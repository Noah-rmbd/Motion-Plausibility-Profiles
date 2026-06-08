import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from integrate_predictions import integrate_predictions_into_json
from gowalla_trajectory_computation import compute_unnormalized_statistics

# Global input dimension is intrinsic to the features we have
INPUT_DIM = 6         # Input features: [speed, elapsed_time, distance, acceleration, sin_bearing, cos_bearing]

class TrajectoryDataset(Dataset):
    """
    Custom PyTorch Dataset for trajectory sequence data.
    Wraps trajectory matrices and their corresponding identifier lists.
    """
    def __init__(self, trajectories, ids, feature_indices=None):
        if feature_indices is None:
            feature_indices = [0, 1, 2, 3, 4, 5]
        self.feature_indices = feature_indices
        # Convert list of NumPy trajectory matrices to Float32 tensors, slicing only requested features
        self.trajectories = [torch.tensor(t.astype(np.float32)[:, self.feature_indices]) for t in trajectories]
        self.ids = ids # List of identifiers: [trajectory_id, trans_id_1, trans_id_2, ...]
        
    def __len__(self):
        return len(self.trajectories)
        
    def __getitem__(self, idx):
        return self.trajectories[idx], self.ids[idx]

def collate_fn(batch):
    """
    Collate function to assemble variable-length trajectory sequences into a batch.
    Required for PyTorch DataLoader when working with variable-length sequences.
    
    Orders sequences in descending order of length, which is a requirement for 
    efficient packed sequence processing via `pack_padded_sequence`.
    """
    # Sort batch by sequence length (descending) for pack_padded_sequence
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, ids = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Pad shorter sequences with zeros to make them equal length in the tensor representation
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, lengths, ids

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder network for trajectory reconstruction and anomaly detection.
    
    The network compresses input trajectory sequences into a lower-dimensional latent 
    representation (Encoder) and then reconstructs the original sequence (Decoder). 
    Highly anomalous trajectories exhibit high reconstruction error.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super(LSTMAutoencoder, self).__init__()
        # Encoder: Compresses the trajectory features into hidden states
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        # Decoder: Reconstructs the sequence from the latent space representation
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        # Output Projection: Maps decoder hidden state back to the original feature dimension
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 1. ENCODE
        # Pack the padded sequence to skip processing of pad tokens
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hidden, cell) = self.encoder(packed_x)
        
        # 2. LATENT REPRESENTATION
        # We take the final hidden state of the top layer as the compressed vector
        # hidden shape: (num_layers, batch_size, hidden_dim)
        top_hidden = hidden[-1] # (batch_size, hidden_dim)
        
        # Repeat the latent representation seq_len times to act as inputs for the decoder sequence
        decoder_input = top_hidden.unsqueeze(1).repeat(1, seq_len, 1) # (batch_size, seq_len, hidden_dim)
        
        # 3. DECODE
        # Pack decoder inputs
        packed_decoder_input = pack_padded_sequence(decoder_input, lengths.cpu(), batch_first=True, enforce_sorted=True)
        # Run decoder initialized with encoder's final hidden states
        decoder_output, _ = self.decoder(packed_decoder_input, (hidden, cell))
        
        # Pad back the packed output sequences to a standard tensor format
        unpacked_output, _ = pad_packed_sequence(decoder_output, batch_first=True) # (batch_size, seq_len, hidden_dim)
        
        # 4. RECONSTRUCT
        # Project hidden states to original feature space
        reconstructed = self.output_layer(unpacked_output) # (batch_size, seq_len, input_dim)
        return reconstructed

def train_autoencoder(model, dataloader, criterion, optimizer, device):
    """
    Standard PyTorch training loop for one epoch.
    Trains the LSTM Autoencoder model to reconstruct normal trajectories.
    """
    model.train()
    total_loss = 0
    for batch_idx, (sequences, lengths, _) in enumerate(dataloader):
        sequences = sequences.to(device)
        
        optimizer.zero_grad()
        # Forward pass: reconstruct the input trajectories
        reconstructed = model(sequences, lengths)
        
        # Create a boolean mask to ignore padded timesteps when calculating loss
        mask = torch.arange(sequences.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.to(device).unsqueeze(2).expand_as(sequences)
        
        # Compute Mean Squared Error (MSE) only on valid, unpadded timesteps
        loss = criterion(reconstructed[mask], sequences[mask])
        
        # Backward pass & optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"    Batch {batch_idx + 1}/{len(dataloader)} Loss: {loss.item():.6f}")
            
    return total_loss / len(dataloader)

def predict_anomalies(model, dataloader, criterion, device, ignore_first_transition_error=True, feature_indices=None):
    """
    Inference loop to compute reconstruction errors for all trajectories.
    Also extracts transition-level MSE errors for step-specific analyses.
    """
    if feature_indices is None:
        feature_indices = [0, 1, 2, 3, 4, 5]
        
    model.eval()
    all_errors = []
    all_ids = []
    transition_errors = {}
    
    with torch.no_grad():
        for sequences, lengths, ids in dataloader:
            sequences = sequences.to(device)
            # Reconstruct the batch of trajectories
            reconstructed = model(sequences, lengths)
            
            # Compute step-wise squared errors
            squared_errors = (reconstructed - sequences) ** 2
            # Average errors over the 6 features to get MSE per step
            mse_per_step = squared_errors.mean(dim=2) # (batch_size, seq_len)
            
            # Map sequence-level and step-level errors
            for i in range(len(lengths)):
                valid_len = lengths[i].item()
                # Trajectory error is the mean MSE of its valid timesteps (optionally ignoring the first transition)
                if ignore_first_transition_error and valid_len > 1:
                    seq_error = mse_per_step[i, 1:valid_len].mean().item()
                else:
                    seq_error = mse_per_step[i, :valid_len].mean().item()
                all_errors.append(seq_error)
                all_ids.append(ids[i])
                
                # Extract errors for individual transitions inside this trajectory
                traj_info = ids[i]  # Format: [trajectory_id, trans_id_1, trans_id_2, ...]
                for j in range(valid_len):
                    trans_id = traj_info[j + 1]
                    step_err = mse_per_step[i, j].item()
                    
                    sliced_errs = squared_errors[i, j].tolist()
                    padded_errs = [0.0] * 6
                    for idx, f_idx in enumerate(feature_indices):
                        padded_errs[f_idx] = sliced_errs[idx]
                        
                    transition_errors[trans_id] = {
                        'mse': step_err,
                        'features': padded_errs
                    }
                
    return np.array(all_errors), all_ids, transition_errors

def load_and_filter_dataset(name, traj_file, id_file, trans_id_file):
    """
    Loads dataset pickles and separates plausible from "intrinsically implausible" trajectories.
    Returns lists for plausible and implausible trajectories so they can both be evaluated.
    """
    print(f"--- Loading & Filtering {name} ---")
    
    # Load dataset pickle files
    with open(traj_file, 'rb') as f:
        trajectories = pickle.load(f)
    with open(id_file, 'rb') as f:
        traj_ids = pickle.load(f)
    with open(trans_id_file, 'rb') as f:
        transitions_ids = pickle.load(f)
        
    assert len(trajectories) == len(traj_ids), f"Mismatch between trajectories and IDs for {name}!"
    
    plausible_trajectories = []
    plausible_traj_ids = []
    
    implausible_trajectories = []
    implausible_traj_ids = []
    implausible_info = {}
    
    for traj, t_id in zip(trajectories, traj_ids):    
        tid = t_id[0]
        
        # Check for NaN values in trajectory
        if np.isnan(traj).any():
            continue

        traj_trans_ids = t_id[1:]
        is_implausible = False
        worst_trans_id = None
        
        # Check all transitions inside the trajectory
        for tid in traj_trans_ids:
            # If transition_plausibility score is 0, it means it was flagged as impossible by logic checks
            if transitions_ids[tid][5] == 0.0:
                is_implausible = True
                worst_trans_id = tid
                break
                
        if is_implausible:
            implausible_trajectories.append(traj)
            implausible_traj_ids.append(t_id)
            implausible_info[t_id[0]] = {
                'least_plausible_transition_id': worst_trans_id,
                'plausibility_reason': transitions_ids[worst_trans_id][6] if len(transitions_ids[worst_trans_id]) > 6 else None
            }
        else:
            plausible_trajectories.append(traj)
            plausible_traj_ids.append(t_id)

    print(f"Filtered out {len(implausible_trajectories)} intrinsically implausible trajectories.")
    return plausible_trajectories, plausible_traj_ids, implausible_trajectories, implausible_traj_ids, implausible_info

def evaluate_and_save_dataset(name, model, plausible_trajectories, plausible_traj_ids, implausible_trajectories, implausible_traj_ids, implausible_info, output_file, device, batch_size=256, anomaly_percentile=97, ignore_first_transition_error=True, verbose=True, feature_indices=None):
    """
    Evaluates model reconstruction errors, calculates the anomaly threshold,
    flags anomalies, and saves the final predictions into a pickle file.
    """
    if verbose:
        print(f"--- Evaluating {name} ---")
    
    criterion = nn.MSELoss()
    
    # 1. Evaluate plausible trajectories
    if verbose:
        print("Predicting anomalies for plausible trajectories...")
    dataset_p = TrajectoryDataset(plausible_trajectories, plausible_traj_ids, feature_indices=feature_indices)
    loader_p = DataLoader(dataset_p, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    errors_p, ids_p, transition_errors_p = predict_anomalies(model, loader_p, criterion, device, ignore_first_transition_error, feature_indices)
    
    # Compute threshold for anomaly tagging (Top 5% highest MSE errors)
    threshold = np.percentile(errors_p, anomaly_percentile)
    if verbose:
        print(f"Threshold ({anomaly_percentile}th percentile): {threshold:.6f}")
    
    # 2. Evaluate implausible trajectories
    if len(implausible_trajectories) > 0:
        if verbose:
            print("Predicting anomalies for intrinsically implausible trajectories...")
        dataset_i = TrajectoryDataset(implausible_trajectories, implausible_traj_ids, feature_indices=feature_indices)
        loader_i = DataLoader(dataset_i, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        errors_i, ids_i, transition_errors_i = predict_anomalies(model, loader_i, criterion, device, ignore_first_transition_error, feature_indices)
    else:
        errors_i, ids_i, transition_errors_i = [], [], {}

    results = []
    num_anomalies = len(implausible_trajectories)
    
    # Add plausible results
    for err, t_id in zip(errors_p, ids_p):
        is_unplausible = bool(err > threshold)
        if is_unplausible:
            num_anomalies += 1
            
        traj_trans_ids = t_id[1:]
        worst_trans_id = None
        if len(traj_trans_ids) > 0 and is_unplausible:
            worst_trans_id = max(traj_trans_ids, key=lambda tid: transition_errors_p.get(tid, {'mse': 0.0})['mse'])
            
        results.append({
            'trajectory_id': int(t_id[0]),
            'reconstruction_error': round(float(err), 5),
            'is_unplausible': is_unplausible,
            'transition_errors': {int(tid): round(float(transition_errors_p.get(tid, {'mse': 0.0})['mse']), 5) for tid in traj_trans_ids},
            'transition_feature_errors': {int(tid): [round(float(val), 5) for val in transition_errors_p.get(tid, {'features': [0.0]*6})['features']] for tid in traj_trans_ids},
            'least_plausible_transition_id': worst_trans_id if is_unplausible else None,
            'plausibility_reason': "High reconstruction error" if is_unplausible else None
        })
        
    # Add implausible results
    for err, t_id in zip(errors_i, ids_i):
        traj_trans_ids = t_id[1:]
        info = implausible_info.get(t_id[0], {})
        
        results.append({
            'trajectory_id': int(t_id[0]),
            'reconstruction_error': round(float(err), 5),
            'is_unplausible': True,
            'transition_errors': {int(tid): round(float(transition_errors_i.get(tid, {'mse': 0.0})['mse']), 5) for tid in traj_trans_ids},
            'transition_feature_errors': {int(tid): [round(float(val), 5) for val in transition_errors_i.get(tid, {'features': [0.0]*6})['features']] for tid in traj_trans_ids},
            'least_plausible_transition_id': info.get('least_plausible_transition_id'),
            'plausibility_reason': info.get('plausibility_reason')
        })
        
    if verbose:
        print(f"Found {num_anomalies} unplausible trajectories out of {len(plausible_trajectories) + len(implausible_trajectories)} total.")
    
    # Ensure target output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    import json
    if verbose:
        print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        metadata = {
            'metadata': {
                'threshold': round(float(threshold), 5),
                'percentile': anomaly_percentile
            }
        }
        f.write(json.dumps(metadata) + '\n')
        for r in results:
            f.write(json.dumps(r) + '\n')
        
    if verbose:
        print(f"Finished {name}.\n")

def analyze_anomalies(anomaly_scores_path, trips_file, traj_ids_file, anomaly_percentile=97, verbose=True):
    """
    Loads predicted anomalies, filters them, and generates distribution graphs based 
    on their unnormalized physical characteristics for human verification.
    """
    if not verbose:
        return
        
    print(f"\n--- Analyzing Anomalies for {anomaly_scores_path} ---")
    
    import json
    anomaly_scores = []
    with open(anomaly_scores_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if 'metadata' in obj:
                continue
            anomaly_scores.append(obj)
        
    unplausible_traj_ids = set()
    for item in anomaly_scores:
        # Ignore intrinsically implausible trajectories (reconstruction_error == -1.0)
        # so we only plot statistics for the LSTM-predicted anomalies.
        if item['is_unplausible'] and item['reconstruction_error'] != -1.0:
            tid = item['trajectory_id'] if isinstance(item['trajectory_id'], int) else item['trajectory_id'][0]
            unplausible_traj_ids.add(tid)
            
    # Load unnormalized data for descriptive stats mapping
    with open(trips_file, 'rb') as f:
        unnormalized_trips = pickle.load(f)
        
    with open(traj_ids_file, 'rb') as f:
        trajectories_id = pickle.load(f)
        
    anomalous_trajectories = []
    for traj in trajectories_id:
        if traj[0] in unplausible_traj_ids:
            anomalous_trajectories.append(traj)
            
    print(f"Loaded {len(anomalous_trajectories)} unplausible trajectories for analysis.")
    if len(anomalous_trajectories) == 0:
        return
    
    compute_unnormalized_statistics(
        trips=unnormalized_trips,
        trajectories_id=anomalous_trajectories,
        title_prefix="(Anomalies Only)",
        plot_path="vis/gowalla_anomalies_distributions_v2_threshold="+str(anomaly_percentile)+".png"
    )
    print("--- Finished Anomaly Analysis ---\n")

def prepare_datasets(base_dir="Preprocessed_Dataset"):
    # 1. iNaturalist dataset paths
    inat_traj = os.path.join(base_dir, "unnormalized_trajectories.pkl")
    inat_ids = os.path.join(base_dir, "trajectories_id.pkl")
    inat_trans = os.path.join(base_dir, "transitions_id.pkl")
    
    # 2. Gowalla dataset paths
    gowalla_traj = os.path.join(base_dir, "gowalla_unnormalized_trajectories.pkl")
    gowalla_ids = os.path.join(base_dir, "gowalla_trajectories_id.pkl")
    gowalla_trans = os.path.join(base_dir, "gowalla_trips_id.pkl")
    
    # Load and filter both datasets
    combined_plausible_trajectories = []
    combined_plausible_traj_ids = []
    
    inat_loaded = False
    inat_plausible_traj, inat_plausible_ids = None, None
    inat_implausible_traj, inat_implausible_ids, inat_implausible_info = None, None, None
    if os.path.exists(inat_traj) and os.path.exists(inat_ids) and os.path.exists(inat_trans):
        inat_plausible_traj, inat_plausible_ids, inat_implausible_traj, inat_implausible_ids, inat_implausible_info = load_and_filter_dataset(
            "iNaturalist", inat_traj, inat_ids, inat_trans
        )
        combined_plausible_trajectories.extend(inat_plausible_traj)
        combined_plausible_traj_ids.extend(inat_plausible_ids)
        inat_loaded = True
    else:
        print("iNaturalist dataset files not found.")
        
    gowalla_loaded = False
    gowalla_plausible_traj, gowalla_plausible_ids = None, None
    gowalla_implausible_traj, gowalla_implausible_ids, gowalla_implausible_info = None, None, None
    gowalla_ids_data = None
    if os.path.exists(gowalla_traj) and os.path.exists(gowalla_ids) and os.path.exists(gowalla_trans):
        gowalla_plausible_traj, gowalla_plausible_ids, gowalla_implausible_traj, gowalla_implausible_ids, gowalla_implausible_info = load_and_filter_dataset(
            "Gowalla", gowalla_traj, gowalla_ids, gowalla_trans
        )
        with open(gowalla_ids, 'rb') as f:
            gowalla_ids_data = pickle.load(f)
        combined_plausible_trajectories.extend(gowalla_plausible_traj)
        combined_plausible_traj_ids.extend(gowalla_plausible_ids)
        gowalla_loaded = True
    else:
        print("Gowalla dataset files not found.")
        
    if not inat_loaded and not gowalla_loaded:
        print("No datasets loaded. Exiting.")
        return None
        
    # --- Compute global scaler and normalize all trajectories ---
    from sklearn.preprocessing import MinMaxScaler
    print("Fitting global scalers on combined unnormalized datasets...")
    all_transitions = []
    for traj in combined_plausible_trajectories:
        all_transitions.extend(traj)
    all_transitions = np.array(all_transitions, dtype=float)
    
    speed_scaler = MinMaxScaler()
    elapsed_time_scaler = MinMaxScaler()
    distance_scaler = MinMaxScaler()
    acceleration_scaler = MinMaxScaler()
    
    speeds = all_transitions[:, 0]
    valid_speeds = speeds[speeds <= 900.0]
    if len(valid_speeds) > 0:
        speed_scaler.fit(np.log1p(valid_speeds).reshape(-1, 1))
    else:
        speed_scaler.fit(np.log1p(speeds).reshape(-1, 1))
        
    elapsed_time_scaler.fit(np.log1p(all_transitions[:, 1]).reshape(-1, 1))
    distance_scaler.fit(np.log1p(all_transitions[:, 2]).reshape(-1, 1))
    
    # Acceleration can be negative, so we use signed log1p
    accel = all_transitions[:, 3]
    accel_log = np.sign(accel) * np.log1p(np.abs(accel))
    acceleration_scaler.fit(accel_log.reshape(-1, 1))
    
    os.makedirs("Models_weights_and_results", exist_ok=True)
    scalers_path = "Models_weights_and_results/combined_scalers.pkl"
    with open(scalers_path, "wb") as f:
        pickle.dump((speed_scaler, elapsed_time_scaler, distance_scaler, acceleration_scaler), f)
        
    def normalize_trajs(trajs):
        normalized = []
        for traj in trajs:
            traj_arr = np.array(traj, dtype=float)
            size = traj_arr.shape[0]
            new_traj = np.zeros((size, 6), dtype=float)
            
            new_traj[:, 0] = speed_scaler.transform(np.log1p(traj_arr[:, 0]).reshape(-1, 1)).reshape(-1)
            new_traj[:, 1] = elapsed_time_scaler.transform(np.log1p(traj_arr[:, 1]).reshape(-1, 1)).reshape(-1)
            new_traj[:, 2] = distance_scaler.transform(np.log1p(traj_arr[:, 2]).reshape(-1, 1)).reshape(-1)
            
            traj_accel = traj_arr[:, 3]
            traj_accel_log = np.sign(traj_accel) * np.log1p(np.abs(traj_accel))
            new_traj[:, 3] = acceleration_scaler.transform(traj_accel_log.reshape(-1, 1)).reshape(-1)
            new_traj[:, 4] = (np.sin(traj_arr[:, 4]) + 1.0) / 2.0
            new_traj[:, 5] = (np.cos(traj_arr[:, 4]) + 1.0) / 2.0
            normalized.append(new_traj)
        return normalized

    print("Normalizing trajectories...")
    combined_normalized_trajectories = normalize_trajs(combined_plausible_trajectories)
    inat_normalized_traj = normalize_trajs(inat_plausible_traj) if inat_loaded else None
    inat_normalized_impl_traj = normalize_trajs(inat_implausible_traj) if inat_loaded else None
    gowalla_normalized_traj = normalize_trajs(gowalla_plausible_traj) if gowalla_loaded else None
    gowalla_normalized_impl_traj = normalize_trajs(gowalla_implausible_traj) if gowalla_loaded else None
    
    return {
        "inat_loaded": inat_loaded,
        "gowalla_loaded": gowalla_loaded,
        "combined_normalized_trajectories": combined_normalized_trajectories,
        "combined_plausible_traj_ids": combined_plausible_traj_ids,
        "inat_normalized_traj": inat_normalized_traj,
        "inat_plausible_ids": inat_plausible_ids,
        "inat_normalized_impl_traj": inat_normalized_impl_traj,
        "inat_implausible_ids": inat_implausible_ids,
        "inat_implausible_info": inat_implausible_info,
        "gowalla_normalized_traj": gowalla_normalized_traj,
        "gowalla_plausible_ids": gowalla_plausible_ids,
        "gowalla_normalized_impl_traj": gowalla_normalized_impl_traj,
        "gowalla_implausible_ids": gowalla_implausible_ids,
        "gowalla_implausible_info": gowalla_implausible_info,
        "gowalla_ids_data": gowalla_ids_data
    }

def train_evaluate_model(
    hidden_dim=32,
    num_layers=2,
    batch_size=256,
    num_epochs=10,
    learning_rate=1e-3,
    anomaly_percentile=97,
    ignore_first_transition_error=True,
    model_save_path="Models_weights_and_results/lstm_autoencoder_combinedv5.pth",
    run_eval=True,
    preprocessed_data=None,
    pre_trained_model=None,
    verbose=True,
    feature_indices=None,
    model_name="full_model"
):
    """
    Orchestrator function that trains the LSTM autoencoder and evaluates the datasets.
    Designed to be modular for hyperparameter tuning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Note: We intentionally do NOT use MPS on Mac ARM because PyTorch's MPS implementation 
    # of LSTM for packed sequences is extremely unoptimized, running 50x slower than CPU.
    if verbose:
        print(f"Using device: {device}")
    
    # Preprocessed dataset directory
    base_dir = "Preprocessed_Dataset"
    
    if feature_indices is None:
        feature_indices = [0, 1, 2, 3, 4, 5]

    
    inat_traj_json = os.path.join(base_dir, "trajectories_list.json")
    inat_out = os.path.join("Models_weights_and_results", f"anomaly_scores_inat_{model_name}.jsonl")
    
    gowalla_traj_json = os.path.join(base_dir, "gowalla_trajectories_list.json")
    gowalla_out = os.path.join("Models_weights_and_results", f"anomaly_scores_gowalla_{model_name}.jsonl")
    
    if preprocessed_data is None:
        preprocessed_data = prepare_datasets(base_dir)
        
    if preprocessed_data is None:
        return
        
    inat_loaded = preprocessed_data["inat_loaded"]
    gowalla_loaded = preprocessed_data["gowalla_loaded"]
    combined_normalized_trajectories = preprocessed_data["combined_normalized_trajectories"]
    combined_plausible_traj_ids = preprocessed_data["combined_plausible_traj_ids"]
    
    inat_normalized_traj = preprocessed_data["inat_normalized_traj"]
    inat_plausible_ids = preprocessed_data["inat_plausible_ids"]
    inat_normalized_impl_traj = preprocessed_data["inat_normalized_impl_traj"]
    inat_implausible_ids = preprocessed_data["inat_implausible_ids"]
    inat_implausible_info = preprocessed_data["inat_implausible_info"]
    
    gowalla_normalized_traj = preprocessed_data["gowalla_normalized_traj"]
    gowalla_plausible_ids = preprocessed_data["gowalla_plausible_ids"]
    gowalla_normalized_impl_traj = preprocessed_data["gowalla_normalized_impl_traj"]
    gowalla_implausible_ids = preprocessed_data["gowalla_implausible_ids"]
    gowalla_implausible_info = preprocessed_data["gowalla_implausible_info"]
    gowalla_ids = preprocessed_data["gowalla_ids_data"]

        
    # Initialize combined autoencoder model
    if pre_trained_model is not None:
        print("Using provided pre-trained model...")
        model = pre_trained_model
    else:
        print("Initializing model...")
        input_dim = len(feature_indices)
        model = LSTMAutoencoder(input_dim, hidden_dim, num_layers).to(device)
        
        if os.path.exists(model_save_path):
            print(f"Loading combined model weights from {model_save_path}...")
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        else:
            print(f"Training combined model from scratch on {len(combined_normalized_trajectories)} trajectories...")
            combined_dataset = TrajectoryDataset(combined_normalized_trajectories, combined_plausible_traj_ids, feature_indices=feature_indices)
            dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            for epoch in range(num_epochs):
                loss = train_autoencoder(model, dataloader, criterion, optimizer, device)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}")
                
            os.makedirs("Models_weights_and_results", exist_ok=True)
            print(f"Saving combined model to {model_save_path}...")
            torch.save(model.state_dict(), model_save_path)
        
    if not run_eval:
        return model
        
    # Evaluate iNaturalist
    if inat_loaded:
        evaluate_and_save_dataset(
            "iNaturalist", model, inat_normalized_traj, inat_plausible_ids, inat_normalized_impl_traj, inat_implausible_ids, inat_implausible_info, inat_out, device,
            batch_size=batch_size, anomaly_percentile=anomaly_percentile, ignore_first_transition_error=ignore_first_transition_error, verbose=verbose, feature_indices=feature_indices
        )
        
    # Evaluate Gowalla
    if gowalla_loaded:
        evaluate_and_save_dataset(
            "Gowalla", model, gowalla_normalized_traj, gowalla_plausible_ids, gowalla_normalized_impl_traj, gowalla_implausible_ids, gowalla_implausible_info, gowalla_out, device,
            batch_size=batch_size, anomaly_percentile=anomaly_percentile, ignore_first_transition_error=ignore_first_transition_error, verbose=verbose, feature_indices=feature_indices
        )
        
        # Analyze the anomalies using the unnormalized values and print stats
        gowalla_unnormalized_trips = os.path.join(base_dir, "gowalla_unnormalized_trips.pkl")
        
        # We need to write gowalla_ids_data somewhere or load it if we modified it? Wait, gowalla_ids is already passed from datasets
        analyze_anomalies(gowalla_out, gowalla_unnormalized_trips, os.path.join(base_dir, "gowalla_trajectories_id.pkl"), anomaly_percentile, verbose=verbose)
        
        # Evaluate synthetic anomalies
        from generate_synthetic_anomalies import run_synthetic_evaluation
        run_synthetic_evaluation(model_save_path, gowalla_out, inat_out, model_name, feature_indices)

        
    return model

if __name__ == "__main__":
    train_evaluate_model()