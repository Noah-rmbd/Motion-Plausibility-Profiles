import os
import pickle
import json
import math
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from trajectory_prediction_LSTM import LSTMAutoencoder, TrajectoryDataset, collate_fn

IGNORE_FIRST_TRANSITION_ERROR = True

MODEL_CONFIGS = {
    "full_model": [0, 1, 2, 3, 4, 5],
    "dist_time_bearing": [1, 2, 4, 5],
    "dist_time": [1, 2],
}

EPOCH_CONFIGS = [5, 10, 15]
BATCH_SIZE_CONFIGS = [256]

# Calculate bearing between two points
def compute_bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return math.atan2(x, y)

# Destination point given start point, distance, and bearing
def destination_point(lat, lon, distance, bearing):
    R = 6371000.0 # Earth radius
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    
    d_R = distance / R
    
    lat2 = math.asin(math.sin(lat1) * math.cos(d_R) +
                     math.cos(lat1) * math.sin(d_R) * math.cos(bearing))
                     
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d_R) * math.cos(lat1),
                             math.cos(d_R) - math.sin(lat1) * math.sin(lat2))
                             
    return math.degrees(lat2), math.degrees(lon2)

def normalize_trajectory(trajectory, scalers):
    speed_scaler, elapsed_time_scaler, distance_scaler, acceleration_scaler = scalers
    trajectory = np.array(trajectory)
    trajectory_size = np.shape(trajectory)[0]
    
    new_traj = np.zeros((trajectory_size, 6), dtype=float)
    traj_speeds = trajectory[:, 0].astype(float)
    new_traj[:, 0] = speed_scaler.transform(np.log1p(traj_speeds).reshape(-1, 1)).reshape(trajectory_size,)
    new_traj[:, 1] = elapsed_time_scaler.transform(np.log1p(trajectory[:, 1]).reshape(-1, 1)).reshape(trajectory_size,)
    new_traj[:, 2] = distance_scaler.transform(np.log1p(trajectory[:, 2]).reshape(-1, 1)).reshape(trajectory_size,)
    
    traj_accel = trajectory[:, 3].astype(float)
    traj_accel_log = np.sign(traj_accel) * np.log1p(np.abs(traj_accel))
    new_traj[:, 3] = acceleration_scaler.transform(traj_accel_log.reshape(-1, 1)).reshape(trajectory_size,)
    new_traj[:, 4] = (np.sin(trajectory[:, 4].astype(float)) + 1.0) / 2.0
    new_traj[:, 5] = (np.cos(trajectory[:, 4].astype(float)) + 1.0) / 2.0
    return new_traj

def generate_back_and_forth(trajectories):
    synthetic = []
    for traj in trajectories[:50]:
        new_traj = [list(trans) for trans in traj]
        if len(new_traj) >= 2:
            num_events = np.random.randint(1, 3)
            for _ in range(num_events):
                start_idx = np.random.randint(0, len(new_traj) - 1)
                new_traj[start_idx][4] = np.pi + np.random.normal(0, 0.1)
                new_traj[start_idx + 1][4] = np.pi + np.random.normal(0, 0.1)
        synthetic.append(np.array(new_traj))
    return synthetic

def generate_abnormal_speeds(trajectories):
    synthetic = []
    for traj in trajectories[:50]:
        new_traj = [list(trans) for trans in traj]
        if len(new_traj) >= 2:
            seq_len = np.random.randint(2, min(5, len(new_traj) + 1))
            start_idx = np.random.randint(0, len(new_traj) - seq_len + 1)
            for i in range(start_idx, start_idx + seq_len):
                speed_kmh = np.random.uniform(500, 850)
                elapsed_s = np.random.uniform(1, 5) * 3600
                distance_m = speed_kmh * (elapsed_s / 3600) * 1000
                new_traj[i][0] = speed_kmh
                new_traj[i][1] = elapsed_s
                new_traj[i][2] = distance_m
                new_traj[i][3] = np.random.normal(0, 0.5)
        synthetic.append(np.array(new_traj))
    return synthetic

def generate_time_to_first_fix_delay(trajectories):
    synthetic = []
    for traj in trajectories[:50]:
        if len(traj) < 4:
            continue
        new_traj = [list(trans) for trans in traj]
        
        # Simulates an inaccurate first point (P1), making Transition 0 (P1 -> P2) absurdly long and fast
        added_dist = np.random.uniform(5000, 20000) # 5km to 20km error jump
        new_traj[0][2] += added_dist
        
        # Recalculate speed (dist / time * 3.6 for km/h)
        elapsed_time = new_traj[0][1]
        if elapsed_time > 0:
            new_speed_kmh = (new_traj[0][2] / elapsed_time) * 3.6
            new_traj[0][0] = new_speed_kmh
            # Recalculate acceleration, assuming a rest start (0 m/s)
            new_traj[0][3] = (new_speed_kmh / 3.6) / elapsed_time
            
        synthetic.append(np.array(new_traj))
    return synthetic

def evaluate_recall(model, dataloader, threshold, device, num_samples, feature_indices=None):
    if feature_indices is None:
        feature_indices = [0, 1, 2, 3, 4, 5]
    model.eval()
    errors = [0.0] * num_samples
    traj_is_unplausible = [False] * num_samples
    step_errors = [None] * num_samples
    step_feature_errors = [None] * num_samples
    
    with torch.no_grad():
        for sequences, lengths, ids in dataloader:
            sequences = sequences.to(device)
            reconstructed = model(sequences, lengths)
            squared_errors = (reconstructed - sequences) ** 2
            mse_per_step = squared_errors.mean(dim=2)
            
            for i in range(sequences.size(0)):
                orig_idx = ids[i][0]
                valid_len = lengths[i].item()
                # Trajectory error is the mean MSE of its valid timesteps (optionally ignoring the first transition)
                if IGNORE_FIRST_TRANSITION_ERROR and valid_len > 1:
                    seq_error = mse_per_step[i, 1:valid_len].mean().item()
                else:
                    seq_error = mse_per_step[i, :valid_len].mean().item()
                
                errors[orig_idx] = seq_error
                traj_is_unplausible[orig_idx] = (seq_error > threshold)
                
                step_errs = mse_per_step[i, :valid_len].tolist()
                step_errors[orig_idx] = step_errs
                
                step_feat_errs_sliced = squared_errors[i, :valid_len].tolist()
                padded_step_feat_errs = []
                for s_err in step_feat_errs_sliced:
                    padded_errs = [0.0] * 6
                    for idx, f_idx in enumerate(feature_indices):
                        padded_errs[f_idx] = s_err[idx]
                    padded_step_feat_errs.append(padded_errs)
                
                step_feature_errors[orig_idx] = padded_step_feat_errs
                
    errors = np.array(errors)
    detected = np.sum(errors > threshold)
    recall = (detected / len(errors)) * 100 if len(errors) > 0 else 0
    return errors, recall, traj_is_unplausible, step_errors, step_feature_errors

def run_synthetic_evaluation(model_path='Models_weights_and_results/lstm_autoencoder_combinedv5.pth',
                             anomaly_scores_path='Models_weights_and_results/anomaly_scores_gowallav5.jsonl',
                             inat_scores_path='Models_weights_and_results/anomaly_scores_inat.jsonl',
                             model_name="full_model",
                             feature_indices=None):
    if feature_indices is None:
        feature_indices = [0, 1, 2, 3, 4, 5]
    
    np.random.seed(1)
    print(f"\n--- Starting Synthetic Anomaly Evaluation for {model_name} ({os.path.basename(model_path)}) ---")
    
    # 1. Load data
    unnormalized_trips_path = 'Preprocessed_Dataset/gowalla_unnormalized_trips.pkl'
    trajectories_id_path = 'Preprocessed_Dataset/gowalla_trajectories_id.pkl'
    trips_id_path = 'Preprocessed_Dataset/gowalla_trips_id.pkl'

    
    if not (os.path.exists(unnormalized_trips_path) and os.path.exists(trajectories_id_path) and os.path.exists(trips_id_path)):
        print("Missing required unnormalized Gowalla files.")
        return
        
    print("Loading unnormalized trips & trajectory IDs...")
    with open(unnormalized_trips_path, 'rb') as f:
        trips = pickle.load(f)
    with open(trajectories_id_path, 'rb') as f:
        trajectories_id = pickle.load(f)
    with open(trips_id_path, 'rb') as f:
        trips_id = pickle.load(f)
        
    # Load global scalers
    print("Loading global scalers...")
    scalers_path = 'Models_weights_and_results/combined_scalers.pkl'
    if not os.path.exists(scalers_path):
        print(f"Missing combined scalers file at {scalers_path}.")
        return
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    # Reconstruct unnormalized trajectories
    print("Reconstructing unnormalized trajectories...")
    all_trajectories = []
    for traj_info in trajectories_id:
        trip_ids = traj_info[1:]
        traj_data = [trips[idx] for idx in trip_ids]
        all_trajectories.append(traj_data)
        
    # Filter for clean trajectories (not intrinsically implausible)
    scores = []
    file_threshold = None
    with open(anomaly_scores_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if 'metadata' in obj:
                file_threshold = obj['metadata'].get('threshold')
                continue
            scores.append(obj)
        
    clean_traj_indices = []
    for idx, item in enumerate(scores):
        if item['reconstruction_error'] != -1.0 and not item['is_unplausible'] and len(all_trajectories[idx]) >= 4:
            clean_traj_indices.append(idx)
                
    # Select up to 1000 clean trajectories to mutate
    np.random.seed(42)
    selected_indices = np.random.choice(clean_traj_indices, min(1000, len(clean_traj_indices)), replace=False)
    
    # Track the selected baseline trajectory info
    base_trajectories = [all_trajectories[idx] for idx in selected_indices]
    base_trajectories_id = [trajectories_id[idx] for idx in selected_indices]
    
    print(f"Selected {len(base_trajectories)} clean trajectories as baseline for anomaly injection.")
    
    # 2. Generate synthetic profiles
    print("\nGenerating synthetic anomaly profiles...")
    profiles = {
        "Back & Forth Shape": generate_back_and_forth(base_trajectories),
        "Abnormal Speed Sequence (Flights)": generate_abnormal_speeds(base_trajectories),
        "Time-to-First-Fix Delay": generate_time_to_first_fix_delay(base_trajectories)
    }
    
    # Keep track of corresponding base ID list for each profile
    profiles_base_ids = {
        "Back & Forth Shape": base_trajectories_id,
        "Abnormal Speed Sequence (Flights)": base_trajectories_id,
        "Time-to-First-Fix Delay": base_trajectories_id
    }
    
    # Normalize each set of trajectories
    normalized_profiles = {}
    for name, list_traj in profiles.items():
        normalized_list = [normalize_trajectory(t, scalers) for t in list_traj]
        normalized_profiles[name] = normalized_list
        print(f"  - {name}: {len(normalized_list)} trajectories")
        
    # 3. Load model and anomaly threshold
    print("\nLoading model & anomaly scores threshold...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(feature_indices)
    model = LSTMAutoencoder(input_dim, 32, 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    if file_threshold is not None:
        threshold = file_threshold
    else:
        unplausible_errors = [
            item['reconstruction_error'] for item in scores 
            if item['is_unplausible'] and item['reconstruction_error'] != -1.0 
            and item.get('plausibility_reason') == "High reconstruction error"
        ]
        threshold = min(unplausible_errors) if unplausible_errors else 0.019100
    # 4. Evaluate False Positives using jimw3 (user_id 98904) as True Negatives
    fp_count = 0
    tn_count = 0
    if os.path.exists('Preprocessed_Dataset/trajectories_id.pkl') and os.path.exists('Preprocessed_Dataset/transitions_id.pkl') and os.path.exists(inat_scores_path):
        with open('Preprocessed_Dataset/trajectories_id.pkl', 'rb') as f:
            inat_traj_id_list = pickle.load(f)
        with open('Preprocessed_Dataset/transitions_id.pkl', 'rb') as f:
            inat_trans_id_list = pickle.load(f)
            
        jimw3_traj_ids = set()
        for traj_id_info in inat_traj_id_list:
            if len(traj_id_info) > 1:
                tid = traj_id_info[0]
                first_trans_id = traj_id_info[1]
                user_id = inat_trans_id_list[first_trans_id][4]
                if user_id == '98904':
                    jimw3_traj_ids.add(tid)
        
        inat_scores = []
        with open(inat_scores_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if 'metadata' in obj:
                    continue
                inat_scores.append(obj)
            
        for score in inat_scores:
            tid = score['trajectory_id'] if isinstance(score['trajectory_id'], int) else score['trajectory_id'][0]
            if tid in jimw3_traj_ids:
                if score['is_unplausible']:
                    fp_count += 1
                else:
                    tn_count += 1
    print(f"False Positives baseline (jimw3 / 98904): {fp_count} FP, {tn_count} TN")
    
    # 5. Evaluate each profile
    print("\n--- Model Evaluation (Recall, Precision, F1) ---")
    results = {}
    evaluation_details = {}
    
    for name, list_traj in normalized_profiles.items():
        # Build dummy IDs
        dummy_ids = [[i, i] for i in range(len(list_traj))]
        dataset = TrajectoryDataset(list_traj, dummy_ids, feature_indices=feature_indices)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        errors, recall, traj_is_unplausible, step_errors, step_feature_errors = evaluate_recall(model, dataloader, threshold, device, len(list_traj), feature_indices)
        
        tp = np.sum(traj_is_unplausible)
        precision = (tp / (tp + fp_count)) * 100 if (tp + fp_count) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[name] = {
            'mean_mse': np.mean(errors),
            'median_mse': np.median(errors),
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
        
        # Save exact evaluation arrays for JSON creation
        evaluation_details[name] = {
            'traj_is_unplausible': traj_is_unplausible,
            'step_errors': step_errors,
            'step_feature_errors': step_feature_errors
        }
        
    # Print formatted results
    print(f"{'Anomaly Profile':<35} | {'Mean MSE':<12} | {'Median MSE':<12} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10}")
    print("-" * 105)
    for name, res in results.items():
        print(f"{name:<35} | {res['mean_mse']:<12.6f} | {res['median_mse']:<12.6f} | {res['recall']:<8.2f}% | {res['precision']:<8.2f}% | {res['f1']:<8.2f}%")
        
    # 6. Export JSON visualization datasets
    print("\nGenerating JSON visualization files...")
    
    # Load gowalla_observations.json to resolve coordinates of start points
    print("Loading gowalla_observations.json...")
    with open('Preprocessed_Dataset/gowalla_observations.json', 'r') as f:
        orig_obs = json.load(f)
    obs_coords = {o['observation_id']: (o['lat'], o['long']) for o in orig_obs}
    
    # Load original timestamps
    obs_timestamps = {}
    if os.path.exists('Preprocessed_Dataset/obs_timestamps.json'):
        with open('Preprocessed_Dataset/obs_timestamps.json', 'r') as f:
            obs_timestamps = json.load(f)
            
    synthetic_users = []
    synthetic_observations = []
    synthetic_transitions = []
    synthetic_trajectories = []
    
    global_trans_id = 0
    global_traj_id = 0
    
    for name, list_traj in profiles.items():
        profile_key = name.replace(" Shape", "").replace(" Sequence", "").replace(" (Flights)", "").replace(" ", "_").replace("&", "And")
        user_id = f"{profile_key}_Anomalies"
        
        eval_det = evaluation_details[name]
        base_tids = profiles_base_ids[name]
        
        total_obs = 0
        
        for t_idx, traj in enumerate(list_traj):
            orig_tid_info = base_tids[t_idx]
            orig_first_trip_id = orig_tid_info[1]
            orig_first_trip = trips_id[orig_first_trip_id]
            
            orig_obs_0 = orig_first_trip[1]
            orig_obs_1 = orig_first_trip[2]
            
            # Start coordinate and date
            start_lat, start_lon = obs_coords.get(orig_obs_0, (30.2679, -97.7493)) # Gowalla defaults
            orig_date = orig_first_trip[3]
            
            # Find original start bearing
            lat1, lon1 = obs_coords.get(orig_obs_1, (start_lat, start_lon))
            initial_bearing = compute_bearing(start_lat, start_lon, lat1, lon1)
            
            # Reconstruct spatial points based on modified properties
            current_lat = start_lat
            current_lon = start_lon
            current_bearing = initial_bearing
            
            traj_obs_ids = []
            
            # Save starting observation
            start_obs_id = f"synth_{profile_key}_{t_idx}_0"
            synthetic_observations.append({
                "user_id": user_id,
                "observation_id": start_obs_id,
                "date": orig_date,
                "lat": current_lat,
                "long": current_lon
            })
            # Map timestamps
            orig_time_0 = obs_timestamps.get(orig_obs_0, "12:00:00")
            obs_timestamps[start_obs_id] = orig_time_0
            traj_obs_ids.append(start_obs_id)
            total_obs += 1
            
            list_of_trans_ids = []
            
            # Propagate steps
            for k in range(len(traj)):
                trans = traj[k]
                speed_kmh, elapsed_time, distance, accel, bearing_change = trans[0], trans[1], trans[2], trans[3], trans[4]
                
                # Update bearing
                current_bearing = current_bearing + bearing_change
                current_bearing = (current_bearing + math.pi) % (2 * math.pi) - math.pi
                
                # Next location
                next_lat, next_lon = destination_point(current_lat, current_lon, distance, current_bearing)
                
                obs_id_next = f"synth_{profile_key}_{t_idx}_{k+1}"
                synthetic_observations.append({
                    "user_id": user_id,
                    "observation_id": obs_id_next,
                    "date": orig_date,
                    "lat": next_lat,
                    "long": next_lon
                })
                # Set timestamp
                orig_next_obs_id = trips_id[orig_tid_info[k+1]][2]
                obs_timestamps[obs_id_next] = obs_timestamps.get(orig_next_obs_id, "12:00:00")
                
                traj_obs_ids.append(obs_id_next)
                total_obs += 1
                
                # Fetch anomaly stats for this step
                is_unplausible_flag = eval_det['traj_is_unplausible'][t_idx]
                step_err = eval_det['step_errors'][t_idx][k]
                step_feat_err = eval_det['step_feature_errors'][t_idx][k]
                
                # A transition is flagged as unplausible in the UI if it is the "worst" transition in an anomalous trajectory
                # Find index of worst step in this trajectory
                worst_step_idx = np.argmax(eval_det['step_errors'][t_idx])
                is_trans_unplausible = bool(is_unplausible_flag and (k == worst_step_idx))
                
                # Add synthetic transition object
                trans_obj = {
                    "transition_id": global_trans_id,
                    "user_id": user_id,
                    "observation_id1": start_obs_id if k == 0 else f"synth_{profile_key}_{t_idx}_{k}",
                    "observation_id2": obs_id_next,
                    "date": orig_date,
                    "speed": float(speed_kmh),
                    "elapsed_time": float(elapsed_time),
                    "distance": float(distance),
                    "acceleration": float(accel),
                    "bearing_change": float(bearing_change),
                    "transition_plausibility": 1,
                    "plausibility_reason": None,
                    "is_unplausible": is_trans_unplausible,
                    "reconstruction_error": float(step_err),
                    "reconstruction_feature_errors": step_feat_err
                }
                synthetic_transitions.append(trans_obj)
                list_of_trans_ids.append(global_trans_id)
                global_trans_id += 1
                
                current_lat, current_lon = next_lat, next_lon
                
            # Add synthetic trajectory object
            traj_obj = {
                "trajectory_id": global_traj_id,
                "source_profile_index": t_idx,
                "user_id": user_id,
                "date": orig_date,
                "transitions": list_of_trans_ids,
                "is_unplausible": bool(is_unplausible_flag)
            }
            synthetic_trajectories.append(traj_obj)
            global_traj_id += 1
            
        # Add user
        synthetic_users.append({
            "username": user_id,
            "nb_observations": total_obs
        })

    # Save visualization JSON files
    print("Writing synthetic JSON files to Preprocessed_Dataset/...")
    with open("Preprocessed_Dataset/synthetic_users_list.json", "w") as f:
        json.dump(synthetic_users, f, indent=4)
    with open("Preprocessed_Dataset/synthetic_transitions_list.json", "w") as f:
        json.dump(synthetic_transitions, f, indent=4)
    with open("Preprocessed_Dataset/synthetic_trajectories_list.json", "w") as f:
        json.dump(synthetic_trajectories, f, indent=4)
    with open("Preprocessed_Dataset/synthetic_observations.json", "w") as f:
        json.dump(synthetic_observations, f, indent=4)
        
    # Save modified obs_timestamps
    with open("Preprocessed_Dataset/obs_timestamps.json", "w") as f:
        json.dump(obs_timestamps, f, indent=4)
        
    print("Export complete. 4 synthetic dataset JSONs successfully saved.")
    
    # 7. Export anomaly scores JSONL for SQL database ingestion
    # This file follows the same format as the iNat/Gowalla JSONL files so that
    # build_database.py --model-score can ingest synthetic model scores into
    # model_trajectory_scores and model_transition_scores tables.
    synth_jsonl_path = f"Models_weights_and_results/anomaly_scores_synthetic_{model_name}.jsonl"
    print(f"\nWriting synthetic anomaly scores to {synth_jsonl_path}...")
    
    with open(synth_jsonl_path, 'w', encoding='utf-8') as f:
        # Write metadata header line
        metadata = {'metadata': {'threshold': float(threshold), 'percentile': 0}}
        f.write(json.dumps(metadata) + '\n')
        
        # Write one line per synthetic trajectory
        for traj_obj in synthetic_trajectories:
            t_idx = traj_obj.get('source_profile_index', traj_obj['trajectory_id'])
            user_id = traj_obj['user_id']
            trans_ids = traj_obj['transitions']
            
            # Identify which profile this trajectory belongs to
            profile_name = None
            for name in profiles.keys():
                profile_key = name.replace(" Shape", "").replace(" Sequence", "").replace(" (Flights)", "").replace(" ", "_").replace("&", "And")
                if user_id == f"{profile_key}_Anomalies":
                    profile_name = name
                    break
            
            if profile_name is None:
                continue
                
            eval_det = evaluation_details[profile_name]
            is_unplausible = eval_det['traj_is_unplausible'][t_idx]
            step_errs = eval_det['step_errors'][t_idx]
            step_feat_errs = eval_det['step_feature_errors'][t_idx]
            
            # Compute trajectory-level error (mean of step errors, optionally ignoring first)
            if IGNORE_FIRST_TRANSITION_ERROR and len(step_errs) > 1:
                traj_error = float(np.mean(step_errs[1:]))
            else:
                traj_error = float(np.mean(step_errs))
            
            # Find worst transition
            worst_idx = int(np.argmax(step_errs))
            worst_trans_id = trans_ids[worst_idx] if is_unplausible else None
            
            transition_errors = {}
            transition_feature_errors = {}
            for k, tid in enumerate(trans_ids):
                transition_errors[str(tid)] = round(float(step_errs[k]), 5)
                transition_feature_errors[str(tid)] = [round(float(v), 5) for v in step_feat_errs[k]]
            
            result = {
                'trajectory_id': traj_obj['trajectory_id'],
                'reconstruction_error': round(traj_error, 5),
                'is_unplausible': bool(is_unplausible),
                'transition_errors': transition_errors,
                'transition_feature_errors': transition_feature_errors,
                'least_plausible_transition_id': worst_trans_id,
                'plausibility_reason': "High reconstruction error" if is_unplausible else None
            }
            f.write(json.dumps(result) + '\n')
    
    print(f"Synthetic JSONL export complete ({synth_jsonl_path}).")

def run_all_synthetic_evaluations():
    for base_model_name, feature_indices in MODEL_CONFIGS.items():
        for epochs in EPOCH_CONFIGS:
            for batch_size in BATCH_SIZE_CONFIGS:
                model_name = f"{base_model_name}_{epochs}_{batch_size}"
                run_synthetic_evaluation(
                    model_path=f"Models_weights_and_results/lstm_autoencoder_{model_name}.pth",
                    anomaly_scores_path=f"Models_weights_and_results/anomaly_scores_gowalla_{model_name}.jsonl",
                    inat_scores_path=f"Models_weights_and_results/anomaly_scores_inat_{model_name}.jsonl",
                    model_name=model_name,
                    feature_indices=feature_indices,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate synthetic anomalies for one or all LSTM models.")
    parser.add_argument("--all", action="store_true", help="Evaluate synthetic anomalies for every benchmark model.")
    args = parser.parse_args()

    if args.all:
        run_all_synthetic_evaluations()
    else:
        run_synthetic_evaluation()
