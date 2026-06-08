import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_feature_errors(anomaly_file, output_prefix="vis/feature_errors"):
    if not os.path.exists(anomaly_file):
        print(f"File {anomaly_file} not found.")
        return

    print(f"Loading {anomaly_file}...")
    scores = []
    with open(anomaly_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if 'metadata' in obj:
                continue
            scores.append(obj)

    # 6 features: Speed, Elapsed Time, Distance, Acceleration, Bear(sin), Bear(cos)
    feature_names = ['Speed', 'Elapsed Time', 'Distance', 'Acceleration', 'Bear(sin)', 'Bear(cos)']
    
    all_feature_errors = {i: [] for i in range(6)}
    unplausible_feature_errors = {i: [] for i in range(6)}

    # We can also collect errors just for the "least plausible" transition to see what drives the anomaly most directly
    least_plausible_feature_errors = {i: [] for i in range(6)}

    # Track trajectory-level overall reconstruction error
    all_traj_errors = []
    unplausible_traj_errors = []

    print("Extracting feature errors...")
    for item in scores:
        # Skip intrinsically implausible trajectories which have [0.0]*6 default errors
        if item['reconstruction_error'] == -1.0:
            continue
            
        is_unplausible = item['is_unplausible']
        worst_tid = item.get('least_plausible_transition_id')
        
        traj_err = item['reconstruction_error']
        all_traj_errors.append(traj_err)
        if is_unplausible:
            unplausible_traj_errors.append(traj_err)
        
        feature_errors_dict = item.get('transition_feature_errors', {})
        
        for tid, f_errors in feature_errors_dict.items():
            if not f_errors or len(f_errors) != 6:
                continue
                
            for i in range(6):
                all_feature_errors[i].append(f_errors[i])
                
                if is_unplausible:
                    unplausible_feature_errors[i].append(f_errors[i])
                    
            if is_unplausible and tid == worst_tid:
                for i in range(6):
                    least_plausible_feature_errors[i].append(f_errors[i])

    print(f"Collected {len(all_feature_errors[0])} total transitions.")
    print(f"Collected {len(unplausible_feature_errors[0])} transitions from unplausible trajectories.")
    print(f"Collected {len(least_plausible_feature_errors[0])} worst transitions from unplausible trajectories.")

    # Create plots
    os.makedirs("vis", exist_ok=True)
    
    print("Generating distribution plots for all transitions...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Feature-wise Mean Squared Errors (MSE) - All Transitions', fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        if len(all_feature_errors[i]) == 0:
            continue
            
        all_errs = np.array(all_feature_errors[i])
        max_val = np.percentile(all_errs, 99)
        if max_val == 0:
            max_val = np.max(all_errs) if len(all_errs) > 0 else 1.0
            
        bins = np.linspace(0, max_val, 50)
        ax.hist(all_errs, bins=bins, alpha=0.7, label='All Transitions', density=True, color='blue')
        
        ax.set_title(feature_names[i])
        ax.set_xlabel('MSE')
        ax.set_ylabel('Density')
        ax.legend()
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_all = f"{output_prefix}_all_distribution.png"
    plt.savefig(save_path_all)
    plt.close()
    print(f"Saved plot to {save_path_all}")
    
    print("Generating distribution plots for unplausible transitions...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Feature-wise Mean Squared Errors (MSE) - Unplausible Trajectories', fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        unplaus_errs = np.array(unplausible_feature_errors[i])
        worst_errs = np.array(least_plausible_feature_errors[i])
        
        if len(unplaus_errs) == 0:
            continue
            
        max_val = np.percentile(unplaus_errs, 99)
        if max_val == 0:
            max_val = np.max(unplaus_errs) if len(unplaus_errs) > 0 else 1.0
            
        bins = np.linspace(0, max_val, 50)
        
        ax.hist(unplaus_errs, bins=bins, alpha=0.6, label='Transitions in Unplausible Traj', density=True, color='orange')
        if len(worst_errs) > 0:
            ax.hist(worst_errs, bins=bins, alpha=0.6, label='Worst Transition in Unplausible Traj', density=True, color='red')
            
        ax.set_title(feature_names[i])
        ax.set_xlabel('MSE')
        ax.set_ylabel('Density')
        ax.legend()
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_unplaus = f"{output_prefix}_unplausible_distribution.png"
    plt.savefig(save_path_unplaus)
    plt.close()
    print(f"Saved plot to {save_path_unplaus}")
    
    # Calculate and print statistics
    print("\n--- Summary Statistics (Mean MSE) ---")
    print(f"{'Feature':<15} | {'All Transitions':<15} | {'Unplausible Traj':<18} | {'Worst Transition':<15}")
    print("-" * 70)
    for i in range(6):
        mean_all = np.mean(all_feature_errors[i]) if len(all_feature_errors[i]) > 0 else 0
        mean_unplaus = np.mean(unplausible_feature_errors[i]) if len(unplausible_feature_errors[i]) > 0 else 0
        mean_worst = np.mean(least_plausible_feature_errors[i]) if len(least_plausible_feature_errors[i]) > 0 else 0
        
        print(f"{feature_names[i]:<15} | {mean_all:<15.6f} | {mean_unplaus:<18.6f} | {mean_worst:<15.6f}")

    # Plot for trajectory reconstruction errors
    print("Generating split trajectory error distribution plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Trajectory Reconstruction Error (MSE) Distribution Comparison', fontsize=16)
    
    all_traj_arr = np.array(all_traj_errors)
    unplaus_traj_arr = np.array(unplausible_traj_errors)
    
    # Left subplot: All trajectories
    max_all_val = np.percentile(all_traj_arr, 99) if len(all_traj_arr) > 0 else 1.0
    if max_all_val == 0:
        max_all_val = np.max(all_traj_arr) if len(all_traj_arr) > 0 else 1.0
    bins_all = np.linspace(0, max_all_val, 50)
    ax1.hist(all_traj_arr, bins=bins_all, alpha=0.7, label='All Trajectories', density=True, color='blue')
    ax1.set_title("All Trajectories")
    ax1.set_xlabel('MSE')
    ax1.set_ylabel('Density')
    ax1.legend()
    
    # Right subplot: Unplausible trajectories
    if len(unplaus_traj_arr) > 0:
        max_unplaus_val = np.percentile(unplaus_traj_arr, 99)
        if max_unplaus_val == 0:
            max_unplaus_val = np.max(unplaus_traj_arr) if len(unplaus_traj_arr) > 0 else 1.0
        bins_unplaus = np.linspace(0, max_unplaus_val, 50)
        ax2.hist(unplaus_traj_arr, bins=bins_unplaus, alpha=0.7, label='Unplausible Trajectories', density=True, color='orange')
        ax2.set_title("Unplausible Trajectories")
        ax2.set_xlabel('MSE')
        ax2.set_ylabel('Density')
        ax2.legend()
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_traj = f"{output_prefix}_trajectory_error_distribution.png"
    plt.savefig(save_path_traj)
    plt.close()
    print(f"Saved trajectory error plot to {save_path_traj}")
    
    print("\n--- Trajectory Reconstruction Error Statistics ---")
    print(f"All Trajectories Mean MSE: {np.mean(all_traj_arr):.6f}")
    if len(unplaus_traj_arr) > 0:
        print(f"Unplausible Trajectories Mean MSE: {np.mean(unplaus_traj_arr):.6f}")

if __name__ == "__main__":
    # You can change this to anomaly_scores_inat.jsonl if needed
    analyze_feature_errors("Models_weights_and_results/anomaly_scores_gowallav5.jsonl", "vis/gowalla_feature_errors")
