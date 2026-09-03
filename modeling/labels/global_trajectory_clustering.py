import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler

def log(message):
    print(f"[global_trajectory_clustering] {message}", flush=True)

FEATURES = [
    "efficiency",
    "detour",
    "reversal_freq",
    "turning_entropy",
    "turn_rate",
    "mean_turn_angle",
    "convex_hull_area",
    "radius_of_gyration",
    "num_stops",
    "duration_stops",
    "stop_ratio",
    "median_speed",
    "var_speed",
    "compression_ratio",
    "temporal_sparsity",
    "displacement_ratio"
]

def load_features(features_dir, dataset):
    path = Path(features_dir) / dataset / "trajectory_features.parquet"
    if not path.exists():
        log(f"Warning: {path} not found.")
        return None
    return pd.read_parquet(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features/global_v1")
    parser.add_argument("--predictions-dir", type=str, default="artifacts/predictions")
    parser.add_argument("--datasets", nargs="+", default=["inat", "gowalla", "synthetic"])
    parser.add_argument("--comparison-group", type=str, default="unified_evaluation")
    args = parser.parse_args()

    # We will train on inat and gowalla (combined) using use_for_training == True
    # Then predict on all requested datasets
    
    train_dfs = []
    eval_dfs = {}
    
    for ds in args.datasets:
        df = load_features(args.features_dir, ds)
        if df is not None:
            eval_dfs[ds] = df
            if ds in ["inat", "gowalla"]:
                train_dfs.append(df[df["use_for_training"]])
                
    if not train_dfs:
        log("No training data found.")
        return
        
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # Check for NaNs and fill with 0 (e.g. from 0 division)
    X_train = train_df[FEATURES].fillna(0).values
    
    log(f"Fitting scaler on {len(X_train)} trajectories...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    log(f"Training Isolation Forest...")
    iforest = IsolationForest(random_state=42, contamination=0.05)
    iforest.fit(X_train_scaled)
    
    log(f"Training OneClassSVM (subsampled to max 10000 samples)...")
    if len(X_train_scaled) > 10000:
        rng = np.random.default_rng(42)
        svm_idx = rng.choice(len(X_train_scaled), size=10000, replace=False)
        X_svm_train = X_train_scaled[svm_idx]
    else:
        X_svm_train = X_train_scaled
        
    ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    ocsvm.fit(X_svm_train)
    
    models = {
        "global_iforest": (iforest, lambda m, X: -m.decision_function(X)), # Negative because higher score must mean more anomalous
        "global_ocsvm": (ocsvm, lambda m, X: -m.decision_function(X))
    }
    
    for model_id, (model, score_fn) in models.items():
        # Write config.json so build_database.py can pick it up
        model_artifacts_dir = Path("artifacts/models") / model_id
        model_artifacts_dir.mkdir(parents=True, exist_ok=True)
        config_path = model_artifacts_dir / "config.json"
        
        config = {
            "model_type": f"global_clustering_{model_id.split('_')[1]}",
            "feature_set": "global_v1",
            "features": FEATURES,
            "architecture": {
                "algorithm": model_id.split("_")[1],
                "trajectory_score": "Global clustering anomaly"
            },
            "training": {"datasets": ["inat", "gowalla"]},
            "evaluation": {"datasets": args.datasets},
            "experiment": {"comparison_group": args.comparison_group}
        }
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

    for ds, df in eval_dfs.items():
        X_eval = df[FEATURES].fillna(0).values
        X_eval_scaled = scaler.transform(X_eval)
        
        for model_id, (model, score_fn) in models.items():
            scores = score_fn(model, X_eval_scaled)
            
            out_df = pd.DataFrame({
                "model_id": model_id,
                "dataset": ds,
                "trajectory_id": df["trajectory_id"],
                "score": scores,
                "score_type": "global_clustering"
            })
            
            out_dir = Path(args.predictions_dir) / model_id / ds
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "trajectory_scores.parquet"
            out_df.to_parquet(out_path)
            log(f"Saved {len(out_df)} predictions for {model_id} on {ds} to {out_path}")
            
            empty_trans_df = pd.DataFrame(columns=[
                "model_id", "dataset", "trajectory_id", 
                "transition_order", "transition_id", "score"
            ])
            empty_trans_df.to_parquet(out_dir / "transition_scores.parquet")

if __name__ == "__main__":
    main()
