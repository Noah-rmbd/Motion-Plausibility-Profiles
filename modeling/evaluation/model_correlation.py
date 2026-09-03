import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.framework.score_aggregation import select_trajectory_scores

def compute_model_flags(prediction_root, dataset, anomaly_percentage):
    prediction_root = Path(prediction_root)
    model_flags = {}
    
    for model_dir in sorted(prediction_root.iterdir()):
        if not model_dir.is_dir():
            continue
            
        model_id = model_dir.name
        score_path = model_dir / dataset / "trajectory_scores.parquet"
        transition_path = model_dir / dataset / "transition_scores.parquet"
        
        if not score_path.exists():
            continue
            
        score_frame = pd.read_parquet(score_path)
        if score_frame.empty:
            continue
            
        if transition_path.exists():
            transition_frame = pd.read_parquet(transition_path)
        else:
            transition_frame = pd.DataFrame(columns=["trajectory_id", "transition_order", "score"])
            
        try:
            selected = select_trajectory_scores(
                score_frame,
                transition_frame,
                aggregation="mean",
                ignore_first_transition=False,
                transition_score_mode="mean",
            )
        except Exception as e:
            print(f"Warning: Failed to process {model_id}: {e}")
            continue
            
        scores = selected["selected_score"].to_numpy(dtype=float)
        if len(scores) == 0:
            continue
            
        threshold = np.percentile(scores, 100.0 - anomaly_percentage)
        flagged_ids = set(selected.loc[selected["selected_score"] >= threshold, "trajectory_id"])
        model_flags[model_id] = flagged_ids
        
    return model_flags

import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Compute model correlation (agreement) matrix.")
    parser.add_argument("--prediction-root", type=str, default="artifacts/predictions")
    parser.add_argument("--dataset", type=str, default="gowalla")
    parser.add_argument("--anomaly-percentage", type=float, default=5.0, help="Top N percent of trajectories to flag")
    parser.add_argument("--output-dir", type=str, default="artifacts/stats")
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.prediction_root} for dataset '{args.dataset}'")
    model_flags = compute_model_flags(args.prediction_root, args.dataset, args.anomaly_percentage)
    
    models = sorted(list(model_flags.keys()))
    if not models:
        print("No models found!")
        return
        
    print(f"Found {len(models)} models.")
    
    jaccard_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    count_matrix = pd.DataFrame(index=models, columns=models, dtype=int)
    
    for m1 in models:
        for m2 in models:
            set1 = model_flags[m1]
            set2 = model_flags[m2]
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            count_matrix.loc[m1, m2] = intersection
            jaccard_matrix.loc[m1, m2] = (intersection / union * 100) if union > 0 else 0.0
            
    print("\n" + "="*120)
    print(f"JACCARD SIMILARITY MATRIX (%) - Dataset: {args.dataset} | Top {args.anomaly_percentage}% Flagged")
    print("="*120)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1500)
    print(jaccard_matrix.round(1))
    
    print("\n" + "="*120)
    print(f"INTERSECTION COUNT MATRIX - Dataset: {args.dataset} | Top {args.anomaly_percentage}% Flagged")
    print("="*120)
    print(count_matrix)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    jaccard_out = out_dir / f"model_correlation_jaccard_{args.dataset}_{args.anomaly_percentage}.csv"
    count_out = out_dir / f"model_correlation_counts_{args.dataset}_{args.anomaly_percentage}.csv"
    
    jaccard_matrix.to_csv(jaccard_out)
    count_matrix.to_csv(count_out)
    print(f"\nSaved matrices to {out_dir}")
    
    # Plot PNG
    short_models = [m.replace("v7_unified_", "").replace("_all_motion_features_combined_e15", "") for m in models]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        jaccard_matrix, 
        annot=True, 
        fmt=".1f", 
        cmap="YlGnBu",
        xticklabels=short_models,
        yticklabels=short_models,
        cbar_kws={'label': 'Jaccard Similarity (%)'},
        square=True
    )
    plt.title(f"Model Jaccard Similarity ({args.dataset.capitalize()}, Top {args.anomaly_percentage}%)")
    plt.tight_layout()
    
    png_out = out_dir / f"model_correlation_jaccard_{args.dataset}_{args.anomaly_percentage}.png"
    plt.savefig(png_out, dpi=300)
    print(f"Saved heatmap PNG to {png_out}")

if __name__ == "__main__":
    main()
