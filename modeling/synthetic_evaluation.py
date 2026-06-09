import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def evaluate_synthetic_predictions(
    model_id,
    prediction_root="artifacts/predictions",
    feature_root="data/features",
    feature_set="motion_v1",
    reference_datasets=("inat", "gowalla"),
    anomaly_percentages=(1.0, 5.0, 10.0),
):
    model_prediction_dir = Path(prediction_root) / model_id
    synthetic_scores = pd.read_parquet(
        model_prediction_dir / "synthetic" / "trajectory_scores.parquet",
        columns=["trajectory_id", "score"],
    )
    synthetic_trajectories = pd.read_parquet(
        Path(feature_root) / feature_set / "synthetic" / "trajectories.parquet",
        columns=["trajectory_id", "anomaly_profile"],
    )
    synthetic = synthetic_scores.merge(
        synthetic_trajectories,
        on="trajectory_id",
        how="left",
        validate="one_to_one",
    )
    if synthetic["anomaly_profile"].isna().any():
        raise ValueError("Synthetic prediction rows are missing anomaly profile metadata")

    reference_frames = []
    for dataset in reference_datasets:
        path = model_prediction_dir / dataset / "trajectory_scores.parquet"
        reference_frames.append(pd.read_parquet(path, columns=["score"]).assign(dataset=dataset))
    reference = pd.concat(reference_frames, ignore_index=True)
    if reference.empty or synthetic.empty:
        raise ValueError("Reference and synthetic prediction sets must both be non-empty")

    labels = np.concatenate(
        [
            np.zeros(len(reference), dtype=np.int8),
            np.ones(len(synthetic), dtype=np.int8),
        ]
    )
    scores = np.concatenate(
        [
            reference["score"].to_numpy(dtype=float),
            synthetic["score"].to_numpy(dtype=float),
        ]
    )
    metrics = {
        "model_id": model_id,
        "feature_set": feature_set,
        "reference_datasets": list(reference_datasets),
        "reference_trajectories": len(reference),
        "synthetic_trajectories": len(synthetic),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "synthetic_score_mean": float(synthetic["score"].mean()),
        "synthetic_score_median": float(synthetic["score"].median()),
        "threshold_metrics": [],
    }

    reference_scores = reference["score"].to_numpy(dtype=float)
    for percentage in anomaly_percentages:
        if not 0.0 < percentage < 100.0:
            raise ValueError(f"Anomaly percentage must be between 0 and 100: {percentage}")
        threshold = float(np.percentile(reference_scores, 100.0 - percentage))
        detected = synthetic["score"] >= threshold
        profile_recall = (
            synthetic.assign(detected=detected)
            .groupby("anomaly_profile", sort=True)["detected"]
            .mean()
        )
        metrics["threshold_metrics"].append(
            {
                "anomaly_percentage": float(percentage),
                "threshold": threshold,
                "recall": float(detected.mean()),
                "recall_by_profile": {
                    profile: float(recall)
                    for profile, recall in profile_recall.items()
                },
            }
        )

    output_path = model_prediction_dir / "synthetic" / "metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic anomaly scores against real reference scores."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prediction-root", default="artifacts/predictions")
    parser.add_argument("--feature-root", default="data/features")
    parser.add_argument("--feature-set", default="motion_v1")
    parser.add_argument("--reference-datasets", nargs="+", default=["inat", "gowalla"])
    parser.add_argument("--anomaly-percentages", nargs="+", type=float, default=[1.0, 5.0, 10.0])
    args = parser.parse_args()

    metrics = evaluate_synthetic_predictions(
        model_id=args.model_id,
        prediction_root=args.prediction_root,
        feature_root=args.feature_root,
        feature_set=args.feature_set,
        reference_datasets=args.reference_datasets,
        anomaly_percentages=args.anomaly_percentages,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
