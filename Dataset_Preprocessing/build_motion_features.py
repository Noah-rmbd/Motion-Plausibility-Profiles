import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from .cleaning import (
        build_trajectory_exclusions,
        select_valid_trajectory_ids,
    )
except ImportError:
    from cleaning import build_trajectory_exclusions, select_valid_trajectory_ids


PHYSICAL_COLUMNS = [
    "speed_kmh",
    "elapsed_time_s",
    "distance_m",
    "acceleration_m_s2",
    "bearing_change_rad",
]

FEATURE_COLUMNS = [
    "speed",
    "elapsed_time",
    "distance",
    "acceleration",
    "bearing_sin",
    "bearing_cos",
]

DEFAULT_DATASETS = ["inat", "gowalla"]


def signed_log1p(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def fit_scalers(frames):
    all_transitions = pd.concat([frame[PHYSICAL_COLUMNS] for frame in frames], ignore_index=True)

    scalers = {
        "speed": MinMaxScaler(),
        "elapsed_time": MinMaxScaler(),
        "distance": MinMaxScaler(),
        "acceleration": MinMaxScaler(),
    }

    speed_values = all_transitions["speed_kmh"].to_numpy(dtype=float)
    valid_speed_values = speed_values[speed_values <= 900.0]
    speed_fit_values = valid_speed_values if len(valid_speed_values) else speed_values
    scalers["speed"].fit(np.log1p(speed_fit_values).reshape(-1, 1))
    scalers["elapsed_time"].fit(np.log1p(all_transitions["elapsed_time_s"]).to_numpy().reshape(-1, 1))
    scalers["distance"].fit(np.log1p(all_transitions["distance_m"]).to_numpy().reshape(-1, 1))
    scalers["acceleration"].fit(signed_log1p(all_transitions["acceleration_m_s2"]).reshape(-1, 1))

    return scalers


def transform_features(transitions, scalers):
    features = pd.DataFrame(index=transitions.index)
    features["speed"] = scalers["speed"].transform(
        np.log1p(transitions["speed_kmh"]).to_numpy().reshape(-1, 1)
    ).reshape(-1)
    features["elapsed_time"] = scalers["elapsed_time"].transform(
        np.log1p(transitions["elapsed_time_s"]).to_numpy().reshape(-1, 1)
    ).reshape(-1)
    features["distance"] = scalers["distance"].transform(
        np.log1p(transitions["distance_m"]).to_numpy().reshape(-1, 1)
    ).reshape(-1)
    features["acceleration"] = scalers["acceleration"].transform(
        signed_log1p(transitions["acceleration_m_s2"]).reshape(-1, 1)
    ).reshape(-1)
    features["bearing_sin"] = (np.sin(transitions["bearing_change_rad"].to_numpy(dtype=float)) + 1.0) / 2.0
    features["bearing_cos"] = (np.cos(transitions["bearing_change_rad"].to_numpy(dtype=float)) + 1.0) / 2.0
    return features


def read_real_dataset(processed_dir, dataset):
    base = Path(processed_dir) / dataset
    transitions = pd.read_parquet(
        base / "transitions.parquet",
        columns=["transition_id", *PHYSICAL_COLUMNS, "transition_plausibility", "plausibility_reason"],
    )
    trajectory_transitions = pd.read_parquet(
        base / "trajectory_transitions.parquet",
        columns=["trajectory_id", "transition_order", "transition_id"],
    )
    trajectories = pd.read_parquet(
        base / "trajectories.parquet",
        columns=["trajectory_id", "user_id", "n_transitions", "start_observation_id", "end_observation_id"],
    )
    return transitions, trajectory_transitions, trajectories


def select_valid_real_trajectories(transitions, trajectory_transitions, trajectories, min_transitions):
    valid_ids, valid_by_transitions = select_valid_trajectory_ids(
        transitions,
        trajectory_transitions,
        trajectories,
        min_transitions,
    )
    exclusions = build_trajectory_exclusions(
        trajectories,
        valid_ids,
        valid_by_transitions,
        min_transitions,
    )
    return valid_ids, exclusions


def build_real_feature_dataset(dataset, processed_dir, output_dir, scalers, min_transitions):
    transitions, trajectory_transitions, trajectories = read_real_dataset(processed_dir, dataset)
    valid_ids, exclusions = select_valid_real_trajectories(
        transitions,
        trajectory_transitions,
        trajectories,
        min_transitions,
    )

    valid_mapping = trajectory_transitions.loc[trajectory_transitions["trajectory_id"].isin(valid_ids)].copy()
    valid_transitions = valid_mapping.merge(transitions.drop(columns=["plausibility_reason"]), on="transition_id", how="left")
    valid_transitions = valid_transitions.sort_values(["trajectory_id", "transition_order"])

    feature_values = transform_features(valid_transitions, scalers)
    transition_features = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "trajectory_id": valid_transitions["trajectory_id"].to_numpy(),
                    "transition_order": valid_transitions["transition_order"].to_numpy(),
                    "transition_id": valid_transitions["transition_id"].to_numpy(),
                }
            ),
            feature_values.reset_index(drop=True),
        ],
        axis=1,
    )

    trajectory_features = trajectories.loc[trajectories["trajectory_id"].isin(valid_ids)].copy()
    trajectory_features.insert(0, "dataset", dataset)
    trajectory_features["source"] = "real"
    trajectory_features["use_for_training"] = True
    trajectory_features["is_synthetic_anomaly"] = False
    trajectory_features["anomaly_profile"] = None
    trajectory_features = trajectory_features.sort_values("trajectory_id")

    dataset_dir = Path(output_dir) / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    transition_features.to_parquet(dataset_dir / "transition_features.parquet", index=False, compression="zstd")
    trajectory_features.to_parquet(dataset_dir / "trajectories.parquet", index=False, compression="zstd")

    exclusions.insert(0, "dataset", dataset)
    return {
        "dataset": dataset,
        "valid_trajectories": int(len(trajectory_features)),
        "valid_transitions": int(len(transition_features)),
        "excluded_trajectories": int(len(exclusions)),
    }, exclusions


def collect_valid_real_transitions(processed_dir, dataset, min_transitions):
    transitions, trajectory_transitions, trajectories = read_real_dataset(processed_dir, dataset)
    valid_ids, exclusions = select_valid_real_trajectories(
        transitions,
        trajectory_transitions,
        trajectories,
        min_transitions,
    )
    valid_transition_ids = trajectory_transitions.loc[
        trajectory_transitions["trajectory_id"].isin(valid_ids),
        "transition_id",
    ]
    valid_transitions = transitions.loc[transitions["transition_id"].isin(valid_transition_ids), PHYSICAL_COLUMNS].copy()
    return valid_transitions, exclusions


def build_synthetic_feature_dataset(processed_dir, output_dir, scalers):
    transitions, trajectory_transitions, trajectories = read_real_dataset(processed_dir, "synthetic")
    metadata_path = Path(processed_dir) / "synthetic" / "synthetic_metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} is missing. Generate synthetic data with "
            "dataset_preprocessing.synthetic_anomalies before building features."
        )
    synthetic_metadata = pd.read_parquet(metadata_path)
    synthetic_transitions = trajectory_transitions.merge(
        transitions.drop(columns=["transition_plausibility", "plausibility_reason"]),
        on="transition_id",
        how="left",
    )
    synthetic_transitions = synthetic_transitions.sort_values(["trajectory_id", "transition_order"])

    feature_values = transform_features(synthetic_transitions, scalers)
    transition_features = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": "synthetic",
                    "trajectory_id": synthetic_transitions["trajectory_id"].to_numpy(),
                    "transition_order": synthetic_transitions["transition_order"].to_numpy(),
                    "transition_id": synthetic_transitions["transition_id"].to_numpy(),
                }
            ),
            feature_values.reset_index(drop=True),
        ],
        axis=1,
    )
    trajectory_features = trajectories.merge(
        synthetic_metadata.drop(columns=["dataset"]),
        on="trajectory_id",
        how="left",
        validate="one_to_one",
    ).sort_values("trajectory_id")
    trajectory_features.insert(0, "dataset", "synthetic")
    trajectory_features["source"] = "synthetic"
    trajectory_features["use_for_training"] = False
    trajectory_features["is_synthetic_anomaly"] = True
    if trajectory_features["anomaly_profile"].isna().any():
        raise ValueError("Synthetic metadata is missing for one or more trajectories")

    dataset_dir = Path(output_dir) / "synthetic"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    transition_features.to_parquet(dataset_dir / "transition_features.parquet", index=False, compression="zstd")
    trajectory_features.to_parquet(dataset_dir / "trajectories.parquet", index=False, compression="zstd")

    return {
        "dataset": "synthetic",
        "valid_trajectories": int(len(trajectory_features)),
        "valid_transitions": int(len(transition_features)),
        "excluded_trajectories": 0,
    }


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build normalized motion features from canonical Parquet data.")
    parser.add_argument("--processed-dir", default="data/processed_parquet")
    parser.add_argument("--feature-set", default="motion_v1")
    parser.add_argument("--output-root", default="data/features")
    parser.add_argument("--scaler-root", default="artifacts/scalers")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--min-transitions", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_root) / args.feature_set
    scaler_path = Path(args.scaler_root) / f"{args.feature_set}.pkl"
    run_dir = Path(args.run_root) / args.feature_set

    valid_real_frames = []
    all_exclusions = []
    prefit_summaries = []
    for dataset in args.datasets:
        valid_transitions, exclusions = collect_valid_real_transitions(
            args.processed_dir,
            dataset,
            args.min_transitions,
        )
        valid_real_frames.append(valid_transitions)
        exclusions.insert(0, "dataset", dataset)
        all_exclusions.append(exclusions)
        prefit_summaries.append(
            {
                "dataset": dataset,
                "valid_transitions_for_scaler": int(len(valid_transitions)),
                "excluded_trajectories": int(len(exclusions)),
            }
        )

    scalers = fit_scalers(valid_real_frames)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_path.open("wb") as handle:
        pickle.dump(scalers, handle)

    summaries = []
    for dataset in args.datasets:
        summary, _ = build_real_feature_dataset(
            dataset,
            args.processed_dir,
            output_dir,
            scalers,
            args.min_transitions,
        )
        summaries.append(summary)

    if not args.skip_synthetic:
        summaries.append(build_synthetic_feature_dataset(args.processed_dir, output_dir, scalers))

    run_dir.mkdir(parents=True, exist_ok=True)
    if all_exclusions:
        pd.concat(all_exclusions, ignore_index=True).to_parquet(
            run_dir / "exclusions.parquet",
            index=False,
            compression="zstd",
        )

    config = {
        "feature_set": args.feature_set,
        "feature_columns": FEATURE_COLUMNS,
        "physical_columns": PHYSICAL_COLUMNS,
        "datasets": args.datasets,
        "synthetic_enabled": not args.skip_synthetic,
        "min_transitions": args.min_transitions,
        "cleaning_policy": {
            "real_training_data": "trajectory has at least min_transitions and all transitions are rule-plausible",
            "synthetic_data": "transformed with fitted real-data scaler; never used to fit scalers",
        },
        "scaler_path": str(scaler_path),
        "prefit_summary": prefit_summaries,
        "output_summary": summaries,
    }
    write_json(run_dir / "config.json", config)

    print(json.dumps(config["output_summary"], indent=2))


if __name__ == "__main__":
    main()
