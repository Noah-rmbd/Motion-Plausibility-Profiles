import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .cleaning import (
        build_trajectory_exclusions,
        select_valid_trajectory_ids,
    )
    from .splitting import (
        DEFAULT_CALIBRATION_FRACTION,
        DEFAULT_SPLIT_SEED,
        user_split,
    )
    from modeling.plausible_labels import (
        DEFAULT_LABEL_PATH,
        resolve_plausible_trajectories_from_processed,
    )
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from cleaning import build_trajectory_exclusions, select_valid_trajectory_ids
    from splitting import (
        DEFAULT_CALIBRATION_FRACTION,
        DEFAULT_SPLIT_SEED,
        user_split,
    )
    from modeling.plausible_labels import (
        DEFAULT_LABEL_PATH,
        resolve_plausible_trajectories_from_processed,
    )


PHYSICAL_COLUMNS = [
    "speed_kmh",
    "elapsed_time_s",
    "distance_m",
    "acceleration_m_s2",
    "bearing_change_rad",
]

FEATURE_COLUMNS = [
    "speed",
    "speed_stationary",
    "speed_0_5",
    "speed_5_10",
    "speed_10_25",
    "speed_25_80",
    "speed_80_140",
    "speed_140_200",
    "speed_200_plus",
    "elapsed_time",
    "distance",
    "acceleration",
    "acceleration_valid",
    "bearing_sin",
    "bearing_cos",
    "bearing_valid",
]

DEFAULT_DATASETS = ["inat", "gowalla"]


def signed_log1p(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def transform_features(transitions):
    features = pd.DataFrame(index=transitions.index)
    speed = transitions["speed_kmh"].to_numpy(dtype=float)
    features["speed"] = np.log1p(speed)
    features["speed_stationary"] = (speed == 0.0).astype(np.float32)
    features["speed_0_5"] = ((speed > 0.0) & (speed < 5.0)).astype(np.float32)
    features["speed_5_10"] = ((speed >= 5.0) & (speed < 10.0)).astype(np.float32)
    features["speed_10_25"] = ((speed >= 10.0) & (speed < 25.0)).astype(np.float32)
    features["speed_25_80"] = ((speed >= 25.0) & (speed < 80.0)).astype(np.float32)
    features["speed_80_140"] = ((speed >= 80.0) & (speed < 140.0)).astype(np.float32)
    features["speed_140_200"] = ((speed >= 140.0) & (speed < 200.0)).astype(np.float32)
    features["speed_200_plus"] = (speed >= 200.0).astype(np.float32)
    features["elapsed_time"] = np.log1p(
        transitions["elapsed_time_s"].to_numpy(dtype=float)
    )
    features["distance"] = np.log1p(
        transitions["distance_m"].to_numpy(dtype=float)
    )
    acceleration_valid = transitions["acceleration_valid"].to_numpy(dtype=bool)
    acceleration = signed_log1p(
        transitions["acceleration_m_s2"].to_numpy(dtype=float)
    )
    features["acceleration"] = np.where(acceleration_valid, acceleration, 0.0)
    features["acceleration_valid"] = acceleration_valid.astype(np.float32)
    bearing_valid = transitions["bearing_change_valid"].to_numpy(dtype=bool)
    bearing = transitions["bearing_change_rad"].to_numpy(dtype=float)
    features["bearing_sin"] = np.where(bearing_valid, np.sin(bearing), 0.0)
    features["bearing_cos"] = np.where(bearing_valid, np.cos(bearing), 0.0)
    features["bearing_valid"] = bearing_valid.astype(np.float32)
    return features


def read_real_dataset(processed_dir, dataset):
    base = Path(processed_dir) / dataset
    transitions = pd.read_parquet(
        base / "transitions.parquet",
        columns=[
            "transition_id",
            *PHYSICAL_COLUMNS,
            "acceleration_valid",
            "bearing_change_valid",
            "transition_plausibility",
            "plausibility_reason",
        ],
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


def assign_splits(
    dataset,
    trajectories,
    calibration_fraction,
    split_seed,
):
    return trajectories["user_id"].map(
        lambda user_id: user_split(
            dataset,
            user_id,
            calibration_fraction=calibration_fraction,
            seed=split_seed,
        )
    )


def build_real_feature_dataset(
    dataset,
    processed_dir,
    output_dir,
    min_transitions,
    calibration_fraction,
    split_seed,
    plausible_labels_path,
):
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

    feature_values = transform_features(valid_transitions)
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
    trajectory_features["split"] = assign_splits(
        dataset,
        trajectory_features,
        calibration_fraction,
        split_seed,
    )
    reviewed_plausible_ids = set()
    if plausible_labels_path and Path(plausible_labels_path).exists():
        reviewed = resolve_plausible_trajectories_from_processed(
            processed_root=processed_dir,
            label_path=plausible_labels_path,
        )
        reviewed_plausible_ids = set(
            reviewed.loc[
                reviewed["dataset"].astype(str).eq(str(dataset)),
                "trajectory_id",
            ].astype(int)
        )
        reviewed_plausible_ids &= set(trajectory_features["trajectory_id"])
        trajectory_features.loc[
            trajectory_features["trajectory_id"].isin(reviewed_plausible_ids),
            "split",
        ] = "calibration"
    trajectory_features["use_for_training"] = trajectory_features["split"].eq("fit")
    trajectory_features["use_for_calibration"] = trajectory_features["split"].eq(
        "calibration"
    )
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
        "reviewed_plausible_trajectories": int(len(reviewed_plausible_ids)),
    }, exclusions


def build_synthetic_feature_dataset(processed_dir, output_dir):
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

    feature_values = transform_features(synthetic_transitions)
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
    trajectory_features["split"] = "benchmark"
    trajectory_features["use_for_training"] = False
    trajectory_features["use_for_calibration"] = False
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
    parser = argparse.ArgumentParser(
        description="Build unscaled model features from canonical Parquet data."
    )
    parser.add_argument("--processed-dir", default="data/processed_parquet")
    parser.add_argument("--feature-set", default="motion_v2")
    parser.add_argument("--output-root", default="data/features")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--min-transitions", type=int, default=2)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--plausible-labels", default=str(DEFAULT_LABEL_PATH))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_root) / args.feature_set
    run_dir = Path(args.run_root) / args.feature_set

    all_exclusions = []
    summaries = []
    for dataset in args.datasets:
        summary, exclusions = build_real_feature_dataset(
            dataset,
            args.processed_dir,
            output_dir,
            args.min_transitions,
            args.calibration_fraction,
            args.split_seed,
            args.plausible_labels,
        )
        summaries.append(summary)
        all_exclusions.append(exclusions)

    if not args.skip_synthetic:
        summaries.append(build_synthetic_feature_dataset(args.processed_dir, output_dir))

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
        "calibration_fraction": args.calibration_fraction,
        "split_seed": args.split_seed,
        "plausible_labels": args.plausible_labels,
        "cleaning_policy": {
            "real_training_data": (
                "trajectory has at least min_transitions, all transitions are "
                "rule-plausible, and is not manually reviewed plausible"
            ),
            "synthetic_data": "same deterministic feature transform; never used to fit model scalers",
        },
        "scaling_policy": (
            "unscaled log/angle features are stored once; every model fits a RobustScaler "
            "on its selected real fit population and stores it beside the model"
        ),
        "output_summary": summaries,
    }
    write_json(run_dir / "config.json", config)

    print(json.dumps(config["output_summary"], indent=2))


if __name__ == "__main__":
    main()
