import argparse
import json
from pathlib import Path

import pandas as pd

from modeling.data import validate_training_trajectories
from modeling.experiments import expanded_runs


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_empty(frame, message):
    if not frame.empty:
        raise ValueError(f"{message}: {len(frame)} rows")


def audit_real_dataset(feature_root, processed_root, feature_set, dataset):
    feature_base = Path(feature_root) / feature_set / dataset
    processed_base = Path(processed_root) / dataset
    trajectories = pd.read_parquet(feature_base / "trajectories.parquet")
    training = validate_training_trajectories(
        trajectory_table=trajectories,
        feature_root=feature_root,
        feature_set=feature_set,
        dataset=dataset,
        processed_root=processed_root,
    )

    transition_features = pd.read_parquet(
        feature_base / "transition_features.parquet",
        columns=["transition_id"],
    )
    processed_transitions = pd.read_parquet(
        processed_base / "transitions.parquet",
        columns=["transition_id", "transition_plausibility"],
    )
    checked_transitions = transition_features.merge(
        processed_transitions,
        on="transition_id",
        how="left",
        validate="many_to_one",
    )
    assert_empty(
        checked_transitions.loc[checked_transitions["transition_plausibility"].isna()],
        f"{dataset} feature transitions missing from processed transitions",
    )
    assert_empty(
        checked_transitions.loc[checked_transitions["transition_plausibility"].ne(1)],
        f"{dataset} feature transitions include deterministic-rule failures",
    )

    return {
        "dataset": dataset,
        "feature_trajectories": int(len(trajectories)),
        "training_trajectories": int(len(training)),
        "calibration_trajectories": int(trajectories["use_for_calibration"].sum()),
        "feature_transitions": int(len(transition_features)),
    }


def audit_synthetic_dataset(feature_root, feature_set, training_ids_by_dataset):
    synthetic_path = Path(feature_root) / feature_set / "synthetic" / "trajectories.parquet"
    if not synthetic_path.exists():
        return {"dataset": "synthetic", "present": False}

    synthetic = pd.read_parquet(synthetic_path)
    if synthetic["use_for_training"].any():
        raise ValueError("Synthetic trajectories are marked use_for_training")
    if synthetic["use_for_calibration"].any():
        raise ValueError("Synthetic trajectories are marked use_for_calibration")
    if not synthetic["split"].astype(str).eq("benchmark").all():
        raise ValueError("Synthetic trajectories must all use split=benchmark")

    overlaps = {}
    for dataset, training_ids in training_ids_by_dataset.items():
        source_ids = set(
            synthetic.loc[
                synthetic["source_dataset"].astype(str).eq(str(dataset)),
                "source_trajectory_id",
            ]
            .dropna()
            .astype(int)
            .tolist()
        )
        overlap = source_ids & training_ids
        if overlap:
            overlaps[dataset] = sorted(overlap)[:10]
    if overlaps:
        raise ValueError(
            "Synthetic source trajectories overlap training trajectories: "
            + json.dumps(overlaps, sort_keys=True)
        )

    return {
        "dataset": "synthetic",
        "present": True,
        "trajectories": int(len(synthetic)),
        "source_controls": int(
            synthetic[["source_dataset", "source_trajectory_id"]]
            .drop_duplicates()
            .shape[0]
        ),
    }


def audit_experiment_config(config_path, allowed_datasets):
    experiment = load_json(config_path)
    disallowed = []
    for run in expanded_runs(experiment):
        training_datasets = set(run["training"]["datasets"])
        evaluation_datasets = set(run["evaluation"]["datasets"])
        bad = (training_datasets | evaluation_datasets) - allowed_datasets
        if bad:
            disallowed.append(
                {
                    "model_id": run["model_id"],
                    "datasets": sorted(bad),
                }
            )
    if disallowed:
        raise ValueError(
            f"{config_path} contains disallowed datasets: "
            + json.dumps(disallowed[:10], sort_keys=True)
        )
    return {
        "config": str(config_path),
        "runs": len(expanded_runs(experiment)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit leakage boundaries before fitting or reporting models."
    )
    parser.add_argument("--feature-root", default="data/features")
    parser.add_argument("--processed-root", default="data/processed_parquet")
    parser.add_argument("--feature-set", default="motion_v2")
    parser.add_argument("--datasets", nargs="+", default=["inat", "gowalla"])
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "modeling/configs/configurable_model_matrix.json",
            "modeling/configs/lstm_forecaster_experiments.json",
            "modeling/configs/lagmm_experiments.json",
        ],
    )
    args = parser.parse_args()

    training_ids_by_dataset = {}
    real_results = []
    for dataset in args.datasets:
        result = audit_real_dataset(
            args.feature_root,
            args.processed_root,
            args.feature_set,
            dataset,
        )
        real_results.append(result)
        training_ids_by_dataset[dataset] = set(
            pd.read_parquet(
                Path(args.feature_root) / args.feature_set / dataset / "trajectories.parquet",
                columns=["trajectory_id", "use_for_training"],
            )
            .loc[lambda frame: frame["use_for_training"], "trajectory_id"]
            .astype(int)
            .tolist()
        )

    allowed_datasets = set(args.datasets) | {"synthetic"}
    config_results = [
        audit_experiment_config(Path(path), allowed_datasets)
        for path in args.configs
    ]
    result = {
        "status": "ok",
        "feature_set": args.feature_set,
        "real_datasets": real_results,
        "synthetic": audit_synthetic_dataset(
            args.feature_root,
            args.feature_set,
            training_ids_by_dataset,
        ),
        "configs": config_results,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
