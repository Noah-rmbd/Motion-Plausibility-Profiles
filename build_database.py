import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from modeling.experiments import expanded_runs
from modeling.score_aggregation import (
    AGGREGATION_COLUMNS,
    TRANSITION_SCORE_MODES,
    aggregate_transition_scores,
    select_trajectory_scores,
    uses_native_trajectory_score,
)


SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE datasets (
    dataset TEXT PRIMARY KEY,
    observation_count INTEGER NOT NULL,
    transition_count INTEGER NOT NULL,
    trajectory_count INTEGER NOT NULL
);

CREATE TABLE users (
    dataset TEXT NOT NULL,
    user_id TEXT NOT NULL,
    observation_count INTEGER NOT NULL,
    trajectory_count INTEGER NOT NULL,
    first_timestamp TEXT,
    last_timestamp TEXT,
    PRIMARY KEY (dataset, user_id)
);

CREATE TABLE trajectories (
    dataset TEXT NOT NULL,
    trajectory_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    n_transitions INTEGER NOT NULL,
    start_observation_id TEXT NOT NULL,
    end_observation_id TEXT NOT NULL,
    start_timestamp TEXT,
    end_timestamp TEXT,
    PRIMARY KEY (dataset, trajectory_id)
);

CREATE TABLE models (
    model_id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    feature_set TEXT NOT NULL,
    features_json TEXT NOT NULL,
    architecture_json TEXT NOT NULL,
    training_json TEXT NOT NULL,
    evaluation_json TEXT NOT NULL,
    experiment_json TEXT NOT NULL,
    prediction_root TEXT NOT NULL,
    final_loss REAL
);

CREATE TABLE model_datasets (
    model_id TEXT NOT NULL,
    dataset TEXT NOT NULL,
    score_count INTEGER NOT NULL,
    score_min REAL NOT NULL,
    score_max REAL NOT NULL,
    score_mean REAL NOT NULL,
    threshold_curve_json TEXT NOT NULL,
    PRIMARY KEY (model_id, dataset)
);

CREATE INDEX idx_users_dataset_count ON users(dataset, observation_count DESC);
CREATE INDEX idx_users_search ON users(dataset, user_id);
CREATE INDEX idx_trajectories_user ON trajectories(dataset, user_id, start_timestamp);
"""


def parquet_count(path):
    return pq.ParquetFile(path).metadata.num_rows


def insert_frame(conn, table, frame):
    frame.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1_000)


def import_parquet_batches(conn, path, table, columns, transform=None, batch_size=100_000):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        if transform:
            frame = transform(frame)
        insert_frame(conn, table, frame)


def import_dataset(conn, processed_root, dataset):
    base = Path(processed_root) / dataset
    print(f"Importing canonical dataset: {dataset}")

    trajectories = pd.read_parquet(base / "trajectories.parquet")
    trajectories["user_id"] = trajectories["user_id"].astype(str)
    required_observation_ids = set(trajectories["start_observation_id"])
    required_observation_ids.update(trajectories["end_observation_id"])
    observation_times = {}
    user_aggregates = {}
    observations_parquet = pq.ParquetFile(base / "observations.parquet")
    for batch in observations_parquet.iter_batches(
        batch_size=100_000,
        columns=["user_id", "observation_id", "timestamp"],
    ):
        frame = batch.to_pandas()
        frame["user_id"] = frame["user_id"].astype(str)
        for user_id, group in frame.groupby("user_id", sort=False):
            timestamps = group["timestamp"].dropna()
            aggregate = user_aggregates.setdefault(
                user_id,
                {"observation_count": 0, "first_timestamp": None, "last_timestamp": None},
            )
            aggregate["observation_count"] += len(group)
            if not timestamps.empty:
                batch_min = timestamps.min()
                batch_max = timestamps.max()
                aggregate["first_timestamp"] = min(
                    value for value in (aggregate["first_timestamp"], batch_min) if value is not None
                )
                aggregate["last_timestamp"] = max(
                    value for value in (aggregate["last_timestamp"], batch_max) if value is not None
                )
        selected = frame.loc[frame["observation_id"].isin(required_observation_ids)]
        observation_times.update(zip(selected["observation_id"], selected["timestamp"]))

    trajectories["start_timestamp"] = trajectories["start_observation_id"].map(observation_times)
    trajectories["end_timestamp"] = trajectories["end_observation_id"].map(observation_times)
    insert_frame(
        conn,
        "trajectories",
        trajectories[
            [
                "dataset",
                "trajectory_id",
                "user_id",
                "n_transitions",
                "start_observation_id",
                "end_observation_id",
                "start_timestamp",
                "end_timestamp",
            ]
        ],
    )
    trajectory_counts = trajectories.groupby("user_id").size().to_dict()
    users = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "user_id": user_id,
                "observation_count": values["observation_count"],
                "trajectory_count": trajectory_counts.get(user_id, 0),
                "first_timestamp": values["first_timestamp"],
                "last_timestamp": values["last_timestamp"],
            }
            for user_id, values in user_aggregates.items()
        ]
    )
    insert_frame(conn, "users", users)
    conn.execute(
        "INSERT INTO datasets VALUES (?, ?, ?, ?)",
        (
            dataset,
            parquet_count(base / "observations.parquet"),
            parquet_count(base / "transitions.parquet"),
            parquet_count(base / "trajectories.parquet"),
        ),
    )
    conn.commit()


def threshold_curve(scores):
    anomaly_percentages = np.linspace(0.1, 25.0, 250)
    thresholds = np.percentile(scores, 100.0 - anomaly_percentages)
    return [
        {"percentage": round(float(percentage), 1), "threshold": float(threshold)}
        for percentage, threshold in zip(anomaly_percentages, thresholds)
    ]


def import_models(
    conn,
    model_root,
    prediction_root,
    feature_root,
    comparison_group=None,
    datasets=None,
    model_ids=None,
    model_types=None,
):
    model_root = Path(model_root)
    prediction_root = Path(prediction_root)
    allowed_datasets = set(datasets or [])
    allowed_model_ids = set(model_ids or [])
    allowed_model_types = set(model_types or [])
    for config_path in sorted(model_root.glob("*/config.json")):
        model_dir = config_path.parent
        model_id = model_dir.name
        if allowed_model_ids and model_id not in allowed_model_ids:
            continue
        prediction_dir = prediction_root / model_id
        if not prediction_dir.exists():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if allowed_model_types and config.get("model_type") not in allowed_model_types:
            continue
        if (
            comparison_group
            and config.get("experiment", {}).get("comparison_group")
            != comparison_group
        ):
            continue
        if allowed_datasets:
            training_datasets = set(config.get("training", {}).get("datasets", []))
            if training_datasets - allowed_datasets:
                continue
        history_path = model_dir / "training_history.parquet"
        final_loss = None
        if history_path.exists():
            history = pd.read_parquet(history_path)
            if not history.empty:
                final_loss = float(history.iloc[-1]["loss"])
        conn.execute(
            "INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                model_id,
                config["model_type"],
                config["feature_set"],
                json.dumps(config["features"]),
                json.dumps(config.get("architecture", {})),
                json.dumps(config["training"]),
                json.dumps(config["evaluation"]),
                json.dumps(config.get("experiment", {})),
                str(prediction_root),
                final_loss,
            ),
        )

        for score_path in sorted(prediction_dir.glob("*/trajectory_scores.parquet")):
            dataset = score_path.parent.name
            if allowed_datasets and dataset not in allowed_datasets:
                continue
            feature_dataset_dir = (
                Path(feature_root)
                / config["feature_set"]
                / dataset
            )
            feature_paths = [
                feature_dataset_dir / "trajectories.parquet",
                feature_dataset_dir / "transition_features.parquet",
            ]
            if (
                dataset == "synthetic"
                and any(
                    path.exists()
                    and path.stat().st_mtime > score_path.stat().st_mtime
                    for path in feature_paths
                )
            ):
                print(
                    f"Skipping stale synthetic predictions: {model_id}"
                )
                continue
            score_frame = pd.read_parquet(score_path)
            transition_frame = pd.read_parquet(
                score_path.with_name("transition_scores.parquet")
            )
            selected_scores = select_trajectory_scores(
                score_frame,
                transition_frame,
                aggregation="mean",
                ignore_first_transition=False,
                transition_score_mode="mean",
            )
            scores = selected_scores["selected_score"].to_numpy(dtype=float)
            if not len(score_frame):
                continue
            calibration_path = (
                Path(feature_root)
                / config["feature_set"]
                / dataset
                / "trajectories.parquet"
            )
            calibration_ids = None
            if calibration_path.exists() and dataset != "synthetic":
                calibration_frame = pd.read_parquet(
                    calibration_path,
                    columns=["trajectory_id", "use_for_calibration"],
                )
                calibration_ids = set(
                    calibration_frame.loc[
                        calibration_frame["use_for_calibration"],
                        "trajectory_id",
                    ]
                )
            curves = {}
            if uses_native_trajectory_score(score_frame):
                native_scores = select_trajectory_scores(
                    score_frame,
                    transition_frame,
                    aggregation="mean",
                    ignore_first_transition=False,
                    transition_score_mode="mean",
                )
                if calibration_ids is not None:
                    native_scores = native_scores.loc[
                        native_scores["trajectory_id"].isin(calibration_ids)
                    ]
                native_curve = threshold_curve(
                    native_scores["selected_score"].to_numpy(dtype=float)
                )
                for score_mode in TRANSITION_SCORE_MODES:
                    curves[score_mode] = {}
                    for first_key in ("included", "excluded"):
                        curves[score_mode][first_key] = {
                            aggregation: native_curve
                            for aggregation in AGGREGATION_COLUMNS
                        }
            else:
                for score_mode in TRANSITION_SCORE_MODES:
                    curves[score_mode] = {}
                    for first_key in ("included", "excluded"):
                        curves[score_mode][first_key] = {}
                        aggregated = aggregate_transition_scores(
                            transition_frame,
                            ignore_first_transition=first_key == "excluded",
                            transition_score_mode=score_mode,
                        )
                        if calibration_ids is not None:
                            aggregated = aggregated.loc[
                                aggregated["trajectory_id"].isin(calibration_ids)
                            ]
                        for aggregation, score_column in AGGREGATION_COLUMNS.items():
                            curves[score_mode][first_key][aggregation] = threshold_curve(
                                aggregated[score_column].to_numpy(dtype=float)
                            )
            conn.execute(
                "INSERT INTO model_datasets VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    dataset,
                    len(scores),
                    float(scores.min()),
                    float(scores.max()),
                    float(scores.mean()),
                    json.dumps(curves),
                ),
            )
        conn.commit()
        print(f"Indexed model: {model_id}")


def build_database(
    db_path,
    processed_root,
    model_root,
    prediction_root,
    feature_root,
    comparison_group,
    datasets,
    experiment_config=None,
    model_ids=None,
    model_types=None,
    artifact_sources=None,
):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = db_path.with_suffix(f"{db_path.suffix}.tmp")
    temp_path.unlink(missing_ok=True)
    selected_model_ids = set(model_ids or [])
    indexed_model_ids = None
    if experiment_config:
        experiment_path = Path(experiment_config)
        with experiment_path.open("r", encoding="utf-8") as handle:
            experiment = json.load(handle)
        indexed_model_ids = [run["model_id"] for run in expanded_runs(experiment)]
    if selected_model_ids:
        if indexed_model_ids is None:
            indexed_model_ids = sorted(selected_model_ids)
        else:
            indexed_model_ids = [
                model_id
                for model_id in indexed_model_ids
                if model_id in selected_model_ids
            ]
    sources = artifact_sources or [(model_root, prediction_root)]

    conn = sqlite3.connect(temp_path)
    try:
        conn.executescript(SCHEMA)
        for dataset in datasets:
            import_dataset(conn, processed_root, dataset)
        for source_model_root, source_prediction_root in sources:
            import_models(
                conn,
                source_model_root,
                source_prediction_root,
                feature_root,
                comparison_group,
                datasets,
                indexed_model_ids,
                model_types,
            )
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()
    temp_path.replace(db_path)
    print(f"Built visualization database: {db_path}")


def parse_artifact_source(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--artifact-source must be formatted as model_root=prediction_root"
        )
    model_root, prediction_root = value.split("=", 1)
    if not model_root or not prediction_root:
        raise argparse.ArgumentTypeError(
            "--artifact-source requires both model_root and prediction_root"
        )
    return model_root, prediction_root


def main():
    parser = argparse.ArgumentParser(
        description="Build the visualization SQLite cache from canonical Parquet artifacts."
    )
    parser.add_argument("--db-path", default="artifacts/app/plausibility.db")
    parser.add_argument("--processed-root", default="data/processed_parquet")
    parser.add_argument("--model-root", default="artifacts/models")
    parser.add_argument("--prediction-root", default="artifacts/predictions")
    parser.add_argument(
        "--artifact-source",
        action="append",
        type=parse_artifact_source,
        help=(
            "Additional artifact source formatted as model_root=prediction_root. "
            "Can be repeated to build one DB containing models from multiple runs."
        ),
    )
    parser.add_argument("--feature-root", default="data/features")
    parser.add_argument(
        "--comparison-group",
        default="configurable_pipeline_v3",
        help="Only index models from this experiment group. Use an empty value for all.",
    )
    parser.add_argument(
        "--experiment-config",
        default="modeling/configs/configurable_model_matrix.json",
        help=(
            "Only index model IDs expanded from this experiment config. "
            "Use an empty value to index every artifact matching --comparison-group."
        ),
    )
    parser.add_argument(
        "--model-id",
        nargs="+",
        help="Only index these model IDs. When used with --experiment-config, the filters are intersected.",
    )
    parser.add_argument(
        "--model-type",
        nargs="+",
        help="Only index models whose model_type matches one of these values.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["inat", "gowalla", "synthetic"],
    )
    args = parser.parse_args()
    build_database(
        db_path=args.db_path,
        processed_root=args.processed_root,
        model_root=args.model_root,
        prediction_root=args.prediction_root,
        feature_root=args.feature_root,
        comparison_group=args.comparison_group or None,
        datasets=args.datasets,
        experiment_config=args.experiment_config or None,
        model_ids=args.model_id,
        model_types=args.model_type,
        artifact_sources=args.artifact_source,
    )


if __name__ == "__main__":
    main()
