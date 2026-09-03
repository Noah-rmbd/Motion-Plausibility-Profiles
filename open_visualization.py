import argparse
import csv
import hashlib
import json
import mimetypes
import sqlite3
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.plausible_labels import (
    EXTENDED_COLUMNS,
    resolve_plausible_trajectories,
    resolve_plausible_trajectories_from_processed,
)
from modeling.score_aggregation import (
    AGGREGATION_COLUMNS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_TOP_K,
    TRANSITION_SCORE_MODES,
    aggregate_transition_scores,
    select_trajectory_scores,
    transition_error_scores,
)
from modeling.model_diagnostics import build_model_population_stats
from modeling.synthetic_evaluation import evaluate_synthetic_predictions


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "interactive_visualisation"
DB_PATH = ROOT / "artifacts/app/plausibility.db"
PREDICTION_ROOT = ROOT / "artifacts/predictions"
PROCESSED_ROOT = ROOT / "data/processed_parquet"
STATS_ROOT = ROOT / "artifacts/stats/processed_parquet"
PLAUSIBLE_LABELS_PATH = ROOT / "data/labels/plausible_trajectories.csv"
CUSTOM_METRICS_CACHE = {}
TRAJECTORY_BUBBLE_CACHE = {}
TRAJECTORY_SCORE_FILTER_CACHE = {}
CUSTOM_METRICS_CACHE_DIR = "custom_metric_variants"

STATS_METRICS = {
    "speed": "transition_metrics",
    "acceleration": "transition_metrics",
    "distance": "transition_metrics",
    "elapsed_time": "transition_metrics",
    "bearing_change": "transition_metrics",
    "trajectory_n_transitions": "trajectory_n_transitions",
    "avg_elapsed_time": "avg_elapsed_time",
}


def connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def json_value(value):
    if value is None:
        return None
    return json.loads(value)


def finite_json_number(value):
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def query_bool(query, name, default=False):
    raw = query.get(name, [None])[0]
    if raw is None:
        return default
    return str(raw).lower() in {"1", "true", "yes", "on"}


def query_list(query, name):
    values = query.get(name, [])
    if not values:
        return None
    result = []
    for value in values:
        result.extend(
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        )
    return result or None


def model_prediction_root(model_id):
    with connection() as conn:
        row = conn.execute(
            "SELECT prediction_root FROM models WHERE model_id = ?",
            (model_id,),
        ).fetchone()
    if row is None:
        return PREDICTION_ROOT
    return Path(row["prediction_root"])


def synthetic_metrics(model_id):
    prediction_root = model_prediction_root(model_id)
    path = prediction_root / model_id / "synthetic" / "metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("metrics_schema_version") != 3:
        return None
    return metrics


def custom_metrics_cache_payload(
    model_id,
    feature_set,
    aggregation,
    transition_score_mode,
    included_features,
    include_first_transition,
    reference_datasets,
):
    return {
        "schema": 1,
        "model_id": model_id,
        "feature_set": feature_set,
        "aggregation": aggregation,
        "transition_score_mode": transition_score_mode,
        "included_features": list(included_features or []),
        "include_first_transition": bool(include_first_transition),
        "reference_datasets": list(reference_datasets),
    }


def custom_metrics_cache_path(model_id, payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    digest = hashlib.sha256(encoded).hexdigest()[:24]
    return (
        model_prediction_root(model_id)
        / model_id
        / "synthetic"
        / CUSTOM_METRICS_CACHE_DIR
        / f"{digest}.json"
    )


def custom_metrics_source_paths(model_id, feature_set, reference_datasets):
    prediction_root = model_prediction_root(model_id) / model_id
    paths = [
        prediction_root / "synthetic" / "trajectory_scores.parquet",
        prediction_root / "synthetic" / "transition_scores.parquet",
        ROOT / "data/features" / feature_set / "synthetic" / "trajectories.parquet",
        PLAUSIBLE_LABELS_PATH,
    ]
    for dataset in reference_datasets:
        paths.extend(
            [
                prediction_root / dataset / "trajectory_scores.parquet",
                prediction_root / dataset / "transition_scores.parquet",
                ROOT / "data/features" / feature_set / dataset / "trajectories.parquet",
            ]
        )
    review_path = ROOT / "data/labels/trajectory_reviews.csv"
    if review_path.exists():
        paths.append(review_path)
    return [path for path in paths if path.exists()]


def read_custom_metrics_cache(path, source_paths):
    if not path.exists():
        return None
    cache_mtime = path.stat().st_mtime
    if any(source.stat().st_mtime > cache_mtime for source in source_paths):
        return None
    with path.open("r", encoding="utf-8") as handle:
        cached = json.load(handle)
    metrics = cached.get("metrics")
    if not metrics:
        return None
    metrics = dict(metrics)
    metrics["metrics_origin"] = "on_demand_disk_cache"
    return metrics


def write_custom_metrics_cache(path, payload, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    cached_metrics = dict(metrics)
    cached_metrics["metrics_origin"] = "on_demand_disk_cache"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_schema": 1,
                "request": payload,
                "metrics": cached_metrics,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            separators=(",", ":"),
        )


def model_record(row, datasets):
    return {
        "model_id": row["model_id"],
        "model_type": row["model_type"],
        "feature_set": row["feature_set"],
        "features": json_value(row["features_json"]),
        "architecture": json_value(row["architecture_json"]),
        "training": json_value(row["training_json"]),
        "evaluation": json_value(row["evaluation_json"]),
        "experiment": json_value(row["experiment_json"]),
        "prediction_root": row["prediction_root"],
        "final_loss": row["final_loss"],
        "synthetic_metrics": None,
        "datasets": datasets,
    }


def read_scores(model_id, dataset, table, trajectory_ids):
    if not trajectory_ids:
        return pd.DataFrame()
    path = model_prediction_root(model_id) / model_id / dataset / f"{table}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(
        path,
        filters=[("trajectory_id", "in", [int(value) for value in trajectory_ids])],
    )


def highest_scoring_window_transition_ids(group, window_size=DEFAULT_WINDOW_SIZE):
    ordered = group.sort_values("transition_order").copy()
    ordered["window_score"] = (
        ordered["selected_score"].rolling(window_size, min_periods=1).mean()
    )
    end_position = int(ordered["window_score"].to_numpy().argmax())
    start_position = max(0, end_position - window_size + 1)
    return set(
        ordered.iloc[start_position : end_position + 1]["transition_id"]
    )


def canonical_payload(dataset, trajectory_ids, model_id=None):
    if not trajectory_ids:
        return {
            "dataset": dataset,
            "trajectories": [],
            "transitions": [],
            "observations": [],
            "trajectory_scores": [],
            "transition_scores": [],
        }
    placeholders = ",".join("?" for _ in trajectory_ids)
    with connection() as conn:
        trajectories = [
            dict(row)
            for row in conn.execute(
                f"SELECT * FROM trajectories WHERE dataset = ? AND trajectory_id IN ({placeholders})",
                [dataset, *trajectory_ids],
            )
        ]
    existing_ids = [int(row["trajectory_id"]) for row in trajectories]
    if not existing_ids:
        return {
            "dataset": dataset,
            "trajectories": [],
            "transitions": [],
            "observations": [],
            "trajectory_scores": [],
            "transition_scores": [],
        }

    base = PROCESSED_ROOT / dataset
    mapping_frame = pd.read_parquet(
        base / "trajectory_transitions.parquet",
        filters=[("trajectory_id", "in", existing_ids)],
    ).sort_values(["trajectory_id", "transition_order"])
    transition_ids = mapping_frame["transition_id"].tolist()
    if not transition_ids:
        return {
            "dataset": dataset,
            "trajectories": trajectories,
            "transitions": [],
            "observations": [],
            "trajectory_scores": read_scores(
                model_id, dataset, "trajectory_scores", existing_ids
            ).to_dict("records") if model_id else [],
            "transition_scores": [],
        }
    transition_frame = pd.read_parquet(
        base / "transitions.parquet",
        filters=[("transition_id", "in", transition_ids)],
    )
    mappings = mapping_frame.merge(
        transition_frame.drop(columns=["dataset"]),
        on="transition_id",
        how="left",
        validate="one_to_one",
    ).to_dict("records")
    observation_ids = pd.unique(
        pd.concat(
            [transition_frame["observation_id1"], transition_frame["observation_id2"]],
            ignore_index=True,
        )
    ).tolist()
    observations = pd.read_parquet(
        base / "observations.parquet",
        columns=["observation_id", "user_id", "timestamp", "lat", "lon", "is_obscured"],
        filters=[("observation_id", "in", observation_ids)],
    ).to_dict("records")
    return {
        "dataset": dataset,
        "trajectories": trajectories,
        "transitions": mappings,
        "observations": observations,
        "trajectory_scores": read_scores(
            model_id, dataset, "trajectory_scores", existing_ids
        ).to_dict("records") if model_id else [],
        "transition_scores": read_scores(
            model_id, dataset, "transition_scores", existing_ids
        ).to_dict("records") if model_id else [],
    }


def start_date_for_trajectory(row):
    timestamp = row.get("start_timestamp")
    if timestamp:
        return str(timestamp)[:10].replace("-", "/")
    return "*"


def read_label_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {column: row.get(column, "") for column in EXTENDED_COLUMNS}
            for row in reader
        ]


def write_label_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda row: (
            row.get("dataset", ""),
            row.get("user_id", ""),
            row.get("start_date", ""),
            row.get("trajectory_id", ""),
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXTENDED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def mercator_pixels(latitudes, longitudes, zoom):
    latitudes = np.clip(np.asarray(latitudes, dtype=float), -85.05112878, 85.05112878)
    longitudes = np.asarray(longitudes, dtype=float)
    scale = 256 * (2 ** int(zoom))
    lat_rad = np.radians(latitudes)
    x = (longitudes + 180.0) / 360.0 * scale
    y = (
        1.0
        - np.log(np.tan(lat_rad) + (1.0 / np.cos(lat_rad))) / np.pi
    ) / 2.0 * scale
    return x, y


def derive_trajectory_centers(dataset):
    base = PROCESSED_ROOT / dataset
    required = [
        base / "trajectory_transitions.parquet",
        base / "transitions.parquet",
        base / "observations.parquet",
    ]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(
            f"No trajectory barycentre file for dataset '{dataset}', and raw "
            "processed parquet files are not available to derive it."
        )

    mapping = pd.read_parquet(
        base / "trajectory_transitions.parquet",
        columns=["trajectory_id", "transition_id", "transition_order"],
    )
    transitions = pd.read_parquet(
        base / "transitions.parquet",
        columns=["transition_id", "observation_id1", "observation_id2"],
    )
    observations = pd.read_parquet(
        base / "observations.parquet",
        columns=["observation_id", "lat", "lon"],
    ).set_index("observation_id")
    joined = mapping.merge(transitions, on="transition_id", how="left")
    joined = joined.sort_values(["trajectory_id", "transition_order"])

    centers = []
    for trajectory_id, group in joined.groupby("trajectory_id", sort=False):
        observation_ids = [group.iloc[0]["observation_id1"]]
        observation_ids.extend(group["observation_id2"].tolist())
        coords = observations.reindex(observation_ids)[["lat", "lon"]].dropna()
        if coords.empty:
            continue
        centers.append(
            {
                "dataset": dataset,
                "trajectory_id": int(trajectory_id),
                "lat": float(coords["lat"].mean()),
                "lon": float(coords["lon"].mean()),
            }
        )
    return pd.DataFrame(
        centers,
        columns=["dataset", "trajectory_id", "lat", "lon"],
    )


UNPLAUSIBLE_TRAJECTORY_CACHE = {}
FLIGHT_SPEED_TRAJECTORY_CACHE = {}


def load_unplausible_trajectory_ids(dataset):
    if dataset in UNPLAUSIBLE_TRAJECTORY_CACHE:
        return UNPLAUSIBLE_TRAJECTORY_CACHE[dataset]
    base = PROCESSED_ROOT / dataset
    mapping_path = base / "trajectory_transitions.parquet"
    transitions_path = base / "transitions.parquet"
    if not mapping_path.exists() or not transitions_path.exists():
        UNPLAUSIBLE_TRAJECTORY_CACHE[dataset] = set()
        return set()
    mapping = pd.read_parquet(mapping_path, columns=["trajectory_id", "transition_id"])
    transitions = pd.read_parquet(transitions_path, columns=["transition_id", "transition_plausibility"])
    unplausible_trans_ids = set(
        transitions.loc[transitions["transition_plausibility"] == 0, "transition_id"]
    )
    unplausible_ids = set(
        mapping.loc[mapping["transition_id"].isin(unplausible_trans_ids), "trajectory_id"]
    )
    UNPLAUSIBLE_TRAJECTORY_CACHE[dataset] = unplausible_ids
    return unplausible_ids


def load_speed_filtered_trajectory_ids(dataset, min_speed_kmh=200.0, only_plausible=True):
    cache_key = (dataset, float(min_speed_kmh), bool(only_plausible))
    if cache_key in FLIGHT_SPEED_TRAJECTORY_CACHE:
        return FLIGHT_SPEED_TRAJECTORY_CACHE[cache_key]
    base = PROCESSED_ROOT / dataset
    mapping_path = base / "trajectory_transitions.parquet"
    transitions_path = base / "transitions.parquet"
    if not mapping_path.exists() or not transitions_path.exists():
        FLIGHT_SPEED_TRAJECTORY_CACHE[cache_key] = set()
        return set()
    mapping = pd.read_parquet(mapping_path, columns=["trajectory_id", "transition_id"])
    transitions = pd.read_parquet(transitions_path, columns=["transition_id", "speed_kmh"])
    high_speed_trans_ids = set(
        transitions.loc[transitions["speed_kmh"] > float(min_speed_kmh), "transition_id"]
    )
    high_speed_traj_ids = set(
        mapping.loc[mapping["transition_id"].isin(high_speed_trans_ids), "trajectory_id"]
    )
    if only_plausible:
        unplausible_ids = load_unplausible_trajectory_ids(dataset)
        high_speed_traj_ids = high_speed_traj_ids - unplausible_ids
    FLIGHT_SPEED_TRAJECTORY_CACHE[cache_key] = high_speed_traj_ids
    return high_speed_traj_ids


def load_trajectory_bubble_frame(dataset):
    if dataset in TRAJECTORY_BUBBLE_CACHE:
        return TRAJECTORY_BUBBLE_CACHE[dataset]

    center_path = PROCESSED_ROOT / dataset / "trajectory_centers.parquet"
    centers = pd.read_parquet(center_path) if center_path.exists() else derive_trajectory_centers(dataset)
    centers = centers[
        ["dataset", "trajectory_id", "lat", "lon"]
    ].dropna(subset=["lat", "lon"])
    centers = centers[
        np.isfinite(centers["lat"].to_numpy())
        & np.isfinite(centers["lon"].to_numpy())
    ]

    with connection() as conn:
        trajectories = pd.read_sql_query(
            """
            SELECT dataset, trajectory_id, user_id, n_transitions,
                   start_timestamp, end_timestamp
            FROM trajectories
            WHERE dataset = ? AND n_transitions > 2
            """,
            conn,
            params=(dataset,),
        )

    frame = centers.merge(
        trajectories,
        on=["dataset", "trajectory_id"],
        how="inner",
        validate="one_to_one",
    )
    frame["date"] = frame["start_timestamp"].fillna("").str[:10].str.replace("-", "/")
    unplausible_ids = load_unplausible_trajectory_ids(dataset)
    frame["is_unplausible"] = frame["trajectory_id"].isin(unplausible_ids)
    TRAJECTORY_BUBBLE_CACHE[dataset] = frame
    return frame


def score_threshold_for_settings(
    model_id,
    dataset,
    percentage,
    aggregation,
    transition_score_mode,
    included_features,
    include_first_transition,
):
    first_key = "included" if include_first_transition else "excluded"
    prediction_root = model_prediction_root(model_id)
    with connection() as conn:
        model_row = conn.execute(
            """
            SELECT feature_set, evaluation_json
            FROM models WHERE model_id = ?
            """,
            (model_id,),
        ).fetchone()
    if model_row is None:
        raise ValueError(f"Unknown model: {model_id}")

    threshold_reference = dataset
    custom_feature_subset = included_features is not None
    if dataset == "synthetic" and not custom_feature_subset:
        model_metrics = synthetic_metrics(model_id) or {}
        if model_metrics.get("metrics_schema_version") != 3:
            raise ValueError(
                f"{model_id}: synthetic metrics use an obsolete evaluation schema"
            )
        aggregation_metrics = (
            model_metrics.get("scoring", {})
            .get(transition_score_mode, {})
            .get(first_key, {})
            .get(aggregation)
        )
        if aggregation_metrics is None:
            aggregation_metrics = model_metrics.get("aggregations", {}).get(
                aggregation,
                model_metrics,
            )
        metric_point = min(
            aggregation_metrics.get("threshold_metrics", []),
            key=lambda item: abs(item["anomaly_percentage"] - percentage),
            default=None,
        )
        threshold = (
            metric_point["threshold"]
            if metric_point and "threshold" in metric_point
            else float("inf")
        )
        threshold_reference = "unlabeled held-out real score distribution"
    elif custom_feature_subset and dataset == "synthetic":
        threshold = float("inf")
        threshold_reference = "custom feature subsets require real-dataset calibration"
    elif custom_feature_subset:
        calibration_path = (
            ROOT
            / "data/features"
            / model_row["feature_set"]
            / dataset
            / "trajectories.parquet"
        )
        calibration_frame = pd.read_parquet(
            calibration_path,
            columns=["trajectory_id", "use_for_calibration"],
        )
        calibration_ids = set(
            calibration_frame.loc[
                calibration_frame["use_for_calibration"],
                "trajectory_id",
            ].astype(int)
        )
        all_transition_scores = pd.read_parquet(
            prediction_root / model_id / dataset / "transition_scores.parquet"
        )
        calibration_scores = select_trajectory_scores(
            pd.read_parquet(
                prediction_root / model_id / dataset / "trajectory_scores.parquet"
            ),
            all_transition_scores,
            aggregation=aggregation,
            ignore_first_transition=not include_first_transition,
            transition_score_mode=transition_score_mode,
            included_features=included_features,
        )
        calibration_scores = calibration_scores.loc[
            calibration_scores["trajectory_id"].isin(calibration_ids)
        ]
        if calibration_scores.empty:
            raise ValueError("Calibration score population is empty")
        threshold = float(
            calibration_scores["selected_score"].quantile(
                1.0 - percentage / 100.0
            )
        )
        threshold_reference = "held-out real score distribution"
    else:
        with connection() as conn:
            row = conn.execute(
                """
                SELECT threshold_curve_json FROM model_datasets
                WHERE model_id = ? AND dataset = ?
                """,
                (model_id, dataset),
            ).fetchone()
        curves = json_value(row["threshold_curve_json"]) if row else {}
        curve = (
            curves.get(transition_score_mode, {})
            .get(first_key, {})
            .get(aggregation, [])
            if isinstance(curves, dict)
            else curves
        )
        if not curve and isinstance(curves, dict):
            curve = curves.get(aggregation, [])
        threshold = min(
            curve,
            key=lambda item: abs(item["percentage"] - percentage),
            default={"threshold": float("inf")},
        )["threshold"]
    return float(threshold), threshold_reference


def model_score_filter_frame(
    model_id,
    dataset,
    percentage,
    aggregation,
    transition_score_mode,
    included_features,
    include_first_transition,
):
    if aggregation not in AGGREGATION_COLUMNS:
        raise ValueError(f"Unknown trajectory aggregation: {aggregation}")
    if transition_score_mode not in TRANSITION_SCORE_MODES:
        raise ValueError(f"Unknown transition score mode: {transition_score_mode}")
    cache_key = json.dumps(
        {
            "model_id": model_id,
            "dataset": dataset,
            "percentage": percentage,
            "aggregation": aggregation,
            "transition_score_mode": transition_score_mode,
            "included_features": included_features or [],
            "include_first_transition": include_first_transition,
        },
        sort_keys=True,
    )
    if cache_key in TRAJECTORY_SCORE_FILTER_CACHE:
        return TRAJECTORY_SCORE_FILTER_CACHE[cache_key]

    prediction_root = model_prediction_root(model_id)
    model_dir = prediction_root / model_id / dataset
    trajectory_path = model_dir / "trajectory_scores.parquet"
    transition_path = model_dir / "transition_scores.parquet"
    if not trajectory_path.exists() or not transition_path.exists():
        raise FileNotFoundError(
            f"No predictions found for model '{model_id}' on dataset '{dataset}'."
        )
    trajectory_scores = pd.read_parquet(trajectory_path)
    transition_scores = pd.read_parquet(transition_path)
    selected_scores = select_trajectory_scores(
        trajectory_scores,
        transition_scores,
        aggregation=aggregation,
        ignore_first_transition=not include_first_transition,
        transition_score_mode=transition_score_mode,
        included_features=included_features,
    )
    threshold, threshold_reference = score_threshold_for_settings(
        model_id=model_id,
        dataset=dataset,
        percentage=percentage,
        aggregation=aggregation,
        transition_score_mode=transition_score_mode,
        included_features=included_features,
        include_first_transition=include_first_transition,
    )
    selected_scores = selected_scores.copy()
    selected_scores["selected_score"] = selected_scores["selected_score"].astype(float)
    selected_scores["model_unplausible"] = selected_scores["selected_score"] >= threshold
    payload = {
        "scores": selected_scores[
            ["trajectory_id", "selected_score", "model_unplausible"]
        ],
        "threshold": threshold,
        "threshold_reference": threshold_reference,
    }
    TRAJECTORY_SCORE_FILTER_CACHE[cache_key] = payload
    return payload


def parse_parent_bubbles(raw_parent):
    if not raw_parent:
        return []
    parents = []
    for item in raw_parent.split(";"):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError("Invalid trajectory bubble parent descriptor")
        parents.append(
            {
                "zoom": int(parts[0]),
                "cell_size": int(parts[1]),
                "grid_x": int(parts[2]),
                "grid_y": int(parts[3]),
            }
        )
    return parents


def validate_plausible_trajectory(dataset, trajectory_id):
    if dataset == "synthetic":
        return False, "Synthetic trajectories cannot be reviewed as plausible."
    with connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM trajectories
            WHERE dataset = ? AND trajectory_id = ?
            """,
            (dataset, trajectory_id),
        ).fetchone()
    if row is None:
        return False, f"Unknown trajectory: {dataset}/{trajectory_id}"
    row = dict(row)
    if int(row["n_transitions"]) <= 2:
        return False, "Trajectory has 2 or fewer transitions."

    base = PROCESSED_ROOT / dataset
    mapping = pd.read_parquet(
        base / "trajectory_transitions.parquet",
        filters=[("trajectory_id", "=", int(trajectory_id))],
    )
    if mapping.empty:
        return False, "Trajectory has no transition mapping."
    transitions = pd.read_parquet(
        base / "transitions.parquet",
        columns=["transition_id", "transition_plausibility", "plausibility_reason"],
        filters=[("transition_id", "in", mapping["transition_id"].tolist())],
    )
    invalid = transitions.loc[transitions["transition_plausibility"] != 1]
    if not invalid.empty:
        reasons = sorted(
            {
                value or "deterministic plausibility policy"
                for value in invalid["plausibility_reason"].dropna().astype(str)
            }
        )
        return False, "Deterministic-rule violation: " + "; ".join(reasons)
    return True, row


def append_plausible_labels(dataset, user_id, trajectory_ids, notes=""):
    existing_rows = read_label_rows(PLAUSIBLE_LABELS_PATH)
    existing_keys = {
        (row.get("dataset", ""), str(row.get("trajectory_id", "")))
        for row in existing_rows
        if row.get("trajectory_id")
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    added = []
    existing = []
    rejected = []
    for raw_id in trajectory_ids:
        trajectory_id = int(raw_id)
        valid, result = validate_plausible_trajectory(dataset, trajectory_id)
        if not valid:
            rejected.append(
                {
                    "trajectory_id": trajectory_id,
                    "reason": result,
                }
            )
            continue
        row = result
        if str(row["user_id"]) != str(user_id):
            rejected.append(
                {
                    "trajectory_id": trajectory_id,
                    "reason": "Trajectory does not belong to the selected user.",
                }
            )
            continue
        key = (dataset, str(trajectory_id))
        if key in existing_keys:
            existing.append(trajectory_id)
            continue
        label_row = {
            "dataset": dataset,
            "user_id": str(user_id),
            "start_date": start_date_for_trajectory(row),
            "trajectory_id": str(trajectory_id),
            "source": "interactive_visualization",
            "created_at": timestamp,
            "notes": notes,
        }
        existing_rows.append(label_row)
        existing_keys.add(key)
        added.append(label_row)
    if added:
        write_label_rows(PLAUSIBLE_LABELS_PATH, existing_rows)
    try:
        label_path = str(PLAUSIBLE_LABELS_PATH.relative_to(ROOT))
    except ValueError:
        label_path = str(PLAUSIBLE_LABELS_PATH)
    return {
        "label_path": label_path,
        "added": added,
        "existing": existing,
        "rejected": rejected,
    }


def plausible_label_metadata_by_trajectory(dataset, user_id, trajectory_rows):
    rows = read_label_rows(PLAUSIBLE_LABELS_PATH)
    if not rows or dataset == "synthetic":
        return {}
    trajectories = {}
    for row in trajectory_rows:
        trajectory_id = int(row["trajectory_id"])
        trajectories[trajectory_id] = {
            "trajectory_id": trajectory_id,
            "user_id": str(row["user_id"]),
            "start_date": start_date_for_trajectory(row),
            "labels": [],
        }
    for label in rows:
        if label.get("dataset") != dataset:
            continue
        label_user = str(label.get("user_id", ""))
        if label_user not in {"", "*", str(user_id)}:
            continue
        label_trajectory_id = str(label.get("trajectory_id", "")).strip()
        label_start_date = str(label.get("start_date", "")).strip()
        for trajectory_id, trajectory in trajectories.items():
            if label_trajectory_id:
                if str(trajectory_id) != label_trajectory_id:
                    continue
            elif label_start_date and label_start_date != "*":
                if trajectory["start_date"] != label_start_date.replace("-", "/"):
                    continue
            trajectory["labels"].append(label)
    return {
        trajectory_id: {
            "is_plausible_label": True,
            "plausible_label_count": len(item["labels"]),
            "plausible_label_sources": sorted(
                {
                    label.get("source") or "manual"
                    for label in item["labels"]
                }
            ),
            "plausible_label_notes": [
                label.get("notes", "")
                for label in item["labels"]
                if label.get("notes")
            ],
        }
        for trajectory_id, item in trajectories.items()
        if item["labels"]
    }


def get_benchmark_user_data(user_id):
    if user_id == "category:plausible":
        plausible_df = resolve_plausible_trajectories_from_processed()
        all_obs, all_trans, all_trajs = [], [], []
        baseline_reasons = {}
        for dataset, group in plausible_df.groupby("dataset"):
            traj_ids = group["trajectory_id"].tolist()
            payload = canonical_payload(dataset, traj_ids)
            base = PROCESSED_ROOT / dataset
            obs_ids = set()
            for t in payload["transitions"]:
                obs_ids.add(t["observation_id1"])
                obs_ids.add(t["observation_id2"])
            filtered_obs = pd.read_parquet(
                base / "observations.parquet",
                columns=["observation_id", "user_id", "timestamp", "lat", "lon", "is_obscured"],
                filters=[("observation_id", "in", list(obs_ids))] if obs_ids else None,
            )
            
            traj_by_id = {row["trajectory_id"]: row for row in payload["trajectories"]}
            for row in payload["transitions"]:
                trajectory = traj_by_id[row["trajectory_id"]]
                baseline_unplausible = row.get("transition_plausibility") == 0
                if baseline_unplausible:
                    reason = row.get("plausibility_reason") or "Deterministic plausibility policy"
                    baseline_reasons.setdefault((dataset, row["trajectory_id"]), set()).add(reason)
                all_trans.append({
                    **row,
                    "user_id": user_id,
                    "date": (trajectory["start_timestamp"] or "")[:10].replace("-", "/"),
                    "speed": row["speed_kmh"],
                    "elapsed_time": row["elapsed_time_s"],
                    "distance": row["distance_m"],
                    "acceleration": row["acceleration_m_s2"],
                    "bearing_change": row["bearing_change_rad"],
                    "baseline_unplausible": baseline_unplausible,
                    "model_unplausible": False,
                    "is_unplausible": baseline_unplausible,
                    "reviewed_plausible": True,
                    "plausible_label": {"is_plausible_label": True},
                })
            
            mapping = {}
            for row in payload["transitions"]:
                mapping.setdefault(row["trajectory_id"], []).append(row["transition_id"])
                
            for row in payload["trajectories"]:
                unp = (dataset, row["trajectory_id"]) in baseline_reasons
                all_trajs.append({
                    **row,
                    "date": (row["start_timestamp"] or "")[:10].replace("-", "/"),
                    "transitions": mapping.get(row["trajectory_id"], []),
                    "baseline_unplausible": unp,
                    "baseline_reasons": sorted(baseline_reasons.get((dataset, row["trajectory_id"]), set())),
                    "model_unplausible": False,
                    "is_unplausible": unp,
                    "reviewed_plausible": True,
                    "plausible_label": {"is_plausible_label": True},
                })
            
            for row in filtered_obs.to_dict("records"):
                all_obs.append({
                    **row,
                    "date": row["timestamp"],
                    "long": row["lon"],
                })
        return all_obs, all_trans, all_trajs

    elif user_id == "category:unplausible":
        review_path = ROOT / "data/labels/trajectory_reviews.csv"
        if not review_path.exists():
            return [], [], []
        reviews = pd.read_csv(review_path, dtype=str)
        unplausible_reviews = reviews.loc[reviews["label"].str.strip().eq("unplausible")]
        all_obs, all_trans, all_trajs = [], [], []
        baseline_reasons = {}
        for dataset, group in unplausible_reviews.groupby("dataset"):
            traj_ids = group["trajectory_id"].dropna().astype(int).tolist()
            if not traj_ids:
                continue
            payload = canonical_payload(dataset, traj_ids)
            base = PROCESSED_ROOT / dataset
            obs_ids = set()
            for t in payload["transitions"]:
                obs_ids.add(t["observation_id1"])
                obs_ids.add(t["observation_id2"])
            filtered_obs = pd.read_parquet(
                base / "observations.parquet",
                columns=["observation_id", "user_id", "timestamp", "lat", "lon", "is_obscured"],
                filters=[("observation_id", "in", list(obs_ids))] if obs_ids else None,
            )
            
            traj_by_id = {row["trajectory_id"]: row for row in payload["trajectories"]}
            for row in payload["transitions"]:
                trajectory = traj_by_id[row["trajectory_id"]]
                baseline_unplausible = row.get("transition_plausibility") == 0
                if baseline_unplausible:
                    reason = row.get("plausibility_reason") or "Deterministic plausibility policy"
                    baseline_reasons.setdefault((dataset, row["trajectory_id"]), set()).add(reason)
                all_trans.append({
                    **row,
                    "user_id": user_id,
                    "date": (trajectory["start_timestamp"] or "")[:10].replace("-", "/"),
                    "speed": row["speed_kmh"],
                    "elapsed_time": row["elapsed_time_s"],
                    "distance": row["distance_m"],
                    "acceleration": row["acceleration_m_s2"],
                    "bearing_change": row["bearing_change_rad"],
                    "baseline_unplausible": baseline_unplausible,
                    "model_unplausible": False,
                    "is_unplausible": True,
                    "reviewed_plausible": False,
                    "plausible_label": None,
                })
            
            mapping = {}
            for row in payload["transitions"]:
                mapping.setdefault(row["trajectory_id"], []).append(row["transition_id"])
                
            for row in payload["trajectories"]:
                unp = (dataset, row["trajectory_id"]) in baseline_reasons
                all_trajs.append({
                    **row,
                    "date": (row["start_timestamp"] or "")[:10].replace("-", "/"),
                    "transitions": mapping.get(row["trajectory_id"], []),
                    "baseline_unplausible": unp,
                    "baseline_reasons": sorted(baseline_reasons.get((dataset, row["trajectory_id"]), set())),
                    "model_unplausible": False,
                    "is_unplausible": True,
                    "reviewed_plausible": False,
                    "plausible_label": None,
                })
            
            for row in filtered_obs.to_dict("records"):
                all_obs.append({
                    **row,
                    "date": row["timestamp"],
                    "long": row["lon"],
                })
        return all_obs, all_trans, all_trajs

    else:
        profile_name = user_id.split(":")[-1] if ":" in user_id else user_id
        with connection() as conn:
            trajectory_rows = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM trajectories WHERE dataset = 'synthetic' AND (user_id = ? OR user_id LIKE ?) AND n_transitions > 2 ORDER BY trajectory_id",
                    (user_id, f"%:{profile_name}")
                )
            ]
        trajectory_ids = [row["trajectory_id"] for row in trajectory_rows]
        payload = canonical_payload("synthetic", trajectory_ids)
        base = PROCESSED_ROOT / "synthetic"
        obs_ids = set()
        for t in payload["transitions"]:
            obs_ids.add(t["observation_id1"])
            obs_ids.add(t["observation_id2"])
        filtered_obs = pd.read_parquet(
            base / "observations.parquet",
            columns=["observation_id", "user_id", "timestamp", "lat", "lon", "is_obscured"],
            filters=[("observation_id", "in", list(obs_ids))] if obs_ids else None,
        )
        trajectory_by_id = {row["trajectory_id"]: row for row in payload["trajectories"]}
        transitions = []
        baseline_reasons = {}
        for row in payload["transitions"]:
            trajectory = trajectory_by_id[row["trajectory_id"]]
            baseline_unplausible = row.get("transition_plausibility") == 0
            if baseline_unplausible:
                reason = row.get("plausibility_reason") or "Deterministic plausibility policy"
                baseline_reasons.setdefault(row["trajectory_id"], set()).add(reason)
            transitions.append({
                **row,
                "user_id": user_id,
                "date": (trajectory["start_timestamp"] or "")[:10].replace("-", "/"),
                "speed": row["speed_kmh"],
                "elapsed_time": row["elapsed_time_s"],
                "distance": row["distance_m"],
                "acceleration": row["acceleration_m_s2"],
                "bearing_change": row["bearing_change_rad"],
                "baseline_unplausible": baseline_unplausible,
                "model_unplausible": False,
                "is_unplausible": baseline_unplausible,
                "reviewed_plausible": False,
                "plausible_label": None,
            })
        mapping = {}
        for row in payload["transitions"]:
            mapping.setdefault(row["trajectory_id"], []).append(row["transition_id"])
        trajectories = [
            {
                **row,
                "date": (row["start_timestamp"] or "")[:10].replace("-", "/"),
                "transitions": mapping.get(row["trajectory_id"], []),
                "baseline_unplausible": row["trajectory_id"] in baseline_reasons,
                "baseline_reasons": sorted(baseline_reasons.get(row["trajectory_id"], set())),
                "model_unplausible": False,
                "is_unplausible": row["trajectory_id"] in baseline_reasons,
                "reviewed_plausible": False,
                "plausible_label": None,
            }
            for row in trajectory_rows
        ]
        observations = [
            {
                **row,
                "date": row["timestamp"],
                "long": row["lon"],
            }
            for row in filtered_obs.to_dict("records")
        ]
        return observations, transitions, trajectories


class VisualizationHandler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/api/catalog":
                return self.catalog()
            if parsed.path == "/api/models":
                return self.models()
            if parsed.path == "/api/model_metrics":
                return self.model_metrics(query)
            if parsed.path == "/api/users":
                return self.users(query)
            if parsed.path == "/api/user_data":
                return self.user_data(query)
            if parsed.path == "/api/timeline":
                return self.timeline(query)
            if parsed.path == "/api/trajectories":
                return self.trajectories(query)
            if parsed.path == "/api/trajectory_bubbles":
                return self.trajectory_bubbles(query)
            if parsed.path == "/api/trajectory_score_distribution":
                return self.trajectory_score_distribution(query)
            if parsed.path == "/api/stats":
                return self.stats(query)
            return self.static_file(parsed.path)
        except (ValueError, KeyError) as exc:
            return self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            return self.send_json({"error": str(exc)}, 500)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in {
            "/api/scores",
            "/api/model_stats",
            "/api/plausible_trajectories",
        }:
            return self.send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/api/plausible_trajectories":
                return self.plausible_trajectories(request)
            if parsed.path == "/api/model_stats":
                return self.model_stats(request)
            return self.scores(request)
        except (ValueError, KeyError) as exc:
            return self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            return self.send_json({"error": str(exc)}, 500)

    def catalog(self):
        with connection() as conn:
            datasets = [dict(row) for row in conn.execute("SELECT * FROM datasets ORDER BY dataset")]
            model_rows = conn.execute("SELECT * FROM models ORDER BY model_id").fetchall()
            models = []
            for row in model_rows:
                model_datasets = [
                    dict(dataset_row)
                    for dataset_row in conn.execute(
                        "SELECT * FROM model_datasets WHERE model_id = ? ORDER BY dataset",
                        (row["model_id"],),
                    )
                ]
                for item in model_datasets:
                    item.pop("threshold_curve_json", None)
                models.append(model_record(row, model_datasets))
        self.send_json({"datasets": datasets, "models": models})

    def models(self):
        with connection() as conn:
            rows = conn.execute("SELECT * FROM models ORDER BY model_id").fetchall()
            models = []
            for row in rows:
                datasets = [
                    dict(dataset_row)
                    for dataset_row in conn.execute(
                        "SELECT * FROM model_datasets WHERE model_id = ? ORDER BY dataset",
                        (row["model_id"],),
                    )
                ]
                for item in datasets:
                    item.pop("threshold_curve_json", None)
                models.append(model_record(row, datasets))
        self.send_json(models)

    def model_metrics(self, query):
        model_id = query.get("model_id", [""])[0]
        if not model_id:
            raise ValueError("model_id is required")
        with connection() as conn:
            model_row = conn.execute(
                """
                SELECT feature_set, features_json, evaluation_json
                FROM models WHERE model_id = ?
                """,
                (model_id,),
            ).fetchone()
        if model_row is None:
            raise ValueError(f"Unknown model: {model_id}")
        included_features = query_list(query, "transition_score_features")
        aggregation = query.get("aggregation", [""])[0]
        transition_score_mode = query.get("transition_score_mode", [""])[0]
        include_first_transition = query_bool(
            query,
            "include_first_transition",
            default=True,
        )
        if not included_features and not aggregation and not transition_score_mode:
            self.send_json(synthetic_metrics(model_id))
            return
        if not aggregation:
            aggregation = "mean"
        if not transition_score_mode:
            transition_score_mode = "mean"
        model_features = json_value(model_row["features_json"])
        if included_features and set(included_features) == set(model_features):
            included_features = None
        evaluation = json_value(model_row["evaluation_json"]) or {}
        reference_datasets = [
            dataset
            for dataset in evaluation.get("datasets", ["inat", "gowalla"])
            if dataset != "synthetic"
        ] or ["inat", "gowalla"]
        cache_key = json.dumps(
            {
                "model_id": model_id,
                "aggregation": aggregation,
                "transition_score_mode": transition_score_mode,
                "included_features": included_features or [],
                "include_first_transition": include_first_transition,
                "reference_datasets": reference_datasets,
            },
            sort_keys=True,
        )
        if cache_key in CUSTOM_METRICS_CACHE:
            self.send_json(CUSTOM_METRICS_CACHE[cache_key])
            return
        cache_payload = custom_metrics_cache_payload(
            model_id=model_id,
            feature_set=model_row["feature_set"],
            aggregation=aggregation,
            transition_score_mode=transition_score_mode,
            included_features=included_features,
            include_first_transition=include_first_transition,
            reference_datasets=reference_datasets,
        )
        disk_cache_path = custom_metrics_cache_path(model_id, cache_payload)
        source_paths = custom_metrics_source_paths(
            model_id=model_id,
            feature_set=model_row["feature_set"],
            reference_datasets=reference_datasets,
        )
        cached_metrics = read_custom_metrics_cache(disk_cache_path, source_paths)
        if cached_metrics is not None:
            CUSTOM_METRICS_CACHE[cache_key] = cached_metrics
            self.send_json(cached_metrics)
            return
        metrics = evaluate_synthetic_predictions(
            model_id=model_id,
            prediction_root=model_prediction_root(model_id),
            feature_root=ROOT / "data/features",
            processed_root=PROCESSED_ROOT,
            feature_set=model_row["feature_set"],
            reference_datasets=reference_datasets,
            plausible_labels_path=PLAUSIBLE_LABELS_PATH,
            aggregation=aggregation,
            transition_score_mode=transition_score_mode,
            included_features=included_features,
            ignore_first_transition=not include_first_transition,
            _write_output=False,
        )
        metrics["metrics_origin"] = "on_demand"
        write_custom_metrics_cache(disk_cache_path, cache_payload, metrics)
        CUSTOM_METRICS_CACHE[cache_key] = metrics
        self.send_json(metrics)




    def users(self, query):
        dataset = query.get("dataset", [""])[0]
        search = query.get("search", [""])[0].strip()
        limit = min(int(query.get("limit", ["100"])[0]), 500)
        if not dataset:
            raise ValueError("dataset is required")
        if dataset in {"synthetic", "benchmark"}:
            review_unplausible_count = 0
            review_path = ROOT / "data/labels/trajectory_reviews.csv"
            if review_path.exists():
                reviews = pd.read_csv(review_path, dtype=str)
                if "label" in reviews.columns:
                    review_unplausible_count = len(reviews.loc[reviews["label"].str.strip().eq("unplausible")])
            plausible_count = len(resolve_plausible_trajectories_from_processed())
            items = [
                {"user_id": "category:plausible", "username": "Labelled as plausible", "observation_count": 0, "trajectory_count": plausible_count},
                {"user_id": "category:unplausible", "username": "Labelled as unplausible", "observation_count": 0, "trajectory_count": review_unplausible_count},
                {"user_id": "profile:back_and_forth", "username": "back_and_forth", "observation_count": 0, "trajectory_count": 100},
                {"user_id": "profile:abnormal_speed", "username": "abnormal_speed", "observation_count": 0, "trajectory_count": 100},
                {"user_id": "profile:initial_fix_jump", "username": "initial_fix_jump", "observation_count": 0, "trajectory_count": 100},
                {"user_id": "profile:coordinate_jitter", "username": "coordinate_jitter", "observation_count": 0, "trajectory_count": 100},
                {"user_id": "profile:timestamp_jitter", "username": "timestamp_jitter", "observation_count": 0, "trajectory_count": 100},
            ]
            if search:
                items = [item for item in items if search.lower() in item["username"].lower()]
            self.send_json(
                [
                    {
                        **row,
                        "nb_observations": row["observation_count"],
                        "nb_trajectories": row["trajectory_count"],
                    }
                    for row in items
                ]
            )
            return
        sql = """
            SELECT u.dataset, u.user_id, u.observation_count, COUNT(t.trajectory_id) AS trajectory_count, u.first_timestamp, u.last_timestamp
            FROM users u
            JOIN trajectories t ON u.dataset = t.dataset AND u.user_id = t.user_id
            WHERE u.dataset = ? AND t.n_transitions > 2
        """
        params = [dataset]
        if search:
            sql += " AND u.user_id LIKE ?"
            params.append(f"%{search}%")
        sql += " GROUP BY u.dataset, u.user_id HAVING trajectory_count > 0 ORDER BY trajectory_count DESC LIMIT ?"
        params.append(limit)
        with connection() as conn:
            rows = [dict(row) for row in conn.execute(sql, params)]
        self.send_json(
            [
                {
                    **row,
                    "username": row["user_id"],
                    "nb_observations": row["observation_count"],
                    "nb_trajectories": row["trajectory_count"],
                }
                for row in rows
            ]
        )

    def user_data(self, query):
        dataset = query.get("dataset", [""])[0]
        user_id = query.get("user_id", [""])[0]
        if not dataset or not user_id:
            raise ValueError("dataset and user_id are required")
        if dataset in {"synthetic", "benchmark"} or user_id.startswith("category:") or user_id.startswith("profile:"):
            observations, transitions, trajectories = get_benchmark_user_data(user_id)
            self.send_json(
                {
                    "observations": observations,
                    "obscured_observations": [row for row in observations if row.get("is_obscured")],
                    "transitions": transitions,
                    "trajectories": trajectories,
                }
            )
            return
        with connection() as conn:
            trajectory_rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT * FROM trajectories
                    WHERE dataset = ? AND user_id = ? AND n_transitions > 2
                    ORDER BY start_timestamp, trajectory_id
                    """,
                    (dataset, user_id),
                )
            ]
        trajectory_ids = [row["trajectory_id"] for row in trajectory_rows]
        plausible_labels = plausible_label_metadata_by_trajectory(
            dataset,
            user_id,
            trajectory_rows,
        )
        payload = canonical_payload(dataset, trajectory_ids)
        base = PROCESSED_ROOT / dataset
        all_observations = pd.read_parquet(
            base / "observations.parquet",
            columns=["observation_id", "user_id", "timestamp", "lat", "lon", "is_obscured"],
            filters=[("user_id", "=", str(user_id))],
        )
        trajectory_by_id = {row["trajectory_id"]: row for row in trajectory_rows}
        transitions = []
        baseline_reasons = {}
        for row in payload["transitions"]:
            trajectory = trajectory_by_id[row["trajectory_id"]]
            baseline_unplausible = row.get("transition_plausibility") == 0
            if baseline_unplausible:
                reason = row.get("plausibility_reason") or "Deterministic plausibility policy"
                baseline_reasons.setdefault(row["trajectory_id"], set()).add(reason)
            transitions.append(
                {
                    **row,
                    "user_id": user_id,
                    "date": (trajectory["start_timestamp"] or "")[:10].replace("-", "/"),
                    "speed": row["speed_kmh"],
                    "elapsed_time": row["elapsed_time_s"],
                    "distance": row["distance_m"],
                    "acceleration": row["acceleration_m_s2"],
                    "bearing_change": row["bearing_change_rad"],
                    "baseline_unplausible": baseline_unplausible,
                    "model_unplausible": False,
                    "is_unplausible": baseline_unplausible,
                    "reviewed_plausible": row["trajectory_id"] in plausible_labels,
                    "plausible_label": plausible_labels.get(row["trajectory_id"]),
                }
            )
        mapping = {}
        for row in payload["transitions"]:
            mapping.setdefault(row["trajectory_id"], []).append(row["transition_id"])
        trajectories = [
            {
                **row,
                "date": (row["start_timestamp"] or "")[:10].replace("-", "/"),
                "transitions": mapping.get(row["trajectory_id"], []),
                "baseline_unplausible": row["trajectory_id"] in baseline_reasons,
                "baseline_reasons": sorted(
                    baseline_reasons.get(row["trajectory_id"], set())
                ),
                "model_unplausible": False,
                "is_unplausible": row["trajectory_id"] in baseline_reasons,
                "reviewed_plausible": row["trajectory_id"] in plausible_labels,
                "plausible_label": plausible_labels.get(row["trajectory_id"]),
            }
            for row in trajectory_rows
        ]
        observations = [
            {
                **row,
                "date": row["timestamp"],
                "long": row["lon"],
            }
            for row in all_observations.to_dict("records")
        ]
        self.send_json(
            {
                "observations": observations,
                "obscured_observations": [
                    row for row in observations if row["is_obscured"]
                ],
                "transitions": transitions,
                "trajectories": trajectories,
            }
        )


    def timeline(self, query):
        dataset = query.get("dataset", [""])[0]
        user_id = query.get("user_id", [""])[0]
        model_id = query.get("model_id", [""])[0]
        if not dataset or not user_id:
            raise ValueError("dataset and user_id are required")
        with connection() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT trajectory_id, n_transitions, start_timestamp, end_timestamp
                    FROM trajectories
                    WHERE dataset = ? AND user_id = ? AND n_transitions > 2
                    ORDER BY start_timestamp, trajectory_id
                    """,
                    (dataset, user_id),
                )
            ]
        if model_id and rows:
            score_frame = read_scores(
                model_id,
                dataset,
                "trajectory_scores",
                [row["trajectory_id"] for row in rows],
            )
            scores = score_frame.set_index("trajectory_id").to_dict("index")
            for row in rows:
                score = scores.get(row["trajectory_id"])
                if score:
                    row.update({key: value for key, value in score.items() if key != "dataset"})
        self.send_json(rows)

    def trajectories(self, query):
        dataset = query.get("dataset", [""])[0]
        raw_ids = query.get("trajectory_ids", [""])[0]
        model_id = query.get("model_id", [""])[0]
        trajectory_ids = [int(value) for value in raw_ids.split(",") if value]
        if not dataset or not trajectory_ids:
            raise ValueError("dataset and trajectory_ids are required")
        self.send_json(canonical_payload(dataset, trajectory_ids, model_id))

    def trajectory_bubbles(self, query):
        dataset = query.get("dataset", [""])[0]
        if not dataset:
            raise ValueError("dataset is required")

        zoom = max(0, min(24, int(float(query.get("zoom", ["2"])[0]))))
        cell_size = max(32, min(240, int(float(query.get("cell_size", ["90"])[0]))))
        parent_descriptors = parse_parent_bubbles(query.get("parents", [""])[0])

        try:
            frame = load_trajectory_bubble_frame(dataset)
        except FileNotFoundError as exc:
            self.send_json({"error": str(exc)}, 404)
            return
        filtered = frame
        score_payload = None

        user_id = query.get("user_id", [""])[0]
        model_id = query.get("model_id", [""])[0]
        exclude_raw = query.get("exclude_trajectory_ids", [""])[0]
        if exclude_raw:
            exclude_ids = {int(val) for val in exclude_raw.split(",") if val.strip().isdigit()}
            if exclude_ids:
                filtered = filtered[~filtered["trajectory_id"].isin(exclude_ids)]
        if user_id:
            filtered = filtered[filtered["user_id"].astype(str) == str(user_id)]
        min_transition_speed = query.get("min_transition_speed", [""])[0]
        if min_transition_speed:
            try:
                min_speed_val = float(min_transition_speed)
                exclude_unplausible = query_bool(query, "exclude_unplausible", default=True)
                speed_filtered_ids = load_speed_filtered_trajectory_ids(
                    dataset, min_speed_val, only_plausible=exclude_unplausible
                )
                filtered = filtered[filtered["trajectory_id"].isin(speed_filtered_ids)]
                if exclude_unplausible and "is_unplausible" in filtered.columns:
                    filtered = filtered[~filtered["is_unplausible"]]
            except (ValueError, TypeError):
                pass
        if model_id:
            percentage = float(query.get("anomaly_percentage", ["5.0"])[0])
            aggregation = query.get("aggregation", ["mean"])[0]
            transition_score_mode = query.get("transition_score_mode", ["mean"])[0]
            included_features = query_list(query, "transition_score_features")
            include_first_transition = query_bool(
                query,
                "include_first_transition",
                default=True,
            )
            try:
                score_payload = model_score_filter_frame(
                    model_id=model_id,
                    dataset=dataset,
                    percentage=percentage,
                    aggregation=aggregation,
                    transition_score_mode=transition_score_mode,
                    included_features=included_features,
                    include_first_transition=include_first_transition,
                )
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, 404)
                return
            score_frame = score_payload["scores"]
            filtered = filtered.merge(
                score_frame,
                on="trajectory_id",
                how="inner",
                validate="one_to_one",
            )
            raw_score_min = query.get("score_min", [""])[0]
            raw_score_max = query.get("score_max", [""])[0]
            if raw_score_min != "":
                filtered = filtered[
                    filtered["selected_score"] >= float(raw_score_min)
                ]
            if raw_score_max != "":
                filtered = filtered[
                    filtered["selected_score"] <= float(raw_score_max)
                ]

        for parent in parent_descriptors:
            x, y = mercator_pixels(
                filtered["lat"].to_numpy(),
                filtered["lon"].to_numpy(),
                parent["zoom"],
            )
            filtered = filtered[
                (np.floor(x / parent["cell_size"]).astype(np.int64) == parent["grid_x"])
                & (np.floor(y / parent["cell_size"]).astype(np.int64) == parent["grid_y"])
            ]

        has_bounds = all(name in query for name in ("south", "north", "west", "east"))
        if has_bounds:
            south = float(query["south"][0])
            north = float(query["north"][0])
            west = float(query["west"][0])
            east = float(query["east"][0])
            filtered = filtered[
                (filtered["lat"] >= south)
                & (filtered["lat"] <= north)
            ]
            if west <= east:
                filtered = filtered[
                    (filtered["lon"] >= west)
                    & (filtered["lon"] <= east)
                ]
            else:
                filtered = filtered[
                    (filtered["lon"] >= west)
                    | (filtered["lon"] <= east)
                ]

        if filtered.empty:
            self.send_json(
                {
                    "dataset": dataset,
                    "total_points": int(len(frame)),
                    "visible_points": 0,
                    "cluster_zoom": zoom,
                    "cell_size": cell_size,
                    "threshold": finite_json_number(score_payload["threshold"]) if score_payload else None,
                    "threshold_reference": (
                        score_payload["threshold_reference"]
                        if score_payload else None
                    ),
                    "clusters": [],
                }
            )
            return

        clustered = filtered.copy()
        x, y = mercator_pixels(
            clustered["lat"].to_numpy(),
            clustered["lon"].to_numpy(),
            zoom,
        )
        clustered["_grid_x"] = np.floor(x / cell_size).astype(np.int64)
        clustered["_grid_y"] = np.floor(y / cell_size).astype(np.int64)

        clusters = []
        for (grid_x, grid_y), group in clustered.groupby(["_grid_x", "_grid_y"], sort=False):
            count = int(len(group))
            item = {
                "count": count,
                "lat": float(group["lat"].mean()),
                "lon": float(group["lon"].mean()),
                "bounds": {
                    "south": float(group["lat"].min()),
                    "north": float(group["lat"].max()),
                    "west": float(group["lon"].min()),
                    "east": float(group["lon"].max()),
                },
                "parent": {
                    "zoom": zoom,
                    "cell_size": cell_size,
                    "grid_x": int(grid_x),
                    "grid_y": int(grid_y),
                },
            }
            if "selected_score" in group:
                item["score_summary"] = {
                    "min": float(group["selected_score"].min()),
                    "max": float(group["selected_score"].max()),
                    "mean": float(group["selected_score"].mean()),
                }
            if count <= 10:
                item["trajectory_ids"] = [
                    int(value) for value in group["trajectory_id"].tolist()
                ]
            if count == 1:
                row = group.iloc[0]
                is_unplausible = bool(row.get("is_unplausible", False))
                item.update(
                    {
                        "trajectory_id": int(row["trajectory_id"]),
                        "user_id": str(row["user_id"]),
                        "n_transitions": int(row["n_transitions"]),
                        "start_timestamp": row.get("start_timestamp"),
                        "end_timestamp": row.get("end_timestamp"),
                        "date": row.get("date"),
                        "is_unplausible": is_unplausible,
                        "baseline_unplausible": is_unplausible,
                    }
                )
                if "selected_score" in row:
                    item["model_score"] = float(row["selected_score"])
                if "model_unplausible" in row:
                    item["model_unplausible"] = bool(row["model_unplausible"])
            clusters.append(item)

        clusters.sort(key=lambda item: item["count"], reverse=True)
        self.send_json(
            {
                "dataset": dataset,
                "total_points": int(len(frame)),
                "visible_points": int(len(filtered)),
                "cluster_zoom": zoom,
                "cell_size": cell_size,
                "threshold": finite_json_number(score_payload["threshold"]) if score_payload else None,
                "threshold_reference": (
                    score_payload["threshold_reference"]
                    if score_payload else None
                ),
                "clusters": clusters,
            }
        )

    def trajectory_score_distribution(self, query):
        dataset = query.get("dataset", [""])[0]
        model_id = query.get("model_id", [""])[0]
        if not dataset or not model_id:
            raise ValueError("dataset and model_id are required")
        percentage = float(query.get("anomaly_percentage", ["5.0"])[0])
        aggregation = query.get("aggregation", ["mean"])[0]
        transition_score_mode = query.get("transition_score_mode", ["mean"])[0]
        included_features = query_list(query, "transition_score_features")
        include_first_transition = query_bool(
            query,
            "include_first_transition",
            default=True,
        )
        distribution_datasets = (
            ["inat", "gowalla"]
            if dataset in {"combined_real", "real_combined", "inat+gowalla"}
            else [dataset]
        )
        score_payloads = [
            model_score_filter_frame(
                model_id=model_id,
                dataset=score_dataset,
                percentage=percentage,
                aggregation=aggregation,
                transition_score_mode=transition_score_mode,
                included_features=included_features,
                include_first_transition=include_first_transition,
            )
            for score_dataset in distribution_datasets
        ]
        filtered_scores = []
        for score_dataset, payload in zip(distribution_datasets, score_payloads):
            bubble_frame = load_trajectory_bubble_frame(score_dataset)
            valid_ids = set(bubble_frame["trajectory_id"])
            frame_scores = payload["scores"]
            valid_scores = frame_scores.loc[
                frame_scores["trajectory_id"].isin(valid_ids), "selected_score"
            ]
            filtered_scores.append(valid_scores.dropna().astype(float))
        scores = pd.concat(filtered_scores, ignore_index=True)
        total_points = sum(
            int(len(load_trajectory_bubble_frame(score_dataset)))
            for score_dataset in distribution_datasets
        )
        threshold_reference = (
            "iNaturalist + Gowalla score distribution"
            if len(distribution_datasets) > 1
            else score_payloads[0]["threshold_reference"]
        )
        threshold = (
            float(scores.quantile(1.0 - percentage / 100.0))
            if len(distribution_datasets) > 1 and not scores.empty
            else score_payloads[0]["threshold"]
        )
        if scores.empty:
            self.send_json(
                {
                    "dataset": dataset,
                    "distribution_datasets": distribution_datasets,
                    "model_id": model_id,
                    "count": 0,
                    "total_points": total_points,
                    "min": None,
                    "max": None,
                    "threshold": finite_json_number(threshold),
                    "threshold_reference": threshold_reference,
                    "histogram": [],
                }
            )
            return
        counts, edges = np.histogram(scores.to_numpy(), bins=min(60, max(10, int(np.sqrt(len(scores))))))
        histogram = [
            {
                "bin_left": float(edges[index]),
                "bin_right": float(edges[index + 1]),
                "count": int(count),
            }
            for index, count in enumerate(counts)
        ]
        self.send_json(
            {
                "dataset": dataset,
                "distribution_datasets": distribution_datasets,
                "model_id": model_id,
                "count": int(len(scores)),
                "total_points": total_points,
                "min": float(scores.min()),
                "max": float(scores.max()),
                "threshold": finite_json_number(threshold),
                "threshold_reference": threshold_reference,
                "histogram": histogram,
            }
        )

    def plausible_trajectories(self, request):
        dataset = request.get("dataset", "")
        user_id = request.get("user_id", "")
        trajectory_ids = request.get("trajectory_ids", [])
        if not dataset or not user_id or not trajectory_ids:
            raise ValueError("dataset, user_id, and trajectory_ids are required")
        payload = append_plausible_labels(
            dataset=dataset,
            user_id=user_id,
            trajectory_ids=trajectory_ids,
            notes=request.get("notes", ""),
        )
        self.send_json(payload)

    def stats(self, query):
        dataset = query.get("dataset", ["inat"])[0]
        subset = query.get("subset", ["clean_real"])[0]
        metric = query.get("metric", ["speed"])[0]
        if metric not in STATS_METRICS:
            raise ValueError(f"Unknown statistics metric: {metric}")
        summary_path = STATS_ROOT / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"{summary_path} does not exist. Run dataset_preprocessing/parquet_dataset_stats.py."
            )
        with summary_path.open("r", encoding="utf-8") as handle:
            complete_summary = json.load(handle)
        try:
            subset_summary = complete_summary["datasets"][dataset][subset]
        except KeyError as exc:
            raise ValueError(f"Unknown statistics dataset/subset: {dataset}/{subset}") from exc
        metric_group = STATS_METRICS[metric]
        metric_summary = (
            subset_summary[metric_group][metric]
            if metric_group == "transition_metrics"
            else subset_summary[metric_group]
        )
        histogram_path = STATS_ROOT / "histograms" / f"{dataset}_{subset}_{metric}.csv"
        histogram = pd.read_csv(histogram_path).to_dict("records")
        self.send_json(
            {
                "dataset": dataset,
                "subset": subset,
                "metric": metric,
                "n_trajectories": subset_summary["n_trajectories"],
                "n_transitions": subset_summary["n_transitions"],
                "summary": metric_summary,
                "data_quality": subset_summary.get("data_quality", {}),
                "histogram": histogram,
            }
        )

    def model_stats(self, request):
        model_id = request["model_id"]
        with connection() as conn:
            row = conn.execute(
                "SELECT feature_set FROM models WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown model: {model_id}")
        prediction_root = model_prediction_root(model_id)
        payload = build_model_population_stats(
            model_id=model_id,
            dataset=request["dataset"],
            population=request.get("population", "model_flagged"),
            metric=request.get("metric", "model_score"),
            anomaly_percentage=float(request.get("anomaly_percentage", 5.0)),
            aggregation=request.get("aggregation", "mean"),
            transition_score_mode=request.get(
                "transition_score_mode",
                "mean",
            ),
            included_features=request.get("transition_score_features"),
            include_first_transition=bool(
                request.get("include_first_transition", True)
            ),
            prediction_root=prediction_root,
            feature_root=ROOT / "data/features",
            processed_root=PROCESSED_ROOT,
            feature_set=row["feature_set"],
        )
        self.send_json(payload)

    def scores(self, request):
        dataset = request["dataset"]
        model_id = request["model_id"]
        trajectory_ids = [int(value) for value in request.get("trajectory_ids", [])]
        percentage = float(request.get("anomaly_percentage", 5.0))
        aggregation = request.get("aggregation", "mean")
        transition_score_mode = request.get("transition_score_mode", "mean")
        included_features = request.get("transition_score_features")
        include_first_transition = bool(
            request.get("include_first_transition", True)
        )
        if aggregation not in AGGREGATION_COLUMNS:
            raise ValueError(f"Unknown trajectory aggregation: {aggregation}")
        if transition_score_mode not in TRANSITION_SCORE_MODES:
            raise ValueError(
                f"Unknown transition score mode: {transition_score_mode}"
            )
        first_key = "included" if include_first_transition else "excluded"
        with connection() as conn:
            model_row = conn.execute(
                """
                SELECT evaluation_json
                FROM models WHERE model_id = ?
                """,
                (model_id,),
            ).fetchone()
        evaluation = json_value(model_row["evaluation_json"]) if model_row else {}
        prediction_root = model_prediction_root(model_id)
        trajectory_scores = read_scores(
            model_id, dataset, "trajectory_scores", trajectory_ids
        )
        transition_scores = read_scores(
            model_id, dataset, "transition_scores", trajectory_ids
        )
        selected_trajectory_scores = select_trajectory_scores(
            trajectory_scores,
            transition_scores,
            aggregation=aggregation,
            ignore_first_transition=not include_first_transition,
            transition_score_mode=transition_score_mode,
            included_features=included_features,
        )
        transition_scores = transition_scores.copy()
        transition_scores["selected_score"] = transition_error_scores(
            transition_scores,
            mode=transition_score_mode,
            included_features=included_features,
        )
        trajectory_scores = trajectory_scores.merge(
            selected_trajectory_scores,
            on="trajectory_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_selected"),
        )
        threshold_reference = dataset
        custom_feature_subset = included_features is not None
        if dataset == "synthetic" and model_row and not custom_feature_subset:
            model_metrics = synthetic_metrics(model_id) or {}
            if model_metrics.get("metrics_schema_version") != 3:
                raise ValueError(
                    f"{model_id}: synthetic metrics use an obsolete evaluation schema"
                )
            aggregation_metrics = (
                model_metrics.get("scoring", {})
                .get(transition_score_mode, {})
                .get(first_key, {})
                .get(aggregation)
            )
            if aggregation_metrics is None:
                aggregation_metrics = model_metrics.get("aggregations", {}).get(
                    aggregation,
                    model_metrics,
                )
            metric_point = min(
                aggregation_metrics.get("threshold_metrics", []),
                key=lambda item: abs(item["anomaly_percentage"] - percentage),
                default=None,
            )
            if metric_point and "threshold" in metric_point:
                threshold = metric_point["threshold"]
                threshold_reference = "unlabeled held-out real score distribution"
            else:
                threshold = float("inf")
        elif custom_feature_subset and dataset == "synthetic":
            threshold = float("inf")
            threshold_reference = (
                "custom feature subsets require real-dataset calibration"
            )
        elif custom_feature_subset:
            with connection() as conn:
                feature_set_row = conn.execute(
                    "SELECT feature_set FROM models WHERE model_id = ?",
                    (model_id,),
                ).fetchone()
            if feature_set_row is None:
                raise ValueError(f"Unknown model: {model_id}")
            calibration_path = (
                ROOT
                / "data/features"
                / feature_set_row["feature_set"]
                / dataset
                / "trajectories.parquet"
            )
            calibration_frame = pd.read_parquet(
                calibration_path,
                columns=["trajectory_id", "use_for_calibration"],
            )
            calibration_ids = set(
                calibration_frame.loc[
                    calibration_frame["use_for_calibration"],
                    "trajectory_id",
                ].astype(int)
            )
            all_transition_scores = pd.read_parquet(
                prediction_root
                / model_id
                / dataset
                / "transition_scores.parquet"
            )
            calibration_scores = select_trajectory_scores(
                pd.read_parquet(
                    prediction_root
                    / model_id
                    / dataset
                    / "trajectory_scores.parquet"
                ),
                all_transition_scores,
                aggregation=aggregation,
                ignore_first_transition=not include_first_transition,
                transition_score_mode=transition_score_mode,
                included_features=included_features,
            )
            calibration_scores = calibration_scores.loc[
                calibration_scores["trajectory_id"].isin(calibration_ids)
            ]
            if calibration_scores.empty:
                raise ValueError("Calibration score population is empty")
            threshold = float(
                calibration_scores["selected_score"].quantile(
                    1.0 - percentage / 100.0
                )
            )
            threshold_reference = "held-out real score distribution"
        else:
            with connection() as conn:
                row = conn.execute(
                    """
                    SELECT threshold_curve_json FROM model_datasets
                    WHERE model_id = ? AND dataset = ?
                    """,
                    (model_id, dataset),
                ).fetchone()
            curves = json_value(row["threshold_curve_json"]) if row else {}
            curve = (
                curves.get(transition_score_mode, {})
                .get(first_key, {})
                .get(aggregation, [])
                if isinstance(curves, dict)
                else curves
            )
            if not curve and isinstance(curves, dict):
                curve = curves.get(aggregation, [])
            threshold = min(
                curve,
                key=lambda item: abs(item["percentage"] - percentage),
                default={"threshold": float("inf")},
            )["threshold"]
        trajectory_result = {}
        anomalous_ids = set()
        for score in trajectory_scores.to_dict("records"):
            selected_score = float(score["selected_score"])
            anomalous = selected_score >= threshold
            if anomalous:
                anomalous_ids.add(score["trajectory_id"])
            trajectory_result[str(score["trajectory_id"])] = {
                **score,
                "model_score": selected_score,
                "score_type": score.get(
                    "score_type",
                    "reconstruction_error",
                ),
                "reconstruction_error": float(
                    score.get("reconstruction_score", selected_score)
                ),
                "aggregation": aggregation,
                "is_unplausible": anomalous,
            }
        ranked_steps = {}
        for trajectory_id, group in transition_scores.groupby("trajectory_id", sort=False):
            eligible = (
                group
                if include_first_transition
                else group.loc[group["transition_order"] > 0]
            )
            if eligible.empty:
                eligible = group
            if aggregation == "window_max":
                ranked_steps[trajectory_id] = highest_scoring_window_transition_ids(
                    eligible
                )
            else:
                contribution_count = DEFAULT_TOP_K if aggregation == "top_k" else 1
                ranked_steps[trajectory_id] = set(
                    eligible.nlargest(contribution_count, "selected_score")[
                        "transition_id"
                    ]
                )
        transition_result = {}
        selected_feature_keys = (
            {f"error_{feature}" for feature in included_features}
            if included_features is not None
            else None
        )
        for score in transition_scores.to_dict("records"):
            feature_keys = [
                key
                for key in score
                if key.startswith("error_")
                and (selected_feature_keys is None or key in selected_feature_keys)
            ]
            is_least_plausible = (
                score["trajectory_id"] in anomalous_ids
                and score["transition_id"] in ranked_steps.get(score["trajectory_id"], set())
            )
            transition_result[str(score["transition_id"])] = {
                **score,
                "mse": score["selected_score"],
                "features": [score[key] for key in feature_keys],
                "feature_names": [key.removeprefix("error_") for key in feature_keys],
                "is_unplausible": is_least_plausible,
            }
        self.send_json(
            {
                "threshold": threshold,
                "threshold_reference": threshold_reference,
                "transition_score_mode": transition_score_mode,
                "transition_score_features": included_features,
                "include_first_transition": include_first_transition,
                "trajectories": trajectory_result,
                "transitions": transition_result,
            }
        )

    def static_file(self, request_path):
        relative = "main.html" if request_path in {"", "/"} else request_path.lstrip("/")
        path = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT not in path.parents and path != STATIC_ROOT:
            return self.send_json({"error": "not found"}, 404)
        if not path.exists() or not path.is_file():
            return self.send_json({"error": "not found"}, 404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def main():
    global DB_PATH, PREDICTION_ROOT, PROCESSED_ROOT, STATS_ROOT
    parser = argparse.ArgumentParser(description="Serve the trajectory plausibility visualization.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument("--prediction-root", default=str(PREDICTION_ROOT))
    parser.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    parser.add_argument("--stats-root", default=str(STATS_ROOT))
    args = parser.parse_args()
    DB_PATH = Path(args.db_path)
    PREDICTION_ROOT = Path(args.prediction_root)
    PROCESSED_ROOT = Path(args.processed_root)
    STATS_ROOT = Path(args.stats_root)
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} does not exist. Run build_database.py first.")
    server = ThreadingHTTPServer((args.host, args.port), VisualizationHandler)
    print(f"Visualization: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
