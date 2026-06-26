import argparse
import csv
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

STATS_METRICS = {
    "speed": "transition_metrics",
    "acceleration": "transition_metrics",
    "distance": "transition_metrics",
    "elapsed_time": "transition_metrics",
    "bearing_change": "transition_metrics",
    "trajectory_n_transitions": "trajectory_n_transitions",
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
    synthetic_features = (
        ROOT / "data/features/motion_v2/synthetic/trajectories.parquet"
    )
    if (
        synthetic_features.exists()
        and synthetic_features.stat().st_mtime > path.stat().st_mtime
    ):
        return None
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("metrics_schema_version") != 3:
        return None
    return metrics


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

    base = PROCESSED_ROOT / dataset
    mapping_frame = pd.read_parquet(
        base / "trajectory_transitions.parquet",
        filters=[("trajectory_id", "in", trajectory_ids)],
    ).sort_values(["trajectory_id", "transition_order"])
    transition_ids = mapping_frame["transition_id"].tolist()
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
        "trajectories": trajectories,
        "transitions": mappings,
        "observations": observations,
        "trajectory_scores": read_scores(
            model_id, dataset, "trajectory_scores", trajectory_ids
        ).to_dict("records") if model_id else [],
        "transition_scores": read_scores(
            model_id, dataset, "transition_scores", trajectory_ids
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
            WHERE dataset = ? AND n_transitions > 0
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
    if int(row["n_transitions"]) <= 0:
        return False, "Trajectory has no transition."

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
        CUSTOM_METRICS_CACHE[cache_key] = metrics
        self.send_json(metrics)

    def users(self, query):
        dataset = query.get("dataset", [""])[0]
        search = query.get("search", [""])[0].strip()
        limit = min(int(query.get("limit", ["100"])[0]), 500)
        if not dataset:
            raise ValueError("dataset is required")
        sql = "SELECT * FROM users WHERE dataset = ? AND trajectory_count > 0"
        params = [dataset]
        if search:
            sql += " AND user_id LIKE ?"
            params.append(f"%{search}%")
        sql += " ORDER BY trajectory_count DESC LIMIT ?"
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
        with connection() as conn:
            trajectory_rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT * FROM trajectories
                    WHERE dataset = ? AND user_id = ? AND n_transitions > 0
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
                    WHERE dataset = ? AND user_id = ? AND n_transitions > 0
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
        if user_id:
            filtered = filtered[filtered["user_id"].astype(str) == str(user_id)]
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
            if count <= 50:
                item["trajectory_ids"] = [
                    int(value) for value in group["trajectory_id"].tolist()
                ]
            if count == 1:
                row = group.iloc[0]
                item.update(
                    {
                        "trajectory_id": int(row["trajectory_id"]),
                        "user_id": str(row["user_id"]),
                        "n_transitions": int(row["n_transitions"]),
                        "start_timestamp": row.get("start_timestamp"),
                        "end_timestamp": row.get("end_timestamp"),
                        "date": row.get("date"),
                    }
                )
                if "selected_score" in row:
                    item["model_score"] = float(row["selected_score"])
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
        score_payload = model_score_filter_frame(
            model_id=model_id,
            dataset=dataset,
            percentage=percentage,
            aggregation=aggregation,
            transition_score_mode=transition_score_mode,
            included_features=included_features,
            include_first_transition=include_first_transition,
        )
        scores = score_payload["scores"]["selected_score"].dropna().astype(float)
        total_points = int(len(load_trajectory_bubble_frame(dataset)))
        if scores.empty:
            self.send_json(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "count": 0,
                    "total_points": total_points,
                    "min": None,
                    "max": None,
                    "threshold": finite_json_number(score_payload["threshold"]),
                    "threshold_reference": score_payload["threshold_reference"],
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
                "model_id": model_id,
                "count": int(len(scores)),
                "total_points": total_points,
                "min": float(scores.min()),
                "max": float(scores.max()),
                "threshold": finite_json_number(score_payload["threshold"]),
                "threshold_reference": score_payload["threshold_reference"],
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
                f"{summary_path} does not exist. Run Dataset_Preprocessing/parquet_dataset_stats.py."
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
                and score["transition_id"] in ranked_steps[score["trajectory_id"]]
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
