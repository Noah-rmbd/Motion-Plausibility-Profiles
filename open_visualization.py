import argparse
import json
import mimetypes
import sqlite3
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "interactive_visualisation"
DB_PATH = ROOT / "artifacts/app/plausibility.db"
PREDICTION_ROOT = ROOT / "artifacts/predictions"
PROCESSED_ROOT = ROOT / "data/processed_parquet"


def connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def json_value(value):
    if value is None:
        return None
    return json.loads(value)


def model_record(row, datasets):
    return {
        "model_id": row["model_id"],
        "model_type": row["model_type"],
        "feature_set": row["feature_set"],
        "features": json_value(row["features_json"]),
        "architecture": json_value(row["architecture_json"]),
        "training": json_value(row["training_json"]),
        "evaluation": json_value(row["evaluation_json"]),
        "final_loss": row["final_loss"],
        "synthetic_metrics": json_value(row["synthetic_metrics_json"]),
        "datasets": datasets,
    }


def read_scores(model_id, dataset, table, trajectory_ids):
    if not trajectory_ids:
        return pd.DataFrame()
    path = PREDICTION_ROOT / model_id / dataset / f"{table}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(
        path,
        filters=[("trajectory_id", "in", [int(value) for value in trajectory_ids])],
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


class VisualizationHandler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
            if parsed.path == "/api/users":
                return self.users(query)
            if parsed.path == "/api/user_data":
                return self.user_data(query)
            if parsed.path == "/api/timeline":
                return self.timeline(query)
            if parsed.path == "/api/trajectories":
                return self.trajectories(query)
            return self.static_file(parsed.path)
        except (ValueError, KeyError) as exc:
            return self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            return self.send_json({"error": str(exc)}, 500)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/scores":
            return self.send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
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
                    {
                        **dict(dataset_row),
                        "threshold_curve": json_value(dataset_row["threshold_curve_json"]),
                    }
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
                    {
                        **dict(dataset_row),
                        "threshold_curve": json_value(dataset_row["threshold_curve_json"]),
                    }
                    for dataset_row in conn.execute(
                        "SELECT * FROM model_datasets WHERE model_id = ? ORDER BY dataset",
                        (row["model_id"],),
                    )
                ]
                for item in datasets:
                    item.pop("threshold_curve_json", None)
                models.append(model_record(row, datasets))
        self.send_json(models)

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
        sql += " ORDER BY observation_count DESC LIMIT ?"
        params.append(limit)
        with connection() as conn:
            rows = [dict(row) for row in conn.execute(sql, params)]
        self.send_json(
            [
                {
                    **row,
                    "username": row["user_id"],
                    "nb_observations": row["observation_count"],
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
                    WHERE dataset = ? AND user_id = ?
                    ORDER BY start_timestamp, trajectory_id
                    """,
                    (dataset, user_id),
                )
            ]
        trajectory_ids = [row["trajectory_id"] for row in trajectory_rows]
        payload = canonical_payload(dataset, trajectory_ids)
        base = PROCESSED_ROOT / dataset
        all_observations = pd.read_parquet(
            base / "observations.parquet",
            columns=["observation_id", "user_id", "timestamp", "lat", "lon", "is_obscured"],
            filters=[("user_id", "=", str(user_id))],
        )
        trajectory_by_id = {row["trajectory_id"]: row for row in trajectory_rows}
        transitions = []
        for row in payload["transitions"]:
            trajectory = trajectory_by_id[row["trajectory_id"]]
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
                "is_unplausible": False,
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
                    WHERE dataset = ? AND user_id = ?
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

    def scores(self, request):
        dataset = request["dataset"]
        model_id = request["model_id"]
        trajectory_ids = [int(value) for value in request.get("trajectory_ids", [])]
        percentage = float(request.get("anomaly_percentage", 5.0))
        trajectory_scores = read_scores(
            model_id, dataset, "trajectory_scores", trajectory_ids
        )
        transition_scores = read_scores(
            model_id, dataset, "transition_scores", trajectory_ids
        )
        with connection() as conn:
            row = conn.execute(
                """
                SELECT threshold_curve_json FROM model_datasets
                WHERE model_id = ? AND dataset = ?
                """,
                (model_id, dataset),
            ).fetchone()
        curve = json_value(row["threshold_curve_json"]) if row else []
        threshold = min(
            curve,
            key=lambda item: abs(item["percentage"] - percentage),
            default={"threshold": float("inf")},
        )["threshold"]
        trajectory_result = {}
        anomalous_ids = set()
        for score in trajectory_scores.to_dict("records"):
            anomalous = float(score["score"]) >= threshold
            if anomalous:
                anomalous_ids.add(score["trajectory_id"])
            trajectory_result[str(score["trajectory_id"])] = {
                **score,
                "reconstruction_error": score["score"],
                "is_unplausible": anomalous,
            }
        highest_steps = {}
        for score in transition_scores.to_dict("records"):
            current = highest_steps.get(score["trajectory_id"])
            if current is None or score["score"] > current["score"]:
                highest_steps[score["trajectory_id"]] = score
        transition_result = {}
        for score in transition_scores.to_dict("records"):
            feature_keys = [key for key in score if key.startswith("error_")]
            is_least_plausible = (
                score["trajectory_id"] in anomalous_ids
                and highest_steps[score["trajectory_id"]]["transition_id"]
                == score["transition_id"]
            )
            transition_result[str(score["transition_id"])] = {
                **score,
                "mse": score["score"],
                "features": [score[key] for key in feature_keys],
                "feature_names": [key.removeprefix("error_") for key in feature_keys],
                "is_unplausible": is_least_plausible,
            }
        self.send_json(
            {
                "threshold": threshold,
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
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def main():
    global DB_PATH, PREDICTION_ROOT, PROCESSED_ROOT
    parser = argparse.ArgumentParser(description="Serve the trajectory plausibility visualization.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument("--prediction-root", default=str(PREDICTION_ROOT))
    parser.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    args = parser.parse_args()
    DB_PATH = Path(args.db_path)
    PREDICTION_ROOT = Path(args.prediction_root)
    PROCESSED_ROOT = Path(args.processed_root)
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} does not exist. Run build_database.py first.")
    server = ThreadingHTTPServer((args.host, args.port), VisualizationHandler)
    print(f"Visualization: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
