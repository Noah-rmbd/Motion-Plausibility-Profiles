import argparse
import json
import mimetypes
import random
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

import open_visualization as visualization
from modeling.plausible_labels import DEFAULT_LABEL_PATH, read_plausible_label_table
from modeling.review_labels import (
    DEFAULT_REVIEW_LABEL_PATH,
    REVIEW_COLUMNS,
    VALID_LABELS,
    read_review_label_table,
    write_review_label_table,
)


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "trajectory_review_tool"
LABEL_PATH = DEFAULT_REVIEW_LABEL_PATH
DEFAULT_DATASETS = ("inat", "gowalla")
REVIEWABLE_TRAJECTORY_CACHE = {}


def reviewed_keys():
    keys = set()
    reviews = read_review_label_table(LABEL_PATH)
    for row in reviews.itertuples(index=False):
        if row.trajectory_id:
            keys.add((str(row.dataset), int(row.trajectory_id)))

    legacy = read_plausible_label_table(DEFAULT_LABEL_PATH)
    for row in legacy.itertuples(index=False):
        if row.trajectory_id:
            keys.add((str(row.dataset), int(row.trajectory_id)))
    return keys


def start_date_for_row(row):
    timestamp = row.get("start_timestamp")
    if timestamp:
        return str(timestamp)[:10].replace("-", "/")
    return "*"


def reviewable_trajectory_ids(dataset, min_transitions=3):
    cache_key = (str(dataset), int(min_transitions), str(visualization.PROCESSED_ROOT))
    if cache_key in REVIEWABLE_TRAJECTORY_CACHE:
        return REVIEWABLE_TRAJECTORY_CACHE[cache_key]

    base = visualization.PROCESSED_ROOT / dataset
    trajectories = pd.read_parquet(
        base / "trajectories.parquet",
        columns=["trajectory_id", "n_transitions"],
    )
    eligible_ids = set(
        trajectories.loc[
            trajectories["n_transitions"].astype(int) >= int(min_transitions),
            "trajectory_id",
        ].astype(int)
    )
    mapping = pd.read_parquet(
        base / "trajectory_transitions.parquet",
        columns=["trajectory_id", "transition_id"],
        filters=[("trajectory_id", "in", list(eligible_ids))] if eligible_ids else None,
    )
    if mapping.empty:
        REVIEWABLE_TRAJECTORY_CACHE[cache_key] = frozenset()
        return REVIEWABLE_TRAJECTORY_CACHE[cache_key]
    transitions = pd.read_parquet(
        base / "transitions.parquet",
        columns=["transition_id", "transition_plausibility"],
        filters=[("transition_id", "in", mapping["transition_id"].astype(int).tolist())],
    )
    merged = mapping.merge(transitions, on="transition_id", how="left")
    valid_by_trajectory = merged.groupby("trajectory_id")[
        "transition_plausibility"
    ].min().eq(1)
    valid_ids = frozenset(
        int(trajectory_id)
        for trajectory_id in valid_by_trajectory.index[valid_by_trajectory]
    )
    REVIEWABLE_TRAJECTORY_CACHE[cache_key] = valid_ids
    return valid_ids


def candidate_for_review(datasets=DEFAULT_DATASETS, min_transitions=3):
    datasets = [dataset for dataset in datasets if dataset in DEFAULT_DATASETS]
    if not datasets:
        raise ValueError("At least one real dataset is required")
    rng = random.SystemRandom()
    blocked = reviewed_keys()
    shuffled_datasets = list(datasets)
    rng.shuffle(shuffled_datasets)

    with visualization.connection() as conn:
        for dataset in shuffled_datasets:
            reviewable_ids = set(reviewable_trajectory_ids(dataset, min_transitions))
            if not reviewable_ids:
                continue
            blocked_ids = [trajectory_id for item_dataset, trajectory_id in blocked if item_dataset == dataset]
            candidate_ids = sorted(reviewable_ids - set(blocked_ids))
            if not candidate_ids:
                continue
            candidate_frame = pd.read_sql_query(
                """
                SELECT *
                FROM trajectories
                WHERE dataset = ? AND n_transitions >= ?
                """,
                conn,
                params=(dataset, min_transitions),
            )
            candidate_frame = candidate_frame.loc[
                candidate_frame["trajectory_id"].astype(int).isin(candidate_ids)
            ]
            users = candidate_frame["user_id"].astype(str).drop_duplicates().tolist()
            if not users:
                continue
            rng.shuffle(users)
            for user_id in users:
                user_frame = candidate_frame.loc[
                    candidate_frame["user_id"].astype(str).eq(str(user_id))
                ]
                if not user_frame.empty:
                    return user_frame.sample(n=1, random_state=rng.randrange(2**32)).iloc[0].to_dict()
    return None


def build_review_payload(row):
    dataset = row["dataset"]
    trajectory_id = int(row["trajectory_id"])
    payload = visualization.canonical_payload(dataset, [trajectory_id])
    transitions = payload["transitions"]
    baseline_invalid = [
        transition
        for transition in transitions
        if transition.get("transition_plausibility") != 1
    ]
    if baseline_invalid:
        raise ValueError(
            f"Trajectory {dataset}/{trajectory_id} violates deterministic rules"
        )
    payload["review"] = {
        "dataset": dataset,
        "user_id": str(row["user_id"]),
        "trajectory_id": trajectory_id,
        "n_transitions": int(row["n_transitions"]),
        "start_timestamp": row.get("start_timestamp"),
        "end_timestamp": row.get("end_timestamp"),
        "start_date": start_date_for_row(row),
        "baseline_invalid_transitions": len(baseline_invalid),
        "baseline_reasons": sorted(
            {
                transition.get("plausibility_reason") or "deterministic policy"
                for transition in baseline_invalid
            }
        ),
    }
    return payload


def save_review_label(dataset, user_id, trajectory_id, label, notes=""):
    label = str(label).strip().lower()
    if label not in VALID_LABELS:
        raise ValueError(f"Unknown review label: {label}")
    if dataset not in DEFAULT_DATASETS:
        raise ValueError("Only real iNaturalist and Gowalla trajectories can be reviewed")

    with visualization.connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM trajectories
            WHERE dataset = ? AND trajectory_id = ?
            """,
            (dataset, int(trajectory_id)),
        ).fetchone()
    if row is None:
        raise ValueError(f"Unknown trajectory: {dataset}/{trajectory_id}")
    row = dict(row)
    if str(row["user_id"]) != str(user_id):
        raise ValueError("Trajectory does not belong to the submitted user")
    if int(trajectory_id) not in reviewable_trajectory_ids(dataset):
        raise ValueError(
            "This trajectory violates deterministic rules and cannot be reviewed "
            "for the manual benchmark"
        )

    labels = read_review_label_table(LABEL_PATH)
    duplicate = (
        labels["dataset"].astype(str).eq(str(dataset))
        & labels["trajectory_id"].astype(str).eq(str(trajectory_id))
    )
    labels = labels.loc[~duplicate].copy()
    new_row = {
        "dataset": dataset,
        "user_id": str(user_id),
        "start_date": start_date_for_row(row),
        "trajectory_id": str(int(trajectory_id)),
        "label": label,
        "source": "trajectory_review_tool",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": str(notes or ""),
    }
    labels = pd.concat(
        [labels, pd.DataFrame([new_row], columns=REVIEW_COLUMNS)],
        ignore_index=True,
    )
    write_review_label_table(labels, LABEL_PATH)
    return new_row


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def send_json(self, payload, status=200):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/api/next":
                datasets = query.get("datasets", [",".join(DEFAULT_DATASETS)])[0]
                dataset_list = [item.strip() for item in datasets.split(",") if item.strip()]
                row = candidate_for_review(dataset_list)
                if row is None:
                    return self.send_json({"done": True})
                return self.send_json(build_review_payload(row))
            return self.static_file(parsed.path)
        except (ValueError, KeyError) as exc:
            return self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            return self.send_json({"error": str(exc)}, 500)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/review":
            return self.send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            label = save_review_label(
                dataset=request["dataset"],
                user_id=request["user_id"],
                trajectory_id=request["trajectory_id"],
                label=request["label"],
                notes=request.get("notes", ""),
            )
            return self.send_json({"saved": label})
        except (ValueError, KeyError) as exc:
            return self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            return self.send_json({"error": str(exc)}, 500)

    def static_file(self, request_path):
        relative = "main.html" if request_path in {"", "/"} else request_path.lstrip("/")
        path = (STATIC_ROOT / relative).resolve()
        if not path.is_file() or STATIC_ROOT.resolve() not in path.parents:
            return self.send_json({"error": "not found"}, 404)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global LABEL_PATH
    parser = argparse.ArgumentParser(
        description="Launch a Tinder-style trajectory benchmark review tool."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--db-path", default=str(visualization.DB_PATH))
    parser.add_argument("--processed-root", default=str(visualization.PROCESSED_ROOT))
    parser.add_argument("--label-path", default=str(DEFAULT_REVIEW_LABEL_PATH))
    args = parser.parse_args()

    visualization.DB_PATH = Path(args.db_path)
    visualization.PROCESSED_ROOT = Path(args.processed_root)
    LABEL_PATH = Path(args.label_path)
    if not visualization.DB_PATH.exists():
        raise FileNotFoundError(
            f"{visualization.DB_PATH} does not exist. Run build_database.py first."
        )

    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Trajectory review tool: http://{args.host}:{args.port}")
    print(f"Writing labels to: {LABEL_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
