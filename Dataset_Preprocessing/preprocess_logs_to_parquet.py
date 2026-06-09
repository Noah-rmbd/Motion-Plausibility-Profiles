import argparse
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .utils import compute_bearing
except ImportError:
    from utils import compute_bearing


DEFAULT_INPUTS = {
    "inat": Path("data/raw/iNaturalist/frequent-poster.log"),
    "gowalla": Path("data/raw/gowalla/gowalla_frequent_poster.log"),
}

FALLBACK_INPUTS = {
    "inat": Path("data/processed/iNaturalist/frequent-poster.log"),
    "gowalla": Path("data/processed/gowalla/gowalla_frequent_poster.log"),
}

OBSERVATION_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("user_id", pa.string()),
        ("observation_id", pa.string()),
        ("timestamp", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("is_obscured", pa.bool_()),
    ]
)

TRANSITION_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("transition_id", pa.int64()),
        ("observation_id1", pa.string()),
        ("observation_id2", pa.string()),
        ("speed_kmh", pa.float64()),
        ("elapsed_time_s", pa.float64()),
        ("distance_m", pa.float64()),
        ("acceleration_m_s2", pa.float64()),
        ("bearing_change_rad", pa.float64()),
        ("transition_plausibility", pa.int8()),
        ("plausibility_reason", pa.string()),
    ]
)

TRAJECTORY_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("trajectory_id", pa.int64()),
        ("user_id", pa.string()),
        ("n_transitions", pa.int64()),
        ("start_observation_id", pa.string()),
        ("end_observation_id", pa.string()),
    ]
)

TRAJECTORY_TRANSITION_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("trajectory_id", pa.int64()),
        ("transition_order", pa.int64()),
        ("transition_id", pa.int64()),
    ]
)


class ParquetBatchWriter:
    def __init__(self, path, schema, batch_size):
        self.path = Path(path)
        self.schema = schema
        self.batch_size = batch_size
        self.rows = []
        self.writer = None
        self.count = 0

    def append(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table)
        self.count += len(self.rows)
        self.rows = []

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()


def parse_float(value):
    value = value.strip()
    if value in {"", "N/A"}:
        return None
    return float(value)


def parse_metric(value, suffix):
    value = value.strip()
    if value in {"", "N/A"}:
        return None
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return parse_float(value)


def normalize_bearing_delta(delta):
    return (delta + math.pi) % (2 * math.pi) - math.pi


def transition_plausibility(speed_kmh, elapsed_time_s, distance_m, acceleration_m_s2):
    if speed_kmh > 900.0:
        return 0, "Too high speed"
    if elapsed_time_s > 86400.0:
        return 0, "Elapsed time exceeds 24 hours"
    if distance_m > 10000000.0:
        return 0, "Distance exceeds 10,000 km"
    if abs(acceleration_m_s2) > 50.0:
        return 0, "Acceleration exceeds 50 m/s^2"
    if (
        (speed_kmh > 30.0 and elapsed_time_s <= 30)
        or (speed_kmh > 200.0 and elapsed_time_s <= 300)
        or (speed_kmh > 300.0 and elapsed_time_s <= 3600)
    ):
        return 0, "This speed cannot be reached with any vehicle during such a small amount of time"
    return 1, None


def is_obscured_observation(observation_id, status):
    status = status.strip().lower()
    return observation_id.startswith("iN-o") or status == "obscured"


def observation_row(dataset, user_id, observation_id, date, time, lat, lon, is_obscured):
    return {
        "dataset": dataset,
        "user_id": user_id,
        "observation_id": observation_id,
        "timestamp": f"{date} {time}" if date and time else None,
        "lat": lat,
        "lon": lon,
        "is_obscured": is_obscured,
    }


def preprocess_log(dataset, input_path, output_dir, batch_size=100_000):
    input_path = Path(input_path)
    output_dir = Path(output_dir) / dataset
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    observations = ParquetBatchWriter(output_dir / "observations.parquet", OBSERVATION_SCHEMA, batch_size)
    transitions = ParquetBatchWriter(output_dir / "transitions.parquet", TRANSITION_SCHEMA, batch_size)
    trajectories = ParquetBatchWriter(output_dir / "trajectories.parquet", TRAJECTORY_SCHEMA, batch_size)
    trajectory_transitions = ParquetBatchWriter(
        output_dir / "trajectory_transitions.parquet",
        TRAJECTORY_TRANSITION_SCHEMA,
        batch_size,
    )

    current_user = None
    current_date = None
    previous_point = None
    transition_id = 0
    trajectory_id = 0
    pending_transitions = []

    def finish_trajectory():
        nonlocal trajectory_id, pending_transitions
        if not pending_transitions:
            return

        first = pending_transitions[0]
        last = pending_transitions[-1]
        trajectories.append(
            {
                "dataset": dataset,
                "trajectory_id": trajectory_id,
                "user_id": current_user,
                "n_transitions": len(pending_transitions),
                "start_observation_id": first["observation_id1"],
                "end_observation_id": last["observation_id2"],
            }
        )
        for order, transition in enumerate(pending_transitions):
            transitions.append(transition)
            trajectory_transitions.append(
                {
                    "dataset": dataset,
                    "trajectory_id": trajectory_id,
                    "transition_order": order,
                    "transition_id": transition["transition_id"],
                }
            )
        trajectory_id += 1
        pending_transitions = []

    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("="):
                continue

            parts = [part.strip() for part in line.split(",")]

            if len(parts) == 2:
                finish_trajectory()
                current_user = parts[0]
                current_date = None
                previous_point = None
                continue

            if len(parts) not in {6, 8}:
                continue

            observation_id = parts[0]
            lat = parse_float(parts[1])
            lon = parse_float(parts[2])
            date = parts[3]
            time = parts[4]
            status = parts[5] if len(parts) == 6 else ""
            obscured = is_obscured_observation(observation_id, status)

            observations.append(
                observation_row(dataset, current_user, observation_id, date, time, lat, lon, obscured)
            )

            if obscured or lat is None or lon is None:
                continue

            if len(parts) == 6:
                if status != "first entry of the day":
                    continue
                if date != current_date:
                    finish_trajectory()
                    current_date = date
                previous_point = {
                    "observation_id": observation_id,
                    "lat": lat,
                    "lon": lon,
                    "speed_ms": 0.0,
                    "bearing": None,
                }
                continue

            if date != current_date or previous_point is None:
                continue

            elapsed_time_s = parse_metric(parts[5], "s")
            distance_m = parse_metric(parts[6], "m")
            if elapsed_time_s is None or distance_m is None:
                continue

            if elapsed_time_s > 0:
                speed_kmh = (distance_m / elapsed_time_s) * 3.6
            else:
                speed_kmh = parse_metric(parts[7], "km/h") or 0.0
                if speed_kmh < 0:
                    speed_kmh = 0.0

            speed_ms = speed_kmh / 3.6
            acceleration_m_s2 = (
                (speed_ms - previous_point["speed_ms"]) / elapsed_time_s
                if elapsed_time_s > 0
                else 0.0
            )

            bearing = compute_bearing(previous_point["lat"], previous_point["lon"], lat, lon)
            if previous_point["bearing"] is None:
                bearing_change_rad = 0.0
            else:
                bearing_change_rad = normalize_bearing_delta(bearing - previous_point["bearing"])

            plausible, reason = transition_plausibility(
                speed_kmh,
                elapsed_time_s,
                distance_m,
                acceleration_m_s2,
            )

            pending_transitions.append(
                {
                    "dataset": dataset,
                    "transition_id": transition_id,
                    "observation_id1": previous_point["observation_id"],
                    "observation_id2": observation_id,
                    "speed_kmh": speed_kmh,
                    "elapsed_time_s": elapsed_time_s,
                    "distance_m": distance_m,
                    "acceleration_m_s2": acceleration_m_s2,
                    "bearing_change_rad": bearing_change_rad,
                    "transition_plausibility": plausible,
                    "plausibility_reason": reason,
                }
            )
            transition_id += 1

            previous_point = {
                "observation_id": observation_id,
                "lat": lat,
                "lon": lon,
                "speed_ms": speed_ms,
                "bearing": bearing,
            }

    finish_trajectory()

    observations.close()
    transitions.close()
    trajectories.close()
    trajectory_transitions.close()

    return {
        "observations": observations.count,
        "transitions": transitions.count,
        "trajectories": trajectories.count,
        "trajectory_transitions": trajectory_transitions.count,
    }


def preprocess_many(datasets, output_dir, batch_size):
    results = {}
    for dataset, input_path in datasets.items():
        print(f"Preprocessing {dataset}: {input_path}")
        results[dataset] = preprocess_log(
            dataset=dataset,
            input_path=input_path,
            output_dir=output_dir,
            batch_size=batch_size,
        )
        print(f"  {results[dataset]}")
    return results


def parse_dataset_args(values):
    if not values:
        datasets = {}
        for name, default_path in DEFAULT_INPUTS.items():
            datasets[name] = default_path if default_path.exists() else FALLBACK_INPUTS[name]
        return datasets

    datasets = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Dataset arguments must use name=path format, got: {value}")
        name, path = value.split("=", 1)
        datasets[name] = Path(path)
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess frequent-poster logs into canonical Parquet datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset input in name=path format. Can be repeated. Defaults to current iNat and Gowalla logs.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_parquet",
        help="Directory where dataset subdirectories and Parquet files are written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Number of rows per Parquet write batch.",
    )
    args = parser.parse_args()

    datasets = parse_dataset_args(args.dataset)
    preprocess_many(datasets, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
