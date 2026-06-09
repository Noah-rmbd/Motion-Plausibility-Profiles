import argparse
import json
import math
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .cleaning import select_valid_trajectory_ids
    from .preprocess_logs_to_parquet import (
        OBSERVATION_SCHEMA,
        TRAJECTORY_SCHEMA,
        TRAJECTORY_TRANSITION_SCHEMA,
        TRANSITION_SCHEMA,
        normalize_bearing_delta,
        transition_plausibility,
    )
    from .utils import compute_bearing
except ImportError:
    from cleaning import select_valid_trajectory_ids
    from preprocess_logs_to_parquet import (
        OBSERVATION_SCHEMA,
        TRAJECTORY_SCHEMA,
        TRAJECTORY_TRANSITION_SCHEMA,
        TRANSITION_SCHEMA,
        normalize_bearing_delta,
        transition_plausibility,
    )
    from utils import compute_bearing


PROFILE_NAMES = ("back_and_forth", "abnormal_speed", "initial_fix_jump")
METADATA_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("trajectory_id", pa.int64()),
        ("source_dataset", pa.string()),
        ("source_trajectory_id", pa.int64()),
        ("anomaly_profile", pa.string()),
        ("anomaly_start_order", pa.int64()),
        ("anomaly_end_order", pa.int64()),
        ("generation_seed", pa.int64()),
    ]
)


def destination_point(lat, lon, distance_m, bearing_rad):
    earth_radius_m = 6_371_000.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    angular_distance = distance_m / earth_radius_m
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def write_parquet(rows, path, schema):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def load_clean_source_trajectories(processed_root, source_dataset, min_transitions, count, rng):
    source_dir = Path(processed_root) / source_dataset
    trajectories = pd.read_parquet(source_dir / "trajectories.parquet")
    trajectory_transitions = pd.read_parquet(source_dir / "trajectory_transitions.parquet")
    transition_flags = pd.read_parquet(
        source_dir / "transitions.parquet",
        columns=["transition_id", "transition_plausibility"],
    )
    valid_ids, _ = select_valid_trajectory_ids(
        transition_flags,
        trajectory_transitions,
        trajectories,
        min_transitions,
    )
    candidates = np.asarray(sorted(valid_ids), dtype=np.int64)
    if len(candidates) < count:
        raise ValueError(
            f"{source_dataset}: requested {count} source trajectories but only "
            f"{len(candidates)} satisfy the cleaning policy"
        )
    selected_ids = np.sort(rng.choice(candidates, size=count, replace=False))

    selected_mapping = trajectory_transitions.loc[
        trajectory_transitions["trajectory_id"].isin(selected_ids)
    ].sort_values(["trajectory_id", "transition_order"])
    transition_ids = selected_mapping["transition_id"].unique().tolist()
    transitions = pd.read_parquet(
        source_dir / "transitions.parquet",
        filters=[("transition_id", "in", transition_ids)],
    )
    observation_ids = pd.unique(
        pd.concat([transitions["observation_id1"], transitions["observation_id2"]], ignore_index=True)
    ).tolist()
    observations = pd.read_parquet(
        source_dir / "observations.parquet",
        filters=[("observation_id", "in", observation_ids)],
    )
    source_trajectories = trajectories.loc[trajectories["trajectory_id"].isin(selected_ids)]
    return source_trajectories, selected_mapping, transitions, observations


def inject_back_and_forth(frame, rng):
    output = frame.copy()
    start = int(rng.integers(1, max(2, len(output) - 1)))
    end = min(start + 1, len(output) - 1)
    output.loc[start:end, "bearing_change_rad"] = [
        normalize_bearing_delta(math.pi + rng.normal(0.0, 0.08))
        for _ in range(end - start + 1)
    ]
    return output, start, end


def inject_abnormal_speed(frame, rng):
    output = frame.copy()
    max_length = min(4, len(output) - 1)
    length = int(rng.integers(2, max_length + 1))
    start = int(rng.integers(1, len(output) - length + 1))
    end = start + length - 1
    speeds = rng.uniform(500.0, 850.0, size=length)
    elapsed = rng.uniform(300.0, 1_800.0, size=length)
    output.loc[start:end, "elapsed_time_s"] = elapsed
    output.loc[start:end, "distance_m"] = speeds * elapsed / 3.6
    return output, start, end


def inject_initial_fix_jump(frame, rng):
    output = frame.copy()
    end = min(1, len(output) - 1)
    jump_distance = float(rng.uniform(5_000.0, 20_000.0))
    output.loc[0, "distance_m"] += jump_distance
    if end == 1:
        output.loc[1, "distance_m"] += jump_distance
        output.loc[1, "bearing_change_rad"] = normalize_bearing_delta(
            math.pi + rng.normal(0.0, 0.05)
        )
    return output, 0, end


PROFILE_INJECTORS = {
    "back_and_forth": inject_back_and_forth,
    "abnormal_speed": inject_abnormal_speed,
    "initial_fix_jump": inject_initial_fix_jump,
}


def recompute_dependent_motion(frame):
    output = frame.copy()
    elapsed = output["elapsed_time_s"].to_numpy(dtype=float)
    distance = output["distance_m"].to_numpy(dtype=float)
    speed_kmh = np.divide(distance, elapsed, out=np.zeros_like(distance), where=elapsed > 0) * 3.6
    speed_m_s = speed_kmh / 3.6
    previous_speed = np.concatenate(([0.0], speed_m_s[:-1]))
    acceleration = np.divide(
        speed_m_s - previous_speed,
        elapsed,
        out=np.zeros_like(speed_m_s),
        where=elapsed > 0,
    )
    output["speed_kmh"] = speed_kmh
    output["acceleration_m_s2"] = acceleration
    return output


def parse_start_timestamp(value):
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.Timestamp("2000-01-01 00:00:00")
    return timestamp


def build_synthetic_trajectory(
    source_dataset,
    source_trajectory_id,
    source_transitions,
    source_observations,
    profile,
    trajectory_id,
    first_transition_id,
    seed,
):
    rng = np.random.default_rng(seed)
    mutated, anomaly_start, anomaly_end = PROFILE_INJECTORS[profile](source_transitions, rng)
    mutated = recompute_dependent_motion(mutated)

    observations_by_id = source_observations.set_index("observation_id")
    first_source = source_transitions.iloc[0]
    first_point = observations_by_id.loc[first_source["observation_id1"]]
    second_point = observations_by_id.loc[first_source["observation_id2"]]
    current_lat = float(first_point["lat"])
    current_lon = float(first_point["lon"])
    current_bearing = compute_bearing(
        current_lat,
        current_lon,
        float(second_point["lat"]),
        float(second_point["lon"]),
    )
    current_timestamp = parse_start_timestamp(first_point["timestamp"])
    user_id = f"synthetic:{profile}"
    observation_prefix = f"synthetic-{trajectory_id}"
    observation_rows = [
        {
            "dataset": "synthetic",
            "user_id": user_id,
            "observation_id": f"{observation_prefix}-0",
            "timestamp": current_timestamp.isoformat(sep=" "),
            "lat": current_lat,
            "lon": current_lon,
            "is_obscured": False,
        }
    ]
    transition_rows = []
    mapping_rows = []

    for order, row in mutated.reset_index(drop=True).iterrows():
        current_bearing = normalize_bearing_delta(
            current_bearing + float(row["bearing_change_rad"])
        )
        next_lat, next_lon = destination_point(
            current_lat,
            current_lon,
            float(row["distance_m"]),
            current_bearing,
        )
        current_timestamp += timedelta(seconds=float(row["elapsed_time_s"]))
        transition_id = first_transition_id + order
        observation_id1 = f"{observation_prefix}-{order}"
        observation_id2 = f"{observation_prefix}-{order + 1}"
        plausible, reason = transition_plausibility(
            float(row["speed_kmh"]),
            float(row["elapsed_time_s"]),
            float(row["distance_m"]),
            float(row["acceleration_m_s2"]),
        )
        transition_rows.append(
            {
                "dataset": "synthetic",
                "transition_id": transition_id,
                "observation_id1": observation_id1,
                "observation_id2": observation_id2,
                "speed_kmh": float(row["speed_kmh"]),
                "elapsed_time_s": float(row["elapsed_time_s"]),
                "distance_m": float(row["distance_m"]),
                "acceleration_m_s2": float(row["acceleration_m_s2"]),
                "bearing_change_rad": float(row["bearing_change_rad"]),
                "transition_plausibility": plausible,
                "plausibility_reason": reason,
            }
        )
        mapping_rows.append(
            {
                "dataset": "synthetic",
                "trajectory_id": trajectory_id,
                "transition_order": order,
                "transition_id": transition_id,
            }
        )
        observation_rows.append(
            {
                "dataset": "synthetic",
                "user_id": user_id,
                "observation_id": observation_id2,
                "timestamp": current_timestamp.isoformat(sep=" "),
                "lat": next_lat,
                "lon": next_lon,
                "is_obscured": False,
            }
        )
        current_lat, current_lon = next_lat, next_lon

    trajectory_row = {
        "dataset": "synthetic",
        "trajectory_id": trajectory_id,
        "user_id": user_id,
        "n_transitions": len(transition_rows),
        "start_observation_id": observation_rows[0]["observation_id"],
        "end_observation_id": observation_rows[-1]["observation_id"],
    }
    metadata_row = {
        "dataset": "synthetic",
        "trajectory_id": trajectory_id,
        "source_dataset": source_dataset,
        "source_trajectory_id": int(source_trajectory_id),
        "anomaly_profile": profile,
        "anomaly_start_order": anomaly_start,
        "anomaly_end_order": anomaly_end,
        "generation_seed": seed,
    }
    return observation_rows, transition_rows, trajectory_row, mapping_rows, metadata_row


def generate_synthetic_dataset(
    processed_root,
    output_dir,
    source_dataset="gowalla",
    trajectories_per_profile=50,
    min_transitions=4,
    seed=42,
    profiles=PROFILE_NAMES,
):
    rng = np.random.default_rng(seed)
    source_trajectories, mapping, transitions, observations = load_clean_source_trajectories(
        processed_root=processed_root,
        source_dataset=source_dataset,
        min_transitions=min_transitions,
        count=trajectories_per_profile,
        rng=rng,
    )
    transition_lookup = transitions.set_index("transition_id")
    observation_lookup = observations.set_index("observation_id", drop=False)

    all_observations = []
    all_transitions = []
    all_trajectories = []
    all_mappings = []
    all_metadata = []
    next_transition_id = 0

    for profile_index, profile in enumerate(profiles):
        if profile not in PROFILE_INJECTORS:
            raise ValueError(f"Unknown anomaly profile: {profile}")
        for source_index, source in enumerate(source_trajectories.itertuples(index=False)):
            source_mapping = mapping.loc[mapping["trajectory_id"] == source.trajectory_id]
            source_transition_frame = transition_lookup.loc[
                source_mapping["transition_id"].to_numpy()
            ].reset_index()
            source_observation_ids = pd.unique(
                pd.concat(
                    [
                        source_transition_frame["observation_id1"],
                        source_transition_frame["observation_id2"],
                    ],
                    ignore_index=True,
                )
            )
            source_observation_frame = observation_lookup.loc[source_observation_ids].reset_index(
                drop=True
            )
            trajectory_id = profile_index * trajectories_per_profile + source_index
            trajectory_seed = seed + trajectory_id + 1
            generated = build_synthetic_trajectory(
                source_dataset=source_dataset,
                source_trajectory_id=source.trajectory_id,
                source_transitions=source_transition_frame,
                source_observations=source_observation_frame,
                profile=profile,
                trajectory_id=trajectory_id,
                first_transition_id=next_transition_id,
                seed=trajectory_seed,
            )
            observation_rows, transition_rows, trajectory_row, mapping_rows, metadata_row = generated
            all_observations.extend(observation_rows)
            all_transitions.extend(transition_rows)
            all_trajectories.append(trajectory_row)
            all_mappings.extend(mapping_rows)
            all_metadata.append(metadata_row)
            next_transition_id += len(transition_rows)

    output_dir = Path(output_dir)
    temp_dir = output_dir.with_name(f"{output_dir.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    write_parquet(all_observations, temp_dir / "observations.parquet", OBSERVATION_SCHEMA)
    write_parquet(all_transitions, temp_dir / "transitions.parquet", TRANSITION_SCHEMA)
    write_parquet(all_trajectories, temp_dir / "trajectories.parquet", TRAJECTORY_SCHEMA)
    write_parquet(
        all_mappings,
        temp_dir / "trajectory_transitions.parquet",
        TRAJECTORY_TRANSITION_SCHEMA,
    )
    write_parquet(all_metadata, temp_dir / "synthetic_metadata.parquet", METADATA_SCHEMA)

    config = {
        "source_dataset": source_dataset,
        "trajectories_per_profile": trajectories_per_profile,
        "min_source_transitions": min_transitions,
        "seed": seed,
        "profiles": list(profiles),
        "output": {
            "observations": len(all_observations),
            "transitions": len(all_transitions),
            "trajectories": len(all_trajectories),
        },
    }
    with (temp_dir / "generation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    if output_dir.exists():
        backup_dir = output_dir.with_name(f"{output_dir.name}.previous")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        output_dir.replace(backup_dir)
        temp_dir.replace(output_dir)
        shutil.rmtree(backup_dir)
    else:
        temp_dir.replace(output_dir)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate canonical synthetic anomalies from clean processed trajectories."
    )
    parser.add_argument("--processed-root", default="data/processed_parquet")
    parser.add_argument("--output-dir", default="data/processed_parquet/synthetic")
    parser.add_argument("--source-dataset", default="gowalla")
    parser.add_argument("--trajectories-per-profile", type=int, default=50)
    parser.add_argument("--min-transitions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profiles", nargs="+", choices=PROFILE_NAMES, default=list(PROFILE_NAMES))
    args = parser.parse_args()

    config = generate_synthetic_dataset(
        processed_root=args.processed_root,
        output_dir=args.output_dir,
        source_dataset=args.source_dataset,
        trajectories_per_profile=args.trajectories_per_profile,
        min_transitions=args.min_transitions,
        seed=args.seed,
        profiles=args.profiles,
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
