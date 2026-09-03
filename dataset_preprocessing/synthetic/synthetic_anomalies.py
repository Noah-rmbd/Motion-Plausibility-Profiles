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
    from dataset_preprocessing.features.cleaning import select_valid_trajectory_ids
    from dataset_preprocessing.ingestion.preprocess_logs_to_parquet import (
        OBSERVATION_SCHEMA,
        TRAJECTORY_SCHEMA,
        TRAJECTORY_TRANSITION_SCHEMA,
        TRANSITION_SCHEMA,
        normalize_bearing_delta,
        transition_plausibility,
    )
    from dataset_preprocessing.features.sessionization import (
        DEFAULT_MAX_INACTIVITY_SECONDS,
        anomalous_session_range,
    )
    from dataset_preprocessing.features.splitting import user_split
    from dataset_preprocessing.utils.utils import compute_bearing
    from modeling.plausible_labels import (
        DEFAULT_LABEL_PATH,
        resolve_plausible_trajectories_from_processed,
    )
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from dataset_preprocessing.features.cleaning import select_valid_trajectory_ids
    from dataset_preprocessing.ingestion.preprocess_logs_to_parquet import (
        OBSERVATION_SCHEMA,
        TRAJECTORY_SCHEMA,
        TRAJECTORY_TRANSITION_SCHEMA,
        TRANSITION_SCHEMA,
        normalize_bearing_delta,
        transition_plausibility,
    )
    from dataset_preprocessing.features.sessionization import (
        DEFAULT_MAX_INACTIVITY_SECONDS,
        anomalous_session_range,
    )
    from dataset_preprocessing.features.splitting import user_split
    from dataset_preprocessing.utils.utils import compute_bearing
    from modeling.plausible_labels import (
        DEFAULT_LABEL_PATH,
        resolve_plausible_trajectories_from_processed,
    )


PROFILE_NAMES = ("back_and_forth", "abnormal_speed", "initial_fix_jump", "coordinate_jitter", "timestamp_jitter")
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


def load_clean_source_trajectories(
    processed_root,
    source_dataset,
    min_transitions,
    count,
    rng,
    max_inactivity_seconds=DEFAULT_MAX_INACTIVITY_SECONDS,
    plausible_labels_path=DEFAULT_LABEL_PATH,
):
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
    calibration_trajectories = trajectories.loc[
        trajectories["trajectory_id"].isin(valid_ids)
    ].copy()
    calibration_trajectories = calibration_trajectories.loc[
        calibration_trajectories["user_id"].map(
            lambda user_id: user_split(source_dataset, user_id) == "calibration"
        )
    ]
    elapsed_times = pd.read_parquet(
        source_dir / "transitions.parquet",
        columns=["transition_id", "elapsed_time_s"],
    )
    if max_inactivity_seconds is not None and max_inactivity_seconds > 0:
        inactivity_ids = elapsed_times.loc[
            elapsed_times["elapsed_time_s"] > max_inactivity_seconds,
            "transition_id",
        ]
        trajectories_with_gaps = trajectory_transitions.loc[
            trajectory_transitions["transition_id"].isin(inactivity_ids),
            "trajectory_id",
        ].unique()
        calibration_trajectories = calibration_trajectories.loc[
            ~calibration_trajectories["trajectory_id"].isin(trajectories_with_gaps)
        ]
    invalid_elapsed_ids = elapsed_times.loc[
        elapsed_times["elapsed_time_s"] <= 0,
        "transition_id",
    ]
    trajectories_with_invalid_elapsed = trajectory_transitions.loc[
        trajectory_transitions["transition_id"].isin(invalid_elapsed_ids),
        "trajectory_id",
    ].unique()
    calibration_trajectories = calibration_trajectories.loc[
        ~calibration_trajectories["trajectory_id"].isin(
            trajectories_with_invalid_elapsed
        )
    ]
    candidates = (
        calibration_trajectories.sort_values(["user_id", "trajectory_id"])
        .groupby("user_id", sort=False)
        .sample(n=1, random_state=int(rng.integers(0, 2**32 - 1)))
    )
    if plausible_labels_path and Path(plausible_labels_path).exists():
        reviewed = resolve_plausible_trajectories_from_processed(
            processed_root=processed_root,
            label_path=plausible_labels_path,
        )
        reviewed_ids = set(
            reviewed.loc[
                reviewed["dataset"].astype(str).eq(str(source_dataset)),
                "trajectory_id",
            ].astype(int)
        )
        reviewed_candidates = candidates.loc[
            candidates["trajectory_id"].isin(reviewed_ids)
        ]
        if len(reviewed_candidates) >= count:
            candidates = reviewed_candidates
    if len(candidates) < count:
        raise ValueError(
            f"{source_dataset}: requested {count} source trajectories but only "
            f"{len(candidates)} calibration users satisfy the cleaning policy"
        )
    selected_ids = np.sort(
        rng.choice(
            candidates["trajectory_id"].to_numpy(dtype=np.int64),
            size=count,
            replace=False,
        )
    )

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


def haversine_distance(lat1, lon1, lat2, lon2):
    import math
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def inject_back_and_forth(frame, source_observations, rng):
    output = frame.copy()
    max_length = min(8, len(output) - 1)
    min_length = min(4, max_length)
    length = int(rng.integers(min_length, max_length + 1))
    start = int(rng.integers(1, len(output) - length + 1))
    end = start + length - 1
    source_distance = output.loc[start:end, "distance_m"].replace(0, np.nan)
    segment_distance = float(
        np.clip(source_distance.median(), 50.0, 500.0)
        if source_distance.notna().any()
        else rng.uniform(50.0, 200.0)
    )
    segment_speed = float(rng.uniform(5.0, 40.0))
    output.loc[start:end, "bearing_change_rad"] = [
        normalize_bearing_delta(math.pi + rng.normal(0.0, 0.02))
        for _ in range(end - start + 1)
    ]
    output.loc[start:end, "distance_m"] = segment_distance
    output.loc[start:end, "elapsed_time_s"] = (
        segment_distance / (segment_speed / 3.6)
    )
    return output, start, end, None

def inject_abnormal_speed(frame, source_observations, rng):
    output = frame.copy()
    max_length = min(4, len(output) - 1)
    length = int(rng.integers(1, max_length + 1))
    start = int(rng.integers(1, len(output) - length + 1))
    end = start + length - 1

    regime = int(rng.integers(1, 5))
    
    if regime == 1:
        # Moderate Speed Surge (60 - 140 km/h)
        target_speeds = rng.uniform(60.0, 140.0, size=length)
        elapsed = rng.uniform(31.0, 250.0, size=length)
    elif regime == 2:
        # High-Speed Burst (141 - 250 km/h)
        target_speeds = rng.uniform(141.0, 250.0, size=length)
        elapsed = rng.uniform(305.0, 900.0, size=length)
    elif regime == 3:
        # Fast Transit Shift (251 - 380 km/h, capped under 400 km/h)
        target_speeds = rng.uniform(251.0, 380.0, size=length)
        elapsed = rng.uniform(3605.0, 7200.0, size=length)
    else:
        # Relative Speed Multiplier (2.5x to 6.0x baseline, capped at 380 km/h)
        orig_speeds = np.divide(
            output.loc[start:end, "distance_m"].to_numpy(dtype=float),
            output.loc[start:end, "elapsed_time_s"].to_numpy(dtype=float),
            out=np.zeros(length),
            where=output.loc[start:end, "elapsed_time_s"].to_numpy(dtype=float) > 0
        ) * 3.6
        multipliers = rng.uniform(2.5, 6.0, size=length)
        target_speeds = np.clip(np.maximum(orig_speeds, 25.0) * multipliers, 60.0, 380.0)
        
        elapsed = np.zeros(length)
        for i in range(length):
            spd = target_speeds[i]
            if spd <= 200.0:
                elapsed[i] = rng.uniform(31.0, 300.0)
            elif spd <= 300.0:
                elapsed[i] = rng.uniform(305.0, 1500.0)
            else:
                elapsed[i] = rng.uniform(3605.0, 7200.0)

    output.loc[start:end, "elapsed_time_s"] = elapsed
    output.loc[start:end, "distance_m"] = target_speeds * elapsed / 3.6

    # Cap acceleration for the transition immediately following the anomaly
    if end + 1 < len(output):
        next_speed = output.loc[end + 1, "distance_m"] / output.loc[end + 1, "elapsed_time_s"] * 3.6
        speed_diff_m_s = abs(target_speeds[-1] - next_speed) / 3.6
        min_elapsed_for_accel = speed_diff_m_s / 45.0
        if output.loc[end + 1, "elapsed_time_s"] < min_elapsed_for_accel:
            output.loc[end + 1, "elapsed_time_s"] = min_elapsed_for_accel

    return output, start, end, None

def inject_initial_fix_jump(frame, source_observations, rng):
    output = frame.copy()
    observations_by_id = source_observations.set_index("observation_id")
    true_observation = observations_by_id.loc[
        output.iloc[1]["observation_id2"]
    ]
    true_lat = float(true_observation["lat"])
    true_lon = float(true_observation["lon"])

    correction_speed = rng.uniform(40.0, 90.0)
    correction_elapsed = rng.uniform(30.0, 60.0)
    correction_distance = correction_speed * correction_elapsed / 3.6
    jump_bearing = rng.uniform(-math.pi, math.pi)
    stale_p1_lat, stale_p1_lon = destination_point(
        true_lat,
        true_lon,
        correction_distance,
        jump_bearing,
    )
    stale_speed = rng.uniform(0.0, 3.0)
    stale_elapsed = rng.uniform(30.0, 90.0)
    stale_distance = stale_speed * stale_elapsed / 3.6
    stale_p0_lat, stale_p0_lon = destination_point(
        stale_p1_lat,
        stale_p1_lon,
        stale_distance,
        rng.uniform(-math.pi, math.pi),
    )

    output.loc[0, "distance_m"] = haversine_distance(
        stale_p0_lat,
        stale_p0_lon,
        stale_p1_lat,
        stale_p1_lon,
    )
    output.loc[0, "elapsed_time_s"] = stale_elapsed
    output.loc[0, "bearing_change_rad"] = 0.0
    output.loc[1, "distance_m"] = haversine_distance(
        stale_p1_lat,
        stale_p1_lon,
        true_lat,
        true_lon,
    )
    output.loc[1, "elapsed_time_s"] = correction_elapsed
    output.loc[1, "bearing_change_rad"] = 0.0

    initial_offset = {
        "p0_lat": stale_p0_lat,
        "p0_lon": stale_p0_lon,
        "p1_lat": stale_p1_lat,
        "p1_lon": stale_p1_lon,
        "p2_lat": true_lat,
        "p2_lon": true_lon,
    }
    return output, 0, 1, initial_offset

def inject_coordinate_jitter(frame, source_observations, rng):
    output = frame.copy()
    max_length = min(3, len(output) - 1)
    length = int(rng.integers(1, max_length + 1))
    start = int(rng.integers(1, len(output) - length + 1))
    end = start + length - 1

    observations_by_id = source_observations.set_index("observation_id")
    coords = []
    first_obs_id = output.iloc[0]["observation_id1"]
    coords.append((float(observations_by_id.loc[first_obs_id]["lat"]), 
                   float(observations_by_id.loc[first_obs_id]["lon"])))
    for _, row in output.iterrows():
        obs_id2 = row["observation_id2"]
        coords.append((float(observations_by_id.loc[obs_id2]["lat"]), 
                       float(observations_by_id.loc[obs_id2]["lon"])))
    
    for i in range(start, end + 1):
        lat, lon = coords[i]
        
        # Ensure we don't violate max speed rules by capping distance based on available time
        t_in = output.loc[i - 1, "elapsed_time_s"] if i - 1 >= 0 else 86400.0
        t_out = output.loc[i, "elapsed_time_s"] if i < len(output) else 86400.0
        t_min = min(t_in, t_out)
        
        # Max safe speed for ANY time gap is < 30 km/h (8.33 m/s)
        # We cap jitter at 8.0 m/s * t_min to safely pass the deterministic rules
        max_safe_jitter = t_min * 8.0
        
        if max_safe_jitter < 10.0:
            jitter_dist = rng.uniform(1.0, max(2.0, max_safe_jitter))
        else:
            jitter_dist = rng.uniform(min(50.0, max_safe_jitter * 0.2), min(1000.0, max_safe_jitter))
            
        jitter_bearing = rng.uniform(-math.pi, math.pi)
        coords[i] = destination_point(lat, lon, jitter_dist, jitter_bearing)
        
    previous_bearing = None
    for i in range(len(output)):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i+1]
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        bearing = compute_bearing(lat1, lon1, lat2, lon2) if dist > 1e-3 else None
        
        output.loc[i, "distance_m"] = dist
        if i == 0:
            output.loc[i, "bearing_change_rad"] = 0.0
        else:
            if previous_bearing is not None and bearing is not None:
                output.loc[i, "bearing_change_rad"] = normalize_bearing_delta(bearing - previous_bearing)
            else:
                output.loc[i, "bearing_change_rad"] = 0.0
                
        if bearing is not None:
            previous_bearing = bearing

    return output, start, end, None

def inject_timestamp_jitter(frame, source_observations, rng):
    output = frame.copy()
    idx = int(rng.integers(0, len(output) - 1))
    t1 = output.loc[idx, "elapsed_time_s"]
    t2 = output.loc[idx + 1, "elapsed_time_s"]
    d1 = output.loc[idx, "distance_m"]
    d2 = output.loc[idx + 1, "distance_m"]
    
    t_total = t1 + t2
    
    def min_valid_time(d):
        if d <= 250.0: return max(0.1, d / 8.33)
        elif d <= 16650.0: return max(30.1, d / 55.5)
        elif d <= 299880.0: return max(300.1, d / 83.3)
        else: return max(3600.1, d / 250.0)
        
    mt1 = min_valid_time(d1)
    mt2 = min_valid_time(d2)
    
    if mt1 + mt2 > t_total:
        raise ValueError("Cannot jitter timestamp without violating deterministic rules")
        
    valid_min = max(2.0, mt1)
    valid_max = min(t_total - 2.0, t_total - mt2)
    
    if valid_max <= valid_min + 1.0:
        raise ValueError("Valid time range too narrow")
        
    margin = min(60.0, (valid_max - valid_min) * 0.1)
    if rng.random() < 0.5:
        new_t1 = rng.uniform(valid_min, valid_min + margin)
    else:
        new_t1 = rng.uniform(valid_max - margin, valid_max)
    
    new_t1 = max(valid_min, min(new_t1, valid_max))
    new_t2 = t_total - new_t1
    
    output.loc[idx, "elapsed_time_s"] = new_t1
    output.loc[idx + 1, "elapsed_time_s"] = new_t2
    
    return output, idx, idx + 1, None

PROFILE_INJECTORS = {
    "back_and_forth": inject_back_and_forth,
    "abnormal_speed": inject_abnormal_speed,
    "initial_fix_jump": inject_initial_fix_jump,
    "coordinate_jitter": inject_coordinate_jitter,
    "timestamp_jitter": inject_timestamp_jitter,
}


def recompute_dependent_motion(frame):
    output = frame.copy()
    elapsed = output["elapsed_time_s"].to_numpy(dtype=float)
    distance = output["distance_m"].to_numpy(dtype=float)
    speed_kmh = np.divide(distance, elapsed, out=np.zeros_like(distance), where=elapsed > 0) * 3.6
    speed_m_s = speed_kmh / 3.6
    previous_speed = np.concatenate(([np.nan], speed_m_s[:-1]))
    acceleration_valid = np.isfinite(previous_speed) & (elapsed > 0)
    acceleration = np.divide(
        speed_m_s - previous_speed,
        elapsed,
        out=np.zeros_like(speed_m_s),
        where=acceleration_valid,
    )
    output["speed_kmh"] = speed_kmh
    output["acceleration_m_s2"] = acceleration
    output["acceleration_valid"] = acceleration_valid
    moving = distance > 1e-3
    bearing_valid = moving & np.concatenate(([False], moving[:-1]))
    output["bearing_change_valid"] = bearing_valid
    output.loc[~bearing_valid, "bearing_change_rad"] = 0.0
    return output


def parse_start_timestamp(value):
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.Timestamp("2000-01-01 00:00:00")
    return timestamp


def retain_anomalous_session(
    observation_rows,
    transition_rows,
    metadata_row,
    max_inactivity_seconds,
):
    start, end = anomalous_session_range(
        [row["elapsed_time_s"] for row in transition_rows],
        metadata_row["anomaly_start_order"],
        metadata_row["anomaly_end_order"],
        max_inactivity_seconds,
    )
    selected_transitions = [
        dict(row) for row in transition_rows[start:end]
    ]
    selected_transitions[0]["acceleration_m_s2"] = 0.0
    selected_transitions[0]["acceleration_valid"] = False
    selected_transitions[0]["bearing_change_rad"] = 0.0
    selected_transitions[0]["bearing_change_valid"] = False
    selected_observation_ids = {
        selected_transitions[0]["observation_id1"],
        *[row["observation_id2"] for row in selected_transitions],
    }
    selected_observations = [
        row for row in observation_rows
        if row["observation_id"] in selected_observation_ids
    ]
    selected_metadata = {
        **metadata_row,
        "anomaly_start_order": metadata_row["anomaly_start_order"] - start,
        "anomaly_end_order": metadata_row["anomaly_end_order"] - start,
    }
    return selected_observations, selected_transitions, selected_metadata


def build_synthetic_trajectory(
    source_dataset,
    source_trajectory_id,
    source_transitions,
    source_observations,
    profile,
    trajectory_id,
    first_transition_id,
    seed,
    max_inactivity_seconds=DEFAULT_MAX_INACTIVITY_SECONDS,
):
    rng = np.random.default_rng(seed)
    mutated, anomaly_start, anomaly_end, initial_offset = PROFILE_INJECTORS[profile](source_transitions, source_observations, rng)
    mutated = recompute_dependent_motion(mutated)

    observations_by_id = source_observations.set_index("observation_id")
    first_source = source_transitions.iloc[0]
    first_point = observations_by_id.loc[first_source["observation_id1"]]
    second_point = observations_by_id.loc[first_source["observation_id2"]]

    if initial_offset:
        current_lat = initial_offset["p0_lat"]
        current_lon = initial_offset["p0_lon"]
        current_bearing = compute_bearing(
            current_lat, current_lon, initial_offset["p1_lat"], initial_offset["p1_lon"]
        )
    else:
        current_lat = float(first_point["lat"])
        current_lon = float(first_point["lon"])
        current_bearing = compute_bearing(
            current_lat,
            current_lon,
            float(second_point["lat"]),
            float(second_point["lon"]),
        )

    current_timestamp = parse_start_timestamp(first_point["timestamp"])
    user_id = f"synthetic:{source_dataset}:{profile}"
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
    previous_segment_bearing = None

    for order, row in mutated.reset_index(drop=True).iterrows():
        if initial_offset and order == 0:
            next_lat = initial_offset["p1_lat"]
            next_lon = initial_offset["p1_lon"]
        elif initial_offset and order == 1:
            next_lat = initial_offset["p2_lat"]
            next_lon = initial_offset["p2_lon"]
            current_bearing = compute_bearing(current_lat, current_lon, next_lat, next_lon)
        else:
            if order > 0:
                current_bearing = normalize_bearing_delta(
                    current_bearing + float(row["bearing_change_rad"])
                )
            next_lat, next_lon = destination_point(
                current_lat,
                current_lon,
                float(row["distance_m"]),
                current_bearing,
            )

        has_motion = float(row["distance_m"]) > 1e-3
        segment_bearing = (
            compute_bearing(current_lat, current_lon, next_lat, next_lon)
            if has_motion
            else None
        )
        bearing_change_valid = (
            segment_bearing is not None and previous_segment_bearing is not None
        )
        bearing_change_rad = (
            normalize_bearing_delta(segment_bearing - previous_segment_bearing)
            if bearing_change_valid
            else 0.0
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
                "acceleration_valid": bool(row["acceleration_valid"]),
                "bearing_change_rad": float(bearing_change_rad),
                "bearing_change_valid": bearing_change_valid,
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
        previous_segment_bearing = segment_bearing

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
    observation_rows, transition_rows, metadata_row = retain_anomalous_session(
        observation_rows,
        transition_rows,
        metadata_row,
        max_inactivity_seconds,
    )
    invalid_reasons = sorted(
        {
            row["plausibility_reason"] or "deterministic plausibility policy"
            for row in transition_rows
            if row["transition_plausibility"] != 1
        }
    )
    if invalid_reasons:
        raise ValueError(
            "generated trajectory violates deterministic plausibility rules: "
            + "; ".join(invalid_reasons)
        )
    mapping_rows = [
        {
            "dataset": "synthetic",
            "trajectory_id": trajectory_id,
            "transition_order": order,
            "transition_id": row["transition_id"],
        }
        for order, row in enumerate(transition_rows)
    ]
    trajectory_row = {
        "dataset": "synthetic",
        "trajectory_id": trajectory_id,
        "user_id": user_id,
        "n_transitions": len(transition_rows),
        "start_observation_id": observation_rows[0]["observation_id"],
        "end_observation_id": observation_rows[-1]["observation_id"],
    }
    return observation_rows, transition_rows, trajectory_row, mapping_rows, metadata_row


def generate_synthetic_dataset(
    processed_root,
    output_dir,
    source_datasets=("inat", "gowalla"),
    trajectories_per_profile=50,
    min_transitions=4,
    seed=42,
    profiles=PROFILE_NAMES,
    max_inactivity_seconds=DEFAULT_MAX_INACTIVITY_SECONDS,
    plausible_labels_path=DEFAULT_LABEL_PATH,
):
    rng = np.random.default_rng(seed)
    source_tables = {}
    for source_dataset in source_datasets:
        source_trajectories, mapping, transitions, observations = (
            load_clean_source_trajectories(
                processed_root=processed_root,
                source_dataset=source_dataset,
                min_transitions=min_transitions,
                count=trajectories_per_profile,
                rng=rng,
                max_inactivity_seconds=max_inactivity_seconds,
                plausible_labels_path=plausible_labels_path,
            )
        )
        source_tables[source_dataset] = {
            "trajectories": source_trajectories,
            "mapping": mapping,
            "transitions": transitions.set_index("transition_id"),
            "observations": observations.set_index(
                "observation_id",
                drop=False,
            ),
        }

    all_observations = []
    all_transitions = []
    all_trajectories = []
    all_mappings = []
    all_metadata = []
    next_transition_id = 0

    next_trajectory_id = 0
    for source_dataset in source_datasets:
        tables = source_tables[source_dataset]
        for profile in profiles:
            if profile not in PROFILE_INJECTORS:
                raise ValueError(f"Unknown anomaly profile: {profile}")
            for source in tables["trajectories"].itertuples(index=False):
                source_mapping = tables["mapping"].loc[
                    tables["mapping"]["trajectory_id"] == source.trajectory_id
                ]
                source_transition_frame = tables["transitions"].loc[
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
                source_observation_frame = tables["observations"].loc[
                    source_observation_ids
                ].reset_index(drop=True)
                last_error = None
                generated = None
                for attempt in range(50):
                    trajectory_seed = seed + ((next_trajectory_id + 1) * 1000) + attempt
                    try:
                        generated = build_synthetic_trajectory(
                            source_dataset=source_dataset,
                            source_trajectory_id=source.trajectory_id,
                            source_transitions=source_transition_frame,
                            source_observations=source_observation_frame,
                            profile=profile,
                            trajectory_id=next_trajectory_id,
                            first_transition_id=next_transition_id,
                            seed=trajectory_seed,
                            max_inactivity_seconds=max_inactivity_seconds,
                        )
                        break
                    except ValueError as exc:
                        last_error = exc
                if generated is None:
                    raise ValueError(
                        f"Failed to generate {profile} from {source_dataset} "
                        f"trajectory {source.trajectory_id}: {last_error}"
                    ) from last_error
                (
                    observation_rows,
                    transition_rows,
                    trajectory_row,
                    mapping_rows,
                    metadata_row,
                ) = generated
                all_observations.extend(observation_rows)
                all_transitions.extend(transition_rows)
                all_trajectories.append(trajectory_row)
                all_mappings.extend(mapping_rows)
                all_metadata.append(metadata_row)
                next_transition_id = max(
                    row["transition_id"] for row in transition_rows
                ) + 1
                next_trajectory_id += 1

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
        "source_datasets": list(source_datasets),
        "trajectories_per_profile_per_dataset": trajectories_per_profile,
        "min_source_transitions": min_transitions,
        "seed": seed,
        "profiles": list(profiles),
        "max_inactivity_seconds": max_inactivity_seconds,
        "plausible_labels": str(plausible_labels_path) if plausible_labels_path else None,
        "deterministic_plausibility_required": True,
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
    parser.add_argument(
        "--source-datasets",
        nargs="+",
        default=["inat", "gowalla"],
    )
    parser.add_argument("--trajectories-per-profile", type=int, default=50)
    parser.add_argument("--min-transitions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-inactivity-hours",
        type=float,
        default=DEFAULT_MAX_INACTIVITY_SECONDS / 3600,
    )
    parser.add_argument("--profiles", nargs="+", choices=PROFILE_NAMES, default=list(PROFILE_NAMES))
    parser.add_argument("--plausible-labels", default=str(DEFAULT_LABEL_PATH))
    args = parser.parse_args()

    config = generate_synthetic_dataset(
        processed_root=args.processed_root,
        output_dir=args.output_dir,
        source_datasets=args.source_datasets,
        trajectories_per_profile=args.trajectories_per_profile,
        min_transitions=args.min_transitions,
        seed=args.seed,
        profiles=args.profiles,
        max_inactivity_seconds=(
            args.max_inactivity_hours * 3600
            if args.max_inactivity_hours > 0
            else None
        ),
        plausible_labels_path=args.plausible_labels,
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
