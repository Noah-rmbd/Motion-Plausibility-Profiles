import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

try:
    from dataset_preprocessing.features.build_motion_features import select_valid_real_trajectories, read_real_dataset, assign_splits
    from dataset_preprocessing.features.splitting import DEFAULT_CALIBRATION_FRACTION, DEFAULT_SPLIT_SEED
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from dataset_preprocessing.features.build_motion_features import select_valid_real_trajectories, read_real_dataset, assign_splits
    from dataset_preprocessing.features.splitting import DEFAULT_CALIBRATION_FRACTION, DEFAULT_SPLIT_SEED

DEFAULT_DATASETS = ["inat", "gowalla"]

def log(message):
    print(f"[build_global_trajectory_features] {message}", flush=True)

def point_line_distance(point, start, end):
    point = np.array(point)
    start = np.array(start)
    end = np.array(end)
    if np.all(start == end):
        return np.linalg.norm(point - start)
    num = np.abs((end[0] - start[0]) * (start[1] - point[1]) - (end[1] - start[1]) * (start[0] - point[0]))
    den = np.linalg.norm(end - start)
    return num / den

def rdp_count(coords, epsilon):
    n = len(coords)
    if n <= 2:
        return n
    start = coords[0]
    end = coords[-1]
    line_vec = end - start
    line_len_sq = np.dot(line_vec, line_vec)
    vecs = coords[1:-1] - start
    if line_len_sq == 0:
        dists = np.linalg.norm(vecs, axis=1)
    else:
        cross_prod = np.abs(vecs[:, 0] * line_vec[1] - vecs[:, 1] * line_vec[0])
        dists = cross_prod / np.sqrt(line_len_sq)
        
    if len(dists) == 0:
        return 2
        
    max_idx = np.argmax(dists)
    max_d = dists[max_idx]
    if max_d >= epsilon:
        idx = max_idx + 1
        return rdp_count(coords[:idx+1], epsilon) + rdp_count(coords[idx:], epsilon) - 1
    else:
        return 2

def compute_gini(array):
    if len(array) == 0:
        return 0.0
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    s = np.sum(array)
    if s > 0:
        return ((np.sum((2 * index - n  - 1) * array)) / (n * s))
    return 0.0

def compute_entropy(counts):
    total = np.sum(counts)
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def process_trajectory(group):
    group = group.sort_values("transition_order")
    trav_dist = group["distance_m"].sum()
    
    n_trans = len(group)
    lats = np.empty(n_trans + 1, dtype=np.float64)
    lons = np.empty(n_trans + 1, dtype=np.float64)
    lats[0] = group["lat1"].iloc[0]
    lats[1:] = group["lat2"].values
    lons[0] = group["lon1"].iloc[0]
    lons[1:] = group["lon2"].values
    
    mean_lat = np.mean(lats)
    lat_scale = 111320.0
    lon_scale = 111320.0 * np.cos(np.radians(mean_lat))
    
    x = lons * lon_scale
    y = lats * lat_scale
    coords = np.column_stack((x, y))
    
    straight_line_dist = np.linalg.norm(coords[-1] - coords[0]) if len(coords) > 1 else 0.0
    efficiency = straight_line_dist / trav_dist if trav_dist > 0 else 0.0
    detour = 1.0 / efficiency if efficiency > 0 else 0.0
    
    max_pairwise = np.max(pdist(coords)) if len(coords) > 1 else 0.0
    displacement_ratio = straight_line_dist / max_pairwise if max_pairwise > 0 else 0.0
    
    valid_bearings = group[group["bearing_change_valid"]]["bearing_change_rad"].dropna()
    bearing_deg = np.degrees(valid_bearings)
    reversals = np.sum(np.abs(bearing_deg) > 150)
    reversal_freq = reversals / n_trans if n_trans > 0 else 0.0
    
    u_turn = np.sum(np.abs(bearing_deg) > 150)
    left = np.sum((bearing_deg >= -150) & (bearing_deg < -30))
    right = np.sum((bearing_deg > 30) & (bearing_deg <= 150))
    straight = np.sum((bearing_deg >= -30) & (bearing_deg <= 30))
    entropy = compute_entropy(np.array([u_turn, left, right, straight]))
    
    sum_bearing_change = np.sum(np.abs(bearing_deg))
    turn_rate = sum_bearing_change / trav_dist if trav_dist > 0 else 0.0
    mean_turn = np.mean(np.abs(bearing_deg)) if len(bearing_deg) > 0 else 0.0
    
    area = 0.0
    if len(coords) >= 3:
        try:
            hull = ConvexHull(coords)
            area = hull.volume
        except Exception:
            area = 0.0
            
    centroid = np.mean(coords, axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1))) if len(coords) > 0 else 0.0
    
    stops = group[group["speed_kmh"] < 1.0]
    num_stops = len(stops)
    dur_stops = stops["elapsed_time_s"].sum()
    stop_ratio = dur_stops / group["elapsed_time_s"].sum() if group["elapsed_time_s"].sum() > 0 else 0.0
    
    median_speed = group["speed_kmh"].median()
    var_speed = group["speed_kmh"].var() if n_trans > 1 else 0.0
    
    num_compressed = rdp_count(coords, 10.0)
    comp_ratio = (num_compressed - 1) / n_trans if n_trans > 0 else 0.0
    
    gini = compute_gini(group["elapsed_time_s"].values)
    
    return pd.Series({
        "efficiency": efficiency,
        "detour": detour,
        "reversal_freq": reversal_freq,
        "turning_entropy": entropy,
        "turn_rate": turn_rate,
        "mean_turn_angle": mean_turn,
        "convex_hull_area": area,
        "radius_of_gyration": rg,
        "num_stops": num_stops,
        "duration_stops": dur_stops,
        "stop_ratio": stop_ratio,
        "median_speed": median_speed,
        "var_speed": var_speed,
        "compression_ratio": comp_ratio,
        "temporal_sparsity": gini,
        "displacement_ratio": displacement_ratio,
    })

def build_global_features_for_dataset(dataset, processed_dir, output_dir, min_transitions=2):
    log(f"Building global trajectory features for {dataset}")
    base = Path(processed_dir) / dataset
    
    transitions, trajectory_transitions, trajectories = read_real_dataset(processed_dir, dataset)
    full_transitions = pd.read_parquet(base / "transitions.parquet", columns=["transition_id", "observation_id1", "observation_id2"])
    transitions = transitions.merge(full_transitions, on="transition_id", how="left")
    
    observations = pd.read_parquet(base / "observations.parquet", columns=["observation_id", "lat", "lon"])
    
    valid_ids, exclusions = select_valid_real_trajectories(
        transitions, trajectory_transitions, trajectories, min_transitions
    )
    
    valid_mapping = trajectory_transitions.loc[trajectory_transitions["trajectory_id"].isin(valid_ids)]
    valid_transitions = valid_mapping.merge(transitions.drop(columns=["plausibility_reason"]), on="transition_id", how="left")
    
    # Pre-join lat and lon for observation_id1 and observation_id2 to eliminate in-loop lookups
    obs_coords = observations[["observation_id", "lat", "lon"]]
    valid_transitions = valid_transitions.merge(
        obs_coords.rename(columns={"observation_id": "observation_id1", "lat": "lat1", "lon": "lon1"}),
        on="observation_id1",
        how="left"
    ).merge(
        obs_coords.rename(columns={"observation_id": "observation_id2", "lat": "lat2", "lon": "lon2"}),
        on="observation_id2",
        how="left"
    )
    
    log(f"{dataset}: computing features for {len(valid_ids):,} trajectories")
    features_df = valid_transitions.groupby("trajectory_id").apply(process_trajectory).reset_index()
    
    final_trajectories = trajectories[trajectories["trajectory_id"].isin(valid_ids)].copy()
    
    if dataset != "synthetic":
        final_trajectories["split"] = assign_splits(dataset, final_trajectories, DEFAULT_CALIBRATION_FRACTION, DEFAULT_SPLIT_SEED)
        final_trajectories["use_for_training"] = final_trajectories["split"] == "fit"
        final_trajectories["use_for_calibration"] = final_trajectories["split"] == "calibration"
    else:
        final_trajectories["split"] = "benchmark"
        final_trajectories["use_for_training"] = False
        final_trajectories["use_for_calibration"] = False
        
    final_trajectories = final_trajectories.merge(features_df, on="trajectory_id", how="left")
    
    out_dir = Path(output_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "trajectory_features.parquet"
    final_trajectories.to_parquet(out_path)
    log(f"{dataset}: saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", "--processed-root", dest="processed_dir", type=str, default="data/processed_parquet")
    parser.add_argument("--output-dir", type=str, default="data/features/global_v1")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    args = parser.parse_args()
    
    for ds in args.datasets:
        build_global_features_for_dataset(ds, args.processed_dir, args.output_dir)

if __name__ == "__main__":
    main()
