import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dataset_preprocessing.features.cleaning import select_valid_trajectory_ids
except ModuleNotFoundError:
    from dataset_preprocessing.cleaning import select_valid_trajectory_ids


METRICS = {
    "speed_kmh": "speed",
    "acceleration_m_s2": "acceleration",
    "distance_m": "distance",
    "elapsed_time_s": "elapsed_time",
    "bearing_change_rad": "bearing_change",
}

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
DEFAULT_DATASETS = ["inat", "gowalla", "diverse"]


def numeric_summary(series):
    values = series.dropna().astype(float)
    if values.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": {},
        }

    quantiles = values.quantile(QUANTILES)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": {f"p{int(q * 100):02d}": float(quantiles.loc[q]) for q in QUANTILES},
    }


def histogram(series, bins):
    values = series.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return pd.DataFrame(columns=["bin_left", "bin_right", "count"])

    counts, edges = np.histogram(values, bins=bins)
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts.astype(int),
        }
    )


def read_dataset(processed_dir, dataset):
    base = Path(processed_dir) / dataset
    transitions = pd.read_parquet(
        base / "transitions.parquet",
        columns=["transition_id", *METRICS.keys(), "transition_plausibility"],
    )
    trajectories = pd.read_parquet(
        base / "trajectories.parquet",
        columns=["trajectory_id", "n_transitions"],
    )
    trajectory_transitions = pd.read_parquet(
        base / "trajectory_transitions.parquet",
        columns=["trajectory_id", "transition_order", "transition_id"],
    )
    validity = trajectory_transitions.merge(
        transitions[["transition_id", "elapsed_time_s", "distance_m"]],
        on="transition_id",
        how="left",
        validate="one_to_one",
    ).sort_values(["trajectory_id", "transition_order"])
    if "acceleration_valid" not in transitions:
        transitions["acceleration_valid"] = False
    if "bearing_change_valid" not in transitions:
        transitions["bearing_change_valid"] = False
    derived_acceleration_valid = (
        validity["transition_order"].gt(0) & validity["elapsed_time_s"].gt(0)
    )
    previous_moving = validity.groupby("trajectory_id", sort=False)["distance_m"].shift().gt(1e-3)
    derived_bearing_valid = validity["distance_m"].gt(1e-3) & previous_moving
    validity["derived_acceleration_valid"] = derived_acceleration_valid
    validity["derived_bearing_valid"] = derived_bearing_valid
    transitions = transitions.drop(
        columns=["acceleration_valid", "bearing_change_valid"],
        errors="ignore",
    ).merge(
        validity[
            [
                "transition_id",
                "derived_acceleration_valid",
                "derived_bearing_valid",
            ]
        ],
        on="transition_id",
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "derived_acceleration_valid": "acceleration_valid",
            "derived_bearing_valid": "bearing_change_valid",
        }
    )
    return transitions, trajectories, trajectory_transitions


def clean_trajectory_ids(transitions, trajectories, trajectory_transitions, min_transitions):
    valid_ids, _ = select_valid_trajectory_ids(
        transitions,
        trajectory_transitions,
        trajectories,
        min_transitions,
    )
    return valid_ids


def subset_tables(transitions, trajectories, trajectory_transitions, valid_ids=None):
    if valid_ids is None:
        return transitions, trajectories, trajectory_transitions

    clean_trajectories = trajectories.loc[trajectories["trajectory_id"].isin(valid_ids)].copy()
    clean_trajectory_transitions = trajectory_transitions.loc[
        trajectory_transitions["trajectory_id"].isin(valid_ids)
    ].copy()
    clean_transition_ids = clean_trajectory_transitions["transition_id"]
    clean_transitions = transitions.loc[transitions["transition_id"].isin(clean_transition_ids)].copy()
    return clean_transitions, clean_trajectories, clean_trajectory_transitions


def describe_subset(dataset, subset_name, transitions, trajectories, trajectory_transitions, output_dir, bins):
    merged = trajectory_transitions.merge(
        transitions[["transition_id", "elapsed_time_s"]],
        on="transition_id",
        how="left",
    )
    avg_elapsed_series = merged.groupby("trajectory_id", sort=False)["elapsed_time_s"].mean()

    subset_summary = {
        "dataset": dataset,
        "subset": subset_name,
        "n_transitions": int(len(transitions)),
        "n_trajectories": int(len(trajectories)),
        "transition_metrics": {},
        "trajectory_n_transitions": numeric_summary(trajectories["n_transitions"]),
        "avg_elapsed_time": numeric_summary(avg_elapsed_series),
        "data_quality": {
            "undefined_acceleration": int((~transitions["acceleration_valid"]).sum()),
            "undefined_bearing_change": int((~transitions["bearing_change_valid"]).sum()),
            "stationary_nonzero_bearing_change": int(
                (
                    transitions["distance_m"].le(1e-3)
                    & transitions["bearing_change_rad"].abs().gt(1e-12)
                ).sum()
            ),
            "zero_elapsed_time": int(transitions["elapsed_time_s"].le(0).sum()),
        },
    }

    histogram_dir = Path(output_dir) / "histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)

    for column, metric_name in METRICS.items():
        metric_values = transitions[column]
        if column == "acceleration_m_s2":
            metric_values = metric_values.loc[transitions["acceleration_valid"]]
        elif column == "bearing_change_rad":
            metric_values = metric_values.loc[transitions["bearing_change_valid"]]
        subset_summary["transition_metrics"][metric_name] = numeric_summary(metric_values)
        histogram(metric_values, bins).to_csv(
            histogram_dir / f"{dataset}_{subset_name}_{metric_name}.csv",
            index=False,
        )

    histogram(trajectories["n_transitions"], bins).to_csv(
        histogram_dir / f"{dataset}_{subset_name}_trajectory_n_transitions.csv",
        index=False,
    )
    histogram(avg_elapsed_series, bins).to_csv(
        histogram_dir / f"{dataset}_{subset_name}_avg_elapsed_time.csv",
        index=False,
    )

    return subset_summary


def format_number(value):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.4f}"


def text_report(summary):
    lines = []
    for dataset, dataset_summary in summary["datasets"].items():
        lines.append(f"{dataset}")
        for subset_name, subset_summary in dataset_summary.items():
            lines.append(f"  {subset_name}")
            lines.append(f"    trajectories: {subset_summary['n_trajectories']:,}")
            lines.append(f"    transitions: {subset_summary['n_transitions']:,}")
            traj_stats = subset_summary["trajectory_n_transitions"]
            lines.append(
                "    trajectory transitions: "
                f"mean={format_number(traj_stats['mean'])}, "
                f"median={format_number(traj_stats['quantiles'].get('p50'))}, "
                f"p95={format_number(traj_stats['quantiles'].get('p95'))}, "
                f"max={format_number(traj_stats['max'])}"
            )
            for metric_name, metric_stats in subset_summary["transition_metrics"].items():
                lines.append(
                    f"    {metric_name}: "
                    f"mean={format_number(metric_stats['mean'])}, "
                    f"median={format_number(metric_stats['quantiles'].get('p50'))}, "
                    f"p95={format_number(metric_stats['quantiles'].get('p95'))}, "
                    f"p99={format_number(metric_stats['quantiles'].get('p99'))}, "
                    f"max={format_number(metric_stats['max'])}"
                )
        lines.append("")
    return "\n".join(lines)


def build_stats(processed_dir, datasets, output_dir, min_transitions, bins):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "processed_dir": str(processed_dir),
        "min_transitions_for_clean_real": min_transitions,
        "cleaning_policy": "trajectory has at least min_transitions and all transitions are rule-plausible",
        "datasets": {},
    }

    for dataset in datasets:
        transitions, trajectories, trajectory_transitions = read_dataset(processed_dir, dataset)
        valid_ids = clean_trajectory_ids(transitions, trajectories, trajectory_transitions, min_transitions)

        all_transitions, all_trajectories, all_traj_trans = subset_tables(transitions, trajectories, trajectory_transitions)
        clean_transitions, clean_trajectories, clean_traj_trans = subset_tables(
            transitions,
            trajectories,
            trajectory_transitions,
            valid_ids,
        )

        summary["datasets"][dataset] = {
            "all": describe_subset(dataset, "all", all_transitions, all_trajectories, all_traj_trans, output_dir, bins),
            "clean_real": describe_subset(dataset, "clean_real", clean_transitions, clean_trajectories, clean_traj_trans, output_dir, bins),
        }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report = text_report(summary)
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(report)

    return summary, report


def main():
    parser = argparse.ArgumentParser(description="Extract stats from canonical Parquet trajectory datasets.")
    parser.add_argument("--processed-dir", default="data/processed_parquet")
    parser.add_argument("--output-dir", default="artifacts/stats/processed_parquet")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--min-transitions", type=int, default=2)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()

    _, report = build_stats(
        processed_dir=args.processed_dir,
        datasets=args.datasets,
        output_dir=args.output_dir,
        min_transitions=args.min_transitions,
        bins=args.bins,
    )
    print(report)


if __name__ == "__main__":
    main()
