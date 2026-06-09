import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from Dataset_Preprocessing.cleaning import select_valid_trajectory_ids
except ModuleNotFoundError:
    from cleaning import select_valid_trajectory_ids


METRICS = {
    "speed_kmh": "speed",
    "acceleration_m_s2": "acceleration",
    "distance_m": "distance",
    "elapsed_time_s": "elapsed_time",
    "bearing_change_rad": "bearing_change",
}

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
DEFAULT_DATASETS = ["inat", "gowalla"]


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
        columns=["trajectory_id", "transition_id"],
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
        return transitions, trajectories

    clean_trajectories = trajectories.loc[trajectories["trajectory_id"].isin(valid_ids)].copy()
    clean_transition_ids = trajectory_transitions.loc[
        trajectory_transitions["trajectory_id"].isin(valid_ids),
        "transition_id",
    ]
    clean_transitions = transitions.loc[transitions["transition_id"].isin(clean_transition_ids)].copy()
    return clean_transitions, clean_trajectories


def describe_subset(dataset, subset_name, transitions, trajectories, output_dir, bins):
    subset_summary = {
        "dataset": dataset,
        "subset": subset_name,
        "n_transitions": int(len(transitions)),
        "n_trajectories": int(len(trajectories)),
        "transition_metrics": {},
        "trajectory_n_transitions": numeric_summary(trajectories["n_transitions"]),
    }

    histogram_dir = Path(output_dir) / "histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)

    for column, metric_name in METRICS.items():
        subset_summary["transition_metrics"][metric_name] = numeric_summary(transitions[column])
        histogram(transitions[column], bins).to_csv(
            histogram_dir / f"{dataset}_{subset_name}_{metric_name}.csv",
            index=False,
        )

    histogram(trajectories["n_transitions"], bins).to_csv(
        histogram_dir / f"{dataset}_{subset_name}_trajectory_n_transitions.csv",
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

        all_transitions, all_trajectories = subset_tables(transitions, trajectories, trajectory_transitions)
        clean_transitions, clean_trajectories = subset_tables(
            transitions,
            trajectories,
            trajectory_transitions,
            valid_ids,
        )

        summary["datasets"][dataset] = {
            "all": describe_subset(dataset, "all", all_transitions, all_trajectories, output_dir, bins),
            "clean_real": describe_subset(dataset, "clean_real", clean_transitions, clean_trajectories, output_dir, bins),
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
