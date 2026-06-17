from pathlib import Path

import numpy as np
import pandas as pd

from modeling.score_aggregation import (
    AGGREGATION_COLUMNS,
    TRANSITION_SCORE_MODES,
    aggregate_transition_scores,
    feature_error_columns,
    select_trajectory_scores,
    transition_error_scores,
)


TRANSITION_METRICS = {
    "speed": ("speed_kmh", None),
    "acceleration": ("acceleration_m_s2", "acceleration_valid"),
    "distance": ("distance_m", None),
    "elapsed_time": ("elapsed_time_s", None),
    "bearing_change": ("bearing_change_rad", "bearing_change_valid"),
}
TRAJECTORY_METRICS = {
    "trajectory_n_transitions",
    "max_speed",
    "total_distance",
    "total_elapsed_time",
    "model_score",
}
TRAJECTORY_POPULATIONS = {"model_flagged", "model_accepted"}
TRANSITION_POPULATIONS = {"transition_flagged", "transition_accepted"}


def numeric_summary(values):
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": {},
        }
    quantiles = series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
        "quantiles": {
            f"p{int(value * 100):02d}": float(quantiles.loc[value])
            for value in quantiles.index
        },
    }


def shared_histograms(selected, comparison, bins=80):
    selected = pd.Series(selected, dtype=float).dropna().to_numpy()
    comparison = pd.Series(comparison, dtype=float).dropna().to_numpy()
    if not len(comparison):
        return [], []
    lower, upper = np.quantile(comparison, [0.005, 0.995])
    if not np.isfinite(lower) or not np.isfinite(upper):
        return [], []
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    edges = np.linspace(lower, upper, bins + 1)

    def build(values):
        clipped = np.clip(values, lower, upper)
        counts, _ = np.histogram(clipped, bins=edges)
        return [
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": int(count),
            }
            for left, right, count in zip(edges[:-1], edges[1:], counts)
        ]

    return build(selected), build(comparison)


def trajectory_physical_metrics(processed_root, dataset, trajectory_ids):
    base = Path(processed_root) / dataset
    mapping = pd.read_parquet(
        base / "trajectory_transitions.parquet",
        columns=["trajectory_id", "transition_id"],
        filters=[("trajectory_id", "in", [int(value) for value in trajectory_ids])],
    )
    transitions = pd.read_parquet(
        base / "transitions.parquet",
        columns=[
            "transition_id",
            "speed_kmh",
            "distance_m",
            "elapsed_time_s",
        ],
        filters=[
            (
                "transition_id",
                "in",
                mapping["transition_id"].astype(int).tolist(),
            )
        ],
    )
    return (
        mapping.merge(
            transitions,
            on="transition_id",
            how="left",
            validate="many_to_one",
        )
        .groupby("trajectory_id", sort=False)
        .agg(
            max_speed=("speed_kmh", "max"),
            total_distance=("distance_m", "sum"),
            total_elapsed_time=("elapsed_time_s", "sum"),
        )
        .reset_index()
    )


def population_values(
    metric,
    processed_root,
    dataset,
    trajectory_table,
    trajectory_ids,
    score_table,
):
    selected_ids = set(int(value) for value in trajectory_ids)
    if metric == "model_score":
        return score_table.loc[
            score_table["trajectory_id"].isin(selected_ids),
            "selected_score",
        ]
    if metric == "trajectory_n_transitions":
        return trajectory_table.loc[
            trajectory_table["trajectory_id"].isin(selected_ids),
            "n_transitions",
        ]
    if metric in {"max_speed", "total_distance", "total_elapsed_time"}:
        physical = trajectory_physical_metrics(
            processed_root,
            dataset,
            selected_ids,
        )
        return physical[metric]

    column, valid_column = TRANSITION_METRICS[metric]
    base = Path(processed_root) / dataset
    mapping = pd.read_parquet(
        base / "trajectory_transitions.parquet",
        columns=["trajectory_id", "transition_id"],
        filters=[("trajectory_id", "in", list(selected_ids))],
    )
    columns = ["transition_id", column]
    if valid_column:
        columns.append(valid_column)
    transitions = pd.read_parquet(
        base / "transitions.parquet",
        columns=columns,
        filters=[
            (
                "transition_id",
                "in",
                mapping["transition_id"].astype(int).tolist(),
            )
        ],
    )
    if valid_column:
        transitions = transitions.loc[transitions[valid_column]]
    return transitions[column]


def feature_error_summary(transition_scores, selected_ids):
    columns = feature_error_columns(transition_scores)
    if not columns:
        return []
    selected = transition_scores.loc[
        transition_scores["trajectory_id"].isin(selected_ids),
        columns,
    ].mean()
    comparison = transition_scores[columns].mean()
    rows = []
    for column in columns:
        selected_mean = float(selected[column])
        comparison_mean = float(comparison[column])
        rows.append(
            {
                "feature": column.removeprefix("error_"),
                "mean_error": selected_mean,
                "all_mean_error": comparison_mean,
                "lift": (
                    selected_mean / comparison_mean
                    if comparison_mean > 0
                    else None
                ),
            }
        )
    return sorted(rows, key=lambda row: row["mean_error"], reverse=True)


def transition_population_values(metric, processed_root, dataset, transition_table):
    if metric == "model_score":
        return transition_table["selected_score"]
    if metric in TRAJECTORY_METRICS:
        raise ValueError(
            f"{metric} is a trajectory-level metric and cannot be used for "
            "transition-level model populations"
        )
    if transition_table.empty:
        return pd.Series(dtype=float)

    column, valid_column = TRANSITION_METRICS[metric]
    base = Path(processed_root) / dataset
    columns = ["transition_id", column]
    if valid_column:
        columns.append(valid_column)
    transitions = pd.read_parquet(
        base / "transitions.parquet",
        columns=columns,
        filters=[
            (
                "transition_id",
                "in",
                transition_table["transition_id"].astype(int).tolist(),
            )
        ],
    )
    values = transition_table[["transition_id"]].merge(
        transitions,
        on="transition_id",
        how="left",
        validate="many_to_one",
    )
    if valid_column:
        values = values.loc[values[valid_column]]
    return values[column]


def feature_error_summary_by_transitions(selected_transition_scores, comparison_scores):
    columns = feature_error_columns(comparison_scores)
    if not columns or selected_transition_scores.empty:
        return []
    selected = selected_transition_scores[columns].mean()
    comparison = comparison_scores[columns].mean()
    rows = []
    for column in columns:
        selected_mean = float(selected[column])
        comparison_mean = float(comparison[column])
        rows.append(
            {
                "feature": column.removeprefix("error_"),
                "mean_error": selected_mean,
                "all_mean_error": comparison_mean,
                "lift": (
                    selected_mean / comparison_mean
                    if comparison_mean > 0
                    else None
                ),
            }
        )
    return sorted(rows, key=lambda row: row["mean_error"], reverse=True)


def build_model_population_stats(
    model_id,
    dataset,
    population,
    metric,
    anomaly_percentage,
    aggregation,
    transition_score_mode,
    include_first_transition,
    prediction_root="artifacts/predictions",
    feature_root="data/features",
    processed_root="data/processed_parquet",
    feature_set="motion_v2",
):
    if dataset not in {"inat", "gowalla"}:
        raise ValueError("Model population statistics require a real dataset")
    if population not in {*TRAJECTORY_POPULATIONS, *TRANSITION_POPULATIONS}:
        raise ValueError(f"Unknown model population: {population}")
    if metric not in {*TRANSITION_METRICS, *TRAJECTORY_METRICS}:
        raise ValueError(f"Unknown model statistics metric: {metric}")
    if aggregation not in AGGREGATION_COLUMNS:
        raise ValueError(f"Unknown trajectory aggregation: {aggregation}")
    if transition_score_mode not in TRANSITION_SCORE_MODES:
        raise ValueError(f"Unknown transition score mode: {transition_score_mode}")
    if not 0 < anomaly_percentage < 100:
        raise ValueError("Anomaly percentage must be between 0 and 100")

    transition_scores = pd.read_parquet(
        Path(prediction_root)
        / model_id
        / dataset
        / "transition_scores.parquet"
    )
    transition_scores = transition_scores.copy()
    transition_scores["selected_score"] = transition_error_scores(
        transition_scores,
        mode=transition_score_mode,
    )
    scores = select_trajectory_scores(
        pd.read_parquet(
            Path(prediction_root)
            / model_id
            / dataset
            / "trajectory_scores.parquet"
        ),
        transition_scores,
        aggregation=aggregation,
        ignore_first_transition=not include_first_transition,
        transition_score_mode=transition_score_mode,
    )
    trajectories = pd.read_parquet(
        Path(feature_root) / feature_set / dataset / "trajectories.parquet",
        columns=[
            "trajectory_id",
            "n_transitions",
            "use_for_calibration",
        ],
    )
    scores = scores.merge(
        trajectories,
        on="trajectory_id",
        how="inner",
        validate="one_to_one",
    )
    calibration_scores = scores.loc[
        scores["use_for_calibration"],
        "selected_score",
    ]
    if population in TRANSITION_POPULATIONS:
        if not include_first_transition:
            transition_scores = transition_scores.loc[
                transition_scores["transition_order"] > 0
            ].copy()
        calibration_ids = set(
            trajectories.loc[
                trajectories["use_for_calibration"],
                "trajectory_id",
            ].astype(int)
        )
        calibration_transition_scores = transition_scores.loc[
            transition_scores["trajectory_id"].isin(calibration_ids),
            "selected_score",
        ]
        if calibration_transition_scores.empty:
            raise ValueError("Transition calibration population is empty")
        threshold = float(
            np.percentile(
                calibration_transition_scores,
                100.0 - anomaly_percentage,
            )
        )
        transition_scores["model_flagged"] = transition_scores[
            "selected_score"
        ].ge(threshold)
        selected_mask = (
            transition_scores["model_flagged"]
            if population == "transition_flagged"
            else ~transition_scores["model_flagged"]
        )
        selected_transitions = transition_scores.loc[selected_mask].copy()
        selected_values = transition_population_values(
            metric,
            processed_root,
            dataset,
            selected_transitions,
        )
        comparison_values = transition_population_values(
            metric,
            processed_root,
            dataset,
            transition_scores,
        )
        selected_histogram, comparison_histogram = shared_histograms(
            selected_values,
            comparison_values,
        )
        return {
            "dataset": dataset,
            "subset": population,
            "metric": metric,
            "model_id": model_id,
            "anomaly_percentage": float(anomaly_percentage),
            "threshold": threshold,
            "threshold_level": "transition",
            "aggregation": aggregation,
            "transition_score_mode": transition_score_mode,
            "include_first_transition": bool(include_first_transition),
            "n_trajectories": int(
                selected_transitions["trajectory_id"].nunique()
            ),
            "n_transitions": int(len(selected_transitions)),
            "total_trajectories": int(
                transition_scores["trajectory_id"].nunique()
            ),
            "total_transitions": int(len(transition_scores)),
            "flagged_transitions": int(transition_scores["model_flagged"].sum()),
            "flagged_transition_rate": float(
                transition_scores["model_flagged"].mean()
            ),
            "summary": numeric_summary(selected_values),
            "comparison_summary": numeric_summary(comparison_values),
            "histogram": selected_histogram,
            "comparison_histogram": comparison_histogram,
            "comparison_label": "All scored transitions",
            "feature_errors": feature_error_summary_by_transitions(
                selected_transitions,
                transition_scores,
            ),
            "data_quality": {},
        }

    threshold = float(
        np.percentile(calibration_scores, 100.0 - anomaly_percentage)
    )
    scores["model_flagged"] = scores["selected_score"].ge(threshold)
    selected_mask = (
        scores["model_flagged"]
        if population == "model_flagged"
        else ~scores["model_flagged"]
    )
    selected_ids = scores.loc[selected_mask, "trajectory_id"].astype(int).tolist()
    all_ids = scores["trajectory_id"].astype(int).tolist()

    selected_values = population_values(
        metric,
        processed_root,
        dataset,
        trajectories,
        selected_ids,
        scores,
    )
    comparison_values = population_values(
        metric,
        processed_root,
        dataset,
        trajectories,
        all_ids,
        scores,
    )
    selected_histogram, comparison_histogram = shared_histograms(
        selected_values,
        comparison_values,
    )
    return {
        "dataset": dataset,
        "subset": population,
        "metric": metric,
        "model_id": model_id,
        "anomaly_percentage": float(anomaly_percentage),
        "threshold": threshold,
        "threshold_level": "trajectory",
        "aggregation": aggregation,
        "transition_score_mode": transition_score_mode,
        "include_first_transition": bool(include_first_transition),
        "n_trajectories": int(len(selected_ids)),
        "n_transitions": int(
            trajectories.loc[
                trajectories["trajectory_id"].isin(selected_ids),
                "n_transitions",
            ].sum()
        ),
        "total_trajectories": int(len(scores)),
        "flagged_trajectories": int(scores["model_flagged"].sum()),
        "flagged_rate": float(scores["model_flagged"].mean()),
        "summary": numeric_summary(selected_values),
        "comparison_summary": numeric_summary(comparison_values),
        "histogram": selected_histogram,
        "comparison_histogram": comparison_histogram,
        "comparison_label": "All model-eligible trajectories",
        "feature_errors": feature_error_summary(
            transition_scores,
            set(selected_ids),
        ),
        "data_quality": {},
    }
