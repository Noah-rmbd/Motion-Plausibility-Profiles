import numpy as np
import pandas as pd


AGGREGATION_COLUMNS = {
    "mean": "score_mean",
    "max": "score_max",
    "top_k": "score_top_k",
    "window_max": "score_window_max",
}
TRANSITION_SCORE_MODES = (
    "mean",
    "weighted",
    "mean_no_bearing",
    "weighted_no_bearing",
    "mean_no_acceleration",
    "weighted_no_acceleration",
)
DEFAULT_TOP_K = 3
DEFAULT_WINDOW_SIZE = 5


def feature_error_columns(frame):
    return [column for column in frame.columns if column.startswith("error_")]


def feature_group(feature_name):
    if feature_name.startswith("speed_"):
        return "speed_profile"
    if feature_name in {"bearing_sin", "bearing_cos", "bearing_valid"}:
        return "bearing"
    return feature_name


def score_mode_base(mode):
    if mode.startswith("weighted"):
        return "weighted"
    return "mean"


def excluded_feature_groups(mode):
    groups = set()
    if mode.endswith("_no_bearing"):
        groups.add("bearing")
    if mode.endswith("_no_acceleration"):
        groups.add("acceleration")
    return groups


def transition_error_scores(transition_scores, mode="mean"):
    if mode not in TRANSITION_SCORE_MODES:
        raise ValueError(f"Unknown transition score mode: {mode}")
    error_columns = feature_error_columns(transition_scores)
    if not error_columns:
        if "score" not in transition_scores:
            raise ValueError("Transition scores contain no feature-wise errors")
        return transition_scores["score"].astype(float)
    excluded_groups = excluded_feature_groups(mode)
    error_columns = [
        column
        for column in error_columns
        if feature_group(column.removeprefix("error_")) not in excluded_groups
    ]
    if not error_columns:
        raise ValueError(f"{mode} excludes every available feature-error column")
    if score_mode_base(mode) == "mean":
        return pd.Series(
            transition_scores[error_columns].to_numpy(dtype=float).mean(axis=1),
            index=transition_scores.index,
        )

    grouped_columns = {}
    for column in error_columns:
        feature_name = column.removeprefix("error_")
        grouped_columns.setdefault(feature_group(feature_name), []).append(column)
    group_scores = [
        transition_scores[columns].to_numpy(dtype=float).mean(axis=1)
        for columns in grouped_columns.values()
    ]
    return pd.Series(
        np.vstack(group_scores).mean(axis=0),
        index=transition_scores.index,
    )


def rolling_window_max_score(scores, window_size):
    values = np.asarray(scores, dtype=float)
    if len(values) == 0:
        return np.nan
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    positions = np.arange(len(values))
    starts = np.maximum(0, positions - window_size + 1)
    lengths = positions - starts + 1
    window_means = (cumulative[positions + 1] - cumulative[starts]) / lengths
    return float(window_means.max())


def aggregate_transition_scores(
    transition_scores,
    ignore_first_transition=True,
    top_k=DEFAULT_TOP_K,
    window_size=DEFAULT_WINDOW_SIZE,
    transition_score_mode="mean",
):
    frame = transition_scores[["trajectory_id", "transition_order"]].copy()
    frame["score"] = transition_error_scores(
        transition_scores,
        mode=transition_score_mode,
    ).to_numpy()
    if ignore_first_transition:
        eligible = frame.loc[frame["transition_order"] > 0]
        trajectories_with_eligible_steps = set(eligible["trajectory_id"])
        fallback = frame.loc[~frame["trajectory_id"].isin(trajectories_with_eligible_steps)]
        frame = pd.concat([eligible, fallback], ignore_index=True)

    grouped = frame.groupby("trajectory_id", sort=False)["score"]
    aggregates = grouped.agg(score_mean="mean", score_max="max").reset_index()
    top_k_scores = (
        frame.sort_values(["trajectory_id", "score"], ascending=[True, False])
        .groupby("trajectory_id", sort=False)
        .head(top_k)
        .groupby("trajectory_id", sort=False)["score"]
        .mean()
        .rename("score_top_k")
        .reset_index()
    )
    window_frame = frame.sort_values(["trajectory_id", "transition_order"]).copy()
    window_scores = (
        window_frame.groupby("trajectory_id", sort=False)["score"]
        .agg(lambda scores: rolling_window_max_score(scores, window_size))
        .rename("score_window_max")
        .reset_index()
    )
    return aggregates.merge(
        top_k_scores,
        on="trajectory_id",
        how="left",
        validate="one_to_one",
    ).merge(
        window_scores,
        on="trajectory_id",
        how="left",
        validate="one_to_one",
    )


def uses_native_trajectory_score(trajectory_scores):
    return (
        not trajectory_scores.empty
        and "score_type" in trajectory_scores.columns
        and trajectory_scores["score_type"].notna().all()
    )


def select_trajectory_scores(
    trajectory_scores,
    transition_scores,
    aggregation="mean",
    ignore_first_transition=True,
    transition_score_mode="mean",
):
    if aggregation not in AGGREGATION_COLUMNS:
        raise ValueError(f"Unknown trajectory aggregation: {aggregation}")
    if uses_native_trajectory_score(trajectory_scores):
        return trajectory_scores[["trajectory_id", "score"]].rename(
            columns={"score": "selected_score"}
        )
    score_column = AGGREGATION_COLUMNS[aggregation]
    return aggregate_transition_scores(
        transition_scores,
        ignore_first_transition=ignore_first_transition,
        transition_score_mode=transition_score_mode,
    )[["trajectory_id", score_column]].rename(
        columns={score_column: "selected_score"}
    )
