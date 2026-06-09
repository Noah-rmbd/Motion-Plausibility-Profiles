import numpy as np


def select_valid_trajectory_ids(transitions, trajectory_transitions, trajectories, min_transitions):
    transition_flags = transitions[["transition_id", "transition_plausibility"]]
    mapped = trajectory_transitions.merge(transition_flags, on="transition_id", how="left")
    grouped = mapped.groupby("trajectory_id", sort=False)["transition_plausibility"]

    has_only_plausible_transitions = grouped.min().eq(1)
    has_no_missing_transition = grouped.count().eq(grouped.size())
    valid_by_transitions = has_only_plausible_transitions & has_no_missing_transition

    valid_ids = set(valid_by_transitions[valid_by_transitions].index)
    valid_ids &= set(trajectories.loc[trajectories["n_transitions"] >= min_transitions, "trajectory_id"])
    return valid_ids, valid_by_transitions


def build_trajectory_exclusions(trajectories, valid_ids, valid_by_transitions, min_transitions):
    excluded_ids = set(trajectories["trajectory_id"]) - valid_ids
    exclusions = trajectories.loc[
        trajectories["trajectory_id"].isin(excluded_ids),
        ["trajectory_id", "n_transitions"],
    ].copy()

    invalid_transition_ids = set(valid_by_transitions[~valid_by_transitions].index)
    exclusions["exclusion_reason"] = np.where(
        exclusions["n_transitions"] < min_transitions,
        "too_few_transitions",
        "contains_implausible_transition",
    )
    exclusions.loc[
        exclusions["trajectory_id"].isin(invalid_transition_ids)
        & (exclusions["exclusion_reason"] != "too_few_transitions"),
        "exclusion_reason",
    ] = "contains_implausible_transition"
    return exclusions
