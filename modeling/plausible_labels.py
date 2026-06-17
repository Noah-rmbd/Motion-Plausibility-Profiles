from pathlib import Path

import pandas as pd


DEFAULT_LABEL_PATH = Path("data/labels/plausible_trajectories.csv")


def normalize_date(value):
    return str(value).strip().replace("-", "/")


def resolve_plausible_trajectories(
    feature_root="data/features",
    processed_root="data/processed_parquet",
    feature_set="motion_v2",
    label_path=DEFAULT_LABEL_PATH,
):
    labels = pd.read_csv(label_path, dtype=str)
    required = {"dataset", "user_id", "start_date"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            f"{label_path} is missing columns: {', '.join(sorted(missing))}"
        )

    resolved = []
    for dataset, dataset_labels in labels.groupby("dataset", sort=True):
        trajectories = pd.read_parquet(
            Path(feature_root) / feature_set / dataset / "trajectories.parquet",
            columns=[
                "trajectory_id",
                "user_id",
                "start_observation_id",
                "use_for_calibration",
            ],
        )
        trajectories["user_id"] = trajectories["user_id"].astype(str)
        user_ids = dataset_labels["user_id"].astype(str).unique().tolist()
        candidates = trajectories.loc[trajectories["user_id"].isin(user_ids)].copy()
        if not candidates["use_for_calibration"].all():
            invalid_users = sorted(
                candidates.loc[
                    ~candidates["use_for_calibration"],
                    "user_id",
                ].unique()
            )
            raise ValueError(
                "Manually plausible trajectories must be calibration-only; "
                f"training rows found for {dataset}: {invalid_users}"
            )

        observations = pd.read_parquet(
            Path(processed_root) / dataset / "observations.parquet",
            columns=["observation_id", "user_id", "timestamp"],
            filters=[("user_id", "in", user_ids)],
        )
        observations["user_id"] = observations["user_id"].astype(str)
        starts = observations.rename(
            columns={
                "observation_id": "start_observation_id",
                "timestamp": "start_timestamp",
            }
        )[["start_observation_id", "start_timestamp"]]
        candidates = candidates.merge(
            starts,
            on="start_observation_id",
            how="left",
            validate="one_to_one",
        )
        candidates["start_date"] = candidates["start_timestamp"].str[:10].map(
            normalize_date
        )

        for label in dataset_labels.itertuples(index=False):
            user_candidates = candidates.loc[
                candidates["user_id"].eq(str(label.user_id))
            ]
            if label.start_date != "*":
                user_candidates = user_candidates.loc[
                    user_candidates["start_date"].eq(
                        normalize_date(label.start_date)
                    )
                ]
            if user_candidates.empty:
                raise ValueError(
                    "No calibration trajectory matches plausible label "
                    f"{dataset}:{label.user_id}:{label.start_date}"
                )
            resolved.append(
                user_candidates[
                    ["trajectory_id", "user_id", "start_date"]
                ].assign(dataset=dataset)
            )

    if not resolved:
        raise ValueError(f"No plausible trajectories resolved from {label_path}")
    return (
        pd.concat(resolved, ignore_index=True)
        .drop_duplicates(["dataset", "trajectory_id"])
        .sort_values(["dataset", "trajectory_id"])
        .reset_index(drop=True)
    )
