from pathlib import Path

import pandas as pd


DEFAULT_LABEL_PATH = Path("data/labels/plausible_trajectories.csv")
BASE_COLUMNS = ["dataset", "user_id", "start_date"]
EXTENDED_COLUMNS = [
    "dataset",
    "user_id",
    "start_date",
    "trajectory_id",
    "source",
    "created_at",
    "notes",
]


def normalize_date(value):
    return str(value).strip().replace("-", "/")


def read_plausible_label_table(label_path=DEFAULT_LABEL_PATH):
    path = Path(label_path)
    labels_list = []
    if path.exists():
        labels_list.append(pd.read_csv(path, dtype=str))
    
    review_path = path.parent / "trajectory_reviews.csv"
    if review_path.exists():
        reviews = pd.read_csv(review_path, dtype=str)
        if "label" in reviews.columns:
            plausible_reviews = reviews.loc[reviews["label"].str.strip().eq("plausible")].copy()
            if not plausible_reviews.empty:
                labels_list.append(plausible_reviews)

    if not labels_list:
        return pd.DataFrame(columns=EXTENDED_COLUMNS)

    labels = pd.concat(labels_list, ignore_index=True)
    required = {"dataset", "user_id", "start_date"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            f"{label_path} is missing columns: {', '.join(sorted(missing))}"
        )
    for column in EXTENDED_COLUMNS:
        if column not in labels.columns:
            labels[column] = ""
    labels = labels[EXTENDED_COLUMNS].fillna("")
    labels["dataset"] = labels["dataset"].astype(str).str.strip()
    labels["user_id"] = labels["user_id"].astype(str).str.strip()
    labels["start_date"] = labels["start_date"].astype(str).str.strip()
    labels["trajectory_id"] = labels["trajectory_id"].astype(str).str.strip()
    labels = labels.drop_duplicates(subset=["dataset", "user_id", "start_date", "trajectory_id"]).reset_index(drop=True)
    return labels


def _attach_start_dates(candidates, processed_root, dataset, user_ids=None):
    observation_filters = None
    if user_ids:
        observation_filters = [("user_id", "in", [str(value) for value in user_ids])]
    observations = pd.read_parquet(
        Path(processed_root) / dataset / "observations.parquet",
        columns=["observation_id", "user_id", "timestamp"],
        filters=observation_filters,
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
    return candidates


def _resolve_dataset_labels(dataset, dataset_labels, candidates, processed_root):
    candidates = candidates.copy()
    candidates["user_id"] = candidates["user_id"].astype(str)
    resolved = []
    user_ids = (
        dataset_labels.loc[
            dataset_labels["trajectory_id"].eq(""),
            "user_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    candidates_with_dates = None

    for label in dataset_labels.itertuples(index=False):
        trajectory_id = str(label.trajectory_id).strip()
        if trajectory_id:
            try:
                trajectory_id_value = int(trajectory_id)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid trajectory_id in plausible label: {trajectory_id}"
                ) from exc
            user_candidates = candidates.loc[
                candidates["trajectory_id"].eq(trajectory_id_value)
            ].copy()
            if str(label.user_id).strip() not in {"", "*"}:
                user_candidates = user_candidates.loc[
                    user_candidates["user_id"].eq(str(label.user_id))
                ]
            if user_candidates.empty:
                raise ValueError(
                    "No trajectory matches plausible label "
                    f"{dataset}:{label.user_id}:{trajectory_id}"
                )
            if "start_date" not in user_candidates.columns:
                label_start_date = str(label.start_date).strip()
                if label_start_date and label_start_date != "*":
                    user_candidates["start_date"] = normalize_date(label_start_date)
                else:
                    user_candidates = _attach_start_dates(
                        user_candidates,
                        processed_root,
                        dataset,
                        user_ids=user_candidates["user_id"].astype(str).unique(),
                    )
        else:
            if candidates_with_dates is None:
                date_candidates = candidates.loc[
                    candidates["user_id"].isin(user_ids)
                ].copy()
                candidates_with_dates = _attach_start_dates(
                    date_candidates,
                    processed_root,
                    dataset,
                    user_ids=user_ids,
                )
            user_candidates = candidates_with_dates.loc[
                candidates_with_dates["user_id"].eq(str(label.user_id))
            ]
            if label.start_date != "*":
                user_candidates = user_candidates.loc[
                    user_candidates["start_date"].eq(
                        normalize_date(label.start_date)
                    )
                ]
            if user_candidates.empty:
                raise ValueError(
                    "No trajectory matches plausible label "
                    f"{dataset}:{label.user_id}:{label.start_date}"
                )
        resolved.append(
            user_candidates[
                ["trajectory_id", "user_id", "start_date"]
            ].assign(dataset=dataset)
        )

    if not resolved:
        return pd.DataFrame(columns=["dataset", "trajectory_id", "user_id", "start_date"])
    return pd.concat(resolved, ignore_index=True)


def resolve_plausible_trajectories_from_processed(
    processed_root="data/processed_parquet",
    label_path=DEFAULT_LABEL_PATH,
):
    labels = read_plausible_label_table(label_path)
    resolved = []
    for dataset, dataset_labels in labels.groupby("dataset", sort=True):
        trajectories = pd.read_parquet(
            Path(processed_root) / dataset / "trajectories.parquet",
            columns=[
                "trajectory_id",
                "user_id",
                "start_observation_id",
                "n_transitions",
            ],
        )
        resolved.append(
            _resolve_dataset_labels(
                dataset,
                dataset_labels,
                trajectories,
                processed_root,
            )
        )
    if not resolved:
        return pd.DataFrame(columns=["dataset", "trajectory_id", "user_id", "start_date"])
    return (
        pd.concat(resolved, ignore_index=True)
        .drop_duplicates(["dataset", "trajectory_id"])
        .sort_values(["dataset", "trajectory_id"])
        .reset_index(drop=True)
    )


def resolve_plausible_trajectories(
    feature_root="data/features",
    processed_root="data/processed_parquet",
    feature_set="motion_v2",
    label_path=DEFAULT_LABEL_PATH,
):
    labels = read_plausible_label_table(label_path)

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
        candidates = _resolve_dataset_labels(
            dataset,
            dataset_labels,
            trajectories,
            processed_root,
        )
        candidates = candidates.merge(
            trajectories[["trajectory_id", "use_for_calibration"]],
            on="trajectory_id",
            how="left",
            validate="many_to_one",
        )
        candidates = candidates.loc[candidates["use_for_calibration"].fillna(False)]
        if candidates.empty:
            continue
        resolved.append(candidates[["trajectory_id", "user_id", "start_date", "dataset"]])

    if not resolved:
        raise ValueError(f"No plausible trajectories resolved from {label_path}")
    return (
        pd.concat(resolved, ignore_index=True)
        .drop_duplicates(["dataset", "trajectory_id"])
        .sort_values(["dataset", "trajectory_id"])
        .reset_index(drop=True)
    )
