from pathlib import Path

import pandas as pd

from modeling.labels.plausible_labels import _resolve_dataset_labels


DEFAULT_REVIEW_LABEL_PATH = Path("data/labels/trajectory_reviews.csv")
VALID_LABELS = {"plausible", "unplausible", "unsure"}
BENCHMARK_LABELS = {"plausible", "unplausible"}
REVIEW_COLUMNS = [
    "dataset",
    "user_id",
    "start_date",
    "trajectory_id",
    "label",
    "source",
    "created_at",
    "notes",
]


def read_review_label_table(label_path=DEFAULT_REVIEW_LABEL_PATH):
    path = Path(label_path)
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    labels = pd.read_csv(path, dtype=str)
    required = {"dataset", "user_id", "trajectory_id", "label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            f"{label_path} is missing columns: {', '.join(sorted(missing))}"
        )
    for column in REVIEW_COLUMNS:
        if column not in labels.columns:
            labels[column] = ""
    labels = labels[REVIEW_COLUMNS].fillna("")
    labels["dataset"] = labels["dataset"].astype(str).str.strip()
    labels["user_id"] = labels["user_id"].astype(str).str.strip()
    labels["start_date"] = labels["start_date"].astype(str).str.strip()
    labels["trajectory_id"] = labels["trajectory_id"].astype(str).str.strip()
    labels["label"] = labels["label"].astype(str).str.strip().str.lower()
    invalid = sorted(set(labels["label"]) - VALID_LABELS)
    if invalid:
        raise ValueError(
            f"{label_path} contains invalid labels: {', '.join(invalid)}"
        )
    return labels


def write_review_label_table(labels, label_path=DEFAULT_REVIEW_LABEL_PATH):
    path = Path(label_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(labels, columns=REVIEW_COLUMNS).fillna("")
    if not frame.empty:
        frame = frame.sort_values(
            ["dataset", "user_id", "trajectory_id", "label"]
        )
    frame.to_csv(path, index=False)


def resolve_reviewed_trajectories_from_processed(
    processed_root="data/processed_parquet",
    label_path=DEFAULT_REVIEW_LABEL_PATH,
    labels=None,
):
    review_labels = read_review_label_table(label_path)
    if labels is None:
        labels = BENCHMARK_LABELS
    labels = {str(label).lower() for label in labels}
    review_labels = review_labels.loc[review_labels["label"].isin(labels)]
    resolved = []
    for dataset, dataset_labels in review_labels.groupby("dataset", sort=True):
        trajectories = pd.read_parquet(
            Path(processed_root) / dataset / "trajectories.parquet",
            columns=[
                "trajectory_id",
                "user_id",
                "start_observation_id",
                "n_transitions",
            ],
        )
        resolved_frame = _resolve_dataset_labels(
            dataset,
            dataset_labels,
            trajectories,
            processed_root,
        )
        label_lookup = dataset_labels.set_index("trajectory_id")["label"].to_dict()
        resolved_frame["label"] = resolved_frame["trajectory_id"].astype(str).map(
            label_lookup
        )
        resolved.append(resolved_frame)
    if not resolved:
        return pd.DataFrame(
            columns=["dataset", "trajectory_id", "user_id", "start_date", "label"]
        )
    return (
        pd.concat(resolved, ignore_index=True)
        .drop_duplicates(["dataset", "trajectory_id"])
        .sort_values(["dataset", "trajectory_id"])
        .reset_index(drop=True)
    )


def resolve_reviewed_trajectories(
    feature_root="data/features",
    processed_root="data/processed_parquet",
    feature_set="motion_v2",
    label_path=DEFAULT_REVIEW_LABEL_PATH,
    labels=None,
):
    reviewed = resolve_reviewed_trajectories_from_processed(
        processed_root=processed_root,
        label_path=label_path,
        labels=labels,
    )
    if reviewed.empty:
        return reviewed
    resolved = []
    for dataset, dataset_labels in reviewed.groupby("dataset", sort=True):
        trajectories = pd.read_parquet(
            Path(feature_root) / feature_set / dataset / "trajectories.parquet",
            columns=["trajectory_id"],
        )
        resolved.append(
            dataset_labels.merge(
                trajectories,
                on="trajectory_id",
                how="inner",
                validate="one_to_one",
            )
        )
    if not resolved:
        return reviewed.iloc[0:0]
    return (
        pd.concat(resolved, ignore_index=True)
        .drop_duplicates(["dataset", "trajectory_id"])
        .sort_values(["dataset", "trajectory_id"])
        .reset_index(drop=True)
    )
