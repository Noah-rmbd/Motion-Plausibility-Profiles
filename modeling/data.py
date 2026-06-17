from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset


IDENTIFIER_COLUMNS = ["trajectory_id", "transition_order", "transition_id"]
DEFAULT_PROCESSED_ROOT = "data/processed_parquet"
DEFAULT_PLAUSIBLE_LABELS = "data/labels/plausible_trajectories.csv"


def _format_id_sample(ids, limit=10):
    values = sorted(int(value) for value in ids)
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit})"
    return ", ".join(str(value) for value in values[:limit]) + suffix


@lru_cache(maxsize=128)
def _excluded_trajectory_ids(feature_set, dataset):
    path = Path("artifacts/runs") / feature_set / "exclusions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required to prove deterministic-rule exclusions "
            "are absent from training data"
        )
    exclusions = pd.read_parquet(
        path,
        columns=["dataset", "trajectory_id"],
    )
    ids = exclusions.loc[
        exclusions["dataset"].astype(str).eq(str(dataset)),
        "trajectory_id",
    ]
    return frozenset(ids.astype(np.int64).tolist())


@lru_cache(maxsize=128)
def _synthetic_source_trajectory_ids(feature_root, feature_set, dataset):
    path = Path(feature_root) / feature_set / "synthetic" / "trajectories.parquet"
    if not path.exists():
        return frozenset()
    columns = ["source_dataset", "source_trajectory_id"]
    synthetic = pd.read_parquet(path, columns=columns)
    if not set(columns).issubset(synthetic.columns):
        return frozenset()
    ids = synthetic.loc[
        synthetic["source_dataset"].astype(str).eq(str(dataset)),
        "source_trajectory_id",
    ]
    return frozenset(ids.dropna().astype(np.int64).tolist())


@lru_cache(maxsize=128)
def _trusted_plausible_trajectory_ids(
    feature_root,
    processed_root,
    feature_set,
    dataset,
    label_path,
):
    path = Path(label_path)
    if not path.exists():
        return frozenset()
    try:
        from modeling.plausible_labels import resolve_plausible_trajectories

        labels = resolve_plausible_trajectories(
            feature_root=feature_root,
            processed_root=processed_root,
            feature_set=feature_set,
            label_path=path,
        )
    except FileNotFoundError:
        return frozenset()
    ids = labels.loc[
        labels["dataset"].astype(str).eq(str(dataset)),
        "trajectory_id",
    ]
    return frozenset(ids.astype(np.int64).tolist())


def _require_no_overlap(dataset, selected_ids, forbidden_ids, reason):
    overlap = set(selected_ids) & set(forbidden_ids)
    if overlap:
        raise ValueError(
            f"{dataset}: training data includes {len(overlap)} {reason} "
            f"trajectories: {_format_id_sample(overlap)}"
        )


def validate_training_trajectories(
    trajectory_table,
    feature_root,
    feature_set,
    dataset,
    processed_root=DEFAULT_PROCESSED_ROOT,
    plausible_labels_path=DEFAULT_PLAUSIBLE_LABELS,
):
    if dataset == "synthetic":
        raise ValueError("Synthetic trajectories must never be used for training")

    required_columns = {"trajectory_id", "use_for_training"}
    missing = required_columns - set(trajectory_table.columns)
    if missing:
        raise ValueError(
            f"{dataset}: training trajectory table is missing columns: "
            + ", ".join(sorted(missing))
        )

    training_table = trajectory_table.loc[trajectory_table["use_for_training"]].copy()
    if training_table.empty:
        raise ValueError(f"{dataset}: no trajectories are marked use_for_training")

    selected_ids = set(training_table["trajectory_id"].astype(np.int64).tolist())
    if "use_for_calibration" in training_table.columns:
        calibration_ids = training_table.loc[
            training_table["use_for_calibration"],
            "trajectory_id",
        ]
        _require_no_overlap(
            dataset,
            selected_ids,
            calibration_ids.astype(np.int64).tolist(),
            "calibration/benchmark",
        )
    if "is_synthetic_anomaly" in training_table.columns:
        synthetic_ids = training_table.loc[
            training_table["is_synthetic_anomaly"],
            "trajectory_id",
        ]
        _require_no_overlap(
            dataset,
            selected_ids,
            synthetic_ids.astype(np.int64).tolist(),
            "synthetic anomaly",
        )
    if "split" in training_table.columns:
        non_fit_ids = training_table.loc[
            ~training_table["split"].astype(str).eq("fit"),
            "trajectory_id",
        ]
        _require_no_overlap(
            dataset,
            selected_ids,
            non_fit_ids.astype(np.int64).tolist(),
            "non-fit split",
        )
    if "source" in training_table.columns:
        non_real_ids = training_table.loc[
            ~training_table["source"].astype(str).eq("real"),
            "trajectory_id",
        ]
        _require_no_overlap(
            dataset,
            selected_ids,
            non_real_ids.astype(np.int64).tolist(),
            "non-real source",
        )

    _require_no_overlap(
        dataset,
        selected_ids,
        _excluded_trajectory_ids(feature_set, dataset),
        "deterministic-rule-excluded",
    )
    _require_no_overlap(
        dataset,
        selected_ids,
        _synthetic_source_trajectory_ids(str(feature_root), feature_set, dataset),
        "synthetic source-control",
    )
    _require_no_overlap(
        dataset,
        selected_ids,
        _trusted_plausible_trajectory_ids(
            str(feature_root),
            DEFAULT_PROCESSED_ROOT if processed_root is None else str(processed_root),
            feature_set,
            dataset,
            str(plausible_labels_path),
        ),
        "trusted plausible benchmark",
    )
    return training_table


@dataclass(frozen=True)
class TrajectoryMetadata:
    dataset: str
    trajectory_id: int
    transition_orders: np.ndarray
    transition_ids: np.ndarray


class ParquetTrajectoryDataset(Dataset):
    def __init__(
        self,
        feature_root,
        feature_set,
        dataset,
        feature_names,
        context_feature_names=None,
        training_only=False,
        max_trajectories=None,
        sample_trajectories=None,
        seed=42,
    ):
        self.dataset_name = dataset
        context_feature_names = list(context_feature_names or [])
        base = Path(feature_root) / feature_set / dataset
        trajectory_table = pd.read_parquet(base / "trajectories.parquet")
        if training_only:
            trajectory_table = validate_training_trajectories(
                trajectory_table=trajectory_table,
                feature_root=feature_root,
                feature_set=feature_set,
                dataset=dataset,
            )
        trajectory_table = trajectory_table.sort_values("trajectory_id")
        if sample_trajectories is not None and len(trajectory_table) > sample_trajectories:
            trajectory_table = trajectory_table.sample(
                n=sample_trajectories,
                random_state=seed,
            ).sort_values("trajectory_id")
        if max_trajectories is not None:
            trajectory_table = trajectory_table.head(max_trajectories)

        trajectory_ids = trajectory_table["trajectory_id"].to_numpy(dtype=np.int64)
        parquet_path = base / "transition_features.parquet"
        value_columns = list(dict.fromkeys([*feature_names, *context_feature_names]))
        columns = [*IDENTIFIER_COLUMNS, *value_columns]
        if 0 < len(trajectory_ids) <= 10_000:
            feature_table = pd.read_parquet(
                parquet_path,
                columns=columns,
                filters=[("trajectory_id", "in", trajectory_ids.tolist())],
            )
        else:
            feature_table = pd.read_parquet(parquet_path, columns=columns)
            feature_table = feature_table.loc[feature_table["trajectory_id"].isin(trajectory_ids)]
        feature_table = feature_table.sort_values(["trajectory_id", "transition_order"])

        available_ids = set(feature_table["trajectory_id"].unique())
        missing_ids = set(trajectory_ids) - available_ids
        if missing_ids:
            raise ValueError(f"{dataset}: {len(missing_ids)} trajectories have no feature rows")

        counts = feature_table.groupby("trajectory_id", sort=False).size()
        ordered_ids = counts.index.to_numpy(dtype=np.int64)
        self._starts = np.concatenate(([0], np.cumsum(counts.to_numpy(dtype=np.int64))[:-1]))
        self._ends = np.cumsum(counts.to_numpy(dtype=np.int64))
        self._trajectory_ids = ordered_ids
        self._transition_orders = feature_table["transition_order"].to_numpy(dtype=np.int32)
        self._transition_ids = feature_table["transition_id"].to_numpy(dtype=np.int64)
        self._features = feature_table[feature_names].to_numpy(dtype=np.float32)
        self.feature_names = list(feature_names)
        self.context_feature_names = context_feature_names
        self._context_features = (
            feature_table[context_feature_names].to_numpy(dtype=np.float32)
            if context_feature_names
            else None
        )

    def __len__(self):
        return len(self._trajectory_ids)

    def __getitem__(self, index):
        start = self._starts[index]
        end = self._ends[index]
        metadata = TrajectoryMetadata(
            dataset=self.dataset_name,
            trajectory_id=int(self._trajectory_ids[index]),
            transition_orders=self._transition_orders[start:end],
            transition_ids=self._transition_ids[start:end],
        )
        sequence = torch.from_numpy(self._features[start:end])
        if self._context_features is None:
            return sequence, metadata
        context = torch.from_numpy(self._context_features[start:end])
        return sequence, context, metadata


class ForecastingTrajectoryDataset(ParquetTrajectoryDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self._context_features is not None:
            raise ValueError("ForecastingTrajectoryDataset does not support context features")
        sequence, metadata = item
        if len(sequence) < 2:
            raise ValueError(
                f"{metadata.dataset}/{metadata.trajectory_id}: forecasting requires at least 2 transitions"
            )
        input_metadata = TrajectoryMetadata(
            dataset=metadata.dataset,
            trajectory_id=metadata.trajectory_id,
            transition_orders=metadata.transition_orders[1:],
            transition_ids=metadata.transition_ids[1:],
        )
        return sequence[:-1], sequence[1:], input_metadata


def collate_trajectories(batch):
    batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
    sequences, metadata = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, metadata


def collate_forecasting_trajectories(batch):
    batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
    inputs, targets, metadata = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in inputs], dtype=torch.long)
    padded_inputs = pad_sequence(inputs, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)
    return padded_inputs, padded_targets, lengths, metadata


def collate_context_trajectories(batch):
    batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
    sequences, contexts, metadata = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    padded_contexts = pad_sequence(contexts, batch_first=True)
    return padded_sequences, lengths, padded_contexts, metadata


def load_combined_datasets(
    feature_root,
    feature_set,
    datasets,
    feature_names,
    context_feature_names=None,
    training_only=False,
    max_trajectories=None,
    sampling_strategy="all",
    seed=42,
):
    if sampling_strategy not in {"all", "balanced"}:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    sample_trajectories = None
    if training_only and sampling_strategy == "balanced" and len(datasets) > 1:
        counts = []
        for dataset in datasets:
            table = pd.read_parquet(
                Path(feature_root) / feature_set / dataset / "trajectories.parquet",
                columns=["use_for_training"],
            )
            counts.append(int(table["use_for_training"].sum()))
        sample_trajectories = min(counts)
        if max_trajectories is not None:
            sample_trajectories = min(sample_trajectories, max_trajectories)

    loaded = [
        ParquetTrajectoryDataset(
            feature_root=feature_root,
            feature_set=feature_set,
            dataset=dataset,
            feature_names=feature_names,
            context_feature_names=context_feature_names,
            training_only=training_only,
            max_trajectories=None if sample_trajectories is not None else max_trajectories,
            sample_trajectories=sample_trajectories,
            seed=seed + index,
        )
        for index, dataset in enumerate(datasets)
    ]
    if not loaded:
        raise ValueError("At least one dataset is required")
    return loaded[0] if len(loaded) == 1 else ConcatDataset(loaded)


def load_forecasting_datasets(
    feature_root,
    feature_set,
    datasets,
    feature_names,
    training_only=False,
    max_trajectories=None,
    sampling_strategy="all",
    seed=42,
):
    if sampling_strategy not in {"all", "balanced"}:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    sample_trajectories = None
    if training_only and sampling_strategy == "balanced" and len(datasets) > 1:
        counts = []
        for dataset in datasets:
            table = pd.read_parquet(
                Path(feature_root) / feature_set / dataset / "trajectories.parquet",
                columns=["use_for_training"],
            )
            counts.append(int(table["use_for_training"].sum()))
        sample_trajectories = min(counts)
        if max_trajectories is not None:
            sample_trajectories = min(sample_trajectories, max_trajectories)

    loaded = [
        ForecastingTrajectoryDataset(
            feature_root=feature_root,
            feature_set=feature_set,
            dataset=dataset,
            feature_names=feature_names,
            training_only=training_only,
            max_trajectories=None if sample_trajectories is not None else max_trajectories,
            sample_trajectories=sample_trajectories,
            seed=seed + index,
        )
        for index, dataset in enumerate(datasets)
    ]
    if not loaded:
        raise ValueError("At least one dataset is required")
    return loaded[0] if len(loaded) == 1 else ConcatDataset(loaded)
