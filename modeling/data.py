from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset


IDENTIFIER_COLUMNS = ["trajectory_id", "transition_order", "transition_id"]


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
        training_only=False,
        max_trajectories=None,
    ):
        self.dataset_name = dataset
        base = Path(feature_root) / feature_set / dataset
        trajectory_table = pd.read_parquet(base / "trajectories.parquet")
        if training_only:
            trajectory_table = trajectory_table.loc[trajectory_table["use_for_training"]]
        trajectory_table = trajectory_table.sort_values("trajectory_id")
        if max_trajectories is not None:
            trajectory_table = trajectory_table.head(max_trajectories)

        trajectory_ids = trajectory_table["trajectory_id"].to_numpy(dtype=np.int64)
        parquet_path = base / "transition_features.parquet"
        columns = [*IDENTIFIER_COLUMNS, *feature_names]
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
        return torch.from_numpy(self._features[start:end]), metadata


def collate_trajectories(batch):
    batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
    sequences, metadata = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, metadata


def load_combined_datasets(
    feature_root,
    feature_set,
    datasets,
    feature_names,
    training_only=False,
    max_trajectories=None,
):
    loaded = [
        ParquetTrajectoryDataset(
            feature_root=feature_root,
            feature_set=feature_set,
            dataset=dataset,
            feature_names=feature_names,
            training_only=training_only,
            max_trajectories=max_trajectories,
        )
        for dataset in datasets
    ]
    if not loaded:
        raise ValueError("At least one dataset is required")
    return loaded[0] if len(loaded) == 1 else ConcatDataset(loaded)
