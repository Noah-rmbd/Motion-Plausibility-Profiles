import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler
from torch.utils.data import ConcatDataset


DEFAULT_CONTINUOUS_FEATURES = {
    "speed",
    "elapsed_time",
    "distance",
    "acceleration",
}


def _leaf_datasets(dataset):
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from _leaf_datasets(child)
        return
    yield dataset


def _sample_feature(dataset, feature_index, max_rows, seed, valid_index=None):
    arrays = []
    for leaf in _leaf_datasets(dataset):
        values = leaf._features[:, feature_index]
        if valid_index is not None:
            values = values[leaf._features[:, valid_index] >= 0.5]
        if len(values):
            arrays.append(values)
    total_rows = sum(len(array) for array in arrays)
    if total_rows == 0:
        raise ValueError("Cannot fit a scaler on an empty training population")
    if max_rows is None or total_rows <= max_rows:
        return np.concatenate(arrays).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    sampled = []
    remaining = max_rows
    for index, array in enumerate(arrays):
        if index == len(arrays) - 1:
            sample_size = min(len(array), remaining)
        else:
            proportional = round(max_rows * len(array) / total_rows)
            sample_size = min(len(array), max(1, proportional), remaining)
        if sample_size:
            row_indices = rng.choice(len(array), size=sample_size, replace=False)
            sampled.append(array[row_indices])
            remaining -= sample_size
    return np.concatenate(sampled).reshape(-1, 1)


def fit_robust_scaler(
    dataset,
    feature_names,
    quantile_range=(5.0, 95.0),
    max_rows=500_000,
    seed=42,
):
    scaled_features = [
        name for name in feature_names if name in DEFAULT_CONTINUOUS_FEATURES
    ]
    scalers = {}
    fit_row_counts = {}
    for offset, feature_name in enumerate(scaled_features):
        feature_index = feature_names.index(feature_name)
        valid_index = None
        if feature_name == "acceleration" and "acceleration_valid" in feature_names:
            valid_index = feature_names.index("acceleration_valid")
        values = _sample_feature(
            dataset,
            feature_index,
            max_rows,
            seed + offset,
            valid_index=valid_index,
        )
        scalers[feature_name] = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=tuple(quantile_range),
        ).fit(values)
        fit_row_counts[feature_name] = int(len(values))
    return {
        "type": "robust",
        "feature_names": list(feature_names),
        "scaled_features": scaled_features,
        "quantile_range": list(quantile_range),
        "max_fit_rows": max_rows,
        "fit_row_counts": fit_row_counts,
        "scalers": scalers,
    }


def apply_scaler(dataset, bundle):
    expected = list(bundle["feature_names"])
    for leaf in _leaf_datasets(dataset):
        if leaf.feature_names != expected:
            raise ValueError(
                f"Scaler features {expected} do not match dataset features {leaf.feature_names}"
            )
        scaled_features = bundle["scaled_features"]
        if not scaled_features:
            continue
        for feature_name in scaled_features:
            feature_index = expected.index(feature_name)
            leaf._features[:, feature_index] = bundle["scalers"][
                feature_name
            ].transform(
                leaf._features[:, feature_index].reshape(-1, 1)
            ).reshape(-1).astype(np.float32)
        if "acceleration" in expected and "acceleration_valid" in expected:
            acceleration_index = expected.index("acceleration")
            valid_index = expected.index("acceleration_valid")
            invalid = leaf._features[:, valid_index] < 0.5
            leaf._features[invalid, acceleration_index] = 0.0
    return dataset


def save_scaler(path, bundle):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)


def load_scaler(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
