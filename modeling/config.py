import json
from copy import deepcopy
from pathlib import Path


DEFAULT_CONFIG = {
    "model_type": "lstm_autoencoder",
    "feature_set": "motion_v1",
    "features": [
        "speed",
        "elapsed_time",
        "distance",
        "acceleration",
        "bearing_sin",
        "bearing_cos",
    ],
    "architecture": {
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.0,
    },
    "training": {
        "datasets": ["inat", "gowalla"],
        "epochs": 10,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "loss": "masked_mse",
        "seed": 42,
        "device": "auto",
        "num_workers": 0,
        "max_trajectories": None,
    },
    "evaluation": {
        "datasets": ["inat", "gowalla", "synthetic"],
        "batch_size": 256,
        "device": "auto",
        "num_workers": 0,
        "max_trajectories": None,
        "ignore_first_transition": True,
    },
}


def deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_config(path=None):
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            deep_update(config, json.load(handle))
    return config


def write_config(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
