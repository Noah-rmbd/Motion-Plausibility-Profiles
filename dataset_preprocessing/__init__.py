"""Modular data preprocessing, feature engineering, and synthetic anomaly generation."""

import sys

from .ingestion import extract_diverse_log, preprocess_logs_to_parquet
from .features import (
    build_global_trajectory_features,
    build_motion_features,
    cleaning,
    sessionization,
    splitting,
)
from .synthetic import generate_synthetic_anomalies, synthetic_anomalies
from .stats import parquet_dataset_stats, raw_dataset_stats
from .utils import utils

# Register backward-compatibility aliases under dataset_preprocessing.<submodule>
sys.modules.setdefault("dataset_preprocessing.preprocess_logs_to_parquet", preprocess_logs_to_parquet)
sys.modules.setdefault("dataset_preprocessing.extract_diverse_log", extract_diverse_log)
sys.modules.setdefault("dataset_preprocessing.build_motion_features", build_motion_features)
sys.modules.setdefault("dataset_preprocessing.build_global_trajectory_features", build_global_trajectory_features)
sys.modules.setdefault("dataset_preprocessing.sessionization", sessionization)
sys.modules.setdefault("dataset_preprocessing.cleaning", cleaning)
sys.modules.setdefault("dataset_preprocessing.splitting", splitting)
sys.modules.setdefault("dataset_preprocessing.synthetic_anomalies", synthetic_anomalies)
sys.modules.setdefault("dataset_preprocessing.generate_synthetic_anomalies", generate_synthetic_anomalies)
sys.modules.setdefault("dataset_preprocessing.parquet_dataset_stats", parquet_dataset_stats)
sys.modules.setdefault("dataset_preprocessing.raw_dataset_stats", raw_dataset_stats)
sys.modules.setdefault("dataset_preprocessing.utils", utils)
