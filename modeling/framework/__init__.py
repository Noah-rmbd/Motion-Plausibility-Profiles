"""Core modeling framework components: base plugin, dataset collation, losses, scaling, and training."""

from .base import ModelPlugin
from .config import load_config, write_config
from .registry import create_model_plugin, register_model, registered_models
from .losses import concept_feature_weights, get_loss, masked_concept_mse, masked_mse
from .scaling import apply_scaler, fit_robust_scaler, load_scaler, save_scaler
from .torch_training import choose_device, set_global_seed, train_reconstruction_model
from .score_aggregation import (
    aggregate_transition_scores,
    select_trajectory_scores,
    transition_error_scores,
)
from .data import (
    MIN_MODEL_TRAJECTORY_TRANSITIONS,
    collate_trajectories,
    collate_context_trajectories,
    collate_forecasting_trajectories,
    load_combined_datasets,
    load_forecasting_datasets,
    validate_training_trajectories,
)
