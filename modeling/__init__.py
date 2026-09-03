"""Modular trajectory anomaly modeling, evaluation, and benchmarking framework."""

import sys

from .framework import (
    base,
    config,
    data,
    losses,
    registry,
    scaling,
    score_aggregation,
    torch_training,
)
from .evaluation import (
    benchmark,
    compact_predictions,
    data_leakage_audit,
    evaluation,
    experiments,
    length_bias_diagnostics,
    model_correlation,
    model_diagnostics,
    synthetic_evaluation,
)
from .labels import (
    global_trajectory_clustering,
    plausible_labels,
    review_labels,
)

# Register backward-compatibility aliases under modeling.<submodule>
sys.modules.setdefault("modeling.base", base)
sys.modules.setdefault("modeling.config", config)
sys.modules.setdefault("modeling.data", data)
sys.modules.setdefault("modeling.registry", registry)
sys.modules.setdefault("modeling.losses", losses)
sys.modules.setdefault("modeling.scaling", scaling)
sys.modules.setdefault("modeling.torch_training", torch_training)
sys.modules.setdefault("modeling.score_aggregation", score_aggregation)
sys.modules.setdefault("modeling.evaluation", evaluation)
sys.modules.setdefault("modeling.experiments", experiments)
sys.modules.setdefault("modeling.benchmark", benchmark)
sys.modules.setdefault("modeling.synthetic_evaluation", synthetic_evaluation)
sys.modules.setdefault("modeling.model_diagnostics", model_diagnostics)
sys.modules.setdefault("modeling.model_correlation", model_correlation)
sys.modules.setdefault("modeling.length_bias_diagnostics", length_bias_diagnostics)
sys.modules.setdefault("modeling.data_leakage_audit", data_leakage_audit)
sys.modules.setdefault("modeling.compact_predictions", compact_predictions)
sys.modules.setdefault("modeling.plausible_labels", plausible_labels)
sys.modules.setdefault("modeling.review_labels", review_labels)
sys.modules.setdefault("modeling.global_trajectory_clustering", global_trajectory_clustering)
