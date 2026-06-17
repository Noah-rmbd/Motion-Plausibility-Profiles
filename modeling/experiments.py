import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

from modeling.config import load_config, write_config
from modeling.models import (  # noqa: F401
    LAGMMPlugin,
    LSTMAutoencoderPlugin,
    LSTMForecasterPlugin,
    LSTMSeq2SeqAutoencoderPlugin,
    TLSTMAutoencoderPlugin,
    TLSTMSeq2SeqAutoencoderPlugin,
)
from modeling.registry import create_model_plugin
from modeling.synthetic_evaluation import evaluate_synthetic_predictions


def saved_model_matches_config(model_dir, config):
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    required_artifacts = [
        model_dir / "model.pth",
        model_dir / "scaler.pkl",
        model_dir / "training_history.parquet",
    ]
    if not config_path.exists() or not all(path.exists() for path in required_artifacts):
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            saved_config = json.load(handle)
    except json.JSONDecodeError:
        return False
    training_keys = (
        "model_type",
        "feature_set",
        "features",
        "architecture",
        "preprocessing",
        "training",
        "gmm",
        "lagmm_objective",
    )
    return {
        key: saved_config.get(key)
        for key in training_keys
    } == {
        key: config.get(key)
        for key in training_keys
    }


def expanded_runs(experiment):
    if "runs" in experiment:
        return experiment["runs"]

    matrix = experiment["matrix"]
    runs = []
    for architecture in matrix["architectures"]:
        for population in matrix["populations"]:
            for feature_set in matrix["feature_combinations"]:
                model_id_parts = [
                    architecture["id"],
                    feature_set["id"],
                    population["id"],
                    f"e{matrix['epochs']}",
                ]
                model_id_prefix = architecture.get(
                    "model_id_prefix",
                    matrix.get("model_id_prefix"),
                )
                if model_id_prefix:
                    model_id_parts.insert(0, model_id_prefix)
                run = {
                    "model_id": "_".join(model_id_parts),
                    "model_type": architecture["model_type"],
                    "features": architecture.get("features", feature_set["features"]),
                    "architecture": architecture.get("architecture", {}),
                    "gmm": architecture.get("gmm", {}),
                    "lagmm_objective": architecture.get(
                        "lagmm_objective",
                        {},
                    ),
                    "training": {
                        **architecture.get("training", {}),
                        "datasets": population["datasets"],
                        "sampling_strategy": population["sampling_strategy"],
                        "epochs": matrix["epochs"],
                    },
                    "evaluation": {
                        "datasets": matrix.get(
                            "evaluation_datasets",
                            ["inat", "gowalla", "synthetic"],
                        ),
                        "ignore_first_transition": False,
                    },
                    "experiment": {
                        "feature_representation": feature_set["id"],
                        "training_population": population["id"],
                        "comparison_group": matrix.get(
                            "comparison_group",
                            "configurable_pipeline",
                        ),
                    },
                }
                runs.append(run)
    return runs


def run_experiments(
    config_path,
    feature_root,
    model_root,
    prediction_root,
    feature_set=None,
):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        experiment = json.load(handle)
    base_config = load_config(experiment.get("base_config"))
    results = []

    for run in expanded_runs(experiment):
        config = deepcopy(base_config)
        config["model_id"] = run["model_id"]
        for section in (
            "architecture",
            "gmm",
            "lagmm_objective",
            "training",
            "evaluation",
            "experiment",
        ):
            config.setdefault(section, {}).update(run.get(section, {}))
        if "model_type" in run:
            config["model_type"] = run["model_type"]
        if "features" in run:
            config["features"] = run["features"]
        if feature_set is not None:
            config["feature_set"] = feature_set

        model_dir = Path(model_root) / config["model_id"]
        source_model = run.get("source_model")
        if source_model:
            source_dir = Path(model_root) / source_model
            model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dir / "model.pth", model_dir / "model.pth")
            shutil.copy2(source_dir / "scaler.pkl", model_dir / "scaler.pkl")
            history_path = source_dir / "training_history.parquet"
            if history_path.exists():
                shutil.copy2(history_path, model_dir / "training_history.parquet")
        else:
            if saved_model_matches_config(model_dir, config):
                print(f"Reusing existing trained model: {config['model_id']}", flush=True)
            else:
                plugin = create_model_plugin(config["model_type"], config)
                plugin.train(feature_root, model_dir)

        write_config(model_dir / "config.json", config)
        plugin = create_model_plugin(config["model_type"], config)
        reference_datasets = experiment.get(
            "reference_datasets",
            ["inat", "gowalla"],
        )
        missing_reference_datasets = sorted(
            set(reference_datasets) - set(config["evaluation"]["datasets"])
        )
        if missing_reference_datasets:
            raise ValueError(
                "Synthetic evaluation reference datasets must also be present "
                "in evaluation.datasets. Missing: "
                + ", ".join(missing_reference_datasets)
            )
        predictions = plugin.evaluate(
            feature_root,
            model_dir,
            prediction_root,
            config["evaluation"]["datasets"],
        )
        metrics = evaluate_synthetic_predictions(
            model_id=config["model_id"],
            prediction_root=prediction_root,
            feature_root=feature_root,
            feature_set=config["feature_set"],
            reference_datasets=reference_datasets,
            ignore_first_transition=config["evaluation"].get(
                "ignore_first_transition",
                True,
            ),
        )
        results.append(
            {
                "model_id": config["model_id"],
                "predictions": predictions,
                "synthetic_auc": metrics["roc_auc"],
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run an explicit list of comparable model experiments."
    )
    parser.add_argument("config")
    parser.add_argument("--feature-root", default="data/features")
    parser.add_argument("--feature-set")
    parser.add_argument("--model-root", default="artifacts/models")
    parser.add_argument("--prediction-root", default="artifacts/predictions")
    args = parser.parse_args()
    results = run_experiments(
        config_path=args.config,
        feature_root=args.feature_root,
        model_root=args.model_root,
        prediction_root=args.prediction_root,
        feature_set=args.feature_set,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
