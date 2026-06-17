import argparse
import itertools
import json
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
from modeling.synthetic_evaluation import default_anomaly_percentages, evaluate_synthetic_predictions


def run_benchmark(config_path, feature_root, model_root, prediction_root):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        benchmark = json.load(handle)

    base_config = load_config(benchmark.get("base_config"))
    results = []
    for feature_group, epochs, batch_size in itertools.product(
        benchmark["feature_groups"].items(),
        benchmark["epochs"],
        benchmark["batch_sizes"],
    ):
        feature_name, features = feature_group
        model_id = f"{benchmark['name']}_{feature_name}_e{epochs}_b{batch_size}"
        config = deepcopy(base_config)
        config["model_id"] = model_id
        config["features"] = features
        config["training"]["epochs"] = epochs
        config["training"]["batch_size"] = batch_size
        config["evaluation"]["batch_size"] = batch_size

        model_dir = Path(model_root) / model_id
        write_config(model_dir / "config.json", config)
        plugin = create_model_plugin(config["model_type"], config)
        plugin.train(feature_root, model_dir)
        summaries = plugin.evaluate(
            feature_root,
            model_dir,
            prediction_root,
            config["evaluation"]["datasets"],
        )
        result = {"model_id": model_id, "predictions": summaries}
        evaluated_datasets = set(config["evaluation"]["datasets"])
        reference_datasets = benchmark.get("reference_datasets", ["inat", "gowalla"])
        if "synthetic" in evaluated_datasets and set(reference_datasets) <= evaluated_datasets:
            result["synthetic_metrics"] = evaluate_synthetic_predictions(
                model_id=model_id,
                prediction_root=prediction_root,
                feature_root=feature_root,
                feature_set=config["feature_set"],
                reference_datasets=reference_datasets,
                anomaly_percentages=benchmark.get(
                    "anomaly_percentages",
                    default_anomaly_percentages(),
                ),
                ignore_first_transition=config["evaluation"].get(
                    "ignore_first_transition",
                    True,
                ),
            )
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run a model configuration grid.")
    parser.add_argument("config")
    parser.add_argument("--feature-root", default="data/features")
    parser.add_argument("--model-root", default="artifacts/models")
    parser.add_argument("--prediction-root", default="artifacts/predictions")
    args = parser.parse_args()

    results = run_benchmark(
        config_path=args.config,
        feature_root=args.feature_root,
        model_root=args.model_root,
        prediction_root=args.prediction_root,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
