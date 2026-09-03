import argparse
import json
from pathlib import Path

from modeling.framework.config import load_config, write_config
from modeling.models import (  # noqa: F401
    FrechetKernelPlugin,
    IsolationForestPlugin,
    LAGMMPlugin,
    LSTMAutoencoderPlugin,
    LSTMForecasterPlugin,
    LSTMSeq2SeqAutoencoderPlugin,
    OneClassSVMPlugin,
    TLSTMAutoencoderPlugin,
    TLSTMSeq2SeqAutoencoderPlugin,
)
from modeling.framework.registry import create_model_plugin, registered_models
from modeling.evaluation.synthetic_evaluation import evaluate_synthetic_predictions


def apply_cli_overrides(config, args):
    if getattr(args, "model_id", None):
        config["model_id"] = args.model_id
    if getattr(args, "features", None):
        config["features"] = args.features
    if getattr(args, "epochs", None) is not None:
        config["training"]["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        config["training"]["batch_size"] = args.batch_size
        config["evaluation"]["batch_size"] = args.batch_size
    if getattr(args, "max_train_trajectories", None) is not None:
        config["training"]["max_trajectories"] = args.max_train_trajectories
    if getattr(args, "max_eval_trajectories", None) is not None:
        config["evaluation"]["max_trajectories"] = args.max_eval_trajectories
    return config


def resolve_config(args):
    config = apply_cli_overrides(load_config(args.config), args)
    if "model_id" not in config:
        raise ValueError("model_id must be provided in the config or with --model-id")
    return config


def model_paths(args, config):
    model_dir = Path(args.model_root) / config["model_id"]
    return model_dir, Path(args.prediction_root)


def train_command(args):
    config = resolve_config(args)
    model_dir, _ = model_paths(args, config)
    write_config(model_dir / "config.json", config)
    plugin = create_model_plugin(config["model_type"], config)
    plugin.train(args.feature_root, model_dir)
    print(f"Saved model to {model_dir}")


def evaluate_command(args):
    model_dir = Path(args.model_root) / args.model_id
    config = load_config(model_dir / "config.json")
    config["model_id"] = args.model_id
    if args.max_eval_trajectories is not None:
        config["evaluation"]["max_trajectories"] = args.max_eval_trajectories
    datasets = args.datasets or config["evaluation"]["datasets"]
    plugin = create_model_plugin(config["model_type"], config)
    summaries = plugin.evaluate(
        feature_root=args.feature_root,
        model_dir=model_dir,
        prediction_root=args.prediction_root,
        datasets=datasets,
    )
    print(json.dumps(summaries, indent=2))


def evaluate_all_command(args):
    results = []
    failures = []
    config_paths = sorted(Path(args.model_root).glob("*/config.json"))
    for config_path in config_paths:
        model_id = config_path.parent.name
        try:
            config = load_config(config_path)
            experiment = config.get("experiment", {})
            if (
                args.comparison_group
                and experiment.get("comparison_group")
                != args.comparison_group
            ):
                continue
            if args.feature_set and config.get("feature_set") != args.feature_set:
                continue
            if args.model_type and config.get("model_type") != args.model_type:
                continue

            config["model_id"] = model_id
            if args.max_eval_trajectories is not None:
                config["evaluation"]["max_trajectories"] = (
                    args.max_eval_trajectories
                )
            datasets = args.datasets or config["evaluation"]["datasets"]
            print(f"Evaluating {model_id}: {', '.join(datasets)}", flush=True)
            plugin = create_model_plugin(config["model_type"], config)
            summaries = plugin.evaluate(
                feature_root=args.feature_root,
                model_dir=config_path.parent,
                prediction_root=args.prediction_root,
                datasets=datasets,
            )
            result = {
                "model_id": model_id,
                "predictions": summaries,
            }
            if "synthetic" in datasets and not args.skip_metrics:
                metrics = evaluate_synthetic_predictions(
                    model_id=model_id,
                    prediction_root=args.prediction_root,
                    feature_root=args.feature_root,
                    feature_set=config["feature_set"],
                    reference_datasets=args.reference_datasets,
                    ignore_first_transition=config["evaluation"].get(
                        "ignore_first_transition",
                        True,
                    ),
                )
                result["synthetic_auc"] = metrics["roc_auc"]
            results.append(result)
        except Exception as exc:
            failures.append(
                {
                    "model_id": model_id,
                    "error": str(exc),
                }
            )
            print(f"Skipped {model_id}: {exc}", flush=True)

    if not results and not failures:
        raise ValueError("No saved models matched the requested filters")
    print(
        json.dumps(
            {
                "evaluated": results,
                "failures": failures,
            },
            indent=2,
        )
    )


def run_command(args):
    train_command(args)
    evaluation_args = argparse.Namespace(
        model_id=args.model_id,
        model_root=args.model_root,
        prediction_root=args.prediction_root,
        feature_root=args.feature_root,
        datasets=args.datasets,
        max_eval_trajectories=args.max_eval_trajectories,
    )
    evaluate_command(evaluation_args)


def add_shared_paths(parser):
    parser.add_argument("--feature-root", default="data/features")
    parser.add_argument("--model-root", default="artifacts/models")
    parser.add_argument("--prediction-root", default="artifacts/predictions")


def add_training_arguments(parser):
    parser.add_argument("--config")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-trajectories", type=int)
    parser.add_argument("--max-eval-trajectories", type=int)
    parser.add_argument("--datasets", nargs="+")
    add_shared_paths(parser)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate modular trajectory anomaly models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-models")
    list_parser.set_defaults(handler=lambda args: print(json.dumps(registered_models(), indent=2)))

    train_parser = subparsers.add_parser("train")
    add_training_arguments(train_parser)
    train_parser.set_defaults(handler=train_command)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--model-id", required=True)
    evaluate_parser.add_argument("--datasets", nargs="+")
    evaluate_parser.add_argument("--max-eval-trajectories", type=int)
    add_shared_paths(evaluate_parser)
    evaluate_parser.set_defaults(handler=evaluate_command)

    evaluate_all_parser = subparsers.add_parser(
        "evaluate-all",
        help="Evaluate every saved model matching the requested filters.",
    )
    evaluate_all_parser.add_argument(
        "--datasets",
        nargs="+",
        default=["synthetic"],
    )
    evaluate_all_parser.add_argument("--comparison-group")
    evaluate_all_parser.add_argument("--feature-set")
    evaluate_all_parser.add_argument("--model-type")
    evaluate_all_parser.add_argument(
        "--reference-datasets",
        nargs="+",
        default=["inat", "gowalla"],
    )
    evaluate_all_parser.add_argument("--skip-metrics", action="store_true")
    evaluate_all_parser.add_argument("--max-eval-trajectories", type=int)
    add_shared_paths(evaluate_all_parser)
    evaluate_all_parser.set_defaults(handler=evaluate_all_command)

    run_parser = subparsers.add_parser("run")
    add_training_arguments(run_parser)
    run_parser.set_defaults(handler=run_command)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
