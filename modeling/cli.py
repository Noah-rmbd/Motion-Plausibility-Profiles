import argparse
import json
from pathlib import Path

from modeling.config import load_config, write_config
from modeling.models import LSTMAutoencoderPlugin  # noqa: F401
from modeling.registry import create_model_plugin, registered_models


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

    run_parser = subparsers.add_parser("run")
    add_training_arguments(run_parser)
    run_parser.set_defaults(handler=run_command)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

