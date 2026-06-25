import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


STAGES = (
    "preprocess",
    "synthetic",
    "features",
    "stats",
    "models",
    "database",
)

DEFAULT_EXPERIMENT_CONFIG = "modeling/configs/configurable_model_matrix.json"


def real_datasets(datasets):
    return [dataset for dataset in datasets if dataset != "synthetic"]


def infer_comparison_group(experiment_config):
    with Path(experiment_config).open("r", encoding="utf-8") as handle:
        experiment = json.load(handle)
    if "matrix" in experiment:
        return experiment["matrix"].get("comparison_group")
    groups = {
        run.get("experiment", {}).get("comparison_group")
        for run in experiment.get("runs", [])
        if run.get("experiment", {}).get("comparison_group")
    }
    return next(iter(groups)) if len(groups) == 1 else None


def extend(command, *items):
    for item in items:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            command.extend(str(value) for value in item)
        else:
            command.append(str(item))
    return command


def artifact_source_arguments(args):
    if not args.extra_artifact_source:
        return None
    sources = [
        f"{args.model_root}={args.prediction_root}",
        *args.extra_artifact_source,
    ]
    result = []
    for source in sources:
        result.extend(["--artifact-source", source])
    return result


def stage_commands(args):
    python = sys.executable
    synthetic_output = str(Path(args.processed_root) / "synthetic")
    database_experiment_config = (
        "" if args.extra_artifact_source else args.experiment_config
    )
    commands = {
        "preprocess": [
            python,
            "-m",
            "dataset_preprocessing.preprocess_logs_to_parquet",
            "--output-dir",
            args.processed_root,
        ],
        "synthetic": extend(
            [
                python,
                "-m",
                "dataset_preprocessing.synthetic_anomalies",
                "--processed-root",
                args.processed_root,
                "--output-dir",
                synthetic_output,
                "--source-datasets",
            ],
            real_datasets(args.datasets),
        ),
        "features": extend(
            [
                python,
                "-m",
                "dataset_preprocessing.build_motion_features",
                "--processed-dir",
                args.processed_root,
                "--feature-set",
                args.feature_set,
                "--output-root",
                args.feature_root,
                "--run-root",
                args.run_root,
                "--datasets",
            ],
            args.datasets,
        ),
        "stats": extend(
            [
                python,
                "-m",
                "dataset_preprocessing.parquet_dataset_stats",
                "--processed-dir",
                args.processed_root,
                "--output-dir",
                args.stats_root,
                "--datasets",
            ],
            args.datasets,
        ),
        "models": extend(
            [
                python,
                "-m",
                "modeling.experiments",
                args.experiment_config,
                "--feature-root",
                args.feature_root,
                "--feature-set",
                args.feature_set,
                "--model-root",
                args.model_root,
                "--prediction-root",
                args.prediction_root,
                "--metrics",
                args.metrics,
            ],
            ["--model-id", *args.models] if args.models else None,
            ["--model-type", *args.model_types] if args.model_types else None,
            ["--train-datasets", *args.train_datasets]
            if args.train_datasets
            else None,
            ["--eval-datasets", *args.eval_datasets] if args.eval_datasets else None,
            ["--epochs", args.epochs] if args.epochs is not None else None,
        ),
        "database": extend(
            [
                python,
                "build_database.py",
                "--db-path",
                args.db_path,
                "--processed-root",
                args.processed_root,
                "--model-root",
                args.model_root,
                "--prediction-root",
                args.prediction_root,
                "--feature-root",
                args.feature_root,
                "--experiment-config",
                database_experiment_config,
                "--datasets",
            ],
            args.datasets,
            ["--comparison-group", args.comparison_group]
            if args.comparison_group
            else None,
            ["--model-id", *args.models] if args.models else None,
            ["--model-type", *args.model_types] if args.model_types else None,
            artifact_source_arguments(args),
        ),
    }
    return commands


def serve_command(args):
    return [
        sys.executable,
        "open_visualization.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--db-path",
        args.db_path,
        "--prediction-root",
        args.prediction_root,
        "--processed-root",
        args.processed_root,
        "--stats-root",
        args.stats_root,
    ]


def selected_stages(start, stop, skip):
    start_index = STAGES.index(start)
    stop_index = STAGES.index(stop)
    if start_index > stop_index:
        raise ValueError("--from-stage must precede --to-stage")
    return [
        stage
        for stage in STAGES[start_index : stop_index + 1]
        if stage not in skip
    ]


def resolve_args(args):
    if args.run_id:
        if args.model_root is None:
            args.model_root = f"artifacts/models_{args.run_id}"
        if args.prediction_root is None:
            args.prediction_root = f"artifacts/predictions_{args.run_id}"
    args.model_root = args.model_root or "artifacts/models"
    args.prediction_root = args.prediction_root or "artifacts/predictions"

    if (
        args.comparison_group is None
        and args.experiment_config
        and not args.extra_artifact_source
    ):
        args.comparison_group = infer_comparison_group(args.experiment_config)
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Run the canonical preprocessing, training, database, and visualization pipeline."
    )
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--skip", nargs="*", choices=STAGES, default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start open_visualization.py after the selected stages finish.",
    )
    parser.add_argument(
        "--run-id",
        help="Use artifacts/models_<run-id> and artifacts/predictions_<run-id> unless roots are explicitly set.",
    )
    parser.add_argument("--feature-set", default="motion_v2")
    parser.add_argument("--processed-root", default="data/processed_parquet")
    parser.add_argument("--feature-root", default="data/features")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--stats-root", default="artifacts/stats/processed_parquet")
    parser.add_argument("--db-path", default="artifacts/app/plausibility.db")
    parser.add_argument("--model-root")
    parser.add_argument("--prediction-root")
    parser.add_argument(
        "--extra-artifact-source",
        action="append",
        help=(
            "Also index an existing run in the visualization DB, formatted "
            "as model_root=prediction_root. Can be repeated."
        ),
    )
    parser.add_argument(
        "--experiment-config",
        default=DEFAULT_EXPERIMENT_CONFIG,
    )
    parser.add_argument(
        "--comparison-group",
        help="Defaults to the unique comparison_group found in --experiment-config.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["inat", "gowalla", "synthetic"],
        help="Datasets to build features/stats for, evaluate, and index.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Run and index only these model IDs from the experiment config.",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        help="Run only these model_type values from the experiment config.",
    )
    parser.add_argument(
        "--train-datasets",
        nargs="+",
        help="Override training.datasets for selected model runs.",
    )
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        help="Override evaluation.datasets for selected model runs.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--metrics",
        choices=["quick", "full", "skip"],
        default="quick",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = resolve_args(parser.parse_args())

    commands = stage_commands(args)
    stages = selected_stages(args.from_stage, args.to_stage, set(args.skip))
    for index, stage in enumerate(stages, start=1):
        command = commands[stage]
        print(f"[{index}/{len(stages)}] {stage}: {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)

    if args.serve:
        command = serve_command(args)
        print(f"serve: {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
