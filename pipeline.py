import argparse
import shlex
import subprocess
import sys


STAGES = (
    "preprocess",
    "synthetic",
    "features",
    "stats",
    "models",
    "database",
)


def stage_commands(args):
    python = sys.executable
    return {
        "preprocess": [
            python,
            "Dataset_Preprocessing/preprocess_logs_to_parquet.py",
        ],
        "synthetic": [
            python,
            "-m",
            "dataset_preprocessing.synthetic_anomalies",
        ],
        "features": [
            python,
            "-m",
            "dataset_preprocessing.build_motion_features",
            "--feature-set",
            args.feature_set,
        ],
        "stats": [
            python,
            "Dataset_Preprocessing/parquet_dataset_stats.py",
        ],
        "models": [
            python,
            "-m",
            "modeling.experiments",
            args.experiment_config,
            "--feature-set",
            args.feature_set,
        ],
        "database": [
            python,
            "build_database.py",
            "--comparison-group",
            args.comparison_group,
        ],
    }


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


def main():
    parser = argparse.ArgumentParser(
        description="Run the canonical preprocessing, training, and visualization pipeline."
    )
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--skip", nargs="*", choices=STAGES, default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--feature-set", default="motion_v2")
    parser.add_argument(
        "--experiment-config",
        default="modeling/configs/configurable_model_matrix.json",
    )
    parser.add_argument(
        "--comparison-group",
        default="configurable_pipeline_v2",
    )
    args = parser.parse_args()

    commands = stage_commands(args)
    stages = selected_stages(args.from_stage, args.to_stage, set(args.skip))
    for index, stage in enumerate(stages, start=1):
        command = commands[stage]
        print(f"[{index}/{len(stages)}] {stage}: {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
