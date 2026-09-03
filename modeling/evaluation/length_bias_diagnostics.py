import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from modeling.labels.plausible_labels import DEFAULT_LABEL_PATH, resolve_plausible_trajectories
from modeling.framework.score_aggregation import select_trajectory_scores, transition_error_scores
from modeling.evaluation.synthetic_evaluation import classification_metrics


TRANSITION_FEATURES = {
    "speed": "speed_kmh",
    "acceleration": "acceleration_m_s2",
    "distance": "distance_m",
    "elapsed_time": "elapsed_time_s",
    "bearing_change": "bearing_change_rad",
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_configs(model_root, comparison_group, model_type=None):
    for config_path in sorted(Path(model_root).glob("*/config.json")):
        config = load_json(config_path)
        if (
            comparison_group
            and config.get("experiment", {}).get("comparison_group") != comparison_group
        ):
            continue
        if model_type and config.get("model_type") != model_type:
            continue
        yield config_path.parent.name, config


def synthetic_predictions_are_current(model_id, prediction_root, feature_root, feature_set):
    score_path = Path(prediction_root) / model_id / "synthetic" / "trajectory_scores.parquet"
    if not score_path.exists():
        return False
    feature_dataset_dir = Path(feature_root) / feature_set / "synthetic"
    feature_paths = [
        feature_dataset_dir / "trajectories.parquet",
        feature_dataset_dir / "transition_features.parquet",
    ]
    return not any(
        path.exists() and path.stat().st_mtime > score_path.stat().st_mtime
        for path in feature_paths
    )


def read_selected_scores(
    model_id,
    dataset,
    prediction_root,
    aggregation,
    transition_score_mode,
    include_first_transition,
):
    model_dir = Path(prediction_root) / model_id / dataset
    trajectory_path = model_dir / "trajectory_scores.parquet"
    transition_path = model_dir / "transition_scores.parquet"
    if not trajectory_path.exists() or not transition_path.exists():
        return None, None
    trajectory_scores = pd.read_parquet(trajectory_path)
    transition_scores = pd.read_parquet(transition_path)
    selected = select_trajectory_scores(
        trajectory_scores,
        transition_scores,
        aggregation=aggregation,
        ignore_first_transition=not include_first_transition,
        transition_score_mode=transition_score_mode,
    ).rename(columns={"selected_score": "score"})
    return selected, transition_scores


def attach_trajectory_lengths(scores, feature_root, feature_set, dataset):
    trajectories = pd.read_parquet(
        Path(feature_root) / feature_set / dataset / "trajectories.parquet",
        columns=["trajectory_id", "n_transitions", "use_for_calibration"],
    )
    return scores.merge(
        trajectories,
        on="trajectory_id",
        how="inner",
        validate="one_to_one",
    )


def spearman(values, scores):
    frame = pd.DataFrame({"length": values, "score": scores}).dropna()
    if frame["length"].nunique() < 2 or frame["score"].nunique() < 2:
        return np.nan
    return float(frame["length"].corr(frame["score"], method="spearman"))


def pearson_log_length(values, scores):
    frame = pd.DataFrame({"length": values, "score": scores}).dropna()
    if frame["length"].nunique() < 2 or frame["score"].nunique() < 2:
        return np.nan
    return float(np.log1p(frame["length"]).corr(frame["score"], method="pearson"))


def bucket_by_quantiles(lengths, q=4):
    unique = pd.Series(lengths).dropna().nunique()
    if unique < 2:
        return pd.Series(["all"] * len(lengths), index=lengths.index)
    bins = min(q, unique)
    return pd.qcut(lengths, q=bins, duplicates="drop").astype(str)


def threshold_at(scores, anomaly_percentage):
    return float(np.percentile(scores.to_numpy(dtype=float), 100.0 - anomaly_percentage))


def synthetic_bucket_metrics(
    synthetic,
    trusted_reference,
    threshold,
    model_id,
    score_context,
):
    rows = []
    positives = synthetic.assign(label=1)
    negatives = trusted_reference.assign(label=0, anomaly_profile="trusted_plausible")
    evaluation = pd.concat([positives, negatives], ignore_index=True, sort=False)
    evaluation["length_bucket"] = bucket_by_quantiles(evaluation["n_transitions"], q=4)
    for bucket, bucket_frame in evaluation.groupby("length_bucket", sort=False):
        pos = bucket_frame.loc[bucket_frame["label"] == 1]
        neg = bucket_frame.loc[bucket_frame["label"] == 0]
        detected = bucket_frame["score"] >= threshold
        true_positives = int(((bucket_frame["label"] == 1) & detected).sum())
        false_positives = int(((bucket_frame["label"] == 0) & detected).sum())
        false_negatives = int(((bucket_frame["label"] == 1) & ~detected).sum())
        true_negatives = int(((bucket_frame["label"] == 0) & ~detected).sum())
        metrics = classification_metrics(
            true_positives,
            false_positives,
            false_negatives,
            true_negatives,
        )
        auc = np.nan
        if len(pos) and len(neg) and bucket_frame["score"].nunique() > 1:
            auc = float(roc_auc_score(bucket_frame["label"], bucket_frame["score"]))
        rows.append(
            {
                **score_context,
                "model_id": model_id,
                "length_bucket": bucket,
                "length_min": int(bucket_frame["n_transitions"].min()),
                "length_max": int(bucket_frame["n_transitions"].max()),
                "support": int(len(bucket_frame)),
                "synthetic_support": int(len(pos)),
                "trusted_plausible_support": int(len(neg)),
                "roc_auc": auc,
                **metrics,
            }
        )
    return rows


def flagged_transition_feature_rows(
    model_id,
    dataset,
    transition_scores,
    selected_scores,
    feature_root,
    feature_set,
    processed_root,
    anomaly_percentage,
    transition_score_mode,
    include_first_transition,
    score_context,
):
    trajectories = pd.read_parquet(
        Path(feature_root) / feature_set / dataset / "trajectories.parquet",
        columns=["trajectory_id", "use_for_calibration"],
    )
    calibration_ids = set(
        trajectories.loc[
            trajectories["use_for_calibration"],
            "trajectory_id",
        ].astype(int)
    )
    transition_frame = transition_scores.copy()
    transition_frame["selected_score"] = transition_error_scores(
        transition_frame,
        mode=transition_score_mode,
    )
    if not include_first_transition:
        transition_frame = transition_frame.loc[
            transition_frame["transition_order"] > 0
        ].copy()
    calibration_transition_scores = transition_frame.loc[
        transition_frame["trajectory_id"].isin(calibration_ids),
        "selected_score",
    ]
    transition_threshold = threshold_at(
        calibration_transition_scores,
        anomaly_percentage,
    )
    transition_frame["flagged_transition"] = transition_frame["selected_score"].ge(
        transition_threshold
    )
    transition_ids = transition_frame["transition_id"].astype(int).tolist()
    physical = pd.read_parquet(
        Path(processed_root) / dataset / "transitions.parquet",
        columns=["transition_id", *TRANSITION_FEATURES.values()],
        filters=[("transition_id", "in", transition_ids)],
    )
    transition_frame = transition_frame.merge(
        physical,
        on="transition_id",
        how="left",
        validate="one_to_one",
    )
    rows = []
    for feature_name, column in TRANSITION_FEATURES.items():
        all_values = transition_frame[column].dropna()
        flagged_values = transition_frame.loc[
            transition_frame["flagged_transition"],
            column,
        ].dropna()
        if flagged_values.empty or all_values.empty:
            continue
        rows.append(
            {
                **score_context,
                "model_id": model_id,
                "dataset": dataset,
                "feature": feature_name,
                "threshold_level": "transition",
                "transition_threshold": transition_threshold,
                "all_count": int(len(all_values)),
                "flagged_count": int(len(flagged_values)),
                "flagged_rate": float(len(flagged_values) / len(all_values)),
                "all_mean": float(all_values.mean()),
                "flagged_mean": float(flagged_values.mean()),
                "mean_lift": (
                    float(flagged_values.mean() / all_values.mean())
                    if all_values.mean() != 0
                    else np.nan
                ),
                "all_median": float(all_values.median()),
                "flagged_median": float(flagged_values.median()),
                "all_p95": float(all_values.quantile(0.95)),
                "flagged_p95": float(flagged_values.quantile(0.95)),
            }
        )
    return rows, transition_frame


def plot_score_vs_length(frame, output_path, title):
    plot_frame = frame.copy()
    if len(plot_frame) > 3000:
        plot_frame = plot_frame.sample(3000, random_state=42)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=plot_frame,
        x="n_transitions",
        y="score",
        hue="dataset",
        alpha=0.35,
        linewidth=0,
        s=14,
    )
    plt.xscale("log")
    plt.yscale("symlog")
    plt.title(title)
    plt.xlabel("Trajectory transitions, log scale")
    plt.ylabel("Selected anomaly score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_flagged_feature_distributions(frame, output_path, title):
    features = ["speed_kmh", "acceleration_m_s2", "distance_m", "elapsed_time_s"]
    plot_frame = frame.melt(
        id_vars=["flagged_transition"],
        value_vars=[feature for feature in features if feature in frame],
        var_name="feature",
        value_name="value",
    ).dropna()
    if plot_frame.empty:
        return
    plot_frame["population"] = np.where(
        plot_frame["flagged_transition"],
        "flagged transitions",
        "all transitions",
    )
    if len(plot_frame) > 20000:
        plot_frame = plot_frame.sample(20000, random_state=42)
    grid = sns.displot(
        data=plot_frame,
        x="value",
        hue="population",
        col="feature",
        col_wrap=2,
        kind="kde",
        common_norm=False,
        facet_kws={"sharex": False, "sharey": False},
        height=3.0,
    )
    grid.figure.suptitle(title)
    grid.figure.tight_layout()
    grid.figure.savefig(output_path, dpi=160)
    plt.close(grid.figure)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    correlation_rows = []
    bucket_rows = []
    transition_rows = []
    real_score_frames = []
    transition_plot_frames = []
    score_context = {
        "aggregation": args.aggregation,
        "transition_score_mode": args.transition_score_mode,
        "include_first_transition": args.include_first_transition,
        "anomaly_percentage": args.anomaly_percentage,
    }

    for model_id, config in model_configs(
        args.model_root,
        args.comparison_group,
        model_type=args.model_type,
    ):
        feature_set = config["feature_set"]
        if not synthetic_predictions_are_current(
            model_id,
            args.prediction_root,
            args.feature_root,
            feature_set,
        ):
            continue
        metrics_path = (
            Path(args.prediction_root) / model_id / "synthetic" / "metrics.json"
        )
        if not metrics_path.exists():
            continue
        model_summary = {
            **score_context,
            "model_id": model_id,
            "model_type": config["model_type"],
            "feature_set": feature_set,
            "training_datasets": "+".join(config["training"]["datasets"]),
            "sampling_strategy": config["training"].get("sampling_strategy", "all"),
        }
        real_frames = []
        dataset_scores = {}
        transition_score_tables = {}
        for dataset in args.reference_datasets:
            scores, transition_scores = read_selected_scores(
                model_id,
                dataset,
                args.prediction_root,
                args.aggregation,
                args.transition_score_mode,
                args.include_first_transition,
            )
            if scores is None:
                continue
            scores = attach_trajectory_lengths(
                scores,
                args.feature_root,
                feature_set,
                dataset,
            )
            scores["dataset"] = dataset
            dataset_scores[dataset] = scores
            transition_score_tables[dataset] = transition_scores
            real_frames.append(scores)
            correlation_rows.append(
                {
                    **model_summary,
                    "dataset": dataset,
                    "score_length_spearman": spearman(
                        scores["n_transitions"],
                        scores["score"],
                    ),
                    "score_log_length_pearson": pearson_log_length(
                        scores["n_transitions"],
                        scores["score"],
                    ),
                    "n_trajectories": int(len(scores)),
                    "length_median": float(scores["n_transitions"].median()),
                    "score_median": float(scores["score"].median()),
                }
            )
        if real_frames:
            combined_real = pd.concat(real_frames, ignore_index=True)
            real_score_frames.append(
                combined_real.assign(
                    model_id=model_id,
                    model_type=config["model_type"],
                )
            )
            correlation_rows.append(
                {
                    **model_summary,
                    "dataset": "+".join(args.reference_datasets),
                    "score_length_spearman": spearman(
                        combined_real["n_transitions"],
                        combined_real["score"],
                    ),
                    "score_log_length_pearson": pearson_log_length(
                        combined_real["n_transitions"],
                        combined_real["score"],
                    ),
                    "n_trajectories": int(len(combined_real)),
                    "length_median": float(combined_real["n_transitions"].median()),
                    "score_median": float(combined_real["score"].median()),
                }
            )

        synthetic_scores, _ = read_selected_scores(
            model_id,
            "synthetic",
            args.prediction_root,
            args.aggregation,
            args.transition_score_mode,
            args.include_first_transition,
        )
        if synthetic_scores is None:
            continue
        synthetic_features = pd.read_parquet(
            Path(args.feature_root) / feature_set / "synthetic" / "trajectories.parquet",
            columns=[
                "trajectory_id",
                "n_transitions",
                "anomaly_profile",
                "source_dataset",
                "source_trajectory_id",
            ],
        )
        synthetic = synthetic_scores.merge(
            synthetic_features,
            on="trajectory_id",
            how="inner",
            validate="one_to_one",
        )
        plausible_labels = resolve_plausible_trajectories(
            feature_root=args.feature_root,
            processed_root=args.processed_root,
            feature_set=feature_set,
            label_path=args.plausible_labels,
        )
        trusted_frames = []
        for dataset, labels in plausible_labels.groupby("dataset", sort=True):
            if dataset not in dataset_scores:
                continue
            trusted_frames.append(
                dataset_scores[dataset].merge(
                    labels[["trajectory_id"]],
                    on="trajectory_id",
                    how="inner",
                    validate="one_to_one",
                ).assign(reference_dataset=dataset)
            )
        if not trusted_frames:
            continue
        trusted = pd.concat(trusted_frames, ignore_index=True)
        calibration_frames = []
        for dataset, scores in dataset_scores.items():
            trusted_ids = set(
                trusted.loc[
                    trusted["reference_dataset"] == dataset,
                    "trajectory_id",
                ].astype(int)
            )
            calibration_frames.append(
                scores.loc[
                    scores["use_for_calibration"]
                    & ~scores["trajectory_id"].isin(trusted_ids)
                ]
            )
        calibration = pd.concat(calibration_frames, ignore_index=True)
        if calibration.empty:
            continue
        trajectory_threshold = threshold_at(
            calibration["score"],
            args.anomaly_percentage,
        )
        bucket_rows.extend(
            synthetic_bucket_metrics(
                synthetic=synthetic,
                trusted_reference=trusted,
                threshold=trajectory_threshold,
                model_id=model_id,
                score_context=model_summary,
            )
        )
        for dataset in args.reference_datasets:
            if dataset not in transition_score_tables or dataset not in dataset_scores:
                continue
            rows, transition_frame = flagged_transition_feature_rows(
                model_id=model_id,
                dataset=dataset,
                transition_scores=transition_score_tables[dataset],
                selected_scores=dataset_scores[dataset],
                feature_root=args.feature_root,
                feature_set=feature_set,
                processed_root=args.processed_root,
                anomaly_percentage=args.anomaly_percentage,
                transition_score_mode=args.transition_score_mode,
                include_first_transition=args.include_first_transition,
                score_context=model_summary,
            )
            transition_rows.extend(rows)
            if model_id == args.example_model_id and dataset == args.example_dataset:
                transition_plot_frames.append(transition_frame)

        if model_id == args.example_model_id and real_frames:
            plot_score_vs_length(
                pd.concat(real_frames, ignore_index=True),
                plot_dir / f"{model_id}_score_vs_length.png",
                f"{model_id}: score vs trajectory length",
            )

    correlation_frame = pd.DataFrame(correlation_rows)
    bucket_frame = pd.DataFrame(bucket_rows)
    transition_frame = pd.DataFrame(transition_rows)
    correlation_frame.to_csv(output_dir / "score_length_correlations.csv", index=False)
    bucket_frame.to_csv(output_dir / "length_bucket_metrics.csv", index=False)
    transition_frame.to_csv(
        output_dir / "flagged_transition_feature_distributions.csv",
        index=False,
    )

    if not correlation_frame.empty:
        top = correlation_frame.loc[
            correlation_frame["dataset"] == "+".join(args.reference_datasets)
        ].sort_values("score_length_spearman", ascending=False)
        top.to_csv(output_dir / "score_length_correlations_ranked.csv", index=False)
        plt.figure(figsize=(9, max(5, min(14, len(top) * 0.24))))
        sns.barplot(
            data=top.head(args.plot_top_n),
            x="score_length_spearman",
            y="model_id",
            hue="model_type",
            dodge=False,
        )
        plt.axvline(0.0, color="black", linewidth=0.8)
        plt.title("Trajectory score vs length Spearman correlation")
        plt.xlabel("Spearman rho")
        plt.ylabel("Model")
        plt.tight_layout()
        plt.savefig(plot_dir / "score_length_correlation_top.png", dpi=160)
        plt.close()

    if not bucket_frame.empty:
        plt.figure(figsize=(9, max(5, min(14, bucket_frame["model_id"].nunique() * 0.24))))
        top_auc = (
            bucket_frame.groupby("model_id", sort=False)["roc_auc"]
            .mean()
            .sort_values(ascending=False)
            .head(args.plot_top_n)
            .index
        )
        plot_frame = bucket_frame.loc[bucket_frame["model_id"].isin(top_auc)]
        sns.lineplot(
            data=plot_frame,
            x="length_bucket",
            y="f1_score",
            hue="model_id",
            marker="o",
        )
        plt.xticks(rotation=30, ha="right")
        plt.title("Synthetic benchmark F1 by trajectory length bucket")
        plt.tight_layout()
        plt.savefig(plot_dir / "length_bucket_f1_top.png", dpi=160)
        plt.close()

    if transition_plot_frames:
        plot_flagged_feature_distributions(
            pd.concat(transition_plot_frames, ignore_index=True),
            plot_dir / f"{args.example_model_id}_flagged_transition_features.png",
            f"{args.example_model_id}: flagged transitions vs all transitions",
        )

    summary = {
        "output_dir": str(output_dir),
        "correlation_rows": int(len(correlation_frame)),
        "length_bucket_rows": int(len(bucket_frame)),
        "transition_distribution_rows": int(len(transition_frame)),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Quantify trajectory length bias and transition-level flagged distributions."
    )
    parser.add_argument("--model-root", default="artifacts/models")
    parser.add_argument("--prediction-root", default="artifacts/predictions")
    parser.add_argument("--feature-root", default="data/features")
    parser.add_argument("--processed-root", default="data/processed_parquet")
    parser.add_argument("--output-dir", default="artifacts/diagnostics/length_bias")
    parser.add_argument("--comparison-group", default="configurable_pipeline_v2")
    parser.add_argument("--model-type")
    parser.add_argument("--reference-datasets", nargs="+", default=["inat", "gowalla"])
    parser.add_argument("--plausible-labels", default=str(DEFAULT_LABEL_PATH))
    parser.add_argument("--aggregation", default="mean")
    parser.add_argument("--transition-score-mode", default="mean")
    parser.add_argument(
        "--include-first-transition",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--anomaly-percentage", type=float, default=5.0)
    parser.add_argument(
        "--example-model-id",
        default="v2_lstm_ae_mpp_distance_time_bearing_inat_e15",
    )
    parser.add_argument("--example-dataset", default="inat")
    parser.add_argument("--plot-top-n", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
