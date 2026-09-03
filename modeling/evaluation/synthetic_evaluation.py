import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from modeling.framework.score_aggregation import (
    AGGREGATION_COLUMNS,
    TRANSITION_SCORE_MODES,
    aggregate_transition_scores,
    select_trajectory_scores,
)
from modeling.labels.plausible_labels import (
    DEFAULT_LABEL_PATH,
    resolve_plausible_trajectories,
)
from modeling.framework.data import MIN_MODEL_TRAJECTORY_TRANSITIONS
from modeling.labels.review_labels import (
    DEFAULT_REVIEW_LABEL_PATH,
    resolve_reviewed_trajectories,
)


def classification_metrics(
    true_positives,
    false_positives,
    false_negatives,
    true_negatives,
):
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = 2.0 * precision * recall / f1_denominator if f1_denominator else 0.0
    negative_denominator = true_negatives + false_positives
    specificity = (
        true_negatives / negative_denominator
        if negative_denominator
        else 0.0
    )
    return {
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_negatives": int(true_negatives),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2.0),
    }


def default_anomaly_percentages():
    return np.round(np.arange(0.1, 25.0 + 0.1, 0.1), 1).tolist()


def roc_auc(reference, synthetic):
    return float(
        roc_auc_score(
            np.concatenate(
                [
                    np.zeros(len(reference), dtype=np.int8),
                    np.ones(len(synthetic), dtype=np.int8),
                ]
            ),
            np.concatenate(
                [
                    reference["score"].to_numpy(dtype=float),
                    synthetic["score"].to_numpy(dtype=float),
                ]
            ),
        )
    )


def benchmark_metrics(
    calibration,
    trusted_plausible,
    synthetic,
    anomaly_percentages,
    reviewed_unplausible=None,
):
    if calibration.empty:
        raise ValueError("Calibration population must not be empty")
    if trusted_plausible.empty:
        raise ValueError("Trusted plausible population must not be empty")
    calibration_scores = calibration["score"].to_numpy(dtype=float)
    trusted_scores = trusted_plausible["score"].to_numpy(dtype=float)
    synthetic_scores = synthetic["score"].to_numpy(dtype=float)
    if reviewed_unplausible is None:
        reviewed_unplausible = pd.DataFrame(columns=["dataset", "trajectory_id", "score"])
    reviewed_unplausible_scores = reviewed_unplausible["score"].to_numpy(dtype=float)
    result = {
        "threshold_population": "unlabeled_real_calibration",
        "calibration_trajectories": int(len(calibration)),
        "negative_population": "manually_reviewed_plausible",
        "trusted_plausible_trajectories": int(len(trusted_plausible)),
        "reviewed_unplausible_trajectories": int(len(reviewed_unplausible)),
        "roc_auc": roc_auc(trusted_plausible, synthetic),
        "manual_review_roc_auc": (
            roc_auc(trusted_plausible, reviewed_unplausible)
            if not reviewed_unplausible.empty
            else None
        ),
        "threshold_metrics": [],
    }
    for percentage in anomaly_percentages:
        if not 0.0 < percentage < 100.0:
            raise ValueError(
                f"Anomaly percentage must be between 0 and 100: {percentage}"
            )
        threshold = float(np.percentile(calibration_scores, 100.0 - percentage))
        calibration_flagged = int((calibration_scores >= threshold).sum())
        false_positives = int((trusted_scores >= threshold).sum())
        true_negatives = int((trusted_scores < threshold).sum())
        detected = synthetic_scores >= threshold
        true_positives = int(detected.sum())
        false_negatives = int((~detected).sum())
        overall = classification_metrics(
            true_positives,
            false_positives,
            false_negatives,
            true_negatives,
        )
        overall["support"] = int(len(synthetic))
        overall["false_positive_rate"] = float(
            false_positives / len(trusted_plausible)
        )

        profiles = {}
        for profile, profile_frame in synthetic.assign(detected=detected).groupby(
            "anomaly_profile",
            sort=True,
        ):
            profile_true_positives = int(profile_frame["detected"].sum())
            profile_false_negatives = int((~profile_frame["detected"]).sum())
            profile_result = classification_metrics(
                profile_true_positives,
                false_positives,
                profile_false_negatives,
                true_negatives,
            )
            profile_result["comparison"] = (
                "profile_vs_all_trusted_plausible"
            )
            profile_result["support"] = int(len(profile_frame))
            profile_result["false_positive_rate"] = float(
                false_positives / len(trusted_plausible)
            )
            profiles[profile] = profile_result

        threshold_entry = {
            "anomaly_percentage": float(percentage),
            "threshold": threshold,
            "calibration_flagged": calibration_flagged,
            "calibration_flagged_rate": float(
                calibration_flagged / len(calibration)
            ),
            "trusted_plausible": {
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_positive_rate": float(
                    false_positives / len(trusted_plausible)
                ),
                "support": int(len(trusted_plausible)),
            },
            "overall": overall,
            "profiles": profiles,
        }
        if len(reviewed_unplausible_scores):
            reviewed_detected = reviewed_unplausible_scores >= threshold
            trusted_detected = trusted_scores >= threshold
            threshold_entry["manual_review"] = {
                **classification_metrics(
                    int(reviewed_detected.sum()),
                    int(trusted_detected.sum()),
                    int((~reviewed_detected).sum()),
                    int((~trusted_detected).sum()),
                ),
                "support": int(len(reviewed_unplausible_scores)),
            }
        result["threshold_metrics"].append(threshold_entry)
    return result


def paired_source_metrics(synthetic):
    def summarize(frame):
        deltas = frame["score"] - frame["source_score"]
        return {
            "pairs": int(len(frame)),
            "anomaly_above_source_rate": float((deltas > 0).mean()),
            "delta_mean": float(deltas.mean()),
            "delta_median": float(deltas.median()),
            "delta_p05": float(deltas.quantile(0.05)),
            "delta_p95": float(deltas.quantile(0.95)),
        }

    return {
        "overall": summarize(synthetic),
        "profiles": {
            name: summarize(frame)
            for name, frame in synthetic.groupby("anomaly_profile", sort=True)
        },
        "source_datasets": {
            name: summarize(frame)
            for name, frame in synthetic.groupby("source_dataset", sort=True)
        },
    }


def evaluate_synthetic_predictions(
    model_id,
    prediction_root="artifacts/predictions",
    feature_root="data/features",
    processed_root="data/processed_parquet",
    feature_set="motion_v2",
    reference_datasets=("inat", "gowalla"),
    plausible_labels_path=DEFAULT_LABEL_PATH,
    review_labels_path=DEFAULT_REVIEW_LABEL_PATH,
    anomaly_percentages=None,
    aggregation=None,
    transition_score_mode=None,
    included_features=None,
    _write_output=True,
    ignore_first_transition=None,
):
    if aggregation is None:
        scoring_metrics = {
            score_mode: {
                first_key: {
                    name: evaluate_synthetic_predictions(
                        model_id=model_id,
                        prediction_root=prediction_root,
                        feature_root=feature_root,
                        processed_root=processed_root,
                        feature_set=feature_set,
                        reference_datasets=reference_datasets,
                        plausible_labels_path=plausible_labels_path,
                        review_labels_path=review_labels_path,
                        anomaly_percentages=anomaly_percentages,
                        aggregation=name,
                        transition_score_mode=score_mode,
                        included_features=included_features,
                        _write_output=False,
                        ignore_first_transition=first_key == "excluded",
                    )
                    for name in AGGREGATION_COLUMNS
                }
                for first_key in ("included", "excluded")
            }
            for score_mode in TRANSITION_SCORE_MODES
        }
        default_first_key = "excluded" if ignore_first_transition else "included"
        default_metrics = scoring_metrics["mean"][default_first_key]["mean"]
        metrics = {
            "metrics_schema_version": 3,
            "model_id": model_id,
            "feature_set": feature_set,
            "roc_auc": default_metrics["roc_auc"],
            "paired_source_metrics": default_metrics["paired_source_metrics"],
            "scoring": scoring_metrics,
        }
        output_path = Path(prediction_root) / model_id / "synthetic" / "metrics.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, separators=(",", ":"))
        return metrics

    if aggregation not in AGGREGATION_COLUMNS:
        raise ValueError(f"Unknown trajectory aggregation: {aggregation}")
    if transition_score_mode not in TRANSITION_SCORE_MODES:
        raise ValueError(
            f"Unknown transition score mode: {transition_score_mode}"
        )
    if ignore_first_transition is None:
        ignore_first_transition = False
    if anomaly_percentages is None:
        anomaly_percentages = default_anomaly_percentages()
    model_prediction_dir = Path(prediction_root) / model_id
    synthetic_scores = select_trajectory_scores(
        pd.read_parquet(
            model_prediction_dir / "synthetic" / "trajectory_scores.parquet",
        ),
        pd.read_parquet(
            model_prediction_dir / "synthetic" / "transition_scores.parquet",
        ),
        aggregation=aggregation,
        ignore_first_transition=ignore_first_transition,
        transition_score_mode=transition_score_mode,
        included_features=included_features,
    ).rename(columns={"selected_score": "score"})
    synthetic_trajectories = pd.read_parquet(
        Path(feature_root) / feature_set / "synthetic" / "trajectories.parquet",
        columns=[
            "trajectory_id",
            "n_transitions",
            "anomaly_profile",
            "source_dataset",
            "source_trajectory_id",
        ],
    )
    synthetic_trajectories = synthetic_trajectories.loc[
        synthetic_trajectories["n_transitions"].astype(int)
        >= MIN_MODEL_TRAJECTORY_TRANSITIONS
    ]
    synthetic = synthetic_scores.merge(
        synthetic_trajectories,
        on="trajectory_id",
        how="inner",
        validate="one_to_one",
    )
    if synthetic["anomaly_profile"].isna().any():
        raise ValueError("Synthetic prediction rows are missing anomaly profile metadata")

    reference_frames = []
    all_dataset_scores = {}
    eligible_reference_ids = {}
    for dataset in reference_datasets:
        path = model_prediction_dir / dataset / "transition_scores.parquet"
        calibration_ids = pd.read_parquet(
            Path(feature_root) / feature_set / dataset / "trajectories.parquet",
            columns=["trajectory_id", "use_for_calibration", "n_transitions"],
        )
        eligible_ids = calibration_ids.loc[
            calibration_ids["n_transitions"].astype(int)
            >= MIN_MODEL_TRAJECTORY_TRANSITIONS,
            "trajectory_id",
        ].astype(int)
        eligible_reference_ids[dataset] = set(eligible_ids.tolist())
        calibration_ids = calibration_ids.loc[
            calibration_ids["use_for_calibration"]
            & (
                calibration_ids["n_transitions"].astype(int)
                >= MIN_MODEL_TRAJECTORY_TRANSITIONS
            ),
            ["trajectory_id"],
        ]
        dataset_scores = select_trajectory_scores(
            pd.read_parquet(
                model_prediction_dir / dataset / "trajectory_scores.parquet",
            ),
            pd.read_parquet(path),
            aggregation=aggregation,
            ignore_first_transition=ignore_first_transition,
            transition_score_mode=transition_score_mode,
            included_features=included_features,
        ).rename(columns={"selected_score": "score"})
        dataset_scores = dataset_scores.loc[
            dataset_scores["trajectory_id"]
            .astype(int)
            .isin(eligible_reference_ids[dataset])
        ]
        all_dataset_scores[dataset] = dataset_scores
        reference_frames.append(
            dataset_scores.merge(
                calibration_ids,
                on="trajectory_id",
                how="inner",
                validate="one_to_one",
            ).assign(dataset=dataset)
        )
    reference = pd.concat(reference_frames, ignore_index=True)
    eligible_source_mask = pd.Series(False, index=synthetic.index)
    synthetic_source_ids = pd.to_numeric(
        synthetic["source_trajectory_id"],
        errors="coerce",
    )
    for dataset, ids in eligible_reference_ids.items():
        eligible_source_mask |= (
            synthetic["source_dataset"].astype(str).eq(str(dataset))
            & synthetic_source_ids.isin(ids)
        )
    synthetic = synthetic.loc[eligible_source_mask].reset_index(drop=True)
    if reference.empty or synthetic.empty:
        raise ValueError("Reference and synthetic prediction sets must both be non-empty")

    plausible_labels = resolve_plausible_trajectories(
        feature_root=feature_root,
        processed_root=processed_root,
        feature_set=feature_set,
        label_path=plausible_labels_path,
    )
    reviewed_labels = resolve_reviewed_trajectories(
        feature_root=feature_root,
        processed_root=processed_root,
        feature_set=feature_set,
        label_path=review_labels_path,
        labels={"plausible", "unplausible"},
    )
    trusted_frames = []
    for dataset, labels in plausible_labels.groupby("dataset", sort=True):
        labels = labels.loc[
            labels["trajectory_id"].astype(int).isin(
                eligible_reference_ids.get(dataset, set())
            )
        ]
        trusted_frames.append(
            all_dataset_scores[dataset].merge(
                labels[["trajectory_id"]],
                on="trajectory_id",
                how="inner",
                validate="one_to_one",
            ).assign(dataset=dataset)
        )
    reviewed_unplausible_frames = []
    if not reviewed_labels.empty:
        for dataset, labels in reviewed_labels.groupby("dataset", sort=True):
            labels = labels.loc[
                labels["trajectory_id"].astype(int).isin(
                    eligible_reference_ids.get(dataset, set())
                )
            ]
            plausible = labels.loc[labels["label"].eq("plausible")]
            unplausible = labels.loc[labels["label"].eq("unplausible")]
            if not plausible.empty:
                trusted_frames.append(
                    all_dataset_scores[dataset].merge(
                        plausible[["trajectory_id"]],
                        on="trajectory_id",
                        how="inner",
                        validate="one_to_one",
                    ).assign(dataset=dataset)
                )
            if not unplausible.empty:
                reviewed_unplausible_frames.append(
                    all_dataset_scores[dataset].merge(
                        unplausible[["trajectory_id"]],
                        on="trajectory_id",
                        how="inner",
                        validate="one_to_one",
                    ).assign(dataset=dataset)
                )
    source_frames = []
    for dataset, metadata in synthetic.groupby("source_dataset", sort=True):
        source_ids = metadata[["source_trajectory_id"]].drop_duplicates()
        source_frames.append(
            all_dataset_scores[dataset].merge(
                source_ids,
                left_on="trajectory_id",
                right_on="source_trajectory_id",
                how="inner",
                validate="one_to_one",
            )[["trajectory_id", "score"]].assign(dataset=dataset)
        )
    source_reference = pd.concat(source_frames, ignore_index=True)
    expected_source_count = len(
        synthetic[
            ["source_dataset", "source_trajectory_id"]
        ].drop_duplicates()
    )
    if len(source_reference) != expected_source_count:
        raise ValueError(
            f"Expected {expected_source_count} source controls, found "
            f"{len(source_reference)}"
        )

    trusted_reference = pd.concat(trusted_frames, ignore_index=True)
    reviewed_unplausible_reference = (
        pd.concat(reviewed_unplausible_frames, ignore_index=True)
        if reviewed_unplausible_frames
        else trusted_reference.iloc[0:0].copy()
    )
    trusted_ids = trusted_reference[["dataset", "trajectory_id"]].drop_duplicates()
    source_ids = source_reference[["dataset", "trajectory_id"]].drop_duplicates()
    reviewed_unplausible_ids = reviewed_unplausible_reference[
        ["dataset", "trajectory_id"]
    ].drop_duplicates()
    protected_reference_ids = pd.concat(
        [
            trusted_ids.assign(_protected="trusted_plausible"),
            source_ids.assign(_protected="synthetic_source_control"),
            reviewed_unplausible_ids.assign(_protected="reviewed_unplausible"),
        ],
        ignore_index=True,
    ).drop_duplicates(["dataset", "trajectory_id"])
    reference = (
        reference.merge(
            protected_reference_ids,
            on=["dataset", "trajectory_id"],
            how="left",
            validate="one_to_one",
        )
        .loc[lambda frame: frame["_protected"].isna()]
        .drop(columns="_protected")
        .reset_index(drop=True)
    )
    if reference.empty:
        raise ValueError(
            "Calibration population is empty after excluding trusted plausible "
            "and synthetic source-control trajectories"
        )
    synthetic = synthetic.merge(
        source_reference.rename(
            columns={
                "dataset": "source_dataset",
                "trajectory_id": "source_trajectory_id",
                "score": "source_score",
            }
        ),
        on=["source_dataset", "source_trajectory_id"],
        how="left",
        validate="many_to_one",
    )
    if synthetic["source_score"].isna().any():
        raise ValueError("Synthetic trajectories are missing source control scores")

    benchmark = benchmark_metrics(
        reference,
        trusted_reference,
        synthetic,
        anomaly_percentages,
        reviewed_unplausible=reviewed_unplausible_reference,
    )
    metrics = {
        "metrics_schema_version": 3,
        "model_id": model_id,
        "aggregation": aggregation,
        "transition_score_mode": transition_score_mode,
        "transition_score_features": list(included_features or []),
        "first_transition": (
            "excluded" if ignore_first_transition else "included"
        ),
        "feature_set": feature_set,
        "reference_datasets": list(reference_datasets),
        "threshold_population": benchmark["threshold_population"],
        "threshold_excludes_trusted_plausible": True,
        "threshold_excludes_synthetic_sources": True,
        "negative_population": benchmark["negative_population"],
        "paired_source_metrics": paired_source_metrics(synthetic),
        "calibration_trajectories": benchmark["calibration_trajectories"],
        "trusted_references": (
            plausible_labels.groupby(["dataset", "user_id"], sort=True)
            .size()
            .rename("trajectory_count")
            .reset_index()
            .to_dict("records")
        ),
        "trusted_reference_trajectories": len(trusted_reference),
        "reviewed_unplausible_trajectories": benchmark[
            "reviewed_unplausible_trajectories"
        ],
        "synthetic_trajectories": len(synthetic),
        "roc_auc": benchmark["roc_auc"],
        "manual_review_roc_auc": benchmark["manual_review_roc_auc"],
        "synthetic_score_mean": float(synthetic["score"].mean()),
        "synthetic_score_median": float(synthetic["score"].median()),
        "threshold_metrics": benchmark["threshold_metrics"],
    }

    if _write_output:
        output_path = model_prediction_dir / "synthetic" / "metrics.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, separators=(",", ":"))
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic anomaly scores against real reference scores."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prediction-root", default="artifacts/predictions")
    parser.add_argument("--model-root", default="artifacts/models")
    parser.add_argument("--feature-root", default="data/features")
    parser.add_argument("--processed-root", default="data/processed_parquet")
    parser.add_argument("--feature-set", default="motion_v2")
    parser.add_argument("--reference-datasets", nargs="+", default=["inat", "gowalla"])
    parser.add_argument(
        "--plausible-labels",
        default=str(DEFAULT_LABEL_PATH),
    )
    parser.add_argument(
        "--review-labels",
        default=str(DEFAULT_REVIEW_LABEL_PATH),
    )
    parser.add_argument("--anomaly-percentages", nargs="+", type=float)
    parser.add_argument(
        "--ignore-first-transition",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    ignore_first_transition = args.ignore_first_transition
    if ignore_first_transition is None:
        config_path = Path(args.model_root) / args.model_id / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                model_config = json.load(handle)
            ignore_first_transition = model_config.get("evaluation", {}).get(
                "ignore_first_transition",
                True,
            )
        else:
            ignore_first_transition = True

    metrics = evaluate_synthetic_predictions(
        model_id=args.model_id,
        prediction_root=args.prediction_root,
        feature_root=args.feature_root,
        processed_root=args.processed_root,
        feature_set=args.feature_set,
        reference_datasets=args.reference_datasets,
        plausible_labels_path=args.plausible_labels,
        review_labels_path=args.review_labels,
        anomaly_percentages=args.anomaly_percentages,
        ignore_first_transition=ignore_first_transition,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
