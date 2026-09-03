import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from torch.utils.data import ConcatDataset

from modeling.framework.base import ModelPlugin
from modeling.framework.data import load_combined_datasets
from modeling.evaluation.evaluation import _write_batch
from modeling.framework.registry import register_model
from modeling.framework.scaling import apply_scaler, fit_robust_scaler, load_scaler, save_scaler


def _iter_dataset(dataset):
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from _iter_dataset(child)
        return
    for index in range(len(dataset)):
        yield dataset[index]


@register_model
class IsolationForestPlugin(ModelPlugin):
    model_type = "isolation_forest"
    capabilities = {
        "trajectory_scores": True,
        "transition_scores": True,
        "feature_explanations": False,
        "explanation_type": "isolation_score",
    }

    def train(self, feature_root, model_dir):
        model_dir = self.ensure_directory(model_dir)
        training = self.config["training"]
        seed = training.get("seed", 42)

        dataset = load_combined_datasets(
            feature_root=feature_root,
            feature_set=self.config["feature_set"],
            datasets=training["datasets"],
            feature_names=self.config["features"],
            training_only=True,
            max_trajectories=training.get("max_trajectories"),
            sampling_strategy=training.get("sampling_strategy", "all"),
            seed=seed,
        )

        preprocessing = self.config.get("preprocessing", {})
        scaler = fit_robust_scaler(
            dataset,
            self.config["features"],
            quantile_range=preprocessing.get("quantile_range", [5.0, 95.0]),
            max_rows=preprocessing.get("max_scaler_rows", 500_000),
            seed=seed,
        )
        save_scaler(Path(model_dir) / "scaler.pkl", scaler)
        apply_scaler(dataset, scaler)

        all_transitions = []
        for item in _iter_dataset(dataset):
            sequence = item[0]
            seq_np = (
                sequence.numpy()
                if hasattr(sequence, "numpy")
                else np.asarray(sequence, dtype=np.float32)
            )
            if len(seq_np) > 1:
                all_transitions.append(seq_np[1:])
            else:
                all_transitions.append(seq_np)

        if not all_transitions:
            raise ValueError("No training transitions found.")

        X_train = np.vstack(all_transitions).astype(np.float32)

        architecture = self.config.get("architecture", {})
        n_estimators = architecture.get("n_estimators", 100)
        contamination = architecture.get("contamination", "auto")
        max_samples = architecture.get("max_samples", "auto")
        max_train_samples = architecture.get("max_train_samples")

        if max_train_samples and len(X_train) > max_train_samples:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(X_train), size=max_train_samples, replace=False)
            X_train = X_train[indices]

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train)

        with (Path(model_dir) / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)

        pd.DataFrame([{"epoch": 0, "loss": 0.0, "n_samples": len(X_train)}]).to_parquet(
            Path(model_dir) / "training_history.parquet", index=False
        )
        return model

    def load_model(self, model_dir):
        with (Path(model_dir) / "model.pkl").open("rb") as handle:
            return pickle.load(handle)

    def evaluate(self, feature_root, model_dir, prediction_root, datasets):
        evaluation = self.config["evaluation"]
        model = self.load_model(model_dir)
        scaler = load_scaler(Path(model_dir) / "scaler.pkl")
        summaries = []

        for dataset_name in datasets:
            dataset = load_combined_datasets(
                feature_root=feature_root,
                feature_set=self.config["feature_set"],
                datasets=[dataset_name],
                feature_names=self.config["features"],
                training_only=False,
                max_trajectories=evaluation.get("max_trajectories"),
            )
            apply_scaler(dataset, scaler)

            output_dir = Path(prediction_root) / self.config["model_id"] / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            trajectory_temp = output_dir / "trajectory_scores.parquet.tmp"
            transition_temp = output_dir / "transition_scores.parquet.tmp"

            trajectory_writer = None
            transition_writer = None
            trajectory_count = 0
            transition_count = 0

            try:
                trajectory_rows = []
                transition_rows = []
                batch_size = int(evaluation.get("write_batch_size", 2048))

                pending_items = []
                pending_transitions = []
                pending_lengths = []

                def flush_batch():
                    nonlocal trajectory_writer, transition_writer, trajectory_count, transition_count
                    nonlocal pending_items, pending_transitions, pending_lengths, trajectory_rows, transition_rows
                    if not pending_items:
                        return

                    X_batch = np.vstack(pending_transitions).astype(np.float32)
                    scores_batch = -model.score_samples(X_batch)

                    offset = 0
                    for item, length in zip(pending_items, pending_lengths):
                        metadata = item[-1]
                        step_scores = scores_batch[offset : offset + length]
                        offset += length

                        traj_score = float(step_scores[1:].mean()) if length > 1 else float(step_scores.mean())
                        trajectory_rows.append(
                            {
                                "model_id": self.config["model_id"],
                                "dataset": metadata.dataset,
                                "trajectory_id": metadata.trajectory_id,
                                "score": traj_score,
                            }
                        )

                        for idx, transition_id in enumerate(metadata.transition_ids):
                            transition_rows.append(
                                {
                                    "model_id": self.config["model_id"],
                                    "dataset": metadata.dataset,
                                    "trajectory_id": metadata.trajectory_id,
                                    "transition_order": int(metadata.transition_orders[idx]),
                                    "transition_id": int(transition_id),
                                    "score": float(step_scores[idx]),
                                }
                            )

                    trajectory_frame = pd.DataFrame(trajectory_rows)
                    transition_frame = pd.DataFrame(transition_rows)
                    trajectory_writer = _write_batch(
                        trajectory_writer, trajectory_frame, trajectory_temp
                    )
                    transition_writer = _write_batch(
                        transition_writer, transition_frame, transition_temp
                    )
                    trajectory_count += len(trajectory_frame)
                    transition_count += len(transition_frame)

                    pending_items.clear()
                    pending_transitions.clear()
                    pending_lengths.clear()
                    trajectory_rows.clear()
                    transition_rows.clear()

                for item in _iter_dataset(dataset):
                    sequence = item[0]
                    seq_np = (
                        sequence.numpy()
                        if hasattr(sequence, "numpy")
                        else np.asarray(sequence, dtype=np.float32)
                    )
                    pending_items.append(item)
                    pending_transitions.append(seq_np)
                    pending_lengths.append(len(seq_np))

                    if len(pending_items) >= batch_size:
                        flush_batch()

                flush_batch()

            finally:
                if trajectory_writer is not None:
                    trajectory_writer.close()
                if transition_writer is not None:
                    transition_writer.close()

            if trajectory_writer is None or transition_writer is None:
                raise ValueError(f"{dataset_name}: no trajectories available for evaluation")

            trajectory_temp.replace(output_dir / "trajectory_scores.parquet")
            transition_temp.replace(output_dir / "transition_scores.parquet")

            summaries.append(
                {
                    "dataset": dataset_name,
                    "trajectory_scores": trajectory_count,
                    "transition_scores": transition_count,
                }
            )

        return summaries
