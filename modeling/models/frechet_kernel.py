import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset

from modeling.framework.base import ModelPlugin
from modeling.framework.data import load_combined_datasets
from modeling.evaluation.evaluation import _write_batch
from modeling.framework.registry import register_model
from modeling.framework.scaling import apply_scaler, fit_robust_scaler, load_scaler, save_scaler


def _load_pyfrk():
    try:
        return importlib.import_module("pyfrk")
    except ImportError:
        root = Path(__file__).resolve().parents[2]
        version = f"{sys.version_info.major}{sys.version_info.minor}"
        candidates = [
            root
            / "frk/frk_anonymous/build"
            / f"lib.macosx-12.1-arm64-cpython-{version}",
        ]
        for candidate in candidates:
            if candidate.exists():
                sys.path.insert(0, str(candidate))
                try:
                    return importlib.import_module("pyfrk")
                except ImportError:
                    continue
    return None


def _iter_dataset(dataset):
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from _iter_dataset(child)
        return
    for index in range(len(dataset)):
        yield dataset[index]


def _compact_sequence(sequence, max_length):
    sequence = np.asarray(sequence, dtype=np.float32)
    if max_length is None or len(sequence) <= max_length:
        return sequence
    indices = np.linspace(0, len(sequence) - 1, max_length).round().astype(int)
    return sequence[indices]


def _squared_distances(x, y):
    diff = x[:, None, :] - y[None, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff, dtype=np.float64)


def _logaddexp3(a, b, c):
    return np.logaddexp(np.logaddexp(a, b), c)


def _walk_log_counts(max_rows, max_cols):
    counts = np.full((max_rows, max_cols), -np.inf, dtype=np.float64)
    counts[0, :] = 0.0
    counts[:, 0] = 0.0
    for i in range(1, max_rows):
        for j in range(1, max_cols):
            counts[i, j] = _logaddexp3(
                counts[i - 1, j],
                counts[i, j - 1],
                counts[i - 1, j - 1],
            )
    return counts


def _sample_alignment_values(emat, log_counts, rng, diag_wgt):
    i = emat.shape[0] - 1
    j = emat.shape[1] - 1
    values = []
    while i > 0 and j > 0:
        values.append(float(emat[i, j]))
        pr_up = float(np.exp(log_counts[i - 1, j] - log_counts[i, j]))
        pr_left = float(np.exp(log_counts[i, j - 1] - log_counts[i, j]))
        pr_diag = max(0.0, 1.0 - pr_up - pr_left) * diag_wgt
        total = pr_up + pr_left + pr_diag
        draw = rng.random() * total
        if draw <= pr_up:
            i -= 1
        elif draw <= pr_up + pr_left:
            j -= 1
        else:
            i -= 1
            j -= 1
    while i > 0:
        values.append(float(emat[i, j]))
        i -= 1
    while j > 0:
        values.append(float(emat[i, j]))
        j -= 1
    values.append(float(emat[0, 0]))
    return values


def _python_frk_kernel(distance_matrix, gamma, beta, samples, diag_wgt, seed):
    emat = np.exp(-distance_matrix / gamma)
    log_counts = _walk_log_counts(*emat.shape)
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(samples):
        values = np.asarray(
            _sample_alignment_values(emat, log_counts, rng, diag_wgt),
            dtype=np.float64,
        )
        weights = np.exp(-beta * values)
        total += float(np.sum(values * weights) / np.sum(weights))
    return total / samples


class FrechetKernelScorer:
    def __init__(self, max_length, use_extension=True):
        self.pyfrk = _load_pyfrk() if use_extension else None
        self.kernel = self.pyfrk.Kernel(max_length) if self.pyfrk else None

    def __call__(self, x, y, gamma, beta, samples, diag_wgt, seed):
        distance_matrix = _squared_distances(x, y)
        if self.pyfrk is None:
            return _python_frk_kernel(
                distance_matrix,
                gamma=gamma,
                beta=beta,
                samples=samples,
                diag_wgt=diag_wgt,
                seed=seed,
            )
        emat = self.pyfrk.MatrixF64(distance_matrix.shape[0], distance_matrix.shape[1])
        similarities = np.exp(-distance_matrix / gamma)
        for i in range(similarities.shape[0]):
            for j in range(similarities.shape[1]):
                emat.set(i, j, float(similarities[i, j]))
        return float(
            self.kernel.compute(
                emat=emat,
                nsamples=samples,
                beta=beta,
                diag_wgt=diag_wgt,
                seed=seed,
            )
        )


def _nearest_feature_errors(sequence, landmark):
    distances = _squared_distances(sequence, landmark)
    nearest = distances.argmin(axis=1)
    diff = sequence - landmark[nearest]
    return diff * diff


@register_model
class FrechetKernelPlugin(ModelPlugin):
    model_type = "frechet_kernel"
    capabilities = {
        "trajectory_scores": True,
        "transition_scores": True,
        "feature_explanations": True,
        "explanation_type": "nearest_landmark_transition_error",
    }

    def architecture(self):
        architecture = dict(self.config.get("architecture", {}))
        architecture.setdefault("landmarks", 64)
        architecture.setdefault("candidate_landmarks", 8)
        architecture.setdefault("max_sequence_length", 96)
        architecture.setdefault("gamma", 1.0)
        architecture.setdefault("beta", 1.0)
        architecture.setdefault("samples", 100)
        architecture.setdefault("diag_wgt", 1.0)
        architecture.setdefault("score", "one_minus_max_normalized_kernel")
        architecture.setdefault("trajectory_score", "frechet_kernel_dissimilarity")
        architecture.setdefault("use_pyfrk_extension", True)
        return architecture

    def train(self, feature_root, model_dir):
        model_dir = self.ensure_directory(model_dir)
        training = self.config["training"]
        architecture = self.architecture()
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

        items = list(_iter_dataset(dataset))
        if not items:
            raise ValueError("Frechet kernel training requires at least one trajectory")
        rng = np.random.default_rng(seed)
        landmark_count = min(int(architecture["landmarks"]), len(items))
        selected_indices = np.sort(
            rng.choice(len(items), size=landmark_count, replace=False)
        )
        landmarks = []
        max_sequence_length = architecture["max_sequence_length"]
        for index in selected_indices:
            sequence, metadata = items[int(index)]
            full_sequence = sequence.numpy().astype(np.float32)
            landmarks.append(
                {
                    "dataset": metadata.dataset,
                    "trajectory_id": metadata.trajectory_id,
                    "sequence": _compact_sequence(full_sequence, max_sequence_length),
                    "full_sequence": full_sequence,
                    "summary": full_sequence.mean(axis=0),
                }
            )

        scorer = FrechetKernelScorer(
            max_length=max(len(item["sequence"]) for item in landmarks),
            use_extension=architecture.get("use_pyfrk_extension", True),
        )
        self_kernels = []
        for offset, landmark in enumerate(landmarks):
            self_kernels.append(
                scorer(
                    landmark["sequence"],
                    landmark["sequence"],
                    gamma=architecture["gamma"],
                    beta=architecture["beta"],
                    samples=architecture["samples"],
                    diag_wgt=architecture["diag_wgt"],
                    seed=seed + offset,
                )
            )

        state = {
            "model_type": self.model_type,
            "feature_names": list(self.config["features"]),
            "feature_set": self.config["feature_set"],
            "architecture": architecture,
            "landmarks": landmarks,
            "landmark_self_kernels": self_kernels,
        }
        with (Path(model_dir) / "model.pth").open("wb") as handle:
            pickle.dump(state, handle)
        pd.DataFrame(
            [
                {
                    "epoch": 0,
                    "loss": 0.0,
                    "landmarks": landmark_count,
                }
            ]
        ).to_parquet(Path(model_dir) / "training_history.parquet", index=False)
        return state

    def load_model(self, model_dir):
        with (Path(model_dir) / "model.pth").open("rb") as handle:
            state = pickle.load(handle)
        if state["model_type"] != self.model_type:
            raise ValueError(f"Expected {self.model_type}, found {state['model_type']}")
        return state

    def score_sequence(self, sequence, metadata, state, scorer, seed):
        architecture = state["architecture"]
        compact = _compact_sequence(sequence, architecture["max_sequence_length"])
        self_kernel = scorer(
            compact,
            compact,
            gamma=architecture["gamma"],
            beta=architecture["beta"],
            samples=architecture["samples"],
            diag_wgt=architecture["diag_wgt"],
            seed=seed,
        )
        similarities = []
        summaries = np.vstack([item["summary"] for item in state["landmarks"]])
        summary_distance = np.sum((summaries - compact.mean(axis=0)) ** 2, axis=1)
        candidate_count = min(
            int(architecture.get("candidate_landmarks", len(state["landmarks"]))),
            len(state["landmarks"]),
        )
        candidate_indices = np.argsort(summary_distance)[:candidate_count]
        for index in candidate_indices:
            landmark = state["landmarks"][int(index)]
            kernel_value = scorer(
                compact,
                landmark["sequence"],
                gamma=architecture["gamma"],
                beta=architecture["beta"],
                samples=architecture["samples"],
                diag_wgt=architecture["diag_wgt"],
                seed=seed + index + 1,
            )
            denominator = np.sqrt(
                max(self_kernel, 1e-12)
                * max(state["landmark_self_kernels"][index], 1e-12)
            )
            similarities.append((int(index), kernel_value / denominator))
        best_index, best_similarity = max(similarities, key=lambda item: item[1])
        best_similarity = float(np.clip(best_similarity, 0.0, 1.0))
        score = float(1.0 - best_similarity)
        feature_errors = _nearest_feature_errors(
            sequence,
            state["landmarks"][best_index]["full_sequence"],
        )
        transition_scores = feature_errors.mean(axis=1)
        trajectory_row = {
            "model_id": self.config["model_id"],
            "dataset": metadata.dataset,
            "trajectory_id": metadata.trajectory_id,
            "score": score,
            "score_type": "frechet_kernel_dissimilarity",
            "kernel_similarity": best_similarity,
            "nearest_landmark_index": best_index,
        }
        transition_rows = []
        for offset, transition_id in enumerate(metadata.transition_ids):
            row = {
                "model_id": self.config["model_id"],
                "dataset": metadata.dataset,
                "trajectory_id": metadata.trajectory_id,
                "transition_order": int(metadata.transition_orders[offset]),
                "transition_id": int(transition_id),
                "score": float(transition_scores[offset]),
            }
            for feature_index, feature_name in enumerate(state["feature_names"]):
                row[f"error_{feature_name}"] = float(
                    feature_errors[offset, feature_index]
                )
            transition_rows.append(row)
        return trajectory_row, transition_rows

    def evaluate(self, feature_root, model_dir, prediction_root, datasets):
        evaluation = self.config["evaluation"]
        architecture = self.architecture()
        state = self.load_model(model_dir)
        scaler = load_scaler(Path(model_dir) / "scaler.pkl")
        summaries = []
        seed = self.config["training"].get("seed", 42)

        max_kernel_length = max(
            architecture["max_sequence_length"],
            max(len(item["sequence"]) for item in state["landmarks"]),
        )
        scorer = FrechetKernelScorer(
            max_length=max_kernel_length,
            use_extension=architecture.get("use_pyfrk_extension", True),
        )
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
                batch_size = int(evaluation.get("write_batch_size", 512))
                for index, (sequence, metadata) in enumerate(_iter_dataset(dataset)):
                    trajectory_row, rows = self.score_sequence(
                        sequence.numpy().astype(np.float32),
                        metadata,
                        state,
                        scorer,
                        seed=seed + index,
                    )
                    trajectory_rows.append(trajectory_row)
                    transition_rows.extend(rows)
                    if len(trajectory_rows) >= batch_size:
                        trajectory_frame = pd.DataFrame(trajectory_rows)
                        transition_frame = pd.DataFrame(transition_rows)
                        trajectory_writer = _write_batch(
                            trajectory_writer,
                            trajectory_frame,
                            trajectory_temp,
                        )
                        transition_writer = _write_batch(
                            transition_writer,
                            transition_frame,
                            transition_temp,
                        )
                        trajectory_count += len(trajectory_frame)
                        transition_count += len(transition_frame)
                        trajectory_rows = []
                        transition_rows = []
                if trajectory_rows:
                    trajectory_frame = pd.DataFrame(trajectory_rows)
                    transition_frame = pd.DataFrame(transition_rows)
                    trajectory_writer = _write_batch(
                        trajectory_writer,
                        trajectory_frame,
                        trajectory_temp,
                    )
                    transition_writer = _write_batch(
                        transition_writer,
                        transition_frame,
                        transition_temp,
                    )
                    trajectory_count += len(trajectory_frame)
                    transition_count += len(transition_frame)
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
                    "score_type": "frechet_kernel_dissimilarity",
                }
            )
        return summaries
