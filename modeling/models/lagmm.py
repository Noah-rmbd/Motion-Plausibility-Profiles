import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from modeling.framework.base import ModelPlugin
from modeling.framework.data import collate_trajectories, load_combined_datasets
from modeling.evaluation.evaluation import _score_batch, _write_batch
from modeling.framework.losses import concept_feature_weights, get_loss
from modeling.framework.registry import register_model
from modeling.framework.scaling import (
    apply_scaler,
    fit_robust_scaler,
    load_scaler,
    save_scaler,
)
from modeling.framework.torch_training import choose_device, set_global_seed


IMPLEMENTATION = "paper_joint_lagmm_v1"


class LAGMM(nn.Module):
    """Joint LSTM autoencoder and neural GMM estimation network."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        latent_dim=1,
        compression_dims=(64, 32, 16),
        estimation_hidden_dim=10,
        n_components=4,
        estimation_dropout=0.5,
    ):
        super().__init__()
        recurrent_dropout = 0.0
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        encoder_dims = [hidden_dim, *compression_dims, latent_dim]
        self.compression = self._mlp(encoder_dims, final_activation=False)
        decoder_dims = [latent_dim, *reversed(compression_dims), hidden_dim]
        self.decompression = self._mlp(decoder_dims, final_activation=False)
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.estimation = nn.Sequential(
            nn.Linear(latent_dim + 2, estimation_hidden_dim),
            nn.Tanh(),
            nn.Dropout(estimation_dropout),
            nn.Linear(estimation_hidden_dim, n_components),
            nn.Softmax(dim=1),
        )

    @staticmethod
    def _mlp(dimensions, final_activation):
        layers = []
        for index, (input_dim, output_dim) in enumerate(
            zip(dimensions, dimensions[1:])
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2 or final_activation:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def encode(self, sequences, lengths):
        packed = pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, (hidden, _) = self.encoder(packed)
        return self.compression(hidden[-1])

    def decode(self, latent, lengths, max_length):
        context = self.decompression(latent)
        decoder_input = context.unsqueeze(1).expand(-1, max_length, -1)
        packed = pack_padded_sequence(
            decoder_input,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        decoded, _ = self.decoder(packed)
        decoded, _ = pad_packed_sequence(
            decoded,
            batch_first=True,
            total_length=max_length,
        )
        return self.output_layer(decoded)

    def forward(self, sequences, lengths):
        latent = self.encode(sequences, lengths)
        return self.decode(latent, lengths, sequences.size(1))

    def representation(self, sequences, lengths, reconstructed=None):
        if reconstructed is None:
            reconstructed = self(sequences, lengths)
        latent = self.encode(sequences, lengths)
        descriptors = []
        for index, length_value in enumerate(lengths):
            length = int(length_value.item())
            target = sequences[index, :length].reshape(-1)
            prediction = reconstructed[index, :length].reshape(-1)
            relative_distance = (
                (target - prediction).norm()
                / target.norm().clamp_min(1e-8)
            )
            cosine_similarity = F.cosine_similarity(
                target.unsqueeze(0),
                prediction.unsqueeze(0),
                dim=1,
                eps=1e-8,
            ).squeeze(0)
            descriptors.append(
                torch.stack([relative_distance, cosine_similarity])
            )
        return torch.cat([latent, torch.stack(descriptors)], dim=1)

    def estimate_membership(self, representations):
        return self.estimation(representations)


def estimate_gmm_parameters(representations, membership, covariance_epsilon=1e-6):
    component_mass = membership.sum(dim=0).clamp_min(covariance_epsilon)
    phi = component_mass / membership.size(0)
    means = (
        membership.transpose(0, 1) @ representations
        / component_mass.unsqueeze(1)
    )
    differences = representations.unsqueeze(1) - means.unsqueeze(0)
    covariance = torch.einsum(
        "nk,nkd,nke->kde",
        membership,
        differences,
        differences,
    )
    covariance = covariance / component_mass.view(-1, 1, 1)
    identity = torch.eye(
        representations.size(1),
        device=representations.device,
        dtype=representations.dtype,
    )
    covariance = covariance + covariance_epsilon * identity.unsqueeze(0)
    return phi, means, covariance


def sample_energy(representations, phi, means, covariance):
    differences = representations.unsqueeze(1) - means.unsqueeze(0)
    precision = torch.linalg.inv(covariance)
    mahalanobis = torch.einsum(
        "nkd,kde,nke->nk",
        differences,
        precision,
        differences,
    )
    _, log_determinant = torch.linalg.slogdet(covariance)
    dimensions = representations.size(1)
    component_log_probability = (
        torch.log(phi.clamp_min(1e-12)).unsqueeze(0)
        - 0.5
        * (
            mahalanobis
            + log_determinant.unsqueeze(0)
            + dimensions * np.log(2.0 * np.pi)
        )
    )
    return -torch.logsumexp(component_log_probability, dim=1)


def covariance_diagonal_penalty(covariance):
    diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
    return torch.reciprocal(diagonal.clamp_min(1e-12)).sum()


def extract_joint_outputs(model, dataloader, device):
    representations = []
    memberships = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for sequences, lengths, _ in dataloader:
            sequences = sequences.to(device)
            reconstructed = model(sequences, lengths)
            batch_representations = model.representation(
                sequences,
                lengths,
                reconstructed,
            )
            representations.append(batch_representations.cpu())
            memberships.append(
                model.estimate_membership(batch_representations).cpu()
            )
    if not representations:
        raise ValueError("Cannot estimate a Gaussian mixture without trajectories")
    return torch.cat(representations), torch.cat(memberships)


def train_joint_lagmm(model, dataloader, config, device):
    training = config["training"]
    objective = config.get("lagmm_objective", {})
    reconstruction_loss = get_loss(training["loss"])
    feature_weights = concept_feature_weights(config["features"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.0),
    )
    energy_weight = objective.get("energy_weight", 0.1)
    covariance_weight = objective.get("covariance_weight", 0.005)
    covariance_epsilon = objective.get("covariance_epsilon", 1e-6)
    history = []
    model.to(device)

    for epoch in range(1, training["epochs"] + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "energy": 0.0,
            "covariance_penalty": 0.0,
        }
        trajectory_count = 0
        for sequences, lengths, _ in dataloader:
            sequences = sequences.to(device)
            optimizer.zero_grad()
            reconstructed = model(sequences, lengths)
            representations = model.representation(
                sequences,
                lengths,
                reconstructed,
            )
            membership = model.estimate_membership(representations)
            phi, means, covariance = estimate_gmm_parameters(
                representations,
                membership,
                covariance_epsilon,
            )
            reconstruction, _ = reconstruction_loss(
                reconstructed,
                sequences,
                lengths,
                feature_weights=feature_weights,
            )
            energy = sample_energy(
                representations,
                phi,
                means,
                covariance,
            ).mean()
            covariance_penalty = covariance_diagonal_penalty(covariance)
            loss = (
                reconstruction
                + energy_weight * energy
                + covariance_weight * covariance_penalty
            )
            loss.backward()
            optimizer.step()

            batch_size = len(sequences)
            trajectory_count += batch_size
            totals["loss"] += float(loss.item()) * batch_size
            totals["reconstruction_loss"] += (
                float(reconstruction.item()) * batch_size
            )
            totals["energy"] += float(energy.item()) * batch_size
            totals["covariance_penalty"] += (
                float(covariance_penalty.item()) * batch_size
            )

        row = {
            "epoch": epoch,
            **{
                name: value / trajectory_count
                for name, value in totals.items()
            },
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{training['epochs']}: "
            f"loss={row['loss']:.8f} "
            f"reconstruction={row['reconstruction_loss']:.8f} "
            f"energy={row['energy']:.8f}"
        )
    return pd.DataFrame(history)


@register_model
class LAGMMPlugin(ModelPlugin):
    model_type = "lagmm"
    capabilities = {
        "trajectory_scores": True,
        "transition_scores": True,
        "feature_explanations": True,
        "explanation_type": "reconstruction_error",
        "native_trajectory_score": True,
        "trajectory_score_type": "gmm_energy",
        "joint_model": True,
    }

    def collate_fn(self):
        return collate_trajectories

    def build_model(self):
        architecture = self.config["architecture"]
        gmm = self.config.get("gmm", {})
        return LAGMM(
            input_dim=len(self.config["features"]),
            hidden_dim=architecture.get("hidden_dim", 128),
            num_layers=architecture.get("num_layers", 1),
            latent_dim=architecture.get("latent_dim", 1),
            compression_dims=tuple(
                architecture.get("compression_dims", [64, 32, 16])
            ),
            estimation_hidden_dim=architecture.get(
                "estimation_hidden_dim",
                10,
            ),
            n_components=gmm.get("n_components", 4),
            estimation_dropout=architecture.get(
                "estimation_dropout",
                0.5,
            ),
        )

    def train(self, feature_root, model_dir):
        model_dir = self.ensure_directory(model_dir)
        training = self.config["training"]
        seed = training.get("seed", 42)
        set_global_seed(seed)
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
        dataloader = DataLoader(
            dataset,
            batch_size=training["batch_size"],
            shuffle=True,
            num_workers=training.get("num_workers", 0),
            collate_fn=self.collate_fn(),
            generator=torch.Generator().manual_seed(seed),
        )
        device = choose_device(training.get("device", "auto"))
        model = self.build_model()
        history = train_joint_lagmm(model, dataloader, self.config, device)

        checkpoint = {
            "model_type": self.model_type,
            "implementation": IMPLEMENTATION,
            "feature_names": self.config["features"],
            "architecture": self.config["architecture"],
            "state_dict": model.state_dict(),
        }
        torch.save(checkpoint, Path(model_dir) / "model.pth")
        history.to_parquet(
            Path(model_dir) / "training_history.parquet",
            index=False,
        )

        fit_loader = DataLoader(
            dataset,
            batch_size=training["batch_size"],
            shuffle=False,
            num_workers=training.get("num_workers", 0),
            collate_fn=self.collate_fn(),
        )
        representations, membership = extract_joint_outputs(
            model,
            fit_loader,
            device,
        )
        phi, means, covariance = estimate_gmm_parameters(
            representations,
            membership,
            self.config.get("lagmm_objective", {}).get(
                "covariance_epsilon",
                1e-6,
            ),
        )
        artifact = {
            "implementation": IMPLEMENTATION,
            "representation": (
                "one_dimensional_latent_relative_distance_cosine_similarity"
            ),
            "phi": phi.numpy(),
            "means": means.numpy(),
            "covariance": covariance.numpy(),
        }
        with (Path(model_dir) / "gmm.pkl").open("wb") as handle:
            pickle.dump(artifact, handle)
        energies = sample_energy(
            representations,
            phi,
            means,
            covariance,
        )
        diagnostics = {
            "model_family": "joint_lstm_autoencoder_neural_gmm",
            "implementation": IMPLEMENTATION,
            "fit_trajectories": int(len(representations)),
            "representation_dimensions": int(representations.shape[1]),
            "n_components": int(len(phi)),
            "energy_mean": float(energies.mean()),
            "energy_std": float(energies.std()),
        }
        with (Path(model_dir) / "gmm_diagnostics.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(diagnostics, handle, indent=2)
        return model

    def load_model(self, model_dir, device):
        checkpoint = torch.load(
            Path(model_dir) / "model.pth",
            map_location=device,
        )
        if checkpoint.get("implementation") != IMPLEMENTATION:
            raise ValueError(
                "This checkpoint belongs to the previous two-stage LAGMM "
                "implementation and must be retrained."
            )
        model = self.build_model()
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)

    def load_gmm(self, model_dir):
        with (Path(model_dir) / "gmm.pkl").open("rb") as handle:
            artifact = pickle.load(handle)
        if artifact.get("implementation") != IMPLEMENTATION:
            raise ValueError(
                "This GMM artifact belongs to the previous two-stage LAGMM "
                "implementation and must be retrained."
            )
        return artifact

    def evaluate(self, feature_root, model_dir, prediction_root, datasets):
        evaluation = self.config["evaluation"]
        device = choose_device(
            evaluation.get(
                "device",
                self.config["training"].get("device", "auto"),
            )
        )
        model = self.load_model(model_dir, device)
        scaler = load_scaler(Path(model_dir) / "scaler.pkl")
        gmm_artifact = self.load_gmm(model_dir)
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
            dataloader = DataLoader(
                dataset,
                batch_size=evaluation["batch_size"],
                shuffle=False,
                num_workers=evaluation.get("num_workers", 0),
                collate_fn=self.collate_fn(),
            )
            summaries.append(
                self.write_predictions(
                    model,
                    dataloader,
                    gmm_artifact,
                    device,
                    prediction_root,
                    dataset_name,
                )
            )
        return summaries

    def write_predictions(
        self,
        model,
        dataloader,
        gmm_artifact,
        device,
        prediction_root,
        dataset,
    ):
        output_dir = Path(prediction_root) / self.config["model_id"] / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = output_dir / "trajectory_scores.parquet"
        transition_path = output_dir / "transition_scores.parquet"
        trajectory_temp = trajectory_path.with_suffix(".parquet.tmp")
        transition_temp = transition_path.with_suffix(".parquet.tmp")
        trajectory_writer = None
        transition_writer = None
        trajectory_count = 0
        transition_count = 0
        phi = torch.as_tensor(gmm_artifact["phi"], device=device)
        means = torch.as_tensor(gmm_artifact["means"], device=device)
        covariance = torch.as_tensor(
            gmm_artifact["covariance"],
            device=device,
        )

        model.eval()
        model.to(device)
        try:
            with torch.no_grad():
                for sequences, lengths, metadata_batch in dataloader:
                    sequences = sequences.to(device)
                    reconstructed = model(sequences, lengths)
                    representations = model.representation(
                        sequences,
                        lengths,
                        reconstructed,
                    )
                    energy = sample_energy(
                        representations,
                        phi,
                        means,
                        covariance,
                    )
                    membership = model.estimate_membership(representations)
                    components = membership.argmax(dim=1)
                    trajectory_scores, transition_scores = _score_batch(
                        sequences=sequences,
                        lengths=lengths,
                        metadata_batch=metadata_batch,
                        reconstructed=reconstructed,
                        feature_names=self.config["features"],
                        ignore_first_transition=self.config["evaluation"].get(
                            "ignore_first_transition",
                            False,
                        ),
                        model_id=self.config["model_id"],
                    )
                    trajectory_scores["reconstruction_score"] = (
                        trajectory_scores["score"]
                    )
                    trajectory_scores["score"] = (
                        energy.cpu().numpy().astype(np.float32)
                    )
                    trajectory_scores["score_type"] = "gmm_energy"
                    trajectory_scores["gmm_component"] = (
                        components.cpu().numpy().astype(np.int16)
                    )
                    trajectory_scores["gmm_component_probability"] = (
                        membership.max(dim=1).values.cpu().numpy()
                        .astype(np.float32)
                    )
                    trajectory_writer = _write_batch(
                        trajectory_writer,
                        trajectory_scores,
                        trajectory_temp,
                    )
                    transition_writer = _write_batch(
                        transition_writer,
                        transition_scores,
                        transition_temp,
                    )
                    trajectory_count += len(trajectory_scores)
                    transition_count += len(transition_scores)
        finally:
            if trajectory_writer is not None:
                trajectory_writer.close()
            if transition_writer is not None:
                transition_writer.close()
        if trajectory_writer is None or transition_writer is None:
            raise ValueError(f"{dataset}: no trajectories available for evaluation")
        trajectory_temp.replace(trajectory_path)
        transition_temp.replace(transition_path)
        return {
            "dataset": dataset,
            "trajectory_scores": trajectory_count,
            "transition_scores": transition_count,
            "score_type": "gmm_energy",
        }
