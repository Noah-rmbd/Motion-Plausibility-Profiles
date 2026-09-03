from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from modeling.framework.base import ModelPlugin
from modeling.framework.data import collate_trajectories, load_combined_datasets
from modeling.evaluation.evaluation import write_reconstruction_predictions
from modeling.framework.registry import register_model
from modeling.framework.scaling import apply_scaler, fit_robust_scaler, load_scaler, save_scaler
from modeling.framework.torch_training import choose_device, set_global_seed, train_reconstruction_model


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, sequences, lengths):
        packed = pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, (hidden, cell) = self.encoder(packed)
        return hidden, cell

    def forward(self, sequences, lengths):
        hidden, cell = self.encode(sequences, lengths)
        latent = hidden[-1]
        decoder_input = latent.unsqueeze(1).expand(-1, sequences.size(1), -1)
        packed_decoder = pack_padded_sequence(
            decoder_input,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        decoded, _ = self.decoder(packed_decoder, (hidden, cell))
        decoded, _ = pad_packed_sequence(decoded, batch_first=True)
        return self.output_layer(decoded)


@register_model
class LSTMAutoencoderPlugin(ModelPlugin):
    model_type = "lstm_autoencoder"
    capabilities = {
        "trajectory_scores": True,
        "transition_scores": True,
        "feature_explanations": True,
        "explanation_type": "reconstruction_error",
    }

    def build_model(self):
        architecture = self.config["architecture"]
        return LSTMAutoencoder(
            input_dim=len(self.config["features"]),
            hidden_dim=architecture["hidden_dim"],
            num_layers=architecture["num_layers"],
            dropout=architecture.get("dropout", 0.0),
        )

    def context_feature_names(self):
        return []

    def collate_fn(self):
        return collate_trajectories

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
            context_feature_names=self.context_feature_names(),
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
        history = train_reconstruction_model(model, dataloader, self.config, device)

        checkpoint = {
            "model_type": self.model_type,
            "feature_names": self.config["features"],
            "context_feature_names": self.context_feature_names(),
            "architecture": self.config["architecture"],
            "preprocessing": {
                "type": scaler["type"],
                "scaled_features": scaler["scaled_features"],
                "quantile_range": scaler["quantile_range"],
            },
            "state_dict": model.state_dict(),
        }
        torch.save(checkpoint, Path(model_dir) / "model.pth")
        history.to_parquet(Path(model_dir) / "training_history.parquet", index=False)
        return model

    def load_model(self, model_dir, device):
        checkpoint = torch.load(Path(model_dir) / "model.pth", map_location=device)
        if checkpoint["model_type"] != self.model_type:
            raise ValueError(f"Expected {self.model_type}, found {checkpoint['model_type']}")
        model = self.build_model()
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)

    def evaluate(self, feature_root, model_dir, prediction_root, datasets):
        evaluation = self.config["evaluation"]
        device = choose_device(evaluation.get("device", self.config["training"].get("device", "auto")))
        model = self.load_model(model_dir, device)
        scaler = load_scaler(Path(model_dir) / "scaler.pkl")
        summaries = []

        for dataset_name in datasets:
            dataset = load_combined_datasets(
                feature_root=feature_root,
                feature_set=self.config["feature_set"],
                datasets=[dataset_name],
                feature_names=self.config["features"],
                context_feature_names=self.context_feature_names(),
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
            summary = write_reconstruction_predictions(
                model=model,
                dataloader=dataloader,
                feature_names=self.config["features"],
                device=device,
                ignore_first_transition=evaluation.get("ignore_first_transition", True),
                prediction_root=prediction_root,
                model_id=self.config["model_id"],
                dataset=dataset_name,
            )
            summaries.append(summary)
        return summaries
