from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from modeling.framework.base import ModelPlugin
from modeling.framework.data import collate_forecasting_trajectories, load_forecasting_datasets
from modeling.evaluation.evaluation import write_forecast_predictions
from modeling.framework.losses import concept_feature_weights
from modeling.framework.registry import register_model
from modeling.framework.scaling import apply_scaler, fit_robust_scaler, load_scaler, save_scaler
from modeling.framework.torch_training import choose_device, set_global_seed


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, sequences, lengths):
        packed = pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        encoded, _ = self.recurrent(packed)
        encoded, _ = pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=sequences.size(1),
        )
        return self.output_layer(encoded)


def masked_forecast_loss(predictions, targets, lengths, feature_weights):
    timestep_mask = torch.arange(targets.size(1), device=targets.device).unsqueeze(0)
    timestep_mask = timestep_mask < lengths.to(targets.device).unsqueeze(1)
    squared_errors = (predictions - targets) ** 2
    weights = feature_weights.to(targets.device).view(1, 1, -1)
    weighted_sum = (squared_errors * weights * timestep_mask.unsqueeze(2)).sum()
    denominator = timestep_mask.sum() * weights.sum()
    return weighted_sum / denominator, float(denominator.item())


def train_forecasting_model(model, dataloader, config, device):
    feature_weights = concept_feature_weights(config["features"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    epochs = config["training"]["epochs"]
    history = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss = 0.0
        value_count = 0
        for inputs, targets, lengths, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs, lengths)
            loss, batch_value_count = masked_forecast_loss(
                predictions,
                targets,
                lengths,
                feature_weights=feature_weights,
            )
            loss.backward()
            optimizer.step()

            weighted_loss += loss.item() * batch_value_count
            value_count += batch_value_count

        epoch_loss = weighted_loss / value_count if value_count else float("nan")
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"Epoch {epoch}/{epochs}: loss={epoch_loss:.8f}")

    return pd.DataFrame(history)


@register_model
class LSTMForecasterPlugin(ModelPlugin):
    model_type = "lstm_forecaster"
    capabilities = {
        "trajectory_scores": True,
        "transition_scores": True,
        "feature_explanations": True,
        "explanation_type": "forecast_error",
        "forecast_horizon": 1,
    }

    def build_model(self):
        architecture = self.config["architecture"]
        return LSTMForecaster(
            input_dim=len(self.config["features"]),
            hidden_dim=architecture["hidden_dim"],
            num_layers=architecture["num_layers"],
            dropout=architecture.get("dropout", 0.0),
        )

    def train(self, feature_root, model_dir):
        model_dir = self.ensure_directory(model_dir)
        training = self.config["training"]
        seed = training.get("seed", 42)
        set_global_seed(seed)
        dataset = load_forecasting_datasets(
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
            collate_fn=collate_forecasting_trajectories,
            generator=torch.Generator().manual_seed(seed),
        )
        device = choose_device(training.get("device", "auto"))
        model = self.build_model()
        history = train_forecasting_model(model, dataloader, self.config, device)
        checkpoint = {
            "model_type": self.model_type,
            "feature_names": self.config["features"],
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
        device = choose_device(
            evaluation.get("device", self.config["training"].get("device", "auto"))
        )
        model = self.load_model(model_dir, device)
        scaler = load_scaler(Path(model_dir) / "scaler.pkl")
        summaries = []

        for dataset_name in datasets:
            dataset = load_forecasting_datasets(
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
                collate_fn=collate_forecasting_trajectories,
            )
            summaries.append(
                write_forecast_predictions(
                    model=model,
                    dataloader=dataloader,
                    feature_names=self.config["features"],
                    device=device,
                    prediction_root=prediction_root,
                    model_id=self.config["model_id"],
                    dataset=dataset_name,
                )
            )
        return summaries
