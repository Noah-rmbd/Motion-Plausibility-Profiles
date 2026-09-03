import math

import torch
from torch import nn

from modeling.framework.data import collate_context_trajectories
from modeling.models.lstm_autoencoder import LSTMAutoencoderPlugin
from modeling.framework.registry import register_model


class TimeAwareLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_decomposition = nn.Linear(hidden_dim, hidden_dim)
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    @staticmethod
    def inverse_log_decay(elapsed_time):
        return 1.0 / torch.log(math.e + elapsed_time)

    def forward(self, inputs, elapsed_time, state):
        hidden, cell = state
        short_memory = torch.tanh(self.memory_decomposition(cell))
        decay = self.inverse_log_decay(elapsed_time).expand_as(short_memory)
        adjusted_cell = cell - short_memory + short_memory * decay

        input_gate, forget_gate, candidate, output_gate = self.gates(
            torch.cat([inputs, hidden], dim=1)
        ).chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        candidate = torch.tanh(candidate)
        output_gate = torch.sigmoid(output_gate)

        cell = forget_gate * adjusted_cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class TimeAwareLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TimeAwareLSTMCell(
                    input_dim if layer_index == 0 else hidden_dim,
                    hidden_dim,
                )
                for layer_index in range(num_layers)
            ]
        )

    def forward(self, sequences, lengths, elapsed_times, initial_state=None):
        batch_size, max_length, _ = sequences.shape
        if initial_state is None:
            states = [
                (
                    sequences.new_zeros(batch_size, self.hidden_dim),
                    sequences.new_zeros(batch_size, self.hidden_dim),
                )
                for _ in range(self.num_layers)
            ]
        else:
            states = list(initial_state)

        outputs = []
        for timestep in range(max_length):
            layer_input = sequences[:, timestep]
            elapsed_time = elapsed_times[:, timestep]
            active = (timestep < lengths.to(sequences.device)).unsqueeze(1)
            next_states = []
            for layer_index, cell in enumerate(self.layers):
                previous_hidden, previous_cell = states[layer_index]
                hidden, memory = cell(
                    layer_input,
                    elapsed_time,
                    (previous_hidden, previous_cell),
                )
                hidden = torch.where(active, hidden, previous_hidden)
                memory = torch.where(active, memory, previous_cell)
                next_states.append((hidden, memory))
                layer_input = hidden
                if layer_index < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            states = next_states
            outputs.append(torch.where(active, layer_input, torch.zeros_like(layer_input)))

        return torch.stack(outputs, dim=1), states


class TLSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout=0.0,
        time_scale_seconds=3600.0,
        decay_function="inverse_log",
    ):
        super().__init__()
        if time_scale_seconds <= 0:
            raise ValueError("time_scale_seconds must be positive")
        if decay_function != "inverse_log":
            raise ValueError(f"Unsupported T-LSTM decay function: {decay_function}")
        self.time_scale_seconds = float(time_scale_seconds)
        self.encoder = TimeAwareLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
        )
        self.decoder = TimeAwareLSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def elapsed_hours(self, elapsed_time_context):
        log_seconds = elapsed_time_context[..., :1].clamp_min(0.0)
        seconds = torch.expm1(log_seconds)
        return seconds / self.time_scale_seconds

    def forward(self, sequences, lengths, context):
        elapsed_time = self.elapsed_hours(context)
        _, encoder_state = self.encoder(sequences, lengths, elapsed_time)
        latent = encoder_state[-1][0]
        decoder_input = latent.unsqueeze(1).expand(-1, sequences.size(1), -1)
        decoded, _ = self.decoder(
            decoder_input,
            lengths,
            elapsed_time,
            initial_state=encoder_state,
        )
        return self.output_layer(decoded)


@register_model
class TLSTMAutoencoderPlugin(LSTMAutoencoderPlugin):
    model_type = "t_lstm_autoencoder"
    capabilities = {
        **LSTMAutoencoderPlugin.capabilities,
        "time_aware_recurrence": True,
        "time_context": "elapsed_time",
    }

    def context_feature_names(self):
        return ["elapsed_time"]

    def collate_fn(self):
        return collate_context_trajectories

    def build_model(self):
        architecture = self.config["architecture"]
        return TLSTMAutoencoder(
            input_dim=len(self.config["features"]),
            hidden_dim=architecture["hidden_dim"],
            num_layers=architecture["num_layers"],
            dropout=architecture.get("dropout", 0.0),
            time_scale_seconds=architecture.get("time_scale_seconds", 3600.0),
            decay_function=architecture.get("decay_function", "inverse_log"),
        )
