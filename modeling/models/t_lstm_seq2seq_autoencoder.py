import torch
from torch import nn

from modeling.models.t_lstm_autoencoder import (
    TLSTMAutoencoder,
    TLSTMAutoencoderPlugin,
    TimeAwareLSTM,
)
from modeling.framework.registry import register_model


class TLSTMSeq2SeqAutoencoder(TLSTMAutoencoder):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout=0.0,
        time_scale_seconds=3600.0,
        decay_function="inverse_log",
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            time_scale_seconds=time_scale_seconds,
            decay_function=decay_function,
        )
        self.decoder = TimeAwareLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.start_token = nn.Parameter(torch.zeros(input_dim))

    def encode(self, sequences, lengths, elapsed_time):
        _, state = self.encoder(sequences, lengths, elapsed_time)
        return state

    def decode_with_teacher_forcing(
        self,
        sequences,
        lengths,
        elapsed_time,
        state,
    ):
        start = self.start_token.view(1, 1, -1).expand(
            sequences.size(0),
            1,
            -1,
        )
        decoder_input = torch.cat([start, sequences[:, :-1]], dim=1)
        decoded, _ = self.decoder(
            decoder_input,
            lengths,
            elapsed_time,
            initial_state=state,
        )
        return self.output_layer(decoded)

    def decode_autoregressively(self, lengths, elapsed_time, state):
        batch_size, max_length, _ = elapsed_time.shape
        decoder_input = self.start_token.view(1, 1, -1).expand(
            batch_size,
            1,
            -1,
        )
        outputs = []
        device_lengths = lengths.to(elapsed_time.device)
        for timestep in range(max_length):
            step_lengths = (device_lengths > timestep).to(dtype=torch.long)
            decoded, state = self.decoder(
                decoder_input,
                step_lengths,
                elapsed_time[:, timestep : timestep + 1],
                initial_state=state,
            )
            prediction = self.output_layer(decoded)
            outputs.append(prediction)
            decoder_input = prediction
        return torch.cat(outputs, dim=1)

    def forward(self, sequences, lengths, context):
        elapsed_time = self.elapsed_hours(context)
        state = self.encode(sequences, lengths, elapsed_time)
        if self.training:
            return self.decode_with_teacher_forcing(
                sequences,
                lengths,
                elapsed_time,
                state,
            )
        return self.decode_autoregressively(lengths, elapsed_time, state)


@register_model
class TLSTMSeq2SeqAutoencoderPlugin(TLSTMAutoencoderPlugin):
    model_type = "t_lstm_seq2seq_autoencoder"
    capabilities = {
        **TLSTMAutoencoderPlugin.capabilities,
        "seq2seq_decoder": True,
        "teacher_forcing": True,
        "autoregressive_evaluation": True,
    }

    def build_model(self):
        architecture = self.config["architecture"]
        return TLSTMSeq2SeqAutoencoder(
            input_dim=len(self.config["features"]),
            hidden_dim=architecture["hidden_dim"],
            num_layers=architecture["num_layers"],
            dropout=architecture.get("dropout", 0.0),
            time_scale_seconds=architecture.get("time_scale_seconds", 3600.0),
            decay_function=architecture.get("decay_function", "inverse_log"),
        )
