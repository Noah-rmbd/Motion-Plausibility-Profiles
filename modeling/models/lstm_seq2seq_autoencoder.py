import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modeling.models.lstm_autoencoder import LSTMAutoencoderPlugin
from modeling.registry import register_model


class LSTMSeq2SeqAutoencoder(nn.Module):
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
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.start_token = nn.Parameter(torch.zeros(input_dim))

    def encode(self, sequences, lengths):
        packed = pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, state = self.encoder(packed)
        return state

    def decode_with_teacher_forcing(self, sequences, lengths, state):
        start = self.start_token.view(1, 1, -1).expand(sequences.size(0), 1, -1)
        decoder_input = torch.cat([start, sequences[:, :-1]], dim=1)
        packed = pack_padded_sequence(
            decoder_input,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        decoded, _ = self.decoder(packed, state)
        decoded, _ = pad_packed_sequence(
            decoded,
            batch_first=True,
            total_length=sequences.size(1),
        )
        return self.output_layer(decoded)

    def decode_autoregressively(self, sequences, state):
        decoder_input = self.start_token.view(1, 1, -1).expand(
            sequences.size(0),
            1,
            -1,
        )
        outputs = []
        for _ in range(sequences.size(1)):
            decoded, state = self.decoder(decoder_input, state)
            prediction = self.output_layer(decoded)
            outputs.append(prediction)
            decoder_input = prediction
        return torch.cat(outputs, dim=1)

    def forward(self, sequences, lengths):
        state = self.encode(sequences, lengths)
        if self.training:
            return self.decode_with_teacher_forcing(sequences, lengths, state)
        return self.decode_autoregressively(sequences, state)


@register_model
class LSTMSeq2SeqAutoencoderPlugin(LSTMAutoencoderPlugin):
    model_type = "lstm_seq2seq_autoencoder"

    def build_model(self):
        architecture = self.config["architecture"]
        return LSTMSeq2SeqAutoencoder(
            input_dim=len(self.config["features"]),
            hidden_dim=architecture["hidden_dim"],
            num_layers=architecture["num_layers"],
            dropout=architecture.get("dropout", 0.0),
        )
