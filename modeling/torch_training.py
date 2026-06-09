import random

import numpy as np
import pandas as pd
import torch

from modeling.losses import get_loss


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_reconstruction_model(model, dataloader, config, device):
    loss_function = get_loss(config["training"]["loss"])
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

        for sequences, lengths, _ in dataloader:
            sequences = sequences.to(device)
            optimizer.zero_grad()
            reconstructed = model(sequences, lengths)
            loss, batch_value_count = loss_function(reconstructed, sequences, lengths)
            loss.backward()
            optimizer.step()

            weighted_loss += loss.item() * batch_value_count
            value_count += batch_value_count

        epoch_loss = weighted_loss / value_count if value_count else float("nan")
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"Epoch {epoch}/{epochs}: loss={epoch_loss:.8f}")

    return pd.DataFrame(history)
