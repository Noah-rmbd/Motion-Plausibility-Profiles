from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def _score_batch(
    sequences,
    lengths,
    metadata_batch,
    reconstructed,
    feature_names,
    ignore_first_transition,
    model_id,
):
    squared_errors = (reconstructed - sequences) ** 2
    transition_mse = squared_errors.mean(dim=2)
    trajectory_rows = []
    transition_rows = []

    for batch_index, metadata in enumerate(metadata_batch):
        length = int(lengths[batch_index].item())
        feature_errors = squared_errors[batch_index, :length].cpu().numpy()
        step_scores = transition_mse[batch_index, :length].cpu().numpy()
        trajectory_slice = slice(1, length) if ignore_first_transition and length > 1 else slice(0, length)
        trajectory_feature_errors = feature_errors[trajectory_slice].mean(axis=0)

        trajectory_row = {
            "model_id": model_id,
            "dataset": metadata.dataset,
            "trajectory_id": metadata.trajectory_id,
            "score": float(step_scores[trajectory_slice].mean()),
        }
        for feature_index, feature_name in enumerate(feature_names):
            trajectory_row[f"error_{feature_name}"] = float(trajectory_feature_errors[feature_index])
        trajectory_rows.append(trajectory_row)

        for index, transition_id in enumerate(metadata.transition_ids):
            transition_row = {
                "model_id": model_id,
                "dataset": metadata.dataset,
                "trajectory_id": metadata.trajectory_id,
                "transition_order": int(metadata.transition_orders[index]),
                "transition_id": int(transition_id),
                "score": float(step_scores[index]),
            }
            for feature_index, feature_name in enumerate(feature_names):
                transition_row[f"error_{feature_name}"] = float(feature_errors[index, feature_index])
            transition_rows.append(transition_row)

    return pd.DataFrame(trajectory_rows), pd.DataFrame(transition_rows)


def _write_batch(writer, dataframe, path):
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def write_reconstruction_predictions(
    model,
    dataloader,
    feature_names,
    device,
    ignore_first_transition,
    prediction_root,
    model_id,
    dataset,
):
    output_dir = Path(prediction_root) / model_id / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory_scores.parquet"
    transition_path = output_dir / "transition_scores.parquet"
    trajectory_temp = output_dir / "trajectory_scores.parquet.tmp"
    transition_temp = output_dir / "transition_scores.parquet.tmp"
    trajectory_writer = None
    transition_writer = None
    trajectory_count = 0
    transition_count = 0

    model.eval()
    model.to(device)
    try:
        with torch.no_grad():
            for sequences, lengths, metadata_batch in dataloader:
                sequences = sequences.to(device)
                reconstructed = model(sequences, lengths)
                trajectory_scores, transition_scores = _score_batch(
                    sequences=sequences,
                    lengths=lengths,
                    metadata_batch=metadata_batch,
                    reconstructed=reconstructed,
                    feature_names=feature_names,
                    ignore_first_transition=ignore_first_transition,
                    model_id=model_id,
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
    }
