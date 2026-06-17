import argparse
from pathlib import Path

import pandas as pd


def compact_predictions(prediction_root="artifacts/predictions"):
    updated = []
    for path in sorted(Path(prediction_root).glob("*/*/*_scores.parquet")):
        frame = pd.read_parquet(path)
        float_columns = frame.select_dtypes(include=["float64"]).columns
        frame[float_columns] = frame[float_columns].astype("float32")
        if "transition_order" in frame:
            frame["transition_order"] = frame["transition_order"].astype("int32")
        frame.to_parquet(path, index=False, compression="zstd")
        updated.append(path)
    return updated


def main():
    parser = argparse.ArgumentParser(description="Compact model prediction Parquet files.")
    parser.add_argument("--prediction-root", default="artifacts/predictions")
    args = parser.parse_args()
    updated = compact_predictions(args.prediction_root)
    print(f"Compacted {len(updated)} prediction files")


if __name__ == "__main__":
    main()
