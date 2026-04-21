from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import get_config
from src.utils import clean_text, ensure_directory, save_json

RANDOM_STATE = 42


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    config = get_config()
    dataset_path = Path(path or config.raw_data_path)
    dataframe = pd.read_csv(dataset_path)
    required_columns = {"text", "label"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    dataframe = dataframe.copy()
    dataframe["text"] = dataframe["text"].astype(str).map(clean_text)
    dataframe["label"] = dataframe["label"].astype(str).str.strip()
    dataframe = dataframe.dropna(subset=["text", "label"]).drop_duplicates().reset_index(drop=True)
    return dataframe


def prepare_dataset(
    *,
    data_path: str | Path | None = None,
    train_out: str | Path | None = None,
    test_out: str | Path | None = None,
    summary_out: str | Path | None = None,
    test_size: float = 0.25,
) -> dict:
    config = get_config()
    dataframe = load_dataset(data_path)
    train_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=dataframe["label"],
    )

    train_path = Path(train_out or config.train_split_path)
    test_path = Path(test_out or config.test_split_path)
    summary_path = Path(summary_out or config.prepared_summary_path)

    ensure_directory(train_path.parent)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "raw_rows": int(len(dataframe)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "labels": sorted(dataframe["label"].unique().tolist()),
        "class_distribution": dataframe["label"].value_counts().sort_index().to_dict(),
    }
    save_json(summary, summary_path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MediChat train/test data splits.")
    parser.add_argument("--data-path", default=str(get_config().raw_data_path))
    parser.add_argument("--train-out", default=str(get_config().train_split_path))
    parser.add_argument("--test-out", default=str(get_config().test_split_path))
    parser.add_argument("--summary-out", default=str(get_config().prepared_summary_path))
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    result = prepare_dataset(
        data_path=args.data_path,
        train_out=args.train_out,
        test_out=args.test_out,
        summary_out=args.summary_out,
        test_size=args.test_size,
    )
    print(result)
