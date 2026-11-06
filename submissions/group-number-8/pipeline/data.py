"""Loading and the fixed stratified validation split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import SPLIT_SEED, VAL_SIZE


def load_split(path: Path, labeled: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"id", "text"} | ({"label"} if labeled else set())
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "id": df["id"].astype(int),
            "text": df["text"].fillna("").astype(str),
        }
    )
    if labeled:
        out["label"] = df["label"].astype(int)
        if not set(out["label"].unique()).issubset({0, 1}):
            raise ValueError(f"Expected labels 0 and 1 in {path}")
    return out


def validation_split(
    df: pd.DataFrame, seed: int = SPLIT_SEED, val_size: int = VAL_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_partition, val_partition). Validation is fixed by seed."""
    if len(df) <= val_size:
        raise ValueError(
            f"Need more than {val_size} labeled rows to hold out validation; got {len(df)}"
        )
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["label"].to_numpy(),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def exact_text_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[int, int]:
    """Count training rows that copy a validation passage.

    Returns (exact stripped match, case-folded stripped match).
    """
    val_text = set(val_df["text"].str.strip())
    exact = int(train_df["text"].str.strip().isin(val_text).sum())
    val_folded = set(val_df["text"].str.strip().str.casefold())
    folded = int(train_df["text"].str.strip().str.casefold().isin(val_folded).sum())
    return exact, folded
