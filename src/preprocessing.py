"""Data loading, validation, and sklearn preprocessing helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_data(path) -> pd.DataFrame:
    """Load the credit card fraud CSV."""
    df = pd.read_csv(path)
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series | None]:
    """
    Split features and optional target.
    Returns (X, y) where y is None if Class column is absent (inference).
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].astype(int)
        return X, y
    return X, None


def stratified_train_test_split(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 stratified split on fraud class."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Scale Amount and Time; pass V1–V28 through unchanged (already transformed).
    """
    scale_cols = ["Time", "Amount"]
    pass_cols = [c for c in FEATURE_COLUMNS if c not in scale_cols]

    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scale_cols),
            ("pass", "passthrough", pass_cols),
        ],
        remainder="drop",
    )


def get_feature_names_after_preprocess() -> list[str]:
    """Feature names in order after ColumnTransformer (for importance plots)."""
    scale_cols = ["Time", "Amount"]
    pass_cols = [c for c in FEATURE_COLUMNS if c not in scale_cols]
    return scale_cols + pass_cols


def validate_upload_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate uploaded CSV has required feature columns.
    Class is optional (for scoring-only uploads).
    """
    if df.empty:
        return False, "Uploaded file is empty."

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"

    for col in FEATURE_COLUMNS:
        if not np.issubdtype(df[col].dtype, np.number):
            return False, f"Column '{col}' must be numeric."

    if TARGET_COLUMN in df.columns:
        if not set(df[TARGET_COLUMN].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            return False, f"'{TARGET_COLUMN}' must contain only 0 and 1."

    return True, "OK"


def subsample_for_training(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    """Optional subsample while preserving class ratio."""
    if max_rows is None or len(df) <= max_rows:
        return df
    _, sample = train_test_split(
        df,
        train_size=max_rows,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    return sample.reset_index(drop=True)
