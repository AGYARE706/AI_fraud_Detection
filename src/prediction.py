"""Load trained model and score transactions."""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import DEFAULT_THRESHOLD, FEATURE_COLUMNS, MODEL_PATH, TARGET_COLUMN
from src.preprocessing import split_features_target, validate_upload_schema
from src.utilities import apply_risk_columns, load_metadata


def model_exists() -> bool:
    return MODEL_PATH.exists()


def load_model():
    """Load persisted sklearn pipeline."""
    if not model_exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run: python train_model.py"
        )
    return joblib.load(MODEL_PATH)


def get_inference_threshold() -> float:
    meta = load_metadata()
    if meta and "threshold" in meta:
        return float(meta["threshold"])
    return DEFAULT_THRESHOLD


def predict_proba(df: pd.DataFrame, model=None) -> pd.Series:
    """Return fraud probability (class 1) for each row."""
    if model is None:
        model = load_model()
    X, _ = split_features_target(df)
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=df.index, name="fraud_probability")


def score_transactions(
    df: pd.DataFrame,
    model=None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Score all rows: adds fraud_probability, risk_level, is_flagged, alert, predicted_fraud.
    """
    ok, msg = validate_upload_schema(df)
    if not ok:
        raise ValueError(msg)

    if model is None:
        model = load_model()
    if threshold is None:
        threshold = get_inference_threshold()

    probabilities = predict_proba(df, model=model)
    return apply_risk_columns(df, probabilities, threshold)


def score_uploaded_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Score upload; returns (scored_df, error_message).
    Strips Class for prediction but keeps it in output if present.
    """
    ok, msg = validate_upload_schema(df)
    if not ok:
        return df, msg
    try:
        scored = score_transactions(df)
        return scored, None
    except FileNotFoundError as e:
        return df, str(e)
    except Exception as e:
        return df, f"Scoring failed: {e}"
