"""Train XGBoost fraud classifier and persist model + metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    DATA_PATH,
    MAX_TRAIN_ROWS,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.preprocessing import (
    build_preprocessor,
    get_feature_names_after_preprocess,
    load_data,
    split_features_target,
    stratified_train_test_split,
    subsample_for_training,
)


def build_classifier(scale_pos_weight: float) -> XGBClassifier:
    """XGBoost with class imbalance weighting."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Pick threshold that maximizes F1 on validation probabilities."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def train_pipeline(data_path=None) -> dict:
    """
    End-to-end training: load data, fit pipeline, evaluate, save artifacts.
    Returns metadata dict.
    """
    path = data_path or DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download from Kaggle and place as data/creditcard.csv"
        )

    df = load_data(path)
    df = subsample_for_training(df, MAX_TRAIN_ROWS)
    X, y = split_features_target(df)
    if y is None:
        raise ValueError("Training data must include 'Class' column.")

    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)
    # Validation slice from train for threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    preprocessor = build_preprocessor()
    classifier = build_classifier(scale_pos_weight)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val.values, val_proba)

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "pr_auc": float(average_precision_score(y_test, test_proba)),
    }

    cm = confusion_matrix(y_test, test_pred)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "fraud_rate_train": float(y_train.mean()),
        "scale_pos_weight": scale_pos_weight,
        "feature_names": get_feature_names_after_preprocess(),
        "model_type": "XGBClassifier",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def print_training_report(metadata: dict) -> None:
    """Console summary after training."""
    m = metadata["metrics"]
    print("\n=== Fraud Model Training Complete ===")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Threshold:   {metadata['threshold']:.2f}")
    print(f"PR-AUC:      {m['pr_auc']:.4f}")
    print(f"ROC-AUC:     {m['roc_auc']:.4f}")
    print(f"Precision:   {m['precision']:.4f}")
    print(f"Recall:      {m['recall']:.4f}")
    print(f"F1:          {m['f1']:.4f}")
    print("Confusion matrix [TN, FP], [FN, TP]:")
    for row in metadata["confusion_matrix"]:
        print(f"  {row}")
