"""Central configuration for paths, thresholds, and feature columns."""

from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"

TARGET_COLUMN = "Class"
FEATURE_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Decision threshold (tuned during training; fallback for inference)
DEFAULT_THRESHOLD = 0.5

# Alert when fraud probability exceeds this (Critical tier + toast)
ALERT_THRESHOLD = 0.85

# Risk bands: (upper_bound_exclusive, label) — checked in ascending order
RISK_BANDS = [
    (0.30, "Low"),
    (0.60, "Medium"),
    (0.85, "High"),
]

# Training
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_TRAIN_ROWS = None  # Use full dataset; set int to subsample for quick dev

# Live simulation defaults
SIM_BATCH_SIZE = 1
SIM_MAX_QUEUE = 500
