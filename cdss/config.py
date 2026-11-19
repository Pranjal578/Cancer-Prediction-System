"""Central paths and constants for the breast-cancer prediction application."""

from __future__ import annotations

from pathlib import Path

CDSS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CDSS_ROOT.parent
DATA_DIR = CDSS_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = CDSS_ROOT / "artifacts" / "breast"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODEL_PIPELINES_PATH = ARTIFACTS_DIR / "model_pipelines.joblib"

WDBC_PATH = RAW_DATA_DIR / "wdbc.data"

RANDOM_STATE = 42
TEST_SIZE = 0.20
# Validation is a proportion of the 80% development partition. Together with
# TEST_SIZE this gives a stratified 60/20/20 train/validation/test split.
VALIDATION_SIZE = 0.25
SELECTION_METRIC = "recall"
MODEL_NAMES = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine",
    "K-Nearest Neighbors",
)


def ensure_project_directories() -> None:
    """Create output folders used by training without touching raw data."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
