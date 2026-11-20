"""Central configuration for the research ML framework."""
import os
from pathlib import Path

# Safe temporary directories for sandboxed file writes
os.environ.setdefault("SKB_DATA_DIRECTORY", "/tmp/skrub_data")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("TABPFN_CACHE_DIR", "/tmp/tabpfn_cache")

RESEARCH_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_ROOT.parent
WDBC_PATH = PROJECT_ROOT / "cdss" / "data" / "raw" / "wdbc.data"
ARTIFACTS_DIR = RESEARCH_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
RESULTS_DIR = ARTIFACTS_DIR / "results"

RANDOM_STATE = 42
SEEDS = [42, 123, 7]
TEST_SIZE = 0.20
N_FOLDS = 5
N_OPTUNA_TRIALS = 30

MODEL_NAMES = ["TabPFN", "CatBoost", "FT-Transformer", "XGBoost"]
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]
CLASS_NAMES = ["Benign", "Malignant"]
POSITIVE_CLASS = 1

def ensure_directories():
    for d in [MODELS_DIR, REPORTS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
