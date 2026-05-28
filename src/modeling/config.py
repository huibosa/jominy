from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELING_DATA_DIR = DATA_DIR / "modeling"
OUTPUT_DIR = PROJECT_ROOT / "output" / "modeling"
FOLDS_DIR = OUTPUT_DIR / "folds"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

RANDOM_SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3

TARGET_J9 = "J9"
TARGET_J15 = "J15"
TARGET_DELTA = "delta"
GROUP_COL = "base_heat_id"

LONG_SOURCE_PATH = DATA_DIR / "jominy_cleaned.csv"
CHEMISTRY_ONLY_PATH = DATA_DIR / "jominy_chemistry_only.csv"
SPECIMEN_TABLE_PATH = MODELING_DATA_DIR / "specimen_table.parquet"
J9_DATASET_PATH = MODELING_DATA_DIR / "j9_dataset.parquet"
DELTA_DATASET_PATH = MODELING_DATA_DIR / "delta_dataset.parquet"
