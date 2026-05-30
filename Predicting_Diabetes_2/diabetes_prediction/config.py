from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_FILE = DATA_DIR / "diabetes.csv"
RANDOM_STATE = 42
TARGET_COLUMN = "Diabetes_Status"
AGE_BINS = [0, 25, 60, 100]
AGE_LABELS = ["Young", "Adult", "Elderly"]
DROP_COLUMNS = ["id", "frame", "location", "bp.2s", "bp.2d"]
