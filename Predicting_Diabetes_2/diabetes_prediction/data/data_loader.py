import pandas as pd
from pathlib import Path


def load_data(file_path: Path) -> pd.DataFrame:
    """Load dataset from a CSV file path."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)
    return df
