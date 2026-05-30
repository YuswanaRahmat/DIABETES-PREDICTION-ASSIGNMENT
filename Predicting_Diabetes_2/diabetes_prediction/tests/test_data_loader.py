import pandas as pd

from diabetes_prediction.data.data_loader import load_data
from diabetes_prediction.config import DATA_FILE


def test_load_data_returns_dataframe():
    df = load_data(DATA_FILE)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "id" in df.columns
