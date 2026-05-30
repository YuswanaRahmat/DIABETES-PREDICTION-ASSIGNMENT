import pandas as pd

from diabetes_prediction.data.data_loader import load_data
from diabetes_prediction.preprocessing.preprocessing import clean_data, create_target, encode_categories
from diabetes_prediction.config import DATA_FILE


def test_clean_data_fills_missing_values():
    df = load_data(DATA_FILE)
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0
    assert "BMI" in cleaned.columns


def test_create_target_adds_diabetes_status():
    df = load_data(DATA_FILE)
    cleaned = clean_data(df)
    target_df = create_target(cleaned)
    assert "Diabetes_Status" in target_df.columns
    assert target_df["Diabetes_Status"].nunique() > 1


def test_encode_categories_maps_strings():
    df = load_data(DATA_FILE)
    cleaned = clean_data(df)
    encoded = encode_categories(cleaned)
    assert encoded["gender"].dtype.kind in "iu"
    assert encoded["age_group"].dtype.kind in "iu"
