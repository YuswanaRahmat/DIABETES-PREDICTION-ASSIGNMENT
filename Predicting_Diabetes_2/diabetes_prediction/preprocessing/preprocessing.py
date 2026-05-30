from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from diabetes_prediction.config import AGE_BINS, AGE_LABELS, DROP_COLUMNS, RANDOM_STATE, TARGET_COLUMN


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_columns = [column for column in DROP_COLUMNS if column in df.columns]
    df = df.drop(columns=drop_columns, errors="ignore")

    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna("UNKNOWN")

    df["BMI"] = (df["weight"] / (df["height"] ** 2)) * 703
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


def categorize_patient(row: pd.Series) -> str:
    glyhb = row["glyhb"]
    stab_glu = row["stab.glu"]

    if glyhb <= 5.7 or (glyhb <= 5.7 and stab_glu <= 99):
        return "Normal"
    if 5.7 < glyhb < 6.5 or (5.7 < glyhb < 6.5 and 99 < stab_glu < 125):
        return "Prediabetes"
    if glyhb >= 6.5 or (glyhb >= 6.5 and stab_glu >= 125):
        return "Diabetes"
    return "Unknown"


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = df.apply(categorize_patient, axis=1)
    return df


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.lower().map({"male": 1, "female": 2}).fillna(0)
    if "age_group" in df.columns:
        df["age_group"] = df["age_group"].astype(str).map({"Young": 1, "Adult": 2, "Elderly": 3}).fillna(0)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    return df[feature_columns]


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = [column for column in df.select_dtypes(include=[np.number]).columns if column != TARGET_COLUMN]
    if numeric_columns:
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


def build_target_vector(df: pd.DataFrame) -> np.ndarray:
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(df[TARGET_COLUMN].astype(str))


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = clean_data(df)
    df = create_target(df)
    df = encode_categories(df)
    df = scale_features(df)

    X = build_feature_matrix(df)
    y = build_target_vector(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def split_train_test(df: pd.DataFrame, test_size: float = 0.2):
    df = clean_data(df)
    df = create_target(df)
    df = encode_categories(df)
    df = scale_features(df)
    X = build_feature_matrix(df)
    y = build_target_vector(df)
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
