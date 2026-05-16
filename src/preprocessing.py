"""
Preprocessing module for the Laptop Price Prediction project.

Handles:
- Cleaning categorical columns
- Feature engineering (PPI from screen resolution)
- Encoding categorical variables
- Preparing data for modeling
"""

import numpy as np
import pandas as pd


# Columns we drop because they have too many unique values
# (would explode the feature count when one-hot encoded)
DROP_COLS = ["Product", "CPU_model", "GPU_model"]

# Categorical columns that need one-hot encoding
CATEGORICAL_COLS = [
    "Company", "TypeName", "OS", "Screen",
    "Touchscreen", "IPSpanel", "RetinaDisplay",
    "CPU_company", "GPU_company",
    "PrimaryStorageType", "SecondaryStorageType",
]


def add_ppi_feature(df):
    """
    Engineer PPI (pixels per inch) from ScreenW, ScreenH, and Inches.
    PPI is a better single feature than the three columns separately.
    """
    df = df.copy()
    df["PPI"] = (np.sqrt(df["ScreenW"] ** 2 + df["ScreenH"] ** 2) / df["Inches"]).round(2)
    # Drop the originals now that we have PPI
    df = df.drop(columns=["ScreenW", "ScreenH"])
    return df


def convert_yes_no(df):
    """Convert Yes/No columns to 1/0 for modeling."""
    df = df.copy()
    for col in ["Touchscreen", "IPSpanel", "RetinaDisplay"]:
        df[col] = (df[col] == "Yes").astype(int)
    return df


def clean_data(df):
    """
    Full cleaning pipeline:
    1. Drop high-cardinality columns
    2. Engineer PPI feature
    3. Convert Yes/No columns to 1/0
    """
    df = df.copy()
    df = df.drop(columns=DROP_COLS)
    df = add_ppi_feature(df)
    df = convert_yes_no(df)
    return df


def encode_features(df, fit_columns=None):
    """
    One-hot encode categorical columns.

    If fit_columns is provided (used during prediction), reindex to match
    those columns exactly. This ensures prediction data has the same
    feature set as training data.
    """
    encoded = pd.get_dummies(
        df,
        columns=[c for c in CATEGORICAL_COLS if c in df.columns],
        drop_first=True
    )
    # Convert bool columns to int (cleaner)
    bool_cols = encoded.select_dtypes(include="bool").columns
    encoded[bool_cols] = encoded[bool_cols].astype(int)

    if fit_columns is not None:
        encoded = encoded.reindex(columns=fit_columns, fill_value=0)

    return encoded, list(encoded.columns)


def prepare_training_data(df, target_col="Price_euros"):
    """
    Full pipeline: clean + encode + split into X and y.
    Returns: X (features), y (target), feature_columns (list)
    """
    df = clean_data(df)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_encoded, feature_columns = encode_features(X)
    return X_encoded, y, feature_columns


def prepare_single_prediction(user_input, feature_columns):
    """
    Convert a single laptop's specs (dict) into model-ready format.
    Used by the Streamlit app for predictions.
    """
    df = pd.DataFrame([user_input])
    df = clean_data(df)
    encoded, _ = encode_features(df, fit_columns=feature_columns)
    return encoded