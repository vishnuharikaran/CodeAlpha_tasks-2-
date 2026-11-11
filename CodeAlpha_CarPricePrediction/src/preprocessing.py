"""Feature engineering and scikit-learn preprocessing pipelines."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REFERENCE_YEAR = 2026

NUMERIC_FEATURES = ["Present_Price", "Kms_Driven", "Owner", "Car_Age"]
CATEGORICAL_FEATURES = ["Fuel_Type", "Seller_Type", "Transmission"]
MODEL_INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RAW_INPUT_FEATURES = [
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
]

VALID_FUEL_TYPES = ("Petrol", "Diesel", "CNG")
VALID_SELLER_TYPES = ("Dealer", "Individual")
VALID_TRANSMISSIONS = ("Manual", "Automatic")


def engineer_features(
    df: pd.DataFrame, reference_year: int = REFERENCE_YEAR
) -> pd.DataFrame:
    """Create model-ready features from raw/cleaned vehicle records.

    Derives ``Car_Age = reference_year - Year`` and drops redundant or
    high-cardinality columns (``Year``, ``Car_Name``). Never includes
    ``Selling_Price`` in the returned feature matrix.

    Args:
        df: DataFrame containing raw vehicle columns.
        reference_year: Reference year used to calculate ``Car_Age``.

    Returns:
        DataFrame with engineered input features in canonical column order.
    """
    if "Year" not in df.columns:
        raise ValueError("Input DataFrame must contain the 'Year' column to derive 'Car_Age'.")

    featured = df.copy()
    featured["Car_Age"] = reference_year - featured["Year"]

    missing = [col for col in MODEL_INPUT_FEATURES if col not in featured.columns]
    if missing:
        raise ValueError(f"Missing required feature columns after engineering: {missing}")

    return featured[MODEL_INPUT_FEATURES].copy()


def build_preprocessor() -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for numeric and categorical features.

    Numeric features are standardized with ``StandardScaler`` and categorical
    features are one-hot encoded with ``handle_unknown='ignore'`` so unseen
    categories during evaluation or inference never fail.
    """
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_clean_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return human-readable output feature names from a fitted preprocessor."""
    raw_names = preprocessor.get_feature_names_out()
    return [name.replace("num__", "").replace("cat__", "") for name in raw_names]
