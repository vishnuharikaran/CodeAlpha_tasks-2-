"""Dataset loading and data-cleaning utilities for Car Price Prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "car data.csv"

TARGET_COLUMN = "Selling_Price"
EXPECTED_COLUMNS = [
    "Car_Name",
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
]

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho"
)


def load_car_data(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load the CarDekho used-car dataset and verify expected columns.

    Args:
        dataset_path: Optional path to ``car data.csv``. Defaults to
            ``dataset/car data.csv`` inside the project root.

    Returns:
        Loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the target path.
        ValueError: If expected columns are missing from the loaded CSV.
    """
    path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            f"Please download 'car data.csv' from Kaggle ({KAGGLE_DATASET_URL}) "
            f"and place it at '{DEFAULT_DATASET_PATH}'."
        )

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset at '{path}' is missing expected columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def clean_car_data(
    df: pd.DataFrame, max_valid_year: int = 2026
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the loaded dataset without discarding legitimate outlier observations.

    Steps performed:
    1. Strip leading/trailing whitespace on string columns.
    2. Ensure appropriate numeric data types.
    3. Remove exact duplicate rows.
    4. Drop rows with missing values (if any exist).
    5. Filter physically impossible values (e.g., negative prices or mileage).

    Args:
        df: Raw CarDekho DataFrame.
        max_valid_year: Upper bound for valid manufacturing year.

    Returns:
        Tuple of ``(cleaned_df, cleaning_report)``.
    """
    cleaned = df.copy()
    initial_rows = len(cleaned)

    # Normalize string columns
    for col in ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # Enforce numeric types
    cleaned["Year"] = pd.to_numeric(cleaned["Year"], errors="coerce").astype("Int64")
    cleaned["Selling_Price"] = pd.to_numeric(cleaned["Selling_Price"], errors="coerce")
    cleaned["Present_Price"] = pd.to_numeric(cleaned["Present_Price"], errors="coerce")
    cleaned["Kms_Driven"] = pd.to_numeric(cleaned["Kms_Driven"], errors="coerce").astype("Int64")
    cleaned["Owner"] = pd.to_numeric(cleaned["Owner"], errors="coerce").astype("Int64")

    missing_before_drop = int(cleaned.isnull().sum().sum())
    if missing_before_drop > 0:
        cleaned = cleaned.dropna().copy()
    rows_after_missing = len(cleaned)

    duplicate_count = int(cleaned.duplicated().sum())
    if duplicate_count > 0:
        cleaned = cleaned.drop_duplicates().copy()
    rows_after_duplicates = len(cleaned)

    # Validate realistic domain bounds without removing valid luxury/high-mileage cars
    valid_mask = (
        (cleaned["Year"] >= 1980)
        & (cleaned["Year"] <= max_valid_year)
        & (cleaned["Present_Price"] > 0)
        & (cleaned["Selling_Price"] >= 0)
        & (cleaned["Kms_Driven"] >= 0)
        & (cleaned["Owner"] >= 0)
    )
    invalid_rows = int((~valid_mask).sum())
    cleaned = cleaned.loc[valid_mask].reset_index(drop=True)

    # Convert nullable Int64 back to standard int64 after validation
    for int_col in ["Year", "Kms_Driven", "Owner"]:
        cleaned[int_col] = cleaned[int_col].astype("int64")

    report: dict[str, Any] = {
        "initial_rows": initial_rows,
        "missing_values_found": missing_before_drop,
        "rows_dropped_for_missing": initial_rows - rows_after_missing,
        "duplicate_rows_removed": duplicate_count,
        "invalid_domain_rows_removed": invalid_rows,
        "final_rows": len(cleaned),
    }
    return cleaned, report
