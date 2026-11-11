"""Car selling price inference module using the saved scikit-learn pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    REFERENCE_YEAR,
    VALID_FUEL_TYPES,
    VALID_SELLER_TYPES,
    VALID_TRANSMISSIONS,
    engineer_features,
)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_car_price_model.joblib"

SAMPLE_CAR_INPUT: dict[str, Any] = {
    "Car_Name": "ciaz",
    "Year": 2017,
    "Present_Price": 9.85,
    "Kms_Driven": 6900,
    "Fuel_Type": "Petrol",
    "Seller_Type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0,
}


def load_model_bundle(
    model_path: str | Path | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Load the saved fitted pipeline and metadata dictionary from disk.

    Raises:
        FileNotFoundError: If the model artifact does not exist yet.
    """
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{path}'.\n"
            "Please train and save the model first by running:\n"
            "  python -m src.train"
        )

    loaded = joblib.load(path)
    if isinstance(loaded, dict) and "pipeline" in loaded:
        return loaded["pipeline"], loaded.get("metadata", {})
    if isinstance(loaded, Pipeline):
        return loaded, {"reference_year": REFERENCE_YEAR, "selected_model": "Saved Pipeline"}
    raise TypeError(f"Unexpected object type loaded from '{path}': {type(loaded)}")


def validate_car_input(car_features: dict[str, Any], reference_year: int) -> pd.DataFrame:
    """Validate user-provided car attributes and convert them to a single-row DataFrame."""
    required_keys = [
        "Year",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner",
    ]
    missing = [k for k in required_keys if k not in car_features]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    year = int(car_features["Year"])
    present_price = float(car_features["Present_Price"])
    kms_driven = int(car_features["Kms_Driven"])
    owner = int(car_features["Owner"])
    fuel_type = str(car_features["Fuel_Type"]).strip()
    seller_type = str(car_features["Seller_Type"]).strip()
    transmission = str(car_features["Transmission"]).strip()

    if year < 1980 or year > reference_year:
        raise ValueError(f"'Year' must be between 1980 and {reference_year}, got {year}.")
    if present_price <= 0:
        raise ValueError(f"'Present_Price' must be positive, got {present_price}.")
    if kms_driven < 0:
        raise ValueError(f"'Kms_Driven' cannot be negative, got {kms_driven}.")
    if owner < 0:
        raise ValueError(f"'Owner' cannot be negative, got {owner}.")
    if fuel_type not in VALID_FUEL_TYPES:
        raise ValueError(f"'Fuel_Type' must be one of {VALID_FUEL_TYPES}, got '{fuel_type}'.")
    if seller_type not in VALID_SELLER_TYPES:
        raise ValueError(f"'Seller_Type' must be one of {VALID_SELLER_TYPES}, got '{seller_type}'.")
    if transmission not in VALID_TRANSMISSIONS:
        raise ValueError(f"'Transmission' must be one of {VALID_TRANSMISSIONS}, got '{transmission}'.")

    return pd.DataFrame(
        [
            {
                "Year": year,
                "Present_Price": present_price,
                "Kms_Driven": kms_driven,
                "Fuel_Type": fuel_type,
                "Seller_Type": seller_type,
                "Transmission": transmission,
                "Owner": owner,
            }
        ]
    )


def predict_car_price(
    car_features: dict[str, Any],
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Predict used-car selling price (in Lakhs INR) from input characteristics."""
    pipeline, metadata = load_model_bundle(model_path)
    reference_year = int(metadata.get("reference_year", REFERENCE_YEAR))
    selected_model = str(metadata.get("selected_model", "Saved Model"))

    raw_df = validate_car_input(car_features, reference_year=reference_year)
    featured_df = engineer_features(raw_df, reference_year=reference_year)

    predicted_price = float(pipeline.predict(featured_df)[0])
    predicted_price = max(0.0, predicted_price)

    return {
        "predicted_selling_price": round(predicted_price, 4),
        "unit": "Lakhs INR",
        "model_name": selected_model,
        "car_age": int(featured_df.loc[0, "Car_Age"]),
        "input_features": car_features,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict used-car selling price.")
    parser.add_argument("--year", type=int, default=SAMPLE_CAR_INPUT["Year"])
    parser.add_argument("--present-price", type=float, default=SAMPLE_CAR_INPUT["Present_Price"])
    parser.add_argument("--kms-driven", type=int, default=SAMPLE_CAR_INPUT["Kms_Driven"])
    parser.add_argument("--fuel-type", type=str, default=SAMPLE_CAR_INPUT["Fuel_Type"])
    parser.add_argument("--seller-type", type=str, default=SAMPLE_CAR_INPUT["Seller_Type"])
    parser.add_argument("--transmission", type=str, default=SAMPLE_CAR_INPUT["Transmission"])
    parser.add_argument("--owner", type=int, default=SAMPLE_CAR_INPUT["Owner"])
    return parser.parse_args()


def main() -> None:
    """CLI entry point for single-car price prediction."""
    args = _parse_args()
    car_input = {
        "Year": args.year,
        "Present_Price": args.present_price,
        "Kms_Driven": args.kms_driven,
        "Fuel_Type": args.fuel_type,
        "Seller_Type": args.seller_type,
        "Transmission": args.transmission,
        "Owner": args.owner,
    }
    result = predict_car_price(car_input)
    print("=== Car Price Prediction ===")
    print(f"Model Used              : {result['model_name']}")
    print(f"Input Characteristics   : {result['input_features']}")
    print(f"Derived Car Age         : {result['car_age']} years")
    print(
        f"Predicted Selling Price : {result['predicted_selling_price']:.2f} {result['unit']} "
        f"(raw: {result['predicted_selling_price']:.4f})"
    )


if __name__ == "__main__":
    main()
