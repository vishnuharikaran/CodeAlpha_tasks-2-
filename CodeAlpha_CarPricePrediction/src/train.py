"""End-to-end training, cross-validation, evaluation, and model saving script."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import TARGET_COLUMN, clean_car_data, load_car_data
from src.evaluate import (
    calculate_regression_metrics,
    extract_feature_importance,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_model_comparison,
    plot_residual_distribution,
    plot_residuals_vs_predicted,
)
from src.preprocessing import (
    MODEL_INPUT_FEATURES,
    RAW_INPUT_FEATURES,
    REFERENCE_YEAR,
    build_preprocessor,
    engineer_features,
    get_clean_feature_names,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_PLOTS_DIR = OUTPUTS_DIR / "model_evaluation"
VIS_PLOTS_DIR = OUTPUTS_DIR / "visualizations"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_car_price_model.joblib"


def get_candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Return the five candidate regression estimators with fixed random state."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=random_state
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=random_state
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=200, random_state=random_state
        ),
    }


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform an 80:20 train-test split stratified across target price quintiles."""
    price_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=price_bins,
    )


def train_and_evaluate(
    dataset_path: str | Path | None = None,
    save_outputs: bool = True,
    show_plots: bool = False,
) -> dict[str, Any]:
    """Run full data loading, preprocessing, training, CV, evaluation, and saving."""
    raw_df = load_car_data(dataset_path)
    cleaned_df, cleaning_report = clean_car_data(raw_df, max_valid_year=REFERENCE_YEAR)

    X = engineer_features(cleaned_df, reference_year=REFERENCE_YEAR)
    y = cleaned_df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    # Keep raw metadata columns for the test prediction export
    test_raw_df = cleaned_df.loc[X_test.index].copy()

    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    candidate_models = get_candidate_models(random_state=RANDOM_STATE)

    records: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Pipeline] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
            return_train_score=False,
        )

        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_metrics = calculate_regression_metrics(y_train, y_train_pred)
        test_metrics = calculate_regression_metrics(y_test, y_test_pred)

        fitted_pipelines[model_name] = pipeline
        test_predictions[model_name] = y_test_pred

        records.append(
            {
                "Model": model_name,
                "CV_R2_Mean": round(float(np.mean(cv_results["test_r2"])), 4),
                "CV_R2_Std": round(float(np.std(cv_results["test_r2"])), 4),
                "CV_RMSE_Mean": round(
                    float(-np.mean(cv_results["test_neg_root_mean_squared_error"])), 4
                ),
                "Train_R2": round(train_metrics["R2_Score"], 4),
                "Test_MAE": round(test_metrics["MAE"], 4),
                "Test_MSE": round(test_metrics["MSE"], 4),
                "Test_RMSE": round(test_metrics["RMSE"], 4),
                "Test_R2": round(test_metrics["R2_Score"], 4),
            }
        )

    comparison_df = (
        pd.DataFrame(records)
        .sort_values(by=["Test_R2", "CV_R2_Mean"], ascending=[False, False])
        .reset_index(drop=True)
    )

    best_model_name = str(comparison_df.loc[0, "Model"])
    best_pipeline = fitted_pipelines[best_model_name]
    best_test_pred = test_predictions[best_model_name]
    best_test_metrics = calculate_regression_metrics(y_test, best_test_pred)
    best_row = comparison_df.loc[0].to_dict()

    # Build test predictions export table
    prediction_results_df = test_raw_df[
        ["Car_Name", "Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner"]
    ].copy()
    prediction_results_df["Car_Age"] = X_test["Car_Age"].values
    prediction_results_df["Actual_Selling_Price"] = y_test.values
    prediction_results_df["Predicted_Selling_Price"] = np.round(best_test_pred, 4)
    prediction_results_df["Residual"] = np.round(
        y_test.values - best_test_pred, 4
    )
    prediction_results_df["Absolute_Error"] = np.round(
        np.abs(y_test.values - best_test_pred), 4
    )
    prediction_results_df = prediction_results_df.reset_index(drop=True)

    importance_df = extract_feature_importance(best_pipeline)

    # Also compute Extra Trees / Random Forest importance if best is not tree-based (for EDA #10)
    tree_importance_model_name = (
        best_model_name
        if hasattr(best_pipeline.named_steps["model"], "feature_importances_")
        else "Extra Trees Regressor"
    )
    tree_importance_df = extract_feature_importance(
        fitted_pipelines[tree_importance_model_name]
    )

    transformed_feature_names = get_clean_feature_names(
        best_pipeline.named_steps["preprocessor"]
    )

    metadata: dict[str, Any] = {
        "selected_model": best_model_name,
        "evaluation_metrics": {k: round(v, 4) for k, v in best_test_metrics.items()},
        "cross_validation_metrics": {
            "CV_R2_Mean": best_row["CV_R2_Mean"],
            "CV_R2_Std": best_row["CV_R2_Std"],
            "CV_RMSE_Mean": best_row["CV_RMSE_Mean"],
        },
        "raw_input_features": RAW_INPUT_FEATURES,
        "engineered_features": MODEL_INPUT_FEATURES,
        "transformed_feature_names": transformed_feature_names,
        "reference_year": REFERENCE_YEAR,
        "dataset_info": {
            **cleaning_report,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "target_column": TARGET_COLUMN,
            "target_unit": "Lakhs INR",
        },
        "training_config": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_splits": CV_SPLITS,
            "stratified_by_target_quintiles": True,
        },
    }

    if save_outputs:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        EVAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        VIS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        comparison_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
        prediction_results_df.to_csv(OUTPUTS_DIR / "prediction_results.csv", index=False)

        with open(EVAL_PLOTS_DIR / "model_metadata.json", "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

        joblib.dump(
            {"pipeline": best_pipeline, "metadata": metadata},
            DEFAULT_MODEL_PATH,
        )

        plot_model_comparison(
            comparison_df,
            save_path=EVAL_PLOTS_DIR / "model_comparison.png",
            show=show_plots,
        )
        plot_actual_vs_predicted(
            y_test,
            best_test_pred,
            model_name=best_model_name,
            save_path=EVAL_PLOTS_DIR / "actual_vs_predicted.png",
            show=show_plots,
        )
        plot_residual_distribution(
            y_test,
            best_test_pred,
            model_name=best_model_name,
            save_path=EVAL_PLOTS_DIR / "residual_distribution.png",
            show=show_plots,
        )
        plot_residuals_vs_predicted(
            y_test,
            best_test_pred,
            model_name=best_model_name,
            save_path=EVAL_PLOTS_DIR / "residuals_vs_predicted.png",
            show=show_plots,
        )
        if importance_df is not None:
            plot_feature_importance(
                importance_df,
                model_name=best_model_name,
                save_path=EVAL_PLOTS_DIR / "selected_model_feature_importance.png",
                show=show_plots,
            )
        if tree_importance_df is not None:
            plot_feature_importance(
                tree_importance_df,
                model_name=tree_importance_model_name,
                save_path=VIS_PLOTS_DIR / "10_feature_importance.png",
                show=show_plots,
            )

    return {
        "cleaned_df": cleaned_df,
        "cleaning_report": cleaning_report,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "comparison_df": comparison_df,
        "fitted_pipelines": fitted_pipelines,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "best_test_pred": best_test_pred,
        "prediction_results_df": prediction_results_df,
        "importance_df": importance_df,
        "tree_importance_df": tree_importance_df,
        "tree_importance_model_name": tree_importance_model_name,
        "metadata": metadata,
    }


def main() -> None:
    """CLI entry point for training and evaluating car price prediction models."""
    results = train_and_evaluate(save_outputs=True, show_plots=False)
    print("Dataset Cleaning Report:", results["cleaning_report"])
    print("\nModel Comparison (sorted by Test R²):")
    print(results["comparison_df"].to_string(index=False))
    print(f"\nBest Model Selected: {results['best_model_name']}")
    print("Evaluation Metrics:", results["metadata"]["evaluation_metrics"])
    print(f"Saved model bundle to: {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
