"""Model evaluation metrics, diagnostics, and visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocessing import get_clean_feature_names

sns.set_theme(style="whitegrid", context="notebook", palette="deep")


def calculate_regression_metrics(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> dict[str, float]:
    """Compute MAE, MSE, RMSE, and R2 Score for regression predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2_Score": r2}


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    """Extract sorted feature importances from a fitted pipeline if supported."""
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")
    if model is None or preprocessor is None:
        return None

    feature_names = get_clean_feature_names(preprocessor)
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None

    df_imp = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)
    return df_imp.reset_index(drop=True)


def _finalize_figure(fig: plt.Figure, save_path: Path | None, show: bool) -> None:
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot side-by-side comparison of Test R2 Score and Test RMSE across models."""
    ordered = comparison_df.sort_values(by="Test_R2", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    bars_r2 = axes[0].barh(ordered["Model"], ordered["Test_R2"], color="#2b6cb0")
    axes[0].set_title("Model Comparison: Test R² Score (Higher is Better)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("R² Score")
    axes[0].set_ylabel("Model")
    axes[0].set_xlim(0, min(1.05, max(1.0, ordered["Test_R2"].max() + 0.1)))
    for bar in bars_r2:
        w = bar.get_width()
        axes[0].text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9)

    ordered_rmse = comparison_df.sort_values(by="Test_RMSE", ascending=False)
    bars_rmse = axes[1].barh(ordered_rmse["Model"], ordered_rmse["Test_RMSE"], color="#dd6b20")
    axes[1].set_title("Model Comparison: Test RMSE (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("RMSE (Lakhs INR)")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, ordered_rmse["Test_RMSE"].max() * 1.18)
    for bar in bars_rmse:
        w = bar.get_width()
        axes[1].text(w + 0.04, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9)

    _finalize_figure(fig, save_path, show)
    return fig


def plot_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot actual vs predicted selling prices with the ideal 45-degree reference line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.8, edgecolors="white", linewidth=0.6, s=65, color="#2b6cb0")
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Prediction (y = x)")

    r2 = r2_score(y_true, y_pred)
    ax.set_title(f"Actual vs Predicted Selling Price ({model_name} | R² = {r2:.4f})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Selling Price (Lakhs INR)")
    ax.set_ylabel("Predicted Selling Price (Lakhs INR)")
    ax.legend()

    _finalize_figure(fig, save_path, show)
    return fig


def plot_residual_distribution(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot histogram and KDE of prediction residuals (Actual - Predicted)."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.histplot(residuals, kde=True, bins=18, color="#319795", ax=ax)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.8, label="Zero Residual")
    ax.set_title(f"Residual Distribution ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Residual (Actual - Predicted, Lakhs INR)")
    ax.set_ylabel("Frequency")
    ax.legend()

    _finalize_figure(fig, save_path, show)
    return fig


def plot_residuals_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot residuals against predicted values to check homoscedasticity."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(y_pred, residuals, alpha=0.8, edgecolors="white", linewidth=0.6, s=60, color="#805ad5")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.8)
    ax.set_title(f"Residuals vs Predicted Values ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Selling Price (Lakhs INR)")
    ax.set_ylabel("Residual (Actual - Predicted, Lakhs INR)")

    _finalize_figure(fig, save_path, show)
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot horizontal bar chart of feature importances."""
    ordered = importance_df.sort_values(by="Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(ordered["Feature"], ordered["Importance"], color="#2f855a")
    ax.set_title(f"Feature Importance ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, ordered["Importance"].max() * 1.18)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9)

    _finalize_figure(fig, save_path, show)
    return fig
