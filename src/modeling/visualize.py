# pyright: basic

from importlib import import_module
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.metrics import PredictionErrorDisplay


plt = import_module("matplotlib.pyplot")


def save_prediction_error_plots(y_true, y_pred, title_prefix: str, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 5))
    PredictionErrorDisplay.from_predictions(y_true, y_pred=y_pred, kind="actual_vs_predicted", ax=ax)
    ax.set_title(f"{title_prefix}: Actual vs Predicted")
    actual_path = output_dir / f"actual_vs_predicted_{stem}.png"
    fig.savefig(actual_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(actual_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    PredictionErrorDisplay.from_predictions(y_true, y_pred=y_pred, kind="residual_vs_predicted", ax=ax)
    ax.set_title(f"{title_prefix}: Residuals vs Predicted")
    residual_path = output_dir / f"residuals_vs_predicted_{stem}.png"
    fig.savefig(residual_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(residual_path)

    return paths


def save_qq_plot(residuals, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_metric_bar_chart(df: pd.DataFrame, output_path: Path, title: str, x_col: str, y_col: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = df.copy()
    plot_df[x_col] = plot_df[x_col].astype(str)
    plot_df.plot.bar(x=x_col, y=y_col, ax=ax, legend=False, color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel(y_col)
    ax.set_xlabel("")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
