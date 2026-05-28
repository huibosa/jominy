# pyright: basic

import pandas as pd

from modeling.config import METRICS_DIR, OUTPUT_DIR, PREDICTIONS_DIR, REPORTS_DIR
from modeling.reporting import build_summary_report_html, summarize_metric_artifacts


def _load_metrics(metrics_dir):
    baselines_path = metrics_dir / "baselines.csv"
    comparison_path = metrics_dir / "model_comparison.csv"
    baselines = pd.read_csv(baselines_path) if baselines_path.exists() else pd.DataFrame()
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    return baselines, comparison


def _build_comparison_chart_frame(baselines: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if not baselines.empty and {"target", "mae", "model"}.issubset(baselines.columns):
        baseline_summary = (
            baselines.loc[baselines["target"].isin(["J9", "J15", "delta_clipped"]), ["model", "target", "mae"]]
            .groupby(["model", "target"], as_index=False)
            .agg(mae=("mae", "mean"))
        )
        baseline_summary["series"] = "baseline"
        frames.append(baseline_summary.rename(columns={"model": "label"}))

    if not comparison.empty and {"target", "mae"}.issubset(comparison.columns):
        label_col = "label" if "label" in comparison.columns else "model"
        comparison_summary = (
            comparison.loc[:, [label_col, "target", "mae"]]
            .groupby([label_col, "target"], as_index=False)
            .agg(mae=("mae", "mean"))
        )
        comparison_summary["series"] = "comparison"
        comparison_summary = comparison_summary.rename(columns={label_col: "label"})
        frames.append(comparison_summary)

    if not frames:
        return pd.DataFrame(columns=["label", "target", "mae", "series", "chart_label"])

    chart_df = pd.concat(frames, ignore_index=True)
    chart_df["chart_label"] = chart_df["label"] + " (" + chart_df["series"] + ")"
    return chart_df.sort_values(["target", "mae", "chart_label"], ignore_index=True)


def generate_visual_report(metrics_dir, predictions_path, figures_dir, reports_dir):
    modeling_visualize = __import__("modeling.visualize", fromlist=["unused"])
    save_metric_bar_chart = modeling_visualize.save_metric_bar_chart
    save_prediction_error_plots = modeling_visualize.save_prediction_error_plots
    save_qq_plot = modeling_visualize.save_qq_plot

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    baselines, comparison = _load_metrics(metrics_dir)
    predictions = pd.read_parquet(predictions_path)

    figure_paths: list = []
    if {"j9_true", "j9_pred"}.issubset(predictions.columns):
        figure_paths.extend(save_prediction_error_plots(predictions["j9_true"], predictions["j9_pred"], "J9", figures_dir, "j9"))

    if {"j15_true", "j15_pred"}.issubset(predictions.columns):
        figure_paths.extend(save_prediction_error_plots(predictions["j15_true"], predictions["j15_pred"], "J15", figures_dir, "j15"))
        residuals = predictions["j15_true"] - predictions["j15_pred"]
        figure_paths.append(save_qq_plot(residuals, figures_dir / "qq_residuals_j15.png", "J15 Residuals Q-Q Plot"))

    chart_df = _build_comparison_chart_frame(baselines, comparison)
    if not chart_df.empty:
        j9_chart = chart_df.loc[chart_df["target"] == "J9", ["chart_label", "mae"]]
        if not j9_chart.empty:
            figure_paths.append(
                save_metric_bar_chart(
                    j9_chart,
                    figures_dir / "model_comparison_mae.png",
                    "Average J9 MAE by Model Family",
                    x_col="chart_label",
                    y_col="mae",
                )
            )

    metrics = summarize_metric_artifacts(
        baselines=baselines,
        comparison=comparison,
        prediction_rows=len(predictions),
    )
    figures = [{"title": path.stem.replace("_", " ").title(), "filename": path.name} for path in figure_paths]
    html = build_summary_report_html(metrics=metrics, figures=figures, title="Jominy Modeling Summary Report")
    html_path = reports_dir / "summary_report.html"
    html_path.write_text(html, encoding="utf-8")

    return {
        "figures": figure_paths,
        "html_path": html_path,
        "metrics": metrics,
    }


def main() -> None:
    outputs = generate_visual_report(
        metrics_dir=METRICS_DIR,
        predictions_path=PREDICTIONS_DIR / "cv_predictions.parquet",
        figures_dir=OUTPUT_DIR / "figures",
        reports_dir=REPORTS_DIR,
    )
    print(f"Wrote {len(outputs['figures'])} figures to {OUTPUT_DIR / 'figures'}")
    print(f"Wrote HTML summary to {outputs['html_path']}")


if __name__ == "__main__":
    main()
