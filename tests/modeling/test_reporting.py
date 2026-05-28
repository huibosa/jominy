# pyright: basic

from importlib import import_module


def test_add_candidate_notes_annotates_comparison_labels() -> None:
    pd = import_module("pandas")
    modeling_reporting = import_module("modeling.reporting")

    df = pd.DataFrame(
        {
            "label": ["ridge_core", "ridge_full", "pls_full", "hgbr_full"],
        }
    )

    annotated = modeling_reporting.add_candidate_notes(df)

    assert annotated["selection_policy"].tolist() == ["benchmark_only"] * 4
    assert annotated["candidate_note"].tolist() == [
        "Core-feature ridge benchmark; not the fixed full-feature export reference.",
        "Full-feature ridge reference for Task 5 benchmarking; final production choice remains deferred.",
        "Benchmark-only PLS challenger with external standardization and internal scaling disabled.",
        "Benchmark-only shallow HGBR challenger constrained for the small grouped dataset.",
    ]


def test_summarize_metric_artifacts_handles_baseline_and_comparison_outputs() -> None:
    pd = import_module("pandas")
    modeling_reporting = import_module("modeling.reporting")

    baselines = pd.DataFrame(
        {
            "fold": [0, 1, 0, 1, 0],
            "model": ["ridge", "ridge", "ridge", "ridge", "ridge"],
            "target": ["J9", "J9", "J15", "J15", "delta_clipped"],
            "mae": [1.1, 1.3, 1.4, 1.6, 0.8],
            "rmse": [1.3, 1.5, 1.6, 1.8, 1.0],
            "r2": [0.4, 0.5, 0.3, 0.2, 0.6],
        }
    )
    comparison = pd.DataFrame(
        {
            "label": ["ridge_full", "pls_full", "ridge_full", "pls_full"],
            "target": ["J9", "J9", "delta", "delta"],
            "mae": [1.2, 1.0, 0.9, 0.7],
        }
    )

    metrics = modeling_reporting.summarize_metric_artifacts(baselines=baselines, comparison=comparison, prediction_rows=491)

    metric_map = {row["metric"]: row["value"] for row in metrics}

    assert metric_map["rows_in_prediction_artifact"] == 491
    assert metric_map["baseline_best_j15_mae"] == 1.5
    assert metric_map["baseline_best_j9_mae"] == 1.2
    assert metric_map["baseline_best_delta_clipped_mae"] == 0.8
    assert metric_map["comparison_best_j9_candidate"] == "pls_full"
    assert metric_map["comparison_best_delta_candidate"] == "pls_full"


def test_build_summary_report_html_embeds_metrics_images_and_warning() -> None:
    modeling_reporting = import_module("modeling.reporting")

    html = modeling_reporting.build_summary_report_html(
        metrics=[{"metric": "baseline_best_j15_mae", "value": 1.23}],
        figures=[{"title": "Residual Plot", "filename": "residuals_vs_predicted_j15.png"}],
        title="Jominy Summary",
    )

    assert "Jominy Summary" in html
    assert "baseline_best_j15_mae" in html
    assert "residuals_vs_predicted_j15.png" in html
    assert "deployment warning" in html.lower()
    assert "chemistry-only pool" in html
    assert "<html" in html.lower()
