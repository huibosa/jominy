# pyright: basic

from importlib import import_module, util
from pathlib import Path


def _load_script_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "generate_visual_report.py"
    spec = util.spec_from_file_location("generate_visual_report", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load generate_visual_report.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_save_prediction_error_plots_and_qq_plot_write_pngs(tmp_path: Path) -> None:
    pd = import_module("pandas")
    modeling_visualize = import_module("modeling.visualize")

    preds = pd.DataFrame(
        {
            "y_true": [40.0, 38.5, 36.0, 34.5],
            "y_pred": [39.4, 38.8, 35.2, 35.1],
        }
    )

    paths = modeling_visualize.save_prediction_error_plots(
        preds["y_true"],
        preds["y_pred"],
        title_prefix="J15",
        output_dir=tmp_path,
        stem="j15",
    )
    qq_path = modeling_visualize.save_qq_plot(
        preds["y_true"] - preds["y_pred"],
        tmp_path / "qq_residuals_j15.png",
        "J15 Residuals Q-Q Plot",
    )

    assert [path.name for path in paths] == [
        "actual_vs_predicted_j15.png",
        "residuals_vs_predicted_j15.png",
    ]
    assert all(path.exists() for path in [*paths, qq_path])


def test_generate_visual_report_writes_html_and_expected_figures(tmp_path: Path) -> None:
    pd = import_module("pandas")
    script_module = _load_script_module()

    metrics_dir = tmp_path / "metrics"
    predictions_dir = tmp_path / "predictions"
    figures_dir = tmp_path / "figures"
    reports_dir = tmp_path / "reports"
    metrics_dir.mkdir()
    predictions_dir.mkdir()

    pd.DataFrame(
        {
            "fold": [0, 1, 0, 1, 0],
            "model": ["ridge", "ridge", "ridge", "ridge", "ridge"],
            "target": ["J9", "J9", "J15", "J15", "delta_clipped"],
            "mae": [1.0, 1.2, 1.4, 1.6, 0.9],
            "rmse": [1.2, 1.4, 1.6, 1.8, 1.1],
            "r2": [0.4, 0.5, 0.2, 0.3, 0.6],
        }
    ).to_csv(metrics_dir / "baselines.csv", index=False)
    pd.DataFrame(
        {
            "fold": [0, 1, 0, 1],
            "model": ["ridge", "pls", "ridge", "pls"],
            "target": ["J9", "J9", "delta", "delta"],
            "mae": [1.1, 0.95, 0.8, 0.7],
            "label": ["ridge_full", "pls_full", "ridge_full", "pls_full"],
        }
    ).to_csv(metrics_dir / "model_comparison.csv", index=False)
    pd.DataFrame(
        {
            "炉号": ["H1", "H2", "H3"],
            "base_heat_id": ["H1", "H2", "H3"],
            "outer_fold": [0, 1, 2],
            "j9_true": [40.0, 38.5, 37.0],
            "j9_pred": [39.8, 38.1, 36.8],
            "delta_true": [6.0, 5.5, 5.0],
            "delta_pred": [5.8, 5.4, 4.9],
            "delta_pred_clipped": [5.8, 5.4, 4.9],
            "j15_true": [34.0, 33.0, 32.0],
            "j15_pred": [34.0, 32.7, 31.9],
        }
    ).to_parquet(predictions_dir / "cv_predictions.parquet", index=False)

    outputs = script_module.generate_visual_report(
        metrics_dir=metrics_dir,
        predictions_path=predictions_dir / "cv_predictions.parquet",
        figures_dir=figures_dir,
        reports_dir=reports_dir,
    )

    assert sorted(path.name for path in outputs["figures"]) == [
        "actual_vs_predicted_j15.png",
        "actual_vs_predicted_j9.png",
        "model_comparison_mae.png",
        "qq_residuals_j15.png",
        "residuals_vs_predicted_j15.png",
        "residuals_vs_predicted_j9.png",
    ]
    assert outputs["html_path"].exists()
    html = outputs["html_path"].read_text(encoding="utf-8")
    assert "comparison_best_j9_candidate" in html
    assert "chemistry-only pool" in html
