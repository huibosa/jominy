# pyright: basic

from importlib import import_module


def test_summarize_metrics_aggregates_by_label_and_target() -> None:
    pd = import_module("pandas")
    pytest = import_module("pytest")
    modeling_train = import_module("modeling.train")

    metrics = pd.DataFrame(
        {
            "label": ["ridge_full", "ridge_full", "pls_full", "pls_full"],
            "target": ["J9", "J9", "J9", "J9"],
            "mae": [1.0, 2.0, 0.5, 1.5],
            "rmse": [1.1, 2.1, 0.7, 1.7],
            "r2": [0.1, 0.3, 0.2, 0.4],
        }
    )

    summary = modeling_train.summarize_metrics(metrics, label_col="label")

    assert summary["label"].tolist() == ["pls_full", "ridge_full"]
    assert summary["target"].tolist() == ["J9", "J9"]
    assert summary["mae_mean"].tolist() == [1.0, 1.5]
    assert summary["rmse_mean"].tolist() == [1.2, 1.6]
    assert summary["r2_mean"].tolist() == [pytest.approx(0.3), pytest.approx(0.2)]
    assert summary["mae_std"].tolist() == [pytest.approx(0.70710678), pytest.approx(0.70710678)]


def test_reuse_outer_folds_assigns_delta_rows_from_j9_group_map() -> None:
    pd = import_module("pandas")
    modeling_train = import_module("modeling.train")

    j9_df = pd.DataFrame(
        {
            "base_heat_id": ["H1", "H2", "H3"],
            "outer_fold": [0, 1, 0],
        }
    )
    delta_df = pd.DataFrame(
        {
            "炉号": ["S1", "S2"],
            "base_heat_id": ["H1", "H2"],
            "delta": [1.0, 2.0],
        }
    )

    reused = modeling_train.reuse_outer_folds(delta_df, j9_df)

    assert reused["outer_fold"].tolist() == [0, 1]
