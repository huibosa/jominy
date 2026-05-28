# pyright: basic

import json
from importlib import import_module


class LinearFeatureModel:
    def __init__(self, feature_name: str, offset: float = 0.0) -> None:
        self.feature_name = feature_name
        self.offset = offset

    def predict(self, frame):
        return frame[self.feature_name] + self.offset


def test_assemble_pair_predictions_clips_negative_delta_to_zero() -> None:
    pd = import_module("pandas")
    modeling_predict = import_module("modeling.predict")
    df = pd.DataFrame(
        {
            "base_heat_id": ["H1", "H2"],
            "j9_pred": [40.0, 35.0],
            "delta_pred": [-1.5, 4.0],
        }
    )

    result = modeling_predict.assemble_pair_predictions(df)

    assert result.loc[0, "delta_pred_clipped"] == 0.0
    assert result.loc[0, "j15_pred"] == 40.0
    assert result.loc[1, "j15_pred"] == 31.0
    assert (result["j9_pred"] >= result["j15_pred"]).all()


def test_assemble_pair_predictions_adds_j15_true_when_truth_columns_present() -> None:
    pd = import_module("pandas")
    modeling_predict = import_module("modeling.predict")
    df = pd.DataFrame(
        {
            "j9_true": [40.0, 35.0],
            "delta_true": [2.0, 4.0],
            "j9_pred": [39.5, 34.5],
            "delta_pred": [1.5, -1.0],
        }
    )

    result = modeling_predict.assemble_pair_predictions(df)

    assert result["j15_true"].tolist() == [38.0, 31.0]
    assert result["j15_pred"].tolist() == [38.0, 34.5]


def test_regression_metrics_returns_expected_values() -> None:
    modeling_evaluate = import_module("modeling.evaluate")
    metrics = modeling_evaluate.regression_metrics([1.0, 2.0, 3.0], [1.0, 3.0, 2.0])

    assert metrics == {"mae": 2.0 / 3.0, "rmse": (2.0 / 3.0) ** 0.5, "r2": 0.0}


def test_monotonic_violation_rate_counts_j15_above_j9() -> None:
    pd = import_module("pandas")
    modeling_evaluate = import_module("modeling.evaluate")
    df = pd.DataFrame(
        {
            "j9_pred": [40.0, 35.0, 30.0],
            "j15_pred": [39.0, 36.0, 31.0],
        }
    )

    assert modeling_evaluate.monotonic_violation_rate(df) == 2.0 / 3.0


def test_clipped_delta_metrics_uses_clipped_predictions() -> None:
    pd = import_module("pandas")
    modeling_evaluate = import_module("modeling.evaluate")
    df = pd.DataFrame(
        {
            "delta_true": [1.0, 3.0],
            "delta_pred_clipped": [0.0, 5.0],
        }
    )

    metrics = modeling_evaluate.clipped_delta_metrics(df)

    assert metrics == {"mae": 1.5, "rmse": 2.5**0.5, "r2": -1.5}


def test_load_model_pair_reads_exported_models_and_manifest(tmp_path) -> None:
    joblib = import_module("joblib")
    modeling_predict = import_module("modeling.predict")

    expected_j9_model = LinearFeatureModel("C", offset=1.0)
    expected_delta_model = LinearFeatureModel("Mn", offset=-0.5)
    expected_manifest = {
        "feature_names": ["C", "Mn"],
        "selection_policy": "fixed_v1_ridge",
        "postprocessing": "J15 = J9 - max(0, delta) guarantees J9 >= J15",
    }

    joblib.dump(expected_j9_model, tmp_path / "final_j9_model.joblib")
    joblib.dump(expected_delta_model, tmp_path / "final_delta_model.joblib")
    (tmp_path / "feature_manifest.json").write_text(json.dumps(expected_manifest), encoding="utf-8")

    j9_model, delta_model, manifest = modeling_predict.load_model_pair(tmp_path)

    assert j9_model.predict.__self__.feature_name == "C"
    assert delta_model.predict.__self__.feature_name == "Mn"
    assert manifest == expected_manifest


def test_predict_j9_j15_reconstructs_monotonic_pair_predictions() -> None:
    pd = import_module("pandas")
    modeling_predict = import_module("modeling.predict")
    features = pd.DataFrame(
        {
            "炉号": ["H1", "H2"],
            "C": [40.0, 35.0],
            "Mn": [-2.0, 4.0],
            "unused": [1, 2],
        }
    )

    result = modeling_predict.predict_j9_j15(
        features=features,
        j9_model=LinearFeatureModel("C"),
        delta_model=LinearFeatureModel("Mn"),
        feature_names=["C", "Mn"],
    )

    assert result["炉号"].tolist() == ["H1", "H2"]
    assert result["j9_pred"].tolist() == [40.0, 35.0]
    assert result["delta_pred"].tolist() == [-2.0, 4.0]
    assert result["delta_pred_clipped"].tolist() == [0.0, 4.0]
    assert result["j15_pred"].tolist() == [40.0, 31.0]
    assert (result["j9_pred"] >= result["j15_pred"]).all()
