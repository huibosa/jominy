# pyright: basic

from importlib import import_module


def test_build_pls_pipeline_uses_explicit_external_scaling() -> None:
    modeling_pipelines = import_module("modeling.pipelines")

    pipe = modeling_pipelines.build_pls_pipeline(["C", "Mn"], n_components=2)

    preprocessor = pipe.named_steps["preprocessor"]
    transformer_name, transformer, columns = preprocessor.transformers[0]

    assert transformer_name == "num"
    assert columns == ["C", "Mn"]
    assert [step_name for step_name, _ in transformer.steps] == ["imputer", "scaler"]
    assert pipe.named_steps["model"].scale is False


def test_build_hist_gbr_pipeline_uses_shallow_small_dataset_settings() -> None:
    modeling_pipelines = import_module("modeling.pipelines")

    pipe = modeling_pipelines.build_hist_gbr_pipeline(["C", "Mn"])

    preprocessor = pipe.named_steps["preprocessor"]
    transformer_name, transformer, columns = preprocessor.transformers[0]
    model = pipe.named_steps["model"]

    assert transformer_name == "num"
    assert columns == ["C", "Mn"]
    assert transformer.strategy == "median"
    assert model.learning_rate == 0.05
    assert model.max_depth == 3
    assert model.max_iter == 200
    assert model.max_leaf_nodes == 15
    assert model.min_samples_leaf == 10
