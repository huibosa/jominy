from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(filename: str, module_name: str):
    script_path = PROJECT_ROOT / "scripts" / filename
    spec = spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_robustness_inputs_reuses_j9_outer_folds_for_subsets() -> None:
    pandas = __import__("pandas")
    module = _load_script_module("run_robustness_checks.py", "run_robustness_checks")

    j9_df = pandas.DataFrame(
        {
            "炉号": [f"S{i}" for i in range(10)],
            "base_heat_id": [f"H{i // 2}" for i in range(10)],
            "J9": [40.0 + i for i in range(10)],
            "has_pair": [True, False, True, False, True, True, False, True, True, False],
        }
    )
    delta_df = pandas.DataFrame(
        {
            "炉号": ["S0", "S2", "S4", "S5", "S7", "S8"],
            "base_heat_id": ["H0", "H1", "H2", "H2", "H3", "H4"],
            "delta": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        }
    )

    j9_folded, j9_paired_only, delta_folded = module.prepare_robustness_inputs(j9_df, delta_df)

    j9_fold_map = j9_folded.set_index("炉号")["outer_fold"].to_dict()
    assert j9_paired_only["outer_fold"].tolist() == [j9_fold_map[heat] for heat in j9_paired_only["炉号"]]
    assert delta_folded["outer_fold"].tolist() == [j9_fold_map[heat] for heat in delta_folded["炉号"]]
    assert (delta_folded.groupby("base_heat_id")["outer_fold"].nunique() == 1).all()


def test_build_distribution_shift_markdown_includes_extrapolation_warning() -> None:
    pandas = __import__("pandas")
    module = _load_script_module("analyze_distribution_shift.py", "analyze_distribution_shift")

    summary = pandas.DataFrame(
        {
            "feature": ["C", "Cr"],
            "labeled_mean": [0.2, 1.1],
            "chemistry_only_mean": [0.45, 0.4],
            "mean_gap": [-0.25, 0.7],
            "labeled_min": [0.1, 0.8],
            "chemistry_only_min": [0.2, 0.1],
            "labeled_max": [0.3, 1.4],
            "chemistry_only_max": [0.7, 0.8],
        }
    )

    markdown = module.build_distribution_shift_markdown(summary)

    assert "Extrapolation warning" in markdown
    assert "chemistry-only pool" in markdown
    assert "not a validated supervised test set" in markdown
    assert "outside the labeled specimen-level chemistry coverage" in markdown
