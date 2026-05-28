"""Smoke test for scripts/generate_eda_report.py.

Runs main() on the real cleaned parquet files and verifies the three
output artifacts exist with sane structure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_eda_report.py"
DATA_DIR = REPO_ROOT / "data"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_eda_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_eda_report"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def eda_module():
    return _load_script_module()


@pytest.fixture(scope="module")
def eda_outputs(eda_module, tmp_path_factory):
    if not (DATA_DIR / "jominy_cleaned.parquet").exists():
        pytest.skip("jominy_cleaned.parquet not present; run scripts/clean_jominy.py first.")
    out_dir = tmp_path_factory.mktemp("eda")
    return eda_module.main(data_dir=DATA_DIR, output_dir=out_dir)


def test_pdf_exists_and_has_pages(eda_outputs):
    pdf = eda_outputs["pdf"]
    assert pdf.exists()
    assert pdf.stat().st_size > 5_000


def test_html_exists_and_is_self_contained(eda_outputs):
    html = eda_outputs["html"]
    assert html.exists()
    body = html.read_text(encoding="utf-8")
    # Plotly bundle should be inlined in the first figure block.
    assert "plotly" in body.lower()
    # TL;DR block should be present.
    assert "TL;DR" in body
    # All seven family titles should appear.
    expected_titles = [
        "Hardness distributions",
        "Element distributions",
        "Missingness analysis",
        "J9 vs J15 monotonicity",
        "Element correlation",
        "Element vs hardness",
        "Distribution shift",
    ]
    for title in expected_titles:
        assert title in body, f"missing section: {title}"
    # Each family must have a notes block with all three subsections.
    assert body.count("<section class='notes'>") == 7
    assert body.count("<h3>Description</h3>") == 7
    assert body.count("<h3>Insights</h3>") == 7
    assert body.count("<h3>Recommendations</h3>") == 7


def test_all_seven_pngs_exist(eda_outputs):
    figures_dir = eda_outputs["figures_dir"]
    expected = {
        "hardness_histograms.png",
        "element_histograms.png",
        "missingness.png",
        "monotonicity.png",
        "correlation.png",
        "element_vs_hardness.png",
        "distribution_shift.png",
    }
    found = {p.name for p in figures_dir.glob("*.png")}
    assert expected <= found, f"missing PNGs: {expected - found}"
    for name in expected:
        assert (figures_dir / name).stat().st_size > 1_000


def test_summary_and_tldr_lines(eda_module):
    cleaned, chem_only = eda_module.load_data(DATA_DIR)
    summary = eda_module.compute_summary(cleaned, chem_only)
    assert summary["n_labeled_specimens"] > 0
    assert summary["n_unlabeled_specimens"] > summary["n_labeled_specimens"]
    assert summary["mono_pct"] == pytest.approx(100.0)
    assert len(eda_module.tldr_lines(summary)) == 5
