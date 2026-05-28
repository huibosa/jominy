# pyright: basic

import json

import pandas as pd

from modeling.config import FOLDS_DIR, GROUP_COL, METRICS_DIR, REPORTS_DIR
from modeling.config import DELTA_DATASET_PATH, J9_DATASET_PATH
from modeling.features import CORE_PLUS_FLAGS, FULL_FEATURES
from modeling.pipelines import build_ridge_pipeline
from modeling.splits import attach_outer_fold_ids
from modeling.train import make_splits_from_outer_folds, reuse_outer_folds, run_sklearn_cv, summarize_metrics


def load_saved_fold_map() -> pd.DataFrame:
    folds_path = FOLDS_DIR / "outer_folds.json"
    if not folds_path.exists():
        return pd.DataFrame(columns=[GROUP_COL, "outer_fold"])

    payload = json.loads(folds_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for fold_entry in payload:
        fold_id = int(fold_entry["fold"])
        for group in fold_entry.get("valid_groups", []):
            rows.append({GROUP_COL: group, "outer_fold": fold_id})

    fold_map = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    if not fold_map.empty and fold_map[GROUP_COL].duplicated().any():
        raise ValueError("Saved outer_folds.json must contain one fold assignment per base_heat_id")
    return fold_map


def prepare_robustness_inputs(j9_df: pd.DataFrame, delta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    saved_fold_map = load_saved_fold_map()
    if saved_fold_map.empty:
        j9_folded = attach_outer_fold_ids(j9_df)
    else:
        try:
            j9_folded = reuse_outer_folds(j9_df, saved_fold_map)
        except ValueError:
            j9_folded = attach_outer_fold_ids(j9_df)

    j9_paired_only = reuse_outer_folds(j9_df.loc[j9_df["has_pair"]].reset_index(drop=True), j9_folded)
    delta_folded = reuse_outer_folds(delta_df, j9_folded)
    return j9_folded, j9_paired_only, delta_folded


def build_robustness_report(summary: pd.DataFrame) -> str:
    table = summary.to_markdown(index=False)
    return "\n".join(
        [
            "# Jominy Robustness Checks",
            "",
            "These checks keep the Task 4 shared outer-fold behavior intact:",
            "- J9 folds reuse the saved J9-derived assignment when available.",
            "- Paired-only J9 and delta subsets inherit those same outer folds instead of recomputing subset-specific grouped CV.",
            "",
            table,
            "",
        ]
    )


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    j9_df = pd.read_parquet(J9_DATASET_PATH)
    delta_df = pd.read_parquet(DELTA_DATASET_PATH)
    j9_folded, j9_paired_only, delta_folded = prepare_robustness_inputs(j9_df, delta_df)

    checks: list[pd.DataFrame] = []
    for label, dataframe, feature_names, target in [
        ("j9_core", j9_folded, CORE_PLUS_FLAGS, "J9"),
        ("j9_full", j9_folded, FULL_FEATURES, "J9"),
        ("j9_paired_only", j9_paired_only, FULL_FEATURES, "J9"),
        ("delta_core", delta_folded, CORE_PLUS_FLAGS, "delta"),
        ("delta_full", delta_folded, FULL_FEATURES, "delta"),
    ]:
        metrics_df, _ = run_sklearn_cv(
            dataframe,
            target=target,
            feature_names=feature_names,
            splits=make_splits_from_outer_folds(dataframe),
            build_pipeline=lambda columns: build_ridge_pipeline(columns, alpha=1.0),
        )
        metrics_df["label"] = label
        checks.append(metrics_df)

    result = pd.concat(checks, ignore_index=True)
    summary = summarize_metrics(result, label_col="label")
    result.to_csv(METRICS_DIR / "robustness.csv", index=False)
    (REPORTS_DIR / "robustness.md").write_text(build_robustness_report(summary), encoding="utf-8")
    print(f"Wrote robustness metrics to {METRICS_DIR / 'robustness.csv'}")
    print(f"Wrote robustness report to {REPORTS_DIR / 'robustness.md'}")


if __name__ == "__main__":
    main()
