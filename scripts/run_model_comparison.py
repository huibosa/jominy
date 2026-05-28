# pyright: basic

import pandas as pd

from modeling.config import DELTA_DATASET_PATH, J9_DATASET_PATH, METRICS_DIR, REPORTS_DIR
from modeling.features import CORE_PLUS_FLAGS, FULL_FEATURES
from modeling.pipelines import build_hist_gbr_pipeline, build_pls_pipeline, build_ridge_pipeline
from modeling.reporting import add_candidate_notes, write_markdown_table
from modeling.splits import attach_outer_fold_ids
from modeling.train import make_splits_from_outer_folds, reuse_outer_folds, run_sklearn_cv, summarize_metrics


def evaluate_family(df: pd.DataFrame, target: str, splits) -> pd.DataFrame:
    runs: list[pd.DataFrame] = []
    challengers = [
        ("ridge_core", CORE_PLUS_FLAGS, lambda columns: build_ridge_pipeline(columns, alpha=1.0)),
        ("ridge_full", FULL_FEATURES, lambda columns: build_ridge_pipeline(columns, alpha=1.0)),
        ("pls_full", FULL_FEATURES, lambda columns: build_pls_pipeline(columns, n_components=4)),
        ("hgbr_full", FULL_FEATURES, build_hist_gbr_pipeline),
    ]

    for label, feature_names, builder in challengers:
        metrics_df, _ = run_sklearn_cv(
            df,
            target=target,
            feature_names=feature_names,
            splits=splits,
            build_pipeline=builder,
        )
        metrics_df["label"] = label
        runs.append(metrics_df)

    return pd.concat(runs, ignore_index=True)


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    j9_df = attach_outer_fold_ids(pd.read_parquet(J9_DATASET_PATH))
    j9_splits = make_splits_from_outer_folds(j9_df)

    delta_df = reuse_outer_folds(pd.read_parquet(DELTA_DATASET_PATH), j9_df)
    delta_splits = make_splits_from_outer_folds(delta_df)

    comparison_frames: list[pd.DataFrame] = [
        evaluate_family(j9_df, target="J9", splits=j9_splits),
        evaluate_family(delta_df, target="delta", splits=delta_splits),
    ]
    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison = add_candidate_notes(comparison)

    summary = summarize_metrics(comparison, label_col="label")
    summary = add_candidate_notes(summary)
    summary = summary[
        [
            "label",
            "target",
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "r2_mean",
            "selection_policy",
            "candidate_note",
        ]
    ]

    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    write_markdown_table(REPORTS_DIR / "model_comparison.md", "Jominy Model Comparison", summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
