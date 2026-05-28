# pyright: basic

import pandas as pd

from modeling.config import DELTA_DATASET_PATH, J9_DATASET_PATH, METRICS_DIR, PREDICTIONS_DIR
from modeling.evaluate import clipped_delta_metrics, monotonic_violation_rate, regression_metrics
from modeling.features import FULL_FEATURES
from modeling.pipelines import build_ridge_pipeline
from modeling.predict import assemble_pair_predictions
from modeling.splits import attach_outer_fold_ids, save_outer_splits
from modeling.train import make_splits_from_outer_folds, metrics_row, reuse_outer_folds, run_mean_baseline, run_sklearn_cv


def main() -> None:
    j9_df = attach_outer_fold_ids(pd.read_parquet(J9_DATASET_PATH))
    j9_splits = make_splits_from_outer_folds(j9_df)
    save_outer_splits(j9_df, j9_splits)

    delta_df = reuse_outer_folds(pd.read_parquet(DELTA_DATASET_PATH), j9_df)
    delta_splits = make_splits_from_outer_folds(delta_df)

    j9_mean = run_mean_baseline(j9_df, target="J9", splits=j9_splits)
    delta_mean = run_mean_baseline(delta_df, target="delta", splits=delta_splits)

    j9_ridge: pd.DataFrame
    j9_preds: pd.DataFrame
    j9_ridge, j9_preds = run_sklearn_cv(
        j9_df,
        target="J9",
        feature_names=FULL_FEATURES,
        splits=j9_splits,
        build_pipeline=lambda columns: build_ridge_pipeline(columns, alpha=1.0),
    )
    delta_ridge: pd.DataFrame
    delta_preds: pd.DataFrame
    delta_ridge, delta_preds = run_sklearn_cv(
        delta_df,
        target="delta",
        feature_names=FULL_FEATURES,
        splits=delta_splits,
        build_pipeline=lambda columns: build_ridge_pipeline(columns, alpha=1.0),
    )

    pair_preds = (
        j9_preds.rename(columns={"J9_true": "j9_true", "J9_pred": "j9_pred"})
        .merge(
            delta_preds[["炉号", "delta_true", "delta_pred"]],
            on="炉号",
            how="inner",
        )
        .sort_values(["outer_fold", "炉号"])
        .reset_index(drop=True)
    )
    pair_preds = assemble_pair_predictions(pair_preds)

    j15_rows = []
    clipped_delta_rows = []
    monotonic_rows = []
    outer_folds = [int(fold_id) for fold_id in sorted(pair_preds["outer_fold"].astype(int).unique().tolist())]
    for fold_id in outer_folds:
        fold_df = pair_preds.loc[pair_preds["outer_fold"] == fold_id].reset_index(drop=True)
        j15_rows.append(metrics_row("ridge", "J15", fold_id, regression_metrics(fold_df["j15_true"], fold_df["j15_pred"])))
        clipped_delta_rows.append(metrics_row("ridge", "delta_clipped", fold_id, clipped_delta_metrics(fold_df)))
        monotonic_rows.append(
            {
                "fold": fold_id,
                "model": "ridge",
                "target": "monotonicity",
                "violation_rate": monotonic_violation_rate(fold_df),
            }
        )

    if monotonic_violation_rate(pair_preds) != 0.0:
        raise ValueError("Post-processed predictions must satisfy J9 >= J15 for every row")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pd.concat(
        [
            j9_mean,
            delta_mean,
            j9_ridge,
            delta_ridge,
            pd.DataFrame(j15_rows),
            pd.DataFrame(clipped_delta_rows),
            pd.DataFrame(monotonic_rows),
        ],
        ignore_index=True,
    ).to_csv(METRICS_DIR / "baselines.csv", index=False)
    pair_preds.to_parquet(PREDICTIONS_DIR / "cv_predictions.parquet", index=False)
    print("Wrote baseline metrics and predictions")


if __name__ == "__main__":
    main()
