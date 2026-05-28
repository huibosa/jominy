# pyright: basic

import numpy as np
import pandas as pd

from .config import GROUP_COL
from .evaluate import regression_metrics


def metrics_row(model_name: str, target: str, fold, metrics: dict) -> dict:
    return {"fold": fold, "model": model_name, "target": target, **metrics}


def reuse_outer_folds(target_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    fold_map = source_df[[GROUP_COL, "outer_fold"]].drop_duplicates()
    if fold_map[GROUP_COL].duplicated().any():
        raise ValueError("source_df must provide exactly one outer_fold per base_heat_id")

    folded = target_df.merge(fold_map, on=GROUP_COL, how="left")
    if folded["outer_fold"].isna().any():
        missing_groups = sorted(folded.loc[folded["outer_fold"].isna(), GROUP_COL].unique().tolist())
        raise ValueError(f"Missing outer_fold assignments for groups: {missing_groups}")

    folded["outer_fold"] = folded["outer_fold"].astype(int)
    return folded


def make_splits_from_outer_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    if "outer_fold" not in df.columns:
        raise ValueError("df must contain an outer_fold column")

    splits = []
    fold_ids = sorted(df["outer_fold"].unique().tolist())
    for fold_id in fold_ids:
        valid_idx = np.flatnonzero(df["outer_fold"].to_numpy() == fold_id)
        train_idx = np.flatnonzero(df["outer_fold"].to_numpy() != fold_id)
        splits.append((train_idx, valid_idx))
    return splits


def run_mean_baseline(df: pd.DataFrame, target: str, splits) -> pd.DataFrame:
    rows = []
    for fold_id, (train_idx, valid_idx) in enumerate(splits):
        train = df.iloc[train_idx]
        valid = df.iloc[valid_idx]
        pred = np.repeat(train[target].mean(), len(valid))
        rows.append(metrics_row("mean", target, fold_id, regression_metrics(valid[target], pred)))
    return pd.DataFrame(rows)


def run_sklearn_cv(df: pd.DataFrame, target: str, feature_names, splits, build_pipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    for fold_id, (train_idx, valid_idx) in enumerate(splits):
        train = df.iloc[train_idx]
        valid = df.iloc[valid_idx]

        pipe = build_pipeline(feature_names)
        pipe.fit(train[list(feature_names)], train[target])
        pred = np.asarray(pipe.predict(valid[list(feature_names)])).reshape(-1)

        rows.append(
            metrics_row(
                pipe.named_steps["model"].__class__.__name__.lower(),
                target,
                fold_id,
                regression_metrics(valid[target], pred),
            )
        )

        keep_cols = [column for column in ["炉号", GROUP_COL, "outer_fold"] if column in valid.columns]
        fold_preds = valid[keep_cols].copy()
        fold_preds[f"{target}_true"] = valid[target].to_numpy()
        fold_preds[f"{target}_pred"] = pred
        preds.append(fold_preds)

    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarize_metrics(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    summary = (
        df.groupby([label_col, "target"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
        )
        .sort_values(["target", "mae_mean", label_col], ignore_index=True)
    )
    summary["mae_std"] = summary["mae_std"].fillna(0.0)
    return summary
