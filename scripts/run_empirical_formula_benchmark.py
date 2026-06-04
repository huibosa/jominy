# pyright: basic
"""Benchmark Sheet 1 empirical formulas against project model baselines.

Run from the repository root:
    uv run --with pandas,pyarrow,scikit-learn,xgboost scripts/run_empirical_formula_benchmark.py

The project dependency group also works:
    uv run --group backend-build python scripts/run_empirical_formula_benchmark.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from modeling.config import (  # noqa: E402
    DELTA_DATASET_PATH,
    FOLDS_DIR,
    GROUP_COL,
    J9_DATASET_PATH,
    METRICS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)
from modeling.features import FULL_FEATURES  # noqa: E402

FOLDS_PATH = FOLDS_DIR / "outer_folds.json"
BASELINES_PATH = METRICS_DIR / "baselines.csv"
BLEND_OOF_PATH = PROJECT_ROOT / "output" / "modeling" / "predictions" / "blend_oof.csv"
PER_FOLD_PATH = METRICS_DIR / "empirical_formula_per_fold.csv"
MONOTONICITY_PATH = METRICS_DIR / "empirical_formula_monotonicity.csv"
REPORT_PATH = REPORTS_DIR / "empirical_formula_comparison.md"

MO_VALUES = (0.0, 0.01)
NI_MODES = ("ni_zero", "ni_median_fold", "ni_complete_case")
BLEND_LABEL = "blend_xgb_pls3_w0.70"
FORMULA_LABEL = "formula_d20_d21"
F20_LABEL = "formula_f20"


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_outer_folds() -> list[dict[str, Any]]:
    with FOLDS_PATH.open("r", encoding="utf-8") as handle:
        folds = json.load(handle)
    if len(folds) != 5:
        raise ValueError(f"Expected 5 outer folds in {FOLDS_PATH}, found {len(folds)}")
    return folds


def attach_saved_outer_folds(df: pd.DataFrame, folds: list[dict[str, Any]]) -> pd.DataFrame:
    spec_to_fold: dict[str, int] = {}
    for fold in folds:
        fold_id = int(fold["fold"])
        for spec in fold["valid_specs"]:
            if spec in spec_to_fold:
                raise ValueError(f"Specimen {spec!r} appears in more than one fold")
            spec_to_fold[str(spec)] = fold_id

    out = df.copy()
    out["outer_fold"] = out["炉号"].map(spec_to_fold)
    if out["outer_fold"].isna().any():
        missing = sorted(out.loc[out["outer_fold"].isna(), "炉号"].astype(str).tolist())
        raise ValueError(f"Rows missing saved outer_fold assignment: {missing[:10]}")
    out["outer_fold"] = out["outer_fold"].astype(int)
    return out


def fold_train_valid(df: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_groups = set(fold["train_groups"])
    valid_specs = set(fold["valid_specs"])
    train = df.loc[df[GROUP_COL].isin(train_groups)].copy()
    valid = df.loc[df["炉号"].isin(valid_specs)].copy()
    if valid.empty:
        raise ValueError(f"No validation rows for fold {fold['fold']}")
    return train, valid


def checked_median(df: pd.DataFrame, column: str, fold_id: int) -> float:
    value = df[column].median(skipna=True)
    if pd.isna(value):
        raise ValueError(f"Fold {fold_id} has no non-null training values for {column}")
    return float(value)


def ni_values(valid: pd.DataFrame, train: pd.DataFrame, mode: str, fold_id: int) -> pd.Series:
    if mode == "ni_zero":
        return valid["Ni"].fillna(0.0)
    if mode == "ni_median_fold":
        return valid["Ni"].fillna(checked_median(train, "Ni", fold_id))
    if mode == "ni_complete_case":
        return valid["Ni"]
    raise ValueError(f"Unknown Ni handling mode: {mode}")


def predict_d20_d21(valid: pd.DataFrame, train: pd.DataFrame, mo_value: float, ni_mode: str, fold_id: int) -> tuple[pd.Series, pd.Series]:
    ni = ni_values(valid, train, ni_mode, fold_id)
    j9 = (
        15.0674
        + 33.20605 * valid["C"]
        + 8.109032 * valid["Mn"]
        - 23.10789 * valid["S"]
        + 5.1356 * valid["Cr"]
        - 52.31636 * mo_value
        + 7.402463 * ni
        + 8.935094 * valid["Cu"]
    )
    j15 = (
        7.624122
        + 22.65873 * valid["C"]
        + 8.872051 * valid["Mn"]
        - 32.4852 * valid["S"]
        + 5.171636 * valid["Cr"]
        - 47.65877 * mo_value
        + 6.868623 * ni
        + 12.69098 * valid["Cu"]
    )
    return j9, j15


def predict_f20(valid: pd.DataFrame, train: pd.DataFrame, mo_value: float, ni_mode: str, fold_id: int) -> pd.Series:
    ni = ni_values(valid, train, ni_mode, fold_id)
    w = valid["W"].fillna(checked_median(train, "W", fold_id))
    return (
        14.70641
        + 32.46597 * valid["C"]
        + 8.183352 * valid["Mn"]
        + 2.545434 * valid["Si"]
        + 2.904236 * valid["P"]
        - 23.44407 * valid["S"]
        - 26.30782 * w
        + 5.295791 * valid["Cr"]
        - 53.94057 * mo_value
        + 8.173319 * ni
        - 4.605883 * valid["Ti"]
        + 8.82092 * valid["Cu"]
    )


def metric_row(
    *,
    label: str,
    target: str,
    fold: int,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    eval_scope: str,
    ni_handling: str | None,
    mo_value: float | None,
    source: str,
) -> dict[str, Any]:
    if len(y_true) == 0:
        raise ValueError(f"No rows to score for {label}/{target}/fold={fold}/{eval_scope}")
    metrics = regression_metrics(y_true, y_pred)
    return {
        "label": label,
        "target": target,
        "fold": fold,
        **metrics,
        "n_rows": int(len(y_true)),
        "eval_scope": eval_scope,
        "ni_handling": ni_handling,
        "mo_value": mo_value,
        "source": source,
    }


def monotonicity_row(
    *,
    label: str,
    fold: int,
    j9_pred: pd.Series | np.ndarray,
    j15_pred: pd.Series | np.ndarray,
    ni_handling: str | None,
    mo_value: float | None,
    source: str,
) -> dict[str, Any]:
    if len(j9_pred) == 0:
        raise ValueError(f"No paired rows for monotonicity: {label}/fold={fold}")
    return {
        "label": label,
        "fold": fold,
        "n_pairs": int(len(j9_pred)),
        "violation_rate": float((np.asarray(j15_pred) > np.asarray(j9_pred)).mean()),
        "ni_handling": ni_handling,
        "mo_value": mo_value,
        "source": source,
    }


def evaluate_formulas(j9_df: pd.DataFrame, folds: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    mono_rows: list[dict[str, Any]] = []

    for ni_mode in NI_MODES:
        eval_scope = "complete_case_ni" if ni_mode == "ni_complete_case" else "full"
        for mo_value in MO_VALUES:
            for fold in folds:
                fold_id = int(fold["fold"])
                train, valid = fold_train_valid(j9_df, fold)
                valid_eval = valid.dropna(subset=["Ni"]).copy() if ni_mode == "ni_complete_case" else valid

                d20_j9, d21_j15 = predict_d20_d21(valid_eval, train, mo_value, ni_mode, fold_id)
                metric_rows.append(
                    metric_row(
                        label=FORMULA_LABEL,
                        target="J9",
                        fold=fold_id,
                        y_true=valid_eval["J9"],
                        y_pred=d20_j9,
                        eval_scope=eval_scope,
                        ni_handling=ni_mode,
                        mo_value=mo_value,
                        source="recomputed_here",
                    )
                )

                j15_eval = valid_eval.loc[valid_eval["J15"].notna()]
                metric_rows.append(
                    metric_row(
                        label=FORMULA_LABEL,
                        target="J15",
                        fold=fold_id,
                        y_true=j15_eval["J15"],
                        y_pred=d21_j15.loc[j15_eval.index],
                        eval_scope=eval_scope,
                        ni_handling=ni_mode,
                        mo_value=mo_value,
                        source="recomputed_here",
                    )
                )

                mono_rows.append(
                    monotonicity_row(
                        label=FORMULA_LABEL,
                        fold=fold_id,
                        j9_pred=d20_j9.loc[j15_eval.index],
                        j15_pred=d21_j15.loc[j15_eval.index],
                        ni_handling=ni_mode,
                        mo_value=mo_value,
                        source="recomputed_here",
                    )
                )

                f20_j9 = predict_f20(valid_eval, train, mo_value, ni_mode, fold_id)
                metric_rows.append(
                    metric_row(
                        label=F20_LABEL,
                        target="J9",
                        fold=fold_id,
                        y_true=valid_eval["J9"],
                        y_pred=f20_j9,
                        eval_scope=eval_scope,
                        ni_handling=ni_mode,
                        mo_value=mo_value,
                        source="recomputed_here",
                    )
                )

    return metric_rows, mono_rows


def row_count_for_fold(df: pd.DataFrame, target: str, fold_id: int) -> int:
    subset = df.loc[df["outer_fold"] == fold_id]
    if target in subset.columns:
        subset = subset.loc[subset[target].notna()]
    return int(len(subset))


def add_mean_baseline_rows(j9_df: pd.DataFrame, folds: list[dict[str, Any]], baselines: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mean_j9 = baselines.loc[(baselines["model"] == "mean") & (baselines["target"] == "J9")]
    if len(mean_j9) != 5:
        raise ValueError(f"Expected 5 mean/J9 rows in {BASELINES_PATH}, found {len(mean_j9)}")
    for _, row in mean_j9.sort_values("fold").iterrows():
        fold_id = int(row["fold"])
        rows.append(
            {
                "label": "mean_baseline",
                "target": "J9",
                "fold": fold_id,
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "r2": float(row["r2"]),
                "n_rows": row_count_for_fold(j9_df, "J9", fold_id),
                "eval_scope": "full",
                "ni_handling": None,
                "mo_value": None,
                "source": "baselines.csv",
            }
        )

    for fold in folds:
        fold_id = int(fold["fold"])
        train, valid = fold_train_valid(j9_df, fold)
        train_j15 = train.loc[train["J15"].notna(), "J15"]
        valid_j15 = valid.loc[valid["J15"].notna()]
        pred = np.repeat(float(train_j15.mean()), len(valid_j15))
        rows.append(
            metric_row(
                label="mean_baseline",
                target="J15",
                fold=fold_id,
                y_true=valid_j15["J15"],
                y_pred=pred,
                eval_scope="full",
                ni_handling=None,
                mo_value=None,
                source="recomputed_here",
            )
        )
    return rows


def add_ridge_rows(j9_df: pd.DataFrame, baselines: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    mono_rows: list[dict[str, Any]] = []
    for target in ("J9", "J15"):
        ridge = baselines.loc[(baselines["model"] == "ridge") & (baselines["target"] == target)].copy()
        if len(ridge) != 5:
            raise ValueError(f"Expected 5 ridge/{target} rows in {BASELINES_PATH}, found {len(ridge)}")
        for _, row in ridge.sort_values("fold").iterrows():
            fold_id = int(row["fold"])
            metric_rows.append(
                {
                    "label": "ridge_full",
                    "target": target,
                    "fold": fold_id,
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "r2": float(row["r2"]),
                    "n_rows": row_count_for_fold(j9_df, target, fold_id),
                    "eval_scope": "full",
                    "ni_handling": None,
                    "mo_value": None,
                    "source": "baselines.csv",
                }
            )

    ridge_mono = baselines.loc[(baselines["model"] == "ridge") & (baselines["target"] == "monotonicity")].copy()
    if len(ridge_mono) != 5:
        raise ValueError(f"Expected 5 ridge/monotonicity rows in {BASELINES_PATH}, found {len(ridge_mono)}")
    for _, row in ridge_mono.sort_values("fold").iterrows():
        fold_id = int(row["fold"])
        mono_rows.append(
            {
                "label": "ridge_full",
                "fold": fold_id,
                "n_pairs": row_count_for_fold(j9_df, "J15", fold_id),
                "violation_rate": float(row["violation_rate"]),
                "ni_handling": None,
                "mo_value": None,
                "source": "baselines.csv",
            }
        )
    return metric_rows, mono_rows


def scaled_pipe(model: Any) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                list(FULL_FEATURES),
            )
        ]
    )
    return Pipeline([("preprocessor", pre), ("model", model)])


def imputed_pipe(model: Any) -> Pipeline:
    pre = ColumnTransformer(transformers=[("num", SimpleImputer(strategy="median"), list(FULL_FEATURES))])
    return Pipeline([("preprocessor", pre), ("model", model)])


def fit_predict_j9_blend(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    from xgboost import XGBRegressor

    xgb = imputed_pipe(
        XGBRegressor(
            n_estimators=1500,
            max_depth=3,
            learning_rate=0.005,
            subsample=0.55,
            colsample_bytree=0.5,
            reg_lambda=2.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    )
    pls = scaled_pipe(PLSRegression(n_components=3, scale=False))
    x_train = train[FULL_FEATURES]
    y_train = train["J9"].to_numpy()
    x_valid = valid[FULL_FEATURES]
    xgb.fit(x_train, y_train)
    pls.fit(x_train, y_train)
    return 0.70 * np.asarray(xgb.predict(x_valid)).reshape(-1) + 0.30 * np.asarray(pls.predict(x_valid)).reshape(-1)


def fit_predict_delta_blend(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    from xgboost import XGBRegressor

    xgb = imputed_pipe(
        XGBRegressor(
            n_estimators=800,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    )
    bayes = scaled_pipe(BayesianRidge())
    x_train = train[FULL_FEATURES]
    y_train = train["delta"].to_numpy()
    x_valid = valid[FULL_FEATURES]
    xgb.fit(x_train, y_train)
    bayes.fit(x_train, y_train)
    return 0.60 * np.asarray(xgb.predict(x_valid)).reshape(-1) + 0.40 * np.asarray(bayes.predict(x_valid)).reshape(-1)


def evaluate_blend(
    j9_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    folds: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    mono_rows: list[dict[str, Any]] = []
    blend_pred_frames: list[pd.DataFrame] = []
    fold_seconds: dict[int, float] = {}

    for fold in folds:
        fold_id = int(fold["fold"])
        started = time.perf_counter()
        j9_train, j9_valid = fold_train_valid(j9_df, fold)
        delta_train, delta_valid = fold_train_valid(delta_df, fold)

        j9_pred = fit_predict_j9_blend(j9_train, j9_valid)
        metric_rows.append(
            metric_row(
                label=BLEND_LABEL,
                target="J9",
                fold=fold_id,
                y_true=j9_valid["J9"],
                y_pred=j9_pred,
                eval_scope="full",
                ni_handling=None,
                mo_value=None,
                source="retrained_in_script",
            )
        )

        delta_pred = fit_predict_delta_blend(delta_train, delta_valid)
        j9_pred_df = j9_valid[["炉号", GROUP_COL, "outer_fold", "J9"]].copy()
        j9_pred_df = j9_pred_df.rename(columns={"J9": "j9_true"})
        j9_pred_df["j9_pred"] = j9_pred
        delta_pred_df = delta_valid[["炉号", "delta"]].copy().rename(columns={"delta": "delta_true"})
        delta_pred_df["delta_pred"] = delta_pred
        pair_preds = j9_pred_df.merge(delta_pred_df, on="炉号", how="inner").sort_values("炉号").reset_index(drop=True)
        pair_preds["delta_pred_clipped"] = pair_preds["delta_pred"].clip(lower=0.0)
        pair_preds["j15_pred"] = pair_preds["j9_pred"] - pair_preds["delta_pred_clipped"]
        pair_preds["j15_true"] = pair_preds["j9_true"] - pair_preds["delta_true"]

        metric_rows.append(
            metric_row(
                label=BLEND_LABEL,
                target="J15",
                fold=fold_id,
                y_true=pair_preds["j15_true"],
                y_pred=pair_preds["j15_pred"],
                eval_scope="full",
                ni_handling=None,
                mo_value=None,
                source="retrained_in_script",
            )
        )
        mono_rows.append(
            monotonicity_row(
                label=BLEND_LABEL,
                fold=fold_id,
                j9_pred=pair_preds["j9_pred"],
                j15_pred=pair_preds["j15_pred"],
                ni_handling=None,
                mo_value=None,
                source="retrained_in_script",
            )
        )

        fold_oof = j9_pred_df[["炉号", GROUP_COL, "outer_fold", "j9_true", "j9_pred"]].copy()
        fold_oof = fold_oof.rename(columns={"outer_fold": "fold"})
        blend_pred_frames.append(fold_oof)
        fold_seconds[fold_id] = time.perf_counter() - started

    retrained_oof = pd.concat(blend_pred_frames, ignore_index=True)
    stored_oof = pd.read_csv(BLEND_OOF_PATH)
    fold_ids = sorted(int(fold_id) for fold_id in retrained_oof["fold"].unique().tolist())
    retrained_mae = pd.Series(
        {
            fold_id: float(
                mean_absolute_error(
                    retrained_oof.loc[retrained_oof["fold"] == fold_id, "j9_true"],
                    retrained_oof.loc[retrained_oof["fold"] == fold_id, "j9_pred"],
                )
            )
            for fold_id in fold_ids
        }
    )
    stored_mae = pd.Series(
        {fold_id: float(stored_oof.loc[stored_oof["fold"] == fold_id, "abs_err"].mean()) for fold_id in fold_ids}
    )
    deltas = (retrained_mae - stored_mae).abs()
    max_delta = float(deltas.max())
    bad = deltas.loc[deltas > 0.05]
    if not bad.empty:
        raise RuntimeError(
            "Blend retrain does not agree with stored blend_oof.csv. "
            f"retrained_mae={retrained_mae.to_dict()}, stored_mae={stored_mae.to_dict()}, "
            f"violating_folds={bad.index.astype(int).tolist()}, abs_deltas={deltas.to_dict()}"
        )

    return metric_rows, mono_rows, {"max_fold_mae_delta": max_delta, "fold_seconds": fold_seconds}


def aggregate_metrics(per_fold: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_fold.groupby(["label", "target", "eval_scope", "ni_handling", "mo_value"], dropna=False, as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            n_rows_total=("n_rows", "sum"),
            source=("source", lambda values: ", ".join(sorted(set(str(v) for v in values)))),
        )
        .sort_values(["target", "eval_scope", "mae_mean", "label"], ignore_index=True)
    )
    return grouped


def aggregate_monotonicity(monotonicity: pd.DataFrame) -> pd.DataFrame:
    return (
        monotonicity.groupby(["label", "ni_handling", "mo_value"], dropna=False, as_index=False)
        .agg(
            n_pairs_total=("n_pairs", "sum"),
            violation_rate_mean=("violation_rate", "mean"),
            violation_rate_std=("violation_rate", "std"),
            source=("source", lambda values: ", ".join(sorted(set(str(v) for v in values)))),
        )
        .sort_values(["violation_rate_mean", "label", "mo_value", "ni_handling"], ignore_index=True)
    )


def fmt_float(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_mo(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2f}"


def fmt_text(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return str(value)


def notes_for(label: str, target: str) -> str:
    if label == "mean_baseline":
        return "Mean baseline; J9 from baselines.csv, J15 recomputed here."
    if label == "ridge_full":
        return "Full-feature ridge rows from baselines.csv."
    if label == BLEND_LABEL:
        return "Production blend retrained per outer fold."
    if label == FORMULA_LABEL:
        return "Sheet 1 D20 for J9 and D21 for J15."
    if label == F20_LABEL:
        return "Sheet 1 F20; J9 only."
    return f"Benchmark row for {target}."


def table_from_records(records: list[dict[str, Any]], columns: list[str]) -> str:
    if not records:
        return "_No rows._\n"
    widths = {column: len(column) for column in columns}
    string_rows: list[dict[str, str]] = []
    for record in records:
        string_record: dict[str, str] = {}
        for column in columns:
            value = str(record.get(column, ""))
            value = value.replace("|", "\\|").replace("\n", "<br>")
            string_record[column] = value
            widths[column] = max(widths[column], len(value))
        string_rows.append(string_record)

    header = "| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |"
    separator = "| " + " | ".join("-" * widths[column] for column in columns) + " |"
    body = ["| " + " | ".join(row[column].ljust(widths[column]) for column in columns) + " |" for row in string_rows]
    return "\n".join([header, separator, *body]) + "\n"


def metric_records(summary: pd.DataFrame, target: str, eval_scope: str) -> list[dict[str, Any]]:
    subset = summary.loc[(summary["target"] == target) & (summary["eval_scope"] == eval_scope)].copy()
    label_order = {"mean_baseline": 0, "ridge_full": 1, BLEND_LABEL: 2, FORMULA_LABEL: 3, F20_LABEL: 4}
    ni_order = {"NA": 0, "ni_zero": 1, "ni_median_fold": 2, "ni_complete_case": 3}
    subset["_label_order"] = subset["label"].map(label_order).fillna(99)
    subset["_ni_order"] = subset["ni_handling"].map(lambda value: ni_order.get(fmt_text(value), 99))
    subset = subset.sort_values(["_label_order", "mo_value", "_ni_order", "mae_mean"], na_position="first")

    records: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        records.append(
            {
                "label": row["label"],
                "target": row["target"],
                "ni_handling": fmt_text(row["ni_handling"]),
                "mo_value": fmt_mo(row["mo_value"]),
                "mae_mean ± std": f"{fmt_float(row['mae_mean'])} ± {fmt_float(row['mae_std'])}",
                "rmse_mean": fmt_float(row["rmse_mean"]),
                "r2_mean": fmt_float(row["r2_mean"]),
                "n_rows_total": int(row["n_rows_total"]),
                "notes": notes_for(str(row["label"]), target),
            }
        )
    return records


def monotonicity_records(mono_summary: pd.DataFrame) -> list[dict[str, Any]]:
    label_order = {"ridge_full": 0, BLEND_LABEL: 1, FORMULA_LABEL: 2}
    ni_order = {"NA": 0, "ni_zero": 1, "ni_median_fold": 2, "ni_complete_case": 3}
    subset = mono_summary.copy()
    subset["_label_order"] = subset["label"].map(label_order).fillna(99)
    subset["_ni_order"] = subset["ni_handling"].map(lambda value: ni_order.get(fmt_text(value), 99))
    subset = subset.sort_values(["_label_order", "mo_value", "_ni_order"], na_position="first")
    records: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        records.append(
            {
                "label": row["label"],
                "ni_handling": fmt_text(row["ni_handling"]),
                "mo_value": fmt_mo(row["mo_value"]),
                "violation_rate_mean": f"{100.0 * float(row['violation_rate_mean']):.2f}%",
                "violation_rate_std": f"{100.0 * float(row['violation_rate_std']):.2f}%" if not pd.isna(row["violation_rate_std"]) else "",
                "n_pairs_total": int(row["n_pairs_total"]),
                "source": row["source"],
            }
        )
    return records


def sensitivity_note(summary: pd.DataFrame) -> str:
    full_formula = summary.loc[
        (summary["eval_scope"] == "full")
        & (summary["label"].isin([FORMULA_LABEL, F20_LABEL]))
        & (summary["ni_handling"].isin(["ni_zero", "ni_median_fold"]))
    ]
    pivot = full_formula.pivot_table(
        index=["label", "target", "mo_value"], columns="ni_handling", values="mae_mean", aggfunc="first"
    )
    max_fill_delta = float((pivot["ni_zero"] - pivot["ni_median_fold"]).abs().max())
    return (
        f"Across full-scope formula rows, changing Ni null handling from zero-fill to fold-median fill "
        f"moves mean MAE by at most {max_fill_delta:.4f} HRC. Complete-case rows below use a smaller "
        "population and should be read as a scope sensitivity, not an imputation-only comparison."
    )


def write_report(summary: pd.DataFrame, mono_summary: pd.DataFrame, blend_info: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    j9_records = metric_records(summary, "J9", "full")
    j15_records = metric_records(summary, "J15", "full")
    sensitivity_records = []
    for target in ("J9", "J15"):
        sensitivity_records.extend(metric_records(summary, target, "complete_case_ni"))

    fold_seconds = blend_info["fold_seconds"]
    runtime_text = ", ".join(f"fold {fold}: {seconds:.1f}s" for fold, seconds in sorted(fold_seconds.items()))
    total_runtime = sum(float(seconds) for seconds in fold_seconds.values())

    sections = [
        "# Empirical Formula vs Current Model Benchmark\n",
        "## Protocol\n",
        "- Empirical formulas are hard-coded from `data/empirical-formula.xlsx` Sheet 1 cells D20/D21/F20.\n",
        "- Evaluation uses the saved 5-fold GroupKFold split in `output/modeling/folds/outer_folds.json`.\n",
        "- J9 full scope contains all 566 J9-labelled rows; J15/pair scope contains the 491 rows with J15 labels.\n",
        "- Per-fold metric summaries use pandas sample standard deviation (`ddof=1`).\n",
        "- `formula_f20` is J9-only because Sheet 1 has no F20 J15 counterpart.\n",
        "- `pls_full` / `hgbr_full` J15 rows are intentionally not reported because this repo has no saved J15 artifacts for them.\n\n",
        "## J9 headline table (`eval_scope=full`)\n\n",
        table_from_records(
            j9_records,
            ["label", "ni_handling", "mo_value", "mae_mean ± std", "rmse_mean", "r2_mean", "n_rows_total", "notes"],
        ),
        "\n## J15 table (`eval_scope=full`)\n\n",
        table_from_records(
            j15_records,
            ["label", "ni_handling", "mo_value", "mae_mean ± std", "rmse_mean", "r2_mean", "n_rows_total", "notes"],
        ),
        "\n## Monotonicity on paired rows\n\n",
        "Rows report the fold-mean violation rate for `J15_pred > J9_pred`; ridge and blend are clipped/post-processed and are 0% by construction.\n\n",
        table_from_records(
            monotonicity_records(mono_summary),
            ["label", "ni_handling", "mo_value", "violation_rate_mean", "violation_rate_std", "n_pairs_total", "source"],
        ),
        "\n## Ni sensitivity (`eval_scope=complete_case_ni`)\n\n",
        sensitivity_note(summary),
        "\n\n",
        table_from_records(
            sensitivity_records,
            ["label", "target", "ni_handling", "mo_value", "mae_mean ± std", "rmse_mean", "r2_mean", "n_rows_total", "notes"],
        ),
        "\n## Reproducibility notes\n\n",
        f"- Blend retrain cross-check against `output/modeling/predictions/blend_oof.csv`: max per-fold |ΔMAE| = {blend_info['max_fold_mae_delta']:.6f} HRC.\n",
        f"- Blend retrain wall time in this run: {total_runtime:.1f}s ({runtime_text}).\n",
        "- Ridge rows come from `output/modeling/metrics/baselines.csv`, generated by `scripts/run_baselines.py`.\n",
        "- Blend hyperparameters match `webapp/backend/train_models.py`: J9 = 0.70·XGB + 0.30·PLS(3); delta = 0.60·XGB + 0.40·BayesianRidge; J15 = J9 − max(0, delta).\n",
        "- Re-run command: `uv run --with pandas,pyarrow,scikit-learn,xgboost scripts/run_empirical_formula_benchmark.py` (or `uv run --group backend-build python scripts/run_empirical_formula_benchmark.py`).\n",
    ]
    REPORT_PATH.write_text("".join(sections), encoding="utf-8")


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    folds = load_outer_folds()
    j9_df = attach_saved_outer_folds(pd.read_parquet(J9_DATASET_PATH), folds)
    delta_df = attach_saved_outer_folds(pd.read_parquet(DELTA_DATASET_PATH), folds)
    baselines = pd.read_csv(BASELINES_PATH)

    metric_rows, mono_rows = evaluate_formulas(j9_df, folds)
    metric_rows.extend(add_mean_baseline_rows(j9_df, folds, baselines))

    ridge_metric_rows, ridge_mono_rows = add_ridge_rows(j9_df, baselines)
    metric_rows.extend(ridge_metric_rows)
    mono_rows.extend(ridge_mono_rows)

    blend_metric_rows, blend_mono_rows, blend_info = evaluate_blend(j9_df, delta_df, folds)
    metric_rows.extend(blend_metric_rows)
    mono_rows.extend(blend_mono_rows)

    per_fold = pd.DataFrame(metric_rows)
    monotonicity = pd.DataFrame(mono_rows)
    per_fold = per_fold[
        ["label", "target", "fold", "mae", "rmse", "r2", "n_rows", "eval_scope", "ni_handling", "mo_value", "source"]
    ].sort_values(["target", "label", "eval_scope", "mo_value", "ni_handling", "fold"], na_position="first")
    monotonicity = monotonicity[
        ["label", "fold", "n_pairs", "violation_rate", "ni_handling", "mo_value", "source"]
    ].sort_values(["label", "mo_value", "ni_handling", "fold"], na_position="first")

    per_fold.to_csv(PER_FOLD_PATH, index=False)
    monotonicity.to_csv(MONOTONICITY_PATH, index=False)

    summary = aggregate_metrics(per_fold)
    mono_summary = aggregate_monotonicity(monotonicity)
    write_report(summary, mono_summary, blend_info)

    print(f"blend retrain agrees with blend_oof.csv: max |Δ| = {blend_info['max_fold_mae_delta']:.6f} HRC")
    print(f"Wrote {PER_FOLD_PATH}")
    print(f"Wrote {MONOTONICITY_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
