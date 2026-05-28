# pyright: basic
"""
Round 7: refine the winning blend xgb_v3 (n=1500, lr=0.005, ss=0.55, cs=0.5) + pls_n3.

  - Finer weight grid around 0.65-0.75
  - Bag (multi-seed XGB) inside the blend
  - Try alternative linear partner: Ridge (best alpha) and Lasso
  - Apply blend to delta target as well
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from modeling.config import DELTA_DATASET_PATH, J9_DATASET_PATH  # noqa: E402
from modeling.features import FULL_FEATURES  # noqa: E402
from model_experiments import (  # noqa: E402
    RANDOM_SEED,
    append_row,
    imputed_pipe,
    scaled_pipe,
    section,
)


def cv_blend(df, target, build_xgb, build_linear, weights, label_prefix, notes_prefix):
    X = df[FULL_FEATURES]
    y = df[target].to_numpy()
    groups = df["base_heat_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    fold_xgb_pred = []
    fold_lin_pred = []
    fold_y = []
    for tr, va in gkf.split(X, y, groups):
        xgb_pipe = build_xgb()
        lin_pipe = build_linear()
        xgb_pipe.fit(X.iloc[tr], y[tr])
        lin_pipe.fit(X.iloc[tr], y[tr])
        fold_xgb_pred.append(np.asarray(xgb_pipe.predict(X.iloc[va])).reshape(-1))
        fold_lin_pred.append(np.asarray(lin_pipe.predict(X.iloc[va])).reshape(-1))
        fold_y.append(y[va])

    best = (1e9, None, None, None)
    for w in weights:
        maes, rmses, r2s = [], [], []
        for x_pred, l_pred, y_va in zip(fold_xgb_pred, fold_lin_pred, fold_y):
            pred = w * x_pred + (1 - w) * l_pred
            maes.append(mean_absolute_error(y_va, pred))
            rmses.append(math.sqrt(mean_squared_error(y_va, pred)))
            r2s.append(r2_score(y_va, pred))
        means = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)), "r2": float(np.mean(r2s))}
        stds = {"mae": float(np.std(maes)), "rmse": float(np.std(rmses)), "r2": float(np.std(r2s))}
        label = f"{label_prefix}_w{w:.2f}"
        notes = f"{notes_prefix}; w_xgb={w:.2f}"
        append_row(target, label, "FULL", means, stds, notes)
        if means["mae"] < best[0]:
            best = (means["mae"], label, means, stds)
    return best


def cv_blend_bag(df, target, build_xgb_seed, seeds, build_linear, weights, label_prefix, notes_prefix):
    """Like cv_blend but XGB is replaced by mean over multiple seeds."""
    X = df[FULL_FEATURES]
    y = df[target].to_numpy()
    groups = df["base_heat_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    fold_xgb_pred = []
    fold_lin_pred = []
    fold_y = []
    for tr, va in gkf.split(X, y, groups):
        bag = np.zeros(len(va))
        for s in seeds:
            pipe = build_xgb_seed(s)
            pipe.fit(X.iloc[tr], y[tr])
            bag += np.asarray(pipe.predict(X.iloc[va])).reshape(-1)
        bag /= len(seeds)
        lin_pipe = build_linear()
        lin_pipe.fit(X.iloc[tr], y[tr])
        fold_xgb_pred.append(bag)
        fold_lin_pred.append(np.asarray(lin_pipe.predict(X.iloc[va])).reshape(-1))
        fold_y.append(y[va])

    best = (1e9, None, None, None)
    for w in weights:
        maes, rmses, r2s = [], [], []
        for x_pred, l_pred, y_va in zip(fold_xgb_pred, fold_lin_pred, fold_y):
            pred = w * x_pred + (1 - w) * l_pred
            maes.append(mean_absolute_error(y_va, pred))
            rmses.append(math.sqrt(mean_squared_error(y_va, pred)))
            r2s.append(r2_score(y_va, pred))
        means = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)), "r2": float(np.mean(r2s))}
        stds = {"mae": float(np.std(maes)), "rmse": float(np.std(rmses)), "r2": float(np.std(r2s))}
        label = f"{label_prefix}_w{w:.2f}"
        append_row(target, label, "FULL", means, stds, f"{notes_prefix}; w_xgb={w:.2f}")
        if means["mae"] < best[0]:
            best = (means["mae"], label, means, stds)
    return best


def main() -> None:
    j9 = pd.read_parquet(J9_DATASET_PATH)
    delta = pd.read_parquet(DELTA_DATASET_PATH)

    try:
        from xgboost import XGBRegressor

        XGB_LEADER_PARAMS = dict(
            n_estimators=1500, max_depth=3, learning_rate=0.005,
            subsample=0.55, colsample_bytree=0.5, reg_lambda=2.0,
            random_state=RANDOM_SEED, n_jobs=-1,
        )

        section("J9: fine blend grid xgb_leader + pls3")
        best = cv_blend(
            j9, "J9",
            lambda: imputed_pipe(XGBRegressor(**XGB_LEADER_PARAMS)),
            lambda: scaled_pipe(PLSRegression(n_components=3, scale=False)),
            np.linspace(0.55, 0.85, 13).tolist(),
            "blendF_xgb_pls3",
            "fine grid",
        )
        print("Best fine xgb+pls3 blend:", best[1], best[2])

        section("J9: blend xgb_leader + ridge")
        best = cv_blend(
            j9, "J9",
            lambda: imputed_pipe(XGBRegressor(**XGB_LEADER_PARAMS)),
            lambda: scaled_pipe(Ridge(alpha=1.0)),
            np.linspace(0.55, 0.85, 7).tolist(),
            "blendF_xgb_ridge",
            "ridge alpha=1.0",
        )
        print("Best xgb+ridge blend:", best[1], best[2])

        section("J9: blend xgb_leader + bayesianridge")
        best = cv_blend(
            j9, "J9",
            lambda: imputed_pipe(XGBRegressor(**XGB_LEADER_PARAMS)),
            lambda: scaled_pipe(BayesianRidge()),
            np.linspace(0.55, 0.85, 7).tolist(),
            "blendF_xgb_bayes",
            "bayesian ridge",
        )
        print("Best xgb+bayes blend:", best[1], best[2])

        section("J9: blend (10-seed bagged xgb) + pls3")
        seeds10 = list(range(10))
        best = cv_blend_bag(
            j9, "J9",
            lambda s: imputed_pipe(XGBRegressor(**{**XGB_LEADER_PARAMS, "random_state": s})),
            seeds10,
            lambda: scaled_pipe(PLSRegression(n_components=3, scale=False)),
            np.linspace(0.6, 0.8, 5).tolist(),
            "blendBag10_xgb_pls3",
            "10-seed XGB bag + PLS3",
        )
        print("Best bagged blend:", best[1], best[2])

        section("delta: blend leader + pls5")
        XGB_DELTA = dict(
            n_estimators=800, max_depth=3, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        best = cv_blend(
            delta, "delta",
            lambda: imputed_pipe(XGBRegressor(**XGB_DELTA)),
            lambda: scaled_pipe(PLSRegression(n_components=5, scale=False)),
            np.linspace(0.2, 0.8, 7).tolist(),
            "blendF_delta_xgb_pls5",
            "delta blend",
        )
        print("Best delta blend xgb+pls5:", best[1], best[2])

        best = cv_blend(
            delta, "delta",
            lambda: imputed_pipe(XGBRegressor(**XGB_DELTA)),
            lambda: scaled_pipe(BayesianRidge()),
            np.linspace(0.2, 0.8, 7).tolist(),
            "blendF_delta_xgb_bayes",
            "delta blend",
        )
        print("Best delta blend xgb+bayes:", best[1], best[2])
    except ImportError:
        print("xgboost missing")

    print("\nDone (round 7).")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Elapsed: {time.time() - t0:.1f}s")
