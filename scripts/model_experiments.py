# pyright: basic
"""
Iterative model experiments for J9 hardenability prediction.

Runs each candidate through 5-fold GroupKFold on base_heat_id and appends a row
to HISTORY.md. Best run is tracked across the whole script's invocation.

Run: uv run --with pandas,pyarrow,scikit-learn,xgboost,lightgbm scripts/model_experiments.py [--quick]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from modeling.config import DELTA_DATASET_PATH, J9_DATASET_PATH  # noqa: E402
from modeling.features import CORE_FEATURES, CORE_PLUS_FLAGS, FULL_FEATURES  # noqa: E402

HISTORY_PATH = PROJECT_ROOT / "HISTORY.md"
RANDOM_SEED = 42
N_FOLDS = 5

# Models we've already entered into HISTORY.md (in-memory dedupe by (run_id, target))
_seen_run_ids: set[tuple[str, str]] = set()


def scaled_pipe(model, features=FULL_FEATURES) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                list(features),
            )
        ]
    )
    return Pipeline([("preprocessor", pre), ("model", model)])


def imputed_pipe(model, features=FULL_FEATURES) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), list(features))]
    )
    return Pipeline([("preprocessor", pre), ("model", model)])


def cv_score(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    pipe_factory: Callable[[], Pipeline],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (means, stds) of MAE/RMSE/R² across N_FOLDS GroupKFold."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    maes, rmses, r2s = [], [], []
    X = df[features]
    y = df[target].to_numpy()
    groups = df["base_heat_id"].to_numpy()

    for tr, va in gkf.split(X, y, groups):
        pipe = pipe_factory()
        pipe.fit(X.iloc[tr], y[tr])
        pred = np.asarray(pipe.predict(X.iloc[va])).reshape(-1)
        maes.append(mean_absolute_error(y[va], pred))
        rmses.append(math.sqrt(mean_squared_error(y[va], pred)))
        r2s.append(r2_score(y[va], pred))
    means = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)), "r2": float(np.mean(r2s))}
    stds = {"mae": float(np.std(maes)), "rmse": float(np.std(rmses)), "r2": float(np.std(r2s))}
    return means, stds


def append_row(
    target: str,
    label: str,
    feature_set: str,
    means: dict[str, float],
    stds: dict[str, float],
    notes: str = "",
) -> None:
    """Append a markdown leaderboard row to HISTORY.md."""
    key = (label, target)
    if key in _seen_run_ids:
        return
    _seen_run_ids.add(key)

    n = sum(1 for k in _seen_run_ids if k[1] == target)
    row = (
        f"| {n} | `{label}` | {feature_set} | "
        f"{means['mae']:.4f} | {stds['mae']:.4f} | "
        f"{means['rmse']:.4f} | {means['r2']:.4f} | {notes} |\n"
    )
    section_header = f"## {target} leaderboard (live)"
    text = HISTORY_PATH.read_text(encoding="utf-8")
    if section_header not in text:
        text += (
            f"\n{section_header}\n\n"
            "| # | Model | Features | MAE_mean | MAE_std | RMSE_mean | R²_mean | Notes |\n"
            "|---|-------|----------|---------:|--------:|----------:|--------:|-------|\n"
        )
    HISTORY_PATH.write_text(text + row, encoding="utf-8")
    print(
        f"[{target}] {label:30s} feat={feature_set:11s} "
        f"MAE={means['mae']:.4f}±{stds['mae']:.4f} "
        f"RMSE={means['rmse']:.4f} R²={means['r2']:+.4f}  {notes}"
    )


def run(
    target: str,
    df: pd.DataFrame,
    label: str,
    features: list[str],
    feature_set_name: str,
    pipe_factory: Callable[[], Pipeline],
    notes: str = "",
) -> tuple[float, str]:
    means, stds = cv_score(df, target, features, pipe_factory)
    append_row(target, label, feature_set_name, means, stds, notes)
    return means["mae"], label


def section(title: str) -> None:
    print(f"\n=== {title} ===")
    HISTORY_PATH.write_text(
        HISTORY_PATH.read_text(encoding="utf-8") + f"\n<!-- {title} -->\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Skip slow models (GP, big MLPs).")
    ap.add_argument("--target", default="J9", choices=["J9", "delta", "both"])
    args = ap.parse_args()

    j9 = pd.read_parquet(J9_DATASET_PATH)
    delta = pd.read_parquet(DELTA_DATASET_PATH)

    print(f"J9: {j9.shape}, delta: {delta.shape}, FULL_FEATURES={len(FULL_FEATURES)}")

    targets: list[tuple[str, pd.DataFrame]] = []
    if args.target in ("J9", "both"):
        targets.append(("J9", j9))
    if args.target in ("delta", "both"):
        targets.append(("delta", delta))

    leaderboard: list[tuple[float, str, str]] = []  # (mae, label, target)

    for target, df in targets:
        section(f"{target}: linear family")
        run(target, df, "ridge_a1.0_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(Ridge(alpha=1.0)), "baseline reproduction")
        run(target, df, "ridge_a1.0_core", CORE_PLUS_FLAGS, "CORE+flag", lambda: scaled_pipe(Ridge(alpha=1.0), CORE_PLUS_FLAGS))
        run(target, df, "ridge_a1.0_core7", CORE_FEATURES, "CORE7", lambda: scaled_pipe(Ridge(alpha=1.0), CORE_FEATURES))
        run(target, df, "linreg_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(LinearRegression()))
        run(target, df, "lasso_a0.01_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(Lasso(alpha=0.01, max_iter=20000)))
        run(target, df, "elasticnet_a0.05_l1_0.5", FULL_FEATURES, "FULL", lambda: scaled_pipe(ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000)))
        run(target, df, "bayesian_ridge_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(BayesianRidge()))
        run(target, df, "huber_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(HuberRegressor(max_iter=500)))
        run(target, df, "pls_n4_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(PLSRegression(n_components=4, scale=False)), "baseline reproduction")
        run(target, df, "pls_n3_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(PLSRegression(n_components=3, scale=False)))
        run(target, df, "pls_n5_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(PLSRegression(n_components=5, scale=False)))
        run(target, df, "pls_n6_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(PLSRegression(n_components=6, scale=False)))

        # poly + ridge
        for deg in (2, 3):
            run(
                target,
                df,
                f"poly{deg}_ridge_a5_core",
                CORE_FEATURES,
                "CORE7",
                lambda d=deg: Pipeline(
                    [
                        ("imp", ColumnTransformer([("n", SimpleImputer(strategy="median"), CORE_FEATURES)])),
                        ("sc", StandardScaler()),
                        ("poly", PolynomialFeatures(degree=d, interaction_only=False, include_bias=False)),
                        ("model", Ridge(alpha=5.0)),
                    ]
                ),
                f"degree={deg} interactions+squared",
            )

        section(f"{target}: tree ensembles")
        run(
            target,
            df,
            "rf_300",
            FULL_FEATURES,
            "FULL",
            lambda: imputed_pipe(
                RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)
            ),
        )
        run(
            target,
            df,
            "rf_500_md8",
            FULL_FEATURES,
            "FULL",
            lambda: imputed_pipe(
                RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=4, random_state=RANDOM_SEED, n_jobs=-1)
            ),
        )
        run(
            target,
            df,
            "extratrees_500",
            FULL_FEATURES,
            "FULL",
            lambda: imputed_pipe(
                ExtraTreesRegressor(n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)
            ),
        )
        run(
            target,
            df,
            "gbr_default",
            FULL_FEATURES,
            "FULL",
            lambda: imputed_pipe(
                GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED)
            ),
        )
        run(
            target,
            df,
            "hgbr_baseline",
            FULL_FEATURES,
            "FULL",
            lambda: imputed_pipe(
                HistGradientBoostingRegressor(
                    learning_rate=0.05, max_depth=3, max_leaf_nodes=15, min_samples_leaf=10, max_iter=200, random_state=RANDOM_SEED
                )
            ),
            "baseline reproduction",
        )
        run(
            target,
            df,
            "hgbr_lr0.03_iter500",
            FULL_FEATURES,
            "FULL",
            lambda: imputed_pipe(
                HistGradientBoostingRegressor(
                    learning_rate=0.03, max_depth=4, max_leaf_nodes=31, min_samples_leaf=8, max_iter=500, random_state=RANDOM_SEED, l2_regularization=0.1
                )
            ),
        )
        run(
            target,
            df,
            "hgbr_lr0.05_md5",
            FULL_FEATURES,
            "FULL",
            lambda: imputed_pipe(
                HistGradientBoostingRegressor(
                    learning_rate=0.05, max_depth=5, max_leaf_nodes=31, min_samples_leaf=5, max_iter=400, random_state=RANDOM_SEED, l2_regularization=1.0
                )
            ),
        )

        # XGBoost / LightGBM if available
        try:
            from xgboost import XGBRegressor

            for params in [
                dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1),
                dict(n_estimators=800, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=2.0, random_state=RANDOM_SEED, n_jobs=-1),
                dict(n_estimators=1000, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1),
            ]:
                tag = f"xgb_n{params['n_estimators']}_lr{params['learning_rate']}_md{params['max_depth']}"
                run(target, df, tag, FULL_FEATURES, "FULL", lambda p=params: imputed_pipe(XGBRegressor(**p)))
        except ImportError:
            print("xgboost not installed — skipping")

        try:
            from lightgbm import LGBMRegressor

            for params in [
                dict(n_estimators=500, learning_rate=0.05, num_leaves=15, max_depth=-1, min_child_samples=8, reg_alpha=0.1, reg_lambda=0.1, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1),
                dict(n_estimators=800, learning_rate=0.03, num_leaves=31, max_depth=6, min_child_samples=10, reg_alpha=0.0, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1),
            ]:
                tag = f"lgbm_n{params['n_estimators']}_lr{params['learning_rate']}_lv{params['num_leaves']}"
                run(target, df, tag, FULL_FEATURES, "FULL", lambda p=params: imputed_pipe(LGBMRegressor(**p)))
        except ImportError:
            print("lightgbm not installed — skipping")

        section(f"{target}: kernel & instance models")
        run(target, df, "kridge_rbf_a1_g0.1", FULL_FEATURES, "FULL", lambda: scaled_pipe(KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)))
        run(target, df, "kridge_rbf_a0.5_g0.05", FULL_FEATURES, "FULL", lambda: scaled_pipe(KernelRidge(alpha=0.5, kernel="rbf", gamma=0.05)))
        run(target, df, "kridge_poly2_a1", FULL_FEATURES, "FULL", lambda: scaled_pipe(KernelRidge(alpha=1.0, kernel="polynomial", degree=2)))
        run(target, df, "svr_rbf_C1_g0.1", FULL_FEATURES, "FULL", lambda: scaled_pipe(SVR(kernel="rbf", C=1.0, gamma=0.1, epsilon=0.2)))
        run(target, df, "svr_rbf_C5_g0.05", FULL_FEATURES, "FULL", lambda: scaled_pipe(SVR(kernel="rbf", C=5.0, gamma=0.05, epsilon=0.1)))
        run(target, df, "svr_linear_C1", FULL_FEATURES, "FULL", lambda: scaled_pipe(SVR(kernel="linear", C=1.0, epsilon=0.2)))
        run(target, df, "knn_k7", FULL_FEATURES, "FULL", lambda: scaled_pipe(KNeighborsRegressor(n_neighbors=7, weights="distance")))
        run(target, df, "knn_k15", FULL_FEATURES, "FULL", lambda: scaled_pipe(KNeighborsRegressor(n_neighbors=15, weights="distance")))

        if not args.quick:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 1e2))
            run(
                target,
                df,
                "gp_rbf_white",
                FULL_FEATURES,
                "FULL",
                lambda: scaled_pipe(
                    GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-8, n_restarts_optimizer=2, random_state=RANDOM_SEED)
                ),
                "may be slow",
            )

        section(f"{target}: MLP")
        run(
            target,
            df,
            "mlp_64x32_relu",
            FULL_FEATURES,
            "FULL",
            lambda: scaled_pipe(
                MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=2000, early_stopping=True, random_state=RANDOM_SEED)
            ),
        )
        run(
            target,
            df,
            "mlp_128x64_tanh",
            FULL_FEATURES,
            "FULL",
            lambda: scaled_pipe(
                MLPRegressor(hidden_layer_sizes=(128, 64), activation="tanh", alpha=1e-2, learning_rate_init=5e-4, max_iter=3000, early_stopping=True, random_state=RANDOM_SEED)
            ),
        )
        if not args.quick:
            run(
                target,
                df,
                "mlp_256x128x64_relu",
                FULL_FEATURES,
                "FULL",
                lambda: scaled_pipe(
                    MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=1e-2, learning_rate_init=5e-4, max_iter=3000, early_stopping=True, random_state=RANDOM_SEED)
                ),
            )

        section(f"{target}: stacking")
        # stacking the three best families
        try:
            from xgboost import XGBRegressor

            xgb = XGBRegressor(n_estimators=600, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=RANDOM_SEED, n_jobs=-1)
            xgb_estimator = ("xgb", imputed_pipe(xgb))
        except ImportError:
            xgb_estimator = (
                "hgbr",
                imputed_pipe(HistGradientBoostingRegressor(learning_rate=0.05, max_depth=3, max_leaf_nodes=15, min_samples_leaf=10, max_iter=200, random_state=RANDOM_SEED)),
            )

        estimators = [
            ("pls", scaled_pipe(PLSRegression(n_components=4, scale=False))),
            ("ridge", scaled_pipe(Ridge(alpha=1.0))),
            xgb_estimator,
        ]
        run(
            target,
            df,
            "stack_pls_ridge_xgb",
            FULL_FEATURES,
            "FULL",
            lambda: StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1),
            "stacking three best families",
        )

    print("\nDone. See HISTORY.md")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Elapsed: {time.time() - t0:.1f}s")
