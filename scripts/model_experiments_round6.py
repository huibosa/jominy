# pyright: basic
"""
Round 6: LightGBM with proper params (analogue of XGBoost leader),
plus stacked ensemble of best PLS + best XGB + best Ridge with passthrough.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from modeling.config import J9_DATASET_PATH, DELTA_DATASET_PATH  # noqa: E402
from modeling.features import FULL_FEATURES  # noqa: E402
from model_experiments import (  # noqa: E402
    RANDOM_SEED,
    append_row,
    imputed_pipe,
    run,
    scaled_pipe,
    section,
)


def main() -> None:
    j9 = pd.read_parquet(J9_DATASET_PATH)
    delta = pd.read_parquet(DELTA_DATASET_PATH)

    section("J9: LightGBM analogue of leader")
    try:
        from lightgbm import LGBMRegressor

        for n_est, lr, num_leaves in [
            (800, 0.01, 7),
            (800, 0.01, 15),
            (1500, 0.005, 7),
            (1500, 0.005, 15),
            (1500, 0.005, 31),
            (2000, 0.005, 7),
        ]:
            p = dict(
                n_estimators=n_est, learning_rate=lr, num_leaves=num_leaves,
                max_depth=-1, min_child_samples=8,
                subsample=0.55, subsample_freq=1, colsample_bytree=0.5,
                reg_lambda=2.0, reg_alpha=0.0,
                random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
            )
            run(
                "J9", j9,
                f"lgbm_n{n_est}_lr{lr}_lv{num_leaves}",
                FULL_FEATURES, "FULL",
                lambda pp=p: imputed_pipe(LGBMRegressor(**pp)),
            )
    except ImportError:
        print("lightgbm missing — skipping")

    section("J9: cross-family stacking with imputed front")
    try:
        from xgboost import XGBRegressor

        # Build a stacking with three diverse base learners and a Ridge meta-learner.
        # Use a Pipeline-wrapped final estimator with an imputer so passthrough
        # original features (which have NaN) don't break the meta-fit.
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        meta_pipe = Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )

        xgb_leader = lambda: imputed_pipe(  # noqa: E731
            XGBRegressor(
                n_estimators=1500, max_depth=3, learning_rate=0.005,
                subsample=0.55, colsample_bytree=0.5, reg_lambda=2.0,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
        )
        estimators = [
            ("pls3", scaled_pipe(PLSRegression(n_components=3, scale=False))),
            ("bayes", scaled_pipe(BayesianRidge())),
            ("xgb", xgb_leader()),
        ]
        run(
            "J9", j9,
            "stack_v2_pls3_bayes_xgb",
            FULL_FEATURES, "FULL",
            lambda: StackingRegressor(estimators=estimators, final_estimator=meta_pipe, cv=5, n_jobs=1, passthrough=True),
            "passthrough+imputed Ridge meta",
        )

        # Greedy weighted average — manual: optimize convex weights on training fold predictions
        section("J9: convex-weighted blend of pls3 + xgb")
        gkf = GroupKFold(n_splits=5)
        X = j9[FULL_FEATURES]
        y = j9["J9"].to_numpy()
        groups = j9["base_heat_id"].to_numpy()

        # Sweep over grid
        best_label = None
        best_means: dict[str, float] = {}
        best_stds: dict[str, float] = {}
        for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            maes, rmses, r2s = [], [], []
            for tr, va in gkf.split(X, y, groups):
                pls = scaled_pipe(PLSRegression(n_components=3, scale=False))
                xgb_pipe = xgb_leader()
                pls.fit(X.iloc[tr], y[tr])
                xgb_pipe.fit(X.iloc[tr], y[tr])
                p_pls = np.asarray(pls.predict(X.iloc[va])).reshape(-1)
                p_xgb = np.asarray(xgb_pipe.predict(X.iloc[va])).reshape(-1)
                pred = w * p_xgb + (1 - w) * p_pls
                maes.append(mean_absolute_error(y[va], pred))
                rmses.append(math.sqrt(mean_squared_error(y[va], pred)))
                r2s.append(r2_score(y[va], pred))
            means = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)), "r2": float(np.mean(r2s))}
            stds = {"mae": float(np.std(maes)), "rmse": float(np.std(rmses)), "r2": float(np.std(r2s))}
            label = f"blend_xgb{w}_pls{1 - w}"
            append_row("J9", label, "FULL", means, stds, "convex blend")
            if best_label is None or means["mae"] < best_means["mae"]:
                best_label = label
                best_means = means
                best_stds = stds
        print(f"Best blend: {best_label} -> {best_means}")

        # Three-way blend with bayesian ridge
        section("J9: three-way blend xgb + pls3 + bayes")
        best_three: tuple[float, str] | None = None
        for w_xgb in (0.4, 0.5, 0.6, 0.7):
            for w_pls in (0.1, 0.2, 0.3):
                w_bayes = 1.0 - w_xgb - w_pls
                if w_bayes < 0.0 or w_bayes > 0.6:
                    continue
                maes = []
                rmses = []
                r2s = []
                for tr, va in gkf.split(X, y, groups):
                    pls = scaled_pipe(PLSRegression(n_components=3, scale=False))
                    bayes = scaled_pipe(BayesianRidge())
                    xgb_pipe = xgb_leader()
                    pls.fit(X.iloc[tr], y[tr])
                    bayes.fit(X.iloc[tr], y[tr])
                    xgb_pipe.fit(X.iloc[tr], y[tr])
                    p_pls = np.asarray(pls.predict(X.iloc[va])).reshape(-1)
                    p_bayes = np.asarray(bayes.predict(X.iloc[va])).reshape(-1)
                    p_xgb = np.asarray(xgb_pipe.predict(X.iloc[va])).reshape(-1)
                    pred = w_xgb * p_xgb + w_pls * p_pls + w_bayes * p_bayes
                    maes.append(mean_absolute_error(y[va], pred))
                    rmses.append(math.sqrt(mean_squared_error(y[va], pred)))
                    r2s.append(r2_score(y[va], pred))
                means = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)), "r2": float(np.mean(r2s))}
                stds = {"mae": float(np.std(maes)), "rmse": float(np.std(rmses)), "r2": float(np.std(r2s))}
                label = f"blend3_xgb{w_xgb}_pls{w_pls}_bayes{w_bayes:.1f}"
                append_row("J9", label, "FULL", means, stds, "3-way blend")
                if best_three is None or means["mae"] < best_three[0]:
                    best_three = (means["mae"], label)
        print(f"Best 3-way blend: {best_three}")
    except ImportError:
        pass

    section("delta: LightGBM and final blends")
    try:
        from lightgbm import LGBMRegressor

        for n_est, lr, num_leaves in [(800, 0.01, 15), (1500, 0.005, 15), (1500, 0.005, 31)]:
            p = dict(
                n_estimators=n_est, learning_rate=lr, num_leaves=num_leaves,
                max_depth=-1, min_child_samples=8,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_lambda=2.0, reg_alpha=0.0,
                random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
            )
            run(
                "delta", delta,
                f"lgbm_n{n_est}_lr{lr}_lv{num_leaves}",
                FULL_FEATURES, "FULL",
                lambda pp=p: imputed_pipe(LGBMRegressor(**pp)),
            )
    except ImportError:
        pass

    print("\nDone (round 6). See HISTORY.md")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Elapsed: {time.time() - t0:.1f}s")
