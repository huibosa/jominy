# pyright: basic
"""
Round 4: tune around new winner xgb_n800_lr0.01_md3 with ss0.6/cs0.6 (MAE 1.7386).

Plan:
  - Tighter sweep around (ss, cs) in [0.5..0.8]
  - max_depth 2 + 3 + min_child_weight sweep
  - Multi-seed bagging of best XGB
  - Delta sweep (still missing from prior runs)
  - Fixed stacking with imputation in front of passthrough features
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from modeling.config import DELTA_DATASET_PATH, J9_DATASET_PATH  # noqa: E402
from modeling.features import FULL_FEATURES  # noqa: E402
from model_experiments import (  # noqa: E402
    RANDOM_SEED,
    cv_score,
    imputed_pipe,
    run,
    scaled_pipe,
    section,
    append_row,
)


def main() -> None:
    j9 = pd.read_parquet(J9_DATASET_PATH)
    delta = pd.read_parquet(DELTA_DATASET_PATH)

    section("J9: XGBoost subsample/colsample fine sweep")
    try:
        from xgboost import XGBRegressor

        for ss in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
            for cs in (0.5, 0.6, 0.7, 0.8):
                p = dict(
                    n_estimators=800, max_depth=3, learning_rate=0.01,
                    subsample=ss, colsample_bytree=cs, reg_lambda=2.0,
                    random_state=RANDOM_SEED, n_jobs=-1,
                )
                run(
                    "J9", j9,
                    f"xgb_v2_ss{ss}_cs{cs}",
                    FULL_FEATURES, "FULL",
                    lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
                )

        # md=2 with the tight sweep
        for ss in (0.6, 0.7):
            for cs in (0.6, 0.7):
                p = dict(
                    n_estimators=800, max_depth=2, learning_rate=0.01,
                    subsample=ss, colsample_bytree=cs, reg_lambda=2.0,
                    random_state=RANDOM_SEED, n_jobs=-1,
                )
                run(
                    "J9", j9,
                    f"xgb_md2_ss{ss}_cs{cs}",
                    FULL_FEATURES, "FULL",
                    lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
                )

        # min_child_weight effect on the leader
        for mcw in (1, 3, 5, 10):
            p = dict(
                n_estimators=800, max_depth=3, learning_rate=0.01,
                subsample=0.6, colsample_bytree=0.6, reg_lambda=2.0,
                min_child_weight=mcw, random_state=RANDOM_SEED, n_jobs=-1,
            )
            run(
                "J9", j9,
                f"xgb_leader_mcw{mcw}",
                FULL_FEATURES, "FULL",
                lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
            )
    except ImportError:
        print("xgboost missing — skipping")

    section("J9: multi-seed bag of leader XGBoost")
    # Bag the leader across 10 seeds and average. Implement manually so we measure
    # CV score of the bagged predictor.
    try:
        from xgboost import XGBRegressor
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import math

        seeds = [42, 7, 13, 21, 99, 123, 314, 2026, 5, 11]
        X = j9[FULL_FEATURES]
        y = j9["J9"].to_numpy()
        groups = j9["base_heat_id"].to_numpy()
        gkf = GroupKFold(n_splits=5)
        maes, rmses, r2s = [], [], []
        for tr, va in gkf.split(X, y, groups):
            preds = np.zeros(len(va))
            for s in seeds:
                pipe = imputed_pipe(
                    XGBRegressor(
                        n_estimators=800, max_depth=3, learning_rate=0.01,
                        subsample=0.65, colsample_bytree=0.65, reg_lambda=2.0,
                        random_state=s, n_jobs=-1,
                    )
                )
                pipe.fit(X.iloc[tr], y[tr])
                preds += np.asarray(pipe.predict(X.iloc[va])).reshape(-1)
            preds /= len(seeds)
            maes.append(mean_absolute_error(y[va], preds))
            rmses.append(math.sqrt(mean_squared_error(y[va], preds)))
            r2s.append(r2_score(y[va], preds))
        means = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)), "r2": float(np.mean(r2s))}
        stds = {"mae": float(np.std(maes)), "rmse": float(np.std(rmses)), "r2": float(np.std(r2s))}
        append_row("J9", "xgb_bag_10seeds", "FULL", means, stds, "10-seed bag of leader")
    except ImportError:
        print("xgboost missing — skipping bag")

    section("J9: stacking — imputed front")
    try:
        from xgboost import XGBRegressor

        xgb_leader = imputed_pipe(
            XGBRegressor(
                n_estimators=800, max_depth=3, learning_rate=0.01,
                subsample=0.6, colsample_bytree=0.6, reg_lambda=2.0,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
        )
        # final estimator that handles NaN — HGBR
        estimators = [
            ("pls3", scaled_pipe(PLSRegression(n_components=3, scale=False))),
            ("ridge", scaled_pipe(Ridge(alpha=1.0))),
            ("xgb", xgb_leader),
        ]
        run(
            "J9", j9,
            "stack_pls3_ridge_xgbLeader",
            FULL_FEATURES, "FULL",
            lambda: StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1),
        )
        # HGBR as final
        run(
            "J9", j9,
            "stack_pls3_xgb_finalHGBR",
            FULL_FEATURES, "FULL",
            lambda: StackingRegressor(
                estimators=[
                    ("pls3", scaled_pipe(PLSRegression(n_components=3, scale=False))),
                    ("xgb", xgb_leader),
                ],
                final_estimator=HistGradientBoostingRegressor(
                    learning_rate=0.05, max_depth=3, max_iter=100, random_state=RANDOM_SEED
                ),
                cv=5,
                n_jobs=1,
            ),
        )
    except ImportError:
        pass

    section("delta: round 4")
    for n in (2, 3, 4, 5):
        run(
            "delta", delta,
            f"pls_n{n}_full",
            FULL_FEATURES, "FULL",
            lambda nn=n: scaled_pipe(PLSRegression(n_components=nn, scale=False)),
        )
    run("delta", delta, "ridge_a1.0_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(Ridge(alpha=1.0)))
    run("delta", delta, "bayesian_ridge_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(BayesianRidge()))
    run(
        "delta", delta,
        "gbr_n200_md3_lr0.05",
        FULL_FEATURES, "FULL",
        lambda: imputed_pipe(GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED)),
    )
    try:
        from xgboost import XGBRegressor
        for n_est, lr, md, ss, cs in [
            (800, 0.01, 3, 0.6, 0.6),
            (800, 0.01, 3, 0.8, 0.8),
            (500, 0.02, 3, 0.8, 0.8),
            (1200, 0.01, 3, 0.6, 0.6),
            (2000, 0.005, 3, 0.6, 0.6),
        ]:
            p = dict(
                n_estimators=n_est, max_depth=md, learning_rate=lr,
                subsample=ss, colsample_bytree=cs, reg_lambda=2.0,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
            run(
                "delta", delta,
                f"xgb_n{n_est}_lr{lr}_md{md}_ss{ss}_cs{cs}",
                FULL_FEATURES, "FULL",
                lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
            )
    except ImportError:
        pass

    print("\nDone (round 4). See HISTORY.md")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Elapsed: {time.time() - t0:.1f}s")
