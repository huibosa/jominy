# pyright: basic
"""
Round 3: continue from where round 2 was killed.

Round 2 new leader: xgb_n800_lr0.01_md3  MAE=1.7445.

This round (kept memory-light):
  - XGBoost: even slower lr (0.005, 0.008), n_estimators up to 2000
  - Stacking with new leader (pls3 + bayes + xgb_n800_lr0.01_md3)
  - Delta target sweep
  - Different XGBoost regularization
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import BayesianRidge, Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from modeling.config import DELTA_DATASET_PATH, J9_DATASET_PATH  # noqa: E402
from modeling.features import FULL_FEATURES  # noqa: E402
from model_experiments import (  # noqa: E402
    RANDOM_SEED,
    imputed_pipe,
    run,
    scaled_pipe,
    section,
)


def main() -> None:
    j9 = pd.read_parquet(J9_DATASET_PATH)
    delta = pd.read_parquet(DELTA_DATASET_PATH)

    section("J9: XGBoost slower learning rates")
    try:
        from xgboost import XGBRegressor

        # n=1200 finishes (lr 0.02, 0.03 not yet recorded)
        for n_est, lr, md in [
            (1200, 0.02, 3),
            (1200, 0.02, 4),
            (1500, 0.01, 3),
            (1500, 0.01, 4),
            (2000, 0.005, 3),
            (2000, 0.005, 4),
            (2000, 0.008, 3),
            (3000, 0.005, 3),
        ]:
            p = dict(
                n_estimators=n_est, max_depth=md, learning_rate=lr,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
            run(
                "J9", j9,
                f"xgb_n{n_est}_lr{lr}_md{md}",
                FULL_FEATURES, "FULL",
                lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
            )

        # different regularization styles for the leader
        for reg_lambda, reg_alpha in [(0.5, 0.5), (5.0, 0.0), (5.0, 0.5), (10.0, 0.0)]:
            p = dict(
                n_estimators=800, max_depth=3, learning_rate=0.01,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
            run(
                "J9", j9,
                f"xgb_n800_lr0.01_md3_l{reg_lambda}_a{reg_alpha}",
                FULL_FEATURES, "FULL",
                lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
            )

        # different subsample/colsample
        for ss, cs in [(0.6, 0.6), (0.7, 0.7), (1.0, 0.6), (0.6, 1.0), (1.0, 1.0)]:
            p = dict(
                n_estimators=800, max_depth=3, learning_rate=0.01,
                subsample=ss, colsample_bytree=cs, reg_lambda=2.0,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
            run(
                "J9", j9,
                f"xgb_n800_lr0.01_md3_ss{ss}_cs{cs}",
                FULL_FEATURES, "FULL",
                lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
            )
    except ImportError:
        print("xgboost missing — skipping")

    section("J9: stacking with new leader")
    try:
        from xgboost import XGBRegressor

        xgb_leader = lambda: imputed_pipe(  # noqa: E731
            XGBRegressor(
                n_estimators=800, max_depth=3, learning_rate=0.01,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
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
            "stack_pls3_bayes_xgbLeader",
            FULL_FEATURES, "FULL",
            lambda: StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1),
            "stack new leader",
        )
        # passthrough=False vs True
        estimators2 = [
            ("pls3", scaled_pipe(PLSRegression(n_components=3, scale=False))),
            ("xgb", xgb_leader()),
        ]
        run(
            "J9", j9,
            "stack_pls3_xgbLeader_pt",
            FULL_FEATURES, "FULL",
            lambda: StackingRegressor(estimators=estimators2, final_estimator=BayesianRidge(), cv=5, n_jobs=1, passthrough=True),
            "stack with passthrough=True",
        )
    except ImportError:
        pass

    section("delta: round 3")
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
        for n_est, lr, md in [(800, 0.01, 3), (500, 0.02, 3), (1200, 0.01, 3)]:
            p = dict(
                n_estimators=n_est, max_depth=md, learning_rate=lr,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
            run(
                "delta", delta,
                f"xgb_n{n_est}_lr{lr}_md{md}",
                FULL_FEATURES, "FULL",
                lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
            )
    except ImportError:
        pass

    print("\nDone (round 3). See HISTORY.md")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Elapsed: {time.time() - t0:.1f}s")
