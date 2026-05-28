# pyright: basic
"""
Round 5: refine around the new leader xgb_v2_ss0.55_cs0.5 (MAE 1.7263 on J9).

  - Even tighter (ss, cs) sweep
  - lr/n_estimators sweep at ss=0.55, cs=0.5
  - multi-seed bag of new best
  - MAE objective ('reg:absoluteerror')
  - delta multi-seed bag
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
    run,
    section,
)


def bag_cv(df, target, features, build_pipe, seeds, label, notes=""):
    X = df[features]
    y = df[target].to_numpy()
    groups = df["base_heat_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    maes, rmses, r2s = [], [], []
    for tr, va in gkf.split(X, y, groups):
        preds = np.zeros(len(va))
        for s in seeds:
            pipe = build_pipe(s)
            pipe.fit(X.iloc[tr], y[tr])
            preds += np.asarray(pipe.predict(X.iloc[va])).reshape(-1)
        preds /= len(seeds)
        maes.append(mean_absolute_error(y[va], preds))
        rmses.append(math.sqrt(mean_squared_error(y[va], preds)))
        r2s.append(r2_score(y[va], preds))
    means = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)), "r2": float(np.mean(r2s))}
    stds = {"mae": float(np.std(maes)), "rmse": float(np.std(rmses)), "r2": float(np.std(r2s))}
    append_row(target, label, "FULL", means, stds, notes)


def main() -> None:
    j9 = pd.read_parquet(J9_DATASET_PATH)
    delta = pd.read_parquet(DELTA_DATASET_PATH)

    section("J9: very tight (ss, cs) sweep")
    try:
        from xgboost import XGBRegressor

        for ss in (0.45, 0.5, 0.55, 0.6):
            for cs in (0.45, 0.5, 0.55):
                p = dict(
                    n_estimators=800, max_depth=3, learning_rate=0.01,
                    subsample=ss, colsample_bytree=cs, reg_lambda=2.0,
                    random_state=RANDOM_SEED, n_jobs=-1,
                )
                run(
                    "J9", j9,
                    f"xgb_v3_ss{ss}_cs{cs}",
                    FULL_FEATURES, "FULL",
                    lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
                )

        section("J9: lr/n_est sweep at (0.55, 0.5)")
        for n_est, lr in [
            (500, 0.01), (1000, 0.01), (1200, 0.01),
            (800, 0.005), (1500, 0.005), (2000, 0.005),
            (800, 0.015), (800, 0.02),
        ]:
            p = dict(
                n_estimators=n_est, max_depth=3, learning_rate=lr,
                subsample=0.55, colsample_bytree=0.5, reg_lambda=2.0,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
            run(
                "J9", j9,
                f"xgb_v3_n{n_est}_lr{lr}",
                FULL_FEATURES, "FULL",
                lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
            )

        section("J9: reg:absoluteerror objective")
        for ss in (0.55, 0.6):
            for cs in (0.5, 0.6):
                p = dict(
                    n_estimators=800, max_depth=3, learning_rate=0.01,
                    subsample=ss, colsample_bytree=cs, reg_lambda=2.0,
                    objective="reg:absoluteerror",
                    random_state=RANDOM_SEED, n_jobs=-1,
                )
                run(
                    "J9", j9,
                    f"xgb_mae_ss{ss}_cs{cs}",
                    FULL_FEATURES, "FULL",
                    lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
                )

        section("J9: bag of new leader (20 seeds)")
        seeds20 = list(range(20))
        bag_cv(
            j9, "J9", FULL_FEATURES,
            lambda s: imputed_pipe(
                XGBRegressor(
                    n_estimators=800, max_depth=3, learning_rate=0.01,
                    subsample=0.55, colsample_bytree=0.5, reg_lambda=2.0,
                    random_state=s, n_jobs=-1,
                )
            ),
            seeds20,
            "xgb_bag20_v3",
            notes="20-seed bag at ss=0.55, cs=0.5",
        )
        # bag of MAE-objective leader candidates
        bag_cv(
            j9, "J9", FULL_FEATURES,
            lambda s: imputed_pipe(
                XGBRegressor(
                    n_estimators=800, max_depth=3, learning_rate=0.01,
                    subsample=0.55, colsample_bytree=0.5, reg_lambda=2.0,
                    objective="reg:absoluteerror",
                    random_state=s, n_jobs=-1,
                )
            ),
            seeds20,
            "xgb_bag20_mae",
            notes="20-seed bag, MAE objective",
        )

        section("delta: bag of leader")
        bag_cv(
            delta, "delta", FULL_FEATURES,
            lambda s: imputed_pipe(
                XGBRegressor(
                    n_estimators=800, max_depth=3, learning_rate=0.01,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                    random_state=s, n_jobs=-1,
                )
            ),
            seeds20,
            "xgb_bag20_delta",
            notes="20-seed bag of delta leader",
        )
    except ImportError:
        print("xgboost missing — skipping")

    print("\nDone (round 5). See HISTORY.md")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Elapsed: {time.time() - t0:.1f}s")
