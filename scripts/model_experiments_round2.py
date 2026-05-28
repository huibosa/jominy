# pyright: basic
"""
Round 2: refine around the leaders found in round 1.

Round 1 leaders (J9):
  pls_n3_full          MAE 1.7555
  pls_n4_full          MAE 1.7617
  stack_pls_ridge_xgb  MAE 1.7635
  pls_n5_full          MAE 1.7641
  xgb_n800_lr0.02_md3  MAE 1.7688
  bayesian_ridge_full  MAE 1.7696

This round: feature engineering (interactions/ratios driven by hardenability physics),
PLS sweep, fine XGBoost tuning, stack with PLS-n3 instead of n=4, target transformations.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from modeling.config import DELTA_DATASET_PATH, J9_DATASET_PATH  # noqa: E402
from modeling.features import CORE_FEATURES, FULL_FEATURES  # noqa: E402
from model_experiments import (  # noqa: E402
    RANDOM_SEED,
    cv_score,
    imputed_pipe,
    run,
    scaled_pipe,
    section,
)


def add_engineered_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add hardenability-physics-motivated interaction terms.

    Hardenability is driven by carbon plus key alloying elements that suppress
    pearlite formation. Common empirical hardenability factors (e.g. Grossmann
    multipliers, the Ideal Diameter formula) multiply per-element factors,
    so log-element and element products should help.
    """
    out = df.copy()
    eps = 1e-6
    for col in ["C", "Mn", "Cr", "Ni", "Cu", "Si"]:
        out[f"log_{col}"] = np.log(out[col].fillna(0) + eps)
    # interactions
    out["C_x_Mn"] = out["C"] * out["Mn"]
    out["C_x_Cr"] = out["C"] * out["Cr"]
    out["C_x_Mn_x_Cr"] = out["C"] * out["Mn"] * out["Cr"]
    out["Cr_x_Mn"] = out["Cr"] * out["Mn"]
    out["Cr_x_Ni"] = out["Cr"] * out["Ni"].fillna(0)
    out["sumCEQ"] = out["C"] + out["Mn"] / 6.0 + (out["Cr"] + out["Ni"].fillna(0)) / 5.0  # carbon-equivalent style
    new_cols = [
        "log_C", "log_Mn", "log_Cr", "log_Ni", "log_Cu", "log_Si",
        "C_x_Mn", "C_x_Cr", "C_x_Mn_x_Cr", "Cr_x_Mn", "Cr_x_Ni", "sumCEQ",
    ]
    return out, new_cols


def main() -> None:
    j9 = pd.read_parquet(J9_DATASET_PATH)
    delta = pd.read_parquet(DELTA_DATASET_PATH)

    # ----- engineered features -----
    j9_eng, new_eng = add_engineered_features(j9)
    delta_eng, _ = add_engineered_features(delta)
    feats_eng_full = FULL_FEATURES + new_eng
    feats_eng_core = CORE_FEATURES + new_eng

    print(f"engineered features added: {new_eng}")
    print(f"FULL+ENG = {len(feats_eng_full)} features; CORE+ENG = {len(feats_eng_core)} features\n")

    # ===== J9 =====
    section("J9: feature engineering — PLS")
    for n in (2, 3, 4, 5, 6, 8, 10):
        run(
            "J9",
            j9_eng,
            f"pls_n{n}_full+eng",
            feats_eng_full,
            "FULL+ENG",
            lambda nn=n: scaled_pipe(PLSRegression(n_components=nn, scale=False), feats_eng_full),
        )

    section("J9: feature engineering — Ridge / Lasso / ElasticNet")
    for alpha in (0.5, 1.0, 2.0, 5.0, 10.0):
        run(
            "J9",
            j9_eng,
            f"ridge_a{alpha}_full+eng",
            feats_eng_full,
            "FULL+ENG",
            lambda a=alpha: scaled_pipe(Ridge(alpha=a), feats_eng_full),
        )
    run(
        "J9",
        j9_eng,
        "elasticnet_a0.05_l1_0.7_full+eng",
        feats_eng_full,
        "FULL+ENG",
        lambda: scaled_pipe(ElasticNet(alpha=0.05, l1_ratio=0.7, max_iter=20000), feats_eng_full),
    )
    run(
        "J9",
        j9_eng,
        "bayesian_ridge_full+eng",
        feats_eng_full,
        "FULL+ENG",
        lambda: scaled_pipe(BayesianRidge(), feats_eng_full),
    )

    section("J9: XGBoost tight sweep around xgb_n800_lr0.02_md3")
    try:
        from xgboost import XGBRegressor

        for n_est in (500, 800, 1200):
            for lr in (0.01, 0.02, 0.03):
                for md in (3, 4):
                    p = dict(
                        n_estimators=n_est, max_depth=md, learning_rate=lr,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                        reg_alpha=0.0, random_state=RANDOM_SEED, n_jobs=-1,
                    )
                    run(
                        "J9",
                        j9,
                        f"xgb_n{n_est}_lr{lr}_md{md}",
                        FULL_FEATURES,
                        "FULL",
                        lambda pp=p: imputed_pipe(XGBRegressor(**pp)),
                    )
    except ImportError:
        print("xgboost missing — skipping")

    section("J9: stacking variants around new leader")
    # try stacking around pls_n3 instead of n=4
    try:
        from xgboost import XGBRegressor

        xgb_best = lambda: imputed_pipe(  # noqa: E731
            XGBRegressor(
                n_estimators=800, max_depth=3, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
        )
    except ImportError:
        xgb_best = lambda: imputed_pipe(  # noqa: E731
            HistGradientBoostingRegressor(
                learning_rate=0.05, max_depth=3, max_leaf_nodes=15,
                min_samples_leaf=10, max_iter=200, random_state=RANDOM_SEED,
            )
        )

    estimators_a = [
        ("pls3", scaled_pipe(PLSRegression(n_components=3, scale=False))),
        ("bayes", scaled_pipe(BayesianRidge())),
        ("xgb", xgb_best()),
    ]
    run(
        "J9",
        j9,
        "stack_pls3_bayes_xgb",
        FULL_FEATURES,
        "FULL",
        lambda: StackingRegressor(estimators=estimators_a, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1),
        "stack pls3+bayes+xgb",
    )

    estimators_b = [
        ("pls3", scaled_pipe(PLSRegression(n_components=3, scale=False))),
        ("pls5", scaled_pipe(PLSRegression(n_components=5, scale=False))),
        ("ridge", scaled_pipe(Ridge(alpha=1.0))),
    ]
    run(
        "J9",
        j9,
        "stack_pls3_pls5_ridge",
        FULL_FEATURES,
        "FULL",
        lambda: StackingRegressor(estimators=estimators_b, final_estimator=BayesianRidge(), cv=5, n_jobs=-1),
        "all-linear stack",
    )

    section("J9: target log transform on PLS")
    # log-target regression — useful when noise is multiplicative
    j9_logt = j9.copy()
    j9_logt["J9_log"] = np.log(j9_logt["J9"])
    means_log, stds_log = cv_score(
        j9_logt,
        "J9_log",
        FULL_FEATURES,
        lambda: scaled_pipe(PLSRegression(n_components=3, scale=False)),
    )
    # convert back to original-scale MAE/RMSE/R² using simple test-time exp
    # we'll just record the log-scale metrics for now
    from model_experiments import append_row
    append_row("J9", "pls_n3_logtarget", "FULL", means_log, stds_log, "log-target; metrics on log scale (not directly comparable)")

    # ===== delta =====
    section("delta: round 2")
    for n in (2, 3, 4, 5):
        run(
            "delta",
            delta,
            f"pls_n{n}_full",
            FULL_FEATURES,
            "FULL",
            lambda nn=n: scaled_pipe(PLSRegression(n_components=nn, scale=False)),
        )
    run("delta", delta, "ridge_a1.0_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(Ridge(alpha=1.0)))
    run("delta", delta, "bayesian_ridge_full", FULL_FEATURES, "FULL", lambda: scaled_pipe(BayesianRidge()))
    run("delta", delta_eng, "pls_n3_full+eng", feats_eng_full, "FULL+ENG", lambda: scaled_pipe(PLSRegression(n_components=3, scale=False), feats_eng_full))
    run("delta", delta_eng, "pls_n5_full+eng", feats_eng_full, "FULL+ENG", lambda: scaled_pipe(PLSRegression(n_components=5, scale=False), feats_eng_full))
    run("delta", delta_eng, "ridge_a1_full+eng", feats_eng_full, "FULL+ENG", lambda: scaled_pipe(Ridge(alpha=1.0), feats_eng_full))
    run("delta", delta_eng, "bayes_full+eng", feats_eng_full, "FULL+ENG", lambda: scaled_pipe(BayesianRidge(), feats_eng_full))
    run(
        "delta",
        delta,
        "gbr_n200_md3_lr0.05",
        FULL_FEATURES,
        "FULL",
        lambda: imputed_pipe(
            GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED)
        ),
    )
    try:
        from xgboost import XGBRegressor
        run(
            "delta",
            delta,
            "xgb_n800_lr0.02_md3",
            FULL_FEATURES,
            "FULL",
            lambda: imputed_pipe(
                XGBRegressor(n_estimators=800, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=RANDOM_SEED, n_jobs=-1)
            ),
        )
    except ImportError:
        pass

    print("\nDone (round 2). See HISTORY.md")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Elapsed: {time.time() - t0:.1f}s")
