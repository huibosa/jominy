"""Train the production blend on all data and persist to disk.

J9: 0.70 * XGB(n=1500, lr=0.005, md=3, ss=0.55, cs=0.5, l2=2.0) + 0.30 * PLS(3)
delta: 0.60 * XGB(n=800, lr=0.01, md=3, ss=0.8, cs=0.8, l2=2.0) + 0.40 * BayesianRidge

J15 = J9 - max(0, delta)  (enforces J9 >= J15)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from modeling.config import DELTA_DATASET_PATH, J9_DATASET_PATH  # noqa: E402
from modeling.features import FULL_FEATURES  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RANDOM_SEED = 42


def scaled_pipe(model):
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


def imputed_pipe(model):
    pre = ColumnTransformer(transformers=[("num", SimpleImputer(strategy="median"), list(FULL_FEATURES))])
    return Pipeline([("preprocessor", pre), ("model", model)])


def train_j9_blend(j9_df: pd.DataFrame) -> dict:
    from xgboost import XGBRegressor

    X = j9_df[FULL_FEATURES]
    y = j9_df["J9"].to_numpy()
    xgb = imputed_pipe(
        XGBRegressor(
            n_estimators=1500, max_depth=3, learning_rate=0.005,
            subsample=0.55, colsample_bytree=0.5, reg_lambda=2.0,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    )
    pls = scaled_pipe(PLSRegression(n_components=3, scale=False))
    xgb.fit(X, y)
    pls.fit(X, y)
    return {"xgb": xgb, "pls": pls, "weights": {"xgb": 0.70, "pls": 0.30}}


def train_delta_blend(delta_df: pd.DataFrame) -> dict:
    from xgboost import XGBRegressor

    X = delta_df[FULL_FEATURES]
    y = delta_df["delta"].to_numpy()
    xgb = imputed_pipe(
        XGBRegressor(
            n_estimators=800, max_depth=3, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    )
    bayes = scaled_pipe(BayesianRidge())
    xgb.fit(X, y)
    bayes.fit(X, y)
    return {"xgb": xgb, "bayes": bayes, "weights": {"xgb": 0.60, "bayes": 0.40}}


def feature_stats(df: pd.DataFrame) -> dict:
    """Per-feature observed range; used for input validation in the API."""
    stats = {}
    for col in FULL_FEATURES:
        if col.endswith("_missing"):
            stats[col] = {"min": 0, "max": 1, "median": 0, "is_flag": True}
        else:
            s = df[col].dropna()
            stats[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median()),
                "p01": float(s.quantile(0.01)),
                "p99": float(s.quantile(0.99)),
                "is_flag": False,
            }
    return stats


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    j9_df = pd.read_parquet(J9_DATASET_PATH)
    delta_df = pd.read_parquet(DELTA_DATASET_PATH)

    print(f"J9 training rows: {len(j9_df)}")
    print(f"delta training rows: {len(delta_df)}")

    j9_models = train_j9_blend(j9_df)
    delta_models = train_delta_blend(delta_df)

    joblib.dump(j9_models, MODELS_DIR / "j9_blend.joblib")
    joblib.dump(delta_models, MODELS_DIR / "delta_blend.joblib")

    stats = feature_stats(j9_df)
    (MODELS_DIR / "feature_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "features": FULL_FEATURES,
                "j9_train_rows": int(len(j9_df)),
                "delta_train_rows": int(len(delta_df)),
                "j9_blend": {"xgb_weight": 0.70, "pls_weight": 0.30},
                "delta_blend": {"xgb_weight": 0.60, "bayes_weight": 0.40},
                "expected_oof_metrics": {
                    "J9": {"mae": 1.7106, "rmse": 2.2135, "r2": 0.4808},
                    "delta": {"mae": 1.0940, "rmse": 1.4363, "r2": 0.0012},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
