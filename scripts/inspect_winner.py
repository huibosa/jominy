# pyright: basic
"""Cross-validated prediction quality of the winning blend on J9."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from modeling.config import J9_DATASET_PATH  # noqa: E402
from modeling.features import FULL_FEATURES  # noqa: E402
from model_experiments import imputed_pipe, scaled_pipe  # noqa: E402


def main() -> None:
    j9 = pd.read_parquet(J9_DATASET_PATH)
    X = j9[FULL_FEATURES]
    y = j9["J9"].to_numpy()
    groups = j9["base_heat_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    from xgboost import XGBRegressor

    XGB = dict(
        n_estimators=1500, max_depth=3, learning_rate=0.005,
        subsample=0.55, colsample_bytree=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=-1,
    )

    # Out-of-fold predictions
    oof_xgb = np.zeros(len(y))
    oof_pls = np.zeros(len(y))
    fold_id = np.full(len(y), -1, dtype=int)
    for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
        x_pipe = imputed_pipe(XGBRegressor(**XGB))
        p_pipe = scaled_pipe(PLSRegression(n_components=3, scale=False))
        x_pipe.fit(X.iloc[tr], y[tr])
        p_pipe.fit(X.iloc[tr], y[tr])
        oof_xgb[va] = np.asarray(x_pipe.predict(X.iloc[va])).reshape(-1)
        oof_pls[va] = np.asarray(p_pipe.predict(X.iloc[va])).reshape(-1)
        fold_id[va] = k

    blend = 0.70 * oof_xgb + 0.30 * oof_pls
    err = blend - y
    abserr = np.abs(err)

    print("=== Overall (OOF, n=566) ===")
    print(f"MAE  = {mean_absolute_error(y, blend):.4f}")
    print(f"RMSE = {math.sqrt(mean_squared_error(y, blend)):.4f}")
    print(f"R²   = {r2_score(y, blend):.4f}")
    print(f"Bias (mean residual) = {err.mean():+.4f} HRC")
    print(f"Median |error|        = {np.median(abserr):.4f}")
    print(f"P75 |error|           = {np.percentile(abserr, 75):.4f}")
    print(f"P90 |error|           = {np.percentile(abserr, 90):.4f}")
    print(f"P95 |error|           = {np.percentile(abserr, 95):.4f}")
    print(f"Max |error|           = {abserr.max():.4f}")
    print(f"Within ±1 HRC: {(abserr <= 1).mean() * 100:.1f}%")
    print(f"Within ±2 HRC: {(abserr <= 2).mean() * 100:.1f}%")
    print(f"Within ±3 HRC: {(abserr <= 3).mean() * 100:.1f}%")
    print(f"Within ±5 HRC: {(abserr <= 5).mean() * 100:.1f}%")

    # Compare to single-model baselines
    print("\n=== Per-component OOF ===")
    for name, pred in [("xgb_only", oof_xgb), ("pls_only", oof_pls), ("blend_0.70_0.30", blend)]:
        print(f"{name:18s}  MAE={mean_absolute_error(y, pred):.4f}  RMSE={math.sqrt(mean_squared_error(y, pred)):.4f}  R²={r2_score(y, pred):+.4f}")

    # Error by hardness bin
    print("\n=== Error by J9 range ===")
    bins = [(29.9, 32), (32, 35), (35, 38), (38, 41), (41, 45.2)]
    for lo, hi in bins:
        m = (y >= lo) & (y < hi)
        n = m.sum()
        if n == 0:
            continue
        mae = mean_absolute_error(y[m], blend[m])
        bias = (blend[m] - y[m]).mean()
        print(f"  J9 ∈ [{lo:.0f}, {hi:.1f})  n={n:3d}  MAE={mae:.3f}  bias={bias:+.3f}")

    # Worst predictions
    j9_pred = j9.copy()
    j9_pred["pred"] = blend
    j9_pred["err"] = err
    j9_pred["abs_err"] = abserr
    j9_pred["fold"] = fold_id
    worst = j9_pred.nlargest(10, "abs_err")[["炉号", "C", "Mn", "Cr", "J9", "pred", "err", "fold"]]
    print("\n=== 10 worst predictions ===")
    print(worst.to_string(index=False))

    # Residual stats by fold
    print("\n=== Per-fold MAE ===")
    for k in range(5):
        m = fold_id == k
        print(f"  fold {k}  n={m.sum():3d}  MAE={mean_absolute_error(y[m], blend[m]):.4f}  RMSE={math.sqrt(mean_squared_error(y[m], blend[m])):.4f}")

    # Save predictions
    out_path = PROJECT_ROOT / "output" / "modeling" / "predictions" / "blend_oof.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    j9_pred[["炉号", "base_heat_id", "fold", "C", "Mn", "Cr", "J9", "pred", "err", "abs_err"]].to_csv(out_path, index=False)
    print(f"\nWrote OOF predictions to {out_path}")


if __name__ == "__main__":
    main()
