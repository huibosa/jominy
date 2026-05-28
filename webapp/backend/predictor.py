"""Prediction logic — load the persisted blend models and return J9 / J15."""
from __future__ import annotations

import json
import sys as _sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _models_dir() -> Path:
    if getattr(_sys, "frozen", False):
        # PyInstaller --onedir: _MEIPASS is the _internal/ dir next to main.exe
        return Path(getattr(_sys, "_MEIPASS")) / "models"
    # Dev mode: models/ is one level above backend/
    return Path(__file__).resolve().parents[1] / "models"

MODELS_DIR = _models_dir()


class Predictor:
    def __init__(self) -> None:
        self.j9 = joblib.load(MODELS_DIR / "j9_blend.joblib")
        self.delta = joblib.load(MODELS_DIR / "delta_blend.joblib")
        self.metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
        self.feature_stats = json.loads((MODELS_DIR / "feature_stats.json").read_text(encoding="utf-8"))
        self.features: list[str] = self.metadata["features"]

    def predict(self, composition: dict[str, float | None]) -> dict:
        """Predict J9, J15, delta from a chemical composition.

        composition: dict mapping element symbols (C, Si, Mn, ...) to wt% values
        (None or missing for unmeasured trace elements).

        Missingness flags for V/Ti/W/Al/B are auto-derived from None values.
        """
        row = self._build_row(composition)
        df = pd.DataFrame([row], columns=self.features)

        j9_xgb = float(np.asarray(self.j9["xgb"].predict(df)).reshape(-1)[0])
        j9_pls = float(np.asarray(self.j9["pls"].predict(df)).reshape(-1)[0])
        j9 = self.j9["weights"]["xgb"] * j9_xgb + self.j9["weights"]["pls"] * j9_pls

        delta_xgb = float(np.asarray(self.delta["xgb"].predict(df)).reshape(-1)[0])
        delta_bayes = float(np.asarray(self.delta["bayes"].predict(df)).reshape(-1)[0])
        delta = self.delta["weights"]["xgb"] * delta_xgb + self.delta["weights"]["bayes"] * delta_bayes

        delta_clipped = max(0.0, delta)
        j15 = j9 - delta_clipped

        warnings = self._check_inputs(composition)

        return {
            "J9": round(j9, 2),
            "J15": round(j15, 2),
            "delta": round(delta_clipped, 2),
            "components": {
                "j9_xgb": round(j9_xgb, 2),
                "j9_pls": round(j9_pls, 2),
                "delta_xgb": round(delta_xgb, 2),
                "delta_bayes": round(delta_bayes, 2),
            },
            "warnings": warnings,
            "expected_mae": {
                "J9": self.metadata["expected_oof_metrics"]["J9"]["mae"],
                "delta": self.metadata["expected_oof_metrics"]["delta"]["mae"],
            },
        }

    def _build_row(self, composition: dict[str, float | None]) -> dict[str, float | int]:
        """Map user composition to the model's feature schema."""
        row: dict[str, float | int] = {}
        for feat in self.features:
            if feat.endswith("_missing"):
                element = feat.replace("_missing", "")
                v = composition.get(element)
                row[feat] = 1 if v is None else 0
            else:
                v = composition.get(feat)
                row[feat] = float("nan") if v is None else float(v)
        return row

    def _check_inputs(self, composition: dict[str, float | None]) -> list[str]:
        warnings: list[str] = []
        for element, val in composition.items():
            if val is None:
                continue
            stats = self.feature_stats.get(element)
            if not stats or stats.get("is_flag"):
                continue
            if val < stats["p01"] or val > stats["p99"]:
                warnings.append(
                    f"{element}={val:g} is outside the typical training range "
                    f"[{stats['p01']:.4g}, {stats['p99']:.4g}] — extrapolation, prediction may be unreliable."
                )
        return warnings
