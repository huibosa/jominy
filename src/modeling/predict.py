# pyright: basic

import json
from pathlib import Path

import joblib
import pandas as pd


def assemble_pair_predictions(df):
    out = df.copy()
    out["delta_pred_clipped"] = out["delta_pred"].clip(lower=0.0)
    out["j15_pred"] = out["j9_pred"] - out["delta_pred_clipped"]
    if {"j9_true", "delta_true"}.issubset(out.columns):
        out["j15_true"] = out["j9_true"] - out["delta_true"]
    return out


def load_model_pair(model_dir: Path):
    j9_model = joblib.load(model_dir / "final_j9_model.joblib")
    delta_model = joblib.load(model_dir / "final_delta_model.joblib")
    with (model_dir / "feature_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return j9_model, delta_model, manifest


def predict_j9_j15(features: pd.DataFrame, j9_model, delta_model, feature_names):
    model_features = features.loc[:, list(feature_names)]
    out = pd.DataFrame(
        {
            "j9_pred": j9_model.predict(model_features),
            "delta_pred": delta_model.predict(model_features),
        },
        index=features.index,
    )
    if "炉号" in features.columns:
        out["炉号"] = features["炉号"]
    return assemble_pair_predictions(out)
