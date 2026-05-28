# pyright: basic

import json

import joblib
import pandas as pd

from modeling.config import DELTA_DATASET_PATH, J9_DATASET_PATH, MODELS_DIR, REPORTS_DIR
from modeling.features import FULL_FEATURES
from modeling.pipelines import build_ridge_pipeline


def _build_manifest() -> dict:
    return {
        "feature_names": FULL_FEATURES,
        "selection_policy": "fixed_v1_ridge",
        "j9_model": "ridge",
        "delta_model": "ridge",
        "postprocessing": "J15 = J9 - max(0, delta) guarantees J9 >= J15",
    }


def _build_report_text() -> str:
    return "\n".join(
        [
            "# Final Model",
            "",
            "Selected model family: ridge for J9 and ridge for delta.",
            "",
            "Selection policy: fixed-v1 ridge export. PLS and shallow HGBR remain benchmark-only challengers in this plan and are not auto-promoted to production.",
            "",
            "Post-processing reconstructs final J15 predictions as `J9 - max(0, delta)`, guaranteeing `J9 >= J15` for every exported inference result.",
            "",
        ]
    )


def main() -> None:
    j9_df = pd.read_parquet(J9_DATASET_PATH)
    delta_df = pd.read_parquet(DELTA_DATASET_PATH)

    j9_model = build_ridge_pipeline(FULL_FEATURES, alpha=1.0)
    delta_model = build_ridge_pipeline(FULL_FEATURES, alpha=1.0)
    j9_model.fit(j9_df[FULL_FEATURES], j9_df["J9"])
    delta_model.fit(delta_df[FULL_FEATURES], delta_df["delta"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(j9_model, MODELS_DIR / "final_j9_model.joblib")
    joblib.dump(delta_model, MODELS_DIR / "final_delta_model.joblib")

    manifest = _build_manifest()
    with (MODELS_DIR / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    (REPORTS_DIR / "final_model.md").write_text(_build_report_text(), encoding="utf-8")

    print(f"Wrote {MODELS_DIR / 'final_j9_model.joblib'}")
    print(f"Wrote {MODELS_DIR / 'final_delta_model.joblib'}")
    print(f"Wrote {MODELS_DIR / 'feature_manifest.json'}")
    print(f"Wrote {REPORTS_DIR / 'final_model.md'}")


if __name__ == "__main__":
    main()
