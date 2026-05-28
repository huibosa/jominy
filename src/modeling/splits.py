# pyright: basic

import json
from importlib import import_module

from .config import FOLDS_DIR, GROUP_COL, OUTER_FOLDS


def make_outer_splits(df):
    sklearn_model_selection = import_module("sklearn.model_selection")
    splitter = sklearn_model_selection.GroupKFold(n_splits=OUTER_FOLDS)
    groups = df[GROUP_COL]
    return list(splitter.split(df, groups=groups))


def attach_outer_fold_ids(df):
    out = df.copy()
    out["outer_fold"] = -1
    column_index = out.columns.get_loc("outer_fold")
    for fold_id, (_, valid_idx) in enumerate(make_outer_splits(df)):
        out.iloc[valid_idx, column_index] = fold_id
    return out


def save_outer_splits(df, splits) -> None:
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    payload = []
    for fold_id, (train_idx, valid_idx) in enumerate(splits):
        payload.append(
            {
                "fold": fold_id,
                "train_groups": sorted(df.iloc[train_idx][GROUP_COL].unique().tolist()),
                "valid_groups": sorted(df.iloc[valid_idx][GROUP_COL].unique().tolist()),
                "valid_specs": sorted(df.iloc[valid_idx]["炉号"].tolist()) if "炉号" in df.columns else [],
            }
        )
    with (FOLDS_DIR / "outer_folds.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
