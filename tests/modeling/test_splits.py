# pyright: basic

from importlib import import_module


def test_make_outer_splits_has_no_group_overlap() -> None:
    pd = import_module("pandas")
    modeling_splits = import_module("modeling.splits")
    df = pd.DataFrame(
        {
            "base_heat_id": [f"H{i}" for i in range(20)],
            "value": range(20),
        }
    )

    splits = modeling_splits.make_outer_splits(df)

    assert len(splits) == 5
    for train_idx, valid_idx in splits:
        train_groups = set(df.iloc[train_idx]["base_heat_id"])
        valid_groups = set(df.iloc[valid_idx]["base_heat_id"])
        assert train_groups.isdisjoint(valid_groups)


def test_attach_outer_fold_ids_reuses_one_fold_assignment() -> None:
    pd = import_module("pandas")
    modeling_splits = import_module("modeling.splits")
    df = pd.DataFrame(
        {
            "炉号": [f"S{i}" for i in range(10)],
            "base_heat_id": [f"H{i // 2}" for i in range(10)],
            "value": range(10),
        }
    )

    folded = modeling_splits.attach_outer_fold_ids(df)

    assert "outer_fold" in folded.columns
    grouped = folded.groupby("base_heat_id")["outer_fold"].nunique()
    assert (grouped == 1).all()
