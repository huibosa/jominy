# pyright: basic

from importlib import import_module


def test_build_specimen_table_produces_expected_counts() -> None:
    modeling_data = import_module("modeling.data")
    long_df = modeling_data.load_cleaned_long()
    specimen_df = modeling_data.build_specimen_table(long_df)

    assert specimen_df["炉号"].nunique() == len(specimen_df)
    assert specimen_df["base_heat_id"].nunique() <= len(specimen_df)
    assert specimen_df["has_j9"].sum() == 566
    assert specimen_df["has_pair"].sum() == 491
    assert (specimen_df.loc[specimen_df["has_pair"], "delta"] >= 0).all()
