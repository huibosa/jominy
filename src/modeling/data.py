# pyright: basic

from importlib import import_module

from .config import DELTA_DATASET_PATH, J9_DATASET_PATH, LONG_SOURCE_PATH, SPECIMEN_TABLE_PATH
from .features import FULL_FEATURES


def load_cleaned_long():
    pd = import_module("pandas")
    return pd.read_csv(LONG_SOURCE_PATH)


def build_specimen_table(long_df):
    feature_cols = ["炉号", "base_heat_id", *FULL_FEATURES]
    base = long_df[feature_cols].drop_duplicates(subset="炉号").copy()

    hardness = (
        long_df.pivot_table(index="炉号", columns="distance", values="hardness", aggfunc="first")
        .rename(columns={9: "J9", 15: "J15"})
        .reset_index()
    )

    specimen_df = base.merge(hardness, on="炉号", how="left")
    specimen_df["has_j9"] = specimen_df["J9"].notna()
    specimen_df["has_j15"] = specimen_df["J15"].notna()
    specimen_df["has_pair"] = specimen_df["has_j9"] & specimen_df["has_j15"]
    specimen_df["delta"] = specimen_df["J9"] - specimen_df["J15"]

    return specimen_df.sort_values(["base_heat_id", "炉号"]).reset_index(drop=True)


def make_j9_dataset(specimen_df):
    return specimen_df.loc[specimen_df["has_j9"]].reset_index(drop=True)


def make_delta_dataset(specimen_df):
    return specimen_df.loc[specimen_df["has_pair"]].reset_index(drop=True)


def save_modeling_tables(specimen_df) -> None:
    SPECIMEN_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    specimen_df.to_parquet(SPECIMEN_TABLE_PATH, index=False)
    specimen_df.to_csv(SPECIMEN_TABLE_PATH.with_suffix(".csv"), index=False)
    make_j9_dataset(specimen_df).to_parquet(J9_DATASET_PATH, index=False)
    make_delta_dataset(specimen_df).to_parquet(DELTA_DATASET_PATH, index=False)
