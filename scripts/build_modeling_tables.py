# pyright: basic

from importlib import import_module


def main() -> None:
    modeling_data = import_module("modeling.data")
    long_df = modeling_data.load_cleaned_long()
    specimen_df = modeling_data.build_specimen_table(long_df)
    j9_df = modeling_data.make_j9_dataset(specimen_df)
    delta_df = modeling_data.make_delta_dataset(specimen_df)

    assert specimen_df["炉号"].nunique() == len(specimen_df)
    assert len(j9_df) == 566
    assert len(delta_df) == 491
    assert (delta_df["delta"] >= 0).all()

    modeling_data.save_modeling_tables(specimen_df)
    print("specimen_rows=", len(specimen_df))
    print("j9_rows=", len(j9_df))
    print("delta_rows=", len(delta_df))


if __name__ == "__main__":
    main()
