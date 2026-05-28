# pyright: basic

import matplotlib.pyplot as plt
import pandas as pd
from typing import Any

from modeling.config import CHEMISTRY_ONLY_PATH, J9_DATASET_PATH, OUTPUT_DIR, REPORTS_DIR


FEATURES = ["C", "Si", "Mn", "P", "S", "Cu", "Ni", "Cr", "V", "Ti", "W", "Al", "B"]


def _format_markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _dataframe_to_markdown(table: pd.DataFrame) -> str:
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def summarize_feature_shift(labeled: pd.DataFrame, chemistry_only: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for feature in features:
        labeled_values = labeled[feature].dropna()
        chemistry_values = chemistry_only[feature].dropna()

        labeled_min = float(labeled_values.min())
        labeled_max = float(labeled_values.max())

        rows.append(
            {
                "feature": feature,
                "labeled_mean": float(labeled_values.mean()),
                "chemistry_only_mean": float(chemistry_values.mean()),
                "mean_gap": float(labeled_values.mean() - chemistry_values.mean()),
                "labeled_min": labeled_min,
                "chemistry_only_min": float(chemistry_values.min()),
                "labeled_max": labeled_max,
                "chemistry_only_max": float(chemistry_values.max()),
                "chemistry_only_below_labeled_min": float((chemistry_values < labeled_min).mean()),
                "chemistry_only_above_labeled_max": float((chemistry_values > labeled_max).mean()),
                "chemistry_only_outside_labeled_range": float(
                    ((chemistry_values < labeled_min) | (chemistry_values > labeled_max)).mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("feature", ignore_index=True)


def build_distribution_shift_markdown(summary: pd.DataFrame) -> str:
    if "chemistry_only_outside_labeled_range" in summary.columns:
        top_outside = summary.sort_values(
            ["chemistry_only_outside_labeled_range", "feature"],
            ascending=[False, True],
            ignore_index=True,
        ).head(5)
    else:
        top_outside = summary.sort_values("feature", ignore_index=True).head(5)
    return "\n".join(
        [
            "# Distribution Shift",
            "",
            "## Extrapolation warning",
            "Predictions on the chemistry-only pool may extrapolate outside the labeled specimen-level chemistry coverage.",
            "The chemistry-only pool is not a validated supervised test set, so this report is a coverage check rather than evidence of production accuracy.",
            "Treat any chemistry-only composition outside the labeled ranges as an extrapolation case requiring extra caution.",
            "",
            "## Feature summary",
            _dataframe_to_markdown(summary),
            "",
            "## Highest outside-range features",
            _dataframe_to_markdown(top_outside),
            "",
        ]
    )


def save_distribution_shift_figure(summary: pd.DataFrame, output_path) -> None:
    plot_df = summary.loc[summary["feature"].isin(["C", "Cr"]), ["feature", "labeled_mean", "chemistry_only_mean"]].copy()
    plot_df = plot_df.sort_values("feature", ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax = plot_df.set_index("feature")[["labeled_mean", "chemistry_only_mean"]].plot.bar(
        figsize=(8, 5),
        color=["#4c78a8", "#f58518"],
        title="Labeled vs Chemistry-Only Mean Chemistry for C and Cr",
    )
    ax.set_xlabel("")
    ax.set_ylabel("weight percent")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    labeled = pd.read_parquet(J9_DATASET_PATH)
    chemistry_only = pd.read_csv(CHEMISTRY_ONLY_PATH)
    summary = summarize_feature_shift(labeled, chemistry_only, FEATURES)

    report_path = REPORTS_DIR / "distribution_shift.md"
    report_path.write_text(build_distribution_shift_markdown(summary), encoding="utf-8")

    figure_path = figures_dir / "distribution_shift_C_Cr.png"
    save_distribution_shift_figure(summary, figure_path)

    print(f"Wrote distribution-shift report to {report_path}")
    print(f"Wrote distribution-shift figure to {figure_path}")


if __name__ == "__main__":
    main()
