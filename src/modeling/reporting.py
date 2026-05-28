# pyright: basic

from html import escape
from pathlib import Path


_CANDIDATE_NOTES = {
    "ridge_core": "Core-feature ridge benchmark; not the fixed full-feature export reference.",
    "ridge_full": "Full-feature ridge reference for Task 5 benchmarking; final production choice remains deferred.",
    "pls_full": "Benchmark-only PLS challenger with external standardization and internal scaling disabled.",
    "hgbr_full": "Benchmark-only shallow HGBR challenger constrained for the small grouped dataset.",
}


def write_markdown_table(path: Path, title: str, dataframe) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(dataframe.to_markdown(index=False))
        handle.write("\n")


def add_candidate_notes(df, label_col: str = "label"):
    out = df.copy()
    out["selection_policy"] = "benchmark_only"
    out["candidate_note"] = out[label_col].map(_CANDIDATE_NOTES).fillna("Benchmark-only comparison candidate.")
    return out


def _round_metric(value, digits: int = 4):
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        return value
    return round(float(value), digits)


def _best_mae_value(df, target: str):
    subset = df.loc[df["target"] == target]
    if subset.empty or "mae" not in subset.columns:
        return None
    return _round_metric(subset["mae"].mean())


def _best_candidate_row(df, target: str):
    if df.empty or "mae" not in df.columns:
        return None

    label_col = "label" if "label" in df.columns else "model"
    summary = (
        df.loc[df["target"] == target, [label_col, "mae"]]
        .groupby(label_col, as_index=False)
        .agg(mae=("mae", "mean"))
        .sort_values(["mae", label_col], ignore_index=True)
    )
    if summary.empty:
        return None
    row = summary.iloc[0]
    return {"name": row[label_col], "mae": _round_metric(row["mae"])}


def summarize_metric_artifacts(baselines, comparison, prediction_rows: int) -> list[dict]:
    metrics: list[dict] = [
        {"metric": "rows_in_prediction_artifact", "value": int(prediction_rows)},
    ]

    baseline_targets = [
        ("J9", "baseline_best_j9_mae"),
        ("J15", "baseline_best_j15_mae"),
        ("delta_clipped", "baseline_best_delta_clipped_mae"),
    ]
    for target, metric_name in baseline_targets:
        value = _best_mae_value(baselines, target)
        if value is not None:
            metrics.append({"metric": metric_name, "value": value})

    comparison_targets = [
        ("J9", "comparison_best_j9_candidate", "comparison_best_j9_mae"),
        ("delta", "comparison_best_delta_candidate", "comparison_best_delta_mae"),
    ]
    for target, name_metric, mae_metric in comparison_targets:
        best = _best_candidate_row(comparison, target)
        if best is not None:
            metrics.append({"metric": name_metric, "value": best["name"]})
            metrics.append({"metric": mae_metric, "value": best["mae"]})

    return metrics


def build_summary_report_html(metrics, figures, title: str) -> str:
    metric_rows = "".join(
        f"<tr><td>{escape(str(row['metric']))}</td><td>{escape(str(row['value']))}</td></tr>"
        for row in metrics
    )
    figure_blocks = "".join(
        (
            "<section class=\"figure-card\">"
            f"<h3>{escape(str(item['title']))}</h3>"
            f"<img src=\"../figures/{escape(str(item['filename']))}\" "
            f"alt=\"{escape(str(item['title']))}\">"
            "</section>"
        )
        for item in figures
    )
    return f"""
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <title>{escape(title)}</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 1100px; color: #222; line-height: 1.5; }}
      h1, h2, h3 {{ color: #12324a; }}
      table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
      th, td {{ border: 1px solid #ccd6dd; padding: 0.65rem; text-align: left; }}
      th {{ background: #eef4f8; }}
      .warning {{ background: #fff4db; border-left: 5px solid #d18b00; padding: 1rem 1.25rem; margin: 1.5rem 0 2rem; }}
      .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }}
      .figure-card {{ border: 1px solid #dde5ea; border-radius: 8px; padding: 1rem; background: #fafcfd; }}
      img {{ max-width: 100%; height: auto; border: 1px solid #dde5ea; background: white; }}
      code {{ background: #f2f5f7; padding: 0.1rem 0.25rem; border-radius: 4px; }}
    </style>
  </head>
  <body>
    <h1>{escape(title)}</h1>
    <p>Task 6 visual summary for the current Jominy modeling artifacts generated from grouped cross-validation outputs.</p>
    <section class=\"warning\">
      <h2>Deployment Warning</h2>
      <p>
        These supervised metrics come only from the labeled Jominy evaluation set. Deployment beyond that range requires caution:
        the chemistry-only pool is not a validated test set, and predictions on that pool may extrapolate outside the labeled chemistry coverage.
        Treat the chemistry-only pool as an out-of-distribution coverage check, not as evidence of production accuracy.
      </p>
    </section>
    <h2>Key Metrics</h2>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{metric_rows}</tbody>
    </table>
    <h2>Figures</h2>
    <div class=\"figure-grid\">{figure_blocks}</div>
  </body>
</html>
""".strip()
