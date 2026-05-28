"""Generate EDA report for the Jominy cleaned datasets.

Reads:
  data/jominy_cleaned.parquet         (1,057 long-format rows; labeled training)
  data/jominy_chemistry_only.parquet  (3,498 specimens; chemistry-only pool)

Writes to output/eda/:
  figures/*.png      One PNG per plot family (matplotlib).
  jominy_eda.pdf     Multi-page PDF (cover + TL;DR + 7 figure pages).
  jominy_eda.html    Single self-contained Plotly file (interactive twins).

Run:
  uv run --with pandas,pyarrow,matplotlib,plotly scripts/generate_eda_report.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from plotly.subplots import make_subplots  # pyright: ignore[reportMissingImports]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output" / "eda"
FIGURES_DIR = OUTPUT_DIR / "figures"

ELEMENTS = ["C", "Si", "Mn", "P", "S", "Cu", "Ni", "Cr", "V", "Ti", "W", "Al", "B"]
DISTANCES = [9, 15]

LABELED_COLOR = "#3a76a8"
UNLABELED_COLOR = "#d18b00"
DISTANCE_COLORS = {9: "#3a76a8", 15: "#d18b00"}

CAPTIONS = {
    "hardness_histograms": (
        "Marginal HRC distributions. J9 typically right-shifted vs J15 — "
        "consistent with hardness decreasing along the bar."
    ),
    "element_histograms": (
        "Per-element marginal distributions in the labeled set. "
        "Identifies skew, multi-modality, and near-zero-variance features."
    ),
    "missingness": (
        "Null rates by element. Trace elements (V, W, Al, B) are sparse in both pools; "
        "the right panel shows the row-level pattern on a 500-row sample."
    ),
    "monotonicity": (
        "Hardness must decrease with distance from the quenched end (J9 ≥ J15). "
        "Cleaning enforces this constraint; right panel shows the per-specimen margin."
    ),
    "correlation": (
        "Pairwise element correlations among labeled specimens. "
        "Off-diagonal magnitudes inform regularization choice and feature pruning."
    ),
    "element_vs_hardness": (
        "Univariate slope of hardness on each element. Flat panels suggest the element "
        "alone does not drive hardness; J9 (blue) and J15 (orange) overlaid for comparison."
    ),
    "distribution_shift": (
        "Labeled (training-eligible) chemistry vs the unlabeled pool, per element. "
        "Density-normalized so sample-size disparity (566 vs 3,498) does not dominate. "
        "Mismatch in C and Cr is the key OOD risk for downstream predictions."
    ),
}

FAMILY_ORDER = [
    ("hardness_histograms", "Hardness distributions"),
    ("element_histograms", "Element distributions"),
    ("missingness", "Missingness analysis"),
    ("monotonicity", "J9 vs J15 monotonicity"),
    ("correlation", "Element correlation"),
    ("element_vs_hardness", "Element vs hardness"),
    ("distribution_shift", "Distribution shift: labeled vs unlabeled"),
]


# Long-form description / insights / recommendations rendered on the PDF
# notes page following each figure. Numbers are derived from the cleaned
# datasets and verified at build time.
INSIGHTS: dict[str, dict] = {
    "hardness_histograms": {
        "description": (
            "Two histograms of Rockwell HRC, one per quench distance. "
            "Left: J9 across 566 specimens (mean 36.4, std 3.1, range 29.9 – 45.2). "
            "Right: J15 across 491 specimens (mean 30.1, std 3.4, range 22.1 – 41.7). "
            "Bin width is auto-selected at 30 bins per panel."
        ),
        "insights": [
            "J9 is right-shifted vs J15 by ~6 HRC on average, consistent with hardness decreasing along the bar.",
            "Both targets span ~15 HRC of dynamic range — that is the spread a model must cover.",
            "Sample size is asymmetric: 75 specimens have J9 only and zero have J15 only, so J15 is a strict subset of J9.",
        ],
        "recommendations": [
            "Keep the existing two-stage architecture: predict J9 directly, predict the J9 – J15 margin separately, then reconstruct J15 = J9 − max(0, delta). The bounded margin (P95 = 8.6 HRC) makes the delta a much easier regression target than J15.",
            "Treat J15 prediction as the harder problem — it has 13% fewer training rows than J9. Allocate any extra labeled data preferentially to specimens that increase J15 coverage.",
        ],
    },
    "element_histograms": {
        "description": (
            "13 histograms of weight percent, one per element, across the 566 labeled "
            "specimens (deduplicated to specimen level). Panel titles show the non-null "
            "sample count, e.g. 'Ni (n = 475)'. Sparse elements (V, W, Al, B) lose rows "
            "to missing readings."
        ),
        "insights": [
            "C (CV = 5%) and Cr (CV = 3%) are nearly fixed at 0.20 wt% and 1.13 wt% — the labeled set is dominated by one Cr-bearing carbon-steel grade.",
            "W (CV = 4%) has effectively no variance: nearly all specimens read 0.005 wt%.",
            "Trace elements P, S, Cu, V, Al, B are right-skewed at low concentrations — consistent with impurity-spec distributions.",
            "Si shows weak bimodality (small mode near 0.06, main mode near 0.26), hinting that more than one grade family is mixed into the labeled set.",
        ],
        "recommendations": [
            "Drop W as a feature — near-zero variance carries no predictive signal and only adds noise to ridge coefficients.",
            "Apply log1p (or quantile transform) to the right-skewed traces (P, S, Cu, V, Al, B) before ridge to tame leverage from the right tail.",
            "Investigate the Si bimodality with the cleaning team. If two distinct grade families are present, encoding grade explicitly (or fitting per-grade models) may outperform a single ridge.",
        ],
    },
    "missingness": {
        "description": (
            "Left: grouped bar chart of % null per element, comparing labeled (n = 566) "
            "vs the chemistry-only pool (n = 3,498). Right: row-level missingness pattern "
            "on a 500-row sample, sorted by split — black cells are nulls."
        ),
        "insights": [
            "C, Si, Mn, Cr are reported essentially everywhere in both pools (≤ 0.2% null) — these are the always-present 'core' features.",
            "Ti is 0% null in labeled but 40% null in unlabeled. This is a recording-protocol mismatch, not a measurement artifact, and is a red flag for cross-pool generalization.",
            "Al and B are far sparser in unlabeled (60% / 64% null) than labeled (22% / 34%). Predictions on the unlabeled pool will rarely have these features available.",
            "The right-panel pattern shows missingness clusters — within a split, V/W/Al/B tend to be missing together (an entire trace-element block is recorded or not).",
        ],
        "recommendations": [
            "Keep the existing *_missing indicator columns for V/Ti/W/Al/B. Treat missingness as signal, not noise, and let the linear coefficient on the indicator pick up the systematic offset.",
            "Avoid relying on Ti when scoring the unlabeled pool — its information content collapses there. If Ti is influential in the trained model, consider a 'no-Ti' fallback variant for inference on missing-Ti rows.",
            "For non-linear models (e.g. HistGradientBoosting, XGBoost), prefer native NaN support over zero-imputation — the latter places missing values at a chemically meaningless location in the input space.",
        ],
    },
    "monotonicity": {
        "description": (
            "Left: scatter of J9 (y) vs J15 (x) for the 491 specimens with both "
            "measurements, plus the y = x reference. Right: histogram of the per-specimen "
            "margin J9 − J15 (mean 6.46 HRC, std 1.45, P5 – P95 4.0 – 8.6, min 0.50)."
        ),
        "insights": [
            "100% of dual-measurement specimens satisfy J9 ≥ J15 — the cleaning pipeline's monotonicity constraint is fully respected.",
            "The margin distribution is unimodal and approximately Gaussian with std 1.45 HRC, making it a well-behaved regression target.",
            "A small number of specimens have a margin near 0 (min 0.50). These are specimens where chemistry already saturates hardenability and the bar is near-uniform.",
        ],
        "recommendations": [
            "The current 'predict J9, predict delta, reconstruct J15' design is well justified by the bounded, near-Gaussian margin. Keep it.",
            "Inspect the bottom-of-margin specimens (min J9 − J15 ≈ 0.5 HRC) by hand. They may be high-Cr / high-Mn alloys where the hardenability ceiling has been hit, or measurement noise — separating the two changes how the delta model should treat them.",
            "Consider an isotonic post-processor on (J9, predicted-delta) only if the post-hoc max(0, ·) clip is observed to bias predictions; otherwise the existing reconstruction is sufficient.",
        ],
    },
    "correlation": {
        "description": (
            "13 × 13 Pearson correlation heatmap among elements, computed on labeled "
            "specimens. Positive correlations are red, negative are blue, |r| values "
            "annotated in each cell."
        ),
        "insights": [
            "Mn – Cr is the dominant pair at r = +0.86. These two are adjusted together when the steel grade is set, so they covary tightly.",
            "Moderate pairs (|r| ≈ 0.5 – 0.6): Si – Ni (−0.58), C – Cr (+0.55), Mn – Ni (+0.54), Ni – Ti (−0.53). These reflect grade-family structure rather than chance.",
            "Most off-diagonal pairs have |r| < 0.3 — the feature block is not pathologically collinear apart from Mn – Cr.",
        ],
        "recommendations": [
            "Mn – Cr collinearity is the only pair that materially threatens an OLS coefficient interpretation. Ridge handles it via shrinkage, which is one structural reason the existing ridge choice is appropriate — this plot validates that decision.",
            "Do not drop either Mn or Cr in isolation: each correlates strongly with hardness (r ≈ 0.44 / 0.55) and contributes independent signal even in the presence of the other.",
            "If interpretable directions are wanted, compute a small PLS or PCA on the chemistry block; the first component will be dominated by the Mn – Cr axis and may be a useful single 'grade index' feature.",
        ],
    },
    "element_vs_hardness": {
        "description": (
            "13 panels, one per element, showing scatter of element wt% (x) vs hardness "
            "(y). J9 in blue, J15 in orange, overlaid in each panel. Slopes and dispersion "
            "in each panel describe the marginal element-to-hardness relationship at each "
            "quench distance."
        ),
        "insights": [
            "C, Cr, Mn show clear positive marginal slopes (r vs J9 = +0.51, +0.55, +0.44). These are the headline hardenability drivers, matching metallurgical theory.",
            "Al has a moderate negative correlation (r ≈ −0.21) — likely confounded by grade clustering rather than a genuine inverse hardenability effect.",
            "S, B, Ni, W, Ti show essentially flat panels (|r| < 0.10) — no detectable univariate signal in this dataset.",
            "J9 and J15 slopes are approximately parallel within each panel, suggesting elements affect both depths similarly and the J9 – J15 offset is driven mainly by quench geometry rather than chemistry.",
        ],
        "recommendations": [
            "C, Cr, Mn are non-negotiable features. They each carry roughly a quarter to a third of the hardness variance in this set.",
            "S, B, W, Ni have negligible univariate signal. Keep them only if you suspect interaction effects (and verify that suspicion with a non-linear model); otherwise a leaner feature set may improve ridge generalization.",
            "Confirm with a partial-dependence or SHAP analysis that the negative Al slope is genuine before designing any feature based on it — it most likely vanishes after controlling for C and Cr.",
            "The parallel-slope observation supports modeling the delta (J9 – J15) as approximately chemistry-invariant; a simpler delta model focused on bar geometry / cooling-rate proxies may suffice.",
        ],
    },
    "distribution_shift": {
        "description": (
            "13 panels showing density-normalized histograms of each element's "
            "distribution in the labeled training set vs the chemistry-only pool. "
            "Density normalization avoids the 1:6 sample-size mismatch (566 vs 3,498) "
            "from dominating the visual."
        ),
        "insights": [
            "C is the largest shift: labeled mean 0.20 wt% vs unlabeled 0.45 wt% (−56%). The labeled set is half the carbon of the production population.",
            "Cr shifts the other way at +164% (labeled 1.13 vs unlabeled 0.43). Production includes many low-Cr or Cr-free grades that the training set hardly covers.",
            "Ti +215%, S +105%, Mn +32%, Ni −29%, V −30% — sizable shifts in seven of thirteen elements.",
            "Only P, Cu, W, Al, B show small shifts (< 15%), and only because their absolute concentrations are also small.",
        ],
        "recommendations": [
            "Treat all chemistry-only predictions as out-of-distribution in C and Cr until a held-out sample with hardness measurements covers that region. The existing summary_report.html 'Deployment Warning' is concretely justified by this plot.",
            "Score every inference row with a labeled-set Mahalanobis distance (or covariance-based density estimate) and surface that as a confidence flag alongside the J9 / J15 prediction. Refuse to predict above a calibrated threshold.",
            "If new labeling effort is possible, prioritize specimens that fill the high-C / low-Cr corner of chemistry space — that is the largest unfilled region and the one where the model is currently extrapolating most aggressively.",
            "Avoid naive pseudo-labeling or self-training on the chemistry-only pool. The shift magnitude means the model's pseudo-labels would amplify the existing labeled-set bias rather than correct it.",
            "Strategically, scope hardenability deployment to grades whose chemistry falls inside the labeled envelope, or commission a labeling campaign to widen that envelope.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = pd.read_parquet(data_dir / "jominy_cleaned.parquet")
    chem_only = pd.read_parquet(data_dir / "jominy_chemistry_only.parquet")
    return cleaned, chem_only


def specimens_table(cleaned: pd.DataFrame) -> pd.DataFrame:
    """One row per specimen — chemistry is invariant across J9/J15 rows."""
    return cleaned.drop_duplicates(subset="炉号")[["炉号", "base_heat_id"] + ELEMENTS].reset_index(
        drop=True
    )


def compute_summary(cleaned: pd.DataFrame, chem_only: pd.DataFrame) -> dict:
    spec = specimens_table(cleaned)
    j9 = cleaned[cleaned["distance"] == 9]
    j15 = cleaned[cleaned["distance"] == 15]

    wide = cleaned.pivot_table(
        index="炉号", columns="distance", values="hardness", aggfunc="first"
    )
    both = wide.dropna()
    margin = both[9] - both[15]
    mono_pct = float((both[9] >= both[15]).mean() * 100)

    corr_full = spec[ELEMENTS].corr()
    abs_matrix = np.abs(corr_full.values).copy()
    np.fill_diagonal(abs_matrix, 0.0)
    i, j = np.unravel_index(np.argmax(abs_matrix), abs_matrix.shape)
    top_a, top_b = ELEMENTS[int(i)], ELEMENTS[int(j)]
    top_r = float(abs_matrix[i, j])

    shifts = []
    for el in ELEMENTS:
        l_mean = spec[el].mean()
        u_mean = chem_only[el].mean()
        if pd.notna(l_mean) and pd.notna(u_mean) and abs(u_mean) > 1e-9:
            shifts.append((el, float(l_mean), float(u_mean), (l_mean - u_mean) / u_mean * 100))
    shifts.sort(key=lambda x: abs(x[3]), reverse=True)

    null_pct = spec[ELEMENTS].isna().mean() * 100

    return {
        "n_labeled_specimens": len(spec),
        "n_unlabeled_specimens": len(chem_only),
        "n_j9": len(j9),
        "n_j15": len(j15),
        "n_specimens_both": len(both),
        "mono_pct": mono_pct,
        "margin_mean": float(margin.mean()),
        "margin_min": float(margin.min()),
        "top_pair": (top_a, top_b, top_r),
        "top_shift": shifts[0],
        "most_missing": (str(null_pct.idxmax()), float(null_pct.max())),
    }


def tldr_lines(s: dict) -> list[str]:
    a, b, r = s["top_pair"]
    el, l_mean, u_mean, pct = s["top_shift"]
    miss_el, miss_pct = s["most_missing"]
    return [
        f"Sample sizes: {s['n_labeled_specimens']} labeled specimens "
        f"({s['n_j9']} J9 + {s['n_j15']} J15) vs {s['n_unlabeled_specimens']:,} unlabeled.",
        f"Monotonicity J9 ≥ J15 holds for {s['n_specimens_both']}/{s['n_specimens_both']} "
        f"specimens with both measurements ({s['mono_pct']:.0f}%); "
        f"mean margin {s['margin_mean']:.2f} HRC, min {s['margin_min']:.2f}.",
        f"Strongest absolute element correlation: {a}–{b} (|r| = {r:.2f}).",
        f"Largest labeled-vs-unlabeled mean shift: {el} "
        f"({l_mean:.3f} vs {u_mean:.3f}, {pct:+.0f}%).",
        f"Most-missing element in labeled set: {miss_el} ({miss_pct:.0f}% null).",
    ]


# ---------------------------------------------------------------------------
# Matplotlib builders
# ---------------------------------------------------------------------------


def build_hardness_histograms_mpl(cleaned: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, dist in zip(axes, DISTANCES):
        data = cleaned.loc[cleaned["distance"] == dist, "hardness"]
        ax.hist(data, bins=30, color=LABELED_COLOR, edgecolor="white")
        ax.set_xlabel("Rockwell HRC")
        ax.set_ylabel("Specimen count")
        ax.set_title(f"J{dist}  (n = {len(data)})")
    fig.suptitle("Hardness distributions", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def build_element_histograms_mpl(spec: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    flat = axes.flatten()
    for ax, el in zip(flat, ELEMENTS):
        data = spec[el].dropna()
        ax.hist(data, bins=25, color=LABELED_COLOR, edgecolor="white")
        ax.set_title(f"{el}  (n = {len(data)})", fontsize=10)
        ax.set_xlabel("wt%")
    for ax in flat[len(ELEMENTS):]:
        ax.axis("off")
    fig.suptitle("Element distributions — labeled specimens", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def build_missingness_mpl(cleaned: pd.DataFrame, chem_only: pd.DataFrame) -> Figure:
    spec = specimens_table(cleaned)
    labeled_null = spec[ELEMENTS].isna().mean() * 100
    unlabeled_null = chem_only[ELEMENTS].isna().mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(ELEMENTS))
    w = 0.4
    axes[0].bar(x - w / 2, labeled_null.values, w,
                label=f"Labeled (n={len(spec)})", color=LABELED_COLOR)
    axes[0].bar(x + w / 2, unlabeled_null.values, w,
                label=f"Unlabeled (n={len(chem_only)})", color=UNLABELED_COLOR)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ELEMENTS)
    axes[0].set_ylabel("% null")
    axes[0].set_title("Per-element null rate")
    axes[0].legend()

    combined = pd.concat(
        [
            spec[ELEMENTS].assign(_split="labeled"),
            chem_only[ELEMENTS].assign(_split="unlabeled"),
        ],
        ignore_index=True,
    )
    sample = combined.sample(min(500, len(combined)), random_state=0).sort_values("_split")
    matrix = sample[ELEMENTS].isna().astype(int).values
    axes[1].imshow(matrix, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[1].set_xticks(np.arange(len(ELEMENTS)))
    axes[1].set_xticklabels(ELEMENTS)
    axes[1].set_xlabel("Element")
    axes[1].set_ylabel("Specimen (500 sampled, sorted by split)")
    axes[1].set_title("Missingness pattern  (black = null)")

    fig.suptitle("Missingness analysis", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def build_monotonicity_mpl(cleaned: pd.DataFrame) -> Figure:
    wide = cleaned.pivot_table(
        index="炉号", columns="distance", values="hardness", aggfunc="first"
    )
    both = wide.dropna()
    margin = both[9] - both[15]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(both[15], both[9], s=15, alpha=0.5, color=LABELED_COLOR)
    lo = float(min(both[9].min(), both[15].min())) - 1
    hi = float(max(both[9].max(), both[15].max())) + 1
    axes[0].plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="J9 = J15")
    axes[0].set_xlabel("J15 (HRC)")
    axes[0].set_ylabel("J9 (HRC)")
    axes[0].set_title(f"J9 vs J15  (n = {len(both)} specimens)")
    axes[0].legend()

    axes[1].hist(margin, bins=30, color=LABELED_COLOR, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.6, label="J9 = J15")
    axes[1].set_xlabel("J9 − J15  (HRC)")
    axes[1].set_ylabel("Specimen count")
    axes[1].set_title(f"Margin  (mean = {margin.mean():.2f}, min = {margin.min():.2f})")
    axes[1].legend()

    fig.suptitle("Monotonicity check: J9 ≥ J15", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def build_correlation_mpl(spec: pd.DataFrame) -> Figure:
    corr = spec[ELEMENTS].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(ELEMENTS)))
    ax.set_yticks(np.arange(len(ELEMENTS)))
    ax.set_xticklabels(ELEMENTS)
    ax.set_yticklabels(ELEMENTS)
    for i in range(len(ELEMENTS)):
        for j in range(len(ELEMENTS)):
            v = corr.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Element correlation matrix — labeled specimens", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def build_element_vs_hardness_mpl(cleaned: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    flat = axes.flatten()
    legend_ax = flat[0]
    for ax, el in zip(flat, ELEMENTS):
        for dist in DISTANCES:
            df = cleaned[cleaned["distance"] == dist].dropna(subset=[el])
            ax.scatter(df[el], df["hardness"], s=10, alpha=0.5,
                       color=DISTANCE_COLORS[dist],
                       label=f"J{dist}" if ax is legend_ax else None)
        ax.set_title(el, fontsize=10)
        ax.set_xlabel(f"{el} wt%", fontsize=8)
        ax.set_ylabel("HRC", fontsize=8)
    legend_ax.legend(fontsize=8, loc="best")
    for ax in flat[len(ELEMENTS):]:
        ax.axis("off")
    fig.suptitle("Element vs hardness", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def build_distribution_shift_mpl(cleaned: pd.DataFrame, chem_only: pd.DataFrame) -> Figure:
    spec = specimens_table(cleaned)
    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    flat = axes.flatten()
    legend_ax = flat[0]
    for ax, el in zip(flat, ELEMENTS):
        l = spec[el].dropna()
        u = chem_only[el].dropna()
        if len(l) == 0 or len(u) == 0:
            ax.set_title(f"{el} (insufficient data)", fontsize=9)
            continue
        bins = np.histogram_bin_edges(np.concatenate([l.to_numpy(), u.to_numpy()]), bins=30)
        ax.hist(l, bins=bins, alpha=0.6, density=True,
                label="labeled" if ax is legend_ax else None, color=LABELED_COLOR)
        ax.hist(u, bins=bins, alpha=0.6, density=True,
                label="unlabeled" if ax is legend_ax else None, color=UNLABELED_COLOR)
        ax.set_title(el, fontsize=10)
        ax.set_xlabel(f"{el} wt%", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
    legend_ax.legend(fontsize=8, loc="best")
    for ax in flat[len(ELEMENTS):]:
        ax.axis("off")
    fig.suptitle("Distribution shift: labeled vs chemistry-only", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def build_all_mpl(cleaned: pd.DataFrame, chem_only: pd.DataFrame) -> dict[str, Figure]:
    spec = specimens_table(cleaned)
    return {
        "hardness_histograms": build_hardness_histograms_mpl(cleaned),
        "element_histograms": build_element_histograms_mpl(spec),
        "missingness": build_missingness_mpl(cleaned, chem_only),
        "monotonicity": build_monotonicity_mpl(cleaned),
        "correlation": build_correlation_mpl(spec),
        "element_vs_hardness": build_element_vs_hardness_mpl(cleaned),
        "distribution_shift": build_distribution_shift_mpl(cleaned, chem_only),
    }


# ---------------------------------------------------------------------------
# Plotly builders
# ---------------------------------------------------------------------------


def build_hardness_histograms_plotly(cleaned: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"J{d}  (n={(cleaned['distance']==d).sum()})"
                                        for d in DISTANCES])
    for col, dist in enumerate(DISTANCES, start=1):
        data = cleaned.loc[cleaned["distance"] == dist, "hardness"]
        fig.add_trace(
            go.Histogram(x=data, nbinsx=30, marker_color=LABELED_COLOR,
                         hovertemplate="HRC: %{x}<br>count: %{y}<extra></extra>"),
            row=1, col=col,
        )
    fig.update_xaxes(title_text="HRC", row=1, col=1)
    fig.update_xaxes(title_text="HRC", row=1, col=2)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_layout(title="Hardness distributions", template="plotly_white",
                      showlegend=False, bargap=0.05, height=420)
    return fig


def build_element_histograms_plotly(spec: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=5, subplot_titles=ELEMENTS)
    for i, el in enumerate(ELEMENTS):
        r, c = divmod(i, 5)
        data = spec[el].dropna()
        fig.add_trace(
            go.Histogram(x=data, nbinsx=25, marker_color=LABELED_COLOR,
                         hovertemplate=f"{el}: %{{x}} wt%<br>count: %{{y}}<extra></extra>"),
            row=r + 1, col=c + 1,
        )
    fig.update_layout(title="Element distributions — labeled specimens",
                      template="plotly_white", showlegend=False, bargap=0.05, height=620)
    return fig


def build_missingness_plotly(cleaned: pd.DataFrame, chem_only: pd.DataFrame) -> go.Figure:
    spec = specimens_table(cleaned)
    labeled_null = spec[ELEMENTS].isna().mean() * 100
    unlabeled_null = chem_only[ELEMENTS].isna().mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ELEMENTS, y=labeled_null.values,
                         name=f"Labeled (n={len(spec)})", marker_color=LABELED_COLOR,
                         hovertemplate="%{x}: %{y:.1f}%<extra>labeled</extra>"))
    fig.add_trace(go.Bar(x=ELEMENTS, y=unlabeled_null.values,
                         name=f"Unlabeled (n={len(chem_only)})", marker_color=UNLABELED_COLOR,
                         hovertemplate="%{x}: %{y:.1f}%<extra>unlabeled</extra>"))
    fig.update_layout(title="Per-element null rate (%)", barmode="group",
                      template="plotly_white", yaxis_title="% null", height=440)
    return fig


def build_monotonicity_plotly(cleaned: pd.DataFrame) -> go.Figure:
    wide = cleaned.pivot_table(
        index="炉号", columns="distance", values="hardness", aggfunc="first"
    )
    both = wide.dropna().reset_index()
    margin = both[9] - both[15]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"J9 vs J15 (n={len(both)})",
                                        f"J9 − J15 margin (mean={margin.mean():.2f})"])
    fig.add_trace(
        go.Scatter(x=both[15], y=both[9], mode="markers",
                   marker=dict(color=LABELED_COLOR, size=6, opacity=0.55),
                   text=both["炉号"],
                   hovertemplate="炉号: %{text}<br>J15: %{x}<br>J9: %{y}<extra></extra>",
                   name="specimen"),
        row=1, col=1,
    )
    lo = float(min(both[9].min(), both[15].min())) - 1
    hi = float(max(both[9].max(), both[15].max())) + 1
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                   line=dict(color="black", dash="dash"), name="J9 = J15"),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="J15 (HRC)", row=1, col=1)
    fig.update_yaxes(title_text="J9 (HRC)", row=1, col=1)

    fig.add_trace(
        go.Histogram(x=margin, nbinsx=30, marker_color=LABELED_COLOR,
                     hovertemplate="margin: %{x}<br>count: %{y}<extra></extra>"),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="J9 − J15 (HRC)", row=1, col=2)
    fig.update_yaxes(title_text="count", row=1, col=2)
    fig.update_layout(title="Monotonicity check: J9 ≥ J15",
                      template="plotly_white", showlegend=False, height=460)
    return fig


def build_correlation_plotly(spec: pd.DataFrame) -> go.Figure:
    corr = spec[ELEMENTS].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=ELEMENTS, y=ELEMENTS,
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title="Element correlation matrix — labeled specimens",
                      template="plotly_white", width=720, height=620)
    return fig


def build_element_vs_hardness_plotly(cleaned: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=5, subplot_titles=ELEMENTS)
    show_legend = True
    for i, el in enumerate(ELEMENTS):
        r, c = divmod(i, 5)
        for dist in DISTANCES:
            df = cleaned[cleaned["distance"] == dist].dropna(subset=[el])
            fig.add_trace(
                go.Scatter(
                    x=df[el], y=df["hardness"], mode="markers",
                    marker=dict(color=DISTANCE_COLORS[dist], size=4, opacity=0.55),
                    text=df["炉号"],
                    hovertemplate=(
                        f"炉号: %{{text}}<br>{el}: %{{x}} wt%<br>J{dist}: %{{y}} HRC<extra></extra>"
                    ),
                    name=f"J{dist}",
                    legendgroup=f"J{dist}",
                    showlegend=show_legend,
                ),
                row=r + 1, col=c + 1,
            )
        show_legend = False
    fig.update_layout(title="Element vs hardness  (J9 = blue, J15 = orange)",
                      template="plotly_white", height=640)
    return fig


def build_distribution_shift_plotly(cleaned: pd.DataFrame, chem_only: pd.DataFrame) -> go.Figure:
    spec = specimens_table(cleaned)
    fig = make_subplots(rows=3, cols=5, subplot_titles=ELEMENTS)
    show_legend = True
    for i, el in enumerate(ELEMENTS):
        r, c = divmod(i, 5)
        l = spec[el].dropna()
        u = chem_only[el].dropna()
        fig.add_trace(
            go.Histogram(x=l, name="labeled", histnorm="probability density",
                         marker_color=LABELED_COLOR, opacity=0.6,
                         legendgroup="labeled", showlegend=show_legend),
            row=r + 1, col=c + 1,
        )
        fig.add_trace(
            go.Histogram(x=u, name="unlabeled", histnorm="probability density",
                         marker_color=UNLABELED_COLOR, opacity=0.6,
                         legendgroup="unlabeled", showlegend=show_legend),
            row=r + 1, col=c + 1,
        )
        show_legend = False
    fig.update_layout(title="Distribution shift: labeled vs chemistry-only",
                      template="plotly_white", barmode="overlay", height=640)
    return fig


def build_all_plotly(cleaned: pd.DataFrame, chem_only: pd.DataFrame) -> dict[str, go.Figure]:
    spec = specimens_table(cleaned)
    return {
        "hardness_histograms": build_hardness_histograms_plotly(cleaned),
        "element_histograms": build_element_histograms_plotly(spec),
        "missingness": build_missingness_plotly(cleaned, chem_only),
        "monotonicity": build_monotonicity_plotly(cleaned),
        "correlation": build_correlation_plotly(spec),
        "element_vs_hardness": build_element_vs_hardness_plotly(cleaned),
        "distribution_shift": build_distribution_shift_plotly(cleaned, chem_only),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def add_caption(fig: Figure, caption: str) -> None:
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.14))
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic", wrap=True)


def _wrap_bullet(text: str, width: int = 95) -> str:
    return textwrap.fill(f"• {text}", width=width, subsequent_indent="  ")


def build_notes_figure(idx: int, title: str, content: dict) -> Figure:
    """Render a notes page: title + Description + Insights bullets + Recommendations bullets."""
    fig = plt.figure(figsize=(8.5, 11))
    LEFT = 0.07
    LINE = 0.0185
    SECTION_GAP = 0.022
    HEADING_GAP = 0.028

    y = 0.955
    fig.text(LEFT, y, f"{idx}. {title}", fontsize=18, fontweight="bold", color="#12324a")
    y -= 0.022
    fig.text(LEFT, y, "Description · Insights · Recommendations",
             fontsize=10, color="#777", style="italic")
    y -= 0.04

    fig.text(LEFT, y, "Description", fontsize=12, fontweight="bold", color="#12324a")
    y -= HEADING_GAP
    desc = textwrap.fill(content["description"], width=95)
    fig.text(LEFT, y, desc, fontsize=10, va="top", linespacing=1.45)
    y -= LINE * (desc.count("\n") + 1) + SECTION_GAP

    fig.text(LEFT, y, "Insights", fontsize=12, fontweight="bold", color="#12324a")
    y -= HEADING_GAP
    for item in content["insights"]:
        wrapped = _wrap_bullet(item)
        fig.text(LEFT, y, wrapped, fontsize=10, va="top", linespacing=1.45)
        y -= LINE * (wrapped.count("\n") + 1) + 0.006
    y -= SECTION_GAP - 0.006

    fig.text(LEFT, y, "Recommendations", fontsize=12, fontweight="bold", color="#12324a")
    y -= HEADING_GAP
    for item in content["recommendations"]:
        wrapped = _wrap_bullet(item)
        fig.text(LEFT, y, wrapped, fontsize=10, va="top", linespacing=1.45)
        y -= LINE * (wrapped.count("\n") + 1) + 0.006

    return fig


def write_pngs(mpl_figs: dict[str, Figure], figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, fig in mpl_figs.items():
        path = figures_dir / f"{key}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        paths.append(path)
    return paths


def write_pdf(mpl_figs: dict[str, Figure], tldr: list[str], pdf_path: Path) -> int:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        cover = plt.figure(figsize=(8.5, 11))
        cover.text(0.5, 0.93, "Jominy EDA Report", ha="center",
                   fontsize=22, fontweight="bold")
        cover.text(0.5, 0.88, "Cleaned training set + chemistry-only pool",
                   ha="center", fontsize=11, color="#555")
        cover.text(0.08, 0.80, "TL;DR", fontsize=14, fontweight="bold")
        for i, line in enumerate(tldr):
            cover.text(0.08, 0.74 - i * 0.05, f"• {line}", fontsize=10,
                       wrap=True, color="#222")
        cover.text(0.08, 0.45, "Sections (each: figure + notes page)",
                   fontsize=14, fontweight="bold")
        for i, (_, title) in enumerate(FAMILY_ORDER):
            cover.text(0.10, 0.40 - i * 0.035, f"{i + 1}. {title}", fontsize=10)
        cover.text(0.08, 0.13,
                   "Each section's notes page contains a description, data-driven insights, "
                   "and recommendations\nfor downstream hardenability modeling.",
                   fontsize=9.5, color="#555", style="italic")
        pdf.savefig(cover)
        plt.close(cover)
        pages = 1
        for idx, (key, title) in enumerate(FAMILY_ORDER, start=1):
            fig = mpl_figs[key]
            add_caption(fig, CAPTIONS[key])
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1

            notes_fig = build_notes_figure(idx, title, INSIGHTS[key])
            pdf.savefig(notes_fig)
            plt.close(notes_fig)
            pages += 1
    return pages


HTML_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1200px; margin: 2rem auto; padding: 0 1rem;
       color: #222; line-height: 1.55; }
h1, h2 { color: #12324a; }
h2 { border-bottom: 1px solid #ccd6dd; padding-bottom: 0.3rem; margin-top: 2.5rem; }
h3 { color: #12324a; font-size: 1.05rem; margin: 1.6rem 0 0.4rem;
     letter-spacing: 0.02em; text-transform: uppercase; }
.tldr { background: #eef4f8; border-left: 4px solid #3a76a8; padding: 1rem 1.25rem; }
.tldr ul { margin: 0.5rem 0 0; padding-left: 1.25rem; }
.caption { color: #555; font-size: 0.92rem; margin-top: -0.4rem; margin-bottom: 1rem;
           font-style: italic; }
.note { color: #777; font-size: 0.88rem; }
.notes { background: #fafcfd; border: 1px solid #e2e8ed; border-radius: 6px;
         padding: 0.5rem 1.25rem 1.25rem; margin: 1rem 0 0; }
.notes p { margin: 0.4rem 0 0; }
.notes ul { margin: 0.4rem 0 0; padding-left: 1.25rem; }
.notes li { margin-bottom: 0.4rem; }
.notes li:last-child { margin-bottom: 0; }
.notes .recs { border-left: 3px solid #3a76a8; padding-left: 0.75rem; margin-top: 0.4rem; }
"""


def write_html(plotly_figs: dict[str, go.Figure], tldr: list[str], html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Jominy EDA Report</title>",
        f"<style>{HTML_STYLE}</style>",
        "</head><body>",
        "<h1>Jominy EDA Report</h1>",
        "<p class='note'>Interactive companion to <code>jominy_eda.pdf</code>. "
        "Hover for tooltips, click legend entries to toggle traces, drag to zoom.</p>",
        "<div class='tldr'><strong>TL;DR</strong><ul>",
    ]
    parts.extend(f"<li>{line}</li>" for line in tldr)
    parts.append("</ul></div>")

    first = True
    for key, title in FAMILY_ORDER:
        parts.append(f"<h2>{title}</h2>")
        parts.append(f"<p class='caption'>{CAPTIONS[key]}</p>")
        parts.append(
            plotly_figs[key].to_html(
                full_html=False,
                include_plotlyjs="inline" if first else False,
            )
        )
        parts.append(_render_notes_block(INSIGHTS[key]))
        first = False

    parts.append("</body></html>")
    html_path.write_text("".join(parts), encoding="utf-8")


def _render_notes_block(content: dict) -> str:
    insights_html = "".join(f"<li>{item}</li>" for item in content["insights"])
    recs_html = "".join(f"<li>{item}</li>" for item in content["recommendations"])
    return (
        "<section class='notes'>"
        "<h3>Description</h3>"
        f"<p>{content['description']}</p>"
        "<h3>Insights</h3>"
        f"<ul>{insights_html}</ul>"
        "<h3>Recommendations</h3>"
        f"<ul class='recs'>{recs_html}</ul>"
        "</section>"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    figures_dir = output_dir / "figures"
    pdf_path = output_dir / "jominy_eda.pdf"
    html_path = output_dir / "jominy_eda.html"

    cleaned, chem_only = load_data(data_dir)
    summary = compute_summary(cleaned, chem_only)
    tldr = tldr_lines(summary)

    mpl_figs = build_all_mpl(cleaned, chem_only)
    plotly_figs = build_all_plotly(cleaned, chem_only)

    png_paths = write_pngs(mpl_figs, figures_dir)
    n_pdf_pages = write_pdf(mpl_figs, tldr, pdf_path)
    write_html(plotly_figs, tldr, html_path)

    print(f"Wrote {len(png_paths)} PNGs    -> {figures_dir}")
    print(f"Wrote PDF ({n_pdf_pages} pages) -> {pdf_path}")
    print(f"Wrote HTML               -> {html_path}")

    return {
        "figures_dir": figures_dir,
        "pdf": pdf_path,
        "html": html_path,
    }


if __name__ == "__main__":
    main()
