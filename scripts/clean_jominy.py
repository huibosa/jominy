#!/usr/bin/env python3
"""Clean Jominy end-quench test data and merge into long format.

J9 = hardness at 9mm from quenched end
J15 = hardness at 15mm from quenched end

Output:
  - jominy_cleaned.parquet: long-format dataset for training (hardness not null)
  - jominy_chemistry_only.parquet: rows without hardness (for representativeness checks)
  - cleaning_report.json: step-by-step audit
  - quarantine.json: quarantined rows with reasons

Usage:
    uv run --with pandas,openpyxl,pyarrow scripts/clean_jominy.py
"""

import json
import re
from pathlib import Path

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
GRADES = {"J9": 9, "J15": 15}  # grade name -> distance in mm
ELEM_COLS = ["C", "Si", "Mn", "P", "S", "Cu", "Ni", "Cr", "V", "Ti", "W", "Al", "B"]
# Elements with high null rates — add missingness flags for these
SPARSE_ELEM_COLS = ["V", "Ti", "W", "Al", "B"]
# Lab tolerance for element agreement (relative) — values within this are considered same measurement
ELEM_TOLERANCE = 0.05  # 5% relative difference


def load_raw_data(grade: str) -> pd.DataFrame:
    """Load the master 数据汇总 file for a given grade."""
    path = DATA_DIR / f"{grade}-数据汇总.xlsx"
    df = pd.read_excel(path)
    return df


def save_outputs(
    df_clean: pd.DataFrame,
    df_chem_only: pd.DataFrame,
    report: dict,
    quarantine: list[dict],
) -> None:
    """Save all output artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(OUTPUT_DIR / "jominy_cleaned.parquet", index=False)
    df_clean.to_csv(OUTPUT_DIR / "jominy_cleaned.csv", index=False)
    if len(df_chem_only) > 0:
        df_chem_only.to_parquet(OUTPUT_DIR / "jominy_chemistry_only.parquet", index=False)
        df_chem_only.to_csv(OUTPUT_DIR / "jominy_chemistry_only.csv", index=False)
    with open(OUTPUT_DIR / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "quarantine.json", "w", encoding="utf-8") as f:
        json.dump(quarantine, f, ensure_ascii=False, indent=2)


def remove_junk_rows(df: pd.DataFrame, grade: str) -> tuple[pd.DataFrame, dict]:
    """Remove rows that are duplicate headers or have non-coercible element values."""
    before = len(df)

    # First, try to coerce element columns to numeric
    for col in ELEM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Identify full duplicate header rows (炉号 is '炉号' and all elements became NaN after coercion)
    mask_header = df["炉号"] == "炉号"

    # After coercion, non-numeric strings become NaN. But we need to distinguish
    # "was a valid number stored as string" (now numeric) from "was actual text" (now NaN).
    # Rows where ALL elements are NaN after coercion AND 炉号 is non-standard are junk.
    mask_all_elem_nan = df[ELEM_COLS].isna().all(axis=1)
    mask_non_standard_heat = ~df["炉号"].str.match(r'^[A-Za-z0-9]', na=False)

    junk_mask = mask_header | (mask_all_elem_nan & mask_non_standard_heat)

    n_headers = mask_header.sum()
    n_junk = junk_mask.sum()

    df_clean = df[~junk_mask].copy()

    removed = before - len(df_clean)
    step_report = {
        "step": "remove_junk_rows",
        "rows_before": before,
        "rows_after": len(df_clean),
        "rows_removed": removed,
        "duplicate_headers": int(n_headers),
    }
    print(f"  remove_junk_rows: removed {removed} rows ({n_headers} header rows)")
    return df_clean, step_report


def normalize_heat_number(df: pd.DataFrame, grade: str, quarantine: list) -> tuple[pd.DataFrame, dict]:
    """Normalize 炉号 format, derive base_heat_id, quarantine invalid entries."""
    before = len(df)

    # Strip whitespace and normalize full-width hyphens
    df["炉号"] = df["炉号"].str.strip().str.replace("−", "-", regex=False).str.replace("—", "-", regex=False)

    # Quarantine bare numeric IDs (not auto-drop — check first)
    mask_bare_number = df["炉号"].str.match(r'^\d+$', na=False)
    bare_rows = df[mask_bare_number]
    for _, row in bare_rows.iterrows():
        has_hardness = pd.notna(row[grade])
        has_chemistry = row[ELEM_COLS].notna().any()
        reason = "bare_numeric_id"
        if has_hardness or has_chemistry:
            reason += "_has_data"  # flag for review — might be valid with weird ID
        quarantine.append({
            "grade": grade, "step": "normalize_heat_number",
            "炉号": row["炉号"], "reason": reason,
            "has_hardness": bool(has_hardness), "has_chemistry": bool(has_chemistry),
        })

    # Quarantine steel grade name
    grade_names = {"20CrMnTiH"}
    mask_grade_name = df["炉号"].isin(grade_names)
    grade_rows = df[mask_grade_name]
    for _, row in grade_rows.iterrows():
        quarantine.append({
            "grade": grade, "step": "normalize_heat_number",
            "炉号": row["炉号"], "reason": "steel_grade_name_not_heat_id",
        })

    # Remove quarantined rows
    quarantine_mask = mask_bare_number | mask_grade_name
    df_clean = df[~quarantine_mask].copy()

    # Derive base_heat_id: strip suffix from P-pattern heats
    # P23708592-H → P23708592, P23602033-6 → P23602033
    # 423V2-735 stays 423V2-735 (not a P-pattern, suffix is part of the code)
    def extract_base_heat(heat_id: str) -> str:
        if pd.isna(heat_id):
            return heat_id
        m = re.match(r'^(P\d{8})(-.+)?$', str(heat_id))
        if m:
            return m.group(1)
        return str(heat_id)  # non-P patterns: keep as-is

    df_clean["base_heat_id"] = df_clean["炉号"].apply(extract_base_heat)

    # Report
    standard_pattern = df_clean["炉号"].str.match(r'^P\d{8}$', na=False)
    suffixed_pattern = df_clean["炉号"].str.match(r'^P\d{8}-.+$', na=False)
    n_standard = standard_pattern.sum()
    n_suffixed = suffixed_pattern.sum()
    n_non_standard = (~standard_pattern & ~suffixed_pattern).sum()
    n_base_heats = df_clean["base_heat_id"].nunique()
    n_specimens = df_clean["炉号"].nunique()

    removed = before - len(df_clean)
    step_report = {
        "step": "normalize_heat_number",
        "rows_before": before,
        "rows_after": len(df_clean),
        "rows_removed": removed,
        "quarantined_bare_numeric": int(mask_bare_number.sum()),
        "quarantined_grade_name": int(mask_grade_name.sum()),
        "standard_heats": int(n_standard),
        "suffixed_heats": int(n_suffixed),
        "other_heats": int(n_non_standard),
        "unique_base_heats": int(n_base_heats),
        "unique_specimens": int(n_specimens),
    }
    print(f"  normalize_heat_number: removed {removed} rows")
    print(f"    {n_standard} standard, {n_suffixed} suffixed, {n_non_standard} other 炉号")
    print(f"    {n_base_heats} base heats, {n_specimens} specimens")
    return df_clean, step_report


def resolve_chemistry(df: pd.DataFrame, grade: str) -> tuple[pd.DataFrame, dict]:
    """Resolve chemistry per exact raw 炉号 using most-complete-row + tolerance backfill."""
    before = len(df)

    # Count specimens with inconsistent elements before resolution
    variation_count = 0
    ambiguous_count = 0
    variation_examples = []

    resolved_records = []

    for heat_id, group in df.groupby("炉号"):
        if len(group) == 1:
            # Single row — use as-is
            resolved_records.append(group.iloc[0])
            continue

        # Find the most complete row (most non-null element values)
        completeness = group[ELEM_COLS].notna().sum(axis=1)
        best_idx = completeness.idxmax()
        canonical = group.loc[best_idx].copy()

        # Check for disagreements and attempt tolerance backfill
        has_variation = False
        has_ambiguous = False

        for col in ELEM_COLS:
            if pd.notna(canonical[col]):
                continue  # Already have a value from the most-complete row

            # Canonical is null — check if other rows agree within tolerance
            non_null_vals = group[col].dropna().unique()

            if len(non_null_vals) == 0:
                continue  # All null — leave as null

            if len(non_null_vals) == 1:
                # Single unique value — safe to backfill
                canonical[col] = non_null_vals[0]
                continue

            # Multiple unique values — check if they agree within tolerance
            min_val = non_null_vals.min()
            max_val = non_null_vals.max()
            if min_val > 0:
                rel_diff = (max_val - min_val) / min_val
                if rel_diff <= ELEM_TOLERANCE:
                    # Within tolerance — use mean
                    canonical[col] = non_null_vals.mean()
                else:
                    # Material disagreement — leave as null, flag
                    has_ambiguous = True
            else:
                # min_val is 0 or negative (shouldn't happen for element %, but be safe)
                has_ambiguous = True

            has_variation = True

        if has_variation:
            variation_count += 1
        if has_ambiguous:
            ambiguous_count += 1
            if len(variation_examples) < 5:
                example_cols = []
                for col in ELEM_COLS:
                    vals = group[col].dropna().unique()
                    if len(vals) > 1:
                        min_v = vals.min()
                        max_v = vals.max()
                        rel_d = (max_v - min_v) / min_v if min_v > 0 else float('inf')
                        example_cols.append({
                            "element": col,
                            "values": vals.tolist(),
                            "rel_diff_pct": round(rel_d * 100, 1),
                            "within_tolerance": rel_d <= ELEM_TOLERANCE,
                        })
                variation_examples.append({
                    "heat": heat_id,
                    "n_rows": len(group),
                    "conflicts": example_cols,
                })

        resolved_records.append(canonical)

    # Build resolved DataFrame — one chemistry row per 炉号
    resolved_df = pd.DataFrame(resolved_records).reset_index(drop=True)

    # Now merge resolved chemistry back with original hardness values
    # Keep ALL rows (multiple hardness measurements per 炉号), but with unified chemistry
    chemistry_only = resolved_df[["炉号", "base_heat_id"] + ELEM_COLS]
    # Drop duplicates on 炉号 since chemistry is now 1:1
    chemistry_only = chemistry_only.drop_duplicates(subset=["炉号"])

    # Merge back with original dataframe (keeping all hardness rows)
    df_result = df[["炉号", grade]].merge(
        chemistry_only, on="炉号", how="left"
    )

    step_report = {
        "step": "resolve_chemistry",
        "rows_before": before,
        "rows_after": len(df_result),
        "specimens_with_variation": variation_count,
        "specimens_with_ambiguous_elements": ambiguous_count,
        "method": "most_complete_row_plus_tolerance_backfill",
        "tolerance_pct": ELEM_TOLERANCE * 100,
        "variation_examples": variation_examples,
    }
    print(f"  resolve_chemistry: {variation_count} specimens had varying elements, {ambiguous_count} with ambiguous (material disagreement)")
    print(f"    Method: most-complete-row + {ELEM_TOLERANCE*100}% tolerance backfill")
    return df_result, step_report


def aggregate_hardness(df: pd.DataFrame, grade: str) -> tuple[pd.DataFrame, dict]:
    """Aggregate repeated hardness measurements per specimen × distance."""
    before = len(df)
    before_heats = df["炉号"].nunique()

    # Remove exact duplicate rows first (same heat, same elements, same hardness)
    df_dedup = df.drop_duplicates()
    n_exact_dupes = before - len(df_dedup)

    # Aggregate: one row per 炉号, hardness = mean of replicates
    agg_dict = {grade: ["mean", "count", "std"]}
    for col in ELEM_COLS:
        agg_dict[col] = "first"  # chemistry already resolved — identical across rows
    agg_dict["base_heat_id"] = "first"

    grouped = df_dedup.groupby("炉号", sort=False).agg(agg_dict)

    # Flatten multi-level columns
    grouped.columns = [
        f"{col}_{agg}" if agg != "first" else col
        for col, agg in grouped.columns
    ]

    # Rename aggregated columns
    grouped = grouped.rename(columns={
        f"{grade}_mean": "hardness",
        f"{grade}_count": "hardness_n",
        f"{grade}_std": "hardness_std",
    })

    grouped = grouped.reset_index()

    # Flag specimens with high replicate spread
    # (std > 2 HRC suggests inconsistent measurements)
    high_spread = (grouped["hardness_std"].dropna() > 2.0).sum()

    # Report
    n_multi = (grouped["hardness_n"] > 1).sum()
    n_single = (grouped["hardness_n"] == 1).sum()

    step_report = {
        "step": "aggregate_hardness",
        "rows_before": before,
        "rows_after": len(grouped),
        "exact_duplicates_removed": int(n_exact_dupes),
        "specimens_with_single_measurement": int(n_single),
        "specimens_with_multiple_measurements": int(n_multi),
        "specimens_with_high_spread_std_gt_2": int(high_spread),
    }
    print(f"  aggregate_hardness: {n_exact_dupes} exact dupes removed, {n_multi} specimens with multiple measurements")
    print(f"    {high_spread} specimens with high replicate spread (std > 2 HRC)")
    return grouped, step_report


def filter_null_targets(df: pd.DataFrame, grade: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split into: (1) rows with hardness (training data), (2) rows without (chemistry-only)."""
    mask_has_hardness = df["hardness"].notna()

    df_with_target = df[mask_has_hardness].copy()
    df_chem_only = df[~mask_has_hardness].copy()

    # Report null rates in training data
    null_rates = {}
    for col in ELEM_COLS:
        null_rates[col] = {
            "null_count": int(df_with_target[col].isna().sum()),
            "null_pct": round(df_with_target[col].isna().mean() * 100, 1),
        }

    # Compare chemistry distributions between labeled and unlabeled (selection bias check)
    bias_check = {}
    for col in ["C", "Mn", "Cr"]:  # Key elements for hardenability
        labeled = df_with_target[col].dropna()
        unlabeled = df_chem_only[col].dropna()
        if len(labeled) > 0 and len(unlabeled) > 0:
            bias_check[col] = {
                "labeled_mean": round(float(labeled.mean()), 4),
                "unlabeled_mean": round(float(unlabeled.mean()), 4),
                "diff_pct": round(abs(labeled.mean() - unlabeled.mean()) / labeled.mean() * 100, 1),
            }

    step_report = {
        "step": "filter_null_targets",
        "rows_with_hardness": len(df_with_target),
        "rows_chemistry_only": len(df_chem_only),
        "null_rates_in_training": null_rates,
        "selection_bias_check": bias_check,
    }
    print(f"  filter_null_targets: {len(df_with_target)} with hardness, {len(df_chem_only)} chemistry-only")
    for col, info in bias_check.items():
        print(f"    Selection bias — {col}: labeled={info['labeled_mean']}, unlabeled={info['unlabeled_mean']}, diff={info['diff_pct']}%")
    return df_with_target, df_chem_only, step_report


def enforce_types_and_flags(df: pd.DataFrame, grade: str) -> tuple[pd.DataFrame, dict]:
    """Enforce numeric types, add missingness flags, validate ranges."""
    before = len(df)

    # Enforce numeric types
    for col in ELEM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["hardness"] = pd.to_numeric(df["hardness"], errors="coerce")
    df["hardness_n"] = df["hardness_n"].astype(int)
    df["hardness_std"] = pd.to_numeric(df["hardness_std"], errors="coerce").fillna(0.0)

    # Add missingness flags for sparse elements
    for col in SPARSE_ELEM_COLS:
        df[f"{col}_missing"] = df[col].isna()

    # Validate ranges (flag but don't remove)
    range_checks = {
        "C": (0.1, 0.3),
        "Si": (0.1, 0.5),
        "Mn": (0.5, 1.5),
        "Cr": (0.8, 1.5),
        "B": (0.0001, 0.01),  # Boron is in 0.00x range — check for misplaced decimals
        "hardness": (15, 50),
    }
    range_flags = {}
    for col, (lo, hi) in range_checks.items():
        valid_mask = df[col].notna()
        out_of_range = (valid_mask & ((df[col] < lo) | (df[col] > hi))).sum()
        if out_of_range > 0:
            range_flags[col] = {
                "expected_range": [lo, hi],
                "out_of_range_count": int(out_of_range),
                "out_of_range_values": df[valid_mask & ((df[col] < lo) | (df[col] > hi))][col].unique().tolist()[:5],
            }

    step_report = {
        "step": "enforce_types_and_flags",
        "rows": len(df),
        "missingness_flags_added": [f"{c}_missing" for c in SPARSE_ELEM_COLS],
        "range_flags": range_flags,
    }
    print(f"  enforce_types_and_flags: added {len(SPARSE_ELEM_COLS)} missingness flags")
    if range_flags:
        print(f"    Range flags: {list(range_flags.keys())}")
    return df, step_report


def merge_and_qc(cleaned_grades: dict, chemistry_only_grades: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Merge per-grade DataFrames into long format, run final QC checks."""
    # --- Merge to long format ---
    frames = []
    all_elem_cols_with_flags = ELEM_COLS + [f"{c}_missing" for c in SPARSE_ELEM_COLS]

    for grade, distance in GRADES.items():
        df = cleaned_grades[grade]
        available_cols = [c for c in ["炉号", "base_heat_id", "hardness", "hardness_n", "hardness_std"] + all_elem_cols_with_flags if c in df.columns]
        df_long = df[available_cols].copy()
        df_long["distance"] = distance
        frames.append(df_long)

    merged = pd.concat(frames, ignore_index=True)

    # Sort by 炉号 then distance
    merged = merged.sort_values(["炉号", "distance"]).reset_index(drop=True)

    # Reorder columns
    col_order = ["炉号", "base_heat_id", "distance", "hardness", "hardness_n", "hardness_std"] + ELEM_COLS + [f"{c}_missing" for c in SPARSE_ELEM_COLS]
    col_order = [c for c in col_order if c in merged.columns]
    merged = merged[col_order]

    # --- Monotonicity check: J9 >= J15 for same specimen ---
    # Pivot to compare distances
    pivot = merged.pivot_table(index="炉号", columns="distance", values="hardness")
    if 9 in pivot.columns and 15 in pivot.columns:
        both_present = pivot[[9, 15]].dropna()
        violations = (both_present[9] < both_present[15]).sum()
        violation_examples = both_present[both_present[9] < both_present[15]].head(5)
        violation_list = []
        for heat_id, row in violation_examples.iterrows():
            violation_list.append({
                "炉号": heat_id,
                "J9": round(float(row[9]), 1),
                "J15": round(float(row[15]), 1),
            })
    else:
        violations = 0
        violation_list = []

    # --- Coverage statistics ---
    total_rows = len(merged)
    unique_specimens = merged["炉号"].nunique()
    unique_base_heats = merged["base_heat_id"].nunique()

    heat_distances = merged.groupby("炉号")["distance"].nunique()
    both_distances = (heat_distances == 2).sum()
    one_distance = (heat_distances == 1).sum()

    j9_specimens = set(cleaned_grades["J9"]["炉号"].unique()) if "J9" in cleaned_grades else set()
    j15_specimens = set(cleaned_grades["J15"]["炉号"].unique()) if "J15" in cleaned_grades else set()
    shared = j9_specimens & j15_specimens

    # --- Chemistry-only artifact ---
    chem_frames = []
    for grade, df in chemistry_only_grades.items():
        chem_cols = ["炉号", "base_heat_id"] + ELEM_COLS + [f"{c}_missing" for c in SPARSE_ELEM_COLS]
        available_cols = [c for c in chem_cols if c in df.columns]
        chem_frames.append(df[available_cols].copy())

    chem_merged = pd.concat(chem_frames, ignore_index=True).drop_duplicates(subset=["炉号"])
    chem_merged = chem_merged.sort_values("炉号").reset_index(drop=True)

    # --- Build report ---
    step_report = {
        "step": "merge_and_qc",
        "total_rows": total_rows,
        "unique_specimens": unique_specimens,
        "unique_base_heats": unique_base_heats,
        "specimens_with_both_distances": int(both_distances),
        "specimens_with_one_distance": int(one_distance),
        "j9_only_specimens": len(j9_specimens - j15_specimens),
        "j15_only_specimens": len(j15_specimens - j9_specimens),
        "shared_specimens": len(shared),
        "monotonicity_violations": int(violations),
        "monotonicity_violation_examples": violation_list,
        "hardness_stats": {
            "mean": round(float(merged["hardness"].mean()), 2),
            "std": round(float(merged["hardness"].std()), 2),
            "min": round(float(merged["hardness"].min()), 2),
            "max": round(float(merged["hardness"].max()), 2),
        },
        "chemistry_only_rows": len(chem_merged),
    }

    print(f"\n  merge_and_qc:")
    print(f"    Total rows: {total_rows}")
    print(f"    Unique specimens: {unique_specimens}, base heats: {unique_base_heats}")
    print(f"    Both distances: {both_distances}, one distance: {one_distance}")
    print(f"    Monotonicity violations (J9 < J15): {violations}")
    if violations > 0:
        print(f"    ⚠  Examples: {violation_list[:3]}")
    print(f"    Hardness range: {merged['hardness'].min():.1f} - {merged['hardness'].max():.1f}")
    print(f"    Chemistry-only artifact: {len(chem_merged)} rows")

    return merged, chem_merged, step_report


def main():
    report = {"steps": []}
    quarantine_log = []
    cleaned_grades = {}
    chemistry_only_grades = {}

    for grade, distance in GRADES.items():
        print(f"\n{'='*60}")
        print(f"Processing {grade} (distance={distance}mm)")
        print(f"{'='*60}")

        grade_report = {"grade": grade, "distance_mm": distance, "steps": []}
        df = load_raw_data(grade)
        initial_rows = len(df)
        grade_report["initial_rows"] = initial_rows
        print(f"Loaded {initial_rows} rows")

        # Task 2: Remove junk rows
        df, step_report = remove_junk_rows(df, grade)
        grade_report["steps"].append(step_report)

        # Task 3: Normalize 炉号 + derive base_heat_id
        df, step_report = normalize_heat_number(df, grade, quarantine_log)
        grade_report["steps"].append(step_report)

        # Task 4: Resolve chemistry per specimen
        df, step_report = resolve_chemistry(df, grade)
        grade_report["steps"].append(step_report)

        # Task 5: Aggregate repeated hardness
        df, step_report = aggregate_hardness(df, grade)
        grade_report["steps"].append(step_report)

        # Task 6: Filter null targets, save chemistry-only
        df, df_chem_only, step_report = filter_null_targets(df, grade)
        grade_report["steps"].append(step_report)
        chemistry_only_grades[grade] = df_chem_only

        # Task 7: Type enforcement + missingness flags
        df, step_report = enforce_types_and_flags(df, grade)
        grade_report["steps"].append(step_report)

        print(f"Final for {grade}: {len(df)} rows (removed {initial_rows - len(df) - len(df_chem_only)})")
        grade_report["final_rows"] = len(df)

        cleaned_grades[grade] = df
        report["steps"].append(grade_report)

    # Task 8: Merge + QC + Save
    merged, chem_merged, merge_report = merge_and_qc(cleaned_grades, chemistry_only_grades)
    report["merge"] = merge_report

    save_outputs(merged, chem_merged, report, quarantine_log)

    print(f"\n{'='*60}")
    print(f"Saved: data/jominy_cleaned.parquet ({len(merged)} rows)")
    print(f"Saved: data/jominy_chemistry_only.parquet ({len(chem_merged)} rows)")
    print(f"Saved: data/cleaning_report.json")
    print(f"Saved: data/quarantine.json ({len(quarantine_log)} entries)")


if __name__ == "__main__":
    main()
