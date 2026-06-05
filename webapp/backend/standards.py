"""Hardcoded GB/T 5216-2013 standard compositions for steel grade families.

Each entry maps element symbol → midpoint value (wt%):
  - Specified range   → midpoint
  - Upper-bound only  → upper_limit / 2
  - Lower-bound only  → representative typical midpoint
  - Not in standard   → key absent (model treats element as missing)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# GB/T 5216-2013 — 20CrMnTiH family
# Applies to: 20CrMnTi, 20CrMnTiH, and all processing / quality variants
# (suffixes: (QC), (SH), (DCH), (DCY), (ZH), (LD), H3, H4, …)
# ---------------------------------------------------------------------------
_20CRMNTI: dict[str, float] = {
    "C":  0.200,  # 0.17–0.23  → midpoint
    "Si": 0.270,  # 0.17–0.37  → midpoint
    "Mn": 0.950,  # 0.80–1.10  → midpoint
    "P":  0.018,  # ≤ 0.035   → 0.035 / 2
    "S":  0.018,  # ≤ 0.035   → 0.035 / 2
    "Cu": 0.150,  # ≤ 0.30    → 0.30  / 2  (residual element)
    "Ni": 0.150,  # ≤ 0.30    → 0.30  / 2  (residual element)
    "Cr": 1.150,  # 1.00–1.30  → midpoint
    "Ti": 0.070,  # 0.04–0.10  → midpoint
    "Al": 0.035,  # Als ≥ 0.020 (lower-bound only); 0.035 = representative typical midpoint
    # V, W, B: not specified in GB/T 5216-2013 for this grade family
}


def grade_lookup(grade: str) -> dict[str, float] | None:
    """Return the GB standard midpoint composition for *grade*, or None if unknown.

    Matching rule: any grade string containing ``'20CrMnTi'`` (case-insensitive)
    is treated as the 20CrMnTiH family.  This covers the 16 known variants in
    the production data (processing suffixes, hardenability bands H3/H4, etc.).
    """
    if "20crmnti" in grade.lower():
        return _20CRMNTI
    return None
