"""Batch XLSX/XLS parsing, normalisation, and row classification.

No FastAPI imports — pure parsing logic, easy to unit-test.
HTTP error mapping lives in main.py.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised for any recoverable parse failure → HTTP 400."""


class UnsupportedFormatError(Exception):
    """Raised when the file magic is unrecognised → HTTP 415."""


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(header: bytes) -> Literal["xlsx", "xls", "html"]:
    """Inspect the first bytes of a file and return its format.

    Args:
        header: First 64 bytes (or full content if shorter).

    Returns:
        One of ``"xlsx"``, ``"xls"``, ``"html"``.

    Raises:
        UnsupportedFormatError: If the magic is unrecognised.
    """
    if header[:4] == b"PK\x03\x04":
        return "xlsx"
    if header[:4] == b"\xd0\xcf\x11\xe0":
        return "xls"
    # HTML-as-xls: starts with < after stripping leading whitespace
    stripped = header.lstrip()
    if stripped and stripped[0:1] == b"<":
        return "html"
    raise UnsupportedFormatError(
        "unrecognised file format — upload a .xls or .xlsx file"
    )


# ---------------------------------------------------------------------------
# DataFrame reader
# ---------------------------------------------------------------------------

def read_dataframe(content: bytes, fmt: str) -> pd.DataFrame:
    """Read a file's bytes into a DataFrame.

    First sheet only. All paths use header=0 so column names are always
    taken from row 0.  Excel paths use dtype=object to preserve string
    blanks for per-cell pd.to_numeric coercion downstream.

    Args:
        content: Raw file bytes.
        fmt: One of ``"xlsx"``, ``"xls"``, ``"html"``.

    Returns:
        A non-empty DataFrame.

    Raises:
        ParseError: On any read failure or if the result is empty.
    """
    buf = io.BytesIO(content)
    try:
        if fmt == "xlsx":
            df = pd.read_excel(buf, engine="openpyxl", sheet_name=0, header=0, dtype=object)
        elif fmt == "xls":
            df = pd.read_excel(buf, engine="xlrd", sheet_name=0, header=0, dtype=object)
        else:  # html
            # pd.read_html returns a list; we want the first table.
            # flavor="lxml" + header=0 required: the HTML files use <td> not
            # <th>/<thead> for headers, so header=None would name cols 0,1,2,…
            tables = pd.read_html(buf, flavor="lxml", header=0)
            if not tables:
                raise ParseError("file contains no data rows")
            df = tables[0]
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"could not read file: {exc}") from exc

    if df.empty:
        raise ParseError("file contains no data rows")
    return df


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

# Canonical names (uppercase) → alternative lowercase aliases (as seen in the
# production XLS files: lh=炉号, gh=钢号, c=C, mn=Mn, …)
_ID_ALIASES = {"lh", "炉号"}
_GRADE_ALIASES = {"gh", "钢号"}

_REQUIRED = ["C", "Si", "Mn", "P", "S", "Cu", "Ni", "Cr"]
_OPTIONAL = ["V", "Ti", "W", "Al", "B"]
_ALL_CHEMISTRY = _REQUIRED + _OPTIONAL


@dataclass
class ColumnMap:
    id_column: str | None          # raw DataFrame column name for the heat ID
    grade_column: str | None       # raw column name for the steel grade (optional)
    required: dict[str, str]       # {"C": <raw col>, "Si": <raw col>, …} for present required
    optional: dict[str, str]       # {"V": <raw col>, …} for present optional
    missing_required: list[str]    # required elements with NO column in the file at all


def normalize_columns(df: pd.DataFrame) -> ColumnMap:
    """Map raw DataFrame column names to canonical element names.

    Matching is case-insensitive exact.

    Args:
        df: The raw DataFrame from read_dataframe.

    Returns:
        A ColumnMap describing which columns are present and how they map.

    Raises:
        ParseError: If no heat-ID column is found.
    """
    # Build lookup: normalised → raw
    lookup: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        lookup[key] = str(col)

    def find(aliases: set[str]) -> str | None:
        for alias in aliases:
            if alias in lookup:
                return lookup[alias]
        return None

    id_column = find(_ID_ALIASES)
    if id_column is None:
        looked_for = " / ".join(sorted(_ID_ALIASES))
        raise ParseError(
            f"heat-ID column not found — expected one of: {looked_for}"
        )

    grade_column = find(_GRADE_ALIASES)

    required: dict[str, str] = {}
    optional: dict[str, str] = {}
    missing_required: list[str] = []

    for elem in _REQUIRED:
        raw = lookup.get(elem.lower())
        if raw is not None:
            required[elem] = raw
        else:
            missing_required.append(elem)

    for elem in _OPTIONAL:
        raw = lookup.get(elem.lower())
        if raw is not None:
            optional[elem] = raw

    return ColumnMap(
        id_column=id_column,
        grade_column=grade_column,
        required=required,
        optional=optional,
        missing_required=missing_required,
    )


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedRow:
    id: str
    id_synthesized: bool
    grade: str | None
    composition: dict[str, float | None]   # ALL 13 keys, None where missing
    missing_required: list[str]
    status: Literal["ok", "insufficient", "empty"]


def _coerce(value: object) -> float | None:
    """Coerce a single cell value to float, returning None on failure."""
    if value is None:
        return None
    result = pd.to_numeric(value, errors="coerce")
    if pd.isna(result):
        return None
    return float(result)


def parse_rows(df: pd.DataFrame, mapping: ColumnMap) -> list[ParsedRow]:
    """Parse every DataFrame row into a ParsedRow.

    Returns ALL rows, including those classified "empty".  The caller
    (main.py route) is responsible for filtering empties out of the
    response payload while counting them in the summary.

    Classification logic:
    - ``"empty"``: all 13 chemistry cells are NaN **and**
      ``mapping.missing_required`` is empty (i.e. chemistry columns exist
      but this row has nothing in any of them).
    - ``"insufficient"``: one or more required elements are missing
      (either the column is absent from the file, or the cell is NaN).
    - ``"ok"``: all required elements are present and numeric.

    Args:
        df: Raw DataFrame.
        mapping: Column map from normalize_columns.

    Returns:
        List of ParsedRow, one per spreadsheet row.
    """
    rows: list[ParsedRow] = []

    for row_idx, (_, series) in enumerate(df.iterrows(), start=1):
        # --- Heat ID ---
        id_synthesized = False
        raw_id = series.get(mapping.id_column) if mapping.id_column else None
        id_str = str(raw_id).strip() if (raw_id is not None and not _is_blank(raw_id)) else ""
        if not id_str:
            id_str = f"row-{row_idx}"
            id_synthesized = True

        # --- Grade ---
        grade: str | None = None
        if mapping.grade_column:
            raw_grade = series.get(mapping.grade_column)
            if raw_grade is not None and not _is_blank(raw_grade):
                grade = str(raw_grade).strip() or None

        # --- Chemistry ---
        composition: dict[str, float | None] = {}
        for elem in _ALL_CHEMISTRY:
            if elem in mapping.required:
                composition[elem] = _coerce(series.get(mapping.required[elem]))
            elif elem in mapping.optional:
                composition[elem] = _coerce(series.get(mapping.optional[elem]))
            else:
                composition[elem] = None

        # --- Empty-row detection ---
        # A row is "empty" only when chemistry columns exist in the file but
        # every cell on this row is NaN.  If columns are absent altogether
        # (mapping.missing_required non-empty), rows are classified as
        # "insufficient" so they still show up with meaningful status.
        all_chemistry_null = all(v is None for v in composition.values())
        if all_chemistry_null and not mapping.missing_required:
            rows.append(ParsedRow(
                id=id_str, id_synthesized=id_synthesized, grade=grade,
                composition=composition, missing_required=[],
                status="empty",
            ))
            continue

        # --- Missing-required detection ---
        # Union of file-level missing columns and row-level NaN required cells.
        row_missing = list(mapping.missing_required)
        for elem in _REQUIRED:
            if elem not in mapping.missing_required and composition.get(elem) is None:
                row_missing.append(elem)

        status: Literal["ok", "insufficient", "empty"] = (
            "insufficient" if row_missing else "ok"
        )

        rows.append(ParsedRow(
            id=id_str, id_synthesized=id_synthesized, grade=grade,
            composition=composition, missing_required=row_missing,
            status=status,
        ))

    return rows


def _is_blank(value: object) -> bool:
    """Return True if value is a float/None NaN or a whitespace-only string."""
    if value is None:
        return True
    if isinstance(value, float):
        return pd.isna(value)
    if isinstance(value, str):
        return value.strip() == ""
    return False


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_rows(rows: list[ParsedRow]) -> tuple[list[ParsedRow], int]:
    """Keep the first occurrence of each heat ID, drop subsequent duplicates.

    Rows with synthesized IDs (``row-N``) are unique by construction and are
    never dropped.

    Args:
        rows: All rows from parse_rows(), including ``"empty"`` ones.

    Returns:
        ``(deduplicated_list, number_of_duplicate_rows_removed)``
    """
    seen: set[str] = set()
    result: list[ParsedRow] = []
    deduped = 0
    for row in rows:
        if row.id_synthesized:
            # Synthesized IDs are row-index-based — always unique.
            result.append(row)
            continue
        if row.id in seen:
            deduped += 1
            continue
        seen.add(row.id)
        result.append(row)
    return result, deduped
