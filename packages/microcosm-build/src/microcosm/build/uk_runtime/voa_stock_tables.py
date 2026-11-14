"""VOA's council tax stock of properties by small area: CTSOP 1.1, 3.1 and 4.1.

The stock cell builder (microcosm#1524) counts homes the way VOA does. Each
table is one CSV member of a zip that Chronicle registers raw-only
(PolicyEngine/chronicle#415), one row per area (and per band in 3.1 and 4.1)
for England and Wales at 31 March 2026:

- **CTSOP 1.1**: properties by band, A to I. The builder's cell count is its
  figure as printed.
- **CTSOP 3.1**: properties by band and property type, 43 type × bedroom
  columns. The builder reads only type shares from it, grouped into Price
  Paid's four types.
- **CTSOP 4.1**: properties by band and build period, for the variants that
  use VOA's build-period mix.

VOA rounds each count to the nearest ten, prints ``[c]`` for a count from 1
to 4 and ``[z]`` where a band does not apply (band I outside Wales). The
readers read ``[z]`` as zero and ``[c]`` as a declared value that the
caller passes in, and count the cells they imputed. The 2025 zip used other
markers (``-`` and ``..``), read as zero as the home-stock builder does.

This module is loaded by file path from the builder tool, so it imports no
``uk_runtime`` sibling.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "BANDS",
    "BandCounts",
    "BuildPeriods",
    "CTSOP1_1_MEMBER",
    "CTSOP3_1_MEMBER",
    "CTSOP4_1_MEMBER",
    "GEOGRAPHIES",
    "PRICE_PAID_TYPES",
    "TypeShares",
    "VOA_TYPE_GROUPS",
    "band_counts",
    "build_periods",
    "classify_build_periods",
    "classify_type_columns",
    "member_header",
    "parse_counts",
    "read_member",
    "type_shares",
]

#: The 2026 members; the 2025 CTSOP1.1 member was CTSOP1.1/CTSOP1_1_2025_03_31.csv.
CTSOP1_1_MEMBER = "CTSOP_1_1/CTSOP1_1_2026_03_31.csv"
CTSOP3_1_MEMBER = "CTSOP_3_1/CTSOP3_1_2026_03_31.csv"
CTSOP4_1_MEMBER = "CTSOP_4_1/CTSOP4_1_2026_03_31.csv"

BANDS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I")
GEOGRAPHIES: tuple[str, ...] = (
    "ENGWAL",
    "NATL",
    "REGL",
    "CTYMET",
    "LAUA",
    "MSOA",
    "LSOA",
    "UNMD",
)
SUPPRESSED = "[c]"
NOT_APPLICABLE = "[z]"
_ZERO_MARKERS = (NOT_APPLICABLE, "-", "..", "")
#: Price Paid's four types, in the home-stock library's ``ptype`` order.
PRICE_PAID_TYPES: tuple[str, ...] = ("D", "S", "T", "F")

#: How CTSOP 3.1's property-type columns group, first match wins. Bungalows
#: are split across the three house types by the house mix, and annexes,
#: caravans and unknown types across all four types (:func:`type_shares`).
#: A column that matches no rule is refused, so a renamed column fails here
#: rather than vanishing from the shares; the ``discover`` stage prints the
#: header against these rules.
VOA_TYPE_GROUPS: tuple[tuple[str, str], ...] = (
    (r"^.*_total$", "total"),
    (r"^bungalow", "bungalow"),
    (r"^flat", "F"),
    (r"^house_ter", "T"),
    (r"^house_semi", "S"),
    (r"^house_det", "D"),
    (r"^annexe", "other"),
    (r"^caravan", "other"),
    (r"^unknown", "other"),
)
_KEY_COLUMNS = ("geography", "ba_code", "ecode", "area_name")
_KINDS = ("D", "S", "T", "F", "bungalow", "other")
_LABELS = ("All", *BANDS)
_BUILD_PERIOD = re.compile(
    r"^bp_(?:pre_(?P<before>\d{4})|(?P<first>\d{4})_(?P<last>\d{4})|"
    r"(?P<year>\d{4})|(?P<unknown>unkw))$"
)


def read_member(zip_path: Path, member: str) -> pd.DataFrame:
    """One CTSOP CSV member, every column as text, markers kept as printed."""
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as stream:
        return pd.read_csv(
            stream, dtype=str, encoding="utf-8-sig", keep_default_na=False
        )


def member_header(zip_path: Path, member: str) -> list[str]:
    """The header row of a CTSOP CSV member, read without the rest."""
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as stream:
        text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
        return next(csv.reader(text))


def parse_counts(
    column: pd.Series, *, suppressed_value: float
) -> tuple[np.ndarray, np.ndarray]:
    """Counts as printed, with the markers read; and which cells were ``[c]``.

    Thousands separators are dropped. Any other text is refused.
    """
    text = column.astype(str).str.strip().str.replace(",", "", regex=False)
    suppressed = (text == SUPPRESSED).to_numpy()
    zero = text.isin(_ZERO_MARKERS).to_numpy()
    numeric = ~(suppressed | zero)
    values = np.zeros(len(text), dtype=float)
    values[suppressed] = float(suppressed_value)
    parsed = pd.to_numeric(text[numeric], errors="coerce").to_numpy(dtype=float)
    bad = ~(np.isfinite(parsed) & (parsed >= 0))
    if bad.any():
        raise ValueError(
            f"Unexpected CTSOP cells: {sorted(set(text[numeric][bad]))[:5]}"
        )
    values[numeric] = parsed
    return values, suppressed


def _rows(table: pd.DataFrame, geography: str) -> pd.DataFrame:
    if geography not in GEOGRAPHIES:
        raise ValueError(f"CTSOP has no geography {geography!r}.")
    missing = [column for column in _KEY_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"The CTSOP table lacks {missing}.")
    return table[table["geography"] == geography]


@dataclass(frozen=True)
class BandCounts:
    """Properties by band per area (CTSOP 1.1), as printed."""

    codes: tuple[str, ...]
    counts: np.ndarray
    all_properties: np.ndarray
    suppressed_cells: int
    index: Mapping[str, int] = field(repr=False, default_factory=dict)


def band_counts(
    table: pd.DataFrame, *, geography: str, suppressed_value: float
) -> BandCounts:
    """Each area's count in bands A to I and all properties (CTSOP 1.1)."""
    rows = _rows(table, geography)
    codes = rows["ecode"].astype(str).tolist()
    if len(set(codes)) != len(codes):
        raise ValueError(f"CTSOP 1.1 repeats a {geography} area.")
    columns = [f"band_{band.lower()}" for band in BANDS]
    missing = [c for c in [*columns, "all_properties"] if c not in rows.columns]
    if missing:
        raise ValueError(f"CTSOP 1.1 lacks {missing}.")
    counts = np.zeros((len(rows), len(BANDS)))
    suppressed = 0
    for j, column in enumerate(columns):
        counts[:, j], flags = parse_counts(
            rows[column], suppressed_value=suppressed_value
        )
        suppressed += int(flags.sum())
    total, flags = parse_counts(
        rows["all_properties"], suppressed_value=suppressed_value
    )
    suppressed += int(flags.sum())
    return BandCounts(
        codes=tuple(codes),
        counts=counts,
        all_properties=total,
        suppressed_cells=suppressed,
        index={code: i for i, code in enumerate(codes)},
    )


def classify_type_columns(
    columns: Sequence[str], rules: Sequence[tuple[str, str]] = VOA_TYPE_GROUPS
) -> dict[str, str]:
    """Each CTSOP 3.1 property-type column's group; an unmatched one is refused.

    The key columns, ``band`` and ``all_properties`` are left out.
    """
    compiled = [(re.compile(pattern), group) for pattern, group in rules]
    out: dict[str, str] = {}
    unmatched = []
    for column in columns:
        if column in (*_KEY_COLUMNS, "band", "all_properties"):
            continue
        group = next((g for pattern, g in compiled if pattern.match(column)), None)
        if group is None:
            unmatched.append(column)
        else:
            out[column] = group
    if unmatched:
        raise ValueError(f"CTSOP 3.1 columns match no type group: {unmatched}")
    if not {"D", "S", "T", "F"} <= set(out.values()):
        raise ValueError("CTSOP 3.1 lacks a column for one of the four types.")
    return out


def _mix(counts: np.ndarray, *fallbacks: np.ndarray) -> np.ndarray:
    """Shares of ``counts`` along the last axis, else of each fallback in turn.

    Each fallback broadcasts against ``counts``; where none has a positive
    total, the shares are equal.
    """
    out = np.zeros(counts.shape)
    pending = np.ones(counts.shape[:-1], dtype=bool)
    for candidate in (counts, *fallbacks):
        candidate = np.broadcast_to(candidate, counts.shape)
        total = candidate.sum(axis=-1)
        usable = pending & (total > 0)
        out[usable] = candidate[usable] / total[usable][..., None]
        pending &= ~usable
    out[pending] = 1.0 / counts.shape[-1]
    return out


def _allocate(
    counts: np.ndarray,
    house_fallbacks: Sequence[np.ndarray],
    type_fallbacks: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Bungalows into the house types, then other types into all four.

    ``counts`` ends in the six kinds (D, S, T, F, bungalow, other). Returns
    the four types before and after the other types are spread.
    """
    houses = counts[..., :3]
    house_mix = _mix(houses, *house_fallbacks)
    typed = np.concatenate(
        [houses + counts[..., 4:5] * house_mix, counts[..., 3:4]], axis=-1
    )
    final = typed + counts[..., 5:6] * _mix(typed, *type_fallbacks)
    return typed, final


def _grouped_type_counts(
    frame: pd.DataFrame, groups: Mapping[str, str], *, suppressed_value: float
) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    """Kind counts per area × label, checked against ``all_properties``."""
    area = frame["ecode"].astype(str).to_numpy()
    band = frame["band"].astype(str).str.strip().to_numpy()
    if len(set(zip(area, band, strict=True))) != len(frame):
        raise ValueError("CTSOP 3.1 repeats an area and band.")
    unknown = sorted(set(band) - set(_LABELS))
    if unknown:
        raise ValueError(f"CTSOP 3.1 has band labels {unknown}.")
    codes = sorted(set(area))
    where = {code: i for i, code in enumerate(codes)}
    a = np.array([where[code] for code in area], dtype=np.int64)
    b = np.array([_LABELS.index(value) for value in band], dtype=np.int64)
    shape = (len(codes), len(_LABELS))
    counts = np.zeros((*shape, len(_KINDS)))
    nonzero = np.zeros(shape)
    imputed = np.zeros(shape)
    for column, group in groups.items():
        values, flags = parse_counts(frame[column], suppressed_value=suppressed_value)
        if group == "total":
            continue
        np.add.at(counts, (a, b, _KINDS.index(group)), values)
        np.add.at(nonzero, (a, b), (values > 0) & ~flags)
        np.add.at(imputed, (a, b), flags)
    printed = np.zeros(shape)
    printed[a, b], _ = parse_counts(
        frame["all_properties"], suppressed_value=suppressed_value
    )
    # Each printed count is within 5 of its true value, an imputed [c] within
    # 1.5 and the printed total within 5; a printed zero is exact.
    tolerance = 5.0 * nonzero + 1.5 * imputed + 5.0
    present = np.zeros(shape, dtype=bool)
    present[a, b] = True
    outside = present & (np.abs(counts.sum(axis=2) - printed) > tolerance)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} CTSOP 3.1 rows do not add up to all_properties "
            "within rounding: a type column is missing or counted twice."
        )
    return codes, counts, printed, int(imputed.sum())


@dataclass(frozen=True)
class TypeShares:
    """Shares of Price Paid's four types within each area × band (CTSOP 3.1).

    ``shares`` is areas × bands × (D, S, T, F) and ``totals`` areas × bands,
    3.1's own all-properties count. ``receipt`` counts the imputed cells and
    the rows that fell back to a wider mix.
    """

    codes: tuple[str, ...]
    shares: np.ndarray
    totals: np.ndarray
    receipt: dict[str, object]
    index: Mapping[str, int] = field(repr=False, default_factory=dict)


def type_shares(
    table: pd.DataFrame,
    *,
    geography: str,
    suppressed_value: float,
    parent_of: Mapping[str, str] | None = None,
    parent_geography: str = "LAUA",
    rules: Sequence[tuple[str, str]] = VOA_TYPE_GROUPS,
) -> TypeShares:
    """Type shares within each area × band, from CTSOP 3.1's type columns.

    Bungalows join the three house types in the row's house mix, else the
    area's (its "All" row), else the parent's (``parent_of`` maps an area to
    its ``parent_geography`` row), else in equal thirds. Annexes, caravans
    and unknown types then spread over all four types in the same order of
    mixes. A band the area has no property in takes the area's shares, then
    the parent's. Every row's columns must add up to its ``all_properties``
    within VOA's rounding.
    """
    if "band" not in table.columns or "all_properties" not in table.columns:
        raise ValueError("CTSOP 3.1 lacks the band or all_properties column.")
    groups = classify_type_columns(list(table.columns), rules)
    codes, counts, printed, imputed = _grouped_type_counts(
        _rows(table, geography), groups, suppressed_value=suppressed_value
    )
    parent = np.zeros((len(codes), len(_KINDS)))
    unparented = len(codes)
    if parent_of is not None:
        parent_codes, parent_counts, _, _ = _grouped_type_counts(
            _rows(table, parent_geography), groups, suppressed_value=suppressed_value
        )
        where = {code: i for i, code in enumerate(parent_codes)}
        rows = np.array(
            [where.get(parent_of.get(code, ""), -1) for code in codes], dtype=np.int64
        )
        found = rows >= 0
        parent[found] = parent_counts[rows[found], 0]
        unparented = int((~found).sum())
    parent_typed, parent_final = _allocate(parent, (), ())
    area = counts[:, 0]
    area_typed, area_final = _allocate(area, (parent[:, :3],), (parent_typed,))
    bands = counts[:, 1:]
    _, final = _allocate(
        bands,
        (area[:, None, :3], parent[:, None, :3]),
        (area_typed[:, None], parent_typed[:, None]),
    )
    shares = _mix(final, area_final[:, None], parent_final[:, None])
    receipt = {
        "areas": len(codes),
        "geography": geography,
        "imputed_suppressed_cells": imputed,
        "area_bands_without_stock": int((final.sum(axis=-1) <= 0).sum()),
        "areas_without_parent_row": unparented if parent_of is not None else None,
        "type_columns": dict(sorted(groups.items())),
    }
    return TypeShares(
        codes=tuple(codes),
        shares=shares,
        totals=printed[:, 1:],
        receipt=receipt,
        index={code: i for i, code in enumerate(codes)},
    )


def classify_build_periods(
    columns: Sequence[str],
) -> dict[str, tuple[int | None, int | None]]:
    """Each CTSOP 4.1 build-period column's first and last year.

    ``bp_pre_1900`` has no first year, ``bp_unkw`` neither year. A column
    that is neither a key, ``band``, ``all_properties`` nor a build period
    is refused.
    """
    out: dict[str, tuple[int | None, int | None]] = {}
    unknown = []
    for column in columns:
        if column in (*_KEY_COLUMNS, "band", "all_properties"):
            continue
        match = _BUILD_PERIOD.match(column)
        if match is None:
            unknown.append(column)
        elif match["before"]:
            out[column] = (None, int(match["before"]) - 1)
        elif match["first"]:
            out[column] = (int(match["first"]), int(match["last"]))
        elif match["year"]:
            out[column] = (int(match["year"]), int(match["year"]))
        else:
            out[column] = (None, None)
    if unknown:
        raise ValueError(f"CTSOP 4.1 columns are not build periods: {unknown}")
    return out


@dataclass(frozen=True)
class BuildPeriods:
    """Properties by band and build period per area (CTSOP 4.1)."""

    codes: tuple[str, ...]
    periods: tuple[str, ...]
    spans: tuple[tuple[int | None, int | None], ...]
    counts: np.ndarray
    receipt: dict[str, object]
    index: Mapping[str, int] = field(repr=False, default_factory=dict)


def build_periods(
    table: pd.DataFrame, *, geography: str, suppressed_value: float
) -> BuildPeriods:
    """Each area × band's count by period; ``counts`` is areas × bands × periods."""
    spans = classify_build_periods(list(table.columns))
    rows = _rows(table, geography)
    rows = rows[rows["band"].astype(str).str.strip().isin(BANDS)]
    codes = sorted(set(rows["ecode"].astype(str)))
    where = {code: i for i, code in enumerate(codes)}
    periods = tuple(spans)
    counts = np.zeros((len(codes), len(BANDS), len(periods)))
    a = np.array([where[code] for code in rows["ecode"].astype(str)], dtype=np.int64)
    b = np.array(
        [BANDS.index(value) for value in rows["band"].astype(str).str.strip()],
        dtype=np.int64,
    )
    if len(set(zip(a.tolist(), b.tolist(), strict=True))) != len(rows):
        raise ValueError("CTSOP 4.1 repeats an area and band.")
    imputed = 0
    for k, column in enumerate(periods):
        values, flags = parse_counts(rows[column], suppressed_value=suppressed_value)
        counts[a, b, k] = values
        imputed += int(flags.sum())
    return BuildPeriods(
        codes=tuple(codes),
        periods=periods,
        spans=tuple(spans[p] for p in periods),
        counts=counts,
        receipt={
            "areas": len(codes),
            "geography": geography,
            "periods": len(periods),
            "imputed_suppressed_cells": imputed,
        },
        index=where,
    )
