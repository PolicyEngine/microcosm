"""Region and legal status for synthetic UK firms, from ONS UK Business 2025.

Two published ONS tables give the firm attributes policyengine-uk's Firm
entity needs but the firm generator does not draw:

- Table 8 counts enterprises by SIC division, region and the seven ONS
  turnover bands; it gives ``ons_region_code``.
- Table 13 counts enterprises by broad industry group, legal status and its
  own ten turnover bands; it gives ``ons_legal_status`` (ONS codes 1 to 7).
- Table 12 counts enterprises by the same industry groups, legal status and
  seven employment sizebands; when present it conditions ``ons_legal_status``
  on the firm's ONS ``employment`` as well as its turnover.

:func:`assign_uk_firm_attributes` draws both after calibration, so it never
changes a weight or an existing column. Region is drawn conditional on the
firm's (SIC division, Table 8 band) cell and legal status conditional on its
(industry group, Table 13 band) cell. The two draws are independent given
those cells: no published table crosses region with legal status by industry,
so any dependence beyond what sector and turnover carry is not modelled
(Table 15, region by legal status by turnover, is held out to check it).

No published table crosses legal status with both turnover and employment, so
the legal status weights combine the two tables under an assumption: within an
industry group and legal status, a firm's employment band is independent of
its turnover band. The firm's Table 13 distribution (cell, row or UK total, as
below) is multiplied, status by status, by Table 12's share of that status's
enterprises in the group that fall in the firm's employment band,
``P(employment band | group, status)``. Without that factor, a sole
proprietor's draw ignored headcount and firms of any size became sole
proprietors at their turnover cell's rate. When every product is zero (no
status in the turnover distribution has enterprises in that employment band)
the factor is dropped and the Table 13 distribution alone is used; a division
without a Table 12 group uses Table 12's UK total over groups. Employment
bands are parsed from Table 12's labels like turnover bands ("0-4" is
``[0, 5)``).

Band edges and industry groups are parsed from the tables' own labels. A band
label ``"lo-hi"`` in thousands of pounds is the interval ``[lo, hi + 1)`` and
``"lo+"`` is open above, as in the firm generator's ONS bands, where "50-99"
is ``[50, 100)``. An industry-group key ``"01-03"`` covers SIC divisions 1 to
3 inclusive and ``"45"`` covers division 45 alone.

Fallback (an assumption; see ``docs/firm-support-plan.md``): when a firm's
cell holds no enterprises, or its division or group is absent from the table,
the draw uses the row total of its division or group over every band, then the
UK total over every division and band.
Out-of-frame firms (``in_frame`` False, or zero or negative turnover: the
HMRC Negative_or_Zero traders appended before calibration) have no ONS band,
and were allocated to sectors by the sector shares of the frame, so they draw
from their division's or group's row total, then the UK total; they never use
a band cell. :func:`uk_firm_attribute_fallback_counts` reports which level
each firm used.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

__all__ = [
    "UK_FIRM_ATTRIBUTE_FALLBACK_LEVELS",
    "UK_FIRM_LEGAL_STATUS_EMPLOYMENT_FILE",
    "UK_FIRM_LEGAL_STATUS_TURNOVER_FILE",
    "UK_FIRM_REGION_TURNOVER_FILE",
    "UKFirmAttributeSourceData",
    "assign_uk_firm_attributes",
    "industry_group_divisions",
    "parse_turnover_band_lowers",
    "read_uk_firm_attribute_source_data",
    "turnover_band_index",
    "uk_firm_attribute_fallback_counts",
]

UK_FIRM_REGION_TURNOVER_FILE = "ons_region_turnover.csv"
UK_FIRM_LEGAL_STATUS_TURNOVER_FILE = "ons_legal_status_turnover.csv"
UK_FIRM_LEGAL_STATUS_EMPLOYMENT_FILE = "ons_legal_status_employment.csv"

#: The distribution each firm's draw used, most specific first.
UK_FIRM_ATTRIBUTE_FALLBACK_LEVELS: tuple[str, ...] = ("cell", "row", "uk")
_CELL, _ROW, _UK = range(len(UK_FIRM_ATTRIBUTE_FALLBACK_LEVELS))

_REGION_COLUMNS = ("sic_division", "region_code", "turnover_band", "enterprises")
_LEGAL_STATUS_COLUMNS = (
    "industry_group",
    "legal_status",
    "turnover_band",
    "enterprises",
)
_LEGAL_STATUS_EMPLOYMENT_COLUMNS = (
    "industry_group",
    "legal_status",
    "employment_band",
    "enterprises",
)
_CLOSED_BAND = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_OPEN_BAND = re.compile(r"^\s*(\d+)\s*\+\s*$")
_GROUP_KEY = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


def parse_turnover_band_lowers(labels: object) -> dict[str, float]:
    """Each band label's inclusive lower edge in thousands of pounds.

    Labels are ``"lo-hi"`` (the interval ``[lo, hi + 1)``) or one open top band
    ``"lo+"``. The bands must start at zero and be contiguous; anything else,
    a ``"Total"`` column included, raises. Returned in ascending order.
    """

    closed: list[tuple[float, float, str]] = []
    open_bands: list[tuple[float, str]] = []
    for label in dict.fromkeys(str(label) for label in labels):
        if match := _CLOSED_BAND.match(label):
            lower, upper = float(match.group(1)), float(match.group(2))
            if upper < lower:
                raise ValueError(f"turnover band {label!r} is reversed")
            closed.append((lower, upper, label))
        elif match := _OPEN_BAND.match(label):
            open_bands.append((float(match.group(1)), label))
        else:
            raise ValueError(f"unparseable turnover band label {label!r}")
    if len(open_bands) != 1:
        raise ValueError(
            f"expected exactly one open turnover band, got {len(open_bands)}"
        )
    closed.sort()
    expected = 0.0
    lowers: dict[str, float] = {}
    for lower, upper, label in closed:
        if lower != expected:
            raise ValueError(
                f"turnover band {label!r} starts at {lower:g}, expected {expected:g}"
            )
        lowers[label] = lower
        expected = upper + 1.0
    open_lower, open_label = open_bands[0]
    if open_lower != expected:
        raise ValueError(
            f"open turnover band {open_label!r} starts at {open_lower:g}, "
            f"expected {expected:g}"
        )
    lowers[open_label] = open_lower
    return lowers


def turnover_band_index(
    turnover_k: np.ndarray, band_lowers: Mapping[str, float]
) -> np.ndarray:
    """Index into ``band_lowers`` (ascending) of each positive turnover.

    Uses lower edges only, so a turnover between one band's published upper
    label and the next band's lower edge (99.5 between "50-99" and "100-249")
    falls in the lower band. Callers route non-positive turnover elsewhere.
    """

    lowers = np.fromiter(band_lowers.values(), dtype=np.float64)
    return (
        np.searchsorted(lowers, np.asarray(turnover_k, dtype=np.float64), "right") - 1
    )


def industry_group_divisions(keys: object) -> dict[int, str]:
    """Map each SIC division to the industry-group key that covers it.

    A key ``"lo-hi"`` covers divisions ``lo`` to ``hi`` inclusive and a key
    ``"nn"`` covers one division. Overlapping or unparseable keys raise.
    """

    division_group: dict[int, str] = {}
    for key in dict.fromkeys(str(key) for key in keys):
        match = _GROUP_KEY.match(key)
        if match is None:
            raise ValueError(f"unparseable industry group key {key!r}")
        lower = int(match.group(1))
        upper = int(match.group(2)) if match.group(2) is not None else lower
        if upper < lower:
            raise ValueError(f"industry group key {key!r} is reversed")
        for division in range(lower, upper + 1):
            if division in division_group:
                raise ValueError(
                    f"SIC division {division} is in industry groups "
                    f"{division_group[division]!r} and {key!r}"
                )
            division_group[division] = key
    return division_group


@dataclass(frozen=True)
class _CellTable:
    """Enterprise counts as ``counts[row, band, category]``."""

    row_index: Mapping[int, int]
    band_lowers: Mapping[str, float]
    categories: tuple[object, ...]
    counts: np.ndarray


def _cell_table(
    frame: pd.DataFrame,
    *,
    row_key: pd.Series,
    category: pd.Series,
    band_column: str = "turnover_band",
) -> _CellTable:
    band_lowers = parse_turnover_band_lowers(frame[band_column])
    band_position = {label: i for i, label in enumerate(band_lowers)}
    row_values = sorted(set(row_key))
    row_index = {value: i for i, value in enumerate(row_values)}
    categories = tuple(sorted(set(category)))
    category_index = {value: i for i, value in enumerate(categories)}
    enterprises = pd.to_numeric(frame["enterprises"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(enterprises).all() or (enterprises < 0).any():
        raise ValueError("enterprise counts must be finite and non-negative")
    counts = np.zeros(
        (len(row_values), len(band_lowers), len(categories)), dtype=np.float64
    )
    np.add.at(
        counts,
        (
            row_key.map(row_index).to_numpy(dtype=np.int64),
            frame[band_column].map(band_position).to_numpy(dtype=np.int64),
            category.map(category_index).to_numpy(dtype=np.int64),
        ),
        enterprises,
    )
    if counts.sum() <= 0:
        raise ValueError("an ONS attribute table holds no enterprises")
    return _CellTable(
        row_index=MappingProxyType(row_index),
        band_lowers=MappingProxyType(band_lowers),
        categories=categories,
        counts=counts,
    )


def _group_status_table(frame: pd.DataFrame, band_column: str) -> _CellTable:
    groups = frame["industry_group"].astype(str).str.strip()
    statuses = frame["legal_status"].astype(str).str.strip().astype(int)
    if not statuses.between(1, 7).all():
        raise ValueError("ONS legal status codes run from 1 to 7")
    # Row ids are the industry groups' positions in key order; the division
    # map sends each division to its group's position.
    group_keys = sorted(set(groups))
    industry_group_divisions(group_keys)
    position = {key: i for i, key in enumerate(group_keys)}
    return _cell_table(
        frame,
        row_key=groups.map(position),
        category=statuses,
        band_column=band_column,
    )


@dataclass(frozen=True)
class UKFirmAttributeSourceData:
    """The ONS UK Business 2025 tables the attribute draws read.

    ``region_turnover`` is Table 8 in long form (``sic_division``,
    ``region_code``, ``turnover_band``, ``enterprises``) and
    ``legal_status_turnover`` is Table 13 (``industry_group``,
    ``legal_status``, ``turnover_band``, ``enterprises``). The optional
    ``legal_status_employment`` is Table 12 (``industry_group``,
    ``legal_status``, ``employment_band``, ``enterprises``); without it legal
    status is conditioned on turnover alone. Keys are strings, except
    ``legal_status`` (int) and ``enterprises`` (float).
    """

    region_turnover: pd.DataFrame
    legal_status_turnover: pd.DataFrame
    legal_status_employment: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        tables = [
            ("region_turnover", self.region_turnover, _REGION_COLUMNS),
            (
                "legal_status_turnover",
                self.legal_status_turnover,
                _LEGAL_STATUS_COLUMNS,
            ),
        ]
        if self.legal_status_employment is not None:
            tables.append(
                (
                    "legal_status_employment",
                    self.legal_status_employment,
                    _LEGAL_STATUS_EMPLOYMENT_COLUMNS,
                )
            )
        for name, frame, columns in tables:
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise ValueError(f"{name} is missing columns {missing}")
        # Parse once here so a malformed table fails on read, not on draw.
        self._region_table()
        status_table = self._legal_status_table()
        if self.legal_status_employment is not None:
            employment_table = self._legal_status_employment_table()
            if employment_table.categories != status_table.categories:
                raise ValueError(
                    "legal_status_employment and legal_status_turnover hold "
                    "different legal status codes"
                )

    def _region_table(self) -> _CellTable:
        frame = self.region_turnover
        return _cell_table(
            frame,
            row_key=frame["sic_division"].astype(str).str.strip().astype(int),
            category=frame["region_code"].astype(str).str.strip(),
        )

    def _legal_status_table(self) -> _CellTable:
        return _group_status_table(self.legal_status_turnover, "turnover_band")

    def _legal_status_employment_table(self) -> _CellTable:
        if self.legal_status_employment is None:
            raise ValueError("no legal_status_employment table")
        return _group_status_table(self.legal_status_employment, "employment_band")

    def _division_group_rows(self, frame: pd.DataFrame | None = None) -> dict[int, int]:
        if frame is None:
            frame = self.legal_status_turnover
        groups = sorted(set(frame["industry_group"].astype(str).str.strip()))
        position = {key: i for i, key in enumerate(groups)}
        return {
            division: position[key]
            for division, key in industry_group_divisions(groups).items()
        }


def read_uk_firm_attribute_source_data(
    directory: str | Path,
) -> UKFirmAttributeSourceData:
    """Read Tables 8, 13 and, when present, 12 from ``directory``.

    ``directory`` holds the fixture CSVs. Table 12
    (:data:`UK_FIRM_LEGAL_STATUS_EMPLOYMENT_FILE`) is optional; without it
    legal status is drawn on turnover alone.
    """

    directory = Path(directory)
    employment_path = directory / UK_FIRM_LEGAL_STATUS_EMPLOYMENT_FILE
    legal_status_employment = None
    if employment_path.is_file():
        legal_status_employment = pd.read_csv(
            employment_path,
            dtype={
                "industry_group": str,
                "legal_status": int,
                "employment_band": str,
                "enterprises": float,
            },
            keep_default_na=False,
        )
    return UKFirmAttributeSourceData(
        region_turnover=pd.read_csv(
            directory / UK_FIRM_REGION_TURNOVER_FILE,
            dtype={
                "sic_division": str,
                "region_code": str,
                "turnover_band": str,
                "enterprises": float,
            },
            keep_default_na=False,
        ),
        legal_status_turnover=pd.read_csv(
            directory / UK_FIRM_LEGAL_STATUS_TURNOVER_FILE,
            dtype={
                "industry_group": str,
                "legal_status": int,
                "turnover_band": str,
                "enterprises": float,
            },
            keep_default_na=False,
        ),
        legal_status_employment=legal_status_employment,
    )


def _firm_inputs(firms: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(division, turnover_k, banded)`` per firm; banded firms are in frame."""

    for column in ("sic_code", "annual_turnover_k"):
        if column not in firms.columns:
            raise ValueError(f"firms frame is missing {column!r}")
    division = firms["sic_code"].astype(str).str.strip().astype(int).to_numpy()
    turnover = firms["annual_turnover_k"].to_numpy(dtype=np.float64)
    if np.isnan(turnover).any():
        raise ValueError("annual_turnover_k holds NaN")
    banded = turnover > 0
    if "in_frame" in firms.columns:
        banded &= firms["in_frame"].to_numpy(dtype=bool)
    return division, turnover, banded


def _resolve(
    table: _CellTable,
    rows: np.ndarray,
    turnover: np.ndarray,
    banded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Each firm's distribution id and fallback level.

    ``rows`` holds the table row of each firm, or -1 when the table has none.
    Distribution ids index :func:`_distributions`: cells first
    (``row * n_bands + band``), then row totals, then the UK total.
    """

    n_rows, n_bands, _ = table.counts.shape
    cell_total = table.counts.sum(axis=2)
    row_total = cell_total.sum(axis=1)
    has_row = rows >= 0
    safe_row = np.where(has_row, rows, 0)
    band = np.zeros(len(rows), dtype=np.int64)
    band[banded] = turnover_band_index(turnover[banded], table.band_lowers)
    cell_ok = banded & has_row & (cell_total[safe_row, band] > 0)
    row_ok = ~cell_ok & has_row & (row_total[safe_row] > 0)
    level = np.full(len(rows), _UK, dtype=np.int64)
    level[row_ok] = _ROW
    level[cell_ok] = _CELL
    distribution = np.full(len(rows), n_rows * n_bands + n_rows, dtype=np.int64)
    distribution[row_ok] = n_rows * n_bands + safe_row[row_ok]
    distribution[cell_ok] = safe_row[cell_ok] * n_bands + band[cell_ok]
    return distribution, level


def _distributions(table: _CellTable) -> np.ndarray:
    """Counts per distribution id: cells, then row totals, then the UK total."""

    n_rows, n_bands, n_categories = table.counts.shape
    return np.concatenate(
        [
            table.counts.reshape(n_rows * n_bands, n_categories),
            table.counts.sum(axis=1),
            table.counts.sum(axis=(0, 1))[None, :],
        ]
    )


def _draw(
    rng: np.random.Generator,
    distribution: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """One categorical draw per distribution, in ascending id order."""

    category = np.empty(len(distribution), dtype=np.int64)
    order = np.argsort(distribution, kind="stable")
    ids, starts = np.unique(distribution[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    for distribution_id, start, end in zip(ids, starts, ends, strict=True):
        weights = counts[distribution_id]
        category[order[start:end]] = rng.choice(
            len(weights), size=end - start, p=weights / weights.sum()
        )
    return category


def _map_rows(lookup: Mapping[int, int], division: np.ndarray) -> np.ndarray:
    """Table row of each firm's division, or -1, mapped once per division."""

    unique, inverse = np.unique(division, return_inverse=True)
    rows = np.array([lookup.get(int(d), -1) for d in unique], dtype=np.int64)
    return rows[inverse]


def _region_rows(
    source: UKFirmAttributeSourceData, division: np.ndarray
) -> tuple[_CellTable, np.ndarray]:
    table = source._region_table()
    return table, _map_rows(table.row_index, division)


def _legal_status_rows(
    source: UKFirmAttributeSourceData, division: np.ndarray
) -> tuple[_CellTable, np.ndarray]:
    table = source._legal_status_table()
    return table, _map_rows(source._division_group_rows(), division)


def _employment(firms: pd.DataFrame) -> np.ndarray:
    if "employment" not in firms.columns:
        raise ValueError("firms frame is missing 'employment'")
    employment = firms["employment"].to_numpy(dtype=np.float64)
    if np.isnan(employment).any():
        raise ValueError("employment holds NaN")
    if (employment < 0).any():
        raise ValueError("employment must be non-negative")
    return employment


def _employment_weighting(
    source: UKFirmAttributeSourceData,
    division: np.ndarray,
    employment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Each firm's employment key and ``P(band | group, status)`` per key.

    Keys index the rows of the returned likelihood: ``row * n_bands + band``,
    where row is the firm's Table 12 group, or the extra UK row after the
    groups when its division has none.
    """

    table = source._legal_status_employment_table()
    n_rows, n_bands, _ = table.counts.shape
    counts = np.concatenate([table.counts, table.counts.sum(axis=0)[None]], axis=0)
    status_total = counts.sum(axis=1, keepdims=True)
    likelihood = np.divide(
        counts,
        status_total,
        out=np.zeros_like(counts),
        where=status_total > 0,
    )
    rows = _map_rows(
        source._division_group_rows(source.legal_status_employment), division
    )
    rows = np.where(rows >= 0, rows, n_rows)
    band = turnover_band_index(employment, table.band_lowers)
    return rows * n_bands + band, likelihood.reshape((n_rows + 1) * n_bands, -1)


def _legal_status_distributions(
    source: UKFirmAttributeSourceData,
    division: np.ndarray,
    turnover: np.ndarray,
    banded: np.ndarray,
    employment: np.ndarray | None,
) -> tuple[_CellTable, np.ndarray, np.ndarray, np.ndarray]:
    """``(table, distribution, counts, employment_dropped)`` for legal status.

    Without Table 12 (or ``employment``) these are the Table 13 distributions.
    With it, each distribution is a (Table 13 distribution, employment key)
    pair, numbered in ascending pair order, whose weights are the Table 13
    counts times the employment likelihood, or the Table 13 counts alone when
    that product is all zero (``employment_dropped`` is True for those firms).
    """

    table, rows = _legal_status_rows(source, division)
    distribution, _ = _resolve(table, rows, turnover, banded)
    counts = _distributions(table)
    dropped = np.zeros(len(distribution), dtype=bool)
    if source.legal_status_employment is None or employment is None:
        return table, distribution, counts, dropped
    key, likelihood = _employment_weighting(source, division, employment)
    pair = distribution * len(likelihood) + key
    unique, inverse = np.unique(pair, return_inverse=True)
    weights = counts[unique // len(likelihood)] * likelihood[unique % len(likelihood)]
    empty = weights.sum(axis=1) <= 0
    weights[empty] = counts[unique[empty] // len(likelihood)]
    return table, inverse.reshape(-1), weights, empty[inverse.reshape(-1)]


def assign_uk_firm_attributes(
    firms: pd.DataFrame,
    source: UKFirmAttributeSourceData,
    *,
    seed: int,
) -> pd.DataFrame:
    """Return a copy of ``firms`` with ``ons_region_code`` and ``ons_legal_status``.

    Run after calibration: weights and every existing column are unchanged.
    One generator, ``default_rng(seed + 2)``, draws region first and legal
    status second, each with one categorical draw per distribution (cell, row
    total or UK total; for legal status, paired with the firm's employment
    band when Table 12 is present) in ascending distribution order, so the
    result depends only on the frame, the tables and the seed. See the module
    docstring for the conditioning, the independence assumptions and the
    fallback. With Table 12, ``firms`` needs a non-negative ``employment``.
    """

    division, turnover, banded = _firm_inputs(firms)
    employment = (
        _employment(firms) if source.legal_status_employment is not None else None
    )
    rng = np.random.default_rng(seed + 2)
    region_table, region_rows = _region_rows(source, division)
    region_distribution, _ = _resolve(region_table, region_rows, turnover, banded)
    region = _draw(rng, region_distribution, _distributions(region_table))
    status_table, status_distribution, status_counts, _ = _legal_status_distributions(
        source, division, turnover, banded, employment
    )
    status = _draw(rng, status_distribution, status_counts)

    result = firms.copy()
    result["ons_region_code"] = pd.Series(
        np.asarray(region_table.categories, dtype=object)[region],
        index=firms.index,
        dtype=object,
    )
    result["ons_legal_status"] = pd.Series(
        np.asarray(status_table.categories, dtype=np.int64)[status],
        index=firms.index,
        dtype=np.int64,
    )
    return result


def uk_firm_attribute_fallback_counts(
    firms: pd.DataFrame,
    source: UKFirmAttributeSourceData,
) -> pd.DataFrame:
    """Firms and base weight per attribute and fallback level.

    One row per (``attribute``, ``level``) with ``firms`` (rows) and
    ``weight`` (sum of ``base_weight``, or of ones when the frame has none).
    ``level`` is one of :data:`UK_FIRM_ATTRIBUTE_FALLBACK_LEVELS`. The levels
    are those :func:`assign_uk_firm_attributes` uses; no random draw is made.
    """

    division, turnover, banded = _firm_inputs(firms)
    weight = (
        firms["base_weight"].to_numpy(dtype=np.float64)
        if "base_weight" in firms.columns
        else np.ones(len(firms), dtype=np.float64)
    )
    n_levels = len(UK_FIRM_ATTRIBUTE_FALLBACK_LEVELS)
    records = []
    for attribute, (table, rows) in (
        ("ons_region_code", _region_rows(source, division)),
        ("ons_legal_status", _legal_status_rows(source, division)),
    ):
        _, level = _resolve(table, rows, turnover, banded)
        firm_counts = np.bincount(level, minlength=n_levels)
        weights = np.bincount(level, weights=weight, minlength=n_levels)
        for index, name in enumerate(UK_FIRM_ATTRIBUTE_FALLBACK_LEVELS):
            records.append(
                {
                    "attribute": attribute,
                    "level": name,
                    "firms": int(firm_counts[index]),
                    "weight": float(weights[index]),
                }
            )
    return pd.DataFrame.from_records(
        records, columns=["attribute", "level", "firms", "weight"]
    )
