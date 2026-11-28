"""UK firm employees paid at or below the secondary threshold, from ASHE.

policyengine-uk's ``firm_class_1_earnings_above_secondary_threshold`` removes
the employees supplied in ``firm_employees_below_secondary_threshold`` (int)
and their pay in ``firm_paybill_below_secondary_threshold`` (float) before its
average-pay approximation. Left at 0, every employee's unused threshold is
netted against others' pay, which understates the employer NIC base of firms
with low-paid staff. This module imputes the two inputs from one published
table: ONS ASHE Table 4.7a (*Annual pay - Gross*), 2025 revised edition,
sheet ``All``, tax year 2024-25, the same table ``firm_paybill`` uses. The
extracted CSV and its provenance sit in
``tests/fixtures/uk/firm_pay_distribution_2024/``
(``tools/extract_uk_firm_pay_distribution_table.py`` re-creates it).

Method, all of it assumption beyond the source (status ``assumed``):

- a section's annual pay quantile function is piecewise linear through
  (0, 0 pounds) and its published percentiles (10, 20, 25, 30, 40, 50, 60,
  70, 75, 80, 90). A percentile ONS marks ``x`` or ``:`` is skipped, so the
  line runs between its published neighbours (section A has no p10 in 2025,
  so its line runs from 0 pounds at p0 to p20). The straight line from 0
  pounds at p0 to the first published percentile is a guess: ASHE publishes
  nothing below p10. In most sections the 2024-25 threshold (9,100 pounds)
  is below p10, so the share is 10% x 9,100 / p10 and the mean pay below is
  exactly 4,550 pounds: both rest wholly on that guess;
- the share of a section's jobs paid at or below the threshold is the
  percentile at which that line reaches the threshold, and their mean pay is
  the line's average below that percentile;
- a section with no published percentiles (section U in 2025), one whose
  highest published percentile is below the threshold, and ``UNKNOWN`` take
  the ``ALL EMPLOYEES`` row, as ``firm_paybill`` does;
- every firm in a section has that section's share, whatever its size. The
  section total is unbiased, but a micro firm whose draw puts every employee
  below the threshold has the rest of its paybill charged with no threshold
  at all in policyengine-uk, which makes the split lumpy per firm (it matters
  for the Employment Allowance);
- the head count is an integer (policyengine-uk ``value_type = int``):
  ``floor(employees x share)`` plus one with probability equal to the
  remainder, drawn from a fixed-seed generator of its own
  (``UK_FIRM_PAY_DISTRIBUTION_SEED``), so the weighted total stays close to
  the expected value and the firm generator's stream is untouched;
- the paybill is that count times the section's mean pay below the
  threshold, capped at the firm's ``firm_paybill`` where the frame has one;
- ASHE annual pay covers employees on adult rates in the same job for more
  than a year, so short-tenure and youth-rate jobs, which are lower paid,
  are left out and the share leans low; annual pay at or below the annual
  threshold is not the same as no secondary contributions, which are
  assessed per earnings period.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_SIC_SECTION_UNKNOWN,
    UK_FIRM_SIC_SECTIONS,
    uk_firm_sic_sections,
)

#: The extracted ASHE table's file name in a pay distribution source directory.
UK_FIRM_PAY_DISTRIBUTION_FILE = "ashe_section_annual_pay_percentiles.csv"
#: Percentiles the table carries, in order (``p50`` is the sheet's median).
UK_FIRM_PAY_DISTRIBUTION_PERCENTILES = (10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90)
#: Its columns, in order.
UK_FIRM_PAY_DISTRIBUTION_SOURCE_COLUMNS = (
    "sic_section",
    "description",
    "jobs_thousand",
    "mean",
    *(f"p{p}" for p in UK_FIRM_PAY_DISTRIBUTION_PERCENTILES),
)
#: ``sic_section`` of the table's ``ALL EMPLOYEES`` row.
UK_FIRM_PAY_DISTRIBUTION_ALL_EMPLOYEES = "ALL"
#: ONS markers for a cell it does not publish.
UK_FIRM_PAY_DISTRIBUTION_MARKERS = ("x", ":")
#: The tax year the table describes (ASHE 2025 is the year to 5 April 2025).
UK_FIRM_PAY_DISTRIBUTION_TAX_YEAR = "2024-25"
UK_FIRM_PAY_DISTRIBUTION_STATUS = "assumed"
UK_FIRM_PAY_DISTRIBUTION_METHOD = (
    "share of section jobs at or below the secondary threshold from a "
    "piecewise-linear ONS ASHE 2025 Table 4.7a percentile curve (0 pounds at "
    "p0), times firm_employees, rounded by a fixed-seed draw; paybill is the "
    "count times the section's mean pay below the threshold"
)
#: Seed of the rounding generator; not the firm generator's seed.
UK_FIRM_PAY_DISTRIBUTION_SEED = 20242025

_SECTION_LETTERS = tuple(
    section
    for section in UK_FIRM_SIC_SECTIONS
    if section != UK_FIRM_SIC_SECTION_UNKNOWN
)


@dataclass(frozen=True)
class UKFirmPayDistributionSource:
    """Published annual gross pay percentiles (pounds) per SIC section.

    ``section_percentiles`` maps a section, or ``ALL``, to its published
    ``{percentile: pay}`` (unpublished cells left out; a section with none
    has an empty mapping). ``sha256`` is the CSV's hash.
    """

    section_percentiles: Mapping[str, Mapping[int, float]]
    file: str
    sha256: str


def read_uk_firm_pay_distribution_source(
    path: str | Path,
) -> UKFirmPayDistributionSource:
    """Read the ASHE percentile table from a directory or the CSV itself.

    Raises ``ValueError`` unless the file has exactly
    ``UK_FIRM_PAY_DISTRIBUTION_SOURCE_COLUMNS``, sections A-U and ``ALL``
    once each, every percentile cell a positive number or an ONS marker,
    published percentiles that do not fall as the percentile rises, and a
    fully published ``ALL`` row.
    """

    path = Path(path)
    if path.is_dir():
        path = path / UK_FIRM_PAY_DISTRIBUTION_FILE
    payload = path.read_bytes()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != UK_FIRM_PAY_DISTRIBUTION_SOURCE_COLUMNS:
        raise ValueError(
            f"UK firm pay distribution columns {list(frame.columns)} are not "
            f"{list(UK_FIRM_PAY_DISTRIBUTION_SOURCE_COLUMNS)}."
        )
    keys = frame["sic_section"].str.strip().tolist()
    expected = [*_SECTION_LETTERS, UK_FIRM_PAY_DISTRIBUTION_ALL_EMPLOYEES]
    if sorted(keys) != sorted(expected):
        raise ValueError(
            f"UK firm pay distribution rows {keys} are not sections A-U and "
            f"{UK_FIRM_PAY_DISTRIBUTION_ALL_EMPLOYEES!r}, once each."
        )
    curves: dict[str, Mapping[int, float]] = {}
    for key, (_, row) in zip(keys, frame.iterrows(), strict=True):
        curve: dict[int, float] = {}
        for percentile in UK_FIRM_PAY_DISTRIBUTION_PERCENTILES:
            cell = row[f"p{percentile}"].strip()
            if cell in UK_FIRM_PAY_DISTRIBUTION_MARKERS:
                continue
            try:
                value = float(cell)
            except ValueError:
                raise ValueError(
                    f"UK firm pay distribution section {key} p{percentile} "
                    f"{cell!r} is neither a number nor an ONS marker."
                ) from None
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"UK firm pay distribution section {key} p{percentile} "
                    "must be positive."
                )
            curve[percentile] = value
        if np.any(np.diff(list(curve.values())) < 0):
            raise ValueError(
                f"UK firm pay distribution section {key} percentiles fall."
            )
        curves[key] = MappingProxyType(curve)
    if len(curves[UK_FIRM_PAY_DISTRIBUTION_ALL_EMPLOYEES]) != len(
        UK_FIRM_PAY_DISTRIBUTION_PERCENTILES
    ):
        raise ValueError("UK firm pay distribution ALL row is not fully published.")
    return UKFirmPayDistributionSource(
        section_percentiles=MappingProxyType(dict(sorted(curves.items()))),
        file=path.name,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _share_and_mean(
    curve: Mapping[int, float], threshold_gbp: float
) -> tuple[float, float] | None:
    """Share of jobs at or below the threshold and their mean pay.

    ``None`` when the curve cannot place the threshold: no published
    percentile, or the threshold above the highest one.
    """

    if not curve or threshold_gbp > max(curve.values()):
        return None
    points = np.array([0.0, *curve]) / 100.0
    pays = np.array([0.0, *curve.values()])
    if threshold_gbp <= 0:
        return 0.0, 0.0
    # First knot at or above the threshold; pay rises strictly across it.
    upper = int(np.searchsorted(pays, threshold_gbp, side="left"))
    lower = upper - 1
    share = points[lower] + (points[upper] - points[lower]) * (
        threshold_gbp - pays[lower]
    ) / (pays[upper] - pays[lower])
    # Area under the quantile line from 0 to ``share``: trapezoids.
    knots_u = np.array([*points[:upper], share])
    knots_pay = np.array([*pays[:upper], threshold_gbp])
    area = float(np.sum(np.diff(knots_u) * (knots_pay[1:] + knots_pay[:-1]) / 2.0))
    return float(share), area / float(share)


def uk_firm_pay_below_threshold_by_section(
    source: UKFirmPayDistributionSource, *, threshold_gbp: float
) -> pd.DataFrame:
    """Per section A-U, ``UNKNOWN`` and ``ALL``: share and mean pay below.

    Columns ``sic_section``, ``share_at_or_below`` (0-1),
    ``mean_pay_at_or_below_gbp`` and ``level`` (``section``, or
    ``all_employees`` where the section takes the ``ALL`` row).
    """

    threshold = float(threshold_gbp)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("UK firm secondary threshold must be non-negative.")
    all_key = UK_FIRM_PAY_DISTRIBUTION_ALL_EMPLOYEES
    fallback = _share_and_mean(source.section_percentiles[all_key], threshold)
    if fallback is None:
        raise ValueError(
            f"UK firm secondary threshold {threshold} is above the highest "
            "published all-employees percentile."
        )
    records = []
    for key in (*_SECTION_LETTERS, UK_FIRM_SIC_SECTION_UNKNOWN, all_key):
        own = _share_and_mean(source.section_percentiles.get(key, {}), threshold)
        share, mean = fallback if own is None else own
        level = "all_employees" if own is None or key == all_key else "section"
        records.append((key, share, mean, level))
    return pd.DataFrame(
        records,
        columns=[
            "sic_section",
            "share_at_or_below",
            "mean_pay_at_or_below_gbp",
            "level",
        ],
    )


def _employees_and_sections(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if {"firm_employees", "firm_sic_section"} <= set(frame.columns):
        employees = frame["firm_employees"].to_numpy()
        sections = frame["firm_sic_section"].astype(str).to_numpy(object)
        if not np.isin(sections, UK_FIRM_SIC_SECTIONS).all():
            raise ValueError(
                "UK firm firm_sic_section must be SICSection member names."
            )
    elif {"employment", "sic_code"} <= set(frame.columns):
        employees = frame["employment"].to_numpy()
        sections = uk_firm_sic_sections(frame["sic_code"])
    else:
        raise ValueError(
            "UK firm pay distribution needs firm_employees and "
            "firm_sic_section, or employment and sic_code."
        )
    as_float = employees.astype(np.float64)
    if (
        not np.all(np.isfinite(as_float))
        or np.any(as_float != np.round(as_float))
        or np.any(as_float < 0)
    ):
        raise ValueError("UK firm employees must be non-negative whole numbers.")
    return as_float.astype(np.int64), sections


def assign_uk_firm_pay_below_secondary_threshold(
    table_or_firms: pd.DataFrame,
    source: UKFirmPayDistributionSource,
    *,
    threshold_gbp: float,
    seed: int = UK_FIRM_PAY_DISTRIBUTION_SEED,
) -> dict[str, np.ndarray]:
    """The two policyengine-uk below-threshold inputs per row, in row order.

    ``firm_employees_below_secondary_threshold`` (int64) is
    ``floor(employees x share)`` plus a fixed-seed Bernoulli draw on the
    remainder, never above the firm's employees.
    ``firm_paybill_below_secondary_threshold`` (float64, pounds) is that
    count times the section's mean pay at or below ``threshold_gbp``, capped
    at ``firm_paybill`` when ``table_or_firms`` has that column. The frame is
    a policyengine-uk firm table (``firm_employees``, ``firm_sic_section``)
    or generator rows (``employment``, ``sic_code``). The draw depends on row
    order.
    """

    employees, sections = _employees_and_sections(table_or_firms)
    by_section = uk_firm_pay_below_threshold_by_section(
        source, threshold_gbp=threshold_gbp
    ).set_index("sic_section")
    share = by_section["share_at_or_below"].reindex(sections).to_numpy(np.float64)
    mean = by_section["mean_pay_at_or_below_gbp"].reindex(sections).to_numpy(np.float64)
    expected = employees * share
    whole = np.floor(expected)
    draws = np.random.default_rng(seed).random(len(employees))
    count = (whole + (draws < expected - whole)).astype(np.int64)
    count = np.minimum(count, employees)
    paybill = count * mean
    if "firm_paybill" in table_or_firms.columns:
        paybill = np.minimum(
            paybill, table_or_firms["firm_paybill"].to_numpy(np.float64)
        )
    return {
        "firm_employees_below_secondary_threshold": count,
        "firm_paybill_below_secondary_threshold": paybill,
    }


__all__ = [
    "UK_FIRM_PAY_DISTRIBUTION_ALL_EMPLOYEES",
    "UK_FIRM_PAY_DISTRIBUTION_FILE",
    "UK_FIRM_PAY_DISTRIBUTION_MARKERS",
    "UK_FIRM_PAY_DISTRIBUTION_METHOD",
    "UK_FIRM_PAY_DISTRIBUTION_PERCENTILES",
    "UK_FIRM_PAY_DISTRIBUTION_SEED",
    "UK_FIRM_PAY_DISTRIBUTION_SOURCE_COLUMNS",
    "UK_FIRM_PAY_DISTRIBUTION_STATUS",
    "UK_FIRM_PAY_DISTRIBUTION_TAX_YEAR",
    "UKFirmPayDistributionSource",
    "assign_uk_firm_pay_below_secondary_threshold",
    "read_uk_firm_pay_distribution_source",
    "uk_firm_pay_below_threshold_by_section",
]
