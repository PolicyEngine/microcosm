"""UK firm paybill: employees times the ONS ASHE section mean annual gross pay.

policyengine-uk's ``firm_paybill`` is a firm's annual Class 1 earnings in
pounds, before employer contributions. The experimental UK firm generator draws
no pay, so this module imputes it as

    firm_paybill = firm_employees x mean annual gross pay of the firm's section

from one published table: ONS ASHE Table 4.7a (*Annual pay - Gross*), 2025
revised edition, sheet ``All`` (all employee jobs, full-time and part-time),
column ``Mean``. ASHE annual estimates cover the tax year ending 5 April of the
reference year, so the 2025 edition is tax year 2024-25. The extracted CSV and
its provenance sit in ``tests/fixtures/uk/firm_paybill_2024/``
(``tools/extract_uk_firm_paybill_table.py`` re-creates it).

Choices and caveats, all of them assumptions beyond the source:

- the all-employees mean, not the full-time mean: the generator's employment
  is a headcount that includes part-time staff, so a full-time mean would
  overstate pay in part-time-heavy sections (accommodation, retail, health);
- every firm in a section pays that section's mean, whatever its size,
  region or turnover;
- ASHE annual pay covers employees on adult rates in the same job for more
  than a year, which leaves out short-tenure and youth-rate jobs (lower
  paid), so the mean leans high;
- the generator's employment comes from ONS employment size bands, which
  count working proprietors as well as employees, so a proprietor without
  a payroll gets a paybill they would not pay as Class 1 earnings;
- a section ONS does not publish (``x`` or ``:``; section U in 2025) and the
  ``UNKNOWN`` section take the table's ``ALL EMPLOYEES`` mean.
  ``uk_firm_paybill_fallback_counts`` counts the firms that do.

The table is for tax year 2024-25 (``UK_FIRM_PAYBILL_TAX_YEAR``), and
``build_uk_firm_dataset`` refuses a paybill source for any other
``data_vintage``.

Measured on the 20-iteration thinned 2024-25 test build (seed 42), the
paybill is kept, and disclosed, where it is least like a payroll:

- out-of-frame zero-turnover traders (``in_frame`` false) carry about 2.43
  million weighted employees and about 103.8 billion pounds of paybill,
  about 9.7% of the total. In-frame paybill is about 4% below ASHE's jobs
  times its all-employees mean, so the build's excess over ASHE comes from
  the out-of-frame stratum;
- no firm has a zero headcount, so working proprietors get a paybill: sole
  traders carry about 123 billion pounds and partnerships about 45 billion
  pounds, about 16% of the total together.

On the full 2024-25 build (2,946,015 rows, 1,000 epochs) the weighted
paybill is 1.0255 trillion pounds and no firm falls back to the
all-employees mean.
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

#: The extracted ASHE table's file name in a paybill source directory.
UK_FIRM_PAYBILL_FILE = "ashe_section_annual_pay.csv"
#: Its columns, in order.
UK_FIRM_PAYBILL_SOURCE_COLUMNS = (
    "sic_section",
    "description",
    "jobs_thousand",
    "mean_annual_gross_pay_gbp",
    "marker",
)
#: ``sic_section`` of the table's ``ALL EMPLOYEES`` row.
UK_FIRM_PAYBILL_ALL_EMPLOYEES = "ALL"
#: The tax year the table's means describe: ASHE 2025 covers the year to
#: 5 April 2025. A build's ``data_vintage`` must match it.
UK_FIRM_PAYBILL_TAX_YEAR = "2024-25"
#: The receipt's ``uk_firm_paybill`` status and method.
UK_FIRM_PAYBILL_STATUS = "assumed"
UK_FIRM_PAYBILL_METHOD = (
    "firm_employees x ONS ASHE 2025 Table 4.7a section mean annual gross pay "
    "(all employee jobs)"
)
#: Where a firm's mean pay comes from: its section's published mean, or the
#: all-employees mean for an unpublished or unknown section.
UK_FIRM_PAYBILL_LEVELS = ("section", "all_employees")

_SECTION_LETTERS = tuple(
    section
    for section in UK_FIRM_SIC_SECTIONS
    if section != UK_FIRM_SIC_SECTION_UNKNOWN
)


@dataclass(frozen=True)
class UKFirmPaybillSource:
    """Mean annual gross pay (pounds) per SIC section from ASHE Table 4.7a.

    ``section_mean`` holds the sections ONS publishes; ``unpublished`` maps
    each other section A-U to its ONS marker. ``sha256`` is the CSV's hash.
    """

    section_mean: Mapping[str, float]
    all_employees_mean: float
    unpublished: Mapping[str, str]
    file: str
    sha256: str


def read_uk_firm_paybill_source(path: str | Path) -> UKFirmPaybillSource:
    """Read the ASHE section table from a directory or the CSV itself.

    Raises ``ValueError`` unless the file has exactly
    ``UK_FIRM_PAYBILL_SOURCE_COLUMNS``, each of sections A-U once and one
    ``ALL`` row, a positive finite mean wherever no ONS marker stands, and
    no mean beside a marker.
    """

    path = Path(path)
    if path.is_dir():
        path = path / UK_FIRM_PAYBILL_FILE
    payload = path.read_bytes()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != UK_FIRM_PAYBILL_SOURCE_COLUMNS:
        raise ValueError(
            f"UK firm paybill source columns {list(frame.columns)} are not "
            f"{list(UK_FIRM_PAYBILL_SOURCE_COLUMNS)}."
        )
    keys = frame["sic_section"].str.strip().tolist()
    expected = [*_SECTION_LETTERS, UK_FIRM_PAYBILL_ALL_EMPLOYEES]
    if sorted(keys) != sorted(expected):
        raise ValueError(
            f"UK firm paybill source rows {keys} are not sections A-U and "
            f"{UK_FIRM_PAYBILL_ALL_EMPLOYEES!r}, once each."
        )
    means: dict[str, float] = {}
    unpublished: dict[str, str] = {}
    for key, mean, marker in zip(
        keys,
        frame["mean_annual_gross_pay_gbp"].str.strip(),
        frame["marker"].str.strip(),
        strict=True,
    ):
        if marker:
            if mean:
                raise ValueError(
                    f"UK firm paybill section {key} has a mean and marker {marker!r}."
                )
            unpublished[key] = marker
            continue
        try:
            value = float(mean)
        except ValueError:
            raise ValueError(
                f"UK firm paybill section {key} mean {mean!r} is not a number."
            ) from None
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"UK firm paybill section {key} mean must be positive.")
        means[key] = value
    if UK_FIRM_PAYBILL_ALL_EMPLOYEES not in means:
        raise ValueError("UK firm paybill source has no all-employees mean.")
    all_employees = means.pop(UK_FIRM_PAYBILL_ALL_EMPLOYEES)
    return UKFirmPaybillSource(
        section_mean=MappingProxyType(dict(sorted(means.items()))),
        all_employees_mean=all_employees,
        unpublished=MappingProxyType(dict(sorted(unpublished.items()))),
        file=path.name,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _employees_and_sections(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Headcount and ``SICSection`` names from a firm table or generator rows."""

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
            "UK firm paybill needs firm_employees and firm_sic_section, or "
            "employment and sic_code."
        )
    as_float = employees.astype(np.float64)
    if (
        not np.all(np.isfinite(as_float))
        or np.any(as_float != np.round(as_float))
        or np.any(as_float < 0)
    ):
        raise ValueError("UK firm employees must be non-negative whole numbers.")
    return as_float.astype(np.int64), sections


def _uses_section_mean(sections: np.ndarray, source: UKFirmPaybillSource) -> np.ndarray:
    return np.isin(sections, list(source.section_mean))


def uk_firm_paybill_mean_pay(
    table_or_firms: pd.DataFrame, source: UKFirmPaybillSource
) -> np.ndarray:
    """Each row's mean annual gross pay (pounds), in row order.

    A section ``source`` publishes gives its mean; any other section,
    ``UNKNOWN`` included, gives ``source.all_employees_mean``.
    """

    _, sections = _employees_and_sections(table_or_firms)
    lookup = {**source.section_mean}
    return np.array(
        [lookup.get(section, source.all_employees_mean) for section in sections],
        dtype=np.float64,
    )


def assign_uk_firm_paybill(
    table_or_firms: pd.DataFrame, source: UKFirmPaybillSource
) -> np.ndarray:
    """Annual paybill in pounds per row, in row order: employees x mean pay.

    ``table_or_firms`` is a policyengine-uk firm table (``firm_employees``,
    ``firm_sic_section``) or the generator's rows (``employment``,
    ``sic_code``). A firm with no employees has a zero paybill. Sections
    without a published mean use the all-employees mean; see
    ``uk_firm_paybill_fallback_counts``.
    """

    employees, _ = _employees_and_sections(table_or_firms)
    return employees * uk_firm_paybill_mean_pay(table_or_firms, source)


def uk_firm_paybill_fallback_counts(
    table_or_firms: pd.DataFrame, source: UKFirmPaybillSource
) -> pd.DataFrame:
    """Firms, weight and paybill per mean-pay level.

    One row per ``UK_FIRM_PAYBILL_LEVELS`` entry with ``firms`` (rows),
    ``firms_with_employees`` (rows with a positive headcount), ``weight``
    (sum of ``firm_weight``, or of ones when the frame has none) and
    ``weighted_paybill_gbp``.
    """

    employees, sections = _employees_and_sections(table_or_firms)
    level = np.where(_uses_section_mean(sections, source), 0, 1)
    weight = (
        table_or_firms["firm_weight"].to_numpy(dtype=np.float64)
        if "firm_weight" in table_or_firms.columns
        else np.ones(len(table_or_firms), dtype=np.float64)
    )
    paybill = assign_uk_firm_paybill(table_or_firms, source)
    n_levels = len(UK_FIRM_PAYBILL_LEVELS)
    firms = np.bincount(level, minlength=n_levels)
    employing = np.bincount(level, weights=employees > 0, minlength=n_levels)
    weights = np.bincount(level, weights=weight, minlength=n_levels)
    totals = np.bincount(level, weights=weight * paybill, minlength=n_levels)
    return pd.DataFrame(
        {
            "level": list(UK_FIRM_PAYBILL_LEVELS),
            "firms": firms.astype(np.int64),
            "firms_with_employees": employing.astype(np.int64),
            "weight": weights,
            "weighted_paybill_gbp": totals,
        }
    )


__all__ = [
    "UK_FIRM_PAYBILL_ALL_EMPLOYEES",
    "UK_FIRM_PAYBILL_FILE",
    "UK_FIRM_PAYBILL_LEVELS",
    "UK_FIRM_PAYBILL_METHOD",
    "UK_FIRM_PAYBILL_SOURCE_COLUMNS",
    "UK_FIRM_PAYBILL_STATUS",
    "UK_FIRM_PAYBILL_TAX_YEAR",
    "UKFirmPaybillSource",
    "assign_uk_firm_paybill",
    "read_uk_firm_paybill_source",
    "uk_firm_paybill_fallback_counts",
    "uk_firm_paybill_mean_pay",
]
