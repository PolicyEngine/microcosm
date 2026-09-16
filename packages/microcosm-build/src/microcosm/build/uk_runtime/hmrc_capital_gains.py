"""HMRC's published joint distribution of capital gains by taxable income.

The FRS does not ask about capital gains, so a UK build has to impute them.
What it imputes against decides whether the result can answer a capital gains
question at all: the tax is held by very few people, with roughly 2,000
taxpayers above £5m of gains holding £22.7bn of them, so a source that cannot
reach those bands produces a population whose total is right and whose shape
is wrong.

HMRC table 3.1 publishes the joint distribution directly — individuals and
gains by size of gain crossed with taxable income — which is the surface an
imputation conditioned on income needs, and it carries the top bands.

Read the published basis before using these facts. Gains are after losses and
attributed gains but **before** the annual exempt amount. Taxable income is
after reliefs and the personal allowance. Trusts are excluded, and only
individuals with a CGT liability appear at all, which is why the table shows
almost nobody below £10,000 of gains rather than the many people whose gains
fall under the exempt amount.

Counts publish in thousands and amounts in £ millions; both convert here, so
callers see people and pounds. Cells too small to publish carry a suppression
marker instead of a count, and those come back as ``None`` rather than zero —
counts publish in thousands, so a suppressed count means "fewer than 1,000
people", not "nobody".
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from microcosm.build.uk_runtime.ledger_fact_vendoring import load_vendored_resource
from microcosm.build.uk_runtime.ods_tables import ODSTable, read_ods_tables
from microcosm.calibrate.geography_constants import UK_REGION_TIER, UK_REGION_TIER_ENUM

__all__ = [
    "HMRC_CGT_AGE_BAND_LOWER_BOUNDS",
    "HMRC_CGT_CONDITIONING_RESOURCE",
    "HMRC_CGT_CONDITIONING_TAX_YEAR",
    "HMRC_CGT_SIZE_BAND_LOWER_BOUNDS",
    "HMRCCGTAgeBand",
    "HMRCCGTConditioningFacts",
    "HMRCCGTRegionRow",
    "HMRCCGTSizeBand",
    "HMRCCGTTable1Totals",
    "load_hmrc_cgt_conditioning_facts",
    "HMRCCapitalGainsBandTotal",
    "HMRCCapitalGainsIncomeTotal",
    "HMRC_CGT_BUILD_PERIOD",
    "HMRC_CGT_GAIN_BAND_LOWER_BOUNDS",
    "HMRC_CGT_INCOME_BAND_LOWER_BOUNDS",
    "HMRC_CGT_JOINT_ODS_FILENAME",
    "HMRC_CGT_JOINT_ODS_SHA256",
    "HMRC_CGT_JOINT_ODS_SIZE_BYTES",
    "HMRC_CGT_JOINT_ODS_URL",
    "HMRC_CGT_JOINT_SHEET_NAMES",
    "HMRC_CGT_PUBLICATION_URL",
    "HMRC_CGT_SOURCE_LABEL",
    "HMRC_CGT_SOURCE_VINTAGE",
    "HMRC_CGT_TOTAL_GAINS_GBP",
    "HMRC_CGT_TOTAL_INDIVIDUALS",
    "HMRCCapitalGainsCell",
    "HMRCCapitalGainsJointDistribution",
    "HMRCCapitalGainsSourceProvenance",
    "materialize_hmrc_capital_gains_joint_distribution",
]

HMRC_CGT_PUBLICATION_URL = (
    "https://www.gov.uk/government/statistics/capital-gains-tax-statistics"
)
HMRC_CGT_JOINT_ODS_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "6878ac62760bf6cedaf5bd93/Table_3_2025_Size_of_gain_by_income.ods"
)
HMRC_CGT_JOINT_ODS_FILENAME = "Table_3_2025_Size_of_gain_by_income.ods"
HMRC_CGT_JOINT_ODS_SHA256 = (
    "8e75c00bab949348a7238fea6d995f626c85e5d02813b46606dd7fea85e9d0c3"
)
HMRC_CGT_JOINT_ODS_SIZE_BYTES = 11_996
HMRC_CGT_SOURCE_LABEL = "HMRC capital gains table 3"
HMRC_CGT_SOURCE_VINTAGE = "2023-24"
HMRC_CGT_BUILD_PERIOD = "2024"

#: Worksheet per tax year, newest first. The publication carries four years,
#: so a stage can fit each year rather than aging one forward.
HMRC_CGT_JOINT_SHEET_NAMES: dict[str, str] = {
    "2023-24": "3_1_2023-24",
    "2022-23": "3_2_2022-23",
    "2021-22": "3_3_2021-22",
    "2020-21": "3_4_2020-21",
}

#: Lower bound of each published band of gains, in pounds.
HMRC_CGT_GAIN_BAND_LOWER_BOUNDS: tuple[int, ...] = (
    0,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
)

#: Lower bound of each published band of taxable income, in pounds.
HMRC_CGT_INCOME_BAND_LOWER_BOUNDS: tuple[int, ...] = (
    0,
    37_700,
    50_000,
    100_000,
    150_000,
    200_000,
)

#: Published 2023-24 totals, for reconciling a parse against the publication.
HMRC_CGT_TOTAL_INDIVIDUALS = 359_000.0
HMRC_CGT_TOTAL_GAINS_GBP = 62_921_000_000.0

_COUNT_UNIT = 1_000.0
_AMOUNT_UNIT = 1_000_000.0

_BAND_COLUMN = 0
_BAND_COLUMN_HEADING = "Range of gain"
_ALL_ROW_LABEL = "All"

#: Counts publish to the nearest thousand and amounts to the nearest million,
#: so marginals need not sum exactly to the published totals. HMRC says as
#: much on the face of the table.
_COUNT_RECONCILIATION_TOLERANCE = 10_000.0
_AMOUNT_RECONCILIATION_TOLERANCE = 50_000_000.0


@dataclass(frozen=True)
class HMRCCapitalGainsCell:
    """One published cell of the joint distribution, in people and pounds."""

    gain_lower_bound: int
    income_lower_bound: int
    individuals: float | None
    gains: float | None

    @property
    def individuals_suppressed(self) -> bool:
        """Whether HMRC withheld the count as too small to publish.

        Counts publish in thousands, so a suppressed count means fewer than
        1,000 people, not zero.
        """
        return self.individuals is None

    @property
    def gains_suppressed(self) -> bool:
        """Whether HMRC withheld the amount.

        Rarer than a suppressed count, and it does happen: 2022-23 withholds
        an amount in the lowest band of gains at the lowest band of income.
        """
        return self.gains is None


@dataclass(frozen=True)
class HMRCCapitalGainsBandTotal:
    """One published all-incomes row total for a band of gains.

    The row totals come straight from the published "All incomes" pair rather
    than summing the six income cells, so a band whose cells include a
    suppressed amount still has its full published total here. The imputation
    uses these as the fallback mean where a cell's own count is withheld.
    """

    gain_lower_bound: int
    individuals: float | None
    gains: float


@dataclass(frozen=True)
class HMRCCapitalGainsIncomeTotal:
    """One published All-row total for a band of taxable income.

    The column totals come from the published All row rather than summing the
    ten cells above it, so a column containing suppressed cells still has its
    full published taxpayer count. An allocation can reconcile suppressed
    cells against these instead of inventing a count for them.
    """

    income_lower_bound: int
    individuals: float | None
    gains: float | None


@dataclass(frozen=True)
class HMRCCapitalGainsSourceProvenance:
    """What was read, and what it hashed to when read."""

    local_path: Path
    sha256: str
    size_bytes: int
    sheet_name: str
    source_vintage: str
    build_period: str


@dataclass(frozen=True)
class HMRCCapitalGainsJointDistribution:
    """The joint distribution, with the provenance of the file behind it."""

    cells: tuple[HMRCCapitalGainsCell, ...]
    band_totals: tuple[HMRCCapitalGainsBandTotal, ...]
    income_totals: tuple[HMRCCapitalGainsIncomeTotal, ...]
    source: HMRCCapitalGainsSourceProvenance
    total_individuals: float
    total_gains: float

    @property
    def unpublished_gains(self) -> float:
        """Gains inside suppressed cells, as the published total less the cells.

        A stage fitting to these cells is fitting to slightly less than the
        whole, and this is how much it is missing.
        """
        return self.total_gains - sum(
            cell.gains for cell in self.cells if cell.gains is not None
        )

    def cell(
        self, *, gain_lower_bound: int, income_lower_bound: int
    ) -> HMRCCapitalGainsCell:
        """Return one cell by its band bounds."""
        for cell in self.cells:
            if (
                cell.gain_lower_bound == gain_lower_bound
                and cell.income_lower_bound == income_lower_bound
            ):
                return cell
        raise KeyError(
            f"No published cell for gains from {gain_lower_bound} "
            f"and income from {income_lower_bound}."
        )

    def gains_by_band(self) -> dict[int, float]:
        """Published gains in each band of gains, from the row totals."""
        return {total.gain_lower_bound: total.gains for total in self.band_totals}

    def band_total(self, gain_lower_bound: int) -> HMRCCapitalGainsBandTotal:
        """Return one published row total by its band bound."""
        for total in self.band_totals:
            if total.gain_lower_bound == gain_lower_bound:
                return total
        raise KeyError(f"No published band total for gains from {gain_lower_bound}.")

    def income_total(self, income_lower_bound: int) -> HMRCCapitalGainsIncomeTotal:
        """Return one published column total by its income band bound."""
        for total in self.income_totals:
            if total.income_lower_bound == income_lower_bound:
                return total
        raise KeyError(
            f"No published income total for incomes from {income_lower_bound}."
        )


def _fingerprint(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _numeric(value: object, *, label: str) -> float | None:
    """Return a published number, or None where the cell is suppressed."""
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return None
        raise ValueError(f"{label} holds unexpected text {value!r}.")
    if value is None:
        raise ValueError(f"{label} is empty.")
    raise ValueError(f"{label} holds an unexpected value {value!r}.")


def _locate_header_row(table: ODSTable) -> int:
    """Find the header by its band column heading rather than by position.

    The number of notes above the header varies by year — 2020-21 carries one
    fewer than 2023-24 — so a fixed row index reads notes as data on some
    sheets and refuses others.
    """
    matches = [
        index
        for index, row in enumerate(table.rows)
        if row
        and isinstance(row[_BAND_COLUMN], str)
        and row[_BAND_COLUMN].startswith(_BAND_COLUMN_HEADING)
    ]
    if not matches:
        raise ValueError(
            f"{HMRC_CGT_SOURCE_LABEL} sheet {table.sheet_name} has no column "
            f"headed {_BAND_COLUMN_HEADING!r}; the published layout has moved."
        )
    if len(matches) > 1:
        raise ValueError(
            f"{HMRC_CGT_SOURCE_LABEL} sheet {table.sheet_name} has "
            f"{len(matches)} columns headed {_BAND_COLUMN_HEADING!r}."
        )
    return matches[0]


def _validate_headers(table: ODSTable, *, header_row: int) -> None:
    header = table.rows[header_row]
    label = table.sheet_name
    expected_columns = 1 + 2 * (len(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS) + 1)
    if len(header) < expected_columns:
        raise ValueError(
            f"{HMRC_CGT_SOURCE_LABEL} sheet {label} has {len(header)} header "
            f"columns, fewer than the {expected_columns} the published "
            "income bands require."
        )


def materialize_hmrc_capital_gains_joint_distribution(
    path: Path,
    *,
    tax_year: str = HMRC_CGT_SOURCE_VINTAGE,
    build_period: str = HMRC_CGT_BUILD_PERIOD,
    verify_fingerprint: bool = True,
) -> HMRCCapitalGainsJointDistribution:
    """Read the joint distribution of gains by taxable income.

    Args:
        path: Local copy of the published ODS.
        tax_year: Which published year to read, as ``"2023-24"``.
        build_period: The build period these facts are declared against. The
            latest published tax year 2023-24 is replayed against build period
            2024 (signed, microcosm#723).
        verify_fingerprint: Whether to require the pinned hash and size. Only
            a caller reading a different vintage should turn this off, and it
            then owns checking what it read.

    Raises:
        ValueError: If the artifact or the published layout does not match
            what this module was written against.
    """
    if tax_year not in HMRC_CGT_JOINT_SHEET_NAMES:
        raise ValueError(
            f"{HMRC_CGT_SOURCE_LABEL} publishes {sorted(HMRC_CGT_JOINT_SHEET_NAMES)}, "
            f"not {tax_year!r}."
        )

    # Check the artifact before opening it, so a wrong file fails on its
    # identity rather than somewhere inside a parse.
    sha256, size_bytes = _fingerprint(path)
    if verify_fingerprint:
        if size_bytes != HMRC_CGT_JOINT_ODS_SIZE_BYTES:
            raise ValueError(
                f"{HMRC_CGT_SOURCE_LABEL} at {path} is {size_bytes} bytes, "
                f"not the pinned {HMRC_CGT_JOINT_ODS_SIZE_BYTES}."
            )
        if sha256 != HMRC_CGT_JOINT_ODS_SHA256:
            raise ValueError(
                f"{HMRC_CGT_SOURCE_LABEL} at {path} hashes to {sha256}, "
                f"not the pinned {HMRC_CGT_JOINT_ODS_SHA256}."
            )

    sheet_name = HMRC_CGT_JOINT_SHEET_NAMES[tax_year]
    tables = read_ods_tables(path, label=HMRC_CGT_SOURCE_LABEL)
    if sheet_name not in tables:
        raise ValueError(
            f"{HMRC_CGT_SOURCE_LABEL} has no sheet {sheet_name!r}; it carries "
            f"{sorted(tables)}."
        )
    table = tables[sheet_name]
    header_row = _locate_header_row(table)
    _validate_headers(table, header_row=header_row)
    first_band_row = header_row + 1

    cells: list[HMRCCapitalGainsCell] = []
    band_totals: list[HMRCCapitalGainsBandTotal] = []
    total_individuals: float | None = None
    total_gains: float | None = None

    for offset, gain_lower_bound in enumerate(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS):
        row = first_band_row + offset
        published_bound = table.cell(row, _BAND_COLUMN)
        if not isinstance(published_bound, float) or int(published_bound) != (
            gain_lower_bound
        ):
            raise ValueError(
                f"{HMRC_CGT_SOURCE_LABEL} sheet {sheet_name} row {row} opens "
                f"band {published_bound!r}, not the expected {gain_lower_bound}."
            )
        for index, income_lower_bound in enumerate(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS):
            count_column = 1 + 2 * index
            where = (
                f"{HMRC_CGT_SOURCE_LABEL} sheet {sheet_name} "
                f"row {row} column {count_column}"
            )
            individuals = _numeric(table.cell(row, count_column), label=where)
            gains = _numeric(table.cell(row, count_column + 1), label=f"{where} amount")
            cells.append(
                HMRCCapitalGainsCell(
                    gain_lower_bound=gain_lower_bound,
                    income_lower_bound=income_lower_bound,
                    individuals=(
                        None if individuals is None else individuals * _COUNT_UNIT
                    ),
                    gains=None if gains is None else gains * _AMOUNT_UNIT,
                )
            )
        all_incomes_column = 1 + 2 * len(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS)
        where = (
            f"{HMRC_CGT_SOURCE_LABEL} sheet {sheet_name} row {row} "
            f"column {all_incomes_column}"
        )
        row_individuals = _numeric(table.cell(row, all_incomes_column), label=where)
        row_gains = _numeric(
            table.cell(row, all_incomes_column + 1), label=f"{where} amount"
        )
        if row_gains is None:
            raise ValueError(
                f"{where} suppresses the all-incomes amount, which is unexpected."
            )
        band_totals.append(
            HMRCCapitalGainsBandTotal(
                gain_lower_bound=gain_lower_bound,
                individuals=(
                    None if row_individuals is None else row_individuals * _COUNT_UNIT
                ),
                gains=row_gains * _AMOUNT_UNIT,
            )
        )

    all_row = first_band_row + len(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS)
    if str(table.cell(all_row, _BAND_COLUMN)).strip() != _ALL_ROW_LABEL:
        raise ValueError(
            f"{HMRC_CGT_SOURCE_LABEL} sheet {sheet_name} does not close with "
            f"an {_ALL_ROW_LABEL!r} row at row {all_row}."
        )
    income_totals = []
    for index, income_lower_bound in enumerate(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS):
        count_column = 1 + 2 * index
        where = (
            f"{HMRC_CGT_SOURCE_LABEL} sheet {sheet_name} All row column {count_column}"
        )
        column_individuals = _numeric(table.cell(all_row, count_column), label=where)
        column_gains = _numeric(
            table.cell(all_row, count_column + 1), label=f"{where} amount"
        )
        income_totals.append(
            HMRCCapitalGainsIncomeTotal(
                income_lower_bound=income_lower_bound,
                individuals=(
                    None
                    if column_individuals is None
                    else column_individuals * _COUNT_UNIT
                ),
                gains=None if column_gains is None else column_gains * _AMOUNT_UNIT,
            )
        )
    total_column = 1 + 2 * len(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS)
    total_individuals = _numeric(
        table.cell(all_row, total_column), label=f"{sheet_name} total individuals"
    )
    total_gains = _numeric(
        table.cell(all_row, total_column + 1), label=f"{sheet_name} total gains"
    )
    if total_individuals is None or total_gains is None:
        raise ValueError(f"{HMRC_CGT_SOURCE_LABEL} suppresses a published total.")
    total_individuals *= _COUNT_UNIT
    total_gains *= _AMOUNT_UNIT

    # Cells can only sum to less than the published total, by whatever sits
    # inside suppressed cells. Summing to more means the parse has gone wrong.
    summed_gains = sum(cell.gains for cell in cells if cell.gains is not None)
    if summed_gains > total_gains + _AMOUNT_RECONCILIATION_TOLERANCE:
        raise ValueError(
            f"{HMRC_CGT_SOURCE_LABEL} sheet {sheet_name} cells sum to "
            f"{summed_gains} of gains, above a published total of {total_gains}."
        )

    return HMRCCapitalGainsJointDistribution(
        cells=tuple(cells),
        band_totals=tuple(band_totals),
        income_totals=tuple(income_totals),
        source=HMRCCapitalGainsSourceProvenance(
            local_path=path.resolve(),
            sha256=sha256,
            size_bytes=size_bytes,
            sheet_name=sheet_name,
            source_vintage=tax_year,
            build_period=build_period,
        ),
        total_individuals=total_individuals,
        total_gains=total_gains,
    )


# ---------------------------------------------------------------------------
# Vendored 2024-25 conditioning facts (Tables 2.1a, 6, 5 and 1)
# ---------------------------------------------------------------------------

#: The per-concern vendored resource carrying the 2024-25 CGT rows the
#: amounts imputation rakes to and the donor stage weights by. Regenerated
#: from the pinned Chronicle feed by ``tools/vendor_uk_ledger_facts.py``;
#: never edited by hand.
HMRC_CGT_CONDITIONING_RESOURCE = "hmrc_cgt_conditioning_facts.json"

#: Tax year (start year) of every conditioning row: the FY2024-25
#: individual observations the calibration targets fit (microcosm#889).
HMRC_CGT_CONDITIONING_TAX_YEAR = 2024

#: Lower bound of each published Table 2.1a size-of-gain band, in pounds.
#: The bands below £12,300 exist because the annual exempt amount moved
#: (£12,300 → £6,000 → £3,000); Table 3's ten bands start at 0 and 10,000.
HMRC_CGT_SIZE_BAND_LOWER_BOUNDS: tuple[int, ...] = (
    0,
    3_000,
    6_000,
    10_000,
    12_300,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
)

#: Lower bound of each published Table 6 age band (age at the end of the
#: tax year). The top band is open.
HMRC_CGT_AGE_BAND_LOWER_BOUNDS: tuple[int, ...] = (0, 16, 25, 35, 45, 55, 65, 75, 85)

_SIZE_BAND_RECORD_SET_PREFIX = "hmrc.cgt_size_of_gain_2026.table2_1a.ty2024."
_AGE_BAND_RECORD_SET_ID = "hmrc.cgt_table6_2026.age.ty2024"
_REGION_RECORD_SET_ID = "hmrc.cgt_table5_2026.country_region.ty2024"
_TABLE1_RECORD_SET_ID = "hmrc.cgt_statistics_2026.table1.ty2024"
_UK_GEOGRAPHY_ID = "K02000001"


@dataclass(frozen=True)
class HMRCCGTSizeBand:
    """One Table 2.1a row: individuals with a liability by size of gain."""

    lower_bound: int
    upper_bound: int | None
    taxpayers: float
    gains: float


@dataclass(frozen=True)
class HMRCCGTAgeBand:
    """One Table 6 row: individuals with a liability by age band."""

    lower_bound: int
    upper_bound: int | None
    taxpayers: float
    gains: float
    tax: float


@dataclass(frozen=True)
class HMRCCGTRegionRow:
    """One Table 5 region-tier row: all taxpayers (including trusts)."""

    geography_id: str
    region: str
    taxpayers: float
    gains: float
    tax: float


@dataclass(frozen=True)
class HMRCCGTTable1Totals:
    """Table 1 lines for the year: individuals, trusts and their total."""

    individuals_taxpayers: float
    individuals_gains: float
    individuals_tax: float
    trusts_taxpayers: float
    trusts_gains: float
    trusts_tax: float
    total_taxpayers: float
    total_gains: float
    total_tax: float

    def individuals_share(self, measure: str) -> float:
        """Individuals ÷ total for ``taxpayers``, ``gains`` or ``tax``.

        The share that restates a Table 5 all-taxpayer cell on the
        individuals basis of the national targets, on the declared
        assumption that trusts are spread across areas like all taxpayers.
        """

        numerator = getattr(self, f"individuals_{measure}")
        denominator = getattr(self, f"total_{measure}")
        if denominator <= 0:
            raise ValueError(f"Table 1 total {measure} must be positive.")
        return numerator / denominator


@dataclass(frozen=True)
class HMRCCGTConditioningFacts:
    """The vendored 2024-25 rows, typed, with the resource digest they came from."""

    tax_year: int
    size_bands: tuple[HMRCCGTSizeBand, ...]
    age_bands: tuple[HMRCCGTAgeBand, ...]
    regions: tuple[HMRCCGTRegionRow, ...]
    table1: HMRCCGTTable1Totals
    resource: str
    resource_sha256: str
    source_commit: str

    def size_band(self, lower_bound: int) -> HMRCCGTSizeBand:
        for band in self.size_bands:
            if band.lower_bound == lower_bound:
                return band
        raise KeyError(f"No Table 2.1a band with lower bound {lower_bound}.")

    def region(self, name: str) -> HMRCCGTRegionRow:
        for row in self.regions:
            if row.region == name:
                return row
        raise KeyError(f"No Table 5 row for region {name!r}.")

    def size_bands_aggregated(
        self, bounds: tuple[int, ...]
    ) -> dict[int, tuple[float, float]]:
        """Table 2.1a taxpayers and gains re-binned onto coarser lower bounds.

        Each published band folds into the coarser band whose lower bound
        is the largest not above its own, so Table 3's ten bands (0 and
        10,000 absorb 3,000/6,000 and 12,300) receive the 2024-25 levels.
        """

        if tuple(sorted(bounds)) != tuple(bounds) or bounds[0] != 0:
            raise ValueError("Aggregation bounds must be ascending and start at 0.")
        folded = {bound: [0.0, 0.0] for bound in bounds}
        for band in self.size_bands:
            target = max(bound for bound in bounds if bound <= band.lower_bound)
            folded[target][0] += band.taxpayers
            folded[target][1] += band.gains
        return {bound: (people, gains) for bound, (people, gains) in folded.items()}


_BAND_VALUE_ID = re.compile(
    r"^(?P<prefix>[a-z_]+?)_(?P<lower>\d+)_(?:to_(?P<upper>\d+)|(?P<open>plus))$"
)


def _band_bounds_from_value_id(value_id: str, *, prefix: str) -> tuple[int, int | None]:
    """The inclusive lower and exclusive upper edge a band value id states.

    The vendored projection carries the publisher's band as its dimension
    value id (``gain_3000_to_5999``, ``age_85_plus``): the inclusive
    integer upper limit becomes the exclusive edge one past it, and a
    ``_plus`` band is open.
    """

    match = _BAND_VALUE_ID.match(value_id)
    if match is None or match.group("prefix") != prefix:
        raise ValueError(
            f"Vendored CGT band value id {value_id!r} is not of the form "
            f"{prefix}_<lower>_to_<upper> or {prefix}_<lower>_plus."
        )
    lower = int(match.group("lower"))
    if match.group("open"):
        return lower, None
    return lower, int(match.group("upper")) + 1


def _band_value_id(row: Mapping[str, Any], dimension: str) -> str:
    value = (row.get("dimensions") or {}).get(dimension)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Vendored CGT row {row.get('aggregate_fact_key')!r} carries no "
            f"{dimension!r} dimension."
        )
    return value


def _value(row: Mapping[str, Any]) -> float:
    value = row.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"Vendored CGT row {row.get('aggregate_fact_key')!r} has no numeric value."
        )
    return float(value)


def _measure_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("measure_id") or (row.get("layout") or {}).get("measure_id") or ""
    )


def _record_set_id(row: Mapping[str, Any]) -> str:
    return str((row.get("layout") or {}).get("record_set_id") or "")


def load_hmrc_cgt_conditioning_facts(
    resource: str = HMRC_CGT_CONDITIONING_RESOURCE,
) -> HMRCCGTConditioningFacts:
    """Type the vendored 2024-25 rows and refuse a roster that has drifted.

    Every band roster, the region set and the Table 1 lines are checked
    against the module constants and the region tier, so a re-vendored
    resource whose publisher shape moved fails here rather than raking to
    the wrong cells.
    """

    payload = load_vendored_resource(resource)
    digest = hashlib.sha256(
        files("microcosm.build.uk").joinpath(resource).read_bytes()
    ).hexdigest()
    rows = list(payload["rows"])

    size_cells: dict[object, dict[str, float]] = {}
    for row in rows:
        if not _record_set_id(row).startswith(_SIZE_BAND_RECORD_SET_PREFIX):
            continue
        lower, upper = _band_bounds_from_value_id(
            _band_value_id(row, "cgt_gain_band"), prefix="gain"
        )
        measure = {"taxpayers_individuals": "taxpayers", "gains_individuals": "gains"}[
            _measure_id(row)
        ]
        cell = size_cells.setdefault((lower, upper), {})
        if measure in cell:
            raise ValueError(f"Table 2.1a: duplicate {measure!r} row for {lower}.")
        cell[measure] = _value(row)
    size_bands = tuple(
        HMRCCGTSizeBand(
            lower_bound=lower,
            upper_bound=upper,
            taxpayers=cell["taxpayers"],
            gains=cell["gains"],
        )
        for (lower, upper), cell in sorted(
            size_cells.items(), key=lambda item: item[0][0]
        )
    )
    if (
        tuple(band.lower_bound for band in size_bands)
        != HMRC_CGT_SIZE_BAND_LOWER_BOUNDS
    ):
        raise ValueError(
            "Table 2.1a band roster drifted: expected lower bounds "
            f"{HMRC_CGT_SIZE_BAND_LOWER_BOUNDS}, got "
            f"{tuple(band.lower_bound for band in size_bands)}."
        )
    for earlier, later in zip(size_bands, size_bands[1:], strict=False):
        if earlier.upper_bound != later.lower_bound:
            raise ValueError("Table 2.1a bands are not contiguous.")
    if size_bands[-1].upper_bound is not None:
        raise ValueError("Table 2.1a top band must be open.")

    age_cells: dict[object, dict[str, float]] = {}
    age_totals: dict[str, float] = {}
    for row in rows:
        if _record_set_id(row) != _AGE_BAND_RECORD_SET_ID:
            continue
        measure = {
            "taxpayers_individuals": "taxpayers",
            "gains_individuals": "gains",
            "tax_individuals": "tax",
        }[_measure_id(row)]
        if (row.get("layout") or {}).get("table_record_kind") == "total":
            if measure in age_totals:
                raise ValueError(f"Table 6: duplicate all-ages {measure!r} row.")
            age_totals[measure] = _value(row)
            continue
        lower, upper = _band_bounds_from_value_id(
            _band_value_id(row, "age_band"), prefix="age"
        )
        cell = age_cells.setdefault((lower, upper), {})
        if measure in cell:
            raise ValueError(f"Table 6: duplicate {measure!r} row for age {lower}.")
        cell[measure] = _value(row)
    age_bands = tuple(
        HMRCCGTAgeBand(
            lower_bound=lower,
            upper_bound=upper,
            taxpayers=cell["taxpayers"],
            gains=cell["gains"],
            tax=cell["tax"],
        )
        for (lower, upper), cell in sorted(
            age_cells.items(), key=lambda item: item[0][0]
        )
    )
    if tuple(band.lower_bound for band in age_bands) != HMRC_CGT_AGE_BAND_LOWER_BOUNDS:
        raise ValueError(
            "Table 6 age-band roster drifted: expected lower bounds "
            f"{HMRC_CGT_AGE_BAND_LOWER_BOUNDS}, got "
            f"{tuple(band.lower_bound for band in age_bands)}."
        )
    if age_bands[-1].upper_bound is not None:
        raise ValueError("Table 6 top age band must be open.")

    region_cells: dict[str, dict[str, float]] = {}
    for row in rows:
        if _record_set_id(row) != _REGION_RECORD_SET_ID:
            continue
        geography = row.get("geography") or {}
        geography_id = str(geography.get("id") or "")
        if geography_id not in UK_REGION_TIER_ENUM:
            continue
        measure = {
            "taxpayers_total": "taxpayers",
            "gains_total": "gains",
            "tax_total": "tax",
        }[_measure_id(row)]
        cell = region_cells.setdefault(geography_id, {})
        if measure in cell:
            raise ValueError(f"Table 5: duplicate {measure!r} row for {geography_id}.")
        cell[measure] = _value(row)
    regions = tuple(
        HMRCCGTRegionRow(
            geography_id=geography_id,
            region=UK_REGION_TIER_ENUM[geography_id],
            taxpayers=region_cells[geography_id]["taxpayers"],
            gains=region_cells[geography_id]["gains"],
            tax=region_cells[geography_id]["tax"],
        )
        for _, geography_id in UK_REGION_TIER
        if geography_id in region_cells
    )
    if len(regions) != len(UK_REGION_TIER):
        missing = sorted(
            geography_id
            for _, geography_id in UK_REGION_TIER
            if geography_id not in region_cells
        )
        raise ValueError(f"Table 5 lacks region-tier area(s) {missing}.")

    table1_values: dict[str, float] = {}
    for row in rows:
        if _record_set_id(row) != _TABLE1_RECORD_SET_ID:
            continue
        measure = _measure_id(row)
        if measure in table1_values:
            raise ValueError(f"Table 1: duplicate {measure!r} row.")
        table1_values[measure] = _value(row)
    try:
        table1 = HMRCCGTTable1Totals(
            individuals_taxpayers=table1_values["individuals_count"],
            individuals_gains=table1_values["individuals_gains"],
            individuals_tax=table1_values["individuals_tax"],
            trusts_taxpayers=table1_values["trusts_count"],
            trusts_gains=table1_values["trusts_gains"],
            trusts_tax=table1_values["trusts_tax"],
            total_taxpayers=table1_values["total_taxpayers"],
            total_gains=table1_values["total_gains"],
            total_tax=table1_values["total_tax"],
        )
    except KeyError as missing:
        raise ValueError(f"Table 1 lacks the {missing} line.") from None

    # The three individual surfaces describe one universe; a vendored
    # resource whose bands no longer sum to the Table 1 individuals line
    # would rake to inconsistent margins.
    for label, total in (
        ("Table 2.1a", sum(band.taxpayers for band in size_bands)),
        ("Table 6", sum(band.taxpayers for band in age_bands)),
        ("Table 6 all-ages", age_totals.get("taxpayers", float("nan"))),
    ):
        if abs(total - table1.individuals_taxpayers) > _COUNT_RECONCILIATION_TOLERANCE:
            raise ValueError(
                f"{label} taxpayers {total} do not reconcile with Table 1 "
                f"individuals {table1.individuals_taxpayers}."
            )

    return HMRCCGTConditioningFacts(
        tax_year=HMRC_CGT_CONDITIONING_TAX_YEAR,
        size_bands=size_bands,
        age_bands=age_bands,
        regions=regions,
        table1=table1,
        resource=resource,
        resource_sha256=digest,
        source_commit=str(payload["source_fact_feed"]["source_commit"]),
    )
