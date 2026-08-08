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
from dataclasses import dataclass
from pathlib import Path

from populace.build.uk_runtime.ods_tables import ODSTable, read_ods_tables

__all__ = [
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
HMRC_CGT_BUILD_PERIOD = "2023"

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
        build_period: The build period these facts are declared against.
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
