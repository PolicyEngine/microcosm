"""Strict HMRC SPI income-band targets for the UK national build.

The official 2023-24 collated ODS publishes counts in thousands of people and
amounts in GBP millions.  This module converts those source units to people
and GBP while retaining the exact 13-band source surface.  It deliberately
has no network or ``policyengine-uk-data`` dependency: callers must supply the
local ODS artifact whose runtime hash is recorded in the returned provenance.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

__all__ = [
    "HMRCIncomeBandTargetRecord",
    "HMRCIncomeSourceProvenance",
    "HMRCIncomeTargetSet",
    "HMRC_SPI_BUILD_PERIOD",
    "HMRC_SPI_COLLATED_ODS_URL",
    "HMRC_SPI_INCOME_BAND_LOWER_BOUNDS",
    "HMRC_SPI_INCOME_COMPONENTS",
    "HMRC_SPI_PUBLICATION_URL",
    "HMRC_SPI_SOURCE_TAX_YEAR",
    "HMRC_SPI_SOURCE_TAX_YEAR_START",
    "HMRC_SPI_SOURCE_VINTAGE",
    "HMRC_SPI_TARGET_RECORD_COUNT",
    "materialize_hmrc_spi_income_band_targets",
]

HMRC_SPI_PUBLICATION_URL = (
    "https://www.gov.uk/government/statistics/"
    "personal-incomes-statistics-for-the-tax-year-2023-to-2024"
)
HMRC_SPI_COLLATED_ODS_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "69f1f12d2fae53a03709682f/Collated_Tables_3_1_to_3_11_2324.ods"
)
HMRC_SPI_SOURCE_VINTAGE = "2023-24"
HMRC_SPI_SOURCE_TAX_YEAR = HMRC_SPI_SOURCE_VINTAGE
HMRC_SPI_SOURCE_TAX_YEAR_START = 2023
HMRC_SPI_BUILD_PERIOD = "2023"

HMRC_SPI_INCOME_BAND_LOWER_BOUNDS = (
    12_570,
    15_000,
    20_000,
    30_000,
    40_000,
    50_000,
    70_000,
    100_000,
    150_000,
    200_000,
    300_000,
    500_000,
    1_000_000,
)
HMRC_SPI_INCOME_COMPONENTS = (
    "employment_income",
    "self_employment_income",
    "state_pension",
    "private_pension_income",
    "property_income",
    "savings_interest_income",
    "dividend_income",
    "other_investment_income",
)
HMRC_SPI_TARGET_RECORD_COUNT = (
    len(HMRC_SPI_INCOME_BAND_LOWER_BOUNDS) * len(HMRC_SPI_INCOME_COMPONENTS) * 2
)

HMRCIncomeMeasure = Literal["count", "amount"]
HMRCIncomeUnit = Literal["people", "GBP"]


@dataclass(frozen=True)
class HMRCIncomeSourceProvenance:
    """Runtime identity and explicit period mapping for one local HMRC ODS."""

    local_path: Path
    sha256: str
    publication_url: str
    ods_url: str
    source_vintage: str
    source_tax_year: str
    source_tax_year_start: int
    build_period: str
    table_names: tuple[str, ...]


@dataclass(frozen=True)
class HMRCIncomeBandTargetRecord:
    """One source-backed count or amount for one total-income band."""

    name: str
    component: str
    measure: HMRCIncomeMeasure
    unit: HMRCIncomeUnit
    value: float
    period: str
    total_income_lower_bound: int
    total_income_upper_bound: int | None


@dataclass(frozen=True)
class HMRCIncomeTargetSet:
    """A complete HMRC target surface and its shared source provenance."""

    source: HMRCIncomeSourceProvenance
    targets: tuple[HMRCIncomeBandTargetRecord, ...]


@dataclass(frozen=True)
class _ComponentColumns:
    component: str
    count_position: int
    amount_position: int


@dataclass(frozen=True)
class _TableLayout:
    sheet_name: str
    components: tuple[_ComponentColumns, ...]


_TABLE_LAYOUTS = (
    _TableLayout(
        sheet_name="Table_3_6",
        components=(
            _ComponentColumns("self_employment_income", 1, 2),
            _ComponentColumns("employment_income", 4, 5),
            _ComponentColumns("state_pension", 7, 8),
            _ComponentColumns("private_pension_income", 10, 11),
        ),
    ),
    _TableLayout(
        sheet_name="Table_3_7",
        components=(
            _ComponentColumns("property_income", 1, 2),
            _ComponentColumns("savings_interest_income", 4, 5),
            _ComponentColumns("dividend_income", 7, 8),
            # Table 3.7's fourth pair is "Other income".  The enhanced-FRS
            # parser previously stopped at dividends, silently omitting these
            # published positions 10/11 from the target family.
            _ComponentColumns("other_investment_income", 10, 11),
        ),
    ),
)

_FIRST_DATA_ROW = 5
_SOURCE_SCALE: dict[HMRCIncomeMeasure, float] = {
    "count": 1_000.0,
    "amount": 1_000_000.0,
}
_UNIT_BY_MEASURE: dict[HMRCIncomeMeasure, HMRCIncomeUnit] = {
    "count": "people",
    "amount": "GBP",
}


def materialize_hmrc_spi_income_band_targets(
    ods_path: str | Path,
    *,
    build_period: int | str,
) -> HMRCIncomeTargetSet:
    """Parse the complete 2023-24 HMRC SPI income-band target surface.

    The source tax year is mapped by its tax-year start, so 2023-24 is valid
    only for the Populace build period ``"2023"``.  Any missing, duplicate, or
    malformed source record raises; this function never narrows the family.
    """

    period = str(build_period)
    if period != HMRC_SPI_BUILD_PERIOD:
        raise ValueError(
            f"HMRC SPI source tax year {HMRC_SPI_SOURCE_TAX_YEAR} starts in "
            f"{HMRC_SPI_SOURCE_TAX_YEAR_START} and maps to build period "
            f"{HMRC_SPI_BUILD_PERIOD!r}; got {period!r}."
        )

    _validate_component_layouts()
    path = Path(ods_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HMRC SPI collated ODS not found: {path}.")

    source = HMRCIncomeSourceProvenance(
        local_path=path,
        sha256=_sha256(path),
        publication_url=HMRC_SPI_PUBLICATION_URL,
        ods_url=HMRC_SPI_COLLATED_ODS_URL,
        source_vintage=HMRC_SPI_SOURCE_VINTAGE,
        source_tax_year=HMRC_SPI_SOURCE_TAX_YEAR,
        source_tax_year_start=HMRC_SPI_SOURCE_TAX_YEAR_START,
        build_period=period,
        table_names=tuple(layout.sheet_name for layout in _TABLE_LAYOUTS),
    )

    records: list[HMRCIncomeBandTargetRecord] = []
    for layout in _TABLE_LAYOUTS:
        frame = pd.read_excel(
            path,
            sheet_name=layout.sheet_name,
            engine="odf",
            header=None,
        )
        records.extend(_records_from_table(frame, layout=layout, period=period))

    targets = tuple(records)
    _validate_target_surface(targets)
    return HMRCIncomeTargetSet(source=source, targets=targets)


def _records_from_table(
    frame: pd.DataFrame,
    *,
    layout: _TableLayout,
    period: str,
) -> list[HMRCIncomeBandTargetRecord]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{layout.sheet_name} must load as a pandas DataFrame, "
            f"got {type(frame).__name__}."
        )
    if frame.shape[1] == 0:
        raise ValueError(f"{layout.sheet_name} has no columns.")

    missing_positions = [
        f"{component.component} count (position {component.count_position})"
        for component in layout.components
        if component.count_position >= frame.shape[1]
    ]
    missing_positions.extend(
        f"{component.component} amount (position {component.amount_position})"
        for component in layout.components
        if component.amount_position >= frame.shape[1]
    )
    if missing_positions:
        raise ValueError(
            f"{layout.sheet_name} is missing component column(s): {missing_positions}."
        )

    band_rows = _strict_band_rows(frame, sheet_name=layout.sheet_name)
    records: list[HMRCIncomeBandTargetRecord] = []
    upper_bounds = (*HMRC_SPI_INCOME_BAND_LOWER_BOUNDS[1:], None)
    for lower_bound, upper_bound in zip(
        HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
        upper_bounds,
        strict=True,
    ):
        row_position = band_rows[lower_bound]
        for component in layout.components:
            for measure, column_position in (
                ("count", component.count_position),
                ("amount", component.amount_position),
            ):
                source_value = _strict_positive_value(
                    frame.iat[row_position, column_position],
                    sheet_name=layout.sheet_name,
                    component=component.component,
                    measure=measure,
                    lower_bound=lower_bound,
                )
                value = source_value * _SOURCE_SCALE[measure]
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"{layout.sheet_name} component {component.component!r} "
                        f"{measure} at lower band {lower_bound} does not convert "
                        "to a finite positive target."
                    )
                records.append(
                    HMRCIncomeBandTargetRecord(
                        name=_target_name(
                            component.component,
                            measure,
                            lower_bound,
                            upper_bound,
                        ),
                        component=component.component,
                        measure=measure,
                        unit=_UNIT_BY_MEASURE[measure],
                        value=value,
                        period=period,
                        total_income_lower_bound=lower_bound,
                        total_income_upper_bound=upper_bound,
                    )
                )
    return records


def _strict_band_rows(frame: pd.DataFrame, *, sheet_name: str) -> dict[int, int]:
    if len(frame) <= _FIRST_DATA_ROW:
        raise ValueError(f"{sheet_name} has no income-band rows.")

    parsed: list[tuple[int, int]] = []
    for row_position in range(_FIRST_DATA_ROW, len(frame)):
        lower_bound = _integer_or_none(frame.iat[row_position, 0])
        if lower_bound is not None:
            parsed.append((lower_bound, row_position))

    actual = tuple(lower_bound for lower_bound, _ in parsed)
    counts = Counter(actual)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{sheet_name} contains duplicate income lower band(s): {duplicates}."
        )

    expected_set = set(HMRC_SPI_INCOME_BAND_LOWER_BOUNDS)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected or len(actual) != len(expected_set):
        raise ValueError(
            f"{sheet_name} must contain exactly the 13 published income lower "
            f"bands; missing={missing}, unexpected={unexpected}, "
            f"observed_count={len(actual)}."
        )
    if actual != HMRC_SPI_INCOME_BAND_LOWER_BOUNDS:
        raise ValueError(
            f"{sheet_name} income lower bands are out of published order: {actual}."
        )
    return dict(parsed)


def _validate_component_layouts() -> None:
    components = [
        component.component
        for layout in _TABLE_LAYOUTS
        for component in layout.components
    ]
    counts = Counter(components)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    missing = sorted(set(HMRC_SPI_INCOME_COMPONENTS) - set(components))
    unexpected = sorted(set(components) - set(HMRC_SPI_INCOME_COMPONENTS))
    if (
        duplicates
        or missing
        or unexpected
        or len(components) != len(HMRC_SPI_INCOME_COMPONENTS)
    ):
        raise ValueError(
            "HMRC SPI table layout must define each required component exactly "
            f"once; duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}."
        )


def _validate_target_surface(
    targets: tuple[HMRCIncomeBandTargetRecord, ...],
) -> None:
    keys = [
        (target.total_income_lower_bound, target.component, target.measure)
        for target in targets
    ]
    counts = Counter(keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate HMRC SPI target record(s): {duplicates}.")

    expected_keys = {
        (lower_bound, component, measure)
        for lower_bound in HMRC_SPI_INCOME_BAND_LOWER_BOUNDS
        for component in HMRC_SPI_INCOME_COMPONENTS
        for measure in ("count", "amount")
    }
    actual_keys = set(keys)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected or len(targets) != HMRC_SPI_TARGET_RECORD_COUNT:
        raise ValueError(
            "HMRC SPI target surface is incomplete; "
            f"expected {HMRC_SPI_TARGET_RECORD_COUNT} records, got "
            f"{len(targets)}, missing={missing}, unexpected={unexpected}."
        )


def _strict_positive_value(
    value: object,
    *,
    sheet_name: str,
    component: str,
    measure: HMRCIncomeMeasure,
    lower_bound: int,
) -> float:
    if isinstance(value, bool):
        numeric = math.nan
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(
            f"{sheet_name} component {component!r} {measure} at lower band "
            f"{lower_bound} must be a finite positive number; got {value!r}."
        )
    return numeric


def _integer_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _target_name(
    component: str,
    measure: HMRCIncomeMeasure,
    lower_bound: int,
    upper_bound: int | None,
) -> str:
    formatted_lower = f"{lower_bound:_}"
    formatted_upper = "inf" if upper_bound is None else f"{upper_bound:_}"
    component_label = component if measure == "amount" else f"{component}_count"
    return f"hmrc/{component_label}_income_band_{formatted_lower}_to_{formatted_upper}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
