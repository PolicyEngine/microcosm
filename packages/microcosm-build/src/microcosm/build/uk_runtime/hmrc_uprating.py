"""Declared uprating of HMRC SPI band rows to the UK calibration year.

The SPI income-by-band facts (Tables 3.6 and 3.7, tax year 2023-24) are the
only published distribution of taxable income by component, and the
calibration binds them at period 2025. Bound as published they hold 2023-24
nominal levels against a spine the engine has already uprated to 2025, which
the v20 national build measured as a 13 percent income-tax shortfall and
state-pension bands 8 to 18 percent over (microcosm#280 lane, assessment of
2026-09-21). Every reference in the family now declares an ``uprating_index``
and one of the two appliers below transports its value:

* amount rows: the ratio of a policyengine-uk parameter between the fact's
  opening-year instant and the calibration year's, the same calendar-year
  instants the SPI donor rebasing uses (``spi_income._spi_income_uprating_factors``).
  The OBR indices are calendar-year series keyed at 1 January, so the engine's
  period 2025 is calendar 2025 for incomes; the state pension rate is the
  tax-year rate the engine pays for the whole year, so its rows move by that
  rate's ratio;
* count rows: HMRC's own projection of taxpayer numbers by total-income band
  (Income Tax liabilities statistics, Table 2.5), as the calendar-year window
  of the two overlapping tax years over the fact's year, read from the
  vendored ``hmrc_itl_taxpayer_counts.json`` resource so the factor is
  refused when the resource lags the feed pin.

María's ruling of 2026-09-22: the calibration binds at calendar 2025; facts
published by fiscal or tax year use the months to the end of that year.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import cache, partial
from typing import Any

from microcosm.build.ledger_targets import (
    CALENDAR_YEAR_WINDOW_WEIGHTS,
    LedgerTargetReference,
    TargetRegistry,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

UK_ENGINE_PARAMETER_INDEX_PREFIX = "policyengine_uk.parameter:"
UK_ENGINE_INDEX_BASIS = (
    "policyengine-uk parameter ratio between 1 January of the fact's opening "
    "year and 1 January of the calibration year (the SPI donor rebasing convention)"
)

#: The parameter paths a UK reference may name behind the prefix. One per SPI
#: income component the family binds, each the index the engine itself uprates
#: that variable with, plus the new State Pension weekly rate for the state
#: pension rows (the engine pays the tax-year rate for the whole period).
UK_ENGINE_INDEX_PARAMETERS: tuple[str, ...] = (
    "gov.economic_assumptions.indices.obr.average_earnings",
    "gov.economic_assumptions.indices.obr.per_capita.mixed_income",
    "gov.economic_assumptions.indices.obr.per_capita.gdp",
    "gov.economic_assumptions.indices.obr.private_pension_index",
    "gov.economic_assumptions.indices.ons.household_interest_income",
    "gov.dwp.state_pension.new_state_pension.amount",
)
UK_ENGINE_INDEX_CONCEPTS: tuple[str, ...] = tuple(
    f"{UK_ENGINE_PARAMETER_INDEX_PREFIX}{path}" for path in UK_ENGINE_INDEX_PARAMETERS
)

UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT = (
    "hmrc.itl_2026.taxpayer_count_growth_by_total_income_band"
)
UK_HMRC_TAXPAYER_COUNTS_RESOURCE = "hmrc_itl_taxpayer_counts.json"
UK_HMRC_TAXPAYER_COUNT_CONCEPT = "hmrc.spi_taxpayer_count"
UK_HMRC_TAXPAYER_COUNT_MEASURE_ID = "total_taxpayer_count"
UK_HMRC_TAXPAYER_GROWTH_BASIS = (
    "HMRC Income Tax liabilities statistics Table 2.5 taxpayers by total-income "
    "band: the calendar-year window of the two tax years overlapping the "
    "calibration year over the SPI fact's year, in the Table 2.5 band that "
    "contains the SPI band"
)
UK_INCOME_UPRATING_ADJUDICATION = (
    "microcosm#280 lane (María, 2026-09-22): the calibration binds at calendar "
    "2025; facts published by fiscal or tax year use the months to the end of "
    "that year; policy amounts move by the rate the engine pays at the period"
)

#: SPI Table 3.6/3.7 band lower edge -> the Table 2.5 bands (lower, upper) it
#: sits in. Table 2.5 merges 30-40k with 40-50k, 50-70k with 70-100k and
#: 200-300k with 300-500k, and splits 1m+ into 1m-2m and 2m+ (the SPI band
#: takes both). Upper edges are the publisher's; None is open-ended.
SPI_BAND_TO_ITL_BANDS: Mapping[int, tuple[tuple[int, int | None], ...]] = {
    12_570: ((12_570, 15_000),),
    15_000: ((15_000, 20_000),),
    20_000: ((20_000, 30_000),),
    30_000: ((30_000, 50_000),),
    40_000: ((30_000, 50_000),),
    50_000: ((50_000, 100_000),),
    70_000: ((50_000, 100_000),),
    100_000: ((100_000, 150_000),),
    150_000: ((150_000, 200_000),),
    200_000: ((200_000, 500_000),),
    300_000: ((200_000, 500_000),),
    500_000: ((500_000, 1_000_000),),
    1_000_000: ((1_000_000, 2_000_000), (2_000_000, None)),
}

ParameterValue = Callable[[str, str], float]
_LEDGER_FILTER_PREFIX = "ledger_filter_"
_ANNUAL_PERIOD_TYPES = frozenset(("tax_year", "fiscal_year"))


@cache
def _engine_system() -> Any:
    from policyengine_uk import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


def engine_parameter_value(path: str, instant: str) -> float:
    """The pinned engine's value of ``path`` at ``instant`` (``YYYY-MM-DD``)."""

    node = _engine_system().parameters(instant)
    for part in path.split("."):
        node = getattr(node, part)
    value = float(node)
    if not value > 0:
        raise ValueError(
            f"policyengine-uk parameter {path!r} is {value!r} at {instant}; an "
            "uprating index must be positive."
        )
    return value


def _engine_version() -> str:
    try:
        return importlib.metadata.version("policyengine-uk")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - refused below
        return "unknown"


def _opening_year(spec: Any, reference: LedgerTargetReference) -> int:
    period_type = str(spec.metadata.get("ledger_period_type") or "")
    if period_type not in _ANNUAL_PERIOD_TYPES:
        raise ValueError(
            f"UK target {spec.name!r}: uprating_index {reference.uprating_index!r} "
            f"transports fiscal- or tax-year facts; the resolved fact is a "
            f"{period_type or 'untyped'} period."
        )
    raw = str(spec.metadata.get("ledger_fact_period") or "")
    if not raw[:4].isdigit():
        raise ValueError(
            f"UK target {spec.name!r}: cannot read an opening year from the "
            f"resolved fact period {raw!r}."
        )
    return int(raw[:4])


def _target_year(reference: LedgerTargetReference) -> int:
    period = str(reference.period or "")
    if not period[:4].isdigit():
        raise ValueError(
            f"UK reference {reference.name!r}: a declared uprating needs a "
            "calendar-year target period."
        )
    return int(period[:4])


def align_hmrc_row_by_engine_index(
    reference: LedgerTargetReference,
    registry: TargetRegistry,
    *,
    parameter_path: str,
    parameter_value: ParameterValue = engine_parameter_value,
) -> TargetRegistry:
    """Move an amount row from its opening year to the calibration year.

    The factor is ``parameter(target-01-01) / parameter(opening-01-01)``; both
    values, the instants, the engine version and the pre-alignment value are
    written on the spec so the receipt reproduces without the engine.
    """

    expected = f"{UK_ENGINE_PARAMETER_INDEX_PREFIX}{parameter_path}"
    if reference.uprating_index != expected:
        return registry
    target_year = _target_year(reference)
    aligned = []
    for spec in registry.specs:
        opening_year = _opening_year(spec, reference)
        if opening_year >= target_year:
            raise ValueError(
                f"UK target {spec.name!r}: the fact opens in {opening_year}, not "
                f"before the calibration year {target_year}; nothing to uprate."
            )
        from_instant = f"{opening_year}-01-01"
        to_instant = f"{target_year}-01-01"
        from_value = parameter_value(parameter_path, from_instant)
        to_value = parameter_value(parameter_path, to_instant)
        factor = to_value / from_value
        metadata = {
            **spec.metadata,
            "uprating_index": expected,
            "uprating_index_basis": UK_ENGINE_INDEX_BASIS,
            "uprating_index_parameter": parameter_path,
            "uprating_index_engine": f"policyengine-uk {_engine_version()}",
            "uprating_index_from_instant": from_instant,
            "uprating_index_to_instant": to_instant,
            "uprating_index_from_value": f"{from_value:.15g}",
            "uprating_index_to_value": f"{to_value:.15g}",
            "uprating_factor": f"{factor:.15g}",
            "ledger_value_before_alignment": f"{spec.value:.15g}",
            "uprating_adjudication": UK_INCOME_UPRATING_ADJUDICATION,
        }
        aligned.append(replace(spec, value=spec.value * factor, metadata=metadata))
    return TargetRegistry(aligned, country="uk")


def _spi_band_lower_edge(spec: Any, reference: LedgerTargetReference) -> int:
    key = f"{_LEDGER_FILTER_PREFIX}total_income_lower_bound"
    raw = str(spec.metadata.get(key) or "").replace(",", "")
    if not raw.isdigit():
        raise ValueError(
            f"UK target {spec.name!r}: uprating_index {reference.uprating_index!r} "
            "needs the SPI total-income band lower bound on the spec "
            f"({key}); found {raw!r}."
        )
    return int(raw)


def _vendored_taxpayer_count_rows() -> list[Mapping[str, Any]]:
    return vendored_rows(
        UK_HMRC_TAXPAYER_COUNTS_RESOURCE,
        concept=UK_HMRC_TAXPAYER_COUNT_CONCEPT,
        period_type="tax_year",
    )


def _taxpayer_count(
    rows: list[Mapping[str, Any]],
    *,
    band: tuple[int, int | None],
    opening_year: int,
    spec_name: str,
) -> tuple[float, str]:
    lower, upper = band
    matches = [
        row
        for row in rows
        if str(row.get("measure_id")) == UK_HMRC_TAXPAYER_COUNT_MEASURE_ID
        and int((row.get("period") or {}).get("value", -1)) == opening_year
        and (row.get("dimensions") or {}).get("total_income_lower_bound") == lower
        and (row.get("dimensions") or {}).get("total_income_upper_bound") == upper
    ]
    if len(matches) != 1:
        raise ValueError(
            f"UK target {spec_name!r}: Table 2.5 taxpayer count for band "
            f"{lower}-{upper if upper is not None else 'inf'} opening in "
            f"{opening_year} matched {len(matches)} vendored rows; expected one."
        )
    (row,) = matches
    value = float(row["value"])
    if not value > 0:
        raise ValueError(
            f"UK target {spec_name!r}: Table 2.5 taxpayer count is {value!r}."
        )
    return value, str(row.get("source_record_id") or "")


def align_hmrc_count_row_by_taxpayer_growth(
    reference: LedgerTargetReference,
    registry: TargetRegistry,
    *,
    count_rows: list[Mapping[str, Any]] | None = None,
) -> TargetRegistry:
    """Move an SPI count row by HMRC's projected taxpayer growth in its band.

    factor = window(count in target-1, count in target) / count in the
    fact's opening year, where the window is the calendar-year weighting of
    ``ledger_targets.CALENDAR_YEAR_WINDOW_WEIGHTS`` and the counts are the
    Table 2.5 band containing the SPI band (summed where the SPI band spans
    two Table 2.5 bands).
    """

    if reference.uprating_index != UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT:
        return registry
    target_year = _target_year(reference)
    rows = _vendored_taxpayer_count_rows() if count_rows is None else count_rows
    aligned = []
    for spec in registry.specs:
        opening_year = _opening_year(spec, reference)
        if opening_year >= target_year:
            raise ValueError(
                f"UK target {spec.name!r}: the fact opens in {opening_year}, not "
                f"before the calibration year {target_year}; nothing to uprate."
            )
        lower_edge = _spi_band_lower_edge(spec, reference)
        bands = SPI_BAND_TO_ITL_BANDS.get(lower_edge)
        if bands is None:
            raise ValueError(
                f"UK target {spec.name!r}: SPI band lower edge {lower_edge} has no "
                "declared Table 2.5 band."
            )
        years = {opening_year: 0.0}
        record_ids: list[str] = []
        for offset in CALENDAR_YEAR_WINDOW_WEIGHTS:
            years[target_year + offset] = 0.0
        for band in bands:
            for year in years:
                value, record_id = _taxpayer_count(
                    rows, band=band, opening_year=year, spec_name=spec.name
                )
                years[year] += value
                record_ids.append(record_id)
        window = sum(
            weight * years[target_year + offset]
            for offset, weight in CALENDAR_YEAR_WINDOW_WEIGHTS.items()
        )
        factor = window / years[opening_year]
        metadata = {
            **spec.metadata,
            "uprating_index": UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT,
            "uprating_index_basis": UK_HMRC_TAXPAYER_GROWTH_BASIS,
            "uprating_index_resource": UK_HMRC_TAXPAYER_COUNTS_RESOURCE,
            "uprating_index_itl_bands": ";".join(
                f"{lower}-{upper if upper is not None else 'inf'}"
                for lower, upper in bands
            ),
            "uprating_index_counts_by_opening_year": ";".join(
                f"{year}={years[year]:.15g}" for year in sorted(years)
            ),
            "uprating_index_window_weights": ";".join(
                f"{target_year + offset}={weight:.15g}"
                for offset, weight in sorted(CALENDAR_YEAR_WINDOW_WEIGHTS.items())
            ),
            "uprating_index_source_record_ids": ",".join(record_ids),
            "uprating_factor": f"{factor:.15g}",
            "ledger_value_before_alignment": f"{spec.value:.15g}",
            "uprating_adjudication": UK_INCOME_UPRATING_ADJUDICATION,
        }
        aligned.append(replace(spec, value=spec.value * factor, metadata=metadata))
    return TargetRegistry(aligned, country="uk")


def hmrc_uprating_appliers() -> dict[str, Any]:
    """The appliers this module contributes to ``UK_UPRATING_APPLIERS``."""

    appliers: dict[str, Any] = {
        f"{UK_ENGINE_PARAMETER_INDEX_PREFIX}{path}": partial(
            align_hmrc_row_by_engine_index, parameter_path=path
        )
        for path in UK_ENGINE_INDEX_PARAMETERS
    }
    appliers[UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT] = (
        align_hmrc_count_row_by_taxpayer_growth
    )
    return appliers


__all__ = [
    "SPI_BAND_TO_ITL_BANDS",
    "UK_ENGINE_INDEX_CONCEPTS",
    "UK_ENGINE_INDEX_PARAMETERS",
    "UK_ENGINE_PARAMETER_INDEX_PREFIX",
    "UK_HMRC_TAXPAYER_COUNTS_RESOURCE",
    "UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT",
    "align_hmrc_count_row_by_taxpayer_growth",
    "align_hmrc_row_by_engine_index",
    "engine_parameter_value",
    "hmrc_uprating_appliers",
]
