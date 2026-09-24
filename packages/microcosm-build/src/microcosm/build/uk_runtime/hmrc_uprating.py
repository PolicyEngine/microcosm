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
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import cache, partial
from importlib.resources import files
from typing import Any

from microcosm.build.ledger_targets import (
    CALENDAR_YEAR_WINDOW_WEIGHTS,
    LedgerTargetReference,
    TargetRegistry,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

UK_ENGINE_PARAMETER_INDEX_PREFIX = "policyengine_uk_parameter:"
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
UK_HMRC_TOTAL_INCOME_GROWTH_INDEX_CONCEPT = (
    "hmrc.itl_2026.total_income_growth_by_total_income_band"
)
UK_HMRC_TOTAL_TAX_GROWTH_INDEX_CONCEPT = (
    "hmrc.itl_2026.total_tax_growth_by_total_income_band"
)
UK_HMRC_TAXPAYER_COUNTS_RESOURCE = "hmrc_itl_taxpayer_counts.json"
UK_HMRC_TAXPAYER_COUNT_CONCEPT = "hmrc.spi_taxpayer_count"
UK_HMRC_TAXPAYER_COUNT_MEASURE_ID = "total_taxpayer_count"
#: Table 2.5 growth indices: the vendored resource carries the three Table 2.5
#: measures for every band and year, and each index concept reads one of them.
UK_HMRC_ITL_GROWTH_MEASURES: Mapping[str, tuple[str, str]] = {
    UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT: (
        "hmrc.spi_taxpayer_count",
        "total_taxpayer_count",
    ),
    UK_HMRC_TOTAL_INCOME_GROWTH_INDEX_CONCEPT: (
        "hmrc.spi_total_income_before_tax_amount",
        "total_income_amount",
    ),
    UK_HMRC_TOTAL_TAX_GROWTH_INDEX_CONCEPT: (
        "hmrc.spi_total_tax_amount",
        "total_tax_amount",
    ),
}
UK_HMRC_ITL_GROWTH_BASIS = (
    "HMRC Income Tax liabilities statistics Table 2.5 {measure} by total-income "
    "band: the calendar-year window of the two tax years overlapping the "
    "calibration year over the SPI fact's year, summed over the Table 2.5 bands "
    "the SPI band spans"
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
#: Every Table 2.5 band edge, for a band the lower-edge table does not carry
#: (the regional Table 3.11 top band is 200,000 and over).
ITL_BANDS: tuple[tuple[int, int | None], ...] = (
    (12_570, 15_000),
    (15_000, 20_000),
    (20_000, 30_000),
    (30_000, 50_000),
    (50_000, 100_000),
    (100_000, 150_000),
    (150_000, 200_000),
    (200_000, 500_000),
    (500_000, 1_000_000),
    (1_000_000, 2_000_000),
    (2_000_000, None),
)


def itl_bands_spanned(
    lower: int, upper: int | None, *, open_top: bool = False
) -> tuple[tuple[int, int | None], ...]:
    """The Table 2.5 bands an SPI band spans, refused on a mismatch.

    A band spelled by its lower edge alone is an SPI Table 3.6/3.7 band and
    takes the declared table; ``open_top`` is a publisher's "and over" band
    and takes every Table 2.5 band from its lower edge up.
    """

    if open_top:
        spanned = tuple(band for band in ITL_BANDS if band[0] >= lower)
        if not spanned or spanned[0][0] != lower:
            raise ValueError(f"no Table 2.5 band opens at {lower}.")
        return spanned
    declared = SPI_BAND_TO_ITL_BANDS.get(lower)
    if upper is None:
        if declared is None:
            raise ValueError(f"no declared Table 2.5 band for the SPI band at {lower}.")
        return declared
    if declared is not None and (declared[-1][1] is None or upper <= declared[-1][1]):
        # An SPI band spelled with both edges (Table 3.3, 3.11) sits inside the
        # Table 2.5 band its lower edge declares.
        return declared
    spanned = tuple(
        band
        for band in ITL_BANDS
        if band[0] >= lower and band[1] is not None and band[1] <= upper
    )
    if not spanned or spanned[0][0] != lower or spanned[-1][1] != upper:
        raise ValueError(f"the SPI band {lower}-{upper} does not tile Table 2.5 bands.")
    return spanned


ParameterValue = Callable[[str, str], float]
_LEDGER_FILTER_PREFIX = "ledger_filter_"
_ANNUAL_PERIOD_TYPES = frozenset(("tax_year", "fiscal_year"))


_UK_PACKAGE = "microcosm.build.uk"
#: The vendored engine values the compile reads instead of the engine: one
#: value per declared parameter per 1 January of ``UK_ENGINE_PIN_YEARS``,
#: written by ``UK_ENGINE_PINS_TOOL`` from the installed policyengine-uk and
#: held in lockstep with it by a ``requires_uk`` test. The compile itself
#: therefore runs, and reproduces, without the engine (the fast CI tiers
#: carry no country package).
UK_ENGINE_PINS_RESOURCE = "hmrc_uprating_engine_pins.json"
UK_ENGINE_PINS_TOOL = "tools/pin_uk_uprating_engine_values.py"
#: The 1 January instants the pins carry: every opening year an SPI or Table
#: 2.5 fact can have, and every calibration year the lane binds.
UK_ENGINE_PIN_YEARS: tuple[int, ...] = tuple(range(2019, 2027))


@cache
def _engine_system() -> Any:
    from policyengine_uk import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


def installed_engine_version() -> str:
    """The policyengine-uk version the environment carries (``absent`` if none)."""

    try:
        return importlib.metadata.version("policyengine-uk")
    except importlib.metadata.PackageNotFoundError:
        return "absent"


@cache
def live_engine_parameter_value(path: str, instant: str) -> float:
    """The installed engine's value of ``path`` at ``instant`` (``YYYY-MM-DD``).

    Only the pin tool and its lockstep test read the engine; the compile reads
    the vendored pins. Cached per (path, instant): the engine materialises a
    whole parameter snapshot per instant, which is seconds of work.
    """

    node = _engine_system().parameters(instant)
    for part in path.split("."):
        node = getattr(node, part)
    return float(node)


@cache
def engine_parameter_pins() -> Mapping[str, Any]:
    """The vendored engine values (``UK_ENGINE_PINS_RESOURCE``), read once."""

    text = (
        files(_UK_PACKAGE).joinpath(UK_ENGINE_PINS_RESOURCE).read_text(encoding="utf-8")
    )
    payload = json.loads(text)
    engine = payload.get("engine") or {}
    if (
        payload.get("schema_version") != 1
        or engine.get("package") != "policyengine-uk"
        or not str(engine.get("version") or "").strip()
        or tuple(payload.get("values") or ()) != UK_ENGINE_INDEX_PARAMETERS
    ):
        raise ValueError(
            f"{UK_ENGINE_PINS_RESOURCE} does not carry schema 1 policyengine-uk pins "
            f"for exactly {list(UK_ENGINE_INDEX_PARAMETERS)}; re-run "
            f"{UK_ENGINE_PINS_TOOL} with the uk extra installed."
        )
    return payload


def engine_parameter_value(path: str, instant: str) -> float:
    """The pinned engine's value of ``path`` at ``instant``, from the vendored pins."""

    try:
        value = float(engine_parameter_pins()["values"][path][instant])
    except KeyError:
        raise ValueError(
            f"policyengine-uk parameter {path!r} at {instant} is not in the "
            f"vendored engine pins ({UK_ENGINE_PINS_RESOURCE}); re-run "
            f"{UK_ENGINE_PINS_TOOL} with the uk extra installed."
        ) from None
    if not value > 0:
        raise ValueError(
            f"policyengine-uk parameter {path!r} is {value!r} at {instant}; an "
            "uprating index must be positive."
        )
    return value


def _engine_version() -> str:
    return str(engine_parameter_pins()["engine"]["version"])


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


def _refuse_fact_after_target(spec: Any, opening_year: int, target_year: int) -> None:
    if opening_year > target_year:
        raise ValueError(
            f"UK target {spec.name!r}: the fact opens in {opening_year}, after the "
            f"calibration year {target_year}; a declared uprating never moves a "
            "value backwards."
        )


def _identity_aligned(
    spec: Any, reference: LedgerTargetReference, opening_year: int
) -> Any:
    """A fact that opens in the calibration year binds as published, receipted.

    The production-2023 parity surface compiles the same references at their
    own period; the declared index then reads as an identity, so the row stays
    on the surface with its factor written rather than dropping out.
    """

    metadata = {
        **spec.metadata,
        "uprating_index": str(reference.uprating_index),
        "uprating_index_basis": "identity: the fact opens in the calibration year",
        "uprating_index_from_instant": f"{opening_year}-01-01",
        "uprating_index_to_instant": f"{opening_year}-01-01",
        "uprating_factor": "1",
        "ledger_value_before_alignment": f"{spec.value:.15g}",
        "uprating_adjudication": UK_INCOME_UPRATING_ADJUDICATION,
    }
    return replace(spec, metadata=metadata)


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
        _refuse_fact_after_target(spec, opening_year, target_year)
        if opening_year == target_year:
            aligned.append(_identity_aligned(spec, reference, opening_year))
            continue
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


@cache
def _vendored_itl_rows(concept: str) -> tuple[Mapping[str, Any], ...]:
    """The vendored Table 2.5 rows of one concept, read (and pin-checked) once."""

    return tuple(
        vendored_rows(
            UK_HMRC_TAXPAYER_COUNTS_RESOURCE, concept=concept, period_type="tax_year"
        )
    )


_BAND_VALUE_ID = re.compile(r"(?:^|_)band_(\d+)(?:_(\d+|plus))?$")


def _spi_band_edges(
    spec: Any, reference: LedgerTargetReference
) -> tuple[int, int | None, bool]:
    """``(lower, upper, open)`` of the row's band.

    The publisher's value id says which universe the row belongs to: the SPI
    band tables spell a band by its lower edge alone (``band_200000``, upper
    edge implied by the next band), the liabilities and regional tables spell
    both edges (``band_200000_300000``) or an open top (``band_200000_plus``).
    The ledger filters supply the lower edge when the id carries none.
    """

    lower = _spi_band_lower_edge(spec, reference)
    value_id = str(spec.metadata.get("ledger_layout_groupby_value_id") or "")
    match = _BAND_VALUE_ID.search(value_id)
    if match is None or match.group(2) is None:
        return lower, None, False
    if match.group(2) == "plus":
        return lower, None, True
    return lower, int(match.group(2)), False


def _itl_value(
    rows: list[Mapping[str, Any]],
    *,
    measure_id: str,
    band: tuple[int, int | None],
    opening_year: int,
    spec_name: str,
) -> tuple[float, str]:
    lower, upper = band
    matches = [
        row
        for row in rows
        if str(row.get("measure_id")) == measure_id
        and int((row.get("period") or {}).get("value", -1)) == opening_year
        and (row.get("dimensions") or {}).get("total_income_lower_bound") == lower
        and (row.get("dimensions") or {}).get("total_income_upper_bound") == upper
    ]
    if len(matches) != 1:
        raise ValueError(
            f"UK target {spec_name!r}: Table 2.5 {measure_id} for band "
            f"{lower}-{upper if upper is not None else 'inf'} opening in "
            f"{opening_year} matched {len(matches)} vendored rows; expected one."
        )
    (row,) = matches
    value = float(row["value"])
    if not value > 0:
        raise ValueError(
            f"UK target {spec_name!r}: Table 2.5 {measure_id} is {value!r}."
        )
    return value, str(row.get("source_record_id") or "")


def align_hmrc_row_by_itl_growth(
    reference: LedgerTargetReference,
    registry: TargetRegistry,
    *,
    index_concept: str,
    rows: list[Mapping[str, Any]] | None = None,
) -> TargetRegistry:
    """Move an SPI band row by HMRC's projected growth of a Table 2.5 measure.

    factor = window(measure in target-1, measure in target) / measure in the
    fact's opening year, where the window is the calendar-year weighting of
    ``ledger_targets.CALENDAR_YEAR_WINDOW_WEIGHTS`` and the measure is summed
    over the Table 2.5 bands the SPI band spans. Counts follow taxpayer
    numbers; total income and tax rows (the regional Table 3.11 anchors)
    follow HMRC's projected total income and liabilities in the band.
    """

    if reference.uprating_index != index_concept:
        return registry
    concept, measure_id = UK_HMRC_ITL_GROWTH_MEASURES[index_concept]
    target_year = _target_year(reference)
    rows = list(_vendored_itl_rows(concept)) if rows is None else rows
    aligned = []
    for spec in registry.specs:
        opening_year = _opening_year(spec, reference)
        _refuse_fact_after_target(spec, opening_year, target_year)
        if opening_year == target_year:
            aligned.append(_identity_aligned(spec, reference, opening_year))
            continue
        lower_edge, upper_edge, open_top = _spi_band_edges(spec, reference)
        try:
            bands = itl_bands_spanned(lower_edge, upper_edge, open_top=open_top)
        except ValueError as error:
            raise ValueError(f"UK target {spec.name!r}: {error}") from error
        years = {opening_year: 0.0}
        record_ids: list[str] = []
        for offset in CALENDAR_YEAR_WINDOW_WEIGHTS:
            years[target_year + offset] = 0.0
        for band in bands:
            for year in years:
                value, record_id = _itl_value(
                    rows,
                    measure_id=measure_id,
                    band=band,
                    opening_year=year,
                    spec_name=spec.name,
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
            "uprating_index": index_concept,
            "uprating_index_basis": UK_HMRC_ITL_GROWTH_BASIS.format(measure=measure_id),
            "uprating_index_resource": UK_HMRC_TAXPAYER_COUNTS_RESOURCE,
            "uprating_index_measure_id": measure_id,
            "uprating_index_itl_bands": ";".join(
                f"{lower}-{upper if upper is not None else 'inf'}"
                for lower, upper in bands
            ),
            "uprating_index_values_by_opening_year": ";".join(
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
    for index_concept in UK_HMRC_ITL_GROWTH_MEASURES:
        appliers[index_concept] = partial(
            align_hmrc_row_by_itl_growth, index_concept=index_concept
        )
    return appliers


def align_hmrc_count_row_by_taxpayer_growth(
    reference: LedgerTargetReference,
    registry: TargetRegistry,
    *,
    count_rows: list[Mapping[str, Any]] | None = None,
) -> TargetRegistry:
    """The taxpayer-count index, kept under its own name for the SPI count rows."""

    return align_hmrc_row_by_itl_growth(
        reference,
        registry,
        index_concept=UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT,
        rows=count_rows,
    )


__all__ = [
    "SPI_BAND_TO_ITL_BANDS",
    "UK_ENGINE_INDEX_CONCEPTS",
    "UK_ENGINE_INDEX_PARAMETERS",
    "UK_ENGINE_PARAMETER_INDEX_PREFIX",
    "UK_HMRC_TAXPAYER_COUNTS_RESOURCE",
    "UK_HMRC_ITL_GROWTH_MEASURES",
    "UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT",
    "UK_HMRC_TOTAL_INCOME_GROWTH_INDEX_CONCEPT",
    "UK_HMRC_TOTAL_TAX_GROWTH_INDEX_CONCEPT",
    "align_hmrc_count_row_by_taxpayer_growth",
    "align_hmrc_row_by_engine_index",
    "align_hmrc_row_by_itl_growth",
    "itl_bands_spanned",
    "engine_parameter_pins",
    "engine_parameter_value",
    "installed_engine_version",
    "live_engine_parameter_value",
    "hmrc_uprating_appliers",
]
