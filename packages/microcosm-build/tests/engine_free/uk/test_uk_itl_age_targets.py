"""HMRC Income Tax payers by age: ITL Tables 2.1 and 2.2 (microcosm#1069).

The two ``hmrc_itl_age`` targets bind Table 2.2's twelve region-tier areas,
and the SPI State Pension band rows move by the growth of that level over
Table 2.1's United Kingdom count. These tests hold the vendored rows to the
publisher's own accounting and the contract to that design.
"""

from __future__ import annotations

import json
import math
from importlib.resources import files

import pytest

from microcosm.build.ledger_targets import CALENDAR_YEAR_WINDOW_WEIGHTS
from microcosm.build.uk_runtime.hmrc_uprating import (
    UK_HMRC_ITL_UK_GEOGRAPHY_ID,
    UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT,
    UK_HMRC_TAXPAYER_COUNT_CONCEPT,
    UK_HMRC_TAXPAYER_COUNTS_RESOURCE,
    UK_SPI_STATE_PENSION_AGE_AMOUNT_INDEX_CONCEPT,
    align_hmrc_row_by_resident_state_pension_age_growth,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.calibrate.geography_constants import UK_REGION_TIER
from test_support.microcosm_build.uk_hmrc_uprating import (
    TargetRegistry,
    _reference,
    _spec,
)

NATIONS = ("E92000001", "W92000004", "S92000003", "N92000002")
ENGLISH_REGIONS = tuple(code for level, code in UK_REGION_TIER if level == "region")
REGION_TIER = tuple(code for _, code in UK_REGION_TIER)
MEASURES = ("state_pension_age", "age_65_and_over")
YEARS = (2023, 2024, 2025, 2026)


def _table() -> dict[tuple[str, str, int], float]:
    table: dict[tuple[str, str, int], float] = {}
    for row in vendored_rows(
        UK_HMRC_TAXPAYER_COUNTS_RESOURCE,
        concept=UK_HMRC_TAXPAYER_COUNT_CONCEPT,
        period_type="tax_year",
    ):
        if row["measure_id"] not in MEASURES:
            continue
        key = (row["measure_id"], row["geography"]["id"], row["period"]["value"])
        assert key not in table, key
        table[key] = float(row["value"])
    return table


def _half_unit(value: float) -> float:
    """Half the last published unit of a figure given to three significant
    figures, the most a rounded figure can be off by."""

    return 0.5 * 10 ** (math.floor(math.log10(value)) - 2)


def _window(table, measure: str, geography: str, target_year: int = 2025) -> float:
    return sum(
        weight * table[(measure, geography, target_year + offset)]
        for offset, weight in CALENDAR_YEAR_WINDOW_WEIGHTS.items()
    )


def _targets() -> dict[str, dict]:
    contract = json.loads(
        files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text(encoding="utf-8")
    )
    return {target["target_id"]: target for target in contract["targets"]}


def test_the_vendored_age_rows_cover_every_area_year_and_measure() -> None:
    table = _table()
    areas = {UK_HMRC_ITL_UK_GEOGRAPHY_ID, *NATIONS, *REGION_TIER}
    assert set(table) == {
        (measure, area, year)
        for measure in MEASURES
        for area in areas
        for year in YEARS
    }
    # The published figures are in thousands to three significant figures,
    # the premise of the rounding tolerance below.
    for value in table.values():
        assert value > 0
        assert value % (2 * _half_unit(value)) == 0


@pytest.mark.parametrize("year", YEARS)
@pytest.mark.parametrize("measure", MEASURES)
def test_the_nations_and_the_region_tier_sum_to_at_most_the_united_kingdom(
    measure: str, year: int
) -> None:
    """Table 2.2 leaves out taxpayers who live abroad or whose region is not
    known, whom Table 2.1's United Kingdom column counts, so the bound areas
    never sum to more than the United Kingdom row; England is its nine
    regions, to the rounding of each published figure."""

    table = _table()
    united_kingdom = table[(measure, UK_HMRC_ITL_UK_GEOGRAPHY_ID, year)]
    assert sum(table[(measure, nation, year)] for nation in NATIONS) <= united_kingdom
    assert sum(table[(measure, area, year)] for area in REGION_TIER) <= united_kingdom
    england = table[(measure, "E92000001", year)]
    regions = [table[(measure, region, year)] for region in ENGLISH_REGIONS]
    tolerance = _half_unit(england) + sum(map(_half_unit, regions))
    assert abs(sum(regions) - england) <= tolerance


def test_taxpayers_of_state_pension_age_are_among_those_aged_65_and_over() -> None:
    # State Pension age is 66 from October 2020 and rises to 67 from 2026.
    table = _table()
    for (measure, area, year), value in table.items():
        if measure == "state_pension_age":
            assert value <= table[("age_65_and_over", area, year)], (area, year)


def test_the_spi_state_pension_rows_move_by_the_resident_level_over_the_uk_count() -> (
    None
):
    """The calendar-2025 resident level over 2023-24's United Kingdom count:
    8.84175m / 8.16m, United Kingdom growth 9.005m / 8.16m times the resident
    share 8.84175m / 9.005m."""

    table = _table()
    opening = table[("state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID, 2023)]
    resident = sum(_window(table, "state_pension_age", area) for area in REGION_TIER)
    united_kingdom = _window(table, "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID)
    assert opening == 8_160_000
    assert resident == pytest.approx(8_841_750)
    assert united_kingdom == pytest.approx(9_005_000)
    assert resident < united_kingdom

    registry = TargetRegistry(
        (
            _spec(
                name="hmrc/state_pension_count_income_band_x",
                value=1.0,
                lower_bound=12_570,
            ),
        ),
        country="uk",
    )
    (spec,) = align_hmrc_row_by_resident_state_pension_age_growth(
        _reference(UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT), registry
    ).specs
    assert float(spec.metadata["uprating_factor"]) == pytest.approx(resident / opening)
    assert float(spec.metadata["uprating_index_united_kingdom_growth"]) == (
        pytest.approx(united_kingdom / opening)
    )
    assert float(spec.metadata["uprating_index_resident_share"]) == pytest.approx(
        resident / united_kingdom
    )


@pytest.mark.parametrize(
    ("target_id", "measure_id", "band", "binding_filter"),
    [
        (
            "hmrc.itl.state_pension_age_taxpayers",
            "state_pension_age",
            "state_pension_age",
            {"variable": "is_SP_age", "equals": True},
        ),
        (
            "hmrc.itl.taxpayers_aged_65_and_over",
            "age_65_and_over",
            "65_and_over",
            {"variable": "age", "operator": ">=", "value": 65},
        ),
    ],
)
def test_the_age_targets_bind_the_region_tier_at_the_calendar_2025_window(
    target_id: str, measure_id: str, band: str, binding_filter: dict
) -> None:
    target = _targets()[target_id]
    assert target["family"] == "hmrc_itl_age"
    assert target["geography_levels"] == ["country", "region"]
    assert target["value_operation"] == "calendar_year_window"
    assert target["assertion_policy"] == "allow_source_projection"
    selector = target["ledger_selector"]
    assert selector["source_concept"] == UK_HMRC_TAXPAYER_COUNT_CONCEPT
    assert selector["source_measure_id"] == measure_id
    assert selector["dimension_values"] == {"age_band": band}
    binding = target["bindings"]["policyengine"]
    assert binding["value_variable"] == "person_count"
    # Taxpayers on the tax measure the SPI and Income Tax liabilities rows use.
    assert binding["filters"] == [
        {
            "variable": "income_tax_before_winter_fuel_payment_charge_annual",
            "operator": ">",
            "value": 0,
        },
        binding_filter,
    ]
    assert not any(
        str(predicate.get("variable")) == "country" for predicate in binding["filters"]
    )


def test_the_spi_state_pension_rows_declare_the_state_pension_age_indices() -> None:
    targets = _targets()
    count = targets["hmrc.spi.state_pension.count_by_total_income_band"]
    amount = targets["hmrc.spi.state_pension.amount_by_total_income_band"]
    assert count["uprating_index"] == UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT
    assert amount["uprating_index"] == UK_SPI_STATE_PENSION_AGE_AMOUNT_INDEX_CONCEPT
    # The private pension rows keep Table 2.5's band growth: their recipients
    # are not one age group HMRC projects.
    assert (
        targets["hmrc.spi.private_pension_income.count_by_total_income_band"][
            "uprating_index"
        ]
        == "hmrc.itl_2026.taxpayer_count_growth_by_total_income_band"
    )


def test_the_parity_receipt_routes_only_the_age_and_state_pension_band_rows() -> None:
    from tools import build_uk_ledger_compile_parity_signed_differences as parity_tool

    names = {
        "hmrc.itl.state_pension_age_taxpayers@E12000001": "ledger_only",
        "hmrc.itl.taxpayers_aged_65_and_over@N92000002": "ledger_only",
        "hmrc/state_pension_count_income_band_12_570_to_15_000": "calibration_drift",
        "hmrc/state_pension_income_band_100_000_to_150_000": "calibration_drift",
        # Look-alikes that keep the general rationales.
        "hmrc/state_pension_credit_count_income_band_12_570_to_15_000": (
            "calibration_drift"
        ),
        "hmrc/private_pension_income_count_income_band_12_570_to_15_000": (
            "calibration_drift"
        ),
        "hmrc.itl.higher_rate_taxpayers": "ledger_only",
    }
    report = {"differences": [{"name": n, "kind": k} for n, k in names.items()]}
    parity_tool._add_signed_rationale_notes(
        report, fixture_resource="registry_parity_fixture_2025.json"
    )
    reasons = {row["name"]: row.get("reason") for row in report["differences"]}
    age = parity_tool._ITL_AGE_LEDGER_ONLY_RATIONALE
    state_pension = parity_tool._SPI_STATE_PENSION_UPRATED_DRIFT_RATIONALE
    assert reasons["hmrc.itl.state_pension_age_taxpayers@E12000001"] == age
    assert reasons["hmrc.itl.taxpayers_aged_65_and_over@N92000002"] == age
    assert (
        reasons["hmrc/state_pension_count_income_band_12_570_to_15_000"]
        == state_pension
    )
    assert reasons["hmrc/state_pension_income_band_100_000_to_150_000"] == (
        state_pension
    )
    for name in (
        "hmrc/state_pension_credit_count_income_band_12_570_to_15_000",
        "hmrc/private_pension_income_count_income_band_12_570_to_15_000",
    ):
        assert reasons[name] == parity_tool._SPI_UPRATED_DRIFT_RATIONALE
    assert reasons["hmrc.itl.higher_rate_taxpayers"] not in (age, state_pension)
