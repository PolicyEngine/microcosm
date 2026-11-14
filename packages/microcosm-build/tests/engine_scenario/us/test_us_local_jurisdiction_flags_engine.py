"""Local taxes reach households placed through the block-ladder line.

Synthetic households are pinned to fixture blocks in Yonkers, Wilmington,
Denver and Philadelphia (and a neighbour of each), placed by
``with_household_us_geography_ladder``, and run through
``PolicyEngineUSEngine.materialize``. Each local tax or benefit is nonzero
for the resident and zero (or at the nonresident rate) for the neighbour,
and stripping the derived flags returns the resident to the neighbour's
result, so the flags, not other geography, drive the difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.us_runtime.geography_ladder import (
    with_household_us_geography_ladder,
)
from microcosm.frame import Frame
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine
from test_support.microcosm_build.us_local_jurisdictions import (
    block_ladder,
    households_on_blocks,
    us_frame,
)

PERIOD = 2026
FLAGS = {
    "household": ("in_denver", "in_wilmington", "in_yonkers"),
    "person": ("pa_philadelphia_wage_tax_resident",),
}
# Every person carries every input column, so no table holds a null the
# engine would refuse.
NO_INPUTS = {
    "age": 40,
    "employment_income": 0.0,
    "rent": 0.0,
    "social_security_retirement": 0.0,
    "pa_philadelphia_wage_tax_taxable_wages": 0.0,
}
WORKER = {**NO_INPUTS, "employment_income": 100_000.0}
PHILADELPHIA_WORKER = {
    **NO_INPUTS,
    "employment_income": 50_000.0,
    "pa_philadelphia_wage_tax_taxable_wages": 50_000.0,
}
# Elderly low-income renter: Denver's renter relief needs rent, an elderly or
# disabled member, and income under the AMI-based limit.
DENVER_RENTER = {
    **NO_INPUTS,
    "age": 72,
    "rent": 14_400.0,
    "social_security_retirement": 9_000.0,
}
CASES = (
    ("yonkers", WORKER),
    ("new_rochelle_westchester", WORKER),
    ("wilmington", WORKER),
    ("new_castle_unincorporated", WORKER),
    ("denver", DENVER_RENTER),
    ("aurora_arapahoe", DENVER_RENTER),
    ("philadelphia", PHILADELPHIA_WORKER),
    ("delaware_county_pa", PHILADELPHIA_WORKER),
)
ROW = {name: position for position, (name, _) in enumerate(CASES)}
OUTPUTS = (
    "ny_yonkers_income_tax",
    "de_wilmington_earned_income_tax",
    "co_denver_property_tax_relief_renter_eligible",
    "co_denver_property_tax_relief",
    "pa_philadelphia_wage_tax",
    "in_denver",
    "in_wilmington",
    "in_yonkers",
    "pa_philadelphia_wage_tax_resident",
)


@pytest.fixture(scope="module")
def placed() -> Frame:
    frame = us_frame(
        households_on_blocks([name for name, _ in CASES]),
        person_inputs=[[dict(person)] for _, person in CASES],
    )
    return with_household_us_geography_ladder(frame, block_ladder(), seed=0)


def _without_flags(frame: Frame) -> Frame:
    tables = {entity: frame.table(entity) for entity in frame.entities}
    for entity, flags in FLAGS.items():
        tables[entity] = tables[entity].drop(columns=list(flags))
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
    )


@pytest.fixture(scope="module")
def engine() -> PolicyEngineUSEngine:
    return PolicyEngineUSEngine()


@pytest.fixture(scope="module")
def with_flags(engine, placed) -> dict[str, np.ndarray]:
    return dict(engine.materialize(placed, OUTPUTS, PERIOD))


@pytest.fixture(scope="module")
def without_flags(engine, placed) -> dict[str, np.ndarray]:
    return dict(engine.materialize(_without_flags(placed), OUTPUTS, PERIOD))


def test_the_engine_loads_the_derived_flags(with_flags) -> None:
    for flag, resident in (
        ("in_yonkers", "yonkers"),
        ("in_wilmington", "wilmington"),
        ("in_denver", "denver"),
        ("pa_philadelphia_wage_tax_resident", "philadelphia"),
    ):
        assert np.flatnonzero(with_flags[flag]).tolist() == [ROW[resident]], flag


def test_yonkers_resident_surcharge_is_nonzero_only_in_yonkers(
    with_flags, without_flags
) -> None:
    tax = with_flags["ny_yonkers_income_tax"]

    assert tax[ROW["yonkers"]] > 0
    assert tax[ROW["new_rochelle_westchester"]] == 0
    assert without_flags["ny_yonkers_income_tax"][ROW["yonkers"]] == 0


def test_wilmington_earned_income_tax_is_nonzero_only_in_wilmington(
    with_flags, without_flags
) -> None:
    tax = with_flags["de_wilmington_earned_income_tax"]

    assert tax[ROW["wilmington"]] > 0
    assert tax[ROW["new_castle_unincorporated"]] == 0
    assert without_flags["de_wilmington_earned_income_tax"][ROW["wilmington"]] == 0


def test_denver_renter_relief_reaches_only_the_denver_renter(
    with_flags, without_flags
) -> None:
    eligible = with_flags["co_denver_property_tax_relief_renter_eligible"]
    relief = with_flags["co_denver_property_tax_relief"]

    assert eligible[ROW["denver"]] > 0
    assert relief[ROW["denver"]] > 0
    assert eligible[ROW["aurora_arapahoe"]] == 0
    assert relief[ROW["aurora_arapahoe"]] == 0
    assert without_flags["co_denver_property_tax_relief"][ROW["denver"]] == 0


def test_philadelphia_wage_tax_uses_the_resident_rate_for_residents(
    with_flags, without_flags
) -> None:
    from policyengine_us import CountryTaxBenefitSystem

    rates = CountryTaxBenefitSystem().parameters(PERIOD).gov.local.pa.philadelphia
    resident_rate = rates.tax.income.resident_rate
    nonresident_rate = rates.tax.income.nonresident_rate
    wages = PHILADELPHIA_WORKER["pa_philadelphia_wage_tax_taxable_wages"]
    tax = with_flags["pa_philadelphia_wage_tax"]

    assert resident_rate != nonresident_rate
    assert tax[ROW["philadelphia"]] == pytest.approx(wages * resident_rate)
    assert tax[ROW["delaware_county_pa"]] == pytest.approx(wages * nonresident_rate)
    assert without_flags["pa_philadelphia_wage_tax"][
        ROW["philadelphia"]
    ] == pytest.approx(wages * nonresident_rate)
