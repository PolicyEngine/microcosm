"""The SNAP caseload composition holdout's measures on the locked policyengine-us.

The release scorer asks the engine for each measure at the release year. These
tests pin what the engine returns for that request, on households whose SNAP
position is known, and what the holdout makes of it (microcosm#647).
"""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.snap_caseload_composition_holdout import (
    SNAP_CASELOAD_COMPOSITION_HOLDOUT_VARIABLES,
    snap_caseload_composition_holdout_payload,
)

PERIOD = 2024
MONTHS = [f"{PERIOD}-{month:02d}" for month in range(1, 13)]
JANUARY, DECEMBER = MONTHS[0], MONTHS[-1]
# 7 CFR 273.10(e)(2)(ii)(C): the minimum monthly allotment for a one- or
# two-person household, $23 in fiscal years 2024 and 2025.
_MINIMUM_MONTHLY_ALLOTMENT = 23
_NAMES = (
    "within",
    "asset_only",
    "income_failing",
    "no_take_up",
    "not_eligible",
    "october",
)


def _household(
    name,
    *,
    state_fips,
    earnings,
    assets=0.0,
    takes_up=True,
    reported=(),
    undocumented_adult=False,
) -> dict:
    """A parent of 35 and a child of 6 in their own household and SPM unit.

    The state is given as ``state_fips`` and reported receipt month by month,
    as a release H5 stores them. ``undocumented_adult`` adds a second adult,
    whom SNAP excludes from the unit.
    """
    adult, child = f"{name}_adult", f"{name}_child"
    people = {
        adult: {
            "age": {PERIOD: 35},
            "employment_income": {PERIOD: earnings},
            "bank_account_assets": {PERIOD: assets},
        },
        child: {"age": {PERIOD: 6}},
    }
    if undocumented_adult:
        people[f"{name}_other_adult"] = {
            "age": {PERIOD: 40},
            "immigration_status": {PERIOD: "UNDOCUMENTED"},
        }
    members = list(people)
    return {
        "people": people,
        "tax_units": {f"{name}_tax_unit": {"members": members}},
        "families": {f"{name}_family": {"members": members}},
        "marital_units": {
            f"{member}_marital_unit": {"members": [member]} for member in members
        },
        "spm_units": {
            f"{name}_spm_unit": {
                "members": members,
                "takes_up_snap_if_eligible": {PERIOD: takes_up},
                "receives_snap": {month: month in reported for month in MONTHS},
            }
        },
        "households": {
            f"{name}_household": {
                "members": members,
                "state_fips": {PERIOD: state_fips},
            }
        },
    }


@pytest.fixture(scope="module")
def simulation():
    """Six households, in ``_NAMES`` order.

    In the locked engine's rules, California (06) and Washington (53) confer
    categorical eligibility with no resource test, and Washington's reaches a
    household over both federal income limits. Texas (48) keeps a resource
    test. Missouri (29) applies the federal limits alone, which rise each
    October. The unit without the take-up flag holds a third member, an
    undocumented adult, whom SNAP excludes.
    """
    from policyengine_us import Simulation

    households = (
        _household("within", state_fips=6, earnings=12_000),
        _household(
            "asset_only",
            state_fips=6,
            earnings=12_000,
            assets=10_000.0,
            reported=MONTHS,
        ),
        _household("income_failing", state_fips=53, earnings=32_000),
        _household(
            "no_take_up",
            state_fips=6,
            earnings=12_000,
            takes_up=False,
            undocumented_adult=True,
        ),
        _household("not_eligible", state_fips=48, earnings=12_000, assets=10_000.0),
        # Gross income sits between the limits of fiscal years 2024 and 2025.
        _household("october", state_fips=29, earnings=26_200, reported=[JANUARY]),
    )
    situation: dict = {}
    for household in households:
        for entity, members in household.items():
            situation.setdefault(entity, {}).update(members)
    return Simulation(situation=situation)


def _units(simulation) -> pd.DataFrame:
    units = pd.DataFrame(
        {
            variable: np.asarray(simulation.calculate(variable, PERIOD))
            for variable in SNAP_CASELOAD_COMPOSITION_HOLDOUT_VARIABLES
        },
        index=list(_NAMES),
    )
    units["weight"] = 1.0
    return units


def test_a_year_request_reads_monthly_tests_in_december(simulation) -> None:
    """Pin the engine rule the holdout's wording rests on (microcosm#647).

    policyengine-core reports a monthly true-or-false variable for a year as
    its December value, and a monthly amount as the year's total. If an engine
    bump changes either, this test fails and the holdout's definitions, which
    say "in December", must change with it.
    """
    units = _units(simulation)
    position = _NAMES.index("october")

    def month(variable: str, period: str):
        return np.asarray(simulation.calculate(variable, period))[position]

    # Eligible from October only, so the year reads eligible.
    assert not month("is_snap_eligible", JANUARY)
    assert month("is_snap_eligible", DECEMBER)
    assert units.loc["october", "is_snap_eligible"]
    assert units.loc["october", "meets_snap_gross_income_test"]
    # Reported receipt in January only, so the year reads no receipt.
    assert month("receives_snap", JANUARY)
    assert not month("receives_snap", DECEMBER)
    assert not units.loc["october", "receives_snap"]
    # The benefit is the three eligible months' total.
    assert month("snap", JANUARY) == 0
    assert month("snap", DECEMBER) > 0
    assert units.loc["october", "snap"] == pytest.approx(3 * month("snap", DECEMBER))
    for variable in (
        "is_snap_eligible",
        "meets_snap_gross_income_test",
        "meets_snap_net_income_test",
        "meets_snap_asset_test",
        "takes_up_snap_if_eligible",
        "receives_snap",
    ):
        assert units[variable].dtype == np.bool_, variable


def test_holdout_groups_follow_the_engine_on_households(simulation) -> None:
    units = _units(simulation)

    # Categorical eligibility carries the unit over the resource limit and the
    # unit over both income limits; the second gets the minimum allotment.
    assert units.loc["asset_only", "is_snap_eligible"]
    assert not units.loc["asset_only", "meets_snap_asset_test"]
    assert units.loc["income_failing", "is_snap_eligible"]
    assert not units.loc["income_failing", "meets_snap_gross_income_test"]
    assert not units.loc["income_failing", "meets_snap_net_income_test"]
    assert units.loc["income_failing", "snap"] == pytest.approx(
        12 * _MINIMUM_MONTHLY_ALLOTMENT
    )
    # Texas tests resources, so the same unit is not eligible there.
    assert not units.loc["not_eligible", "is_snap_eligible"]
    assert units.loc["not_eligible", "snap"] == 0
    # An eligible unit without the take-up flag receives nothing.
    assert units.loc["no_take_up", "is_snap_eligible"]
    assert units.loc["no_take_up", "snap"] == 0
    # Persons are the members SNAP does not exclude: the unit of three with
    # an undocumented adult counts two.
    np.testing.assert_array_equal(
        np.asarray(simulation.calculate("spm_unit_size", PERIOD)), [2, 2, 2, 3, 2, 2]
    )
    np.testing.assert_array_equal(units["snap_unit_size"], [2] * 6)
    assert units["snap_unit_size"].dtype.kind == "i"
    np.testing.assert_array_equal(units["spm_unit_state_fips"], [6, 6, 53, 6, 48, 29])

    payload = snap_caseload_composition_holdout_payload(units, period=PERIOD)
    national = payload["national"]
    assert payload["status"] == "available"
    take_up = national["take_up"]
    assert take_up["eligible_weighted_units"] == 5
    assert take_up["flagged_weighted_units"] == 4
    assert take_up["receiving_weighted_units"] == 4
    assert take_up["eligible_weighted_persons"] == 10
    assert take_up["receiving_share_of_eligible_persons"] == pytest.approx(0.8)
    composition = national["composition"]
    assert composition["income_failing"]["weighted_units"] == 1
    assert composition["income_failing"]["average_monthly_benefit"] == pytest.approx(
        _MINIMUM_MONTHLY_ALLOTMENT
    )
    assert composition["asset_only"]["weighted_units"] == 1
    assert composition["within_federal_limits"]["weighted_units"] == 2
    assert national["caseload_source"]["reported_receipt"]["weighted_units"] == 1
    assert national["caseload_source"]["take_up_fill"]["weighted_units"] == 3
    assert national["benefit_outside_december_eligibility"]["weighted_units"] == 0
    assert [row["state_fips"] for row in payload["states"]] == ["06", "29", "48", "53"]
    assert payload["states_at_or_above_saturation_threshold"] == {
        "flagged_share_of_eligible_units": ["29", "53"],
        "receiving_share_of_eligible_units": ["29", "53"],
        "receiving_share_of_eligible_persons": ["29", "53"],
    }
