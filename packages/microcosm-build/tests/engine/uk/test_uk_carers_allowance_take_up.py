"""Engine lockstep for the Carer's Allowance take-up stage (microcosm#1319).

The stage opens the claim flag to carers whose care hours meet the engine's
condition, who care for a person the engine holds severely disabled
(microcosm#1418), who report no allowance, and whom the engine's earnings test
holds gainfully employed at the survey year; in Scotland the flag is Carer
Support Payment's. These tests fail if policyengine-uk moves the hours
condition or the earnings limit, adds a condition the stage does not read, or
stops applying Carer Support Payment's own earnings limit, on which drawing
Scotland rests.
"""

from __future__ import annotations

from importlib import metadata

import numpy as np
import pandas as pd

from microcosm.build.stochastic_assignment import weight_balanced_uniforms
from microcosm.build.uk_runtime import carers_allowance_take_up as ca
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

_TEST = ca.CARERS_ALLOWANCE_EARNINGS_TEST


def _simulation(people: dict, period: int, region: str = "WALES"):
    """One-person benefit units and households, inputs at ``period``."""

    import policyengine_uk

    situation = {
        "people": {
            name: {variable: {period: value} for variable, value in inputs.items()}
            for name, inputs in people.items()
        },
        "benunits": {f"b_{name}": {"members": [name]} for name in people},
        "households": {
            f"h_{name}": {"members": [name], "region": {period: region}}
            for name in people
        },
    }
    return policyengine_uk.Simulation(situation=situation)


def _carers(weekly_pay, *, hours: float = 35.0) -> dict:
    return {
        f"carer_{index}": {
            "age": 40,
            "care_hours": hours,
            "employment_income": pay * 52,
            "would_claim_carers_allowance": True,
        }
        for index, pay in enumerate(weekly_pay)
    }


def test_the_policy_is_the_engines_hours_condition_and_limit() -> None:
    assert ca.uk_carers_allowance_policy(2024) == ca.UKCarersAllowancePolicy(
        min_hours=35.0,
        earnings_limit=151.0,
        weekly_rate=81.90,
        instant="2024-04-30",
        source=(
            f"policyengine-uk {metadata.version('policyengine-uk')} "
            "gov.dwp.carers_allowance at 2024-04-30"
        ),
    )
    # The April 2025 rise the drawn carers reach in the baseline.
    assert ca.uk_carers_allowance_policy(2025).earnings_limit == 196.0


def test_the_hours_condition_is_the_engines() -> None:
    policy = ca.uk_carers_allowance_policy(2024)
    # The FRS care-hours band edges, and one hour under the line.
    hours = [0.0, 5.0, 10.0, 20.0, 34.0, 35.0, 50.0, 100.0]
    simulation = _simulation(
        {
            f"carer_{index}": {
                "age": 40,
                "care_hours": value,
                "would_claim_carers_allowance": True,
            }
            for index, value in enumerate(hours)
        },
        2024,
    )
    assert not np.asarray(simulation.calculate(_TEST, 2024)).any()
    entitled = np.asarray(simulation.calculate("carers_allowance_pre_overlap", 2024))
    np.testing.assert_array_equal(entitled > 0, np.asarray(hours) >= policy.min_hours)


def test_the_earnings_limit_is_the_only_bar_left_for_a_non_reporting_carer() -> None:
    policy = ca.uk_carers_allowance_policy(2024)
    weekly_pay = np.asarray([100.0, 151.0, 152.0, 180.0, 196.0, 197.0, 400.0])
    survey = _simulation(_carers(weekly_pay), 2024)
    gainful = np.asarray(survey.calculate(_TEST, 2024))
    # Below the personal allowance pay is its own net earnings; above it the
    # engine's PAYE and Class 1 still leave 400 a week over the limit.
    np.testing.assert_array_equal(gainful, weekly_pay > policy.earnings_limit)
    entitled = np.asarray(survey.calculate("carers_allowance_pre_overlap", 2024))
    # For a 35-hour carer outside Scotland who would claim, the earnings test
    # is the whole difference between entitled and not: the stage's eligible
    # set is exactly the carers the limit alone keeps out.
    np.testing.assert_array_equal(entitled > 0, ~gainful)

    # The April 2025 rise: carers between the two limits, eligible for the
    # draw at the survey year, are entitled at 2025.
    later = _simulation(_carers(weekly_pay), 2025)
    entitled_2025 = np.asarray(later.calculate("carers_allowance_pre_overlap", 2025))
    rise = (weekly_pay > 151.0) & (weekly_pay <= 196.0)
    assert rise.sum() == 3
    assert (entitled_2025[rise] > 0).all()
    assert (entitled_2025[weekly_pay > 196.0] == 0).all()


def test_a_drawn_scottish_carer_is_paid_only_under_the_support_payments_limit() -> None:
    simulation = _simulation(_carers([100.0, 400.0]), 2025, region="SCOTLAND")
    # The stage reads Scotland off the household region; the engine's country
    # follows it. A top-level calculate returns the enum decoded.
    country = simulation.calculate("country", 2025, decode_enums=True)
    assert [str(value) for value in country] == ["SCOTLAND", "SCOTLAND"]
    # Scotland is drawn (María's ruling of 2026-10-23): with the flag true,
    # Carer Support Payment pays a carer under its own earnings limit and not
    # one over it (policyengine-uk 2.146.1, PolicyEngine/policyengine-uk#2473),
    # and the allowance pays neither.
    support = np.asarray(
        simulation.calculate("carer_support_payment_pre_overlap", 2025)
    )
    assert support[0] > 0
    assert support[1] == 0
    assert np.asarray(simulation.calculate(_TEST, 2025))[1]
    assert (
        np.asarray(simulation.calculate("carers_allowance_pre_overlap", 2025))[1] == 0
    )
    assert "In Scotland the flag is Carer Support Payment's" in (
        ca.CARERS_ALLOWANCE_ELIGIBILITY
    )


class _CertainContract:
    """Every eligible carer claims, so the flags are the eligible set."""

    def rate(self, key: str, build_year: int | None = None) -> float:
        assert key == ca.CARERS_ALLOWANCE_NEWLY_ENTITLED_RATE_KEY
        return 1.0


def _frame():
    """Five 35-hour carers, each living with an 85-year-old in a benefit unit
    of their own: over the limit in Wales and in Scotland, under it in Wales,
    a reporter, and one over the limit in Wales whose household member draws
    no qualifying benefit. The other four members draw the higher rate of
    Attendance Allowance."""

    weekly_pay = np.asarray([165.0, 165.0, 120.0, 100.0, 165.0])
    reported = np.asarray([0.0, 0.0, 0.0, 81.9 * 52, 0.0])
    carers = pd.DataFrame(
        {
            "person_id": [11, 21, 31, 41, 51],
            "person_benunit_id": [1, 2, 3, 4, 5],
            "person_household_id": [1, 2, 3, 4, 5],
            "age": [40.0] * 5,
            "employment_income": weekly_pay * 52,
            "care_hours": [35.0] * 5,
            ca.CARERS_ALLOWANCE_IN_HOUSEHOLD_HOURS: [35.0] * 5,
            "carers_allowance_reported": reported,
            "would_claim_carers_allowance": reported > 0,
            "aa_category": ["NONE"] * 5,
            **{name: [0.0] * 5 for name in ca.CARERS_ALLOWANCE_MEANS_TESTED_REPORTED},
        }
    )
    members = pd.DataFrame(
        {
            "person_id": [12, 22, 32, 42, 52],
            "person_benunit_id": [6, 7, 8, 9, 10],
            "person_household_id": [1, 2, 3, 4, 5],
            "age": [85.0] * 5,
            "employment_income": [0.0] * 5,
            "care_hours": [0.0] * 5,
            ca.CARERS_ALLOWANCE_IN_HOUSEHOLD_HOURS: [0.0] * 5,
            "carers_allowance_reported": [0.0] * 5,
            "would_claim_carers_allowance": [False] * 5,
            "aa_category": ["HIGHER"] * 4 + ["NONE"],
            **{name: [0.0] * 5 for name in ca.CARERS_ALLOWANCE_MEANS_TESTED_REPORTED},
        }
    )
    benunit = pd.DataFrame({"benunit_id": list(range(1, 11))})
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5],
            "region": ["WALES", "SCOTLAND", "WALES", "WALES", "WALES"],
            "council_tax": [0.0] * 5,
            "tenure_type": ["OWNED_OUTRIGHT"] * 5,
            "rent": [0.0] * 5,
            "household_source_id": [1, 2, 3, 4, 5],
        }
    )
    return uk_national_frame(
        person=pd.concat([carers, members], ignore_index=True),
        benunit=benunit,
        household=household,
        household_weights=np.ones(5),
        weight_kind=WeightKind.DESIGN,
        time_period="2024",
    )


def test_the_stage_reaches_the_carers_the_2025_rise_entitles() -> None:
    engine = PolicyEngineUKEngine()
    result = ca.redraw_carers_allowance_take_up(
        _frame(), engine=engine, contract=_CertainContract()
    )
    flag = result.frame.table("person")["would_claim_carers_allowance"].tolist()
    # Scotland is drawn as Wales is (María's ruling of 2026-10-23); the carer
    # whose household member draws no qualifying benefit is not, and nor is
    # any member (microcosm#1418).
    assert flag == [True, True, False, True, False] + [False] * 5
    assert result.evidence()["weekly_earnings_limit"] == 151.0
    # Projected from the survey year as the released file is, the Welsh carer
    # over the 2024-25 limit and under 2025-26's is entitled and would claim;
    # before the stage the flag kept the allowance from them.
    before = engine.materialize(
        _frame(), ["carers_allowance_pre_overlap"], 2025, dataset_period=2024
    )["carers_allowance_pre_overlap"]
    after = engine.materialize(
        result.frame, ["carers_allowance_pre_overlap"], 2025, dataset_period=2024
    )["carers_allowance_pre_overlap"]
    assert before[0] == 0
    assert after[0] > 0
    # The Scottish carer is paid Carer Support Payment under its own 2025-26
    # limit, not the allowance; the carer under the limit who reports nothing
    # stays unpaid.
    support = engine.materialize(
        result.frame, ["carer_support_payment_pre_overlap"], 2025, dataset_period=2024
    )["carer_support_payment_pre_overlap"]
    assert support[1] > 0
    assert after[1] == 0
    assert after[2] == 0


def test_the_balanced_draw_is_policyengine_uks_stratified_uniform() -> None:
    """Without a start, the helper places records as policyengine-uk's
    stratified_uniform does (which the engine uses to place birthdays); the
    stage adds a start per stratum (the independent review of microcosm#1420,
    finding 4)."""

    from policyengine_uk.utils.stochastic import stratified_uniform

    rng = np.random.default_rng(7)
    n = 2_000
    strata = rng.integers(0, 16, n)
    draws = rng.random(n)
    weights = rng.uniform(0.0, 8.0, n)
    weights[strata == 3] = 0.0
    np.testing.assert_array_equal(
        weight_balanced_uniforms(strata, draws, weights),
        stratified_uniform(strata, draws, weights),
    )
