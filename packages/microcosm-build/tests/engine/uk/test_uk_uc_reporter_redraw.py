"""Tests split from packages/microcosm-build/tests/test_uk_uc_reporter_redraw.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_uc_reporter_redraw import *


def test_live_engine_screen_reads_the_recorded_capital_with_property() -> None:
    """The screen's capital is the FRS carrier plus the unit's property share,
    as uc_capital_coherence records it: unit 103 holds no financial capital
    but its household's other property is over the UC limit (microcosm#1095)."""

    person = pd.DataFrame(
        {
            "person_id": [1001, 1002, 1003],
            "person_benunit_id": [101, 102, 103],
            "person_household_id": [1, 2, 3],
            "age": [40, 40, 40],
            "is_benunit_head": [True, True, True],
            "is_uc_claimant": [True, True, True],
            UC_REPORTER_REDRAW_OUTPUT: [0.0, 0.0, 0.0],
            "employment_income": [0.0, 0.0, 0.0],
            "self_employment_income": [0.0, 0.0, 0.0],
            "savings_interest_income": [0.0, 0.0, 0.0],
            "dividend_income": [0.0, 0.0, 0.0],
            "property_income": [0.0, 0.0, 0.0],
            "other_investment_income": [0.0, 0.0, 0.0],
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [101, 102, 103],
            "frs_benunit_capital": [0.0, 16_001.0, 0.0],
            "is_married": [False, False, False],
            support_channel_column("benunit"): ["frs", "frs", "frs"],
            support_clone_index_column("benunit"): [0, 0, 0],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "region": ["LONDON", "LONDON", "LONDON"],
            "council_tax": [0.0, 0.0, 0.0],
            "tenure_type": ["OWNED_OUTRIGHT", "OWNED_OUTRIGHT", "OWNED_OUTRIGHT"],
            "rent": [0.0, 0.0, 0.0],
            "savings": [100_000.0, 100_000.0, 0.0],
            "other_residential_property_value": [0.0, 0.0, 16_001.0],
            "non_residential_property_value": [0.0, 0.0, 0.0],
            "corporate_wealth": [0.0, 0.0, 0.0],
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.ones(3),
        weight_kind=WeightKind.DESIGN,
        time_period="2024",
    )
    materialized = _materialize_screen_inputs(
        frame,
        engine=PolicyEngineUKEngine(),
        person=person,
        benunit=benunit,
        household=household,
    )
    screen = _positive_pre_takeup_award_screen(
        materialized["uc_maximum_amount"],
        materialized["uc_income_reduction"],
        expected=3,
    )

    np.testing.assert_array_equal(screen, [True, False, False])
