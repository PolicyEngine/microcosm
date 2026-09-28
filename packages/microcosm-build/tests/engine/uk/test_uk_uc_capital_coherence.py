"""Tests split from packages/microcosm-build/tests/test_uk_uc_capital_coherence.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_uc_capital_coherence import *


def test_engine_uses_reported_capital_and_sentinel_routes_to_residual_proxy() -> None:
    person = pd.DataFrame(
        {
            "person_id": [1001, 1002, 1003],
            "person_benunit_id": [101, 102, 103],
            "person_household_id": [1, 2, 3],
            "age": [40, 40, 40],
            "is_benunit_head": [True, True, True],
            "universal_credit_reported": [10.0, 10.0, 10.0],
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [101, 102, 103],
            "uc_reported_capital": [0.0, 16_000.0, -1.0],
            "frs_benunit_capital": [0.0, 16_000.0, -1.0],
            "would_claim_uc": [True, True, True],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "region": ["LONDON", "LONDON", "LONDON"],
            "council_tax": [0.0, 0.0, 0.0],
            "tenure_type": ["OWNED_OUTRIGHT", "OWNED_OUTRIGHT", "OWNED_OUTRIGHT"],
            "rent": [0.0, 0.0, 0.0],
            "savings": [100_000.0, 100_000.0, 7_000.0],
            "other_residential_property_value": [0.0, 0.0, 0.0],
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

    result = PolicyEngineUKEngine().materialize(frame, ["uc_assessable_capital"], 2024)[
        "uc_assessable_capital"
    ]

    np.testing.assert_array_equal(result, np.asarray([0.0, 16_000.0, 7_000.0]))
