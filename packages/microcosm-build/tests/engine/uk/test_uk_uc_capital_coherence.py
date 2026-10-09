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


def _capital_frame(uc_capital, pension_credit_capital):
    """Pensioner parents with their working-age child's unit, and a lone
    parent whose 17-year-old dependant is no claimant or partner."""

    person = pd.DataFrame(
        {
            "person_id": [1001, 1002, 1003, 2001, 2002],
            "person_benunit_id": [101, 101, 102, 201, 201],
            "person_household_id": [1, 1, 1, 2, 2],
            "age": [70, 68, 30, 40, 17],
            "is_benunit_head": [True, False, True, True, False],
            "is_uc_claimant": [True, True, True, True, False],
            "universal_credit_reported": 0.0,
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [101, 102, 201],
            "uc_reported_capital": uc_capital,
            "pension_credit_reported_capital": pension_credit_capital,
            "would_claim_uc": True,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["LONDON", "LONDON"],
            "council_tax": [0.0, 0.0],
            "tenure_type": ["OWNED_OUTRIGHT", "OWNED_OUTRIGHT"],
            "rent": [0.0, 0.0],
            "savings": [0.0, 0.0],
            "corporate_wealth": [0.0, 0.0],
            "other_residential_property_value": [90_000.0, 50_000.0],
            "non_residential_property_value": [10_000.0, 0.0],
            "owned_land": [30_000.0, 5_000.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.ones(2),
        weight_kind=WeightKind.DESIGN,
        time_period="2024",
    )


def _assessable_capital(frame):
    values = PolicyEngineUKEngine().materialize(
        frame, ["uc_assessable_capital", "pension_credit_assessable_capital"], 2024
    )
    return (
        np.asarray(values["uc_assessable_capital"], dtype=float),
        np.asarray(values["pension_credit_assessable_capital"], dtype=float),
    )


def test_recorded_property_shares_match_the_engines_own_proxies() -> None:
    """With no financial capital, the engine's proxies assess each unit
    exactly its property share, so recording the share changes nothing,
    whether every unit records capital or only some do (microcosm#1095)."""

    from microcosm.build.uk_runtime.frs_take_up import uk_take_up_population_policy

    unavailable = [-1.0, -1.0, -1.0]
    proxied = _capital_frame(unavailable, unavailable)
    tables = {
        entity: proxied.table(entity) for entity in ("person", "benunit", "household")
    }
    uc_share = uc_property_capital_share(
        tables["person"], tables["benunit"], tables["household"]
    )
    pension_credit_share = pension_credit_property_capital_share(
        tables["person"],
        tables["benunit"],
        tables["household"],
        qualifying_age=uk_take_up_population_policy(2024).state_pension_age,
    )
    # UC shares other property by claimants and partners (the dependant owns
    # none); Pension Credit adds land and shares by members at its age.
    np.testing.assert_allclose(
        uc_share, [200_000.0 / 3, 100_000.0 / 3, 50_000.0], rtol=1e-12
    )
    np.testing.assert_array_equal(pension_credit_share, [130_000.0, 0.0, 0.0])

    uc_proxy, pension_credit_proxy = _assessable_capital(proxied)
    np.testing.assert_allclose(uc_proxy, uc_share, rtol=1e-6)
    np.testing.assert_allclose(pension_credit_proxy, pension_credit_share, rtol=1e-6)

    recorded = _capital_frame(
        recorded_capital_with_property(np.zeros(3), uc_share),
        recorded_capital_with_property(np.zeros(3), pension_credit_share),
    )
    uc_recorded, pension_credit_recorded = _assessable_capital(recorded)
    np.testing.assert_allclose(uc_recorded, uc_proxy, rtol=1e-6)
    np.testing.assert_allclose(pension_credit_recorded, pension_credit_proxy, rtol=1e-6)

    # The parents record and their child's unit stays proxied: the UC proxy
    # shares out only what the recorded unit did not take.
    mixed = _capital_frame(
        [uc_share[0], -1.0, uc_share[2]],
        [pension_credit_share[0], -1.0, pension_credit_share[2]],
    )
    uc_mixed, pension_credit_mixed = _assessable_capital(mixed)
    np.testing.assert_allclose(uc_mixed, uc_proxy, rtol=1e-6)
    np.testing.assert_allclose(pension_credit_mixed, pension_credit_proxy, rtol=1e-6)

    # A unit that reports Universal Credit keeps its receipt: it records the
    # carrier alone, the engine assesses it no property, and the shares the
    # other units record stand.
    reporter_keeps = _capital_frame(
        [uc_share[0], 0.0, uc_share[2]],
        recorded_capital_with_property(np.zeros(3), pension_credit_share),
    )
    uc_reporter_keeps, _ = _assessable_capital(reporter_keeps)
    np.testing.assert_allclose(
        uc_reporter_keeps, [uc_share[0], 0.0, uc_share[2]], rtol=1e-6
    )
