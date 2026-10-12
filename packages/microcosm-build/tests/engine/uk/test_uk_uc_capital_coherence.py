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
    """Pensioner parents, one of them the household reference person, with
    their working-age child's unit, and a lone parent with a 17-year-old
    dependant."""

    person = pd.DataFrame(
        {
            "person_id": [1001, 1002, 1003, 2001, 2002],
            "person_benunit_id": [101, 101, 102, 201, 201],
            "person_household_id": [1, 1, 1, 2, 2],
            "age": [70, 68, 30, 40, 17],
            "is_benunit_head": [True, False, True, True, False],
            "is_household_head": [True, False, False, True, False],
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


def test_the_engine_assesses_the_recorded_property_and_no_other_unit_takes_it() -> None:
    """The household reference person's unit records the household's property
    at 90% of its value, and policyengine-uk takes each recorded figure as the
    unit's assessable capital. Its own proxies, which apply only to a unit on
    the sentinel, would put a third of the parents' property on their child's
    unit (microcosm#1095)."""

    unavailable = [-1.0, -1.0, -1.0]
    proxied = _capital_frame(unavailable, unavailable)
    tables = {
        entity: proxied.table(entity) for entity in ("person", "benunit", "household")
    }
    uc_share = uc_property_capital_share(
        tables["person"], tables["benunit"], tables["household"]
    )
    pension_credit_share = pension_credit_property_capital_share(
        tables["person"], tables["benunit"], tables["household"]
    )
    # UC counts other property; Pension Credit adds land. The parents' unit
    # and the lone parent's hold their households' property; the child's unit
    # holds none.
    np.testing.assert_allclose(uc_share, [90_000.0, 0.0, 45_000.0], rtol=1e-12)
    np.testing.assert_allclose(
        pension_credit_share, [117_000.0, 0.0, 49_500.0], rtol=1e-12
    )

    recorded = _capital_frame(
        recorded_capital_with_property(np.zeros(3), uc_share),
        recorded_capital_with_property(np.zeros(3), pension_credit_share),
    )
    uc_recorded, pension_credit_recorded = _assessable_capital(recorded)
    np.testing.assert_allclose(uc_recorded, uc_share, rtol=1e-6)
    # The engine assesses Pension Credit capital only for a unit with a member
    # at its qualifying age, so the lone parent's recorded figure is not read.
    np.testing.assert_allclose(
        pension_credit_recorded, [pension_credit_share[0], 0.0, 0.0], rtol=1e-6
    )

    # The engine's UC proxy shares household 1's gross property over its three
    # claimants and partners, so the child's unit would be assessed a third.
    uc_proxy, _ = _assessable_capital(proxied)
    np.testing.assert_allclose(
        uc_proxy, [200_000.0 / 3, 100_000.0 / 3, 50_000.0], rtol=1e-6
    )

    # A unit that reports Universal Credit keeps its receipt: it records the
    # carrier alone, the engine assesses it no property, and its child's unit,
    # which records its own carrier, takes none of it.
    reporter_keeps = _capital_frame(
        [0.0, 0.0, uc_share[2]],
        recorded_capital_with_property(np.zeros(3), pension_credit_share),
    )
    uc_reporter_keeps, _ = _assessable_capital(reporter_keeps)
    np.testing.assert_allclose(uc_reporter_keeps, [0.0, 0.0, uc_share[2]], rtol=1e-6)
