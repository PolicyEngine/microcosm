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
            "other_residential_property_secured_debt": [0.0, 0.0, 0.0],
            "non_residential_property_secured_debt": [0.0, 0.0, 0.0],
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


def _capital_frame(uc_capital, pension_credit_capital, *, child_heads=False):
    """Pensioner parents with their working-age child's unit, and a lone
    parent with a 17-year-old dependant. One of the parents is the household
    reference person, or the child where ``child_heads``."""

    person = pd.DataFrame(
        {
            "person_id": [1001, 1002, 1003, 2001, 2002],
            "person_benunit_id": [101, 101, 102, 201, 201],
            "person_household_id": [1, 1, 1, 2, 2],
            "age": [70, 68, 30, 40, 17],
            "is_benunit_head": [True, False, True, True, False],
            "is_household_head": [not child_heads, False, child_heads, True, False],
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
            "other_residential_property_secured_debt": [0.0, 0.0],
            "non_residential_property_secured_debt": [0.0, 0.0],
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
    unit's complete assessable capital. Its own property share, which reaches
    only a unit on the sentinel, would put a third of the parents' property on
    their child's unit (microcosm#1095)."""

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

    # Since policyengine-uk 2.136.1 the engine values property apart, at 90%
    # of its value less secured debt, and shares it over the household's
    # claimants and partners, so the child's unit would be assessed a third.
    uc_proxy, _ = _assessable_capital(proxied)
    np.testing.assert_allclose(uc_proxy, [60_000.0, 30_000.0, 45_000.0], rtol=1e-6)

    # A unit that reports Universal Credit keeps its receipt: it records the
    # carrier alone, the engine assesses it no property, and its child's unit,
    # which records its own carrier, takes none of it.
    reporter_keeps = _capital_frame(
        [0.0, 0.0, uc_share[2]],
        recorded_capital_with_property(np.zeros(3), pension_credit_share),
    )
    uc_reporter_keeps, _ = _assessable_capital(reporter_keeps)
    np.testing.assert_allclose(uc_reporter_keeps, [0.0, 0.0, uc_share[2]], rtol=1e-6)


def test_a_unit_on_the_sentinel_beside_a_recording_unit_takes_the_engines_proxy() -> (
    None
):
    """A unit on the unavailable sentinel stays on policyengine-uk's own rules.
    Since 2.136.1 Universal Credit gives it its claimant and partner share of
    the household's property at 90%, whatever the other units record, so
    beside a recording reference person's unit it counts that property a
    second time; the Pension Credit proxy, which subtracts nothing, does the
    same. The build records every unit, so no unit is on the sentinel
    (microcosm#1095)."""

    unavailable = [-1.0, -1.0, -1.0]
    tables = {
        entity: _capital_frame(unavailable, unavailable).table(entity)
        for entity in ("person", "benunit", "household")
    }
    uc_share = uc_property_capital_share(
        tables["person"], tables["benunit"], tables["household"]
    )
    pension_credit_share = pension_credit_property_capital_share(
        tables["person"], tables["benunit"], tables["household"]
    )

    # The parents record and their child's unit is on the sentinel: the child
    # takes a third of household 1's property at 90%, which the parents'
    # recorded 90,000 already holds.
    uc_mixed, pension_credit_mixed = _assessable_capital(
        _capital_frame(
            [uc_share[0], -1.0, uc_share[2]],
            [pension_credit_share[0], -1.0, pension_credit_share[2]],
        )
    )
    np.testing.assert_allclose(uc_mixed, [90_000.0, 30_000.0, 45_000.0], rtol=1e-6)
    np.testing.assert_allclose(pension_credit_mixed, [117_000.0, 0.0, 0.0], rtol=1e-6)


def test_property_held_by_a_working_age_reference_person_leaves_pension_credit() -> (
    None
):
    """Where the working-age child is the household reference person, the
    child's unit records the household's property and the parents record none.
    policyengine-uk reads Pension Credit capital only for a unit with a member
    at the qualifying age, so no Pension Credit assessment counts it. Parents
    on the sentinel take the engine's own property share instead, beside the
    child's recorded figure (microcosm#1095)."""

    unavailable = [-1.0, -1.0, -1.0]
    tables = {
        entity: _capital_frame(unavailable, unavailable, child_heads=True).table(entity)
        for entity in ("person", "benunit", "household")
    }
    uc_share = uc_property_capital_share(
        tables["person"], tables["benunit"], tables["household"]
    )
    pension_credit_share = pension_credit_property_capital_share(
        tables["person"], tables["benunit"], tables["household"]
    )
    np.testing.assert_allclose(uc_share, [0.0, 90_000.0, 45_000.0], rtol=1e-12)
    np.testing.assert_allclose(
        pension_credit_share, [0.0, 117_000.0, 49_500.0], rtol=1e-12
    )

    uc_recorded, pension_credit_recorded = _assessable_capital(
        _capital_frame(
            recorded_capital_with_property(np.zeros(3), uc_share),
            recorded_capital_with_property(np.zeros(3), pension_credit_share),
            child_heads=True,
        )
    )
    np.testing.assert_allclose(uc_recorded, uc_share, rtol=1e-6)
    np.testing.assert_array_equal(pension_credit_recorded, [0.0, 0.0, 0.0])

    # The parents on the sentinel: Pension Credit assesses them the gross
    # 130,000 (property and land), which the child's unit also records at 90%.
    uc_mixed, pension_credit_mixed = _assessable_capital(
        _capital_frame(
            [-1.0, uc_share[1], uc_share[2]],
            [-1.0, pension_credit_share[1], pension_credit_share[2]],
            child_heads=True,
        )
    )
    # Universal Credit gives the parents two thirds of the property at 90%.
    np.testing.assert_allclose(uc_mixed, [60_000.0, 90_000.0, 45_000.0], rtol=1e-6)
    np.testing.assert_allclose(pension_credit_mixed, [130_000.0, 0.0, 0.0], rtol=1e-6)
