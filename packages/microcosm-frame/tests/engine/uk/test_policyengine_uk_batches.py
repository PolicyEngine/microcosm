"""policyengine-uk calls over more than one clone, clone by clone (microcosm#931).

A frame whose households carry the clone index with more than one clone runs
one simulation per clone through the adapter: each clone's simulation reads
the pool's share denominator (``c_b = S_b / S`` at the frame's own weights),
the UK declaration's guard judges it under the spine's pass, and the values
come back in the frame's order. Held here against one simulation over the
same frame without the column (the adapter's single-simulation path), at the
dataset's year and projected a year on: the share formulas within the valid
float32 bound, household taxes within its per-row form, household-local
values bitwise, people in the frame's order and groups by id. What only the
pool defines is refused before any simulation is built, and a frame with one
clone takes the single simulation it always has. Every household here is made
up.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.adapters import policyengine_uk as adapter_module
from microcosm.frame.adapters.engine_population import (
    SPINE_BATCHED_PASS,
    PopulationGuardError,
    frame_key_sums,
    household_batches,
    pool_share_atol,
    pool_share_rtol,
)
from microcosm.frame.adapters.policyengine_uk import UK_SCHEMA, PolicyEngineUKEngine
from microcosm.frame.adapters.policyengine_uk_population import (
    UK_ENGINE_POPULATION,
    UK_SPINE_ENGINE_BATCH_COLUMN,
)

D, P = 2024, 2025
CLONES, HOUSEHOLDS = 3, 12
COICOP = UK_ENGINE_POPULATION.key_adds["consumption"]
REGIONS = ("LONDON", "NORTH_WEST", "SCOTLAND", "WALES", "SOUTH_EAST")
MONTHS = "months_since_last_birthday"
ATTENDANCE = "attends_private_school"

#: Each share result and the allocation key it divides by: business rates
#: and corporate SDLT are shareholding times a receipt.
SHARES = {
    "shareholding": "corporate_sector_wealth",
    "corporate_land_value": "corporate_sector_wealth",
    "consumption_shareholding": "consumption",
    "business_rates": "corporate_sector_wealth",
    "corporate_sdlt": "corporate_sector_wealth",
}
SHARE_DEPENDENT_TAXES = ("business_rates", "corporate_sdlt")
#: What a household's own records decide, at each entity, ids included.
LOCAL = (
    "person_id",
    "income_tax",
    "national_insurance",
    "employment_income",
    "age_under_18",
    "benunit_id",
    "benunit_tax",
    "num_adults",
    "universal_credit",
    "LHA_category",
    "household_id",
    "household_market_income",
    "household_benefits",
    "household_weight",
)
REQUESTED = (*SHARES, "household_tax", *LOCAL)


def _tables(store):
    """Three clones of the same households, each drawing its own keys, pay
    and weights. Household ids interleave the clones, benefit unit ids sort
    the other way, and the people are listed out of order."""

    persons, households = [], []
    for clone in range(CLONES):
        for h in range(HOUSEHOLDS):
            household_id = 10 * h + clone + 1
            benunit_id = 5_000 - 2 * household_id
            draw = np.random.default_rng(1009 * clone + 17 * h)
            shared = np.random.default_rng(5_000 + h)
            row = {
                "household_id": household_id,
                UK_SPINE_ENGINE_BATCH_COLUMN: clone,
                "weight": 40_000.0 * (1 + h % 5) / CLONES * draw.uniform(0.7, 1.3),
                "region": REGIONS[h % len(REGIONS)],
                "council_tax": 1_400.0 + 90 * (h % 5),
                "rent": 9_000.0 if h % 2 else 0.0,
                "tenure_type": "RENT_PRIVATELY" if h % 2 else "OWNED_OUTRIGHT",
                "corporate_wealth": float(draw.lognormal(10, 1.6)) if h % 2 else 0.0,
                "private_pension_wealth": (
                    float(draw.lognormal(11, 1.1)) if h % 3 else 0.0
                ),
            }
            for column in COICOP:
                row[column] = float(draw.uniform(300, 6_000))
            households.append(row)
            for p in range(1 + h % 4):
                age = (41, 38, 12, 9)[p]
                pay = np.random.default_rng(31 * h + p + 997 * clone)
                person = {
                    "person_id": 10 * household_id + p,
                    "person_benunit_id": benunit_id,
                    "person_household_id": household_id,
                    "age": age,
                    "gender": "MALE" if (h + p) % 2 else "FEMALE",
                    "employment_income": (
                        float(pay.uniform(0, 60_000)) if age >= 18 and h % 3 else 0.0
                    ),
                    "attends_private_school_random_draw": float(shared.uniform()),
                }
                if MONTHS in store:
                    person[MONTHS] = float(shared.integers(0, 12))
                if ATTENDANCE in store:
                    person[ATTENDANCE] = bool(age < 18 and shared.uniform() < 0.4)
                persons.append(person)
    person = pd.DataFrame(persons).sample(frac=1.0, random_state=7)
    household = pd.DataFrame(households).sort_values("household_id")
    benunit = pd.DataFrame(
        {"benunit_id": np.sort(person["person_benunit_id"].unique())}
    )
    return (
        person.reset_index(drop=True),
        benunit,
        household.reset_index(drop=True),
    )


def _frame(*, clones=True, single=False, store=(MONTHS,)):
    person, benunit, household = _tables(store)
    weights = household.pop("weight").to_numpy()
    if single:
        household[UK_SPINE_ENGINE_BATCH_COLUMN] = 0
    if not clones:
        household = household.drop(columns=[UK_SPINE_ENGINE_BATCH_COLUMN])
    return Frame(
        tables={"person": person, "benunit": benunit, "household": household},
        schema=UK_SCHEMA,
        weights={"household": Weights(values=weights, kind=WeightKind.DESIGN)},
    )


def _quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **kwargs)


def _household_tax_terms(frame, period, dataset_period):
    """The pool's household tax terms, mapped to households, for the per-row
    bound on household_tax."""

    from policyengine_uk import Microsimulation
    from policyengine_uk.variables.gov.hmrc.household_tax import (
        HOUSEHOLD_TAX_VARIABLES,
    )

    simulation = _quiet(
        Microsimulation,
        dataset=PolicyEngineUKEngine()._build_dataset(frame, dataset_period),
    )
    return {
        name: np.asarray(
            _quiet(simulation.calculate, name, period, map_to="household"),
            dtype=np.float64,
        )
        for name in HOUSEHOLD_TAX_VARIABLES
    }


@pytest.fixture(scope="module")
def system():
    from policyengine_uk import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


@pytest.fixture(
    scope="module", params=[(D, None), (P, D)], ids=["dataset_year", "projected"]
)
def runs(request):
    period, dataset_period = request.param
    frame, pool = _frame(), _frame(clones=False)
    engine = PolicyEngineUKEngine()
    batched = _quiet(
        engine.materialize, frame, REQUESTED, period, dataset_period=dataset_period
    )
    single = _quiet(
        PolicyEngineUKEngine().materialize,
        pool,
        REQUESTED,
        period,
        dataset_period=dataset_period,
    )
    dataset_year = period if dataset_period is None else dataset_period
    return SimpleNamespace(
        frame=frame,
        engine=engine,
        batched=batched,
        single=single,
        period=period,
        years=period - dataset_year,
        terms=_household_tax_terms(pool, period, dataset_year),
    )


def _share_rtol(name, years):
    return pool_share_rtol(
        HOUSEHOLDS,
        CLONES * HOUSEHOLDS,
        key_inputs=len(UK_ENGINE_POPULATION.key_adds[SHARES[name]]),
        projection_years=years,
    )


def test_the_clones_hold_different_shares_of_the_keys():
    # Uncorrected, each clone's simulation would divide by its own totals:
    # the shares the data gives each clone are well off a third.
    frame = _frame()
    batches = household_batches(frame, UK_SPINE_ENGINE_BATCH_COLUMN)
    sums = frame_key_sums(frame, UK_ENGINE_POPULATION, batches)
    for formula in UK_ENGINE_POPULATION.corrected:
        pool = sum(sums[batch.key][formula] for batch in batches)
        shares = [sums[batch.key][formula] / pool for batch in batches]
        assert max(abs(share - 1 / CLONES) for share in shares) > 0.01, formula


def test_share_results_are_the_pools_to_float32_rounding(runs):
    for name in SHARES:
        batched, single = runs.batched[name], runs.single[name]
        assert batched.dtype == single.dtype, name
        assert np.any(single != 0), name
        np.testing.assert_allclose(
            batched,
            single,
            rtol=_share_rtol(name, runs.years),
            atol=0,
            err_msg=f"{name} at {runs.period}",
        )


def test_household_taxes_are_the_pools_within_the_per_row_bound(runs):
    terms = runs.terms
    atol = pool_share_atol(
        sum(np.abs(terms[name]) for name in SHARE_DEPENDENT_TAXES),
        sum(np.abs(values) for values in terms.values()),
        share_rtol=max(_share_rtol(name, runs.years) for name in SHARE_DEPENDENT_TAXES),
        additions=len(terms),
    )
    gap = np.abs(
        runs.batched["household_tax"].astype(np.float64)
        - runs.single["household_tax"].astype(np.float64)
    )
    assert np.all(gap <= atol), f"household_tax at {runs.period}"
    assert np.any(terms["business_rates"] != 0)


def test_household_local_values_are_the_pools_bitwise(runs):
    for name in LOCAL:
        batched, single = runs.batched[name], runs.single[name]
        assert type(batched) is type(single), name
        assert batched.dtype == single.dtype, name
        np.testing.assert_array_equal(batched, single, err_msg=name)
    assert np.any(runs.single["income_tax"] != 0)


def test_people_come_back_in_the_frames_order_and_groups_by_id(runs):
    frame = runs.frame
    for entity, name in (
        ("person", "person_id"),
        ("benunit", "benunit_id"),
        ("household", "household_id"),
    ):
        np.testing.assert_array_equal(
            runs.batched[name], frame.table(entity)[name].to_numpy(), err_msg=entity
        )
    # The frame lists people out of id order and benefit units against the
    # households' order, so neither is the batches' concatenation.
    people = frame.table("person")["person_id"].to_numpy()
    assert not np.array_equal(people, np.sort(people))


def test_the_call_ran_in_three_judged_batches(runs):
    assert runs.engine.simulation_batches(runs.frame) == CLONES
    assert runs.engine.batch_record() == {
        "by": UK_SPINE_ENGINE_BATCH_COLUMN,
        "batched_calls": 1,
        "batches": [CLONES],
        "pass": SPINE_BATCHED_PASS.name,
        "declaration_sha256": UK_ENGINE_POPULATION.sha256,
    }


def test_a_frame_with_one_clone_takes_one_simulation_bitwise(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a frame with one clone never runs in batches")

    monkeypatch.setattr(adapter_module, "materialize_in_batches", refuse)
    engine = PolicyEngineUKEngine()
    one = _quiet(
        engine.materialize, _frame(single=True), REQUESTED, P, dataset_period=D
    )
    plain = _quiet(
        PolicyEngineUKEngine().materialize,
        _frame(clones=False),
        REQUESTED,
        P,
        dataset_period=D,
    )
    for name in REQUESTED:
        assert type(one[name]) is type(plain[name]), name
        assert one[name].dtype == plain[name].dtype, name
        np.testing.assert_array_equal(one[name], plain[name], err_msg=name)
    assert engine.simulation_batches(_frame(single=True)) == 1
    assert engine.batch_record() is None


def test_stored_attendance_may_be_asked_for_in_batches():
    store = (MONTHS, ATTENDANCE)
    frame = _frame(store=store)
    batched = _quiet(
        PolicyEngineUKEngine().materialize,
        frame,
        [ATTENDANCE, "household_tax"],
        P,
        dataset_period=D,
    )
    stored = frame.table("person")[ATTENDANCE].to_numpy()
    assert stored.any()
    np.testing.assert_array_equal(batched[ATTENDANCE], stored)


def _refusing_engine(monkeypatch, system, **kwargs):
    """An adapter that refuses to build any simulation."""

    def refuse(**_kwargs):
        raise AssertionError("no simulation is built")

    engine = PolicyEngineUKEngine(**kwargs)
    engine._system = system
    monkeypatch.setattr(
        engine,
        "_import_policyengine_uk",
        lambda: SimpleNamespace(Microsimulation=refuse),
    )
    return engine


@pytest.mark.parametrize(
    ("names", "period", "store", "expected"),
    [
        (
            ["income_tax", ATTENDANCE],
            P,
            (MONTHS,),
            {("inert_request", ATTENDANCE, str(P))},
        ),
        (
            ["income_tax", "household_income_decile"],
            P,
            (MONTHS,),
            {("refused_request", "household_income_decile", str(P))},
        ),
        (
            ["income_tax"],
            2026,
            (MONTHS,),
            {
                ("window_period", None, "2026"),
                ("window_request", "income_tax", "2026"),
            },
        ),
        (
            ["income_tax"],
            P,
            (),
            {
                ("stored_not_input", MONTHS, str(D)),
                ("stored_not_input", MONTHS, str(P)),
            },
        ),
    ],
    ids=["attendance_unstored", "decile", "rebalancing_year", "months_unstored"],
)
def test_what_only_the_pool_defines_is_refused_before_any_simulation(
    monkeypatch, system, names, period, store, expected
):
    engine = _refusing_engine(monkeypatch, system)
    with pytest.raises(PopulationGuardError) as raised:
        engine.materialize(_frame(store=store), names, period, dataset_period=D)
    found = {(v.kind, v.name, v.period) for v in raised.value.verdict.violations}
    assert found == expected
    assert engine.batch_record() is None


def test_a_frame_with_fewer_clones_than_the_spine_is_refused(monkeypatch, system):
    engine = _refusing_engine(monkeypatch, system, clones=CLONES + 1)
    with pytest.raises(ValueError, match="no engine call runs per clone"):
        engine.materialize(_frame(), ["income_tax"], P, dataset_period=D)
    with pytest.raises(ValueError, match="no engine call runs per clone"):
        engine.simulation_batches(_frame(single=True))


def test_the_adapters_parameters_are_the_engines(system):
    engine = PolicyEngineUKEngine()
    node = engine.parameters(P).gov.simulation.private_school_vat
    expected = system.parameters(P).gov.simulation.private_school_vat
    assert node.private_school_factor == expected.private_school_factor
