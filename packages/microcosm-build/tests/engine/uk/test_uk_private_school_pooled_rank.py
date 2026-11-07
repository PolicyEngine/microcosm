"""Private-school attendance ranked over the pool (microcosm#931).

policyengine-uk's ``attends_private_school`` ranks household net income over
the people a simulation holds. Once a dense spine's engine calls run clone by
clone, one clone's simulation would rank that clone alone, so stage 52
restates the formula's last step over the whole frame from engine values each
household's own records decide, the engine's own parameters and microdf's
weighted percentile rank.

The pool here is three clones of twelve made-up households: five alike in
every clone, whose incomes tie across clones, and the rest drawing their own
pay and weights, which spread about one to ten; clone 1 also holds a household
no other clone has. Most households have several children, members tie on
their household's income, and one household in each clone has no weight.
Household ids interleave the clones, benefit unit ids sort the other way and
the people are listed out of order. Each clone's people weigh less than the
formula's one million, so one clone's simulation imputes nobody, while the
pool's people weigh more.

Each child's draw is set 3% under or 3% over the rate the formula gives that
child over the pool, read from the formula's own percentiles and the engine's
parameters, so a rank, a rate, its interpolation or the factor a few percent
off the formula's would store other flags. A spy on microdf's
``percentile_rank``, patched on the class, sees both the formula's call and
the restatement's.

On it: the restatement hands microdf's rank the formula's net incomes and
weights and gets its percentiles back, bit for bit, in one simulation and
clone by clone, and its flags are the formula's; the stage run clone by clone
stores what one simulation over the pool stores; a population factor of zero,
or 6% lower, breaks the parity check; below the floor nobody attends on either
route, while design weights just under it impute on both, since the formula
reads the engine's weights at the calibration year; and a frame of one clone
stores what the stage stored before, the engine's attendance alone.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.private_school_attendance import (
    PRIVATE_SCHOOL_ATTENDANCE_DRAW,
    PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE,
    PRIVATE_SCHOOL_ATTENDANCE_HOUSEHOLD_INPUTS,
    PRIVATE_SCHOOL_ATTENDANCE_OUTPUT,
    PRIVATE_SCHOOL_ATTENDANCE_PERSON_INPUTS,
    PRIVATE_SCHOOL_ATTENDANCE_POOLED_ROUTE,
    restate_private_school_attendance,
    store_private_school_attendance,
)
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
from microcosm.frame.adapters.policyengine_uk_population import (
    UK_SPINE_ENGINE_BATCH_COLUMN,
)

K = 3
SOURCES = 12
#: Sources whose records are the same in every clone.
ALIKE = 5
#: The household with no weight, in every clone.
WEIGHTLESS = 8
#: The household only clone 1 holds.
EXTRA = SOURCES
#: Each source's children, the household only clone 1 holds last.
CHILDREN = (0, 3, 2, 4, 1, 3, 4, 2, 3, 4, 0, 3, 2)
#: A household's weight before its own spread, about one to ten.
WEIGHT = 9_000.0
D, P = 2024, 2025
CLONE = UK_SPINE_ENGINE_BATCH_COLUMN
REGIONS = ("LONDON", "NORTH_WEST", "SCOTLAND", "WALES", "SOUTH_EAST")
INPUTS = (
    *PRIVATE_SCHOOL_ATTENDANCE_PERSON_INPUTS,
    *PRIVATE_SCHOOL_ATTENDANCE_HOUSEHOLD_INPUTS,
)
#: Where a child's draw sits against the rate the formula gives that child:
#: 3% under it (the child attends) or 3% over it (the child does not).
MARGINS = (0.97, 1.03)
#: Every child's draw until the draws are set: no percentile's rate admits it.
UNSET_DRAW = 0.5
#: The draw of a child of no weight, which every positive percentile admits.
SMALL_DRAW = 0.001
#: The people's design weights in all for the floor test: under the formula's
#: one million, while the engine's own weights at the calibration year, grown
#: from them, are over it.
NEAR_FLOOR = 0.9985e6
#: The percentiles the engine gives its attendance rates at.
BREAKPOINTS = tuple(range(0, 101, 5))


def _pool(
    draws: Mapping[int, float] | None = None,
    *,
    scale: float = 1.0,
    clone_column: bool = True,
    one_clone: bool = False,
    clones: tuple[int, ...] = tuple(range(K)),
):
    """The made-up pool at the dataset year. ``draws`` gives each child's
    draw by person id (without it every child draws UNSET_DRAW); ``scale``
    multiplies every weight; ``clone_column`` keeps the clone index on the
    households and ``one_clone`` sets it to one clone for all; ``clones``
    keeps those clones' households only."""

    persons, benunits, households, weights = [], [], [], []
    for clone in clones:
        for source in (*range(SOURCES), *((EXTRA,) if clone == 1 else ())):
            # Household ids interleave the clones; benefit unit ids sort the
            # other way.
            household_id = 10 * source + clone + 1
            benunit_id = 5_000 - 2 * household_id
            alike = source < ALIKE
            drawn = np.random.default_rng(
                7_000 + source + (0 if alike else 101 * clone)
            )
            spread = float(10 ** drawn.uniform(-0.5, 0.5))
            weights.append(0.0 if source == WEIGHTLESS else WEIGHT * scale * spread)
            households.append(
                {
                    "household_id": household_id,
                    **({CLONE: 0 if one_clone else clone} if clone_column else {}),
                    "region": REGIONS[source % len(REGIONS)],
                    "council_tax": 1_400.0 + 100.0 * (source % 4),
                    "rent": 8_400.0 if source % 2 else 0.0,
                    "tenure_type": "RENT_PRIVATELY" if source % 2 else "OWNED_OUTRIGHT",
                }
            )
            adults, children = 1 + source % 2, CHILDREN[source]
            benunits.append(
                {
                    "benunit_id": benunit_id,
                    "dependent_children": children,
                    "is_married": adults == 2,
                }
            )
            for p in range(adults + children):
                child = p >= adults
                person_id = 10 * household_id + p
                # Everyone takes a draw from the household's stream, so alike
                # households stay alike; a child's own is replaced.
                draw = float(drawn.uniform())
                if child:
                    draw = UNSET_DRAW if draws is None else draws[person_id]
                # Some adults have no pay at all, so their households' incomes
                # sit low; the others' pay is the source's, or the clone's own.
                works = not child and (source + p) % 3 != 0
                persons.append(
                    {
                        "person_id": person_id,
                        "person_benunit_id": benunit_id,
                        "person_household_id": household_id,
                        "age": 3 + 3 * (p - adults) if child else 34 + 5 * p,
                        "is_benunit_head": p == 0,
                        "is_parent": children > 0 and not child,
                        "employment_income": (
                            float(drawn.uniform(4_000, 90_000)) if works else 0.0
                        ),
                        "self_employment_income": (
                            float(drawn.uniform(2_000, 30_000))
                            if p == 0 and source % 4 == 1
                            else 0.0
                        ),
                        # Stored at the root, as the spine stores it.
                        "months_since_last_birthday": float(drawn.integers(0, 12)),
                        PRIVATE_SCHOOL_ATTENDANCE_DRAW: draw,
                    }
                )
    person = pd.DataFrame(persons).sample(frac=1.0, random_state=11)
    household = pd.DataFrame(households)
    order = np.argsort(household["household_id"].to_numpy(), kind="stable")
    benunit = pd.DataFrame(benunits).sort_values("benunit_id")
    return uk_national_frame(
        person=person.reset_index(drop=True),
        benunit=benunit.reset_index(drop=True),
        household=household.iloc[order].reset_index(drop=True),
        time_period=D,
        household_weights=np.asarray(weights, dtype=float)[order],
    )


def _store(frame, engine=None):
    return store_private_school_attendance(
        frame, engine=engine or PolicyEngineUKEngine(), calibration_year=P
    )


def _stored(result) -> np.ndarray:
    return result.frame.table("person")[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT].to_numpy()


def _member_of(frame) -> np.ndarray:
    """Each person's household's row in the frame's household table."""

    return pd.Index(frame.table("household")["household_id"]).get_indexer(
        frame.table("person")["person_household_id"]
    )


def _person_weight_total(frame) -> float:
    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    return float(weights[_member_of(frame)].sum())


def _rate(parameters, percentile: np.ndarray) -> np.ndarray:
    """The formula's attendance rate at each percentile: the engine's rates at
    every fifth percentile, interpolated, times its population factor."""

    vat = parameters.gov.simulation.private_school_vat
    rates = [float(vat.private_school_attendance_rate[str(p)]) for p in BREAKPOINTS]
    return np.interp(percentile, BREAKPOINTS, rates) * vat.private_school_factor


@contextmanager
def _ranks():
    """Every call of microdf's weighted percentile rank while open, patched on
    the class so the formula's call and the restatement's are both seen: the
    values ranked, their weights and the ranks returned."""

    from microdf import MicroSeries

    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    rank = MicroSeries.percentile_rank

    @functools.wraps(rank)
    def spy(self):
        ranked = rank(self)
        calls.append(
            (
                pd.Series.to_numpy(self, copy=True),
                self.weights.to_numpy(copy=True),
                pd.Series.to_numpy(ranked, copy=True),
            )
        )
        return ranked

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MicroSeries, "percentile_rank", spy)
        yield calls


def _assert_same_rank(seen, expected) -> None:
    """One rank call is another, bit for bit: the same values and weights, in
    the same order and types, and the same ranks back."""

    for got, want in zip(seen, expected, strict=True):
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(got, want)


@pytest.fixture(scope="module")
def draws():
    """Each child's draw, set 3% under or over the rate the formula gives it
    over the pool, and who attends then. The rates come from the formula's own
    percentiles, seen in its call of microdf's rank in one simulation over the
    pool, and the engine's parameters at the calibration year; the draws
    change nothing else the formula reads. A child of no weight, at
    percentile 0 whose rate is zero, draws SMALL_DRAW."""

    unset = _pool(clone_column=False)
    engine = PolicyEngineUKEngine()
    with _ranks() as calls:
        values = engine.materialize(
            unset,
            [PRIVATE_SCHOOL_ATTENDANCE_OUTPUT, *INPUTS],
            P,
            dataset_period=str(D),
        )
    # The formula ranks once.
    ((_, _, ranked),) = calls
    person_ids = unset.table("person")["person_id"].to_numpy()
    weighted = np.asarray(values["household_weight"])[_member_of(unset)] > 0
    percentile = np.zeros(len(person_ids), dtype=np.int64)
    percentile[weighted] = np.clip(ranked, 0, 100).astype(np.int64)
    rate = _rate(engine.parameters(P), percentile)
    children = np.asarray(values["age_under_18"], dtype=bool)
    # By place in the household, so households alike in every clone keep
    # alike draws.
    margin = np.asarray(MARGINS)[person_ids % 10 % 2]
    draw = np.where(rate > 0, rate * margin, SMALL_DRAW)
    return SimpleNamespace(
        by_person=dict(zip(person_ids[children], draw[children], strict=True)),
        attends=children & (rate > 0) & (margin < 1),
    )


@pytest.fixture(scope="module")
def pool(draws):
    """The pool without the clone index: one simulation over it."""

    return _pool(draws.by_person, clone_column=False)


@pytest.fixture(scope="module")
def engine_values(pool):
    """One simulation's attendance over the pool, beside the inputs the
    restatement reads, at the calibration year from the dataset year, as the
    stage asks for them; the engine's parameters there; and the formula's
    call of microdf's rank."""

    engine = PolicyEngineUKEngine()
    with _ranks() as calls:
        values = engine.materialize(
            pool,
            [PRIVATE_SCHOOL_ATTENDANCE_OUTPUT, *INPUTS],
            P,
            dataset_period=str(D),
        )
    (rank,) = calls
    return SimpleNamespace(
        values=values,
        parameters=engine.parameters(P),
        rank=rank,
        flags=np.asarray(values[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT]).astype(bool),
    )


@pytest.fixture(scope="module")
def stored_single(pool):
    """The stage over the pool in one simulation, and the rank calls made."""

    with _ranks() as calls:
        result = _store(pool)
    return SimpleNamespace(result=result, ranks=calls)


def test_the_pool_is_under_the_floor_per_clone_and_over_it_in_all():
    for clone in range(K):
        assert _person_weight_total(_pool(clones=(clone,))) < 1e6
    assert _person_weight_total(_pool()) > 1e6
    assert _person_weight_total(_pool(scale=0.25)) < 1e6


def test_the_restatement_is_the_engines_formula_bit_for_bit(pool, engine_values, draws):
    with _ranks() as calls:
        restated = restate_private_school_attendance(
            pool, engine_values.values, engine_values.parameters
        )

    # One call of microdf's rank, handed the formula's net incomes and
    # weights in the formula's order and types, and returning its percentiles.
    (rank,) = calls
    _assert_same_rank(rank, engine_values.rank)
    assert restated.dtype == bool
    np.testing.assert_array_equal(restated, engine_values.flags)
    # Every child sits where its draw was set: 3% under its rate it attends,
    # 3% over it not, so these flags pin the rank, the rates, their
    # interpolation and the factor to a few percent.
    np.testing.assert_array_equal(engine_values.flags, draws.attends)
    person = pool.table("person")
    children = person["age"].to_numpy() < 18
    # Some children attend and some do not; no adult does.
    assert engine_values.flags[children].any()
    assert not engine_values.flags[children].all()
    assert not engine_values.flags[~children].any()
    # The weightless household's children keep percentile 0, whose rate is
    # zero, so even a draw every positive percentile admits leaves them out.
    source = (person["person_household_id"].to_numpy() - 1) // 10
    weightless = source == WEIGHTLESS
    assert (weightless & children).any()
    assert not engine_values.flags[weightless].any()
    # The rank meets ties: households alike in every clone, and members of one
    # household, share a net income.
    weighted = (
        np.asarray(engine_values.values["household_weight"])[_member_of(pool)] > 0
    )
    net_income, _, _ = engine_values.rank
    tied = pd.DataFrame({"net": net_income, "household": _member_of(pool)[weighted]})
    assert (tied.groupby("net")["household"].nunique() > 1).any()
    assert tied.groupby("household")["net"].size().max() > 1


def test_one_simulation_ranks_in_the_formula_then_the_restatement(
    engine_values, stored_single
):
    # The engine's formula ranks the pool, then the restatement ranks it
    # again, from the same values and weights to the same percentiles.
    formula, restatement = stored_single.ranks
    _assert_same_rank(formula, engine_values.rank)
    _assert_same_rank(restatement, engine_values.rank)


def test_the_stage_run_clone_by_clone_stores_what_one_simulation_stores(
    draws, engine_values, stored_single
):
    frame = _pool(draws.by_person)
    engine = PolicyEngineUKEngine()
    assert engine.simulation_batches(frame) == K

    with _ranks() as calls:
        batched = _store(frame, engine)

    # No clone's simulation computes attendance, so the one rank is the
    # restatement's over the pool: the formula's own, bit for bit.
    (rank,) = calls
    _assert_same_rank(rank, engine_values.rank)
    single = stored_single.result
    np.testing.assert_array_equal(_stored(single), engine_values.flags)
    np.testing.assert_array_equal(_stored(batched), _stored(single))
    assert batched.rank_route == PRIVATE_SCHOOL_ATTENDANCE_POOLED_ROUTE
    assert batched.engine_parity_checked is False
    assert single.rank_route == PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE
    assert single.engine_parity_checked is True
    pupils = single.evidence()["weighted_pupils"]
    assert batched.evidence()["weighted_pupils"] == pupils
    # Why the stage restates: one clone's own simulation ranks that clone
    # alone, and its people weigh under the floor, so it imputes nobody where
    # the pool imputes some of the same people.
    alone = _pool(draws.by_person, clone_column=False, clones=(1,))
    own = PolicyEngineUKEngine().materialize(
        alone, [PRIVATE_SCHOOL_ATTENDANCE_OUTPUT], P, dataset_period=str(D)
    )
    assert not np.asarray(own[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT]).any()
    in_clone = frame.table("household").set_index("household_id")[CLONE]
    of_clone_1 = (
        frame.table("person")["person_household_id"].map(in_clone).to_numpy() == 1
    )
    assert _stored(batched)[of_clone_1].any()


class _PerturbedEngine(PolicyEngineUKEngine):
    """The adapter, its parameter tree giving another population factor: a
    tree the engine's own simulations do not hold."""

    def __init__(self, factor):
        super().__init__()
        self._factor = factor

    def parameters(self, period):
        node = super().parameters(period).gov.simulation.private_school_vat
        perturbed = SimpleNamespace(
            private_school_attendance_rate=node.private_school_attendance_rate,
            private_school_factor=self._factor(node.private_school_factor),
        )
        return SimpleNamespace(
            gov=SimpleNamespace(
                simulation=SimpleNamespace(private_school_vat=perturbed)
            )
        )


@pytest.mark.parametrize(
    "factor",
    [lambda factor: 0.0, lambda factor: 0.94 * factor],
    ids=["no_factor", "factor_six_percent_lower"],
)
def test_a_perturbed_parameter_breaks_the_parity_check(pool, stored_single, factor):
    # The engine's own flags have pupils, each 3% under its rate, so a
    # restatement with no factor, or one 6% lower, differs from them.
    assert _stored(stored_single.result).any()

    with pytest.raises(ValueError, match="differs from the engine's flags"):
        _store(pool, _PerturbedEngine(factor))


def test_below_the_floor_nobody_attends_on_either_route(draws):
    clones = _pool(draws.by_person, scale=0.25)
    engine = PolicyEngineUKEngine()
    assert engine.simulation_batches(clones) == K

    batched = _store(clones, engine)
    single = _store(_pool(draws.by_person, scale=0.25, clone_column=False))

    assert batched.rank_route == PRIVATE_SCHOOL_ATTENDANCE_POOLED_ROUTE
    assert single.rank_route == PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE
    assert not _stored(batched).any()
    assert not _stored(single).any()


def test_the_floor_reads_the_engines_weights_at_the_calibration_year(draws):
    """The formula's one-million floor reads the engine's weights at the
    calibration year, grown from the dataset year's design weights. Design
    weights just under the floor are over it there, so both routes impute,
    as one simulation's formula does."""

    scale = NEAR_FLOOR / _person_weight_total(_pool())
    frame = _pool(draws.by_person, scale=scale, clone_column=False)
    assert _person_weight_total(frame) < 1e6

    single = _store(frame)
    batched = _store(_pool(draws.by_person, scale=scale))

    # The engine's own flags, stored once the restatement equalled them, have
    # pupils, so its formula read weights over the floor; clone by clone, the
    # restatement reads the same weights.
    assert single.rank_route == PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE
    assert _stored(single).any()
    assert batched.rank_route == PRIVATE_SCHOOL_ATTENDANCE_POOLED_ROUTE
    np.testing.assert_array_equal(_stored(batched), _stored(single))


def test_a_frame_of_one_clone_stores_the_engines_flags_as_before(draws, engine_values):
    frame = _pool(draws.by_person, one_clone=True)
    engine = PolicyEngineUKEngine()
    assert engine.simulation_batches(frame) == 1

    result = _store(frame, engine)

    # What the stage asked the engine for before it restated anything, its
    # attendance alone over the same frame, is what it stores; one simulation
    # over the pool without the clone index gives the same.
    before = PolicyEngineUKEngine().materialize(
        frame, [PRIVATE_SCHOOL_ATTENDANCE_OUTPUT], P, dataset_period=str(D)
    )
    np.testing.assert_array_equal(
        _stored(result),
        np.asarray(before[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT]).astype(bool),
    )
    np.testing.assert_array_equal(_stored(result), engine_values.flags)
    assert result.rank_route == PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE
    assert result.engine_parity_checked is True
