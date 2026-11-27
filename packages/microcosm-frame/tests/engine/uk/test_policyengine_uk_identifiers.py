"""The ids each policyengine-uk simulation holds, through the adapter (microcosm#931).

policyengine-core builds a simulation's entities and memberships from the
int64 id columns it is given, and stores those columns as the values of
policyengine-uk's five id variables, converted to int32 by numpy's
``astype``, which wraps an id outside the type without an error. Two ids that
wrap to one value therefore merge no entity, but every formula that reads the
stored id sees one key: policyengine-uk hashes its Warm Home Discount draw on
household_id, its Universal Credit deduction draws on benunit_id and its
birth-month spread on person_id. The adapter gives each simulation the
frame's ids where the store keeps them apart, and 1 plus each id's rank among
that simulation's own ids for an entity whose stored ids would repeat
(``microcosm.frame.adapters.engine_identifiers``). Held here through the
installed engine:

- the adapter's int32 declaration is the engine's (test 11);
- ids past 2**31 that stay apart once stored, laid out as copy ids are, reach
  the engine as they did before the rule: every value is a direct
  Microsimulation's over the frame's engine tables, at 2024 and at 2025 as the
  dataset's year and at 2025 projected from 2024, and the engine holds the
  frame's rows in its order (test 12);
- households whose ids store as one value (7 and 7 + 2**32) take ranks, so
  their draws differ and every value is the same call's over the frame
  renumbered by hand: at the dataset's year (test 13), projected a year on
  through policyengine-uk's uprating, whose council-tax and student-loan
  joins read the raw memberships (test 13b), and clone by clone, each batch
  ranking its own rows (test 14). All three fail where the adapter passes the
  frame's ids through unchanged;
- policyengine-uk's ``EntityIdWarning`` is filtered at both of the adapter's
  construction sites, and a direct Microsimulation over the same tables still
  records it (test 15).

Every household and id here is made up. Assertion messages name variables,
entities and periods only, never an id or a value.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from microcosm.frame import Frame
from microcosm.frame.adapters.engine_identifiers import (
    identifier_columns,
    stored_identifier_repeats,
)
from microcosm.frame.adapters.engine_population import household_batches
from microcosm.frame.adapters.policyengine_uk import (
    UK_ENGINE_IDENTIFIERS,
    UK_SCHEMA,
    PolicyEngineUKEngine,
    frame_isa_income_contract,
    frame_savings_income_projection,
)
from microcosm.frame.adapters.policyengine_uk_population import (
    UK_SPINE_ENGINE_BATCH_COLUMN,
)
from microcosm.frame.materialize import engine_tables
from test_support.microcosm_frame.engine_identifier_frames import (
    COLLIDING_CLONE_HOUSEHOLD_IDS,
    COLLIDING_HOUSEHOLD_IDS,
    clone_layout_frame,
    colliding_clones_frame,
    colliding_household_frame,
    copy_layout_frame,
    relabelled,
    wrapped,
)

D, P = 2024, 2025
INT32 = np.dtype(np.int32)
#: The id-keyed draws policyengine-uk hashes on stored ids: households',
#: benefit units' (two), and the birth-month spread on people's.
ID_DRAWS = (
    "warm_home_discount_random_draw",
    "uc_deduction_random_draw",
    "uc_deduction_type_random_draw",
    "months_since_last_birthday",
)
#: Test 12's requests: the draws, values over the whole structure, each
#: household's member count and an echoed household input.
END_TO_END = (
    *ID_DRAWS,
    "hbai_household_net_income",
    "universal_credit",
    "income_tax",
    "household_count_people",
    "council_tax",
)
#: Tests 13 and 13b's requests: test 12's, the two person inputs the uprating
#: joins rewrite, and the person and benefit-unit ids, which fit int32 and so
#: may be asked for.
COLLIDING = (
    *END_TO_END,
    "council_tax_benefit_reported",
    "student_loan_plan",
    "person_id",
    "benunit_id",
)
#: Test 14's requests, household-local and free of the pool's shares, as
#: test_policyengine_uk_batches.py splits LOCAL from SHARE_DEPENDENT_TAXES: a
#: standalone call over one batch gets no pool shares, so values that depend
#: on them are not compared. months_since_last_birthday is stored, as the
#: spine's batched pass requires.
BATCH_LOCAL = (
    *ID_DRAWS,
    "income_tax",
    "national_insurance",
    "universal_credit",
    "household_count_people",
    "council_tax",
)

#: The colliding frame's people, as colliding_household_frame lists them.
COLLIDING_PEOPLE = (34, 31, 33, 35, 36, 32)
#: Each of those people's inputs. People 31 and 36 hold no student loan, are
#: tertiary-educated and aged 31, an age at which policyengine-uk's
#: projection from 2024 to 2025 adds new holders (its extended 2012 intake),
#: and both draw a loan on their person ids, so the plan each takes follows
#: the region of its household: the first (Scotland) and the third (Wales).
#: The colliding households' adults report Council Tax Benefit, which the
#: projection carries by their household's council-tax factor.
COLLIDING_INPUTS = {
    "age": {34: 38.0, 31: 31.0, 33: 45.0, 35: 12.0, 36: 31.0, 32: 6.0},
    "employment_income": {
        34: 31_000.0,
        31: 24_500.0,
        33: 18_000.0,
        35: 0.0,
        36: 12_000.0,
        32: 0.0,
    },
    "highest_education": {
        34: "UPPER_SECONDARY",
        31: "TERTIARY",
        33: "TERTIARY",
        35: "PRIMARY",
        36: "TERTIARY",
        32: "PRIMARY",
    },
    "student_loan_plan": dict.fromkeys(COLLIDING_PEOPLE, "NONE"),
    "student_loan_repayments": dict.fromkeys(COLLIDING_PEOPLE, 0.0),
    "council_tax_benefit_reported": {
        34: 185.5,
        31: 210.0,
        33: 0.0,
        35: 0.0,
        36: 96.25,
        32: 0.0,
    },
}
#: The colliding frame's households, in row order (7, 9 and 7 + 2**32): the
#: first and third in different nations, the second in Northern Ireland,
#: whose council tax policyengine-uk holds flat.
COLLIDING_HOUSEHOLDS = {
    "region": ["SCOTLAND", "NORTHERN_IRELAND", "WALES"],
    "council_tax": [1_210.0, 0.0, 1_450.25],
    "rent": [0.0, 4_800.0, 7_800.0],
    "tenure_type": ["OWNED_OUTRIGHT", "RENT_FROM_COUNCIL", "RENT_PRIVATELY"],
}


def _quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **kwargs)


def _check(condition, what: str) -> None:
    """Assert ``condition``; a failure names only ``what``, never the ids or
    values the condition compared."""

    holds = bool(condition)
    assert holds, what


def _assert_same(left, right, what: str) -> None:
    """Two returned arrays are equal bit for bit, and of one dtype; a failure
    names only ``what``."""

    left, right = np.asarray(left), np.asarray(right)
    same = left.dtype == right.dtype and left.shape == right.shape
    if same and left.dtype.kind == "O":
        same = left.tolist() == right.tolist()
    elif same:
        same = left.tobytes() == right.tobytes()
    assert same, f"{what} differs"


def _snapshot(frame: Frame) -> dict:
    return {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}


def _assert_unchanged(frame: Frame, before: dict) -> None:
    for entity, table in before.items():
        unchanged = frame.table(entity).equals(table) and list(
            frame.table(entity).dtypes
        ) == list(table.dtypes)
        assert unchanged, f"the frame's {entity} table changed"


def _with_columns(frame: Frame, entity: str, columns) -> Frame:
    """``frame`` with ``columns`` added to its ``entity`` table, in row order."""

    tables = {name: frame.table(name).copy() for name in frame.entities}
    for column, values in columns.items():
        tables[entity][column] = list(values)
    return Frame(
        tables,
        frame.schema,
        {name: frame.weights_for(name) for name in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _household_inputs(count: int, **columns) -> dict:
    """The household inputs policyengine-uk's projection of a dataset reads
    (region, council tax, rent and tenure), one value each unless given."""

    inputs = {
        "region": ["LONDON"] * count,
        "council_tax": [1_400.0] * count,
        "rent": [0.0] * count,
        "tenure_type": ["OWNED_OUTRIGHT"] * count,
    }
    inputs.update(columns)
    return inputs


def _member_counts(frame: Frame) -> np.ndarray:
    """Each household row's own number of people in the frame."""

    members = frame.table("person")["person_household_id"].value_counts()
    return frame.table("household")["household_id"].map(members).to_numpy()


def _dataset(frame: Frame, tables, fiscal_year: int):
    """The dataset the adapter built before the rank rule: its engine tables'
    copies, as they come."""

    from policyengine_uk.data import UKSingleYearDataset

    return UKSingleYearDataset(
        person=tables["person"].copy(),
        benunit=tables["benunit"].copy(),
        household=tables["household"].copy(),
        fiscal_year=fiscal_year,
        isa_income_contract=frame_isa_income_contract(frame),
        savings_income_projection=frame_savings_income_projection(frame),
    )


def _direct_values(frame: Frame, names, period: int, dataset_period: int | None):
    """``names`` from a Microsimulation built directly over the frame's
    engine tables: the adapter's path before the rank rule."""

    from policyengine_uk import Microsimulation

    tables = engine_tables(frame, weighted_entities=("household",))
    dataset = _dataset(
        frame, tables, period if dataset_period is None else dataset_period
    )
    simulation = _quiet(Microsimulation, dataset=dataset)
    return {
        name: np.asarray(_quiet(simulation.calculate, name, period=period))
        for name in names
    }


def _entity_id_warned(call, *args, **kwargs) -> bool:
    """Whether ``call`` raised policyengine-uk's ``EntityIdWarning``, under
    ``simplefilter("always")``: a boolean, never the warning's text."""

    from policyengine_uk.utils.entity_ids import EntityIdWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call(*args, **kwargs)
    return any(issubclass(item.category, EntityIdWarning) for item in caught)


@pytest.fixture(scope="module")
def system():
    from policyengine_uk import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


@pytest.fixture(scope="module")
def engine():
    return PolicyEngineUKEngine()


# ---------------------------------------------------------------------------
# Test 11: the declaration is the installed engine's
# ---------------------------------------------------------------------------


def test_the_id_declaration_is_the_installed_engines(system):
    from policyengine_uk.utils.entity_ids import ENTITY_ID_VARIABLES

    columns = identifier_columns(UK_SCHEMA)
    assert len(ENTITY_ID_VARIABLES) == len(set(ENTITY_ID_VARIABLES)) == len(columns)
    assert set(ENTITY_ID_VARIABLES) == set(columns)
    for column in columns:
        variable = system.variables[column]
        assert variable.value_type is int, column
        assert np.dtype(variable.dtype) == UK_ENGINE_IDENTIFIERS.stored_dtype, column
    assert UK_ENGINE_IDENTIFIERS.stored_dtype == INT32
    adapter = PolicyEngineUKEngine()
    adapter._system = system
    adapter._require_identifier_dtypes()


# ---------------------------------------------------------------------------
# Test 12: ids that stay apart once stored reach the engine as before
# ---------------------------------------------------------------------------


def _copy_layout_frame() -> Frame:
    """Test 12's frame: every entity's ids laid out as copy ids are, six
    households of different sizes in different regions, adults in work."""

    frame = copy_layout_frame(
        household_columns=_household_inputs(
            6,
            region=[
                "LONDON",
                "SCOTLAND",
                "WALES",
                "NORTH_WEST",
                "SOUTH_EAST",
                "NORTHERN_IRELAND",
            ],
            council_tax=[1_210.0, 1_307.5, 1_405.0, 1_502.5, 1_600.0, 1_697.5],
            rent=[0.0, 6_600.0, 0.0, 8_400.0, 0.0, 5_100.0],
            tenure_type=[
                "OWNED_OUTRIGHT",
                "RENT_PRIVATELY",
                "OWNED_WITH_MORTGAGE",
                "RENT_PRIVATELY",
                "OWNED_OUTRIGHT",
                "RENT_FROM_COUNCIL",
            ],
        )
    )
    age = frame.table("person")["age"].to_numpy()
    pay = 7_000.0 + 2_750.0 * np.arange(len(age))
    return _with_columns(
        frame, "person", {"employment_income": np.where(age >= 18, pay, 0.0)}
    )


@pytest.mark.parametrize(
    ("period", "dataset_period"),
    [(D, None), (P, None), (P, D)],
    ids=["2024", "2025", "2025_projected_from_2024"],
)
def test_ids_that_stay_apart_once_stored_give_the_direct_simulations_values(
    engine, period, dataset_period
):
    # The end-to-end half of the proof that a frame without a stored repeat
    # reaches the engine unchanged: ids past 2**31 that stay apart once
    # stored are handed to the engine as they were before the rank rule, so
    # every value, id-keyed draws included, is a direct Microsimulation's over
    # the frame's engine tables.
    frame = _copy_layout_frame()
    tables = engine_tables(frame, weighted_entities=("household",))
    # The layout's own claims: every entity's ids pass 2**31 and stay apart
    # once stored, and the households' stored ids are out of int64 order.
    assert stored_identifier_repeats(UK_SCHEMA, tables, UK_ENGINE_IDENTIFIERS) == ()
    for entity in UK_SCHEMA.entities:
        ids = tables[entity][UK_SCHEMA.entity_id_column(entity)]
        _check(int(ids.max()) > 2**31, f"the {entity} ids fit int32")
    stored = [wrapped(value, INT32) for value in tables["household"]["household_id"]]
    _check(stored != sorted(stored), "the household ids store in int64 order")
    before = _snapshot(frame)

    values = _quiet(
        engine.materialize, frame, END_TO_END, period, dataset_period=dataset_period
    )

    direct = _direct_values(frame, END_TO_END, period, dataset_period)
    for name in END_TO_END:
        _assert_same(values[name], direct[name], f"{name} at {period}")
    # Frame truth, which the comparison cannot give, since an engine that
    # built its groups from stored ids would reorder both sides alike: each
    # household row's own member count (all different), and at the dataset's
    # year an echoed household input with distinct values, in row order.
    counts = _member_counts(frame)
    _check(len(set(counts.tolist())) == len(counts), "two households are one size")
    _assert_same(
        values["household_count_people"].astype(np.int64),
        counts.astype(np.int64),
        "household_count_people against the frame's members",
    )
    if dataset_period is None:
        council_tax = frame.table("household")["council_tax"].to_numpy()
        _check(
            len(set(council_tax.tolist())) == len(council_tax),
            "two households pay one council tax",
        )
        _assert_same(
            values["council_tax"].astype(np.float64),
            council_tax,
            "council_tax against the frame's column",
        )
    _check(np.any(values["income_tax"] != 0), f"no income tax at {period}")
    _assert_unchanged(frame, before)


# ---------------------------------------------------------------------------
# Tests 13 and 13b: households whose ids store as one value
# ---------------------------------------------------------------------------


def _colliding_frame(household_ids=COLLIDING_HOUSEHOLD_IDS) -> Frame:
    """Tests 13 and 13b's frame: households 7, 9 and 7 + 2**32, or
    ``household_ids`` for the same frame renumbered by hand."""

    return colliding_household_frame(
        household_ids,
        person_columns={
            column: [by_person[person] for person in COLLIDING_PEOPLE]
            for column, by_person in COLLIDING_INPUTS.items()
        },
        household_columns=COLLIDING_HOUSEHOLDS,
    )


def _colliding_runs(engine, period: int, dataset_period: int | None):
    """One call over the colliding frame, and the same call over it
    renumbered by hand to households 1, 2 and 3."""

    frame = _colliding_frame()
    # The layout's own claims: the people are listed as the inputs assume,
    # and the first and third households store as one int32 value.
    _check(
        tuple(frame.table("person")["person_id"].tolist()) == COLLIDING_PEOPLE,
        "the people are not listed as the inputs assume",
    )
    first, _second, third = COLLIDING_HOUSEHOLD_IDS
    _check(
        wrapped(first, INT32) == wrapped(third, INT32),
        "the first and third households store apart",
    )
    before = _snapshot(frame)
    values = _quiet(
        engine.materialize, frame, COLLIDING, period, dataset_period=dataset_period
    )
    hand = _quiet(
        engine.materialize,
        _colliding_frame((1, 2, 3)),
        COLLIDING,
        period,
        dataset_period=dataset_period,
    )
    return SimpleNamespace(frame=frame, before=before, values=values, hand=hand)


def _assert_ranked(runs, period: int) -> None:
    """What tests 13 and 13b share."""

    values, hand, frame = runs.values, runs.hand, runs.frame
    draws = values["warm_home_discount_random_draw"]
    # The colliding households' draws differ: without the rank rule both hash
    # the one stored id and are equal.
    _check(
        draws[0] != draws[2], f"the colliding households' draws are equal at {period}"
    )
    # Ranks keep order and change nothing else: every value is the hand
    # renumbering's, bit for bit.
    for name in COLLIDING:
        _assert_same(values[name], hand[name], f"{name} at {period}")
    _assert_same(
        values["household_count_people"].astype(np.int64),
        _member_counts(frame).astype(np.int64),
        "household_count_people against the frame's members",
    )
    _assert_unchanged(frame, runs.before)


def test_households_whose_ids_store_together_take_ranks(engine):
    # Test 13, at the dataset's year.
    runs = _colliding_runs(engine, P, None)
    _assert_ranked(runs, P)
    council_tax = runs.frame.table("household")["council_tax"].to_numpy()
    _assert_same(
        runs.values["council_tax"].astype(np.float64),
        council_tax,
        "council_tax against the frame's column",
    )


def test_households_whose_ids_store_together_take_ranks_when_projected(engine):
    # Test 13b: as test 13, projected from 2024, as uc_take_up and
    # pension_credit_take_up call. policyengine-uk's uprating then maps each
    # household's council-tax factor onto its people by person_household_id,
    # and merges region onto people by it for new student-loan holders: joins
    # on the raw membership column, which a membership map that missed its
    # group's new id would misattach.
    runs = _colliding_runs(engine, P, D)
    _assert_ranked(runs, P)
    # The joins are exercised: the colliding households' council-tax factors
    # differ, and their new borrowers take different plans.
    row = {person: index for index, person in enumerate(COLLIDING_PEOPLE)}
    reported = runs.values["council_tax_benefit_reported"].astype(np.float64)
    given = COLLIDING_INPUTS["council_tax_benefit_reported"]
    factors = [reported[row[person]] / given[person] for person in (31, 34)]
    _check(factors[0] != factors[1], "the colliding households share a factor")
    plans = runs.values["student_loan_plan"]
    _check(plans[row[31]] != plans[row[36]], "the new borrowers take one plan")
    _check(
        "NONE" not in (plans[row[31]], plans[row[36]]), "a new borrower takes no plan"
    )


# ---------------------------------------------------------------------------
# Test 14: clone by clone, each batch ranks its own rows
# ---------------------------------------------------------------------------


def _clones_frame() -> Frame:
    """Test 14's frame: two clones, each holding two households whose ids
    store as one int32 value, adults in work."""

    return colliding_clones_frame(
        person_columns={
            "employment_income": [26_000.0, 0.0, 38_500.0, 14_250.0, 0.0, 47_000.0]
        },
        household_columns=_household_inputs(
            4,
            region=["LONDON", "LONDON", "WALES", "WALES"],
            council_tax=[1_380.0, 1_392.5, 1_455.0, 1_467.5],
        ),
    )


def _scatter(frame: Frame, entity: str, parts) -> np.ndarray:
    """Per-batch values on ``frame``'s ``entity`` rows, each placed by the
    batch frame's own id."""

    column = UK_SCHEMA.entity_id_column(entity)
    placed = {}
    for batch_ids, values in parts:
        for identifier, value in zip(batch_ids, values, strict=True):
            _check(identifier not in placed, f"two batches return one {entity} row")
            placed[identifier] = value
    ids = [int(value) for value in frame.table(entity)[column]]
    _check(set(placed) == set(ids), f"the batches miss a {entity} row")
    return np.asarray(
        [placed[identifier] for identifier in ids], dtype=parts[0][1].dtype
    )


def test_each_batch_ranks_its_own_rows(engine):
    # Test 14: within each clone's batch the two households' ids store as one
    # value (7, and 1000000007 for clone 1), while people and benefit units
    # pass 2**31 and stay apart. Each batch ranks its own households, so its
    # values are those of a standalone call over the batch renumbered by hand
    # to households 1 and 2.
    frame = _clones_frame()
    before = _snapshot(frame)
    _check(engine.simulation_batches(frame) == 2, "the frame runs in other batches")

    values = _quiet(engine.materialize, frame, BATCH_LOCAL, P)

    batches = household_batches(frame, UK_SPINE_ENGINE_BATCH_COLUMN)
    _check(
        [batch.key for batch in batches] == list(COLLIDING_CLONE_HOUSEHOLD_IDS),
        "the batches follow other clones",
    )
    draws = values["warm_home_discount_random_draw"]
    entities = {name: engine.variable_metadata(name).entity for name in BATCH_LOCAL}
    parts = {name: [] for name in BATCH_LOCAL}
    for batch, households in zip(
        batches, COLLIDING_CLONE_HOUSEHOLD_IDS.values(), strict=True
    ):
        part = frame.select(batch.persons)
        _check(
            part.table("household")["household_id"].tolist() == list(households),
            "a batch holds other households",
        )
        _check(
            len({wrapped(value, INT32) for value in households}) == 1,
            "a batch's households store apart",
        )
        # The batch's two draws differ: without the rank rule both hash the
        # one stored id and are equal.
        own = draws[batch.households]
        _check(own[0] != own[1], f"a batch's two draws are equal at {P}")
        alone = _quiet(
            engine.materialize, relabelled(part, "household", (1, 2)), BATCH_LOCAL, P
        )
        for name in BATCH_LOCAL:
            entity = entities[name]
            ids = part.table(entity)[UK_SCHEMA.entity_id_column(entity)]
            parts[name].append(([int(value) for value in ids], alone[name]))
    for name in BATCH_LOCAL:
        expected = _scatter(frame, entities[name], parts[name])
        _assert_same(values[name], expected, f"{name} at {P}")
    _check(np.any(values["income_tax"] != 0), f"no income tax at {P}")
    _assert_unchanged(frame, before)


# ---------------------------------------------------------------------------
# Test 15: the warning is filtered at both construction sites
# ---------------------------------------------------------------------------


def test_the_id_warning_is_filtered_where_one_simulation_is_built(engine):
    # Test 15, unbatched: the shifted clone layout, whose ids pass 2**31 and
    # stay apart.
    from policyengine_uk import Microsimulation

    frame = clone_layout_frame()
    frame = _with_columns(frame, "household", _household_inputs(frame.n("household")))
    tables = engine_tables(frame, weighted_entities=("household",))
    # policyengine-uk still warns on ids outside int32 ...
    _check(
        _entity_id_warned(Microsimulation, dataset=_dataset(frame, tables, P)),
        "a direct Microsimulation no longer warns",
    )
    # ... and the adapter's simulation does not pass it on.
    _check(
        not _entity_id_warned(engine.materialize, frame, ["income_tax"], P),
        "the adapter's simulation passed the warning on",
    )


def test_the_id_warning_is_filtered_where_batches_are_built(engine):
    # Test 15, batched: test 14's frame, whose people pass 2**31 in every
    # batch. This is the site the spine's calls over more than one clone use.
    from policyengine_uk import Microsimulation

    frame = _clones_frame()
    batches = household_batches(frame, UK_SPINE_ENGINE_BATCH_COLUMN)
    part = frame.select(batches[0].persons)
    tables = engine_tables(part, weighted_entities=("household",))
    _check(
        _entity_id_warned(Microsimulation, dataset=_dataset(part, tables, P)),
        "a direct Microsimulation over one batch no longer warns",
    )
    _check(engine.simulation_batches(frame) == 2, "the frame runs in other batches")
    _check(
        not _entity_id_warned(engine.materialize, frame, ["income_tax"], P),
        "the adapter's batched simulations passed the warning on",
    )
