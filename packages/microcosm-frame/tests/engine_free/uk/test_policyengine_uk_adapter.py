"""Tests split from packages/microcosm-frame/tests/test_policyengine_uk_adapter.py."""

# ruff: noqa: F403, F405
import sys
import types
import warnings

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.adapters import policyengine_uk as adapter_module
from microcosm.frame.adapters.engine_identifiers import (
    identifier_columns,
    inexact_identifier_entities,
    stored_identifier_repeats,
)
from microcosm.frame.adapters.engine_population import SPINE_BATCHED_PASS
from microcosm.frame.adapters.policyengine_uk import UK_ENGINE_IDENTIFIERS
from microcosm.frame.adapters.policyengine_uk_population import (
    UK_ENGINE_POPULATION,
    UK_SPINE_ENGINE_BATCH_COLUMN,
)
from microcosm.frame.materialize import engine_tables
from test_support.microcosm_frame.engine_identifier_frames import (
    COLLIDING_CLONE_HOUSEHOLD_IDS,
    clone_layout_frame,
    colliding_clones_frame,
    relabelled,
    wrapped,
)
from test_support.microcosm_frame.policyengine_uk_adapter import *


def test_policyengine_uk_adapter_satisfies_rules_protocol_without_importing_engine() -> (
    None
):
    adapter = PolicyEngineUKEngine()

    assert isinstance(adapter, RulesEngine)
    assert adapter.country == "uk"
    assert adapter.entity_schema() == UK_SCHEMA


def test_policyengine_uk_adapter_export_side_is_not_implemented() -> None:
    adapter = PolicyEngineUKEngine()

    with pytest.raises(NotImplementedError, match="write_uk_national_frame"):
        adapter.write_dataset(object(), "unused.h5", period=2023)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "defect", ["missing", "not_loaded", "changed", "entity", "dtype", "period"]
)
def test_adapter_refuses_unconsumed_explicit_uc_roles(monkeypatch, defect):
    definition = SimpleNamespace(
        entity=SimpleNamespace(key="person"), value_type=bool, definition_period="year"
    )
    variables = {
        "is_uc_claimant": definition,
        "universal_credit": SimpleNamespace(
            entity=SimpleNamespace(key="benunit"), value_type=float
        ),
    }
    if defect == "missing":
        del variables["is_uc_claimant"]
    elif defect == "entity":
        definition.entity.key = "benunit"
    elif defect == "dtype":
        definition.value_type = float
    elif defect == "period":
        definition.definition_period = "month"
    calls = []

    def calculate(name, *args, **kwargs):
        calls.append(name)
        return np.array([False if defect == "changed" else True])

    simulation = SimpleNamespace(
        tax_benefit_system=SimpleNamespace(variables=variables),
        input_variables=[] if defect == "not_loaded" else ["is_uc_claimant"],
        calculate=calculate,
    )
    engine = PolicyEngineUKEngine()
    engine._system = simulation.tax_benefit_system
    monkeypatch.setattr(engine, "_build_dataset", lambda *args: object())
    monkeypatch.setattr(
        engine,
        "_import_policyengine_uk",
        lambda: SimpleNamespace(Microsimulation=lambda **kwargs: simulation),
    )
    bundle = SimpleNamespace(
        table=lambda entity: pd.DataFrame({"is_uc_claimant": [True]}),
        n=lambda entity: 1,
    )
    with pytest.raises(ValueError, match="is_uc_claimant"):
        engine.materialize(bundle, ["universal_credit"], 2025)
    assert "universal_credit" not in calls


def _person_role():
    return SimpleNamespace(
        entity=SimpleNamespace(key="person"), value_type=bool, definition_period="year"
    )


_ROLES = ("is_uc_claimant", "is_claimant_or_partner", "is_hbai_dependent_child")
_SUPPLIED = {
    "is_uc_claimant": np.array([True, False]),
    "is_claimant_or_partner": np.array([True, False]),
    "is_hbai_dependent_child": np.array([False, True]),
}


def _role_engine(monkeypatch, variables, *, loaded, values):
    calls = []

    def calculate(name, *args, **kwargs):
        calls.append(name)
        return np.asarray(values[name]) if name in values else np.zeros(2)

    simulation = SimpleNamespace(
        tax_benefit_system=SimpleNamespace(variables=variables),
        input_variables=loaded,
        calculate=calculate,
    )
    engine = PolicyEngineUKEngine()
    engine._system = simulation.tax_benefit_system
    monkeypatch.setattr(engine, "_build_dataset", lambda *args: object())
    monkeypatch.setattr(
        engine,
        "_import_policyengine_uk",
        lambda: SimpleNamespace(Microsimulation=lambda **kwargs: simulation),
    )
    bundle = SimpleNamespace(
        table=lambda entity: pd.DataFrame(_SUPPLIED), n=lambda entity: 2
    )
    return engine, bundle, calls


_OUTPUT = SimpleNamespace(entity=SimpleNamespace(key="person"), value_type=float)


def test_the_validated_roles_are_the_supplied_person_types() -> None:
    assert tuple(name for name, _ in SUPPLIED_ROLE_INPUTS) == _ROLES


@pytest.mark.parametrize("role", ["is_claimant_or_partner", "is_hbai_dependent_child"])
@pytest.mark.parametrize(
    "defect", ["missing", "not_loaded", "changed", "entity", "dtype", "period"]
)
def test_adapter_refuses_unconsumed_person_type_roles(monkeypatch, role, defect):
    # policyengine-uk 2.107.0 reads both person types across its programmes;
    # a supplied role must reach the model as an input, unchanged.
    variables = {name: _person_role() for name in _ROLES} | {"income_tax": _OUTPUT}
    if defect == "missing":
        del variables[role]
    elif defect == "entity":
        variables[role].entity.key = "benunit"
    elif defect == "dtype":
        variables[role].value_type = float
    elif defect == "period":
        variables[role].definition_period = "month"
    loaded = [name for name in _ROLES if not (defect == "not_loaded" and name == role)]
    values = dict(_SUPPLIED)
    if defect == "changed":
        values[role] = ~values[role]
    engine, bundle, calls = _role_engine(
        monkeypatch, variables, loaded=loaded, values=values
    )
    with pytest.raises(ValueError, match=role):
        engine.materialize(bundle, ["income_tax"], 2025)
    assert "income_tax" not in calls


def test_adapter_accepts_person_type_roles_the_model_reads(monkeypatch):
    variables = {name: _person_role() for name in _ROLES} | {"income_tax": _OUTPUT}
    engine, bundle, calls = _role_engine(
        monkeypatch, variables, loaded=list(_ROLES), values=_SUPPLIED
    )
    engine.materialize(bundle, ["income_tax"], 2025)
    assert calls == [*_ROLES, "income_tax"]


# ---------------------------------------------------------------------------
# Calls over more than one clone run clone by clone (microcosm#931)
# ---------------------------------------------------------------------------


def _clone_frame(clones=(0, 1, 2), *, column=True, missing=False):
    """Two households per clone, one person each, clones interleaved by id.
    Everything is made up."""

    households = len(clones) * 2
    ids = np.arange(1, households + 1, dtype="int64")
    household = pd.DataFrame({"household_id": ids})
    if column:
        index = np.array([clones[i % len(clones)] for i in range(households)])
        household[UK_SPINE_ENGINE_BATCH_COLUMN] = (
            pd.array(index, dtype="Int64") if missing else index
        )
        if missing:
            household.loc[0, UK_SPINE_ENGINE_BATCH_COLUMN] = pd.NA
    person = pd.DataFrame(
        {
            "person_id": ids[::-1] * 10,
            "person_benunit_id": ids[::-1],
            "person_household_id": ids[::-1],
            "age": np.full(households, 40.0),
        }
    )
    return Frame(
        tables={
            "person": person,
            "benunit": pd.DataFrame({"benunit_id": ids}),
            "household": household,
        },
        schema=EntitySchema(group_entities=("benunit", "household")),
        weights={
            "household": Weights(
                values=np.full(households, 100.0), kind=WeightKind.DESIGN
            )
        },
    )


def _no_engine(monkeypatch, engine):
    def refuse():
        raise AssertionError("no engine is built")

    monkeypatch.setattr(engine, "_import_policyengine_uk", refuse)


def _record(batches):
    return {
        "by": UK_SPINE_ENGINE_BATCH_COLUMN,
        "batches": batches,
        "pass": SPINE_BATCHED_PASS.name,
        "declaration_sha256": UK_ENGINE_POPULATION.sha256,
    }


def test_whether_a_call_batches_is_decided_from_the_frame(monkeypatch):
    engine = PolicyEngineUKEngine()
    _no_engine(monkeypatch, engine)
    assert engine.simulation_batches(_clone_frame(column=False)) == 1
    assert engine.simulation_batches(_clone_frame((0,))) == 1
    assert engine.simulation_batches(_clone_frame((0, 1, 2))) == 3
    # The record is the frame's, with no call run: names and the clone count.
    assert engine.batch_record(_clone_frame(column=False)) is None
    assert engine.batch_record(_clone_frame((0,))) is None
    assert engine.batch_record(_clone_frame((0, 1, 2))) == _record(3)


@pytest.mark.parametrize(
    ("expected", "clones", "batches"),
    [
        (3, (0, 1, 2), 3),
        (3, (0,), None),
        (3, (0, 1), None),
        (1, (0,), 1),
        (1, (0, 1), None),
    ],
)
def test_the_spines_clone_count_refuses_any_other_frame(
    monkeypatch, expected, clones, batches
):
    # Part of the pool (one clone, or fewer than the spine's) or another
    # spine's frame is refused before any engine is built; a frame without
    # the column (a stage before the clones exist) runs as it always has.
    engine = PolicyEngineUKEngine(clones=expected)
    assert engine.simulation_batches(_clone_frame(column=False)) == 1
    frame = _clone_frame(clones)
    if batches is not None:
        assert engine.simulation_batches(frame) == batches
        return
    with pytest.raises(ValueError, match="no engine call runs per clone"):
        engine.simulation_batches(frame)
    monkeypatch.setattr(
        engine,
        "_import_policyengine_uk",
        lambda: SimpleNamespace(Microsimulation=lambda **kwargs: pytest.fail()),
    )
    with pytest.raises(ValueError, match="no engine call runs per clone"):
        engine.materialize(frame, ["age"], 2025)


@pytest.mark.parametrize("clones", [0, -1, True, np.bool_(True), 2.0, "3"])
def test_the_spines_clone_count_is_a_positive_whole_number(clones):
    with pytest.raises(ValueError, match="positive whole number"):
        PolicyEngineUKEngine(clones=clones)


@pytest.mark.parametrize("clones", [3, np.int64(3), np.int32(3)])
def test_the_spines_clone_count_may_be_any_whole_number_type(monkeypatch, clones):
    engine = PolicyEngineUKEngine(clones=clones)
    _no_engine(monkeypatch, engine)
    assert engine.simulation_batches(_clone_frame((0, 1, 2))) == 3
    with pytest.raises(ValueError, match="no engine call runs per clone"):
        engine.simulation_batches(_clone_frame((0, 1)))


def test_a_clone_index_with_missing_households_is_refused(monkeypatch):
    engine = PolicyEngineUKEngine()
    _no_engine(monkeypatch, engine)
    with pytest.raises(ValueError, match="clone each belongs to"):
        engine.simulation_batches(_clone_frame((0, 1), missing=True))


def _one_simulation_engine(monkeypatch):
    """An adapter whose engine is a stub: one call builds one simulation."""

    built = []

    class Simulation:
        tax_benefit_system = SimpleNamespace(variables={})
        input_variables = ()

        def __init__(self, dataset):
            built.append(dataset)

        def calculate(self, name, period=None):
            return np.arange(len(built[-1]), dtype=np.float32)

    engine = PolicyEngineUKEngine()
    monkeypatch.setattr(
        engine,
        "_import_policyengine_uk",
        lambda: SimpleNamespace(Microsimulation=Simulation),
    )
    monkeypatch.setattr(
        engine, "_build_dataset", lambda bundle, period: list(range(bundle.n("person")))
    )
    monkeypatch.setattr(
        engine,
        "variable_metadata",
        lambda name: SimpleNamespace(entity="person"),
    )
    return engine, built


@pytest.mark.parametrize("clones", [None, (0,)])
def test_a_frame_with_one_clone_takes_one_simulation(monkeypatch, clones):
    engine, built = _one_simulation_engine(monkeypatch)

    def refuse(*args, **kwargs):
        raise AssertionError("one clone never runs in batches")

    monkeypatch.setattr(adapter_module, "materialize_in_batches", refuse)
    frame = _clone_frame(column=False) if clones is None else _clone_frame(clones)
    values = engine.materialize(frame, ["age"], 2025)
    assert len(built) == 1 and len(built[0]) == frame.n("person")
    np.testing.assert_array_equal(values["age"], np.arange(frame.n("person")))
    assert engine.batch_record(frame) is None


def test_a_frame_with_clones_runs_through_the_shared_batches(monkeypatch):
    engine, built = _one_simulation_engine(monkeypatch)
    calls = []

    def batched(bundle, variables, period, **kwargs):
        calls.append((bundle, variables, period, kwargs))
        return {"age": np.zeros(bundle.n("person"))}

    monkeypatch.setattr(adapter_module, "materialize_in_batches", batched)
    roles = []
    monkeypatch.setattr(
        adapter_module,
        "validate_uc_claimant_input",
        lambda simulation, person, period: roles.append((len(person), period)),
    )
    frame = _clone_frame((0, 1, 2))
    engine.materialize(frame, ["age"], 2025, dataset_period=2023)
    ((bundle, variables, period, kwargs),) = calls
    assert bundle is frame and list(variables) == ["age"] and period == 2025
    assert kwargs["declaration"] is UK_ENGINE_POPULATION
    assert kwargs["policy"] is SPINE_BATCHED_PASS
    assert kwargs["dataset_year"] == 2023
    assert [batch.key for batch in kwargs["batches"]] == [0, 1, 2]
    assert kwargs["entity_of"]("age") == "person"
    # Every batch is held to the adapter's own parameters wherever the UK
    # declaration's restatements read them.
    assert kwargs["reference_parameters"] == engine.parameters
    # Each batch's engine is built from the batch's own frame, at the
    # dataset's year, and its roles are checked against the batch's people.
    batch = frame.select(kwargs["batches"][1].persons)
    kwargs["build"](batch)
    assert built == [list(range(batch.n("person")))]
    assert roles == [(batch.n("person"), 2025)]
    engine.materialize(frame, ["age"], 2025)
    assert calls[-1][3]["dataset_year"] == 2025
    # The calls this adapter ran leave no trace in the record: it is the
    # frame's, as a fresh adapter reads it.
    assert engine.batch_record(frame) == _record(3)
    assert PolicyEngineUKEngine().batch_record(frame) == _record(3)
    with pytest.raises(ValueError, match="projects a dataset forward only"):
        engine.materialize(frame, ["age"], 2023, dataset_period=2025)


def test_the_parameters_are_the_engines_own():
    engine = PolicyEngineUKEngine()
    engine._system = SimpleNamespace(parameters=lambda period: ("tree at", period))
    assert engine.parameters(2025) == ("tree at", 2025)


# ---------------------------------------------------------------------------
# The ids each simulation holds (microcosm#931)
# ---------------------------------------------------------------------------


def _identifier_system(dtype=np.int32, **dtypes):
    """A stand-in tax-benefit system whose five id variables store as
    ``dtype``, or as ``dtypes[column]``."""

    return SimpleNamespace(
        variables={
            column: SimpleNamespace(dtype=dtypes.get(column, dtype))
            for column in identifier_columns(UK_SCHEMA)
        }
    )


def _ids(frame) -> list[int]:
    """Every id of every entity of ``frame``, as Python ints."""

    return [
        int(value)
        for entity in UK_SCHEMA.entities
        for value in frame.table(entity)[UK_SCHEMA.entity_id_column(entity)]
    ]


def _assert_count_free(message: str, numbers) -> None:
    """A refusal quotes none of the frame's multi-digit ids or their stored
    values (single digits are never checked: a type name can hold one). A
    failure names none of them either."""

    quoted = any(str(number) in message for number in numbers if abs(number) >= 10)
    assert not quoted, "the refusal quotes an id or a stored value"


def _snapshot(frame) -> dict[str, pd.DataFrame]:
    return {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}


def _assert_unchanged(frame, before) -> None:
    for entity, table in before.items():
        pd.testing.assert_frame_equal(frame.table(entity), table, check_exact=True)


def test_the_engine_declaration_is_int32_and_held_to_the_engine_once(monkeypatch):
    # policyengine-uk declares its five id variables ``value_type = int``,
    # which policyengine-core stores as int32.
    assert UK_ENGINE_IDENTIFIERS.stored_dtype == np.dtype("int32")
    assert dict(identifier_columns(UK_SCHEMA)) == {
        "person_id": "person",
        "benunit_id": "benunit",
        "person_benunit_id": "benunit",
        "household_id": "household",
        "person_household_id": "household",
    }
    # The adapter holds the declaration to its own engine once: a second
    # check would build an engine, which _no_engine refuses.
    engine = PolicyEngineUKEngine()
    _no_engine(monkeypatch, engine)
    engine._system = _identifier_system()
    engine._require_identifier_dtypes()
    engine._system = None
    engine._require_identifier_dtypes()
    # An engine that stores an id column otherwise is refused, every time.
    other = PolicyEngineUKEngine()
    _no_engine(monkeypatch, other)
    other._system = _identifier_system(person_household_id=np.int64)
    for _ in range(2):
        with pytest.raises(ValueError) as excinfo:
            other._require_identifier_dtypes()
        assert "'person_household_id' as int64, not as the declared int32" in str(
            excinfo.value
        )
    # The dataset builder holds the declaration before it builds anything.
    built = []
    data = types.ModuleType("policyengine_uk.data")
    data.UKSingleYearDataset = lambda **kwargs: built.append(kwargs)
    monkeypatch.setitem(sys.modules, "policyengine_uk.data", data)
    with pytest.raises(ValueError, match="declared int32"):
        other._build_dataset(clone_layout_frame(), 2025)
    assert built == []


def test_ids_that_stay_apart_once_stored_reach_the_engine_as_its_tables(monkeypatch):
    # The structural half of the proof that a frame without a stored repeat
    # reaches the engine unchanged: a frame whose ids pass 2**31 but stay
    # apart once stored, as K = 1's SPI copies do, reaches the engine as
    # exactly the tables it did before.
    frame = clone_layout_frame()
    before = _snapshot(frame)
    tables = engine_tables(frame, weighted_entities=("household",))
    # The layout's own claims.
    for entity in UK_SCHEMA.entities:
        ids = [int(value) for value in tables[entity][f"{entity}_id"]]
        assert min(ids) > 2**31
        assert len({wrapped(value, np.int32) for value in ids}) == len(ids)
    assert stored_identifier_repeats(UK_SCHEMA, tables, UK_ENGINE_IDENTIFIERS) == ()
    assert (
        inexact_identifier_entities(UK_SCHEMA, tables, UK_ENGINE_IDENTIFIERS)
        == UK_SCHEMA.entities
    )
    engine = PolicyEngineUKEngine()
    _no_engine(monkeypatch, engine)
    made = []

    def recording(bundle, **kwargs):
        made.append(engine_tables(bundle, **kwargs))
        return made[-1]

    monkeypatch.setattr(adapter_module, "engine_tables", recording)

    given = engine._dataset_tables(frame)

    # The very objects engine_tables made, not equal copies.
    (returned,) = made
    assert list(given) == list(returned)
    assert all(given[entity] is returned[entity] for entity in returned)
    assert list(given) == list(tables)
    for entity, table in tables.items():
        pd.testing.assert_frame_equal(given[entity], table, check_exact=True)
    # And the dataset is built from those tables, after the declaration is
    # checked, with no engine built.
    built = []
    data = types.ModuleType("policyengine_uk.data")
    data.UKSingleYearDataset = lambda **kwargs: built.append(kwargs) or "dataset"
    monkeypatch.setitem(sys.modules, "policyengine_uk.data", data)
    engine._system = _identifier_system()
    assert engine._build_dataset(frame, 2025) == "dataset"
    (kwargs,) = built
    assert kwargs["fiscal_year"] == 2025
    for entity, table in tables.items():
        pd.testing.assert_frame_equal(kwargs[entity], table, check_exact=True)
    _assert_unchanged(frame, before)


def test_each_batch_ranks_its_own_rows_where_its_household_ids_store_together(
    monkeypatch,
):
    # Each clone holds two households whose ids store as one int32 value;
    # each batch numbers its own households 1 to n in id order.
    frame = colliding_clones_frame()
    before = _snapshot(frame)
    engine = PolicyEngineUKEngine()
    _no_engine(monkeypatch, engine)
    batches = engine._clone_batches(frame)
    assert [batch.key for batch in batches] == list(COLLIDING_CLONE_HOUSEHOLD_IDS)
    for batch, households in zip(
        batches, COLLIDING_CLONE_HOUSEHOLD_IDS.values(), strict=True
    ):
        part = frame.select(batch.persons)
        tables = engine_tables(part, weighted_entities=("household",))
        # The layout's own claims: the batch's households store together,
        # and nothing else repeats once stored.
        assert part.table("household")["household_id"].tolist() == list(households)
        assert len({wrapped(value, np.int32) for value in households}) == 1
        assert stored_identifier_repeats(UK_SCHEMA, tables, UK_ENGINE_IDENTIFIERS) == (
            "household",
        )

        given = engine._dataset_tables(part)

        # Exactly the batch renumbered by hand: households 1 and 2, each
        # membership its household's new id.
        engine_id = {household: rank + 1 for rank, household in enumerate(households)}
        hand = engine_tables(
            relabelled(part, "household", list(engine_id.values())),
            weighted_entities=("household",),
        )
        assert list(given) == list(hand)
        for entity in UK_SCHEMA.entities:
            pd.testing.assert_frame_equal(given[entity], hand[entity], check_exact=True)
        assert given["household"]["household_id"].tolist() == [1, 2]
        assert given["person"]["person_household_id"].tolist() == [
            engine_id[int(value)] for value in tables["person"]["person_household_id"]
        ]
        # Persons and benefit units pass through; nothing but the two
        # household id columns moves.
        pd.testing.assert_frame_equal(
            given["benunit"], tables["benunit"], check_exact=True
        )
        for entity, column in (
            ("person", "person_household_id"),
            ("household", "household_id"),
        ):
            pd.testing.assert_frame_equal(
                given[entity].drop(columns=column),
                tables[entity].drop(columns=column),
                check_exact=True,
            )
    _assert_unchanged(frame, before)


def test_each_batch_dataset_is_ranked_over_its_own_rows(monkeypatch):
    # As above, through materialize's batched site and _build_dataset: the
    # batches are handed the frame itself, and each batch's dataset numbers
    # that batch's households 1 and 2, people and benefit units passing
    # through.
    frame = colliding_clones_frame()
    before = _snapshot(frame)
    built = []
    data = types.ModuleType("policyengine_uk.data")
    data.UKSingleYearDataset = lambda **kwargs: built.append(kwargs) or "dataset"
    monkeypatch.setitem(sys.modules, "policyengine_uk.data", data)
    engine = PolicyEngineUKEngine()
    engine._system = _identifier_system()
    monkeypatch.setattr(
        engine,
        "_import_policyengine_uk",
        lambda: SimpleNamespace(Microsimulation=lambda dataset: SimpleNamespace()),
    )
    handed = []

    def batched(bundle, variables, period, **kwargs):
        handed.append(bundle)
        for batch in kwargs["batches"]:
            kwargs["build"](bundle.select(batch.persons))
        return {}

    monkeypatch.setattr(adapter_module, "materialize_in_batches", batched)
    engine.materialize(frame, ["age"], 2025)
    assert len(handed) == 1 and handed[0] is frame
    batches = engine._clone_batches(frame)
    assert len(built) == len(batches) == len(COLLIDING_CLONE_HOUSEHOLD_IDS)
    for batch, kwargs in zip(batches, built, strict=True):
        part = engine_tables(
            frame.select(batch.persons), weighted_entities=("household",)
        )
        engine_id = {
            int(household): rank + 1
            for rank, household in enumerate(part["household"]["household_id"])
        }
        assert kwargs["fiscal_year"] == 2025
        assert kwargs["household"]["household_id"].tolist() == [1, 2]
        assert kwargs["person"]["person_household_id"].tolist() == [
            engine_id[int(value)] for value in part["person"]["person_household_id"]
        ]
        for entity in ("person", "benunit"):
            column = f"{entity}_id"
            assert kwargs[entity][column].tolist() == part[entity][column].tolist()
    _assert_unchanged(frame, before)


@pytest.mark.parametrize("build", [clone_layout_frame, colliding_clones_frame])
@pytest.mark.parametrize(
    "variables",
    [["household_id"], ["age", "person_household_id"], ["benunit_id"]],
)
def test_an_id_request_the_engine_cannot_answer_is_refused_before_any_engine(
    monkeypatch, build, variables
):
    # The engine would hold wrapped ids, or ranks for the colliding clones'
    # households, for these entities, not the frame's.
    frame = build()
    engine = PolicyEngineUKEngine()
    _no_engine(monkeypatch, engine)
    with pytest.raises(ValueError, match="cannot return") as excinfo:
        engine.materialize(frame, variables, 2025)
    message = str(excinfo.value)
    for name in variables:
        if name != "age":
            assert repr(name) in message
    ids = _ids(frame)
    _assert_count_free(message, ids + [wrapped(value, np.int32) for value in ids])


def test_an_id_request_the_engine_answers_reaches_it_with_every_name(monkeypatch):
    # The person ids fit int32, though the households' and benefit units' do
    # not: a person id request reaches the engine, and so does every name of
    # a one-shot iterator, on either path.
    engine, built = _one_simulation_engine(monkeypatch)
    frame = clone_layout_frame(shifted=("benunit", "household"))
    values = engine.materialize(frame, ["person_id"], 2025)
    assert len(built) == 1 and list(values) == ["person_id"]
    values = engine.materialize(frame, iter(["age", "person_id"]), 2025)
    assert len(built) == 2 and list(values) == ["age", "person_id"]
    calls = []
    monkeypatch.setattr(
        adapter_module,
        "materialize_in_batches",
        lambda bundle, variables, period, **kwargs: calls.append(list(variables)),
    )
    engine.materialize(_clone_frame((0, 1, 2)), iter(["age", "person_id"]), 2025)
    assert calls == [["age", "person_id"]]


def _construction_engine(monkeypatch, module):
    """An adapter whose engine module is ``module`` and whose datasets are
    stubs; batched calls hand the first batch to its build closure."""

    engine = PolicyEngineUKEngine()
    monkeypatch.setattr(engine, "_import_policyengine_uk", lambda: module)
    monkeypatch.setattr(
        engine, "_build_dataset", lambda bundle, period: list(range(bundle.n("person")))
    )
    monkeypatch.setattr(
        engine, "variable_metadata", lambda name: SimpleNamespace(entity="person")
    )

    def batched(bundle, variables, period, **kwargs):
        kwargs["build"](bundle.select(kwargs["batches"][0].persons))
        return {}

    monkeypatch.setattr(adapter_module, "materialize_in_batches", batched)
    return engine


def _simulation_class(seen, emit=()):
    """A stand-in Microsimulation recording the warning filters it was
    constructed under, and emitting a made-up warning of each ``emit``
    category while it is constructed."""

    class Simulation:
        tax_benefit_system = SimpleNamespace(variables={})
        input_variables = ()

        def __init__(self, dataset):
            seen.append(list(warnings.filters))
            for category in emit:
                warnings.warn("made-up construction warning", category, stacklevel=2)
            self._rows = len(dataset)

        def calculate(self, name, period=None):
            return np.zeros(self._rows)

    return Simulation


def _module_without_the_warning(shape):
    module = types.ModuleType("policyengine_uk")
    if shape in {"no_entity_ids", "not_a_warning"}:
        module.utils = types.ModuleType("policyengine_uk.utils")
    if shape == "not_a_warning":
        module.utils.entity_ids = types.ModuleType("policyengine_uk.utils.entity_ids")
        module.utils.entity_ids.EntityIdWarning = object
    return module


@pytest.mark.parametrize("shape", ["no_utils", "no_entity_ids", "not_a_warning"])
@pytest.mark.parametrize("clones", [None, (0, 1, 2)])
def test_an_engine_module_without_the_id_warning_class_builds_nothing(
    monkeypatch, shape, clones
):
    # The filter fails closed at both construction sites, before any
    # simulation is constructed.
    seen = []
    module = _module_without_the_warning(shape)
    module.Microsimulation = _simulation_class(seen)
    engine = _construction_engine(monkeypatch, module)
    frame = _clone_frame(column=False) if clones is None else _clone_frame(clones)
    with pytest.raises(ImportError) as excinfo:
        engine.materialize(frame, ["age"], 2025)
    assert "policyengine_uk.utils.entity_ids.EntityIdWarning" in str(excinfo.value)
    assert seen == []


@pytest.mark.parametrize("clones", [None, (0, 1, 2)])
def test_a_stand_in_engine_constructs_without_a_filter(monkeypatch, clones):
    # A stand-in that is not a module and carries no warning class, as the
    # stubs above are, constructs under the filters it was called with.
    seen = []
    stand_in = SimpleNamespace(Microsimulation=_simulation_class(seen))
    engine = _construction_engine(monkeypatch, stand_in)
    frame = _clone_frame(column=False) if clones is None else _clone_frame(clones)
    outer = list(warnings.filters)
    engine.materialize(frame, ["age"], 2025)
    assert seen == [outer]


@pytest.mark.parametrize("clones", [None, (0, 1, 2)])
def test_the_id_warning_is_filtered_only_while_a_simulation_is_built(
    monkeypatch, clones
):
    # The warning class is read from the engine module the adapter imported;
    # other warnings pass, and the filter ends with the construction.
    class EntityIdWarning(UserWarning):
        pass

    seen = []
    module = types.ModuleType("policyengine_uk")
    module.utils = SimpleNamespace(
        entity_ids=SimpleNamespace(EntityIdWarning=EntityIdWarning)
    )
    module.Microsimulation = _simulation_class(
        seen, emit=(EntityIdWarning, RuntimeWarning)
    )
    engine = _construction_engine(monkeypatch, module)
    frame = _clone_frame(column=False) if clones is None else _clone_frame(clones)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.materialize(frame, ["age"], 2025)
        warnings.warn("made-up later warning", EntityIdWarning, stacklevel=1)
    assert len(seen) == 1
    assert [
        (item.category, str(item.message))
        for item in caught
        if str(item.message).startswith("made-up")
    ] == [
        (RuntimeWarning, "made-up construction warning"),
        (EntityIdWarning, "made-up later warning"),
    ]
