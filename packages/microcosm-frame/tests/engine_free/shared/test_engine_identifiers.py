"""The ids each engine simulation holds for a frame (microcosm#931).

``microcosm.frame.adapters.engine_identifiers`` hands a simulation the tables it
was given where no entity's ids repeat once the engine stores them, and gives
an entity whose stored ids would repeat 1 plus each id's rank, each membership
taking its group's new id. Held here on neutral schemas: ids that stay apart
pass through as the same objects, a repeat renumbers only its entity, a seeded
sweep over ids of every NumPy and nullable integer dtype keeps order and
memberships, an id request the engine cannot answer is refused without reading
a table it need not, and every refusal names entities, columns and types, never
an id or a count. Every id here is made up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import EntitySchema, Frame, LinkSpec, WeightKind, Weights
from microcosm.frame.adapters import engine_identifiers
from microcosm.frame.adapters.engine_identifiers import (
    EngineIdentifiers,
    EngineIdentifierTables,
    engine_identifier_tables,
    identifier_columns,
    inexact_identifier_entities,
    rank_identifier_tables,
    refuse_inexact_identifier_requests,
    require_stored_dtypes,
    stored_identifier_repeats,
)
from test_support.microcosm_frame.engine_identifier_frames import (
    COLLIDING_HOUSEHOLD_IDS,
    FAMILY_SCHEMA,
    NEUTRAL_SCHEMAS,
    SWEEP_STORED_DTYPES,
    WRAP,
    colliding_household_tables,
    layout_tables,
    sweep_frame,
    sweep_tables,
    wrapped,
)

INT32 = EngineIdentifiers(np.int32)
#: Sweep cases per schema and stored type.
SWEEP_CASES = 100


def _schema_id(schema) -> str:
    return "-".join(schema.group_entities)


def _copies(tables):
    return {entity: table.copy(deep=True) for entity, table in tables.items()}


def _assert_unchanged(tables, before) -> None:
    assert list(tables) == list(before)
    for entity, table in tables.items():
        pd.testing.assert_frame_equal(table, before[entity], check_exact=True)


def _id_values(schema, tables) -> dict[str, list[int]]:
    """Every id and membership column's values, as Python ints in row order."""

    person = tables[schema.person_entity]
    values = {}
    for entity in schema.entities:
        column = schema.entity_id_column(entity)
        values[column] = [int(value) for value in tables[entity][column]]
    for group in schema.group_entities:
        column = schema.membership_column(group)
        values[column] = [int(value) for value in person[column]]
    return values


def _order(values: list[int]) -> list[int]:
    return sorted(range(len(values)), key=values.__getitem__)


def _assert_count_free(message: str, *numbers: int | float) -> None:
    """A refusal quotes none of the frame's multi-digit ids, wrapped values or
    row counts (single digits are never checked: a type name can hold one).
    A failure names none of them either."""

    quoted = any(str(number) in message for number in numbers)
    assert not quoted, "the refusal quotes an id, a wrapped value or a row count"


# ---------------------------------------------------------------------------
# Test 1: ids that stay apart once stored pass through.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["inside", "copies"])
@pytest.mark.parametrize("schema", NEUTRAL_SCHEMAS, ids=_schema_id)
def test_ids_that_stay_apart_once_stored_pass_through(schema, layout):
    tables = layout_tables(schema, layout)
    before = _copies(tables)
    stored = {
        column: [wrapped(value, np.int32) for value in values]
        for column, values in _id_values(schema, tables).items()
    }
    # The layout's own claims: every entity's ids stay apart once stored; the
    # copies pass 2**31 for every entity, and some wrap negative.
    for entity in schema.entities:
        column = schema.entity_id_column(entity)
        assert len(set(stored[column])) == len(stored[column])
        if layout == "copies":
            assert max(_id_values(schema, tables)[column]) > 2**31
            assert min(stored[column]) < 0
    assert stored_identifier_repeats(schema, tables, INT32) == ()

    result = engine_identifier_tables(schema, tables, INT32)

    assert isinstance(result, EngineIdentifierTables)
    assert result.renumbered == ()
    assert list(result.tables) == list(tables)
    for entity, table in tables.items():
        assert result.tables[entity] is table
    _assert_unchanged(tables, before)
    # Passing through does not mean the engine holds them exactly.
    assert inexact_identifier_entities(schema, tables, INT32) == (
        () if layout == "inside" else schema.entities
    )


# ---------------------------------------------------------------------------
# Test 2: a stored repeat renumbers its entity by rank.
# ---------------------------------------------------------------------------


def test_a_stored_repeat_renumbers_that_entity_by_rank():
    tables = colliding_household_tables()
    before = _copies(tables)
    first, second, third = COLLIDING_HOUSEHOLD_IDS
    assert wrapped(first, np.int32) == wrapped(third, np.int32)
    assert stored_identifier_repeats(FAMILY_SCHEMA, tables, INT32) == ("household",)

    result = engine_identifier_tables(FAMILY_SCHEMA, tables, INT32)

    assert result.renumbered == ("household",)
    household = result.tables["household"]
    assert household["household_id"].tolist() == [1, 2, 3]
    assert household["household_id"].dtype == np.dtype(np.int64)
    assert household["rent"].tolist() == tables["household"]["rent"].tolist()
    # Each person's membership is their household's engine id, 1 plus its
    # rank, never the 0-based rank.
    engine_id = {first: 1, second: 2, third: 3}
    person = result.tables["person"]
    frame_memberships = tables["person"]["person_household_id"].tolist()
    assert person["person_household_id"].tolist() == [
        engine_id[value] for value in frame_memberships
    ]
    assert person["person_household_id"].tolist() == [3, 1, 2, 1, 3, 2]
    assert person["person_household_id"].dtype == np.dtype(np.int64)
    # People keep their order, ids and other memberships.
    for column in ("person_id", "person_family_id", "age"):
        pd.testing.assert_series_equal(person[column], tables["person"][column])
    assert result.tables["family"] is tables["family"]
    assert household is not tables["household"]
    assert person is not tables["person"]
    assert list(result.tables) == list(tables)
    assert stored_identifier_repeats(FAMILY_SCHEMA, result.tables, INT32) == ()
    _assert_unchanged(tables, before)


# ---------------------------------------------------------------------------
# Test 3: a seeded sweep over ids of every NumPy and nullable integer dtype.
# ---------------------------------------------------------------------------


def _assert_engine_ids(frame, given, returned, *, renumbered, stored_dtype) -> None:
    """``returned`` holds ``frame``'s ids with exactly ``renumbered`` ranked.

    An engine id is the value in the returned table, not its stored form: an
    entity that passes through outside the type stores in another order.
    """

    schema = frame.schema
    engine = _id_values(schema, returned)
    assert list(returned) == list(given)
    for entity in schema.entities:
        column = schema.entity_id_column(entity)
        frame_ids, engine_ids = frame.ids[entity], engine[column]
        # Sorting rows by engine id is sorting them by frame id.
        assert _order(engine_ids) == _order(frame_ids)
        if entity in renumbered:
            assert sorted(engine_ids) == list(range(1, len(frame_ids) + 1))
            assert returned[entity][column].dtype == np.dtype(np.int64)
        else:
            assert engine_ids == frame_ids
            assert returned[entity][column].dtype == given[entity][column].dtype
        if stored_dtype is not None:
            stored = [wrapped(value, stored_dtype) for value in engine_ids]
            assert len(set(stored)) == len(stored), entity
        value_column = f"{entity}_value"
        pd.testing.assert_series_equal(
            returned[entity][value_column], given[entity][value_column]
        )
    person = schema.person_entity
    for group in schema.group_entities:
        column = schema.membership_column(group)
        # Each engine membership is the engine id of the group the frame's
        # membership names; a 0-based map fails this.
        engine_of = dict(
            zip(frame.ids[group], engine[schema.id_column(group)], strict=True)
        )
        assert engine[column] == [
            engine_of[value] for value in frame.memberships[group]
        ]
        expected = (
            np.dtype(np.int64) if group in renumbered else given[person][column].dtype
        )
        assert returned[person][column].dtype == expected
        assert (returned[group] is given[group]) == (group not in renumbered)
    assert (returned[person] is given[person]) == (not renumbered)


@pytest.mark.parametrize("stored_dtype", SWEEP_STORED_DTYPES, ids=str)
@pytest.mark.parametrize("schema", NEUTRAL_SCHEMAS, ids=_schema_id)
def test_a_seeded_sweep_ranks_exactly_the_entities_whose_stored_ids_repeat(
    schema, stored_dtype
):
    identifiers = EngineIdentifiers(stored_dtype)
    rng = np.random.default_rng([931, len(schema.entities), stored_dtype.itemsize])
    decisions = set()
    for _case in range(SWEEP_CASES):
        frame = sweep_frame(rng, schema)
        tables = sweep_tables(frame, rng)
        before = _copies(tables)
        expected = tuple(
            entity
            for entity in schema.entities
            if len({wrapped(value, stored_dtype) for value in frame.ids[entity]})
            < len(frame.ids[entity])
        )
        inexact = tuple(
            entity
            for entity in schema.entities
            if any(wrapped(value, stored_dtype) != value for value in frame.ids[entity])
        )
        assert stored_identifier_repeats(schema, tables, identifiers) == expected
        assert inexact_identifier_entities(schema, tables, identifiers) == inexact
        assert set(expected) <= set(inexact)

        result = engine_identifier_tables(schema, tables, identifiers)

        # An entity renumbers exactly when its stored ids repeat, and after,
        # no entity's do.
        assert result.renumbered == expected
        _assert_engine_ids(
            frame, tables, result.tables, renumbered=expected, stored_dtype=stored_dtype
        )
        # The same ids in other drawn dtypes: the same decision and ranks.
        retyped = sweep_tables(frame, rng)
        again = engine_identifier_tables(schema, retyped, identifiers)
        assert again.renumbered == expected
        assert _id_values(schema, again.tables) == _id_values(schema, result.tables)
        # The rank step ranks exactly the entities it is given.
        chosen = tuple(entity for entity in schema.entities if rng.random() < 0.5)
        ranked = rank_identifier_tables(schema, tables, chosen)
        _assert_engine_ids(frame, tables, ranked, renumbered=chosen, stored_dtype=None)
        assert _id_values(schema, rank_identifier_tables(schema, retyped, chosen)) == (
            _id_values(schema, ranked)
        )
        _assert_unchanged(tables, before)
        decisions.add(bool(expected))
    # The sweep reaches both branches.
    assert decisions == {False, True}


# ---------------------------------------------------------------------------
# Test 4: inexact entities, and id requests the engine cannot answer.
# ---------------------------------------------------------------------------


def _inexact_tables():
    """People fit int32; household 9 + 2**32 lies outside it; family 1.5 is
    not a whole number. Neither entity's stored ids repeat."""

    outside = 9 + WRAP
    return {
        "person": pd.DataFrame(
            {
                "person_id": np.asarray([21, 22, 23], dtype=np.int64),
                "person_family_id": np.asarray([1.5, 3.0, 3.0]),
                "person_household_id": np.asarray([7, outside, 7], dtype=np.int64),
            }
        ),
        "family": pd.DataFrame({"family_id": np.asarray([1.5, 3.0])}),
        "household": pd.DataFrame(
            {"household_id": np.asarray([7, outside], dtype=np.int64)}
        ),
    }


def _reader(tables, reads):
    def table(entity):
        reads.append(entity)
        return tables[entity]

    return table


def test_inexact_entities_and_the_id_requests_the_engine_cannot_answer():
    assert dict(identifier_columns(FAMILY_SCHEMA)) == {
        "person_id": "person",
        "family_id": "family",
        "person_family_id": "family",
        "household_id": "household",
        "person_household_id": "household",
    }
    assert list(identifier_columns(FAMILY_SCHEMA)) == [
        "person_id",
        "family_id",
        "person_family_id",
        "household_id",
        "person_household_id",
    ]
    # A group whose id column is another group's membership column.
    clashing = EntitySchema(group_entities=("household", "person_household"))
    with pytest.raises(ValueError, match="'person_household_id' twice"):
        identifier_columns(clashing)
    tables = _inexact_tables()
    assert stored_identifier_repeats(FAMILY_SCHEMA, tables, INT32) == ()
    assert inexact_identifier_entities(FAMILY_SCHEMA, tables, INT32) == (
        "family",
        "household",
    )
    # An entity whose stored ids repeat is inexact too.
    assert inexact_identifier_entities(
        FAMILY_SCHEMA, colliding_household_tables(), INT32
    ) == ("household",)

    refused = {
        "household_id": "household",
        "person_household_id": "household",
        "family_id": "family",
        "person_family_id": "family",
    }
    for name, entity in refused.items():
        reads = []
        with pytest.raises(ValueError, match="cannot return") as excinfo:
            refuse_inexact_identifier_requests(
                FAMILY_SCHEMA,
                iter([name, "age"]),
                INT32,
                table=_reader(tables, reads),
            )
        message = str(excinfo.value)
        assert repr(name) in message and repr(entity) in message
        assert "int32" in message
        _assert_count_free(message, 9 + WRAP, 1.5, 21, 22, 23)
        assert reads == [entity]

    # The persons fit, and a variable that is no id needs no table at all.
    reads = []
    refuse_inexact_identifier_requests(
        FAMILY_SCHEMA, ["person_id", "age"], INT32, table=_reader(tables, reads)
    )
    assert reads == ["person"]

    def refuse_to_read(entity):
        raise AssertionError("no table is read for a request with no id")

    refuse_inexact_identifier_requests(
        FAMILY_SCHEMA, ["age", "income"], INT32, table=refuse_to_read
    )
    refuse_inexact_identifier_requests(FAMILY_SCHEMA, [], INT32, table=refuse_to_read)

    # Each entity is read once, and only the inexact entities' columns are
    # named.
    reads = []
    with pytest.raises(ValueError) as excinfo:
        refuse_inexact_identifier_requests(
            FAMILY_SCHEMA,
            ["person_id", "household_id", "person_household_id", "household_id"],
            INT32,
            table=_reader(tables, reads),
        )
    assert reads == ["person", "household"]
    message = str(excinfo.value)
    assert "['household_id', 'person_household_id']" in message
    assert "'person_id'" not in message
    with pytest.raises(TypeError, match="not one name"):
        refuse_inexact_identifier_requests(
            FAMILY_SCHEMA, "household_id", INT32, table=refuse_to_read
        )


# ---------------------------------------------------------------------------
# Test 5: refusals name entities, columns and types, never an id or a count.
# ---------------------------------------------------------------------------


def test_an_entity_with_more_rows_than_the_type_numbers_is_refused():
    # 200 people (more than int8 numbers from 1), two of whose ids store as one
    # value in int8: 1000 and 1256 both store as -24.
    person_ids = [1000 + index for index in range(199)] + [1256]
    assert wrapped(1000, np.int8) == wrapped(1256, np.int8) == -24
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.asarray(person_ids, dtype=np.int64),
                "person_family_id": np.ones(200, dtype=np.int64),
                "person_household_id": np.asarray([1, 2] * 100, dtype=np.int64),
            }
        ),
        "family": pd.DataFrame({"family_id": np.asarray([1], dtype=np.int64)}),
        "household": pd.DataFrame({"household_id": np.asarray([1, 2], dtype=np.int64)}),
    }
    identifiers = EngineIdentifiers(np.int8)
    assert stored_identifier_repeats(FAMILY_SCHEMA, tables, identifiers) == ("person",)
    with pytest.raises(ValueError, match="more rows than int8") as excinfo:
        engine_identifier_tables(FAMILY_SCHEMA, tables, identifiers)
    message = str(excinfo.value)
    assert "'person'" in message
    _assert_count_free(message, 200, 199, 1000, 1198, 1256, -24, 232, 127, 128)


@pytest.mark.parametrize(
    "stored_dtype",
    [np.float64, "float32", np.bool_, pd.Int32Dtype(), object, "no such type"],
    ids=str,
)
def test_a_declaration_that_is_no_integer_type_is_refused(stored_dtype):
    with pytest.raises(TypeError, match="must be a NumPy integer type"):
        EngineIdentifiers(stored_dtype)


def test_a_declaration_is_normalised_and_held_to_the_engine():
    assert EngineIdentifiers("int32").stored_dtype == np.dtype(np.int32)
    assert EngineIdentifiers(np.int32) == EngineIdentifiers(np.dtype("int32"))
    assert EngineIdentifiers(np.uint16).stored_dtype == np.dtype(np.uint16)
    require_stored_dtypes(
        INT32,
        {"person_id": np.int32, "family_id": np.dtype("int32"), "household_id": "i4"},
    )
    require_stored_dtypes(INT32, {})
    with pytest.raises(ValueError, match="declared int32") as excinfo:
        require_stored_dtypes(
            INT32,
            {
                "person_id": np.int32,
                "household_id": np.int64,
                "person_household_id": pd.Int32Dtype(),
            },
        )
    message = str(excinfo.value)
    assert "'household_id' as int64" in message
    assert "'person_household_id' as Int32Dtype()" in message
    assert "'person_id'" not in message
    with pytest.raises(TypeError, match="EngineIdentifiers declaration"):
        require_stored_dtypes(np.dtype(np.int32), {"person_id": np.int32})


def test_engine_ids_that_still_repeat_after_ranking_raise(monkeypatch):
    # Stands for a later edit that broke the rank step: it now returns the
    # tables as given.
    monkeypatch.setattr(
        engine_identifiers,
        "rank_identifier_tables",
        lambda schema, tables, entities: dict(tables),
    )
    with pytest.raises(RuntimeError, match="still repeat") as excinfo:
        engine_identifier_tables(FAMILY_SCHEMA, colliding_household_tables(), INT32)
    message = str(excinfo.value)
    assert "'household'" in message and "int32" in message
    _assert_count_free(message, 7 + WRAP)


def test_the_rank_step_refuses_what_it_cannot_rank():
    tables = colliding_household_tables()
    with pytest.raises(ValueError, match="undeclared entities") as excinfo:
        rank_identifier_tables(FAMILY_SCHEMA, tables, ["household", "tax_unit"])
    assert "['tax_unit']" in str(excinfo.value)
    with pytest.raises(TypeError, match="not one"):
        rank_identifier_tables(FAMILY_SCHEMA, tables, "household")
    # A membership that names no household (a Frame rules it out).
    stray = 11 + WRAP
    person = tables["person"].copy()
    person.loc[0, "person_household_id"] = stray
    with pytest.raises(ValueError, match="does not hold") as excinfo:
        rank_identifier_tables(
            FAMILY_SCHEMA, {**tables, "person": person}, ["household"]
        )
    _assert_count_free(str(excinfo.value), stray, 7 + WRAP)
    # Ids the tables themselves repeat have no rank (a Frame rules it out too).
    household = pd.DataFrame(
        {"household_id": np.asarray([9, 9 + WRAP, 9 + WRAP], dtype=np.int64)}
    )
    with pytest.raises(ValueError, match="repeat, so have no rank") as excinfo:
        rank_identifier_tables(
            FAMILY_SCHEMA, {**tables, "household": household}, ["household"]
        )
    _assert_count_free(str(excinfo.value), 9 + WRAP)
    # A declared link's table that names an entity to renumber: the step maps
    # only the person table's memberships, so it refuses rather than hand
    # back a link table holding the old ids.
    linked = EntitySchema(
        group_entities=("family", "household"),
        links=(LinkSpec("visits", "person", "household"),),
    )
    visits = pd.DataFrame(
        {
            "person_id": np.asarray([11, 13], dtype=np.int64),
            "household_id": np.asarray([7, 7 + WRAP], dtype=np.int64),
        }
    )
    with_link = {**tables, "visits": visits}
    with pytest.raises(ValueError, match="link tables") as excinfo:
        rank_identifier_tables(linked, with_link, ["household"])
    message = str(excinfo.value)
    assert "['household']" in message and "['visits']" in message
    _assert_count_free(message, 7 + WRAP, 11, 13)
    with pytest.raises(ValueError, match="link tables"):
        engine_identifier_tables(linked, with_link, INT32)
    # A link that names no entity to renumber, or whose table is not among
    # the tables, is left as it is.
    assert rank_identifier_tables(linked, with_link, ["family"])["visits"] is visits
    assert engine_identifier_tables(linked, tables, INT32).renumbered == ("household",)
    # Nothing to rank: the same objects back.
    unranked = rank_identifier_tables(FAMILY_SCHEMA, tables, [])
    assert all(unranked[entity] is table for entity, table in tables.items())


def test_the_functions_take_tables_never_a_frame():
    frame = Frame(
        colliding_household_tables(),
        FAMILY_SCHEMA,
        {"household": Weights(values=np.ones(3), kind=WeightKind.DESIGN)},
    )
    with pytest.raises(TypeError, match="map each entity to its table"):
        stored_identifier_repeats(FAMILY_SCHEMA, frame, INT32)
    with pytest.raises(ValueError, match="no 'household' table"):
        engine_identifier_tables(
            FAMILY_SCHEMA,
            {
                entity: table
                for entity, table in colliding_household_tables().items()
                if entity != "household"
            },
            INT32,
        )
