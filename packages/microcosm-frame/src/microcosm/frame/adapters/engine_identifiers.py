"""The ids each engine simulation holds for a frame's entities (microcosm#931).

policyengine-core keeps a dataset's ids in two places. Where a country package
builds a simulation from the tables an adapter hands it, as the package this
rule was first written for does, the entities, and each person's membership of
them, are built from the id and membership columns as given. Core also stores
those columns as the values of the id variables, each converted by numpy's
``astype`` to its variable's type, which wraps an id outside the type without
an error. Two ids that wrap to one stored value therefore merge no entity, but
they become one key for every formula that reads the stored id (a draw hashed
on it, a tie broken on it), and one entity for anything rebuilt from the
stored values. An adapter that adopts this rule first checks how its engine
builds its entities.

An adapter declares the type with :class:`EngineIdentifiers`, and
:func:`require_stored_dtypes` holds the declaration to the installed engine.
The rule, :func:`engine_identifier_tables`, takes each simulation's own tables:

- an entity whose ids stay apart once stored keeps them, so a simulation in
  which no entity's stored ids would repeat receives the tables it was given,
  and its input is unchanged byte for byte;
- an entity whose stored ids would repeat takes 1 plus each id's rank among
  the entity's ids in that simulation, in id order, and each person's
  membership of it takes the new id of the group it names (1 plus the group's
  rank, never the 0-based rank). Ranks are distinct, fit the type and keep id
  order, so the engine holds every entity's rows in the frame's order.

The frame is never changed, and no result is ever mapped back by the engine's
ids: an adapter returns values by position or by the frame's own ids. The
engine holds wrapped ids or ranks for an entity whose ids it cannot store
exactly, so :func:`refuse_inexact_identifier_requests` refuses a request for
that entity's id or membership variable rather than return them.

:func:`rank_identifier_tables` is the step that ranks and maps, and it decides
nothing. A release rule that renumbers every entity of a released file takes
the same step with its own ranker; moving such a rule onto this function is
owed.

Every function here that reads ids takes a schema and entity tables (a mapping
from entity to table, or a reader for one), never a Frame, so a caller holding
only a table dict can use it. Refusals name entities, links, columns,
variables and types, never an id or a count.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.frame.schema import EntitySchema

__all__ = [
    "EngineIdentifierTables",
    "EngineIdentifiers",
    "engine_identifier_tables",
    "identifier_columns",
    "inexact_identifier_entities",
    "rank_identifier_tables",
    "refuse_inexact_identifier_requests",
    "require_stored_dtypes",
    "stored_identifier_repeats",
]

_INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True)
class EngineIdentifiers:
    """How an engine stores a schema's id and membership columns.

    The engine converts every id and membership column to ``stored_dtype`` with
    numpy's ``astype``, as policyengine-core converts a stored input: an
    integer outside the type wraps and a float inside it is truncated, without
    an error; a float outside the type converts as the machine's numpy
    converts it (C leaves that cast undefined), so whether such ids repeat once
    stored is the machine's, as it is in the engine.

    Attributes:
        stored_dtype: The integer type the columns are stored as, normalised by
            ``np.dtype``.

    Raises:
        TypeError: If ``stored_dtype`` is not a NumPy integer type.
    """

    stored_dtype: np.dtype

    def __post_init__(self) -> None:
        try:
            dtype = np.dtype(self.stored_dtype)
        except TypeError:
            dtype = None
        if dtype is None or not np.issubdtype(dtype, np.integer):
            raise TypeError(
                "EngineIdentifiers.stored_dtype must be a NumPy integer type, "
                f"got {self.stored_dtype!r}."
            )
        object.__setattr__(self, "stored_dtype", dtype)


@dataclass(frozen=True)
class EngineIdentifierTables:
    """The tables one engine simulation receives, and the entities they renumber.

    Attributes:
        tables: Entity -> the table the simulation receives, in the given
            tables' order. Where nothing is renumbered these are the given
            tables themselves. Otherwise each renumbered entity's table, and
            the person table that holds the memberships, are copies, and every
            other table is the given object.
        renumbered: The entities whose ids are 1 plus their rank, in schema
            order; empty where every entity passes its ids through.
    """

    tables: dict[str, pd.DataFrame]
    renumbered: tuple[str, ...]


def identifier_columns(schema: EntitySchema) -> Mapping[str, str]:
    """Each id and membership column of ``schema``, and the entity whose ids it holds.

    The person entity's id column holds the person entity's ids. Each group's
    id column (``{group}_id``) and the person table's membership column of it
    (``person_{group}_id``) hold the group's. In schema order: the person id,
    then each group's id and membership columns.

    Raises:
        ValueError: If two of the columns share a name.
    """

    columns = {schema.person_id_column: schema.person_entity}
    for group in schema.group_entities:
        for column in (schema.id_column(group), schema.membership_column(group)):
            if column in columns:
                raise ValueError(f"The schema names the id column {column!r} twice.")
            columns[column] = group
    return MappingProxyType(columns)


def stored_identifier_repeats(
    schema: EntitySchema,
    tables: Mapping[str, pd.DataFrame],
    identifiers: EngineIdentifiers,
) -> tuple[str, ...]:
    """The entities whose id column repeats a value once the engine stores it.

    Each entity's ids are converted as policyengine-core converts a stored
    input, ``np.asarray(column).astype(stored_dtype)``; a float outside the
    type converts as the engine's does, without a warning. A repeat is found on
    a sorted copy, where equal values sit side by side. Distinct ids that wrap
    to one value repeat, and so do ids the tables themselves repeat.

    Returns:
        The entities, in schema order: names only.

    Raises:
        TypeError: If the engine could not store an entity's ids at all.
        ValueError: If a table or an id column is missing.
    """

    stored_dtype = _stored_dtype(identifiers)
    _require_tables(tables)
    repeats = []
    for entity in schema.entities:
        column, values = _entity_ids(schema, tables, entity)
        stored = _stored(values, stored_dtype, entity=entity, column=column)
        ordered = np.sort(stored)
        if bool((ordered[1:] == ordered[:-1]).any()):
            repeats.append(entity)
    return tuple(repeats)


def inexact_identifier_entities(
    schema: EntitySchema,
    tables: Mapping[str, pd.DataFrame],
    identifiers: EngineIdentifiers,
) -> tuple[str, ...]:
    """The entities with an id the engine's conversion changes.

    An id is changed when it lies outside ``stored_dtype`` or is not a whole
    number, and the engine then holds another value for it. Distinct whole
    numbers inside the type convert to themselves, so every entity
    :func:`stored_identifier_repeats` names is one here, unless the tables
    themselves repeat its ids.

    Returns:
        The entities, in schema order: names only.

    Raises:
        TypeError: If the engine could not store an entity's ids at all.
        ValueError: If a table or an id column is missing.
    """

    stored_dtype = _stored_dtype(identifiers)
    _require_tables(tables)
    inexact = []
    for entity in schema.entities:
        column, values = _entity_ids(schema, tables, entity)
        if not _exact(values, stored_dtype, entity=entity, column=column):
            inexact.append(entity)
    return tuple(inexact)


def rank_identifier_tables(
    schema: EntitySchema,
    tables: Mapping[str, pd.DataFrame],
    entities: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """``tables`` with each named entity's ids replaced by 1 plus their rank.

    The step that ranks and maps. It decides nothing:
    :func:`engine_identifier_tables` chooses the entities. Each named entity's
    id column becomes 1 plus each id's rank among the entity's ids, ascending,
    as int64, so a group table sorted by id is numbered 1 to n in row order.
    For a named group, the person table's membership column becomes the engine
    id of the group each person names, 1 plus that group's rank, found by
    ``searchsorted`` on the sorted ids. Never the 0-based rank. Rows keep their
    order and index, and every other column is unchanged. A declared link's
    table is never mapped, so one among ``tables`` that names an entity to
    renumber is refused rather than handed back holding the old ids.

    Args:
        schema: The entity structure.
        tables: Entity -> table. Nothing is written into them.
        entities: The entities to renumber, any declared ones in any order.

    Returns:
        Entity -> table, in the order of ``tables``: copies of the named
        entities' tables and, when a group is named, of the person table. Every
        other table is the given object.

    Raises:
        TypeError: If ``entities`` is one name rather than a collection.
        ValueError: If an entity is not declared; if ``tables`` hold a
            declared link's table that names an entity to renumber; or if a
            named entity's ids are missing or repeat, or a membership names no
            group of a named entity, which a Frame rules out.
    """

    _require_tables(tables)
    if isinstance(entities, str):
        raise TypeError("entities must be a collection of entity names, not one.")
    named = set(entities)
    undeclared = sorted(named - set(schema.entities))
    if undeclared:
        raise ValueError(
            f"Cannot rank the undeclared entities {undeclared}; the schema "
            f"declares {list(schema.entities)}."
        )
    linked = [
        link
        for link in schema.links
        if link.name in tables and named & {link.left_entity, link.right_entity}
    ]
    if linked:
        sides = {
            side for link in linked for side in (link.left_entity, link.right_entity)
        }
        raise ValueError(
            f"Cannot renumber {sorted(named & sides)}: the tables hold the link "
            f"tables {[link.name for link in linked]}, whose ids this step does "
            "not map."
        )
    result = dict(tables)
    person_entity = schema.person_entity
    person_columns: dict[str, np.ndarray] = {}
    for entity in (entity for entity in schema.entities if entity in named):
        table = _entity_table(tables, entity)
        column = schema.entity_id_column(entity)
        ranks, ordered = _ranks(table, entity, column)
        if entity == person_entity:
            person_columns[column] = ranks
            continue
        renumbered = table.copy()
        renumbered[column] = ranks
        result[entity] = renumbered
        membership = schema.membership_column(entity)
        person = _entity_table(tables, person_entity)
        members = _complete_values(person, person_entity, membership)
        person_columns[membership] = 1 + _positions(
            ordered,
            members,
            group=entity,
            person_entity=person_entity,
            column=membership,
        )
    if person_columns:
        person = _entity_table(tables, person_entity).copy()
        for column, values in person_columns.items():
            person[column] = values
        result[person_entity] = person
    return result


def engine_identifier_tables(
    schema: EntitySchema,
    tables: Mapping[str, pd.DataFrame],
    identifiers: EngineIdentifiers,
) -> EngineIdentifierTables:
    """The tables one simulation receives: the engine's rule over its own tables.

    Pass ``tables`` as the engine would receive them (an adapter's engine
    copies of the frame's tables). An entity whose ids would repeat once stored
    (:func:`stored_identifier_repeats`) is renumbered by
    :func:`rank_identifier_tables`; every other entity passes its ids through.
    When nothing repeats, the result holds the given tables themselves.

    Raises:
        TypeError: If the engine could not store an entity's ids at all.
        ValueError: If a table or an id column is missing, or an entity to
            renumber has more rows than the stored type can number from 1 or
            is named by a declared link whose table ``tables`` hold.
        RuntimeError: If any entity's engine ids still repeat once stored, so
            the rank step no longer keeps them apart.
    """

    stored_dtype = _stored_dtype(identifiers)
    repeats = stored_identifier_repeats(schema, tables, identifiers)
    if not repeats:
        return EngineIdentifierTables(tables=dict(tables), renumbered=())
    capacity = int(np.iinfo(stored_dtype).max)
    crowded = [entity for entity in repeats if len(tables[entity]) > capacity]
    if crowded:
        raise ValueError(
            f"The ids of {crowded} repeat once the engine stores them as "
            f"{stored_dtype}, and those entities have more rows than "
            f"{stored_dtype} can number from 1, so their ranks would not fit it "
            "either."
        )
    ranked = rank_identifier_tables(schema, tables, repeats)
    still = stored_identifier_repeats(schema, ranked, identifiers)
    if still:
        raise RuntimeError(
            f"After renumbering {list(repeats)}, the engine ids of {list(still)} "
            f"still repeat once stored as {stored_dtype}."
        )
    return EngineIdentifierTables(tables=ranked, renumbered=repeats)


def refuse_inexact_identifier_requests(
    schema: EntitySchema,
    variables: Iterable[str],
    identifiers: EngineIdentifiers,
    *,
    table: Callable[[str], pd.DataFrame],
) -> None:
    """Refuse a request for an id the engine cannot return as the frame holds it.

    ``variables`` is first intersected with :func:`identifier_columns`; when no
    requested name is an id or membership column, this returns having read no
    table. Otherwise ``table(entity)`` is read once for each requested column's
    entity, and the request is refused when that entity's ids are inexact
    (:func:`inexact_identifier_entities`): the engine holds wrapped ids or
    ranks for it, not the frame's.

    Args:
        schema: The entity structure.
        variables: The requested variable names, iterated once.
        identifiers: How the engine stores the id and membership columns.
        table: Reads one entity's table, such as ``Frame.table``.

    Raises:
        TypeError: If ``variables`` is one name rather than a collection.
        ValueError: Naming the refused variables, their entities and the type.
    """

    stored_dtype = _stored_dtype(identifiers)
    if isinstance(variables, str):
        raise TypeError("variables must be a collection of names, not one name.")
    columns = identifier_columns(schema)
    requested = [name for name in dict.fromkeys(variables) if name in columns]
    if not requested:
        return
    exact: dict[str, bool] = {}
    refused = []
    for name in requested:
        entity = columns[name]
        if entity not in exact:
            column = schema.entity_id_column(entity)
            values = _values(table(entity), entity, column)
            exact[entity] = _exact(values, stored_dtype, entity=entity, column=column)
        if not exact[entity]:
            refused.append(name)
    if refused:
        entities = list(dict.fromkeys(columns[name] for name in refused))
        raise ValueError(
            f"The engine stores ids as {stored_dtype} and cannot return "
            f"{refused} as the frame holds them: the ids of {entities} are not "
            f"all whole numbers inside {stored_dtype}, so the engine holds other "
            "values for them."
        )


def require_stored_dtypes(
    identifiers: EngineIdentifiers, dtypes: Mapping[str, object]
) -> None:
    """Refuse an engine that stores an id column as a type other than the declared one.

    Args:
        identifiers: The adapter's declaration.
        dtypes: Column -> the type the installed engine stores it as (such as a
            variable's ``dtype``), normalised by ``np.dtype``. It must hold
            every column of ``identifier_columns(schema)``; a column left out
            is not checked.

    Raises:
        ValueError: Naming each differing column, its type and the declared
            type.
    """

    declared = _stored_dtype(identifiers)
    differing = []
    for column, given in dtypes.items():
        try:
            dtype = np.dtype(given)
        except TypeError:
            differing.append(f"{column!r} as {given!r}")
            continue
        if dtype != declared:
            differing.append(f"{column!r} as {dtype}")
    if differing:
        raise ValueError(
            f"The installed engine stores {', '.join(differing)}, not as the "
            f"declared {declared}; review the adapter's id declaration against "
            "the engine."
        )


def _stored_dtype(identifiers: object) -> np.dtype:
    if not isinstance(identifiers, EngineIdentifiers):
        raise TypeError(
            "identifiers must be an EngineIdentifiers declaration, got "
            f"{type(identifiers).__name__}."
        )
    return identifiers.stored_dtype


def _require_tables(tables: object) -> None:
    if not isinstance(tables, Mapping):
        raise TypeError(
            f"tables must map each entity to its table, got {type(tables).__name__}."
        )


def _entity_table(tables: Mapping[str, pd.DataFrame], entity: str) -> pd.DataFrame:
    if entity not in tables:
        raise ValueError(f"The tables hold no {entity!r} table.")
    return tables[entity]


def _values(table: pd.DataFrame, entity: str, column: str) -> np.ndarray:
    """A column's values as numpy reads them, which is how the engine reads them."""

    if column not in table.columns:
        raise ValueError(f"The {entity!r} table lacks the column {column!r}.")
    return np.asarray(table[column])


def _complete_values(table: pd.DataFrame, entity: str, column: str) -> np.ndarray:
    values = _values(table, entity, column)
    if bool(table[column].isna().any()):
        raise ValueError(f"The {entity!r} column {column!r} has missing values.")
    return values


def _entity_ids(
    schema: EntitySchema, tables: Mapping[str, pd.DataFrame], entity: str
) -> tuple[str, np.ndarray]:
    column = schema.entity_id_column(entity)
    return column, _values(_entity_table(tables, entity), entity, column)


def _stored(
    values: np.ndarray, stored_dtype: np.dtype, *, entity: str, column: str
) -> np.ndarray:
    """``values`` as the engine stores them: numpy's ``astype``, as core converts."""

    try:
        with np.errstate(invalid="ignore"):
            return values.astype(stored_dtype)
    except (TypeError, ValueError, OverflowError):
        # numpy's own message can quote a value, so it is not chained.
        raise TypeError(
            f"The engine cannot store the {entity!r} ids in {column!r} "
            f"({values.dtype}) as {stored_dtype}."
        ) from None


def _exact(
    values: np.ndarray, stored_dtype: np.dtype, *, entity: str, column: str
) -> bool:
    """Whether every value converts to itself: a whole number inside the type.

    The bounds are compared in the values' own type, or in float64 for a
    narrower float, so no comparison rounds.
    """

    if values.size == 0:
        return True
    info = np.iinfo(stored_dtype)
    if values.dtype.kind in "iu":
        if np.can_cast(values.dtype, stored_dtype, casting="safe"):
            return True
        held = np.iinfo(values.dtype)
        low, high = max(info.min, held.min), min(info.max, held.max)
        return bool(((values >= low) & (values <= high)).all())
    if values.dtype.kind == "f":
        if values.dtype.itemsize < 8:
            values = values.astype(np.float64)
        with np.errstate(invalid="ignore"):
            # Both bounds are powers of two, which a float holds exactly.
            inside = (values >= info.min) & (values < info.max + 1)
            whole = np.isfinite(values) & (np.floor(values) == values)
        return bool((whole & inside).all())
    stored = _stored(values, stored_dtype, entity=entity, column=column)
    return bool(np.all(stored.astype(object) == values.astype(object)))


def _ranks(
    table: pd.DataFrame, entity: str, column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Each row's 1-based rank among the entity's ids, and the ids sorted."""

    values = _complete_values(table, entity, column)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    if bool((ordered[1:] == ordered[:-1]).any()):
        raise ValueError(f"The {entity!r} ids in {column!r} repeat, so have no rank.")
    ranks = np.empty(values.size, dtype=np.int64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.int64)
    return ranks, ordered


def _positions(
    ordered: np.ndarray,
    members: np.ndarray,
    *,
    group: str,
    person_entity: str,
    column: str,
) -> np.ndarray:
    """Each member's 0-based position among the sorted group ids ``ordered``."""

    left, right = _comparable(ordered, members)
    positions = np.searchsorted(left, right)
    found = positions < left.size
    found[found] = left[positions[found]] == right[found]
    if not found.all():
        raise ValueError(
            f"The {person_entity!r} column {column!r} names a {group!r} id the "
            f"{group!r} table does not hold."
        )
    return positions.astype(np.int64)


def _comparable(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``left`` and ``right`` in one type that compares their values exactly.

    numpy compares int64 with uint64 through float64, which merges ids above
    2**53, so integers are brought to int64 or uint64 where both fit, and
    anything else to Python objects.
    """

    if left.dtype == right.dtype:
        return left, right
    if left.dtype.kind in "iu" and right.dtype.kind in "iu":
        if _fits_int64(left) and _fits_int64(right):
            return left.astype(np.int64), right.astype(np.int64)
        if _non_negative(left) and _non_negative(right):
            return left.astype(np.uint64), right.astype(np.uint64)
    return left.astype(object), right.astype(object)


def _fits_int64(values: np.ndarray) -> bool:
    if values.dtype.kind == "i" or values.size == 0:
        return True
    return values.dtype.itemsize < 8 or int(values.max()) <= _INT64_MAX


def _non_negative(values: np.ndarray) -> bool:
    return values.dtype.kind == "u" or values.size == 0 or int(values.min()) >= 0
