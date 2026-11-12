"""Invented frames and id layouts for the engine-identifier tests (microcosm#931).

``microcosm.frame.adapters.engine_identifiers`` gives each engine simulation
the frame's ids where the engine's stored type keeps them apart, and 1 plus
their rank where it would not. Its shared tests, the UK adapter's engine-free
tests and the UK engine tier share these layouts:

- :data:`COPY_IDS`: source ids and copies carrying the ordinals 10**11, 10**12
  and 10**13, laid out as a frame's copy ids are. The copies pass 2**31, all
  six stay apart once stored as int32, and 3 + 10**12 wraps negative, so they
  convert out of int64 order. :data:`COPY_GROUP_IDS` and
  :data:`COPY_PERSON_IDS` lay out other entities' ids the same way;
- :data:`COLLIDING_HOUSEHOLD_IDS`: 7, 9 and 7 + 2**32, the first and last of
  which store as 7;
- :data:`COLLIDING_CLONE_HOUSEHOLD_IDS`: two clones, each holding two
  households whose ids store together (h and h + 2**32, plus 10**9 in clone
  1);
- two neutral schemas, and a seeded sweep of small tables whose id columns
  take any NumPy or nullable integer dtype that holds their values
  (``concept_id_dtypes.ID_DTYPES``).

:func:`wrapped` is the tests' own reading of numpy's conversion: an integer's
low bits, two's complement. The UK builders import the UK adapter's names only
when called, so a shared test that imports this module loads no UK module.
Every id and value here is made up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from test_support.microcosm_frame.concept_id_dtypes import (
    ID_DTYPES,
    INT64_MAX,
    INT64_MIN,
    UINT64_MAX,
    holds,
    typed,
)

#: The span of a 32-bit store: two ids this far apart store as one value.
WRAP = 2**32
#: The offset the invented clone layouts add to every id of clone 1.
CLONE_OFFSET = 10**9
#: The shift that carries an invented layout's ids past 2**31.
SHIFT = 10**11

#: Ids laid out as copy ids are (source ids 3 and 5, and copies carrying the
#: ordinals 10**11, 10**12 and 10**13), ascending.
COPY_IDS = (3, 5, 3 + 10**11, 5 + 10**11, 3 + 10**12, 3 + 10**13)
#: The same layout from the source ids 4 and 6, for a second group entity.
COPY_GROUP_IDS = (4, 6, 4 + 10**11, 6 + 10**11, 4 + 10**12, 4 + 10**13)
#: The same layout from the source ids 1 and 2, for a third group entity.
COPY_THIRD_GROUP_IDS = (1, 2, 1 + 10**11, 2 + 10**11, 1 + 10**12, 1 + 10**13)
#: Twenty-one person ids in the same layout: the odd source ids 11 to 23, each
#: also carrying the ordinals 10**11 and 10**12.
COPY_PERSON_IDS = tuple(
    source + ordinal for ordinal in (0, 10**11, 10**12) for source in range(11, 24, 2)
)
#: Ids of the same shapes inside int32, which store as themselves.
INSIDE_IDS = (2, 3, 5, 7, 11, 13)
INSIDE_GROUP_IDS = (1, 4, 6, 8, 9, 10)
INSIDE_THIRD_GROUP_IDS = (12, 14, 15, 16, 18, 20)
INSIDE_PERSON_IDS = tuple(range(101, 122))
#: Members of each of the six households in the layouts above, in id order:
#: all different, so a household matched to another's row would show.
LAYOUT_MEMBERS = (3, 1, 4, 2, 6, 5)

#: Three households, the first and last of which store as 7 in int32.
COLLIDING_HOUSEHOLD_IDS = (7, 9, 7 + WRAP)
#: Two clones of two households each; within each clone both ids store as one
#: int32 value (7 for clone 0, 1000000007 for clone 1).
COLLIDING_CLONE_HOUSEHOLD_IDS: Mapping[int, tuple[int, int]] = {
    0: (7, 7 + WRAP),
    1: (7 + CLONE_OFFSET, 7 + WRAP + CLONE_OFFSET),
}

#: A person entity and two groups, and one with a third group.
FAMILY_SCHEMA = EntitySchema(group_entities=("family", "household"))
THREE_GROUP_SCHEMA = EntitySchema(group_entities=("family", "household", "dwelling"))
NEUTRAL_SCHEMAS = (FAMILY_SCHEMA, THREE_GROUP_SCHEMA)

#: The stored types the sweep declares: narrow, so repeats are common.
SWEEP_STORED_DTYPES = (np.dtype(np.int8), np.dtype(np.int16))

_LAYOUTS: Mapping[str, Mapping[str, tuple[int, ...]]] = {
    "inside": {
        "person": INSIDE_PERSON_IDS,
        "household": INSIDE_IDS,
        "family": INSIDE_GROUP_IDS,
        "dwelling": INSIDE_THIRD_GROUP_IDS,
    },
    "copies": {
        "person": COPY_PERSON_IDS,
        "household": COPY_IDS,
        "family": COPY_GROUP_IDS,
        "dwelling": COPY_THIRD_GROUP_IDS,
    },
}
#: Ages by place in a household of the layouts: two adults, then children.
_AGES = (40.0, 38.0, 12.0, 9.0, 6.0, 3.0)
_ROW_SEED = 931


def wrapped(value: int, dtype: np.dtype) -> int:
    """``value`` as numpy's ``astype`` stores it in the integer ``dtype``.

    Its low bits, read as two's complement for a signed type: the tests'
    reading of the conversion, written apart from numpy's.
    """

    dtype = np.dtype(dtype)
    bits = 8 * dtype.itemsize
    low = int(value) % 2**bits
    if dtype.kind == "i" and low >= 2 ** (bits - 1):
        low -= 2**bits
    return low


def _people(person_ids: Sequence[int], members: Sequence[int]):
    """(person id, household place, place in the household) per person row.

    The ids are taken in order, household by household, then the rows are
    listed in a fixed scrambled order, so they are not in id order.
    """

    if sum(members) != len(person_ids):
        raise ValueError("The layout's members do not match its person ids.")
    rows = []
    ids = iter(person_ids)
    for household, size in enumerate(members):
        for place in range(size):
            rows.append((next(ids), household, place))
    order = np.random.default_rng(_ROW_SEED).permutation(len(rows))
    return [rows[index] for index in order]


# ---------------------------------------------------------------------------
# Neutral tables
# ---------------------------------------------------------------------------


def layout_tables(schema: EntitySchema, layout: str) -> dict[str, pd.DataFrame]:
    """Six households' tables, their ids ``"inside"`` int32 or laid out as
    ``"copies"`` are.

    Six households of :data:`LAYOUT_MEMBERS` people; family ``i`` is household
    ``i``'s, and a third group spreads the people over its six ids. Every id is
    int64, groups ascending, people out of id order.
    """

    ids = _LAYOUTS[layout]
    rows = _people(ids["person"], LAYOUT_MEMBERS)
    person: dict[str, object] = {
        "person_id": np.asarray([row[0] for row in rows], dtype=np.int64)
    }
    places = {
        "family": [row[1] for row in rows],
        "household": [row[1] for row in rows],
        "dwelling": [(5 * position) % 6 for position, _row in enumerate(rows)],
    }
    for group in schema.group_entities:
        group_ids = ids[group]
        person[schema.membership_column(group)] = np.asarray(
            [group_ids[place] for place in places[group]], dtype=np.int64
        )
    person["age"] = np.asarray([_AGES[row[2]] for row in rows])
    tables = {"person": pd.DataFrame(person)}
    for group in schema.group_entities:
        group_ids = np.asarray(ids[group], dtype=np.int64)
        tables[group] = pd.DataFrame(
            {schema.id_column(group): group_ids, f"{group}_size": np.arange(6) + 1.5}
        )
    return tables


def colliding_household_tables() -> dict[str, pd.DataFrame]:
    """Tables of households 7, 9 and 7 + 2**32, people out of id order.

    The first and last households store as 7. Families and people fit int32.
    """

    first, second, third = COLLIDING_HOUSEHOLD_IDS
    person = pd.DataFrame(
        {
            "person_id": np.asarray([13, 11, 15, 12, 14, 16], dtype=np.int64),
            "person_family_id": np.asarray([4, 1, 3, 2, 4, 3], dtype=np.int64),
            "person_household_id": np.asarray(
                [third, first, second, first, third, second], dtype=np.int64
            ),
            "age": [40.0, 38.0, 12.0, 9.0, 6.0, 3.0],
        }
    )
    family = pd.DataFrame(
        {"family_id": np.asarray([1, 2, 3, 4], dtype=np.int64), "rooms": 3.0}
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray(COLLIDING_HOUSEHOLD_IDS, dtype=np.int64),
            "rent": [700.0, 650.0, 900.0],
        }
    )
    return {"person": person, "family": family, "household": household}


# ---------------------------------------------------------------------------
# The seeded sweep
# ---------------------------------------------------------------------------


class SweepFrame(NamedTuple):
    """One sweep case's invented ids, as Python ints in row order.

    ``ids[entity]`` lists an entity's ids in its table's row order (groups
    ascending, as a Frame keeps them; people in any order), and
    ``memberships[group]`` each person row's id of ``group``. Every group has a
    member. Each entity's ids lie wholly in int64's range or wholly in
    uint64's, so some integer dtype holds them.
    """

    schema: EntitySchema
    ids: dict[str, list[int]]
    memberships: dict[str, list[int]]


#: Values within one of a power of two where int8, int16 or int32 wraps, and
#: the 64-bit extremes.
_SWEEP_EDGES = (
    127,
    128,
    255,
    256,
    32767,
    32768,
    65535,
    65536,
    2**31 - 1,
    2**31,
    2**32 - 1,
    2**32,
    2**63 - 1,
    2**63,
    2**64 - 1,
)
#: Spans whose multiples wrap to one value in int8 (all three) or int16 (the
#: last two).
_SWEEP_SPANS = (256, 65536, WRAP)
_SWEEP_ORDINALS = (CLONE_OFFSET, 10**11, 10**12, 10**13)


def _sweep_ids(rng: np.random.Generator, count: int) -> list[int]:
    """``count`` distinct ids, many a span apart from another so they wrap."""

    unsigned = bool(rng.random() < 0.2)
    low, high = (0, UINT64_MAX) if unsigned else (INT64_MIN, INT64_MAX)
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < count:
        branch = int(rng.integers(0, 5 if unsigned else 4))
        if branch == 1 and values:
            base = values[int(rng.integers(0, len(values)))]
            span = _SWEEP_SPANS[int(rng.integers(0, len(_SWEEP_SPANS)))]
            value = base + int(rng.integers(-2, 3)) * span
        elif branch == 2:
            edge = _SWEEP_EDGES[int(rng.integers(0, len(_SWEEP_EDGES)))]
            value = edge if unsigned or rng.random() < 0.5 else -edge
        elif branch == 3:
            ordinal = _SWEEP_ORDINALS[int(rng.integers(0, len(_SWEEP_ORDINALS)))]
            value = int(rng.integers(0, 301)) + ordinal
        elif branch == 4:
            value = UINT64_MAX - int(rng.integers(0, 2**62))
        else:
            value = int(rng.integers(0 if unsigned else -300, 301))
        if low <= value <= high and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def sweep_frame(
    rng: np.random.Generator, schema: EntitySchema, *, max_people: int = 40
) -> SweepFrame:
    """A sweep case: up to ``max_people`` people, each group drawn its own size."""

    people = int(rng.integers(1, max_people + 1))
    ids = {schema.person_entity: _sweep_ids(rng, people)}
    memberships: dict[str, list[int]] = {}
    for group in schema.group_entities:
        count = int(rng.integers(1, people + 1))
        group_ids = sorted(_sweep_ids(rng, count))
        # The first ``count`` people of a shuffle take one group each, so every
        # group has a member; the rest join any.
        order = rng.permutation(people)
        place = np.empty(people, dtype=np.int64)
        place[order[:count]] = np.arange(count)
        place[order[count:]] = rng.integers(0, count, size=people - count)
        ids[group] = group_ids
        memberships[group] = [group_ids[index] for index in place]
    return SweepFrame(schema, ids, memberships)


def _drawn_column(rng: np.random.Generator, values: list[int]):
    dtypes = [dtype for dtype in ID_DTYPES if holds(dtype, values)]
    return typed(values, dtypes[int(rng.integers(0, len(dtypes)))])


def sweep_tables(
    frame: SweepFrame, rng: np.random.Generator
) -> dict[str, pd.DataFrame]:
    """``frame``'s tables, each id and membership column of a dtype drawn
    from those in ``ID_DTYPES`` that hold its values, beside one made-up
    value column per table."""

    schema = frame.schema
    person_ids = frame.ids[schema.person_entity]
    person: dict[str, object] = {
        schema.person_id_column: _drawn_column(rng, person_ids)
    }
    for group in schema.group_entities:
        person[schema.membership_column(group)] = _drawn_column(
            rng, frame.memberships[group]
        )
    person[f"{schema.person_entity}_value"] = np.arange(len(person_ids)) + 0.5
    tables = {schema.person_entity: pd.DataFrame(person)}
    for group in schema.group_entities:
        group_ids = frame.ids[group]
        tables[group] = pd.DataFrame(
            {
                schema.id_column(group): _drawn_column(rng, group_ids),
                f"{group}_value": np.arange(len(group_ids)) + 0.25,
            }
        )
    return tables


# ---------------------------------------------------------------------------
# UK frames
# ---------------------------------------------------------------------------


def _uk_names():
    """The UK adapter's schema and batch column, imported when a UK frame is
    built so that importing this module loads no UK module."""

    from microcosm.frame.adapters.policyengine_uk import UK_SCHEMA
    from microcosm.frame.adapters.policyengine_uk_population import (
        UK_SPINE_ENGINE_BATCH_COLUMN,
    )

    return UK_SCHEMA, UK_SPINE_ENGINE_BATCH_COLUMN


def uk_frame(
    *,
    household_ids: Sequence[int],
    benunit_ids: Sequence[int],
    people: Sequence[tuple[int, int, int]],
    person_columns: Mapping[str, Sequence[object]] | None = None,
    benunit_columns: Mapping[str, Sequence[object]] | None = None,
    household_columns: Mapping[str, Sequence[object]] | None = None,
    household_weights: Sequence[float] | None = None,
) -> Frame:
    """A Frame of the UK adapter's schema, its ids int64 as the spine's are.

    Households and benefit units take their ids in the order given (ascending,
    as a Frame keeps them); ``people`` are (person id, benefit unit id,
    household id) rows in the order given. Each ``*_columns`` mapping adds
    columns in its table's row order; a NumPy or pandas array keeps its dtype.
    Households weigh 100 each unless ``household_weights`` are given.
    """

    schema, _batch_column = _uk_names()
    person = pd.DataFrame(
        {
            "person_id": np.asarray([row[0] for row in people], dtype=np.int64),
            "person_benunit_id": np.asarray([row[1] for row in people], dtype=np.int64),
            "person_household_id": np.asarray(
                [row[2] for row in people], dtype=np.int64
            ),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.asarray(benunit_ids, dtype=np.int64)})
    household = pd.DataFrame(
        {"household_id": np.asarray(household_ids, dtype=np.int64)}
    )
    for table, columns in (
        (person, person_columns),
        (benunit, benunit_columns),
        (household, household_columns),
    ):
        for name, values in (columns or {}).items():
            # A NumPy or pandas array keeps its dtype; anything else is a list.
            keeps = isinstance(values, np.ndarray | pd.api.extensions.ExtensionArray)
            table[name] = values if keeps else list(values)
    weights = (
        np.full(len(household), 100.0)
        if household_weights is None
        else np.asarray(household_weights, dtype=np.float64)
    )
    return Frame(
        tables={"person": person, "benunit": benunit, "household": household},
        schema=schema,
        weights={"household": Weights(values=weights, kind=WeightKind.DESIGN)},
    )


def clone_layout_frame(
    clones: int = 3,
    *,
    shift: int = SHIFT,
    shifted: Sequence[str] = ("person", "benunit", "household"),
) -> Frame:
    """The UK engine-free adapter tests' clone layout, shifted.

    Two households per clone and one person each, as ``_clone_frame`` lays
    them out (household and benefit-unit ids 1 to n, people listed in reverse
    with ids ten times their household's), without the clone column. Every id
    of an entity in ``shifted`` is moved by ``shift``, so, with the default,
    the ids pass 2**31 and stay apart once stored as int32, as K = 1's SPI
    copies do.
    """

    count = 2 * clones
    ids = np.arange(1, count + 1, dtype=np.int64)

    def moved(entity: str, values: np.ndarray) -> list[int]:
        return [int(value) + (shift if entity in shifted else 0) for value in values]

    reverse = ids[::-1]
    people = list(
        zip(
            moved("person", reverse * 10),
            moved("benunit", reverse),
            moved("household", reverse),
            strict=True,
        )
    )
    return uk_frame(
        household_ids=moved("household", ids),
        benunit_ids=moved("benunit", ids),
        people=people,
        person_columns={"age": [40.0] * count},
    )


def copy_layout_frame(
    *,
    person_columns: Mapping[str, Sequence[object]] | None = None,
    benunit_columns: Mapping[str, Sequence[object]] | None = None,
    household_columns: Mapping[str, Sequence[object]] | None = None,
) -> Frame:
    """A UK frame whose every entity's ids are laid out as copy ids are.

    Households :data:`COPY_IDS` with :data:`LAYOUT_MEMBERS` people each (all
    different), one benefit unit each (:data:`COPY_GROUP_IDS`), people
    :data:`COPY_PERSON_IDS` out of id order. The copies' ids pass 2**31; every
    id stays apart once stored as int32, and the ids convert out of int64
    order. People carry ``age`` (two adults, then children); extra columns go
    in row order.
    """

    rows = _people(COPY_PERSON_IDS, LAYOUT_MEMBERS)
    people = [(row[0], COPY_GROUP_IDS[row[1]], COPY_IDS[row[1]]) for row in rows]
    columns = {"age": [_AGES[row[2]] for row in rows], **(person_columns or {})}
    return uk_frame(
        household_ids=COPY_IDS,
        benunit_ids=COPY_GROUP_IDS,
        people=people,
        person_columns=columns,
        benunit_columns=benunit_columns,
        household_columns=household_columns,
        household_weights=[100.0 + 10.0 * index for index in range(len(COPY_IDS))],
    )


def colliding_household_frame(
    household_ids: Sequence[int] = COLLIDING_HOUSEHOLD_IDS,
    *,
    person_columns: Mapping[str, Sequence[object]] | None = None,
    benunit_columns: Mapping[str, Sequence[object]] | None = None,
    household_columns: Mapping[str, Sequence[object]] | None = None,
) -> Frame:
    """A UK frame of households 7, 9 and 7 + 2**32, or ``household_ids``.

    Pass ``(1, 2, 3)`` for the same frame renumbered by hand. The households
    hold two, one and three people, one benefit unit each; benefit units and
    people fit int32, and people are listed out of id order. People carry
    ``age``; extra columns go in row order (people as listed: the third
    household's adult, the first's adult, the second's, the third's two
    children, the first's child).
    """

    first, second, third = household_ids
    people = [
        (34, 23, third),
        (31, 21, first),
        (33, 22, second),
        (35, 23, third),
        (36, 23, third),
        (32, 21, first),
    ]
    columns = {
        "age": [38.0, 40.0, 70.0, 12.0, 9.0, 6.0],
        **(person_columns or {}),
    }
    return uk_frame(
        household_ids=household_ids,
        benunit_ids=(21, 22, 23),
        people=people,
        person_columns=columns,
        benunit_columns=benunit_columns,
        household_columns=household_columns,
        household_weights=(110.0, 90.0, 130.0),
    )


def colliding_clones_frame(
    *,
    person_columns: Mapping[str, Sequence[object]] | None = None,
    benunit_columns: Mapping[str, Sequence[object]] | None = None,
    household_columns: Mapping[str, Sequence[object]] | None = None,
) -> Frame:
    """A UK frame of two clones whose households store together.

    The households carry the clone column. Clone 0 holds households 7 and
    7 + 2**32 and clone 1 the same plus 10**9, so within each clone both
    household ids store as one int32 value. The first household of each clone
    holds an adult and a child, the second one adult, each household one
    benefit unit. Benefit-unit and person ids pass 2**31 and stay apart once
    stored, clone 1's being clone 0's plus 10**9. People are listed out of id
    order with the clones interleaved, and carry ``age`` and the
    ``months_since_last_birthday`` the spine's batched pass requires.
    """

    _schema, batch_column = _uk_names()
    households, clones, benunits, people, ages, months = [], [], [], [], [], []
    for clone, (paired, wrapped_pair) in COLLIDING_CLONE_HOUSEHOLD_IDS.items():
        offset = clone * CLONE_OFFSET
        first_unit, second_unit = 11 + 10**11 + offset, 11 + 10**12 + offset
        households += [paired, wrapped_pair]
        clones += [clone, clone]
        benunits += [first_unit, second_unit]
        people += [
            (3 + 10**11 + offset, first_unit, paired),
            (3 + 10**12 + offset, first_unit, paired),
            (5 + 10**11 + offset, second_unit, wrapped_pair),
        ]
        ages += [40.0, 9.0, 35.0]
        months += [4.0 + clone, 10.0 - clone, 7.0 + 3 * clone]
    household_order = np.argsort(households)
    benunit_order = np.argsort(benunits)
    # Clone 1's second household, clone 0's child, clone 1's adult, and so on.
    rows = (5, 1, 3, 2, 4, 0)
    return uk_frame(
        household_ids=[households[index] for index in household_order],
        benunit_ids=[benunits[index] for index in benunit_order],
        people=[people[row] for row in rows],
        person_columns={
            "age": [ages[row] for row in rows],
            "months_since_last_birthday": [months[row] for row in rows],
            **(person_columns or {}),
        },
        benunit_columns=benunit_columns,
        household_columns={
            batch_column: np.asarray(
                [clones[index] for index in household_order], dtype=np.int64
            ),
            **(household_columns or {}),
        },
        household_weights=(100.0, 120.0, 140.0, 160.0),
    )


def relabelled(frame: Frame, entity: str, ids: Sequence[int]) -> Frame:
    """``frame`` renumbered by hand: ``entity``'s ids replaced row for row by
    ``ids``, and each membership of it mapped to its group's new id through a
    dict, independently of the shared module's ranks. A group's new ids must
    be ascending, as a Frame keeps them."""

    schema = frame.schema
    tables = {name: frame.table(name).copy() for name in frame.entities}
    column = schema.entity_id_column(entity)
    old = [int(value) for value in tables[entity][column]]
    new = dict(zip(old, (int(value) for value in ids), strict=True))
    tables[entity][column] = np.asarray([new[value] for value in old], dtype=np.int64)
    if entity != schema.person_entity:
        membership = schema.membership_column(entity)
        person = tables[schema.person_entity]
        person[membership] = np.asarray(
            [new[int(value)] for value in person[membership]], dtype=np.int64
        )
    return Frame(
        tables,
        schema,
        {name: frame.weights_for(name) for name in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
