"""Synthetic UK id frames for the copy-id tests (microcosm#931).

The clone-first spine (#931 group (b)) will mint geographic clone ``k`` of
every FRS row at ``id + k x 10**9`` (the copy-id table's
``geographic_support`` field) before any copy layer runs. This module builds
such frames from synthetic sernums, with ids as ``frs_spine`` makes them
(``benunit_id`` sernum x 100 + b, ``person_id`` sernum x 1000 + p), and copies
households through the six copy layers in the spine's order: the SPI copy,
the income band donor, the high-value copies, the CGT support copies, the
incidence twin and the residential arms. Each layer picks its households by
source id alone, so a K=1 frame's copies are exactly its clone 0's in a K=3
frame. ``rule="table"`` mints every copy through the copy-id table, as the
stages do; ``rule="retired"`` replays the rule the table replaced (ten to the
digit count of the largest id the layer sees), for comparison.

:func:`assert_copy_ids_follow_lineage` checks a real stage's output: every new
household is matched to its root in the stage's input by lineage columns, never
by id arithmetic, and all five id columns must equal the root's plus the copy's
ordinal times the field's declared radix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime import copy_id_table
from microcosm.frame import Frame

__all__ = [
    "COPY_PATH_COLUMNS",
    "GEOGRAPHIC_RADIX",
    "IdTables",
    "assert_copy_ids_follow_lineage",
    "base_id_tables",
    "clone_first_tables",
    "copy_chain",
    "copy_households",
    "id_frame",
]

GEOGRAPHIC_RADIX = copy_id_table.radix("geographic_support")
#: Each layer's ordinal on the household table, in the spine's order.
COPY_PATH_COLUMNS = ("spi", "hv", "cgt", "twin", "arm")
_PERSON_ID_COLUMNS = ("person_id", "person_household_id", "person_benunit_id")


@dataclass(frozen=True)
class IdTables:
    """The five id columns of a UK frame, with each household's copy path."""

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame

    def ids(self, entity: str) -> np.ndarray:
        column = {
            "person": "person_id",
            "benunit": "benunit_id",
            "household": "household_id",
        }[entity]
        return getattr(self, entity)[column].to_numpy(dtype=np.int64)


def base_id_tables(n_households: int = 60) -> IdTables:
    """Synthetic FRS ids: sernums 1..n, one to four people in one or two units."""

    people: list[tuple[int, int, int]] = []
    benunits: list[int] = []
    for sernum in range(1, n_households + 1):
        size = 1 + sernum % 4
        units = min(size, 1 + sernum % 2)
        benunits.extend(sernum * 100 + b for b in range(1, units + 1))
        people.extend(
            (sernum * 1000 + p, sernum, sernum * 100 + min(p, units))
            for p in range(1, size + 1)
        )
    sernums = np.arange(1, n_households + 1, dtype=np.int64)
    household = pd.DataFrame(
        {
            "household_id": sernums,
            "source_household_id": sernums,
            "household_clone_index": np.zeros(n_households, dtype=np.int64),
            **{
                column: np.zeros(n_households, dtype=np.int64)
                for column in COPY_PATH_COLUMNS
            },
        }
    )
    person = pd.DataFrame(people, columns=list(_PERSON_ID_COLUMNS)).astype("int64")
    benunit = pd.DataFrame({"benunit_id": np.asarray(benunits, dtype=np.int64)})
    return IdTables(person=person, benunit=benunit, household=household)


def clone_first_tables(tables: IdTables, n_clones: int) -> IdTables:
    """``n_clones`` geographic clones of every row, clone-major.

    Clone ``k`` takes every id plus ``k x 10**9`` and ``household_clone_index``
    ``k``; ``source_household_id`` stays the clone-0 id.
    """

    persons, benunits, households = (
        [tables.person],
        [tables.benunit],
        [tables.household],
    )
    for k in range(1, n_clones):
        person = tables.person.copy()
        for column in _PERSON_ID_COLUMNS:
            person[column] = copy_id_table.mint(person[column], "geographic_support", k)
        benunit = tables.benunit.copy()
        benunit["benunit_id"] = copy_id_table.mint(
            benunit["benunit_id"], "geographic_support", k
        )
        household = tables.household.copy()
        household["household_id"] = copy_id_table.mint(
            household["household_id"], "geographic_support", k
        )
        household["household_clone_index"] = k
        persons.append(person)
        benunits.append(benunit)
        households.append(household)
    return IdTables(
        person=pd.concat(persons, ignore_index=True),
        benunit=pd.concat(benunits, ignore_index=True),
        household=pd.concat(households, ignore_index=True),
    )


def id_frame(tables: IdTables, *, time_period: str = "2024") -> Frame:
    """``tables`` as a UK national frame at unit household weights."""

    from microcosm.build.uk_runtime.national_frame import uk_national_frame

    return uk_national_frame(
        person=tables.person,
        benunit=tables.benunit,
        household=tables.household,
        household_weights=np.ones(len(tables.household)),
        time_period=time_period,
    )


def _largest_id(tables: IdTables) -> int:
    return max(
        int(tables.person[list(_PERSON_ID_COLUMNS)].to_numpy().max()),
        int(tables.benunit["benunit_id"].max()),
        int(tables.household["household_id"].max()),
    )


def copy_households(
    tables: IdTables,
    *,
    field: str,
    path_column: str,
    ordinals: dict[int, Sequence[int]],
    rule: str,
) -> IdTables:
    """Append copies of households at each ordinal, as the stages mint them.

    ``ordinals`` maps a household's position to the ordinals it is copied at;
    each copy carries its household's path with ``path_column`` set to the
    ordinal. ``rule="table"`` mints through the copy-id table's ``field``;
    ``rule="retired"`` offsets by ten to the digit count of the largest id
    in ``tables``, computed once for the layer as the stages did.
    """

    if rule == "retired":
        multiplier = 10 ** max(1, len(str(_largest_id(tables))))
    elif rule != "table":
        raise ValueError(f"Unknown copy-id rule {rule!r}.")
    household = tables.household
    persons, benunits, households = [tables.person], [tables.benunit], [household]
    for k in sorted({k for values in ordinals.values() for k in values}):
        positions = [p for p, values in ordinals.items() if k in values]
        copied = household.iloc[positions].copy()
        ids = set(copied["household_id"].tolist())
        members = tables.person.loc[
            tables.person["person_household_id"].isin(ids)
        ].copy()
        units = tables.benunit.loc[
            tables.benunit["benunit_id"].isin(set(members["person_benunit_id"]))
        ].copy()

        def minted(values: pd.Series, k: int = k) -> np.ndarray:
            if rule == "table":
                return copy_id_table.mint(values, field, k)
            return values.to_numpy(dtype=np.int64) + k * multiplier

        for column in _PERSON_ID_COLUMNS:
            members[column] = minted(members[column])
        units["benunit_id"] = minted(units["benunit_id"])
        copied["household_id"] = minted(copied["household_id"])
        copied[path_column] = k
        persons.append(members)
        benunits.append(units)
        households.append(copied)
    result = IdTables(
        person=pd.concat(persons, ignore_index=True),
        benunit=pd.concat(benunits, ignore_index=True),
        household=pd.concat(households, ignore_index=True),
    )
    for entity in ("person", "benunit", "household"):
        if len(np.unique(result.ids(entity))) != len(result.ids(entity)):
            raise AssertionError(f"Minted {entity} ids are not unique.")
    return result


def copy_chain(
    tables: IdTables, *, rule: str = "table", layers: int | None = None
) -> IdTables:
    """The six copy layers in the spine's order, each picking by source id.

    ``layers`` stops after that many layers (the SPI copy and the band donor
    count as two). The CGT layer gives one household eleven copies besides
    its root, so its largest ordinal has two digits, and the twin of every
    household whose source id is a multiple of six becomes two, four or
    eight arms.
    """

    def spi_copies(row) -> tuple[int, ...]:
        return (1,) if row.spi == 0 and row.source_household_id % 3 != 0 else ()

    def band_donors(row) -> tuple[int, ...]:
        return (2,) if row.spi == 0 and row.source_household_id % 3 == 0 else ()

    def high_value(row) -> tuple[int, ...]:
        return tuple(range(1, 6)) if row.source_household_id % 7 == 1 else ()

    def cgt_support(row) -> tuple[int, ...]:
        source = int(row.source_household_id)
        if source % 4 != 2:
            return ()
        return tuple(range(1, 12 if source == 2 else 2 + source % 3))

    def twins(row) -> tuple[int, ...]:
        return (1,)

    def arms(row) -> tuple[int, ...]:
        source = int(row.source_household_id)
        if row.twin != 1 or source % 6 != 0:
            return ()
        return tuple(range(1, 2 ** (1 + (source // 6) % 3)))

    steps = (
        # The SPI copy of base-channel households, then the income band donors
        # of the others, both in the spi_support_channel field.
        ("spi_support_channel", "spi", spi_copies),
        ("spi_support_channel", "spi", band_donors),
        ("high_value_support", "hv", high_value),
        ("cgt_support_split", "cgt", cgt_support),
        ("cgt_incidence_clone", "twin", twins),
        ("cgt_residential_split", "arm", arms),
    )
    for field, column, ordinals_of in steps[:layers]:
        ordinals = {
            position: ordinals_of(row)
            for position, row in enumerate(tables.household.itertuples(index=False))
        }
        tables = copy_households(
            tables,
            field=field,
            path_column=column,
            ordinals={p: values for p, values in ordinals.items() if values},
            rule=rule,
        )
    return tables


def _entity_tables(
    source: Frame | Mapping[str, pd.DataFrame],
) -> Mapping[str, pd.DataFrame]:
    if isinstance(source, Frame):
        return {entity: source.table(entity) for entity in source.entities}
    return source


def assert_copy_ids_follow_lineage(
    before: Frame | Mapping[str, pd.DataFrame],
    after: Frame | Mapping[str, pd.DataFrame],
    *,
    field: str,
    key_columns: Sequence[str],
    ordinal_column: str | None = None,
) -> int:
    """Every new household is its root's copy at ``ordinal x radix(field)``.

    The root is the household of ``before`` whose ``key_columns`` equal the
    copy's (the stages copy them unchanged), never one found from the copy's
    id. The copy's ordinal is ``ordinal_column`` on the new row, or 1 for a
    single-copy layer. Members pair up in person_id order, and each copy's
    benefit units are its root's at the same offset. Ids are read as Python
    integers, exact above 2**53. Returns the number of copies checked.
    """

    radix = copy_id_table.radix(field)
    keys = list(key_columns)
    before_tables, after_tables = _entity_tables(before), _entity_tables(after)
    root_household = before_tables["household"]
    root_keys = list(zip(*(root_household[c].tolist() for c in keys), strict=True))
    if len(set(root_keys)) != len(root_keys):
        raise AssertionError("The lineage key does not name one root household.")
    roots = dict(zip(root_keys, root_household["household_id"].tolist(), strict=True))
    household = after_tables["household"]
    new = household.loc[~household["household_id"].isin(root_household["household_id"])]
    ordinals = (
        [1] * len(new) if ordinal_column is None else new[ordinal_column].tolist()
    )
    person = after_tables["person"]
    root_person = before_tables["person"]
    benunit_ids = set(after_tables["benunit"]["benunit_id"].tolist())
    copy_keys = zip(*(new[c].tolist() for c in keys), strict=True)
    for copy, key, ordinal in zip(
        new["household_id"].tolist(), copy_keys, ordinals, strict=True
    ):
        root = roots[key]
        offset = int(ordinal) * radix
        assert copy == root + offset, (copy, root, ordinal)
        members = person.loc[person["person_household_id"] == copy]
        root_members = root_person.loc[root_person["person_household_id"] == root]
        assert len(members) == len(root_members)
        for column in _PERSON_ID_COLUMNS:
            assert sorted(members[column].tolist()) == sorted(
                value + offset for value in root_members[column].tolist()
            ), column
        assert {
            value + offset for value in root_members["person_benunit_id"].tolist()
        } <= benunit_ids
    return len(new)
