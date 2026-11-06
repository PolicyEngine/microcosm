"""Each record's true source, read the same way in every country.

A build copies survey records: support copies of a household, the arms of a
split, and clones that each take an area of their own. Every copy keeps the
survey household it was made from in ``source_household_id``
(:data:`~microcosm.frame.concepts.SOURCE_HOUSEHOLD_ID_COLUMN`), its true
source, and each copying step marks its copies on the household table with a
marker column, an ordinal column, or both (:class:`CopyBranch`). A country
declares its copy lineage once (:class:`CopyLineage`) and passes it in; this
module names no country's columns.

- :func:`record_sources` gives every record of any entity its true source,
  through household membership: the key a disclosure floor counts distinct
  sources by, so that a household's copies and clones count once
  (microcosm#931). A frame without the source column is refused, since its
  copies would count as sources of their own, unless the caller runs before
  the first copying step and says so (``before_copies``): each household is
  then its own source, and its id serves, as long as the frame carries no
  declared copy column either.
- :func:`copy_paths` gives every household its ordered copy path, read from
  the declared columns, never inferred from ids, weights or values.
- :func:`replica_keys` and :func:`collapse_replicas` keep one training row per
  true source and path without the named branches, with the rows' weights
  summed, so a fit trained on the frame sees each survey record once however
  many geographic clones carry it. A frame without such replicas keeps its
  rows and weights value for value.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.frame import EntitySchema
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN

__all__ = [
    "CopyBranch",
    "CopyLineage",
    "collapse_replicas",
    "copy_paths",
    "record_sources",
    "replica_keys",
]


@dataclass(frozen=True)
class CopyBranch:
    """One copying step: a branch of every copy's path, in path order.

    ``index`` names the copy's ordinal column, zero on an original, and
    ``marker`` the column marking the step's copies: ``copy`` is its value on
    a copy and ``original`` its value on an original. A step declares either
    column or both; with both, a row is a copy exactly when its ordinal is
    nonzero, and a marker-only step's copy takes ordinal one.
    """

    name: str
    index: str | None = None
    marker: str | None = None
    copy: object = True
    original: object = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A copy branch needs a name.")
        if self.index is None and self.marker is None:
            raise ValueError(f"Copy branch {self.name!r} declares no column.")
        if self.copy == self.original:
            raise ValueError(
                f"Copy branch {self.name!r} marks copies and originals alike."
            )

    @property
    def columns(self) -> tuple[str, ...]:
        """The household columns the step writes."""
        return tuple(c for c in (self.index, self.marker) if c is not None)


@dataclass(frozen=True)
class CopyLineage:
    """How a country's copies record where they come from.

    ``entity`` is the group entity copies are made of (a household), whose
    table carries the true-source column ``source`` and every branch's
    columns; ``branches`` are the copying steps in path order.
    """

    entity: str
    branches: tuple[CopyBranch, ...]
    source: str = SOURCE_HOUSEHOLD_ID_COLUMN

    def __post_init__(self) -> None:
        names = [branch.name for branch in self.branches]
        if len(set(names)) != len(names):
            raise ValueError("Copy branches need distinct names.")
        columns = self.copy_columns
        if len(set(columns)) != len(columns) or self.source in columns:
            raise ValueError("Copy branches need distinct columns of their own.")

    @property
    def copy_columns(self) -> tuple[str, ...]:
        """Every column a copying step writes, in path order."""
        return tuple(c for branch in self.branches for c in branch.columns)


def record_sources(
    tables: Mapping[str, pd.DataFrame],
    schema: EntitySchema,
    entity: str,
    lineage: CopyLineage,
    *,
    before_copies: bool = False,
) -> np.ndarray:
    """Every ``entity`` record's true source, in the table's row order.

    A household reads its own source; a person reads its household's, through
    ``schema.membership_column``; any other group reads the household its
    members share, and a unit with no members or members in several
    households is refused. A null source, or a person whose household is
    absent, is refused: it would drop out of a distinct count, so a cell would
    read fewer sources than it rests on. Without the source column the frame
    is refused, unless ``before_copies`` says the caller runs before the first
    copying step and the frame carries none of the declared copy columns.
    """

    household = tables[lineage.entity]
    household_ids = household[schema.id_column(lineage.entity)]
    if lineage.source in household.columns:
        sources = household[lineage.source]
        if sources.isna().any():
            raise ValueError(
                f"{lineage.entity}.{lineage.source} is null on some rows: every "
                "record needs its true source."
            )
    else:
        copied = [c for c in lineage.copy_columns if c in household.columns]
        if copied:
            raise ValueError(
                f"The {lineage.entity} table carries copy lineage {copied} "
                f"without {lineage.source!r}, so its copies have no true source."
            )
        if not before_copies:
            raise ValueError(
                f"The {lineage.entity} table lacks {lineage.source!r}, the true "
                "source every record is counted by."
            )
        sources = household_ids
    if entity == lineage.entity:
        return sources.to_numpy()
    by_household = pd.Series(sources.to_numpy(), index=household_ids.to_numpy())
    person = tables[schema.person_entity]
    members = person[schema.membership_column(lineage.entity)]
    if not members.isin(by_household.index).all():
        raise ValueError(
            f"Every {schema.person_entity} needs a {lineage.entity} on the frame."
        )
    if entity == schema.person_entity:
        return members.map(by_household).to_numpy()
    units = pd.DataFrame(
        {
            "unit": person[schema.membership_column(entity)].to_numpy(),
            "household": members.to_numpy(),
        }
    ).drop_duplicates()
    if units["unit"].duplicated().any():
        raise ValueError(
            f"A {entity} with members in several {lineage.entity}s has no source."
        )
    unit_households = pd.Series(
        units["household"].to_numpy(), index=units["unit"].to_numpy()
    )
    unit_ids = tables[entity][schema.id_column(entity)]
    if not unit_ids.isin(unit_households.index).all():
        raise ValueError(f"A {entity} without members has no source.")
    return unit_ids.map(unit_households).map(by_household).to_numpy()


def copy_paths(
    household: pd.DataFrame,
    lineage: CopyLineage,
    *,
    require: Iterable[str] = (),
) -> tuple[np.ndarray, list[tuple[tuple[str, int], ...]]]:
    """Every household's true source and ordered copy path, in row order.

    A path lists, in branch order, ``(branch, ordinal)`` for each step the
    household is a copy in. A step whose columns are all absent has not run
    and adds nothing; a step with only some of its columns, or a step named in
    ``require`` that is absent, is refused. So are a null, negative or
    non-integer ordinal, a marker value the step does not declare, and a
    marker that disagrees with its ordinal: the path is read from the declared
    columns, never inferred from ids, weights or values.
    """

    required = set(require)
    unknown = required - {branch.name for branch in lineage.branches}
    if unknown:
        raise ValueError(f"No copy branch is named {sorted(unknown)}.")
    if lineage.source not in household.columns:
        raise ValueError(f"Copy paths need {lineage.entity}.{lineage.source}.")
    sources = household[lineage.source]
    if sources.isna().any():
        raise ValueError(f"{lineage.entity}.{lineage.source} is null on some rows.")
    steps: list[tuple[str, np.ndarray]] = []
    for branch in lineage.branches:
        present = [c for c in branch.columns if c in household.columns]
        if not present:
            if branch.name in required:
                raise ValueError(
                    f"The {lineage.entity} table lacks the {branch.name!r} "
                    f"step's column(s) {list(branch.columns)}."
                )
            continue
        if len(present) != len(branch.columns):
            raise ValueError(
                f"The {branch.name!r} step needs both {list(branch.columns)}; "
                f"the {lineage.entity} table carries only {present}."
            )
        steps.append((branch.name, _branch_ordinals(household, branch)))
    paths = [
        tuple((name, int(ordinals[row])) for name, ordinals in steps if ordinals[row])
        for row in range(len(household))
    ]
    return sources.to_numpy(), paths


def _branch_ordinals(household: pd.DataFrame, branch: CopyBranch) -> np.ndarray:
    ordinals = None
    if branch.index is not None:
        values = household[branch.index]
        if values.isna().any() or not pd.api.types.is_integer_dtype(values.dtype):
            raise ValueError(f"The {branch.name!r} ordinals must be integers.")
        ordinals = values.to_numpy(dtype=np.int64)
        if (ordinals < 0).any():
            raise ValueError(f"The {branch.name!r} step refuses negative ordinals.")
    if branch.marker is None:
        return ordinals
    marker = household[branch.marker]
    if marker.isna().any():
        raise ValueError(f"The {branch.name!r} marker is null on some rows.")
    if isinstance(branch.copy, bool) and not pd.api.types.is_bool_dtype(marker.dtype):
        raise ValueError(f"The {branch.name!r} marker must be boolean.")
    is_copy = marker.eq(branch.copy).to_numpy(dtype=bool)
    if not (is_copy | marker.eq(branch.original).to_numpy(dtype=bool)).all():
        raise ValueError(
            f"The {branch.name!r} marker holds a value other than "
            f"{branch.copy!r} and {branch.original!r}."
        )
    if ordinals is None:
        return is_copy.astype(np.int64)
    if not np.array_equal(is_copy, ordinals != 0):
        raise ValueError(f"The {branch.name!r} marker and ordinal disagree.")
    return ordinals


def replica_keys(
    tables: Mapping[str, pd.DataFrame],
    schema: EntitySchema,
    entity: str,
    lineage: CopyLineage,
    *,
    merge: Iterable[str],
) -> np.ndarray:
    """A key per ``entity`` record that only its replicas in ``merge`` share.

    A household's key is its true source and its copy path without the
    ``merge`` branches; a person's or unit's adds its rank among its
    household's members or units by id, which every copy keeps, since a copy
    remaps ids in order. Two records share a key exactly when one is the
    other's replica in the ``merge`` branches alone.
    """

    merged = set(merge)
    unknown = merged - {branch.name for branch in lineage.branches}
    if unknown:
        raise ValueError(f"No copy branch is named {sorted(unknown)}.")
    household = tables[lineage.entity]
    sources, paths = copy_paths(household, lineage)
    household_keys = pd.Series(
        [
            repr((source, tuple(step for step in path if step[0] not in merged)))
            for source, path in zip(sources.tolist(), paths, strict=True)
        ],
        index=household[schema.id_column(lineage.entity)].to_numpy(),
        dtype=object,
    )
    if entity == lineage.entity:
        return household_keys.to_numpy()
    person = tables[schema.person_entity]
    members = person[schema.membership_column(lineage.entity)]
    if not members.isin(household_keys.index).all():
        raise ValueError(
            f"Every {schema.person_entity} needs a {lineage.entity} on the frame."
        )
    if entity == schema.person_entity:
        ids = person[schema.person_id_column]
        owners = members
    else:
        units = pd.DataFrame(
            {
                "unit": person[schema.membership_column(entity)].to_numpy(),
                "household": members.to_numpy(),
            }
        ).drop_duplicates()
        if units["unit"].duplicated().any():
            raise ValueError(
                f"A {entity} with members in several {lineage.entity}s has no key."
            )
        by_unit = pd.Series(units["household"].to_numpy(), index=units["unit"])
        ids = tables[entity][schema.id_column(entity)]
        if not ids.isin(by_unit.index).all():
            raise ValueError(f"A {entity} without members has no key.")
        owners = ids.map(by_unit)
    rank = (
        pd.Series(ids.to_numpy())
        .groupby(owners.to_numpy(), sort=False)
        .rank(method="first")
        .astype(np.int64)
    )
    keys = owners.map(household_keys).to_numpy(dtype=object)
    return np.asarray(
        [f"{key}#{position}" for key, position in zip(keys, rank, strict=True)],
        dtype=object,
    )


def collapse_replicas(
    keys: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray,
    *,
    inputs: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first of each key's ``rows``, carrying the key's summed weight.

    ``rows`` marks the records a fit trains on and ``weights`` their weights,
    both over every record. The first selected record of each key, in row
    order, stays selected with the selected weights of its key summed
    exactly (:func:`math.fsum`, so the sum does not depend on row order); its
    replicas leave the selection. Weights off the selection are returned
    unchanged. When no two selected records share a key, the selection and
    weights are returned as they were, value for value.

    ``inputs`` holds, one row per record, what the fit reads: its predictors
    and targets, or the columns they derive from. Replicas then fold only
    where they agree on every one of those columns. Replicas that differ stay
    rows of their own: each distinct variant keeps its first record, with
    the summed weight of the replicas that share it, so the fit sees every
    variant at its share and nothing is dropped (microcosm#931). A missing
    value agrees only with a missing value.
    """

    keys = np.asarray(keys, dtype=object)
    selected = np.asarray(rows, dtype=bool)
    values = np.asarray(weights, dtype=np.float64)
    if not (len(keys) == len(selected) == len(values)):
        raise ValueError("Replica keys, rows and weights must align.")
    positions = np.flatnonzero(selected)
    if inputs is None:
        codes, uniques = pd.factorize(
            pd.Series(keys[positions], dtype=object), sort=False
        )
        groups = len(uniques)
    else:
        if not isinstance(inputs, pd.DataFrame) or len(inputs) != len(keys):
            raise ValueError("Replica inputs must be a frame with one row per record.")
        if not inputs.columns.is_unique:
            raise ValueError("Replica inputs must name each column once.")
        table = inputs.iloc[positions].reset_index(drop=True)
        table.columns = [f"input:{index}" for index in range(table.shape[1])]
        for column in table.columns:
            if isinstance(table[column].dtype, pd.CategoricalDtype):
                table[column] = table[column].astype(object)
        table.insert(0, "key", pd.Series(keys[positions], dtype=object))
        codes = (
            table.groupby(list(table.columns), sort=False, dropna=False, observed=True)
            .ngroup()
            .to_numpy()
        )
        groups = int(codes.max()) + 1 if len(codes) else 0
    if groups == len(positions):
        return selected, values
    first = np.full(groups, -1, dtype=np.int64)
    members: list[list[float]] = [[] for _ in range(groups)]
    for position, code in zip(positions.tolist(), codes.tolist(), strict=True):
        if first[code] < 0:
            first[code] = position
        members[code].append(float(values[position]))
    kept = np.zeros(len(selected), dtype=bool)
    kept[first] = True
    summed = values.copy()
    summed[first] = [math.fsum(group) for group in members]
    return kept, summed
