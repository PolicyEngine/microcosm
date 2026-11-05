"""Each record's true source, read the same way in every country.

A build copies survey records: support copies of a household, the arms of a
split, and clones that each take an area of their own. Every copy keeps the
survey household it was made from in ``source_household_id``
(:data:`~microcosm.frame.concepts.SOURCE_HOUSEHOLD_ID_COLUMN`), its true
source, and each copying step marks its copies on the household table with a
marker column, an ordinal column, or both (:class:`CopyBranch`). A country
declares its copy lineage once (:class:`CopyLineage`) and passes it in; this
module names no country's columns.

:func:`record_sources` gives every record of any entity its true source,
through household membership: the key a disclosure floor counts distinct
sources by, so that a household's copies and clones count once
(microcosm#931). A frame without the source column is refused, since its
copies would count as sources of their own, unless the caller runs before the
first copying step and says so (``before_copies``): each household is then its
own source, and its id serves, as long as the frame carries no declared copy
column either.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.frame import EntitySchema
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN

__all__ = [
    "CopyBranch",
    "CopyLineage",
    "record_sources",
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
