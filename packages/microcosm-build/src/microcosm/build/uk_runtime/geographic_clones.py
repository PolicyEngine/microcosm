"""The UK geographic clones' column names, and a frame's clone count.

Each geographic clone of a household carries its index, 0 to ``K - 1``, on
the household, its benefit units and its people, under a per-entity name
(:func:`ladder_clone_index_column`); the written artifact keeps the legacy
schema's one shared name (:data:`ARTIFACT_CLONE_INDEX_COLUMN`). The UK copy
lineage reads the household's index as its ``geographic_support`` branch
(``household_lineage.UK_COPY_LINEAGE``), and :func:`geographic_clone_count`
reads ``K`` from it. The spine's ``geographic_expand`` stage writes the
indices and every household's source columns
(:data:`UK_GEOGRAPHIC_SOURCE_COLUMNS`) and mints the clones' ids in the
copy-id table's field :data:`GEOGRAPHIC_COPY_ID_FIELD`, and its
``geographic_identity`` stage each clone's geography key
(:data:`GEOGRAPHY_HOUSEHOLD_KEY_COLUMN`), which the dense line's
``atomic_area_support`` names ``IDENTITY_COLUMN``.

The two index names lived in ``rowwise_dataset`` and the count in
``cgt_support``, which re-export them. This module holds the names and the
count alone and imports nothing first-party, as ``annuity_trust_column``
holds its column's name. The stages that count by lineage read the
household's name through ``household_lineage``, and their nodes hash this
module with the counting modules (``graph_kernels._COUNTING_MODULES``). The
CGT support split, the support-clip stages and property components read
``K`` through the count, and the WAS stages, which count nothing by lineage,
hash this module as a helper. The full build's population and export
kernels, which read the names, hash it as well. So an edit here re-keys all
of them, without binding the dense line's ``rowwise_dataset`` or
``atomic_area_support`` into the spine (microcosm#931).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "ARTIFACT_CLONE_INDEX_COLUMN",
    "GEOGRAPHIC_COPY_ID_FIELD",
    "GEOGRAPHY_HOUSEHOLD_KEY_COLUMN",
    "SOURCE_HOUSEHOLD_KEY_COLUMN",
    "SOURCE_YEAR_COLUMN",
    "UK_GEOGRAPHIC_SOURCE_COLUMNS",
    "geographic_clone_count",
    "ladder_clone_index_column",
]

#: Clone-index column name in the written artifact (one name per table —
#: the legacy single-year schema, unchanged by the Frame carrier).
ARTIFACT_CLONE_INDEX_COLUMN = "clone_index"
#: The household's survey year, which ``geographic_expand`` writes on every
#: household.
SOURCE_YEAR_COLUMN = "source_year"
#: ``f"{source_year}:{source_household_id}"``, which ``geographic_expand``
#: writes on every household.
SOURCE_HOUSEHOLD_KEY_COLUMN = "source_household_key"
#: The households' source columns ``geographic_expand`` writes on every
#: household: the source id (each household's clone-0 id), its year and its
#: key. The source id is ``microcosm.frame.concepts.SOURCE_HOUSEHOLD_ID_COLUMN``,
#: spelled here so that this module imports nothing first-party; a test holds
#: the two spellings equal.
UK_GEOGRAPHIC_SOURCE_COLUMNS = (
    "source_household_id",
    SOURCE_YEAR_COLUMN,
    SOURCE_HOUSEHOLD_KEY_COLUMN,
)
#: The copy-id table's field the clones' ids are minted in: clone ``k`` of a
#: row takes the row's id at ordinal ``k`` there, and every later copy keeps
#: its clone's ordinal (``copy_id_table``; spelled here so that this module
#: imports nothing first-party).
GEOGRAPHIC_COPY_ID_FIELD = "geographic_support"
#: Each clone's geography key, which ``geographic_identity`` writes on every
#: household of the spine and the dense line's ``uk.full.identity`` on every
#: household of its pool (``atomic_area_support.IDENTITY_COLUMN`` is this
#: name).
GEOGRAPHY_HOUSEHOLD_KEY_COLUMN = "geography_household_key"


def ladder_clone_index_column(entity: str) -> str:
    """The in-memory, per-entity clone-index column on the ladder frame.

    Frame enforces globally unique column names across entity tables, so the
    artifact's shared ``clone_index`` name cannot appear on all three tables
    in memory; the writer renames these back at the export boundary.
    """

    return f"{entity}_{ARTIFACT_CLONE_INDEX_COLUMN}"


def geographic_clone_count(household: pd.DataFrame, *, stage: str | None = None) -> int:
    """The frame's geographic clone count ``K``: one more than its largest index.

    The clone-first spine's ``geographic_expand`` stage (microcosm#931) gives
    every household the index of its geographic clone in
    ``household_clone_index``, 0 to ``K - 1``; a household table without that
    column is one clone. Refuses a non-integer index column, a household
    without an index, a negative index and indices that are not exactly 0 to
    ``K - 1`` (an index below the largest that no household carries), since
    a count above the clones the frame holds would widen whatever the caller
    scales by it. Each refusal is a fixed sentence naming the column, the
    calling ``stage`` where one is given and, for a column that is not
    integer, its dtype: never a count, an id or a value.
    """

    column = ladder_clone_index_column("household")
    if column not in household.columns:
        return 1
    values = household[column]
    rule = (
        f"{stage or 'A spine'} counts K geographic clones as the {column!r} "
        "values 0 to K - 1 (microcosm#931)."
    )
    if pd.api.types.is_bool_dtype(values.dtype) or not (
        pd.api.types.is_integer_dtype(values.dtype)
    ):
        raise ValueError(
            f"{column!r} has dtype {values.dtype}, not an integer one. {rule}"
        )
    if bool(values.isna().any()):
        raise ValueError(f"A household carries no {column!r}. {rule}")
    indices = values.to_numpy(dtype=np.int64)
    if not indices.size:
        return 1
    if bool((indices < 0).any()):
        raise ValueError(f"A household carries a negative {column!r}. {rule}")
    clone_count = int(indices.max()) + 1
    if int(np.unique(indices).size) != clone_count:
        raise ValueError(
            f"An index from 0 to the largest is carried by no household. {rule}"
        )
    return clone_count
