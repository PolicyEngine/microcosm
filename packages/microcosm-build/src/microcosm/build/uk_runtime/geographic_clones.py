"""The UK geographic clone index columns' names, and a frame's clone count.

Each geographic clone of a household carries its index, 0 to ``K - 1``, on
the household, its benefit units and its people, under a per-entity name
(:func:`ladder_clone_index_column`); the written artifact keeps the legacy
schema's one shared name (:data:`ARTIFACT_CLONE_INDEX_COLUMN`). The UK copy
lineage reads the household's index as its ``geographic_support`` branch
(``household_lineage.UK_COPY_LINEAGE``), and :func:`geographic_clone_count`
reads ``K`` from it.

The two names lived in ``rowwise_dataset`` and the count in ``cgt_support``,
which re-export them. This module holds the three alone and imports nothing
first-party, as ``annuity_trust_column`` holds its column's name. The stages
that count by lineage read the household's name through ``household_lineage``,
and their nodes hash this module with the counting modules
(``graph_kernels._COUNTING_MODULES``); the CGT support split, which reads
``K`` through the count, hashes it too. The full build's population and export
kernels, which read the names, hash it as well. So an edit here re-keys all
of them, without binding the dense line's ``rowwise_dataset`` into the spine
(microcosm#931).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "ARTIFACT_CLONE_INDEX_COLUMN",
    "geographic_clone_count",
    "ladder_clone_index_column",
]

#: Clone-index column name in the written artifact (one name per table —
#: the legacy single-year schema, unchanged by the Frame carrier).
ARTIFACT_CLONE_INDEX_COLUMN = "clone_index"


def ladder_clone_index_column(entity: str) -> str:
    """The in-memory, per-entity clone-index column on the ladder frame.

    Frame enforces globally unique column names across entity tables, so the
    artifact's shared ``clone_index`` name cannot appear on all three tables
    in memory; the writer renames these back at the export boundary.
    """

    return f"{entity}_{ARTIFACT_CLONE_INDEX_COLUMN}"


def geographic_clone_count(household: pd.DataFrame) -> int:
    """The frame's geographic clone count ``K``: one more than its largest index.

    The clone-first spine (microcosm#931) gives every household the index of
    its geographic clone in ``household_clone_index``, 0 to ``K - 1``; a
    household table without that column, as on every national spine today,
    is one clone. Refuses a non-integer index column, a household without an
    index, a negative index and indices that are not exactly 0 to ``K - 1``
    (an index below the largest that no household carries). Each refusal
    gives a count or the column's dtype, never an id or a value.
    """

    column = ladder_clone_index_column("household")
    if column not in household.columns:
        return 1
    values = household[column]
    if pd.api.types.is_bool_dtype(values.dtype) or not (
        pd.api.types.is_integer_dtype(values.dtype)
    ):
        raise ValueError(
            "The CGT support split counts geographic clones on an integer "
            f"{column!r} column; it has dtype {values.dtype}."
        )
    rule = (
        f"the CGT support split counts K geographic clones as the {column!r} "
        "values 0 to K - 1 (microcosm#931)."
    )
    unindexed = int(values.isna().sum())
    if unindexed:
        raise ValueError(f"{unindexed} household(s) carry no {column!r}; {rule}")
    indices = values.to_numpy(dtype=np.int64)
    if not indices.size:
        return 1
    negative = int((indices < 0).sum())
    if negative:
        raise ValueError(f"{negative} household(s) carry a negative {column!r}; {rule}")
    clone_count = int(indices.max()) + 1
    absent = clone_count - int(np.unique(indices).size)
    if absent:
        raise ValueError(
            f"{absent} of the {clone_count} indices from 0 to the largest are "
            f"carried by no household; {rule}"
        )
    return clone_count
