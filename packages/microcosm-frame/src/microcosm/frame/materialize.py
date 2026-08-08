"""Materialize a frame's entity tables for a rules-engine dataset.

Rules engines consume flat per-entity tables with weights carried as a
``{entity}_weight`` column; the frame carries them as typed
:class:`~microcosm.frame.weights.Weights`. This module is the one shared
materializer for that boundary: the engine adapters delegate to it, and
country build runtimes that write engine-readable artifacts without an
adapter class call it directly.

The typed weights are authoritative. Any ``{entity}_weight`` column already
on a table is overwritten — never trusted — so a stale or leftover column
can never override calibrated weights on export. Overwriting assigns in
place, which preserves the column's existing position; a weight column
that is *added* (the entity table carried none) is appended last.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from microcosm.frame.bundle import Frame

__all__ = ["engine_tables"]

_WEIGHT_COLUMN_SUFFIX = "_weight"


def engine_tables(
    frame: Frame,
    *,
    weighted_entities: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Copy the frame's tables and materialize typed weights as columns.

    Args:
        frame: The frame to materialize. Every entity table is copied, so
            mutating the result never touches the frame.
        weighted_entities: Entities whose typed weights to materialize as
            ``{entity}_weight`` columns. Defaults to
            :attr:`Frame.weighted_entities` — every entity carrying its own
            explicit weights. Pass a subset to pin an export contract that
            materializes fewer (an entity outside the frame's weighted set
            raises, exactly as :meth:`Frame.weights_for` does; inherited
            weights are never materialized implicitly).

    Returns:
        Entity name -> copied table, in :attr:`Frame.entities` order, with
        each selected entity's ``{entity}_weight`` column overwritten from
        its typed weights.
    """

    tables = {name: frame.table(name).copy() for name in frame.entities}
    selected = (
        frame.weighted_entities
        if weighted_entities is None
        else tuple(weighted_entities)
    )
    for entity in selected:
        tables[entity][f"{entity}{_WEIGHT_COLUMN_SUFFIX}"] = frame.weights_for(
            entity
        ).values
    return tables
