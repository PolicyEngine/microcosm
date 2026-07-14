"""Per-record nonzero shares for US frames: the evidence the parity gate runs on.

:func:`populace.build.gates.parity_gate` compares, for every layer the incumbent
populates, whether the candidate populates it too — share-based, so an
exported-but-all-zero column fails. That comparison needs one number per column:
the share of records carrying signal (a non-zero number, or a ``True`` boolean).
This module computes exactly that number, mirroring the canonical definition in
:func:`populace.build.plan._nonzero_share` so the pinned incumbent reference and
a live candidate frame are measured the same way.

Unlike :func:`populace.build.us_runtime.input_mass.us_input_mass_totals`, which
sums *weighted mass* per column, the parity share is an *unweighted record share*
(the fraction of an entity table's rows that carry signal). The parity contract
is about whether a layer is populated at all, not how much aggregate mass it
carries — a rare-but-real input is a populated layer even at a tiny share.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from populace.frame import Frame

__all__ = ["us_nonzero_shares", "nonzero_share"]


def nonzero_share(column: pd.Series) -> float:
    """Share of records carrying signal: non-zero numbers or ``True`` booleans.

    Non-numeric, non-boolean columns report the share of non-empty string
    values. This is the same definition the build plan records for produced
    columns (:func:`populace.build.plan._nonzero_share`), kept in lockstep so
    the parity reference and candidate measurements never diverge on convention.
    """
    if column.dtype == bool:
        return float(column.mean())
    if pd.api.types.is_numeric_dtype(column):
        values = pd.to_numeric(column, errors="coerce").fillna(0.0)
        return float((values != 0.0).mean())
    return float((column.astype(str).str.len() > 0).mean())


def us_nonzero_shares(
    frame: Frame,
    *,
    columns: Iterable[str] | None = None,
) -> dict[str, float]:
    """Per-record nonzero share for each value column of a US frame.

    Every non-structural column is measured under :func:`nonzero_share`;
    structural columns (entity ids and person membership ids) are skipped
    because they are reconstruction scaffolding, not populated layers.

    Args:
        frame: A US-schema frame.
        columns: Optional restriction — when given, only these columns are
            measured. Pass the engine's input-variable list (formula-owned
            excluded) so raw source-survey scratch columns that never persist,
            and formula-owned outputs the engine computes rather than stores,
            do not enter the parity universe.

    Returns:
        Column name -> share of records with a non-zero (or ``True``) value.
        The mapping is flat because the frame already enforces globally unique
        column names across entity tables.
    """
    schema = frame.schema
    structural = {schema.person_id_column}
    for group in schema.group_entities:
        structural.add(schema.id_column(group))
        structural.add(schema.membership_column(group))
    requested = None if columns is None else {str(name) for name in columns}

    shares: dict[str, float] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        for column in table.columns:
            if column in structural:
                continue
            if requested is not None and column not in requested:
                continue
            shares[column] = nonzero_share(table[column])

    # Frame weights are typed pipeline state, not ordinary table data. The US
    # adapter materializes the authoritative vector as ``household_weight``
    # when writing PolicyEngine H5, and the coverage gate mirrors that export
    # behavior; mirror it here too so the parity universe measures what is
    # persisted instead of reading the deliberately absent table column as an
    # all-zero layer (the Build M sparse failure: the reference's finished
    # export carries the column, the candidate keeps it as typed state).
    if (
        (requested is None or "household_weight" in requested)
        and "household_weight" not in shares
        and "household" in frame.weighted_entities
    ):
        shares["household_weight"] = nonzero_share(
            pd.Series(frame.weights_for("household").values)
        )
    return shares
