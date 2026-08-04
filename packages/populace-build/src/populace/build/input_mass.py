"""Weighted per-column input mass for populace frames.

These totals feed :func:`populace.build.gates.input_mass_parity_gate`:
comparing a derived artifact's persisted-input totals against its dense
parent (or a certified reference release) catches input bases that a sparse
selection or a rebuilt base pipeline silently zeroes — the failure mode of
populace issue #278, where the sparse default release carried ~$0 in
IRA-contribution, HSA, pension-contribution, and childcare inputs while
hitting its own calibration target surface.

The computation is schema-driven and country-agnostic: it reads only the
frame's entity layout (id and membership columns) and each entity's resolved
weights, so US and UK callers share one implementation instead of forking it.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from populace.frame import Frame

__all__ = ["input_mass_totals"]


def input_mass_totals(
    frame: Frame,
    *,
    columns: Iterable[str] | None = None,
) -> dict[str, float]:
    """Weighted totals of the frame's numeric and boolean value columns.

    Every non-structural numeric column is summed under the owning entity's
    effective weights (household weights broadcast through membership for
    entities without their own vector); boolean columns total their weighted
    ``True`` mass. String/enum columns and structural columns (entity ids and
    person membership ids) are skipped.

    Args:
        frame: A populace frame in any country schema.
        columns: Optional restriction — when given, only these columns are
            totalled. Pass the engine's input-variable list on raw build
            frames so source-survey scratch columns that never persist do not
            enter the comparison.

    Returns:
        Column name -> weighted total. The mapping is flat because the frame
        already enforces globally unique column names across entity tables.
    """

    schema = frame.schema
    structural = {schema.person_id_column}
    for group in schema.group_entities:
        structural.add(schema.id_column(group))
        structural.add(schema.membership_column(group))
    requested = None if columns is None else {str(name) for name in columns}

    totals: dict[str, float] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
        for column in table.columns:
            if column in structural:
                continue
            if requested is not None and column not in requested:
                continue
            values = table[column]
            if pd.api.types.is_bool_dtype(values):
                numeric = values.fillna(False).to_numpy(dtype=np.float64)
            elif pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce")
                numeric = numeric.fillna(0.0).to_numpy(dtype=np.float64)
            else:
                continue
            totals[column] = float(numeric @ weights)
    return totals
