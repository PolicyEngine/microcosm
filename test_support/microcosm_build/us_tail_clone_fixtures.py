"""Build capital-gains own-tail copies with the production clone operator.

A PUF-support base built without a selection source keeps every support copy:
the native ASEC record (clone index 0), its primary PUF-detail copy (clone
index 1) and, for the households the capital-gains tail transfer selects, an
own-tail copy of the PUF-detail household (clone index 2). The tail copy keeps
its source IDs and the ``puf_tax_detail`` channel, so a historical
(non-assembled) frame carries two PUF-role rows per selected source unit.

Stage tests use :func:`with_capital_gains_tail_copies` to reproduce that layout
through ``puf_capital_gains_tail._clone_and_transfer`` itself rather than a
hand-written approximation of it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

import microcosm.build.us_runtime.puf_capital_gains_tail as tail_module
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS,
    PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS,
)
from microcosm.frame import Frame

TAIL_CLONE_INDEX = 2


def _with_zero_capital_gains_columns(frame: Frame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS:
        if column not in tables["person"]:
            tables["person"][column] = 0.0
    for column in PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS:
        if column not in tables["tax_unit"]:
            tables["tax_unit"][column] = 0.0
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def with_capital_gains_tail_copies(
    frame: Frame,
    source_household_ids: Sequence[int],
    *,
    weight_share: float = 0.5,
) -> Frame:
    """Add one own-tail copy per selected source household.

    ``frame`` must be a PUF-support clone (clone indices 0 and 1). Each selected
    source household's primary PUF-detail copy is split: ``weight_share`` of
    its household weight moves to a new clone-index-2 copy whose first
    tax-unit member receives a nonzero capital-gains vector, exactly as the
    production transfer writes it.
    """

    frame = _with_zero_capital_gains_columns(frame)
    household = frame.table("household")
    clone_index = pd.to_numeric(
        household["household_support_clone_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    source = household["household_source_id"].to_numpy(dtype=np.int64)
    weights = frame.weights_for("household").values
    person = frame.table("person")
    rows: list[dict[str, object]] = []
    for position, source_household_id in enumerate(source_household_ids):
        match = np.flatnonzero((clone_index == 1) & (source == source_household_id))
        if len(match) != 1:
            raise ValueError(
                "Tail fixture needs exactly one primary PUF-detail copy of "
                f"source household {source_household_id}."
            )
        row = int(match[0])
        household_id = int(household["household_id"].iloc[row])
        members = person.loc[person["person_household_id"].eq(household_id)]
        tax_unit_id = int(members["person_tax_unit_id"].iloc[0])
        assignment: dict[str, object] = {
            "recipient_household_id": household_id,
            "recipient_tax_unit_id": tax_unit_id,
            "assigned_weight": float(weights[row]) * float(weight_share),
            "donor_source_id": 900_000 + position,
            tail_module._TAIL_SYNTHETIC_COLUMN: False,
            "filing_status_code": 1,
            tail_module._TAIL_AGI_BAND_INDEX_COLUMN: 0,
        }
        for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS:
            assignment[column] = 1_000_000.0 + position
        for column in PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS:
            assignment[column] = 0.0
        rows.append(assignment)
    transferred, _receipt = tail_module._clone_and_transfer(frame, pd.DataFrame(rows))
    return transferred
