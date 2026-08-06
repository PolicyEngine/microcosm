"""The UK national build's Frame carrier: schema, construction, provenance.

DESIGN.md makes :class:`populace.frame.Frame` the atom of the stack —
structure established once at assembly, every operator working on the
validated bundle. This module is the UK national build's construction seam
for that carrier (#612): the schema constant, the constructor helper (which
is where Frame's linkage validation runs), the metadata accessors, the UK
residue validation, and the provenance record that travels *beside* the
Frame. Provenance cannot live in the Frame: ``Frame.metadata`` freezes to
plain values (no ``Path``), ``__slots__`` blocks attributes, and Frame
operations hard-construct ``Frame(...)`` so a subclass sheds its type.

Mass accounting rule for stages: a table-reshaping stage hard-constructs a
new Frame via :func:`uk_national_frame` with an explicitly extended
``mass_log``; a weight-only update on unchanged tables goes through
:meth:`Frame.with_weights`, which enforces the forward-only kind transition
and appends the record itself — and must also refresh the persisted
``household_weight`` column, because :func:`validate_uk_national_frame`
holds the column equal to the typed vector (the staging H5 exports the
column, so a silent disagreement would ship the wrong weights).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from populace.frame import (
    EntitySchema,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
)

__all__ = [
    "UK_NATIONAL_SCHEMA",
    "UK_TIME_PERIOD_METADATA_KEY",
    "UKStagingProvenance",
    "uk_household_weight_kind",
    "uk_national_frame",
    "uk_time_period",
    "validate_uk_national_frame",
]

UK_NATIONAL_SCHEMA = EntitySchema(group_entities=("benunit", "household"))

UK_TIME_PERIOD_METADATA_KEY = "time_period"


@dataclass(frozen=True)
class _UKSourceFileFingerprint:
    """Cheap stable-file identity used to bind a prior hash to an H5 load."""

    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


def _uk_source_file_fingerprint(path: Path) -> _UKSourceFileFingerprint:
    stat = path.stat()
    return _UKSourceFileFingerprint(
        device=stat.st_dev,
        inode=stat.st_ino,
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


@dataclass(frozen=True)
class UKStagingProvenance:
    """Where a UK national frame's bytes came from, carried beside it.

    The same idiom as ``UKCertifiedCandidateIdentity``: provenance is its own
    frozen record next to the data, never a field smuggled into the carrier.
    """

    source_h5: Path
    fingerprint: _UKSourceFileFingerprint


def uk_national_frame(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    time_period: int | str,
    weight_kind: WeightKind = WeightKind.DESIGN,
    mass_log: tuple[MassChangeRecord, ...] = (),
) -> Frame:
    """Assemble the UK national tables into a validated Frame.

    The typed household weights are built from the ``household_weight``
    column, which stays on the table (the frame permits it because typed
    weights exist) so the staging H5 keeps its column order on export. The
    Frame constructor enforces the structural invariants the shadow carrier
    never checked: group ids unique and sorted ascending, membership equality
    in both directions, global column uniqueness, and weight health.
    """

    if "household_weight" not in household.columns:
        raise ValueError(
            "household must carry a household_weight column; the UK staging "
            "artifact exports it as a real column."
        )
    # Validate on a stripped copy but store the caller's exact value: the
    # carrier must never rewrite payload it merely transports.
    period = "" if time_period is None else str(time_period)
    if not period.strip():
        raise ValueError("UK national frame time_period must be a non-empty string.")
    weights = Weights(
        values=household["household_weight"].to_numpy(dtype="float64"),
        kind=weight_kind,
    )
    return Frame(
        tables={"person": person, "benunit": benunit, "household": household},
        schema=UK_NATIONAL_SCHEMA,
        weights={"household": weights},
        mass_log=mass_log,
        metadata={UK_TIME_PERIOD_METADATA_KEY: period},
    )


def uk_time_period(frame: Frame) -> str:
    """The frame's UK time period label (e.g. ``"2023"``), from metadata."""

    period = frame.metadata.get(UK_TIME_PERIOD_METADATA_KEY)
    if not isinstance(period, str) or not period.strip():
        raise ValueError(
            "UK national frame time_period metadata must be a non-empty string."
        )
    return period


def uk_household_weight_kind(frame: Frame) -> WeightKind:
    """The kind of the frame's explicit household weights."""

    return frame.weights_for("household").kind


def validate_uk_national_frame(frame: Frame) -> None:
    """The UK residue on top of Frame's own construction-time validation.

    Frame already enforced linkage, column uniqueness, and weight health when
    the frame was built; this checks what only the UK contract knows: the
    entity set, the time-period metadata, agreement between the persisted
    ``household_weight`` column and the typed vector, and agreement between
    the weight total and the latest household :class:`MassChangeRecord`.
    """

    if not isinstance(frame, Frame):
        raise TypeError("UK national stages must operate on Frame instances.")
    if tuple(frame.entities) != UK_NATIONAL_SCHEMA.entities:
        raise ValueError(
            f"UK national frame entities must be {UK_NATIONAL_SCHEMA.entities}; "
            f"got {tuple(frame.entities)}."
        )
    uk_time_period(frame)
    weights = frame.weights_for("household")
    household = frame.table("household")
    if "household_weight" not in household.columns:
        raise ValueError(
            "UK national frame household table must carry the exported "
            "household_weight column."
        )
    column = household["household_weight"].to_numpy(dtype="float64")
    if not np.array_equal(column, weights.values):
        raise ValueError(
            "household.household_weight column disagrees with the frame's "
            "typed weights; a stage that replaced weights must refresh the "
            "exported column."
        )
    household_records = [
        record for record in frame.mass_log if record.entity == "household"
    ]
    if household_records and not np.isclose(
        household_records[-1].new_total,
        float(weights.values.sum()),
        rtol=1e-9,
        atol=0.0,
    ):
        raise ValueError(
            "UK national frame household weight total disagrees with the "
            "latest household MassChangeRecord."
        )
