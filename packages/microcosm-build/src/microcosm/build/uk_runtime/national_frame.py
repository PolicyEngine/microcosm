"""The UK national build's Frame carrier: schema, construction, provenance.

DESIGN.md makes :class:`microcosm.frame.Frame` the atom of the stack —
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

The ``household_weight`` column itself is a materialized export contract,
not carrier state (``engine_tables`` regenerates it from the typed weights
at every export boundary). Dropping it from the in-build tables — the #612
increment-2 charter item — is assessed and deferred: this module's own
contract makes the column load-bearing (required at construction, asserted
equal to the typed vector at validation), the #611-owned gate modules and
the spi/rowwise reader surface still read it, and the drop would re-open
the #618 carrier review for no behavioural gain. It is sequenced behind the
#611 consumer half and the reader moves, not silently abandoned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.frame import (
    EntitySchema,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
    engine_tables,
)

__all__ = [
    "UK_EXPORTED_WEIGHT_COLUMNS",
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

# The persisted weight columns of the materialized export contract above:
# real columns on the tables, but plumbing rather than input mass. Consumers
# that total column mass (the input-mass parity surface, its reference
# emitter) must exclude them — the frame schema cannot, because the column
# is deliberately load-bearing here.
UK_EXPORTED_WEIGHT_COLUMNS = frozenset({"household_weight"})


@dataclass(frozen=True)
class _UKSourceFileFingerprint:
    """Cheap stable-file identity used to bind a prior hash to an H5 load.

    Scope-reduced by #612 increment 3: stat identity (device/inode/mtime)
    cannot survive a file copy or a machine move, so it no longer carries
    any *run* identity — that role belongs to content addressing
    (``uk_frame_content_identity`` in the descent fences, the checkpointed
    build's content-addressed run config). What stays here is exactly what
    stat identity is good at: the mid-read race guard (the file must not
    change while it is being loaded) and re-binding the certified-candidate
    hash to the same on-disk file within one process.
    """

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


@dataclass(frozen=True)
class _UKGateSurface:
    """The duck-attr evidence surface the legacy UK gate modules read.

    Mirrors the national build's gate-evidence adapter: the three entity
    tables plus the metadata gates consult (``time_period``,
    ``household_weight_kind``, ``mass_log``). Lives here — the Frame-carrier
    seam — as the single copy shared by the battery bindings and the legacy
    report evaluator, neither of which may import the other.
    """

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    time_period: str
    household_weight_kind: WeightKind
    mass_log: tuple[MassChangeRecord, ...]


def _uk_gate_surface(frame: Frame) -> _UKGateSurface:
    if not isinstance(frame, Frame):
        raise TypeError("UK gate evidence must be a Frame instance.")
    tables = engine_tables(frame)
    return _UKGateSurface(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        time_period=uk_time_period(frame),
        household_weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
    )


def validate_uk_national_frame(frame: Frame) -> None:
    """Structural revalidation plus the UK residue, at every seam.

    ``Frame.table`` returns the stored tables, not copies, so a stage that
    mutates them in place can break the invariants construction proved —
    the retired shadow-carrier validator rechecked structure at every seam,
    and this validator must be no weaker: it re-runs every constructor
    invariant via :meth:`Frame.revalidate`, then checks what only the UK
    contract knows — the exact export schema (person/benunit/household,
    household-only typed weights, no links), the time-period metadata,
    agreement between the persisted ``household_weight`` column and the
    typed vector, and agreement between the weight total and the latest
    household :class:`MassChangeRecord`.
    """

    if not isinstance(frame, Frame):
        raise TypeError("UK national stages must operate on Frame instances.")
    frame.revalidate()
    if tuple(frame.entities) != UK_NATIONAL_SCHEMA.entities:
        raise ValueError(
            f"UK national frame entities must be {UK_NATIONAL_SCHEMA.entities}; "
            f"got {tuple(frame.entities)}."
        )
    if tuple(frame.weighted_entities) != ("household",):
        raise ValueError(
            "UK national frames carry household typed weights only; got "
            f"explicit weights for {tuple(frame.weighted_entities)}. The "
            "staging artifact exports household_weight alone, and the loader "
            "rejects any other reserved weight column."
        )
    if frame.schema.links:
        raise ValueError(
            "UK national frames declare no links; the staging writer "
            "persists entity tables only and would silently drop them."
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
