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
and appends the record itself.

The ``household_weight`` column itself is a materialized export contract,
not carrier state (``engine_tables`` regenerates it from the typed weights
at every export boundary). The H5 loader validates stored-vs-typed
agreement before consuming the column into the Frame's typed vector.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame import (
    EntitySchema,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
    engine_tables,
    put_frame_table,
    read_frame_table,
)

__all__ = [
    "UK_EXPORTED_WEIGHT_COLUMNS",
    "UK_HOUSEHOLD_WEIGHT_KIND_ATTR",
    "UK_MASS_LOG_ATTR",
    "UK_NATIONAL_H5_TABLES",
    "UK_NATIONAL_SCHEMA",
    "UK_TIME_PERIOD_METADATA_KEY",
    "UKNationalStage",
    "UKStagingProvenance",
    "load_uk_national_frame",
    "uk_household_weight_kind",
    "uk_national_frame",
    "uk_time_period",
    "validate_uk_national_frame",
    "write_uk_national_frame",
]

UK_NATIONAL_SCHEMA = EntitySchema(group_entities=("benunit", "household"))

UK_TIME_PERIOD_METADATA_KEY = "time_period"

UK_NATIONAL_H5_TABLES = ("person", "benunit", "household", "time_period")
UK_HOUSEHOLD_WEIGHT_KIND_ATTR = "populace_household_weight_kind"
UK_MASS_LOG_ATTR = "populace_mass_log_json"

# The persisted weight columns of the materialized export contract above:
# real columns on the tables, but plumbing rather than input mass. Consumers
# that total column mass (the input-mass parity surface, its reference
# emitter) must exclude them — the frame schema cannot, because the column
# is deliberately load-bearing here.
UK_EXPORTED_WEIGHT_COLUMNS = frozenset({"household_weight"})


def _assert_uk_benunit_nesting(person: pd.DataFrame) -> None:
    required = {"person_benunit_id", "person_household_id"}
    if not required.issubset(person.columns):
        # Belt-and-braces, not an escape hatch: at the validator seam the
        # Frame guarantees both membership columns exist, and at the
        # construction seam a missing column is refused a line later by the
        # kernel's own linkage validation, which names it.
        return
    placements = person[["person_benunit_id", "person_household_id"]].drop_duplicates()
    split = placements.loc[
        placements["person_benunit_id"].duplicated(keep=False),
        "person_benunit_id",
    ].drop_duplicates()
    if not split.empty:
        raise ValueError(
            "each benunit must belong to exactly one household; split "
            f"benunit id(s): {split.head(5).tolist()}."
        )


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


def _weight_kind_from_stored(value: object) -> WeightKind:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return WeightKind(str(value))
    except ValueError as exc:
        raise ValueError(f"Unknown stored UK household weight kind {value!r}.") from exc


def _read_weight_metadata(path: Path) -> tuple[object, object]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - UK H5 runtime dependency
        raise RuntimeError("h5py is required to read UK national metadata.") from exc
    with h5py.File(path, mode="r") as file:
        return (
            file.attrs.get(
                UK_HOUSEHOLD_WEIGHT_KIND_ATTR,
                WeightKind.DESIGN.value,
            ),
            file.attrs.get(UK_MASS_LOG_ATTR, "[]"),
        )


def _write_weight_metadata(
    path: Path,
    *,
    weight_kind: WeightKind,
    mass_log: tuple[MassChangeRecord, ...],
) -> None:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - UK H5 runtime dependency
        raise RuntimeError("h5py is required to write UK national metadata.") from exc
    with h5py.File(path, mode="r+") as file:
        file.attrs[UK_HOUSEHOLD_WEIGHT_KIND_ATTR] = weight_kind.value
        file.attrs[UK_MASS_LOG_ATTR] = json.dumps(
            [_mass_change_record_payload(record) for record in mass_log],
            sort_keys=True,
        )


def _mass_log_from_stored(value: object) -> tuple[MassChangeRecord, ...]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError("Stored UK microcosm mass log is not valid JSON.") from exc
    if not isinstance(payload, list):
        raise ValueError("Stored UK microcosm mass log must be a JSON list.")
    records: list[MassChangeRecord] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Stored UK microcosm mass-log entries must be objects.")
        try:
            records.append(
                MassChangeRecord(
                    entity=str(entry["entity"]),
                    old_total=float(entry["old_total"]),
                    new_total=float(entry["new_total"]),
                    declared_factor=(
                        None
                        if entry.get("declared_factor") is None
                        else float(entry["declared_factor"])
                    ),
                    reason=str(entry["reason"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Stored UK microcosm mass-log entry is malformed."
            ) from exc
    return tuple(records)


def _mass_change_record_payload(record: MassChangeRecord) -> dict[str, object]:
    return {
        "entity": record.entity,
        "old_total": record.old_total,
        "new_total": record.new_total,
        "declared_factor": record.declared_factor,
        "reason": record.reason,
    }


@dataclass(frozen=True)
class UKStagingProvenance:
    """Where a UK national frame's bytes came from, carried beside it.

    The same idiom as ``UKCertifiedCandidateIdentity``: provenance is its own
    frozen record next to the data, never a field smuggled into the carrier.
    """

    source_h5: Path
    fingerprint: _UKSourceFileFingerprint


@dataclass(frozen=True)
class UKNationalStage:
    """One named ``Frame -> Frame`` national transform.

    The June national driver is retired, but a few stage factories still expose
    this small callable container as their compatibility surface.
    """

    name: str
    transform: Callable[[Frame], Frame]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("UKNationalStage.name must be non-empty.")
        if not callable(self.transform):
            raise TypeError("UKNationalStage.transform must be callable.")

    def run(self, frame: Frame) -> Frame:
        result = self.transform(frame)
        if not isinstance(result, Frame):
            raise TypeError(
                f"UK national stage {self.name!r} must return a microcosm Frame, "
                f"got {type(result).__name__}."
            )
        return result


def _read_uk_national_tables(
    path: str | Path,
) -> tuple[dict[str, Any], _UKSourceFileFingerprint, Path]:
    """Read a compact UK single-year H5's payload with the race guard."""

    requested_path = Path(path).expanduser()
    if requested_path.suffix != ".h5":
        raise ValueError("UK national dataset path must end with '.h5'.")
    input_path = requested_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"UK national dataset not found: {input_path}.")

    fingerprint_before = _uk_source_file_fingerprint(input_path)
    stored_kind, stored_mass_log = _read_weight_metadata(input_path)
    with pd.HDFStore(input_path, mode="r") as store:
        keys = {key.lstrip("/") for key in store.keys()}
        missing = sorted(set(UK_NATIONAL_H5_TABLES) - keys)
        if missing:
            raise ValueError(f"UK national dataset is missing table(s): {missing}.")
        raw_period = store["time_period"]
        if len(raw_period) != 1:
            raise ValueError(
                "UK national dataset time_period must contain exactly one value."
            )
        payload = {
            "person": read_frame_table(store, "person"),
            "benunit": read_frame_table(store, "benunit"),
            "household": read_frame_table(store, "household"),
            "time_period": str(raw_period.iloc[0]),
            "household_weight_kind": _weight_kind_from_stored(stored_kind),
            "mass_log": _mass_log_from_stored(stored_mass_log),
        }
    fingerprint_after = _uk_source_file_fingerprint(input_path)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError(
            "UK national source H5 changed while it was being loaded; refusing "
            "to bind mixed or stale bytes to build stages."
        )
    return payload, fingerprint_after, input_path


def load_uk_national_frame(
    path: str | Path,
) -> tuple[Frame, UKStagingProvenance]:
    """Load a compact UK single-year H5 as a validated Frame plus provenance."""

    payload, fingerprint, input_path = _read_uk_national_tables(path)
    frame = uk_national_frame(
        person=payload["person"],
        benunit=payload["benunit"],
        household=payload["household"],
        time_period=payload["time_period"],
        weight_kind=payload["household_weight_kind"],
        mass_log=payload["mass_log"],
    )
    validate_uk_national_frame(frame)
    return frame, UKStagingProvenance(source_h5=input_path, fingerprint=fingerprint)


def uk_national_frame(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    time_period: int | str,
    weight_kind: WeightKind = WeightKind.DESIGN,
    household_weights: Sequence[float] | np.ndarray | pd.Series | None = None,
    mass_log: tuple[MassChangeRecord, ...] = (),
) -> Frame:
    """Assemble the UK national tables into a validated Frame.

    The typed household weights are built from ``household_weights`` when
    supplied, otherwise from a legacy input ``household_weight`` column. The
    stored carrier table never keeps that export column; materialization
    regenerates it from the typed vector. The Frame constructor enforces the
    structural invariants the shadow carrier never checked: group ids unique
    and sorted ascending, membership equality in both directions, global
    column uniqueness, weight health, and the UK invariant that a benunit's
    members share a household.
    """

    period = "" if time_period is None else str(time_period)
    if not period.strip():
        raise ValueError("UK national frame time_period must be a non-empty string.")
    if household_weights is None and "household_weight" not in household.columns:
        raise ValueError(
            "uk_national_frame requires household_weights or a household_weight "
            "input column to seed the typed household vector."
        )
    weight_values = (
        household_weights
        if household_weights is not None
        else household["household_weight"].to_numpy(dtype="float64")
    )
    weights = Weights(
        values=np.asarray(weight_values, dtype="float64"),
        kind=weight_kind,
    )
    _assert_uk_benunit_nesting(person)
    carrier_household = household.drop(columns=["household_weight"], errors="ignore")
    return Frame(
        tables={"person": person, "benunit": benunit, "household": carrier_household},
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


def _write_uk_single_year_tables(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    time_period: str,
    weight_kind: WeightKind,
    mass_log: tuple[MassChangeRecord, ...],
    path: Path,
) -> Path:
    """The one physical writer for every UK single-year H5."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.h5")
    try:
        with pd.HDFStore(temporary_path) as store:
            put_frame_table(
                store,
                "person",
                person,
                preferred_format="table",
                data_columns=True,
            )
            put_frame_table(
                store,
                "benunit",
                benunit,
                preferred_format="table",
                data_columns=True,
            )
            put_frame_table(
                store,
                "household",
                household,
                preferred_format="table",
                data_columns=True,
            )
            store.put(
                "time_period",
                pd.Series([time_period]),
                format="table",
                data_columns=True,
            )
        _write_weight_metadata(
            temporary_path, weight_kind=weight_kind, mass_log=mass_log
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def write_uk_national_frame(frame: Frame, path: str | Path) -> Path:
    """Atomically write a validated UK national Frame as a staging H5."""

    validate_uk_national_frame(frame)
    output_path = Path(path)
    if output_path.suffix != ".h5":
        raise ValueError("UK national staging path must end with '.h5'.")
    tables = engine_tables(frame, weighted_entities=("household",))
    return _write_uk_single_year_tables(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
        path=output_path,
    )


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
    the UK invariant that a benunit's members share a household, and
    agreement between the weight total and the latest household
    :class:`MassChangeRecord`.
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
    _assert_uk_benunit_nesting(frame.table("person"))
    uk_time_period(frame)
    weights = frame.weights_for("household")
    household = frame.table("household")
    reserved = sorted(UK_EXPORTED_WEIGHT_COLUMNS & set(household.columns))
    if reserved:
        raise ValueError(
            "UK national Frame carrier must not persist exported weight "
            f"column(s): {reserved}; use typed weights instead."
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
