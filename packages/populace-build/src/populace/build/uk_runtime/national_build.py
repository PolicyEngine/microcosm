"""National UK build orchestration with batched terminal release gates.

This module is deliberately table-oriented. UK source stages operate on the
same person, benunit, and household tables persisted by a PolicyEngine-UK
single-year H5, including ``household_weight`` as a real export column. The
local-geography clone remains a separate downstream build product.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import populace.build.uk_runtime.national_frame as _national_frame
import populace.build.uk_runtime.release_input_coverage as _release_input_coverage
from populace.build.gates import GateReport, GateResult
from populace.build.uk_runtime.national_frame import (
    UKStagingProvenance,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from populace.build.uk_runtime.release_input_coverage import (
    PolicyEngineUKCoverageEngine,
    assert_uk_release_input_coverage_build_stages,
    assert_uk_release_input_coverage_manifest_current,
)
from populace.build.uk_runtime.terminal_gates import (
    UKInputMassParityPolicy,
    UKInputMassReference,
    UKQRFTailConcentrationPolicy,
    UKReleaseParityEvidence,
    uk_terminal_gate_report,
    write_uk_terminal_gate_report,
)
from populace.frame import Frame, MassChangeRecord, WeightKind, engine_tables

# Retained as the existing library-test monkeypatch seam. Production terminal
# evaluation resolves the same function inside terminal_gates so its policy
# attestation can identify the builtin evaluator.
uk_release_input_coverage_gate = _release_input_coverage.uk_release_input_coverage_gate

__all__ = [
    "UKNationalBuildResult",
    "UKNationalDataset",
    "UKNationalStage",
    "UKStagingProvenance",
    "build_uk_national_dataset",
    "load_uk_national_dataset",
    "load_uk_national_frame",
    "uk_household_weight_kind",
    "uk_national_frame",
    "uk_time_period",
    "validate_uk_national_dataset",
    "validate_uk_national_frame",
    "write_uk_national_dataset",
    "write_uk_national_frame",
]

UK_NATIONAL_H5_TABLES = ("person", "benunit", "household", "time_period")
UK_HOUSEHOLD_WEIGHT_KIND_ATTR = "populace_household_weight_kind"
UK_MASS_LOG_ATTR = "populace_mass_log_json"


# The fingerprint pair moved to national_frame with the Frame carrier; the
# re-import keeps this module's existing consumers (hmrc_restoration binds
# certified candidates to it) on their current import path.
_UKSourceFileFingerprint = _national_frame._UKSourceFileFingerprint
_uk_source_file_fingerprint = _national_frame._uk_source_file_fingerprint


@dataclass(frozen=True)
class UKNationalDataset:
    """Explicit entity tables at one point in the national build pipeline."""

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    time_period: str
    household_weight_kind: WeightKind = WeightKind.DESIGN
    mass_log: tuple[MassChangeRecord, ...] = ()
    _source_h5: Path | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _source_file_fingerprint: _UKSourceFileFingerprint | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def source_h5(self) -> Path | None:
        """Resolved source path set by the H5 loader, never by table callers."""

        return self._source_h5

    @property
    def source_file_fingerprint(self) -> _UKSourceFileFingerprint | None:
        """Stable identity of the file bytes opened by the H5 loader."""

        return self._source_file_fingerprint

    def with_tables(
        self,
        *,
        person: pd.DataFrame | None = None,
        benunit: pd.DataFrame | None = None,
        household: pd.DataFrame | None = None,
        time_period: int | str | None = None,
        household_weight_kind: WeightKind | None = None,
        mass_log: tuple[MassChangeRecord, ...] | None = None,
    ) -> UKNationalDataset:
        """Return a dataset with selected tables replaced."""

        result = UKNationalDataset(
            person=self.person if person is None else person,
            benunit=self.benunit if benunit is None else benunit,
            household=self.household if household is None else household,
            time_period=(self.time_period if time_period is None else str(time_period)),
            household_weight_kind=(
                self.household_weight_kind
                if household_weight_kind is None
                else household_weight_kind
            ),
            mass_log=self.mass_log if mass_log is None else tuple(mass_log),
        )
        object.__setattr__(result, "_source_h5", self._source_h5)
        object.__setattr__(
            result,
            "_source_file_fingerprint",
            self._source_file_fingerprint,
        )
        return result


@dataclass(frozen=True)
class UKNationalStage:
    """One named, deterministic ``Frame -> Frame`` national transform."""

    name: str
    transform: Callable[[Frame], Frame]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("UKNationalStage.name must be non-empty.")
        if not callable(self.transform):
            raise TypeError("UKNationalStage.transform must be callable.")

    def run(self, frame: Frame) -> Frame:
        """Apply this stage and require an explicit Frame result."""

        result = self.transform(frame)
        if not isinstance(result, Frame):
            raise TypeError(
                f"UK national stage {self.name!r} must return a populace Frame, "
                f"got {type(result).__name__}."
            )
        return result


@dataclass(frozen=True)
class UKNationalBuildResult:
    """A gated national staging artifact and its execution evidence."""

    frame: Frame
    provenance: UKStagingProvenance
    input_h5: Path
    staging_h5: Path
    stage_names: tuple[str, ...]
    terminal_gates: GateReport
    terminal_gate_path: Path

    @property
    def input_coverage(self) -> GateResult:
        """Backward-compatible projection of the consolidated gate report."""

        return next(
            result
            for result in self.terminal_gates.results
            if result.name == "uk_release_input_coverage"
        )

    @property
    def input_coverage_path(self) -> Path:
        """Backward-compatible alias for :attr:`terminal_gate_path`."""

        return self.terminal_gate_path


def _read_uk_national_tables(
    path: str | Path,
) -> tuple[dict[str, Any], _UKSourceFileFingerprint, Path]:
    """Read a compact UK single-year H5's payload with the race guard.

    Shared body of both loaders: suffix check, symlink resolution, and the
    fingerprint-before/after guard binding the returned tables to one stable
    set of bytes.
    """

    requested_path = Path(path).expanduser()
    if requested_path.suffix != ".h5":
        raise ValueError("UK national dataset path must end with '.h5'.")
    # Hugging Face cache entries retain the requested ``.h5`` name as a
    # symlink whose content-addressed blob target has no suffix. Validate the
    # caller-facing artifact name before resolving it, while binding all
    # provenance and stable-byte checks to the actual opened file.
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
            "person": store["person"],
            "benunit": store["benunit"],
            "household": store["household"],
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


def load_uk_national_dataset(path: str | Path) -> UKNationalDataset:
    """Load and validate a compact UK single-year H5."""

    payload, fingerprint, input_path = _read_uk_national_tables(path)
    dataset = UKNationalDataset(
        person=payload["person"],
        benunit=payload["benunit"],
        household=payload["household"],
        time_period=payload["time_period"],
        household_weight_kind=payload["household_weight_kind"],
        mass_log=payload["mass_log"],
    )
    object.__setattr__(dataset, "_source_h5", input_path)
    object.__setattr__(dataset, "_source_file_fingerprint", fingerprint)
    validate_uk_national_dataset(dataset)
    return dataset


def load_uk_national_frame(
    path: str | Path,
) -> tuple[Frame, UKStagingProvenance]:
    """Load a compact UK single-year H5 as a validated Frame plus provenance.

    Frame construction is where the structural invariants are enforced
    (linkage in both directions, sorted group ids, column uniqueness, weight
    health); :func:`validate_uk_national_frame` adds the UK residue. The
    provenance record travels beside the frame — it is the same source-path
    and fingerprint identity the shadow carrier smuggled in private fields.
    """

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


def validate_uk_national_dataset(dataset: UKNationalDataset) -> None:
    """Validate UK entity IDs, memberships, weights, and period metadata."""

    if not isinstance(dataset, UKNationalDataset):
        raise TypeError(
            "UK national stages must operate on UKNationalDataset instances."
        )
    for name in ("person", "benunit", "household"):
        if not isinstance(getattr(dataset, name), pd.DataFrame):
            raise TypeError(f"UK national {name} table must be a pandas DataFrame.")
    if not isinstance(dataset.time_period, str) or not dataset.time_period.strip():
        raise ValueError("UK national dataset time_period must be a non-empty string.")
    if not isinstance(dataset.household_weight_kind, WeightKind):
        raise TypeError(
            "UK national dataset household_weight_kind must be a WeightKind."
        )
    if not isinstance(dataset.mass_log, tuple) or any(
        not isinstance(record, MassChangeRecord) for record in dataset.mass_log
    ):
        raise TypeError(
            "UK national dataset mass_log must be a tuple of MassChangeRecord."
        )

    _require_columns(
        dataset.person,
        ("person_id", "person_household_id", "person_benunit_id"),
        label="person",
    )
    _require_columns(dataset.benunit, ("benunit_id",), label="benunit")
    _require_columns(
        dataset.household,
        ("household_id", "household_weight"),
        label="household",
    )
    _require_unique(dataset.person, "person_id", label="person")
    _require_unique(dataset.benunit, "benunit_id", label="benunit")
    _require_unique(dataset.household, "household_id", label="household")

    missing_households = sorted(
        set(dataset.person["person_household_id"])
        - set(dataset.household["household_id"])
    )
    if missing_households:
        raise ValueError(
            "person.person_household_id contains value(s) absent from household: "
            f"{missing_households[:5]}."
        )
    missing_benunits = sorted(
        set(dataset.person["person_benunit_id"]) - set(dataset.benunit["benunit_id"])
    )
    if missing_benunits:
        raise ValueError(
            "person.person_benunit_id contains value(s) absent from benunit: "
            f"{missing_benunits[:5]}."
        )

    weights = pd.to_numeric(dataset.household["household_weight"], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("household.household_weight must contain finite numbers.")
    if (weights < 0).any():
        raise ValueError("household.household_weight must be non-negative.")
    if not (weights > 0).any():
        raise ValueError(
            "household.household_weight must retain at least one positive value."
        )
    household_records = [
        record for record in dataset.mass_log if record.entity == "household"
    ]
    if household_records and not np.isclose(
        household_records[-1].new_total,
        float(weights.sum()),
        rtol=1e-9,
        atol=0.0,
    ):
        raise ValueError(
            "UK national dataset household_weight total disagrees with the "
            "latest household MassChangeRecord."
        )


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
    """The one physical writer for every UK single-year H5.

    Tables and the weight-kind/mass-log attrs must land together: writing
    them into a temporary file and renaming keeps a metadata failure from
    leaving a complete-looking attr-less H5 that would silently default to
    DESIGN semantics on the next read.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.h5")
    try:
        with pd.HDFStore(temporary_path) as store:
            store.put("person", person, format="table", data_columns=True)
            store.put("benunit", benunit, format="table", data_columns=True)
            store.put("household", household, format="table", data_columns=True)
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


def write_uk_national_dataset(
    dataset: UKNationalDataset,
    path: str | Path,
) -> Path:
    """Atomically write a validated UK national single-year staging H5."""

    validate_uk_national_dataset(dataset)
    output_path = Path(path)
    if output_path.suffix != ".h5":
        raise ValueError("UK national staging path must end with '.h5'.")
    return _write_uk_single_year_tables(
        person=dataset.person,
        benunit=dataset.benunit,
        household=dataset.household,
        time_period=dataset.time_period,
        weight_kind=dataset.household_weight_kind,
        mass_log=dataset.mass_log,
        path=output_path,
    )


def write_uk_national_frame(frame: Frame, path: str | Path) -> Path:
    """Atomically write a validated UK national Frame as a staging H5.

    The engine-facing payload is materialized through the shared
    :func:`populace.frame.engine_tables`, so the typed weights are
    authoritative and the ``household_weight`` column is overwritten in
    place (preserving its position, and therefore the artifact's column
    order).
    """

    validate_uk_national_frame(frame)
    output_path = Path(path)
    if output_path.suffix != ".h5":
        raise ValueError("UK national staging path must end with '.h5'.")
    tables = engine_tables(frame)
    return _write_uk_single_year_tables(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
        path=output_path,
    )


def build_uk_national_dataset(
    *,
    input_h5: str | Path,
    staging_h5: str | Path,
    release_id: str,
    calibration_diagnostics_sha256: str,
    stages: Sequence[UKNationalStage] = (),
    coverage_engine: Any | None = None,
    parity_evidence: UKReleaseParityEvidence | None = None,
    input_mass_reference: UKInputMassReference | None = None,
    input_mass_policy: UKInputMassParityPolicy | None = None,
    qrf_tail_policy: UKQRFTailConcentrationPolicy | None = None,
    terminal_gate_path: str | Path | None = None,
    input_coverage_path: str | Path | None = None,
) -> UKNationalBuildResult:
    """Run ordered national stages, hard-gate the result, and stage an H5."""

    requested_input_path = Path(input_h5).expanduser()
    input_path = requested_input_path.resolve()
    staging_path = Path(staging_h5).resolve()
    if input_path == staging_path:
        raise ValueError("input_h5 and staging_h5 must differ.")
    if staging_path.suffix != ".h5":
        raise ValueError("UK national staging path must end with '.h5'.")

    if terminal_gate_path is not None and input_coverage_path is not None:
        raise ValueError(
            "terminal_gate_path and input_coverage_path are mutually exclusive; "
            "input_coverage_path is a compatibility alias."
        )
    legacy_input_coverage_output = input_coverage_path is not None
    requested_gate_path = (
        terminal_gate_path
        if terminal_gate_path is not None
        else (
            input_coverage_path
            if input_coverage_path is not None
            else staging_path.with_suffix(".terminal_gates.json")
        )
    )
    diagnostic_path = Path(requested_gate_path).resolve()
    if diagnostic_path in {input_path, staging_path}:
        raise ValueError(
            "terminal_gate_path must differ from the input and staging H5 paths."
        )

    materialized_stages = tuple(stages)
    _validate_stages(materialized_stages)
    staging_path.unlink(missing_ok=True)
    diagnostic_path.unlink(missing_ok=True)

    engine = (
        coverage_engine
        if coverage_engine is not None
        else PolicyEngineUKCoverageEngine()
    )
    # Mirrors the US cheap preflight: graph or reference drift aborts before
    # source stages and, once added, before national target-registry compilation.
    assert_uk_release_input_coverage_manifest_current(engine=engine)
    assert_uk_release_input_coverage_build_stages(
        tuple(stage.name for stage in materialized_stages)
    )
    frame, provenance = load_uk_national_frame(requested_input_path)
    # Stages whose fences bind the loaded bytes (the SPI stage's
    # certified-candidate check) receive the load provenance explicitly —
    # it travels beside the frame, never inside it.
    for stage in materialized_stages:
        binder = getattr(stage.transform, "bind_staging_provenance", None)
        if callable(binder):
            binder(provenance)
    for stage in materialized_stages:
        frame = stage.run(frame)
        validate_uk_national_frame(frame)

    # Mirrors the US final-export placement: evaluate every evidenced gate in
    # one batch after all stages and immediately before the staging writer.
    fit_weight_records, require_fit_weight_records = _stage_fit_weight_records(
        materialized_stages
    )
    terminal_gates = uk_terminal_gate_report(
        engine_tables(frame),
        engine,
        release_id=release_id,
        calibration_diagnostics_sha256=calibration_diagnostics_sha256,
        fit_weight_records=fit_weight_records,
        require_fit_weight_records=require_fit_weight_records,
        parity_evidence=parity_evidence,
        input_mass_reference=input_mass_reference,
        input_mass_policy=input_mass_policy,
        qrf_tail_policy=qrf_tail_policy,
    )
    write_uk_terminal_gate_report(terminal_gates, diagnostic_path)
    if legacy_input_coverage_output:
        input_coverage = next(
            gate
            for gate in terminal_gates.results
            if gate.name == "uk_release_input_coverage"
        )
        _write_input_coverage_diagnostic(diagnostic_path, input_coverage)
    if not terminal_gates.passed:
        raise RuntimeError(
            "Release gates failed: " + "; ".join(terminal_gates.failures)
        )

    write_uk_national_frame(frame, staging_path)
    return UKNationalBuildResult(
        frame=frame,
        provenance=provenance,
        input_h5=input_path,
        staging_h5=staging_path,
        stage_names=tuple(stage.name for stage in materialized_stages),
        terminal_gates=terminal_gates,
        terminal_gate_path=diagnostic_path,
    )


def _validate_stages(stages: tuple[UKNationalStage, ...]) -> None:
    names: set[str] = set()
    for stage in stages:
        if not isinstance(stage, UKNationalStage):
            raise TypeError(
                "UK national stages must be UKNationalStage instances, "
                f"got {type(stage).__name__}."
            )
        if stage.name in names:
            raise ValueError(f"Duplicate UK national stage {stage.name!r}.")
        names.add(stage.name)


def _stage_fit_weight_records(
    stages: tuple[UKNationalStage, ...],
) -> tuple[tuple[object, ...] | None, bool]:
    """Return real fit evidence, requiring it only when HMRC executed."""

    hmrc_stage = next(
        (stage for stage in stages if stage.name == "hmrc_spi_income"),
        None,
    )
    if hmrc_stage is None:
        return (None, False)
    try:
        records = getattr(hmrc_stage.transform, "fit_weight_records", None)
        return (() if records is None else tuple(records), True)
    except Exception:  # noqa: BLE001 - the terminal report must name the failure
        return ((), True)


def _write_input_coverage_diagnostic(path: Path, gate: GateResult) -> None:
    """Write the byte-compatible origin/main schema for the legacy alias."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "enforced": True,
        "input_coverage": {
            "passed": gate.passed,
            "failures": list(gate.failures),
            "details": dict(gate.details),
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} table is missing column(s): {missing}.")


def _require_unique(frame: pd.DataFrame, column: str, *, label: str) -> None:
    if frame[column].isna().any():
        raise ValueError(f"{label}.{column} contains missing values.")
    if frame[column].duplicated().any():
        duplicates = frame.loc[frame[column].duplicated(), column].unique()
        raise ValueError(
            f"{label}.{column} must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
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
        raise ValueError("Stored UK populace mass log is not valid JSON.") from exc
    if not isinstance(payload, list):
        raise ValueError("Stored UK populace mass log must be a JSON list.")
    records: list[MassChangeRecord] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Stored UK populace mass-log entries must be objects.")
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
            raise ValueError("Stored UK populace mass-log entry is malformed.") from exc
    return tuple(records)


def _mass_change_record_payload(record: MassChangeRecord) -> dict[str, object]:
    return {
        "entity": record.entity,
        "old_total": record.old_total,
        "new_total": record.new_total,
        "declared_factor": record.declared_factor,
        "reason": record.reason,
    }
