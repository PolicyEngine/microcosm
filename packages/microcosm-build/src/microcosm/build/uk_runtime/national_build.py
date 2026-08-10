"""National UK build orchestration with batched terminal release gates.

UK source stages run ``Frame -> Frame`` on the national carrier assembled by
:mod:`microcosm.build.uk_runtime.national_frame`; the staging H5 persists the
same person, benunit, and household tables PolicyEngine-UK reads, including
``household_weight`` as a real export column materialized from the frame's
typed weights. The local-geography clone remains a separate downstream build
product with its own carrier.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import microcosm.build.uk_runtime.national_frame as _national_frame
import microcosm.build.uk_runtime.release_input_coverage as _release_input_coverage
from microcosm.build.frame_sampling import (
    validate_sample_fraction,
    validate_sample_seed,
)
from microcosm.build.gates import GateReport, GateResult
from microcosm.build.uk_runtime.national_frame import (
    UKStagingProvenance,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.national_sampling import (
    UK_SAMPLE_SEED_DEFAULT,
    sample_uk_national_frame,
)
from microcosm.build.uk_runtime.release_input_coverage import (
    PolicyEngineUKCoverageEngine,
    assert_uk_release_input_coverage_build_stages,
    assert_uk_release_input_coverage_manifest_current,
)
from microcosm.build.uk_runtime.terminal_gates import (
    UKInputMassParityPolicy,
    UKInputMassReference,
    UKQRFTailConcentrationPolicy,
    UKReleaseParityEvidence,
    uk_terminal_gate_report,
    write_uk_terminal_gate_report,
)
from microcosm.frame import Frame, MassChangeRecord, WeightKind, engine_tables

# Retained as the existing library-test monkeypatch seam. Production terminal
# evaluation resolves the same function inside terminal_gates so its policy
# attestation can identify the builtin evaluator.
uk_release_input_coverage_gate = _release_input_coverage.uk_release_input_coverage_gate

__all__ = [
    "UKNationalBuildResult",
    "UKNationalStage",
    "UKStagingProvenance",
    "build_uk_national_dataset",
    "load_uk_national_frame",
    "uk_household_weight_kind",
    "uk_national_frame",
    "uk_time_period",
    "validate_uk_national_frame",
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
                f"UK national stage {self.name!r} must return a microcosm Frame, "
                f"got {type(result).__name__}."
            )
        return result


@dataclass(frozen=True)
class _UKGateEvidence:
    """The duck-attr evidence surface the UK gate battery consumes today.

    Exactly the shadow carrier's read surface — the three entity tables plus
    the weight-kind, period, and mass-log metadata — materialized from the
    frame. The gate modules stay deliberately duck-typed (#611 owns their
    Frame typing); until that lands, this adapter is the one place the legacy
    evidence shape survives, so a gate that reads ``household_weight_kind``
    or ``time_period`` sees the frame's real values rather than a fallback.
    """

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    time_period: str
    household_weight_kind: WeightKind
    mass_log: tuple[MassChangeRecord, ...]


def _uk_gate_evidence(frame: Frame) -> _UKGateEvidence:
    """Materialize the gate battery's evidence surface from the frame."""

    tables = engine_tables(frame)
    return _UKGateEvidence(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        time_period=uk_time_period(frame),
        household_weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
    )


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
    #: The #627 rung receipt; ``None`` on a full-scale (fraction 1.0) build.
    sampling_receipt: Mapping[str, object] | None = None

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


def write_uk_national_frame(frame: Frame, path: str | Path) -> Path:
    """Atomically write a validated UK national Frame as a staging H5.

    The engine-facing payload is materialized through the shared
    :func:`microcosm.frame.engine_tables`, so the typed weights are
    authoritative and the ``household_weight`` column is overwritten in
    place (preserving its position, and therefore the artifact's column
    order).
    """

    validate_uk_national_frame(frame)
    output_path = Path(path)
    if output_path.suffix != ".h5":
        raise ValueError("UK national staging path must end with '.h5'.")
    # The export contract is pinned, not inherited: household weights are the
    # one materialized vector, so a frame that somehow reached this point
    # with other typed weights cannot grow reserved columns the UK loader
    # rejects (validate_uk_national_frame refuses such frames anyway).
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
    reviewed_degenerate_exclusions: Mapping[str, str] | None = None,
    terminal_gate_path: str | Path | None = None,
    input_coverage_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    run_config: Mapping[str, object] | None = None,
    sample_fraction: float = 1.0,
    sample_seed: int = UK_SAMPLE_SEED_DEFAULT,
) -> UKNationalBuildResult:
    """Run ordered national stages, hard-gate the result, and stage an H5.

    Without ``checkpoint_dir`` the build is the destructive single-process
    monolith it always was. With ``checkpoint_dir`` each stage boundary
    persists a lossless Frame checkpoint through the outer stage runtime and
    completed stages are resumed from their checkpoints instead of re-run —
    which requires ``run_config``, the content-addressed identity of the run
    (input digest, seeds, source digests): resuming under a different
    configuration is refused by the runtime, and an unpinned resume is
    exactly the drift hazard checkpoints exist to prevent, so a checkpointed
    build without a ``run_config`` is refused here.

    ``sample_fraction`` below 1.0 is the #627 scale ladder: the loaded frame
    is sampled at clone-family grain (see
    :func:`~microcosm.build.uk_runtime.national_sampling.sample_uk_national_frame`)
    before provenance binding, so the certified-candidate fence attests the
    frame the stages actually consume. At 1.0 the sampler is never invoked —
    full-scale builds are structurally byte-invariant to it. A checkpointed
    sampled run must carry the fraction and seed inside ``run_config`` (the
    driver does); otherwise two rungs pointed at one checkpoint directory
    would silently resume across each other.
    """

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
    if checkpoint_dir is not None and run_config is None:
        raise ValueError(
            "a checkpointed UK national build requires run_config: the "
            "content-addressed run identity is what makes a resume safe."
        )
    validate_sample_fraction(sample_fraction, label="UK sample")
    validate_sample_seed(sample_seed, label="UK sample")
    if (
        checkpoint_dir is not None
        and sample_fraction != 1.0
        and "sampling" not in run_config
    ):
        raise ValueError(
            "a checkpointed rung build requires the sampling identity inside "
            "run_config: two rungs pointed at one checkpoint directory must "
            "refuse, never cross-resume."
        )
    frame, provenance = load_uk_national_frame(requested_input_path)
    sampling_receipt: Mapping[str, object] | None = None
    if sample_fraction != 1.0:
        # Sample before provenance binding: the fence attests the sampled
        # frame, and the stages never learn a rung existed.
        frame, sampling_receipt = sample_uk_national_frame(
            frame, fraction=sample_fraction, seed=sample_seed
        )
    # Stages whose fences bind the loaded bytes (the SPI stage's
    # certified-candidate check) receive the load provenance and the loaded
    # frame explicitly — provenance travels beside the frame, never inside
    # it, and binding records the loaded frame's content identity so the
    # fence can assert descent from this exact load. Bindings are
    # single-use; the stage consumes them.
    for stage in materialized_stages:
        binder = getattr(stage.transform, "bind_staging_provenance", None)
        if callable(binder):
            binder(provenance, frame)
    if checkpoint_dir is None:
        for stage in materialized_stages:
            frame = stage.run(frame)
            validate_uk_national_frame(frame)
    else:
        frame = _run_stages_checkpointed(
            materialized_stages,
            frame=frame,
            checkpoint_dir=Path(checkpoint_dir),
            run_config=run_config,
        )

    # Mirrors the US final-export placement: evaluate every evidenced gate in
    # one batch after all stages and immediately before the staging writer.
    fit_weight_records, require_fit_weight_records = _stage_fit_weight_records(
        materialized_stages
    )
    terminal_gates = uk_terminal_gate_report(
        _uk_gate_evidence(frame),
        engine,
        release_id=release_id,
        calibration_diagnostics_sha256=calibration_diagnostics_sha256,
        fit_weight_records=fit_weight_records,
        require_fit_weight_records=require_fit_weight_records,
        parity_evidence=parity_evidence,
        input_mass_reference=input_mass_reference,
        input_mass_policy=input_mass_policy,
        qrf_tail_policy=qrf_tail_policy,
        reviewed_degenerate_exclusions=reviewed_degenerate_exclusions,
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
        sampling_receipt=sampling_receipt,
    )


def _run_stages_checkpointed(
    stages: tuple[UKNationalStage, ...],
    *,
    frame: Frame,
    checkpoint_dir: Path,
    run_config: Mapping[str, object],
) -> Frame:
    """Run the national stages through the outer stage runtime.

    Each boundary persists a lossless Frame checkpoint (frame metadata rides
    the stage record, per ``uk_runtime.stage_checkpoints``); stages the run
    context already records as complete are resumed from their checkpoints —
    transforms that expose ``resume_from_checkpoint`` rehydrate their
    downstream evidence (the retained-leaves descent identities, the SPI
    fit-weight audit records) from the record instead of re-running.
    """

    from microcosm.build.outer_stage_runtime import (
        Stage as OuterStage,
    )
    from microcosm.build.outer_stage_runtime import (
        StagePipeline,
        StageRuntime,
    )
    from microcosm.build.uk_runtime.stage_checkpoints import (
        UK_FRAME_METADATA_KEY,
        load_uk_stage_checkpoint,
        uk_stage_metadata,
    )

    pipeline = StagePipeline(
        tuple(
            OuterStage(stage.name, f"UK national stage {stage.name}")
            for stage in stages
        )
    )
    runtime = StageRuntime(checkpoint_dir, pipeline, run_config=dict(run_config))
    completed = set(runtime.context.completed)
    for stage in stages:
        if stage.name in completed:
            loaded = load_uk_stage_checkpoint(runtime, stage.name)
            resume = getattr(stage.transform, "resume_from_checkpoint", None)
            if callable(resume):
                extra = {
                    key: value
                    for key, value in loaded.metadata.items()
                    if key != UK_FRAME_METADATA_KEY
                }
                resume(extra, loaded.frame)
            frame = loaded.frame
            continue
        frame = stage.run(frame)
        validate_uk_national_frame(frame)
        extra_metadata: dict[str, object] = {}
        hook = getattr(stage.transform, "checkpoint_metadata", None)
        if callable(hook):
            extra_metadata = dict(hook())
        runtime.complete(
            stage.name,
            frame,
            metadata=uk_stage_metadata(frame, extra=extra_metadata),
        )
    return frame


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
