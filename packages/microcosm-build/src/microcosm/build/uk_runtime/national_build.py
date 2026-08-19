"""National UK build orchestration over the shared gate battery.

UK source stages run ``Frame -> Frame`` on the national carrier assembled by
:mod:`microcosm.build.uk_runtime.national_frame`; the staging H5 persists the
same person, benunit, and household tables PolicyEngine-UK reads, including
``household_weight`` as a real export column materialized from the frame's
typed weights. The local-geography clone remains a separate downstream build
product with its own carrier.

Gates run through :class:`microcosm.build.gate_battery.GateBatteryRun` over
the declared ``uk/gates.json`` spec: the preflight phase before the frame
loads, the terminal phase after the last stage and immediately before the
staging writer. Every declared entry appears in the persisted schema-4
report — evidence the build cannot supply is a named ``evidence_absent``
gap, blocking release candidates only — and the report is on disk before
any blocking decision raises.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

import microcosm.build.uk_runtime.national_frame as _national_frame
from microcosm.build.country_spec import load_country_spec
from microcosm.build.frame_sampling import (
    validate_sample_fraction,
    validate_sample_seed,
)
from microcosm.build.gate_battery import (
    BlockingMode,
    EvidenceContext,
    GateBatteryBlockedError,
    GateBatteryRun,
    GateBinding,
    GatePhaseReport,
)
from microcosm.build.gates import GateResult
from microcosm.build.plan import Stage as PlanStage
from microcosm.build.plan import StagePlan, StageRecord
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
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
)
from microcosm.build.uk_runtime.weighted_integrity import (
    UKInputMassReference,
    UKReviewedExclusion,
    exclusion_evaluation_date,
)
from microcosm.frame import (
    Frame,
    MassChangeRecord,
    WeightKind,
    engine_tables,
    put_frame_table,
    read_frame_table,
)

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
    """Deprecated shim for one named ``Frame -> Frame`` national transform.

    UK national builds now execute shared :class:`microcosm.build.plan.Stage`
    entries assembled into a :class:`microcosm.build.plan.StagePlan`. This
    wrapper remains for one release so existing callers can pass a stage name
    and transform; :func:`build_uk_national_dataset` converts it internally to
    a shared stage with empty consumes/produces and no donor.
    """

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
class UKNationalBuildResult:
    """A gated national staging artifact and its execution evidence."""

    frame: Frame
    provenance: UKStagingProvenance
    input_h5: Path
    staging_h5: Path
    stage_names: tuple[str, ...]
    #: The in-memory phase reports, declared order (preflight, terminal).
    phase_reports: tuple[GatePhaseReport, ...]
    #: The schema-4 payload the battery persisted at ``terminal_gate_path``
    #: (in compatibility-alias mode the file is last-written as the schema-1
    #: alias; this field always carries the full battery payload).
    gate_report: Mapping[str, object]
    terminal_gate_path: Path
    #: Shared stage evidence records, one per national build stage.
    stage_records: tuple[StageRecord, ...] = ()
    #: The #627 rung receipt; ``None`` on a full-scale (fraction 1.0) build.
    sampling_receipt: Mapping[str, object] | None = None

    @property
    def input_coverage(self) -> GateResult:
        """Backward-compatible projection of the coverage gate's verdict."""

        for report in self.phase_reports:
            for outcome in report.outcomes:
                if (
                    outcome.entry.id == "uk_release_input_coverage"
                    and outcome.result is not None
                ):
                    return outcome.result
        raise LookupError("uk_release_input_coverage did not evaluate in this build.")

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
    stages: Sequence[UKNationalStage | PlanStage] | StagePlan = (),
    coverage_engine: Any | None = None,
    input_mass_reference: UKInputMassReference | None = None,
    reviewed_input_mass_exclusions: (
        Mapping[str, Mapping[str, UKReviewedExclusion]] | None
    ) = None,
    reviewed_qrf_tail_exclusions: Mapping[str, UKReviewedExclusion] | None = None,
    reviewed_degenerate_exclusions: Mapping[str, UKReviewedExclusion] | None = None,
    terminal_gate_path: str | Path | None = None,
    input_coverage_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    run_config: Mapping[str, object] | None = None,
    sample_fraction: float = 1.0,
    sample_seed: int = UK_SAMPLE_SEED_DEFAULT,
    release_candidate: bool = False,
    now: date | None = None,
    gate_registry: Mapping[str, GateBinding] | None = None,
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

    ``release_candidate`` is the battery's second blocking axis: a candidate
    build treats every ``evidence_absent`` gap as blocking, a dev build
    records the gap and continues. A sampled rung is structurally
    non-releasable, so requesting both is refused. ``now`` is the shared
    exclusion-expiry clock (default: today, UTC), threaded to every
    exclusion-consuming gate so one report carries one evaluation date; it
    is resolved once when the battery is armed — before the stages — where
    the legacy aggregator resolved it after them, so a receipt expiring
    mid-build is judged by the date the build started.
    ``gate_registry`` overrides the binding registry (tests only).
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

    stage_plan = _coerce_stage_plan(stages)
    materialized_stages = stage_plan.stages
    _validate_stages(materialized_stages)
    # Invariant: no destructive step precedes argument validation. Every
    # configuration refusal sits above the sidecar unlinks and the battery,
    # so a misconfigured run can neither delete a previous report nor write
    # a new one (the #658 --degenerate-exclusions ordering bug, generalized).
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
    if release_candidate and sample_fraction != 1.0:
        raise ValueError(
            "a sampled rung build is structurally non-releasable (#627); "
            "release_candidate requires sample_fraction == 1.0."
        )
    if release_candidate and legacy_input_coverage_output:
        raise ValueError(
            "input_coverage_path is a compatibility alias whose schema-1 "
            "payload is last-written over the report path; a release "
            "candidate must keep its signed schema-4 report, so the two "
            "are mutually exclusive."
        )
    engine = (
        coverage_engine
        if coverage_engine is not None
        else PolicyEngineUKCoverageEngine()
    )
    # The clock and the battery construction validate their inputs (the
    # date's type; release identity, spec parameters, release_evidence
    # values), so they sit inside the no-destruction-before-validation
    # fence too: the unlinks come strictly last.
    evaluation_date = exclusion_evaluation_date(now)
    battery = GateBatteryRun(
        load_country_spec("uk").gates,
        release_id=release_id,
        report_path=diagnostic_path,
        release_candidate=release_candidate,
        registry=UK_GATE_REGISTRY if gate_registry is None else gate_registry,
        release_evidence={
            "calibration_diagnostics_sha256": calibration_diagnostics_sha256
        },
    )
    staging_path.unlink(missing_ok=True)
    diagnostic_path.unlink(missing_ok=True)
    # Mirrors the US cheap preflight: graph or reference drift blocks before
    # source stages — now with the refusal persisted as a schema-4 report.
    battery.run_phase(
        "preflight",
        EvidenceContext(
            artifacts={
                "coverage_engine": engine,
                "build_stage_names": tuple(stage.name for stage in materialized_stages),
            }
        ),
    )
    battery.enforce("preflight", mode=BlockingMode.BLOCKS_ARTIFACT)
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
        frame, stage_records = _validating_stage_plan(stage_plan).run(frame)
    else:
        frame, stage_records = _run_stages_checkpointed(
            materialized_stages,
            frame=frame,
            checkpoint_dir=Path(checkpoint_dir),
            run_config=run_config,
        )

    # Mirrors the US final-export placement: evaluate every declared gate in
    # one batch after all stages and immediately before the staging writer.
    artifacts: dict[str, object] = {
        "coverage_engine": engine,
        "exclusions_evaluated_on": evaluation_date,
        # The nonnegative gate derives its required columns from the stages
        # this build actually scheduled (same roster the preflight coverage
        # gate attests).
        "build_stage_names": tuple(stage.name for stage in materialized_stages),
        "rules_engine": engine,
    }
    brma_domain = _brma_enum_domain(engine)
    if brma_domain is None and "brma" in frame.table("household"):
        brma_domain = tuple(
            sorted(
                str(value)
                for value in frame.table("household")["brma"].dropna().unique()
            )
        )
    if brma_domain is not None:
        artifacts["brma_enum_domain"] = brma_domain
    fit_weight_records = _stage_fit_weight_records(materialized_stages)
    if fit_weight_records is not None:
        artifacts["fit_weight_records"] = fit_weight_records
    if input_mass_reference is not None:
        artifacts["input_mass_reference"] = input_mass_reference
    if reviewed_input_mass_exclusions is not None:
        artifacts["reviewed_input_mass_exclusions"] = reviewed_input_mass_exclusions
    if reviewed_qrf_tail_exclusions is not None:
        artifacts["reviewed_qrf_tail_exclusions"] = reviewed_qrf_tail_exclusions
    if reviewed_degenerate_exclusions is not None:
        artifacts["reviewed_degenerate_exclusions"] = reviewed_degenerate_exclusions
    terminal = battery.run_phase(
        "terminal", EvidenceContext(frame=frame, artifacts=artifacts)
    )
    coverage_outcome = next(
        outcome
        for outcome in terminal.outcomes
        if outcome.entry.id == "uk_release_input_coverage"
    )
    if legacy_input_coverage_output and coverage_outcome.result is None:
        raise RuntimeError(
            "uk_release_input_coverage did not evaluate; the schema-1 "
            "compatibility alias has no verdict to serialize."
        )
    try:
        battery.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
    except GateBatteryBlockedError as blocked:
        # The alias consumer reads the schema-1 shape at this exact path, in
        # the blocked case too — same last-write order as the legacy flow.
        # A failing alias write must not displace the typed block: the block
        # is the build's outcome, the write failure rides along as its cause.
        if legacy_input_coverage_output and coverage_outcome.result is not None:
            try:
                _write_input_coverage_diagnostic(
                    diagnostic_path, coverage_outcome.result
                )
            except Exception as write_error:  # noqa: BLE001 - keep the block typed
                raise blocked from write_error
        raise
    if legacy_input_coverage_output:
        _write_input_coverage_diagnostic(diagnostic_path, coverage_outcome.result)
    gate_report = battery.report_payload()
    attestation = gate_report["attestation"]
    signing_error = (
        attestation.get("signing_error") if isinstance(attestation, Mapping) else None
    )
    if signing_error is not None and sample_fraction == 1.0:
        # A rung build may proceed unsigned (its report honestly says
        # shippable: false, and a rung is structurally non-releasable); a
        # full-scale build keeps the legacy guarantee — no staging artifact
        # without an attested report. The unsigned report is already on disk.
        raise RuntimeError(
            "UK terminal gate report is unsigned and this is a full-scale "
            f"build; refusing to stage. {signing_error} The unsigned report "
            f"was written to {diagnostic_path}."
        )

    write_uk_national_frame(frame, staging_path)
    return UKNationalBuildResult(
        frame=frame,
        provenance=provenance,
        input_h5=input_path,
        staging_h5=staging_path,
        stage_names=tuple(stage.name for stage in materialized_stages),
        phase_reports=tuple(
            battery.phase_report(phase) for phase in battery.phases_evaluated
        ),
        gate_report=gate_report,
        terminal_gate_path=diagnostic_path,
        stage_records=stage_records,
        sampling_receipt=sampling_receipt,
    )


def _run_stages_checkpointed(
    stages: tuple[PlanStage, ...],
    *,
    frame: Frame,
    checkpoint_dir: Path,
    run_config: Mapping[str, object],
) -> tuple[Frame, tuple[StageRecord, ...]]:
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
    records: list[StageRecord] = []
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
            frame, stage_records = _resume_stage_record(stage, frame, loaded.frame)
            validate_uk_national_frame(frame)
            records.extend(stage_records)
            continue
        frame, stage_records = _validating_stage_plan(StagePlan((stage,))).run(frame)
        records.extend(stage_records)
        extra_metadata: dict[str, object] = {}
        hook = getattr(stage.transform, "checkpoint_metadata", None)
        if callable(hook):
            extra_metadata = dict(hook())
        runtime.complete(
            stage.name,
            frame,
            metadata=uk_stage_metadata(frame, extra=extra_metadata),
        )
    return frame, tuple(records)


def _coerce_stage_plan(
    stages: Sequence[UKNationalStage | PlanStage] | StagePlan,
) -> StagePlan:
    """Normalize legacy UK stages and shared stages to one StagePlan."""

    if isinstance(stages, StagePlan):
        return stages
    materialized = tuple(_coerce_stage(stage) for stage in stages)
    names: set[str] = set()
    for stage in materialized:
        if stage.name in names:
            raise ValueError(f"Duplicate UK national stage {stage.name!r}.")
        names.add(stage.name)
    return StagePlan(materialized)


def _coerce_stage(stage: UKNationalStage | PlanStage) -> PlanStage:
    if isinstance(stage, PlanStage):
        return stage
    if isinstance(stage, UKNationalStage):
        return PlanStage(name=stage.name, transform=stage.transform)
    raise TypeError(
        "UK national stages must be shared Stage or UKNationalStage instances, "
        f"got {type(stage).__name__}."
    )


def _validating_stage_plan(plan: StagePlan) -> StagePlan:
    """Return a plan whose stages validate the UK national frame after each run."""

    return StagePlan(_validating_stage(stage) for stage in plan.stages)


def _validating_stage(stage: PlanStage) -> PlanStage:
    def transform(frame: Frame) -> Frame:
        result = stage.transform(frame)
        if not isinstance(result, Frame):
            return result
        validate_uk_national_frame(result)
        return result

    return PlanStage(
        name=stage.name,
        transform=transform,
        produces=stage.produces,
        consumes=stage.consumes,
        donor=stage.donor,
    )


def _resume_stage_record(
    stage: PlanStage,
    previous: Frame,
    loaded: Frame,
) -> tuple[Frame, tuple[StageRecord, ...]]:
    plan = StagePlan(
        (
            PlanStage(
                name=stage.name,
                transform=lambda _frame: loaded,
                produces=stage.produces,
                consumes=stage.consumes,
                donor=stage.donor,
            ),
        )
    )
    return plan.run(previous)


def _validate_stages(stages: tuple[PlanStage, ...]) -> None:
    names: set[str] = set()
    for stage in stages:
        if not isinstance(stage, PlanStage):
            raise TypeError(
                "UK national stages must be shared Stage instances, "
                f"got {type(stage).__name__}."
            )
        if stage.name in names:
            raise ValueError(f"Duplicate UK national stage {stage.name!r}.")
        names.add(stage.name)


#: Stage names that perform production fits and therefore owe the terminal
#: weights audit their :class:`FitWeightRecord` evidence even when a swapped
#: or hollow transform stops exposing it.
_UK_FITTING_STAGE_NAMES = frozenset({"hmrc_spi_income", "was_wealth"})


def _stage_fit_weight_records(
    stages: tuple[PlanStage, ...],
) -> tuple[object, ...] | None:
    """The weights-audit evidence artifact, aggregated across fitting stages.

    A stage counts as fitting when its name is a declared fitting stage
    (HMRC SPI income, WAS wealth) or its transform exposes
    ``fit_weight_records``; each contributes records in stage order.
    ``None`` (no fitting stage scheduled) leaves the artifact unsupplied,
    so the audit is a named ``evidence_absent`` gap. A present fitting
    stage always supplies the artifact — records that are missing,
    unreadable, or empty coerce to ``()``, which the UK audit binding
    fails: an absent audit is not a passing audit.
    """

    fitting_stages = tuple(
        stage
        for stage in stages
        if stage.name in _UK_FITTING_STAGE_NAMES
        or hasattr(stage.transform, "fit_weight_records")
    )
    if not fitting_stages:
        return None
    collected: list[object] = []
    for stage in fitting_stages:
        try:
            records = tuple(stage.transform.fit_weight_records or ())
        except Exception:  # noqa: BLE001 - unreadable records coerce to ()
            # and fail the audit as missing evidence rather than crashing
            # the batch.
            return ()
        if not records:
            # A scheduled fitting stage with no records is missing evidence;
            # it must fail the audit, not be absorbed by another stage's
            # records.
            return ()
        collected.extend(records)
    return tuple(collected)


def _brma_enum_domain(engine: object) -> tuple[str, ...] | None:
    variable_getter = getattr(engine, "_variable", None)
    if not callable(variable_getter):
        return None
    try:
        variable = variable_getter("brma")
    except Exception:
        return None
    possible_values = getattr(variable, "possible_values", None)
    members = getattr(possible_values, "__members__", None)
    if isinstance(members, Mapping):
        return tuple(str(name) for name in members)
    if possible_values is None:
        return None
    return tuple(str(getattr(value, "name", value)) for value in possible_values)


def _write_input_coverage_diagnostic(path: Path, gate: GateResult) -> None:
    """Write the byte-compatible origin/main schema for the legacy alias.

    Atomic like every other writer on this surface: the alias last-writes
    over the gate-report path, and a crash mid-write must not leave
    truncated JSON where a consumer expects a report.
    """

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
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


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
