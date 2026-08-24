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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    UK_HOUSEHOLD_WEIGHT_KIND_ATTR as UK_HOUSEHOLD_WEIGHT_KIND_ATTR,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_MASS_LOG_ATTR as UK_MASS_LOG_ATTR,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_NATIONAL_H5_TABLES as UK_NATIONAL_H5_TABLES,
)
from microcosm.build.uk_runtime.national_frame import (
    UKStagingProvenance as UKStagingProvenance,
)
from microcosm.build.uk_runtime.national_frame import (
    _mass_log_from_stored as _mass_log_from_stored,
)
from microcosm.build.uk_runtime.national_frame import (
    _read_uk_national_tables as _read_uk_national_tables,
)
from microcosm.build.uk_runtime.national_frame import (
    _read_weight_metadata as _read_weight_metadata,
)
from microcosm.build.uk_runtime.national_frame import (
    _weight_kind_from_stored as _weight_kind_from_stored,
)
from microcosm.build.uk_runtime.national_frame import (
    _write_uk_single_year_tables as _write_uk_single_year_tables,
)
from microcosm.build.uk_runtime.national_frame import (
    _write_weight_metadata as _write_weight_metadata,
)
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame as load_uk_national_frame,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.national_frame import (
    write_uk_national_frame as write_uk_national_frame,
)
from microcosm.build.uk_runtime.national_sampling import (
    UK_SAMPLE_SEED_DEFAULT,
    sample_uk_national_frame,
)
from microcosm.build.uk_runtime.parity_reference import EfrsParityReference
from microcosm.build.uk_runtime.release_input_coverage import (
    PolicyEngineUKCoverageEngine,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    UKInputMassReference,
    UKReviewedExclusion,
    exclusion_evaluation_date,
)
from microcosm.calibrate import TargetRegistry
from microcosm.frame import Frame

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
    ledger_target_registry: Mapping[int | str, TargetRegistry] | None = None,
    parity_reference: EfrsParityReference | None = None,
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
    preflight_artifacts: dict[str, object] = {
        "coverage_engine": engine,
        "build_stage_names": tuple(stage.name for stage in materialized_stages),
    }
    if ledger_target_registry is not None:
        preflight_artifacts["uk_ledger_compiled_registries"] = dict(
            ledger_target_registry
        )
    battery.run_phase(
        "preflight",
        EvidenceContext(artifacts=preflight_artifacts),
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
    student_loan_plan_domain = _engine_enum_domain(engine, "student_loan_plan")
    if student_loan_plan_domain is not None:
        artifacts["student_loan_plan_enum_domain"] = student_loan_plan_domain
    fit_weight_records = _stage_fit_weight_records(materialized_stages)
    if fit_weight_records is not None:
        artifacts["fit_weight_records"] = fit_weight_records
    calibration_evidence = _stage_calibration_evidence(materialized_stages)
    if calibration_evidence is not None:
        artifacts["national_calibration"] = calibration_evidence
        parity_evidence = _stage_parity_evidence(
            materialized_stages,
            frame=frame,
            parity_reference=parity_reference,
        )
        if parity_evidence is not None:
            artifacts["parity_evidence"] = parity_evidence
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


def _stage_calibration_evidence(
    stages: tuple[PlanStage, ...],
) -> Mapping[str, object] | None:
    for stage in stages:
        if stage.name == "national_calibration":
            manifest = getattr(stage.transform, "manifest", None)
            if not isinstance(manifest, Mapping):
                raise RuntimeError(
                    "national_calibration stage did not produce a manifest; "
                    "refusing to build calibration gate evidence."
                )
            return dict(manifest)
    return None


def _stage_parity_evidence(
    stages: tuple[PlanStage, ...],
    *,
    frame: Frame,
    parity_reference: EfrsParityReference | None,
) -> object | None:
    """Build the parity-trio evidence from independently sourced sides.

    The candidate side comes from the staged frame and the solve diagnostics;
    the reference side comes from the frozen parity instrument and the
    declared registry. The two sides must never alias each other — a copied
    reference would make the trio pass by construction.
    """

    for stage in stages:
        if stage.name != "national_calibration":
            continue
        if parity_reference is None:
            return None
        diagnostics = getattr(stage.transform, "diagnostics", None)
        if not isinstance(diagnostics, tuple):
            raise RuntimeError(
                "national_calibration stage did not produce target diagnostics; "
                "refusing to build parity evidence."
            )
        registry = getattr(stage.transform, "registry", None)
        specs = getattr(registry, "specs", None)
        if specs is None:
            raise RuntimeError(
                "national_calibration stage carries no compiled registry; "
                "refusing to build parity evidence."
            )
        target_relative_errors = {
            str(row["name"]): float(row["relative_error"]) for row in diagnostics
        }
        return SimpleNamespace(
            candidate_columns={
                f"{entity}.{column}"
                for entity in frame.entities
                for column in frame.table(entity).columns
            },
            reference_columns={
                f"{entity}.{name}"
                for name, entity in parity_reference.input_entities.items()
            },
            candidate_targets=set(target_relative_errors),
            # Solver diagnostics label rows as ``name@period``; the declared
            # side is compared at the same labeled grain so a target bound at
            # the wrong period cannot satisfy the surface.
            reference_targets={f"{spec.name}@{spec.period}" for spec in specs},
            target_relative_errors=target_relative_errors,
        )
    return None


def _brma_enum_domain(engine: object) -> tuple[str, ...] | None:
    return _engine_enum_domain(engine, "brma")


def _engine_enum_domain(engine: object, variable_name: str) -> tuple[str, ...] | None:
    variable_getter = getattr(engine, "_variable", None)
    if not callable(variable_getter):
        return None
    try:
        variable = variable_getter(variable_name)
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
