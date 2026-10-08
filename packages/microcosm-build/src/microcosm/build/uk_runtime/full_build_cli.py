"""Execute the single UK full build; all geographies are selected by default.

Numerical operations and verdicts belong to the composed graph. This module
resolves requests, executes graph endpoints and atomically materializes their
stored artifacts. Publication and signing remain explicit external services.

``--release-role`` declares which UK dataset line the run builds and is
required (microcosm#823): ``dense`` is the K-clone joint national + local
surface under the local doctrine; ``national`` is the certified national
line, the bound spine checkpoint solved as bound under the national doctrine
(:mod:`microcosm.build.uk_runtime.graph_national`), with the seam-shaped
evidence (``build_record.json``, the signed calibration-seam gate report,
the diagnostics, the frozen registries, the schema-4 national manifest;
:mod:`microcosm.build.uk_runtime.national_role`) materialised from the
graph's stored artifacts. Both lines are built here through the graph. The
role supplies every unset solve default and refuses the other role's flags
through :mod:`microcosm.build.uk_runtime.rowwise_cli`.
``tools/build_uk_rowwise_candidate.py`` and ``tools/build_uk_full.py`` are
stubs over :func:`main`, so every UK line is built by this one command.

A non-dry dense run is wrapped in the rowwise tool's operational envelope:
the Logbook attempt (a spooled row under ``<out>/logbook-spool`` on every
terminal outcome, an error receipt on failure), the version 2 staging
telemetry with the run's sampling evidence, stage events around each graph
phase and per-epoch rows from the dense solve and, with
``--dataset-households``, the size search and refit (rows tagged with their
``phase``), and the staged-dataset delivery of the published bundle.
The bundle carries ``rowwise_candidate_manifest.json`` projected from the
graph's stored artifacts
(:func:`~microcosm.build.uk_runtime.graph_terminal.rowwise_candidate_manifest_from_graph`),
so the dense release pre-flight and assembler read a graph build as they
read a rowwise-tool build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from microcosm.build.artifact_files import file_artifact, materialize_bytes
from microcosm.build.run_outcome import (
    BuildRefusedError,
    Classified,
    GateBlock,
    RunOutcome,
    classify_failure,
    classify_return,
)
from microcosm.build.termination import BuildTerminatedError, raise_on_sigterm
from microcosm.graph import (
    ArtifactInput,
    ContentStore,
    Graph,
    KernelRegistry,
    SourceRef,
    compile_graph,
    graph_to_json,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import load_raw_bytes

from ..logbook_adoption import (
    AttemptState,
    append_phase,
    git_code_pin,
    local_artifact_reference,
    preflight_digest,
    resolve_predecessor,
    role_pins_digest,
    sha256_argument,
)
from ..staging_cli import (
    add_staged_dataset_arguments,
    add_staging_arguments,
    validate_staged_dataset_arguments,
    validate_staging_arguments,
)
from ..staging_dataset import SHA256SUMS_FILENAME, refresh_sha256sums_entry
from . import country_adapter, national_role
from .atomic_area_support import (
    IDENTITY_COLUMN,
    uk_atomic_assignment_definition,
)
from .atomic_area_support import (
    SOURCES as ATOMIC_SUPPORT_SOURCES,
)
from .atomic_area_support import (
    SYSTEMS as ATOMIC_SUPPORT_SYSTEMS,
)
from .calibration_run import (
    _write_json,
    load_bound_spine_checkpoint,
    new_uk_calibration_attempt_id,
    runtime_provenance,
    spine_provenance_from_sidecar,
)
from .chronicle_feed import load_uk_chronicle_feed
from .frs_release import load_uk_frs_release
from .full_certification import (
    append_uk_full_certification_node,
    register_uk_full_certification_kernel,
)
from .graph_build import (
    SPINE_PROVENANCE_TYPE,
    UKFullBuildConfig,
    UKFullGraph,
    bound_spine_graph,
    register_uk_full_kernels,
    uk_full_graph,
)
from .graph_calibration import UKGraphCalibrationConfig
from .graph_national import (
    NATIONAL_GATES_NODE,
    NATIONAL_READBACK_NODE,
    NATIONAL_TARGETS_NODE,
    UKNationalBuildConfig,
    UKNationalGraph,
    add_uk_national_readback,
    graph_payload,
    national_build_record,
    national_result_from_manifest,
    national_run_config,
    register_uk_national_kernels,
    replay_uk_dense_gate_battery,
    replay_uk_national_gate_battery,
    uk_national_graph,
)
from .graph_population import GEOGRAPHY_ASSIGNMENTS
from .graph_targets import TARGET_SELECTION_TYPE, registry_from_payload
from .graph_terminal import (
    FULL_DIAGNOSTICS_CSV_TYPE,
    FULL_DIAGNOSTICS_TYPE,
    FULL_GATE_REPORT_TYPE,
    FULL_HOLDOUT_TYPE,
    FULL_SUPPORT_CSV_TYPE,
    add_uk_export_continuation,
    add_uk_export_preparation,
    append_uk_full_gate_nodes,
    decode_full_gate_report,
    materialize_uk_export,
    materialize_uk_terminal_artifacts,
    register_uk_full_gate_kernels,
    register_uk_terminal_kernels,
    rowwise_candidate_manifest_from_graph,
)
from .national_doctrine import uk_doctrine_with_overrides
from .national_frame import (
    load_uk_national_frame,
    uk_release_export_frame,
    write_uk_national_frame,
)
from .national_sampling import UK_SAMPLE_RUNG_TOKENS
from .rowwise_cli import (
    MANIFEST_FILENAME,
    REPOSITORY,
    candidate_clone_counts_argument,
    candidate_identity_digest,
    git_commit,
    git_dirty,
    json_text,
    new_candidate_build_id,
    output_paths,
    posture_of,
    record_candidate_attempt,
    record_candidate_error,
    refuse_national_role_arguments,
    resolve_role_arguments,
    rowwise_parameters,
    validate_cli_args,
)
from .rowwise_posture import (
    UK_ROWWISE_DENSE_POSTURE,
    UK_ROWWISE_RELEASE_ROLES,
    uk_rowwise_posture,
)
from .rowwise_staging import (
    STAGED_DATASET_PHASES,
    STAGING_UPLOAD_INTERVAL_SECONDS,
    block_staging_run_bundle,
    create_staging_run_bundle,
    emit_calibration_progress,
    fail_staging_run_bundle,
    finalize_staging_run_bundle,
    gate_statuses,
    graph_progress,
    preflight_staged_dataset,
    replace_manifest,
    stage,
    stage_dataset,
    stage_sample,
    staging_delivery,
    staging_epoch_every,
    start_telemetry_emitter,
    thinned_epochs,
)
from .size_checkpoint import uk_size_checkpoint_identity
from .staging import UK_STAGED_DATASET_REPOSITORY, UK_STAGING_REPOSITORY

# The names the rowwise tool's tests call on either driver.
_validate_cli_args = validate_cli_args
_refuse_national_role_arguments = refuse_national_role_arguments
__all__ = [
    "UK_ROWWISE_DENSE_POSTURE",
    "PreparedUKFullBuild",
    "PreparedUKNationalBuild",
    "execute_full_build",
    "execute_national_build",
    "main",
    "parse_args",
    "prepare_full_build",
    "prepare_national_build",
    "uk_rowwise_posture",
]

#: The graph's full terminal gate document (every declared UK gate), as the
#: evidence materialization names it. The package binds its bytes; the signed
#: local battery report takes the posture's ``*.local_gates.json`` name.
FULL_GATE_REPORT_FILENAME = "uk.full.gates.calibrated.gate_report.json"

#: CLI labels for the three UK atomic-area support systems, in SYSTEMS order.
_SUPPORT_ARGUMENTS = dict(
    zip(ATOMIC_SUPPORT_SYSTEMS, ("ew", "scotland", "ni"), strict=True)
)


def _run_graph_with_progress(compiled, **kwargs):
    """Run a graph and emit one lightweight update per completed node."""

    completed = 0
    previous = time.monotonic()
    total = len(compiled.order)

    def observe(node_id, _population) -> None:
        nonlocal completed, previous
        now = time.monotonic()
        completed += 1
        graph_progress(
            node_id=node_id,
            done=completed,
            total=total,
            elapsed_seconds=now - previous,
        )
        previous = now

    return run_graph(
        compiled,
        _population_observer=observe,
        _population_observer_detach=False,
        **kwargs,
    )


def _target_geographies(value: str) -> tuple[str, ...] | None:
    if value == "all":
        return None
    levels = tuple(value.split(","))
    if (
        not levels
        or len(levels) != len(set(levels))
        or set(levels) - {"country", "region", "constituency", "la"}
    ):
        raise argparse.ArgumentTypeError(
            "Use all or a comma-separated subset of country,region,constituency,la."
        )
    return levels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the request and bind the declared role's defaults.

    Value checks that need the whole namespace (the refusal tables, the
    size-selection rules, the mandatory Ledger pins) run in
    :func:`~microcosm.build.uk_runtime.rowwise_cli.validate_cli_args` from
    :func:`main`, as on the rowwise tool, so the role tests can parse and
    validate in two steps.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-role",
        choices=UK_ROWWISE_RELEASE_ROLES,
        required=True,
        help=(
            "Which UK dataset line this run builds: 'dense' (the K-clone joint "
            "national + local surface under the local doctrine, built through "
            "the graph) or 'national' (the certified national line, dispatched "
            "to the retained calibration seam). The role supplies every unset "
            "solve default and refuses the other role's flags."
        ),
    )
    population = parser.add_mutually_exclusive_group(required=True)
    population.add_argument(
        "--input-h5",
        type=Path,
        help="Canonical spine checkpoint with bound build and gate sidecars.",
    )
    population.add_argument(
        "--spine-request",
        type=Path,
        help="JSON array of raw FRS spine arguments; these stages execute in the same graph.",
    )
    parser.add_argument("--input-sidecar", type=Path)
    parser.add_argument("--input-spine-gates", type=Path)
    parser.add_argument(
        "--input-sha256",
        type=sha256_argument,
        help="Pinned SHA-256 of --input-h5 (required with --input-h5).",
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        help="Full-UK OA geography ladder NPZ (required by the dense role).",
    )
    parser.add_argument("--ladder-sha256", type=sha256_argument)
    parser.add_argument(
        "--geography-assignment",
        choices=GEOGRAPHY_ASSIGNMENTS,
        default="atomic",
        help=(
            "Dense role: atomic (default) is the identity-keyed single-stage "
            "draw on the three UK atomic-area supports; legacy is the sequential "
            "ladder draw, for measurement builds only."
        ),
    )
    for system, label in _SUPPORT_ARGUMENTS.items():
        parser.add_argument(
            f"--atomic-support-{label}",
            type=Path,
            dest=f"atomic_support_{label}",
            help=f"Atomic-area support NPZ for {system} (dense role, required with atomic).",
        )
        parser.add_argument(
            f"--atomic-support-sha256-{label}",
            type=sha256_argument,
            dest=f"atomic_support_sha256_{label}",
        )
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        help="Complete Chronicle artifact directory matching the committed feed pins.",
    )
    parser.add_argument("--ledger-facts-sha256", type=sha256_argument)
    parser.add_argument("--ledger-manifest-sha256", type=sha256_argument)
    parser.add_argument("--measure-exclusions", type=Path)
    parser.add_argument("--register-json", type=Path)
    parser.add_argument("--input-mass-reference", type=Path)
    parser.add_argument(
        "--native-scorecard",
        type=Path,
        help="Measured incumbent-surface comparison bound to immutable candidate.json and its exact output bytes.",
    )
    parser.add_argument(
        "--matched-size-scorecard",
        type=Path,
        help="Additional measured comparison with both populations at requested k.",
    )
    parser.add_argument(
        "--target-geographies",
        type=_target_geographies,
        default=None,
        metavar="all|country,...",
        help="Default all: calibrate all applicable geographies together. country is an explicit filter in this same build.",
    )
    parser.add_argument(
        "--households-only",
        action="store_true",
        help="Bind only Chronicle census-household constituency targets.",
    )
    parser.add_argument(
        "--n-clones",
        type=int,
        help="Geographic pool copies K; defaults to the role's doctrine clone count.",
    )
    parser.add_argument(
        "--candidate-clone-counts",
        type=candidate_clone_counts_argument,
        help="Dry-run only comma-separated candidate clone counts.",
    )
    parser.add_argument(
        "--dataset-households",
        type=int,
        help="Exact exported household count k via informed L0, draw and refit.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Optional pool sampling before cloning. Cannot resample an already sampled source spine.",
    )
    parser.add_argument(
        "--sample-seed", type=int, help="Dense role only; defaults to the spine seed."
    )
    parser.add_argument("--seed", type=int, help="Defaults to the role's seed.")
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--selection-pi-hi", type=float, default=1.0)
    parser.add_argument(
        "--selection-initial-lambda",
        type=float,
        default=None,
        help=(
            "Warm-start the informed L0 budget search at this penalty (the "
            "selected_l0_lambda of a previous search on the same pool and "
            "targets, microcosm#1115). The search probes it first and stops "
            "there when the draw is feasible and within tolerance; a stale "
            "value costs probes, not feasibility (a warm and a cold search may "
            "settle on different penalties inside the budget window). Requires "
            "--dataset-households."
        ),
    )
    parser.add_argument(
        "--baseline-pi-floor",
        type=float,
        default=0.0,
        help=(
            "Floor on the inclusion probability the refit's Horvitz-Thompson "
            "baseline divides each selected row's dense weight by (microcosm#355). "
            "0 (default) is the untrimmed baseline. Requires --dataset-households."
        ),
    )
    parser.add_argument(
        "--no-size-checkpoint",
        action="store_true",
        help=(
            "Do not materialize the dense solve and the informed L0 search as "
            "size_selection_checkpoint.{npz,json} in --out after a "
            "--dataset-households run."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, help="Defaults to the role's doctrine solve length."
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Defaults to the role's learning rate."
    )
    parser.add_argument(
        "--target-weight-rule",
        choices=("uniform", "grain_equal", "family_equal"),
        help="Defaults to the role's doctrine; any other admitted rule is a receipted override.",
    )
    parser.add_argument("--engine-blocks", type=int, default=1)
    parser.add_argument("--source-year", type=int)
    parser.add_argument("--calibration-year", type=int)
    parser.add_argument("--source-lineage-modulus", type=int)
    parser.add_argument(
        "--expected-constituency-vintage",
        help="Dense role only: constituency vintage required from the ladder.",
    )
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--release-candidate", action="store_true")
    parser.add_argument("--review-date", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--resume-size-checkpoint",
        type=Path,
        help="Import an identity-verified historical size search, skipping its dense solve and search.",
    )
    # The national role's own knobs: the dense refusal table names them and
    # the national dispatch (``national_role``) reads them.
    parser.add_argument(
        "--target-loss-cap",
        type=float,
        help="National role only: receipted override of the seam doctrine's per-target loss cap.",
    )
    parser.add_argument(
        "--allow-unpinned-feed",
        action="store_true",
        help="National role only: allow a Ledger artifact whose feed commit is not the committed Chronicle pin.",
    )
    parser.add_argument(
        "--incumbent-h5",
        type=Path,
        help="National role only: the incumbent dataset the finished candidate is evaluated against.",
    )
    parser.add_argument(
        "--incumbent-sha256",
        help="National role only: the incumbent's SHA-256, verified before it is read.",
    )
    parser.add_argument(
        "--incumbent-label",
        default="enhanced_frs_2024_25",
        help="National role only: the incumbent's label in the score receipt.",
    )
    parser.add_argument(
        "--logbook-prev-row-digest",
        type=sha256_argument,
        help=(
            "Optional current Logbook chain head. If omitted, "
            "POPULACE_LOGBOOK_PREV_ROW_DIGEST is used, then genesis null."
        ),
    )
    parser.add_argument(
        "--graph-store",
        type=Path,
        help="Persistent shared graph store; default <out>/.graph-store.",
    )
    parser.add_argument(
        "--resume", choices=("auto", "require", "forbid"), default="auto"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the request and print the compiled operation inventory without fitting or writing files.",
    )
    add_staging_arguments(
        parser,
        repository=UK_STAGING_REPOSITORY,
        default_upload_interval_seconds=STAGING_UPLOAD_INTERVAL_SECONDS,
    )
    add_staged_dataset_arguments(parser, repository=UK_STAGED_DATASET_REPOSITORY)
    args = parser.parse_args(argv)
    validate_staging_arguments(parser, args)
    validate_staged_dataset_arguments(parser, args)
    if args.input_h5 is None and any(
        (args.input_sidecar, args.input_spine_gates, args.input_sha256)
    ):
        parser.error("Input H5 sidecar/pin options require --input-h5.")
    if args.matched_size_scorecard is not None and args.dataset_households is None:
        parser.error("A matched-size scorecard requires --dataset-households.")
    if args.resume_size_checkpoint and args.input_h5 is None:
        parser.error(
            "Historical size checkpoints bind an input H5; raw builds resume using --graph-store."
        )
    resolve_role_arguments(args)
    return args


_parse_args = parse_args


def _pin(path: Path, expected: str | None = None) -> dict:
    record = file_artifact(path)
    if expected is not None and expected != record["sha256"]:
        raise ValueError(f"Input digest differs from the requested pin: {path}.")
    return {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}


def _input_record(path: Path, pin: dict, *, pinned: bool) -> dict:
    """The rowwise manifest's input record: path, digest, bytes and the pin flag."""
    return {
        "path": str(path.resolve()),
        "sha256": str(pin["sha256"]),
        "bytes": int(pin["size_bytes"]),
        "pin_verified": bool(pinned),
    }


def _prepare_geography(
    args: argparse.Namespace, pins: dict, sources: dict[str, Path]
) -> tuple[dict | None, dict]:
    """Pin the three supports and declare the identity-keyed assignment.

    The definition is the UK adapter's declaration over the admitted support
    bytes at the build seed; its sha256 (and every keyed draw) moves iff a
    support artifact or the seed moves. Each support enters ``pins`` as a
    flat pin role named by its graph source, so the Logbook input-pins
    digest, the candidate identity digest and the size-checkpoint identity
    all bind it. Legacy builds bind nothing here.
    """
    if args.geography_assignment != "atomic":
        return None, {"assignment": "legacy"}
    # A release candidate builds on the registered supports (sources.yaml,
    # the provenance register): the operator's pin proves the bytes are the
    # ones requested, the register proves they are the ones reviewed.
    register = (
        country_adapter.uk_atomic_support_register()
        if getattr(args, "release_candidate", False)
        else None
    )
    support_pins = {}
    payloads = {}
    for system, label in _SUPPORT_ARGUMENTS.items():
        path = getattr(args, f"atomic_support_{label}")
        if path is None:
            raise ValueError(
                "Atomic geography assignment requires --atomic-support-ew, "
                "--atomic-support-scotland and --atomic-support-ni."
            )
        pin = _pin(path, getattr(args, f"atomic_support_sha256_{label}"))
        if register is not None and pin != register[system]:
            raise ValueError(
                f"--release-candidate requires the registered atomic-area support "
                f"for --atomic-support-{label}: sha256 {pin['sha256']} "
                f"({pin['size_bytes']} bytes) is not the sources.yaml row "
                f"{register[system]['sha256']} ({register[system]['size_bytes']} bytes)."
            )
        pins[ATOMIC_SUPPORT_SOURCES[system]] = pin
        support_pins[system] = pin
        payloads[system] = load_raw_bytes(path)
        sources[ATOMIC_SUPPORT_SOURCES[system]] = path
    definition = uk_atomic_assignment_definition(payloads, seed=args.seed)
    return definition, {
        "assignment": "atomic",
        "definition_sha256": hashlib.sha256(canonical_json(definition)).hexdigest(),
        "support_pins": support_pins,
        "identity": IDENTITY_COLUMN,
        "stream": list(definition["stream"]),
    }


@dataclass(frozen=True)
class PreparedUKFullBuild:
    full: UKFullGraph
    kernels: KernelRegistry
    sources: dict[str, Path]
    bindings: dict
    spine_provenance: ArtifactInput | None = None
    comparison_sources: dict[str, Path] | None = None
    pins: dict | None = None
    inputs: dict | None = None


def _stderr_progress(line: str) -> None:
    """Solver progress (epoch losses), as the rowwise tool prints them."""
    print(line, file=sys.stderr, flush=True)


def _solve_observer(args: argparse.Namespace, telemetry):
    """Readable stderr lines plus the thinned staging rows of every solve.

    The dense solve, the informed size search and the refit share this one
    observer; the size events carry the ``phase`` the stderr line names and
    the staging row records.
    """
    from .solve_progress import uk_solve_progress_callback

    sinks = [uk_solve_progress_callback(_stderr_progress)]
    sinks.append(
        thinned_epochs(emit_calibration_progress, every=staging_epoch_every(args))
    )
    if telemetry is not None:
        sinks.append(
            thinned_epochs(
                telemetry.calibration_progress, every=staging_epoch_every(args)
            )
        )

    def observer(event: dict[str, object]) -> None:
        for sink in sinks:
            sink(event)

    return observer


def _require_gate_kernel_completed(manifest, node_id: str) -> None:
    """Surface a gate battery that recorded an exception instead of a report.

    A GATE kernel preserves a failure inside its receipt rather than raising,
    so the executor completes the node with no artifacts; the driver then met
    the missing artifact as a bare KeyError (first licensed graph build). Name
    the node and the recorded exception instead.
    """

    receipt = dict(manifest.nodes[node_id].receipt or {})
    evidence = receipt.get("evidence")
    if receipt.get("outcome") == "fail" and isinstance(evidence, dict):
        exception_type = evidence.get("exception_type")
        if exception_type:
            raise RuntimeError(
                f"{node_id} failed inside the gate battery: {exception_type}: "
                f"{evidence.get('message')}"
            )


def prepare_full_build(
    args: argparse.Namespace, *, telemetry=None, attempt: dict | None = None
) -> PreparedUKFullBuild:
    from .calibration_run import load_bound_spine_checkpoint
    from .graph import uk_spine_endpoint
    from .spine_build import (
        _rules_engine,
        _rules_engine_provenance,
        parse_uk_spine_args,
        prepare_uk_spine_execution,
    )

    posture = posture_of(args)
    release = load_uk_frs_release()
    pins = {"ladder": _pin(args.ladder, args.ladder_sha256)}
    inputs = {
        "ladder": _input_record(
            args.ladder, pins["ladder"], pinned=args.ladder_sha256 is not None
        )
    }
    sources = {"uk_ladder": args.ladder, "uk_ledger_facts": args.ledger_facts}
    provenance = None
    if args.input_h5 is not None:
        pins["dataset"] = _pin(args.input_h5, args.input_sha256)
        inputs["dataset"] = _input_record(
            args.input_h5, pins["dataset"], pinned=args.input_sha256 is not None
        )
        frame, _ = load_uk_national_frame(args.input_h5)
        sidecar_path = args.input_sidecar or args.input_h5.with_suffix(".build.json")
        gates_path = args.input_spine_gates or args.input_h5.with_suffix(
            ".spine_gates.json"
        )
        sidecar = load_bound_spine_checkpoint(
            sidecar_path,
            frame,
            gate_report_path=gates_path,
            input_sha256=pins["dataset"]["sha256"],
        )
        spine = bound_spine_graph(frame)
        endpoint = "uk.full.spine_checkpoint"
        weight_kind = frame.weights_for("household").kind.value
        time_period = str(frame.metadata["time_period"])
        source_fraction = float((sidecar.get("sampling") or {}).get("fraction", 1.0))
        stages = tuple(sidecar["stages"])
        _rules_engine()  # the uk extra must be installed; the identity is provenance
        engine_identity = hashlib.sha256(
            canonical_json(_rules_engine_provenance())
        ).hexdigest()
        kernels = KernelRegistry()
        sources.update(
            uk_spine=args.input_h5,
            uk_spine_evidence=sidecar_path,
            uk_spine_gates=gates_path,
        )
        provenance = ArtifactInput(
            "spine_provenance", endpoint, "spine_provenance", SPINE_PROVENANCE_TYPE
        )
    else:
        raw_arguments = json.loads(args.spine_request.read_text())
        if not isinstance(raw_arguments, list) or not all(
            isinstance(x, str) for x in raw_arguments
        ):
            raise ValueError(
                "The spine request must be a JSON array of command arguments."
            )
        # --spine-h5 is an output control of the checkpoint command. Preparation
        # uses it only for path configuration; the full graph writes its own H5.
        if "--spine-h5" not in raw_arguments and not any(
            x.startswith("--spine-h5=") for x in raw_arguments
        ):
            raw_arguments += ["--spine-h5", str(args.out / "spine.h5")]
        raw = parse_uk_spine_args(raw_arguments)
        if args.release_candidate:
            raw.release_candidate = True
        prepared = prepare_uk_spine_execution(raw)
        spine, kernels = prepared.graph, prepared.kernels
        sources.update(prepared.sources)
        endpoint = uk_spine_endpoint(spine).population
        weight_kind = "importance"
        time_period = prepared.frs_release.time_period
        source_fraction = raw.sample_fraction
        stages = prepared.stage_names
        engine_identity = prepared.engine_identity
        inputs["dataset"] = {
            "path": None,
            "sha256": None,
            "bytes": None,
            "pin_verified": False,
            "spine_request": str(args.spine_request.resolve()),
        }
    # The atomic-area supports are admitted after the spine so a spine
    # refusal (sidecar, digest) is reported before a support one.
    definition, geography_binding = _prepare_geography(args, pins, sources)
    stage(
        telemetry,
        "input_pinning",
        "completed",
        dataset_sha256=inputs["dataset"]["sha256"],
        ladder_sha256=pins["ladder"]["sha256"],
    )
    config = UKFullBuildConfig(
        calibration_year=args.calibration_year or release.calibration_year,
        time_period=time_period,
        source_year=args.source_year
        if args.source_year is not None
        else int(time_period),
        geography_levels=args.target_geographies,
        target_families=("census_households/constituency",)
        if args.households_only
        else None,
        n_clones=args.n_clones,
        sample_fraction=args.sample_fraction,
        source_sample_fraction=source_fraction,
        sample_seed=args.sample_seed,
        seed=args.seed,
        engine_blocks=args.engine_blocks,
        constituency_vintage=args.expected_constituency_vintage,
        source_lineage_modulus=args.source_lineage_modulus,
        geography_assignment=args.geography_assignment,
        source_vintage=release.vintage,
        calibration=UKGraphCalibrationConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            dataset_households=args.dataset_households,
            selection_seed=args.selection_seed,
            selection_pi_hi=args.selection_pi_hi,
            selection_initial_lambda=args.selection_initial_lambda,
            baseline_pi_floor=args.baseline_pi_floor,
            target_weight_rule=args.target_weight_rule,
        ),
    )
    args._calibration_year = int(config.calibration_year)
    if attempt is not None:
        state = attempt["state"]
        attempt["code_pin"] = git_code_pin(REPOSITORY)
        state.input_pins_digest = role_pins_digest(pins)
        append_phase(state, "configured")
        append_phase(state, "inputs_pinned")
        state.identity_digest = candidate_identity_digest(
            pins=pins, args=args, source_year=config.source_year
        )
    if args.release_candidate and config.effective_sample_fraction != 1.0:
        raise ValueError(
            "Sampled builds cannot request release-candidate certification."
        )
    feed = load_uk_chronicle_feed()
    for supplied, committed in (
        (args.ledger_facts_sha256, feed.facts_sha256),
        (args.ledger_manifest_sha256, feed.manifest_sha256),
    ):
        if supplied is not None and supplied != committed:
            raise ValueError(
                "Requested Chronicle pin differs from the committed full-build feed."
            )
    optional = []
    for name, path in (
        ("uk_measure_exclusions", args.measure_exclusions),
        ("uk_frozen_register", args.register_json),
    ):
        if path is not None:
            sources[name] = path
            optional.append(name)
    if args.input_mass_reference is not None:
        sources["uk_input_mass_reference"] = args.input_mass_reference
        spine = replace(
            spine,
            sources=(
                *spine.sources,
                SourceRef("uk_input_mass_reference", "raw-bytes-v1"),
            ),
        )
    checkpoint_identity = (
        None
        if args.resume_size_checkpoint is None
        else uk_size_checkpoint_identity(
            args, pins=pins, source_year=config.source_year
        )
    )
    full = uk_full_graph(
        config,
        spine=spine,
        spine_population=endpoint,
        spine_weight_kind=weight_kind,
        optional_target_sources=tuple(optional),
        review_date=args.review_date.isoformat(),
        checkpoint_identity=checkpoint_identity,
        atomic_geography_definition=definition,
    )
    if args.resume_size_checkpoint:
        from .size_checkpoint import (
            SIZE_CHECKPOINT_ARRAYS_FILENAME,
            SIZE_CHECKPOINT_MANIFEST_FILENAME,
        )

        sources["uk_size_checkpoint_manifest"] = (
            args.resume_size_checkpoint / SIZE_CHECKPOINT_MANIFEST_FILENAME
        )
        sources["uk_size_checkpoint_arrays"] = (
            args.resume_size_checkpoint / SIZE_CHECKPOINT_ARRAYS_FILENAME
        )
    graph = append_uk_full_gate_nodes(
        full.graph,
        calibration=full.calibration,
        spine_stage_names=stages,
        engine_identity=engine_identity,
        review_date=args.review_date,
        sample_fraction=config.effective_sample_fraction,
        release_candidate=args.release_candidate,
        spine_provenance=provenance,
        skip_holdout=args.skip_holdout,
    )
    bindings = {
        "schema": "microcosm.uk.full-build-request.v1",
        "release_role": posture.role,
        "configuration": asdict(config),
        "target_scope": "all"
        if config.geography_levels is None
        else list(config.geography_levels),
        "review_date": args.review_date.isoformat(),
        "engine_identity": engine_identity,
        "release_candidate": args.release_candidate,
        "skip_holdout": args.skip_holdout,
        "targets": {
            "chronicle": {
                "facts_sha256": feed.facts_sha256,
                "manifest_sha256": feed.manifest_sha256,
            },
            "paired_ladder_sha256": pins["ladder"]["sha256"],
        },
        "geography": geography_binding,
    }
    graph = add_uk_export_preparation(
        graph,
        population=full.population,
        bindings=bindings,
        artifact_inputs=(
            ArtifactInput(
                "full_gates",
                "uk.full.gates.calibrated",
                "gate_report",
                FULL_GATE_REPORT_TYPE,
            ),
        ),
    )
    register_uk_full_kernels(
        kernels,
        progress_callback=None if args.dry_run else _solve_observer(args, telemetry),
    )
    # The gate batteries take the coverage engine, as the release-cut producer
    # hands them (tools/certify_uk_release_cut.py): its variables() enumerates
    # the loadable inputs the coverage manifest is checked against, whereas the
    # spine adapter's enumerates the frame's non-computed columns (found by the
    # first licensed graph build: sixteen live inputs reported as drift).
    from .release_input_coverage import PolicyEngineUKCoverageEngine

    register_uk_full_gate_kernels(
        kernels,
        coverage_engine=PolicyEngineUKCoverageEngine(),
        engine_identity=engine_identity,
    )
    register_uk_terminal_kernels(kernels)
    register_uk_full_certification_kernel(kernels)
    full = replace(full, graph=graph)
    compile_graph(graph)
    comparisons = {
        name: path
        for name, path in (
            ("uk_native_scorecard", args.native_scorecard),
            ("uk_matched_size_scorecard", args.matched_size_scorecard),
        )
        if path is not None
    }
    return PreparedUKFullBuild(
        full, kernels, sources, bindings, provenance, comparisons, pins, inputs
    )


def _through(graph: Graph, endpoint: str) -> Graph:
    """Execute an actual ancestor-closed checkpoint of the one declared graph."""
    compiled = compile_graph(graph)
    needed, pending = {endpoint}, [endpoint]
    while pending:
        for parent in compiled.predecessors[pending.pop()]:
            if parent not in needed:
                needed.add(parent)
                pending.append(parent)
    return replace(
        graph, nodes=tuple(node for node in graph.nodes if node.id in needed)
    )


def _payload(manifest, store, node: str, artifact: str) -> bytes:
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[artifact])


def _materialize_evidence(manifest, store, out: Path) -> dict:
    inventory = {}
    for node_id, receipt in manifest.nodes.items():
        for name, key in receipt.opaque_artifacts.items():
            payload = store.load_bytes(key)
            suffix = ".json" if payload.startswith((b"{", b"[")) else ".artifact"
            filename = f"{node_id}.{name}{suffix}"
            if Path(filename).name != filename:
                raise ValueError(
                    "Graph artifact name cannot be materialized as a bundle filename."
                )
            inventory[f"{node_id}/{name}"] = {
                "key": key,
                **materialize_bytes(payload, out / filename),
            }
    materialize_bytes(canonical_json(inventory), out / "evidence-index.json")
    return inventory


def _persist_checkpoint(manifest, store, args, phase: str) -> None:
    """Keep receipts and small evidence durable if a later kernel refuses."""
    payload = manifest.to_json_bytes()
    materialize_bytes(payload, args.out / f"{phase}.graph.json")
    attempt = Path(args.attempt_evidence)
    materialize_bytes(payload, attempt / f"{phase}.graph.json")
    evidence = {}
    for node_id, receipt in manifest.nodes.items():
        for name, key in receipt.opaque_artifacts.items():
            if name not in {"gate_report", "stage_evidence", "spine_provenance"}:
                continue
            filename = f"{node_id}.{name}.json"
            if Path(filename).name != filename:
                raise ValueError("Checkpoint artifact names must be simple filenames.")
            evidence[f"{node_id}/{name}"] = {
                "key": key,
                **materialize_bytes(store.load_bytes(key), attempt / filename),
            }
    materialize_bytes(
        canonical_json({"phase": phase, "artifacts": evidence}),
        attempt / "evidence-index.json",
    )


def _argument_sources(args: argparse.Namespace) -> dict[str, Path]:
    """The path-valued request arguments, for the output-location check
    before (or without) a prepared build."""
    return {
        name: path
        for name, path in (
            ("input_h5", args.input_h5),
            ("spine_request", args.spine_request),
            ("input_sidecar", args.input_sidecar),
            ("input_spine_gates", args.input_spine_gates),
            ("ladder", args.ladder),
            ("ledger_facts", args.ledger_facts),
            ("measure_exclusions", args.measure_exclusions),
            ("register_json", args.register_json),
            ("input_mass_reference", args.input_mass_reference),
            ("native_scorecard", args.native_scorecard),
            ("matched_size_scorecard", args.matched_size_scorecard),
            ("resume_size_checkpoint", args.resume_size_checkpoint),
            *(
                (f"atomic_support_{label}", getattr(args, f"atomic_support_{label}"))
                for label in _SUPPORT_ARGUMENTS.values()
            ),
        )
        if path is not None
    }


def _output_locations(prepared: PreparedUKFullBuild | None, args: argparse.Namespace):
    output = args.out.resolve()
    graph_store = (args.graph_store or output / ".graph-store").resolve()
    sources = (
        _argument_sources(args)
        if prepared is None
        else {**prepared.sources, **(prepared.comparison_sources or {})}
    )
    for source in sources.values():
        source = source.resolve()
        if source.is_relative_to(output):
            raise ValueError(
                "Full-build output directory must not contain an input source."
            )
        if source.is_dir() and (
            output.is_relative_to(source) or graph_store.is_relative_to(source)
        ):
            raise ValueError(
                "Output files and graph store cannot alter a declared source directory."
            )
    return output, graph_store


def _release_stem(posture) -> tuple[str, str]:
    """The role's dataset stem and gate-report filename for the FRS vintage."""
    vintage = load_uk_frs_release().vintage
    return (
        posture.dataset_filename(vintage).removesuffix(".h5"),
        posture.gate_report_filename(vintage),
    )


def execute_full_build(
    prepared: PreparedUKFullBuild,
    args: argparse.Namespace,
    *,
    telemetry=None,
    attempt: dict | None = None,
) -> int:
    """Stage complete files, then publish their completion marker last.

    With an ``attempt`` (a non-dry run from :func:`main`) the published
    bundle is then staged as a dataset, the staging telemetry is closed, the
    rowwise manifest gains both receipts and the Logbook row is spooled.
    """
    if args.dry_run:
        return _execute_full_build(prepared, args)
    from microcosm.build.artifact_files import publish_staged_bundle

    # The rung is known once the source spine's own sampling has been read
    # (preparation); it is staged before any graph phase, where the rowwise
    # tool staged it after its sampling step.
    stage_sample(
        telemetry, sample_fraction=prepared.full.config.effective_sample_fraction
    )
    output, graph_store = _output_locations(prepared, args)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Operational attempt identity never enters a scientific node/cache key.
    args.attempt_evidence = graph_store / "uk-full-attempts" / uuid.uuid4().hex
    record: dict = {}
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.full-build-", dir=output.parent
    ) as temporary:
        staged_args = argparse.Namespace(**vars(args))
        staged_args.out = Path(temporary)
        staged_args.graph_store = graph_store
        staged_args.published_out = output
        status = _execute_full_build(
            prepared, staged_args, telemetry=telemetry, attempt=attempt, record=record
        )
        if not (staged_args.out / "build.json").exists():
            materialize_bytes(
                canonical_json(
                    {
                        "schema_version": 1,
                        "kind": "uk_full_build_refused",
                        "request": prepared.bindings,
                        "release_authorized": False,
                        "artifact_permitted": False,
                    }
                ),
                staged_args.out / "build.json",
            )
        staged = {
            ("manifest" if path.name == "build.json" else path.name): path
            for path in staged_args.out.iterdir()
            if path.is_file()
        }
        destinations = {role: output / path.name for role, path in staged.items()}
        publish_staged_bundle(staged, destinations, completion_role="manifest")
    stem, _ = _release_stem(posture_of(args))
    if status == 0:
        print(
            f"UK full build: {output / f'{stem}.h5'}; "
            f"target scope {prepared.bindings['target_scope']}.",
            file=sys.stderr,
        )
    if attempt is not None:
        _close_attempt(
            args,
            attempt,
            output=output,
            stem=stem,
            status=status,
            record=record,
            telemetry=telemetry,
        )
    return status


def _refresh_completion(output: Path, roles: Mapping[str, Path]) -> None:
    """Re-issue the completion marker over the files the close step rewrote.

    ``publish_staged_bundle`` places ``build.json`` last, binding the staged
    bytes. The close step then appends the staging receipts to the rowwise
    manifest (both lines) and the delivery summary to the build record (the
    national line, whose release assembler reads it there), so the marker is
    re-issued with those files' final digests: it stays the bundle's binding
    of what is on disk rather than of what was published a moment earlier.
    """
    marker = output / "build.json"
    completion = json.loads(marker.read_text())
    for role, path in roles.items():
        entry = dict(completion.get(role) or {})
        entry.pop("note", None)
        completion[role] = {**entry, **file_artifact(path)}
    materialize_bytes(canonical_json(completion), marker)


def _close_attempt(
    args: argparse.Namespace,
    attempt: dict,
    *,
    output: Path,
    stem: str,
    status: int,
    record: dict,
    telemetry,
) -> None:
    """Stage the published bundle, close the telemetry, spool the Logbook row."""
    state: AttemptState = attempt["state"]
    manifest = record.get("manifest")
    blocked = bool(record.get("blocking_failures"))
    classified = classify_return(status, record.get("gate_block"))
    if manifest is not None:
        append_phase(state, "published")
        args._gate_report = {"gates": record.get("gate_rows", {})}
        staged_dataset = stage_dataset(
            args,
            manifest=manifest,
            output_paths={"manifest": output / MANIFEST_FILENAME},
            run_id=state.build_id if telemetry is None else telemetry.run_id,
            staging_bundle=telemetry,
        )
        append_phase(state, STAGED_DATASET_PHASES[staged_dataset["status"]])
        try:
            _close_staging(args, telemetry, classified, record)
        finally:
            manifest["staging_delivery"] = staging_delivery(telemetry)
            manifest["staged_dataset"] = staged_dataset
            replace_manifest(output / MANIFEST_FILENAME, manifest)
            if (output / SHA256SUMS_FILENAME).is_file():
                refresh_sha256sums_entry(output, MANIFEST_FILENAME)
            _refresh_completion(
                output, {"rowwise_candidate_manifest": output / MANIFEST_FILENAME}
            )
        state.artifact_location = local_artifact_reference(
            output / f"{stem}.h5", repository_hint=REPOSITORY
        )
    else:
        _close_staging(args, telemetry, classified, record)
    spool_path = record_candidate_attempt(
        state=state,
        started_at=attempt["started_at"],
        started_ts=attempt["started_ts"],
        seed=args.seed,
        code_pin=str(attempt["code_pin"]),
        disposition=classified.disposition,
        predecessor=attempt["predecessor"],
        spool_dir=output / "logbook-spool",
        rung=UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
    )
    print(f"Wrote Logbook row: {spool_path}", file=sys.stderr)
    if manifest is not None:
        print(json_text(manifest), end="")
    if blocked:
        print(
            "Gate battery blocked the artifact at f100; evidence bundle "
            f"written, artifact unreleasable: {record['blocking_failures'][:5]}",
            file=sys.stderr,
        )
    elif classified.block is not None:
        print(
            f"Gate battery refused the candidate at {classified.block.phase}: "
            f"{list(classified.block.blocking_gate_ids)[:5]}",
            file=sys.stderr,
        )


def _apply_graph_gate_verdicts(
    state: AttemptState,
    gate_rows: dict,
    report_path: Path,
    *,
    outcome_index: Mapping[str, int] | None = None,
    scoped_report_path: Path | None = None,
    scoped_gate_ids: Collection[str] = (),
) -> None:
    """Record each gate's verdict with a receipt that resolves in its report.

    A battery report keys its gates (``#/gates/<id>``); a graph gate document
    lists outcomes (``#/report/outcomes/<index>``), so its caller passes the
    outcome positions. Gates a signed scoped report also carries (the dense
    line's local battery) point at that report instead.
    """
    receipt = local_artifact_reference(report_path, repository_hint=REPOSITORY)
    scoped_receipt = (
        None
        if scoped_report_path is None
        else local_artifact_reference(scoped_report_path, repository_hint=REPOSITORY)
    )

    def reference(gate_id: str) -> str:
        if scoped_receipt is not None and gate_id in scoped_gate_ids:
            return f"{scoped_receipt}#/gates/{gate_id}"
        if outcome_index is not None and gate_id in outcome_index:
            return f"{receipt}#/report/outcomes/{outcome_index[gate_id]}"
        return f"{receipt}#/gates/{gate_id}"

    state.gate_verdicts = {
        str(gate_id): {
            "verdict": str(payload["status"]),
            "receipt": reference(str(gate_id)),
        }
        for gate_id, payload in gate_rows.items()
    }


#: Why a filtered dense build ships no local gate report.
LOCAL_GATE_REPORT_ABSENCE = (
    "The target filter selected no local targets, so the local fit claim the "
    "signed local gate report attests does not apply; no report is written."
)


def _local_gate_report_state(report: Mapping | None) -> str:
    """How the build left its local gate report: signed, unsigned, or absent."""
    if report is None:
        return "not_written"
    attestation = report.get("attestation") or {}
    return "unsigned" if "signing_error" in attestation else "signed"


def _graph_gate_rows(document: Mapping) -> tuple[dict, dict[str, int]]:
    """A graph gate document's outcomes as rows by gate id, with their positions."""
    outcomes = document["report"]["outcomes"]
    rows = {
        str(outcome["id"]): {k: v for k, v in outcome.items() if k != "id"}
        for outcome in outcomes
    }
    return rows, {str(outcome["id"]): index for index, outcome in enumerate(outcomes)}


def _close_staging(
    args: argparse.Namespace, telemetry, classified: Classified, record: dict
) -> None:
    """Close both telemetry destinations the way the attempt ended."""
    if classified.outcome is RunOutcome.BLOCKED:
        block_staging_run_bundle(
            args,
            telemetry,
            classified.block,
            gate_statuses=record.get("gate_statuses"),
        )
    elif classified.outcome is RunOutcome.COMPLETED:
        finalize_staging_run_bundle(args, telemetry)
    else:
        fail_staging_run_bundle(
            telemetry,
            BuildRefusedError(
                "The build returned a non-zero status without a recorded gate block."
            ),
        )


def _materialize_size_checkpoint(
    manifest, store, *, args: argparse.Namespace, pins: dict, source_year: int
) -> dict | None:
    """Write ``size_selection_checkpoint.{npz,json}`` from the stored solve.

    The dense result and the informed L0 search are graph artifacts; the
    checkpoint the rowwise tool writes before the exact-count draw is
    materialized from them with the same identity a resume presents through
    ``--resume-size-checkpoint``. A full-pool search (k equal to the pool)
    carries no gate probabilities and writes no checkpoint.
    """
    from microcosm.calibrate.artifacts import (
        decode_calibration_result,
        decode_problem,
    )
    from microcosm.frame import Frame

    from .dataset_size import UKSizeSelection
    from .size_checkpoint import write_uk_size_checkpoint

    if "uk.full.size_search" not in manifest.nodes:
        return None
    selection_meta = json.loads(
        _payload(manifest, store, "uk.full.size_search", "selection")
    )
    if selection_meta.get("method") != "contribution_informed_l0":
        return None
    pool = manifest.population("uk.full.pool")
    problem = decode_problem(_payload(manifest, store, "uk.full.problem", "problem"))
    initial = Frame(
        {entity: pool.table(entity) for entity in pool.entities},
        pool.schema,
        {"household": problem.problem.initial_weights},
        pool.strata,
        mass_log=pool.mass_log,
        metadata=pool.metadata,
    )
    dense = decode_calibration_result(
        _payload(manifest, store, "uk.full.dense", "result"),
        frame=initial,
        problem=problem,
    )
    search = decode_calibration_result(
        _payload(manifest, store, "uk.full.size_search", "result"),
        frame=initial,
        problem=problem,
    )
    selection = UKSizeSelection(
        selection=search,
        protected=np.asarray(selection_meta["protected"], dtype=bool),
        households=int(selection_meta["households"]),
        epochs=int(selection_meta["epochs"]),
        learning_rate=float(selection_meta["learning_rate"]),
        seed=int(selection_meta["seed"]),
        search_pi_hi=float(selection_meta["pi_hi"]),
    )
    return write_uk_size_checkpoint(
        args.out,
        frame=initial,
        dense=dense,
        selection=selection,
        identity=uk_size_checkpoint_identity(args, pins=pins, source_year=source_year),
        provenance={
            "graph_artifacts": {
                "dense": manifest.nodes["uk.full.dense"].opaque_artifacts["result"],
                "search": manifest.nodes["uk.full.size_search"].opaque_artifacts[
                    "result"
                ],
            }
        },
    )


def _execute_full_build(
    prepared: PreparedUKFullBuild,
    args: argparse.Namespace,
    *,
    telemetry=None,
    attempt: dict | None = None,
    record: dict | None = None,
) -> int:
    full, kernels, sources = prepared.full, prepared.kernels, prepared.sources
    if args.dry_run:
        print(json.dumps(full.operation_inventory(), indent=2))
        return 0
    posture = posture_of(args)
    state: AttemptState | None = None if attempt is None else attempt["state"]
    record = {} if record is None else record
    stem, gate_report_name = _release_stem(posture)
    published_root = Path(getattr(args, "published_out", args.out))
    args.out.mkdir(parents=True, exist_ok=True)
    store = ContentStore(args.graph_store or args.out / ".graph-store")
    graph = full.graph
    materialize_bytes(graph_to_json(graph).encode(), args.out / "graph.json")
    materialize_bytes(
        canonical_json(full.operation_inventory()), args.out / "operations.json"
    )
    materialize_bytes(
        canonical_json(prepared.bindings), args.attempt_evidence / "request.json"
    )
    # Preserve source evidence before downstream transforms can raise. Each is
    # an ancestor-closed checkpoint of this graph, with the same node keys/RNG.
    resume = args.resume
    node_ids = {node.id for node in graph.nodes}
    for endpoint in (
        "spine.gates.assembled",
        "spine.gates.transferred",
        "uk.full.spine_checkpoint",
    ):
        if endpoint not in node_ids:
            continue
        checkpoint = _run_graph_with_progress(
            compile_graph(_through(graph, endpoint)),
            sources=sources,
            store=store,
            kernels=kernels,
            resume=resume,
        )
        _persist_checkpoint(checkpoint, store, args, endpoint)
        resume = "require" if args.resume == "require" else "auto"
    # Persist preflight outcomes before any solver can reject them.
    stage(telemetry, "target_compilation", "started")
    preflight_graph = _through(graph, "uk.full.gates.preflight")
    preflight = _run_graph_with_progress(
        compile_graph(preflight_graph),
        sources=sources,
        store=store,
        kernels=kernels,
        resume=resume,
    )
    _persist_checkpoint(preflight, store, args, "preflight")
    _materialize_evidence(preflight, store, args.out)
    if "uk.full.target_selection" in preflight.nodes:
        selection = json.loads(
            _payload(preflight, store, "uk.full.target_selection", "selection")
        )
        included = selection.get("receipt", {}).get("included", [])
        stage(
            telemetry,
            "target_compilation",
            "completed",
            selected_target_count=len(included),
        )
    if state is not None:
        append_phase(state, "targets_bound")
    preflight_bytes = _payload(
        preflight, store, "uk.full.gates.preflight", "gate_report"
    )
    _, admission = decode_full_gate_report(preflight_bytes)
    if not admission["artifact_permitted"]:
        # Refused before solving: a gate block at phase ``preflight``, recorded
        # the way the terminal battery's is, so no record reads it as a pass.
        preflight_rows, preflight_index = _graph_gate_rows(json.loads(preflight_bytes))
        blocking = list(admission["enforced_blocking"]) or ["unidentified_gate"]
        record["gate_rows"] = preflight_rows
        record["gate_statuses"] = gate_statuses({"gates": preflight_rows})
        record["gate_block"] = GateBlock.of("preflight", blocking)
        if state is not None:
            _apply_graph_gate_verdicts(
                state,
                preflight_rows,
                published_root / "uk.full.gates.preflight.gate_report.json",
                outcome_index=preflight_index,
            )
            append_phase(state, "candidate_blocked_at_preflight")
        stage(
            telemetry,
            "preflight_gates",
            "completed",
            gate_statuses=record["gate_statuses"],
            blocking_failure_count=len(blocking),
        )
        return 1
    stage(
        telemetry,
        "calibration",
        "started",
        dataset_households=args.dataset_households,
        epochs=int(args.epochs),
        epoch_every=staging_epoch_every(args),
        resumed_from_checkpoint=args.resume_size_checkpoint is not None,
    )
    manifest = _run_graph_with_progress(
        compile_graph(_through(graph, "uk.full.gates.calibrated")),
        sources=sources,
        store=store,
        kernels=kernels,
        resume="require" if args.resume == "require" else "auto",
    )
    _persist_checkpoint(manifest, store, args, "numerical")
    _materialize_evidence(manifest, store, args.out)
    _require_gate_kernel_completed(manifest, "uk.full.gates.calibrated")
    terminal_files = materialize_uk_terminal_artifacts(
        manifest, store, directory=args.out, stem=stem
    )
    gate_report_bytes = _payload(
        manifest, store, "uk.full.gates.calibrated", "gate_report"
    )
    full_gate_report_path = args.out / FULL_GATE_REPORT_FILENAME
    gate_report_path = args.out / gate_report_name
    terminal_files["full_gates"] = {
        **materialize_bytes(gate_report_bytes, full_gate_report_path),
        "graph_artifact_key": manifest.nodes[
            "uk.full.gates.calibrated"
        ].opaque_artifacts["gate_report"],
    }
    if state is not None:
        append_phase(state, "solved")
    diagnostics = json.loads(
        _payload(manifest, store, "uk.full.gates.calibrated", "calibration_diagnostics")
    )
    stage(
        telemetry,
        "calibration",
        "completed",
        final_loss=diagnostics.get("final_loss"),
        n_nonzero=diagnostics.get("n_nonzero"),
        realized_households=diagnostics.get("n_records"),
    )
    if (
        args.dataset_households is not None
        and not args.no_size_checkpoint
        and args.resume_size_checkpoint is None
    ):
        receipt = _materialize_size_checkpoint(
            manifest,
            store,
            args=args,
            pins=prepared.pins or {},
            source_year=full.config.source_year,
        )
        if receipt is not None and state is not None:
            append_phase(state, "size_selection_checkpointed")
    _, enforcement = decode_full_gate_report(gate_report_bytes)
    gate_document = json.loads(gate_report_bytes)
    # The signed local battery report, written before any refusal so a
    # blocked candidate leaves it too; the graph's enforcement still decides
    # the exit status. A filtered build that drops the local fit claim writes
    # none.
    local_report = replay_uk_dense_gate_battery(
        gate_document,
        report_path=gate_report_path,
        release_id=None if state is None else state.build_id,
        release_candidate=bool(args.release_candidate),
        posture=posture,
    )
    gate_rows, gate_index = _graph_gate_rows(gate_document)
    record["gate_rows"] = gate_rows
    record["blocking_failures"] = list(enforcement["enforced_blocking"])
    record["gate_statuses"] = gate_statuses({"gates": gate_rows})
    if enforcement["enforced_blocking"]:
        record["gate_block"] = GateBlock.of(
            "terminal", enforcement["enforced_blocking"]
        )
    if state is not None:
        _apply_graph_gate_verdicts(
            state,
            gate_rows,
            published_root / FULL_GATE_REPORT_FILENAME,
            outcome_index=gate_index,
            scoped_report_path=None
            if local_report is None
            else published_root / gate_report_name,
            scoped_gate_ids=posture.gate_scope,
        )
        append_phase(
            state,
            "candidate_blocked"
            if enforcement["enforced_blocking"]
            else "candidate_gated",
        )
    stage(
        telemetry,
        "gate_battery",
        "completed",
        gate_statuses=record["gate_statuses"],
        blocking_failure_count=len(enforcement["enforced_blocking"]),
        diagnostic_failure_count=len(enforcement["diagnostic_failures"]),
        local_gate_report=_local_gate_report_state(local_report),
    )
    if not enforcement["artifact_permitted"]:
        return 1
    stage(telemetry, "output_bundle", "started")
    manifest = _run_graph_with_progress(
        compile_graph(graph),
        sources=sources,
        store=store,
        kernels=kernels,
        resume="require" if args.resume == "require" else "auto",
    )
    descriptor = json.loads(
        _payload(manifest, store, "uk.full.export.prepare", "export_descriptor")
    )
    dataset = args.out / f"{stem}.h5"
    frame = manifest.population(full.population)
    materialize_uk_export(frame, descriptor, dataset)
    graph = add_uk_export_continuation(
        graph,
        population=full.population,
        manifest_binding={
            "graph_sha256": hashlib.sha256(graph_to_json(graph).encode()).hexdigest()
        },
        artifact_inputs=(
            ArtifactInput(
                "full_gates",
                "uk.full.gates.calibrated",
                "gate_report",
                FULL_GATE_REPORT_TYPE,
            ),
            ArtifactInput(
                "diagnostics",
                "uk.full.gates.calibrated",
                "calibration_diagnostics",
                FULL_DIAGNOSTICS_TYPE,
            ),
            ArtifactInput("holdout", "uk.full.holdout", "holdout", FULL_HOLDOUT_TYPE),
            ArtifactInput(
                "target_diagnostics",
                "uk.full.gates.calibrated",
                "target_diagnostics_csv",
                FULL_DIAGNOSTICS_CSV_TYPE,
            ),
            ArtifactInput(
                "area_support",
                "uk.full.gates.calibrated",
                "area_support_csv",
                FULL_SUPPORT_CSV_TYPE,
            ),
            ArtifactInput(
                "target_registry",
                "uk.full.target_selection",
                "selection",
                TARGET_SELECTION_TYPE,
            ),
        ),
        evidence_files={
            "full_gates": FULL_GATE_REPORT_FILENAME,
            "diagnostics": terminal_files["calibration_diagnostics"]["filename"],
            "holdout": terminal_files["holdout"]["filename"],
            "target_diagnostics": terminal_files["target_diagnostics"]["filename"],
            "area_support": terminal_files["area_support"]["filename"],
            "target_registry": terminal_files["target_registry"]["filename"],
        },
    )
    evidence_sources = {
        "exported_evidence_full_gates": full_gate_report_path,
        "exported_evidence_diagnostics": args.out
        / terminal_files["calibration_diagnostics"]["filename"],
        "exported_evidence_holdout": args.out / terminal_files["holdout"]["filename"],
        "exported_evidence_target_diagnostics": args.out
        / terminal_files["target_diagnostics"]["filename"],
        "exported_evidence_area_support": args.out
        / terminal_files["area_support"]["filename"],
        "exported_evidence_target_registry": args.out
        / terminal_files["target_registry"]["filename"],
    }
    comparisons = prepared.comparison_sources or {}
    graph = append_uk_full_certification_node(
        graph,
        population=full.population,
        spine_provenance=prepared.spine_provenance,
        comparison_sources=tuple(comparisons),
    )
    final = _run_graph_with_progress(
        compile_graph(graph),
        sources={
            **sources,
            "exported_dataset": dataset,
            **evidence_sources,
            **comparisons,
        },
        store=store,
        kernels=kernels,
        resume="auto",
    )
    final.save(args.out / "build.graph.json")
    materialize_bytes(graph_to_json(graph).encode(), args.out / "graph.json")
    _materialize_evidence(final, store, args.out)
    package = json.loads(_payload(final, store, "uk.full.package", "package_inventory"))
    candidate_file = materialize_bytes(
        canonical_json(package), args.out / "candidate.json"
    )
    certification_payload = _payload(
        final, store, "uk.full.certification", "certification_readiness"
    )
    certification_file = materialize_bytes(
        certification_payload, args.out / "certification.json"
    )

    def output_record(path: Path) -> dict:
        artifact = file_artifact(path)
        return {
            "path": str(published_root / path.name),
            "sha256": artifact["sha256"],
            "bytes": int(artifact["size_bytes"]),
        }

    outputs = {
        "dataset": output_record(dataset),
        "calibration_diagnostics": output_record(
            args.out / terminal_files["calibration_diagnostics"]["filename"]
        ),
        "local_gate_report": None
        if local_report is None
        else output_record(gate_report_path),
        "full_gate_report": output_record(full_gate_report_path),
        "solve_diagnostics": output_record(
            args.out / terminal_files["target_diagnostics"]["filename"]
        ),
        "area_support_summary": output_record(
            args.out / terminal_files["area_support"]["filename"]
        ),
        "holdout": output_record(args.out / terminal_files["holdout"]["filename"]),
        "target_registry": output_record(
            args.out / terminal_files["target_registry"]["filename"]
        ),
    }
    inputs = prepared.inputs or {
        "dataset": {"path": None, "sha256": None, "bytes": None, "pin_verified": False},
        "ladder": {"path": None, "sha256": None, "bytes": None, "pin_verified": False},
    }
    rowwise_manifest = rowwise_candidate_manifest_from_graph(
        final,
        store,
        args=args,
        posture=posture,
        pins=prepared.pins or {},
        terminal_files=terminal_files,
        frame=frame,
        outputs=outputs,
        source_year=full.config.source_year,
        inputs=inputs,
        ladder_provenance=json.loads(
            _payload(final, store, "uk.full.target_compilation", "surface")
        ).get("ladder_provenance", {}),
        code={"git_commit": git_commit(), "git_dirty": git_dirty()},
        runtime=runtime_provenance(),
        created_at=datetime.now(UTC).isoformat(),
        local_gate_report_absence=None
        if local_report is not None
        else LOCAL_GATE_REPORT_ABSENCE,
    )
    materialize_bytes(
        json_text(rowwise_manifest).encode(), args.out / MANIFEST_FILENAME
    )
    record["manifest"] = rowwise_manifest
    completion = {
        **package,
        "kind": "uk_full_build_completion",
        "release_role": posture.role,
        "candidate_manifest": candidate_file,
        "rowwise_candidate_manifest": {
            **file_artifact(args.out / MANIFEST_FILENAME),
            "note": "staging receipts are appended after publication",
        },
        "certification": {
            **certification_file,
            "graph_artifact_key": final.nodes["uk.full.certification"].opaque_artifacts[
                "certification_readiness"
            ],
        },
    }
    materialize_bytes(canonical_json(completion), args.out / "build.json")
    stage(
        telemetry,
        "output_bundle",
        "completed",
        output_bytes={
            key: int(entry["bytes"])
            for key, entry in outputs.items()
            if entry is not None
        },
    )
    return (
        0 if package["readback_passed"] and not enforcement["enforced_blocking"] else 1
    )


# ---------------------------------------------------------------------------
# The national release role through the graph
# ---------------------------------------------------------------------------


@dataclass
class PreparedUKNationalBuild:
    national: UKNationalGraph
    kernels: KernelRegistry
    sources: dict[str, Path]
    bindings: dict
    pins: dict
    inputs: dict
    sidecar_path: Path
    gates_path: Path
    engine_identity: str
    source_pins: dict


def _ledger_facts_pin(path: Path) -> dict:
    facts = path / "consumer_facts.jsonl" if path.is_dir() else path
    artifact = file_artifact(facts)
    return {"sha256": artifact["sha256"], "size_bytes": int(artifact["size_bytes"])}


def refuse_occupied_national_output(args: argparse.Namespace) -> None:
    """Refuse an occupied ``--out`` before any national attempt exists.

    The publication step replaces whatever is at ``--out``, so a second run into
    a candidate's directory would silently replace the first candidate's bytes
    under the Logbook row that names them (review C2); the retired seam refused
    it before any work (``_validate_output_paths``). The refusal runs with the
    other argument refusals, before the attempt id is minted and telemetry
    opens, so it writes nothing into the occupied directory: no failure
    sidecar, no Logbook row, no staging run (microcosm#1057 review round 2,
    item 9). Dry runs write nothing and are not refused.
    """

    national_role.validate_output_paths(
        output_paths(
            args.out,
            posture=posture_of(args),
            vintage=str(load_uk_frs_release().vintage),
        ),
        # The input's existence is checked inside the attempt, as before; the
        # collision check needs only its path.
        input_h5=Path(args.input_h5).expanduser(),
        ladder_path=None,
    )


def prepare_national_build(
    args: argparse.Namespace, *, telemetry=None, attempt: dict | None = None
) -> PreparedUKNationalBuild:
    """Bind the national request to its graph: the checkpoint, the doctrine, the register."""
    from .spine_build import _rules_engine, _rules_engine_provenance

    posture = posture_of(args)
    national_role.require_bound_input(args)
    release = load_uk_frs_release()
    input_h5 = national_role.require_file(args.input_h5, label="--input-h5")
    pins = {"dataset": _pin(input_h5, args.input_sha256)}
    inputs = {"dataset": _input_record(input_h5, pins["dataset"], pinned=True)}
    frame, _ = load_uk_national_frame(input_h5)
    sidecar_path = args.input_sidecar or input_h5.with_suffix(".build.json")
    gates_path = args.input_spine_gates or input_h5.with_suffix(".spine_gates.json")
    load_bound_spine_checkpoint(
        sidecar_path,
        frame,
        gate_report_path=gates_path,
        input_sha256=pins["dataset"]["sha256"],
    )
    spine = bound_spine_graph(frame)
    _rules_engine()  # the uk extra must be installed; the identity is provenance
    engine_identity = hashlib.sha256(
        canonical_json(_rules_engine_provenance())
    ).hexdigest()
    stage(
        telemetry,
        "input_pinning",
        "completed",
        dataset_sha256=pins["dataset"]["sha256"],
    )
    time_period = str(frame.metadata["time_period"])
    calibration_year = int(args.calibration_year or release.calibration_year)
    doctrine, overrides = uk_doctrine_with_overrides(
        **national_role.national_doctrine_overrides(args)
    )
    if doctrine.target_weight_rule != args.target_weight_rule:
        raise RuntimeError(
            "national doctrine override did not bind the requested rule."
        )
    args._calibration_year = calibration_year
    args._frs_vintage = str(release.vintage)
    args._doctrine_override_receipt = overrides
    # An occupied ``--out`` is refused in ``_national_main`` before the attempt
    # opens (``refuse_occupied_national_output``), so a refused run leaves
    # nothing in the first candidate's directory.
    config = UKNationalBuildConfig(
        calibration_year=calibration_year,
        time_period=time_period,
        source_year=national_role.source_year(
            args.source_year, time_period=str(release.time_period)
        ),
        doctrine=doctrine,
        doctrine_overrides=overrides,
        allow_unpinned_feed=bool(args.allow_unpinned_feed),
    )
    sources = {
        "uk_spine": input_h5,
        "uk_spine_evidence": sidecar_path,
        "uk_spine_gates": gates_path,
        "uk_ledger_facts": args.ledger_facts,
    }
    optional = []
    for name, path in (
        ("uk_measure_exclusions", args.measure_exclusions),
        ("uk_frozen_register", args.register_json),
    ):
        if path is not None:
            sources[name] = path
            optional.append(name)
    national = uk_national_graph(
        config,
        spine=spine,
        spine_population="uk.full.spine_checkpoint",
        optional_target_sources=tuple(optional),
        review_date=args.review_date.isoformat(),
        posture=posture,
        ledger_pins={
            "facts_sha256": args.ledger_facts_sha256,
            "manifest_sha256": args.ledger_manifest_sha256,
        },
    )
    kernels = KernelRegistry()
    register_uk_full_kernels(
        kernels,
        progress_callback=None if args.dry_run else _solve_observer(args, telemetry),
    )
    register_uk_national_kernels(kernels)
    compile_graph(national.graph)
    source_pins = {
        "input_h5": {
            "sha256": pins["dataset"]["sha256"],
            "size_bytes": int(pins["dataset"]["size_bytes"]),
        },
        "ledger_facts": _ledger_facts_pin(Path(args.ledger_facts)),
    }
    bindings = {
        "schema": "microcosm.uk.national-build-request.v1",
        "release_role": posture.role,
        "release_id": posture.release_id,
        "configuration": config.payload(),
        "review_date": args.review_date.isoformat(),
        "engine_identity": engine_identity,
        "inputs": inputs,
        "targets": {"ledger": dict(source_pins["ledger_facts"])},
    }
    if attempt is not None:
        state = attempt["state"]
        attempt["code_pin"] = git_code_pin(REPOSITORY)
        state.input_pins_digest = role_pins_digest(source_pins)
        append_phase(state, "configured")
        append_phase(state, "input_sha_verified")
        append_phase(state, "input_loaded")
        append_phase(state, "input_sidecar_bound")
    return PreparedUKNationalBuild(
        national,
        kernels,
        sources,
        bindings,
        pins,
        inputs,
        sidecar_path,
        gates_path,
        engine_identity,
        source_pins,
    )


def execute_national_build(
    prepared: PreparedUKNationalBuild,
    args: argparse.Namespace,
    *,
    telemetry=None,
    attempt: dict | None = None,
) -> int:
    """Stage complete files, then publish their completion marker last."""
    if args.dry_run:
        return _execute_national_build(prepared, args)
    from microcosm.build.artifact_files import publish_staged_bundle

    stage_sample(telemetry, sample_fraction=args.sample_fraction)
    output, graph_store = _output_locations(None, args)
    output.parent.mkdir(parents=True, exist_ok=True)
    args.attempt_evidence = graph_store / "uk-national-attempts" / uuid.uuid4().hex
    record: dict = {}
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.national-build-", dir=output.parent
    ) as temporary:
        staged_args = argparse.Namespace(**vars(args))
        staged_args.out = Path(temporary)
        staged_args.graph_store = graph_store
        staged_args.published_out = output
        status = _execute_national_build(
            prepared, staged_args, telemetry=telemetry, attempt=attempt, record=record
        )
        if not (staged_args.out / "build.json").exists():
            materialize_bytes(
                canonical_json(
                    {
                        "schema_version": 1,
                        "kind": "uk_national_build_refused",
                        "request": prepared.bindings,
                        "release_authorized": False,
                        "artifact_permitted": False,
                    }
                ),
                staged_args.out / "build.json",
            )
        staged = {
            ("manifest" if path.name == "build.json" else path.name): path
            for path in staged_args.out.iterdir()
            if path.is_file()
        }
        destinations = {role: output / path.name for role, path in staged.items()}
        publish_staged_bundle(staged, destinations, completion_role="manifest")
    if status == 0:
        print(
            f"UK national build: {output / record['dataset_name']}; the "
            "release verdict is tools/certify_uk_release_cut.py's.",
            file=sys.stderr,
        )
    if attempt is not None:
        _close_national_attempt(
            args,
            attempt,
            output=output,
            status=status,
            record=record,
            telemetry=telemetry,
        )
    return status


def _execute_national_build(
    prepared: PreparedUKNationalBuild,
    args: argparse.Namespace,
    *,
    telemetry=None,
    attempt: dict | None = None,
    record: dict | None = None,
) -> int:
    from microcosm.build.gate_battery import GateBatteryBlockedError
    from microcosm.build.logbook import canonical_json_bytes
    from microcosm.diagnostics import DiagnosticsWriteFailure

    from .diagnostics import write_uk_calibration_diagnostics
    from .target_support import write_uk_target_support_sidecars

    national, kernels, sources = prepared.national, prepared.kernels, prepared.sources
    if args.dry_run:
        print(json.dumps(national.operation_inventory(), indent=2))
        return 0
    posture = posture_of(args)
    state: AttemptState | None = None if attempt is None else attempt["state"]
    record = {} if record is None else record
    vintage = args._frs_vintage
    paths = output_paths(args.out, posture=posture, vintage=vintage)
    published_root = Path(getattr(args, "published_out", args.out))
    published = output_paths(published_root, posture=posture, vintage=vintage)
    record["dataset_name"] = paths["dataset"].name
    args.out.mkdir(parents=True, exist_ok=True)
    store = ContentStore(args.graph_store or args.out / ".graph-store")
    graph = national.graph
    materialize_bytes(graph_to_json(graph).encode(), args.out / "graph.json")
    materialize_bytes(
        canonical_json(national.operation_inventory()), args.out / "operations.json"
    )
    materialize_bytes(
        canonical_json(prepared.bindings), args.attempt_evidence / "request.json"
    )
    build_id = "dry-run" if state is None else state.build_id
    resume = args.resume
    # 1. The bound checkpoint: its provenance artifact is the solve's admission.
    stage(telemetry, "input_loading", "started")
    checkpoint = _run_graph_with_progress(
        compile_graph(_through(graph, "uk.full.spine_checkpoint")),
        sources=sources,
        store=store,
        kernels=kernels,
        resume=resume,
    )
    _persist_checkpoint(checkpoint, store, args, "uk.full.spine_checkpoint")
    resume = "require" if args.resume == "require" else "auto"
    spine_frame = checkpoint.population("uk.full.spine_checkpoint")
    stage(
        telemetry,
        "input_loading",
        "completed",
        entity_row_counts={
            entity: int(len(spine_frame.table(entity)))
            for entity in spine_frame.entities
        },
    )
    # 2. The register: compiled inside the graph, frozen beside the outputs.
    stage(telemetry, "target_compilation", "started")
    targets = _run_graph_with_progress(
        compile_graph(_through(graph, NATIONAL_TARGETS_NODE)),
        sources=sources,
        store=store,
        kernels=kernels,
        resume=resume,
    )
    registry_document = json.loads(
        graph_payload(targets, store, NATIONAL_TARGETS_NODE, "registry")
    )
    national_registry = registry_from_payload(registry_document["national_registry"])
    band_edge_registry = registry_from_payload(registry_document["band_edge_registry"])
    stage(
        telemetry,
        "target_compilation",
        "completed",
        compiled_target_count=len(band_edge_registry.specs),
        active_target_count=len(national_registry.specs),
    )
    # The frozen register is the scorer's input: the same artifact this run
    # solved against, by content hash; the full compiled register beside it
    # keeps the band edges a pruned scoring surface must never redraw (#803).
    national_registry.to_json(paths["national_registry"])
    band_edge_registry.to_json(paths["contract_registry"])
    source_year = national.config.source_year
    run_config = national_run_config(
        posture=posture,
        config=national.config,
        registry_document=registry_document,
        driver_parameters=rowwise_parameters(args, source_year=source_year),
    )
    if state is not None:
        state.identity_digest = hashlib.sha256(
            canonical_json_bytes(run_config)
        ).hexdigest()
        append_phase(state, "targets_bound")
    inputs = {
        "national_registry": national_registry,
        "band_edge_registry": band_edge_registry,
        "calibration_year": int(registry_document["calibration_year"]),
        "measure_exclusions": dict(registry_document["measure_exclusions"]),
        "chronicle_provenance": dict(registry_document["chronicle_provenance"]),
    }
    record["inputs"] = inputs
    record["incumbent"] = national_role.incumbent_arguments(args)
    # 3. The solve and the seam battery, evaluated in the graph.
    stage(
        telemetry,
        "calibration",
        "started",
        epochs=int(args.epochs),
        epoch_every=staging_epoch_every(args),
    )
    manifest = _run_graph_with_progress(
        compile_graph(_through(graph, NATIONAL_GATES_NODE)),
        sources=sources,
        store=store,
        kernels=kernels,
        resume=resume,
    )
    _persist_checkpoint(manifest, store, args, "numerical")
    _materialize_evidence(manifest, store, args.out)
    _require_gate_kernel_completed(manifest, NATIONAL_GATES_NODE)
    if state is not None:
        append_phase(state, "national_calibration_solved")
    frame = manifest.population(national.population)
    result, problem = national_result_from_manifest(
        manifest, store, frame=frame, registry=national_registry
    )
    evidence = json.loads(
        graph_payload(manifest, store, NATIONAL_GATES_NODE, "calibration_evidence")
    )
    gate_document = json.loads(
        graph_payload(manifest, store, NATIONAL_GATES_NODE, "gate_report")
    )
    stage(
        telemetry,
        "calibration",
        "completed",
        target_count=len(national_registry.specs),
        entity_row_counts={
            entity: int(len(frame.table(entity))) for entity in frame.entities
        },
    )
    # 4. The seam diagnostics, written through the seam writer with the
    # build block; the digest is measured from the file, never declared.
    sidecar = json.loads(prepared.sidecar_path.read_text(encoding="utf-8"))
    spine_provenance = spine_provenance_from_sidecar(prepared.sidecar_path, sidecar)
    input_posture = {
        "tier": "staging_candidate",
        "sha256": prepared.source_pins["input_h5"]["sha256"],
        "size_bytes": prepared.source_pins["input_h5"]["size_bytes"],
    }
    build_block = {
        "build_id": build_id,
        "code_pin": "unresolved-local-git-code-pin"
        if attempt is None
        else str(attempt["code_pin"]),
        "runtime": runtime_provenance(),
        "source_pins": dict(prepared.source_pins),
        "ledger": run_config["ledger"],
        "input_posture": input_posture,
        "doctrine": run_config["doctrine"],
        "doctrine_overrides": run_config["doctrine_overrides"],
        "measure_exclusions": inputs["measure_exclusions"],
        "measure_resolution": evidence["calibration"].get("measure_resolution"),
        "register": evidence["register"],
        "spine_provenance": spine_provenance,
        "score_vs_enhanced_frs": None,
    }
    stage(telemetry, "diagnostics", "started")
    diagnostics_outcome = write_uk_calibration_diagnostics(
        result,
        paths["calibration_diagnostics"],
        frame,
        target_geography_levels=evidence["target_geography_levels"],
        target_registry=national_registry,
        build=build_block,
    )
    if isinstance(diagnostics_outcome, DiagnosticsWriteFailure):
        raise RuntimeError(
            "UK release validation requires calibration diagnostics, but their "
            f"canonical write failed [{diagnostics_outcome.error_code}]: "
            f"{diagnostics_outcome.message}"
        )
    diagnostics_sha = diagnostics_outcome.sha256
    if state is not None:
        append_phase(state, "diagnostics_written")
    stage(
        telemetry,
        "diagnostics",
        "completed",
        target_count=len(evidence["diagnostics"]),
    )
    # 5. The solve's compiled system beside the diagnostics, before the
    # terminal battery, so a blocked attempt keeps its anatomy readable.
    stage(telemetry, "target_support_sidecars", "started")
    sidecars = write_uk_target_support_sidecars(result, args.out)
    if state is not None:
        append_phase(state, "target_support_sidecars_written")
    stage(
        telemetry,
        "target_support_sidecars",
        "completed",
        files=sorted(str(path.name) for path in sidecars.values()),
    )
    # 6. The seam battery, replayed from the graph's phase report and signed.
    stage(telemetry, "release_check_evaluation", "started")
    gate_report_path = paths["terminal_gates"]
    try:
        gate_report = replay_uk_national_gate_battery(
            gate_document,
            report_path=gate_report_path,
            release_id=posture.release_id,
            diagnostics_sha256=diagnostics_sha,
            posture=posture,
        )
    except GateBatteryBlockedError as blocked:
        record["blocking_failures"] = list(blocked.failures)
        written = json.loads(gate_report_path.read_text(encoding="utf-8"))
        gate_rows = written.get("gates", {})
        record["gate_report"] = written
        record["gate_statuses"] = gate_statuses({"gates": gate_rows})
        record["gate_block"] = GateBlock.of(
            blocked.phase,
            blocked.blocking_gate_ids
            or [
                gate_id
                for gate_id, row in gate_rows.items()
                if row.get("status") not in {"passed", "not_applicable"}
            ]
            or ["unidentified_gate"],
            blocking_failure_count=len(blocked.failures),
        )
        if state is not None:
            _apply_graph_gate_verdicts(
                state, gate_rows, published_root / gate_report_path.name
            )
            append_phase(state, "calibration_gates_evaluated")
            append_phase(state, "candidate_blocked")
        stage(
            telemetry,
            "release_check_evaluation",
            "completed",
            check_count=len(gate_rows),
            gate_statuses=record["gate_statuses"],
            blocking_failure_count=len(blocked.failures),
        )
        print(
            "Calibration-seam battery blocked the national artifact; evidence "
            f"bundle written, no H5: {blocked.failures[:5]}",
            file=sys.stderr,
        )
        return 1
    gate_rows = gate_report["gates"]
    record["gate_report"] = gate_report
    if state is not None:
        _apply_graph_gate_verdicts(
            state, gate_rows, published_root / gate_report_path.name
        )
        append_phase(state, "calibration_gates_evaluated")
    stage(
        telemetry,
        "release_check_evaluation",
        "completed",
        check_count=len(gate_rows),
    )
    # 7. The H5, then the graph's readback of it.
    stage(telemetry, "candidate_h5_creation", "started")
    # The release boundary (microcosm#1063 c9): the reviewed export exclusions
    # leave the candidate here, and the graph readback expects the same.
    write_uk_national_frame(uk_release_export_frame(frame), paths["dataset"])
    if state is not None:
        append_phase(state, "staging_h5_written")
    stage(
        telemetry,
        "candidate_h5_creation",
        "completed",
        size_bytes=paths["dataset"].stat().st_size,
    )
    continued = add_uk_national_readback(graph, population=national.population)
    final = _run_graph_with_progress(
        compile_graph(continued),
        sources={**sources, "exported_dataset": paths["dataset"]},
        store=store,
        kernels=kernels,
        resume="auto",
    )
    _require_gate_kernel_completed(final, NATIONAL_READBACK_NODE)
    readback = json.loads(
        graph_payload(final, store, NATIONAL_READBACK_NODE, "readback")
    )
    if not readback["passed"]:
        raise RuntimeError(
            "UK national H5 readback failed: " + "; ".join(readback["failures"])
        )
    final.save(args.out / "build.graph.json")
    _materialize_evidence(final, store, args.out)
    # 8. The build record the release-cut certifier reads.
    stage(telemetry, "build_record_creation", "started")
    artifacts = {
        "staging_h5": {
            "path": str(published["dataset"]),
            "sha256": file_artifact(paths["dataset"])["sha256"],
        },
        "diagnostics_json": {
            "path": str(published["calibration_diagnostics"]),
            "sha256": diagnostics_sha,
        },
        "terminal_gate_json": {
            "path": str(published["terminal_gates"]),
            "sha256": file_artifact(gate_report_path)["sha256"],
        },
    }
    graph_keys = {
        node_id: dict(receipt.opaque_artifacts)
        for node_id, receipt in final.nodes.items()
    }
    build_record = national_build_record(
        posture=posture,
        build_id=build_id,
        run_config=run_config,
        source_pins=prepared.source_pins,
        input_posture=input_posture,
        spine_provenance=spine_provenance,
        register=evidence["register"],
        calibration=evidence["calibration"],
        gate_report=gate_report,
        artifacts=artifacts,
        staging_delivery=staging_delivery(telemetry),
        graph={"artifacts": graph_keys, "readback": readback},
    )
    _write_json(paths["build_record"], build_record)
    if state is not None:
        append_phase(state, "build_record_written")
        state.artifact_location = local_artifact_reference(
            published["dataset"], repository_hint=REPOSITORY
        )
    stage(
        telemetry,
        "build_record_creation",
        "completed",
        size_bytes=paths["build_record"].stat().st_size,
    )
    # 9. The national manifest over the seam-shaped evidence.
    diagnostics = json.loads(paths["calibration_diagnostics"].read_text())
    manifest_payload = national_role.national_manifest(
        args,
        posture=posture,
        build_id=build_id,
        build_record=build_record,
        build_record_path=paths["build_record"],
        gate_report=gate_report,
        diagnostics=diagnostics,
        inputs=inputs,
        doctrine_overrides=national.config.doctrine_overrides,
        output_paths=paths,
        input_artifact=prepared.inputs["dataset"],
        source_year=source_year,
        reported_paths=published,
        graph={"artifacts": graph_keys, "readback": readback},
    )
    materialize_bytes(json_text(manifest_payload).encode(), paths["manifest"])
    record["manifest"] = manifest_payload
    record["build_record"] = build_record
    completion = {
        "schema_version": 1,
        "kind": "uk_national_build_completion",
        "release_role": posture.role,
        "release_id": posture.release_id,
        "build_id": build_id,
        "dataset": {
            **file_artifact(paths["dataset"]),
            "path": str(published["dataset"]),
        },
        "build_record": {
            **file_artifact(paths["build_record"]),
            "path": str(published["build_record"]),
        },
        "rowwise_candidate_manifest": {
            **file_artifact(paths["manifest"]),
            "note": "staging receipts are appended after publication",
        },
        "graph": {"artifacts": graph_keys, "readback": readback},
        "release_authorized": False,
    }
    materialize_bytes(canonical_json(completion), args.out / "build.json")
    return 0


def _close_national_attempt(
    args: argparse.Namespace,
    attempt: dict,
    *,
    output: Path,
    status: int,
    record: dict,
    telemetry,
) -> None:
    """Stage the published bundle, evaluate, close the telemetry, spool the row.

    The seam's order: the manifest and the staged bundle first, the
    incumbent evaluation after the bundle is on the Hub and before the
    telemetry completes, then the build record and the manifest rewritten
    with the delivery summary and the sums refreshed.
    """
    from microcosm.build.staging_v2 import validate_staging_delivery

    posture = posture_of(args)
    state: AttemptState = attempt["state"]
    manifest = record.get("manifest")
    classified = classify_return(status, record.get("gate_block"))
    paths = output_paths(output, posture=posture, vintage=args._frs_vintage)
    if manifest is not None:
        append_phase(state, "published")
        args._gate_report = record["gate_report"]
        staged_dataset = stage_dataset(
            args,
            manifest=manifest,
            output_paths=paths,
            run_id=state.build_id if telemetry is None else telemetry.run_id,
            staging_bundle=telemetry,
        )
        append_phase(state, STAGED_DATASET_PHASES[staged_dataset["status"]])
        evaluation = national_role.evaluate_against_incumbent(
            args,
            incumbent=record.get("incumbent"),
            inputs=record["inputs"],
            output_paths=paths,
            telemetry=telemetry,
            calibration_year=int(record["inputs"]["calibration_year"]),
            out_dir=output,
        )
        try:
            _close_staging(args, telemetry, classified, record)
        finally:
            delivery = staging_delivery(telemetry)
            build_record = json.loads(paths["build_record"].read_text())
            build_record["staging_delivery"] = validate_staging_delivery(delivery)
            _write_json(paths["build_record"], build_record)
            manifest["outputs"]["build_record"] = national_role.artifact_info(
                paths["build_record"]
            )
            manifest["build_record"] = {
                "path": str(paths["build_record"]),
                "sha256": manifest["outputs"]["build_record"]["sha256"],
            }
            manifest["staging_delivery"] = delivery
            manifest["staged_dataset"] = staged_dataset
            manifest["evaluation"] = evaluation
            if evaluation.get("status") == "completed":
                manifest["outputs"]["score_receipt"] = evaluation["receipt"]
            replace_manifest(paths["manifest"], manifest)
            if (output / SHA256SUMS_FILENAME).is_file():
                refresh_sha256sums_entry(output, paths["build_record"].name)
                if evaluation.get("status") == "completed":
                    national_role.list_sha256sums_entry(
                        output, paths["score_receipt"].name
                    )
                refresh_sha256sums_entry(output, MANIFEST_FILENAME)
            _refresh_completion(
                output,
                {
                    "build_record": paths["build_record"],
                    "rowwise_candidate_manifest": paths["manifest"],
                },
            )
    else:
        _close_staging(args, telemetry, classified, record)
    spool_path = record_candidate_attempt(
        state=state,
        started_at=attempt["started_at"],
        started_ts=attempt["started_ts"],
        seed=args.seed,
        code_pin=str(attempt["code_pin"]),
        disposition=classified.disposition,
        predecessor=attempt["predecessor"],
        spool_dir=output / "logbook-spool",
        rung="f100",
        pipeline=posture.pipeline,
    )
    print(f"Wrote Logbook row: {spool_path}", file=sys.stderr)
    if manifest is not None:
        print(json_text(manifest), end="")


def _run_with_telemetry(
    args: argparse.Namespace, *, build_id: str, operation: Callable[[], int]
) -> int:
    """Report the command's actual outcome, then release its telemetry handle."""
    emitter = start_telemetry_emitter(
        args,
        build_id=build_id,
        run_kind="dry_run" if args.dry_run else "calibration",
    )
    emitter.transition_stage(
        "preflight",
        message=f"Validating UK {posture_of(args).role} build inputs and configuration.",
    )
    try:
        status = operation()
    except BaseException as error:
        # The attempt's own close-out has normally closed the emitter with its
        # classification already; an error raised before the attempt took over
        # (its preflight digest, say) is classified the same way here.
        if emitter.available:
            classified = classify_failure(error)
            if classified.block is not None:
                emitter.block(
                    phase=classified.block.phase,
                    blocking_gate_ids=list(classified.block.blocking_gate_ids),
                    blocking_failure_count=classified.block.blocking_failure_count,
                )
            else:
                emitter.fail(
                    error,
                    failure_class=classified.failure_class or "build_failure",
                    error_code=classified.error_code,
                )
        raise
    else:
        if emitter.available:
            if status == 0:
                emitter.complete()
            else:
                emitter.fail(
                    RuntimeError(f"UK build returned exit status {status}."),
                    failure_class="build_failure",
                )
        return status
    finally:
        emitter.close()


def _national_main(args: argparse.Namespace) -> int:
    """Run the national build with one telemetry lifecycle."""
    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    build_id = new_uk_calibration_attempt_id(timestamp=started_ts)
    return _run_with_telemetry(
        args,
        build_id=build_id,
        operation=lambda: _national_attempt(
            args, build_id=build_id, started_at=started_at, started_ts=started_ts
        ),
    )


def _national_attempt(
    args: argparse.Namespace, *, build_id: str, started_at: float, started_ts: datetime
) -> int:
    """Prepare, execute, and record the national build's artifacts."""
    posture = posture_of(args)
    digest = preflight_digest(posture.pipeline)
    state = AttemptState(
        # The staging run id and Logbook row use the same attempt id.
        build_id=build_id,
        identity_digest=digest,
        input_pins_digest=digest,
        phases_reached=["attempt_started"],
        gate_verdicts={
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-terminal-receipt",
            }
        },
    )
    if args.dry_run:
        try:
            status = national_role.national_dry_run(
                args,
                operation_inventory=lambda: prepare_national_build(
                    args
                ).national.operation_inventory(),
            )
        except BaseException as error:
            fail_staging_run_bundle(None, error)
            raise
        finalize_staging_run_bundle(args, None)
        return status

    try:
        national_role.require_bound_input(args)
        # These checks must precede input loading. The emitter is already
        # running so a refusal is visible as a failed attempt.
        refuse_occupied_national_output(args)
        preflight_staged_dataset(args)
        predecessor = resolve_predecessor(args.logbook_prev_row_digest)
        telemetry = create_staging_run_bundle(args, build_id=state.build_id)
    except BaseException as error:
        fail_staging_run_bundle(None, error)
        raise
    attempt = {
        "state": state,
        "started_at": started_at,
        "started_ts": started_ts,
        "code_pin": "unresolved-local-git-code-pin",
        "predecessor": predecessor,
    }
    try:
        prepared = prepare_national_build(args, telemetry=telemetry, attempt=attempt)
        return execute_national_build(
            prepared, args, telemetry=telemetry, attempt=attempt
        )
    except KeyboardInterrupt as interrupt:
        # An operator interrupt is a discarded attempt, not a failed one; the
        # seam recorded the row so and re-raised (every terminal disposition
        # records a row), and the staging run closes as failed (INTERRUPTED).
        _record_failure(
            args,
            interrupt,
            state=state,
            attempt=attempt,
            pipeline=posture.pipeline,
            disposition=classify_failure(interrupt).disposition,
        )
        fail_staging_run_bundle(telemetry, interrupt)
        raise
    except Exception as error:
        _record_failure(
            args,
            error,
            state=state,
            attempt=attempt,
            pipeline=posture.pipeline,
            disposition=classify_failure(error).disposition,
        )
        fail_staging_run_bundle(telemetry, error)
        print(f"UK national build failed: {error}", file=sys.stderr)
        return 1


def _record_failure(
    args: argparse.Namespace,
    error: BaseException,
    *,
    state: AttemptState,
    attempt: dict,
    pipeline: str,
    prepared: PreparedUKFullBuild | None = None,
    disposition: str = "failed",
) -> None:
    """The failure sidecar and the terminal Logbook row, when the output is safe.

    ``disposition`` is ``failed`` for an error and ``discarded`` for an
    operator interrupt; the dense line passes its ``prepared`` build so the
    output check sees the declared sources.
    """
    try:
        _output_locations(prepared, args)
    except ValueError:
        return
    materialize_bytes(
        canonical_json(
            {
                "schema": "microcosm.uk.full-build-failure.v1",
                "error_type": type(error).__name__,
                "message": str(error),
                "evidence_directory": str(args.attempt_evidence)
                if hasattr(args, "attempt_evidence")
                else None,
                "release_authorized": False,
            }
        ),
        args.out / "failure.json",
    )
    if state.spool_path is None:
        record_candidate_error(
            error=error,
            state=state,
            started_at=attempt["started_at"],
            started_ts=attempt["started_ts"],
            seed=args.seed,
            code_pin=str(attempt["code_pin"]),
            predecessor=attempt["predecessor"],
            base_dir=args.out.resolve(),
            spool_dir=args.out.resolve() / "logbook-spool",
            rung=UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
            pipeline=pipeline,
            disposition=disposition,
        )


def _dry_run(args: argparse.Namespace) -> int:
    """Plan without solving or writing; loop over candidate clone counts."""
    counts = args.candidate_clone_counts
    if counts is None:
        return execute_full_build(prepare_full_build(args), args)
    inventories = []
    for count in counts:
        planned = argparse.Namespace(**vars(args))
        planned.n_clones = int(count)
        prepared = prepare_full_build(planned)
        inventories.append(
            {"n_clones": int(count), **prepared.full.operation_inventory()}
        )
    print(json.dumps(inventories, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one UK build; a SIGTERM is recorded like Ctrl-C, then exits 143."""
    with raise_on_sigterm():
        try:
            return _main(argv)
        except BuildTerminatedError as terminated:
            raise SystemExit(terminated.exit_code) from terminated


def _main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_cli_args(args)
    posture = posture_of(args)
    if posture.role == "national":
        # The national line: the bound checkpoint solved as bound under the
        # national doctrine, on the same graph (microcosm#823 line).
        return _national_main(args)
    if args.candidate_clone_counts is not None and not args.dry_run:
        raise ValueError("--candidate-clone-counts is valid only with --dry-run.")
    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    build_id = new_candidate_build_id(
        seed=args.seed,
        timestamp=started_ts,
        rung=UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
    )
    return _run_with_telemetry(
        args,
        build_id=build_id,
        operation=lambda: _dense_attempt(
            args, build_id=build_id, started_at=started_at, started_ts=started_ts
        ),
    )


def _dense_attempt(
    args: argparse.Namespace, *, build_id: str, started_at: float, started_ts: datetime
) -> int:
    """Prepare, execute, and record the dense build's artifacts."""
    posture = posture_of(args)
    digest = preflight_digest(posture.pipeline)
    state = AttemptState(
        build_id=build_id,
        identity_digest=digest,
        input_pins_digest=digest,
        phases_reached=["attempt_started"],
        gate_verdicts={
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-terminal-receipt",
            }
        },
    )
    if args.dry_run:
        # Dry runs plan without solving or writing and record no Logbook row,
        # but their hosted telemetry still has a complete lifecycle.
        try:
            status = _dry_run(args)
        except BaseException as error:
            fail_staging_run_bundle(None, error)
            raise
        finalize_staging_run_bundle(args, None)
        return status

    try:
        # The credential and chain checks still precede input loading, while
        # the emitter records their failures.
        preflight_staged_dataset(args)
        predecessor = resolve_predecessor(args.logbook_prev_row_digest)
        telemetry = create_staging_run_bundle(args, build_id=state.build_id)
    except BaseException as error:
        fail_staging_run_bundle(None, error)
        raise
    attempt = {
        "state": state,
        "started_at": started_at,
        "started_ts": started_ts,
        "code_pin": "unresolved-local-git-code-pin",
        "predecessor": predecessor,
    }
    prepared = None
    try:
        prepared = prepare_full_build(args, telemetry=telemetry, attempt=attempt)
        return execute_full_build(prepared, args, telemetry=telemetry, attempt=attempt)
    except KeyboardInterrupt as interrupt:
        # As on the national line: a discarded row, the staging run failed,
        # the interrupt re-raised.
        _record_failure(
            args,
            interrupt,
            state=state,
            attempt=attempt,
            pipeline=posture.pipeline,
            prepared=prepared,
            disposition=classify_failure(interrupt).disposition,
        )
        fail_staging_run_bundle(telemetry, interrupt)
        raise
    except Exception as error:
        _record_failure(
            args,
            error,
            state=state,
            attempt=attempt,
            pipeline=posture.pipeline,
            prepared=prepared,
            disposition=classify_failure(error).disposition,
        )
        fail_staging_run_bundle(telemetry, error)
        print(f"UK full build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
