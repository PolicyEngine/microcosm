"""The UK rowwise driver's national release role (microcosm#823).

``microcosm-build-uk --release-role national`` builds the certified national
line by dispatching to the retained calibration seam
(:func:`~microcosm.build.uk_runtime.calibration_run.run_uk_calibration`): no
cloning, national targets only, the seam doctrine with explicit flags as
receipted overrides, the six calibration-seam gates, the seam-shaped
``build_record.json`` that ``tools/certify_uk_release_cut.py`` certifies, a
frozen ``national_target_registry.json`` for the scorer, and the same
staging telemetry (run id = the attempt id) and staged bundle as the dense
role. The driver adds the pinned input, the rowwise manifest beside the
seam's evidence, and the optional end-of-build evaluation against the
incumbent (microcosm#578 rule 1 on the common surface).

These functions were moved from ``tools/build_uk_rowwise_candidate.py``
(microcosm#901 phase 4) and consume only the shared rowwise command surface
(:mod:`~microcosm.build.uk_runtime.rowwise_cli`,
:mod:`~microcosm.build.uk_runtime.rowwise_staging`) and package APIs; the
graph full-build driver (:mod:`~microcosm.build.uk_runtime.full_build_cli`)
dispatches here before any graph preparation. The private spellings are
kept as aliases so the drivers' tests can patch the names they always did.
The posture-driven graph national path (the next PR on this line) replaces
this dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.logbook_adoption import atomic_write_json
from microcosm.build.staging_dataset import (
    SHA256SUMS_FILENAME,
    parse_sha256sums,
    refresh_sha256sums_entry,
)
from microcosm.build.staging_v2 import StagingTelemetryV2
from microcosm.build.uk_runtime.calibration_run import (
    UKCalibrationRunPaths,
    new_uk_calibration_attempt_id,
    run_uk_calibration,
    runtime_provenance,
)
from microcosm.build.uk_runtime.chronicle_feed import (
    require_committed_uk_chronicle_feed_pin,
)
from microcosm.build.uk_runtime.diagnostics import uk_fit_by_family
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.ledger_targets import compile_uk_target_registry
from microcosm.build.uk_runtime.measure_simulation import (
    UKMeasureResolver,
    apply_uk_calibration_measure_exclusions,
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.national_doctrine import uk_doctrine_with_overrides
from microcosm.build.uk_runtime.rowwise_cli import (
    git_commit,
    git_dirty,
    json_text,
    output_paths,
    posture_of,
    rowwise_parameters,
)
from microcosm.build.uk_runtime.rowwise_posture import UKRowwisePosture
from microcosm.build.uk_runtime.rowwise_staging import (
    add_staging_artifact,
    create_staging_telemetry,
    fail_staging_telemetry,
    finalize_staging_telemetry,
    gate_statuses,
    preflight_staged_dataset,
    replace_manifest,
    stage,
    stage_dataset,
    staging_delivery,
    staging_epoch_every,
    thinned_epochs,
)
from microcosm.calibrate import TargetRegistry

__all__ = ["national_dry_run", "run_national_role"]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must hold a JSON object.")
    return payload


def _ledger_facts_pin(artifact: Any) -> dict[str, object]:
    facts_path = (
        artifact.path / "consumer_facts.jsonl"
        if artifact.path.is_dir()
        else artifact.path
    )
    return {"sha256": artifact.facts_sha256, "size_bytes": facts_path.stat().st_size}


def _load_national_target_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """The national role's target surface: the pinned Ledger artifact, compiled.

    The artifact must be the committed Chronicle feed pin unless
    ``--allow-unpinned-feed`` records a reviewed diagnostic run; the
    compiled register, less the measure exclusions, is the solve surface and
    the full compiled register keeps the band edges; ``--register-json``
    requires the re-derived register to be the frozen scoring surface.
    """

    artifact = load_ledger_consumer_artifact(
        args.ledger_facts,
        expected_facts_sha256=args.ledger_facts_sha256,
        expected_manifest_sha256=args.ledger_manifest_sha256,
    )
    pin = require_committed_uk_chronicle_feed_pin(
        artifact.facts_sha256,
        manifest_sha256=artifact.manifest_sha256,
        allow_unpinned_feed=bool(args.allow_unpinned_feed),
    )
    calibration_year = int(load_uk_frs_release().calibration_year)
    compilation = compile_uk_target_registry(
        artifact.facts, target_period=calibration_year
    )
    if compilation.unsupported:
        raise SystemExit(
            f"{len(compilation.unsupported)} national target references "
            "failed to compile"
        )
    exclusions = load_uk_calibration_measure_exclusions(args.measure_exclusions)
    registry, exclusion_receipt = apply_uk_calibration_measure_exclusions(
        compilation.registry, exclusions
    )
    if args.register_json is not None:
        try:
            frozen = TargetRegistry.from_json(args.register_json)
        except ValueError as error:
            raise SystemExit(
                f"error: frozen scoring register is unusable: {error}"
            ) from error
        if frozen.version != registry.version:
            raise SystemExit(
                "re-derived register differs from the frozen scoring register: "
                f"{registry.version} vs {frozen.version}"
            )
    return {
        "artifact": artifact,
        "calibration_year": calibration_year,
        "national_registry": registry,
        "band_edge_registry": compilation.registry,
        "measure_exclusions": exclusion_receipt,
        "chronicle_feed_pin": pin.to_dict(),
    }


def _national_doctrine_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """The receipted per-run overrides of the seam doctrine, explicit flags only."""

    explicit = args._explicit_arguments
    overrides: dict[str, Any] = {}
    if "epochs" in explicit:
        overrides["epochs"] = int(args.epochs)
    if "learning_rate" in explicit:
        overrides["learning_rate"] = float(args.learning_rate)
    if "target_weight_rule" in explicit:
        overrides["target_weight_rule"] = str(args.target_weight_rule)
    if args.target_loss_cap is not None:
        overrides["target_loss_cap"] = float(args.target_loss_cap)
    return overrides


def _require_bound_input(args: argparse.Namespace) -> None:
    """The national role builds from a bound spine checkpoint, never a request.

    The graph driver admits ``--spine-request`` as the dense build's
    population source; the seam engine reads a pinned ``--input-h5``.
    """

    if args.input_h5 is None:
        raise ValueError(
            "--release-role national builds from a bound spine checkpoint: pass "
            "--input-h5 with --input-sha256 (--spine-request executes the spine "
            "stages only in the graph dense build)."
        )


def national_dry_run(args: argparse.Namespace) -> int:
    """Compile the national target surface and print the plan; write nothing."""

    posture = posture_of(args)
    input_h5 = _require_file(args.input_h5, label="--input-h5")
    input_artifact = _artifact_info(input_h5)
    _verify_requested_pin("--input-h5", input_artifact, requested=args.input_sha256)
    frs_release = load_uk_frs_release()
    inputs = _load_national_target_inputs(args)
    doctrine, doctrine_overrides = uk_doctrine_with_overrides(
        **_national_doctrine_overrides(args)
    )
    plan = {
        "schema_version": 3,
        "build_kind": "uk_national_calibrated_candidate_plan",
        "release_role": posture.role,
        "release_id": posture.release_id,
        "dry_run": True,
        "calibration_year": inputs["calibration_year"],
        "inputs": {"dataset": dict(input_artifact)},
        "targets": {
            "chronicle": inputs["artifact"].provenance(),
            "compiled": len(inputs["band_edge_registry"].specs),
            "active": len(inputs["national_registry"].specs),
            "excluded": len(inputs["measure_exclusions"]),
            "register_sha256": inputs["national_registry"].version,
        },
        "doctrine": {
            field: getattr(doctrine, field)
            for field in (
                "epochs",
                "learning_rate",
                "max_weight_ratio",
                "seed",
                "target_loss_cap",
                "scale_rule",
                "target_weight_rule",
                "mass_rule",
                "l0_lambda",
            )
        },
        "doctrine_overrides": dict(doctrine_overrides),
        "parameters": rowwise_parameters(
            args,
            source_year=_source_year(
                args.source_year, time_period=str(frs_release.time_period)
            ),
        ),
        "engine": "not_run",
        "incumbent": _incumbent_arguments(args),
        "releasable": False,
    }
    print(json_text(plan))
    return 0


_national_dry_run = national_dry_run


def run_national_role(args: argparse.Namespace) -> int:
    """Serve ``--release-role national``: the dry-run plan or the seam build.

    The validated request arrives from the driver's ``main`` before any graph
    preparation. A dry run plans without solving, writing or staging; a build
    runs the staged-dataset pre-flight (argument refusals cost nothing; the
    credential check reaches the Hub, so it runs last, still before any input
    is read) and then the seam build under the driver's posture.
    """

    _require_bound_input(args)
    if args.dry_run:
        return national_dry_run(args)
    preflight_staged_dataset(args)
    return _run_national_role(args)


def _run_national_role(args: argparse.Namespace) -> int:
    """Build the national line: the calibration seam under the driver's posture.

    The seam library (:func:`run_uk_calibration`) resolves the measures from
    the input file, solves under the seam doctrine, runs the six
    calibration-seam gates, writes the H5, the diagnostics, the signed gate
    report, the build record and the Logbook row exactly as the retired
    seam command did, so a national cut built here is bit-for-bit the seam's.
    The driver adds what the dense role has: the pinned input, the role's
    doctrine and overrides, staging telemetry under the attempt id, the
    rowwise manifest beside the seam's evidence, and the staged bundle.
    """

    posture = posture_of(args)
    out_dir = args.out.expanduser().resolve()
    input_h5 = _require_file(args.input_h5, label="--input-h5")
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"--out must be a directory path, got {out_dir}.")
    input_artifact = _artifact_info(input_h5)
    _verify_requested_pin("--input-h5", input_artifact, requested=args.input_sha256)
    incumbent = _incumbent_arguments(args)
    # The attempt id is minted before telemetry opens so the staging run id
    # and the Logbook row agree, as on the dense role.
    build_id = new_uk_calibration_attempt_id(timestamp=datetime.now(UTC))
    telemetry = create_staging_telemetry(args, build_id=build_id)
    try:
        return _run_national_attempt(
            args,
            posture=posture,
            out_dir=out_dir,
            input_h5=input_h5,
            input_artifact=input_artifact,
            build_id=build_id,
            telemetry=telemetry,
            incumbent=incumbent,
        )
    except BaseException as error:
        fail_staging_telemetry(telemetry, error)
        raise


def _run_national_attempt(
    args: argparse.Namespace,
    *,
    posture: UKRowwisePosture,
    out_dir: Path,
    input_h5: Path,
    input_artifact: Mapping[str, Any],
    build_id: str,
    telemetry: StagingTelemetryV2 | None,
    incumbent: Mapping[str, Any] | None = None,
) -> int:
    stage(
        telemetry,
        "input_pinning",
        "completed",
        dataset_sha256=input_artifact["sha256"],
    )
    if telemetry is not None:
        telemetry.set_sample({"mode": "full"})
    frs_release = load_uk_frs_release()
    calibration_year = int(frs_release.calibration_year)
    args._calibration_year = calibration_year
    args._frs_vintage = str(frs_release.vintage)
    source_year = _source_year(
        args.source_year, time_period=str(frs_release.time_period)
    )
    paths_by_role = output_paths(out_dir, posture=posture, vintage=args._frs_vintage)
    _validate_output_paths(paths_by_role, input_h5=input_h5, ladder_path=None)
    stage(telemetry, "target_compilation", "started")
    inputs = _load_national_target_inputs(args)
    stage(
        telemetry,
        "target_compilation",
        "completed",
        compiled_target_count=len(inputs["band_edge_registry"].specs),
        active_target_count=len(inputs["national_registry"].specs),
    )
    doctrine, doctrine_overrides = uk_doctrine_with_overrides(
        **_national_doctrine_overrides(args)
    )
    if doctrine.target_weight_rule != args.target_weight_rule:
        raise RuntimeError(
            "national doctrine override did not bind the requested rule."
        )
    args._doctrine_override_receipt = doctrine_overrides
    out_dir.mkdir(parents=True, exist_ok=True)
    # The frozen register is the scorer's input: the same artifact this run
    # solved against, by content hash.
    inputs["national_registry"].to_json(paths_by_role["national_registry"])
    # The full compiled register beside it: the band edges a pruned scoring
    # surface must never redraw (#803), for the end-of-build evaluation and
    # for a re-score by hand (--band-edge-registry-json).
    inputs["band_edge_registry"].to_json(paths_by_role["contract_registry"])
    resolver = UKMeasureResolver(
        simulation_source=input_h5,
        scratch_dir=out_dir,
        year=calibration_year,
        frame=None,
    )
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=paths_by_role["dataset"],
        diagnostics_json=paths_by_role["calibration_diagnostics"],
        build_record_json=paths_by_role["build_record"],
        terminal_gate_json=paths_by_role["terminal_gates"],
    )
    evidence: dict[str, Any] = {}

    def publish_manifest() -> None:
        # The seam has written its evidence; the manifest describes it and the
        # staged bundle (every manifest-registered output) follows, as on the
        # dense role. The staged copy predates the evidence blocks appended
        # below; staged_manifest.json describes the remote side.
        args._gate_report = _read_json(paths.terminal_gate_json)
        manifest = _national_manifest(
            args,
            posture=posture,
            build_id=build_id,
            build_record=_read_json(paths.build_record_json),
            build_record_path=paths.build_record_json,
            gate_report=args._gate_report,
            diagnostics=_read_json(paths.diagnostics_json),
            inputs=inputs,
            doctrine_overrides=doctrine_overrides,
            output_paths=paths_by_role,
            input_artifact=input_artifact,
            source_year=source_year,
        )
        replace_manifest(paths_by_role["manifest"], manifest)
        evidence["manifest"] = manifest
        evidence["staged_dataset"] = stage_dataset(
            args,
            manifest=manifest,
            output_paths=paths_by_role,
            run_id=build_id if telemetry is None else telemetry.run_id,
            telemetry=telemetry,
        )

    def evaluate() -> None:
        # After the bundle is staged (the evaluation never blocks staging),
        # before the telemetry completes (the receipt rides it as an artifact).
        evidence["evaluation"] = _evaluate_against_incumbent(
            args,
            incumbent=incumbent,
            inputs=inputs,
            output_paths=paths_by_role,
            telemetry=telemetry,
            calibration_year=calibration_year,
            out_dir=out_dir,
        )

    def finalize_staging() -> None:
        publish_manifest()
        evaluate()
        finalize_staging_telemetry(args, telemetry)

    def event_callback(stage_id: str, status: str, details: Mapping[str, Any]) -> None:
        stage(telemetry, stage_id, status, **dict(details))

    result = run_uk_calibration(
        paths=paths,
        build_id=build_id,
        input_sha256=str(args.input_sha256),
        ledger_artifact=inputs["artifact"],
        register_registry=inputs["national_registry"],
        band_edge_registry=inputs["band_edge_registry"],
        calibration_year=calibration_year,
        exclusion_receipt=inputs["measure_exclusions"],
        doctrine=doctrine,
        doctrine_overrides=doctrine_overrides,
        measure_resolver=resolver,
        source_pins={
            "input_h5": {
                "sha256": str(args.input_sha256),
                "size_bytes": int(input_artifact["bytes"]),
            },
            "ledger_facts": _ledger_facts_pin(inputs["artifact"]),
        },
        run_config_extra={
            "release_role": posture.role,
            "calibration_year": calibration_year,
            "allow_unpinned_feed": bool(args.allow_unpinned_feed),
            "chronicle_feed_pin": inputs["chronicle_feed_pin"],
            "rowwise_driver_parameters": rowwise_parameters(
                args, source_year=source_year
            ),
        },
        release_id=posture.release_id,
        logbook_prev_row_digest=args.logbook_prev_row_digest,
        progress_callback=(
            None
            if telemetry is None
            else thinned_epochs(
                telemetry.calibration_progress, every=staging_epoch_every(args)
            )
        ),
        event_callback=None if telemetry is None else event_callback,
        staging_delivery=staging_delivery(telemetry),
        staging_finalizer=None if telemetry is None else finalize_staging,
        staging_delivery_provider=(
            None if telemetry is None else (lambda: telemetry.delivery_summary)
        ),
    )
    if telemetry is None:
        publish_manifest()
        evaluate()
    manifest = evidence["manifest"]
    # The seam rewrote its record with the delivery summary after the
    # finalizer; the manifest binds the record as it now is.
    manifest["outputs"]["build_record"] = _artifact_info(paths.build_record_json)
    manifest["build_record"] = {
        "path": str(paths.build_record_json),
        "sha256": result.build_record_sha256,
    }
    manifest["staging_delivery"] = staging_delivery(telemetry)
    manifest["staged_dataset"] = evidence["staged_dataset"]
    evaluation = evidence.get("evaluation", {"status": "not_requested"})
    manifest["evaluation"] = evaluation
    if evaluation.get("status") == "completed":
        manifest["outputs"]["score_receipt"] = evaluation["receipt"]
    replace_manifest(paths_by_role["manifest"], manifest)
    if (out_dir / SHA256SUMS_FILENAME).is_file():
        # The uploaded copies list the files as uploaded; the local sums
        # list the record, the receipt and the manifest as they now are,
        # evidence included.
        refresh_sha256sums_entry(out_dir, paths.build_record_json.name)
        if evaluation.get("status") == "completed":
            _list_sha256sums_entry(out_dir, paths_by_role["score_receipt"].name)
        refresh_sha256sums_entry(out_dir, paths_by_role["manifest"].name)
    print(json_text(manifest))
    return 0


def _national_fit_by_family(
    diagnostics: Mapping[str, Any], registry: TargetRegistry
) -> list[dict[str, object]]:
    families = {str(spec.name): str(spec.family or "") for spec in registry.specs}
    rows = []
    for row in diagnostics.get("targets", []):
        if not isinstance(row, Mapping):
            continue
        error = row.get("relative_error")
        if error is None:
            continue
        # The diagnostics name a target by its register spec name and, on
        # period-suffixed rows, by a materialized name; the family lives on
        # the spec.
        labels = [
            str(label)
            for label in (row.get("target_name"), row.get("name"))
            if label is not None
        ]
        family = next((families[label] for label in labels if label in families), "")
        rows.append(
            {
                "target_name": labels[0] if labels else "",
                "family": family,
                "abs_relative_error": abs(float(error)),
            }
        )
    if not rows:
        return []
    return uk_fit_by_family(pd.DataFrame(rows))


def _national_manifest(
    args: argparse.Namespace,
    *,
    posture: UKRowwisePosture,
    build_id: str,
    build_record: Mapping[str, Any],
    build_record_path: Path,
    gate_report: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    inputs: Mapping[str, Any],
    doctrine_overrides: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    input_artifact: Mapping[str, Any],
    source_year: int,
) -> dict[str, Any]:
    """The rowwise manifest of a national-role run, over the seam's evidence."""

    calibration = build_record["calibration"]
    solve = calibration["solve"]
    statuses = gate_statuses(gate_report)
    abs_errors = np.asarray(
        [
            abs(float(row["relative_error"]))
            for row in diagnostics.get("targets", [])
            if isinstance(row, Mapping) and row.get("relative_error") is not None
        ],
        dtype=np.float64,
    )
    fit_rows = _national_fit_by_family(diagnostics, inputs["national_registry"])
    outputs = {
        "dataset": _artifact_info(output_paths["dataset"]),
        "calibration_diagnostics": _artifact_info(
            output_paths["calibration_diagnostics"]
        ),
        "build_record": _artifact_info(build_record_path),
        "terminal_gate_report": _artifact_info(output_paths["terminal_gates"]),
        "national_target_registry": _artifact_info(output_paths["national_registry"]),
        "national_contract_registry": _artifact_info(output_paths["contract_registry"]),
    }
    return {
        "schema_version": 4,
        "build_kind": "uk_national_calibrated_candidate",
        "release_role": posture.role,
        "release_id": posture.release_id,
        "build_id": build_id,
        "candidate_scope": "national",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "parameters": rowwise_parameters(args, source_year=source_year),
        "inputs": {"dataset": dict(input_artifact)},
        "identity": {
            "spine": {
                **dict(input_artifact),
                "spine_provenance": dict(build_record["spine_provenance"]),
            },
            "targets": {"chronicle": inputs["artifact"].provenance()},
            "code": {"git_commit": git_commit(), "git_dirty": git_dirty()},
            "runtime": runtime_provenance(),
            "sampling": {"mode": "full"},
            "survey_year": source_year,
            "calibration_year": int(inputs["calibration_year"]),
        },
        "sampling": {"mode": "full"},
        "outputs": outputs,
        "weights": dict(calibration["weights"]),
        "solve": {
            "n_targets": int(solve["n_targets"]),
            "n_targets_by_kind": {
                "national": int(solve["n_targets"]),
                "local": 0,
                "ladder": 0,
            },
            "n_households": int(solve["n_households"]),
            "pool_households": int(solve["n_households"]),
            "initial_loss": float(solve["initial_loss"]),
            "final_loss": float(solve["final_loss"]),
            "n_nonzero": int(solve["n_nonzero"]),
            "max_abs_relative_error": (
                float(abs_errors.max()) if abs_errors.size else None
            ),
            "median_abs_relative_error": (
                float(np.median(abs_errors)) if abs_errors.size else None
            ),
            "effective_sample_size": calibration.get("effective_sample_size"),
            "max_weight_ratio": calibration.get("max_weight_ratio"),
            "target_weight_rule": str(args.target_weight_rule),
            "target_weight_rule_override": dict(
                doctrine_overrides.get("target_weight_rule", {})
            ),
            "doctrine_overrides": dict(doctrine_overrides),
            "measure_resolution": calibration.get("measure_resolution"),
        },
        "fit": {
            "national_by_family": fit_rows,
            "weakest_families": sorted(
                fit_rows,
                key=lambda row: (
                    -float(row["worst_abs_relative_error"]),
                    row["family"],
                ),
            )[:10],
        },
        "gate": {
            "scope": list(posture.gate_scope),
            "posture": posture.gate_posture,
            "release_id": gate_report.get("release_id"),
            "statuses": statuses,
        },
        "failing_gate_ids": sorted(
            gate_id for gate_id, status in statuses.items() if status != "passed"
        ),
        "releasable": False,
        "release_posture": {
            "release_candidate": False,
            "shippable_by": "tools/certify_uk_release_cut.py",
            "calibration_seam_gates_passed": bool(statuses)
            and all(status == "passed" for status in statuses.values()),
        },
        "measure_exclusions": dict(inputs["measure_exclusions"]),
        "build_record": {
            "path": str(build_record_path),
            "sha256": outputs["build_record"]["sha256"],
        },
    }


def _verify_requested_pin(
    label: str,
    artifact: Mapping[str, Any],
    *,
    requested: str | None,
) -> None:
    if requested is None:
        artifact["pin_verified"] = False
        return
    measured = str(artifact["sha256"])
    if measured != requested:
        raise SystemExit(
            f"error: {label} sha mismatch: measured {measured}, pinned {requested}"
        )
    artifact["pin_verified"] = True


def _load_candidate_evaluator(importer=importlib.import_module):
    """The common-surface scorer (microcosm#967), loaded when an incumbent is given.

    Lazy so the driver imports without it; a build asked to evaluate refuses
    up front, before any solve, when the scorer is not in the tree.
    """

    try:
        return importer("microcosm.build.uk_runtime.candidate_score")
    except ImportError as error:
        raise ValueError(
            "--incumbent-h5 needs the common-surface scorer (microcosm#967): "
            "microcosm.build.uk_runtime.candidate_score is not in this tree."
        ) from error


def _incumbent_arguments(args: argparse.Namespace) -> dict[str, Any] | None:
    """The national role's optional incumbent, pinned and verified up front."""

    if args.incumbent_h5 is None and args.incumbent_sha256 is None:
        return None
    if args.incumbent_h5 is None or args.incumbent_sha256 is None:
        raise ValueError(
            "--incumbent-h5 and --incumbent-sha256 must be given together."
        )
    _load_candidate_evaluator()
    incumbent_h5 = _require_file(args.incumbent_h5, label="--incumbent-h5")
    info = _artifact_info(incumbent_h5)
    _verify_requested_pin("--incumbent-h5", info, requested=args.incumbent_sha256)
    return {
        "path": str(incumbent_h5),
        "sha256": info["sha256"],
        "bytes": info["bytes"],
        "label": str(args.incumbent_label),
    }


def _list_sha256sums_entry(out_dir: Path, name: str) -> None:
    """List a file the build wrote after the sidecars in the local sums."""

    sums_path = out_dir / SHA256SUMS_FILENAME
    entries = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    if name in {listed for _, listed in entries}:
        refresh_sha256sums_entry(out_dir, name)
        return
    digest = _artifact_info(out_dir / name)["sha256"]
    sums_path.write_text(
        sums_path.read_text(encoding="utf-8") + f"{digest}  {name}\n",
        encoding="utf-8",
    )


#: Receipt blocks that are record arrays (lists of mappings): the reviewed
#: telemetry artifact policy refuses them, and the verdict, the pruned block
#: and the aggregates carry everything a reviewer reads. The full receipt stays
#: beside the outputs and in the staged bundle.
_SCORE_RECEIPT_TELEMETRY_EXCLUDED = (
    "target_drift",
    "signed_asymmetries",
    "measure_resolution",
)


def _score_receipt_telemetry_summary(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in score.items()
        if key not in _SCORE_RECEIPT_TELEMETRY_EXCLUDED
    }


def _evaluate_against_incumbent(
    args: argparse.Namespace,
    *,
    incumbent: Mapping[str, Any] | None,
    inputs: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    telemetry: StagingTelemetryV2 | None,
    calibration_year: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Score the finished candidate against the incumbent; never fail the build.

    Runs after the staged bundle is on the Hub and before the telemetry
    completes, so the receipt rides the run as a reviewed artifact. Rows the
    incumbent cannot materialize are pruned from both arms and warned about
    by name; the verdict (microcosm#578 rule 1 on the common surface) is
    what the release-cut certifier requires to be ``passed``. An error is
    recorded and warned, never raised: staging and the exit code stand.
    """

    if incumbent is None:
        return {
            "status": "not_requested",
            "note": (
                "no --incumbent-h5: the rule-1 score receipt the release-cut "
                "certifier needs was not produced; score the candidate with "
                "tools/score_uk_national_candidate.py before certification."
            ),
        }
    stage(
        telemetry,
        "incumbent_evaluation",
        "started",
        incumbent_sha256=incumbent["sha256"],
    )
    candidate = _artifact_info(output_paths["dataset"])
    try:
        module = _load_candidate_evaluator()
        score = module.evaluate_uk_candidate_against_incumbent(
            candidate_h5=output_paths["dataset"],
            incumbent_h5=Path(incumbent["path"]),
            candidate_sha256=candidate["sha256"],
            incumbent_sha256=incumbent["sha256"],
            target_registry=inputs["national_registry"],
            calibration_year=calibration_year,
            measure_resolver_factory=module.uk_default_measure_resolver_factory(
                out_dir, calibration_year
            ),
            candidate_label=output_paths["dataset"].stem,
            incumbent_label=incumbent["label"],
            band_edge_registry=inputs["band_edge_registry"],
        )
        atomic_write_json(output_paths["score_receipt"], score)
    except Exception as error:  # noqa: BLE001 - the evaluation never fails a finished build
        message = f"{type(error).__name__}: {error}"[:600]
        print(
            f"warning: the incumbent evaluation failed ({message}); the build's "
            "evidence and staging are unaffected, and the candidate cannot be "
            "certified until it is re-scored with "
            "tools/score_uk_national_candidate.py.",
            file=sys.stderr,
            flush=True,
        )
        stage(telemetry, "incumbent_evaluation", "failed", error=message)
        return {
            "status": "error",
            "error": message,
            "incumbent": dict(incumbent),
            "receipt": None,
        }
    warning = module.pruned_warning(score)
    if warning is not None:
        print(warning, file=sys.stderr, flush=True)
    evaluation = score["evaluation"]
    pruned = score["incumbent_unresolvable_pruned"]
    add_staging_artifact(
        telemetry,
        "score_vs_incumbent",
        _score_receipt_telemetry_summary(score),
        artifact_kind="aggregate_diagnostics",
        classification="aggregate",
    )
    stage(
        telemetry,
        "incumbent_evaluation",
        "completed",
        verdict=evaluation["verdict"],
        n_scored=int(evaluation["scored_surface"]["n_scored"]),
        n_pruned=int(evaluation["scored_surface"]["n_pruned"]),
    )
    return {
        "status": "completed",
        "verdict": evaluation["verdict"],
        "rule_1": dict(evaluation["rule_1"]),
        "scored_surface": dict(evaluation["scored_surface"]),
        "pruned_measures": list(pruned["measures"]),
        "pruned_families": dict(pruned["families"]),
        "receipt": _artifact_info(output_paths["score_receipt"]),
        "incumbent": dict(incumbent),
    }


def _validate_output_paths(
    output_paths: Mapping[str, Path],
    *,
    input_h5: Path,
    ladder_path: Path | None,
) -> None:
    resolved = {name: path.resolve() for name, path in output_paths.items()}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("candidate output paths must be distinct.")
    protected = {input_h5.resolve()}
    if ladder_path is not None:
        protected.add(ladder_path.resolve())
    collisions = sorted(str(path) for path in resolved.values() if path in protected)
    if collisions:
        raise ValueError(
            "candidate outputs must differ from --input-h5 and --ladder; "
            f"collision(s): {collisions}."
        )
    existing = sorted(str(path) for path in resolved.values() if path.exists())
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing candidate artifact(s): {existing}."
        )


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} artifact not found: {resolved}.")
    return resolved


def _source_year(requested: int | None, *, time_period: str) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("--source-year must be positive.")
        return requested
    prefix = str(time_period).strip()[:4]
    if len(prefix) != 4 or not prefix.isdigit():
        raise ValueError(
            "Could not infer source year from input H5 time_period; pass --source-year."
        )
    return int(prefix)


def _artifact_info(
    path: Path,
    *,
    reported_path: Path | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str((reported_path or path).resolve()),
        "sha256": digest.hexdigest(),
        "bytes": int(path.stat().st_size),
    }
