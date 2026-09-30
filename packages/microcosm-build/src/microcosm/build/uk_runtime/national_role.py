"""The UK build driver's national release role (microcosm#823): evidence shapes.

``microcosm-build-uk --release-role national`` builds the certified national
line on the shared executable graph (:mod:`.graph_national`; the driver's
national path lives in :mod:`.full_build_cli`). This module holds what the
role adds around the graph: the dry-run plan, the receipted doctrine
overrides, the schema-4 national manifest over the seam-shaped evidence, the
optional end-of-build evaluation against the incumbent (microcosm#578 rule 1
on the common surface), the pinned-input and output-path checks and the
local sums. These functions were moved from
``tools/build_uk_rowwise_candidate.py`` (microcosm#901 phase 4) and consume
only the shared rowwise command surface (:mod:`.rowwise_cli`,
:mod:`.rowwise_staging`) and package APIs; the private spellings are kept
as aliases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.logbook_adoption import atomic_write_json
from microcosm.build.staging_dataset import (
    SHA256SUMS_FILENAME,
    parse_sha256sums,
    refresh_sha256sums_entry,
)
from microcosm.build.staging_v2 import StagingTelemetryV2
from microcosm.build.uk_runtime.calibration_run import runtime_provenance
from microcosm.build.uk_runtime.diagnostics import uk_fit_by_family
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.full_targets import load_uk_national_target_inputs
from microcosm.build.uk_runtime.national_doctrine import uk_doctrine_with_overrides
from microcosm.build.uk_runtime.rowwise_cli import (
    git_commit,
    git_dirty,
    json_text,
    posture_of,
    rowwise_parameters,
)
from microcosm.build.uk_runtime.rowwise_posture import UKRowwisePosture
from microcosm.build.uk_runtime.rowwise_staging import (
    add_staging_artifact,
    gate_statuses,
    stage,
)
from microcosm.calibrate import TargetRegistry

__all__ = [
    "evaluate_against_incumbent",
    "incumbent_arguments",
    "national_doctrine_overrides",
    "national_dry_run",
    "national_manifest",
    "require_bound_input",
]


def load_national_target_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """The national role's target surface, compiled as the graph node compiles it.

    The dry run plans over it; a build compiles inside
    ``uk.full.national_targets`` with the same loader and records the result
    as the registry artifact.
    """

    return load_uk_national_target_inputs(
        args.ledger_facts,
        expected_facts_sha256=args.ledger_facts_sha256,
        expected_manifest_sha256=args.ledger_manifest_sha256,
        measure_exclusions=args.measure_exclusions,
        register_json=args.register_json,
        calibration_year=int(load_uk_frs_release().calibration_year),
        exclusions_evaluated_on=args.review_date,
        allow_unpinned_feed=bool(args.allow_unpinned_feed),
    )


_load_national_target_inputs = load_national_target_inputs


def national_doctrine_overrides(args: argparse.Namespace) -> dict[str, Any]:
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


_national_doctrine_overrides = national_doctrine_overrides


def require_bound_input(args: argparse.Namespace) -> None:
    """The national role builds from a bound spine checkpoint, never a request.

    The graph driver admits ``--spine-request`` as the dense build's
    population source; the national line solves a pinned ``--input-h5``
    checkpoint as bound.
    """

    if args.input_h5 is None:
        raise ValueError(
            "--release-role national builds from a bound spine checkpoint: pass "
            "--input-h5 with --input-sha256 (--spine-request executes the spine "
            "stages only in the graph dense build)."
        )


_require_bound_input = require_bound_input


def national_dry_run(
    args: argparse.Namespace,
    *,
    operation_inventory: Callable[[], Mapping[str, Any]] | None = None,
) -> int:
    """Compile the national target surface and print the plan; write nothing.

    ``operation_inventory`` (the driver's) adds the compiled graph's node
    inventory to the plan.
    """

    posture = posture_of(args)
    input_h5 = require_file(args.input_h5, label="--input-h5")
    input_artifact = artifact_info(input_h5)
    verify_requested_pin("--input-h5", input_artifact, requested=args.input_sha256)
    frs_release = load_uk_frs_release()
    inputs = load_national_target_inputs(args)
    doctrine, doctrine_overrides = uk_doctrine_with_overrides(
        **national_doctrine_overrides(args)
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
            "chronicle": inputs["chronicle_provenance"],
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
            source_year=source_year(
                args.source_year, time_period=str(frs_release.time_period)
            ),
        ),
        "engine": "not_run",
        "incumbent": incumbent_arguments(args),
        "releasable": False,
    }
    if operation_inventory is not None:
        plan["graph"] = dict(operation_inventory())
    print(json_text(plan))
    return 0


_national_dry_run = national_dry_run


def national_fit_by_family(
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


_national_fit_by_family = national_fit_by_family


def national_manifest(
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
    reported_paths: Mapping[str, Path] | None = None,
    graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The rowwise manifest of a national-role run, over the seam-shaped evidence.

    ``inputs`` carries the compiled surface (``national_registry``,
    ``band_edge_registry``, ``calibration_year``, ``measure_exclusions``,
    ``chronicle_provenance``); ``reported_paths`` names the published
    locations when the files were written to a staging directory first;
    ``graph`` names the graph artifacts the evidence stood on.
    """

    reported = dict(output_paths) if reported_paths is None else dict(reported_paths)
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
    fit_rows = national_fit_by_family(diagnostics, inputs["national_registry"])

    def output(role: str) -> dict[str, Any]:
        return artifact_info(output_paths[role], reported_path=reported[role])

    outputs = {
        "dataset": output("dataset"),
        "calibration_diagnostics": output("calibration_diagnostics"),
        "build_record": artifact_info(
            build_record_path, reported_path=reported["build_record"]
        ),
        "terminal_gate_report": output("terminal_gates"),
        "national_target_registry": output("national_registry"),
        "national_contract_registry": output("contract_registry"),
    }
    manifest = {
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
            "targets": {"chronicle": dict(inputs["chronicle_provenance"])},
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
            "path": str(reported["build_record"]),
            "sha256": outputs["build_record"]["sha256"],
        },
    }
    if graph is not None:
        manifest["graph"] = dict(graph)
    return manifest


_national_manifest = national_manifest


def verify_requested_pin(
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


_verify_requested_pin = verify_requested_pin


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


def incumbent_arguments(args: argparse.Namespace) -> dict[str, Any] | None:
    """The national role's optional incumbent, pinned and verified up front."""

    if args.incumbent_h5 is None and args.incumbent_sha256 is None:
        return None
    if args.incumbent_h5 is None or args.incumbent_sha256 is None:
        raise ValueError(
            "--incumbent-h5 and --incumbent-sha256 must be given together."
        )
    _load_candidate_evaluator()
    incumbent_h5 = require_file(args.incumbent_h5, label="--incumbent-h5")
    info = artifact_info(incumbent_h5)
    verify_requested_pin("--incumbent-h5", info, requested=args.incumbent_sha256)
    return {
        "path": str(incumbent_h5),
        "sha256": info["sha256"],
        "bytes": info["bytes"],
        "label": str(args.incumbent_label),
    }


_incumbent_arguments = incumbent_arguments


def list_sha256sums_entry(out_dir: Path, name: str) -> None:
    """List a file the build wrote after the sidecars in the local sums."""

    sums_path = out_dir / SHA256SUMS_FILENAME
    entries = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    if name in {listed for _, listed in entries}:
        refresh_sha256sums_entry(out_dir, name)
        return
    digest = artifact_info(out_dir / name)["sha256"]
    sums_path.write_text(
        sums_path.read_text(encoding="utf-8") + f"{digest}  {name}\n",
        encoding="utf-8",
    )


_list_sha256sums_entry = list_sha256sums_entry


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


def evaluate_against_incumbent(
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
    candidate = artifact_info(output_paths["dataset"])
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
        "receipt": artifact_info(output_paths["score_receipt"]),
        "incumbent": dict(incumbent),
    }


_evaluate_against_incumbent = evaluate_against_incumbent


def validate_output_paths(
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


_validate_output_paths = validate_output_paths


def require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} artifact not found: {resolved}.")
    return resolved


_require_file = require_file


def source_year(requested: int | None, *, time_period: str) -> int:
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


_source_year = source_year


def artifact_info(
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


_artifact_info = artifact_info
