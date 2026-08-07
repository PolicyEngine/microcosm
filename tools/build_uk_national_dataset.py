"""Build the national UK staging file with the guarded HMRC/SPI family."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from itertools import combinations
from pathlib import Path

import pandas as pd

from microcosm.build.uk_runtime.frs_hmrc_leaves import (
    UKFRSHMRCRetainedLeavesStageTransform,
)
from microcosm.build.uk_runtime.hmrc_replay import write_hmrc_replay_report
from microcosm.build.uk_runtime.hmrc_restoration import (
    UKHMRCIncomeStageTransform,
    verify_certified_uk_candidate,
)
from microcosm.build.uk_runtime.national_build import (
    UKNationalStage,
    build_uk_national_dataset,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_time_period,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
    UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE,
    UKInputMassParityPolicy,
    UKQRFTailConcentrationPolicy,
    load_uk_input_mass_reference,
    load_uk_reviewed_exclusion_register,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="Compact UK single-year H5 supplying the national base tables.",
    )
    parser.add_argument(
        "--staging-h5",
        type=Path,
        required=True,
        help="Caller-owned path for the gated national staging H5.",
    )
    parser.add_argument(
        "--release-id",
        required=True,
        help="Canonical release id to bind into the signed terminal report.",
    )
    parser.add_argument(
        "--calibration-diagnostics-sha256",
        required=True,
        help=(
            "Lowercase SHA-256 of the exact calibration_diagnostics.json bytes "
            "that will ship with this release."
        ),
    )
    parser.add_argument(
        "--frs-raw-dir",
        type=Path,
        required=True,
        help=(
            "Raw FRS 2023-24 directory containing adult.tab and benefits.tab "
            "for source-faithful retained HMRC leaves."
        ),
    )
    parser.add_argument(
        "--spi-tab",
        type=Path,
        required=True,
        help="Licensed UKDS SPI 2022-23 donor named put2223uk.tab.",
    )
    parser.add_argument(
        "--hmrc-ods",
        type=Path,
        required=True,
        help="Official HMRC Personal Incomes 2023-24 collated ODS.",
    )
    gate_output = parser.add_mutually_exclusive_group()
    gate_output.add_argument(
        "--terminal-gates-json",
        type=Path,
        help=(
            "Consolidated terminal-gate report path. Defaults beside "
            "--staging-h5 with suffix '.terminal_gates.json'."
        ),
    )
    gate_output.add_argument(
        "--input-coverage-json",
        type=Path,
        help=(
            "Legacy schema-1 input-coverage diagnostic path. Cannot be "
            "supplied with the preferred terminal-gate option."
        ),
    )
    parser.add_argument(
        "--hmrc-evidence-json",
        type=Path,
        help=(
            "HMRC stage evidence path. Defaults beside --staging-h5 with "
            "suffix '.hmrc_income.json'."
        ),
    )
    parser.add_argument(
        "--hmrc-replay-json",
        type=Path,
        help=(
            "Aggregate-only 208-fact replay report path. Defaults beside "
            "--staging-h5 with suffix '.hmrc_replay.json'."
        ),
    )
    parser.add_argument(
        "--build-record-json",
        type=Path,
        help=(
            "Aggregate, path-free staging build record. Defaults beside "
            "--staging-h5 with suffix '.build.json'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qrf-estimators", type=int, default=100)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Persist a lossless Frame checkpoint at every stage boundary and "
            "resume completed stages from it (#612 increment 3). The run is "
            "pinned by a content-addressed run config (input digest, seed, "
            "QRF estimators, raw-source digests); rerunning the same command "
            "against the same directory resumes, a changed configuration is "
            "refused. Omit for the destructive single-process build."
        ),
    )
    parser.add_argument(
        "--input-mass-reference-json",
        type=Path,
        help=(
            "Frozen weighted per-column reference totals (schema_version 1: "
            "identity + totals) emitted by the #609 measurement tooling. "
            "Supplying it arms the input_mass_parity terminal gate; both "
            "--input-mass-* thresholds are then required."
        ),
    )
    parser.add_argument(
        "--input-mass-relative-tolerance",
        type=float,
        help=(
            "Maximum |candidate - reference| / |reference| per column before "
            "input_mass_parity fails. No default: the boundary comes from the "
            "#609 measurement pass, not from the US 0.5."
        ),
    )
    parser.add_argument(
        "--input-mass-minimum-reference-total",
        type=float,
        help=(
            "Reference-mass floor (GBP-scale) below which a column is not "
            "checked. No default: the US 1e9 is a USD figure against a "
            "different pool and must not be inherited (#609)."
        ),
    )
    parser.add_argument(
        "--input-mass-exclusions",
        type=Path,
        help=(
            "Reviewed input-mass exclusion register overriding the committed "
            f"{UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE}. Stale entries fail "
            "the gate; dormant entries are reported."
        ),
    )
    parser.add_argument(
        "--qrf-tail-top-k",
        type=int,
        help=(
            "Tail size for the qrf_tail_concentration terminal gate. "
            "Supplying the three --qrf-tail-* thresholds together arms the "
            "gate; no defaults are inherited from the US #462 calibration."
        ),
    )
    parser.add_argument(
        "--qrf-tail-max-top-share",
        type=float,
        help=(
            "Blocking share of weighted |mass| the top-k records may carry "
            "per declared QRF output, in (0, 1). Measured per #609."
        ),
    )
    parser.add_argument(
        "--qrf-tail-min-nonzero-records",
        type=int,
        help=(
            "Columns with fewer weighted carriers are reported as thin and "
            "not checked; must exceed --qrf-tail-top-k. Measured per #609."
        ),
    )
    parser.add_argument(
        "--qrf-tail-exclusions",
        type=Path,
        help=(
            "Reviewed QRF tail-concentration exclusion register overriding "
            f"the committed {UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE}. Stale "
            "entries fail the gate; dormant entries are reported."
        ),
    )
    return parser.parse_args()


def _weighted_integrity_arguments(args: argparse.Namespace) -> dict[str, object]:
    """Assemble the increment-4 gate arguments, requiring complete arming."""

    def parser_error(message: str) -> None:
        raise SystemExit(f"error: {message}")

    arguments: dict[str, object] = {}
    input_mass_thresholds = (
        args.input_mass_relative_tolerance,
        args.input_mass_minimum_reference_total,
    )
    input_mass_requested = args.input_mass_reference_json is not None or any(
        value is not None for value in input_mass_thresholds
    )
    if input_mass_requested:
        if args.input_mass_reference_json is None or any(
            value is None for value in input_mass_thresholds
        ):
            parser_error(
                "arming input_mass_parity requires --input-mass-reference-json, "
                "--input-mass-relative-tolerance, and "
                "--input-mass-minimum-reference-total together."
            )
        arguments["input_mass_reference"] = load_uk_input_mass_reference(
            args.input_mass_reference_json
        )
        arguments["input_mass_policy"] = UKInputMassParityPolicy(
            relative_tolerance=args.input_mass_relative_tolerance,
            minimum_reference_total=args.input_mass_minimum_reference_total,
            reviewed_exclusions=load_uk_reviewed_exclusion_register(
                args.input_mass_exclusions,
                resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
            ),
        )
    elif args.input_mass_exclusions is not None:
        parser_error(
            "--input-mass-exclusions requires the input_mass_parity gate to be armed."
        )
    qrf_thresholds = (
        args.qrf_tail_top_k,
        args.qrf_tail_max_top_share,
        args.qrf_tail_min_nonzero_records,
    )
    if any(value is not None for value in qrf_thresholds):
        if any(value is None for value in qrf_thresholds):
            parser_error(
                "arming qrf_tail_concentration requires --qrf-tail-top-k, "
                "--qrf-tail-max-top-share, and --qrf-tail-min-nonzero-records "
                "together."
            )
        arguments["qrf_tail_policy"] = UKQRFTailConcentrationPolicy(
            top_k=args.qrf_tail_top_k,
            max_top_share=args.qrf_tail_max_top_share,
            min_nonzero_records=args.qrf_tail_min_nonzero_records,
            reviewed_exclusions=load_uk_reviewed_exclusion_register(
                args.qrf_tail_exclusions,
                resource=UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE,
            ),
        )
    elif args.qrf_tail_exclusions is not None:
        parser_error(
            "--qrf-tail-exclusions requires the qrf_tail_concentration gate "
            "to be armed."
        )
    return arguments


def main() -> int:
    args = _parse_args()
    legacy_input_coverage_path = args.input_coverage_json
    terminal_gate_path = (
        None
        if legacy_input_coverage_path is not None
        else (
            args.terminal_gates_json
            or args.staging_h5.with_suffix(".terminal_gates.json")
        )
    )
    gate_output_path = legacy_input_coverage_path or terminal_gate_path
    assert gate_output_path is not None
    evidence_path = args.hmrc_evidence_json or args.staging_h5.with_suffix(
        ".hmrc_income.json"
    )
    replay_path = args.hmrc_replay_json or args.staging_h5.with_suffix(
        ".hmrc_replay.json"
    )
    build_record_path = args.build_record_json or args.staging_h5.with_suffix(
        ".build.json"
    )
    retained_leaves_transform = (
        UKFRSHMRCRetainedLeavesStageTransform.from_raw_frs_directory(args.frs_raw_dir)
    )
    _validate_distinct_paths(
        evidence_path=evidence_path,
        replay_path=replay_path,
        terminal_gate_path=gate_output_path,
        input_h5=args.input_h5,
        staging_h5=args.staging_h5,
        spi_tab=args.spi_tab,
        hmrc_ods=args.hmrc_ods,
        adult_tab=retained_leaves_transform.adult_tab_path,
        benefits_tab=retained_leaves_transform.benefits_tab_path,
        build_record_path=build_record_path,
        input_mass_reference_path=args.input_mass_reference_json,
        input_mass_exclusions_path=args.input_mass_exclusions,
        qrf_tail_exclusions_path=args.qrf_tail_exclusions,
    )
    # Read-only gate inputs are materialized before any sidecar unlink so a
    # path collision cannot consume a just-deleted file.
    weighted_integrity_arguments = _weighted_integrity_arguments(args)
    candidate = verify_certified_uk_candidate(args.input_h5)
    evidence_path.unlink(missing_ok=True)
    replay_path.unlink(missing_ok=True)
    build_record_path.unlink(missing_ok=True)
    hmrc_transform = UKHMRCIncomeStageTransform(
        spi_tab_path=args.spi_tab,
        hmrc_ods_path=args.hmrc_ods,
        certified_candidate=candidate,
        retained_leaves_transform=retained_leaves_transform,
        seed=args.seed,
        qrf_estimators=args.qrf_estimators,
    )
    try:
        # This staging path performs no calibration and therefore has no real
        # target-surface or target-fit evidence. Leave parity_evidence absent;
        # the terminal report omits that trio instead of inventing passes.
        # The weighted-integrity pair (#609) follows the same rule: it joins
        # the battery only when the caller arms it with a frozen reference
        # and measured thresholds.
        gate_path_argument = (
            {"input_coverage_path": legacy_input_coverage_path}
            if legacy_input_coverage_path is not None
            else {"terminal_gate_path": terminal_gate_path}
        )
        checkpoint_arguments: dict[str, object] = {}
        if args.checkpoint_dir is not None:
            checkpoint_arguments = {
                "checkpoint_dir": args.checkpoint_dir,
                "run_config": _staging_run_config(
                    args,
                    candidate=candidate,
                    retained_leaves_transform=retained_leaves_transform,
                    hmrc_transform=hmrc_transform,
                ),
            }
        result = build_uk_national_dataset(
            input_h5=args.input_h5,
            staging_h5=args.staging_h5,
            release_id=args.release_id,
            calibration_diagnostics_sha256=args.calibration_diagnostics_sha256,
            stages=(
                UKNationalStage(
                    name="frs_hmrc_retained_leaves",
                    transform=retained_leaves_transform,
                ),
                UKNationalStage(
                    name="hmrc_spi_income",
                    transform=hmrc_transform,
                ),
            ),
            **gate_path_argument,
            **weighted_integrity_arguments,
            **checkpoint_arguments,
        )
    except RuntimeError as error:
        if (
            _is_final_release_gate_failure(error)
            and retained_leaves_transform.last_result is not None
            and hmrc_transform.last_result is not None
        ):
            _write_stage_reports(
                evidence_path=evidence_path,
                replay_path=replay_path,
                candidate=candidate,
                retained_leaves_transform=retained_leaves_transform,
                hmrc_transform=hmrc_transform,
            )
        raise
    _write_stage_reports(
        evidence_path=evidence_path,
        replay_path=replay_path,
        candidate=candidate,
        retained_leaves_transform=retained_leaves_transform,
        hmrc_transform=hmrc_transform,
    )
    assert hmrc_transform.last_result is not None  # guarded by report writer
    hmrc_evidence = {
        "passed": True,
        "summary": _replay_summary(hmrc_transform.last_result),
    }
    artifact_paths = {
        "input_h5": result.input_h5,
        "staging_h5": result.staging_h5,
        "terminal_gates": gate_output_path,
        "hmrc_evidence": evidence_path,
        "hmrc_replay": replay_path,
        "spi_donor": args.spi_tab,
        "hmrc_surface": args.hmrc_ods,
        "frs_adult": retained_leaves_transform.adult_tab_path,
        "frs_benefits": retained_leaves_transform.benefits_tab_path,
    }
    artifacts = {role: _artifact_info(path) for role, path in artifact_paths.items()}
    build_record = _aggregate_build_record(
        result=result,
        artifacts=artifacts,
        retained_evidence=retained_leaves_transform.last_result.evidence(),
        family_evidence=hmrc_transform.last_result.evidence(),
        seed=args.seed,
        qrf_estimators=args.qrf_estimators,
    )
    _write_json(build_record_path, build_record)
    payload = {
        "schema_version": 4,
        "build_kind": "uk_national_staging_dataset",
        "stages": list(result.stage_names),
        "terminal_gates": result.terminal_gates.to_manifest(),
        "input_coverage": {
            "passed": result.input_coverage.passed,
            "failures": list(result.input_coverage.failures),
            "details": dict(result.input_coverage.details),
        },
        "artifacts": {
            **artifacts,
            "build_record": _artifact_info(build_record_path),
        },
        "hmrc_replay": hmrc_evidence,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _staging_run_config(
    args: argparse.Namespace,
    *,
    candidate: object,
    retained_leaves_transform: UKFRSHMRCRetainedLeavesStageTransform,
    hmrc_transform: UKHMRCIncomeStageTransform,
) -> dict[str, object]:
    """The content-addressed identity of a checkpointed staging run.

    Everything that determines the stage outputs is pinned by content, not
    by path or stat: the verified certified-candidate digest, the raw-source
    digests (read from the transforms' own resolved paths, so the pinned
    files are exactly the files the stages consume), the seeds, the release
    coordinates, and the builder code identity (packaged sources plus the
    numeric-dependency versions). The stage runtime refuses to resume a
    checkpoint directory under a different config, so a drifted input,
    parameter, code change, or environment upgrade can never blend into an
    old run's prefix.
    """

    from microcosm.build.code_identity import builder_code_identity

    def _digest(path: str | Path) -> dict[str, object]:
        info = _artifact_info(path)
        return {"sha256": info["sha256"], "size_bytes": info["size_bytes"]}

    return {
        "build_kind": "uk_national_staging_dataset",
        "release_id": str(args.release_id),
        "calibration_diagnostics_sha256": str(args.calibration_diagnostics_sha256),
        "seed": int(args.seed),
        "qrf_estimators": int(args.qrf_estimators),
        "certified_candidate": {
            "sha256": str(candidate.sha256),
            "size_bytes": int(candidate.size_bytes),
        },
        "sources": {
            "adult_tab": _digest(retained_leaves_transform.adult_tab_path),
            "benefits_tab": _digest(retained_leaves_transform.benefits_tab_path),
            "spi_tab": _digest(hmrc_transform.spi_tab_path),
            "hmrc_ods": _digest(hmrc_transform.hmrc_ods_path),
        },
        "code_identity": builder_code_identity(
            Path(__file__).resolve().parents[1],
            tool_path=Path(__file__).resolve(),
            distributions=(
                "h5py",
                "numpy",
                "pandas",
                "quantile-forest",
                "scikit-learn",
                "tables",
            ),
        ),
    }


def _artifact_info(path: str | Path) -> dict[str, str | int]:
    artifact = Path(path).resolve()
    digest = hashlib.sha256()
    with artifact.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(artifact),
        "sha256": digest.hexdigest(),
        "size_bytes": artifact.stat().st_size,
    }


def _aggregate_build_record(
    *,
    result: object,
    artifacts: dict[str, dict[str, str | int]],
    retained_evidence: dict[str, object],
    family_evidence: dict[str, object],
    seed: int,
    qrf_estimators: int,
) -> dict[str, object]:
    """Return commit-safe aggregate evidence for one successful staging build."""

    details = dict(result.input_coverage.details)
    safe_artifacts = {
        role: {
            "sha256": str(info["sha256"]),
            "size_bytes": int(info["size_bytes"]),
        }
        for role, info in artifacts.items()
    }
    safe_artifacts["staging_h5"]["retention"] = "local_untracked"
    retained_sources = dict(retained_evidence.get("sources", {}))
    family_sources = dict(family_evidence.get("sources", {}))
    mass_changes = [
        {
            "entity": record.entity,
            "old_total": float(record.old_total),
            "new_total": float(record.new_total),
            "declared_factor": (
                None
                if record.declared_factor is None
                else float(record.declared_factor)
            ),
            "reason": record.reason,
        }
        for record in result.frame.mass_log
    ]
    household_weights = pd.to_numeric(
        result.frame.table("household")["household_weight"], errors="raise"
    )
    return {
        "schema_version": 2,
        "build_kind": "uk_national_staging_dataset",
        "status": "passed",
        "stages": list(result.stage_names),
        "parameters": {
            "seed": int(seed),
            "qrf_estimators": int(qrf_estimators),
        },
        "dataset": {
            "time_period": uk_time_period(result.frame),
            "entity_rows": {
                "person": len(result.frame.table("person")),
                "benunit": len(result.frame.table("benunit")),
                "household": len(result.frame.table("household")),
            },
            "household_weight_kind": uk_household_weight_kind(result.frame).value,
            "household_weight_total": float(household_weights.sum()),
            "mass_changes": mass_changes,
        },
        "source_rows": {
            "frs_adult": int(dict(retained_sources.get("adult", {})).get("rows", 0)),
            "frs_benefits": int(
                dict(retained_sources.get("benefits", {})).get("rows", 0)
            ),
            "spi_donor_used": int(
                dict(family_sources.get("spi_donor", {})).get("rows_used", 0)
            ),
        },
        "source_vintages": dict(family_evidence.get("source_vintages", {})),
        "terminal_gates": result.terminal_gates.to_manifest(),
        "input_coverage": {
            "passed": bool(result.input_coverage.passed),
            "failures": list(result.input_coverage.failures),
            "required_columns": int(details.get("required_columns", 0)),
            "reviewed_exclusion_columns": len(
                dict(details.get("reviewed_exclusions", {}))
            ),
            "missing": list(details.get("missing", ())),
            "degenerate_required": list(details.get("degenerate_required", ())),
            "insufficient_effective_mass": list(
                details.get("insufficient_effective_mass", ())
            ),
            "stale_exclusions": list(details.get("stale_exclusions", ())),
            "effective_mass_policy": dict(details.get("effective_mass_policy", {})),
            "family_effective_mass": dict(details.get("family_effective_mass", {})),
            "family_build_state": dict(details.get("family_build_state", {})),
        },
        "hmrc_replay": {
            "summary": dict(
                dict(family_evidence.get("targets", {})).get("classification", {})
            ),
            "post_draw_identity": dict(family_evidence.get("post_draw_identity", {})),
        },
        "artifacts": safe_artifacts,
    }


def _write_json(path: str | Path, payload: dict[str, object]) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_stage_reports(
    *,
    evidence_path: Path,
    replay_path: Path,
    candidate: object,
    retained_leaves_transform: UKFRSHMRCRetainedLeavesStageTransform,
    hmrc_transform: UKHMRCIncomeStageTransform,
) -> None:
    retained_result = retained_leaves_transform.last_result
    hmrc_result = hmrc_transform.last_result
    if retained_result is None or hmrc_result is None:
        raise RuntimeError(
            "UK national HMRC stages did not both complete; refusing to write "
            "partial or stale aggregate evidence."
        )
    payload = {
        "schema_version": 2,
        "base_candidate": {
            "path": str(candidate.path),
            "filename": candidate.filename,
            "tier": candidate.tier,
            "revision": candidate.revision,
            "sha256": candidate.sha256,
            "size_bytes": candidate.size_bytes,
        },
        "retained_leaves": retained_result.evidence(),
        "family": hmrc_result.evidence(),
    }
    _write_json(evidence_path, payload)
    # A checkpoint-resumed SPI stage carries no report object, only the
    # payload its real report produced at completion time; _write_json and
    # write_hmrc_replay_report share the exact serialization (indent=2,
    # sort_keys, trailing newline), so the resumed sidecar is byte-identical.
    replay_report = getattr(hmrc_result, "replay_report", None)
    if replay_report is not None:
        write_hmrc_replay_report(replay_report, replay_path)
    else:
        _write_json(replay_path, dict(_resumed_replay_payload(hmrc_result)))


def _resumed_replay_payload(hmrc_result: object) -> dict[str, object]:
    payload = getattr(hmrc_result, "replay_payload", None)
    if not isinstance(payload, dict):
        raise RuntimeError(
            "resumed SPI restoration carries no replay payload; the "
            "checkpoint record cannot feed the driver's stage reports."
        )
    return payload


def _replay_summary(hmrc_result: object) -> dict[str, object]:
    report = getattr(hmrc_result, "replay_report", None)
    if report is not None:
        return dict(report.summary)
    summary = _resumed_replay_payload(hmrc_result).get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("resumed SPI replay payload carries no summary block.")
    return dict(summary)


def _is_final_release_gate_failure(error: RuntimeError) -> bool:
    """Match only the national seam's post-stage, pre-staging hard gate."""

    return str(error).startswith("Release gates failed:")


def _validate_distinct_paths(
    *,
    evidence_path: Path,
    replay_path: Path,
    terminal_gate_path: Path,
    input_h5: Path,
    staging_h5: Path,
    spi_tab: Path,
    hmrc_ods: Path,
    adult_tab: Path,
    benefits_tab: Path,
    build_record_path: Path,
    input_mass_reference_path: Path | None,
    input_mass_exclusions_path: Path | None,
    qrf_tail_exclusions_path: Path | None,
) -> None:
    paths = {
        "--input-h5": input_h5.resolve(),
        "--staging-h5": staging_h5.resolve(),
        "--spi-tab": spi_tab.resolve(),
        "--hmrc-ods": hmrc_ods.resolve(),
        "--frs-raw-dir/adult.tab": adult_tab.resolve(),
        "--frs-raw-dir/benefits.tab": benefits_tab.resolve(),
        "--build-record-json": build_record_path.resolve(),
        "--terminal-gates-json/--input-coverage-json": terminal_gate_path.resolve(),
        "--hmrc-evidence-json": evidence_path.resolve(),
        "--hmrc-replay-json": replay_path.resolve(),
    }
    paths.update(
        (label, path.resolve())
        for label, path in {
            "--input-mass-reference-json": input_mass_reference_path,
            "--input-mass-exclusions": input_mass_exclusions_path,
            "--qrf-tail-exclusions": qrf_tail_exclusions_path,
        }.items()
        if path is not None
    )
    collisions = [
        (left_label, right_label, left_path, right_path)
        for (left_label, left_path), (right_label, right_path) in combinations(
            paths.items(),
            2,
        )
        if _paths_alias(left_path, right_path)
    ]
    if collisions:
        details = "; ".join(
            f"{left_label}, {right_label} -> {left_path} == {right_path}"
            for left_label, right_label, left_path, right_path in collisions
        )
        raise ValueError(
            "UK national build input, staging, and sidecar paths must be "
            f"pairwise distinct: {details}."
        )


def _paths_alias(left: Path, right: Path) -> bool:
    """Conservatively identify aliases before any build path is unlinked."""

    if left == right:
        return True
    try:
        if left.exists() and right.exists() and left.samefile(right):
            return True
    except OSError:
        # The case-folded resolved identity below remains a safe fallback for
        # a path that changes between the existence and samefile checks.
        pass
    # macOS volumes are commonly case-insensitive even though Path.resolve()
    # preserves caller casing. Reject case-only distinctions everywhere: a
    # destructive build tool has no legitimate need for them, and doing so
    # also protects outputs that do not exist yet.
    return str(left).casefold() == str(right).casefold()


if __name__ == "__main__":
    raise SystemExit(main())
