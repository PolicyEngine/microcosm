"""Build the national UK staging file with the guarded HMRC/SPI family."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from itertools import combinations
from pathlib import Path

import pandas as pd

from populace.build.uk_runtime.frs_hmrc_leaves import (
    UKFRSHMRCRetainedLeavesStageTransform,
)
from populace.build.uk_runtime.hmrc_replay import write_hmrc_replay_report
from populace.build.uk_runtime.hmrc_restoration import (
    UKHMRCIncomeStageTransform,
    verify_certified_uk_candidate,
)
from populace.build.uk_runtime.national_build import (
    UKNationalStage,
    build_uk_national_dataset,
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
    parser.add_argument(
        "--input-coverage-json",
        type=Path,
        help=(
            "Coverage diagnostic path. Defaults beside --staging-h5 with "
            "suffix '.input_coverage.json'."
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    coverage_path = args.input_coverage_json or args.staging_h5.with_suffix(
        ".input_coverage.json"
    )
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
        coverage_path=coverage_path,
        input_h5=args.input_h5,
        staging_h5=args.staging_h5,
        spi_tab=args.spi_tab,
        hmrc_ods=args.hmrc_ods,
        adult_tab=retained_leaves_transform.adult_tab_path,
        benefits_tab=retained_leaves_transform.benefits_tab_path,
        build_record_path=build_record_path,
    )
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
        result = build_uk_national_dataset(
            input_h5=args.input_h5,
            staging_h5=args.staging_h5,
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
            input_coverage_path=coverage_path,
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
        "summary": dict(hmrc_transform.last_result.replay_report.summary),
    }
    artifact_paths = {
        "input_h5": result.input_h5,
        "staging_h5": result.staging_h5,
        "input_coverage": coverage_path,
        "hmrc_evidence": evidence_path,
        "hmrc_replay": replay_path,
        "spi_donor": args.spi_tab,
        "hmrc_surface": args.hmrc_ods,
        "frs_adult": retained_leaves_transform.adult_tab_path,
        "frs_benefits": retained_leaves_transform.benefits_tab_path,
    }
    artifacts = {
        role: _artifact_info(path) for role, path in artifact_paths.items()
    }
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
        "schema_version": 3,
        "build_kind": "uk_national_staging_dataset",
        "stages": list(result.stage_names),
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
        for record in result.dataset.mass_log
    ]
    household_weights = pd.to_numeric(
        result.dataset.household["household_weight"], errors="raise"
    )
    return {
        "schema_version": 1,
        "build_kind": "uk_national_staging_dataset",
        "status": "passed",
        "stages": list(result.stage_names),
        "parameters": {
            "seed": int(seed),
            "qrf_estimators": int(qrf_estimators),
        },
        "dataset": {
            "time_period": str(result.dataset.time_period),
            "entity_rows": {
                "person": len(result.dataset.person),
                "benunit": len(result.dataset.benunit),
                "household": len(result.dataset.household),
            },
            "household_weight_kind": result.dataset.household_weight_kind.value,
            "household_weight_total": float(household_weights.sum()),
            "mass_changes": mass_changes,
        },
        "source_rows": {
            "frs_adult": int(
                dict(retained_sources.get("adult", {})).get("rows", 0)
            ),
            "frs_benefits": int(
                dict(retained_sources.get("benefits", {})).get("rows", 0)
            ),
            "spi_donor_used": int(
                dict(family_sources.get("spi_donor", {})).get("rows_used", 0)
            ),
        },
        "source_vintages": dict(family_evidence.get("source_vintages", {})),
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
            "effective_mass_policy": dict(
                details.get("effective_mass_policy", {})
            ),
            "family_effective_mass": dict(
                details.get("family_effective_mass", {})
            ),
            "family_build_state": dict(details.get("family_build_state", {})),
        },
        "hmrc_replay": {
            "summary": dict(
                dict(family_evidence.get("targets", {})).get(
                    "classification", {}
                )
            ),
            "post_draw_identity": dict(
                family_evidence.get("post_draw_identity", {})
            ),
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
            "revision": candidate.revision,
            "sha256": candidate.sha256,
            "size_bytes": candidate.size_bytes,
        },
        "retained_leaves": retained_result.evidence(),
        "family": hmrc_result.evidence(),
    }
    _write_json(evidence_path, payload)
    write_hmrc_replay_report(hmrc_result.replay_report, replay_path)


def _is_final_release_gate_failure(error: RuntimeError) -> bool:
    """Match only the national seam's post-stage, pre-staging hard gate."""

    return str(error).startswith("Release gates failed: Input coverage failed:")


def _validate_distinct_paths(
    *,
    evidence_path: Path,
    replay_path: Path,
    coverage_path: Path,
    input_h5: Path,
    staging_h5: Path,
    spi_tab: Path,
    hmrc_ods: Path,
    adult_tab: Path,
    benefits_tab: Path,
    build_record_path: Path,
) -> None:
    paths = {
        "--input-h5": input_h5.resolve(),
        "--staging-h5": staging_h5.resolve(),
        "--spi-tab": spi_tab.resolve(),
        "--hmrc-ods": hmrc_ods.resolve(),
        "--frs-raw-dir/adult.tab": adult_tab.resolve(),
        "--frs-raw-dir/benefits.tab": benefits_tab.resolve(),
        "--build-record-json": build_record_path.resolve(),
        "--input-coverage-json": coverage_path.resolve(),
        "--hmrc-evidence-json": evidence_path.resolve(),
        "--hmrc-replay-json": replay_path.resolve(),
    }
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
