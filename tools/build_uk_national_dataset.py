"""Build the national UK staging file with the guarded HMRC/SPI family."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qrf-estimators", type=int, default=100)
    parser.add_argument("--calibration-epochs", type=int, default=256)
    parser.add_argument("--calibration-learning-rate", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidate = verify_certified_uk_candidate(args.input_h5)
    coverage_path = args.input_coverage_json or args.staging_h5.with_suffix(
        ".input_coverage.json"
    )
    evidence_path = args.hmrc_evidence_json or args.staging_h5.with_suffix(
        ".hmrc_income.json"
    )
    _validate_distinct_paths(
        evidence_path=evidence_path,
        coverage_path=coverage_path,
        input_h5=args.input_h5,
        staging_h5=args.staging_h5,
        spi_tab=args.spi_tab,
        hmrc_ods=args.hmrc_ods,
    )
    evidence_path.unlink(missing_ok=True)
    transform = UKHMRCIncomeStageTransform(
        spi_tab_path=args.spi_tab,
        hmrc_ods_path=args.hmrc_ods,
        certified_candidate=candidate,
        seed=args.seed,
        qrf_estimators=args.qrf_estimators,
        calibration_epochs=args.calibration_epochs,
        calibration_learning_rate=args.calibration_learning_rate,
    )
    result = build_uk_national_dataset(
        input_h5=args.input_h5,
        staging_h5=args.staging_h5,
        stages=(UKNationalStage(name="hmrc_spi_income", transform=transform),),
        input_coverage_path=coverage_path,
    )
    if transform.last_result is None:  # pragma: no cover - defensive
        raise RuntimeError("HMRC/SPI national stage completed without evidence.")
    hmrc_evidence = {
        "schema_version": 1,
        "base_candidate": {
            "path": str(candidate.path),
            "filename": candidate.filename,
            "revision": candidate.revision,
            "sha256": candidate.sha256,
            "size_bytes": candidate.size_bytes,
        },
        "family": transform.last_result.evidence(),
    }
    _write_json(evidence_path, hmrc_evidence)
    payload = {
        "schema_version": 2,
        "build_kind": "uk_national_staging_dataset",
        "stages": list(result.stage_names),
        "input_coverage": {
            "passed": result.input_coverage.passed,
            "failures": list(result.input_coverage.failures),
            "details": dict(result.input_coverage.details),
        },
        "artifacts": {
            "input_h5": _artifact_info(result.input_h5),
            "staging_h5": _artifact_info(result.staging_h5),
            "input_coverage": _artifact_info(coverage_path),
            "hmrc_evidence": _artifact_info(evidence_path),
            "spi_donor": _artifact_info(args.spi_tab),
            "hmrc_surface": _artifact_info(args.hmrc_ods),
        },
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


def _validate_distinct_paths(
    *,
    evidence_path: Path,
    coverage_path: Path,
    input_h5: Path,
    staging_h5: Path,
    spi_tab: Path,
    hmrc_ods: Path,
) -> None:
    evidence = evidence_path.resolve()
    reserved = {
        path.resolve()
        for path in (coverage_path, input_h5, staging_h5, spi_tab, hmrc_ods)
    }
    if evidence in reserved:
        raise ValueError(
            "--hmrc-evidence-json must differ from every input, staging, and "
            "coverage path."
        )


if __name__ == "__main__":
    raise SystemExit(main())
