"""Run the standalone national UK staging seam.

Family-specific stages are added explicitly by later coverage milestones. The
initial tool is a gated pass-through that proves the orchestration and H5
boundary without changing the local-geography clone product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from populace.build.uk_runtime import build_uk_national_dataset


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
        "--input-coverage-json",
        type=Path,
        help=(
            "Coverage diagnostic path. Defaults beside --staging-h5 with "
            "suffix '.input_coverage.json'."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    coverage_path = args.input_coverage_json or args.staging_h5.with_suffix(
        ".input_coverage.json"
    )
    result = build_uk_national_dataset(
        input_h5=args.input_h5,
        staging_h5=args.staging_h5,
        stages=(),
        input_coverage_path=coverage_path,
    )
    payload = {
        "schema_version": 1,
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


if __name__ == "__main__":
    raise SystemExit(main())
