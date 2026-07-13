#!/usr/bin/env python3
"""Run SNAP-specific post-build acceptance checks on a US release candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from populace.build.us_runtime.snap_release_acceptance import (
    assemble_us_snap_release_acceptance,
    us_snap_participation_validation,
)

DATASET_FILENAME = "populace_us_2024.h5"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-dir",
        type=Path,
        required=True,
        help="Candidate releases/<release-id> directory.",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        help=(
            f"Candidate H5. Defaults to <release-root>/artifacts/{DATASET_FILENAME}."
        ),
    )
    parser.add_argument(
        "--target-relative-tolerance",
        type=float,
        default=0.10,
        help="Maximum absolute relative error for every state SNAP target.",
    )
    parser.add_argument(
        "--participation-tolerance-pp",
        type=float,
        default=10.0,
        help=(
            "Advisory absolute percentage-point tolerance for FNS FY2022 "
            "eligible-person participation rates."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output JSON. Defaults to <release-dir>/us_snap_release_acceptance.json."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def main() -> int:
    args = _parse_args()
    release_dir = args.release_dir.resolve()
    h5_path = (
        args.h5.resolve()
        if args.h5 is not None
        else release_dir.parent.parent / "artifacts" / DATASET_FILENAME
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else release_dir / "us_snap_release_acceptance.json"
    )
    required = {
        "build_manifest": release_dir / "build_manifest.json",
        "calibration_diagnostics": release_dir / "calibration_diagnostics.json",
        "snap_state_take_up": release_dir / "us_snap_state_take_up.json",
        "h5": h5_path,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SNAP release acceptance input(s): {missing}")

    build_manifest = _load_json(required["build_manifest"])
    release_id = str(build_manifest.get("build_id") or release_dir.name)
    participation = us_snap_participation_validation(
        h5_path,
        tolerance_percentage_points=args.participation_tolerance_pp,
    )
    payload = assemble_us_snap_release_acceptance(
        release_id=release_id,
        build_manifest=build_manifest,
        snap_state_take_up=_load_json(required["snap_state_take_up"]),
        calibration_diagnostics=_load_json(required["calibration_diagnostics"]),
        h5_path=h5_path,
        participation_validation=participation,
        target_relative_tolerance=args.target_relative_tolerance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "release_id": release_id,
                "passed": payload["passed"],
                "required_failures": payload["required_failures"],
                "participation_states_outside_advisory_tolerance": participation[
                    "states_outside_advisory_tolerance"
                ],
            },
            indent=2,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
