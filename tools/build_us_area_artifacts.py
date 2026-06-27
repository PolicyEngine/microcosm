"""Build Populace-owned US state and congressional-district H5 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from populace.build.us_runtime.area_artifacts import (
    assert_complete_area_artifacts,
    congressional_district_artifact_specs,
    load_policyengine_us_h5_frame,
    state_artifact_specs,
    write_area_artifacts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--states", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--congressional-districts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Diagnostic only: write artifacts for observed states/districts "
            "without requiring the full 51-state / 436-district release set."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.allow_incomplete and (
        not args.states or not args.congressional_districts
    ):
        raise SystemExit(
            "--no-states and --no-congressional-districts are diagnostic-only; "
            "pass --allow-incomplete to write a partial area artifact manifest."
        )
    frame = load_policyengine_us_h5_frame(args.input_h5)
    specs = []
    if args.states:
        specs.extend(
            state_artifact_specs(frame, require_complete=not args.allow_incomplete)
        )
    if args.congressional_districts:
        specs.extend(
            congressional_district_artifact_specs(
                frame, require_complete=not args.allow_incomplete
            )
        )
    results = write_area_artifacts(
        frame,
        specs,
        output_root=args.out,
        period=args.period,
    )
    if not args.allow_incomplete:
        assert_complete_area_artifacts(results)
    manifest = {
        "input_h5": str(args.input_h5),
        "period": args.period,
        "n_artifacts": len(results),
        "artifacts": [
            {
                "key": result.key,
                "path": result.path,
                "kind": result.kind,
                "sha256": result.sha256,
                "n_households": result.n_households,
                "n_persons": result.n_persons,
            }
            for result in results
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "area_artifacts_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "n_artifacts": len(results)}, indent=2))


if __name__ == "__main__":
    main()
