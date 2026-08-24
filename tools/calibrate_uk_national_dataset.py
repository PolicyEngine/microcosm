"""Calibrate an existing UK national spine H5 without replaying source stages.

This driver is intentionally thin: it verifies pinned inputs, compiles the
Ledger target registry, applies the reviewed measure-exclusion register, records
any explicit doctrine overrides, and delegates the calibration/gate/logbook
work to :func:`microcosm.build.uk_runtime.calibration_run.run_uk_calibration`.

Signed deviation for v1: no sampling rungs and no checkpointing. This seam runs
full-scale only; scale ladders and resumable source-stage checkpoints belong to
the spine build lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.uk_runtime.calibration_run import (
    UKCalibrationRunPaths,
    run_uk_calibration,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.ledger_targets import compile_uk_target_registry
from microcosm.build.uk_runtime.measure_simulation import (
    UKMeasureResolver,
    apply_uk_calibration_measure_exclusions,
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.national_doctrine import uk_doctrine_with_overrides

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CANONICAL_UK_RELEASE_ID = re.compile(r"populace-uk-[1-9][0-9]*-[a-z0-9_]+-k[1-9][0-9]*")
_UK_JUNE_RELEASE_ID = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact = load_ledger_consumer_artifact(
        args.ledger_facts,
        expected_facts_sha256=args.ledger_facts_sha256,
        expected_manifest_sha256=args.ledger_manifest_sha256,
    )
    calibration_year = load_uk_frs_release().calibration_year
    compilation = compile_uk_target_registry(
        artifact.facts, target_period=calibration_year
    )
    if compilation.unsupported:
        raise SystemExit(
            f"{len(compilation.unsupported)} target references failed to compile"
        )
    _compare_frozen_register(args.register_json, compilation.registry)
    exclusions = load_uk_calibration_measure_exclusions(args.measure_exclusions)
    registry, exclusion_receipt = apply_uk_calibration_measure_exclusions(
        compilation.registry, exclusions
    )
    overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "target_weight_rule": args.target_weight_rule,
            "learning_rate": args.learning_rate,
            "target_loss_cap": args.target_loss_cap,
        }.items()
        if value is not None
    }
    doctrine, doctrine_overrides = uk_doctrine_with_overrides(**overrides)
    paths = UKCalibrationRunPaths(
        input_h5=args.input_h5,
        staging_h5=args.staging_h5,
        diagnostics_json=args.diagnostics_json,
        build_record_json=args.build_record_json,
        terminal_gate_json=args.terminal_gate_json,
    )
    # The resolver reads the input H5 to build its simulation, so the pin is
    # verified here first — no bytes are consumed before they match the CLI sha.
    measured_input_sha = _sha256_file(args.input_h5)
    if measured_input_sha != args.input_sha256:
        raise SystemExit(
            "error: --input-h5 sha mismatch: "
            f"measured {measured_input_sha}, pinned {args.input_sha256}"
        )
    resolver = UKMeasureResolver(
        simulation_source=args.input_h5,
        scratch_dir=args.staging_h5.parent,
        year=calibration_year,
        frame=None,
    )
    result = run_uk_calibration(
        paths=paths,
        input_sha256=args.input_sha256,
        ledger_artifact=artifact,
        register_registry=registry,
        calibration_year=calibration_year,
        exclusion_receipt=exclusion_receipt,
        doctrine=doctrine,
        doctrine_overrides=doctrine_overrides,
        measure_resolver=resolver,
        source_pins={
            "input_h5": {
                "sha256": args.input_sha256,
                "size_bytes": args.input_h5.stat().st_size,
            },
            "ledger_facts": _ledger_facts_pin(artifact),
        },
        run_config_extra={"calibration_year": calibration_year},
        release_candidate=args.release_candidate,
        release_id=args.release_id,
        logbook_prev_row_digest=args.logbook_prev_row_digest,
    )
    summary = {
        "staging_h5_sha256": result.staging_sha256,
        "diagnostics_sha256": result.diagnostics_sha256,
        "terminal_gate_sha256": result.terminal_gate_sha256,
        "build_record_sha256": result.build_record_sha256,
        "gate_verdicts": result.build_record["gate_summary"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True, type=_sha256)
    parser.add_argument("--ledger-facts", required=True, type=Path)
    parser.add_argument("--ledger-facts-sha256", required=True, type=_sha256)
    parser.add_argument("--ledger-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--staging-h5", required=True, type=Path)
    parser.add_argument("--diagnostics-json", required=True, type=Path)
    parser.add_argument("--build-record-json", required=True, type=Path)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--terminal-gate-json", type=Path)
    parser.add_argument("--register-json", type=Path)
    parser.add_argument("--measure-exclusions", type=Path)
    parser.add_argument("--release-candidate", action="store_true")
    parser.add_argument("--logbook-prev-row-digest", type=_sha256)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--target-weight-rule")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--target-loss-cap", type=float)
    args = parser.parse_args(argv)
    args.terminal_gate_json = args.terminal_gate_json or args.staging_h5.with_suffix(
        ".terminal_gates.json"
    )
    override_flags = {
        "--epochs": args.epochs,
        "--target-weight-rule": args.target_weight_rule,
        "--learning-rate": args.learning_rate,
        "--target-loss-cap": args.target_loss_cap,
    }
    passed_overrides = [flag for flag, value in override_flags.items() if value is not None]
    if args.release_candidate and passed_overrides:
        parser.error(
            "--release-candidate is refused with doctrine override flag(s): "
            + ", ".join(passed_overrides)
        )
    if (
        not args.release_candidate
        and (_CANONICAL_UK_RELEASE_ID.fullmatch(args.release_id) or args.release_id == _UK_JUNE_RELEASE_ID)
    ):
        parser.error("canonical UK release ids require --release-candidate")
    _validate_distinct_paths(
        {
            "--input-h5": args.input_h5,
            "--ledger-facts": args.ledger_facts,
            "--staging-h5": args.staging_h5,
            "--diagnostics-json": args.diagnostics_json,
            "--build-record-json": args.build_record_json,
            "--terminal-gate-json": args.terminal_gate_json,
            **({"--register-json": args.register_json} if args.register_json else {}),
            **(
                {"--measure-exclusions": args.measure_exclusions}
                if args.measure_exclusions
                else {}
            ),
        }
    )
    return args


def _sha256(value: str) -> str:
    if not _SHA256.fullmatch(value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_distinct_paths(paths: dict[str, Path]) -> None:
    resolved = {label: path.expanduser().resolve() for label, path in paths.items()}
    for (left_label, left), (right_label, right) in combinations(resolved.items(), 2):
        if _paths_alias(left, right):
            raise SystemExit(
                f"error: {left_label} and {right_label} must be distinct paths "
                f"({left} aliases {right})"
            )


def _paths_alias(left: Path, right: Path) -> bool:
    if str(left).casefold() == str(right).casefold():
        return True
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except FileNotFoundError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino)


def _compare_frozen_register(path: Path | None, registry: Any) -> None:
    if path is None:
        return
    payload = {
        "country": registry.country,
        "version": registry.version,
        "specs": [
            {field: value for field, value in spec.__dict__.items() if value is not None}
            for spec in registry.specs
        ],
    }
    derived = hashlib.sha256(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    frozen = hashlib.sha256(path.read_bytes()).hexdigest()
    if derived != frozen:
        raise SystemExit(
            "re-derived register differs from the frozen scoring register: "
            f"{derived} vs {frozen}"
        )


def _ledger_facts_pin(artifact: Any) -> dict[str, object]:
    facts_path = (
        artifact.path / "consumer_facts.jsonl" if artifact.path.is_dir() else artifact.path
    )
    return {"sha256": artifact.facts_sha256, "size_bytes": facts_path.stat().st_size}


if __name__ == "__main__":
    raise SystemExit(main())
