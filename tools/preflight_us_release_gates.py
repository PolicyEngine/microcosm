"""Preflight the US release gates that are statically knowable without a solve.

Each Build M release attempt (9/10/11) spent ~2h of calibration to surface one
gate-group failure already determined by the base pool, the frozen selection,
and the registry. This tool recovers those signals in minutes and reports, per
check, PASS / FAIL / AT-RISK with the measured numbers. Run it:

- when the base build exits (before launching a release),
- before any release launch, and
- after any change to the selection-source manifest or the target/coverage
  registry.

Example (read-only against the artifacts)::

    uv run python tools/preflight_us_release_gates.py \
        --base-h5 out/base-m/base_populace_us_2024_puf_support.h5 \
        --selection-source-manifest inputs/buildm_keogh_swap_selection_source.json \
        --export-input-mass-reference-h5 forensics/populace_us_2024.h5

Exit code: 1 on any static-check FAIL, 2 on static AT-RISK only, 0 clean. A
carried red base-pool battery is human-review evidence and does not by itself
change that exit code; ``--release-manifest`` also supports a manifest-only
publication review when those static inputs do not apply.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the release tool's export-mass register and engine input-variable
# surface — never re-declared here (this file sits beside it in tools/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "packages" / "microcosm-build" / "src")
)

from microcosm.build.us_runtime.release_gate_preflight import (
    run_preflight,  # noqa: E402
)

_ALLOW_GATE_FAILED_BASE_POOL_FLAG = "--allow-gate-failed-base-pool"
_CARRIED_BATTERY_PAYLOAD_KEY = "carried_base_pool_agreement_battery"


def _json_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _load_carried_base_pool_battery(
    path: Path,
) -> dict[str, object] | None:
    """Read the non-blocking red pool verdict carried by a release manifest."""

    release_manifest = _json_object(
        json.loads(path.read_text()),
        label=f"release manifest {path}",
    )
    build = release_manifest.get("build")
    if not isinstance(build, dict):
        return None
    exact_k_ladder = build.get("exact_k_ladder")
    if not isinstance(exact_k_ladder, dict):
        return None
    pool = exact_k_ladder.get("pool")
    agreement_gate_reference = exact_k_ladder.get("agreement_gate_reference")
    if not isinstance(pool, dict):
        return None
    carries_gate_failed_override = (
        pool.get("allow_gate_failed_base_pool") is True
        or pool.get("status") == "gate_failed"
        or (
            isinstance(agreement_gate_reference, dict)
            and agreement_gate_reference.get("battery_status") == "red"
        )
    )
    if not carries_gate_failed_override:
        return None
    if (
        pool.get("status") != "gate_failed"
        or pool.get("simulation_ready") is not False
        or pool.get("allow_gate_failed_base_pool") is not True
    ):
        raise ValueError(
            f"Release manifest {path} has an incoherent gate-failed base-pool "
            "carriage receipt."
        )
    gate_reference = _json_object(
        agreement_gate_reference,
        label=(
            f"release manifest {path} build.exact_k_ladder."
            "agreement_gate_reference"
        ),
    )
    failures = gate_reference.get("failures")
    failure_count = gate_reference.get("failure_count")
    gates_json_sha256 = gate_reference.get("gates_json_sha256")
    verdict = gate_reference.get("verdict")
    if (
        gate_reference.get("passed") is not False
        or gate_reference.get("battery_status") != "red"
        or not isinstance(failures, list)
        or not all(isinstance(failure, str) for failure in failures)
        or type(failure_count) is not int
        or failure_count != len(failures)
        or not isinstance(gates_json_sha256, str)
        or len(gates_json_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in gates_json_sha256
        )
        or not isinstance(verdict, dict)
        or verdict.get("passed") is not False
        or gate_reference.get("diagnostics_sha256") != gates_json_sha256
        or pool.get("agreement_diagnostics_sha256") != gates_json_sha256
    ):
        raise ValueError(
            f"Release manifest {path} has an incomplete or inconsistent "
            "carried red agreement-battery verdict."
        )
    return {
        "battery_status": "red",
        "pool_status": "gate_failed",
        "simulation_ready": False,
        "allow_gate_failed_base_pool": True,
        "flag": _ALLOW_GATE_FAILED_BASE_POOL_FLAG,
        "gates_json_sha256": gates_json_sha256,
        "failure_count": failure_count,
        "failures": list(failures),
        "agreement_gate_reference": dict(gate_reference),
        "publication_decision": "human_review_required",
        "affects_exit_code": False,
    }


def _carried_battery_banner(carried: dict[str, object]) -> str:
    count = int(carried["failure_count"])
    noun = "FAILURE" if count == 1 else "FAILURES"
    lines = [
        "=" * 72,
        f"CARRIED BASE-POOL AGREEMENT BATTERY: RED — {count} {noun}",
        "Pool status: gate_failed; simulation_ready: false",
        f"Build opt-in used: {carried['flag']}",
        f"Pool gates JSON SHA-256: {carried['gates_json_sha256']}",
        "Publication decision: HUMAN REVIEW REQUIRED",
        "This carried verdict does not alter the preflight exit code.",
    ]
    lines.extend(f"  FAILURE: {failure}" for failure in carried["failures"])
    lines.append("=" * 72)
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Statically preview the US release gates (selection carryover, "
            "zero-support, export-mass parity risk, reform-coverage smoke "
            "support) without running a calibration solve."
        )
    )
    parser.add_argument(
        "--base-h5",
        type=Path,
        help=(
            "Base pool H5 (read-only). Required with "
            "--selection-source-manifest unless --release-manifest is used "
            "for a manifest-only publication review."
        ),
    )
    parser.add_argument(
        "--selection-source-manifest",
        type=Path,
        help=(
            "Frozen selection-source manifest JSON. Required with --base-h5 "
            "for the existing static preflight checks."
        ),
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        help=(
            "Optional built release_manifest.json. A carried gate-failed "
            "base-pool battery is displayed as prominent, non-blocking "
            "evidence for the human publication decision. It may be used "
            "alone for publication review or alongside the existing static "
            "preflight inputs."
        ),
    )
    parser.add_argument(
        "--export-input-mass-reference-h5",
        type=Path,
        default=None,
        help=(
            "Reference H5 for the export-mass parity band (read-only). Omit to "
            "skip the parity-risk check."
        ),
    )
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        default=None,
        help=(
            "Optional Ledger consumer facts feed (directory or "
            "consumer_facts.jsonl). Required to preview zero-support — the "
            "compiled fiscal-target surface comes from it. Omitted -> the "
            "zero-support check is SKIPPED."
        ),
    )
    parser.add_argument(
        "--ledger-facts-sha256",
        default=None,
        help="Optional pin: expected SHA-256 of consumer_facts.jsonl.",
    )
    parser.add_argument(
        "--target-period",
        default=2024,
        help="Build period the fiscal targets are compiled for (default 2024).",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.5,
        help=(
            "Export-mass parity band half-width (matches the release tool's "
            "--input-mass-relative-tolerance; default 0.5)."
        ),
    )
    parser.add_argument(
        "--minimum-reference-total",
        type=float,
        default=1e9,
        help=(
            "Reference-mass floor below which parity is not checked (matches "
            "the release tool's --input-mass-minimum-reference-total; default "
            "1e9)."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the machine-readable report JSON to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    has_base_h5 = args.base_h5 is not None
    has_selection_manifest = args.selection_source_manifest is not None
    if has_base_h5 != has_selection_manifest:
        parser.error(
            "--base-h5 and --selection-source-manifest must be provided together."
        )
    if not has_base_h5 and args.release_manifest is None:
        parser.error(
            "provide --base-h5 with --selection-source-manifest, or provide "
            "--release-manifest for publication review."
        )
    carried = (
        _load_carried_base_pool_battery(args.release_manifest)
        if args.release_manifest is not None
        else None
    )
    if not has_base_h5 and carried is None:
        parser.error(
            "manifest-only publication review requires a release manifest "
            "carrying an explicit gate-failed base-pool verdict."
        )
    if carried is not None:
        print(_carried_battery_banner(carried))

    try:
        target_period: int | str = int(args.target_period)
    except (TypeError, ValueError):
        target_period = args.target_period

    if has_base_h5:
        report = run_preflight(
            base_h5=args.base_h5,
            selection_source_manifest=args.selection_source_manifest,
            export_input_mass_reference_h5=args.export_input_mass_reference_h5,
            ledger_facts=args.ledger_facts,
            ledger_facts_sha256=args.ledger_facts_sha256,
            target_period=target_period,
            relative_tolerance=args.relative_tolerance,
            minimum_reference_total=args.minimum_reference_total,
        )
        print(report.human_table())
        payload = report.to_dict()
        exit_code = report.exit_code
    else:
        print("US release-gate preflight (manifest-only publication review)")
        print("Static base/selection checks: NOT RUN")
        print("Automated exit: 0 (carried evidence requires human review)")
        payload = {
            "status": "PASS",
            "exit_code": 0,
            "inputs": {"release_manifest": str(args.release_manifest)},
            "checks": [],
            "static_checks_run": False,
        }
        exit_code = 0
    if carried is not None:
        payload[_CARRIED_BATTERY_PAYLOAD_KEY] = carried
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=False))
        print(f"\nWrote machine-readable report to {args.json_out}")
    else:
        print("\n--- machine-readable report ---")
        print(json.dumps(payload, indent=2, sort_keys=False))
    if carried is not None:
        print(
            "\nCARRIED RED BATTERY: HUMAN REVIEW REQUIRED; "
            f"automated preflight exit remains {exit_code}."
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
