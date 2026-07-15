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

Exit code: 1 on any FAIL, 2 on AT-RISK only, 0 clean.
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
    0, str(Path(__file__).resolve().parents[1] / "packages" / "populace-build" / "src")
)

from populace.build.us_runtime.release_gate_preflight import (  # noqa: E402
    run_preflight,
)


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
        required=True,
        type=Path,
        help="Base pool H5 (read-only).",
    )
    parser.add_argument(
        "--selection-source-manifest",
        required=True,
        type=Path,
        help="Frozen selection-source manifest JSON.",
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
    args = _parser().parse_args(argv)
    try:
        target_period: int | str = int(args.target_period)
    except (TypeError, ValueError):
        target_period = args.target_period

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
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=False))
        print(f"\nWrote machine-readable report to {args.json_out}")
    else:
        print("\n--- machine-readable report ---")
        print(json.dumps(payload, indent=2, sort_keys=False))

    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
