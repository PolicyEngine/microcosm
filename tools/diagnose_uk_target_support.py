"""Per-target weight-stretch anatomy of a UK national calibration attempt.

Reads the target-support sidecars a national calibration attempt writes beside
its diagnostics before the terminal battery runs (``target_support_matrix.npz``,
``target_support_vectors.npz``, ``target_support_manifest.json``; see
``microcosm.build.uk_runtime.target_support``) and reports, for each requested
target: the target, design and final aggregates, the carrier count, the top
carrier concentration, the carrier weight-ratio distribution, and the share of
the target's final mass sitting on households stretched beyond declared
multiples of their design weight (3x and 5x by default), each beside the
frame-wide share of weight above the same multiple. This is the read-only
instrument behind the microcosm#890 acceptance line (the share of England
bus-fare mass on households stretched more than 3x) and the microcosm#930
measurement; it never modifies an artifact and never runs a solve.

When ``--diagnostics`` names the attempt's ``calibration_diagnostics.json``,
every recomputed final estimate is checked against the recorded one before a
report is printed, so a sidecar from another attempt is refused.

Example::

    uv run --no-sync python tools/diagnose_uk_target_support.py \\
        --attempt-dir data/ukds/acceptance/930-nts-bus/round-1 \\
        --diagnostics data/ukds/acceptance/930-nts-bus/round-1/calibration_diagnostics.json \\
        --prefix dft.bus_fare_receipts --prefix scotgov.bus --json anatomy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from microcosm.build.uk_runtime.target_support import (
    DEFAULT_STRETCH_THRESHOLDS,
    TargetSupportError,
    anatomy_for_targets,
    load_uk_target_support_sidecars,
    realised_max_weight_ratio,
    verify_against_diagnostics,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--attempt-dir",
        required=True,
        type=Path,
        help="directory holding the target-support sidecars",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="the attempt's calibration_diagnostics.json (verifies the sidecars)",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="a compiled row name (name or name@period); repeatable",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="report every row whose name starts with this prefix; repeatable",
    )
    parser.add_argument(
        "--stretch",
        type=float,
        action="append",
        default=None,
        help="weight-ratio threshold for the stretched-mass buckets; repeatable",
    )
    parser.add_argument("--top", type=int, default=10, help="carriers listed per row")
    parser.add_argument("--json", type=Path, default=None, help="write the reports")
    return parser.parse_args(argv)


def _percent(value: object, *, signed: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sidecars = load_uk_target_support_sidecars(args.attempt_dir)
    names: list[str] = list(args.target)
    for prefix in args.prefix:
        names.extend(n for n in sidecars.names if n.startswith(prefix))
    if not names:
        raise SystemExit("name at least one --target or --prefix.")
    seen: set[str] = set()
    ordered = [n for n in names if not (n in seen or seen.add(n))]
    thresholds = tuple(args.stretch) if args.stretch else DEFAULT_STRETCH_THRESHOLDS
    reports = anatomy_for_targets(
        sidecars, ordered, stretch_thresholds=thresholds, top=args.top
    )
    if args.diagnostics is not None:
        diagnostics = json.loads(args.diagnostics.read_text())
        verify_against_diagnostics(reports, diagnostics)
    max_ratio = realised_max_weight_ratio(sidecars)
    print(
        f"attempt {args.attempt_dir}: {len(sidecars.names)} compiled rows, "
        f"{len(sidecars.final_weights)} records, realised max weight ratio "
        f"{max_ratio if max_ratio is None else f'{max_ratio:.3f}'}"
    )
    for report in reports:
        buckets = ", ".join(
            f">{key}: mass {_percent(b['stretched_mass'])} "
            f"(frame {_percent(b['frame_weight_share'])}), carriers "
            f"{_percent(b['stretched_carriers'])}"
            for key, b in report["stretched"].items()
        )
        ratio = report["carrier_weight_ratio"]
        print(
            f"- {report['name']}: design {_percent(report['design_relative_error'], signed=True)}"
            f" -> final {_percent(report['final_relative_error'], signed=True)}; "
            f"carriers {report['carrier_count']}, top-1 {_percent(report['top_1_share'])}, "
            f"top-5 {_percent(report['top_5_share'])}; ratio median "
            f"{ratio['median'] if ratio['median'] is None else f'{ratio["median"]:.2f}'}, "
            f"p90 {ratio['p90'] if ratio['p90'] is None else f'{ratio["p90"]:.2f}'}, "
            f"max {ratio['max'] if ratio['max'] is None else f'{ratio["max"]:.2f}'}; "
            f"{buckets}"
        )
    if args.json is not None:
        args.json.write_text(json.dumps(reports, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TargetSupportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
