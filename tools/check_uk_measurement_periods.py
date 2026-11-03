"""Check how a finished UK national build measured each plain-column row.

Reads the build's ``calibration_diagnostics.json`` (the rows) and
``build_record.json`` (the measure resolver's attached routes), the contract,
and the stored columns of the spine the build read. It writes a receipt with
each row's measurement period and value route, and names any row whose
stored value was read at a period the frame does not hold (microcosm#1324).
Warn only: it prints the rows and exits 0 unless ``--strict`` is given.

    uv run --no-sync python tools/check_uk_measurement_periods.py \
        --run <run-dir> --spine-h5 <spine.h5> [--out receipt.json] [--strict]

The receipt carries target names, periods and route labels only; keep it on
the private lane with the run directory all the same.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources as importlib_resources
from pathlib import Path

import pandas as pd

from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.measurement_period import uk_value_read_routes


def _stored_columns(spine_h5: Path) -> dict[str, list[str]]:
    columns: dict[str, list[str]] = {}
    with pd.HDFStore(spine_h5, "r") as store:
        for entity in ("person", "benunit", "household"):
            storer = store.get_storer(f"/{entity}")
            columns[entity] = [str(name) for name in storer.non_index_axes[0][1]]
    return columns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--spine-h5", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--frame-period")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    diagnostics = json.loads((args.run / "calibration_diagnostics.json").read_text())
    record = json.loads((args.run / "build_record.json").read_text())
    attached = record["calibration"]["measure_resolution"]["attached"]
    contract = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text()
    )
    targets = {target["target_id"]: target for target in contract["targets"]}
    rows = [
        (
            str(row["name"]),
            str(row["metadata"].get("contract_target_id")),
            str(row["entity"]),
        )
        for row in diagnostics["targets"]
    ]
    periods = {str(row["period"]) for row in diagnostics["targets"]}
    if len(periods) != 1:
        raise SystemExit(
            f"rows carry more than one calibration period: {sorted(periods)}"
        )
    receipt = uk_value_read_routes(
        targets,
        rows,
        stored_columns=_stored_columns(args.spine_h5),
        frame_period=args.frame_period or load_uk_frs_release().time_period,
        calibration_period=periods.pop(),
        attached=attached,
    )
    out = args.out or args.run / "measurement_period_receipt.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    routes: dict[str, int] = {}
    for row in receipt["rows"]:
        for route in row["value_routes"].values():
            key = route.split(":", 1)[0]
            routes[key] = routes.get(key, 0) + 1
    print(f"wrote {out}: {len(receipt['rows'])} plain-column rows; routes {routes}")
    off = receipt["stored_reads_off_period"]
    if off:
        print(
            f"warning: {len(off)} row(s) read a stored value at a period the frame "
            f"does not hold: {off[:20]}{' ...' if len(off) > 20 else ''}",
            file=sys.stderr,
        )
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
