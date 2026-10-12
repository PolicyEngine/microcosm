"""Copy the #1113 measurement outputs into the repository as aggregates only.

Usage: python sanitize_outputs.py <measurement dir> <outputs dir>

The stage's ``donor_floor`` receipt records each donor column's exact minimum,
which is licensed unit-record information, so it is dropped. Household dumps
and solved weight vectors are never copied. Paths inside the outputs are
machine-local.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Output files that hold aggregates only once ``DROPPED`` keys are removed.
AGGREGATE_OUTPUTS = (
    "m2_m3_desk.json",
    "m4_report.json",
    "solve_base.json",
    "solve_drop4.json",
    "stage_dev_run.json",
    "stage_dev_run.group-b.json",
    "stage_dev_run.base-7f235941c.json",
    "stage_dev_run.review-r2.json",
)
DROPPED = ("donor_floor",)


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in DROPPED}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    for name in AGGREGATE_OUTPUTS:
        path = source / name
        if not path.exists():
            print("missing", name)
            continue
        payload = _strip(json.loads(path.read_text()))
        (target / name).write_text(
            json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n"
        )
        print("wrote", name)


if __name__ == "__main__":
    main()
