"""Copy the #1359 measurement outputs into the repository as aggregates only.

Usage: python sanitize_outputs.py <measurement dir> <outputs dir>

The stage receipts carry no unit record by construction: shares, positive
shares and weighted totals over the frame or the donor, the donor support as a
band and any count under ten withheld. As for #1113, a ``donor_floor``,
``donor_min`` or ``donor_max`` key (licensed unit-record extremes) is dropped
wherever it appears, and so is ``output_sha256``, the digest of each written
column, which ``compare_runs.py`` reads in place and reduces to which columns
two runs leave identical. The ``tree`` and ``spine`` fields keep only their
names, since the paths are machine-local. Household dumps are never copied.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Output files that hold aggregates only once ``DROPPED`` keys are removed.
AGGREGATE_OUTPUTS = (
    "stage_dev_run.base-74a2a378e.json",
    "stage_dev_run.head-restack.json",
    "stage_dev_run.head-split.json",
    "stage_dev_run.head-main-rebase.json",
    "alcohol_channel_weights.json",
)
DROPPED = ("donor_floor", "donor_min", "donor_max", "output_sha256")
#: Keys whose string value is a machine-local path; only the file name stays.
PATH_KEYS = ("spine", "tree")


def _strip(value):
    if isinstance(value, dict):
        return {
            k: (Path(v).name if k in PATH_KEYS and isinstance(v, str) else _strip(v))
            for k, v in value.items()
            if k not in DROPPED
        }
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
