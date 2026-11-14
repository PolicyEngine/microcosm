"""Count the people in the restored Medicaid statuses who report Medicaid.

Usage:
  python -I medicaid_reporters.py LABELS.pkl --archive-dir <dir> [--out OUT.json]

LABELS.pkl is a ``--labels-out`` file from ``stage_receipt_pooled.py``. The
script reads ``NOW_MCAID`` (Medicaid at interview) from the three ASEC person
files that script runs on, after checking each archive against the same
SHA-256, and joins them to the labels on ``(source_year, PERIDNUM)`` as that
script keys them. It then sums the run's person weights, by status, over the
statuses that the reform-coverage probe
``hr1_medicaid_humanitarian_eligibility_restoration`` restores.
``microcosm.build.us_runtime.cps_carried`` maps ``NOW_MCAID`` to
``has_medicaid_health_coverage_at_interview``, the column the Medicaid take-up
stage anchors on. The output holds aggregates only, in millions of people.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# The archives and SHA-256 values of stage_receipt_pooled.py.
ARCHIVES = (
    (
        2023,
        "asecpub24csv.zip",
        "pppub24.csv",
        "cdb39cdac34bef99dd0940ab28e306f692404c2eea44d85dfd634214872a0a09",
    ),
    (
        2024,
        "asecpub25csv.zip",
        "pppub25.csv",
        "318845a2b5e0034eb2973898de1738f4df0025727de38499e7669cb9c0deef0b",
    ),
    (
        2025,
        "asecpub26csv.zip",
        "pppub26.csv",
        "fe819d0d2fc4470c282e76c247b8aa8b6054811619730307e9f2fe6186707d93",
    ),
)
# The statuses the probe adds to the 2027 eligible list.
RESTORED = (
    "REFUGEE",
    "ASYLEE",
    "DEPORTATION_WITHHELD",
    "PAROLED_ONE_YEAR",
    "CONDITIONAL_ENTRANT",
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("labels", type=Path)
parser.add_argument(
    "--archive-dir",
    type=Path,
    action="append",
    required=True,
    help="directory holding ASEC zips (repeatable; each zip is found by name)",
)
parser.add_argument("--out", type=Path)
args = parser.parse_args()


def _archive_path(name: str) -> Path:
    found = [d / name for d in args.archive_dir if (d / name).is_file()]
    if not found:
        raise SystemExit(f"{name} is in none of {args.archive_dir}")
    return found[0]


labels = pd.read_pickle(args.labels)
key = labels["source_year"].astype(str) + ":" + labels["source_person_id"].astype(str)
if key.duplicated().any():
    raise SystemExit(f"{args.labels}: duplicate person keys")

parts = []
archives = []
for income_year, archive_name, member, pinned in ARCHIVES:
    path = _archive_path(archive_name)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != pinned:
        raise SystemExit(f"{path}: SHA-256 {digest} is not the pinned {pinned}")
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(member))
        with archive.open(name) as handle:
            frame = pd.read_csv(handle, usecols=["PERIDNUM", "NOW_MCAID"])
    # stage_receipt_pooled.py reads PERIDNUM as an integer, then as text.
    frame.index = str(income_year) + ":" + frame["PERIDNUM"].astype(str)
    parts.append(frame["NOW_MCAID"])
    archives.append(
        {"income_year": income_year, "archive": archive_name, "sha256": digest}
    )
now_mcaid = pd.concat(parts)
if now_mcaid.index.duplicated().any():
    raise SystemExit("the ASEC files repeat a (source_year, PERIDNUM) key")
if set(now_mcaid.index) != set(key):
    raise SystemExit("the labels and the ASEC files hold different persons")

reports = now_mcaid.reindex(key).to_numpy() == 1
weight = labels["person_weight"].to_numpy(dtype=np.float64)
status = labels["immigration_status_str"].to_numpy()
rows = {}
for name in (*RESTORED, "all restored"):
    mask = np.isin(status, RESTORED) if name == "all restored" else status == name
    rows[name] = {
        "rows": int(mask.sum()),
        "persons_millions": round(float(weight[mask].sum()) / 1e6, 6),
        "now_mcaid_persons_millions": round(
            float(weight[mask & reports].sum()) / 1e6, 6
        ),
    }
    print(
        f"{name:22s} {rows[name]['persons_millions']:7.3f}M people, "
        f"{rows[name]['now_mcaid_persons_millions']:6.3f}M report Medicaid at "
        f"interview ({rows[name]['rows']} rows)"
    )

if args.out is not None:
    receipt = {
        "labels_file": args.labels.name,
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "labels_rows": int(len(labels)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "archives": archives,
        "column": "NOW_MCAID == 1",
        "statuses": rows,
    }
    args.out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
