"""Base-weight vs calibrated QRF tail shares for route A run 310842b986d7.

The measurement behind experiments/us-release-dry-run-margin-evidence.md. It
reads machine-local route A artifacts (paths below), so it is a record of the
method, not a CI step. Usage: ``python <this file> OUT.json``.

Reads only the needed fields from the raw base H5 (h5py field slices) and the
release's saved calibrated household weights, computes the release's own
tail_concentration_gate at both weightings, and checks the calibrated side
against the release's recorded qrf_tail_concentration.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

from microcosm.build.gates import tail_concentration_gate

RUN = Path(
    "/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/overnight-20260923/"
    "route-a/run-310842b986d7"
)
BASE = RUN / "base-out/base_populace_us_2024_puf_support.h5"
REL = RUN / (
    "release-out/populace-us-2024-0581707-310842b986d7-20260926T165326Z/releases/"
    "populace-us-2024-0581707-310842b986d7-20260926T165326Z"
)
recorded = json.loads((REL / "qrf_tail_concentration.json").read_text())
surface = recorded["surface"]
details = recorded["tail_concentration"]["details"]
columns = sorted(set(surface["checked_sparse_columns"]) | set(details["thin_columns"]))

f = h5py.File(BASE, "r")
hh = f["household/table"].fields(["household_id", "household_weight"])[:]
hh_ids = hh["household_id"]
hh_w = hh["household_weight"].astype(np.float64)

person_fields = set(f["person/table"].dtype.names)
tax_fields = set(f["tax_unit/table"].dtype.names)
p_cols = [c for c in columns if c in person_fields]
t_cols = [c for c in columns if c in tax_fields]
missing = [c for c in columns if c not in person_fields and c not in tax_fields]
person = f["person/table"].fields(
    ["person_household_id", "person_tax_unit_id", *p_cols]
)[:]
tax = f["tax_unit/table"].fields(["tax_unit_id", *t_cols])[:]

order = np.argsort(hh_ids)


def hh_index(ids):
    pos = np.searchsorted(hh_ids[order], ids)
    return order[pos]


person_hh = hh_index(person["person_household_id"])
# tax unit -> household through its first member
tu_ids = tax["tax_unit_id"]
first = {}
for tu, h in zip(person["person_tax_unit_id"], person_hh, strict=True):
    first.setdefault(int(tu), int(h))
tu_hh = np.array([first[int(t)] for t in tu_ids])

cal_ids = np.load(REL / "final_household_weight_ids.npy")
cal_w_raw = np.load(REL / "final_household_weights.npy").astype(np.float64)
cal_w = np.full(hh_w.shape, np.nan)
cal_w[hh_index(cal_ids)] = cal_w_raw
assert np.isfinite(cal_w).all(), "calibrated weights do not cover every base household"
assert np.array_equal(np.sort(cal_ids), np.sort(hh_ids))


def weighted(w_hh):
    vals, wts = {}, {}
    for c in p_cols:
        vals[c] = person[c].astype(np.float64)
        wts[c] = w_hh[person_hh]
    for c in t_cols:
        vals[c] = tax[c].astype(np.float64)
        wts[c] = w_hh[tu_hh]
    return vals, wts


kw = dict(top_k=100, max_top_share=0.75, min_nonzero_records=500)
bv, bw = weighted(hh_w)
cv, cw = weighted(cal_w)
base_gate = tail_concentration_gate(bv, bw, **kw)
cal_gate = tail_concentration_gate(cv, cw, **kw)

rows = []
max_abs_diff = 0.0
for c in sorted(set(p_cols) | set(t_cols)):
    b = base_gate.details["top_share"].get(c)
    k = cal_gate.details["top_share"].get(c)
    r = details["top_share"].get(c)
    bc = base_gate.details["carrier_counts"].get(
        c, base_gate.details["thin_columns"].get(c)
    )
    kc = cal_gate.details["carrier_counts"].get(
        c, cal_gate.details["thin_columns"].get(c)
    )
    rc = details["carrier_counts"].get(c, details["thin_columns"].get(c))
    if k is not None and r is not None:
        max_abs_diff = max(max_abs_diff, abs(k - r))
    rows.append(
        {
            "column": c,
            "entity": "person" if c in p_cols else "tax_unit",
            "base_share": b,
            "calibrated_share_recomputed": k,
            "calibrated_share_recorded": r,
            "shift": (None if b is None or k is None else k - b),
            "carriers_base": bc,
            "carriers_calibrated_recomputed": kc,
            "carriers_recorded": rc,
        }
    )

out = {
    "run": "route A 310842b986d7 release 20260926T165326Z",
    "columns_not_in_raw_base": missing,
    "max_abs_diff_recomputed_vs_recorded_calibrated_share": max_abs_diff,
    "rows": rows,
}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("not in raw base:", missing)
print("max |recomputed - recorded| calibrated share:", max_abs_diff)
print(
    f"{'column':45s} {'base':>6s} {'cal':>6s} {'rec':>6s} {'shift':>7s} {'nB':>6s} {'nC':>6s} {'nR':>6s}"
)


def fmt(x):
    return "  -   " if x is None else f"{x:6.3f}"


for r in sorted(rows, key=lambda r: -(r["calibrated_share_recorded"] or 0)):
    sh = "   -   " if r["shift"] is None else f"{r['shift']:+7.3f}"
    print(
        f"{r['column']:45s} {fmt(r['base_share'])} {fmt(r['calibrated_share_recomputed'])} {fmt(r['calibrated_share_recorded'])} {sh} {str(r['carriers_base']):>6s} {str(r['carriers_calibrated_recomputed']):>6s} {str(r['carriers_recorded']):>6s}"
    )
