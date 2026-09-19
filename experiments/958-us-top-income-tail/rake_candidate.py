"""PROTOTYPE reweighting: tilt a candidate's household weights onto the compiled SOI Table 1.1
size-of-AGI targets (the rows microcosm#958's target change binds), at the 2024 build period.

Stand-in for the Microcosm solve, NOT the solve: it minimizes entropy distance from the starting
weights subject to the 16 band rows, holds every household with no tax unit at or above $100k
fixed, and enforces the solver's hard cap (weight <= 5x start; calibration.yaml max_weight_ratio).
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from policyengine_us import Microsimulation

src, specs_path, out, receipt_path = sys.argv[1:5]
CAP = 5.0
YEAR = 2024
warnings.filterwarnings("ignore")

cache = os.path.splitext(os.path.basename(src))[0] + ".agi2024.npz"
if os.path.exists(cache):
    z = np.load(cache)
    agi, tu_id = z["agi"], z["tu_id"]
else:
    sim = Microsimulation(dataset=src)
    agi = sim.calculate("adjusted_gross_income", YEAR).values.astype(float)
    tu_id = sim.calculate("tax_unit_id", YEAR).values.astype("int64")
    np.savez(cache, agi=agi, tu_id=tu_id)
pe = (
    pd.read_hdf(src, "person", columns=["person_tax_unit_id", "person_household_id"])
    .drop_duplicates("person_tax_unit_id")
    .set_index("person_tax_unit_id")
)
hh = pd.read_hdf(src, "household")
hid = hh.household_id.to_numpy()
w0 = hh.household_weight.to_numpy(float)
pos = pd.Series(np.arange(len(hid)), index=hid)
tu_pos = pos.loc[pe.loc[tu_id].person_household_id.to_numpy()].to_numpy()
specs = [
    s
    for s in json.load(open(specs_path))
    if s["metadata"].get("requires_agi_size_distribution_rebase") == "true"
]
rows, tgt, names = [], [], []
for s in specs:
    lo, hi = (
        float(s["metadata"]["agi_lower_bound"]),
        float(s["metadata"]["agi_upper_bound"]),
    )
    m = (agi >= lo) & (agi < hi)
    v = m.astype(float) if s["metadata"]["measure_mode"] == "indicator_sum" else agi * m
    rows.append(np.bincount(tu_pos, weights=v, minlength=len(hid)))
    tgt.append(float(s["value"]))
    names.append(s["name"])
comp_path = sys.argv[5] if len(sys.argv) > 5 else None
n_band_rows = len(rows)
if comp_path:
    cols = {
        "wages": ["employment_income_before_lsr"],
        "net_capital_gains": [
            "long_term_capital_gains_before_response",
            "short_term_capital_gains",
        ],
    }
    need = sorted({c for v in cols.values() for c in v})
    pp = pd.read_hdf(src, "person", columns=["person_tax_unit_id"] + need)
    tu_comp = pp.groupby("person_tax_unit_id")[need].sum().reindex(tu_id)
    for c in json.load(open(comp_path)):
        lo, hi = float(c["lower"]), float(c["upper"])
        m = (agi >= lo) & (agi < hi)
        v = tu_comp[cols[c["component"]]].sum(axis=1).to_numpy() * m
        rows.append(np.bincount(tu_pos, weights=v, minlength=len(hid)))
        tgt.append(float(c["target_2024"]))
        names.append(c["name"])
rows_by_household = np.vstack(rows)
t = np.array(tgt)
adj = (rows_by_household[:n_band_rows] != 0).any(axis=0)
before = rows_by_household @ w0
# Row scaling so entries are O(1): counts stay indicators, amounts divide by the band's target mean.
scale = np.array(
    [1.0 if s["metadata"]["measure_mode"] == "indicator_sum" else 1.0 for s in specs]
)
mean_by_band = {}
for s, tt in zip(specs, t, strict=False):
    key = (s["metadata"]["agi_lower_bound"], s["metadata"]["agi_upper_bound"])
    mean_by_band.setdefault(key, {})[s["metadata"]["measure_mode"]] = tt
for j, s in enumerate(specs):
    key = (s["metadata"]["agi_lower_bound"], s["metadata"]["agi_upper_bound"])
    if s["metadata"]["measure_mode"] != "indicator_sum":
        scale[j] = mean_by_band[key]["sum"] / mean_by_band[key]["indicator_sum"]
scale = np.concatenate([scale, np.ones(len(t) - len(scale))])
for j in range(n_band_rows, len(t)):
    nz = rows_by_household[j, adj][rows_by_household[j, adj] != 0]
    scale[j] = np.abs(nz).mean() if len(nz) else 1.0
design = rows_by_household[:, adj] / scale[:, None]
b = t / scale
base = w0[adj]
w = base.copy()
free = np.ones(adj.sum(), bool)
mu = np.zeros(len(t))


def dual(mu, b_free, base_free, rhs_free):
    e = np.clip(b_free.T @ mu, -40, 40)
    return float((base_free * np.exp(e)).sum() - rhs_free @ mu)


for _outer in range(40):
    b_free, base_free = design[:, free], base[free]
    rhs_free = b - design[:, ~free] @ w[~free]
    for _it in range(500):
        e = np.clip(b_free.T @ mu, -40, 40)
        w_free = base_free * np.exp(e)
        g = b_free @ w_free - rhs_free
        if np.abs(g / b).max() < 1e-9:
            break
        hessian = (b_free * w_free) @ b_free.T
        step = np.linalg.solve(
            hessian + 1e-9 * np.trace(hessian) / len(b) * np.eye(len(b)), g
        )
        f0 = dual(mu, b_free, base_free, rhs_free)
        a_ = 1.0
        while a_ > 1e-8 and not dual(
            mu - a_ * step, b_free, base_free, rhs_free
        ) < f0 - 1e-4 * a_ * (g @ step):
            a_ *= 0.5
        mu = mu - a_ * step
    w[free] = base_free * np.exp(np.clip(b_free.T @ mu, -40, 40))
    over = free & (w > CAP * base * (1 + 1e-9))
    if not over.any():
        break
    w[over] = CAP * base[over]
    free &= ~over
w1 = w0.copy()
w1[adj] = w
# Conserve total household mass: households with no tax unit at or above $100k absorb the difference uniformly.
absorb = (w0.sum() - w1[adj].sum()) / w0[~adj].sum()
w1[~adj] = w0[~adj] * absorb
after = rows_by_household @ w1
rep = {
    "source": src,
    "households": int(len(w0)),
    "adjustable_households": int(adj.sum()),
    "capped_at_5x": int((~free).sum()),
    "max_weight_ratio": float((w1[w0 > 0] / w0[w0 > 0]).max()),
    "min_weight_ratio": float((w1[w0 > 0] / w0[w0 > 0]).min()),
    "below_100k_uniform_factor": float(absorb),
    "weight_total_before_m": float(w0.sum() / 1e6),
    "weight_total_after_m": float(w1.sum() / 1e6),
    "converged_max_abs_rel_error": float(np.abs(after / t - 1).max()),
    "rows": [
        {
            "name": n,
            "target": float(tt),
            "before": float(b),
            "after": float(x),
            "before_rel": float(b / tt - 1),
            "after_rel": float(x / tt - 1),
        }
        for n, tt, b, x in zip(names, t, before, after, strict=False)
    ],
}
json.dump(rep, open(receipt_path, "w"), indent=1)
hh["household_weight"] = w1
for k in ["household", "person", "tax_unit", "spm_unit", "family", "marital_unit"]:
    (hh if k == "household" else pd.read_hdf(src, k)).to_hdf(
        out, key=k, mode="w" if k == "household" else "a", format="table"
    )
pd.read_hdf(src, "_time_period").to_hdf(
    out, key="_time_period", mode="a", format="table"
)
print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
for r in rep["rows"]:
    print(
        f"  {r['name']:<64} target {r['target']:.4g} "
        f"before {100 * r['before_rel']:+7.1f}% after {100 * r['after_rel']:+8.4f}%"
    )
