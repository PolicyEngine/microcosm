"""Score the 37% -> 39.6% top-rate reform on a populace_us_2024-format H5.

Runs policyengine_us.Microsimulation directly at the analysis year, which is
what policyengine's per-year dataset builder does input-by-input before it
simulates. The harness is accepted only if it reproduces the policyengine
6.0.0 result on the certified file (+$23.5B in 2026, $905B base).
"""

import json
import sys
import time
import warnings

import numpy as np
from policyengine_core.reforms import Reform
from policyengine_us import Microsimulation

warnings.filterwarnings("ignore")

path, year, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t0 = time.time()
THRESHOLDS_2026 = {
    "SINGLE": 640600.0,
    "JOINT": 768700.0,
    "SEPARATE": 384350.0,
    "HEAD_OF_HOUSEHOLD": 640600.0,
    "SURVIVING_SPOUSE": 768700.0,
}
reform = Reform.from_dict(
    {"gov.irs.income.bracket.rates.7": {"2026-01-01.2100-12-31": 0.396}},
    country_id="us",
)
base = Microsimulation(dataset=path)
ref = Microsimulation(dataset=path, reform=reform)
w = base.calculate("tax_unit_weight", year).values.astype(float)


def tu(sim, v):
    return sim.calculate(v, year).values.astype(float)


it0, it1 = tu(base, "income_tax"), tu(ref, "income_tax")
agi = tu(base, "adjusted_gross_income")
ti = tu(base, "taxable_income")
cgx = tu(base, "capital_gains_excluded_from_taxable_income")
fs = base.calculate("filing_status", year).values.astype(str)
ordinary = np.maximum(0, ti - cgx)
thr = np.array([THRESHOLDS_2026.get(s, np.inf) for s in fs])
above = np.maximum(0, ordinary - thr)
res = {
    "dataset": path,
    "year": year,
    "baseline_income_tax_bn": float((it0 * w).sum() / 1e9),
    "top_rate_revenue_change_bn": float(((it1 - it0) * w).sum() / 1e9),
    "agi_total_bn": float((agi * w).sum() / 1e9),
    "tax_units_weighted_m": float(w.sum() / 1e6),
    "ordinary_income_above_37pct_threshold_bn": float((above * w).sum() / 1e9),
    "returns_in_37pct_bracket": float(w[above > 0].sum()),
    "records_in_37pct_bracket": int((above > 0).sum()),
    "agi_bands": {},
}
edges = [1e5, 2e5, 5e5, 1e6, 1.5e6, 2e6, 5e6, 1e7, np.inf]
for lo, hi in zip(edges[:-1], edges[1:], strict=False):
    m = (agi >= lo) & (agi < hi)
    res["agi_bands"][f"{lo:.0f}-{hi:.0f}"] = {
        "records": int(m.sum()),
        "returns": float(w[m].sum()),
        "agi_bn": float((agi * w)[m].sum() / 1e9),
        "above_threshold_bn": float((above * w)[m].sum() / 1e9),
    }
res["seconds"] = time.time() - t0
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "agi_bands"}, indent=1))
for k, v in res["agi_bands"].items():
    print(
        f"  {k:>22}: n={v['records']:6d} returns={v['returns']:12,.0f} AGI=${v['agi_bn']:8,.0f}B above-threshold=${v['above_threshold_bn']:6,.0f}B"
    )
