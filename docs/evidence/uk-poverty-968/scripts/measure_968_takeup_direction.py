"""microcosm#968 follow-up (Vahid point 3): which way does a realistic residual allocation move poverty?
Residual UC claimants among entitled non-reporters ranked by entitlement size (largest first = DWP's
finding that non-take-up concentrates in small entitlements), v20 at 2024, calibrated weights."""

import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from policyengine_uk import Microsimulation

PATH = "/Users/mariajuaristi/Desktop/PolicyEngine/runs/uk-623-first-calibrated/spine-assessment-v20/microcosm_uk_2024.h5"
YEAR = 2024


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


with pd.HDFStore(PATH, "r") as st:
    person = st["person"]
    h5hh = st["household"]
    bu = st["benunit"]
hid = h5hh["household_id"].to_numpy()
p_idx = pd.Index(hid).get_indexer(person["person_household_id"].to_numpy())
bid = bu["benunit_id"].to_numpy()
b_of_p = pd.Index(bid).get_indexer(person["person_benunit_id"].to_numpy())
base_claim = bu["would_claim_uc"].to_numpy(bool)
anchor = np.zeros(len(bid), bool)
anchor[
    np.unique(
        b_of_p[
            pd.to_numeric(person["universal_credit_reported"], errors="coerce")
            .fillna(0)
            .to_numpy()
            > 0
        ]
    )
] = True


def run(claim, label):
    sim = Microsimulation(dataset=PATH)
    sim.set_input("would_claim_uc", YEAR, claim.astype(bool))
    hh = lambda v: np.asarray(sim.calculate(v, YEAR, map_to="household").values, float)
    hw = hh("household_weight")
    pw = hw[p_idx]
    eq = hh("equiv_hbai_household_net_income")[p_idx]
    thr = float(hh("poverty_threshold_bhc")[0])
    uc_b = np.asarray(sim.calculate("universal_credit", YEAR).values, float)
    bw = np.asarray(sim.calculate("benunit_weight", YEAR).values, float)
    med = wmedian(eq, pw)
    age = np.asarray(sim.calculate("age", YEAR).values, float)
    child = age < 18
    r = {
        "label": label,
        "uc_families_m": float(bw[uc_b > 0].sum() / 1e6),
        "uc_bn": float((uc_b * bw).sum() / 1e9),
        "rel_bhc_pct": 100 * float(pw[eq < 0.6 * med].sum() / pw.sum()),
        "abs_bhc_pct": 100 * float(pw[eq < thr].sum() / pw.sum()),
        "child_rel_pct": 100
        * float(pw[child & (eq < 0.6 * med)].sum() / pw[child].sum()),
        "median_week": med / 52,
    }
    print(json.dumps(r), flush=True)
    return r, uc_b


res = []
r0, _ = run(base_claim, "baseline draw (v20 file)")
res.append(r0)
r_all, uc_all = run(np.ones(len(bid), bool), "everyone entitled claims")
res.append(r_all)
entitled = uc_all > 0
pool = entitled & ~anchor
n = int((base_claim & pool).sum())
order = np.argsort(-uc_all)
pool_order = [i for i in order if pool[i]]
largest = np.zeros(len(bid), bool)
largest[pool_order[:n]] = True
smallest = np.zeros(len(bid), bool)
smallest[pool_order[-n:]] = True
res.append(
    run(
        anchor | largest | (base_claim & ~pool),
        "residual to the LARGEST entitlements (DWP take-up pattern)",
    )[0]
)
res.append(
    run(
        anchor | smallest | (base_claim & ~pool),
        "residual to the SMALLEST entitlements (opposite)",
    )[0]
)
# mean entitlement of the pool by chosen set
w_b = None
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("pool", int(pool.sum()), "n", n, "DONE")
