"""microcosm#968 experiment: HBAI-basis swap. Replace the engine's simulated means-tested
benefits with the FRS-reported amounts (same records, same weights, year 2024) and re-measure
BHC poverty. Quantifies the 'benchmark basis differs' layer."""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from importlib.metadata import version

from policyengine_uk import Microsimulation

ROOT = "/Users/mariajuaristi/Desktop/PolicyEngine"
HF = "/Users/mariajuaristi/.cache/huggingface/hub"
FILES = {
    "v20_national": f"{ROOT}/runs/uk-623-first-calibrated/spine-assessment-v20/microcosm_uk_2024.h5",
    "efrs_1_57_3": f"{HF}/models--policyengine--policyengine-uk-data-private/snapshots/25af520a6651b8812fef56964a42a79a3f9f515a/enhanced_frs_2024_25.h5",
}
SPINE_S = f"{ROOT}/data/ukds/acceptance/spine-s-main-rebuild/spine-s.h5"
# engine variable -> reported person-level input column(s)
SWAP = {
    "universal_credit": ["universal_credit_reported"],
    "housing_benefit": ["housing_benefit_reported"],
    "pension_credit": ["pension_credit_reported"],
    "child_tax_credit": ["child_tax_credit_reported"],
    "working_tax_credit": ["working_tax_credit_reported"],
    "income_support": ["income_support_reported"],
    "jsa_income": ["jsa_income_reported"],
    "esa_income": ["esa_income_reported"],
    "council_tax_benefit": ["council_tax_benefit_reported"],
    "child_benefit": ["child_benefit_reported"],
}
YEAR = 2024


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


def wshare(m, w):
    return float(w[m].sum() / w.sum())


out = {
    "engine": version("policyengine-uk"),
    "year": YEAR,
    "swap_set": list(SWAP),
    "files": {},
}
for name, path in FILES.items():
    t0 = time.time()
    sim = Microsimulation(dataset=path)
    hh = lambda v: np.asarray(sim.calculate(v, YEAR, map_to="household").values, float)
    with pd.HDFStore(path, "r") as st:
        person = st["person"]
        h5hh = st["household"]
    hid = h5hh["household_id"].to_numpy()
    p_idx = pd.Index(hid).get_indexer(person["person_household_id"].to_numpy())
    assert (p_idx >= 0).all()
    hw = hh("household_weight")
    pw = hw[p_idx]
    assert len(hw) == len(hid) and np.allclose(
        hw, h5hh["household_weight"].to_numpy(float)
    ), "sim household order != H5 order"
    assert len(np.asarray(sim.calculate("age", YEAR).values)) == len(person), (
        "sim person count != H5"
    )
    equiv = hh("household_equivalisation_bhc")
    thr = float(hh("poverty_threshold_bhc")[0])
    base_income = hh("hbai_household_net_income")
    h5_p_idx = p_idx

    def reported_hh(cols):
        v = sum(
            pd.to_numeric(person[c], errors="coerce").fillna(0).to_numpy(float)
            for c in cols
            if c in person
        )
        return np.bincount(h5_p_idx, weights=v, minlength=len(hid))

    sim_amt = {k: hh(k) for k in SWAP}
    rep_amt = {k: reported_hh(cols) for k, cols in SWAP.items()}
    weightings = {"calibrated": hw}
    if name == "v20_national":
        with pd.HDFStore(SPINE_S, "r") as st:
            design = st["household"]["household_weight"].to_numpy(float)
        weightings["spine_s_design"] = design * hw.sum() / design.sum()
    entry = {
        "totals_bn": {
            k: {
                "simulated": float((sim_amt[k] * hw).sum() / 1e9),
                "reported": float((rep_amt[k] * hw).sum() / 1e9),
                "simulated_recipient_hh_m": float(hw[sim_amt[k] > 0].sum() / 1e6),
                "reported_recipient_hh_m": float(hw[rep_amt[k] > 0].sum() / 1e6),
            }
            for k in SWAP
        },
        "weightings": {},
    }
    for wname, w in weightings.items():
        pwx = w[p_idx]

        def rates(income):
            eq = income / equiv
            eqp = eq[p_idx]
            med = wmedian(eqp, pwx)
            return {
                "abs_bhc_pct": 100 * wshare(eqp < thr, pwx),
                "rel_bhc_pct": 100 * wshare(eqp < 0.6 * med, pwx),
                "median_week": med / 52,
                "fixed_line_60pct_of_base_median_pct": None,
            }

        base = rates(base_income)
        base_med = base["median_week"] * 52
        res = {"engine_simulated": base}
        full = base_income - sum(sim_amt.values()) + sum(rep_amt.values())
        r = rates(full)
        r["fixed_line_60pct_of_base_median_pct"] = 100 * wshare(
            (full / equiv)[p_idx] < 0.6 * base_med, pwx
        )
        res["all_reported"] = r
        for k in SWAP:
            inc = base_income - sim_amt[k] + rep_amt[k]
            r = rates(inc)
            r["fixed_line_60pct_of_base_median_pct"] = 100 * wshare(
                (inc / equiv)[p_idx] < 0.6 * base_med, pwx
            )
            res[f"only_{k}_reported"] = r
        entry["weightings"][wname] = res
        print(
            name,
            wname,
            "abs",
            round(base["abs_bhc_pct"], 2),
            "->",
            round(res["all_reported"]["abs_bhc_pct"], 2),
            "| rel",
            round(base["rel_bhc_pct"], 2),
            "->",
            round(res["all_reported"]["rel_bhc_pct"], 2),
            flush=True,
        )
    out["files"][name] = entry
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(name, "done", round(time.time() - t0), "s", flush=True)
print("ALL DONE")
