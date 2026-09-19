"""PROXY AGI by band on a Frame checkpoint (numeric columns only). Proxy = sum of person-level
AGI-entering income inputs present at that stage; NOT engine AGI."""

import json
import sys

import h5py
import numpy as np
import pandas as pd

INC = [
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "taxable_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "long_term_capital_gains_before_response",
    "short_term_capital_gains",
    "non_sch_d_capital_gains",
    "taxable_private_pension_income",
    "taxable_ira_distributions",
    "taxable_401k_distributions",
    "taxable_403b_distributions",
    "taxable_sep_distributions",
    "keogh_distributions",
    "alimony_income",
    "miscellaneous_income",
    "rental_income",
    "farm_operations_income",
    "farm_income",
    "farm_rent_income",
    "unemployment_compensation",
    "estate_income",
    "partnership_income",
    "s_corp_income",
    "salt_refund_income",
]


def load(p):
    f = h5py.File(p, "r")
    g = f["_populace_frame_checkpoint"]
    md = json.loads(bytes(g["metadata_json"][()]).decode())
    out = {}
    for i, t in enumerate(md["tables"]):
        cols = {}
        for j, c in enumerate(t["columns"]):
            if c.get("encoding") == "numpy":
                node = g[f"tables/t{i:05d}/columns/c{j:05d}"]
                cols[c["name"]] = (
                    node[()]
                    if isinstance(node, h5py.Dataset)
                    else node[list(node.keys())[0]][()]
                )
        out[t.get("name") or t.get("entity") or str(i)] = pd.DataFrame(cols)
    w = g["weights/w00000"][()]
    return out, w, md


p = sys.argv[1]
T, w, md = load(p)
print("file", p.split("/")[-1], "| tables", {k: v.shape for k, v in T.items()})
pe = T["person"]
hh = T["household"].copy()
hh["w"] = w
use = [c for c in INC if c in pe.columns]
print(
    "income cols present:",
    len(use),
    "missing:",
    [c for c in INC if c not in pe.columns],
)
if use:
    pe = pe.assign(proxy=pe[use].sum(axis=1))
    wcol = "employment_income_before_lsr"
    print("PROXY BASIS: policyengine income inputs")
else:
    pe = pe.assign(proxy=pe["PTOTVAL"].astype(float))
    wcol = "WSAL_VAL"
    print("PROXY BASIS: ASEC PTOTVAL (total person income); raw ASEC stage")
    for c in [
        "WSAL_VAL",
        "SEMP_VAL",
        "INT_VAL",
        "DIV_VAL",
        "CAP_VAL",
        "RNT_VAL",
        "PTOTVAL",
        "AGI",
    ]:
        if c in pe.columns:
            print(
                "  max",
                c,
                f"= {pe[c].max():.0f}",
                "| n>=1M:",
                int((pe[c] >= 1e6).sum()),
                "| n>=5M:",
                int((pe[c] >= 5e6).sum()),
            )
tu = pe.groupby("person_tax_unit_id").agg(
    proxy=("proxy", "sum"), hid=("person_household_id", "first"), wages=(wcol, "sum")
)
ci = "household_support_clone_index"
hcols = ["household_id", "w"] + ([ci] if ci in hh.columns else [])
tu = tu.join(hh[hcols].set_index("household_id"), on="hid")
edges = [-np.inf, 0, 5e5, 1e6, 1.5e6, 2e6, 5e6, 1e7, np.inf]
lab = ["<0", "0-500k", "500k-1M", "1M-1.5M", "1.5M-2M", "2M-5M", "5M-10M", "10M+"]
tu["band"] = pd.cut(tu.proxy, edges, labels=lab, right=False)
res = {}


def tab(d, name):
    g = d.groupby("band", observed=False)
    t = pd.DataFrame(
        {
            "n_unw": g.size(),
            "units_w": g.w.sum(),
            "proxy_B": g.apply(lambda x: (x.proxy * x.w).sum()) / 1e9,
        }
    )
    print(
        "\n==",
        name,
        "| hh weight sum %.1fM" % (hh.w.sum() / 1e6),
        "| tax units",
        len(d),
        "| max proxy $%.1fM" % (d.proxy.max() / 1e6),
        "| max wages $%.2fM" % (d.wages.max() / 1e6),
    )
    print(t.round(1).to_string())
    res[name] = json.loads(t.to_json())


tab(tu, "ALL")
if ci in tu.columns:
    for k, d in tu.groupby(ci):
        tab(d, f"clone_index={k}")
json.dump(res, open(sys.argv[2], "w"), indent=1)
