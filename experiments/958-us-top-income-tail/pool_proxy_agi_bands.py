"""PROXY AGI by band on a base pool H5 (pandas HDFStore layout).
Proxy = sum of person-level income inputs that enter AGI (no ALDs, no SS taxability,
no loss limits). It is a locator for the top tail, NOT engine AGI."""

import json
import sys

import numpy as np
import pandas as pd

p = sys.argv[1]
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
cols = pd.read_hdf(p, "person", start=0, stop=1).columns
use = [c for c in INC if c in cols]
miss = [c for c in INC if c not in cols]
pe = pd.read_hdf(
    p,
    "person",
    columns=use
    + ["person_tax_unit_id", "person_household_id", "person_support_channel"],
)
hh = pd.read_hdf(
    p,
    "household",
    columns=[
        "household_id",
        "household_weight",
        "household_support_channel",
        "household_support_clone_index",
    ],
)
pe["proxy"] = pe[use].sum(axis=1)
tu = pe.groupby("person_tax_unit_id").agg(
    proxy=("proxy", "sum"),
    hh=("person_household_id", "first"),
    wages=("employment_income_before_lsr", "sum"),
)
tu = tu.join(hh.set_index("household_id"), on="hh")
edges = [-np.inf, 0, 5e5, 1e6, 1.5e6, 2e6, 5e6, 1e7, np.inf]
lab = ["<0", "0-500k", "500k-1M", "1M-1.5M", "1.5M-2M", "2M-5M", "5M-10M", "10M+"]
tu["band"] = pd.cut(tu.proxy, edges, labels=lab, right=False)
out = {}


def tab(d, name):
    g = d.groupby("band", observed=False)
    t = pd.DataFrame(
        {
            "n_unw": g.size(),
            "units_w": g.household_weight.sum(),
            "proxy_B": g.apply(lambda x: (x.proxy * x.household_weight).sum()) / 1e9,
        }
    )
    print(
        "\n==",
        name,
        "| tax units",
        len(d),
        "| max proxy AGI $%.1fM" % (d.proxy.max() / 1e6),
        "| max wages $%.2fM" % (d.wages.max() / 1e6),
    )
    print(t.round(1).to_string())
    out[name] = json.loads(t.to_json())


print("file", p)
print("missing cols", miss)
tab(tu, "ALL")
for ch, d in tu.groupby("household_support_channel"):
    tab(d, "channel=" + str(ch))
for ci, d in tu.groupby("household_support_clone_index"):
    tab(d, "clone_index=" + str(ci))
json.dump(out, open(sys.argv[2], "w"), indent=1)
