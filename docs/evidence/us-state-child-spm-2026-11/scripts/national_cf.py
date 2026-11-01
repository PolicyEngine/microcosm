"""One national simulation of the default file per process: base, aged pooled incomes, or the 2021 CTC reform.
Usage: national_cf.py <mode>   mode in base, aged, reform"""

import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import extract_state as ex
from policyengine_core.data import Dataset
from policyengine_core.reforms import Reform
from policyengine_us import Microsimulation

mode = sys.argv[1]
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
H5 = S + "9a814a3b3b53c0ecd6e1737b6ec862c31300ef6f/populace_us_2024.h5"
ex.load(H5)
t = ex.tables
people = t["person"]
households = t["household"]
if mode == "aged":
    # Census resources of the child at the 20th percentile, 2022 and 2023 to 2024 (raw ASEC, this session).
    F = {2022: 46508 / 42020, 2023: 46508 / 44342, 2024: 1.0}
    hc = households.set_index("household_id")["household_support_channel"]
    ch = people["person_household_id"].map(hc).values
    pat = re.compile(
        r"(income|dividend|_distributions|^social_security_|benefits|compensation|capital_gains|non_sch_d)"
    )
    cols = [
        c
        for c in people.columns
        if c in ex.known
        and pat.search(c)
        and "last_year" not in c
        and people[c].dtype.kind == "f"
    ]
    print("aged columns", len(cols), cols, flush=True)
    f = people["source_year"].map(F).values
    f = np.where(ch == "asec", f, 1.0)
    people = people.copy()
    for c in cols:
        people[c] = people[c].values * f
    t["person"] = people
    print("asec share aged >1", (f > 1).mean(), flush=True)
# Same flat frame as state_file, with no state filter.
orig = households


class All:
    pass


columns = {
    n: people[n].values
    for n in people.columns
    if n in ex.known or n == "person_id" or n.startswith("person_")
}


def add_group(table, group):
    ids = f"{group}_id"
    membership = people[f"person_{group}_id"].values
    rows = table.set_index(ids).index.get_indexer(membership)
    for name in table.columns:
        if name in ex.known:
            columns[name] = table[name].values[rows]
    columns[ids] = membership


add_group(households, "household")
for g in ex.GROUPS:
    add_group(t[g], g)
frame = pd.DataFrame({f"{k}__2024": v for k, v in columns.items()})
kw = {}
if mode == "reform":
    d = "2025-01-01.2025-12-31"
    kw["reform"] = Reform.from_dict(
        {
            "gov.irs.credits.ctc.amount.arpa[0].amount": {d: 3600},
            "gov.irs.credits.ctc.amount.arpa[1].amount": {d: 3000},
            "gov.irs.credits.ctc.refundable.fully_refundable": {d: True},
            "gov.irs.credits.ctc.phase_out.arpa.in_effect": {d: True},
        },
        country_id="us",
    )
t0 = time.time()
sim = Microsimulation(dataset=Dataset.from_dataframe(frame, 2024), **kw)
res = {"spm_unit_id": sim.calculate("spm_unit_id", 2024).values}
for y in (2024, 2025) if mode != "reform" else (2025,):
    for n in (
        "spm_unit_is_in_spm_poverty",
        "spm_unit_net_income",
        "spm_unit_spm_threshold",
    ):
        res[f"{n}__{y}"] = np.asarray(
            sim.calculate(n, y, map_to="spm_unit").values, float
        )
pd.DataFrame(res).to_parquet(f"{SP}/run/cf_{mode}.parquet")
print(mode, "seconds", round(time.time() - t0), flush=True)
