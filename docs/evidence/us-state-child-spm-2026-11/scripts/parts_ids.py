"""Split one state's households into parts for acs_parts.sh: contiguous in the split's household order, with about equal
numbers of people.

policyengine-us 2.90.2 resolves parent links within a household or a tax unit, and reads across households only when a tax
unit spans households or a Medicaid claiming tax unit is set (_parent_links.py). Model formulas cannot draw random numbers
(policyengine-core 3.34.2's random raises). So this refuses a state where any tax unit, SPM unit, family or marital unit spans
households, or where any table carries a claiming tax unit column. acs_parts_cmp.py checks the parts against whole-state runs.
Stops itself with exit 75 if peak RSS passes RSS_CAP_GIB (default 8).
Usage: parts_ids.py <fips> <parts> <folder>
"""

import os
import resource
import sys
import threading
import time

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
fips, K, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
GROUPS = ("tax_unit", "spm_unit", "family", "marital_unit")
CAP = float(os.environ.get("RSS_CAP_GIB", "8")) * 2**30


def watch():
    while True:
        time.sleep(2)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > CAP:
            print("rss stop parts_ids", fips, flush=True)
            os._exit(75)


threading.Thread(target=watch, daemon=True).start()
h = pd.read_pickle(f"{SP}/run/acs_split/{fips:02d}-household.pkl.gz")
p = pd.read_pickle(f"{SP}/run/acs_split/{fips:02d}-person.pkl.gz")
claiming = [c for c in [*h.columns, *p.columns] if "claiming" in c]
for g in GROUPS:
    claiming += [
        c
        for c in pd.read_pickle(f"{SP}/run/acs_split/{fips:02d}-{g}.pkl.gz").columns
        if "claiming" in c
    ]
assert not claiming, f"claiming tax unit columns {claiming}"
p = p.loc[
    p.person_household_id.isin(h.household_id),
    ["person_household_id", *[f"person_{g}_id" for g in GROUPS]],
]
for g in GROUPS:
    spans = p.groupby(f"person_{g}_id").person_household_id.nunique()
    assert (spans == 1).all(), (
        f"{int((spans > 1).sum())} of {len(spans)} {g}s span households"
    )
people = (
    p.person_household_id.value_counts().reindex(h.household_id, fill_value=0).values
)
edges = np.searchsorted(np.cumsum(people), np.arange(1, K) * people.sum() / K)
parts = np.split(h.household_id.values, edges)
assert len(parts) == K and all(len(ids) for ids in parts), [len(ids) for ids in parts]
os.makedirs(out, exist_ok=True)
for k, ids in enumerate(parts):
    np.save(f"{out}/p{k}.tmp.npy", ids)
    os.replace(f"{out}/p{k}.tmp.npy", f"{out}/p{k}.npy")
    print(
        fips,
        "part",
        k,
        "households",
        len(ids),
        "people",
        int(people[np.isin(h.household_id.values, ids)].sum()),
        flush=True,
    )
