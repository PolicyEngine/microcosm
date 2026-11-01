"""One ACS-local state at current policyengine-us: unit results for 2025, with the 2021 CTC reform, keyed for clustering.
Usage: acs_one.py <fips>"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SP = os.path.dirname(os.path.abspath(__file__))
from policyengine_core.data import Dataset
from policyengine_core.reforms import Reform
from policyengine_us import Microsimulation

fips = int(sys.argv[1])
TAG = os.environ.get("ACS_TAG", "acs")
Y = 2025
f = f"{SP}/run/{TAG}/{fips:02d}.parquet"
if os.path.exists(f):
    sys.exit(0)
t0 = time.time()
frame = pd.read_pickle(f"{SP}/run/{TAG}_cuts/{fips:02d}.pkl")
for c in frame.columns:
    if c.startswith("parent_") and c.endswith("_id__2024"):
        frame[c] = frame[c].fillna(-1).astype(np.int64)
if "spm_unit_spm_universe_status__2024" not in frame:
    # Declared per docs/us-spm-measurement-universe.md: ACS group quarters OUTSIDE, everything else INCLUDED.
    gq = frame.get("TYPEHUGQ__2024")
    frame["spm_unit_spm_universe_status__2024"] = "INCLUDED"
d = "2025-01-01.2025-12-31"
ref = Reform.from_dict(
    {
        "gov.irs.credits.ctc.amount.arpa[0].amount": {d: 3600},
        "gov.irs.credits.ctc.amount.arpa[1].amount": {d: 3000},
        "gov.irs.credits.ctc.refundable.fully_refundable": {d: True},
        "gov.irs.credits.ctc.phase_out.arpa.in_effect": {d: True},
    },
    country_id="us",
)
base = Microsimulation(dataset=Dataset.from_dataframe(frame, 2024))
ids = base.calculate("spm_unit_id", 2024).values
age = base.calculate("age", 2024).values
pu = base.calculate("person_spm_unit_id", 2024).values
res = {
    "spm_unit_id": ids,
    "kids": pd.Series(age < 18).groupby(pu).sum().reindex(ids).values,
}
res["poor"] = np.asarray(
    base.calculate("spm_unit_is_in_spm_poverty", Y, map_to="spm_unit").values, float
)
res["net"] = np.asarray(
    base.calculate("spm_unit_net_income", Y, map_to="spm_unit").values, float
)
res["thr"] = np.asarray(
    base.calculate("spm_unit_spm_threshold", Y, map_to="spm_unit").values, float
)
res["hh"] = (
    base.calculate("household_id", 2024, map_to="spm_unit").values if False else None
)
res.pop("hh")
del base
sim = Microsimulation(dataset=Dataset.from_dataframe(frame, 2024), reform=ref)
res["poor_r"] = np.asarray(
    sim.calculate("spm_unit_is_in_spm_poverty", Y, map_to="spm_unit").values, float
)
pd.DataFrame(res).to_parquet(f)
print(fips, round(time.time() - t0), flush=True)
