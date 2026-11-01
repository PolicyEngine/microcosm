"""ACS-local file with rent on the right households, one state at a time, simulated in memory so no cut reaches disk.
Donors take the base record's right rent; ACS renter heads take twelve times native monthly contract rent. Weights unchanged.
Base simulations for all 51 first; then the 2021 CTC reform for the top-gap states. Pauses while free space reads under 30 GB."""

import gc
import glob
import os
import subprocess
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

S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
FREE = os.path.expanduser("~/reviews/us-hub/scripts/free_gb.py")
Y = 2025
TOP = [11, 10, 47, 24, 12, 44, 50, 29, 22, 30, 28, 15, 5, 25, 23]


def wait_for_space():
    while True:
        gb = float(
            subprocess.run(
                ["python3", FREE], capture_output=True, text=True
            ).stdout.split()[0]
        )
        if gb >= 30:
            return gb
        print("pause, free", gb, flush=True)
        time.sleep(120)


ex.load(S + "725d89993f6b2dc4821fd5c3620f0f3d4d630102/populace_us_2024_acs_local.h5")
de = pd.read_hdf(
    [x for x in glob.glob(S + "15cefac4*/*.h5") if "calib" not in x][0], "household"
)[["household_id", "household_source_id"]]
right = (
    pd.read_parquet(f"{SP}/run/rentfix_dense.parquet")
    .merge(de, on="household_id")
    .drop_duplicates("household_source_id")
    .set_index("household_source_id")
    .rent_right
)
h = ex.tables["household"]
p = ex.tables["person"]
acs = h.household_support_channel.astype(str).str.startswith("acs").values
hr = pd.Series(
    np.where(
        acs,
        12
        * pd.to_numeric(h.acs_monthly_contract_rent, errors="coerce").fillna(0).values,
        h.household_source_id.map(right).values,
    ),
    index=h.household_id.values,
)
assert hr.notna().all()
head = p.is_household_head.astype(bool).values
rent = np.zeros(len(p))
rent[head] = p.person_household_id[head].map(hr).values
print(
    "rent was",
    round(p.pre_subsidy_rent.sum() / 1e9, 2),
    "now",
    round(rent.sum() / 1e9, 2),
    flush=True,
)
p["pre_subsidy_rent"] = rent
ex.tables["person"] = p
sizes = h.groupby("state_fips").size().sort_values()
order = TOP + [int(f) for f in sizes.index if int(f) not in TOP]
os.makedirs(f"{SP}/run/acsrent", exist_ok=True)
os.makedirs(f"{SP}/run/acsrent_r", exist_ok=True)


def frame_for(fips):
    frame = ex.state_file(fips)
    for c in frame.columns:
        if c.startswith("parent_") and c.endswith("_id__2024"):
            frame[c] = frame[c].fillna(-1).astype(np.int64)
    if "spm_unit_spm_universe_status__2024" not in frame:
        frame["spm_unit_spm_universe_status__2024"] = "INCLUDED"
    return frame


for fips in order:
    f = f"{SP}/run/acsrent/{fips:02d}.parquet"
    if os.path.exists(f):
        continue
    gb = wait_for_space()
    t0 = time.time()
    sim = Microsimulation(dataset=Dataset.from_dataframe(frame_for(fips), 2024))
    ids = sim.calculate("spm_unit_id", 2024).values
    age = sim.calculate("age", 2024).values
    pu = sim.calculate("person_spm_unit_id", 2024).values
    res = {
        "spm_unit_id": ids,
        "kids": pd.Series(age < 18).groupby(pu).sum().reindex(ids).values,
    }
    res["poor"] = np.asarray(
        sim.calculate("spm_unit_is_in_spm_poverty", Y, map_to="spm_unit").values, float
    )
    res["net"] = np.asarray(
        sim.calculate("spm_unit_net_income", Y, map_to="spm_unit").values, float
    )
    res["thr"] = np.asarray(
        sim.calculate("spm_unit_spm_threshold", Y, map_to="spm_unit").values, float
    )
    res["snap"] = np.asarray(
        sim.calculate("snap", 2024, map_to="spm_unit").values, float
    )
    res["house"] = np.asarray(
        sim.calculate(
            "spm_unit_capped_housing_subsidy", 2024, map_to="spm_unit"
        ).values,
        float,
    )
    pd.DataFrame(res).to_parquet(f)
    del sim
    gc.collect()
    print(fips, round(time.time() - t0), "free", gb, flush=True)
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
for fips in TOP:
    f = f"{SP}/run/acsrent_r/{fips:02d}.parquet"
    if os.path.exists(f):
        continue
    wait_for_space()
    t0 = time.time()
    sim = Microsimulation(
        dataset=Dataset.from_dataframe(frame_for(fips), 2024), reform=ref
    )
    pd.DataFrame(
        {
            "spm_unit_id": sim.calculate("spm_unit_id", 2024).values,
            "poor_r": np.asarray(
                sim.calculate(
                    "spm_unit_is_in_spm_poverty", Y, map_to="spm_unit"
                ).values,
                float,
            ),
        }
    ).to_parquet(f)
    del sim
    gc.collect()
    print("reform", fips, round(time.time() - t0), flush=True)
print("done", flush=True)
