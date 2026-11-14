"""Why do Census housing recipients get no subsidy in the model? Real state cuts of the default file, 2024."""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import extract_state as ex
from policyengine_core.data import Dataset
from policyengine_us import Microsimulation

S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
ex.load(glob.glob(S + "9a814a3b*/populace_us_2024.h5")[0])
VARS = "receives_housing_assistance takes_up_housing_assistance_if_eligible is_eligible_for_housing_assistance housing_assistance hud_hap hud_max_subsidy hud_ttp hud_gross_rent spm_unit_allocated_housing_subsidy spm_unit_allocated_tenant_payment spm_unit_capped_housing_subsidy spm_unit_tenure_type".split()
rows = []
for fips in (25, 17, 6, 48):
    sim = Microsimulation(dataset=Dataset.from_dataframe(ex.state_file(fips), 2024))
    d = {"spm_unit_id": sim.calculate("spm_unit_id", 2024).values, "state": fips}
    for v in VARS:
        try:
            d[v] = sim.calculate(v, 2024).values
        except Exception as e:
            print("fail", v, str(e)[:90])
    rows.append(pd.DataFrame(d))
    del sim
d = pd.concat(rows, ignore_index=True)
d["spm_unit_tenure_type"] = d.spm_unit_tenure_type.astype(str)
d.to_parquet(f"{SP}/run/house_sim.parquet")
u = pd.read_parquet(
    f"{SP}/run/default_units.parquet",
    columns=[
        "spm_unit_id",
        "kids",
        "w1",
        "household_support_channel",
        "SPM_CAPHOUSESUB",
        "SPM_TENMORTSTATUS",
    ],
)
d = d.merge(u, on="spm_unit_id")
d = d[
    (d.kids > 0)
    & (d.household_support_channel == "asec")
    & (d.SPM_CAPHOUSESUB.fillna(0) > 0)
].copy()
d["W"] = d.w1 * d.kids
a = lambda m: round(float(np.average(m, weights=d.W)), 3)
print(
    "census recipients",
    len(d),
    "flag",
    a(d.receives_housing_assistance),
    "eligible",
    a(d.is_eligible_for_housing_assistance),
    "takeup",
    a(d.takes_up_housing_assistance_if_eligible),
)
print(
    "housing_assistance>0",
    a(d.housing_assistance > 0),
    "hap>0",
    a(d.hud_hap > 0),
    "max_subsidy>0",
    a(d.hud_max_subsidy > 0),
    "gross_rent>0",
    a(d.hud_gross_rent > 0),
    "gross_rent>ttp",
    a(d.hud_gross_rent > d.hud_ttp),
    "capped>0",
    a(d.spm_unit_capped_housing_subsidy > 0),
)
print(
    "tenure",
    d.groupby("spm_unit_tenure_type").W.sum().div(d.W.sum()).round(3).to_dict(),
)
print(
    "Census tenure code",
    d.groupby("SPM_TENMORTSTATUS").W.sum().div(d.W.sum()).round(3).to_dict(),
)
for n in (
    "hud_gross_rent",
    "hud_ttp",
    "hud_max_subsidy",
    "hud_hap",
    "spm_unit_capped_housing_subsidy",
    "SPM_CAPHOUSESUB",
):
    print(
        n,
        "mean",
        round(float(np.average(d[n], weights=d.W))),
        "p50",
        round(float(d[n].median())),
    )
