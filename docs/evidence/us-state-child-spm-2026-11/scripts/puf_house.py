"""Housing receipt on the two support halves of the dense file: same source households, ASEC measured vs PUF clone reimputed."""

import glob
import os

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
f = [x for x in glob.glob(S + "15cefac4*/*.h5") if "calib" not in x][0]
s = pd.read_hdf(f, "spm_unit")
print([c for c in s.columns if "source" in c or "support" in c][:8])
ch = s.spm_unit_support_channel.astype(str)
cl = s.spm_unit_support_clone_index
a = (
    s[(ch == "asec")]
    .set_index("spm_unit_source_id")
    .receives_housing_assistance.astype(bool)
)
b = (
    s[(ch == "puf_tax_detail") & (cl == 1)]
    .set_index("spm_unit_source_id")
    .receives_housing_assistance.astype(bool)
)
print(
    "unweighted rate asec",
    round(a.mean(), 4),
    "puf clone",
    round(b.mean(), 4),
    "units",
    len(a),
    len(b),
)
m = pd.concat([a.rename("asec"), b.rename("puf")], axis=1).dropna()
print(pd.crosstab(m.asec, m.puf, normalize="index").round(3).to_string())
print(
    "rate of puf given asec=1",
    round(m.puf[m.asec.astype(bool)].mean(), 3),
    "given asec=0",
    round(m.puf[~m.asec.astype(bool)].mean(), 3),
)
u = pd.read_parquet(
    f"{SP}/run/dense_units.parquet",
    columns=[
        "spm_unit_id",
        "SPM_RESOURCES",
        "SPM_POVTHRESHOLD",
        "kids",
        "household_support_channel",
    ],
)
u = u.merge(s[["spm_unit_id", "receives_housing_assistance"]], on="spm_unit_id")
u["low"] = u.SPM_RESOURCES < 2 * u.SPM_POVTHRESHOLD
print(
    u[u.kids > 0]
    .groupby(["household_support_channel", "low"])
    .receives_housing_assistance.apply(lambda v: v.astype(bool).mean())
    .round(4)
    .to_dict()
)
