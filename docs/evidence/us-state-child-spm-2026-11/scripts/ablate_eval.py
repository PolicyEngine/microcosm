"""State and national child SPM rates under each stored calibration re-solve (weights only; record flags unchanged)."""

import glob
import json
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser("~/reviews/pe30-population-2026-09-27/out/calib_resolve")
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
h = pd.read_hdf(
    S + "9a814a3b3b53c0ecd6e1737b6ec862c31300ef6f/populace_us_2024.h5", "household"
)[["household_id"]]
pos = pd.Series(np.arange(len(h)), index=h.household_id.values)
u = pd.read_parquet(f"{SP}/run/default_units.parquet")
u = u[u.kids > 0].copy()
ix = pos.reindex(u.household_id.values).values
assert not np.isnan(ix).any()
ix = ix.astype(int)
sr = pd.read_csv(f"{SP}/asec/state_raw.csv", index_col=0)
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
E = (u.SPM_RESOURCES.fillna(0) < u.SPM_POVTHRESHOLD.fillna(0)).values.astype(float)
P25 = u["spm_unit_is_in_spm_poverty__2025"].values.astype(float)
P24 = u["spm_unit_is_in_spm_poverty__2024"].values.astype(float)
st = u.state.values
k = u.kids.values
z = np.load(
    S + "8ab57ffc2ca41d8af631ff62ebbce95e966299f3/populace_us_2024_calibration.npz",
    allow_pickle=True,
)


def ev(w):
    W = w[ix] * k
    den = pd.Series(W).groupby(st).sum()
    out = {}
    for n, f in (("E", E), ("P24", P24), ("P25", P25)):
        r = 100 * pd.Series(W * f).groupby(st).sum() / den
        out[n] = r
        out["nat" + n] = 100 * (W * f).sum() / W.sum()
    out["kids"] = W.sum()
    wh = np.bincount(ix, weights=None)  # unused  # noqa: F841
    return out


rows = []
base = {"initial": z["initial_household_weight"], "published": z["household_weight"]}
for n, w in base.items():
    rows.append((n, ev(np.asarray(w, float))))
for f in sorted(glob.glob(D + "/*.npz")):
    n = os.path.basename(f)[:-4]
    rows.append((n, ev(np.load(f)["weights"])))
pool = sr["avg_22_24"]
cen = c.rate
T = []
for n, o in rows:
    T.append(
        dict(
            run=n,
            natE=o["natE"],
            natP24=o["natP24"],
            natP25=o["natP25"],
            kids_m=o["kids"] / 1e6,
            E_pool_mean=(o["E"] - pool).mean(),
            E_pool_mad=(o["E"] - pool).abs().mean(),
            E_pool_corr=np.corrcoef(o["E"], pool)[0, 1],
            P25_cen_mean=(o["P25"] - cen).mean(),
            P25_cen_mad=(o["P25"] - cen).abs().mean(),
            P25_cen_corr=np.corrcoef(o["P25"], cen)[0, 1],
        )
    )
T = pd.DataFrame(T).set_index("run")
T.to_csv(f"{SP}/run/ablate_eval.csv")
pd.set_option("display.width", 220)
print(T.round(2).sort_values("natE").to_string())
pickle_path = f"{SP}/run/ablate_states.pkl"
pd.to_pickle(
    {n: {k_: v for k_, v in o.items() if k_ in ("E", "P24", "P25")} for n, o in rows},
    pickle_path,
)
