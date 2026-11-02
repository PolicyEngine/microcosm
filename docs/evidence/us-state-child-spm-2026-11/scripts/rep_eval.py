"""Waterfall by state with replicate-calibration standard errors, plus the noise-only test on replicate SEs."""

import glob
import itertools
import json
import math
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
H5 = S + "9a814a3b3b53c0ecd6e1737b6ec862c31300ef6f/populace_us_2024.h5"
hh = pd.read_hdf(H5, "household")["household_id"].to_numpy()
p = (
    pd.read_hdf(H5, "person")[
        ["person_household_id", "source_year", "source_household_id"]
    ]
    .drop_duplicates("person_household_id")
    .set_index("person_household_id")
    .reindex(hh)
)
clu = pd.factorize(p.source_year.astype(str) + ":" + p.source_household_id.astype(str))[
    0
]
ncl = clu.max() + 1
z = np.load(
    S + "8ab57ffc2ca41d8af631ff62ebbce95e966299f3/populace_us_2024_calibration.npz",
    allow_pickle=True,
)
w0 = z["initial_household_weight"].astype(float)
w1 = z["household_weight"].astype(float)
pos = pd.Series(np.arange(len(hh)), index=hh)
u = pd.read_parquet(f"{SP}/run/default_units.parquet")
u = u[u.kids > 0].copy()
ix = pos.reindex(u.household_id.values).values.astype(int)
k = u.kids.values.astype(float)
st = u.state.values
states = sorted(set(st))
sidx = pd.Categorical(st, categories=states).codes
c = lambda n: u[n].fillna(0).values.astype(float)
q = lambda n, y=2024: u[f"{n}__{y}"].values.astype(float)
Rc = c("SPM_RESOURCES")
Tc = c("SPM_POVTHRESHOLD")
Tp = q("spm_unit_spm_threshold")
Rp = q("spm_unit_net_income")
ratio = Tp / Tc
pf = pd.Series(ratio).groupby(u.source_year.values).transform("median").values
Ty = Tc * pf
P24 = q("spm_unit_is_in_spm_poverty", 2024)
P25 = q("spm_unit_is_in_spm_poverty", 2025)


# Three factors: price year of thresholds, threshold rest, resources.
def fl(a, b, r):
    T = (Tp if b else Ty) if a else (Tp / pf if b else Tc)
    return ((Rp if r else Rc) < T).astype(float)


F = {bits: fl(*bits) for bits in itertools.product([0, 1], repeat=3)}
assert np.abs(F[(1, 1, 1)] - P24).mean() < 1e-3, np.abs(F[(1, 1, 1)] - P24).mean()


def shap(i):
    out = 0
    for bits in itertools.product([0, 1], repeat=2):
        s = sum(bits)
        w = math.factorial(s) * math.factorial(2 - s) / math.factorial(3)
        a = bits[:i] + (1,) + bits[i:]
        b = bits[:i] + (0,) + bits[i:]
        out = out + w * (F[a] - F[b])
    return out


Sh = [shap(i) for i in range(3)]
E = F[(0, 0, 0)]
stages = {
    "E": E,
    "priceyear": Sh[0],
    "threshold_rest": Sh[1],
    "resources": Sh[2],
    "year": P25 - P24,
    "P25": P25,
    "P24": P24,
}


def rate(W, f):  # noqa: N803
    return (
        100
        * np.bincount(sidx, W * f, minlength=51)
        / np.bincount(sidx, W, minlength=51)
    )


def stageset(w, w_init):
    W = w[ix] * k
    out = {n: rate(W, f) for n, f in stages.items()}
    out["D"] = rate(w_init[ix] * k, E)
    return out


base = stageset(w1, w0)
reps = []
fs = sorted(glob.glob(f"{SP}/run/rep/r*.npz"))
for f in fs:
    r = int(os.path.basename(f)[1:4])
    w = np.load(f)["weights"]
    m = np.random.default_rng(20261101 + r).poisson(1.0, ncl)[clu].astype(float)
    wi = w0 * m
    wi *= w0.sum() / wi.sum()
    reps.append(stageset(w, wi))
print("replicates", len(reps))
names = list(base)
A = {n: np.array([r[n] for r in reps]) for n in names}
se = {n: pd.Series(A[n].std(0, ddof=1), index=states) for n in names}
seW = pd.Series((A["E"] - A["D"]).std(0, ddof=1), index=states)  # noqa: N816
b = {n: pd.Series(base[n], index=states) for n in names}
cen = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
sr = pd.read_csv(f"{SP}/asec/state_raw.csv", index_col=0)
noise = pd.read_csv(f"{SP}/run/default_noise.csv", index_col=0)
T = pd.DataFrame(
    {
        "census": cen.rate,
        "pool": sr.avg_22_24,
        "D": b["D"],
        "E": b["E"],
        "priceyear": b["priceyear"],
        "threshold_rest": b["threshold_rest"],
        "resources": b["resources"],
        "year": b["year"],
        "model": b["P25"],
    }
)
T["se_model_rep"] = se["P25"]
T["se_model_lin"] = noise.se
T["rep_bias"] = A["P25"].mean(0) - b["P25"].values
T["se_E_rep"] = se["E"]
T["se_D_rep"] = se["D"]
T["se_weights_step"] = seW
for n in ("priceyear", "threshold_rest", "resources", "year"):
    T["se_" + n] = se[n]
T.to_csv(f"{SP}/run/waterfall.csv")
print(
    "ratio rep/lin median",
    (T.se_model_rep / T.se_model_lin).median().round(3),
    "median rep se",
    T.se_model_rep.median().round(2),
    "rms",
    np.sqrt((T.se_model_rep**2).mean()).round(2),
)
print("mean rep bias", T.rep_bias.mean().round(2))
top = [
    "DC",
    "DE",
    "TN",
    "MD",
    "FL",
    "RI",
    "VT",
    "MO",
    "LA",
    "MT",
    "MS",
    "HI",
    "AR",
    "CA",
    "TX",
]
pd.set_option("display.width", 220)
print(
    T.loc[
        top,
        [
            "census",
            "pool",
            "D",
            "E",
            "priceyear",
            "threshold_rest",
            "resources",
            "year",
            "model",
        ],
    ]
    .round(1)
    .to_string()
)
print(
    T.loc[
        top,
        [
            "se_D_rep",
            "se_weights_step",
            "se_E_rep",
            "se_priceyear",
            "se_threshold_rest",
            "se_resources",
            "se_year",
            "se_model_rep",
            "se_model_lin",
        ],
    ]
    .round(1)
    .to_string()
)
print(
    T[
        [
            "census",
            "pool",
            "D",
            "E",
            "priceyear",
            "threshold_rest",
            "resources",
            "year",
            "model",
        ]
    ]
    .mean()
    .round(2)
    .to_dict()
)
d = T.copy()
cse = cen.moe / 1.645
d["gap"] = d.model - d.census
v = d.se_model_rep**2 + cse**2
w = 1 / v
mu = (w * d.gap).sum() / w.sum()
Q = (w * (d.gap - mu) ** 2).sum()
t2 = max(0, (Q - 50) / (w.sum() - (w**2).sum() / w.sum()))
ws = 1 / (v + t2)
mur = (ws * d.gap).sum() / ws.sum()
print(
    "DL rep SE: Q",
    round(Q, 1),
    "tau",
    round(np.sqrt(t2), 2),
    "mu_re",
    round(mur, 2),
    "se_mu",
    round(np.sqrt(1 / ws.sum()), 2),
)
rng = np.random.default_rng(1)
sy = cen.rate
cv = cse**2
w_ = 1 / cv
m0 = (w_ * sy).sum() / w_.sum()
q0 = (w_ * (sy - m0) ** 2).sum()
t0 = max(0, (q0 - 50) / (w_.sum() - (w_**2).sum() / w_.sum()))
Bc = t0 / (t0 + cv)
tr = m0 + Bc * (sy - m0)
ps = np.sqrt(Bc * cv)
for lab, sh, tx in (
    ("noise + common shift", d.gap.mean(), 0.0),
    ("noise + shift + tau", d.gap.mean(), np.sqrt(t2)),
):
    Sn = 20000
    Tr = tr.values + ps.values * rng.standard_normal((Sn, 51))
    C = Tr + cse.values * rng.standard_normal((Sn, 51))
    M = (
        Tr
        + sh
        + tx * rng.standard_normal((Sn, 51))
        + d.se_model_rep.values * rng.standard_normal((Sn, 51))
    )
    g = M - C
    qq = lambda a: np.round(np.percentile(a, [5, 50, 95]), 2)
    print(
        lab,
        "MAD",
        qq(np.abs(g).mean(1)),
        "sd",
        qq(g.std(1, ddof=1)),
        "beyond",
        qq((np.abs(g) > cen.moe.values).sum(1)),
        "corr",
        qq([np.corrcoef(M[i], C[i])[0, 1] for i in range(3000)]),
    )
