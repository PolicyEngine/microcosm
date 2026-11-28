"""State misses that persist: dense initial-weight gaps against ACS-row gaps, which share no household.

The covariance comes from overlap_fits.csv, which overlap.py writes.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

SP = os.path.dirname(os.path.abspath(__file__))
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
cse = c.moe / 1.645
fits = pd.read_csv(f"{SP}/run/overlap_fits.csv")


def cov(weights):
    return fits[
        (fits.file == "dense") & (fits.weights == weights) & (fits.rows == "all")
    ].mean_cov.iloc[0]


s = pd.read_csv(f"{SP}/run/initial_fit_states.csv", index_col=0)
e = pd.read_csv(f"{SP}/run/acs_eval.csv", index_col=0)
t = pd.DataFrame(
    {
        "dense_initial": s.dense_W0,
        "dense_initial_se": s.dense_W0_se,
        "acs_rows": e.acs_only - e.census,
        "acs_rows_se": e.acs_only_se,
        "census_se": cse,
    }
)
t["z_dense"] = t.dense_initial / np.sqrt(t.dense_initial_se**2 + t.census_se**2)
t["z_acs"] = t.acs_rows / np.sqrt(t.acs_rows_se**2 + t.census_se**2)
t.to_csv(f"{SP}/run/state_persistence.csv")
# ACS rows share no household with Census, so the gaps' sampling covariance equals Var(Census) less Cov(dense, Census).
print(
    f"gap corr dense initial, ACS rows {t.dense_initial.corr(t.acs_rows):.2f}; p {stats.pearsonr(t.dense_initial, t.acs_rows)[1]:.3f}"
)
print(
    f"  Census noise alone would give about {((t.census_se**2).mean() - cov('initial')) / (t.dense_initial.std() * t.acs_rows.std()):.2f}"
)
print(f"gap corr dense published, ACS rows {s.dense_W1.corr(t.acs_rows):.2f}")
print(
    f"  Census noise alone would give about {((t.census_se**2).mean() - cov('published')) / (s.dense_W1.std() * t.acs_rows.std()):.2f}"
)
rng = np.random.default_rng(0)
same = lambda z1, z2, th: (
    (np.sign(z1) == np.sign(z2)) & (np.abs(z1) > th) & (np.abs(z2) > th)
).sum()
for th in (1.28, 1.645, 1.96):
    obs = same(t.z_dense.values, t.z_acs.values, th)
    sim = np.array(
        [
            same(t.z_dense.values, t.z_acs.values[rng.permutation(51)], th)
            for _ in range(20000)
        ]
    )
    pick = t.index[
        (np.sign(t.z_dense) == np.sign(t.z_acs))
        & (t.z_dense.abs() > th)
        & (t.z_acs.abs() > th)
    ].tolist()
    print(
        f"|z|>{th}: {obs} states {pick}, permutation mean {sim.mean():.2f}, p {(sim >= obs).mean():.3f}"
    )
t["joint_z"] = np.where(
    np.sign(t.z_dense) == np.sign(t.z_acs),
    np.minimum(t.z_dense.abs(), t.z_acs.abs()),
    0,
)


def largest(z2):
    return np.where(
        np.sign(t.z_dense.values) == np.sign(z2),
        np.minimum(np.abs(t.z_dense.values), np.abs(z2)),
        0,
    ).max()


sim = np.array([largest(t.z_acs.values[rng.permutation(51)]) for _ in range(200000)])
print(
    f"largest joint z {t.joint_z.max():.3f}, permutation median {np.median(sim):.2f}, p {(sim >= t.joint_z.max()).mean():.3f}"
)
t.to_csv(f"{SP}/run/state_persistence.csv")
print(
    t.sort_values("joint_z", ascending=False)
    .head(8)[["dense_initial", "acs_rows", "z_dense", "z_acs", "joint_z"]]
    .round(2)
    .to_string()
)
