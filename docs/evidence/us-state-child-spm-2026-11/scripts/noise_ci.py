"""How much of the state gap variance noise explains: null test with exact tail probabilities, a Q-profile interval on tau, and the same fits on the ACS-local file."""

import json
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

SP = os.path.dirname(os.path.abspath(__file__))
cen = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
cse = cen.moe / 1.645
T = pd.read_csv(f"{SP}/run/waterfall.csv", index_col=0)
out = {}


def fit(g, se, lab):
    v = se**2 + cse.reindex(g.index) ** 2
    k = len(g)

    def Q(t2):  # noqa: N802
        w = 1 / (v + t2)
        m = (w * g).sum() / w.sum()
        return (w * (g - m) ** 2).sum()

    Q0 = Q(0.0)
    w = 1 / v
    dl = max(0, (Q0 - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))

    def bound(p):
        target = stats.chi2.ppf(p, k - 1)
        if Q(0.0) < target:
            return 0.0
        return optimize.brentq(lambda t: Q(t) - target, 0.0, 400.0)

    lo, hi = bound(0.975), bound(0.025)
    var = g.var(ddof=1)
    r = dict(
        Q=Q0,
        df=k - 1,
        p=float(stats.chi2.sf(Q0, k - 1)),
        tau=np.sqrt(dl),
        tau_lo=np.sqrt(lo),
        tau_hi=np.sqrt(hi),
        var=var,
        noise_share=1 - dl / var,
        noise_share_lo=1 - hi / var,
        noise_share_hi=1 - lo / var,
        mean_v=float(v.mean()),
    )
    out[lab] = r
    print(lab, {a: round(b, 3) for a, b in r.items()})


g = T.model - T.census
se = T.se_model_rep
fit(g, se, "default_replicate")
fit(g, np.sqrt(T.se_model_rep**2 + T.rep_bias**2), "default_replicate_rmse")
fit(g, T.se_model_lin, "default_linearized")
print(
    "replicate over linearized: median",
    round((T.se_model_rep / T.se_model_lin).median(), 2),
    "range",
    round((T.se_model_rep / T.se_model_lin).min(), 2),
    round((T.se_model_rep / T.se_model_lin).max(), 2),
)
print(
    "mean abs replicate bias",
    round(T.rep_bias.abs().mean(), 2),
    "largest",
    T.rep_bias.abs().sort_values().tail(3).round(2).to_dict(),
)
d = pd.read_csv(f"{SP}/run/dense_fit.csv", index_col=0)
fit(d.model - d.rate, d.se, "dense_published")
e = pd.read_csv(f"{SP}/run/acs_eval.csv", index_col=0)
fit(e.acs_local - e.census, e.se, "acs_local_all")
fit(e.acs_only - e.census, e.acs_only_se, "acs_rows")
fit(e.donor_only - e.census, e.donor_only_se, "acs_local_donors")
# null test on the default file: common shift plus noise
rng = np.random.default_rng(1)
S = 20000
sy = cen.rate.reindex(T.index)
cv = cse.reindex(T.index) ** 2
w_ = 1 / cv
m0 = (w_ * sy).sum() / w_.sum()
q0 = (w_ * (sy - m0) ** 2).sum()
t0 = max(0, (q0 - 50) / (w_.sum() - (w_**2).sum() / w_.sum()))
Bc = t0 / (t0 + cv)
tr = m0 + Bc * (sy - m0)
ps = np.sqrt(Bc * cv)
moe = cen.moe.reindex(T.index).values


def null(se, sh, tx, obs_g):
    Tr = tr.values + ps.values * rng.standard_normal((S, 51))
    C = Tr + np.sqrt(cv.values) * rng.standard_normal((S, 51))
    M = Tr + sh + tx * rng.standard_normal((S, 51)) + se * rng.standard_normal((S, 51))
    h = M - C
    mad = np.abs(h).mean(1)
    sd = h.std(1, ddof=1)
    b = (np.abs(h) > moe).sum(1)
    q = lambda a: [round(float(x), 2) for x in np.percentile(a, [2.5, 5, 50, 95, 97.5])]
    ob = dict(
        mad=float(obs_g.abs().mean()),
        sd=float(obs_g.std()),
        beyond=int((obs_g.abs() > moe).sum()),
    )
    return dict(
        mad=q(mad),
        sd=q(sd),
        beyond=q(b),
        obs=ob,
        p_mad=float((mad >= ob["mad"]).mean()),
        p_sd=float((sd >= ob["sd"]).mean()),
        p_beyond=float((b >= ob["beyond"]).mean()),
    )


n0 = null(T.se_model_rep.values, g.mean(), 0.0, g)
n1 = null(T.se_model_rep.values, g.mean(), out["default_replicate"]["tau"], g)
print("null shift+noise", n0)
print("null shift+noise+tau", n1)
out["null_shift_noise"] = n0
out["null_shift_noise_tau"] = n1
# the ACS-local file under the same null
ea = e.reindex(T.index)
ga = ea.acs_local - ea.census
na = null(ea.se.values, ga.mean(), 0.0, ga)
print("ACS-local null", na)
out["acs_null_shift_noise"] = na
json.dump(out, open(f"{SP}/run/noise_ci.json", "w"), indent=1)
