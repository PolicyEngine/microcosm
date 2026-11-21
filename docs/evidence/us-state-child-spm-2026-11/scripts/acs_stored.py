"""Stored dashboard audits (real Modal microsim runs, pe-us 2.30.1 and 1.825.2) against Table 18, and against the deck's default cut."""

import json
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
n = pd.read_csv(f"{SP}/run/default_noise.csv", index_col=0)


def load(f):
    d = pd.DataFrame(json.load(open(f"{SP}/dash/{f}.json"))).set_index("state")
    return d


F = {
    "o1017@2.30.1 2025": "state_audit_o1017_2301_y25_2025_raw",
    "buildp@1.825.2 2025": "state_audit_buildp_v2_y25_2025_raw",
    "o1017@1.825.2 2025": "state_audit_o1017_18252_2025_raw",
    "o1017@2.30.1 2024": "state_audit_o1017_2301_2024_raw",
    "buildp@1.825.2 2024": "state_audit_buildp_v2_2024_raw",
    "o1017@2.2.1 2024": "state_audit_o1017_221_2024_raw",
}
res = {}
for k, f in F.items():
    d = load(f)
    col = "poverty_child_measured" if "poverty_child_measured" in d else "poverty_child"
    print(
        f,
        {x: d[x].iloc[0] for x in ("pe_us", "slice_tag", "year")},
        "hh median",
        d.n_households.median(),
        "hh min",
        d.n_households.min(),
        d.status.value_counts().to_dict(),
    )
    r = d[col]
    g = r - c.rate
    nat = (r * d.pop_under_18).sum() / d.pop_under_18.sum()
    res[k] = r
    print(
        f"{k:24s} mean {g.mean():+.2f} MAD {g.abs().mean():.2f} sd {g.std():.2f} corr {np.corrcoef(r, c.rate)[0, 1]:.3f} beyond MOE {(g.abs() > c.moe).sum()} national {nat:.2f} corr(gap,defaultgap) {np.corrcoef(g, n.gap)[0, 1]:.2f}"
    )
R = pd.DataFrame(res)
R["census"] = c.rate
R["moe"] = c.moe
R["default"] = n.model
R["default_se"] = n.se
R["gap_acs"] = R["o1017@2.30.1 2025"] - R.census
R.to_csv(f"{SP}/run/acs_stored.csv")
print(
    R[
        [
            "census",
            "moe",
            "default",
            "default_se",
            "o1017@2.30.1 2025",
            "buildp@1.825.2 2025",
            "gap_acs",
        ]
    ]
    .sort_values("gap_acs")
    .round(1)
    .to_string()
)
# Noise-only check for ACS-local: what MAD would Census's sampling error alone produce?
rng = np.random.default_rng(1)
cse = c.moe / 1.645
print(
    "MAD from Census error alone",
    np.abs(rng.standard_normal((20000, 51)) * cse.values).mean(1).mean(),
    "with common shift of",
    R.gap_acs.mean(),
)
sh = R.gap_acs.mean()
print(
    "MAD Census error + shift",
    np.abs(sh + rng.standard_normal((20000, 51)) * cse.values).mean(1).mean(),
    "beyond",
    (np.abs(sh + rng.standard_normal((20000, 51)) * cse.values) > c.moe.values)
    .sum(1)
    .mean(),
)
