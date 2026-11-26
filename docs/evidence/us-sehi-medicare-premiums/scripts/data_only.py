"""Stored-data tabulations (no engine): who the Medicare proxy withholds, and what they report."""

import json
import sys

import numpy as np
import pandas as pd

h5, out_path = sys.argv[1], sys.argv[2]
p = pd.read_hdf(h5, "person")
h = pd.read_hdf(h5, "household")
w = (
    p["person_household_id"]
    .map(h.set_index("household_id")["household_weight"])
    .to_numpy(float)
)
assert np.isfinite(w).all()
f = lambda c: p[c].to_numpy(float)
rep = f("health_insurance_premiums_without_medicare_part_b")
pem = f("medicare_part_b_premiums")
sc = f("self_employment_income_before_lsr") + f(
    "sstb_self_employment_income_before_lsr"
)
farm, part = f("farm_operations_income"), f("partnership_self_employment_net_earnings")
age, ssdi = f("age"), f("social_security_disability")
esi = p["has_esi"].to_numpy(bool)
takes = p["takes_up_medicare_if_eligible"].to_numpy(bool)
shipped = f("health_insurance_premiums")
channel = (
    p["person_support_channel"].astype(str).to_numpy()
    if "person_support_channel" in p
    else None
)
ss_cols = [
    c
    for c in (
        "social_security_retirement",
        "social_security_disability",
        "social_security_survivors",
        "social_security_dependents",
    )
    if c in p
]
ss = sum(f(c) for c in ss_cols)
proxy = (age >= 65) | (ssdi > 0)
aged = age >= 65
biz = {"schedule_c": sc > 0, "farm": farm > 0, "partnership": part > 0}
widened = biz["schedule_c"] | biz["farm"] | biz["partnership"]


def m(mask, value=None):
    row = {"people": round(float(w[mask].sum())), "rows": int(mask.sum())}
    if value is not None:
        row["amount"] = round(float((value * w)[mask].sum()))
    return row


out = {
    "dataset": h5,
    "rows": len(p),
    "people": round(float(w.sum())),
    "social_security_columns": ss_cols,
}
assert np.allclose(
    np.where(biz["schedule_c"] & ~proxy & ~esi, rep, 0.0), shipped, rtol=0, atol=1e-9
)
out["shipped_attribution_is_schedule_c_rule"] = True
for name, flag in (("schedule_c_rule", biz["schedule_c"]), ("widened_rule", widened)):
    g = flag & proxy
    o = g & ~esi
    out[name] = {
        "self_employed": m(flag),
        "in_medicare_proxy": m(g),
        "in_proxy_age_65_plus": m(flag & aged),
        "in_proxy_under_65_with_ssdi": m(flag & ~aged & (ssdi > 0)),
        "in_proxy_with_measured_esi": m(g & esi),
        "in_proxy_outside_esi": m(o),
        "in_proxy_outside_esi_takes_up_medicare": m(o & takes),
        "in_proxy_outside_esi_reported_premium_positive": m(o & (rep > 0), rep),
        "in_proxy_outside_esi_reported_premium_zero": m(o & ~(rep > 0)),
        "in_proxy_outside_esi_stored_medicare_premium_positive": m(o & (pem > 0), pem),
        "in_proxy_outside_esi_stored_medicare_premium_zero": m(o & ~(pem > 0)),
        "in_proxy_outside_esi_no_social_security": m(o & ~(ss > 0)),
        "in_proxy_outside_esi_no_social_security_reported_positive": m(
            o & ~(ss > 0) & (rep > 0), rep
        ),
        "in_proxy_outside_esi_no_social_security_stored_medicare_positive": m(
            o & ~(ss > 0) & (pem > 0), pem
        ),
        "in_proxy_outside_esi_by_business": {
            "schedule_c": m(o & biz["schedule_c"], rep),
            "farm_without_schedule_c_profit": m(
                o & ~biz["schedule_c"] & biz["farm"], rep
            ),
            "partnership_without_schedule_c_or_farm_profit": m(
                o & ~biz["schedule_c"] & ~biz["farm"] & biz["partnership"], rep
            ),
        },
        "attributed_now": m(flag & ~proxy & ~esi & (rep > 0), rep),
    }
    if channel is not None:
        out[name]["in_proxy_outside_esi_by_channel"] = {
            c: {
                "reported": m(o & (channel == c) & (rep > 0), rep),
                "stored_medicare": m(o & (channel == c) & (pem > 0), pem),
                "all": m(o & (channel == c)),
            }
            for c in sorted(set(channel))
        }
out["stored_medicare_premium"] = {
    "all": m(pem > 0, pem),
    "by_channel": None
    if channel is None
    else {c: m((pem > 0) & (channel == c), pem) for c in sorted(set(channel))},
    "takes_up_medicare": m(takes & proxy),
    "age_65_plus": m(aged),
    "positive_among_age_65_plus": m(aged & (pem > 0), pem),
    "positive_outside_proxy": m(~proxy & (pem > 0), pem),
    "quantiles_positive": [
        float(q) for q in np.quantile(pem[pem > 0], [0.1, 0.25, 0.5, 0.75, 0.9])
    ],
}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(out, indent=1))
