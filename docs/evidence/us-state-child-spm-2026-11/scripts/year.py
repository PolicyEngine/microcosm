"""Exact Shapley of the PE 2024 to PE 2025 step, national and by state, on final weights. Also the threshold growth."""

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "default"
u = pd.read_parquet(f"{SP}/run/{TAG}_units.parquet")
u = u[u.kids > 0].copy()
p = lambda n, y: u[f"{n}__{y}"].values.astype(float)
W = (u.w1 * u.kids).values
st = u.state.values


def parts(y):
    snap = p("snap", y)
    ctc = p("refundable_ctc", y)
    eitc = p("eitc", y)
    fed = p("spm_unit_federal_tax", y) + ctc + eitc  # federal before those two credits
    net = p("spm_unit_net_income", y)
    mkt = p("spm_unit_market_income", y)
    ben = p("spm_unit_benefits", y) - snap
    pay = p("spm_unit_payroll_tax", y) + p("spm_unit_self_employment_tax", y)
    sl = p("spm_unit_state_tax", y) + p("spm_unit_local_tax", y)
    exp = p("spm_unit_spm_expenses", y)
    rest = net - (mkt + ben + snap - fed + ctc + eitc - pay - sl - exp)
    return dict(
        market=mkt,
        benefits=ben + rest,
        snap=snap,
        fed_precredit=-fed,
        ctc=ctc,
        eitc=eitc,
        paystate=-pay - sl,
        exp=-exp,
        threshold=p("spm_unit_spm_threshold", y),
    )


a, b = parts(2024), parts(2025)
assert (
    np.abs(
        sum(v for k, v in a.items() if k != "threshold")
        - p("spm_unit_net_income", 2024)
    ).max()
    < 1
)
FACT = list(a)


def poor(bits):
    R = sum((b[k] if s else a[k]) for k, s in zip(FACT, bits) if k != "threshold")
    T = b["threshold"] if bits[FACT.index("threshold")] else a["threshold"]
    return R < T


states = sorted(set(st))
den = pd.Series(W).groupby(st).sum()


def rate(flag):
    return 100 * pd.Series(flag * W).groupby(st).sum() / den


vals = {}
nat = {}
for bits in itertools.product([0, 1], repeat=len(FACT)):
    f = poor(bits)
    vals[bits] = rate(f)
    nat[bits] = 100 * (f * W).sum() / W.sum()
n = len(FACT)
out = pd.DataFrame(index=states)
nout = {}
for i, fk in enumerate(FACT):
    tot = 0
    nt = 0
    for bits in itertools.product([0, 1], repeat=n - 1):
        s = sum(bits)
        w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
        A = bits[:i] + (1,) + bits[i:]
        B = bits[:i] + (0,) + bits[i:]
        tot = tot + w * (vals[A] - vals[B])
        nt = nt + w * (nat[A] - nat[B])
    out[fk] = tot
    nout[fk] = nt
print(
    "national Shapley",
    {k: round(v, 2) for k, v in nout.items()},
    "sum",
    round(sum(nout.values()), 2),
    "from",
    round(nat[(0,) * n], 2),
    "to",
    round(nat[(1,) * n], 2),
)
print(
    out.loc[["MD", "DC", "DE", "MO", "MI", "NC", "OK", "NV", "TX", "OR", "CA", "HI"]]
    .round(1)
    .to_string()
)
print("mean", out.mean().round(2).to_dict())
for tn, dv in (("threshold", None),):  # noqa: B007
    r = p("spm_unit_spm_threshold", 2025) / p("spm_unit_spm_threshold", 2024)
    print("threshold growth", pd.Series(r).describe().round(4).to_dict())
    print(
        "by tenure",
        pd.Series(r).groupby(u.SPM_TENMORTSTATUS.values).median().round(4).to_dict(),
    )
    print(
        "ref thresh growth",
        pd.Series(
            p("spm_unit_reference_spm_threshold", 2025)
            / p("spm_unit_reference_spm_threshold", 2024)
        )
        .groupby(u.SPM_TENMORTSTATUS.values)
        .median()
        .round(4)
        .to_dict(),
    )
    print(
        "geo adj change",
        pd.Series(
            p("spm_unit_geographic_adjustment", 2025)
            / p("spm_unit_geographic_adjustment", 2024)
        )
        .describe()
        .round(4)
        .to_dict(),
    )
g = lambda n, y: (u[f"{n}__{y}"] * W).sum() / W.sum()
for n in [
    "employment_income",
    "spm_unit_market_income",
    "spm_unit_benefits",
    "snap",
    "refundable_ctc",
    "eitc",
    "spm_unit_federal_tax",
    "spm_unit_net_income",
]:
    print(
        n, round(g(n, 2024)), round(g(n, 2025)), round(g(n, 2025) / g(n, 2024) - 1, 4)
    )
lost = (u["refundable_ctc__2024"] > 0) & (u["refundable_ctc__2025"] == 0)
print(
    "units with refundable CTC 2024 and none in 2025: child-weight share",
    round(W[lost].sum() / W.sum(), 4),
    "n",
    lost.sum(),
)
if "ctc__2025" in u:
    pass
