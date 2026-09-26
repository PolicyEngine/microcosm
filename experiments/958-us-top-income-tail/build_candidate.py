"""PROTOTYPE candidate: certified populace_us_2024 plus an own-tail stratum of PUF donors.

NOT a Microcosm build. It grafts tail households onto the certified file so the effect of
(a) an own-tail stratum and (b) the SOI Table 1.1 size-of-AGI targets on the top-rate score
can be measured in minutes. Two variants share this harness:

  cg_only     main's rule today: donors above the weighted q99.5 of positive ST+LT gains;
              only the five capital-gains fields transfer (puf_capital_gains_tail.py).
  full_vector proposed rule: donors with proxy AGI >= --agi-floor; the whole tax-detail
              income and deduction vector transfers.

Each tail household is a clone of a high-AGI single-tax-unit certified household of the same
joint/non-joint class, carrying the donor's own weight.
"""

import argparse
import json
import warnings

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--certified", required=True)
ap.add_argument("--donor", required=True)
ap.add_argument(
    "--certified-agi",
    required=True,
    help="parquet: tax_unit_id, agi_2024, filing_status",
)
ap.add_argument("--variant", choices=["cg_only", "full_vector"], required=True)
ap.add_argument("--agi-floor", type=float, default=5e6)
ap.add_argument(
    "--max-donors",
    type=int,
    default=0,
    help="0 = all; else a weight-preserving systematic subsample",
)
ap.add_argument("--out", required=True)
ap.add_argument("--receipt", required=True)
a = ap.parse_args()

TABLES = ["household", "person", "tax_unit", "spm_unit", "family", "marital_unit"]
T = {t: pd.read_hdf(a.certified, t) for t in TABLES}
time_period = pd.read_hdf(a.certified, "_time_period")
pe = T["person"]

# ---------------- donor
f = h5py.File(a.donor, "r")


def arr(k):
    g = f[k]
    return g[list(g.keys())[0]][()] if isinstance(g, h5py.Group) else g[()]


d_ptu = arr("person_tax_unit_id")
d_tuid = arr("tax_unit_id")
# donor person variable -> certified person column
PERSON_MAP = {
    "employment_income": "employment_income_before_lsr",
    "self_employment_income": "self_employment_income_before_lsr",
    "taxable_interest_income": "taxable_interest_income",
    "tax_exempt_interest_income": "tax_exempt_interest_income",
    "qualified_dividend_income": "qualified_dividend_income",
    "non_qualified_dividend_income": "non_qualified_dividend_income",
    "long_term_capital_gains": "long_term_capital_gains_before_response",
    "short_term_capital_gains": "short_term_capital_gains",
    "long_term_capital_gains_on_collectibles": "long_term_capital_gains_on_collectibles",
    "non_sch_d_capital_gains": "non_sch_d_capital_gains",
    "taxable_pension_income": "taxable_private_pension_income",
    "taxable_ira_distributions": "taxable_ira_distributions",
    "partnership_s_corp_income": "partnership_income",
    "rental_income": "rental_income",
    "farm_income": "farm_income",
    "farm_operations_income": "farm_operations_income",
    "farm_rent_income": "farm_rent_income",
    "estate_income": "estate_income",
    "miscellaneous_income": "miscellaneous_income",
    "alimony_income": "alimony_income",
    "salt_refund_income": "salt_refund_income",
    "charitable_cash_donations": "charitable_cash_donations",
    "charitable_non_cash_donations": "charitable_non_cash_donations",
    "real_estate_taxes": "real_estate_taxes",
    "home_mortgage_interest": "home_mortgage_interest",
    "investment_interest_expense": "investment_interest_expense",
    "investment_income_elected_form_4952": "investment_income_elected_form_4952",
    "w2_wages_from_qualified_business": "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property": "unadjusted_basis_qualified_property",
    "qualified_reit_and_ptp_income": "qualified_reit_and_ptp_income",
    "qualified_bdc_income": "qualified_bdc_income",
}
ZERO_ON_TRANSFER = [
    "s_corp_income",
    "partnership_self_employment_net_earnings",
    "sstb_self_employment_income_before_lsr",
    "taxable_401k_distributions",
    "taxable_403b_distributions",
    "taxable_sep_distributions",
    "keogh_distributions",
]
CG_PERSON = {
    k: PERSON_MAP[k]
    for k in [
        "long_term_capital_gains",
        "short_term_capital_gains",
        "long_term_capital_gains_on_collectibles",
        "non_sch_d_capital_gains",
    ]
}
AGI_PROXY = [
    "employment_income",
    "self_employment_income",
    "taxable_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "long_term_capital_gains",
    "short_term_capital_gains",
    "non_sch_d_capital_gains",
    "taxable_pension_income",
    "taxable_ira_distributions",
    "partnership_s_corp_income",
    "rental_income",
    "farm_income",
    "miscellaneous_income",
    "estate_income",
    "alimony_income",
    "taxable_unemployment_compensation",
    "farm_rent_income",
    "salt_refund_income",
]
need = sorted(set(PERSON_MAP) | set(AGI_PROXY))
dp = pd.DataFrame({k: arr(k).astype(float) for k in need})
dp["tu"] = d_ptu
dtu = dp.groupby("tu").sum().reindex(d_tuid)
dtu["weight"] = arr("household_weight").astype(float)
dtu["filing_status"] = [x.decode() for x in arr("filing_status")]
dtu["unrecaptured_section_1250_gain"] = arr("unrecaptured_section_1250_gain").astype(
    float
)
dtu["proxy_agi"] = dtu[AGI_PROXY].sum(axis=1)
dtu = dtu[dtu.weight > 0]


def wq(v, w, q):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) / w.sum()
    return v[np.searchsorted(c, q)]


if a.variant == "cg_only":
    comb = (dtu.long_term_capital_gains + dtu.short_term_capital_gains).to_numpy()
    pos = comb > 0
    boundary = wq(comb[pos], dtu.weight.to_numpy()[pos], 0.995)
    tail = dtu[comb > boundary].copy()
    rule = {
        "rule": "st_plus_lt_gains_above_weighted_q99.5",
        "boundary": float(boundary),
    }
else:
    tail = dtu[dtu.proxy_agi >= a.agi_floor].copy()
    rule = {"rule": "proxy_agi_at_or_above_floor", "floor": a.agi_floor}
tail = tail.sort_index()
if a.max_donors and len(tail) > a.max_donors:
    # systematic PPS-free subsample within proxy-AGI order; weights scaled so weighted returns AND weighted proxy AGI are preserved per decile
    tail = tail.sort_values("proxy_agi")
    tail["g"] = pd.qcut(np.arange(len(tail)), 10, labels=False)
    parts = []
    for _, grp in tail.groupby("g"):
        step = max(1, int(round(len(grp) / (a.max_donors / 10))))
        sub = grp.iloc[step // 2 :: step].copy()
        sub["weight"] *= grp.weight.sum() / sub.weight.sum()
        parts.append(sub)
    tail = pd.concat(parts).drop(columns="g").sort_index()
tail = tail.reset_index().rename(columns={"tu": "donor_tax_unit_id"})

# ---------------- recipients
cagi = pd.read_parquet(a.certified_agi).set_index("tax_unit_id")
tu_per_hh = pe.groupby("person_household_id").person_tax_unit_id.nunique()
one_tu = set(tu_per_hh[tu_per_hh == 1].index)
p1 = pe.drop_duplicates("person_tax_unit_id")[
    ["person_tax_unit_id", "person_household_id"]
].set_index("person_tax_unit_id")
cand = cagi.join(p1, how="inner")
cand = cand[cand.person_household_id.isin(one_tu)]
cand["joint"] = cand.filing_status.eq("JOINT")
pools = {
    j: cand[cand.joint == j]
    .sort_values("agi_2024", ascending=False)
    .head(n)
    .person_household_id.to_numpy()
    for j, n in ((True, 600), (False, 250))
}
tail["joint"] = tail.filing_status.eq("JOINT")
rec = np.empty(len(tail), dtype=np.int64)
for j in (True, False):
    idx = np.flatnonzero(tail.joint.to_numpy() == j)
    rec[idx] = pools[j][np.arange(len(idx)) % len(pools[j])]
tail["recipient_household_id"] = rec

# ---------------- clone
n = len(tail)
clone_no = np.arange(n)
BIG = int(
    max(T[t][f"{t}_id"].max() for t in TABLES if t != "person")
    + pe.person_id.max()
    + 10
)


def offset(i):
    return (i + 1) * 0 + 0


hh_index = T["household"].set_index("household_id")
persons_by_hh = pe.groupby("person_household_id").indices
out = {t: [T[t]] for t in TABLES}
ID_COLS = {
    "household": "person_household_id",
    "tax_unit": "person_tax_unit_id",
    "spm_unit": "person_spm_unit_id",
    "family": "person_family_id",
    "marital_unit": "person_marital_unit_id",
}
next_id = {t: int(T[t][f"{t}_id"].max()) + 1 for t in TABLES if t != "person"}
next_pid = int(pe.person_id.max()) + 1
ent_index = {t: T[t].set_index(f"{t}_id") for t in TABLES if t != "person"}
new_rows = {t: [] for t in TABLES}
transfer_cols = (
    list(CG_PERSON.values())
    if a.variant == "cg_only"
    else list(PERSON_MAP.values()) + ZERO_ON_TRANSFER
)
src_map = CG_PERSON if a.variant == "cg_only" else PERSON_MAP
for _i, row in enumerate(tail.itertuples(index=False)):
    ppos = persons_by_hh[row.recipient_household_id]
    p = pe.iloc[ppos].copy()
    remap = {}
    for t, col in ID_COLS.items():
        olds = p[col].unique()
        news = np.arange(next_id[t], next_id[t] + len(olds))
        next_id[t] += len(olds)
        m = dict(zip(olds, news, strict=False))
        remap[t] = m
        p[col] = p[col].map(m)
        e = ent_index[t].loc[olds].reset_index()
        e[f"{t}_id"] = e[f"{t}_id"].map(m)
        ci = f"{t}_support_clone_index"
        if ci in e.columns:
            e[ci] = 2
        if t == "household":
            e["household_weight"] = row.weight
        if t == "tax_unit" and "unrecaptured_section_1250_gain" in e.columns:
            e["unrecaptured_section_1250_gain"] = row.unrecaptured_section_1250_gain
        new_rows[t].append(e)
    p["person_id"] = np.arange(next_pid, next_pid + len(p))
    next_pid += len(p)
    if "person_support_clone_index" in p.columns:
        p["person_support_clone_index"] = 2
    for c in transfer_cols:
        if c in p.columns:
            p[c] = 0.0
    head = p.index[0]
    ages = p["age"].to_numpy()
    head = p.index[int(np.argmax(ages))]
    for src, dst in src_map.items():
        if dst in p.columns:
            p.at[head, dst] = float(getattr(row, src))
    new_rows["person"].append(p)
for t in TABLES:
    add = pd.concat(new_rows[t], ignore_index=True)
    for c in T[t].columns:
        add[c] = add[c].astype(T[t][c].dtype)
    out[t] = pd.concat([T[t], add[T[t].columns]], ignore_index=True)

for t in TABLES:
    out[t].to_hdf(a.out, key=t, mode="a" if t != TABLES[0] else "w", format="table")
time_period.to_hdf(a.out, key="_time_period", mode="a", format="table")
receipt = {
    "variant": a.variant,
    **rule,
    "tail_donors": int(n),
    "tail_weighted_returns": float(tail.weight.sum()),
    "tail_weighted_proxy_agi_bn": float((tail.weight * tail.proxy_agi).sum() / 1e9),
    "tail_joint_share": float(tail.joint.mean()),
    "recipient_pool_sizes": {str(k): int(len(v)) for k, v in pools.items()},
    "households_out": int(len(out["household"])),
    "persons_out": int(len(out["person"])),
}
json.dump(receipt, open(a.receipt, "w"), indent=1)
print(json.dumps(receipt, indent=1))
