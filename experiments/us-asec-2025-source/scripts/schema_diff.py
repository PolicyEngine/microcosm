"""Schema, value-range and coverage diff between two processed CPS ASEC stores.

usage: schema_diff.py OLD.h5 NEW.h5 OUT.json
Reports, per table: row counts, added/dropped columns, dtype changes, and for every
shared column a value summary (min, max, distinct count, zero share) with flags for
range changes and code-set changes on low-cardinality columns. Also measures the
NOW_* coverage recodes, county identification (GTCO) and H_IDNUM rotation overlap.
"""

import json
import sys

import numpy as np
import pandas as pd

old_p, new_p, out_p = sys.argv[1:4]
LOW_CARD = 60


def summ(s):
    if s.dtype.kind in "iufb":
        v = s.to_numpy()
        return dict(
            kind="num",
            min=float(np.nanmin(v)),
            max=float(np.nanmax(v)),
            n_unique=int(s.nunique()),
            zero_share=float((v == 0).mean()),
            neg_share=float((v < 0).mean()),
        )
    return dict(
        kind="str", n_unique=int(s.nunique()), sample=[str(x) for x in s.head(3)]
    )


res = {}
with pd.HDFStore(old_p, "r") as so, pd.HDFStore(new_p, "r") as sn:
    tables = {
        k.strip("/"): (so[k], sn[k])
        for k in sorted(set(so.keys()) | set(sn.keys()))
        if k in so and k in sn
    }
    res["keys"] = dict(old=sorted(so.keys()), new=sorted(sn.keys()))
for name, (a, b) in tables.items():
    shared = [c for c in a.columns if c in b.columns]
    t = dict(
        rows=[len(a), len(b)],
        n_columns=[a.shape[1], b.shape[1]],
        dropped=sorted(set(a.columns) - set(b.columns)),
        added=sorted(set(b.columns) - set(a.columns)),
        index_name=[a.index.name, b.index.name],
        dtype_changes={
            c: [str(a[c].dtype), str(b[c].dtype)]
            for c in shared
            if str(a[c].dtype) != str(b[c].dtype)
        },
        column_order_same_for_shared=[c for c in b.columns if c in shared] == shared,
    )
    range_flags, code_flags = {}, {}
    for c in shared:
        sa, sb = summ(a[c]), summ(b[c])
        if sa["kind"] == "num" and sb["kind"] == "num":
            if sb["min"] < sa["min"] or sb["max"] > sa["max"]:
                range_flags[c] = dict(
                    old=[sa["min"], sa["max"]], new=[sb["min"], sb["max"]]
                )
            if max(sa["n_unique"], sb["n_unique"]) <= LOW_CARD:
                ca, cb = set(a[c].unique().tolist()), set(b[c].unique().tolist())
                if ca != cb:
                    code_flags[c] = dict(
                        new_codes=sorted(cb - ca), gone_codes=sorted(ca - cb)
                    )
    t["range_widened"] = range_flags
    t["code_set_changed"] = code_flags
    t["added_summaries"] = {c: summ(b[c]) for c in t["added"]}
    t["dropped_summaries_old"] = {c: summ(a[c]) for c in t["dropped"]}
    res[name] = t


def cov(path):
    with pd.HDFStore(path, "r") as s:
        p, h = s["person"], s["household"]
    w = p["A_FNLWGT"] / 100.0
    u65 = p["A_AGE"] < 65
    now = sorted(c for c in p.columns if c.startswith("NOW_"))
    d = {"NOW_columns": now}
    for c in now:
        v = p[c]
        d[c] = dict(
            codes=sorted(map(int, v.unique())),
            w_yes_u65_m=round(float(w[(v == 1) & u65].sum()) / 1e6, 2),
        )
    hw = h["HSUP_WGT"] / 100.0 if "HSUP_WGT" in h else None
    geo = {}
    for col in [
        "GTCO",
        "GTCBSA",
        "GTCSA",
        "GESTFIPS",
        "GTMETSTA",
        "GTINDVPC",
        "GTCOUNTY",
    ]:
        if col in h:
            geo[col] = dict(
                dtype=str(h[col].dtype),
                n_unique=int(h[col].nunique()),
                nonzero_share_hh=float((h[col] != 0).mean()),
                nonzero_share_weighted=float(hw[h[col] != 0].sum() / hw.sum())
                if hw is not None
                else None,
            )
    if "GTCO" in h:
        cty = h.loc[h["GTCO"] != 0, "GESTFIPS"] * 1000 + h.loc[h["GTCO"] != 0, "GTCO"]
        geo["distinct_identified_counties"] = int(cty.nunique())
        geo["identified_county_fips"] = sorted(map(int, cty.unique()))
    d["geo"] = geo
    d["households"] = len(h)
    d["persons"] = len(p)
    d["H_IDNUM"] = dict(
        present="H_IDNUM" in h,
        dtype=str(h["H_IDNUM"].dtype) if "H_IDNUM" in h else None,
        unique=int(h["H_IDNUM"].nunique()) if "H_IDNUM" in h else None,
    )
    return d, h


co, ho = cov(old_p)
cn, hn = cov(new_p)
res["coverage"] = dict(old=co, new=cn)
if "H_IDNUM" in ho and "H_IDNUM" in hn:
    ids_o, ids_n = set(ho["H_IDNUM"].astype(str)), set(hn["H_IDNUM"].astype(str))
    shared_ids = ids_o & ids_n
    hw_n = hn["HSUP_WGT"] / 100.0
    in_old = hn["H_IDNUM"].astype(str).isin(ids_o)
    res["rotation"] = dict(
        new_households=len(hn),
        old_households=len(ho),
        shared_H_IDNUM=len(shared_ids),
        share_of_new_in_old=float(in_old.mean()),
        share_of_new_in_old_weighted=float(hw_n[in_old].sum() / hw_n.sum()),
        share_of_old_in_new=float(ho["H_IDNUM"].astype(str).isin(ids_n).mean()),
    )
    if "H_MIS" in hn:
        res["rotation"]["by_new_H_MIS"] = {
            int(k): float(v) for k, v in in_old.groupby(hn["H_MIS"]).mean().items()
        }
    # same-address check on matched households: state must agree
    m = hn[in_old][["H_IDNUM", "GESTFIPS"]].merge(
        ho[["H_IDNUM", "GESTFIPS"]], on="H_IDNUM", suffixes=("_new", "_old")
    )
    res["rotation"]["matched_state_agreement"] = (
        float((m["GESTFIPS_new"] == m["GESTFIPS_old"]).mean()) if len(m) else None
    )
json.dump(res, open(out_p, "w"), indent=1, default=str)
print("wrote", out_p)
