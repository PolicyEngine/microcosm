"""Measure the income-year 2025 processed ASEC file against income year 2024.

usage: measure_pool.py OLD_2024.h5 NEW_2025.h5 COUNTY_LIST_CSV OUT.json

Households and persons, H_IDNUM rotation overlap (validated by reference-person
sex and age), county identification (GTCO != 0), and the coded counties against
the Census identified-county list for the matching survey year.
"""

import csv
import json
import sys

import numpy as np
import pandas as pd

old_p, new_p, county_csv, out_p = sys.argv[1:5]


def load(p):
    h = pd.read_hdf(p, "household")
    per = pd.read_hdf(p, "person")
    return h, per


res = {}
for label, path, survey in (("income_2024", old_p, 2025), ("income_2025", new_p, 2026)):
    h, per = load(path)
    w = h["HSUP_WGT"] / 100.0
    coded = h["GTCO"] != 0
    fips = (h.loc[coded, "GESTFIPS"] * 1000 + h.loc[coded, "GTCO"]).astype(int)
    listed = {
        int(r["state_fips"]) * 1000 + int(r["county_code"])
        for r in csv.DictReader(open(county_csv))
        if int(r["asec_year"]) == survey
    }
    coded_set = set(fips.unique().tolist())
    res[label] = dict(
        survey_year=survey,
        households=int(len(h)),
        persons=int(len(per)),
        weighted_households_millions=round(float(w.sum()) / 1e6, 3),
        county_identified_households=int(coded.sum()),
        county_identified_share=float(coded.mean()),
        county_identified_share_weighted=float(w[coded].sum() / w.sum()),
        distinct_coded_counties=len(coded_set),
        census_list4_counties=len(listed),
        coded_and_listed=len(coded_set & listed),
        coded_not_listed=len(coded_set - listed),
        listed_not_coded=len(listed - coded_set),
        coded_not_listed_fips=sorted(coded_set - listed),
        # PEINUSYR top codes, whose intervals Census re-bins every year.
        peinusyr_top_code_persons={
            int(code): int((per["PEINUSYR"] == code).sum()) for code in (27, 28, 29)
        },
    )

ho, po = load(old_p)
hn, pn = load(new_p)


def ref(h, p):
    """Households with their reference person's age and sex."""
    persons = p[p.A_EXPRRP.isin([1, 2])].drop_duplicates("PH_SEQ")
    return h[["H_SEQ", "H_IDNUM", "HSUP_WGT"]].merge(
        persons[["PH_SEQ", "A_AGE", "A_SEX"]],
        left_on="H_SEQ",
        right_on="PH_SEQ",
        how="left",
    )


o, n = ref(ho, po), ref(hn, pn)
in_old = n["H_IDNUM"].isin(set(o["H_IDNUM"]))
m = n.merge(o, on="H_IDNUM", suffixes=("_new", "_old"))
rng = np.random.default_rng(20260927)
a = n.iloc[rng.integers(0, len(n), 20000)].reset_index(drop=True)
b = o.iloc[rng.integers(0, len(o), 20000)].reset_index(drop=True)
d_rand = a.A_AGE.values - b.A_AGE.values
res["rotation_income_2025_vs_2024"] = dict(
    shared_H_IDNUM=int(in_old.sum()),
    share_of_2025_households=float(in_old.mean()),
    share_of_2025_households_weighted=float(
        n.HSUP_WGT[in_old].sum() / n.HSUP_WGT.sum()
    ),
    share_of_2024_households=float(o["H_IDNUM"].isin(set(n["H_IDNUM"])).mean()),
    linked_reference_person_sex_agrees=float((m.A_SEX_new == m.A_SEX_old).mean()),
    linked_reference_person_age_plus_0_to_2=float(
        (m.A_AGE_new - m.A_AGE_old).between(0, 2).mean()
    ),
    random_pairs_sex_agrees=float((a.A_SEX.values == b.A_SEX.values).mean()),
    random_pairs_age_plus_0_to_2=float(((d_rand >= 0) & (d_rand <= 2)).mean()),
    shared_PERIDNUM=int(len(set(pn.PERIDNUM) & set(po.PERIDNUM))),
    share_of_2025_persons=float(pn.PERIDNUM.isin(set(po.PERIDNUM)).mean()),
)
json.dump(res, open(out_p, "w"), indent=1)
print(json.dumps(res, indent=1))
