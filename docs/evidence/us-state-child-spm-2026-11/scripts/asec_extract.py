import os
import zipfile

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/PolicyEngine/_buildm-runtime/inputs/asec_education")
SPMC = [
    "SPM_ACTC",
    "SPM_BBSUBVAL",
    "SPM_CAPHOUSESUB",
    "SPM_CAPWKCCXPNS",
    "SPM_CHILDCAREXPNS",
    "SPM_CHILDSUPPD",
    "SPM_EITC",
    "SPM_ENGVAL",
    "SPM_EQUIVSCALE",
    "SPM_FAMTYPE",
    "SPM_FEDTAX",
    "SPM_FEDTAXBC",
    "SPM_FICA",
    "SPM_GEOADJ",
    "SPM_ID",
    "SPM_MEDXPNS",
    "SPM_NUMADULTS",
    "SPM_NUMKIDS",
    "SPM_NUMPER",
    "SPM_POOR",
    "SPM_POVTHRESHOLD",
    "SPM_RESOURCES",
    "SPM_SCHLUNCH",
    "SPM_SNAPSUB",
    "SPM_STTAX",
    "SPM_TENMORTSTATUS",
    "SPM_TOTVAL",
    "SPM_WEIGHT",
    "SPM_WICVAL",
    "SPM_WKXPNS",
]
out = []
for sy in (23, 24, 25, 26):
    z = zipfile.ZipFile(f"{ROOT}/asecpub{sy}csv.zip")
    names = z.namelist()
    pn = [n for n in names if n.lower().endswith(f"pppub{sy}.csv")][0]
    hn = [n for n in names if n.lower().endswith(f"hhpub{sy}.csv")][0]
    with z.open(pn) as f:
        hdr = pd.read_csv(f, nrows=0).columns.tolist()
    cols = [
        c
        for c in hdr
        if c
        in set(
            ["PERIDNUM", "PH_SEQ", "A_AGE", "A_FNLWGT", "MARSUPWT", "A_LINENO", "P_SEQ"]
            + SPMC
        )
        or c in ("SPM_Poor",)
    ]
    # case-insensitive map
    u = {c.upper(): c for c in hdr}
    want = [
        "PERIDNUM",
        "PH_SEQ",
        "A_AGE",
        "A_FNLWGT",
        "MARSUPWT",
        "P_SEQ",
        "A_LINENO",
    ] + SPMC
    cols = [u[c] for c in want if c in u]
    miss = [c for c in want if c not in u]
    print(sy, "missing", miss)
    with z.open(pn) as f:
        p = pd.read_csv(f, usecols=cols, dtype={u["PERIDNUM"]: str})
        p.columns = [c.upper() for c in p.columns]
    with z.open(hn) as f:
        h0 = pd.read_csv(f, nrows=0).columns.tolist()
    uh = {c.upper(): c for c in h0}
    with z.open(hn) as f:
        h = pd.read_csv(f, usecols=[uh["H_SEQ"], uh["GESTFIPS"], uh["HSUP_WGT"]])
        h.columns = (
            ["H_SEQ", "GESTFIPS", "HSUP_WGT"]
            if list(h.columns)[0].upper() == "H_SEQ"
            else [c.upper() for c in h.columns]
        )
    h.columns = [c.upper() for c in h.columns]
    p = p.merge(h, left_on="PH_SEQ", right_on="H_SEQ", how="left")
    p["income_year"] = 2000 + sy - 1
    print(sy, p.shape, p.GESTFIPS.isna().sum())
    ch = p[p.A_AGE < 18]
    print(
        "  child SPM rate MARSUPWT",
        np.average(ch.SPM_POOR, weights=ch.MARSUPWT),
        " A_FNLWGT",
        np.average(ch.SPM_POOR, weights=ch.A_FNLWGT),
        " SPM_WEIGHT",
        np.average(ch.SPM_POOR, weights=ch.SPM_WEIGHT),
    )
    print("  all SPM rate MARSUPWT", np.average(p.SPM_POOR, weights=p.MARSUPWT))
    out.append(p)
a = pd.concat(out, ignore_index=True)
a.to_parquet(SP + "/asec/raw_asec_2022_2025.parquet")
print(a.shape)
