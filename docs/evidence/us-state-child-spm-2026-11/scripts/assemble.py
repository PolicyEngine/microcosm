"""Join each state's microsimulation output to Census's fields on the same units."""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
B = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
FIPS = {
    "AL": 1,
    "AK": 2,
    "AZ": 4,
    "AR": 5,
    "CA": 6,
    "CO": 8,
    "CT": 9,
    "DE": 10,
    "DC": 11,
    "FL": 12,
    "GA": 13,
    "HI": 15,
    "ID": 16,
    "IL": 17,
    "IN": 18,
    "IA": 19,
    "KS": 20,
    "KY": 21,
    "LA": 22,
    "ME": 23,
    "MD": 24,
    "MA": 25,
    "MI": 26,
    "MN": 27,
    "MS": 28,
    "MO": 29,
    "MT": 30,
    "NE": 31,
    "NV": 32,
    "NH": 33,
    "NJ": 34,
    "NM": 35,
    "NY": 36,
    "NC": 37,
    "ND": 38,
    "OH": 39,
    "OK": 40,
    "OR": 41,
    "PA": 42,
    "RI": 44,
    "SC": 45,
    "SD": 46,
    "TN": 47,
    "TX": 48,
    "UT": 49,
    "VT": 50,
    "VA": 51,
    "WA": 53,
    "WV": 54,
    "WI": 55,
    "WY": 56,
}
INV = {v: k for k, v in FIPS.items()}
CENSUS = """SPM_POOR SPM_RESOURCES SPM_POVTHRESHOLD SPM_TOTVAL SPM_SNAPSUB SPM_CAPHOUSESUB SPM_SCHLUNCH SPM_ENGVAL SPM_WICVAL SPM_BBSUBVAL
SPM_FICA SPM_FEDTAXBC SPM_EITC SPM_ACTC SPM_STTAX SPM_CAPWKCCXPNS SPM_MEDXPNS SPM_CHILDSUPPD SPM_GEOADJ SPM_EQUIVSCALE SPM_TENMORTSTATUS SPM_WEIGHT SPM_NUMPER""".split()


def people_table(h5, npz=None):
    with pd.HDFStore(h5, "r") as s:
        h = s.select("/household")
        keep = [
            "person_household_id",
            "person_spm_unit_id",
            "age",
            "source_year",
            "source_household_id",
        ] + [c for c in CENSUS]
        pp = s.select("/person")
        pp = pp[[c for c in keep if c in pp.columns]]
    if npz:
        z = np.load(npz, allow_pickle=True)
        assert np.allclose(h.household_weight.values, z["household_weight"])
        h["w0"] = z["initial_household_weight"]
    h["w1"] = h.household_weight
    h["state"] = h.state_fips.map(INV)
    return h, pp


def units(h, pp):
    """One row per SPM unit, with its Census fields from the first member."""
    pp = pp.merge(
        h[
            ["household_id", "state", "household_support_channel"]
            + (["w0"] if "w0" in h else [])
            + ["w1"]
        ],
        left_on="person_household_id",
        right_on="household_id",
    )
    g = pp.groupby("person_spm_unit_id", sort=False)
    u = g.first().reset_index().rename(columns={"person_spm_unit_id": "spm_unit_id"})
    u["kids"] = g.age.apply(lambda a: (a < 18).sum()).values
    u["people"] = g.size().values
    return u


if __name__ == "__main__":
    tag = sys.argv[1]
    h5 = sys.argv[2]
    npz = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    h, pp = people_table(h5, npz)
    u = units(h, pp)
    files = sorted(glob.glob(f"{SP}/run/{tag}/*.parquet"))
    pe = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(len(files), "states", len(pe), "units", len(u))
    m = u.merge(pe, on="spm_unit_id", how="inner", validate="one_to_one")
    assert (m.kids == m.n_children).all()
    m.to_parquet(f"{SP}/run/{tag}_units.parquet")
    # Reproduce the published run.
    base = pd.DataFrame(json.load(open(SP + "/state_baseline.json"))["rows"]).set_index(
        "state"
    )
    k = m[m.kids > 0]
    r = k.groupby("state").apply(
        lambda d: (
            100
            * np.average(d["spm_unit_is_in_spm_poverty__2025"], weights=d.w1 * d.kids)
        ),
        include_groups=False,
    )
    cmp = pd.DataFrame({"mine": r, "slides": base.child_spm_percent}).dropna()
    cmp["d"] = cmp.mine - cmp.slides
    print(
        "reproduction of state_baseline.json: max abs diff",
        cmp.d.abs().max(),
        "n",
        len(cmp),
    )
