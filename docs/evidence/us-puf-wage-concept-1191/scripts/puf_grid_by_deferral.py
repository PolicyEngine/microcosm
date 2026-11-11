"""Lattice membership of PUF-clone tax-unit wage totals, split by whether the
unit holds any stored traditional 401(k) deferral (microcosm#1191 audit).

Usage: python puf_grid_by_deferral.py <frame.h5> <scalar> <out.json>

Reports lattice membership for stored wages and for wages plus or minus stored
desired deferrals. Membership does not establish donor identity or whether a
deferral was added.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

H5, SCALAR, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with pd.HDFStore(H5, mode="r") as store:
    person = store.select(
        "person",
        columns=[
            "person_tax_unit_id",
            "person_support_channel",
            "person_support_clone_index",
            "employment_income_before_lsr",
            "traditional_401k_contributions_desired",
        ],
    )
mask = (person["person_support_channel"].astype(str) == "puf_tax_detail") & (
    person["person_support_clone_index"].astype(int) == 1
)
units = (
    person[mask]
    .groupby("person_tax_unit_id")[
        ["employment_income_before_lsr", "traditional_401k_contributions_desired"]
    ]
    .sum()
)
units = units[units["employment_income_before_lsr"] > 0]
wage = units["employment_income_before_lsr"].to_numpy()
deferral = units["traditional_401k_contributions_desired"].to_numpy()


def on_grid(values):
    multiple = values / SCALAR
    nearest = np.round(multiple)
    return np.abs(multiple - nearest) <= 1e-11 * np.maximum(np.abs(nearest), 1.0)


digest = hashlib.sha256()
with open(H5, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
has = deferral > 0
out = {
    "h5": os.path.basename(H5),
    "h5_sha256": digest.hexdigest(),
    "scalar": SCALAR,
    "units_with_wages_count": int(len(wage)),
    "units_with_a_deferral_count": int(has.sum()),
    "on_grid_share_units_with_a_deferral": float(on_grid(wage[has]).mean()),
    "on_grid_share_units_without_a_deferral": float(on_grid(wage[~has]).mean()),
    "on_grid_share_of_wage_plus_deferral_units_with_a_deferral": float(
        on_grid((wage + deferral)[has]).mean()
    ),
    "on_grid_share_of_wage_minus_deferral_units_with_a_deferral": float(
        on_grid((wage - deferral)[has]).mean()
    ),
}
with open(OUT, "w") as handle:
    json.dump(out, handle, indent=1)
print(json.dumps(out, indent=1))
