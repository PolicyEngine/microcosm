"""PUF-channel membership on the archived generator's rounding lattice (microcosm#1191 audit).

Usage: python puf_grid_check.py <frame.h5> <soi_targets.csv> <out.json>

Tests whether stored tax-unit totals lie on whole-dollar lattices scaled by
the archived generator's uprating factors, at a relative tolerance of 1e-11. A
continuous-uniform random control is reported. Membership does not establish
donor identity or whether a deferral was added.

The scalar tested is the archived generator's own: policyengine-us-data at
42ed5d45, datasets/puf/uprate_puf.py::get_growth(variable, 2015, year) read
from its storage/calibration_targets/soi_targets.csv, i.e. growth of the SOI
aggregate divided by growth of the return count. 2022 and 2023 scalars and a
uniform random control are reported beside 2021.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

H5, SOI, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
COLUMNS = {
    "employment_income_before_lsr": "employment_income",
    "qualified_dividend_income": "qualified_dividends",
    "taxable_private_pension_income": "taxable_pension_income",
    "taxable_ira_distributions": "ira_distributions",
}
TOLERANCE = 1e-11


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


soi = pd.read_csv(SOI)
soi["Value"] = soi["Value"].astype(float)


def soi_aggregate(variable, year, is_count):
    rows = (
        (soi.Variable == variable)
        & (soi.Year == year)
        & (soi["Filing status"] == "All")
        & (soi["AGI lower bound"] == -np.inf)
        & (soi["AGI upper bound"] == np.inf)
        & (soi["Count"] == is_count)
        & (~soi["Taxable only"])
    )
    return soi[rows].iloc[0].Value


def get_growth(variable, from_year, to_year):
    aggregate = soi_aggregate(variable, to_year, False) / soi_aggregate(
        variable, from_year, False
    )
    returns = soi_aggregate("count", to_year, True) / soi_aggregate(
        "count", from_year, True
    )
    return aggregate / returns


def on_grid(values, step):
    multiple = values / step
    nearest = np.round(multiple)
    return np.abs(multiple - nearest) <= TOLERANCE * np.maximum(nearest, 1.0)


with pd.HDFStore(H5, mode="r") as store:
    person = store.select(
        "person",
        columns=[
            "person_tax_unit_id",
            "person_support_channel",
            "person_support_clone_index",
            *COLUMNS,
        ],
    )
channel = person["person_support_channel"].astype(str)
clone = person["person_support_clone_index"].astype(int)
out = {
    "h5": os.path.basename(H5),
    "h5_sha256": sha256(H5),
    "soi_targets_sha256": sha256(SOI),
    "relative_tolerance": TOLERANCE,
}
rng = np.random.default_rng(0)

for column, soi_variable in COLUMNS.items():
    scalars = {
        year: float(get_growth(soi_variable, 2015, year)) for year in (2021, 2022, 2023)
    }
    entry = {
        "soi_variable": soi_variable,
        "get_growth_from_2015": scalars,
        "groups": {},
    }
    for label, mask in {
        "puf_tax_detail/clone1": (channel == "puf_tax_detail") & (clone == 1),
        "asec/clone0": (channel == "asec") & (clone == 0),
    }.items():
        totals = (
            person[mask]
            .groupby("person_tax_unit_id")[column]
            .sum()
            .to_numpy(dtype=np.float64)
        )
        totals = totals[totals > 0]
        if not len(totals):
            entry["groups"][label] = {"tax_units_with_positive_total_count": 0}
            continue
        group = {"tax_units_with_positive_total_count": int(len(totals))}
        for year, scalar in scalars.items():
            group[f"share_whole_source_dollars_at_{year}_scalar"] = float(
                on_grid(totals, scalar).mean()
            )
        control = rng.uniform(
            np.percentile(totals, 5), np.percentile(totals, 95), 200_000
        )
        group["uniform_random_control_share_at_2021_scalar"] = float(
            on_grid(control, scalars[2021]).mean()
        )
        hit = on_grid(totals, scalars[2021])
        source = totals / scalars[2021]
        group["on_grid_distinct_values_count"] = int(
            np.unique(np.round(source[hit])).size
        )
        group["off_grid_count"] = int((~hit).sum())
        group["off_grid_distinct_values_count"] = int(np.unique(totals[~hit]).size)
        bands = {}
        for name, low, high in (
            ("under_5", 0.0, 5.0),
            ("5_to_10k", 5.0, 10_000.0),
            ("10k_to_100k", 10_000.0, 100_000.0),
            ("100k_and_over", 100_000.0, np.inf),
        ):
            rows = hit & (source >= low - 1e-6) & (source < high - 1e-6)
            if not rows.any():
                bands[name] = {"on_grid_count": 0}
                continue
            value = source[rows]
            bands[name] = {
                "on_grid_count": int(rows.sum()),
                "distinct_values_if_under_5": sorted(
                    {float(v) for v in np.round(value, 6)}
                )
                if high <= 5
                else None,
                "share_multiple_of_10": float(on_grid(value, 10.0).mean()),
                "share_multiple_of_100": float(on_grid(value, 100.0).mean()),
                "share_multiple_of_1000": float(on_grid(value, 1000.0).mean()),
            }
        group["implied_source_amount_bands_at_2021_scalar"] = bands
        entry["groups"][label] = group
    out[column] = entry

with open(OUT, "w") as handle:
    json.dump(out, handle, indent=1)
print(json.dumps(out, indent=1))
