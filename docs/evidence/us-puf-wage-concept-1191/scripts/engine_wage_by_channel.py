"""Engine wage concepts by support channel on a US release H5 (microcosm#1191 audit).

Usage: python engine_wage_by_channel.py <populace_us_2024.h5> <out.json>

Runs the locked policyengine-us on the file the way the release's own
reform-validation factory does (USSingleYearDataset, county SPM selection) and
sums person-level wage concepts by person_support_channel at the file's own
weights. It also compares, person by person, the engine's
``pre_tax_contributions`` and ``irs_employment_income`` with the same amounts
rebuilt from stored columns by ``gross_up_reference.engine_pre_tax``. Nothing
is recalibrated.
"""

import hashlib
import importlib.metadata as md
import json
import os
import sys

import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from policyengine_us.data import USSingleYearDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gross_up_reference import engine_pre_tax  # noqa: E402

H5, OUT = sys.argv[1], sys.argv[2]
PERIOD = 2024
VARIABLES = [
    "employment_income",
    "employment_income_before_lsr",
    "irs_employment_income",
    "pre_tax_contributions",
    "traditional_401k_contributions",
    "traditional_403b_contributions",
    "pre_tax_health_insurance_premiums",
    "health_savings_account_payroll_contributions",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


sim = Microsimulation(
    dataset=USSingleYearDataset(file_path=H5), spm={"geography_kind": "county"}
)
weight = np.asarray(sim.calculate("person_weight", PERIOD), dtype=np.float64)
person_id = np.asarray(sim.calculate("person_id", PERIOD))
values = {v: np.asarray(sim.calculate(v, PERIOD), dtype=np.float64) for v in VARIABLES}

with pd.HDFStore(H5, mode="r") as store:
    stored = store.select(
        "person",
        columns=[
            "person_id",
            "person_support_channel",
            "person_support_clone_index",
            "age",
            "employment_income_before_lsr",
            "traditional_401k_contributions_desired",
            "roth_401k_contributions_desired",
        ],
    )
stored = stored.set_index("person_id").reindex(person_id)
channel = stored["person_support_channel"].astype(str).to_numpy()

out = {
    "h5": os.path.basename(H5),
    "h5_sha256": sha256(H5),
    "period": PERIOD,
    "policyengine_us": md.version("policyengine-us"),
    "policyengine_core": md.version("policyengine-core"),
    "persons_count": int(len(weight)),
    "persons_without_channel_count": int(stored["person_support_channel"].isna().sum()),
    "units": "USD weighted sums",
    "all": {v: float((weight * values[v]).sum()) for v in VARIABLES},
    "by_channel": {},
}
gross, taxed, pre_tax = (
    values["employment_income"],
    values["irs_employment_income"],
    values["pre_tax_contributions"],
)
for name in np.unique(channel):
    mask = channel == name
    entry = {v: float((weight * values[v])[mask].sum()) for v in VARIABLES}
    entry["weighted_persons_count"] = float(weight[mask].sum())
    entry["gross_minus_taxed"] = float((weight * (gross - taxed))[mask].sum())
    entry["lost_to_zero_floor"] = float(
        (weight * np.maximum(pre_tax - gross, 0.0))[mask].sum()
    )
    entry["persons_with_pre_tax_contributions_count"] = float(
        weight[mask & (pre_tax > 0)].sum()
    )
    out["by_channel"][name] = entry

# Per-person differential: stored columns through the reference formula
# against the engine. The engine holds float variables as float32.
desired = stored["traditional_401k_contributions_desired"].to_numpy(np.float64)
roth = stored["roth_401k_contributions_desired"].to_numpy(np.float64)
age = stored["age"].to_numpy(np.float64)
wage = stored["employment_income_before_lsr"].to_numpy(np.float64)
rebuilt_pre_tax = engine_pre_tax(desired, roth, age)
rebuilt_taxed = np.maximum(wage - rebuilt_pre_tax, 0.0)
differential = {}
for name, engine_values, rebuilt in (
    ("pre_tax_contributions", pre_tax, rebuilt_pre_tax),
    ("irs_employment_income", taxed, rebuilt_taxed),
):
    gap = np.abs(engine_values - rebuilt)
    # Two float32 units in the last place of the larger operand, plus a cent.
    allowed = 2 * np.spacing(np.float32(np.maximum(wage, rebuilt_pre_tax))) + 0.01
    differential[name] = {
        "max_abs_difference": float(gap.max()),
        "persons_over_float32_bound_count": int((gap > allowed).sum()),
        "persons_count": int(gap.size),
    }
out["per_person_differential_vs_stored_columns"] = differential

with open(OUT, "w") as handle:
    json.dump(out, handle, indent=1)
print(json.dumps(out, indent=1))
