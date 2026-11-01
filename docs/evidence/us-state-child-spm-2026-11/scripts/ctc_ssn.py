"""Do units that lose the refundable child tax credit between 2024 and 2025 lack an adult with a valid SSN card?
pe-us: gov.irs.credits.ctc.adult_ssn_requirement_applies turns true on 2025-01-01; eligible cards are CITIZEN and NON_CITIZEN_VALID_EAD."""

import glob
import os

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
f = glob.glob(S + "9a814a3b*/populace_us_2024.h5")[0]
p = pd.read_hdf(f, "person")
print([c for c in p.columns if "ssn" in c.lower() or "immigration" in c.lower()][:8])
col = [c for c in p.columns if c.startswith("ssn_card_type")][0]
print(p[col].astype(str).value_counts(normalize=True).round(4).to_dict())
p["valid"] = (
    p[col].astype(str).isin(["CITIZEN", "NON_CITIZEN_VALID_EAD"])
    if p[col].dtype == object or str(p[col].dtype) in ("str", "string", "category")
    else False
)
agecol = [c for c in p.columns if c == "age" or c.startswith("age__")][0]
a = p[p[agecol] >= 18]
unitcol = [c for c in p.columns if c.startswith("person_spm_unit_id")][0]
ok = a.groupby(unitcol).valid.any()
pk = p[p[agecol] < 18].groupby(unitcol).size()
u = pd.read_parquet(
    f"{SP}/run/default_units.parquet",
    columns=[
        "spm_unit_id",
        "kids",
        "w1",
        "state",
        "refundable_ctc__2024",
        "refundable_ctc__2025",
        "spm_unit_is_in_spm_poverty__2024",
        "spm_unit_is_in_spm_poverty__2025",
        "household_support_channel",
    ],
)
u = u[u.kids > 0].copy()
u["W"] = u.w1 * u.kids
u["adult_ok"] = u.spm_unit_id.map(ok).fillna(False).astype(bool)
lost = (u.refundable_ctc__2024 > 0) & (u.refundable_ctc__2025 <= 0)
print(
    "lost share of child weight",
    round(u.W[lost].sum() / u.W.sum() * 100, 2),
    "units",
    int(lost.sum()),
)
print(
    "lost units with no adult holding a valid card, share of lost child weight",
    round(u.W[lost & ~u.adult_ok].sum() / u.W[lost].sum(), 3),
)
print(
    "all child units with no valid-card adult, share of child weight",
    round(u.W[~u.adult_ok].sum() / u.W.sum() * 100, 2),
)
print(
    u[~u.adult_ok]
    .groupby("household_support_channel")
    .W.sum()
    .div(u.W[~u.adult_ok].sum())
    .round(3)
    .to_dict()
)
fl = (u.spm_unit_is_in_spm_poverty__2024 == 0) & (
    u.spm_unit_is_in_spm_poverty__2025 == 1
)
print(
    "flips into poverty, child weight share",
    round(u.W[fl].sum() / u.W.sum() * 100, 2),
    "of which lost CTC",
    round(u.W[fl & lost].sum() / u.W[fl].sum(), 3),
    "of which lost and no valid adult",
    round(u.W[fl & lost & ~u.adult_ok].sum() / u.W[fl].sum(), 3),
)
