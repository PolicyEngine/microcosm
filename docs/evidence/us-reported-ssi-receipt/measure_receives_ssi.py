"""Measure what survey-reported SSI receipt (`receives_ssi`) changes in PE-US.

Reads the pinned default dataset (populace_us_2024.h5, sha256 6496cc43...) and
runs PolicyEngine-US on a household subset, twice per code version:

* ``base``: the dataset as published (no ``receives_ssi`` column, so the
  engine default ``False`` applies);
* ``receipt``: the same rows plus ``person.receives_ssi = SSI_VAL > 0``.

Subset = every household containing an SSI reporter (any row with SSI_VAL > 0)
plus every Colorado household. ``receives_ssi`` is a person input and a static
microsimulation has no cross-household interaction, so households without a
reporter cannot change: the weighted receipt deltas over the subset equal the
national deltas. The script checks that claim on the Colorado non-reporter
households, whose delta must be exactly zero. Colorado totals are complete
because every Colorado household is in the subset.

Usage (one code version per process):
    PYTHONPATH=<pe-us tree> python measure_receives_ssi.py <label> <out.json>
"""

from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import pandas as pd

H5 = (
    "/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/"
    "snapshots/9a814a3b3b53c0ecd6e1737b6ec862c31300ef6f/populace_us_2024.h5"
)
YEARS = (2024, 2026)
CO_FIPS = 8
AGE_BANDS = ((0, 18, "under_18"), (18, 65, "18_64"), (65, 200, "65_plus"))
AGGREGATES = (
    "household_net_income",
    "household_benefits",
    "household_state_benefits",
    "spm_unit_net_income",
    "ssi",
    "snap",
    "tanf",
    "wic",
    "medicaid",
    "chip",
    "aca_ptc",
    "lifeline",
    "head_start",
    "early_head_start",
    "free_school_meals",
    "reduced_price_school_meals",
)
COUNTS = (
    ("medicaid_enrolled", "person"),
    ("is_ssi_recipient_for_medicaid", "person"),
    ("is_snap_eligible", "spm_unit"),
)


LIQUID_ASSETS = ("bank_account_assets", "stock_assets", "bond_assets")
#: ASEC RESNSSI1/RESNSSI2 codes 3 and 4: SSI received "on behalf of a disabled
#: (blind) child" (2024 ASEC data dictionary, census.gov cpsmar24.pdf). A row
#: whose only reasons are these is a payee, not a recipient.
ON_BEHALF_OF_CHILD_CODES = (3, 4)


def own_ssi_receipt(person: pd.DataFrame) -> np.ndarray:
    """SSI_VAL > 0, excluding reporters who get SSI only on behalf of a child."""

    amount = pd.to_numeric(person["SSI_VAL"], errors="coerce").fillna(0.0)
    reason_1 = pd.to_numeric(person["RESNSSI1"], errors="coerce").fillna(0)
    reason_2 = pd.to_numeric(person["RESNSSI2"], errors="coerce").fillna(0)
    on_behalf_only = reason_1.isin(ON_BEHALF_OF_CHILD_CODES) & reason_2.isin(
        (0, *ON_BEHALF_OF_CHILD_CODES)
    )
    return ((amount > 0) & ~on_behalf_only).to_numpy(dtype=bool)


def subset_dataset(
    add_receipt: bool,
    zero_reporter_household_assets: bool = False,
    receipt_channel: str | None = None,
    anchor_puf_criteria: bool = False,
    receipt_rule: str = "own",
):
    from policyengine_us.data.dataset_schema import USSingleYearDataset

    full = USSingleYearDataset(file_path=H5)
    person = full.person.copy()
    reported = pd.to_numeric(person["SSI_VAL"], errors="coerce").fillna(0.0) > 0
    reporter_households = set(person.loc[reported, "person_household_id"])
    co_households = set(
        full.household.loc[full.household["state_fips"] == CO_FIPS, "household_id"]
    )
    keep = reporter_households | co_households
    person = person[person["person_household_id"].isin(keep)].copy()
    if zero_reporter_household_assets:
        # Upper-bound diagnostic only: every person in a reporter's household
        # holds no liquid assets, so the SSI resource test (own or deemed)
        # cannot bind for any reporter.
        in_reporter_household = person["person_household_id"].isin(reporter_households)
        for column in LIQUID_ASSETS:
            if column not in person.columns:
                raise KeyError(f"{column} is not a person column of the dataset")
            person.loc[in_reporter_household, column] = 0.0
    row_reported = (
        own_ssi_receipt(person)
        if receipt_rule == "own"
        else (
            pd.to_numeric(person["SSI_VAL"], errors="coerce").fillna(0.0) > 0
        ).to_numpy(dtype=bool)
    )
    channel = person["person_support_channel"].astype(str).to_numpy()
    if anchor_puf_criteria:
        # Counterfactual only: extend the ASEC under-65 reporter anchor of
        # meets_ssi_disability_criteria to the reporters' PUF-channel clones.
        under_65 = pd.to_numeric(person["age"], errors="coerce").to_numpy() < 65
        anchor = row_reported & under_65 & (channel == "puf_tax_detail")
        person["meets_ssi_disability_criteria"] = (
            person["meets_ssi_disability_criteria"].to_numpy(dtype=bool) | anchor
        )
    if add_receipt:
        receipt = row_reported
        if receipt_channel is not None:
            receipt = receipt & (channel == receipt_channel)
        person["receives_ssi"] = receipt

    def keep_ids(table, id_col, person_col):
        return table[table[id_col].isin(set(person[person_col]))].copy()

    ds = USSingleYearDataset(
        person=person,
        household=keep_ids(full.household, "household_id", "person_household_id"),
        tax_unit=keep_ids(full.tax_unit, "tax_unit_id", "person_tax_unit_id"),
        spm_unit=keep_ids(full.spm_unit, "spm_unit_id", "person_spm_unit_id"),
        family=keep_ids(full.family, "family_id", "person_family_id"),
        marital_unit=keep_ids(
            full.marital_unit, "marital_unit_id", "person_marital_unit_id"
        ),
        time_period=2024,
    )
    return ds, reporter_households, co_households


def ssp_variables(system) -> list[str]:
    pattern = re.compile(r"^[a-z]{2}_(ssp|state_supplement|ssi_state_supplement)$")
    return sorted(v for v in system.variables if pattern.match(v))


def person_frame(sim, year: int) -> pd.DataFrame:
    month = f"{year}-01"
    frame = pd.DataFrame(
        {
            "person_id": sim.calculate("person_id", year).values,
            "household_id": sim.calculate("household_id", year, map_to="person").values,
            "state_fips": sim.calculate("state_fips", year, map_to="person").values,
            "weight": sim.calculate("person_weight", year).values,
            "age": sim.calculate("age", year).values,
            "ssi": sim.calculate("ssi", year).values,
            "uncapped_ssi_jan": sim.calculate("uncapped_ssi", month).values,
            "abd_jan": sim.calculate("is_ssi_aged_blind_disabled", month).values,
            "resource_ok_jan": sim.calculate("meets_ssi_resource_test", month).values,
            "eligible_jan": sim.calculate("is_ssi_eligible", month).values,
            "receives_ssi_jan": sim.calculate("receives_ssi", month).values,
            "receives_ssi_jul": sim.calculate("receives_ssi", f"{year}-07").values,
            "takes_up": sim.calculate("takes_up_ssi_if_eligible", year).values,
            "disability_criteria": sim.calculate(
                "meets_ssi_disability_criteria", year
            ).values,
            "countable_resources_jan": sim.calculate(
                "ssi_countable_resources", month
            ).values,
            "countable_income_jan": sim.calculate("ssi_countable_income", month).values,
        }
    )
    return frame


def weighted(series, weights) -> float:
    return float(np.sum(np.asarray(series, dtype=float) * np.asarray(weights)))


VARIANTS = {
    "base": {"add_receipt": False},
    "receipt": {"add_receipt": True},
    "receipt_any": {"add_receipt": True, "receipt_rule": "any"},
    "receipt_assets0": {"add_receipt": True, "zero_reporter_household_assets": True},
    "receipt_asec_only": {"add_receipt": True, "receipt_channel": "asec"},
    "criteria_anchor": {"add_receipt": False, "anchor_puf_criteria": True},
    "receipt_criteria_anchor": {"add_receipt": True, "anchor_puf_criteria": True},
}


def run(label: str, variant: str) -> dict:
    from policyengine_us import Microsimulation

    started = time.time()
    ds, reporter_households, co_households = subset_dataset(**VARIANTS[variant])
    sim = Microsimulation(dataset=ds, dataset_end_year=max(YEARS))
    out: dict = {
        "label": label,
        "variant": variant,
        "households": int(len(ds.household)),
        "persons": int(len(ds.person)),
        "years": {},
    }
    ssps = ssp_variables(sim.tax_benefit_system)
    out["ssp_variables"] = ssps
    raw_ssi_val = (
        pd.to_numeric(ds.person["SSI_VAL"], errors="coerce").fillna(0.0).to_numpy()
    )
    for year in YEARS:
        pf = person_frame(sim, year)
        pf["ssi_val_reported"] = raw_ssi_val
        pf["channel"] = ds.person["person_support_channel"].astype(str).to_numpy()
        pf["any_reporter"] = raw_ssi_val > 0
        pf["reporter"] = own_ssi_receipt(ds.person)
        pf["payee_only"] = pf["any_reporter"] & ~pf["reporter"]
        pf["in_co"] = pf["state_fips"] == CO_FIPS
        pf["reporter_household"] = pf["household_id"].isin(reporter_households)
        year_out: dict = {"aggregates": {}, "ssp": {}, "counts": {}}
        hh_weight = sim.calculate("household_weight", year).values
        hh_id = sim.calculate("household_id", year).values
        hh_state = sim.calculate("state_fips", year).values
        hh_reporter = np.isin(hh_id, list(reporter_households))
        for name in AGGREGATES:
            if name not in sim.tax_benefit_system.variables:
                continue
            values = sim.calculate(name, year, map_to="household").values
            year_out["aggregates"][name] = {
                "all_subset": weighted(values, hh_weight),
                "reporter_households": weighted(values * hh_reporter, hh_weight),
                "co_households": weighted(values * (hh_state == CO_FIPS), hh_weight),
                "co_non_reporter_households": weighted(
                    values * ((hh_state == CO_FIPS) & ~hh_reporter), hh_weight
                ),
                "household_values": values.tolist(),
            }
        year_out["household_ids"] = hh_id.tolist()
        year_out["household_weights"] = hh_weight.tolist()
        for name, _entity in COUNTS:
            if name not in sim.tax_benefit_system.variables:
                continue
            var = sim.tax_benefit_system.variables[name]
            period = f"{year}-01" if var.definition_period == "month" else year
            values = sim.calculate(name, period, map_to="household").values
            year_out["counts"][name] = {
                "period": str(period),
                "reporter_households": weighted(values * hh_reporter, hh_weight),
                "household_values": values.tolist(),
            }
        for name in ssps:
            values = sim.calculate(name, year, map_to="household").values
            by_state = (
                pd.DataFrame({"state": hh_state, "v": values * hh_weight})
                .groupby("state")["v"]
                .sum()
            )
            year_out["ssp"][name] = {
                "all_subset": weighted(values, hh_weight),
                "reporter_households": weighted(values * hh_reporter, hh_weight),
                "co_households": weighted(values * (hh_state == CO_FIPS), hh_weight),
                "by_state": {
                    str(int(k)): float(v) for k, v in by_state.items() if v != 0
                },
                "household_values": values.tolist(),
            }
        # Reporter diagnostics (person grain).
        rep = pf[pf["reporter"]]
        w = rep["weight"]
        diag = {
            "reporter_persons_weighted": float(w.sum()),
            "reporter_rows": int(len(rep)),
            "reported_ssi_val_weighted": weighted(rep["ssi_val_reported"], w),
            "modeled_ssi_weighted": weighted(rep["ssi"], w),
            "modeled_ssi_positive_weighted": float(w[rep["ssi"] > 0].sum()),
            "takes_up_true_weighted": float(w[rep["takes_up"].astype(bool)].sum()),
            "receives_ssi_jan_true_weighted": float(
                w[rep["receives_ssi_jan"].astype(bool)].sum()
            ),
            "receives_ssi_jul_true_weighted": float(
                w[rep["receives_ssi_jul"].astype(bool)].sum()
            ),
            "non_reporter_receives_ssi_true_weighted": float(
                pf.loc[
                    ~pf["reporter"] & pf["receives_ssi_jan"].astype(bool), "weight"
                ].sum()
            ),
            "by_age": {},
            "zero_reasons_jan": {},
        }
        for lo, hi, key in AGE_BANDS:
            band = rep[(rep["age"] >= lo) & (rep["age"] < hi)]
            bw = band["weight"]
            diag["by_age"][key] = {
                "reporters_weighted": float(bw.sum()),
                "modeled_ssi_positive_weighted": float(bw[band["ssi"] > 0].sum()),
                "reported_ssi_val_weighted": weighted(band["ssi_val_reported"], bw),
                "modeled_ssi_weighted": weighted(band["ssi"], bw),
            }
        # Why do January-ineligible reporters get no payment? First failing test.
        jan_zero = rep[rep["uncapped_ssi_jan"] <= 0]
        zw = jan_zero["weight"]
        not_abd = ~jan_zero["abd_jan"].astype(bool)
        abd_resource_fail = jan_zero["abd_jan"].astype(bool) & ~jan_zero[
            "resource_ok_jan"
        ].astype(bool)
        other_categorical = (
            jan_zero["abd_jan"].astype(bool)
            & jan_zero["resource_ok_jan"].astype(bool)
            & ~jan_zero["eligible_jan"].astype(bool)
        )
        income = jan_zero["eligible_jan"].astype(bool)
        diag["zero_reasons_jan"] = {
            "uncapped_ssi_jan_zero_weighted": float(zw.sum()),
            "not_aged_blind_disabled": float(zw[not_abd].sum()),
            "not_abd_under_65_no_disability_criteria": float(
                zw[
                    not_abd
                    & (jan_zero["age"] < 65)
                    & ~jan_zero["disability_criteria"].astype(bool)
                ].sum()
            ),
            "abd_fails_resource_test": float(zw[abd_resource_fail].sum()),
            "abd_resources_ok_other_categorical_fail": float(
                zw[other_categorical].sum()
            ),
            "eligible_but_income_exceeds_benefit": float(zw[income].sum()),
            "resource_fail_median_countable_resources": (
                float(
                    jan_zero.loc[abd_resource_fail, "countable_resources_jan"].median()
                )
                if abd_resource_fail.any()
                else None
            ),
        }
        diag["by_channel"] = {}
        for channel, group in rep.groupby("channel"):
            gw = group["weight"]
            under65 = group["age"] < 65
            diag["by_channel"][channel] = {
                "reporter_rows": int(len(group)),
                "reporters_weighted": float(gw.sum()),
                "modeled_ssi_positive_weighted": float(gw[group["ssi"] > 0].sum()),
                "under_65_weighted": float(gw[under65].sum()),
                "under_65_disability_criteria_weighted": float(
                    gw[under65 & group["disability_criteria"].astype(bool)].sum()
                ),
            }
        payee = pf[pf["payee_only"]]
        diag["payee_only"] = {
            "rows": int(len(payee)),
            "weighted": float(payee["weight"].sum()),
            "reported_ssi_val_weighted": weighted(
                payee["ssi_val_reported"], payee["weight"]
            ),
            "modeled_ssi_positive_weighted": float(
                payee.loc[payee["ssi"] > 0, "weight"].sum()
            ),
            "modeled_ssi_weighted": weighted(payee["ssi"], payee["weight"]),
            "disability_criteria_weighted": float(
                payee.loc[payee["disability_criteria"].astype(bool), "weight"].sum()
            ),
        }
        # Colorado reporters.
        co_rep = rep[rep["in_co"]]
        diag["colorado"] = {
            "reporters_weighted": float(co_rep["weight"].sum()),
            "reporter_rows": int(len(co_rep)),
            "reporters_modeled_ssi_zero_weighted": float(
                co_rep.loc[co_rep["ssi"] <= 0, "weight"].sum()
            ),
            "reporters_age_under_60_weighted": float(
                co_rep.loc[co_rep["age"] < 60, "weight"].sum()
            ),
        }
        # Reported versus modeled amounts among reporters with a modeled payment.
        paid = rep[rep["ssi"] > 0]
        if len(paid):
            ratio = paid["ssi"] / paid["ssi_val_reported"]
            order = np.argsort(ratio.to_numpy())
            cw = np.cumsum(paid["weight"].to_numpy()[order])
            median = float(ratio.to_numpy()[order][np.searchsorted(cw, cw[-1] / 2)])
            diag["paid_reporters_modeled_over_reported_weighted_median"] = median
        year_out["reporter_diagnostics"] = diag
        out["years"][str(year)] = year_out
    out["seconds"] = round(time.time() - started, 1)
    return out


if __name__ == "__main__":
    label, path = sys.argv[1], sys.argv[2]
    variants = sys.argv[3].split(",") if len(sys.argv) > 3 else list(VARIANTS)
    results = [run(label, variant) for variant in variants]
    with open(path, "w") as handle:
        json.dump(results, handle)
    for result in results:
        print(
            result["label"],
            result["variant"],
            result["households"],
            result["persons"],
            result["seconds"],
            "s",
        )
