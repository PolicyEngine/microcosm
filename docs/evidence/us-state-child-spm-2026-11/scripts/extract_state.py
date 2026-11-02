"""Run one state's cut of a Populace file and store SPM-unit results.

Follows state_baseline.py on PolicyEngine/policyengine-slides#95: one state at a
time, as a flat file with group columns on each member's row.
"""

import sys
import time
import warnings
from importlib.metadata import version
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from policyengine_core.data import Dataset
from policyengine_us import CountryTaxBenefitSystem, Microsimulation

DATA_YEAR = 2024
YEARS = (2024, 2025)
GROUPS = ("tax_unit", "spm_unit", "family", "marital_unit")
SPM_VARS = """spm_unit_is_in_spm_poverty spm_unit_net_income spm_unit_spm_threshold
spm_unit_reference_spm_threshold spm_unit_unadjusted_spm_threshold spm_unit_geographic_adjustment
spm_measurement_adults spm_measurement_children spm_unit_market_income spm_unit_benefits
spm_unit_taxes spm_unit_spm_expenses spm_unit_payroll_tax spm_unit_self_employment_tax
spm_unit_federal_tax spm_unit_state_tax spm_unit_local_tax child_support_expense
spm_unit_medical_out_of_pocket_expenses spm_unit_capped_work_childcare_expenses
snap wic free_school_meals reduced_price_school_meals spm_unit_capped_housing_subsidy
spm_unit_energy_subsidy tanf ssi social_security total_unemployment_compensation
child_support_received workers_compensation educational_assistance acp ebb""".split()
# Mapped up from person or tax unit to the SPM unit.
UP_VARS = """eitc refundable_ctc employment_income self_employment_income capital_gains
partnership_s_corp_income""".split()

tables = None
known = None
path = None


def load(h5):
    global tables, known, path
    path = h5
    known = set(CountryTaxBenefitSystem().variables)
    with pd.HDFStore(h5, "r") as store:
        tables = {n: store.select(n) for n in ("household", "person", *GROUPS)}


def state_file(fips):
    households = tables["household"]
    households = households[households["state_fips"] == fips]
    people = tables["person"]
    people = people[
        people["person_household_id"].isin(households["household_id"])
    ].reset_index(drop=True)
    columns = {
        n: people[n].values
        for n in people.columns
        if n in known or n == "person_id" or n.startswith("person_")
    }

    def add_group(table, group):
        ids = f"{group}_id"
        membership = people[f"person_{group}_id"].values
        rows = table.set_index(ids).index.get_indexer(membership)
        for name in table.columns:
            if name in known:
                columns[name] = table[name].values[rows]
        columns[ids] = membership

    add_group(households, "household")
    for group in GROUPS:
        add_group(tables[group], group)
    return pd.DataFrame({f"{k}__{DATA_YEAR}": v for k, v in columns.items()})


def run(args):
    fips, out = args
    target = Path(out) / f"{fips:02d}.parquet"
    if target.exists():
        return fips, "cached"
    started = time.time()
    sim = Microsimulation(dataset=Dataset.from_dataframe(state_file(fips), DATA_YEAR))
    ids = sim.calculate("spm_unit_id", DATA_YEAR).values
    frame = {"spm_unit_id": ids}
    age = sim.calculate("age", DATA_YEAR).values
    # Children per SPM unit from the person rows.
    person_unit = sim.calculate("person_spm_unit_id", DATA_YEAR).values
    frame["n_children"] = (
        pd.Series(age < 18).groupby(person_unit).sum().reindex(ids).values
    )
    frame["n_people"] = pd.Series(age).groupby(person_unit).size().reindex(ids).values
    for year in YEARS:
        for name in SPM_VARS:
            frame[f"{name}__{year}"] = np.asarray(
                sim.calculate(name, year, map_to="spm_unit").values, dtype=float
            )
        for name in UP_VARS:
            frame[f"{name}__{year}"] = np.asarray(
                sim.calculate(name, year, map_to="spm_unit").values, dtype=float
            )
    frame = pd.DataFrame(frame)
    frame["state_fips"] = fips
    frame.to_parquet(target)
    return fips, round(time.time() - started)


def main(h5, out, only=None, workers=3):
    Path(out).mkdir(parents=True, exist_ok=True)
    load(h5)
    fips = sorted(int(v) for v in tables["household"]["state_fips"].unique())
    if only:
        fips = [f for f in fips if f in only]
    print(
        "policyengine-us", version("policyengine-us"), "states", len(fips), flush=True
    )
    with Pool(workers, initializer=load, initargs=(h5,)) as pool:
        for fips_done, seconds in pool.imap_unordered(run, [(f, out) for f in fips]):
            print(fips_done, seconds, flush=True)


if __name__ == "__main__":
    only = (
        [int(v) for v in sys.argv[3].split(",")]
        if len(sys.argv) > 3 and sys.argv[3]
        else None
    )
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    main(sys.argv[1], sys.argv[2], only, workers)
