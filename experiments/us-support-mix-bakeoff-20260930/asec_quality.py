"""CPS ASEC response and processing quality by income year (receipt script).

Writes ``asec_quality.json`` next to this file. Two local sources:

- ``census_cps_{2018..2024}.h5`` (policyengine-us-data processed ASEC, keyed by
  income year): households, survey year, distinct and repeated ``H_IDNUM``,
  household-weight dispersion, homeownership, unemployment-compensation
  recipients, and the earnings imputation flag ``I_ERNVAL`` (9 = whole record
  imputed, 1-8 = item allocated) among wage earners.
- the Census public-use ASEC CSV archives ``asecpub{22..26}csv.zip`` (income
  years 2021-2025): the whole-supplement imputation flag ``FL_665`` (persons,
  ``MARSUPWT``-weighted) and the weighted bachelor's-or-higher share among
  adults 25 and over (``A_HGA`` >= 43).

Run: ``uv run python experiments/us-support-mix-bakeoff-20260930/asec_quality.py``
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

STORAGE = Path("/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage")
ZIPS = Path("/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/asec_education")
OUT = Path(__file__).resolve().parent / "asec_quality.json"


def h5_metrics(income_year: int, previous_ids: set | None) -> tuple[dict, set]:
    path = STORAGE / f"census_cps_{income_year}.h5"
    with pd.HDFStore(path, "r") as store:
        hh = store["household"][["H_IDNUM", "H_YEAR", "H_TENURE", "HSUP_WGT"]]
        person = store["person"][["A_FNLWGT", "I_ERNVAL", "UC_VAL", "WSAL_VAL"]]
    w = hh["HSUP_WGT"].to_numpy(np.float64) / 100
    ids = set(hh["H_IDNUM"].astype(str))
    earners = person[person["WSAL_VAL"] > 0]
    ew = earners["A_FNLWGT"].to_numpy(np.float64)
    flag = earners["I_ERNVAL"].to_numpy()
    metrics = {
        "income_year": income_year,
        "survey_year": int(hh["H_YEAR"].iloc[0]),
        "households": int(len(hh)),
        "distinct_h_idnum": len(ids),
        "share_households_in_prior_file": (
            None if previous_ids is None else len(ids & previous_ids) / len(ids)
        ),
        "household_weight_total_millions": float(w.sum() / 1e6),
        "household_weight_cv": float(w.std() / w.mean()),
        "household_weight_kish_share": float(w.sum() ** 2 / np.square(w).sum() / len(w)),
        "homeownership_weighted": float(w[hh["H_TENURE"].to_numpy() == 1].sum() / w.sum()),
        "unemployment_compensation_recipients": int((person["UC_VAL"] > 0).sum()),
        "earners_whole_record_imputed_weighted": float(ew[flag == 9].sum() / ew.sum()),
        "earners_item_allocated_weighted": float(ew[(flag >= 1) & (flag <= 8)].sum() / ew.sum()),
    }
    return metrics, ids


def zip_metrics(survey_year: int) -> dict:
    path = ZIPS / f"asecpub{survey_year % 100:02d}csv.zip"
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.lower().startswith("pppub") and n.lower().endswith(".csv"))
        with archive.open(name) as handle:
            person = pd.read_csv(handle, usecols=["FL_665", "MARSUPWT", "A_AGE", "A_HGA"])
    w = person["MARSUPWT"].to_numpy(np.float64)
    adults = person["A_AGE"].to_numpy() >= 25
    return {
        "survey_year": survey_year,
        "income_year": survey_year - 1,
        "persons": int(len(person)),
        "whole_supplement_imputed_weighted": float(w[person["FL_665"].to_numpy() != 1].sum() / w.sum()),
        "bachelors_or_higher_25plus_weighted": float(
            w[adults & (person["A_HGA"].to_numpy() >= 43)].sum() / w[adults].sum()
        ),
    }


def main() -> None:
    files, previous = [], None
    for year in range(2018, 2025):
        metrics, previous = h5_metrics(year, previous)
        files.append(metrics)
    pools = {}
    for years in ((2022, 2023, 2024), (2020, 2021, 2022, 2023, 2024)):
        households, distinct = 0, set()
        for year in years:
            with pd.HDFStore(STORAGE / f"census_cps_{year}.h5", "r") as store:
                ids = store["household"]["H_IDNUM"].astype(str)
            households += len(ids)
            distinct |= set(ids)
        pools["-".join(map(str, years))] = {"households": households, "distinct_h_idnum": len(distinct)}
    archives = [zip_metrics(year) for year in range(2022, 2027) if (ZIPS / f"asecpub{year % 100:02d}csv.zip").exists()]
    payload = {
        "processed_h5_by_income_year": files,
        "pools": pools,
        "public_use_archives_by_survey_year": archives,
        "combined_unweighted_response_rate_by_survey_year": {
            "source": "CPS ASEC technical documentation, Source and Accuracy statements "
            "(combined basic-CPS x ASEC household nonresponse), transcribed 2026-09-28",
            "2019": 0.676, "2020": 0.611, "2021": 0.650, "2022": 0.614,
            "2023": 0.596, "2024": 0.593, "2025": 0.601,
        },
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
