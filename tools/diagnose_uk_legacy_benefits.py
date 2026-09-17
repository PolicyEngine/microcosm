"""Decompose the UK legacy-benefit rows on a spine or a calibrated dataset.

Housing Benefit, Jobseeker's Allowance and Employment and Support Allowance
are carried by the frame as FRS-reported receipt. This tool reports where the
reporters sit (support channel, tenure, pension age, the ``would_claim_uc``
draw) and how much of that receipt the engine pays, against the DWP caseload
facts the contract binds. With ``--frs-raw-dir`` it adds the raw FRS 2024-25
benchmarks (benefit codes 94 HB, 14 JSA, 16 ESA, 13 Carer's Allowance; VAR2 1/3
contributory, 2/4 income-based). Development diagnostic: it never recalibrates
and writes only an aggregate JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

GB_REGIONS = (
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "WALES",
    "SCOTLAND",
)
SOCIAL_TENURES = ("RENT_FROM_COUNCIL", "RENT_FROM_HA")
RENTING_TENURES = SOCIAL_TENURES + ("RENT_PRIVATELY",)
REPORTED = {
    "housing_benefit": "housing_benefit_reported",
    "jsa_contrib": "jsa_contrib_reported",
    "jsa_income": "jsa_income_reported",
    "esa_contrib": "esa_contrib_reported",
    "esa_income": "esa_income_reported",
    "carers_allowance": "carers_allowance_reported",
    "universal_credit": "universal_credit_reported",
}
ENGINE = {
    "housing_benefit": "housing_benefit",
    "jsa_contrib": "jsa_contrib",
    "jsa_income": "jsa_income",
    "esa_contrib": "esa_contrib",
    "esa_income": "esa_income",
    "esa": "esa",
}
RAW_BENEFIT_CODES = {
    "housing_benefit": 94,
    "jsa": 14,
    "esa": 16,
    "carers_allowance": 13,
}


def _spine_tables(path: str) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_hdf(path, f"/{name}")
        for name in ("person", "benunit", "household")
    }


def _benunit_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per benefit unit with the reporter, age, tenure and channel facts."""

    person = tables["person"]
    household = tables["household"].set_index("household_id")
    person = person.assign(
        weight=person["person_household_id"].map(household["household_weight"]),
        tenure=person["person_household_id"].map(household["tenure_type"]).astype(str),
        region=person["person_household_id"].map(household["region"]).astype(str),
        channel=person["person_household_id"]
        .map(household["household_support_channel"])
        .astype(str),
        rent=person["person_household_id"].map(household["rent"]),
    )
    adults = person[person["age"] >= 16]
    ages = adults.groupby("person_benunit_id")["age"].agg(["min", "max"])
    rows = person.groupby("person_benunit_id").agg(
        weight=("weight", "first"),
        tenure=("tenure", "first"),
        region=("region", "first"),
        channel=("channel", "first"),
        rent=("rent", "first"),
        household_head=("is_household_head", "max"),
        **{name: (column, "sum") for name, column in REPORTED.items()},
    )
    rows["all_pension_age"] = ages["min"].reindex(rows.index).fillna(0) >= 66
    rows["any_pension_age"] = ages["max"].reindex(rows.index).fillna(0) >= 66
    benunit = tables["benunit"].set_index("benunit_id")
    rows["would_claim_uc"] = (
        benunit["would_claim_uc"].reindex(rows.index).fillna(False).astype(bool)
    )
    return rows


def _age_class(rows: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            rows["all_pension_age"],
            "all_pension_age",
            np.where(rows["any_pension_age"], "mixed_age", "working_age"),
        ),
        index=rows.index,
    )


def _split(rows: pd.DataFrame, mask: pd.Series, by: list[str]) -> dict:
    selected = rows[mask]
    if selected.empty:
        return {}
    grouped = selected.groupby(by, observed=True)["weight"].sum()
    return {
        " | ".join(map(str, key if isinstance(key, tuple) else (key,))): float(v)
        for key, v in grouped.items()
    }


def measure_spine(path: str) -> dict:
    rows = _benunit_frame(_spine_tables(path))
    rows["age_class"] = _age_class(rows)
    rows["renting"] = rows["tenure"].isin(RENTING_TENURES)
    gb = rows["region"].isin(GB_REGIONS)
    result: dict[str, object] = {
        "dataset": str(path),
        "weights": "design",
        "reporters": {},
    }
    for name in REPORTED:
        mask = (rows[name] > 0) & gb
        entry = {
            "units": float(rows.loc[mask, "weight"].sum()),
            "reported_gbp": float((rows[name] * rows["weight"])[mask].sum()),
            "by_channel": _split(rows, mask, ["channel"]),
            "by_channel_tenure": _split(rows, mask, ["channel", "tenure"]),
            "by_age_class": _split(rows, mask, ["age_class"]),
            "by_age_class_would_claim_uc": _split(
                rows, mask, ["age_class", "would_claim_uc"]
            ),
        }
        if name == "housing_benefit":
            entry["renting_units"] = float(
                rows.loc[mask & rows["renting"], "weight"].sum()
            )
            entry["renting_not_drawn_into_uc"] = float(
                rows.loc[
                    mask & rows["renting"] & ~rows["would_claim_uc"], "weight"
                ].sum()
            )
            entry["non_head_units"] = float(
                rows.loc[mask & (rows["household_head"] == 0), "weight"].sum()
            )
            entry["household_rent_zero_units"] = float(
                rows.loc[mask & (rows["rent"] <= 0), "weight"].sum()
            )
        result["reporters"][name] = entry
    result["would_claim_uc"] = {
        "share_all_units": float(
            np.average(rows["would_claim_uc"], weights=rows["weight"])
        ),
        "all_pension_age_units": float(
            rows.loc[rows["all_pension_age"], "weight"].sum()
        ),
        "all_pension_age_drawn": float(
            rows.loc[rows["all_pension_age"] & rows["would_claim_uc"], "weight"].sum()
        ),
        "share_among_units_with_a_working_age_adult": float(
            np.average(
                rows.loc[~rows["all_pension_age"], "would_claim_uc"],
                weights=rows.loc[~rows["all_pension_age"], "weight"],
            )
        ),
    }
    result["household_mass_by_channel"] = _split(
        rows.drop_duplicates(subset=["channel"]).assign(weight=0),
        rows["weight"] >= 0,
        ["channel"],
    )
    household = _spine_tables(path)["household"]
    result["household_mass_by_channel"] = {
        str(k): float(v)
        for k, v in household.groupby("household_support_channel")["household_weight"]
        .sum()
        .items()
    }
    return result


def _first_person_label(simulation, variable: str, year: int) -> np.ndarray:
    labels = np.asarray(simulation.calculate(variable, year, map_to="person")).astype(
        str
    )
    codes, uniques = pd.factorize(pd.Series(labels))
    first = simulation.populations["benunit"].value_from_first_person(codes)
    return np.asarray(uniques)[first].astype(str)


def measure_engine(dataset: str, year: int) -> dict:
    """Engine-paid legacy benefits on a dataset the engine can load (spine or calibrated)."""

    from policyengine_uk import Microsimulation

    simulation = Microsimulation(dataset=dataset)
    benunit = simulation.populations["benunit"]
    weight = np.asarray(
        simulation.calculate("benunit_weight", year).values, dtype=float
    )
    region = _first_person_label(simulation, "region", year)
    tenure = np.asarray(
        simulation.calculate("benunit_tenure_type", year).values
    ).astype(str)
    gb = np.isin(region, GB_REGIONS)
    would_claim = np.asarray(
        simulation.calculate("would_claim_uc", year).values, dtype=bool
    )
    pension_age = np.asarray(simulation.calculate("is_SP_age", year).values, dtype=bool)
    adult = np.asarray(simulation.calculate("age", year).values) >= 16
    all_pension_age = benunit.all(pension_age | ~adult) & benunit.any(adult)
    any_pension_age = benunit.any(pension_age & adult)
    age_class = np.where(
        all_pension_age,
        "all_pension_age",
        np.where(any_pension_age, "mixed_age", "working_age"),
    )
    social = np.isin(tenure, SOCIAL_TENURES)
    private = tenure == "RENT_PRIVATELY"

    def count(mask: np.ndarray) -> float:
        return float(weight[mask].sum())

    def by(mask: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        return {
            str(label): count(mask & (labels == label))
            for label in sorted(set(labels[mask]))
        }

    result: dict[str, object] = {
        "dataset": str(dataset),
        "year": year,
        "weights": "as loaded",
        "engine": {},
    }
    for name, variable in ENGINE.items():
        amount = np.asarray(
            simulation.calculate(variable, year, map_to="benunit").values, dtype=float
        )
        paid = (amount > 0) & gb
        entry = {
            "variable": variable,
            "units_gb": count(paid),
            "gbp": float((amount * weight)[paid].sum()),
            "by_age_class": by(paid, age_class),
        }
        if name == "housing_benefit":
            reported = benunit.sum(
                np.asarray(
                    simulation.calculate("housing_benefit_reported", year).values,
                    dtype=float,
                )
            )
            reporters = (reported > 0) & gb
            entry.update(
                {
                    "social_rented_units": count(paid & social),
                    "private_rented_units": count(paid & private),
                    "reporter_units_gb": count(reporters),
                    "reporter_gbp": float((reported * weight)[reporters].sum()),
                    "reporters_by_age_class": by(reporters, age_class),
                    "reporters_drawn_into_uc_by_age_class": by(
                        reporters & would_claim, age_class
                    ),
                    "reporters_renting_not_drawn_by_age_class": by(
                        reporters & (social | private) & ~would_claim, age_class
                    ),
                }
            )
        result["engine"][name] = entry
    result["would_claim_uc"] = {
        "share_all_units": float(np.average(would_claim, weights=weight)),
        "all_pension_age_drawn": count(would_claim & all_pension_age),
        "paid_uc_all_pension_age": count(
            (np.asarray(simulation.calculate("universal_credit", year).values) > 0)
            & all_pension_age
        ),
    }
    return result


def measure_raw_frs(raw_dir: str) -> dict:
    """Raw FRS 2024-25 reporter benchmarks by tenure and claim type (design weights)."""

    raw = Path(raw_dir)
    weight: dict[str, float] = {}
    tenure: dict[str, str] = {}
    with open(raw / "househol.tab") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            weight[row["SERNUM"]] = float(row.get("gross4") or 0)
            tenure[row["SERNUM"]] = row.get("ptentyp2", "").strip()
    units: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    with open(raw / "benefits.tab") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            code = row["BENEFIT"].strip()
            for name, expected in RAW_BENEFIT_CODES.items():
                if code == str(expected):
                    var2 = row.get("VAR2", "").strip()
                    units[(name, var2)].add((row["SERNUM"], row["BENUNIT"]))
    result: dict[str, object] = {
        "raw_dir": str(raw),
        "weights": "gross4",
        "benefit_units": {},
    }
    for (name, var2), keys in sorted(units.items()):
        total = sum(weight.get(sernum, 0.0) for sernum, _ in keys)
        by_tenure: dict[str, float] = defaultdict(float)
        for sernum, _ in keys:
            by_tenure[tenure.get(sernum, "?")] += weight.get(sernum, 0.0)
        result["benefit_units"][f"{name}|VAR2={var2 or 'none'}"] = {
            "records": len(keys),
            "units": total,
            "by_ptentyp2": dict(sorted(by_tenure.items())),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spine",
        help="spine H5 with the person/benunit/household tables (design weights)",
    )
    parser.add_argument(
        "--dataset",
        help="an H5 the engine loads (spine or calibrated) for the engine-paid side",
    )
    parser.add_argument(
        "--frs-raw-dir",
        help="licensed FRS 2024-25 tab directory for the raw benchmarks",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.spine or args.dataset or args.frs_raw_dir):
        parser.error("give at least one of --spine, --dataset, --frs-raw-dir")
    result: dict[str, object] = {"year": args.year}
    if args.spine:
        result["spine"] = measure_spine(args.spine)
    if args.dataset:
        result["engine"] = measure_engine(args.dataset, args.year)
    if args.frs_raw_dir:
        result["raw_frs"] = measure_raw_frs(args.frs_raw_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        key: {
            name: entry.get("units", entry.get("units_gb"))
            for name, entry in section.get(
                "reporters", section.get("engine", {})
            ).items()
        }
        for key, section in result.items()
        if isinstance(section, dict) and ("reporters" in section or "engine" in section)
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
