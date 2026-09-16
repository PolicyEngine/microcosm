"""Size Universal Credit element support on a calibrated UK dataset.

Counts Great Britain benefit units carrying each UC element among paid UC
awards (``universal_credit > 0``), the comparison basis of the DWP element
targets, and reports the unconditioned counts the T4 admin legs used, so the
two readings are never confused again. Development diagnostic: it never
recalibrates and writes only an aggregate JSON.
"""

from __future__ import annotations

import argparse
import json
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
ELEMENTS = {
    "lcwra": "uc_LCWRA_element",
    "carer": "uc_carer_element",
    "housing": "uc_housing_costs_element",
    "childcare": "uc_childcare_element",
    "child": "uc_child_element",
    "deductions": "uc_deductions",
}


def _benunit(simulation, variable: str, year: int) -> np.ndarray:
    return np.asarray(
        simulation.calculate(variable, year, map_to="benunit").values, dtype=float
    )


def _first_person_label(simulation, variable: str, year: int) -> np.ndarray:
    """Household categorical carried to benefit units through the first member."""

    labels = np.asarray(simulation.calculate(variable, year, map_to="person")).astype(
        str
    )
    person_benunit = np.asarray(
        simulation.calculate("benunit_id", year, map_to="person").values
    )
    benunit_ids = np.asarray(simulation.calculate("benunit_id", year).values)
    first = pd.Series(labels).groupby(person_benunit).first()
    return first.reindex(benunit_ids).to_numpy().astype(str)


def measure(dataset: str, year: int) -> dict:
    from policyengine_uk import Microsimulation

    simulation = Microsimulation(dataset=dataset)
    weight = _benunit(simulation, "benunit_weight", year)
    region = _first_person_label(simulation, "region", year)
    tenure = _first_person_label(simulation, "tenure_type", year)
    universal_credit = _benunit(simulation, "universal_credit", year)
    would_claim = _benunit(simulation, "would_claim_uc", year) > 0
    eligible = _benunit(simulation, "is_uc_eligible", year) > 0
    maximum = _benunit(simulation, "uc_maximum_amount", year)
    earnings = _benunit(simulation, "employment_income", year) + _benunit(
        simulation, "self_employment_income", year
    )

    great_britain = np.isin(region, GB_REGIONS)
    paid = (universal_credit > 0) & great_britain
    entitled = would_claim & eligible & (maximum > 0) & great_britain
    social = np.isin(tenure, SOCIAL_TENURES)
    private = tenure == "RENT_PRIVATELY"

    def count(mask: np.ndarray) -> float:
        return float(weight[mask].sum())

    elements: dict[str, dict[str, float]] = {}
    for name, variable in ELEMENTS.items():
        amount = _benunit(simulation, variable, year)
        positive = amount > 0
        elements[name] = {
            "variable": variable,
            "paid_uc_gb": count(positive & paid),
            "entitled_proxy_gb": count(positive & entitled),
            "unconditioned_gb": count(positive & great_britain),
            "amount_among_paid_gbp": float((amount * weight)[paid].sum()),
            "share_of_paid": count(positive & paid) / count(paid),
        }
    housing = _benunit(simulation, ELEMENTS["housing"], year) > 0
    elements["housing_social_rented"] = {
        "variable": "uc_housing_costs_element (benunit tenure social)",
        "paid_uc_gb": count(housing & social & paid),
        "unconditioned_gb": count(housing & social & great_britain),
    }
    elements["housing_private_rented"] = {
        "variable": "uc_housing_costs_element (benunit tenure private)",
        "paid_uc_gb": count(housing & private & paid),
        "unconditioned_gb": count(housing & private & great_britain),
    }
    return {
        "dataset": str(dataset),
        "year": year,
        "basis": {
            "paid_uc_gb": "universal_credit > 0 and first member in a GB region",
            "entitled_proxy_gb": (
                "would_claim_uc and is_uc_eligible and uc_maximum_amount > 0; "
                "not an open-claim identification"
            ),
            "unconditioned_gb": "element > 0 with no UC award condition",
        },
        "paid_uc_gb": count(paid),
        "entitled_proxy_gb": count(entitled),
        "paid_with_earnings_share": count(paid & (earnings > 0)) / count(paid),
        "elements": elements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="calibrated UK H5 path")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.dataset, args.year)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["elements"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
