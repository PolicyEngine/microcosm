"""Size the benefit-cap reduction tail on a UK spine or calibrated dataset.

DWP's benefit-cap Table 4 counts UC households capped by the amount capped
per monthly assessment period. The engine's ``benefit_cap_reduction`` is an
annual amount, so the bands are the monthly edges times twelve, half-open on
the lower edge and closed on the upper. The tool reports the band counts on
the paid-claim GB basis the contract binds, the mean and median reduction,
the composition of the tail (family type, children, elements, regions) and,
on a spine, how many source households stand behind each band. Development
diagnostic: it never recalibrates and writes only an aggregate JSON.
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
# DWP Table 4 monthly bands: (band id, lower edge exclusive, upper edge inclusive).
BANDS = (
    ("up_to_100", 0, 100),
    ("100_01_to_200", 100, 200),
    ("200_01_to_300", 200, 300),
    ("300_01_to_400", 300, 400),
    ("400_01_to_500", 400, 500),
    ("500_01_to_600", 500, 600),
    ("600_01_to_700", 600, 700),
    ("700_01_to_800", 700, 800),
    ("800_01_to_900", 800, 900),
    ("900_01_to_1000", 900, 1000),
    ("1000_01_to_1100", 1000, 1100),
    ("1100_01_to_1200", 1100, 1200),
    ("1200_01_to_1300", 1200, 1300),
    ("1300_01_and_above", 1300, None),
)
TAIL_EDGE = 1300
COMPONENTS = {
    "uc_maximum_amount": "uc_maximum_amount",
    "housing_element": "uc_housing_costs_element",
    "child_element": "uc_child_element",
    "standard_allowance": "uc_standard_allowance",
    "child_benefit": "child_benefit",
}


def _first_person_label(simulation, variable: str, year: int) -> np.ndarray:
    labels = np.asarray(simulation.calculate(variable, year, map_to="person")).astype(
        str
    )
    codes, uniques = pd.factorize(pd.Series(labels))
    first = simulation.populations["benunit"].value_from_first_person(codes)
    return np.asarray(uniques)[first].astype(str)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(np.interp(0.5, cumulative, values[order]))


def measure(dataset: str, year: int, spine: str | None = None) -> dict:
    from policyengine_uk import Microsimulation

    simulation = Microsimulation(dataset=dataset)
    benunit = simulation.populations["benunit"]
    weight = np.asarray(
        simulation.calculate("benunit_weight", year).values, dtype=float
    )
    region = _first_person_label(simulation, "region", year)
    universal_credit = np.asarray(
        simulation.calculate("universal_credit", year).values, dtype=float
    )
    reduction = np.asarray(
        simulation.calculate("benefit_cap_reduction", year).values, dtype=float
    )
    family_type = np.asarray(simulation.calculate("family_type", year).values).astype(
        str
    )
    is_child = np.asarray(simulation.calculate("is_child", year).values, dtype=bool)
    children = benunit.sum(is_child.astype(float))
    monthly = reduction / 12.0
    great_britain = np.isin(region, GB_REGIONS)
    capped = (reduction > 0) & great_britain
    paid = capped & (universal_credit > 0)

    def count(mask: np.ndarray) -> float:
        return float(weight[mask].sum())

    def share_by(mask: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        total = count(mask)
        return {
            str(label): count(mask & (labels == label)) / total if total else 0.0
            for label in sorted(set(labels[mask]))
        }

    bands: dict[str, dict[str, float]] = {}
    for band, lower, upper in BANDS:
        in_band = capped & (monthly > lower)
        if upper is not None:
            in_band &= monthly <= upper
        bands[band] = {
            "units_gb": count(in_band),
            "paid_uc_gb": count(in_band & paid),
            "share_of_capped": count(in_band) / count(capped) if count(capped) else 0.0,
            "records": int(in_band.sum()),
        }
    tail = capped & (monthly > TAIL_EDGE)
    components = {}
    for name, variable in COMPONENTS.items():
        amount = np.asarray(
            simulation.calculate(variable, year, map_to="benunit").values, dtype=float
        )
        components[name] = {
            "tail_mean_monthly_gbp": float(
                np.average(amount[tail] / 12.0, weights=weight[tail])
            )
            if tail.any()
            else 0.0,
            "capped_mean_monthly_gbp": float(
                np.average(amount[capped] / 12.0, weights=weight[capped])
            )
            if capped.any()
            else 0.0,
        }
    result = {
        "dataset": str(dataset),
        "year": year,
        "basis": {
            "capped": "benefit_cap_reduction > 0 and first member in a GB region",
            "paid_uc_gb": "capped and universal_credit > 0",
            "bands": "monthly reduction = annual benefit_cap_reduction / 12; "
            "lower edge exclusive, upper edge inclusive, last band open",
        },
        "capped_units_gb": count(capped),
        "paid_uc_gb": count(paid),
        "mean_monthly_reduction_gbp": float(
            np.average(monthly[capped], weights=weight[capped])
        )
        if capped.any()
        else 0.0,
        "median_monthly_reduction_gbp": _weighted_median(
            monthly[capped], weight[capped]
        )
        if capped.any()
        else 0.0,
        "bands": bands,
        "tail": {
            "edge_monthly_gbp": TAIL_EDGE,
            "units_gb": count(tail),
            "share_of_capped": count(tail) / count(capped) if count(capped) else 0.0,
            "records": int(tail.sum()),
            "mean_children": float(np.average(children[tail], weights=weight[tail]))
            if tail.any()
            else 0.0,
            "family_type_shares": share_by(tail, family_type),
            "region_shares": share_by(tail, region),
        },
        "capped": {
            "mean_children": float(np.average(children[capped], weights=weight[capped]))
            if capped.any()
            else 0.0,
            "family_type_shares": share_by(capped, family_type),
            "region_shares": share_by(capped, region),
        },
        "components": components,
    }
    if spine:
        result["support"] = _support(spine, simulation, year, capped, tail, monthly)
    return result


def _support(
    spine: str, simulation, year: int, capped: np.ndarray, tail: np.ndarray, monthly
) -> dict:
    """Distinct source households behind the capped units, from the spine tables.

    The join is positional: the engine's household population keeps the spine
    household table's row order, so each benefit unit's household position
    (through its first member) indexes the spine's ``household_source_id``
    directly. No id is cast, so no two households can collide.
    """

    household = pd.read_hdf(spine, "/household")
    if "household_source_id" not in household.columns:
        return {"available": False}
    household_population = simulation.populations["household"]
    person_household_position = np.asarray(
        household_population.members_entity_id, dtype=np.int64
    )
    benunit = simulation.populations["benunit"]
    positions = np.asarray(
        benunit.value_from_first_person(person_household_position), dtype=np.int64
    )
    if len(household) != household_population.count or positions.max() >= len(
        household
    ):
        return {
            "available": False,
            "reason": "engine household order does not match the spine household table",
            "spine_households": int(len(household)),
            "engine_households": int(household_population.count),
        }
    source = household["household_source_id"].to_numpy()[positions]
    out = {
        "available": True,
        "join": "positional (engine household order is spine table order)",
        "capped_records": int(capped.sum()),
        "capped_source_households": int(len(set(source[capped]))),
        "tail_records": int(tail.sum()),
        "tail_source_households": int(len(set(source[tail]))),
        "bands": {},
    }
    for band, lower, upper in BANDS:
        in_band = capped & (monthly > lower)
        if upper is not None:
            in_band &= monthly <= upper
        out["bands"][band] = int(len(set(source[in_band])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="an H5 the engine loads")
    parser.add_argument(
        "--spine",
        help="the spine H5 whose household table carries household_source_id "
        "(support counts; only meaningful when --dataset is that spine)",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.dataset, args.year, spine=args.spine)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "capped_units_gb": result["capped_units_gb"],
                "mean": result["mean_monthly_reduction_gbp"],
                "tail_share": result["tail"]["share_of_capped"],
                "bands": {k: v["units_gb"] for k, v in result["bands"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
