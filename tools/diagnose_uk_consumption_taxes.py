"""Measure the consumption-tax and domestic-energy rows on a UK dataset.

The levels of these rows are set by the build or the engine, not by the
weights (microcosm#1113, microcosm#1123 item 3): ``obr.vat`` by an engine
coverage factor (policyengine-uk#1996), ``obr.fuel_duties_cars`` by fuel
that includes business-paid car fuel, and the ONS electricity and gas rows
by a published-source residual against the stage's DESNZ volume at QEP
prices. Rows held out of the objective by the measure-exclusion register stay
measured here: the tool reports each row's estimate on any H5 the engine
loads, against the value the run's contract registry declares, beside the
household fuel and energy composition the rows used to pull (gas-connected
share, fuel-car households with no fuel, consumption total). Development
diagnostic: it never recalibrates and writes only an aggregate JSON.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.lcfs_consumption import UK_LCFS_COICOP_DIVISION_COLUMNS

#: Held-out rows and the engine variables whose weighted sum measures them.
ROWS: Mapping[str, tuple[str, ...]] = {
    "obr.vat": ("vat",),
    "obr.fuel_duties_cars": ("fuel_duty",),
    "ons.household_electricity_expenditure": ("electricity_consumption",),
    "ons.household_gas_expenditure": ("gas_consumption",),
}


def composition(household: pd.DataFrame) -> dict[str, float]:
    """Weighted fuel and energy composition from the stored household table."""

    weight = household["household_weight"].to_numpy(dtype=float)
    total = float(weight.sum())
    if total <= 0:
        raise ValueError("household weights sum to zero.")
    petrol = household["petrol_spending"].to_numpy(dtype=float)
    diesel = household["diesel_spending"].to_numpy(dtype=float)
    fuel = petrol + diesel
    gas = household["gas_consumption"].to_numpy(dtype=float)
    flagged = household["has_fuel_consumption"].to_numpy(dtype=bool)
    flagged_weight = float(weight[flagged].sum())
    out = {
        "households": total,
        "gas_connected_share": float(weight[gas > 0].sum()) / total,
        "petrol_positive_share": float(weight[petrol > 0].sum()) / total,
        "diesel_positive_share": float(weight[diesel > 0].sum()) / total,
        "fuel_car_household_share": flagged_weight / total,
        "fuel_car_zero_fuel_share": (
            float(weight[flagged & (fuel <= 0)].sum()) / flagged_weight
            if flagged_weight > 0
            else 0.0
        ),
        "consumption_stored": float(
            np.dot(
                weight,
                household[list(UK_LCFS_COICOP_DIVISION_COLUMNS)]
                .to_numpy(dtype=float)
                .sum(axis=1),
            )
        ),
    }
    for column in (
        "petrol_spending",
        "diesel_spending",
        "electricity_consumption",
        "gas_consumption",
    ):
        out[f"{column}_stored"] = float(
            np.dot(weight, household[column].to_numpy(dtype=float))
        )
    return out


def compare(
    estimates: Mapping[str, float], targets: Mapping[str, float]
) -> dict[str, dict[str, float | None]]:
    """Each row's estimate beside its declared value and relative error."""

    rows: dict[str, dict[str, float | None]] = {}
    for name, estimate in estimates.items():
        target = targets.get(name)
        rows[name] = {
            "estimate": float(estimate),
            "target": None if target is None else float(target),
            "relative_error": (
                None if not target else float(estimate) / float(target) - 1.0
            ),
        }
    return rows


def registry_values(path: Path) -> dict[str, float]:
    """Declared values of the held-out rows from a run's contract registry."""

    payload = json.loads(path.read_text())
    values: dict[str, float] = {}
    for spec in payload.get("specs", []):
        if spec.get("name") in ROWS and spec.get("value") is not None:
            values[str(spec["name"])] = float(spec["value"])
    return values


def engine_estimates(dataset: str, year: int) -> dict[str, float]:
    """Weighted engine totals of each held-out row at ``year``."""

    from policyengine_uk import Microsimulation

    simulation = Microsimulation(dataset=dataset)
    estimates: dict[str, float] = {}
    for name, variables in ROWS.items():
        estimates[name] = float(
            sum(
                float(simulation.calculate(variable, year).sum())
                for variable in variables
            )
        )
    return estimates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="an H5 the engine loads")
    parser.add_argument(
        "--registry",
        type=Path,
        help="the run's national_contract_registry.json (declared row values)",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    household = pd.read_hdf(args.dataset, "household")
    targets = registry_values(args.registry) if args.registry else {}
    report = {
        "dataset": args.dataset,
        "year": args.year,
        "rows": compare(engine_estimates(args.dataset, args.year), targets),
        "composition": composition(household),
    }
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report["rows"], indent=1))


if __name__ == "__main__":
    main()
