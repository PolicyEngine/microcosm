"""Measure the consumption-tax and domestic-energy rows on a UK dataset.

The build or the engine sets the levels these rows read (microcosm#1113,
microcosm#1123 item 3): ``obr.vat`` an engine coverage factor
(policyengine-uk#1996, #2187; held as a diagnostic, and measured with VAT on
private school fees from policyengine-uk 2.157.1), ``obr.fuel_duties_cars``
fuel that includes business-paid car fuel, the ONS electricity row the
stage's level, the kWh ONS household spending buys (microcosm#1418), the ONS
gas row the stage's Energy Trends level, and OBR's excise receipts the
clearances and bases the stage levels the excise inputs to at prior weights
(microcosm#1359), measured at their own FY2024-25 period. The tool measures
each row whether the measure-exclusion register holds it out of the
objective, as it holds the gas row while ONS 04.5.2 counts bottled gas
(microcosm#1095) and the excise rows (María's ruling of 2026-10-23; the
betting and gaming receipts also because the solve cannot yet measure their
2024 duty, microcosm#1420), or the
national binds it, as it binds the household road-fuel row (ONS 07.2.2) in
place of the cars receipts and the ONS electricity row (microcosm#1418). It
reports each row's
estimate on a national H5 that ``load_uk_national_frame`` accepts (the engine loads
it too), against the value the run's contract
registry declares, beside the household fuel and energy composition the rows
used to pull (gas-connected share, fuel-car households with no fuel,
consumption total). Development diagnostic: it never recalibrates and writes
only an aggregate JSON.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.lcfs_consumption import UK_LCFS_COICOP_DIVISION_COLUMNS

#: The measured rows, held out or bound, with the engine variables whose
#: weighted sum measures each.
ROWS: Mapping[str, tuple[str, ...]] = {
    "obr.vat": ("vat", "private_school_vat"),
    "obr.fuel_duties_cars": ("fuel_duty",),
    "ons.household_electricity_expenditure": ("electricity_consumption",),
    "ons.household_gas_expenditure": ("gas_consumption",),
    "ons.household_road_fuel_expenditure": ("petrol_spending", "diesel_spending"),
    # microcosm#1359: the duties on the excise step's quantities and bases.
    "obr.beer_and_cider_duties": ("beer_duty", "cider_duty", "sparkling_cider_duty"),
    "obr.wine_duties": ("wine_duty", "other_fermented_duty"),
    "obr.spirits_duties": ("spirits_duty",),
    "obr.tobacco_duties": ("tobacco_duty",),
    "obr.betting_and_gaming_duties": ("gambling_duty",),
}


def declared_row_years() -> dict[str, int]:
    """Each measured row's own period, where its binding declares one."""

    from importlib.resources import files

    payload = json.loads(
        files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text(encoding="utf-8")
    )
    years: dict[str, int] = {}
    for target in payload["targets"]:
        period = target["bindings"]["policyengine"].get("measurement_period")
        if target["target_id"] in ROWS and period is not None:
            years[str(target["target_id"])] = int(period)
    return years


#: Rows measured at their own period, not ``--year``: every row whose binding
#: declares a measurement_period, read from the contract so the tool measures
#: what the national binds. The excise receipts sit at FY2024-25, the year of
#: the stored quantities, which policyengine-uk charges as model year 2024 and
#: holds flat after it (microcosm#1359); road fuel and electricity at calendar
#: 2024, on the frame's own columns (microcosm#1324, #1418).
ROW_YEARS: Mapping[str, int] = declared_row_years()


def composition(
    household: pd.DataFrame, weights: pd.Series | None = None
) -> dict[str, float]:
    """Weighted fuel and energy composition from the stored household table.

    ``weights`` are the solved weights by household id, as the estimates use
    (the sixth re-review of microcosm#1420, 3): a release stores them 0.3%
    lower, at the dataset's year. Without them the stored weights are used.
    """

    if weights is None:
        weight = household["household_weight"].to_numpy(dtype=float)
    else:
        ids = household["household_id"].to_numpy()
        if set(ids.tolist()) != set(weights.index.tolist()):
            raise ValueError(
                "The household table's households and the frame's do not match."
            )
        weight = weights.reindex(ids).to_numpy(dtype=float)
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
    """Declared values of the measured rows from a run's contract registry."""

    payload = json.loads(path.read_text())
    values: dict[str, float] = {}
    for spec in payload.get("specs", []):
        if spec.get("name") in ROWS and spec.get("value") is not None:
            values[str(spec["name"])] = float(spec["value"])
    return values


def registry_row_years(path: Path) -> dict[str, int]:
    """Each measured row's period as the run's own registry declares it.

    A run built before a row's binding moved its period measured the row at
    the run's period, and its registry value is that period's fact, so the
    tool reads the period the run used, not the package's (the re-review of
    microcosm#1420, 5). A row the registry gives no period is measured at
    ``--year``.
    """

    payload = json.loads(path.read_text())
    years: dict[str, int] = {}
    for spec in payload.get("specs", []):
        period = (spec.get("metadata") or {}).get("measurement_period")
        if spec.get("name") in ROWS and period is not None:
            years[str(spec["name"])] = int(period)
    return years


def solved_household_weights(dataset: str, year: int) -> pd.Series:
    """The household weights the dataset was solved with, by household id.

    Read as the certifier reads them (``load_uk_national_frame`` then
    ``uk_solve_year_frame``): a release H5 stores the solved weights divided
    by the engine's projection from its own year, and an H5 written as solved
    (a national released before that, or the engine scratch file) stores them
    as they are (the fourth re-review of microcosm#1420, 4).
    """

    from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
    from microcosm.build.uk_runtime.weight_year import uk_solve_year_frame

    frame, _provenance = load_uk_national_frame(dataset)
    frame = uk_solve_year_frame(frame, solve_year=year)
    return pd.Series(
        np.asarray(frame.weights_for("household").values, dtype=float),
        index=frame.table("household")["household_id"].to_numpy(),
    )


def engine_estimates(
    dataset: str,
    year: int,
    row_years: Mapping[str, int] | None = None,
    weights: pd.Series | None = None,
) -> dict[str, float]:
    """Each measured row's engine total at its own period or ``year``.

    Every row is weighted with the solved weights, the calibration's, read by
    household id (``solved_household_weights`` unless ``weights`` is given):
    the engine's own weights at an earlier period read a release's 2024 rows
    0.3% low (the re-reviews of microcosm#1420, 5 and 4).
    """

    from policyengine_uk import Microsimulation

    periods = ROW_YEARS if row_years is None else row_years
    simulation = Microsimulation(dataset=dataset)
    by_id = solved_household_weights(dataset, year) if weights is None else weights
    households = np.asarray(simulation.calculate("household_id", year).values)
    if set(households.tolist()) != set(by_id.index.tolist()):
        raise ValueError("The engine's households and the frame's do not match.")
    weights = by_id.reindex(households).to_numpy(dtype=float)
    estimates: dict[str, float] = {}
    for name, variables in ROWS.items():
        period = periods.get(name, year)
        estimates[name] = float(
            sum(
                float(
                    np.dot(
                        np.asarray(
                            simulation.calculate(
                                variable, period, map_to="household"
                            ).values,
                            dtype=float,
                        ),
                        weights,
                    )
                )
                for variable in variables
            )
        )
    return estimates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="a national H5 that load_uk_national_frame accepts",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="the run's national_contract_registry.json (declared row values)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=(
            "the calibration year (default: the --registry run's, else the FRS "
            "release's)"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.year is None and args.registry:
        periods = {
            spec.get("period")
            for spec in json.loads(args.registry.read_text()).get("specs", [])
            if spec.get("period") is not None
        }
        if len(periods) == 1:
            args.year = int(periods.pop())
        elif len(periods) > 1:
            parser.error(
                f"the registry's specs span the periods {sorted(map(str, periods))}; "
                "pass --year"
            )
    if args.year is None:
        from microcosm.build.uk_runtime.frs_release import load_uk_frs_release

        args.year = int(load_uk_frs_release().calibration_year)
    household = pd.read_hdf(args.dataset, "household")
    weights = solved_household_weights(args.dataset, args.year)
    targets = registry_values(args.registry) if args.registry else {}
    row_years = registry_row_years(args.registry) if args.registry else ROW_YEARS
    estimates = engine_estimates(args.dataset, args.year, row_years, weights=weights)
    report = {
        "dataset": args.dataset,
        "year": args.year,
        "row_years": dict(row_years),
        "rows": compare(estimates, targets),
        "composition": composition(household, weights),
    }
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report["rows"], indent=1))


if __name__ == "__main__":
    main()
