"""The engine's duty on the committed declaration's national levels, against OBR.

Usage, from the tree (its packages first on PYTHONPATH, policyengine-uk as locked):
  <venv>/bin/python <this> <out.json>

One household holds every national level the committed ``level_excise_quantities``
declaration resolves to from the vendored ``excise_facts.json``. Every duty is
linear in its inputs, so its duty is the frame's at prior weights however the
step shares the levels out. Each OBR row is read through its own binding's
variables, at the binding's measurement period, against the value the national
compile-parity receipt records from the pinned feed. Public figures only: the
levels are published national quantities and nothing here reads microdata.
"""

from __future__ import annotations

import importlib.resources
import json
import sys
from pathlib import Path

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.lcfs_consumption import (
    UK_LCFS_EXCISE_TOBACCO_OUTPUTS,
    UK_PE_UK_WEEKS_IN_YEAR,
    excise_levels,
    excise_quantities_operation,
)

ROWS = (
    "obr.beer_and_cider_duties",
    "obr.wine_duties",
    "obr.spirits_duties",
    "obr.tobacco_duties",
    "obr.betting_and_gaming_duties",
)
DUTIES = (
    "beer_duty",
    "cider_duty",
    "sparkling_cider_duty",
    "wine_duty",
    "other_fermented_duty",
    "spirits_duty",
    "alcohol_duty",
    "tobacco_duty",
    "general_betting_duty",
    "pool_betting_duty",
    "remote_gaming_duty",
    "machine_games_duty",
    "bingo_duty",
    "lottery_duty",
    "gaming_duty",
    "gambling_duty",
    "household_tax",
)


def _resource(name: str) -> dict:
    return json.loads(
        importlib.resources.files("microcosm.build.uk").joinpath(name).read_text()
    )


def main() -> None:
    from policyengine_uk import Simulation

    declared = excise_quantities_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    resolved = excise_levels(declared)
    year = int(declared["period_value"])
    inputs = {
        column: level.annual / UK_PE_UK_WEEKS_IN_YEAR
        if column in UK_LCFS_EXCISE_TOBACCO_OUTPUTS
        else level.annual
        for column, level in resolved.quantities.items()
    }
    inputs.update(
        {
            column: float(draught["share"])
            for column, draught in resolved.draught.items()
        }
    )
    simulation = Simulation(
        situation={
            "people": {"adult": {"age": {year: 40}}},
            "benunits": {"benunit": {"members": ["adult"]}},
            "households": {
                "household": {
                    "members": ["adult"],
                    **{column: {year: value} for column, value in inputs.items()},
                }
            },
        }
    )
    duty = {name: float(simulation.calculate(name, year)[0]) for name in DUTIES}
    bindings = {
        target["target_id"]: target["bindings"]["policyengine"]
        for target in _resource("uk_population_targets.json")["targets"]
    }
    published = {
        row["name"]: row["ledger_value"]
        for row in _resource(
            "ledger_compile_parity_incumbent_2025_signed_differences.json"
        )["differences"]
        if row["name"] in ROWS
    }
    rows = {}
    for name in ROWS:
        binding = bindings[name]
        variables = (
            binding["value_expression"].split(" + ")
            if "value_expression" in binding
            else [binding["value_variable"]]
        )
        engine = sum(duty[variable] for variable in variables)
        rows[name] = {
            "variables": variables,
            "measurement_period": binding["measurement_period"],
            "engine": engine,
            "obr": published[name],
            "gap": engine / published[name] - 1.0,
        }
    report = {
        "year": year,
        "national_levels": inputs,
        "duties": duty,
        "rows": rows,
        "three_duties": duty["alcohol_duty"]
        + duty["tobacco_duty"]
        + duty["gambling_duty"],
    }
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for name, row in rows.items():
        print(f"{name}: {row['engine'] / 1e9:.3f}bn against {row['obr'] / 1e9:.3f}bn")


if __name__ == "__main__":
    main()
