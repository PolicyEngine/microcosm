"""Count and cost of the State Pension income tax allowance on one dataset.

Runs uk-autumn-budget-dashboard-2026's
``state_pension_only_income_tax_allowance`` through policyengine.py
(``pe.uk.managed_microsimulation(dataset=..., allow_unmanaged=True)``, the
dashboard's ``pipeline.build_microsimulation``) for 2027-28 to 2031-32, in
two designs:

- ``sole_income``, PR #182's: ``other_income_limit=0``,
  ``total_income_at_most_allowance=False``, ``disregard_savings_interest=True``;
- ``total_income``, the module's default on the dashboard's main since then.

Cost is the change in ``gov_balance`` summed over households, as
``BudgetaryImpactCalculator`` computes it (negative is a cost). People
directly affected follow ``AgeImpactCalculator``: a change of at least £1 a
year in the person's own income tax, National Insurance, student loan
repayment or employment income, among people in a household with an income
decile, weighted by ``person_weight``. Writes one JSON of aggregates.

Needs the policyengine.py venv and the dashboard's ``src`` on ``sys.path``:

    PYTHONPATH=<dashboard>/src <pepy-venv>/bin/python allowance_cost.py \
        --dataset <national.h5> --label candidate --output <receipt.json>
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

YEARS = (2027, 2028, 2029, 2030, 2031)
DESIGNS = {
    "sole_income": {
        "other_income_limit": 0,
        "total_income_at_most_allowance": False,
        "disregard_savings_interest": True,
    },
    "total_income": {},
}
DIRECT_EFFECT_VARIABLES = (
    "income_tax",
    "national_insurance",
    "student_loan_repayment",
    "employment_income",
)
DIRECT_EFFECT_MIN_GBP = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _simulation(dataset: Path):
    import policyengine as pe

    return pe.uk.managed_microsimulation(dataset=str(dataset), allow_unmanaged=True)


def _person(sim, variable: str, year: int) -> np.ndarray:
    return np.asarray(
        sim.calculate(variable, period=year, map_to="person").values, dtype=float
    )


def _baseline(sim) -> dict[int, dict[str, object]]:
    out = {}
    for year in YEARS:
        out[year] = {
            "gov_balance": float(sim.calculate("gov_balance", period=year).sum()),
            "person": {
                name: _person(sim, name, year)
                for name in (
                    *DIRECT_EFFECT_VARIABLES,
                    "person_weight",
                    "household_income_decile",
                    "is_SP_age",
                    "state_pension",
                )
            },
        }
    return out


def _design(dataset: Path, baseline, arguments) -> list[dict[str, float]]:
    from uk_budget_data.budget_measures.state_pension_only_income_tax_allowance import (
        state_pension_only_income_tax_allowance,
    )

    sim = state_pension_only_income_tax_allowance(_simulation(dataset), **arguments)
    rows = []
    for year in YEARS:
        base = baseline[year]
        person = base["person"]
        own_change = np.zeros_like(person["person_weight"])
        for name in DIRECT_EFFECT_VARIABLES:
            own_change = np.maximum(
                own_change, np.abs(_person(sim, name, year) - person[name])
            )
        direct = own_change >= DIRECT_EFFECT_MIN_GBP
        weight = person["person_weight"]
        in_decile = person["household_income_decile"] >= 1
        tax_cut = person["income_tax"] - _person(sim, "income_tax", year)
        cost = float(sim.calculate("gov_balance", period=year).sum()) - float(
            base["gov_balance"]
        )
        affected = float(weight[direct & in_decile].sum())
        rows.append(
            {
                "year": year,
                "fiscal_year": f"{year}-{str(year + 1)[-2:]}",
                "gov_balance_change_gbp": cost,
                "people_directly_affected": affected,
                "people_directly_affected_any_decile": float(weight[direct].sum()),
                "income_tax_cut_gbp": float((weight * tax_cut).sum()),
                "mean_gain_per_affected_gbp": (
                    float((weight * tax_cut)[direct & in_decile].sum()) / affected
                    if affected
                    else 0.0
                ),
                "state_pension_age_recipients": float(
                    weight[
                        (person["is_SP_age"] > 0) & (person["state_pension"] > 0)
                    ].sum()
                ),
            }
        )
    del sim
    gc.collect()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dashboard-commit", required=True)
    args = parser.parse_args()
    sim = _simulation(args.dataset)
    baseline = _baseline(sim)
    del sim
    gc.collect()
    receipt = {
        "label": args.label,
        "dataset": args.dataset.name,
        "dataset_sha256": _sha256(args.dataset),
        "dashboard_commit": args.dashboard_commit,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("policyengine", "policyengine-uk", "policyengine-core")
        },
        "designs": {
            name: {
                "arguments": arguments,
                "years": _design(args.dataset, baseline, arguments),
            }
            for name, arguments in DESIGNS.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    for name, design in receipt["designs"].items():
        for row in design["years"]:
            print(
                name,
                row["fiscal_year"],
                f"{row['gov_balance_change_gbp'] / 1e6:,.0f}m",
                f"{row['people_directly_affected']:,.0f}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
