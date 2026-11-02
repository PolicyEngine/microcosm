#!/usr/bin/env python3
"""Generate NZ's reusable pre-calibration gross-up node declarations.

This generates graph data only. The pinned rules engine evaluates the tax
schedule. The packaged NZ transport graph remains a separate wiring task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NZ = ROOT / "packages/microcosm-build/src/microcosm/build/nz"
PROGRAMMES = (
    "jobseeker_support",
    "sole_parent_support",
    "supported_living_payment",
)
DISCLOSURE = (
    "Annualised Schedule 1, before credits, 52-week benefit gross-up, "
    "each benefit inverted independently of other income."
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _owned(column, dtype="float64"):
    return {"entity": "person", "column": column, "dtype": dtype}


def _slice(columns):
    return {"entity": "person", "columns": columns}


def _selector(field):
    return {"binding": "income_tax_schedule_1", "field": field}


def documents(nz=NZ):
    bindings_document = json.loads((nz / "axiom_rules_bindings.json").read_text())
    bindings = {row["id"]: row for row in bindings_document["bindings"]}
    rulespec = bindings_document["rulespec"]
    rulespec_url = (
        f"https://github.com/{rulespec['repository']}/blob/{rulespec['commit']}"
    )
    msd_url = json.loads((nz / "msd_benefit_rates_20260401.json").read_text())[
        "source"
    ]["url"]
    closure = json.loads((nz / "axiom_input_closure.json").read_text())
    scenarios = json.loads((nz / "scenarios.json").read_text())
    weeks = scenarios["gap_conventions"]["annualisation"]["weeks_per_year"]
    tax = bindings["income_tax_schedule_1"]
    nets = [f"{programme}_net_weekly_payment" for programme in PROGRAMMES]
    gross = [f"{programme}_annual_gross" for programme in PROGRAMMES]
    cents = [f"{column}_cents" for column in gross]
    rate_inputs = [
        entry["input"]
        for entry in closure["entries"]
        if entry["rulespec_path"] == bindings["main_benefits_rates"]["rulespec_path"]
    ]
    nodes = [
        {
            "id": "nz.benefits.rates",
            "kernel": "simulate.rules_by_ref@1",
            "population": "nz.open",
            "structural": "none",
            "rules_binding": "main_benefits_rates",
            "inputs": [_slice(rate_inputs)],
            "outputs": [_owned(column) for column in nets],
            "params": {"variables": nets},
        }
    ]
    for programme, net, annual in zip(PROGRAMMES, nets, gross, strict=True):
        terms = [
            {
                "source_entity": "person",
                "source_column": net,
                "coefficient": weeks,
                "entity": "person",
                "column": annual,
            },
            {
                "variable": tax["variables"][0],
                "coefficient": 1,
                "entity": "person",
                "column": annual,
            },
            {"trial": True, "coefficient": -1, "entity": "person", "column": annual},
        ]
        nodes.append(
            {
                "id": f"nz.gross_up.{programme}",
                "kernel": "simulate.solve_zero@1",
                "population": "nz.open",
                "structural": "none",
                "inputs": [_slice([net])],
                "outputs": [_owned(annual)],
                "params": {
                    "engine_ref": _selector("engine_ref"),
                    "period": _selector("period"),
                    "variables": tax["variables"],
                    "input_overrides": "[]",
                    "output_coefficients": _json(terms),
                    "solve_input": _json(
                        {"entity": "person", "column": "taxable_income"}
                    ),
                    "bracket": [0, 1000000],
                    "tolerance": 0.0000001,
                    "iterations": 48,
                    "quantum": "0.01",
                    "tie_rule": "half_even",
                    "certificate_margin": "0.000001",
                    "quantized_residual_bound": "0.0050001",
                    "extra_evaluation_budget": 9,
                },
            }
        )
    minor_units = {
        "entity": "person",
        "columns": [
            {
                "source_column": annual,
                "output_column": minor,
                "quantum": "0.01",
                "rounding": "exact",
            }
            for annual, minor in zip(gross, cents, strict=True)
        ],
    }
    nzs_source = "nz_super_weekly_amount_before_tax"
    nzs_cents = f"{nzs_source}_cents"
    nodes.extend(
        [
            {
                "id": "nz.gross_up.annual_cents",
                "kernel": "money.minor_units@1",
                "population": "nz.open",
                "structural": "none",
                "inputs": [_slice(gross)],
                "outputs": [_owned(column, "int64") for column in cents],
                "params": {
                    "engine_ref": _selector("engine_ref"),
                    "period": _selector("period"),
                    "declaration": _json(minor_units),
                },
            },
            {
                "id": "nz.gross_up.super_cents",
                "kernel": "money.minor_units@1",
                "population": "nz.open",
                "structural": "none",
                "inputs": [_slice([nzs_source])],
                "outputs": [_owned(nzs_cents, "int64")],
                "params": {
                    "declaration": _json(
                        {
                            "entity": "person",
                            "columns": [
                                {
                                    "source_column": nzs_source,
                                    "output_column": nzs_cents,
                                    "quantum": "0.01",
                                    "rounding": "half_even",
                                    "source_tax_basis": "before_tax",
                                }
                            ],
                        }
                    )
                },
            },
        ]
    )
    programmes = [*PROGRAMMES, "nz_super"]
    flags = [f"receives_{programme}" for programme in programmes]
    transfer_columns = [
        f"{programme}_annual_gross_transfer_cents" for programme in programmes
    ]
    transfers = {
        "entity": "person",
        "weeks_per_year": weeks,
        "exposure": {"central": 1, "convention": "full_year_stock_equivalent"},
        "transfers": [
            {
                "flag_column": flag,
                "source_column": source,
                "source_basis": "annual_gross_cents",
                "output_column": output,
            }
            for flag, source, output in zip(
                flags[:3], cents, transfer_columns[:3], strict=True
            )
        ]
        + [
            {
                "flag_column": flags[-1],
                "source_column": nzs_cents,
                "source_basis": "weekly_before_tax_cents",
                "source_tax_basis": "before_tax",
                "output_column": transfer_columns[-1],
            }
        ],
    }
    nodes.append(
        {
            "id": "nz.benefits.annual_transfers",
            "kernel": "money.annual_transfers@1",
            "population": "nz.open",
            "structural": "none",
            "inputs": [_slice([*flags, *cents, nzs_cents])],
            "outputs": [_owned(column, "int64") for column in transfer_columns],
            "params": {"declaration": _json(transfers)},
        }
    )
    gross_up = {
        "schema_version": 1,
        "country": "nz",
        "generator": "tools/build_nz_benefit_gross_up.py",
        "status": "reusable graph declarations; packaged transport graph wiring remains pending",
        "disclosure": DISCLOSURE,
        "monotonicity": "For the pinned Schedule 1, every marginal tax rate falls below one; 52*n + T(G) - G decreases or stays constant as each person's own G increases.",
        "authoritative_value": "money.minor_units@1 recovers annual integer cents exactly from certified float64 quantum multiples and preserves the bridge's float64 output contract.",
        "certificate_contract": "certificate_margin bounds the absolute difference between each computed residual and the exact residual of the declared terms at every evaluated point. Boundary residuals must exceed the margin in magnitude; ambiguous boundaries refuse, and the residual bound includes the margin. A zero margin asserts exact outputs and permits half-even ties.",
        "weekly_gross_reporting": {
            "source": "annual_gross_cents",
            "divisor": weeks,
            "tie_rule": "half_even",
            "reporting_only": True,
            "persist": False,
        },
        "nodes": nodes,
    }
    stages = {
        "schema_version": 1,
        "stages": [
            {
                "id": "nz.precal.main_benefit_rates",
                "rules_binding": "main_benefits_rates",
                "rulespec_path": bindings["main_benefits_rates"]["rulespec_path"],
                "outputs": [
                    {
                        "entity": "person",
                        "column": column,
                        "purpose": "calibration_measure",
                    }
                    for column in nets
                ],
                "citation": (
                    f"{rulespec_url}/{bindings['main_benefits_rates']['rulespec_path']} "
                    f"(SHA-256 {bindings['main_benefits_rates']['sha256']}); "
                    f"{msd_url}. The rates stage supplies after-tax payments after abatement."
                ),
            },
            {
                "id": "nz.precal.benefit_gross_up",
                "rules_binding": "income_tax_schedule_1",
                "rulespec_path": tax["rulespec_path"],
                "outputs": [],
                "derived_outputs": [
                    {
                        "entity": "person",
                        "column": column,
                        "purpose": "calibration_measure",
                    }
                    for column in [*gross, *cents]
                ],
                "citation": (
                    f"{rulespec_url}/{tax['rulespec_path']} (SHA-256 {tax['sha256']}); "
                    f"{msd_url}. Each gross-up inverts annualised Schedule 1 before credits "
                    "over 52 weeks independently of other income and recovers authoritative annual cents."
                ),
            },
        ],
    }
    return {"benefit_gross_up.json": gross_up, "precalibration_stages.json": stages}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    for name, document in documents().items():
        path = NZ / name
        text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise SystemExit(f"Generated resource is stale: {path}")
        else:
            path.write_text(text)
    print("NZ benefit gross-up resources match the generator.")


if __name__ == "__main__":
    main()
