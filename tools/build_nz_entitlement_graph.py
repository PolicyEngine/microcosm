#!/usr/bin/env python3
"""Generate NZ's post-calibration entitlement branch declarations.

The bridge evaluations preserve the committed reg 17/18 recipes. No kernel
currently assembles their nonlinear expressions, so the graph deliberately
leaves the two final bridge inputs without a producer. The hub must close
that conflict before activation; this generator never substitutes a schedule.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NZ = ROOT / "packages/microcosm-build/src/microcosm/build/nz"
AS_BINDING = "accommodation_supplement"
BRIDGE_COLUMNS = (
    "accommodation_supplement_base_rate_weekly_amount",
    "accommodation_supplement_non_beneficiary_income_cutout_weekly_amount",
)
PAYMENT = "accommodation_supplement_rounded_weekly_payment"
POSITIVE = "accommodation_supplement_positive_payment_indicator"
ENTITIES = ["person", "household", "family"]


def read(nz, name):
    return json.loads((nz / f"{name}.json").read_text())


def select(resource, encoding="json", *path):
    result = {"resource": resource, "encoding": encoding}
    if path:
        result["path"] = list(path)
    return result


def slices(entity, columns):
    from build_nz_transport_graph import slice_

    return slice_(entity, sorted(set(columns)))


def owned(entity, column, dtype="float64", *, rewrite=False):
    result = {"entity": entity, "column": column, "dtype": dtype}
    if rewrite:
        result["rewrite"] = True
    return result


def node(id, kernel, population=None, **fields):
    result = {"id": id, "kernel": kernel, "structural": "none"}
    if population:
        result["population"] = population
    return {**result, **fields}


def boundary(id, base):
    return node(
        id,
        "transport.boundary@1",
        structural="filter",
        base=base,
        inputs=[slices("person", ["age"])],
        params={},
    )


def binding(name, field="engine_ref"):
    return {"binding": name, "field": field}


def _overrides(rows):
    result = []
    for row in rows:
        item = {"entity": "person", "column": row["variable"]}
        if "unit_attribute" in row:
            entity, _, column = row["unit_attribute"].partition(".")
            item.update(source_entity=entity, source_column=column)
        else:
            item["value"] = row["value"]
        result.append(item)
    return result


def bridge_evaluations(nz, population, prefix):
    """Expose the declared engine calls while leaving final assembly pending."""
    bridge = read(nz, "as_rate_bridge")
    bindings = {row["id"]: row for row in read(nz, "axiom_rules_bindings")["bindings"]}
    closure = read(nz, "axiom_input_closure")
    rate_path = bindings["main_benefits_rates"]["rulespec_path"]
    rate_inputs = [
        entry["input"]
        for entry in closure["entries"]
        if entry["rulespec_path"] == rate_path
    ]
    family_inputs = [
        "n_dependent_children",
        "is_partnered",
        "is_single",
        "is_sole_parent",
        "youngest_dependent_child_age",
    ]
    result = []
    evaluations = []
    reg17, reg18 = bridge["outputs"]
    for index, term in enumerate(reg17["terms"]):
        evaluations.append((f"reg17_{index}", term))
    for branch in reg18["branches"]:
        for evaluation in branch["evaluations"]:
            evaluations.append((f"reg18_{branch['id']}_{evaluation['id']}", evaluation))
    for label, evaluation in evaluations:
        name = evaluation["binding"]
        entity = bindings[name]["entity"]
        inputs = [slices("person", rate_inputs), slices("family", family_inputs)]
        overrides = _overrides(evaluation["input_overrides"])
        if entity == "family":
            for override in overrides:
                override["entity"] = "family"
        variable = evaluation["variable"]
        column = f"bridge_evaluation_{label}"
        result.append(
            node(
                f"{prefix}.bridge.evaluate.{label}",
                "simulate.counterfactual@1",
                population,
                sources=["rulespec_nz"],
                inputs=inputs,
                outputs=[owned(entity, column)],
                params={
                    "engine_ref": binding(name),
                    "period": binding(name, "period"),
                    "variables": [variable],
                    "input_overrides": overrides,
                    "output_coefficients": [
                        {
                            "variable": variable,
                            "coefficient": 1,
                            "entity": entity,
                            "column": column,
                        }
                    ],
                    "resource_sha256": select("as_rate_bridge", "sha256"),
                },
                description="The engine evaluates this declared bridge component; the hub still must supply the approved final assembly.",
            )
        )
    return result


def group_node(nz, population, prefix, *, rewrite=False, scenario=True):
    skeleton = read(nz, "transport_graph")
    baseline = next(row for row in skeleton["nodes"] if row["id"] == "nz.as.inputs")
    result = copy.deepcopy(baseline)
    result.update(id=f"{prefix}.inputs", population=population)
    result["kernel"] = (
        "transport.scenario_override@1" if rewrite else "concepts.encode_groups@1"
    )
    for output in result["outputs"]:
        if rewrite:
            output["rewrite"] = True
        else:
            output.pop("rewrite", None)
    if scenario:
        result["params"]["knobs"] = {"scenario": "encoding_knobs", "encoding": "json"}
        result["params"]["scenario_sha256"] = {
            "scenario": "encoding_knobs",
            "encoding": "sha256",
        }
    return result


def reporting_nodes(nz, population, prefix, scenario, *, baseline=False):
    classifiers = read(nz, "reporting_classifiers")
    income = classifiers["income_decile"]
    income_columns = {
        component["column"]
        for component in income["components"]
        if component["entity"] == "person"
    }
    income_columns.update(
        component["receipt_column"]
        for component in income["components"]
        if "receipt_column" in component
    )
    result = [
        node(
            f"{prefix}.income_decile",
            "reporting.income_decile@1"
            if baseline
            else "reporting.carry_income_decile@1",
            population,
            inputs=[
                slices("person", ["age", *income_columns] if baseline else ["age"]),
                slices("household", ["source_household_id"]),
                slices("family", ["family_type"]),
            ],
            outputs=[owned("family", "income_decile", "int64")],
            params={
                "classifier": select("reporting_classifiers", "json", "income_decile"),
                "classifier_sha256": select("reporting_classifiers", "sha256"),
                "scenario": scenario,
                "entities": ENTITIES,
            },
        )
    ]
    if not baseline:
        result[0]["artifact_inputs"] = [
            {
                "name": "baseline_cells",
                "producer": "nz.scn.S0.income_decile",
                "artifact": "baseline_cells",
            }
        ]
    flags = {
        column
        for predicate in classifiers["receipt_group"]["predicates"].values()
        for column in predicate["columns"]
    }
    result.append(
        node(
            f"{prefix}.receipt_group",
            "reporting.receipt_group@1",
            population,
            inputs=[slices("person", flags), slices("family", ["family_type"])],
            outputs=[owned("family", "receipt_group", "string")],
            params={
                "classifier": select("reporting_classifiers", "json", "receipt_group"),
                "classifier_sha256": select("reporting_classifiers", "sha256"),
                "scenario": scenario,
            },
        )
    )
    return result


def as_nodes(nz, population, prefix, scenario, *, baseline=False):
    bindings = {row["id"]: row for row in read(nz, "axiom_rules_bindings")["bindings"]}
    closure = read(nz, "axiom_input_closure")
    as_inputs = [
        entry["input"]
        for entry in closure["entries"]
        if entry["rulespec_path"] == bindings[AS_BINDING]["rulespec_path"]
    ]
    variables = bindings[AS_BINDING]["variables"]
    judgment = {
        "accommodation_supplement_non_rate_entitlement_requirements",
        "accommodation_supplement_cash_assets_within_limit",
        "accommodation_supplement_social_housing_exclusion_not_applies",
        "accommodation_supplement_other_funding_exclusion_not_applies",
    }
    return [
        *bridge_evaluations(nz, population, prefix),
        node(
            f"{prefix}.as",
            "simulate.rules_by_ref@1",
            population,
            sources=["rulespec_nz"],
            inputs=[slices("person", ["age"]), slices("family", as_inputs)],
            outputs=[
                owned(
                    "family", variable, "int64" if variable in judgment else "float64"
                )
                for variable in variables
            ],
            rules_binding=AS_BINDING,
            params={"variables": variables},
        ),
        *reporting_nodes(nz, population, prefix, scenario, baseline=baseline),
        node(
            f"{prefix}.positive_payment",
            "calibration.prepare@1",
            population,
            inputs=[slices("family", [PAYMENT])],
            outputs=[owned("family", POSITIVE, "bool")],
            params={
                "bindings": select("holdout_comparator_bindings"),
                "references": select("holdout_references"),
                "references_sha256": select("holdout_references", "sha256"),
            },
        ),
        node(
            f"{prefix}.gap",
            "takeup.gap@1",
            population,
            sources=["nz_holdout_facts"],
            inputs=[
                slices("person", ["age"]),
                slices("household", ["source_household_id"]),
                slices(
                    "family",
                    [
                        PAYMENT,
                        POSITIVE,
                        "family_type",
                        "income_decile",
                        "receipt_group",
                    ],
                ),
            ],
            params={
                "country": "nz",
                "references": select("holdout_references"),
                "references_sha256": select("holdout_references", "sha256"),
                "entities": ENTITIES,
                "weight_entity": "household",
                "annualisation_factors": select("gap_annualisation"),
                "annualisation_sha256": select("gap_annualisation", "sha256"),
                "scenario": scenario,
            },
        ),
    ]


def documents(nz=NZ):
    from build_nz_transport_graph import calibration_nodes

    scenarios = read(nz, "scenarios")["scenarios"]
    entitlement = [
        boundary("nz.scn.{scenario}", "nz.as"),
        *as_nodes(nz, "nz.scn.{scenario}", "nz.scn.{scenario}", "{scenario}"),
    ]
    for scenario in scenarios:
        if scenario["tier"] == "entitlement":
            prefix = f"nz.scn.{scenario['id']}"
            rows = [boundary(prefix, "nz.as")]
            if scenario["id"] in {"S1", "S2", "S3"}:
                rows.append(group_node(nz, prefix, prefix, rewrite=True))
            rows.extend(
                as_nodes(
                    nz, prefix, prefix, scenario["id"], baseline=scenario["id"] == "S0"
                )
            )
        else:
            prefix = f"nz.v.{scenario['id']}"
            resource = (
                "money_normalization_hold_fixed"
                if scenario["id"] == "VHF"
                else "money_normalization"
            )
            rows = [
                boundary(prefix, "nz.open"),
                *calibration_nodes(prefix, prefix, money_resource=resource),
            ]
            for row in rows:
                if row["kernel"] == "money.normalize@1":
                    row["params"]["profile_selection"] = {
                        "scenario": "knobs",
                        "encoding": "json",
                    }
            calibrated = next(
                row["id"] for row in rows if row["kernel"] == "calibrate.ordered_adam@1"
            )
            population = prefix + ".housing"
            rows.extend(
                [
                    boundary(population, calibrated),
                    group_node(nz, population, prefix),
                    *as_nodes(nz, population, prefix, scenario["id"]),
                ]
            )
        scenario["nodes"] = rows
    return {
        "entitlement_graph": {
            "schema_version": 1,
            "nodes": [],
            "scenario_nodes": entitlement,
            "scenario_gap": "nz.scn.{scenario}.gap",
            "variant_gap": "nz.v.{variant}.gap",
            "bands": {
                "id": "nz.as.bands",
                "population": "nz.terminal",
                "baseline_scenario": "S0",
            },
            "package": {"id": "nz.package", "artifact_name": "bands"},
            "checkpoints": ["nz.as.bands"],
        },
        "gap_annualisation": {
            "msd_accommodation_supplement_recipients_national": 1,
            "msd_annual_report_accommodation_assistance_expenditure": 52,
        },
        "holdout_comparator_bindings": {
            "schema_version": 1,
            "required_inputs": {"family": [PAYMENT]},
            "input_domains": {"family": {PAYMENT: {"kind": "nonnegative_number"}}},
            "columns": [
                {
                    "entity": "family",
                    "name": POSITIVE,
                    "predicates": [{"column": PAYMENT, "operator": "gt", "value": 0}],
                }
            ],
        },
        "scenarios": {**read(nz, "scenarios"), "scenarios": scenarios},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    differs = []
    for name, document in documents().items():
        path = NZ / f"{name}.json"
        text = json.dumps(document, indent=2) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != text:
                differs.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(text)
    if differs:
        raise SystemExit(
            "Generated NZ entitlement resources differ: " + ", ".join(differs)
        )
    print(
        "NZ entitlement resources match."
        if args.check
        else "Wrote NZ entitlement resources."
    )


if __name__ == "__main__":
    main()
