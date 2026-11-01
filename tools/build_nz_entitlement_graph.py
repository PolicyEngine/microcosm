#!/usr/bin/env python3
"""Generate NZ's post-calibration entitlement branch declarations.

The Accommodation Supplement bridge follows ``as_rate_bridge.json``. On each
population that runs the AS engine, a ``calibration.prepare@1`` node writes
the family masks, a ``simulate.counterfactual@1`` node writes the reg 18(3)(b)
sole-parent gross rate and a ``simulate.solve_zero@1`` node writes the reg 18
cutout. The shared ``nz.as.bridge`` population, a FILTER branch of ``nz.as``,
runs these once for every entitlement scenario, and each variant repeats them
on its own housing population. An extension node on ``nz.as`` itself would
re-key the skeleton's ``nz.terminal``, which composition refuses. Each
branch runs its own reg 17 base-rate counterfactual. Every coefficient, mask
and override comes from the resource; this generator adds no schedule.
"""

from __future__ import annotations

import argparse
import copy
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NZ = ROOT / "packages/microcosm-build/src/microcosm/build/nz"
AS_BINDING = "accommodation_supplement"
PAYMENT = "accommodation_supplement_rounded_weekly_payment"
POSITIVE = "accommodation_supplement_positive_payment_indicator"
ENTITIES = ["person", "household", "family"]
#: The FILTER branch of nz.as on which the shared bridge nodes run.
SHARED_BRIDGE = "nz.as.bridge"


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


def _unit(attribute):
    entity, _, column = attribute.partition(".")
    return entity, column


def _mask(row):
    entity, column = _unit(row["unit_attribute"])
    return {"entity": entity, "column": column, "equals": row["equals"]}


def _overrides(rows, entity="person"):
    """Translate the resource's override rows into kernel override records."""
    result = []
    for row in rows:
        item = {"entity": entity, "column": row["variable"]}
        if "unit_attribute" in row:
            source_entity, column = _unit(row["unit_attribute"])
            item.update(source_entity=source_entity, source_column=column)
        else:
            item["value"] = row["value"]
        if "mask" in row:
            item["mask"] = _mask(row["mask"])
        result.append(item)
    return result


def _coefficient(value):
    return float(Fraction(value)) if isinstance(value, str) else value


def _term(row, target, primary, bindings):
    """One output coefficient on the family target column."""
    if "source" in row:
        entity, column = _unit(row["source"])
        return {
            "source_entity": entity,
            "source_column": column,
            "coefficient": _coefficient(row["coefficient"]),
            "entity": "family",
            "column": target,
        }
    term = {
        "variable": row["variable"],
        "coefficient": _coefficient(row["coefficient"]),
        "entity": "family",
        "column": target,
    }
    if row["binding"] != primary:
        term["engine_ref"] = binding(row["binding"])
    if bindings[row["binding"]]["entity"] == "person":
        term["aggregation"] = row["aggregation"]
    if "mask" in row:
        term["mask"] = _mask(row["mask"])
    return term


def _bridge_inputs(nz, *extra):
    bindings = {row["id"]: row for row in read(nz, "axiom_rules_bindings")["bindings"]}
    closure = read(nz, "axiom_input_closure")
    rate_path = bindings["main_benefits_rates"]["rulespec_path"]
    rate_inputs = [
        entry["input"]
        for entry in closure["entries"]
        if entry["rulespec_path"] == rate_path and entry["entity"] == "Person"
    ]
    family = [
        "n_dependent_children",
        "is_partnered",
        "is_single",
        "is_sole_parent",
        "youngest_dependent_child_age",
        *(
            column["name"]
            for column in read(nz, "as_rate_bridge")["mask_bindings"]["columns"]
        ),
        *extra,
    ]
    return [slices("person", rate_inputs), slices("family", family)]


def bridge_masks(nz, population, prefix):
    """The family booleans the equality-only bridge masks compare."""
    bridge = read(nz, "as_rate_bridge")
    return node(
        f"{prefix}.bridge.masks",
        "calibration.prepare@1",
        population,
        inputs=[
            slices(entity, columns)
            for entity, columns in bridge["mask_bindings"]["required_inputs"].items()
        ],
        outputs=[
            owned(column["entity"], column["name"], "bool")
            for column in bridge["mask_bindings"]["columns"]
        ],
        params={
            "bindings": select("as_rate_bridge", "json", "mask_bindings"),
            "references": select("holdout_references"),
            "references_sha256": select("holdout_references", "sha256"),
        },
    )


def _counterfactual_params(primary, overrides, terms, components=()):
    params = {
        "engine_ref": binding(primary),
        "period": binding(primary, "period"),
        "variables": list(
            dict.fromkeys(
                term["variable"]
                for term in terms
                if "variable" in term and "engine_ref" not in term
            )
        ),
        "input_overrides": overrides,
        "output_coefficients": terms,
    }
    if components:
        params["components"] = list(components)
    params["resource_sha256"] = select("as_rate_bridge", "sha256")
    return params


def cutout_nodes(nz, population, prefix):
    """Reg 18: the sole-parent gross rate, then one zero-solve per population."""
    bridge = read(nz, "as_rate_bridge")
    bindings = {row["id"]: row for row in read(nz, "axiom_rules_bindings")["bindings"]}
    reg18 = bridge["outputs"][1]
    gross = reg18["gross_rate"]
    gross_term = {
        "variable": gross["variable"],
        "coefficient": gross["coefficient"],
        "entity": "family",
        "column": gross["column"],
        "aggregation": gross["aggregation"],
        "mask": _mask(gross["mask"]),
    }
    solve = reg18["solve"]
    target = reg18["target_input"]
    terms = [_term(row, target, solve["binding"], bindings) for row in solve["terms"]]
    params = _counterfactual_params(
        solve["binding"], _overrides(solve["input_overrides"]), terms
    )
    resource = params.pop("resource_sha256")
    params.update(
        solve_input=solve["solve_input"],
        bracket=solve["bracket"],
        tolerance=solve["tolerance"],
        iterations=solve["iterations"],
        resource_sha256=resource,
    )
    return [
        node(
            f"{prefix}.bridge.{gross['id']}",
            gross["kernel"],
            population,
            sources=["rulespec_nz"],
            inputs=_bridge_inputs(nz),
            outputs=[owned("family", gross["column"])],
            params=_counterfactual_params(
                gross["binding"], _overrides(gross["input_overrides"]), [gross_term]
            ),
            description="The engine evaluates the reg 18(3)(b) sole-parent Jobseeker Support rate for sole parents.",
        ),
        node(
            f"{prefix}.bridge.cutout",
            reg18["kernel"],
            population,
            sources=["rulespec_nz"],
            inputs=_bridge_inputs(nz, gross["column"]),
            outputs=[owned("family", target)],
            params=params,
            description="The engine's Jobseeker Support residual reaches zero at the reg 18 non-beneficiary income cutout.",
        ),
    ]


def base_rate_node(nz, population, prefix):
    """Reg 17: one counterfactual over the main-benefit and tax-credit modules."""
    bridge = read(nz, "as_rate_bridge")
    bindings = {row["id"]: row for row in read(nz, "axiom_rules_bindings")["bindings"]}
    reg17 = bridge["outputs"][0]
    primary = reg17["primary_binding"]
    target = reg17["target_input"]
    components = []
    for component in reg17["components"]:
        name = component["binding"]
        components.append(
            {
                "engine_ref": binding(name),
                "period": binding(name, "period"),
                "variables": [
                    term["variable"]
                    for term in reg17["terms"]
                    if term["binding"] == name
                ],
                "input_overrides": _overrides(
                    component["input_overrides"], bindings[name]["entity"]
                ),
            }
        )
    terms = [_term(row, target, primary, bindings) for row in reg17["terms"]]
    return node(
        f"{prefix}.bridge.base_rate",
        reg17["kernel"],
        population,
        sources=["rulespec_nz"],
        inputs=_bridge_inputs(nz),
        outputs=[owned("family", target)],
        params=_counterfactual_params(
            primary, _overrides(reg17["input_overrides"]), terms, components
        ),
        description="The engine evaluates the reg 17 base rate from zero-income Jobseeker Support and the eldest-child family tax credit.",
    )


def shared_bridge_nodes(nz, population, prefix):
    """Masks, the sole-parent gross rate and the cutout, once per population."""
    return [bridge_masks(nz, population, prefix), *cutout_nodes(nz, population, prefix)]


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
        base_rate_node(nz, population, prefix),
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
    # An extension node on nz.as would re-key the skeleton's nz.terminal
    # FILTER, so the shared bridge runs on a FILTER branch of nz.as that every
    # entitlement scenario descends from.
    shared = [
        boundary(SHARED_BRIDGE, "nz.as"),
        *shared_bridge_nodes(nz, SHARED_BRIDGE, "nz.as"),
    ]
    entitlement = [
        boundary("nz.scn.{scenario}", SHARED_BRIDGE),
        *as_nodes(nz, "nz.scn.{scenario}", "nz.scn.{scenario}", "{scenario}"),
    ]
    for scenario in scenarios:
        if scenario["tier"] == "entitlement":
            prefix = f"nz.scn.{scenario['id']}"
            rows = [boundary(prefix, SHARED_BRIDGE)]
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
                    *shared_bridge_nodes(nz, population, prefix),
                    *as_nodes(nz, population, prefix, scenario["id"]),
                ]
            )
        scenario["nodes"] = rows
    return {
        "entitlement_graph": {
            "schema_version": 1,
            "nodes": shared,
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
