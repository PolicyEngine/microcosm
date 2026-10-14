"""Synthetic entitlement graph over invented donor support and rules.

The numbers in ``TOY_PROGRAMS`` and ``TOY_DECLARATIONS`` are test data, never
statutory rates or destination microdata. The adapters read their complete
programs from the pinned fixture tree on every call. Scenario knobs are
separate selected subtrees, so a scenario edit cannot change sibling keys.
"""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from microcosm.build.transport.cli import EXPORT_FILENAME, through
from microcosm.build.transport.compose import compose_transport_graph
from microcosm.build.transport.registry import build_transport_registry
from microcosm.build.transport.terminal_kernels import materialize_export
from microcosm.frame import Frame, VariableMetadata
from microcosm.graph import ContentStore, compile_graph, run_graph
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.transport_composed import (
    COUNTRY,
    PERIOD,
    ComposedFixture,
    ToyRulesEngine,
    _edge,
    _fact,
    _node,
    _owned,
    _references,
    _select,
    _slice,
    make_composed_fixture,
)
from test_support.microcosm_build.transport_graph import reference_row, write_facts
from test_support.microcosm_build.transport_population import UNIT_RULE

INPUT_FIELDS = (
    "as_assets",
    "as_costs",
    "as_income",
    "as_beneficiary",
    "as_super",
    "as_s66",
    "as_s67",
    "as_maximum",
    "as_credit_units",
)
AS_INPUTS = (*INPUT_FIELDS, "as_base_rate", "as_cutout")
AS_OUTPUTS = {
    "toy_as_weekly": "float64",
    "toy_as_unit_count": "int64",
    "toy_as_eligible": "int64",
    "toy_as_single": "int64",
    "toy_as_other": "int64",
    "toy_as_excluded": "int64",
}

# Invented fixture programs. Every output-affecting amount is serialized into
# the RuleSpec source tree, so these examples exercise content-bound execution.
TOY_PROGRAMS = {
    "entitlement": {
        "path": "toy/as.yaml",
        "entity": "family",
        "kind": "as",
        "policy": {
            "asset_limit": 350.0,
            "cost_share": 0.25,
            "payment_share": 0.7,
            "income_floor": 10.0,
            "income_abatement": 0.2,
            "family_type": "single",
        },
        "defaults": {
            "as_assets": {"source": "input_assets", "value": 200.0},
            "as_costs": {"source": "input_rent", "value": 300.0},
            "as_income": {"value": 80.0},
            "as_beneficiary": {"value": 0.0},
            "as_super": {"value": 0.0},
            "as_s66": {"value": 0.0},
            "as_s67": {"value": 0.0},
            "as_maximum": {"value": 90.0},
            "as_credit_units": {"value": 1.0},
        },
        "outputs": {
            **{f"seed_{name}": {"dtype": "float64"} for name in INPUT_FIELDS},
            **{name: {"dtype": dtype} for name, dtype in AS_OUTPUTS.items()},
        },
    },
    "bridge_rates": {
        "path": "toy/rates.yaml",
        "entity": "person",
        "kind": "rates",
        "policy": {"rate": 40.0, "alternative_rate": 60.0, "abatement": 0.25},
        "outputs": {"toy_bridge_benefit": {"dtype": "float64"}},
    },
    "bridge_credit": {
        "path": "toy/credits.yaml",
        "entity": "family",
        "kind": "credit",
        "policy": {"credit": 16.0},
        "outputs": {"toy_bridge_credit": {"dtype": "float64"}},
    },
}
TOY_DECLARATIONS = {
    "base_credit_coefficient": 0.25,
    # The invented cutout 160 is exact on this dyadic bisection grid. Eight
    # fixed iterations preserve the graph semantics; core solver tests cover
    # general brackets and tighter accuracy separately.
    "solve_bracket": [0.0, 256.0],
    "solve_tolerance": 1.0,
    "solve_iterations": 8,
    "annualisation": 4.0,
    "target_factors": {"V1": 0.95, "V2": 1.05},
    "alternative_receipt_rate": 0.2,
}


class EntitlementToyEngine(ToyRulesEngine):
    """AS-shaped toy arithmetic using only the public ``RulesEngine`` protocol."""

    def cache_identity(self) -> Mapping:
        return {"formula": "synthetic-as-and-bridge-v1"}

    def variable_metadata(self, name: str) -> VariableMetadata:
        program = self.program()
        dtype = program["outputs"][name]["dtype"]
        return VariableMetadata(
            name,
            program["entity"],
            "int" if dtype == "int64" else "float",
            "year",
        )

    def variables(self) -> Sequence[str]:
        kind = self.program()["kind"]
        if kind == "rates":
            return ("input_income", "bridge_alternative")
        if kind == "credit":
            return ("as_credit_units",)
        return (*AS_INPUTS, "input_assets", "input_rent")

    def materialize(self, bundle: Frame, variables, period):
        self.calls.append((tuple(variables), period))
        program = self.program()
        table = bundle.table(program["entity"])
        policy = program["policy"]

        def column(name, fallback=0):
            if name in table:
                return table[name].to_numpy(dtype=np.float64)
            return np.full(len(table), fallback, dtype=np.float64)

        if program["kind"] == "rates":
            rate = np.where(
                column("bridge_alternative") != 0,
                policy["alternative_rate"],
                policy["rate"],
            )
            result = {
                "toy_bridge_benefit": np.maximum(
                    rate - column("input_income") * policy["abatement"], 0
                )
            }
        elif program["kind"] == "credit":
            result = {
                "toy_bridge_credit": column("as_credit_units", 1) * policy["credit"]
            }
        else:
            result = {}
            for name, row in program["defaults"].items():
                result[f"seed_{name}"] = (
                    column(name, row["value"])
                    if name in table
                    else column(row.get("source", name), row["value"])
                )
            # Seed-only counterfactuals need no derived AS inputs.
            if set(variables).intersection(AS_OUTPUTS):
                eligible = (
                    (column("as_assets") <= policy["asset_limit"])
                    & (column("as_s66") == 0)
                    & (column("as_s67") == 0)
                )
                costs = np.maximum(
                    column("as_costs") - column("as_base_rate") * policy["cost_share"],
                    0,
                )
                amount = np.minimum(
                    column("as_maximum"), costs * policy["payment_share"]
                )
                reduction = (
                    np.maximum(column("as_income") - policy["income_floor"], 0)
                    * policy["income_abatement"]
                )
                amount = np.maximum(
                    amount - np.where(column("as_beneficiary") != 0, 0, reduction), 0
                )
                eligible &= (column("as_beneficiary") != 0) | (
                    column("as_income") < column("as_cutout")
                )
                result["toy_as_weekly"] = np.where(eligible, amount, 0)
                result["toy_as_unit_count"] = np.ones(len(table), dtype=np.int64)
                result["toy_as_eligible"] = np.where(eligible, 1, -1)
                result["toy_as_single"] = (
                    table["family_type"].to_numpy() == policy["family_type"]
                ).astype(np.int64)
                result["toy_as_other"] = (
                    (column("as_beneficiary") == 0) & (column("as_super") == 0)
                ).astype(np.int64)
                result["toy_as_excluded"] = (
                    (column("as_s66") != 0) | (column("as_s67") != 0)
                ).astype(np.int64)
        return {
            name: np.asarray(result[name], dtype=program["outputs"][name]["dtype"])
            for name in variables
        }


def _binding(name, field="engine_ref"):
    return {"binding": name, "field": field}


def _terms(variables, columns):
    return [
        {"variable": variable, "coefficient": 1, "entity": "family", "column": column}
        for variable, column in zip(variables, columns, strict=True)
    ]


def _seed(population, id, *, scenario=False, fields=INPUT_FIELDS):
    return _node(
        id,
        "simulate.counterfactual@1",
        population=population,
        sources=["rulespec_nz"],
        inputs=[
            _slice("person", ["age"]),
            _slice("family", fields if scenario else ["input_assets", "input_rent"]),
        ],
        outputs=[
            _owned("family", name, "float64", rewrite=scenario) for name in fields
        ],
        params={
            "engine_ref": _binding("entitlement"),
            "period": _binding("entitlement", "period"),
            "variables": [f"seed_{name}" for name in fields],
            "input_overrides": {
                "scenario": "knobs",
                "encoding": "json",
                "path": ["input_overrides"],
            }
            if scenario
            else _select("entitlement_seed", "json", "input_overrides"),
            "output_coefficients": _terms([f"seed_{name}" for name in fields], fields),
            "resource_sha256": {"scenario": "knobs", "encoding": "sha256"}
            if scenario
            else _select("entitlement_seed", "sha256"),
        },
    )


def _bridge(population, prefix):
    slices = [
        _slice("person", ["age", "input_income"]),
        _slice("family", ["as_credit_units"]),
    ]
    base = _node(
        f"{prefix}.bridge.base_rate",
        "simulate.counterfactual@1",
        population=population,
        sources=["rulespec_nz"],
        inputs=slices,
        outputs=[_owned("family", "as_base_rate", "float64")],
        params={
            "engine_ref": _binding("bridge_rates"),
            "period": _binding("bridge_rates", "period"),
            "variables": ["toy_bridge_benefit"],
            "resource_sha256": _select("entitlement_bridge", "sha256"),
            "input_overrides": [
                {"entity": "person", "column": "input_income", "value": 0}
            ],
            "components": [
                {
                    "engine_ref": _binding("bridge_credit"),
                    "period": _binding("bridge_credit", "period"),
                    "variables": ["toy_bridge_credit"],
                    "input_overrides": [
                        {"entity": "family", "column": "as_credit_units", "value": 1}
                    ],
                }
            ],
            "output_coefficients": [
                {
                    "variable": "toy_bridge_benefit",
                    "coefficient": 1,
                    "entity": "family",
                    "column": "as_base_rate",
                    "aggregation": "max",
                },
                {
                    "engine_ref": _binding("bridge_credit"),
                    "variable": "toy_bridge_credit",
                    "coefficient": TOY_DECLARATIONS["base_credit_coefficient"],
                    "entity": "family",
                    "column": "as_base_rate",
                },
            ],
        },
    )
    cutout = _node(
        f"{prefix}.bridge.cutout",
        "simulate.solve_zero@1",
        population=population,
        sources=["rulespec_nz"],
        inputs=slices,
        outputs=[_owned("family", "as_cutout", "float64")],
        params={
            "engine_ref": _binding("bridge_rates"),
            "period": _binding("bridge_rates", "period"),
            "variables": ["toy_bridge_benefit"],
            "resource_sha256": _select("entitlement_bridge", "sha256"),
            "input_overrides": [
                {"entity": "person", "column": "bridge_alternative", "value": 0}
            ],
            "output_coefficients": [
                {
                    "variable": "toy_bridge_benefit",
                    "coefficient": 1,
                    "entity": "family",
                    "column": "as_cutout",
                    "aggregation": "max",
                }
            ],
            "solve_input": {"entity": "person", "column": "input_income"},
            "bracket": TOY_DECLARATIONS["solve_bracket"],
            "tolerance": TOY_DECLARATIONS["solve_tolerance"],
            "iterations": TOY_DECLARATIONS["solve_iterations"],
        },
    )
    return [base, cutout]


def _as_and_gap(population, prefix, scenario):
    return [
        _node(
            f"{prefix}.as",
            "simulate.rules_by_ref@1",
            population=population,
            sources=["rulespec_nz"],
            inputs=[
                _slice("person", ["age"]),
                _slice("family", [*AS_INPUTS, "family_type"]),
            ],
            outputs=[
                _owned("family", name, dtype) for name, dtype in AS_OUTPUTS.items()
            ],
            rules_binding="entitlement",
            params={"variables": list(AS_OUTPUTS)},
        ),
        _node(
            f"{prefix}.gap",
            "takeup.gap@1",
            population=population,
            sources=["nz_holdout_facts"],
            inputs=[
                _slice("person", ["age"]),
                _slice("household", ["rent"]),
                _slice("family", [*AS_OUTPUTS, "as_beneficiary", "as_super"]),
            ],
            params={
                "country": COUNTRY,
                "references": _select("entitlement_gap"),
                "references_sha256": _select("entitlement_gap", "sha256"),
                "entities": ["person", "household", "family"],
                "weight_entity": "household",
                "annualisation_factors": _select("entitlement_annualisation"),
                "annualisation_sha256": _select("entitlement_annualisation", "sha256"),
                "scenario": scenario,
            },
        ),
    ]


def _boundary(id, base):
    return _node(
        id,
        "transport.boundary@1",
        structural="filter",
        base=base,
        inputs=[_slice("person", ["age"])],
        params={},
    )


def _scenario_nodes(fields):
    nodes = [_boundary("nz.scn.{scenario}", "nz.as")]
    if fields:
        nodes.append(
            _seed(
                "nz.scn.{scenario}",
                "nz.scn.{scenario}.override",
                scenario=True,
                fields=fields,
            )
        )
    return [
        *nodes,
        *_as_and_gap("nz.scn.{scenario}", "nz.scn.{scenario}", "{scenario}"),
    ]


def _entitlement_document(skeleton):
    shared = [_seed("nz.as", "nz.as.seed"), *_bridge("nz.as", "nz.as")]
    shared += [
        _boundary("nz.validate", "nz.calibrate"),
        _node(
            "nz.validate.wff",
            "simulate.counterfactual@1",
            population="nz.validate",
            sources=["rulespec_nz"],
            inputs=[_slice("person", ["age"])],
            outputs=[_owned("family", "toy_wff_credit", "float64")],
            params={
                "engine_ref": _binding("bridge_credit"),
                "period": _binding("bridge_credit", "period"),
                "variables": ["toy_bridge_credit"],
                "input_overrides": [
                    {"entity": "family", "column": "as_credit_units", "value": 1}
                ],
                "output_coefficients": _terms(
                    ["toy_bridge_credit"], ["toy_wff_credit"]
                ),
            },
        ),
        _node(
            "nz.validate.wff_tripwire",
            "takeup.compare@1",
            population="nz.validate",
            sources=["nz_holdout_facts"],
            inputs=[
                _slice("person", ["age"]),
                _slice("household", ["rent"]),
                _slice("family", ["toy_wff_credit"]),
            ],
            params={
                "country": COUNTRY,
                "references": _select("wff_holdout_references"),
                "references_sha256": _select("wff_holdout_references", "sha256"),
                "entities": ["person", "household", "family"],
                "weight_entity": "household",
            },
        ),
    ]
    return {
        "schema_version": 1,
        "nodes": shared,
        # A newly added scenario in the C2 fixture declares an asset override.
        "scenario_nodes": _scenario_nodes(("as_assets",)),
        "scenario_gap": "nz.scn.{scenario}.gap",
        "variant_gap": "nz.v.{variant}.gap",
        "bands": {
            "id": "nz.as.bands",
            "population": "nz.terminal",
            "baseline_scenario": "S0",
        },
        "package": {
            "id": "nz.package",
            "artifact_name": "bands",
            "inputs": [_edge("wff_tripwire", "nz.validate.wff_tripwire", "comparison")],
        },
        "checkpoints": ["nz.as.bands", "nz.validate.wff_tripwire"],
    }


def _variant_nodes(skeleton, variant):
    """An actual independent target/problem/calibration branch, then AS."""
    by_id = {node["id"]: node for node in skeleton["nodes"]}
    prefix = f"nz.v.{variant}"
    nodes = [_boundary(prefix, "nz.open")]
    if variant == "V3":
        receipt = copy.deepcopy(by_id["nz.takeup.assign"])
        receipt.update(id=f"{prefix}.receipt", population=prefix)
        for output in receipt["outputs"]:
            output["rewrite"] = True
        receipt["params"] = {
            "receipt_contract": _select("variant_receipt", "json", "receipts"),
            "receipt_contract_sha256": _select("variant_receipt", "sha256"),
        }
        nodes.append(receipt)
    targets = copy.deepcopy(by_id["nz.targets.compile"])
    targets.update(id=f"{prefix}.targets", population=prefix)
    reference = "target_references" if variant == "V3" else f"variant_{variant}_targets"
    targets["params"] = {
        "country": COUNTRY,
        "references": _select(reference),
        "references_sha256": _select(reference, "sha256"),
    }
    problem = copy.deepcopy(by_id["nz.targets.problem"])
    problem.update(
        id=f"{prefix}.problem",
        population=prefix,
        artifact_inputs=[_edge("surface", targets["id"])],
    )
    calibrated = copy.deepcopy(by_id["nz.calibrate"])
    calibrated.update(
        id=f"{prefix}.calibrate",
        base=prefix,
        artifact_inputs=[_edge("problem", problem["id"])],
    )
    as_population = f"{prefix}.housing"
    group = copy.deepcopy(by_id["nz.as.inputs"])
    group.update(id=f"{prefix}.inputs", population=as_population)
    nodes += [
        targets,
        problem,
        calibrated,
        _boundary(as_population, calibrated["id"]),
        group,
        _seed(as_population, f"{prefix}.seed"),
        *_bridge(as_population, prefix),
        *_as_and_gap(as_population, prefix, variant),
    ]
    return nodes


@dataclass(frozen=True)
class EntitlementFixture(ComposedFixture):
    """Prepared composed toy graph with real entitlement extension kernels."""

    def graph(self, *, extended=True):
        return compose_transport_graph(self.spec, self.config)

    def reprepare(self, spec=None):
        from microcosm.build.transport.entitlement import entitlement_extension

        spec = self.spec if spec is None else spec
        bindings = spec["resources"]["axiom_rules_bindings"]
        root = self.sources["rulespec_nz"]
        engines = {
            row["id"]: (
                EntitlementToyEngine if row["id"] in TOY_PROGRAMS else ToyRulesEngine
            )(root, row)
            for row in bindings["bindings"]
        }
        registry = build_transport_registry(
            bindings, root, unit_rule=UNIT_RULE, engines_by_binding=engines
        )
        config = replace(
            self.config,
            engine_refs=registry.engine_refs,
            extensions=(entitlement_extension(spec),),
        )
        return replace(
            self,
            spec=spec,
            config=config,
            registry=registry,
            engines_by_binding=engines,
        )

    def relocated(self, root):
        shutil.copytree(self.root, root)
        sources = {
            name: root / path.relative_to(self.root)
            for name, path in self.sources.items()
        }
        return replace(self, root=root, sources=sources).reprepare()


def make_entitlement_fixture(root: Path, *, seed_overrides=None, variants=True):
    base = make_composed_fixture(root)
    spec = copy.deepcopy(base.spec)
    resources = spec["resources"]
    tree = base.sources["rulespec_nz"]
    for name, raw in TOY_PROGRAMS.items():
        program = {key: value for key, value in raw.items() if key != "path"}
        program["relations"] = []
        payload = canonical_json(program)
        path = tree / raw["path"]
        path.write_bytes(payload)
        resources["axiom_rules_bindings"]["bindings"].append(
            {
                "id": name,
                "rulespec_path": raw["path"],
                "sha256": sha256(payload).hexdigest(),
                "entity": raw["entity"],
                "engine_entity": raw["entity"].capitalize(),
                "period": PERIOD,
                "variables": list(raw["outputs"]),
            }
        )
    resources["entitlement_seed"] = {
        "input_overrides": [
            {"entity": "family", "column": name, "value": value}
            for name, value in (seed_overrides or {}).items()
        ]
    }
    resources["entitlement_bridge"] = copy.deepcopy(TOY_DECLARATIONS)
    gap_rows = [
        reference_row("toy_as_expenditure", entity="family", measure="toy_as_weekly"),
        reference_row(
            "toy_as_recipients",
            entity="family",
            measure="toy_as_unit_count",
            filter_="toy_as_weekly",
        ),
        reference_row(
            "toy_as_single_expenditure",
            entity="family",
            measure="toy_as_weekly",
            filter_="toy_as_single",
        ),
        reference_row(
            "toy_as_beneficiary_expenditure",
            entity="family",
            measure="toy_as_weekly",
            filter_="as_beneficiary",
        ),
        reference_row(
            "toy_as_super_expenditure",
            entity="family",
            measure="toy_as_weekly",
            filter_="as_super",
        ),
        reference_row(
            "toy_as_other_expenditure",
            entity="family",
            measure="toy_as_weekly",
            filter_="toy_as_other",
        ),
        reference_row(
            "toy_as_excluded_units",
            entity="family",
            measure="toy_as_unit_count",
            filter_="toy_as_excluded",
        ),
    ]
    for row, cell in zip(
        gap_rows[2:],
        (
            {"income_decile": "toy_decile_1", "family_type": "single"},
            {"receipt_stratum": "beneficiary"},
            {"receipt_stratum": "superannuitant"},
            {"receipt_stratum": "other"},
            {"eligibility_stratum": "excluded_s66_or_s67"},
        ),
        strict=True,
    ):
        # LedgerTargetReference metadata is a flat string-to-string mapping.
        row["metadata"].update(cell)
    resources["entitlement_gap"] = _references(gap_rows)
    # Recipient target counts positivity, rather than summing -1/1 judgments.
    resources["entitlement_annualisation"] = {
        row["name"]: TOY_DECLARATIONS["annualisation"]
        if row["measure"] == "toy_as_weekly"
        else 1
        for row in gap_rows
    }
    resources["wff_holdout_references"] = _references(
        [
            reference_row(
                "toy_wff_expenditure", entity="family", measure="toy_wff_credit"
            )
        ]
    )
    old_facts = [
        json.loads(row)
        for row in base.sources["nz_holdout_facts"].read_text().splitlines()
    ]
    write_facts(
        base.sources["nz_holdout_facts"],
        [
            *old_facts,
            _fact("toy_as_expenditure", 100.0, "family"),
            _fact("toy_as_recipients", 15.0, "family"),
            *[_fact(row["name"], 0.0, "family") for row in gap_rows[2:]],
            _fact("toy_wff_expenditure", 200.0, "family"),
        ],
    )
    resources["scenarios"] = {
        "scenarios": [
            {"id": "S0", "tier": "entitlement", "knobs": {"input_overrides": []}},
            {
                "id": "S1",
                "tier": "entitlement",
                "knobs": {
                    "input_overrides": [
                        {"entity": "family", "column": "as_assets", "value": 0.0}
                    ]
                },
            },
            {
                "id": "S2",
                "tier": "entitlement",
                "knobs": {
                    "input_overrides": [
                        {"entity": "family", "column": "as_costs", "value": 150.0}
                    ]
                },
            },
            {
                "id": "S3",
                "tier": "entitlement",
                "knobs": {
                    "input_overrides": [
                        {"entity": "family", "column": "as_maximum", "value": 60.0}
                    ]
                },
            },
            {
                "id": "S4",
                "tier": "entitlement",
                "knobs": {
                    "input_overrides": [
                        {"entity": "family", "column": "as_costs", "value": 450.0}
                    ]
                },
            },
            {
                "id": "S5",
                "tier": "entitlement",
                "knobs": {
                    "input_overrides": [
                        {"entity": "family", "column": "as_super", "value": 1},
                        {"entity": "family", "column": "as_beneficiary", "value": 0},
                    ]
                },
            },
        ]
    }
    document = _entitlement_document(resources["transport_graph"])
    for row in resources["scenarios"]["scenarios"]:
        row["nodes"] = _scenario_nodes(
            tuple(item["column"] for item in row["knobs"]["input_overrides"])
        )
    # The alternate base-rate scenario runs the engine again; policy-derived
    # bridge columns are never overwritten by a data-only transform.
    s5 = resources["scenarios"]["scenarios"][-1]
    alternate = copy.deepcopy(document["nodes"][1])
    alternate.update(id="nz.scn.{scenario}.base_rate", population="nz.scn.{scenario}")
    alternate["outputs"][0]["rewrite"] = True
    alternate["params"]["input_overrides"].append(
        {"entity": "person", "column": "bridge_alternative", "value": 1}
    )
    s5["nodes"].insert(2, alternate)
    if variants:
        facts = [
            json.loads(row)
            for row in base.sources["nz_calibration_facts"].read_text().splitlines()
        ]
        fact_by_name = {
            row["semantic_fact_key"].rsplit(":", 1)[-1]: row for row in facts
        }
        for variant, factor in TOY_DECLARATIONS["target_factors"].items():
            refs = copy.deepcopy(resources["target_references"])
            for row in refs["target_references"]:
                original = row["name"]
                name = f"{variant}_{original}"
                row.update(name=name, ledger_fact_key=f"toy.aggregate_fact.v1:{name}")
                facts.append(
                    _fact(name, fact_by_name[original]["value"] * factor, row["entity"])
                )
            resources[f"variant_{variant}_targets"] = refs
        write_facts(base.sources["nz_calibration_facts"], facts)
        receipt = copy.deepcopy(resources["receipt_contract"])
        receipt["receipts"]["programs"][0]["rate"] = TOY_DECLARATIONS[
            "alternative_receipt_rate"
        ]
        resources["variant_receipt"] = receipt
        for variant in ("V1", "V2", "V3"):
            resources["scenarios"]["scenarios"].append(
                {
                    "id": variant,
                    "tier": "calibration",
                    "knobs": {},
                    "nodes": _variant_nodes(resources["transport_graph"], variant),
                }
            )
    resources["entitlement_graph"] = document
    fixture = EntitlementFixture(
        base.root,
        spec,
        base.config,
        base.registry,
        base.sources,
        base.engines_by_binding,
    )
    return fixture.reprepare()


def run_entitlement_graph(
    root, fixture, *, store=None, exported=None, graph=None, resume="auto"
):
    """Execute the complete graph with one pinned H5 outer-service input."""
    root.mkdir(parents=True, exist_ok=True)
    graph = fixture.graph() if graph is None else graph
    store = (
        ContentStore(root / "store", codecs=fixture.registry.codecs)
        if store is None
        else store
    )
    if exported is None:
        prepared = run_graph(
            compile_graph(through(graph, "nz.export.prepare")),
            sources=fixture.sources,
            store=store,
            kernels=fixture.registry.kernels,
        )
        descriptor = json.loads(
            store.load_bytes(
                prepared.nodes["nz.export.prepare"].opaque_artifacts[
                    "export_descriptor"
                ]
            )
        )
        version = compile_graph(graph).versions["nz.export.prepare"]
        exported = root / EXPORT_FILENAME
        materialize_export(prepared.populations[version], descriptor, exported)
    manifest = run_graph(
        compile_graph(graph),
        sources=fixture.source_mapping(exported),
        store=store,
        kernels=fixture.registry.kernels,
        resume=resume,
    )
    return manifest, store, exported
