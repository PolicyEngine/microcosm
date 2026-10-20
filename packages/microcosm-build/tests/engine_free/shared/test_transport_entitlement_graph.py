"""Executed entitlement invariants and cache locality on a synthetic graph.

The composed graph uses the public engine protocol, all four new kernels,
six entitlement scenarios, three independently calibrated variants and a
computed toy WFF tripwire. Every amount belongs to synthetic fixture data.
These tests make no claim about statutory correctness or real donor outputs.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.transport.compose import (
    compose_transport_graph,
    validate_transport_activation,
)
from microcosm.build.transport.takeup_kernels import (
    decode_bands,
    decode_cell_diagnostics,
    decode_gap,
    reporting_cell_declarations,
)
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN
from microcosm.graph import compile_graph, graph_to_json
from microcosm.graph.keys import node_key, source_content_key
from test_support.microcosm_build.transport_entitlement import (
    AS_INPUTS,
    AS_OUTPUTS,
    EXPORT_FILENAME,
    TOY_DECLARATIONS,
    make_entitlement_fixture,
    run_entitlement_graph,
)

EXECUTED = settings(
    max_examples=2,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
NONNEGATIVE = st.floats(
    min_value=0, max_value=1500, allow_nan=False, allow_infinity=False
)
# A unit increase is representable throughout NONNEGATIVE's float64 range.
POSITIVE_INCREMENT = st.floats(
    min_value=1, max_value=1500, allow_nan=False, allow_infinity=False
)


def _case_root(tmp_path):
    """Give every Hypothesis execution an independent cold input/store tree."""
    root = tmp_path / f"case-{len(tuple(tmp_path.iterdir()))}"
    root.mkdir()
    return root


def _artifact(manifest, store, node, name):
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[name])


def _frame(manifest, population="nz.scn.S0"):
    return manifest.populations[population]


def _misses(manifest):
    return {name for name, receipt in manifest.nodes.items() if not receipt.hit}


def _keys(manifest):
    return {name: receipt.key for name, receipt in manifest.nodes.items()}


def _seeded(fixture, overrides):
    spec = copy.deepcopy(fixture.spec)
    spec["resources"]["entitlement_seed"]["input_overrides"] = [
        {"entity": "family", "column": name, "value": value}
        for name, value in overrides.items()
    ]
    return fixture.reprepare(spec)


def _declared_keys(fixture, graph, exported):
    compiled = compile_graph(graph)
    sources = {
        name: source_content_key(name, path)
        for name, path in fixture.source_mapping(exported).items()
    }
    keys = {}
    for name in compiled.order:
        kernel = fixture.registry.kernels.get(graph.node(name).kernel)
        keys[name] = node_key(
            compiled,
            name,
            keys,
            kernel.implementation_hash(),
            sources,
            kernel_capabilities=kernel.capabilities,
        )
    return keys


def _cell_support_fixture(tmp_path):
    fixture = make_entitlement_fixture(tmp_path / "fixture", variants=False)
    spec = copy.deepcopy(fixture.spec)
    resources = spec["resources"]
    grid = {
        "entity": "family",
        "measure": "toy_as_weekly",
        "period": "2026-27",
        "annualisation_factor": 52,
        "tables": [
            {
                "name": "toy_type",
                "axes": [
                    {
                        "column": "family_type",
                        "values": [
                            "single",
                            "couple",
                            "sole_parent",
                            "couple_with_children",
                        ],
                    }
                ],
            }
        ],
    }
    resources["scenarios"]["gap_conventions"] = {"reporting_cells": grid}
    resources["gates"]["gates"].append(
        {
            "id": "toy_cell_support",
            "gate": "cell_support",
            "phase": "terminal",
            "criticality": "release_blocking",
            "evidence_absent_blocks": True,
            "parameters": {
                "required_scenarios": [
                    row["id"] for row in resources["scenarios"]["scenarios"]
                ],
                "required_cells": [
                    row["name"] for row in reporting_cell_declarations(grid)
                ],
                # CREATE owns and carries the declared source-household concept.
                "aggregation_column": SOURCE_HOUSEHOLD_ID_COLUMN,
                "required_weight_entity": "household",
                "required_weight_kind": "calibrated",
                "minimum_cell_ess": 100,
                "maximum_contribution_cv": 2,
                "maximum_single_donor_dollar_share": 0.1,
                "target_minimum_cell_ess": 400,
                "target_maximum_single_donor_dollar_share": 0.05,
            },
        }
    )
    return fixture.reprepare(spec)


def test_cell_support_graph_enforces_scenario_complete_diagnostics_and_suppresses_cells(
    tmp_path,
):
    fixture = _cell_support_fixture(tmp_path)
    graph = fixture.graph()
    collector = "nz.as.bands.cell_diagnostics"
    gate = graph.node("nz.gates.terminal")
    assert any(
        edge.name == "cell_diagnostics" and edge.producer == collector
        for edge in gate.artifact_inputs
    )
    manifest, store, exported = run_entitlement_graph(
        tmp_path / "supported", fixture, graph=graph
    )
    assert exported.exists()
    diagnostics = decode_cell_diagnostics(
        _artifact(manifest, store, collector, "cell_diagnostics")
    )
    assert {row["scenario"] for row in diagnostics["scenarios"]} == {
        row["id"] for row in fixture.spec["resources"]["scenarios"]["scenarios"]
    }
    assert all(
        cell["suppressed"]
        for scenario in diagnostics["scenarios"]
        for cell in scenario["cells"]
    )
    assert (
        decode_bands(_artifact(manifest, store, "nz.as.bands", "bands"))["bands"] == []
    )
    report = json.loads(_artifact(manifest, store, "nz.gates.terminal", "gate_report"))
    assert report["enforcement"]["artifact_permitted"] is True


def test_cell_support_graph_adds_diagnostics_to_explicit_gap_output_declarations(
    tmp_path,
):
    fixture = _cell_support_fixture(tmp_path)
    spec = copy.deepcopy(fixture.spec)
    entitlement = spec["resources"]["entitlement_graph"]
    node_groups = [
        entitlement.get(template, [])
        for template in ("nodes", "scenario_nodes", "variant_nodes")
    ]
    node_groups.extend(
        scenario.get("nodes", [])
        for scenario in spec["resources"]["scenarios"]["scenarios"]
    )
    for declared_nodes in node_groups:
        for row in declared_nodes:
            if row["kernel"] == "takeup.gap@1":
                row["artifact_outputs"] = [
                    {
                        "name": "gap",
                        "type": {"name": "microcosm.takeup.gap", "schema_version": 1},
                    }
                ]
    graph = fixture.reprepare(spec).graph()
    compile_graph(graph)
    assert all(
        {output.name for output in node.artifact_outputs} == {"gap", "cell_diagnostics"}
        for node in graph.nodes
        if node.kernel == "takeup.gap@1"
    )


def _with_cell_target(fixture, **overrides):
    """Return the fixture's spec with a tracked cell-target gate appended."""
    spec = copy.deepcopy(fixture.spec)
    gates = spec["resources"]["gates"]["gates"]
    contract = next(g for g in gates if g["gate"] == "cell_support")["parameters"]
    parameters = {
        "required_scenarios": list(contract["required_scenarios"]),
        "required_cells": list(contract["required_cells"]),
        "aggregation_column": contract["aggregation_column"],
        "required_weight_entity": "household",
        "required_weight_kind": "calibrated",
        "minimum_cell_ess": contract["target_minimum_cell_ess"],
        "maximum_single_donor_dollar_share": contract[
            "target_maximum_single_donor_dollar_share"
        ],
    }
    parameters.update(overrides)
    gates.append(
        {
            "id": "toy_cell_support_target",
            "gate": "cell_support_target",
            "phase": "terminal",
            "criticality": "diagnostic",
            "parameters": parameters,
        }
    )
    return spec


def test_cell_support_target_matching_the_contract_builds(tmp_path):
    fixture = _cell_support_fixture(tmp_path)
    fixture.reprepare(_with_cell_target(fixture)).graph()


@pytest.mark.parametrize(
    "overrides",
    [
        {"minimum_cell_ess": 1_000_000},
        {"maximum_single_donor_dollar_share": 1},
        {"required_cells": ["toy_type/single"]},
        {"required_scenarios": ["S0"]},
        {"aggregation_column": "household_id"},
    ],
    ids=["ess", "share", "cells", "scenarios", "aggregation"],
)
def test_cell_support_target_drift_from_the_contract_is_refused(tmp_path, overrides):
    fixture = _cell_support_fixture(tmp_path)
    with pytest.raises(ValueError, match="the two must agree"):
        fixture.reprepare(_with_cell_target(fixture, **overrides)).graph()


def test_cell_support_contract_target_drift_is_refused(tmp_path):
    # The reviewer's mutant: change only the contract's copy of the target.
    fixture = _cell_support_fixture(tmp_path)
    spec = _with_cell_target(fixture)
    contract = next(
        g for g in spec["resources"]["gates"]["gates"] if g["gate"] == "cell_support"
    )
    contract["parameters"]["target_minimum_cell_ess"] = 1_000_000
    with pytest.raises(ValueError, match="the two must agree"):
        fixture.reprepare(spec).graph()


def test_cell_support_target_lists_in_another_order_build(tmp_path):
    fixture = _cell_support_fixture(tmp_path)
    contract = _cell_support_contract(fixture.spec)
    fixture.reprepare(
        _with_cell_target(
            fixture,
            required_cells=list(reversed(contract["required_cells"])),
            required_scenarios=list(reversed(contract["required_scenarios"])),
        )
    ).graph()


def test_cell_support_target_with_null_not_applicable_is_checked(tmp_path):
    # The gate parser treats "not_applicable": null as an active gate.
    fixture = _cell_support_fixture(tmp_path)
    spec = _with_cell_target(fixture, minimum_cell_ess=1_000_000)
    spec["resources"]["gates"]["gates"][-1]["not_applicable"] = None
    with pytest.raises(ValueError, match="the two must agree"):
        fixture.reprepare(spec).graph()


@pytest.mark.parametrize(
    "key",
    [
        "minimum_cell_ess",
        "maximum_single_donor_dollar_share",
        "required_cells",
        "required_scenarios",
        "aggregation_column",
    ],
)
def test_cell_support_target_omitting_a_shared_value_is_refused(tmp_path, key):
    fixture = _cell_support_fixture(tmp_path)
    spec = _with_cell_target(fixture)
    del spec["resources"]["gates"]["gates"][-1]["parameters"][key]
    with pytest.raises(ValueError, match="must both declare"):
        fixture.reprepare(spec).graph()


@pytest.mark.parametrize("malformed", ["mixed", "mappings", "string"])
def test_cell_support_target_malformed_lists_are_refused_by_name(tmp_path, malformed):
    fixture = _cell_support_fixture(tmp_path)
    cells = list(_cell_support_contract(fixture.spec)["required_cells"])
    overrides = {
        "mixed": {"required_cells": [*cells, 1]},
        "mappings": {"required_cells": [{"name": cell} for cell in cells]},
        # The same characters as one cell name, as a list: equal only if a
        # string were sorted like a list.
        "string": {"required_cells": sorted(cells[0])},
    }[malformed]
    if malformed == "string":
        spec = _with_cell_target(fixture, **overrides)
        _cell_support_contract(spec)["required_cells"] = cells[0]
        # A string contract list is itself malformed, so the contract's shape
        # check refuses it before the two copies are compared.
        expected = "non-empty list of non-empty strings"
    else:
        spec = _with_cell_target(fixture, **overrides)
        expected = "the two must agree"
    with pytest.raises(ValueError, match=expected):
        fixture.reprepare(spec).graph()


def test_cell_support_contract_with_null_not_applicable_is_the_contract(tmp_path):
    fixture = _cell_support_fixture(tmp_path)
    spec = copy.deepcopy(fixture.spec)
    next(g for g in spec["resources"]["gates"]["gates"] if g["gate"] == "cell_support")[
        "not_applicable"
    ] = None
    nodes = {node.id: node.kernel for node in fixture.reprepare(spec).graph().nodes}
    assert nodes == {node.id: node.kernel for node in fixture.graph().nodes}
    assert "nz.as.bands.cell_diagnostics" in nodes


def _cell_support_contract(spec):
    return next(
        g for g in spec["resources"]["gates"]["gates"] if g["gate"] == "cell_support"
    )["parameters"]


@pytest.mark.parametrize(
    "key",
    [
        "minimum_cell_ess",
        "maximum_contribution_cv",
        "maximum_single_donor_dollar_share",
        "target_minimum_cell_ess",
        "target_maximum_single_donor_dollar_share",
        "required_cells",
        "required_scenarios",
        "aggregation_column",
        "required_weight_kind",
        "required_weight_entity",
    ],
)
def test_cell_support_contract_omitting_a_key_is_refused_by_name(tmp_path, key):
    fixture = _cell_support_fixture(tmp_path)
    spec = copy.deepcopy(fixture.spec)
    del _cell_support_contract(spec)[key]
    with pytest.raises(ValueError, match=f"omits .*'{key}'"):
        fixture.reprepare(spec).graph()


@pytest.mark.parametrize("malformed", ["mappings", "mixed", "empty", "string", "blank"])
@pytest.mark.parametrize("key", ["required_cells", "required_scenarios"])
def test_cell_support_contract_malformed_lists_are_refused_by_name(
    tmp_path, key, malformed
):
    fixture = _cell_support_fixture(tmp_path)
    spec = copy.deepcopy(fixture.spec)
    contract = _cell_support_contract(spec)
    values = list(contract[key])
    contract[key] = {
        "mappings": [{"name": value} for value in values],
        "mixed": [*values, 1],
        "empty": [],
        "string": values[0],
        "blank": [*values, ""],
    }[malformed]
    with pytest.raises(ValueError, match="non-empty list of non-empty strings"):
        fixture.reprepare(spec).graph()


def test_matching_mapping_lists_on_both_gates_are_refused_by_name(tmp_path):
    # Round 3 review of #1202: matching mapping lists passed the drift guard
    # and then raised a bare TypeError at the set comparison.
    fixture = _cell_support_fixture(tmp_path)
    cells = [
        {"name": c} for c in _cell_support_contract(fixture.spec)["required_cells"]
    ]
    spec = _with_cell_target(fixture, required_cells=cells)
    _cell_support_contract(spec)["required_cells"] = copy.deepcopy(cells)
    with pytest.raises(ValueError, match="non-empty list of non-empty strings"):
        fixture.reprepare(spec).graph()


@pytest.mark.parametrize("absence", ["diagnostics", "scenario", "lineage"])
def test_cell_support_graph_absence_prevents_export(tmp_path, absence):
    fixture = _cell_support_fixture(tmp_path)
    graph = fixture.graph()
    if absence == "diagnostics":
        gate = graph.node("nz.gates.terminal")
        changed = replace(
            gate,
            artifact_inputs=tuple(
                edge for edge in gate.artifact_inputs if edge.name != "cell_diagnostics"
            ),
        )
        graph = replace(
            graph,
            nodes=tuple(
                changed if node.id == gate.id else node for node in graph.nodes
            ),
        )
    elif absence == "scenario":
        collector = graph.node("nz.as.bands.cell_diagnostics")
        changed = replace(collector, artifact_inputs=collector.artifact_inputs[:-1])
        graph = replace(
            graph,
            nodes=tuple(
                changed if node.id == collector.id else node for node in graph.nodes
            ),
        )
    else:
        changed_nodes = []
        for node in graph.nodes:
            if node.kernel == "takeup.gap@1":
                node = replace(
                    node,
                    params={
                        **node.params,
                        "source_household_column": "missing_source_identity",
                    },
                )
            changed_nodes.append(node)
        graph = replace(graph, nodes=tuple(changed_nodes))
    with pytest.raises((ValueError, RuntimeError), match="gate|scenario|lineage"):
        run_entitlement_graph(tmp_path / "blocked", fixture, graph=graph)
    assert not (tmp_path / "blocked" / EXPORT_FILENAME).exists()


@EXECUTED
@example(income=0, costs=1000, assets=0, beneficiary=False)
@given(
    income=NONNEGATIVE, costs=NONNEGATIVE, assets=NONNEGATIVE, beneficiary=st.booleans()
)
def test_as_bounds_weighted_total_exact_gap_labelled_bands_and_determinism(
    tmp_path, income, costs, assets, beneficiary
):
    """Bounds, weighted E, exact E-A, labelled order and deterministic bytes."""
    tmp_path = _case_root(tmp_path)
    fixture = make_entitlement_fixture(
        tmp_path / "fixture",
        seed_overrides={
            "as_income": income,
            "as_costs": costs,
            "as_assets": assets,
            "as_beneficiary": int(beneficiary),
        },
    )
    graph = fixture.graph()
    assert graph_to_json(graph) == graph_to_json(fixture.graph())
    compiled = compile_graph(graph)
    for root in (
        node.id for node in graph.nodes if node.kernel.startswith("calibrate.")
    ):
        pending, ancestors = [root], set()
        while pending:
            name = pending.pop()
            if name in ancestors:
                continue
            ancestors.add(name)
            pending.extend(compiled.predecessors[name])
        for name in ancestors:
            node = graph.node(name)
            assert "nz_holdout_facts" not in node.sources
            assert (
                node.params.get("engine_ref")
                != fixture.registry.engine_refs["entitlement"]
            )
            assert not {
                column
                for item in node.inputs
                if item.entity == "family"
                for column in item.columns
            }.intersection(AS_INPUTS)
    first, store, exported = run_entitlement_graph(
        tmp_path / "first", fixture, graph=graph
    )
    assert _keys(first) == _declared_keys(fixture, graph, exported)
    for scenario in fixture.spec["resources"]["scenarios"]["scenarios"]:
        name = scenario["id"]
        population = (
            f"nz.scn.{name}"
            if scenario["tier"] == "entitlement"
            else f"nz.v.{name}.housing"
        )
        frame = _frame(first, population)
        family = frame.table("family")
        amounts = family["toy_as_weekly"].to_numpy()
        assert np.all(amounts >= 0)
        assert np.all(amounts <= family["as_maximum"].to_numpy())
        prefix = (
            f"nz.scn.{name}" if scenario["tier"] == "entitlement" else f"nz.v.{name}"
        )
        gap = decode_gap(_artifact(first, store, f"{prefix}.gap", "gap"))
        rows = {row["name"]: row for row in gap["gaps"]}
        expenditure = rows["toy_as_expenditure"]
        factor = fixture.spec["resources"]["entitlement_annualisation"][
            "toy_as_expenditure"
        ]
        expected = np.sum(frame.resolve_weights("family").values * amounts * factor)
        # The kernel aggregates each household's families before weighting;
        # float64 reassociation is allowed only on this independent total.
        assert expenditure["E"] == pytest.approx(float(expected), rel=1e-12, abs=1e-12)
        references = {
            row["name"]: row
            for row in fixture.spec["resources"]["entitlement_gap"]["target_references"]
        }
        for row in gap["gaps"]:
            assert row["gap"] == row["E"] - row["A"]
            assert row["metadata"] == references[row["name"]]["metadata"]
            values = (
                amounts if row["measure"] == "toy_as_weekly" else np.ones(len(family))
            )
            if row["filter"] is not None:
                values = values * (family[row["filter"]].to_numpy() != 0)
            expected_cell = np.sum(
                frame.resolve_weights("family").values
                * values
                * row["annualisation_factor"]
            )
            assert row["E"] == pytest.approx(float(expected_cell), rel=1e-12, abs=1e-12)
            if row["measure"] == "toy_as_unit_count":
                assert row["annualisation_factor"] == 1
    bands = decode_bands(_artifact(first, store, "nz.as.bands", "bands"))
    labels = {row["scenario"] for row in bands["scenarios"]}
    assert labels == {
        row["id"] for row in fixture.spec["resources"]["scenarios"]["scenarios"]
    }
    for row in bands["bands"]:
        assert row["low"]["value"] <= row["baseline"]["value"] <= row["high"]["value"]
        assert row["baseline"]["scenario"] == "S0"
        for bound in ("low", "baseline", "high"):
            assert row[bound]["scenario"] in labels
    # Independent cold store: equality checks actual execution, not memoization.
    second, second_store, _ = run_entitlement_graph(
        tmp_path / "second", fixture, graph=graph, exported=exported
    )
    assert _keys(first) == _keys(second)
    for name, receipt in first.nodes.items():
        assert receipt.frame_key == second.nodes[name].frame_key
        for artifact in receipt.opaque_artifacts:
            assert _artifact(first, store, name, artifact) == _artifact(
                second, second_store, name, artifact
            )
    warm, _, _ = run_entitlement_graph(
        tmp_path / "warm", fixture, store=store, exported=exported
    )
    assert _misses(warm) == set()


@EXECUTED
@given(
    excess=st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False)
)
def test_as_is_zero_for_excess_assets_and_each_statutory_exclusion(tmp_path, excess):
    """An encoded asset limit or either independent exclusion zeros AS."""
    tmp_path = _case_root(tmp_path)
    fixture = make_entitlement_fixture(tmp_path / "fixture", variants=False)
    _, store, exported = run_entitlement_graph(tmp_path / "initial", fixture)
    policy = fixture.engines_by_binding["entitlement"].program()["policy"]
    for case in (
        {"as_assets": policy["asset_limit"] + excess},
        {"as_assets": 0, "as_s66": 1},
        {"as_assets": 0, "as_s67": 1},
    ):
        changed = _seeded(fixture, case)
        result, _, _ = run_entitlement_graph(
            tmp_path / "excluded", changed, store=store, exported=exported
        )
        assert np.all(_frame(result).table("family")["toy_as_weekly"].to_numpy() == 0)
        assert result.nodes["nz.calibrate"].hit is True


@EXECUTED
@example(income=80, increment=10)
@given(income=NONNEGATIVE, increment=POSITIVE_INCREMENT)
def test_nonbeneficiary_as_is_nonincreasing_in_income(tmp_path, income, increment):
    tmp_path = _case_root(tmp_path)
    fixture = make_entitlement_fixture(
        tmp_path / "fixture",
        variants=False,
        seed_overrides={"as_income": income, "as_beneficiary": 0, "as_assets": 0},
    )
    low, store, exported = run_entitlement_graph(tmp_path / "low", fixture)
    changed = _seeded(
        fixture, {"as_income": income + increment, "as_beneficiary": 0, "as_assets": 0}
    )
    high, _, _ = run_entitlement_graph(
        tmp_path / "high", changed, store=store, exported=exported
    )
    assert np.all(
        _frame(high).table("family")["toy_as_weekly"].to_numpy()
        <= _frame(low).table("family")["toy_as_weekly"].to_numpy()
    )


@EXECUTED
@example(costs=40, increment=20)
@given(costs=NONNEGATIVE, increment=POSITIVE_INCREMENT)
def test_as_is_nondecreasing_in_qualifying_costs_until_the_cap(
    tmp_path, costs, increment
):
    tmp_path = _case_root(tmp_path)
    fixture = make_entitlement_fixture(
        tmp_path / "fixture",
        variants=False,
        seed_overrides={"as_costs": costs, "as_income": 0, "as_assets": 0},
    )
    low, store, exported = run_entitlement_graph(tmp_path / "low", fixture)
    changed = _seeded(
        fixture, {"as_costs": costs + increment, "as_income": 0, "as_assets": 0}
    )
    high, _, _ = run_entitlement_graph(
        tmp_path / "high", changed, store=store, exported=exported
    )
    before = _frame(low).table("family")["toy_as_weekly"].to_numpy()
    after = _frame(high).table("family")["toy_as_weekly"].to_numpy()
    assert np.all(after >= before)
    assert np.all(after <= _frame(high).table("family")["as_maximum"].to_numpy())


@EXECUTED
@given(
    asset_knob=st.floats(
        min_value=1, max_value=1000, allow_nan=False, allow_infinity=False
    )
)
def test_a3_s1_knob_misses_only_s1_bands_and_package(tmp_path, asset_knob):
    tmp_path = _case_root(tmp_path)
    fixture = make_entitlement_fixture(tmp_path / "fixture")
    initial, store, exported = run_entitlement_graph(tmp_path / "initial", fixture)
    spec = copy.deepcopy(fixture.spec)
    spec["resources"]["scenarios"]["scenarios"][1]["knobs"]["input_overrides"][0][
        "value"
    ] = asset_knob
    changed = fixture.reprepare(spec)
    result, _, _ = run_entitlement_graph(
        tmp_path / "changed", changed, store=store, exported=exported
    )
    expected = {name for name in initial.nodes if name.startswith("nz.scn.S1.")} | {
        "nz.as.bands",
        "nz.package",
    }
    assert _misses(result) == expected
    assert result.nodes["nz.scn.S1"].hit is True
    assert {
        name
        for name in initial.nodes
        if initial.nodes[name].key != result.nodes[name].key
    } == expected


@EXECUTED
@given(newlines=st.integers(min_value=1, max_value=8))
def test_a6_holdout_bytes_miss_only_gaps_validation_bands_and_package(
    tmp_path, newlines
):
    tmp_path = _case_root(tmp_path)
    fixture = make_entitlement_fixture(tmp_path / "fixture")
    initial, store, exported = run_entitlement_graph(tmp_path / "initial", fixture)
    source = fixture.sources["nz_holdout_facts"]
    source.write_bytes(source.read_bytes() + b"\n" * newlines)
    result, _, _ = run_entitlement_graph(
        tmp_path / "changed", fixture, store=store, exported=exported
    )
    expected = {name for name in initial.nodes if name.endswith(".gap")} | {
        "nz.validate.wff_tripwire",
        "nz.as.bands",
        "nz.package",
    }
    assert _misses(result) == expected
    assert {
        name
        for name in initial.nodes
        if initial.nodes[name].key != result.nodes[name].key
    } == expected
    assert result.nodes["nz.calibrate"].hit is True


@EXECUTED
@given(assets=NONNEGATIVE)
def test_c2_adding_scenario_rekeys_only_bands_and_package(tmp_path, assets):
    tmp_path = _case_root(tmp_path)
    fixture = make_entitlement_fixture(tmp_path / "fixture")
    initial, store, exported = run_entitlement_graph(tmp_path / "initial", fixture)
    spec = copy.deepcopy(fixture.spec)
    spec["resources"]["scenarios"]["scenarios"].append(
        {
            "id": "SX",
            "tier": "entitlement",
            "knobs": {
                "input_overrides": [
                    {"entity": "family", "column": "as_assets", "value": assets}
                ]
            },
        }
    )
    changed = fixture.reprepare(spec)
    result, _, _ = run_entitlement_graph(
        tmp_path / "changed", changed, store=store, exported=exported
    )
    assert {
        name
        for name in initial.nodes
        if initial.nodes[name].key != result.nodes[name].key
    } == {"nz.as.bands", "nz.package"}
    new = {
        name
        for name in result.nodes
        if name == "nz.scn.SX" or name.startswith("nz.scn.SX.")
    }
    assert _misses(result) == new | {"nz.as.bands", "nz.package"}
    assert result.nodes["nz.calibrate"].hit is True


def test_bridges_and_as_graph_match_direct_toy_engine_calls(tmp_path):
    """A differential independently calls the adapter on the executed frame."""
    fixture = make_entitlement_fixture(tmp_path / "fixture", variants=False)
    manifest, store, _ = run_entitlement_graph(tmp_path / "run", fixture)
    (tmp_path / "node-durations.json").write_text(
        json.dumps(
            {name: receipt.wall_time for name, receipt in manifest.nodes.items()},
            sort_keys=True,
        )
    )
    frame = _frame(manifest)
    table = frame.table("family")
    rate_policy = fixture.engines_by_binding["bridge_rates"].program()["policy"]
    credit_policy = fixture.engines_by_binding["bridge_credit"].program()["policy"]
    expected_base = (
        rate_policy["rate"]
        + credit_policy["credit"] * TOY_DECLARATIONS["base_credit_coefficient"]
    )
    assert np.all(table["as_base_rate"].to_numpy() == expected_base)
    assert table["as_cutout"].to_numpy() == pytest.approx(
        rate_policy["rate"] / rate_policy["abatement"],
        abs=TOY_DECLARATIONS["solve_tolerance"],
    )
    engine = fixture.engines_by_binding["entitlement"]
    direct = engine.materialize(frame, tuple(AS_OUTPUTS), "2026-27")
    for name in AS_OUTPUTS:
        np.testing.assert_array_equal(table[name].to_numpy(), direct[name])
    alternative = (
        _frame(manifest, "nz.scn.S5").table("family")["as_base_rate"].to_numpy()
    )
    assert np.all(
        alternative
        == rate_policy["alternative_rate"]
        + credit_policy["credit"] * TOY_DECLARATIONS["base_credit_coefficient"]
    )
    comparison = json.loads(
        _artifact(manifest, store, "nz.validate.wff_tripwire", "comparison")
    )
    validation_frame = _frame(manifest, "nz.validate")
    expected_credit = np.sum(
        validation_frame.resolve_weights("family").values
        * validation_frame.table("family")["toy_wff_credit"].to_numpy()
    )
    assert comparison["comparisons"][0]["model"] == pytest.approx(
        expected_credit, rel=1e-12
    )
    receipt = json.loads(_artifact(manifest, store, "nz.package", "receipt"))
    assert (
        receipt["artifacts"]["bands"]["key"]
        == manifest.nodes["nz.as.bands"].opaque_artifacts["bands"]
    )
    assert (
        receipt["artifacts"]["wff_tripwire"]["key"]
        == manifest.nodes["nz.validate.wff_tripwire"].opaque_artifacts["comparison"]
    )


def test_automatic_extension_installation_and_package_edge_restrictions(tmp_path):
    """Automatic composition equals explicit installation; receipt edges are narrow."""
    fixture = make_entitlement_fixture(tmp_path / "fixture", variants=False)
    explicit = fixture.graph()
    automatic = compose_transport_graph(
        fixture.spec, replace(fixture.config, extensions=())
    )
    assert graph_to_json(explicit) == graph_to_json(automatic)
    extension = fixture.config.extensions[0]
    target, edge = extension.package_inputs[0]
    skeleton_output = explicit.node("nz.targets.compile").artifact_outputs[0]
    invalid = (
        (
            replace(extension, package_inputs=(("nz.export.prepare", edge),)),
            "only a skeleton package node",
        ),
        (
            replace(
                extension, package_inputs=((target, replace(edge, name="diagnostics")),)
            ),
            "new and distinct",
        ),
        (
            replace(
                extension,
                package_inputs=(
                    (
                        target,
                        replace(
                            edge,
                            producer="nz.targets.compile",
                            artifact="surface",
                            type=skeleton_output.type,
                        ),
                    ),
                ),
            ),
            "from extension nodes",
        ),
    )
    for changed, message in invalid:
        with pytest.raises(ValueError, match=message):
            compose_transport_graph(
                fixture.spec, replace(fixture.config, extensions=(changed,))
            )
    spec = copy.deepcopy(fixture.spec)
    rows = spec["resources"]["transport_graph"]["nodes"]
    consumer = copy.deepcopy(next(row for row in rows if row["id"] == "nz.package"))
    consumer["id"] = "nz.package.consumer"
    consumer["artifact_inputs"].append(
        {"name": "prior_receipt", "producer": "nz.package", "artifact": "receipt"}
    )
    rows.append(consumer)
    with pytest.raises(ValueError, match="terminal skeleton package node"):
        compose_transport_graph(spec, fixture.config)


@pytest.mark.parametrize("template", ["scenario_nodes", "variant_nodes"])
def test_activation_ignores_unused_absent_tier_template(tmp_path, template):
    """Explicit scenario nodes replace their tier's entire unused template."""
    fixture = make_entitlement_fixture(tmp_path / "fixture")
    spec = copy.deepcopy(fixture.spec)
    document = spec["resources"]["entitlement_graph"]
    unused = copy.deepcopy(document["scenario_nodes"][0])
    unused["params"]["unused"] = {"resource": "unused_absent"}
    document[template] = [unused]
    validate_transport_activation(spec)
    assert graph_to_json(
        compose_transport_graph(spec, fixture.config)
    ) == graph_to_json(fixture.graph())


@pytest.mark.parametrize("location", ["common", "scenario", "variant", "template"])
@pytest.mark.parametrize("gap", ["null", "path", "name"])
def test_activation_checks_every_instantiated_entitlement_selection(
    tmp_path, location, gap
):
    """G6's resource preflight also covers the G7 rows that will be used."""
    fixture = make_entitlement_fixture(tmp_path / "fixture")
    spec = copy.deepcopy(fixture.spec)
    document = spec["resources"]["entitlement_graph"]
    scenarios = spec["resources"]["scenarios"]["scenarios"]
    if location == "common":
        node = document["nodes"][0]
    elif location == "template":
        scenarios[0].pop("nodes")
        node = document["scenario_nodes"][0]
    else:
        tier = "entitlement" if location == "scenario" else "calibration"
        node = next(row for row in scenarios if row["tier"] == tier)["nodes"][0]
    spec["resources"]["selected_probe"] = {"value": None}
    selection = {"resource": "selected_probe", "path": ["value"]}
    if gap == "path":
        selection["path"] = ["absent"]
    elif gap == "name":
        selection["resource"] = None
    node["params"]["probe"] = selection
    match = {
        "null": "selected_probe.*not activated",
        "path": "selected_probe.*no selected path",
        "name": "must name a nonempty string",
    }[gap]
    with pytest.raises(ValueError, match=match):
        validate_transport_activation(spec)
