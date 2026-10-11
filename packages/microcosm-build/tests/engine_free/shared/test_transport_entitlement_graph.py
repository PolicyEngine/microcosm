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
from microcosm.build.transport.takeup_kernels import decode_bands, decode_gap
from microcosm.graph import compile_graph, graph_to_json
from microcosm.graph.keys import node_key, source_content_key
from test_support.microcosm_build.transport_entitlement import (
    AS_INPUTS,
    AS_OUTPUTS,
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
