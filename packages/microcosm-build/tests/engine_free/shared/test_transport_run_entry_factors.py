"""Run entry checks the same scenario money factors that normalization selects."""

from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.money_normalization import (
    selected_money_factors,
    validate_money_normalization,
)
from microcosm.build.reference_factors import (
    require_current_factors,
    resolve_reference_factor,
    validate_factor_context,
)
from microcosm.build.transport import cli as cli_module
from microcosm.build.transport import compose as compose_module
from microcosm.build.transport import entitlement as entitlement_module
from microcosm.build.transport import money_kernels as money_kernels_module
from microcosm.build.transport.codecs import register_transport_codecs
from microcosm.build.transport.money_kernels import MONEY_NORMALIZE
from microcosm.graph import (
    SOURCE_CODECS,
    ContentStore,
    KernelContext,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.money_normalization import (
    DIMENSION,
    PROFILES,
    add_money_columns,
    factor_id,
    graph_declaration,
    money_factor_context,
    money_graph,
    money_registry,
    money_sources,
    normalize_node,
    write_money_feed,
)
from test_support.microcosm_build.reference_factor_transport import (
    factor_transport_fixture,
)
from test_support.microcosm_build.transport_composed import make_composed_fixture
from test_support.microcosm_build.transport_entitlement import make_entitlement_fixture
from test_support.microcosm_build.transport_graph import (
    reference_document,
    reference_row,
)
from test_support.microcosm_build.transport_population import (
    donor_pin,
    write_population_sources,
)

TODAY = date(2026, 10, 24)
CONTEXT_RESOURCE = "aging-evidence.json"


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(cli_module, "_today", lambda: TODAY)


@pytest.fixture(scope="module")
def money_evidence(tmp_path_factory):
    path, pin, identity, facts = write_money_feed(
        tmp_path_factory.mktemp("run-entry-money-feed")
    )
    return path, pin, identity, facts, money_factor_context(identity, facts)


def _json(document):
    return canonical_json(document).decode("utf-8")


def _selection(resource, encoding="json"):
    return {"resource": resource, "encoding": encoding}


def _money_row(
    node_id="toy.money.normalize", *, selection=None, context=CONTEXT_RESOURCE
):
    return {
        "id": node_id,
        "kernel": "money.normalize@1",
        "population": "toy.money",
        "params": {
            "declaration": _selection("money_normalization"),
            "profile_selection": _json({}) if selection is None else selection,
            "factor_context": _selection(context),
        },
    }


def _money_spec(context, *, scenarios=()):
    scenarios = copy.deepcopy(list(scenarios))
    for scenario in scenarios:
        scenario.setdefault("tier", "calibration")
    spec = {
        "country": "xx",
        "resources": {
            "money_normalization": graph_declaration(),
            CONTEXT_RESOURCE: copy.deepcopy(context),
            "transport_graph": {"nodes": [_money_row()], "sources": []},
            "target_references": reference_document(
                [reference_row("toy_count", entity="person", measure="people")]
            ),
            "reference_profiles": {
                "schema_version": 1,
                "reference_resource": "target_references",
                "factor_context_resource": CONTEXT_RESOURCE,
                "dimensions": {},
            },
            "scenarios": {
                "knob_domains": {DIMENSION: list(PROFILES)},
                "scenarios": scenarios,
            },
            "entitlement_graph": {
                "schema_version": 1,
                "variant_nodes": [
                    _money_row(
                        "toy.{variant}.normalize",
                        selection={"scenario": "knobs", "encoding": "json"},
                    )
                ],
                "variant_gap": "toy.{variant}.gap",
                "bands": {
                    "id": "toy.money.bands",
                    "population": "toy.money",
                    "baseline_scenario": scenarios[0]["id"]
                    if scenarios
                    else "baseline",
                },
            },
        },
    }
    if not scenarios:
        del spec["resources"]["entitlement_graph"]
    return spec


def _with_expiry(evidence, profile, until):
    _, _, identity, facts, context = evidence
    context = copy.deepcopy(context)
    entry = next(
        row
        for row in context["factors"]
        if row["declaration"]["id"] == factor_id(profile)
    )
    entry["declaration"]["validity"]["valid_until"] = until.isoformat()
    factor, receipt = resolve_reference_factor(entry["declaration"], facts, identity)
    entry.update(operational_factor=factor, receipt=receipt)
    return context


def _later_steps(monkeypatch):
    steps = []
    for owner, name in (
        (cli_module, "validate_transport_activation"),
        (cli_module, "build_transport_registry"),
        (cli_module, "prepare_create_outputs"),
        (cli_module.TRANSPORT_CREATE, "run"),
    ):
        step = Mock(side_effect=AssertionError(f"Run entry must refuse before {name}."))
        monkeypatch.setattr(owner, name, step)
        steps.append(step)
    return steps


@pytest.mark.parametrize("resume", ["auto", "forbid"])
@pytest.mark.parametrize("entrypoint", ["cli", "library"])
def test_expired_scenario_money_factor_refuses_before_create(
    money_evidence, tmp_path, monkeypatch, capsys, resume, entrypoint
):
    context = _with_expiry(money_evidence, "low", TODAY - timedelta(days=1))
    spec = _money_spec(
        context, scenarios=[{"id": "variant", "knobs": {DIMENSION: "low"}}]
    )
    # The default factor remains current; only the variant selects the expired one.
    require_current_factors(
        validate_factor_context(context, [factor_id("central")]), today=TODAY
    )
    later = _later_steps(monkeypatch)
    if entrypoint == "cli":
        monkeypatch.setattr(cli_module, "load_transport_spec", lambda source: spec)
        out = tmp_path / "out"
        assert (
            cli_module.main(["--country", "xx", "--out", str(out), "--resume", resume])
            == 1
        )
        error = capsys.readouterr().err
        assert not out.exists()
    else:
        with pytest.raises(ValueError) as refusal:
            cli_module.prepare_transport_build(spec, sources={}, resume=resume)
        error = str(refusal.value)
    assert factor_id("low") in error
    assert factor_id("central") not in error
    assert "reviewed re-declaration" in error
    for step in later:
        step.assert_not_called()


@pytest.mark.parametrize("resume", ["auto", "forbid"])
def test_expired_default_money_factor_refuses_without_scenarios(
    money_evidence, monkeypatch, resume
):
    spec = _money_spec(
        _with_expiry(money_evidence, "central", TODAY - timedelta(days=1))
    )
    del spec["resources"]["scenarios"]
    later = _later_steps(monkeypatch)
    with pytest.raises(ValueError, match=factor_id("central")):
        cli_module.prepare_transport_build(spec, sources={}, resume=resume)
    for step in later:
        step.assert_not_called()


@pytest.mark.parametrize("remaining_days", [0, 1])
def test_current_money_factor_includes_valid_until_today(
    money_evidence, remaining_days
):
    context = _with_expiry(
        money_evidence, "central", TODAY + timedelta(days=remaining_days)
    )
    spec = _money_spec(context)
    cli_module._require_current_run_factors(spec, resume="auto")
    cli_module._require_current_run_factors(spec, resume="forbid")


def _identity_declaration():
    declaration = graph_declaration()
    for family in declaration["index_family_factors"]:
        declaration["index_family_factors"][family] = {
            "identity": True,
            "reason": "The toy family keeps its donor basis.",
        }
    return declaration


@pytest.mark.parametrize("with_scenarios", [False, True])
def test_identity_money_families_select_no_factor(
    money_evidence, monkeypatch, with_scenarios
):
    spec = _money_spec(
        money_evidence[-1],
        scenarios=[
            {"id": profile, "knobs": {DIMENSION: profile}} for profile in PROFILES
        ]
        if with_scenarios
        else (),
    )
    resources = spec["resources"]
    resources["money_normalization"] = _identity_declaration()
    del resources["reference_profiles"]
    captured = Mock()
    monkeypatch.setattr(cli_module, "require_current_factors", captured)
    cli_module._require_current_run_factors(spec, resume="auto")
    captured.assert_called_once_with({}, today=TODAY)


def test_money_declaration_and_context_accept_json_resource_names(money_evidence):
    spec = _money_spec(money_evidence[-1])
    resources = spec["resources"]
    for name in ("money_normalization", "reference_profiles", "scenarios"):
        resources[f"{name}.json"] = resources.pop(name)
    cli_module._require_current_run_factors(spec, resume="auto")


def test_without_money_resource_preserves_reference_result_and_refusal(
    tmp_path, monkeypatch
):
    _, _, _, _, spec = factor_transport_fixture(tmp_path)
    selected = {}
    for view in compose_module.reference_profile_views(spec):
        selected.update(view[2]["selected_factors"])
    captured = Mock(wraps=require_current_factors)
    monkeypatch.setattr(cli_module, "require_current_factors", captured)
    cli_module._require_current_run_factors(spec, resume="auto")
    captured.assert_called_once_with(selected, today=TODAY)
    expired_today = date(2100, 1, 1)
    monkeypatch.setattr(cli_module, "_today", lambda: expired_today)
    with pytest.raises(ValueError) as original:
        require_current_factors(selected, today=expired_today)
    with pytest.raises(ValueError) as run_entry:
        cli_module._require_current_run_factors(spec, resume="auto")
    assert str(run_entry.value) == str(original.value)


def test_require_returns_before_money_selection_or_clock(money_evidence, monkeypatch):
    spec = _money_spec(
        _with_expiry(money_evidence, "central", TODAY - timedelta(days=1))
    )
    steps = []
    for owner, name in (
        (cli_module, "reference_profile_views"),
        (cli_module, "money_node_factor_entries"),
        (entitlement_module, "selected_money_factors"),
        (cli_module, "_today"),
    ):
        step = Mock(side_effect=AssertionError(f"require must return before {name}."))
        monkeypatch.setattr(owner, name, step)
        steps.append(step)
    cli_module._require_current_run_factors(spec, resume="require")
    for step in steps:
        step.assert_not_called()


KNOBS = st.fixed_dictionaries(
    {},
    optional={
        DIMENSION: st.sampled_from(PROFILES),
        "index": st.sampled_from(PROFILES),
    },
)


@settings(
    max_examples=12,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@example(choices=[({}, {DIMENSION: "low"}), ({DIMENSION: "high"}, {})])
@given(choices=st.lists(st.tuples(KNOBS, KNOBS), min_size=1, max_size=3))
def test_each_scenario_run_entry_selection_equals_kernel_selection(
    money_evidence, monkeypatch, choices
):
    path, pin, _, _, context = money_evidence
    scenarios = [
        {
            "id": f"variant-{index}",
            "knobs": knobs,
            "nodes": [
                _money_row(
                    f"toy.variant-{index}.normalize",
                    selection=_selection(f"node_selection_{index}"),
                )
            ],
        }
        for index, (knobs, _) in enumerate(choices)
    ]
    spec = _money_spec(context, scenarios=scenarios)
    for index, (_, selection) in enumerate(choices):
        spec["resources"][f"node_selection_{index}"] = selection
    declaration = spec["resources"]["money_normalization"]
    checked = []

    def select_for_entry(document, selection):
        result = selected_money_factors(document, selection)
        checked.append((copy.deepcopy(selection), result))
        return result

    monkeypatch.setattr(entitlement_module, "selected_money_factors", select_for_entry)
    cli_module._require_current_run_factors(spec, resume="auto")
    assert checked[0] == ({}, selected_money_factors(declaration, {}))
    assert {_json(selection) for selection, _ in checked} == {
        _json(selection) for selection in [{}, *(choice[1] for choice in choices)]
    }
    selected_in_kernel = []

    def select_for_kernel(document, selection):
        result = selected_money_factors(document, selection)
        selected_in_kernel.append((copy.deepcopy(selection), result))
        return result

    monkeypatch.setattr(
        money_kernels_module, "selected_money_factors", select_for_kernel
    )
    for scenario, (_, selection) in zip(scenarios, choices, strict=True):
        knobs = scenario["knobs"]
        entry_choice = next(choice for choice in checked if choice[0] == selection)
        # The entitlement factory uses this parameter resolver for scenario
        # selectors before handing the resolved params to compose._node.
        profile_selection = entitlement_module._parameter(
            spec, None, scenario["nodes"][0]["params"]["profile_selection"], scenario
        )
        template = normalize_node(
            f"toy.{scenario['id']}.normalize",
            f"toy.{scenario['id']}",
            declaration,
            context,
            pin,
        )
        node = compose_module._node(
            spec,
            {
                "id": template.id,
                "kernel": template.kernel,
                "population": template.population,
                "sources": list(template.sources),
                "inputs": [
                    {"entity": item.entity, "columns": list(item.columns)}
                    for item in template.inputs
                ],
                "outputs": [
                    {
                        "entity": item.entity,
                        "column": item.column,
                        "dtype": item.dtype,
                        "rewrite": item.rewrite,
                    }
                    for item in template.outputs
                ],
            },
            compose_module.TransportGraphConfig(
                engine_refs={}, create_outputs=template.outputs
            ),
            {},
            resolved_params={
                **template.params,
                "profile_selection": profile_selection,
            },
            profile_selection=knobs,
        )
        assert json.loads(node.params["profile_selection"]) == selection
        kernel_context = KernelContext(
            node=node,
            tables={
                "person": pd.DataFrame(
                    {
                        "person_id": [1],
                        "interest_income": [3.0],
                        "rental_income": [-2.0],
                        "farm_self_employment_income": [1.0],
                    }
                )
            },
            weights={},
            strata=pd.Series([], dtype=object),
            params=node.params,
            rng=np.random.default_rng(0),
            sources={"money_facts": path},
            frame_metadata={"donor_time_period": 2024},
        )
        result = MONEY_NORMALIZE.run(kernel_context)
        expected = selected_money_factors(declaration, selection)
        assert entry_choice == (selection, expected)
        assert selected_in_kernel[-1] == entry_choice
        assert result.receipt["profile"] == expected[0]
        assert {row["factor_id"] for row in result.receipt["factors"].values()} == {
            value for value in expected[1].values() if value is not None
        }


@pytest.fixture(scope="module")
def reference_and_money_spec(tmp_path_factory, money_evidence):
    _, _, _, _, spec = factor_transport_fixture(
        tmp_path_factory.mktemp("run-entry-reference-feed")
    )
    resources = spec["resources"]
    profiles = resources["reference_profiles"]["dimensions"]["index"]["profiles"]
    profiles["central"]["transformations"] = []
    profiles["high"]["transformations"] = []
    context = resources.pop("factor_context")
    # Each entry retains its own feed identity; the optional shared identity
    # would incorrectly assert that these two independent toy feeds coincide.
    context.pop("feed_identity")
    context["factors"].extend(copy.deepcopy(money_evidence[-1]["factors"]))
    resources[CONTEXT_RESOURCE] = context
    resources["reference_profiles"]["factor_context_resource"] = CONTEXT_RESOURCE
    resources["money_normalization"] = graph_declaration()
    resources["scenarios"]["knob_domains"] = {
        DIMENSION: list(PROFILES),
        "index": list(PROFILES),
    }
    return spec


@settings(
    max_examples=12,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)
@example(choices=[])
@example(choices=[({DIMENSION: "low", "index": "low"}, {DIMENSION: "high"})])
@given(choices=st.lists(st.tuples(KNOBS, KNOBS), max_size=4))
def test_checked_factors_equal_exact_reference_and_money_union(
    reference_and_money_spec, monkeypatch, choices
):
    spec = copy.deepcopy(reference_and_money_spec)
    resources = spec["resources"]
    domains = resources["scenarios"]["knob_domains"]
    assert (
        tuple(resources["money_normalization"]["factor_profile"]["profiles"])
        == PROFILES
    )
    assert (
        tuple(resources["reference_profiles"]["dimensions"]["index"]["profiles"])
        == PROFILES
    )
    assert all(
        value in domains[name]
        for knobs, selection in choices
        for selection in (knobs, selection)
        for name, value in selection.items()
    )
    money_spec = _money_spec(
        resources[CONTEXT_RESOURCE],
        scenarios=[
            {
                "id": f"variant-{index}",
                "knobs": knobs,
                "nodes": [
                    _money_row(
                        f"toy.variant-{index}.normalize",
                        selection=_selection(f"node_selection_{index}"),
                    )
                ],
            }
            for index, (knobs, _) in enumerate(choices)
        ],
    )
    for name in ("transport_graph", "entitlement_graph"):
        if name in money_spec["resources"]:
            resources[name] = money_spec["resources"][name]
    resources["scenarios"]["scenarios"] = money_spec["resources"]["scenarios"][
        "scenarios"
    ]
    for index, (_, selection) in enumerate(choices):
        resources[f"node_selection_{index}"] = selection
    expected = {}
    for view in compose_module.reference_profile_views(spec):
        expected.update(view[2].get("selected_factors", {}))
    money_ids = set()
    for selection in [{}, *(choice[1] for choice in choices)]:
        _, families = selected_money_factors(
            resources["money_normalization"], selection
        )
        money_ids.update(value for value in families.values() if value is not None)
    expected.update(validate_factor_context(resources[CONTEXT_RESOURCE], money_ids))
    captured = Mock()
    with monkeypatch.context() as patch:
        patch.setattr(cli_module, "require_current_factors", captured)
        cli_module._require_current_run_factors(spec, resume="auto")
    checked = {}
    for call in captured.call_args_list:
        assert call.kwargs == {"today": TODAY}
        checked.update(call.args[0])
    assert checked == expected


def test_require_replays_warm_money_graph_after_selected_factors_expire(
    money_evidence, tmp_path, monkeypatch
):
    path, pin, _, _, context = money_evidence
    spec = _money_spec(
        context,
        scenarios=[
            {"id": profile, "knobs": {DIMENSION: profile}} for profile in PROFILES
        ],
    )
    population = write_population_sources(
        tmp_path / "population", time_period=2024, edit_tables=add_money_columns
    )
    register_transport_codecs(SOURCE_CODECS)
    graph = money_graph(population, graph_declaration(), context, pin)
    registry = money_registry()
    store = ContentStore(tmp_path / "store")
    compiled = compile_graph(graph)
    sources = money_sources(population, path)
    cli_module._require_current_run_factors(spec, resume="auto")
    warm = run_graph(compiled, sources=sources, store=store, kernels=registry)
    assert warm.nodes["toy.v1.normalize"].receipt["profile"] == "low"
    calls = []
    for ref in registry.refs():
        step = Mock(side_effect=AssertionError("require must replay stored work."))
        monkeypatch.setattr(registry.get(ref), "run", step)
        calls.append(step)
    monkeypatch.setattr(cli_module, "_today", lambda: date(2100, 1, 1))
    cli_module._require_current_run_factors(spec, resume="require")
    replay = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="require"
    )
    assert all(row.hit for row in replay.nodes.values())
    for name in warm.nodes:
        assert replay.nodes[name].key == warm.nodes[name].key
        assert replay.nodes[name].receipt == warm.nodes[name].receipt
    for step in calls:
        step.assert_not_called()
    # The first declared node selects central; every profile has expired by now.
    with pytest.raises(ValueError, match=factor_id("central")):
        cli_module._require_current_run_factors(spec, resume="auto")


def _explicit_selection_spec(context):
    spec = _money_spec(context, scenarios=[{"id": "variant", "knobs": {}}])
    resources = spec["resources"]
    resources["transport_graph"]["nodes"] = []
    resources["variant_selection"] = {DIMENSION: "low"}
    resources["entitlement_graph"]["variant_nodes"][0]["params"][
        "profile_selection"
    ] = _selection("variant_selection")
    return spec


@pytest.mark.parametrize("resume", ["auto", "forbid"])
@pytest.mark.parametrize("entrypoint", ["cli", "library"])
def test_explicit_node_selection_refuses_before_create(
    money_evidence, tmp_path, monkeypatch, capsys, resume, entrypoint
):
    context = _with_expiry(money_evidence, "low", TODAY - timedelta(days=1))
    spec = _explicit_selection_spec(context)
    later = _later_steps(monkeypatch)
    if entrypoint == "cli":
        monkeypatch.setattr(cli_module, "load_transport_spec", lambda source: spec)
        out = tmp_path / "out"
        assert (
            cli_module.main(["--country", "xx", "--out", str(out), "--resume", resume])
            == 1
        )
        error = capsys.readouterr().err
        assert not out.exists()
    else:
        with pytest.raises(ValueError) as refusal:
            cli_module.prepare_transport_build(spec, sources={}, resume=resume)
        error = str(refusal.value)
    assert factor_id("low") in error
    assert factor_id("central") not in error
    for step in later:
        step.assert_not_called()


def test_unused_expired_factor_does_not_refuse_explicit_selection(money_evidence):
    context = _with_expiry(money_evidence, "central", TODAY - timedelta(days=1))
    cli_module._require_current_run_factors(
        _explicit_selection_spec(context), resume="auto"
    )


@pytest.mark.parametrize("node_expired", [True, False])
def test_money_node_uses_its_own_context(money_evidence, node_expired):
    expired = _with_expiry(money_evidence, "central", TODAY - timedelta(days=1))
    current = money_evidence[-1]
    spec = _money_spec(current if node_expired else expired)
    spec["resources"]["node_context"] = expired if node_expired else current
    spec["resources"]["transport_graph"]["nodes"][0]["params"]["factor_context"] = (
        _selection("node_context")
    )
    if node_expired:
        with pytest.raises(ValueError, match=factor_id("central")):
            cli_module._require_current_run_factors(spec, resume="auto")
    else:
        cli_module._require_current_run_factors(spec, resume="auto")


@pytest.mark.parametrize("expired_first", [True, False])
def test_two_node_contexts_keep_both_validity_checks(
    money_evidence, monkeypatch, expired_first
):
    spec = _money_spec(money_evidence[-1])
    resources = spec["resources"]
    resources["expired_context"] = _with_expiry(
        money_evidence, "central", TODAY - timedelta(days=1)
    )
    rows = [
        _money_row("toy.expired.normalize", context="expired_context"),
        _money_row("toy.current.normalize"),
    ]
    resources["transport_graph"]["nodes"] = rows if expired_first else rows[::-1]
    today = Mock(return_value=TODAY)
    monkeypatch.setattr(cli_module, "_today", today)
    with pytest.raises(ValueError, match=factor_id("central")):
        cli_module._require_current_run_factors(spec, resume="auto")
    today.assert_called_once_with()


def test_sibling_declaration_selects_its_own_factors(money_evidence, monkeypatch):
    spec = _explicit_selection_spec(
        _with_expiry(money_evidence, "central", TODAY - timedelta(days=1))
    )
    resources = spec["resources"]
    sibling = graph_declaration()
    sibling_context = copy.deepcopy(money_evidence[-1])
    _, _, identity, facts, _ = money_evidence
    for entry in sibling_context["factors"]:
        entry["declaration"]["id"] = f"sibling_{entry['declaration']['id']}"
        factor, receipt = resolve_reference_factor(
            entry["declaration"], facts, identity
        )
        entry.update(operational_factor=factor, receipt=receipt)
    for family in sibling["index_family_factors"].values():
        family["factors"] = {
            profile: f"sibling_{selected}"
            for profile, selected in family["factors"].items()
        }
    resources["money_normalization_sibling"] = sibling
    resources["sibling_context"] = sibling_context
    resources["variant_selection"] = {DIMENSION: "high"}
    params = resources["entitlement_graph"]["variant_nodes"][0]["params"]
    params["declaration"] = _selection("money_normalization_sibling")
    params["factor_context"] = _selection("sibling_context")
    captured = Mock(wraps=require_current_factors)
    monkeypatch.setattr(cli_module, "require_current_factors", captured)
    cli_module._require_current_run_factors(spec, resume="auto")
    assert {key for call in captured.call_args_list for key in call.args[0]} == {
        f"sibling_{factor_id('high')}"
    }


def test_literal_skeleton_selection_uses_selected_factor(money_evidence):
    spec = _money_spec(_with_expiry(money_evidence, "low", TODAY - timedelta(days=1)))
    spec["resources"]["transport_graph"]["nodes"][0]["params"]["profile_selection"] = (
        _json({DIMENSION: "low"})
    )
    with pytest.raises(ValueError, match=factor_id("low")):
        cli_module._require_current_run_factors(spec, resume="auto")


def test_money_node_needs_no_reference_profiles_resource(money_evidence):
    spec = _money_spec(money_evidence[-1])
    del spec["resources"]["reference_profiles"]
    cli_module._require_current_run_factors(spec, resume="auto")


def test_money_resource_without_money_nodes_selects_no_factors(
    money_evidence, monkeypatch
):
    spec = _money_spec(
        _with_expiry(money_evidence, "central", TODAY - timedelta(days=1))
    )
    spec["resources"]["transport_graph"]["nodes"] = []
    captured = Mock(wraps=require_current_factors)
    monkeypatch.setattr(cli_module, "require_current_factors", captured)
    cli_module._require_current_run_factors(spec, resume="auto")
    captured.assert_called_once_with({}, today=TODAY)


def _row_for_placement(spec, placement):
    if placement == "skeleton":
        return spec["resources"]["transport_graph"]["nodes"][0]
    return spec["resources"]["entitlement_graph"]["variant_nodes"][0]


@pytest.mark.parametrize("placement", ["skeleton", "entitlement"])
@pytest.mark.parametrize(
    "parameter", ["declaration", "profile_selection", "factor_context"]
)
@pytest.mark.parametrize(
    "selector",
    [
        {"binding": "toy", "field": "engine_ref"},
        {"prepared": "engine_refs"},
    ],
    ids=["binding", "prepared"],
)
def test_unavailable_money_parameter_refuses_with_node_name(
    money_evidence, placement, parameter, selector
):
    spec = _money_spec(money_evidence[-1], scenarios=[{"id": "variant", "knobs": {}}])
    row = _row_for_placement(spec, placement)
    row["params"][parameter] = selector
    with pytest.raises(ValueError) as refusal:
        cli_module._require_current_run_factors(spec, resume="auto")
    assert row["id"].replace("{variant}", "variant") in str(refusal.value)
    assert parameter in str(refusal.value)


@pytest.mark.parametrize("placement", ["skeleton", "entitlement"])
@pytest.mark.parametrize(
    "parameter", ["declaration", "profile_selection", "factor_context"]
)
@pytest.mark.parametrize("value", [None, 3, "not JSON", "[]"])
def test_money_parameter_requires_json_object_with_node_name(
    money_evidence, placement, parameter, value
):
    spec = _money_spec(money_evidence[-1], scenarios=[{"id": "variant", "knobs": {}}])
    row = _row_for_placement(spec, placement)
    row["params"][parameter] = value
    with pytest.raises(ValueError) as refusal:
        cli_module._require_current_run_factors(spec, resume="auto")
    assert row["id"].replace("{variant}", "variant") in str(refusal.value)
    assert parameter in str(refusal.value)


def test_every_family_factor_of_a_node_is_checked(money_evidence):
    declaration = graph_declaration()
    declaration["index_family_factors"]["capital_income"]["factors"]["central"] = (
        factor_id("low")
    )
    spec = _money_spec(_with_expiry(money_evidence, "low", TODAY - timedelta(days=1)))
    spec["resources"]["money_normalization"] = declaration
    with pytest.raises(ValueError, match=factor_id("low")):
        cli_module._require_current_run_factors(spec, resume="auto")


def test_invalid_money_declaration_refuses_with_node_name(money_evidence):
    spec = _money_spec(money_evidence[-1])
    spec["resources"]["money_normalization"]["bogus"] = 1
    with pytest.raises(ValueError, match="toy.money.normalize"):
        cli_module._require_current_run_factors(spec, resume="auto")


def test_unknown_money_profile_refuses_with_node_name(money_evidence):
    spec = _money_spec(money_evidence[-1])
    spec["resources"]["transport_graph"]["nodes"][0]["params"]["profile_selection"] = (
        _json({DIMENSION: "nope"})
    )
    with pytest.raises(ValueError, match="toy.money.normalize.*profile_selection"):
        cli_module._require_current_run_factors(spec, resume="auto")


def test_missing_selected_factor_refuses_with_node_name(money_evidence):
    context = copy.deepcopy(money_evidence[-1])
    context["factors"] = [
        row
        for row in context["factors"]
        if row["declaration"]["id"] != factor_id("central")
    ]
    with pytest.raises(
        ValueError,
        match="toy.money.normalize.*factor_context.*Missing selected factor",
    ):
        cli_module._require_current_run_factors(_money_spec(context), resume="auto")


def test_skeleton_mapping_selection_refuses_like_composition(money_evidence):
    spec = _money_spec(money_evidence[-1])
    spec["resources"]["transport_graph"]["nodes"][0]["params"]["profile_selection"] = {
        DIMENSION: "low"
    }
    with pytest.raises(
        ValueError,
        match="toy.money.normalize.*must be a resource or prepared selection",
    ):
        cli_module._require_current_run_factors(spec, resume="auto")


@pytest.mark.parametrize("placement", ["skeleton", "entitlement"])
def test_non_object_graph_row_refuses_cleanly(money_evidence, placement):
    spec = _money_spec(money_evidence[-1], scenarios=[{"id": "variant", "knobs": {}}])
    if placement == "skeleton":
        spec["resources"]["transport_graph"]["nodes"].append("not a node")
    else:
        spec["resources"]["entitlement_graph"]["nodes"] = ["not a node"]
    with pytest.raises(TypeError, match="graph node must be an object"):
        cli_module._require_current_run_factors(spec, resume="auto")


def test_identity_node_context_refuses_at_run_entry_as_the_kernel_does(money_evidence):
    declaration = _identity_declaration()
    spec = _money_spec(money_evidence[-1])
    spec["resources"]["money_normalization"] = declaration
    spec["resources"]["transport_graph"]["nodes"][0]["params"]["factor_context"] = "{}"
    message = "Factor context requires schema_version 1"
    with pytest.raises(ValueError, match=f"factor_context.*{message}"):
        cli_module._require_current_run_factors(spec, resume="auto")

    template = normalize_node(
        "toy.node", "toy.population", declaration, money_evidence[-1], money_evidence[1]
    )
    node = compose_module._node(
        {"country": "xx", "resources": {}},
        {
            "id": "toy.node",
            "kernel": "money.normalize@1",
            "population": "toy.population",
            "sources": list(template.sources),
            "inputs": [
                {"entity": item.entity, "columns": list(item.columns)}
                for item in template.inputs
            ],
            "outputs": [
                {
                    "entity": item.entity,
                    "column": item.column,
                    "dtype": item.dtype,
                    "rewrite": item.rewrite,
                }
                for item in template.outputs
            ],
        },
        compose_module.TransportGraphConfig(
            engine_refs={}, create_outputs=template.outputs
        ),
        {},
        resolved_params={**template.params, "factor_context": "{}"},
    )
    context = KernelContext(
        node=node,
        tables={
            "person": pd.DataFrame(
                {
                    "person_id": [1],
                    "interest_income": [3.0],
                    "rental_income": [-2.0],
                    "farm_self_employment_income": [1.0],
                }
            )
        },
        weights={},
        strata=pd.Series([], dtype=object),
        params=node.params,
        rng=np.random.default_rng(0),
        sources={"money_facts": money_evidence[0]},
        frame_metadata={"donor_time_period": 2024},
    )
    with pytest.raises(ValueError, match=message):
        MONEY_NORMALIZE.run(context)


def _composed_money_case(root, evidence, *, fixture_factory=make_composed_fixture):
    """Add real normalization rows to the synthetic composed transport fixture."""
    toy = fixture_factory(root)
    spec = copy.deepcopy(toy.spec)
    resources = spec["resources"]
    with pd.HDFStore(toy.sources["donor_pool"], mode="a") as store:
        store.put("_time_period", pd.Series([2024]), format="table")
    resources["donor_pin"] = donor_pin(toy.sources["donor_pool"])
    declaration = graph_declaration()
    declaration["columns"] = {"person": ["interest_income"]}
    declaration["index_family_factors"] = {
        "capital_income": declaration["index_family_factors"]["capital_income"]
    }
    for name in ("rental_income", "farm_self_employment_income"):
        declaration["pending"]["person"][name] = {
            "awaits": ["a donor that carries it"],
            "reason": "The synthetic donor has no such column.",
        }
    resources.update(
        money_normalization=declaration,
        entry_context=copy.deepcopy(evidence[-1]),
        node_context=copy.deepcopy(evidence[-1]),
        money_selection={DIMENSION: "central"},
        money_feed_pin=evidence[1],
        reference_profiles={
            "schema_version": 1,
            "reference_resource": "target_references",
            "factor_context_resource": "entry_context",
            "dimensions": {},
        },
    )
    blueprint = resources["transport_graph"]
    row = next(
        row for row in blueprint["nodes"] if row["id"] == "nz.transport.currency"
    )
    row.update(
        kernel="money.normalize@1",
        sources=["money_facts"],
        outputs=[
            {
                "entity": "person",
                "column": "interest_income",
                "dtype": "float64",
                "rewrite": True,
            },
            {"entity": "person", "column": "interest_income_cents", "dtype": "int64"},
        ],
        params={
            "declaration": _selection("money_normalization"),
            "declaration_sha256": _selection("money_normalization", "sha256"),
            "factor_context": _selection("node_context"),
            "factor_context_sha256": _selection("node_context", "sha256"),
            "profile_selection": _selection("money_selection"),
            "chronicle_feed_pin": _selection("money_feed_pin"),
        },
    )
    blueprint["sources"].append(
        {"name": "money_facts", "codec": "ledger-consumer-artifact-v1"}
    )
    return toy, spec, {**toy.sources, "money_facts": evidence[0]}


@pytest.fixture(scope="module")
def real_money_case(tmp_path_factory, money_evidence):
    toy, spec, sources = _composed_money_case(
        tmp_path_factory.mktemp("run-entry-real-composition"),
        money_evidence,
        fixture_factory=make_entitlement_fixture,
    )
    resources = spec["resources"]
    resources["variant_money_selection"] = {DIMENSION: "low"}
    resources["variant_context"] = _with_expiry(money_evidence, "low", TODAY)
    row = copy.deepcopy(
        next(
            row
            for row in resources["transport_graph"]["nodes"]
            if row["id"] == "nz.transport.currency"
        )
    )
    row.update(id="nz.v.V1.money.normalize", population="nz.v.V1.money")
    row["params"]["profile_selection"] = _selection("variant_money_selection")
    row["params"]["factor_context"] = _selection("variant_context")
    row["params"]["factor_context_sha256"] = _selection("variant_context", "sha256")
    variant = next(
        scenario
        for scenario in resources["scenarios"]["scenarios"]
        if scenario["id"] == "V1"
    )
    assert variant["knobs"] == {}
    variant["nodes"][:0] = [
        {
            "id": "nz.v.V1.money",
            "kernel": "transport.boundary@1",
            "structural": "filter",
            "base": "nz.create",
            "inputs": [{"entity": "person", "columns": ["age"]}],
        },
        row,
    ]
    return toy, spec, sources


def _factor_context_pairs(entries):
    return {
        (factor, _json(entry)) for group in entries for factor, entry in group.items()
    }


def test_declared_money_entries_equal_real_factory_parameters(real_money_case):
    toy, spec, _ = real_money_case
    graph = compose_module.compose_transport_graph(spec, toy.config)
    expected = []
    selections = []
    for node in graph.nodes:
        if node.kernel != "money.normalize@1":
            continue
        declaration = validate_money_normalization(
            json.loads(node.params["declaration"])
        )
        selection = json.loads(node.params["profile_selection"])
        selections.append(selection)
        _, factors = selected_money_factors(declaration, selection)
        expected.append(
            validate_factor_context(
                json.loads(node.params["factor_context"]),
                [factor for factor in factors.values() if factor is not None],
            )
        )
    assert {DIMENSION: "low"} in selections
    assert _factor_context_pairs(
        entitlement_module.money_node_factor_entries(spec)
    ) == (_factor_context_pairs(expected))


def test_prepared_real_graph_receipt_uses_checked_factor(
    real_money_case, tmp_path, monkeypatch
):
    toy, spec, sources = real_money_case
    captured = Mock(wraps=require_current_factors)
    monkeypatch.setattr(cli_module, "require_current_factors", captured)
    prepared = cli_module.prepare_transport_build(
        spec, sources=sources, engines_by_binding=toy.engines_by_binding, resume="auto"
    )
    result = run_graph(
        compile_graph(cli_module.through(prepared.graph, "nz.v.V1.money.normalize")),
        sources=sources,
        kernels=prepared.registry.kernels,
        store=ContentStore(tmp_path / "store", codecs=prepared.registry.codecs),
    )
    receipt = result.nodes["nz.v.V1.money.normalize"].receipt
    assert receipt["profile"] == "low"
    receipt_ids = {row["factor_id"] for row in receipt["factors"].values()}
    assert receipt_ids == {factor_id("low")}
    checked = [call.args[0] for call in captured.call_args_list]
    assert receipt_ids <= {factor for group in checked for factor in group}
    expected_context = validate_factor_context(
        spec["resources"]["variant_context"], receipt_ids
    )
    assert _factor_context_pairs([expected_context]) <= _factor_context_pairs(checked)
