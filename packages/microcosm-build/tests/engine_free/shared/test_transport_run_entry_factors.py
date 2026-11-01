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

from microcosm.build.money_normalization import selected_money_factors
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
from test_support.microcosm_build.transport_graph import (
    reference_document,
    reference_row,
)
from test_support.microcosm_build.transport_population import write_population_sources

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


def _money_spec(context, *, scenarios=()):
    return {
        "country": "xx",
        "resources": {
            "money_normalization": graph_declaration(),
            CONTEXT_RESOURCE: copy.deepcopy(context),
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
                "scenarios": copy.deepcopy(list(scenarios)),
            },
        },
    }


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


@pytest.mark.parametrize("with_scenarios", [False, True])
def test_identity_money_families_need_no_factor_context(monkeypatch, with_scenarios):
    declaration = graph_declaration()
    for family in declaration["index_family_factors"]:
        declaration["index_family_factors"][family] = {
            "identity": True,
            "reason": "The toy family keeps its donor basis.",
        }
    resources = {"money_normalization": declaration}
    if with_scenarios:
        resources["scenarios"] = {
            "scenarios": [
                {"id": profile, "knobs": {DIMENSION: profile}} for profile in PROFILES
            ]
        }
    captured = Mock()
    monkeypatch.setattr(cli_module, "require_current_factors", captured)
    cli_module._require_current_run_factors(
        {"country": "xx", "resources": resources}, resume="auto"
    )
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
    for name in ("reference_profile_views", "selected_money_factors", "_today"):
        step = Mock(side_effect=AssertionError(f"require must return before {name}."))
        monkeypatch.setattr(cli_module, name, step)
        steps.append(step)
    cli_module._require_current_run_factors(spec, resume="require")
    for step in steps:
        step.assert_not_called()


def test_each_scenario_run_entry_selection_equals_kernel_selection(
    money_evidence, monkeypatch
):
    path, pin, _, _, context = money_evidence
    scenarios = [
        {"id": profile, "knobs": {DIMENSION: profile, "unrelated": "kept"}}
        for profile in PROFILES
    ]
    spec = _money_spec(context, scenarios=scenarios)
    declaration = spec["resources"]["money_normalization"]
    checked = []

    def select_for_entry(document, selection):
        result = selected_money_factors(document, selection)
        checked.append((copy.deepcopy(selection), result))
        return result

    monkeypatch.setattr(cli_module, "selected_money_factors", select_for_entry)
    cli_module._require_current_run_factors(spec, resume="auto")
    assert checked[0] == ({}, selected_money_factors(declaration, {}))
    assert len(checked) == len(scenarios) + 1
    selected_in_kernel = []

    def select_for_kernel(document, selection):
        result = selected_money_factors(document, selection)
        selected_in_kernel.append((copy.deepcopy(selection), result))
        return result

    monkeypatch.setattr(
        money_kernels_module, "selected_money_factors", select_for_kernel
    )
    for scenario, entry_choice in zip(scenarios, checked[1:], strict=True):
        knobs = scenario["knobs"]
        # The entitlement factory uses this parameter resolver for scenario
        # selectors before handing the resolved params to compose._node.
        profile_selection = entitlement_module._parameter(
            spec, None, {"scenario": "knobs", "encoding": "json"}, scenario
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
        assert json.loads(node.params["profile_selection"]) == knobs
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
        expected = selected_money_factors(declaration, knobs)
        assert entry_choice == (knobs, expected)
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


KNOBS = st.fixed_dictionaries(
    {},
    optional={
        DIMENSION: st.sampled_from(PROFILES),
        "index": st.sampled_from(PROFILES),
    },
)


@settings(
    max_examples=24,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)
@example(knob_sets=[])
@example(knob_sets=[{DIMENSION: "low", "index": "low"}, {DIMENSION: "high"}])
@given(knob_sets=st.lists(KNOBS, max_size=6))
def test_checked_factors_equal_exact_reference_and_money_union(
    reference_and_money_spec, monkeypatch, knob_sets
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
        for selection in knob_sets
        for name, value in selection.items()
    )
    resources["scenarios"]["scenarios"] = [
        {"id": f"variant-{index}", "knobs": knobs}
        for index, knobs in enumerate(knob_sets)
    ]
    expected = {}
    for view in compose_module.reference_profile_views(spec):
        expected.update(view[2].get("selected_factors", {}))
    money_ids = set()
    for selection in [{}, *knob_sets]:
        _, families = selected_money_factors(
            resources["money_normalization"], selection
        )
        money_ids.update(value for value in families.values() if value is not None)
    expected.update(validate_factor_context(resources[CONTEXT_RESOURCE], money_ids))
    captured = Mock()
    with monkeypatch.context() as patch:
        patch.setattr(cli_module, "require_current_factors", captured)
        cli_module._require_current_run_factors(spec, resume="auto")
    captured.assert_called_once_with(expected, today=TODAY)


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
    with pytest.raises(ValueError, match=factor_id("low")):
        cli_module._require_current_run_factors(spec, resume="auto")
