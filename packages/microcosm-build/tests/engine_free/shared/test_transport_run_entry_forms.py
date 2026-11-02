"""Run entry resolves every money parameter form as composition does."""

from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from microcosm.build.money_normalization import (
    selected_money_factors,
    validate_money_normalization,
)
from microcosm.build.reference_factors import (
    resolve_reference_factor,
    validate_factor_context,
)
from microcosm.build.transport import cli, compose, entitlement
from microcosm.graph import ContentStore, compile_graph, run_graph
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.money_normalization import (
    DIMENSION,
    factor_id,
    graph_declaration,
    money_factor_context,
    write_money_feed,
)
from test_support.microcosm_build.transport_entitlement import make_entitlement_fixture
from test_support.microcosm_build.transport_population import donor_pin

TODAY = date(2026, 10, 24)


def _json(d):
    return canonical_json(d).decode("utf-8")


def sel(resource, encoding="json"):
    return {"resource": resource, "encoding": encoding}


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    path, pin, identity, facts = write_money_feed(tmp_path_factory.mktemp("feed"))
    return path, pin, identity, facts, money_factor_context(identity, facts)


@pytest.fixture(autouse=True)
def today(monkeypatch):
    monkeypatch.setattr(cli, "_today", lambda: TODAY)


def with_expiry(evidence, profile, until):
    _, _, identity, facts, context = evidence
    context = copy.deepcopy(context)
    entry = next(
        r for r in context["factors"] if r["declaration"]["id"] == factor_id(profile)
    )
    entry["declaration"]["validity"]["valid_until"] = until.isoformat()
    factor, receipt = resolve_reference_factor(entry["declaration"], facts, identity)
    entry.update(operational_factor=factor, receipt=receipt)
    return context


@pytest.fixture(scope="module")
def base_case(tmp_path_factory, evidence):
    toy = make_entitlement_fixture(tmp_path_factory.mktemp("composed"))
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
    row = next(r for r in blueprint["nodes"] if r["id"] == "nz.transport.currency")
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
            "declaration": sel("money_normalization"),
            "declaration_sha256": sel("money_normalization", "sha256"),
            "factor_context": sel("node_context"),
            "factor_context_sha256": sel("node_context", "sha256"),
            "profile_selection": sel("money_selection"),
            "chronicle_feed_pin": sel("money_feed_pin"),
        },
    )
    blueprint["sources"].append(
        {"name": "money_facts", "codec": "ledger-consumer-artifact-v1"}
    )
    resources["variant_context"] = copy.deepcopy(evidence[-1])
    variant_row = copy.deepcopy(row)
    variant_row.update(id="nz.v.V1.money.normalize", population="nz.v.V1.money")
    variant_row["params"]["factor_context"] = sel("variant_context")
    variant_row["params"]["factor_context_sha256"] = sel("variant_context", "sha256")
    variant = next(s for s in resources["scenarios"]["scenarios"] if s["id"] == "V1")
    variant["nodes"][:0] = [
        {
            "id": "nz.v.V1.money",
            "kernel": "transport.boundary@1",
            "structural": "filter",
            "base": "nz.create",
            "inputs": [{"entity": "person", "columns": ["age"]}],
        },
        variant_row,
    ]
    return toy, spec, {**toy.sources, "money_facts": evidence[0]}


def parts(spec):
    res = spec["resources"]
    variant = next(s for s in res["scenarios"]["scenarios"] if s["id"] == "V1")
    vrow = next(r for r in variant["nodes"] if r["id"] == "nz.v.V1.money.normalize")
    srow = next(
        r for r in res["transport_graph"]["nodes"] if r["id"] == "nz.transport.currency"
    )
    return res, variant, vrow, srow


def form_knobs(res, v, r, s):
    v["knobs"][DIMENSION] = "low"
    r["params"]["profile_selection"] = {"scenario": "knobs", "encoding": "json"}
    return "low"


def form_knobs_path(res, v, r, s):
    v["knobs"]["nest"] = {DIMENSION: "high"}
    r["params"]["profile_selection"] = {
        "scenario": "knobs",
        "path": ["nest"],
        "encoding": "json",
    }
    return "high"


def form_row_path(res, v, r, s):
    v["money_selection"] = {DIMENSION: "high"}
    r["params"]["profile_selection"] = {
        "scenario": "row",
        "path": ["money_selection"],
        "encoding": "json",
    }
    return "high"


def form_row_key(res, v, r, s):
    v["money_selection"] = {DIMENSION: "low"}
    r["params"]["profile_selection"] = {
        "scenario": "money_selection",
        "encoding": "json",
    }
    return "low"


def form_inline(res, v, r, s):
    r["params"]["profile_selection"] = {DIMENSION: "high"}
    return "high"


def form_inline_empty(res, v, r, s):
    r["params"]["profile_selection"] = {}
    return "central"


def form_res_path(res, v, r, s):
    res["bundle"] = {"v": {DIMENSION: "low"}}
    r["params"]["profile_selection"] = {
        "resource": "bundle",
        "path": ["v"],
        "encoding": "json",
    }
    return "low"


def form_res_text(res, v, r, s):
    res["bundle"] = {"t": _json({DIMENSION: "low"})}
    r["params"]["profile_selection"] = {"resource": "bundle", "path": ["t"]}
    return "low"


def form_literal(res, v, r, s):
    r["params"]["profile_selection"] = _json({DIMENSION: "high"})
    return "high"


def form_token(res, v, r, s):
    res["sel_V1"] = {DIMENSION: "low"}
    r["params"]["profile_selection"] = sel("sel_{variant}")
    return "low"


def form_suffix(res, v, r, s):
    res["sel.json"] = {DIMENSION: "low"}
    r["params"]["profile_selection"] = sel("sel.json")
    return "low"


def form_scenario_text(res, v, r, s):
    v["sel_text"] = _json({DIMENSION: "high"})
    r["params"]["profile_selection"] = {"scenario": "sel_text"}
    return "high"


def form_ctx_row(res, v, r, s):
    v["ctx"] = copy.deepcopy(res["variant_context"])
    r["params"]["profile_selection"] = _json({DIMENSION: "low"})
    r["params"]["factor_context"] = {
        "scenario": "row",
        "path": ["ctx"],
        "encoding": "json",
    }
    r["params"]["factor_context_sha256"] = {
        "scenario": "row",
        "path": ["ctx"],
        "encoding": "sha256",
    }
    return "low"


def form_ctx_inline(res, v, r, s):
    r["params"]["profile_selection"] = _json({DIMENSION: "high"})
    r["params"]["factor_context"] = copy.deepcopy(res["variant_context"])
    r["params"]["factor_context_sha256"] = "a" * 64
    return "high"


def form_decl_token(res, v, r, s):
    res["money_normalization_V1"] = copy.deepcopy(res["money_normalization"])
    r["params"]["profile_selection"] = _json({DIMENSION: "low"})
    r["params"]["declaration"] = sel("money_normalization_{variant}")
    r["params"]["declaration_sha256"] = sel("money_normalization_{variant}", "sha256")
    return "low"


def form_decl_path(res, v, r, s):
    res["bundle"] = {"d": copy.deepcopy(res["money_normalization"])}
    r["params"]["profile_selection"] = _json({DIMENSION: "high"})
    r["params"]["declaration"] = {
        "resource": "bundle",
        "path": ["d"],
        "encoding": "json",
    }
    r["params"]["declaration_sha256"] = sel("bundle", "sha256")
    return "high"


def form_skeleton_literal(res, v, r, s):
    s["params"]["profile_selection"] = _json({DIMENSION: "low"})
    return "central"


def form_skeleton_path(res, v, r, s):
    res["bundle"] = {"v": {DIMENSION: "high"}}
    s["params"]["profile_selection"] = {
        "resource": "bundle",
        "path": ["v"],
        "encoding": "json",
    }
    return "central"


def form_skeleton_text(res, v, r, s):
    res["bundle"] = {"t": _json({DIMENSION: "high"})}
    s["params"]["profile_selection"] = {"resource": "bundle", "path": ["t"]}
    return "central"


FORMS = [
    form_knobs,
    form_knobs_path,
    form_row_path,
    form_row_key,
    form_inline,
    form_inline_empty,
    form_res_path,
    form_res_text,
    form_literal,
    form_token,
    form_suffix,
    form_scenario_text,
    form_ctx_row,
    form_ctx_inline,
    form_decl_token,
    form_decl_path,
    form_skeleton_literal,
    form_skeleton_path,
    form_skeleton_text,
]


def pairs(groups):
    return sorted((k, _json(v)) for g in groups for k, v in g.items())


@pytest.mark.parametrize("form", FORMS, ids=lambda f: f.__name__)
def test_walk_equals_composed_nodes_and_receipts(
    base_case, form, tmp_path, monkeypatch
):
    toy, base, sources = base_case
    spec = copy.deepcopy(base)
    res, variant, vrow, srow = parts(spec)
    profile = form(res, variant, vrow, srow)
    calls = []
    original = entitlement._EntitlementFactory.__call__
    monkeypatch.setattr(
        entitlement._EntitlementFactory,
        "__call__",
        lambda self, *a: calls.append(1) or original(self, *a),
    )
    walk = entitlement.money_node_factor_entries(spec)
    graph = compose.compose_transport_graph(spec, toy.config)
    assert calls, "real factory did not run"
    expected = []
    for node in graph.nodes:
        if node.kernel != "money.normalize@1":
            continue
        declaration = validate_money_normalization(
            json.loads(node.params["declaration"])
        )
        _, factors = selected_money_factors(
            declaration, json.loads(node.params["profile_selection"])
        )
        expected.append(
            validate_factor_context(
                json.loads(node.params["factor_context"]),
                sorted({f for f in factors.values() if f}),
            )
        )
    assert [sorted(g) for g in walk] == [sorted(g) for g in expected]
    assert pairs(walk) == pairs(expected)
    assert len(walk) == 2
    prepared = cli.prepare_transport_build(
        spec, sources=sources, engines_by_binding=toy.engines_by_binding, resume="auto"
    )
    result = run_graph(
        compile_graph(cli.through(prepared.graph, "nz.v.V1.money.normalize")),
        sources=sources,
        kernels=prepared.registry.kernels,
        store=ContentStore(tmp_path / "store", codecs=prepared.registry.codecs),
    )
    receipt = result.nodes["nz.v.V1.money.normalize"].receipt
    assert receipt["profile"] == profile
    assert {row["factor_id"] for row in receipt["factors"].values()} == set(walk[1])


@pytest.mark.parametrize(
    "form",
    [
        form_knobs,
        form_row_path,
        form_inline,
        form_res_path,
        form_res_text,
        form_literal,
        form_token,
        form_scenario_text,
    ],
    ids=lambda f: f.__name__,
)
def test_expiring_the_selected_factor_refuses_and_the_other_passes(
    base_case, evidence, form
):
    _, base, _ = base_case
    spec = copy.deepcopy(base)
    res, variant, vrow, srow = parts(spec)
    profile = form(res, variant, vrow, srow)
    res["variant_context"] = with_expiry(evidence, profile, TODAY - timedelta(days=1))
    with pytest.raises(ValueError, match=factor_id(profile)):
        cli._require_current_run_factors(spec, resume="auto")
    other = "central" if profile != "central" else "low"
    res["variant_context"] = with_expiry(evidence, other, TODAY - timedelta(days=1))
    cli._require_current_run_factors(spec, resume="auto")


NAMES = ("declaration", "profile_selection", "factor_context")
BAD = {
    "missing": None,
    "int": 3,
    "bool": True,
    "float": 1.5,
    "list": ["a"],
    "list_map": [{"a": 1}],
    "empty_list": [],
    "bad_json": "not JSON",
    "json_list": "[]",
    "json_num": "3",
    "json_null": "null",
    "json_nan": "NaN",
    "empty_str": "",
    "binding": {"binding": "toy", "field": "engine_ref"},
    "binding_only": {"binding": "toy"},
    "prepared": {"prepared": "engine_refs"},
    "prepared_fp": {"prepared": "fingerprint"},
    "prepared_extra": {"prepared": "engine_refs", "x": 1},
    "missing_resource": {"resource": "nope", "encoding": "json"},
    "resource_nonstr": {"resource": 5},
    "resource_bad_path": {
        "resource": "money_selection",
        "path": ["nope"],
        "encoding": "json",
    },
    "resource_path_int": {"resource": "money_selection", "path": 5, "encoding": "json"},
    "resource_encoding_list": {"resource": "money_selection", "encoding": ["a"]},
    "resource_encoding_bad": {"resource": "money_selection", "encoding": "yaml"},
    "resource_sha": {"resource": "money_selection", "encoding": "sha256"},
    "resource_value_object": {"resource": "money_selection"},
    "scenario_missing_key": {"scenario": "nope", "encoding": "json"},
    "scenario_object_value": {"scenario": "knobs"},
    "scenario_sha": {"scenario": "knobs", "encoding": "sha256"},
    "scenario_unhashable": {"scenario": ["a"], "encoding": "json"},
    "scenario_extra": {"scenario": "knobs", "encoding": "json", "x": 1},
}


@pytest.mark.parametrize("placement", ["skeleton", "entitlement"])
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("case", sorted(BAD))
def test_every_bad_parameter_refuses_with_node_and_parameter(
    base_case, placement, name, case, tmp_path, monkeypatch, capsys
):
    _, base, _ = base_case
    spec = copy.deepcopy(base)
    res, variant, vrow, srow = parts(spec)
    row = srow if placement == "skeleton" else vrow
    value = BAD[case]
    if case == "missing":
        del row["params"][name]
    else:
        row["params"][name] = value
    node_id = row["id"]
    with pytest.raises(ValueError) as refusal:
        cli._require_current_run_factors(spec, resume="auto")
    message = str(refusal.value)
    assert node_id in message and name in message, message
    monkeypatch.setattr(cli, "load_transport_spec", lambda source: spec)
    for owner, attribute in (
        (cli, "validate_transport_activation"),
        (cli, "build_transport_registry"),
        (cli, "prepare_create_outputs"),
    ):
        monkeypatch.setattr(
            owner, attribute, Mock(side_effect=AssertionError(attribute))
        )
    out = tmp_path / "out"
    assert cli.main(["--country", "xx", "--out", str(out), "--resume", "auto"]) == 1
    error = capsys.readouterr().err
    assert (
        error.startswith("transport build refused: Money node ")
        and node_id in error
        and name in error
    ), error
    assert not out.exists()


@pytest.mark.parametrize("placement", ["skeleton", "entitlement"])
@pytest.mark.parametrize(
    "case",
    [
        "empty_declaration",
        "unknown_profile",
        "unhashable_profile",
        "null_profile",
        "missing_factor",
        "context_schema",
        "params_not_object",
        "tampered_receipt",
    ],
)
def test_validator_refusals_name_node(base_case, evidence, placement, case):
    _, base, _ = base_case
    spec = copy.deepcopy(base)
    res, variant, vrow, srow = parts(spec)
    row = srow if placement == "skeleton" else vrow
    ctx_name = row["params"]["factor_context"]["resource"]
    label = {
        "empty_declaration": "declaration",
        "unknown_profile": "profile_selection",
        "unhashable_profile": "profile_selection",
        "null_profile": "profile_selection",
        "missing_factor": "factor_context",
        "context_schema": "factor_context",
        "params_not_object": "declaration",
        "tampered_receipt": "factor_context",
    }[case]
    if case == "empty_declaration":
        res["bad"] = {}
        row["params"]["declaration"] = sel("bad")
    elif case == "unknown_profile":
        row["params"]["profile_selection"] = _json({DIMENSION: "nope"})
    elif case == "unhashable_profile":
        row["params"]["profile_selection"] = _json({DIMENSION: ["a"]})
    elif case == "null_profile":
        row["params"]["profile_selection"] = _json({DIMENSION: None})
    elif case == "missing_factor":
        res[ctx_name]["factors"] = [
            f
            for f in res[ctx_name]["factors"]
            if f["declaration"]["id"] != factor_id("central")
        ]
    elif case == "context_schema":
        res[ctx_name]["schema_version"] = 2
    elif case == "params_not_object":
        row["params"] = "x"
    elif case == "tampered_receipt":
        for f in res[ctx_name]["factors"]:
            f["receipt"]["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError) as refusal:
        cli._require_current_run_factors(spec, resume="auto")
    message = str(refusal.value)
    assert row["id"] in message and label in message, message
