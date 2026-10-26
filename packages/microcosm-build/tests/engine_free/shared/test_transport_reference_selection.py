"""Selection closure, parked-row compilation and frozen legacy differentials."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from copy import deepcopy
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.transport import compose, takeup_kernels, target_kernels
from microcosm.build.transport.artifact_types import COMPARISON_TYPE
from microcosm.build.transport.reference_selection import (
    exemption_rows,
    selection_resources,
)
from microcosm.graph import (
    ArtifactOutput,
    Capabilities,
    Determinism,
    Graph,
    KernelContext,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.keys import node_key
from test_support.microcosm_build.transport_graph import (
    reference_document,
    reference_row,
    toy_fact,
    toy_frame,
    write_facts,
)
from test_support.paths import paths_for

SHA = "7" * 64
CONFIG = compose.TransportGraphConfig({}, (Owned("person", "person_id", "int64"),))


def _document(names=("used", "parked")):
    return reference_document(
        [
            reference_row(
                name,
                entity="person",
                measure="person_count",
                activation_status="pending" if name == "parked" else "active",
            )
            for name in names
        ]
    )


def _exemption(name="parked", status="pending"):
    return {"name": name, "activation_status": status, "reason": "Awaiting producer."}


def _selection(*rows):
    return {"exempt": list(rows or (_exemption(),))}


def _node(kernel="targets.compile@1", id="reader", **params):
    return {
        "id": id,
        "kernel": kernel,
        "population": "open",
        "params": {
            "country": "xx",
            "references": {"resource": "target_references", "encoding": "json"},
            "references_sha256": {
                "resource": "target_references",
                "encoding": "sha256",
            },
            **params,
        },
    }


def _spec(*nodes):
    return {
        "country": "xx",
        "resources": {
            "transport_graph": {"nodes": list(nodes or (_node(),))},
            "benefit_unit_rule": {"dependent_child": {"max_age": 18}},
            "axiom_rules_bindings": {"engine": {"commit": "a", "wheel_sha256": "b"}},
            "scenarios": {"scenarios": []},
            "precal_references": _document(("precal",)),
            "target_references": _document(),
            "holdout_references": _document(("holdout",)),
            "receipt_contract": {
                "receipts": {
                    "seed_column": "seed",
                    "programs": [
                        {
                            "program": "benefit",
                            "output": "receives_benefit",
                            "judgment_column": "eligible",
                            "payment_column": "payment",
                            "rate": 1,
                        }
                    ],
                    "exclusion_groups": [],
                }
            },
            "reference_selection": {
                "schema_version": 1,
                "resources": {"target_references": _selection()},
            },
        },
    }


def _legacy(namespace, names):
    """Execute only frozen pure definitions, with today's shared dependencies.

    Holding the implementation digest constant below isolates composition's
    node-key contract. Editing an attested kernel module intentionally changes
    its implementation digest, outside the legacy selection-path guarantee.
    """
    path = (
        paths_for("microcosm-build").tests
        / "fixtures/reference_selection_legacy.py.txt"
    )
    tree = ast.parse(path.read_text())
    definitions = [
        ast.get_source_segment(path.read_text(), node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    scope = dict(vars(namespace))
    exec("from __future__ import annotations\n" + "\n\n".join(definitions), scope)
    return scope


def test_resource_read_does_not_select_parked_rows():
    spec = _spec()
    compose.validate_transport_activation(spec)
    node = compose._node(spec, _node(), CONFIG, {})
    assert json.loads(node.params["reference_selection"]) == _selection()
    assert not compose._reference_selection_closure(spec)["target_references"]


def _v2_spec():
    assignment = {
        "id": "assign",
        "kernel": "takeup.assign@1",
        "population": "open",
        "params": {
            "receipt_contract": {
                "resource": "receipt_contract",
                "path": ["receipts"],
                "encoding": "json",
            },
            "receipt_contract_sha256": {
                "resource": "receipt_contract",
                "encoding": "sha256",
            },
        },
    }
    spec = _spec(_node(), assignment)
    spec["resources"]["receipt_contract"]["receipts"] = {
        "contract_version": 2,
        "seed_column": "seed",
        "default_assignment_profile": "central",
        "assignment_knob": "receipt_profile",
        "assignment_profiles": {
            "central": {"draw_key_suffix": "central"},
            "reverse": {"draw_key_suffix": "reverse", "exclusion_order": "reversed"},
        },
        "programs": [
            {
                "program": "benefit",
                "output": "receives_benefit",
                "judgment_columns": ["eligible"],
                "judgment_reduction": "any",
                "payment_column": "payment",
                "strata": [{"target": "used", "filter": "adult"}],
                "outside_strata": "not_assigned",
            }
        ],
        "exclusion_groups": [],
        "unmodelled_programs": [],
    }
    spec["resources"]["scenarios"]["central_knobs"] = {"receipt_profile": "central"}
    spec["resources"]["calibration_bindings"] = {
        "schema_version": 1,
        "required_inputs": {"person": ["age"]},
        "input_domains": {"person": {"age": {"kind": "nonnegative_integer"}}},
        "columns": [
            {
                "entity": "person",
                "name": "adult",
                "predicates": [{"column": "age", "operator": "ge", "value": 18}],
            }
        ],
    }
    return spec, assignment


def test_v2_preflight_checks_default_profile_and_stratum_predicate():
    spec, _ = _v2_spec()
    compose.validate_transport_activation(spec)
    spec["resources"]["scenarios"]["central_knobs"]["receipt_profile"] = "reverse"
    with pytest.raises(
        ValueError, match="default_assignment_profile must equal central_knobs"
    ):
        compose.validate_transport_activation(spec)
    spec["resources"]["scenarios"]["central_knobs"]["receipt_profile"] = "central"
    spec["resources"]["receipt_contract"]["receipts"]["programs"][0]["strata"][0][
        "filter"
    ] = "unbound"
    with pytest.raises(
        ValueError,
        match="Receipt stratum 'unbound' must name calibration_bindings predicates",
    ):
        compose.validate_transport_activation(spec)


def test_v2_node_binds_predicates_and_resolves_declared_scenario_knob():
    spec, assignment = _v2_spec()
    node = compose._node(
        spec, assignment, CONFIG, {}, profile_selection={"receipt_profile": "reverse"}
    )
    assert (
        json.loads(node.params["calibration_bindings"])
        == spec["resources"]["calibration_bindings"]
    )
    assert node.params["calibration_bindings_sha256"] == compose._resource_digest(
        spec, "calibration_bindings"
    )
    assert json.loads(node.params["profile_selection"]) == {
        "receipt_profile": "reverse"
    }


def test_v2_unstratified_preflight_needs_no_predicate_resource():
    spec, assignment = _v2_spec()
    program = spec["resources"]["receipt_contract"]["receipts"]["programs"][0]
    program.pop("strata")
    program.pop("outside_strata")
    program["target"] = "used"
    spec["resources"].pop("calibration_bindings")
    compose.validate_transport_activation(spec)
    node = compose._node(spec, assignment, CONFIG, {})
    assert "calibration_bindings" not in node.params


def test_exempt_selected_passes_mutant_comparison_refuses_before_filtering():
    spec = _spec(_node("takeup.compare@1", "explicit-comparison"))
    with pytest.raises(
        ValueError, match="explicit-comparison.*exempt.*target_references.parked"
    ):
        compose.validate_transport_activation(spec)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("receipt_contract", {"programs": [{"target": "parked"}]}),
        (
            "receipt_contract",
            {"programs": [{"strata": [{"target": "parked", "filter": "age"}]}]},
        ),
        ("bands", {"bands": [{"reference": "parked"}]}),
        ("annualisation_factors", {"parked": 52}),
        ("target", "parked"),
        ("reference", "parked"),
    ],
    ids=[
        "receipt-target",
        "receipt-stratum",
        "quantile-band",
        "annualisation-factor",
        "direct-target",
        "direct-reference",
    ],
)
def test_required_target_exempted_mutant_names_consumer(parameter, value):
    spec = _spec(_node(id="required-consumer", **{parameter: value}))
    with pytest.raises(ValueError, match="required-consumer.*exempt.*parked"):
        compose.validate_transport_activation(spec)


def test_mass_reference_rows_are_required():
    spec = _spec()
    spec["resources"]["mass_references"] = _document(("parked",))
    spec["resources"]["reference_selection"]["resources"]["mass_references"] = (
        _selection()
    )
    with pytest.raises(
        ValueError, match="mass_references.*exempt.*mass_references.parked"
    ):
        compose.validate_transport_activation(spec)


@pytest.mark.parametrize("tier", ["entitlement", "calibration"])
def test_closure_uses_instantiated_scenario_and_variant_rows(tier):
    spec = _spec()
    spec["resources"]["scenarios"]["scenarios"] = [
        {"id": "case", "tier": tier, "knobs": {}}
    ]
    template = "scenario_nodes" if tier == "entitlement" else "variant_nodes"
    spec["resources"]["entitlement_graph"] = {
        "schema_version": 1,
        "nodes": [],
        template: [_node("takeup.compare@1", "compare-{scenario}-{variant}")],
        "scenario_gap": "gap-{scenario}",
        "variant_gap": "gap-{variant}",
    }
    with pytest.raises(ValueError, match="compare-.*case.*exempt.*parked"):
        compose.validate_transport_activation(spec)
    # Explicit nodes replace the parked-row template, matching the factory.
    spec["resources"]["scenarios"]["scenarios"][0]["nodes"] = [_node(id="reader-case")]
    compose.validate_transport_activation(spec)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("active", "Exempt reference target_references.parked is active"),
        ("stale", "Stale exemption for target_references.parked"),
        ("unknown", "Unknown exempt reference target_references.unknown"),
    ],
    ids=["active", "stale", "unknown"],
)
def test_exempt_active_row_and_stale_unknown_exemptions_refuse(mutation, message):
    spec = _spec()
    if mutation == "active":
        spec["resources"]["target_references"]["target_references"][1]["metadata"][
            "activation_status"
        ] = "active"
    elif mutation == "stale":
        spec["resources"]["reference_selection"]["resources"]["target_references"] = (
            _selection(_exemption(status="old"))
        )
    else:
        spec["resources"]["reference_selection"]["resources"]["target_references"] = (
            _selection(_exemption("unknown"))
        )
    with pytest.raises(ValueError, match=message):
        compose.validate_transport_activation(spec)


def test_nonexempt_placeholder_keeps_legacy_gap_message():
    spec = _spec()
    spec["resources"]["reference_selection"]["resources"] = {}
    with pytest.raises(
        ValueError, match="placeholder reference target_references.parked"
    ):
        compose.validate_transport_activation(spec)


def test_absent_selection_preserves_complete_legacy_activation_gap_order():
    spec = _spec()
    spec["resources"].pop("reference_selection")
    spec["resources"]["holdout_references"] = _document(("parked",))
    spec["resources"]["precal_references"] = _document(("parked",))
    # The assigned baseline already visits the reference-resource set in
    # sorted order, including mandatory resources selected by multiple params.
    with pytest.raises(ValueError) as error:
        compose.validate_transport_activation(spec)
    assert str(error.value) == (
        "Transport spec is not activated: "
        "placeholder reference holdout_references.parked; "
        "placeholder reference precal_references.parked; "
        "placeholder reference target_references.parked."
    )


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ({"exempt": [], "ignored": True}, "exactly an exempt list"),
        ({"exempt": "parked"}, "exempt must be a list"),
        (
            {"exempt": [{"name": "parked"}]},
            "declares name, activation_status and reason",
        ),
        (_selection({**_exemption(), "reason": ""}), "reason must be nonempty text"),
        (_selection(_exemption(), _exemption()), "Repeated reference exemption"),
    ],
)
def test_exemption_schema_is_closed(selection, message):
    with pytest.raises(ValueError, match=message):
        exemption_rows(selection)


def test_selection_resource_schema_and_filename_aliases():
    assert selection_resources(
        {"schema_version": 1, "resources": {"a.json": _selection()}}
    ) == {"a": _selection()}
    with pytest.raises(ValueError, match="Repeated reference selection resource"):
        selection_resources(
            {
                "schema_version": 1,
                "resources": {"a": _selection(), "a.json": _selection()},
            }
        )
    with pytest.raises(ValueError, match="schema_version 1 and a resources object"):
        selection_resources({"schema_version": 1, "resources": {}, "ignored": True})


def test_compile_validates_full_document_and_records_exemptions(tmp_path):
    facts = write_facts(
        tmp_path / "facts.jsonl", [toy_fact("used", 3, entity="person")]
    )
    payload, registry = target_kernels.compile_target_surface(
        facts,
        _document(),
        country="xx",
        references_sha256=SHA,
        reference_selection=_selection(),
    )
    surface = target_kernels.decode_target_surface(payload)
    assert [spec.name for spec in registry.specs] == ["used"]
    assert [row["reference"] for row in surface.trace] == ["used"]
    assert surface.reference_selection == _selection()
    with pytest.raises(ValueError, match="Unknown exempt reference references.unknown"):
        target_kernels.compile_target_surface(
            facts,
            _document(),
            country="xx",
            references_sha256=SHA,
            reference_selection=_selection(_exemption("unknown")),
        )


def test_direct_comparison_request_for_exempt_row_refuses(tmp_path):
    node = Node(
        "direct-comparison",
        "takeup.compare@1",
        population="open",
        sources=("facts",),
        params={
            "country": "xx",
            "references": canonical_json(_document()).decode(),
            "references_sha256": SHA,
            "reference_selection": canonical_json(_selection()).decode(),
            "entities": ("person", "household"),
            "weight_entity": "person",
        },
        artifact_outputs=(ArtifactOutput("comparison", COMPARISON_TYPE),),
    )
    context = KernelContext(
        node=node,
        params=node.params,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object),
        rng=np.random.default_rng(0),
        sources={"facts": tmp_path / "not-opened.jsonl"},
        artifacts={},
    )
    with pytest.raises(
        ValueError, match="direct-comparison.*exempt reference references.parked"
    ):
        target_kernels.TAKEUP_COMPARE.run(context)


@pytest.mark.parametrize("request_parked", [False, True])
def test_gap_compares_only_annualisation_selected_rows(tmp_path, request_parked):
    document = _document()
    document["target_references"][0]["measure"] = "is_adult"
    facts = write_facts(
        tmp_path / "facts.jsonl", [toy_fact("used", 3, entity="person")]
    )
    factors = {"used": 52, **({"parked": 52} if request_parked else {})}
    node = Node(
        "explicit-gap",
        "takeup.gap@1",
        population="open",
        sources=("facts",),
        params={
            "country": "xx",
            "references": canonical_json(document).decode(),
            "references_sha256": SHA,
            "reference_selection": canonical_json(_selection()).decode(),
            "entities": ("person", "household", "family"),
            "weight_entity": "household",
            "annualisation_factors": canonical_json(factors).decode(),
            "scenario": "central",
        },
        artifact_outputs=(ArtifactOutput("gap", takeup_kernels.GAP_TYPE),),
    )
    frame = toy_frame()
    context = KernelContext(
        node=node,
        params=node.params,
        tables={entity: frame.table(entity) for entity in node.params["entities"]},
        weights={"household": frame.weights_for("household")},
        strata=frame.strata,
        rng=np.random.default_rng(0),
        sources={"facts": facts},
        artifacts={},
    )
    if request_parked:
        with pytest.raises(
            ValueError, match="explicit-gap.*exempt reference references.parked"
        ):
            takeup_kernels.TAKEUP_GAP.run(context)
    else:
        result = takeup_kernels.TAKEUP_GAP.run(context)
        assert [row["name"] for row in json.loads(result.artifacts["gap"])["gaps"]] == [
            "used"
        ]


@settings(
    max_examples=12,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
@given(
    st.lists(
        st.from_regex(r"parked_[a-z]{1,6}", fullmatch=True),
        min_size=1,
        max_size=5,
        unique=True,
    )
)
def test_exempt_rows_never_enter_surface(tmp_path, names):
    rows = _document(("used",))["target_references"] + [
        reference_row(
            name, entity="person", measure="person_count", activation_status="pending"
        )
        for name in names
    ]
    selection = _selection(*(_exemption(name) for name in names))
    facts = write_facts(
        tmp_path / "facts.jsonl", [toy_fact("used", 3, entity="person")]
    )
    payload, registry = target_kernels.compile_target_surface(
        facts,
        reference_document(rows),
        country="xx",
        references_sha256=SHA,
        reference_selection=selection,
    )
    assert {spec.name for spec in registry.specs} == {"used"}
    assert {
        row["name"]
        for row in target_kernels.decode_target_surface(payload).reference_selection[
            "exempt"
        ]
    } == set(names)


@pytest.mark.parametrize("country", ["xx", "be"])
def test_legacy_selection_path_changed_mutant_frozen_surface_params_and_keys(
    tmp_path, country
):
    if country == "xx":
        document = _document(("used",))
        fact = toy_fact("used", 3, entity="person")
        spec = {"country": country, "resources": {"target_references": document}}
    else:
        # Keep an actual active scalar row from the packaged country verbatim;
        # supply only its synthetic matching fact, never a production feed.
        spec = compose.load_transport_spec(country)
        document = deepcopy(compose.transport_resource(spec, "target_references"))
        document = {
            key: value
            for key, value in document.items()
            if key in target_kernels.REFERENCE_DOCUMENT_KEYS
        }
        reference = next(
            row
            for row in document["target_references"]
            if row["name"] == "spf_finances_pit_total"
        )
        document["target_references"] = [reference]
        fact = toy_fact(reference["name"], 3, entity=reference["entity"])
        fact["period"] = {"type": "tax_year", "value": reference["period"]}
        fact["geography"]["id"] = country.upper()
        for section in ("source", "observed_measure"):
            fact[section]["source_name"] = reference["ledger_selector"]["source_name"]
        fact["observed_measure"]["source_measure_id"] = reference["ledger_selector"][
            "source_measure_id"
        ]
        spec["resources"]["target_references"] = document
    facts = write_facts(tmp_path / "facts.jsonl", [fact])
    legacy_targets = _legacy(
        target_kernels, {"compile_target_surface", "encode_target_surface"}
    )
    before, _ = legacy_targets["compile_target_surface"](
        facts, document, country=country, references_sha256=SHA
    )
    after, _ = target_kernels.compile_target_surface(
        facts, document, country=country, references_sha256=SHA
    )
    assert before == after
    legacy_compose = _legacy(compose, {"_node", "_reference_activation_gaps"})
    row = _node()
    row["params"]["country"] = country
    before_node = legacy_compose["_node"](spec, row, CONFIG, {})
    after_node = compose._node(spec, row, CONFIG, {})
    assert before_node.params == after_node.params
    assert "reference_selection" not in after_node.params
    assert list(
        legacy_compose["_reference_activation_gaps"](document, "target_references")
    ) == list(compose._reference_activation_gaps(document, "target_references"))
    keys = []
    for node in (before_node, after_node):
        graph = Graph(
            country,
            (SourceRef("seed", "raw-bytes-v1"),),
            (
                Node(
                    "open",
                    "fixture.create@1",
                    sources=("seed",),
                    structural=StructuralDelta.CREATE,
                    outputs=CONFIG.create_outputs,
                ),
                node,
            ),
        )
        keys.append(
            node_key(
                compile_graph(graph),
                node.id,
                {"open": SHA},
                SHA,
                {},
                kernel_capabilities=Capabilities(Determinism.DETERMINISTIC),
            )
        )
    assert keys[0] == keys[1]


@pytest.mark.parametrize(
    ("module_name", "kernel_name"),
    [
        ("column_kernels", "TAKEUP_ASSIGN"),
        ("gate_kernels", "GateBatteryKernel"),
        ("geography_kernels", "GEOGRAPHY_SUPPORT_FROM_FACTS"),
        ("population_kernels", "TRANSPORT_CREATE"),
        ("population_kernels", "TRANSPORT_QUANTILE_MAP"),
        ("takeup_kernels", "TAKEUP_GAP"),
        ("target_kernels", "TARGETS_COMPILE"),
        ("target_kernels", "TARGETS_PROBLEM"),
        ("target_kernels", "TAKEUP_COMPARE"),
        ("terminal_kernels", "DIAGNOSTICS_CALIBRATION"),
    ],
)
def test_surface_consumers_hash_selection_and_profile_modules(
    monkeypatch, module_name, kernel_name
):
    # source_hash reads only each explicitly supplied module's bytes. Cover
    # every kernel reaching decode_target_surface, including through helpers.
    module = importlib.import_module(f"microcosm.build.transport.{module_name}")
    hashed = set()

    def capture(*objects, **kwargs):
        for obj in objects:
            defining_module = (
                obj if isinstance(obj, ModuleType) else inspect.getmodule(obj)
            )
            hashed.add(defining_module.__name__)
        return SHA

    monkeypatch.setattr(module, "source_hash", capture)
    kernel = getattr(module, kernel_name)
    if kernel_name == "GateBatteryKernel":
        kernel = kernel({})
    kernel.implementation_hash()
    assert {
        "microcosm.build.reference_profiles",
        "microcosm.build.transport.reference_selection",
    } <= hashed


def _compiled_surface_with_exemption(tmp_path):
    facts = write_facts(
        tmp_path / "facts.jsonl", [toy_fact("used", 3, entity="person")]
    )
    payload, _ = target_kernels.compile_target_surface(
        facts,
        _document(),
        country="xx",
        references_sha256=SHA,
        reference_selection=_selection(),
    )
    return payload


def test_surface_decode_refuses_active_exemption(tmp_path):
    document = json.loads(_compiled_surface_with_exemption(tmp_path))
    document["reference_selection"] = _selection(_exemption(status="active"))
    with pytest.raises(ValueError, match="cannot exempt an active reference"):
        target_kernels.decode_target_surface(canonical_json(document))


def test_surface_decode_refuses_exempt_compiled_reference(tmp_path):
    document = json.loads(_compiled_surface_with_exemption(tmp_path))
    document["reference_selection"] = _selection(_exemption("used"))
    with pytest.raises(ValueError, match="An exempt reference entered"):
        target_kernels.decode_target_surface(canonical_json(document))


def test_resolution_filters_transformed_exempt_reference(tmp_path):
    from microcosm.build.reference_profiles import (
        reference_document_sha256,
        resolve_reference_profiles,
    )

    def selector(key, category=None):
        value = {
            "aggregate_fact_key": key,
            "source_name": "publisher",
            "source_measure_id": "count",
            "source_concept": "people",
            "entity_name": "person",
            "record_set_id": "people",
            "period_type": "calendar_year",
            "period_value": 2023,
            "geography_level": "region",
            "geography_id": "a",
            "geography_vintage": "v",
            "dimensions": [] if category is None else ["category"],
        }
        if category is not None:
            value["dimension_values"] = {"category": category}
        return value

    document = _document()
    parked = document["target_references"][1]
    parked.pop("ledger_fact_key", None)
    parked["ledger_selector"] = selector("atom", 1)
    transformation = {
        "reference_names": ["parked"],
        "operation": "allocated_residual_by_ratio",
        "value_operands": [
            {"role": "base", "expected_member_count": 1},
            {"role": "total", **selector("total")},
            {
                "role": "partition",
                "member_selectors": [selector("atom", 1)],
                "allocation_member_keys": ["atom"],
            },
        ],
        "metadata": {"allocation_ruling": "Declared allocation."},
    }
    declaration = {
        "schema_version": 1,
        "dimensions": {
            "allocation": {
                "default": "allocated",
                "profiles": {"allocated": {"transformations": [transformation]}},
            }
        },
    }
    effective, _, receipt = resolve_reference_profiles(document, declaration, {})
    assert receipt["transformed_names"] == ["parked"]
    facts = write_facts(
        tmp_path / "facts.jsonl", [toy_fact("used", 3, entity="person")]
    )
    payload, registry = target_kernels.compile_target_surface(
        facts,
        effective,
        country="xx",
        references_sha256=reference_document_sha256(effective),
        reference_resolution=receipt,
        reference_selection=_selection(),
    )
    surface = target_kernels.decode_target_surface(payload)
    assert [spec.name for spec in registry.specs] == ["used"]
    assert surface.reference_resolution == receipt
    assert surface.references_sha256 == receipt["effective_references_sha256"]


def test_v2_activation_refuses_unsupported_calibration_input_domain():
    spec, _ = _v2_spec()
    domain = spec["resources"]["calibration_bindings"]["input_domains"]["person"]["age"]
    domain["kind"] = "integer"
    with pytest.raises(
        ValueError, match="Calibration input person.age has an unsupported domain"
    ):
        compose.validate_transport_activation(spec)


@pytest.mark.parametrize("knob_scope", ["central", "scenario"])
def test_v2_assignment_node_key_uses_only_assignment_knob(knob_scope):
    spec, assignment = _v2_spec()
    spec["resources"]["scenarios"]["central_knobs"]["unrelated"] = "before"
    changed = deepcopy(spec)
    before_selection = {"unrelated": "before"} if knob_scope == "scenario" else {}
    after_selection = {"unrelated": "after"} if knob_scope == "scenario" else {}
    if knob_scope == "central":
        changed["resources"]["scenarios"]["central_knobs"]["unrelated"] = "after"

    def assignment_key(selected_spec, selection):
        node = compose._node(
            selected_spec, assignment, CONFIG, {}, profile_selection=selection
        )
        graph = Graph(
            "xx",
            (SourceRef("seed", "raw-bytes-v1"),),
            (
                Node(
                    "open",
                    "fixture.create@1",
                    sources=("seed",),
                    structural=StructuralDelta.CREATE,
                    outputs=CONFIG.create_outputs,
                ),
                node,
            ),
        )
        return node_key(
            compile_graph(graph),
            node.id,
            {"open": SHA},
            SHA,
            {},
            kernel_capabilities=Capabilities(Determinism.DETERMINISTIC),
        )

    baseline_key = assignment_key(spec, before_selection)
    assert baseline_key == assignment_key(changed, after_selection)
    assert baseline_key != assignment_key(
        changed, {**after_selection, "receipt_profile": "reverse"}
    )


def test_v2_assignment_node_refuses_unresolved_assignment_knob():
    spec, assignment = _v2_spec()
    del spec["resources"]["scenarios"]["central_knobs"]["receipt_profile"]
    with pytest.raises(ValueError, match="'receipt_profile' is not a resolved knob"):
        compose._node(spec, assignment, CONFIG, {}, profile_selection={})


def test_full_effective_resolution_hash_survives_exemption_filtering(tmp_path):
    from microcosm.build.reference_profiles import (
        reference_document_sha256,
        resolve_reference_profiles,
    )

    document = _document()
    declaration = {
        "schema_version": 1,
        "reference_resource": "target_references",
        "dimensions": {},
    }
    # A resolver receipt is validated against the full document before parked
    # rows leave compilation, never against a reconstructed active subset.
    effective, _, receipt = resolve_reference_profiles(document, declaration, {})
    facts = write_facts(
        tmp_path / "facts.jsonl", [toy_fact("used", 3, entity="person")]
    )
    payload, _ = target_kernels.compile_target_surface(
        facts,
        effective,
        country="xx",
        references_sha256=reference_document_sha256(effective),
        reference_resolution=receipt,
        reference_selection=_selection(),
    )
    surface = target_kernels.decode_target_surface(payload)
    assert surface.references_sha256 == receipt["effective_references_sha256"]
    assert surface.reference_resolution == receipt


def test_v2_activation_refuses_cyclic_exclusion_groups():
    spec, _ = _v2_spec()
    contract = spec["resources"]["receipt_contract"]["receipts"]
    second = deepcopy(contract["programs"][0])
    second.update(program="second", output="receives_second")
    second["strata"][0]["target"] = "second_target"
    contract["programs"].append(second)
    contract["exclusion_groups"] = [
        ["receives_benefit", "receives_second"],
        ["receives_second", "receives_benefit"],
    ]
    with pytest.raises(ValueError, match="cyclic order"):
        compose.validate_transport_activation(spec)
