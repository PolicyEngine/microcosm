"""A diagnostic cannot turn missing semantic evidence into population readiness."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.concept_coverage import (
    build_concept_coverage,
    validate_concept_coverage,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.adapters.axiom import AxiomEngine
from microcosm.frame.rules import EngineInput, InputInventory

ADDRESS = "zz:statutes/example#input.earnings"
FINGERPRINTS = [
    {"role": "entry_module", "name": "example.yaml", "sha256": "a" * 64},
    {
        "role": "rulespec_root_yaml_and_toolchain",
        "name": "rulespec-zz",
        "sha256": "b" * 64,
    },
]


class InventoryProvider:
    """Protocol fixture only; no tax calculation or replacement Axiom evaluator."""

    def input_inventory(self):
        return InputInventory(
            inputs=(
                EngineInput(
                    name="earnings",
                    entity="person",
                    engine_entity="Person",
                    canonical_request_name=ADDRESS,
                    request_names=(ADDRESS, "zz:statutes/another#input.earnings"),
                ),
            ),
            fingerprints=tuple(deepcopy(FINGERPRINTS)),
            mapped_entities=("person", "household"),
            entity_discovery=(
                {
                    "entity": "person",
                    "engine_entity": "Person",
                    "status": "complete",
                    "root_input_count": 1,
                },
                {
                    "entity": "household",
                    "engine_entity": "Household",
                    "status": "no_derived_program",
                    "root_input_count": None,
                },
            ),
            runtime={
                "engine": "protocol_fixture",
                "wrapper_distribution_version": None,
                "native_distribution_version": None,
                "core_version": None,
                "python": None,
                "platform": None,
                "numpy": None,
            },
        )


@pytest.fixture
def dataset():
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [1],
                    "person_household_id": [1],
                    "earnings": [100.0],
                    "wealth_for_future_reforms": [200.0],
                }
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([1.0]), WeightKind.DESIGN)},
    )


def column_binding(name="earnings"):
    return {
        "input": {
            "entity": "person",
            "engine_entity": "Person",
            "name": "earnings",
            "canonical_request_name": ADDRESS,
        },
        "artifact_fingerprints": deepcopy(FINGERPRINTS),
        "column": {"entity": "person", "name": name},
    }


def fact_binding(relationship="proxy"):
    scope = {
        "statistic": "total",
        "entity": "person",
        "universe": "resident workers",
        "unit": "EUR",
        "geography": "BE",
        "period": "income-year:2025",
        "measurement_kind": "flow",
        "reference_instant": None,
        "accounting_basis": "income_year",
        "entity_definition": {
            "uri": "https://example.org/entities",
            "sha256": "d" * 64,
            "locator": "Person definition",
        },
        "universe_definition": {
            "uri": "https://example.org/universes",
            "sha256": "e" * 64,
            "locator": "Resident workers definition",
        },
    }
    return {
        "input": {
            "entity": "person",
            "engine_entity": "Person",
            "name": "earnings",
            "canonical_request_name": ADDRESS,
        },
        "artifact_fingerprints": deepcopy(FINGERPRINTS),
        "asserted_relationship": relationship,
        "source": {
            "fact_id": "chronicle:example/fact",
            "concept_id": "publisher:gross-earnings",
            "vintage": "publication:2026-01-01",
            "artifact_sha256": "b" * 64,
            "scope": scope,
        },
        "target": {
            "concept_id": None,
            "legal_vintage": "rulespec:revision-1",
            "scope": deepcopy(scope),
        },
        "transformation": {
            "kind": "identity",
            "description": "This is an authored identity claim, not verified equivalence.",
        },
        "evidence": [
            {
                "uri": "https://example.org/publisher/definition",
                "sha256": "c" * 64,
                "locator": "Table 1 definition",
                "claim": "Publisher defines gross earnings for resident workers.",
            }
        ],
    }


def test_missing_metadata_and_no_dataset_remain_unknown():
    report = build_concept_coverage(InventoryProvider())
    item = report["inputs"][0]
    assert set(item["metadata"].values()) == {None}
    assert item["column_status"] == "unassessed"
    assert report["dataset"]["status"] == "not_supplied"
    assert report["readiness"] == {
        "certified": False,
        "population_schema_ready": False,
        "reason": "diagnostic_only",
    }
    assert report["fact_bindings"] == []
    validate_concept_coverage(report)


def test_matching_names_do_not_automatically_bind_columns(dataset):
    report = build_concept_coverage(InventoryProvider(), dataset=dataset)
    assert report["dataset"]["status"] == "schema_inspected"
    assert report["inputs"][0]["column_status"] == "unassessed"
    assert report["column_bindings"] == []


@pytest.mark.parametrize(
    "name,status", [("earnings", "present"), ("missing", "absent")]
)
def test_only_explicit_binding_can_measure_column_presence(dataset, name, status):
    report = build_concept_coverage(
        InventoryProvider(), dataset=dataset, column_bindings=[column_binding(name)]
    )
    assert report["inputs"][0]["column_status"] == status
    assert report["inputs"][0]["data_origin"] == "unassessed"
    assert report["dataset"]["extra_columns"] == "permitted"


def test_column_binding_without_dataset_does_not_measure_absence():
    report = build_concept_coverage(
        InventoryProvider(), column_bindings=[column_binding("missing")]
    )
    assert report["inputs"][0]["column_status"] == "unassessed"


def test_proxy_evidence_is_an_unverified_assertion_not_readiness():
    report = build_concept_coverage(InventoryProvider(), fact_bindings=[fact_binding()])
    assert report["inputs"][0]["semantic_status"] == "unverified_assertions"
    assert report["inputs"][0]["metadata"]["concept_id"] is None
    assert report["readiness"]["population_schema_ready"] is False


@pytest.mark.parametrize("relationship", ["exact", "proxy"])
def test_non_unresolved_bindings_require_pinned_evidence(relationship):
    binding = fact_binding(relationship)
    binding["evidence"] = []
    with pytest.raises(ValueError, match="evidence"):
        build_concept_coverage(InventoryProvider(), fact_bindings=[binding])


@pytest.mark.parametrize("relationship", ["exact", "proxy"])
def test_unknown_semantics_preserve_assertion_but_machine_classify_unresolved(
    relationship,
):
    report = build_concept_coverage(
        InventoryProvider(), fact_bindings=[fact_binding(relationship)]
    )
    binding = report["fact_bindings"][0]
    assert binding["asserted_relationship"] == relationship
    assert binding["effective_relationship"] == "unresolved"
    assert binding["classification_reason"] == "target_semantics_unavailable"


@pytest.mark.parametrize(
    "field", ["statistic", "entity", "universe", "unit", "geography", "period"]
)
def test_exact_statistical_scope_mismatch_fails_closed(field):
    binding = fact_binding("exact")
    binding["target"]["scope"][field] = "different"
    with pytest.raises(ValueError, match="scope"):
        build_concept_coverage(InventoryProvider(), fact_bindings=[binding])


def test_unknown_and_duplicate_input_bindings_rejected():
    binding = column_binding()
    binding["input"]["name"] = "computed_tax"
    with pytest.raises(ValueError, match="unknown input"):
        build_concept_coverage(InventoryProvider(), column_bindings=[binding])
    with pytest.raises(ValueError, match="duplicate"):
        build_concept_coverage(
            InventoryProvider(), column_bindings=[column_binding(), column_binding()]
        )


def test_manifest_is_closed_and_digest_is_verified():
    report = build_concept_coverage(InventoryProvider())
    for path in [(), ("inputs", 0, "metadata"), ("dataset",)]:
        changed = deepcopy(report)
        target = changed
        for key in path:
            target = target[key]
        target["invented"] = True
        with pytest.raises(ValueError):
            validate_concept_coverage(changed)
    changed = deepcopy(report)
    changed["inputs"][0]["name"] = "changed"
    with pytest.raises(ValueError, match="digest"):
        validate_concept_coverage(changed)


def test_provenance_and_content_digest_are_deterministic():
    first = build_concept_coverage(InventoryProvider())
    assert build_concept_coverage(InventoryProvider()) == first
    assert len(first["content_sha256"]) == 64
    assert (
        next(
            item["sha256"]
            for item in first["fingerprints"]
            if item["role"] == "entry_module"
        )
        == "a" * 64
    )


def test_readiness_cannot_be_promoted_by_payload_edit():
    report = build_concept_coverage(InventoryProvider())
    report["readiness"]["certified"] = True
    with pytest.raises(ValueError):
        validate_concept_coverage(report)


@pytest.mark.parametrize("binding_type", ["column", "fact"])
def test_bindings_cannot_be_replayed_against_different_artifact(binding_type):
    binding = column_binding() if binding_type == "column" else fact_binding()
    binding["artifact_fingerprints"][0]["sha256"] = "f" * 64
    argument = "column_bindings" if binding_type == "column" else "fact_bindings"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        build_concept_coverage(InventoryProvider(), **{argument: [binding]})


def test_noncanonical_alias_binding_is_rejected_not_merged():
    binding = fact_binding()
    binding["input"]["canonical_request_name"] = "zz:statutes/another#input.earnings"
    with pytest.raises(ValueError, match="non-canonical alias"):
        build_concept_coverage(InventoryProvider(), fact_bindings=[binding])


def test_conflicting_target_concepts_at_one_slot_fail_closed():
    first = fact_binding()
    first["target"]["concept_id"] = "concept:gross"
    second = deepcopy(first)
    second["source"]["fact_id"] = "chronicle:another/fact"
    second["target"]["concept_id"] = "concept:net"
    with pytest.raises(ValueError, match="conflicting target concept"):
        build_concept_coverage(InventoryProvider(), fact_bindings=[first, second])


def test_stock_without_reference_instant_is_rejected():
    binding = fact_binding()
    binding["source"]["scope"]["measurement_kind"] = "stock"
    with pytest.raises(ValueError, match="reference instant"):
        build_concept_coverage(InventoryProvider(), fact_bindings=[binding])


def test_stock_flow_and_assessment_income_year_cannot_be_asserted_exact():
    for field, value in (
        ("measurement_kind", "stock"),
        ("accounting_basis", "assessment_year"),
    ):
        binding = fact_binding("exact")
        binding["source"]["scope"][field] = value
        if field == "measurement_kind":
            binding["source"]["scope"]["reference_instant"] = "2025-12-31"
        with pytest.raises(ValueError, match="scope"):
            build_concept_coverage(InventoryProvider(), fact_bindings=[binding])


def test_asserted_exact_requires_identity_transformation():
    binding = fact_binding("exact")
    binding["transformation"]["kind"] = "declared_conversion"
    with pytest.raises(ValueError, match="identity transformation"):
        build_concept_coverage(InventoryProvider(), fact_bindings=[binding])


@pytest.mark.parametrize("field", ["entity_definition", "universe_definition"])
def test_assertions_need_pinned_population_definitions(field):
    binding = fact_binding()
    binding["source"]["scope"][field] = None
    with pytest.raises(ValueError, match=field):
        build_concept_coverage(InventoryProvider(), fact_bindings=[binding])


def test_unsupported_discovery_refuses_instead_of_reporting_zero():
    with pytest.raises(TypeError, match="does not support input discovery"):
        build_concept_coverage(object())


@pytest.mark.parametrize("count", [0, 2])
def test_partial_discovery_cannot_drop_or_invent_an_input(count):
    class PartialProvider(InventoryProvider):
        def input_inventory(self):
            original = super().input_inventory()
            discovery = deepcopy(original.entity_discovery)
            discovery[0]["root_input_count"] = count
            return replace(original, entity_discovery=discovery)

    with pytest.raises(ValueError, match="root input count"):
        build_concept_coverage(PartialProvider())


def test_zero_inputs_requires_successful_runtime_enumeration():
    class EmptyProvider(InventoryProvider):
        def input_inventory(self):
            original = super().input_inventory()
            discovery = deepcopy(original.entity_discovery)
            discovery[0]["root_input_count"] = 0
            return replace(original, inputs=(), entity_discovery=discovery)

    report = build_concept_coverage(EmptyProvider())
    assert report["root_input_count"] == 0
    assert report["inputs"] == []
    assert report["entity_discovery"][0]["root_input_count"] is None
    assert report["entity_discovery"][1]["status"] == "complete"


def test_completely_failed_discovery_cannot_look_like_zero_inputs():
    class FailedProvider(InventoryProvider):
        def input_inventory(self):
            original = super().input_inventory()
            discovery = tuple(
                {**item, "status": "no_derived_program", "root_input_count": None}
                for item in original.entity_discovery
            )
            return replace(original, inputs=(), entity_discovery=discovery)

    with pytest.raises(ValueError, match="successfully enumerated"):
        build_concept_coverage(FailedProvider())


def test_no_program_is_not_successfully_enumerated_zero():
    class FalseZeroProvider(InventoryProvider):
        def input_inventory(self):
            original = super().input_inventory()
            discovery = deepcopy(original.entity_discovery)
            discovery[1]["root_input_count"] = 0
            return replace(original, entity_discovery=discovery)

    with pytest.raises(ValueError, match="uncompiled entity"):
        build_concept_coverage(FalseZeroProvider())


def test_discovery_without_provenance_cannot_look_complete():
    class UnpinnedProvider(InventoryProvider):
        def input_inventory(self):
            return replace(super().input_inventory(), fingerprints=())

    with pytest.raises(
        ValueError, match="missing or conflicting artifact fingerprints"
    ):
        build_concept_coverage(UnpinnedProvider())


def _dataset_with_column(label):
    return Frame(
        {
            "person": pd.DataFrame(
                {"person_id": [1], "person_household_id": [1], label: [10.0]}
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([1.0]), WeightKind.DESIGN)},
    )


def test_integer_column_label_cannot_be_coerced_into_a_string_match():
    dataset = _dataset_with_column(123)
    assert 123 in dataset.table("person").columns
    assert "123" not in dataset.table("person").columns
    with pytest.raises(ValueError, match="non-string column"):
        build_concept_coverage(
            InventoryProvider(),
            dataset=dataset,
            column_bindings=[column_binding("123")],
        )


def test_actual_string_column_label_remains_a_valid_explicit_match():
    dataset = _dataset_with_column("123")
    report = build_concept_coverage(
        InventoryProvider(), dataset=dataset, column_bindings=[column_binding("123")]
    )
    assert report["inputs"][0]["column_status"] == "present"


@pytest.mark.parametrize("binding_kind", ["column", "fact"])
def test_real_axiom_binding_cannot_replay_between_native_root_entities(binding_kind):
    """The same address/file pins can name inputs on different native roots."""
    pytest.importorskip("axiom_rules_engine")
    pytest.importorskip("axiom_rules_engine_dense")
    root = (Path(__file__).parent / "fixtures/rulespec-zz").resolve()
    module = root / "zz/policies/tests/shared_input.yaml"
    schema = EntitySchema(group_entities=("household",))
    normal = AxiomEngine(module, schema, rulespec_roots=(root,))
    swapped = AxiomEngine(
        module,
        schema,
        rulespec_roots=(root,),
        entity_names={"person": "Household", "household": "Person"},
    )
    before = normal.input_inventory()
    after = swapped.input_inventory()
    original = next(item for item in before.inputs if item.entity == "person")
    remapped = next(item for item in after.inputs if item.entity == "person")
    assert original.engine_entity == "Person"
    assert remapped.engine_entity == "Household"
    assert original.canonical_request_name == remapped.canonical_request_name
    assert before.fingerprints == after.fingerprints
    binding = (
        column_binding("shared_input") if binding_kind == "column" else fact_binding()
    )
    binding["input"] = {
        "entity": original.entity,
        "engine_entity": original.engine_entity,
        "name": original.name,
        "canonical_request_name": original.canonical_request_name,
    }
    binding["artifact_fingerprints"] = [dict(item) for item in before.fingerprints]
    argument = "column_bindings" if binding_kind == "column" else "fact_bindings"
    dataset = _dataset_with_column("shared_input")
    first = build_concept_coverage(normal, dataset=dataset, **{argument: [binding]})
    person = next(item for item in first["inputs"] if item["entity"] == "person")
    assert person["engine_entity"] == "Person"
    assert first[argument][0]["input"]["engine_entity"] == "Person"
    if binding_kind == "column":
        assert person["column_status"] == "present"
    else:
        assert first[argument][0]["effective_relationship"] == "unresolved"
    with pytest.raises(ValueError, match="engine entity"):
        build_concept_coverage(swapped, dataset=dataset, **{argument: [binding]})
