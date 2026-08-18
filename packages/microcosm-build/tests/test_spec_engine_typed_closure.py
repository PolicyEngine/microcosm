"""Exact compiler-facing column, artifact, and scope closure gates."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any

import pytest

from microcosm.build.spec_engine import load_schema_registry
from microcosm.build.spec_engine.model import EntitySpec, thaw_json
from microcosm.build.spec_engine.typed_closure import (
    TypedClosureError,
    TypedClosureResult,
    compile_producer_outputs,
    resolve_typed_closure,
)
from tools.us_bundle_generation.contracts import build_take_up
from tools.us_bundle_generation.core import build_catalogs
from tools.us_bundle_generation.imputation import build_imputation

pytest.importorskip(
    "policyengine_us",
    reason="live-engine oracle: the wheels gate's venv installs no engine",
)

_ENTITY_IDS = (
    "person",
    "tax_unit",
    "spm_unit",
    "household",
    "family",
    "benunit",
    "marital_unit",
    "frame",
)
_ENTITIES = tuple(EntitySpec(entity_id) for entity_id in _ENTITY_IDS)


@pytest.fixture(scope="module")
def typed_resources() -> dict[str, Any]:
    return {
        "imputation": build_imputation(),
        "catalogs": build_catalogs(),
        "take_up": build_take_up(),
    }


@pytest.fixture(scope="module")
def typed_result(typed_resources: dict[str, Any]) -> TypedClosureResult:
    return resolve_typed_closure(typed_resources, entities=_ENTITIES)


def test_us_typed_closure_has_exact_inventory_counts(
    typed_resources: dict[str, Any],
    typed_result: TypedClosureResult,
) -> None:
    assert len(typed_result.columns) == 173
    assert len(typed_result.artifacts) == 84
    assert len(typed_result.scopes) == 7
    assert Counter(artifact.kind for artifact in typed_result.artifacts) == {
        "producer_node": 38,
        "virtual_output": 18,
        "virtual_resource_binding": 28,
    }

    graph = typed_resources["imputation"]["producer_graph"]
    assert sum(len(node["outputs"]) for node in graph["nodes"]) == 92
    compiled_outputs = compile_producer_outputs(typed_resources)
    assert sum(len(rows) for rows in compiled_outputs.values()) == 227
    assert sum(len(node["virtual_resources"]) for node in graph["nodes"]) == 75
    column_keys = {column.key for column in typed_result.columns}
    output_artifact_ids = {
        artifact.id
        for artifact in typed_result.artifacts
        if artifact.kind == "virtual_output"
    }
    for node in graph["nodes"]:
        for output in compiled_outputs[node["id"]]:
            key = f"{output['entity']}.{output['column']}"
            is_virtual = (
                output["column"].startswith("@")
                and output["column"] != "@resolved_weight"
            )
            assert (key in column_keys, key in output_artifact_ids) == (
                not is_virtual,
                is_virtual,
            )

    resource_artifacts = {
        artifact.id: artifact
        for artifact in typed_result.artifacts
        if artifact.kind == "virtual_resource_binding"
    }
    assert set(resource_artifacts) == {
        resource["id"]
        for node in graph["nodes"]
        for resource in node["virtual_resources"]
    }
    assert (
        sum(
            len(thaw_json(artifact.binding)["rows"])
            for artifact in resource_artifacts.values()
        )
        == 75
    )


def test_us_scope_specs_are_exact_finite_predicates(
    typed_result: TypedClosureResult,
) -> None:
    assert [(scope.id, scope.predicate_space) for scope in typed_result.scopes] == [
        ("acs_source", "producer_origin_clone_or_receipt"),
        ("asec_source", "producer_origin_clone_or_receipt"),
        ("puf_clone", "producer_origin_clone_or_receipt"),
        ("receipt", "producer_origin_clone_or_receipt"),
        ("whole_pool", "producer_origin_clone_or_receipt"),
        ("asec_rows", "take_up_support_channel"),
        ("puf_support_rows", "take_up_support_channel"),
    ]
    take_up_scopes = {
        scope.id: set(thaw_json(scope.predicate)["atoms"])
        for scope in typed_result.scopes
        if scope.predicate_space == "take_up_support_channel"
    }
    assert take_up_scopes["asec_rows"].isdisjoint(take_up_scopes["puf_support_rows"])
    assert set().union(*take_up_scopes.values()) == {
        "support_channel:asec",
        "support_channel:puf_tax_detail",
    }


@pytest.mark.parametrize(
    ("coverage_scope", "message"),
    [
        ("puf_clone", "duplicates its family-owned output"),
        ("whole_pool", "conflicts with its family-owned output"),
    ],
)
def test_family_owned_output_mirrors_refuse_before_typed_compilation(
    typed_resources: dict[str, Any],
    coverage_scope: str,
    message: str,
) -> None:
    mutated = copy.deepcopy(typed_resources)
    primary = next(
        family
        for family in mutated["imputation"]["families"]
        if family["stage"] == "primary_puf_qrf"
    )
    target = next(
        row
        for row in primary["targets"]
        if row["output_coverage_scope"] == "puf_clone"
    )
    node = next(
        row
        for row in mutated["imputation"]["producer_graph"]["nodes"]
        if row["id"] == primary["execution_contract"]
    )
    node["outputs"].append(
        {
            "entity": target["entity"],
            "column": target["name"],
            "coverage_scope": coverage_scope,
            "temporary": False,
            "validation_only": False,
        }
    )

    with pytest.raises(TypedClosureError, match=message):
        compile_producer_outputs(mutated)


def test_scope_registries_are_explicit_and_equal_the_compiler_defaults(
    typed_resources: dict[str, Any],
) -> None:
    registry = load_schema_registry()
    for domain, schema_id, registry_path in (
        (typed_resources["imputation"], "imputation.schema.json", "producer_graph"),
        (typed_resources["take_up"], "take_up.schema.json", None),
    ):
        registry.validate(domain, schema_id)
        omitted = copy.deepcopy(domain)
        container = omitted if registry_path is None else omitted[registry_path]
        explicit = container.pop("scope_registry")
        registry.validate(omitted, schema_id)
        defaulted = registry.validate_and_inject_defaults(omitted, schema_id)
        defaulted_container = (
            defaulted if registry_path is None else defaulted[registry_path]
        )
        assert defaulted_container["scope_registry"] == explicit


def test_missing_physical_column_contract_refuses(
    typed_resources: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(typed_resources)
    mutated["catalogs"]["columns"] = [
        row
        for row in mutated["catalogs"]["columns"]
        if row["key"] != "family.@resolved_weight"
    ]
    with pytest.raises(TypedClosureError, match="missing compiled physical output"):
        resolve_typed_closure(mutated, entities=_ENTITIES)


def test_expired_catalog_metadata_waiver_refuses(
    typed_resources: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(typed_resources)
    mutated["catalogs"]["metadata_waivers"][0]["expires_on"] = "2000-01-01"
    with pytest.raises(TypedClosureError, match="metadata waiver expired"):
        resolve_typed_closure(mutated, entities=_ENTITIES)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda resource: resource.update(
                {"id": "unknown.@post_clone_source_execution_config"}
            ),
            "unknown entity prefix",
        ),
        (
            lambda resource: resource["binding"].pop("resource_kind"),
            "binding/resource_kind",
        ),
    ],
)
def test_dangling_virtual_resource_binding_refuses(
    typed_resources: dict[str, Any],
    mutation,
    message: str,
) -> None:
    mutated = copy.deepcopy(typed_resources)
    resource = mutated["imputation"]["producer_graph"]["nodes"][0]["virtual_resources"][
        0
    ]
    mutation(resource)
    with pytest.raises(TypedClosureError, match=message):
        resolve_typed_closure(mutated, entities=_ENTITIES)


def test_dangling_producer_scope_reference_refuses(
    typed_resources: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(typed_resources)
    mutated["imputation"]["producer_graph"]["nodes"][0]["outputs"][0][
        "coverage_scope"
    ] = "missing_scope"
    with pytest.raises(TypedClosureError, match="dangling producer row scope"):
        resolve_typed_closure(mutated, entities=_ENTITIES)


def _replace_scope(value: object, old: str, new: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                key in {"coverage_scope", "required_scope", "row_scope"}
                and child == old
            ):
                value[key] = new
            else:
                _replace_scope(child, old, new)
    elif isinstance(value, list):
        for child in value:
            _replace_scope(child, old, new)


def test_orphan_producer_scope_refuses(typed_resources: dict[str, Any]) -> None:
    mutated = copy.deepcopy(typed_resources)
    graph = mutated["imputation"]["producer_graph"]
    _replace_scope(graph, "receipt", "whole_pool")
    with pytest.raises(TypedClosureError, match=r"orphan scopes \['receipt'\]"):
        resolve_typed_closure(mutated, entities=_ENTITIES)


def test_mixed_take_up_scope_overlap_refuses(
    typed_resources: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(typed_resources)
    mixed = next(
        program
        for program in mutated["take_up"]["programs"]
        if program["ownership"] == "mixed"
    )
    mixed["segments"][1]["row_scope"] = mixed["segments"][0]["row_scope"]
    with pytest.raises(TypedClosureError, match="scope predicates overlap"):
        resolve_typed_closure(mutated, entities=_ENTITIES)
