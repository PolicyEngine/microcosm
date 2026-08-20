from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from microcosm.build.spec_engine.resolver import (
    KernelRegistry,
    SpecResolutionError,
    resolve_cross_references,
)
from microcosm.build.spec_engine.typed_closure import compile_producer_outputs

Mutation = Callable[[dict[str, Any]], None]


def _target(
    name: str,
    entity: str = "person",
    *,
    output_coverage_scope: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "entity": entity,
        "requires_concepts": [],
    }
    if output_coverage_scope is not None:
        result["output_coverage_scope"] = output_coverage_scope
    return result


def _gap_target(name: str, order_index: int) -> dict[str, object]:
    return {
        **_target(name),
        "producer_binding": {
            "operator": "gap_operator",
            "order_index": order_index,
            "execution_scope": "whole_pool",
            "stage": "gap_fill_stacked_spine",
        },
    }


def _valid_resources() -> dict[str, Any]:
    primary_node = "primary_puf_qrf"
    late_node = "transfer:person/late"
    return {
        "bundle": {
            "country": "xx",
            "identity_generation": 1,
            "seed_protocol": "legacy-v1",
        },
        "sources": {
            "sources": [],
            "stages": [{"stage": "source_stage"}],
        },
        "vintages": {"records": []},
        "spine": {"channels": [], "support_roles": []},
        "catalogs": {"columns": []},
        "imputation": {
            "predictor_blocks": {"base": {"columns": ["base_column"]}},
            "models": {"model": {}},
            "concepts": {},
            "waiver_records": [],
            "transfer_execution": {
                "profiles": {"early_profile": {}, "late_profile": {}}
            },
            "gap_fill_schedule": {"directions": [{"name": "forward"}]},
            "chaining": {
                "split_after": [
                    {
                        "family": "early/forward/person/family",
                        "after_target": "early_2",
                    }
                ],
            },
            "families": [
                {
                    "id": "early/forward/person/family",
                    "stage": "gap_fill_stacked_spine",
                    "entities": ["person"],
                    "direction": "forward",
                    "donor": {},
                    "model": "model",
                    "predictors": ["base"],
                    "max_targets_per_fit": 2,
                    "execution_contract": "early_profile",
                    "targets": [
                        _gap_target("early_1", 0),
                        _gap_target("early_2", 1),
                        _gap_target("early_3", 2),
                    ],
                },
                {
                    "id": "primary/puf",
                    "stage": "primary_puf_qrf",
                    "entities": ["person"],
                    "donor": {},
                    "model": "model",
                    "predictors": ["base"],
                    "chaining": "base_plus_preceding_declared_targets",
                    "execution_contract": primary_node,
                    "targets": [
                        _target("primary_1", output_coverage_scope="puf_clone"),
                        _target("primary_2", output_coverage_scope="puf_clone"),
                    ],
                },
                {
                    "id": "late/person/late",
                    "stage": "late_producer_dag",
                    "entities": ["person"],
                    "runtime_name": late_node,
                    "donor": {},
                    "model": "model",
                    "predictors": ["base"],
                    "max_targets_per_fit": 2,
                    "execution_contract": "late_profile",
                    "targets": [
                        _target("late_1", output_coverage_scope="whole_pool")
                    ],
                },
            ],
            "producer_graph": {
                "external_stages": ["external_input"],
                "scope_coverage": {
                    "declared": {
                        "puf_clone": ["puf_clone"],
                        "whole_pool": ["whole_pool"],
                    }
                },
                "nodes": [
                    {
                        "id": primary_node,
                        "name": primary_node,
                        "kind": "primary_puf",
                        "inputs": [{"producing_stage": "external_input"}],
                        "virtual_resources": [
                            {
                                "binding": {
                                    "source_stage_ref": {
                                        "stage_id": "source_stage",
                                    }
                                }
                            }
                        ],
                    },
                    {
                        "id": late_node,
                        "name": late_node,
                        "kind": "late_transfer",
                        "inputs": [{"producing_stage": primary_node}],
                    },
                ],
                "resource_semantics": {},
            },
        },
    }


def _resolve(resources: dict[str, Any]) -> None:
    resolve_cross_references(
        resources,
        kernel_registry=KernelRegistry.from_ids([]),
    )


def _imputation(resources: dict[str, Any]) -> dict[str, Any]:
    return resources["imputation"]


def _graph(resources: dict[str, Any]) -> dict[str, Any]:
    return _imputation(resources)["producer_graph"]


def _families(resources: dict[str, Any]) -> list[dict[str, Any]]:
    return _imputation(resources)["families"]


def _rename_first_node(resources: dict[str, Any]) -> None:
    _graph(resources)["nodes"][0]["name"] = "renamed"


def _dangle_input_stage(resources: dict[str, Any]) -> None:
    _graph(resources)["nodes"][0]["inputs"][0]["producing_stage"] = "missing"


def _add_stale_dependencies(resources: dict[str, Any]) -> None:
    _graph(resources)["nodes"][1]["depends_on"] = ["primary_puf_qrf"]


def _add_stale_edges(resources: dict[str, Any]) -> None:
    _graph(resources)["edges"] = [["primary_puf_qrf", "transfer:person/late"]]


def _add_stale_order(resources: dict[str, Any]) -> None:
    _graph(resources)["order"] = ["primary_puf_qrf", "transfer:person/late"]


def _add_stale_waves(resources: dict[str, Any]) -> None:
    _graph(resources)["waves"] = [
        ["primary_puf_qrf"],
        ["transfer:person/late"],
    ]


def _add_stale_input_inventories(resources: dict[str, Any]) -> None:
    _graph(resources)["input_inventories"] = {}


def _add_stale_incomparable_policy(resources: dict[str, Any]) -> None:
    _graph(resources)["incomparable_node_policy"] = {}


def _add_stale_transfer_groups(resources: dict[str, Any]) -> None:
    _graph(resources)["transfer_groups"] = []


def _add_stale_write_scopes(resources: dict[str, Any]) -> None:
    _graph(resources)["nodes"][0]["write_scopes"] = []


def _make_cycle(resources: dict[str, Any]) -> None:
    graph = _graph(resources)
    late = graph["nodes"][1]["id"]
    graph["nodes"][0]["inputs"].append({"producing_stage": late})


def _add_stale_semantic_order(resources: dict[str, Any]) -> None:
    _graph(resources)["resource_semantics"]["producer_order"] = [
        node["id"] for node in _graph(resources)["nodes"]
    ]


def _mismatch_family_entities(resources: dict[str, Any]) -> None:
    _families(resources)[2]["entities"] = ["tax_unit"]


def _dangle_gap_contract(resources: dict[str, Any]) -> None:
    _families(resources)[0]["execution_contract"] = "missing"


def _bind_primary_to_profile(resources: dict[str, Any]) -> None:
    _families(resources)[1]["execution_contract"] = "early_profile"


def _dangle_direction(resources: dict[str, Any]) -> None:
    _families(resources)[0]["direction"] = "reverse"


def _dangle_late_runtime(resources: dict[str, Any]) -> None:
    _families(resources)[2]["runtime_name"] = "transfer:person/missing"


def _wrong_late_node_kind(resources: dict[str, Any]) -> None:
    _graph(resources)["nodes"][1]["kind"] = "post_clone_source"


def _remove_primary_output_scope(resources: dict[str, Any]) -> None:
    _families(resources)[1]["targets"][0].pop("output_coverage_scope")


def _add_gap_output_scope(resources: dict[str, Any]) -> None:
    _families(resources)[0]["targets"][0]["output_coverage_scope"] = "whole_pool"


def _dangle_family_output_scope(resources: dict[str, Any]) -> None:
    _families(resources)[2]["targets"][0]["output_coverage_scope"] = "missing"


def _duplicate_family_owned_output(resources: dict[str, Any]) -> None:
    _graph(resources)["nodes"][0].setdefault("outputs", []).append(
        {
            "entity": "person",
            "column": "primary_1",
            "coverage_scope": "puf_clone",
        }
    )


def _conflict_with_family_owned_output(resources: dict[str, Any]) -> None:
    _graph(resources)["nodes"][0].setdefault("outputs", []).append(
        {
            "entity": "person",
            "column": "primary_1",
            "coverage_scope": "whole_pool",
        }
    )


def _duplicate_structural_output(resources: dict[str, Any]) -> None:
    output = {
        "entity": "person",
        "column": "person_id",
        "coverage_scope": "whole_pool",
    }
    _graph(resources)["nodes"][0].setdefault("outputs", []).extend(
        [output, copy.deepcopy(output)]
    )


def _duplicate_late_family_link(resources: dict[str, Any]) -> None:
    duplicate = copy.deepcopy(_families(resources)[2])
    duplicate["id"] = "late/person/duplicate"
    _families(resources).append(duplicate)


def _add_orphan_primary_node(resources: dict[str, Any]) -> None:
    _graph(resources)["nodes"].append(
        {
            "id": "primary_puf_orphan",
            "name": "primary_puf_orphan",
            "kind": "primary_puf",
            "inputs": [{"producing_stage": "external_input"}],
        }
    )


def _dangle_split_family(resources: dict[str, Any]) -> None:
    _imputation(resources)["chaining"]["split_after"][0]["family"] = "missing"


def _dangle_split_target(resources: dict[str, Any]) -> None:
    _imputation(resources)["chaining"]["split_after"][0]["after_target"] = "missing"


def _split_off_boundary(resources: dict[str, Any]) -> None:
    _imputation(resources)["chaining"]["split_after"][0]["after_target"] = "early_1"


def _add_stale_primary_tuples(resources: dict[str, Any]) -> None:
    _imputation(resources)["chaining"]["primary_effective_predictor_tuples"] = []


def _dangle_source_stage_reference(resources: dict[str, Any]) -> None:
    _graph(resources)["nodes"][0]["virtual_resources"][0]["binding"][
        "source_stage_ref"
    ]["stage_id"] = "missing"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_rename_first_node, "node id must equal name"),
        (_dangle_input_stage, "dangling stage reference"),
        (_add_stale_dependencies, "stale equality-only dependency assertion"),
        (_add_stale_edges, "stale compiler-derived fields"),
        (_add_stale_order, "stale compiler-derived fields"),
        (_add_stale_waves, "stale compiler-derived fields"),
        (_add_stale_input_inventories, "stale compiler-derived fields"),
        (_add_stale_incomparable_policy, "stale compiler-derived fields"),
        (_add_stale_transfer_groups, "stale compiler-derived fields"),
        (_add_stale_write_scopes, "stale compiler-derived assertion"),
        (_make_cycle, "producer graph contains a cycle"),
        (_add_stale_semantic_order, "stale derived assertion"),
        (_mismatch_family_entities, "must exactly cover target entities"),
        (_dangle_gap_contract, "dangling transfer-execution profile"),
        (_bind_primary_to_profile, "dangling primary producer-node"),
        (_dangle_direction, "dangling gap-fill direction"),
        (_dangle_late_runtime, "dangling late producer-node"),
        (_wrong_late_node_kind, "is not a late_transfer producer"),
        (_remove_primary_output_scope, "output_coverage_scope: expected identifier"),
        (_add_gap_output_scope, "only primary and late family targets"),
        (_dangle_family_output_scope, "dangling graph scope reference"),
        (_duplicate_family_owned_output, "duplicates family-owned output"),
        (_conflict_with_family_owned_output, "conflicts with family-owned output"),
        (_duplicate_structural_output, "duplicate authored producer output"),
        (_duplicate_late_family_link, "duplicate late-family runtime name"),
        (_add_orphan_primary_node, "primary-PUF nodes without families"),
        (_dangle_split_family, "dangling family reference"),
        (_dangle_split_target, "is not in family"),
        (_split_off_boundary, "is not a declared family-end"),
        (_add_stale_primary_tuples, "stale derived assertion"),
        (_dangle_source_stage_reference, "dangling source stage reference"),
    ],
)
def test_imputation_semantic_mutations_refuse_deterministically(
    mutation: Mutation,
    message: str,
) -> None:
    resources = _valid_resources()
    mutation(resources)

    with pytest.raises(SpecResolutionError, match=message):
        _resolve(resources)


def test_imputation_semantic_fixture_resolves() -> None:
    _resolve(_valid_resources())


def test_family_owned_outputs_compile_in_deterministic_materialized_order() -> None:
    compiled = compile_producer_outputs(_valid_resources())

    assert compiled["primary_puf_qrf"] == (
        {
            "entity": "person",
            "column": "primary_1",
            "coverage_scope": "puf_clone",
            "temporary": False,
            "validation_only": False,
        },
        {
            "entity": "person",
            "column": "primary_2",
            "coverage_scope": "puf_clone",
            "temporary": False,
            "validation_only": False,
        },
    )
    assert compiled["transfer:person/late"] == (
        {
            "entity": "person",
            "column": "late_1",
            "coverage_scope": "whole_pool",
            "temporary": False,
            "validation_only": False,
        },
    )


def test_empty_imputation_domain_resolves_for_minimal_country_bundle() -> None:
    _resolve(
        {
            "bundle": {
                "country": "xx",
                "identity_generation": 1,
                "seed_protocol": "legacy-v1",
            },
            "imputation": {},
        }
    )


def test_resolution_error_text_is_stable_across_repeated_runs() -> None:
    resources = _valid_resources()
    _add_stale_edges(resources)

    with pytest.raises(SpecResolutionError) as first:
        _resolve(copy.deepcopy(resources))
    with pytest.raises(SpecResolutionError) as second:
        _resolve(copy.deepcopy(resources))

    assert str(first.value) == str(second.value)
    assert str(first.value) == (
        "imputation/producer_graph: stale compiler-derived fields ['edges']"
    )


def _resources_with_reviewed_vintages() -> dict[str, Any]:
    resources = _valid_resources()
    resources["vintages"] = {
        "records": [
            {
                "id": "source_period",
                "kind": "tax_period_ref",
                "authority_ref": {
                    "kind": "source_record",
                    "source": "source:source",
                    "authority": "source_period",
                },
                "compatible_with": ["vintage:target_period"],
            },
            {
                "id": "target_period",
                "kind": "target_period_ref",
                "authority_ref": {
                    "kind": "dataset_run",
                    "pointer": "/dataset_run/target_period",
                },
                "compatible_with": ["vintage:source_period"],
            },
        ]
    }
    resources["sources"]["sources"] = [
        {
            "id": "source",
            "sha256": "0" * 64,
            "vintages": ["vintage:source_period", "vintage:target_period"],
            "vintage_authorities": [
                {"id": "source_period", "kind": "tax_period", "value": 2015}
            ],
        }
    ]
    resources["bundle"]["dataset_run"] = {"target_period": 2024}
    return resources


def test_reviewed_vintage_compatibility_resolves() -> None:
    _resolve(_resources_with_reviewed_vintages())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda resources: resources["vintages"]["records"][1].update(
                {"compatible_with": ["vintage:source_period", "vintage:missing"]}
            ),
            "dangling vintage reference",
        ),
        (
            lambda resources: resources["vintages"]["records"][1].update(
                {"compatible_with": ["vintage:missing"]}
            ),
            "not reciprocal",
        ),
        (
            lambda resources: resources["vintages"]["records"][0].update(
                {"kind": "geography_vintage_ref"}
            ),
            "incompatible kind pair",
        ),
    ],
)
def test_vintage_compatibility_mutations_refuse(
    mutation: Mutation,
    message: str,
) -> None:
    resources = _resources_with_reviewed_vintages()
    mutation(resources)
    with pytest.raises(SpecResolutionError, match=message):
        _resolve(resources)


def test_composite_source_requires_pairwise_vintage_compatibility() -> None:
    resources = _resources_with_reviewed_vintages()
    resources["vintages"]["records"].extend(
        [
            {
                "id": "survey_period",
                "kind": "survey_period_ref",
                "authority_ref": {
                    "kind": "source_record",
                    "source": "source:source",
                    "authority": "survey_period",
                },
                "compatible_with": ["vintage:target_period"],
            }
        ]
    )
    resources["vintages"]["records"][1]["compatible_with"].append(
        "vintage:survey_period"
    )
    resources["sources"]["sources"][0]["vintages"] = [
        "vintage:source_period",
        "vintage:survey_period",
    ]
    resources["sources"]["sources"][0]["vintage_authorities"].append(
        {"id": "survey_period", "kind": "survey_period", "value": 2023}
    )

    with pytest.raises(SpecResolutionError, match="lack a reviewed compatibility"):
        _resolve(resources)


def _resources_with_catalog() -> dict[str, Any]:
    resources = _valid_resources()
    keys = sorted(
        {
            f"{target['entity']}.{target['name']}"
            for family in _families(resources)
            for target in family["targets"]
        }
    )
    resources["catalogs"] = {
        "columns": [
            {
                "key": key,
                "contract": {
                    "entity": key.split(".", 1)[0],
                    "dtype": "float64",
                    "period": "year",
                    "nullable": False,
                },
            }
            for key in keys
        ]
    }
    return resources


def test_catalog_contracts_resolve_against_typed_outputs() -> None:
    _resolve(_resources_with_catalog())


def test_duplicate_catalog_key_refuses() -> None:
    resources = _resources_with_catalog()
    resources["catalogs"]["columns"].append(
        copy.deepcopy(resources["catalogs"]["columns"][0])
    )
    with pytest.raises(SpecResolutionError, match="duplicate key"):
        _resolve(resources)


def test_catalog_key_entity_prefix_mismatch_refuses() -> None:
    resources = _resources_with_catalog()
    resources["catalogs"]["columns"][0]["contract"]["entity"] = "tax_unit"
    with pytest.raises(SpecResolutionError, match="entity prefix"):
        _resolve(resources)


def test_catalog_key_absent_from_compiled_outputs_refuses() -> None:
    resources = _resources_with_catalog()
    resources["catalogs"]["columns"].append(
        {
            "key": "person.not_a_typed_output",
            "contract": {
                "entity": "person",
                "dtype": "float64",
                "period": "year",
                "nullable": False,
            },
        }
    )
    with pytest.raises(SpecResolutionError, match="absent from compiled typed outputs"):
        _resolve(resources)


def test_missing_imputation_target_catalog_contract_refuses() -> None:
    resources = _resources_with_catalog()
    resources["catalogs"]["columns"].pop()
    with pytest.raises(SpecResolutionError, match="missing imputation target"):
        _resolve(resources)
