"""Fixture-scale contract tests for the generic spec-engine executor."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.spec_engine.brokers as broker_module
from microcosm.build.spec_engine.brokers import (
    AmbientAccessError,
    BrokerAccessError,
    BrokerOwner,
    BrokerSession,
    DeclaredSource,
)
from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.compiler_ir import (
    CompiledNode,
    ResolvedParam,
    SeedSiteIR,
    TransitiveNodeSlice,
    current_compiler_ir_abi,
    row_classifier_contract,
)
from microcosm.build.spec_engine.executor import (
    CapabilityError,
    Effect,
    ExecutionContext,
    ExecutorError,
    ImmutableFrameProjection,
    KernelPatch,
    NodeOrderingError,
    PatchScopeError,
    RegisteredKernel,
    RegisteredRowClassifier,
    RowClassification,
    RunProvenanceIdentity,
    StructuralDelta,
    StructuralDiffError,
    ValidatedPatch,
    WeightState,
    apply_patch,
    build_run_provenance_identity,
    diff_projections,
    execute_node,
    node_reuse_identity,
    order_nodes,
)
from microcosm.build.spec_engine.loader import load_bundle
from microcosm.build.spec_engine.model import FrozenMap, freeze_json, thaw_json
from microcosm.build.spec_engine.scope_algebra import ClosedScopeRegistry

_IMPLEMENTATION_SHA256 = "a" * 64
_SEED_PROTOCOL_SHA256 = "c" * 64
_NODE_SLICE_DOMAIN = "microcosm.spec-engine.node-slice.v1"
_NODE_KEY_DOMAIN = "microcosm.spec-engine.static-node-key.v1"


def _executor_global_helper() -> float:
    """Patch target for the transitive prebound-ambient regression test."""

    return 0.0


class _StrataWithMetadata(pd.Series):
    _metadata = ["executor_secret"]

    @property
    def _constructor(self):
        return _StrataWithMetadata


_PRESERVE_MUTATIONS: Mapping[str, tuple[str, str, str]] = {
    "entity_keys": ("preserve", "entity_keys_valid", "entity_keys_unchanged"),
    "cardinality": (
        "preserve",
        "entity_cardinality_valid",
        "entity_cardinality_unchanged",
    ),
    "links": ("preserve", "links_valid", "links_unchanged"),
    "memberships": (
        "preserve",
        "memberships_valid",
        "memberships_unchanged",
    ),
    "order": ("preserve", "entity_order_valid", "entity_order_unchanged"),
    "weights": ("preserve", "weights_valid", "weights_unchanged"),
    "mass_history": (
        "preserve",
        "mass_history_valid",
        "mass_history_unchanged",
    ),
}
_DELTA_MUTATIONS: Mapping[StructuralDelta, Mapping[str, tuple[str, str, str]]] = {
    StructuralDelta.FILTER: {
        "entity_keys": (
            "filter_entity_keys",
            "entity_keys_valid",
            "remaining_entity_keys_unique",
        ),
        "cardinality": (
            "filter_entity_rows",
            "entity_cardinality_valid",
            "entity_cardinality_filtered",
        ),
        "links": (
            "filter_link_rows",
            "links_valid",
            "links_reference_surviving_keys",
        ),
        "memberships": (
            "filter_membership_rows",
            "memberships_valid",
            "memberships_reference_surviving_keys",
        ),
        "order": (
            "filter_rows_preserving_order",
            "entity_order_valid",
            "surviving_entity_order_preserved",
        ),
        "weights": (
            "filter_row_weights",
            "weights_valid",
            "weights_aligned_to_surviving_keys",
        ),
    },
    StructuralDelta.EXPAND: {
        "entity_keys": (
            "append_remapped_clone_keys",
            "native_entity_keys_unique",
            "all_entity_keys_unique",
        ),
        "cardinality": (
            "expand_complete_household_graphs",
            "native_clone_index_zero",
            "clone_roles_materialized",
        ),
        "links": (
            "append_relinked_clone_links",
            "links_valid",
            "clone_links_reference_remapped_keys",
        ),
        "memberships": (
            "append_relinked_clone_memberships",
            "native_memberships_valid",
            "clone_memberships_reference_remapped_keys",
        ),
        "order": (
            "append_clone_blocks_preserving_native_order",
            "native_entity_order_valid",
            "clone_blocks_follow_native_rows",
        ),
        "weights": (
            "split_mass_across_clone_descendants",
            "native_household_mass_finite",
            "household_mass_conserved",
        ),
    },
    StructuralDelta.RELINK: {
        "links": ("relink_references", "links_valid", "links_valid"),
        "memberships": (
            "relink_memberships",
            "memberships_valid",
            "memberships_valid",
        ),
    },
    StructuralDelta.REORDER: {
        "order": ("reorder_rows", "entity_order_valid", "entity_order_permuted"),
        "weights": (
            "realign_row_weights",
            "weights_valid",
            "weights_preserve_key_mapping",
        ),
    },
    StructuralDelta.REWEIGHT: {
        "weights": ("replace_weights", "weights_valid", "weights_valid"),
        "mass_history": (
            "append_mass_history",
            "mass_history_valid",
            "mass_history_extended",
        ),
    },
}


def _fixture_scope_registry() -> ClosedScopeRegistry:
    return ClosedScopeRegistry(
        "fixture_rows",
        (
            "origin:a/clone:0",
            "origin:b/clone:0",
            "origin:b/clone:1",
        ),
        {
            "a_rows": ("origin:a/clone:0",),
            "b_rows": ("origin:b/clone:0", "origin:b/clone:1"),
            "whole": (
                "origin:a/clone:0",
                "origin:b/clone:0",
                "origin:b/clone:1",
            ),
        },
    )


@pytest.fixture
def projection() -> ImmutableFrameProjection:
    return ImmutableFrameProjection(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "group_id": [10, 10, 20],
                    "value": [10.0, 20.0, 30.0],
                }
            ),
            "group": pd.DataFrame(
                {
                    "group_key": [10, 20],
                    "group_value": [100.0, 200.0],
                }
            ),
        },
        entity_keys={"person": "person_id", "group": "group_key"},
        membership_columns={"person": ("group_id",)},
        membership_targets={"person": {"group_id": "group"}},
        links={
            "membership": pd.DataFrame(
                {"person_id": [1, 2, 3], "group_id": [10, 10, 20]}
            )
        },
        link_targets={"membership": {"person_id": "person", "group_id": "group"}},
        weights={"person": WeightState([1.0, 2.0, 3.0], "design")},
        strata=pd.Series(["x", "y", "z"], name="stratum"),
        strata_entity="person",
        mass_history=(),
        metadata={"sealed": {"version": 1}},
        virtual_receipts={},
        row_atoms={
            "person": {
                1: {"origin:a/clone:0"},
                2: {"origin:a/clone:0"},
                3: {"origin:b/clone:0"},
            },
            "group": {
                10: {"origin:a/clone:0"},
                20: {"origin:b/clone:0"},
            },
        },
    )


def _frozen_mapping(value: object) -> FrozenMap:
    frozen = freeze_json(value)
    assert isinstance(frozen, FrozenMap)
    return frozen


def _mutation_row(triple: tuple[str, str, str]) -> dict[str, str]:
    return dict(
        zip(("operation", "precondition", "postcondition"), triple, strict=True)
    )


def _mutations(
    delta: StructuralDelta,
    changed: tuple[str, ...],
    overrides: Mapping[str, tuple[str, str, str]] | None = None,
) -> dict[str, object]:
    selected = _DELTA_MUTATIONS.get(delta, {})
    result = {
        axis: _mutation_row(
            selected[axis] if axis in changed else _PRESERVE_MUTATIONS[axis]
        )
        for axis in _PRESERVE_MUTATIONS
    }
    for axis, triple in (overrides or {}).items():
        result[axis] = _mutation_row(triple)
    return result


def _scope(
    column: str,
    row_scope: str = "whole",
    *,
    entity: str = "person",
    mode: str = "column_cells",
) -> dict[str, object]:
    return {
        "entity": entity,
        "column": column,
        "row_scope": row_scope,
        "mode": mode,
        "cell_segments": [
            {
                "predicate": "coverage_scope",
                "coverage_scope": row_scope,
                "write_policy": "declared_output_write",
            }
        ],
    }


def _input(
    entity: str,
    column: str,
    *,
    required_scope: str = "whole",
    alternatives: list[list[dict[str, object]]] | None = None,
    tolerated_absence_receipts: list[str] | None = None,
) -> dict[str, object]:
    return {
        "entity": entity,
        "column": column,
        "producing_stage": "fixture_input",
        "required_scope": required_scope,
        "alternatives": [] if alternatives is None else alternatives,
        "tolerated_absence_receipts": (
            [] if tolerated_absence_receipts is None else tolerated_absence_receipts
        ),
    }


def _resolved_param(path: str, value: object) -> ResolvedParam:
    return ResolvedParam(
        path=path,
        value_sha256=sha256_json(value),
        value=freeze_json(value),
    )


def _node(
    delta: StructuralDelta | str,
    scopes: list[dict[str, object]],
    *,
    node_id: str = "fixture_node",
    execution_rank: int = 0,
    depends_on: tuple[str, ...] = (),
    changed_mutations: tuple[str, ...] = (),
    mutation_overrides: Mapping[str, tuple[str, str, str]] | None = None,
    determinism: str = "deterministic",
    numeric_reproducibility: str = "bitwise",
    effects: list[str] | None = None,
    retry_safety: str = "idempotent",
    seed_streams: tuple[str, ...] = (),
    omit_capability: str | None = None,
    scope_registry: ClosedScopeRegistry | None = None,
    inputs: tuple[dict[str, object], ...] | None = None,
    outputs: tuple[dict[str, object], ...] = (),
    column_contracts: tuple[dict[str, object], ...] = (),
    transitive_nodes: tuple[TransitiveNodeSlice, ...] = (),
) -> CompiledNode:
    structural_delta = StructuralDelta(delta)
    capabilities: dict[str, object] = {
        "determinism": determinism,
        "numeric_reproducibility": numeric_reproducibility,
        "effects": ["none"] if effects is None else effects,
        "structural_delta": structural_delta.value,
        "retry_safety": retry_safety,
    }
    if omit_capability is not None:
        capabilities.pop(omit_capability)
    mutations = _mutations(structural_delta, changed_mutations, mutation_overrides)
    node_inputs = (
        (
            {
                "entity": "person",
                "column": "value",
                "producing_stage": "fixture_input",
                "required_scope": "whole",
                "alternatives": [],
                "tolerated_absence_receipts": [],
            },
            {
                "entity": "group",
                "column": "group_value",
                "producing_stage": "fixture_input",
                "required_scope": "whole",
                "alternatives": [],
                "tolerated_absence_receipts": [],
            },
        )
        if inputs is None
        else inputs
    )
    node_outputs = outputs
    node_column_contracts = column_contracts
    if structural_delta is StructuralDelta.JOIN and not node_outputs:
        node_outputs = (
            {
                "entity": "person",
                "column": "joined",
                "coverage_scope": "whole",
                "temporary": False,
                "validation_only": False,
            },
        )
    if structural_delta is StructuralDelta.JOIN and not node_column_contracts:
        node_column_contracts = (
            {
                "key": "person.joined",
                "entity": "person",
                "dtype": "int64",
                "unit": "count",
                "period": "year",
                "vintage": "vintage:fixture",
                "nullable": False,
                "domain": "fixture",
                "public_stability": "internal",
                "unit_waiver": None,
            },
        )
    authored = {
        "id": node_id,
        "name": node_id,
        "kind": "fixture",
        "kernel": "kernel:fixture",
        "capabilities": capabilities,
        "mutations": mutations,
        "depends_on": list(depends_on),
        "inputs": list(node_inputs),
        "outputs": list(node_outputs),
        "virtual_resources": [],
    }
    registry = scope_registry or _fixture_scope_registry()
    compiler_ir_abi_object = current_compiler_ir_abi()
    classifier_ref, classifier_sha256 = row_classifier_contract(
        compiler_ir_abi_object,
        _frozen_mapping(registry.to_wire()),
    )
    sites = tuple(
        SeedSiteIR(
            id=f"fixture_site_{index}",
            stream=stream,
            contract=_frozen_mapping(
                {
                    "value_source": "run_request.fixture_seed",
                    "default": 17,
                    "rng_family": "numpy.random.Generator(PCG64)",
                    "rng_version": "fixture-v1",
                    "kernel": "legacy_v1_direct_draws",
                    "seed_material": ["fixture_seed"],
                    "consumption_order": ["row_order"],
                    "reset_boundary": "fresh_generator_per_call",
                    "draw_condition": "always",
                    "derivation": "direct_integer_seed",
                }
            ),
            owners=(("producer_node", node_id),),
        )
        for index, stream in enumerate(seed_streams)
    )
    contract_params = (
        (
            _resolved_param(
                f"/compiled/columns@producer={node_id}",
                list(node_column_contracts),
            ),
        )
        if node_column_contracts
        else ()
    )
    params = (
        _resolved_param(f"/imputation/producer_graph/nodes/{node_id}", authored),
        _resolved_param(
            f"/compiled/producer_graph/nodes/{node_id}/depends_on",
            list(depends_on),
        ),
        _resolved_param(
            f"/compiled/producer_graph/nodes/{node_id}/inputs",
            list(node_inputs),
        ),
        _resolved_param(
            f"/compiled/producer_graph/nodes/{node_id}/outputs",
            list(node_outputs),
        ),
        _resolved_param(
            f"/compiled/producer_graph/nodes/{node_id}/kernel",
            {
                "ref": "kernel:fixture",
                "implementation_sha256": _IMPLEMENTATION_SHA256,
            },
        ),
        _resolved_param(
            f"/compiled/producer_graph/nodes/{node_id}/write_scopes", scopes
        ),
        _resolved_param(
            "/imputation/producer_graph/scope_registry", registry.to_wire()
        ),
        *contract_params,
        _resolved_param(
            f"/compiled/producer_graph/nodes/{node_id}/execution_rank",
            execution_rank,
        ),
        _resolved_param(
            f"/compiled/producer_graph/nodes/{node_id}/row_classifier",
            {
                "ref": classifier_ref,
                "implementation_sha256": classifier_sha256,
            },
        ),
        _resolved_param(
            f"/compiled/effective_seed_grant@producer={node_id}",
            {
                "configuration_references": [],
                "grant_sources": [],
                "sites": [site.to_wire() for site in sites],
            },
        ),
    )
    node_slice_sha256 = sha256_json(
        {
            "domain": _NODE_SLICE_DOMAIN,
            "resolved_params": [param.to_wire() for param in params],
            "transitive_nodes": [row.to_wire() for row in transitive_nodes],
        }
    )
    compiler_ir_abi = compiler_ir_abi_object.to_wire()
    node_key = sha256_json(
        {
            "domain": _NODE_KEY_DOMAIN,
            "compiler_ir_abi": compiler_ir_abi,
            "node_slice_sha256": node_slice_sha256,
            "kernel": {
                "ref": "kernel:fixture",
                "implementation_sha256": _IMPLEMENTATION_SHA256,
            },
            "seed_protocol_sha256": _SEED_PROTOCOL_SHA256,
        }
    )
    return CompiledNode(
        id=node_id,
        execution_rank=execution_rank,
        node_key=node_key,
        node_slice_sha256=node_slice_sha256,
        kernel_ref="kernel:fixture",
        kernel_implementation_sha256=_IMPLEMENTATION_SHA256,
        depends_on=depends_on,
        inputs=tuple(_frozen_mapping(row) for row in node_inputs),
        outputs=tuple(_frozen_mapping(row) for row in node_outputs),
        capabilities=_frozen_mapping(capabilities),
        mutations=_frozen_mapping(mutations),
        write_scopes=tuple(_frozen_mapping(scope) for scope in scopes),
        scope_registry=_frozen_mapping(registry.to_wire()),
        row_classifier_ref=classifier_ref,
        row_classifier_implementation_sha256=classifier_sha256,
        compiler_ir_abi=_frozen_mapping(compiler_ir_abi),
        seed_protocol_sha256=_SEED_PROTOCOL_SHA256,
        seed_sites=sites,
        seed_streams=seed_streams,
        resolved_params=params,
        transitive_nodes=transitive_nodes,
    )


def _transitive_slice(node: CompiledNode) -> TransitiveNodeSlice:
    return TransitiveNodeSlice(
        id=node.id,
        local_slice_sha256=sha256_json(
            [param.to_wire() for param in node.resolved_params]
        ),
    )


def _run(
    node: CompiledNode,
    projection: ImmutableFrameProjection,
    kernel,
    *,
    context: ExecutionContext | None = None,
    row_classifiers: Mapping[str, RegisteredRowClassifier] | None = None,
):
    selected_context = context or ExecutionContext()
    if selected_context.run_provenance_identity is None:
        selected_context = replace(
            selected_context,
            run_provenance_identity=_fixture_run_provenance(),
        )
    return execute_node(
        node,
        projection,
        kernels={"kernel:fixture": RegisteredKernel(kernel, _IMPLEMENTATION_SHA256)},
        row_classifiers=row_classifiers,
        context=selected_context,
    )


def _fixture_run_provenance() -> RunProvenanceIdentity:
    return build_run_provenance_identity(
        identity_generation=0,
        source_grammar_receipt=None,
        spec_binding=None,
        authority_versions={"fixture": 1},
        code_inventory_digest="a" * 64,
        artifact_protocol_inventory={"fixture": "v1"},
        run_request={"rung": "fixture"},
        execution_receipt={"backend": "fixture"},
    )


def _broker_session(node: CompiledNode, **kwargs) -> BrokerSession:
    return BrokerSession.for_compiled_node(
        node,
        run_provenance_identity=_fixture_run_provenance().to_wire(),
        **kwargs,
    )


def _behavior_session(node: CompiledNode, *, fixture_seed: int = 17) -> BrokerSession:
    return _broker_session(node, run_inputs={"fixture_seed": fixture_seed})


def _row_classifier() -> RegisteredRowClassifier:
    def classify(
        entity: str,
        _table: pd.DataFrame,
        _key: str,
        added_ids: frozenset[object],
        _registry: ClosedScopeRegistry,
    ) -> dict[object, RowClassification]:
        source_rows = {
            ("person", 4): np.int64(3),
            ("group", 30): np.int64(20),
        }
        return {
            row_id: RowClassification(
                atoms=frozenset({"origin:b/clone:1"}),
                source_row_id=source_rows[(entity, row_id)],
            )
            for row_id in added_ids
        }

    registry = _fixture_scope_registry()
    _, classifier_sha256 = row_classifier_contract(
        current_compiler_ir_abi(),
        _frozen_mapping(registry.to_wire()),
    )
    return RegisteredRowClassifier(classify, classifier_sha256, "fixture_rows")


def _structural_context() -> Mapping[str, RegisteredRowClassifier]:
    return {"classifier:fixture_rows": _row_classifier()}


def _none_patch(
    projection: ImmutableFrameProjection,
    _context: ExecutionContext,
) -> KernelPatch:
    person = projection.table("person")
    person.loc[person.person_id.eq(1), "value"] = 11.0
    return KernelPatch(StructuralDelta.NONE, tables={"person": person})


def _filter_patch(
    projection: ImmutableFrameProjection,
    _context: ExecutionContext,
) -> KernelPatch:
    person_table = projection.table("person")
    group_table = projection.table("group")
    link_table = projection.link("membership")
    person = person_table.loc[person_table["person_id"].ne(3)].copy()
    group = group_table.loc[group_table["group_key"].ne(20)].copy()
    link = link_table.loc[link_table["person_id"].ne(3)].copy()
    return KernelPatch(
        StructuralDelta.FILTER,
        tables={"person": person, "group": group},
        links={"membership": link},
        weights={"person": WeightState([1.0, 2.0], "design")},
        replace_strata=True,
        strata=projection.strata.iloc[:2],  # type: ignore[union-attr]
    )


def _expand_patch(
    projection: ImmutableFrameProjection,
    _context: ExecutionContext,
) -> KernelPatch:
    person = pd.concat(
        [
            projection.table("person"),
            pd.DataFrame({"person_id": [4], "group_id": [30], "value": [30.0]}),
        ],
        ignore_index=True,
    )
    group = pd.concat(
        [
            projection.table("group"),
            pd.DataFrame({"group_key": [30], "group_value": [200.0]}),
        ],
        ignore_index=True,
    )
    strata = pd.concat(
        [projection.strata, pd.Series(["w"], name="stratum")],  # type: ignore[list-item]
        ignore_index=True,
    )
    link = pd.concat(
        [
            projection.link("membership"),
            pd.DataFrame({"person_id": [4], "group_id": [30]}),
        ],
        ignore_index=True,
    )
    return KernelPatch(
        StructuralDelta.EXPAND,
        tables={"person": person, "group": group},
        links={"membership": link},
        weights={"person": WeightState([1.0, 2.0, 1.5, 1.5], "design")},
        replace_strata=True,
        strata=strata,
    )


def _join_patch(
    projection: ImmutableFrameProjection,
    _context: ExecutionContext,
) -> KernelPatch:
    person = projection.table("person")
    person["joined"] = [100, 200, 300]
    return KernelPatch(StructuralDelta.JOIN, tables={"person": person})


def _relink_patch(
    projection: ImmutableFrameProjection,
    _context: ExecutionContext,
) -> KernelPatch:
    person = projection.table("person")
    person.loc[person.person_id.eq(1), "group_id"] = 20
    link = projection.link("membership")
    link.loc[link.person_id.eq(1), "group_id"] = 20
    return KernelPatch(
        StructuralDelta.RELINK,
        tables={"person": person},
        links={"membership": link},
    )


def _reorder_patch(
    projection: ImmutableFrameProjection,
    _context: ExecutionContext,
) -> KernelPatch:
    person = (
        projection.table("person")
        .set_index("person_id", drop=False)
        .loc[[3, 1, 2]]
        .reset_index(drop=True)
    )
    return KernelPatch(
        StructuralDelta.REORDER,
        tables={"person": person},
        weights={"person": WeightState([3.0, 1.0, 2.0], "design")},
        replace_strata=True,
        strata=pd.Series(["z", "x", "y"], name="stratum"),
    )


def _reweight_patch(
    _projection: ImmutableFrameProjection,
    _context: ExecutionContext,
) -> KernelPatch:
    return KernelPatch(
        StructuralDelta.REWEIGHT,
        weights={"person": WeightState([2.0, 2.0, 3.0], "importance")},
        mass_history=({"reason": "fixture", "factor": 7 / 6},),
    )


def _filter_or_expand_scopes() -> list[dict[str, object]]:
    return [
        _scope("person_id", "b_rows"),
        _scope("group_id", "b_rows"),
        _scope("value", "b_rows"),
        _scope("group_key", "b_rows", entity="group"),
        _scope("group_value", "b_rows", entity="group"),
        _scope("@resolved_weight", "b_rows", mode="resolved_weight"),
    ]


@pytest.mark.parametrize(
    ("delta", "kernel", "scopes", "changed_mutations", "row_classifiers"),
    [
        (StructuralDelta.NONE, _none_patch, [_scope("value", "a_rows")], (), None),
        (
            StructuralDelta.FILTER,
            _filter_patch,
            _filter_or_expand_scopes(),
            ("entity_keys", "cardinality", "links", "memberships", "order", "weights"),
            None,
        ),
        (
            StructuralDelta.EXPAND,
            _expand_patch,
            _filter_or_expand_scopes(),
            ("entity_keys", "cardinality", "links", "memberships", "order", "weights"),
            _structural_context(),
        ),
        (StructuralDelta.JOIN, _join_patch, [_scope("joined")], (), None),
        (
            StructuralDelta.RELINK,
            _relink_patch,
            [_scope("group_id", "a_rows", mode="structural_column")],
            ("links", "memberships"),
            None,
        ),
        (
            StructuralDelta.REORDER,
            _reorder_patch,
            [_scope("value"), _scope("@resolved_weight", mode="resolved_weight")],
            ("order", "weights"),
            None,
        ),
        (
            StructuralDelta.REWEIGHT,
            _reweight_patch,
            [_scope("@resolved_weight", mode="resolved_weight")],
            ("weights", "mass_history"),
            None,
        ),
    ],
)
def test_one_kernel_per_structural_delta_is_applied_transactionally(
    delta: StructuralDelta,
    kernel,
    scopes: list[dict[str, object]],
    changed_mutations: tuple[str, ...],
    row_classifiers: Mapping[str, RegisteredRowClassifier] | None,
    projection: ImmutableFrameProjection,
) -> None:
    original = projection.detached_copy()
    validated = _run(
        _node(delta, scopes, changed_mutations=changed_mutations),
        projection,
        kernel,
        row_classifiers=row_classifiers,
    )
    result = apply_patch(projection, validated)
    assert not validated.diff.empty
    assert diff_projections(projection, original).empty
    assert not diff_projections(projection, result).empty


@pytest.mark.parametrize(
    (
        "declared",
        "returned",
        "kernel",
        "scopes",
        "changed_mutations",
        "row_classifiers",
    ),
    [
        (
            StructuralDelta.NONE,
            StructuralDelta.JOIN,
            _join_patch,
            [_scope("joined")],
            (),
            None,
        ),
        (
            StructuralDelta.FILTER,
            StructuralDelta.EXPAND,
            _expand_patch,
            _filter_or_expand_scopes(),
            ("entity_keys", "cardinality", "links", "memberships", "order", "weights"),
            _structural_context(),
        ),
        (
            StructuralDelta.EXPAND,
            StructuralDelta.FILTER,
            _filter_patch,
            _filter_or_expand_scopes(),
            ("entity_keys", "cardinality", "links", "memberships", "order", "weights"),
            _structural_context(),
        ),
        (
            StructuralDelta.JOIN,
            StructuralDelta.NONE,
            _none_patch,
            [_scope("value")],
            (),
            None,
        ),
        (
            StructuralDelta.RELINK,
            StructuralDelta.NONE,
            _none_patch,
            [_scope("value")],
            ("links", "memberships"),
            None,
        ),
        (
            StructuralDelta.REORDER,
            StructuralDelta.NONE,
            _none_patch,
            [_scope("value")],
            ("order", "weights"),
            None,
        ),
        (
            StructuralDelta.REWEIGHT,
            StructuralDelta.NONE,
            _none_patch,
            [_scope("value")],
            ("weights", "mass_history"),
            None,
        ),
    ],
)
def test_every_structural_delta_refuses_a_mismatched_patch_kind(
    declared: StructuralDelta,
    returned: StructuralDelta,
    kernel,
    scopes: list[dict[str, object]],
    changed_mutations: tuple[str, ...],
    row_classifiers: Mapping[str, RegisteredRowClassifier] | None,
    projection: ImmutableFrameProjection,
) -> None:
    def wrong_kind(view, selected_context):
        return replace(
            kernel(view, selected_context),
            structural_delta=returned,
        )

    with pytest.raises(StructuralDiffError, match="patch delta"):
        _run(
            _node(declared, scopes, changed_mutations=changed_mutations),
            projection,
            wrong_kind,
            row_classifiers=row_classifiers,
        )


def test_adversarial_out_of_scope_row_write_is_refused_without_mutation(
    projection: ImmutableFrameProjection,
) -> None:
    def kernel(view, _context):
        person = view.table("person")
        person.loc[person.person_id.eq(3), "value"] = -1.0
        return KernelPatch(StructuralDelta.NONE, tables={"person": person})

    original = projection.detached_copy()
    with pytest.raises(PatchScopeError, match="outside row scope"):
        _run(
            _node(StructuralDelta.NONE, [_scope("value", "a_rows")]),
            projection,
            kernel,
        )
    assert diff_projections(original, projection).empty


def test_adversarial_wrong_column_and_entity_are_refused(
    projection: ImmutableFrameProjection,
) -> None:
    def wrong_column(view, _context):
        person = view.table("person")
        person.loc[0, "group_id"] = 20
        return KernelPatch(StructuralDelta.NONE, tables={"person": person})

    with pytest.raises(PatchScopeError, match="no compiled write scope"):
        _run(
            _node(StructuralDelta.NONE, [_scope("value")]),
            projection,
            wrong_column,
        )

    def wrong_entity(view, _context):
        group = view.table("group")
        group.loc[0, "group_value"] = -1.0
        return KernelPatch(StructuralDelta.NONE, tables={"group": group})

    with pytest.raises(PatchScopeError, match="no compiled write scope"):
        _run(
            _node(StructuralDelta.NONE, [_scope("value")]),
            projection,
            wrong_entity,
        )


def test_expand_refuses_an_undeclared_added_row_column(
    projection: ImmutableFrameProjection,
) -> None:
    scopes = [
        scope
        for scope in _filter_or_expand_scopes()
        if not (scope["entity"] == "person" and scope["column"] == "value")
    ]
    with pytest.raises(PatchScopeError, match="person.value"):
        _run(
            _node(
                StructuralDelta.EXPAND,
                scopes,
                changed_mutations=(
                    "entity_keys",
                    "cardinality",
                    "links",
                    "memberships",
                    "order",
                    "weights",
                ),
            ),
            projection,
            _expand_patch,
            row_classifiers=_structural_context(),
        )


def test_link_only_and_out_of_scope_relinks_are_refused(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(
        StructuralDelta.RELINK,
        [_scope("group_id", "a_rows", mode="structural_column")],
        changed_mutations=("links", "memberships"),
    )

    def link_only(view, _context):
        link = view.link("membership")
        link.loc[link.person_id.eq(1), "group_id"] = 20
        return KernelPatch(StructuralDelta.RELINK, links={"membership": link})

    with pytest.raises(PatchScopeError, match="not an exact changed membership mirror"):
        _run(node, projection, link_only)

    def wrong_row(view, _context):
        person = view.table("person")
        person.loc[person.person_id.eq(1), "group_id"] = 20
        person.loc[person.person_id.eq(3), "group_id"] = 10
        link = view.link("membership")
        link.loc[link.person_id.eq(1), "group_id"] = 20
        link.loc[link.person_id.eq(3), "group_id"] = 10
        return KernelPatch(
            StructuralDelta.RELINK,
            tables={"person": person},
            links={"membership": link},
        )

    with pytest.raises(PatchScopeError, match="outside row scope"):
        _run(node, projection, wrong_row)


def test_full_diff_refuses_dtype_column_order_metadata_and_virtual_escapes(
    projection: ImmutableFrameProjection,
) -> None:
    def dtype_kernel(view, _context):
        person = view.table("person")
        person["value"] = person.value.astype("float32")
        return KernelPatch(StructuralDelta.NONE, tables={"person": person})

    with pytest.raises(StructuralDiffError, match="dtype"):
        _run(_node(StructuralDelta.NONE, [_scope("value")]), projection, dtype_kernel)

    def removed_column(view, _context):
        return KernelPatch(
            StructuralDelta.NONE,
            tables={"person": view.table("person").drop(columns="value")},
        )

    with pytest.raises(StructuralDiffError, match="column removal"):
        _run(
            _node(StructuralDelta.NONE, [_scope("value")]),
            projection,
            removed_column,
        )

    def reordered_columns(view, _context):
        person = view.table("person")[["person_id", "value", "group_id"]]
        return KernelPatch(StructuralDelta.NONE, tables={"person": person})

    with pytest.raises(StructuralDiffError, match="column order"):
        _run(
            _node(StructuralDelta.NONE, [_scope("value")]),
            projection,
            reordered_columns,
        )

    def index_kernel(view, _context):
        person = view.table("person")
        person.index = pd.Index([10, 11, 12])
        strata = view.strata
        assert strata is not None
        strata.index = person.index
        return KernelPatch(
            StructuralDelta.NONE,
            tables={"person": person},
            replace_strata=True,
            strata=strata,
        )

    with pytest.raises(
        StructuralDiffError,
        match=(
            "stable entity rows changed index labels|changed frame structure|"
            "axis metadata"
        ),
    ):
        _run(_node(StructuralDelta.NONE, [_scope("value")]), projection, index_kernel)

    def metadata_kernel(_view, _context):
        return KernelPatch(StructuralDelta.NONE, metadata={"changed": True})

    with pytest.raises(StructuralDiffError, match="metadata"):
        _run(
            _node(StructuralDelta.NONE, [_scope("value")]),
            projection,
            metadata_kernel,
        )

    def virtual_kernel(_view, _context):
        return KernelPatch(
            StructuralDelta.NONE,
            virtual_writes={("frame", "@receipt"): {"ok": True}},
        )

    with pytest.raises(PatchScopeError, match="undeclared virtual"):
        _run(_node(StructuralDelta.NONE, [_scope("value")]), projection, virtual_kernel)


def test_declared_virtual_receipt_is_allowed(
    projection: ImmutableFrameProjection,
) -> None:
    def kernel(_view, _context):
        return KernelPatch(
            StructuralDelta.NONE,
            virtual_writes={("frame", "@receipt"): {"ok": True}},
        )

    node = _node(
        StructuralDelta.NONE,
        [_scope("@receipt", entity="frame", mode="virtual_receipt")],
    )
    result = apply_patch(projection, _run(node, projection, kernel))
    assert result.virtual_receipts[("frame", "@receipt")] == {"ok": True}


@pytest.mark.parametrize(
    ("node", "context", "message"),
    [
        (
            _node(
                StructuralDelta.NONE,
                [_scope("value")],
                omit_capability="retry_safety",
            ),
            ExecutionContext(),
            "exactly five",
        ),
        (
            _node(
                StructuralDelta.NONE,
                [_scope("value")],
                effects=["none", "declared_source_read"],
            ),
            ExecutionContext(granted_effects=frozenset({Effect.DECLARED_SOURCE_READ})),
            "exclusive",
        ),
        (
            _node(
                StructuralDelta.NONE,
                [_scope("value")],
                determinism="seeded",
            ),
            ExecutionContext(),
            "no effective compiled RNG grant",
        ),
        (
            _node(
                StructuralDelta.NONE,
                [_scope("value")],
                seed_streams=("fixture",),
            ),
            ExecutionContext(),
            "unexpectedly has RNG grants",
        ),
        (
            _node(
                StructuralDelta.NONE,
                [_scope("value")],
                retry_safety="attempt_scoped",
            ),
            ExecutionContext(),
            "requires an attempt scope",
        ),
        (
            _node(
                StructuralDelta.NONE,
                [_scope("value")],
                retry_safety="nonretryable",
            ),
            ExecutionContext(attempt=1, resumed=True),
            "cannot be retried",
        ),
    ],
)
def test_capability_failures_happen_before_kernel_dispatch(
    node: CompiledNode,
    context: ExecutionContext,
    message: str,
    projection: ImmutableFrameProjection,
) -> None:
    calls = 0

    def kernel(_view, _context):
        nonlocal calls
        calls += 1
        return KernelPatch(StructuralDelta.NONE)

    with pytest.raises(CapabilityError, match=message):
        _run(node, projection, kernel, context=context)
    assert calls == 0


def test_unknown_and_axis_incompatible_mutation_tuples_fail_before_dispatch(
    projection: ImmutableFrameProjection,
) -> None:
    calls = 0

    def kernel(_view, _context):
        nonlocal calls
        calls += 1
        return KernelPatch(StructuralDelta.NONE)

    for triple in (
        ("invented", "entity_keys_valid", "entity_keys_unchanged"),
        ("replace_weights", "weights_valid", "weights_valid"),
    ):
        node = _node(
            StructuralDelta.NONE,
            [_scope("value")],
            mutation_overrides={"entity_keys": triple},
        )
        with pytest.raises(CapabilityError, match="unknown or axis-incompatible"):
            _run(node, projection, kernel)
    assert calls == 0


def test_delta_incompatible_valid_mutation_fails_before_dispatch(
    projection: ImmutableFrameProjection,
) -> None:
    calls = 0

    def kernel(_view, _context):
        nonlocal calls
        calls += 1
        return KernelPatch(StructuralDelta.NONE)

    node = _node(
        StructuralDelta.NONE,
        [_scope("value")],
        mutation_overrides={
            "entity_keys": (
                "filter_entity_keys",
                "entity_keys_valid",
                "remaining_entity_keys_unique",
            )
        },
    )
    with pytest.raises(CapabilityError, match="incompatible mutation operations"):
        _run(node, projection, kernel)
    assert calls == 0


def test_lifted_contract_tampering_is_refused_before_dispatch(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value")])
    changed = replace(
        node,
        capabilities=_frozen_mapping(
            {
                "determinism": "nondeterministic",
                "numeric_reproducibility": "unspecified",
                "effects": ["none"],
                "structural_delta": "none",
                "retry_safety": "idempotent",
            }
        ),
    )
    calls = 0

    def kernel(_view, _context):
        nonlocal calls
        calls += 1
        return KernelPatch(StructuralDelta.NONE)

    with pytest.raises(CapabilityError, match="capability lift differs"):
        _run(
            changed,
            projection,
            kernel,
            context=ExecutionContext(require_byte_equivalence=False),
        )
    assert calls == 0


def test_resolved_param_node_slice_node_key_and_rank_tamper_are_refused(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value")])
    first = node.resolved_params[0]
    cases = (
        (
            replace(
                node,
                resolved_params=(
                    replace(first, value_sha256="0" * 64),
                    *node.resolved_params[1:],
                ),
            ),
            "resolved-param digest",
        ),
        (replace(node, node_slice_sha256="0" * 64), "node-slice digest"),
        (replace(node, node_key="0" * 64), "node key differs"),
        (replace(node, execution_rank=1), "execution-rank lift differs"),
    )
    calls = 0

    def kernel(_view, _context):
        nonlocal calls
        calls += 1
        return KernelPatch(StructuralDelta.NONE)

    for tampered, message in cases:
        with pytest.raises(CapabilityError, match=message):
            _run(tampered, projection, kernel)
    assert calls == 0


def test_node_bound_scope_registry_refuses_broader_direct_and_param_substitution(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value", "a_rows")])
    broader = ClosedScopeRegistry(
        "fixture_rows",
        (
            "origin:a/clone:0",
            "origin:b/clone:0",
            "origin:b/clone:1",
        ),
        {
            "a_rows": (
                "origin:a/clone:0",
                "origin:b/clone:0",
                "origin:b/clone:1",
            ),
            "b_rows": ("origin:b/clone:0", "origin:b/clone:1"),
            "whole": (
                "origin:a/clone:0",
                "origin:b/clone:0",
                "origin:b/clone:1",
            ),
        },
    )

    direct = replace(node, scope_registry=_frozen_mapping(broader.to_wire()))
    with pytest.raises(CapabilityError, match="scope-registry lift differs"):
        _run(direct, projection, _none_patch)

    registry_index = next(
        index
        for index, param in enumerate(node.resolved_params)
        if param.path.endswith("/scope_registry")
    )
    replaced_params = list(node.resolved_params)
    replaced_params[registry_index] = _resolved_param(
        "/imputation/producer_graph/scope_registry", broader.to_wire()
    )
    direct_and_param = replace(
        node,
        scope_registry=_frozen_mapping(broader.to_wire()),
        resolved_params=tuple(replaced_params),
    )
    with pytest.raises(CapabilityError, match="node-slice digest"):
        _run(direct_and_param, projection, _none_patch)


def test_private_nested_object_mutation_is_detected_without_aliasing_caller(
    projection: ImmutableFrameProjection,
) -> None:
    parts = projection._parts()
    person = parts["tables"]["person"]
    person["payload"] = [
        {"nested": [1]},
        {"nested": [2]},
        {"nested": [3]},
    ]
    nested_projection = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]

    def malicious(view, _context):
        view._tables["person"].at[0, "payload"]["nested"].append(99)
        return KernelPatch(StructuralDelta.NONE)

    with pytest.raises(ExecutorError, match="mutated its immutable input"):
        _run(
            _node(StructuralDelta.NONE, [_scope("payload", "a_rows")]),
            nested_projection,
            malicious,
        )
    assert nested_projection.table("person").at[0, "payload"] == {"nested": [1]}


def test_kernel_forged_row_atoms_are_refused(
    projection: ImmutableFrameProjection,
) -> None:
    def kernel(_view, _context):
        return KernelPatch(
            StructuralDelta.NONE,
            row_atoms={
                "person": {
                    1: {"origin:b/clone:0"},
                    2: {"origin:a/clone:0"},
                    3: {"origin:b/clone:0"},
                }
            },
        )

    with pytest.raises(PatchScopeError, match="kernels cannot issue row-scope"):
        _run(_node(StructuralDelta.NONE, [_scope("value")]), projection, kernel)


def _validated_none_patch(projection: ImmutableFrameProjection):
    return _run(
        _node(StructuralDelta.NONE, [_scope("value", "a_rows")]),
        projection,
        _none_patch,
    )


def test_validated_patch_base_result_and_kernel_patch_tamper_seals(
    projection: ImmutableFrameProjection,
) -> None:
    base_tamper = _validated_none_patch(projection)
    base_tamper._base._tables["person"].loc[0, "value"] = 101.0
    with pytest.raises(ExecutorError, match="base seal was mutated"):
        apply_patch(projection, base_tamper)

    result_tamper = _validated_none_patch(projection)
    result_tamper._result._tables["person"].loc[0, "value"] = 102.0
    with pytest.raises(ExecutorError, match="result seal was mutated"):
        apply_patch(projection, result_tamper)

    patch_tamper = _validated_none_patch(projection)
    patch_tamper.patch.tables["person"].loc[0, "value"] = 103.0
    with pytest.raises(ExecutorError, match="patch receipt was mutated"):
        apply_patch(projection, patch_tamper)


def test_apply_patch_refuses_swapped_and_aborted_broker_receipts(
    projection: ImmutableFrameProjection,
) -> None:
    first = _validated_none_patch(projection)

    def resealed_receipt(**changes):
        provisional = replace(
            first.broker_receipt,
            receipt_sha256="",
            **changes,
        )
        return replace(
            provisional,
            receipt_sha256=sha256_json(provisional.body_wire()),
        )

    second = _run(
        _node(
            StructuralDelta.NONE,
            [_scope("value", "a_rows")],
            node_id="other_fixture_node",
        ),
        projection,
        _none_patch,
    )
    with pytest.raises(ExecutorError, match="receipt owner differs"):
        apply_patch(projection, replace(first, broker_receipt=second.broker_receipt))

    non_node_owner = resealed_receipt(
        owner=replace(first.broker_receipt.owner, kind="source_stage")
    )
    with pytest.raises(ExecutorError, match="receipt owner differs"):
        apply_patch(projection, replace(first, broker_receipt=non_node_owner))

    wrong_node_key = resealed_receipt(node_key="f" * 64)
    with pytest.raises(ExecutorError, match="receipt node key differs"):
        apply_patch(projection, replace(first, broker_receipt=wrong_node_key))

    aborted = resealed_receipt(status="aborted")
    aborted.validate()
    with pytest.raises(ExecutorError, match="complete broker receipt"):
        apply_patch(projection, replace(first, broker_receipt=aborted))

    with pytest.raises(ExecutorError, match="invalid type"):
        apply_patch(
            projection,
            replace(first, broker_receipt=object()),  # type: ignore[arg-type]
        )


def test_apply_patch_requires_exact_receipt_attempt_and_scope(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value", "a_rows")])
    first = _run(
        node,
        projection,
        _none_patch,
        context=ExecutionContext(attempt=3, attempt_scope="attempt:3"),
    )
    assert (first.attempt, first.attempt_scope) == (3, "attempt:3")
    assert (first.broker_receipt.attempt, first.broker_receipt.attempt_scope) == (
        3,
        "attempt:3",
    )

    for context, expected_message in (
        (ExecutionContext(attempt=4, attempt_scope="attempt:3"), "attempt differs"),
        (ExecutionContext(attempt=3, attempt_scope="attempt:other"), "scope differs"),
    ):
        other = _run(node, projection, _none_patch, context=context)
        with pytest.raises(ExecutorError, match=expected_message):
            apply_patch(
                projection,
                replace(first, broker_receipt=other.broker_receipt),
            )


def test_patch_cannot_be_replayed_on_a_different_base(
    projection: ImmutableFrameProjection,
) -> None:
    validated = _validated_none_patch(projection)
    parts = projection._parts()
    parts["tables"]["person"].loc[1, "value"] = 999.0
    different = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]
    with pytest.raises(ExecutorError, match="base projection differs"):
        apply_patch(different, validated)


def test_reorder_requires_weights_realigned_by_stable_id(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(
        StructuralDelta.REORDER,
        [_scope("value"), _scope("@resolved_weight", mode="resolved_weight")],
        changed_mutations=("order", "weights"),
    )
    good = apply_patch(projection, _run(node, projection, _reorder_patch))
    ids = good.table("person")["person_id"].tolist()
    assert dict(zip(ids, good.weights_for("person").values, strict=True)) == {
        1: 1.0,
        2: 2.0,
        3: 3.0,
    }

    def bad(view, _context):
        person = (
            view.table("person")
            .set_index("person_id", drop=False)
            .loc[[3, 1, 2]]
            .reset_index(drop=True)
        )
        return KernelPatch(
            StructuralDelta.REORDER,
            tables={"person": person},
            replace_strata=True,
            strata=pd.Series(["z", "x", "y"], name="stratum"),
        )

    with pytest.raises(StructuralDiffError, match="reorder changed.*weights"):
        _run(node, projection, bad)


def test_join_cannot_smuggle_an_added_row(
    projection: ImmutableFrameProjection,
) -> None:
    def kernel(view, _context):
        person = view.table("person")
        person["joined"] = [100, 200, 300]
        person = pd.concat(
            [
                person,
                pd.DataFrame(
                    {
                        "person_id": [4],
                        "group_id": [20],
                        "value": [40.0],
                        "joined": [400],
                    }
                ),
            ],
            ignore_index=True,
        )
        return KernelPatch(
            StructuralDelta.JOIN,
            tables={"person": person},
            weights={"person": WeightState([1.0, 2.0, 3.0, 1.0], "design")},
            replace_strata=True,
            strata=pd.Series(["x", "y", "z", "w"], name="stratum"),
        )

    scopes = [
        _scope(column, "whole")
        for column in ("person_id", "group_id", "value", "joined")
    ] + [_scope("@resolved_weight", mode="resolved_weight")]
    with pytest.raises(
        (StructuralDiffError, PatchScopeError),
        match="row-preserving|added rows",
    ):
        _run(
            _node(StructuralDelta.JOIN, scopes),
            projection,
            kernel,
            row_classifiers=_structural_context(),
        )


def test_nan_and_nullable_values_are_exactly_unchanged(
    projection: ImmutableFrameProjection,
) -> None:
    parts = projection._parts()
    person = parts["tables"]["person"]
    person["float_missing"] = [np.nan, 1.0, np.nan]
    person["nullable"] = pd.Series([pd.NA, "x", pd.NA], dtype="string")
    missing_projection = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]

    def identity(view, _context):
        return KernelPatch(
            StructuralDelta.NONE,
            tables={"person": view.table("person")},
        )

    validated = _run(
        _node(
            StructuralDelta.NONE,
            [_scope("value"), _scope("float_missing"), _scope("nullable")],
        ),
        missing_projection,
        identity,
    )
    assert validated.diff.empty
    assert diff_projections(
        missing_projection, apply_patch(missing_projection, validated)
    ).empty


def test_deterministic_compiled_rank_kahn_order_ignores_input_permutation() -> None:
    a = _node(
        StructuralDelta.NONE,
        [_scope("a_value")],
        node_id="a",
        execution_rank=0,
    )
    b = _node(
        StructuralDelta.NONE,
        [_scope("b_value")],
        node_id="b",
        execution_rank=1,
    )
    c = _node(
        StructuralDelta.NONE,
        [_scope("c_value")],
        node_id="c",
        execution_rank=2,
        depends_on=("b",),
        transitive_nodes=(_transitive_slice(b),),
    )
    for permutation in ((c, b, a), (b, a, c), (a, c, b)):
        assert [node.id for node in order_nodes(permutation)] == ["a", "b", "c"]


def test_order_requires_contiguous_ranks_and_lower_dependency_rank() -> None:
    rank_zero = _node(
        StructuralDelta.NONE,
        [_scope("zero")],
        node_id="zero",
        execution_rank=0,
    )
    rank_two = _node(
        StructuralDelta.NONE,
        [_scope("two")],
        node_id="two",
        execution_rank=2,
    )
    with pytest.raises(NodeOrderingError, match="contiguous"):
        order_nodes((rank_two, rank_zero))

    later = _node(
        StructuralDelta.NONE,
        [_scope("later")],
        node_id="later",
        execution_rank=1,
    )
    earlier = _node(
        StructuralDelta.NONE,
        [_scope("earlier")],
        node_id="earlier",
        execution_rank=0,
        depends_on=("later",),
        transitive_nodes=(_transitive_slice(later),),
    )
    with pytest.raises(NodeOrderingError, match="lower execution rank"):
        order_nodes((later, earlier))


def test_incomparable_exact_and_structural_write_conflicts_are_refused() -> None:
    left = _node(
        StructuralDelta.NONE,
        [_scope("value", "a_rows")],
        node_id="left",
        execution_rank=0,
    )
    right = _node(
        StructuralDelta.NONE,
        [_scope("value", "a_rows")],
        node_id="right",
        execution_rank=1,
    )
    with pytest.raises(NodeOrderingError, match="overlapping exact writes"):
        order_nodes((right, left))

    changed = ("entity_keys", "cardinality", "links", "memberships", "order", "weights")
    structural_left = _node(
        StructuralDelta.FILTER,
        [_scope("left_value", "a_rows")],
        node_id="structural_left",
        execution_rank=0,
        changed_mutations=changed,
    )
    structural_right = _node(
        StructuralDelta.FILTER,
        [_scope("right_value", "b_rows")],
        node_id="structural_right",
        execution_rank=1,
        changed_mutations=changed,
    )
    with pytest.raises(NodeOrderingError, match="overlapping structural resources"):
        order_nodes((structural_right, structural_left))


def test_executor_source_has_no_legacy_escape_or_country_literals() -> None:
    source_root = Path(__file__).parents[1] / "src/microcosm/build/spec_engine"
    source = "\n".join(
        (source_root / filename).read_text(encoding="utf-8")
        for filename in ("executor.py", "scope_algebra.py")
    )
    assert "structural_effect" not in source
    for forbidden in ("snap", "medicaid", "tax_unit", "asec", "acs", "puf"):
        assert forbidden not in source.lower()


def test_diff_is_exact_even_for_tolerance_bound_nodes(
    projection: ImmutableFrameProjection,
) -> None:
    def tiny_change(view, _context):
        person = view.table("person")
        person.loc[0, "value"] = np.nextafter(person.loc[0, "value"], np.inf)
        return KernelPatch(StructuralDelta.NONE, tables={"person": person})

    node = _node(
        StructuralDelta.NONE,
        [_scope("value", "a_rows")],
        numeric_reproducibility="tolerance_bound",
    )
    validated = _run(node, projection, tiny_change)
    assert len(validated.diff.tables[0].cell_changes) == 1


@pytest.mark.parametrize(
    ("before_value", "after_value"),
    [
        (0.0, -0.0),
        (1, True),
        (None, np.nan),
    ],
    ids=("positive-zero-to-negative-zero", "integer-to-bool", "none-to-nan"),
)
def test_cell_diff_is_type_and_bit_exact(
    projection: ImmutableFrameProjection,
    before_value: object,
    after_value: object,
) -> None:
    parts = projection._parts()
    person = parts["tables"]["person"]
    person["exact_object"] = pd.Series(
        [before_value, "unchanged", object()],
        dtype=object,
    )
    before = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]
    after_parts = before._parts()
    after_parts["tables"]["person"].at[0, "exact_object"] = after_value
    after = ImmutableFrameProjection(**after_parts)  # type: ignore[arg-type]

    changes = [
        change
        for table in diff_projections(before, after).tables
        for change in table.cell_changes
        if change.column == "exact_object"
    ]
    assert len(changes) == 1
    assert changes[0].row_id == 1


@pytest.mark.parametrize("surface", ["table", "link"])
def test_diff_detects_column_index_class_substitution(
    projection: ImmutableFrameProjection,
    surface: str,
) -> None:
    parts = projection._parts()
    if surface == "table":
        frame = parts["tables"]["person"]
    else:
        frame = parts["links"]["membership"]
    assert type(frame.columns) is pd.Index
    frame.columns = pd.CategoricalIndex(frame.columns)
    changed = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]

    diff = diff_projections(projection, changed)
    assert not diff.empty
    if surface == "table":
        assert [row.entity for row in diff.tables] == ["person"]
    else:
        assert [row.name for row in diff.links] == ["membership"]


@pytest.mark.parametrize("surface", ["name", "attrs"])
def test_none_refuses_strata_axis_metadata_changes(
    projection: ImmutableFrameProjection,
    surface: str,
) -> None:
    def kernel(view, _context):
        strata = view.strata
        assert strata is not None
        if surface == "name":
            strata.name = "substituted_stratum"
        else:
            strata.attrs["substituted"] = True
        return KernelPatch(
            StructuralDelta.NONE,
            replace_strata=True,
            strata=strata,
        )

    with pytest.raises(StructuralDiffError, match="strata"):
        _run(_node(StructuralDelta.NONE, [_scope("value")]), projection, kernel)


def test_none_refuses_a_declared_but_structural_added_column(
    projection: ImmutableFrameProjection,
) -> None:
    def kernel(view, _context):
        person = view.table("person")
        person["smuggled"] = pd.Series([1, 2, 3], dtype="int64")
        return KernelPatch(StructuralDelta.NONE, tables={"person": person})

    with pytest.raises(StructuralDiffError, match="only structural_delta join"):
        _run(
            _node(StructuralDelta.NONE, [_scope("smuggled")]),
            projection,
            kernel,
        )


@pytest.mark.parametrize(
    ("delta", "surface"),
    [
        (StructuralDelta.FILTER, "table"),
        (StructuralDelta.FILTER, "link"),
        (StructuralDelta.EXPAND, "table"),
        (StructuralDelta.EXPAND, "link"),
    ],
)
def test_cardinality_delta_refuses_stable_table_or_link_index_rewrite(
    projection: ImmutableFrameProjection,
    delta: StructuralDelta,
    surface: str,
) -> None:
    base_kernel = _filter_patch if delta is StructuralDelta.FILTER else _expand_patch

    def kernel(view, context):
        patch = base_kernel(view, context)
        tables = dict(patch.tables)
        links = dict(patch.links)
        strata = patch.strata
        if surface == "table":
            person = tables["person"].copy()
            person.index = pd.Index(
                range(100, 100 + len(person)),
                name="substituted_index",
            )
            tables["person"] = person
            assert strata is not None
            strata = strata.copy()
            strata.index = person.index
        else:
            membership = links["membership"].copy()
            membership.index = pd.Index(
                range(100, 100 + len(membership)),
                name="substituted_index",
            )
            links["membership"] = membership
        return replace(patch, tables=tables, links=links, strata=strata)

    with pytest.raises(StructuralDiffError, match="index|axis metadata"):
        _run(
            _node(
                delta,
                _filter_or_expand_scopes(),
                changed_mutations=(
                    "entity_keys",
                    "cardinality",
                    "links",
                    "memberships",
                    "order",
                    "weights",
                ),
            ),
            projection,
            kernel,
            row_classifiers=(
                _structural_context() if delta is StructuralDelta.EXPAND else None
            ),
        )


def test_filter_weights_must_preserve_surviving_key_alignment(
    projection: ImmutableFrameProjection,
) -> None:
    def kernel(view, context):
        patch = _filter_patch(view, context)
        return replace(
            patch,
            weights={"person": WeightState([2.0, 1.0], "design")},
        )

    with pytest.raises(
        StructuralDiffError,
        match="weights_aligned_to_surviving_keys",
    ):
        scopes = [
            (
                _scope("@resolved_weight", "whole", mode="resolved_weight")
                if scope["column"] == "@resolved_weight"
                else scope
            )
            for scope in _filter_or_expand_scopes()
        ]
        _run(
            _node(
                StructuralDelta.FILTER,
                scopes,
                changed_mutations=(
                    "entity_keys",
                    "cardinality",
                    "links",
                    "memberships",
                    "order",
                    "weights",
                ),
            ),
            projection,
            kernel,
        )


def test_expand_clone_blocks_must_follow_all_native_rows(
    projection: ImmutableFrameProjection,
) -> None:
    def kernel(view, context):
        patch = _expand_patch(view, context)
        person = patch.tables["person"].iloc[[0, 3, 1, 2]].reset_index(drop=True)
        assert patch.strata is not None
        strata = patch.strata.iloc[[0, 3, 1, 2]].reset_index(drop=True)
        link = patch.links["membership"].iloc[[0, 3, 1, 2]].reset_index(drop=True)
        return replace(
            patch,
            tables={**patch.tables, "person": person},
            links={**patch.links, "membership": link},
            weights={"person": WeightState([1.0, 1.5, 2.0, 1.5], "design")},
            strata=strata,
        )

    with pytest.raises(
        StructuralDiffError,
        match="clone_blocks_follow_native_rows|stable entity rows",
    ):
        _run(
            _node(
                StructuralDelta.EXPAND,
                _filter_or_expand_scopes(),
                changed_mutations=(
                    "entity_keys",
                    "cardinality",
                    "links",
                    "memberships",
                    "order",
                    "weights",
                ),
            ),
            projection,
            kernel,
            row_classifiers=_structural_context(),
        )


def test_expand_mass_must_be_conserved_per_parent_lineage(
    projection: ImmutableFrameProjection,
) -> None:
    def kernel(view, context):
        patch = _expand_patch(view, context)
        # Row 4 is a descendant of row 3 in this fixture.  The global mass is
        # unchanged at six, but half a unit moves from row 1's unrelated
        # lineage to the row 3/4 lineage.
        return replace(
            patch,
            weights={"person": WeightState([0.5, 2.0, 1.5, 2.0], "design")},
        )

    scopes = [
        (
            _scope("@resolved_weight", "whole", mode="resolved_weight")
            if scope["column"] == "@resolved_weight"
            else scope
        )
        for scope in _filter_or_expand_scopes()
    ]
    with pytest.raises(StructuralDiffError, match="household_mass_conserved"):
        _run(
            _node(
                StructuralDelta.EXPAND,
                scopes,
                changed_mutations=(
                    "entity_keys",
                    "cardinality",
                    "links",
                    "memberships",
                    "order",
                    "weights",
                ),
            ),
            projection,
            kernel,
            row_classifiers=_structural_context(),
        )


def test_expand_native_clone_index_zero_precondition_is_enforced(
    projection: ImmutableFrameProjection,
) -> None:
    parts = projection._parts()
    # A pre-existing/native row carrying a nonzero clone index is the
    # deliberate precondition violation.
    parts["row_atoms"]["person"][3] = {"origin:b/clone:1"}
    invalid_native = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]
    with pytest.raises(StructuralDiffError, match="native_clone_index_zero"):
        _run(
            _node(
                StructuralDelta.EXPAND,
                _filter_or_expand_scopes(),
                changed_mutations=(
                    "entity_keys",
                    "cardinality",
                    "links",
                    "memberships",
                    "order",
                    "weights",
                ),
            ),
            invalid_native,
            _expand_patch,
            row_classifiers=_structural_context(),
        )


def test_incomparable_structural_and_cell_writes_conflict() -> None:
    structural = _node(
        StructuralDelta.FILTER,
        [_scope("person_id", "b_rows")],
        node_id="structural",
        execution_rank=0,
        changed_mutations=(
            "entity_keys",
            "cardinality",
            "links",
            "memberships",
            "order",
            "weights",
        ),
    )
    cell = _node(
        StructuralDelta.NONE,
        [_scope("value", "b_rows")],
        node_id="cell",
        execution_rank=1,
    )
    with pytest.raises(NodeOrderingError, match="structural.*cell|cell.*structural"):
        order_nodes((cell, structural))


def test_dependency_input_and_output_direct_field_tamper_are_refused_before_dispatch(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value")])
    cases = (
        replace(node, depends_on=("forged_dependency",)),
        replace(
            node,
            inputs=(_frozen_mapping({"entity": "group", "column": "group_value"}),),
        ),
        replace(
            node,
            outputs=(_frozen_mapping({"entity": "person", "column": "value"}),),
        ),
    )
    calls = 0

    def kernel(_view, _context):
        nonlocal calls
        calls += 1
        return KernelPatch(StructuralDelta.NONE)

    for tampered in cases:
        with pytest.raises(CapabilityError, match="lift differs|transitive slice"):
            _run(tampered, projection, kernel)
    assert calls == 0


def test_expand_refuses_a_substituted_row_classifier_implementation(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(
        StructuralDelta.EXPAND,
        _filter_or_expand_scopes(),
        changed_mutations=(
            "entity_keys",
            "cardinality",
            "links",
            "memberships",
            "order",
            "weights",
        ),
    )
    expected = _row_classifier()
    substituted = RegisteredRowClassifier(
        expected.function,
        "d" * 64,
        expected.predicate_space,
    )
    calls = 0

    def kernel(view, context):
        nonlocal calls
        calls += 1
        return _expand_patch(view, context)

    with pytest.raises(
        (CapabilityError, PatchScopeError),
        match="classifier.*(digest|authority|compiled|implementation)",
    ):
        _run(
            node,
            projection,
            kernel,
            row_classifiers={node.row_classifier_ref: substituted},
        )
    assert calls == 0


def test_kernel_cannot_read_an_undeclared_projected_column(
    projection: ImmutableFrameProjection,
) -> None:
    declared_input = {
        "entity": "person",
        "column": "value",
        "producing_stage": "fixture_input",
        "required_scope": "whole",
        "alternatives": [],
        "tolerated_absence_receipts": [],
    }

    def kernel(view, _context):
        leaked = view.table("group")["group_value"].iloc[0]
        person = view.table("person")
        person.loc[person.person_id.eq(1), "value"] = leaked
        return KernelPatch(StructuralDelta.NONE, tables={"person": person})

    with pytest.raises(ExecutorError, match="undeclared.*read"):
        _run(
            _node(
                StructuralDelta.NONE,
                [_scope("value", "a_rows")],
                inputs=(declared_input,),
            ),
            projection,
            kernel,
        )


@pytest.mark.parametrize(
    ("dtype", "values", "message"),
    [
        ("int64", [100.0, 200.0, 300.0], "dtype"),
        ("float64", [100.0, np.nan, 300.0], "nullab"),
    ],
)
def test_compiled_output_contract_enforces_dtype_and_nullability(
    projection: ImmutableFrameProjection,
    dtype: str,
    values: list[float],
    message: str,
) -> None:
    output = {
        "entity": "person",
        "column": "joined",
        "coverage_scope": "whole",
        "temporary": False,
        "validation_only": False,
    }
    column_contract = {
        "key": "person.joined",
        "entity": "person",
        "dtype": dtype,
        "unit": "count",
        "period": "year",
        "vintage": "vintage:fixture",
        "nullable": False,
        "domain": "fixture",
        "public_stability": "internal",
        "unit_waiver": None,
    }
    captured_values = tuple(values)

    def kernel(view, _context):
        person = view.table("person")
        person["joined"] = pd.Series(captured_values, dtype="float64")
        return KernelPatch(StructuralDelta.JOIN, tables={"person": person})

    with pytest.raises(StructuralDiffError, match=message):
        _run(
            _node(
                StructuralDelta.JOIN,
                [_scope("joined")],
                outputs=(output,),
                column_contracts=(column_contract,),
            ),
            projection,
            kernel,
        )


def test_existing_output_contract_enforces_non_nullability(
    projection: ImmutableFrameProjection,
) -> None:
    output = {
        "entity": "person",
        "column": "value",
        "coverage_scope": "whole",
        "temporary": False,
        "validation_only": False,
    }
    column_contract = {
        "key": "person.value",
        "entity": "person",
        "dtype": "float64",
        "unit": "count",
        "period": "year",
        "vintage": "vintage:fixture",
        "nullable": False,
        "domain": "fixture",
        "public_stability": "internal",
        "unit_waiver": None,
    }

    def kernel(view, _context):
        person = view.table("person")
        person.loc[person.person_id.eq(1), "value"] = np.nan
        return KernelPatch(StructuralDelta.NONE, tables={"person": person})

    with pytest.raises(StructuralDiffError, match="nullab"):
        _run(
            _node(
                StructuralDelta.NONE,
                [_scope("value", "a_rows")],
                outputs=(output,),
                column_contracts=(column_contract,),
            ),
            projection,
            kernel,
        )


def test_empty_node_collection_has_a_total_empty_order() -> None:
    assert order_nodes(()) == ()


@pytest.mark.parametrize(
    ("surface", "representation"),
    [
        ("entity", "categorical_dtype"),
        ("entity", "flags"),
        ("link", "categorical_dtype"),
        ("link", "flags"),
    ],
)
def test_full_diff_refuses_piggybacked_unmodeled_pandas_state(
    projection: ImmutableFrameProjection,
    surface: str,
    representation: str,
) -> None:
    parts = projection._parts()
    node_inputs: tuple[dict[str, object], ...] | None = None
    if surface == "entity" and representation == "categorical_dtype":
        parts["tables"]["person"]["category"] = pd.Categorical(
            ["a", "b", "a"],
            categories=["a", "b"],
            ordered=False,
        )
        node_inputs = (
            _input("person", "value"),
            _input("person", "category"),
            _input("group", "group_value"),
        )
    elif surface == "link" and representation == "categorical_dtype":
        parts["links"]["membership"]["person_id"] = pd.Categorical(
            [1, 2, 3],
            categories=[1, 2, 3],
            ordered=False,
        )
    changed_projection = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]

    def kernel(view, _context):
        person = view.table("person")
        person.loc[person.person_id.eq(1), "value"] = 11.0
        patch_tables = {"person": person}
        patch_links = {}
        if surface == "entity":
            if representation == "categorical_dtype":
                person["category"] = person["category"].cat.reorder_categories(
                    ["b", "a"],
                    ordered=False,
                )
            else:
                person.flags.allows_duplicate_labels = False
        else:
            membership = view.link("membership")
            if representation == "categorical_dtype":
                membership["person_id"] = membership[
                    "person_id"
                ].cat.reorder_categories(
                    [3, 2, 1],
                    ordered=False,
                )
            else:
                membership.flags.allows_duplicate_labels = False
            patch_links["membership"] = membership
        return KernelPatch(
            StructuralDelta.NONE,
            tables=patch_tables,
            links=patch_links,
        )

    with pytest.raises(
        ExecutorError,
        match="dtype|flag|contract|metadata|attrs|modeled|structural|link",
    ):
        _run(
            _node(
                StructuralDelta.NONE,
                [_scope("value", "a_rows")],
                inputs=node_inputs,
            ),
            changed_projection,
            kernel,
        )


@pytest.mark.parametrize(
    ("delta", "base_kernel", "scopes", "changed_mutations", "classifiers"),
    [
        (
            StructuralDelta.FILTER,
            _filter_patch,
            _filter_or_expand_scopes(),
            ("entity_keys", "cardinality", "links", "memberships", "order", "weights"),
            None,
        ),
        (
            StructuralDelta.EXPAND,
            _expand_patch,
            _filter_or_expand_scopes(),
            ("entity_keys", "cardinality", "links", "memberships", "order", "weights"),
            _structural_context(),
        ),
        (StructuralDelta.JOIN, _join_patch, [_scope("joined")], (), None),
        (
            StructuralDelta.RELINK,
            _relink_patch,
            [_scope("group_id", "a_rows", mode="structural_column")],
            ("links", "memberships"),
            None,
        ),
        (
            StructuralDelta.REORDER,
            _reorder_patch,
            [_scope("value"), _scope("@resolved_weight", mode="resolved_weight")],
            ("order", "weights"),
            None,
        ),
    ],
)
def test_structural_delta_refuses_unrelated_stable_strata_metadata(
    projection: ImmutableFrameProjection,
    delta: StructuralDelta,
    base_kernel,
    scopes: list[dict[str, object]],
    changed_mutations: tuple[str, ...],
    classifiers: Mapping[str, RegisteredRowClassifier] | None,
) -> None:
    def kernel(view, context):
        patch = base_kernel(view, context)
        strata = patch.strata if patch.replace_strata else view.strata
        assert strata is not None
        strata = strata.copy()
        strata.name = "smuggled_name"
        strata.attrs["smuggled"] = True
        return replace(
            patch,
            replace_strata=True,
            strata=strata,
        )

    with pytest.raises(StructuralDiffError, match="strata"):
        _run(
            _node(delta, scopes, changed_mutations=changed_mutations),
            projection,
            kernel,
            row_classifiers=classifiers,
        )


def test_reorder_refuses_strata_subclass_metadata_change(
    projection: ImmutableFrameProjection,
) -> None:
    parts = projection._parts()
    strata = _StrataWithMetadata(parts["strata"])
    strata.name = parts["strata"].name
    strata.executor_secret = "base"
    parts["strata"] = strata
    subclass_projection = ImmutableFrameProjection(  # type: ignore[arg-type]
        **parts
    )

    def kernel(view, _context):
        person = (
            view.table("person")
            .set_index("person_id", drop=False)
            .loc[[3, 1, 2]]
            .reset_index(drop=True)
        )
        reordered_strata = _StrataWithMetadata(["z", "x", "y"], name="stratum")
        reordered_strata.executor_secret = "smuggled"
        return KernelPatch(
            StructuralDelta.REORDER,
            tables={"person": person},
            weights={"person": WeightState([3.0, 1.0, 2.0], "design")},
            replace_strata=True,
            strata=reordered_strata,
        )

    node = _node(
        StructuralDelta.REORDER,
        [
            _scope("value"),
            _scope("@resolved_weight", mode="resolved_weight"),
        ],
        changed_mutations=("order", "weights"),
    )
    with pytest.raises(StructuralDiffError, match="strata|metadata"):
        _run(node, subclass_projection, kernel)


def test_reorder_refuses_unrelated_unscoped_entity_index_rewrite(
    projection: ImmutableFrameProjection,
) -> None:
    def kernel(view, context):
        patch = _reorder_patch(view, context)
        group = view.table("group")
        group.index = pd.Index([100, 101])
        return replace(
            patch,
            tables={**patch.tables, "group": group},
        )

    with pytest.raises(ExecutorError, match="index|scope|axis metadata"):
        _run(
            _node(
                StructuralDelta.REORDER,
                [
                    _scope("value"),
                    _scope("@resolved_weight", mode="resolved_weight"),
                ],
                changed_mutations=("order", "weights"),
            ),
            projection,
            kernel,
        )


def test_none_refuses_representation_change_to_an_entity_key() -> None:
    key_projection = ImmutableFrameProjection(
        {
            "person": pd.DataFrame(
                {
                    "person_id": pd.Series([0.0, 1.0], dtype=object),
                    "value": [10.0, 20.0],
                }
            )
        },
        entity_keys={"person": "person_id"},
        weights={"person": WeightState([1.0, 1.0], "design")},
        row_atoms={
            "person": {
                0.0: {"origin:a/clone:0"},
                1.0: {"origin:a/clone:0"},
            }
        },
    )

    def kernel(view, _context):
        person = view.table("person")
        person.at[0, "person_id"] = -0.0
        return KernelPatch(StructuralDelta.NONE, tables={"person": person})

    with pytest.raises(StructuralDiffError, match="key|entity"):
        _run(
            _node(
                StructuralDelta.NONE,
                [_scope("person_id", mode="structural_column")],
                inputs=(_input("person", "value"),),
            ),
            key_projection,
            kernel,
        )


def test_relink_requires_representation_exact_membership_mirror() -> None:
    exact_projection = ImmutableFrameProjection(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "group_id": pd.Series([1.0, 0.0], dtype=object),
                    "value": [10.0, 20.0],
                }
            ),
            "group": pd.DataFrame(
                {
                    "group_key": pd.Series([0.0, 1.0], dtype=object),
                    "group_value": [100.0, 200.0],
                }
            ),
        },
        entity_keys={"person": "person_id", "group": "group_key"},
        membership_columns={"person": ("group_id",)},
        membership_targets={"person": {"group_id": "group"}},
        links={
            "membership": pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "group_id": pd.Series([1.0, 0.0], dtype=object),
                }
            )
        },
        link_targets={"membership": {"person_id": "person", "group_id": "group"}},
        weights={"person": WeightState([1.0, 1.0], "design")},
        row_atoms={
            "person": {
                1: {"origin:a/clone:0"},
                2: {"origin:a/clone:0"},
            },
            "group": {
                0.0: {"origin:a/clone:0"},
                1.0: {"origin:a/clone:0"},
            },
        },
    )

    def kernel(view, _context):
        person = view.table("person")
        person["group_id"] = pd.Series([-0.0, 1.0], dtype=object)
        membership = view.link("membership")
        membership["group_id"] = pd.Series([+0.0, 1.0], dtype=object)
        return KernelPatch(
            StructuralDelta.RELINK,
            tables={"person": person},
            links={"membership": membership},
        )

    with pytest.raises(ExecutorError, match="membership|link|exact|reference"):
        _run(
            _node(
                StructuralDelta.RELINK,
                [_scope("group_id", mode="structural_column")],
                changed_mutations=("links", "memberships"),
                inputs=(
                    _input("person", "value"),
                    _input("group", "group_value"),
                ),
            ),
            exact_projection,
            kernel,
        )


def test_authored_kernel_pin_refuses_direct_ref_and_digest_substitution(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value", "a_rows")])
    forged_ref = "kernel:forged"
    forged_sha256 = "b" * 64
    forged_key = sha256_json(
        {
            "domain": _NODE_KEY_DOMAIN,
            "compiler_ir_abi": current_compiler_ir_abi().to_wire(),
            "node_slice_sha256": node.node_slice_sha256,
            "kernel": {
                "ref": forged_ref,
                "implementation_sha256": forged_sha256,
            },
            "seed_protocol_sha256": node.seed_protocol_sha256,
        }
    )
    forged_node = replace(
        node,
        kernel_ref=forged_ref,
        kernel_implementation_sha256=forged_sha256,
        node_key=forged_key,
    )
    calls = 0

    def forged_kernel(view, _context):
        nonlocal calls
        calls += 1
        person = view.table("person")
        person.loc[person.person_id.eq(1), "value"] = 999.0
        return KernelPatch(StructuralDelta.NONE, tables={"person": person})

    with pytest.raises(CapabilityError, match="kernel.*lift|kernel.*pin"):
        execute_node(
            forged_node,
            projection,
            kernels={
                forged_ref: RegisteredKernel(forged_kernel, forged_sha256),
            },
            context=ExecutionContext(run_provenance_identity=_fixture_run_provenance()),
        )
    assert calls == 0


def test_projection_refuses_a_missing_required_physical_input(
    projection: ImmutableFrameProjection,
) -> None:
    parts = projection._parts()
    parts["tables"]["group"] = parts["tables"]["group"].drop(columns="group_value")
    missing_input = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]
    calls = 0

    def kernel(view, context):
        nonlocal calls
        calls += 1
        return _none_patch(view, context)

    with pytest.raises(
        CapabilityError, match="input.*group.*group_value|missing.*input"
    ):
        _run(
            _node(StructuralDelta.NONE, [_scope("value", "a_rows")]),
            missing_input,
            kernel,
        )
    assert calls == 0


@pytest.mark.parametrize("ambient_kind", ["environment", "clock"])
def test_executor_refuses_ambient_environment_and_clock_before_patch_application(
    ambient_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    projection: ImmutableFrameProjection,
) -> None:
    monkeypatch.setenv("MICROCOSM_EXECUTOR_SENTINEL", "visible-outside-guard")
    if ambient_kind == "clock":
        monkeypatch.setattr(time, "time", lambda: 123.5)
    node = _node(StructuralDelta.NONE, [_scope("value", "a_rows")])
    brokers = _broker_session(node)
    original = projection.detached_copy()

    def kernel(view, context):
        assert context.brokers is not None
        if ambient_kind == "environment":
            os.environ.get("MICROCOSM_EXECUTOR_SENTINEL")
        else:
            time.time()
        return _none_patch(view, context)

    with pytest.raises(AmbientAccessError, match=f"ambient {ambient_kind}"):
        _run(
            node,
            projection,
            kernel,
            context=ExecutionContext(brokers=brokers),
        )
    assert diff_projections(original, projection).empty
    assert brokers.receipt.status == "aborted"
    assert brokers.receipt.events[-1].disposition == "refused"
    assert os.environ["MICROCOSM_EXECUTOR_SENTINEL"] == "visible-outside-guard"
    if ambient_kind == "clock":
        assert time.time() == 123.5


def test_caught_ambient_refusal_still_aborts_the_executor_receipt(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value", "a_rows")])
    brokers = _broker_session(node)

    def kernel(view, context):
        try:
            time.time()
        except AmbientAccessError:
            pass
        return _none_patch(view, context)

    with pytest.raises(BrokerAccessError, match="recorded a refused access"):
        _run(
            node,
            projection,
            kernel,
            context=ExecutionContext(brokers=brokers),
        )
    assert brokers.receipt.status == "aborted"
    assert brokers.receipt.events[-1].disposition == "refused"


def test_seeded_kernel_can_only_draw_from_its_compiled_site_token(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(
        StructuralDelta.NONE,
        [_scope("value", "a_rows")],
        determinism="seeded",
        seed_streams=("fixture",),
    )
    brokers = _broker_session(node, run_inputs={"fixture_seed": 17})

    def kernel(view, context):
        token = context.brokers.rng.token("fixture_site_0")
        with context.brokers.rng.generator(token) as generator:
            assert generator.integers(0, 100, size=3).tolist() == [74, 84, 10]
        return _none_patch(view, context)

    validated = _run(
        node,
        projection,
        kernel,
        context=ExecutionContext(brokers=brokers),
    )
    assert validated.broker_receipt is brokers.receipt
    rng_events = [
        event for event in validated.broker_receipt.events if event.broker == "rng"
    ]
    assert rng_events[0].resource == "fixture_site_0"
    assert rng_events[0].details["stream"] == "stream:fixture"


def test_seeded_kernel_private_numpy_rng_is_refused(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(
        StructuralDelta.NONE,
        [_scope("value", "a_rows")],
        determinism="seeded",
        seed_streams=("fixture",),
    )
    brokers = _broker_session(node, run_inputs={"fixture_seed": 17})

    def kernel(view, context):
        np.random.default_rng(17)
        return _none_patch(view, context)

    with pytest.raises(AmbientAccessError, match="ambient rng"):
        _run(
            node,
            projection,
            kernel,
            context=ExecutionContext(brokers=brokers),
        )
    assert brokers.receipt.status == "aborted"


def test_executor_refuses_prebound_clock_alias_before_dispatch(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value", "a_rows")])
    brokers = _broker_session(node)
    captured_time = time.time
    calls = 0

    def kernel(view, context):
        nonlocal calls
        calls += 1
        captured_time()
        return _none_patch(view, context)

    with pytest.raises(AmbientAccessError, match="captures prohibited ambient"):
        _run(
            node,
            projection,
            kernel,
            context=ExecutionContext(brokers=brokers),
        )
    assert calls == 0
    assert brokers.receipt.status == "aborted"
    assert brokers.receipt.events[-1].reason_code == "prebound_ambient_access"


def test_executor_refuses_clock_alias_reached_through_global_helper(
    monkeypatch: pytest.MonkeyPatch,
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value", "a_rows")])
    brokers = _broker_session(node)
    captured_time = time.time

    def helper() -> float:
        return captured_time()

    monkeypatch.setitem(globals(), "_executor_global_helper", helper)

    def kernel(view, context):
        _executor_global_helper()
        return _none_patch(view, context)

    with pytest.raises(AmbientAccessError, match="captures prohibited ambient"):
        _run(
            node,
            projection,
            kernel,
            context=ExecutionContext(brokers=brokers),
        )
    assert brokers.receipt.status == "aborted"


def test_row_classifier_runs_under_the_same_ambient_guard(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(
        StructuralDelta.EXPAND,
        _filter_or_expand_scopes(),
        changed_mutations=(
            "entity_keys",
            "cardinality",
            "links",
            "memberships",
            "order",
            "weights",
        ),
    )
    good = _row_classifier()
    calls = 0

    def classify(entity, table, key, added_ids, registry):
        nonlocal calls
        calls += 1
        time.monotonic()
        return good.function(entity, table, key, added_ids, registry)

    bad = RegisteredRowClassifier(
        classify,
        good.implementation_sha256,
        good.predicate_space,
    )
    brokers = _broker_session(node)
    with pytest.raises(AmbientAccessError, match="ambient clock"):
        _run(
            node,
            projection,
            _expand_patch,
            row_classifiers={node.row_classifier_ref: bad},
            context=ExecutionContext(brokers=brokers),
        )
    assert calls == 1
    assert brokers.receipt.status == "aborted"


def test_row_classifier_cannot_capture_kernel_broker_authority(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(
        StructuralDelta.EXPAND,
        _filter_or_expand_scopes(),
        changed_mutations=(
            "entity_keys",
            "cardinality",
            "links",
            "memberships",
            "order",
            "weights",
        ),
    )
    good = _row_classifier()
    brokers = _broker_session(node)

    def classify(entity, table, key, added_ids, registry):
        try:
            brokers.environment.get("FORBIDDEN")
        except BrokerAccessError:
            pass
        return good.function(entity, table, key, added_ids, registry)

    classifier = RegisteredRowClassifier(
        classify,
        good.implementation_sha256,
        good.predicate_space,
    )
    with pytest.raises(AmbientAccessError, match="captures prohibited ambient"):
        _run(
            node,
            projection,
            _expand_patch,
            row_classifiers={node.row_classifier_ref: classifier},
            context=ExecutionContext(brokers=brokers),
        )
    assert brokers.receipt.status == "aborted"
    assert brokers.receipt.events[-1].reason_code == "prebound_ambient_access"


def test_kernel_receives_only_a_tainting_public_broker_projection(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value", "a_rows")])
    brokers = _broker_session(node)

    def kernel(view, context):
        try:
            _ = context.brokers._log
        except BrokerAccessError:
            pass
        try:
            _ = context.brokers.rng._issued
        except BrokerAccessError:
            pass
        return _none_patch(view, context)

    with pytest.raises(BrokerAccessError, match="recorded a refused access"):
        _run(
            node,
            projection,
            kernel,
            context=ExecutionContext(brokers=brokers),
        )
    assert brokers.receipt.status == "aborted"
    assert {event.reason_code for event in brokers.receipt.events} == {
        "broker_authority_internals_prohibited"
    }


def test_executor_refuses_module_reexport_of_original_ambient_primitive(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value", "a_rows")])
    brokers = _broker_session(node)

    def kernel(view, context):
        broker_module._ORIGINAL_BUILTINS_OPEN("forbidden", "rb")  # noqa: SLF001
        return _none_patch(view, context)

    with pytest.raises(AmbientAccessError, match="captures prohibited ambient"):
        _run(
            node,
            projection,
            kernel,
            context=ExecutionContext(brokers=brokers),
        )
    assert brokers.receipt.status == "aborted"
    assert brokers.receipt.events[-1].reason_code == "prebound_ambient_access"


def test_broker_receipt_is_outside_patch_and_node_identity_seals(
    projection: ImmutableFrameProjection,
) -> None:
    node = _node(
        StructuralDelta.NONE,
        [_scope("value", "a_rows")],
        determinism="seeded",
        seed_streams=("fixture",),
    )
    results: list[ValidatedPatch] = []
    for draw_count in (1, 2):
        brokers = _broker_session(node, run_inputs={"fixture_seed": 17})

        def kernel(view, context, count=draw_count):
            token = context.brokers.rng.token("fixture_site_0")
            with context.brokers.rng.generator(token) as generator:
                for _ in range(count):
                    generator.random(1)
            return _none_patch(view, context)

        results.append(
            _run(
                node,
                projection,
                kernel,
                context=ExecutionContext(brokers=brokers),
            )
        )
    first, second = results
    assert first.broker_receipt.receipt_sha256 != second.broker_receipt.receipt_sha256
    assert first.node_key == second.node_key == node.node_key
    assert first._patch_sha256 == second._patch_sha256
    assert first._result_sha256 == second._result_sha256
    assert first._envelope_sha256 != second._envelope_sha256
    with pytest.raises(ExecutorError, match="envelope seal"):
        apply_patch(
            projection,
            replace(first, broker_receipt=second.broker_receipt),
        )
    behavior_session = _behavior_session(node)
    reuse = node_reuse_identity(
        node,
        behavior_relevant_run_inputs={"sample_fraction": 0.01},
        transitive_input_content_hashes={"fixture_input": "d" * 64},
        implementation_dependency_sha256="e" * 64,
        rng_behavior_inputs=behavior_session.rng.behavior_identity,
        source_behavior_inputs=behavior_session.source_behavior_identity,
        artifact_materializer_abis={"fixture_output": "materializer-v1"},
    )
    assert reuse.key not in {
        first.broker_receipt.receipt_sha256,
        second.broker_receipt.receipt_sha256,
    }


def test_node_reuse_changes_with_seed_but_not_run_provenance_generation() -> None:
    node = _node(
        StructuralDelta.NONE,
        [_scope("value", "a_rows")],
        determinism="seeded",
        seed_streams=("fixture",),
    )

    def reuse(seed: int):
        behavior_session = _behavior_session(node, fixture_seed=seed)
        return node_reuse_identity(
            node,
            behavior_relevant_run_inputs={"rung": "f004"},
            transitive_input_content_hashes={"fixture_input": "d" * 64},
            implementation_dependency_sha256="e" * 64,
            rng_behavior_inputs=behavior_session.rng.behavior_identity,
            source_behavior_inputs=behavior_session.source_behavior_identity,
            artifact_materializer_abis={"fixture_output": "materializer-v1"},
            output_sensitive_backend_abi={"numeric_backend": "cpu-bitwise-v1"},
        )

    common_provenance = {
        "authority_versions": {"stacked_authority": 10},
        "code_inventory_digest": "a" * 64,
        "artifact_protocol_inventory": {"parquet": "fixture-v1"},
        "execution_receipt": {"resolved_backend": "cpu"},
    }
    generation_zero_provenance = build_run_provenance_identity(
        identity_generation=0,
        source_grammar_receipt=None,
        spec_binding=None,
        run_request={"config_authority": "constants", "rung": "f004"},
        **common_provenance,
    )
    generation_one_provenance = build_run_provenance_identity(
        identity_generation=1,
        source_grammar_receipt={
            "schema_version": 3,
            "canonicalizer_version": 1,
            "migration_chain": [{"id": "fixture-v2-v3", "sha256": "b" * 64}],
        },
        spec_binding={
            "country": "us",
            "schema_id": "country-spec",
            "schema_version": 3,
            "canonicalizer_version": 1,
            "spec_sha256": "f" * 64,
            "attestation": "mirror-attested",
        },
        run_request={"config_authority": "bundle", "rung": "f004"},
        **common_provenance,
    )
    first = reuse(17)
    second = reuse(17)
    changed_seed = reuse(18)
    assert isinstance(generation_zero_provenance, RunProvenanceIdentity)
    assert generation_zero_provenance.promotable is False
    assert generation_one_provenance.promotable is True
    assert generation_zero_provenance.to_wire() != generation_one_provenance.to_wire()
    assert set(generation_one_provenance.to_wire()) == {
        "identity_generation",
        "source_grammar_receipt",
        "spec_binding",
        "authority_versions",
        "code_inventory_digest",
        "artifact_protocol_inventory",
        "run_request",
        "execution_receipt",
    }
    assert set(generation_one_provenance.to_wire()["spec_binding"]) == {
        "country",
        "schema_id",
        "schema_version",
        "canonicalizer_version",
        "spec_sha256",
        "attestation",
    }
    assert first.key == second.key
    assert changed_seed.key != first.key
    payload = thaw_json(first.payload)
    assert "identity_generation" not in payload
    rendered = str(payload)
    for forbidden in (
        "config_authority",
        "run_provenance_identity",
        "spec_binding",
        "spec_sha256",
        "broker_receipt",
        "access_log",
    ):
        assert forbidden not in rendered


def test_node_reuse_binds_declared_source_content_but_not_its_path(
    tmp_path: Path,
) -> None:
    node = _node(
        StructuralDelta.NONE,
        [_scope("value", "a_rows")],
        effects=["declared_source_read"],
    )

    def session_for(path: Path, payload: bytes) -> BrokerSession:
        path.write_bytes(payload)
        return _broker_session(
            node,
            sources=(
                DeclaredSource(
                    id="fixture_source",
                    path=path,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    byte_size=len(payload),
                ),
            ),
        )

    def reuse(session: BrokerSession, hashes: Mapping[str, str] | None = None):
        return node_reuse_identity(
            node,
            behavior_relevant_run_inputs={},
            transitive_input_content_hashes={} if hashes is None else hashes,
            implementation_dependency_sha256="e" * 64,
            rng_behavior_inputs=session.rng.behavior_identity,
            source_behavior_inputs=session.source_behavior_identity,
            artifact_materializer_abis={},
        )

    first = session_for(tmp_path / "first.bin", b"same bytes")
    relocated = session_for(tmp_path / "relocated.bin", b"same bytes")
    changed = session_for(tmp_path / "changed.bin", b"changed bytes")
    assert reuse(first).key == reuse(relocated).key
    assert reuse(first).key != reuse(changed).key
    payload = thaw_json(reuse(first).payload)
    assert (
        payload["transitive_input_content_hashes"]["declared_source:fixture_source"]
        == hashlib.sha256(b"same bytes").hexdigest()
    )
    assert str(tmp_path) not in str(payload)
    with pytest.raises(CapabilityError, match="conflicts with declared source"):
        reuse(first, {"declared_source:fixture_source": "f" * 64})
    with pytest.raises(CapabilityError, match="different broker sessions"):
        node_reuse_identity(
            node,
            behavior_relevant_run_inputs={},
            transitive_input_content_hashes={},
            implementation_dependency_sha256="e" * 64,
            rng_behavior_inputs=first.rng.behavior_identity,
            source_behavior_inputs=relocated.source_behavior_identity,
            artifact_materializer_abis={},
        )


def test_node_reuse_requires_an_exact_broker_issued_rng_identity() -> None:
    node = _node(
        StructuralDelta.NONE,
        [_scope("value", "a_rows")],
        determinism="seeded",
        seed_streams=("fixture",),
    )
    common = {
        "behavior_relevant_run_inputs": {"rung": "f004"},
        "transitive_input_content_hashes": {"fixture_input": "d" * 64},
        "implementation_dependency_sha256": "e" * 64,
        "artifact_materializer_abis": {"fixture_output": "materializer-v1"},
    }
    behavior_session = _behavior_session(node)
    common["source_behavior_inputs"] = behavior_session.source_behavior_identity
    with pytest.raises(TypeError, match="broker-issued"):
        node_reuse_identity(
            node,
            rng_behavior_inputs={"fixture_seed": 17},  # type: ignore[arg-type]
            **common,
        )

    wrong_owner = behavior_session.rng.behavior_identity
    object.__setattr__(
        wrong_owner, "owner", BrokerOwner("source_stage", "fixture_node")
    )
    with pytest.raises(CapabilityError, match="owner differs"):
        node_reuse_identity(node, rng_behavior_inputs=wrong_owner, **common)

    forged_session = _behavior_session(node)
    forged = forged_session.rng.behavior_identity
    rows = [thaw_json(site) for site in forged.sites]
    rows[0]["opaque_receipt_digest"] = "f" * 64
    object.__setattr__(
        forged,
        "sites",
        tuple(freeze_json(row) for row in rows),
    )
    with pytest.raises(CapabilityError, match="not broker-issued"):
        node_reuse_identity(
            node,
            rng_behavior_inputs=forged,
            source_behavior_inputs=forged_session.source_behavior_identity,
            **{
                key: value
                for key, value in common.items()
                if key != "source_behavior_inputs"
            },
        )


@pytest.mark.parametrize("generation", [-1, 2, True, "1"])
def test_run_provenance_identity_refuses_unknown_generations(
    generation: object,
) -> None:
    with pytest.raises(ExecutorError, match="identity_generation"):
        build_run_provenance_identity(
            identity_generation=generation,  # type: ignore[arg-type]
            source_grammar_receipt=None,
            spec_binding=None,
            authority_versions={},
            code_inventory_digest="a" * 64,
            artifact_protocol_inventory={},
            run_request={},
            execution_receipt={},
        )


def test_generation_one_provenance_requires_the_binding_receipts() -> None:
    with pytest.raises(ExecutorError, match="generation 1 provenance requires"):
        build_run_provenance_identity(
            identity_generation=1,
            source_grammar_receipt=None,
            spec_binding=None,
            authority_versions={},
            code_inventory_digest="a" * 64,
            artifact_protocol_inventory={},
            run_request={},
            execution_receipt={},
        )


def test_run_provenance_accepts_the_resolved_us_spec_binding() -> None:
    resolved = load_bundle("us")
    provenance = build_run_provenance_identity(
        identity_generation=1,
        source_grammar_receipt=resolved.grammar_receipt.to_wire(),
        spec_binding=resolved.spec_binding.to_wire(),
        authority_versions={"stacked_authority": 10},
        code_inventory_digest="a" * 64,
        artifact_protocol_inventory={"parquet": "fixture-v1"},
        run_request={"config_authority": "bundle", "rung": "f004"},
        execution_receipt={"resolved_backend": "cpu"},
    )
    assert provenance.to_wire()["spec_binding"] == resolved.spec_binding.to_wire()

    bundle_binding = resolved.spec_binding.to_wire()
    bundle_binding["attestation"] = "bundle-authoritative"
    bundle_provenance = build_run_provenance_identity(
        identity_generation=1,
        source_grammar_receipt=resolved.grammar_receipt.to_wire(),
        spec_binding=bundle_binding,
        authority_versions={"stacked_authority": 10},
        code_inventory_digest="a" * 64,
        artifact_protocol_inventory={"parquet": "fixture-v1"},
        run_request={"config_authority": "bundle", "rung": "f004"},
        execution_receipt={"resolved_backend": "cpu"},
    )
    assert (
        bundle_provenance.to_wire()["spec_binding"]["attestation"]
        == "bundle-authoritative"
    )

    invalid_binding = resolved.spec_binding.to_wire()
    invalid_binding["attestation"] = "self-asserted"
    with pytest.raises(ExecutorError, match="attestation"):
        build_run_provenance_identity(
            identity_generation=1,
            source_grammar_receipt=resolved.grammar_receipt.to_wire(),
            spec_binding=invalid_binding,
            authority_versions={"stacked_authority": 10},
            code_inventory_digest="a" * 64,
            artifact_protocol_inventory={"parquet": "fixture-v1"},
            run_request={"config_authority": "bundle", "rung": "f004"},
            execution_receipt={"resolved_backend": "cpu"},
        )


@pytest.mark.parametrize(
    "forbidden",
    [
        "identity_generation",
        "config_authority",
        "run_provenance_identity",
        "spec_binding",
        "spec_sha256",
        "broker_receipt",
        "broker_access_log",
    ],
)
def test_node_reuse_identity_refuses_provenance_and_operational_receipt_fields(
    forbidden: str,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value", "a_rows")])
    behavior_session = _behavior_session(node)
    with pytest.raises(CapabilityError, match="provenance fields"):
        node_reuse_identity(
            node,
            behavior_relevant_run_inputs={"nested": {forbidden: "forbidden"}},
            transitive_input_content_hashes={"fixture_input": "d" * 64},
            implementation_dependency_sha256="e" * 64,
            rng_behavior_inputs=behavior_session.rng.behavior_identity,
            source_behavior_inputs=behavior_session.source_behavior_identity,
            artifact_materializer_abis={"fixture_output": "materializer-v1"},
        )


@pytest.mark.parametrize(
    "aliased_marker",
    [
        "microcosm.spec-engine.broker-access-receipt.v1",
        "operational",
    ],
)
def test_node_reuse_identity_refuses_receipt_markers_under_aliases(
    aliased_marker: str,
) -> None:
    node = _node(StructuralDelta.NONE, [_scope("value", "a_rows")])
    behavior_session = _behavior_session(node)
    with pytest.raises(CapabilityError, match="operational receipt markers"):
        node_reuse_identity(
            node,
            behavior_relevant_run_inputs={"opaque_alias": aliased_marker},
            transitive_input_content_hashes={"fixture_input": "d" * 64},
            implementation_dependency_sha256="e" * 64,
            rng_behavior_inputs=behavior_session.rng.behavior_identity,
            source_behavior_inputs=behavior_session.source_behavior_identity,
            artifact_materializer_abis={"fixture_output": "materializer-v1"},
        )


def test_projection_requires_one_complete_or_of_and_input_alternative(
    projection: ImmutableFrameProjection,
) -> None:
    parts = projection._parts()
    parts["tables"]["group"] = parts["tables"]["group"].drop(columns="group_value")
    incomplete = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]
    effective_input = _input(
        "person",
        "@effective:fixture",
        alternatives=[
            [
                {
                    "entity": "person",
                    "column": "value",
                    "value_kind": "finite_numeric",
                },
                {
                    "entity": "group",
                    "column": "group_value",
                    "value_kind": "finite_numeric",
                },
            ],
            [
                {
                    "entity": "person",
                    "column": "fallback",
                    "value_kind": "column_present",
                }
            ],
        ],
    )
    calls = 0

    def kernel(view, context):
        nonlocal calls
        calls += 1
        return _none_patch(view, context)

    with pytest.raises(CapabilityError, match="alternative|effective|input"):
        _run(
            _node(
                StructuralDelta.NONE,
                [_scope("value", "a_rows")],
                inputs=(effective_input,),
            ),
            incomplete,
            kernel,
        )
    assert calls == 0


def test_absence_receipt_tolerates_missing_but_not_present_invalid_input(
    projection: ImmutableFrameProjection,
) -> None:
    receipt_id = "optional_input:fixture:value"
    effective_input = _input(
        "person",
        "@effective:value",
        alternatives=[
            [
                {
                    "entity": "person",
                    "column": "value",
                    "value_kind": "finite_numeric",
                }
            ]
        ],
        tolerated_absence_receipts=[receipt_id],
    )
    node = _node(
        StructuralDelta.NONE,
        [_scope("group_value", entity="group")],
        inputs=(effective_input, _input("group", "group_value")),
    )
    calls = 0

    def kernel(view, _context):
        nonlocal calls
        calls += 1
        group = view.table("group")
        group.loc[group.group_key.eq(10), "group_value"] = 101.0
        return KernelPatch(StructuralDelta.NONE, tables={"group": group})

    invalid_parts = projection._parts()
    invalid_parts["tables"]["person"]["value"] = [10.0, 20.0, np.inf]
    invalid_parts["virtual_receipts"] = {("frame", receipt_id): {"absent": True}}
    invalid = ImmutableFrameProjection(**invalid_parts)  # type: ignore[arg-type]
    with pytest.raises(CapabilityError, match="input|finite"):
        _run(node, invalid, kernel)
    assert calls == 0

    missing_parts = projection._parts()
    missing_parts["tables"]["person"]["value"] = [10.0, 20.0, np.nan]
    missing_parts["virtual_receipts"] = {
        ("frame", receipt_id): {"absent": True}
    }
    missing = ImmutableFrameProjection(**missing_parts)  # type: ignore[arg-type]
    validated = _run(node, missing, kernel)
    assert not validated.diff.empty
    assert calls == 1

    absent_parts = projection._parts()
    absent_parts["tables"]["person"] = absent_parts["tables"]["person"].drop(
        columns="value"
    )
    absent_parts["virtual_receipts"] = {("frame", receipt_id): {"absent": True}}
    absent = ImmutableFrameProjection(**absent_parts)  # type: ignore[arg-type]
    validated = _run(node, absent, kernel)
    assert not validated.diff.empty
    assert calls == 2


@pytest.mark.parametrize(
    "invalid_values",
    (
        pd.Series([10.0, 20.0, np.inf], dtype="float64"),
        pd.Series([True, False, True], dtype="bool"),
        pd.Series(["10", "20", "invalid"], dtype="string"),
    ),
    ids=("infinite", "boolean", "non_numeric"),
)
def test_absence_receipt_refuses_each_present_non_null_invalid_numeric_input(
    projection: ImmutableFrameProjection,
    invalid_values: pd.Series,
) -> None:
    receipt_id = "optional_input:fixture:value"
    effective_input = _input(
        "person",
        "@effective:value",
        alternatives=[
            [
                {
                    "entity": "person",
                    "column": "value",
                    "value_kind": "finite_numeric",
                }
            ]
        ],
        tolerated_absence_receipts=[receipt_id],
    )
    node = _node(
        StructuralDelta.NONE,
        [_scope("group_value", entity="group")],
        inputs=(effective_input, _input("group", "group_value")),
    )
    parts = projection._parts()
    parts["tables"]["person"]["value"] = invalid_values
    parts["virtual_receipts"] = {("frame", receipt_id): {"absent": True}}
    invalid = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]

    with pytest.raises(CapabilityError, match="input|finite"):
        _run(node, invalid, _none_patch)


@pytest.mark.parametrize(
    ("required_scope", "accepted"),
    [("a_rows", True), ("b_rows", False)],
)
def test_input_value_kind_is_enforced_only_on_its_required_scope(
    projection: ImmutableFrameProjection,
    required_scope: str,
    accepted: bool,
) -> None:
    parts = projection._parts()
    parts["tables"]["person"]["value"] = [10.0, 20.0, np.inf]
    scoped_projection = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]
    effective_input = _input(
        "person",
        "@effective:finite_value",
        required_scope=required_scope,
        alternatives=[
            [
                {
                    "entity": "person",
                    "column": "value",
                    "value_kind": "finite_numeric",
                }
            ]
        ],
    )
    node = _node(
        StructuralDelta.NONE,
        [_scope("value", "a_rows")],
        inputs=(effective_input, _input("group", "group_value")),
    )
    if accepted:
        validated = _run(node, scoped_projection, _none_patch)
        assert not validated.diff.empty
    else:
        with pytest.raises(CapabilityError, match="finite|value_kind|input"):
            _run(node, scoped_projection, _none_patch)


def test_validated_patch_envelope_seals_its_node_binding(
    projection: ImmutableFrameProjection,
) -> None:
    validated = _validated_none_patch(projection)
    forged = replace(validated, node_id="forged_node")
    with pytest.raises(ExecutorError, match="node|envelope|seal|receipt|owner"):
        apply_patch(projection, forged)


def test_expand_refuses_weight_kind_change_during_mass_split(
    projection: ImmutableFrameProjection,
) -> None:
    def kernel(view, context):
        patch = _expand_patch(view, context)
        return replace(
            patch,
            weights={
                "person": WeightState(
                    [1.0, 2.0, 1.5, 1.5],
                    "importance",
                )
            },
        )

    scopes = [
        (
            _scope("@resolved_weight", "whole", mode="resolved_weight")
            if scope["column"] == "@resolved_weight"
            else scope
        )
        for scope in _filter_or_expand_scopes()
    ]
    with pytest.raises(StructuralDiffError, match="weight.*kind|kind.*weight"):
        _run(
            _node(
                StructuralDelta.EXPAND,
                scopes,
                changed_mutations=(
                    "entity_keys",
                    "cardinality",
                    "links",
                    "memberships",
                    "order",
                    "weights",
                ),
            ),
            projection,
            kernel,
            row_classifiers=_structural_context(),
        )


def test_order_refuses_dependency_omitted_from_transitive_node_slice() -> None:
    ancestor = _node(
        StructuralDelta.NONE,
        [_scope("ancestor_value")],
        node_id="ancestor",
        execution_rank=0,
    )
    descendant = _node(
        StructuralDelta.NONE,
        [_scope("descendant_value")],
        node_id="descendant",
        execution_rank=1,
        depends_on=("ancestor",),
        transitive_nodes=(),
    )
    with pytest.raises(
        (NodeOrderingError, CapabilityError),
        match="transitive|ancestor|closure",
    ):
        order_nodes((descendant, ancestor))
