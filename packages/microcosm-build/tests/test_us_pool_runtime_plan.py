"""Compiler-capability gates for the US physical pool runtime plan."""

from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError, replace

import pytest

from microcosm.build.spec_engine import (
    compile_runtime_authorities,
    compile_spec,
    load_bundle,
)
from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.model import FrozenMap
from microcosm.build.us_runtime.pool_runtime_plan import (
    OperationalReceiptsSidecar,
    USPoolRuntimePlan,
    USPoolRuntimePlanError,
)
from microcosm.build.us_runtime.spec_authority import (
    USSpecAuthority,
    _capability_sha256,
    compile_us_spec_authority,
)


@pytest.fixture(scope="module")
def authority() -> USSpecAuthority:
    return compile_us_spec_authority(
        compile_runtime_authorities(compile_spec(load_bundle("us")))
    )


@pytest.fixture(scope="module")
def plan(authority: USSpecAuthority) -> USPoolRuntimePlan:
    return USPoolRuntimePlan.from_spec_authority(authority)


def _replace_map(value: FrozenMap, **changes: object) -> FrozenMap:
    entries = dict(value.entries)
    entries.update(changes)
    return FrozenMap(tuple(sorted(entries.items())))


def _resign_execution(value: FrozenMap, **changes: object) -> FrozenMap:
    changed = _replace_map(value, **changes)
    unsigned = FrozenMap(
        tuple((key, item) for key, item in changed.entries if key != "sha256")
    )
    return _replace_map(changed, sha256=sha256_json(unsigned))


def _reseal_authority(
    authority: USSpecAuthority,
    **changes: object,
) -> USSpecAuthority:
    surfaces = {
        "_behavior": authority._behavior,
        "_projections": authority._projections,
        "_declared_sources": authority._declared_sources,
        "_generated_authorities": authority._generated_authorities,
        "_vintage_authorities": authority._vintage_authorities,
        "_execution_abi": authority._execution_abi,
        "_seed_stream_map": authority._seed_stream_map,
        "_nodes": authority._nodes,
    }
    surfaces.update(changes)
    seal = _capability_sha256(
        authority_sha256=authority.authority_sha256,
        spec_sha256=authority.spec_sha256,
        identity_generation=authority.identity_generation,
        behavior=surfaces["_behavior"],
        projections=surfaces["_projections"],
        declared_sources=surfaces["_declared_sources"],
        generated_authorities=surfaces["_generated_authorities"],
        vintage_authorities=surfaces["_vintage_authorities"],
        execution_abi=surfaces["_execution_abi"],
        seed_stream_map=surfaces["_seed_stream_map"],
        nodes=surfaces["_nodes"],
    )
    return replace(authority, **changes, _seal_sha256=seal)


def test_plan_narrows_every_pool_authority_without_copying_compiler_maps(
    authority: USSpecAuthority,
    plan: USPoolRuntimePlan,
) -> None:
    spine = authority.behavior_resource("spine")
    assert plan.authority_sha256 == authority.authority_sha256
    assert plan.spec_sha256 == authority.spec_sha256
    assert plan.identity_generation == 1
    assert plan.seed_stream_map is authority.seed_stream_map

    assert plan.sources.declared is authority.declared_sources
    assert plan.sources.contract is authority.behavior_resource("sources")
    assert plan.sources.vintage_contract is authority.behavior_resource("vintages")
    assert plan.sources.vintages is authority.vintage_authorities
    assert plan.support_spine.source_pool is spine["support_source_pool"]
    assert (
        plan.support_spine.source_pool_metadata
        is (spine["support_source_pool_metadata"])
    )
    assert plan.assembly_sampling.assembly is spine["assembly"]
    assert plan.assembly_sampling.sampling_contract is spine["sampling"]
    assert plan.assembly_sampling.runtime_sampling is authority.sampling
    assert plan.publication.contract is authority.behavior_resource("publication")
    assert plan.publication.runtime is authority.publication
    assert plan.execution.pipeline is authority.execution_abi["pipeline"]
    assert plan.execution.code_abi is authority.execution_abi["code_abi"]
    assert plan.execution.stacked_authority is authority.stacked_authority
    assert plan.execution.checkpoint_static_components is (
        authority.stacked_checkpoint_static_components
    )
    assert plan.imputation.contract is authority.behavior_resource("imputation")
    assert plan.imputation.runtime is authority.imputation
    assert plan.imputation.nodes is authority.nodes
    assert plan.take_up.contract is authority.behavior_resource("take_up")
    assert plan.take_up.runtime is authority.take_up
    assert (
        plan.remaining_stage.manifest
        is (
            authority.generated_authorities["engine_abi_lock"][
                "remaining_stage_input_manifest"
            ]
        )
    )
    assert plan.battery.contract is authority.behavior_resource("battery")
    assert plan.battery.runtime is authority.battery
    assert plan.battery.components is authority.battery_components


def test_execution_stages_and_checkpoints_are_typed_exact_compiler_order(
    authority: USSpecAuthority,
    plan: USPoolRuntimePlan,
) -> None:
    execution = authority.execution_abi
    assert tuple(item.id for item in plan.execution.operations) == tuple(
        execution["pipeline"]["operator_order"]
    )
    assert tuple(item.contract for item in plan.execution.operations) == tuple(
        execution["operations"]
    )
    assert tuple(item.contract for item in plan.execution.stages) == tuple(
        execution["logical_stages"]
    )
    assert tuple(item.contract for item in plan.execution.checkpoints) == tuple(
        execution["durable_checkpoints"]
    )
    assert tuple(
        operation_id
        for stage in plan.execution.stages
        for operation_id in stage.operation_ids
    ) == tuple(item.id for item in plan.execution.operations)
    assert tuple(item.id for item in plan.execution.checkpoints) == tuple(
        item.id for item in plan.execution.stages if item.durable
    )
    assert tuple(
        item.operational_receipts_sidecar for item in plan.execution.stages
    ) == (
        OperationalReceiptsSidecar.FORBIDDEN,
        OperationalReceiptsSidecar.REQUIRED,
        OperationalReceiptsSidecar.REQUIRED,
        OperationalReceiptsSidecar.NOT_APPLICABLE,
    )
    assert tuple(
        item.operational_receipts_sidecar for item in plan.execution.checkpoints
    ) == tuple(
        item.operational_receipts_sidecar
        for item in plan.execution.stages
        if item.durable
    )

    for operation in plan.execution.operations:
        assert plan.execution.require_operation(operation.id) is operation
    for stage in plan.execution.stages:
        assert plan.execution.require_stage(stage.id) is stage
    for checkpoint in plan.execution.checkpoints:
        assert plan.execution.require_checkpoint(checkpoint.id) is checkpoint


def test_generic_lookups_do_not_create_program_named_authority_accessors(
    plan: USPoolRuntimePlan,
) -> None:
    source = plan.sources.declared_rows[0]
    stage = plan.sources.stage_rows[0]
    node = plan.imputation.nodes[0]
    assert plan.sources.require_source(str(source["id"])) is source
    assert plan.sources.require_stage(str(stage["stage"])) is stage
    assert plan.imputation.require_node(node.id) is node

    programs = plan.take_up.runtime["programs"]
    assert isinstance(programs, tuple)
    variable_names = {
        str(row["variable"]) for row in programs if isinstance(row, FrozenMap)
    }
    assert variable_names
    public_names = set(dir(plan)) | set(dir(plan.take_up))
    assert variable_names.isdisjoint(public_names)


def test_plan_is_recursively_immutable(plan: USPoolRuntimePlan) -> None:
    with pytest.raises(FrozenInstanceError):
        plan.identity_generation = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.execution.stages[0].id = "changed"  # type: ignore[misc]


def test_plan_seal_rejects_replaced_domain_authority(
    plan: USPoolRuntimePlan,
) -> None:
    with pytest.raises(USPoolRuntimePlanError, match="plan seal"):
        replace(
            plan,
            sources=replace(plan.sources, declared=FrozenMap()),
        )
    with pytest.raises(TypeError):
        plan.publication.contract["release"] = FrozenMap()  # type: ignore[index]


def test_factory_does_not_read_files_or_invoke_legacy_adapter(
    authority: USSpecAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import microcosm.build.spec_engine.legacy_adapter as legacy_adapter

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("runtime-plan compilation attempted ambient loading")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(legacy_adapter, "compile_to_legacy_payload", forbidden)
    plan = USPoolRuntimePlan.from_spec_authority(authority)
    assert plan.authority_sha256 == authority.authority_sha256


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("absent_execution", "physical pipeline is absent"),
        ("missing_stages", "logical_stages.*at least one row"),
        ("missing_projection", "required compiler projection.*publication"),
        ("missing_remaining_manifest", "remaining_stage_input_manifest"),
        ("resume_missing_reuse", "identity fields differ"),
        ("resume_candidate_order", "candidate order must reverse"),
        ("producer_order", "pipeline producer order differs"),
        ("extra_nested_nodes", "only the producer graph operation"),
        ("receipt_policy_mismatch", "receipt sidecar policy differs"),
    ],
)
def test_plan_fails_closed_on_incomplete_compiler_authority(
    authority: USSpecAuthority,
    mutation: str,
    match: str,
) -> None:
    if mutation == "absent_execution":
        candidate = _reseal_authority(
            authority,
            _execution_abi=_replace_map(authority.execution_abi, present=False),
        )
    elif mutation == "missing_stages":
        candidate = _reseal_authority(
            authority,
            _execution_abi=_resign_execution(
                authority.execution_abi,
                logical_stages=(),
            ),
        )
    elif mutation == "missing_projection":
        projections = FrozenMap(
            tuple(
                (key, value)
                for key, value in authority._projections.entries
                if key != "publication"
            )
        )
        candidate = _reseal_authority(authority, _projections=projections)
    elif mutation == "missing_remaining_manifest":
        generated = _replace_map(
            authority.generated_authorities,
            engine_abi_lock=_replace_map(
                authority.generated_authorities["engine_abi_lock"],
                remaining_stage_input_manifest=None,
            ),
        )
        candidate = _reseal_authority(
            authority,
            _generated_authorities=generated,
        )
    elif mutation == "resume_missing_reuse":
        predicate = _replace_map(
            authority.execution_abi["resume_predicate"],
            identity_fields=(
                "artifact_protocol",
                "input_pins",
                "sampling_request",
                "clone_attachment_request",
            ),
        )
        candidate = _reseal_authority(
            authority,
            _execution_abi=_resign_execution(
                authority.execution_abi,
                resume_predicate=predicate,
            ),
        )
    elif mutation == "resume_candidate_order":
        predicate = authority.execution_abi["resume_predicate"]
        candidate = _reseal_authority(
            authority,
            _execution_abi=_resign_execution(
                authority.execution_abi,
                resume_predicate=_replace_map(
                    predicate,
                    candidate_order=tuple(reversed(predicate["candidate_order"])),
                ),
            ),
        )
    elif mutation == "producer_order":
        pipeline = authority.execution_abi["pipeline"]
        candidate = _reseal_authority(
            authority,
            _execution_abi=_resign_execution(
                authority.execution_abi,
                pipeline=_replace_map(
                    pipeline,
                    producer_order=tuple(reversed(pipeline["producer_order"])),
                ),
            ),
        )
    elif mutation == "extra_nested_nodes":
        operations = authority.execution_abi["operations"]
        producer_operation = next(
            row for row in operations if row["nested_producer_nodes"]
        )
        nonproducer_index = next(
            index
            for index, row in enumerate(operations)
            if row is not producer_operation
        )
        changed_operations = list(operations)
        changed_operations[nonproducer_index] = _replace_map(
            operations[nonproducer_index],
            nested_producer_nodes=(authority.nodes[0].id,),
        )
        candidate = _reseal_authority(
            authority,
            _execution_abi=_resign_execution(
                authority.execution_abi,
                operations=tuple(changed_operations),
            ),
        )
    else:
        checkpoints = authority.execution_abi["durable_checkpoints"]
        changed_checkpoints = list(checkpoints)
        changed_checkpoints[0] = _replace_map(
            checkpoints[0],
            operational_receipts_sidecar="required",
        )
        candidate = _reseal_authority(
            authority,
            _execution_abi=_resign_execution(
                authority.execution_abi,
                durable_checkpoints=tuple(changed_checkpoints),
            ),
        )

    with pytest.raises(USPoolRuntimePlanError, match=match):
        USPoolRuntimePlan.from_spec_authority(candidate)


def test_factory_rejects_non_authority() -> None:
    with pytest.raises(TypeError, match="USSpecAuthority"):
        USPoolRuntimePlan.from_spec_authority(FrozenMap())  # type: ignore[arg-type]
