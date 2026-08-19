"""Compiler-sealed authorities consumed by the US pool orchestrator.

The pool build has many authority domains, but it must have only one source of
truth in bundle mode.  :class:`USPoolRuntimePlan` narrows a
:class:`~microcosm.build.us_runtime.spec_authority.USSpecAuthority` into the
domain capabilities used by the physical build.  It does not load bundle
files, constants, a legacy payload, or packaged JSON.

All JSON-shaped values remain :class:`FrozenMap` objects owned by the compiler;
the adapter neither thaws nor reconstructs them.  Pipeline stages and durable
checkpoints receive small typed wrappers so orchestration can dispatch without
stringly-shaped dictionary traversal.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from typing import Protocol

from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.compiler_ir import (
    EXECUTION_RESUME_IDENTITY_FIELDS,
    EXECUTION_RESUME_INTEGRITY_VALIDATORS,
    CompiledNode,
    SeedStreamMap,
)
from microcosm.build.spec_engine.model import (
    FrozenMap,
    FrozenValue,
    ResourceKind,
    thaw_json,
)
from microcosm.build.us_runtime.spec_authority import (
    USAuthorityProjection,
    USSpecAuthority,
    USSpecAuthorityError,
)


class USPoolRuntimePlanError(ValueError):
    """The compiler capability cannot produce a closed US pool plan."""


class OperationalReceiptsSidecar(StrEnum):
    """Compiler-authored presence contract for one stage receipt sidecar."""

    FORBIDDEN = "forbidden"
    REQUIRED = "required"
    NOT_APPLICABLE = "not_applicable"


_RESUME_PREDICATE_FIELDS = frozenset(
    {
        "candidate_order",
        "required_artifact_roles_by_stage",
        "identity_fields",
        "integrity_validators",
        "last_durable_stage",
    }
)


class _Identified(Protocol):
    id: str


def _map(value: FrozenValue | object, *, location: str) -> FrozenMap:
    if not isinstance(value, FrozenMap):
        raise USPoolRuntimePlanError(f"{location}: compiler-sealed object required")
    return value


def _map_tuple(
    value: FrozenValue | object,
    *,
    location: str,
    nonempty: bool = False,
) -> tuple[FrozenMap, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(item, FrozenMap) for item in value
    ):
        raise USPoolRuntimePlanError(
            f"{location}: compiler-sealed object array required"
        )
    if nonempty and not value:
        raise USPoolRuntimePlanError(f"{location}: at least one row required")
    return value


def _string_tuple(
    value: FrozenValue | object,
    *,
    location: str,
    nonempty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise USPoolRuntimePlanError(f"{location}: non-empty string array required")
    if nonempty and not value:
        raise USPoolRuntimePlanError(f"{location}: at least one value required")
    return value


def _string(value: FrozenValue | object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise USPoolRuntimePlanError(f"{location}: non-empty string required")
    return value


def _ordinal(value: FrozenValue | object, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise USPoolRuntimePlanError(f"{location}: non-negative integer required")
    return value


def _optional_string(value: FrozenValue | object, *, location: str) -> str | None:
    if value is None:
        return None
    return _string(value, location=location)


def _receipt_sidecar_policy(
    value: FrozenValue | object,
    *,
    location: str,
) -> OperationalReceiptsSidecar:
    try:
        return OperationalReceiptsSidecar(_string(value, location=location))
    except ValueError as error:
        raise USPoolRuntimePlanError(
            f"{location}: unknown operational receipt sidecar policy"
        ) from error


def _require_unique_ids(
    rows: tuple[_Identified, ...],
    *,
    location: str,
) -> None:
    ids = tuple(row.id for row in rows)
    if len(ids) != len(set(ids)):
        raise USPoolRuntimePlanError(f"{location}: duplicate ids are forbidden")


def _row_by_id(
    rows: tuple[FrozenMap, ...],
    row_id: str,
    *,
    id_field: str,
    location: str,
) -> FrozenMap:
    if not isinstance(row_id, str) or not row_id:
        raise USPoolRuntimePlanError(f"{location}: lookup id must be non-empty")
    matches = tuple(row for row in rows if row.get(id_field) == row_id)
    if len(matches) != 1:
        raise USPoolRuntimePlanError(
            f"{location}: {row_id!r} must match exactly once; matched {len(matches)}"
        )
    return matches[0]


def _projection(
    authority: USSpecAuthority,
    projection: USAuthorityProjection,
) -> FrozenMap:
    try:
        return authority.projection(projection)
    except USSpecAuthorityError as error:
        raise USPoolRuntimePlanError(
            f"required compiler projection {projection.value!r} is absent"
        ) from error


@dataclass(frozen=True, slots=True)
class SourceAuthority:
    """Source registry, stage contracts, and resolved source vintages."""

    declared: FrozenMap
    contract: FrozenMap
    vintage_contract: FrozenMap
    vintages: FrozenMap

    @property
    def declared_rows(self) -> tuple[FrozenMap, ...]:
        return _map_tuple(
            self.declared.get("sources"),
            location="sources/declared/sources",
            nonempty=True,
        )

    @property
    def stage_rows(self) -> tuple[FrozenMap, ...]:
        return _map_tuple(
            self.contract.get("stages"),
            location="sources/contract/stages",
            nonempty=True,
        )

    def require_source(self, source_id: str) -> FrozenMap:
        """Return one source by declared id, refusing absence or ambiguity."""

        return _row_by_id(
            self.declared_rows,
            source_id,
            id_field="id",
            location="sources/declared/sources",
        )

    def require_stage(self, stage_id: str) -> FrozenMap:
        """Return one source-stage contract by id."""

        return _row_by_id(
            self.stage_rows,
            stage_id,
            id_field="stage",
            location="sources/contract/stages",
        )


@dataclass(frozen=True, slots=True)
class SupportSpineAuthority:
    """Narrow authority for support-source composition."""

    source_pool: FrozenMap
    source_pool_metadata: FrozenMap
    roles: tuple[FrozenMap, ...]
    channels: tuple[FrozenMap, ...]


@dataclass(frozen=True, slots=True)
class AssemblySamplingAuthority:
    """Stacked assembly and deterministic sampling authorities."""

    assembly: FrozenMap
    sampling_contract: FrozenMap
    runtime_sampling: FrozenMap
    seed_site_bindings: tuple[FrozenMap, ...]


@dataclass(frozen=True, slots=True)
class PublicationAuthority:
    """Normative publication contract and legacy-compatible runtime view."""

    contract: FrozenMap
    runtime: FrozenMap


@dataclass(frozen=True, slots=True)
class ExecutionOperation:
    """One physical operation in compiler total order."""

    id: str
    ordinal: int
    stage_id: str
    nested_producer_nodes: tuple[str, ...]
    contract: FrozenMap = field(repr=False)


@dataclass(frozen=True, slots=True)
class ExecutionStage:
    """One compiler-declared logical stage."""

    id: str
    ordinal: int
    operation_ids: tuple[str, ...]
    durable: bool
    operational_receipts_sidecar: OperationalReceiptsSidecar
    producer_graph_operation: str | None
    producer_node_ids: tuple[str, ...]
    contract: FrozenMap = field(repr=False)


@dataclass(frozen=True, slots=True)
class DurableCheckpoint:
    """One resumable physical checkpoint boundary."""

    id: str
    ordinal: int
    after_operation: str
    covered_operation_ids: tuple[str, ...]
    operational_receipts_sidecar: OperationalReceiptsSidecar
    artifact_roles: tuple[str, ...]
    contract: FrozenMap = field(repr=False)


@dataclass(frozen=True, slots=True)
class ExecutionAuthority:
    """Typed physical execution, checkpoint, and comparison contract."""

    abi_sha256: str
    pipeline: FrozenMap
    operations: tuple[ExecutionOperation, ...]
    stages: tuple[ExecutionStage, ...]
    checkpoints: tuple[DurableCheckpoint, ...]
    code_abi: FrozenMap
    artifact_vector: tuple[FrozenMap, ...]
    receipt_comparison_vector: tuple[FrozenMap, ...]
    resume_predicate: FrozenMap
    stacked_authority: FrozenMap
    checkpoint_static_components: FrozenMap

    def require_operation(self, operation_id: str) -> ExecutionOperation:
        """Return exactly one operation by compiler id."""

        return self._require_typed(
            self.operations,
            operation_id,
            location="execution/operations",
        )

    def require_stage(self, stage_id: str) -> ExecutionStage:
        """Return exactly one logical stage by compiler id."""

        return self._require_typed(
            self.stages,
            stage_id,
            location="execution/logical_stages",
        )

    def require_checkpoint(self, stage_id: str) -> DurableCheckpoint:
        """Return exactly one durable checkpoint by logical-stage id."""

        return self._require_typed(
            self.checkpoints,
            stage_id,
            location="execution/durable_checkpoints",
        )

    @staticmethod
    def _require_typed[T: _Identified](
        rows: tuple[T, ...],
        row_id: str,
        *,
        location: str,
    ) -> T:
        if not isinstance(row_id, str) or not row_id:
            raise USPoolRuntimePlanError(f"{location}: lookup id must be non-empty")
        matches = tuple(row for row in rows if row.id == row_id)
        if len(matches) != 1:
            raise USPoolRuntimePlanError(
                f"{location}: {row_id!r} must match exactly once; "
                f"matched {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class ImputationProducerAuthority:
    """Imputation contracts plus the compiler-ordered producer graph."""

    contract: FrozenMap
    runtime: FrozenMap
    producer_graph: FrozenMap
    nodes: tuple[CompiledNode, ...]

    def require_node(self, node_id: str) -> CompiledNode:
        """Return one compiled producer node by id."""

        if not isinstance(node_id, str) or not node_id:
            raise USPoolRuntimePlanError(
                "imputation/producer_graph: lookup id must be non-empty"
            )
        matches = tuple(node for node in self.nodes if node.id == node_id)
        if len(matches) != 1:
            raise USPoolRuntimePlanError(
                "imputation/producer_graph: "
                f"{node_id!r} must match exactly once; matched {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class TakeUpAuthority:
    """Typed-step take-up contract with its compatibility projection."""

    contract: FrozenMap
    runtime: FrozenMap


@dataclass(frozen=True, slots=True)
class RemainingStageAuthority:
    """Compiler-generated remaining-stage engine input manifest."""

    engine: FrozenMap
    manifest: FrozenMap
    receipt: FrozenMap
    rows: tuple[FrozenMap, ...]


@dataclass(frozen=True, slots=True)
class BatteryAuthority:
    """Battery gates and compiler-resolved metric/component authority."""

    contract: FrozenMap
    runtime: FrozenMap
    components: FrozenMap


def _compile_operations(execution_abi: FrozenMap) -> tuple[ExecutionOperation, ...]:
    rows = _map_tuple(
        execution_abi.get("operations"),
        location="execution_abi/operations",
        nonempty=True,
    )
    operations = tuple(
        ExecutionOperation(
            id=_string(row.get("id"), location=f"execution_abi/operations/{index}/id"),
            ordinal=_ordinal(
                row.get("ordinal"),
                location=f"execution_abi/operations/{index}/ordinal",
            ),
            stage_id=_string(
                row.get("stage_ref"),
                location=f"execution_abi/operations/{index}/stage_ref",
            ),
            nested_producer_nodes=_string_tuple(
                row.get("nested_producer_nodes"),
                location=(f"execution_abi/operations/{index}/nested_producer_nodes"),
            ),
            contract=row,
        )
        for index, row in enumerate(rows)
    )
    _require_unique_ids(operations, location="execution_abi/operations")
    if tuple(row.ordinal for row in operations) != tuple(range(len(operations))):
        raise USPoolRuntimePlanError(
            "execution_abi/operations: ordinals must be contiguous total order"
        )
    return operations


def _compile_stages(execution_abi: FrozenMap) -> tuple[ExecutionStage, ...]:
    rows = _map_tuple(
        execution_abi.get("logical_stages"),
        location="execution_abi/logical_stages",
        nonempty=True,
    )
    stages: list[ExecutionStage] = []
    for index, row in enumerate(rows):
        durable = row.get("durable_checkpoint")
        if not isinstance(durable, bool):
            raise USPoolRuntimePlanError(
                f"execution_abi/logical_stages/{index}/durable_checkpoint: "
                "boolean required"
            )
        stages.append(
            ExecutionStage(
                id=_string(
                    row.get("id"),
                    location=f"execution_abi/logical_stages/{index}/id",
                ),
                ordinal=_ordinal(
                    row.get("ordinal"),
                    location=f"execution_abi/logical_stages/{index}/ordinal",
                ),
                operation_ids=_string_tuple(
                    row.get("operations"),
                    location=f"execution_abi/logical_stages/{index}/operations",
                    nonempty=True,
                ),
                durable=durable,
                operational_receipts_sidecar=_receipt_sidecar_policy(
                    row.get("operational_receipts_sidecar"),
                    location=(
                        "execution_abi/logical_stages/"
                        f"{index}/operational_receipts_sidecar"
                    ),
                ),
                producer_graph_operation=_optional_string(
                    row.get("producer_graph_operation"),
                    location=(
                        f"execution_abi/logical_stages/{index}/producer_graph_operation"
                    ),
                ),
                producer_node_ids=_string_tuple(
                    row.get("producer_nodes"),
                    location=f"execution_abi/logical_stages/{index}/producer_nodes",
                ),
                contract=row,
            )
        )
    result = tuple(stages)
    _require_unique_ids(result, location="execution_abi/logical_stages")
    if tuple(row.ordinal for row in result) != tuple(range(len(result))):
        raise USPoolRuntimePlanError(
            "execution_abi/logical_stages: ordinals must be contiguous total order"
        )
    return result


def _compile_checkpoints(execution_abi: FrozenMap) -> tuple[DurableCheckpoint, ...]:
    rows = _map_tuple(
        execution_abi.get("durable_checkpoints"),
        location="execution_abi/durable_checkpoints",
        nonempty=True,
    )
    checkpoints = tuple(
        DurableCheckpoint(
            id=_string(
                row.get("id"),
                location=f"execution_abi/durable_checkpoints/{index}/id",
            ),
            ordinal=_ordinal(
                row.get("ordinal"),
                location=f"execution_abi/durable_checkpoints/{index}/ordinal",
            ),
            after_operation=_string(
                row.get("after_operation"),
                location=(f"execution_abi/durable_checkpoints/{index}/after_operation"),
            ),
            covered_operation_ids=_string_tuple(
                row.get("covers_operations"),
                location=(
                    f"execution_abi/durable_checkpoints/{index}/covers_operations"
                ),
                nonempty=True,
            ),
            operational_receipts_sidecar=_receipt_sidecar_policy(
                row.get("operational_receipts_sidecar"),
                location=(
                    "execution_abi/durable_checkpoints/"
                    f"{index}/operational_receipts_sidecar"
                ),
            ),
            artifact_roles=_string_tuple(
                row.get("artifact_roles"),
                location=f"execution_abi/durable_checkpoints/{index}/artifact_roles",
                nonempty=True,
            ),
            contract=row,
        )
        for index, row in enumerate(rows)
    )
    _require_unique_ids(checkpoints, location="execution_abi/durable_checkpoints")
    if tuple(row.ordinal for row in checkpoints) != tuple(range(len(checkpoints))):
        raise USPoolRuntimePlanError(
            "execution_abi/durable_checkpoints: ordinals must be contiguous"
        )
    return checkpoints


def _compile_resume_predicate(
    execution_abi: FrozenMap,
    *,
    checkpoints: tuple[DurableCheckpoint, ...],
) -> FrozenMap:
    predicate = _map(
        execution_abi.get("resume_predicate"),
        location="execution_abi/resume_predicate",
    )
    fields = frozenset(predicate)
    if fields != _RESUME_PREDICATE_FIELDS:
        raise USPoolRuntimePlanError(
            "execution_abi/resume_predicate: fields differ from the closed contract"
        )
    checkpoint_ids = tuple(row.id for row in checkpoints)
    if _string_tuple(
        predicate.get("candidate_order"),
        location="execution_abi/resume_predicate/candidate_order",
        nonempty=True,
    ) != tuple(reversed(checkpoint_ids)):
        raise USPoolRuntimePlanError(
            "execution_abi/resume_predicate: candidate order must reverse checkpoints"
        )
    if predicate.get("last_durable_stage") != checkpoint_ids[-1]:
        raise USPoolRuntimePlanError(
            "execution_abi/resume_predicate: last durable stage differs"
        )
    required_roles = _map(
        predicate.get("required_artifact_roles_by_stage"),
        location=("execution_abi/resume_predicate/required_artifact_roles_by_stage"),
    )
    if frozenset(required_roles) != frozenset(checkpoint_ids):
        raise USPoolRuntimePlanError(
            "execution_abi/resume_predicate: checkpoint artifact-role keys differ"
        )
    for checkpoint in checkpoints:
        if (
            _string_tuple(
                required_roles.get(checkpoint.id),
                location=(
                    "execution_abi/resume_predicate/required_artifact_roles_by_stage/"
                    f"{checkpoint.id}"
                ),
                nonempty=True,
            )
            != checkpoint.artifact_roles
        ):
            raise USPoolRuntimePlanError(
                "execution_abi/resume_predicate: checkpoint artifact roles differ"
            )
    if (
        _string_tuple(
            predicate.get("identity_fields"),
            location="execution_abi/resume_predicate/identity_fields",
            nonempty=True,
        )
        != EXECUTION_RESUME_IDENTITY_FIELDS
    ):
        raise USPoolRuntimePlanError(
            "execution_abi/resume_predicate: identity fields differ"
        )
    if (
        _string_tuple(
            predicate.get("integrity_validators"),
            location="execution_abi/resume_predicate/integrity_validators",
            nonempty=True,
        )
        != EXECUTION_RESUME_INTEGRITY_VALIDATORS
    ):
        raise USPoolRuntimePlanError(
            "execution_abi/resume_predicate: integrity validators differ"
        )
    return predicate


def _compile_execution(
    authority: USSpecAuthority,
    *,
    nodes: tuple[CompiledNode, ...],
) -> ExecutionAuthority:
    execution_abi = authority.execution_abi
    if execution_abi.get("present") is not True:
        raise USPoolRuntimePlanError("execution_abi: physical pipeline is absent")
    abi_sha256 = _string(
        execution_abi.get("sha256"),
        location="execution_abi/sha256",
    )
    unsigned = FrozenMap(
        tuple((key, value) for key, value in execution_abi.entries if key != "sha256")
    )
    if sha256_json(unsigned) != abi_sha256:
        raise USPoolRuntimePlanError(
            "execution_abi: sha256 does not seal the supplied physical plan"
        )

    pipeline = _map(execution_abi.get("pipeline"), location="execution_abi/pipeline")
    operations = _compile_operations(execution_abi)
    stages = _compile_stages(execution_abi)
    checkpoints = _compile_checkpoints(execution_abi)

    pipeline_order = _string_tuple(
        pipeline.get("operator_order"),
        location="execution_abi/pipeline/operator_order",
        nonempty=True,
    )
    operation_order = tuple(row.id for row in operations)
    stage_order = tuple(
        operation_id for stage in stages for operation_id in stage.operation_ids
    )
    if pipeline_order != operation_order or stage_order != operation_order:
        raise USPoolRuntimePlanError(
            "execution_abi: logical stages must exactly partition operator order"
        )
    by_operation = {row.id: row for row in operations}
    for stage in stages:
        if stage.durable and stage.operational_receipts_sidecar not in {
            OperationalReceiptsSidecar.FORBIDDEN,
            OperationalReceiptsSidecar.REQUIRED,
        }:
            raise USPoolRuntimePlanError(
                f"execution_abi/logical_stages/{stage.id}: durable stage "
                "requires a forbidden or required receipt policy"
            )
        if (
            not stage.durable
            and stage.operational_receipts_sidecar
            != OperationalReceiptsSidecar.NOT_APPLICABLE
        ):
            raise USPoolRuntimePlanError(
                f"execution_abi/logical_stages/{stage.id}: non-durable stage "
                "requires a not_applicable receipt policy"
            )
        if any(by_operation[item].stage_id != stage.id for item in stage.operation_ids):
            raise USPoolRuntimePlanError(
                f"execution_abi/logical_stages/{stage.id}: operation stage mismatch"
            )
        if stage.producer_graph_operation is None:
            if stage.producer_node_ids:
                raise USPoolRuntimePlanError(
                    f"execution_abi/logical_stages/{stage.id}: producer nodes "
                    "require a producer graph operation"
                )
        elif stage.producer_graph_operation not in stage.operation_ids:
            raise USPoolRuntimePlanError(
                f"execution_abi/logical_stages/{stage.id}: producer graph "
                "operation is outside the stage"
            )
    producer_stages = tuple(
        stage for stage in stages if stage.producer_graph_operation is not None
    )
    if len(producer_stages) != 1:
        raise USPoolRuntimePlanError(
            "execution_abi: exactly one producer-graph stage is required"
        )
    node_order = tuple(node.id for node in nodes)
    producer_stage = producer_stages[0]
    if (
        _string_tuple(
            pipeline.get("producer_order"),
            location="execution_abi/pipeline/producer_order",
            nonempty=True,
        )
        != node_order
    ):
        raise USPoolRuntimePlanError(
            "execution_abi: pipeline producer order differs from compiled nodes"
        )
    if producer_stage.producer_node_ids != node_order:
        raise USPoolRuntimePlanError(
            "execution_abi: producer stage order differs from compiled node order"
        )
    if (
        by_operation[producer_stage.producer_graph_operation].nested_producer_nodes
        != node_order
    ):
        raise USPoolRuntimePlanError(
            "execution_abi: producer operation order differs from compiled nodes"
        )
    for operation in operations:
        if (
            operation.id != producer_stage.producer_graph_operation
            and operation.nested_producer_nodes
        ):
            raise USPoolRuntimePlanError(
                "execution_abi: only the producer graph operation may nest nodes"
            )

    checkpoint_by_id = {row.id: row for row in checkpoints}
    durable_stage_ids = tuple(stage.id for stage in stages if stage.durable)
    if tuple(row.id for row in checkpoints) != durable_stage_ids:
        raise USPoolRuntimePlanError(
            "execution_abi: durable checkpoints differ from durable logical stages"
        )
    covered: list[str] = []
    for stage in stages:
        covered.extend(stage.operation_ids)
        checkpoint = checkpoint_by_id.get(stage.id)
        if checkpoint is None:
            continue
        if checkpoint.after_operation != stage.operation_ids[
            -1
        ] or checkpoint.covered_operation_ids != tuple(covered):
            raise USPoolRuntimePlanError(
                f"execution_abi/durable_checkpoints/{stage.id}: boundary mismatch"
            )
        if (
            checkpoint.operational_receipts_sidecar
            != stage.operational_receipts_sidecar
        ):
            raise USPoolRuntimePlanError(
                f"execution_abi/durable_checkpoints/{stage.id}: operational "
                "receipt sidecar policy differs from its stage"
            )

    seed_map_sha256 = sha256_json(authority.seed_stream_map.to_wire())
    if pipeline.get("seed_stream_map_sha256") != seed_map_sha256:
        raise USPoolRuntimePlanError(
            "execution_abi: pipeline seed map digest differs from compiler map"
        )

    return ExecutionAuthority(
        abi_sha256=abi_sha256,
        pipeline=pipeline,
        operations=operations,
        stages=stages,
        checkpoints=checkpoints,
        code_abi=_map(
            execution_abi.get("code_abi"),
            location="execution_abi/code_abi",
        ),
        artifact_vector=_map_tuple(
            execution_abi.get("normative_artifact_vector"),
            location="execution_abi/normative_artifact_vector",
            nonempty=True,
        ),
        receipt_comparison_vector=_map_tuple(
            execution_abi.get("receipt_comparison_vector"),
            location="execution_abi/receipt_comparison_vector",
            nonempty=True,
        ),
        resume_predicate=_compile_resume_predicate(
            execution_abi,
            checkpoints=checkpoints,
        ),
        stacked_authority=_projection(
            authority,
            USAuthorityProjection.STACKED_AUTHORITY,
        ),
        checkpoint_static_components=_projection(
            authority,
            USAuthorityProjection.STACKED_CHECKPOINT_STATIC_COMPONENTS,
        ),
    )


def _compile_sources(authority: USSpecAuthority) -> SourceAuthority:
    contract = authority.behavior_resource(ResourceKind.SOURCES)
    vintage_contract = authority.behavior_resource(ResourceKind.VINTAGES)
    _map_tuple(
        contract.get("sources"),
        location="sources/sources",
        nonempty=True,
    )
    _map_tuple(
        contract.get("stages"),
        location="sources/stages",
        nonempty=True,
    )
    _map(contract.get("stage_asset"), location="sources/stage_asset")
    _map(contract.get("stage_manifest"), location="sources/stage_manifest")
    _map_tuple(
        vintage_contract.get("records"),
        location="vintages/records",
        nonempty=True,
    )
    vintages = authority.vintage_authorities
    _map(vintages.get("records"), location="vintage_authorities/records")
    result = SourceAuthority(
        declared=authority.declared_sources,
        contract=contract,
        vintage_contract=vintage_contract,
        vintages=vintages,
    )
    # Force the closed registry shapes now rather than at first lookup.
    _ = result.declared_rows
    _ = result.stage_rows
    return result


def _compile_support_spine(spine: FrozenMap) -> SupportSpineAuthority:
    return SupportSpineAuthority(
        source_pool=_map(
            spine.get("support_source_pool"),
            location="spine/support_source_pool",
        ),
        source_pool_metadata=_map(
            spine.get("support_source_pool_metadata"),
            location="spine/support_source_pool_metadata",
        ),
        roles=_map_tuple(
            spine.get("support_roles"),
            location="spine/support_roles",
            nonempty=True,
        ),
        channels=_map_tuple(
            spine.get("channels"),
            location="spine/channels",
            nonempty=True,
        ),
    )


def _compile_assembly_sampling(
    authority: USSpecAuthority,
    *,
    spine: FrozenMap,
) -> AssemblySamplingAuthority:
    return AssemblySamplingAuthority(
        assembly=_map(spine.get("assembly"), location="spine/assembly"),
        sampling_contract=_map(
            spine.get("sampling"),
            location="spine/sampling",
        ),
        runtime_sampling=_projection(authority, USAuthorityProjection.SAMPLING),
        seed_site_bindings=_map_tuple(
            spine.get("seed_site_bindings"),
            location="spine/seed_site_bindings",
            nonempty=True,
        ),
    )


def _compile_imputation(authority: USSpecAuthority) -> ImputationProducerAuthority:
    contract = authority.behavior_resource(ResourceKind.IMPUTATION)
    graph = _map(
        contract.get("producer_graph"),
        location="imputation/producer_graph",
    )
    authored_nodes = _map_tuple(
        graph.get("nodes"),
        location="imputation/producer_graph/nodes",
        nonempty=True,
    )
    compiled_nodes = authority.nodes
    if not compiled_nodes:
        raise USPoolRuntimePlanError(
            "imputation/producer_graph: compiled nodes are absent"
        )
    authored_ids = tuple(
        _string(
            row.get("id"),
            location=f"imputation/producer_graph/nodes/{index}/id",
        )
        for index, row in enumerate(authored_nodes)
    )
    compiled_ids = tuple(node.id for node in compiled_nodes)
    if set(authored_ids) != set(compiled_ids):
        raise USPoolRuntimePlanError(
            "imputation/producer_graph: authored and compiled node sets differ"
        )
    if tuple(node.execution_rank for node in compiled_nodes) != tuple(
        range(len(compiled_nodes))
    ):
        raise USPoolRuntimePlanError(
            "imputation/producer_graph: compiled execution ranks are not contiguous"
        )
    return ImputationProducerAuthority(
        contract=contract,
        runtime=_projection(authority, USAuthorityProjection.IMPUTATION),
        producer_graph=graph,
        nodes=compiled_nodes,
    )


def _compile_remaining_stage(authority: USSpecAuthority) -> RemainingStageAuthority:
    engine_lock = _map(
        authority.generated_authorities.get("engine_abi_lock"),
        location="generated_authorities/engine_abi_lock",
    )
    manifest = _map(
        engine_lock.get("remaining_stage_input_manifest"),
        location=(
            "generated_authorities/engine_abi_lock/remaining_stage_input_manifest"
        ),
    )
    receipt = _map(
        manifest.get("receipt"),
        location=(
            "generated_authorities/engine_abi_lock/"
            "remaining_stage_input_manifest/receipt"
        ),
    )
    rows = _map_tuple(
        manifest.get("rows"),
        location=(
            "generated_authorities/engine_abi_lock/remaining_stage_input_manifest/rows"
        ),
        nonempty=True,
    )
    return RemainingStageAuthority(
        engine=_map(
            engine_lock.get("engine"),
            location="generated_authorities/engine_abi_lock/engine",
        ),
        manifest=manifest,
        receipt=receipt,
        rows=rows,
    )


@dataclass(frozen=True, slots=True)
class USPoolRuntimePlan:
    """Closed, immutable pool plan derived from one compiler authority."""

    authority_sha256: str
    spec_sha256: str
    identity_generation: int
    sources: SourceAuthority
    support_spine: SupportSpineAuthority
    assembly_sampling: AssemblySamplingAuthority
    publication: PublicationAuthority
    execution: ExecutionAuthority
    imputation: ImputationProducerAuthority
    take_up: TakeUpAuthority
    remaining_stage: RemainingStageAuthority
    battery: BatteryAuthority
    seed_stream_map: SeedStreamMap
    _seal_sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        if self._seal_sha256 != _plan_sha256(self):
            raise USPoolRuntimePlanError(
                "US pool runtime plan seal differs from its compiler authorities"
            )

    @classmethod
    def from_spec_authority(
        cls,
        authority: USSpecAuthority,
    ) -> USPoolRuntimePlan:
        """Compile the physical pool plan from the sole US capability."""

        if not isinstance(authority, USSpecAuthority):
            raise TypeError(
                "USPoolRuntimePlan.from_spec_authority requires USSpecAuthority"
            )
        if authority.identity_generation != 1:
            raise USPoolRuntimePlanError(
                "US pool runtime plan requires identity_generation 1"
            )
        for name, value in (
            ("authority_sha256", authority.authority_sha256),
            ("spec_sha256", authority.spec_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise USPoolRuntimePlanError(
                    f"US pool runtime plan requires a sealed {name}"
                )
        if not isinstance(authority.seed_stream_map, SeedStreamMap):
            raise USPoolRuntimePlanError(
                "US pool runtime plan requires a typed SeedStreamMap"
            )

        spine = authority.behavior_resource(ResourceKind.SPINE)
        publication_contract = authority.behavior_resource(ResourceKind.PUBLICATION)
        take_up_contract = authority.behavior_resource(ResourceKind.TAKE_UP)
        battery_contract = authority.behavior_resource(ResourceKind.BATTERY)
        imputation = _compile_imputation(authority)

        values = dict(
            authority_sha256=authority.authority_sha256,
            spec_sha256=authority.spec_sha256,
            identity_generation=authority.identity_generation,
            sources=_compile_sources(authority),
            support_spine=_compile_support_spine(spine),
            assembly_sampling=_compile_assembly_sampling(authority, spine=spine),
            publication=PublicationAuthority(
                contract=publication_contract,
                runtime=_projection(authority, USAuthorityProjection.PUBLICATION),
            ),
            execution=_compile_execution(authority, nodes=imputation.nodes),
            imputation=imputation,
            take_up=TakeUpAuthority(
                contract=take_up_contract,
                runtime=_projection(authority, USAuthorityProjection.TAKE_UP),
            ),
            remaining_stage=_compile_remaining_stage(authority),
            battery=BatteryAuthority(
                contract=battery_contract,
                runtime=_projection(authority, USAuthorityProjection.BATTERY),
                components=_projection(
                    authority,
                    USAuthorityProjection.BATTERY_COMPONENTS,
                ),
            ),
            seed_stream_map=authority.seed_stream_map,
        )
        return cls(**values, _seal_sha256=_plan_values_sha256(values))


def _plan_wire_value(value: object) -> object:
    if isinstance(value, FrozenMap):
        return thaw_json(value)
    if isinstance(value, (CompiledNode, SeedStreamMap)):
        return value.to_wire()
    if isinstance(value, tuple):
        return [_plan_wire_value(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: _plan_wire_value(getattr(value, item.name))
            for item in fields(value)
            if item.name != "_seal_sha256"
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise USPoolRuntimePlanError(
        f"US pool runtime plan contains an unsupported value: {type(value).__name__}"
    )


def _plan_values_sha256(values: dict[str, object]) -> str:
    return sha256_json(
        {
            "domain": "microcosm.us-runtime.pool-runtime-plan.v1",
            "plan": {key: _plan_wire_value(value) for key, value in values.items()},
        }
    )


def _plan_sha256(plan: USPoolRuntimePlan) -> str:
    return _plan_values_sha256(
        {
            item.name: getattr(plan, item.name)
            for item in fields(plan)
            if item.name != "_seal_sha256"
        }
    )


__all__ = [
    "AssemblySamplingAuthority",
    "BatteryAuthority",
    "DurableCheckpoint",
    "ExecutionAuthority",
    "ExecutionOperation",
    "ExecutionStage",
    "ImputationProducerAuthority",
    "OperationalReceiptsSidecar",
    "PublicationAuthority",
    "RemainingStageAuthority",
    "SourceAuthority",
    "SupportSpineAuthority",
    "TakeUpAuthority",
    "USPoolRuntimePlan",
    "USPoolRuntimePlanError",
]
