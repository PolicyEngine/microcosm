"""Bundle-only bridge from the US pool DAG to the generic node executor.

The pool's physical callbacks still return rich legacy result objects.  This
module keeps those callbacks behind the compiler-owned executor boundary:

* a full :class:`~microcosm.frame.Frame` is narrowed to the compiled node's
  projection;
* the callback is prebound as a broker-owned :class:`PhysicalOperation`;
* a small registered kernel requests that exact operation and returns its
  declarative patch;
* the validated projection is merged back into the checked legacy frame while
  preserving every non-frame result sidecar.

The execution journal is operational evidence.  It is intentionally separate
from artifact bytes, plan identity, and semantic node-reuse identity.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType

from microcosm.build.spec_engine.brokers import (
    BrokerSession,
    DeclaredSource,
    KernelBrokerSession,
    PhysicalOperation,
    RNGInvocation,
)
from microcosm.build.spec_engine.compiler_ir import (
    CompiledNode,
    SeedOwnerIR,
    SeedStreamMap,
)
from microcosm.build.spec_engine.executor import (
    ExecutionContext,
    KernelPatch,
    RegisteredKernel,
    RegisteredRowClassifier,
    RunProvenanceIdentity,
    apply_patch,
    execute_node,
    order_nodes,
)
from microcosm.build.spec_engine.model import thaw_json
from microcosm.build.spec_engine.scope_algebra import ClosedScopeRegistry
from microcosm.build.us_runtime.pool_frame_projection import (
    classify_added_support_rows,
    frame_to_projection,
    legacy_result_to_patch,
    merge_projection_into_frame,
    projection_to_frame,
)
from microcosm.frame import Frame

__all__ = [
    "USPoolPhysicalExecutionRecord",
    "USPoolPhysicalExecutor",
    "USPoolPhysicalExecutorError",
]


_JOURNAL_DOMAIN = "microcosm.us-pool.physical-executor-journal.v1"
_INPUT_BINDING_DOMAIN = "microcosm.us-pool.physical-input-binding.v1"


class USPoolPhysicalExecutorError(ValueError):
    """A pool callback cannot be dispatched under its compiled node."""


@dataclass(frozen=True, slots=True)
class USPoolPhysicalExecutionRecord:
    """Non-normative evidence for one successfully validated node."""

    execution_index: int
    node_id: str
    node_key: str
    input_binding_sha256: str
    patch_sha256: str
    result_projection_sha256: str
    broker_receipt_sha256: str

    def to_wire(self) -> dict[str, object]:
        return {
            "execution_index": self.execution_index,
            "node_id": self.node_id,
            "node_key": self.node_key,
            "input_binding_sha256": self.input_binding_sha256,
            "patch_sha256": self.patch_sha256,
            "result_projection_sha256": self.result_projection_sha256,
            "broker_receipt_sha256": self.broker_receipt_sha256,
            "status": "complete",
        }


def _pickle_sha256(value: object) -> str:
    try:
        payload = pickle.dumps(value, protocol=5)
    except (AttributeError, pickle.PickleError, TypeError, ValueError) as error:
        raise USPoolPhysicalExecutorError(
            "physical executor input cannot be content-bound"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _normalized_available_inputs(
    available_inputs: Mapping[object, object],
) -> tuple[tuple[tuple[str, str], object], ...]:
    rows: list[tuple[tuple[str, str], object]] = []
    for raw_key, value in available_inputs.items():
        if isinstance(raw_key, tuple) and len(raw_key) == 2:
            entity, column = raw_key
        elif isinstance(raw_key, str) and "." in raw_key:
            entity, column = raw_key.split(".", 1)
        else:
            raise USPoolPhysicalExecutorError(
                "available-input keys must be (entity, column) pairs or "
                "'entity.column' strings"
            )
        if not isinstance(entity, str) or not entity:
            raise USPoolPhysicalExecutorError(
                "available-input entities must be non-empty strings"
            )
        if not isinstance(column, str) or not column:
            raise USPoolPhysicalExecutorError(
                "available-input columns must be non-empty strings"
            )
        rows.append(((entity, column), copy.deepcopy(value)))
    keys = [key for key, _value in rows]
    if len(keys) != len(set(keys)):
        raise USPoolPhysicalExecutorError(
            "available-input keys repeat after canonical normalization"
        )
    return tuple(sorted(rows, key=lambda row: row[0]))


def _available_inputs_with_absence(
    node: CompiledNode,
    available_inputs: Mapping[object, object],
    absence_receipts: Mapping[str, object] | None,
) -> dict[tuple[str, str], object]:
    """Bind each supplied absence receipt to its unique compiled entity."""

    merged = dict(_normalized_available_inputs(available_inputs))
    if absence_receipts is None:
        return merged
    if not isinstance(absence_receipts, Mapping):
        raise USPoolPhysicalExecutorError("absence_receipts must be a mapping")

    receipt_entities: dict[str, set[str]] = {}
    for input_index, frozen in enumerate(node.inputs):
        value = thaw_json(frozen)
        if not isinstance(value, dict):
            raise USPoolPhysicalExecutorError(
                f"compiled node {node.id!r} input {input_index} is not an object"
            )
        entity = value.get("entity")
        receipts = value.get("tolerated_absence_receipts")
        if not isinstance(entity, str) or not entity:
            raise USPoolPhysicalExecutorError(
                f"compiled node {node.id!r} input {input_index} has no entity"
            )
        if not isinstance(receipts, list) or any(
            not isinstance(receipt, str) or not receipt for receipt in receipts
        ):
            raise USPoolPhysicalExecutorError(
                f"compiled node {node.id!r} input {input_index} has invalid "
                "tolerated absence receipts"
            )
        for receipt in receipts:
            receipt_entities.setdefault(receipt, set()).add(entity)

    for receipt_id, receipt in absence_receipts.items():
        if not isinstance(receipt_id, str) or not receipt_id:
            raise USPoolPhysicalExecutorError(
                "absence receipt ids must be non-empty strings"
            )
        entities = receipt_entities.get(receipt_id, set())
        if not entities:
            raise USPoolPhysicalExecutorError(
                f"absence receipt {receipt_id!r} is not declared by node {node.id!r}"
            )
        if len(entities) != 1:
            raise USPoolPhysicalExecutorError(
                f"absence receipt {receipt_id!r} is ambiguous across entities "
                f"{sorted(entities)!r}"
            )
        key = (next(iter(entities)), receipt_id)
        if key in merged:
            raise USPoolPhysicalExecutorError(
                f"absence receipt {receipt_id!r} duplicates an available input"
            )
        merged[key] = copy.deepcopy(receipt)
    return merged


def _input_binding_sha256(
    *,
    node: CompiledNode,
    frame: Frame,
    available_inputs: Mapping[object, object],
) -> str:
    return _pickle_sha256(
        {
            "domain": _INPUT_BINDING_DOMAIN,
            "node_id": node.id,
            "node_key": node.node_key,
            "frame_sha256": _pickle_sha256(frame),
            "available_inputs": _normalized_available_inputs(available_inputs),
        }
    )


def _node_capabilities(node: CompiledNode) -> dict[str, object]:
    value = thaw_json(node.capabilities)
    if not isinstance(value, dict):  # pragma: no cover - compiler IR is typed
        raise USPoolPhysicalExecutorError(
            f"compiled node {node.id!r} capabilities are not an object"
        )
    return value


def _node_outputs(node: CompiledNode) -> tuple[tuple[str, str], ...]:
    outputs: list[tuple[str, str]] = []
    for index, frozen in enumerate(node.outputs):
        value = thaw_json(frozen)
        if not isinstance(value, dict):
            raise USPoolPhysicalExecutorError(
                f"compiled node {node.id!r} output {index} is not an object"
            )
        entity = value.get("entity")
        column = value.get("column")
        if not isinstance(entity, str) or not entity:
            raise USPoolPhysicalExecutorError(
                f"compiled node {node.id!r} output {index} has no entity"
            )
        if not isinstance(column, str) or not column:
            raise USPoolPhysicalExecutorError(
                f"compiled node {node.id!r} output {index} has no column"
            )
        outputs.append((entity, column))
    return tuple(outputs)


def _declared_virtual_writes(
    node: CompiledNode,
    result: object,
    result_frame: Frame,
) -> dict[tuple[str, str], object]:
    """Resolve every declared virtual output without guessing its meaning."""

    writes: dict[tuple[str, str], object] = {}
    for entity, column in _node_outputs(node):
        if not column.startswith("@"):
            continue
        if column == "@resolved_weight":
            # The codec represents this through KernelPatch.weights.  Resolve it
            # here so a missing entity/weight fails before dispatch, but do not
            # mislabel the weight surface as a virtual-receipt write.
            result_frame.resolve_weights(entity)
            continue
        if entity == "frame":
            metadata_key = column.removeprefix("@")
            if metadata_key not in result_frame.metadata:
                raise USPoolPhysicalExecutorError(
                    f"legacy result for {node.id!r} lacks declared frame metadata "
                    f"{metadata_key!r}"
                )
            writes[(entity, column)] = copy.deepcopy(
                result_frame.metadata[metadata_key]
            )
            continue
        if entity == "person" and column.startswith("@source_receipt:"):
            operator = column.removeprefix("@source_receipt:")
            if node.id != f"source:{operator}":
                raise USPoolPhysicalExecutorError(
                    f"source receipt {column!r} is not owned by node {node.id!r}"
                )
            receipt = getattr(result, "receipt", None)
            if not isinstance(receipt, Mapping):
                raise USPoolPhysicalExecutorError(
                    f"legacy result for {node.id!r} lacks a mapping receipt"
                )
            writes[(entity, column)] = copy.deepcopy(dict(receipt))
            continue
        raise USPoolPhysicalExecutorError(
            f"compiled node {node.id!r} has unsupported virtual output "
            f"{entity}.{column}; refusing to infer a receipt"
        )
    return writes


def _make_physical_kernel(
    input_binding_sha256: str,
) -> Callable[[object, ExecutionContext], KernelPatch]:
    """Return the tiny generic kernel bound only to an input digest."""

    def run(_projection: object, context: ExecutionContext) -> KernelPatch:
        brokers = context.brokers
        if not isinstance(brokers, KernelBrokerSession):
            raise USPoolPhysicalExecutorError(
                "physical kernel requires the executor's narrowed broker view"
            )
        patch = brokers.run_physical_operation(
            input_binding_sha256=input_binding_sha256
        )
        if not isinstance(patch, KernelPatch):
            raise USPoolPhysicalExecutorError(
                "physical operation did not return a KernelPatch"
            )
        return patch

    return run


def _replacement_result(result: object, frame: Frame) -> object:
    if is_dataclass(result) and not isinstance(result, type):
        try:
            return replace(result, frame=frame)
        except (TypeError, ValueError) as error:
            raise USPoolPhysicalExecutorError(
                "legacy dataclass result does not expose a replaceable frame field"
            ) from error
    original = getattr(result, "frame", None)
    if not isinstance(original, Frame):
        raise USPoolPhysicalExecutorError(
            "legacy callback result must expose a Frame as .frame"
        )
    if _pickle_sha256(original) != _pickle_sha256(frame):
        raise USPoolPhysicalExecutorError(
            "validated frame differs from the non-dataclass legacy result"
        )
    return result


def _per_node_values(
    values: Mapping[str, object] | None,
    *,
    node_ids: frozenset[str],
    label: str,
) -> dict[str, object]:
    result = {} if values is None else dict(values)
    unknown = set(result) - node_ids
    if unknown:
        raise USPoolPhysicalExecutorError(
            f"{label} names unknown compiled nodes: {sorted(unknown)!r}"
        )
    return result


class USPoolPhysicalExecutor:
    """Execute one exact compiler-ordered US physical producer sequence."""

    def __init__(
        self,
        nodes: Sequence[CompiledNode],
        *,
        run_provenance_identity: RunProvenanceIdentity,
        seed_stream_map: SeedStreamMap,
        run_inputs: Mapping[str, int] | None = None,
        sink_roots_by_node: Mapping[str, Sequence[Path | str]] | None = None,
        sources_by_node: Mapping[str, Sequence[DeclaredSource]] | None = None,
        environment_by_node: Mapping[str, Mapping[str, str | None]] | None = None,
        clocks_by_node: Mapping[str, Mapping[str, float]] | None = None,
        attempt: int = 0,
        resumed: bool = False,
        require_byte_equivalence: bool = True,
    ) -> None:
        rows = tuple(nodes)
        if not rows or any(not isinstance(node, CompiledNode) for node in rows):
            raise USPoolPhysicalExecutorError(
                "physical executor requires a non-empty CompiledNode sequence"
            )
        ordered = order_nodes(rows)
        if tuple(node.id for node in rows) != tuple(node.id for node in ordered):
            raise USPoolPhysicalExecutorError(
                "physical executor nodes must already be in compiled execution order"
            )
        if not isinstance(run_provenance_identity, RunProvenanceIdentity):
            raise USPoolPhysicalExecutorError(
                "physical executor requires a typed RunProvenanceIdentity"
            )
        if not isinstance(seed_stream_map, SeedStreamMap):
            raise USPoolPhysicalExecutorError(
                "physical executor requires the compiled SeedStreamMap"
            )
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise USPoolPhysicalExecutorError("attempt must be a non-negative integer")
        if not isinstance(resumed, bool) or not isinstance(
            require_byte_equivalence, bool
        ):
            raise USPoolPhysicalExecutorError(
                "resumed and require_byte_equivalence must be booleans"
            )

        node_ids = frozenset(node.id for node in rows)
        self._nodes = rows
        self._run_provenance_identity = run_provenance_identity
        self._seed_stream_map = seed_stream_map
        self._run_inputs = MappingProxyType(dict(run_inputs or {}))
        self._sink_roots = _per_node_values(
            sink_roots_by_node,
            node_ids=node_ids,
            label="sink_roots_by_node",
        )
        self._sources = _per_node_values(
            sources_by_node,
            node_ids=node_ids,
            label="sources_by_node",
        )
        self._environments = _per_node_values(
            environment_by_node,
            node_ids=node_ids,
            label="environment_by_node",
        )
        self._clocks = _per_node_values(
            clocks_by_node,
            node_ids=node_ids,
            label="clocks_by_node",
        )
        self._attempt = attempt
        self._resumed = resumed
        self._require_byte_equivalence = require_byte_equivalence
        self._next_index = 0
        self._journal: list[USPoolPhysicalExecutionRecord] = []
        self._failed_node_id: str | None = None

    @property
    def journal(self) -> tuple[USPoolPhysicalExecutionRecord, ...]:
        return tuple(self._journal)

    @property
    def remaining_node_ids(self) -> tuple[str, ...]:
        return tuple(node.id for node in self._nodes[self._next_index :])

    def seed_site_ids(self, producer_name: str) -> tuple[str, ...]:
        """Return the compiled direct RNG sites for one physical producer."""

        node = next((row for row in self._nodes if row.id == producer_name), None)
        if node is None:
            raise USPoolPhysicalExecutorError(
                f"unknown physical producer {producer_name!r}"
            )
        return tuple(site.id for site in node.seed_sites)

    def seed_owner(self, kind: str, owner_id: str) -> SeedOwnerIR:
        """Resolve one supplemental owner from the executor's compiled map."""

        owner = self._seed_stream_map.owner(kind, owner_id)
        if owner is None:
            raise USPoolPhysicalExecutorError(
                f"compiled seed owner {(kind, owner_id)!r} is absent"
            )
        return owner

    def _expected_node(self, producer_name: str) -> CompiledNode:
        if self._failed_node_id is not None:
            raise USPoolPhysicalExecutorError(
                f"physical executor is sealed by failed node "
                f"{self._failed_node_id!r}"
            )
        if self._next_index == len(self._nodes):
            raise USPoolPhysicalExecutorError(
                f"physical executor is already complete; unexpected "
                f"producer {producer_name!r}"
            )
        node = self._nodes[self._next_index]
        if producer_name != node.id:
            raise USPoolPhysicalExecutorError(
                f"physical producer order mismatch: expected {node.id!r}, "
                f"got {producer_name!r}"
            )
        return node

    def dispatch(
        self,
        producer_name: str,
        before_frame: Frame,
        available_inputs: Mapping[object, object],
        operation: Callable[[Frame, KernelBrokerSession], object],
        *,
        absence_receipts: Mapping[str, object] | None = None,
        rng_invocation_plan: Mapping[str, Sequence[RNGInvocation]] | None = None,
        rng_invocation_plan_factory: (
            Callable[[Frame], Mapping[str, Sequence[RNGInvocation]]] | None
        ) = None,
        supplemental_seed_owners: Sequence[SeedOwnerIR] = (),
        rng_invocation_plans_by_owner: Mapping[
            tuple[str, str], Mapping[str, Sequence[RNGInvocation]]
        ]
        | None = None,
    ) -> object:
        """Run one legacy callback through ``execute_node`` exactly once."""

        if not isinstance(producer_name, str) or not producer_name:
            raise USPoolPhysicalExecutorError(
                "physical producer name must be non-empty"
            )
        if not isinstance(before_frame, Frame):
            raise USPoolPhysicalExecutorError(
                "physical executor before_frame must be a Frame"
            )
        if not isinstance(available_inputs, Mapping):
            raise USPoolPhysicalExecutorError(
                "physical executor available_inputs must be a mapping"
            )
        if not callable(operation):
            raise USPoolPhysicalExecutorError(
                "physical executor operation must be callable"
            )
        if rng_invocation_plan_factory is not None and not callable(
            rng_invocation_plan_factory
        ):
            raise USPoolPhysicalExecutorError(
                "RNG invocation plan factory must be callable"
            )
        if rng_invocation_plan is not None and rng_invocation_plan_factory is not None:
            raise USPoolPhysicalExecutorError(
                "RNG invocation plan and factory are mutually exclusive"
            )
        if (
            rng_invocation_plan is not None
            or rng_invocation_plan_factory is not None
        ) and (supplemental_seed_owners or rng_invocation_plans_by_owner is not None):
            raise USPoolPhysicalExecutorError(
                "direct RNG plans cannot be combined with owner-scoped plans"
            )
        node = self._expected_node(producer_name)
        executor_inputs = _available_inputs_with_absence(
            node,
            available_inputs,
            absence_receipts,
        )
        input_binding_sha256 = _input_binding_sha256(
            node=node,
            frame=before_frame,
            available_inputs=executor_inputs,
        )
        projection = frame_to_projection(
            before_frame,
            node=node,
            available_inputs=executor_inputs,
        )
        effective_rng_invocation_plan = rng_invocation_plan
        if rng_invocation_plan_factory is not None:
            planner_frame = projection_to_frame(
                projection,
                schema=before_frame.schema,
            )
            if not isinstance(planner_frame, Frame):
                self._failed_node_id = node.id
                raise USPoolPhysicalExecutorError(
                    f"RNG planner frame for {node.id!r} is not a Frame"
                )
            try:
                planned = rng_invocation_plan_factory(planner_frame)
            except Exception:
                self._failed_node_id = node.id
                raise
            if not isinstance(planned, Mapping):
                self._failed_node_id = node.id
                raise USPoolPhysicalExecutorError(
                    f"RNG planner for {node.id!r} did not return a mapping"
                )
            try:
                normalized_plan = {
                    site_id: tuple(invocations)
                    for site_id, invocations in planned.items()
                }
            except TypeError as error:
                self._failed_node_id = node.id
                raise USPoolPhysicalExecutorError(
                    f"RNG planner for {node.id!r} returned non-sequence invocations"
                ) from error
            if any(
                not isinstance(site_id, str)
                or not site_id
                or any(
                    not isinstance(invocation, RNGInvocation)
                    for invocation in invocations
                )
                for site_id, invocations in normalized_plan.items()
            ):
                self._failed_node_id = node.id
                raise USPoolPhysicalExecutorError(
                    f"RNG planner for {node.id!r} returned invalid invocations"
                )
            effective_rng_invocation_plan = normalized_plan
        narrow_frame = projection_to_frame(projection, schema=before_frame.schema)
        if not isinstance(narrow_frame, Frame):
            self._failed_node_id = node.id
            raise USPoolPhysicalExecutorError(
                f"physical callback frame for {node.id!r} is not a Frame"
            )
        result_box: list[object] = []

        def physical_call(context: KernelBrokerSession) -> object:
            result = operation(narrow_frame, context)
            result_frame = getattr(result, "frame", None)
            if not isinstance(result_frame, Frame):
                raise USPoolPhysicalExecutorError(
                    f"legacy callback for {node.id!r} did not return .frame"
                )
            writes = _declared_virtual_writes(node, result, result_frame)
            patch = legacy_result_to_patch(
                before_frame,
                result_frame,
                node=node,
                virtual_writes=writes,
            )
            result_box.append(result)
            return patch

        physical_operation = PhysicalOperation(
            function=physical_call,
            implementation_sha256=node.kernel_implementation_sha256,
            input_binding_sha256=input_binding_sha256,
            policy="broker-only",
            sink_roots=tuple(self._sink_roots.get(node.id, ())),
        )
        capabilities = _node_capabilities(node)
        effects = capabilities.get("effects")
        if not isinstance(effects, list) or any(
            not isinstance(effect, str) for effect in effects
        ):
            raise USPoolPhysicalExecutorError(
                f"compiled node {node.id!r} effects are not a string array"
            )
        retry_safety = capabilities.get("retry_safety")
        attempt_scope = (
            f"us-pool-physical:{self._attempt}:{node.id}"
            if retry_safety == "attempt_scoped"
            else None
        )
        broker_session = BrokerSession.for_compiled_node(
            node,
            run_provenance_identity=self._run_provenance_identity.to_wire(),
            run_inputs=self._run_inputs,
            rng_invocation_plan=effective_rng_invocation_plan,
            seed_stream_map=self._seed_stream_map,
            supplemental_seed_owners=tuple(supplemental_seed_owners),
            rng_invocation_plans_by_owner=rng_invocation_plans_by_owner,
            sources=tuple(self._sources.get(node.id, ())),
            environment=self._environments.get(node.id),
            clocks=self._clocks.get(node.id),
            attempt=self._attempt,
            attempt_scope=attempt_scope,
            require_byte_equivalence=self._require_byte_equivalence,
            physical_operation=physical_operation,
        )
        registry = ClosedScopeRegistry.from_wire(thaw_json(node.scope_registry))
        context = ExecutionContext(
            attempt=self._attempt,
            resumed=self._resumed,
            attempt_scope=attempt_scope,
            granted_effects=frozenset(effect for effect in effects if effect != "none"),
            require_byte_equivalence=self._require_byte_equivalence,
            brokers=broker_session,
            run_provenance_identity=self._run_provenance_identity,
        )
        kernel = RegisteredKernel(
            _make_physical_kernel(input_binding_sha256),
            node.kernel_implementation_sha256,
        )
        row_classifiers = None
        if capabilities.get("structural_delta") == "expand":
            classifier = RegisteredRowClassifier(
                classify_added_support_rows,
                node.row_classifier_implementation_sha256,
                registry.predicate_space,
            )
            row_classifiers = {node.row_classifier_ref: classifier}
        try:
            validated_patch = execute_node(
                node,
                projection,
                kernels={node.kernel_ref: kernel},
                row_classifiers=row_classifiers,
                context=context,
            )
            if len(result_box) != 1:
                raise USPoolPhysicalExecutorError(
                    f"physical operation for {node.id!r} was not invoked exactly once"
                )
            validated_projection = apply_patch(projection, validated_patch)
            result = result_box[0]
            legacy_result_frame = getattr(result, "frame", None)
            assert isinstance(legacy_result_frame, Frame)
            merged = merge_projection_into_frame(
                validated_projection,
                before_frame=before_frame,
                legacy_result_frame=legacy_result_frame,
                node=node,
                validated_metadata=legacy_result_frame.metadata,
            )
            returned = _replacement_result(result, merged)
        except Exception:
            self._failed_node_id = node.id
            raise

        self._journal.append(
            USPoolPhysicalExecutionRecord(
                execution_index=self._next_index,
                node_id=node.id,
                node_key=node.node_key,
                input_binding_sha256=input_binding_sha256,
                patch_sha256=validated_patch._patch_sha256,
                result_projection_sha256=validated_patch._result_sha256,
                broker_receipt_sha256=(
                    validated_patch.broker_receipt.receipt_sha256
                ),
            )
        )
        self._next_index += 1
        return returned

    def complete(self) -> tuple[USPoolPhysicalExecutionRecord, ...]:
        """Seal the dispatcher only after every compiled node was consumed."""

        if self._failed_node_id is not None:
            raise USPoolPhysicalExecutorError(
                f"physical executor cannot complete after failed node "
                f"{self._failed_node_id!r}"
            )
        if self._next_index != len(self._nodes):
            raise USPoolPhysicalExecutorError(
                "physical executor is incomplete; remaining nodes="
                f"{list(self.remaining_node_ids)!r}"
            )
        return self.journal

    def journal_wire(self) -> dict[str, object]:
        """Return typed operational evidence after exact completion."""

        records = self.complete()
        return {
            "domain": _JOURNAL_DOMAIN,
            "schema_version": 1,
            "surface": "operational",
            "node_count": len(records),
            "nodes": [record.to_wire() for record in records],
            "status": "complete",
        }
