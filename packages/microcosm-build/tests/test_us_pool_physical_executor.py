"""Fixture-scale tests for bundle physical-node dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.pool_physical_executor as physical_module
from microcosm.build.spec_engine.compiler_ir import CompiledNode
from microcosm.build.spec_engine.executor import KernelPatch
from microcosm.build.spec_engine.model import FrozenMap, freeze_json
from microcosm.build.spec_engine.scope_algebra import ClosedScopeRegistry
from microcosm.build.us_runtime.pool_physical_executor import (
    USPoolPhysicalExecutor,
    USPoolPhysicalExecutorError,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

_SHA = "a" * 64


def _frozen(value: object) -> FrozenMap:
    result = freeze_json(value)
    assert isinstance(result, FrozenMap)
    return result


def _input(
    entity: str,
    column: str,
    *,
    tolerated: tuple[str, ...] = (),
) -> FrozenMap:
    return _frozen(
        {
            "entity": entity,
            "column": column,
            "required_scope": "whole_pool",
            "alternatives": [],
            "tolerated_absence_receipts": list(tolerated),
        }
    )


def _node(
    node_id: str,
    rank: int,
    *,
    depends_on: tuple[str, ...] = (),
    inputs: tuple[FrozenMap, ...] = (),
    outputs: tuple[tuple[str, str], ...] = (),
) -> CompiledNode:
    registry = ClosedScopeRegistry(
        "fixture_pool_rows",
        ("origin:fixture/clone:0", "receipt:virtual"),
        {
            "whole_pool": ("origin:fixture/clone:0", "receipt:virtual"),
        },
    )
    return CompiledNode(
        id=node_id,
        execution_rank=rank,
        node_key=_SHA,
        node_slice_sha256=_SHA,
        kernel_ref=f"kernel:{node_id}",
        kernel_implementation_sha256=_SHA,
        depends_on=depends_on,
        inputs=inputs,
        outputs=tuple(
            _frozen(
                {
                    "entity": entity,
                    "column": column,
                    "coverage_scope": "whole_pool",
                }
            )
            for entity, column in outputs
        ),
        capabilities=_frozen(
            {
                "determinism": "deterministic",
                "numeric_reproducibility": "bitwise",
                "effects": ["none"],
                "structural_delta": "join",
                "retry_safety": "idempotent",
            }
        ),
        mutations=_frozen({}),
        write_scopes=(),
        scope_registry=_frozen(registry.to_wire()),
        row_classifier_ref="",
        row_classifier_implementation_sha256="",
        compiler_ir_abi=_frozen({}),
        seed_protocol_sha256=_SHA,
        seed_sites=(),
        seed_streams=(),
        resolved_params=(),
        transitive_nodes=(),
    )


def _frame(*, value: float = 1.0) -> Frame:
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.array([1], dtype=np.int64),
                    "person_household_id": np.array([10], dtype=np.int64),
                    "person_support_channel": ["fixture"],
                    "person_support_clone_index": np.array([0], dtype=np.int64),
                    "value": np.array([value], dtype=np.float64),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.array([10], dtype=np.int64),
                    "household_support_channel": ["fixture"],
                    "household_support_clone_index": np.array([0], dtype=np.int64),
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                np.array([1.0], dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


@dataclass(frozen=True)
class _LegacyResult:
    frame: Frame
    receipt: Mapping[str, object]
    sidecar: tuple[str, ...]


class _FakeKernelBroker:
    def __init__(self, operation) -> None:
        self.operation = operation
        self.calls = 0

    def run_physical_operation(self, *, input_binding_sha256: str) -> object:
        assert input_binding_sha256 == self.operation.input_binding_sha256
        self.calls += 1
        if self.calls != 1:
            raise AssertionError("physical callback repeated")
        return self.operation.function()


@pytest.fixture
def dispatch_harness(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    evidence: dict[str, object] = {
        "execute_calls": 0,
        "projection_inputs": None,
        "virtual_writes": None,
    }
    projection = object()
    validated_projection = object()
    fake_session = object()

    monkeypatch.setattr(physical_module, "order_nodes", lambda nodes: tuple(nodes))

    def project(
        _frame: Frame,
        *,
        node: CompiledNode,
        available_inputs: Mapping[object, object],
    ) -> object:
        assert isinstance(node, CompiledNode)
        evidence["projection_inputs"] = dict(available_inputs)
        return projection

    monkeypatch.setattr(physical_module, "frame_to_projection", project)

    def encode(
        _before: Frame,
        _result: Frame,
        *,
        node: CompiledNode,
        virtual_writes: Mapping[object, object],
    ) -> KernelPatch:
        assert isinstance(node, CompiledNode)
        evidence["virtual_writes"] = dict(virtual_writes)
        return KernelPatch("join")

    monkeypatch.setattr(physical_module, "legacy_result_to_patch", encode)

    def make_session(
        _node: CompiledNode,
        **kwargs: object,
    ) -> object:
        evidence["physical_operation"] = kwargs["physical_operation"]
        return fake_session

    monkeypatch.setattr(
        physical_module.BrokerSession,
        "for_compiled_node",
        make_session,
    )
    monkeypatch.setattr(physical_module, "KernelBrokerSession", _FakeKernelBroker)

    def execute(
        node: CompiledNode,
        incoming_projection: object,
        *,
        kernels: Mapping[str, object],
        row_classifiers: object,
        context: object,
    ) -> object:
        assert incoming_projection is projection
        assert row_classifiers is None
        assert context.brokers is fake_session
        operation = evidence["physical_operation"]
        broker = _FakeKernelBroker(operation)
        kernel_context = replace(context, brokers=broker)
        patch = kernels[node.kernel_ref].function(projection, kernel_context)
        assert isinstance(patch, KernelPatch)
        assert broker.calls == 1
        evidence["execute_calls"] = int(evidence["execute_calls"]) + 1
        return SimpleNamespace(
            _patch_sha256="b" * 64,
            _result_sha256="c" * 64,
            broker_receipt=SimpleNamespace(receipt_sha256="d" * 64),
        )

    monkeypatch.setattr(physical_module, "execute_node", execute)
    monkeypatch.setattr(
        physical_module,
        "apply_patch",
        lambda incoming, _patch: (
            validated_projection if incoming is projection else None
        ),
    )

    def merge(
        incoming: object,
        *,
        before_frame: Frame,
        legacy_result_frame: Frame,
        node: CompiledNode,
        validated_metadata: Mapping[str, object],
    ) -> Frame:
        assert incoming is validated_projection
        assert isinstance(before_frame, Frame)
        assert isinstance(node, CompiledNode)
        assert dict(validated_metadata) == dict(legacy_result_frame.metadata)
        return legacy_result_frame

    monkeypatch.setattr(physical_module, "merge_projection_into_frame", merge)
    return evidence


def test_dispatch_uses_executor_broker_and_preserves_result_sidecars(
    dispatch_harness: dict[str, object],
) -> None:
    absence_id = "optional_input:fixture:missing"
    node = _node(
        "source:fixture_operator",
        0,
        inputs=(
            _input("person", "@resource"),
            _input("person", "optional_value", tolerated=(absence_id,)),
        ),
        outputs=(
            ("person", "value"),
            ("person", "@source_receipt:fixture_operator"),
        ),
    )
    executor = USPoolPhysicalExecutor(
        (node,),
        run_provenance_identity=physical_module.RunProvenanceIdentity(
            identity_generation=0,
            source_grammar_receipt=None,
            spec_binding=None,
            authority_versions=_frozen({"fixture": 1}),
            code_inventory_digest=_SHA,
            artifact_protocol_inventory=_frozen({"fixture": "v1"}),
            run_request=_frozen({"fixture": True}),
            execution_receipt=_frozen({"backend": "fixture"}),
        ),
    )
    callback_calls = 0

    def operation() -> _LegacyResult:
        nonlocal callback_calls
        callback_calls += 1
        return _LegacyResult(
            frame=_frame(value=2.0),
            receipt={"sha256": "e" * 64},
            sidecar=("preserved",),
        )

    result = executor.dispatch(
        node.id,
        _frame(),
        {"person.@resource": {"version": 1}},
        operation,
        absence_receipts={absence_id: {"status": "tolerated"}},
    )

    assert callback_calls == 1
    assert isinstance(result, _LegacyResult)
    assert result.sidecar == ("preserved",)
    assert dispatch_harness["execute_calls"] == 1
    assert dispatch_harness["projection_inputs"] == {
        ("person", "@resource"): {"version": 1},
        ("person", absence_id): {"status": "tolerated"},
    }
    assert dispatch_harness["virtual_writes"] == {
        ("person", "@source_receipt:fixture_operator"): {
            "sha256": "e" * 64
        }
    }
    journal = executor.complete()
    assert [record.node_id for record in journal] == [node.id]
    assert journal[0].patch_sha256 == "b" * 64
    assert journal[0].broker_receipt_sha256 == "d" * 64
    assert executor.journal_wire()["surface"] == "operational"
    with pytest.raises(USPoolPhysicalExecutorError, match="already complete"):
        executor.dispatch(node.id, _frame(), {}, operation)


def test_dispatch_enforces_exact_order_and_complete_consumption(
    dispatch_harness: dict[str, object],
) -> None:
    first = _node("first", 0)
    second = _node("second", 1, depends_on=("first",))
    executor = USPoolPhysicalExecutor(
        (first, second),
        run_provenance_identity=physical_module.RunProvenanceIdentity(
            identity_generation=0,
            source_grammar_receipt=None,
            spec_binding=None,
            authority_versions=_frozen({"fixture": 1}),
            code_inventory_digest=_SHA,
            artifact_protocol_inventory=_frozen({"fixture": "v1"}),
            run_request=_frozen({"fixture": True}),
            execution_receipt=_frozen({"backend": "fixture"}),
        ),
    )

    with pytest.raises(USPoolPhysicalExecutorError, match="order mismatch"):
        executor.dispatch(second.id, _frame(), {}, lambda: None)
    with pytest.raises(USPoolPhysicalExecutorError, match="incomplete"):
        executor.complete()
    assert executor.remaining_node_ids == ("first", "second")
    assert dispatch_harness["execute_calls"] == 0


def test_absence_receipt_mapping_refuses_undeclared_or_ambiguous_ids(
    dispatch_harness: dict[str, object],
) -> None:
    receipt_id = "optional_input:fixture:ambiguous"
    node = _node(
        "fixture",
        0,
        inputs=(
            _input("person", "person_value", tolerated=(receipt_id,)),
            _input("household", "household_value", tolerated=(receipt_id,)),
        ),
    )
    executor = USPoolPhysicalExecutor(
        (node,),
        run_provenance_identity=physical_module.RunProvenanceIdentity(
            identity_generation=0,
            source_grammar_receipt=None,
            spec_binding=None,
            authority_versions=_frozen({"fixture": 1}),
            code_inventory_digest=_SHA,
            artifact_protocol_inventory=_frozen({"fixture": "v1"}),
            run_request=_frozen({"fixture": True}),
            execution_receipt=_frozen({"backend": "fixture"}),
        ),
    )

    with pytest.raises(USPoolPhysicalExecutorError, match="ambiguous"):
        executor.dispatch(
            node.id,
            _frame(),
            {},
            lambda: None,
            absence_receipts={receipt_id: {"status": "tolerated"}},
        )
    with pytest.raises(USPoolPhysicalExecutorError, match="not declared"):
        executor.dispatch(
            node.id,
            _frame(),
            {},
            lambda: None,
            absence_receipts={"unknown": {}},
        )
    assert dispatch_harness["execute_calls"] == 0
