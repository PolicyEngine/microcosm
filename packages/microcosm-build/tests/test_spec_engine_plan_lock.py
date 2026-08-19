"""Canonical emission and rejection gates for ``plan.lock.json``."""

from __future__ import annotations

import copy
import json

import pytest

from microcosm.build.spec_engine.compiler_ir import CompiledSpecIR, compile_spec
from microcosm.build.spec_engine.errors import SpecValidationError
from microcosm.build.spec_engine.loader import load_bundle
from microcosm.build.spec_engine.plan_lock import (
    PlanLockError,
    assert_plan_lock_current,
    assert_plan_lock_payload_current,
    emit_plan_lock,
    plan_lock_bytes,
    plan_lock_payload,
)
from microcosm.build.spec_engine.schemas import load_schema_registry


@pytest.fixture(scope="module")
def compiled_us() -> CompiledSpecIR:
    return compile_spec(load_bundle("us"))


def test_plan_lock_is_complete_and_closed_world(compiled_us: CompiledSpecIR) -> None:
    payload = plan_lock_payload(compiled_us)
    assert set(payload) == {
        "compiler_ir_abi",
        "spec_binding",
        "surfaces",
        "typed_inventory",
        "authorities",
        "stage_dag",
        "producer_graph",
        "seed_stream_map",
        "nodes",
    }
    assert payload["spec_binding"]["attestation"] == "mirror-attested"
    assert len(payload["producer_graph"]["nodes"]) == 38
    assert len(payload["producer_graph"]["authored"]["ownership_matrix"]) == 18
    assert len(payload["seed_stream_map"]["sites"]) == 72
    assert len(payload["nodes"]) == 38

    mutated = copy.deepcopy(payload)
    mutated["ignored_field"] = True
    with pytest.raises(SpecValidationError, match="Additional properties"):
        load_schema_registry().validate(
            mutated, "locks.schema.json#/$defs/plan_lock"
        )


def test_plan_lock_emits_and_asserts_exact_canonical_bytes(
    compiled_us: CompiledSpecIR,
    tmp_path,
) -> None:
    path = emit_plan_lock(compiled_us, tmp_path / "plan.lock.json")
    assert path.read_bytes() == plan_lock_bytes(compiled_us)
    assert assert_plan_lock_current(compiled_us, path) == path
    assert json.loads(path.read_bytes()) == plan_lock_payload(compiled_us)


def test_plan_lock_compilation_is_reproducible(compiled_us: CompiledSpecIR) -> None:
    assert plan_lock_bytes(compiled_us) == plan_lock_bytes(compiled_us)
    assert plan_lock_payload(compiled_us)["compiler_ir_abi"] == (
        compiled_us.compiler_ir_abi.to_wire()
    )


def test_semantically_valid_hand_edit_is_rejected_field_by_field(
    compiled_us: CompiledSpecIR,
) -> None:
    observed = copy.deepcopy(plan_lock_payload(compiled_us))
    observed["nodes"][0]["node_key"] = "0" * 64
    with pytest.raises(
        PlanLockError,
        match=r"/nodes/0/node_key: expected=",
    ):
        assert_plan_lock_payload_current(compiled_us, observed)


def test_noncanonical_json_bytes_are_rejected(
    compiled_us: CompiledSpecIR,
    tmp_path,
) -> None:
    path = tmp_path / "plan.lock.json"
    payload = plan_lock_payload(compiled_us)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(PlanLockError, match="canonical emitted byte form"):
        assert_plan_lock_current(compiled_us, path)


def test_duplicate_json_key_is_rejected_before_comparison(
    compiled_us: CompiledSpecIR,
    tmp_path,
) -> None:
    path = tmp_path / "plan.lock.json"
    path.write_text(
        '{"compiler_ir_abi":{},"compiler_ir_abi":{}}', encoding="utf-8"
    )
    with pytest.raises(PlanLockError, match="duplicate JSON key"):
        assert_plan_lock_current(compiled_us, path)
