"""Portable provenance and stable manifest-identity contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from microcosm.graph.decl import StructuralDelta
from microcosm.graph.kernel import Capabilities, Determinism, SeedSource
from microcosm.graph.manifest import Decision, NodeReceipt, RunManifest


def _capabilities() -> Capabilities:
    return Capabilities(
        determinism=Determinism.SEEDED,
        seed_source=SeedSource.EXECUTOR,
        structural=StructuralDelta.NONE,
        dependencies=("numpy",),
    )


def _receipt(key: str, *, hit: bool = False, wall_time: float = 0.2) -> NodeReceipt:
    return NodeReceipt(
        key=key,
        hit=hit,
        seed=42,
        kernel_ref="toy.model@1",
        kernel_impl_hash="c" * 64,
        capabilities=_capabilities(),
        receipt={"rows": 3, "labels": ["a", "b"]},
        artifacts={("person", "x"): "d" * 64},
        wall_time=wall_time,
    )


def test_manifest_json_round_trip_and_convenient_lookup() -> None:
    population = object()
    manifest = RunManifest(
        country="toy",
        nodes={"b": _receipt("b" * 64), "a": _receipt("a" * 64)},
        decisions=(Decision("reviewer", "publish", "approved", "2026-09-01"),),
        started_at="2026-09-01T12:00:00Z",
        finished_at="2026-09-01T12:00:01Z",
        host="runner-1",
        populations={"survey": population},  # type: ignore[dict-item]
    )
    restored = RunManifest.from_json(manifest.to_json())
    assert restored == manifest
    assert restored.to_json() == manifest.to_json()
    assert manifest.nodes["a"] is manifest.node("a")
    assert manifest.receipts["a"] is manifest.receipt("a")
    assert manifest["a"].artifacts[("person", "x")] == "d" * 64
    assert manifest.population("survey") is population
    with pytest.raises(KeyError, match="not attached"):
        restored.population("survey")


def test_manifest_key_excludes_every_operational_field() -> None:
    cold = RunManifest(
        country="toy",
        nodes={
            "a": _receipt("a" * 64, hit=False, wall_time=2.0),
            "b": _receipt("b" * 64, hit=False, wall_time=3.0),
        },
        started_at="first",
        finished_at="later",
        host="host-a",
    )
    warm = RunManifest(
        country="renamed descriptive label",
        nodes={
            "b": replace(
                cold.nodes["b"], hit=True, wall_time=0.01, receipt={"memoized": True}
            ),
            "a": replace(
                cold.nodes["a"], hit=True, wall_time=0.01, receipt={"memoized": True}
            ),
        },
        started_at="second",
        finished_at="soon",
        host="host-b",
    )
    assert cold.key == warm.key
    assert cold.to_json() != warm.to_json()


def test_decisions_change_manifest_identity_but_not_node_identity() -> None:
    receipt = _receipt("a" * 64)
    bare = RunManifest("toy", {"a": receipt})
    decided = RunManifest(
        "toy",
        {"a": receipt},
        decisions=(Decision("owner", "release", "yes", "2026-09-01"),),
    )
    assert bare.nodes["a"].key == decided.nodes["a"].key
    assert bare.key != decided.key


def test_decision_and_node_mapping_order_are_not_identity() -> None:
    first = Decision("a", "review", "yes", "1")
    second = Decision("b", "review", "yes", "2")
    forward = RunManifest(
        "toy",
        {"a": _receipt("a" * 64), "b": _receipt("b" * 64)},
        (first, second),
    )
    reversed_ = RunManifest(
        "toy",
        {"b": _receipt("b" * 64), "a": _receipt("a" * 64)},
        (second, first),
    )
    assert forward.key == reversed_.key


def test_serialized_content_key_detects_provenance_tampering() -> None:
    manifest = RunManifest(
        "toy",
        {"a": _receipt("a" * 64)},
        (Decision("owner", "release", "yes", "2026-09-01"),),
    )
    payload = json.loads(manifest.to_json())
    payload["decisions"][0]["text"] = "no"
    with pytest.raises(ValueError, match="content key mismatch"):
        RunManifest.from_json(json.dumps(payload))


def test_receipts_and_nested_payloads_are_immutable() -> None:
    receipt = _receipt("a" * 64)
    with pytest.raises(FrozenInstanceError):
        receipt.hit = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        receipt.artifacts[("person", "y")] = "e" * 64  # type: ignore[index]
    with pytest.raises(TypeError):
        receipt.receipt["rows"] = 4  # type: ignore[index]
    assert receipt.node_key == receipt.key
    assert receipt.store_hit == receipt.hit
    assert receipt.implementation_hash == receipt.kernel_impl_hash
    assert receipt.artifact_keys == receipt.artifacts
