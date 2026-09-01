"""Portable provenance and stable manifest-identity contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

import microcosm.graph as graph_api
from microcosm.graph.decl import StructuralDelta
from microcosm.graph.kernel import Capabilities, Determinism, SeedSource
from microcosm.graph.manifest import Decision, NodeReceipt, RunManifest
from microcosm.graph.population import MassRecord


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
        frame_key="e" * 64,
        weight_key="f" * 64,
        opaque_artifacts={"diagnostics": "1" * 64},
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
    with pytest.raises(TypeError):
        receipt.opaque_artifacts["extra"] = "2" * 64  # type: ignore[index]
    assert receipt.node_key == receipt.key
    assert receipt.store_hit == receipt.hit
    assert receipt.implementation_hash == receipt.kernel_impl_hash
    assert receipt.artifact_keys == receipt.artifacts


def test_optional_artifact_identities_round_trip_and_allow_legacy_absence() -> None:
    manifest = RunManifest("toy", {"a": _receipt("a" * 64)})
    restored = RunManifest.from_json(manifest.to_json())
    receipt = restored.nodes["a"]
    assert receipt.frame_key == "e" * 64
    assert receipt.weight_key == "f" * 64
    assert receipt.opaque_artifacts == {"diagnostics": "1" * 64}

    legacy_payload = json.loads(manifest.to_json())
    del legacy_payload["nodes"]["a"]["frame_key"]
    del legacy_payload["nodes"]["a"]["weight_key"]
    del legacy_payload["nodes"]["a"]["opaque_artifacts"]
    legacy = RunManifest.from_json(json.dumps(legacy_payload))
    assert legacy.nodes["a"].frame_key is None
    assert legacy.nodes["a"].weight_key is None
    assert legacy.nodes["a"].opaque_artifacts == {}


def test_transient_mass_ledgers_are_immutable_and_not_portable_identity() -> None:
    record = MassRecord(
        node_id="calibrate",
        operation="REWEIGHT",
        policy="preserve_total",
        before_total=3.0,
        after_total=3.0,
        before_by_stratum=(("all", 3.0),),
        after_by_stratum=(("all", 3.0),),
        entity="household",
    )
    manifest = RunManifest(
        "toy",
        {"a": _receipt("a" * 64)},
        mass_ledgers={"calibrated": (record,)},
    )
    without_ledger = RunManifest("toy", manifest.nodes)
    assert manifest.key == without_ledger.key
    assert manifest.to_json() == without_ledger.to_json()
    assert manifest.mass_ledger("calibrated") == (record,)
    with pytest.raises(TypeError):
        manifest.mass_ledgers["other"] = ()  # type: ignore[index]
    restored = RunManifest.from_json(manifest.to_json())
    with pytest.raises(KeyError, match="not attached"):
        restored.mass_ledger("calibrated")


def test_package_exports_runtime_implementations_and_failures() -> None:
    assert graph_api.ContentStore.__module__.endswith(".store")
    assert graph_api.RunManifest is RunManifest
    assert graph_api.NodeReceipt is NodeReceipt
    assert graph_api.Decision is Decision
    assert graph_api.run_graph.__module__.endswith(".executor")
    assert graph_api.describe.__module__.endswith(".view")
    assert graph_api.StoreCorrupt.__module__.endswith(".store")
    assert graph_api.StoreUnavailable.__module__.endswith(".store")
    assert graph_api.NodeRejected.__module__.endswith(".executor")
