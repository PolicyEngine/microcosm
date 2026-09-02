"""Portable provenance and stable manifest-identity contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.graph as graph_api
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph.decl import StructuralDelta
from microcosm.graph.kernel import Capabilities, Determinism, KernelRole, SeedSource
from microcosm.graph.manifest import Decision, NodeReceipt, PopulationView, RunManifest
from microcosm.graph.population import MassRecord


def _capabilities(role: KernelRole = KernelRole.COMPUTE) -> Capabilities:
    return Capabilities(
        determinism=Determinism.SEEDED,
        seed_source=SeedSource.EXECUTOR,
        structural=StructuralDelta.NONE,
        role=role,
        dependencies=("numpy",),
    )


def _frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2], dtype=np.int64),
            "person_household_id": np.asarray([10, 20], dtype=np.int64),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray([10, 20], dtype=np.int64),
            "size": np.asarray([1, 1], dtype=np.int64),
        }
    )
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                np.asarray([1.0, 2.0], dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["a", "b"], name="stratum"),
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


def _persisted_manifest(
    store: graph_api.ContentStore,
    *,
    tier: str = "evidence",
    gate_outcome: str = "fail",
    release_outcome: str | None = None,
) -> RunManifest:
    gate_artifact = "2" * 64
    release_artifact = "3" * 64
    for key, name in (
        (gate_artifact, "gate_verdict"),
        (release_artifact, "tier"),
    ):
        store.put_column(
            key,
            pd.Series([name], index=pd.Index([1]), dtype="string"),
            declared_dtype="string",
        )
    gate = NodeReceipt(
        key="a" * 64,
        hit=False,
        seed=1,
        kernel_ref="gate@1",
        kernel_impl_hash="b" * 64,
        capabilities=_capabilities(KernelRole.GATE),
        receipt={"outcome": gate_outcome, "evidence": {"fixture": True}},
        artifacts={("release", "gate_verdict"): gate_artifact},
    )
    release = NodeReceipt(
        key="c" * 64,
        hit=False,
        seed=2,
        kernel_ref="release@1",
        kernel_impl_hash="d" * 64,
        capabilities=_capabilities(KernelRole.RELEASE),
        receipt={
            "tier": tier,
            "outcome": release_outcome or ("pass" if tier == "certified" else "fail"),
            "gate_ancestry": ["gate"],
        },
        artifacts={("release", "tier"): release_artifact},
    )
    return RunManifest("toy", {"release": release, "gate": gate})


def test_manifest_json_round_trip_and_population_view() -> None:
    raw = _frame()
    manifest = RunManifest(
        country="toy",
        nodes={"b": _receipt("b" * 64), "a": _receipt("a" * 64)},
        decisions=(Decision("reviewer", "publish", "approved", "2026-09-01"),),
        started_at="2026-09-01T12:00:00Z",
        finished_at="2026-09-01T12:00:01Z",
        host="runner-1",
        populations={"survey": raw, "filtered": raw},
    )
    restored = RunManifest.from_json(manifest.to_json())
    assert restored == manifest
    assert restored.to_json() == manifest.to_json()
    assert manifest.nodes["a"] is manifest.node("a")
    assert manifest.receipts["a"] is manifest.receipt("a")
    assert manifest["a"].artifacts[("person", "x")] == "d" * 64

    survey = manifest.population("survey")
    filtered = manifest.population("filtered")
    assert type(survey) is type(filtered) is PopulationView
    assert isinstance(survey, Frame)
    assert manifest.population("survey") is survey
    assert type(raw) is Frame
    assert not hasattr(raw, "household")
    assert survey.person is raw.person
    assert survey.household is raw.table("household")
    assert survey.table("household") is raw.table("household")
    assert survey.weights_for("household") is raw.weights_for("household")
    assert survey.strata is raw.strata
    with pytest.raises(AttributeError, match="PopulationView.*missing"):
        _ = survey.missing

    with pytest.raises(KeyError, match="not attached"):
        restored.population("survey")
    with pytest.raises(TypeError, match="values must be Frame"):
        RunManifest(
            "toy",
            {"a": _receipt("a" * 64)},
            populations={"survey": object()},  # type: ignore[dict-item]
        )


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


def test_original_signed_decision_records_remain_mapping_compatible() -> None:
    record = {
        "name": "publish",
        "owner": "maria",
        "signature": "toy-signature-0001",
    }
    manifest = RunManifest(
        "toy",
        {"a": _receipt("a" * 64)},
        decisions=(record,),  # type: ignore[arg-type]
    )

    assert [dict(decision) for decision in manifest.decisions] == [record]
    restored = RunManifest.from_json(manifest.to_json())
    assert [dict(decision) for decision in restored.decisions] == [record]
    assert restored.key == manifest.key


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


def test_saved_manifest_persists_and_rederives_release_fields(tmp_path: Path) -> None:
    store = graph_api.ContentStore(tmp_path / "store")
    manifest = _persisted_manifest(store)
    path = tmp_path / "manifest.json"
    manifest.save(path)

    document = json.loads(path.read_text())
    assert document["schema_version"] == 1
    assert document["key"] == manifest.key
    assert document["tier"] == "evidence"
    assert document["known_failures"] == ["gate"]
    assert document["content_addressed"] == {
        "node_keys": ["a" * 64, "c" * 64],
        "decisions": [],
    }

    restored = RunManifest.load(path, store)
    assert restored.key == manifest.key
    assert restored.tier == "evidence"
    assert restored.known_failures == ("gate",)
    with pytest.raises(graph_api.NodeRejectedError, match="evidence"):
        RunManifest.load_certified(path, store)


def test_certified_loader_checks_unreached_before_tier(tmp_path: Path) -> None:
    store = graph_api.ContentStore(tmp_path / "store")
    certified = _persisted_manifest(
        store, tier="certified", gate_outcome="not_applicable"
    )
    certified_path = tmp_path / "certified.json"
    certified.save(certified_path)
    assert RunManifest.load_certified(certified_path, store).key == certified.key

    unreached = _persisted_manifest(
        store,
        tier="certified",
        gate_outcome="pass",
        release_outcome="unreached",
    )
    unreached_path = tmp_path / "unreached.json"
    unreached.save(unreached_path)
    with pytest.raises(graph_api.NodeRejectedError, match="unreached"):
        RunManifest.load_certified(unreached_path, store)


def test_loader_rederives_tier_from_gate_receipts(tmp_path: Path) -> None:
    store = graph_api.ContentStore(tmp_path / "store")
    manifest = _persisted_manifest(store, tier="certified", gate_outcome="pass")
    path = tmp_path / "manifest.json"
    manifest.save(path)
    document = json.loads(path.read_text())
    document["nodes"]["gate"]["receipt"]["outcome"] = "fail"
    document["known_failures"] = ["gate"]
    path.write_text(json.dumps(document))

    with pytest.raises(graph_api.StoreCorruptError, match=manifest.key):
        RunManifest.load_certified(path, store)


def test_loader_wraps_noncanonical_body_with_manifest_key(tmp_path: Path) -> None:
    store = graph_api.ContentStore(tmp_path / "store")
    manifest = _persisted_manifest(store)
    path = tmp_path / "manifest.json"
    manifest.save(path)
    document = json.loads(path.read_text())
    document["content_addressed"]["node_keys"][0] = float("nan")
    path.write_text(json.dumps(document))

    with pytest.raises(graph_api.StoreCorruptError, match=manifest.key):
        RunManifest.load(path, store)


def test_known_failures_includes_explicitly_rejected_nodes() -> None:
    rejected = replace(
        _receipt("a" * 64),
        capabilities=_capabilities(KernelRole.COMPUTE),
        receipt={"rejected": True},
    )

    assert RunManifest("toy", {"compute": rejected}).known_failures == ("compute",)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("tier", "certified"),
        ("schema_version", True),
        ("known_failures", []),
        ("key", "0" * 64),
        ("content_addressed", {"node_keys": [], "decisions": []}),
    ],
)
def test_load_rejects_every_persisted_projection_mismatch(
    tmp_path: Path, field: str, replacement: object
) -> None:
    store = graph_api.ContentStore(tmp_path / "store")
    manifest = _persisted_manifest(store)
    path = tmp_path / "manifest.json"
    manifest.save(path)
    document = json.loads(path.read_text())
    document[field] = replacement
    path.write_text(json.dumps(document))

    with pytest.raises(graph_api.StoreCorruptError, match=manifest.key):
        RunManifest.load(path, store)


@pytest.mark.parametrize("artifact_kind", ["column", "frame", "weight", "opaque"])
def test_load_requires_every_manifest_artifact(
    tmp_path: Path, artifact_kind: str
) -> None:
    missing_key = "9" * 64
    receipt = NodeReceipt(
        key="a" * 64,
        hit=False,
        seed=1,
        kernel_ref="compute@1",
        kernel_impl_hash="b" * 64,
        capabilities=_capabilities(KernelRole.COMPUTE),
        artifacts=({("person", "x"): missing_key} if artifact_kind == "column" else {}),
        frame_key=missing_key if artifact_kind == "frame" else None,
        weight_key=missing_key if artifact_kind == "weight" else None,
        opaque_artifacts=(
            {"diagnostic": missing_key} if artifact_kind == "opaque" else {}
        ),
    )
    manifest = RunManifest("toy", {"compute": receipt})
    path = tmp_path / "manifest.json"
    manifest.save(path)

    with pytest.raises(graph_api.StoreMissError):
        RunManifest.load(path, graph_api.ContentStore(tmp_path / "store"))

    assert json.loads(path.read_text())["tier"] is None


def test_package_exports_runtime_implementations_and_failures() -> None:
    assert graph_api.ContentStore.__module__.endswith(".store")
    assert graph_api.RunManifest is RunManifest
    assert graph_api.PopulationView is PopulationView
    assert graph_api.NodeReceipt is NodeReceipt
    assert graph_api.Decision is Decision
    assert graph_api.run_graph.__module__.endswith(".executor")
    assert graph_api.describe.__module__.endswith(".view")
    assert graph_api.StoreCorruptError.__module__.endswith(".errors")
    assert graph_api.StoreUnavailableError.__module__.endswith(".errors")
    assert graph_api.NodeRejectedError.__module__.endswith(".errors")
    assert graph_api.StoreCorrupt is graph_api.StoreCorruptError
    assert graph_api.StoreUnavailable is graph_api.StoreUnavailableError
    assert graph_api.NodeRejected is graph_api.NodeRejectedError
