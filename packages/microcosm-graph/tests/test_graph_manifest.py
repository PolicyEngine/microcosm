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
from microcosm.graph.canonical import canonical_json, sha256_domain
from microcosm.graph.decl import StructuralDelta
from microcosm.graph.kernel import Capabilities, Determinism, KernelRole, SeedSource
from microcosm.graph.manifest import Decision, NodeReceipt, PopulationView, RunManifest
from microcosm.graph.population import MassRecord

MANIFEST_FIXTURES = Path(__file__).parent / "fixtures" / "manifests"


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


def _frame_with_colliding_entity(name: str) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2], dtype=np.int64),
            f"person_{name}_id": np.asarray([10, 20], dtype=np.int64),
        }
    )
    group = pd.DataFrame(
        {
            f"{name}_id": np.asarray([10, 20], dtype=np.int64),
            f"{name}_value": np.asarray([100, 200], dtype=np.int64),
        }
    )
    return Frame(
        {"person": person, name: group},
        EntitySchema(group_entities=(name,)),
        {
            name: Weights(
                np.asarray([1.0, 2.0], dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["a", "b"], name="stratum"),
        metadata={"source": "collision fixture"},
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
            "requires_decisions": [],
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


@pytest.mark.parametrize("entity_name", ["metadata", "schema", "table", "entities"])
def test_population_view_entity_accessor_handles_frame_attribute_collisions(
    entity_name: str,
) -> None:
    raw = _frame_with_colliding_entity(entity_name)
    view = RunManifest(
        country="toy",
        nodes={"a": _receipt("a" * 64)},
        populations={"survey": raw},
    ).population("survey")

    assert view.entity(entity_name) is raw.table(entity_name)
    # The colliding name resolves to the Frame member, exactly as on a Frame,
    # and inherited Frame operations keep working on the view.
    assert getattr(view, entity_name) is not raw.table(entity_name)
    assert view.n("person") == raw.n("person")
    assert list(view.entities) == list(raw.entities)


def test_manifest_key_authenticates_receipts_and_excludes_run_metadata() -> None:
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
    metadata_only = RunManifest(
        country="renamed descriptive label",
        nodes=cold.nodes,
        started_at="second",
        finished_at="soon",
        host="host-b",
    )
    assert cold.key == metadata_only.key
    # A warm run that changed nothing but the run-level fields shares the key.
    run_level_only = replace(
        warm,
        nodes={
            node_id: replace(receipt, receipt=cold.nodes[node_id].receipt)
            for node_id, receipt in warm.nodes.items()
        },
    )
    assert cold.key == run_level_only.key
    assert cold.to_json() != run_level_only.to_json()
    # A changed kernel receipt is content, so it moves the key.
    assert cold.key != warm.key


def test_decisions_are_provenance_outside_manifest_and_node_identity() -> None:
    receipt = _receipt("a" * 64)
    bare = RunManifest("toy", {"a": receipt})
    decided = RunManifest(
        "toy",
        {"a": receipt},
        decisions=(Decision("owner", "release", "yes", "2026-09-01"),),
    )
    assert bare.nodes["a"].key == decided.nodes["a"].key
    assert bare.key == decided.key


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


def test_serialized_decisions_are_provenance_outside_content_key() -> None:
    manifest = RunManifest(
        "toy",
        {"a": _receipt("a" * 64)},
        (Decision("owner", "release", "yes", "2026-09-01"),),
    )
    payload = json.loads(manifest.to_json())
    payload["decisions"][0]["text"] = "no"
    restored = RunManifest.from_json(json.dumps(payload))
    assert restored.key == manifest.key
    assert restored.decisions[0].text == "no"


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


def test_optional_artifact_identities_round_trip_and_missing_fields_fail_key() -> None:
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
    with pytest.raises(ValueError, match="node 'a'"):
        RunManifest.from_json(json.dumps(legacy_payload))


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
    assert document["schema_version"] == 2
    assert document["key"] == manifest.key
    assert document["tier"] == "evidence"
    assert document["known_failures"] == ["gate"]
    body = document["content_addressed"]
    assert body["tier"] == "evidence"
    # The body is the serialized receipts less their run-level fields.
    assert body["nodes"] == {
        node_id: {k: v for k, v in receipt.items() if k not in ("hit", "wall_time")}
        for node_id, receipt in document["nodes"].items()
    }
    assert "decisions" not in body

    restored = RunManifest.load(path, store)
    assert restored.key == manifest.key
    assert restored.tier == "evidence"
    assert restored.known_failures == ("gate",)
    with pytest.raises(graph_api.NodeRejectedError, match="evidence"):
        RunManifest.load_certified(path, store)


def test_from_json_rejects_coordinated_gate_and_tier_tampering(
    tmp_path: Path,
) -> None:
    store = graph_api.ContentStore(tmp_path / "store")
    manifest = _persisted_manifest(store)
    document = json.loads(manifest.to_json())
    document["nodes"]["gate"]["receipt"]["outcome"] = "pass"
    document["nodes"]["release"]["receipt"].update(
        {"tier": "certified", "outcome": "pass"}
    )
    document["tier"] = "certified"
    document["known_failures"] = []
    tampered = json.dumps(document)

    with pytest.raises(ValueError, match="gate"):
        RunManifest.from_json(tampered)

    path = tmp_path / "tampered.json"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(graph_api.StoreCorruptError, match="gate"):
        RunManifest.load_certified(path, store)


def test_schema_v2_body_rejects_canonically_distinct_numeric_value(
    tmp_path: Path,
) -> None:
    store = graph_api.ContentStore(tmp_path / "store")
    manifest = _persisted_manifest(store)
    document = json.loads(manifest.to_json())
    document["content_addressed"]["nodes"]["gate"]["seed"] = 1.0
    document["key"] = sha256_domain(
        "manifest", canonical_json(document["content_addressed"])
    )

    with pytest.raises(ValueError, match="node 'gate'"):
        RunManifest.from_json(json.dumps(document))


def test_v1_manifest_relabel_with_fabricated_tolerance_fails_key() -> None:
    path = MANIFEST_FIXTURES / "v1_tolerance_bound_without_tolerance.json"
    document = json.loads(path.read_text())
    document["schema_version"] = 2
    receipt = document["nodes"]["fit_qrf"]
    receipt["legacy_capabilities"] = False
    receipt["capabilities"]["tolerance"] = {
        "rtol": 1e-6,
        "atol": 1e-9,
        "ulps": 2,
    }

    with pytest.raises(ValueError, match="fit_qrf"):
        RunManifest.from_json(json.dumps(document))


def test_untouched_certified_manifest_round_trips_and_loads_certified(
    tmp_path: Path,
) -> None:
    store = graph_api.ContentStore(tmp_path / "store")
    manifest = _persisted_manifest(
        store,
        tier="certified",
        gate_outcome="pass",
    )

    restored = RunManifest.from_json(manifest.to_json())
    assert restored.to_json() == manifest.to_json()
    path = tmp_path / "certified.json"
    manifest.save(path)
    assert RunManifest.load_certified(path, store).key == manifest.key


def test_v1_tolerance_bound_manifest_loads_as_legacy_cache_miss(
    tmp_path: Path,
) -> None:
    path = MANIFEST_FIXTURES / "v1_tolerance_bound_without_tolerance.json"
    raw_capabilities = json.loads(path.read_text())["nodes"]["fit_qrf"]["capabilities"]
    store = graph_api.ContentStore(tmp_path / "store")

    manifest = RunManifest.load(path, store)
    node = manifest.nodes["fit_qrf"]

    assert node.hit is False
    assert node.legacy_capabilities is True
    assert set(node.capabilities) == set(raw_capabilities)
    assert node.capabilities["numeric"] == "tolerance_bound"
    assert tuple(node.capabilities["dependencies"]) == tuple(
        raw_capabilities["dependencies"]
    )
    with pytest.raises(ValueError, match=r"legacy capabilities.*omit.*tolerance"):
        replace(
            node,
            capabilities={**raw_capabilities, "tolerance": None},
            legacy_capabilities=True,
        )
    with pytest.raises(
        graph_api.NodeRejectedError,
        match=r"unreached.*legacy_capabilities.*tolerance",
    ):
        RunManifest.load_certified(path, store)

    emitted = json.loads(manifest.to_json())
    emitted_node = emitted["nodes"]["fit_qrf"]
    assert emitted["schema_version"] == 2
    assert emitted_node["legacy_capabilities"] is True
    assert emitted_node["capabilities"] == raw_capabilities
    assert "tolerance" not in emitted_node["capabilities"]
    assert RunManifest.from_json(manifest.to_json()).to_json() == manifest.to_json()


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


def test_certified_loader_requires_authenticated_decision_requirements(
    tmp_path: Path,
) -> None:
    store = graph_api.ContentStore(tmp_path / "store")
    manifest = _persisted_manifest(store, tier="certified", gate_outcome="pass")
    path = tmp_path / "missing-requirements.json"
    document = json.loads(manifest.to_json())
    del document["nodes"]["release"]["receipt"]["requires_decisions"]
    del document["content_addressed"]["nodes"]["release"]["receipt"][
        "requires_decisions"
    ]
    document["key"] = sha256_domain(
        "manifest", canonical_json(document["content_addressed"])
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        graph_api.NodeRejectedError,
        match="authenticated requires_decisions",
    ):
        RunManifest.load_certified(path, store)


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
    document["content_addressed"]["nodes"]["gate"]["seed"] = float("nan")
    path.write_text(json.dumps(document))

    with pytest.raises(graph_api.StoreCorruptError, match=manifest.key):
        RunManifest.load(path, store)


def test_load_wraps_oversized_tolerance_integer_as_store_corrupt(
    tmp_path: Path,
) -> None:
    store = graph_api.ContentStore(tmp_path / "store")
    manifest = _persisted_manifest(store)
    path = tmp_path / "manifest.json"
    document = json.loads(manifest.to_json())
    oversized = int("9" * 400)
    for gate in (
        document["nodes"]["gate"],
        document["content_addressed"]["nodes"]["gate"],
    ):
        gate["capabilities"]["numeric"] = "tolerance_bound"
        gate["capabilities"]["tolerance"] = {
            "rtol": oversized,
            "atol": 0,
            "ulps": 0,
        }
    document["key"] = sha256_domain(
        "manifest", canonical_json(document["content_addressed"])
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        graph_api.StoreCorruptError,
        match="representable as a finite float",
    ):
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
        ("content_addressed", {"nodes": {}, "tier": None}),
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
