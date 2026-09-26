"""Invented seven-node runner checks; never genuine source or target admission."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import test_us_national_age_counts as fixture

from microcosm.build.us_runtime import survey_age_calibration as stage
from microcosm.calibrate.registry import TargetRegistry
from microcosm.frame import WeightKind
from microcosm.graph import Capabilities, ContentStore, Determinism, graph_to_json
from microcosm.graph.artifact_edges import descriptor
from microcosm.graph.decl import ArtifactType
from microcosm.graph.keys import _capabilities_projection
from microcosm.graph.manifest import NodeReceipt, RunManifest
from microcosm.graph.store import StoreCorrupt


@pytest.mark.parametrize("resume", [False, None, "off", "forbid", {}])
def test_unsupported_resume_refuses_before_source(tmp_path, monkeypatch, resume):
    def forbidden(*args, **kwargs):
        pytest.fail("source execution reached")

    monkeypatch.setattr(
        stage.source_graph, "run_authenticated_survey_population", forbidden
    )
    with pytest.raises(ValueError, match="RESUME"):
        stage.run_survey_age_calibration(
            tmp_path,
            snapshot_root=tmp_path / "snapshot",
            store_root=tmp_path / "store",
            fraction=None,
            seed_value=0,
            target_registry=fixture.fixture_registry(),
            epochs=2,
            learning_rate=0.1,
            resume=resume,
        )
    assert not tuple(tmp_path.iterdir())


def test_genuine_registry_refuses_before_source(tmp_path, monkeypatch):
    registry = TargetRegistry(
        [
            replace(s, metadata={**s.metadata, "evidence_scope": "source_documented"})
            for s in fixture.fixture_registry()
        ],
        country="us",
    )

    def forbidden(*args, **kwargs):
        pytest.fail("source execution reached")

    monkeypatch.setattr(
        stage.source_graph, "run_authenticated_survey_population", forbidden
    )
    with pytest.raises(ValueError, match="INVENTED_AGE_REGISTRY_REQUIRED"):
        stage.run_survey_age_calibration(
            tmp_path,
            snapshot_root=tmp_path / "snapshot",
            store_root=tmp_path / "store",
            fraction=None,
            seed_value=0,
            target_registry=registry,
            epochs=2,
            learning_rate=0.1,
        )
    assert not tuple(tmp_path.iterdir())


def test_actual_seven_node_calibration_preserves_full_population(tmp_path, monkeypatch):
    from test_us_graph_survey_population import authenticated_arguments

    arguments = authenticated_arguments(tmp_path, monkeypatch)
    arguments["seed_value"] = arguments.pop("seed")
    result = stage.run_survey_age_calibration(
        **arguments,
        target_registry=fixture.fixture_registry(),
        epochs=12,
        learning_rate=0.1,
    )
    assert len(result.compiled.order) == len(result.manifest.nodes) == 7
    checked = result.successor.checked_view()
    initial = checked.previous
    current = checked.current
    assert checked.previous is initial
    assert current.frame.n("household") == 12 and current.frame.n("person") == 18
    assert current.frame.weights_for("household").kind is WeightKind.CALIBRATED
    stage.budgets._same_nonweight(initial.frame, current.frame)
    np.testing.assert_array_equal(
        current.design_weights["household"], initial.design_weights["household"]
    )
    assert not set(stage.ages._COLUMNS) & set(current.frame.table("household"))
    assert np.array_equal(
        current.frame.weights_for("household").values == 0,
        initial.frame.weights_for("household").values == 0,
    )
    assert result.release_eligible is False
    assert result.manifest.node(stage.numerical.CALIBRATION_NODE).store_hit is False
    assert result.diagnostics["verification"]["optimizer_rerun"] is False
    create_receipt = result.manifest.node(stage.source_graph.CREATE_NODE)
    preparation_payload = ContentStore(arguments["store_root"]).load_bytes(
        create_receipt.opaque_artifacts["preparation"]
    )
    assert (
        stage._sha(preparation_payload) == create_receipt.receipt["preparation_sha256"]
    )
    preparation = stage.json.loads(preparation_payload)
    assert preparation["protocol"] == "microcosm.us.survey-population-preparation.v2"
    normalization = preparation["origins"]["observed_age_normalization"]
    assert normalization["rule"]["relation"] == "numeric_identity"
    assert normalization["rule"]["temporal_adjustment"] is False
    assert normalization["sources"]["asec"]["common_age_preexisting"] is False
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    exports = {
        "manifest.json": result.manifest.to_json_bytes(),
        "compiled-graph.json": graph_to_json(result.compiled.graph).encode("utf-8"),
        "diagnostics.json": stage.canonical_json(result.diagnostics),
        "source-preparation.json": preparation_payload,
    }
    for name, payload in exports.items():
        assert len(payload) <= 2 * 1024**2
        (evidence / name).write_bytes(payload)
    (evidence / "execution.json").write_bytes(
        stage.canonical_json(
            {
                "candidate": "invented6HH-age-calibration",
                "release_eligible": False,
                "calibration": "invented_age_targets_only",
                "graph_nodes": len(result.compiled.order),
                "counts": {
                    "households": current.frame.n("household"),
                    "persons": current.frame.n("person"),
                },
                "manifest_key": result.manifest.key,
                "sha256": {
                    name: stage._sha(payload) for name, payload in exports.items()
                },
                "node_store_hits": {
                    name: row.store_hit for name, row in result.manifest.nodes.items()
                },
                "node_annotations": {
                    stage.source_graph.CREATE_NODE: {
                        "kind": "observed_age_normalization",
                        "rule": normalization["rule"],
                        "evidence_file": "source-preparation.json",
                        "evidence_sha256": stage._sha(preparation_payload),
                    }
                },
                "optimizer_history_schema_checked_only": list(
                    stage.diagnostic_check.HISTORY_FIELDS
                ),
            }
        )
    )


def test_original_budget_mutation_after_receiving_verification_refuses(
    tmp_path, monkeypatch
):
    from test_us_graph_survey_population import authenticated_arguments

    arguments = authenticated_arguments(tmp_path, monkeypatch)
    arguments["seed_value"] = arguments.pop("seed")
    issued = []
    mutated = []
    freeze = stage.budgets.freeze_survey_origin_budget.__code__
    verify = stage.budgets.verify_survey_weight_only_successor.__code__

    def observe_return(frame, event, value):
        if event != "return":
            return
        if frame.f_code is freeze and type(value) is stage.budgets.SamplingOriginBudget:
            issued.append(value)
        elif (
            frame.f_code is verify
            and type(value) is stage.budgets.SamplingOriginSuccessor
            and len(issued) == 2
        ):
            # The receiving budget has finished verification; invalidate only
            # the original capsule, leaving both populations unchanged.
            object.__setattr__(issued[0], "payload", b"changed after receiving I/O")
            mutated.append(True)

    previous_profile = sys.getprofile()
    sys.setprofile(observe_return)
    try:
        with pytest.raises(
            stage.budgets.SurveyOriginBudgetError, match="UNISSUED_OR_CHANGED"
        ):
            stage.run_survey_age_calibration(
                **arguments,
                target_registry=fixture.fixture_registry(),
                epochs=12,
                learning_rate=0.1,
            )
    finally:
        sys.setprofile(previous_profile)
    assert len(issued) == 2 and mutated == [True]


def portable_values():
    """Real receipt/manifest values, not a source or execution authority fixture."""
    row = NodeReceipt(
        key="a" * 64,
        hit=False,
        seed=4,
        kernel_ref="invented@1",
        kernel_impl_hash="b" * 64,
        capabilities=Capabilities(determinism=Determinism.DETERMINISTIC),
        receipt={"release_eligible": False},
        frame_key="c" * 64,
        weight_key="d" * 64,
        artifacts={("household", "x"): "e" * 64},
        opaque_artifacts={"counts": "f" * 64},
    )
    state = stage.json.loads(stage._node_identity(row))
    state["artifacts"] = {(e, c): k for e, c, k in state["artifacts"]}
    manifest = RunManifest(country="us", nodes={"invented": row})
    compiled = SimpleNamespace(graph=SimpleNamespace(country="us"), versions={})
    return manifest, compiled, {"invented": state}


@pytest.mark.parametrize(
    "field",
    [
        "key",
        "kernel_ref",
        "kernel_impl_hash",
        "seed",
        "frame_key",
        "weight_key",
        "artifacts",
        "opaque_artifacts",
        "receipt",
        "capabilities",
        "typed_artifacts",
        "legacy_capabilities",
        "country",
        "decisions",
    ],
)
def test_portable_receipt_check_refuses_each_late_field(field):
    manifest, compiled, expected = portable_values()
    stage._check_manifest(manifest, compiled, expected)
    row = manifest.node("invented")
    if field == "country":
        object.__setattr__(manifest, field, "xx")
    elif field == "decisions":
        object.__setattr__(manifest, field, ("invented",))
    elif field == "capabilities":
        object.__setattr__(row.capabilities, "consumes_se", True)
    else:
        changed = {
            "seed": True,
            "artifacts": {},
            "opaque_artifacts": {},
            "receipt": {"release_eligible": True},
            "typed_artifacts": {"invented": {}},
            "legacy_capabilities": True,
        }.get(field, "9" * 64)
        object.__setattr__(row, field, changed)
    with pytest.raises(ValueError):
        stage._check_manifest(manifest, compiled, expected)


def test_bounded_diagnostic_store_read_and_hash_check(tmp_path):
    store = ContentStore(tmp_path)
    key = "a" * 64
    store.put_bytes(key, b'{"invented":true}')
    assert stage._load_diagnostics(store, key) == b'{"invented":true}'
    (store.object_path(key) / "payload.bin").write_bytes(b'{"invented":null}')
    with pytest.raises(StoreCorrupt):
        stage._load_diagnostics(store, key)


def test_store_payload_limit_precedes_hashing_or_buffer_read(tmp_path, monkeypatch):
    store = ContentStore(tmp_path)
    key = "a" * 64
    store.put_bytes(key, b"12345")
    monkeypatch.setattr(stage.diagnostic_check, "MAX_DIAGNOSTIC_BYTES", 4)
    monkeypatch.setattr(
        stage, "_verified_meta", lambda *a, **k: pytest.fail("store hashing reached")
    )
    with pytest.raises(ValueError, match="DIAGNOSTIC_STORE_LIMIT"):
        stage._load_diagnostics(store, key)


def test_store_payload_symlink_refuses(tmp_path):
    store = ContentStore(tmp_path / "store")
    key = "a" * 64
    store.put_bytes(key, b"123")
    payload = store.object_path(key) / "payload.bin"
    payload.unlink()
    other = tmp_path / "other"
    other.write_bytes(b"123")
    payload.symlink_to(other)
    with pytest.raises(OSError):
        stage._load_diagnostics(store, key)


@pytest.mark.parametrize("field", ["receipt", "opaque_artifacts", "capabilities"])
def test_actual_seven_node_terminal_manifest_mutation_refuses(
    tmp_path, monkeypatch, field
):
    """Expensive actual source fixture: requires a separately approved workload."""
    from test_us_graph_survey_population import authenticated_arguments

    actual_graph = stage.run_graph
    actual_verify = stage.budgets.verify_survey_weight_only_successor
    returned = []
    changed = []

    def capture(compiled, **kwargs):
        manifest = actual_graph(compiled, **kwargs)
        if len(compiled.order) == 7:
            returned.append(manifest)
        return manifest

    def late(value):
        actual_verify(value)
        if returned:
            row = returned[-1].node(stage.numerical.CALIBRATION_NODE)
            if field == "capabilities":
                object.__setattr__(row.capabilities, "consumes_se", True)
                assert row.capabilities.consumes_se is True
            else:
                value = (
                    {**row.receipt, "release_eligible": True}
                    if field == "receipt"
                    else {}
                )
                object.__setattr__(row, field, value)
                assert getattr(row, field) == value
            changed.append(True)

    class _LateBudgets:
        """The budgets module seals its own functions in its producer check, so
        the late seam is injected through the stage's alias, never by editing
        the sealed module."""

        verify_survey_weight_only_successor = staticmethod(late)

        def __getattr__(self, name):
            return getattr(actual_budgets, name)

    actual_budgets = stage.budgets
    monkeypatch.setattr(stage, "run_graph", capture)
    monkeypatch.setattr(stage, "budgets", _LateBudgets())
    arguments = authenticated_arguments(tmp_path, monkeypatch)
    arguments["seed_value"] = arguments.pop("seed")
    try:
        with pytest.raises(ValueError, match="MANIFEST_NODE_STATE"):
            stage.run_survey_age_calibration(
                **arguments,
                target_registry=fixture.fixture_registry(),
                epochs=12,
                learning_rate=0.1,
            )
    finally:
        if field == "capabilities" and returned:
            object.__setattr__(
                returned[-1].node(stage.numerical.CALIBRATION_NODE).capabilities,
                "consumes_se",
                False,
            )
    assert changed


def _portable_array_values():
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        dependencies=("invented-alpha", "invented-beta"),
    )
    output = descriptor(
        producer="invented",
        artifact="counts",
        type_=ArtifactType("invented.counts", 1),
        producer_key="a" * 64,
        capabilities=capabilities,
    )
    row = NodeReceipt(
        key="a" * 64,
        hit=False,
        seed=4,
        kernel_ref="invented@1",
        kernel_impl_hash="b" * 64,
        capabilities=capabilities,
        typed_artifacts={"inputs": {}, "outputs": {"counts": output}},
        opaque_artifacts={"counts": output["key"]},
        receipt={
            "release_eligible": False,
            "capabilities": _capabilities_projection(capabilities),
            "declared_columns": ["age", "weight"],
            "mass": {
                "rows": [["urban", 1.5], ["rural", 2.5]],
                "nested": [{"ids": [10, 20]}],
            },
        },
    )
    # Match the country runner's detached JSON expectations, not a surrogate
    # NodeReceipt implementation or a fabricated source authority capsule.
    state = stage.json.loads(stage._node_identity(row))
    state["artifacts"] = {(e, c): k for e, c, k in state["artifacts"]}
    manifest = RunManifest(country="us", nodes={"invented": row})
    compiled = SimpleNamespace(graph=SimpleNamespace(country="us"), versions={})
    return manifest, compiled, {"invented": state}


def test_portable_manifest_accepts_real_frozen_receipt_arrays():
    manifest, compiled, expected = _portable_array_values()
    actual = manifest.node("invented")
    assert isinstance(actual.receipt["capabilities"]["dependencies"], tuple)
    assert isinstance(
        expected["invented"]["receipt"]["capabilities"]["dependencies"], list
    )
    assert isinstance(expected["invented"]["capabilities"]["dependencies"], list)
    assert isinstance(actual.receipt["mass"]["rows"][0], tuple)
    assert isinstance(expected["invented"]["receipt"]["mass"]["rows"][0], list)
    before = stage.canonical_json(expected["invented"]["receipt"])
    stage._check_manifest(manifest, compiled, expected)
    assert stage.canonical_json(expected["invented"]["receipt"]) == before


@pytest.mark.parametrize(
    "change",
    ["dependency", "array_value", "array_order", "nested_mass", "typed_schema"],
)
def test_portable_manifest_array_normalization_preserves_values(change):
    manifest, compiled, expected = _portable_array_values()
    stage._check_manifest(manifest, compiled, expected)
    row = manifest.node("invented")
    receipt = stage.json.loads(stage.canonical_json(row.receipt))
    typed = stage.json.loads(stage.canonical_json(row.typed_artifacts))
    if change == "dependency":
        receipt["capabilities"]["dependencies"][0] = "changed-dependency"
    elif change == "array_value":
        receipt["declared_columns"][0] = "changed-column"
    elif change == "array_order":
        receipt["declared_columns"].reverse()
    elif change == "nested_mass":
        receipt["mass"]["rows"][0][1] = 99.0
    else:
        typed["outputs"]["counts"]["type"]["schema_version"] = 2
    changed_row = replace(row, receipt=receipt, typed_artifacts=typed)
    changed_manifest = replace(manifest, nodes={"invented": changed_row})
    with pytest.raises(ValueError):
        stage._check_manifest(changed_manifest, compiled, expected)


def test_portable_manifest_still_checks_complete_canonical_scalar_types():
    manifest, compiled, expected = _portable_array_values()
    stage._check_manifest(manifest, compiled, expected)
    row = manifest.node("invented")
    receipt = stage.json.loads(stage.canonical_json(row.receipt))
    receipt["release_eligible"] = 0
    changed_row = replace(row, receipt=receipt)
    changed_manifest = replace(manifest, nodes={"invented": changed_row})
    # Python mapping equality equates False and 0. The complete canonical
    # comparison after _check_node_states must continue rejecting that change.
    with pytest.raises(ValueError, match="MANIFEST_CANONICAL_VALUES"):
        stage._check_manifest(changed_manifest, compiled, expected)
