"""UK exports and staging order, independent of the runs application's code."""

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

import test_support.microcosm_build.uk_full_build_cli as full_build_support
from microcosm.build.uk_runtime import (
    full_build_cli,
    orrery_publication,
    rowwise_staging,
)
from microcosm.build.uk_runtime.orrery_contract import save_uk_graph_schema
from microcosm.graph import (
    ContentStore,
    Graph,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    collect_execution_evidence,
    load_run_evidence,
)
from microcosm.graph.execution_overlay import load_execution_overlay
from microcosm.graph.orrery import orrery_json_from_schema
from test_support.microcosm_build.telemetry import fake_publication_inventory

#: The execution overlay's file name, which the collector's inputs will name.
OVERLAY_NAME = "graph.execution.json"


def fixture(tmp_path, **options):
    out = tmp_path / "temporary"
    out.mkdir()
    graph = Graph(
        "uk",
        (SourceRef("uk_spine", "raw-bytes-v1"),),
        (
            Node(
                "uk.full.spine_checkpoint",
                "fixture@1",
                sources=("uk_spine",),
                structural=StructuralDelta.CREATE,
                outputs=(Owned("person", "age", "int64"),),
            ),
        ),
    )
    save_uk_graph_schema(graph, out, scope="national")
    args = Namespace(
        out=out,
        published_out=tmp_path / "out",
        attempt_evidence=tmp_path / "attempt",
        graph_store=tmp_path / "store",
        no_staging=False,
        no_staged_dataset=False,
        staging_local_only=True,
        publish_orrery=None,
        **options,
    )
    return args


def test_local_export_preserves_exact_graph_without_a_run_id_or_network(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path)
    monkeypatch.setattr(rowwise_staging, "_ACTIVE_EMITTER", None)
    (args.out / "private.h5").write_bytes(b"licensed population")
    (args.out / "raw.json").write_text('{"person_id":123}')
    result = orrery_publication.finalize_graph(args, manifest=None)
    receipt = result.receipt
    assert receipt["status"] == "skipped"
    assert receipt["recorded_execution"] is False
    preserved = args.attempt_evidence / "orrery-publication"
    inventory = json.loads((preserved / "orrery.upload.json").read_bytes())
    assert "run_id" not in inventory
    assert {f["name"] for f in inventory["files"]} == {
        "graph.orrery.json",
        "graph.schema.json",
    }
    assert (preserved / "graph.orrery.json").read_bytes() == (
        args.out / "graph.orrery.json"
    ).read_bytes()
    assert not (preserved / "private.h5").exists()
    assert not (preserved / "raw.json").exists()


def test_publication_precedes_manifest_registration_and_receipt_is_a_snapshot(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path)
    args.publish_orrery = True
    events = []

    class Emitter:
        publication_inventory = staticmethod(fake_publication_inventory)

        def publish_graph(self, directory, inventory, *, wait_seconds):
            assert not (args.out / "orrery.publication.json").exists()
            events.append("publication_attempt")
            return {
                "publication_id": inventory["publication_id"],
                "status": "pending",
                "error_code": "credential_missing",
            }

    monkeypatch.setattr(rowwise_staging, "_ACTIVE_EMITTER", Emitter())
    manifest = {"outputs": {}}
    receipt = orrery_publication.finalize_graph(args, manifest=manifest).receipt
    events.append("registered")
    assert events == ["publication_attempt", "registered"]
    assert receipt["status"] == "pending"
    names = {entry["path"].split("/")[-1] for entry in manifest["outputs"].values()}
    assert {
        "graph.orrery.json",
        "graph.schema.json",
        "orrery.publication.json",
        "orrery.upload.json",
    } <= names
    snapshot = (args.out / "orrery.publication.json").read_bytes()
    (
        args.attempt_evidence / "orrery-publication" / "publication.status.json"
    ).write_text('{"status":"published"}')
    assert (args.out / "orrery.publication.json").read_bytes() == snapshot


def fail(*args, **kwargs):
    raise RuntimeError("kernel failed")


def test_failure_path_preserves_graph_before_temporary_cleanup(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    staged = []
    monkeypatch.setattr(
        orrery_publication,
        "stage_graph_evidence",
        lambda args, output, **kwargs: staged.append(output),
    )

    with pytest.raises(RuntimeError, match="kernel failed"):
        full_build_cli._run_with_graph_publication(
            fail, None, args, telemetry=None, attempt=None, record={}
        )
    bundle = args.attempt_evidence / orrery_publication.FAILURE_BUNDLE_DIRECTORY
    assert staged == [bundle]
    assert (bundle / "graph.orrery.json").is_file()
    assert (bundle / "orrery.evidence-manifest.json").is_file()
    assert (bundle / "orrery.failure.json").is_file()
    evidence = json.loads((bundle / "orrery.evidence-manifest.json").read_bytes())
    assert {entry["path"] for entry in evidence["outputs"].values()} == {
        str(bundle / name) for name in evidence["outputs"]
    }


def test_failure_leaves_an_earlier_successful_bundle_unchanged(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    monkeypatch.setattr(
        orrery_publication, "stage_graph_evidence", lambda *a, **k: None
    )
    args.published_out.mkdir()
    previous = {
        "build.json": b'{"kind":"previous-complete-build"}',
        "graph.orrery.json": b'{"previous":"graph"}',
        "orrery.publication.json": b'{"previous":"receipt"}',
    }
    for name, payload in previous.items():
        (args.published_out / name).write_bytes(payload)

    with pytest.raises(RuntimeError, match="kernel failed"):
        full_build_cli._run_with_graph_publication(
            fail, None, args, telemetry=None, attempt=None, record={}
        )
    assert {
        path.name: path.read_bytes() for path in args.published_out.iterdir()
    } == previous


def test_interrupt_preserves_local_evidence_without_waiting_or_staging(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path)
    args.publish_orrery = True
    waits = []

    class Emitter:
        publication_inventory = staticmethod(fake_publication_inventory)

        def publish_graph(self, directory, inventory, *, wait_seconds):
            waits.append(wait_seconds)
            return {
                "publication_id": inventory["publication_id"],
                "status": "pending",
                "error_code": None,
            }

    monkeypatch.setattr(rowwise_staging, "_ACTIVE_EMITTER", Emitter())
    monkeypatch.setattr(
        orrery_publication,
        "stage_graph_evidence",
        lambda *a, **k: pytest.fail("An interrupt started Hugging Face staging"),
    )

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        full_build_cli._run_with_graph_publication(
            interrupt, None, args, telemetry=None, attempt=None, record={}
        )
    assert waits == [0]
    bundle = args.attempt_evidence / orrery_publication.FAILURE_BUNDLE_DIRECTORY
    assert (bundle / "graph.orrery.json").is_file()


def test_failure_evidence_error_does_not_replace_the_build_error(
    tmp_path, monkeypatch, capsys
):
    args = fixture(tmp_path)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(orrery_publication, "finalize_graph", broken)
    with pytest.raises(RuntimeError, match="kernel failed"):
        full_build_cli._run_with_graph_publication(
            fail, None, args, telemetry=None, attempt=None, record={}
        )
    assert "Graph failure evidence was not preserved (OSError)" in (
        capsys.readouterr().err
    )


def test_no_emitter_delegates_publication_without_a_run_id(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    args.publish_orrery = True
    monkeypatch.setattr(rowwise_staging, "_ACTIVE_EMITTER", None)
    calls = []

    def publish(directory, inventory, **options):
        assert "run_id" not in inventory
        assert options == {"wait_seconds": 0}
        calls.append(directory)
        return {"status": "pending", "error_code": "credential_missing"}

    monkeypatch.setattr(orrery_publication, "publish_graph", publish)
    receipt = orrery_publication.finalize_graph(args, manifest=None).receipt
    assert receipt["status"] == "pending"
    assert calls == [args.attempt_evidence / "orrery-publication"]


@pytest.mark.parametrize(
    "explicit,local,disabled,expected",
    [
        (None, False, False, False),
        (None, True, False, False),
        (None, False, True, False),
        (True, False, False, True),
        (True, True, True, True),
        (False, False, False, False),
    ],
)
def test_publication_is_opt_in_whatever_the_staging(
    explicit, local, disabled, expected
):
    """María's ruling of 2026-10-26: a build publishes its graph to the
    public Orrery only on ``--publish-orrery``, staged or not."""

    args = Namespace(
        publish_orrery=explicit, staging_local_only=local, no_staging=disabled
    )
    assert orrery_publication.publication_enabled(args) is expected


@pytest.mark.parametrize("active", [False, True])
def test_unavailable_inventory_preserves_graph_without_claiming_a_queued_job(
    tmp_path, monkeypatch, active
):
    args = fixture(tmp_path)
    args.publish_orrery = True
    calls = []

    def unavailable(directory, roles):
        calls.append(directory)
        return None

    def must_not_publish(*args, **options):
        pytest.fail("publication attempted without a measured inventory")

    if active:
        from types import SimpleNamespace

        emitter = SimpleNamespace(
            publication_inventory=unavailable, publish_graph=must_not_publish
        )
        monkeypatch.setattr(rowwise_staging, "_ACTIVE_EMITTER", emitter)
        monkeypatch.setattr(
            orrery_publication,
            "publication_inventory",
            lambda *a, **k: pytest.fail("started another inventory service"),
        )
    else:
        monkeypatch.setattr(rowwise_staging, "_ACTIVE_EMITTER", None)
        monkeypatch.setattr(orrery_publication, "publication_inventory", unavailable)
    monkeypatch.setattr(orrery_publication, "publish_graph", must_not_publish)
    manifest = {"outputs": {}}
    result = orrery_publication.finalize_graph(args, manifest=manifest)
    assert result.receipt["status"] == "failed"
    assert result.receipt["publication_id"] is None
    assert result.receipt["error_code"] == "local_service_unavailable"
    preserved = args.attempt_evidence / "orrery-publication"
    assert calls == [preserved]
    for name in ("graph.schema.json", "graph.orrery.json"):
        assert (preserved / name).read_bytes() == (args.out / name).read_bytes()
        assert f"orrery/{name}" in manifest["outputs"]
    assert "orrery.upload.json" not in result.files
    assert "orrery/orrery.publication.json" in manifest["outputs"]


def test_enqueue_failure_is_recorded_without_replacing_graph_exports(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path)
    args.publish_orrery = True

    class Emitter:
        publication_inventory = staticmethod(fake_publication_inventory)

        def publish_graph(self, directory, inventory, *, wait_seconds):
            return {
                "version": 1,
                "publication_id": inventory["publication_id"],
                "status": "failed",
                "error_code": "local_service_unavailable",
            }

    monkeypatch.setattr(rowwise_staging, "_ACTIVE_EMITTER", Emitter())
    result = orrery_publication.finalize_graph(args, manifest=None)
    assert result.receipt["status"] == "failed"
    assert result.receipt["error_code"] == "local_service_unavailable"
    assert result.receipt["publication_id"]
    assert (args.out / "graph.orrery.json").is_file()
    assert (args.out / "orrery.upload.json").is_file()
    assert (args.out / "orrery.evidence-manifest.json").is_file()


def synthetic_build_request(tmp_path, monkeypatch, failure=None):
    """A synthetic dense build request whose run records execution."""
    pytest.importorskip("tables")
    full_build_support.patch_certification(monkeypatch)
    args = full_build_support.arguments(tmp_path)
    return args, full_build_support.prepared(tmp_path, failure)


def synthetic_full_build(tmp_path, monkeypatch, failure=None):
    """Run a synthetic dense build; return its arguments and exit status."""
    args, build = synthetic_build_request(tmp_path, monkeypatch, failure)
    return args, full_build_support.cli.execute_full_build(build, args)


def test_execution_overlay_reprojects_to_the_exported_orrery_artifact(
    tmp_path, monkeypatch
):
    args, status = synthetic_full_build(tmp_path, monkeypatch)
    assert status == 0
    schema = json.loads((args.out / "graph.schema.json").read_bytes())
    store = ContentStore(args.out / ".graph-store", create=False)
    collected = collect_execution_evidence(
        schema,
        runs=load_run_evidence(args.out / "execution.evidence.json", store=store),
        store=store,
    )

    overlay = load_execution_overlay(args.out / OVERLAY_NAME)

    assert overlay == collected
    assert (
        orrery_json_from_schema(schema, execution=overlay).encode()
        == (args.out / "graph.orrery.json").read_bytes()
    )


def test_execution_overlay_is_registered_but_not_published(tmp_path, monkeypatch):
    args, status = synthetic_full_build(tmp_path, monkeypatch)
    assert status == 0
    overlay_path = args.out / OVERLAY_NAME
    manifest = json.loads(
        (args.out / full_build_support.cli.MANIFEST_FILENAME).read_text()
    )
    registered = manifest["outputs"][f"orrery/{OVERLAY_NAME}"]
    assert Path(registered["path"]) == overlay_path
    assert registered["sha256"] == hashlib.sha256(overlay_path.read_bytes()).hexdigest()
    assert registered["bytes"] == overlay_path.stat().st_size
    preserved = args.attempt_evidence / "orrery-publication"
    inventory = json.loads((preserved / "orrery.upload.json").read_bytes())
    assert (preserved / "graph.orrery.json").is_file()
    assert not (preserved / OVERLAY_NAME).exists()
    assert OVERLAY_NAME not in {entry["name"] for entry in inventory["files"]}


def test_refused_build_registers_its_execution_overlay_in_its_evidence_record(
    tmp_path, monkeypatch
):
    args, status = synthetic_full_build(
        tmp_path, monkeypatch, "uk_local_geography_ladder_post_calibration"
    )
    assert status == 1
    evidence = json.loads(
        (args.out / orrery_publication.EVIDENCE_MANIFEST_NAME).read_bytes()
    )
    overlay_path = args.out / OVERLAY_NAME
    assert evidence["orrery_publication"]["recorded_execution"] is True
    assert evidence["outputs"][OVERLAY_NAME]["sha256"] == (
        hashlib.sha256(overlay_path.read_bytes()).hexdigest()
    )


def test_failed_build_keeps_its_execution_overlay_in_the_failure_bundle(
    tmp_path, monkeypatch
):
    args, build = synthetic_build_request(tmp_path, monkeypatch)

    def refuse_manifest(*arguments, **options):
        raise RuntimeError("manifest projection failed")

    monkeypatch.setattr(
        full_build_support.cli, "rowwise_candidate_manifest_from_graph", refuse_manifest
    )
    with pytest.raises(RuntimeError, match="manifest projection failed"):
        full_build_support.cli.execute_full_build(build, args)
    bundle = args.attempt_evidence / orrery_publication.FAILURE_BUNDLE_DIRECTORY
    evidence = json.loads(
        (bundle / orrery_publication.EVIDENCE_MANIFEST_NAME).read_bytes()
    )
    assert evidence["orrery_publication"]["recorded_execution"] is True
    assert Path(evidence["outputs"][OVERLAY_NAME]["path"]) == bundle / OVERLAY_NAME
    assert load_execution_overlay(bundle / OVERLAY_NAME)["phases"]
    assert not (args.attempt_evidence / "orrery-publication" / OVERLAY_NAME).exists()


def test_build_without_recorded_execution_writes_no_execution_overlay(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path)
    monkeypatch.setattr(rowwise_staging, "_ACTIVE_EMITTER", None)
    manifest = {"outputs": {}}
    result = orrery_publication.finalize_graph(args, manifest=manifest)
    assert result.receipt["recorded_execution"] is False
    assert not (args.out / OVERLAY_NAME).exists()
    assert OVERLAY_NAME not in result.files
    assert f"orrery/{OVERLAY_NAME}" not in manifest["outputs"]
