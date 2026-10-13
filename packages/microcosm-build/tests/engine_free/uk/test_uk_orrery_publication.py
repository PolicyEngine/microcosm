"""UK exports and staging order, independent of the runs application's code."""

import json
from argparse import Namespace

import pytest

from microcosm.build.uk_runtime import (
    full_build_cli,
    orrery_publication,
    rowwise_staging,
)
from microcosm.build.uk_runtime.orrery_contract import save_uk_graph_schema
from microcosm.graph import Graph, Node, Owned, SourceRef, StructuralDelta


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


@pytest.mark.parametrize(
    "explicit,local,disabled,expected",
    [
        (None, False, False, True),
        (None, True, False, False),
        (None, False, True, False),
        (True, True, True, True),
        (False, False, False, False),
    ],
)
def test_publication_has_explicit_local_only_and_opt_out_controls(
    explicit, local, disabled, expected
):
    args = Namespace(
        publish_orrery=explicit, staging_local_only=local, no_staging=disabled
    )
    assert orrery_publication.publication_enabled(args) is expected
