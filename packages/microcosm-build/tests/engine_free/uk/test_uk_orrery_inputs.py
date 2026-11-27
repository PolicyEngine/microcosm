"""UK builds, and the inputs command, write only the collector's Orrery inputs.

The collector builds the Orrery graph files from a build's graph schema,
execution overlay and the upstream files the schema links. Microcosm keeps no
projection, so no build writes a graph document, a publication inventory or a
receipt for one.
"""

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

import test_support.microcosm_build.uk_full_build_cli as full_build_support
from microcosm.build.uk_runtime import full_build_cli, orrery_inputs, rowwise_staging
from microcosm.build.uk_runtime.orrery_contract import (
    public_summary_findings,
    save_uk_graph_schema,
)
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

SCHEMA_NAME = "graph.schema.json"
OVERLAY_NAME = "graph.execution.json"
EVIDENCE_INDEX_NAME = "execution.evidence.json"
#: Files only Microcosm's own projection and the emitter's graph module needed.
RETIRED_NAMES = frozenset(
    {"graph.orrery.json", "orrery.upload.json", "orrery.publication.json"}
)
REFUSED_BUILD_CHECK = "uk_local_geography_ladder_post_calibration"


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spine_checkpoint_graph() -> Graph:
    return Graph(
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


def build_directory_with_upstream_files(tmp_path: Path) -> Path:
    """A build directory whose saved schema links two recorded checkpoint files."""
    out = tmp_path / "temporary"
    out.mkdir()
    producer = tmp_path / "spine"
    producer.mkdir()
    declaration = producer / "spine.graph.json"
    declaration.write_text('{"graph": "spine"}')
    run_manifest = producer / "spine.manifest.json"
    run_manifest.write_text('{"manifest": "spine"}')
    sidecar = producer / "spine.build.json"
    sidecar.write_text(
        json.dumps(
            {
                "graph_declaration": {
                    "path": declaration.name,
                    "sha256": sha256_hex(declaration),
                },
                "graph_manifest": {
                    "path": run_manifest.name,
                    "sha256": sha256_hex(run_manifest),
                },
            }
        )
    )
    save_uk_graph_schema(
        spine_checkpoint_graph(), out, scope="national", sidecar_path=sidecar
    )
    return out


def plain_build_directory(tmp_path: Path) -> Path:
    """A build directory whose saved schema links no upstream file."""
    out = tmp_path / "temporary"
    out.mkdir()
    save_uk_graph_schema(spine_checkpoint_graph(), out, scope="national")
    return out


def upstream_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("upstream-*.json"))


def graph_step_arguments(tmp_path: Path, out: Path) -> Namespace:
    return Namespace(
        out=out,
        published_out=tmp_path / "out",
        attempt_evidence=tmp_path / "attempt",
        graph_store=tmp_path / "store",
        no_staging=False,
        no_staged_dataset=False,
        staging_local_only=True,
    )


def synthetic_full_build(tmp_path, monkeypatch, failure=None):
    """Run a synthetic dense build that records execution; return its
    arguments and exit status."""
    pytest.importorskip("tables")
    full_build_support.patch_certification(monkeypatch)
    args = full_build_support.arguments(tmp_path)
    build = full_build_support.prepared(tmp_path, failure)
    return args, full_build_support.cli.execute_full_build(build, args)


def regress_full_build_summaries(monkeypatch) -> None:
    """Make the full build's check summaries carry their failure lines."""
    from microcosm.build.uk_runtime import orrery_contract
    from microcosm.build.uk_runtime.graph_terminal import FULL_GATE_REPORT_TYPE

    key = (FULL_GATE_REPORT_TYPE.name, FULL_GATE_REPORT_TYPE.schema_version)
    projected = orrery_contract.UK_SUMMARY_PROVIDERS[key]

    def regressed(context):
        data = projected(context)
        for row in data["tables"]["gates"]:
            row["failures"] = ["a line"]
        return data

    monkeypatch.setitem(orrery_contract.UK_SUMMARY_PROVIDERS, key, regressed)


def registered_orrery_names(outputs: dict) -> set[str]:
    return {key.split("/", 1)[1] for key in outputs if key.startswith("orrery/")}


def test_graph_step_registers_the_schema_and_its_upstream_files_only(tmp_path):
    out = build_directory_with_upstream_files(tmp_path)
    args = graph_step_arguments(tmp_path, out)
    manifest = {"outputs": {}}

    result = orrery_inputs.finalize_graph(args, manifest=manifest)

    upstream = upstream_names(out)
    assert len(upstream) == 2
    assert sorted(result.files) == sorted([SCHEMA_NAME, *upstream])
    assert registered_orrery_names(manifest["outputs"]) == set(result.files)
    assert "orrery_publication" not in manifest
    assert not any((out / name).exists() for name in RETIRED_NAMES)
    assert not (args.attempt_evidence / "orrery-publication").exists()


def test_a_finished_build_registers_the_collectors_inputs_and_no_document(
    tmp_path, monkeypatch
):
    args, status = synthetic_full_build(tmp_path, monkeypatch)
    assert status == 0

    manifest = json.loads((args.out / full_build_cli.MANIFEST_FILENAME).read_text())
    registered = registered_orrery_names(manifest["outputs"])
    assert {SCHEMA_NAME, OVERLAY_NAME, EVIDENCE_INDEX_NAME} <= registered
    assert not RETIRED_NAMES & registered
    assert "orrery_publication" not in manifest
    assert not any((args.out / name).exists() for name in RETIRED_NAMES)
    assert not (args.attempt_evidence / "orrery-publication").exists()


def test_a_refused_builds_evidence_record_names_no_document(tmp_path, monkeypatch):
    args, status = synthetic_full_build(tmp_path, monkeypatch, REFUSED_BUILD_CHECK)
    assert status == 1

    evidence = json.loads(
        (args.out / orrery_inputs.EVIDENCE_MANIFEST_NAME).read_bytes()
    )
    assert {SCHEMA_NAME, OVERLAY_NAME} <= set(evidence["outputs"])
    assert not RETIRED_NAMES & set(evidence["outputs"])
    assert "orrery_publication" not in evidence


def test_a_failed_builds_evidence_bundle_holds_no_document(tmp_path):
    out = build_directory_with_upstream_files(tmp_path)
    args = graph_step_arguments(tmp_path, out)

    def fail(*arguments, **options):
        raise RuntimeError("kernel failed")

    with pytest.raises(RuntimeError, match="kernel failed"):
        full_build_cli._run_with_graph_inputs(
            fail, None, args, telemetry=None, attempt=None, record={}
        )

    bundle = args.attempt_evidence / orrery_inputs.FAILURE_BUNDLE_DIRECTORY
    names = {path.name for path in bundle.iterdir()}
    assert {SCHEMA_NAME, *upstream_names(out)} <= names
    assert not RETIRED_NAMES & names
    evidence = json.loads((bundle / orrery_inputs.EVIDENCE_MANIFEST_NAME).read_bytes())
    assert "orrery_publication" not in evidence


def test_unpublishable_summaries_leave_no_inputs_and_the_build_says_so(
    tmp_path, monkeypatch, capsys
):
    regress_full_build_summaries(monkeypatch)

    args, status = synthetic_full_build(tmp_path, monkeypatch)

    assert status == 0
    manifest = json.loads((args.out / full_build_cli.MANIFEST_FILENAME).read_text())
    assert not {SCHEMA_NAME, OVERLAY_NAME} & registered_orrery_names(
        manifest["outputs"]
    )
    assert not (args.out / OVERLAY_NAME).exists()
    assert "hands the collector no Orrery inputs" in capsys.readouterr().err


def test_inputs_command_writes_what_a_finished_build_hands_over(
    tmp_path, monkeypatch, capsys
):
    args, status = synthetic_full_build(tmp_path, monkeypatch)
    assert status == 0
    output = tmp_path / "inputs"

    exit_code = orrery_inputs.main(
        [
            "--schema",
            str(args.out / SCHEMA_NAME),
            "--evidence",
            str(args.out / EVIDENCE_INDEX_NAME),
            "--store",
            str(args.out / ".graph-store"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.is_dir()
    expected = sorted([SCHEMA_NAME, OVERLAY_NAME, *upstream_names(args.out)])
    assert sorted(path.name for path in output.iterdir()) == expected
    for name in expected:
        assert (output / name).read_bytes() == (args.out / name).read_bytes()
    assert str(output.resolve()) in capsys.readouterr().out


def test_inputs_command_without_evidence_writes_the_schema_and_upstream_files(
    tmp_path,
):
    out = build_directory_with_upstream_files(tmp_path)
    output = tmp_path / "inputs"

    exit_code = orrery_inputs.main(
        ["--schema", str(out / SCHEMA_NAME), "--output", str(output)]
    )

    assert exit_code == 0
    assert output.is_dir()
    expected = sorted([SCHEMA_NAME, *upstream_names(out)])
    assert len(expected) == 3
    assert sorted(path.name for path in output.iterdir()) == expected
    for name in expected:
        assert (output / name).read_bytes() == (out / name).read_bytes()


def test_inputs_command_refuses_an_existing_output(tmp_path, capsys):
    out = build_directory_with_upstream_files(tmp_path)
    output = tmp_path / "inputs"
    output.mkdir()

    with pytest.raises(SystemExit) as refusal:
        orrery_inputs.main(
            ["--schema", str(out / SCHEMA_NAME), "--output", str(output)]
        )

    assert refusal.value.code == 2
    assert "already exists" in capsys.readouterr().err
    assert list(output.iterdir()) == []


def test_inputs_command_needs_the_evidence_and_its_store_together(tmp_path, capsys):
    out = build_directory_with_upstream_files(tmp_path)
    output = tmp_path / "inputs"

    with pytest.raises(SystemExit) as refusal:
        orrery_inputs.main(
            [
                "--schema",
                str(out / SCHEMA_NAME),
                "--evidence",
                str(out / EVIDENCE_INDEX_NAME),
                "--output",
                str(output),
            ]
        )

    assert refusal.value.code == 2
    assert "together" in capsys.readouterr().err
    assert not output.exists()


def test_inputs_command_refuses_an_upstream_file_whose_bytes_changed(tmp_path, capsys):
    out = build_directory_with_upstream_files(tmp_path)
    (out / upstream_names(out)[0]).write_text('{"changed": true}')
    output = tmp_path / "inputs"

    with pytest.raises(SystemExit) as refusal:
        orrery_inputs.main(
            ["--schema", str(out / SCHEMA_NAME), "--output", str(output)]
        )

    assert refusal.value.code == 2
    assert "digest" in capsys.readouterr().err
    assert not output.exists()


def test_inputs_command_refuses_summaries_a_build_would_not_hand_over(
    tmp_path, monkeypatch, capsys
):
    regress_full_build_summaries(monkeypatch)
    args, status = synthetic_full_build(tmp_path, monkeypatch)
    assert status == 0
    capsys.readouterr()
    output = tmp_path / "inputs"

    with pytest.raises(SystemExit) as refusal:
        orrery_inputs.main(
            [
                "--schema",
                str(args.out / SCHEMA_NAME),
                "--evidence",
                str(args.out / EVIDENCE_INDEX_NAME),
                "--store",
                str(args.out / ".graph-store"),
                "--output",
                str(output),
            ]
        )

    assert refusal.value.code == 2
    assert "summaries" in capsys.readouterr().err
    assert not output.exists()


def fail(*arguments, **options):
    raise RuntimeError("kernel failed")


def test_failure_path_preserves_graph_inputs_before_temporary_cleanup(
    tmp_path, monkeypatch
):
    args = graph_step_arguments(tmp_path, plain_build_directory(tmp_path))
    staged = []
    monkeypatch.setattr(
        orrery_inputs,
        "stage_graph_evidence",
        lambda args, output, **kwargs: staged.append(output),
    )

    with pytest.raises(RuntimeError, match="kernel failed"):
        full_build_cli._run_with_graph_inputs(
            fail, None, args, telemetry=None, attempt=None, record={}
        )
    bundle = args.attempt_evidence / orrery_inputs.FAILURE_BUNDLE_DIRECTORY
    assert staged == [bundle]
    assert (bundle / SCHEMA_NAME).is_file()
    assert (bundle / orrery_inputs.EVIDENCE_MANIFEST_NAME).is_file()
    assert (bundle / orrery_inputs.FAILURE_MARKER_NAME).is_file()
    evidence = json.loads((bundle / orrery_inputs.EVIDENCE_MANIFEST_NAME).read_bytes())
    assert {entry["path"] for entry in evidence["outputs"].values()} == {
        str(bundle / name) for name in evidence["outputs"]
    }


def test_failure_leaves_an_earlier_successful_bundle_unchanged(tmp_path, monkeypatch):
    args = graph_step_arguments(tmp_path, plain_build_directory(tmp_path))
    monkeypatch.setattr(orrery_inputs, "stage_graph_evidence", lambda *a, **k: None)
    args.published_out.mkdir()
    previous = {
        "build.json": b'{"kind":"previous-complete-build"}',
        "graph.orrery.json": b'{"previous":"graph"}',
        "orrery.publication.json": b'{"previous":"receipt"}',
    }
    for name, payload in previous.items():
        (args.published_out / name).write_bytes(payload)

    with pytest.raises(RuntimeError, match="kernel failed"):
        full_build_cli._run_with_graph_inputs(
            fail, None, args, telemetry=None, attempt=None, record={}
        )
    assert {
        path.name: path.read_bytes() for path in args.published_out.iterdir()
    } == previous


def test_interrupt_preserves_local_evidence_without_staging(tmp_path, monkeypatch):
    args = graph_step_arguments(tmp_path, plain_build_directory(tmp_path))
    monkeypatch.setattr(
        orrery_inputs,
        "stage_graph_evidence",
        lambda *a, **k: pytest.fail("An interrupt started Hugging Face staging"),
    )

    def interrupt(*arguments, **options):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        full_build_cli._run_with_graph_inputs(
            interrupt, None, args, telemetry=None, attempt=None, record={}
        )
    bundle = args.attempt_evidence / orrery_inputs.FAILURE_BUNDLE_DIRECTORY
    assert (bundle / SCHEMA_NAME).is_file()


def test_failure_evidence_error_does_not_replace_the_build_error(
    tmp_path, monkeypatch, capsys
):
    args = graph_step_arguments(tmp_path, plain_build_directory(tmp_path))

    def broken(*arguments, **options):
        raise OSError("disk full")

    monkeypatch.setattr(orrery_inputs, "finalize_graph", broken)
    with pytest.raises(RuntimeError, match="kernel failed"):
        full_build_cli._run_with_graph_inputs(
            fail, None, args, telemetry=None, attempt=None, record={}
        )
    assert "Graph failure evidence was not preserved (OSError)" in (
        capsys.readouterr().err
    )


def test_a_graph_step_failure_registers_nothing_and_says_so(tmp_path, capsys):
    out = build_directory_with_upstream_files(tmp_path)
    (out / upstream_names(out)[0]).write_text('{"changed": true}')
    args = graph_step_arguments(tmp_path, out)
    manifest = {"outputs": {}}

    result = orrery_inputs.finalize_graph(args, manifest=manifest)

    assert result.files == ()
    assert manifest["outputs"] == {}
    assert "Orrery inputs were not written (ValueError)" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--publish-orrery", "--no-publish-orrery"])
def test_the_retired_orrery_upload_flags_are_refused(tmp_path, capsys, flag):
    """``--publish-orrery`` asked for an upload to the public Orrery, which
    builds no longer make: the collector builds the graph files from the
    inputs the staged bundle submits. Neither it nor ``--no-publish-orrery``
    is an option any more."""

    with pytest.raises(SystemExit) as refused:
        full_build_support.arguments(tmp_path, flag)
    assert refused.value.code == 2
    assert f"unrecognized arguments: {flag}" in capsys.readouterr().err


def test_a_build_request_has_no_orrery_upload_setting(tmp_path):
    assert not hasattr(full_build_support.arguments(tmp_path), "publish_orrery")


def test_a_refused_build_hands_over_check_verdicts_without_their_failure_lines(
    tmp_path, monkeypatch
):
    """Vahid's review of #1396, M1: a failure line can state a row's figures
    or an area's support, so the overlay's summaries count the lines."""
    args, status = synthetic_full_build(tmp_path, monkeypatch, REFUSED_BUILD_CHECK)
    assert status == 1

    summaries = load_execution_overlay(args.out / OVERLAY_NAME)["summaries"]
    checks = [
        row
        for summary in summaries
        for row in summary["data"].get("tables", {}).get("gates", [])
    ]
    failed = [row for row in checks if row["status"] == "failed"]
    assert failed and all(row["failure_count"] > 0 for row in failed)
    assert not any("failures" in row for row in checks)
    assert public_summary_findings(summaries) == []


def test_execution_overlay_file_equals_the_collected_overlay(tmp_path, monkeypatch):
    args, status = synthetic_full_build(tmp_path, monkeypatch)
    assert status == 0
    schema = json.loads((args.out / SCHEMA_NAME).read_bytes())
    store = ContentStore(args.out / ".graph-store", create=False)
    collected = collect_execution_evidence(
        schema,
        runs=load_run_evidence(args.out / EVIDENCE_INDEX_NAME, store=store),
        store=store,
    )

    assert load_execution_overlay(args.out / OVERLAY_NAME) == collected


def test_execution_overlay_is_registered_in_the_build_manifest(tmp_path, monkeypatch):
    args, status = synthetic_full_build(tmp_path, monkeypatch)
    assert status == 0
    overlay_path = args.out / OVERLAY_NAME
    manifest = json.loads((args.out / full_build_cli.MANIFEST_FILENAME).read_text())
    registered = manifest["outputs"][f"orrery/{OVERLAY_NAME}"]
    assert Path(registered["path"]) == overlay_path
    assert registered["sha256"] == sha256_hex(overlay_path)
    assert registered["bytes"] == overlay_path.stat().st_size


def test_refused_build_registers_its_execution_overlay_in_its_evidence_record(
    tmp_path, monkeypatch
):
    args, status = synthetic_full_build(tmp_path, monkeypatch, REFUSED_BUILD_CHECK)
    assert status == 1
    evidence = json.loads(
        (args.out / orrery_inputs.EVIDENCE_MANIFEST_NAME).read_bytes()
    )
    assert evidence["outputs"][OVERLAY_NAME]["sha256"] == sha256_hex(
        args.out / OVERLAY_NAME
    )


def test_failed_build_keeps_its_execution_overlay_in_the_failure_bundle(
    tmp_path, monkeypatch
):
    pytest.importorskip("tables")
    full_build_support.patch_certification(monkeypatch)
    args = full_build_support.arguments(tmp_path)
    build = full_build_support.prepared(tmp_path)

    def refuse_manifest(*arguments, **options):
        raise RuntimeError("manifest projection failed")

    monkeypatch.setattr(
        full_build_support.cli, "rowwise_candidate_manifest_from_graph", refuse_manifest
    )
    with pytest.raises(RuntimeError, match="manifest projection failed"):
        full_build_support.cli.execute_full_build(build, args)
    bundle = args.attempt_evidence / orrery_inputs.FAILURE_BUNDLE_DIRECTORY
    evidence = json.loads((bundle / orrery_inputs.EVIDENCE_MANIFEST_NAME).read_bytes())
    assert Path(evidence["outputs"][OVERLAY_NAME]["path"]) == bundle / OVERLAY_NAME
    assert load_execution_overlay(bundle / OVERLAY_NAME)["phases"]


def test_build_without_recorded_execution_writes_no_execution_overlay(tmp_path):
    args = graph_step_arguments(tmp_path, plain_build_directory(tmp_path))
    manifest = {"outputs": {}}

    result = orrery_inputs.finalize_graph(args, manifest=manifest)

    assert result.files == (SCHEMA_NAME,)
    assert not (args.out / OVERLAY_NAME).exists()
    assert f"orrery/{OVERLAY_NAME}" not in manifest["outputs"]


def test_failure_evidence_is_handed_to_the_builds_emitter_with_each_files_role(
    tmp_path, monkeypatch
):
    """A failed build's graph evidence is staged like a dataset bundle: the
    build's emitter commits it under ``staged/<run_id>/``, and the build
    makes no Hub call of its own."""
    from microcosm_emitter_interface import CollectorProvenance

    from microcosm.build.telemetry_emitter import TelemetryRun
    from test_support.microcosm_build.emitter import FakeEmitterClient
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter
    from test_support.microcosm_build.uk_rowwise_candidate import _FakeHub

    output = tmp_path / orrery_inputs.FAILURE_BUNDLE_DIRECTORY
    output.mkdir()
    schema = output / SCHEMA_NAME
    schema.write_text('{"nodes": []}')
    evidence = {
        "schema_version": 1,
        "build_kind": "uk_graph_evidence",
        "releasable": False,
        "outputs": {
            schema.name: {
                "path": str(schema),
                "sha256": sha256_hex(schema),
                "bytes": schema.stat().st_size,
            }
        },
    }
    (output / orrery_inputs.EVIDENCE_MANIFEST_NAME).write_text(json.dumps(evidence))
    hub = _FakeHub()
    client = FakeEmitterClient(hub_files=hub.files, hub_api=hub)
    run = TelemetryRun(run_id="uk-attempt", country_code="GB", pipeline="uk")
    monkeypatch.setattr(
        rowwise_staging, "_ACTIVE_EMITTER", FakeTelemetryEmitter(run, client=client)
    )
    args = Namespace(
        no_staging=False,
        no_staged_dataset=False,
        staging_local_only=False,
        staged_dataset_repo_id="policyengine/populace-uk-private",
    )

    delivery = orrery_inputs.stage_graph_evidence(args, output, run_id="uk-attempt")

    assert delivery["status"] == "uploaded"
    assert delivery["revision"] == hub.sha
    (request,) = client.bundle_requests
    assert {item.name: item.role for item in request.files} == {
        SCHEMA_NAME: "orrery_schema",
        orrery_inputs.EVIDENCE_MANIFEST_NAME: "build_manifest",
        "staged_manifest.json": "staged_manifest",
        "sha256sums.txt": "checksums",
    }
    # The bundle names the build for the collector, so its Orrery inputs are
    # submitted once the commit lands; a failed build is no release candidate.
    assert request.collector == CollectorProvenance(
        country_code="UK", build_id="uk-attempt", licensed_data=True, release=None
    )
    assert hub.commits[0]["paths"] == sorted(
        f"staged/uk-attempt/{item.name}" for item in request.files
    )


def break_presentation_contract(out: Path) -> None:
    """Make the saved schema's presentation contract name an unknown operation."""
    schema_path = out / SCHEMA_NAME
    schema = json.loads(schema_path.read_bytes())
    groups = schema["extensions"]["microcosm.presentation"]["groups"]
    group = next(group for group in groups if "operations" in group)
    group["operations"].append("uk.no_such_operation")
    schema_path.write_text(json.dumps(schema))


def test_graph_step_hands_over_nothing_for_an_invalid_presentation_contract(
    tmp_path, capsys
):
    """The collector's projection refuses an invalid contract, so the build
    must not hand it over."""
    out = plain_build_directory(tmp_path)
    break_presentation_contract(out)
    manifest = {"outputs": {}}

    result = orrery_inputs.finalize_graph(
        graph_step_arguments(tmp_path, out), manifest=manifest
    )

    assert result.files == ()
    assert manifest["outputs"] == {}
    assert "Orrery inputs were not written (ValueError)" in capsys.readouterr().err


def test_inputs_command_refuses_an_invalid_presentation_contract(tmp_path, capsys):
    out = plain_build_directory(tmp_path)
    break_presentation_contract(out)
    output = tmp_path / "inputs"

    with pytest.raises(SystemExit) as refusal:
        orrery_inputs.main(
            ["--schema", str(out / SCHEMA_NAME), "--output", str(output)]
        )

    assert refusal.value.code == 2
    assert "unknown group member" in capsys.readouterr().err
    assert not output.exists()


def test_a_graph_step_that_registers_nothing_drops_an_earlier_evidence_record(
    tmp_path,
):
    """A refused build may reuse an output directory; its evidence record
    must never stand in for one this attempt did not write."""
    out = plain_build_directory(tmp_path)
    break_presentation_contract(out)
    args = graph_step_arguments(tmp_path, out)
    args.published_out.mkdir()
    earlier = args.published_out / orrery_inputs.EVIDENCE_MANIFEST_NAME
    earlier.write_text('{"outputs": {"graph.schema.json": {}}}')

    result = orrery_inputs.finalize_graph(args, manifest=None)

    assert result.files == ()
    assert not earlier.exists()
    assert not (out / orrery_inputs.EVIDENCE_MANIFEST_NAME).exists()


def test_inputs_command_refuses_a_schema_with_a_duplicate_key(tmp_path, capsys):
    out = plain_build_directory(tmp_path)
    schema_path = out / SCHEMA_NAME
    schema_path.write_text(
        schema_path.read_text().replace("{", '{"protocol": "duplicate", ', 1)
    )
    output = tmp_path / "inputs"

    with pytest.raises(SystemExit) as refusal:
        orrery_inputs.main(["--schema", str(schema_path), "--output", str(output)])

    assert refusal.value.code == 2
    assert "Duplicate JSON key" in capsys.readouterr().err
    assert not output.exists()


def test_inputs_command_refuses_a_schema_behind_a_symbolic_link(tmp_path):
    out = plain_build_directory(tmp_path)
    link = tmp_path / "linked.schema.json"
    link.symlink_to(out / SCHEMA_NAME)
    output = tmp_path / "inputs"

    with pytest.raises(SystemExit) as refusal:
        orrery_inputs.main(["--schema", str(link), "--output", str(output)])

    assert refusal.value.code == 2
    assert not output.exists()


def test_inputs_command_refuses_a_missing_upstream_file(tmp_path, capsys):
    out = build_directory_with_upstream_files(tmp_path)
    (out / upstream_names(out)[0]).unlink()
    output = tmp_path / "inputs"

    with pytest.raises(SystemExit) as refusal:
        orrery_inputs.main(
            ["--schema", str(out / SCHEMA_NAME), "--output", str(output)]
        )

    assert refusal.value.code == 2
    assert "not a regular file" in capsys.readouterr().err
    assert not output.exists()
    assert not any(tmp_path.glob(".inputs.*.partial"))


def test_inputs_command_refuses_an_upstream_file_behind_a_symbolic_link(
    tmp_path, capsys
):
    """An upstream file is read once, as the schema is: never through a link."""
    out = build_directory_with_upstream_files(tmp_path)
    name = upstream_names(out)[0]
    elsewhere = tmp_path / "elsewhere.json"
    (out / name).rename(elsewhere)
    (out / name).symlink_to(elsewhere)
    output = tmp_path / "inputs"

    with pytest.raises(SystemExit) as refusal:
        orrery_inputs.main(
            ["--schema", str(out / SCHEMA_NAME), "--output", str(output)]
        )

    assert refusal.value.code == 2
    assert name in capsys.readouterr().err
    assert not output.exists()
