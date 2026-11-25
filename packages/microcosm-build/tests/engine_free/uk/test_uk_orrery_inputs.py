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
from microcosm.build.uk_runtime import full_build_cli, orrery_inputs
from microcosm.build.uk_runtime.orrery_contract import save_uk_graph_schema
from microcosm.graph import Graph, Node, Owned, SourceRef, StructuralDelta

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
        publish_orrery=None,
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
        full_build_cli._run_with_graph_publication(
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
            str(args.graph_store),
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
                str(args.graph_store),
                "--output",
                str(output),
            ]
        )

    assert refusal.value.code == 2
    assert "summaries" in capsys.readouterr().err
    assert not output.exists()
