"""Declaration-only Orrery publications for UK builds from before run bindings.

Fixtures reproduce the files a UK dense or national build wrote before
microcosm#1144: the declaration, its operation inventory and run manifests,
the candidate manifest carrying the code pin, the completion record binding
that manifest by digest and, for a staged build, ``sha256sums.txt``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from microcosm_emitter.orrery.service.queue import publication_inventory

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.graph import uk_spine_endpoint, uk_spine_graph
from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig, uk_full_graph
from microcosm.build.uk_runtime.graph_calibration import UKGraphCalibrationConfig
from microcosm.build.uk_runtime.graph_evidence import add_uk_spine_gate_nodes
from microcosm.build.uk_runtime.graph_terminal import append_uk_full_gate_nodes
from microcosm.build.uk_runtime.orrery_publication import INVENTORY_NAME
from microcosm.build.uk_runtime.orrery_reconstruction import (
    GRAPH_NAME,
    REPORT_NAME,
    SCHEMA_NAME,
    main,
    reconstruct_uk_publication,
)
from microcosm.build.uk_runtime.rowwise_cli import MANIFEST_FILENAME, json_text
from microcosm.graph import Graph, Node, Owned, SourceRef, StructuralDelta
from microcosm.graph.canonical import canonical_json
from microcosm.graph.reconstruction import (
    RECONSTRUCTION_EXTENSION,
    validate_reconstruction,
)
from microcosm.graph.schema import validate_graph_schema
from microcosm.graph.serialize import graph_to_json
from test_support.paths import paths_for

_RECONSTRUCTION_VERIFY = (
    paths_for("microcosm-build").repository
    / "tools"
    / "orrery-contract"
    / "verify-reconstruction.mjs"
)
RECORDED_COMMIT = "4bf7af406a1b2c3d4e5f60718293a4b5c6d7e8f9"
OTHER_COMMIT = "7f235941c0000000000000000000000000000000"


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pre_1144_declaration() -> Graph:
    return Graph(
        "uk",
        (SourceRef("uk_spine", "raw-bytes-v1", "Recorded FRS spine"),),
        (
            Node(
                "uk.full.spine_checkpoint",
                "fixture.checkpoint@1",
                sources=("uk_spine",),
                structural=StructuralDelta.CREATE,
                outputs=(Owned("household", "household_weight", "float64"),),
                description="Load the recorded spine checkpoint",
            ),
            Node(
                "uk.full.gates.calibrated",
                "fixture.gates@1",
                population="uk.full.spine_checkpoint",
                description="Evaluate the calibrated gate battery",
            ),
        ),
    )


def write_pre_1144_build(
    directory: Path,
    *,
    graph: Graph | None = None,
    git_commit: str | None = RECORDED_COMMIT,
    git_dirty: bool | None = False,
    release_role: str = "dense",
    staged: bool = True,
) -> Path:
    """Write the files a UK build left in its output directory before #1144."""

    directory.mkdir(parents=True)
    declared = pre_1144_declaration() if graph is None else graph
    (directory / "graph.json").write_bytes(graph_to_json(declared).encode())
    (directory / "operations.json").write_bytes(
        canonical_json([{"id": node.id} for node in declared.nodes])
    )
    (directory / "build.graph.json").write_text('{"schema_version":3}')
    (directory / "preflight.graph.json").write_text('{"schema_version":3}')
    (directory / "evidence-index.json").write_text("{}")
    (directory / ".graph-store").mkdir()
    (directory / "candidate.h5").write_bytes(b"licensed population")
    code = {"git_commit": git_commit, "git_dirty": git_dirty}
    manifest = {
        "schema_version": 4,
        "build_kind": "uk_rowwise_calibrated_candidate",
        "release_role": release_role,
        "release_id": f"microcosm-uk-2025-{release_role}",
        "created_at": "2026-10-01T12:00:00+00:00",
        **code,
        "identity": {"code": code},
        "outputs": {"dataset": {"path": "candidate.h5"}},
    }
    manifest_bytes = json_text(manifest).encode()
    (directory / MANIFEST_FILENAME).write_bytes(manifest_bytes)
    (directory / "build.json").write_bytes(
        canonical_json(
            {
                "kind": "uk_full_build_completion",
                "release_role": release_role,
                "rowwise_candidate_manifest": {
                    "filename": MANIFEST_FILENAME,
                    "sha256": sha256_hex(manifest_bytes),
                    "size_bytes": len(manifest_bytes),
                    "note": "staging receipts are appended after publication",
                },
            }
        )
    )
    if staged:
        (directory / "sha256sums.txt").write_text(
            f"{sha256_hex(b'licensed population')}  candidate.h5\n"
            f"{sha256_hex(manifest_bytes)}  {MANIFEST_FILENAME}\n"
        )
    return directory


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def inputs_by_name(marker: dict) -> dict[str, dict]:
    return {item["name"]: item for item in marker["inputs"]}


def test_dense_build_reconstructs_the_normal_publication_layout(tmp_path):
    build = write_pre_1144_build(tmp_path / "build")
    output = tmp_path / "publication"

    reconstruct_uk_publication(build, output)

    assert sorted(path.name for path in output.iterdir()) == sorted(
        [GRAPH_NAME, SCHEMA_NAME, INVENTORY_NAME, REPORT_NAME]
    )
    inventory = read_json(output / INVENTORY_NAME)
    # The emitter recomputes the inventory before it queues a directory; the
    # reconstructed one must be exactly what it would compute.
    assert inventory == publication_inventory(
        output,
        {GRAPH_NAME: "graph", SCHEMA_NAME: "schema"},
        publication_id=inventory["publication_id"],
    )
    schema = validate_graph_schema(read_json(output / SCHEMA_NAME))
    assert (output / SCHEMA_NAME).read_bytes() == canonical_json(schema)
    document = read_json(output / GRAPH_NAME)
    assert document["schemaVersion"] == "graph-explorer/v1"
    assert document["metadata"]["microcosm"]["graph_sha256"] == sha256_hex(
        (build / "graph.json").read_bytes()
    )
    assert "reconstructed declaration" in document["title"]
    assert not {"activities", "artifacts"} & document.keys()
    assert all("statuses" not in node for node in document["nodes"])
    # Nothing from the build other than the declaration's projection is copied.
    assert not (output / "candidate.h5").exists()


def test_marker_names_the_recorded_revision_and_every_input_digest(tmp_path):
    build = write_pre_1144_build(tmp_path / "build")
    output = tmp_path / "publication"

    reconstruct_uk_publication(build, output)

    schema = read_json(output / SCHEMA_NAME)
    marker = validate_reconstruction(schema)
    assert marker is not None
    document = read_json(output / GRAPH_NAME)
    assert (
        document["metadata"]["microcosm"]["extensions"][RECONSTRUCTION_EXTENSION]
        == marker
    )
    assert marker["source_revision"] == {
        "git_commit": RECORDED_COMMIT,
        "git_dirty": False,
        "origin": "recorded",
    }
    inputs = inputs_by_name(marker)
    assert set(inputs) == {
        "build.json",
        "graph.json",
        MANIFEST_FILENAME,
        "sha256sums.txt",
    }
    for name, item in inputs.items():
        payload = (build / name).read_bytes()
        assert item["sha256"] == sha256_hex(payload)
        assert item["bytes"] == len(payload)
    assert inputs["graph.json"]["role"] == "declaration"
    assert inputs["graph.json"]["verification"] == "no_recorded_digest"
    assert inputs[MANIFEST_FILENAME]["role"] == "code_pin"
    assert inputs[MANIFEST_FILENAME]["verification"] == "recorded_digest_matched"


def test_candidate_manifest_disagreeing_with_build_record_is_refused(tmp_path):
    build = write_pre_1144_build(tmp_path / "build", staged=False)
    completion = read_json(build / "build.json")
    completion["rowwise_candidate_manifest"]["sha256"] = "0" * 64
    (build / "build.json").write_bytes(canonical_json(completion))

    with pytest.raises(ValueError, match="does not match the digest recorded in"):
        reconstruct_uk_publication(build, tmp_path / "publication")
    assert not (tmp_path / "publication").exists()


def test_candidate_manifest_disagreeing_with_sha256sums_is_refused(tmp_path):
    build = write_pre_1144_build(tmp_path / "build")
    (build / "sha256sums.txt").write_text(f"{'0' * 64}  {MANIFEST_FILENAME}\n")

    with pytest.raises(ValueError, match="sha256sums.txt"):
        reconstruct_uk_publication(build, tmp_path / "publication")
    assert not (tmp_path / "publication").exists()


@pytest.mark.parametrize(
    "recorded_file", ["graph.schema.json", "execution.evidence.json"]
)
def test_build_with_recorded_orrery_evidence_is_refused(tmp_path, recorded_file):
    build = write_pre_1144_build(tmp_path / "build")
    (build / recorded_file).write_text("{}")

    with pytest.raises(ValueError, match="python -m microcosm.graph.orrery"):
        reconstruct_uk_publication(build, tmp_path / "publication")


def test_staged_bundle_without_a_declaration_is_refused_naming_what_is_missing(
    tmp_path,
):
    # A staged bundle (staged/<run_id>/) never carried graph.json before #1144.
    bundle = write_pre_1144_build(tmp_path / "bundle")
    (bundle / "graph.json").unlink()

    with pytest.raises(ValueError, match="graph.json") as refusal:
        reconstruct_uk_publication(bundle, tmp_path / "publication")
    assert MANIFEST_FILENAME in str(refusal.value)
    assert not (tmp_path / "publication").exists()


def test_caller_revision_must_match_the_recorded_revision(tmp_path):
    build = write_pre_1144_build(tmp_path / "build")

    with pytest.raises(ValueError, match="does not match the recorded revision"):
        reconstruct_uk_publication(
            build, tmp_path / "publication", revision=OTHER_COMMIT
        )
    reconstruct_uk_publication(build, tmp_path / "confirmed", revision=RECORDED_COMMIT)
    marker = validate_reconstruction(read_json(tmp_path / "confirmed" / SCHEMA_NAME))
    assert marker["source_revision"]["origin"] == "recorded"


def test_caller_revision_supplies_an_unrecorded_revision_as_asserted(tmp_path):
    build = write_pre_1144_build(tmp_path / "build", git_commit=None, git_dirty=None)

    report = reconstruct_uk_publication(
        build, tmp_path / "publication", revision=OTHER_COMMIT
    )

    marker = validate_reconstruction(read_json(tmp_path / "publication" / SCHEMA_NAME))
    assert marker["source_revision"] == {
        "git_commit": OTHER_COMMIT,
        "git_dirty": None,
        "origin": "caller_asserted",
    }
    assert any("--revision" in warning for warning in report["warnings"])


@pytest.mark.parametrize("malformed_revision", ["4bf7af406", RECORDED_COMMIT.upper()])
def test_malformed_caller_revision_is_refused(tmp_path, malformed_revision):
    build = write_pre_1144_build(tmp_path / "build")

    with pytest.raises(ValueError, match="full lowercase git commit"):
        reconstruct_uk_publication(
            build, tmp_path / "publication", revision=malformed_revision
        )


def test_build_without_any_revision_is_refused(tmp_path):
    build = write_pre_1144_build(tmp_path / "build", staged=False)
    (build / MANIFEST_FILENAME).unlink()
    (build / "build.json").unlink()

    with pytest.raises(ValueError, match="pass --revision"):
        reconstruct_uk_publication(build, tmp_path / "publication")


def test_summary_reports_unused_evidence_missing_files_and_warnings(tmp_path):
    build = write_pre_1144_build(
        tmp_path / "build", git_dirty=True, release_role="national", staged=False
    )

    report = reconstruct_uk_publication(build, tmp_path / "publication")

    assert report == read_json(tmp_path / "publication" / REPORT_NAME)
    assert report["status"] == "reconstructed"
    assert report["basis"] == "declaration_only"
    assert report["build"]["release_role"] == "national"
    assert report["declaration"]["operations"] == 2
    assert report["declaration"]["sources"] == 1
    assert [item["name"] for item in report["missing"]] == ["sha256sums.txt"]
    assert [item["name"] for item in report["unused_evidence"]] == [
        ".graph-store",
        "build.graph.json",
        "evidence-index.json",
        "operations.json",
        "preflight.graph.json",
    ]
    assert all(item["reason"] for item in report["missing"] + report["unused_evidence"])
    assert any("uncommitted changes" in warning for warning in report["warnings"])
    assert any("graph.json" in warning for warning in report["warnings"])
    published = {
        item["name"]
        for item in read_json(tmp_path / "publication" / INVENTORY_NAME)["files"]
    }
    assert REPORT_NAME not in published


def test_reconstruction_is_deterministic(tmp_path):
    build = write_pre_1144_build(tmp_path / "build")

    reconstruct_uk_publication(build, tmp_path / "first")
    reconstruct_uk_publication(build, tmp_path / "second")

    for name in (GRAPH_NAME, SCHEMA_NAME, INVENTORY_NAME, REPORT_NAME):
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()


def test_existing_output_is_refused_and_failure_leaves_no_partial_output(
    tmp_path, monkeypatch
):
    build = write_pre_1144_build(tmp_path / "build")
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        reconstruct_uk_publication(build, existing)
    assert list(existing.iterdir()) == []

    def interrupted_rename(source, destination):
        raise OSError("interrupted before the publication was moved into place")

    # Every file is written, then the move into place fails: nothing remains.
    monkeypatch.setattr(os, "rename", interrupted_rename)
    with pytest.raises(OSError, match="interrupted"):
        reconstruct_uk_publication(build, tmp_path / "publication")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["build", "existing"]


def test_declaration_behind_a_symbolic_link_is_refused(tmp_path):
    build = write_pre_1144_build(tmp_path / "build")
    elsewhere = tmp_path / "elsewhere.json"
    (build / "graph.json").rename(elsewhere)
    (build / "graph.json").symlink_to(elsewhere)

    with pytest.raises(ValueError, match="regular file"):
        reconstruct_uk_publication(build, tmp_path / "publication")


def test_cli_writes_the_publication_and_prints_a_summary(tmp_path, capsys):
    build = write_pre_1144_build(tmp_path / "build")
    output = tmp_path / "publication"

    assert main(["--build", str(build), "--output", str(output)]) == 0

    printed = capsys.readouterr().out
    assert str(output) in printed
    assert RECORDED_COMMIT in printed
    assert "declaration only" in printed
    assert (output / GRAPH_NAME).is_file()
    with pytest.raises(SystemExit) as refusal:
        main(["--build", str(build), "--output", str(output)])
    assert refusal.value.code == 2
    assert "already exists" in capsys.readouterr().err


def test_maintained_uk_dense_declaration_reconstructs_within_orrery_limits(tmp_path):
    spine = add_uk_spine_gate_nodes(
        uk_spine_graph(source_mode="split"),
        spec=load_country_spec("uk"),
        engine_identity="declaration-only",
    )
    full = uk_full_graph(
        UKFullBuildConfig(
            calibration_year=2025,
            geography_assignment="legacy",
            calibration=UKGraphCalibrationConfig(),
        ),
        spine=spine,
    )
    declared = append_uk_full_gate_nodes(
        full.graph,
        calibration=full.calibration,
        spine_stage_names=uk_spine_endpoint(spine).stage_names,
        engine_identity="declaration-only",
        review_date="2026-10-08",
    )
    build = write_pre_1144_build(tmp_path / "build", graph=declared)
    output = tmp_path / "publication"

    report = reconstruct_uk_publication(build, output)

    assert report["declaration"]["operations"] == len(declared.nodes)
    result = subprocess.run(
        ["node", str(_RECONSTRUCTION_VERIFY)],
        input=(output / GRAPH_NAME).read_text(encoding="utf-8"),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "Orrery accepted the reconstructed declaration.\n"
