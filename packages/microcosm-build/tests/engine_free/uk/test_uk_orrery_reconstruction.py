"""Declaration-only Orrery publications for UK builds from before run bindings.

Fixtures replay the files a UK dense or national build wrote before
microcosm#1144, in the order that era's ``full_build_cli`` wrote them: the
declaration, its operation inventory and run manifests, the candidate manifest
carrying the code pin, the completion record binding that manifest by digest,
for a staged build ``sha256sums.txt``, and then the close step, which appends
the staging receipts to the manifest.
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
from microcosm.build.staging_dataset import refresh_sha256sums_entry
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
NATIONAL_BUILD_ID = "uk-national-20260930T120000Z-s42"
# The completion-record layouts UK builds wrote before #1144:
# - DENSE_STALE_COMPLETION: the dense driver merged in #901 (430187fbc, merged
#   2026-09-29). ``build.json`` binds the candidate manifest as first written,
#   with STAGING_RECEIPTS_NOTE; the close step then rewrites the manifest and
#   refreshes only ``sha256sums.txt``, so that digest is stale.
# - DENSE_REFRESHED_COMPLETION: from cb0d39045 (#1057, merged 2026-09-30),
#   ``_refresh_completion`` reissues ``build.json`` over the final bytes and
#   removes the note.
# - NATIONAL_COMPLETION: the national graph path, which arrived in #1057 with
#   the refresh, so its ``build.json`` (``uk_national_build_completion``) is
#   always refreshed, its ``build_record`` entry included.
DENSE_STALE_COMPLETION = "dense_stale_completion"
DENSE_REFRESHED_COMPLETION = "dense_refreshed_completion"
NATIONAL_COMPLETION = "national_completion"
STAGING_RECEIPTS_NOTE = "staging receipts are appended after publication"


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


def file_entry(path: Path) -> dict[str, object]:
    """A completion-record entry, as ``artifact_files.file_artifact`` wrote it."""

    payload = path.read_bytes()
    return {
        "filename": path.name,
        "sha256": sha256_hex(payload),
        "size_bytes": len(payload),
    }


def output_entry(path: Path) -> dict[str, object]:
    """A candidate manifest ``outputs`` entry, as both drivers wrote it."""

    payload = path.read_bytes()
    return {"path": str(path), "sha256": sha256_hex(payload), "bytes": len(payload)}


def write_pre_1144_build(
    directory: Path,
    *,
    graph: Graph | None = None,
    git_commit: str | None = RECORDED_COMMIT,
    git_dirty: bool | None = False,
    layout: str = DENSE_REFRESHED_COMPLETION,
    staged: bool = True,
) -> Path:
    """Replay the files a UK build left in its output directory before #1144.

    The order is the driver's: the candidate manifest, ``build.json`` over it
    (noting that staging receipts follow), the staging sidecars when the
    bundle was staged, then the close step, which appends ``staging_delivery``
    and ``staged_dataset`` to the manifest whether or not it staged. Only the
    refreshed layouts then reissue ``build.json``.
    """

    national = layout == NATIONAL_COMPLETION
    release_role = "national" if national else "dense"
    release_id = f"microcosm-uk-2025-{release_role}"
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
    dataset = directory / (
        "microcosm_uk_2024_25.h5" if national else "microcosm_uk_2024_25_local.h5"
    )
    dataset.write_bytes(b"licensed population")
    outputs = {"dataset": output_entry(dataset)}
    code = {"git_commit": git_commit, "git_dirty": git_dirty}
    manifest: dict[str, object] = {
        "schema_version": 4,
        "release_role": release_role,
        "release_id": release_id,
        "created_at": "2026-09-30T12:00:00+00:00",
        **code,
        "identity": {"code": code},
        "outputs": outputs,
    }
    if national:
        build_record = directory / "build_record.json"
        build_record.write_text(json_text({"build_id": NATIONAL_BUILD_ID}))
        outputs["build_record"] = output_entry(build_record)
        manifest.update(
            build_kind="uk_national_calibrated_candidate", build_id=NATIONAL_BUILD_ID
        )
    else:
        manifest.update(build_kind="uk_rowwise_calibrated_candidate")
    manifest_path = directory / MANIFEST_FILENAME
    manifest_path.write_text(json_text(manifest))
    manifest_entry = {**file_entry(manifest_path), "note": STAGING_RECEIPTS_NOTE}
    if national:
        completion: dict[str, object] = {
            "schema_version": 1,
            "kind": "uk_national_build_completion",
            "release_role": release_role,
            "release_id": release_id,
            "build_id": NATIONAL_BUILD_ID,
            "dataset": {**file_entry(dataset), "path": str(dataset)},
            "build_record": {**file_entry(build_record), "path": str(build_record)},
            "rowwise_candidate_manifest": manifest_entry,
            "graph": {"artifacts": {}, "readback": {"passed": True}},
            "release_authorized": False,
        }
    else:
        candidate = directory / "candidate.json"
        candidate.write_bytes(canonical_json({"readback_passed": True}))
        completion = {
            "readback_passed": True,
            "kind": "uk_full_build_completion",
            "release_role": release_role,
            "candidate_manifest": file_entry(candidate),
            "rowwise_candidate_manifest": manifest_entry,
        }
    (directory / "build.json").write_bytes(canonical_json(completion))
    sums = directory / "sha256sums.txt"
    if staged:
        # ``write_sidecars``: every output, the manifest as staged and the staged
        # manifest, sorted by name.
        (directory / "staged_manifest.json").write_text('{"run_id":"fixture"}')
        listed = sorted(
            [Path(str(entry["path"])).name for entry in outputs.values()]
            + [MANIFEST_FILENAME, "staged_manifest.json"]
        )
        sums.write_text(
            "".join(
                f"{sha256_hex((directory / name).read_bytes())}  {name}\n"
                for name in listed
            )
        )
    # The close step (``_close_attempt`` / ``_close_national_attempt``).
    manifest["staging_delivery"] = {"mode": "hub" if staged else "disabled"}
    manifest["staged_dataset"] = {"status": "uploaded" if staged else "disabled"}
    rewritten = {"rowwise_candidate_manifest": manifest_path}
    if national:
        build_record.write_text(
            json_text(
                {"build_id": NATIONAL_BUILD_ID, "staging_delivery": {"mode": "hub"}}
            )
        )
        outputs["build_record"] = output_entry(build_record)
        rewritten["build_record"] = build_record
    manifest_path.write_text(json_text(manifest))
    if sums.is_file():
        for path in rewritten.values():
            refresh_sha256sums_entry(directory, path.name)
    if layout != DENSE_STALE_COMPLETION:
        # ``_refresh_completion``: drop the note and bind the final bytes.
        for role, path in rewritten.items():
            entry = dict(completion[role])  # type: ignore[arg-type]
            entry.pop("note", None)
            completion[role] = {**entry, **file_entry(path)}
        (directory / "build.json").write_bytes(canonical_json(completion))
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
    assert not {path.name for path in build.iterdir()} & {
        path.name for path in output.iterdir()
    }


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


@pytest.mark.parametrize("layout", [DENSE_REFRESHED_COMPLETION, NATIONAL_COMPLETION])
def test_candidate_manifest_disagreeing_with_a_refreshed_build_record_is_refused(
    tmp_path, layout
):
    build = write_pre_1144_build(tmp_path / "build", layout=layout, staged=False)
    completion = read_json(build / "build.json")
    completion["rowwise_candidate_manifest"]["sha256"] = "0" * 64
    (build / "build.json").write_bytes(canonical_json(completion))

    with pytest.raises(
        ValueError, match="does not match the digest recorded in build.json"
    ):
        reconstruct_uk_publication(build, tmp_path / "publication")
    assert not (tmp_path / "publication").exists()


def test_only_the_staging_receipts_note_marks_a_completion_digest_as_stale(tmp_path):
    build = write_pre_1144_build(tmp_path / "build", layout=DENSE_STALE_COMPLETION)
    completion = read_json(build / "build.json")
    completion["rowwise_candidate_manifest"]["note"] = "checked by hand"
    (build / "build.json").write_bytes(canonical_json(completion))

    with pytest.raises(
        ValueError, match="does not match the digest recorded in build.json"
    ):
        reconstruct_uk_publication(build, tmp_path / "publication")


@pytest.mark.parametrize("layout", [DENSE_REFRESHED_COMPLETION, DENSE_STALE_COMPLETION])
def test_candidate_manifest_disagreeing_with_sha256sums_is_refused(tmp_path, layout):
    # A stale completion digest is skipped, but the digest list still binds.
    build = write_pre_1144_build(tmp_path / "build", layout=layout)
    (build / "sha256sums.txt").write_text(f"{'0' * 64}  {MANIFEST_FILENAME}\n")

    with pytest.raises(
        ValueError, match="does not match the digest recorded in sha256sums.txt"
    ):
        reconstruct_uk_publication(build, tmp_path / "publication")
    assert not (tmp_path / "publication").exists()


def test_dense_build_closed_before_the_completion_refresh_is_checked_against_the_digest_list(
    tmp_path,
):
    build = write_pre_1144_build(tmp_path / "build", layout=DENSE_STALE_COMPLETION)
    stale_entry = read_json(build / "build.json")["rowwise_candidate_manifest"]
    # What #901's driver left: the note, beside the digest of the manifest as it
    # was before the close step appended the staging receipts.
    assert stale_entry["note"] == STAGING_RECEIPTS_NOTE
    assert stale_entry["sha256"] != sha256_hex((build / MANIFEST_FILENAME).read_bytes())

    report = reconstruct_uk_publication(build, tmp_path / "publication")

    marker = validate_reconstruction(read_json(tmp_path / "publication" / SCHEMA_NAME))
    assert marker["source_revision"]["git_commit"] == RECORDED_COMMIT
    manifest_input = inputs_by_name(marker)[MANIFEST_FILENAME]
    assert manifest_input["verification"] == "fallback_digest_matched"
    assert report["inputs"] == marker["inputs"]
    assert report["stale_recorded_digests"] == [
        {
            "name": MANIFEST_FILENAME,
            "recorded_in": "build.json",
            "sha256": stale_entry["sha256"],
            "bytes": stale_entry["size_bytes"],
            "note": STAGING_RECEIPTS_NOTE,
            "checked_against": ["sha256sums.txt"],
        }
    ]
    assert any(
        "build.json" in warning and "checked against sha256sums.txt" in warning
        for warning in report["warnings"]
    )


def test_stale_completion_digest_without_a_digest_list_leaves_the_manifest_unchecked(
    tmp_path,
):
    # With staging disabled the driver wrote no sidecars, but the close step
    # still appended the receipts, so no current digest covers the manifest.
    build = write_pre_1144_build(
        tmp_path / "build", layout=DENSE_STALE_COMPLETION, staged=False
    )

    report = reconstruct_uk_publication(build, tmp_path / "publication")

    marker = validate_reconstruction(read_json(tmp_path / "publication" / SCHEMA_NAME))
    manifest_input = inputs_by_name(marker)[MANIFEST_FILENAME]
    assert manifest_input["verification"] == "only_stale_digest_recorded"
    assert [item["checked_against"] for item in report["stale_recorded_digests"]] == [
        []
    ]
    assert any(
        "nothing the build recorded checks" in warning for warning in report["warnings"]
    )


def test_national_build_reconstructs_from_its_completion_record(tmp_path):
    build = write_pre_1144_build(tmp_path / "build", layout=NATIONAL_COMPLETION)
    completion = read_json(build / "build.json")
    assert completion["kind"] == "uk_national_build_completion"
    assert "note" not in completion["rowwise_candidate_manifest"]
    assert completion["build_record"]["sha256"] == sha256_hex(
        (build / "build_record.json").read_bytes()
    )

    report = reconstruct_uk_publication(build, tmp_path / "publication")

    marker = validate_reconstruction(read_json(tmp_path / "publication" / SCHEMA_NAME))
    inputs = inputs_by_name(marker)
    assert set(inputs) == {
        "build.json",
        "graph.json",
        MANIFEST_FILENAME,
        "sha256sums.txt",
    }
    assert inputs[MANIFEST_FILENAME]["verification"] == "recorded_digest_matched"
    assert report["stale_recorded_digests"] == []
    assert report["build"] == {
        "build_kind": "uk_national_calibrated_candidate",
        "release_role": "national",
        "release_id": "microcosm-uk-2025-national",
        "build_id": NATIONAL_BUILD_ID,
        "created_at": "2026-09-30T12:00:00+00:00",
    }


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
        tmp_path / "build", git_dirty=True, layout=NATIONAL_COMPLETION, staged=False
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


@pytest.mark.parametrize(
    "output_parts",
    [
        ("build",),
        ("build", "publication"),
        ("elsewhere", "..", "build", "publication"),
        ("link-to-build", "publication"),
    ],
    ids=["the_build_itself", "inside", "through_a_parent_reference", "through_a_link"],
)
def test_output_inside_the_build_directory_is_refused(tmp_path, output_parts):
    build = write_pre_1144_build(tmp_path / "build")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "link-to-build").symlink_to(build, target_is_directory=True)
    listing = sorted(path.name for path in build.iterdir())

    with pytest.raises(ValueError, match="inside the build directory"):
        reconstruct_uk_publication(build, tmp_path.joinpath(*output_parts))

    # The build is only read: nothing was written into it, not even staging.
    assert sorted(path.name for path in build.iterdir()) == listing


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


def test_cli_refuses_a_declaration_that_is_not_a_json_object(tmp_path, capsys):
    build = write_pre_1144_build(tmp_path / "build", git_commit=None, git_dirty=None)
    (build / "graph.json").write_bytes(b"[]")
    output = tmp_path / "publication"

    arguments = ["--build", str(build), "--output", str(output)]
    with pytest.raises(SystemExit) as refusal:
        main([*arguments, "--revision", OTHER_COMMIT])

    assert refusal.value.code == 2
    assert "recorded declaration is not a graph" in capsys.readouterr().err
    assert not output.exists()


def test_cli_reconstructs_a_build_closed_before_the_completion_refresh(
    tmp_path, capsys
):
    build = write_pre_1144_build(tmp_path / "build", layout=DENSE_STALE_COMPLETION)
    output = tmp_path / "publication"

    arguments = ["--build", str(build), "--output", str(output)]
    assert main([*arguments, "--revision", RECORDED_COMMIT]) == 0

    printed = capsys.readouterr().out
    assert f"{MANIFEST_FILENAME} (fallback digest matched)" in printed


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
