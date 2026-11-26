"""MOCK_METADATA_ONLY full source-parent orchestration, never qualification.

The actual builder, certifier, drift checks, strict contract and publisher run.
Only profile inventory and worker/loader/H5-count/Git/Hub transport are mocked.
Dummy artifacts are not HDF5 or wheels; their hashes describe their real bytes.
No test executes a worker, model, producer Git command or publication service.
The certifier writes its own child evidence; no saved passing child is supplied.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build import release_publication as publisher
from microcosm.data import compatibility_recertification as recert
from microcosm.data import release as release_module
from microcosm.data import source_parent_replay as replay
from microcosm.data.contract import (
    US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
    validate_release_dir,
)
from test_support.microcosm_build.emitter import FakeEmitterClient
from test_support.microcosm_data.compatibility_recertification import (
    CRITICAL_METADATA,
    HOUSEHOLDS,
    PathsInfoHub,
    fake_replay,
    score_document,
    seed_published_parent,
)
from test_support.microcosm_data.release import (
    _calibration_diagnostics,
    _source_coverage_diagnostics,
)
from test_support.microcosm_data.source_parent_replay import (
    file_sha,
    make_mock_source_parent,
    mock_bytes,
    mock_current_record,
    mock_sha,
    write_json,
)
from test_support.paths import paths_for

MOCK_COMMIT = mock_sha("MOCK_NOT_A_REAL_PRODUCER_COMMIT")[:40]
MOCK_CHILD = "populace-us-2024-MOCK_METADATA_ONLY-flow-child"
MOCK_DECLARER = "MOCK_METADATA_ONLY synthetic test declarer"
MOCK_MODEL = "0.2.0+mock.metadata.only"


def _initialize_mock_source(tmp_path):
    """Complete a synthetic source fixture before freezing its original bytes."""
    root = tmp_path / "MOCK_METADATA_ONLY_source"
    root.mkdir()
    source = make_mock_source_parent(root)
    diagnostics = _calibration_diagnostics()
    for row in diagnostics["targets"]:
        if row["name"] in CRITICAL_METADATA:
            row["metadata"] = {
                **row["metadata"],
                **CRITICAL_METADATA[row["name"]],
            }
    diagnostics["build"] = {
        "post_export_scoring": {"spm": {"geography_kind": "county"}}
    }
    write_json(source.release_dir / "calibration_diagnostics.json", diagnostics)
    write_json(
        source.release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        _source_coverage_diagnostics(),
    )
    calibration = source.release_dir / "MOCK_METADATA_ONLY_calibration.npz"
    calibration.write_bytes(mock_bytes("NOT_AN_NPZ"))
    manifest = json.loads((source.release_dir / "release_manifest.json").read_text())
    manifest.update(
        data_package={"name": "microcosm-data", "version": "0.1+mock.metadata.only"},
        compatible_model_packages=[
            {"name": "policyengine-us", "specifier": "==0.1.0+mock.metadata.only"}
        ],
        compatible_core_packages=[
            {"name": "policyengine-core", "specifier": "==0.1.0+mock.metadata.only"}
        ],
    )
    manifest["build"].update(
        built_at="2026-01-01T00:00:00+00:00",
        built_with_model_package={
            "name": "policyengine-us",
            "version": "0.1.0+mock.metadata.only",
        },
        built_with_core_package={
            "name": "policyengine-core",
            "version": "0.1.0+mock.metadata.only",
        },
    )
    # The source lane places every evidence artifact beside its manifests;
    # only its selected dummy dataset lives in artifact_root.
    manifest["artifacts"] = {
        path.stem: {
            "kind": (
                "evidence"
                if path.name == "build_manifest.json"
                else "calibration"
                if path == calibration
                else "diagnostics"
            ),
            "path": path.name,
            "repo_id": recert.REPO_ID,
            "revision": source.release_dir.name,
            "sha256": file_sha(path),
        }
        for path in sorted(source.release_dir.iterdir())
        if path.name != "release_manifest.json"
    }
    dataset = next(source.artifact_root.iterdir())
    manifest["artifacts"]["mock_dataset"] = {
        "kind": "microdata",
        "path": dataset.name,
        "repo_id": recert.REPO_ID,
        "revision": source.release_dir.name,
        "sha256": file_sha(dataset),
    }
    write_json(source.release_dir / "release_manifest.json", manifest)
    profile = dict(source.profile)
    profile["release_manifest_sha256"] = file_sha(
        source.release_dir / "release_manifest.json"
    )
    originals = {
        path: path.read_bytes()
        for path in (*source.release_dir.iterdir(), dataset, source.context.ancestor_h5)
    }
    return replace(source, profile=profile, original_bytes=originals)


def _mock_packages(tmp_path, label, versions):
    """Receipt-shaped mock identities with true hashes of labeled dummy files."""
    result = {}
    for name, version in versions.items():
        wheel = tmp_path / f"MOCK_METADATA_ONLY_{label}_{name}.whl"
        wheel.write_bytes(mock_bytes(f"NOT_A_WHEEL:{label}:{name}"))
        content = tmp_path / f"MOCK_METADATA_ONLY_{label}_{name}_source.txt"
        content.write_bytes(mock_bytes(f"NOT_INSTALLED_PACKAGE_SOURCE:{label}:{name}"))
        result[name] = {
            "version": version,
            "files": 1,
            "content_sha256": file_sha(content),
            "wheel_sha256": file_sha(wheel),
        }
    return result


def _fresh_pairs(calls, before):
    fresh = calls[before:]
    assert fresh and len(fresh) % 2 == 0
    assert fresh == ["legacy", "current"] * (len(fresh) // 2)


@pytest.fixture
def mock_flow(tmp_path, monkeypatch):
    import tools.build_us_compatibility_recertification as tool

    # The producer validator lazily imports these metadata modules. Let h5py
    # finish its platform probes before the strict Git-only subprocess mock.
    importlib.import_module("microcosm.data.h5_boolean_append")
    importlib.import_module("microcosm.data.h5_enrichment")

    source = _initialize_mock_source(tmp_path)
    calls = []
    control = {"refuse_current_register": False}
    monkeypatch.setattr(replay, "_reviewed_profile", lambda: source.profile)

    def worker_transport(context, *, mode, release_dir, artifact_root):
        assert context is source.context
        assert release_dir.is_dir() and artifact_root.is_dir()
        calls.append(mode)
        if mode == "legacy":
            return {
                "legacy_actual": copy.deepcopy(source.legacy),
                "legacy_runtime": dict(source.runtime),
            }
        assert mode == "current"
        record = mock_current_record(source)
        if control["refuse_current_register"]:
            record["checks"].remove("current_stored_input_register")
        return record

    monkeypatch.setattr(replay, "_run_worker", worker_transport)
    checkout = paths_for("microcosm-build").repository
    inventory = {
        path: (checkout / path).read_bytes() for path in recert.PRODUCER_SOURCE_FILES
    }
    producer = {
        "git_commit": MOCK_COMMIT,
        "git_dirty": False,
        "source_files_sha256": {
            path: hashlib.sha256(content).hexdigest()
            for path, content in inventory.items()
        },
    }

    def git_transport(argv, **kwargs):
        # Keep the producer validator active. Answer only its native Git reads
        # using actual executing source bytes; no real producer is qualified.
        assert kwargs == {"check": True, "capture_output": True}
        if argv == ["git", "rev-parse", "--show-toplevel"]:
            output = str(checkout).encode() + b"\n"
        elif argv == ["git", "rev-parse", "--verify", MOCK_COMMIT + "^{commit}"]:
            output = MOCK_COMMIT.encode() + b"\n"
        elif len(argv) == 3 and argv[:2] == ["git", "show"]:
            commit, path = argv[2].split(":", 1)
            assert commit == MOCK_COMMIT and path in inventory
            output = inventory[path]
        else:
            pytest.fail(f"unexpected native process in metadata test: {argv!r}")
        return subprocess.CompletedProcess(argv, 0, stdout=output)

    monkeypatch.setattr(subprocess, "run", git_transport)
    monkeypatch.setattr(tool, "_producer_identity", lambda: copy.deepcopy(producer))
    baseline = _mock_packages(
        tmp_path,
        "baseline",
        {
            "policyengine-us": "0.1.0+mock.metadata.only",
            "policyengine-core": "0.1.0+mock.metadata.only",
            "spm-calculator": "0.1.0+mock.metadata.only",
        },
    )
    tested = _mock_packages(
        tmp_path,
        "tested",
        {
            "policyengine-us": MOCK_MODEL,
            "policyengine-core": "0.2.0+mock.metadata.only",
            "spm-calculator": "0.1.0+mock.metadata.only",
        },
    )
    wheels = tuple(
        tmp_path / f"MOCK_METADATA_ONLY_tested_{name}.whl" for name in tested
    )
    loader_calls = []
    loader_answer = fake_replay(packages=tested)

    def loader_transport(
        candidate_h5, *, require_wheels=False, compatibility_wheels=()
    ):
        assert Path(candidate_h5).read_bytes() == mock_bytes("dataset")
        if require_wheels:
            assert tuple(compatibility_wheels) == wheels
        loader_calls.append(bool(require_wheels))
        return loader_answer(candidate_h5)

    monkeypatch.setattr(recert, "run_recertification_loader_replay", loader_transport)

    def household_metadata_transport(path):
        assert Path(path).read_bytes() == mock_bytes("dataset")
        return HOUSEHOLDS  # Mock metadata count, never an H5 observation.

    monkeypatch.setattr(recert, "h5_household_count", household_metadata_transport)
    output = tmp_path / "MOCK_METADATA_ONLY_pending"
    before = len(calls)
    report = tool.build_candidate(
        parent_release_dir=source.release_dir,
        parent_artifact_root=source.artifact_root,
        output_dir=output,
        build_id=MOCK_CHILD,
        source_parent_replay=source.context,
    )
    _fresh_pairs(calls, before)
    pending = output / "releases" / recert.CHILD_RELEASE
    artifacts = output / "artifacts"
    scores = {}
    for label, packages, variant, withheld, value in (
        ("baseline", baseline, recert.VARIANT_AS_STORED, (), 1000.0),
        ("tested", tested, recert.VARIANT_AS_STORED, (), 1100.0),
        (
            "counterfactual",
            tested,
            recert.VARIANT_WITHOUT_NEWLY_READ,
            ("tax_unit_role_input",),
            1040.0,
        ),
    ):
        document = score_document(
            pending,
            packages=packages,
            variant=variant,
            withheld=withheld,
            value=lambda metric_id, value=value: value,
        )
        document["code"] = copy.deepcopy(producer)
        scores[label] = tmp_path / f"MOCK_METADATA_ONLY_{label}_scores.json"
        write_json(scores[label], document)
    certified = tmp_path / "MOCK_METADATA_ONLY_certified" / recert.CHILD_RELEASE
    before = len(calls)
    recert.certify_recertification(
        pending,
        certified,
        artifact_root=artifacts,
        compatibility_wheels=wheels,
        baseline_scores=scores["baseline"],
        tested_scores=scores["tested"],
        counterfactual_scores=scores["counterfactual"],
        compatible_model_specifier=f"policyengine-us=={MOCK_MODEL}",
        compatibility_claim_declared_by=MOCK_DECLARER,
        source_parent_replay=source.context,
    )
    _fresh_pairs(calls, before)
    assert loader_calls and all(loader_calls)
    hub = PathsInfoHub()
    hub._download_dir = tmp_path / "MOCK_METADATA_ONLY_hub_cache"
    seed_published_parent(hub, source.release_dir, source.artifact_root)
    monkeypatch.setattr(publisher, "_hub_api", lambda: hub)

    def repository_transport(repo):
        assert repo == recert.REPO_ID
        return sorted(hub._files("main"))

    monkeypatch.setattr(publisher, "_repository_files", repository_transport)
    emitter = FakeEmitterClient(
        modules={"huggingface"},
        collector_receipts=[
            {"status": "skipped", "error_code": "MOCK_METADATA_ONLY_NO_COLLECTOR"}
        ],
    )
    return SimpleNamespace(
        source=source,
        pending=pending,
        certified=certified,
        artifacts=artifacts,
        report=report,
        calls=calls,
        control=control,
        wheels=wheels,
        emitter=emitter,
    )


def test_mock_child_runs_complete_builder_certifier_preflight_publisher_flow(mock_flow):
    flow = mock_flow
    mapping = flow.report["parent"]["source_parent_artifact_map"]
    original = json.loads(
        (flow.source.release_dir / "release_manifest.json").read_text()
    )["artifacts"]
    assert {row["source_key"] for row in mapping} == set(original)
    assert len({row["carried_key"] for row in mapping}) == len(mapping)
    assert len({row["carried_path"] for row in mapping}) == len(mapping)
    inverse = {}
    for row in mapping:
        path = flow.certified / row["carried_path"]
        if not path.is_file():
            path = flow.artifacts / row["carried_path"]
        assert file_sha(path) == row["sha256"]
        assert row["kind"] == original[row["source_key"]]["kind"]
        inverse[row["source_key"]] = (row["source_path"], path.read_bytes())
    for key, (name, content) in inverse.items():
        source_file = flow.source.release_dir / name
        if not source_file.is_file():
            source_file = flow.source.artifact_root / name
        assert name == original[key]["path"] and content == source_file.read_bytes()
    for filename in ("release_manifest.json", "build_manifest.json"):
        assert (flow.certified / ("parent_" + filename)).read_bytes() == (
            flow.source.release_dir / filename
        ).read_bytes()
    for operation in (
        validate_release_dir,
        release_module.prepare_build_folder_release,
        publisher.publish_release_build,
    ):
        before = len(flow.calls)
        options = {
            "artifact_root": flow.artifacts,
            "compatibility_wheels": flow.wheels,
            "source_parent_replay": flow.source.context,
        }
        if operation is publisher.publish_release_build:
            options.update(client=flow.emitter, collector_wait_seconds=0)
        result = operation(flow.certified, **options)
        _fresh_pairs(flow.calls, before)
    assert result.receipt["status"] == "published"
    assert result.wanted_current is False
    assert len(flow.emitter.release_requests) == 1
    request = flow.emitter.release_requests[0]
    assert request.make_current is False and request.build == MOCK_CHILD
    names = {item.name for item in request.files}
    assert {row["carried_path"] for row in mapping} <= names
    assert replay.SOURCE_PARENT_EVIDENCE_FILE in names
    assert recert.DRIFT_FILE in names and recert.RECEIPT_FILE in names
    drift = json.loads((flow.certified / recert.DRIFT_FILE).read_text())
    assert drift["rows"] and drift["newly_read_columns"] == ["tax_unit_role_input"]
    for row in drift["rows"]:
        assert row["baseline"] == 1000.0 and row["tested"] == 1100.0
        assert row[recert.VARIANT_WITHOUT_NEWLY_READ] == 1040.0
        assert row["newly_read_inputs_effect"] == 60.0
        assert row["other_engine_changes_effect"] == 40.0
    assert not any("latest" in path for _, path in flow.emitter.hub_files)
    assert all(
        path.read_bytes() == value for path, value in flow.source.original_bytes.items()
    )


def test_saved_pass_cannot_hide_fresh_current_register_refusal(mock_flow):
    flow = mock_flow
    evidence = flow.certified / replay.SOURCE_PARENT_EVIDENCE_FILE
    saved = evidence.read_bytes()
    flow.control["refuse_current_register"] = True
    before = len(flow.calls)
    with pytest.raises(ValueError, match="current|checks"):
        publisher.publish_release_build(
            flow.certified,
            artifact_root=flow.artifacts,
            compatibility_wheels=flow.wheels,
            source_parent_replay=flow.source.context,
            client=flow.emitter,
            collector_wait_seconds=0,
        )
    _fresh_pairs(flow.calls, before)
    assert flow.emitter.release_requests == []
    assert flow.emitter.hub_files == {}
    assert evidence.read_bytes() == saved
    assert all(
        path.read_bytes() == value for path, value in flow.source.original_bytes.items()
    )
