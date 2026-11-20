"""US source-parent orchestration with explicit MOCK_METADATA_ONLY callouts.

No fixture here contains a real dataset, wheel, original interpreter or original
source checkout. Mock results certify only schema/dispatch/order. The boundary
tests at the end run the real bootstrap under the test interpreter and stop at
refusals; all real native and original-runtime runs are UNRUN.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib.machinery
import json
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import pytest

from microcosm.data import source_parent_replay as replay
from test_support.git_repositories import NO_BACKGROUND_MAINTENANCE
from test_support.microcosm_data.source_parent_replay import (
    carry_mock_source_parent,
    file_sha,
    make_mock_source_parent,
    mock_current_record,
    write_json,
)
from test_support.paths import paths_for


@pytest.fixture
def source_parent(tmp_path, monkeypatch):
    source = make_mock_source_parent(tmp_path)
    monkeypatch.setattr(replay, "_reviewed_profile", lambda: source.profile)
    return source


def _snapshot(source):
    return {path: path.read_bytes() for path in source.original_bytes}


def _forbid_worker(monkeypatch):
    calls = []

    def unexpected(*args, **kwargs):
        calls.append(kwargs.get("mode"))
        pytest.fail("source metadata must refuse before any worker/native I/O")

    monkeypatch.setattr(replay, "_run_worker", unexpected)
    return calls


def _repin(source, filename):
    fields = {
        "release_manifest.json": "release_manifest_sha256",
        "build_manifest.json": "build_manifest_sha256",
        "source_enrichment.json": "source_report_sha256",
        "source_enrichment_compatibility.json": "native_receipt_sha256",
    }
    source.profile[fields[filename]] = file_sha(source.release_dir / filename)


def test_mock_lineage_is_refused_by_the_unpatched_production_inventory(tmp_path):
    source = make_mock_source_parent(tmp_path)
    with pytest.raises(ValueError):
        replay.describe_source_parent_profile(source.release_dir)
    assert _snapshot(source) == source.original_bytes


def test_test_local_profile_patch_authenticates_only_mock_metadata(source_parent):
    assert replay.describe_source_parent_profile(source_parent.release_dir) == (
        source_parent.profile
    )
    assert _snapshot(source_parent) == source_parent.original_bytes


@pytest.mark.parametrize(
    "filename",
    [
        "release_manifest.json",
        "build_manifest.json",
        "source_enrichment.json",
        "source_enrichment_compatibility.json",
        "parent_release_manifest.json",
        "parent_build_manifest.json",
    ],
)
def test_modified_original_metadata_refuses_before_worker(
    source_parent, monkeypatch, filename
):
    path = source_parent.release_dir / filename
    path.write_bytes(path.read_bytes() + b"MOCK_METADATA_ONLY_TAMPER")
    before = _snapshot(source_parent)
    calls = _forbid_worker(monkeypatch)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
    assert calls == []
    assert _snapshot(source_parent) == before


@pytest.mark.parametrize("mutation", ["missing", "symlink"])
def test_missing_or_symlinked_original_refuses_before_worker(
    source_parent, monkeypatch, tmp_path, mutation
):
    path = source_parent.release_dir / "source_enrichment_compatibility.json"
    content = path.read_bytes()
    path.unlink()
    if mutation == "symlink":
        outside = tmp_path / "MOCK_METADATA_ONLY_external.json"
        outside.write_bytes(content)
        path.symlink_to(outside)
    calls = _forbid_worker(monkeypatch)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
    assert calls == []
    if mutation == "missing":
        assert not path.exists()
    else:
        assert path.is_symlink()
        assert outside.read_bytes() == content


@pytest.mark.parametrize(
    "mutation",
    ["receipt_parent", "recertification", "local", "annual", "wrong_operation"],
)
def test_repinning_does_not_admit_an_unsupported_parent_type(
    source_parent, monkeypatch, mutation
):
    filename = "release_manifest.json"
    path = source_parent.release_dir / filename
    manifest = json.loads(path.read_text())
    if mutation == "receipt_parent":
        manifest["release_type"] = "receipt_enrichment"
    elif mutation == "recertification":
        manifest["release_type"] = "compatibility_recertification"
    elif mutation == "local":
        manifest["dataset_role"] = "local_area"
    elif mutation == "annual":
        manifest["annual_projection"] = {"MOCK_METADATA_ONLY": True}
    else:
        filename = "source_enrichment.json"
        path = source_parent.release_dir / filename
        manifest = json.loads(path.read_text())
        manifest["operation"] = "MOCK_METADATA_ONLY_unknown_operation"
    write_json(path, manifest)
    _repin(source_parent, filename)
    before = _snapshot(source_parent)
    calls = _forbid_worker(monkeypatch)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
    assert calls == []
    assert _snapshot(source_parent) == before


@pytest.mark.parametrize("mutation", ["recursive", "local"])
def test_source_parent_has_one_national_calibration_ancestor(
    source_parent, monkeypatch, mutation
):
    path = source_parent.release_dir / "parent_release_manifest.json"
    ancestor = json.loads(path.read_text())
    ancestor["release_type" if mutation == "recursive" else "dataset_role"] = (
        "source_enrichment" if mutation == "recursive" else "local_area"
    )
    write_json(path, ancestor)
    manifest_path = source_parent.release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["parent_release_manifest"]["sha256"] = file_sha(path)
    write_json(manifest_path, manifest)
    _repin(source_parent, "release_manifest.json")
    calls = _forbid_worker(monkeypatch)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
    assert calls == []


def test_artifact_namespace_refuses_before_worker_or_output(source_parent, monkeypatch):
    path = source_parent.release_dir / "release_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["mock_destination_collision"] = {
        **manifest["artifacts"]["source_enrichment"],
        "path": "ancestor_release_manifest.json",
    }
    write_json(path, manifest)
    _repin(source_parent, "release_manifest.json")
    before = _snapshot(source_parent)
    calls = _forbid_worker(monkeypatch)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
    assert calls == []
    assert _snapshot(source_parent) == before


def test_missing_typed_context_never_runs_a_worker(source_parent, monkeypatch):
    calls = _forbid_worker(monkeypatch)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=None,
        )
    assert calls == []
    assert _snapshot(source_parent) == source_parent.original_bytes


def _leaf_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaf_paths(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaf_paths(item, (*prefix, index))
    else:
        yield prefix


def _legacy_leaf_paths():
    from test_support.microcosm_data.source_parent_replay import (
        mock_legacy_receipt,
        mock_sha,
    )

    return tuple(_leaf_paths(mock_legacy_receipt(mock_sha("dataset"))))


LEGACY_LEAF_PATHS = _legacy_leaf_paths()


def test_mock_legacy_fixture_accounts_for_every_observed_historical_field():
    assert len(LEGACY_LEAF_PATHS) == 59
    assert len({path[0] for path in LEGACY_LEAF_PATHS}) == 11
    assert len([path for path in LEGACY_LEAF_PATHS if path[0] == "checks"]) == 30
    assert len([path for path in LEGACY_LEAF_PATHS if "direct_url" in path]) == 4


@pytest.mark.parametrize("path", LEGACY_LEAF_PATHS)
def test_each_legacy_leaf_change_refuses_before_current_worker(
    source_parent, monkeypatch, path
):
    legacy = copy.deepcopy(source_parent.legacy)
    node = legacy
    for key in path[:-1]:
        node = node[key]
    old = node[path[-1]]
    node[path[-1]] = old + 1 if type(old) is int else old + "::MOCK_TAMPER"
    calls = []

    def transport(context, *, mode, release_dir, artifact_root):
        calls.append(mode)
        assert context is source_parent.context
        assert release_dir == source_parent.release_dir
        assert artifact_root == source_parent.artifact_root
        if mode != "legacy":
            pytest.fail("a changed original receipt must not reach current worker")
        return {"legacy_actual": legacy, "legacy_runtime": source_parent.runtime}

    monkeypatch.setattr(replay, "_run_worker", transport)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
    assert calls == ["legacy"]
    assert _snapshot(source_parent) == source_parent.original_bytes


@pytest.mark.parametrize(
    "mutation",
    ["unknown_top", "missing_top", "unknown_nested", "check_order", "bool_schema"],
)
def test_legacy_total_field_types_and_order_are_exact(
    source_parent, monkeypatch, mutation
):
    legacy = copy.deepcopy(source_parent.legacy)
    if mutation == "unknown_top":
        legacy["MOCK_ignored"] = "forbidden"
    elif mutation == "missing_top":
        del legacy["external_package_publication"]
    elif mutation == "unknown_nested":
        legacy["packages"]["policyengine"]["direct_url"]["ignored"] = True
    elif mutation == "check_order":
        legacy["checks"][0], legacy["checks"][1] = (
            legacy["checks"][1],
            legacy["checks"][0],
        )
    else:
        legacy["schema_version"] = True
    calls = []

    def transport(context, *, mode, release_dir, artifact_root):
        calls.append(mode)
        if mode != "legacy":
            pytest.fail("legacy field accounting must refuse before current worker")
        return {"legacy_actual": legacy, "legacy_runtime": source_parent.runtime}

    monkeypatch.setattr(replay, "_run_worker", transport)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
    assert calls == ["legacy"]
    assert _snapshot(source_parent) == source_parent.original_bytes


@pytest.mark.parametrize("mutation", ["missing", "unknown", "wrong_version"])
def test_legacy_runtime_is_complete_and_exact(source_parent, monkeypatch, mutation):
    runtime = dict(source_parent.runtime)
    if mutation == "missing":
        del runtime["hdf5"]
    elif mutation == "unknown":
        runtime["ignored"] = "MOCK_METADATA_ONLY"
    else:
        runtime["python"] += "::MOCK_TAMPER"
    calls = []

    def transport(context, *, mode, release_dir, artifact_root):
        calls.append(mode)
        if mode != "legacy":
            pytest.fail("wrong original runtime must refuse before current worker")
        return {"legacy_actual": source_parent.legacy, "legacy_runtime": runtime}

    monkeypatch.setattr(replay, "_run_worker", transport)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
    assert calls == ["legacy"]


@pytest.mark.parametrize("failing_mode", ["legacy", "current"])
def test_worker_refusal_never_changes_original_bytes(
    source_parent, monkeypatch, failing_mode
):
    calls = []

    def transport(context, *, mode, release_dir, artifact_root):
        calls.append(mode)
        if mode == failing_mode:
            raise ValueError("MOCK_METADATA_ONLY worker refusal")
        return {
            "legacy_actual": copy.deepcopy(source_parent.legacy),
            "legacy_runtime": dict(source_parent.runtime),
        }

    monkeypatch.setattr(replay, "_run_worker", transport)
    for _ in range(2):
        with pytest.raises(ValueError, match="MOCK_METADATA_ONLY worker refusal"):
            replay.qualify_source_parent(
                source_parent.release_dir,
                artifact_root=source_parent.artifact_root,
                context=source_parent.context,
            )
    expected = ["legacy"] if failing_mode == "legacy" else ["legacy", "current"]
    assert calls == expected * 2
    assert _snapshot(source_parent) == source_parent.original_bytes


@pytest.mark.parametrize("mutation", ["missing_receipt", "extra_field", "passed_only"])
def test_worker_envelope_cannot_substitute_a_saved_pass(
    source_parent, monkeypatch, mutation
):
    result = {
        "legacy_actual": source_parent.legacy,
        "legacy_runtime": source_parent.runtime,
    }
    if mutation == "missing_receipt":
        del result["legacy_actual"]
    elif mutation == "extra_field":
        result["ignored"] = "MOCK_METADATA_ONLY"
    else:
        result = {"status": "passed"}
    calls = []

    def transport(context, *, mode, release_dir, artifact_root):
        calls.append(mode)
        if mode != "legacy":
            pytest.fail("an incomplete worker envelope cannot reach current worker")
        return result

    monkeypatch.setattr(replay, "_run_worker", transport)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
    assert calls == ["legacy"]


def test_worker_transport_stages_fresh_isolated_hash_bound_calls(
    source_parent, monkeypatch
):
    from pathlib import Path
    from types import SimpleNamespace

    calls = []

    def subprocess_transport(command, **options):
        config_path = Path(command[command.index("--config") + 1])
        output = Path(command[command.index("--output") + 1])
        config = json.loads(config_path.read_text())
        assert command[0] == str(source_parent.context.original_python)
        assert command[1:3] == ["-I", "-X"]
        assert command[3].startswith("pycache_prefix=")
        assert command[command.index("--config-sha256") + 1] == file_sha(config_path)
        helper = Path(replay.__file__)
        worker = helper.with_name("source_parent_worker.py")
        assert command[command.index("--helper-sha256") + 1] == file_sha(helper)
        assert command[command.index("--worker-sha256") + 1] == file_sha(worker)
        assert config["context"] == source_parent.context.to_dict()
        assert config["release_dir"] == str(source_parent.release_dir)
        assert config["artifact_root"] == str(source_parent.artifact_root)
        assert options["timeout"] == source_parent.context.worker_timeout_seconds
        assert options["check"] is False
        assert options["capture_output"] is True
        assert not output.exists()
        calls.append((config["mode"], config_path, output))
        write_json(output, {"MOCK_METADATA_ONLY_transport": len(calls)})
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(replay.subprocess, "run", subprocess_transport)
    for mode in ("legacy", "current", "legacy"):
        actual = replay._run_worker(
            source_parent.context,
            mode=mode,
            release_dir=source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
        )
        assert actual == {"MOCK_METADATA_ONLY_transport": len(calls)}
    assert [mode for mode, _, _ in calls] == ["legacy", "current", "legacy"]
    assert len({path.parent for _, path, _ in calls}) == 3
    assert all(not path.parent.exists() for _, path, _ in calls)
    assert _snapshot(source_parent) == source_parent.original_bytes


@pytest.mark.parametrize("failure", ["nonzero", "missing_output", "symlink_output"])
def test_worker_transport_failure_cleans_only_owned_staging(
    source_parent, monkeypatch, tmp_path, failure
):
    from pathlib import Path
    from types import SimpleNamespace

    paths = []
    outside = tmp_path / "MOCK_METADATA_ONLY_unowned.json"
    outside.write_bytes(b"MOCK_METADATA_ONLY:unowned bytes\n")
    before = outside.read_bytes()

    def subprocess_transport(command, **options):
        output = Path(command[command.index("--output") + 1])
        paths.append(output)
        if failure == "symlink_output":
            output.symlink_to(outside)
        return SimpleNamespace(
            returncode=1 if failure == "nonzero" else 0,
            stderr="MOCK_METADATA_ONLY_refusal",
        )

    monkeypatch.setattr(replay.subprocess, "run", subprocess_transport)
    with pytest.raises(ValueError):
        replay._run_worker(
            source_parent.context,
            mode="legacy",
            release_dir=source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
        )
    assert len(paths) == 1
    assert not paths[0].parent.exists()
    assert outside.read_bytes() == before
    assert _snapshot(source_parent) == source_parent.original_bytes


def _mock_transport(source, monkeypatch, current=None):
    calls = []

    def transport(context, *, mode, release_dir, artifact_root):
        calls.append(mode)
        assert context is source.context
        assert release_dir.name == source.release_dir.name
        for name in ("release_manifest.json", "source_enrichment_compatibility.json"):
            assert (release_dir / name).read_bytes() == (
                source.release_dir / name
            ).read_bytes()
        if mode == "legacy":
            return {
                "legacy_actual": copy.deepcopy(source.legacy),
                "legacy_runtime": dict(source.runtime),
            }
        assert mode == "current"
        return copy.deepcopy(
            current if current is not None else mock_current_record(source)
        )

    monkeypatch.setattr(replay, "_run_worker", transport)
    return calls


def test_mock_orchestration_calls_both_workers_fresh_and_keeps_receipts_separate(
    source_parent, monkeypatch
):
    calls = _mock_transport(source_parent, monkeypatch)
    records = []
    for _ in range(2):
        evidence = replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
        record = evidence.to_dict()
        assert record["legacy_actual"] == source_parent.legacy
        assert record["legacy_runtime"] == source_parent.runtime
        assert "stored_inputs" not in record["legacy_actual"]
        assert "stored_inputs" in record["current"]["native_receipt"]
        assert record["current"]["profile"] == record["profile"]
        assert (
            record["current"]["native_receipt"]["packages"]
            == (record["legacy_actual"]["packages"])
        )
        records.append(record)
    assert calls == ["legacy", "current", "legacy", "current"]
    assert records[0] == records[1]
    assert records[0] is not records[1]
    assert _snapshot(source_parent) == source_parent.original_bytes


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_current_field",
        "unknown_current_field",
        "wrong_current_schema",
        "reused_legacy_record",
        "missing_native_register",
        "missing_register_check",
        "wrong_register_digest",
        "failed_native_status",
        "wrong_native_dataset",
        "changed_package_origin",
        "changed_core_version",
        "missing_current_producer",
        "dirty_current_producer",
        "unknown_compatibility_field",
    ],
)
def test_current_worker_must_supply_distinct_complete_fresh_metadata(
    source_parent, monkeypatch, mutation
):
    current = mock_current_record(source_parent)
    if mutation == "missing_current_field":
        del current["compatibility"]
    elif mutation == "unknown_current_field":
        current["ignored"] = "MOCK_METADATA_ONLY"
    elif mutation == "wrong_current_schema":
        current["schema_version"] = 1
    elif mutation == "reused_legacy_record":
        current = copy.deepcopy(source_parent.legacy)
    elif mutation == "missing_native_register":
        del current["native_receipt"]["stored_inputs"]
    elif mutation == "missing_register_check":
        current["native_receipt"]["checks"].remove(
            "country:defines_every_stored_model_input"
        )
    elif mutation == "wrong_register_digest":
        current["native_receipt"]["stored_inputs"]["register_sha256"] = "not-a-sha"
    elif mutation == "failed_native_status":
        current["native_receipt"]["status"] = "failed"
    elif mutation == "wrong_native_dataset":
        current["native_receipt"]["dataset_sha256"] = "not-the-mock-dataset"
    elif mutation == "changed_package_origin":
        current["native_receipt"]["packages"]["policyengine"]["direct_url"]["url"] += (
            "/MOCK_TAMPER"
        )
    elif mutation == "changed_core_version":
        current["native_receipt"]["packages"]["policyengine-core"]["version"] += ".1"
    elif mutation == "missing_current_producer":
        del current["producer_code"]
    elif mutation == "dirty_current_producer":
        current["producer_code"]["git_dirty"] = True
    else:
        current["compatibility"]["ignored"] = "MOCK_METADATA_ONLY"
    calls = _mock_transport(source_parent, monkeypatch, current)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
    assert calls == ["legacy", "current"]
    assert _snapshot(source_parent) == source_parent.original_bytes


def _carried_source(source, tmp_path, monkeypatch):
    calls = _mock_transport(source, monkeypatch)
    evidence = replay.qualify_source_parent(
        source.release_dir, artifact_root=source.artifact_root, context=source.context
    ).to_dict()
    carried = carry_mock_source_parent(source, tmp_path, evidence)
    calls.clear()
    return carried, calls


def test_saved_evidence_is_reconstructed_and_replayed_each_time(
    source_parent, monkeypatch, tmp_path
):
    from microcosm.data import compatibility_recertification as recert

    carried, calls = _carried_source(source_parent, tmp_path, monkeypatch)
    before = {
        path: path.read_bytes()
        for root in (carried.release_dir, carried.artifact_root)
        for path in root.iterdir()
    }
    for _ in range(2):
        result = recert.replay_parent_contract(
            carried.release_dir,
            carried.parent,
            artifact_root=carried.artifact_root,
            source_parent_replay=source_parent.context,
        )
        assert result["status"] == "v2_source_parent_qualified"
        assert result["evidence_sha256"] == file_sha(
            carried.release_dir / replay.SOURCE_PARENT_EVIDENCE_FILE
        )
    assert calls == ["legacy", "current", "legacy", "current"]
    assert {path: path.read_bytes() for path in before} == before
    assert _snapshot(source_parent) == source_parent.original_bytes


@pytest.mark.parametrize("field", ["runner_sha256", "register_sha256"])
def test_stale_saved_current_evidence_refuses_against_fresh_replay(
    source_parent, monkeypatch, tmp_path, field
):
    from test_support.microcosm_data.source_parent_replay import mock_sha

    carried, calls = _carried_source(source_parent, tmp_path, monkeypatch)
    record = copy.deepcopy(carried.evidence)
    native = record["current"]["native_receipt"]
    target = native["stored_inputs"] if field == "register_sha256" else native
    target[field] = mock_sha("MOCK_stale_evidence")
    write_json(carried.release_dir / replay.SOURCE_PARENT_EVIDENCE_FILE, record)
    with pytest.raises(ValueError, match="fresh complete replay"):
        replay.verify_source_parent_evidence(
            carried.release_dir,
            carried.parent,
            artifact_root=carried.artifact_root,
            context=source_parent.context,
        )
    assert calls == ["legacy", "current"]
    assert _snapshot(source_parent) == source_parent.original_bytes


@pytest.mark.parametrize(
    "failure", ["missing_evidence", "tamper_root", "symlink_root", "map_collision"]
)
def test_invalid_carried_evidence_refuses_before_worker(
    source_parent, monkeypatch, tmp_path, failure
):
    carried, calls = _carried_source(source_parent, tmp_path, monkeypatch)
    if failure == "missing_evidence":
        (carried.release_dir / replay.SOURCE_PARENT_EVIDENCE_FILE).unlink()
    elif failure in {"tamper_root", "symlink_root"}:
        path = carried.artifact_root / "microcosm_us_2024.h5"
        if failure == "tamper_root":
            path.write_bytes(b"MOCK_METADATA_ONLY_tampered_dataset")
        else:
            path.unlink()
            original = next(iter(source_parent.artifact_root.iterdir()))
            path.symlink_to(original)
    else:
        mapping = carried.parent["source_parent_artifact_map"]
        mapping[0]["carried_path"] = "MOCK_METADATA_ONLY_wrong_inverse.json"
    with pytest.raises(ValueError):
        replay.verify_source_parent_evidence(
            carried.release_dir,
            carried.parent,
            artifact_root=carried.artifact_root,
            context=source_parent.context,
        )
    assert calls == []
    assert _snapshot(source_parent) == source_parent.original_bytes


def test_parent_replay_requires_context_and_cannot_apply_it_to_calibration(
    source_parent, monkeypatch, tmp_path
):
    from microcosm.data import compatibility_recertification as recert

    carried, calls = _carried_source(source_parent, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="explicit original context"):
        recert.replay_parent_contract(
            carried.release_dir, carried.parent, artifact_root=carried.artifact_root
        )
    calibration = {**carried.parent, "release_type": "calibration"}
    with pytest.raises(ValueError, match="misplaced"):
        recert.replay_parent_contract(
            carried.release_dir,
            calibration,
            artifact_root=carried.artifact_root,
            source_parent_replay=source_parent.context,
        )
    assert calls == []


def test_mock_fresh_claim_measurement_is_separate_from_the_original_report(
    tmp_path, monkeypatch
):
    source = make_mock_source_parent(tmp_path, with_claim=True)
    monkeypatch.setattr(replay, "_reviewed_profile", lambda: source.profile)
    calls = _mock_transport(source, monkeypatch)
    record = replay.qualify_source_parent(
        source.release_dir, artifact_root=source.artifact_root, context=source.context
    ).to_dict()
    original = json.loads((source.release_dir / "source_enrichment.json").read_text())
    assert (
        record["current"]["compatibility"]["publisher_claims"]
        == (original["compatibility"]["publisher_claims"])
    )
    assert record["current"]["compatibility"]["claim_measurement"] == {
        "model": {
            "specifier": ">=0.1,<0.2",
            "measured_versions": ["0.1.0+mock.metadata.only"],
        }
    }
    assert "claim_measurement" not in original["compatibility"]
    assert calls == ["legacy", "current"]
    assert _snapshot(source) == source.original_bytes


@pytest.mark.parametrize(
    "mutation",
    ["missing", "specifier", "measured_version", "unmeasured_version", "core_claim"],
)
def test_saved_claim_cannot_widen_or_change_what_fresh_mock_probe_returned(
    tmp_path, monkeypatch, mutation
):
    source = make_mock_source_parent(tmp_path, with_claim=True)
    monkeypatch.setattr(replay, "_reviewed_profile", lambda: source.profile)
    carried, calls = _carried_source(source, tmp_path, monkeypatch)
    record = copy.deepcopy(carried.evidence)
    compatibility = record["current"]["compatibility"]
    measurement = compatibility["claim_measurement"]["model"]
    if mutation == "missing":
        del compatibility["claim_measurement"]
    elif mutation == "specifier":
        measurement["specifier"] = ">=0.1,<0.3"
    elif mutation == "measured_version":
        measurement["measured_versions"] = ["0.1.1+mock.metadata.only"]
    elif mutation == "unmeasured_version":
        measurement["measured_versions"].append("0.1.1+mock.metadata.only")
    else:
        compatibility["publisher_claims"]["core"] = {
            "name": "policyengine-core",
            "specifier": ">=0.1,<0.2",
        }
    write_json(carried.release_dir / replay.SOURCE_PARENT_EVIDENCE_FILE, record)
    with pytest.raises(ValueError):
        replay.verify_source_parent_evidence(
            carried.release_dir,
            carried.parent,
            artifact_root=carried.artifact_root,
            context=source.context,
        )
    assert calls == ["legacy", "current"]
    assert _snapshot(source) == source.original_bytes


# --- Review-repair boundary cases (C1/A2/A3/A4) ------------------------------
#
# Unlike the MOCK_METADATA_ONLY orchestration tests above, nothing below
# replaces _run_worker, subprocess.run, the profile, catalogs, certificates or
# the current Data implementation. The workers still refuse here because no
# authentic original interpreter, wheel, checkout or H5 is staged; real
# original-runtime and current-qualification replay remains UNRUN/HOLD and
# none of these tests qualifies a source parent.

_REPOSITORY = paths_for("microcosm-data").repository
_DATA_PREFIX = "packages/microcosm-data/src/"


@pytest.mark.parametrize("alias_of", ["source_enrichment", "build_manifest"])
def test_case_alias_of_a_parent_file_refuses_before_worker(
    source_parent, monkeypatch, alias_of
):
    path = source_parent.release_dir / "release_manifest.json"
    manifest = json.loads(path.read_text())
    original = manifest["artifacts"][alias_of]
    manifest["artifacts"]["mock_case_alias"] = {
        **original,
        "path": original["path"].replace(alias_of, alias_of.title()),
    }
    write_json(path, manifest)
    _repin(source_parent, "release_manifest.json")
    before = _snapshot(source_parent)
    calls = _forbid_worker(monkeypatch)
    with pytest.raises(ValueError):
        replay.qualify_source_parent(
            source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
            context=source_parent.context,
        )
    assert calls == []
    assert _snapshot(source_parent) == before


def _git_run(repository, *args):
    subprocess.run(
        [
            "git",
            *NO_BACKGROUND_MAINTENANCE,
            "-c",
            "user.name=MOCK_METADATA_ONLY",
            "-c",
            "user.email=mock-metadata-only@invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            *args,
        ],
        check=True,
        capture_output=True,
    )


def _staged_checkout(tmp_path, *, committed_links=()):
    """A throwaway committed Data closure using the repository's ignore rules."""
    checkout = tmp_path / "MOCK_METADATA_ONLY_checkout"
    data = checkout / _DATA_PREFIX / "microcosm/data"
    data.mkdir(parents=True)
    (data / "__init__.py").write_text('"""MOCK_METADATA_ONLY data closure."""\n')
    (data / "mock_module.py").write_text("VALUE = 'MOCK_METADATA_ONLY'\n")
    (checkout / ".gitignore").write_bytes((_REPOSITORY / ".gitignore").read_bytes())
    for relative, target in committed_links:
        link = checkout / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    _git_run(checkout, "init", "-q")
    _git_run(checkout, "add", "-A")
    _git_run(checkout, "commit", "-q", "-m", "MOCK_METADATA_ONLY closure")
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return checkout, commit


def _status_is_clean(checkout):
    return not subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
    ).stdout.strip()


def test_clean_committed_data_closure_authenticates(tmp_path):
    checkout, commit = _staged_checkout(tmp_path)
    replay._authenticate_checkout(checkout, commit=commit, declared={})


_SOURCELESS = (
    # Gitignored (*.py[cod]) sourceless bytecode git status cannot report.
    "microcosm/__init__.pyc",
    "microcosm/data/mock_planted.pyc",
    "microcosm/data/mock_planted.pyo",
    "microcosm/data/mock_package/__init__.pyc",
)


@pytest.mark.parametrize("relative", _SOURCELESS)
def test_ignored_sourceless_bytecode_refuses_checkout(tmp_path, relative):
    import py_compile

    checkout, commit = _staged_checkout(tmp_path)
    source = tmp_path / "MOCK_METADATA_ONLY_planted.py"
    source.write_text("raise SystemExit('MOCK_METADATA_ONLY planted bytecode')\n")
    target = checkout / _DATA_PREFIX / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(str(source), cfile=str(target), doraise=True)
    assert _status_is_clean(checkout)
    with pytest.raises(ValueError):
        replay._authenticate_checkout(checkout, commit=commit, declared={})


@pytest.mark.parametrize("suffix", sorted(set(importlib.machinery.EXTENSION_SUFFIXES)))
def test_ignored_native_extension_refuses_checkout(tmp_path, suffix):
    checkout, commit = _staged_checkout(tmp_path)
    name = f"mock_native{suffix}"
    (checkout / ".git/info/exclude").write_text(f"{name}\n")
    (checkout / _DATA_PREFIX / "microcosm/data" / name).write_bytes(
        b"MOCK_METADATA_ONLY not a native module"
    )
    assert _status_is_clean(checkout)
    with pytest.raises(ValueError):
        replay._authenticate_checkout(checkout, commit=commit, declared={})


def test_ignored_unrecorded_source_refuses_checkout(tmp_path):
    checkout, commit = _staged_checkout(tmp_path)
    (checkout / ".git/info/exclude").write_text("mock_unrecorded.py\n")
    (checkout / _DATA_PREFIX / "microcosm/data/mock_unrecorded.py").write_text(
        "VALUE = 'MOCK_METADATA_ONLY unrecorded'\n"
    )
    assert _status_is_clean(checkout)
    with pytest.raises(ValueError):
        replay._authenticate_checkout(checkout, commit=commit, declared={})


@pytest.mark.parametrize("target", ["module_dir", "empty_dir", "tracked_module"])
def test_committed_symlink_in_source_closure_refuses_checkout(tmp_path, target):
    outside = tmp_path / "MOCK_METADATA_ONLY_outside"
    outside.mkdir()
    if target == "module_dir":
        (outside / "mock_escape.py").write_text("VALUE = 'MOCK_METADATA_ONLY'\n")
        link, destination = "mock_linked", outside
    elif target == "empty_dir":
        link, destination = "mock_linked_empty", outside
    else:
        link, destination = "mock_alias.py", Path("mock_module.py")
    checkout, commit = _staged_checkout(
        tmp_path,
        committed_links=[(f"{_DATA_PREFIX}microcosm/data/{link}", destination)],
    )
    assert _status_is_clean(checkout)
    with pytest.raises(ValueError):
        replay._authenticate_checkout(checkout, commit=commit, declared={})


@pytest.mark.parametrize("mutation", ["modified", "untracked", "wrong_commit"])
def test_dirty_or_wrong_commit_checkout_still_refuses(tmp_path, mutation):
    checkout, commit = _staged_checkout(tmp_path)
    data = checkout / _DATA_PREFIX / "microcosm/data"
    if mutation == "modified":
        (data / "mock_module.py").write_text("VALUE = 'MOCK_METADATA_ONLY changed'\n")
    elif mutation == "untracked":
        (data / "mock_untracked.py").write_text("VALUE = 'MOCK_METADATA_ONLY'\n")
    else:
        commit = "0" * 40
    with pytest.raises(ValueError):
        replay._authenticate_checkout(checkout, commit=commit, declared={})


def _staging_root(tmp_path, monkeypatch):
    staging = tmp_path / "MOCK_METADATA_ONLY_staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))
    return staging


@pytest.mark.parametrize("mode", ["legacy", "current"])
def test_real_worker_subprocess_refuses_unreviewed_lineage_in_fixed_helper(
    source_parent, monkeypatch, tmp_path, mode
):
    """Real transport, bootstrap and helper; the in-process profile patch does
    not reach the child, so the fixed helper must refuse the mock lineage."""
    staging = _staging_root(tmp_path, monkeypatch)
    context = dataclasses.replace(
        source_parent.context, original_python=Path(sys.executable)
    )
    with pytest.raises(ValueError) as refused:
        replay._run_worker(
            context,
            mode=mode,
            release_dir=source_parent.release_dir,
            artifact_root=source_parent.artifact_root,
        )
    message = str(refused.value)
    assert f"{mode} source-parent worker refused" in message
    # The bootstrap accepted the real helper/worker/config digests and the
    # fixed helper loaded without any model/data import before authentication.
    assert "_worker_main" in message
    assert "only the reviewed tagged source parent is supported" in message
    assert "loaded before replay authentication" not in message
    assert "ModuleNotFoundError" not in message
    assert list(staging.iterdir()) == []
    assert _snapshot(source_parent) == source_parent.original_bytes


def _bootstrap_command(tmp_path, config_path, output, **digests):
    helper = Path(replay.__file__)
    worker = helper.with_name("source_parent_worker.py")
    values = {
        "config": file_sha(config_path),
        "helper": file_sha(helper),
        "worker": file_sha(worker),
        **digests,
    }
    cache = tmp_path / "MOCK_METADATA_ONLY_bytecode"
    cache.mkdir(exist_ok=True)
    return [
        sys.executable,
        "-I",
        "-X",
        f"pycache_prefix={cache}",
        str(worker),
        "--config",
        str(config_path),
        "--config-sha256",
        values["config"],
        "--helper-sha256",
        values["helper"],
        "--worker-sha256",
        values["worker"],
        "--output",
        str(output),
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "helper",
        "worker",
        "config",
        "existing_output",
        "symlink_output",
        "symlink_config",
    ],
)
def test_real_bootstrap_refuses_changed_inputs_or_reused_output(
    source_parent, tmp_path, mutation
):
    work = tmp_path / "MOCK_METADATA_ONLY_bootstrap"
    work.mkdir()
    config = work / "config.json"
    write_json(
        config,
        {
            "context": source_parent.context.to_dict(),
            "mode": "legacy",
            "release_dir": str(source_parent.release_dir),
            "artifact_root": str(source_parent.artifact_root),
        },
    )
    output = work / "actual.json"
    outside = tmp_path / "MOCK_METADATA_ONLY_unowned.json"
    outside.write_bytes(b"MOCK_METADATA_ONLY:unowned\n")
    digests = {}
    if mutation in {"helper", "worker", "config"}:
        digests[mutation] = "0" * 64
    elif mutation == "existing_output":
        output.write_bytes(b"MOCK_METADATA_ONLY:stale output\n")
    elif mutation == "symlink_output":
        output.symlink_to(outside)
    else:
        real = work / "real_config.json"
        config.rename(real)
        config.symlink_to(real)
    # Digests read through the config symlink, so only the link itself differs.
    command = _bootstrap_command(tmp_path, config, output, **digests)
    before = output.read_bytes() if output.is_file() else None
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert "_worker_main" not in result.stderr
    if before is None and not output.is_symlink():
        assert not output.exists()
    elif output.is_symlink():
        assert outside.read_bytes() == b"MOCK_METADATA_ONLY:unowned\n"
    else:
        assert output.read_bytes() == before
    assert _snapshot(source_parent) == source_parent.original_bytes


# --- C1: the current Data closure's first-party dependencies are bound -----


def _committed(prefix):
    listed = subprocess.run(
        ["git", "-C", str(_REPOSITORY), "ls-files", "-z", "--", prefix],
        check=True,
        capture_output=True,
    ).stdout
    return {name for name in listed.decode().split("\0") if name}


def _module_level_shards(path):
    """First-party namespace shards a module imports while it is imported."""
    shards = set()

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                for field in ("body", "orelse", "finalbody"):
                    child = getattr(node, field, None)
                    if isinstance(child, list):
                        visit(child)
                for handler in getattr(node, "handlers", ()):
                    visit(handler.body)
            for name in names:
                parts = name.split(".")
                if parts[0] == "microcosm" and len(parts) > 1:
                    shards.add(parts[1])

    visit(ast.parse(path.read_bytes()).body)
    return shards


def test_current_data_closure_imports_diagnostics_at_module_level():
    # Premise of C1 from committed source: importing current microcosm.data
    # imports microcosm.diagnostics, which the original 743683b0 runtime lacks.
    shards = set()
    for name in _committed(_DATA_PREFIX):
        if name.endswith(".py"):
            shards |= _module_level_shards(_REPOSITORY / name)
    assert "diagnostics" in shards


def test_current_producer_inventory_binds_every_first_party_closure_it_imports():
    """Every committed file of each first-party shard the current Data closure
    imports at module level must be authenticated and bound in the current
    producer inventory (SOURCE_PARENT_IMPLEMENTATION_FILES feeds
    producer_code.source_files_sha256), including py.typed and JSON schemas."""
    shards = set()
    for name in _committed(_DATA_PREFIX):
        if name.endswith(".py"):
            shards |= _module_level_shards(_REPOSITORY / name)
    bound = set(replay.SOURCE_PARENT_IMPLEMENTATION_FILES)
    # The plan binds the complete committed current Data AND Diagnostics
    # source/resource closures, not only the modules the replay names.
    assert "diagnostics" in shards
    for shard in sorted(shards | {"data"}):
        committed = _committed(f"packages/microcosm-{shard}/src/")
        assert committed, shard
        assert committed <= bound, (shard, sorted(committed - bound))


# --- C1: genuine committed current Data + Diagnostics import, one subprocess --
#
# The fixed helper under test runs in a fresh "-I -S" interpreter (the
# bootstrap's file-location load) against a clean sparse clone of the
# committed HEAD. "-S" keeps the test environment's editable first-party .pth
# paths off sys.path; only third-party site directories are appended, so the
# microcosm namespace can come only from roots the helper authenticated. It
# stops after importing microcosm.data.source_enrichment: no model, H5,
# original runtime or qualification runs, and nothing here is native evidence.
# A missing third-party dependency fails as BLOCKED; it is never skipped.

_CURRENT_IMPORT_PROBE = r"""
import importlib.util, json, sys
from pathlib import Path

helper, checkout, commit, *dependency_paths = sys.argv[1:]
sys.path.extend(dependency_paths)
spec = importlib.util.spec_from_file_location("_source_parent_fixed_helper", helper)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
enrichment = module._import_authenticated_sources(
    Path(checkout), commit=commit, declared={}, current=True
)
print(json.dumps({
    "runner": enrichment.__file__,
    "files": {
        name: getattr(item, "__file__", None)
        for name, item in sys.modules.items()
        if name == "microcosm" or name.startswith("microcosm.")
    },
    "namespace": list(sys.modules["microcosm"].__path__),
    "loaded": sorted(sys.modules),
}))
"""


def _git_quiet(*args):
    return subprocess.run(
        ["git", *NO_BACKGROUND_MAINTENANCE, *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_committed_clone(tmp_path):
    """Sparse, object-sharing clone of HEAD; reads the repository, never writes it."""
    commit = _git_quiet("-C", str(_REPOSITORY), "rev-parse", "HEAD")
    common = Path(
        _git_quiet(
            "-C",
            str(_REPOSITORY),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
    )
    clone = (tmp_path / "MOCK_METADATA_ONLY_current_clone").resolve()
    _git_quiet("clone", "--quiet", "--shared", "--no-checkout", str(common), str(clone))
    directories = sorted(
        {str(Path(path).parent) for path in replay.SOURCE_PARENT_IMPLEMENTATION_FILES}
        | {"packages/microcosm-data/src", "packages/microcosm-diagnostics/src"}
    )
    _git_quiet("-C", str(clone), "sparse-checkout", "set", "--cone", *directories)
    _git_quiet("-C", str(clone), "checkout", "--quiet", "--detach", commit)
    assert not _git_quiet(
        "-C", str(clone), "status", "--porcelain", "--untracked-files=all"
    )
    return clone, commit


def test_current_mode_imports_only_authenticated_data_and_diagnostics(tmp_path):
    clone, commit = _clean_committed_clone(tmp_path)
    cache = tmp_path / "MOCK_METADATA_ONLY_bytecode"
    cache.mkdir()
    paths = sysconfig.get_paths()
    dependencies = list(dict.fromkeys((paths["purelib"], paths["platlib"])))
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-X",
            f"pycache_prefix={cache}",
            "-c",
            _CURRENT_IMPORT_PROBE,
            str(Path(replay.__file__)),
            str(clone),
            commit,
            *dependencies,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if (
        result.returncode != 0
        and "ModuleNotFoundError" in result.stderr
        and ("microcosm" not in result.stderr.rsplit("ModuleNotFoundError", 1)[1])
    ):
        pytest.fail(
            f"BLOCKED: third-party dependency unavailable: {result.stderr[-2000:]}"
        )
    assert result.returncode == 0, result.stderr[-4000:]
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    data = clone / _DATA_PREFIX
    diagnostics = clone / "packages/microcosm-diagnostics/src"
    assert Path(observed["runner"]) == data / "microcosm/data/source_enrichment.py"
    assert {Path(path) for path in observed["namespace"]} == {
        data / "microcosm",
        diagnostics / "microcosm",
    }
    files = observed["files"]
    assert "microcosm.diagnostics" in files
    for name, path in files.items():
        if name == "microcosm":
            continue
        shard = name.split(".")[1]
        assert shard in {"data", "diagnostics"}, name
        root = data if shard == "data" else diagnostics
        assert path is not None and Path(path).is_relative_to(root), (name, path)
    country = {"policyengine", "policyengine_core", "policyengine_us", "spm_calculator"}
    assert not [name for name in observed["loaded"] if name.split(".")[0] in country]
    assert not _git_quiet(
        "-C", str(clone), "status", "--porcelain", "--untracked-files=all"
    )
