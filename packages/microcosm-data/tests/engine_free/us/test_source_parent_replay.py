"""US source-parent orchestration with explicit MOCK_METADATA_ONLY callouts.

No fixture here contains a real dataset, wheel, interpreter or source checkout.
Mock results certify only schema/dispatch/order; all real native runs are UNRUN.
"""

from __future__ import annotations

import copy
import json

import pytest

from microcosm.data import source_parent_replay as replay
from test_support.microcosm_data.source_parent_replay import (
    carry_mock_source_parent,
    file_sha,
    make_mock_source_parent,
    mock_current_record,
    write_json,
)


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
