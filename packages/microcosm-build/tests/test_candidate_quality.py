"""Local candidate evidence never promotes or certifies a population."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from microcosm.build.candidate_quality import (
    CandidateQualityError,
    read_candidate_quality,
    summarize_pool_diagnostics,
    write_candidate_quality,
)


def metadata():
    return {
        "country": "us",
        "candidate_release_id": "candidate-fixture",
        "run_id": "run-fixture",
        "created_at": "2026-09-05T00:00:00Z",
        "execution_status": "completed",
        "release_eligibility": {
            "status": "excluded",
            "reason": "Failed historical fixture; diagnostics only.",
        },
        "population": {"sha256": "a" * 64, "period": "2024", "entity": "household"},
    }


def attachment(tmp_path, name="diagnostics.json", **changes):
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 7, "targets": []}) + "\n")
    value = {
        "status": "available",
        "source": str(path),
        "scope": "population",
        "identity": {
            "candidate_release_id": "candidate-fixture",
            "population_sha256": "a" * 64,
        },
    }
    value.update(changes)
    return value


def test_exact_bytes_are_hashed_and_direct_read_does_not_need_index(tmp_path):
    supplied = attachment(tmp_path)
    original = Path(supplied["source"]).read_bytes()
    root = tmp_path / "quality"
    result = write_candidate_quality(
        metadata(), {"calibration_diagnostics": supplied}, root
    )
    assert result["path"] == f"manifests/{result['sha256']}.json"
    payload = read_candidate_quality(root, result["sha256"])
    saved = payload["attachments"]["calibration_diagnostics"]
    assert saved["sha256"] == hashlib.sha256(original).hexdigest()
    assert (root / saved["path"]).read_bytes() == original
    assert "source" not in saved
    assert payload["execution_status"] == "completed"
    assert payload["release_eligibility"]["status"] == "excluded"
    assert payload["attachments"]["support"]["status"] == "missing"
    (root / "index.json").unlink()
    assert read_candidate_quality(root, result["sha256"]) == payload
    assert not any(
        p.name in {"latest.json", "latest_staging.json", "runs.json"}
        for p in root.rglob("*")
    )


@pytest.mark.parametrize("part", ["manifest", "attachment"])
def test_changed_bytes_fail_closed(tmp_path, part):
    root = tmp_path / "quality"
    result = write_candidate_quality(
        metadata(), {"calibration_diagnostics": attachment(tmp_path)}, root
    )
    payload = read_candidate_quality(root, result["sha256"])
    target = root / (
        result["path"]
        if part == "manifest"
        else payload["attachments"]["calibration_diagnostics"]["path"]
    )
    target.write_text("{}")
    with pytest.raises(CandidateQualityError, match="digest"):
        read_candidate_quality(root, result["sha256"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_release_id", "another-candidate"),
        ("population_sha256", "b" * 64),
        ("run_id", "another-run"),
    ],
)
def test_population_identity_mismatch_writes_nothing(tmp_path, field, value):
    supplied = attachment(tmp_path)
    supplied["identity"][field] = value
    root = tmp_path / "quality"
    with pytest.raises(CandidateQualityError, match="identity"):
        write_candidate_quality(metadata(), {"calibration_diagnostics": supplied}, root)
    assert not root.exists()


@pytest.mark.parametrize("status", ["pending", "missing", "failed"])
def test_unavailable_evidence_requires_reason_and_no_fabricated_metrics(
    tmp_path, status
):
    root = tmp_path / "quality"
    with pytest.raises(CandidateQualityError, match="reason"):
        write_candidate_quality(metadata(), {"support": {"status": status}}, root)
    result = write_candidate_quality(
        metadata(),
        {"support": {"status": status, "reason": "No measured support artifact."}},
        root,
    )
    assert read_candidate_quality(root, result["sha256"])["attachments"]["support"] == {
        "status": status,
        "reason": "No measured support artifact.",
    }


def test_failed_pre_population_run_can_carry_run_evidence(tmp_path):
    meta = metadata()
    meta.update(execution_status="failed", population=None)
    supplied = attachment(
        tmp_path,
        scope="run",
        identity={
            "candidate_release_id": meta["candidate_release_id"],
            "run_id": meta["run_id"],
        },
    )
    result = write_candidate_quality(
        meta, {"build_manifest": supplied}, tmp_path / "quality"
    )
    assert (
        read_candidate_quality(tmp_path / "quality", result["sha256"])["population"]
        is None
    )
    with pytest.raises(CandidateQualityError, match="population"):
        write_candidate_quality(
            meta, {"support": attachment(tmp_path)}, tmp_path / "bad"
        )


def test_model_only_evidence_does_not_claim_a_population(tmp_path):
    meta = metadata()
    meta["provenance"] = {
        "model_package": "policyengine-us",
        "model_version": "1.819.0",
    }
    supplied = attachment(
        tmp_path,
        scope="model",
        identity={"model_package": "policyengine-us", "model_version": "1.819.0"},
    )
    result = write_candidate_quality(
        meta, {"validation_reach": supplied}, tmp_path / "quality"
    )
    assert (
        "population_sha256"
        not in read_candidate_quality(tmp_path / "quality", result["sha256"])[
            "attachments"
        ]["validation_reach"]["identity"]
    )


def test_embedded_release_id_alias_must_match_candidate(tmp_path):
    supplied = attachment(tmp_path)
    Path(supplied["source"]).write_text(json.dumps({"release_id": "other-release"}))
    with pytest.raises(CandidateQualityError, match="release_id"):
        write_candidate_quality(
            metadata(), {"calibration_diagnostics": supplied}, tmp_path / "quality"
        )


@pytest.mark.parametrize("vintage", [2024, "", "   "])
def test_geography_vintage_must_be_nonempty_string(tmp_path, vintage):
    meta = metadata()
    meta["geography"] = {"levels": ["national"], "vintage": vintage}
    with pytest.raises(CandidateQualityError, match="vintage"):
        write_candidate_quality(meta, {}, tmp_path / "quality")


def test_completion_and_fit_do_not_infer_eligibility(tmp_path):
    meta = metadata()
    meta["release_eligibility"] = {
        "status": "not_evaluated",
        "reason": "Acceptance not run.",
    }
    result = write_candidate_quality(
        meta, {"calibration_diagnostics": attachment(tmp_path)}, tmp_path / "quality"
    )
    assert (
        read_candidate_quality(tmp_path / "quality", result["sha256"])[
            "release_eligibility"
        ]
        == meta["release_eligibility"]
    )
    meta["execution_status"] = "failed"
    meta["release_eligibility"]["status"] = "eligible"
    with pytest.raises(CandidateQualityError, match="eligible"):
        write_candidate_quality(meta, {}, tmp_path / "bad")


def test_source_must_be_aggregate_json_and_existing_content_is_immutable(tmp_path):
    supplied = attachment(tmp_path)
    Path(supplied["source"]).write_bytes(b"not json")
    with pytest.raises(CandidateQualityError, match="JSON"):
        write_candidate_quality(metadata(), {"support": supplied}, tmp_path / "quality")
    supplied = attachment(tmp_path)
    root = tmp_path / "quality"
    result = write_candidate_quality(
        metadata(), {"calibration_diagnostics": supplied}, root
    )
    (root / result["path"]).write_text("corrupt")
    with pytest.raises(CandidateQualityError, match="immutable"):
        write_candidate_quality(metadata(), {"calibration_diagnostics": supplied}, root)


def test_pool_summary_preserves_failure_units_and_never_certifies():
    manifest = {
        "release_id": "candidate-fixture",
        "status": "gate_failed",
        "simulation_ready": False,
        "provenance_counts": {"household": {"rows": 8}},
        "pool_h5": {"sha256": "a" * 64},
    }
    gates = {"passed": False, "failures": ["a", "a", "b"], "comparisons": [{"id": "x"}]}
    original = copy.deepcopy(gates)
    summary = summarize_pool_diagnostics(manifest, gates)
    assert summary["failure_string_count"] == 3
    assert "failed_comparisons" not in summary
    assert summary["simulation_ready"] is False
    assert summary["failures"] == ["a", "a", "b"]
    assert gates == original


def _bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    payload = {
        "schema_version": "cross_dataset.frontend_bundle.v1",
        "run_id": "evaluation-run",
        "snapshot_id": "chronicle-fixture",
        "source_ids": ["candidate-source"],
        "jurisdictions": ["US"],
        "fact_count": 0,
        "page_count": 0,
        "page_size": 100,
        "partitions": {"facts": []},
    }
    for name in ("summary", "groups", "fact_index"):
        data = json.dumps(
            {
                "schema_version": payload["schema_version"],
                "run_id": payload["run_id"],
                "snapshot_id": payload["snapshot_id"],
            }
        ).encode()
        path = root / f"{name}.json"
        path.write_bytes(data)
        payload["partitions"][name] = {
            "path": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    path = root / "manifest.json"
    path.write_text(json.dumps(payload))
    supplied = attachment(tmp_path)
    supplied["source"] = str(path)
    supplied["identity"].update(
        run_id="evaluation-run",
        snapshot_id="chronicle-fixture",
        source_id="candidate-source",
    )
    return payload, supplied


def test_cross_dataset_bundle_keeps_partitions_and_independent_evaluation_run(tmp_path):
    bundle, supplied = _bundle(tmp_path)
    root = tmp_path / "quality"
    meta = metadata()
    meta["provenance"] = {"chronicle_snapshot_id": "calibration-snapshot"}
    result = write_candidate_quality(meta, {"cross_dataset": supplied}, root)
    manifest = read_candidate_quality(root, result["sha256"])
    attached = manifest["attachments"]["cross_dataset"]
    assert attached["path"] == f"artifacts/{attached['sha256']}/manifest.json"
    assert attached["identity"]["run_id"] != manifest["run_id"]
    saved = root / attached["path"]
    assert json.loads(saved.read_text()) == bundle
    (saved.parent / "summary.json").write_text("{}")
    with pytest.raises(CandidateQualityError, match="digest"):
        read_candidate_quality(root, result["sha256"])


@pytest.mark.parametrize("defect", ["path", "digest", "snapshot", "source"])
def test_cross_dataset_import_refuses_mixed_or_unsafe_partitions(tmp_path, defect):
    bundle, supplied = _bundle(tmp_path)
    if defect == "path":
        bundle["partitions"]["summary"]["path"] = "../diagnostics.json"
    elif defect == "digest":
        bundle["partitions"]["summary"]["sha256"] = "f" * 64
    elif defect == "snapshot":
        bundle["snapshot_id"] = "another-snapshot"
    else:
        supplied["identity"]["source_id"] = "another-source"
    Path(supplied["source"]).write_text(json.dumps(bundle))
    with pytest.raises(CandidateQualityError):
        write_candidate_quality(
            metadata(), {"cross_dataset": supplied}, tmp_path / "quality"
        )
    assert not (tmp_path / "quality").exists()


def test_candidate_index_updates_selection_but_keeps_previous_manifest(tmp_path):
    root = tmp_path / "quality"
    first = write_candidate_quality(metadata(), {}, root)
    meta = metadata()
    meta["created_at"] = "2026-09-05T00:01:00Z"
    second = write_candidate_quality(meta, {}, root)
    index = json.loads((root / "index.json").read_text())
    assert len(index["candidates"]) == 1
    assert index["candidates"][0]["manifest"] == second
    assert (
        read_candidate_quality(root, first["sha256"])["created_at"]
        != meta["created_at"]
    )


def test_local_cli_resolves_attachment_relative_to_its_map(tmp_path):
    import importlib.util

    cli_path = (
        Path(__file__).resolve().parents[3] / "tools" / "build_candidate_quality.py"
    )
    spec = importlib.util.spec_from_file_location(
        "candidate_quality_cli_test", cli_path
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    supplied = attachment(tmp_path)
    supplied["source"] = "diagnostics.json"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata()))
    (tmp_path / "attachments.json").write_text(
        json.dumps({"calibration_diagnostics": supplied})
    )
    root = tmp_path / "quality"
    result = cli.main(
        [
            "--metadata",
            str(tmp_path / "metadata.json"),
            "--attachments",
            str(tmp_path / "attachments.json"),
            "--out",
            str(root),
            "--no-index",
        ]
    )
    assert (
        read_candidate_quality(root, result["sha256"])["candidate_release_id"]
        == "candidate-fixture"
    )
    assert not (root / "index.json").exists()


def _bound_metadata():
    meta = metadata()
    meta["provenance"] = {
        "model_package": "invented-model",
        "model_version": "1",
        "chronicle_snapshot_id": "calibration-snapshot",
    }
    return meta


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "other-run"),
        ("model_version", "2"),
        ("snapshot_id", "other-snapshot"),
        ("country", "uk"),
        ("country", {"code": "uk", "label": "Invented country block"}),
    ],
)
def test_omitted_sidecar_fields_cannot_hide_embedded_identity_conflicts(
    tmp_path, field, value
):
    supplied = attachment(tmp_path)
    Path(supplied["source"]).write_text(json.dumps({"schema_version": 7, field: value}))
    root = tmp_path / "quality"
    with pytest.raises(CandidateQualityError, match="identity"):
        write_candidate_quality(
            _bound_metadata(), {"calibration_diagnostics": supplied}, root
        )
    assert not root.exists()


def test_matching_present_fields_preserve_bytes_and_optional_fields_stay_optional(
    tmp_path,
):
    supplied = attachment(tmp_path)
    raw = b' {"country":{"code":"us","label":"Invented"}, "model_version":"1", "targets":[]}\n'
    Path(supplied["source"]).write_bytes(raw)
    root = tmp_path / "quality"
    result = write_candidate_quality(
        _bound_metadata(), {"calibration_diagnostics": supplied}, root
    )
    saved = read_candidate_quality(root, result["sha256"])["attachments"][
        "calibration_diagnostics"
    ]
    assert (root / saved["path"]).read_bytes() == raw


@pytest.mark.parametrize(
    "field,value",
    [("release_id", "other"), ("run_id", "other"), ("model_version", "2")],
)
def test_authenticated_readback_rejects_resealed_embedded_conflicts(
    tmp_path, field, value
):
    root = tmp_path / "quality"
    result = write_candidate_quality(
        _bound_metadata(), {"calibration_diagnostics": attachment(tmp_path)}, root
    )
    manifest = read_candidate_quality(root, result["sha256"])
    raw = json.dumps({field: value}).encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = f"artifacts/{digest}.json"
    (root / path).write_bytes(raw)
    manifest["attachments"]["calibration_diagnostics"].update(path=path, sha256=digest)
    raw_manifest = json.dumps(manifest).encode()
    manifest_digest = hashlib.sha256(raw_manifest).hexdigest()
    (root / "manifests" / f"{manifest_digest}.json").write_bytes(raw_manifest)
    with pytest.raises(CandidateQualityError, match="identity"):
        read_candidate_quality(root, manifest_digest)


@pytest.mark.parametrize(
    "raw",
    [
        b"\xef\xbb\xbf{}",
        "{}".encode("utf-16"),
        b'{"x":"\xff"}',
        b'{"x":[{"value":1e400}]}',
        b'{"x":-1e400}',
        b'{"x":' + b"9" * 400 + b"}",
        b'{"x":' + b"[" * 1500 + b"0" + b"]" * 1500 + b"}",
    ],
)
def test_unsupported_or_nonfinite_json_is_refused_without_rewriting(tmp_path, raw):
    supplied = attachment(tmp_path)
    Path(supplied["source"]).write_bytes(raw)
    with pytest.raises(CandidateQualityError, match="JSON"):
        write_candidate_quality(metadata(), {"support": supplied}, tmp_path / "quality")
    assert Path(supplied["source"]).read_bytes() == raw
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-09-05 00:00:00+00:00",
        "2026-09-05T00:00:00",
        "2026-02-30T00:00:00Z",
        "2026-09-05T00:00:00+01:99",
        "2026-09-05T00:00:00+24:00",
        "0000-01-01T00:00:00Z",
    ],
)
def test_timestamp_transport_contract_matches_reader(tmp_path, timestamp):
    meta = metadata()
    meta["created_at"] = timestamp
    with pytest.raises(CandidateQualityError):
        write_candidate_quality(meta, {}, tmp_path / "quality")


def test_whitespace_geography_level_is_not_a_declaration(tmp_path):
    meta = metadata()
    meta["geography"] = {"levels": [" \n"]}
    with pytest.raises(CandidateQualityError, match="geography level"):
        write_candidate_quality(meta, {}, tmp_path / "quality")


@pytest.mark.parametrize("scope", ["run", "model"])
def test_payload_cannot_claim_population_from_another_scope(tmp_path, scope):
    identity = (
        {"candidate_release_id": "candidate-fixture", "run_id": "run-fixture"}
        if scope == "run"
        else {"model_package": "invented-model", "model_version": "1"}
    )
    supplied = attachment(tmp_path, scope=scope, identity=identity)
    Path(supplied["source"]).write_text(json.dumps({"population_sha256": "a" * 64}))
    with pytest.raises(CandidateQualityError, match="scope"):
        write_candidate_quality(
            _bound_metadata(), {"validation_reach": supplied}, tmp_path / "quality"
        )


def test_model_inspection_run_is_independent_of_population_build(tmp_path):
    supplied = attachment(
        tmp_path,
        scope="model",
        identity={
            "model_package": "invented-model",
            "model_version": "1",
            "run_id": "model-inspection",
        },
    )
    Path(supplied["source"]).write_text(
        json.dumps({"model_version": "1", "run_id": "model-inspection"})
    )
    result = write_candidate_quality(
        _bound_metadata(), {"validation_reach": supplied}, tmp_path / "quality"
    )
    assert (
        read_candidate_quality(tmp_path / "quality", result["sha256"])["attachments"][
            "validation_reach"
        ]["scope"]
        == "model"
    )


def test_consumer_unsupported_bundle_path_fails_before_writes(tmp_path):
    bundle, supplied = _bundle(tmp_path)
    source = Path(supplied["source"]).parent
    (source / "summary.json").rename(source / "summary with space.json")
    bundle["partitions"]["summary"]["path"] = "summary with space.json"
    Path(supplied["source"]).write_text(json.dumps(bundle))
    with pytest.raises(CandidateQualityError, match="relative"):
        write_candidate_quality(
            metadata(), {"cross_dataset": supplied}, tmp_path / "quality"
        )
    assert not (tmp_path / "quality").exists()


def test_fifo_json_refuses_without_waiting_for_a_writer(tmp_path):
    import os
    import signal

    if not hasattr(os, "mkfifo"):
        pytest.skip("POSIX FIFO probe")
    supplied = attachment(tmp_path)
    path = Path(supplied["source"])
    path.unlink()
    os.mkfifo(path)

    def timeout(signum, frame):
        raise TimeoutError("Reader waited on a FIFO")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 1)
    try:
        with pytest.raises(CandidateQualityError, match="regular"):
            write_candidate_quality(
                metadata(), {"support": supplied}, tmp_path / "quality"
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_generated_manifest_and_control_json_are_bounded(tmp_path, monkeypatch):
    import sys

    module = sys.modules[write_candidate_quality.__module__]
    monkeypatch.setattr(module, "MAX_ATTACHMENT_BYTES", 64)
    with pytest.raises(CandidateQualityError, match="manifest.*bound"):
        write_candidate_quality(metadata(), {}, tmp_path / "quality")
    assert not (tmp_path / "quality").exists()
    control = tmp_path / "control.json"
    control.write_bytes(b" " * 65)
    with pytest.raises(CandidateQualityError, match="bounded"):
        module.read_candidate_quality_json(control)


@pytest.mark.parametrize(
    "defect", ["row", "duplicate", "path", "date", "eligibility", "symlink", "oversize"]
)
def test_invalid_existing_index_refuses_before_immutable_writes(
    tmp_path, monkeypatch, defect
):
    import sys

    module = sys.modules[write_candidate_quality.__module__]
    root = tmp_path / "quality"
    write_candidate_quality(metadata(), {}, root)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    if defect == "row":
        index["candidates"] = ["not an object"]
    elif defect == "duplicate":
        index["candidates"] *= 2
    elif defect == "path":
        index["candidates"][0]["manifest"]["path"] = "../outside.json"
    elif defect == "date":
        index["candidates"][0]["updated_at"] = "bad"
    elif defect == "eligibility":
        index["candidates"][0]["execution_status"] = "failed"
        index["candidates"][0]["release_eligibility"]["status"] = "eligible"
    elif defect == "oversize":
        monkeypatch.setattr(module, "MAX_ATTACHMENT_BYTES", 4096)
        index["padding"] = "x" * 4096
    index_path.write_text(json.dumps(index))
    if defect == "symlink":
        outside = tmp_path / "outside-index.json"
        index_path.rename(outside)
        index_path.symlink_to(outside)
    before = {p.name for p in (root / "manifests").iterdir()}
    meta = metadata()
    meta["created_at"] = "2026-09-05T00:01:00Z"
    with pytest.raises(CandidateQualityError):
        write_candidate_quality(meta, {}, root)
    assert {p.name for p in (root / "manifests").iterdir()} == before


def test_bundle_total_includes_its_manifest(tmp_path, monkeypatch):
    import sys

    module = sys.modules[write_candidate_quality.__module__]
    monkeypatch.setattr(module, "MAX_ATTACHMENT_BYTES", 1024)
    bundle, supplied = _bundle(tmp_path)
    root = Path(supplied["source"]).parent
    identity = {key: bundle[key] for key in ("schema_version", "run_id", "snapshot_id")}
    bundle["page_count"] = 1
    bundle["partitions"]["facts"] = [{"path": "facts.json", "count": 0, "page": 1}]
    descriptors = [
        bundle["partitions"][key] for key in ("summary", "groups", "fact_index")
    ] + bundle["partitions"]["facts"]
    for descriptor in descriptors:
        value = {**identity, "padding": ""}
        value["padding"] = "x" * (1024 - len(json.dumps(value).encode()))
        raw = json.dumps(value).encode()
        assert len(raw) == 1024
        (root / descriptor["path"]).write_bytes(raw)
        descriptor["sha256"] = hashlib.sha256(raw).hexdigest()
    Path(supplied["source"]).write_text(json.dumps(bundle))
    with pytest.raises(
        CandidateQualityError, match="Cross-dataset aggregate size bound"
    ):
        write_candidate_quality(
            metadata(), {"cross_dataset": supplied}, tmp_path / "quality"
        )
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize(
    "manifest",
    [
        {"pool_h5": None},
        {"provenance_counts": None},
        {"provenance_counts": {"household": None}},
        {"provenance_counts": {"household": 8}},
        {"pool_h5": "not-an-object", "provenance_counts": []},
        {},
    ],
)
def test_pool_summary_leaves_unexpected_receipt_fields_null(manifest):
    summary = summarize_pool_diagnostics(manifest, {})
    assert summary["population_sha256"] is None
    assert summary["household_record_count"] is None
    assert summary["failures"] is None
    assert summary["failure_string_count"] is None


def test_evaluation_snapshot_alias_cannot_contradict_its_descriptor(tmp_path):
    _, supplied = _bundle(tmp_path)
    supplied["identity"]["chronicle_snapshot_id"] = "another-evaluation"
    with pytest.raises(CandidateQualityError, match="chronicle_snapshot_id"):
        write_candidate_quality(
            metadata(), {"cross_dataset": supplied}, tmp_path / "quality"
        )
    assert not (tmp_path / "quality").exists()


def test_timestamp_preserves_real_leap_day_and_timezone(tmp_path):
    meta = metadata()
    meta["created_at"] = "2024-02-29T12:00:00.123+05:30"
    result = write_candidate_quality(meta, {}, tmp_path / "quality")
    assert (
        read_candidate_quality(tmp_path / "quality", result["sha256"])["created_at"]
        == meta["created_at"]
    )


@pytest.mark.parametrize(
    "field", ["candidate_release_id", "release_id", "population_sha256"]
)
def test_model_descriptor_cannot_claim_candidate_or_population_aliases(tmp_path, field):
    supplied = attachment(
        tmp_path,
        scope="model",
        identity={
            "model_package": "invented-model",
            "model_version": "1",
            field: "another-population",
        },
    )
    with pytest.raises(CandidateQualityError, match="Model-scoped"):
        write_candidate_quality(
            metadata(), {"validation_reach": supplied}, tmp_path / "quality"
        )
