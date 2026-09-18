"""Exercise private candidate assembly through the real enrichment contract."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.spm_role_source import SpmRoleSourceResult
from microcosm.data import source_enrichment as contract
from microcosm.data.h5_enrichment import file_sha256
from microcosm.data.publish_cli import _staging_undelivered
from tools import build_us_spm_role_enrichment as builder

pytest.importorskip("tables")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    parent = tmp_path / "parent.h5"
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_spm_unit_id": [1, 1, 2],
            "age": [12.0, 16.0, 45.0],
            "person_weight": [2.0, 2.0, 1.0],
        }
    )
    with pd.HDFStore(parent, "w") as store:
        store.put("person", persons, format="table", data_columns=True)
        store.put(
            "spm_unit",
            pd.DataFrame({"spm_unit_id": [1, 2]}),
            format="table",
            data_columns=True,
        )
    parent_sha = file_sha256(parent)
    parent_release = tmp_path / "parent-release"
    parent_release.mkdir()
    parent_payloads = {
        "parent_release_manifest.json": {
            "schema_version": 1,
            "compatible_model_packages": [{"name": "old-model", "specifier": "==0.0"}],
        },
        "parent_build_manifest.json": {
            "build_id": contract.PARENT_BUILD_ID,
            "code": {"git_commit": "parent-only-claim"},
        },
        "calibration_diagnostics.json": {"schema_version": 5, "targets": []},
        "us_source_coverage.json": {"schema_version": 1},
    }
    pins = {}
    for name, payload in parent_payloads.items():
        path = parent_release / name.removeprefix("parent_")
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        pins[name] = file_sha256(path)
    for module in (builder, contract):
        monkeypatch.setattr(module, "PARENT_DATASET_SHA256", parent_sha)
        monkeypatch.setattr(module, "PARENT_FILES", pins)
    roles = np.array([False, True, True], dtype=bool)
    evidence = persons[["person_id", "person_spm_unit_id"]].copy()
    evidence["is_spm_independence_role"] = roles
    reference = tmp_path / "reference.csv"
    reference.write_text(evidence.to_csv(index=False))
    evidence_sha = file_sha256(reference)
    monkeypatch.setattr(builder, "REFERENCE_EVIDENCE_SHA256", evidence_sha)
    monkeypatch.setattr(contract, "SOURCE_EVIDENCE_SHA256", evidence_sha)
    counts = {
        "persons_joined": 3,
        "native_spm_units": 2,
        "total_source_people": 3,
        "total_source_units": 2,
        "minor_only_units_resolved": 1,
        "classification_changed_units_vs_age_only": 1,
    }
    monkeypatch.setattr(contract, "EXPECTED_COUNTS", counts)
    monkeypatch.setattr(contract, "CENSUS_PERSON_PINS", {2025: "a" * 64})
    provenance = {
        **counts,
        "dataset_sha256": parent_sha,
        "unmatched_persons": 0,
        "adult_child_person_count_mismatch_units": 0,
        "complete_source_membership_units": 2,
        "weights_used": False,
        "ages_changed": False,
        "primitive_column": contract.ROLE_VARIABLE,
        "evidence_column": contract.EVIDENCE_COLUMN,
        "source_checks": [
            {
                "survey_year": 2025,
                "csv_sha256": "a" * 64,
                "persons": 3,
                "units": 2,
                "adult_child_person_count_mismatch_units": 0,
                **contract.CENSUS_ARCHIVE_PINS[2025],
            }
        ],
    }
    reconstructed = SpmRoleSourceResult(roles, evidence, provenance)
    source_paths = {2024: tmp_path / "synthetic-pinned-source.csv"}
    source_paths[2024].write_text("Synthetic source derivation is separately tested.\n")
    calls = []

    def derive(actual_parent, actual_sources, *, expected_parent_sha256):
        calls.append((actual_parent, actual_sources, expected_parent_sha256))
        assert actual_parent == parent
        assert actual_sources == source_paths
        assert expected_parent_sha256 == parent_sha
        return reconstructed

    monkeypatch.setattr(builder, "derive_spm_role_source", derive)
    code = {
        "git_commit": "c" * 40,
        "git_dirty": True,
        "source_files_sha256": {"synthetic-producer.py": "d" * 64},
    }
    monkeypatch.setattr(builder, "_producer_identity", lambda: code)
    return SimpleNamespace(
        parent=parent,
        parent_sha=parent_sha,
        parent_release=parent_release,
        source_paths=source_paths,
        reference=reference,
        reconstructed=reconstructed,
        output=tmp_path / "candidate",
        code=code,
        calls=calls,
        release_id="populace-us-2024-synthetic-source-enrichment",
    )


def _build(inputs, **overrides):
    kwargs = {
        "parent_h5": inputs.parent,
        "parent_release_dir": inputs.parent_release,
        "source_paths": inputs.source_paths,
        "output_dir": inputs.output,
        "release_id": inputs.release_id,
        "reference_evidence_csv": inputs.reference,
    }
    return builder.build_candidate(**(kwargs | overrides))


def _release(inputs):
    return inputs.output / "releases" / inputs.release_id


def _assert_clean_failure(inputs):
    assert not inputs.output.exists()
    assert list(inputs.output.parent.glob(".spm-enrichment-*")) == []


def test_builder_assembles_new_candidate_through_real_contract(inputs):
    before = inputs.parent.read_bytes()
    report = _build(inputs)
    release = _release(inputs)
    candidate = inputs.output / "artifacts" / "populace_us_2024.h5"
    assert inputs.parent.read_bytes() == before
    assert inputs.calls == [(inputs.parent, inputs.source_paths, inputs.parent_sha)]
    assert report["compatibility"] == {"status": "pending"}
    assert report["dataset"]["sha256"] == file_sha256(candidate)
    enriched = pd.read_hdf(candidate, "person")
    pd.testing.assert_frame_equal(
        enriched.drop(columns=[contract.ROLE_VARIABLE]),
        pd.read_hdf(inputs.parent, "person"),
    )
    assert enriched[contract.ROLE_VARIABLE].tolist() == [False, True, True]
    assert enriched[contract.ROLE_VARIABLE].dtype == np.dtype(bool)
    assert (
        release / contract.SOURCE_EVIDENCE_FILE
    ).read_bytes() == inputs.reference.read_bytes()
    for name in contract.PARENT_FILES:
        assert (release / name).read_bytes() == (
            inputs.parent_release / name.removeprefix("parent_")
        ).read_bytes()
    build = json.loads((release / "build_manifest.json").read_text())
    assert build["code"] == inputs.code
    assert build["calibration"]["mode"] == "inherited"
    assert build["calibration"]["diagnostics_schema_version"] == 5
    assert build["staging"]["enabled"] is False
    assert _staging_undelivered(release) is False
    manifest = json.loads((release / "release_manifest.json").read_text())
    assert manifest["compatible_core_packages"] == []
    assert manifest["compatible_model_packages"] == []
    assert manifest["build"] == {"build_id": inputs.release_id}
    assert manifest["data_package"] == {
        "name": "microcosm-data",
        "version": metadata.version("microcosm-data"),
    }
    assert {entry["revision"] for entry in manifest["artifacts"].values()} == {
        inputs.release_id
    }
    for path in (
        candidate,
        release / contract.SOURCE_EVIDENCE_FILE,
        release / contract.SOURCE_PROVENANCE_FILE,
    ):
        assert path.stat().st_mode & 0o777 == 0o400
    assert list(inputs.output.parent.glob(".spm-enrichment-*")) == []


def test_reference_csv_is_optional_but_reconstruction_is_mandatory(inputs):
    _build(inputs, reference_evidence_csv=None)
    assert len(inputs.calls) == 1
    assert (
        _release(inputs) / contract.SOURCE_EVIDENCE_FILE
    ).read_bytes() == inputs.reference.read_bytes()


def test_repeat_build_has_identical_artifacts_and_receipts(inputs):
    first = _build(inputs)
    second_output = inputs.output.with_name("candidate-repeat")
    second = _build(inputs, output_dir=second_output)
    assert first == second
    first_files = {
        str(path.relative_to(inputs.output)): path.read_bytes()
        for path in inputs.output.rglob("*")
        if path.is_file()
    }
    second_files = {
        str(path.relative_to(second_output)): path.read_bytes()
        for path in second_output.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


@pytest.mark.parametrize(
    "release_id",
    [
        "wrong-country",
        contract.PARENT_BUILD_ID,
        "populace-us-2024-sub/path",
        "populace-us-2024-sub\\path",
    ],
)
def test_rejects_parent_or_unsafe_release_identifier(inputs, release_id):
    with pytest.raises(ValueError, match="new bare US 2024"):
        _build(inputs, release_id=release_id)
    assert inputs.calls == []
    _assert_clean_failure(inputs)


@pytest.mark.parametrize("kind", ["directory", "dangling_symlink"])
def test_existing_output_is_never_replaced(inputs, kind):
    if kind == "directory":
        inputs.output.mkdir()
        (inputs.output / "sentinel").write_text("keep")
    else:
        inputs.output.symlink_to(inputs.output.parent / "missing-target")
    with pytest.raises(FileExistsError, match="already exists"):
        _build(inputs)
    if kind == "directory":
        assert (inputs.output / "sentinel").read_text() == "keep"
    else:
        assert inputs.output.is_symlink()
    assert inputs.calls == []


@pytest.mark.parametrize(
    "tamper", ["parent", "parent_evidence", "reference", "reconstruction"]
)
def test_wrong_input_identity_refuses_before_output_creation(inputs, tamper):
    if tamper == "parent":
        with inputs.parent.open("ab") as stream:
            stream.write(b"tamper")
        message = "exact reviewed BuildP"
    elif tamper == "parent_evidence":
        (inputs.parent_release / "calibration_diagnostics.json").write_text(
            '{"schema_version":6}'
        )
        message = "Parent evidence SHA-256 mismatch"
    elif tamper == "reference":
        inputs.reference.write_text("wrong independent oracle")
        message = "Reference evidence CSV SHA-256 mismatch"
    else:
        inputs.reconstructed.evidence.loc[0, "is_spm_independence_role"] = True
        message = "Reconstructed source evidence differs"
    with pytest.raises(ValueError, match=message):
        _build(inputs)
    _assert_clean_failure(inputs)


def test_real_contract_rejects_role_array_disagreeing_with_evidence_and_cleans_up(
    inputs,
):
    inputs.reconstructed.role[0] = True
    with pytest.raises(
        ValueError, match="source person evidence does not match native"
    ):
        _build(inputs)
    _assert_clean_failure(inputs)


def test_gate_failure_removes_staged_artifacts(inputs, monkeypatch):
    def gate(*args, **kwargs):
        assert Path(args[0], "build_manifest.json").is_file()
        raise ValueError("synthetic final release gate refused")

    monkeypatch.setattr(builder, "validate_source_enrichment_candidate", gate)
    with pytest.raises(ValueError, match="synthetic final release gate refused"):
        _build(inputs)
    _assert_clean_failure(inputs)


def test_output_appearing_during_build_is_preserved(inputs, monkeypatch):
    real_gate = builder.validate_source_enrichment_candidate

    def gate(*args, **kwargs):
        result = real_gate(*args, **kwargs)
        inputs.output.mkdir()
        (inputs.output / "sentinel").write_text("concurrent owner")
        return result

    monkeypatch.setattr(builder, "validate_source_enrichment_candidate", gate)
    with pytest.raises(FileExistsError, match="Output appeared"):
        _build(inputs)
    assert (inputs.output / "sentinel").read_text() == "concurrent owner"
    assert list(inputs.output.parent.glob(".spm-enrichment-*")) == []


def test_changed_producer_source_refuses_after_validation(inputs, monkeypatch):
    identities = iter(
        [
            inputs.code,
            inputs.code | {"source_files_sha256": {"synthetic-producer.py": "e" * 64}},
        ]
    )
    monkeypatch.setattr(builder, "_producer_identity", lambda: next(identities))
    with pytest.raises(ValueError, match="Producer source files changed"):
        _build(inputs)
    _assert_clean_failure(inputs)


def test_producer_receipt_hashes_the_loaded_checkout():
    identity = builder._producer_identity()
    root = Path(builder.__file__).resolve().parents[1]
    assert identity["source_files_sha256"] == {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in builder._PRODUCER_FILES
    }
    assert len(identity["git_commit"]) == 40


def test_producer_rejects_imported_code_from_another_checkout(monkeypatch):
    def foreign_derivation(*args, **kwargs):
        pytest.fail("foreign source reconstruction must not execute")

    monkeypatch.setattr(builder, "derive_spm_role_source", foreign_derivation)
    with pytest.raises(ValueError, match="execute this checkout"):
        builder._producer_identity()


def test_real_builder_manifest_certifies_prepares_and_publishes(
    inputs, tmp_path, monkeypatch
):
    """Exercise the emitted manifest; synthetic source/runtime probes stay local."""
    from importlib import import_module

    from microcosm.data import release as publisher

    fake_hub = import_module("packages.microcosm-data.tests.test_release").FakeHub
    _build(inputs)
    candidate_release = _release(inputs)
    artifact_root = inputs.output / "artifacts"
    dataset = artifact_root / "populace_us_2024.h5"
    original_files = {
        path.name: path.read_bytes()
        for path in candidate_release.iterdir()
        if path.is_file()
    }
    manifest = json.loads(original_files["release_manifest.json"])
    assert manifest["default_datasets"] == {"national": "populace_us_2024"}
    assert manifest["artifacts"]["build_manifest"]["kind"] == "evidence"
    assert manifest["artifacts"]["build_manifest"]["path"] == "build_manifest.json"
    assert manifest["artifacts"]["populace_us_2024"]["kind"] == "microdata"

    # Source derivation/producer identity and wheel probes have their own tests.
    # These seams keep this builder-to-publisher test independent of microdata,
    # installed country packages and a clean developer checkout.
    wheels = tuple(tmp_path / f"{name}.whl" for name in contract.COMPATIBILITY_PACKAGES)
    compatibility_calls = []
    receipt = {
        "status": "passed",
        "dataset_sha256": file_sha256(dataset),
        "packages": {
            "policyengine-us": {"version": "1.999.0"},
            "policyengine-core": {"version": "3.99.0"},
            "policyengine": {"version": "5.99.0"},
            "spm-calculator": {"version": "1.0.0"},
        },
    }

    def probe(path, *, require_wheels, compatibility_wheels):
        assert Path(path) == dataset
        assert require_wheels is True
        assert compatibility_wheels == wheels
        compatibility_calls.append(path)
        return receipt

    def check_synthetic_producer(code):
        assert code == inputs.code

    monkeypatch.setattr(contract, "run_native_loader_compatibility", probe)
    monkeypatch.setattr(
        contract, "_check_producer_source_identity", check_synthetic_producer
    )
    monkeypatch.setattr(publisher, "_hf_api", lambda: pytest.fail("must use FakeHub"))
    certified = contract.certify_source_enrichment(
        candidate_release,
        tmp_path / "certified" / inputs.release_id,
        parent_h5=inputs.parent,
        artifact_root=artifact_root,
        compatibility_wheels=wheels,
    )
    assert original_files == {
        path.name: path.read_bytes()
        for path in candidate_release.iterdir()
        if path.is_file()
    }
    prepared = publisher.prepare_release(
        certified,
        parent_h5=inputs.parent,
        artifact_root=artifact_root,
        compatibility_wheels=wheels,
    )
    assert prepared.filenames.count("build_manifest.json") == 1
    assert "populace_us_2024.h5" not in prepared.filenames
    assert prepared.root_artifacts == {"populace_us_2024.h5": file_sha256(dataset)}
    hub = fake_hub()
    pointer = publisher.publish_release(
        certified,
        "policyengine/populace-us",
        parent_h5=inputs.parent,
        artifact_root=artifact_root,
        compatibility_wheels=wheels,
        api=hub,
        notify=False,
        updated_at="2026-09-09T00:00:00+00:00",
    )
    assert (
        len(compatibility_calls) == 4
    )  # Certification, its replay, preflight, publish.
    manifest = json.loads((certified / "release_manifest.json").read_text())
    prefix = f"releases/{inputs.release_id}/"
    expected_hashes = {
        entry["path"]
        if entry["kind"] == "microdata"
        else prefix + entry["path"]: entry["sha256"]
        for entry in manifest["artifacts"].values()
    }
    commits = [event for kind, event in hub.events if kind == "create_commit"]
    assert [commit["revision"] for commit in commits] == [
        f"release-staging/{inputs.release_id}",
        "main",
    ]
    for commit in commits:
        paths = commit["paths"]
        assert len(paths) == len(set(paths))
        assert [path for path in paths if path.endswith(".h5")] == [dataset.name]
        assert set(paths) - {"latest.json"} == set(expected_hashes) | {
            prefix + "release_manifest.json"
        }
    for path, expected in expected_hashes.items():
        contents = [
            content for uploaded_path, content in hub.uploads if uploaded_path == path
        ]
        assert len(contents) == 2  # Once in staging and once in main's atomic commit.
        assert all(
            hashlib.sha256(content).hexdigest() == expected for content in contents
        )
    assert hub.tags == [{"tag": inputs.release_id, "revision": commits[0]["commit"]}]
    assert "latest.json" not in commits[0]["paths"]
    assert commits[-1]["paths"][-1] == "latest.json"
    assert hub.uploads[-1][0] == "latest.json"
    assert json.loads(hub.uploads[-1][1]) == pointer
    assert pointer["release_id"] == inputs.release_id
    assert pointer["paths"] == {
        name: f"{prefix}{name}.json"
        for name in (
            "build_manifest",
            "release_manifest",
            "calibration_diagnostics",
            "us_source_coverage",
        )
    }
