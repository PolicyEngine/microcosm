"""The inheritance lane preserves calibration and rejects rehashed tampering."""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings

import h5py
import numpy as np
import pytest

from microcosm.data import source_enrichment as enrichment
from microcosm.data.contract import ReleaseContractError, validate_release_dir
from microcosm.data.h5_enrichment import append_native_spm_role, compare_h5_enrichment
from microcosm.data.publish_cli import main as publish_main
from microcosm.data.release import publish_release


def _write(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True))


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")
    parent = tmp_path / "parent.h5"
    people = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_spm_unit_id": [1, 1, 2],
            "person_household_id": [1, 1, 2],
            "person_weight": [2.0, 2.0, 1.0],
            "age": [12, 16, 45],
        }
    )
    with pd.HDFStore(parent, "w") as store:
        store.put("_time_period", pd.Series([2024]), format="table")
        store.put("person", people, format="table", data_columns=True)
        for entity in ("household", "spm_unit", "tax_unit", "family", "marital_unit"):
            store.put(
                entity,
                pd.DataFrame({f"{entity}_id": [1, 2], f"{entity}_weight": [2.0, 1.0]}),
                format="table",
                data_columns=True,
            )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    h5 = artifact_root / "populace_us_2024.h5"
    append_native_spm_role(
        parent,
        h5,
        np.array([False, True, True]),
        expected_parent_sha256=enrichment.sha256_file(parent),
    )
    release = tmp_path / "populace-us-2024-test-enrichment"
    release.mkdir()
    parent_payloads = {
        "parent_release_manifest.json": {"schema_version": 1},
        "parent_build_manifest.json": {"calibration": {"old": True}},
        "calibration_diagnostics.json": {"schema_version": 5, "targets": []},
        "us_source_coverage.json": {"schema_version": 1},
    }
    for name, payload in parent_payloads.items():
        _write(release / name, payload)
    pins = {name: enrichment.sha256_file(release / name) for name in parent_payloads}
    monkeypatch.setattr(enrichment, "PARENT_FILES", pins)
    monkeypatch.setattr(
        enrichment, "PARENT_DATASET_SHA256", enrichment.sha256_file(parent)
    )
    counts = {
        "persons_joined": 3,
        "native_spm_units": 2,
        "total_source_people": 3,
        "total_source_units": 2,
        "minor_only_units_resolved": 0,
        "classification_changed_units_vs_age_only": 1,
    }
    monkeypatch.setattr(enrichment, "EXPECTED_COUNTS", counts)
    evidence = release / enrichment.SOURCE_EVIDENCE_FILE
    evidence.write_text(
        "person_id,person_spm_unit_id,is_spm_independence_role\n1,1,False\n2,1,True\n3,2,True\n"
    )
    monkeypatch.setattr(
        enrichment, "SOURCE_EVIDENCE_SHA256", enrichment.sha256_file(evidence)
    )
    provenance = {
        **counts,
        "dataset_sha256": enrichment.PARENT_DATASET_SHA256,
        "unmatched_persons": 0,
        "adult_child_person_count_mismatch_units": 0,
        "complete_source_membership_units": 2,
        "weights_used": False,
        "ages_changed": False,
        "primitive_column": enrichment.ROLE_VARIABLE,
        "evidence_column": enrichment.EVIDENCE_COLUMN,
        "source_checks": [
            {
                "survey_year": year,
                "csv_sha256": csv_sha,
                "adult_child_person_count_mismatch_units": 0,
                **enrichment.CENSUS_ARCHIVE_PINS[year],
            }
            for year, csv_sha in enrichment.CENSUS_PERSON_PINS.items()
        ],
    }
    _write(release / enrichment.SOURCE_PROVENANCE_FILE, provenance)
    dataset = {"filename": h5.name, "sha256": enrichment.sha256_file(h5)}
    build = {
        "build_id": release.name,
        "release_type": "source_enrichment",
        "dataset": dataset,
        "code": {"git_commit": "a" * 40, "git_dirty": False},
        "calibration": {
            "mode": "inherited",
            "parent_build_id": enrichment.PARENT_BUILD_ID,
            "diagnostics_sha256": pins["calibration_diagnostics.json"],
            "diagnostics_schema_version": 5,
        },
    }
    _write(release / "build_manifest.json", build)
    report = {
        "schema_version": 1,
        "release_type": "source_enrichment",
        "operation": "add_native_spm_independent_minor_role",
        "parent": {
            "build_id": enrichment.PARENT_BUILD_ID,
            "repo_id": "policyengine/populace-us",
            "revision": enrichment.PARENT_BUILD_ID,
            "dataset_sha256": enrichment.PARENT_DATASET_SHA256,
            "calibration_diagnostics_schema_version": 5,
            "files": pins,
        },
        "dataset": dataset,
        "added_variable": {
            "name": enrichment.ROLE_VARIABLE,
            "entity": "person",
            "dtype": "bool",
            "age_gate_applied": False,
        },
        "preservation": compare_h5_enrichment(parent, h5),
        "source": {
            "person_evidence_filename": enrichment.SOURCE_EVIDENCE_FILE,
            "person_evidence_sha256": enrichment.sha256_file(evidence),
            "provenance_filename": enrichment.SOURCE_PROVENANCE_FILE,
            "provenance_sha256": enrichment.sha256_file(
                release / enrichment.SOURCE_PROVENANCE_FILE
            ),
        },
        "reconciliation": provenance,
        "compatibility": {"status": "pending"},
    }
    _write(release / enrichment.SOURCE_ENRICHMENT_FILE, report)
    manifest = {
        "schema_version": 1,
        "release_type": "source_enrichment",
        "build": {"build_id": release.name},
        "compatible_core_packages": [],
        "compatible_model_packages": [],
        "default_datasets": {"national": "dataset"},
        "artifacts": {},
    }
    for path in [
        *map(lambda name: release / name, pins),
        release / enrichment.SOURCE_ENRICHMENT_FILE,
        evidence,
        release / enrichment.SOURCE_PROVENANCE_FILE,
        h5,
    ]:
        manifest["artifacts"]["dataset" if path == h5 else path.stem] = {
            "kind": "microdata" if path == h5 else "diagnostics",
            "path": path.name,
            "repo_id": "policyengine/populace-us",
            "revision": release.name,
            "sha256": enrichment.sha256_file(path),
        }
    _write(release / "release_manifest.json", manifest)
    return release, parent, artifact_root


def _validate(candidate, **kwargs):
    release, parent, root = candidate
    return enrichment.validate_source_enrichment_candidate(
        release, parent_h5=parent, artifact_root=root, **kwargs
    )


def _refresh(release, filename):
    manifest_path = release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["artifacts"].values():
        if entry["path"] == filename:
            entry["sha256"] = enrichment.sha256_file(release / filename)
    _write(manifest_path, manifest)


def test_pending_candidate_is_valid_but_actual_release_gate_refuses(candidate):
    assert _validate(candidate)["compatibility"]["status"] == "pending"
    release, parent, root = candidate
    with pytest.raises(ReleaseContractError, match="compatibility is pending"):
        validate_release_dir(release, parent_h5=parent, artifact_root=root)


def test_publisher_refuses_before_constructing_hub_client(candidate, monkeypatch):
    import microcosm.data.release as release_module

    def no_hub():
        pytest.fail("publisher contacted Hub for a pending enrichment")

    monkeypatch.setattr(release_module, "_hf_api", no_hub)
    release, parent, root = candidate
    with pytest.raises(ReleaseContractError, match="compatibility is pending"):
        publish_release(
            release, "policyengine/populace-us", parent_h5=parent, artifact_root=root
        )


def test_preflight_runs_actual_gate_and_never_publishes(candidate, monkeypatch):
    import microcosm.data.publish_cli as cli

    monkeypatch.setattr(
        cli, "publish_release", lambda *a, **kw: pytest.fail("preflight published")
    )
    release, parent, root = candidate
    with pytest.raises(ReleaseContractError, match="compatibility is pending"):
        publish_main(
            [
                str(release),
                "--parent-h5",
                str(parent),
                "--artifact-root",
                str(root),
                "--preflight-only",
            ]
        )


def test_requires_actual_parent_and_h5(candidate):
    release, _, root = candidate
    with pytest.raises(ReleaseContractError, match="requires parent_h5"):
        enrichment.validate_source_enrichment_candidate(
            release, parent_h5=None, artifact_root=root
        )


def test_schema_five_cannot_be_relabelled_even_with_rehashed_manifests(candidate):
    release, _, _ = candidate
    diagnostics = release / "calibration_diagnostics.json"
    _write(diagnostics, {"schema_version": 6, "targets": []})
    _refresh(release, diagnostics.name)
    with pytest.raises(ReleaseContractError, match="reviewed parent bytes"):
        _validate(candidate)


@pytest.mark.parametrize("field", ["person_weight", "person_spm_unit_id", "age"])
def test_preservation_rejects_changed_old_values(candidate, field):
    release, _, root = candidate
    h5 = root / "populace_us_2024.h5"
    with h5py.File(h5, "r+") as handle:
        table = handle["person/table"]
        row = table[0]
        row[field] += 1
        table[0] = row
    with pytest.raises(ReleaseContractError, match="exact H5 preservation failed"):
        _validate(candidate)


def test_rehashed_role_evidence_attack_still_fails_reviewed_source_pin(candidate):
    release, parent, root = candidate
    path = release / enrichment.SOURCE_EVIDENCE_FILE
    path.write_text(path.read_text().replace("1,1,False", "1,1,True"))
    h5 = root / "populace_us_2024.h5"
    with h5py.File(h5, "r+") as handle:
        table = handle["person/table"]
        row = table[0]
        row[enrichment.ROLE_VARIABLE] = True
        table[0] = row
    report_path = release / enrichment.SOURCE_ENRICHMENT_FILE
    report = json.loads(report_path.read_text())
    report["source"]["person_evidence_sha256"] = enrichment.sha256_file(path)
    report["dataset"]["sha256"] = enrichment.sha256_file(h5)
    report["preservation"] = compare_h5_enrichment(parent, h5)
    _write(report_path, report)
    _refresh(release, path.name)
    _refresh(release, report_path.name)
    with pytest.raises(
        ReleaseContractError, match="independently reviewed Census-derived"
    ):
        _validate(candidate)


def test_pending_candidate_cannot_copy_old_model_claims(candidate):
    release, _, _ = candidate
    path = release / "release_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["build"]["built_with_model_package"] = {
        "name": "policyengine-us",
        "version": "1.764.6",
    }
    manifest["compatible_model_packages"] = [
        {"name": "policyengine-us", "specifier": "==1.764.6"}
    ]
    _write(path, manifest)
    with pytest.raises(
        ReleaseContractError, match="must not claim model/Core compatibility"
    ):
        _validate(candidate)


def test_fabricated_passed_receipt_is_replayed(candidate, monkeypatch):
    release, _, _ = candidate
    report_path = release / enrichment.SOURCE_ENRICHMENT_FILE
    report = json.loads(report_path.read_text())
    receipt = release / enrichment.COMPATIBILITY_FILE
    _write(receipt, {"status": "passed"})
    report["compatibility"] = {
        "status": "passed",
        "filename": receipt.name,
        "sha256": enrichment.sha256_file(receipt),
    }
    _write(report_path, report)
    _refresh(release, report_path.name)
    calls = []

    def actual_test(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("actual country role is unavailable")

    monkeypatch.setattr(enrichment, "run_native_loader_compatibility", actual_test)
    with pytest.raises(
        ReleaseContractError, match="actual country role is unavailable"
    ):
        _validate(candidate, require_compatibility=True)
    assert calls == [{"require_wheels": True, "compatibility_wheels": ()}]


def test_unknown_release_type_cannot_fall_back_to_calibration(candidate):
    release, _, _ = candidate
    path = release / "release_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["release_type"] = "trust_parent"
    _write(path, manifest)
    with pytest.raises(ReleaseContractError, match="unknown release_type"):
        validate_release_dir(release)


def _qualify_candidate(candidate, tmp_path, monkeypatch, **claim):
    from importlib import metadata

    release, parent, root = candidate
    receipt = {
        "status": "passed",
        "dataset_sha256": enrichment.sha256_file(root / "populace_us_2024.h5"),
        "packages": {
            "policyengine-us": {"version": "1.999.0"},
            "policyengine-core": {"version": "3.99.0"},
            "policyengine": {"version": "5.99.0"},
            "spm-calculator": {"version": "1.0.0"},
        },
    }
    calls = []

    def actual_test(*args, **kwargs):
        calls.append(kwargs)
        return receipt

    monkeypatch.setattr(enrichment, "run_native_loader_compatibility", actual_test)
    monkeypatch.setattr(metadata, "version", lambda name: "0.1.0")
    monkeypatch.setattr(
        enrichment, "_check_producer_source_identity", lambda code: None
    )
    output = tmp_path / "certified" / release.name
    result = enrichment.certify_source_enrichment(
        release,
        output,
        parent_h5=parent,
        artifact_root=root,
        compatibility_wheels=(tmp_path / "country.whl",),
        **claim,
    )
    assert result == output
    return output, calls


def test_annual_cut_preserves_source_enrichment_qualification(
    candidate, tmp_path, monkeypatch
):
    from microcosm.data.release import prepare_release

    from .test_annual_projections import add_annual_extension

    output, calls = _qualify_candidate(candidate, tmp_path, monkeypatch)
    _, parent, root = candidate
    tag = add_annual_extension(output, root)
    previous_calls = len(calls)
    prepared = prepare_release(
        output,
        artifact_root=root,
        parent_h5=parent,
        compatibility_wheels=(tmp_path / "country.whl",),
        tag_name=tag,
        update_latest=False,
    )
    assert len(calls) == previous_calls + 1
    assert calls[-1]["require_wheels"] is True
    assert {
        "annual_manifest.json",
        "annual_acceptance.json",
        "projection_2025.json",
        enrichment.COMPATIBILITY_FILE,
    }.issubset(prepared.filenames)
    assert set(prepared.root_artifacts) == {"populace_us_2024.h5", "annual_2025.h5"}


@pytest.mark.parametrize(
    "problem", ["base_path", "extra_microdata", "missing_parent", "absent_annual"]
)
def test_annual_source_enrichment_retains_original_constraints(
    candidate, tmp_path, monkeypatch, problem
):
    from .test_annual_projections import add_annual_extension

    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch)
    _, parent, root = candidate
    add_annual_extension(output, root)
    path = output / "release_manifest.json"
    manifest = json.loads(path.read_text())
    if problem == "base_path":
        key, artifact = next(
            (key, value)
            for key, value in manifest["artifacts"].items()
            if value["path"] == enrichment.SOURCE_EVIDENCE_FILE
        )
        manifest["artifacts"][key]["path"] = (
            f"releases/{output.name}/{artifact['path']}"
        )
    elif problem == "extra_microdata":
        manifest["artifacts"]["extra"] = {
            **manifest["artifacts"]["dataset"],
            "path": "extra.h5",
        }
        shutil.copyfile(root / "populace_us_2024.h5", output / "extra.h5")
    elif problem == "missing_parent":
        (output / "parent_build_manifest.json").unlink()
    else:
        del manifest["metadata"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError):
        enrichment.validate_source_enrichment_candidate(
            output,
            parent_h5=parent,
            artifact_root=root,
            require_compatibility=True,
            compatibility_wheels=(tmp_path / "country.whl",),
        )


def test_certification_writes_new_bundle_and_preflight_replays(
    candidate, tmp_path, monkeypatch
):
    import microcosm.data.release as release_module

    monkeypatch.setattr(
        release_module,
        "_hf_api",
        lambda: pytest.fail("successful preflight constructed a Hub client"),
    )
    release, parent, root = candidate
    original_report = (release / enrichment.SOURCE_ENRICHMENT_FILE).read_bytes()
    output, calls = _qualify_candidate(candidate, tmp_path, monkeypatch)
    assert (release / enrichment.SOURCE_ENRICHMENT_FILE).read_bytes() == original_report
    assert (output / enrichment.SOURCE_EVIDENCE_FILE).read_bytes() == (
        release / enrichment.SOURCE_EVIDENCE_FILE
    ).read_bytes()
    assert (
        publish_main(
            [
                str(output),
                "--parent-h5",
                str(parent),
                "--artifact-root",
                str(root),
                "--compatibility-wheel",
                str(tmp_path / "country.whl"),
                "--preflight-only",
            ]
        )
        == 0
    )
    assert len(calls) == 3  # Test, staged revalidation, then real publisher preflight.
    assert all(call["require_wheels"] for call in calls)


def test_certified_enrichment_publishes_complete_bundle_and_root_h5(
    candidate, tmp_path, monkeypatch
):
    from .test_release import FakeHub

    _, parent, root = candidate
    release, compatibility_calls = _qualify_candidate(candidate, tmp_path, monkeypatch)
    manifest = json.loads((release / "release_manifest.json").read_text())
    dataset = manifest["artifacts"]["dataset"]
    prefix = f"releases/{release.name}/"
    release_paths = {
        f"{prefix}{path.name}" for path in release.iterdir() if path.is_file()
    }
    assert f"{prefix}{dataset['path']}" not in release_paths
    hub = FakeHub()
    calls_before_publication = len(compatibility_calls)

    pointer = publish_release(
        release,
        "policyengine/populace-us",
        parent_h5=parent,
        artifact_root=root,
        compatibility_wheels=(tmp_path / "country.whl",),
        api=hub,
        notify=False,
    )

    assert len(compatibility_calls) == calls_before_publication + 1
    assert compatibility_calls[-1] == {
        "require_wheels": True,
        "compatibility_wheels": (tmp_path / "country.whl",),
    }
    commits = [event for kind, event in hub.events if kind == "create_commit"]
    assert [commit["revision"] for commit in commits] == [
        f"release-staging/{release.name}",
        "main",
    ]
    # Immutable staging and main promotion each upload the same bundle once.
    # Real file collection must deduplicate contract/manifest entries and keep
    # the H5 at the root rather than creating a second release-local copy.
    for commit in commits:
        paths = commit["paths"]
        assert len(paths) == len(set(paths))
        assert [path for path in paths if path.endswith(".h5")] == [dataset["path"]]
        assert set(paths) - {"latest.json"} == release_paths | {dataset["path"]}
    assert "latest.json" not in commits[0]["paths"]
    assert commits[1]["paths"][-1] == "latest.json"
    assert hub.tags == [{"tag": release.name, "revision": commits[0]["commit"]}]
    assert hub.events[-1] == ("create_commit", commits[1])

    expected_hashes = {
        entry["path"] if key == "dataset" else f"{prefix}{entry['path']}": entry[
            "sha256"
        ]
        for key, entry in manifest["artifacts"].items()
    }
    for path, expected_sha in expected_hashes.items():
        uploaded = [content for name, content in hub.uploads if name == path]
        assert len(uploaded) == 2  # Exactly once in each atomic commit.
        assert all(
            hashlib.sha256(content).hexdigest() == expected_sha for content in uploaded
        )
    assert hub.uploads[-1][0] == "latest.json"
    assert sum(path == "latest.json" for path, _ in hub.uploads) == 1
    assert json.loads(hub.uploads[-1][1]) == pointer
    assert pointer["release_id"] == release.name
    assert pointer["paths"] == {
        name: f"{prefix}{name}.json"
        for name in (
            "build_manifest",
            "release_manifest",
            "calibration_diagnostics",
            "us_source_coverage",
        )
    }


def test_certified_enrichment_preflight_and_publish_reject_tag_mismatch_identically(
    candidate, tmp_path, monkeypatch
):
    import microcosm.data.release as release_module

    _, parent, root = candidate
    release, _ = _qualify_candidate(candidate, tmp_path, monkeypatch)
    monkeypatch.setattr(
        release_module,
        "_hf_api",
        lambda: pytest.fail("tag mismatch constructed a Hub client"),
    )
    wheels = (tmp_path / "country.whl",)
    with pytest.raises(ValueError, match="tag_name must match") as published:
        publish_release(
            release,
            "policyengine/populace-us",
            parent_h5=parent,
            artifact_root=root,
            compatibility_wheels=wheels,
            tag_name="unmatched-tag",
            notify=False,
        )
    with pytest.raises(type(published.value)) as preflight:
        publish_main(
            [
                str(release),
                "--parent-h5",
                str(parent),
                "--artifact-root",
                str(root),
                "--compatibility-wheel",
                str(wheels[0]),
                "--tag-name",
                "unmatched-tag",
                "--preflight-only",
            ]
        )
    assert str(preflight.value) == str(published.value)


@pytest.mark.parametrize("duplicate", ["identical", "different", "symlink"])
@pytest.mark.parametrize(
    "entrypoint", ["candidate", "preflight", "publisher", "publisher_existing_client"]
)
def test_release_local_h5_duplicate_is_rejected_before_hub_activity(
    candidate, tmp_path, monkeypatch, duplicate, entrypoint
):
    import microcosm.data.release as release_module

    _, parent, root = candidate
    release, _ = _qualify_candidate(candidate, tmp_path, monkeypatch)
    # The unduplicated fixture passes the full gate: pending compatibility or
    # synthetic source identity must not accidentally account for rejection.
    validate_release_dir(release, parent_h5=parent, artifact_root=root)
    h5 = root / "populace_us_2024.h5"
    local = release / h5.name
    if duplicate == "symlink":
        local.symlink_to(h5)
    else:
        shutil.copyfile(h5, local)
        if duplicate == "different":
            with h5py.File(local, "r+") as handle:
                row = handle["person/table"][0]
                row["person_weight"] += 1
                handle["person/table"][0] = row
    monkeypatch.setattr(
        release_module,
        "_hf_api",
        lambda: pytest.fail("duplicate reached Hub client construction"),
    )

    class NoHubActivity:
        def __getattr__(self, name):
            pytest.fail(f"duplicate accessed supplied Hub client: {name}")

    monkeypatch.setattr(
        enrichment,
        "run_native_loader_compatibility",
        lambda *a, **kw: pytest.fail("duplicate reached compatibility probing"),
    )
    with pytest.raises(ReleaseContractError, match="release-local H5"):
        if entrypoint == "candidate":
            _validate((release, parent, root))
        elif entrypoint == "preflight":
            publish_main(
                [
                    str(release),
                    "--parent-h5",
                    str(parent),
                    "--artifact-root",
                    str(root),
                    "--preflight-only",
                ]
            )
        else:
            publish_release(
                release,
                "policyengine/populace-us",
                parent_h5=parent,
                artifact_root=root,
                notify=False,
                api=NoHubActivity()
                if entrypoint == "publisher_existing_client"
                else None,
            )


@pytest.mark.parametrize("year", [2023, 2024, 2025])
@pytest.mark.parametrize(
    "field", ["archive_sha256", "official_archive_url", "member", "income_year"]
)
def test_resealed_provenance_rejects_archive_identity_from_another_year(
    candidate, year, field
):
    other_year = 2023 if year != 2023 else 2024
    replacement = enrichment.CENSUS_ARCHIVE_PINS[other_year][field]
    _assert_resealed_archive_rejected(candidate, year, field, replacement)


@pytest.mark.parametrize("year", [2023, 2024, 2025])
def test_resealed_provenance_rejects_unpinned_archive_sha(candidate, year):
    _assert_resealed_archive_rejected(candidate, year, "archive_sha256", "0" * 64)


def _assert_resealed_archive_rejected(candidate, year, field, replacement):
    assert _validate(candidate)["compatibility"]["status"] == "pending"
    release, _, _ = candidate
    provenance_path = release / enrichment.SOURCE_PROVENANCE_FILE
    provenance = json.loads(provenance_path.read_text())
    row = next(row for row in provenance["source_checks"] if row["survey_year"] == year)
    row[field] = replacement
    _write(provenance_path, provenance)
    report_path = release / enrichment.SOURCE_ENRICHMENT_FILE
    report = json.loads(report_path.read_text())
    report["source"]["provenance_sha256"] = enrichment.sha256_file(provenance_path)
    report["reconciliation"] = provenance
    _write(report_path, report)
    _refresh(release, provenance_path.name)
    _refresh(release, report_path.name)
    with pytest.raises(
        ReleaseContractError, match=f"Census {year} pinned archive {field} differs"
    ):
        _validate(candidate)


def test_native_compatibility_requires_wheels_for_all_runtime_packages():
    with pytest.raises(ValueError, match="exact installed policyengine-us"):
        enrichment._runtime_package_identities((), require_wheels=True)


@pytest.mark.parametrize("distribution_name", ["policyengine-us", "spm-calculator"])
def test_wheel_identity_rejects_changed_installed_source(
    tmp_path, monkeypatch, distribution_name
):
    from importlib import metadata
    from zipfile import ZipFile

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 2\n")
    normalized = distribution_name.replace("-", "_")
    wheel = tmp_path / f"{normalized}-1.0-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{normalized}-1.0.dist-info/METADATA",
            f"Name: {distribution_name}\nVersion: 1.0\n",
        )
        archive.writestr("pkg/module.py", "VALUE = 1\n")

    class Installed:
        version = "1.0"
        files = ["pkg/module.py"]

        def read_text(self, name):
            return None

        def locate_file(self, relative):
            return tmp_path / relative

    monkeypatch.setattr(metadata, "distribution", lambda name: Installed())
    with pytest.raises(ValueError, match="installed source differs from wheel"):
        enrichment._runtime_package_identities((wheel,), require_wheels=False)


def test_three_wheels_cannot_omit_calculator(tmp_path, monkeypatch):
    monkeypatch.setattr(enrichment, "_wheel_files", lambda path: (path.stem, "1.0", {}))
    wheels = tuple(
        tmp_path / f"{name}.whl"
        for name in ("policyengine-us", "policyengine-core", "policyengine")
    )
    with pytest.raises(ValueError, match="spm-calculator wheels"):
        enrichment._runtime_package_identities(wheels, require_wheels=True)


def test_registered_native_role_is_owned_by_calculator_wheel(tmp_path, monkeypatch):
    import importlib
    from importlib import metadata
    from types import SimpleNamespace

    monkeypatch.setattr(
        metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda relative: tmp_path / relative),
    )
    imported = []

    def module(name):
        imported.append(name)
        return SimpleNamespace(__file__=str(tmp_path / name / "__init__.py"))

    monkeypatch.setattr(importlib, "import_module", module)
    paths = {
        "country_loader": tmp_path / "policyengine_us/data/dataset_schema.py",
        "wrapper_loader": tmp_path / "policyengine/tax_benefit_models/us/datasets.py",
        "native_role": tmp_path / "spm_calculator/policyengine_adapter.py",
    }
    enrichment._check_loaded_source_ownership(paths)
    assert set(imported) == {
        "policyengine_us",
        "policyengine_core",
        "policyengine",
        "spm_calculator",
    }
    # A same-named country class or source outside the verified calculator is
    # not the production calculator-owned primitive.
    for replacement in (
        tmp_path / "policyengine_us/native_role.py",
        tmp_path / "unverified_calculator/policyengine_adapter.py",
    ):
        with pytest.raises(ValueError, match="native_role.*spm-calculator wheel"):
            enrichment._check_loaded_source_ownership(
                paths | {"native_role": replacement}
            )


def test_evidence_tier_cannot_bypass_enrichment_gate(candidate):
    from microcosm.data.contract import validate_evidence_release_dir

    release, _, _ = candidate
    with pytest.raises(ReleaseContractError, match="evidence tier does not accept"):
        validate_evidence_release_dir(release)


def test_core_probe_rejects_silently_discarded_native_input(candidate, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    import pandas as pd

    _, _, root = candidate
    with pd.HDFStore(root / "populace_us_2024.h5", "r") as store:
        tables = {
            entity: store[entity]
            for entity in (
                "person",
                "household",
                "tax_unit",
                "spm_unit",
                "family",
                "marital_unit",
            )
        }
    for entity in ("tax_unit", "family", "marital_unit"):
        tables["person"][f"person_{entity}_id"] = tables["person"][
            "person_household_id"
        ]
    calls = []

    class IgnoresInput:
        def __init__(self, dataset):
            calls.append(dataset)
            self.dataset = dataset

        def calculate(self, variable, year):
            return np.zeros(len(self.dataset.person), dtype=bool)

    fake_country = ModuleType("policyengine_us")
    fake_country.Microsimulation = IgnoresInput
    fake_data = ModuleType("policyengine_us.data")
    fake_data.USSingleYearDataset = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "policyengine_us", fake_country)
    monkeypatch.setitem(sys.modules, "policyengine_us.data", fake_data)
    with pytest.raises(ValueError, match="discarded the supplied native person role"):
        enrichment._check_native_input_precedence(SimpleNamespace(**tables))
    assert len(calls) == 2
    assert all(len(dataset.person) == 2 for dataset in calls)
    assert all(
        dataset.person["person_spm_unit_id"].tolist() == [1, 1] for dataset in calls
    )


def test_producer_identity_rejects_fabricated_clean_commit():
    code = {
        "git_commit": "0" * 40,
        "git_dirty": False,
        "source_files_sha256": {
            name: "a" * 64 for name in enrichment.PRODUCER_SOURCE_FILES
        },
    }
    with pytest.raises(ValueError, match="cannot be authenticated"):
        enrichment._check_producer_source_identity(code)


def test_producer_identity_rejects_uncommitted_source_mutation(tmp_path, monkeypatch):
    import hashlib
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[3]
    commit = "a" * 40
    sources = {
        name: (root / name).read_bytes() for name in enrichment.PRODUCER_SOURCE_FILES
    }
    for name, content in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    changed = tmp_path / enrichment.PRODUCER_SOURCE_FILES[0]
    changed.write_bytes(changed.read_bytes() + b"\n# uncommitted source edit\n")
    code = {
        "git_commit": commit,
        "git_dirty": False,
        "source_files_sha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sources.items()
        },
    }

    def git_result(args, **kwargs):
        if args[1:] == ["rev-parse", "--show-toplevel"]:
            return SimpleNamespace(stdout=str(tmp_path).encode())
        if args[1] == "rev-parse":
            return SimpleNamespace(stdout=commit.encode())
        return SimpleNamespace(stdout=sources[args[2].split(":", 1)[1]])

    monkeypatch.setattr(subprocess, "run", git_result)
    with pytest.raises(ValueError, match="checkout source differs"):
        enrichment._check_producer_source_identity(code)


DECLARED_RANGE = "policyengine-us>=1.999.0,<2"
DECLARED_BY = "PolicyEngine data release owner"
DECLARED_ENTRY = {
    "name": "policyengine-us",
    "specifier": ">=1.999.0,<2",
    "basis": enrichment.PUBLISHER_CLAIM_BASIS,
    "declared_by": DECLARED_BY,
}


def _declare(**overrides):
    return {
        "compatible_model_specifier": DECLARED_RANGE,
        "compatibility_claim_declared_by": DECLARED_BY,
        **overrides,
    }


def _certified(output):
    return (
        json.loads((output / "release_manifest.json").read_text()),
        json.loads((output / enrichment.SOURCE_ENRICHMENT_FILE).read_text()),
    )


def test_default_certification_pins_the_tested_version_and_declares_nothing(
    candidate, tmp_path, monkeypatch
):
    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch)
    manifest, report = _certified(output)
    assert manifest["compatible_model_packages"] == [
        {"name": "policyengine-us", "specifier": "==1.999.0"}
    ]
    assert manifest["compatible_core_packages"] == [
        {"name": "policyengine-core", "specifier": "==3.99.0"}
    ]
    assert "publisher_claims" not in report["compatibility"]


def test_declared_model_range_is_emitted_verbatim_and_replayed_by_preflight(
    candidate, tmp_path, monkeypatch
):
    import microcosm.data.release as release_module

    monkeypatch.setattr(
        release_module,
        "_hf_api",
        lambda: pytest.fail("successful preflight constructed a Hub client"),
    )
    release, parent, root = candidate
    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    manifest, report = _certified(output)
    assert manifest["compatible_model_packages"] == [DECLARED_ENTRY]
    assert report["compatibility"]["publisher_claims"] == {"model": DECLARED_ENTRY}
    # The claim widens the binding; it never restates what was measured.
    assert manifest["build"]["built_with_model_package"] == {
        "name": "policyengine-us",
        "version": "1.999.0",
    }
    assert manifest["compatible_core_packages"] == [
        {"name": "policyengine-core", "specifier": "==3.99.0"}
    ]
    assert (
        publish_main(
            [
                str(output),
                "--parent-h5",
                str(parent),
                "--artifact-root",
                str(root),
                "--compatibility-wheel",
                str(tmp_path / "country.whl"),
                "--preflight-only",
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    ("specifier", "message"),
    [
        ("policyengine-us>=2.0,<3", "excludes the tested policyengine-us version"),
        ("policyengine-uk>=1.999.0,<2", "declares compatibility for the built-with"),
        ("policyengine-us>=1.999.0", "reaches 2.0.0 and beyond"),
        ("policyengine-us>=1.999.0,!=99999", "reaches 2.0.0 and beyond"),
        ("policyengine-us>=1.999.0,<99998", "reaches 2.0.0 and beyond"),
        # Excluding exactly the next major walks past that probe while still
        # certifying every release after it, which the far-future probe catches.
        ("policyengine-us>=1.999.0,!=2.0.0", "still admits 99999.0.0"),
        # Bounded above and open below: it covers the tested version and every
        # release that ever preceded it, back to the first.
        ("policyengine-us<2", "must also state a lower bound"),
        ("policyengine-us", "needs a PEP 440 specifier"),
        ("policyengine-us[us]>=1.999.0,<2", "bare name and specifier"),
        (
            'policyengine-us>=1.999.0,<2; python_version>"3"',
            "bare name and specifier",
        ),
        (
            "policyengine-us@https://example.invalid/pe.whl",
            "bare name and specifier",
        ),
        ("policyengine-us>=oops", "is not a PEP 508 requirement"),
    ],
)
def test_certification_refuses_an_unsound_claim(
    candidate, tmp_path, monkeypatch, specifier, message
):
    with pytest.raises(ValueError, match=message):
        _qualify_candidate(
            candidate,
            tmp_path,
            monkeypatch,
            **_declare(compatible_model_specifier=specifier),
        )


@pytest.mark.parametrize("specifier", [None, "", "   ", ","])
def test_a_report_claim_without_a_usable_specifier_is_refused(specifier):
    """The entry builder shares the parser's rules, reached from the report side.

    A claim read back out of a certified report never passes through
    :func:`parse_compatibility_claim_requirement`, so the one specifier
    validator has to hold on this path too.
    """
    with pytest.raises(ValueError, match="needs a PEP 440 specifier|must constrain"):
        enrichment.compatibility_claim_entry(
            specifier,
            package="policyengine-us",
            version="1.999.0",
            declared_by=DECLARED_BY,
        )


def test_a_whole_major_range_is_accepted_on_purpose():
    """`>=2.0.1,<3` is bounded at the next major, and the guard allows it.

    Not an oversight and not a probe that missed: the tooling's line is the next
    major version, while the runbook recommends bounding at the next minor. A
    reader who finds this range certified is looking at a deliberate ceiling.
    """
    assert enrichment.compatibility_claim_entry(
        ">=2.0.1,<3",
        package="policyengine-us",
        version="2.0.1",
        declared_by=DECLARED_BY,
    ) == {
        "name": "policyengine-us",
        "specifier": ">=2.0.1,<3",
        "basis": enrichment.PUBLISHER_CLAIM_BASIS,
        "declared_by": DECLARED_BY,
    }


@pytest.mark.parametrize(
    "specifier", [">=2.0.1,<2.1", "~=2.0.1", "==2.0.*", ">=2.0.1,<3"]
)
def test_a_bounded_claim_over_a_2_0_1_build_is_accepted(specifier):
    assert (
        enrichment.compatibility_claim_entry(
            specifier,
            package="policyengine-us",
            version="2.0.1",
            declared_by=DECLARED_BY,
        )["specifier"]
        == specifier
    )


@pytest.mark.parametrize(
    "specifier",
    [
        ">=2.0.1",
        ">=2.0.1,<3.0.1",
        # Excludes the next major by name, and certifies 4.x and 5.x anyway.
        ">=2.0.1,!=3.0.0",
        # Bounded above, open below: it certifies every release back to the
        # first one ever made, including versions predating the loader path
        # certification measures.
        "<2.1",
        "<=2.0.5",
    ],
)
def test_an_unbounded_claim_over_a_2_0_1_build_is_refused(specifier):
    with pytest.raises(
        ValueError,
        match="reaches 3.0.0|still admits 99999.0.0|must also state a lower bound",
    ):
        enrichment.compatibility_claim_entry(
            specifier,
            package="policyengine-us",
            version="2.0.1",
            declared_by=DECLARED_BY,
        )


@pytest.mark.parametrize(
    ("specifier", "admits"),
    [
        # Excludes the next major and the far-future probe by name, and
        # certifies 4.x and 5.x anyway.
        (">=2.0.1,!=3.0.0,!=99999.0.0", "5.0"),
        # The exact mirror below: excludes the zero probe by name, and
        # certifies the 0.x releases the lower bound exists to keep out.
        ("<2.1,!=0", "0.9.0"),
    ],
)
def test_the_documented_probe_residue_is_still_exactly_that(specifier, admits):
    """Characterization: probes bound a claim, they do not prove one bounded.

    The runbook says a specifier that names the probe versions and excludes
    them passes while admitting others, above and below alike. Pinning both
    mirrors keeps that sentence honest and stops the residue widening past
    what is written down. Passes before and after the lower-bound probe; it
    describes the guard's stated limit rather than a change to it.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    assert (
        enrichment.compatibility_claim_entry(
            specifier,
            package="policyengine-us",
            version="2.0.1",
            declared_by=DECLARED_BY,
        )["specifier"]
        == specifier
    )
    assert Version(admits) in SpecifierSet(specifier)


def test_the_lower_bound_probe_sits_at_the_tested_version_epoch():
    """A claim may mix epochs, and only a probe at the tested epoch catches it.

    `>=2.0.1,<1!2.1` over a `1!2.0.1` build is open below within epoch 1 — it
    admits `1!0` — while excluding a bare `Version("0")`, which sorts under
    every epoch-1 release. A probe at plain zero would accept it. (`<1!2.1` on
    its own is refused either way, so it is not the case that pins the carry.)
    """
    with pytest.raises(ValueError, match="must also state a lower bound"):
        enrichment.compatibility_claim_entry(
            ">=2.0.1,<1!2.1",
            package="policyengine-us",
            version="1!2.0.1",
            declared_by=DECLARED_BY,
        )
    assert (
        enrichment.compatibility_claim_entry(
            ">=1!2.0.1,<1!2.1",
            package="policyengine-us",
            version="1!2.0.1",
            declared_by=DECLARED_BY,
        )["specifier"]
        == ">=1!2.0.1,<1!2.1"
    )


@pytest.mark.parametrize(
    ("specifier", "accepted"),
    [
        # A prerelease sorts below its own release, so the ranges written the
        # usual way exclude the build they were written for. Not a prerelease
        # exclusion: `packaging` matches prereleases by default, following
        # PEP 440's recommendation, and `SpecifierSet.contains` says so.
        (">=2.0.1,<2.1", False),
        ("~=2.0.1", False),
        # Naming the prerelease, or matching the series with a prefix, works.
        (">=2.0.1rc1,<2.1", True),
        ("==2.0.*", True),
        # And the default pin certification writes with no options always does.
        ("==2.0.1rc1", True),
    ],
)
def test_a_prerelease_build_takes_a_range_only_if_the_range_reaches_it(
    specifier, accepted
):
    """Characterization: what a claim over a prerelease build can say.

    Pins the behaviour the runbook now describes. Both outcomes are ordering,
    not a prerelease rule, so a `packaging` release that changed either would
    fail here rather than silently rewrite the runbook.
    """

    def claim():
        return enrichment.compatibility_claim_entry(
            specifier,
            package="policyengine-us",
            version="2.0.1rc1",
            declared_by=DECLARED_BY,
        )

    if accepted:
        assert claim()["specifier"] == specifier
    else:
        with pytest.raises(ValueError, match="excludes the tested"):
            claim()


@pytest.mark.parametrize("declared_by", [None, "   ", "x" * 201, "two\nlines"])
def test_claim_must_record_an_accountable_declarer(
    candidate, tmp_path, monkeypatch, declared_by
):
    with pytest.raises(ValueError, match="who declared it|accountable for it"):
        _qualify_candidate(
            candidate,
            tmp_path,
            monkeypatch,
            **_declare(compatibility_claim_declared_by=declared_by),
        )


def test_declarer_without_a_specifier_is_refused(candidate, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="accountable for it"):
        _qualify_candidate(
            candidate,
            tmp_path,
            monkeypatch,
            **_declare(compatible_model_specifier=None),
        )


def test_manifest_widened_after_certification_has_no_declaration_to_stand_on(
    candidate, tmp_path, monkeypatch
):
    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch)
    _, parent, root = candidate
    path = output / "release_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["compatible_model_packages"] = [DECLARED_ENTRY]
    _write(path, manifest)
    with pytest.raises(
        ReleaseContractError, match="must pin exactly the tested version unless"
    ):
        enrichment.validate_source_enrichment_candidate(
            output,
            parent_h5=parent,
            artifact_root=root,
            require_compatibility=True,
            compatibility_wheels=(tmp_path / "country.whl",),
        )


def test_manifest_must_match_the_claim_the_report_declares(
    candidate, tmp_path, monkeypatch
):
    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    _, parent, root = candidate
    path = output / "release_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["compatible_model_packages"] = [
        {**DECLARED_ENTRY, "specifier": ">=1.999.0,<3"}
    ]
    _write(path, manifest)
    with pytest.raises(ReleaseContractError, match="must match the declared publisher"):
        enrichment.validate_source_enrichment_candidate(
            output,
            parent_h5=parent,
            artifact_root=root,
            require_compatibility=True,
            compatibility_wheels=(tmp_path / "country.whl",),
        )


def test_report_claim_widened_after_certification_is_revalidated(
    candidate, tmp_path, monkeypatch
):
    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    _, parent, root = candidate
    unbounded = {**DECLARED_ENTRY, "specifier": ">=1.999.0"}
    report_path = output / enrichment.SOURCE_ENRICHMENT_FILE
    report = json.loads(report_path.read_text())
    report["compatibility"]["publisher_claims"]["model"] = unbounded
    _write(report_path, report)
    manifest_path = output / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["compatible_model_packages"] = [unbounded]
    _write(manifest_path, manifest)
    _refresh(output, enrichment.SOURCE_ENRICHMENT_FILE)
    with pytest.raises(ReleaseContractError, match="reaches 2.0.0 and beyond"):
        enrichment.validate_source_enrichment_candidate(
            output,
            parent_h5=parent,
            artifact_root=root,
            require_compatibility=True,
            compatibility_wheels=(tmp_path / "country.whl",),
        )


def test_report_claim_that_constrains_nothing_is_refused(
    candidate, tmp_path, monkeypatch
):
    """``,`` parses as an empty specifier set, so the validator refuses it."""
    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    _, parent, root = candidate
    empty = {**DECLARED_ENTRY, "specifier": ","}
    report_path = output / enrichment.SOURCE_ENRICHMENT_FILE
    report = json.loads(report_path.read_text())
    report["compatibility"]["publisher_claims"]["model"] = empty
    _write(report_path, report)
    manifest_path = output / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["compatible_model_packages"] = [empty]
    _write(manifest_path, manifest)
    _refresh(output, enrichment.SOURCE_ENRICHMENT_FILE)
    with pytest.raises(ReleaseContractError, match="must constrain the version"):
        enrichment.validate_source_enrichment_candidate(
            output,
            parent_h5=parent,
            artifact_root=root,
            require_compatibility=True,
            compatibility_wheels=(tmp_path / "country.whl",),
        )


@pytest.mark.parametrize(
    "claims",
    [
        [],
        {},
        {"bogus": DECLARED_ENTRY},
        "model",
        # Core is not a field a publisher may widen, so a forged core claim is
        # refused rather than honoured: origin/main pinned Core unconditionally.
        {"core": {**DECLARED_ENTRY, "name": "policyengine-core"}},
        {"model": DECLARED_ENTRY, "core": DECLARED_ENTRY},
    ],
)
def test_malformed_publisher_claims_block_certification_readback(
    candidate, tmp_path, monkeypatch, claims
):
    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    _, parent, root = candidate
    report_path = output / enrichment.SOURCE_ENRICHMENT_FILE
    report = json.loads(report_path.read_text())
    report["compatibility"]["publisher_claims"] = claims
    _write(report_path, report)
    _refresh(output, enrichment.SOURCE_ENRICHMENT_FILE)
    with pytest.raises(ReleaseContractError, match="publisher_claims must map"):
        enrichment.validate_source_enrichment_candidate(
            output,
            parent_h5=parent,
            artifact_root=root,
            require_compatibility=True,
            compatibility_wheels=(tmp_path / "country.whl",),
        )


def test_pending_candidate_cannot_declare_a_publisher_claim(candidate):
    release, _, _ = candidate
    path = release / enrichment.SOURCE_ENRICHMENT_FILE
    report = json.loads(path.read_text())
    report["compatibility"]["publisher_claims"] = {"model": DECLARED_ENTRY}
    _write(path, report)
    _refresh(release, enrichment.SOURCE_ENRICHMENT_FILE)
    with pytest.raises(
        ReleaseContractError, match="pending source enrichment must not declare"
    ):
        _validate(candidate)


def test_declared_range_certifies_a_later_patch_for_both_readers(
    candidate, tmp_path, monkeypatch
):
    """The point of the range: the consumers accept a version it covers."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    from microcosm.data.loader import _package_certification

    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    manifest, _ = _certified(output)
    certification = _package_certification(
        manifest,
        field="compatible_model_packages",
        package_name="policyengine-us",
        built_field="built_with_model_package",
        release_id=output.name,
    )
    assert certification.specifiers == (">=1.999.0,<2",)
    assert certification.built_version == "1.999.0"

    def wrapper_specifier_matches(version, specifier):
        # policyengine.provenance.manifest._specifier_matches, mirrored.
        return Version(version) in SpecifierSet(specifier)

    claim = manifest["compatible_model_packages"][0]["specifier"]
    assert wrapper_specifier_matches("1.999.1", claim)
    assert wrapper_specifier_matches("1.999.0", claim)
    assert not wrapper_specifier_matches("2.0.0", claim)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--compatible-model-specifier", DECLARED_RANGE], "declared together"),
        (["--compatibility-claim-declared-by", DECLARED_BY], "declared together"),
        (
            [
                "--compatible-model-specifier",
                DECLARED_RANGE,
                "--compatibility-claim-declared-by",
                DECLARED_BY,
            ],
            "applies to --certify",
        ),
    ],
)
def test_cli_refuses_a_half_declared_or_uncertified_claim(
    candidate, capsys, argv, message
):
    release, parent, root = candidate
    with pytest.raises(SystemExit):
        enrichment.main(
            [
                "--release-dir",
                str(release),
                "--parent-h5",
                str(parent),
                "--artifact-root",
                str(root),
                *argv,
            ]
        )
    assert message in capsys.readouterr().err


def test_cli_certify_declares_the_claim_it_was_given(
    candidate, tmp_path, monkeypatch, capsys
):
    from importlib import metadata

    release, parent, root = candidate
    monkeypatch.setattr(
        enrichment,
        "run_native_loader_compatibility",
        lambda *args, **kwargs: {
            "status": "passed",
            "dataset_sha256": enrichment.sha256_file(root / "populace_us_2024.h5"),
            "packages": {
                "policyengine-us": {"version": "1.999.0"},
                "policyengine-core": {"version": "3.99.0"},
                "policyengine": {"version": "5.99.0"},
                "spm-calculator": {"version": "1.0.0"},
            },
        },
    )
    monkeypatch.setattr(metadata, "version", lambda name: "0.1.0")
    monkeypatch.setattr(
        enrichment, "_check_producer_source_identity", lambda code: None
    )
    output = tmp_path / "certified" / release.name
    assert (
        enrichment.main(
            [
                "--certify",
                "--release-dir",
                str(release),
                "--output-dir",
                str(output),
                "--parent-h5",
                str(parent),
                "--artifact-root",
                str(root),
                "--compatibility-wheel",
                str(tmp_path / "country.whl"),
                "--compatible-model-specifier",
                DECLARED_RANGE,
                "--compatibility-claim-declared-by",
                DECLARED_BY,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "certified_bundle": str(output),
        "published": False,
    }
    manifest, report = _certified(output)
    assert manifest["compatible_model_packages"] == [DECLARED_ENTRY]
    assert report["compatibility"]["publisher_claims"] == {"model": DECLARED_ENTRY}


def test_the_certify_cli_reports_a_narrowing_it_just_caused(
    candidate, tmp_path, monkeypatch, capsys
):
    """The run that narrows says so in its verdict, not only in a warning.

    Its `RuntimeWarning` is the live signal, and it is the one thing this
    reporting path cannot rely on: stderr under CI, or nothing at all under
    `PYTHONWARNINGS=ignore`. All three verdicts carry the record, so whatever
    captures stdout has it too.
    """
    _, parent, root = candidate
    declared, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    capsys.readouterr()
    output = tmp_path / "recertified-cli" / declared.name
    with pytest.warns(RuntimeWarning, match="narrows the policyengine-us"):
        assert (
            enrichment.main(
                [
                    "--certify",
                    "--release-dir",
                    str(declared),
                    "--output-dir",
                    str(output),
                    "--parent-h5",
                    str(parent),
                    "--artifact-root",
                    str(root),
                    "--compatibility-wheel",
                    str(tmp_path / "country.whl"),
                ]
            )
            == 0
        )
    assert json.loads(capsys.readouterr().out) == {
        "certified_bundle": str(output),
        "published": False,
        "narrowed_claims": NARROWED_RECORD,
    }


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        ([], {}),
        ("not a mapping", {}),
        ({}, {}),
        ({"compatibility": 3}, {}),
        ({"compatibility": {}}, {}),
        ({"compatibility": {"narrowed_claims": [1, 2]}}, {}),
        ({"compatibility": {"narrowed_claims": "model"}}, {}),
        ({"compatibility": {"narrowed_claims": {"model": {}}}}, {"model": {}}),
    ],
)
def test_both_narrowing_readers_agree_on_what_counts_as_a_record(
    tmp_path, report, expected
):
    """One tolerance, so the three verdicts cannot disagree about one bundle.

    The validation path has the report in hand and the publisher paths read it
    off disk; if they applied different rules, a malformed record would show up
    in one verdict and not the others.
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / enrichment.SOURCE_ENRICHMENT_FILE).write_text(json.dumps(report))
    assert enrichment._narrowed_claims(report) == expected
    assert enrichment.recorded_narrowed_claims(bundle) == expected


def test_core_stays_pinned_when_the_model_range_is_declared(
    candidate, tmp_path, monkeypatch
):
    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    manifest, report = _certified(output)
    assert manifest["compatible_core_packages"] == [
        {"name": "policyengine-core", "specifier": "==3.99.0"}
    ]
    assert set(report["compatibility"]["publisher_claims"]) == {"model"}


def test_parenthesised_requirement_records_the_bare_specifier(
    candidate, tmp_path, monkeypatch
):
    output, _ = _qualify_candidate(
        candidate,
        tmp_path,
        monkeypatch,
        **_declare(compatible_model_specifier="policyengine-us (>=1.999.0,<2)"),
    )
    manifest, _ = _certified(output)
    assert manifest["compatible_model_packages"] == [DECLARED_ENTRY]


def _recertify(source, candidate, tmp_path, monkeypatch, name, **claim):
    """Certify an already-certified bundle again, as a re-release would."""
    _, parent, root = candidate
    output = tmp_path / name / source.name
    enrichment.certify_source_enrichment(
        source,
        output,
        parent_h5=parent,
        artifact_root=root,
        compatibility_wheels=(tmp_path / "country.whl",),
        **claim,
    )
    return output


def test_recertifying_without_the_flag_warns_that_it_narrows_the_claim(
    candidate, tmp_path, monkeypatch
):
    """A declared range must not vanish into an exact pin without a word."""
    declared, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    with pytest.warns(RuntimeWarning, match="narrows the policyengine-us") as caught:
        narrowed = _recertify(declared, candidate, tmp_path, monkeypatch, "recertified")
    assert REMEDIATION in _narrowing_warning(caught, "policyengine-us")
    manifest, report = _certified(narrowed)
    assert manifest["compatible_model_packages"] == [
        {"name": "policyengine-us", "specifier": "==1.999.0"}
    ]
    # Recorded in the bundle too, so the narrowing survives the terminal that
    # printed the warning.
    assert report["compatibility"]["narrowed_claims"] == {
        "model": {
            "previous_specifiers": [">=1.999.0,<2"],
            "emitted_specifier": "==1.999.0",
            "first_version_no_longer_covered": "1.999.1",
        }
    }


REMEDIATION = "Pass --compatible-model-specifier"


def _narrowing_warning(caught, package):
    """The one narrowing warning ``caught`` holds for ``package``."""
    messages = [
        str(entry.message)
        for entry in caught
        if f"the {package} compatibility" in str(entry.message)
    ]
    assert len(messages) == 1, messages
    return messages[0]


def test_the_flags_remediation_is_offered_only_when_the_flags_were_missing(
    candidate, tmp_path, monkeypatch
):
    """Advice to pass the flags is noise to the run that just passed them.

    Re-certifying with a tighter range — dropping a patch release found bad
    after the fact — is a legitimate narrowing, and the warning naming what it
    gives up is the point. The remediation belongs to the run that forgot the
    options, not the one that used them.
    """
    wider, _ = _qualify_candidate(
        candidate,
        tmp_path,
        monkeypatch,
        **_declare(compatible_model_specifier="policyengine-us>=1.998.0,<2"),
    )
    with pytest.warns(RuntimeWarning) as forgot:
        _recertify(wider, candidate, tmp_path, monkeypatch, "reverted")
    assert REMEDIATION in _narrowing_warning(forgot, "policyengine-us")

    with pytest.warns(RuntimeWarning) as retightened:
        _recertify(wider, candidate, tmp_path, monkeypatch, "tightened", **_declare())
    message = _narrowing_warning(retightened, "policyengine-us")
    assert "1.998.0" in message
    assert REMEDIATION not in message


def test_recertifying_with_the_same_range_narrows_nothing(
    candidate, tmp_path, monkeypatch
):
    declared, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        again = _recertify(
            declared, candidate, tmp_path, monkeypatch, "recertified", **_declare()
        )
    assert not [entry for entry in caught if "narrows" in str(entry.message)]
    manifest, report = _certified(again)
    assert manifest["compatible_model_packages"] == [DECLARED_ENTRY]
    assert "narrowed_claims" not in report["compatibility"]


LOST_CORE = {
    "previous_specifiers": ["==3.99.0"],
    "emitted_specifier": "==3.100.0",
    "first_version_no_longer_covered": "3.99.0",
}


def test_a_moved_core_pin_is_reported_as_a_pin_not_a_claim():
    """Core has no claim to narrow: `CLAIM_FIELD` is `model` and only `model`.

    The loop that names lost coverage runs over Core too, and it should — if a
    bundle's Core pin ever moved it would drop every consumer on the old one.
    But Core's entry is always the exact tested pin, so calling that a narrowed
    "claim" would name a thing no producer can declare.
    """
    core = enrichment._narrowing_notice(
        "core", "policyengine-core", LOST_CORE, offer_flags=False
    )
    assert core == (
        "certification moves the policyengine-core compatibility pin this "
        "bundle already carried: ==3.99.0 covered 3.99.0 and the ==3.100.0 "
        "this run emits does not."
    )
    assert "claim" not in core
    model = enrichment._narrowing_notice(
        enrichment.CLAIM_FIELD,
        "policyengine-us",
        {
            "previous_specifiers": [">=1.999.0,<2"],
            "emitted_specifier": "==1.999.0",
            "first_version_no_longer_covered": "1.999.1",
        },
        offer_flags=True,
    )
    assert model.startswith("certification narrows the policyengine-us ")
    assert model.endswith(
        REMEDIATION
        + " with --compatibility-claim-declared-by to keep a declared range."
    )


def test_a_moved_core_runtime_is_refused_before_certification_narrows_anything(
    candidate, tmp_path, monkeypatch
):
    """Why the Core branch above has no end-to-end test: it cannot be reached.

    Re-certification validates the input bundle first, and that gate re-runs
    the loader qualification and requires the recorded receipt to equal the
    current runtime. A Core version that moved fails there, before the emitted
    pin could differ from the one the bundle carries. The Core branch stays in
    the loop as defence in depth — a silent revert is what the warning exists
    to prevent — but this is the wall it sits behind.
    """
    _, parent, root = candidate
    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch)
    monkeypatch.setattr(
        enrichment,
        "run_native_loader_compatibility",
        lambda *args, **kwargs: {
            "status": "passed",
            "dataset_sha256": enrichment.sha256_file(root / "populace_us_2024.h5"),
            "packages": {
                "policyengine-us": {"version": "1.999.0"},
                "policyengine-core": {"version": "3.100.0"},
                "policyengine": {"version": "5.99.0"},
                "spm-calculator": {"version": "1.0.0"},
            },
        },
    )
    with pytest.raises(ReleaseContractError, match="receipt differs from actual"):
        _recertify(output, candidate, tmp_path, monkeypatch, "newer-core")


def test_first_certification_narrows_nothing(candidate, tmp_path, monkeypatch):
    """A pending candidate declares no compatibility, so there is none to lose."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch)
    assert not [entry for entry in caught if "narrows" in str(entry.message)]
    _, report = _certified(output)
    assert "narrowed_claims" not in report["compatibility"]


NARROWED_RECORD = {
    "model": {
        "previous_specifiers": [">=1.999.0,<2"],
        "emitted_specifier": "==1.999.0",
        "first_version_no_longer_covered": "1.999.1",
    }
}


def _narrowed_bundle(candidate, tmp_path, monkeypatch):
    """A certified bundle whose report records a narrowing, and a clean one."""
    declared, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    with pytest.warns(RuntimeWarning, match="narrows the policyengine-us"):
        narrowed = _recertify(declared, candidate, tmp_path, monkeypatch, "recertified")
    return narrowed, declared


def test_validation_output_surfaces_a_recorded_narrowing(
    candidate, tmp_path, monkeypatch, capsys
):
    """The record has to reach someone after the terminal that printed it.

    `certify_source_enrichment` warns on stderr, where CI noise or
    `PYTHONWARNINGS=ignore` buries it, and records `narrowed_claims` in the
    bundle. Validation is the next gate a later operator runs, so it reads the
    record back instead of reporting only `passed`.
    """
    _, parent, root = candidate
    narrowed, declared = _narrowed_bundle(candidate, tmp_path, monkeypatch)

    def _validate(bundle):
        assert (
            enrichment.main(
                [
                    "--release-dir",
                    str(bundle),
                    "--parent-h5",
                    str(parent),
                    "--artifact-root",
                    str(root),
                ]
            )
            == 0
        )
        return json.loads(capsys.readouterr().out)

    assert _validate(narrowed) == {
        "valid": True,
        "compatibility": "passed",
        "narrowed_claims": NARROWED_RECORD,
    }
    # A bundle that gave nothing up says nothing, so the key's presence is the
    # signal rather than an empty object every run has to read past.
    assert _validate(declared) == {"valid": True, "compatibility": "passed"}


def test_publish_preflight_surfaces_a_recorded_narrowing(
    candidate, tmp_path, monkeypatch, capsys
):
    """The publisher's own preflight is where the later operator actually is."""
    import microcosm.data.release as release_module

    monkeypatch.setattr(
        release_module,
        "_hf_api",
        lambda: pytest.fail("successful preflight constructed a Hub client"),
    )
    _, parent, root = candidate
    narrowed, declared = _narrowed_bundle(candidate, tmp_path, monkeypatch)

    def _preflight(bundle):
        assert (
            publish_main(
                [
                    str(bundle),
                    "--parent-h5",
                    str(parent),
                    "--artifact-root",
                    str(root),
                    "--compatibility-wheel",
                    str(tmp_path / "country.whl"),
                    "--preflight-only",
                ]
            )
            == 0
        )
        return json.loads(capsys.readouterr().out)

    assert _preflight(narrowed) == {
        "valid": True,
        "published": False,
        "narrowed_claims": NARROWED_RECORD,
    }
    assert _preflight(declared) == {"valid": True, "published": False}


def test_publication_says_on_stderr_what_the_preflight_says_in_json(
    candidate, tmp_path, monkeypatch, capsys
):
    """Publication is reachable without ever running the preflight.

    `tools/publish_release.sh` passes its arguments straight through, so the
    runbook's "remove --preflight-only" step is a habit rather than a gate. The
    JSON verdict belongs to whatever parses stdout; the operator reading the
    terminal gets the same record on stderr, on both paths.
    """
    from microcosm.data import publish_cli

    _, parent, root = candidate
    narrowed, declared = _narrowed_bundle(candidate, tmp_path, monkeypatch)
    monkeypatch.setattr(
        publish_cli, "publish_release", lambda *args, **kwargs: {"published": True}
    )
    tail = [
        "--parent-h5",
        str(parent),
        "--artifact-root",
        str(root),
        "--compatibility-wheel",
        str(tmp_path / "country.whl"),
    ]
    assert publish_main([str(narrowed), *tail]) == 0
    published = capsys.readouterr()
    assert json.loads(published.out) == {"published": True}
    assert "compatibility narrowing" in published.err
    assert ">=1.999.0,<2" in published.err

    assert publish_main([str(declared), *tail]) == 0
    assert "narrowing" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "claim",
    [
        {"compatible_model_specifier": "policyengine-uk>=1.999.0,<2"},
        {"compatibility_claim_declared_by": "  "},
        # A bare package name carries no specifier at all, which is checkable
        # without the tested version and so must be refused at the door.
        {"compatible_model_specifier": "policyengine-us"},
    ],
)
def test_an_unsound_claim_costs_no_qualification_run(
    candidate, tmp_path, monkeypatch, claim
):
    """The checks that need no tested version run before the long probe."""
    from importlib import metadata

    release, parent, root = candidate
    monkeypatch.setattr(
        enrichment,
        "run_native_loader_compatibility",
        lambda *args, **kwargs: pytest.fail("an unsound claim ran qualification"),
    )
    monkeypatch.setattr(
        enrichment,
        "validate_source_enrichment_candidate",
        lambda *args, **kwargs: pytest.fail(
            "an unsound claim ran candidate validation"
        ),
    )
    monkeypatch.setattr(metadata, "version", lambda name: "0.1.0")
    output = tmp_path / "certified" / release.name
    with pytest.raises(ValueError):
        enrichment.certify_source_enrichment(
            release,
            output,
            parent_h5=parent,
            artifact_root=root,
            compatibility_wheels=(tmp_path / "country.whl",),
            **_declare(**claim),
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "forged",
    [
        {**DECLARED_ENTRY, "note": "approved verbally"},
        {**DECLARED_ENTRY, "name": "policyengine_us"},
        {k: v for k, v in DECLARED_ENTRY.items() if k != "basis"},
    ],
)
def test_declared_claim_carries_only_the_validated_fields(
    candidate, tmp_path, monkeypatch, forged
):
    """A report entry the validator cannot rebuild exactly is refused."""
    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch, **_declare())
    _, parent, root = candidate
    report_path = output / enrichment.SOURCE_ENRICHMENT_FILE
    report = json.loads(report_path.read_text())
    report["compatibility"]["publisher_claims"]["model"] = forged
    _write(report_path, report)
    # The manifest keeps the canonical entry, so only the report-shape guard
    # can refuse this: the manifest-vs-report comparison is satisfied.
    _refresh(output, enrichment.SOURCE_ENRICHMENT_FILE)
    assert json.loads((output / "release_manifest.json").read_text())[
        "compatible_model_packages"
    ] == [DECLARED_ENTRY]
    with pytest.raises(ReleaseContractError, match="must record only"):
        enrichment.validate_source_enrichment_candidate(
            output,
            parent_h5=parent,
            artifact_root=root,
            require_compatibility=True,
            compatibility_wheels=(tmp_path / "country.whl",),
        )


def test_producer_validation_is_not_the_tamper_control(
    candidate, tmp_path, monkeypatch
):
    """A coordinated report+manifest edit validates, and that is not the guard.

    The report's only integrity anchor is its ``artifacts`` entry in the very
    manifest it authenticates, so an editor who rewrites both and restamps the
    hash produces a bundle this validator accepts — exactly as, before publisher
    claims existed, one who rewrote the exact pin did. What stands between an
    edited bundle and the Hub is :func:`_check_producer_source_identity`, the
    publish preflight, and the human publication decision. Pinned here so a
    later reader does not mistake the cross-check for a seal.
    """
    output, _ = _qualify_candidate(candidate, tmp_path, monkeypatch)
    _, parent, root = candidate
    widened = {
        "name": "policyengine-us",
        "specifier": ">=1.999.0,<2",
        "basis": enrichment.PUBLISHER_CLAIM_BASIS,
        "declared_by": "nobody who ran certification",
    }
    report_path = output / enrichment.SOURCE_ENRICHMENT_FILE
    report = json.loads(report_path.read_text())
    report["compatibility"]["publisher_claims"] = {"model": widened}
    _write(report_path, report)
    manifest_path = output / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["compatible_model_packages"] = [widened]
    _write(manifest_path, manifest)
    _refresh(output, enrichment.SOURCE_ENRICHMENT_FILE)
    enrichment.validate_source_enrichment_candidate(
        output,
        parent_h5=parent,
        artifact_root=root,
        require_compatibility=True,
        compatibility_wheels=(tmp_path / "country.whl",),
    )
