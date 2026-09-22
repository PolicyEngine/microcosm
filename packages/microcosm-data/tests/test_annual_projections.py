"""Annual publication binds accepted years to exact native H5 artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil

import h5py
import numpy as np
import pytest

from microcosm.data.annual_projections import validate_annual_projection_extension
from microcosm.data.contract import ReleaseContractError, validate_release_dir


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _h5(path, year, *, person_id=1, weight=10.0):
    with h5py.File(path, "w") as store:
        store.create_dataset(
            "_time_period/table",
            data=np.array([(0, year)], dtype=[("index", "i8"), ("values", "i8")]),
        )
        for entity in (
            "person",
            "household",
            "tax_unit",
            "spm_unit",
            "family",
            "marital_unit",
        ):
            fields = [("index", "i8"), (f"{entity}_id", "i8")]
            row = [0, person_id if entity == "person" else 1]
            if entity == "person":
                fields.extend(
                    [("person_household_id", "i8"), ("employment_income", "f8")]
                )
                row.extend([1, 100.0 * (year - 2023)])
            if entity == "household":
                fields.append(("household_weight", "f8"))
                row.append(weight)
            store.create_dataset(
                f"{entity}/table", data=np.array([tuple(row)], dtype=fields)
            )


@pytest.fixture
def candidate(tmp_path):
    release = tmp_path / "release-id"
    release.mkdir()
    root = tmp_path / "artifacts"
    root.mkdir()
    mapping = {
        "populace_us_2024": {"2024": "populace_us_2024", "2025": "populace_us_2025"}
    }
    manifest = {
        "metadata": {"dataset_years": mapping},
        "artifacts": {},
        "default_datasets": {"national": "populace_us_2024"},
        "build": {
            "build_id": release.name,
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": "2.6.15",
            },
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.32.5",
            },
        },
    }
    evidence = {
        "schema_version": 1,
        "kind": "us_annual_static_aging_candidate",
        "status": "complete",
        "base": {
            "dataset": "populace_us_2024",
            "year": 2024,
            "parent_release": release.name,
        },
        "metadata": {"dataset_years": mapping},
        "artifacts": {},
        "model": {
            "version": "2.6.15",
            "commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
        },
        "runtime": {
            "versions": {"policyengine-us": "2.6.15", "policyengine-core": "3.32.5"}
        },
    }
    acceptance = {
        "schema_version": 1,
        "kind": "us_annual_projection_acceptance",
        "status": "passed",
        "years": {},
        "model": dict(evidence["model"]),
        "runtime": dict(evidence["runtime"]["versions"]),
    }
    for year in (2024, 2025):
        key = f"populace_us_{year}"
        path = root / f"{key}.h5"
        _h5(path, year)
        sha = _sha(path)
        manifest["artifacts"][key] = {
            "path": path.name,
            "sha256": sha,
            "revision": f"{release.name}-annual-20260919T220000Z-a1b2c3d4",
            "kind": "microdata",
        }
        evidence["artifacts"][key] = {
            "year": year,
            "sha256": sha,
            "rows": {
                name: 1
                for name in (
                    "person",
                    "household",
                    "tax_unit",
                    "spm_unit",
                    "family",
                    "marital_unit",
                )
            },
        }
        acceptance["years"][str(year)] = {
            "dataset": key,
            "sha256": sha,
            "checks": {
                name: "passed"
                for name in (
                    "schema",
                    "source_identity",
                    "year",
                    "demographics",
                    "input_aggregates",
                    "runtime",
                )
            },
        }
    evidence["base"]["sha256"] = manifest["artifacts"]["populace_us_2024"]["sha256"]
    projection_path = release / "projection_2025.json"
    projection_path.write_text(
        json.dumps({"base_year": 2024, "year": 2025, "factors": {}})
    )
    evidence["artifacts"]["populace_us_2025"]["projection_receipt"] = {
        "path": projection_path.name,
        "sha256": _sha(projection_path),
    }
    manifest["artifacts"]["projection_2025"] = {
        "path": f"releases/{release.name}/{projection_path.name}",
        "sha256": _sha(projection_path),
        "revision": f"{release.name}-annual-20260919T220000Z-a1b2c3d4",
        "kind": "diagnostics",
    }

    def save():
        evidence_path = release / "annual_manifest.json"
        evidence_path.write_text(json.dumps(evidence))
        acceptance["projection_manifest_sha256"] = _sha(evidence_path)
        acceptance_path = release / "annual_acceptance.json"
        acceptance_path.write_text(json.dumps(acceptance))
        for key, field, path in (
            ("annual_manifest", "annual_projection_manifest", evidence_path),
            ("annual_acceptance", "annual_projection_acceptance", acceptance_path),
        ):
            manifest["metadata"][field] = key
            manifest["artifacts"][key] = {
                "path": f"releases/{release.name}/{path.name}",
                "sha256": _sha(path),
                "revision": f"{release.name}-annual-20260919T220000Z-a1b2c3d4",
            }
        (release / "release_manifest.json").write_text(json.dumps(manifest))

    save()
    return release, root, manifest, evidence, acceptance, save


def add_annual_extension(release, root):
    """Augment an otherwise valid release without replacing its base evidence."""
    manifest = json.loads((release / "release_manifest.json").read_text())
    base_key = manifest["default_datasets"]["national"]
    base = manifest["artifacts"][base_key]
    projected = root / "annual_2025.h5"
    shutil.copyfile(root / base["path"], projected)
    with h5py.File(projected, "r+") as store:
        row = store["_time_period/table"][:]
        row["values"] = 2025
        store["_time_period/table"][:] = row
        rows = {
            entity: len(store[f"{entity}/table"])
            for entity in (
                "person",
                "household",
                "tax_unit",
                "spm_unit",
                "family",
                "marital_unit",
            )
        }
    runtime = {
        "policyengine-us": manifest["build"]["built_with_model_package"]["version"],
        "policyengine-core": manifest["build"]["built_with_core_package"]["version"],
    }
    model = {
        "version": runtime["policyengine-us"],
        "commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
    }
    mapping = {base_key: {"2024": base_key, "2025": "annual_2025"}}
    evidence = {
        "schema_version": 1,
        "kind": "us_annual_static_aging_candidate",
        "status": "complete",
        "base": {
            "dataset": base_key,
            "year": 2024,
            "sha256": base["sha256"],
            "parent_release": release.name,
        },
        "metadata": {"dataset_years": mapping},
        "model": model,
        "runtime": {"versions": runtime},
        "artifacts": {},
    }
    acceptance = {
        "schema_version": 1,
        "kind": "us_annual_projection_acceptance",
        "status": "passed",
        "model": model,
        "runtime": runtime,
        "years": {},
    }
    manifest["artifacts"]["annual_2025"] = {
        "kind": "microdata",
        "path": projected.name,
        "sha256": _sha(projected),
        "repo_id": base["repo_id"],
    }
    for year, key in mapping[base_key].items():
        sha = manifest["artifacts"][key]["sha256"]
        evidence["artifacts"][key] = {"year": int(year), "sha256": sha, "rows": rows}
        acceptance["years"][year] = {
            "dataset": key,
            "sha256": sha,
            "checks": {
                name: "passed"
                for name in (
                    "schema",
                    "source_identity",
                    "year",
                    "demographics",
                    "input_aggregates",
                    "runtime",
                )
            },
        }
    projection_path = release / "projection_2025.json"
    projection_path.write_text(
        json.dumps({"base_year": 2024, "year": 2025, "factors": {}})
    )
    evidence["artifacts"]["annual_2025"]["projection_receipt"] = {
        "path": projection_path.name,
        "sha256": _sha(projection_path),
    }
    manifest["artifacts"]["projection_2025"] = {
        "kind": "diagnostics",
        "path": f"releases/{release.name}/{projection_path.name}",
        "sha256": _sha(projection_path),
        "repo_id": base["repo_id"],
    }
    evidence_path = release / "annual_manifest.json"
    evidence_path.write_text(json.dumps(evidence))
    acceptance["projection_manifest_sha256"] = _sha(evidence_path)
    acceptance_path = release / "annual_acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance))
    manifest["metadata"] = {
        **manifest.get("metadata", {}),
        "dataset_years": mapping,
        "annual_projection_manifest": "annual_manifest",
        "annual_projection_acceptance": "annual_acceptance",
    }
    for key, path in (
        ("annual_manifest", evidence_path),
        ("annual_acceptance", acceptance_path),
    ):
        manifest["artifacts"][key] = {
            "kind": "diagnostics",
            "path": f"releases/{release.name}/{path.name}",
            "sha256": _sha(path),
            "repo_id": base["repo_id"],
        }
    tag = f"{release.name}-annual-20260919T220000Z-a1b2c3d4"
    for artifact in manifest["artifacts"].values():
        artifact["revision"] = tag
    (release / "release_manifest.json").write_text(json.dumps(manifest))
    return tag


def test_unrelated_release_retains_existing_contract(tmp_path):
    validate_annual_projection_extension(tmp_path, {})


def test_accepted_files_match_year_identity_and_hash(candidate):
    release, root, manifest, *_ = candidate
    result = validate_annual_projection_extension(release, manifest, artifact_root=root)
    assert result.revision == f"{release.name}-annual-20260919T220000Z-a1b2c3d4"
    assert result.projected_artifacts == {"populace_us_2025"}
    assert set(result.additional_artifacts) == {
        "annual_manifest",
        "annual_acceptance",
        "populace_us_2025",
        "projection_2025",
    }


def test_annual_family_rejects_missing_middle_year(candidate):
    release, root, manifest, evidence, acceptance, save = candidate
    old_key, key = "populace_us_2025", "populace_us_2026"
    mapping = manifest["metadata"]["dataset_years"]["populace_us_2024"]
    del mapping["2025"]
    mapping["2026"] = key
    path = root / f"{key}.h5"
    _h5(path, 2026)
    sha = _sha(path)
    artifact = manifest["artifacts"][key] = manifest["artifacts"].pop(old_key)
    artifact.update(path=path.name, sha256=sha)
    record = evidence["artifacts"][key] = evidence["artifacts"].pop(old_key)
    record.update(year=2026, sha256=sha)
    receipt_path = release / "projection_2026.json"
    receipt_path.write_text(
        json.dumps({"base_year": 2024, "year": 2026, "factors": {}})
    )
    record["projection_receipt"] = {
        "path": receipt_path.name,
        "sha256": _sha(receipt_path),
    }
    receipt_artifact = manifest["artifacts"]["projection_2026"] = manifest[
        "artifacts"
    ].pop("projection_2025")
    receipt_artifact.update(
        path=f"releases/{release.name}/{receipt_path.name}", sha256=_sha(receipt_path)
    )
    accepted = acceptance["years"]["2026"] = acceptance["years"].pop("2025")
    accepted.update(dataset=key, sha256=sha)
    save()
    with pytest.raises(ValueError, match="cover every year"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


@pytest.mark.parametrize(
    "revision",
    [
        None,
        [],
        "release-id",
        "release-id-annual-20260919T220000Z-invalid",
        "other-annual-20260919T220000Z-a1b2c3d4",
    ],
)
def test_annual_extension_requires_immutable_uniform_cut(candidate, revision):
    release, root, manifest, *_ = candidate
    manifest["artifacts"]["populace_us_2025"]["revision"] = revision
    with pytest.raises(ValueError, match="revision|annual cut"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


def test_nested_root_artifact_cannot_be_shadowed_by_release_file(candidate):
    release, root, manifest, *_ = candidate
    relative = "annual/populace_us_2025.h5"
    (root / "annual").mkdir()
    (release / "annual").mkdir()
    shutil.copyfile(root / "populace_us_2025.h5", root / relative)
    shutil.copyfile(root / relative, release / relative)
    manifest["artifacts"]["populace_us_2025"]["path"] = relative
    with pytest.raises(ValueError, match="exact release prefix"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_reference",
        "undeclared",
        "missing_file",
        "bare_path",
        "candidate_hash",
        "changed_bytes",
        "wrong_year",
        "duplicate_path",
    ],
)
def test_per_year_projection_receipt_must_be_delivered(candidate, problem):
    release, root, manifest, evidence, _, save = candidate
    record = evidence["artifacts"]["populace_us_2025"]
    artifact = manifest["artifacts"]["projection_2025"]
    path = release / "projection_2025.json"
    if problem == "missing_reference":
        del record["projection_receipt"]
    elif problem == "undeclared":
        del manifest["artifacts"]["projection_2025"]
    elif problem == "missing_file":
        path.unlink()
    elif problem == "bare_path":
        artifact["path"] = path.name
    elif problem == "candidate_hash":
        record["projection_receipt"]["sha256"] = "0" * 64
    elif problem == "changed_bytes":
        path.write_text("{}")
    elif problem == "wrong_year":
        path.write_text(json.dumps({"base_year": 2024, "year": 2031}))
        artifact["sha256"] = record["projection_receipt"]["sha256"] = _sha(path)
    else:
        manifest["artifacts"]["duplicate_receipt"] = dict(artifact)
    save()
    with pytest.raises(ValueError, match="projection receipt|sha256"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


@pytest.mark.parametrize(
    "problem", ["missing_year", "failed_runtime", "changed_sha", "extra_year"]
)
def test_incomplete_or_stale_acceptance_refuses(candidate, problem):
    release, root, manifest, _, acceptance, save = candidate
    if problem == "missing_year":
        del acceptance["years"]["2025"]
    elif problem == "failed_runtime":
        acceptance["years"]["2025"]["checks"]["runtime"] = "pending"
    elif problem == "changed_sha":
        acceptance["years"]["2025"]["sha256"] = "0" * 64
    else:
        acceptance["years"]["2035"] = acceptance["years"]["2025"]
    save()
    with pytest.raises(ValueError, match="acceptance"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


@pytest.mark.parametrize("problem", ["year", "person", "negative_weight", "nan_weight"])
def test_native_content_checks_do_not_trust_receipt_claims(candidate, problem):
    release, root, manifest, evidence, acceptance, save = candidate
    key = "populace_us_2025"
    path = root / f"{key}.h5"
    _h5(
        path,
        2035 if problem == "year" else 2025,
        person_id=2 if problem == "person" else 1,
        weight=-1
        if problem == "negative_weight"
        else float("nan")
        if problem == "nan_weight"
        else 10,
    )
    sha = _sha(path)
    manifest["artifacts"][key]["sha256"] = sha
    evidence["artifacts"][key]["sha256"] = sha
    acceptance["years"]["2025"]["sha256"] = sha
    save()
    with pytest.raises(ValueError, match="year|person_id|weights"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


def test_changed_h5_bytes_refuse(candidate):
    release, root, manifest, *_ = candidate
    with (root / "populace_us_2025.h5").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


def test_source_enrichment_cannot_skip_annual_checks(candidate):
    release, root, manifest, _, _, save = candidate
    manifest["release_type"] = "source_enrichment"
    manifest["metadata"]["dataset_years"] = None
    save()
    with pytest.raises(ReleaseContractError, match="dataset_years"):
        validate_release_dir(release, artifact_root=root)


def test_missing_acceptance_does_not_certify_candidate(candidate):
    release, root, manifest, *_ = candidate
    del manifest["metadata"]["annual_projection_acceptance"]
    with pytest.raises(ValueError, match="acceptance"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


@pytest.mark.parametrize(
    "path", ["../outside.h5", "/absolute.h5", "releases/../outside.h5"]
)
def test_noncanonical_artifact_paths_refuse(candidate, path):
    release, root, manifest, *_ = candidate
    manifest["artifacts"]["populace_us_2025"]["path"] = path
    with pytest.raises(ValueError, match="relative path"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


@pytest.mark.parametrize(
    "path", ["releases/release-id/nested/2025.h5", "releases/another/2025.h5"]
)
def test_release_paths_must_match_the_publishers_upload_layout(candidate, path):
    release, root, manifest, *_ = candidate
    manifest["artifacts"]["populace_us_2025"]["path"] = path
    with pytest.raises(ValueError, match="bare filenames|another release"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


def test_bare_release_local_path_cannot_publish_at_wrong_location(candidate):
    release, root, manifest, *_ = candidate
    source = root / "populace_us_2025.h5"
    (release / source.name).write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="exact release prefix"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


@pytest.mark.parametrize(
    "problem",
    ["other_family", "invented_parent", "wrong_model", "wrong_acceptance_model"],
)
def test_annual_evidence_must_belong_to_certified_base_and_runtime(candidate, problem):
    release, root, manifest, evidence, acceptance, save = candidate
    if problem == "other_family":
        manifest["default_datasets"]["national"] = "another_base"
    elif problem == "invented_parent":
        evidence["base"]["parent_release"] = "invented-not-a-release"
    elif problem == "wrong_model":
        manifest["build"]["built_with_model_package"]["version"] = "2.2.1"
    else:
        acceptance["model"]["source_tree_sha256"] = "c" * 64
    save()
    with pytest.raises(ValueError, match="certified national base|model|runtime"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)


@pytest.mark.parametrize("specifier", [None, "", " ", ","])
def test_empty_compatibility_claim_cannot_admit_unvalidated_model(candidate, specifier):
    release, root, manifest, *_ = candidate
    manifest["build"]["built_with_model_package"]["version"] = "2.2.1"
    manifest["compatible_model_packages"] = [
        {"name": "policyengine-us", "specifier": specifier}
    ]
    with pytest.raises(ValueError, match="nonempty specifiers"):
        validate_annual_projection_extension(release, manifest, artifact_root=root)
