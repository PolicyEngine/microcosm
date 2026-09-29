"""Annual-projection tests using shared support."""

# ruff: noqa: F403, F405
from test_support.microcosm_data.annual_projections import *


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
