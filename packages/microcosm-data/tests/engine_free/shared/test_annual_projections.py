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


# --- The annual release's build folders (microcosm#1570) --------------------


def _contract_failures(release, root):
    with pytest.raises(ReleaseContractError) as caught:
        validate_release_dir(release, artifact_root=root)
    return "\n".join(caught.value.failures)


def test_an_annual_build_folder_passes_the_release_contract(
    annual_build_folder, artifact_root
):
    """The national build's own contract runs on the carried files, and the
    annual checks on the added ones, which pin no revision."""
    assert validate_release_dir(annual_build_folder, artifact_root=artifact_root) == []
    manifest = json.loads((annual_build_folder / "release_manifest.json").read_text())
    result = validate_annual_projection_extension(
        annual_build_folder, manifest, artifact_root=artifact_root
    )
    assert result.revision is None
    assert result.projected_artifacts == {"microcosm_us_2025"}
    assert set(result.additional_artifacts) == {
        "annual_manifest",
        "annual_acceptance",
        "microcosm_us_2025",
        "projection_2025",
    }
    assert result.additional_artifacts["microcosm_us_2025"] == (
        artifact_root / "microcosm_us_2025.h5"
    )
    assert result.additional_artifacts["projection_2025"] == (
        annual_build_folder / "projection_2025.json"
    )


def test_an_annual_build_of_a_moved_build_keeps_its_dataset_file_name(
    release_dir, artifact_root
):
    """A build the switch moved keeps ``populace_us_2024.h5``, and an annual
    build of it carries that file under the same name and key."""
    national = national_build_folder(release_dir, artifact_root, prefix="populace")
    annual = annual_build_folder_of(national, artifact_root)

    assert validate_release_dir(annual, artifact_root=artifact_root) == []
    manifest = json.loads((annual / "release_manifest.json").read_text())
    assert manifest["artifacts"]["populace_us_2024"]["path"] == "populace_us_2024.h5"
    assert manifest["metadata"]["dataset_years"] == {
        "populace_us_2024": {"2024": "populace_us_2024", "2025": "microcosm_us_2025"}
    }


def test_a_build_of_the_annual_release_must_declare_dataset_years(
    annual_build_folder, artifact_root
):
    rewrite_annual_release_manifest(
        annual_build_folder, lambda manifest: manifest["metadata"].pop("dataset_years")
    )

    failures = _contract_failures(annual_build_folder, artifact_root)

    assert "dataset_years" in failures
    assert US_ANNUAL_RELEASE in failures


def test_a_build_folder_build_of_another_release_cannot_declare_dataset_years(
    national_build, artifact_root
):
    """Only the annual release's builds declare annual years in the
    build-folder layout; a national build that does is refused."""

    def declare_annual_years(manifest):
        manifest.setdefault("metadata", {})["dataset_years"] = {
            "microcosm_us_2024": {"2024": "microcosm_us_2024"}
        }

    path = national_build / "release_manifest.json"
    manifest = json.loads(path.read_text())
    declare_annual_years(manifest)
    path.write_text(json.dumps(manifest))

    failures = _contract_failures(national_build, artifact_root)

    assert "dataset_years" in failures
    assert US_ANNUAL_RELEASE in failures
    assert US_NATIONAL_RELEASE in failures


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        ("missing", "annual_projection_base"),
        ("other_release", "microcosm-us-2024-evidence"),
        ("revision", "build-folder form"),
        ("other_base_build", "base build"),
        ("build_id_form", "-annual-"),
        ("parent_release", "parent_release"),
        ("reference_sha256", "sha256"),
        ("candidate_sha256", "projection base hash"),
        ("reference_path", "path"),
        ("reference_repo_id", "repo_id"),
    ],
)
def test_the_annual_base_reference_names_the_projected_national_build(
    annual_build_folder, artifact_root, problem, expected
):
    """The release manifest names its base national build in the build-folder
    form; the build ID, the candidate and the base-year H5 must agree with it."""
    other_build = "populace-us-2024-0000000-20260101"

    def change_reference(manifest):
        reference = manifest["metadata"]["annual_projection_base"]
        if problem == "missing":
            del manifest["metadata"]["annual_projection_base"]
        elif problem == "other_release":
            reference["release"] = "microcosm-us-2024-evidence"
        elif problem == "revision":
            reference["revision"] = "main"
        elif problem == "other_base_build":
            reference["build"] = other_build
        elif problem == "build_id_form":
            manifest["build"] = f"{RELEASE_ID}-annual-latest"
        elif problem == "reference_sha256":
            reference["sha256"] = "0" * 64
        elif problem == "reference_path":
            reference["path"] = "populace_us_2024.h5"
        elif problem == "reference_repo_id":
            reference["repo_id"] = "policyengine/populace-us-private"

    if problem == "parent_release":
        rewrite_annual_evidence(
            annual_build_folder,
            lambda evidence: evidence["base"].update(parent_release=other_build),
        )
    elif problem == "candidate_sha256":
        rewrite_annual_evidence(
            annual_build_folder,
            lambda evidence: evidence["base"].update(sha256="0" * 64),
        )
    else:
        rewrite_annual_release_manifest(annual_build_folder, change_reference)
    if problem == "build_id_form":
        path = annual_build_folder / "build_manifest.json"
        build = json.loads(path.read_text())
        build["build"] = f"{RELEASE_ID}-annual-latest"
        path.write_text(json.dumps(build))

    failures = _contract_failures(annual_build_folder, artifact_root)

    assert "annual projection extension" in failures
    assert expected in failures


@pytest.mark.parametrize("problem", ["revision", "release_prefix"])
def test_annual_build_folder_files_have_bare_names_and_no_revision(
    annual_build_folder, artifact_root, problem
):
    def change(manifest):
        if problem == "revision":
            manifest["artifacts"]["microcosm_us_2025"]["revision"] = ANNUAL_BUILD
        else:
            manifest["artifacts"]["projection_2025"]["path"] = (
                f"releases/{US_ANNUAL_RELEASE}/projection_2025.json"
            )

    rewrite_annual_release_manifest(annual_build_folder, change)

    failures = _contract_failures(annual_build_folder, artifact_root)

    assert "annual projection extension" in failures
    assert ("pins a revision" if problem == "revision" else "bare file name") in (
        failures
    )


def test_an_annual_build_folder_covers_every_year_from_its_base(
    annual_build_folder, artifact_root
):
    years = {"2024": "microcosm_us_2024", "2026": "microcosm_us_2025"}
    rewrite_annual_evidence(
        annual_build_folder,
        lambda evidence: evidence["metadata"].update(
            dataset_years={"microcosm_us_2024": years}
        ),
    )
    rewrite_annual_release_manifest(
        annual_build_folder,
        lambda manifest: manifest["metadata"].update(
            dataset_years={"microcosm_us_2024": years}
        ),
    )

    failures = _contract_failures(annual_build_folder, artifact_root)

    assert "cover every year" in failures


@pytest.mark.parametrize(
    ("artifact", "file_name"),
    [
        # A receipt-qualified donor's dataset is never an annual base.
        ("microcosm_us_2024", "microcosm_us_2024_receipt_qualified.h5"),
        # A projected year is produced under its microcosm_ name.
        ("microcosm_us_2025", "populace_us_2025.h5"),
    ],
)
def test_an_annual_build_declares_its_dataset_files(
    annual_build_folder, artifact_root, artifact, file_name
):
    manifest = json.loads((annual_build_folder / "release_manifest.json").read_text())
    current = manifest["artifacts"][artifact]["path"]
    shutil.copyfile(artifact_root / current, artifact_root / file_name)

    def rename(manifest):
        manifest["artifacts"][artifact]["path"] = file_name
        if artifact == "microcosm_us_2024":
            manifest["metadata"]["annual_projection_base"]["path"] = file_name

    rewrite_annual_release_manifest(annual_build_folder, rename)
    if artifact == "microcosm_us_2024":
        path = annual_build_folder / "build_manifest.json"
        build = json.loads(path.read_text())
        build["dataset"]["filename"] = file_name
        path.write_text(json.dumps(build))

    failures = _contract_failures(annual_build_folder, artifact_root)

    assert file_name in failures
    expected = "populace_us_2024.h5" if artifact == "microcosm_us_2024" else current
    assert expected in failures
