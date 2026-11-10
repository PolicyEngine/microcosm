"""Tests split from packages/microcosm-build/tests/test_us_receipt_enrichment_release.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_receipt_enrichment_release import *


def test_candidate_carries_the_qualified_child_as_its_one_microdata_artifact(
    lineage,
) -> None:
    report = _build(lineage)

    release = _release(lineage)
    manifest = json.loads((release / "release_manifest.json").read_text())
    microdata = {
        key: entry
        for key, entry in manifest["artifacts"].items()
        if entry["kind"] == "microdata"
    }
    child_sha = file_sha256(lineage.child)
    assert microdata == {
        "microcosm_us_2024_receipt_qualified": {
            "kind": "microdata",
            "path": "microcosm_us_2024_receipt_qualified.h5",
            "repo_id": "policyengine/populace-us",
            "role": "dataset",
            "sha256": child_sha,
        }
    }
    assert manifest["default_datasets"] == {
        "national": "microcosm_us_2024_receipt_qualified"
    }
    assert (manifest["release"], manifest["build"]) == (
        US_RECEIPT_DONOR_RELEASE_ID,
        _RELEASE_ID,
    )
    assert not any("revision" in entry for entry in manifest["artifacts"].values())
    assert "build_manifest" not in manifest["artifacts"]
    assert manifest["release_type"] == "source_enrichment"
    assert "dataset_role" not in manifest
    assert sorted(path.name for path in release.iterdir()) == sorted(
        [
            "build_manifest.json",
            "calibration_diagnostics.json",
            contract.RECEIPT_QUALIFICATION_FILE,
            "parent_release_manifest.json",
            "release_manifest.json",
            contract.SOURCE_ENRICHMENT_FILE,
            "us_source_coverage.json",
        ]
    )
    assert file_sha256(_artifacts(lineage) / contract.RECEIPT_DATASET_FILENAME) == (
        child_sha
    )
    receipt_bytes = (
        lineage.qualified / contract.RECEIPT_QUALIFICATION_FILE
    ).read_bytes()
    assert (release / contract.RECEIPT_QUALIFICATION_FILE).read_bytes() == receipt_bytes
    receipt = json.loads(receipt_bytes)
    assert report["preservation"] == receipt["preservation"]
    # The parent is a build in the national release's folder, not a tag.
    assert report["parent"] == {
        "repo_id": "policyengine/populace-us",
        "release": US_NATIONAL_RELEASE_ID,
        "build": "populace-us-2024-spm-20260915",
        "dataset_sha256": file_sha256(lineage.parent),
        "calibration_diagnostics_schema_version": 5,
        "files": contract.RECEIPT_PARENT_FILES,
    }
    assert report["compatibility"] == {"status": "pending"}
    build = json.loads((release / "build_manifest.json").read_text())
    assert (build["release"], build["build"]) == (
        US_RECEIPT_DONOR_RELEASE_ID,
        _RELEASE_ID,
    )
    assert build["calibration"] == {
        "mode": "inherited",
        "parent_build_id": "populace-us-2024-spm-20260915",
        "diagnostics_sha256": contract.RECEIPT_PARENT_FILES[
            "calibration_diagnostics.json"
        ],
        "diagnostics_schema_version": 5,
    }
    assert _validate(lineage)["operation"] == contract.RECEIPT_OPERATION


def test_the_local_chain_accepts_the_manifest_as_a_published_donor(lineage) -> None:
    _build(lineage)
    manifest = _release(lineage) / "release_manifest.json"

    identity = _legacy_staging_module()._donor_release_identity(
        manifest, file_sha256(lineage.child)
    )

    # The donor is a build of its own release, named by release and build.
    published = json.loads(manifest.read_text())
    assert (published["release"], published["build"]) == (
        US_RECEIPT_DONOR_RELEASE_ID,
        _RELEASE_ID,
    )
    assert identity["artifact"] == "microcosm_us_2024_receipt_qualified"
    assert identity["repo_id"] == "policyengine/populace-us"
    assert identity["sha256"] == file_sha256(lineage.child)
    with pytest.raises(SystemExit, match="does not match its release manifest"):
        _legacy_staging_module()._donor_release_identity(
            manifest, file_sha256(lineage.parent)
        )


def test_pending_candidate_cannot_pass_the_publication_contract(lineage) -> None:
    _build(lineage)

    with pytest.raises(ReleaseContractError, match="compatibility is pending"):
        validate_release_dir(
            _release(lineage),
            parent_h5=lineage.parent,
            artifact_root=_artifacts(lineage),
        )


def test_certification_checks_the_role_and_all_three_receipts_natively(
    lineage, monkeypatch
) -> None:
    _build(lineage)
    calls: list = []

    certified = _certify(lineage, monkeypatch, calls)

    assert calls and all(
        inputs
        == {
            "is_spm_independent_minor_role": "person",
            "receives_wic": "person",
            "receives_snap": "spm_unit",
            "receives_tanf": "spm_unit",
        }
        for inputs in calls
    )
    validate_release_dir(
        certified,
        parent_h5=lineage.parent,
        artifact_root=_artifacts(lineage),
        compatibility_wheels=_WHEELS,
    )


def test_the_donor_is_a_build_of_its_own_release_never_the_national_default(
    lineage, monkeypatch
) -> None:
    _build(lineage)
    certified = _certify(lineage, monkeypatch, [])
    options = {"parent_h5": lineage.parent, "compatibility_wheels": _WHEELS}
    # The release contract replays the H5 comparison and the loader checks,
    # which take the parent H5 and the tested wheels.
    monkeypatch.setattr(
        publisher,
        "validate_release_dir",
        lambda release_dir, **kwargs: validate_release_dir(
            release_dir, **{**options, **kwargs}
        ),
    )
    monkeypatch.setattr(publisher, "_hf_api", lambda: pytest.fail("use no Hub"))

    prepared = publisher.prepare_build_folder_release(
        certified,
        artifact_root=_artifacts(lineage),
        repository="policyengine/populace-us",
    )

    assert (prepared.release, prepared.build) == (
        US_RECEIPT_DONOR_RELEASE_ID,
        _RELEASE_ID,
    )
    assert {
        item.name: item.path for item in prepared.files if item.role == "dataset"
    } == {
        contract.RECEIPT_DATASET_FILENAME: _artifacts(lineage)
        / contract.RECEIPT_DATASET_FILENAME
    }
    # The tag publisher, which writes latest.json, refuses it.
    with pytest.raises(ValueError, match="microcosm-publish-release-build"):
        publisher.prepare_release(
            certified, artifact_root=_artifacts(lineage), **options
        )
    # The same build named as a build of the national release is refused: it
    # is not that release's lineage, and its file is not that release's
    # dataset file.
    national = lineage.tmp / "national" / US_NATIONAL_RELEASE_ID
    shutil.copytree(certified, national)
    for name in ("release_manifest.json", "build_manifest.json"):
        _rewrite(national, name, lambda payload: payload.update(release=national.name))
    with pytest.raises(ReleaseContractError) as refusal:
        validate_release_dir(national, artifact_root=_artifacts(lineage), **options)
    message = str(refusal.value)
    assert (
        f"a reported-receipt enrichment is a build of {US_RECEIPT_DONOR_RELEASE_ID}, "
        f"never of {US_NATIONAL_RELEASE_ID}"
    ) in message
    assert "must declare one microdata artifact, 'microcosm_us_2024.h5'" in message


def test_builder_refuses_any_parent_but_the_pinned_default(
    lineage, monkeypatch
) -> None:
    for module in (contract, builder):
        monkeypatch.setattr(module, "RECEIPT_PARENT_DATASET_SHA256", "e" * 64)

    with pytest.raises(ValueError, match="Only the exact pinned national default"):
        _build(lineage)
    assert not lineage.output.exists()


def test_builder_refuses_a_qualification_of_another_parent(lineage) -> None:
    receipt_path = lineage.qualified / contract.RECEIPT_QUALIFICATION_FILE
    receipt_path.chmod(0o600)
    receipt = json.loads(receipt_path.read_text())
    receipt["parent"]["sha256"] = contract.PARENT_DATASET_SHA256
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="parent is not the pinned national default"):
        _build(lineage)


def test_builder_refuses_changed_parent_evidence(lineage) -> None:
    (lineage.parent_release / "us_source_coverage.json").write_text("{}")

    with pytest.raises(ValueError, match="Parent evidence SHA-256 mismatch"):
        _build(lineage)


@pytest.mark.parametrize("release_id", ["populace-us-2024-spm-20260915", "other-id"])
def test_builder_refuses_a_reused_or_foreign_release_id(lineage, release_id) -> None:
    with pytest.raises(ValueError, match="Choose a new bare US 2024"):
        _build(lineage, release_id=release_id)


def test_contract_refuses_a_parent_other_than_the_pinned_default(
    lineage, monkeypatch
) -> None:
    _build(lineage)
    monkeypatch.setattr(contract, "RECEIPT_PARENT_DATASET_SHA256", "e" * 64)

    with pytest.raises(ReleaseContractError) as refusal:
        _validate(lineage)

    message = str(refusal.value)
    assert "parent H5 SHA256 differs from the pinned national default" in message
    assert "must match the pinned national default identity" in message
    assert "must name the pinned national default as its parent" in message


def test_contract_replays_the_h5_comparison_rather_than_trusting_the_receipt(
    lineage,
) -> None:
    _build(lineage)
    candidate = _artifacts(lineage) / contract.RECEIPT_DATASET_FILENAME
    candidate.chmod(0o600)
    with h5py.File(candidate, "r+") as h5:
        table = h5["household/table"]
        record = qualification._raw_rows(table, 0, 1)
        record["household_weight"] += 1.0
        selection = table.id.get_space()
        selection.select_hyperslab((0,), (1,))
        table.id.write(
            h5py.h5s.create_simple((1,)), selection, record, mtype=table.id.get_type()
        )
    # Re-seal every recorded digest, so only the replay can see the change.
    release = _release(lineage)
    tampered = file_sha256(candidate)
    for name in ("build_manifest.json", contract.SOURCE_ENRICHMENT_FILE):
        _rewrite(
            release, name, lambda payload: payload["dataset"].update(sha256=tampered)
        )
    _rewrite(
        release,
        contract.RECEIPT_QUALIFICATION_FILE,
        lambda payload: payload["dataset"].update(sha256=tampered),
    )
    _rewrite(
        release,
        "release_manifest.json",
        lambda payload: payload["artifacts"][
            "microcosm_us_2024_receipt_qualified"
        ].update(sha256=tampered),
    )

    with pytest.raises(
        ReleaseContractError,
        match="Existing HDF dataset values changed: household/table",
    ):
        _validate(lineage)


def test_contract_recounts_the_receipt_totals_from_the_h5(lineage) -> None:
    _build(lineage)

    def inflate(receipt):
        counts = receipt["counts"]["receives_wic"]
        counts["true"] += 1
        counts["false"] -= 1

    _rewrite(_release(lineage), contract.RECEIPT_QUALIFICATION_FILE, inflate)

    with pytest.raises(
        ReleaseContractError,
        match="counts for receives_wic differ from the enriched H5",
    ):
        _validate(lineage)


def test_contract_refuses_a_receipt_with_no_true_channel_value(lineage) -> None:
    _build(lineage)

    def silence(receipt):
        receipt["counts"]["receives_tanf"]["gate_selected_channel"]["true"] = 0

    _rewrite(_release(lineage), contract.RECEIPT_QUALIFICATION_FILE, silence)

    with pytest.raises(ReleaseContractError, match="records no true receives_tanf"):
        _validate(lineage)


def test_contract_refuses_a_receipt_whose_preservation_differs(lineage) -> None:
    _build(lineage)

    def restate(receipt):
        receipt["preservation"]["datasets_checked"] += 1

    _rewrite(_release(lineage), contract.RECEIPT_QUALIFICATION_FILE, restate)

    with pytest.raises(ReleaseContractError, match="preservation must equal"):
        _validate(lineage)


def test_contract_refuses_an_unbound_qualification_receipt(lineage) -> None:
    _build(lineage)
    receipt = _release(lineage) / contract.RECEIPT_QUALIFICATION_FILE
    receipt.chmod(0o600)
    receipt.write_text(receipt.read_text() + "\n")

    with pytest.raises(ReleaseContractError, match="must bind the immutable"):
        _validate(lineage)


def test_contract_refuses_publishing_over_the_default_root_file(lineage) -> None:
    _build(lineage)
    release = _release(lineage)
    for name in ("build_manifest.json", contract.SOURCE_ENRICHMENT_FILE):
        _rewrite(
            release,
            name,
            lambda payload: payload["dataset"].update(filename="populace_us_2024.h5"),
        )

    with pytest.raises(ReleaseContractError, match="never over the national default"):
        _validate(lineage)


def test_contract_refuses_a_changed_added_variable_list(lineage) -> None:
    _build(lineage)
    _rewrite(
        _release(lineage),
        contract.SOURCE_ENRICHMENT_FILE,
        lambda payload: payload["added_variables"].reverse(),
    )

    with pytest.raises(ReleaseContractError, match="added_variables must declare"):
        _validate(lineage)


def test_an_unknown_operation_is_still_judged_as_the_role_lane(lineage) -> None:
    _build(lineage)
    _rewrite(
        _release(lineage),
        contract.SOURCE_ENRICHMENT_FILE,
        lambda payload: payload.update(operation="add_reported_receipt_input"),
    )

    with pytest.raises(
        ReleaseContractError, match="operation must add only the native SPM role"
    ):
        _validate(lineage)


def test_publication_authenticates_both_producers(lineage, monkeypatch) -> None:
    _build(lineage)
    certified = _certify(lineage, monkeypatch, [])
    # Restore the real authenticator: both recorded identities are the
    # fixture's invented commits, which this checkout cannot resolve.
    monkeypatch.undo()
    failures: list[str] = []

    contract._check_receipt_producers(
        json.loads((certified / "build_manifest.json").read_text()),
        certified,
        failures,
    )

    assert len(failures) == 2
    assert failures[0].startswith("producer source identity failed")
    assert failures[1].startswith("qualification producer source identity failed")
    assert all("cannot be authenticated" in failure for failure in failures)


def test_precedence_probe_supplies_every_receipt_on_its_own_entity(
    lineage, monkeypatch
) -> None:
    calls = _fake_engine(monkeypatch, honours=set(contract.RECEIPT_COLUMNS))

    contract._check_native_input_precedence(
        _country(lineage), native_inputs=contract.RECEIPT_COLUMNS
    )

    assert len(calls) == 2
    for supplied, dataset in zip((False, True), calls, strict=True):
        assert dataset.person["receives_wic"].tolist() == [supplied] * len(
            dataset.person
        )
        for column in ("receives_snap", "receives_tanf"):
            assert dataset.spm_unit[column].tolist() == [supplied] * len(
                dataset.spm_unit
            )


def test_precedence_probe_refuses_a_discarded_spm_unit_receipt(
    lineage, monkeypatch
) -> None:
    _fake_engine(monkeypatch, honours={"receives_wic", "receives_tanf"})

    with pytest.raises(
        ValueError, match="discarded the supplied native spm_unit input receives_snap"
    ):
        contract._check_native_input_precedence(
            _country(lineage), native_inputs=contract.RECEIPT_COLUMNS
        )


def test_receipt_sources_must_come_from_the_country_wheel(
    tmp_path, monkeypatch
) -> None:
    import importlib

    monkeypatch.setattr(
        metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda relative: tmp_path / relative),
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=str(tmp_path / name / "__init__.py")),
    )
    packages = contract._loaded_source_packages(contract.RECEIPT_NATIVE_INPUTS)
    assert packages == {
        **contract.LOADED_SOURCE_PACKAGES,
        "receives_wic": "policyengine-us",
        "receives_snap": "policyengine-us",
        "receives_tanf": "policyengine-us",
    }
    paths = {
        "country_loader": tmp_path / "policyengine_us/data/dataset_schema.py",
        "wrapper_loader": tmp_path / "policyengine/tax_benefit_models/us/datasets.py",
        "native_role": tmp_path / "spm_calculator/policyengine_adapter.py",
        "receives_wic": tmp_path / "policyengine_us/variables/wic.py",
        "receives_snap": tmp_path / "policyengine_us/variables/snap.py",
        "receives_tanf": tmp_path / "policyengine_us/variables/tanf.py",
    }
    contract._check_loaded_source_ownership(paths, packages)
    with pytest.raises(ValueError, match="receives_snap.*policyengine-us wheel"):
        contract._check_loaded_source_ownership(
            paths | {"receives_snap": tmp_path / "elsewhere/snap.py"}, packages
        )


def test_the_contract_pins_the_qualification_tools_exact_inventory() -> None:
    assert contract.RECEIPT_QUALIFICATION_SOURCE_FILES == qualification._PRODUCER_FILES
    for relative in (
        *contract.RECEIPT_QUALIFICATION_SOURCE_FILES,
        *contract.RECEIPT_RELEASE_PRODUCER_FILES,
    ):
        assert (_ROOT / relative).is_file(), relative
    assert builder._PRODUCER_FILES[0] == "tools/build_us_receipt_enrichment_release.py"


def test_the_receipt_lineage_inherits_build_ps_own_calibration_bytes() -> None:
    assert (
        contract.RECEIPT_PARENT_FILES["calibration_diagnostics.json"]
        == (contract.PARENT_FILES["calibration_diagnostics.json"])
    )
    assert (
        contract.RECEIPT_PARENT_FILES["us_source_coverage.json"]
        == (contract.PARENT_FILES["us_source_coverage.json"])
    )
    assert contract.RECEIPT_PARENT_DATASET_SHA256 == (
        qualification.SPM_ROLE_PARENT_SHA256
    )
    assert contract.RECEIPT_COLUMNS == qualification.QUALIFIED_COLUMNS
    assert contract.RECEIPT_QUALIFICATION_FILE == qualification.RECEIPT_FILENAME
    # A new donor declares its release's dataset file; the qualification
    # names the pinned parent's child as a tagged donor published it.
    assert contract.RECEIPT_DATASET_FILENAME == "microcosm_us_2024_receipt_qualified.h5"
    assert contract.TAGGED_RECEIPT_DATASET_FILENAME == (
        "populace_us_2024_receipt_qualified.h5"
    )
    assert contract.RECEIPT_RELEASE_ID == US_RECEIPT_DONOR_RELEASE_ID


def test_the_shared_verifier_is_the_one_the_qualification_calls() -> None:
    from microcosm.data import h5_boolean_append

    assert qualification.compare_boolean_append is (
        h5_boolean_append.compare_boolean_append
    )
    assert qualification.DonorQualificationError is (
        h5_boolean_append.BooleanAppendError
    )
