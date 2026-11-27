"""Tests split from packages/microcosm-data/tests/test_release.py."""

# ruff: noqa: F403, F405

# The annual build-folder fixtures, which build on this module's release_dir.
from test_support.microcosm_data.annual_projections import (  # noqa: F401
    US_ANNUAL_RELEASE,
    annual_build_folder,
    national_build,
)
from test_support.microcosm_data.release import *


def test_prepare_build_folder_release_lists_each_file_with_its_role(
    release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    national = _as_uk_national_build_folder_release(release_dir)
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: []
    )

    prepared = release_module.prepare_build_folder_release(
        national, artifact_root=artifact_root
    )

    assert prepared.repository == "policyengine/populace-uk-private"
    assert (prepared.release, prepared.build) == (
        UK_NATIONAL_RELEASE_ID,
        UK_NATIONAL_BUILD,
    )
    assert prepared.release_role == "national"
    files = {item.name: item for item in prepared.files}
    assert {name: item.role for name, item in files.items()} == {
        "release_manifest.json": "release_manifest",
        "build_manifest.json": "build_manifest",
        "populace_us_2024.h5": "dataset",
        "populace_us_2024_calibration.npz": "calibration_weights",
        "calibration_diagnostics.json": "calibration_diagnostics",
        "us_source_coverage.json": "us_source_coverage",
        "release_certification.json": "release_certification",
        **{name: key for key, name in GATE_EVIDENCE_FILES.items()},
        US_STORED_INPUTS_EVIDENCE_FILE: US_STORED_INPUTS_EVIDENCE_KEY,
    }
    assert files["populace_us_2024.h5"].path == artifact_root / "populace_us_2024.h5"
    assert files["build_manifest.json"].path == national / "build_manifest.json"
    for item in prepared.files:
        assert item.sha256 == _sha256(item.path)
        assert item.bytes == item.path.stat().st_size


def test_prepare_build_folder_release_publishes_the_checksum_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    from test_support.microcosm_data.build_folders import register_uk_local_release

    from .test_local_area_contract import _write_local_bundle

    register_uk_local_release(monkeypatch, UK_LOCAL_RELEASE_ID)
    release_dir = _write_local_bundle(tmp_path, release_id=UK_LOCAL_RELEASE_ID)

    prepared = release_module.prepare_build_folder_release(
        release_dir, artifact_root=tmp_path
    )

    roles = {item.name: item.role for item in prepared.files}
    assert roles["sha256sums.txt"] == "checksums"
    assert roles["populace_us_2024_acs_local.h5"] == "dataset"
    assert prepared.release_role == "local_area"


def test_build_folder_identity_says_where_a_tagged_release_publishes(
    release_dir: Path,
) -> None:
    """A directory named after a US build ID is no release: US builds
    publish into the US releases, and the other UK releases take no new
    builds, once their repositories record their builds in releases.json."""
    with pytest.raises(
        ValueError, match="does not publish into build folders"
    ) as refusal:
        release_module.build_folder_identity(release_dir)

    assert "microcosm-us-2024-national" in str(refusal.value)
    assert "tools/publish_release.sh" not in str(refusal.value)
    assert "take no new builds" in str(refusal.value)


def test_no_build_folder_publication_takes_an_unregistered_uk_local_area_release(
    artifact_root: Path, tmp_path: Path
) -> None:
    """Build-folder publication takes no complete bundle of a UK local-area
    release that the registry does not name, and each refusal points to the
    registry (microcosm#1586)."""
    from .test_local_area_contract import _write_local_bundle

    unregistered = _write_local_bundle(tmp_path, release_id=UK_LOCAL_RELEASE_ID)
    refusals = []

    with pytest.raises(
        ValueError, match="does not publish into build folders"
    ) as refusal:
        release_module.build_folder_identity(unregistered)
    refusals.append(refusal)
    with pytest.raises(
        ValueError, match="does not publish into build folders"
    ) as refusal:
        release_module.prepare_build_folder_release(
            unregistered, artifact_root=artifact_root
        )
    refusals.append(refusal)

    for refusal in refusals:
        assert release_module.UNREGISTERED_UK_LOCAL_RELEASE_ADVICE in str(refusal.value)
        assert "take no new builds" not in str(refusal.value)


def test_prepare_build_folder_release_refuses_a_tagged_release(
    release_dir: Path, artifact_root: Path
) -> None:
    with pytest.raises(ValueError, match="does not publish into build folders"):
        release_module.prepare_build_folder_release(
            release_dir, artifact_root=artifact_root
        )


def test_prepare_build_folder_release_refuses_changed_or_missing_files(
    release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    national = _as_uk_national_build_folder_release(release_dir)
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: []
    )
    (artifact_root / "populace_us_2024.h5").write_bytes(b"other bytes")
    with pytest.raises(ValueError, match="populace_us_2024.h5 has sha256"):
        release_module.prepare_build_folder_release(
            national, artifact_root=artifact_root
        )
    (artifact_root / "populace_us_2024.h5").unlink()
    with pytest.raises(FileNotFoundError, match="populace_us_2024.h5"):
        release_module.prepare_build_folder_release(
            national, artifact_root=artifact_root
        )


def test_prepare_build_folder_release_publishes_to_one_repository(
    release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    national = _as_uk_national_build_folder_release(release_dir)
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: []
    )
    with pytest.raises(ValueError, match="names policyengine/populace-uk-private"):
        release_module.prepare_build_folder_release(
            national, artifact_root=artifact_root, repository="policyengine/other"
        )
    manifest_path = national / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["calibration_diagnostics"]["repo_id"] = "policyengine/other"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="more than one repository"):
        release_module.prepare_build_folder_release(
            national, artifact_root=artifact_root
        )


@pytest.mark.parametrize("status_code", [None, 403, 503])
def test_a_hub_that_could_not_be_asked_is_not_an_absent_file(
    status_code: int | None,
) -> None:
    """LocalEntryNotFoundError is also an EntryNotFoundError and a
    FileNotFoundError, but says nothing about the file; only the Hub's answer
    that it is absent counts (review of #1571)."""
    from test_support.microcosm_data import hub_errors

    assert not release_module._is_absent_entry_error(
        hub_errors.hub_unreachable("r", "f", status_code=status_code)
    )
    assert release_module._is_absent_entry_error(
        hub_errors.missing_on_hub("r", "f", "main")
    )


#: Synthetic contract warnings, worded as the dense line's disclosure walk
#: worded them until María's ruling of 2026-10-26 dropped it: a file and the
#: paths in it, never a value (microcosm#1301).
_CONTRACT_WARNINGS = [
    "synthetic.json carries 1 per-area support figure(s) (rows, ESS or distinct "
    "sources), which a dense release should withhold (microcosm#1301); first "
    "at ['areas[0].rows'].",
    "other.json cannot be read and walked as JSON, so the disclosure walk "
    "cannot rule out a per-area support figure in it (microcosm#1301).",
]


def _microcosm_named_artifacts(artifact_root: Path) -> Path:
    """The artifact root with the dataset and weights under the names new US
    builds declare."""
    for name in ("populace_us_2024.h5", "populace_us_2024_calibration.npz"):
        (artifact_root / name).rename(
            artifact_root / name.replace("populace_", "microcosm_")
        )
    return artifact_root


def test_prepare_build_folder_release_lists_a_us_national_build(
    release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    national = _as_us_national_build_folder_release(release_dir)
    root = _microcosm_named_artifacts(artifact_root)
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: []
    )

    prepared = release_module.prepare_build_folder_release(national, artifact_root=root)

    assert prepared.repository == "policyengine/populace-us"
    assert (prepared.release, prepared.build) == (US_NATIONAL_RELEASE_ID, RELEASE_ID)
    assert prepared.release_role == "national"
    roles = {item.name: item.role for item in prepared.files}
    assert roles["microcosm_us_2024.h5"] == "dataset"
    assert roles["microcosm_us_2024_calibration.npz"] == "calibration_weights"


def test_prepare_build_folder_release_lists_the_bound_gate_evidence(
    release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    """Route A remediation PR-3: a build publishes the files its release
    manifest declares, so the gate verdicts ship because the manifest binds
    them, each from the build's own directory."""
    national = _as_us_national_build_folder_release(release_dir)
    root = _microcosm_named_artifacts(artifact_root)
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: []
    )

    prepared = release_module.prepare_build_folder_release(national, artifact_root=root)

    files = {item.name: item for item in prepared.files}
    for name in GATE_EVIDENCE_FILES.values():
        assert files[name].path == national / name


def test_prepare_build_folder_release_repeats_the_contract_warnings_and_proceeds(
    release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    """A release contract warning is repeated as a warning and listed with the
    build, never refused (microcosm#1301)."""
    national = _as_us_national_build_folder_release(release_dir)
    root = _microcosm_named_artifacts(artifact_root)
    monkeypatch.setattr(
        release_module,
        "validate_release_dir",
        lambda _path, **_kwargs: list(_CONTRACT_WARNINGS),
    )

    with pytest.warns(UserWarning, match="release contract: synthetic.json carries"):
        prepared = release_module.prepare_build_folder_release(
            national, artifact_root=root
        )

    assert prepared.contract_warnings == tuple(_CONTRACT_WARNINGS)


def test_prepare_build_folder_release_publishes_a_build_whose_diagnostics_failed(
    release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    """A build whose calibration diagnostics generation failed still publishes
    its dataset, with a warning, and lists no diagnostics file."""
    national = _as_us_national_build_folder_release(release_dir)
    root = _microcosm_named_artifacts(artifact_root)
    (national / "calibration_diagnostics.json").unlink()
    manifest_path = national / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"].pop("calibration_diagnostics")
    manifest["calibration_diagnostics"] = {
        "status": "failed",
        "expected_schema_version": 8,
        "error_code": "validation_error",
        "message": "Target hierarchy is incomplete.",
    }
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: []
    )

    with pytest.warns(UserWarning, match="generation failed"):
        prepared = release_module.prepare_build_folder_release(
            national, artifact_root=root
        )

    names = {item.name for item in prepared.files}
    assert prepared.calibration_diagnostics_available is False
    assert "calibration_diagnostics.json" not in names
    assert "microcosm_us_2024.h5" in names


def test_prepare_build_folder_release_lists_an_annual_build(
    annual_build_folder: Path,  # noqa: F811
    artifact_root: Path,
) -> None:
    """An annual build passes its contract and publishes, as a build of the
    annual release whose role is national, the base national build's files,
    one H5 per projected year, the candidate manifest, its acceptance report
    and the per-year receipts."""
    prepared = release_module.prepare_build_folder_release(
        annual_build_folder, artifact_root=artifact_root
    )

    assert (prepared.release, prepared.build) == (
        US_ANNUAL_RELEASE,
        f"{RELEASE_ID}-annual-20261110T120000Z-a1b2c3d4",
    )
    assert prepared.release_role == "national"
    files = {item.name: item for item in prepared.files}
    assert files["microcosm_us_2024.h5"].role == "dataset"
    assert files["microcosm_us_2025.h5"].role == "dataset"
    assert files["microcosm_us_2025.h5"].path == artifact_root / "microcosm_us_2025.h5"
    assert {
        "annual_manifest.json",
        "annual_acceptance.json",
        "projection_2025.json",
        "microcosm_us_2024_calibration.npz",
        "calibration_diagnostics.json",
    } <= set(files)


def test_the_build_folder_refusal_advice_names_every_us_release() -> None:
    from microcosm.data.contract import US_RELEASES

    for release in US_RELEASES:
        assert release.removeprefix("microcosm-us-2024") in (
            release_module.TAGGED_RELEASE_ADVICE
        )


def test_prepare_build_folder_release_refuses_a_build_of_another_role(
    release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    """An ACS local-area build declares the local-area role; the national
    release, whose role is national, refuses it."""
    national = _as_us_national_build_folder_release(release_dir)
    manifest_path = national / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_role"] = "non_default_local_area"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: []
    )

    with pytest.raises(ValueError, match="role"):
        release_module.prepare_build_folder_release(
            national, artifact_root=_microcosm_named_artifacts(artifact_root)
        )


def test_prepare_build_folder_release_checks_an_evidence_build_with_its_contract(
    release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    """An evidence-tier build ships its failures on purpose, so the build-
    folder publisher checks it with the evidence contract, never the
    certified one."""
    national = _as_us_national_build_folder_release(release_dir)
    evidence = national.rename(national.with_name("microcosm-us-2024-evidence"))
    for name in ("release_manifest.json", "build_manifest.json"):
        manifest = json.loads((evidence / name).read_text())
        manifest["release"] = "microcosm-us-2024-evidence"
        (evidence / name).write_text(json.dumps(manifest))
    checked = []
    monkeypatch.setattr(
        release_module,
        "validate_release_dir",
        lambda _path, **_kwargs: pytest.fail("the certified contract ran"),
    )
    monkeypatch.setattr(
        release_module,
        "validate_evidence_release_dir",
        lambda path: checked.append(Path(path)),
    )

    prepared = release_module.prepare_build_folder_release(
        evidence, artifact_root=_microcosm_named_artifacts(artifact_root)
    )

    assert checked == [evidence]
    assert prepared.release == "microcosm-us-2024-evidence"
    assert prepared.release_role == "national"


def _with_release_type(release_dir: Path, release_type: str) -> Path:
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["release_type"] = release_type
    manifest_path.write_text(json.dumps(manifest))
    return release_dir


def test_prepare_build_folder_release_passes_parent_inputs_to_the_contract(
    release_dir: Path, artifact_root: Path, tmp_path: Path, monkeypatch
) -> None:
    """A source enrichment's contract replays its parent's H5 and the tested
    wheels, so the build-folder publisher passes them through."""
    national = _with_release_type(
        _as_us_national_build_folder_release(release_dir), "source_enrichment"
    )
    passed = {}

    def contract(_path, **kwargs):
        passed.update(kwargs)
        return []

    monkeypatch.setattr(release_module, "validate_release_dir", contract)
    parent_h5 = tmp_path / "parent.h5"
    wheel = tmp_path / "policyengine_us-2.2.1-py3-none-any.whl"

    release_module.prepare_build_folder_release(
        national,
        artifact_root=_microcosm_named_artifacts(artifact_root),
        parent_h5=parent_h5,
        compatibility_wheels=(wheel,),
    )

    assert passed["parent_h5"] == parent_h5
    assert passed["compatibility_wheels"] == (wheel,)


def test_prepare_build_folder_release_refuses_parent_inputs_for_a_plain_build(
    release_dir: Path, artifact_root: Path, tmp_path: Path
) -> None:
    national = _as_us_national_build_folder_release(release_dir)

    with pytest.raises(ValueError, match="source_enrichment"):
        release_module.prepare_build_folder_release(
            national,
            artifact_root=_microcosm_named_artifacts(artifact_root),
            parent_h5=tmp_path / "parent.h5",
        )
