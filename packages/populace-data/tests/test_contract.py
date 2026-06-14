"""The release contract: every published release looks the same, loudly.

These are behavioral tests against the failure modes already observed on the
Hub: a release with no build manifest at all (1abddeb), and two coexisting
release-manifest schemas (an unversioned early shape next to
``schema_version: 1``). A valid release passes silently; every broken release
fails with each violation named.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from populace.data import (
    RELEASE_MANIFEST_SCHEMA_VERSION,
    REQUIRED_RELEASE_FILES,
    ReleaseContractError,
    validate_release_dir,
)

RELEASE_ID = "populace-us-2024-9f1260b-20260611"


def _build_manifest(release_id: str = RELEASE_ID) -> dict:
    return {
        "build_id": release_id,
        "builder": "populace",
        "dataset": {"filename": "populace_us_2024.h5", "sha256": "dc75c0"},
        "calibration": {
            "filename": "populace_us_2024_calibration.npz",
            "sha256": "a3da2f",
        },
        "gates": {"exported_nonzero": {"passed": True}},
    }


def _release_manifest(release_id: str = RELEASE_ID) -> dict:
    return {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {"name": "populace-data", "version": "0.1.0"},
        "build": {"build_id": release_id},
        "artifacts": {
            "populace_us_2024": {
                "kind": "microdata",
                "path": "populace_us_2024.h5",
                "repo_id": "policyengine/populace-us",
                "sha256": "dc75c0",
            }
        },
    }


def _calibration_diagnostics() -> dict:
    return {
        "schema_version": 1,
        "weight_entity": "household",
        "options": {"epochs": 120},
        "loss_trajectory": [1.0, 0.5],
        "skipped": [],
        "targets": [
            {
                "name": "population",
                "target": 1.0,
                "initial_estimate": 0.8,
                "final_estimate": 1.0,
                "relative_error": 0.0,
                "within_tolerance": True,
            }
        ],
    }


@pytest.fixture
def release_dir(tmp_path: Path) -> Path:
    """A complete, contract-valid release directory."""
    directory = tmp_path / "releases" / RELEASE_ID
    directory.mkdir(parents=True)
    (directory / "build_manifest.json").write_text(json.dumps(_build_manifest()))
    (directory / "release_manifest.json").write_text(json.dumps(_release_manifest()))
    (directory / "calibration_diagnostics.json").write_text(
        json.dumps(_calibration_diagnostics())
    )
    return directory


def test_a_complete_release_passes(release_dir: Path) -> None:
    validate_release_dir(release_dir)


@pytest.mark.parametrize("filename", REQUIRED_RELEASE_FILES)
def test_each_required_file_is_named_when_missing(
    release_dir: Path, filename: str
) -> None:
    (release_dir / filename).unlink()
    with pytest.raises(ReleaseContractError, match=filename):
        validate_release_dir(release_dir)


def test_the_1abddeb_shape_is_rejected(release_dir: Path) -> None:
    """The regression: a release with only an unversioned release manifest."""
    (release_dir / "build_manifest.json").unlink()
    (release_dir / "calibration_diagnostics.json").unlink()
    (release_dir / "release_manifest.json").write_text(
        json.dumps(
            {
                "release_id": RELEASE_ID,
                "country_id": "us",
                "artifacts": {},
                "validation": {},
            }
        )
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "build_manifest.json" in failures
    assert "schema_version" in failures


def test_schema_drift_is_rejected_by_version(release_dir: Path) -> None:
    manifest = _release_manifest()
    manifest["schema_version"] = RELEASE_MANIFEST_SCHEMA_VERSION + 1
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="schema_version"):
        validate_release_dir(release_dir)


def test_build_id_mismatch_names_both_ids(release_dir: Path) -> None:
    (release_dir / "build_manifest.json").write_text(
        json.dumps(_build_manifest("populace-us-2024-other-20260101"))
    )
    with pytest.raises(ReleaseContractError, match="populace-us-2024-other"):
        validate_release_dir(release_dir)


def test_release_manifest_build_id_must_match_directory(
    release_dir: Path,
) -> None:
    manifest = _release_manifest("populace-us-2024-other-20260101")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="build.build_id"):
        validate_release_dir(release_dir)


def test_artifact_entries_must_carry_provenance(release_dir: Path) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"].pop("sha256")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="sha256"):
        validate_release_dir(release_dir)


def test_unparseable_manifest_is_a_named_failure(release_dir: Path) -> None:
    (release_dir / "build_manifest.json").write_text("{not json")
    with pytest.raises(ReleaseContractError, match="not valid JSON"):
        validate_release_dir(release_dir)


def test_malformed_calibration_diagnostics_is_rejected(
    release_dir: Path,
) -> None:
    (release_dir / "calibration_diagnostics.json").write_text(
        json.dumps({"schema_version": 1})
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "calibration_diagnostics.json" in failures
    assert "targets" in failures


def test_all_failures_reported_at_once(release_dir: Path) -> None:
    """A publisher sees the full repair list, not one failure per run."""
    (release_dir / "calibration_diagnostics.json").unlink()
    manifest = _release_manifest()
    del manifest["schema_version"]
    manifest["artifacts"] = {}
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    assert len(excinfo.value.failures) >= 3


def test_a_missing_directory_is_a_contract_error(tmp_path: Path) -> None:
    with pytest.raises(ReleaseContractError, match="is not a directory"):
        validate_release_dir(tmp_path / "releases" / "nope")
