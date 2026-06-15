"""The release contract: every published release looks the same, loudly.

These are behavioral tests against the failure modes already observed on the
Hub: a release with no build manifest at all (1abddeb), and two coexisting
release-manifest schemas (an unversioned early shape next to
``schema_version: 1``). A valid release passes silently; every broken release
fails with each violation named.
"""

import hashlib
import json
from pathlib import Path

import pytest

from populace.data import (
    RELEASE_MANIFEST_SCHEMA_VERSION,
    US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
    ReleaseContractError,
    required_release_files,
    validate_release_dir,
)

RELEASE_ID = "populace-us-2024-9f1260b-20260611"
UK_RELEASE_ID = "populace-uk-2024-9f1260b-20260611"
GIT_COMMIT = "5fa48f07436a806ad75ff76fd22cfb8613bddbe0"
DATASET_SHA = "d" * 64
CALIBRATION_SHA = "a" * 64
DIAGNOSTICS_SHA = "c" * 64
SOURCE_COVERAGE_SHA = "9" * 64
TARGET_SURFACE_SHA = "e" * 64
REGISTRY_VERSION = "registryabc123"


def _model_package(release_id: str) -> tuple[str, str]:
    if release_id.startswith("populace-uk-"):
        return ("policyengine-uk", "2.89.0")
    return ("policyengine-us", "1.729.0")


def _build_manifest(release_id: str = RELEASE_ID) -> dict:
    model_package, model_version = _model_package(release_id)
    return {
        "build_id": release_id,
        "builder": "populace",
        "build_sha": GIT_COMMIT[:7],
        "code": {
            "repository": "PolicyEngine/populace",
            "git_commit": GIT_COMMIT,
            "git_dirty": False,
        },
        "runtime": {
            "python": "3.14.0",
            "policyengine-core": "3.19.0",
            model_package: model_version,
        },
        "dataset": {"filename": "populace_us_2024.h5", "sha256": DATASET_SHA},
        "calibration": {
            "filename": "populace_us_2024_calibration.npz",
            "sha256": CALIBRATION_SHA,
            "target_surface": {"sha256": TARGET_SURFACE_SHA, "n_targets": 1},
            "target_registry": {"version": REGISTRY_VERSION, "n_specs": 1},
        },
        "gates": {"exported_nonzero": {"passed": True}},
    }


def _release_manifest(
    release_id: str = RELEASE_ID,
    *,
    diagnostics_sha: str = DIAGNOSTICS_SHA,
    source_coverage_sha: str = SOURCE_COVERAGE_SHA,
) -> dict:
    model_package, model_version = _model_package(release_id)
    manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {"name": "populace-data", "version": "0.1.0"},
        "default_datasets": {"national": "populace_us_2024"},
        "build": {
            "build_id": release_id,
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.19.0",
            },
            "built_with_model_package": {
                "name": model_package,
                "version": model_version,
            },
        },
        "artifacts": {
            "populace_us_2024": {
                "kind": "microdata",
                "path": "populace_us_2024.h5",
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "sha256": DATASET_SHA,
            },
            "populace_us_2024_calibration": {
                "kind": "calibration",
                "path": "populace_us_2024_calibration.npz",
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "sha256": CALIBRATION_SHA,
            },
            "calibration_diagnostics": {
                "kind": "diagnostics",
                "path": "calibration_diagnostics.json",
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "sha256": diagnostics_sha,
            },
        },
    }
    if release_id.startswith("populace-us-"):
        manifest["artifacts"]["us_source_coverage"] = {
            "kind": "diagnostics",
            "path": US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
            "repo_id": "policyengine/populace-us",
            "revision": release_id,
            "sha256": source_coverage_sha,
        }
    return manifest


def _calibration_diagnostics() -> dict:
    return {
        "schema_version": 2,
        "weight_entity": "household",
        "options": {"epochs": 120},
        "target_surface": {
            "schema_version": 1,
            "weight_entity": "household",
            "n_targets": 1,
            "n_records": 2,
            "constraint_matrix": {"rows": 1, "columns": 2, "nnz": 2},
            "sha256": TARGET_SURFACE_SHA,
            "names_sha256": "b" * 64,
            "values_sha256": "f" * 64,
        },
        "target_registry": {
            "country": "us",
            "version": REGISTRY_VERSION,
            "n_specs": 1,
        },
        "loss_trajectory": [1.0, 0.5],
        "skipped": [],
        "targets": [
            {
                "name": "population@2024",
                "target_name": "population",
                "period": 2024,
                "entity": "household",
                "aggregation": "count",
                "measure": None,
                "filter": None,
                "source": "Census PEP 2024",
                "metadata": {},
                "target": 1.0,
                "compiled_target": 1.0,
                "initial_estimate": 0.8,
                "final_estimate": 1.0,
                "relative_error": 0.0,
                "within_tolerance": True,
                "registry": {"family": "cbo"},
            }
        ],
    }


def _source_coverage_diagnostics() -> dict:
    return {
        "schema_version": 1,
        "classification": "release_gate",
        "source_contract": {
            "name": "us_source_coverage",
            "arch_commit": "5fa48f07436a806ad75ff76fd22cfb8613bddbe0",
        },
        "gate": {
            "name": "us_source_coverage",
            "passed": True,
            "failures": [],
        },
        "coverage_summary": {
            "hard_target": {
                "families": 9,
                "package_aliases": 38,
                "covered_package_aliases": 38,
                "missing_package_aliases": 0,
                "reviewed_excluded_package_aliases": 0,
            },
            "validation_only": {"families": 6, "activated_families": 0},
            "source_gap": {"families": 6, "missing_source_packages": 11},
        },
        "hard_target_families": {"population_age_sex": {}},
        "validation_only_families": {"census_cps_spm": {}},
        "source_gap_families": {"usda_wic": {}},
        "active_target_aliases": ["census-pep-2024-national-age-sex"],
        "active_target_families": [],
        "missing_hard_targets": [],
        "reviewed_exclusions": {},
        "validation_only_activated": [],
        "fiscal_target_sources": {
            "cbo": {
                "label": "Congressional Budget Office revenue projections",
                "target_count": 1,
                "sources": ["Census PEP 2024"],
                "reference_urls": ["https://example.test/source"],
            }
        },
    }


@pytest.fixture
def release_dir(tmp_path: Path) -> Path:
    """A complete, contract-valid release directory."""
    directory = tmp_path / "releases" / RELEASE_ID
    directory.mkdir(parents=True)
    (directory / "build_manifest.json").write_text(json.dumps(_build_manifest()))
    (directory / "calibration_diagnostics.json").write_text(
        json.dumps(_calibration_diagnostics())
    )
    (directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(
        json.dumps(_source_coverage_diagnostics())
    )
    diagnostics_sha = _sha256(directory / "calibration_diagnostics.json")
    source_coverage_sha = _sha256(directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE)
    (directory / "release_manifest.json").write_text(
        json.dumps(
            _release_manifest(
                diagnostics_sha=diagnostics_sha,
                source_coverage_sha=source_coverage_sha,
            )
        )
    )
    return directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_and_refresh_manifest_hash(
    release_dir: Path,
    *,
    filename: str,
    artifact_key: str,
    payload: dict,
) -> None:
    (release_dir / filename).write_text(json.dumps(payload))
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][artifact_key]["sha256"] = _sha256(release_dir / filename)
    manifest_path.write_text(json.dumps(manifest))


def test_a_complete_release_passes(release_dir: Path) -> None:
    validate_release_dir(release_dir)


@pytest.mark.parametrize("filename", required_release_files(RELEASE_ID))
def test_each_required_file_is_named_when_missing(
    release_dir: Path, filename: str
) -> None:
    (release_dir / filename).unlink()
    with pytest.raises(ReleaseContractError, match=filename):
        validate_release_dir(release_dir)


def test_non_us_release_does_not_require_us_source_coverage(tmp_path: Path) -> None:
    directory = tmp_path / "releases" / UK_RELEASE_ID
    directory.mkdir(parents=True)
    (directory / "build_manifest.json").write_text(
        json.dumps(_build_manifest(UK_RELEASE_ID))
    )
    (directory / "calibration_diagnostics.json").write_text(
        json.dumps(_calibration_diagnostics())
    )
    (directory / "release_manifest.json").write_text(
        json.dumps(
            _release_manifest(
                UK_RELEASE_ID,
                diagnostics_sha=_sha256(directory / "calibration_diagnostics.json"),
            )
        )
    )

    validate_release_dir(directory)
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE not in required_release_files(
        UK_RELEASE_ID
    )


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


def test_release_manifest_must_list_calibration_diagnostics(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"].pop("calibration_diagnostics")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="calibration_diagnostics"):
        validate_release_dir(release_dir)


def test_release_manifest_must_list_us_source_coverage_for_us_release(
    release_dir: Path,
) -> None:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["artifacts"].pop("us_source_coverage")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE in failures


def test_release_manifest_local_calibration_diagnostics_hash_must_match(
    release_dir: Path,
) -> None:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["artifacts"]["calibration_diagnostics"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "artifact 'calibration_diagnostics' declares sha256" in failures
    assert "calibration_diagnostics.json" in failures


def test_release_manifest_local_us_source_coverage_hash_must_match(
    release_dir: Path,
) -> None:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["artifacts"]["us_source_coverage"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "artifact 'us_source_coverage' declares sha256" in failures
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE in failures


def test_build_manifest_requires_clean_git_commit(release_dir: Path) -> None:
    manifest = _build_manifest()
    manifest["code"]["git_dirty"] = True
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "code.git_dirty" in failures


def test_target_surface_hash_must_match_between_manifest_and_diagnostics(
    release_dir: Path,
) -> None:
    manifest = _build_manifest()
    manifest["calibration"]["target_surface"]["sha256"] = "1" * 64
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "target_surface.sha256 must match" in failures


def test_target_registry_version_must_match_between_manifest_and_diagnostics(
    release_dir: Path,
) -> None:
    manifest = _build_manifest()
    manifest["calibration"]["target_registry"]["version"] = "other"
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "target_registry.version must match" in failures


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


def test_malformed_us_source_coverage_diagnostics_is_rejected(
    release_dir: Path,
) -> None:
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(
        json.dumps({"schema_version": 1})
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE in failures
    assert "coverage_summary" in failures


def test_failed_us_source_coverage_diagnostics_is_rejected(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["gate"] = {
        "name": "us_source_coverage",
        "passed": False,
        "failures": ["social_security_ssi/ssa-ssi-table-7b1-2024 missing"],
    }
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "gate.passed must be true" in failures
    assert "gate.failures must be empty" in failures


def test_us_source_coverage_reviewed_exclusions_need_reasons(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["reviewed_exclusions"] = {"ssa-ssi-table-7b1-2024": ""}
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "reviewed_exclusions need non-empty string reasons" in failures


def test_us_source_coverage_requires_fiscal_target_sources(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    del payload["fiscal_target_sources"]
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "fiscal_target_sources" in failures


def test_us_source_coverage_must_cover_calibrated_families(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["fiscal_target_sources"] = {}
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "fiscal_target_sources must cover every calibrated target family" in failures


def test_us_source_coverage_must_not_claim_uncalibrated_families(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["fiscal_target_sources"]["irs_soi"] = {
        "label": "Internal Revenue Service Statistics of Income",
        "target_count": 1,
        "sources": ["SOI publication"],
        "reference_urls": ["https://example.test/soi"],
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=payload,
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "unexpected ['irs_soi']" in failures


def test_us_source_coverage_target_counts_match_calibration(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["fiscal_target_sources"]["cbo"]["target_count"] = 2
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=payload,
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "fiscal_target_sources['cbo'].target_count is 2" in failures
    assert "has 1 calibrated target(s)" in failures


def test_build_manifest_requires_runtime_versions(release_dir: Path) -> None:
    manifest = _build_manifest()
    del manifest["runtime"]
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "runtime" in failures


def test_build_manifest_rejects_unknown_runtime_versions(release_dir: Path) -> None:
    manifest = _build_manifest()
    manifest["runtime"]["policyengine-us"] = "not-installed"
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "runtime.policyengine-us" in failures


def test_release_manifest_must_include_dataset_root_artifact(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    del manifest["artifacts"]["populace_us_2024"]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "dataset root artifact" in failures


def test_release_manifest_requires_policyengine_certification_shape(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    del manifest["data_package"]
    del manifest["default_datasets"]
    del manifest["build"]["built_with_model_package"]
    del manifest["artifacts"]["populace_us_2024"]["revision"]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "data_package" in failures
    assert "default_datasets" in failures
    assert "build.built_with_model_package" in failures
    assert "artifact 'populace_us_2024' is missing 'revision'" in failures


def test_release_manifest_default_dataset_must_name_artifact(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["default_datasets"]["national"] = "missing_dataset"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "default_datasets.national" in failures
    assert "missing_dataset" in failures


def test_release_manifest_default_dataset_must_be_microdata_root_artifact(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["default_datasets"]["national"] = "calibration_diagnostics"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "default_datasets.national" in failures
    assert "not 'microdata'" in failures
    assert "dataset root artifact" in failures


def test_release_manifest_default_dataset_hash_must_match_build_manifest(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "default dataset artifact" in failures
    assert "matching build_manifest.json" in failures


def test_release_manifest_artifact_revisions_must_pin_release_tag(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"]["revision"] = "main"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "artifact 'populace_us_2024' revision" in failures
    assert RELEASE_ID in failures


def test_release_manifest_root_artifact_hashes_match_build_manifest(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "sha256 matching build_manifest.json" in failures


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
