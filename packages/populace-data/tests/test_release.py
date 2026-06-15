"""Publishing behavior: contract-gated uploads and a last-written pointer.

The fake Hub client records every upload in order, so the suite asserts the
real guarantees — an invalid release uploads nothing, and ``latest.json``
lands strictly after the files it points at — rather than implementation
details.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from populace.data import ReleaseContractError
from populace.data.contract import (
    US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
    required_release_files,
)
from populace.data.release import (
    LATEST_POINTER_PATH,
    LATEST_POINTER_SCHEMA_VERSION,
    latest_pointer_payload,
    latest_release,
    publish_release,
)

RELEASE_ID = "populace-us-2024-9f1260b-20260611"
GIT_COMMIT = "5fa48f07436a806ad75ff76fd22cfb8613bddbe0"
DATASET_SHA = "cfe0edd307e479920c6a177b316f944bc27839f89e081ede5218a32d6b6b16d8"
CALIBRATION_SHA = "ac31f2be76a0f8dc4da89b6935aa4b8b1b2e1bd4eb3d03b809333084f25b376e"
TARGET_SURFACE_SHA = "e" * 64
REGISTRY_VERSION = "registryabc123"


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


class FakeHub:
    """Records uploads in order; serves downloads from what was uploaded."""

    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes]] = []
        self.tags: list[dict[str, str | None]] = []

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type) -> None:
        assert repo_type == "dataset"
        assert repo_id == "policyengine/populace-us"
        if isinstance(path_or_fileobj, bytes):
            content = path_or_fileobj
        else:
            content = Path(path_or_fileobj).read_bytes()
        self.uploads.append((path_in_repo, content))
        return {"commit_hash": f"commit-{len(self.uploads)}"}

    def create_tag(self, *, repo_id, tag, repo_type, revision=None) -> None:
        assert repo_type == "dataset"
        assert repo_id == "policyengine/populace-us"
        self.tags.append({"tag": tag, "revision": revision})

    def hf_hub_download(self, *, repo_id, filename, repo_type) -> str:
        assert repo_type == "dataset"
        for path_in_repo, content in reversed(self.uploads):
            if path_in_repo == filename:
                local = self._download_dir / filename
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(content)
                return str(local)
        raise FileNotFoundError(filename)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    fake = FakeHub()
    fake._download_dir = tmp_path / "hub-cache"
    return fake


@pytest.fixture
def release_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "releases" / RELEASE_ID
    directory.mkdir(parents=True)
    (directory / "build_manifest.json").write_text(
        json.dumps(
            {
                "build_id": RELEASE_ID,
                "build_sha": GIT_COMMIT[:7],
                "code": {
                    "repository": "PolicyEngine/populace",
                    "git_commit": GIT_COMMIT,
                    "git_dirty": False,
                },
                "runtime": {
                    "python": "3.14.0",
                    "policyengine-us": "1.729.0",
                    "policyengine-core": "3.19.0",
                },
                "dataset": {
                    "filename": "populace_us_2024.h5",
                    "sha256": DATASET_SHA,
                },
                "calibration": {
                    "filename": "populace_us_2024_calibration.npz",
                    "sha256": CALIBRATION_SHA,
                    "target_surface": {
                        "sha256": TARGET_SURFACE_SHA,
                        "n_targets": 1,
                    },
                    "target_registry": {
                        "version": REGISTRY_VERSION,
                        "n_specs": 1,
                    },
                },
                "gates": {"parity_gaps": 0},
            }
        )
    )
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
            {
                "schema_version": 1,
                "data_package": {"name": "populace-data", "version": "0.1.0"},
                "default_datasets": {"national": "populace_us_2024"},
                "build": {
                    "build_id": RELEASE_ID,
                    "built_with_core_package": {
                        "name": "policyengine-core",
                        "version": "3.19.0",
                    },
                    "built_with_model_package": {
                        "name": "policyengine-us",
                        "version": "1.729.0",
                    },
                },
                "artifacts": {
                    "populace_us_2024": {
                        "kind": "microdata",
                        "path": "populace_us_2024.h5",
                        "repo_id": "policyengine/populace-us",
                        "revision": RELEASE_ID,
                        "sha256": DATASET_SHA,
                    },
                    "populace_us_2024_calibration": {
                        "kind": "calibration",
                        "path": "populace_us_2024_calibration.npz",
                        "repo_id": "policyengine/populace-us",
                        "revision": RELEASE_ID,
                        "sha256": CALIBRATION_SHA,
                    },
                    "calibration_diagnostics": {
                        "kind": "diagnostics",
                        "path": "calibration_diagnostics.json",
                        "repo_id": "policyengine/populace-us",
                        "revision": RELEASE_ID,
                        "sha256": diagnostics_sha,
                    },
                    "us_source_coverage": {
                        "kind": "diagnostics",
                        "path": US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
                        "repo_id": "policyengine/populace-us",
                        "revision": RELEASE_ID,
                        "sha256": source_coverage_sha,
                    },
                },
            }
        )
    )
    return directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    (directory / "populace_us_2024.h5").write_bytes(b"h5 payload")
    (directory / "populace_us_2024_calibration.npz").write_bytes(b"npz payload")
    return directory


def test_pointer_payload_names_every_contract_file() -> None:
    payload = latest_pointer_payload(RELEASE_ID, updated_at="2026-06-11T13:53:15+00:00")
    assert payload["schema_version"] == LATEST_POINTER_SCHEMA_VERSION
    assert payload["release_id"] == RELEASE_ID
    assert set(payload["paths"]) == {
        name.removesuffix(".json") for name in required_release_files(RELEASE_ID)
    }
    assert (
        payload["paths"]["build_manifest"]
        == f"releases/{RELEASE_ID}/build_manifest.json"
    )
    assert (
        payload["paths"]["us_source_coverage"]
        == f"releases/{RELEASE_ID}/{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE}"
    )


def test_publish_uploads_pointer_last(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    uploaded_paths = [path for path, _ in hub.uploads]
    assert uploaded_paths[-1] == LATEST_POINTER_PATH
    assert hub.tags == [{"tag": RELEASE_ID, "revision": "commit-6"}]
    for filename in required_release_files(RELEASE_ID):
        assert f"releases/{RELEASE_ID}/{filename}" in uploaded_paths[:-1]


def test_publish_uploads_root_artifacts_before_release_files(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    uploaded_paths = [path for path, _ in hub.uploads]
    assert uploaded_paths[:2] == [
        "populace_us_2024.h5",
        "populace_us_2024_calibration.npz",
    ]
    assert (
        uploaded_paths.index("populace_us_2024.h5")
        < uploaded_paths.index(f"releases/{RELEASE_ID}/build_manifest.json")
        < uploaded_paths.index(LATEST_POINTER_PATH)
    )


def test_publish_requires_artifact_root_for_root_artifacts(
    hub: FakeHub, release_dir: Path
) -> None:
    with pytest.raises(ValueError, match="pass artifact_root"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
        )
    assert hub.uploads == []


def test_missing_root_artifact_uploads_nothing(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    (artifact_root / "populace_us_2024_calibration.npz").unlink()
    with pytest.raises(FileNotFoundError, match="populace_us_2024_calibration"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )
    assert hub.uploads == []


def test_root_artifact_hash_mismatch_uploads_nothing(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    (artifact_root / "populace_us_2024.h5").write_bytes(b"wrong payload")
    with pytest.raises(ValueError, match="release artifact 'populace_us_2024.h5'"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )
    assert hub.uploads == []


def test_release_tag_is_created_before_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        create_tag=True,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    assert hub.tags == [{"tag": RELEASE_ID, "revision": "commit-6"}]
    assert hub.uploads[-1][0] == LATEST_POINTER_PATH


def test_release_id_artifact_revision_requires_release_tag(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    with pytest.raises(ValueError, match="must create the matching Hugging Face tag"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            create_tag=False,
        )
    assert hub.uploads == []
    assert hub.tags == []


def test_release_id_artifact_revision_rejects_tag_name_override(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    with pytest.raises(ValueError, match="tag_name must match the release id"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            tag_name="different-tag",
        )
    assert hub.uploads == []
    assert hub.tags == []


def test_invalid_release_uploads_nothing(hub: FakeHub, release_dir: Path) -> None:
    (release_dir / "build_manifest.json").unlink()
    with pytest.raises(ReleaseContractError):
        publish_release(release_dir, "policyengine/populace-us", api=hub)
    assert hub.uploads == []


def test_invalid_calibration_diagnostics_uploads_nothing(
    hub: FakeHub, release_dir: Path
) -> None:
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    with pytest.raises(ReleaseContractError, match="calibration_diagnostics"):
        publish_release(release_dir, "policyengine/populace-us", api=hub)
    assert hub.uploads == []


def test_nonstandard_nan_calibration_diagnostics_uploads_nothing(
    hub: FakeHub, release_dir: Path
) -> None:
    (release_dir / "calibration_diagnostics.json").write_text(
        '{"schema_version": 2, "targets": [], "loss_trajectory": [NaN], '
        '"skipped": [], "options": {}}'
    )
    with pytest.raises(ReleaseContractError, match="calibration_diagnostics"):
        publish_release(release_dir, "policyengine/populace-us", api=hub)
    assert hub.uploads == []


def test_extra_files_ride_along_before_the_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        extra_files=("calibration_diagnostics.json",),
    )
    uploaded_paths = [path for path, _ in hub.uploads]
    extra = f"releases/{RELEASE_ID}/calibration_diagnostics.json"
    assert extra in uploaded_paths
    assert uploaded_paths.index(extra) < uploaded_paths.index(LATEST_POINTER_PATH)


def test_missing_extra_file_fails_loudly(hub: FakeHub, release_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="support_audit"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            extra_files=("support_audit.json",),
        )
    assert hub.uploads == []


def test_publish_then_resolve_round_trips(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    published = publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    pointer = latest_release("policyengine/populace-us", api=hub)
    assert pointer.release_id == RELEASE_ID
    assert pointer.updated_at == "2026-06-11T13:53:15+00:00"
    assert pointer.paths == published["paths"]


def test_future_pointer_schema_is_refused(hub: FakeHub) -> None:
    hub.uploads.append(
        (
            LATEST_POINTER_PATH,
            json.dumps({"schema_version": LATEST_POINTER_SCHEMA_VERSION + 1}).encode(),
        )
    )
    with pytest.raises(ValueError, match="Upgrade populace-data"):
        latest_release("policyengine/populace-us", api=hub)


def test_pointer_without_release_id_is_refused(hub: FakeHub) -> None:
    hub.uploads.append(
        (
            LATEST_POINTER_PATH,
            json.dumps({"schema_version": LATEST_POINTER_SCHEMA_VERSION}).encode(),
        )
    )
    with pytest.raises(ValueError, match="release_id"):
        latest_release("policyengine/populace-us", api=hub)


def test_pointer_without_contract_paths_is_refused(hub: FakeHub) -> None:
    hub.uploads.append(
        (
            LATEST_POINTER_PATH,
            json.dumps(
                {
                    "schema_version": LATEST_POINTER_SCHEMA_VERSION,
                    "release_id": RELEASE_ID,
                    "paths": {"build_manifest": "releases/x/build_manifest.json"},
                }
            ).encode(),
        )
    )
    with pytest.raises(ValueError, match="paths"):
        latest_release("policyengine/populace-us", api=hub)


def test_pointer_with_swapped_contract_path_is_refused(hub: FakeHub) -> None:
    payload = latest_pointer_payload(RELEASE_ID)
    payload["paths"]["build_manifest"] = (
        f"releases/{RELEASE_ID}/calibration_diagnostics.json"
    )
    hub.uploads.append((LATEST_POINTER_PATH, json.dumps(payload).encode()))

    with pytest.raises(ValueError, match="malformed=\\['build_manifest'\\]"):
        latest_release("policyengine/populace-us", api=hub)
