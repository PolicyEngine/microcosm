"""Publishing behavior: contract-gated uploads and a last-written pointer.

The fake Hub client records every upload in order, so the suite asserts the
real guarantees — an invalid release uploads nothing, and ``latest.json``
lands strictly after the files it points at — rather than implementation
details.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from populace.data import ReleaseContractError
from populace.data.contract import REQUIRED_RELEASE_FILES
from populace.data.release import (
    LATEST_POINTER_PATH,
    LATEST_POINTER_SCHEMA_VERSION,
    latest_pointer_payload,
    latest_release,
    publish_release,
)

RELEASE_ID = "populace-us-2024-9f1260b-20260611"


class FakeHub:
    """Records uploads in order; serves downloads from what was uploaded."""

    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes]] = []

    def upload_file(
        self, *, path_or_fileobj, path_in_repo, repo_id, repo_type
    ) -> None:
        assert repo_type == "dataset"
        assert repo_id == "policyengine/populace-us"
        if isinstance(path_or_fileobj, bytes):
            content = path_or_fileobj
        else:
            content = Path(path_or_fileobj).read_bytes()
        self.uploads.append((path_in_repo, content))

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
                "dataset": {"filename": "populace_us_2024.h5", "sha256": "dc"},
                "gates": {"parity_gaps": 0},
            }
        )
    )
    (directory / "release_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "build": {"build_id": RELEASE_ID},
                "artifacts": {
                    "populace_us_2024": {
                        "path": "populace_us_2024.h5",
                        "repo_id": "policyengine/populace-us",
                        "sha256": "dc",
                    }
                },
            }
        )
    )
    (directory / "sound_ecps_replacement_comparison.json").write_text("{}")
    return directory


def test_pointer_payload_names_every_contract_file() -> None:
    payload = latest_pointer_payload(RELEASE_ID, updated_at="2026-06-11T13:53:15+00:00")
    assert payload["schema_version"] == LATEST_POINTER_SCHEMA_VERSION
    assert payload["release_id"] == RELEASE_ID
    assert set(payload["paths"]) == {
        name.removesuffix(".json") for name in REQUIRED_RELEASE_FILES
    }
    assert (
        payload["paths"]["build_manifest"]
        == f"releases/{RELEASE_ID}/build_manifest.json"
    )


def test_publish_uploads_pointer_last(hub: FakeHub, release_dir: Path) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    uploaded_paths = [path for path, _ in hub.uploads]
    assert uploaded_paths[-1] == LATEST_POINTER_PATH
    for filename in REQUIRED_RELEASE_FILES:
        assert f"releases/{RELEASE_ID}/{filename}" in uploaded_paths[:-1]


def test_invalid_release_uploads_nothing(hub: FakeHub, release_dir: Path) -> None:
    (release_dir / "build_manifest.json").unlink()
    with pytest.raises(ReleaseContractError):
        publish_release(release_dir, "policyengine/populace-us", api=hub)
    assert hub.uploads == []


def test_extra_files_ride_along_before_the_pointer(
    hub: FakeHub, release_dir: Path
) -> None:
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        extra_files=("calibration_diagnostics.json",),
    )
    uploaded_paths = [path for path, _ in hub.uploads]
    extra = f"releases/{RELEASE_ID}/calibration_diagnostics.json"
    assert extra in uploaded_paths
    assert uploaded_paths.index(extra) < uploaded_paths.index(LATEST_POINTER_PATH)


def test_missing_extra_file_fails_loudly(hub: FakeHub, release_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="calibration_diagnostics"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            extra_files=("calibration_diagnostics.json",),
        )


def test_publish_then_resolve_round_trips(
    hub: FakeHub, release_dir: Path
) -> None:
    published = publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
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
            json.dumps(
                {"schema_version": LATEST_POINTER_SCHEMA_VERSION + 1}
            ).encode(),
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
