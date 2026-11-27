"""Microcosm neither reads nor writes ``latest*.json`` pointer files.

Every release repository lists its builds in ``releases.json``, and each
repository's switch to it deleted its pointer files (``latest.json``,
``latest-<name>.json`` and ``latest-evidence.json``). So the registry names no
pointer file, ``microcosm.data`` exports no pointer reader or writer, and the
loader refuses a repository without ``releases.json`` instead of following a
pointer file there.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from pathlib import Path

import huggingface_hub
import pytest

import microcosm.data
import microcosm.data.contract as contract
import microcosm.data.release as release
from microcosm.data import DatasetSpec, load
from microcosm.data.loader import resolve_certified
from test_support.microcosm_data.hub_errors import hub_unreachable, missing_on_hub

US_REPO_ID = "policyengine/populace-us"
#: A build ID of the tagged layout, which a pointer file named by its tag.
TAGGED_BUILD = "populace-us-2024-spm-20260915"
TAGGED_DATASET = b"a dataset at its tag"

#: Every public name the pointer-file reader and writer had.
POINTER_FILE_NAMES = (
    "LATEST_POINTER_PATH",
    "LINE_POINTER_PATH_TEMPLATE",
    "LATEST_EVIDENCE_POINTER_PATH",
    "LATEST_POINTER_SCHEMA_VERSION",
    "RELEASE_TIER_CERTIFIED",
    "RELEASE_TIER_EVIDENCE",
    "LatestPointer",
    "latest_release",
    "latest_line_release",
    "latest_evidence_release",
    "latest_pointer_payload",
    "line_pointer_payload",
    "latest_evidence_pointer_payload",
    "line_pointer_path",
)


def test_a_dataset_spec_names_no_pointer_file() -> None:
    names = {field.name for field in dataclasses.fields(DatasetSpec)}

    assert "pointer_path" not in names
    assert "release" in names


def test_a_dataset_spec_must_name_its_release() -> None:
    """The loader reads every entry's builds through ``releases.json``, so an
    entry without a release has nothing to read."""
    with pytest.raises(TypeError, match="release"):
        DatasetSpec(
            country="zz",
            year=2024,
            hf_repo="policyengine/populace-zz",
            filename="populace_zz_2024.h5",
            engine_module="policyengine_zz.data",
            engine_class="ZZSingleYearDataset",
            engine_package="policyengine-zz",
        )


@pytest.mark.parametrize("name", POINTER_FILE_NAMES)
def test_microcosm_data_exports_no_pointer_file_reader_or_writer(name: str) -> None:
    assert not hasattr(microcosm.data, name)
    assert name not in microcosm.data.__all__
    assert not hasattr(release, name)
    assert name not in release.__all__


@pytest.mark.parametrize(
    "name", ["_read_pointer", "_LINE_RE", "_current_evidence_build"]
)
def test_the_release_module_keeps_no_private_pointer_reader(name: str) -> None:
    assert not hasattr(release, name)


@pytest.mark.parametrize(
    "name",
    [
        "line_for_release_id",
        "dataset_role_for_line",
        "_uk_line_cut_tag_re",
        "_UK_NATIONAL_REVISION_SUFFIX_RE",
    ],
)
def test_the_contract_keeps_no_pointer_file_name_rules(name: str) -> None:
    """Only the pointer files' ``latest-<line>.json`` names and their tags
    used these."""
    assert not hasattr(contract, name)
    assert name not in contract.__all__


def test_load_has_no_unverified_root_option() -> None:
    """No repository holds a root dataset file any more."""
    assert "unverified_root" not in inspect.signature(load).parameters


class _TaggedLayoutHub:
    """A repository in the tagged layout: a pointer file at its root, and the
    build's manifest and dataset at the build's tag, but no ``releases.json``.
    Records every file requested."""

    def __init__(self, tmp_path: Path, *, offline: bool = False) -> None:
        self.tmp_path = tmp_path
        self.offline = offline
        self.requested: list[str] = []
        self.files: dict[tuple[str, str | None], bytes] = {
            ("latest.json", None): json.dumps(
                {
                    "schema_version": 1,
                    "release_id": TAGGED_BUILD,
                    "updated_at": "2026-09-15T00:00:00+00:00",
                    "paths": {
                        name.removesuffix(".json"): f"releases/{TAGGED_BUILD}/{name}"
                        for name in contract.required_release_files(TAGGED_BUILD)
                    },
                }
            ).encode(),
            (f"releases/{TAGGED_BUILD}/release_manifest.json", TAGGED_BUILD): (
                json.dumps(_tagged_release_manifest()).encode()
            ),
            ("populace_us_2024.h5", TAGGED_BUILD): TAGGED_DATASET,
        }

    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str | None = None,
    ) -> str:
        assert (repo_id, repo_type) == (US_REPO_ID, "dataset")
        self.requested.append(filename)
        if filename == "releases.json" and self.offline:
            raise hub_unreachable(repo_id, filename, revision)
        content = self.files.get((filename, revision))
        if content is None:
            raise missing_on_hub(repo_id, filename, revision)
        local = self.tmp_path / (revision or "main") / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
        return str(local)


def _tagged_release_manifest() -> dict:
    return {
        "schema_version": 1,
        "default_datasets": {"national": "populace_us_2024"},
        "build": {
            "build_id": TAGGED_BUILD,
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.26.11",
            },
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": "1.764.6",
            },
        },
        "compatible_core_packages": [
            {"name": "policyengine-core", "specifier": "==3.26.11"}
        ],
        "compatible_model_packages": [
            {"name": "policyengine-us", "specifier": "==1.764.6"}
        ],
        "artifacts": {
            "populace_us_2024": {
                "kind": "microdata",
                "path": "populace_us_2024.h5",
                "repo_id": US_REPO_ID,
                "revision": TAGGED_BUILD,
                "sha256": hashlib.sha256(TAGGED_DATASET).hexdigest(),
            }
        },
    }


def test_the_loader_refuses_a_repository_without_releases_json(
    tmp_path: Path,
) -> None:
    """A repository with only a pointer file is refused, naming
    ``releases.json``; the pointer file is never requested."""
    hub = _TaggedLayoutHub(tmp_path)

    with pytest.raises(ValueError, match="releases.json") as refusal:
        resolve_certified("us", 2024, hub_download=hub)

    assert US_REPO_ID in str(refusal.value)
    assert "latest" not in str(refusal.value)
    assert hub.requested == ["releases.json"]


def test_offline_the_loader_refuses_a_releases_json_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline, a ``releases.json`` that was never cached is refused too,
    saying it is not in the cache; no cached pointer file is read instead."""
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    hub = _TaggedLayoutHub(tmp_path, offline=True)

    with pytest.raises(ValueError, match="releases.json") as refusal:
        resolve_certified("us", 2024, hub_download=hub)

    assert "cache" in str(refusal.value)
    assert hub.requested == ["releases.json"]
