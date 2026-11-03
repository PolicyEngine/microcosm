"""Tests split from packages/microcosm-data/tests/test_loader.py."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

import microcosm.data.loader as loader
from microcosm.data import DEFAULT_VARIANT, download, latest_year, load, resolve
from microcosm.data.contract import (
    NATIONAL_DEFAULT_DATASET_ROLE,
    NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
)
from microcosm.data.registry import DatasetSpec
from microcosm.data.release import (
    LATEST_POINTER_PATH,
    latest_pointer_payload,
)
from test_support.microcosm_data.build_folders import (
    UK_LOCAL_BUILD,
    UK_NATIONAL_BUILD,
    build_folder_release_manifest,
)

#: Release id served by the hub fixture. Any well-formed id works: nothing in
#: this suite may assert which release the live Hub currently points at,
#: because re-certification moves ``latest.json`` without a commit here.
RELEASE_ID = "populace-us-2024-buildi-sparse-rmloss100-6e8e929-20260709T034135Z"
TAGGED_ARTIFACT = b"certified release artifact"
MUTABLE_ROOT_ARTIFACT = b"uncertified in-flight root artifact"
UK_REPO_ID = "policyengine/populace-uk-private"
UK_2023_RELEASE_ID = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
UK_NATIONAL_RELEASE_ID = "microcosm-uk-2024-25-national"
UK_LOCAL_RELEASE_ID = "microcosm-uk-2024-25-local-k15"
UK_TAGGED_ARTIFACT = b"certified UK line artifact"
#: An earlier national build, still listed in releases.json.
UK_EARLIER_BUILD = "uk-frs-calibration-attempt-20260901T000000Z-0badcafe"


def test_resolve_defaults_to_the_latest_year() -> None:
    spec = resolve("us")
    assert spec.year == latest_year("us")
    assert spec.variant == DEFAULT_VARIANT


def test_resolve_is_case_insensitive_on_country() -> None:
    assert resolve("US", 2024).key == ("us", 2024, DEFAULT_VARIANT)


def test_resolve_can_select_uk_compact_dataset() -> None:
    spec = resolve("uk", 2023)
    assert spec.key == ("uk", 2023, DEFAULT_VARIANT)
    assert spec.filename == "populace_uk_2023.h5"


def test_resolve_unknown_year_names_the_published_years() -> None:
    with pytest.raises(
        ValueError,
        match=r"published years for 'us' variant 'compact': \[2024\]",
    ):
        resolve("us", 1999)


def test_resolve_unknown_variant_names_the_published_variants() -> None:
    with pytest.raises(ValueError, match="variant 'local'"):
        resolve("uk", 2023, variant="local")


def test_resolve_unknown_country_names_the_published_countries() -> None:
    with pytest.raises(ValueError, match="published country variants"):
        resolve("atlantis")


def test_latest_year_unknown_country_raises() -> None:
    with pytest.raises(ValueError, match="No microcosm dataset for country"):
        latest_year("atlantis")


def _release_manifest(
    *,
    artifact_content: bytes = TAGGED_ARTIFACT,
    release_id: str = RELEASE_ID,
) -> dict:
    return {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
        "default_datasets": {"national": "populace_us_2024"},
        "build": {
            "build_id": release_id,
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
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "sha256": hashlib.sha256(artifact_content).hexdigest(),
            }
        },
    }


def _uk_release_manifest(
    *,
    release_id: str = UK_2023_RELEASE_ID,
    revision: str = UK_2023_RELEASE_ID,
    filename: str = "microcosm_uk_2024_25.h5",
    artifact_key: str = "microcosm_uk_2024_25",
    dataset_role: str | None = NATIONAL_DEFAULT_DATASET_ROLE,
    default_datasets: dict[str, str] | None = None,
    artifact_content: bytes = UK_TAGGED_ARTIFACT,
) -> dict:
    if default_datasets is None:
        default_datasets = {"national": artifact_key}
    manifest = {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
        "default_datasets": default_datasets,
        "build": {
            "build_id": release_id,
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.27.1",
            },
            "built_with_model_package": {
                "name": "policyengine-uk",
                "version": "2.89.2",
            },
        },
        "compatible_core_packages": [
            {"name": "policyengine-core", "specifier": "==3.27.1"}
        ],
        "compatible_model_packages": [
            {"name": "policyengine-uk", "specifier": "==2.89.2"}
        ],
        "artifacts": {
            artifact_key: {
                "kind": "microdata",
                "path": filename,
                "repo_id": UK_REPO_ID,
                "revision": revision,
                "sha256": hashlib.sha256(artifact_content).hexdigest(),
            }
        },
    }
    if dataset_role is not None:
        manifest["dataset_role"] = dataset_role
    return manifest


def _uk_local_spec() -> DatasetSpec:
    return DatasetSpec(
        country="uk",
        year=2025,
        variant="local",
        hf_repo=UK_REPO_ID,
        filename="microcosm_uk_2024_25_local_k15.h5",
        engine_module="policyengine_uk.data",
        engine_class="UKSingleYearDataset",
        engine_package="policyengine-uk",
        release=UK_LOCAL_RELEASE_ID,
    )


def _uk_build_folder_manifest(
    *,
    release: str = UK_NATIONAL_RELEASE_ID,
    build: str = UK_NATIONAL_BUILD,
    **options,
) -> dict:
    """A release manifest as a build in its own folder carries it."""
    return build_folder_release_manifest(
        _uk_release_manifest(release_id=release, **options),
        release=release,
        build=build,
    )


def _uk_local_manifest() -> dict:
    return _uk_build_folder_manifest(
        release=UK_LOCAL_RELEASE_ID,
        build=UK_LOCAL_BUILD,
        filename="microcosm_uk_2024_25_local_k15.h5",
        artifact_key="microcosm_uk_2024_25_local_k15",
        dataset_role=NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
        default_datasets={},
    )


def _releases_record(
    release: str = UK_NATIONAL_RELEASE_ID,
    *,
    role: str = "national",
    current: str | None = UK_NATIONAL_BUILD,
    builds: dict[str, bool] | None = None,
) -> dict:
    """A releases.json document: ``builds`` maps each build to ``withdrawn``."""
    if builds is None:
        builds = {UK_EARLIER_BUILD: False, UK_NATIONAL_BUILD: False}
    return {
        "schema_version": 1,
        "default": release,
        "releases": {
            release: {
                "role": role,
                "current": current,
                "builds": {
                    build: {
                        "published_at": "2026-09-20T12:00:00+00:00",
                        "withdrawn": withdrawn,
                    }
                    for build, withdrawn in builds.items()
                },
            }
        },
    }


def _build_folder_hub(
    tmp_path: Path,
    *,
    manifest: dict,
    record: dict | bytes | None = None,
    artifact_content: bytes = UK_TAGGED_ARTIFACT,
) -> FakeHubDownload:
    """Serve releases.json and one build folder, all read at ``main``."""
    hub = FakeHubDownload(tmp_path)
    hub.calls.clear()
    hub.files.clear()
    record = _releases_record() if record is None else record
    hub._add(
        tmp_path,
        UK_REPO_ID,
        "releases.json",
        None,
        record if isinstance(record, bytes) else json.dumps(record).encode(),
    )
    folder = f"releases/{manifest['release']}/{manifest['build']}"
    hub._add(
        tmp_path,
        UK_REPO_ID,
        f"{folder}/release_manifest.json",
        None,
        json.dumps(manifest).encode(),
    )
    artifact = next(
        entry
        for entry in manifest["artifacts"].values()
        if entry.get("kind") == "microdata"
    )
    hub._add(
        tmp_path, UK_REPO_ID, f"{folder}/{artifact['path']}", None, artifact_content
    )
    return hub


class FakeHubDownload:
    """Serve mutable and release-tagged files while recording every request."""

    def __init__(self, tmp_path: Path, *, manifest: dict | None = None) -> None:
        pointer = latest_pointer_payload(
            RELEASE_ID, updated_at="2026-07-09T04:00:00+00:00"
        )
        manifest = manifest or _release_manifest()
        self.tmp_path = tmp_path
        self.calls: list[tuple[str, str, str | None]] = []
        self.files: dict[tuple[str, str, str | None], Path] = {}
        self._add(
            tmp_path,
            "policyengine/populace-us",
            LATEST_POINTER_PATH,
            None,
            json.dumps(pointer).encode(),
        )
        self._add(
            tmp_path,
            "policyengine/populace-us",
            pointer["paths"]["release_manifest"],
            RELEASE_ID,
            json.dumps(manifest).encode(),
        )
        self._add(
            tmp_path,
            "policyengine/populace-us",
            "populace_us_2024.h5",
            RELEASE_ID,
            TAGGED_ARTIFACT,
        )
        self._add(
            tmp_path,
            "policyengine/populace-us",
            "populace_us_2024.h5",
            None,
            MUTABLE_ROOT_ARTIFACT,
        )

    def _add(
        self,
        tmp_path: Path,
        repo_id: str,
        filename: str,
        revision: str | None,
        content: bytes,
    ) -> None:
        revision_dir = revision or "main"
        local = tmp_path / "hub" / revision_dir / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
        self.files[(repo_id, filename, revision)] = local

    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str | None = None,
    ) -> str:
        assert repo_type == "dataset"
        self.calls.append((repo_id, filename, revision))
        try:
            return str(self.files[(repo_id, filename, revision)])
        except KeyError as exc:
            raise FileNotFoundError(
                f"{repo_id}/{filename} at revision {revision!r}"
            ) from exc


def _hub_with_pointer(
    tmp_path: Path,
    *,
    repo_id: str,
    pointer_path: str,
    pointer: dict,
    manifest: dict,
) -> FakeHubDownload:
    hub = FakeHubDownload(tmp_path)
    hub.calls.clear()
    hub.files.clear()
    revision = str(pointer.get("revision", pointer["release_id"]))
    hub._add(
        tmp_path,
        repo_id,
        pointer_path,
        None,
        json.dumps(pointer).encode(),
    )
    hub._add(
        tmp_path,
        repo_id,
        pointer["paths"]["release_manifest"],
        revision,
        json.dumps(manifest).encode(),
    )
    return hub


def _patch_engine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_version: str = "1.764.6",
    core_version: str = "3.26.11",
) -> list[str]:
    constructed: list[str] = []

    class FakeDataset:
        def __init__(self, *, file_path: str) -> None:
            constructed.append(file_path)
            self.file_path = file_path

    monkeypatch.setattr(
        loader.importlib,
        "import_module",
        lambda _name: SimpleNamespace(USSingleYearDataset=FakeDataset),
    )
    versions = {
        "policyengine-us": model_version,
        "policyengine-core": core_version,
    }
    monkeypatch.setattr(metadata, "version", versions.__getitem__)
    return constructed


def test_resolve_uk_default_is_the_national_release():
    """The national release is the default UK load, read through
    releases.json; the June 2023 release stays at its year and latest.json."""
    assert latest_year("uk") == 2025
    assert resolve("uk").key == ("uk", 2025, DEFAULT_VARIANT)
    assert resolve("uk").release == UK_NATIONAL_RELEASE_ID
    assert resolve("uk", 2023).release is None
    assert resolve("uk", 2023).pointer_path == "latest.json"
    with pytest.raises(ValueError, match="variant 'national'"):
        resolve("uk", 2025, variant="national")


def test_uk_national_load_follows_the_current_build_in_releases_json(
    tmp_path: Path,
) -> None:
    spec = resolve("uk", 2025)
    hub = _build_folder_hub(tmp_path, manifest=_uk_build_folder_manifest())

    certified = loader._resolve_certified_release(spec, hub_download=hub)

    folder = f"releases/{UK_NATIONAL_RELEASE_ID}/{UK_NATIONAL_BUILD}"
    assert certified.release_id == UK_NATIONAL_RELEASE_ID
    assert certified.build == UK_NATIONAL_BUILD
    assert certified.artifact_path == f"{folder}/microcosm_uk_2024_25.h5"
    assert certified.artifact_revision is None
    assert certified.model.built_version == "2.89.2"
    assert certified.core.built_version == "3.27.1"
    assert hub.calls == [
        (UK_REPO_ID, "releases.json", None),
        (UK_REPO_ID, f"{folder}/release_manifest.json", None),
    ]


def test_a_pinned_build_loads_that_build(tmp_path: Path) -> None:
    hub = _build_folder_hub(
        tmp_path, manifest=_uk_build_folder_manifest(build=UK_EARLIER_BUILD)
    )

    certified = loader._resolve_certified_release(
        resolve("uk", 2025), hub_download=hub, build=UK_EARLIER_BUILD
    )

    assert certified.build == UK_EARLIER_BUILD


def test_a_pinned_withdrawn_build_loads_with_a_warning(tmp_path: Path) -> None:
    record = _releases_record(builds={UK_EARLIER_BUILD: True, UK_NATIONAL_BUILD: False})
    hub = _build_folder_hub(
        tmp_path,
        manifest=_uk_build_folder_manifest(build=UK_EARLIER_BUILD),
        record=record,
    )

    with pytest.warns(UserWarning, match="has been withdrawn"):
        certified = loader._resolve_certified_release(
            resolve("uk", 2025), hub_download=hub, build=UK_EARLIER_BUILD
        )
    assert certified.build == UK_EARLIER_BUILD


@pytest.mark.parametrize(
    ("record", "build", "message"),
    [
        (
            _releases_record(builds={UK_NATIONAL_BUILD: False}),
            UK_EARLIER_BUILD,
            "is not listed in releases.json",
        ),
        (_releases_record(current=None), None, "has no current build"),
        (
            _releases_record("microcosm-uk-2024-25-dense", role="local_area"),
            None,
            "does not list release 'microcosm-uk-2024-25-national'",
        ),
        (b"not json", None, "is not a valid release record"),
        ({"schema_version": 2, "default": None, "releases": {}}, None, "valid"),
        (
            _releases_record(current=UK_EARLIER_BUILD, builds={UK_EARLIER_BUILD: True}),
            None,
            "is not a valid release record",
        ),
    ],
)
def test_releases_json_must_list_the_build_to_load(
    tmp_path: Path, record: dict | bytes, build: str | None, message: str
) -> None:
    hub = _build_folder_hub(
        tmp_path, manifest=_uk_build_folder_manifest(), record=record
    )

    with pytest.raises(ValueError, match=message):
        loader._resolve_certified_release(
            resolve("uk", 2025), hub_download=hub, build=build
        )


@pytest.mark.parametrize("field", ["release", "build"])
def test_a_manifest_naming_another_build_is_refused(tmp_path: Path, field) -> None:
    manifest = _uk_build_folder_manifest()
    hub = _build_folder_hub(tmp_path, manifest=manifest)
    manifest[field] = "uk-frs-calibration-attempt-20260101T000000Z-00000000"
    hub._add(
        tmp_path,
        UK_REPO_ID,
        f"releases/{UK_NATIONAL_RELEASE_ID}/{UK_NATIONAL_BUILD}/release_manifest.json",
        None,
        json.dumps(manifest).encode(),
    )

    with pytest.raises(ValueError, match="names release"):
        loader._resolve_certified_release(resolve("uk", 2025), hub_download=hub)


def test_a_build_folder_artifact_with_a_revision_is_refused(tmp_path: Path) -> None:
    manifest = _uk_build_folder_manifest()
    manifest["artifacts"]["microcosm_uk_2024_25"]["revision"] = "main"
    hub = _build_folder_hub(tmp_path, manifest=manifest)

    with pytest.raises(ValueError, match="pins a revision"):
        loader._resolve_certified_release(resolve("uk", 2025), hub_download=hub)


def test_the_dataset_artifact_must_have_the_dataset_role(tmp_path: Path) -> None:
    manifest = _uk_build_folder_manifest()
    manifest["artifacts"]["microcosm_uk_2024_25"]["role"] = "calibration_weights"
    hub = _build_folder_hub(tmp_path, manifest=manifest)

    with pytest.raises(ValueError, match="not 'dataset'"):
        loader._resolve_certified_release(resolve("uk", 2025), hub_download=hub)


def test_releases_json_role_must_match_the_manifest(tmp_path: Path) -> None:
    hub = _build_folder_hub(
        tmp_path,
        manifest=_uk_build_folder_manifest(),
        record=_releases_record(role="local_area"),
    )

    with pytest.raises(ValueError, match="listed with role 'local_area'"):
        loader._resolve_certified_release(resolve("uk", 2025), hub_download=hub)


def test_a_build_pin_needs_a_build_folder_release() -> None:
    def unexpected_download(**_kwargs) -> str:
        pytest.fail("a refused pin must fail before a Hub request")

    with pytest.raises(ValueError, match="pins a build only for releases"):
        loader._resolve_certified_release(
            resolve("us", 2024), hub_download=unexpected_download, build="x"
        )


def test_unverified_root_loading_is_refused_for_a_build_folder_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        loader,
        "_dataset_class",
        lambda _spec: pytest.fail("refused before the engine is imported"),
    )
    with pytest.raises(ValueError, match="no root file"):
        load("uk", 2025, unverified_root=True)


def test_download_verifies_the_build_folder_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = _build_folder_hub(tmp_path, manifest=_uk_build_folder_manifest())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)

    path = download("uk", 2025)

    folder = f"releases/{UK_NATIONAL_RELEASE_ID}/{UK_NATIONAL_BUILD}"
    assert path.read_bytes() == UK_TAGGED_ARTIFACT
    assert hub.calls[-1] == (UK_REPO_ID, f"{folder}/microcosm_uk_2024_25.h5", None)

    hub._add(
        tmp_path,
        UK_REPO_ID,
        f"{folder}/microcosm_uk_2024_25.h5",
        None,
        b"other bytes",
    )
    with pytest.raises(ValueError, match="sha256 verification failed"):
        download("uk", 2025)


def test_resolve_uk_2023_still_uses_latest_json(tmp_path: Path) -> None:
    spec = resolve("uk", 2023)
    manifest = _uk_release_manifest(
        release_id=UK_2023_RELEASE_ID,
        revision=UK_2023_RELEASE_ID,
        filename="populace_uk_2023.h5",
        artifact_key="populace_uk_2023",
        dataset_role=None,
    )
    pointer = latest_pointer_payload(
        UK_2023_RELEASE_ID,
        updated_at="2026-06-19T03:00:00+00:00",
    )
    hub = _hub_with_pointer(
        tmp_path,
        repo_id=UK_REPO_ID,
        pointer_path=LATEST_POINTER_PATH,
        pointer=pointer,
        manifest=manifest,
    )

    certified = loader._resolve_certified_release(spec, hub_download=hub)

    manifest_path = f"releases/{UK_2023_RELEASE_ID}/release_manifest.json"
    assert spec.pointer_path == LATEST_POINTER_PATH
    assert certified.release_id == UK_2023_RELEASE_ID
    assert certified.artifact_revision == UK_2023_RELEASE_ID
    assert hub.calls == [
        (UK_REPO_ID, LATEST_POINTER_PATH, None),
        (UK_REPO_ID, manifest_path, UK_2023_RELEASE_ID),
    ]


@pytest.mark.parametrize("pointer_path", ["current.json", "latest-national.json"])
def test_resolve_refuses_an_unsupported_pointer_path(pointer_path: str) -> None:
    spec = replace(resolve("us", 2024), pointer_path=pointer_path)

    def unexpected_download(**_kwargs) -> str:
        pytest.fail("an unsupported pointer path must fail before a Hub request")

    with pytest.raises(ValueError, match="Unsupported release pointer path"):
        loader._resolve_certified_release(spec, hub_download=unexpected_download)


def test_local_area_release_selects_its_only_microdata_artifact(
    tmp_path: Path,
) -> None:
    manifest = _uk_local_manifest()
    manifest["artifacts"]["calibration_report"] = {
        "kind": "diagnostics",
        "path": "calibration_diagnostics.json",
        "repo_id": UK_REPO_ID,
        "sha256": "0" * 64,
        "role": "calibration_diagnostics",
    }
    hub = _build_folder_hub(
        tmp_path,
        manifest=manifest,
        record=_releases_record(
            UK_LOCAL_RELEASE_ID,
            role="local_area",
            current=UK_LOCAL_BUILD,
            builds={UK_LOCAL_BUILD: False},
        ),
    )

    certified = loader._resolve_certified_release(_uk_local_spec(), hub_download=hub)

    folder = f"releases/{UK_LOCAL_RELEASE_ID}/{UK_LOCAL_BUILD}"
    assert certified.release_id == UK_LOCAL_RELEASE_ID
    assert certified.artifact_path == f"{folder}/microcosm_uk_2024_25_local_k15.h5"
    assert hub.calls == [
        (UK_REPO_ID, "releases.json", None),
        (UK_REPO_ID, f"{folder}/release_manifest.json", None),
    ]


def test_latest_json_refuses_a_local_area_manifest(tmp_path: Path) -> None:
    manifest = _uk_release_manifest(
        filename="microcosm_uk_2024_25_local_k15.h5",
        artifact_key="microcosm_uk_2024_25_local_k15",
        dataset_role=NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
        default_datasets={},
    )
    pointer = latest_pointer_payload(
        UK_2023_RELEASE_ID,
        updated_at="2026-09-20T12:00:00+00:00",
    )
    hub = _hub_with_pointer(
        tmp_path,
        repo_id=UK_REPO_ID,
        pointer_path=LATEST_POINTER_PATH,
        pointer=pointer,
        manifest=manifest,
    )
    spec = replace(_uk_local_spec(), release=None, pointer_path=LATEST_POINTER_PATH)

    with pytest.raises(
        ValueError,
        match=r"non_default_local_area.*build folder.*latest\.json",
    ):
        loader._resolve_certified_release(spec, hub_download=hub)


def _local_area_hub(tmp_path: Path, manifest: dict) -> FakeHubDownload:
    return _build_folder_hub(
        tmp_path,
        manifest=manifest,
        record=_releases_record(
            UK_LOCAL_RELEASE_ID,
            role="local_area",
            current=UK_LOCAL_BUILD,
            builds={UK_LOCAL_BUILD: False},
        ),
    )


def test_local_area_release_refuses_nonempty_default_datasets(tmp_path: Path) -> None:
    manifest = _uk_local_manifest()
    manifest["default_datasets"] = {"national": "microcosm_uk_2024_25_local_k15"}
    hub = _local_area_hub(tmp_path, manifest)

    with pytest.raises(ValueError, match=r"default_datasets is not empty"):
        loader._resolve_certified_release(_uk_local_spec(), hub_download=hub)


@pytest.mark.parametrize("microdata_count", [0, 2])
def test_local_area_release_requires_exactly_one_microdata_artifact(
    tmp_path: Path,
    microdata_count: int,
) -> None:
    manifest = _uk_local_manifest()
    hub = _local_area_hub(tmp_path, manifest)
    artifact = manifest["artifacts"]["microcosm_uk_2024_25_local_k15"]
    if microdata_count == 0:
        artifact["kind"] = "diagnostics"
    else:
        manifest["artifacts"]["second_microdata"] = {
            "kind": "microdata",
            "path": "second.h5",
            "repo_id": UK_REPO_ID,
            "sha256": "1" * 64,
            "role": "dataset",
        }
    hub._add(
        tmp_path,
        UK_REPO_ID,
        f"releases/{UK_LOCAL_RELEASE_ID}/{UK_LOCAL_BUILD}/release_manifest.json",
        None,
        json.dumps(manifest).encode(),
    )

    with pytest.raises(
        ValueError,
        match=rf"{microdata_count} microdata artifacts; expected exactly one",
    ):
        loader._resolve_certified_release(_uk_local_spec(), hub_download=hub)


def test_loader_refuses_an_unknown_dataset_role(tmp_path: Path) -> None:
    manifest = _uk_build_folder_manifest(dataset_role="research_preview")
    hub = _build_folder_hub(tmp_path, manifest=manifest)

    with pytest.raises(ValueError, match="unknown dataset_role 'research_preview'"):
        loader._resolve_certified_release(resolve("uk", 2025), hub_download=hub)


def test_download_uses_pointer_manifest_and_release_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = FakeHubDownload(tmp_path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)

    artifact = download("us", 2024)

    manifest_path = f"releases/{RELEASE_ID}/release_manifest.json"
    assert artifact.read_bytes() == TAGGED_ARTIFACT
    assert hub.calls == [
        ("policyengine/populace-us", LATEST_POINTER_PATH, None),
        ("policyengine/populace-us", manifest_path, RELEASE_ID),
        ("policyengine/populace-us", "populace_us_2024.h5", RELEASE_ID),
    ]


def test_resolve_certified_release_returns_pointer_pinned_metadata(
    tmp_path: Path,
) -> None:
    """Exact release-metadata equalities are pinned against the hub fixture."""
    hub = FakeHubDownload(tmp_path)

    certified = loader._resolve_certified_release(resolve("us", 2024), hub_download=hub)

    assert certified.release_id == RELEASE_ID
    assert certified.artifact_repo_id == "policyengine/populace-us"
    assert certified.artifact_path == "populace_us_2024.h5"
    assert certified.artifact_revision == RELEASE_ID
    assert certified.artifact_sha256 == hashlib.sha256(TAGGED_ARTIFACT).hexdigest()
    assert certified.model.name == "policyengine-us"
    assert certified.model.built_version == "1.764.6"
    assert certified.core.name == "policyengine-core"
    assert certified.core.built_version == "3.26.11"


def test_recertification_advances_the_pointer_and_resolution_follows(
    tmp_path: Path,
) -> None:
    """Publishing a successor release moves ``latest.json``; resolution follows.

    The pointer is the loader's only entry point — there is no API for
    requesting a superseded release id — so a re-certification changes what
    every consumer resolves without any commit in this repository.
    """
    hub = FakeHubDownload(tmp_path)
    spec = resolve("us", 2024)
    assert (
        loader._resolve_certified_release(spec, hub_download=hub).release_id
        == RELEASE_ID
    )

    successor_id = "populace-us-2024-buildj-sparse-rmloss100-75d5add-20260710T094201Z"
    successor_artifact = b"successor certified release artifact"
    pointer = latest_pointer_payload(
        successor_id, updated_at="2026-07-10T10:00:00+00:00"
    )
    hub._add(
        hub.tmp_path,
        "policyengine/populace-us",
        LATEST_POINTER_PATH,
        None,
        json.dumps(pointer).encode(),
    )
    hub._add(
        hub.tmp_path,
        "policyengine/populace-us",
        pointer["paths"]["release_manifest"],
        successor_id,
        json.dumps(
            _release_manifest(
                artifact_content=successor_artifact, release_id=successor_id
            )
        ).encode(),
    )

    certified = loader._resolve_certified_release(spec, hub_download=hub)

    assert certified.release_id == successor_id
    assert certified.artifact_revision == successor_id
    assert certified.artifact_sha256 == hashlib.sha256(successor_artifact).hexdigest()


def test_pointer_vs_root_divergence_never_serves_mutable_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = FakeHubDownload(tmp_path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)

    artifact = download("us", 2024)

    assert artifact.read_bytes() == TAGGED_ARTIFACT
    assert artifact.read_bytes() != MUTABLE_ROOT_ARTIFACT
    assert (
        "policyengine/populace-us",
        "populace_us_2024.h5",
        None,
    ) not in hub.calls


def test_sha_mismatch_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hub = FakeHubDownload(tmp_path)
    tagged = hub.files[("policyengine/populace-us", "populace_us_2024.h5", RELEASE_ID)]
    tagged.write_bytes(b"tampered tagged artifact")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)

    with pytest.raises(ValueError, match=r"sha256.*release.*expected.*observed"):
        download("us", 2024)


def test_missing_release_manifest_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = FakeHubDownload(tmp_path)
    manifest_path = f"releases/{RELEASE_ID}/release_manifest.json"
    del hub.files[("policyengine/populace-us", manifest_path, RELEASE_ID)]
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)

    with pytest.raises(FileNotFoundError, match=r"release_manifest\.json.*revision"):
        download("us", 2024)


@pytest.mark.parametrize(
    ("model_version", "core_version", "package", "installed", "required"),
    [
        ("1.800.0", "3.26.11", "policyengine-us", "1.800.0", "==1.764.6"),
        ("1.764.6", "3.30.0", "policyengine-core", "3.30.0", "==3.26.11"),
    ],
)
def test_engine_incompatibility_raises_before_dataset_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_version: str,
    core_version: str,
    package: str,
    installed: str,
    required: str,
) -> None:
    hub = FakeHubDownload(tmp_path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)
    constructed = _patch_engine(
        monkeypatch,
        model_version=model_version,
        core_version=core_version,
    )

    with pytest.raises(
        loader._CertifiedPackageCompatibilityError,
        match=rf"{package}.*{installed}.*{required}.*pip install",
    ):
        load("us", 2024)

    assert constructed == []


def test_load_constructs_dataset_after_release_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = FakeHubDownload(tmp_path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)
    constructed = _patch_engine(monkeypatch)

    dataset = load("us", 2024)

    assert dataset.file_path == constructed[0]
    assert Path(dataset.file_path).read_bytes() == TAGGED_ARTIFACT
    assert hub.calls[-1] == (
        "policyengine/populace-us",
        "populace_us_2024.h5",
        RELEASE_ID,
    )


def test_unverified_root_opt_out_warns_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = FakeHubDownload(tmp_path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)
    constructed = _patch_engine(monkeypatch)

    with pytest.warns(
        RuntimeWarning,
        match=r"UNVERIFIED.*latest\.json.*SHA-256.*compatibility",
    ):
        dataset = load("us", 2024, unverified_root=True)

    assert dataset.file_path == constructed[0]
    assert Path(dataset.file_path).read_bytes() == MUTABLE_ROOT_ARTIFACT
    assert hub.calls == [("policyengine/populace-us", "populace_us_2024.h5", None)]


def _hf_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}


def _is_offline_error(exc: Exception) -> bool:
    """Recognize transport/offline failures without hiding bad release metadata."""
    offline_names = {
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "LocalEntryNotFoundError",
        "OfflineModeIsEnabled",
        "ProxyError",
        "ReadTimeout",
    }
    closed_after_retry = isinstance(exc, RuntimeError) and str(exc) == (
        "Cannot send a request, as the client has been closed."
    )
    return (
        isinstance(exc, (ConnectionError, TimeoutError))
        or closed_after_retry
        or any(cls.__name__ in offline_names for cls in type(exc).__mro__)
    )


def _is_missing_pointer_error(exc: BaseException, pointer_path: str) -> bool:
    """The pointer file itself is absent at the repo root: an
    ``EntryNotFoundError`` naming that path (the Hub's answer for a file that
    does not exist at the revision). Repository, auth and manifest errors are
    not this, and must not skip as "not yet promoted"."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    return "EntryNotFoundError" in names and pointer_path in str(exc)


@pytest.mark.skipif(_hf_offline(), reason="Hugging Face offline mode is enabled")
@pytest.mark.parametrize(
    ("country", "year", "variant"),
    [
        ("us", 2024, DEFAULT_VARIANT),
        ("uk", 2023, DEFAULT_VARIANT),
        ("uk", 2025, DEFAULT_VARIANT),
    ],
)
def test_live_latest_pointer_resolves_to_a_coherent_certified_release(
    country: str,
    year: int,
    variant: str,
) -> None:
    """The real pointer and pinned manifest resolve without downloading large H5s.

    Deliberately release-agnostic: re-certification moves ``latest.json`` to a
    new release id without a commit here, so pinning which release is current
    (or its digest) would red main on every publish. Exact-metadata coverage
    lives in :func:`test_resolve_certified_release_returns_pointer_pinned_metadata`;
    this test only proves the currently published chain is coherent. Pinning a
    release *identity* belongs to policyengine.py's certification fixtures,
    which are updated atomically with each certification PR.
    """
    spec = resolve(country, year, variant=variant)
    # The UK 2025 default is a release whose builds publish into their own
    # folders, so releases.json, not a pointer, says which build is current.
    build_folder = spec.release is not None
    try:
        certified = loader._resolve_certified_release(spec)
    except Exception as exc:
        if _is_offline_error(exc):
            pytest.skip(f"Hugging Face metadata unavailable offline: {exc}")
        if isinstance(exc, (RepositoryNotFoundError, GatedRepoError)) or "401" in str(
            exc
        ):
            pytest.skip(
                f"repo requires credentials this environment lacks (private): {exc}"
            )
        if build_folder and _is_missing_pointer_error(exc, "releases.json"):
            # The one-time migration writes the repository's releases.json;
            # until then the release has no record to read. Anything else
            # once it exists (a manifest 404, a bad digest) fails.
            pytest.skip(f"releases.json not yet written in {spec.hf_repo}: {exc}")
        raise

    assert certified.artifact_repo_id == spec.hf_repo
    if build_folder:
        assert certified.release_id == spec.release
        assert certified.artifact_path == (
            f"releases/{spec.release}/{certified.build}/{spec.filename}"
        )
        assert certified.artifact_revision is None
    else:
        assert certified.artifact_path == spec.filename
        assert certified.release_id.startswith(
            # Live certified releases still carry the pre-rename id prefix; new
            # builds may adopt the microcosm- prefix at the next versioned
            # release.
            (f"microcosm-{country}-{spec.year}-", f"populace-{country}-{spec.year}-")
        )
        assert certified.artifact_revision == certified.release_id
    assert re.fullmatch(r"[0-9a-f]{64}", certified.artifact_sha256)
    assert certified.model.name == spec.engine_package


def test_the_national_release_refuses_a_local_area_manifest(tmp_path: Path):
    """releases.json gives each release one role: the national release
    cannot load a local-area manifest on the strength of its name."""
    manifest = _uk_build_folder_manifest(
        dataset_role="non_default_local_area", default_datasets={}
    )
    hub = _build_folder_hub(tmp_path, manifest=manifest)
    with pytest.raises(ValueError, match="listed with role 'national'"):
        loader._resolve_certified_release(resolve("uk", 2025), hub_download=hub)


def test_latest_pointer_refuses_any_artifact_revision_mismatch(tmp_path: Path):
    """One revision per pointer, by intent: the loader is stricter than the
    publisher contract's annual-revision allowance (review of #966)."""
    spec = resolve("uk", 2023)
    manifest = _uk_release_manifest(
        release_id=UK_2023_RELEASE_ID,
        revision=UK_2023_RELEASE_ID,
        filename="populace_uk_2023.h5",
        artifact_key="populace_uk_2023",
        dataset_role=None,
    )
    manifest["artifacts"]["calibration_report"] = {
        "kind": "diagnostics",
        "path": f"releases/{UK_2023_RELEASE_ID}/calibration_diagnostics.json",
        "repo_id": UK_REPO_ID,
        "revision": f"{UK_2023_RELEASE_ID}-annual-2027",
        "sha256": "0" * 64,
    }
    pointer = latest_pointer_payload(
        UK_2023_RELEASE_ID, updated_at="2026-06-19T03:00:00+00:00"
    )
    hub = _hub_with_pointer(
        tmp_path,
        repo_id=UK_REPO_ID,
        pointer_path=LATEST_POINTER_PATH,
        pointer=pointer,
        manifest=manifest,
    )
    with pytest.raises(ValueError, match="not pinned to expected revision"):
        loader._resolve_certified_release(spec, hub_download=hub)
