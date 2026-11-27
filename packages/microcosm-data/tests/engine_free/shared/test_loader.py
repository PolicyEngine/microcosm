"""Tests split from packages/microcosm-data/tests/test_loader.py."""

from __future__ import annotations

import hashlib
import json
import os
import re
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

import microcosm.data.loader as loader
from microcosm.data import DEFAULT_VARIANT, download, latest_year, load, resolve
from microcosm.data.contract import (
    NATIONAL_DEFAULT_DATASET_ROLE,
    NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
)
from microcosm.data.registry import DatasetSpec
from test_support.microcosm_data.build_folders import (
    UK_LOCAL_BUILD,
    UK_NATIONAL_BUILD,
    build_folder_release_manifest,
)
from test_support.microcosm_data.hub_errors import hub_unreachable, missing_on_hub

#: A US build served by the hub fixtures. Any well-formed name works: nothing
#: in this suite may assert which build is current on the live Hub, because a
#: promotion changes releases.json without a commit here.
RELEASE_ID = "populace-us-2024-buildi-sparse-rmloss100-6e8e929-20260709T034135Z"
US_DATASET_BYTES = b"certified release artifact"
UK_REPO_ID = "policyengine/populace-uk-private"
UK_2023_RELEASE_ID = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
#: The UK switch moved the June 2023 build into this release, as its only and
#: current build, named by its old release ID.
UK_2023_RELEASE = "populace-uk-2023"
UK_NATIONAL_RELEASE_ID = "microcosm-uk-2024-25-national"
UK_LOCAL_RELEASE_ID = "microcosm-uk-2024-25-local-k15"
UK_DATASET_BYTES = b"certified UK line artifact"
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
    artifact_content: bytes = US_DATASET_BYTES,
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
    artifact_content: bytes = UK_DATASET_BYTES,
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
    artifact_content: bytes = UK_DATASET_BYTES,
    repo_id: str = UK_REPO_ID,
) -> FakeHubDownload:
    """Serve releases.json and one build folder, all read at ``main``."""
    hub = FakeHubDownload(tmp_path)
    record = _releases_record() if record is None else record
    hub._add(
        tmp_path,
        repo_id,
        "releases.json",
        None,
        record if isinstance(record, bytes) else json.dumps(record).encode(),
    )
    folder = f"releases/{manifest['release']}/{manifest['build']}"
    hub._add(
        tmp_path,
        repo_id,
        f"{folder}/release_manifest.json",
        None,
        json.dumps(manifest).encode(),
    )
    artifact = next(
        entry
        for entry in manifest["artifacts"].values()
        if entry.get("kind") == "microdata"
    )
    hub._add(tmp_path, repo_id, f"{folder}/{artifact['path']}", None, artifact_content)
    return hub


class FakeHubDownload:
    """Serve the files a test adds while recording every request."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.calls: list[tuple[str, str, str | None]] = []
        self.files: dict[tuple[str, str, str | None], Path] = {}

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
            raise missing_on_hub(
                repo_id,
                filename,
                revision,
                message=f"{repo_id}/{filename} at revision {revision!r}",
            ) from exc


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
    releases.json; the June 2023 release stays at its year, read through
    releases.json too."""
    assert latest_year("uk") == 2025
    assert resolve("uk").key == ("uk", 2025, DEFAULT_VARIANT)
    assert resolve("uk").release == UK_NATIONAL_RELEASE_ID
    assert resolve("uk", 2023).release == UK_2023_RELEASE
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


def test_download_verifies_the_build_folder_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = _build_folder_hub(tmp_path, manifest=_uk_build_folder_manifest())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)

    path = download("uk", 2025)

    folder = f"releases/{UK_NATIONAL_RELEASE_ID}/{UK_NATIONAL_BUILD}"
    assert path.read_bytes() == UK_DATASET_BYTES
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


def test_download_certified_returns_the_current_builds_record(
    tmp_path: Path,
) -> None:
    hub = _build_folder_hub(tmp_path, manifest=_uk_build_folder_manifest())

    # The fake build claims policyengine-uk ==2.89.2, which is not installed:
    # a comparison reads a build under its own engine, so nothing refuses.
    path, certified = loader.download_certified("uk", 2025, hub_download=hub)

    folder = f"releases/{UK_NATIONAL_RELEASE_ID}/{UK_NATIONAL_BUILD}"
    assert path.read_bytes() == UK_DATASET_BYTES
    assert certified.release_id == UK_NATIONAL_RELEASE_ID
    assert certified.build == UK_NATIONAL_BUILD
    assert certified.artifact_path == f"{folder}/microcosm_uk_2024_25.h5"
    assert certified.artifact_sha256 == hashlib.sha256(UK_DATASET_BYTES).hexdigest()
    assert certified.model.built_version == "2.89.2"


def _declare_build_file(
    hub: FakeHubDownload,
    tmp_path: Path,
    manifest: dict,
    *,
    key: str,
    name: str,
    content: bytes,
    declared: bytes | None = None,
) -> None:
    """Declare ``name`` in the build's manifest and serve ``content`` for it."""
    manifest["artifacts"][key] = {
        "kind": "diagnostics",
        "path": name,
        "repo_id": UK_REPO_ID,
        "sha256": hashlib.sha256(content if declared is None else declared).hexdigest(),
        "role": key,
    }
    folder = f"releases/{manifest['release']}/{manifest['build']}"
    hub._add(
        tmp_path,
        UK_REPO_ID,
        f"{folder}/release_manifest.json",
        None,
        json.dumps(manifest).encode(),
    )
    hub._add(tmp_path, UK_REPO_ID, f"{folder}/{name}", None, content)


def test_download_build_file_reads_a_declared_file_from_the_build_folder(
    tmp_path: Path,
) -> None:
    manifest = _uk_build_folder_manifest()
    hub = _build_folder_hub(tmp_path, manifest=manifest)
    _declare_build_file(
        hub,
        tmp_path,
        manifest,
        key="release_certification",
        name="release_certification.json",
        content=b'{"shippable": true}',
    )
    _, certified = loader.download_certified("uk", 2025, hub_download=hub)

    path = loader.download_build_file(
        certified, "release_certification", hub_download=hub
    )

    folder = f"releases/{UK_NATIONAL_RELEASE_ID}/{UK_NATIONAL_BUILD}"
    assert path.read_bytes() == b'{"shippable": true}'
    assert hub.calls[-1] == (UK_REPO_ID, f"{folder}/release_certification.json", None)
    with pytest.raises(ValueError, match="declares no artifact 'gate_summary'"):
        loader.download_build_file(certified, "gate_summary", hub_download=hub)


def test_download_build_file_refuses_a_file_whose_digest_differs(
    tmp_path: Path,
) -> None:
    manifest = _uk_build_folder_manifest()
    hub = _build_folder_hub(tmp_path, manifest=manifest)
    _declare_build_file(
        hub,
        tmp_path,
        manifest,
        key="release_certification",
        name="release_certification.json",
        content=b'{"shippable": false}',
        declared=b'{"shippable": true}',
    )
    _, certified = loader.download_certified("uk", 2025, hub_download=hub)

    with pytest.raises(ValueError, match="sha256 verification failed"):
        loader.download_build_file(certified, "release_certification", hub_download=hub)


def test_download_build_file_needs_a_build() -> None:
    without_build = loader.CertifiedRelease(
        release_id=UK_NATIONAL_RELEASE_ID,
        artifact_repo_id=UK_REPO_ID,
        artifact_path="microcosm_uk_2024_25.h5",
        artifact_revision=None,
        artifact_sha256="0" * 64,
        model=loader._PackageCertification("policyengine-uk", ("==2.89.2",), "2.89.2"),
        core=loader._PackageCertification("policyengine-core", ("==3.27.1",), "3.27.1"),
    )
    with pytest.raises(ValueError, match="names no build"):
        loader.download_build_file(without_build, "release_certification")


def test_uk_2023_loads_the_june_build_from_its_folder_after_the_switch(
    tmp_path: Path,
) -> None:
    """The UK switch moved the June 2023 build into release populace-uk-2023,
    as its current build under its old release ID."""
    spec = resolve("uk", 2023)
    assert spec.release == UK_2023_RELEASE
    manifest = _uk_build_folder_manifest(
        release=UK_2023_RELEASE,
        build=UK_2023_RELEASE_ID,
        filename="populace_uk_2023.h5",
        artifact_key="populace_uk_2023",
        dataset_role=None,
    )
    hub = _build_folder_hub(
        tmp_path,
        manifest=manifest,
        record=_releases_record(
            UK_2023_RELEASE,
            current=UK_2023_RELEASE_ID,
            builds={UK_2023_RELEASE_ID: False},
        ),
    )

    certified = loader._resolve_certified_release(spec, hub_download=hub)

    folder = f"releases/{UK_2023_RELEASE}/{UK_2023_RELEASE_ID}"
    assert certified.release_id == UK_2023_RELEASE
    assert certified.build == UK_2023_RELEASE_ID
    assert certified.artifact_path == f"{folder}/populace_uk_2023.h5"
    assert certified.artifact_revision is None
    assert hub.calls == [
        (UK_REPO_ID, "releases.json", None),
        (UK_REPO_ID, f"{folder}/release_manifest.json", None),
    ]


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
    if (
        isinstance(exc, (ConnectionError, TimeoutError))
        or closed_after_retry
        or any(cls.__name__ in offline_names for cls in type(exc).__mro__)
    ):
        return True
    # Offline, the loader refuses a releases.json never cached with a
    # ValueError whose cause is the Hub client's error; follow the causes,
    # each once.
    seen = {id(exc)}
    cause = exc.__cause__
    while isinstance(cause, Exception) and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, (ConnectionError, TimeoutError)) or any(
            cls.__name__ in offline_names for cls in type(cause).__mro__
        ):
            return True
        cause = cause.__cause__
    return False


def test_the_national_release_refuses_a_local_area_manifest(tmp_path: Path):
    """releases.json gives each release one role: the national release
    cannot load a local-area manifest on the strength of its name."""
    manifest = _uk_build_folder_manifest(
        dataset_role="non_default_local_area", default_datasets={}
    )
    hub = _build_folder_hub(tmp_path, manifest=manifest)
    with pytest.raises(ValueError, match="listed with role 'national'"):
        loader._resolve_certified_release(resolve("uk", 2025), hub_download=hub)


US_REPO_ID = "policyengine/populace-us"
US_NATIONAL_RELEASE_ID = "microcosm-us-2024-national"


def _us_build_folder_manifest(*, path: str = "microcosm_us_2024.h5") -> dict:
    """A US national build's release manifest in its own folder, declaring
    its dataset as ``path``."""
    manifest = _release_manifest()
    key = path.removesuffix(".h5")
    entry = manifest["artifacts"].pop("populace_us_2024")
    manifest["artifacts"][key] = {**entry, "path": path}
    manifest["default_datasets"] = {"national": key}
    return build_folder_release_manifest(
        manifest, release=US_NATIONAL_RELEASE_ID, build=RELEASE_ID
    )


def _us_build_folder_hub(tmp_path: Path, manifest: dict) -> FakeHubDownload:
    return _build_folder_hub(
        tmp_path,
        manifest=manifest,
        record=_releases_record(
            US_NATIONAL_RELEASE_ID, current=RELEASE_ID, builds={RELEASE_ID: False}
        ),
        artifact_content=US_DATASET_BYTES,
        repo_id=US_REPO_ID,
    )


def test_us_national_load_follows_the_current_build_in_releases_json(
    tmp_path: Path,
) -> None:
    hub = _us_build_folder_hub(tmp_path, _us_build_folder_manifest())

    certified = loader._resolve_certified_release(resolve("us", 2024), hub_download=hub)

    folder = f"releases/{US_NATIONAL_RELEASE_ID}/{RELEASE_ID}"
    assert certified.release_id == US_NATIONAL_RELEASE_ID
    assert certified.build == RELEASE_ID
    assert certified.artifact_path == f"{folder}/microcosm_us_2024.h5"
    assert hub.calls == [
        (US_REPO_ID, "releases.json", None),
        (US_REPO_ID, f"{folder}/release_manifest.json", None),
    ]


def test_a_us_build_moved_by_the_switch_loads_with_its_earlier_dataset_name(
    tmp_path: Path,
) -> None:
    hub = _us_build_folder_hub(
        tmp_path, _us_build_folder_manifest(path="populace_us_2024.h5")
    )

    certified = loader._resolve_certified_release(resolve("us", 2024), hub_download=hub)

    assert certified.artifact_path.endswith(f"/{RELEASE_ID}/populace_us_2024.h5")


def test_a_us_build_with_another_dataset_name_is_refused(tmp_path: Path) -> None:
    hub = _us_build_folder_hub(
        tmp_path,
        _us_build_folder_manifest(path="microcosm_us_2024_receipt_qualified.h5"),
    )

    with pytest.raises(ValueError, match="microcosm_us_2024.h5"):
        loader._resolve_certified_release(resolve("us", 2024), hub_download=hub)


def test_a_build_pin_is_refused_in_a_repository_without_releases_json(
    tmp_path: Path,
) -> None:
    hub = FakeHubDownload(tmp_path)

    with pytest.raises(ValueError, match="has no releases.json"):
        loader._resolve_certified_release(
            resolve("uk", 2025), hub_download=hub, build=UK_NATIONAL_BUILD
        )
    assert hub.calls == [(UK_REPO_ID, "releases.json", None)]


def test_an_unanswered_releases_json_request_propagates(tmp_path: Path) -> None:
    """An error other than the file's absence, such as an unreachable Hub,
    propagates; it is not read as a repository without releases.json."""
    hub = _build_folder_hub(tmp_path, manifest=_uk_build_folder_manifest())

    def unreachable(**request) -> str:
        if request["filename"] == "releases.json":
            raise ConnectionError("the Hub did not answer")
        return hub(**request)

    with pytest.raises(ConnectionError, match="did not answer"):
        loader._resolve_certified_release(resolve("uk", 2025), hub_download=unreachable)
    assert hub.calls == []


@pytest.mark.parametrize(
    "status_code", [None, 503, 403], ids=["no-answer", "server-error", "forbidden"]
)
def test_a_hub_failure_reading_releases_json_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status_code: int | None
) -> None:
    """Online, huggingface_hub raises LocalEntryNotFoundError when it could
    not ask the Hub, after a 5xx or on a 403. That is not the file's absence,
    so it propagates instead of being refused as a repository without
    releases.json."""
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    hub = _build_folder_hub(tmp_path, manifest=_uk_build_folder_manifest())

    def unreachable(**request) -> str:
        if request["filename"] == "releases.json":
            raise hub_unreachable(UK_REPO_ID, "releases.json", status_code=status_code)
        return hub(**request)

    with pytest.raises(LocalEntryNotFoundError):
        loader._resolve_certified_release(resolve("uk", 2025), hub_download=unreachable)
    assert hub.calls == []


def test_a_missing_release_manifest_raises(tmp_path: Path) -> None:
    hub = _build_folder_hub(tmp_path, manifest=_uk_build_folder_manifest())
    folder = f"releases/{UK_NATIONAL_RELEASE_ID}/{UK_NATIONAL_BUILD}"
    del hub.files[(UK_REPO_ID, f"{folder}/release_manifest.json", None)]

    with pytest.raises(FileNotFoundError, match=r"release_manifest\.json"):
        loader._resolve_certified_release(resolve("uk", 2025), hub_download=hub)


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
    hub = _us_build_folder_hub(tmp_path, _us_build_folder_manifest())
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
    hub = _us_build_folder_hub(tmp_path, _us_build_folder_manifest())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)
    constructed = _patch_engine(monkeypatch)

    dataset = load("us", 2024)

    folder = f"releases/{US_NATIONAL_RELEASE_ID}/{RELEASE_ID}"
    assert dataset.file_path == constructed[0]
    assert Path(dataset.file_path).read_bytes() == US_DATASET_BYTES
    assert hub.calls[-1] == (US_REPO_ID, f"{folder}/microcosm_us_2024.h5", None)


@pytest.mark.skipif(_hf_offline(), reason="Hugging Face offline mode is enabled")
@pytest.mark.parametrize(
    ("country", "year", "variant"),
    [
        ("us", 2024, DEFAULT_VARIANT),
        ("uk", 2023, DEFAULT_VARIANT),
        ("uk", 2025, DEFAULT_VARIANT),
    ],
)
def test_live_releases_json_resolves_to_a_coherent_certified_release(
    country: str,
    year: int,
    variant: str,
) -> None:
    """The real releases.json and the current build's manifest resolve
    without downloading large H5s.

    Deliberately build-agnostic: a promotion changes releases.json without a
    commit here, so pinning which build is current (or its digest) would turn
    main red on every promotion. Exact-metadata coverage lives in the fake-Hub
    tests above; this test only proves the published record and manifest are
    coherent. Pinning a build's identity belongs to policyengine.py's
    certification fixtures, which are updated with each certification PR.
    """
    spec = resolve(country, year, variant=variant)
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
        raise

    assert certified.artifact_repo_id == spec.hf_repo
    assert certified.release_id == spec.release
    assert certified.build is not None
    assert certified.artifact_path in {
        f"releases/{spec.release}/{certified.build}/{name}"
        for name in spec.dataset_filenames
    }
    assert certified.artifact_revision is None
    assert re.fullmatch(r"[0-9a-f]{64}", certified.artifact_sha256)
    assert certified.model.name == spec.engine_package
