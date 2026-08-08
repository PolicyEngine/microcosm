"""Release-native loader resolution, verification, and compatibility checks."""

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
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

import microcosm.data.loader as loader
from microcosm.data import DEFAULT_VARIANT, download, latest_year, load, resolve
from microcosm.data.release import LATEST_POINTER_PATH, latest_pointer_payload

#: Release id served by the hub fixture. Any well-formed id works: nothing in
#: this suite may assert which release the live Hub currently points at,
#: because re-certification moves ``latest.json`` without a commit here.
RELEASE_ID = "populace-us-2024-buildi-sparse-rmloss100-6e8e929-20260709T034135Z"
TAGGED_ARTIFACT = b"certified release artifact"
MUTABLE_ROOT_ARTIFACT = b"uncertified in-flight root artifact"


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
        ValueError,
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


def _engine_available() -> bool:
    try:
        import policyengine_us  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _engine_available(), reason="policyengine-us not installed")
def test_live_load_builds_a_simulation_with_a_sane_population() -> None:
    """The certified dataset loads into PolicyEngine-US with a sane population."""
    from policyengine_us import Microsimulation

    try:
        dataset = load("us", 2024)
    except Exception as exc:  # live network/release availability
        if _is_offline_error(exc):
            pytest.skip(f"Hugging Face dataset unavailable offline: {exc}")
        raise

    sim = Microsimulation(dataset=dataset)
    population = (sim.calculate("age", 2024) >= 0).sum()
    assert 3.0e8 < population < 3.6e8


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


@pytest.mark.skipif(_hf_offline(), reason="Hugging Face offline mode is enabled")
@pytest.mark.parametrize("country", ["us", "uk"])
def test_live_latest_pointer_resolves_to_a_coherent_certified_release(
    country: str,
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
    spec = resolve(country)
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

    assert certified.release_id.startswith(
        # Live certified releases still carry the pre-rename id prefix; new
        # builds may adopt the microcosm- prefix at the next versioned release.
        (f"microcosm-{country}-{spec.year}-", f"populace-{country}-{spec.year}-")
    )
    assert certified.artifact_repo_id == spec.hf_repo
    assert certified.artifact_path == spec.filename
    assert certified.artifact_revision == certified.release_id
    assert re.fullmatch(r"[0-9a-f]{64}", certified.artifact_sha256)
    assert certified.model.name == spec.engine_package
