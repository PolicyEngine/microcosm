"""Resolve, verify, download, and load a certified microcosm release.

The normal path follows the repository's release certificate: resolve
``latest.json``, read that release's manifest at its immutable release tag,
download the manifest-selected dataset at the same tag, verify its SHA-256,
and enforce the certified model/Core versions before constructing the engine
dataset. Mutable root artifacts are available only through the explicitly
unsafe ``load(..., unverified_root=True)`` escape hatch.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from microcosm.data.contract import RELEASE_MANIFEST_SCHEMA_VERSION
from microcosm.data.registry import DEFAULT_VARIANT, REGISTRY, DatasetSpec
from microcosm.data.release import latest_release

__all__ = [
    "available",
    "available_variants",
    "resolve",
    "latest_year",
    "download",
    "load",
]

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_HubDownload = Callable[..., str]


@dataclass(frozen=True)
class _PackageCertification:
    name: str
    specifiers: tuple[str, ...]
    built_version: str


@dataclass(frozen=True)
class _CertifiedRelease:
    release_id: str
    artifact_repo_id: str
    artifact_path: str
    artifact_revision: str
    artifact_sha256: str
    model: _PackageCertification
    core: _PackageCertification


class _CertifiedPackageCompatibilityError(ValueError):
    """Installed package versions do not satisfy a release certificate."""


def available() -> list[tuple[str, int]]:
    """Every published ``(country, year)``, sorted without variant duplicates."""
    return sorted({(country, year) for country, year, _ in REGISTRY})


def available_variants() -> list[tuple[str, int, str]]:
    """Every published ``(country, year, variant)``, sorted."""
    return sorted(REGISTRY)


def latest_year(country: str, *, variant: str = DEFAULT_VARIANT) -> int:
    """The most recent published year for ``country`` and ``variant``.

    Raises:
        ValueError: If no dataset is published for ``country``.
    """
    country = country.lower()
    variant = variant.lower()
    years = [yr for (c, yr, v) in REGISTRY if c == country and v == variant]
    if not years:
        published = sorted({(c, v) for c, _, v in REGISTRY})
        raise ValueError(
            f"No microcosm dataset for country {country!r} with variant "
            f"{variant!r}; published country variants: {published}."
        )
    return max(years)


def resolve(
    country: str,
    year: int | None = None,
    *,
    variant: str = DEFAULT_VARIANT,
) -> DatasetSpec:
    """Look up the :class:`DatasetSpec` for ``country``, ``year``, and variant.

    Args:
        country: Country slug (case-insensitive), e.g. ``"us"``.
        year: Population year; ``None`` selects the latest published year for
            the country.
        variant: Dataset variant; ``"compact"`` is the default fast national
            microsimulation artifact.

    Raises:
        ValueError: If the country (or the country/year pair) is not published,
            naming what *is* available.
    """
    country = country.lower()
    variant = variant.lower()
    if year is None:
        year = latest_year(country, variant=variant)
    spec = REGISTRY.get((country, int(year), variant))
    if spec is None:
        years = sorted(yr for c, yr, v in REGISTRY if c == country and v == variant)
        if years:
            detail = f"published years for {country!r} variant {variant!r}: {years}"
        else:
            published = sorted({(c, v) for c, _, v in REGISTRY})
            detail = (
                f"no datasets for {country!r} variant {variant!r}; "
                f"published country variants: {published}"
            )
        raise ValueError(
            f"No microcosm dataset for ({country!r}, {year}, {variant!r}); {detail}."
        )
    return spec


def _hub_download() -> _HubDownload:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError(
            "microcosm-data needs huggingface_hub to download artifacts; "
            "reinstall microcosm-data with its dependencies."
        ) from exc
    return hf_hub_download


def _load_manifest(path: Path, *, repo_id: str, release_id: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read release manifest for {release_id!r} in {repo_id}: {exc}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Release manifest for {release_id!r} in {repo_id} must be a JSON object."
        )
    return payload


def _required_mapping(
    owner: Mapping[str, Any], field: str, *, release_id: str
) -> Mapping[str, Any]:
    value = owner.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Release {release_id!r} has no valid {field!r} object in "
            "release_manifest.json."
        )
    return value


def _package_certification(
    manifest: Mapping[str, Any],
    *,
    field: str,
    package_name: str,
    built_field: str,
    release_id: str,
) -> _PackageCertification:
    entries = manifest.get(field)
    if not isinstance(entries, list):
        raise ValueError(f"Release {release_id!r} has no {field!r} certification list.")
    specifiers: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("name") != package_name:
            continue
        specifier = entry.get("specifier")
        if not isinstance(specifier, str) or not specifier.strip():
            continue
        try:
            SpecifierSet(specifier)
        except InvalidSpecifier as exc:
            raise ValueError(
                f"Release {release_id!r} has invalid {package_name} compatibility "
                f"specifier {specifier!r}."
            ) from exc
        specifiers.append(specifier)
    if not specifiers:
        raise ValueError(
            f"Release {release_id!r} does not certify compatibility with "
            f"{package_name}."
        )

    build = _required_mapping(manifest, "build", release_id=release_id)
    built_package = _required_mapping(build, built_field, release_id=release_id)
    if built_package.get("name") != package_name:
        raise ValueError(
            f"Release {release_id!r} build provenance does not name {package_name}."
        )
    built_version = built_package.get("version")
    if not isinstance(built_version, str):
        raise ValueError(f"Release {release_id!r} has no built {package_name} version.")
    try:
        parsed_built_version = Version(built_version)
    except InvalidVersion as exc:
        raise ValueError(
            f"Release {release_id!r} has invalid built {package_name} version "
            f"{built_version!r}."
        ) from exc
    if not any(parsed_built_version in SpecifierSet(item) for item in specifiers):
        raise ValueError(
            f"Release {release_id!r} certification for {package_name} does not "
            f"include its built version {built_version}."
        )
    return _PackageCertification(package_name, tuple(specifiers), built_version)


def _certified_release_from_manifest(
    spec: DatasetSpec,
    release_id: str,
    manifest: Mapping[str, Any],
) -> _CertifiedRelease:
    """Validate and select ``spec`` from one release manifest."""
    schema_version = manifest.get("schema_version")
    if schema_version != RELEASE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Release {release_id!r} has release_manifest.json schema_version "
            f"{schema_version!r}; this microcosm-data reads version "
            f"{RELEASE_MANIFEST_SCHEMA_VERSION}. Upgrade microcosm-data."
        )
    build = _required_mapping(manifest, "build", release_id=release_id)
    if build.get("build_id") != release_id:
        raise ValueError(
            f"latest.json points to release {release_id!r}, but its release manifest "
            f"declares build_id {build.get('build_id')!r}."
        )

    default_datasets = _required_mapping(
        manifest, "default_datasets", release_id=release_id
    )
    artifact_key = default_datasets.get("national")
    if not isinstance(artifact_key, str) or not artifact_key:
        raise ValueError(
            f"Release {release_id!r} has no default_datasets.national artifact."
        )
    artifacts = _required_mapping(manifest, "artifacts", release_id=release_id)
    artifact = artifacts.get(artifact_key)
    if not isinstance(artifact, Mapping):
        raise ValueError(
            f"Release {release_id!r} default dataset {artifact_key!r} is missing "
            "from release_manifest.json artifacts."
        )
    if artifact.get("kind") != "microdata":
        raise ValueError(
            f"Release {release_id!r} default artifact {artifact_key!r} has kind "
            f"{artifact.get('kind')!r}, not 'microdata'."
        )

    artifact_repo_id = artifact.get("repo_id")
    artifact_path = artifact.get("path")
    artifact_revision = artifact.get("revision")
    artifact_sha256 = artifact.get("sha256")
    if artifact_repo_id != spec.hf_repo:
        raise ValueError(
            f"Release {release_id!r} default artifact repo {artifact_repo_id!r} "
            f"does not match registry repo {spec.hf_repo!r}."
        )
    if artifact_path != spec.filename:
        raise ValueError(
            f"Release {release_id!r} default artifact path {artifact_path!r} "
            f"does not match registry filename {spec.filename!r}."
        )
    if artifact_revision != release_id:
        raise ValueError(
            f"Release {release_id!r} default artifact revision "
            f"{artifact_revision!r} is not pinned to the release id."
        )
    if not isinstance(artifact_sha256, str) or not _SHA256_PATTERN.fullmatch(
        artifact_sha256
    ):
        raise ValueError(
            f"Release {release_id!r} default artifact has no valid sha256."
        )

    model = _package_certification(
        manifest,
        field="compatible_model_packages",
        package_name=spec.engine_package,
        built_field="built_with_model_package",
        release_id=release_id,
    )
    core = _package_certification(
        manifest,
        field="compatible_core_packages",
        package_name="policyengine-core",
        built_field="built_with_core_package",
        release_id=release_id,
    )
    return _CertifiedRelease(
        release_id=release_id,
        artifact_repo_id=artifact_repo_id,
        artifact_path=artifact_path,
        artifact_revision=artifact_revision,
        artifact_sha256=artifact_sha256.lower(),
        model=model,
        core=core,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_certified_release(
    spec: DatasetSpec, *, hub_download: _HubDownload | None = None
) -> _CertifiedRelease:
    """Resolve and validate the release pointer and its pinned manifest."""
    if hub_download is None:
        hub_download = _hub_download()
    pointer = latest_release(
        spec.hf_repo,
        api=SimpleNamespace(hf_hub_download=hub_download),
    )
    manifest_path = Path(
        hub_download(
            repo_id=spec.hf_repo,
            filename=pointer.paths["release_manifest"],
            repo_type="dataset",
            revision=pointer.release_id,
        )
    )
    manifest = _load_manifest(
        manifest_path,
        repo_id=spec.hf_repo,
        release_id=pointer.release_id,
    )
    return _certified_release_from_manifest(spec, pointer.release_id, manifest)


def _verified_download(spec: DatasetSpec) -> tuple[Path, _CertifiedRelease]:
    hub_download = _hub_download()
    certified = _resolve_certified_release(spec, hub_download=hub_download)
    artifact_path = Path(
        hub_download(
            repo_id=certified.artifact_repo_id,
            filename=certified.artifact_path,
            repo_type="dataset",
            revision=certified.artifact_revision,
        )
    )
    observed_sha256 = _sha256(artifact_path)
    if observed_sha256 != certified.artifact_sha256:
        raise ValueError(
            f"sha256 verification failed for release {certified.release_id!r} "
            f"artifact {certified.artifact_path!r}: expected "
            f"{certified.artifact_sha256}; observed {observed_sha256}."
        )
    return artifact_path, certified


def _unverified_root_download(spec: DatasetSpec) -> Path:
    return Path(
        _hub_download()(
            repo_id=spec.hf_repo,
            filename=spec.filename,
            repo_type="dataset",
        )
    )


def download(
    country: str,
    year: int | None = None,
    *,
    variant: str = DEFAULT_VARIANT,
) -> Path:
    """Download and SHA-verify the latest certified release artifact.

    The mutable ``latest.json`` pointer selects a release; both its manifest
    and dataset artifact are then fetched at ``revision=<release_id>``. Uses
    the Hugging Face cache, so repeated calls do not re-download unchanged
    files.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        ValueError: If release metadata or the artifact digest is invalid.
    """
    spec = resolve(country, year, variant=variant)
    path, _ = _verified_download(spec)
    return path


def _dataset_class(spec: DatasetSpec):
    try:
        module = importlib.import_module(spec.engine_module)
    except ImportError as exc:
        raise ImportError(
            f"microcosm-data needs {spec.engine_package} to load the {spec.country!r} "
            f"population; install it with: pip install "
            f"'microcosm-data[{spec.country}]'."
        ) from exc
    return getattr(module, spec.engine_class)


def _enforce_package_compatibility(
    certification: _PackageCertification, *, release_id: str
) -> None:
    try:
        installed_text = metadata.version(certification.name)
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            f"Cannot verify certified release {release_id!r}: "
            f"{certification.name} is not installed. Install it with: "
            f"pip install '{certification.name}=={certification.built_version}'."
        ) from exc
    try:
        installed = Version(installed_text)
    except InvalidVersion as exc:
        raise ValueError(
            f"Installed {certification.name} version {installed_text!r} is not a "
            "valid PEP 440 version."
        ) from exc
    if any(installed in SpecifierSet(item) for item in certification.specifiers):
        return
    required = " or ".join(certification.specifiers)
    raise _CertifiedPackageCompatibilityError(
        f"Certified release {release_id!r} is incompatible with installed "
        f"{certification.name} {installed_text}; release certification requires "
        f"{required}. Install the certified version with: pip install "
        f"'{certification.name}=={certification.built_version}'."
    )


def load(
    country: str,
    year: int | None = None,
    *,
    variant: str = DEFAULT_VARIANT,
    unverified_root: bool = False,
):
    """Return a certified population as its engine dataset object.

    Args:
        country: Country slug (case-insensitive), e.g. ``"us"``.
        year: Population year; ``None`` loads the latest published year.
        variant: Dataset variant; ``"compact"`` is the default fast national
            microsimulation artifact.
        unverified_root: Load the mutable root convenience artifact while
            bypassing release metadata, SHA-256, and engine certification.
            This unsafe compatibility escape hatch emits a loud warning.

    Raises:
        ImportError: If the country engine is not installed.
        ValueError: If release verification or engine compatibility fails.
    """
    spec = resolve(country, year, variant=variant)
    dataset_class = _dataset_class(spec)
    if unverified_root:
        warnings.warn(
            "UNVERIFIED mutable-root loading bypasses latest.json, the release "
            "manifest, artifact SHA-256, and model/Core compatibility checks. "
            "Do not use this path for production or reproducible analysis.",
            RuntimeWarning,
            stacklevel=2,
        )
        path = _unverified_root_download(spec)
    else:
        path, certified = _verified_download(spec)
        _enforce_package_compatibility(certified.model, release_id=certified.release_id)
        _enforce_package_compatibility(certified.core, release_id=certified.release_id)
    return dataset_class(file_path=str(path))
