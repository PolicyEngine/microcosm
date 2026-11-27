"""Resolve, verify, download, and load a certified microcosm release.

Every registry entry names a release, and the loader reads it through the
repository's ``releases.json``: the record names the release's current build
(or ``build=`` pins a listed one), the loader reads
``releases/<release>/<build>/release_manifest.json``, downloads the dataset
that manifest names from the same folder, verifies its SHA-256, and enforces
the certified model/Core versions before constructing the engine dataset.
Build folders never change after publishing, so every file is read at
``main``.

A repository without ``releases.json`` is refused. The loader reads no
``latest*.json`` pointer file: each repository's switch to ``releases.json``
deleted them. A build published before its repository switched stays
readable by its path at its tag, outside the loader.
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
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from microcosm.data.contract import (
    DATASET_FILE_ROLE,
    NATIONAL_DEFAULT_DATASET_ROLE,
    NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
    RELEASE_MANIFEST_SCHEMA_VERSION,
    RELEASE_ROLES,
)
from microcosm.data.registry import DEFAULT_VARIANT, REGISTRY, DatasetSpec
from microcosm.data.release import missing_on_hub, never_cached_offline
from microcosm.data.releases_record import (
    RELEASES_RECORD_PATH,
    build_folder,
    parse_releases_record,
)

__all__ = [
    "CertifiedRelease",
    "available",
    "available_variants",
    "resolve",
    "latest_year",
    "download",
    "download_build_file",
    "download_certified",
    "load",
    "resolve_certified",
]

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_HubDownload = Callable[..., str]


@dataclass(frozen=True)
class _PackageCertification:
    name: str
    specifiers: tuple[str, ...]
    built_version: str


@dataclass(frozen=True)
class CertifiedRelease:
    """A release's certified dataset: where it is and what it was built with.

    ``build`` is the build that was loaded, ``artifact_path`` is inside its
    folder, and ``artifact_revision`` is ``None``: the folder never changes,
    so it is read at ``main``.
    """

    release_id: str
    artifact_repo_id: str
    artifact_path: str
    artifact_revision: str | None
    artifact_sha256: str
    model: _PackageCertification
    core: _PackageCertification
    build: str | None = None


_CertifiedRelease = CertifiedRelease


class _CertifiedPackageCompatibilityError(ValueError):
    """Installed package versions do not satisfy a release certificate."""


def available() -> list[tuple[str, int]]:
    """Every default-variant ``(country, year)``, sorted."""
    return sorted(
        {
            (country, year)
            for country, year, variant in REGISTRY
            if variant == DEFAULT_VARIANT
        }
    )


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

    # A build-folder manifest records what the build was built with at its
    # top level.
    built_package = _required_mapping(manifest, built_field, release_id=release_id)
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
    *,
    build: str,
    release_role: str,
) -> _CertifiedRelease:
    """Validate and select ``spec`` from one build's release manifest.

    The manifest is that build's, read from its folder in a release whose
    ``releases.json`` entry gives it ``release_role``.
    """
    schema_version = manifest.get("schema_version")
    if schema_version != RELEASE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Release {release_id!r} has release_manifest.json schema_version "
            f"{schema_version!r}; this microcosm-data reads version "
            f"{RELEASE_MANIFEST_SCHEMA_VERSION}. Upgrade microcosm-data."
        )
    folder = build_folder(release_id, build)
    named = (manifest.get("release"), manifest.get("build"))
    if named != (release_id, build):
        raise ValueError(
            f"{folder}/release_manifest.json names release {named[0]!r} and "
            f"build {named[1]!r}, not {release_id!r} and {build!r}."
        )

    dataset_role = manifest.get("dataset_role", NATIONAL_DEFAULT_DATASET_ROLE)
    if dataset_role not in {
        NATIONAL_DEFAULT_DATASET_ROLE,
        NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
    }:
        raise ValueError(
            f"Release {release_id!r} has unknown dataset_role {dataset_role!r}."
        )
    if release_role != RELEASE_ROLES[dataset_role]:
        # releases.json gives a release one role, so a manifest whose role
        # differs is not this release's, whatever its names say.
        raise ValueError(
            f"Release {release_id!r} is listed with role {release_role!r} in "
            f"releases.json, but build {build!r} declares dataset_role "
            f"{dataset_role!r}."
        )
    default_datasets = _required_mapping(
        manifest, "default_datasets", release_id=release_id
    )
    if dataset_role == NATIONAL_DEFAULT_DATASET_ROLE:
        artifact_key = default_datasets.get("national")
        if not isinstance(artifact_key, str) or not artifact_key:
            raise ValueError(
                f"Release {release_id!r} has no default_datasets.national artifact."
            )
        artifact_label = "default"
    else:
        if default_datasets != {}:
            raise ValueError(
                f"Release {release_id!r} declares dataset_role "
                f"{NON_DEFAULT_LOCAL_AREA_DATASET_ROLE!r} but default_datasets "
                "is not empty."
            )
        artifact_label = "local-area"

    artifacts = _required_mapping(manifest, "artifacts", release_id=release_id)
    # A build folder's files are found through releases.json; nothing is
    # tagged, so a manifest that pins a revision is not this layout's.
    pinned = sorted(
        str(key)
        for key, candidate in artifacts.items()
        if not isinstance(candidate, Mapping) or "revision" in candidate
    )
    if pinned:
        raise ValueError(
            f"Release {release_id!r} build {build!r} pins a revision on "
            f"artifact(s) {pinned}; a build folder's files carry none."
        )
    if dataset_role == NON_DEFAULT_LOCAL_AREA_DATASET_ROLE:
        microdata_keys = [
            key
            for key, candidate in artifacts.items()
            if isinstance(candidate, Mapping) and candidate.get("kind") == "microdata"
        ]
        if len(microdata_keys) != 1:
            raise ValueError(
                f"Release {release_id!r} declares dataset_role "
                f"{NON_DEFAULT_LOCAL_AREA_DATASET_ROLE!r} but has "
                f"{len(microdata_keys)} microdata artifacts; expected exactly one."
            )
        artifact_key = microdata_keys[0]

    artifact = artifacts.get(artifact_key)
    if not isinstance(artifact, Mapping):
        raise ValueError(
            f"Release {release_id!r} {artifact_label} dataset {artifact_key!r} is missing "
            "from release_manifest.json artifacts."
        )
    if artifact.get("kind") != "microdata":
        raise ValueError(
            f"Release {release_id!r} {artifact_label} artifact {artifact_key!r} has kind "
            f"{artifact.get('kind')!r}, not 'microdata'."
        )
    if artifact.get("role") != DATASET_FILE_ROLE:
        raise ValueError(
            f"Release {release_id!r} {artifact_label} artifact {artifact_key!r} has "
            f"role {artifact.get('role')!r}, not {DATASET_FILE_ROLE!r}."
        )

    artifact_repo_id = artifact.get("repo_id")
    artifact_path = artifact.get("path")
    artifact_sha256 = artifact.get("sha256")
    if artifact_repo_id != spec.hf_repo:
        raise ValueError(
            f"Release {release_id!r} {artifact_label} artifact repo "
            f"{artifact_repo_id!r} "
            f"does not match registry repo {spec.hf_repo!r}."
        )
    if artifact_path not in spec.dataset_filenames:
        expected = " or ".join(repr(name) for name in spec.dataset_filenames)
        raise ValueError(
            f"Release {release_id!r} {artifact_label} artifact path {artifact_path!r} "
            f"does not match registry filename {expected}."
        )
    if not isinstance(artifact_sha256, str) or not _SHA256_PATTERN.fullmatch(
        artifact_sha256
    ):
        raise ValueError(
            f"Release {release_id!r} {artifact_label} artifact has no valid sha256."
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
        artifact_path=f"{folder}/{artifact_path}",
        artifact_revision=None,
        artifact_sha256=artifact_sha256.lower(),
        model=model,
        core=core,
        build=build,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _releases_record_bytes(spec: DatasetSpec, *, hub_download: _HubDownload) -> bytes:
    """The repository's ``releases.json``, refused when it has none.

    Every release repository lists its builds there, so a repository without
    it is not one the loader reads. The Hub's answer that the file is absent
    (:func:`~microcosm.data.release.missing_on_hub`), or, in offline mode, a
    file that was never cached, is refused with that reason; any other
    failure propagates, a server error included.
    """
    try:
        record_path = hub_download(
            repo_id=spec.hf_repo, filename=RELEASES_RECORD_PATH, repo_type="dataset"
        )
    except Exception as exc:
        if never_cached_offline(exc):
            raise ValueError(
                f"{RELEASES_RECORD_PATH} of {spec.hf_repo} is not in the Hugging "
                "Face cache, and the Hub cannot be asked in offline mode; read "
                f"it once online to load {spec.release!r} offline."
            ) from exc
        if missing_on_hub(exc):
            raise ValueError(
                f"{spec.hf_repo} has no {RELEASES_RECORD_PATH}, so it lists no "
                f"build of {spec.release!r}; the loader reads builds only "
                f"through a repository's {RELEASES_RECORD_PATH}."
            ) from exc
        raise
    return Path(record_path).read_bytes()


def _resolve_build_folder_release(
    spec: DatasetSpec,
    *,
    record_bytes: bytes,
    build: str | None,
    hub_download: _HubDownload,
) -> _CertifiedRelease:
    """Select a build of ``spec.release`` from ``releases.json`` and check it.

    The current build, unless ``build`` pins a listed one; a pinned build
    that has been withdrawn loads with a warning, since a pin is a deliberate
    choice.
    """
    release = spec.release
    try:
        record = parse_releases_record(record_bytes)
    except ValueError as exc:
        raise ValueError(
            f"{RELEASES_RECORD_PATH} in {spec.hf_repo} is not a valid release "
            f"record: {exc}."
        ) from exc
    entry = record.releases.get(release)
    if entry is None:
        raise ValueError(
            f"{RELEASES_RECORD_PATH} in {spec.hf_repo} does not list release "
            f"{release!r}."
        )
    if build is None:
        build = entry.current
        if build is None:
            raise ValueError(
                f"Release {release!r} in {spec.hf_repo} has no current build; "
                "pass build= to load one of its listed builds."
            )
    elif build not in entry.builds:
        raise ValueError(
            f"Build {build!r} of release {release!r} is not listed in "
            f"{RELEASES_RECORD_PATH} in {spec.hf_repo}."
        )
    elif entry.builds[build].withdrawn:
        warnings.warn(
            f"Build {build!r} of release {release!r} has been withdrawn; it is "
            "loaded because build= pins it.",
            UserWarning,
            stacklevel=3,
        )
    manifest_path = Path(
        hub_download(
            repo_id=spec.hf_repo,
            filename=f"{build_folder(release, build)}/release_manifest.json",
            repo_type="dataset",
        )
    )
    manifest = _load_manifest(manifest_path, repo_id=spec.hf_repo, release_id=release)
    return _certified_release_from_manifest(
        spec,
        release,
        manifest,
        build=build,
        release_role=entry.role,
    )


def _resolve_certified_release(
    spec: DatasetSpec,
    *,
    hub_download: _HubDownload | None = None,
    build: str | None = None,
) -> _CertifiedRelease:
    """Resolve and validate the release's selected build and its manifest,
    read through the repository's ``releases.json``."""
    if hub_download is None:
        hub_download = _hub_download()
    return _resolve_build_folder_release(
        spec,
        record_bytes=_releases_record_bytes(spec, hub_download=hub_download),
        build=build,
        hub_download=hub_download,
    )


def _verified_download(
    spec: DatasetSpec,
    *,
    build: str | None = None,
    hub_download: _HubDownload | None = None,
) -> tuple[Path, _CertifiedRelease]:
    if hub_download is None:
        hub_download = _hub_download()
    certified = _resolve_certified_release(spec, hub_download=hub_download, build=build)
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


def download(
    country: str,
    year: int | None = None,
    *,
    variant: str = DEFAULT_VARIANT,
    build: str | None = None,
) -> Path:
    """Download and SHA-verify a certified release artifact.

    The repository's ``releases.json`` names the release's current build (or
    lists the ``build`` it pins), and its manifest and dataset come from that
    build's folder. Uses the Hugging Face cache, so repeated calls do not
    re-download unchanged files.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        ValueError: If the repository has no ``releases.json``, or release
            metadata or the artifact digest is invalid.
    """
    spec = resolve(country, year, variant=variant)
    path, _ = _verified_download(spec, build=build)
    return path


def resolve_certified(
    country: str,
    year: int | None = None,
    *,
    variant: str = DEFAULT_VARIANT,
    build: str | None = None,
    hub_download: _HubDownload | None = None,
) -> CertifiedRelease:
    """Say which build :func:`download_certified` would download, without it.

    Reads only the release record and the build's manifest, with the same
    checks; the dataset itself is neither downloaded nor verified.
    """
    spec = resolve(country, year, variant=variant)
    return _resolve_certified_release(spec, hub_download=hub_download, build=build)


def download_certified(
    country: str,
    year: int | None = None,
    *,
    variant: str = DEFAULT_VARIANT,
    build: str | None = None,
    hub_download: _HubDownload | None = None,
) -> tuple[Path, CertifiedRelease]:
    """Download and SHA-verify a certified dataset, and say which build it is.

    The same resolution and checks as :func:`download`, returning with the
    path the build's record: its release, build, artifact path and SHA-256,
    and the model and Core versions it was built with. Unlike :func:`load`,
    it neither enforces the installed engine's versions nor constructs a
    dataset, so a comparison can read a build under the engine it runs (a UK
    cut compares its candidate with the current build this way).

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        ValueError: If release metadata or the artifact digest is invalid.
    """
    spec = resolve(country, year, variant=variant)
    return _verified_download(spec, build=build, hub_download=hub_download)


def download_build_file(
    certified: CertifiedRelease,
    key: str,
    *,
    hub_download: _HubDownload | None = None,
) -> Path:
    """Download one file a build's release manifest declares, SHA-verified.

    ``key`` names an entry of the build's ``release_manifest.json``
    ``artifacts``; the file is read from the build's own folder, which never
    changes after publishing.

    Raises:
        ValueError: If ``certified`` names no build, the manifest declares no
            such artifact, or its digest differs.
    """
    if certified.build is None:
        raise ValueError(
            f"The record of release {certified.release_id!r} names no build; "
            "only a build in its own folder declares its files there."
        )
    if hub_download is None:
        hub_download = _hub_download()
    folder = build_folder(certified.release_id, certified.build)
    manifest = _load_manifest(
        Path(
            hub_download(
                repo_id=certified.artifact_repo_id,
                filename=f"{folder}/release_manifest.json",
                repo_type="dataset",
            )
        ),
        repo_id=certified.artifact_repo_id,
        release_id=certified.release_id,
    )
    artifacts = _required_mapping(
        manifest, "artifacts", release_id=certified.release_id
    )
    artifact = artifacts.get(key)
    if not isinstance(artifact, Mapping):
        raise ValueError(
            f"Build {certified.build!r} of release {certified.release_id!r} "
            f"declares no artifact {key!r}."
        )
    name = artifact.get("path")
    expected = artifact.get("sha256")
    if not isinstance(name, str) or not name or "/" in name:
        raise ValueError(
            f"Build {certified.build!r} artifact {key!r} has no bare file name."
        )
    if not isinstance(expected, str) or not _SHA256_PATTERN.fullmatch(expected):
        raise ValueError(
            f"Build {certified.build!r} artifact {key!r} has no valid sha256."
        )
    path = Path(
        hub_download(
            repo_id=certified.artifact_repo_id,
            filename=f"{folder}/{name}",
            repo_type="dataset",
        )
    )
    observed = _sha256(path)
    if observed != expected.lower():
        raise ValueError(
            f"sha256 verification failed for build {certified.build!r} artifact "
            f"{key!r}: expected {expected.lower()}; observed {observed}."
        )
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
    build: str | None = None,
):
    """Return a certified population as its engine dataset object.

    Args:
        country: Country slug (case-insensitive), e.g. ``"us"``.
        year: Population year; ``None`` loads the latest published year.
        variant: Dataset variant; ``"compact"`` is the default fast national
            microsimulation artifact.
        build: Load this listed build of the release instead of its current
            one.

    Raises:
        ImportError: If the country engine is not installed.
        ValueError: If release verification or engine compatibility fails.
    """
    spec = resolve(country, year, variant=variant)
    dataset_class = _dataset_class(spec)
    path, certified = _verified_download(spec, build=build)
    _enforce_package_compatibility(certified.model, release_id=certified.release_id)
    _enforce_package_compatibility(certified.core, release_id=certified.release_id)
    return dataset_class(file_path=str(path))
