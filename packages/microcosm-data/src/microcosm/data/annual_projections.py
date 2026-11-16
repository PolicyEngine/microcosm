"""Additional publication checks for explicit annual US dataset families.

The parent release still has to pass its ordinary contract. These checks bind
annual acceptance evidence to the exact files, years, and source population;
they do not turn a candidate build receipt into a calibration certificate.

A family is published in one of two layouts. In the tagged layout its files
are added to the base release's own manifest: projected H5s at the
repository's root, the candidate manifest, the acceptance report and the
per-year receipts under ``releases/<base release>/``, and every artifact pins
one tag ``<base release>-annual-<YYYYMMDDTHHMMSSZ>-<hex8>``. In the
build-folder layout (microcosm#1570) it is a build of its own release,
:data:`~microcosm.data.contract.US_ANNUAL_RELEASE_ID`. Its folder carries the
base national build's files, with the base-year H5 under that build's key and
file name, one H5 per projected year, the candidate manifest, the acceptance
report and every receipt, all as bare file names that pin no revision. Its
build ID is the base build's ID followed by
``-annual-<YYYYMMDDTHHMMSSZ>-<hex8>``, and its release manifest names the base
build in ``metadata.annual_projection_base`` as ``repo_id``, ``release``,
``build``, ``path`` and ``sha256``. In that layout only builds of the annual
release declare ``metadata.dataset_years``, and each of them must. Before an
annual build is handed to the emitter, :func:`require_base_publication` checks
on the Hub that its base build is published with the same base-year H5.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
from packaging.specifiers import SpecifierSet

from microcosm.data.contract import (
    CHECKSUMS_FILE,
    US_ANNUAL_RELEASE_ID,
    US_NATIONAL_RELEASE_ID,
    publishes_to_build_folder,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_YEAR = re.compile(r"[1-9][0-9]{3}\Z")
_ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
_CHECKS = frozenset(
    {"schema", "source_identity", "year", "demographics", "input_aggregates", "runtime"}
)
#: An annual build's ID in the build-folder layout: its base national build's
#: ID, the time it was assembled in UTC, and eight lowercase hex digits.
_ANNUAL_BUILD_ID = re.compile(r"(?P<base>.+)-annual-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}\Z")
#: The fields of ``metadata.annual_projection_base``: the base national build
#: in the build-folder form, and its dataset's file name and SHA-256.
_BASE_REFERENCE_FIELDS = frozenset({"repo_id", "release", "build", "path", "sha256"})
#: The files the emitter publishes for every build itself, which an annual
#: build writes anew instead of carrying from its base build.
_PUBLISHER_FILES = frozenset(
    {"release_manifest.json", "build_manifest.json", CHECKSUMS_FILE}
)


@dataclass(frozen=True)
class AnnualProjectionExtension:
    """Validated publication additions; inherited artifacts keep their contract.

    ``revision`` is the annual tag every artifact pins in the tagged layout,
    and ``None`` for a build in its build folder, whose files pin none.
    """

    revision: str | None
    additional_artifacts: Mapping[str, Path]
    projected_artifacts: frozenset[str]


def annual_release_refusal(release: str, manifest: Mapping) -> str | None:
    """Why a build of ``release`` may not declare, or lack, annual years.

    In the build-folder layout only builds of
    :data:`~microcosm.data.contract.US_ANNUAL_RELEASE_ID` declare
    ``metadata.dataset_years``, and each of them must. ``None`` when the
    build is allowed as it is, which includes every tagged release.
    """
    if not publishes_to_build_folder(release):
        return None
    metadata = manifest.get("metadata")
    declares = isinstance(metadata, Mapping) and "dataset_years" in metadata
    if release == US_ANNUAL_RELEASE_ID and not declares:
        return (
            f"a build of {US_ANNUAL_RELEASE_ID!r} must declare "
            "metadata.dataset_years, the years its H5 files cover"
        )
    if release != US_ANNUAL_RELEASE_ID and declares:
        return (
            f"only builds of {US_ANNUAL_RELEASE_ID!r} declare "
            "metadata.dataset_years in the build-folder layout; a build of "
            f"{release!r} cannot"
        )
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: object, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _read_json(path: Path) -> Mapping:
    return _object(json.loads(path.read_text()), path.name)


def _artifact(
    key: str,
    artifacts: Mapping,
    release_dir: Path,
    artifact_root: Path | None,
    *,
    build_folder: bool = False,
) -> tuple[Path, str]:
    """The local file an annual artifact names, and its checked SHA-256.

    In a build folder the path is a bare file name, read from the release
    directory or else from ``artifact_root``, as the publisher reads it, and
    the artifact pins no revision. In the tagged layout it pins a revision,
    and a ``releases/<release>/`` path is a release file.
    """
    record = _object(artifacts.get(key), f"artifact {key!r}")
    relative = record.get("path")
    sha = record.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or PurePosixPath(relative).is_absolute()
        or PurePosixPath(relative).as_posix() != relative
        or any(part in {".", ".."} for part in relative.split("/"))
    ):
        raise ValueError(f"annual artifact {key!r} must have a clean relative path")
    if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
        raise ValueError(f"annual artifact {key!r} requires sha256")
    if build_folder:
        if "/" in relative:
            raise ValueError(
                f"annual artifact {key!r} path {relative!r} must be a bare file "
                "name in the build's folder"
            )
        if "revision" in record:
            raise ValueError(
                f"annual artifact {key!r} pins a revision; a build folder's "
                "files have no tag"
            )
        local = release_dir / relative
        if not local.is_file():
            if artifact_root is None:
                raise ValueError(
                    f"annual artifact {key!r} is not in the release directory "
                    "and requires artifact_root"
                )
            local = artifact_root / relative
        if not local.is_file() or _sha256(local) != sha:
            raise ValueError(f"annual artifact {key!r} bytes do not match its sha256")
        return local, sha
    if not isinstance(record.get("revision"), str) or not record["revision"]:
        raise ValueError(f"annual artifact {key!r} requires a revision pin")
    prefix = f"releases/{release_dir.name}/"
    if relative.startswith(prefix):
        filename = relative.removeprefix(prefix)
        if not filename or "/" in filename:
            raise ValueError("annual release-local artifacts must be bare filenames")
        local = release_dir / filename
    elif relative.startswith("releases/"):
        raise ValueError("annual artifacts cannot use another release's prefix")
    elif (release_dir / relative).is_file() or (release_dir / relative).is_symlink():
        raise ValueError(
            "annual release-local artifacts require their exact release prefix"
        )
    elif artifact_root is not None:
        local = artifact_root / relative
    else:
        raise ValueError(f"annual artifact {key!r} requires artifact_root")
    if not local.is_file() or _sha256(local) != sha:
        raise ValueError(f"annual artifact {key!r} bytes do not match its sha256")
    return local, sha


def _check_base_reference(
    manifest: Mapping, base: Mapping, base_artifact: Mapping, base_sha: str
) -> None:
    """The base national build an annual build folder names, and its H5.

    ``metadata.annual_projection_base`` names a build of the national release
    in the build-folder form, in the repository of the base-year H5. The
    annual build's ID is that build's ID followed by
    ``-annual-<YYYYMMDDTHHMMSSZ>-<hex8>``, the candidate's
    ``base.parent_release`` is that build, and the reference's ``path`` and
    ``sha256`` are the base-year H5's file name and SHA-256 (which the
    candidate's ``base.sha256`` has already matched).
    """
    from microcosm.data.build_folder_manifests import referenced_build

    reference = _object(manifest.get("metadata"), "release metadata").get(
        "annual_projection_base"
    )
    if not isinstance(reference, Mapping):
        raise ValueError(
            "an annual build names its base national build in "
            "metadata.annual_projection_base"
        )
    try:
        if set(reference) != _BASE_REFERENCE_FIELDS:
            raise ValueError(f"it has the fields {sorted(reference)}")
        referenced_build(reference)
    except ValueError as exc:
        raise ValueError(
            "metadata.annual_projection_base must name the base build in the "
            f"build-folder form ({', '.join(sorted(_BASE_REFERENCE_FIELDS))}, "
            f"no revision): {exc}"
        ) from exc
    base_build = reference["build"]
    if reference["release"] != US_NATIONAL_RELEASE_ID:
        raise ValueError(
            f"metadata.annual_projection_base names release "
            f"{reference['release']!r}; an annual build projects a build of "
            f"{US_NATIONAL_RELEASE_ID!r}"
        )
    if reference["repo_id"] != base_artifact.get("repo_id"):
        raise ValueError(
            f"metadata.annual_projection_base names repo_id "
            f"{reference['repo_id']!r}, but the base-year H5 is in "
            f"{base_artifact.get('repo_id')!r}"
        )
    build = manifest.get("build")
    match = _ANNUAL_BUILD_ID.fullmatch(build) if isinstance(build, str) else None
    if match is None:
        raise ValueError(
            "an annual build's ID is its base build's ID followed by "
            f"-annual-<YYYYMMDDTHHMMSSZ>-<hex8>; got {build!r}"
        )
    if match["base"] != base_build:
        raise ValueError(
            f"annual build {build!r} does not start with its base build "
            f"{base_build!r}, which metadata.annual_projection_base names"
        )
    if base.get("parent_release") != base_build:
        raise ValueError(
            f"the projection candidate's base.parent_release is "
            f"{base.get('parent_release')!r}, not the base build {base_build!r}"
        )
    if reference["path"] != base_artifact.get("path"):
        raise ValueError(
            f"metadata.annual_projection_base path {reference['path']!r} is not "
            f"the base-year H5's file name {base_artifact.get('path')!r}"
        )
    if reference["sha256"] != base_sha:
        raise ValueError(
            "metadata.annual_projection_base sha256 differs from the base-year "
            "H5's sha256"
        )


def _native_year(h5: h5py.File) -> int:
    if "_time_period/table" not in h5:
        raise ValueError("annual H5 requires explicit _time_period metadata")
    stored = h5["_time_period/table"]
    if (
        not isinstance(stored, h5py.Dataset)
        or stored.shape != (1,)
        or "values" not in (stored.dtype.names or ())
        or stored.dtype["values"].kind not in {"i", "u"}
    ):
        raise ValueError("annual H5 _time_period must contain one integer year")
    return int(stored["values"][0])


def _check_native_identity(
    base: Path, projected: Path, year: int, record: Mapping
) -> None:
    with (
        h5py.File(base, mode="r") as original,
        h5py.File(projected, mode="r") as annual,
    ):
        if _native_year(annual) != year:
            raise ValueError(f"annual H5 stored year does not match {year}")
        rows = _object(record.get("rows"), f"{year} row counts")
        for entity in _ENTITIES:
            path = f"{entity}/table"
            if path not in original or path not in annual:
                raise ValueError(f"annual H5 lacks native {entity} table")
            source, target = original[path], annual[path]
            if not isinstance(source, h5py.Dataset) or not isinstance(
                target, h5py.Dataset
            ):
                raise ValueError(f"{entity} table must be a dataset")
            if source.shape != target.shape or rows.get(entity) != len(target):
                raise ValueError(f"{year} {entity} row count differs from its source")
            columns = source.dtype.names
            if not columns or columns != target.dtype.names:
                raise ValueError(
                    f"{year} {entity} native columns differ from its source"
                )
            identity_columns = [
                column
                for column in columns
                if column == f"{entity}_id"
                or (
                    entity == "person"
                    and column.startswith("person_")
                    and column.endswith("_id")
                )
            ]
            if f"{entity}_id" not in identity_columns:
                raise ValueError(f"{year} {entity} lacks its native identity column")
            for column in identity_columns:
                if not np.array_equal(source[column], target[column]):
                    raise ValueError(f"{year} {column} differs from its source")
        weights = annual["household/table"]["household_weight"]
        if (
            not np.isfinite(weights).all()
            or (weights < 0).any()
            or not (weights > 0).any()
        ):
            raise ValueError(
                f"{year} household weights must be finite nonnegative with positive mass"
            )


def validate_annual_projection_extension(
    release_dir: Path | str,
    manifest: Mapping,
    *,
    artifact_root: Path | str | None = None,
) -> AnnualProjectionExtension | None:
    """Check optional annual metadata and every referenced artifact locally.

    Releases without ``metadata.dataset_years`` retain their existing contract,
    except that a build of the annual release must declare it and, in the
    build-folder layout, no other release's build may
    (:func:`annual_release_refusal`). A present annual map requires separate
    passing acceptance evidence. The publisher calls this in addition to the
    base release's validation. A directory named after a release that
    publishes into build folders is checked in that layout; any other in the
    tagged layout.
    """
    release_dir = Path(release_dir)
    refusal = annual_release_refusal(release_dir.name, manifest)
    if refusal is not None:
        raise ValueError(refusal)
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping) or "dataset_years" not in metadata:
        return None
    build_folder = publishes_to_build_folder(release_dir.name)
    root = Path(artifact_root) if artifact_root is not None else None
    families = _object(metadata["dataset_years"], "metadata.dataset_years")
    if not families:
        raise ValueError("metadata.dataset_years must not be empty")
    artifacts = _object(manifest.get("artifacts"), "release artifacts")

    def artifact(key: str) -> tuple[Path, str]:
        return _artifact(key, artifacts, release_dir, root, build_folder=build_folder)

    evidence_key = metadata.get("annual_projection_manifest")
    acceptance_key = metadata.get("annual_projection_acceptance")
    if not isinstance(evidence_key, str) or not isinstance(acceptance_key, str):
        raise ValueError(
            "annual releases require projection manifest and acceptance artifact keys"
        )
    evidence_path, evidence_sha = artifact(evidence_key)
    acceptance_path, _ = artifact(acceptance_key)
    # A tagged release's receipts are its release files; a build folder's
    # files are all bare names in the folder.
    prefix = "" if build_folder else f"releases/{release_dir.name}/"
    for key in (evidence_key, acceptance_key):
        if not artifacts[key]["path"].startswith(prefix):
            raise ValueError("annual receipts require their exact release prefix")
    evidence, acceptance = _read_json(evidence_path), _read_json(acceptance_path)
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 1
        or evidence.get("kind") != "us_annual_static_aging_candidate"
        or evidence.get("status") != "complete"
    ):
        raise ValueError(
            "annual projection manifest must describe a complete schema-1 candidate"
        )
    if (
        type(acceptance.get("schema_version")) is not int
        or acceptance.get("schema_version") != 1
        or acceptance.get("kind") != "us_annual_projection_acceptance"
        or acceptance.get("status") != "passed"
        or acceptance.get("projection_manifest_sha256") != evidence_sha
    ):
        raise ValueError(
            "annual acceptance must pass and bind the exact projection manifest"
        )
    if (
        _object(evidence.get("metadata"), "projection metadata").get("dataset_years")
        != families
    ):
        raise ValueError("annual release coverage differs from its projection evidence")
    base = _object(evidence.get("base"), "projection base")
    base_key = base.get("dataset")
    base_year = base.get("year")
    if (
        not isinstance(base_key, str)
        or type(base_year) is not int
        or not isinstance(base.get("parent_release"), str)
        or not base["parent_release"]
    ):
        raise ValueError(
            "projection base requires dataset, integer year and parent release"
        )
    base_path, base_sha = artifact(base_key)
    if base_sha != base.get("sha256"):
        raise ValueError(
            "projection base hash differs from the release's base artifact"
        )
    defaults = _object(manifest.get("default_datasets"), "release defaults")
    if build_folder:
        # The build's details are at the manifest's top level, and its base
        # is another build, which metadata.annual_projection_base names.
        build = manifest
        if defaults.get("national") != base_key:
            raise ValueError(
                "annual projections must augment this release's certified national base"
            )
        _check_base_reference(manifest, base, artifacts[base_key], base_sha)
    else:
        build = _object(manifest.get("build"), "release build")
        if (
            defaults.get("national") != base_key
            or base["parent_release"] != build.get("build_id")
            or base["parent_release"] != release_dir.name
        ):
            raise ValueError(
                "annual projections must augment this release's certified national base"
            )
    model = _object(evidence.get("model"), "projection model")
    model_identity = {
        name: model.get(name) for name in ("version", "commit", "source_tree_sha256")
    }
    if (
        not isinstance(model_identity["version"], str)
        or not isinstance(model_identity["commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", model_identity["commit"])
        or not isinstance(model_identity["source_tree_sha256"], str)
        or not _SHA256.fullmatch(model_identity["source_tree_sha256"])
        or acceptance.get("model") != model_identity
    ):
        raise ValueError(
            "annual acceptance must bind the projection model's version and source identity"
        )
    runtime = _object(
        _object(evidence.get("runtime"), "projection runtime").get("versions"),
        "projection versions",
    )
    accepted_runtime = _object(acceptance.get("runtime"), "accepted runtime")
    for package, compatible_field, built_field in (
        ("policyengine-us", "compatible_model_packages", "built_with_model_package"),
        ("policyengine-core", "compatible_core_packages", "built_with_core_package"),
    ):
        version = runtime.get(package)
        built = _object(build.get(built_field), f"release {built_field}")
        allowed = built.get("name") == package and built.get("version") == version
        for claim in manifest.get(compatible_field, []):
            claim = _object(claim, f"release {compatible_field} entry")
            if claim.get("name") == package and isinstance(version, str):
                raw_specifier = claim.get("specifier")
                if not isinstance(raw_specifier, str) or not raw_specifier.strip():
                    raise ValueError(
                        "annual compatibility claims require nonempty specifiers"
                    )
                specifier = SpecifierSet(raw_specifier)
                if not tuple(specifier):
                    raise ValueError(
                        "annual compatibility claims require nonempty specifiers"
                    )
                allowed = allowed or version in specifier
        if (
            not isinstance(version, str)
            or accepted_runtime.get(package) != version
            or not allowed
        ):
            raise ValueError(
                f"annual runtime {package} must match acceptance and the release compatibility contract"
            )
    if runtime["policyengine-us"] != model["version"]:
        raise ValueError("projection model and runtime versions differ")
    evidence_artifacts = _object(evidence.get("artifacts"), "projection artifacts")
    accepted_years = _object(acceptance.get("years"), "annual acceptance years")
    all_years: set[str] = set()
    additional_artifacts = {
        evidence_key: evidence_path,
        acceptance_key: acceptance_path,
    }
    projected_artifacts: set[str] = set()
    for family, mapping in families.items():
        if family != base_key:
            raise ValueError(
                "each projection receipt must describe its declared base family"
            )
        years = _object(mapping, f"annual family {family!r}")
        if not years or years.get(str(base_year)) != base_key:
            raise ValueError("annual family must include its unchanged base year")
        if any(
            not isinstance(year, str) or not _YEAR.fullmatch(year) for year in years
        ):
            raise ValueError("annual year keys must be four-digit decimal strings")
        declared_years = {int(year) for year in years}
        if declared_years != set(range(base_year, max(declared_years) + 1)):
            raise ValueError(
                "annual family must cover every year from its base through its last year"
            )
        seen: set[str] = set()
        for year_text, key in years.items():
            if not isinstance(key, str) or key in seen or int(year_text) < base_year:
                raise ValueError(
                    "annual artifacts must be unique and no earlier than their source"
                )
            seen.add(key)
            all_years.add(year_text)
            path, sha = artifact(key)
            if key != base_key:
                if not build_folder and artifacts[key]["path"].startswith("releases/"):
                    raise ValueError(
                        "projected annual H5 artifacts must use root paths"
                    )
                if artifacts[key].get("kind") != "microdata":
                    raise ValueError("projected annual H5 artifacts must be microdata")
                additional_artifacts[key] = path
                projected_artifacts.add(key)
            record = _object(
                evidence_artifacts.get(key), f"projection artifact {key!r}"
            )
            if record.get("sha256") != sha or record.get("year") != int(year_text):
                raise ValueError(
                    f"{year_text} artifact does not match projection evidence"
                )
            if key != base_key:
                receipt = _object(
                    record.get("projection_receipt"),
                    f"{year_text} projection receipt",
                )
                filename = receipt.get("path")
                if (
                    not isinstance(filename, str)
                    or not filename
                    or filename in {".", ".."}
                    or "/" in filename
                    or "\\" in filename
                ):
                    raise ValueError("projection receipt path must be a bare filename")
                matches = [
                    receipt_key
                    for receipt_key, entry in artifacts.items()
                    if isinstance(entry, Mapping)
                    and entry.get("path") == prefix + filename
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"{year_text} projection receipt {filename!r} requires one "
                        "declared artifact with that bare file name"
                        if build_folder
                        else f"{year_text} projection receipt requires one declared "
                        "artifact with its exact release prefix"
                    )
                receipt_key = matches[0]
                receipt_path, receipt_sha = artifact(receipt_key)
                if receipt_sha != receipt.get("sha256"):
                    raise ValueError(
                        f"{year_text} projection receipt hash differs from candidate"
                    )
                receipt_data = _read_json(receipt_path)
                if (
                    receipt_data.get("year") != int(year_text)
                    or receipt_data.get("base_year") != base_year
                ):
                    raise ValueError(
                        f"{year_text} projection receipt has different year/base_year"
                    )
                additional_artifacts[receipt_key] = receipt_path
            accepted = _object(accepted_years.get(year_text), f"{year_text} acceptance")
            checks = _object(accepted.get("checks"), f"{year_text} acceptance checks")
            if (
                accepted.get("dataset") != key
                or accepted.get("sha256") != sha
                or not _CHECKS.issubset(checks)
                or any(checks[name] != "passed" for name in _CHECKS)
            ):
                raise ValueError(
                    f"{year_text} lacks passing acceptance for its exact artifact"
                )
            _check_native_identity(base_path, path, int(year_text), record)
    if set(accepted_years) != all_years:
        raise ValueError("annual acceptance coverage differs from the release coverage")
    if build_folder:
        pinned = sorted(
            str(key)
            for key, record in artifacts.items()
            if isinstance(record, Mapping) and "revision" in record
        )
        if pinned:
            raise ValueError(
                f"annual build-folder artifacts {pinned} pin a revision; a build "
                "folder's files have no tag"
            )
        return AnnualProjectionExtension(
            None, additional_artifacts, frozenset(projected_artifacts)
        )
    revisions = [
        _object(record, f"artifact {key!r}").get("revision")
        for key, record in artifacts.items()
    ]
    if (
        any(not isinstance(revision, str) for revision in revisions)
        or len(set(revisions)) != 1
    ):
        raise ValueError("annual artifacts must share one immutable annual cut tag")
    revision = revisions[0]
    if not isinstance(revision, str) or not re.fullmatch(
        re.escape(release_dir.name) + r"-annual-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}",
        revision,
    ):
        raise ValueError("annual artifacts require an immutable annual cut tag")
    return AnnualProjectionExtension(
        revision, additional_artifacts, frozenset(projected_artifacts)
    )


def is_annual_build(release_dir: Path | str) -> bool:
    """Whether ``release_dir`` is a build of the annual release, by its name."""
    return Path(release_dir).name == US_ANNUAL_RELEASE_ID


def require_base_publication(
    api: object, release_dir: Path | str, *, repo_id: str
) -> None:
    """Refuse unless the annual build's base national build is published.

    The publisher calls this after the build passes its contract and before
    anything is handed over. The base build's ``release_manifest.json`` is read
    where :func:`microcosm.data.build_folder_manifests.referenced_build` says
    for ``metadata.annual_projection_base``: at ``main`` from
    ``releases/microcosm-us-2024-national/<base build>/``, which never
    changes. It must name that release and build; its default dataset must
    be the folder's base-year H5, with the same file name and SHA-256; and
    every other file it declares, apart from the manifests and the checksum
    ledger the emitter writes for each build, must be in the annual build's
    manifest with the same path and SHA-256; and ``releases.json`` must list
    it and not have withdrawn it. An absent manifest refuses; any
    other Hub failure, including one where the Hub could not be asked,
    propagates. ``api`` needs ``hf_hub_download``.
    """
    from microcosm.data.build_folder_manifests import (
        referenced_build,
        require_offered_build,
    )
    from microcosm.data.release import (
        _is_absent_entry_error,
        _is_absent_revision_error,
    )

    manifest = _read_json(Path(release_dir) / "release_manifest.json")
    reference = _object(
        _object(manifest.get("metadata"), "release metadata").get(
            "annual_projection_base"
        ),
        "metadata.annual_projection_base",
    )
    location = referenced_build(reference)
    if location.repo_id != repo_id:
        raise ValueError(
            f"the annual build's base build is in {location.repo_id!r}, not the "
            f"publication repository {repo_id!r}"
        )
    base_key = _object(manifest.get("default_datasets"), "release defaults").get(
        "national"
    )
    local = _object(
        _object(manifest.get("artifacts"), "release artifacts").get(base_key),
        f"artifact {base_key!r}",
    )
    base_build = reference.get("build")
    path_in_repo = f"{location.folder}/release_manifest.json"
    try:
        downloaded = api.hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            revision=location.revision,
        )
    except Exception as exc:
        if not (_is_absent_entry_error(exc) or _is_absent_revision_error(exc)):
            raise
        raise ValueError(
            f"base build {base_build!r} is not published: {path_in_repo} is "
            f"absent at {location.revision!r} in {repo_id} ({exc}). An annual "
            "build is published only after the build it projects."
        ) from exc
    published = _read_json(Path(downloaded))
    if (published.get("release"), published.get("build")) != (
        reference.get("release"),
        base_build,
    ):
        raise ValueError(
            f"{path_in_repo} names release {published.get('release')!r} and "
            f"build {published.get('build')!r}, not the base build "
            f"{base_build!r} of {reference.get('release')!r}"
        )
    published_defaults = published.get("default_datasets")
    published_key = (
        published_defaults.get("national")
        if isinstance(published_defaults, Mapping)
        else None
    )
    published_artifacts = published.get("artifacts")
    dataset = (
        published_artifacts.get(published_key)
        if isinstance(published_artifacts, Mapping) and isinstance(published_key, str)
        else None
    )
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("kind") != "microdata"
        or dataset.get("sha256") != local.get("sha256")
        or dataset.get("path") != local.get("path")
    ):
        published_dataset = dataset if isinstance(dataset, Mapping) else {}
        raise ValueError(
            f"the published base build's dataset ({path_in_repo}) is "
            f"{published_dataset.get('path')!r} with sha256 "
            f"{published_dataset.get('sha256')!r}, not the base-year H5 "
            f"{local.get('path')!r} with sha256 {local.get('sha256')!r}; an "
            "annual build projects exactly the published base build's bytes"
        )
    carried = _object(manifest.get("artifacts"), "release artifacts")
    differing = sorted(
        key
        for key, entry in published_artifacts.items()
        if key != published_key
        and isinstance(entry, Mapping)
        and PurePosixPath(str(entry.get("path", ""))).name not in _PUBLISHER_FILES
        and not (
            isinstance(carried.get(key), Mapping)
            and (carried[key].get("path"), carried[key].get("sha256"))
            == (entry.get("path"), entry.get("sha256"))
        )
    )
    if differing:
        raise ValueError(
            f"the annual build does not carry the published base build's "
            f"{differing} unchanged ({path_in_repo} declares their paths and "
            "SHA-256 values); an annual build carries every file of its base "
            "build as published"
        )
    # A base build its repository has withdrawn, or no longer lists, is not
    # one to project.
    require_offered_build(
        api,
        repo_id=repo_id,
        release=str(reference.get("release")),
        build=str(base_build),
        label="the annual build's base build",
    )
