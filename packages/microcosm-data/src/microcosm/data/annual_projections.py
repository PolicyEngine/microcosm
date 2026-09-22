"""Additional publication checks for explicit annual US dataset families.

The parent release still has to pass its ordinary contract. These checks bind
annual acceptance evidence to the exact files, years, and source population;
they do not turn a candidate build receipt into a calibration certificate.
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

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_YEAR = re.compile(r"[1-9][0-9]{3}\Z")
_ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
_CHECKS = frozenset(
    {"schema", "source_identity", "year", "demographics", "input_aggregates", "runtime"}
)


@dataclass(frozen=True)
class AnnualProjectionExtension:
    """Validated publication additions; inherited artifacts keep their contract."""

    revision: str
    additional_artifacts: Mapping[str, Path]
    projected_artifacts: frozenset[str]


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
    key: str, artifacts: Mapping, release_dir: Path, artifact_root: Path | None
) -> tuple[Path, str]:
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

    Releases without ``metadata.dataset_years`` retain their existing contract.
    A present annual map requires separate passing acceptance evidence. The
    publisher calls this in addition to the base release's validation.
    """
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping) or "dataset_years" not in metadata:
        return
    release_dir = Path(release_dir)
    root = Path(artifact_root) if artifact_root is not None else None
    families = _object(metadata["dataset_years"], "metadata.dataset_years")
    if not families:
        raise ValueError("metadata.dataset_years must not be empty")
    artifacts = _object(manifest.get("artifacts"), "release artifacts")
    evidence_key = metadata.get("annual_projection_manifest")
    acceptance_key = metadata.get("annual_projection_acceptance")
    if not isinstance(evidence_key, str) or not isinstance(acceptance_key, str):
        raise ValueError(
            "annual releases require projection manifest and acceptance artifact keys"
        )
    evidence_path, evidence_sha = _artifact(evidence_key, artifacts, release_dir, root)
    acceptance_path, _ = _artifact(acceptance_key, artifacts, release_dir, root)
    prefix = f"releases/{release_dir.name}/"
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
    base_path, base_sha = _artifact(base_key, artifacts, release_dir, root)
    if base_sha != base.get("sha256"):
        raise ValueError(
            "projection base hash differs from the release's base artifact"
        )
    defaults = _object(manifest.get("default_datasets"), "release defaults")
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
            path, sha = _artifact(key, artifacts, release_dir, root)
            if key != base_key:
                if artifacts[key]["path"].startswith("releases/"):
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
                        f"{year_text} projection receipt requires one declared "
                        "artifact with its exact release prefix"
                    )
                receipt_key = matches[0]
                receipt_path, receipt_sha = _artifact(
                    receipt_key, artifacts, release_dir, root
                )
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
