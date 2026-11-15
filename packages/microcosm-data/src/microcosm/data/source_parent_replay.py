"""Strict, local replay of the reviewed historical US source parent.

The historical receipt is immutable. A fresh original-runtime receipt must
equal it in full, including installation origins. Current source qualification
is separate evidence and never makes an old schema-1 report current evidence.
Importing this module performs no model, HDF, network, or subprocess work.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

SOURCE_PARENT_EVIDENCE_FILE = "source_parent_evidence_v2.json"
SOURCE_PARENT_EVIDENCE_KEY = "source_parent_evidence_v2"
SOURCE_PARENT_QUALIFIED = "v2_source_parent_qualified"
CURRENT_SOURCE_CHECKS = (
    "source_inheritance",
    "exhaustive_h5_preservation",
    "reviewed_source_evidence",
    "current_native_loaders",
    "current_stored_input_register",
    "current_claim_measurement",
    "current_producer_identity",
)
PROFILE_FIELDS = frozenset(
    {
        "build_id",
        "release_manifest_sha256",
        "build_manifest_sha256",
        "source_report_sha256",
        "native_receipt_sha256",
        "dataset_sha256",
        "ancestor_dataset_sha256",
        "producer_commit",
        "producer_runner_sha256",
    }
)
RUNTIME_FIELDS = frozenset(
    {"python", "hdf5", "microcosm-data", "numpy", "pandas", "h5py", "tables"}
)
ARTIFACT_MAP_FIELDS = frozenset(
    {"source_key", "source_path", "carried_key", "carried_path", "kind", "sha256"}
)
_ANCESTOR_NAMES = MappingProxyType(
    {
        "parent_release_manifest": "ancestor_release_manifest",
        "parent_build_manifest": "ancestor_build_manifest",
    }
)

SOURCE_PARENT_IMPLEMENTATION_FILES = (
    "packages/microcosm-data/src/microcosm/data/source_parent_replay.py",
    "packages/microcosm-data/src/microcosm/data/source_parent_worker.py",
    "packages/microcosm-data/src/microcosm/data/compatibility_recertification.py",
    "packages/microcosm-data/src/microcosm/data/source_enrichment.py",
    "tools/build_us_compatibility_recertification.py",
    "packages/microcosm-data/src/microcosm/data/contract.py",
    "packages/microcosm-data/src/microcosm/data/release.py",
    "packages/microcosm-build/src/microcosm/build/release_publication.py",
)
NATIVE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "scope",
        "external_package_publication",
        "canonical_spm_model_acceptance",
        "dataset_sha256",
        "runner_sha256",
        "packages",
        "loaded_source_packages",
        "loaded_source_sha256",
        "checks",
    }
)


def _is_sha(value, length=64):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def _validate_native_receipt(receipt, *, current, profile):
    _exact_keys(
        receipt,
        NATIVE_RECEIPT_FIELDS | ({"stored_inputs"} if current else set()),
        "native receipt",
    )
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("native receipt schema must be integer 1")
    if (
        receipt["status"] != "passed"
        or receipt["scope"] != "native_input_loading_only"
        or receipt["external_package_publication"] != "not_attested"
        or receipt["canonical_spm_model_acceptance"] != "not_attested"
        or receipt["dataset_sha256"] != profile["dataset_sha256"]
        or not _is_sha(receipt["runner_sha256"])
    ):
        raise ValueError("native receipt status/scope/identity is invalid")
    names = {"policyengine", "policyengine-core", "policyengine-us", "spm-calculator"}
    _exact_keys(receipt["packages"], names, "native packages")
    for identity in receipt["packages"].values():
        _exact_keys(
            identity,
            {"version", "source_sha256", "direct_url", "wheel_sha256"},
            "package identity",
        )
        if (
            not isinstance(identity["version"], str)
            or not identity["version"]
            or not all(
                _is_sha(identity[name]) for name in ("source_sha256", "wheel_sha256")
            )
            or not (
                identity["direct_url"] is None
                or isinstance(identity["direct_url"], Mapping)
            )
        ):
            raise ValueError("native package identity is incomplete")
    loaded = {"country_loader", "native_role", "wrapper_loader"}
    _exact_keys(receipt["loaded_source_packages"], loaded, "native loaded packages")
    _exact_keys(receipt["loaded_source_sha256"], loaded, "native loaded sources")
    if not all(_is_sha(value) for value in receipt["loaded_source_sha256"].values()):
        raise ValueError("native loaded source digest is invalid")
    checks = receipt["checks"]
    if (
        not isinstance(checks, (list, tuple))
        or not checks
        or not all(isinstance(value, str) and value for value in checks)
    ):
        raise ValueError("native receipt checks must be explicit ordered strings")
    if current:
        _exact_keys(
            receipt["stored_inputs"],
            {"register_sha256", "registered_non_variables"},
            "current stored inputs",
        )
        registered = receipt["stored_inputs"]["registered_non_variables"]
        if (
            not _is_sha(receipt["stored_inputs"]["register_sha256"])
            or not isinstance(registered, (list, tuple))
            or not all(isinstance(value, str) for value in registered)
        ):
            raise ValueError("current stored-input evidence is malformed")
        if "country:defines_every_stored_model_input" not in checks:
            raise ValueError("current native receipt must attest stored-input checking")
    elif receipt["runner_sha256"] != profile["producer_runner_sha256"]:
        raise ValueError("legacy native runner differs from reviewed source")


def _reviewed_profile() -> Mapping:
    """The sole reviewed lineage; this is not a user-configurable allow-list."""
    return MappingProxyType(
        {
            "build_id": "populace-us-2024-spm-20260915",
            "release_manifest_sha256": "d5c9e2a33d097294af60a816fafb1bdcfbbd81cdda70ae4637daceb829dedc01",
            "build_manifest_sha256": "f3fd734867324bca7619f5f083a8115e32104bea3948d48723b345cfd39fa03b",
            "source_report_sha256": "85329907b2b9c4299855b0970b1f24ede5819d3b16c312eb7400b4203bd71203",
            "native_receipt_sha256": "63109acffd486453c99c476dece6b0154cc5388e5a430830a30744a2249444d1",
            "dataset_sha256": "6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84",
            "ancestor_dataset_sha256": "48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e",
            "producer_commit": "743683b0e121378e93c9b426cd9c79d41d29595b",
            "producer_runner_sha256": "14a63acb510fb43c79a917761b1ed3e7f4b11c08ebf586a8750afb461784f2cd",
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    def invalid_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    value = json.loads(
        path.read_text(),
        object_pairs_hook=_object_pairs,
        parse_constant=invalid_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _plain_json(value):
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("evidence must contain finite JSON values only")


def _same_json(left, right) -> bool:
    """Object ordering is immaterial; JSON value types and array order are not."""

    def encode(value):
        return json.dumps(
            _plain_json(value), sort_keys=True, allow_nan=False, separators=(",", ":")
        )

    return encode(left) == encode(right)


def _freeze_json(value):
    value = _plain_json(value)
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _exact_keys(value, fields, label):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{label} must contain exactly {sorted(fields)}")


def _bare_name(value) -> bool:
    return (
        isinstance(value, str)
        and value not in {"", ".", ".."}
        and Path(value).name == value
        and "/" not in value
        and "\\" not in value
    )


def _regular_file(path: Path) -> None:
    if not path.is_file() or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"{path} must be a regular file without a symlink")


@dataclass(frozen=True)
class SourceParentReplayContext:
    """Explicit local inputs, never a passed bit or caller-selected worker."""

    original_python: Path
    original_checkout: Path
    original_wheels: tuple[Path, ...]
    ancestor_h5: Path
    worker_timeout_seconds: float = 600.0

    def __post_init__(self):
        for name in ("original_python", "original_checkout", "ancestor_h5"):
            raw = getattr(self, name)
            if not isinstance(raw, (str, Path)):
                raise ValueError(f"{name} must be an absolute local path")
            value = Path(raw)
            if not value.is_absolute():
                raise ValueError(f"{name} must be an absolute local path")
            object.__setattr__(self, name, value)
        if not isinstance(self.original_wheels, (list, tuple)) or not all(
            isinstance(path, (str, Path)) for path in self.original_wheels
        ):
            raise ValueError("original_wheels must contain four absolute local paths")
        wheels = tuple(Path(path) for path in self.original_wheels)
        if (
            len(wheels) != 4
            or len(set(wheels)) != 4
            or not all(path.is_absolute() for path in wheels)
        ):
            raise ValueError(
                "original_wheels must contain four distinct absolute paths"
            )
        object.__setattr__(self, "original_wheels", wheels)
        timeout = self.worker_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or (not math.isfinite(timeout) or timeout <= 0)
        ):
            raise ValueError("worker_timeout_seconds must be positive and finite")

    @classmethod
    def from_dict(cls, value: Mapping) -> SourceParentReplayContext:
        required = {
            "original_python",
            "original_checkout",
            "original_wheels",
            "ancestor_h5",
        }
        if (
            not isinstance(value, Mapping)
            or not required.issubset(value)
            or (set(value) - required - {"worker_timeout_seconds"})
        ):
            raise ValueError(
                "source-parent replay config has unknown or missing fields"
            )
        if not isinstance(value["original_wheels"], (list, tuple)):
            raise ValueError("original_wheels must be a JSON array")
        return cls(**dict(value))

    @classmethod
    def from_json(cls, path: Path | str) -> SourceParentReplayContext:
        return cls.from_dict(_read_json(Path(path)))

    def to_dict(self) -> dict:
        return {
            "original_python": str(self.original_python),
            "original_checkout": str(self.original_checkout),
            "original_wheels": [str(path) for path in self.original_wheels],
            "ancestor_h5": str(self.ancestor_h5),
            "worker_timeout_seconds": self.worker_timeout_seconds,
        }


@dataclass(frozen=True)
class SourceParentEvidenceV2:
    """A complete, separate v2 envelope; saved evidence is always replayed."""

    schema_version: int
    kind: str
    policy: str
    profile: Mapping
    artifact_map: tuple[Mapping, ...]
    legacy_actual: Mapping
    legacy_runtime: Mapping
    current: Mapping

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 2
            or (
                self.kind != "source_parent_replay"
                or self.policy != "exact_legacy_receipt"
            )
        ):
            raise ValueError(
                "unsupported source-parent evidence schema, kind or policy"
            )
        _exact_keys(self.profile, PROFILE_FIELDS, "source-parent profile")
        if dict(self.profile) != dict(_reviewed_profile()):
            raise ValueError("source-parent evidence names an unreviewed profile")
        _exact_keys(self.legacy_runtime, RUNTIME_FIELDS, "legacy runtime")
        if not all(
            isinstance(value, str) and value for value in self.legacy_runtime.values()
        ):
            raise ValueError("legacy runtime versions must be nonempty strings")
        _validate_native_receipt(
            self.legacy_actual, current=False, profile=self.profile
        )
        if not isinstance(self.artifact_map, (list, tuple)) or not self.artifact_map:
            raise ValueError("artifact map must be a complete nonempty array")
        seen = {
            name: set()
            for name in ("source_key", "source_path", "carried_key", "carried_path")
        }
        for row in self.artifact_map:
            _exact_keys(row, ARTIFACT_MAP_FIELDS, "artifact-map row")
            for name, values in seen.items():
                if not _bare_name(row[name]) or row[name] in values:
                    raise ValueError(
                        "artifact map must be safe and injective in both directions"
                    )
                values.add(row[name])
            if (
                not _is_sha(row["sha256"])
                or not isinstance(row["kind"], str)
                or not row["kind"]
            ):
                raise ValueError(
                    "artifact map requires explicit kinds and SHA-256 values"
                )
        current_fields = {
            "schema_version",
            "kind",
            "profile",
            "checks",
            "native_receipt",
            "compatibility",
            "producer_code",
        }
        _exact_keys(self.current, current_fields, "current qualification")
        if (
            type(self.current["schema_version"]) is not int
            or self.current["schema_version"] != 2
        ):
            raise ValueError("current qualification must be a distinct schema-2 record")
        _exact_keys(self.current["profile"], PROFILE_FIELDS, "current profile")
        if self.current["kind"] != "source_parent_current_qualification" or (
            dict(self.current["profile"]) != dict(self.profile)
        ):
            raise ValueError("current qualification must bind the same source parent")
        if tuple(self.current["checks"]) != CURRENT_SOURCE_CHECKS:
            raise ValueError("every current source-parent check is required")
        for name in ("native_receipt", "compatibility", "producer_code"):
            if not isinstance(self.current[name], Mapping):
                raise ValueError(f"current {name} must be an object")
        _validate_native_receipt(
            self.current["native_receipt"], current=True, profile=self.profile
        )
        if not _same_json(
            self.current["native_receipt"]["packages"], self.legacy_actual["packages"]
        ):
            raise ValueError(
                "current source replay must use the authenticated original four packages"
            )
        compatibility = self.current["compatibility"]
        if (
            set(compatibility) - {"status", "publisher_claims", "claim_measurement"}
            or compatibility.get("status") != "passed"
        ):
            raise ValueError(
                "current compatibility must be fresh passed claim evidence"
            )
        code = self.current["producer_code"]
        _exact_keys(
            code,
            {"git_commit", "git_dirty", "source_files_sha256"},
            "current producer code",
        )
        hashes = code["source_files_sha256"]
        if (
            not _is_sha(code["git_commit"], 40)
            or code["git_dirty"] is not False
            or not isinstance(hashes, Mapping)
            or not set(SOURCE_PARENT_IMPLEMENTATION_FILES).issubset(hashes)
            or not all(
                isinstance(name, str) and _is_sha(value)
                for name, value in hashes.items()
            )
        ):
            raise ValueError(
                "current producer must record authenticated clean committed sources"
            )
        for field in (
            "profile",
            "artifact_map",
            "legacy_actual",
            "legacy_runtime",
            "current",
        ):
            object.__setattr__(self, field, _freeze_json(getattr(self, field)))

    @classmethod
    def from_dict(cls, value: Mapping) -> SourceParentEvidenceV2:
        _exact_keys(value, cls.__dataclass_fields__, "source-parent evidence")
        return cls(**dict(value))

    def to_dict(self) -> dict:
        return {
            name: _plain_json(getattr(self, name)) for name in self.__dataclass_fields__
        }


def describe_source_parent_profile(release_dir: Path | str) -> dict:
    """Authenticate only the exact reviewed native-role source parent metadata."""
    release_dir = Path(release_dir)
    profile = dict(_reviewed_profile())
    _exact_keys(profile, PROFILE_FIELDS, "reviewed profile")
    if release_dir.name != profile["build_id"]:
        raise ValueError("only the reviewed tagged source parent is supported")
    files = {
        "release_manifest.json": "release_manifest_sha256",
        "build_manifest.json": "build_manifest_sha256",
        "source_enrichment.json": "source_report_sha256",
        "source_enrichment_compatibility.json": "native_receipt_sha256",
    }
    for filename, field in files.items():
        path = release_dir / filename
        _regular_file(path)
        if _sha256(path) != profile[field]:
            raise ValueError(f"unreviewed source-parent {filename}")
    manifest = _read_json(release_dir / "release_manifest.json")
    build = _read_json(release_dir / "build_manifest.json")
    report = _read_json(release_dir / "source_enrichment.json")
    receipt = _read_json(release_dir / "source_enrichment_compatibility.json")
    if manifest.get("release_type") != "source_enrichment" or (
        manifest.get("dataset_role", "national_default") != "national_default"
        or build.get("release_type") != "source_enrichment"
        or build.get("build_id") != profile["build_id"]
        or report.get("operation") != "add_native_spm_independent_minor_role"
        or "annual_projection" in manifest
    ):
        raise ValueError("unknown, recursive, annual or non-national source parent")
    if build.get("code", {}).get("git_commit") != profile["producer_commit"] or (
        build.get("code", {}).get("git_dirty") is not False
    ):
        raise ValueError("source parent must retain its authenticated clean producer")
    if report.get("dataset", {}).get("sha256") != profile["dataset_sha256"] or (
        receipt.get("dataset_sha256") != profile["dataset_sha256"]
        or receipt.get("runner_sha256") != profile["producer_runner_sha256"]
        or report.get("parent", {}).get("dataset_sha256")
        != profile["ancestor_dataset_sha256"]
    ):
        raise ValueError(
            "source parent, ancestor and native runner identities disagree"
        )
    for name in ("parent_release_manifest.json", "parent_build_manifest.json"):
        path = release_dir / name
        _regular_file(path)
        entry = manifest.get("artifacts", {}).get(name.removesuffix(".json"), {})
        if _sha256(path) != entry.get("sha256"):
            raise ValueError(f"source-parent ancestor artifact {name} is not immutable")
    ancestor = _read_json(release_dir / "parent_release_manifest.json")
    if ancestor.get("release_type") not in (None, "calibration") or (
        ancestor.get("dataset_role", "national_default") != "national_default"
    ):
        raise ValueError("a source parent must have exactly one calibration ancestor")
    return profile


def build_source_parent_artifact_map(
    parent_manifest: Mapping,
    *,
    child_dataset_key: str,
    child_dataset_filename: str,
) -> tuple[dict, ...]:
    """Return a complete injective key/path map with an exact inverse."""
    if not _bare_name(child_dataset_key) or not _bare_name(child_dataset_filename):
        raise ValueError("child dataset identifiers must be bare names")
    artifacts = parent_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("parent artifacts must be an object")
    dataset_key = parent_manifest.get("default_datasets", {}).get("national")
    rows, source_paths, keys, paths = [], set(), set(), set()
    reserved = {
        "release_manifest",
        "compatibility_recertification",
        "compatibility_recertification_loader",
        "compatibility_drift",
        SOURCE_PARENT_EVIDENCE_KEY,
        *_ANCESTOR_NAMES.values(),
    }
    for key, entry in artifacts.items():
        if not _bare_name(key) or not isinstance(entry, Mapping):
            raise ValueError("parent artifact keys and entries must be explicit")
        path = entry.get("path")
        if not _bare_name(path) or path in source_paths or key in reserved:
            raise ValueError("unsafe, duplicated or reserved source artifact")
        source_paths.add(path)
        destination_key, destination_path = key, path
        if key in _ANCESTOR_NAMES:
            if path != key + ".json":
                raise ValueError("unknown ancestor alias")
            destination_key = _ANCESTOR_NAMES[key]
            destination_path = destination_key + ".json"
        elif key == "build_manifest":
            if path != "build_manifest.json":
                raise ValueError("unknown build-manifest alias")
            destination_key, destination_path = (
                "parent_build_manifest",
                "parent_build_manifest.json",
            )
        elif path in {
            "parent_release_manifest.json",
            "parent_build_manifest.json",
            "build_manifest.json",
        }:
            raise ValueError("reserved file has an unknown artifact key")
        elif key == dataset_key:
            destination_key, destination_path = (
                child_dataset_key,
                child_dataset_filename,
            )
        if (
            destination_key in keys
            or destination_path in paths
            or (
                destination_key in reserved
                or destination_path == "release_manifest.json"
            )
        ):
            # The two reviewed ancestor destinations are allowed only for their
            # explicitly named source keys, never for another incoming artifact.
            if not (
                key in _ANCESTOR_NAMES
                and destination_key not in keys
                and destination_path not in paths
            ):
                raise ValueError("source-parent artifact mapping collides")
        reserved_paths = {name + ".json" for name in reserved} | {
            "build_manifest.json",
            "parent_release_manifest.json",
            "parent_build_manifest.json",
        }
        if path in reserved_paths and key not in {
            "parent_release_manifest",
            "parent_build_manifest",
            "build_manifest",
        }:
            raise ValueError("reserved filename has an unknown source key")
        kind = entry.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("artifact kind must be a nonempty string")
        sha = entry.get("sha256")
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("artifact map requires an actual SHA-256")
        keys.add(destination_key)
        paths.add(destination_path)
        rows.append(
            {
                "source_key": key,
                "source_path": path,
                "carried_key": destination_key,
                "carried_path": destination_path,
                "kind": kind,
                "sha256": sha,
            }
        )
    if dataset_key not in {row["source_key"] for row in rows}:
        raise ValueError("parent map must include its selected native dataset")
    return tuple(rows)


def verify_source_parent_artifact_map(
    parent_manifest: Mapping,
    artifact_map,
    *,
    child_dataset_key: str,
    child_dataset_filename: str,
) -> None:
    expected = build_source_parent_artifact_map(
        parent_manifest,
        child_dataset_key=child_dataset_key,
        child_dataset_filename=child_dataset_filename,
    )
    if not _same_json(artifact_map, expected):
        raise ValueError("source-parent artifact map is incomplete or changed")


def source_parent_contract_record(evidence_path: Path | str) -> dict:
    """Bind separately replayed v2 evidence; this does not run or accept a worker."""
    evidence_path = Path(evidence_path)
    SourceParentEvidenceV2.from_dict(_read_json(evidence_path))
    return {
        "status": SOURCE_PARENT_QUALIFIED,
        "replaced_failures": [],
        "other_failures": [],
        "evidence_file": SOURCE_PARENT_EVIDENCE_FILE,
        "evidence_sha256": _sha256(evidence_path),
    }


def qualify_source_parent(
    release_dir: Path | str,
    *,
    artifact_root: Path | str,
    context: SourceParentReplayContext,
) -> SourceParentEvidenceV2:
    """Run both fresh workers, never accept a cached receipt or passed flag."""
    if not isinstance(context, SourceParentReplayContext):
        raise ValueError("source parent requires an explicit SourceParentReplayContext")
    release_dir, artifact_root = Path(release_dir), Path(artifact_root)
    profile = describe_source_parent_profile(release_dir)
    manifest = _read_json(release_dir / "release_manifest.json")
    artifact_map = build_source_parent_artifact_map(
        manifest,
        child_dataset_key="microcosm_us_2024",
        child_dataset_filename="microcosm_us_2024.h5",
    )
    dataset_key = manifest["default_datasets"]["national"]
    for key, entry in manifest["artifacts"].items():
        path = (artifact_root if key == dataset_key else release_dir) / entry["path"]
        _regular_file(path)
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"source-parent artifact changed before replay: {key}")
        if key == dataset_key and (
            (release_dir / entry["path"]).exists()
            or (release_dir / entry["path"]).is_symlink()
        ):
            raise ValueError("source-parent H5 must live only in artifact_root")
    original = _read_json(release_dir / "source_enrichment_compatibility.json")
    declared_runtime = _read_json(release_dir / "build_manifest.json")["code"][
        "runtime"
    ]
    legacy = _run_worker(
        context, mode="legacy", release_dir=release_dir, artifact_root=artifact_root
    )
    _exact_keys(legacy, {"legacy_actual", "legacy_runtime"}, "legacy worker output")
    if not _same_json(legacy["legacy_actual"], original) or not _same_json(
        legacy["legacy_runtime"], declared_runtime
    ):
        raise ValueError(
            "fresh legacy receipt/origins/runtime differ from immutable originals"
        )
    current = _run_worker(
        context, mode="current", release_dir=release_dir, artifact_root=artifact_root
    )
    return SourceParentEvidenceV2(
        schema_version=2,
        kind="source_parent_replay",
        policy="exact_legacy_receipt",
        profile=profile,
        artifact_map=artifact_map,
        legacy_actual=legacy["legacy_actual"],
        legacy_runtime=legacy["legacy_runtime"],
        current=current,
    )


def verify_source_parent_evidence(
    release_dir: Path | str,
    parent: Mapping,
    *,
    artifact_root: Path | str,
    context: SourceParentReplayContext,
) -> dict:
    """Reconstruct and rerun both workers before accepting stored v2 evidence."""
    from microcosm.data import compatibility_recertification as recert

    release_dir, artifact_root = Path(release_dir), Path(artifact_root)
    evidence_path = release_dir / SOURCE_PARENT_EVIDENCE_FILE
    _regular_file(evidence_path)
    recorded = SourceParentEvidenceV2.from_dict(_read_json(evidence_path))
    temporary_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(
        prefix=".source-parent-verify-", dir=temporary_root
    ) as tmp:
        staging = Path(tmp)
        parent_dir = recert._reconstructed_parent(release_dir, parent, staging)
        originals = staging / "artifacts"
        originals.mkdir()
        for name, sha in parent["root_artifacts"].items():
            path = artifact_root / recert.child_artifact_path(
                parent, release_dir.name, name
            )
            _regular_file(path)
            if not _bare_name(name) or _sha256(path) != sha:
                raise ValueError("original source-parent root artifact changed")
            os.link(path, originals / name)
        actual = qualify_source_parent(
            parent_dir, artifact_root=originals, context=context
        )
    if not _same_json(actual.to_dict(), recorded.to_dict()):
        raise ValueError("source-parent v2 evidence differs from fresh complete replay")
    return source_parent_contract_record(evidence_path)


def _run_worker(context, *, mode, release_dir, artifact_root):
    """Fixed, locally staged bootstrap; no operator-provided worker is accepted."""
    if mode not in {"legacy", "current"}:
        raise ValueError("unknown source-parent worker mode")
    worker = Path(__file__).with_name("source_parent_worker.py")
    _regular_file(worker)
    _regular_file(Path(__file__))
    temporary_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(
        prefix=".source-parent-worker-", dir=temporary_root
    ) as tmp:
        work = Path(tmp)
        cache = work / "bytecode"
        cache.mkdir()
        output = work / "actual.json"
        config = work / "config.json"
        payload = {
            "context": context.to_dict(),
            "mode": mode,
            "release_dir": str(release_dir),
            "artifact_root": str(artifact_root),
        }
        config.write_text(json.dumps(payload, sort_keys=True, allow_nan=False))
        command = [
            str(context.original_python),
            "-I",
            "-X",
            f"pycache_prefix={cache}",
            str(worker),
            "--config",
            str(config),
            "--config-sha256",
            _sha256(config),
            "--helper-sha256",
            _sha256(Path(__file__)),
            "--worker-sha256",
            _sha256(worker),
            "--output",
            str(output),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=context.worker_timeout_seconds,
            check=False,
        )
        if result.returncode != 0 or not output.is_file() or output.is_symlink():
            raise ValueError(
                f"{mode} source-parent worker refused: {result.stderr.strip()}"
            )
        return _read_json(output)


def _git(checkout: Path, *args: str) -> bytes:
    """Read committed objects without inherited repository-local overrides."""
    env = dict(os.environ)
    local = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.splitlines()
    for name in local:
        env.pop(name, None)
    env["GIT_NO_LAZY_FETCH"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "--no-replace-objects",
            "-C",
            str(checkout),
            *args,
        ],
        env=env,
        check=True,
        capture_output=True,
    ).stdout


def _authenticate_checkout(checkout: Path, *, commit: str, declared: Mapping) -> Path:
    """Authenticate full data-package Python closure, not merely six preimages."""
    if not checkout.is_dir() or checkout.is_symlink():
        raise ValueError("replay requires a locally staged original source checkout")
    if _git(checkout, "rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("replay checkout HEAD differs from its authenticated commit")
    if _git(checkout, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError(
            "replay source checkout must be clean, including untracked files"
        )
    source = checkout / "packages/microcosm-data/src"
    prefix = "packages/microcosm-data/src/"
    tracked = (
        _git(checkout, "ls-tree", "-r", "--name-only", commit, "--", prefix)
        .decode()
        .splitlines()
    )
    tracked_python = {name for name in tracked if name.endswith(".py")}
    actual_python = {str(path.relative_to(checkout)) for path in source.rglob("*.py")}
    if actual_python != tracked_python:
        raise ValueError("replay data Python source closure is incomplete or extended")
    if list(source.rglob("*.so")) or list(source.rglob("*.pyd")):
        raise ValueError("unexpected native module in replay data source tree")
    for relative in sorted(tracked_python | set(declared)):
        path = checkout / relative
        _regular_file(path)
        actual = _sha256(path)
        committed = hashlib.sha256(
            _git(checkout, "show", f"{commit}:{relative}")
        ).hexdigest()
        if actual != committed or (
            relative in declared and actual != declared[relative]
        ):
            raise ValueError(f"replay committed/installed source differs: {relative}")
    return source


def _authenticate_runtime(context, receipt, runtime):
    """Authenticate wheel bytes, installed inventories and original origins first."""
    import platform
    from email.parser import BytesParser
    from importlib import metadata
    from zipfile import ZipFile

    _exact_keys(runtime, RUNTIME_FIELDS, "declared original runtime")
    if platform.python_version() != runtime["python"]:
        raise ValueError("replay Python version differs from original declaration")
    for name in RUNTIME_FIELDS - {"python", "hdf5"}:
        if metadata.version(name) != runtime[name]:
            raise ValueError(f"replay declared dependency version differs: {name}")
    packages = receipt.get("packages")
    names = {"policyengine", "policyengine-core", "policyengine-us", "spm-calculator"}
    _exact_keys(packages, names, "original four package identities")
    wheels = {}
    for path in context.original_wheels:
        _regular_file(path)
        with ZipFile(path) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)):
                raise ValueError("wheel contains duplicate members")
            meta = [name for name in members if name.endswith(".dist-info/METADATA")]
            if len(meta) != 1:
                raise ValueError("original wheel must contain one METADATA")
            info = BytesParser().parsebytes(archive.read(meta[0]))
            name = info["Name"].lower().replace("_", "-")
            if name not in names or name in wheels:
                raise ValueError(
                    "original wheels must identify exactly four unique packages"
                )
            files = {
                name: archive.read(name) for name in members if name.endswith(".py")
            }
        dist = metadata.distribution(name)
        direct = dist.read_text("direct_url.json")
        origin = json.loads(direct) if direct else None
        if origin and origin.get("dir_info", {}).get("editable"):
            raise ValueError("editable original package is not an authenticated wheel")
        installed = sorted(
            str(file) for file in (dist.files or ()) if str(file).endswith(".py")
        )
        if not installed or set(installed) != set(files):
            raise ValueError(f"original installed source inventory differs: {name}")
        digest = hashlib.sha256()
        for relative in installed:
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("unsafe original wheel Python member")
            path_on_disk = Path(dist.locate_file(relative))
            _regular_file(path_on_disk)
            actual = path_on_disk.read_bytes()
            if actual != files[relative]:
                raise ValueError(
                    f"original installed Python source differs: {name}:{relative}"
                )
            digest.update(relative.encode() + b"\0" + actual + b"\0")
        identity = {
            "version": dist.version,
            "source_sha256": digest.hexdigest(),
            "direct_url": origin,
            "wheel_sha256": _sha256(path),
        }
        if info["Version"] != dist.version or not _same_json(identity, packages[name]):
            raise ValueError(f"original full package identity/origin differs: {name}")
        wheels[name] = str(path)
    if set(wheels) != names:
        raise ValueError("original four wheel bytes are required")
    # These imports are permitted only inside this explicitly requested worker,
    # after all Python/package/source identities have been authenticated.
    import h5py

    if h5py.version.hdf5_version != runtime["hdf5"]:
        raise ValueError("replay loaded HDF5 version differs from original declaration")
    return {
        **{
            name: metadata.version(name) for name in RUNTIME_FIELDS - {"python", "hdf5"}
        },
        "python": platform.python_version(),
        "hdf5": h5py.version.hdf5_version,
    }


def _worker_main(payload: Mapping) -> dict:
    """Internal fixed worker body; no user-selected code or cached proof input."""
    _exact_keys(
        payload, {"context", "mode", "release_dir", "artifact_root"}, "worker config"
    )
    context = SourceParentReplayContext.from_dict(payload["context"])
    mode = payload["mode"]
    if mode not in {"legacy", "current"}:
        raise ValueError("unknown source-parent worker mode")
    if any(
        name == "microcosm"
        or name.startswith(("microcosm.", "policyengine", "spm_calculator"))
        for name in sys.modules
    ):
        raise ValueError("model/data package loaded before replay authentication")
    release_dir, artifact_root = (
        Path(payload["release_dir"]),
        Path(payload["artifact_root"]),
    )
    profile = describe_source_parent_profile(release_dir)
    build = _read_json(release_dir / "build_manifest.json")
    receipt = _read_json(release_dir / "source_enrichment_compatibility.json")
    original_source = _authenticate_checkout(
        context.original_checkout,
        commit=profile["producer_commit"],
        declared=build["code"]["source_files_sha256"],
    )
    _regular_file(context.ancestor_h5)
    if _sha256(context.ancestor_h5) != profile["ancestor_dataset_sha256"]:
        raise ValueError("original calibration ancestor H5 differs")
    actual_runtime = _authenticate_runtime(context, receipt, build["code"]["runtime"])
    if mode == "legacy":
        source, checkout = original_source, context.original_checkout
    else:
        checkout = Path(__file__).parents[5]
        commit = _git(checkout, "rev-parse", "HEAD").decode().strip()
        source = _authenticate_checkout(checkout, commit=commit, declared={})
    for name in subprocess.run(
        ["git", "rev-parse", "--local-env-vars"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.splitlines():
        os.environ.pop(name, None)
    os.environ["GIT_NO_LAZY_FETCH"] = "1"
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    os.chdir(checkout)
    sys.path.insert(0, str(source))
    from microcosm.data import source_enrichment

    expected_module = source / "microcosm/data/source_enrichment.py"
    if Path(source_enrichment.__file__) != expected_module:
        raise ValueError("worker loaded a different data runner")
    if mode == "legacy":
        if _sha256(expected_module) != profile["producer_runner_sha256"]:
            raise ValueError("original runner source hash differs")
        source_enrichment.validate_source_enrichment_candidate(
            release_dir,
            parent_h5=context.ancestor_h5,
            artifact_root=artifact_root,
            require_compatibility=True,
            compatibility_wheels=context.original_wheels,
        )
        # The original validator owns its full original contract. Recompute its
        # raw receipt separately rather than instrumenting/patching that code.
        manifest = _read_json(release_dir / "release_manifest.json")
        entry = manifest["artifacts"][manifest["default_datasets"]["national"]]
        actual = source_enrichment.run_native_loader_compatibility(
            artifact_root / entry["path"],
            require_wheels=True,
            compatibility_wheels=context.original_wheels,
        )
        if not _same_json(actual, receipt):
            raise ValueError(
                "complete original native receipt differs, including direct_url"
            )
        result = {"legacy_actual": actual, "legacy_runtime": actual_runtime}
    else:
        result = source_enrichment.qualify_reviewed_source_parent(
            release_dir,
            parent_h5=context.ancestor_h5,
            artifact_root=artifact_root,
            compatibility_wheels=context.original_wheels,
        )
    # No namespace package or subsequently imported data module may escape the
    # authenticated Python closure. Model packages have exact wheel inventories.
    for name, module in tuple(sys.modules.items()):
        if name.startswith("microcosm.data") and getattr(module, "__file__", None):
            path = Path(module.__file__)
            if not path.is_relative_to(source):
                raise ValueError(
                    "replay data import escaped authenticated source closure"
                )
    return result
