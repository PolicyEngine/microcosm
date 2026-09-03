"""Portable authenticated identity for the primary PUF QRF worker."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

PRIMARY_QRF_WORKER_IDENTITY_SCHEMA_VERSION = 1
PRIMARY_QRF_WORKER_MODULE = "microcosm.build.us_runtime.puf_qrf_worker"
PRIMARY_QRF_INTERPRETER_PLACEHOLDER = "{python_interpreter}"
APPROVED_UV_LOCK_SHA256 = (
    "1c1444045960e111433b064abeeb40eebbf55984b7132931d9987697a6f4b606"
)
LEGACY_CAMPAIGN_UV_LOCK_SHA256 = (
    "27f47e385cfa35e2644a37410d1804b361ad9aee123577551c8421547bda65ee"
)
# The plan and reproduced STOP record expose this exact, unambiguous campaign
# identifier.  Do not accept arbitrary suffixes under the approved prefix.
LEGACY_CAMPAIGN_TREE_SHA = "b8819b3f"
LEGACY_WORKER_ATTESTATION_ARTIFACT_KIND = (
    "populace_us_worker_identity_compatibility_attestation"
)
LEGACY_WORKER_ATTESTATION_SCHEMA_VERSION = 1
LEGACY_WORKER_PERMITTED_MISMATCHES = (
    "argv_template[0]",
    "interpreter.executable",
)

_PLAN_SIGNATURE = {
    "gate": "owner-authorization:c27-root-cause:2026-09-03",
    "plan_sha256": ("0a3409cfe1560d56a78ecc9acf012abaeb32621af278d745b674ebf1bee32cf6"),
    "prompt_sha256": (
        "9c1e4508f24d0915c1f3a2942723d3c219c990679227c7d0a315295d5e76efa2"
    ),
    "checklist_sha256": (
        "5ee1f5fb40387cb690c2e85b32b6bd5abed78200f367c253e517a3917c417238"
    ),
    "evidence_sha256": (
        "85345eae623d0081354d746a118c9dc5ddaa89a641238e546d8c8e9f7aabbb44"
    ),
}
_SEMANTIC_ENVIRONMENT_NAMES = (
    "POPULACE_FIT_N_JOBS",
    "POPULACE_FIT_PREDICT_WORKERS",
)
_TRANSITIVE_PACKAGE_RESOURCES = (
    ("microcosm.build.us", "source_stages.json"),
    ("microcosm.build.us", "support_spine.json"),
)
_ATTESTATION_KEYS = {
    "artifact_kind",
    "schema_version",
    "plan_signature",
    "purpose",
    "sealed_manifest_sha256",
    "sealed_pool_h5_sha256",
    "campaign_tree_sha",
    "uv_lock_sha256",
    "installed_transitive_environment_code_sha256",
    "recorded_worker_execution",
    "semantic_identity",
    "semantic_identity_sha256",
    "permitted_mismatches",
}
_SHA256_ALPHABET = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class LegacyWorkerIdentityAuthentication:
    """Validated scoring-only authority for one sealed legacy worker binding."""

    attestation_sha256: str
    campaign_tree_sha: str
    recorded_worker_execution: Mapping[str, object]
    semantic_identity: Mapping[str, object]
    semantic_identity_sha256: str

    def receipt(self) -> dict[str, object]:
        """Return the portable authentication evidence exposed to consumers."""

        interpreter = self.recorded_worker_execution["interpreter"]
        argv = self.recorded_worker_execution["argv_template"]
        assert isinstance(interpreter, Mapping)
        assert isinstance(argv, list)
        return {
            "manifest_schema_version": 9,
            "execution_config_schema_version": 4,
            "worker_execution_schema_version": 0,
            "semantic_identity_sha256": self.semantic_identity_sha256,
            "audit_aliases": {
                "sys_executable": interpreter["executable"],
                "argv_template_0": argv[0],
            },
            "compatibility_attestation_sha256": self.attestation_sha256,
            "purpose": "scoring_only",
        }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_clone(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _require_sha256(value: object, *, boundary: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_ALPHABET for character in value)
    ):
        raise ValueError(f"{boundary} must be a lowercase SHA-256 digest.")
    return value


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    boundary: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{boundary} schema drifted; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}."
        )
    return value


def _require_string(value: object, *, boundary: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{boundary} must be a string.")
    return value


def _repository_root() -> Path | None:
    spec = importlib.util.find_spec(PRIMARY_QRF_WORKER_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"Cannot resolve primary-QRF worker module {PRIMARY_QRF_WORKER_MODULE!r}."
        )
    worker_source = Path(spec.origin).resolve()
    if not worker_source.is_file():
        raise RuntimeError(
            f"Primary-QRF worker source is not a readable file: {worker_source}."
        )
    candidates = [worker_source.parent, *worker_source.parents]
    for candidate in candidates:
        if (candidate / "uv.lock").is_file() and (candidate / "packages").is_dir():
            return candidate
    return None


def _approved_uv_lock_sha256() -> str:
    root = _repository_root()
    if root is None:
        return APPROVED_UV_LOCK_SHA256
    observed = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    if observed != APPROVED_UV_LOCK_SHA256:
        raise RuntimeError(
            "Primary-QRF worker identity found an unapproved uv.lock digest: "
            f"expected {APPROVED_UV_LOCK_SHA256}, got {observed}."
        )
    return observed


def _module_source_index() -> dict[str, Path]:
    namespace = importlib.util.find_spec("microcosm")
    if namespace is None or namespace.submodule_search_locations is None:
        raise RuntimeError("Cannot locate installed microcosm source packages.")
    # These are the namespace roots the running interpreter will actually use;
    # a convenient checkout found via CWD must never substitute different code.
    namespace_roots = sorted(
        {Path(location).resolve() for location in namespace.submodule_search_locations}
    )
    result: dict[str, Path] = {}
    for namespace_root in namespace_roots:
        for source in sorted(namespace_root.rglob("*.py")):
            relative = source.relative_to(namespace_root)
            parts = ["microcosm", *relative.with_suffix("").parts]
            if parts[-1] == "__init__":
                parts.pop()
            if not parts:
                continue
            module_name = ".".join(parts)
            previous = result.setdefault(module_name, source)
            if previous != source:
                raise RuntimeError(
                    f"Duplicate source modules for worker identity: {module_name!r}."
                )
    return result


def _source_imports(
    module_name: str,
    source: Path,
    raw: bytes,
    *,
    index: Mapping[str, Path],
) -> tuple[set[str], set[str]]:
    try:
        tree = ast.parse(raw, filename=str(source))
    except SyntaxError as error:
        raise RuntimeError(
            f"Cannot parse worker dependency source {module_name!r}."
        ) from error
    package = (
        module_name if source.name == "__init__.py" else module_name.rpartition(".")[0]
    )
    internal: set[str] = set()
    external_roots: set[str] = set()
    for node in ast.walk(tree):
        imported: list[str]
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = "." * node.level + (node.module or "")
                try:
                    base = importlib.util.resolve_name(relative, package)
                except (ImportError, ValueError) as error:
                    raise RuntimeError(
                        f"Cannot resolve worker dependency import {relative!r} in "
                        f"{module_name!r}."
                    ) from error
            else:
                base = node.module or ""
            imported = [base]
            imported.extend(
                f"{base}.{alias.name}" if base else alias.name for alias in node.names
            )
        else:
            continue
        for name in imported:
            if name in index:
                internal.add(name)
            elif name.startswith("microcosm"):
                prefixes = name.split(".")
                for length in range(len(prefixes), 0, -1):
                    candidate = ".".join(prefixes[:length])
                    if candidate in index:
                        internal.add(candidate)
                        break
            else:
                root = name.partition(".")[0]
                if root and root not in sys.stdlib_module_names:
                    external_roots.add(root)
    return internal, external_roots


def _with_package_initializers(
    module_names: set[str],
    *,
    index: Mapping[str, Path],
) -> set[str]:
    """Include package initializers Python executes before imported modules."""

    result = set(module_names)
    for module_name in tuple(module_names):
        parts = module_name.split(".")
        for length in range(1, len(parts)):
            package = ".".join(parts[:length])
            source = index.get(package)
            if source is not None and source.name == "__init__.py":
                result.add(package)
    return result


def _worker_package_resource_rows() -> list[dict[str, str]]:
    """Hash package data executed while importing the worker's package graph."""

    rows: list[dict[str, str]] = []
    for package, relative_path in _TRANSITIVE_PACKAGE_RESOURCES:
        spec = importlib.util.find_spec(package)
        locations = (
            tuple(spec.submodule_search_locations or ()) if spec is not None else ()
        )
        candidates = [
            Path(location).resolve() / relative_path
            for location in locations
            if (Path(location).resolve() / relative_path).is_file()
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Primary-QRF worker package resource resolution is ambiguous: "
                f"{package}:{relative_path}."
            )
        try:
            raw = candidates[0].read_bytes()
        except OSError as error:
            raise RuntimeError(
                "Cannot read primary-QRF worker package resource "
                f"{package}:{relative_path}."
            ) from error
        rows.append(
            {
                "resource": f"{package}:{relative_path}",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return rows


def _worker_source_identity() -> tuple[str, str, tuple[str, ...]]:
    index = _module_source_index()
    try:
        worker_source = index[PRIMARY_QRF_WORKER_MODULE]
    except KeyError as error:
        raise RuntimeError(
            f"Cannot locate primary-QRF worker module {PRIMARY_QRF_WORKER_MODULE!r}."
        ) from error
    try:
        worker_raw = worker_source.read_bytes()
    except OSError as error:
        raise RuntimeError("Cannot read primary-QRF worker source.") from error
    worker_sha256 = hashlib.sha256(worker_raw).hexdigest()
    pending, external_roots = _source_imports(
        PRIMARY_QRF_WORKER_MODULE,
        worker_source,
        worker_raw,
        index=index,
    )
    pending = _with_package_initializers(
        pending | {PRIMARY_QRF_WORKER_MODULE},
        index=index,
    )
    visited: set[str] = set()
    rows: list[dict[str, str]] = []
    while pending:
        module_name = min(pending)
        pending.remove(module_name)
        if module_name in visited or module_name == PRIMARY_QRF_WORKER_MODULE:
            continue
        visited.add(module_name)
        source = index[module_name]
        try:
            raw = source.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"Cannot read worker dependency source {module_name!r}."
            ) from error
        rows.append(
            {
                "module": module_name,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        imported, external = _source_imports(
            module_name,
            source,
            raw,
            index=index,
        )
        pending.update(_with_package_initializers(imported, index=index) - visited)
        external_roots.update(external)
    rows.extend(_worker_package_resource_rows())
    return worker_sha256, _canonical_sha256(rows), tuple(sorted(external_roots))


def _portable_record_path(path: PurePosixPath) -> bool:
    if path.is_absolute() or ".." in path.parts:
        return False
    if path.suffix in {".pth", ".pyc"} or "__pycache__" in path.parts:
        return False
    return not any(part.endswith((".dist-info", ".egg-info")) for part in path.parts)


def _record_hash_hex(value: str, *, boundary: str) -> str:
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"{boundary} has malformed base64url content.") from error
    if len(decoded) != hashlib.sha256().digest_size:
        raise RuntimeError(f"{boundary} is not a SHA-256 digest.")
    return decoded.hex()


def _installed_distributions_record_sha256(
    external_roots: Sequence[str],
) -> str:
    by_name: dict[str, metadata.Distribution] = {}
    package_to_distributions = metadata.packages_distributions()
    for distribution in metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if raw_name:
            by_name[canonicalize_name(raw_name)] = distribution

    pending: set[str] = set()
    unavailable_roots: list[str] = []
    resolved_roots: list[tuple[str, tuple[str, ...], tuple[Path, ...]]] = []
    for root in external_roots:
        names = package_to_distributions.get(root)
        spec = importlib.util.find_spec(root)
        if spec is None:
            unavailable_roots.append(root)
            continue
        if not names:
            raise RuntimeError(
                "Primary-QRF worker import resolves outside installed "
                f"distribution metadata: {root!r}."
            )
        locations: list[Path] = []
        if spec.origin is not None:
            locations.append(Path(spec.origin).resolve())
        if spec.submodule_search_locations is not None:
            locations.extend(
                Path(location).resolve() for location in spec.submodule_search_locations
            )
        if not locations:
            raise RuntimeError(
                f"Primary-QRF worker import {root!r} has no resolved location."
            )
        canonical_names = tuple(sorted(canonicalize_name(name) for name in names))
        resolved_roots.append((root, canonical_names, tuple(sorted(set(locations)))))
        pending.update(canonical_names)

    selected: set[str] = set()
    while pending:
        name = min(pending)
        pending.remove(name)
        if name in selected:
            continue
        try:
            distribution = by_name[name]
        except KeyError as error:
            raise RuntimeError(
                f"Primary-QRF worker dependency is not installed: {name}."
            ) from error
        selected.add(name)
        for raw_requirement in distribution.requires or ():
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement as error:
                raise RuntimeError(
                    f"Primary-QRF worker dependency {name!r} has malformed metadata."
                ) from error
            if requirement.marker is not None and not requirement.marker.evaluate(
                {"extra": ""}
            ):
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency not in selected:
                pending.add(dependency)

    distributions: list[dict[str, object]] = []
    portable_paths_by_distribution: dict[str, set[Path]] = {}
    for name in sorted(selected):
        distribution = by_name[name]
        files = distribution.files
        if files is None:
            raise RuntimeError(
                f"Primary-QRF worker distribution has no RECORD: {name}."
            )
        record_rows: list[dict[str, object]] = []
        portable_paths: set[Path] = set()
        for entry in files:
            path = PurePosixPath(str(entry))
            if not _portable_record_path(path):
                continue
            file_hash = entry.hash
            if file_hash is None or file_hash.mode != "sha256":
                raise RuntimeError(
                    f"Primary-QRF worker RECORD lacks SHA-256 for {name}:{path}."
                )
            expected_hex = _record_hash_hex(
                file_hash.value,
                boundary=f"Primary-QRF worker RECORD {name}:{path}",
            )
            installed_path = Path(entry.locate())
            try:
                raw = installed_path.read_bytes()
            except OSError as error:
                raise RuntimeError(
                    f"Primary-QRF worker cannot read installed file {name}:{path}."
                ) from error
            observed_hex = hashlib.sha256(raw).hexdigest()
            if observed_hex != expected_hex:
                raise RuntimeError(
                    f"Primary-QRF worker installed file differs from RECORD: "
                    f"{name}:{path}."
                )
            if entry.size is not None and len(raw) != entry.size:
                raise RuntimeError(
                    f"Primary-QRF worker installed file size differs from RECORD: "
                    f"{name}:{path}."
                )
            portable_paths.add(installed_path.resolve())
            record_rows.append(
                {
                    "path": path.as_posix(),
                    "sha256": expected_hex,
                    "size": entry.size,
                }
            )
        portable_paths_by_distribution[name] = portable_paths
        distributions.append(
            {
                "name": name,
                "version": distribution.version,
                "record_rows": record_rows,
            }
        )
    for root, names, locations in resolved_roots:
        recorded_paths = set().union(
            *(portable_paths_by_distribution.get(name, set()) for name in names)
        )
        for location in locations:
            if location.is_dir():
                is_recorded = any(
                    path.is_relative_to(location) for path in recorded_paths
                )
            else:
                is_recorded = location in recorded_paths
            if not is_recorded:
                raise RuntimeError(
                    "Primary-QRF worker import resolution is not authenticated "
                    f"by its installed RECORD: {root!r} at {location}."
                )
    return _canonical_sha256(
        {
            "distributions": distributions,
            "unavailable_import_roots": sorted(unavailable_roots),
        }
    )


def _canonical_pyvenv_config() -> dict[str, object]:
    path = Path(sys.prefix) / "pyvenv.cfg"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(
            f"Primary-QRF worker cannot read semantic virtualenv config {path}."
        ) from error
    parsed: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        normalized_key = key.strip().lower().replace("_", "-")
        if not separator or not normalized_key or normalized_key in parsed:
            raise RuntimeError(
                f"Primary-QRF worker has malformed pyvenv.cfg line {line_number}."
            )
        parsed[normalized_key] = value.strip()
    implementation = parsed.get("implementation")
    version_text = parsed.get("version-info", parsed.get("version"))
    include_system = parsed.get("include-system-site-packages")
    uv_version = parsed.get("uv")
    if not all(
        isinstance(value, str) and value
        for value in (implementation, version_text, uv_version)
    ):
        raise RuntimeError("Primary-QRF worker pyvenv.cfg lacks semantic uv fields.")
    assert version_text is not None
    try:
        version = [int(part) for part in version_text.split(".")]
    except ValueError as error:
        raise RuntimeError(
            "Primary-QRF worker pyvenv.cfg version is not numeric."
        ) from error
    if len(version) != 3:
        raise RuntimeError("Primary-QRF worker pyvenv.cfg version must be a triplet.")
    if include_system not in {"true", "false"}:
        raise RuntimeError(
            "Primary-QRF worker pyvenv.cfg has invalid include-system-site-packages."
        )
    return {
        "implementation": implementation.casefold(),
        "version": version,
        "include_system_site_packages": include_system == "true",
        "uv_version": uv_version,
    }


def _semantic_environment() -> dict[str, object]:
    fit_jobs_raw = os.environ.get("POPULACE_FIT_N_JOBS")
    if fit_jobs_raw is None:
        fit_jobs = -1
    else:
        try:
            fit_jobs = int(fit_jobs_raw)
        except ValueError as error:
            raise ValueError(
                "POPULACE_FIT_N_JOBS must be a positive integer for the "
                "primary-QRF worker binding."
            ) from error
        if fit_jobs < 1 or str(fit_jobs) != fit_jobs_raw:
            raise ValueError(
                "POPULACE_FIT_N_JOBS must be a canonical positive integer for "
                "the primary-QRF worker binding."
            )
    predict_workers_raw = os.environ.get("POPULACE_FIT_PREDICT_WORKERS")
    if predict_workers_raw is None or not predict_workers_raw.strip():
        predict_workers = os.cpu_count() or 1
        predict_workers_source = "os_cpu_count_fallback"
    else:
        try:
            predict_workers = int(predict_workers_raw)
        except ValueError as error:
            raise ValueError(
                "POPULACE_FIT_PREDICT_WORKERS must be a positive integer for the "
                "primary-QRF worker binding."
            ) from error
        if predict_workers < 1 or str(predict_workers) != predict_workers_raw:
            raise ValueError(
                "POPULACE_FIT_PREDICT_WORKERS must be a canonical positive "
                "integer for the primary-QRF worker binding."
            )
        predict_workers_source = "environment_override"
    return {
        "policy": "inherit_parent_environment_with_bound_fit_controls",
        "overrides": {},
        "semantic_controls": {
            "POPULACE_FIT_N_JOBS": {
                "configured": fit_jobs_raw,
                "resolved": fit_jobs,
            },
            "POPULACE_FIT_PREDICT_WORKERS": {
                "configured": predict_workers_raw,
                "resolved": predict_workers,
                "resolution": predict_workers_source,
            },
        },
        "bound_names": list(_SEMANTIC_ENVIRONMENT_NAMES),
    }


def primary_qrf_worker_semantic_identity(
    *,
    uv_lock_sha256: str | None = None,
) -> dict[str, object]:
    """Build the canonical worker identity, excluding all launcher aliases."""

    if uv_lock_sha256 is None:
        lock_sha256 = _approved_uv_lock_sha256()
    else:
        lock_sha256 = _require_sha256(
            uv_lock_sha256, boundary="primary-QRF worker lock"
        )
        if lock_sha256 not in {
            APPROVED_UV_LOCK_SHA256,
            LEGACY_CAMPAIGN_UV_LOCK_SHA256,
        }:
            raise ValueError("Primary-QRF worker lock is not approved.")
    executable = Path(sys.executable)
    module_sha256, imports_sha256, external_roots = _worker_source_identity()
    record_sha256 = _installed_distributions_record_sha256(external_roots)
    environment_code_sha256 = _canonical_sha256(
        {
            "worker_module_source_sha256": module_sha256,
            "transitive_imports_sha256": imports_sha256,
            "installed_distributions_record_sha256": record_sha256,
        }
    )
    return {
        "interpreter": {
            "bytes_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
            "abi": {
                "soabi": sysconfig.get_config_var("SOABI") or "",
                "abiflags": sys.abiflags,
            },
            "cache_tag": sys.implementation.cache_tag,
            "pyvenv_cfg": _canonical_pyvenv_config(),
        },
        "worker_module": {
            "name": PRIMARY_QRF_WORKER_MODULE,
            "source_sha256": module_sha256,
            "transitive_imports_sha256": imports_sha256,
        },
        "uv_lock_sha256": lock_sha256,
        "installed_distributions_record_sha256": record_sha256,
        "transitive_environment_code_sha256": environment_code_sha256,
        "argv_template": [
            PRIMARY_QRF_INTERPRETER_PLACEHOLDER,
            "-m",
            PRIMARY_QRF_WORKER_MODULE,
            "--checkpoint-dir",
            "{checkpoint_dir}",
            "--target-index",
            "{target_index}",
        ],
        "environment": _semantic_environment(),
    }


def primary_qrf_worker_execution_binding() -> dict[str, object]:
    """Return semantic worker identity plus integrity-bound audit aliases."""

    raw_executable = sys.executable
    executable = Path(raw_executable).absolute()
    semantic_identity = primary_qrf_worker_semantic_identity()
    result = {
        "schema_version": PRIMARY_QRF_WORKER_IDENTITY_SCHEMA_VERSION,
        "semantic_identity": semantic_identity,
        "semantic_identity_sha256": _canonical_sha256(semantic_identity),
        "audit_aliases": {
            "sys_executable": str(executable),
            "sys_prefix": str(Path(sys.prefix).absolute()),
            "argv_template_0": raw_executable,
        },
    }
    validate_primary_qrf_worker_execution_binding(
        result,
        boundary="primary-QRF worker identity construction",
    )
    return result


def _validate_semantic_identity(
    semantic: object, *, boundary: str
) -> Mapping[str, object]:
    value = _require_exact_keys(
        semantic,
        {
            "interpreter",
            "worker_module",
            "uv_lock_sha256",
            "installed_distributions_record_sha256",
            "transitive_environment_code_sha256",
            "argv_template",
            "environment",
        },
        boundary=boundary,
    )
    for key in (
        "uv_lock_sha256",
        "installed_distributions_record_sha256",
        "transitive_environment_code_sha256",
    ):
        _require_sha256(value.get(key), boundary=f"{boundary} {key}")
    worker_module = _require_exact_keys(
        value.get("worker_module"),
        {"name", "source_sha256", "transitive_imports_sha256"},
        boundary=f"{boundary} worker module",
    )
    if worker_module.get("name") != PRIMARY_QRF_WORKER_MODULE:
        raise ValueError(f"{boundary} worker module name changed.")
    _require_sha256(
        worker_module.get("source_sha256"), boundary=f"{boundary} worker source"
    )
    _require_sha256(
        worker_module.get("transitive_imports_sha256"),
        boundary=f"{boundary} transitive imports",
    )
    expected_environment_code = _canonical_sha256(
        {
            "worker_module_source_sha256": worker_module["source_sha256"],
            "transitive_imports_sha256": worker_module["transitive_imports_sha256"],
            "installed_distributions_record_sha256": value[
                "installed_distributions_record_sha256"
            ],
        }
    )
    if value.get("transitive_environment_code_sha256") != expected_environment_code:
        raise ValueError(f"{boundary} transitive environment/code digest changed.")
    interpreter = _require_exact_keys(
        value.get("interpreter"),
        {"bytes_sha256", "implementation", "version", "abi", "cache_tag", "pyvenv_cfg"},
        boundary=f"{boundary} interpreter",
    )
    _require_sha256(
        interpreter.get("bytes_sha256"), boundary=f"{boundary} interpreter bytes"
    )
    _require_string(
        interpreter.get("implementation"), boundary=f"{boundary} implementation"
    )
    _require_string(interpreter.get("cache_tag"), boundary=f"{boundary} cache tag")
    version = interpreter.get("version")
    if (
        not isinstance(version, list)
        or len(version) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in version
        )
    ):
        raise ValueError(f"{boundary} interpreter version must be an integer triplet.")
    abi = _require_exact_keys(
        interpreter.get("abi"), {"soabi", "abiflags"}, boundary=f"{boundary} ABI"
    )
    _require_string(abi.get("soabi"), boundary=f"{boundary} SOABI")
    _require_string(
        abi.get("abiflags"), boundary=f"{boundary} ABI flags", allow_empty=True
    )
    pyvenv = _require_exact_keys(
        interpreter.get("pyvenv_cfg"),
        {"implementation", "version", "include_system_site_packages", "uv_version"},
        boundary=f"{boundary} pyvenv.cfg",
    )
    _require_string(
        pyvenv.get("implementation"), boundary=f"{boundary} pyvenv implementation"
    )
    _require_string(pyvenv.get("uv_version"), boundary=f"{boundary} pyvenv uv version")
    pyvenv_version = pyvenv.get("version")
    if (
        not isinstance(pyvenv_version, list)
        or len(pyvenv_version) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in pyvenv_version
        )
        or type(pyvenv.get("include_system_site_packages")) is not bool
    ):
        raise ValueError(f"{boundary} pyvenv.cfg semantic fields are malformed.")
    expected_argv = [
        PRIMARY_QRF_INTERPRETER_PLACEHOLDER,
        "-m",
        PRIMARY_QRF_WORKER_MODULE,
        "--checkpoint-dir",
        "{checkpoint_dir}",
        "--target-index",
        "{target_index}",
    ]
    if value.get("argv_template") != expected_argv:
        raise ValueError(f"{boundary} semantic argv template changed.")
    environment = _require_exact_keys(
        value.get("environment"),
        {"policy", "overrides", "semantic_controls", "bound_names"},
        boundary=f"{boundary} environment",
    )
    if (
        environment.get("policy")
        != "inherit_parent_environment_with_bound_fit_controls"
        or environment.get("overrides") != {}
        or environment.get("bound_names") != list(_SEMANTIC_ENVIRONMENT_NAMES)
    ):
        raise ValueError(f"{boundary} semantic environment policy changed.")
    controls = _require_exact_keys(
        environment.get("semantic_controls"),
        set(_SEMANTIC_ENVIRONMENT_NAMES),
        boundary=f"{boundary} semantic controls",
    )
    fit = _require_exact_keys(
        controls.get("POPULACE_FIT_N_JOBS"),
        {"configured", "resolved"},
        boundary=f"{boundary} fit jobs",
    )
    predict = _require_exact_keys(
        controls.get("POPULACE_FIT_PREDICT_WORKERS"),
        {"configured", "resolved", "resolution"},
        boundary=f"{boundary} predict workers",
    )
    if fit.get("configured") is not None and not isinstance(fit.get("configured"), str):
        raise ValueError(f"{boundary} configured fit jobs must be a string or null.")
    if predict.get("configured") is not None and not isinstance(
        predict.get("configured"), str
    ):
        raise ValueError(
            f"{boundary} configured predict workers must be a string or null."
        )
    if (
        isinstance(fit.get("resolved"), bool)
        or not isinstance(fit.get("resolved"), int)
        or int(fit["resolved"]) < -1
        or fit.get("resolved") == 0
        or isinstance(predict.get("resolved"), bool)
        or not isinstance(predict.get("resolved"), int)
        or int(predict["resolved"]) < 1
        or predict.get("resolution")
        not in {"environment_override", "os_cpu_count_fallback"}
    ):
        raise ValueError(f"{boundary} resolved worker controls are malformed.")
    return value


def validate_primary_qrf_worker_execution_binding(
    binding: object,
    *,
    boundary: str,
) -> None:
    """Validate the closed v1 worker identity and its semantic self-digest."""

    worker = _require_exact_keys(
        binding,
        {
            "schema_version",
            "semantic_identity",
            "semantic_identity_sha256",
            "audit_aliases",
        },
        boundary=boundary,
    )
    if worker.get("schema_version") != PRIMARY_QRF_WORKER_IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            f"{boundary} requires schema version "
            f"{PRIMARY_QRF_WORKER_IDENTITY_SCHEMA_VERSION}."
        )
    semantic = _validate_semantic_identity(
        worker.get("semantic_identity"), boundary=f"{boundary} semantic_identity"
    )
    semantic_sha256 = _require_sha256(
        worker.get("semantic_identity_sha256"),
        boundary=f"{boundary} semantic_identity_sha256",
    )
    if semantic_sha256 != _canonical_sha256(semantic):
        raise ValueError(f"{boundary} semantic identity SHA-256 mismatch.")
    aliases = _require_exact_keys(
        worker.get("audit_aliases"),
        {"sys_executable", "sys_prefix", "argv_template_0"},
        boundary=f"{boundary} audit aliases",
    )
    for name, value in aliases.items():
        _require_string(value, boundary=f"{boundary} audit alias {name}")
        if name in {"sys_executable", "sys_prefix"} and not Path(value).is_absolute():
            raise ValueError(f"{boundary} audit alias {name} must be absolute.")


def _validated_primary_qrf_worker_semantic_identity(
    binding: object,
    *,
    boundary: str,
) -> Mapping[str, object]:
    validate_primary_qrf_worker_execution_binding(binding, boundary=boundary)
    assert isinstance(binding, Mapping)
    semantic = binding["semantic_identity"]
    assert isinstance(semantic, Mapping)
    return semantic


def primary_qrf_worker_semantic_projection(
    binding: object,
    *,
    boundary: str,
) -> dict[str, object]:
    """Project a binding for identities that must never hash audit aliases."""

    semantic = _validated_primary_qrf_worker_semantic_identity(
        binding, boundary=boundary
    )
    assert isinstance(binding, Mapping)
    return {
        "schema_version": binding["schema_version"],
        "semantic_identity": _json_clone(semantic),
        "semantic_identity_sha256": binding["semantic_identity_sha256"],
    }


def primary_qrf_worker_bindings_semantically_equal(
    left: object,
    right: object,
    *,
    boundary: str,
) -> bool:
    """Compare validated semantic identities while ignoring launcher aliases."""

    return _validated_primary_qrf_worker_semantic_identity(
        left, boundary=f"{boundary} recorded worker"
    ) == _validated_primary_qrf_worker_semantic_identity(
        right, boundary=f"{boundary} live worker"
    )


def legacy_primary_qrf_worker_execution_binding(
    *,
    semantic_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Project the live worker into the frozen schema-4 execution shape."""

    semantic = (
        primary_qrf_worker_semantic_identity(
            uv_lock_sha256=LEGACY_CAMPAIGN_UV_LOCK_SHA256
        )
        if semantic_identity is None
        else semantic_identity
    )
    _validate_semantic_identity(
        semantic, boundary="legacy live worker semantic identity"
    )
    executable = Path(sys.executable)
    worker_module = semantic["worker_module"]
    argv = semantic["argv_template"]
    assert isinstance(worker_module, Mapping)
    assert isinstance(argv, list)
    return {
        "module": worker_module["name"],
        "argv_template": [str(executable), *argv[1:]],
        "interpreter": {
            "executable": str(executable),
            "resolved_executable": str(executable.resolve()),
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": list(sys.version_info[:3]),
        },
        "environment": _json_clone(semantic["environment"]),
    }


def _legacy_worker_execution_mismatch_paths(
    recorded: object,
    live: object,
) -> tuple[str, ...]:
    """Return deterministic leaf paths that differ in two legacy bindings."""

    mismatches: list[str] = []

    def visit(left: object, right: object, path: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right), key=str):
                child = f"{path}.{key}" if path else str(key)
                if key not in left or key not in right:
                    mismatches.append(child)
                else:
                    visit(left[key], right[key], child)
            return
        if (
            isinstance(left, Sequence)
            and not isinstance(left, (str, bytes))
            and isinstance(right, Sequence)
            and not isinstance(right, (str, bytes))
        ):
            for index in range(max(len(left), len(right))):
                child = f"{path}[{index}]"
                if index >= len(left) or index >= len(right):
                    mismatches.append(child)
                else:
                    visit(left[index], right[index], child)
            return
        if left != right:
            mismatches.append(path or "$")

    visit(recorded, live, "")
    return tuple(mismatches)


def authenticate_legacy_worker_identity_attestation(
    path: str | Path,
    *,
    sealed_manifest_sha256: str,
    sealed_pool_h5_sha256: str,
    recorded_worker_execution: object,
    boundary: str,
) -> LegacyWorkerIdentityAuthentication:
    """Authenticate the explicit plan-gated schema-9 scoring exception."""

    attestation_path = Path(path)
    try:
        raw = attestation_path.read_bytes()
        parsed = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{boundary}: worker identity attestation is unreadable."
        ) from error
    attestation = _require_exact_keys(parsed, _ATTESTATION_KEYS, boundary=boundary)
    if (
        attestation.get("artifact_kind") != LEGACY_WORKER_ATTESTATION_ARTIFACT_KIND
        or attestation.get("schema_version") != LEGACY_WORKER_ATTESTATION_SCHEMA_VERSION
        or attestation.get("purpose") != "scoring_only"
        or attestation.get("plan_signature") != _PLAN_SIGNATURE
    ):
        raise ValueError(f"{boundary}: worker identity attestation authority changed.")
    manifest_sha256 = _require_sha256(
        attestation.get("sealed_manifest_sha256"),
        boundary=f"{boundary} sealed manifest",
    )
    pool_sha256 = _require_sha256(
        attestation.get("sealed_pool_h5_sha256"),
        boundary=f"{boundary} sealed pool H5",
    )
    if manifest_sha256 != sealed_manifest_sha256:
        raise ValueError(
            f"{boundary}: worker identity attestation sealed manifest changed."
        )
    if pool_sha256 != sealed_pool_h5_sha256:
        raise ValueError(f"{boundary}: worker identity attestation sealed H5 changed.")
    campaign_tree_sha = attestation.get("campaign_tree_sha")
    if campaign_tree_sha != LEGACY_CAMPAIGN_TREE_SHA:
        raise ValueError(f"{boundary}: worker identity attestation campaign changed.")
    assert isinstance(campaign_tree_sha, str)
    if attestation.get("uv_lock_sha256") != LEGACY_CAMPAIGN_UV_LOCK_SHA256:
        raise ValueError(f"{boundary}: worker identity attestation lock changed.")
    if attestation.get("recorded_worker_execution") != recorded_worker_execution:
        raise ValueError(f"{boundary}: attested recorded worker changed.")
    if attestation.get("permitted_mismatches") != list(
        LEGACY_WORKER_PERMITTED_MISMATCHES
    ):
        raise ValueError(f"{boundary}: permitted worker mismatches changed.")
    semantic = _validate_semantic_identity(
        attestation.get("semantic_identity"),
        boundary=f"{boundary} attested semantic identity",
    )
    semantic_sha256 = _require_sha256(
        attestation.get("semantic_identity_sha256"),
        boundary=f"{boundary} semantic identity",
    )
    if semantic_sha256 != _canonical_sha256(semantic):
        raise ValueError(f"{boundary}: attested semantic identity digest changed.")
    if semantic.get(
        "uv_lock_sha256"
    ) != LEGACY_CAMPAIGN_UV_LOCK_SHA256 or attestation.get(
        "installed_transitive_environment_code_sha256"
    ) != semantic.get("transitive_environment_code_sha256"):
        raise ValueError(f"{boundary}: attested environment/code identity changed.")
    expected_semantic = primary_qrf_worker_semantic_identity(
        uv_lock_sha256=LEGACY_CAMPAIGN_UV_LOCK_SHA256
    )
    if semantic != expected_semantic:
        raise ValueError(f"{boundary}: semantic worker identity changed.")
    live_worker = legacy_primary_qrf_worker_execution_binding(
        semantic_identity=expected_semantic
    )
    if (
        _legacy_worker_execution_mismatch_paths(recorded_worker_execution, live_worker)
        != LEGACY_WORKER_PERMITTED_MISMATCHES
    ):
        raise ValueError(f"{boundary}: legacy worker mismatch set changed.")
    recorded = _json_clone(recorded_worker_execution)
    semantic_copy = _json_clone(semantic)
    assert isinstance(recorded, Mapping)
    assert isinstance(semantic_copy, Mapping)
    return LegacyWorkerIdentityAuthentication(
        attestation_sha256=hashlib.sha256(raw).hexdigest(),
        campaign_tree_sha=campaign_tree_sha,
        recorded_worker_execution=recorded,
        semantic_identity=semantic_copy,
        semantic_identity_sha256=semantic_sha256,
    )


def current_worker_execution_authentication_receipt(
    binding: object,
    *,
    manifest_schema_version: int,
    execution_config_schema_version: int,
    boundary: str,
) -> dict[str, object]:
    """Describe the validated current worker identity without trusting aliases."""

    validate_primary_qrf_worker_execution_binding(binding, boundary=boundary)
    assert isinstance(binding, Mapping)
    aliases = binding["audit_aliases"]
    assert isinstance(aliases, Mapping)
    return {
        "manifest_schema_version": manifest_schema_version,
        "execution_config_schema_version": execution_config_schema_version,
        "worker_execution_schema_version": binding["schema_version"],
        "semantic_identity_sha256": binding["semantic_identity_sha256"],
        "audit_aliases": _json_clone(aliases),
    }


__all__ = [
    "APPROVED_UV_LOCK_SHA256",
    "LEGACY_CAMPAIGN_TREE_SHA",
    "LEGACY_CAMPAIGN_UV_LOCK_SHA256",
    "LEGACY_WORKER_PERMITTED_MISMATCHES",
    "LegacyWorkerIdentityAuthentication",
    "PRIMARY_QRF_INTERPRETER_PLACEHOLDER",
    "PRIMARY_QRF_WORKER_IDENTITY_SCHEMA_VERSION",
    "PRIMARY_QRF_WORKER_MODULE",
    "_legacy_worker_execution_mismatch_paths",
    "authenticate_legacy_worker_identity_attestation",
    "current_worker_execution_authentication_receipt",
    "legacy_primary_qrf_worker_execution_binding",
    "primary_qrf_worker_bindings_semantically_equal",
    "primary_qrf_worker_execution_binding",
    "primary_qrf_worker_semantic_identity",
    "primary_qrf_worker_semantic_projection",
    "validate_primary_qrf_worker_execution_binding",
]
