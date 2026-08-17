"""Country-bundle orchestration for the F0 compiler front end."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from importlib import resources as importlib_resources
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath

from .canonical import (
    CANONICALIZER_VERSION,
    canonical_json_bytes,
    documentation_envelope,
    normalize_and_project,
    sha256_json,
    spec_envelope,
)
from .errors import SpecValidationError
from .model import (
    DOMAIN_TYPE_BY_KIND,
    FileReceipt,
    FrozenMap,
    GrammarReceipt,
    MigrationReceipt,
    ResolvedResource,
    ResolvedSpec,
    ResourceDescriptor,
    ResourceKind,
    Surface,
    SurfaceObjects,
    freeze_json,
    thaw_json,
)
from .resolver import KernelRegistry, resolve_cross_references
from .schemas import (
    SCHEMA_FILENAMES,
    SchemaRegistry,
    assert_schema_id_allowed,
    load_schema_registry,
)
from .yaml12 import load_yaml12

SUPPORTED_SCHEMA_VERSION = 1
BUNDLE_LOCK_FILENAME = "bundle.lock.json"
PLAN_LOCK_FILENAME = "plan.lock.json"
ENGINE_ABI_LOCK_FILENAME = "engine_abi.lock.json"
GENERATED_LOCK_FILENAMES = frozenset(
    {BUNDLE_LOCK_FILENAME, PLAN_LOCK_FILENAME, ENGINE_ABI_LOCK_FILENAME}
)

_PATH_PATTERN = re.compile(r"^[a-z0-9_./-]+$")
_AUTHORED_KINDS = frozenset(
    kind
    for kind in ResourceKind
    if kind not in {ResourceKind.SCHEMA, ResourceKind.LEGACY_JSON}
)
_EMPTY_KERNEL_REGISTRY = KernelRegistry.from_ids(())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bytes(resource: Traversable | Path, *, label: str) -> bytes:
    try:
        return resource.read_bytes()
    except OSError as error:
        raise SpecValidationError(f"unable to read resource: {error}", source=label) from error


def _decode_utf8(raw: bytes, *, label: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SpecValidationError("resource is not UTF-8", source=label) from error


def _resource_child(root: Traversable | Path, relative: PurePosixPath) -> Traversable | Path:
    child: Traversable | Path = root
    for part in relative.parts:
        child = child.joinpath(part)
    return child


def _package_root(country_or_path: str | Path | Traversable) -> Traversable | Path:
    if isinstance(country_or_path, Traversable):
        return country_or_path
    candidate = Path(country_or_path)
    if candidate.exists():
        return candidate.resolve()
    if isinstance(country_or_path, str) and re.fullmatch(r"[a-z]{2}", country_or_path):
        root = importlib_resources.files("microcosm.build").joinpath(country_or_path)
        if root.is_dir():
            return root
    raise SpecValidationError(f"country bundle not found: {country_or_path}")


def _normalized_resource_path(value: object, *, row: int) -> PurePosixPath:
    location = f"country_package.json/resources/{row}/path"
    if not isinstance(value, str) or not value:
        raise SpecValidationError(f"{location}: non-empty string required")
    if not _PATH_PATTERN.fullmatch(value) or "\\" in value:
        raise SpecValidationError(f"{location}: invalid portable resource path {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SpecValidationError(f"{location}: path must be normalized and relative")
    if path.as_posix() != value:
        raise SpecValidationError(f"{location}: path is not normalized")
    if path.name in GENERATED_LOCK_FILENAMES:
        raise SpecValidationError(f"{location}: generated locks cannot be authored resources")
    return path


def _descriptors(manifest: Mapping[str, object]) -> tuple[ResourceDescriptor, ...]:
    rows = manifest.get("resources")
    if not isinstance(rows, list):
        raise SpecValidationError("country_package.json/resources: array required")
    descriptors: list[ResourceDescriptor] = []
    seen_paths: set[PurePosixPath] = set()
    seen_singletons: set[ResourceKind] = set()
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping):
            raise SpecValidationError(
                f"country_package.json/resources/{index}: typed row required"
            )
        path = _normalized_resource_path(value.get("path"), row=index)
        try:
            kind = ResourceKind(str(value.get("kind")))
        except ValueError as error:
            raise SpecValidationError(
                f"country_package.json/resources/{index}/kind: unknown kind"
            ) from error
        schema_id = value.get("schema_id")
        if not isinstance(schema_id, str):
            raise SpecValidationError(
                f"country_package.json/resources/{index}/schema_id: string required"
            )
        assert_schema_id_allowed(kind.value, schema_id)
        if path in seen_paths:
            raise SpecValidationError(
                f"country_package.json/resources/{index}/path: duplicate {path}"
            )
        if kind in _AUTHORED_KINDS and kind in seen_singletons:
            raise SpecValidationError(
                f"country_package.json/resources/{index}/kind: duplicate singleton "
                f"{kind.value!r}"
            )
        if kind in _AUTHORED_KINDS:
            seen_singletons.add(kind)
        expected_suffix = ".json" if kind in {ResourceKind.SCHEMA, ResourceKind.LEGACY_JSON} else ".yaml"
        if path.suffix not in ({".yaml", ".yml"} if expected_suffix == ".yaml" else {".json", ".jsonld"}):
            raise SpecValidationError(
                f"country_package.json/resources/{index}/path: {kind.value} "
                f"resource must use {expected_suffix}"
            )
        seen_paths.add(path)
        descriptors.append(ResourceDescriptor(path=path, kind=kind, schema_id=schema_id))
    if ResourceKind.BUNDLE not in seen_singletons:
        raise SpecValidationError("country_package.json: exactly one bundle resource is required")
    return tuple(sorted(descriptors, key=lambda row: (row.kind.value, row.path.as_posix())))


def _actual_files(root: Traversable | Path) -> frozenset[str]:
    if isinstance(root, Path):
        files: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise SpecValidationError(
                    f"country bundle contains symlink: {path.relative_to(root).as_posix()}"
                )
            if path.is_file():
                files.add(path.relative_to(root).as_posix())
        return frozenset(files)

    def visit(directory: Traversable, prefix: str = "") -> set[str]:
        result: set[str] = set()
        for child in directory.iterdir():
            relative = f"{prefix}/{child.name}" if prefix else child.name
            if child.is_dir():
                result.update(visit(child, relative))
            elif child.is_file():
                result.add(relative)
        return result

    return frozenset(visit(root))


def _assert_file_closure(
    root: Traversable | Path,
    descriptors: tuple[ResourceDescriptor, ...],
) -> None:
    declared = {row.path.as_posix() for row in descriptors} | {"country_package.json"}
    actual = set(_actual_files(root))
    # Generated locks may be present after a compile but never enter the
    # authored composition or their own file-hash closure.
    actual -= GENERATED_LOCK_FILENAMES
    missing = sorted(declared - actual)
    extra = sorted(actual - declared)
    if missing or extra:
        raise SpecValidationError(
            "country package file closure failed: "
            f"missing={missing!r}, undeclared={extra!r}"
        )


def _grammar_receipt(registry: SchemaRegistry, schema_version: int) -> GrammarReceipt:
    schema_set = {
        name: registry.schemas[name]
        for name in sorted(SCHEMA_FILENAMES)
    }
    schema_digest = _sha256_bytes(canonical_json_bytes(schema_set))
    normalizer_digest = _sha256_bytes(
        canonical_json_bytes(
            {
                "id": "schema-v1-defaults-surfaces-and-collections",
                "canonicalizer_version": CANONICALIZER_VERSION,
                "number": "finite-integral-floats-collapse",
                "unicode": "NFC",
                "arrays": "ordered-unless-x-canonical-order-set",
            }
        )
    )
    return GrammarReceipt(
        schema_version=schema_version,
        canonicalizer_version=CANONICALIZER_VERSION,
        migration_chain=(
            MigrationReceipt("approved-schema-catalog-v1", schema_digest),
            MigrationReceipt("resolved-normalization-v1", normalizer_digest),
        ),
    )


def load_bundle(
    country_or_path: str | Path | Traversable,
    *,
    kernel_registry: KernelRegistry | None = None,
    schema_registry: SchemaRegistry | None = None,
) -> ResolvedSpec:
    """Parse, validate, normalize, and resolve one typed country bundle."""

    root = _package_root(country_or_path)
    registry = schema_registry or load_schema_registry()
    manifest_resource = _resource_child(root, PurePosixPath("country_package.json"))
    manifest_raw = _read_bytes(manifest_resource, label="country_package.json")
    manifest_value = load_yaml12(
        _decode_utf8(manifest_raw, label="country_package.json"),
        source="country_package.json",
    )
    if not isinstance(manifest_value, Mapping):
        raise SpecValidationError("manifest root must be an object", source="country_package.json")
    registry.validate(manifest_value, "resource_manifest.schema.json")
    manifest = registry.validate_and_inject_defaults(
        manifest_value, "resource_manifest.schema.json"
    )
    if not isinstance(manifest, Mapping):  # pragma: no cover - schema guarantee
        raise SpecValidationError("manifest root must be an object")
    schema_version = manifest.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise SpecValidationError(
            "country_package.json/schema_version: unsupported grammar "
            f"{schema_version!r}; supported={SUPPORTED_SCHEMA_VERSION}"
        )
    country = manifest.get("country")
    if not isinstance(country, str):  # pragma: no cover - schema guarantee
        raise SpecValidationError("country_package.json/country: string required")
    descriptors = _descriptors(manifest)
    _assert_file_closure(root, descriptors)

    manifest_normalized, manifest_projections = normalize_and_project(
        manifest,
        schema_id="resource_manifest.schema.json",
        registry=registry,
    )
    file_receipt_values: dict[str, object] = {
        "country_package.json": FileReceipt(
            sha256=_sha256_bytes(manifest_raw), byte_size=len(manifest_raw)
        ).to_wire()
    }
    normalized_by_kind: dict[str, object] = {}
    resolved_resources: list[ResolvedResource] = []
    projections_by_surface: dict[Surface, dict[str, object]] = {
        surface: {} for surface in Surface
    }
    manifest_projection_map = thaw_json(manifest_projections)
    assert isinstance(manifest_projection_map, dict)
    for surface in Surface:
        fragment = manifest_projection_map.get(surface.value, {})
        if fragment != {}:
            projections_by_surface[surface]["country_package.json"] = fragment

    for descriptor in descriptors:
        path_text = descriptor.path.as_posix()
        resource = _resource_child(root, descriptor.path)
        if not resource.is_file():
            raise SpecValidationError("declared resource is not a regular file", source=path_text)
        raw = _read_bytes(resource, label=path_text)
        text = _decode_utf8(raw, label=path_text)
        parsed = load_yaml12(text, source=path_text)
        if descriptor.kind is ResourceKind.SCHEMA:
            # Schema resources are grammar inputs and validated when the
            # registry is built; a country bundle does not normally repeat
            # them, but the manifest kind remains closed and meaningful.
            if not isinstance(parsed, Mapping) or parsed.get("$id") != descriptor.schema_id:
                raise SpecValidationError("schema resource $id mismatch", source=path_text)
            normalized = freeze_json(parsed)
            projections = freeze_json(
                {surface.value: (parsed if surface is Surface.NORMATIVE else {}) for surface in Surface}
            )
        elif descriptor.kind is ResourceKind.LEGACY_JSON:
            # Generation-0 compatibility data is transport-visible but is not
            # silently promoted into generation-1 semantic authority.
            normalized = freeze_json(parsed)
            projections = freeze_json(
                {surface.value: {} for surface in Surface}
            )
        else:
            registry.validate(parsed, descriptor.schema_id)
            defaulted = registry.validate_and_inject_defaults(parsed, descriptor.schema_id)
            normalized, projections = normalize_and_project(
                defaulted,
                schema_id=descriptor.schema_id,
                registry=registry,
            )
            normalized_by_kind[descriptor.kind.value] = thaw_json(normalized)
        receipt = FileReceipt(sha256=_sha256_bytes(raw), byte_size=len(raw))
        file_receipt_values[path_text] = receipt.to_wire()
        domain_type = DOMAIN_TYPE_BY_KIND[descriptor.kind]
        domain = domain_type(descriptor=descriptor, value=normalized)
        resolved_resources.append(
            ResolvedResource(
                descriptor=descriptor,
                domain=domain,
                file_receipt=receipt,
                projections=projections if isinstance(projections, FrozenMap) else freeze_json(projections),  # type: ignore[arg-type]
            )
        )
        projection_wire = thaw_json(projections)  # type: ignore[arg-type]
        assert isinstance(projection_wire, dict)
        for surface in Surface:
            fragment = projection_wire.get(surface.value, {})
            if fragment != {}:
                projections_by_surface[surface][path_text] = fragment

    bundle_country = _mapping_country(normalized_by_kind.get("bundle"))
    if bundle_country != country:
        raise SpecValidationError(
            f"bundle country {bundle_country!r} does not match manifest {country!r}"
        )
    resolution = resolve_cross_references(
        normalized_by_kind,
        kernel_registry=kernel_registry or _EMPTY_KERNEL_REGISTRY,
    )
    grammar_receipt = _grammar_receipt(registry, schema_version)
    normative_files = projections_by_surface[Surface.NORMATIVE]
    documentation_files = projections_by_surface[Surface.DOCUMENTATION]
    spec_hash = sha256_json(
        spec_envelope(
            country=country,
            schema_version=schema_version,
            normative_files=normative_files,
        )
    )
    documentation_hash = sha256_json(
        documentation_envelope(
            country=country,
            schema_version=schema_version,
            documentation_files=documentation_files,
        )
    )
    file_receipts = freeze_json(file_receipt_values)
    assert isinstance(file_receipts, FrozenMap)
    package_fingerprint = sha256_json(
        {
            path: receipt["sha256"]
            for path, receipt in sorted(file_receipt_values.items())
        }
    )
    surface_objects = SurfaceObjects(
        **{
            surface.value: freeze_json(projections_by_surface[surface])
            for surface in Surface
        }
    )
    return ResolvedSpec(
        country=country,
        schema_version=schema_version,
        resources=tuple(resolved_resources),
        grammar_receipt=grammar_receipt,
        surfaces=surface_objects,
        entities=resolution.entities,
        artifacts=resolution.artifacts,
        scopes=resolution.scopes,
        columns=resolution.columns,
        references=resolution.references,
        file_receipts=file_receipts,
        package_fingerprint=package_fingerprint,
        spec_sha256=spec_hash,
        documentation_sha256=documentation_hash,
    )


def _mapping_country(value: object) -> str | None:
    if isinstance(value, Mapping):
        country = value.get("country")
        return country if isinstance(country, str) else None
    return None


def bundle_lock_payload(spec: ResolvedSpec) -> dict[str, object]:
    """Return the reproducible emitted bundle lock (never an authored input)."""

    return {
        "grammar_receipt": spec.grammar_receipt.to_wire(),
        "files": thaw_json(spec.file_receipts),
        "spec_sha256": spec.spec_sha256,
    }


def bundle_lock_bytes(spec: ResolvedSpec) -> bytes:
    return canonical_json_bytes(bundle_lock_payload(spec))


def emit_bundle_lock(spec: ResolvedSpec, path: str | Path) -> Path:
    destination = Path(path)
    destination.write_bytes(bundle_lock_bytes(spec))
    return destination


__all__ = [
    "BUNDLE_LOCK_FILENAME",
    "ENGINE_ABI_LOCK_FILENAME",
    "GENERATED_LOCK_FILENAMES",
    "PLAN_LOCK_FILENAME",
    "SUPPORTED_SCHEMA_VERSION",
    "bundle_lock_bytes",
    "bundle_lock_payload",
    "emit_bundle_lock",
    "load_bundle",
]

