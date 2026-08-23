"""Deterministic, lossless re-emission of a resolved country bundle.

Re-emission writes the normalized typed resources, never the authored source
text.  That makes defaults and schema-declared set ordering explicit while
preserving every physical surface carried by :class:`ResolvedSpec`.  The
result is a fresh single-authored bundle: generated locks are emitted outside
the typed manifest and no constants-era or migration generator is imported.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path, PurePosixPath

import yaml

from .canonical import canonical_json_bytes, sha256_json
from .engine_abi import ENGINE_ABI_LOCK_FILENAME
from .errors import SpecValidationError
from .loader import (
    BUNDLE_LOCK_FILENAME,
    PLAN_LOCK_FILENAME,
    bundle_lock_bytes,
)
from .model import FileReceipt, FrozenMap, ResolvedSpec, freeze_json, thaw_json
from .plan_lock import plan_lock_bytes
from .yaml12 import load_yaml12


class _AliasFreeSafeDumper(yaml.SafeDumper):
    """Safe YAML dumper that never represents shared values as aliases."""

    def ignore_aliases(self, data: object) -> bool:
        return True


def canonical_yaml_bytes(value: object, *, source: str = "<resolved>") -> bytes:
    """Serialize one normalized JSON value in the strict YAML 1.2 subset.

    PyYAML's serializer is used only as an encoder.  Its output is immediately
    parsed by the front end's strict loader and compared using canonical JSON
    bytes, so a future serializer change cannot silently alter scalar types or
    introduce a YAML extension that the compiler itself would reject.
    """

    try:
        expected = canonical_json_bytes(value)
        text = yaml.dump(
            value,
            Dumper=_AliasFreeSafeDumper,
            allow_unicode=True,
            default_flow_style=False,
            explicit_end=False,
            sort_keys=True,
            width=4096,
        )
        parsed = load_yaml12(text, source=source)
        observed = canonical_json_bytes(parsed)
    except (TypeError, ValueError, yaml.YAMLError) as error:
        raise SpecValidationError(
            f"unable to re-emit normalized YAML: {error}", source=source
        ) from error
    if observed != expected:
        raise SpecValidationError(
            "re-emitted YAML does not preserve the normalized JSON value",
            source=source,
        )
    return text.encode("utf-8")


def _manifest(spec: ResolvedSpec) -> dict[str, object]:
    return {
        "schema_version": spec.schema_version,
        "country": spec.country,
        "resources": [resource.descriptor.to_wire() for resource in spec.resources],
    }


def _confined_relative_parts(relative: str) -> tuple[str, ...]:
    """Return normalized path components or refuse an escaping output path."""

    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SpecValidationError(
            f"re-emitted resource path must be normalized and relative: {relative!r}"
        )
    return path.parts


def _authored_bundle_bytes(spec: ResolvedSpec) -> dict[str, bytes]:
    files = {"country_package.json": canonical_json_bytes(_manifest(spec)) + b"\n"}
    for resource in spec.resources:
        path = resource.descriptor.path.as_posix()
        _confined_relative_parts(path)
        if path in files:
            raise SpecValidationError(f"duplicate re-emitted resource path {path!r}")
        value = resource.domain.to_wire()
        if resource.descriptor.path.suffix in {".yaml", ".yml"}:
            raw = canonical_yaml_bytes(value, source=path)
        else:
            raw = canonical_json_bytes(value) + b"\n"
            # JSON is inside the strict YAML subset used at the loader boundary.
            parsed = load_yaml12(raw.decode("utf-8"), source=path)
            if canonical_json_bytes(parsed) != canonical_json_bytes(value):
                raise SpecValidationError(
                    "re-emitted JSON does not preserve the normalized value",
                    source=path,
                )
        files[path] = raw
    return files


def _spec_with_emitted_receipts(
    spec: ResolvedSpec,
    authored_files: dict[str, bytes],
) -> ResolvedSpec:
    receipts = {
        path: FileReceipt(
            sha256=hashlib.sha256(raw).hexdigest(),
            byte_size=len(raw),
        ).to_wire()
        for path, raw in sorted(authored_files.items())
    }
    receipt_map = freeze_json(receipts)
    if not isinstance(receipt_map, FrozenMap):  # pragma: no cover - construction
        raise TypeError("emitted file receipts must be an object")
    resource_receipts = {
        path: FileReceipt(
            sha256=str(receipt["sha256"]),
            byte_size=int(receipt["byte_size"]),
        )
        for path, receipt in receipts.items()
        if path != "country_package.json"
    }
    resources = tuple(
        replace(
            resource,
            file_receipt=resource_receipts[resource.descriptor.path.as_posix()],
        )
        for resource in spec.resources
    )
    package_fingerprint = sha256_json(
        {path: receipt["sha256"] for path, receipt in sorted(receipts.items())}
    )
    return replace(
        spec,
        resources=resources,
        file_receipts=receipt_map,
        package_fingerprint=package_fingerprint,
    )


def resolved_bundle_bytes(spec: ResolvedSpec) -> dict[str, bytes]:
    """Return every authored file and generated lock for a resolved bundle.

    ``bundle.lock.json`` is based on the newly encoded file receipts rather
    than the source bundle's transport receipt.  ``plan.lock.json`` is derived
    from the normalized compiler IR.  A generated engine ABI lock is copied
    from the already verified generated authority so a take-up bundle remains
    independently loadable.  None of the lock paths enter the typed manifest.
    """

    if not isinstance(spec, ResolvedSpec):
        raise TypeError("resolved_bundle_bytes requires a ResolvedSpec")
    authored = _authored_bundle_bytes(spec)
    emitted_spec = _spec_with_emitted_receipts(spec, authored)
    files = dict(authored)
    files[BUNDLE_LOCK_FILENAME] = bundle_lock_bytes(emitted_spec)
    files[PLAN_LOCK_FILENAME] = plan_lock_bytes(spec)

    generated = thaw_json(spec.generated_authorities)
    if not isinstance(generated, dict):  # pragma: no cover - model invariant
        raise TypeError("generated authorities must be an object")
    engine_abi_lock = generated.get("engine_abi_lock")
    if engine_abi_lock is not None:
        files[ENGINE_ABI_LOCK_FILENAME] = canonical_json_bytes(engine_abi_lock) + b"\n"
    return {path: files[path] for path in sorted(files)}


def emit_resolved_bundle(spec: ResolvedSpec, destination: str | Path) -> Path:
    """Write a deterministic resolved bundle into a new empty directory."""

    files = resolved_bundle_bytes(spec)
    root = Path(destination)
    if root.is_symlink():
        raise FileExistsError(f"refusing symlink destination: {root}")
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"bundle destination is not a directory: {root}")
        if any(root.iterdir()):
            raise FileExistsError(f"bundle destination is not empty: {root}")
    else:
        root.mkdir(parents=True)
    for relative, raw in files.items():
        target = root.joinpath(*_confined_relative_parts(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return root


__all__ = [
    "canonical_yaml_bytes",
    "emit_resolved_bundle",
    "resolved_bundle_bytes",
]
