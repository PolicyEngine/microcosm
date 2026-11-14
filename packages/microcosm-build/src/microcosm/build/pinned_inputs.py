"""Raw inputs checked against their Chronicle registrations.

Chronicle registers a raw-only input with a manifest that pins each file's
bytes (sha256 and size) without carrying them. A builder that reads such a
file checks its local copy against the pin first, then records what it
verified, so a committed resource names the exact bytes it was built from.

Nothing here is country-specific: the manifest names the publisher, the
licence and the files. The functions moved here from the UK high-value home
stock builder (microcosm#1524), which keeps its old private names as aliases.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import yaml

__all__ = [
    "LICENCE_FIELDS",
    "licence_of",
    "manifest_files",
    "registration",
    "sha256_of_file",
    "verify_pinned_file",
]

#: The licence fields a Chronicle manifest states, in the order a methods
#: note quotes them.
LICENCE_FIELDS: tuple[str, ...] = (
    "license",
    "license_url",
    "attribution",
    "license_evidence",
    "license_verified_at",
)


def sha256_of_file(path: Path | str) -> str:
    """The sha256 of a file's bytes, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_files(manifest_path: Path) -> tuple[dict, dict[str, dict]]:
    """A Chronicle manifest and its file entries, keyed by text.

    YAML reads a year key such as ``2026`` as an integer; the keys are
    returned as text so that every caller sees one spelling.
    """
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{manifest_path} declares no files.")
    return manifest, {str(key): dict(value) for key, value in files.items()}


def verify_pinned_file(
    path: Path, entry: Mapping[str, object], *, skip: bool = False
) -> dict[str, object]:
    """Check a local file against its manifest entry and describe the pin.

    The file must be a regular file, not a link. Its size is always checked;
    its sha256 is checked unless ``skip`` (development only). The result is
    the record a builder stores: file name, sha256, size, the publisher's
    URL and the R2 key.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path} must be a regular local file.")
    size = path.stat().st_size
    if size != int(entry["size_bytes"]):
        raise ValueError(
            f"{path}: size {size} differs from the registered {entry['size_bytes']}."
        )
    if not skip:
        digest = sha256_of_file(path)
        if digest != str(entry["sha256"]):
            raise ValueError(f"{path}: sha256 differs from the registered pin.")
    return {
        "filename": str(entry["filename"]),
        "sha256": str(entry["sha256"]),
        "size_bytes": int(entry["size_bytes"]),
        "source_url": str(entry.get("source_url", "")),
        "r2_key": str(((entry.get("storage") or {}).get("r2") or {}).get("key", "")),
    }


def registration(
    manifest: Mapping[str, object], files: Mapping[str, Mapping[str, object]]
) -> dict:
    """The registration record of a package: its ids and verified files."""
    return {
        "chronicle_source_id": manifest.get("source_id"),
        "chronicle_package_id": manifest.get("package_id"),
        "registration_kind": manifest.get("registration_kind", "committed"),
        "files": {key: files[key] for key in sorted(files)},
    }


def licence_of(
    manifest: Mapping[str, object], *, declared: Mapping[str, object] | None = None
) -> dict[str, str]:
    """The licence a manifest states for its package, as the publisher words it.

    Returns the package id and each of :data:`LICENCE_FIELDS` the manifest
    carries. A methods note builds its attribution text from these fields,
    so it never restates a licence from memory.

    Some older manifests state no licence fields at all. A builder may then
    pass ``declared``, the licence fields it records for that package itself.
    ``license_source`` says which of the two the record quotes, so a reader
    can tell a manifest's words from a builder's. A manifest that states some
    licence fields but no licence is refused, since mixing the two would hide
    a manifest to fix.
    """
    if not manifest.get("package_id"):
        raise ValueError("The manifest names no package_id.")
    package_id = str(manifest["package_id"])
    out = {"package_id": package_id}
    for field in LICENCE_FIELDS:
        value = manifest.get(field)
        if value not in (None, ""):
            out[field] = str(value)
    if "license" in out:
        out["license_source"] = "manifest"
        return out
    if declared is None:
        raise ValueError(f"The manifest of {package_id} states no licence.")
    partial = sorted(set(out) - {"package_id"})
    if partial:
        raise ValueError(
            f"The manifest of {package_id} states {partial} but no licence."
        )
    for field in LICENCE_FIELDS:
        value = declared.get(field)
        if value not in (None, ""):
            out[field] = str(value)
    if "license" not in out:
        raise ValueError(f"The licence declared for {package_id} states no licence.")
    out["license_source"] = "declared"
    return out
