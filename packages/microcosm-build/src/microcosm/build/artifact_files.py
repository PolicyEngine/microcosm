"""Atomic filesystem materialization of declared, portable build artifacts."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .trace import sha256_file


def file_artifact(path: str | Path) -> dict[str, object]:
    """Bind one regular file without loading its entire payload into memory."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Artifact is not a regular file: {source}.")
    before = source.stat()
    digest = sha256_file(source)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError(f"Artifact changed while binding its identity: {source}.")
    return {"filename": source.name, "sha256": digest, "size_bytes": after.st_size}


def materialize_bytes(payload: bytes, path: str | Path) -> dict[str, object]:
    """Write deterministic bytes atomically, including after a graph cache hit."""
    if not isinstance(payload, bytes):
        raise TypeError("Artifact payload must be immutable bytes.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return file_artifact(destination)


def validate_file_inventory(
    inventory: Mapping[str, Mapping[str, object]], *, root: str | Path
) -> None:
    """Check exact bytes and sizes for each named file in a bundle directory."""
    directory = Path(root).resolve()
    for role, expected in inventory.items():
        name = expected.get("filename")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
        ):
            raise ValueError(f"Artifact {role!r} has an invalid bundle filename.")
        path = directory / name
        if path.resolve().parent != directory:
            raise ValueError(
                f"Artifact {role!r} filename escapes its bundle directory."
            )
        if file_artifact(path) != dict(expected):
            raise ValueError(f"Artifact {role!r} identity differs from its inventory.")


def publish_staged_bundle(
    staged: Mapping[str, str | Path],
    destinations: Mapping[str, str | Path],
    *,
    completion_role: str = "manifest",
    expected: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Publish a validated bundle with rollback and the completion marker last.

    The marker is absent while files change. Handled failures, including
    KeyboardInterrupt, restore the prior bundle before restoring its marker.
    This is a transaction over individual atomic renames, not a claim of
    multi-file atomicity across process kill or power loss.
    """
    import shutil

    if set(staged) != set(destinations) or completion_role not in staged:
        raise ValueError(
            "Staged bundle roles must match destinations and include a completion marker."
        )
    source = {role: Path(path) for role, path in staged.items()}
    target = {role: Path(path) for role, path in destinations.items()}
    directories = {path.parent.resolve() for path in target.values()}
    if len(directories) != 1 or len(set(target.values())) != len(target):
        raise ValueError("Bundle destinations must be distinct files in one directory.")
    directory = next(iter(directories))
    inventory = {}
    for role in source:
        if source[role].is_symlink() or target[role].is_symlink():
            raise ValueError("Bundle publication does not accept symlink files.")
        if source[role].resolve() == target[role].resolve():
            raise ValueError("Bundle sources must be separate staging files.")
        if source[role].name != target[role].name:
            raise ValueError("Staged bundle filenames must match their destination.")
        record = file_artifact(source[role])
        if expected is not None and (
            role not in expected
            or any(expected[role].get(key) != value for key, value in record.items())
        ):
            raise ValueError(
                f"Staged artifact {role!r} differs from its declared identity."
            )
        inventory[role] = record
    created = not directory.exists()
    directory.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=".bundle-backup-", dir=directory.parent))
    order = tuple(role for role in staged if role != completion_role) + (
        completion_role,
    )
    saved, published = [], []
    succeeded = False
    try:
        # Remove the old completion marker before replacing any old payload.
        for role in (
            completion_role,
            *[role for role in order if role != completion_role],
        ):
            if target[role].exists():
                if not target[role].is_file():
                    raise ValueError(
                        f"Bundle destination is not a regular file: {target[role]}."
                    )
                target[role].replace(backup / target[role].name)
                saved.append(role)
        for role in order:
            # Recheck just before moving; publication never binds mixed bytes.
            if file_artifact(source[role]) != inventory[role]:
                raise ValueError(
                    f"Staged artifact {role!r} changed during publication."
                )
            source[role].replace(target[role])
            published.append(role)
        succeeded = True
        return inventory
    finally:
        if not succeeded:
            for role in reversed(published):
                target[role].unlink(missing_ok=True)
            # The previous marker returns only after all previous payloads.
            for role in (
                *[role for role in saved if role != completion_role],
                *([completion_role] if completion_role in saved else []),
            ):
                (backup / target[role].name).replace(target[role])
        shutil.rmtree(backup)
        if created and not succeeded:
            try:
                directory.rmdir()
            except OSError:
                pass
