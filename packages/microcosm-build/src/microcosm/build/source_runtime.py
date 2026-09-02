"""Shared interpreter for declarative source-stage manifests.

Country manifests describe source operations as data. This module executes the
generic envelope of that plan: table reads, operation dispatch, and explicit
stop points for staged/cached builds. Operation implementations are injected by
shared runtimes, not named inside manifests.

It also owns the fail-closed root-identity gate (microcosm#848): before a build
reads a raw microdata root it hashes the file it was handed and refuses to
continue unless the bytes are the ones the manifest pins and a Chronicle
registration witnesses. Where a producing run already recorded per-source pins —
the ASEC raw-stage checkpoint does — the recorded pins are cross-checked against
the manifest instead of re-hashing gigabytes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from microcosm.build.source_manifest import (
    ChronicleArtifactReference,
    MicrodataArtifactEntry,
    SourceManifest,
    SourceOperationSpec,
    SourceStageSpec,
    microdata_artifact_entries,
)

__all__ = [
    "MicrodataFileVerification",
    "MicrodataIdentityError",
    "RecordedPinAudit",
    "SourceOperationHandler",
    "SourceRuntimeConfig",
    "SourceRuntimeContext",
    "SourceRuntimeError",
    "UnsupportedSourceOperationError",
    "run_source_stage",
    "sha256_file",
    "verify_microdata_files",
    "verified_chronicle_registrations",
    "verify_recorded_microdata_pins",
]


@dataclass(frozen=True)
class SourceRuntimeConfig:
    """Build knobs available to manifest operation handlers."""

    seed: int = 0
    target_year: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceRuntimeContext:
    """Runtime context passed to injected source operation handlers."""

    config: SourceRuntimeConfig
    tables: Mapping[str, pd.DataFrame]

    def read_table(self, name: str) -> pd.DataFrame:
        """Return a defensive copy of a declared source table."""

        try:
            table = self.tables[name]
        except KeyError as exc:
            available = sorted(self.tables)
            raise SourceRuntimeError(
                f"Source table {name!r} was not provided; available tables: "
                f"{available}."
            ) from exc
        if not isinstance(table, pd.DataFrame):
            raise SourceRuntimeError(
                f"Source table {name!r} must be a pandas DataFrame, got "
                f"{type(table).__name__}."
            )
        return table.copy(deep=True)


SourceOperationHandler = Callable[
    [pd.DataFrame | None, SourceOperationSpec, SourceRuntimeContext],
    pd.DataFrame,
]


class SourceRuntimeError(RuntimeError):
    """Base error for source manifest execution failures."""


class UnsupportedSourceOperationError(SourceRuntimeError):
    """Raised when a manifest operation has no injected runtime handler."""


class MicrodataIdentityError(SourceRuntimeError):
    """Raised when a raw microdata root is not the file the manifest pins.

    This is the fail-closed root-identity gate (microcosm#848). A build reads
    raw microdata only after the bytes on disk are shown to be the registered
    ones; there is no warn-and-continue path, because every downstream artifact
    would otherwise claim a provenance it does not have.
    """


def run_source_stage(
    stage: SourceStageSpec,
    *,
    tables: Mapping[str, pd.DataFrame],
    operation_handlers: Mapping[str, SourceOperationHandler] | None = None,
    config: SourceRuntimeConfig | None = None,
    stop_after: str | None = None,
) -> pd.DataFrame:
    """Execute a source-stage manifest against explicit source tables.

    Args:
        stage: Declarative source-stage contract.
        tables: Table name -> DataFrame inputs. The runtime never discovers
            source data by importing country packages.
        operation_handlers: Injected implementations keyed by operation kind.
        config: Build knobs such as seed and target year.
        stop_after: Optional operation kind at which to return the intermediate
            frame. This lets release builds cache stage prefixes such as
            ``read_table -> disaggregate_aggregate_records`` before later
            uprating/fitting work exists.
    """

    handlers = dict(operation_handlers or {})
    context = SourceRuntimeContext(
        config=config or SourceRuntimeConfig(),
        tables=tables,
    )
    current: pd.DataFrame | None = None

    for operation in stage.operations:
        if operation.kind == "read_table":
            if current is not None:
                raise SourceRuntimeError(
                    f"Stage {stage.stage!r} attempted to read a second primary "
                    "table into an existing frame."
                )
            current = _run_read_table(operation, context)
        else:
            handler = handlers.get(operation.kind)
            if handler is None:
                raise UnsupportedSourceOperationError(
                    f"Stage {stage.stage!r} operation {operation.kind!r} has no "
                    "injected source runtime handler."
                )
            current = handler(current, operation, context)
            if not isinstance(current, pd.DataFrame):
                raise SourceRuntimeError(
                    f"Stage {stage.stage!r} operation {operation.kind!r} returned "
                    f"{type(current).__name__}, expected pandas DataFrame."
                )

        if stop_after == operation.kind:
            if current is None:  # pragma: no cover - defensive
                raise SourceRuntimeError(
                    f"Stage {stage.stage!r} stop point {stop_after!r} produced no "
                    "frame."
                )
            return current.copy(deep=True)

    if current is None:
        raise SourceRuntimeError(f"Stage {stage.stage!r} produced no source frame.")
    if stop_after is not None:
        raise SourceRuntimeError(
            f"Stage {stage.stage!r} did not contain stop point {stop_after!r}."
        )
    return current.copy(deep=True)


def _run_read_table(
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    table = operation.parameters.get("table")
    if not isinstance(table, str) or not table:
        raise SourceRuntimeError("read_table operation requires a non-empty table.")
    return context.read_table(table)


@dataclass(frozen=True)
class MicrodataFileVerification:
    """One raw-microdata root checked against its manifest pin."""

    stage: str
    locator: str
    path: Path
    key: str
    expected_sha256: str
    actual_sha256: str
    registration: ChronicleArtifactReference | None

    @property
    def matched(self) -> bool:
        return self.expected_sha256 == self.actual_sha256

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "locator": self.locator,
            "key": self.key,
            "sha256": self.actual_sha256,
        }
        if self.registration is not None:
            payload["chronicle_artifact"] = self.registration.to_payload()
        return payload


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, streamed so large microdata fits memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_microdata_files(
    source: SourceManifest | SourceStageSpec | Mapping[str, Any],
    files: Mapping[str, str | Path],
    *,
    chunk_size: int = 1 << 20,
) -> tuple[MicrodataFileVerification, ...]:
    """Hash caller-supplied microdata roots and refuse any that is not the pin.

    ``files`` maps a manifest key to the local file a build was handed. A key
    resolves against every microdata entry's ``locator`` and its ``filename``,
    because a caller-supplied private input declares the real name under
    ``filename`` while its locator is the placeholder
    ``"caller-supplied local input"``.

    Raises:
        MicrodataIdentityError: If a key names no pinned microdata root, if a
            named file is missing, or if any file's bytes are not the pinned
            ones. The message carries the publisher, vintage, locator, expected
            and actual digests, so an operator can tell a wrong vintage from a
            corrupted download without rerunning the build.
    """

    entries = microdata_artifact_entries(source)
    by_key: dict[str, list[MicrodataArtifactEntry]] = {}
    for entry in entries:
        if entry.sha256 is None:
            continue
        for key in _entry_keys(entry):
            by_key.setdefault(key, []).append(entry)

    verifications: list[MicrodataFileVerification] = []
    failures: list[str] = []
    for key in sorted(files):
        matches = by_key.get(key)
        if not matches:
            raise MicrodataIdentityError(
                f"{key!r} names no hash-pinned microdata artifact in this "
                f"manifest; pinned keys are {sorted(by_key)}."
            )
        pinned = {entry.sha256 for entry in matches}
        if len(pinned) > 1:
            # The placeholder locator "caller-supplied local input" is shared by
            # every private root, so it names several distinct files. One file
            # cannot satisfy several pins; the caller must key by filename.
            raise MicrodataIdentityError(
                f"{key!r} is ambiguous: it names "
                f"{len(matches)} microdata artifacts pinned to "
                f"{len(pinned)} different files "
                f"({sorted(entry.locator for entry in matches)}). Supply the "
                "declared filename instead."
            )
        path = Path(files[key])
        if not path.is_file():
            raise MicrodataIdentityError(
                f"{key!r} was supplied as {path}, which is not a file."
            )
        actual = sha256_file(path, chunk_size=chunk_size)
        # ``by_key`` only ever holds pinned entries, so this is the one digest
        # every match shares; read it without an assert, which -O would strip
        # out of a fail-closed gate.
        expected = next(iter(pinned))
        if expected is None:  # pragma: no cover - defensive
            raise MicrodataIdentityError(
                f"{key!r} resolved to an unpinned microdata artifact."
            )
        for entry in matches:
            verification = MicrodataFileVerification(
                stage=entry.stage,
                locator=entry.locator,
                path=path,
                key=key,
                expected_sha256=expected,
                actual_sha256=actual,
                registration=entry.chronicle_artifact,
            )
            verifications.append(verification)
            if not verification.matched:
                failures.append(_identity_failure(verification, entry))
    if failures:
        raise MicrodataIdentityError(
            "Raw microdata identity check failed; the build reads different "
            "bytes than the manifest pins:\n" + "\n".join(failures)
        )
    return tuple(verifications)


@dataclass(frozen=True)
class RecordedPinAudit:
    """What a producing run's recorded source pins resolve to."""

    resolved: tuple[ChronicleArtifactReference, ...]
    unregistered: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "chronicle_artifacts": [ref.to_payload() for ref in self.resolved],
            "unregistered_locators": list(self.unregistered),
        }


def verify_recorded_microdata_pins(
    source: SourceManifest | SourceStageSpec | Mapping[str, Any],
    pins: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> RecordedPinAudit:
    """Cross-check pins a checkpoint already recorded against the manifest.

    The ASEC raw-stage checkpoint records a ``sha256``/``member_sha256`` pin per
    source archive it consumed. Re-hashing those archives would cost gigabytes
    of reads for digests the producing run already computed, so this compares
    the recorded pins against the manifest instead.

    Disagreement is fatal: a recorded pin whose locator the manifest pins to
    different bytes stops the build. Absence is not — a locator the manifest
    declares no pin for is reported as unregistered, because that is exactly the
    state ``microdata_pins_pending.json`` records, and this check must not
    invent a registration the repository has not made.
    """

    entries = [
        entry
        for entry in microdata_artifact_entries(source)
        if entry.sha256 is not None
    ]
    resolved: list[ChronicleArtifactReference] = []
    unregistered: list[str] = []
    failures: list[str] = []
    for index, pin in enumerate(pins):
        locator = pin.get("locator")
        sha256 = pin.get("sha256")
        member_sha256 = pin.get("member_sha256")
        matches = [entry for entry in entries if entry.locator == locator]
        if not matches:
            if isinstance(locator, str) and locator not in unregistered:
                unregistered.append(locator)
            continue
        for entry in matches:
            if entry.sha256 != sha256:
                failures.append(
                    f"  pin[{index}] {locator!r} recorded sha256 {sha256!r}; "
                    f"stage {entry.stage!r} pins {entry.sha256}."
                )
                continue
            if (
                entry.member_sha256 is not None
                and member_sha256 is not None
                and entry.member_sha256 != member_sha256
            ):
                failures.append(
                    f"  pin[{index}] {locator!r} recorded member_sha256 "
                    f"{member_sha256!r}; stage {entry.stage!r} pins "
                    f"{entry.member_sha256}."
                )
                continue
            if entry.chronicle_artifact is not None:
                resolved.append(entry.chronicle_artifact)
    if failures:
        raise MicrodataIdentityError(
            f"{context}: recorded raw-microdata pins disagree with the source "
            "manifest:\n" + "\n".join(failures)
        )
    return RecordedPinAudit(
        resolved=tuple(
            sorted(
                set(resolved),
                key=lambda ref: (
                    ref.source_id,
                    ref.package_id,
                    ref.year,
                    ref.sha256,
                ),
            )
        ),
        unregistered=tuple(sorted(unregistered)),
    )


def verified_chronicle_registrations(
    verifications: Sequence[MicrodataFileVerification],
) -> tuple[ChronicleArtifactReference, ...]:
    """The distinct registrations a set of verified files resolves to.

    This is the receipt a build records: not every registration the manifest
    declares, but the ones behind the files this run actually read and hashed.
    """

    return tuple(
        sorted(
            {
                verification.registration
                for verification in verifications
                if verification.registration is not None
            },
            key=lambda ref: (ref.source_id, ref.package_id, ref.year, ref.sha256),
        )
    )


def _entry_keys(entry: MicrodataArtifactEntry) -> tuple[str, ...]:
    filename = entry.artifact.get("filename")
    keys = [entry.locator]
    if isinstance(filename, str) and filename and filename != entry.locator:
        keys.append(filename)
    return tuple(keys)


def _identity_failure(
    verification: MicrodataFileVerification,
    entry: MicrodataArtifactEntry,
) -> str:
    registration = entry.chronicle_artifact
    publisher = (
        f"{registration.source_id}/{registration.package_id}"
        if registration is not None
        else "unregistered"
    )
    vintage = entry.artifact.get("vintage")
    return (
        f"  stage {entry.stage!r} artifact {entry.locator!r} "
        f"(publisher {publisher}, vintage {vintage!r}) supplied as "
        f"{verification.path}: expected SHA-256 "
        f"{verification.expected_sha256}, got {verification.actual_sha256}."
    )
