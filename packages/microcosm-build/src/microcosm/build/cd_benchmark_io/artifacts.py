"""Pinned metadata reads and explicit producer-adapter admission.

There are no genuine producer adapters here. Each required slot needs a
separately reviewed adapter for its actual producer schema and scientific
scope. A generic metadata envelope is not a completed producer admission.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from microcosm.build.cd_benchmark.approval import (
    APPROVAL_SCHEMA,
    ApprovalAuthority,
    ApprovalRecord,
    ApprovalScope,
    ApprovalVerdict,
    ReviewerRole,
)
from microcosm.build.cd_benchmark.binding import (
    ArtifactRef,
    CandidateBinding,
    resolve_binding,
)
from microcosm.build.cd_benchmark.canonical import (
    canonical_bytes,
    digest,
    is_sha256_hex,
    strict_load,
)
from microcosm.build.cd_benchmark.protocol import CompiledProtocol
from microcosm.build.cd_benchmark.reasons import MappingStatus

_ADMISSION_TOKEN = object()

MAX_METADATA_BYTES = 8 * 1024 * 1024
IDENTITY_FIELDS = frozenset(
    (
        "candidate_id",
        "population_artifact_id",
        "population_content_sha256",
        "population_semantic_snapshot_sha256",
        "producer_run_id",
        "producer_git_head",
        "producer_implementation_sha256",
        "model_identity_or_not_invoked",
    )
)


class BoundaryError(ValueError):
    """A required I/O identity or lifecycle condition was not satisfied."""


def require(condition: object, code: str) -> None:
    if not condition:
        raise BoundaryError(code)


def fields(value: object, roster: set | frozenset, code: str) -> dict:
    require(isinstance(value, dict) and set(value) == roster, code)
    return value


def strings(value: object, code: str, *, nonempty: bool = False) -> list[str]:
    require(isinstance(value, list), code)
    require(all(isinstance(x, str) and x.strip() for x in value), code)
    require(len(value) == len(set(value)) and (value or not nonempty), code)
    return value


def stable_path(path: Path) -> None:
    """Reject symlinks in any existing path component, including parents."""
    require(path.is_absolute() and ".." not in path.parts, "artifact_path")
    for item in (path, *path.parents):
        try:
            require(not stat.S_ISLNK(item.lstat().st_mode), "artifact_path_symlink")
        except FileNotFoundError:
            if item != path:
                raise BoundaryError("artifact_path_parent_missing") from None


@dataclass(frozen=True)
class FilePin:
    """An external expected exact file identity; construction grants no authority."""

    path: Path
    size: int
    sha256: str

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))
        require(
            type(self.size) is int and 0 <= self.size <= MAX_METADATA_BYTES,
            "artifact_size",
        )
        require(is_sha256_hex(self.sha256), "artifact_sha256")
        require(self.path.is_absolute(), "artifact_path")


def read_pinned(pin: FilePin) -> bytes:
    """Bounded no-follow read; verify bytes and descriptor/path stability."""
    stable_path(pin.path)
    try:
        fd = os.open(pin.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise BoundaryError("artifact_path_open") from exc
    try:
        before = os.fstat(fd)
        require(
            stat.S_ISREG(before.st_mode) and before.st_size == pin.size, "artifact_size"
        )
        chunks = []
        remaining = pin.size + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        stable_path(pin.path)
        current = pin.path.stat(follow_symlinks=False)

        def identity(s):
            return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

        require(
            identity(before) == identity(after) == identity(current),
            "artifact_path_changed",
        )
        require(len(raw) == pin.size and digest(raw) == pin.sha256, "artifact_digest")
        return raw
    finally:
        os.close(fd)


def document(pin: FilePin) -> dict:
    raw = read_pinned(pin)
    value = strict_load(raw)
    require(
        isinstance(value, dict) and canonical_bytes(value) == raw, "artifact_canonical"
    )
    return value


@dataclass(frozen=True)
class ResolvedAuthority:
    """Approval records loaded from exact external pins, not caller verdicts."""

    authority: ApprovalAuthority
    pins: tuple[FilePin, FilePin]
    _token: object

    def recheck(self):
        require(self._token is _ADMISSION_TOKEN, "approval_loader_required")
        require(
            {r.sha256 for r in self.authority.records} == {p.sha256 for p in self.pins},
            "approval_loaded_records_changed",
        )
        for pin in self.pins:
            read_pinned(pin)


def load_authority(
    protocol: CompiledProtocol,
    *,
    root: FilePin,
    peer: FilePin,
    root_id: str,
    peer_id: str,
) -> ResolvedAuthority:
    """Resolve exact external approval bytes; implementation scope is insufficient."""
    require(root_id != peer_id, "approval_distinct_reviewers")
    records = []
    for pin, role, actor in (
        (root, ReviewerRole.ROOT, root_id),
        (peer, ReviewerRole.PEER, peer_id),
    ):
        doc = fields(
            document(pin),
            {
                "schema",
                "protocol_sha256",
                "verdict",
                "reviewer_role",
                "reviewer_id",
                "scopes",
                "review_artifact_sha256",
            },
            "approval_fields",
        )
        require(
            doc["schema"] == APPROVAL_SCHEMA
            and doc["reviewer_id"] == actor
            and doc["reviewer_role"] == role.value,
            "approval_identity",
        )
        strings(doc["scopes"], "approval_scopes", nonempty=True)
        record = ApprovalRecord(
            doc["protocol_sha256"],
            ApprovalVerdict(doc["verdict"]),
            role,
            actor,
            frozenset(ApprovalScope(x) for x in doc["scopes"]),
            doc["review_artifact_sha256"],
        )
        require(
            record.sha256 == pin.sha256 and record.protocol_sha256 == protocol.sha256,
            "approval_digest",
        )
        records.append(record)
    authority = ApprovalAuthority(tuple(records))
    for action in (
        ApprovalScope.CANDIDATE_SCORING,
        ApprovalScope.PRODUCTION_LEDGER_REGISTRATION,
    ):
        require(
            authority.resolve(protocol.sha256, action).authorized, "approval_action"
        )
    return ResolvedAuthority(authority, (root, peer), _ADMISSION_TOKEN)


@dataclass(frozen=True)
class ProducerEvidence:
    """Result of an explicitly supplied, schema-specific producer adapter.

    Source receipts retain original grain. A final-population adapter must
    authenticate the source-to-current lineage chain, including zero-weight
    rows, instead of asserting that a source receipt describes final clones.
    """

    identity: Mapping[str, str]
    dependencies: Mapping[str, str]


ProducerAdapter = Callable[[str, dict, Mapping[str, dict]], ProducerEvidence]


@dataclass(frozen=True)
class ResolvedCandidate:
    manifest_sha256: str
    manifest_pin: FilePin
    identity_bytes: bytes
    binding: CandidateBinding
    history_bytes: bytes
    artifact_pins: Mapping[str, FilePin]
    _token: object

    @property
    def identity(self) -> dict:
        return strict_load(self.identity_bytes)

    @property
    def history(self) -> dict:
        return strict_load(self.history_bytes)

    def recheck(self) -> None:
        require(self._token is _ADMISSION_TOKEN, "candidate_loader_required")
        for pin in self.artifact_pins.values():
            read_pinned(pin)
        manifest = document(self.manifest_pin)
        require(
            set(self.artifact_pins) == set(manifest["artifacts"]),
            "candidate_loaded_roster_changed",
        )
        for name, pin in self.artifact_pins.items():
            require(
                pin.sha256 == manifest["artifacts"][name]["sha256"]
                and pin.size == manifest["artifacts"][name]["bytes"],
                "candidate_loaded_pin_changed",
            )
        require(
            self.manifest_sha256 == self.manifest_pin.sha256,
            "candidate_loaded_manifest_changed",
        )
        require(
            self.identity == manifest["identity"], "candidate_loaded_identity_changed"
        )
        require(
            self.binding.income_claimed == manifest["income_claimed"],
            "candidate_loaded_binding_changed",
        )
        require(
            {
                k: {"artifact_id": v.artifact_id, "sha256": v.sha256}
                for k, v in self.binding.slots.items()
            }
            == manifest["slots"],
            "candidate_loaded_binding_changed",
        )
        require(
            self.history
            == document(self.artifact_pins[manifest["history_artifact_id"]]),
            "candidate_loaded_history_changed",
        )


def load_candidate(
    protocol: CompiledProtocol,
    *,
    manifest: FilePin,
    expected_identity: Mapping[str, str],
    files: Mapping[str, FilePin],
    adapters: Mapping[str, ProducerAdapter],
) -> ResolvedCandidate:
    """Authenticate pinned metadata and require all actual producer adapters.

    ``manifest`` and ``expected_identity`` are external admissions chosen by
    the integrating reviewer/runner, never read from a candidate as authority.
    The envelope coordinates existing producer evidence; it does not validate
    a scientific mapping. Missing genuine adapters keep scoring closed.
    """
    doc = fields(
        document(manifest),
        {
            "schema",
            "protocol_sha256",
            "identity",
            "slots",
            "artifacts",
            "income_claimed",
            "history_artifact_id",
        },
        "candidate_fields",
    )
    require(
        doc["schema"] == "microcosm.cd_candidate_io.v1"
        and doc["protocol_sha256"] == protocol.sha256,
        "candidate_protocol",
    )
    identity = fields(doc["identity"], IDENTITY_FIELDS, "candidate_identity_fields")
    require(identity == dict(expected_identity), "candidate_expected_identity")
    require(
        all(isinstance(x, str) and x.strip() for x in identity.values()),
        "candidate_identity",
    )
    for key in IDENTITY_FIELDS:
        if key.endswith("sha256"):
            require(is_sha256_hex(identity[key]), "candidate_identity_digest")
    require(
        len(identity["producer_git_head"]) == 40
        and all(c in "0123456789abcdef" for c in identity["producer_git_head"]),
        "candidate_head",
    )
    require(type(doc["income_claimed"]) is bool, "candidate_income")
    require(
        isinstance(doc["slots"], dict) and isinstance(doc["artifacts"], dict),
        "candidate_rosters",
    )
    require(set(files) == set(doc["artifacts"]), "candidate_artifact_roster")
    require("__manifest__" not in files, "candidate_reserved_artifact_id")
    artifacts = {}
    dependencies = {}
    for name, declaration in doc["artifacts"].items():
        fields(
            declaration,
            {"sha256", "bytes", "dependencies"},
            "candidate_artifact_fields",
        )
        require(
            declaration["sha256"] == files[name].sha256
            and declaration["bytes"] == files[name].size,
            "candidate_artifact_pin",
        )
        require(isinstance(declaration["dependencies"], dict), "candidate_dependencies")
        for parent, sha in declaration["dependencies"].items():
            require(
                parent in files and files[parent].sha256 == sha,
                "candidate_dependency_pin",
            )
        artifacts[name] = document(files[name])
        dependencies[name] = declaration["dependencies"]
    slots = {}
    for slot, ref in doc["slots"].items():
        fields(ref, {"artifact_id", "sha256"}, "candidate_slot")
        name = ref["artifact_id"]
        require(
            name in files and files[name].sha256 == ref["sha256"], "candidate_slot_pin"
        )
        slots[slot] = ArtifactRef(name, ref["sha256"])
    binding = CandidateBinding(slots, doc["income_claimed"])
    require(set(adapters) == set(slots), "producer_adapters_missing_or_extra")
    validated = set()
    for slot, ref in slots.items():
        evidence = adapters[slot](
            slot, artifacts[ref.artifact_id], MappingProxyType(artifacts)
        )
        require(isinstance(evidence, ProducerEvidence), "producer_admission_missing")
        require(dict(evidence.identity) == identity, "producer_population_identity")
        require(
            dict(evidence.dependencies) == dependencies[ref.artifact_id],
            "producer_dependencies",
        )
        validated.add(slot)

    class Authenticated:
        def authenticates(self, slot, reference):
            return slot in validated and slots[slot] == reference

    require(
        resolve_binding(protocol, binding, authenticator=Authenticated()).status
        is MappingStatus.APPROVED,
        "candidate_binding_incomplete",
    )
    history_id = doc["history_artifact_id"]
    require(history_id in artifacts, "history_artifact_missing")
    history = fields(
        artifacts[history_id],
        {
            "schema",
            "identity",
            "reservation_complete",
            "ancestry_complete",
            "prior_history_complete",
            "tuning_roots",
            "ancestor_populations",
            "external_exposures",
            "expected_snapshot",
        },
        "history_fields",
    )
    require(
        history["schema"] == "microcosm.cd_exposure_audit_io.v1"
        and history["identity"] == identity,
        "history_identity",
    )
    require(
        all(
            history[k] is True
            for k in (
                "reservation_complete",
                "ancestry_complete",
                "prior_history_complete",
            )
        ),
        "exposure_history_unknown",
    )
    strings(history["tuning_roots"], "history_roots", nonempty=True)
    strings(history["ancestor_populations"], "history_ancestors")
    require(
        all(is_sha256_hex(x) for x in history["ancestor_populations"]),
        "history_ancestor_digest",
    )
    require(isinstance(history["external_exposures"], list), "history_exposures")
    for exposure in history["external_exposures"]:
        fields(
            exposure,
            {"families", "tuning_roots", "populations"},
            "history_exposure_fields",
        )
        for key in exposure:
            strings(exposure[key], "history_exposure_ids", nonempty=True)
    pins = dict(files)
    allowed_families = {f for b in protocol.blocks for f in b.exposure_families}
    require(
        all(
            set(e["families"]) <= allowed_families
            for e in history["external_exposures"]
        ),
        "history_family_root_required",
    )
    return ResolvedCandidate(
        manifest.sha256,
        manifest,
        canonical_bytes(identity),
        binding,
        canonical_bytes(history),
        MappingProxyType(pins),
        _ADMISSION_TOKEN,
    )
