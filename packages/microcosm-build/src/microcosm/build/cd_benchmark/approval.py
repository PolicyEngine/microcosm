"""A compact local representation of the external approval artifacts.

``/approval/transition`` requires each approval artifact to state ``APPROVE``,
the exact protocol SHA, the approving actor or reviewer identity, and the
approved scope, and says plainly that "hashes or caller-provided verdict labels
alone are not approval". The document does not fix a byte format for those
artifacts — the peer review calls that out as an open, non-blocking item — so
this module defines one as an *implementation detail* of this package. It names
the exact protocol, the verdict, the reviewer role and identity, and the scope
enum; it does not amend the contract and does not add a protocol version.

What a local record cannot do is mint external authority. Constructing an
:class:`ApprovalAuthority` asserts that the caller resolved those immutable
external artifacts; this module authenticates none of them, and there is
deliberately **no module-level default, no production authority and no
implicit fallback**. A fixture authority has to be constructed and injected
explicitly at the call site, so no code path acquires approval by omission.

Activation additionally needs a valid ``protocol_registered`` record naming the
same digests (``/exposure_ledger/protocol_registration_payload``). A registration
that names an approval digest the injected authority does not hold is refused:
a ledger line cannot authorize itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from microcosm.build.cd_benchmark.canonical import (
    canonical_bytes,
    digest,
    is_sha256_hex,
)
from microcosm.build.cd_benchmark.protocol import CompiledProtocol
from microcosm.build.cd_benchmark.reasons import Reason, dedupe

__all__ = [
    "APPROVAL_SCHEMA",
    "ApprovalAuthority",
    "ApprovalError",
    "ApprovalRecord",
    "ApprovalResolution",
    "ApprovalScope",
    "ApprovalVerdict",
    "RegistrationClaim",
    "ReviewerRole",
    "resolve_action",
]

APPROVAL_SCHEMA = "microcosm.cd_benchmark_approval.v1"


class ApprovalError(ValueError):
    """An approval or registration representation is malformed."""


class ApprovalVerdict(StrEnum):
    """The verdict an approval artifact states."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"


class ReviewerRole(StrEnum):
    """Which of the two required independent approvals an artifact carries."""

    ROOT = "root"
    PEER = "peer"


class ApprovalScope(StrEnum):
    """Actions an approval may cover, named as ``/approval/transition`` splits them.

    ``INVENTED_FIXTURE_IMPLEMENTATION_ONLY`` is the scope the in-force approval
    artifact states. It never permits candidate scoring: ``CANDIDATE_SCORING``
    is a separate scope that additionally requires authenticated candidate
    bindings, a complete reservation and history audit and a durable evaluation
    attempt, none of which this package supplies.
    """

    INVENTED_FIXTURE_IMPLEMENTATION_ONLY = "invented_fixture_implementation_only"
    CANDIDATE_SCORING = "candidate_scoring"
    PRODUCTION_LEDGER_REGISTRATION = "production_ledger_registration"


@dataclass(frozen=True)
class ApprovalRecord:
    """One external approval artifact, as this package represents it."""

    protocol_sha256: str
    verdict: ApprovalVerdict
    reviewer_role: ReviewerRole
    reviewer_id: str
    scopes: frozenset[ApprovalScope]
    review_artifact_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.protocol_sha256, "protocol_sha256"),
            (self.review_artifact_sha256, "review_artifact_sha256"),
        ):
            if not is_sha256_hex(value):
                raise ApprovalError(f"{label} must be a lowercase SHA-256 digest.")
        if not isinstance(self.verdict, ApprovalVerdict):
            raise ApprovalError("verdict must be an ApprovalVerdict.")
        if not isinstance(self.reviewer_role, ReviewerRole):
            raise ApprovalError("reviewer_role must be a ReviewerRole.")
        if not isinstance(self.reviewer_id, str) or not self.reviewer_id.strip():
            raise ApprovalError("reviewer_id must name the approving actor.")
        scopes = frozenset(self.scopes)
        if not scopes or any(not isinstance(item, ApprovalScope) for item in scopes):
            raise ApprovalError("scopes must be a nonempty set of ApprovalScope.")
        object.__setattr__(self, "scopes", scopes)

    def as_payload(self) -> dict[str, object]:
        """The canonical object this record hashes as."""
        return {
            "schema": APPROVAL_SCHEMA,
            "protocol_sha256": self.protocol_sha256,
            "verdict": str(self.verdict),
            "reviewer_role": str(self.reviewer_role),
            "reviewer_id": self.reviewer_id,
            "scopes": sorted(str(item) for item in self.scopes),
            "review_artifact_sha256": self.review_artifact_sha256,
        }

    @property
    def sha256(self) -> str:
        """The digest a registration record must name to reference this record."""
        return digest(canonical_bytes(self.as_payload()))


@dataclass(frozen=True)
class RegistrationClaim:
    """The ``protocol_registered`` payload, without any ledger side effect."""

    protocol_sha256: str
    protocol_parent_sha256: str
    legacy_parent_artifact_hashes: tuple[str, ...]
    parameter_changes: tuple[Mapping[str, object], ...]
    root_approval_artifact_sha256: str
    peer_approval_artifact_sha256: str
    reference_inventory_sha256: str
    block_definition_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        for value, label in (
            (self.protocol_sha256, "protocol_sha256"),
            (self.protocol_parent_sha256, "protocol_parent_sha256"),
            (self.root_approval_artifact_sha256, "root_approval_artifact_sha256"),
            (self.peer_approval_artifact_sha256, "peer_approval_artifact_sha256"),
            (self.reference_inventory_sha256, "reference_inventory_sha256"),
        ):
            if not is_sha256_hex(value):
                raise ApprovalError(f"{label} must be a lowercase SHA-256 digest.")
        if any(not is_sha256_hex(item) for item in self.legacy_parent_artifact_hashes):
            raise ApprovalError("Legacy parent artifact hashes must be digests.")
        hashes = dict(self.block_definition_hashes)
        if not hashes or any(not is_sha256_hex(item) for item in hashes.values()):
            raise ApprovalError("Block definition hashes must be digests.")
        object.__setattr__(self, "block_definition_hashes", MappingProxyType(hashes))
        object.__setattr__(self, "parameter_changes", tuple(self.parameter_changes))


@dataclass(frozen=True)
class ApprovalResolution:
    """Whether a requested action is authorized, and on what evidence."""

    protocol_sha256: str
    action: ApprovalScope
    authorized: bool
    root: ApprovalRecord | None
    peer: ApprovalRecord | None
    reasons: tuple[Reason, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovalAuthority:
    """A caller-supplied view of externally resolved approval artifacts.

    There is no default instance and no production instance. Building one is an
    explicit assertion by the caller that these artifacts were resolved; this
    class does not authenticate them, and a record here never becomes external
    authority on its own.
    """

    records: tuple[ApprovalRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(item, ApprovalRecord) for item in records):
            raise ApprovalError("An authority holds ApprovalRecord values only.")
        digests = [item.sha256 for item in records]
        if len(set(digests)) != len(digests):
            raise ApprovalError("Duplicate approval records in one authority.")
        object.__setattr__(self, "records", records)

    @classmethod
    def empty(cls) -> ApprovalAuthority:
        """An authority holding nothing; every action stays unapproved."""
        return cls(records=())

    def by_digest(self, value: str) -> ApprovalRecord | None:
        """Return the held record whose canonical digest is ``value``."""
        for item in self.records:
            if item.sha256 == value:
                return item
        return None

    def _role(
        self,
        protocol_sha256: str,
        action: ApprovalScope,
        role: ReviewerRole,
        notes: dict[str, object],
        problems: list[Reason],
    ) -> ApprovalRecord | None:
        matching = [
            item
            for item in self.records
            if item.reviewer_role is role and item.protocol_sha256 == protocol_sha256
        ]
        if any(item.verdict is ApprovalVerdict.REJECT for item in matching):
            notes[f"{role}_rejection_present"] = True
            problems.append(Reason.PROTOCOL_UNAPPROVED)
            return None
        approvals = [
            item
            for item in matching
            if item.verdict is ApprovalVerdict.APPROVE and action in item.scopes
        ]
        if len(approvals) != 1:
            notes[f"{role}_approvals_for_action"] = len(approvals)
            problems.append(Reason.PROTOCOL_UNAPPROVED)
            return None
        return approvals[0]

    def resolve(
        self, protocol_sha256: str, action: ApprovalScope
    ) -> ApprovalResolution:
        """Resolve exactly one root and one peer APPROVE for this SHA and action.

        A verdict for a different protocol digest, a different scope, or from
        only one of the two roles leaves the action ``protocol_unapproved``.
        """
        if not is_sha256_hex(protocol_sha256):
            raise ApprovalError("protocol_sha256 must be a lowercase SHA-256 digest.")
        if not isinstance(action, ApprovalScope):
            raise ApprovalError("action must be an ApprovalScope.")
        problems: list[Reason] = []
        notes: dict[str, object] = {"action": str(action)}
        root = self._role(protocol_sha256, action, ReviewerRole.ROOT, notes, problems)
        peer = self._role(protocol_sha256, action, ReviewerRole.PEER, notes, problems)
        authorized = root is not None and peer is not None
        if authorized:
            notes["root_reviewer_id"] = root.reviewer_id
            notes["peer_reviewer_id"] = peer.reviewer_id
        return ApprovalResolution(
            protocol_sha256=protocol_sha256,
            action=action,
            authorized=authorized,
            root=root,
            peer=peer,
            reasons=dedupe(problems),
            evidence=MappingProxyType(notes),
        )


def _registration_problems(
    protocol: CompiledProtocol,
    registration: RegistrationClaim,
    authority: ApprovalAuthority,
    notes: dict[str, object],
) -> list[Reason]:
    problems: list[Reason] = []
    if registration.protocol_sha256 != protocol.sha256:
        notes["registration_protocol_mismatch"] = True
        problems.append(Reason.IDENTITY_MISMATCH)
    if registration.protocol_parent_sha256 != protocol.parent_sha256:
        notes["registration_parent_mismatch"] = True
        problems.append(Reason.IDENTITY_MISMATCH)
    if registration.reference_inventory_sha256 != protocol.reference_inventory_sha256:
        notes["registration_inventory_mismatch"] = True
        problems.append(Reason.IDENTITY_MISMATCH)
    if dict(registration.block_definition_hashes) != dict(
        protocol.block_definition_hashes
    ):
        notes["registration_block_hash_mismatch"] = True
        problems.append(Reason.IDENTITY_MISMATCH)
    for role, claimed in (
        (ReviewerRole.ROOT, registration.root_approval_artifact_sha256),
        (ReviewerRole.PEER, registration.peer_approval_artifact_sha256),
    ):
        held = authority.by_digest(claimed)
        if held is None or held.reviewer_role is not role:
            notes[f"{role}_artifact_unresolved"] = True
            problems.append(Reason.PROTOCOL_UNAPPROVED)
    return problems


def resolve_action(
    protocol: CompiledProtocol,
    *,
    authority: ApprovalAuthority,
    action: ApprovalScope,
    registration: RegistrationClaim | None,
) -> ApprovalResolution:
    """Decide whether ``action`` is authorized for this exact protocol.

    Three independent conditions, all required. The compiled document must be
    the approved canonical bytes. A ``protocol_registered`` claim must name the
    same protocol, parent, reference inventory, block definitions and approval
    artifact digests. The injected authority must resolve one root and one peer
    ``APPROVE`` naming this digest and covering this action.

    A missing registration is not an oversight this function forgives: without
    one the action stays ``protocol_unapproved``.
    """
    if not isinstance(authority, ApprovalAuthority):
        raise ApprovalError("An explicit ApprovalAuthority must be injected.")
    notes: dict[str, object] = {
        "canonical_protocol": protocol.canonical,
        "registration_supplied": registration is not None,
    }
    problems: list[Reason] = []
    if not protocol.canonical:
        notes["non_canonical_protocol_bytes"] = True
        problems.append(Reason.PROTOCOL_UNAPPROVED)
    if registration is None:
        problems.append(Reason.PROTOCOL_UNAPPROVED)
    else:
        problems.extend(
            _registration_problems(protocol, registration, authority, notes)
        )
    resolution = authority.resolve(protocol.sha256, action)
    problems.extend(resolution.reasons)
    authorized = resolution.authorized and not problems
    return ApprovalResolution(
        protocol_sha256=protocol.sha256,
        action=action,
        authorized=authorized,
        root=resolution.root if authorized else None,
        peer=resolution.peer if authorized else None,
        reasons=dedupe(problems),
        evidence=MappingProxyType({**notes, **dict(resolution.evidence)}),
    )
