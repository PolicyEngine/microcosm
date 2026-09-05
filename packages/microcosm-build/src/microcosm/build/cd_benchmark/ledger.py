"""Immutable exposure-ledger records: validation, chain and state transitions.

This module reads and validates ledger bytes **in memory only**. It opens no
file, creates no ledger, appends to no ledger and takes no lock, which is what
``/exposure_ledger/no_actual_ledger_created`` and
``/exposure_ledger/status`` require of anything written before the real local
contract exists. :func:`build_records` composes a record sequence as bytes so
fixtures can exercise the rules; handing those bytes to a real ledger is a
separate, unimplemented act.

Field and payload rosters are read out of the compiled contract rather than
retyped here, so a roster cannot silently drift from the document. Four event
payloads the document describes in prose instead of a list
(``protocol_amended``, ``evaluation_interrupted``, ``external_exposure_recorded``
and ``ledger_recovery``) get closed rosters defined below; those four are this
package's reading of the prose, and the docs note them as such.

Exposure precedence and attempt terminality follow
``/exposure_ledger/exposure_status_precedence`` and
``/exposure_ledger/attempt_terminality`` exactly. In particular a started
attempt is prior exposure for related later attempts even if it never
completed, an interrupted attempt can never receive a retroactive completion,
and unknown history is never fresh.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from microcosm.build.cd_benchmark.approval import RegistrationClaim
from microcosm.build.cd_benchmark.canonical import (
    CanonicalJsonError,
    canonical_bytes,
    digest,
    is_sha256_hex,
    strict_load,
)
from microcosm.build.cd_benchmark.protocol import (
    CompiledProtocol,
    DistributionBlock,
    PointBlock,
)
from microcosm.build.cd_benchmark.reasons import ExposureStatus, Reason

__all__ = [
    "AMENDMENT_PAYLOAD_FIELDS",
    "AttemptState",
    "EXTERNAL_EXPOSURE_PAYLOAD_FIELDS",
    "EventType",
    "ExposureHistory",
    "INTERRUPTION_PAYLOAD_FIELDS",
    "LEDGER_SCHEMA_VERSION",
    "LedgerError",
    "LedgerRecord",
    "LedgerSnapshot",
    "LedgerVerification",
    "RECOVERY_PAYLOAD_FIELDS",
    "UUID4_PATTERN",
    "VerifiedLedger",
    "build_records",
    "exposed_families",
    "registration_claim",
    "exposure_status",
    "new_evaluation_id",
    "verify_ledger",
]

LEDGER_SCHEMA_VERSION = 1

#: A lowercase canonical UUIDv4 with the version and variant nibbles pinned.
UUID4_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)

#: ``/exposure_ledger/record_fields/recorded_at`` says UTC RFC3339. A ``Z`` or
#: an explicit ``+00:00`` offset is UTC; any other offset is refused.
_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|\+00:00)\Z"
)

#: Closed rosters for the four payloads the document describes in prose.
AMENDMENT_PAYLOAD_FIELDS = (
    "protocol_sha256",
    "protocol_parent_sha256",
    "changes",
    "root_approval_artifact_sha256",
    "peer_approval_artifact_sha256",
)
INTERRUPTION_PAYLOAD_FIELDS = (
    "evaluation_id",
    "attempt_record_sha256",
    "interruption_reason",
    "known_output_hashes",
    "exposure_possible",
)
EXTERNAL_EXPOSURE_PAYLOAD_FIELDS = (
    "prior_evaluation_ids",
    "related_ancestry",
    "block_families",
    "occurred_at",
    "artifacts",
)
RECOVERY_PAYLOAD_FIELDS = (
    "prior_ledger_id",
    "prior_file_sha256",
    "prior_prefix_sha256",
    "prior_prefix_bytes",
    "unresolved_attempt_ids",
    "unresolved_exposure",
)


class LedgerError(ValueError):
    """A ledger record or sequence is not what the contract admits."""


class EventType(StrEnum):
    """``/exposure_ledger/other_event_types`` — the closed event vocabulary."""

    PROTOCOL_REGISTERED = "protocol_registered"
    PROTOCOL_AMENDED = "protocol_amended"
    EVALUATION_ATTEMPT = "evaluation_attempt"
    EVALUATION_COMPLETION = "evaluation_completion"
    EVALUATION_INTERRUPTED = "evaluation_interrupted"
    EXTERNAL_EXPOSURE_RECORDED = "external_exposure_recorded"
    LEDGER_RECOVERY = "ledger_recovery"


@dataclass(frozen=True)
class LedgerRecord:
    """One immutable append-only record. There is no self-hash member."""

    ledger_id: str
    schema_version: int
    sequence: int
    recorded_at: str
    actor_id: str
    event_type: EventType
    payload: Mapping[str, object]
    previous_record_sha256: str | None

    def as_object(self) -> dict[str, object]:
        """The record as its canonical JSON object."""
        return {
            "ledger_id": self.ledger_id,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "recorded_at": self.recorded_at,
            "actor_id": self.actor_id,
            "event_type": str(self.event_type),
            "payload": dict(self.payload),
            "previous_record_sha256": self.previous_record_sha256,
        }

    @property
    def canonical_line(self) -> bytes:
        """Exact canonical record bytes, excluding the terminating LF."""
        return canonical_bytes(self.as_object())

    @property
    def sha256(self) -> str:
        """``SHA256`` of the canonical record bytes, LF excluded."""
        return digest(self.canonical_line)


@dataclass(frozen=True)
class LedgerSnapshot:
    """``/exposure_ledger/snapshot_fields`` for a verified prefix."""

    ledger_id: str
    record_count: int
    tip_sequence: int | None
    tip_record_sha256: str | None
    exact_ndjson_prefix_sha256: str
    prefix_bytes: int


@dataclass(frozen=True)
class AttemptState:
    """One evaluation attempt and its single terminal record, if any."""

    evaluation_id: str
    attempt: LedgerRecord
    terminal: LedgerRecord | None
    terminal_state: str | None

    @property
    def unfinished(self) -> bool:
        """An attempt with neither a completion nor an interruption."""
        return self.terminal is None

    @property
    def interrupted(self) -> bool:
        """Whether this attempt ended in an interruption record."""
        return (
            self.terminal is not None
            and self.terminal.event_type is EventType.EVALUATION_INTERRUPTED
        )


@dataclass(frozen=True)
class VerifiedLedger:
    """A fully validated record sequence with its terminal attempt states."""

    ledger_id: str
    records: tuple[LedgerRecord, ...]
    attempts: Mapping[str, AttemptState]
    raw: bytes

    def snapshot(self) -> LedgerSnapshot:
        """Return the snapshot fields for the whole verified prefix."""
        tip = self.records[-1] if self.records else None
        return LedgerSnapshot(
            ledger_id=self.ledger_id,
            record_count=len(self.records),
            tip_sequence=None if tip is None else tip.sequence,
            tip_record_sha256=None if tip is None else tip.sha256,
            exact_ndjson_prefix_sha256=digest(self.raw),
            prefix_bytes=len(self.raw),
        )


@dataclass(frozen=True)
class LedgerVerification:
    """The outcome of reading ledger bytes: a ledger, or the reasons why not."""

    ledger: VerifiedLedger | None
    reasons: tuple[Reason, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the bytes verified into a usable ledger."""
        return self.ledger is not None


def _closed(
    payload: object,
    roster: Sequence[tuple[str, tuple[str, ...] | None]],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(payload, dict):
        raise LedgerError(f"{label} payload must be an object.")
    names = {name for name, _ in roster}
    if set(payload) != names:
        missing = sorted(names - set(payload))
        unknown = sorted(set(payload) - names)
        raise LedgerError(
            f"{label} payload keys are closed; missing={missing} unknown={unknown}."
        )
    for name, allowed in roster:
        if allowed is not None and payload[name] not in allowed:
            raise LedgerError(
                f"{label} payload field {name!r} must be one of {list(allowed)}."
            )
    return payload


def _roster(
    protocol: CompiledProtocol, event: EventType
) -> tuple[tuple[str, tuple[str, ...] | None], ...]:
    if event is EventType.EVALUATION_ATTEMPT:
        return protocol.attempt_payload_fields
    if event is EventType.EVALUATION_COMPLETION:
        return protocol.completion_payload_fields
    if event is EventType.PROTOCOL_REGISTERED:
        return tuple((name, None) for name in protocol.registration_payload_fields)
    if event is EventType.PROTOCOL_AMENDED:
        return tuple((name, None) for name in AMENDMENT_PAYLOAD_FIELDS)
    if event is EventType.EVALUATION_INTERRUPTED:
        return tuple((name, None) for name in INTERRUPTION_PAYLOAD_FIELDS)
    if event is EventType.EXTERNAL_EXPOSURE_RECORDED:
        return tuple((name, None) for name in EXTERNAL_EXPOSURE_PAYLOAD_FIELDS)
    return tuple((name, None) for name in RECOVERY_PAYLOAD_FIELDS)


def parse_record(value: object, *, protocol: CompiledProtocol) -> LedgerRecord:
    """Validate one record object against the closed field and payload rosters.

    Raises:
        LedgerError: On an unknown or missing field, a bad type, a non-UTC
            timestamp, an unknown event type or a payload key outside the
            event's closed roster.
    """
    if not isinstance(value, dict):
        raise LedgerError("A ledger record must be a JSON object.")
    if set(value) != set(protocol.ledger_record_fields):
        missing = sorted(set(protocol.ledger_record_fields) - set(value))
        unknown = sorted(set(value) - set(protocol.ledger_record_fields))
        raise LedgerError(
            f"Record fields are closed; missing={missing} unknown={unknown}."
        )
    ledger_id = value["ledger_id"]
    if not isinstance(ledger_id, str) or not ledger_id:
        raise LedgerError("ledger_id must be a nonempty stable identifier.")
    version = value["schema_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != LEDGER_SCHEMA_VERSION
    ):
        raise LedgerError(f"schema_version must be {LEDGER_SCHEMA_VERSION}.")
    sequence = value["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise LedgerError("sequence must be a nonnegative integer.")
    recorded_at = value["recorded_at"]
    if not isinstance(recorded_at, str) or not _TIMESTAMP.fullmatch(recorded_at):
        raise LedgerError("recorded_at must be a UTC RFC3339 timestamp.")
    actor_id = value["actor_id"]
    if not isinstance(actor_id, str) or not actor_id:
        raise LedgerError("actor_id must be a declared local actor identity.")
    raw_event = value["event_type"]
    if raw_event not in set(protocol.ledger_event_types):
        raise LedgerError(f"Unknown event_type {raw_event!r}.")
    event = EventType(raw_event)
    previous = value["previous_record_sha256"]
    if previous is not None and not is_sha256_hex(previous):
        raise LedgerError("previous_record_sha256 must be null or a digest.")
    payload = _closed(value["payload"], _roster(protocol, event), str(event))
    if (
        event is EventType.EVALUATION_INTERRUPTED
        and payload["exposure_possible"] is not True
    ):
        raise LedgerError("An interruption must record exposure_possible=true.")
    if event is EventType.EVALUATION_ATTEMPT and not UUID4_PATTERN.fullmatch(
        str(payload["evaluation_id"])
    ):
        raise LedgerError("evaluation_id must be a lowercase canonical UUIDv4.")
    return LedgerRecord(
        ledger_id=ledger_id,
        schema_version=LEDGER_SCHEMA_VERSION,
        sequence=sequence,
        recorded_at=recorded_at,
        actor_id=actor_id,
        event_type=event,
        payload=MappingProxyType(dict(payload)),
        previous_record_sha256=previous,
    )


def build_records(records: Iterable[LedgerRecord]) -> bytes:
    """Serialize records as append-only NDJSON: canonical bytes plus one LF each.

    This produces bytes for fixtures and snapshots. It writes nothing anywhere.
    """
    return b"".join(record.canonical_line + b"\n" for record in records)


def _transitions(
    records: Sequence[LedgerRecord], linked_evaluation_ids: frozenset[str]
) -> dict[str, AttemptState]:
    attempts: dict[str, AttemptState] = {}
    for record in records:
        payload = record.payload
        if record.event_type is EventType.EVALUATION_ATTEMPT:
            evaluation_id = str(payload["evaluation_id"])
            if evaluation_id in attempts or evaluation_id in linked_evaluation_ids:
                raise LedgerError(
                    "Duplicate evaluation_id; a collision is refused, never reused."
                )
            attempts[evaluation_id] = AttemptState(
                evaluation_id=evaluation_id,
                attempt=record,
                terminal=None,
                terminal_state=None,
            )
            continue
        if record.event_type not in {
            EventType.EVALUATION_COMPLETION,
            EventType.EVALUATION_INTERRUPTED,
        }:
            continue
        evaluation_id = str(payload["evaluation_id"])
        state = attempts.get(evaluation_id)
        if state is None:
            raise LedgerError("A terminal record names no recorded attempt.")
        if payload["attempt_record_sha256"] != state.attempt.sha256:
            raise LedgerError("A terminal record's attempt hash does not bind.")
        if not state.unfinished:
            raise LedgerError(
                "An attempt takes exactly one terminal record: a completion "
                "after interruption, an interruption after completion, a second "
                "interruption and a duplicate completion are all refused."
            )
        attempts[evaluation_id] = AttemptState(
            evaluation_id=evaluation_id,
            attempt=state.attempt,
            terminal=record,
            terminal_state=(
                str(payload["terminal_state"])
                if record.event_type is EventType.EVALUATION_COMPLETION
                else "interrupted"
            ),
        )
    return attempts


def verify_ledger(
    data: bytes,
    *,
    protocol: CompiledProtocol,
    linked_evaluation_ids: frozenset[str] = frozenset(),
) -> LedgerVerification:
    """Verify append-only NDJSON bytes into a :class:`VerifiedLedger`.

    Every line must be exactly the canonical serialization of its own record
    followed by one LF; a partial or non-canonical tail is
    ``ledger_integrity_unknown`` and the whole file stays unusable rather than
    being truncated back to a valid prefix. Sequences must start at zero and be
    consecutive, ``previous_record_sha256`` must be null only at sequence zero
    and otherwise cover the complete previous record, and every attempt must
    obey the terminality rule.
    """
    notes: dict[str, object] = {}
    if not isinstance(data, bytes):
        raise LedgerError("Ledger bytes are required.")
    if data.startswith(b"\xef\xbb\xbf"):
        return LedgerVerification(
            None,
            (Reason.LEDGER_INTEGRITY_UNKNOWN,),
            MappingProxyType({"byte_order_mark": True}),
        )
    if data and not data.endswith(b"\n"):
        return LedgerVerification(
            None,
            (Reason.LEDGER_INTEGRITY_UNKNOWN,),
            MappingProxyType({"partial_or_corrupt_tail": True}),
        )
    lines = data.split(b"\n")[:-1] if data else []
    records: list[LedgerRecord] = []
    try:
        for index, line in enumerate(lines):
            record = parse_record(strict_load(line), protocol=protocol)
            if record.canonical_line != line:
                raise LedgerError("A ledger line is not its canonical serialization.")
            if record.sequence != index:
                raise LedgerError(
                    "Ledger sequences must start at 0 and be consecutive."
                )
            if index == 0:
                if record.previous_record_sha256 is not None:
                    raise LedgerError("Sequence 0 has a null previous record hash.")
            else:
                if record.ledger_id != records[0].ledger_id:
                    raise LedgerError("A ledger has one stable ledger_id.")
                if record.previous_record_sha256 != records[-1].sha256:
                    raise LedgerError("The record hash chain is broken or forked.")
            records.append(record)
        attempts = _transitions(records, linked_evaluation_ids)
    except (LedgerError, CanonicalJsonError) as error:
        notes["refusal"] = str(error)
        return LedgerVerification(
            None, (Reason.LEDGER_INTEGRITY_UNKNOWN,), MappingProxyType(notes)
        )
    return LedgerVerification(
        VerifiedLedger(
            ledger_id=records[0].ledger_id if records else "",
            records=tuple(records),
            attempts=MappingProxyType(attempts),
            raw=data,
        ),
        (),
        MappingProxyType(notes),
    )


def new_evaluation_id(
    existing: Iterable[str], *, generator: Callable[[], uuid.UUID] = uuid.uuid4
) -> str:
    """Generate a fresh lowercase canonical UUIDv4 not present in ``existing``.

    ``/exposure_ledger/evaluation_id_generation``: uniqueness is checked across
    the verified ledger and every linked recovery history, a collision is
    refused rather than silently reused, and the identifier supplies uniqueness
    only — never authority or chronology.

    Raises:
        LedgerError: If the generator returns a non-v4 or colliding identifier.
    """
    known = set(existing)
    value = str(generator())
    if not UUID4_PATTERN.fullmatch(value):
        raise LedgerError("Evaluation ids must be lowercase canonical UUIDv4.")
    if value in known:
        raise LedgerError("Evaluation id collision is refused, never reused.")
    return value


def exposed_families(ledger: VerifiedLedger) -> frozenset[str]:
    """Reference families with a recorded related attempt or external exposure.

    A started attempt counts even when it failed or was interrupted: no
    completed numeric score is required for prior exposure.
    """
    families: set[str] = set()
    for record in ledger.records:
        if record.event_type is EventType.EVALUATION_ATTEMPT:
            families.update(
                str(item)
                for item in record.payload["all_derivative_reference_family_ids"]
            )
        elif record.event_type is EventType.EXTERNAL_EXPOSURE_RECORDED:
            families.update(str(item) for item in record.payload["block_families"])
    return frozenset(families)


def registration_claim(record: LedgerRecord) -> RegistrationClaim:
    """Read a validated ``protocol_registered`` record as a registration claim.

    This is the only route from ledger bytes to an activation decision, and it
    carries no authority of its own: the claim still has to be resolved against
    an injected :class:`~microcosm.build.cd_benchmark.approval.ApprovalAuthority`
    that actually holds the approval artifacts the payload names.

    Raises:
        LedgerError: If ``record`` is not a ``protocol_registered`` record, or
            its payload fields are not the shapes a claim needs.
    """
    if record.event_type is not EventType.PROTOCOL_REGISTERED:
        raise LedgerError("Only a protocol_registered record carries a claim.")
    payload = record.payload
    try:
        return RegistrationClaim(
            protocol_sha256=payload["protocol_sha256"],
            protocol_parent_sha256=payload["protocol_parent_sha256"],
            legacy_parent_artifact_hashes=tuple(
                payload["legacy_parent_artifact_hashes_if_first_version"]
            ),
            parameter_changes=tuple(
                payload["parameter_changes_old_new_reason_evidence"]
            ),
            root_approval_artifact_sha256=payload["root_approval_artifact_sha256"],
            peer_approval_artifact_sha256=payload["peer_approval_artifact_sha256"],
            reference_inventory_sha256=payload["reference_inventory_sha256"],
            block_definition_hashes=payload["block_definition_hashes"],
        )
    except (TypeError, ValueError) as error:
        raise LedgerError(f"Registration payload is not a claim: {error}") from None


@dataclass(frozen=True)
class ExposureHistory:
    """The audits ``/exposure_ledger/first_attempt`` requires before freshness."""

    reservation_audit_complete: bool
    ancestry_audit_complete: bool
    prior_history_complete: bool
    related_exposed_families: frozenset[str]

    @property
    def complete(self) -> bool:
        """Whether reservation, ancestry and prior history are all established."""
        return (
            self.reservation_audit_complete
            and self.ancestry_audit_complete
            and self.prior_history_complete
        )


def exposure_status(
    block: PointBlock | DistributionBlock, history: ExposureHistory
) -> tuple[ExposureStatus, tuple[Reason, ...]]:
    """Classify one block's exposure, in the document's precedence order.

    Incomplete history, reservation or ancestry first: that is
    ``exposure_history_unknown`` and scope-incomplete, never a guessed status.
    Otherwise a nonreserved family is ordinary conformity with no holdout claim.
    Otherwise a reserved family with any earlier related attempt or external
    exposure is post-exposure. Only the remaining reserved first-attempt case
    may be fresh, and even then it is same-source, never independent-source
    evidence.
    """
    if not history.complete:
        return (
            ExposureStatus.EXPOSURE_HISTORY_UNKNOWN,
            (Reason.EXPOSURE_HISTORY_UNKNOWN,),
        )
    if not block.reserved:
        return ExposureStatus.NOT_RESERVED_CONFORMITY, ()
    if block.exposure_families & history.related_exposed_families:
        return ExposureStatus.POST_EXPOSURE_CONFORMITY, ()
    return ExposureStatus.FRESH_SAME_SOURCE_HOLDOUT_ELIGIBLE, ()
