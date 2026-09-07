"""Single-writer, append-before-permit CD ledger and immutable attachments.

No comparator is implemented or imported. A permit is usable only inside its
locked context after the attempt has reached fsync. Advisory locking is local
cooperation, not an external witness against a malicious filesystem owner.
"""

from __future__ import annotations

import fcntl
import os
import stat
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from microcosm.build.cd_benchmark.approval import ApprovalScope, resolve_action
from microcosm.build.cd_benchmark.binding import MAPPING_SLOTS
from microcosm.build.cd_benchmark.canonical import canonical_bytes, digest, strict_load
from microcosm.build.cd_benchmark.ledger import (
    EventType,
    ExposureHistory,
    LedgerRecord,
    exposure_status,
    new_evaluation_id,
    registration_claim,
    verify_ledger,
)
from microcosm.build.cd_benchmark.protocol import (
    APPROVED_PROTOCOL_SHA256,
    CompiledProtocol,
)
from microcosm.build.cd_benchmark.reasons import BenchmarkVerdict, Reason

from .artifacts import (
    BoundaryError,
    FilePin,
    ResolvedAuthority,
    ResolvedCandidate,
    fields,
    read_pinned,
    require,
    stable_path,
    strings,
)

MAX_LEDGER_BYTES = 16 * 1024 * 1024


def _now():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _snapshot(ledger):
    s = ledger.snapshot()
    return dict(
        ledger_id=s.ledger_id,
        record_count=s.record_count,
        tip_sequence=s.tip_sequence,
        tip_record_sha256=s.tip_record_sha256,
        exact_NDJSON_prefix_sha256=s.exact_ndjson_prefix_sha256,
        prefix_bytes=s.prefix_bytes,
    )


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        require(count > 0, "write_no_progress")
        view = view[count:]


class DurableLedger:
    """An explicitly located local ledger; construction itself performs no I/O."""

    def __init__(
        self,
        directory: Path,
        ledger_id: str,
        actor_id: str,
        protocol: CompiledProtocol,
        authority: ResolvedAuthority,
        *,
        lock_timeout: float = 0,
    ):
        self.directory = Path(directory)
        require(self.directory.is_absolute(), "ledger_path")
        require(isinstance(ledger_id, str) and ledger_id.strip(), "ledger_identity")
        require(isinstance(actor_id, str) and actor_id.strip(), "ledger_actor")
        require(
            type(lock_timeout) in (int, float) and 0 <= lock_timeout <= 5,
            "lock_timeout",
        )
        require(
            protocol.sha256 == APPROVED_PROTOCOL_SHA256 and protocol.canonical,
            "protocol_identity",
        )
        require(isinstance(authority, ResolvedAuthority), "approval_loader_required")
        authority.recheck()
        for action in (
            ApprovalScope.PRODUCTION_LEDGER_REGISTRATION,
            ApprovalScope.CANDIDATE_SCORING,
        ):
            require(
                authority.authority.resolve(protocol.sha256, action).authorized,
                "approval_action",
            )
        self.ledger_id, self.actor_id = ledger_id, actor_id
        self.protocol, self.authority = protocol, authority.authority
        self._resolved_authority = authority
        self.lock_timeout = lock_timeout
        self._lock_fd = self._ledger_fd = None
        self._directory_identity = None

    def _stable(self):
        stable_path(self.directory)
        s = self.directory.stat()
        require(
            (s.st_dev, s.st_ino) == self._directory_identity, "ledger_directory_changed"
        )
        for name, fd in (
            ("ledger.lock", self._lock_fd),
            ("ledger.ndjson", self._ledger_fd),
        ):
            if fd is None:
                continue
            p = self.directory / name
            stable_path(p)
            actual = p.stat(follow_symlinks=False)
            held = os.fstat(fd)
            require(
                stat.S_ISREG(actual.st_mode)
                and (actual.st_dev, actual.st_ino) == (held.st_dev, held.st_ino),
                "ledger_path_changed",
            )

    @contextmanager
    def locked(self):
        """Bounded stable flock; never unlink its lockfile or infer PID ownership."""
        require(self._lock_fd is None, "ledger_nested_session")
        stable_path(self.directory)
        s = self.directory.stat()
        require(stat.S_ISDIR(s.st_mode), "ledger_directory")
        self._directory_identity = (s.st_dev, s.st_ino)
        path = self.directory / "ledger.lock"
        stable_path(path)
        self._lock_fd = os.open(
            path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
        try:
            require(
                stat.S_ISREG(os.fstat(self._lock_fd).st_mode), "ledger_lock_regular"
            )
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise BoundaryError("ledger_busy") from None
                    time.sleep(min(0.01, max(0, deadline - time.monotonic())))
            self._stable()
            os.fsync(self._lock_fd)
            _sync_directory(self.directory)
            yield self
        finally:
            if self._ledger_fd is not None:
                os.close(self._ledger_fd)
                self._ledger_fd = None
            os.close(self._lock_fd)  # flock releases even after worker death.
            self._lock_fd = None

    def _open(self, *, create=False):
        require(self._lock_fd is not None, "ledger_lock_required")
        self._stable()
        self._resolved_authority.recheck()
        if self._ledger_fd is None:
            path = self.directory / "ledger.ndjson"
            stable_path(path)
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            if create:
                flags |= os.O_CREAT | os.O_EXCL
            try:
                self._ledger_fd = os.open(path, flags, 0o600)
            except FileNotFoundError:
                raise BoundaryError("ledger_history_missing") from None
            if create:
                os.fsync(self._ledger_fd)
                _sync_directory(self.directory)
        self._stable()

    def read(self):
        """Verify the complete current bytes; never discard a malformed tail."""
        self._open()
        before = os.fstat(self._ledger_fd)
        require(
            stat.S_ISREG(before.st_mode) and before.st_size <= MAX_LEDGER_BYTES,
            "ledger_size",
        )
        os.lseek(self._ledger_fd, 0, os.SEEK_SET)
        chunks, remaining = [], MAX_LEDGER_BYTES + 1
        while remaining:
            part = os.read(self._ledger_fd, min(65536, remaining))
            if not part:
                break
            chunks.append(part)
            remaining -= len(part)
        raw = b"".join(chunks)
        after = os.fstat(self._ledger_fd)
        require(
            (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "ledger_changed",
        )
        self._stable()
        checked = verify_ledger(raw, protocol=self.protocol)
        require(checked.ok, "ledger_integrity_unknown")
        ledger = checked.ledger
        require(
            not ledger.records or ledger.ledger_id == self.ledger_id, "ledger_identity"
        )
        # No linked recovery or successor protocol support is asserted here.
        require(
            all(
                r.event_type
                not in (EventType.LEDGER_RECOVERY, EventType.PROTOCOL_AMENDED)
                for r in ledger.records
            ),
            "linked_history_adapter_required",
        )
        return ledger

    def _append(self, event, payload):
        old = self.read()
        rec = LedgerRecord(
            self.ledger_id,
            1,
            len(old.records),
            _now(),
            self.actor_id,
            event,
            payload,
            old.records[-1].sha256 if old.records else None,
        )
        raw = old.raw + rec.canonical_line + b"\n"
        require(
            len(raw) <= MAX_LEDGER_BYTES
            and verify_ledger(raw, protocol=self.protocol).ok,
            "ledger_append_invalid",
        )
        self._stable()
        os.lseek(self._ledger_fd, 0, os.SEEK_END)
        _write_all(self._ledger_fd, rec.canonical_line + b"\n")
        os.fsync(self._ledger_fd)
        require(self.read().raw == raw, "ledger_append_changed")
        return rec

    def registration_payload(self):
        resolution = self.authority.resolve(
            self.protocol.sha256, ApprovalScope.PRODUCTION_LEDGER_REGISTRATION
        )
        return {
            "protocol_sha256": self.protocol.sha256,
            "protocol_parent_sha256": self.protocol.parent_sha256,
            "legacy_parent_artifact_hashes_if_first_version": [],
            "parameter_changes_old_new_reason_evidence": [],
            "root_approval_artifact_sha256": resolution.root.sha256,
            "peer_approval_artifact_sha256": resolution.peer.sha256,
            "reference_inventory_sha256": self.protocol.reference_inventory_sha256,
            "block_definition_hashes": dict(self.protocol.block_definition_hashes),
        }

    def register(self, *, expected_snapshot: dict | None):
        """Explicit registration; None permits a new empty ledger, never recovery."""
        require(self._lock_fd is not None, "ledger_lock_required")
        if expected_snapshot is None:
            require(self._ledger_fd is None, "registration_new_file_required")
            # O_EXCL refuses existing/empty/recreated-path ambiguity. Callers
            # must supply a pinned existing snapshot for all repeated calls.
            self._open(create=True)
        ledger = self.read()
        if expected_snapshot is not None:
            require(
                canonical_bytes(_snapshot(ledger))
                == canonical_bytes(expected_snapshot),
                "ledger_snapshot_mismatch",
            )
        existing = [
            r for r in ledger.records if r.event_type is EventType.PROTOCOL_REGISTERED
        ]
        payload = self.registration_payload()
        if existing:
            require(
                len(existing) == 1 and dict(existing[0].payload) == payload,
                "registration_mismatch",
            )
        else:
            require(
                not ledger.records and expected_snapshot is None,
                "registration_history_missing",
            )
            self._append(EventType.PROTOCOL_REGISTERED, payload)
        return _snapshot(self.read())

    def _registered(self, ledger):
        registrations = [
            r for r in ledger.records if r.event_type is EventType.PROTOCOL_REGISTERED
        ]
        require(
            len(registrations) == 1 and registrations[0].sequence == 0,
            "protocol_unregistered",
        )
        result = resolve_action(
            self.protocol,
            authority=self.authority,
            action=ApprovalScope.CANDIDATE_SCORING,
            registration=registration_claim(registrations[0]),
        )
        require(result.authorized, "approval_action")

    def _interrupt(self, state, reason):
        return self._append(
            EventType.EVALUATION_INTERRUPTED,
            {
                "evaluation_id": state.evaluation_id,
                "attempt_record_sha256": state.attempt.sha256,
                "interruption_reason": reason,
                "known_output_hashes": None,
                "exposure_possible": True,
            },
        )

    def _exposure(self, candidate, ledger):
        history, identity = candidate.history, candidate.identity
        roots = set(history["tuning_roots"])
        populations = set(history["ancestor_populations"]) | {
            identity["population_content_sha256"]
        }
        exposed = set()
        for state in ledger.attempts.values():
            p = state.attempt.payload
            prior_roots = set(p["tuning_family_root_ids"])
            prior_pop = p["population_artifact_id_and_sha256"]["sha256"]
            # Renamed candidates and explicitly linked parent populations count.
            if roots & prior_roots or prior_pop in populations:
                exposed.update(p["all_derivative_reference_family_ids"])
        for event in history["external_exposures"]:
            if roots & set(event["tuning_roots"]) or populations & set(
                event["populations"]
            ):
                exposed.update(event["families"])
        # External ledger events without a complete typed ancestry adapter cannot
        # be silently ignored; refuse rather than guess that they are unrelated.
        require(
            not any(
                r.event_type is EventType.EXTERNAL_EXPOSURE_RECORDED
                for r in ledger.records
            ),
            "external_history_adapter_required",
        )
        audit = ExposureHistory(True, True, True, frozenset(exposed))
        return {
            b.id: exposure_status(b, history=audit)[0].value
            for b in self.protocol.blocks
        }

    @contextmanager
    def attempt(self, candidate: ResolvedCandidate, *, planned_scope: str):
        """Yield only after durable exposure; caller supplies no comparison here.

        All use of the returned permit and any future comparator must remain
        inside this context. Leaving without completion records interruption.
        A hard exit leaves a visible unfinished attempt for the next writer.
        """
        require(self._lock_fd is not None, "ledger_lock_required")
        require(
            isinstance(candidate, ResolvedCandidate), "candidate_admission_required"
        )
        require(
            isinstance(planned_scope, str) and planned_scope.strip(), "planned_scope"
        )
        candidate.recheck()
        ledger = self.read()
        self._registered(ledger)
        require(
            canonical_bytes(candidate.history["expected_snapshot"])
            == canonical_bytes(_snapshot(ledger)),
            "ledger_snapshot_mismatch",
        )
        for state in ledger.attempts.values():
            if state.unfinished:
                self._interrupt(state, "previous_writer_left_unfinished_attempt")
        ledger = self.read()
        statuses = self._exposure(candidate, ledger)
        before = _snapshot(ledger)
        identity = candidate.identity
        evaluation_id = new_evaluation_id(ledger.attempts)
        slots = candidate.binding.slots
        mappings = {name: slots[name].sha256 for name in MAPPING_SLOTS if name in slots}
        payload = {
            "evaluation_id": evaluation_id,
            "protocol_sha256": self.protocol.sha256,
            "candidate_id": identity["candidate_id"],
            "candidate_manifest_sha256": candidate.manifest_sha256,
            "population_artifact_id_and_sha256": {
                "artifact_id": identity["population_artifact_id"],
                "sha256": identity["population_content_sha256"],
            },
            "population_semantic_snapshot_sha256": identity[
                "population_semantic_snapshot_sha256"
            ],
            "producer_run_id": identity["producer_run_id"],
            "model_identity_or_not_invoked": identity["model_identity_or_not_invoked"],
            "tuning_ancestry_manifest_sha256": slots[
                "tuning_ancestry_manifest_sha256"
            ].sha256,
            "tuning_family_root_ids": candidate.history["tuning_roots"],
            "complete_ancestry_audit_sha256": candidate.artifact_pins[
                candidate_history_id(candidate)
            ].sha256,
            "reservation_audit_sha256": slots[
                "reservation_audit_artifact_sha256"
            ].sha256,
            "reference_block_ids_and_hashes": dict(
                self.protocol.block_definition_hashes
            ),
            "all_derivative_reference_family_ids": sorted(
                {f for b in self.protocol.blocks for f in b.exposure_families}
            ),
            "mapping_geography_reducer_artifact_hashes": mappings,
            "ledger_snapshot_before_attempt": before,
            "prior_exposure_classification_per_block": statuses,
            "planned_scope": planned_scope,
            "attempt_state": "started",
        }
        rec = self._append(EventType.EVALUATION_ATTEMPT, payload)
        permit = Permit(self, rec)
        try:
            yield permit
        except BaseException as original:
            try:
                state = self.read().attempts[evaluation_id]
                if state.unfinished:
                    self._interrupt(state, "comparison_or_attachment_raised")
            except BaseException as secondary:
                original.add_note(
                    "Failed to record interruption: "
                    + type(secondary).__name__
                    + "; retain unfinished exposure."
                )
            raise
        finally:
            permit.active = False
            # No exception masking: the except branch already attempted cleanup.
            import sys

            if sys.exc_info()[0] is None:
                state = self.read().attempts[evaluation_id]
                if state.unfinished:
                    self._interrupt(state, "context_left_without_completion")

    def _immutable(self, name, raw):
        require("/" not in name and name not in (".", ".."), "immutable_name")
        self._stable()
        path = self.directory / name
        stable_path(path)
        try:
            fd = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
        except FileExistsError:
            require(
                read_pinned(FilePin(path, len(raw), digest(raw))) == raw,
                "immutable_mismatch",
            )
            # An earlier crash/fsync failure may leave identical bytes without
            # durable data or a durable directory entry. Repair both on reuse.
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                held, located = os.fstat(fd), path.stat(follow_symlinks=False)
                require(
                    stat.S_ISREG(held.st_mode)
                    and (held.st_dev, held.st_ino) == (located.st_dev, located.st_ino),
                    "immutable_path_changed",
                )
                os.fsync(fd)
            finally:
                os.close(fd)
        else:
            try:
                _write_all(fd, raw)
                os.fsync(fd)
            finally:
                os.close(fd)
        _sync_directory(self.directory)
        self._stable()
        require(
            read_pinned(FilePin(path, len(raw), digest(raw))) == raw,
            "immutable_changed",
        )
        return digest(raw)

    def attach(self, evaluation_id):
        """Idempotently reconstruct a terminal envelope; never rerun comparison."""
        ledger = self.read()
        self._registered(ledger)
        require(evaluation_id in ledger.attempts, "evaluation_unknown")
        state = ledger.attempts[evaluation_id]
        require(
            state.terminal is not None
            and state.terminal.event_type is EventType.EVALUATION_COMPLETION,
            "evaluation_not_completed",
        )
        p, c = state.attempt.payload, state.terminal.payload
        before_raw = b"".join(
            r.canonical_line + b"\n" for r in ledger.records[: state.attempt.sequence]
        )
        actual_before = verify_ledger(before_raw, protocol=self.protocol).ledger
        require(
            canonical_bytes(p["ledger_snapshot_before_attempt"])
            == canonical_bytes(_snapshot(actual_before)),
            "attempt_snapshot_mismatch",
        )
        result_path = self.directory / (c["result_payload_sha256"] + ".result.json")
        stable_path(result_path)
        # The completion digest is authority; bounded size is read from regular
        # file metadata, then descriptor/path consistency is checked by read_pinned.
        result_raw = read_pinned(
            FilePin(result_path, result_path.stat().st_size, c["result_payload_sha256"])
        )
        result = strict_load(result_raw)
        require(canonical_bytes(result) == result_raw, "result_canonical")
        expected = _result_identity(p, state.attempt.sha256)
        require(
            set(result)
            == set(expected) | {"block_verdicts_and_scope_reasons", "terminal_state"},
            "result_fields",
        )
        require(
            canonical_bytes({k: result[k] for k in expected})
            == canonical_bytes(expected),
            "result_identity",
        )
        require(
            result["block_verdicts_and_scope_reasons"]
            == c["block_verdicts_and_scope_reasons"]
            and result["terminal_state"] == c["terminal_state"],
            "result_completion_mismatch",
        )
        require(
            c["protocol_sha256"] == p["protocol_sha256"]
            and c["candidate_id"] == p["candidate_id"]
            and c["population_sha256"]
            == p["population_artifact_id_and_sha256"]["sha256"],
            "completion_identity",
        )
        prefix = b"".join(
            r.canonical_line + b"\n"
            for r in ledger.records[: state.terminal.sequence + 1]
        )
        verified = verify_ledger(prefix, protocol=self.protocol).ledger
        self._immutable(digest(prefix) + ".snapshot.ndjson", prefix)
        envelope = dict(
            expected,
            evaluation_completion_record_sha256=state.terminal.sha256,
            ledger_snapshot_through_completion=_snapshot(verified),
            result_payload_sha256=c["result_payload_sha256"],
            scope_and_block_verdicts=result["block_verdicts_and_scope_reasons"],
        )
        envelope["exposure_status_per_block"] = p[
            "prior_exposure_classification_per_block"
        ]
        raw = canonical_bytes(envelope)
        self._immutable(evaluation_id + ".quality.json", raw)
        return strict_load(raw)


def candidate_history_id(candidate):
    # Manifest was already decoded and pinned; reading it again also refuses
    # mutation between admission and attempt. This returns identity metadata only.
    pin = candidate.manifest_pin
    return strict_load(read_pinned(pin))["history_artifact_id"]


def _result_identity(p, attempt_sha):
    return {
        "protocol_sha256": p["protocol_sha256"],
        "candidate_id": p["candidate_id"],
        "candidate_manifest_sha256": p["candidate_manifest_sha256"],
        "population_artifact_id_and_sha256": p["population_artifact_id_and_sha256"],
        "population_semantic_snapshot_sha256": p["population_semantic_snapshot_sha256"],
        "producer_run_id": p["producer_run_id"],
        "model_identity_or_not_invoked": p["model_identity_or_not_invoked"],
        "reference_block_hashes": p["reference_block_ids_and_hashes"],
        "mapping_geography_reducer_hashes": p[
            "mapping_geography_reducer_artifact_hashes"
        ],
        "tuning_ancestry_manifest_sha256": p["tuning_ancestry_manifest_sha256"],
        "evaluation_attempt_record_sha256": attempt_sha,
        "ledger_snapshot_before_attempt": p["ledger_snapshot_before_attempt"],
    }


class Permit:
    """Identity-only handle valid during one durable attempt's locked context."""

    def __init__(self, owner, record):
        self.owner, self.record, self.active = owner, record, True

    @property
    def identity(self):
        require(self.active, "permit_inactive")
        return strict_load(canonical_bytes(dict(self.record.payload)))

    def finish(self, block_verdicts: dict, *, terminal_state: str):
        """Attach nonnumeric verdict metadata from a future approved comparator.

        This method computes no verdict. Failed runs must explicitly retain
        uncomputed blocks; a completed run cannot contain uncomputed blocks.
        """
        require(self.active, "permit_inactive")
        require(terminal_state in ("completed", "failed"), "terminal_state")
        require(
            isinstance(block_verdicts, dict)
            and set(block_verdicts) == set(self.owner.protocol.block_definition_hashes),
            "result_block_roster",
        )
        for value in block_verdicts.values():
            fields(value, {"verdict", "reasons"}, "result_block_fields")
            require(
                value["verdict"]
                in {v.value for v in BenchmarkVerdict} | {"uncomputed"},
                "result_verdict",
            )
            require(
                terminal_state == "failed" or value["verdict"] != "uncomputed",
                "completed_uncomputed",
            )
            strings(value["reasons"], "result_reasons")
            require(
                set(value["reasons"]) <= {r.value for r in Reason}, "result_reasons"
            )
        state = self.owner.read().attempts[self.record.payload["evaluation_id"]]
        require(state.unfinished, "attempt_already_terminal")
        p = self.record.payload
        result = dict(
            _result_identity(p, self.record.sha256),
            block_verdicts_and_scope_reasons=block_verdicts,
            terminal_state=terminal_state,
        )
        raw = canonical_bytes(result)
        sha = self.owner._immutable(digest(raw) + ".result.json", raw)
        self.owner._append(
            EventType.EVALUATION_COMPLETION,
            {
                "evaluation_id": state.evaluation_id,
                "attempt_record_sha256": self.record.sha256,
                "protocol_sha256": p["protocol_sha256"],
                "candidate_id": p["candidate_id"],
                "population_sha256": p["population_artifact_id_and_sha256"]["sha256"],
                "result_payload_sha256": sha,
                "block_verdicts_and_scope_reasons": block_verdicts,
                "completed_at": _now(),
                "terminal_state": terminal_state,
            },
        )
        return self.owner.attach(state.evaluation_id)
