"""A telemetry service decides only its own run's fate from its own credential.

Every emitter service on a host shares one spool, which therefore also holds
other builds' runs. These tests run several producers against one spool and a
fake collector that follows the hosted collector's authorization model
(PolicyEngine/calibration-diagnostics, ``telemetry-service``): the token
exchange refuses a non-member with 403 and a bad token with 401; the first
Hugging Face user to register a run owns it and anyone else gets 403; an
unknown run's events get 404, and a batch the collector cannot accept 422.

Invariants (each checked below; the first four also after every step of a
property test that runs services in any order and lets one run while
another's collector request is in flight):

- No service makes another producer's run local-only because of its own
  credential (missing, refused, or not the run's owner). Another producer's
  run goes local-only only after a collector answer about the run itself.
- While a producer's service is alive, no other service sends any request for
  that producer's run.
- A run goes local-only only when the deciding service's latest collector
  answer is the refusal that justifies it. Two cases have no fresh answer: a
  missing credential, and a credential the collector already refused to that
  service, which is exchanged once and not again.
- Local-only is final: no request about a run reaches the collector after it
  goes local-only, even from a service that listed the run before.
- Once every service has gone, a run left pending is delivered in full by a
  service whose credential owns it, including a service that was refused for
  the run under an earlier login.
"""

from __future__ import annotations

import errno
import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock
from urllib.parse import urlsplit

import pytest
from hypothesis import HealthCheck, event, example, given, settings
from hypothesis import strategies as st

from microcosm.build.telemetry_emitter import LocalTelemetryEmitter, TelemetryRun
from microcosm.build.telemetry_emitter_service import (
    CollectorDelivery,
    EmitterService,
    EventSpool,
)
from microcosm.build.telemetry_emitter_service import collector as collector_module
from microcosm.build.telemetry_emitter_service import leases as leases_module
from microcosm.build.telemetry_emitter_service import main as main_module
from microcosm.build.telemetry_emitter_service.constants import (
    DEFAULT_TOKEN_LIFETIME_SECONDS,
    LOCAL_ONLY_MISSING_CREDENTIAL,
    LOCAL_ONLY_REJECTED_COLLECTOR_AUTHORIZATION,
    LOCAL_ONLY_REJECTED_CREDENTIAL,
    LOCAL_ONLY_REJECTED_EVENTS,
    LOCAL_ONLY_REJECTED_REGISTRATION,
    MAX_RETRY_SECONDS,
    ORPHAN_IDLE_SECONDS,
    OWN_LEASE_UNAVAILABLE_MESSAGE,
    RUN_REGISTRATION_PATH,
    TOKEN_EXCHANGE_PATH,
)
from microcosm.build.telemetry_emitter_service.leases import (
    OwnLeaseUnavailableError,
    ProducerLeases,
)
from microcosm.build.telemetry_protocol import BUILD_COMPLETED_MESSAGE

MEMBER = "hf-member:"
OUTSIDER = "hf-outsider"
EXPIRED = "hf-expired"
CREDENTIALS = (None, OUTSIDER, EXPIRED, MEMBER + "alice", MEMBER + "bob")
DEVELOPMENT_COLLECTOR = "http://127.0.0.1:8080"
EVENT = {
    "timestamp": "2026-10-09T00:00:00+00:00",
    "event_type": "stage",
    "stage_id": "compile",
    "status": "started",
    "message": None,
    "details": {},
}

Run = tuple[str, str]


def _registration(run_id: str, producer_id: str) -> dict[str, Any]:
    return TelemetryRun(
        run_id=run_id,
        country_code="US",
        pipeline="foreign-runs",
        producer_id=producer_id,
    ).as_registration()


def _run_of(registration: dict[str, Any]) -> Run:
    return registration["run_id"], registration["producer_id"]


class FakeCollector:
    """The hosted collector's answers, as its authorization model gives them."""

    def __init__(
        self,
        rejected_run_ids: set[str] | None = None,
        conflicted_run_ids: set[str] | None = None,
    ) -> None:
        self.rejected_run_ids = set(rejected_run_ids or ())
        self.owners: dict[str, str] = {}
        # A conflicted run id was registered before with other metadata, so
        # whoever owns it gets 409 for this registration.
        self.metadata: dict[str, dict[str, Any]] = {
            run_id: {"pipeline": "something-else"}
            for run_id in conflicted_run_ids or ()
        }
        self.producers: set[Run] = set()
        self.accepted: dict[Run, set[str]] = {}
        # (caller's run, kind, run asked about or None, status)
        self.requests: list[tuple[Run, str, Run | None, int]] = []
        # (caller's run, Hugging Face token) for every refused exchange, and
        # where in ``requests`` the first refusal of each falls
        self.refused: list[tuple[Run, str]] = []
        self.refused_at: dict[tuple[Run, str], int] = {}
        # Runs once, while the next request is in flight.
        self.during_next_request: Callable[[], None] | None = None

    @property
    def conflicted_run_ids(self) -> set[str]:
        return {
            run_id
            for run_id, metadata in self.metadata.items()
            if metadata.get("pipeline") == "something-else"
        }

    def post(self, caller: Run, url: str, payload, token: str) -> tuple[int, dict]:
        if self.during_next_request is not None:
            during, self.during_next_request = self.during_next_request, None
            during()
        path = urlsplit(url).path
        if path == TOKEN_EXCHANGE_PATH:
            if not token.startswith(MEMBER):
                self.refused.append((caller, token))
                self.refused_at.setdefault((caller, token), len(self.requests))
            if token.startswith(MEMBER):
                user = token.removeprefix(MEMBER)
                return self._answer(
                    caller,
                    "exchange",
                    None,
                    200,
                    {"access_token": f"session:{user}", "expires_in": 3600},
                )
            return self._answer(
                caller, "exchange", None, 403 if token == OUTSIDER else 401, {}
            )
        user = token.removeprefix("session:")
        if path == RUN_REGISTRATION_PATH:
            run = (payload["run_id"], payload["producer_id"])
            if self.owners.setdefault(run[0], user) != user:
                return self._answer(caller, "register", run, 403, {})
            metadata = {k: v for k, v in payload.items() if k != "producer_id"}
            if self.metadata.setdefault(run[0], metadata) != metadata:
                return self._answer(caller, "register", run, 409, {})
            self.producers.add(run)
            return self._answer(caller, "register", run, 201, {"registered": True})
        run_id = path.split("/")[3]
        events = payload["events"]
        run = (run_id, events[0]["producer_id"])
        if run_id not in self.owners:
            return self._answer(caller, "events", run, 404, {})
        if self.owners[run_id] != user or run not in self.producers:
            return self._answer(caller, "events", run, 403, {})
        if run_id in self.rejected_run_ids:
            return self._answer(caller, "events", run, 422, {})
        self.accepted.setdefault(run, set()).update(e["event_id"] for e in events)
        return self._answer(caller, "events", run, 202, {"accepted": len(events)})

    def _answer(self, caller, kind, run, status, body):
        self.requests.append((caller, kind, run, status))
        return status, body

    def latest(self, caller: Run, before: int) -> tuple[str, Run | None, int] | None:
        """The caller's last request among the first ``before``: kind, run, status."""

        for who, kind, about, status in reversed(self.requests[:before]):
            if who == caller:
                return kind, about, status
        return None

    def asked(self, caller: Run, kind: str, run: Run | None = None) -> list[int]:
        return [
            status
            for who, what, about, status in self.requests
            if who == caller and what == kind and (run is None or about == run)
        ]


@dataclass
class Producer:
    """One build's emitter service: its own spool connection, lease and login."""

    registration: dict[str, Any]
    credential: str | None
    spool: EventSpool
    delivery: CollectorDelivery
    alive: bool = True
    appended: list[str] = field(default_factory=list)

    @property
    def run(self) -> Run:
        return _run_of(self.registration)


class Host:
    """Several builds on one machine, sharing one spool and one collector."""

    def __init__(self, directory: Path, collector: FakeCollector) -> None:
        self.spool_path = directory / "events.sqlite3"
        self.collector = collector
        self.producers: list[Producer] = []
        # (caller's run, run made local-only, reason, caller's login then,
        # number of collector requests by then)
        self.local_only: list[tuple[Run, Run, str, str | None, int]] = []
        # Requests a service sent about another producer's run while it lived.
        self.intrusions: list[tuple[Run, str, Run]] = []
        # Each flush comes after any retry delay a service set itself, as the
        # service's one-second worker ticks would reach it.
        self.clock = 0.0

    def start(
        self,
        credential: str | None,
        run_id: str | None = None,
        producer_id: str | None = None,
    ) -> Producer:
        index = len(self.producers)
        registration = _registration(
            run_id or f"run-{index}", producer_id or f"producer-{index}"
        )
        spool = EventSpool(self.spool_path)
        # As in the service's main(): the lease is taken before registration.
        delivery = CollectorDelivery(
            spool, registration, development_collector_url=DEVELOPMENT_COLLECTOR
        )
        spool.register(registration)
        producer = Producer(registration, credential, spool, delivery)
        make_local_only = spool.make_local_only

        def recorded(run_id: str, producer_id: str, reason: str) -> None:
            self.local_only.append(
                (
                    producer.run,
                    (run_id, producer_id),
                    reason,
                    producer.credential,
                    len(self.collector.requests),
                )
            )
            make_local_only(run_id, producer_id, reason)

        spool.make_local_only = recorded
        self.producers.append(producer)
        return producer

    def emit(self, producer: Producer, count: int = 1) -> None:
        assert producer.alive
        for _ in range(count):
            event = producer.spool.append(producer.registration, EVENT)
            producer.appended.append(event["event_id"])

    def flush(self, producer: Producer) -> bool:
        assert producer.alive

        def post(url, payload, token, *, timeout=5.0):
            status, body = self.collector.post(producer.run, url, payload, token)
            about = self.collector.requests[-1][2]
            owner = self.producer_of(about) if about is not None else None
            if owner is not None and owner is not producer and owner.alive:
                self.intrusions.append((producer.run, url, about))
            return status, body

        self.clock += MAX_RETRY_SECONDS + 1
        with (
            mock.patch.object(
                collector_module, "_huggingface_token", lambda: producer.credential
            ),
            mock.patch.object(collector_module, "_http_post", post),
            mock.patch.object(
                collector_module, "time", SimpleNamespace(monotonic=lambda: self.clock)
            ),
        ):
            return producer.delivery.flush_once()

    def kill(self, producer: Producer) -> None:
        producer.alive = False
        producer.delivery.close()

    def producer_of(self, run: Run) -> Producer | None:
        return next((p for p in self.producers if p.run == run), None)

    def states(self) -> dict[Run, tuple[str, str | None]]:
        with closing(sqlite3.connect(self.spool_path)) as connection:
            return {
                (run_id, producer_id): (state, reason)
                for run_id, producer_id, state, reason in connection.execute(
                    "SELECT run_id, producer_id, upload_state, local_only_reason "
                    "FROM telemetry_runs"
                )
            }

    def queued(self, run: Run) -> int:
        with closing(sqlite3.connect(self.spool_path)) as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM telemetry_events "
                "WHERE run_id = ? AND producer_id = ?",
                run,
            ).fetchone()[0]

    def close(self) -> None:
        for producer in self.producers:
            producer.delivery.close()
            producer.spool._engine.dispose()


@pytest.fixture
def host(tmp_path):
    host = Host(tmp_path, FakeCollector())
    yield host
    host.close()


def _assert_own_credential_only(host: Host) -> None:
    """The safety invariants, over everything recorded so far."""

    assert host.intrusions == []
    rejected = host.collector.rejected_run_ids
    conflicted = host.collector.conflicted_run_ids
    for caller, run, reason, credential, at in host.local_only:
        # Local-only is final: nothing about the run reaches the collector after.
        later = host.collector.requests[at:]
        assert all(about != run for _, _, about, _ in later), (run, later)
        # What this service last heard from the collector before deciding.
        latest = host.collector.latest(caller, at)
        if caller != run:
            # Another producer's run: only the collector's answer about the
            # run itself, never one about this service's credential.
            if reason == LOCAL_ONLY_REJECTED_REGISTRATION:
                assert run[0] in conflicted, (caller, run)
                assert latest == ("register", run, 409), (caller, run, latest)
            else:
                assert reason == LOCAL_ONLY_REJECTED_EVENTS, (caller, run, reason)
                assert run[0] in rejected, (caller, run)
                assert latest == ("events", run, 422), (caller, run, latest)
            continue
        if reason == LOCAL_ONLY_MISSING_CREDENTIAL:
            assert credential is None
        elif reason == LOCAL_ONLY_REJECTED_CREDENTIAL:
            # The collector refused this very login to this service, just now
            # or earlier: a refused login is not exchanged a second time.
            first_refused = host.collector.refused_at.get((caller, credential), at)
            assert first_refused < at, (caller, credential)
        elif reason == LOCAL_ONLY_REJECTED_REGISTRATION:
            assert latest in (("register", run, 403), ("register", run, 409)), (
                run,
                latest,
            )
        elif reason == LOCAL_ONLY_REJECTED_COLLECTOR_AUTHORIZATION:
            assert latest == ("events", run, 403), (run, latest)
        else:
            assert reason == LOCAL_ONLY_REJECTED_EVENTS
            assert latest in (("events", run, 404), ("events", run, 422)), (
                run,
                latest,
            )
    # A refused credential is exchanged once per service, not on every pass.
    refused = host.collector.refused
    assert len(set(refused)) == len(refused), refused


def test_a_service_without_a_credential_leaves_a_live_build_run_pending(host):
    """The 2026-10-09 e7 reproduction: build B's worker had no credential and
    made build A's run local-only."""

    build_a = host.start(MEMBER + "alice")
    build_b = host.start(None)
    host.emit(build_a)
    host.emit(build_b)

    host.flush(build_b)

    assert host.states()[build_a.run] == ("pending", None)
    assert host.states()[build_b.run] == ("local_only", LOCAL_ONLY_MISSING_CREDENTIAL)
    assert host.collector.requests == []
    host.flush(build_a)
    assert host.collector.accepted[build_a.run] == set(build_a.appended)
    _assert_own_credential_only(host)


def test_a_run_made_local_only_after_it_was_listed_is_not_adopted(host):
    """A pass lists the pending runs, then waits on the collector. Meanwhile
    another build's service makes its own run local-only and exits, freeing
    its lease. Found by the two-service test below."""

    build_a = host.start(MEMBER + "alice")
    build_b = host.start(None)
    host.emit(build_a)
    host.emit(build_b)

    def build_b_finishes() -> None:
        host.flush(build_b)
        host.kill(build_b)

    host.collector.during_next_request = build_b_finishes
    host.flush(build_a)

    assert host.states()[build_b.run] == ("local_only", LOCAL_ONLY_MISSING_CREDENTIAL)
    assert all(about != build_b.run for _, _, about, _ in host.collector.requests)
    assert host.collector.accepted[build_a.run] == set(build_a.appended)
    _assert_own_credential_only(host)


@pytest.mark.parametrize("credential", CREDENTIALS)
def test_a_live_build_run_gets_no_request_from_another_service(host, credential):
    build_a = host.start(MEMBER + "alice")
    build_b = host.start(credential)
    host.emit(build_a, 3)
    host.emit(build_b)

    for _ in range(3):
        host.flush(build_b)

    assert all(about != build_a.run for _, _, about, _ in host.collector.requests)
    assert host.states()[build_a.run] == ("pending", None)
    assert host.queued(build_a.run) == 3
    _assert_own_credential_only(host)


@pytest.mark.parametrize(
    ("credential", "own_reason", "exchanges"),
    [
        (None, LOCAL_ONLY_MISSING_CREDENTIAL, 0),
        (OUTSIDER, LOCAL_ONLY_REJECTED_CREDENTIAL, 1),
        (EXPIRED, LOCAL_ONLY_REJECTED_CREDENTIAL, 1),
    ],
)
def test_an_orphan_waits_for_a_credential_that_can_deliver_it(
    host, capsys, credential, own_reason, exchanges
):
    build_a = host.start(MEMBER + "alice")
    host.emit(build_a, 3)
    host.kill(build_a)
    build_b = host.start(credential)
    host.emit(build_b)

    for _ in range(4):
        host.flush(build_b)

    assert host.states()[build_a.run] == ("pending", None)
    assert host.states()[build_b.run] == ("local_only", own_reason)
    assert len(host.collector.asked(build_b.run, "exchange")) == exchanges
    assert capsys.readouterr().err.count("local-only") == 1

    rescuer = host.start(MEMBER + "alice")
    host.flush(rescuer)
    assert host.collector.accepted[build_a.run] == set(build_a.appended)
    assert host.queued(build_a.run) == 0
    assert host.states()[build_a.run] == ("pending", None)
    _assert_own_credential_only(host)


def test_an_orphan_owned_by_another_user_is_passed_over_not_made_local_only(host):
    build_a = host.start(MEMBER + "alice")
    host.emit(build_a, 2)
    host.flush(build_a)
    host.emit(build_a, 2)
    host.kill(build_a)
    build_b = host.start(MEMBER + "bob")
    host.emit(build_b)

    for _ in range(3):
        host.flush(build_b)

    # Bob may not register Alice's run; asking once settles it for Bob.
    assert host.collector.asked(build_b.run, "register", build_a.run) == [403]
    assert host.states()[build_a.run] == ("pending", None)
    assert host.queued(build_a.run) == 2
    assert host.collector.accepted[build_b.run] == set(build_b.appended)

    rescuer = host.start(MEMBER + "alice")
    host.flush(rescuer)
    assert host.collector.accepted[build_a.run] == set(build_a.appended)
    _assert_own_credential_only(host)


@pytest.mark.parametrize("refused", [OUTSIDER, EXPIRED])
def test_a_passed_over_orphan_is_offered_again_once_the_login_changes(host, refused):
    """Once a service's own run is local-only and every other pending run is
    passed over, no delivery reads the login any more. The pass itself must
    notice the new login, or the orphan waits for another service."""

    build_a = host.start(MEMBER + "alice")
    host.emit(build_a)
    host.flush(build_a)
    host.emit(build_a, 2)
    host.kill(build_a)
    build_b = host.start(refused)
    host.emit(build_b)
    for _ in range(2):
        host.flush(build_b)
    assert host.states()[build_b.run] == (
        "local_only",
        LOCAL_ONLY_REJECTED_CREDENTIAL,
    )
    assert host.queued(build_a.run) == 2

    build_b.credential = MEMBER + "alice"
    host.flush(build_b)

    assert host.collector.accepted[build_a.run] == set(build_a.appended)
    assert host.queued(build_a.run) == 0
    # The refused login was exchanged once, and so was the new one.
    assert host.collector.asked(build_b.run, "exchange") == [
        403 if refused == OUTSIDER else 401,
        200,
    ]
    _assert_own_credential_only(host)


def test_a_passed_over_orphan_waits_out_the_session_of_the_login_that_was_refused(
    host,
):
    """A session the collector issued stays in use until it expires, so a new
    login takes effect then, as it does for the service's own run."""

    build_a = host.start(MEMBER + "alice")
    host.emit(build_a)
    host.flush(build_a)
    host.emit(build_a, 2)
    host.kill(build_a)
    build_b = host.start(MEMBER + "bob")
    host.emit(build_b)
    host.flush(build_b)
    assert host.collector.asked(build_b.run, "register", build_a.run) == [403]

    build_b.credential = MEMBER + "alice"
    host.flush(build_b)
    # Bob's session is still what this service sends, and it was refused.
    assert host.collector.asked(build_b.run, "register", build_a.run) == [403]
    assert host.queued(build_a.run) == 2

    host.clock += DEFAULT_TOKEN_LIFETIME_SECONDS
    host.flush(build_b)

    assert host.collector.asked(build_b.run, "register", build_a.run) == [403, 201]
    assert host.collector.accepted[build_a.run] == set(build_a.appended)
    _assert_own_credential_only(host)


def test_a_refusal_for_one_run_does_not_pass_over_a_run_with_similar_ids(host):
    """Run and producer ids may contain colons, so two runs can share the text
    ``run_id:producer_id``. From review: Bob was refused Alice's ``a:b``/``c``
    and then never offered his own ``a``/``b:c``."""

    build_a = host.start(MEMBER + "alice", run_id="a:b", producer_id="c")
    host.emit(build_a)
    host.flush(build_a)
    host.emit(build_a)
    host.kill(build_a)
    build_b = host.start(MEMBER + "bob", run_id="a", producer_id="b:c")
    host.emit(build_b)
    host.flush(build_b)
    host.emit(build_b)
    host.kill(build_b)
    adopter = host.start(MEMBER + "bob")

    for _ in range(2):
        host.flush(adopter)

    assert host.collector.asked(adopter.run, "register", build_a.run) == [403]
    assert host.collector.accepted[build_b.run] == set(build_b.appended)
    assert host.queued(build_b.run) == 0
    assert host.queued(build_a.run) == 1
    _assert_own_credential_only(host)


def test_registering_one_run_does_not_count_for_a_run_with_similar_ids(host):
    """Both orphans are Alice's and neither was registered. The adopter must
    register each: the second is not registered just because the first's ids
    join to the same text."""

    orphans = [
        host.start(MEMBER + "alice", run_id="a:b", producer_id="c"),
        host.start(MEMBER + "alice", run_id="a", producer_id="b:c"),
    ]
    for orphan in orphans:
        host.emit(orphan)
        host.kill(orphan)
    adopter = host.start(MEMBER + "alice")

    for _ in range(2):
        host.flush(adopter)

    for orphan in orphans:
        assert host.collector.asked(adopter.run, "register", orphan.run) == [201]
        assert host.collector.accepted[orphan.run] == set(orphan.appended)
        assert host.states()[orphan.run] == ("pending", None)
    _assert_own_credential_only(host)


def _refuse_locks(patch, refusals: int | None) -> list[int]:
    """Make the next ``refusals`` lock attempts fail (all of them, if ``None``)
    as a system that has run out of locks would."""

    real = leases_module.fcntl
    attempts: list[int] = []

    def flock(descriptor: int, operation: int) -> None:
        attempts.append(descriptor)
        if refusals is None or len(attempts) <= refusals:
            raise OSError(errno.ENOLCK, "No locks available")
        real.flock(descriptor, operation)

    patch.setattr(
        leases_module,
        "fcntl",
        SimpleNamespace(flock=flock, LOCK_EX=real.LOCK_EX, LOCK_NB=real.LOCK_NB),
    )
    return attempts


def test_a_lock_refused_once_at_startup_does_not_expose_the_run(host, monkeypatch):
    """From review: the system refused build A's lock once, so A ran without
    its lease, and the attempt left a lease file nobody held. Build B took
    that for an orphan's and registered A's live run under its own login."""

    with monkeypatch.context() as patch:
        attempts = _refuse_locks(patch, 1)
        build_a = host.start(MEMBER + "alice")
    build_b = host.start(MEMBER + "bob")
    host.emit(build_a)
    host.emit(build_b)

    for _ in range(2):
        host.flush(build_b)

    assert all(about != build_a.run for _, _, about, _ in host.collector.requests)
    host.flush(build_a)
    assert host.collector.accepted[build_a.run] == set(build_a.appended)
    assert host.states()[build_a.run] == ("pending", None)
    _assert_own_credential_only(host)
    # A took its lease on the attempt after the refusal.
    assert len(attempts) == 2


@pytest.mark.parametrize("obstacle", ["locks_refused", "lease_held"])
def test_a_service_that_cannot_take_its_lease_does_not_start(
    tmp_path, monkeypatch, capsys, obstacle
):
    """A lease file nobody holds reads as an exited producer's, and a failed
    attempt can leave one. So a service that cannot take its lease serves
    nothing: its run never has an event another service could adopt."""

    registration = _registration("run-a", "producer-a")
    spool_path = tmp_path / "events.sqlite3"
    arguments = [
        *("--socket", str(tmp_path / "service.sock")),
        *("--spool", str(spool_path)),
        *("--development-collector-url", DEVELOPMENT_COLLECTOR),
        *("--registration-json", json.dumps(registration)),
        *("--parent-pid", str(os.getpid())),
    ]
    served: list[EmitterService] = []
    monkeypatch.setattr(EmitterService, "run", lambda self: served.append(self))
    spools: list[EventSpool] = []

    def opened(path: Path) -> EventSpool:
        spools.append(EventSpool(path))
        return spools[-1]

    monkeypatch.setattr(main_module, "EventSpool", opened)
    monkeypatch.setattr(collector_module, "OWN_LEASE_TIMEOUT_SECONDS", 0.05)
    try:
        with monkeypatch.context() as patch:
            if obstacle == "locks_refused":
                _refuse_locks(patch, None)
                holder = None
            else:
                # An adopter is still delivering an earlier run with these ids.
                holder = ProducerLeases.beside(spool_path).try_acquire(
                    *_run_of(registration), create=True
                )
                assert holder is not None
            with pytest.raises(OwnLeaseUnavailableError):
                CollectorDelivery(
                    opened(spool_path),
                    registration,
                    development_collector_url=DEVELOPMENT_COLLECTOR,
                )
            assert main_module.main(arguments) == 1
            assert served == []
            assert capsys.readouterr().err.strip() == OWN_LEASE_UNAVAILABLE_MESSAGE
            if holder is not None:
                holder.release()

        # With the obstacle gone, the same service starts.
        assert main_module.main(arguments) == 0
        assert [service.registration for service in served] == [registration]
        assert capsys.readouterr().err == ""
    finally:
        for spool in spools:
            spool._engine.dispose()


def test_an_own_lease_waits_out_refused_locks_until_its_deadline(tmp_path, monkeypatch):
    leases = ProducerLeases(tmp_path / "leases")
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    def hold():
        return leases.hold(
            "run-a",
            "producer-a",
            timeout_seconds=1,
            clock=lambda: now[0],
            sleep=sleep,
        )

    with monkeypatch.context() as patch:
        attempts = _refuse_locks(patch, 3)
        lease = hold()
        assert lease is not None
        assert len(attempts) == 4
        lease.release()

    with monkeypatch.context() as patch:
        attempts = _refuse_locks(patch, None)
        assert hold() is None
        # It kept asking until the deadline rather than giving up at once.
        assert len(attempts) > 50

    # The refused attempts left nothing locked.
    lease = leases.try_acquire("run-a", "producer-a", create=False)
    assert lease is not None
    lease.release()


@pytest.mark.parametrize("answer", ["unacceptable_events", "registration_conflict"])
def test_an_orphan_the_collector_refuses_goes_local_only_without_a_warning_here(
    host, capsys, answer
):
    build_a = host.start(MEMBER + "alice")
    host.emit(build_a)
    host.kill(build_a)
    if answer == "unacceptable_events":
        host.collector.rejected_run_ids.add(build_a.run[0])
        reason = LOCAL_ONLY_REJECTED_EVENTS
    else:
        # The run id is already Alice's with other metadata: a 409 reaches
        # only the owner, and every owner would get it.
        host.collector.owners[build_a.run[0]] = "alice"
        host.collector.metadata[build_a.run[0]] = {"pipeline": "something-else"}
        reason = LOCAL_ONLY_REJECTED_REGISTRATION
    build_b = host.start(MEMBER + "alice")
    host.emit(build_b)

    host.flush(build_b)

    assert host.states()[build_a.run] == ("local_only", reason)
    assert host.collector.accepted[build_b.run] == set(build_b.appended)
    # The warning says "this run"; another build's run is not this build's.
    assert "local-only" not in capsys.readouterr().err


def test_a_shared_run_id_refuses_the_second_user_its_own_run_only(host):
    build_a = host.start(MEMBER + "alice", run_id="shared")
    build_b = host.start(MEMBER + "bob", run_id="shared")
    host.emit(build_a)
    host.emit(build_b)

    host.flush(build_a)
    host.flush(build_b)

    assert host.states()[build_a.run] == ("pending", None)
    assert host.states()[build_b.run] == (
        "local_only",
        LOCAL_ONLY_REJECTED_REGISTRATION,
    )
    assert host.collector.accepted[build_a.run] == set(build_a.appended)
    _assert_own_credential_only(host)


def test_a_run_without_a_lease_is_adopted_only_once_idle(host):
    """A run from a checkout that predates leases has no lease file, so only
    its recent activity says whether its service may still be running."""

    legacy = _registration("run-legacy", "producer-legacy")
    older_checkout = EventSpool(host.spool_path)
    older_checkout.register(legacy)
    appended = {older_checkout.append(legacy, EVENT)["event_id"]}
    older_checkout._engine.dispose()
    legacy_run = (legacy["run_id"], legacy["producer_id"])
    adopter = host.start(MEMBER + "alice")

    host.flush(adopter)
    assert all(about != legacy_run for _, _, about, _ in host.collector.requests)

    idle_since = datetime.now(UTC) - timedelta(seconds=ORPHAN_IDLE_SECONDS + 60)
    with closing(sqlite3.connect(host.spool_path)) as connection:
        connection.execute(
            "UPDATE telemetry_runs SET updated_at = ? WHERE run_id = ?",
            (idle_since.isoformat(), legacy["run_id"]),
        )
        connection.commit()
    host.flush(adopter)

    assert host.collector.accepted[legacy_run] == appended
    leases = ProducerLeases.beside(host.spool_path)
    lease = leases.try_acquire(*legacy_run, create=False)
    assert lease is not None, "the adopter must hold the lease only while delivering"
    lease.release()


def test_shutdown_drain_waits_only_for_its_own_run(tmp_path):
    spool = EventSpool(tmp_path / "events.sqlite3")
    own = _registration("run-own", "producer-own")
    other = _registration("run-other", "producer-other")
    spool.register(own)
    spool.register(other)
    spool.append(other, EVENT)
    flushes: list[str] = []
    service = EmitterService(
        socket_path=tmp_path / "unused.sock",
        registration=own,
        spool=spool,
        delivery=SimpleNamespace(flush_once=lambda: flushes.append("other") or False),
        sampler=SimpleNamespace(sample=lambda: None, parent_alive=lambda: True),
        heartbeat_seconds=60,
        drain_seconds=30,
    )

    started = time.monotonic()
    service._drain()
    assert flushes == []
    assert time.monotonic() - started < 5

    spool.append(own, EVENT)

    def deliver_own() -> bool:
        spool.acknowledge(
            [e["event_id"] for e in spool.batch("run-own", "producer-own")]
        )
        flushes.append("own")
        return True

    service.delivery = SimpleNamespace(flush_once=deliver_own)
    service._drain()
    assert flushes == ["own"]
    assert spool.has_deliverable()  # the other build's run, left to its service
    spool._engine.dispose()


def test_a_lease_is_exclusive_until_released(tmp_path):
    leases = ProducerLeases(tmp_path / "leases")
    held = leases.hold("run-a", "producer-a", timeout_seconds=1)
    assert held is not None

    assert leases.try_acquire("run-a", "producer-a", create=False) is None
    assert leases.hold("run-a", "producer-a", timeout_seconds=0.05) is None
    with pytest.raises(FileNotFoundError):
        leases.try_acquire("run-b", "producer-b", create=False)

    held.release()
    taken = leases.try_acquire("run-a", "producer-a", create=False)
    assert taken is not None
    taken.release()


def test_sweep_removes_only_free_leases_of_runs_with_nothing_pending(tmp_path):
    leases = ProducerLeases(tmp_path / "leases")
    live = leases.hold("run-live", "p", timeout_seconds=1)
    for run in (("run-orphan", "p"), ("run-done", "p")):
        leases.try_acquire(*run, create=True).release()

    assert leases.sweep(keep={("run-orphan", "p")}) == 1

    assert leases.exists("run-live", "p")
    assert leases.exists("run-orphan", "p")
    assert not leases.exists("run-done", "p")
    live.release()


def test_a_lease_file_replaced_between_open_and_lock_is_not_trusted(
    tmp_path, monkeypatch
):
    leases = ProducerLeases(tmp_path / "leases")
    path = leases.path("run-a", "producer-a")
    real = leases_module.fcntl
    calls: list[int] = []

    def flock(descriptor: int, operation: int) -> None:
        calls.append(descriptor)
        if len(calls) == 1:
            path.unlink()  # a sweeper removes the file between open and lock
        real.flock(descriptor, operation)

    monkeypatch.setattr(
        leases_module,
        "fcntl",
        SimpleNamespace(flock=flock, LOCK_EX=real.LOCK_EX, LOCK_NB=real.LOCK_NB),
    )
    lease = leases.try_acquire("run-a", "producer-a", create=True)

    assert lease is not None
    assert len(calls) == 2
    # The lock is on the file now at the path, so nobody else can take it.
    monkeypatch.setattr(leases_module, "fcntl", real)
    assert leases.try_acquire("run-a", "producer-a", create=False) is None
    lease.release()

    # Without create, a file removed under us reads as no lease at all.
    calls.clear()
    monkeypatch.setattr(
        leases_module,
        "fcntl",
        SimpleNamespace(flock=flock, LOCK_EX=real.LOCK_EX, LOCK_NB=real.LOCK_NB),
    )
    with pytest.raises(FileNotFoundError):
        leases.try_acquire("run-a", "producer-a", create=False)


def test_a_lease_that_cannot_be_checked_is_not_left_locked(tmp_path, monkeypatch):
    leases = ProducerLeases(tmp_path / "leases")
    leases.try_acquire("run-a", "producer-a", create=True).release()

    def unreadable(path, descriptor):
        raise PermissionError("cannot stat the lease file")

    with monkeypatch.context() as patch:
        patch.setattr(leases_module, "_names_open_file", unreadable)
        with pytest.raises(PermissionError):
            leases.try_acquire("run-a", "producer-a", create=False)

    # The failed attempt closed its descriptor, so the lock went with it.
    lease = leases.try_acquire("run-a", "producer-a", create=False)
    assert lease is not None
    lease.release()


def test_a_killed_service_releases_its_lease(tmp_path):
    leases = ProducerLeases(tmp_path / "leases")
    holder = textwrap.dedent(
        """
        import sys, time
        from microcosm.build.telemetry_emitter_service.leases import ProducerLeases
        lease = ProducerLeases(sys.argv[1]).hold(
            "run-a", "producer-a", timeout_seconds=5
        )
        print("held" if lease is not None else "missing", flush=True)
        time.sleep(600)
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", holder, str(leases.directory)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "held"
        assert leases.try_acquire("run-a", "producer-a", create=False) is None

        process.send_signal(signal.SIGKILL)
        process.wait(timeout=30)

        lease = leases.try_acquire("run-a", "producer-a", create=False)
        assert lease is not None
        lease.release()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
        process.stdout.close()


def _upload_state(spool_path: Path, run_id: str) -> str | None:
    with closing(sqlite3.connect(spool_path)) as connection:
        row = connection.execute(
            "SELECT upload_state FROM telemetry_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return row[0] if row else None


def test_two_real_services_on_one_spool_leave_each_others_runs_alone(
    tmp_path, monkeypatch, real_local_telemetry
):
    """Build A has a credential, build B has none, and both services share one
    spool. A's token exchange is held until B's worker has made a pass over
    A's queued run, and for 1.5 s more (the worker passes every second); A's
    run must still arrive in full."""

    release = threading.Event()
    received: dict[str, list[dict[str, Any]]] = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if self.path.endswith("/exchange"):
                release.wait(timeout=30)
                status = 200
                body = {"access_token": "collector-token", "expires_in": 3600}
            elif self.path == "/v1/runs":
                status, body = 201, {"registered": True}
            else:
                with lock:
                    received.setdefault(payload["events"][0]["run_id"], []).extend(
                        payload["events"]
                    )
                status = 202
                body = {"accepted": len(payload["events"]), "duplicates": 0}
            encoded = json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except OSError:
                pass  # the service gave up waiting and retries

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    spool_path = tmp_path / "shared.sqlite3"
    common = {
        "country_code": "US",
        "pipeline": "test-pipeline",
        "development_collector_url": f"http://127.0.0.1:{server.server_port}",
        "spool_path": spool_path,
        "heartbeat_seconds": 60,
        "startup_timeout_seconds": 60,
    }
    builds: list[LocalTelemetryEmitter] = []
    try:
        monkeypatch.setenv("HF_TOKEN", "hf-build-a-token")
        builds.append(LocalTelemetryEmitter.start(run_id="build-a", **common))
        monkeypatch.delenv("HF_TOKEN")
        builds.append(LocalTelemetryEmitter.start(run_id="build-b", **common))
        build_a, build_b = builds
        assert build_a.available and build_b.available
        build_a.stage("compile", message="Started.")
        build_b.stage("compile", message="Started.")
        # B's first delivery pass, which meets A's queued run too, makes B's own
        # run local-only. Leave time for another pass, then let A deliver.
        deadline = time.monotonic() + 60
        while _upload_state(spool_path, "build-b") != "local_only":
            assert time.monotonic() < deadline, "build B's worker never ran"
            time.sleep(0.1)
        time.sleep(1.5)
        release.set()
        build_a.complete()
        build_b.complete()
        for build in builds:
            build._process.wait(timeout=60)
    finally:
        release.set()
        for build in builds:
            if build._process is not None and build._process.poll() is None:
                build._process.kill()
                build._process.wait(timeout=30)
        server.shutdown()
        server_thread.join(timeout=5)
        server.server_close()

    with closing(sqlite3.connect(spool_path)) as connection:
        states = dict(
            connection.execute(
                "SELECT run_id, COALESCE(local_only_reason, upload_state) "
                "FROM telemetry_runs"
            )
        )
    # Before the fix, B's pass made A's run local-only (the e7 bug), and A,
    # working from the list it read before, uploaded B's local-only run.
    assert states == {
        "build-a": "pending",
        "build-b": LOCAL_ONLY_MISSING_CREDENTIAL,
    }
    assert "build-b" not in received, [
        (e["sequence"], e["message"]) for e in received["build-b"]
    ]
    events = sorted(received["build-a"], key=lambda event: event["sequence"])
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert events[-1]["message"] == BUILD_COMPLETED_MESSAGE


def _label_paths(host: Host) -> None:
    """Report which delivery paths a generated scenario reached."""

    for caller, kind, about, status in host.collector.requests:
        if about is not None and about != caller:
            event(f"adopted orphan: {kind} {status}")
    for caller, run, reason, *_ in host.local_only:
        event(f"{'own' if caller == run else 'adopted'} run local-only: {reason}")


PRODUCER = st.integers(0, 7)
# A new build: its login, and whether it reuses an earlier build's run id.
BUILD = st.tuples(st.sampled_from(CREDENTIALS), st.integers(0, 3))
OPERATIONS = st.lists(
    st.one_of(
        st.tuples(st.just("start"), BUILD),
        st.tuples(st.just("emit"), PRODUCER, st.integers(1, 3)),
        # Flushes are what decide runs' fates, so they come up most often.
        st.tuples(st.just("flush"), PRODUCER),
        st.tuples(st.just("flush"), PRODUCER),
        st.tuples(st.just("flush"), PRODUCER),
        st.tuples(st.just("kill"), PRODUCER),
        st.tuples(st.just("login"), PRODUCER, st.sampled_from(CREDENTIALS)),
        # One service flushes while another's request is in flight, and may
        # then exit: the first service's list of pending runs goes stale.
        st.tuples(st.just("interleave"), PRODUCER, PRODUCER, st.booleans()),
    ),
    max_size=30,
)


def _start_build(host: Host, build: tuple[str | None, int]) -> Producer:
    credential, reuse = build
    shared = (
        host.producers[(reuse - 1) % len(host.producers)].run[0]
        if reuse and host.producers
        else None
    )
    return host.start(credential, run_id=shared)


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    builds=st.lists(BUILD, min_size=2, max_size=4),
    operations=OPERATIONS,
    rejected=st.sets(st.integers(0, 7), max_size=2),
    conflicted=st.sets(st.integers(0, 7), max_size=2),
    # The logins the two last services start under, before Alice's and Bob's.
    rescue_logins=st.tuples(st.sampled_from(CREDENTIALS), st.sampled_from(CREDENTIALS)),
)
# Shrunk from a 1,500-example run against a delivery that trusted its listed
# runs: B makes its own run local-only and exits while A's exchange is in
# flight, and A then uploaded B's run.
@example(
    builds=[(MEMBER + "alice", 0), (None, 0)],
    operations=[("interleave", 0, 1, True)],
    rejected=set(),
    conflicted=set(),
    rescue_logins=(MEMBER + "alice", MEMBER + "bob"),
)
# From review: a service refused for every pending run, whose own run is not
# pending either, never read its login again, so a new login went unseen.
@example(
    builds=[(MEMBER + "alice", 0), (MEMBER + "alice", 0)],
    operations=[],
    rejected=set(),
    conflicted=set(),
    rescue_logins=(OUTSIDER, EXPIRED),
)
def test_no_service_decides_another_producers_run_from_its_own_credential(
    tmp_path_factory, builds, operations, rejected, conflicted, rescue_logins
):
    host = Host(
        tmp_path_factory.mktemp("foreign"),
        FakeCollector(
            {f"run-{index}" for index in rejected},
            {f"run-{index}" for index in conflicted},
        ),
    )
    try:
        # Builds already running, each with an event queued.
        for build in builds:
            host.emit(_start_build(host, build))
        for operation in operations:
            kind = operation[0]
            if kind == "start":
                _start_build(host, operation[1])
                continue
            live = [producer for producer in host.producers if producer.alive]
            if not live:
                continue
            producer = live[operation[1] % len(live)]
            if kind == "emit":
                host.emit(producer, operation[2])
            elif kind == "flush":
                host.flush(producer)
            elif kind == "kill":
                host.kill(producer)
            elif kind == "login":
                producer.credential = operation[2]
            else:
                _, _, other_index, exits = operation
                other = live[other_index % len(live)]
                if other is not producer:

                    def other_flushes(other=other, exits=exits) -> None:
                        host.flush(other)
                        if exits:
                            host.kill(other)

                    host.collector.during_next_request = other_flushes
                host.flush(producer)
                host.collector.during_next_request = None
            _assert_own_credential_only(host)

        # Every service goes. A run still pending is an orphan. Two more
        # services start under any login and make a pass, which may refuse
        # them. Then their logins become Alice's and Bob's and any session
        # they held runs out: between them they deliver each orphan in full.
        for producer in host.producers:
            if producer.alive:
                host.kill(producer)
        before = host.states()
        rescuers = [host.start(login) for login in rescue_logins]
        for rescuer in rescuers:
            host.flush(rescuer)
        _assert_own_credential_only(host)
        for rescuer, user in zip(rescuers, ("alice", "bob"), strict=True):
            rescuer.credential = MEMBER + user
        host.clock += DEFAULT_TOKEN_LIFETIME_SECONDS
        for _ in range(3):
            for rescuer in rescuers:
                host.flush(rescuer)
        _label_paths(host)
        _assert_own_credential_only(host)
        after = host.states()
        for producer in host.producers:
            if producer in rescuers or before[producer.run][0] != "pending":
                continue
            run_id = producer.run[0]
            if host.queued(producer.run) and (
                run_id in host.collector.conflicted_run_ids
            ):
                # Its owner, whoever that is, gets 409: a refusal about the run.
                assert after[producer.run] == (
                    "local_only",
                    LOCAL_ONLY_REJECTED_REGISTRATION,
                ), producer.run
                continue
            if run_id in host.collector.rejected_run_ids and producer.appended:
                if host.queued(producer.run):
                    assert after[producer.run] == (
                        "local_only",
                        LOCAL_ONLY_REJECTED_EVENTS,
                    )
                continue
            assert after[producer.run] == ("pending", None), producer.run
            assert host.queued(producer.run) == 0, producer.run
            assert host.collector.accepted.get(producer.run, set()) >= set(
                producer.appended
            )
    finally:
        host.close()
