"""Per-run isolation of collector delivery from the host's shared spool.

Every emitter service on a host delivers every pending run in one shared
spool, oldest update first. These tests check that a run whose delivery keeps
failing, for any reason, delays only itself; that it costs the collector a
bounded number of requests; that a failed token exchange, which every run
shares, charges no run; and that each error is one line, never a traceback.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import re
import sqlite3
import tempfile
import urllib.error
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from microcosm.build.telemetry_emitter import TelemetryRun
from microcosm.build.telemetry_emitter_service import collector as collector_module
from microcosm.build.telemetry_emitter_service.collector import (
    CollectorDelivery,
    RetryDelay,
)
from microcosm.build.telemetry_emitter_service.constants import (
    BATCH_SIZE,
    DELIVERY_RUN_WARNING,
    INITIAL_RETRY_SECONDS,
    MAX_RETRY_SECONDS,
    REJECTED_CREDENTIAL_MESSAGE,
    RUN_REGISTRATION_PATH,
    TOKEN_EXCHANGE_PATH,
    TOKEN_EXCHANGE_WARNING,
    UPLOAD_STATE_LOCAL_ONLY,
    UPLOAD_STATE_PENDING,
)
from microcosm.build.telemetry_emitter_service.contention import (
    is_transient_spool_error,
)
from microcosm.build.telemetry_emitter_service.models import (
    TelemetryEventRecord,
    TelemetryRunRecord,
)
from microcosm.build.telemetry_emitter_service.spool import EventSpool

_COLLECTOR = "http://127.0.0.1:9"


def _lock_error() -> OperationalError:
    driver_error = sqlite3.OperationalError("database is locked")
    driver_error.sqlite_errorcode = sqlite3.SQLITE_BUSY
    return OperationalError("DELETE FROM telemetry_events ...", {}, driver_error)


def _required_gap(consecutive_failures: int) -> float:
    """The retry delay after a scope's ``consecutive_failures``-th failure in a row."""

    return min(
        MAX_RETRY_SECONDS,
        INITIAL_RETRY_SECONDS * 2 ** (consecutive_failures - 1),
    )


def _warning_pattern(template: str) -> re.Pattern[str]:
    head, rest = template.split("{error_type}")
    middle, tail = rest.split("{error}")
    return re.compile(
        re.escape(head) + r"(\w+)" + re.escape(middle) + ".*" + re.escape(tail) + "$"
    )


_RUN_WARNING = _warning_pattern(DELIVERY_RUN_WARNING)
_EXCHANGE_WARNING = _warning_pattern(TOKEN_EXCHANGE_WARNING)


# --- The retry delay ---------------------------------------------------------


@given(
    gaps=st.lists(
        st.floats(0, 200, allow_nan=False, allow_infinity=False), max_size=20
    ),
    start=st.floats(0, 1e6, allow_nan=False, allow_infinity=False),
)
def test_retry_delay_doubles_to_its_cap_from_each_failure(gaps, start) -> None:
    delay = RetryDelay()
    assert delay.due(start)
    now = start
    for failures, gap in enumerate(gaps, start=1):
        delay.defer(now)
        wait = _required_gap(failures)
        assert delay.next_attempt_at == now + wait
        assert not delay.due(now + wait * 0.999)
        assert delay.due(now + wait)
        now += gap


# --- A simulated host: one shared spool, a scripted collector ----------------
#
# Each run has a script of outcomes, one per request carrying that run, and an
# outcome for every request after the script. A healthy run's requests always
# succeed. The token exchange's outcome is a function of time, as an outage of
# the identity provider would be. Events are appended on scheduled ticks, which
# moves the run to the back of the spool's oldest-update-first order, as a live
# build does, so runs that keep failing sort ahead of the runs being built.

_RUN_FAILURES = ("transport", "timeout", "503", "400", "401", "403", "409", "bug")
_RUN_OUTCOMES = ("ok", "ok", "ok") + _RUN_FAILURES
_EXCHANGE_OUTCOMES = ("ok",) * 6 + (
    "transport",
    "503",
    "malformed",
    "no_token",
    "bug",
    "403",
)


@dataclass(frozen=True)
class _RunSpec:
    key: tuple[str, str]
    corrupt: bool = False
    script: tuple[str, ...] = ()
    then: str = "ok"
    appends: tuple[int, ...] = (0,)

    @property
    def healthy(self) -> bool:
        return not self.corrupt and self.then == "ok" and set(self.script) <= {"ok"}


@dataclass(frozen=True)
class _ScenarioSpec:
    ticks: int
    runs: tuple[_RunSpec, ...]
    exchange: tuple[str, ...] = ()
    token_lifetime: int | None = 3_600
    listing_locked: frozenset[int] = frozenset()
    ack_locked: frozenset[int] = frozenset()
    # Batches smaller than a run's queue leave it pending after an accepted
    # batch, as a backlog after an outage does.
    batch_size: int = BATCH_SIZE

    def exchange_outcome(self, tick: int) -> str:
        return self.exchange[tick] if tick < len(self.exchange) else "ok"


@dataclass
class _RunState:
    spec: _RunSpec
    state: str = UPLOAD_STATE_PENDING
    updated_at: float = -1.0
    events: list[dict[str, object]] = field(default_factory=list)
    appended: int = 0
    responses: int = 0

    def next_outcome(self) -> str:
        script = self.spec.script
        outcome = (
            script[self.responses] if self.responses < len(script) else self.spec.then
        )
        self.responses += 1
        return outcome


@dataclass(frozen=True)
class _Request:
    tick: int
    kind: str  # "exchange", "register" or "events"
    run_key: tuple[str, str] | None
    outcome: str
    event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Pass:
    tick: int
    pending: tuple[tuple[str, str], ...]
    attempted: tuple[tuple[tuple[str, str], bool], ...]
    # The outcome of the token exchange made while attempting each run.
    exchanged: tuple[tuple[tuple[str, str], str], ...]
    raised: Exception | None


class _Host:
    """The spool surface that delivery uses, and the collector, in memory."""

    def __init__(self, spec: _ScenarioSpec, runs: tuple[_RunSpec, ...]) -> None:
        self.spec = spec
        self.runs = {run.key: _RunState(run) for run in runs}
        self.tick = 0
        self.requests: list[_Request] = []
        self.raised_types: dict[str, set[str]] = {"run": set(), "exchange": set()}
        self.local_only: dict[tuple[str, str], str] = {}

    def now(self) -> float:
        return float(self.tick)

    def append_scheduled_events(self) -> None:
        for run in self.runs.values():
            for _ in range(run.spec.appends.count(self.tick)):
                run.appended += 1
                run_id, producer_id = run.spec.key
                run.events.append(
                    {
                        "event_id": f"{run_id}/{producer_id}/{run.appended}",
                        "run_id": run_id,
                        "producer_id": producer_id,
                        "sequence": run.appended,
                    }
                )
                run.updated_at = self.now()

    def pending(self) -> list[tuple[str, str]]:
        ordered = sorted(
            self.runs.values(), key=lambda run: (run.updated_at, run.spec.key)
        )
        return [
            run.spec.key
            for run in ordered
            if run.state == UPLOAD_STATE_PENDING and run.events
        ]

    # EventSpool's delivery surface.

    def pending_run_keys(self) -> list[tuple[str, str]]:
        if self.tick in self.spec.listing_locked:
            raise _lock_error()
        return self.pending()

    def registration(self, run_id: str, producer_id: str) -> dict[str, object]:
        if self.runs[(run_id, producer_id)].spec.corrupt:
            self.raised_types["run"].add("ValueError")
            raise ValueError("telemetry spool JSON must contain an object")
        return {"run_id": run_id, "producer_id": producer_id}

    def batch(self, run_id, producer_id, limit=None):
        limit = self.spec.batch_size if limit is None else limit
        return [
            dict(event) for event in self.runs[(run_id, producer_id)].events[:limit]
        ]

    def acknowledge(self, event_ids) -> None:
        if self.tick in self.spec.ack_locked:
            raise _lock_error()
        acknowledged = set(event_ids)
        for run in self.runs.values():
            run.events = [e for e in run.events if e["event_id"] not in acknowledged]

    def make_local_only(self, run_id, producer_id, reason) -> None:
        run = self.runs[(run_id, producer_id)]
        run.state = UPLOAD_STATE_LOCAL_ONLY
        run.updated_at = self.now()
        self.local_only[(run_id, producer_id)] = reason

    # The collector.

    def post(self, url, payload, bearer_token, *, timeout=None):
        path = url.removeprefix(_COLLECTOR)
        if path == TOKEN_EXCHANGE_PATH:
            return self._exchange()
        if path == RUN_REGISTRATION_PATH:
            kind, event_ids = "register", ()
            run_key = (payload["run_id"], payload["producer_id"])
        else:
            first = payload["events"][0]
            kind, run_key = "events", (first["run_id"], first["producer_id"])
            assert path == f"/v1/runs/{first['run_id']}/events"
            event_ids = tuple(event["event_id"] for event in payload["events"])
        outcome = self.runs[run_key].next_outcome()
        self.requests.append(_Request(self.tick, kind, run_key, outcome, event_ids))
        if outcome == "ok":
            return (201 if kind == "register" else 202), {}
        if outcome == "transport":
            raise urllib.error.URLError("connection refused")
        if outcome == "timeout":
            raise TimeoutError("timed out")
        if outcome == "bug":
            self.raised_types["run"].add("RuntimeError")
            raise RuntimeError("collector bug")
        return int(outcome), {}

    def _exchange(self):
        outcome = self.spec.exchange_outcome(self.tick)
        self.requests.append(_Request(self.tick, "exchange", None, outcome))
        if outcome == "ok":
            response = {"access_token": f"collector-token-{self.tick}"}
            if self.spec.token_lifetime is not None:
                response["expires_in"] = self.spec.token_lifetime
            return 200, response
        if outcome == "malformed":
            # int(None) raises TypeError in the client.
            self.raised_types["exchange"].add("TypeError")
            return 200, {"access_token": "collector-token", "expires_in": None}
        if outcome == "no_token":
            return 200, {"expires_in": 3_600}
        if outcome == "transport":
            raise urllib.error.URLError("connection refused")
        if outcome == "bug":
            self.raised_types["exchange"].add("RuntimeError")
            raise RuntimeError("identity provider bug")
        return int(outcome), {}


@dataclass
class _Trace:
    host: _Host
    passes: list[_Pass]
    stderr: str

    def requests_for(self, run_key):
        return [request for request in self.host.requests if request.run_key == run_key]

    def accepted_events(self, run_key) -> list[tuple[int, tuple[str, ...]]]:
        return [
            (request.tick, request.event_ids)
            for request in self.requests_for(run_key)
            if request.kind == "events" and request.outcome == "ok"
        ]


def _simulate(spec: _ScenarioSpec, *, runs=None) -> _Trace:
    host = _Host(spec, spec.runs if runs is None else runs)
    delivery = CollectorDelivery(
        host,
        development_collector_url=_COLLECTOR,
        clock=host.now,
    )
    # Each run delivery takes up on a pass (its delay has passed), and whether
    # it then held a collector token for that run.
    attempted: list[tuple[tuple[str, str], bool]] = []
    exchanged: list[tuple[tuple[str, str], str]] = []
    deliver_run = delivery._deliver_run
    collector_token = delivery._collector_token

    def recorded_deliver_run(run_key):
        attempted.append((run_key, False))
        return deliver_run(run_key)

    def recorded_collector_token(run_key):
        before = len(host.requests)
        token = collector_token(run_key)
        attempted[-1] = (run_key, token is not None)
        exchanged.extend(
            (run_key, request.outcome)
            for request in host.requests[before:]
            if request.kind == "exchange"
        )
        return token

    delivery._deliver_run = recorded_deliver_run
    delivery._collector_token = recorded_collector_token
    passes: list[_Pass] = []
    stderr = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stderr(stderr):
        patch.setattr(collector_module, "_http_post", host.post)
        patch.setattr(collector_module, "_huggingface_token", lambda: "hf-test-token")
        for tick in range(spec.ticks):
            host.tick = tick
            host.append_scheduled_events()
            pending = tuple(host.pending())
            attempted.clear()
            exchanged.clear()
            try:
                delivery.flush_once()
            except Exception as error:  # checked below: only lock contention
                raised = error
            else:
                raised = None
            passes.append(
                _Pass(tick, pending, tuple(attempted), tuple(exchanged), raised)
            )
    return _Trace(host, passes, stderr.getvalue())


@st.composite
def _scenarios(draw, *, exchange_always_succeeds=False, ack_contention=True):
    ticks = draw(st.integers(1, 150))
    tick = st.integers(0, ticks - 1)
    runs = []
    for index in range(draw(st.integers(1, 6))):
        key = (f"run-{index}", f"producer-{index % 2}")
        appends = tuple(draw(st.lists(tick, min_size=1, max_size=5)))
        if draw(st.booleans()):
            runs.append(_RunSpec(key, appends=appends))
            continue
        runs.append(
            _RunSpec(
                key,
                corrupt=draw(st.booleans()),
                script=tuple(
                    draw(st.lists(st.sampled_from(_RUN_OUTCOMES), max_size=10))
                ),
                then=draw(st.sampled_from(_RUN_OUTCOMES)),
                appends=appends,
            )
        )
    exchange = (
        ()
        if exchange_always_succeeds
        else tuple(draw(st.lists(st.sampled_from(_EXCHANGE_OUTCOMES), max_size=ticks)))
    )
    return _ScenarioSpec(
        ticks=ticks,
        runs=tuple(runs),
        exchange=exchange,
        token_lifetime=draw(st.sampled_from([3_600, 60, None])),
        listing_locked=frozenset(draw(st.sets(tick, max_size=4))),
        ack_locked=(
            frozenset(draw(st.sets(tick, max_size=4)))
            if ack_contention
            else frozenset()
        ),
        batch_size=draw(st.sampled_from([1, 2, BATCH_SIZE])),
    )


def _check_flush_raises_only_lock_contention(trace: _Trace) -> None:
    for record in trace.passes:
        assert record.raised is None or is_transient_spool_error(record.raised)


def _check_no_starvation(trace: _Trace) -> None:
    runs = trace.host.runs
    for record in trace.passes:
        attempted = [run_key for run_key, _ in record.attempted]
        healthy = [key for key in record.pending if runs[key].spec.healthy]
        if record.raised is None:
            # Every pending healthy run is attempted on every pass, whatever
            # the other runs did on this pass or before it.
            assert set(healthy) <= set(attempted)
        elif attempted:
            # Lock contention ended the pass for every run alike: each healthy
            # run listed before the last run attempted was attempted.
            last = record.pending.index(attempted[-1])
            assert {key for key in healthy if record.pending.index(key) < last} <= set(
                attempted
            )
        sent = {
            request.run_key
            for request in trace.host.requests
            if request.tick == record.tick and request.kind == "events"
        }
        # A healthy run given a token sends its batch on the same pass.
        for run_key, had_token in record.attempted:
            if had_token and runs[run_key].spec.healthy:
                assert run_key in sent


def _check_request_bounds(trace: _Trace) -> None:
    requests = trace.host.requests
    per_pass = Counter(
        (request.tick, request.kind, request.run_key)
        for request in requests
        if request.kind != "exchange"
    )
    # At most one registration and one events request per run per pass.
    assert max(per_pass.values(), default=1) == 1
    # No event is accepted twice, even when contention kept an accepted batch
    # in the spool.
    accepted = Counter(
        event_id
        for request in requests
        if (request.kind, request.outcome) == ("events", "ok")
        for event_id in request.event_ids
    )
    assert max(accepted.values(), default=1) == 1
    for run_key in trace.host.runs:
        ticks = sorted({request.tick for request in trace.requests_for(run_key)})
        failures = 0
        for earlier, later in itertools.pairwise(ticks):
            outcomes = [
                (request.kind, request.outcome)
                for request in trace.requests_for(run_key)
                if request.tick == earlier
            ]
            if ("events", "ok") in outcomes:
                failures = 0
                continue
            # The run failed here (a local-only verdict has no later request).
            failures += 1
            assert later - earlier >= _required_gap(failures), (run_key, ticks)
    exchanges = [request for request in requests if request.kind == "exchange"]
    failures = 0
    for earlier, later in itertools.pairwise(exchanges):
        if earlier.outcome == "ok":
            failures = 0
        elif earlier.outcome != "403":
            # A rejected credential makes that one run local-only and neither
            # defers nor resets the exchange; any other failure defers it.
            failures += 1
            assert later.tick - earlier.tick >= _required_gap(failures)
    # More than one exchange in a pass only replaces a token a run's request
    # was refused with (401) or follows a credential rejection (403).
    for tick in {request.tick for request in exchanges}:
        same_pass = [request for request in requests if request.tick == tick]
        replacements = sum(
            (request.kind == "exchange" and request.outcome == "403")
            or (request.kind != "exchange" and request.outcome == "401")
            for request in same_pass
        )
        assert sum(request.kind == "exchange" for request in same_pass) <= (
            1 + replacements
        )


def _check_reports(trace: _Trace) -> None:
    lines = trace.stderr.splitlines()
    assert not any("Traceback" in line for line in lines)
    run_types = [m.group(1) for line in lines if (m := _RUN_WARNING.match(line))]
    exchange_types = [
        m.group(1) for line in lines if (m := _EXCHANGE_WARNING.match(line))
    ]
    # One line per error type and scope, for exactly the types raised.
    assert sorted(run_types) == sorted(trace.host.raised_types["run"])
    assert sorted(exchange_types) == sorted(trace.host.raised_types["exchange"])
    others = [
        line
        for line in lines
        if not _RUN_WARNING.match(line) and not _EXCHANGE_WARNING.match(line)
    ]
    # Every other line is the once-per-run notice of a rejected run.
    assert others == [REJECTED_CREDENTIAL_MESSAGE] * len(trace.host.local_only)


_LOCAL_ONLY_VERDICTS = {("register", "403"), ("register", "409"), ("events", "403")}


def _check_verdicts(trace: _Trace) -> None:
    """Exactly the runs the collector refused for good are made local-only."""

    refused = {
        request.run_key
        for request in trace.host.requests
        if (request.kind, request.outcome) in _LOCAL_ONLY_VERDICTS
    }
    rejected = {
        run_key
        for record in trace.passes
        for run_key, outcome in record.exchanged
        if outcome == "403"
    }
    assert set(trace.host.local_only) == refused | rejected


def _check_runs_retry_exactly_when_due(trace: _Trace) -> None:
    """Each run is attempted on exactly the passes its own delay allows.

    A reference model of one ``RetryDelay`` per run: a failure doubles the
    run's delay, a delivered batch clears it, and nothing else touches it.
    Waiting for a token and a local-only verdict leave it as it was. A batch
    the collector accepted but lock contention kept in the spool is delivered
    once it is removed, on the run's next attempt, which sends nothing.
    """

    delays: dict[tuple[str, str], tuple[float, float]] = {}
    awaiting_removal: set[tuple[str, str]] = set()
    for record in trace.passes:
        attempted = [run_key for run_key, _ in record.attempted]
        if record.raised is not None and not attempted:
            continue  # the listing itself was locked
        delays = {key: delays[key] for key in record.pending if key in delays}
        awaiting_removal &= set(record.pending)
        # Contention ends a pass at the run whose removal from the spool it
        # blocked, always the last one attempted.
        blocked = attempted[-1] if record.raised is not None else None
        considered = (
            record.pending
            if record.raised is None
            else record.pending[: record.pending.index(attempted[-1]) + 1]
        )
        for run_key in considered:
            next_attempt_at, _ = delays.get(run_key, (0.0, INITIAL_RETRY_SECONDS))
            assert (run_key in attempted) is (record.tick >= next_attempt_at), (
                record.tick,
                run_key,
            )
        for run_key, had_token in record.attempted:
            outcomes = {
                (request.kind, request.outcome)
                for request in trace.requests_for(run_key)
                if request.tick == record.tick
            }
            if run_key in awaiting_removal:
                assert not outcomes and not had_token
                if run_key != blocked:
                    awaiting_removal.discard(run_key)
                    delays.pop(run_key, None)
            elif not had_token:
                continue
            elif ("events", "ok") in outcomes:
                if run_key == blocked:
                    awaiting_removal.add(run_key)
                else:
                    delays.pop(run_key, None)
            elif not outcomes & _LOCAL_ONLY_VERDICTS:
                _, seconds = delays.get(run_key, (0.0, INITIAL_RETRY_SECONDS))
                delays[run_key] = (
                    record.tick + seconds,
                    min(MAX_RETRY_SECONDS, seconds * 2),
                )


@settings(max_examples=300, deadline=None)
@given(spec=_scenarios())
@example(
    spec=_ScenarioSpec(
        ticks=150,
        runs=(
            _RunSpec(("stuck", "p"), then="503"),
            _RunSpec(("corrupt", "p"), corrupt=True),
            _RunSpec(("building", "p"), appends=tuple(range(150))),
        ),
    )
).via("a persistently failing and a corrupt run ahead of a live build")
@example(
    spec=_ScenarioSpec(
        ticks=5,
        runs=(_RunSpec(("buggy", "p"), then="bug"),),
        exchange=("bug", "ok"),
    )
).via("the same error type from the exchange, then from a run")
def test_a_failing_run_never_holds_up_another_and_costs_bounded_requests(
    spec,
) -> None:
    trace = _simulate(spec)

    _check_flush_raises_only_lock_contention(trace)
    _check_no_starvation(trace)
    _check_request_bounds(trace)
    _check_runs_retry_exactly_when_due(trace)
    _check_reports(trace)
    _check_verdicts(trace)


@settings(max_examples=200, deadline=None)
@given(spec=_scenarios(exchange_always_succeeds=True, ack_contention=False))
def test_removing_the_failing_runs_never_changes_when_healthy_runs_deliver(
    spec,
) -> None:
    """A failing run delays only itself, stated as a counterfactual.

    With the token exchange working and contention only on listing the
    queue, which the other runs cannot change, the healthy runs deliver the
    same events on the same ticks with or without the other runs present.
    """

    healthy = tuple(run for run in spec.runs if run.healthy)
    shared = _simulate(spec)
    alone = _simulate(spec, runs=healthy)

    for run in healthy:
        assert shared.accepted_events(run.key) == alone.accepted_events(run.key)
        for tick, _ in shared.accepted_events(run.key):
            assert tick not in spec.listing_locked
        # With batches that hold a whole queue, each event is delivered on the
        # first tick from its own on which the listing was not locked.
        if spec.batch_size == BATCH_SIZE:
            assert not shared.host.runs[run.key].events or spec.ticks - 1 in (
                spec.listing_locked
            )


def test_a_persistently_failing_run_settles_to_one_request_a_minute() -> None:
    spec = _ScenarioSpec(
        ticks=3_600,
        runs=(
            _RunSpec(("stuck", "p"), script=("ok",), then="503"),
            _RunSpec(("building", "p"), appends=tuple(range(0, 3_600, 10))),
        ),
    )

    trace = _simulate(spec)

    stuck_ticks = [request.tick for request in trace.requests_for(("stuck", "p"))]
    # Registered and refused at 0, then refused after waits of 1, 2, 4, ... s
    # up to 60 s: 66 requests in an hour, not 3,600.
    expected = [0, 0]
    tick, failures = 0, 0
    while True:
        failures += 1
        tick += int(_required_gap(failures))
        if tick >= spec.ticks:
            break
        expected.append(tick)
    assert stuck_ticks == expected
    assert len(stuck_ticks) == 66
    assert [tick for tick, _ in trace.accepted_events(("building", "p"))] == list(
        range(0, 3_600, 10)
    )


@pytest.mark.parametrize(
    "script",
    [pytest.param((), id="registration"), pytest.param(("ok",), id="events")],
)
def test_a_rejected_token_from_one_run_costs_one_exchange_per_attempt(
    script,
) -> None:
    """A run whose requests are refused 401 replaces the shared token each time.

    Its own delay bounds how often, and the healthy run is never held up.
    """

    spec = _ScenarioSpec(
        ticks=40,
        runs=(
            _RunSpec(("refused", "p"), script=script, then="401"),
            _RunSpec(("building", "p"), appends=tuple(range(40))),
        ),
    )

    trace = _simulate(spec)

    refused = [
        r.tick for r in trace.requests_for(("refused", "p")) if r.outcome == "401"
    ]
    assert refused == [0, 1, 3, 7, 15, 31]
    exchanges = [r.tick for r in trace.host.requests if r.kind == "exchange"]
    # One at the start, then one after each refusal, by whichever run next
    # needs the token.
    assert len(exchanges) == 1 + len(refused)
    assert set(exchanges) <= {0, *refused}
    assert [tick for tick, _ in trace.accepted_events(("building", "p"))] == list(
        range(40)
    )
    _check_reports(trace)


def test_a_working_exchange_resets_the_exchange_delay() -> None:
    """After a token is issued, the next exchange failure waits 1 s again."""

    spec = _ScenarioSpec(
        ticks=40,
        runs=(_RunSpec(("building", "p"), appends=tuple(range(40))),),
        # A 60 s token is replaced 30 s before it expires.
        token_lifetime=60,
        exchange=("transport", "ok") + ("ok",) * 29 + ("503", "ok"),
    )

    trace = _simulate(spec)

    exchanges = [
        (r.tick, r.outcome) for r in trace.host.requests if r.kind == "exchange"
    ]
    assert exchanges == [(0, "transport"), (1, "ok"), (31, "503"), (32, "ok")]
    delivered = [tick for tick, _ in trace.accepted_events(("building", "p"))]
    assert delivered == [tick for tick in range(40) if tick not in {0, 31}]


# --- The real spool ------------------------------------------------------------


def _registration(run_id: str) -> dict[str, object]:
    return TelemetryRun(
        run_id=run_id,
        country_code="US",
        pipeline="test-pipeline",
        producer_id="producer-a",
    ).as_registration()


def _event(stage_id: str = "compile") -> dict[str, object]:
    return {"event_type": "stage", "stage_id": stage_id, "status": "started"}


class _Collector:
    """A loopback collector stand-in recording each request's path."""

    def __init__(self, events_status=None, exchange_response=None) -> None:
        self.paths: list[str] = []
        self._events_status = events_status or {}
        self._exchange_response = exchange_response

    def post(self, url, payload, bearer_token, *, timeout=None):
        path = url.removeprefix(_COLLECTOR)
        self.paths.append(path)
        if path == TOKEN_EXCHANGE_PATH:
            if self._exchange_response is not None:
                return self._exchange_response()
            return 200, {"access_token": "collector-token", "expires_in": 3_600}
        if path == RUN_REGISTRATION_PATH:
            return 201, {}
        return self._events_status.get(path, 202), {}


@pytest.fixture
def collector(monkeypatch):
    def install(**kwargs) -> _Collector:
        fake = _Collector(**kwargs)
        monkeypatch.setattr(collector_module, "_http_post", fake.post)
        monkeypatch.setattr(collector_module, "_huggingface_token", lambda: "hf")
        return fake

    return install


def test_a_corrupt_registration_fails_only_its_own_run(
    tmp_path, collector, capsys
) -> None:
    spool_path = tmp_path / "events.sqlite3"
    spool = EventSpool(spool_path)
    for run_id in ("corrupt", "healthy-a", "healthy-b"):
        spool.register(_registration(run_id))
        spool.append(_registration(run_id), _event())
    connection = sqlite3.connect(spool_path)
    connection.execute(
        "UPDATE telemetry_runs SET registration_json = '[1]' WHERE run_id = 'corrupt'"
    )
    connection.commit()
    connection.close()
    fake = collector()
    clock = [0.0]
    delivery = CollectorDelivery(
        spool, development_collector_url=_COLLECTOR, clock=lambda: clock[0]
    )

    # Decoding every registration at once fails for every run; keys do not.
    with pytest.raises(ValueError, match="must contain an object"):
        spool.pending_runs()
    assert spool.pending_run_keys()[0] == ("corrupt", "producer-a")
    assert delivery.flush_once() is True

    assert spool.pending_run_keys() == [("corrupt", "producer-a")]
    assert "/v1/runs/corrupt/events" not in fake.paths
    assert fake.paths.count(RUN_REGISTRATION_PATH) == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert _RUN_WARNING.match(lines[0]).group(1) == "ValueError"
    assert "Traceback" not in lines[0]
    # It waits out its own delay, then fails again without a request or a line.
    clock[0] = 0.5
    requests = len(fake.paths)
    assert delivery.flush_once() is False
    clock[0] = 1.0
    assert delivery.flush_once() is False
    assert len(fake.paths) == requests
    assert capsys.readouterr().err == ""


def test_a_run_the_collector_keeps_refusing_delays_only_itself(
    tmp_path, collector
) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    for run_id in ("refused", "building"):
        spool.register(_registration(run_id))
    spool.append(_registration("refused"), _event())
    fake = collector(events_status={"/v1/runs/refused/events": 500})
    clock = [0.0]
    delivery = CollectorDelivery(
        spool, development_collector_url=_COLLECTOR, clock=lambda: clock[0]
    )
    delivered: list[int] = []
    refused: list[int] = []

    for tick in range(130):
        clock[0] = float(tick)
        spool.append(_registration("building"), _event(f"stage-{tick}"))
        before = len(fake.paths)
        delivery.flush_once()
        new = fake.paths[before:]
        if "/v1/runs/refused/events" in new:
            refused.append(tick)
        if "/v1/runs/building/events" in new:
            delivered.append(tick)

    assert delivered == list(range(130))
    assert refused == [0, 1, 3, 7, 15, 31, 63, 123]
    assert spool.batch("building", "producer-a") == []
    assert len(spool.batch("refused", "producer-a")) == 1


def test_a_malformed_token_response_charges_no_run(tmp_path, collector, capsys) -> None:
    """The exchange is shared, so it backs off alone and repeats no request."""

    responses = iter(
        [
            (200, {"access_token": "collector-token", "expires_in": None}),
            (200, {"access_token": "collector-token", "expires_in": 3_600}),
        ]
    )
    spool = EventSpool(tmp_path / "events.sqlite3")
    for run_id in ("run-a", "run-b"):
        spool.register(_registration(run_id))
        spool.append(_registration(run_id), _event())
    fake = collector(exchange_response=lambda: next(responses))
    clock = [0.0]
    delivery = CollectorDelivery(
        spool, development_collector_url=_COLLECTOR, clock=lambda: clock[0]
    )

    assert delivery.flush_once() is False
    assert fake.paths == [TOKEN_EXCHANGE_PATH]
    clock[0] = 0.5
    assert delivery.flush_once() is False
    assert fake.paths == [TOKEN_EXCHANGE_PATH]
    lines = capsys.readouterr().err.splitlines()
    assert [_EXCHANGE_WARNING.match(line).group(1) for line in lines] == ["TypeError"]
    # Once the exchange works, every run delivers on that pass: none was
    # charged for the shared failure.
    clock[0] = 1.0
    assert delivery.flush_once() is True
    assert spool.pending_run_keys() == []


def test_a_locked_spool_listing_is_retried_on_the_next_tick() -> None:
    attempts: list[int] = []

    class LockedSpool:
        def pending_run_keys(self):
            attempts.append(1)
            raise _lock_error()

    delivery = CollectorDelivery(LockedSpool(), development_collector_url=_COLLECTOR)
    for _ in range(3):
        with pytest.raises(OperationalError):
            delivery.flush_once()
    assert len(attempts) == 3


def test_contention_after_an_accepted_batch_never_sends_it_twice(
    tmp_path, collector, capsys
) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    for run_id in ("run-a", "run-b"):
        spool.register(_registration(run_id))
        spool.append(_registration(run_id), _event())
    fake = collector()
    acknowledge = spool.acknowledge
    locked = [True]

    def contended_acknowledge(event_ids) -> None:
        if locked[0]:
            locked[0] = False
            raise _lock_error()
        acknowledge(event_ids)

    spool.acknowledge = contended_acknowledge
    delivery = CollectorDelivery(
        spool, development_collector_url=_COLLECTOR, clock=lambda: 0.0
    )

    with pytest.raises(OperationalError):
        delivery.flush_once()
    assert fake.paths == [
        TOKEN_EXCHANGE_PATH,
        RUN_REGISTRATION_PATH,
        "/v1/runs/run-a/events",
    ]
    assert spool.pending_run_keys()[0] == ("run-a", "producer-a")
    # Same instant: nothing was deferred, so both runs go now. run-a's accepted
    # batch is removed from the spool without being sent again.
    assert delivery.flush_once() is True
    assert fake.paths[3:] == [RUN_REGISTRATION_PATH, "/v1/runs/run-b/events"]
    assert spool.pending_run_keys() == []
    assert capsys.readouterr().err == ""


def test_delivery_forgets_runs_that_leave_the_queue(tmp_path, collector) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    spool.register(_registration("run-a"))
    spool.append(_registration("run-a"), _event())
    collector(events_status={"/v1/runs/run-a/events": 503})
    delivery = CollectorDelivery(
        spool, development_collector_url=_COLLECTOR, clock=lambda: 0.0
    )

    delivery.flush_once()
    assert set(delivery._run_delays) == {("run-a", "producer-a")}
    spool.make_local_only("run-a", "producer-a", "test")
    delivery.flush_once()
    assert delivery._run_delays == {}


# --- The pending-run listing -------------------------------------------------


def _legacy_pending_keys(spool: EventSpool) -> list[tuple[str, str, str]]:
    """The join-and-distinct listing ``pending_runs`` used before."""

    statement = (
        select(TelemetryRunRecord)
        .join(TelemetryRunRecord.events)
        .where(TelemetryRunRecord.upload_state == UPLOAD_STATE_PENDING)
        .order_by(TelemetryRunRecord.updated_at)
        .distinct()
    )
    with spool._session_factory() as session:
        return [
            (run.updated_at, run.run_id, run.producer_id)
            for run in session.scalars(statement)
        ]


@settings(max_examples=25, deadline=None)
@given(
    runs=st.lists(
        st.tuples(
            st.sampled_from([UPLOAD_STATE_PENDING, UPLOAD_STATE_LOCAL_ONLY]),
            st.integers(0, 3),  # events
            st.integers(0, 3),  # updated_at, with ties
        ),
        max_size=8,
    )
)
def test_pending_run_keys_list_the_runs_the_old_query_did_in_update_order(
    runs,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        spool = EventSpool(Path(directory) / "events.sqlite3")
        try:
            with spool._session_factory.begin() as session:
                for index, (state, events, updated_at) in enumerate(runs):
                    run_id = f"run-{index}"
                    stamp = f"2026-10-09T00:00:0{updated_at}+00:00"
                    session.add(
                        TelemetryRunRecord(
                            run_id=run_id,
                            producer_id="producer-a",
                            registration=_registration(run_id),
                            next_sequence=events + 1,
                            upload_state=state,
                            local_only_reason=None,
                            updated_at=stamp,
                        )
                    )
                    for sequence in range(1, events + 1):
                        session.add(
                            TelemetryEventRecord(
                                event_id=f"{run_id}-{sequence}",
                                run_id=run_id,
                                producer_id="producer-a",
                                sequence=sequence,
                                payload={"sequence": sequence},
                                created_at=stamp,
                            )
                        )
            legacy = sorted(_legacy_pending_keys(spool))
            keys = spool.pending_run_keys()

            assert keys == [(run_id, producer) for _, run_id, producer in legacy]
            assert [dict(r) for r in spool.pending_runs()] == [
                spool.registration(*key) for key in keys
            ]
        finally:
            spool._engine.dispose()


def test_registration_of_an_unknown_run_is_a_key_error(tmp_path) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    with pytest.raises(KeyError):
        spool.registration("missing", "producer-a")
