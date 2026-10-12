"""How long ``LocalTelemetryEmitter.start`` waits for its service, and what it keeps.

A fake clock drives the real startup loop. The service process, its readiness
ping and ``time`` are fakes, so each example runs in microseconds whatever the
host load, and the wait's contract is asserted exactly:

* a service that answers before the budget runs out is kept, however slowly
  it started;
* a service that exits first is reported with its exit status, at once;
* a live service that never answers is abandoned at the budget, no later than
  one poll after it;
* nothing is sent to a service that never answered.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from microcosm.build import telemetry_emitter as module
from microcosm.build.telemetry_emitter import LocalTelemetryEmitter
from microcosm.build.telemetry_emitter_constants import (
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    SOCKET_FILENAME,
    STARTUP_POLL_SECONDS,
)
from microcosm.build.telemetry_protocol import BUILD_STARTED_MESSAGE

_RUNTIME_DIRECTORIES = itertools.count()
_EXIT_STATUS = 3
# Event times this far apart cannot share a poll, even with float drift in the
# fake clock, so each example has one unambiguous expected outcome.
_SEPARATION = 3 * STARTUP_POLL_SECONDS


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _ServiceProcess:
    """A service process that exits at ``exit_at`` on the fake clock."""

    def __init__(self, clock: _Clock, exit_at: float | None) -> None:
        self._clock = clock
        self._exit_at = exit_at
        self.terminated = False
        self.waited = False

    def poll(self) -> int | None:
        if self.terminated:
            return -15
        if self._exit_at is not None and self._clock.now >= self._exit_at:
            return _EXIT_STATUS
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int | None:
        self.waited = True
        return self.poll()


def _start(tmp_path, *, ready_at: float | None, exit_at: float | None, **budget):
    """Run the real ``start`` against a service ready at ``ready_at``, then close.

    Returns whether the handle was available and what reached the service
    before ``close``.
    """

    clock = _Clock()
    process = _ServiceProcess(clock, exit_at)
    sent: list[dict[str, object]] = []

    def make_runtime_directory(**_kwargs) -> str:
        runtime = tmp_path / f"runtime-{next(_RUNTIME_DIRECTORIES)}"
        runtime.mkdir()
        (runtime / SOCKET_FILENAME).touch()
        return str(runtime)

    def service_ready(_socket_path) -> bool:
        return process.poll() is None and ready_at is not None and clock.now >= ready_at

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            module,
            "time",
            SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep),
        )
        patch.setattr(module.tempfile, "mkdtemp", make_runtime_directory)
        patch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
        patch.setattr(module, "_service_ready", service_ready)
        patch.setattr(module, "runtime_identity", dict)
        patch.setattr(
            LocalTelemetryEmitter, "_send", lambda _self, payload: sent.append(payload)
        )
        emitter = LocalTelemetryEmitter.start(
            run_id="startup-wait",
            country_code="US",
            pipeline="test-pipeline",
            development_collector_url="http://127.0.0.1:1",
            spool_path=tmp_path / "events.sqlite3",
            **budget,
        )
        available = emitter.available
        sent_before_close = list(sent)
        emitter.close()
    return available, process, clock, sent_before_close


_SECONDS = st.floats(min_value=0, max_value=10, allow_nan=False)


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    ready_at=st.none() | _SECONDS,
    exit_at=st.none() | _SECONDS,
    timeout=_SECONDS,
)
def test_start_keeps_exactly_the_services_that_answer_within_budget(
    tmp_path, capsys, real_local_telemetry, ready_at, exit_at, timeout
) -> None:
    events = [time for time in (ready_at, exit_at, timeout) if time is not None]
    assume(all(abs(a - b) > _SEPARATION for a, b in itertools.combinations(events, 2)))
    capsys.readouterr()

    available, process, clock, sent = _start(
        tmp_path, ready_at=ready_at, exit_at=exit_at, startup_timeout_seconds=timeout
    )

    exits_first = exit_at is not None and exit_at < timeout
    horizon = min(timeout, math.inf if exit_at is None else exit_at)
    err = capsys.readouterr().err
    if ready_at is not None and ready_at < horizon:
        assert available
        assert clock.now <= ready_at + STARTUP_POLL_SECONDS + 1e-9
        assert not process.terminated
        assert sent[0]["event"]["message"] == BUILD_STARTED_MESSAGE
        assert err == ""
    else:
        assert not available
        assert clock.now <= horizon + STARTUP_POLL_SECONDS + 1e-9
        assert sent == []
        assert process.waited
        if exits_first:
            assert not process.terminated
            assert f"exited with status {_EXIT_STATUS} before" in err
        else:
            assert process.terminated
            assert f"did not become ready within {timeout:g} s" in err


def test_default_budget_keeps_a_service_that_starts_slowly(
    tmp_path, capsys, real_local_telemetry
) -> None:
    """A service needing 20 s, which the former 3 s budget abandoned, is kept."""

    available, process, clock, _sent = _start(tmp_path, ready_at=20.0, exit_at=None)

    assert available
    assert 20.0 <= clock.now <= 20.0 + STARTUP_POLL_SECONDS + 1e-9
    assert not process.terminated
    assert capsys.readouterr().err == ""


def test_default_budget_abandons_a_service_that_never_answers(
    tmp_path, capsys, real_local_telemetry
) -> None:
    available, process, clock, sent = _start(tmp_path, ready_at=None, exit_at=None)

    assert not available
    assert DEFAULT_STARTUP_TIMEOUT_SECONDS == 30.0
    assert clock.now <= DEFAULT_STARTUP_TIMEOUT_SECONDS + STARTUP_POLL_SECONDS
    assert process.terminated and process.waited
    assert sent == []
    assert "did not become ready within 30 s" in capsys.readouterr().err


def test_a_service_that_exits_is_reported_at_once(
    tmp_path, capsys, real_local_telemetry
) -> None:
    available, process, clock, sent = _start(tmp_path, ready_at=None, exit_at=0.5)

    assert not available
    assert clock.now <= 0.5 + STARTUP_POLL_SECONDS + 1e-9
    assert not process.terminated
    assert sent == []
    err = capsys.readouterr().err
    assert f"exited with status {_EXIT_STATUS} before it became ready" in err
    assert "within" not in err
