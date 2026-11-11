"""A build program's telemetry run (``microcosm.build.program_telemetry``).

The shared fixture replaces the adapter's start with an in-memory emitter
(``fake_telemetry_emitters``). The tests of an emitter that cannot be used
put the adapter's own start back and name a missing emitter command, so no
test runs a real emitter.
"""

import os
import signal
from datetime import UTC, datetime

import pytest
from microcosm_emitter_interface import client as interface_client
from microcosm_emitter_interface import launcher as interface_launcher

from microcosm.build import program_telemetry
from microcosm.build.telemetry_emitter import (
    LocalTelemetryEmitter,
    TelemetryRequiredError,
)
from test_support.microcosm_build.telemetry import ADAPTER_START

#: The interface's own startup, captured before the shared fixture forbids it.
INTERFACE_START_EMITTER = interface_launcher.start_emitter
RUN_ID_VARIABLE = "MICROCOSM_TELEMETRY_RUN_ID"
PARENT_PID_VARIABLE = "MICROCOSM_TELEMETRY_PARENT_PID"


def _event_summary(emitter) -> list[tuple[str, str | None, str]]:
    return [
        (event["event_type"], event["stage_id"], event["status"])
        for event in emitter.events
    ]


def _arrange_unusable_emitter(monkeypatch, tmp_path) -> None:
    """The adapter's own start, looking for an emitter command that is missing."""

    monkeypatch.setattr(LocalTelemetryEmitter, "start", classmethod(ADAPTER_START))
    monkeypatch.setattr(interface_client, "start_emitter", INTERFACE_START_EMITTER)
    monkeypatch.setenv(
        "MICROCOSM_EMITTER_COMMAND", str(tmp_path / "missing" / "microcosm-emitter")
    )


def test_the_variables_keep_their_names() -> None:
    # The Modal stage runner sets the run ID under this name, from a launching
    # checkout that may be newer or older than the tool's pinned commit.
    assert program_telemetry.TELEMETRY_RUN_ID_VARIABLE == RUN_ID_VARIABLE
    assert program_telemetry.TELEMETRY_PARENT_PID_VARIABLE == PARENT_PID_VARIABLE


def test_a_given_run_id_names_the_plan_s_run_as_its_candidate() -> None:
    identity = program_telemetry.telemetry_run_identity(
        "us_acs_local_release",
        environ={RUN_ID_VARIABLE: "route-a.base-2.materialize"},
    )
    # Everything before the last "." is the plan's run_id, which may itself
    # hold dots; the stage names the runner appends hold none.
    assert identity == ("route-a.base-2.materialize", "route-a.base-2")


def test_a_given_run_id_without_a_dot_has_no_candidate() -> None:
    assert program_telemetry.telemetry_run_identity(
        "us_acs_local_release", environ={RUN_ID_VARIABLE: "chosen-by-hand"}
    ) == ("chosen-by-hand", None)


def test_without_a_given_run_id_each_attempt_makes_its_own() -> None:
    now = datetime(2026, 11, 4, 12, 30, 5, tzinfo=UTC)
    first, first_candidate = program_telemetry.telemetry_run_identity(
        "us_puf_support_base", environ={}, now=now
    )
    second, _ = program_telemetry.telemetry_run_identity(
        "us_puf_support_base", environ={RUN_ID_VARIABLE: ""}, now=now
    )
    prefix = "us-puf-support-base-20261104T123005Z-"
    assert first.startswith(prefix) and second.startswith(prefix)
    assert len(first) == len(prefix) + 8
    int(first.removeprefix(prefix), 16)
    assert first != second
    assert first_candidate is None


@pytest.mark.parametrize(
    "given",
    ["has space", "-leading-dash", ".leading-dot", "x" * 193, "slash/inside"],
)
def test_a_given_run_id_the_collector_would_refuse_stops_before_any_emitter(
    monkeypatch, fake_telemetry_emitters, given: str
) -> None:
    monkeypatch.setenv(RUN_ID_VARIABLE, given)
    with pytest.raises(ValueError, match=RUN_ID_VARIABLE):
        with program_telemetry.program_telemetry(pipeline="us_acs_local_release"):
            pytest.fail("the body ran")
    assert fake_telemetry_emitters == []


def test_a_program_that_returns_records_its_stages_and_completes(
    monkeypatch, fake_telemetry_emitters
) -> None:
    monkeypatch.setenv(RUN_ID_VARIABLE, "plan-run.all")
    with program_telemetry.program_telemetry(
        pipeline="us_acs_local_release"
    ) as emitter:
        emitter.transition_stage("materialize", message="Materializing.")
        emitter.transition_stage("calibrate")

    [started] = fake_telemetry_emitters
    assert started is emitter
    assert (
        started.run.run_id,
        started.run.candidate_id,
        started.run.country_code,
        started.run.pipeline,
        started.run.release_id,
        started.run.run_kind,
    ) == ("plan-run.all", "plan-run", "US", "us_acs_local_release", None, "build")
    assert _event_summary(started) == [
        ("stage", "materialize", "started"),
        ("stage", "materialize", "completed"),
        ("stage", "calibrate", "started"),
        ("stage", "calibrate", "completed"),
        ("run", "complete", "completed"),
    ]
    # Recording the outcome stopped the emitter.
    assert not started.available


@pytest.mark.parametrize("code", [None, 0])
def test_a_program_that_exits_with_status_0_completes(
    fake_telemetry_emitters, code
) -> None:
    with pytest.raises(SystemExit):
        with program_telemetry.program_telemetry(
            pipeline="us_acs_local_release"
        ) as emitter:
            emitter.transition_stage("package")
            raise SystemExit(code)
    assert _event_summary(fake_telemetry_emitters[0])[-1] == (
        "run",
        "complete",
        "completed",
    )


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("calibration diverged"),
        SystemExit("Checkpointed stage 'x' failed with exit code 3."),
        SystemExit(2),
        KeyboardInterrupt(),
    ],
    ids=["exception", "exit-message", "exit-code", "interrupt"],
)
def test_a_program_that_raises_records_the_failure_and_re_raises(
    fake_telemetry_emitters, error: BaseException
) -> None:
    with pytest.raises(type(error)) as raised:
        with program_telemetry.program_telemetry(
            pipeline="us_acs_local_release"
        ) as emitter:
            emitter.transition_stage("materialize")
            emitter.transition_stage("calibrate")
            raise error
    assert raised.value is error
    [started] = fake_telemetry_emitters
    assert _event_summary(started) == [
        ("stage", "materialize", "started"),
        ("stage", "materialize", "completed"),
        ("stage", "calibrate", "started"),
        ("stage", "calibrate", "failed"),
        ("run", "failed", "failed"),
    ]
    failure = started.events[-1]["details"]
    assert failure["error_type"] == type(error).__name__
    assert failure["failed_during"] == "calibrate"
    assert not started.available


class UnhandledSigtermError(Exception):
    """The test's own SIGTERM handler ran: nothing in the program took the signal."""


@pytest.fixture
def sigterm_guard():
    """A SIGTERM handler that raises, so a SIGTERM the program misses fails the test.

    Without it, a SIGTERM the program does not handle would end the test
    process itself.
    """

    def reached_the_default(signum, frame) -> None:
        raise UnhandledSigtermError()

    previous = signal.signal(signal.SIGTERM, reached_the_default)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_a_sigterm_is_recorded_as_a_termination_in_the_stage_it_stopped(
    fake_telemetry_emitters, sigterm_guard
) -> None:
    # The Modal runner's budget, a supervisor or kill: the program records
    # the termination in the stage that was running, then exits 143, as the
    # UK builds do.
    outcome: BaseException | None = None
    try:
        with program_telemetry.program_telemetry(
            pipeline="us_acs_local_release"
        ) as emitter:
            emitter.transition_stage("materialize")
            emitter.transition_stage("calibrate")
            os.kill(os.getpid(), signal.SIGTERM)
    except BaseException as raised:  # noqa: BLE001 - the test inspects it
        outcome = raised
    assert isinstance(outcome, SystemExit), repr(outcome)
    assert outcome.code == 128 + signal.SIGTERM
    [started] = fake_telemetry_emitters
    assert _event_summary(started)[-2:] == [
        ("stage", "calibrate", "failed"),
        ("run", "failed", "failed"),
    ]
    failure = started.events[-1]["details"]
    assert (failure["failed_during"], failure["failure_class"]) == (
        "calibrate",
        "terminated",
    )
    assert failure["error_code"] == "TERMINATED"
    assert not started.available


def test_a_program_leaves_the_sigterm_handler_as_it_found_it(
    fake_telemetry_emitters, sigterm_guard
) -> None:
    before = signal.getsignal(signal.SIGTERM)
    with program_telemetry.program_telemetry(pipeline="us_acs_local_release"):
        inside = signal.getsignal(signal.SIGTERM)
    assert inside is not before
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_program_that_must_send_telemetry_stops_before_its_body_without_an_emitter(
    monkeypatch, tmp_path
) -> None:
    _arrange_unusable_emitter(monkeypatch, tmp_path)
    monkeypatch.setenv("MICROCOSM_TELEMETRY_REQUIRED", "1")
    with pytest.raises(TelemetryRequiredError):
        with program_telemetry.program_telemetry(pipeline="us_acs_local_release"):
            pytest.fail("the body ran")


@pytest.mark.parametrize("required", ["0", None])
def test_a_program_that_need_not_send_telemetry_runs_without_an_emitter(
    monkeypatch, capfd, tmp_path, required: str | None
) -> None:
    _arrange_unusable_emitter(monkeypatch, tmp_path)
    if required is None:
        monkeypatch.delenv("MICROCOSM_TELEMETRY_REQUIRED", raising=False)
    else:
        monkeypatch.setenv("MICROCOSM_TELEMETRY_REQUIRED", required)
    ran = []
    with program_telemetry.program_telemetry(
        pipeline="us_acs_local_release"
    ) as emitter:
        assert not emitter.available
        emitter.transition_stage("materialize")
        ran.append("materialize")
    assert ran == ["materialize"]
    assert "command_not_found" in capfd.readouterr().err


def test_a_child_stage_whose_parent_records_it_knows_so() -> None:
    environment = program_telemetry.child_stage_environment({"PATH": "/usr/bin"})
    assert environment == {"PATH": "/usr/bin", PARENT_PID_VARIABLE: str(os.getpid())}
    # In the child, the parent's process ID is its own parent's.
    assert program_telemetry.telemetry_recorded_by_parent(
        {PARENT_PID_VARIABLE: str(os.getppid())}
    )
    # A value left in a shell names some other process: a stage run by hand
    # records its own telemetry.
    for environ in (
        {},
        {PARENT_PID_VARIABLE: str(os.getpid())},
        {PARENT_PID_VARIABLE: ""},
    ):
        assert not program_telemetry.telemetry_recorded_by_parent(environ)
