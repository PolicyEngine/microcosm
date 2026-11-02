"""Microcosm's adapter over ``microcosm-emitter-interface``, without an emitter.

The adapter starts the separately installed ``microcosm-emitter`` command
through the interface's ``EmitterClient``; these tests replace that client
with an in-memory one, or let the interface fail to find the command. A
stand-in command answers ``microcosm-emitter version`` where a test needs an
installed release; no test runs a real emitter.
"""

import json
import os
import traceback
import typing

import pytest
from microcosm_emitter_interface import (
    EmitterClient,
    EmitterStartupError,
    HuggingFaceSettings,
    OrrerySettings,
    TelemetrySettings,
)
from microcosm_emitter_interface import client as interface_client
from microcosm_emitter_interface import launcher as interface_launcher
from microcosm_emitter_interface.launcher import StartupFailureReason

from microcosm.build import telemetry_emitter as adapter
from test_support.microcosm_build.emitter import FakeEmitterClient

START = adapter.LocalTelemetryEmitter.start.__func__
#: The interface's own startup, captured before the shared fixture forbids it.
INTERFACE_START_EMITTER = interface_launcher.start_emitter


@pytest.fixture(autouse=True)
def no_installed_emitter_command(monkeypatch, tmp_path_factory):
    """Name a missing command, so no test finds an emitter installed on PATH."""
    missing = tmp_path_factory.mktemp("no-emitter") / "microcosm-emitter"
    monkeypatch.setenv("MICROCOSM_EMITTER_COMMAND", str(missing))


def _record_starts(monkeypatch, client=None):
    """Replace ``EmitterClient.start``; return the options of every call."""
    starts = []

    def start(cls, **options):
        starts.append(options)
        return client if client is not None else FakeEmitterClient()

    monkeypatch.setattr(adapter.EmitterClient, "start", classmethod(start))
    return starts


def test_starts_one_emitter_with_telemetry_graph_and_hugging_face_modules(
    monkeypatch, tmp_path
):
    client = FakeEmitterClient()
    starts = _record_starts(monkeypatch, client)
    monkeypatch.setattr(adapter, "runtime_identity", lambda: {"commit": "synthetic"})
    path = tmp_path / "jobs.sqlite3"
    emitter = START(
        adapter.LocalTelemetryEmitter,
        run_id="run-one",
        country_code="uk",
        pipeline="full",
        spool_path=path,
        heartbeat_seconds=123,
    )

    assert len(starts) == 1
    options = starts[0]
    assert adapter.EmitterClient is EmitterClient
    assert options["spool_path"] == path
    assert options["startup_timeout"] == 15
    telemetry = options["telemetry"]
    assert isinstance(telemetry, TelemetrySettings)
    assert telemetry.run.run_id == "run-one"
    assert telemetry.run.country_code == "uk"
    assert telemetry.run.pipeline == "full"
    assert telemetry.identity == {"commit": "synthetic"}
    assert telemetry.heartbeat_seconds == 123
    assert telemetry.development_collector_url is None
    assert isinstance(options["orrery"], OrrerySettings)
    assert options["orrery"].development_collector_url is None
    hugging_face = options["hugging_face"]
    assert isinstance(hugging_face, HuggingFaceSettings)
    # The hand-over tables stay in their own SQLite file beside the spool.
    assert hugging_face.configuration(path) == {
        "handover_database_path": str(tmp_path / "jobs.handovers.sqlite3")
    }
    assert emitter.client is client
    assert emitter.run is telemetry.run
    # The emitter records the run start from its configuration; the adapter
    # sends no second start request, and opens no database itself.
    assert client.telemetry_messages == []
    assert not path.exists()


def test_telemetry_requests_reach_the_interface_with_their_fields():
    client = FakeEmitterClient()
    emitter = adapter.LocalTelemetryEmitter(
        run=adapter.TelemetryRun("run-one", "uk", "full"),
        client=client,
        close_service_on_finish=False,
    )

    emitter.stage("stage-that-is-not-in-a-fixed-list", rows=3)
    emitter.transition_stage("solve", message="Solving.", rows=4)
    emitter.progress("solve", done=1, total=2, unit="chunks", elapsed_seconds=0.5)
    emitter.calibration_progress({"kind": "calibration_epoch", "epoch": 1})
    emitter.transition_calibration_progress({"kind": "calibration_epoch", "epoch": 2})
    emitter.emit(event_type="progress", status="completed", stage_id="node")
    emitter.block(
        phase="post_calibration",
        blocking_gate_ids=("uk.local.fit",),
        blocking_failure_count=3,
        gate_statuses={"uk.local.fit": "FAIL"},
    )
    emitter.fail(
        ValueError("synthetic refusal"),
        failed_during="solve",
        failure_class="refused",
        error_code="BUILD_REFUSED",
    )
    emitter.complete()

    assert client.telemetry_messages == [
        {
            "command": "stage",
            "stage_id": "stage-that-is-not-in-a-fixed-list",
            "status": "started",
            "message": None,
            "details": {"rows": 3},
        },
        {
            "command": "transition_stage",
            "stage_id": "solve",
            "status": "running",
            "message": "Solving.",
            "details": {"rows": 4},
        },
        {
            "command": "progress",
            "stage_id": "solve",
            "done": 1,
            "total": 2,
            "unit": "chunks",
            "details": {"elapsed_seconds": 0.5},
        },
        {
            "command": "calibration_progress",
            "event": {"kind": "calibration_epoch", "epoch": 1},
        },
        {
            "command": "transition_calibration_progress",
            "event": {"kind": "calibration_epoch", "epoch": 2},
        },
        {
            "command": "emit",
            "event_type": "progress",
            "stage_id": "node",
            "status": "completed",
            "message": None,
            "details": {},
        },
        {
            "command": "block",
            "phase": "post_calibration",
            "blocking_gate_ids": ["uk.local.fit"],
            "blocking_failure_count": 3,
            "gate_statuses": {"uk.local.fit": "FAIL"},
        },
        {
            "command": "fail",
            "message": "synthetic refusal",
            "error_type": "ValueError",
            "failed_during": "solve",
            "failure_class": "refused",
            "error_code": "BUILD_REFUSED",
        },
        {"command": "complete"},
    ]


def test_a_missing_emitter_command_leaves_the_build_running(
    monkeypatch, capsys, tmp_path
):
    # Let the interface itself look for the command, and fail to find it.
    monkeypatch.setattr(interface_client, "start_emitter", INTERFACE_START_EMITTER)
    # An adapter that still started an in-process host must not start one here.
    monkeypatch.setattr(
        adapter,
        "start_services",
        lambda *args, **options: pytest.fail("started an in-process emitter host"),
        raising=False,
    )
    monkeypatch.setenv(
        "MICROCOSM_EMITTER_COMMAND", str(tmp_path / "missing" / "microcosm-emitter")
    )
    monkeypatch.setattr(adapter, "runtime_identity", lambda: {"commit": "synthetic"})
    emitter = START(
        adapter.LocalTelemetryEmitter,
        run_id="run-one",
        country_code="uk",
        pipeline="full",
        spool_path=tmp_path / "jobs.sqlite3",
    )

    assert not emitter.available
    assert "command_not_found" in capsys.readouterr().err
    emitter.transition_stage("solve")
    assert emitter.publication_inventory(tmp_path, {}) is None
    assert emitter.publish_graph(
        tmp_path,
        {"version": 1, "publication_id": "independent", "files": []},
        wait_seconds=0,
    ) == {
        "version": 1,
        "publication_id": "independent",
        "status": "failed",
        "error_code": "local_service_unavailable",
    }
    emitter.complete()
    emitter.close()
    assert list(tmp_path.iterdir()) == []


def _version_answer(emitter_version):
    """What ``microcosm-emitter version`` prints, for one installed release."""
    description = {"protocol_versions": [1]}
    if emitter_version is not None:
        description["emitter_version"] = emitter_version
    return json.dumps(description)


def _install_emitter_command(monkeypatch, directory, version_answer):
    """Name a stand-in ``microcosm-emitter`` command in MICROCOSM_EMITTER_COMMAND.

    It prints ``version_answer`` for ``microcosm-emitter version``. Started
    any other way, as the emitter itself would be, it creates the returned
    file and exits.
    """
    directory.mkdir()
    answer = directory / "version-answer"
    answer.write_text(version_answer)
    started = directory / "emitter-started"
    command = directory / "microcosm-emitter"
    command.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = version ]; then\n'
        f"  cat '{answer}'\n"
        "  exit 0\n"
        "fi\n"
        f"touch '{started}'\n"
        "exit 1\n"
    )
    command.chmod(0o755)
    monkeypatch.setenv("MICROCOSM_EMITTER_COMMAND", str(command))
    return command, started


def _start_build_emitter(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "runtime_identity", lambda: {"commit": "synthetic"})
    return START(
        adapter.LocalTelemetryEmitter,
        run_id="run-one",
        country_code="uk",
        pipeline="full",
        spool_path=tmp_path / "events.sqlite3",
    )


@pytest.mark.parametrize(
    ("version_answer", "reported"),
    [
        # 0.4.0 to 0.5.1 accept protocol 1 but have no Hugging Face module.
        (_version_answer("0.5.1"), "reports version 0.5.1"),
        (_version_answer("0.4.0"), "reports version 0.4.0"),
        (_version_answer(None), "reports no version"),
        (_version_answer("not-a-version"), "reports version not-a-version"),
        ("not JSON", "reports no version"),
    ],
    ids=["0.5.1", "0.4.0", "no-version", "not-a-version", "not-json"],
)
def test_an_emitter_older_than_0_6_0_is_never_started_and_the_build_continues(
    monkeypatch, capsys, tmp_path, version_answer, reported
):
    # The emitter's telemetry module records the run's start as soon as the
    # emitter starts, before it can answer ``hello``. An emitter started only
    # to be stopped again would leave the collector a run that started and
    # never finished, so the version is read before anything starts.
    client = FakeEmitterClient()
    starts = _record_starts(monkeypatch, client)
    _, started = _install_emitter_command(monkeypatch, tmp_path / "bin", version_answer)
    emitter = _start_build_emitter(monkeypatch, tmp_path)

    assert starts == []
    assert not started.exists()
    assert not emitter.available
    warning = capsys.readouterr().err
    assert f"The installed microcosm-emitter {reported};" in warning
    assert "this build needs 0.6.0 or later" in warning
    assert "uv tool install --upgrade 'microcosm-emitter>=0.6.0'" in warning
    # Nothing reaches an emitter, and nothing raises.
    emitter.transition_stage("solve")
    emitter.complete()
    emitter.close()
    assert client.calls == []


@pytest.mark.parametrize("installed_version", ["0.6.0", "0.7.0", "0.10.0", "1.0.0"])
def test_an_emitter_from_0_6_0_on_starts(
    monkeypatch, capsys, tmp_path, installed_version
):
    client = FakeEmitterClient()
    starts = _record_starts(monkeypatch, client)
    command, started = _install_emitter_command(
        monkeypatch, tmp_path / "bin", _version_answer(installed_version)
    )
    emitter = _start_build_emitter(monkeypatch, tmp_path)

    # The command whose version was read is the one started.
    assert [options["command"] for options in starts] == [str(command)]
    assert not started.exists()
    assert emitter.available
    assert "microcosm-emitter" not in capsys.readouterr().err


@pytest.mark.parametrize("operation", ["inventory", "publish"])
def test_a_short_lived_graph_emitter_older_than_0_6_0_is_never_started(
    monkeypatch, capsys, tmp_path, operation
):
    client = FakeEmitterClient(modules={"orrery"})
    starts = _record_starts(monkeypatch, client)
    _, started = _install_emitter_command(
        monkeypatch, tmp_path / "bin", _version_answer("0.5.1")
    )
    inventory = {"version": 1, "publication_id": "independent", "files": []}

    if operation == "inventory":
        assert adapter.publication_inventory(tmp_path, {}) is None
    else:
        receipt = adapter.publish_graph(tmp_path, inventory, wait_seconds=0)
        assert receipt["status"] == "failed"

    assert starts == []
    assert not started.exists()
    assert client.calls == []
    assert "this build needs 0.6.0 or later" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["default_spool_path", "runtime_identity"])
def test_metadata_or_cache_failure_never_starts_an_emitter(
    monkeypatch, capsys, failure
):
    def fail():
        raise OSError("private setup detail")

    starts = _record_starts(monkeypatch)
    monkeypatch.setattr(adapter, failure, fail)
    emitter = START(
        adapter.LocalTelemetryEmitter,
        run_id="failed",
        country_code="uk",
        pipeline="full",
    )

    assert starts == []
    assert not emitter.available
    assert emitter.client is None
    warning = capsys.readouterr().err
    assert "OSError" in warning
    assert "private setup detail" not in warning
    emitter.fail(ValueError("refused"), error_code="BUILD_REFUSED")
    emitter.close()


@pytest.mark.parametrize("outcome", ["blocked", "failed", "completed"])
def test_graph_publication_precedes_emitter_shutdown_after_an_outcome(
    tmp_path, outcome
):
    client = FakeEmitterClient()
    emitter = adapter.LocalTelemetryEmitter(
        run=adapter.TelemetryRun("synthetic", "uk", "full"),
        client=client,
        close_service_on_finish=False,
    )
    emitter.transition_stage("solve")
    if outcome == "blocked":
        emitter.block(
            phase="post_calibration",
            blocking_gate_ids=["uk.local.fit"],
            blocking_failure_count=3,
        )
    elif outcome == "failed":
        emitter.fail(ValueError("synthetic refusal"), error_code="BUILD_REFUSED")
    else:
        emitter.complete()

    # Recording the outcome leaves the emitter running for later hand-overs.
    assert "close" not in client.calls
    assert emitter.available
    (tmp_path / "graph.orrery.json").write_text("{}")
    inventory = emitter.publication_inventory(tmp_path, {"graph.orrery.json": "graph"})
    assert inventory is not None
    assert inventory["files"][0]["name"] == "graph.orrery.json"
    assert emitter.publish_graph(tmp_path, inventory, wait_seconds=0)["status"] == (
        "pending"
    )
    emitter.close()
    emitter.close()
    assert client.calls[-3:] == [
        "orrery_inventory",
        "hand_over_orrery_inputs",
        "close",
    ]
    assert client.calls.count("close") == 1
    assert not emitter.available


@pytest.mark.parametrize("outcome", ["complete", "fail", "block"])
def test_by_default_an_outcome_shuts_the_emitter_down(outcome):
    client = FakeEmitterClient()
    emitter = adapter.LocalTelemetryEmitter(
        run=adapter.TelemetryRun("synthetic", "us", "release"), client=client
    )
    if outcome == "complete":
        emitter.complete()
    elif outcome == "fail":
        emitter.fail(ValueError("synthetic failure"), error_code="BUILD_FAILED")
    else:
        emitter.block(phase="preflight", blocking_gate_ids=[], blocking_failure_count=1)

    assert client.closed
    assert not emitter.available
    emitter.close()
    assert client.calls.count("close") == 1


def test_an_unavailable_emitter_answers_without_raising(tmp_path):
    emitter = adapter.LocalTelemetryEmitter(
        run=adapter.TelemetryRun("run-one", "uk", "full"), client=None
    )
    value = {"version": 1, "publication_id": "independent", "files": []}

    assert not emitter.available
    emitter.stage("one")
    assert emitter.publication_inventory(tmp_path, {}) is None
    assert emitter.publish_graph(tmp_path, value, wait_seconds=0) == {
        "version": 1,
        "publication_id": "independent",
        "status": "failed",
        "error_code": "local_service_unavailable",
    }
    emitter.block(phase="preflight", blocking_gate_ids=[], blocking_failure_count=1)
    emitter.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operation", ["inventory", "publish"])
def test_graph_requests_without_a_build_emitter_use_a_short_lived_one(
    monkeypatch, tmp_path, operation
):
    client = FakeEmitterClient(modules={"orrery"})
    starts = _record_starts(monkeypatch, client)
    (tmp_path / "graph.orrery.json").write_text("{}")
    roles = {"graph.orrery.json": "graph"}

    if operation == "inventory":
        inventory = adapter.publication_inventory(tmp_path, roles)
        assert inventory is not None
        assert inventory["files"][0]["role"] == "graph"
    else:
        inventory = {"version": 1, "publication_id": "independent", "files": []}
        receipt = adapter.publish_graph(tmp_path, inventory, wait_seconds=0)
        assert receipt["status"] == "pending"

    assert len(starts) == 1
    assert set(starts[0]) >= {"orrery"}
    assert starts[0].get("telemetry") is None
    assert starts[0].get("hugging_face") is None
    assert isinstance(starts[0]["orrery"], OrrerySettings)
    assert client.calls[-1] == "close"


def test_default_test_emitter_records_requests_in_memory(fake_telemetry_emitters):
    emitter = adapter.LocalTelemetryEmitter.start(
        run_id="run-one", country_code="uk", pipeline="full"
    )
    emitter.stage("one")
    emitter.complete()
    assert not emitter.available
    assert emitter.events[0]["stage_id"] == "one"
    assert emitter.events[-1]["event_type"] == "run"
    assert emitter.events[-1]["status"] == "completed"


def test_fixture_intercepts_preimported_build_entrypoint(
    monkeypatch, fake_telemetry_emitters
):
    from microcosm.build.uk_runtime import rowwise_staging
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter

    def fail(*args, **kwargs):
        pytest.fail("test started a real emitter")

    monkeypatch.setattr(adapter.EmitterClient, "start", classmethod(fail))
    for start in (
        rowwise_staging.start_local_telemetry_emitter_service,
        adapter.LocalTelemetryEmitter.start,
    ):
        emitter = start(run_id="fake", country_code="GB", pipeline="test")
        assert isinstance(emitter, FakeTelemetryEmitter)
        assert emitter in fake_telemetry_emitters


def test_fixture_forbids_starting_the_installed_emitter(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="real emitter"):
        EmitterClient.start(
            orrery=OrrerySettings(), spool_path=tmp_path / "events.sqlite3"
        )


def test_fixture_isolates_cached_huggingface_credentials(tmp_path):
    from pathlib import Path

    from huggingface_hub import constants, get_token

    assert Path(constants.HF_TOKEN_PATH).is_relative_to(tmp_path)
    assert get_token() is None


# A build that must send telemetry (MICROCOSM_TELEMETRY_REQUIRED=1) ------------

REQUIRED_VARIABLE = "MICROCOSM_TELEMETRY_REQUIRED"
#: A made-up collector token. Tests never use a real credential.
DUMMY_COLLECTOR_TOKEN = "dummy-collector-token-not-a-credential"
#: A short startup limit, so a ``version`` that never answers is given up on
#: quickly.
SHORT_STARTUP_TIMEOUT_SECONDS = 2.0
#: Each way the installed emitter can be unusable: the reason a build that
#: must send telemetry stops with, and text its error must contain.
UNUSABLE_EMITTERS = {
    "missing": ("command_not_found", "`uv tool install microcosm-emitter`"),
    "too_old": (
        "outdated_emitter",
        "`uv tool install --upgrade "
        f"'microcosm-emitter>={adapter.MINIMUM_EMITTER_VERSION}'`",
    ),
    "version_times_out": ("version_timeout", "did not answer within 2 s"),
    "version_fails": ("version_failed", "exited 3"),
    "no_protocol_1": ("unsupported_protocol_version", "not protocol version 1"),
    "hello_fails": ("hello_failed", "`hello` answer"),
    "no_telemetry_module": (
        "telemetry_module_not_loaded",
        "without its telemetry module",
    ),
    "refuses_collector_token": ("child_exit", "exited before it was ready"),
}
#: The failures ``EmitterClient.start`` reports. Interface 0.7.0 then prints
#: "Local Microcosm emitter failed to start (<reason>); the build will
#: continue." before the adapter's error, which says that line does not apply.
REPORTED_BY_THE_INTERFACE = {"hello_fails", "refuses_collector_token"}
#: For each startup failure interface 0.7.0 can report, text the adapter's
#: error must contain.
INTERFACE_STARTUP_FAILURES = {
    "command_not_found": "`uv tool install microcosm-emitter`",
    "invalid_configuration": "invalid or too large",
    "configuration_timeout": "did not take its module settings",
    "child_exit": '"Service startup failed (startup_failure).", without the cause',
    "readiness_timeout": "was not ready within its startup time limit",
    "unsupported_protocol_version": "`hello` answer",
    "startup_failure": "could not be started",
}


def _install_emitter_script(monkeypatch, directory, version_branch):
    """Name a stand-in ``microcosm-emitter`` whose ``version`` runs ``version_branch``.

    Started any other way, as the emitter itself would be, it creates the
    returned file and exits.
    """
    directory.mkdir()
    started = directory / "emitter-started"
    command = directory / "microcosm-emitter"
    command.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = version ]; then\n'
        f"{version_branch}\n"
        "fi\n"
        f"touch '{started}'\n"
        "exit 1\n"
    )
    command.chmod(0o755)
    monkeypatch.setenv("MICROCOSM_EMITTER_COMMAND", str(command))
    return started


def _install_emitter_that_refuses_its_collector_token(monkeypatch, directory):
    """Name a stand-in for an emitter that cannot use its collector token.

    It answers ``microcosm-emitter version`` with the minimum release.
    Started as the emitter host, it writes the names, never the values, of
    the ``MICROCOSM_COLLECTOR_*`` variables it inherited, prints the
    emitter's refusal and exits 1 before it is ready. Emitter 0.11.0 and
    later do that when ``MICROCOSM_COLLECTOR_TOKEN``,
    ``MICROCOSM_COLLECTOR_TOKEN_EXPIRES_AT`` and ``MICROCOSM_COLLECTOR_ORIGIN``
    are set only in part, cannot be read, or name another collector.
    """
    directory.mkdir()
    inherited_names = directory / "inherited-collector-variables"
    command = directory / "microcosm-emitter"
    command.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = version ]; then\n'
        f"  echo '{_version_answer(adapter.MINIMUM_EMITTER_VERSION)}'\n"
        "  exit 0\n"
        "fi\n"
        "env | cut -d= -f1 | grep '^MICROCOSM_COLLECTOR_' | sort "
        f"> '{inherited_names}'\n"
        "echo 'The Microcosm emitter cannot start: MICROCOSM_COLLECTOR_TOKEN, "
        "MICROCOSM_COLLECTOR_TOKEN_EXPIRES_AT and MICROCOSM_COLLECTOR_ORIGIN must "
        "be set together. It exits before accepting any request from the build.' "
        ">&2\n"
        "exit 1\n"
    )
    command.chmod(0o755)
    monkeypatch.setenv("MICROCOSM_EMITTER_COMMAND", str(command))
    return inherited_names


def _arrange_unusable_emitter(monkeypatch, tmp_path, kind):
    """Make the installed emitter unusable in one of ``UNUSABLE_EMITTERS``' ways.

    Returns the client a recorded ``EmitterClient.start`` hands back, or None
    where the interface's own startup runs (or nothing starts), and the list
    of recorded starts, or None.
    """
    if kind == "missing":
        # The autouse fixture names a missing command; let the interface
        # look for it itself.
        monkeypatch.setattr(interface_client, "start_emitter", INTERFACE_START_EMITTER)
        return None, None
    if kind == "refuses_collector_token":
        monkeypatch.setattr(interface_client, "start_emitter", INTERFACE_START_EMITTER)
        # One of the three variables without the other two.
        monkeypatch.setenv("MICROCOSM_COLLECTOR_TOKEN", DUMMY_COLLECTOR_TOKEN)
        _install_emitter_that_refuses_its_collector_token(monkeypatch, tmp_path / "bin")
        return None, None
    if kind in {"too_old", "version_times_out", "version_fails", "no_protocol_1"}:
        # A required build never starts it: the version answer settles it.
        # Started anyway (as a build that need not send telemetry starts an
        # emitter without protocol 1), it fails as the interface reports.
        starts = _record_starts(
            monkeypatch,
            EmitterClient(
                startup_error=EmitterStartupError("unsupported_protocol_version")
            ),
        )
        if kind == "too_old":
            _install_emitter_command(
                monkeypatch, tmp_path / "bin", _version_answer("0.5.1")
            )
        elif kind == "version_times_out":
            _install_emitter_script(monkeypatch, tmp_path / "bin", "  sleep 30")
        elif kind == "version_fails":
            _install_emitter_script(
                monkeypatch, tmp_path / "bin", "  echo broken >&2\n  exit 3"
            )
        else:
            answer = json.loads(_version_answer(adapter.MINIMUM_EMITTER_VERSION))
            answer["protocol_versions"] = [2]
            _install_emitter_command(monkeypatch, tmp_path / "bin", json.dumps(answer))
        return None, starts
    _install_emitter_command(
        monkeypatch,
        tmp_path / "bin",
        _version_answer(adapter.MINIMUM_EMITTER_VERSION),
    )
    if kind == "hello_fails":
        # What EmitterClient.start returns when ``hello`` fails, although
        # ``version`` listed protocol version 1.
        client = EmitterClient(
            startup_error=EmitterStartupError("unsupported_protocol_version")
        )
    else:
        client = FakeEmitterClient(modules={"orrery", "huggingface"})
    return client, _record_starts(monkeypatch, client)


def _start_with_a_short_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "runtime_identity", lambda: {"commit": "synthetic"})
    return START(
        adapter.LocalTelemetryEmitter,
        run_id="run-one",
        country_code="uk",
        pipeline="full",
        spool_path=tmp_path / "events.sqlite3",
        startup_timeout_seconds=SHORT_STARTUP_TIMEOUT_SECONDS,
    )


def _says_the_interface_line_does_not_apply(message):
    return '"the build will continue"' in message and "does not apply" in message


@pytest.mark.parametrize("kind", list(UNUSABLE_EMITTERS))
def test_a_build_that_must_send_telemetry_stops_when_its_emitter_is_unusable(
    monkeypatch, capfd, tmp_path, kind
):
    client, starts = _arrange_unusable_emitter(monkeypatch, tmp_path, kind)
    monkeypatch.setenv(REQUIRED_VARIABLE, "1")

    with pytest.raises(adapter.TelemetryRequiredError) as raised:
        _start_with_a_short_limit(monkeypatch, tmp_path)

    error = raised.value
    reason, explanation = UNUSABLE_EMITTERS[kind]
    assert error.reason == reason
    message = str(error)
    assert message.startswith(
        "MICROCOSM_TELEMETRY_REQUIRED=1, so this build must send telemetry, but "
    )
    assert explanation in message
    assert message.endswith("The build stops before doing any work.")
    if isinstance(client, FakeEmitterClient):
        # It started without the telemetry module: nothing was sent to it,
        # and it was told to stop.
        assert client.telemetry_messages == []
        assert client.closed
    if client is None and starts is not None:
        assert starts == []
    err = capfd.readouterr().err
    if kind in REPORTED_BY_THE_INTERFACE:
        assert _says_the_interface_line_does_not_apply(message)
    else:
        # The adapter found it before the interface started anything, so
        # nothing printed that the build continues.
        assert "continue" not in err
        assert not _says_the_interface_line_does_not_apply(message)
    if kind in {"version_times_out", "version_fails"}:
        # A version check that failed is not advice to upgrade.
        assert "--upgrade" not in message
    # The message names the command the adapter looked for or ran.
    assert f"`{os.environ['MICROCOSM_EMITTER_COMMAND']}" in message


def test_a_build_that_must_send_telemetry_stops_when_its_emitter_refuses_the_collector_token(
    monkeypatch, capfd, tmp_path
):
    # The interface offers no call that reports whether the emitter
    # authenticated to the collector: the telemetry module registers the run
    # and exchanges credentials later, on its own thread. What a build can
    # see at start is the emitter's refusal of collector-token variables it
    # cannot use: it exits before it is ready.
    _arrange_unusable_emitter(monkeypatch, tmp_path, "refuses_collector_token")
    inherited_names = tmp_path / "bin" / "inherited-collector-variables"
    monkeypatch.setenv(REQUIRED_VARIABLE, "1")

    with pytest.raises(adapter.TelemetryRequiredError) as raised:
        _start_with_a_short_limit(monkeypatch, tmp_path)

    assert raised.value.reason == "child_exit"
    assert raised.value.exit_code == 1
    # The emitter inherited the build's collector-token variable.
    assert inherited_names.read_text().split() == ["MICROCOSM_COLLECTOR_TOKEN"]
    output = capfd.readouterr()
    # The emitter's own reason reaches the build's standard error, and so
    # does the interface's line, which the adapter's error supersedes.
    assert "The Microcosm emitter cannot start: MICROCOSM_COLLECTOR_TOKEN" in (
        output.err
    )
    assert "(child_exit); the build will continue." in output.err
    assert _says_the_interface_line_does_not_apply(str(raised.value))
    shown = output.out + output.err + "".join(traceback.format_exception(raised.value))
    assert DUMMY_COLLECTOR_TOKEN not in shown


def test_every_startup_failure_the_interface_names_has_its_own_message():
    reasons = set(typing.get_args(StartupFailureReason.__value__))
    assert set(INTERFACE_STARTUP_FAILURES) == reasons
    assert set(adapter._STARTUP_PROBLEMS) == reasons


@pytest.mark.parametrize("reason", list(INTERFACE_STARTUP_FAILURES))
def test_a_required_build_s_error_for_each_startup_failure_the_interface_reports(
    monkeypatch, tmp_path, reason
):
    command, _ = _install_emitter_command(
        monkeypatch, tmp_path / "bin", _version_answer(adapter.MINIMUM_EMITTER_VERSION)
    )
    exit_code = 3 if reason == "child_exit" else None
    _record_starts(
        monkeypatch,
        EmitterClient(startup_error=EmitterStartupError(reason, exit_code=exit_code)),
    )
    monkeypatch.setenv(REQUIRED_VARIABLE, "1")

    with pytest.raises(adapter.TelemetryRequiredError) as raised:
        _start_with_a_short_limit(monkeypatch, tmp_path)

    error = raised.value
    # ``version`` listed protocol version 1, so a failed ``hello`` is not an
    # unsupported protocol.
    expected = "hello_failed" if reason == "unsupported_protocol_version" else reason
    assert (error.reason, error.exit_code) == (expected, exit_code)
    message = str(error)
    assert INTERFACE_STARTUP_FAILURES[reason] in message
    assert _says_the_interface_line_does_not_apply(message)
    assert f"`{command}`" in message
    if reason == "child_exit":
        assert "(child_exit, exit code 3)" in message
        assert "set only in part" in message
        assert "traceback" in message


def test_a_required_build_s_error_for_an_emitter_killed_before_it_was_ready(
    monkeypatch, tmp_path
):
    # A signal leaves no message of the emitter's own: the error says so.
    command, _ = _install_emitter_command(
        monkeypatch, tmp_path / "bin", _version_answer(adapter.MINIMUM_EMITTER_VERSION)
    )
    _record_starts(
        monkeypatch,
        EmitterClient(startup_error=EmitterStartupError("child_exit", exit_code=-9)),
    )
    monkeypatch.setenv(REQUIRED_VARIABLE, "1")

    with pytest.raises(adapter.TelemetryRequiredError) as raised:
        _start_with_a_short_limit(monkeypatch, tmp_path)

    assert (raised.value.reason, raised.value.exit_code) == ("child_exit", -9)
    message = str(raised.value)
    assert f"`{command}` was killed by signal 9 before it was ready" in message
    assert "Service startup failed" not in message


#: Each way ``microcosm-emitter version`` can fail in a required build: the
#: stand-in's ``version`` branch, and text the error must contain.
VERSION_FAILURES = {
    "cannot_run": (None, "could not be run (FileNotFoundError)"),
    "exits": ("  echo broken >&2\n  exit 3", "exited 3"),
    "not_json": ("  echo not-json", "printed no JSON object"),
    "not_an_object": ("  echo '[1]'", "printed no JSON object"),
    "no_emitter_version": (
        """  echo '{"protocol_versions": [1]}'""",
        "printed no emitter_version",
    ),
    "not_a_release": (
        """  echo '{"emitter_version": "next", "protocol_versions": [1]}'""",
        "reported version 'next', which is not a release number",
    ),
    "no_protocol_list": (
        f"""  echo '{{"emitter_version": "{adapter.MINIMUM_EMITTER_VERSION}"}}'""",
        "printed no list of protocol versions",
    ),
}


@pytest.mark.parametrize("failure", list(VERSION_FAILURES))
def test_a_required_build_s_error_for_each_way_its_version_check_fails(
    monkeypatch, capfd, tmp_path, failure
):
    version_branch, explanation = VERSION_FAILURES[failure]
    starts = _record_starts(monkeypatch)
    if version_branch is None:
        # Executable, but its interpreter does not exist, so it cannot run.
        (tmp_path / "bin").mkdir()
        command = tmp_path / "bin" / "microcosm-emitter"
        command.write_text("#!/nonexistent/interpreter\n")
        command.chmod(0o755)
        monkeypatch.setenv("MICROCOSM_EMITTER_COMMAND", str(command))
        started = tmp_path / "bin" / "emitter-started"
    else:
        started = _install_emitter_script(
            monkeypatch, tmp_path / "bin", f"{version_branch}\n  exit 0"
        )
        command = tmp_path / "bin" / "microcosm-emitter"
    monkeypatch.setenv(REQUIRED_VARIABLE, "1")

    with pytest.raises(adapter.TelemetryRequiredError) as raised:
        _start_with_a_short_limit(monkeypatch, tmp_path)

    assert raised.value.reason == "version_failed"
    message = str(raised.value)
    assert explanation in message
    assert f"`{command} version`" in message
    # Advice to check that the command runs only when it did not run; never
    # advice to upgrade.
    assert ("Check that" in message and "runs on this machine" in message) is (
        failure == "cannot_run"
    )
    assert "--upgrade" not in message
    assert starts == []
    assert not started.exists()
    assert "continue" not in capfd.readouterr().err


@pytest.mark.parametrize("failure", ["default_spool_path", "runtime_identity"])
def test_a_build_that_must_send_telemetry_stops_when_its_emitter_cannot_be_prepared(
    monkeypatch, capsys, failure
):
    def fail():
        raise OSError("private setup detail")

    starts = _record_starts(monkeypatch)
    monkeypatch.setattr(adapter, failure, fail)
    monkeypatch.setenv(REQUIRED_VARIABLE, "1")

    with pytest.raises(adapter.TelemetryRequiredError) as raised:
        START(
            adapter.LocalTelemetryEmitter,
            run_id="failed",
            country_code="uk",
            pipeline="full",
        )

    assert starts == []
    assert raised.value.reason == "setup_failure"
    assert "OSError" in str(raised.value)
    # Only the error's type is shown, as in the warning without the setting.
    assert "private setup detail" not in "".join(
        traceback.format_exception(raised.value)
    )
    assert "continue" not in capsys.readouterr().err


def test_a_build_that_must_send_telemetry_starts_an_emitter_with_telemetry(
    monkeypatch, tmp_path
):
    client = FakeEmitterClient()
    starts = _record_starts(monkeypatch, client)
    _install_emitter_command(
        monkeypatch,
        tmp_path / "bin",
        _version_answer(adapter.MINIMUM_EMITTER_VERSION),
    )
    monkeypatch.setenv(REQUIRED_VARIABLE, "1")

    emitter = _start_build_emitter(monkeypatch, tmp_path)

    assert len(starts) == 1
    assert emitter.client is client
    assert emitter.available
    emitter.transition_stage("solve")
    assert client.telemetry_messages[-1]["stage_id"] == "solve"


@pytest.mark.parametrize("setting", [None, "0"], ids=["unset", "0"])
@pytest.mark.parametrize("kind", list(UNUSABLE_EMITTERS))
def test_a_build_that_need_not_send_telemetry_continues_without_a_usable_emitter(
    monkeypatch, tmp_path, kind, setting
):
    client, _ = _arrange_unusable_emitter(monkeypatch, tmp_path, kind)
    if setting is None:
        monkeypatch.delenv(REQUIRED_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(REQUIRED_VARIABLE, setting)

    emitter = _start_with_a_short_limit(monkeypatch, tmp_path)

    # The build runs on, and no telemetry reaches an emitter.
    assert emitter.client is None or "telemetry" not in emitter.client.modules
    emitter.transition_stage("solve")
    emitter.complete()
    emitter.close()
    if isinstance(client, FakeEmitterClient):
        assert client.telemetry_messages == []


@pytest.mark.parametrize("setting", ["", "true", "yes", "2", " 1"])
def test_an_unrecognised_telemetry_requirement_stops_the_build(
    monkeypatch, tmp_path, setting
):
    # A mistyped value must not quietly turn the requirement off.
    starts = _record_starts(monkeypatch)
    monkeypatch.setenv(REQUIRED_VARIABLE, setting)

    with pytest.raises(ValueError, match="MICROCOSM_TELEMETRY_REQUIRED must be 1 or 0"):
        _start_build_emitter(monkeypatch, tmp_path)

    assert starts == []


@pytest.mark.parametrize("operation", ["inventory", "publish"])
def test_required_telemetry_leaves_short_lived_graph_emitters_optional(
    monkeypatch, tmp_path, operation
):
    # The setting concerns the build's own emitter, which records its
    # telemetry run; an Orrery-only emitter without one still answers.
    monkeypatch.setattr(interface_client, "start_emitter", INTERFACE_START_EMITTER)
    monkeypatch.setenv(REQUIRED_VARIABLE, "1")
    inventory = {"version": 1, "publication_id": "independent", "files": []}

    if operation == "inventory":
        assert adapter.publication_inventory(tmp_path, {}) is None
    else:
        receipt = adapter.publish_graph(tmp_path, inventory, wait_seconds=0)
        assert receipt["status"] == "failed"


def test_the_shell_s_telemetry_settings_never_reach_a_test():
    # A shell that exports them would change what every adapter test sees;
    # the shared fixture clears them, as it clears Hugging Face credentials.
    for name in (
        REQUIRED_VARIABLE,
        "MICROCOSM_COLLECTOR_TOKEN",
        "MICROCOSM_COLLECTOR_TOKEN_EXPIRES_AT",
        "MICROCOSM_COLLECTOR_ORIGIN",
    ):
        assert name not in os.environ
