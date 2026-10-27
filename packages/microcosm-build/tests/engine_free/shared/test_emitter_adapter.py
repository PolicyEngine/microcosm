"""Microcosm's adapter over ``microcosm-emitter-interface``, without an emitter.

The adapter starts the separately installed ``microcosm-emitter`` command
through the interface's ``EmitterClient``; these tests replace that client
with an in-memory one, or let the interface fail to find the command. A
stand-in command answers ``microcosm-emitter version`` where a test needs an
installed release; no test runs a real emitter.
"""

import json

import pytest
from microcosm_emitter_interface import (
    EmitterClient,
    HuggingFaceSettings,
    OrrerySettings,
    TelemetrySettings,
)
from microcosm_emitter_interface import client as interface_client
from microcosm_emitter_interface import launcher as interface_launcher

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
