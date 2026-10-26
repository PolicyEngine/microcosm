"""Test Microcosm's adapter with emitter APIs replaced, not hosted services."""

from types import SimpleNamespace

import pytest

from microcosm.build import telemetry_emitter as adapter

START = adapter.LocalTelemetryEmitter.start.__func__


def test_selects_both_workers_once_with_shared_path_and_build_identity(
    monkeypatch, tmp_path
):
    calls, messages, queries, closed = [], [], [], []
    transport = SimpleNamespace(send=messages.append)
    inventory = {
        "version": 1,
        "publication_id": "independent",
        "files": [],
    }

    def request(message, **options):
        queries.append((message, options))
        if message["command"] == "inventory":
            return {"inventory": inventory}
        return {
            "receipt": {
                "version": 1,
                "publication_id": "independent",
                "status": "pending",
                "error_code": None,
                "attempts": 1,
            }
        }

    graph_transport = SimpleNamespace(request=request)

    def for_module(name):
        return {"telemetry": transport, "orrery": graph_transport}[name]

    handle = SimpleNamespace(
        client=SimpleNamespace(for_module=for_module), close=lambda: closed.append(True)
    )

    def start(modules, **options):
        calls.append((modules, options))
        return handle

    monkeypatch.setattr(adapter, "start_services", start)
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
    assert len(calls) == 1
    modules, options = calls[0]
    assert set(modules) == {"telemetry", "orrery"}
    assert all(m["spool_path"] == str(path) for m in modules.values())
    assert "registration" not in modules["orrery"]
    assert modules["telemetry"]["registration"]["run_id"] == "run-one"
    assert modules["telemetry"]["heartbeat_seconds"] == 123
    assert modules["telemetry"]["identity"] == {"commit": "synthetic"}
    assert options["startup_timeout"] == 15
    # The service emits one run-start event from its startup configuration.
    # The build adapter must not send a second run-start request.
    assert messages == []
    emitter.stage("stage-that-is-not-in-a-fixed-list")
    assert messages[0]["stage_id"] == "stage-that-is-not-in-a-fixed-list"
    assert emitter.publication_inventory(tmp_path, {}) == inventory
    receipt = emitter.publish_graph(tmp_path, inventory, wait_seconds=0)
    assert receipt["status"] == "pending"
    assert [query[0]["command"] for query in queries] == ["inventory", "enqueue"]
    assert not path.exists()
    emitter.complete()
    emitter.close()
    assert closed == [True]
    assert messages[-1]["command"] == "complete"


def test_disabled_startup_cannot_claim_persistence_or_start_a_replacement_service(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        adapter.GraphPublicationClient,
        "start",
        lambda **options: pytest.fail("started a replacement graph service"),
    )
    emitter = adapter.LocalTelemetryEmitter(
        run=adapter.TelemetryRun("run-one", "uk", "full"),
        transport=None,
    )
    value = {"version": 1, "publication_id": "independent", "files": []}
    assert emitter.publication_inventory(tmp_path, {}) is None
    receipt = emitter.publish_graph(tmp_path, value, wait_seconds=0)
    assert receipt == {
        "version": 1,
        "publication_id": "independent",
        "status": "failed",
        "error_code": "local_service_unavailable",
    }
    assert list(tmp_path.iterdir()) == []


def test_unavailable_graph_socket_returns_failure_without_writing_a_queue(tmp_path):
    from microcosm_emitter.host.errors import LocalServiceError

    def unavailable(*args, **options):
        raise LocalServiceError("socket unavailable")

    emitter = adapter.LocalTelemetryEmitter(
        run=adapter.TelemetryRun("run-one", "uk", "full"),
        transport=SimpleNamespace(send=lambda message: None),
        graph_client=adapter.GraphPublicationClient(
            SimpleNamespace(request=unavailable)
        ),
    )
    assert emitter.publication_inventory(tmp_path, {}) is None
    receipt = emitter.publish_graph(
        tmp_path,
        {"version": 1, "publication_id": "independent", "files": []},
        wait_seconds=0,
    )
    assert receipt["status"] == "failed"
    assert receipt["error_code"] == "local_service_unavailable"
    assert list(tmp_path.iterdir()) == []


def test_start_failure_is_nonfatal_and_does_not_log_exception_payload(
    monkeypatch, capsys, tmp_path
):
    def fail(*args, **kwargs):
        raise RuntimeError("private credential should not be logged")

    monkeypatch.setattr(adapter, "start_services", fail)
    emitter = START(
        adapter.LocalTelemetryEmitter,
        run_id="run-one",
        country_code="uk",
        pipeline="full",
        spool_path=tmp_path / "jobs.sqlite3",
    )
    assert not emitter.available
    warning = capsys.readouterr().err
    assert "RuntimeError" in warning
    assert "private credential" not in warning


@pytest.mark.parametrize("operation", ["complete", "close"])
def test_default_test_emitter_uses_emitter_lifecycle_in_memory(
    fake_telemetry_emitters, operation
):
    emitter = adapter.LocalTelemetryEmitter.start(
        run_id="run-one", country_code="uk", pipeline="full"
    )
    emitter.stage("one")
    getattr(emitter, operation)()
    assert not emitter.available
    assert emitter.events[0]["stage_id"] == "one"


def test_fixture_intercepts_preimported_build_entrypoint(
    monkeypatch, fake_telemetry_emitters
):
    from microcosm.build.uk_runtime import rowwise_staging
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter

    def fail(*args, **kwargs):
        pytest.fail("test started a real emitter")

    monkeypatch.setattr(adapter, "start_services", fail)
    for start in (
        rowwise_staging.start_local_telemetry_emitter_service,
        adapter.LocalTelemetryEmitter.start,
    ):
        emitter = start(run_id="fake", country_code="GB", pipeline="test")
        assert isinstance(emitter, FakeTelemetryEmitter)
        assert emitter in fake_telemetry_emitters


def test_fixture_isolates_cached_huggingface_credentials(tmp_path):
    from pathlib import Path

    from huggingface_hub import constants, get_token

    assert Path(constants.HF_TOKEN_PATH).is_relative_to(tmp_path)
    assert get_token() is None


@pytest.mark.parametrize("failure", ["default_spool_path", "runtime_identity"])
def test_metadata_or_cache_failure_never_starts_a_service(monkeypatch, failure):
    def fail():
        raise OSError("private setup detail")

    monkeypatch.setattr(adapter, failure, fail)
    monkeypatch.setattr(
        adapter, "start_services", lambda *a, **kw: pytest.fail("started after failure")
    )
    emitter = START(
        adapter.LocalTelemetryEmitter,
        run_id="failed",
        country_code="uk",
        pipeline="full",
    )
    assert not emitter.available


def test_graph_receipt_poll_uses_the_existing_module_client(monkeypatch, tmp_path):
    from microcosm_emitter.orrery import client

    calls = []

    def request(message, **options):
        calls.append(message["command"])
        return {
            "receipt": {
                "version": 1,
                "publication_id": "independent",
                "status": (
                    "pending" if message["command"] == "enqueue" else "published"
                ),
                "error_code": None,
                "attempts": 0 if message["command"] == "enqueue" else 1,
            }
        }

    monkeypatch.setattr(client.time, "sleep", lambda duration: None)
    emitter = adapter.LocalTelemetryEmitter(
        run=adapter.TelemetryRun("run-one", "uk", "full"),
        transport=SimpleNamespace(send=lambda message: None),
        graph_client=adapter.GraphPublicationClient(SimpleNamespace(request=request)),
    )
    receipt = emitter.publish_graph(
        tmp_path,
        {"version": 1, "publication_id": "independent", "files": []},
        wait_seconds=1,
    )
    assert receipt["status"] == "published"
    assert calls == ["enqueue", "receipt"]


def test_adapter_assigns_host_cleanup_to_the_build_finally(monkeypatch):
    configured = []

    def initialize(self, **options):
        configured.append(options)

    monkeypatch.setattr(adapter.EmitterTelemetryClient, "__init__", initialize)
    adapter.LocalTelemetryEmitter(
        run=adapter.TelemetryRun("synthetic", "uk", "full"),
        transport=None,
        close_service_on_finish=False,
    )
    assert configured[0]["close_service_on_finish"] is False


@pytest.mark.parametrize("outcome", ["blocked", "failed", "completed"])
def test_graph_publication_precedes_host_cleanup_after_recording_an_outcome(
    monkeypatch, tmp_path, outcome
):
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter

    operations, queries = [], []
    telemetry = FakeTelemetryEmitter(adapter.TelemetryRun("synthetic", "uk", "full"))
    inventory = {"version": 1, "publication_id": "synthetic", "files": []}

    def request(message, **options):
        operations.append(message["command"])
        queries.append(message)
        if message["command"] == "inventory":
            return {"inventory": inventory}
        return {
            "receipt": {
                "version": 1,
                "publication_id": "synthetic",
                "status": "pending",
                "attempts": 1,
                "error_code": None,
            }
        }

    def for_module(name):
        return {
            "telemetry": telemetry,
            "orrery": SimpleNamespace(request=request),
        }[name]

    handle = SimpleNamespace(
        client=SimpleNamespace(for_module=for_module),
        close=lambda: operations.append("close"),
    )
    monkeypatch.setattr(adapter, "start_services", lambda *a, **k: handle)
    monkeypatch.setattr(adapter, "runtime_identity", lambda: {"commit": "synthetic"})
    emitter = START(
        adapter.LocalTelemetryEmitter,
        run_id="synthetic",
        country_code="uk",
        pipeline="full",
        spool_path=tmp_path / "queue.sqlite3",
        close_service_on_finish=False,
    )
    emitter.transition_stage("solve")
    if outcome == "blocked":
        emitter.block(
            phase="post_calibration",
            blocking_gate_ids=["uk.local.fit"],
            blocking_failure_count=3,
            gate_statuses={"uk.local.fit": "FAIL"},
        )
    elif outcome == "failed":
        emitter.fail(
            ValueError("synthetic validation refusal"),
            error_code="BUILD_REFUSED",
            failure_class="refused",
            failed_during="solve",
        )
    else:
        emitter.complete()
    assert operations == []
    assert emitter.available
    final_events = [event for event in telemetry.events if event["event_type"] == "run"]
    assert [event["status"] for event in final_events] == [outcome]
    if outcome == "blocked":
        assert final_events[0]["details"] == {
            "phase": "post_calibration",
            "blocking_gate_ids": ["uk.local.fit"],
            "blocking_failure_count": 3,
            "gate_statuses": {"uk.local.fit": "FAIL"},
        }
        assert telemetry.events[-2]["event_type"] == "stage"
        assert telemetry.events[-2]["status"] == "completed"
    elif outcome == "failed":
        assert final_events[0]["details"]["error_code"] == "BUILD_REFUSED"
        assert final_events[0]["details"]["failure_class"] == "refused"
        assert final_events[0]["details"]["failed_during"] == "solve"
    # A completed telemetry lifecycle must not prevent requests on the other
    # module; the existing build finally owns physical service shutdown.
    try:
        assert emitter.publication_inventory(tmp_path, {}) == inventory
        assert emitter.publish_graph(tmp_path, inventory, wait_seconds=0)["status"] == (
            "pending"
        )
    finally:
        emitter.close()
    emitter.close()
    assert [query["command"] for query in queries] == ["inventory", "enqueue"]
    assert operations == ["inventory", "enqueue", "close"]
    assert not emitter.available
    assert (
        len([event for event in telemetry.events if event["event_type"] == "run"]) == 1
    )


@pytest.mark.parametrize("outcome", ["complete", "fail"])
def test_default_finish_releases_host_without_an_outer_build_cleanup(outcome):
    closed = []
    emitter = adapter.LocalTelemetryEmitter(
        run=adapter.TelemetryRun("synthetic", "us", "release"),
        transport=SimpleNamespace(send=lambda message: None),
        handle=SimpleNamespace(close=lambda: closed.append(True)),
    )
    if outcome == "complete":
        emitter.complete()
    else:
        emitter.fail(ValueError("synthetic failure"), error_code="BUILD_FAILED")
    assert closed == [True]
    assert not emitter.available
    emitter.close()
    assert closed == [True]


@pytest.mark.parametrize("outcome", ["block", "fail"])
def test_outcome_on_disabled_transport_is_safe_and_cannot_restart_service(
    monkeypatch, outcome
):
    monkeypatch.setattr(
        adapter,
        "start_services",
        lambda *a, **k: pytest.fail("outcome started a replacement service"),
    )
    emitter = adapter.LocalTelemetryEmitter(
        run=adapter.TelemetryRun("synthetic", "uk", "full"), transport=None
    )
    if outcome == "block":
        emitter.block(phase="preflight", blocking_gate_ids=[], blocking_failure_count=1)
    else:
        emitter.fail(ValueError("refused"), error_code="BUILD_REFUSED")
    assert not emitter.available
    emitter.close()
