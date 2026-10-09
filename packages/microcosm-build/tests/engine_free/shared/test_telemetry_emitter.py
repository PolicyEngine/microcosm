import json
import os
import socket
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from alembic import command
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
from sqlalchemy.orm import Session

from microcosm.build.telemetry_emitter import (
    LocalTelemetryEmitter,
    TelemetryRun,
)
from microcosm.build.telemetry_emitter_service import (
    CollectorDelivery,
    EmitterService,
    EventSpool,
)
from microcosm.build.telemetry_emitter_service import collector as collector_module
from microcosm.build.telemetry_emitter_service import resources as resources_module
from microcosm.build.telemetry_emitter_service import runtime as runtime_module
from microcosm.build.telemetry_emitter_service import spool as spool_module
from microcosm.build.telemetry_emitter_service.constants import (
    DRAIN_RETRY_SECONDS,
    PRODUCTION_COLLECTOR_URL,
)
from microcosm.build.telemetry_emitter_service.database import (
    create_spool_engine,
    sqlite_database_url,
)
from microcosm.build.telemetry_emitter_service.migrations import (
    alembic_config,
    current_database_revision,
    migration_head_revision,
)
from microcosm.build.telemetry_emitter_service.models import (
    SpoolModel,
    TelemetryEventRecord,
    TelemetryRunRecord,
    serialized_json_length,
)
from microcosm.build.telemetry_protocol import (
    BUILD_COMPLETED_MESSAGE,
    BUILD_STARTED_MESSAGE,
    MAX_TELEMETRY_DETAILS_BYTES,
)
from microcosm.build.telemetry_sanitization import (
    sanitize_details,
    sanitize_json,
    sanitize_text,
)


def _registration(run_id: str = "run-a") -> dict[str, object]:
    return TelemetryRun(
        run_id=run_id,
        country_code="US",
        pipeline="us_fiscal_refresh",
        candidate_id="candidate-a",
        producer_id="producer-a",
    ).as_registration()


def _event(stage_id: str = "compile_targets") -> dict[str, object]:
    return {
        "timestamp": "2026-10-02T10:00:00+00:00",
        "event_type": "stage",
        "stage_id": stage_id,
        "status": "started",
        "message": "Compiling targets.",
        "details": {"batches": 12},
    }


def test_development_collector_must_be_on_loopback() -> None:
    assert collector_module._development_collector_url("http://127.0.0.1:8080") == (
        "http://127.0.0.1:8080"
    )
    for value in ("https://collector.example", "http://192.0.2.1:8080"):
        try:
            collector_module._development_collector_url(value)
        except ValueError as error:
            assert "loopback" in str(error)
        else:
            raise AssertionError("non-loopback development collector was accepted")


def test_production_collector_cannot_be_replaced_by_environment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "MICROCOSM_TELEMETRY_COLLECTOR_URL",
        "https://untrusted.example",
    )

    delivery = CollectorDelivery(EventSpool(tmp_path / "events.sqlite3"))

    assert delivery.collector_url == PRODUCTION_COLLECTOR_URL


def test_token_bearing_http_post_does_not_follow_redirects() -> None:
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            paths.append(self.path)
            self.send_response(307)
            self.send_header("Location", "/credential-leak")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    try:
        status, _ = collector_module._http_post(
            f"http://127.0.0.1:{server.server_port}/exchange",
            {"run_id": "run-a"},
            "hf-private-token",
        )
    finally:
        server.shutdown()
        server_thread.join(timeout=2)
        server.server_close()

    assert status == 307
    assert paths == ["/exchange"]


def test_outbound_payload_redacts_credentials_and_tracebacks() -> None:
    assert sanitize_text("Bearer hf_abcdefghijk") == "[redacted]"
    assert sanitize_json(
        {
            "HF_TOKEN": "hf_abcdefghijk",
            "traceback": "private stack",
            "message": "credential=top-secret",
        }
    ) == {
        "HF_TOKEN": "[redacted]",
        "traceback": "[redacted]",
        "message": "[redacted]",
    }
    bounded = sanitize_details(
        {"done": 2, **{f"large_{index}": "x" * 2_000 for index in range(10)}}
    )
    assert bounded["done"] == 2
    assert bounded["telemetry_details_truncated"] is True
    assert len(json.dumps(bounded).encode()) <= MAX_TELEMETRY_DETAILS_BYTES


def test_event_spool_assigns_stable_sequences_and_acknowledges(tmp_path) -> None:
    spool_path = tmp_path / "events.sqlite3"
    spool = EventSpool(spool_path)
    registration = _registration()
    spool.register(registration)

    first = spool.append(registration, _event())
    second = spool.append(registration, _event("calibrate"))

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert first["producer_id"] == "producer-a"
    assert [event["sequence"] for event in spool.batch("run-a", "producer-a")] == [
        1,
        2,
    ]

    spool.acknowledge([first["event_id"]])
    assert [event["sequence"] for event in spool.batch("run-a", "producer-a")] == [2]
    assert current_database_revision(spool_path) == migration_head_revision()


def test_alembic_schema_matches_sqlalchemy_models(tmp_path) -> None:
    spool_path = tmp_path / "events.sqlite3"
    EventSpool(spool_path)
    engine = create_spool_engine(spool_path)
    try:
        with engine.begin() as connection:
            with alembic_config(connection=connection) as config:
                command.check(config)
    finally:
        engine.dispose()


def test_sequential_stage_updates_close_the_previous_stage() -> None:
    emitter = LocalTelemetryEmitter(
        run=TelemetryRun(
            run_id="run-a",
            country_code="US",
            pipeline="us_fiscal_refresh",
        ),
        process=None,
        socket_path=Path("/unused"),
        runtime_dir=None,
    )
    messages: list[dict[str, object]] = []
    emitter._send = messages.append

    emitter.transition_stage("load", message="Loading.")
    emitter.transition_stage("compile", message="Compiling.")
    emitter.transition_stage("compile", status="completed", batches=4)

    events = [message["event"] for message in messages]
    assert [(event["stage_id"], event["status"]) for event in events] == [
        ("load", "started"),
        ("load", "completed"),
        ("compile", "started"),
        ("compile", "completed"),
    ]


def test_event_spool_keeps_repeated_run_producers_separate(tmp_path) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    first = _registration()
    second = {**first, "producer_id": "producer-b"}
    spool.register(first)
    spool.register(second)

    spool.append(first, _event())
    spool.append(second, _event())

    assert spool.batch("run-a", "producer-a")[0]["sequence"] == 1
    assert spool.batch("run-a", "producer-b")[0]["sequence"] == 1
    assert len(spool.pending_runs()) == 2


def test_pre_eligibility_spool_is_not_uploaded_after_upgrade(tmp_path) -> None:
    path = tmp_path / "events.sqlite3"
    registration = _registration()
    metadata = MetaData()
    runs = Table(
        "telemetry_runs",
        metadata,
        Column("run_id", String, primary_key=True),
        Column("producer_id", String, primary_key=True),
        Column("registration_json", String, nullable=False),
        Column("next_sequence", Integer, nullable=False, default=1),
        Column("updated_at", String, nullable=False),
    )
    events = Table(
        "telemetry_events",
        metadata,
        Column("event_id", String, primary_key=True),
        Column("run_id", String, nullable=False),
        Column("producer_id", String, nullable=False),
        Column("sequence", Integer, nullable=False),
        Column("payload_json", String, nullable=False),
        Column("created_at", String, nullable=False),
    )
    engine = create_engine(sqlite_database_url(path))
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            runs.insert().values(
                run_id="run-a",
                producer_id="producer-a",
                registration_json=json.dumps(registration),
                next_sequence=2,
                updated_at="2026-10-02T10:00:00+00:00",
            )
        )
        connection.execute(
            events.insert().values(
                event_id="old-event",
                run_id="run-a",
                producer_id="producer-a",
                sequence=1,
                payload_json=json.dumps({"event_id": "old-event"}),
                created_at="2026-10-02T10:00:00+00:00",
            )
        )
    engine.dispose()

    spool = EventSpool(path)

    assert spool.has_pending()
    assert spool.pending_runs() == []
    assert current_database_revision(path) == migration_head_revision()


def test_current_pre_alembic_spool_is_adopted_without_losing_events(tmp_path) -> None:
    path = tmp_path / "events.sqlite3"
    registration = _registration()
    engine = create_engine(sqlite_database_url(path))
    SpoolModel.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        session.add(
            TelemetryRunRecord(
                run_id="run-a",
                producer_id="producer-a",
                registration=registration,
                next_sequence=2,
                upload_state="pending",
                local_only_reason=None,
                updated_at="2026-10-02T10:00:00+00:00",
            )
        )
        session.add(
            TelemetryEventRecord(
                event_id="existing-event",
                run_id="run-a",
                producer_id="producer-a",
                sequence=1,
                payload={"event_id": "existing-event", "sequence": 1},
                created_at="2026-10-02T10:00:00+00:00",
            )
        )
    engine.dispose()

    spool = EventSpool(path)

    assert spool.pending_runs() == [registration]
    assert spool.batch("run-a", "producer-a") == [
        {"event_id": "existing-event", "sequence": 1}
    ]
    assert current_database_revision(path) == migration_head_revision()


def test_event_spool_prunes_oldest_events_to_size_limit(
    tmp_path,
    monkeypatch,
) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    registration = _registration()
    spool.register(registration)
    first = spool.append(registration, _event("first"))
    second = spool.append(registration, _event("second"))
    monkeypatch.setattr(
        spool_module,
        "MAX_QUEUED_BYTES",
        serialized_json_length(second),
    )

    spool.prune()

    assert first["event_id"] != second["event_id"]
    assert spool.batch("run-a", "producer-a") == [second]


def test_collector_delivery_exchanges_hf_token_then_flushes(
    tmp_path, monkeypatch
) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    registration = _registration()
    spool.register(registration)
    queued = spool.append(registration, _event())
    requests: list[tuple[str, dict[str, object], str]] = []

    monkeypatch.setattr(collector_module, "_huggingface_token", lambda: "hf-secret")

    def fake_post(url, payload, token, *, timeout=5.0):
        requests.append((url, payload, token))
        if url.endswith("/v1/auth/huggingface/exchange"):
            return 200, {"access_token": "collector-token", "expires_in": 3600}
        if url.endswith("/v1/runs"):
            return 201, {"registered": True}
        return 202, {"accepted": 1, "duplicates": 0}

    monkeypatch.setattr(collector_module, "_http_post", fake_post)

    delivery = CollectorDelivery(
        spool,
        development_collector_url="http://127.0.0.1:8080",
    )
    assert delivery.flush_once()
    assert requests[0][2] == "hf-secret"
    assert requests[0][1] == {}
    assert requests[1] == (
        "http://127.0.0.1:8080/v1/runs",
        registration,
        "collector-token",
    )
    assert requests[2][2] == "collector-token"
    assert requests[2][1] == {"events": [queued]}
    assert not spool.has_pending()


def test_non_org_credential_keeps_event_local(tmp_path, monkeypatch, capsys) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    registration = _registration()
    spool.register(registration)
    spool.append(registration, _event())
    requests: list[dict[str, object]] = []
    monkeypatch.setattr(collector_module, "_huggingface_token", lambda: "hf-outsider")

    def reject(url, payload, token, *, timeout=5.0):
        requests.append(payload)
        return 403, {"detail": "not a member"}

    monkeypatch.setattr(collector_module, "_http_post", reject)
    delivery = CollectorDelivery(
        spool,
        development_collector_url="http://127.0.0.1:8080",
    )

    assert not delivery.flush_once()
    assert not delivery.flush_once()
    assert spool.has_pending()
    assert spool.pending_runs() == []
    assert requests == [{}]
    assert capsys.readouterr().err.count("local-only for this run") == 1


def test_identity_provider_outage_keeps_events_eligible_for_retry(
    tmp_path, monkeypatch
) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    registration = _registration()
    spool.register(registration)
    spool.append(registration, _event())
    monkeypatch.setattr(collector_module, "_huggingface_token", lambda: "hf-member")
    monkeypatch.setattr(
        collector_module,
        "_http_post",
        lambda *args, **kwargs: (503, {"detail": "temporarily unavailable"}),
    )

    assert not CollectorDelivery(spool).flush_once()
    assert spool.pending_runs() == [registration]


def test_missing_credential_never_contacts_collector_and_stays_local_only(
    tmp_path, monkeypatch
) -> None:
    spool_path = tmp_path / "events.sqlite3"
    spool = EventSpool(spool_path)
    registration = _registration()
    spool.register(registration)
    spool.append(registration, _event())
    monkeypatch.setattr(collector_module, "_huggingface_token", lambda: None)
    monkeypatch.setattr(
        collector_module,
        "_http_post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network request")
        ),
    )

    delivery = CollectorDelivery(spool)
    assert not delivery.flush_once()
    assert spool.pending_runs() == []

    reopened = EventSpool(spool_path)
    monkeypatch.setattr(collector_module, "_huggingface_token", lambda: "hf-later")
    assert reopened.pending_runs() == []


def test_resource_sampler_has_a_base_install_fallback(monkeypatch) -> None:
    monkeypatch.setattr(resources_module, "psutil", None)
    sampler = resources_module.ProcessTreeSampler(os.getpid())

    assert sampler.parent_alive()
    sample = sampler.sample()
    assert sample.keys() == {
        "cpu_user_seconds",
        "cpu_system_seconds",
        "rss_bytes",
        "peak_rss_bytes",
    }
    assert all(value >= 0 for value in sample.values())


def test_resource_sampler_keeps_reaped_child_cpu_monotonic(monkeypatch) -> None:
    state = {"child_alive": True}

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return float(self.pid)

        def children(self, recursive=False):
            assert recursive is True
            if self.pid == 10 and state["child_alive"]:
                return [FakeProcess(11)]
            return []

        def cpu_times(self):
            if self.pid == 10:
                return SimpleNamespace(
                    user=10.0,
                    system=2.0,
                    children_user=100.0 if not state["child_alive"] else 0.0,
                    children_system=20.0 if not state["child_alive"] else 0.0,
                )
            return SimpleNamespace(
                user=80.0,
                system=15.0,
                children_user=0.0,
                children_system=0.0,
            )

        def memory_info(self):
            return SimpleNamespace(rss=100)

    monkeypatch.setattr(
        resources_module,
        "psutil",
        SimpleNamespace(Process=FakeProcess, Error=OSError, STATUS_ZOMBIE="zombie"),
    )
    sampler = resources_module.ProcessTreeSampler(10)

    before = sampler.sample()
    state["child_alive"] = False
    after = sampler.sample()

    assert before["cpu_user_seconds"] == 90.0
    assert before["cpu_system_seconds"] == 17.0
    assert after["cpu_user_seconds"] == 110.0
    assert after["cpu_system_seconds"] == 22.0


def test_emitter_startup_timeout_terminates_and_reaps_service(
    tmp_path, monkeypatch, real_local_telemetry
) -> None:
    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

    process = FakeProcess()
    monkeypatch.setattr(
        "microcosm.build.telemetry_emitter.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    emitter = LocalTelemetryEmitter.start(
        run_id="startup-timeout",
        country_code="US",
        pipeline="test-pipeline",
        development_collector_url="http://127.0.0.1:8080",
        spool_path=tmp_path / "events.sqlite3",
        startup_timeout_seconds=0,
    )

    assert not emitter.available
    assert process.terminated
    assert process.waited


class _FakeSampler:
    def sample(self):
        return {
            "cpu_user_seconds": 1.0,
            "cpu_system_seconds": 0.5,
            "rss_bytes": 100,
            "peak_rss_bytes": 120,
        }

    def parent_alive(self):
        return True


class _FakeDelivery:
    def flush_once(self):
        return False


def test_local_socket_acknowledges_after_durable_queue(
    tmp_path, real_local_telemetry
) -> None:
    socket_path = (
        Path(tempfile.mkdtemp(prefix="microcosm-test-", dir="/tmp")) / "e.sock"
    )
    spool = EventSpool(tmp_path / "events.sqlite3")
    registration = _registration()
    emitter = EmitterService(
        socket_path=socket_path,
        registration=registration,
        spool=spool,
        delivery=_FakeDelivery(),
        sampler=_FakeSampler(),
        heartbeat_seconds=60,
        drain_seconds=0,
    )
    thread = threading.Thread(target=emitter.run)
    thread.start()
    deadline = time.monotonic() + 2
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(
            json.dumps({"action": "event", "event": _event()}).encode() + b"\n"
        )
        assert client.recv(16) == b"ok\n"

    assert spool.batch("run-a", "producer-a")[0]["resources"]["rss_bytes"] == 100

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(b'{"action":"close"}\n')
        assert client.recv(16) == b"ok\n"
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_subprocess_exchanges_ambient_token_and_delivers_events(
    tmp_path, monkeypatch, real_local_telemetry
) -> None:
    requests: list[tuple[str, str, dict[str, object]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            requests.append((self.path, self.headers["Authorization"], payload))
            if self.path.endswith("/v1/auth/huggingface/exchange"):
                response = {"access_token": "collector-token", "expires_in": 3600}
                status = 200
            elif self.path == "/v1/runs":
                response = {"registered": True}
                status = 201
            else:
                response = {
                    "accepted": len(payload["events"]),
                    "duplicates": 0,
                }
                status = 202
            body = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    monkeypatch.setenv("HF_TOKEN", "hf-ambient-test-token")
    emitter = LocalTelemetryEmitter.start(
        run_id="subprocess-run",
        country_code="US",
        pipeline="test-pipeline",
        development_collector_url=f"http://127.0.0.1:{server.server_port}",
        spool_path=tmp_path / "subprocess.sqlite3",
        heartbeat_seconds=60,
    )
    assert emitter.available
    emitter.stage("compile", message="Started.")
    emitter.complete()
    assert emitter._process is not None
    emitter._process.wait(timeout=10)
    server.shutdown()
    server_thread.join(timeout=2)
    server.server_close()

    assert requests[0][0] == "/v1/auth/huggingface/exchange"
    assert requests[0][1] == "Bearer hf-ambient-test-token"
    assert requests[0][2] == {}
    assert requests[1][0] == "/v1/runs"
    assert requests[1][1] == "Bearer collector-token"
    ingestion = [request for request in requests if request[0].endswith("/events")]
    assert ingestion
    events = [
        event
        for _, authorization, payload in ingestion
        for event in payload["events"]
        if authorization == "Bearer collector-token"
    ]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    identity = events[0]["details"]["identity"]
    assert identity["host"]["cpu_count"] is not None
    assert "runtime" in identity
    assert events[0]["message"] == BUILD_STARTED_MESSAGE
    assert events[-1]["message"] == BUILD_COMPLETED_MESSAGE
    assert events[-1]["status"] == "completed"


def test_failure_marks_active_stage_failed_before_run_failure():
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter

    emitter = FakeTelemetryEmitter(TelemetryRun("failed-build", "GB", "test"))
    emitter.transition_stage("calibration")
    emitter.fail(ValueError("invalid calibration"))

    assert [(event["event_type"], event["status"]) for event in emitter.events] == [
        ("stage", "started"),
        ("stage", "failed"),
        ("run", "failed"),
    ]
    assert emitter.events[1]["message"] == "invalid calibration"
    assert emitter.events[1]["details"]["error_type"] == "ValueError"
    assert emitter.events[2]["details"]["failed_during"] == "calibration"
    assert not emitter.available


@pytest.mark.parametrize("failure_point", ["mkdtemp", "chmod", "cache_dir"])
def test_early_startup_failure_returns_disabled_handle(
    tmp_path, monkeypatch, capsys, real_local_telemetry, failure_point
):
    from microcosm.build import telemetry_emitter as module

    runtime_dir = tmp_path / "runtime"

    def make_runtime(**kwargs):
        runtime_dir.mkdir()
        return str(runtime_dir)

    def fail(*args, **kwargs):
        raise OSError("telemetry setup refused")

    monkeypatch.setattr(module.tempfile, "mkdtemp", make_runtime)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("spawned after failed setup"),
    )
    if failure_point == "mkdtemp":
        monkeypatch.setattr(module.tempfile, "mkdtemp", fail)
    elif failure_point == "chmod":
        monkeypatch.setattr(Path, "chmod", fail)
    else:
        monkeypatch.setattr(module, "_cache_dir", fail)

    emitter = LocalTelemetryEmitter.start(
        run_id="failed-start",
        country_code="GB",
        pipeline="test",
        development_collector_url="http://127.0.0.1:1",
    )

    assert not emitter.available
    assert not runtime_dir.exists()
    assert "warning" in capsys.readouterr().err.lower()
    emitter.close()


def test_default_fixture_intercepts_preimported_entrypoint_and_class(
    monkeypatch, fake_telemetry_emitters
):
    from microcosm.build import telemetry_emitter as module
    from microcosm.build.uk_runtime import rowwise_staging
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter

    monkeypatch.setenv("HF_TOKEN", "hf-ambient-must-not-be-used")
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("test spawned real emitter"),
    )
    for start in (
        rowwise_staging.start_local_telemetry_emitter_service,
        LocalTelemetryEmitter.start,
    ):
        emitter = start(run_id="fake", country_code="GB", pipeline="test")
        assert isinstance(emitter, FakeTelemetryEmitter)
        assert emitter in fake_telemetry_emitters


def test_default_fixture_isolates_cached_huggingface_credentials(tmp_path):
    from huggingface_hub import constants, get_token

    assert Path(constants.HF_TOKEN_PATH).is_relative_to(tmp_path)
    assert get_token() is None


@pytest.mark.parametrize("collector_url", [None, "https://collector.example"])
def test_real_service_opt_in_requires_explicit_loopback(
    real_local_telemetry, collector_url
):
    with pytest.raises(ValueError, match="loopback"):
        LocalTelemetryEmitter.start(
            run_id="isolated",
            country_code="GB",
            pipeline="test",
            development_collector_url=collector_url,
        )


def test_close_does_not_report_success():
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter

    emitter = FakeTelemetryEmitter(TelemetryRun("cleanup", "GB", "test"))
    emitter.transition_stage("calibration")
    emitter.close()
    assert [(event["event_type"], event["status"]) for event in emitter.events] == [
        ("stage", "started")
    ]


def test_failure_without_active_stage_only_reports_run_failure():
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter

    emitter = FakeTelemetryEmitter(TelemetryRun("failed", "GB", "test"))
    emitter.fail(ValueError("before first stage"), failed_during="preflight")
    assert [(event["event_type"], event["status"]) for event in emitter.events] == [
        ("run", "failed")
    ]
    assert emitter.events[0]["details"]["failed_during"] == "preflight"


@pytest.mark.parametrize("drain_seconds", [0.1, 1.0])
def test_shutdown_retries_wait_after_stop_without_exceeding_deadline(
    tmp_path, monkeypatch, drain_seconds
) -> None:
    clock = SimpleNamespace(now=0.0)
    attempts: list[float] = []

    def flush_once() -> bool:
        attempts.append(clock.now)
        # Simulate a small amount of delivery work, so a broken busy loop
        # still reaches its deadline without relying on real elapsed time.
        clock.now += 0.01
        return False

    def sleep(seconds: float) -> None:
        clock.now += seconds

    monkeypatch.setattr(
        runtime_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep),
    )
    service = EmitterService(
        socket_path=tmp_path / "unused.sock",
        registration=_registration(),
        spool=SimpleNamespace(has_deliverable=lambda: True),
        delivery=SimpleNamespace(flush_once=flush_once),
        sampler=_FakeSampler(),
        heartbeat_seconds=60,
        drain_seconds=drain_seconds,
    )
    service._stop.set()

    service._drain()

    assert attempts
    assert all(
        later - earlier >= DRAIN_RETRY_SECONDS
        for earlier, later in zip(attempts, attempts[1:], strict=False)
    )
    assert clock.now == pytest.approx(drain_seconds)


def test_fallback_sampler_retains_reaped_child_cpu(tmp_path, monkeypatch) -> None:
    proc_root = tmp_path / "proc"

    def write_stat(
        pid: int,
        parent_pid: int,
        user_ticks: int,
        system_ticks: int,
        child_user_ticks: int = 0,
        child_system_ticks: int = 0,
    ) -> Path:
        # Fields start at the state after the parenthesized process name.
        fields = ["0"] * 22
        fields[0] = "S"
        fields[1] = str(parent_pid)
        fields[11:15] = map(
            str, (user_ticks, system_ticks, child_user_ticks, child_system_ticks)
        )
        fields[19] = "1000"
        fields[21] = "4"
        path = proc_root / str(pid) / "stat"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{pid} (build worker) {' '.join(fields)}")
        return path

    write_stat(100, 1, 100, 200)
    child_path = write_stat(101, 100, 500, 300)
    monkeypatch.setattr(resources_module, "psutil", None)
    monkeypatch.setattr(
        resources_module,
        "Path",
        lambda path: proc_root / Path(path).relative_to("/proc"),
    )
    monkeypatch.setattr(
        resources_module,
        "os",
        SimpleNamespace(
            sysconf=lambda name: {"SC_CLK_TCK": 100, "SC_PAGE_SIZE": 4096}[name]
        ),
    )
    sampler = resources_module.ProcessTreeSampler(100)

    before = sampler.sample()
    child_path.unlink()
    write_stat(100, 1, 100, 200, child_user_ticks=500, child_system_ticks=300)
    after = sampler.sample()

    assert before["cpu_user_seconds"] == after["cpu_user_seconds"] == 6.0
    assert before["cpu_system_seconds"] == after["cpu_system_seconds"] == 5.0
    assert before["rss_bytes"] == 8 * 4096
    assert after["rss_bytes"] == 4 * 4096
    assert after["peak_rss_bytes"] == before["rss_bytes"]
