import json
import os
import socket
import sqlite3
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import microcosm.build.telemetry_emitter_service as service_module
from microcosm.build.telemetry_emitter import (
    LocalTelemetryEmitter,
    TelemetryRun,
    _safe_details,
    _safe_json,
    _safe_text,
)
from microcosm.build.telemetry_emitter_service import (
    CollectorDelivery,
    EmitterService,
    EventSpool,
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
    assert service_module._development_collector_url("http://127.0.0.1:8080") == (
        "http://127.0.0.1:8080"
    )
    for value in ("https://collector.example", "http://192.0.2.1:8080"):
        try:
            service_module._development_collector_url(value)
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

    assert delivery.collector_url == service_module.PRODUCTION_COLLECTOR_URL


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
        status, _ = service_module._http_post(
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
    assert _safe_text("Bearer hf_abcdefghijk") == "[redacted]"
    assert _safe_json(
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
    bounded = _safe_details(
        {"done": 2, **{f"large_{index}": "x" * 2_000 for index in range(10)}}
    )
    assert bounded["done"] == 2
    assert bounded["telemetry_details_truncated"] is True
    assert len(json.dumps(bounded).encode()) <= 8_192


def test_event_spool_assigns_stable_sequences_and_acknowledges(tmp_path) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
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
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE telemetry_runs (
            run_id TEXT NOT NULL,
            producer_id TEXT NOT NULL,
            registration_json TEXT NOT NULL,
            next_sequence INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(run_id, producer_id)
        );
        CREATE TABLE telemetry_events (
            event_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            producer_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, producer_id, sequence)
        );
        """
    )
    connection.execute(
        "INSERT INTO telemetry_runs VALUES (?, ?, ?, 2, ?)",
        (
            "run-a",
            "producer-a",
            json.dumps(registration),
            "2026-10-02T10:00:00+00:00",
        ),
    )
    connection.execute(
        "INSERT INTO telemetry_events VALUES (?, ?, ?, 1, ?, ?)",
        (
            "old-event",
            "run-a",
            "producer-a",
            json.dumps({"event_id": "old-event"}),
            "2026-10-02T10:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()

    spool = EventSpool(path)

    assert spool.has_pending()
    assert spool.pending_runs() == []


def test_collector_delivery_exchanges_hf_token_then_flushes(
    tmp_path, monkeypatch
) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    registration = _registration()
    spool.register(registration)
    queued = spool.append(registration, _event())
    requests: list[tuple[str, dict[str, object], str]] = []

    monkeypatch.setattr(service_module, "_huggingface_token", lambda: "hf-secret")

    def fake_post(url, payload, token, *, timeout=5.0):
        requests.append((url, payload, token))
        if url.endswith("/v1/auth/huggingface/exchange"):
            return 200, {"access_token": "collector-token", "expires_in": 3600}
        if url.endswith("/v1/runs"):
            return 201, {"registered": True}
        return 202, {"accepted": 1, "duplicates": 0}

    monkeypatch.setattr(service_module, "_http_post", fake_post)

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
    monkeypatch.setattr(service_module, "_huggingface_token", lambda: "hf-outsider")

    def reject(url, payload, token, *, timeout=5.0):
        requests.append(payload)
        return 403, {"detail": "not a member"}

    monkeypatch.setattr(service_module, "_http_post", reject)
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
    monkeypatch.setattr(service_module, "_huggingface_token", lambda: "hf-member")
    monkeypatch.setattr(
        service_module,
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
    monkeypatch.setattr(service_module, "_huggingface_token", lambda: None)
    monkeypatch.setattr(
        service_module,
        "_http_post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network request")
        ),
    )

    delivery = CollectorDelivery(spool)
    assert not delivery.flush_once()
    assert spool.pending_runs() == []

    reopened = EventSpool(spool_path)
    monkeypatch.setattr(service_module, "_huggingface_token", lambda: "hf-later")
    assert reopened.pending_runs() == []


def test_resource_sampler_has_a_base_install_fallback(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "psutil", None)
    sampler = service_module.ProcessTreeSampler(os.getpid())

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
        service_module,
        "psutil",
        SimpleNamespace(Process=FakeProcess, Error=OSError, STATUS_ZOMBIE="zombie"),
    )
    sampler = service_module.ProcessTreeSampler(10)

    before = sampler.sample()
    state["child_alive"] = False
    after = sampler.sample()

    assert before["cpu_user_seconds"] == 90.0
    assert before["cpu_system_seconds"] == 17.0
    assert after["cpu_user_seconds"] == 110.0
    assert after["cpu_system_seconds"] == 22.0


def test_emitter_startup_timeout_terminates_and_reaps_service(
    tmp_path, monkeypatch
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


def test_local_socket_acknowledges_after_durable_queue(tmp_path) -> None:
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
    tmp_path, monkeypatch
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
    assert events[-1]["status"] == "completed"
