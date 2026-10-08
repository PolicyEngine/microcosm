"""Independent publication delivery in the existing local service."""

import socket
import tempfile
import threading
import time
from pathlib import Path

from microcosm.build.telemetry_emitter_service.auth import CollectorSession
from microcosm.build.telemetry_emitter_service.main import build_parser
from microcosm.build.telemetry_emitter_service.runtime import EmitterService
from microcosm.build.telemetry_emitter_service.spool import EventSpool


def test_slow_graph_upload_does_not_block_telemetry_or_socket(
    tmp_path, real_local_telemetry
):
    entered, release, telemetry_worked = (threading.Event() for _ in range(3))

    class GraphDelivery:
        def flush_once(self):
            entered.set()
            release.wait(5)

    class TelemetryDelivery:
        def flush_once(self):
            telemetry_worked.set()
            return False

    class Sampler:
        def parent_alive(self):
            return True

        def sample(self):
            return {}

    path = Path(tempfile.mkdtemp(prefix="mc-pub-test-", dir="/tmp")) / "e.sock"
    service = EmitterService(
        socket_path=path,
        registration={
            "run_id": "run",
            "producer_id": "producer",
            "country_code": "GB",
            "pipeline": "test",
        },
        spool=EventSpool(tmp_path / "events.sqlite3"),
        delivery=TelemetryDelivery(),
        graph_delivery=GraphDelivery(),
        sampler=Sampler(),
        heartbeat_seconds=60,
        drain_seconds=0,
    )
    thread = threading.Thread(target=service.run)
    thread.start()
    try:
        assert entered.wait(3)
        assert telemetry_worked.wait(3)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(0.5)
            peer.connect(str(path))
            peer.sendall(b'{"action":"ping"}\n')
            assert peer.recv(16) == b"ok\n"
    finally:
        service._stop.set()
        release.set()
        thread.join(3)
    assert not thread.is_alive()


def test_graph_only_service_requires_no_run_registration(tmp_path):
    args = build_parser().parse_args(
        [
            "--socket",
            str(tmp_path / "e.sock"),
            "--spool",
            str(tmp_path / "jobs.sqlite3"),
            "--parent-pid",
            "1",
        ]
    )
    assert args.registration_json is None


def test_both_workers_share_one_collector_exchange():
    calls = []

    def exchange(*args):
        calls.append(args)
        time.sleep(0.01)
        return 200, {"access_token": "synthetic-session", "expires_in": 3600}

    session = CollectorSession(
        "https://collector.example", post=exchange, credential=lambda: "hf-synthetic"
    )
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(session.credential()))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert results == [("ok", "synthetic-session")] * 2
    assert len(calls) == 1
