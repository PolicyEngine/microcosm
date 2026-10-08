"""The local service can host independent publishers without domain branches."""

import socket
import tempfile
import threading
import time
from pathlib import Path

from microcosm.build.emitter_service.auth import CollectorSession
from microcosm.build.emitter_service.components.orrery import OrreryPublicationComponent
from microcosm.build.emitter_service.runtime import EmitterService


class Component:
    def __init__(self, name):
        self.name = name
        self.messages = []
        self.started = False
        self.exit_recorded = False
        self.ran = threading.Event()

    def start(self):
        self.started = True

    def handle(self, message):
        if message.get("action") != self.name:
            return False
        self.messages.append(message)
        return True

    def run(self, stop):
        self.ran.set()
        stop.wait(2)

    def parent_exited(self):
        self.exit_recorded = True


def test_generic_runtime_routes_a_third_component_without_telemetry_registration(
    tmp_path,
):
    parts = [Component("telemetry"), Component("orrery"), Component("future_publisher")]
    service = EmitterService(
        socket_path=tmp_path / "emitter.sock",
        components=parts,
        parent_alive=lambda: True,
    )
    for component in parts:
        service._handle({"action": component.name, "value": 1})
        assert len(component.messages) == 1
    service._handle({"action": "ping"})
    service._handle({"action": "close"})
    assert service._stop.is_set()


def test_slow_orrery_worker_does_not_block_another_publisher_or_socket(
    real_local_telemetry,
):
    blocked = threading.Event()
    entered = threading.Event()

    class SlowDelivery:
        def flush_once(self):
            entered.set()
            blocked.wait(5)

    other = Component("future_publisher")
    directory = Path(tempfile.mkdtemp(prefix="emitter-components-", dir="/tmp"))
    service = EmitterService(
        socket_path=directory / "e.sock",
        components=[OrreryPublicationComponent(SlowDelivery()), other],
        parent_alive=lambda: True,
        shutdown_seconds=0,
    )
    thread = threading.Thread(target=service.run)
    thread.start()
    try:
        assert entered.wait(2)
        assert other.ran.is_set()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(0.5)
            client.connect(str(service.socket_path))
            client.sendall(b'{"action":"ping"}\n')
            assert client.recv(16) == b"ok\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(service.socket_path))
            client.sendall(b'{"action":"close"}\n')
            assert client.recv(16) == b"ok\n"
    finally:
        service._stop.set()
        blocked.set()
        thread.join(2)
    assert not thread.is_alive()


def test_parent_exit_notifies_every_component_before_shutdown(real_local_telemetry):
    parts = [Component("one"), Component("two")]
    service = EmitterService(
        socket_path=Path(tempfile.mkdtemp(prefix="emitter-exit-", dir="/tmp"))
        / "e.sock",
        components=parts,
        parent_alive=lambda: False,
        shutdown_seconds=0,
    )
    service.run()
    assert all(component.exit_recorded for component in parts)


def test_shared_authentication_exchanges_once_across_concurrent_components():
    calls = []

    def exchange(*args):
        calls.append(args)
        time.sleep(0.01)
        return 200, {"access_token": "session", "expires_in": 3600}

    session = CollectorSession(
        "https://collector.example", post=exchange, credential=lambda: "hf-secret"
    )
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(session.credential()))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [("ok", "session"), ("ok", "session")]
    assert len(calls) == 1
    session.invalidate()
    assert session.credential() == ("ok", "session")
    assert len(calls) == 2


def test_general_service_can_be_started_without_a_telemetry_run(tmp_path):
    from microcosm.build.emitter_service.main import build_parser

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
