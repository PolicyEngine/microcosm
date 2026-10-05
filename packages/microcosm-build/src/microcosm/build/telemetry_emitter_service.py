"""Standalone local service that delivers Microcosm telemetry to a collector."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from huggingface_hub import get_token

try:
    import psutil
except ModuleNotFoundError:  # Base installs use the standard-library fallback.
    psutil = None

SCHEMA_VERSION = 1
RETENTION_DAYS = 7
MAX_QUEUED_BYTES = 100 * 1024 * 1024
BATCH_SIZE = 100


def _now() -> str:
    return datetime.now(UTC).isoformat()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep bearer credentials on the explicitly configured origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class EventSpool:
    """Small SQLite queue shared by successive emitter service processes."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        self._connection = sqlite3.connect(
            self.path, timeout=5, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._last_prune_at = 0.0
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS telemetry_runs (
                    run_id TEXT NOT NULL,
                    producer_id TEXT NOT NULL,
                    registration_json TEXT NOT NULL,
                    next_sequence INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, producer_id)
                );
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    producer_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, producer_id, sequence),
                    FOREIGN KEY(run_id, producer_id)
                        REFERENCES telemetry_runs(run_id, producer_id)
                );
                CREATE INDEX IF NOT EXISTS telemetry_events_run_sequence
                    ON telemetry_events(run_id, producer_id, sequence);
                """
            )
        self.prune()

    def register(self, registration: Mapping[str, Any]) -> None:
        run_id = str(registration["run_id"])
        producer_id = str(registration["producer_id"])
        encoded = json.dumps(registration, separators=(",", ":"), sort_keys=True)
        with self._lock, self._connection:
            existing = self._connection.execute(
                """
                SELECT registration_json FROM telemetry_runs
                WHERE run_id = ? AND producer_id = ?
                """,
                (run_id, producer_id),
            ).fetchone()
            if existing is not None:
                old = json.loads(existing["registration_json"])
                if old.get("producer_id") != registration.get("producer_id"):
                    raise ValueError(f"run_id {run_id!r} already has another producer")
            self._connection.execute(
                """
                INSERT INTO telemetry_runs (
                    run_id, producer_id, registration_json, next_sequence, updated_at
                ) VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(run_id, producer_id) DO UPDATE SET
                    registration_json = excluded.registration_json,
                    updated_at = excluded.updated_at
                """,
                (run_id, producer_id, encoded, _now()),
            )

    def append(
        self,
        registration: Mapping[str, Any],
        event: Mapping[str, Any],
        *,
        resources: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_id = str(registration["run_id"])
        producer_id = str(registration["producer_id"])
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT next_sequence FROM telemetry_runs
                WHERE run_id = ? AND producer_id = ?
                """,
                (run_id, producer_id),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            sequence = int(row["next_sequence"])
            event_id = uuid.uuid4().hex
            payload = {
                "schema_version": SCHEMA_VERSION,
                "event_id": event_id,
                "run_id": run_id,
                "producer_id": producer_id,
                "sequence": sequence,
                "timestamp": event.get("timestamp") or _now(),
                "event_type": event["event_type"],
                "stage_id": event.get("stage_id"),
                "status": event["status"],
                "message": event.get("message"),
                "details": event.get("details") or {},
                "resources": resources,
            }
            encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
            self._connection.execute(
                """
                INSERT INTO telemetry_events (
                    event_id, run_id, producer_id, sequence,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, run_id, producer_id, sequence, encoded, _now()),
            )
            self._connection.execute(
                """
                UPDATE telemetry_runs
                SET next_sequence = ?, updated_at = ?
                WHERE run_id = ? AND producer_id = ?
                """,
                (sequence + 1, _now(), run_id, producer_id),
            )
        if time.monotonic() - self._last_prune_at >= 60:
            self.prune()
        return payload

    def pending_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT DISTINCT r.registration_json
                FROM telemetry_runs r
                JOIN telemetry_events e
                  ON e.run_id = r.run_id AND e.producer_id = r.producer_id
                ORDER BY r.updated_at
                """
            ).fetchall()
        return [json.loads(row["registration_json"]) for row in rows]

    def batch(
        self, run_id: str, producer_id: str, limit: int = BATCH_SIZE
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload_json FROM telemetry_events
                WHERE run_id = ? AND producer_id = ?
                ORDER BY sequence LIMIT ?
                """,
                (run_id, producer_id, limit),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def acknowledge(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        with self._lock, self._connection:
            self._connection.execute(
                f"DELETE FROM telemetry_events WHERE event_id IN ({placeholders})",
                event_ids,
            )

    def has_pending(self) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM telemetry_events LIMIT 1"
            ).fetchone()
        return row is not None

    def prune(self) -> None:
        cutoff = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS)).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM telemetry_events WHERE created_at < ?", (cutoff,)
            )
            size_row = self._connection.execute(
                "SELECT COALESCE(SUM(LENGTH(payload_json)), 0) AS bytes "
                "FROM telemetry_events"
            ).fetchone()
            excess = int(size_row["bytes"]) - MAX_QUEUED_BYTES
            if excess > 0:
                rows = self._connection.execute(
                    """
                    SELECT event_id, LENGTH(payload_json) AS bytes
                    FROM telemetry_events ORDER BY created_at, sequence
                    """
                ).fetchall()
                removed = 0
                ids: list[str] = []
                for row in rows:
                    ids.append(row["event_id"])
                    removed += int(row["bytes"])
                    if removed >= excess:
                        break
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    self._connection.execute(
                        f"DELETE FROM telemetry_events "
                        f"WHERE event_id IN ({placeholders})",
                        ids,
                    )
            self._connection.execute(
                """
                DELETE FROM telemetry_runs
                WHERE updated_at < ? AND NOT EXISTS (
                    SELECT 1 FROM telemetry_events e
                    WHERE e.run_id = telemetry_runs.run_id
                      AND e.producer_id = telemetry_runs.producer_id
                )
                """,
                (cutoff,),
            )
        self._last_prune_at = time.monotonic()


def _http_post(
    url: str,
    payload: Mapping[str, Any],
    bearer_token: str,
    *,
    timeout: float = 5.0,
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "User-Agent": "microcosm-telemetry-emitter/1",
        },
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(_NoRedirectHandler)
        with opener.open(request, timeout=timeout) as response:
            body = response.read(1_048_577)
            if len(body) > 1_048_576:
                raise OSError("collector response exceeds 1 MiB")
            try:
                payload = json.loads(body) if body else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {}
            return response.status, payload
    except urllib.error.HTTPError as error:
        body = error.read(1_048_577)
        if len(body) > 1_048_576:
            return error.code, {}
        try:
            detail = json.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = {}
        return error.code, detail


def _huggingface_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        or get_token()
    )


class CollectorDelivery:
    """Authenticate queued runs and deliver idempotent event batches."""

    def __init__(self, collector_url: str, spool: EventSpool) -> None:
        self.collector_url = collector_url.rstrip("/")
        self.spool = spool
        self._tokens: dict[str, tuple[str, float]] = {}
        self._denied: set[str] = set()
        self._warned_no_token = False
        self._warned_denied: set[str] = set()
        self._next_attempt_at = 0.0
        self._retry_seconds = 1.0

    def flush_once(self) -> bool:
        if time.monotonic() < self._next_attempt_at:
            return False
        made_progress = False
        for registration in self.spool.pending_runs():
            run_id = registration["run_id"]
            registration_key = f"{run_id}:{registration['producer_id']}"
            if registration_key in self._denied:
                continue
            token = self._run_token(registration)
            if token is None:
                continue
            events = self.spool.batch(run_id, registration["producer_id"])
            if not events:
                continue
            try:
                status, _ = _http_post(
                    f"{self.collector_url}/v1/runs/{run_id}/events",
                    {"events": events},
                    token,
                )
            except (OSError, TimeoutError):
                self._defer_retry()
                continue
            if status == 202:
                self.spool.acknowledge([event["event_id"] for event in events])
                made_progress = True
                self._retry_seconds = 1.0
            elif status == 401:
                self._tokens.pop(registration_key, None)
            elif status == 403:
                self._denied.add(registration_key)
                self._warn_denied(registration_key)
            else:
                self._defer_retry()
        return made_progress

    def _run_token(self, registration: Mapping[str, Any]) -> str | None:
        run_id = str(registration["run_id"])
        registration_key = f"{run_id}:{registration['producer_id']}"
        cached = self._tokens.get(registration_key)
        if cached is not None and cached[1] > time.monotonic() + 30:
            return cached[0]
        hf_token = _huggingface_token()
        if not hf_token:
            if not self._warned_no_token:
                print(
                    "Microcosm telemetry is local-only: no ambient Hugging Face "
                    "credential was found. The dataset build will continue.",
                    file=sys.stderr,
                    flush=True,
                )
                self._warned_no_token = True
            self._next_attempt_at = time.monotonic() + 60.0
            return None
        try:
            status, response = _http_post(
                f"{self.collector_url}/v1/auth/huggingface/exchange",
                registration,
                hf_token,
            )
        except (OSError, TimeoutError):
            self._defer_retry()
            return None
        if status == 200 and isinstance(response.get("access_token"), str):
            expires_in = max(60, int(response.get("expires_in", 900)))
            token = response["access_token"]
            self._tokens[registration_key] = (token, time.monotonic() + expires_in)
            return token
        if status in {401, 403}:
            self._denied.add(registration_key)
            self._warn_denied(registration_key)
        else:
            self._defer_retry()
        return None

    def _defer_retry(self) -> None:
        self._next_attempt_at = time.monotonic() + self._retry_seconds
        self._retry_seconds = min(60.0, self._retry_seconds * 2)

    def _warn_denied(self, run_id: str) -> None:
        if run_id in self._warned_denied:
            return
        print(
            "Microcosm telemetry is local-only for this run: the ambient "
            "Hugging Face credential was not accepted as a PolicyEngine "
            "organization member. The dataset build will continue.",
            file=sys.stderr,
            flush=True,
        )
        self._warned_denied.add(run_id)


class ProcessTreeSampler:
    """Collect cumulative CPU and resident memory for a build process tree."""

    def __init__(self, parent_pid: int) -> None:
        self.parent_pid = parent_pid
        self._peak_rss = 0
        self._parent_create_time = self._create_time()

    def _create_time(self) -> float | str | None:
        if psutil is not None:
            try:
                return psutil.Process(self.parent_pid).create_time()
            except (psutil.Error, OSError):
                return None
        stat_path = Path(f"/proc/{self.parent_pid}/stat")
        try:
            return stat_path.read_text().rsplit(")", 1)[1].split()[19]
        except (OSError, IndexError):
            return None

    def parent_alive(self) -> bool:
        if psutil is not None:
            try:
                process = psutil.Process(self.parent_pid)
                return (
                    self._parent_create_time is not None
                    and process.create_time() == self._parent_create_time
                    and process.is_running()
                    and process.status() != psutil.STATUS_ZOMBIE
                )
            except (psutil.Error, OSError):
                return False
        try:
            os.kill(self.parent_pid, 0)
        except OSError:
            return False
        current = self._create_time()
        # /proc supplies a process-start tick that protects against PID reuse.
        # On platforms without it, existence is the best standard-library test.
        return self._parent_create_time is None or current == self._parent_create_time

    def sample(self) -> dict[str, Any]:
        if psutil is None:
            return self._fallback_sample()
        processes = []
        try:
            parent = psutil.Process(self.parent_pid)
            processes = [parent, *parent.children(recursive=True)]
        except (psutil.Error, OSError):
            pass
        user = 0.0
        system = 0.0
        rss = 0
        for process in processes:
            try:
                cpu = process.cpu_times()
                user += float(cpu.user) + float(getattr(cpu, "children_user", 0.0))
                system += float(cpu.system) + float(
                    getattr(cpu, "children_system", 0.0)
                )
                rss += int(process.memory_info().rss)
            except (psutil.Error, OSError):
                continue
        self._peak_rss = max(self._peak_rss, rss)
        return {
            "cpu_user_seconds": user,
            "cpu_system_seconds": system,
            "rss_bytes": rss,
            "peak_rss_bytes": self._peak_rss,
        }

    def _fallback_sample(self) -> dict[str, Any]:
        """Sample Linux process trees when the country engine is not installed."""

        process_rows: dict[int, tuple[int, float, float, int]] = {}
        try:
            clock_ticks = float(os.sysconf("SC_CLK_TCK"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            return {
                "cpu_user_seconds": 0.0,
                "cpu_system_seconds": 0.0,
                "rss_bytes": 0,
                "peak_rss_bytes": self._peak_rss,
            }
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                pid = int(stat_path.parent.name)
                fields = stat_path.read_text().rsplit(")", 1)[1].split()
                process_rows[pid] = (
                    int(fields[1]),
                    int(fields[11]) / clock_ticks,
                    int(fields[12]) / clock_ticks,
                    int(fields[21]) * page_size,
                )
            except (OSError, ValueError, IndexError):
                continue
        selected = {self.parent_pid}
        changed = True
        while changed:
            changed = False
            for pid, (parent, *_rest) in process_rows.items():
                if parent in selected and pid not in selected:
                    selected.add(pid)
                    changed = True
        user = sum(process_rows[pid][1] for pid in selected if pid in process_rows)
        system = sum(process_rows[pid][2] for pid in selected if pid in process_rows)
        rss = sum(process_rows[pid][3] for pid in selected if pid in process_rows)
        self._peak_rss = max(self._peak_rss, rss)
        return {
            "cpu_user_seconds": user,
            "cpu_system_seconds": system,
            "rss_bytes": rss,
            "peak_rss_bytes": self._peak_rss,
        }


class EmitterService:
    """Private local socket server with a concurrent delivery worker."""

    def __init__(
        self,
        *,
        socket_path: Path,
        registration: Mapping[str, Any],
        spool: EventSpool,
        delivery: CollectorDelivery,
        sampler: ProcessTreeSampler,
        heartbeat_seconds: float,
        drain_seconds: float = 15.0,
    ) -> None:
        self.socket_path = socket_path
        self.registration = dict(registration)
        self.spool = spool
        self.delivery = delivery
        self.sampler = sampler
        self.heartbeat_seconds = max(1.0, heartbeat_seconds)
        self.drain_seconds = max(0.0, drain_seconds)
        self._stop = threading.Event()
        self._closed_by_client = False
        self._last_stage = "created"

    def run(self) -> None:
        self.spool.register(self.registration)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink(missing_ok=True)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(self.socket_path))
                os.chmod(self.socket_path, 0o600)
                server.listen(16)
                server.settimeout(0.5)
                worker = threading.Thread(target=self._worker, daemon=True)
                worker.start()
                while not self._stop.is_set():
                    try:
                        connection, _ = server.accept()
                    except TimeoutError:
                        continue
                    with connection:
                        connection.settimeout(0.25)
                        try:
                            message = self._read_message(connection)
                            self._handle(message)
                            response = b"ok\n"
                        except Exception:
                            response = b"error\n"
                        try:
                            connection.sendall(response)
                        except OSError:
                            pass
                worker.join(timeout=self.drain_seconds + 1)
        finally:
            self.socket_path.unlink(missing_ok=True)
            try:
                self.socket_path.parent.rmdir()
            except OSError:
                pass

    @staticmethod
    def _read_message(connection: socket.socket) -> dict[str, Any]:
        chunks = bytearray()
        while len(chunks) <= 1_048_576:
            data = connection.recv(65536)
            if not data:
                break
            chunks.extend(data)
            if b"\n" in data:
                break
        if len(chunks) > 1_048_576:
            raise ValueError("local telemetry message exceeds 1 MiB")
        return json.loads(bytes(chunks).split(b"\n", 1)[0])

    def _handle(self, message: Mapping[str, Any]) -> None:
        action = message.get("action")
        if action == "event":
            event = message.get("event")
            if not isinstance(event, Mapping):
                raise ValueError("event must be an object")
            stage_id = event.get("stage_id")
            if isinstance(stage_id, str) and stage_id not in {"complete", "failed"}:
                self._last_stage = stage_id
            self.spool.append(
                self.registration,
                event,
                resources=self.sampler.sample(),
            )
        elif action == "close":
            self._closed_by_client = True
            self._stop.set()
        elif action == "ping":
            return
        else:
            raise ValueError("unsupported local telemetry action")

    def _worker(self) -> None:
        next_heartbeat = time.monotonic() + self.heartbeat_seconds
        while not self._stop.wait(1.0):
            now = time.monotonic()
            # Sample every worker iteration so short-lived build children are
            # much less likely to disappear between stage and heartbeat events.
            self.sampler.sample()
            if now >= next_heartbeat:
                self.spool.append(
                    self.registration,
                    {
                        "timestamp": _now(),
                        "event_type": "heartbeat",
                        "stage_id": self._last_stage,
                        "status": "progress",
                        "message": None,
                        "details": {},
                    },
                    resources=self.sampler.sample(),
                )
                next_heartbeat = now + self.heartbeat_seconds
            self.delivery.flush_once()
            if not self.sampler.parent_alive():
                self.spool.append(
                    self.registration,
                    {
                        "timestamp": _now(),
                        "event_type": "run",
                        "stage_id": "failed",
                        "status": "failed",
                        "message": "The build process exited without reporting completion.",
                        "details": {
                            "failure_class": "unexpected_process_exit",
                            "failed_during": self._last_stage,
                        },
                    },
                    resources=self.sampler.sample(),
                )
                self._stop.set()
                break
        deadline = time.monotonic() + self.drain_seconds
        while self.spool.has_pending() and time.monotonic() < deadline:
            if not self.delivery.flush_once():
                self._stop.wait(0.5)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--collector-url", required=True)
    parser.add_argument("--registration-json", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registration = json.loads(args.registration_json)
    spool = EventSpool(args.spool)
    service = EmitterService(
        socket_path=args.socket,
        registration=registration,
        spool=spool,
        delivery=CollectorDelivery(args.collector_url, spool),
        sampler=ProcessTreeSampler(args.parent_pid),
        heartbeat_seconds=args.heartbeat_seconds,
    )
    service.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
