"""Cut-point-independent RSS and wall-time profiling for build stages."""

from __future__ import annotations

import fcntl
import json
import math
import os
import resource
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

STAGE_PROFILE_FILENAME = "stage_profile.json"
STAGE_PROFILE_SCHEMA_VERSION = 1
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.25
_PS_TIMEOUT_SECONDS = 5.0


def current_rss_bytes() -> int:
    """Return the current process resident set size in bytes.

    Linux exposes current RSS through procfs. macOS does not, so the portable
    fallback asks ``ps`` for the current process's RSS in KiB. Unlike
    ``resource.getrusage``, both paths report current rather than lifetime-peak
    RSS.
    """

    procfs_rss_bytes = _read_procfs_current_rss_bytes()
    if procfs_rss_bytes is not None:
        return procfs_rss_bytes
    try:
        return _read_ps_current_rss_bytes()
    except (OSError, subprocess.SubprocessError):
        # Some sandboxed macOS workers deny process inspection even for the
        # current PID.  Keep profiling non-fatal there; ru_maxrss is a
        # lifetime high-water mark rather than current RSS, but remains a
        # truthful conservative RSS observation.
        return _read_resource_peak_rss_bytes()


def _read_procfs_current_rss_bytes() -> int | None:
    procfs_path = Path("/proc/self/statm")
    if not procfs_path.exists():
        return None
    try:
        fields = procfs_path.read_text(encoding="ascii").split()
        resident_pages = int(fields[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        rss_bytes = resident_pages * page_size
        if rss_bytes <= 0:
            raise ValueError("procfs reported non-positive RSS")
        return rss_bytes
    except (IndexError, OSError, ValueError):
        # Containers can expose an incomplete or unreadable procfs. In that
        # case, use the same ps fallback as macOS.
        return None


def _read_ps_current_rss_bytes() -> int:
    ps_path = shutil.which("ps")
    if ps_path is None:
        raise RuntimeError("Unable to read current RSS: ps executable not found")
    completed = subprocess.run(
        [ps_path, "-o", "rss=", "-p", str(os.getpid())],
        check=True,
        capture_output=True,
        text=True,
        timeout=_PS_TIMEOUT_SECONDS,
    )
    try:
        rss_kib = int(completed.stdout.strip())
    except ValueError as error:
        raise RuntimeError(
            f"Unable to parse current RSS from ps output: {completed.stdout!r}"
        ) from error
    if rss_kib <= 0:
        raise RuntimeError(f"ps reported non-positive current RSS: {rss_kib} KiB")
    return rss_kib * 1024


def _read_resource_peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if peak <= 0:
        raise RuntimeError(f"resource reported non-positive peak RSS: {peak}")
    return peak if os.uname().sysname == "Darwin" else peak * 1024


# Keep a narrow patch seam for tests without adding a test-only argument to the
# public context manager.
_current_rss_bytes = current_rss_bytes


class _RssSampler:
    """Own one stage's background RSS sampling thread."""

    def __init__(
        self,
        *,
        stage_name: str,
        entry_rss_bytes: int,
        sample_interval_seconds: float,
    ) -> None:
        self._sample_interval_seconds = sample_interval_seconds
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._peak_rss_bytes = entry_rss_bytes
        self._sample_count = 0
        self._sampling_error: dict[str, str] | None = None
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name=f"populace-stage-profile-{stage_name}",
            daemon=True,
        )

    def start(self) -> None:
        """Start background sampling."""

        self._thread.start()

    def stop(self) -> None:
        """Stop and join the sampler."""

        self._stop_event.set()
        self._thread.join()

    def result(self, *, exit_rss_bytes: int) -> tuple[int, int, dict[str, str] | None]:
        """Return final sampler state including the synchronous exit sample."""

        with self._state_lock:
            self._peak_rss_bytes = max(self._peak_rss_bytes, exit_rss_bytes)
            return (
                self._peak_rss_bytes,
                self._sample_count,
                self._sampling_error,
            )

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self._sample_interval_seconds):
            try:
                rss_bytes = _current_rss_bytes()
            except Exception as error:  # noqa: BLE001
                with self._state_lock:
                    self._sampling_error = _exception_payload(error)
                return
            with self._state_lock:
                self._peak_rss_bytes = max(self._peak_rss_bytes, rss_bytes)
                self._sample_count += 1


@contextmanager
def profile_stage(
    stage_name: str,
    checkpoint_dir: str | Path,
    *,
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
) -> Iterator[None]:
    """Profile one build stage and atomically merge its result into JSON.

    The stage body is never swallowed: on failure, the exception is recorded,
    the sampler is stopped and joined, and the original exception is re-raised.
    Separate processes coordinate updates with a file lock so a later stage does
    not discard records written by an earlier invocation.

    Args:
        stage_name: Stable stage identifier, such as ``"a"``.
        checkpoint_dir: Directory that owns ``stage_profile.json``.
        sample_interval_seconds: Delay between background current-RSS samples.

    Yields:
        Control to the profiled stage body.
    """

    _validate_inputs(stage_name, sample_interval_seconds)
    checkpoint_path = Path(checkpoint_dir)
    entry_rss_bytes = _current_rss_bytes()
    started_at = time.perf_counter()
    sampler = _RssSampler(
        stage_name=stage_name,
        entry_rss_bytes=entry_rss_bytes,
        sample_interval_seconds=sample_interval_seconds,
    )
    sampler.start()
    stage_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        stage_error = error
        raise
    finally:
        sampler.stop()
        try:
            exit_rss_bytes = _current_rss_bytes()
            peak_rss_bytes, sample_count, sampling_error = sampler.result(
                exit_rss_bytes=exit_rss_bytes
            )
            wall_seconds = time.perf_counter() - started_at
            record = {
                "entry_rss_bytes": entry_rss_bytes,
                "error": (
                    _exception_payload(stage_error) if stage_error is not None else None
                ),
                "exit_rss_bytes": exit_rss_bytes,
                "peak_rss_bytes": peak_rss_bytes,
                "rss_sample_count": sample_count,
                "sampling_error": sampling_error,
                "stage": stage_name,
                "status": "failed" if stage_error is not None else "succeeded",
                "wall_seconds": wall_seconds,
            }
            _merge_profile_record(checkpoint_path, stage_name, record)
        except Exception as profile_error:
            if stage_error is None:
                raise
            stage_error.add_note(
                "Stage profiling also failed: "
                f"{type(profile_error).__name__}: {profile_error}"
            )


def _validate_inputs(stage_name: str, sample_interval_seconds: float) -> None:
    if not isinstance(stage_name, str) or not stage_name.strip():
        raise ValueError("stage_name must be a non-empty string")
    if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0.0:
        raise ValueError("sample_interval_seconds must be finite and positive")


def _exception_payload(error: BaseException) -> dict[str, str]:
    error_type = type(error)
    return {
        "message": str(error),
        "type": f"{error_type.__module__}.{error_type.__qualname__}",
    }


def _merge_profile_record(
    checkpoint_dir: Path,
    stage_name: str,
    record: Mapping[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    profile_path = checkpoint_dir / STAGE_PROFILE_FILENAME
    lock_path = checkpoint_dir / f".{STAGE_PROFILE_FILENAME}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = _load_profile_payload(profile_path)
            stages = dict(payload["stages"])
            stages[stage_name] = dict(record)
            updated_payload = {
                "schema_version": STAGE_PROFILE_SCHEMA_VERSION,
                "stages": stages,
            }
            _atomic_write_json(profile_path, updated_payload)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_profile_payload(profile_path: Path) -> dict[str, Any]:
    if not profile_path.exists():
        return {
            "schema_version": STAGE_PROFILE_SCHEMA_VERSION,
            "stages": {},
        }
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid stage profile JSON: {profile_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Stage profile must contain a JSON object: {profile_path}")
    if payload.get("schema_version") != STAGE_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported stage profile schema version: "
            f"{payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("stages"), dict):
        raise ValueError("Stage profile 'stages' must be a JSON object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_SAMPLE_INTERVAL_SECONDS",
    "STAGE_PROFILE_FILENAME",
    "STAGE_PROFILE_SCHEMA_VERSION",
    "current_rss_bytes",
    "profile_stage",
]
