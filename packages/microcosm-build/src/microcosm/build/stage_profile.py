"""Resident-memory (RSS) and wall-time profiling for build stages.

Each profiled stage gets one record in ``stage_profile.json``.

``entry_rss_bytes`` and ``exit_rss_bytes`` are read by ``current_rss_bytes``
as the stage starts and ends. That is current RSS from ``/proc/self/statm``
where it is readable, and otherwise current RSS from one ``ps`` call. Where
``ps`` gives no usable reading, the reading is the process's lifetime peak RSS
from ``getrusage`` instead of its current RSS. "No usable reading" means no
``ps`` is on ``PATH``; or running it raises ``OSError``, exits non-zero or
times out; or it prints something other than a positive number. That happens
on Linux without ``/proc``, because ``ps`` reads ``/proc`` there, in images
that ship without ``ps``, and in sandboxes that deny ``ps``.

``peak_rss_bytes`` is measured in one of two ways, and ``peak_rss_source``
says which:

- Where ``/proc/self/statm`` is readable (Linux), a background thread reads
  current RSS every ``sample_interval_seconds``. The peak is the largest of
  those samples and the entry and exit readings (``"sampled_current_rss"``).
  Each sample is taken by ``current_rss_bytes``, so if ``/proc`` stops being
  readable during the stage, a sample can come from ``ps`` or ``getrusage``.
  If a sample fails, sampling stops for the rest of the stage and
  ``sampling_error`` says why, so the peak covers only the samples taken
  before that.
- Elsewhere (macOS, or a container that hides ``/proc``), no background thread
  runs, because each reading of current RSS would start a ``ps`` process.
  Instead the process's lifetime peak RSS, ``getrusage(RUSAGE_SELF).ru_maxrss``,
  is read just after the entry reading and again just after the exit reading.
  The peak is the larger of the second ``ru_maxrss`` reading and the entry and
  exit readings. It is the second ``ru_maxrss`` reading unless the entry or
  exit reading disagrees with ``getrusage``.

  - ``"process_peak_rss_reached_during_stage"``: ``ru_maxrss`` rose between
    the two readings. So the process reached its highest RSS so far during
    the stage or, by at most a few pages, during the exit reading's ``ps``
    call. The peak is the stage's own peak.
  - ``"process_peak_rss_reached_before_stage"``: ``ru_maxrss`` did not rise.
    The process reached the recorded peak before the stage, and the stage's
    own peak is at most that. The earlier peak can come from startup and
    imports, or from an earlier profiled stage in the same process. On Linux
    it can also be the parent process's peak: a child started with ``vfork``
    or ``posix_spawn``, as Python's ``subprocess`` normally does there,
    receives the parent's peak in its ``ru_maxrss`` at ``exec``. macOS starts
    a new process's ``ru_maxrss`` near zero.

On either path the record says ``"entry_and_exit_rss"`` when nothing between
the entry and exit readings was measured: the stage ended before the first
background sample, the first sample failed, or ``getrusage`` gave no usable
value. ``sampling_error`` says what failed, if anything did. The peak is the
larger of the entry and exit readings.

In every case ``peak_rss_bytes`` is at least ``entry_rss_bytes`` and
``exit_rss_bytes``.

Records written before ``peak_rss_source`` existed lack the field. Their peak
is the largest of their ``rss_sample_count`` background samples and their
entry and exit readings. Each of those readings was taken the way
``current_rss_bytes`` takes one, which on macOS started a ``ps`` process for
each.
"""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGE_PROFILE_FILENAME = "stage_profile.json"
STAGE_PROFILE_SCHEMA_VERSION = 1
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.25
_PS_TIMEOUT_SECONDS = 5.0

PEAK_RSS_SAMPLED_CURRENT_RSS = "sampled_current_rss"
PEAK_RSS_PROCESS_PEAK_REACHED_DURING_STAGE = "process_peak_rss_reached_during_stage"
PEAK_RSS_PROCESS_PEAK_REACHED_BEFORE_STAGE = "process_peak_rss_reached_before_stage"
PEAK_RSS_ENTRY_AND_EXIT_RSS = "entry_and_exit_rss"


def current_rss_bytes() -> int:
    """Return the current process resident set size in bytes.

    Linux exposes current RSS through procfs. macOS does not, so the portable
    fallback asks ``ps`` for the current process's RSS in KiB. Both report
    current RSS. Where ``ps`` gives no usable reading (it is not on ``PATH``;
    running it raises ``OSError``, exits non-zero or times out; or its output
    is not a positive number), this returns the lifetime peak from
    ``resource.getrusage`` instead, which is at least the current RSS.
    """

    procfs_rss_bytes = _read_procfs_current_rss_bytes()
    if procfs_rss_bytes is not None:
        return procfs_rss_bytes
    try:
        return _read_ps_current_rss_bytes()
    except (OSError, subprocess.SubprocessError, _PsRssUnavailableError):
        # Some sandboxed macOS workers deny process inspection even for the
        # current PID, Linux ps fails without /proc, and slim images ship no
        # ps at all. Keep profiling non-fatal there; ru_maxrss is a lifetime
        # high-water mark rather than current RSS, but remains a truthful
        # conservative RSS observation.
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


class _PsRssUnavailableError(RuntimeError):
    """``ps`` is missing or printed no positive RSS."""


def _read_ps_current_rss_bytes() -> int:
    ps_path = shutil.which("ps")
    if ps_path is None:
        raise _PsRssUnavailableError(
            "Unable to read current RSS: ps executable not found"
        )
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
        raise _PsRssUnavailableError(
            f"Unable to parse current RSS from ps output: {completed.stdout!r}"
        ) from error
    if rss_kib <= 0:
        raise _PsRssUnavailableError(
            f"ps reported non-positive current RSS: {rss_kib} KiB"
        )
    return rss_kib * 1024


def _read_resource_peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if peak <= 0:
        raise RuntimeError(f"resource reported non-positive peak RSS: {peak}")
    return peak if os.uname().sysname == "Darwin" else peak * 1024


# Keep a narrow patch seam for tests without adding a test-only argument to the
# public context manager.
_current_rss_bytes = current_rss_bytes


def _procfs_reports_current_rss() -> bool:
    """Return whether current RSS can be read without starting a process."""

    return _read_procfs_current_rss_bytes() is not None


@dataclass(frozen=True)
class _PeakRss:
    """One stage's peak RSS and how it was measured."""

    peak_rss_bytes: int
    sample_count: int
    sampling_error: dict[str, str] | None
    source: str


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
            name=f"microcosm-stage-profile-{stage_name}",
            daemon=True,
        )

    def start(self) -> None:
        """Start background sampling."""

        self._thread.start()

    def stop(self) -> None:
        """Stop and join the sampler."""

        self._stop_event.set()
        self._thread.join()

    def result(self, *, exit_rss_bytes: int) -> _PeakRss:
        """Return final sampler state including the synchronous exit sample."""

        with self._state_lock:
            self._peak_rss_bytes = max(self._peak_rss_bytes, exit_rss_bytes)
            return _PeakRss(
                peak_rss_bytes=self._peak_rss_bytes,
                sample_count=self._sample_count,
                sampling_error=self._sampling_error,
                source=(
                    PEAK_RSS_SAMPLED_CURRENT_RSS
                    if self._sample_count > 0
                    else PEAK_RSS_ENTRY_AND_EXIT_RSS
                ),
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


class _ProcessPeakRssReader:
    """Read the process's lifetime peak RSS when one stage starts and ends.

    Used where reading current RSS means starting a ``ps`` process. It starts
    no thread and no process. Each ``ru_maxrss`` reading is taken just after
    the entry or exit reading, so that reading is at most the lifetime peak
    read next to it.
    """

    def __init__(self, *, entry_rss_bytes: int) -> None:
        self._entry_rss_bytes = entry_rss_bytes
        self._peak_at_start: int | None = None
        self._sampling_error: dict[str, str] | None = None

    def start(self) -> None:
        """Read the lifetime peak just after the entry reading."""

        self._peak_at_start = self._read()

    def stop(self) -> None:
        """Do nothing; ``result`` reads the lifetime peak after the exit reading."""

    def result(self, *, exit_rss_bytes: int) -> _PeakRss:
        """Read the lifetime peak again and return the stage's peak.

        The peak is never below the entry or exit reading.
        """

        peak_at_end = self._read() if self._sampling_error is None else None
        entry_and_exit_peak = max(self._entry_rss_bytes, exit_rss_bytes)
        if self._peak_at_start is None or peak_at_end is None:
            return _PeakRss(
                peak_rss_bytes=entry_and_exit_peak,
                sample_count=0,
                sampling_error=self._sampling_error,
                source=PEAK_RSS_ENTRY_AND_EXIT_RSS,
            )
        return _PeakRss(
            peak_rss_bytes=max(peak_at_end, entry_and_exit_peak),
            sample_count=0,
            sampling_error=None,
            source=(
                PEAK_RSS_PROCESS_PEAK_REACHED_DURING_STAGE
                if peak_at_end > self._peak_at_start
                else PEAK_RSS_PROCESS_PEAK_REACHED_BEFORE_STAGE
            ),
        )

    def _read(self) -> int | None:
        try:
            return _read_resource_peak_rss_bytes()
        except Exception as error:  # noqa: BLE001
            self._sampling_error = _exception_payload(error)
            return None


@contextmanager
def profile_stage(
    stage_name: str,
    checkpoint_dir: str | Path,
    *,
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
) -> Iterator[None]:
    """Profile one build stage and atomically merge its result into JSON.

    The stage body is never swallowed: on failure, the exception is recorded,
    any sampler thread is stopped and joined, and the original exception is
    re-raised. Separate processes coordinate updates with a file lock so a
    later stage does not discard records written by an earlier invocation.
    The module docstring describes how the peak is measured.

    Args:
        stage_name: Stable stage identifier, such as ``"a"``.
        checkpoint_dir: Directory that owns ``stage_profile.json``.
        sample_interval_seconds: Delay between background current-RSS samples.
            Used only where ``/proc`` reports current RSS.

    Yields:
        Control to the profiled stage body.
    """

    _validate_inputs(stage_name, sample_interval_seconds)
    checkpoint_path = Path(checkpoint_dir)
    samples_current_rss = _procfs_reports_current_rss()
    entry_rss_bytes = _current_rss_bytes()
    started_at = time.perf_counter()
    peak_reader: _RssSampler | _ProcessPeakRssReader
    if samples_current_rss:
        peak_reader = _RssSampler(
            stage_name=stage_name,
            entry_rss_bytes=entry_rss_bytes,
            sample_interval_seconds=sample_interval_seconds,
        )
    else:
        peak_reader = _ProcessPeakRssReader(entry_rss_bytes=entry_rss_bytes)
    peak_reader.start()
    stage_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        stage_error = error
        raise
    finally:
        peak_reader.stop()
        try:
            exit_rss_bytes = _current_rss_bytes()
            peak = peak_reader.result(exit_rss_bytes=exit_rss_bytes)
            wall_seconds = time.perf_counter() - started_at
            record = {
                "entry_rss_bytes": entry_rss_bytes,
                "error": (
                    _exception_payload(stage_error) if stage_error is not None else None
                ),
                "exit_rss_bytes": exit_rss_bytes,
                "peak_rss_bytes": peak.peak_rss_bytes,
                "peak_rss_source": peak.source,
                "rss_sample_count": peak.sample_count,
                "sampling_error": peak.sampling_error,
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
    "PEAK_RSS_ENTRY_AND_EXIT_RSS",
    "PEAK_RSS_PROCESS_PEAK_REACHED_BEFORE_STAGE",
    "PEAK_RSS_PROCESS_PEAK_REACHED_DURING_STAGE",
    "PEAK_RSS_SAMPLED_CURRENT_RSS",
    "STAGE_PROFILE_FILENAME",
    "STAGE_PROFILE_SCHEMA_VERSION",
    "current_rss_bytes",
    "profile_stage",
]
