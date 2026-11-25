"""The shared worker pool for household-batched work, in spawned processes."""

from __future__ import annotations

import contextlib
import gc
import multiprocessing
import os
import signal
import stat
import subprocess
import sys
import threading
import time
import weakref
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import microcosm.build.worker_pool as worker_pool_module
from microcosm.build.worker_pool import (
    WorkerPool,
    WorkerPoolError,
    WorkerPoolSettings,
    WorkerPoolUnitError,
    process_memory_bytes,
)
from test_support.microcosm_build import worker_pool_units as units
from test_support.microcosm_build.worker_pool_identity import (
    assert_identical_across_worker_counts,
    canonical_result_bytes,
)
from test_support.paths import paths_for

MIB = 1024 * 1024
TEST_SECONDS_LIMIT = 120


@pytest.fixture(autouse=True)
def no_worker_left_behind():
    """Stop a hung test, and check every test leaves no worker running."""

    def expire(signum, frame):
        raise TimeoutError(f"worker-pool test ran past {TEST_SECONDS_LIMIT} seconds")

    previous_handler = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, TEST_SECONDS_LIMIT)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        leftover = multiprocessing.active_children()
        for process in leftover:
            process.kill()
            process.join(5)
    assert leftover == []


def test_results_come_back_in_unit_order_whatever_order_they_finish() -> None:
    # Earlier units sleep longer, so they finish after later ones.
    work = [(index, 0.6 - 0.15 * index) for index in range(4)]
    with WorkerPool(
        WorkerPoolSettings(workers=3), unit_function=units.sleep_then_return
    ) as pool:
        assert list(pool.map(work)) == [0, 1, 2, 3]


def test_the_initializer_runs_once_in_each_worker_and_its_state_reaches_each_unit() -> (
    None
):
    payload = np.arange(1_000, dtype=np.float64)
    with WorkerPool(
        WorkerPoolSettings(workers=2),
        initializer=units.initialize_with_payload,
        initargs=(payload,),
        unit_function=units.describe_worker,
    ) as pool:
        results = list(pool.map(range(6)))
        report = pool.report()

    assert [result["unit"] for result in results] == list(range(6))
    assert all(result["initializer_calls"] == 1 for result in results)
    assert all(result["payload_total"] == float(payload.sum()) for result in results)
    worker_pids = {result["pid"] for result in results}
    assert os.getpid() not in worker_pids
    assert len(worker_pids) <= report.worker_starts == 2
    assert report.units_completed == 6


def test_workers_are_replaced_after_their_unit_limit_and_never_outnumber_the_setting() -> (
    None
):
    with WorkerPool(
        WorkerPoolSettings(workers=2, max_units_per_worker=2),
        initializer=units.initialize_with_payload,
        initargs=(np.ones(10),),
        unit_function=units.describe_worker,
    ) as pool:
        results = list(pool.map(range(7)))
        report = pool.report()

    assert [result["unit"] for result in results] == list(range(7))
    units_per_worker = Counter(result["pid"] for result in results)
    assert max(units_per_worker.values()) <= 2
    assert len(units_per_worker) >= 4
    # Each replacement worker ran its own initializer once.
    assert all(result["initializer_calls"] == 1 for result in results)
    assert report.worker_starts >= len(units_per_worker)
    assert report.max_live_workers <= 2


def test_a_second_map_reuses_the_live_workers() -> None:
    with WorkerPool(
        WorkerPoolSettings(workers=1),
        initializer=units.initialize_with_payload,
        initargs=(np.ones(3),),
        unit_function=units.describe_worker,
    ) as pool:
        first = list(pool.map(range(2)))
        second = list(pool.map(range(2, 4)))
        report = pool.report()

    assert [result["unit"] for result in first + second] == [0, 1, 2, 3]
    assert len({result["pid"] for result in first + second}) == 1
    assert report.worker_starts == 1


def test_no_unit_starts_further_ahead_than_the_window_allows() -> None:
    # Unit 0 runs long. With a window of two units, unit 2 may not start until
    # unit 0's result has been handed back, although a third worker is idle.
    work = [(0, 1.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    with WorkerPool(
        WorkerPoolSettings(workers=3, max_units_ahead=2),
        unit_function=units.record_run_times,
    ) as pool:
        results = list(pool.map(work))

    assert [value for value, _, _ in results] == [0, 1, 2, 3]
    unit_zero_ended = results[0][2]
    assert all(started >= unit_zero_ended for _, started, _ in results[2:])


def test_a_worker_that_dies_has_its_unit_run_again_in_a_fresh_worker(tmp_path) -> None:
    markers = [tmp_path / f"unit-{index}" for index in range(3)]
    # Only unit 1 finds no marker, so only its first attempt kills its worker.
    markers[0].write_text("not this one")
    markers[2].write_text("not this one")
    work = [(index, str(marker)) for index, marker in enumerate(markers)]
    with WorkerPool(
        WorkerPoolSettings(workers=2, max_attempts=2),
        unit_function=units.die_on_first_attempt,
    ) as pool:
        results = list(pool.map(work))
        report = pool.report()

    assert [value for value, _ in results] == [0, 1, 2]
    dead_worker = int(markers[1].read_text())
    assert results[1][1] != dead_worker
    assert len(report.failed_attempts) == 1
    failure = report.failed_attempts[0]
    assert (failure.unit_index, failure.attempt, failure.worker_pid) == (
        1,
        1,
        dead_worker,
    )
    assert "exited with code 97" in failure.cause


def test_a_unit_that_keeps_killing_its_worker_stops_the_run_with_its_label() -> None:
    with WorkerPool(
        WorkerPoolSettings(workers=2, max_attempts=2),
        unit_function=units.always_die,
        describe_unit=lambda index, unit: f"fixture batch {index + 1}",
    ) as pool:
        with pytest.raises(
            WorkerPoolUnitError, match=r"fixture batch 1 failed on all 2 attempts"
        ) as failure:
            list(pool.map([0]))

    assert failure.value.unit_index == 0
    assert failure.value.label == "fixture batch 1"
    assert len(failure.value.causes) == 2
    assert all("exited with code 97" in cause for cause in failure.value.causes)


def test_a_worker_over_its_memory_cap_is_stopped_and_its_unit_run_again(
    tmp_path,
) -> None:
    marker = tmp_path / "held"
    cap = 512 * MIB
    with WorkerPool(
        WorkerPoolSettings(workers=1, memory_cap_bytes=cap, memory_poll_seconds=0.05),
        unit_function=units.hold_memory_on_first_attempt,
    ) as pool:
        results = list(pool.map([(7, str(marker), 1_024)]))
        report = pool.report()

    assert results[0][0] == 7
    assert results[0][1] != int(marker.read_text())
    assert len(report.failed_attempts) == 1
    assert "memory cap" in report.failed_attempts[0].cause
    assert report.peak_worker_memory_bytes > cap


def test_a_unit_that_needs_more_than_the_memory_cap_stops_the_run() -> None:
    with WorkerPool(
        WorkerPoolSettings(
            workers=1, memory_cap_bytes=512 * MIB, memory_poll_seconds=0.05
        ),
        unit_function=units.always_hold_memory,
    ) as pool:
        with pytest.raises(WorkerPoolUnitError, match="unit 1/1") as failure:
            list(pool.map([1_024]))

    assert len(failure.value.causes) == 2
    assert all("memory cap" in cause for cause in failure.value.causes)


def test_a_memory_error_runs_the_unit_again_in_a_fresh_worker(tmp_path) -> None:
    marker = tmp_path / "memory-error"
    with WorkerPool(
        WorkerPoolSettings(workers=1), unit_function=units.memory_error_on_first_attempt
    ) as pool:
        results = list(pool.map([(5, str(marker))]))
        report = pool.report()

    assert results[0][0] == 5
    assert results[0][1] != int(marker.read_text())
    assert "MemoryError: fixture allocation failed" in report.failed_attempts[0].cause


def test_a_unit_exception_stops_the_run_without_running_it_again() -> None:
    pool = WorkerPool(
        WorkerPoolSettings(workers=2), unit_function=units.refuse_unit_two
    )
    with pool:
        with pytest.raises(
            WorkerPoolUnitError, match=r"unit 3/4 .*ValueError: fixture unit 2 refuses"
        ) as failure:
            list(pool.map(range(4)))

    assert len(failure.value.causes) == 1
    assert "Traceback" in str(failure.value.__cause__)
    with pytest.raises(WorkerPoolError, match="closed"):
        list(pool.map(range(1)))


def test_a_result_the_worker_cannot_send_stops_the_run() -> None:
    with WorkerPool(
        WorkerPoolSettings(workers=1), unit_function=units.return_unpicklable
    ) as pool:
        with pytest.raises(WorkerPoolUnitError, match="unit 1/1 .*could not be sent"):
            list(pool.map([0]))


def test_an_initializer_failure_stops_the_run_before_any_unit() -> None:
    with WorkerPool(
        WorkerPoolSettings(workers=2),
        initializer=units.fail_to_start,
        unit_function=units.describe_worker,
    ) as pool:
        with pytest.raises(WorkerPoolError, match="fixture initializer refuses"):
            list(pool.map(range(3)))


def test_the_worker_environment_is_set_before_the_worker_imports_anything(
    monkeypatch,
) -> None:
    monkeypatch.delenv(units.ENVIRONMENT_VARIABLE, raising=False)
    settings = WorkerPoolSettings(
        workers=1, worker_environment={units.ENVIRONMENT_VARIABLE: "set for workers"}
    )
    with WorkerPool(settings, unit_function=units.report_environment) as pool:
        (result,) = pool.map([0])

    assert result["at_import"] == "set for workers"
    assert result["at_unit"] == "set for workers"
    assert units.ENVIRONMENT_VARIABLE not in os.environ


def test_without_a_worker_environment_workers_inherit_the_parents(monkeypatch) -> None:
    monkeypatch.setenv(units.ENVIRONMENT_VARIABLE, "the parent's value")
    with WorkerPool(
        WorkerPoolSettings(workers=1), unit_function=units.report_environment
    ) as pool:
        (result,) = pool.map([0])

    assert result["at_import"] == "the parent's value"
    assert os.environ[units.ENVIRONMENT_VARIABLE] == "the parent's value"


def test_closing_an_unfinished_map_stops_every_worker() -> None:
    pool = WorkerPool(
        WorkerPoolSettings(workers=2), unit_function=units.sleep_then_return
    )
    results = pool.map([(index, 0.3) for index in range(6)])
    assert next(results) == 0
    results.close()

    assert multiprocessing.active_children() == []
    with pytest.raises(WorkerPoolError, match="closed"):
        list(pool.map([(0, 0.0)]))


@pytest.mark.parametrize(
    "setting",
    [
        {"workers": 0},
        {"max_units_per_worker": 0},
        {"max_attempts": 0},
        {"memory_cap_bytes": 0},
        {"max_units_ahead": 0},
        {"memory_poll_seconds": 0.0},
        # Longer than a thread can wait for, which would end the memory thread.
        {"memory_poll_seconds": 1e10},
    ],
)
def test_settings_refuse_values_that_cannot_run(setting) -> None:
    with pytest.raises(ValueError, match=next(iter(setting))):
        WorkerPoolSettings(**{"workers": 2, **setting})


def test_functions_a_worker_cannot_import_are_refused_before_any_worker_starts() -> (
    None
):
    with pytest.raises(TypeError, match="importable"):
        WorkerPool(
            WorkerPoolSettings(workers=1), unit_function=lambda state, unit: unit
        )
    with pytest.raises(TypeError, match="importable"):
        WorkerPool(
            WorkerPoolSettings(workers=1),
            initializer=lambda: None,
            unit_function=units.describe_worker,
        )


def test_process_memory_bytes_sees_this_process_grow() -> None:
    before = process_memory_bytes(os.getpid())
    held = np.ones(256 * MIB // 8)
    after = process_memory_bytes(os.getpid())
    assert before is not None and before > 10 * MIB
    assert after - before >= 200 * MIB
    del held


def _held_by_another_thread(lock) -> bool:
    """Whether a thread other than a fresh one holds ``lock`` (an RLock)."""
    acquired = []

    def try_to_acquire() -> None:
        got = lock.acquire(timeout=0.05)
        acquired.append(got)
        if got:
            lock.release()

    thread = threading.Thread(target=try_to_acquire)
    thread.start()
    thread.join()
    return not acquired[0]


def test_the_memory_cap_is_enforced_while_another_worker_starts(monkeypatch) -> None:
    """Starting a worker can block for as long as the worker takes to import.

    The memory thread must still be able to stop a worker over its cap, so a
    start must not hold the lock that thread needs.
    """
    pool = WorkerPool(WorkerPoolSettings(workers=1), unit_function=units.echo)
    starts_under_lock = []
    real_environment_for_start = worker_pool_module._environment_for_start

    @contextlib.contextmanager
    def recording_environment_for_start(environment):
        # Entered around ``process.start()`` in the pool's worker start.
        starts_under_lock.append(_held_by_another_thread(pool._shared.lock))
        with real_environment_for_start(environment):
            yield

    monkeypatch.setattr(
        worker_pool_module, "_environment_for_start", recording_environment_for_start
    )
    with pool:
        list(pool.map([0]))
    assert starts_under_lock == [False]


def test_resuming_a_map_after_the_pool_closed_raises_and_starts_nothing() -> None:
    pool = WorkerPool(
        WorkerPoolSettings(workers=2), unit_function=units.sleep_then_return
    )
    results = pool.map([(index, 0.3) for index in range(6)])
    assert next(results) == 0
    pool.close()
    starts = pool.report().worker_starts

    with pytest.raises(WorkerPoolError, match="closed"):
        next(results)
    assert pool.report().worker_starts == starts
    assert multiprocessing.active_children() == []


def test_a_worker_that_will_not_exit_after_being_stopped_is_killed(monkeypatch) -> None:
    monkeypatch.setattr(worker_pool_module, "_STOP_SECONDS", 1.0)
    started = time.monotonic()
    with WorkerPool(
        WorkerPoolSettings(workers=1, max_units_per_worker=1),
        initializer=units.start_lingering_thread,
        unit_function=units.echo,
    ) as pool:
        results = list(pool.map(range(2)))

    assert results == [0, 1]
    assert time.monotonic() - started < 60


def _unclosed_run() -> list:
    pool = WorkerPool(WorkerPoolSettings(workers=2), unit_function=units.echo)
    return list(pool.map(range(2)))


def test_a_pool_dropped_without_closing_stops_its_workers() -> None:
    assert len(_unclosed_run()) == 2
    gc.collect()
    deadline = time.monotonic() + 30
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert multiprocessing.active_children() == []


def _process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/stat")
    if status.exists():
        # An exited process its new parent has not reaped yet is a zombie.
        return status.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    return True


def test_busy_workers_exit_when_their_parent_is_killed(tmp_path) -> None:
    repository = paths_for("microcosm-build").repository
    environment = {**os.environ, "PYTHONPATH": str(repository)}
    parent = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "test_support.microcosm_build.worker_pool_orphan_parent",
            str(tmp_path),
        ],
        cwd=repository,
        env=environment,
    )
    try:
        deadline = time.monotonic() + 90
        while len(list(tmp_path.glob("worker-*"))) < 2:
            assert parent.poll() is None, "the parent process exited early"
            assert time.monotonic() < deadline, "the workers did not start"
            time.sleep(0.2)
        worker_pids = [
            int(path.name.split("-")[1]) for path in tmp_path.glob("worker-*")
        ]
    finally:
        parent.kill()
        parent.wait()

    deadline = time.monotonic() + 20
    while (
        any(_process_running(pid) for pid in worker_pids)
        and time.monotonic() < deadline
    ):
        time.sleep(0.2)
    still_running = [pid for pid in worker_pids if _process_running(pid)]
    for pid in still_running:
        os.kill(pid, signal.SIGKILL)
    assert still_running == []


def test_taking_every_result_finishes_the_map_without_exhausting_it() -> None:
    with WorkerPool(WorkerPoolSettings(workers=2), unit_function=units.echo) as pool:
        results = pool.map(range(3))
        assert [next(results) for _ in range(3)] == [0, 1, 2]
        results.close()
        assert list(pool.map(range(2))) == [0, 1]


def test_a_unit_the_worker_cannot_unpickle_stops_the_run_with_its_cause() -> None:
    with WorkerPool(
        WorkerPoolSettings(workers=1, max_attempts=3),
        unit_function=units.echo,
    ) as pool:
        with pytest.raises(
            WorkerPoolUnitError, match=r"unit 1/1 .*ValueError: this object cannot"
        ) as failure:
            list(pool.map([units.RefusesToUnpickle()]))
        assert pool.report().worker_starts == 1
    assert len(failure.value.causes) == 1


def test_a_result_the_parent_cannot_unpickle_stops_the_run_with_its_label() -> None:
    with WorkerPool(
        WorkerPoolSettings(workers=1),
        unit_function=units.return_object_the_parent_cannot_unpickle,
    ) as pool:
        with pytest.raises(
            WorkerPoolUnitError, match=r"unit 1/1 .*could not be read in the parent"
        ):
            list(pool.map([0]))


def test_memory_a_unit_leaves_behind_is_not_charged_to_the_next_unit() -> None:
    # The memory thread's first reading comes long after this test ends, so
    # only the reading taken when unit 0 returns can see the 600 MiB.
    settings = WorkerPoolSettings(
        workers=1,
        memory_cap_bytes=400 * MIB,
        memory_poll_seconds=300.0,
        max_attempts=1,
    )
    with WorkerPool(settings, unit_function=units.keep_memory_after_returning) as pool:
        assert list(pool.map([(0, 600), (1, 0)])) == [0, 1]
        report = pool.report()

    assert report.failed_attempts == ()
    assert report.workers_stopped_over_cap == 1


def test_a_worker_environment_value_that_cannot_be_encoded_is_refused() -> None:
    with pytest.raises(ValueError, match="worker_environment"):
        WorkerPoolSettings(
            workers=1,
            worker_environment={
                "MICROCOSM_TEST_OK": "1",
                "MICROCOSM_TEST_BAD": "\ud800",
            },
        )


def test_a_dead_worker_is_noticed_when_something_else_collects_children() -> None:
    previous = signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    try:
        with WorkerPool(
            WorkerPoolSettings(workers=1, max_attempts=2),
            unit_function=units.always_die,
        ) as pool:
            with pytest.raises(WorkerPoolUnitError, match="unit 1/1 failed on all 2"):
                list(pool.map([0]))
    finally:
        signal.signal(signal.SIGCHLD, previous)


def test_a_function_defined_where_workers_cannot_import_it_is_refused(
    monkeypatch,
) -> None:
    script_main = ModuleType("__main__")  # As under ``python -c`` or a REPL.

    def typed_at_the_prompt(state, unit):
        return unit

    typed_at_the_prompt.__module__ = "__main__"
    typed_at_the_prompt.__qualname__ = "typed_at_the_prompt"
    script_main.typed_at_the_prompt = typed_at_the_prompt
    monkeypatch.setitem(sys.modules, "__main__", script_main)

    with pytest.raises(TypeError, match="importable"):
        WorkerPool(WorkerPoolSettings(workers=1), unit_function=typed_at_the_prompt)


def test_a_finished_maps_generator_cannot_release_a_running_maps_turn() -> None:
    with WorkerPool(
        WorkerPoolSettings(workers=2), unit_function=units.sleep_then_return
    ) as pool:
        first = pool.map([(0, 0.0)])
        assert next(first) == 0  # Its last result: the map is over.
        second = pool.map([(10, 0.0), (11, 2.0)])
        assert next(second) == 10
        first.close()  # Must not end the second map's turn.
        with pytest.raises(WorkerPoolError, match="already running"):
            next(pool.map([(20, 0.0)]))
        assert list(second) == [11]


def test_closing_from_inside_the_memory_thread_stops_every_worker(monkeypatch) -> None:
    """A garbage collection can run in any thread, including the memory thread
    while it holds the pool's lock, and close an abandoned map there."""
    pool = WorkerPool(
        WorkerPoolSettings(workers=2, memory_poll_seconds=0.05),
        unit_function=units.sleep_then_return,
    )
    armed = threading.Event()
    outcome: dict[str, str] = {}
    real_has_exited = worker_pool_module._has_exited

    def has_exited(worker):
        # Called by the memory thread with the pool's lock held.
        if (
            armed.is_set()
            and threading.current_thread().name == "worker-pool-memory"
            and "close" not in outcome
        ):
            outcome["close"] = "started"
            try:
                pool.close()
                outcome["close"] = "finished"
            except BaseException as error:
                outcome["close"] = repr(error)
        return real_has_exited(worker)

    monkeypatch.setattr(worker_pool_module, "_has_exited", has_exited)
    results = pool.map([(0, 0.0), (1, 30.0)])
    assert next(results) == 0
    armed.set()
    deadline = time.monotonic() + 30
    while outcome.get("close") in (None, "started") and time.monotonic() < deadline:
        time.sleep(0.05)

    assert outcome.get("close") == "finished"
    assert multiprocessing.active_children() == []
    with pytest.raises(WorkerPoolError, match="closed"):
        next(results)


def test_a_stale_memory_reading_does_not_kill_a_worker_on_a_later_unit(
    monkeypatch, tmp_path
) -> None:
    marker = tmp_path / "unit-0-started"
    real_reading = worker_pool_module.process_memory_bytes
    stale = threading.Event()

    def reading(pid):
        if (
            threading.current_thread().name == "worker-pool-memory"
            and marker.exists()
            and not stale.is_set()
        ):
            # Read during unit 0, acted on after unit 1 has started.
            stale.set()
            time.sleep(1.5)
            return 10 * 1024**3
        return real_reading(pid)

    monkeypatch.setattr(worker_pool_module, "process_memory_bytes", reading)
    settings = WorkerPoolSettings(
        workers=1,
        memory_cap_bytes=400 * MIB,
        memory_poll_seconds=0.05,
        max_attempts=1,
    )
    with WorkerPool(settings, unit_function=units.mark_then_sleep) as pool:
        assert list(pool.map([(0, str(marker), 0.3), (1, "", 3.0)])) == [0, 1]
        assert pool.report().failed_attempts == ()


def test_workers_exchange_messages_over_pipes() -> None:
    """Socket pairs carry large results several times slower on macOS."""
    with WorkerPool(WorkerPoolSettings(workers=1), unit_function=units.echo) as pool:
        assert list(pool.map([0])) == [0]
        (worker,) = pool._shared.workers
        for end in (worker.to_worker, worker.from_worker):
            assert stat.S_ISFIFO(os.fstat(end.fileno()).st_mode)


def test_an_idle_worker_gets_its_next_unit_while_the_caller_holds_a_result() -> None:
    with WorkerPool(
        WorkerPoolSettings(workers=1), unit_function=units.record_run_times
    ) as pool:
        results = pool.map([(0, 0.0), (1, 0.0)])
        next(results)
        time.sleep(1.0)  # The caller works on result 0.
        resumed = time.monotonic()
        (_, unit_one_started, _) = next(results)

    assert unit_one_started < resumed


def test_a_result_already_handed_back_is_not_kept_while_map_waits() -> None:
    with WorkerPool(
        WorkerPoolSettings(workers=1), unit_function=units.array_after_sleep
    ) as pool:
        results = pool.map([0.0, 2.0])
        first = next(results)
        reference = weakref.ref(first)
        del first
        observed: dict[str, bool] = {}
        timer = threading.Timer(
            1.0, lambda: observed.setdefault("alive", reference() is not None)
        )
        timer.start()
        next(results)
        timer.join()

    assert observed == {"alive": False}


def test_a_stopped_worker_is_killed_at_its_deadline_between_maps(monkeypatch) -> None:
    monkeypatch.setattr(worker_pool_module, "_STOP_SECONDS", 1.0)
    with WorkerPool(
        WorkerPoolSettings(workers=1, max_units_per_worker=1, memory_poll_seconds=0.1),
        initializer=units.start_lingering_thread,
        unit_function=units.echo,
    ) as pool:
        assert list(pool.map([0])) == [0]
        (worker,) = pool._shared.workers  # Told to stop; kept alive by a thread.
        deadline = time.monotonic() + 20
        while (
            not worker_pool_module._has_exited(worker) and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        assert worker_pool_module._has_exited(worker)


def test_a_kill_the_operating_system_refuses_does_not_raise(monkeypatch) -> None:
    def refuse(pid, signal_number):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(worker_pool_module.os, "kill", refuse)
    read_end, write_end = os.pipe()
    try:
        worker = SimpleNamespace(
            exited=False,
            pidfd=None,
            pid=999_999,
            process=SimpleNamespace(sentinel=read_end),
        )
        worker_pool_module._kill(worker)
    finally:
        os.close(read_end)
        os.close(write_end)


# ---------------------------------------------------------------------------
# The 1-vs-N byte-identity helper
# ---------------------------------------------------------------------------

_HOUSEHOLDS = 5_003
_BATCH_SIZE = 500


def _household_arrays():
    generator = np.random.default_rng(20261125)
    values = generator.lognormal(10.0, 1.5, _HOUSEHOLDS)
    weights = generator.uniform(0.5, 3.0, _HOUSEHOLDS)
    return values, weights


def _household_batches():
    return [
        (start, min(start + _BATCH_SIZE, _HOUSEHOLDS))
        for start in range(0, _HOUSEHOLDS, _BATCH_SIZE)
    ]


def _join_in_batch_order(parts):
    """Concatenate per-household arrays and sum totals in batch order."""
    total = np.float64(parts[0]["weighted_total"])
    for part in parts[1:]:
        total = np.add(total, part["weighted_total"])
    return {
        "per_household": np.concatenate([part["per_household"] for part in parts]),
        "weighted_total": total,
        "batches": len(parts),
    }


def _pooled_household_run(workers: int, *, max_units_per_worker=None):
    with WorkerPool(
        WorkerPoolSettings(workers=workers, max_units_per_worker=max_units_per_worker),
        initializer=units.initialize_household_arrays,
        initargs=_household_arrays(),
        unit_function=units.score_household_batch,
    ) as pool:
        return _join_in_batch_order(list(pool.map(_household_batches())))


def test_one_and_several_workers_give_byte_identical_results() -> None:
    pooled = assert_identical_across_worker_counts(
        _pooled_household_run, worker_counts=(1, 3)
    )
    state = units.initialize_household_arrays(*_household_arrays())
    in_process = _join_in_batch_order(
        [units.score_household_batch(state, batch) for batch in _household_batches()]
    )

    assert pooled["batches"] == 11
    assert canonical_result_bytes(pooled) == canonical_result_bytes(in_process)


def test_replacing_workers_after_every_unit_keeps_results_byte_identical() -> None:
    assert_identical_across_worker_counts(
        lambda workers: _pooled_household_run(workers, max_units_per_worker=1),
        worker_counts=(1, 4),
    )


def test_the_identity_helper_names_the_first_difference() -> None:
    def run(workers: int):
        return {
            "label": "same",
            "totals": np.array([1.0, 0.0 if workers == 1 else -0.0]),
        }

    with pytest.raises(AssertionError, match=r"workers=3 .*\['totals'\]"):
        assert_identical_across_worker_counts(run, worker_counts=(1, 3))


def test_canonical_bytes_tell_apart_what_equality_hides() -> None:
    assert canonical_result_bytes(np.float64(0.0)) != canonical_result_bytes(
        np.float64(-0.0)
    )
    assert canonical_result_bytes(np.array([np.nan])) != canonical_result_bytes(
        -np.array([np.nan])
    )
    assert canonical_result_bytes(
        np.array([1, 2], dtype=np.int64)
    ) != canonical_result_bytes(np.array([1, 2], dtype=np.int32))
    assert canonical_result_bytes(np.zeros((2, 3))) != canonical_result_bytes(
        np.zeros((3, 2))
    )
    assert canonical_result_bytes({"a": 1, "b": 2}) != canonical_result_bytes(
        {"b": 2, "a": 1}
    )
    assert canonical_result_bytes(1.0) != canonical_result_bytes(1)
    assert canonical_result_bytes((1, 2)) != canonical_result_bytes([1, 2])
    with pytest.raises(TypeError, match="object"):
        canonical_result_bytes(object())


def test_canonical_bytes_keep_field_types_array_kinds_and_pandas_metadata() -> None:
    import enum

    import pandas as pd

    little_endian = np.array([(1.0,)], dtype=[("a", "<f8")])
    assert canonical_result_bytes(little_endian) != canonical_result_bytes(
        little_endian.view([("a", ">f8")])
    )
    assert canonical_result_bytes(little_endian) != canonical_result_bytes(
        little_endian.view([("b", "<f8")])
    )
    assert canonical_result_bytes(
        np.ma.array([1.0, 2.0], mask=[False, True])
    ) != canonical_result_bytes(np.ma.array([1.0, 2.0], mask=[True, False]))
    assert canonical_result_bytes(np.ma.array([1.0, 2.0])) != canonical_result_bytes(
        np.array([1.0, 2.0])
    )
    assert canonical_result_bytes(np.True_) != canonical_result_bytes(True)

    class Size(enum.IntEnum):
        ONE = 1

    assert canonical_result_bytes(Size.ONE) != canonical_result_bytes(1)
    assert canonical_result_bytes(b"ab") != canonical_result_bytes(bytearray(b"ab"))
    assert canonical_result_bytes(
        pd.Series(["a"], dtype=pd.CategoricalDtype(["a", "b"]))
    ) != canonical_result_bytes(pd.Series(["a"], dtype=pd.CategoricalDtype(["a", "c"])))
    assert canonical_result_bytes(
        pd.Series([1.0], index=pd.Index([0], name="household"))
    ) != canonical_result_bytes(pd.Series([1.0], index=pd.Index([0], name="person")))
