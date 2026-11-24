"""Initializers and unit functions that spawned worker-pool workers import by name.

Spawned workers start a fresh interpreter, so everything a test hands the pool
must be importable from a module, not defined inside a test.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

ENVIRONMENT_VARIABLE = "MICROCOSM_TEST_WORKER_POOL_VARIABLE"

#: Read when a worker imports this module, before any pool code runs a unit.
ENVIRONMENT_AT_IMPORT = os.environ.get(ENVIRONMENT_VARIABLE)

_INITIALIZER_CALLS = 0


def initialize_with_payload(payload):
    """Count this process's initializer calls and keep the payload."""
    global _INITIALIZER_CALLS
    _INITIALIZER_CALLS += 1
    return {"payload": payload, "initializer_calls": _INITIALIZER_CALLS}


def echo(state, unit):
    """Return the unit; needs no initializer state."""
    return unit


def describe_worker(state, unit):
    return {
        "unit": unit,
        "pid": os.getpid(),
        "initializer_calls": state["initializer_calls"],
        "payload_total": float(np.sum(state["payload"])),
    }


def sleep_then_return(state, unit):
    value, seconds = unit
    time.sleep(seconds)
    return value


def record_run_times(state, unit):
    """Sleep, then report when this unit started and ended in the worker."""
    value, seconds = unit
    started = time.monotonic()
    time.sleep(seconds)
    return value, started, time.monotonic()


def _first_attempt(marker: str) -> bool:
    path = Path(marker)
    if path.exists():
        return False
    path.write_text(str(os.getpid()))
    return True


def die_on_first_attempt(state, unit):
    value, marker = unit
    if _first_attempt(marker):
        os._exit(97)
    return value, os.getpid()


def always_die(state, unit):
    os._exit(97)


def hold_memory_on_first_attempt(state, unit):
    value, marker, mebibytes = unit
    if _first_attempt(marker):
        held = np.ones(mebibytes * 1024 * 1024 // 8)
        # The pool stops this worker for its memory long before this returns.
        time.sleep(60)
        return float(held[0])
    return value, os.getpid()


def always_hold_memory(state, unit):
    mebibytes = unit
    held = np.ones(mebibytes * 1024 * 1024 // 8)
    time.sleep(60)
    return float(held[0])


def memory_error_on_first_attempt(state, unit):
    value, marker = unit
    if _first_attempt(marker):
        raise MemoryError("fixture allocation failed")
    return value, os.getpid()


def refuse_unit_two(state, unit):
    if unit == 2:
        raise ValueError(f"fixture unit {unit} refuses")
    return unit


def return_unpicklable(state, unit):
    return lambda: unit


def fail_to_start(*args):
    raise RuntimeError("fixture initializer refuses")


def report_environment(state, unit):
    return {
        "at_import": ENVIRONMENT_AT_IMPORT,
        "at_unit": os.environ.get(ENVIRONMENT_VARIABLE),
        "pid": os.getpid(),
    }


def initialize_household_arrays(values, weights):
    """Stand-in for a worker's frame: whole-population arrays sent once."""
    return {"values": values, "weights": weights}


def score_household_batch(state, positions):
    """Per-household and summed results for one contiguous household batch."""
    start, stop = positions
    values = state["values"][start:stop]
    weights = state["weights"][start:stop]
    per_household = np.power(values, 1.5) * weights - np.sqrt(values)
    return {
        "per_household": per_household,
        "weighted_total": np.sum(per_household * weights, dtype=np.float64),
    }


def start_lingering_thread(*args):
    """An initializer whose worker keeps a non-daemon thread after it returns.

    multiprocessing waits for such a thread before the process exits, so a
    worker told to stop stays alive until the pool kills it.
    """
    import threading

    threading.Thread(target=time.sleep, args=(120,), daemon=False).start()


def _refuse_to_load():
    raise ValueError("this object cannot be rebuilt after pickling")


class RefusesToUnpickle:
    """Pickles fine, but raises when it is unpickled."""

    def __reduce__(self):
        return (_refuse_to_load, ())


def return_object_the_parent_cannot_unpickle(state, unit):
    return RefusesToUnpickle()


_KEPT: list = []


def keep_memory_after_returning(state, unit):
    """Return at once, but keep ``mebibytes`` alive in this worker."""
    value, mebibytes = unit
    if mebibytes:
        _KEPT.append(np.ones(mebibytes * 1024 * 1024 // 8))
    else:
        time.sleep(0.5)
    return value


def write_pid_then_sleep(state, unit):
    directory, seconds = unit
    (Path(directory) / f"worker-{os.getpid()}").write_text("running")
    time.sleep(seconds)
    return os.getpid()
