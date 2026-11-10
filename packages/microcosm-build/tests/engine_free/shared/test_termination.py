"""SIGTERM becomes an exception the build's own failure path records."""

from __future__ import annotations

import os
import signal
import threading

import pytest

from microcosm.build.run_outcome import RunOutcome, classify_failure
from microcosm.build.termination import BuildTerminatedError, raise_on_sigterm


def test_sigterm_inside_the_block_raises_and_restores_the_previous_handler():
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(BuildTerminatedError) as raised:
        with raise_on_sigterm():
            os.kill(os.getpid(), signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == previous
    assert raised.value.exit_code == 143
    assert isinstance(raised.value, KeyboardInterrupt)
    assert classify_failure(raised.value).outcome is RunOutcome.TERMINATED


def test_the_handler_fires_once_then_leaves_the_default():
    # ``os.kill`` runs the handler before it returns when the target is this
    # process, so the error surfaces from the call itself.
    with raise_on_sigterm():
        with pytest.raises(BuildTerminatedError):
            os.kill(os.getpid(), signal.SIGTERM)
        # Still inside the block, after the first signal: a second one would
        # now kill the process (the supervisor's escalation).
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


def test_a_block_without_a_signal_leaves_the_handler_as_it_was():
    previous = signal.getsignal(signal.SIGTERM)
    with raise_on_sigterm():
        assert signal.getsignal(signal.SIGTERM) not in (previous, signal.SIG_DFL)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_off_the_main_thread_it_installs_nothing():
    previous = signal.getsignal(signal.SIGTERM)
    seen: list[object] = []

    def worker():
        with raise_on_sigterm():
            seen.append(signal.getsignal(signal.SIGTERM))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen == [previous]
