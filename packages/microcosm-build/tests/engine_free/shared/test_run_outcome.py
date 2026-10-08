"""How a build's end is classified once for staging, telemetry and the Logbook."""

from __future__ import annotations

from pathlib import Path

import pytest

from microcosm.build.gate_battery import GateBatteryBlockedError
from microcosm.build.run_outcome import (
    BuildRefusedError,
    GateBlock,
    RunOutcome,
    classify_failure,
    classify_return,
    logbook_disposition,
)
from microcosm.graph.errors import NodeRejectedError


class SpineGateBlockedError(ValueError):
    """A stand-in matched by name, as the country-agnostic classifier does."""

    def __init__(self, phase: str, blocking_gate_ids: tuple[str, ...]) -> None:
        self.phase = phase
        self.blocking_gate_ids = blocking_gate_ids
        super().__init__(f"Stored {phase} spine gates block downstream execution")


class BuildTerminatedError(KeyboardInterrupt):
    """A stand-in for :mod:`microcosm.build.termination`'s error, matched by name."""


def _wrapped(error: BaseException, wrapper: BaseException) -> BaseException:
    try:
        try:
            raise error
        except BaseException as inner:
            raise wrapper from inner
    except BaseException as outer:
        return outer


def test_a_gate_battery_block_is_blocked_with_its_phase_and_gate_ids():
    error = GateBatteryBlockedError(
        "terminal",
        ["[gate_a] missed", "[gate_b] missed"],
        Path("gates.json"),
        blocking_gate_ids=("gate_a", "gate_b"),
    )
    classified = classify_failure(error)
    assert classified.outcome is RunOutcome.BLOCKED
    assert classified.block == GateBlock("terminal", ("gate_a", "gate_b"), 2)
    assert (classified.error_code, classified.failure_class) == (None, None)
    assert classified.disposition == "failed"


def test_gate_ids_are_read_from_the_failure_lines_when_the_raiser_named_none():
    error = GateBatteryBlockedError("assembled", ["[gate_c] missed"], Path("g.json"))
    assert classify_failure(error).block == GateBlock("assembled", ("gate_c",), 1)


def test_a_spine_block_wrapped_by_the_graph_executor_is_still_blocked():
    error = _wrapped(
        SpineGateBlockedError("transferred", ("spine_gate",)),
        NodeRejectedError("uk.spine.donors rejected"),
    )
    classified = classify_failure(error)
    assert classified.outcome is RunOutcome.BLOCKED
    assert classified.block == GateBlock("transferred", ("spine_gate",), 1)


@pytest.mark.parametrize(
    ("error", "outcome", "code", "failure_class", "disposition"),
    [
        (
            KeyboardInterrupt(),
            RunOutcome.INTERRUPTED,
            "INTERRUPTED",
            "interrupted",
            "discarded",
        ),
        (
            BuildTerminatedError(),
            RunOutcome.TERMINATED,
            "TERMINATED",
            "terminated",
            "discarded",
        ),
        (MemoryError(), RunOutcome.FAILED, "OUT_OF_MEMORY", "out_of_memory", "failed"),
        (
            _wrapped(MemoryError(), RuntimeError("solve failed")),
            RunOutcome.FAILED,
            "OUT_OF_MEMORY",
            "out_of_memory",
            "failed",
        ),
        (
            NodeRejectedError("bad receipt"),
            RunOutcome.FAILED,
            "GRAPH_NODE_FAILED",
            "error",
            "failed",
        ),
        (
            BuildRefusedError("refused"),
            RunOutcome.FAILED,
            "BUILD_REFUSED",
            "refused",
            "failed",
        ),
        (
            ValueError("anything else"),
            RunOutcome.FAILED,
            "BUILD_FAILED",
            "error",
            "failed",
        ),
    ],
)
def test_raised_errors_get_distinct_codes_classes_and_dispositions(
    error, outcome, code, failure_class, disposition
):
    classified = classify_failure(error)
    assert classified.outcome is outcome
    assert (classified.error_code, classified.failure_class) == (code, failure_class)
    assert classified.block is None
    assert classified.disposition == disposition


def test_a_return_is_completed_only_when_it_passed_without_a_block():
    block = GateBlock.of("preflight", ["gate_a"])
    assert classify_return(0, None).outcome is RunOutcome.COMPLETED
    assert classify_return(0, None).disposition == "iterating"
    assert classify_return(1, block).outcome is RunOutcome.BLOCKED
    assert classify_return(1, block).block == block
    refused = classify_return(1, None)
    assert (refused.outcome, refused.error_code, refused.failure_class) == (
        RunOutcome.FAILED,
        "BUILD_REFUSED",
        "refused",
    )


def test_every_outcome_has_a_logbook_disposition():
    assert {outcome: logbook_disposition(outcome) for outcome in RunOutcome} == {
        RunOutcome.COMPLETED: "iterating",
        RunOutcome.BLOCKED: "failed",
        RunOutcome.FAILED: "failed",
        RunOutcome.INTERRUPTED: "discarded",
        RunOutcome.TERMINATED: "discarded",
        RunOutcome.ABORTED: "discarded",
    }


def test_a_gate_block_counts_at_least_one_failure():
    assert (
        GateBlock.of("terminal", [], blocking_failure_count=0).blocking_failure_count
        == 1
    )
