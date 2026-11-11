"""How a build ended, decided once and copied into every record.

The staging run bundle, the hosted telemetry and the Logbook row each describe a
build's end in their own vocabulary. This module is the single place that maps one
build's end onto all three, so they cannot disagree:

* a build whose gates refused its candidate is ``blocked`` (a staging status of its
  own since contract version 3), wherever the refusal surfaced — a phase report the
  build checked itself, or a :class:`GateBatteryBlockedError` /
  ``SpineGateBlockedError`` raised (possibly wrapped by the graph executor);
* a build that raised is ``failed`` with an error code and a failure class drawn
  from the labels the calibration dashboard renders;
* the Logbook keeps its own vocabulary, derived here by :func:`logbook_disposition`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from microcosm.build.gate_battery import GateBatteryBlockedError


class BuildRefusedError(RuntimeError):
    """A build returned a non-zero status without raising and without a gate block.

    No such path exists today (every refusal is a recorded gate block); this is the
    guard that keeps a future one from closing as ``completed``.
    """


class RunOutcome(Enum):
    """How a build attempt ended."""

    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    TERMINATED = "terminated"
    ABORTED = "aborted"


#: The Logbook disposition each outcome records. A blocked candidate is a failed
#: attempt to the Logbook; an operator stop or a rung abort discards the attempt.
_LOGBOOK_DISPOSITIONS = {
    RunOutcome.COMPLETED: "iterating",
    RunOutcome.BLOCKED: "failed",
    RunOutcome.FAILED: "failed",
    RunOutcome.INTERRUPTED: "discarded",
    RunOutcome.TERMINATED: "discarded",
    RunOutcome.ABORTED: "discarded",
}


def logbook_disposition(outcome: RunOutcome) -> str:
    """The Logbook ``disposition`` for one outcome."""

    return _LOGBOOK_DISPOSITIONS[outcome]


@dataclass(frozen=True)
class GateBlock:
    """Which gate phase refused the candidate, and which gates."""

    phase: str
    #: The gates that refused; empty when the raiser named none (the count
    #: still says how many failures there were). Never a placeholder id.
    blocking_gate_ids: tuple[str, ...]
    blocking_failure_count: int
    #: Every gate's status in the refusing report, when that report could be
    #: read; ``None`` otherwise.
    gate_statuses: Mapping[str, str] | None = None

    @classmethod
    def of(
        cls,
        phase: str,
        blocking_gate_ids: Sequence[str],
        *,
        blocking_failure_count: int | None = None,
        gate_statuses: Mapping[str, str] | None = None,
    ) -> GateBlock:
        ids = tuple(str(gate_id) for gate_id in blocking_gate_ids)
        count = len(ids) if blocking_failure_count is None else blocking_failure_count
        return cls(
            phase=str(phase),
            blocking_gate_ids=ids,
            blocking_failure_count=max(1, int(count)),
            gate_statuses=None
            if gate_statuses is None
            else {
                str(gate_id): str(status) for gate_id, status in gate_statuses.items()
            },
        )


@dataclass(frozen=True)
class Classified:
    """One build's end, ready for the staging bundle, the emitter and the Logbook."""

    outcome: RunOutcome
    error_code: str | None = None
    failure_class: str | None = None
    block: GateBlock | None = None

    @property
    def disposition(self) -> str:
        return logbook_disposition(self.outcome)


#: A run whose gate block could not be recorded (the staging contract refused
#: the block's details) closes ``failed`` with this code and class instead, so no
#: path leaves a run ``running``.
UNRECORDED_GATE_BLOCK = Classified(
    RunOutcome.FAILED, "GATE_BLOCK_UNRECORDED", "unrecorded_gate_block"
)


def _cause_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _written_gate_statuses(report_path: object) -> dict[str, str] | None:
    """The gate statuses a battery wrote beside its block, if the file reads."""

    try:
        gates = json.loads(Path(report_path).read_text(encoding="utf-8")).get("gates")
    except (OSError, TypeError, ValueError, AttributeError):
        return None
    if not isinstance(gates, Mapping):
        return None
    return {
        str(gate_id): str(entry.get("status"))
        for gate_id, entry in gates.items()
        if isinstance(entry, Mapping)
    }


def _phase_report_gate_statuses(report: object) -> dict[str, str] | None:
    """The gate statuses of an in-memory phase report (a spine refusal's)."""

    outcomes = getattr(report, "outcomes", None)
    if outcomes is None:
        return None
    try:
        return {
            str(outcome.entry.id): str(getattr(outcome.status, "value", outcome.status))
            for outcome in outcomes
        }
    except AttributeError:
        return None


def _gate_block(error: BaseException) -> GateBlock | None:
    """A gate refusal anywhere in the cause chain, as a :class:`GateBlock`."""

    for link in _cause_chain(error):
        if isinstance(link, GateBatteryBlockedError):
            ids = link.blocking_gate_ids or tuple(
                line.split("]", 1)[0].lstrip("[")
                for line in link.failures
                if line.startswith("[") and "]" in line
            )
            return GateBlock.of(
                link.phase,
                ids,
                blocking_failure_count=max(len(link.failures), len(ids), 1),
                gate_statuses=_written_gate_statuses(link.report_path),
            )
        # SpineGateBlockedError lives in the UK runtime; match it structurally so
        # this module stays country-agnostic.
        if type(link).__name__ == "SpineGateBlockedError":
            ids = tuple(getattr(link, "blocking_gate_ids", ()) or ())
            report = getattr(link, "report", None)
            phase = getattr(link, "phase", None) or getattr(report, "phase", "spine")
            return GateBlock.of(
                phase, ids, gate_statuses=_phase_report_gate_statuses(report)
            )
    return None


def classify_failure(error: BaseException) -> Classified:
    """Classify a raised build error: a gate block, an operator stop, or a failure."""

    block = _gate_block(error)
    if block is not None:
        return Classified(RunOutcome.BLOCKED, block=block)
    for link in _cause_chain(error):
        if type(link).__name__ == "BuildTerminatedError":
            return Classified(RunOutcome.TERMINATED, "TERMINATED", "terminated")
    for link in _cause_chain(error):
        if isinstance(link, KeyboardInterrupt):
            return Classified(RunOutcome.INTERRUPTED, "INTERRUPTED", "interrupted")
    if isinstance(error, BuildRefusedError):
        return Classified(RunOutcome.FAILED, "BUILD_REFUSED", "refused")
    for link in _cause_chain(error):
        if isinstance(link, MemoryError):
            return Classified(RunOutcome.FAILED, "OUT_OF_MEMORY", "out_of_memory")
    for link in _cause_chain(error):
        if type(link).__name__ in {"NodeRejectedError", "NodeRejected"}:
            return Classified(RunOutcome.FAILED, "GRAPH_NODE_FAILED", "error")
    return Classified(RunOutcome.FAILED, "BUILD_FAILED", "error")


def classify_return(status: int, block: GateBlock | None) -> Classified:
    """Classify a build that returned rather than raised.

    A non-zero return with no recorded block is a failure: no path that refuses a
    candidate may close as ``completed``.
    """

    if block is not None:
        return Classified(RunOutcome.BLOCKED, block=block)
    if status == 0:
        return Classified(RunOutcome.COMPLETED)
    return Classified(RunOutcome.FAILED, "BUILD_REFUSED", "refused")
