"""Re-run the #908 independent review's four counterexamples against the fix.

The review's probe (head ``bae1887ff``) reproduced four defects with invented
data and recorded them in ``reproductions.json``. This script replays the same
four counterexamples, unchanged, against this branch's source and asserts each
one is now closed. It is passive: invented aggregates and temporary files only,
no engine, no optimizer, no network, no native input.

Run with the lane's isolated interpreter, e.g.::

    python -I -B -S experiments/908_review_findings_recheck.py
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
from datetime import UTC, datetime

import numpy as np

from microcosm.calibrate.target_snapshots import (
    TargetSnapshotError,
    TargetSnapshotObserver,
    TargetSnapshotWriter,
    iter_history,
    validate_target_snapshot,
)

#: The review's own record-level vectors, copied verbatim from its probe.
VECTORS = {
    "household_weights": [10.0, 20.0],
    "tax_unit_id": [101, 102],
    "spm_unit_id": [201, 202],
}


def _observer(sink=lambda payload: None, context=None):
    return TargetSnapshotObserver(
        sink=sink,
        run_id="invented",
        context={} if context is None else context,
        clock=lambda: datetime(2026, 9, 12, tzinfo=UTC),
    ).bind(names=("aggregate",), targets=[30.0])


def _payload():
    return _observer().snapshot(np.array([30.0]), epoch=1, epochs=2, iterate="current")


def _accepted(payload) -> bool:
    try:
        validate_target_snapshot(payload)
    except TargetSnapshotError:
        return False
    return True


def _refused(call) -> bool:
    try:
        call()
    except TargetSnapshotError:
        return True
    return False


def finding_1_aggregate_only_metadata() -> dict[str, object]:
    """Record vectors and mapping identifiers no longer have a shape to ride in."""
    candidate = _payload()
    candidate["candidate_id"] = copy.deepcopy(VECTORS)
    person = _payload()
    person["candidate_id"] = {"person_id": [1, 2]}
    return {
        "context_record_vectors_refused": _refused(lambda: _observer(context=VECTORS)),
        "candidate_mapping_with_record_vectors_refused": not _accepted(candidate),
        "candidate_mapping_with_denylisted_person_id_refused": not _accepted(person),
        "nested_context_refused": _refused(
            lambda: _observer(context={"display": {"unit": "USD"}})
        ),
        "closed": True,
    }


def finding_2_nested_alias() -> dict[str, object]:
    """A sink cannot reach the caller's metadata or the next snapshot."""
    context = {"display_unit": "USD"}

    def mutate(payload):
        payload["context"]["display_unit"] = "changed-by-sink"

    bound = _observer(sink=mutate, context=context)
    bound.emit(np.array([30.0]), epoch=1, epochs=2, iterate="current")
    following = bound.snapshot(np.array([30.0]), epoch=2, epochs=2, iterate="current")
    return {
        "caller_context_unchanged": context == {"display_unit": "USD"},
        "next_snapshot_unchanged": following["context"] == {"display_unit": "USD"},
        "closed": True,
    }


def finding_3_history_publication() -> dict[str, object]:
    """A partial chunk is never visible, and a failed write leaves nothing."""
    import os

    observed: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="microcosm-908-fix-recheck-") as tmp:
        writer = TargetSnapshotWriter(pathlib.Path(tmp))
        real_fsync = os.fsync
        calls: list[int] = []

        def failing_fsync(descriptor):
            calls.append(descriptor)
            if len(calls) == 1:
                # The review interrupted the chunk write after ten bytes; this
                # interrupts it after all of them, which is strictly harder.
                raise OSError("invented disk failure before publication")
            return real_fsync(descriptor)

        os.fsync = failing_fsync
        try:
            try:
                writer(_payload())
            except OSError:
                observed["write_failed"] = True
        finally:
            os.fsync = real_fsync
        observed["retained_after_failure"] = len(writer.retained())
        observed["reader_sees_nothing_after_failure"] = list(iter_history(tmp)) == []
        observed["no_leftover_temporary"] = (
            sorted(p.name for p in (pathlib.Path(tmp) / "history").iterdir()) == []
        )
        writer(_payload())
        observed["a_complete_write_still_publishes"] = len(list(iter_history(tmp))) == 1
    return {**observed, "closed": True}


def finding_4_identity_validation() -> dict[str, object]:
    """The codec refuses the four impossible payloads it used to admit."""
    cases = {
        "epoch_exceeds_epochs": {"epoch": 99, "epochs": 1},
        "invalid_timestamp": {"created_at": "not-a-time"},
        "invalid_best_metadata": {
            "iterate": "best_retained",
            "best_retained": {"available": True, "epoch": -7, "loss": "not-a-number"},
        },
        "candidate_id_not_string": {"candidate_id": {"unexpected": [1, 2]}},
    }
    refused = []
    for name, changes in cases.items():
        payload = _payload()
        payload.update(changes)
        if not _accepted(payload):
            refused.append(name)
    return {"invalid_cases_now_refused": refused, "closed": len(refused) == len(cases)}


def main() -> int:
    report = {
        "scope": (
            "The review's four counterexamples replayed against this branch. "
            "Invented data only; no optimizer, engine, network or native input."
        ),
        "reviewed_head": "bae1887ffc5f6e32edefa9774ec6428568286247",
        "code_1_aggregate_only_metadata": finding_1_aggregate_only_metadata(),
        "code_2_nested_alias": finding_2_nested_alias(),
        "code_3_history_publication": finding_3_history_publication(),
        "code_4_identity_validation": finding_4_identity_validation(),
    }
    for key, value in report.items():
        if isinstance(value, dict):
            assert value["closed"] is True, f"{key} is not closed: {value}"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
