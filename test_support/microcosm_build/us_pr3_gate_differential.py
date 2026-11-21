"""Main's code and the live code for d2283 PR 3's four gates, on any frame.

d2283 PR 3 edits four gates for the one-population frame: weeks unemployed,
Medicare take-up, retirement contributions and retirement distributions.
Route A's frames, the base builder's pre-clone frame and the stacked pool's
frames must keep main's verdicts and statistics. ``pr3_gate_differential``
grades one frame with both versions of each gate, in one process: main's,
vendored per gate in ``us_pr3_main_oracle_<gate>.py``, and the live one.

A payload is ``{"name", "passed", "failures", "details"}`` as plain data, or
``{"raised": [type name, message]}`` if the gate raised, so a live raise where
main returns reads as a difference. ``payload_sha256`` hashes a payload's
canonical JSON with sha256: sorted keys, no whitespace, and ``repr``'s form
for each float (JSON's ``NaN`` and ``Infinity`` for the non-finite ones).

This module and the four oracles import no d2283 PR 2 file
(``one_population_contract`` or its generator), so they also run on a tree
without PR 2.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType

import numpy as np

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.medicare_take_up import (
    us_medicare_take_up_signal_gate,
)
from microcosm.build.us_runtime.retirement_contributions import (
    us_retirement_contributions_signal_gate,
)
from microcosm.build.us_runtime.retirement_distributions import (
    us_retirement_distributions_signal_gate,
)
from microcosm.build.us_runtime.weeks_unemployed import (
    us_weeks_unemployed_signal_gate,
)
from microcosm.frame import Frame
from test_support.microcosm_build import (
    us_pr3_main_oracle_contributions,
    us_pr3_main_oracle_distributions,
    us_pr3_main_oracle_medicare,
    us_pr3_main_oracle_weeks,
)

__all__ = [
    "PR3_GATES",
    "PR3_GATE_PAIRS",
    "payload_sha256",
    "pr3_gate_differential",
]

Gate = Callable[[Frame], GateResult]

#: The four gates d2283 PR 3 edits, in the plan's order.
PR3_GATES: tuple[str, ...] = (
    "weeks_unemployed",
    "medicare_take_up",
    "retirement_contributions",
    "retirement_distributions",
)

#: Each gate's (main's vendored copy, live) signal gate.
PR3_GATE_PAIRS: Mapping[str, tuple[Gate, Gate]] = MappingProxyType(
    {
        "weeks_unemployed": (
            us_pr3_main_oracle_weeks.us_weeks_unemployed_signal_gate,
            us_weeks_unemployed_signal_gate,
        ),
        "medicare_take_up": (
            us_pr3_main_oracle_medicare.us_medicare_take_up_signal_gate,
            us_medicare_take_up_signal_gate,
        ),
        "retirement_contributions": (
            us_pr3_main_oracle_contributions.us_retirement_contributions_signal_gate,
            us_retirement_contributions_signal_gate,
        ),
        "retirement_distributions": (
            us_pr3_main_oracle_distributions.us_retirement_distributions_signal_gate,
            us_retirement_distributions_signal_gate,
        ),
    }
)


def _plain(value: object) -> object:
    """``value`` as JSON-ready data; refuse any type without one exact form."""

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(
                f"Payload keys must be strings: {sorted(map(repr, value))}."
            )
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    raise TypeError(f"Payload value of type {type(value).__name__} has no plain form.")


def _payload(gate: Gate, frame: Frame) -> dict[str, object]:
    try:
        result = gate(frame)
    except Exception as error:
        return {"raised": [type(error).__name__, str(error)]}
    return _plain(
        {
            "name": result.name,
            "passed": result.passed,
            "failures": result.failures,
            "details": result.details,
        }
    )


def payload_sha256(payload: Mapping[str, object]) -> str:
    """The sha256 of ``payload``'s canonical JSON."""

    canonical = json.dumps(
        _plain(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=True,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def pr3_gate_differential(
    frame: Frame, gates: Iterable[str] = PR3_GATES
) -> dict[str, dict[str, object]]:
    """Grade ``frame`` with main's and the live version of each named gate.

    Returns, per gate in the order given, ``{"oracle": payload, "live":
    payload, "oracle_sha256": ..., "live_sha256": ...}``. Equal digests mean
    equal verdicts, failures and details, bit for bit.
    """

    if isinstance(gates, str):
        raise TypeError("gates must be an iterable of gate names, not one string.")
    comparison: dict[str, dict[str, object]] = {}
    for gate in gates:
        if gate not in PR3_GATE_PAIRS:
            raise ValueError(
                f"Unknown PR 3 gate {gate!r}; expected one of {PR3_GATES}."
            )
        oracle_gate, live_gate = PR3_GATE_PAIRS[gate]
        oracle = _payload(oracle_gate, frame)
        live = _payload(live_gate, frame)
        comparison[gate] = {
            "oracle": oracle,
            "live": live,
            "oracle_sha256": payload_sha256(oracle),
            "live_sha256": payload_sha256(live),
        }
    return comparison
