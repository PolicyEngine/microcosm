"""Main's Medicare take-up gate, vendored as an oracle for d2283 PR 3.

Source: PolicyEngine/microcosm main at
2f45706fc246cc52604c3e541d9143b322349b90,
``packages/microcosm-build/src/microcosm/build/us_runtime/medicare_take_up.py``,
blob ba40e479b42cb56ae6e4ddd6a965140fad40eb89. This module copies its
``_asec_source_mask``, ``us_medicare_take_up_summary`` and
``us_medicare_take_up_signal_gate`` verbatim, in main's order. Main at
cfd5a2a5ea and 7ff5a26bc2, microcosm#1685 at 74ca47864a, microcosm#1676 at
ba410438e6 (the parent build's commit; its earlier head bd4ff48a6b too) and
microcosm#1667 at 931f237777 hold the same blob.

The oracle imports every other name the copies read, by name, from the module
that defines it: ``_CHANNEL_ENROLLED_SHARE_BAND``, ``_ENROLLED_CODE``,
``_OUTPUT``, ``_SOURCE``, ``_WEIGHTED_ENROLLED_SHARE_BAND`` and
``_source_codes`` from the live module; ``BASE_ASEC_SUPPORT_CHANNEL``,
``has_assembled_support_metadata``, ``has_support_role_metadata``,
``support_gate_source_channel_series`` and ``support_role_series`` from
``support_provenance``; ``GateResult``, ``SourceRuntimeError`` and ``Frame``.
The oracle shares these helpers with the live gate: a change to any of them, or
to anything they call, moves both together, so the differential test cannot
detect it. For those names, read main's own file.

Track main only; do not edit otherwise.
``test_support/microcosm_build/us_pr3_gate_differential.py`` grades one frame
with these copies and with the live gate, in one process, and
``tests/engine_free/us/test_us_pr3_route_a_differential.py`` runs it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.medicare_take_up import (
    _CHANNEL_ENROLLED_SHARE_BAND,
    _ENROLLED_CODE,
    _OUTPUT,
    _SOURCE,
    _WEIGHTED_ENROLLED_SHARE_BAND,
    _source_codes,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    has_assembled_support_metadata,
    has_support_role_metadata,
    support_gate_source_channel_series,
    support_role_series,
)
from microcosm.frame import Frame

__all__ = [
    "us_medicare_take_up_signal_gate",
    "us_medicare_take_up_summary",
]


def _asec_source_mask(person: pd.DataFrame) -> np.ndarray:
    """Select physical ASEC rows without changing legacy source semantics."""

    if not has_assembled_support_metadata(person, entity="person"):
        return np.ones(len(person), dtype=bool)
    source_channels = support_gate_source_channel_series(person, entity="person")
    mask = source_channels.eq(BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
    if not mask.any():
        raise SourceRuntimeError(
            "US Medicare take-up support has no physical ASEC source rows."
        )
    return mask


def us_medicare_take_up_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and exact source-reconciliation diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    values = person[_OUTPUT].fillna(False).astype(bool).to_numpy()
    total_weight = float(weights.sum())
    summary: dict[str, object] = {
        "weighted_enrolled_share": (
            float(weights[values].sum()) / total_weight if total_weight > 0 else 0.0
        ),
        "weighted_enrolled_share_band": list(_WEIGHTED_ENROLLED_SHARE_BAND),
        "positive_count": int(np.count_nonzero(values)),
        "unique_count": int(person[_OUTPUT].dropna().nunique()),
        "missing_count": int(person[_OUTPUT].isna().sum()),
    }
    if _SOURCE in person:
        source_mask = _asec_source_mask(person)
        source_values = (
            _source_codes(person.loc[source_mask], _SOURCE) == _ENROLLED_CODE
        )
        summary["source_mismatch_count"] = int(
            np.count_nonzero(values[source_mask] != source_values)
        )
        summary["source_rows"] = int(np.count_nonzero(source_mask))
    if has_support_role_metadata(person, entity="person"):
        channel_shares: dict[str, float] = {}
        channels = support_role_series(person, entity="person").to_numpy()
        for channel in sorted(set(channels.tolist())):
            mask = channels == channel
            channel_weight = float(weights[mask].sum())
            channel_shares[str(channel)] = (
                float(weights[mask & values].sum()) / channel_weight
                if channel_weight > 0
                else 0.0
            )
        summary["channel_weighted_enrolled_shares"] = channel_shares
        summary["channel_weighted_enrolled_share_band"] = list(
            _CHANNEL_ENROLLED_SHARE_BAND
        )
    return summary


def us_medicare_take_up_signal_gate(frame: Frame) -> GateResult:
    """Require measured, nonconstant, source-consistent Medicare enrollment."""

    person = frame.table("person")
    if _OUTPUT not in person:
        return GateResult(
            name="medicare_take_up_input_signal",
            passed=False,
            failures=(f"{_OUTPUT}: missing",),
            details={},
        )
    try:
        summary = us_medicare_take_up_summary(frame)
    except SourceRuntimeError as exc:
        return GateResult(
            name="medicare_take_up_input_signal",
            passed=False,
            failures=(str(exc),),
            details={},
        )
    failures: list[str] = []
    if int(summary["missing_count"]):
        failures.append(f"{_OUTPUT}: missing values")
    if int(summary["unique_count"]) < 2:
        failures.append(f"{_OUTPUT}: constant")
    share = float(summary["weighted_enrolled_share"])
    low, high = _WEIGHTED_ENROLLED_SHARE_BAND
    if not low <= share <= high:
        failures.append(
            f"{_OUTPUT}: weighted enrolled share {share:.6f} outside "
            f"[{low:.3f}, {high:.3f}]"
        )
    mismatches = int(summary.get("source_mismatch_count", 0))
    if mismatches:
        failures.append(f"{_OUTPUT}: {mismatches} MCARE reconciliation mismatch(es)")
    channel_shares = summary.get("channel_weighted_enrolled_shares", {})
    channel_low, channel_high = _CHANNEL_ENROLLED_SHARE_BAND
    for channel, channel_share in dict(channel_shares).items():
        if not channel_low <= float(channel_share) <= channel_high:
            failures.append(
                f"{_OUTPUT}: {channel} weighted enrolled share "
                f"{float(channel_share):.6f} outside "
                f"[{channel_low:.3f}, {channel_high:.3f}]"
            )
    return GateResult(
        name="medicare_take_up_input_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
