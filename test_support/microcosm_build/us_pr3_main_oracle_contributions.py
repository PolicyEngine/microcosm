"""Main's retirement-contributions gate, vendored as an oracle for d2283 PR 3.

Source: PolicyEngine/microcosm main at
2f45706fc246cc52604c3e541d9143b322349b90,
``packages/microcosm-build/src/microcosm/build/us_runtime/retirement_contributions.py``,
blob 01b116da8ad106753257ac50b4919fd392ad158a. This module copies its
``_asec_source_mask``, ``_source_reconciliation_mask``,
``us_retirement_contributions_summary`` and
``us_retirement_contributions_signal_gate`` verbatim, in main's order. Main at
cfd5a2a5ea and 7ff5a26bc2, microcosm#1685 at 74ca47864a, microcosm#1676 at
ba410438e6 (the parent build's commit; its earlier head bd4ff48a6b too) and
microcosm#1667 at 931f237777 hold the same blob.

The oracle imports every other name the copies read, by name, from the module
that defines it: ``_BASE_ASEC_SUPPORT_CHANNEL``, ``_NONZERO_SHARE_BAND``,
``US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS`` and ``_numeric_source`` from the
live module; ``has_assembled_support_metadata``,
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
from microcosm.build.us_runtime.retirement_contributions import (
    _BASE_ASEC_SUPPORT_CHANNEL,
    _NONZERO_SHARE_BAND,
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    _numeric_source,
)
from microcosm.build.us_runtime.support_provenance import (
    has_assembled_support_metadata,
    support_gate_source_channel_series,
    support_role_series,
)
from microcosm.frame import Frame

__all__ = [
    "us_retirement_contributions_signal_gate",
    "us_retirement_contributions_summary",
]


def _asec_source_mask(frame: pd.DataFrame) -> np.ndarray:
    """Select physical ASEC rows while retaining the legacy all-row source."""

    if not has_assembled_support_metadata(frame, entity="person"):
        return np.ones(len(frame), dtype=bool)
    source_channels = support_gate_source_channel_series(frame, entity="person")
    mask = source_channels.eq(_BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
    if not mask.any():
        raise SourceRuntimeError(
            "US retirement-contribution support has no physical ASEC source rows."
        )
    return mask


def _source_reconciliation_mask(frame: pd.DataFrame) -> np.ndarray:
    """Select direct ASEC operator rows whose allocation remains source-exact."""

    source_mask = _asec_source_mask(frame)
    if not has_assembled_support_metadata(frame, entity="person"):
        return source_mask
    roles = support_role_series(frame, entity="person").to_numpy()
    return source_mask & (roles == _BASE_ASEC_SUPPORT_CHANNEL)


def us_retirement_contributions_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and source-allocation diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    contributions = {
        column: pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
    }
    source = np.full(len(person), np.nan, dtype=np.float64)
    source_mask = np.zeros(len(person), dtype=bool)
    reconciliation_mask = np.zeros(len(person), dtype=bool)
    if "RETCB_VAL" in person:
        source_mask = _asec_source_mask(person)
        source[source_mask] = _numeric_source(
            person.loc[source_mask],
            "RETCB_VAL",
        )
        reconciliation_mask = _source_reconciliation_mask(person)
    combined = np.sum(np.column_stack(tuple(contributions.values())), axis=1)

    def _share(values: np.ndarray) -> float:
        positive = np.isfinite(values) & (values > 0)
        return (
            float(weights[positive].sum()) / total_weight if total_weight > 0 else 0.0
        )

    nonfinite = {
        column: int(np.count_nonzero(~np.isfinite(values)))
        for column, values in contributions.items()
    }
    negative = {
        column: int(np.count_nonzero(values < 0))
        for column, values in contributions.items()
    }
    source_positive = reconciliation_mask & (source > 0)
    allocation_mismatch = source_positive & ~np.isclose(
        combined,
        source,
        rtol=1e-9,
        atol=1e-6,
    )
    return {
        "nonzero_shares": {
            column: _share(values) for column, values in contributions.items()
        },
        "weighted_totals": {
            column: float(np.sum(np.nan_to_num(values) * weights))
            for column, values in contributions.items()
        },
        "nonzero_share_band": list(_NONZERO_SHARE_BAND),
        "nonfinite": nonfinite,
        "negative": negative,
        "source_rows": int(np.count_nonzero(source_mask)),
        "source_reconciliation_rows": int(np.count_nonzero(reconciliation_mask)),
        "source_positive_rows": int(np.count_nonzero(source_positive)),
        "allocation_mismatch_rows": int(np.count_nonzero(allocation_mismatch)),
        "source_total": float(
            np.sum(np.where(reconciliation_mask, source, 0.0) * weights)
        ),
        "allocated_total": float(np.sum(np.nan_to_num(combined) * weights)),
    }


def us_retirement_contributions_signal_gate(frame: Frame) -> GateResult:
    """Require every retirement-contribution leaf to carry valid signal."""

    person = frame.table("person")
    missing = [
        column
        for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="retirement_contributions_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_retirement_contributions_summary(frame)
    failures: list[str] = []
    low, high = summary["nonzero_share_band"]
    for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS:
        nonfinite = int(summary["nonfinite"][column])
        negative = int(summary["negative"][column])
        share = float(summary["nonzero_shares"][column])
        if nonfinite:
            failures.append(f"{column}: {nonfinite} nonfinite values.")
        if negative:
            failures.append(f"{column}: {negative} negative values.")
        if not (low <= share <= high):
            failures.append(
                f"{column}: nonzero share {share:.4f} outside plausibility "
                f"band [{low}, {high}]."
            )
    return GateResult(
        name="retirement_contributions_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
