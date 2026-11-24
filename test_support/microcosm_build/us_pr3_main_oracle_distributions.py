"""Main's retirement-distributions gate, vendored as an oracle for d2283 PR 3.

Source: PolicyEngine/microcosm main at
2f45706fc246cc52604c3e541d9143b322349b90,
``packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py``,
blob 7d9659af0df5b4287b6787970537f431cfed9b2b. This module copies its
``_asec_source_mask``, ``_source_reconciliation_mask``,
``us_retirement_distributions_summary`` and
``us_retirement_distributions_signal_gate`` verbatim, in main's order. Main at
cfd5a2a5ea and 7ff5a26bc2, microcosm#1685 at 74ca47864a, microcosm#1676 at
ba410438e6 (the parent build's commit; its earlier head bd4ff48a6b too) and
microcosm#1667 at 931f237777 hold the same blob.

The oracle imports every other name the copies read, by name, from the module
that defines it: ``_BASE_ASEC_SUPPORT_CHANNEL``,
``_EXPECTED_OUTPUT_BY_ACCOUNT_CODE``, ``_NONZERO_SHARE_BANDS``,
``US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS``,
``US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS`` and ``_derived_outputs``
from the live module; ``has_assembled_support_metadata``,
``has_support_role_metadata``, ``support_gate_source_channel_series`` and
``support_role_series`` from ``support_provenance``; ``GateResult``,
``SourceRuntimeError`` and ``Frame``. The oracle shares these helpers with the
live gate: a change to any of them, or to anything they call, moves both
together, so the differential test cannot detect it. For those names, read
main's own file.

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
from microcosm.build.us_runtime.retirement_distributions import (
    _BASE_ASEC_SUPPORT_CHANNEL,
    _EXPECTED_OUTPUT_BY_ACCOUNT_CODE,
    _NONZERO_SHARE_BANDS,
    US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS,
    US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS,
    _derived_outputs,
)
from microcosm.build.us_runtime.support_provenance import (
    has_assembled_support_metadata,
    has_support_role_metadata,
    support_gate_source_channel_series,
    support_role_series,
)
from microcosm.frame import Frame

__all__ = [
    "us_retirement_distributions_signal_gate",
    "us_retirement_distributions_summary",
]


def _asec_source_mask(frame: pd.DataFrame) -> np.ndarray:
    """Select physical ASEC rows while retaining the legacy all-row source."""

    if not has_assembled_support_metadata(frame, entity="person"):
        return np.ones(len(frame), dtype=bool)
    source_channels = support_gate_source_channel_series(frame, entity="person")
    mask = source_channels.eq(_BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
    if not mask.any():
        raise SourceRuntimeError(
            "US retirement-distribution support has no physical ASEC source rows."
        )
    return mask


def _source_reconciliation_mask(frame: pd.DataFrame) -> np.ndarray:
    """Select direct operator rows whose outputs remain measured-source exact."""

    compare = _asec_source_mask(frame).copy()
    if has_support_role_metadata(frame, entity="person"):
        roles = support_role_series(frame, entity="person").to_numpy()
        compare &= roles == _BASE_ASEC_SUPPORT_CHANNEL
    return compare


def us_retirement_distributions_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and exact source-reconciliation diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    values = {
        column: pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
    }

    source_mismatches: dict[str, int] = {}
    source_rows = 0
    source_reconciliation_rows = 0
    if all(
        column in person
        for column in US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS
    ):
        source_mask = _asec_source_mask(person)
        source_expected = _derived_outputs(
            person.loc[source_mask],
            _EXPECTED_OUTPUT_BY_ACCOUNT_CODE,
        )
        expected = {
            column: np.full(len(person), np.nan, dtype=np.float64)
            for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
        }
        for column, source_values in source_expected.items():
            expected[column][source_mask] = source_values
        compare = _source_reconciliation_mask(person)
        source_rows = int(np.count_nonzero(source_mask))
        source_reconciliation_rows = int(np.count_nonzero(compare))
        source_mismatches = {
            column: int(
                np.count_nonzero(
                    compare
                    & ~np.isclose(values[column], expected[column], rtol=0, atol=0)
                )
            )
            for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
        }

    def _share(array: np.ndarray) -> float:
        positive = np.isfinite(array) & (array > 0)
        return (
            float(weights[positive].sum()) / total_weight if total_weight > 0 else 0.0
        )

    return {
        "nonzero_shares": {column: _share(array) for column, array in values.items()},
        "weighted_totals": {
            column: float(np.sum(np.nan_to_num(array) * weights))
            for column, array in values.items()
        },
        "nonzero_share_bands": {
            column: list(_NONZERO_SHARE_BANDS[column])
            for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
        },
        "unique_counts": {
            column: int(person[column].dropna().nunique())
            for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
        },
        "nonfinite": {
            column: int(np.count_nonzero(~np.isfinite(array)))
            for column, array in values.items()
        },
        "negative": {
            column: int(np.count_nonzero(array < 0)) for column, array in values.items()
        },
        "source_rows": source_rows,
        "source_reconciliation_rows": source_reconciliation_rows,
        "source_mismatches": source_mismatches,
    }


def us_retirement_distributions_signal_gate(frame: Frame) -> GateResult:
    """Require every measured account-type leaf to carry valid source signal."""

    person = frame.table("person")
    missing = [
        column
        for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
        if column not in person
    ]
    if missing:
        return GateResult(
            name="retirement_distributions_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_retirement_distributions_summary(frame)
    failures: list[str] = []
    for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS:
        nonfinite = int(summary["nonfinite"][column])
        negative = int(summary["negative"][column])
        unique = int(summary["unique_counts"][column])
        share = float(summary["nonzero_shares"][column])
        low, high = summary["nonzero_share_bands"][column]
        mismatch = int(summary["source_mismatches"].get(column, 0))
        if nonfinite:
            failures.append(f"{column}: {nonfinite} nonfinite values.")
        if negative:
            failures.append(f"{column}: {negative} negative values.")
        if unique < 2:
            failures.append(f"{column}: degenerate with {unique} distinct value(s).")
        if not low <= share <= high:
            failures.append(
                f"{column}: weighted nonzero share {share:.8f} outside "
                f"[{low:.8f}, {high:.8f}]."
            )
        if mismatch:
            failures.append(
                f"{column}: {mismatch} row(s) differ from measured ASEC slots."
            )
    return GateResult(
        name="retirement_distributions_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
