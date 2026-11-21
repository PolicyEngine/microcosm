"""Main's weeks-unemployed gate, vendored as an oracle for d2283 PR 3.

Source: PolicyEngine/microcosm main at
2f45706fc246cc52604c3e541d9143b322349b90,
``packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py``,
blob dd7e3664110f6cde00f41348b6f35c177630469f. This module copies its
``us_weeks_unemployed_summary``, ``_weeks_unemployed_gate_scopes`` and
``us_weeks_unemployed_signal_gate`` verbatim, in main's order. Main at
cfd5a2a5ea and 7ff5a26bc2, microcosm#1685 at 74ca47864a, microcosm#1676 at
ba410438e6 (the parent build's commit; its earlier head bd4ff48a6b too) and
microcosm#1667 at 931f237777 hold the same blob.

The oracle imports every other name the copies read, by name, from the module
that defines it: ``_ASEC_CHANNEL``, ``_CHANNEL_POSITIVE_SHARE_BANDS``,
``_CHANNEL_WEIGHTED_MEAN_WEEKS_BANDS``, ``_OPTIONAL_UC_PREDICTOR``,
``_OUTPUT``, ``_PUF_CHANNEL`` and ``_SOURCE`` from the live module;
``has_assembled_support_metadata``, ``has_support_role_metadata``,
``support_clone_index_column``, ``support_gate_source_channel_series`` and
``support_role_series`` from ``support_provenance``; ``GateResult`` and
``Frame``. The oracle shares these helpers with the live gate: a change to any
of them, or to anything they call, moves both together, so the differential
test cannot detect it. For those names, read main's own file.

Track main only; do not edit otherwise.
``test_support/microcosm_build/us_pr3_gate_differential.py`` grades one frame
with these copies and with the live gate, in one process, and
``tests/engine_free/us/test_us_pr3_route_a_differential.py`` runs it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.support_provenance import (
    has_assembled_support_metadata,
    has_support_role_metadata,
    support_clone_index_column,
    support_gate_source_channel_series,
    support_role_series,
)
from microcosm.build.us_runtime.weeks_unemployed import (
    _ASEC_CHANNEL,
    _CHANNEL_POSITIVE_SHARE_BANDS,
    _CHANNEL_WEIGHTED_MEAN_WEEKS_BANDS,
    _OPTIONAL_UC_PREDICTOR,
    _OUTPUT,
    _PUF_CHANNEL,
    _SOURCE,
)
from microcosm.frame import Frame

__all__ = [
    "us_weeks_unemployed_signal_gate",
    "us_weeks_unemployed_summary",
]

# The copies below stay byte for byte as main has them. ruff format would
# rewrap one line of the summary, in main's file as here, and CI runs only
# ruff check, so formatting stays off from here to the end of the file.
# fmt: off


def us_weeks_unemployed_summary(frame: Frame) -> dict[str, object]:
    """Return integer-domain, source-reconciliation, and channel diagnostics."""

    person = frame.table("person")
    if _OUTPUT not in person:
        raise ValueError(f"US weeks-unemployed summary requires {_OUTPUT!r}.")
    values = pd.to_numeric(person[_OUTPUT], errors="coerce").to_numpy(dtype=np.float64)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    if (
        not np.isfinite(weights).all()
        or (weights < 0.0).any()
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError(
            "US weeks-unemployed summary requires finite nonnegative person "
            "weights with positive total mass."
        )
    finite = np.isfinite(values)
    integer = finite & (values == np.rint(values))
    in_range = integer & (values >= 0.0) & (values <= 52.0)
    positive = in_range & (values > 0.0)

    (
        channel,
        channel_roster,
        source_rows,
        source_reconciliation_rows,
        uc_constraint_rows,
    ) = _weeks_unemployed_gate_scopes(person)
    channels: dict[str, dict[str, float | int]] = {}
    for name in channel_roster:
        mask = channel == name
        channel_weight = float(weights[mask].sum())
        plausibility_role = _ASEC_CHANNEL if name == _ASEC_CHANNEL else _PUF_CHANNEL
        channels[name] = {
            "rows": int(np.count_nonzero(mask)),
            "positive_rows": int(np.count_nonzero(mask & positive)),
            "positive_share": (
                float(weights[mask & positive].sum()) / channel_weight
                if channel_weight > 0.0
                else 0.0
            ),
            "weighted_weeks": float(np.dot(weights[mask], np.nan_to_num(values[mask]))),
            "weighted_mean_weeks": (
                float(np.dot(weights[mask], np.nan_to_num(values[mask])))
                / channel_weight
                if channel_weight > 0.0
                else 0.0
            ),
            "positive_share_band": list(
                _CHANNEL_POSITIVE_SHARE_BANDS[plausibility_role]
            ),
            "weighted_mean_weeks_band": list(
                _CHANNEL_WEIGHTED_MEAN_WEEKS_BANDS[plausibility_role]
            ),
        }

    source_missing = _SOURCE not in person
    source_invalid = source_mismatch = 0
    if not source_missing:
        source_raw = pd.to_numeric(person[_SOURCE], errors="coerce").to_numpy(
            dtype=np.float64
        )
        source_valid = np.isfinite(source_raw) & (source_raw == np.floor(source_raw))
        source_valid &= (source_raw == -1.0) | (
            (source_raw >= 0.0) & (source_raw <= 52.0)
        )
        source_invalid = int(np.count_nonzero(source_rows & ~source_valid))
        expected = np.where(source_raw == -1.0, 0.0, source_raw)
        source_mismatch = int(
            np.count_nonzero(
                source_reconciliation_rows
                & source_valid
                & finite
                & (values != expected)
            )
        )
    uc_constraint_mismatch = 0
    if _OPTIONAL_UC_PREDICTOR in person:
        uc = pd.to_numeric(person[_OPTIONAL_UC_PREDICTOR], errors="coerce").to_numpy(
            dtype=np.float64
        )
        uc_constraint_mismatch = int(
            np.count_nonzero(
                uc_constraint_rows
                & np.isfinite(uc)
                & (uc <= 0.0)
                & finite
                & (values != 0.0)
            )
        )

    return {
        "rows": int(len(person)),
        "missing_or_nonfinite": int(np.count_nonzero(~finite)),
        "noninteger": int(np.count_nonzero(finite & ~integer)),
        "out_of_range": int(np.count_nonzero(integer & ~in_range)),
        "positive_rows": int(np.count_nonzero(positive)),
        "positive_share": float(weights[positive].sum() / weights.sum()),
        "weighted_weeks": float(np.dot(weights, np.nan_to_num(values))),
        "source_missing": source_missing,
        "source_rows": int(np.count_nonzero(source_rows)),
        "source_reconciliation_rows": int(
            np.count_nonzero(source_reconciliation_rows)
        ),
        "source_invalid": source_invalid,
        "source_mismatch_count": source_mismatch,
        "uc_constraint_rows": int(np.count_nonzero(uc_constraint_rows)),
        "uc_constraint_mismatch_count": uc_constraint_mismatch,
        "channel_roster": list(channel_roster),
        "channels": channels,
    }


def _weeks_unemployed_gate_scopes(
    person: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Resolve source channels and the two reviewed constraint scopes."""

    rows = len(person)
    if not has_support_role_metadata(person, entity="person"):
        all_rows = np.ones(rows, dtype=bool)
        return (
            np.full(rows, _ASEC_CHANNEL, dtype=object),
            (_ASEC_CHANNEL,),
            all_rows,
            all_rows.copy(),
            np.zeros(rows, dtype=bool),
        )

    # Validate complete channel/clone metadata even when an assembled frame's
    # raw source channels, rather than its compatibility roles, own the gate.
    roles = support_role_series(person, entity="person")
    assembled = has_assembled_support_metadata(person, entity="person")
    if not assembled:
        channel = roles.to_numpy(dtype=object, copy=True)
        roster = tuple(
            name for name in (_ASEC_CHANNEL, _PUF_CHANNEL) if np.any(channel == name)
        )
        source_rows = channel == _ASEC_CHANNEL
        if not np.any(source_rows):
            raise ValueError("US weeks-unemployed support has no ASEC source rows.")
        return (
            channel,
            roster,
            source_rows,
            source_rows.copy(),
            channel == _PUF_CHANNEL,
        )

    clone_column = support_clone_index_column("person")
    source_channel = support_gate_source_channel_series(
        person,
        entity="person",
    ).to_numpy(dtype=object, copy=True)
    observed = {str(value) for value in source_channel}
    if _ASEC_CHANNEL not in observed:
        raise ValueError("US weeks-unemployed support has no ASEC source rows.")
    roster = (_ASEC_CHANNEL, *sorted(observed - {_ASEC_CHANNEL}))
    clone_index = pd.to_numeric(person[clone_column], errors="raise").to_numpy(
        dtype=np.int64
    )
    native = clone_index == 0
    source_rows = source_channel == _ASEC_CHANNEL
    # Native ASEC rows are the direct LKWEEKS carry. Non-native ASEC rows are
    # the reviewed PUF-detail imputation; native non-ASEC rows are the reviewed
    # post-transfer calibration scope. Other non-native source rows have no UC
    # zero rule in either producer contract.
    source_reconciliation_rows = source_rows & native
    uc_constraint_rows = (source_rows & ~native) | (~source_rows & native)
    return (
        source_channel,
        roster,
        source_rows,
        source_reconciliation_rows,
        uc_constraint_rows,
    )


def us_weeks_unemployed_signal_gate(frame: Frame) -> GateResult:
    """Require exact ASEC carry and integer signal on every support channel."""

    person = frame.table("person")
    if _OUTPUT not in person:
        return GateResult(
            name="weeks_unemployed_signal",
            passed=False,
            failures=(f"person column missing: {_OUTPUT}.",),
            details={"missing": [_OUTPUT]},
        )
    try:
        summary = us_weeks_unemployed_summary(frame)
    except (TypeError, ValueError) as error:
        return GateResult(
            name="weeks_unemployed_signal",
            passed=False,
            failures=(str(error),),
            details={},
        )
    failures: list[str] = []
    for key, label in (
        ("missing_or_nonfinite", "missing or nonfinite value(s)"),
        ("noninteger", "noninteger value(s)"),
        ("out_of_range", "value(s) outside 0--52"),
    ):
        if int(summary[key]):
            failures.append(f"{_OUTPUT}: {summary[key]} {label}.")
    if bool(summary["source_missing"]):
        failures.append("ASEC LKWEEKS source column is missing.")
    if int(summary["source_invalid"]):
        failures.append(
            f"ASEC LKWEEKS has {summary['source_invalid']} invalid source value(s)."
        )
    if int(summary["source_mismatch_count"]):
        failures.append(
            f"{_OUTPUT} has {summary['source_mismatch_count']} ASEC source "
            "reconciliation mismatch(es)."
        )
    if int(summary["uc_constraint_mismatch_count"]):
        failures.append(
            f"{_OUTPUT} has {summary['uc_constraint_mismatch_count']} row(s) "
            "positive outside the unemployment-compensation constraint."
        )
    required_channels = tuple(summary["channel_roster"])
    channels = summary["channels"]
    for name in required_channels:
        channel = channels[name]
        if int(channel["rows"]) == 0:
            failures.append(f"{name} weeks-unemployed support is empty.")
        elif (
            int(channel["positive_rows"]) == 0 or float(channel["weighted_weeks"]) <= 0
        ):
            failures.append(f"{name} weeks_unemployed is default-only.")
        else:
            positive_share = float(channel["positive_share"])
            positive_low, positive_high = channel["positive_share_band"]
            if not (positive_low <= positive_share <= positive_high):
                failures.append(
                    f"{name} weeks_unemployed positive share "
                    f"{positive_share:.6f} outside plausibility band "
                    f"[{positive_low}, {positive_high}]."
                )
            weighted_mean = float(channel["weighted_mean_weeks"])
            mean_low, mean_high = channel["weighted_mean_weeks_band"]
            if not (mean_low <= weighted_mean <= mean_high):
                failures.append(
                    f"{name} weeks_unemployed weighted mean weeks "
                    f"{weighted_mean:.6f} outside plausibility band "
                    f"[{mean_low}, {mean_high}]."
                )
    return GateResult(
        name="weeks_unemployed_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
