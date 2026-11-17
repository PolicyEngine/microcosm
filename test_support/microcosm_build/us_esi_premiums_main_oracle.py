"""Main's ESI premium gates, vendored as the oracle for PR 0's Route A test.

Source: PolicyEngine/microcosm main at
5499382772a41a956f0c0ddc86a27d506436ced6,
``packages/microcosm-build/src/microcosm/build/us_runtime/esi_premiums.py``:
``_clone_disagreements``, ``us_esi_premiums_summary``,
``_missing_source_failure``, ``us_esi_premiums_signal_gate`` and
``us_esi_premiums_anchor_gate``, copied verbatim. Every helper and constant
they read comes from the live module; the slice of microcosm#1199 (d2283
PR 0) left each of those unchanged.

Track main only; do not edit otherwise. This file exists so that one process
grades the same frame with main's gates and the live ones
(``tests/engine_free/us/test_us_esi_premiums_route_a_differential.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.esi_premiums import (
    _CELLS_SHA256,
    _EMPLOYED_SHARE_BAND,
    _FEDERAL,
    _LOCAL_GOVERNMENT,
    _PAYS_ALL,
    _PAYS_NONE,
    _PAYS_SOME,
    _POSITIVE_SHARE_BAND,
    _PRIVATE_EMPLOYER_CODES,
    _RAW_MEAN_BAND,
    _STATE_GOVERNMENT,
    _TIER_BY_CODE,
    ANCHOR_RELATIVE_TOLERANCE,
    EMPLOYER_PREMIUM_ANCHOR,
    EMPLOYER_PREMIUM_CROSS_CHECK,
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    US_ESI_PREMIUMS_OUTPUT_COLUMNS,
    US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS,
    _person_raw_shares,
    _person_with_state,
    _stable_person_keys,
    _weighted,
    load_meps_ic_esi_premium_cells,
    meps_ic_private_active_employer_totals,
)
from microcosm.frame import Frame

__all__ = [
    "us_esi_premiums_anchor_gate",
    "us_esi_premiums_signal_gate",
    "us_esi_premiums_summary",
]


def _clone_disagreements(person: pd.DataFrame) -> int:
    if not {"source_year", "source_household_id", "source_person_id"} <= set(
        person.columns
    ):
        return 0
    keys = _stable_person_keys(person)
    work = person[list(US_ESI_PREMIUMS_OUTPUT_COLUMNS)].copy()
    work["_key"] = keys.to_numpy()
    return int(
        work.groupby("_key", sort=False)[list(US_ESI_PREMIUMS_OUTPUT_COLUMNS)]
        .nunique()
        .gt(1)
        .any(axis=1)
        .sum()
    )


def us_esi_premiums_summary(frame: Frame) -> dict[str, object]:
    """Weighted ESI premium lineage and structure, for gates and manifests.

    It recomputes the raw MEPS-IC shares from the raw ASEC columns, so the
    scale factor, the anchor-universe total and every structural zero are
    checked against the cells rather than trusted. A frame that lacks those
    columns gets the column totals only, with ``source_columns_missing``
    naming what is absent; both gates fail on it.
    """

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    employer = person[US_ESI_EMPLOYER_PREMIUM_COLUMN].to_numpy(dtype=np.float64)
    positive = employer > 0
    total_weight = float(weights.sum())
    holders = float(weights[positive].sum())
    person = _person_with_state(frame)
    summary: dict[str, object] = {
        "employer_premium_total": _weighted(weights, employer),
        "employer_premium_positive_persons": holders,
        "employer_premium_mean_per_positive_person": (
            _weighted(weights, employer, positive) / holders if holders else 0.0
        ),
        "employer_premium_positive_share": holders / total_weight
        if total_weight
        else 0.0,
        "nonfinite_rows": int((~np.isfinite(employer)).sum()),
        "negative_rows": int((employer < 0).sum()),
        "clone_disagreement_source_persons": _clone_disagreements(person),
        "anchor": dict(EMPLOYER_PREMIUM_ANCHOR),
        "cross_check": dict(EMPLOYER_PREMIUM_CROSS_CHECK),
        "cells_sha256": _CELLS_SHA256,
        "source_columns_missing": [
            column
            for column in US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS
            if column not in person.columns
        ],
    }
    if summary["source_columns_missing"]:
        return summary
    codes, raw, premium, contribution = _person_raw_shares(person)
    universe = codes.universe
    holder = codes.policyholder
    other = holder & ~universe
    raw_positive = universe & (raw > 0)
    raw_universe_total = _weighted(weights, raw, universe)
    raw_anchor_total = _weighted(weights, raw)
    scale = (
        summary["employer_premium_total"] / raw_universe_total
        if raw_universe_total
        else float("nan")
    )
    by_tier = {
        tier: _weighted(weights, employer, universe & (codes.tier == code))
        for code, tier in _TIER_BY_CODE.items()
    }
    by_sector = {
        "private": _weighted(
            weights, employer, np.isin(codes.sector, _PRIVATE_EMPLOYER_CODES)
        ),
        "federal": _weighted(weights, employer, codes.sector == _FEDERAL),
        "state_and_local": _weighted(
            weights,
            employer,
            np.isin(codes.sector, [_STATE_GOVERNMENT, _LOCAL_GOVERNMENT]),
        ),
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(
            raw_positive, employer / np.where(raw_positive, raw, 1.0), np.nan
        )
    # MEPS-IC's published employer mean of a cell is premium less average
    # contribution over every enrollee the employer pays for.
    employer_pays = universe & np.isin(codes.hipaid, [_PAYS_ALL, _PAYS_SOME])
    published = _weighted(weights, premium - contribution, employer_pays)
    summary |= {
        "weighted_policyholders": float(weights[holder].sum()),
        "weighted_employed_policyholders": float(weights[universe].sum()),
        "weighted_employed_policyholders_by_payment": {
            name: float(weights[universe & (codes.hipaid == code)].sum())
            for name, code in (
                ("all", _PAYS_ALL),
                ("some", _PAYS_SOME),
                ("none", _PAYS_NONE),
            )
        },
        "weighted_other_policyholders": float(weights[other].sum()),
        "weighted_other_policyholders_with_employer_share": float(
            weights[other & (raw > 0)].sum()
        ),
        "raw_employer_share_total": raw_universe_total,
        "raw_anchor_universe_total": raw_anchor_total,
        "raw_employer_share_mean_per_positive_person": (
            _weighted(weights, raw, raw_positive) / float(weights[raw_positive].sum())
            if raw_positive.any()
            else 0.0
        ),
        "raw_employer_share_band": list(_RAW_MEAN_BAND),
        "raw_over_published_employer_mean": (
            _weighted(weights, raw, employer_pays) / published
            if published
            else float("nan")
        ),
        "scale_factor": scale,
        "scale_factor_spread": (
            float(np.nanmax(ratio) - np.nanmin(ratio)) if raw_positive.any() else 0.0
        ),
        "anchor_universe_employer_total": scale * raw_anchor_total,
        "other_policyholder_employer_total": scale
        * (raw_anchor_total - raw_universe_total),
        "employed_share_of_anchor_universe": (
            raw_universe_total / raw_anchor_total if raw_anchor_total else float("nan")
        ),
        "employer_premium_by_tier": by_tier,
        "employer_premium_by_sector": by_sector,
        "employer_premium_outside_universe_rows": int((positive & ~universe).sum()),
        "employer_premium_where_employer_pays_none_rows": int(
            (positive & (codes.hipaid == _PAYS_NONE)).sum()
        ),
        "employer_premium_without_raw_share_rows": int(
            (positive & ~raw_positive).sum()
        ),
        "meps_ic_private_active_employer_total": (
            meps_ic_private_active_employer_totals(load_meps_ic_esi_premium_cells())
        ),
    }
    return summary


def _missing_source_failure(missing: list[str], consequence: str) -> str:
    return (
        f"assignment provenance incomplete: the person table lacks {missing}, so "
        f"{consequence}. Rebuild the base with the meps_esi_premiums stage; a "
        "release must carry the raw ASEC coverage columns the assignment read."
    )


def us_esi_premiums_signal_gate(frame: Frame) -> GateResult:
    """Structure and scale-invariant signal of the ESI employer premium input.

    Holds on any frame size (smoke builds included). It proves every positive
    value sits in the employed-policyholder universe and equals one common
    multiple of its MEPS-IC cell. Those proofs read the raw ASEC columns, so a
    frame without them fails: it cannot be certified.
    """

    person = frame.table("person")
    missing = [
        column
        for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="esi_premiums_signal",
            passed=False,
            failures=tuple(f"person column missing: {column}." for column in missing),
            details={"missing": missing},
        )
    try:
        summary = us_esi_premiums_summary(frame)
    except SourceRuntimeError as exc:
        return GateResult(
            name="esi_premiums_signal",
            passed=False,
            failures=(str(exc),),
            details={"structural_error": str(exc)},
        )
    failures: list[str] = []
    if summary["source_columns_missing"]:
        failures.append(
            _missing_source_failure(
                list(summary["source_columns_missing"]),
                "the universe, payment-status and MEPS-IC cell checks did not run",
            )
        )
    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        if person[column].dropna().nunique() < 2:
            failures.append(f"{column}: constant column — zero mass or no signal.")
    for key, label in (
        ("nonfinite_rows", "non-finite value(s)"),
        ("negative_rows", "negative value(s)"),
        ("clone_disagreement_source_persons", "support-clone disagreement(s)"),
        (
            "employer_premium_outside_universe_rows",
            "employer premium(s) outside employed policyholders",
        ),
        (
            "employer_premium_where_employer_pays_none_rows",
            "employer premium(s) where the employer pays none",
        ),
        (
            "employer_premium_without_raw_share_rows",
            "employer premium(s) with no MEPS-IC share",
        ),
    ):
        count = int(summary.get(key, 0))
        if count:
            failures.append(f"ESI premiums: {count} {label}.")
    share = float(summary["employer_premium_positive_share"])
    low, high = _POSITIVE_SHARE_BAND
    if not low <= share <= high:
        failures.append(
            f"employer premium positive share {share:.4f} outside [{low}, {high}]."
        )
    if "raw_employer_share_mean_per_positive_person" in summary:
        mean = float(summary["raw_employer_share_mean_per_positive_person"])
        low, high = _RAW_MEAN_BAND
        if not low <= mean <= high:
            failures.append(
                f"raw MEPS-IC employer share per holder ${mean:,.0f} outside "
                f"[${low:,.0f}, ${high:,.0f}]: the cell table's units are broken."
            )
        # Relative to the factor: a float32 round trip leaves ~1e-7.
        spread = float(summary["scale_factor_spread"])
        if spread > 1e-6 * float(summary["scale_factor"]):
            failures.append(
                f"employer premiums are not one common multiple of their MEPS-IC "
                f"cells (scale-factor spread {spread:.3g})."
            )
    return GateResult(
        name="esi_premiums_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def us_esi_premiums_anchor_gate(frame: Frame, *, time_period: int) -> GateResult:
    """Hold a release's weighted ESI employer premium total to its anchor.

    Runs on the calibrated export (dense and sparse). The anchor, BEA NIPA
    7.8 line 17, counts employer contributions for every policyholder, so the
    gate compares the anchor-universe total: the column (employed
    policyholders) plus the same scale factor applied to the other
    policyholders' MEPS-IC shares, recomputed from the raw ASEC columns at the
    frame's weights. It fails when the column is absent or zero-mass, when
    the raw columns are absent, when that total leaves
    ±``ANCHOR_RELATIVE_TOLERANCE`` of the anchor for ``time_period``, or when
    the employed share of it leaves its band. CMS NHE Table 24 and MEPS-IC's
    own private-sector enrollment total are recorded beside the verdict, not
    gated.
    """

    person = frame.table("person")
    missing = [
        column
        for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="esi_premiums_anchor",
            passed=False,
            failures=tuple(f"person column missing: {column}." for column in missing),
            details={"missing": missing},
        )
    try:
        summary = us_esi_premiums_summary(frame)
    except SourceRuntimeError as exc:
        return GateResult(
            name="esi_premiums_anchor",
            passed=False,
            failures=(str(exc),),
            details={"structural_error": str(exc)},
        )
    employer_total = float(summary["employer_premium_total"])
    year = str(int(time_period))
    failures: list[str] = []
    details: dict[str, object] = {
        "time_period": int(time_period),
        "employer_premium_total": employer_total,
        "employer_premium_positive_persons": summary[
            "employer_premium_positive_persons"
        ],
        "employer_premium_mean_per_positive_person": summary[
            "employer_premium_mean_per_positive_person"
        ],
        "relative_tolerance": ANCHOR_RELATIVE_TOLERANCE,
        "anchor": dict(EMPLOYER_PREMIUM_ANCHOR),
        "cross_check": dict(EMPLOYER_PREMIUM_CROSS_CHECK),
    }
    if not employer_total > 0:
        failures.append(f"{US_ESI_EMPLOYER_PREMIUM_COLUMN}: zero weighted mass.")
    anchor = EMPLOYER_PREMIUM_ANCHOR["values"].get(year)
    if anchor is None:
        failures.append(f"no BEA NIPA 7.8 line 17 anchor for {year}.")
    if summary["source_columns_missing"]:
        failures.append(
            _missing_source_failure(
                list(summary["source_columns_missing"]),
                "the anchor-universe total (every policyholder) cannot be computed",
            )
        )
    else:
        universe_total = float(summary["anchor_universe_employer_total"])
        employed_share = float(summary["employed_share_of_anchor_universe"])
        details |= {
            key: summary[key]
            for key in (
                "anchor_universe_employer_total",
                "other_policyholder_employer_total",
                "employed_share_of_anchor_universe",
                "scale_factor",
                "weighted_policyholders",
                "weighted_employed_policyholders",
                "weighted_other_policyholders",
                "employer_premium_by_sector",
            )
        }
        if not np.isfinite(universe_total):
            failures.append(
                "the anchor-universe employer total is not finite: no employed "
                "policyholder carries a MEPS-IC share to recover the scale from."
            )
        elif anchor is not None:
            relative = universe_total / float(anchor) - 1.0
            details |= {"anchor_value": float(anchor), "relative_error": relative}
            if abs(relative) > ANCHOR_RELATIVE_TOLERANCE:
                failures.append(
                    "employer premiums over every policyholder total "
                    f"${universe_total / 1e9:,.1f}B, {relative:+.1%} from BEA NIPA 7.8 "
                    f"line 17 ${float(anchor) / 1e9:,.1f}B ({year}); tolerance "
                    f"±{ANCHOR_RELATIVE_TOLERANCE:.0%}."
                )
        low, high = _EMPLOYED_SHARE_BAND
        details["employed_share_band"] = [low, high]
        if not low <= employed_share <= high:
            failures.append(
                f"employed policyholders carry {employed_share:.4f} of the "
                f"anchor-universe employer total, outside [{low}, {high}]."
            )
        nhe = EMPLOYER_PREMIUM_CROSS_CHECK["values"].get(year)
        if nhe is not None and np.isfinite(universe_total):
            details["cross_check_ratio"] = universe_total / float(nhe)
        active = dict(summary["meps_ic_private_active_employer_total"])
        if year in active:
            private = float(dict(summary["employer_premium_by_sector"])["private"])
            details["private_active_cross_check"] = {
                "meps_ic_private_active_employer_total": active[year],
                "employer_premium_private_sector": private,
                "ratio": private / active[year],
            }
    return GateResult(
        name="esi_premiums_anchor",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
