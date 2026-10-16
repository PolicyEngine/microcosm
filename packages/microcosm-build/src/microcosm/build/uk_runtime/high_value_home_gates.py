"""Diagnostic evidence and gates for high-value homes (microcosm#1240, fix 8).

Two report-only gates read one evidence summary computed once on the
calibrated pool:

- ``high_value_home_benchmarks``: England owner-occupied counts by surcharge
  band at April-2026 prices beside the OBR costing note's Table 1.1, read from
  the vendored ``uk/obr_hvcts_costing_facts.json`` rows, with the note's
  non-owner-occupied share; the regional split of homes over £2m beside the
  derived Land Registry stock; and the rank correlation by constituency of
  homes over £1.5m and £2m against the derived stock and against VOA band H.
- ``high_value_home_support``: how many distinct source households carry each
  constituency's and local authority's homes over £1.5m and £2m, in bands,
  and whether one source household dominates a cell.

The shipped gate details are aggregates only: counts of areas by support
band, correlations, shares and national totals. The per-area table (rows,
distinct sources, weight, largest source share) is written as a candidate
artifact that stays in the run directory; it never enters the diagnostics
payload or the release bundle (UKDS disclosure control).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.uk_runtime.high_value_homes import (
    UKHighValueHomeStockFacts,
    uk_owner_occupied_mask,
)

__all__ = [
    "HIGH_VALUE_HOME_SCOPE",
    "HVCTS_BANDS_2026",
    "HighValueHomeEvidence",
    "high_value_home_benchmarks_gate",
    "high_value_home_support_gate",
    "uk_high_value_home_evidence",
]

#: Surcharge bands at April-2026 prices (Budget 2025 schedule): lower bound,
#: upper bound and the OBR Table 1.1 band id.
HVCTS_BANDS_2026: tuple[tuple[float, float, str], ...] = (
    (2_000_000.0, 2_500_000.0, "gbp_2m_to_2_5m"),
    (2_500_000.0, 3_500_000.0, "gbp_2_5m_to_3_5m"),
    (3_500_000.0, 5_000_000.0, "gbp_3_5m_to_5m"),
    (5_000_000.0, math.inf, "over_gbp_5m"),
)
OBR_PROPERTIES_CONCEPT = "obr.hvcts_costing.properties_in_scope"
OBR_NON_OWNER_SHARE_CONCEPT = "obr.hvcts_costing.non_owner_occupied_share"
_OBR_BAND_IDS = frozenset((*(band_id for _, _, band_id in HVCTS_BANDS_2026), "total"))
SUPPORT_BANDS: tuple[tuple[str, int, int], ...] = (
    ("<10", 0, 10),
    ("10-29", 10, 30),
    ("30+", 30, 10**9),
)
_SUPPORT_THRESHOLDS = ("over_gbp_1_5m", "over_gbp_2m")
_LEVEL_COLUMNS = {
    "constituency": "constituency_code",
    "local_authority": "local_authority_code",
}
REQUIRED_COLUMNS = (
    "region_code",
    "constituency_code",
    "local_authority_code",
    "tenure_type",
    "main_residence_value",
    "source_household_id",
)
#: What the family and its evidence cover until the coverage decision of
#: microcosm#1240 fix 6: carried in the evidence summary and both gates'
#: details, so every reading of these figures names its scope.
HIGH_VALUE_HOME_SCOPE: Mapping[str, str] = {
    "label": "owner_occupied_main_residences_only",
    "note": (
        "The frame values each owner-occupier household's main residence. "
        "Second homes, let property and homes owned by companies or "
        "non-residents are outside this family and this evidence; the OBR "
        "costing note puts the properties in scope that are not owner-occupied "
        "at 40 per cent (microcosm#1240 fix 6)."
    ),
}


@dataclass(frozen=True)
class HighValueHomeEvidence:
    """The gate summary and the per-area support table."""

    summary: dict[str, Any]
    support: pd.DataFrame = field(repr=False)


def _spearman(
    x: Mapping[str, float], y: Mapping[str, float]
) -> tuple[float | None, int]:
    keys = sorted(set(x) & set(y))
    if len(keys) < 3:
        return None, len(keys)
    a = pd.Series([float(x[k]) for k in keys]).rank(method="average")
    b = pd.Series([float(y[k]) for k in keys]).rank(method="average")
    if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return None, len(keys)
    return float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1]), len(keys)


def _weighted_counts(
    codes: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    if not mask.any():
        return {}
    series = pd.Series(weights[mask]).groupby(codes[mask]).sum()
    return {str(code): float(value) for code, value in series.items()}


#: Shipped cells need at least this many distinct source households.
SMALL_CELL_SOURCES = 10
ENGLAND_CODE = "E92000001"


def _suppress_small_cells(
    values: Mapping[str, float], sources: Mapping[str, int]
) -> tuple[dict[str, float | None], list[str]]:
    """Null every cell under the distinct-source floor, plus one more when only one is.

    An empty cell discloses nothing and stays. A single suppressed cell could
    be recovered from the others and the total, so the smallest remaining
    non-empty cell goes with it (complementary suppression). The caller keeps
    the exact values for the gate's own evaluation; only the shipped copy is
    suppressed.
    """
    suppressed = [
        key
        for key in values
        if key != "total" and 0 < int(sources.get(key, 0)) < SMALL_CELL_SOURCES
    ]
    if len(suppressed) == 1:
        remaining = [
            key
            for key in values
            if key != "total" and key not in suppressed and int(sources.get(key, 0)) > 0
        ]
        if remaining:
            suppressed.append(min(remaining, key=lambda key: int(sources.get(key, 0))))
    shipped = {
        key: (None if key in suppressed else value) for key, value in values.items()
    }
    return shipped, sorted(suppressed)


def _obr_band_counts(
    obr_rows: Sequence[Mapping[str, Any]] | None, *, period: str, basis: str
) -> dict[str, float]:
    """Table 1.1 (static) or 1.2 (post_behaviour) counts by band id, in properties.

    Reads the vendored fact rows: the concept, the fiscal year, the behaviour
    basis and England select a row; its band is the ``property_value_band``
    dimension, or the group-by value for the total. A row that matches but
    names an unknown band or unit, or repeats a band, is refused rather than
    skipped.
    """
    if not obr_rows:
        return {}
    out: dict[str, float] = {}
    for row in obr_rows:
        if row.get("concept") != OBR_PROPERTIES_CONCEPT:
            continue
        dimensions = row.get("dimensions") or {}
        if str((row.get("period") or {}).get("value")) != str(period):
            continue
        if dimensions.get("behaviour_basis") != basis:
            continue
        if (row.get("geography") or {}).get("id") != ENGLAND_CODE:
            continue
        record = row.get("source_record_id")
        band = str(
            dimensions.get("property_value_band")
            or (row.get("layout") or {}).get("groupby_value_id")
            or ""
        )
        if band not in _OBR_BAND_IDS:
            raise ValueError(
                f"OBR HVCTS row {record!r} names an unknown band {band!r}."
            )
        if row.get("unit") != "count":
            raise ValueError(
                f"OBR HVCTS row {record!r} is in {row.get('unit')!r}, not a count."
            )
        if band in out:
            raise ValueError(f"OBR HVCTS rows repeat band {band!r} for {period}.")
        out[band] = float(row["value"])
    return out


def _obr_non_owner_share(
    obr_rows: Sequence[Mapping[str, Any]] | None,
) -> float | None:
    """The costing note's non-owner-occupied share of properties in scope.

    The vendored row is in percent; the share is returned as a fraction, or
    ``None`` when the rows carry no such fact.
    """
    rows = [
        row
        for row in obr_rows or ()
        if row.get("concept") == OBR_NON_OWNER_SHARE_CONCEPT
    ]
    if not rows:
        return None
    if len(rows) > 1:
        raise ValueError("OBR HVCTS rows carry more than one non-owner-occupied share.")
    row = rows[0]
    if row.get("unit") != "percent":
        raise ValueError(
            "OBR HVCTS non-owner-occupied share is in "
            f"{row.get('unit')!r}, not percent."
        )
    share = float(row["value"]) / 100.0
    if not 0.0 <= share < 1.0:
        raise ValueError(f"OBR HVCTS non-owner-occupied share {share} is out of range.")
    return share


def uk_high_value_home_evidence(
    household: pd.DataFrame,
    *,
    weights: np.ndarray,
    facts: UKHighValueHomeStockFacts,
    initial_weights: np.ndarray | None = None,
    obr_rows: Sequence[Mapping[str, Any]] | None = None,
    obr_period: str = "2028",
    obr_basis: str = "static",
) -> HighValueHomeEvidence:
    """Compute the gate summary and the per-area support table on the pool."""
    missing = [column for column in REQUIRED_COLUMNS if column not in household.columns]
    if missing:
        raise ValueError(f"high-value home evidence needs household columns {missing}.")
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(household):
        raise ValueError("weights must align with the household rows.")
    thresholds = {t.threshold_id: t for t in facts.thresholds}
    two_m = thresholds["over_gbp_2m"]
    deflator = two_m.basis_gbp / two_m.nominal_gbp
    stored = household["main_residence_value"].to_numpy(dtype=float)
    value_2026 = stored / deflator
    owner = uk_owner_occupied_mask(household["tenure_type"].to_numpy())
    region = household["region_code"].to_numpy().astype(str)
    england = np.char.startswith(region, "E12")
    constituency = household["constituency_code"].to_numpy().astype(str)
    source = household["source_household_id"].to_numpy()
    positive = weights > 0

    # 1. England owner-occupied counts by surcharge band at April-2026 prices.
    # The shipped counts are suppressed where a band rests on fewer than
    # ``SMALL_CELL_SOURCES`` distinct source households, with a second band
    # suppressed when only one would be, so the total cannot recover it.
    bands: dict[str, float] = {}
    band_sources: dict[str, int] = {}
    for lower, upper, band_id in HVCTS_BANDS_2026:
        mask = owner & england & positive & (value_2026 >= lower) & (value_2026 < upper)
        bands[band_id] = float(weights[mask].sum())
        band_sources[band_id] = int(pd.Series(source[mask]).nunique())
    bands["total"] = float(sum(bands[band_id] for _, _, band_id in HVCTS_BANDS_2026))
    band_sources["total"] = int(
        pd.Series(
            source[owner & england & positive & (value_2026 >= HVCTS_BANDS_2026[0][0])]
        ).nunique()
    )
    shipped_bands, suppressed_bands = _suppress_small_cells(bands, band_sources)
    obr = _obr_band_counts(obr_rows, period=obr_period, basis=obr_basis)
    obr_leg: dict[str, Any] = {"status": "awaiting_vendored_facts"}
    if obr_rows and not obr:
        obr_leg = {
            "status": "no_matching_rows",
            "period": obr_period,
            "basis": obr_basis,
        }
    if obr:
        non_owner = _obr_non_owner_share(obr_rows)
        obr_leg = {
            "status": "available",
            "period": obr_period,
            "basis": obr_basis,
            "all_properties": obr,
            "non_owner_occupied_share": non_owner,
            "frame_over_all_properties_ratio": {
                band: (
                    shipped_bands[band] / obr[band]
                    if obr.get(band) and shipped_bands.get(band) is not None
                    else None
                )
                for band in bands
            },
        }
        if non_owner is not None:
            sensitivity = obr.get("total", 0.0) * (1.0 - non_owner)
            obr_leg["owner_occupied_sensitivity_total"] = sensitivity
            # The frame's England owner-occupied total over 2m beside the
            # note's owner-occupied sensitivity (its total less the
            # non-owner-occupied share): the like-for-like level check.
            obr_leg["frame_over_owner_occupied_sensitivity"] = (
                shipped_bands["total"] / sensitivity
                if sensitivity and shipped_bands.get("total") is not None
                else None
            )
        # The derived stock's own England totals over 2m beside Table 1.1: the
        # all-tenure private-sector count is the like-for-like comparison
        # (Table 1.1 counts every residential property in scope), and the
        # owner-occupied count is what the local family binds.
        derived_england = {
            measure: facts.rows_at(
                level="country", measure_id=measure, threshold_id="over_gbp_2m"
            ).get(ENGLAND_CODE)
            for measure in ("private_sector_dwellings", "owner_occupied_homes")
        }
        obr_total = float(obr.get("total") or 0.0)
        obr_leg["derived_stock_england_over_2m"] = {
            **derived_england,
            "private_sector_over_obr_all_properties": (
                derived_england["private_sector_dwellings"] / obr_total
                if obr_total and derived_england["private_sector_dwellings"]
                else None
            ),
            "owner_occupied_over_obr_owner_occupied": (
                derived_england["owner_occupied_homes"]
                / (obr_total * (1.0 - non_owner))
                if obr_total
                and non_owner is not None
                and derived_england["owner_occupied_homes"]
                else None
            ),
        }

    # 2. Regional split of owner-occupied homes over £2m, England. The gate
    # evaluates every region; the shipped shares suppress regions resting on
    # fewer than ``SMALL_CELL_SOURCES`` distinct source households (with a
    # second region suppressed when only one would be).
    over_2m = owner & england & positive & (stored > two_m.basis_gbp)
    frame_region = _weighted_counts(region, weights, over_2m)
    region_sources = {
        code: int(pd.Series(source[over_2m & (region == code)]).nunique())
        for code in frame_region
    }
    frame_total = sum(frame_region.values())
    frame_shares = (
        {code: value / frame_total for code, value in frame_region.items()}
        if frame_total
        else {}
    )
    shipped_shares, suppressed_regions = _suppress_small_cells(
        frame_shares, region_sources
    )
    derived_region = {
        code: value
        for code, value in facts.rows_at(
            level="region",
            measure_id="owner_occupied_homes",
            threshold_id="over_gbp_2m",
        ).items()
        if code.startswith("E12")
    }
    derived_total = sum(derived_region.values())
    derived_shares = (
        {code: value / derived_total for code, value in derived_region.items()}
        if derived_total
        else {}
    )
    share_gaps = {
        code: frame_shares.get(code, 0.0) - derived_shares.get(code, 0.0)
        for code in sorted(set(frame_shares) | set(derived_shares))
    }
    # The gate tests the largest gap over every region, suppressed ones
    # included, but ships only the largest over the unsuppressed regions: the
    # derived shares are public, so a suppressed region's gap would give back
    # its frame share (PR #1270 review round 2, finding 1).
    shipped_gaps = {
        code: (None if code in suppressed_regions else gap)
        for code, gap in share_gaps.items()
    }
    largest_gap_evaluated = max(
        (abs(float(v)) for v in share_gaps.values()), default=0.0
    )
    largest_gap_shipped = max(
        (abs(float(v)) for v in shipped_gaps.values() if v is not None), default=0.0
    )

    # 3. Rank correlations by constituency.
    correlations: dict[str, dict[str, Any]] = {}
    voa_band_h = (facts.benchmarks or {}).get("voa_band_h_by_constituency") or {}
    for threshold_id in _SUPPORT_THRESHOLDS:
        threshold = thresholds[threshold_id]
        derived_seat = facts.rows_at(
            level="constituency",
            measure_id="owner_occupied_homes",
            threshold_id=threshold_id,
        )
        legs: dict[str, Any] = {}
        for label, weight_vector in (("final", weights), ("initial", initial_weights)):
            if weight_vector is None or len(weight_vector) != len(household):
                legs[label] = {"status": "unavailable"}
                continue
            weight_vector = np.asarray(weight_vector, dtype=float)
            frame_seat = _weighted_counts(
                constituency,
                weight_vector,
                owner & (weight_vector > 0) & (stored > threshold.basis_gbp),
            )
            frame_seat = {code: frame_seat.get(code, 0.0) for code in derived_seat}
            rho_derived, n_derived = _spearman(frame_seat, derived_seat)
            rho_voa, n_voa = (
                _spearman(frame_seat, {k: float(v) for k, v in voa_band_h.items()})
                if voa_band_h
                else (None, 0)
            )
            legs[label] = {
                "status": "available",
                "spearman_vs_derived_stock": rho_derived,
                "seats_vs_derived_stock": n_derived,
                "spearman_vs_voa_band_h": rho_voa,
                "seats_vs_voa_band_h": n_voa,
            }
        correlations[threshold_id] = legs

    # 4. Support by area and threshold (the table stays in the run directory).
    records: list[dict[str, Any]] = []
    for level, column in _LEVEL_COLUMNS.items():
        codes = household[column].to_numpy().astype(str)
        for threshold_id in _SUPPORT_THRESHOLDS:
            threshold = thresholds[threshold_id]
            mask = owner & positive & (stored > threshold.basis_gbp)
            table = pd.DataFrame(
                {"area": codes[mask], "source": source[mask], "weight": weights[mask]}
            )
            by_source = (
                table.groupby(["area", "source"], sort=False)["weight"]
                .sum()
                .reset_index()
            )
            grouped = by_source.groupby("area", sort=True)
            summary = grouped.agg(
                nonzero_source_households=("source", "nunique"),
                weight_sum=("weight", "sum"),
                largest_source_weight=("weight", "max"),
            )
            households_by_area = table.groupby("area", sort=True).size()
            for area, row in summary.iterrows():
                records.append(
                    {
                        "geography_level": level,
                        "area_code": str(area),
                        "threshold_id": threshold_id,
                        "nonzero_households": int(households_by_area.get(area, 0)),
                        "nonzero_source_households": int(
                            row["nonzero_source_households"]
                        ),
                        "weight_sum": float(row["weight_sum"]),
                        "max_source_share": float(
                            row["largest_source_weight"] / row["weight_sum"]
                        )
                        if row["weight_sum"] > 0
                        else 0.0,
                    }
                )
    support = pd.DataFrame.from_records(
        records,
        columns=[
            "geography_level",
            "area_code",
            "threshold_id",
            "nonzero_households",
            "nonzero_source_households",
            "weight_sum",
            "max_source_share",
        ],
    )
    support_summary: dict[str, dict[str, Any]] = {}
    for level in _LEVEL_COLUMNS:
        for threshold_id in _SUPPORT_THRESHOLDS:
            derived_areas = facts.rows_at(
                level=level,
                measure_id="owner_occupied_homes",
                threshold_id=threshold_id,
            )
            rows = support[
                (support["geography_level"] == level)
                & (support["threshold_id"] == threshold_id)
            ]
            sources_by_area = dict(
                zip(rows["area_code"], rows["nonzero_source_households"], strict=True)
            )
            dominance_by_area = dict(
                zip(rows["area_code"], rows["max_source_share"], strict=True)
            )
            support_summary[f"{level}:{threshold_id}"] = {
                "derived_target_by_area": derived_areas,
                "sources_by_area": sources_by_area,
                "dominance_by_area": dominance_by_area,
            }

    summary = {
        "schema": "microcosm.build.uk.high-value-home-evidence.v1",
        "scope": dict(HIGH_VALUE_HOME_SCOPE),
        "frame_basis_deflator": deflator,
        "households": int(len(household)),
        "owner_occupied_rows": int(owner.sum()),
        "england_owner_occupied_by_band_2026": shipped_bands,
        "england_bands_suppressed_small_cells": suppressed_bands,
        "obr": obr_leg,
        "regional_over_2m_share": {
            "frame": shipped_shares,
            "frame_regions_suppressed_small_cells": suppressed_regions,
            "derived_stock": derived_shares,
            "gap": shipped_gaps,
            "largest_gap_evaluated": largest_gap_evaluated,
            "largest_gap_shipped": largest_gap_shipped,
            # True when no unsuppressed region attains the evaluated maximum,
            # so the gate keeps the value out of its details and message.
            "largest_gap_in_suppressed_region": largest_gap_evaluated
            > largest_gap_shipped,
            "frame_total": frame_total,
            "derived_total": derived_total,
        },
        "seat_rank_correlation": correlations,
        "support": support_summary,
    }
    return HighValueHomeEvidence(summary=summary, support=support)


def _band_counts(
    sources_by_area: Mapping[str, int], assessed: Sequence[str]
) -> dict[str, int]:
    counts = {label: 0 for label, _, _ in SUPPORT_BANDS}
    for area in assessed:
        n = int(sources_by_area.get(area, 0))
        for label, lower, upper in SUPPORT_BANDS:
            if lower <= n < upper:
                counts[label] += 1
                break
    return counts


def high_value_home_support_gate(
    summary: Mapping[str, Any],
    *,
    minimum_distinct_sources: int = 10,
    minimum_target_homes: float = 100.0,
    dominance_share: float = 0.5,
) -> GateResult:
    """Every assessed area clears the distinct-source floor without a dominant source.

    An area is assessed when the derived stock places at least
    ``minimum_target_homes`` owner-occupied homes above the threshold there,
    so a rural seat with no such homes is not a failure. The details carry
    counts of areas by support band and the number with a dominant source,
    never an area code or a cell value.
    """
    if isinstance(minimum_distinct_sources, bool) or minimum_distinct_sources < 1:
        raise ValueError("minimum_distinct_sources must be a positive integer.")
    if not 0 < dominance_share <= 1:
        raise ValueError("dominance_share must be in (0, 1].")
    failures: list[str] = []
    details: dict[str, Any] = {
        "scope": dict(HIGH_VALUE_HOME_SCOPE),
        "minimum_distinct_sources": minimum_distinct_sources,
        "minimum_target_homes": minimum_target_homes,
        "dominance_share": dominance_share,
        "cells": {},
    }
    for key, cell in (summary.get("support") or {}).items():
        targets = cell.get("derived_target_by_area") or {}
        assessed = sorted(
            area
            for area, value in targets.items()
            if float(value) >= minimum_target_homes
        )
        sources = cell.get("sources_by_area") or {}
        dominance = cell.get("dominance_by_area") or {}
        bands = _band_counts(sources, assessed)
        below = [
            area
            for area in assessed
            if int(sources.get(area, 0)) < minimum_distinct_sources
        ]
        dominant = [
            area
            for area in assessed
            if float(dominance.get(area, 0.0)) > dominance_share
        ]
        details["cells"][key] = {
            "assessed_areas": len(assessed),
            "areas_by_source_band": bands,
            "areas_below_floor": len(below),
            "areas_with_dominant_source": len(dominant),
        }
        if below:
            failures.append(
                f"{key}: {len(below)} of {len(assessed)} assessed areas rest on fewer than "
                f"{minimum_distinct_sources} distinct source households."
            )
        if dominant:
            failures.append(
                f"{key}: {len(dominant)} of {len(assessed)} assessed areas have one source "
                f"household above {dominance_share:.0%} of the cell."
            )
    return GateResult(
        name="high_value_home_support",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )


def high_value_home_benchmarks_gate(
    summary: Mapping[str, Any],
    *,
    minimum_rank_correlation: float = 0.6,
    maximum_regional_share_gap: float = 0.10,
) -> GateResult:
    """Seat ranks track the derived stock and regions split as Land Registry does.

    The OBR leg is informational: Table 1.1 counts every residential property
    in scope for 2028-29, so it bounds the owner-occupied frame from above and
    is reported beside the costing note's non-owner-occupied share.
    """
    failures: list[str] = []
    correlations = summary.get("seat_rank_correlation") or {}
    reported: dict[str, Any] = {}
    for threshold_id, legs in correlations.items():
        final = legs.get("final") or {}
        reported[threshold_id] = final
        rho = final.get("spearman_vs_derived_stock")
        if rho is None:
            failures.append(
                f"{threshold_id}: no seat rank correlation could be measured."
            )
        elif rho < minimum_rank_correlation:
            failures.append(
                f"{threshold_id}: seat rank correlation with the derived stock is "
                f"{rho:.3f}, below {minimum_rank_correlation:.2f}."
            )
    regional = summary.get("regional_over_2m_share") or {}
    gaps = regional.get("gap") or {}
    # ``gap`` carries only the unsuppressed regions, so its maximum is the
    # shipped figure; the evaluated maximum covers every region and decides
    # the outcome, but its value is shipped and printed only when an
    # unsuppressed region attains it.
    shipped_worst = regional.get("largest_gap_shipped")
    if shipped_worst is None:
        shipped_worst = max(
            (abs(float(v)) for v in gaps.values() if v is not None), default=0.0
        )
    shipped_worst = float(shipped_worst)
    worst = regional.get("largest_gap_evaluated")
    worst = shipped_worst if worst is None else float(worst)
    in_suppressed_region = bool(
        regional.get("largest_gap_in_suppressed_region", worst > shipped_worst)
    )
    if worst > maximum_regional_share_gap:
        if in_suppressed_region:
            failures.append(
                "regional split of owner-occupied homes over 2m differs from the derived "
                f"stock by more than {maximum_regional_share_gap:.2f} of England's total "
                "in a region whose share is suppressed in the shipped details."
            )
        else:
            failures.append(
                "regional split of owner-occupied homes over 2m differs from the derived "
                f"stock by up to {worst:.3f} of England's total, above "
                f"{maximum_regional_share_gap:.2f}."
            )
    return GateResult(
        name="high_value_home_benchmarks",
        passed=not failures,
        failures=tuple(failures),
        details={
            "scope": dict(HIGH_VALUE_HOME_SCOPE),
            "minimum_rank_correlation": minimum_rank_correlation,
            "maximum_regional_share_gap": maximum_regional_share_gap,
            "seat_rank_correlation": reported,
            "regional_over_2m_share": {
                "frame": regional.get("frame"),
                "derived_stock": regional.get("derived_stock"),
                "largest_gap": shipped_worst,
            },
            "england_owner_occupied_by_band_2026": summary.get(
                "england_owner_occupied_by_band_2026"
            ),
            "obr": summary.get("obr"),
        },
    )
