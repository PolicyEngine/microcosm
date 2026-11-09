"""PIP assessment points for every recipient (P1 of the November event-data plan, microcosm#1095).

policyengine-uk 2.174.1 decides a PIP component's rate from its points
(``pip_dl_points`` and ``pip_m_points``: 8 for the standard rate, 12 for the
enhanced) wherever no category is supplied for the year (``utils/pip.py``).
The release used to store ``pip_dl_category`` and ``pip_m_category``, which
the engine copies into every projected year, so no reform to the points
thresholds moved anyone. This stage, late in the spine after the SPI channel
(just before trading_allowance_receipts), gives every recipient points
consistent with their stored award, and the release carries
the points instead of the categories, so the engine derives the categories
from them. The spine keeps the categories: the VED exemption and the severe
disability flag read them.

The points come from DWP's Pathways to Work evidence pack, chapter 2, Tables
2.9 and 2.10 (PolicyEngine/chronicle#420 item 5): the PIP caseload's daily
living points in twelve bands and its mobility points by total, by age group,
in England and Wales at the end of January 2025, normal rules.

* A recipient of a component draws its total within the stored award: the
  standard rate from 8 to 11 points, the enhanced from 12 up.
* A recipient of one component draws the other component's total below 8.
* A person with neither component keeps -1, policyengine-uk's "not assessed".

Each age group's distribution spreads a daily living band's share evenly over
its integer points, with the open band (41 and over) at 41, and takes the
mobility totals as published. Restricted to the range the stored award allows,
it is inverted by one uniform per component on the person's own ``person_id``
(``stable_identity_uniforms``, seed 0, salts ``pip_dl_points`` and
``pip_m_points``). An age group with no share in the allowed range takes the
all-ages row, and failing that the range's lowest points. Recipients in
Scotland and Northern Ireland, outside DWP's tables, use England and Wales's,
and pension-age recipients take the 60-and-over row, which carries them (the
package's volumes run past State Pension age). With unchanged thresholds the
engine's category from the points equals the stored category for every
person, which the stage asserts. María signed these choices on 9 November 2026.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    rounded_count,
    small_cell_share,
    source_household_ids,
    support_band,
)
from microcosm.frame import Frame, MassChangeRecord

__all__ = [
    "PIP_AGE_GROUPS",
    "PIP_DAILY_LIVING_BANDS",
    "PIP_MOBILITY_TOTALS",
    "PIP_NOT_ASSESSED",
    "PIP_POINTS_RESOURCE",
    "PIP_POINTS_STAGE_NAME",
    "UKPIPPointsStageTransform",
    "draw_pip_points",
    "load_pip_points_distributions",
    "pip_points_distributions_from_rows",
    "pip_points_operation_parameters",
    "store_pip_points",
]

PIP_POINTS_STAGE_NAME = "pip_points"
PIP_POINTS_RESOURCE = "dwp_pip_points_facts.json"
PIP_POINTS_RECORD_SETS: tuple[str, ...] = (
    "dwp.pathways_pip_points.jan2025.t2_9.daily_living",
    "dwp.pathways_pip_points.jan2025.t2_10.mobility",
)
PIP_POINTS_OPERATION_KIND = "assign_pip_points_by_award"
#: The components, each with its stored category, its points output and its
#: draw's salt.
PIP_COMPONENTS: tuple[tuple[str, str, str], ...] = (
    ("daily_living", "pip_dl_category", "pip_dl_points"),
    ("mobility", "pip_m_category", "pip_m_points"),
)
PIP_POINTS_AGE = "age"
PIP_POINTS_SEED = 0
#: policyengine-uk's points default: the person was not assessed.
PIP_NOT_ASSESSED = -1
#: policyengine-uk's thresholds (gov.dwp.pip): 8 points for the standard rate,
#: 12 for the enhanced.
PIP_STANDARD_POINTS = 8
PIP_ENHANCED_POINTS = 12
#: The points each stored award allows, upper bounds inclusive (None: open).
PIP_AWARD_RANGES: Mapping[str, tuple[int, int | None]] = {
    "NONE": (0, PIP_STANDARD_POINTS - 1),
    "STANDARD": (PIP_STANDARD_POINTS, PIP_ENHANCED_POINTS - 1),
    "ENHANCED": (PIP_ENHANCED_POINTS, None),
}
#: DWP's age groups (Tables 2.9 and 2.10), bounds inclusive; the youngest also
#: takes anyone below 16, the oldest everyone from 60.
PIP_AGE_GROUPS: tuple[tuple[str, int, int | None], ...] = (
    ("16_to_19", 16, 19),
    ("20_to_29", 20, 29),
    ("30_to_39", 30, 39),
    ("40_to_49", 40, 49),
    ("50_to_59", 50, 59),
    ("60_and_over", 60, None),
)
PIP_ALL_AGES = "all"
#: Table 2.9's daily living bands, bounds inclusive (None: open).
PIP_DAILY_LIVING_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("0_to_3", 0, 3),
    ("4_to_7", 4, 7),
    ("8_to_9", 8, 9),
    ("10_to_11", 10, 11),
    ("12_to_13", 12, 13),
    ("14_to_15", 14, 15),
    ("16_to_20", 16, 20),
    ("21_to_25", 21, 25),
    ("26_to_30", 26, 30),
    ("31_to_35", 31, 35),
    ("36_to_40", 36, 40),
    ("41_and_over", 41, None),
)
#: Table 2.10's mobility totals.
PIP_MOBILITY_TOTALS: tuple[int, ...] = (0, 4, 8, 10, 12, 14, 16, 18, 20, 22, 24)
#: Where the open daily living band's recipients sit.
PIP_OPEN_BAND_POINTS = 41
#: A group's shares must add to within this many points of 100. Table 2.9
#: prints whole percentages, and its rows as ported add to 92 (16 to 19),
#: 98 (all ages), 99, 100, 100, 100 and 102 (60 and over); the draw uses
#: each row's shares only relative to one another within an award's range,
#: and the receipt records every row's total. A row further off than this
#: is refused as a broken port.
PIP_SHARE_TOTAL_TOLERANCE = 10.0
_SHARE_CONCEPTS = {
    "daily_living": (
        "dwp.pip_daily_living_points_share",
        "pip_daily_living_points_band",
    ),
    "mobility": ("dwp.pip_mobility_points_share", "pip_mobility_points"),
}
#: Outside DWP's tables (England and Wales only): their recipients use the
#: same distributions, and the receipt compares England and Wales alone.
PIP_OUTSIDE_TABLES = ("SCOTLAND", "NORTHERN_IRELAND")

PIP_POINTS_SCOPE = (
    "every person with pip_dl_category or pip_m_category other than NONE; "
    "Scotland's and Northern Ireland's recipients, outside DWP's tables "
    "(England and Wales), use England and Wales's distributions"
)
PIP_POINTS_RULE = (
    "each component's total, drawn within the stored award (NONE below 8 "
    "points for a recipient of the other component, STANDARD 8 to 11, "
    "ENHANCED 12 up) from the person's age group's distribution: a daily "
    "living band's share spread evenly over its integer points, the open band "
    "at 41, the mobility totals as published; the restricted distribution is "
    "inverted by the person's uniform on person_id; a person with neither "
    "component keeps -1, not assessed"
)
PIP_POINTS_FALLBACK = (
    "an age group with no share in the range a stored award allows takes the "
    "all-ages row, and failing that the range's lowest points"
)
PIP_POINTS_AGE_RULE = (
    "the stored age; below 20 the 16_to_19 group (anyone below 16 too), from "
    "60 the 60_and_over group, which carries pension-age claimants"
)
PIP_POINTS_MASS_CHANGE_REASON = (
    "PIP points stored on the source spine: household weights pass through "
    "unchanged and total household mass is conserved."
)
_MASS_GATE_COUPLING = (
    "The terminal family gate requires a valid mass-conserving MassChangeRecord "
    "carrying exactly this stage-specific reason."
)


def pip_points_operation_parameters() -> tuple[tuple[str, dict[str, object]], ...]:
    """The reviewed operations the manifest must carry verbatim."""

    return (
        (
            "verify_vendored_fact_resource",
            {
                "artifact_role": "pip_points_facts",
                "resource": PIP_POINTS_RESOURCE,
                "feed_pin": "chronicle_feed.json",
                "record_sets": list(PIP_POINTS_RECORD_SETS),
                "source_vintage": "PIP caseload at end January 2025",
                "require_before_source_read": True,
                "runtime_sha256_required": True,
                "fail_on_mismatch": True,
            },
        ),
        (
            PIP_POINTS_OPERATION_KIND,
            {
                "outputs": {name: output for name, _, output in PIP_COMPONENTS},
                "categories": {name: category for name, category, _ in PIP_COMPONENTS},
                "age": PIP_POINTS_AGE,
                "draws": {name: output for name, _, output in PIP_COMPONENTS},
                "seed": PIP_POINTS_SEED,
                "not_assessed": PIP_NOT_ASSESSED,
                "thresholds": {
                    "standard": PIP_STANDARD_POINTS,
                    "enhanced": PIP_ENHANCED_POINTS,
                },
                "age_groups": [name for name, _, _ in PIP_AGE_GROUPS],
                "daily_living_bands": [name for name, _, _ in PIP_DAILY_LIVING_BANDS],
                "mobility_totals": list(PIP_MOBILITY_TOTALS),
                "open_band_points": PIP_OPEN_BAND_POINTS,
                "scope": PIP_POINTS_SCOPE,
                "age_rule": PIP_POINTS_AGE_RULE,
                "fallback": PIP_POINTS_FALLBACK,
                "rule": PIP_POINTS_RULE,
            },
        ),
        (
            "record_mass_conservation_receipt",
            {
                "entity": "household",
                "reason": PIP_POINTS_MASS_CHANGE_REASON,
                "declared_factor": 1.0,
                "gate_coupling": _MASS_GATE_COUPLING,
            },
        ),
    )


def _integer_distribution(
    component: str, shares: Mapping[Any, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Integer points and their weights from one age group's published shares."""

    values: list[int] = []
    weights: list[float] = []
    if component == "daily_living":
        for name, low, high in PIP_DAILY_LIVING_BANDS:
            share = float(shares[name])
            if high is None:
                values.append(PIP_OPEN_BAND_POINTS)
                weights.append(share)
                continue
            span = high - low + 1
            for points in range(low, high + 1):
                values.append(points)
                weights.append(share / span)
    else:
        for total in PIP_MOBILITY_TOTALS:
            values.append(total)
            weights.append(float(shares[total]))
    return np.asarray(values, dtype=np.int64), np.asarray(weights, dtype=np.float64)


def pip_points_distributions_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Each component's integer points distribution by age group (and all ages).

    Refuses a missing, repeated or unexpected cell, a negative or non-finite
    share, and an age group whose shares stray from 100 by more than
    ``PIP_SHARE_TOTAL_TOLERANCE`` (Table 2.9 prints whole percentages).
    """

    groups = [name for name, _, _ in PIP_AGE_GROUPS] + [PIP_ALL_AGES]
    expected = {
        "daily_living": [name for name, _, _ in PIP_DAILY_LIVING_BANDS],
        "mobility": list(PIP_MOBILITY_TOTALS),
    }
    shares: dict[str, dict[str, dict[Any, float]]] = {
        component: {group: {} for group in groups} for component in expected
    }
    for row in rows:
        concept = row.get("concept")
        component = next(
            (
                name
                for name, (share_concept, _) in _SHARE_CONCEPTS.items()
                if share_concept == concept
            ),
            None,
        )
        if component is None:
            continue
        dimensions = row.get("dimensions") or {}
        group = dimensions.get("age_band")
        cell = dimensions.get(_SHARE_CONCEPTS[component][1])
        if group not in shares[component]:
            raise ValueError(
                f"PIP points facts carry an unexpected age group: {group!r}."
            )
        if cell not in expected[component]:
            raise ValueError(
                f"PIP points facts carry an unexpected {component} cell: {cell!r}."
            )
        if cell in shares[component][group]:
            raise ValueError(f"PIP points facts repeat {component} {group} {cell!r}.")
        value = float(row["value"])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"PIP points share {component} {group} {cell!r} is {value!r}."
            )
        shares[component][group][cell] = value
    distributions: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for component, by_group in shares.items():
        distributions[component] = {}
        for group, cells in by_group.items():
            missing = [cell for cell in expected[component] if cell not in cells]
            if missing:
                raise ValueError(
                    f"PIP points facts miss {component} {group} cells: {missing!r}."
                )
            total = sum(cells.values())
            if abs(total - 100.0) > PIP_SHARE_TOTAL_TOLERANCE:
                raise ValueError(
                    f"PIP points shares for {component} {group} add to {total:g}, not about 100."
                )
            distributions[component][group] = _integer_distribution(component, cells)
    return distributions


def load_pip_points_distributions() -> dict[
    str, dict[str, tuple[np.ndarray, np.ndarray]]
]:
    """The distributions from the committed vendored facts (pinned to the feed)."""

    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    return pip_points_distributions_from_rows(vendored_rows(PIP_POINTS_RESOURCE))


def _age_groups(age: np.ndarray) -> np.ndarray:
    age = np.asarray(age, dtype=np.float64)
    conditions = []
    names = []
    for name, _, high in PIP_AGE_GROUPS[:-1]:
        conditions.append(age <= high)
        names.append(name)
    return np.select(conditions, names, default=PIP_AGE_GROUPS[-1][0]).astype(object)


def draw_pip_points(
    *,
    category: np.ndarray,
    recipient: np.ndarray,
    age_group: np.ndarray,
    uniforms: np.ndarray,
    distributions: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, int]:
    """One component's points for each person, and how many fell back.

    A recipient draws within the range their stored ``category`` allows from
    their age group's distribution; anyone else keeps ``PIP_NOT_ASSESSED``.
    """

    category = np.asarray(category).astype(object)
    unknown = sorted(set(category[recipient].tolist()) - set(PIP_AWARD_RANGES))
    if unknown:
        raise ValueError(f"PIP categories are unknown: {unknown!r}.")
    points = np.full(len(category), PIP_NOT_ASSESSED, dtype=np.int64)
    fallbacks = 0
    for group in [name for name, _, _ in PIP_AGE_GROUPS]:
        for award, (low, high) in PIP_AWARD_RANGES.items():
            mask = recipient & (age_group == group) & (category == award)
            if not mask.any():
                continue
            values = weights = None
            for source in (group, PIP_ALL_AGES):
                candidate_values, candidate_weights = distributions[source]
                allowed = candidate_values >= low
                if high is not None:
                    allowed &= candidate_values <= high
                if float(candidate_weights[allowed].sum()) > 0.0:
                    values, weights = (
                        candidate_values[allowed],
                        candidate_weights[allowed],
                    )
                    break
                fallbacks += int(mask.sum()) if source == group else 0
            if values is None:
                values, weights = np.asarray([low], dtype=np.int64), np.asarray([1.0])
            cumulative = np.cumsum(weights) / float(weights.sum())
            index = np.searchsorted(cumulative, uniforms[mask], side="left")
            points[mask] = values[np.minimum(index, len(values) - 1)]
    return points, fallbacks


def _implied_category(points: np.ndarray) -> np.ndarray:
    return np.where(
        points >= PIP_ENHANCED_POINTS,
        "ENHANCED",
        np.where(points >= PIP_STANDARD_POINTS, "STANDARD", "NONE"),
    ).astype(object)


@dataclass(frozen=True)
class UKPIPPointsResult:
    """The frame with the stored points, and the stage's receipt."""

    frame: Frame
    receipt: Mapping[str, object]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": PIP_POINTS_STAGE_NAME,
            "seed": PIP_POINTS_SEED,
            "resource": PIP_POINTS_RESOURCE,
            **dict(self.receipt),
        }


def _shares_receipt(
    *,
    points: np.ndarray,
    component: str,
    recipient: np.ndarray,
    age_group: np.ndarray,
    weights: np.ndarray,
    sources: np.ndarray,
    distributions_rows: Mapping[str, Mapping[Any, float]] | None,
) -> list[dict[str, object]]:
    """Weighted shares by DWP's cells against DWP's, by age group, with support.

    A cell resting on fewer than three source households is withheld, and so
    is the group's smallest other cell where the withheld cells together rest
    on fewer than three, so no share gives one back (UKDS disclosure control).
    """

    if component == "daily_living":
        cells = [(name, low, high) for name, low, high in PIP_DAILY_LIVING_BANDS]
    else:
        cells = [(total, total, total) for total in PIP_MOBILITY_TOTALS]
    out: list[dict[str, object]] = []
    for group in [name for name, _, _ in PIP_AGE_GROUPS]:
        in_group = recipient & (age_group == group)
        group_support = distinct_count(sources, in_group, weights)
        total = float(weights[in_group].sum())
        masks = []
        for _, low, high in cells:
            mask = in_group & (points >= low)
            if high is not None:
                mask &= points <= high
            masks.append(mask)
        supports = [distinct_count(sources, mask, weights) for mask in masks]
        thin = {i for i, n in enumerate(supports) if 0 < n < 3}
        withheld = set(thin)
        if thin:
            together = np.zeros(len(points), dtype=bool)
            for i in thin:
                together |= masks[i]
            if distinct_count(sources, together, weights) < 3:
                rest = [i for i, n in enumerate(supports) if n >= 3]
                if rest:
                    withheld.add(
                        min(rest, key=lambda i: float(weights[masks[i]].sum()))
                    )
        rows: list[dict[str, object]] = []
        for i, (cell, _, _) in enumerate(cells):
            dwp = (
                None
                if distributions_rows is None
                else distributions_rows[group].get(cell)
            )
            entry: dict[str, object] = {
                "cell": cell,
                "dwp_share": None if dwp is None else round(float(dwp) / 100.0, 6),
                "support": support_band(supports[i]),
            }
            if i in withheld:
                entry["share"] = "withheld"
            else:
                share = float(weights[masks[i]].sum()) / total if total > 0.0 else 0.0
                entry["share"] = small_cell_share(share, supports[i])
            rows.append(entry)
        out.append(
            {"age_group": group, "support": support_band(group_support), "cells": rows}
        )
    return out


def store_pip_points(
    frame: Frame,
    *,
    distributions: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    published_shares: Mapping[str, Mapping[str, Mapping[Any, float]]] | None = None,
) -> UKPIPPointsResult:
    """Store each person's PIP points; nothing else moves."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    household = frame.table("household").copy()
    for _, _, output in PIP_COMPONENTS:
        if output in person.columns:
            raise ValueError(
                f"{output!r} is already stored: this stage is its only writer."
            )
    for column in (PIP_POINTS_AGE, *(category for _, category, _ in PIP_COMPONENTS)):
        if column not in person.columns:
            raise KeyError(
                f"person.{column} is missing; the PIP points are drawn on it."
            )
    if "region" not in household.columns:
        raise KeyError(
            "household.region is missing; the receipt compares England and Wales."
        )
    if pd.Series(person["person_id"]).duplicated().any():
        raise ValueError(
            "person_id must be unique: the PIP points draws are keyed on it."
        )
    categories = {
        name: person[category].fillna("NONE").astype(str).to_numpy(dtype=object)
        for name, category, _ in PIP_COMPONENTS
    }
    recipient = np.zeros(len(person), dtype=bool)
    for values in categories.values():
        recipient |= values != "NONE"
    age_group = _age_groups(
        pd.to_numeric(person[PIP_POINTS_AGE], errors="raise").to_numpy()
    )
    ids = person["person_id"].to_numpy()

    household_weights = frame.weights_for("household")
    weight_by_household = dict(
        zip(
            household["household_id"].to_numpy(),
            np.asarray(household_weights.values, dtype=float),
            strict=True,
        )
    )
    source_by_household = dict(
        zip(
            household["household_id"].to_numpy(),
            source_household_ids(household, stage=PIP_POINTS_STAGE_NAME),
            strict=True,
        )
    )
    region_by_household = dict(
        zip(
            household["household_id"].to_numpy(),
            household["region"].astype(str).to_numpy(),
            strict=True,
        )
    )
    weights = person["person_household_id"].map(weight_by_household).to_numpy(float)
    sources = person["person_household_id"].map(source_by_household).to_numpy()
    in_tables = ~person["person_household_id"].map(region_by_household).isin(
        PIP_OUTSIDE_TABLES
    ).to_numpy(bool)

    receipt: dict[str, object] = {
        "rule": PIP_POINTS_RULE,
        "scope": PIP_POINTS_SCOPE,
        "fallback": PIP_POINTS_FALLBACK,
        "age_rule": PIP_POINTS_AGE_RULE,
        # Record counts are rounded, never exact (UKDS CD137 cl. 8).
        "recipients": rounded_count(int(recipient.sum())),
        "recipients_support": support_band(distinct_count(sources, recipient, weights)),
        "components": {},
    }
    for name, category, output in PIP_COMPONENTS:
        uniforms = stable_identity_uniforms(ids, seed=PIP_POINTS_SEED, salt=output)
        points, fallbacks = draw_pip_points(
            category=categories[name],
            recipient=recipient,
            age_group=age_group,
            uniforms=uniforms,
            distributions=distributions[name],
        )
        implied = _implied_category(points)
        mismatches = int((implied != categories[name]).sum())
        if mismatches:
            raise ValueError(
                f"{output} implies a category other than the stored {category} for "
                f"{mismatches} people; the draw must keep every award."
            )
        person[output] = points
        receipt["components"][name] = {
            "output": output,
            "category": category,
            # Zero exact, as the check reads it.
            "category_mismatches": 0,
            "fallbacks": rounded_count(fallbacks),
            "by_award": {
                award: rounded_count(
                    int((recipient & (categories[name] == award)).sum())
                )
                for award in PIP_AWARD_RANGES
            },
            "dwp_share_totals": (
                None
                if published_shares is None
                else {
                    group: round(sum(cells.values()) / 100.0, 6)
                    for group, cells in published_shares[name].items()
                }
            ),
            "england_and_wales": _shares_receipt(
                points=points,
                component=name,
                recipient=recipient & in_tables,
                age_group=age_group,
                weights=weights,
                sources=sources,
                distributions_rows=None
                if published_shares is None
                else published_shares[name],
            ),
        }
    mass = MassChangeRecord(
        entity="household",
        old_total=household_weights.total,
        new_total=household_weights.total,
        declared_factor=1.0,
        reason=PIP_POINTS_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=person,
        benunit=frame.table("benunit").copy(),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=household_weights.values,
        mass_log=(*frame.mass_log, mass),
    )
    validate_uk_national_frame(result)
    return UKPIPPointsResult(frame=result, receipt=receipt)


def published_pip_points_shares(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[Any, float]]]:
    """DWP's shares as published (percent), by component, age group and cell."""

    shares: dict[str, dict[str, dict[Any, float]]] = {
        name: {} for name in _SHARE_CONCEPTS
    }
    for row in rows:
        for name, (concept, cell_key) in _SHARE_CONCEPTS.items():
            if row.get("concept") == concept:
                dimensions = row.get("dimensions") or {}
                shares[name].setdefault(dimensions.get("age_band"), {})[
                    dimensions.get(cell_key)
                ] = float(row["value"])
    return shares


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind the stage's operations closed-world."""

    if stage.stage != PIP_POINTS_STAGE_NAME:
        raise ValueError(
            f"PIP points received stage {stage.stage!r}, expected {PIP_POINTS_STAGE_NAME!r}."
        )
    expected = pip_points_operation_parameters()
    operations = tuple(stage.operations)
    if tuple(operation.kind for operation in operations) != tuple(
        kind for kind, _ in expected
    ):
        raise ValueError(
            f"pip_points operation order drifted: {[operation.kind for operation in operations]!r}."
        )
    for operation, (kind, parameters) in zip(operations, expected, strict=True):
        if dict(operation.parameters) != dict(parameters):
            raise ValueError(
                f"pip_points {kind} parameters drifted: {dict(operation.parameters)!r} != {dict(parameters)!r}."
            )
    if tuple(stage.outputs) != tuple(output for _, _, output in PIP_COMPONENTS):
        raise ValueError(
            f"{PIP_POINTS_STAGE_NAME} must declare exactly the two points outputs."
        )
    if stage.grain != "person":
        raise ValueError(
            f"{PIP_POINTS_STAGE_NAME} is person grain, got {stage.grain!r}."
        )


@dataclass(frozen=True)
class UKPIPPointsStageTransform:
    """Whole-stage callable storing every recipient's PIP points."""

    stage: SourceStageSpec
    #: Read from the vendored facts when not given.
    rows: Sequence[Mapping[str, Any]] | None = field(default=None, repr=False)
    last_result: UKPIPPointsResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_stage_parameters(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        if self.rows is None:
            from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

            rows = vendored_rows(PIP_POINTS_RESOURCE)
        else:
            rows = self.rows
        result = store_pip_points(
            frame,
            distributions=pip_points_distributions_from_rows(rows),
            published_shares=published_pip_points_shares(rows),
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return tuple(output for _, _, output in PIP_COMPONENTS)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
