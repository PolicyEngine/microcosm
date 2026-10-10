"""England's census tenure shares drifted to the latest dwelling stock (#1123).

The census tenure cells hold each authority's 2021 tenure shares; their level
comes from the authority's household reconciliation (the tenure partition).
For England, ONS's subnational dwelling stock by tenure (SPREE, 2022 to 2024)
says how those shares have moved since: a cell's share is drifted by its
category's share of the authority's dwellings in 2024 over the same share in
2022. SPREE counts dwellings, not households, so it moves shares only, never
levels. Authorities outside England have no such series paired with their
census, so their census shares are held (factor one, said so on the spec).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from microcosm.build.ledger_targets import LedgerTargetReference, TargetRegistry
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

UK_TENURE_SHARE_DRIFT_INDEX_CONCEPT = "ons.spree.tenure_share_drift"
UK_TENURE_SPREE_RESOURCE = "ons_spree_tenure_dwellings.json"
SPREE_OPENING_YEAR = 2022
SPREE_CLOSING_YEAR = 2024
SPREE_TOTAL_CONCEPT = "ons.spree.total_dwellings"
SPREE_CONCEPT_BY_TENURE_TARGET: Mapping[str, str] = {
    "ons.tenure.owned_outright": "ons.spree.owned_outright_dwellings",
    "ons.tenure.owned_mortgage": "ons.spree.owned_mortgage_dwellings",
    "ons.tenure.private_rent": "ons.spree.private_rent_dwellings",
    "ons.tenure.social_rent": "ons.spree.social_rent_dwellings",
}
UK_TENURE_SHARE_DRIFT_BASIS = (
    "ONS subnational dwelling stock by tenure (SPREE): the category's share of "
    "the authority's dwellings in {closing} over its share in {opening}, "
    "applied to the census tenure share; dwellings move shares, never levels"
)
UK_TENURE_SHARE_HELD_BASIS = (
    "no dwelling stock by tenure paired with this nation's census in the feed: "
    "the census tenure share is held; the level follows the authority's "
    "household reconciliation"
)


def _spree_values(
    rows: list[Mapping[str, Any]],
) -> dict[tuple[str, str, int], float]:
    values: dict[tuple[str, str, int], float] = {}
    for row in rows:
        concept = str((row.get("observed_measure") or {}).get("source_concept") or "")
        area = str((row.get("geography") or {}).get("id") or "")
        period = row.get("period") or {}
        try:
            year = int(period.get("value"))
        except (TypeError, ValueError):
            continue
        values[(concept, area, year)] = float(row["value"])
    return values


def _cell_area(spec: Any) -> str:
    return str(
        spec.metadata.get("geography_id") or spec.metadata.get("ledger_geography_id")
    )


def align_tenure_by_spree_share_drift(
    reference: LedgerTargetReference,
    registry: TargetRegistry,
    *,
    rows: list[Mapping[str, Any]] | None = None,
) -> TargetRegistry:
    """Drift each English census tenure share by SPREE's share change."""

    if reference.uprating_index != UK_TENURE_SHARE_DRIFT_INDEX_CONCEPT:
        return registry
    target_id = reference.name.split("@", 1)[0]
    category = SPREE_CONCEPT_BY_TENURE_TARGET.get(target_id)
    if category is None:
        raise ValueError(
            f"UK reference {reference.name!r}: {UK_TENURE_SHARE_DRIFT_INDEX_CONCEPT} "
            f"has no SPREE category for {target_id!r}."
        )
    rows = list(vendored_rows(UK_TENURE_SPREE_RESOURCE)) if rows is None else rows
    values = _spree_values(rows)
    aligned = []
    for spec in registry.specs:
        area = _cell_area(spec)
        keys = [
            (concept, area, year)
            for concept in (category, SPREE_TOTAL_CONCEPT)
            for year in (SPREE_OPENING_YEAR, SPREE_CLOSING_YEAR)
        ]
        if all(key in values for key in keys):
            opening_share = (
                values[(category, area, SPREE_OPENING_YEAR)]
                / values[(SPREE_TOTAL_CONCEPT, area, SPREE_OPENING_YEAR)]
            )
            closing_share = (
                values[(category, area, SPREE_CLOSING_YEAR)]
                / values[(SPREE_TOTAL_CONCEPT, area, SPREE_CLOSING_YEAR)]
            )
            if opening_share <= 0.0:
                raise ValueError(
                    f"UK target {spec.name!r}: SPREE {category} share in "
                    f"{SPREE_OPENING_YEAR} is {opening_share!r}."
                )
            factor = closing_share / opening_share
            basis = UK_TENURE_SHARE_DRIFT_BASIS.format(
                opening=SPREE_OPENING_YEAR, closing=SPREE_CLOSING_YEAR
            )
        elif area.startswith("E"):
            raise ValueError(
                f"UK target {spec.name!r}: the English authority {area!r} has no "
                f"SPREE {category} or total in {SPREE_OPENING_YEAR} and "
                f"{SPREE_CLOSING_YEAR}."
            )
        else:
            factor = 1.0
            basis = UK_TENURE_SHARE_HELD_BASIS
        aligned.append(
            replace(
                spec,
                value=spec.value * factor,
                metadata={
                    **spec.metadata,
                    "uprating_index": UK_TENURE_SHARE_DRIFT_INDEX_CONCEPT,
                    "uprating_index_basis": basis,
                    "uprating_index_resource": UK_TENURE_SPREE_RESOURCE,
                    "uprating_factor": f"{factor:.15g}",
                    "ledger_value_before_alignment": f"{spec.value:.15g}",
                },
            )
        )
    return TargetRegistry(aligned, country="uk")


def tenure_drift_appliers() -> dict[str, Any]:
    return {UK_TENURE_SHARE_DRIFT_INDEX_CONCEPT: align_tenure_by_spree_share_drift}
