"""England's census tenure shares drift by SPREE's share change (#1123)."""

from __future__ import annotations

import pytest

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.uk_runtime.tenure_drift import (
    UK_TENURE_SHARE_DRIFT_INDEX_CONCEPT,
    align_tenure_by_spree_share_drift,
)
from microcosm.calibrate import TargetRegistry, TargetSpec


def _row(concept: str, area: str, year: int, value: float) -> dict:
    return {
        "observed_measure": {"source_concept": concept},
        "geography": {"id": area},
        "period": {"type": "calendar_year", "value": year},
        "value": value,
    }


_ROWS = [
    _row("ons.spree.private_rent_dwellings", "E06000001", 2022, 10.0),
    _row("ons.spree.total_dwellings", "E06000001", 2022, 100.0),
    _row("ons.spree.private_rent_dwellings", "E06000001", 2024, 13.2),
    _row("ons.spree.total_dwellings", "E06000001", 2024, 110.0),
]


def _cell(area: str, value: float) -> TargetSpec:
    return TargetSpec(
        name=f"ons.tenure.private_rent@{area}@2025",
        entity="household",
        value=value,
        measure="tenure/private_rent",
        period=2025,
        family="ons_housing",
        source="test",
        metadata={
            "contract_target_id": "ons.tenure.private_rent",
            "geography_id": area,
        },
    )


def _reference(area: str) -> LedgerTargetReference:
    return LedgerTargetReference(
        name=f"ons.tenure.private_rent@{area}",
        ledger_selector={"source_measure_id": "households"},
        entity="household",
        measure="tenure/private_rent",
        period=2025,
        uprating_index=UK_TENURE_SHARE_DRIFT_INDEX_CONCEPT,
    )


def test_an_english_share_moves_with_its_dwelling_share():
    registry = TargetRegistry([_cell("E06000001", 50.0)], country="uk")
    (spec,) = align_tenure_by_spree_share_drift(
        _reference("E06000001"), registry, rows=_ROWS
    ).specs
    # Share 10% in 2022, 12% in 2024: the census share drifts by 1.2.
    assert spec.value == pytest.approx(60.0)
    assert float(spec.metadata["uprating_factor"]) == pytest.approx(1.2)


def test_another_nation_holds_its_census_share():
    registry = TargetRegistry([_cell("S12000005", 50.0)], country="uk")
    (spec,) = align_tenure_by_spree_share_drift(
        _reference("S12000005"), registry, rows=_ROWS
    ).specs
    assert spec.value == 50.0
    assert spec.metadata["uprating_factor"] == "1"
    assert "held" in spec.metadata["uprating_index_basis"]


def test_an_english_authority_missing_from_spree_is_refused():
    registry = TargetRegistry([_cell("E06000002", 50.0)], country="uk")
    with pytest.raises(ValueError, match="no SPREE"):
        align_tenure_by_spree_share_drift(_reference("E06000002"), registry, rows=_ROWS)
