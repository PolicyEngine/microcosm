"""England's census tenure shares drift by SPREE's share change (#1123)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.uk_runtime.tenure_drift import (
    SPREE_CONCEPT_BY_TENURE_TARGET,
    SPREE_MAX_SHARE_DRIFT,
    UK_TENURE_SHARE_DRIFT_INDEX_CONCEPT,
    align_tenure_by_spree_share_drift,
)
from microcosm.build.uk_runtime.uprating_holds import is_uprating_hold
from microcosm.calibrate import TargetRegistry, TargetSpec


def _row(concept: str, area: str, year: int, value: float) -> dict:
    return {
        "concept": concept,
        "geography": {"id": area},
        "period": {"type": "calendar_year", "value": year},
        "value": value,
    }


# Private rent's share goes from 10% to 10.5% (factor 1.05); the other three
# categories keep within the drift bound.
_ROWS = [
    _row("ons.spree.private_rent_dwellings", "E06000001", 2022, 10.0),
    _row("ons.spree.private_rent_dwellings", "E06000001", 2024, 11.55),
    _row("ons.spree.social_rent_dwellings", "E06000001", 2022, 20.0),
    _row("ons.spree.social_rent_dwellings", "E06000001", 2024, 21.45),
    _row("ons.spree.owned_outright_dwellings", "E06000001", 2022, 35.0),
    _row("ons.spree.owned_outright_dwellings", "E06000001", 2024, 38.5),
    _row("ons.spree.owned_mortgage_dwellings", "E06000001", 2022, 35.0),
    _row("ons.spree.owned_mortgage_dwellings", "E06000001", 2024, 38.5),
    _row("ons.spree.total_dwellings", "E06000001", 2022, 100.0),
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
    # Share 10% in 2022, 10.5% in 2024: the census share drifts by 1.05.
    assert spec.value == pytest.approx(52.5)
    assert float(spec.metadata["uprating_factor"]) == pytest.approx(1.05)


def test_another_nation_holds_its_census_share():
    registry = TargetRegistry([_cell("S12000005", 50.0)], country="uk")
    (spec,) = align_tenure_by_spree_share_drift(
        _reference("S12000005"), registry, rows=_ROWS
    ).specs
    assert spec.value == 50.0
    assert spec.metadata["uprating_factor"] == "1"
    assert "held" in spec.metadata["uprating_index_basis"]
    # The applier moved nothing: the cell stays a hold the register declares.
    assert spec.metadata["uprating_hold"] == "census_share_held"
    assert is_uprating_hold(
        replace(spec, metadata={**spec.metadata, "ledger_fact_period": "2022"}), 2025
    )


def test_a_drifted_english_cell_is_not_a_hold():
    registry = TargetRegistry([_cell("E06000001", 50.0)], country="uk")
    (spec,) = align_tenure_by_spree_share_drift(
        _reference("E06000001"), registry, rows=_ROWS
    ).specs
    assert "uprating_hold" not in spec.metadata
    assert not is_uprating_hold(
        replace(spec, metadata={**spec.metadata, "ledger_fact_period": "2021"}), 2025
    )


def test_a_share_that_moves_beyond_the_bound_holds_the_authority():
    # Social stock down 22% in two years is a reclassification, not drift:
    # every tenure share of the authority is held, the in-bound ones too.
    rows = [
        _row("ons.spree.social_rent_dwellings", "E07000092", 2022, 8_710.0),
        _row("ons.spree.social_rent_dwellings", "E07000092", 2024, 6_778.0),
        _row("ons.spree.private_rent_dwellings", "E07000092", 2022, 9_000.0),
        _row("ons.spree.private_rent_dwellings", "E07000092", 2024, 9_600.0),
        _row("ons.spree.owned_outright_dwellings", "E07000092", 2022, 12_000.0),
        _row("ons.spree.owned_outright_dwellings", "E07000092", 2024, 12_800.0),
        _row("ons.spree.owned_mortgage_dwellings", "E07000092", 2022, 11_000.0),
        _row("ons.spree.owned_mortgage_dwellings", "E07000092", 2024, 11_700.0),
        _row("ons.spree.total_dwellings", "E07000092", 2022, 40_710.0),
        _row("ons.spree.total_dwellings", "E07000092", 2024, 40_878.0),
    ]
    registry = TargetRegistry([_cell("E07000092", 50.0)], country="uk")
    (spec,) = align_tenure_by_spree_share_drift(
        _reference("E07000092"), registry, rows=rows
    ).specs
    assert spec.value == 50.0
    assert spec.metadata["uprating_hold"] == "census_share_held"
    assert "reclassification" in spec.metadata["uprating_index_basis"]
    assert f"{SPREE_MAX_SHARE_DRIFT:.0%}" in spec.metadata["uprating_index_basis"]


def test_an_english_authority_missing_from_spree_is_refused():
    registry = TargetRegistry([_cell("E06000002", 50.0)], country="uk")
    with pytest.raises(ValueError, match="no SPREE"):
        align_tenure_by_spree_share_drift(_reference("E06000002"), registry, rows=_ROWS)


@pytest.mark.parametrize("target_id", sorted(SPREE_CONCEPT_BY_TENURE_TARGET))
def test_the_packaged_spree_resource_drifts_an_english_authority(target_id):
    # The vendored rows carry the concept at top level (#1123).
    cell = TargetSpec(
        name=f"{target_id}@E06000001@2025",
        entity="household",
        value=1000.0,
        measure="tenure",
        period=2025,
        family="ons_housing",
        source="test",
        metadata={"contract_target_id": target_id, "geography_id": "E06000001"},
    )
    reference = LedgerTargetReference(
        name=f"{target_id}@E06000001",
        ledger_selector={"source_measure_id": "households"},
        entity="household",
        measure="tenure",
        period=2025,
        uprating_index=UK_TENURE_SHARE_DRIFT_INDEX_CONCEPT,
    )
    (spec,) = align_tenure_by_spree_share_drift(
        reference, TargetRegistry([cell], country="uk")
    ).specs
    factor = float(spec.metadata["uprating_factor"])
    assert 0.5 < factor < 2.0 and factor != 1.0
