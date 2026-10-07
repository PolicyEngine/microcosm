"""The UK national register reconciles across its own grains (#1123)."""

from __future__ import annotations

import math

import pytest

from microcosm.build.uk_runtime.ledger_targets import UK_REGION_TIER_CODES
from microcosm.build.uk_runtime.national_reconciliation import (
    UKBandBridge,
    _apply_band_bridges,
    assert_uk_national_rows_unmoved,
    reconcile_uk_national_registry,
    uk_band_bridges,
)
from microcosm.calibrate import TargetRegistry, TargetSpec


def _spec(name: str, target_id: str, value: float, **metadata) -> TargetSpec:
    return TargetSpec(
        name=name,
        entity="person",
        measure=name,
        value=value,
        period=2025,
        family="hmrc",
        source="test",
        metadata={"contract_target_id": target_id, **metadata},
    )


def _itl(lower: int, upper: str, value: float) -> TargetSpec:
    return _spec(
        f"hmrc.itl.taxpayers_by_total_income_band.band_{lower}_{upper}@2025",
        "hmrc.itl.taxpayers_by_total_income_band",
        value,
    )


def _regional(band: str, area: str, value: float) -> TargetSpec:
    return _spec(
        f"hmrc.spi_region.taxpayers_by_region_{band}@{area}@2025",
        f"hmrc.spi_region.taxpayers_by_region_{band}",
        value,
    )


def test_regional_cells_close_on_the_uk_bands_they_nest_in():
    specs = [
        _itl(30000, "50000", 12_000.0),
        _regional("30000_40000", "E12000001", 3_000.0),
        _regional("30000_40000", "W92000004", 2_000.0),
        _regional("40000_50000", "E12000001", 4_000.0),
        _regional("40000_50000", "W92000004", 2_940.0),
    ]
    bridge = UKBandBridge(
        bridge_id="itl_taxpayers_30000",
        higher_target_id="hmrc.itl.taxpayers_by_total_income_band",
        higher_band_lower_bounds=(30000.0,),
        lower_target_ids=(
            "hmrc.spi_region.taxpayers_by_region_30000_40000",
            "hmrc.spi_region.taxpayers_by_region_40000_50000",
        ),
    )
    (receipt,) = _apply_band_bridges(specs, (bridge,))
    lower = math.fsum(spec.value for spec in specs[1:])
    assert lower == pytest.approx(12_000.0, rel=1e-12)
    assert receipt["declared_factor"] == pytest.approx(12_000.0 / 11_940.0)
    assert float(specs[1].metadata["cross_grain_value_before"]) == 3_000.0
    assert specs[1].metadata["cross_grain_control"] == "itl_taxpayers_30000"
    assert specs[0].value == 12_000.0


def test_an_open_top_regional_band_takes_every_uk_band_above_it():
    specs = [
        _itl(200000, "500000", 300.0),
        _itl(500000, "1000000", 60.0),
        _itl(1000000, "2000000", 25.0),
        _itl(2000000, "plus", 15.0),
        _regional("200000_plus", "E12000007", 396.0),
    ]
    bridge = UKBandBridge(
        bridge_id="itl_taxpayers_200000",
        higher_target_id="hmrc.itl.taxpayers_by_total_income_band",
        higher_band_lower_bounds=(200000.0, 500000.0, 1000000.0, 2000000.0),
        lower_target_ids=("hmrc.spi_region.taxpayers_by_region_200000_plus",),
    )
    _apply_band_bridges(specs, (bridge,))
    assert specs[-1].value == pytest.approx(400.0)


def test_a_bridge_whose_uk_band_is_missing_is_refused():
    specs = [_regional("30000_40000", "E12000001", 3_000.0)]
    bridge = UKBandBridge(
        bridge_id="itl_taxpayers_30000",
        higher_target_id="hmrc.itl.taxpayers_by_total_income_band",
        higher_band_lower_bounds=(30000.0,),
        lower_target_ids=("hmrc.spi_region.taxpayers_by_region_30000_40000",),
    )
    with pytest.raises(ValueError, match="expects ITL band"):
        _apply_band_bridges(specs, (bridge,))


def test_the_declared_band_bridges_tile_the_regional_bands_once_per_measure():
    bridges = uk_band_bridges()
    by_measure: dict[str, list[str]] = {}
    for bridge in bridges:
        by_measure.setdefault(bridge.higher_target_id, []).extend(
            bridge.lower_target_ids
        )
    assert set(by_measure) == {
        "hmrc.itl.taxpayers_by_total_income_band",
        "hmrc.itl.total_income_by_total_income_band",
        "hmrc.itl.income_tax_by_total_income_band",
    }
    for lower_ids in by_measure.values():
        assert len(lower_ids) == len(set(lower_ids)) == 10


def _cgt_region(geography_id: str, level: str, value: float) -> TargetSpec:
    return _spec(
        f"hmrc.cgt.taxpayers_by_region@{geography_id}@2025",
        "hmrc.cgt.taxpayers_by_region",
        value,
        geography_level=level,
        geography_id=geography_id,
    )


def test_region_rows_are_written_back_at_their_uk_control():
    registry = TargetRegistry(
        [
            _cgt_region("K02000001", "country", 120.0),
            *[_cgt_region(code, "region", 9.0) for code in UK_REGION_TIER_CODES],
        ],
        country="uk",
    )
    reconciled, receipt = reconcile_uk_national_registry(registry)
    regions = [spec for spec in reconciled.specs if "K02" not in spec.name]
    assert math.fsum(spec.value for spec in regions) == pytest.approx(120.0, rel=1e-12)
    assert receipt["rows_moved_by_exact_signature"] == len(UK_REGION_TIER_CODES)
    assert float(regions[0].metadata["cross_grain_value_before"]) == 9.0
    assert float(regions[0].metadata["cross_grain_factor"]) == pytest.approx(120 / 108)
    assert regions[0].metadata["cross_grain_control"] == "exact_signature"
    assert reconciled.specs[0].value == 120.0
    assert reconciled.version != registry.version


def test_a_register_with_nothing_to_reconcile_is_returned_unchanged():
    registry = TargetRegistry(
        [_cgt_region("K02000001", "country", 120.0)], country="uk"
    )
    reconciled, receipt = reconcile_uk_national_registry(registry)
    assert reconciled is registry
    assert receipt["band_bridges"] == []
    assert receipt["rows_moved_by_exact_signature"] == 0


def _joint_receipt(lower_grain: str, factor: float) -> dict:
    return {
        "groups": [
            {
                "inconsistency_id": f"x:country_over_{lower_grain}",
                "bridge_id": None,
                "winning_grain": "country",
                "lower_grain": lower_grain,
                "legs": [{"leg": "E12000001", "declared_factor": factor}],
            }
        ]
    }


def test_the_joint_pass_may_not_rescale_a_national_row_again():
    with pytest.raises(ValueError, match="already closed"):
        assert_uk_national_rows_unmoved(_joint_receipt("region", 1.1))
    assert_uk_national_rows_unmoved(_joint_receipt("region", 1.0 + 1e-12))
    assert_uk_national_rows_unmoved(_joint_receipt("constituency", 1.1))
