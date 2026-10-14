"""The UK national register reconciles across its own grains (#1123)."""

from __future__ import annotations

import math

import pytest

from microcosm.build.uk_runtime.ledger_targets import UK_REGION_TIER_CODES
from microcosm.build.uk_runtime.national_reconciliation import (
    UK_TABLE_3_11_UK_ROW,
    UKBandBridge,
    _apply_band_bridges,
    assert_uk_national_rows_unmoved,
    reconcile_uk_national_registry,
    uk_band_bridges,
    uk_table_3_11_uk_rows,
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
        max_factor_shift=0.02,
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
        max_factor_shift=0.02,
    )
    _apply_band_bridges(specs, (bridge,))
    assert specs[-1].value == pytest.approx(400.0)


def test_a_band_bridge_factor_beyond_its_bound_is_refused():
    specs = [
        _itl(30000, "50000", 12_000.0),
        _regional("30000_40000", "E12000001", 11_000.0),
    ]
    bridge = UKBandBridge(
        bridge_id="itl_taxpayers_30000",
        higher_target_id="hmrc.itl.taxpayers_by_total_income_band",
        higher_band_lower_bounds=(30000.0,),
        lower_target_ids=("hmrc.spi_region.taxpayers_by_region_30000_40000",),
        max_factor_shift=0.05,
    )
    with pytest.raises(ValueError, match="beyond the declared 0.05"):
        _apply_band_bridges(specs, (bridge,))
    assert specs[1].value == 11_000.0


def _table_3_11_cell(
    band: str, area: str, published: float, growth: float, upper: int | None
) -> TargetSpec:
    lower = band.split("_", 1)[0]
    return _spec(
        f"hmrc.spi_region.taxpayers_by_region_{band}@{area}@2025",
        f"hmrc.spi_region.taxpayers_by_region_{band}",
        published * growth,
        ledger_layout_measure_id="total_income_count",
        ledger_filter_total_income_lower_bound=lower,
        **({} if upper is None else {"ledger_filter_total_income_upper_bound": upper}),
        uprating_factor=repr(growth),
        ledger_value_before_alignment=repr(published),
    )


def _explained_bridge(band: str, *lower_bounds: float) -> UKBandBridge:
    return UKBandBridge(
        bridge_id=f"itl_taxpayers_{band}",
        higher_target_id="hmrc.itl.taxpayers_by_total_income_band",
        higher_band_lower_bounds=tuple(lower_bounds),
        lower_target_ids=(f"hmrc.spi_region.taxpayers_by_region_{band}",),
        max_factor_shift=0.05,
        explained_by=UK_TABLE_3_11_UK_ROW,
    )


def test_a_band_bridge_factor_is_checked_against_the_tables_own_uk_row():
    # Table 3.11: 1,000 across the regions and 1,030 in the UK row. The band
    # grows 10%, so the projected UK band is 1,133 and the factor is 1.03.
    specs = [
        _itl(12570, "15000", 1_133.0),
        _table_3_11_cell("12570_15000", "E12000001", 600.0, 1.1, 15000),
        _table_3_11_cell("12570_15000", "W92000004", 400.0, 1.1, 15000),
    ]
    bridge = _explained_bridge("12570_15000", 12570.0)
    uk_rows = {("total_income_count", 12570.0): 1_030.0}
    (receipt,) = _apply_band_bridges(specs, (bridge,), uk_rows=uk_rows)
    assert receipt["declared_factor"] == pytest.approx(1.03)
    assert receipt["explained_factor"] == pytest.approx(1.03)
    assert receipt["explained_by"] == UK_TABLE_3_11_UK_ROW


def test_a_factor_the_uk_row_does_not_explain_is_refused_inside_its_bound():
    # The factor is 1.03, inside the declared 5%, but the UK row holds only
    # 1,010: the taxpayers of no region explain one point of it, not three.
    specs = [
        _itl(12570, "15000", 1_133.0),
        _table_3_11_cell("12570_15000", "E12000001", 1_000.0, 1.1, 15000),
    ]
    bridge = _explained_bridge("12570_15000", 12570.0)
    with pytest.raises(ValueError, match="no longer explain the gap"):
        _apply_band_bridges(
            specs, (bridge,), uk_rows={("total_income_count", 12570.0): 1_010.0}
        )
    assert specs[1].value == pytest.approx(1_100.0)


def test_an_open_top_regional_band_is_explained_by_every_uk_band_above_it():
    specs = [
        _itl(200000, "plus", 404.0),
        _table_3_11_cell("200000_plus", "E12000007", 400.0, 1.0, None),
    ]
    uk_rows = {
        ("total_income_count", 150000.0): 999.0,
        ("total_income_count", 200000.0): 300.0,
        ("total_income_count", 300000.0): 80.0,
        ("total_income_count", 1000000.0): 24.0,
        ("total_income_amount", 200000.0): 999.0,
    }
    bridge = _explained_bridge("200000_plus", 200000.0)
    (receipt,) = _apply_band_bridges(specs, (bridge,), uk_rows=uk_rows)
    assert receipt["explained_factor"] == pytest.approx(404.0 / 400.0)


def test_an_explained_bridge_needs_its_cells_table_identity_and_a_uk_row():
    bridge = _explained_bridge("12570_15000", 12570.0)
    unnamed = [
        _itl(12570, "15000", 1_000.0),
        _regional("12570_15000", "E12000001", 1_000.0),
    ]
    with pytest.raises(ValueError, match="does not name its Table 3.11 measure"):
        _apply_band_bridges(unnamed, (bridge,), uk_rows={})
    named = [
        _itl(12570, "15000", 1_000.0),
        _table_3_11_cell("12570_15000", "E12000001", 1_000.0, 1.0, 15000),
    ]
    with pytest.raises(ValueError, match="no United Kingdom row"):
        _apply_band_bridges(named, (bridge,), uk_rows={})


def test_the_vendored_uk_rows_carry_every_band_of_the_three_measures():
    rows = uk_table_3_11_uk_rows()
    measures = {"total_income_count", "total_income_amount", "total_tax_amount"}
    assert {measure for measure, _ in rows} == measures
    assert len(rows) == 39
    # Table 3.11's UK row, 2023-24: 2,850 thousand taxpayers at 12,570-15,000.
    assert rows[("total_income_count", 12570.0)] == 2_850_000.0


def test_every_declared_band_bridge_states_a_bound_and_its_measured_gap():
    bridges = uk_band_bridges()
    assert len(bridges) == 24
    assert {bridge.explained_by for bridge in bridges} == {UK_TABLE_3_11_UK_ROW}
    with pytest.raises(ValueError, match="unknown explanation"):
        uk_band_bridges(
            {
                "band_bridges": [
                    {
                        "bridge_id": "x",
                        "higher_target_id": "h",
                        "higher_band_lower_bounds": [0],
                        "lower_target_ids": [],
                        "max_factor_shift": 0.01,
                        "explained_by": "a_guess",
                    }
                ]
            }
        )
    assert all(0.005 < bridge.max_factor_shift <= 0.063 for bridge in bridges)
    # The lowest band carries most of the taxpayers HMRC places in no region.
    by_id = {bridge.bridge_id: bridge for bridge in bridges}
    assert by_id["itl_income_tax_12570_vs_spi_region_12570"].max_factor_shift == 0.063
    assert by_id["itl_taxpayers_30000_vs_spi_region_30000_40000"].max_factor_shift == (
        0.008
    )
    with pytest.raises(ValueError, match="must declare a max_factor_shift"):
        uk_band_bridges(
            {
                "band_bridges": [
                    {
                        "bridge_id": "x",
                        "higher_target_id": "h",
                        "higher_band_lower_bounds": [0],
                        "lower_target_ids": [],
                    }
                ]
            }
        )


def test_a_bridge_whose_uk_band_is_missing_is_refused():
    specs = [_regional("30000_40000", "E12000001", 3_000.0)]
    bridge = UKBandBridge(
        bridge_id="itl_taxpayers_30000",
        higher_target_id="hmrc.itl.taxpayers_by_total_income_band",
        higher_band_lower_bounds=(30000.0,),
        lower_target_ids=("hmrc.spi_region.taxpayers_by_region_30000_40000",),
        max_factor_shift=0.02,
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


def test_the_uk_composition_takes_the_nations_household_sum():
    """#1123, nations win: the composition partition keeps its shares and
    scales onto the nations' official estimates summed."""

    from microcosm.build.uk_runtime.national_reconciliation import (
        _apply_sum_bridges,
        uk_sum_bridges,
    )

    (bridge,) = uk_sum_bridges()
    specs = [
        _spec(f"{target_id}@x", target_id, value)
        for target_id, value in zip(
            bridge.higher_target_ids, (240.0, 14.0, 26.0, 8.0), strict=True
        )
    ] + [
        _spec(f"{target_id}@K02000001", target_id, 28.0)
        for target_id in bridge.lower_target_ids
    ]
    (receipt,) = _apply_sum_bridges(specs, (bridge,))
    lower = [spec.value for spec in specs[len(bridge.higher_target_ids) :]]
    assert math.fsum(lower) == pytest.approx(288.0)
    assert receipt["declared_factor"] == pytest.approx(288.0 / 280.0)
    assert len(set(lower)) == 1


def test_a_partly_bound_sum_bridge_is_refused():
    from microcosm.build.uk_runtime.national_reconciliation import (
        _apply_sum_bridges,
        uk_sum_bridges,
    )

    (bridge,) = uk_sum_bridges()
    specs = [_spec("a", bridge.higher_target_ids[0], 1.0)]
    with pytest.raises(ValueError, match="partly bound"):
        _apply_sum_bridges(specs, (bridge,))
