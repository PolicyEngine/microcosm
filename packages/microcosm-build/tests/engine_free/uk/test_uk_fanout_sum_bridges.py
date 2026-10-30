"""A banded national target, summed, controls its area cells (#1123)."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from importlib import resources as importlib_resources

import pytest

from microcosm.build.cross_grain import _measurement_signature
from microcosm.build.uk_runtime import ledger_targets
from microcosm.build.uk_runtime.cross_grain_declarations import (
    is_uk_fanout_target,
    load_uk_cross_grain_declarations,
    uk_cross_grain_coverage_violations,
    uk_fanout_sum_bridges,
)
from microcosm.build.uk_runtime.ledger_targets import (
    UK_CROSS_GRAIN_RULE,
    UK_FANOUT_SUM_BRIDGES,
    _assert_uk_fanout_sum_agreement,
    _uk_fanout_sum_controls,
)
from microcosm.build.uk_runtime.national_reconciliation import uk_fanout_sum_controls
from microcosm.calibrate import TargetRegistry, TargetSpec

_EMPLOYMENT = "spi_employment_amount_bands_vs_area"


def _contract() -> dict[str, dict]:
    payload = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text(encoding="utf-8")
    )
    return {target["target_id"]: target for target in payload["targets"]}


def test_each_area_hmrc_row_shares_its_national_bands_measurement():
    contract = _contract()
    fields = UK_CROSS_GRAIN_RULE.signature_fields
    assert {bridge.lower_target_id for bridge in uk_fanout_sum_bridges()} == {
        "hmrc.employment_income.amount",
        "hmrc.employment_income.count",
        "hmrc.self_employment_income.amount",
        "hmrc.self_employment_income.count",
        "hmrc.pension_income.amount",
    }
    for bridge in uk_fanout_sum_bridges():
        lower = contract[bridge.lower_target_id]
        assert not is_uk_fanout_target(lower)
        # A bridge summing several fan-outs measures their sum, so its concept
        # is its own (microcosm#1418: State and private pension income).
        checked = (
            fields
            if len(bridge.higher_target_ids) == 1
            else tuple(field for field in fields if field != "concept")
        )
        for higher_target_id in bridge.higher_target_ids:
            higher = contract[higher_target_id]
            assert is_uk_fanout_target(higher)
            # Taxpayer basis and one counting rule at both grains.
            assert _measurement_signature(higher, checked) == _measurement_signature(
                lower, checked
            )


def test_a_fanout_is_never_an_exact_signature_control():
    declarations = copy.deepcopy(dict(load_uk_cross_grain_declarations()))
    declarations["fanout_sum_bridges"] = []
    violations = uk_cross_grain_coverage_violations(_contract(), declarations)
    assert {
        "kind": "local_target_without_control",
        "target_id": "hmrc.employment_income.amount",
        "where": "contract",
    } in violations


def _band(lower: int, value: float, geography_id: str = "K02000001") -> TargetSpec:
    return TargetSpec(
        name=f"hmrc.spi.employment_income.amount_by_total_income_band.band_{lower}@2025",
        entity="person",
        measure="hmrc/employment_income_income_band",
        value=value,
        period=2025,
        family="hmrc",
        source="test",
        metadata={
            "contract_target_id": (
                "hmrc.spi.employment_income.amount_by_total_income_band"
            ),
            "geography_level": "country",
            "geography_id": geography_id,
        },
    )


def test_the_summed_control_counts_every_band_of_the_full_register():
    registry = TargetRegistry(
        [_band(0, 10.0), _band(50_000, 20.0), _band(1_000_000, 5.0)], country="uk"
    )
    controls = uk_fanout_sum_controls(registry)
    assert controls[_EMPLOYMENT]["control"] == 35.0
    assert len(controls[_EMPLOYMENT]["cells"]) == 3
    assert "spi_self_employment_amount_bands_vs_area" not in controls


_PENSION = "spi_pension_amount_bands_vs_area"
_STATE_PENSION = "hmrc.spi.state_pension.amount_by_total_income_band"
_PRIVATE_PENSION = "hmrc.spi.private_pension_income.amount_by_total_income_band"


def _pension_band(target_id: str, lower: int, value: float) -> TargetSpec:
    return TargetSpec(
        name=f"{target_id}.band_{lower}@2025",
        entity="person",
        measure="hmrc/pension_income_band",
        value=value,
        period=2025,
        family="hmrc",
        source="test",
        metadata={
            "contract_target_id": target_id,
            "geography_level": "country",
            "geography_id": "K02000001",
            "ledger_value_before_alignment": repr(value / 2),
        },
    )


def test_a_bridge_sums_every_fanout_it_names():
    """Pension income by area is the State and private pension bands together
    (microcosm#1418): the control sums both, and refuses with one missing."""

    (bridge,) = [b for b in UK_FANOUT_SUM_BRIDGES if b.bridge_id == _PENSION]
    assert bridge.higher_target_ids == (_STATE_PENSION, _PRIVATE_PENSION)
    assert bridge.control_target_id == f"{_STATE_PENSION}+{_PRIVATE_PENSION}#sum"
    both = TargetRegistry(
        [
            _pension_band(_STATE_PENSION, 0, 10.0),
            _pension_band(_STATE_PENSION, 50_000, 4.0),
            _pension_band(_PRIVATE_PENSION, 0, 20.0),
            _pension_band(_PRIVATE_PENSION, 1_000_000, 6.0),
        ],
        country="uk",
    )
    control = uk_fanout_sum_controls(both)[_PENSION]
    assert control["control"] == 40.0
    assert control["control_before_alignment"] == 20.0
    assert len(control["cells"]) == 4
    one = TargetRegistry([_pension_band(_STATE_PENSION, 0, 10.0)], country="uk")
    with pytest.raises(ValueError, match="carries only"):
        uk_fanout_sum_controls(one)


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (
            {"higher_target_id": "a", "higher_target_ids": ["a", "b"]},
            "exactly one of",
        ),
        ({}, "exactly one of"),
        ({"higher_target_ids": ["a"]}, "two or more distinct"),
        ({"higher_target_ids": ["a", "a"]}, "two or more distinct"),
    ],
)
def test_a_bridge_names_its_fanouts_once(entry, match):
    declarations = {
        "fanout_sum_bridges": [
            {
                "bridge_id": "test",
                "lower_target_id": "c",
                "geography_id": "K02000001",
                "max_factor_shift": 0.02,
                "max_index_shift": 0.01,
                **entry,
            }
        ]
    }
    with pytest.raises(ValueError, match=match):
        uk_fanout_sum_bridges(declarations)


def test_a_surface_binding_the_area_cells_adds_one_country_control_row():
    controls = {_EMPLOYMENT: {"control": 35.0}}
    rows = [{"contract_target_id": "hmrc.employment_income.amount"}]
    bridges, signatures, control_rows = _uk_fanout_sum_controls(rows, controls)
    (bridge,) = bridges
    assert bridge.lower_side == "contract:hmrc.employment_income.amount"
    (synthetic,) = bridge.higher_target_ids
    # The synthetic control groups only through its bridge, and its signature
    # has the contract shape the reconciliation canonicalizes.
    assert _measurement_signature(
        signatures[synthetic], UK_CROSS_GRAIN_RULE.signature_fields
    )
    assert control_rows == [
        {
            "grain": "country",
            "geography_id": "K02000001",
            "target_id": f"contract:{synthetic}",
            "value": 35.0,
            "_output_position": None,
        }
    ]
    assert _uk_fanout_sum_controls([], controls) == ((), {}, [])


def test_a_surface_binding_the_area_cells_without_the_control_is_refused():
    rows = [{"contract_target_id": "hmrc.employment_income.amount"}]
    with pytest.raises(ValueError, match="no summed control"):
        _uk_fanout_sum_controls(rows, {})


def _area(
    level: str, area: str, value: float, before: float | None, **metadata
) -> TargetSpec:
    return TargetSpec(
        name=f"hmrc.employment_income.amount@{area}@2025",
        entity="person",
        measure="hmrc/employment_income/amount",
        value=value,
        period=2025,
        family="hmrc",
        source="test",
        metadata={
            "contract_target_id": "hmrc.employment_income.amount",
            "geography_level": level,
            "geography_id": area,
            **(
                {}
                if before is None
                else {"ledger_value_before_alignment": repr(before)}
            ),
            **metadata,
        },
    )


def _controls(control: float, before: float) -> dict:
    return {_EMPLOYMENT: {"control": control, "control_before_alignment": before}}


def _grown_cells(growth: float) -> TargetRegistry:
    # Cells of 1,000 at source at each grain, grown on the all-ranges index.
    return TargetRegistry(
        [
            _area("constituency", "E14000001", 500.0 * growth, 500.0),
            _area("constituency", "E14000002", 500.0 * growth, 500.0),
            _area("local_authority", "E06000001", 1_000.0 * growth, 1_000.0),
        ],
        country="uk",
    )


def test_the_gate_reads_source_agreement_apart_from_the_two_indices():
    # At source the cells sum to 1,000 against bands of 1,010 (1% apart). The
    # bands grew 15.5% band by band and the cells 15% on the all-ranges index.
    rows = _assert_uk_fanout_sum_agreement(
        _grown_cells(1.15), _controls(1_010.0 * 1.155, 1_010.0)
    )
    assert [(row["geography_level"], row["cells"]) for row in rows] == [
        ("constituency", 2),
        ("local_authority", 1),
    ]
    for row in rows:
        assert row["source_factor"] == pytest.approx(1.01)
        assert row["applied_factor"] == pytest.approx(1.01 * 1.155 / 1.15)
        assert row["index_factor"] == pytest.approx(1.155 / 1.15)
        assert row["max_factor_shift"] == 0.02
        assert row["max_index_shift"] == 0.01


def test_an_index_gap_past_its_bound_is_refused_though_the_sources_agree():
    # The same source agreement, but the bands grew 20% against the cells'
    # 15%: the cells would take 5.4% that no publisher table disagrees by.
    with pytest.raises(ValueError, match="uprating index.*beyond the declared 0.01"):
        _assert_uk_fanout_sum_agreement(
            _grown_cells(1.15), _controls(1_010.0 * 1.2, 1_010.0)
        )


def test_a_moved_cell_that_lost_its_source_value_is_refused():
    registry = TargetRegistry(
        [_area("constituency", "E14000001", 1_150.0, None, uprating_factor="1.15")],
        country="uk",
    )
    with pytest.raises(ValueError, match="no ledger_value_before_alignment"):
        _assert_uk_fanout_sum_agreement(registry, _controls(1_150.0, 1_000.0))
    # A cell no index moved is its own source value.
    registry = TargetRegistry(
        [_area("constituency", "E14000001", 1_000.0, None)], country="uk"
    )
    (row,) = _assert_uk_fanout_sum_agreement(registry, _controls(1_000.0, 1_000.0))
    assert row["source_factor"] == 1.0


def test_a_grain_the_single_winner_rule_reconciles_first_is_still_gated():
    # The constituency cells agree with the bands; the local-authority cells
    # are 3% short at source. The reconciliation parents the local-authority
    # cells on the constituency cells, so its bridge leg would read one.
    (bridge,) = [b for b in UK_FANOUT_SUM_BRIDGES if b.bridge_id == _EMPLOYMENT]
    assert bridge.max_factor_shift == 0.02
    assert bridge.max_index_shift == 0.01
    registry = TargetRegistry(
        [
            _area("constituency", "E14000001", 1_000.0, 1_000.0),
            _area("local_authority", "E06000001", 970.0, 970.0),
        ],
        country="uk",
    )
    with pytest.raises(ValueError, match="local_authority cells.*no longer measure"):
        _assert_uk_fanout_sum_agreement(registry, _controls(1_000.0, 1_000.0))


def _employment_bridge_floored(floor: float) -> list:
    return [
        replace(bridge, min_factor_shift=floor)
        if bridge.bridge_id == _EMPLOYMENT
        else bridge
        for bridge in UK_FANOUT_SUM_BRIDGES
    ]


def test_a_one_way_source_gap_is_bounded_on_both_sides(monkeypatch):
    # The pension bands sit 3.0% to 3.2% above the cells at source, so a
    # factor at or below one is as wrong as one past the top (the independent
    # review of microcosm#1420). The employment bridge stands in, floored at
    # +0.5%: its cells agree with the bands exactly here.
    monkeypatch.setattr(
        ledger_targets, "UK_FANOUT_SUM_BRIDGES", _employment_bridge_floored(0.005)
    )
    with pytest.raises(ValueError, match=r"outside the declared \[1.005, 1.02\]"):
        _assert_uk_fanout_sum_agreement(_grown_cells(1.0), _controls(1_000.0, 1_000.0))
    rows = _assert_uk_fanout_sum_agreement(
        _grown_cells(1.0), _controls(1_010.0, 1_010.0)
    )
    assert {row["min_factor_shift"] for row in rows} == {0.005}


def test_the_pension_bridge_declares_its_one_way_gap():
    (bridge,) = [
        b
        for b in UK_FANOUT_SUM_BRIDGES
        if b.bridge_id == "spi_pension_amount_bands_vs_area"
    ]
    assert bridge.source_factor_bounds == pytest.approx((1.025, 1.037))
    # The other bridges keep a symmetric bound.
    assert all(
        b.min_factor_shift is None
        for b in UK_FANOUT_SUM_BRIDGES
        if b.bridge_id != "spi_pension_amount_bands_vs_area"
    )


@pytest.mark.parametrize("floor", [0.02, 0.05, -0.03, True, "0.01"])
def test_a_floor_outside_the_ceiling_is_refused(floor):
    declarations = {
        "fanout_sum_bridges": [
            {
                "bridge_id": "test",
                "higher_target_id": "a",
                "lower_target_id": "c",
                "geography_id": "K02000001",
                "max_factor_shift": 0.02,
                "max_index_shift": 0.01,
                "min_factor_shift": floor,
            }
        ]
    }
    with pytest.raises(ValueError, match="min_factor_shift in"):
        uk_fanout_sum_bridges(declarations)


def test_cells_the_small_cell_rule_defers_count_on_the_source_side_only():
    # Without the deferred 100 the cells read 4.5% short of the bands at
    # source; counted, they agree, and grown by the bound cells' own index
    # they leave the index factor where it was.
    registry = TargetRegistry(
        [
            _area("constituency", "E14000001", 450.0 * 1.15, 450.0),
            _area("constituency", "E14000002", 450.0 * 1.15, 450.0),
        ],
        country="uk",
    )
    deferred = [
        {
            "target_id": "hmrc.employment_income.amount",
            "geography_level": "constituency",
            "geography_id": "E14000003",
            "value": 100.0,
        }
    ]
    with pytest.raises(ValueError, match="constituency cells.*no longer measure"):
        _assert_uk_fanout_sum_agreement(registry, _controls(1_000.0 * 1.155, 1_000.0))
    (row,) = _assert_uk_fanout_sum_agreement(
        registry, _controls(1_000.0 * 1.155, 1_000.0), deferred_cells=deferred
    )
    assert row["small_cell_deferrals"] == 1
    assert row["source_factor"] == pytest.approx(1.0)
    assert row["index_factor"] == pytest.approx(1.155 / 1.15)
    assert row["cell_total_before_alignment"] == pytest.approx(1_000.0)


def test_a_surface_binding_the_cells_needs_the_control_to_gate_them():
    registry = TargetRegistry(
        [_area("constituency", "E14000001", 1.0, 1.0)], country="uk"
    )
    with pytest.raises(ValueError, match="no summed control"):
        _assert_uk_fanout_sum_agreement(registry, {})
    assert _assert_uk_fanout_sum_agreement(TargetRegistry([], country="uk"), {}) == []


def _committed(resource: str) -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk").joinpath(resource).read_text()
    )


@pytest.mark.parametrize(
    "bridge",
    _committed("cross_grain_declarations.json")["fanout_sum_bridges"],
    ids=lambda bridge: bridge["bridge_id"],
)
def test_each_committed_bridge_holds_its_declared_index_gap(bridge) -> None:
    """The independent review of microcosm#1420, finding 1: the surface's own
    check only runs in a dense build, and these tests use synthetic registers,
    so a declared index gap went stale unseen. The index part of a bridge's
    factor is its national bands' growth over the cells' own index, so it is
    recomputed here from the committed national membership and the cells'
    applier rows (the SPI facts open in 2023; the calibration year is 2025).
    """

    from microcosm.build.uk_runtime import hmrc_uprating

    membership = _committed("target_reference_membership.json")
    contract = {
        target["target_id"]: target
        for target in _committed("uk_population_targets.json")["targets"]
    }
    before = after = 0.0
    for higher in bridge.get("higher_target_ids") or [bridge["higher_target_id"]]:
        for candidate in membership["targets"][higher]["candidates"]:
            assert candidate["status"] == "active", candidate["name"]
            value = float(candidate["resolved_value"])
            uprating = candidate.get("uprating")
            after += value
            before += float(uprating["value_before_uprating"]) if uprating else value
    index = contract[bridge["lower_target_id"]]["uprating_index"]
    concept, measure_id = hmrc_uprating.UK_HMRC_ITL_ALL_BAND_GROWTH_MEASURES[index]
    rows = list(hmrc_uprating._vendored_itl_rows(concept))

    def value_in(year: int) -> float:
        return hmrc_uprating._itl_all_ranges_value(
            rows, measure_id=measure_id, opening_year=year, spec_name=index
        )[0]

    cells = math.fsum(
        weight * value_in(2025 + offset)
        for offset, weight in hmrc_uprating.CALENDAR_YEAR_WINDOW_WEIGHTS.items()
    ) / value_in(2023)
    gap = (after / before) / cells - 1.0
    assert abs(gap) <= bridge["max_index_shift"], (bridge["bridge_id"], gap)


@pytest.mark.parametrize("bridge", uk_fanout_sum_bridges(), ids=lambda b: b.bridge_id)
def test_each_committed_bridge_holds_its_declared_source_gap(bridge) -> None:
    """The re-review of microcosm#1420, D: the source half of a bridge's factor,
    with the pension bridge's one-sided floor, is checked only inside a dense
    build, so a feed re-pin or a change in which cells are deferred could stop
    every dense build with CI green. It is recomputed here, at each grain, from
    the committed registers: the national bands before their index over the
    cells at their published values, the deferred cells that keep a value
    counted, as the surface counts them. The surface also fills an area with no
    fact from the other grain, which this leaves out, so a grain short of an
    area reads a little higher here than in a build."""

    membership = _committed("target_reference_membership.json")
    local = _committed("local_target_reference_membership.json")
    published = {
        row["name"]: float(row["value_before_uprating"])
        for row in local["uprating_holds"]
        if "value_before_uprating" in row
    }
    control = 0.0
    for higher in bridge.higher_target_ids:
        for candidate in membership["targets"][higher]["candidates"]:
            uprating = candidate.get("uprating")
            control += (
                float(uprating["value_before_uprating"])
                if uprating
                else float(candidate["resolved_value"])
            )
    lowest, highest = bridge.source_factor_bounds
    levels = local["targets"][bridge.lower_target_id]["geography_levels"]
    assert set(levels) == {"constituency", "local_authority"}
    for level, payload in levels.items():
        cells = 0.0
        for candidate in payload["candidates"]:
            if candidate["status"] == "active":
                cells += published.get(
                    candidate["name"], float(candidate["resolved_value"])
                )
            elif candidate.get("deferred_value") is not None:
                cells += float(candidate["deferred_value"])
        source = control / cells
        assert lowest <= source <= highest, (bridge.bridge_id, level, source)
