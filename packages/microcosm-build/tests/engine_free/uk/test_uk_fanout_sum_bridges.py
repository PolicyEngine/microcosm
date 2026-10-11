"""A banded national target, summed, controls its area cells (#1123)."""

from __future__ import annotations

import copy
import json
from importlib import resources as importlib_resources

import pytest

from microcosm.build.cross_grain import _measurement_signature
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
    }
    for bridge in uk_fanout_sum_bridges():
        higher = contract[bridge.higher_target_id]
        lower = contract[bridge.lower_target_id]
        assert is_uk_fanout_target(higher)
        assert not is_uk_fanout_target(lower)
        # Taxpayer basis and one counting rule at both grains.
        assert _measurement_signature(higher, fields) == _measurement_signature(
            lower, fields
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


def test_a_surface_binding_the_cells_needs_the_control_to_gate_them():
    registry = TargetRegistry(
        [_area("constituency", "E14000001", 1.0, 1.0)], country="uk"
    )
    with pytest.raises(ValueError, match="no summed control"):
        _assert_uk_fanout_sum_agreement(registry, {})
    assert _assert_uk_fanout_sum_agreement(TargetRegistry([], country="uk"), {}) == []
