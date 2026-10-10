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
    _assert_uk_fanout_sum_factors,
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


def _receipt(factor: float) -> dict:
    return {
        "groups": [
            {
                "inconsistency_id": "x",
                "bridge_id": _EMPLOYMENT,
                "winning_grain": "country",
                "lower_grain": "constituency",
                "legs": [{"leg": "all", "declared_factor": factor}],
            }
        ]
    }


def test_a_bridge_factor_beyond_its_tolerance_is_refused():
    (bridge,) = [b for b in UK_FANOUT_SUM_BRIDGES if b.bridge_id == _EMPLOYMENT]
    assert bridge.max_factor_shift == 0.02
    assert _assert_uk_fanout_sum_factors(_receipt(1.015))[0]["declared_factor"] == (
        1.015
    )
    with pytest.raises(ValueError, match="no longer measure one quantity"):
        _assert_uk_fanout_sum_factors(_receipt(1.03))
