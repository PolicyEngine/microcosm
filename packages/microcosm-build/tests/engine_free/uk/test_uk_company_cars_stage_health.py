"""Stage-health check of the company cars stage (dashboard #316)."""

from __future__ import annotations

import copy

import pytest

from microcosm.build.uk_runtime.receipt_counts import extreme_count, rounded_count
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

PARAMETERS = {"maximum_pay_identity_error_gbp": 0.05}
PASSING = {
    "stage": "company_cars",
    "holders": {
        "frs_channel": 1200,
        "spi_channel_tape": 300,
        "spi_channel_twin_pay": 20,
        "spi_dropped_without_epb": {
            "count": 40,
            "weighted": 30_000.0,
            "weighted_list_price_gbp": 1.2e9,
        },
        "weighted": 900_000.0,
    },
    "spi_carve_out": {"holders": 300, "max_abs_pay_identity_error_gbp": 0.01},
    "checks": {
        "violations": {
            "non_finite_values": 0,
            "negative_values": 0,
            "holder_without_list_price": 0,
            "inputs_without_a_car": 0,
            "unknown_fuel_type": 0,
            "emitting_car_without_co2": 0,
            "electric_car_with_co2": 0,
            "private_fuel_on_an_electric_car": 0,
            "contributions_above_list_price": 0,
            "electric_range_outside_1_to_50_g_km": 0,
            "spi_benefit_above_epb": 0,
        },
        "frs_holders_without_pay": rounded_count(5),
    },
}


def _gate(evidence, parameters=PARAMETERS):
    return uk_stage_health_gate(
        evidence=evidence,
        stage=evidence["stage"],
        check="company_cars",
        parameters=parameters,
    )


def test_a_coherent_receipt_passes() -> None:
    result = _gate(PASSING)
    assert result.passed
    assert result.details["holders"]["spi_channel_tape"] == 300


def test_any_invariant_break_fails_and_is_named() -> None:
    broken = copy.deepcopy(PASSING)
    broken["checks"]["violations"]["electric_car_with_co2"] = 2
    result = _gate(broken)
    assert not result.passed
    assert any("electric_car_with_co2" in failure for failure in result.failures)


def test_a_receipt_that_counts_no_invariant_fails() -> None:
    empty = copy.deepcopy(PASSING)
    empty["checks"]["violations"] = {}
    assert not _gate(empty).passed


def test_spi_pay_off_the_engine_s_benefits_fails() -> None:
    off = copy.deepcopy(PASSING)
    off["spi_carve_out"]["max_abs_pay_identity_error_gbp"] = 0.5
    result = _gate(off)
    assert not result.passed
    assert any("the pay the stage received" in failure for failure in result.failures)


def test_a_banded_invariant_break_fails() -> None:
    broken = copy.deepcopy(PASSING)
    broken["checks"]["violations"]["spi_benefit_above_epb"] = extreme_count(1)
    assert not _gate(broken).passed


def test_a_receipt_without_its_checks_is_refused() -> None:
    bare = copy.deepcopy(PASSING)
    del bare["checks"]
    with pytest.raises(ValueError, match="checks"):
        _gate(bare)
