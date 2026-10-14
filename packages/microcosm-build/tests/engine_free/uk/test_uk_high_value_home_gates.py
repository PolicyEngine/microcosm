"""The high-value homes diagnostic gates (microcosm#1240, fix 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import high_value_homes
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.calibration_run import (
    UK_LOCAL_DIAGNOSTIC_GATE_SCOPE,
    uk_local_gate_scope_exclusions,
)
from microcosm.build.uk_runtime.high_value_home_gates import (
    high_value_home_benchmarks_gate,
    high_value_home_support_gate,
    uk_high_value_home_evidence,
)
from microcosm.build.uk_runtime.release_certification import (
    UK_CERTIFICATION_EXCLUDED_GATE_IDS,
)


def _facts(
    seats: dict[str, float], regions: dict[str, float]
) -> high_value_homes.UKHighValueHomeStockFacts:
    rows = []
    for threshold_id in high_value_homes.THRESHOLD_IDS:
        for code, value in seats.items():
            rows.append(_row("constituency", code, threshold_id, value))
        for code, value in regions.items():
            rows.append(_row("region", code, threshold_id, value))
    return high_value_homes.UKHighValueHomeStockFacts(
        rows=tuple(rows),
        inputs_sha256="",
        rows_sha256="",
        parameters={
            "thresholds_gbp_at_reference_month": {
                "over_gbp_1m": 1e6,
                "over_gbp_1_5m": 1.5e6,
                "over_gbp_2m": 2e6,
            },
            "frame_basis_thresholds": {
                "over_gbp_1m": 0.9e6,
                "over_gbp_1_5m": 1.35e6,
                "over_gbp_2m": 1.8e6,
            },
            "basis_year": 2024,
            "stock_date": "2025-12-31",
            "obr_non_owner_occupied_share": 0.4,
        },
        inputs={},
        generator={},
        benchmarks={
            "voa_band_h_by_constituency": {
                code: value * 2 for code, value in seats.items()
            }
        },
    )


def _row(level: str, code: str, threshold_id: str, value: float) -> dict:
    return {
        "geography": {"level": level, "id": code},
        "observed_measure": {"source_measure_id": "owner_occupied_homes"},
        "dimensions": {"microcosm.value_threshold": threshold_id},
        "value": value,
    }


def _household() -> pd.DataFrame:
    # Three seats in two English regions; seat A is prime, seat C has nothing.
    seats = ["E14000001"] * 6 + ["E14000002"] * 4 + ["E14000003"] * 2
    regions = ["E12000007"] * 6 + ["E12000008"] * 6
    values = [
        3.0e6,
        2.5e6,
        2.0e6,
        1.9e6,
        0.5e6,
        2.2e6,
        2.0e6,
        1.4e6,
        0.3e6,
        0.0,
        0.2e6,
        0.4e6,
    ]
    tenure = (
        ["OWNED_OUTRIGHT"] * 4
        + ["RENT_PRIVATELY", "OWNED_WITH_MORTGAGE"]
        + ["OWNED_OUTRIGHT"] * 3
        + ["RENT_FROM_COUNCIL", "OWNED_OUTRIGHT", "OWNED_OUTRIGHT"]
    )
    source = [1, 2, 3, 4, 5, 6, 1, 7, 8, 9, 10, 11]
    return pd.DataFrame(
        {
            "region_code": regions,
            "constituency_code": seats,
            "local_authority_code": ["E09000020"] * 6 + ["E07000001"] * 6,
            "tenure_type": tenure,
            "main_residence_value": values,
            "source_household_id": source,
        }
    )


def test_evidence_counts_bands_shares_correlations_and_support() -> None:
    facts = _facts(
        {"E14000001": 500.0, "E14000002": 200.0, "E14000003": 0.0},
        {"E12000007": 500.0, "E12000008": 200.0},
    )
    household = _household()
    weights = np.ones(len(household))
    evidence = uk_high_value_home_evidence(household, weights=weights, facts=facts)
    summary = evidence.summary
    bands = summary["england_owner_occupied_by_band_2026"]
    # The toy pool's bands rest on fewer than ten source households, so the
    # shipped copies are suppressed (PR #1270 review, finding 6); the total
    # over two suppressed bands stays, and empty bands stay as zero.
    assert bands["gbp_2m_to_2_5m"] is None and bands["gbp_2_5m_to_3_5m"] is None
    assert bands["gbp_3_5m_to_5m"] == 0.0 and bands["over_gbp_5m"] == 0.0
    assert bands["total"] == pytest.approx(6.0)
    assert summary["england_bands_suppressed_small_cells"] == [
        "gbp_2_5m_to_3_5m",
        "gbp_2m_to_2_5m",
    ]
    assert summary["obr"]["status"] == "awaiting_vendored_facts"
    shares = summary["regional_over_2m_share"]
    assert shares["frame"]["E12000007"] is None
    assert "E12000007" in shares["frame_regions_suppressed_small_cells"]
    assert shares["derived_stock"]["E12000007"] == pytest.approx(5 / 7)
    assert shares["largest_gap_evaluated"] == pytest.approx(abs(5 / 6 - 5 / 7))
    rho = summary["seat_rank_correlation"]["over_gbp_2m"]["final"]
    assert rho["spearman_vs_derived_stock"] == pytest.approx(1.0)
    assert rho["spearman_vs_voa_band_h"] == pytest.approx(1.0)
    assert (
        summary["seat_rank_correlation"]["over_gbp_2m"]["initial"]["status"]
        == "unavailable"
    )
    support = evidence.support
    seat_a = support[
        (support["geography_level"] == "constituency")
        & (support["area_code"] == "E14000001")
        & (support["threshold_id"] == "over_gbp_2m")
    ].iloc[0]
    assert seat_a["nonzero_source_households"] == 5
    assert seat_a["max_source_share"] == pytest.approx(0.2)
    # Seat C has no owner above the thresholds, so it has no support row at all.
    assert "E14000003" not in set(support["area_code"])


def test_support_gate_assesses_only_areas_with_derived_stock_and_reports_bands() -> (
    None
):
    facts = _facts(
        {"E14000001": 500.0, "E14000002": 200.0, "E14000003": 0.0},
        {"E12000007": 500.0, "E12000008": 200.0},
    )
    evidence = uk_high_value_home_evidence(
        _household(), weights=np.ones(12), facts=facts
    )
    result = high_value_home_support_gate(
        evidence.summary, minimum_distinct_sources=3, minimum_target_homes=100.0
    )
    cell = result.details["cells"]["constituency:over_gbp_2m"]
    assert cell["assessed_areas"] == 2
    assert cell["areas_by_source_band"] == {"<10": 2, "10-29": 0, "30+": 0}
    assert cell["areas_below_floor"] == 1  # seat B carries one source above 2m
    assert not result.passed
    assert any("constituency:over_gbp_2m" in failure for failure in result.failures)
    # No area code or cell value reaches the shipped details.
    flat = str(result.details)
    assert "E14000001" not in flat and "E09000020" not in flat
    passing = high_value_home_support_gate(
        evidence.summary,
        minimum_distinct_sources=1,
        minimum_target_homes=100.0,
        dominance_share=1.0,
    )
    assert passing.passed


def test_benchmarks_gate_fails_on_low_correlation_or_regional_gap() -> None:
    summary = {
        "seat_rank_correlation": {
            "over_gbp_2m": {"final": {"spearman_vs_derived_stock": 0.4}}
        },
        "regional_over_2m_share": {
            "frame": {"E12000007": 0.5},
            "derived_stock": {"E12000007": 0.55},
            "gap": {"E12000007": -0.05},
        },
    }
    failed = high_value_home_benchmarks_gate(
        summary, minimum_rank_correlation=0.6, maximum_regional_share_gap=0.1
    )
    assert not failed.passed and len(failed.failures) == 1
    passed = high_value_home_benchmarks_gate(
        summary, minimum_rank_correlation=0.3, maximum_regional_share_gap=0.1
    )
    assert passed.passed
    assert passed.details["regional_over_2m_share"]["largest_gap"] == pytest.approx(
        0.05
    )


def test_gates_are_declared_diagnostic_bound_and_outside_the_release_contract() -> None:
    entries = {entry.id: entry for entry in load_country_spec("uk").gates.gates}
    for gate_id in UK_LOCAL_DIAGNOSTIC_GATE_SCOPE:
        entry = entries[gate_id]
        assert entry.criticality == "diagnostic"
        assert entry.phase == "terminal"
        assert entry.gate in UK_GATE_REGISTRY
        binding = UK_GATE_REGISTRY[entry.gate]
        assert set(entry.parameters) <= binding.parameter_keys
        assert binding.required_artifacts(entry.parameters) == {
            "uk_high_value_home_evidence"
        }
        assert gate_id in UK_CERTIFICATION_EXCLUDED_GATE_IDS
        assert gate_id in uk_local_gate_scope_exclusions()
