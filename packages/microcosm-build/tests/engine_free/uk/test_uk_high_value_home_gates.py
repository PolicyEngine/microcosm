"""The high-value homes diagnostic gates (microcosm#1240, fix 8)."""

from __future__ import annotations

import copy
import json

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
    HIGH_VALUE_HOME_SCOPE,
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
    # The toy pool's 2.5m to 3.5m band rests on two source households, under
    # the floor of three, so its shipped copy is suppressed (PR #1270 review,
    # finding 6), and the 2m to 2.5m band goes with it, since the total less
    # the shipped bands would give it back (microcosm#1306 review). The total
    # rests on five and ships; empty bands stay as zero.
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
    # Both regions are suppressed, so no gap ships and the evaluated maximum
    # is flagged as resting on a suppressed region (review round 2, finding 1).
    assert shares["gap"] == {"E12000007": None, "E12000008": None}
    assert shares["largest_gap_shipped"] == 0.0
    assert shares["largest_gap_in_suppressed_region"] is True
    # The frame total rests on five sources, so it ships beside the withheld
    # shares, which it cannot give back.
    assert shares["frame_total"] == pytest.approx(6.0)
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
    assert cell["areas_by_source_band"] == {"<3": 1, "3-9": 1, "10-29": 0, "30+": 0}
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


def test_benchmarks_gate_keeps_a_suppressed_regions_gap_out_of_what_it_ships() -> None:
    # The evaluated maximum rests on a suppressed region (PR #1270 review
    # round 2, finding 1): the gate still fails on it, but neither the details
    # nor the message carry the value, which beside the public derived shares
    # would give back that region's frame share.
    regional = {
        "frame": {"E12000007": 0.5, "E12000008": None, "E12000009": None},
        "frame_regions_suppressed_small_cells": ["E12000008", "E12000009"],
        "derived_stock": {"E12000007": 0.52, "E12000008": 0.3, "E12000009": 0.18},
        "gap": {"E12000007": -0.02, "E12000008": None, "E12000009": None},
        "largest_gap_evaluated": 0.17,
        "largest_gap_shipped": 0.02,
        "largest_gap_in_suppressed_region": True,
    }
    summary = {
        "seat_rank_correlation": {
            "over_gbp_2m": {"final": {"spearman_vs_derived_stock": 0.9}}
        },
        "regional_over_2m_share": regional,
    }
    failed = high_value_home_benchmarks_gate(summary, maximum_regional_share_gap=0.1)
    assert not failed.passed and len(failed.failures) == 1
    assert "suppressed" in failed.failures[0] and "0.17" not in failed.failures[0]
    assert "0.17" not in json.dumps(failed.details)
    assert failed.details["regional_over_2m_share"]["largest_gap"] == pytest.approx(
        0.02
    )
    passed = high_value_home_benchmarks_gate(summary, maximum_regional_share_gap=0.2)
    assert passed.passed
    assert passed.details["regional_over_2m_share"]["largest_gap"] == pytest.approx(
        0.02
    )
    # When an unsuppressed region attains the maximum, the value is printed.
    regional.update(
        {
            "gap": {"E12000007": -0.17, "E12000008": None, "E12000009": None},
            "largest_gap_shipped": 0.17,
            "largest_gap_in_suppressed_region": False,
        }
    )
    printed = high_value_home_benchmarks_gate(summary, maximum_regional_share_gap=0.1)
    assert not printed.passed and "0.170" in printed.failures[0]
    assert printed.details["regional_over_2m_share"]["largest_gap"] == pytest.approx(
        0.17
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


# The OBR leg reads the vendored costing rows (microcosm#1278: it had looked
# for an older row shape and reported awaiting facts on every build).


def _evidence_with_obr(obr_rows, **kwargs):
    household = _household()
    return uk_high_value_home_evidence(
        household,
        weights=np.ones(len(household)),
        facts=high_value_homes.load_uk_high_value_home_stock_facts(),
        obr_rows=obr_rows,
        **kwargs,
    )


def test_obr_leg_reads_the_vendored_costing_rows() -> None:
    rows = high_value_homes.load_vendored_obr_hvcts_facts()
    assert rows
    facts = high_value_homes.load_uk_high_value_home_stock_facts()
    summary = _evidence_with_obr(rows).summary
    obr = summary["obr"]
    assert obr["status"] == "available"
    assert (obr["period"], obr["basis"]) == ("2028", "static")
    # OBR Table 1.1, static, 2028-29, at April-2026 valuations.
    assert obr["all_properties"] == {
        "gbp_2m_to_2_5m": 71_000.0,
        "gbp_2_5m_to_3_5m": 54_000.0,
        "gbp_3_5m_to_5m": 25_000.0,
        "over_gbp_5m": 15_000.0,
        "total": 165_000.0,
    }
    assert obr["non_owner_occupied_share"] == pytest.approx(0.4)
    assert obr["non_owner_occupied_share_status"] == "available"
    assert obr["non_owner_occupied_share_fiscal_year"] == "2028"
    assert obr["owner_occupied_sensitivity_total"] == pytest.approx(99_000.0)
    # The toy pool's total rests on five source households, above the floor of
    # three, so it ships with its ratio (the ratios are checked on a larger pool
    # below).
    total = summary["england_owner_occupied_by_band_2026"]["total"]
    assert total == pytest.approx(5.0)
    assert obr["frame_over_owner_occupied_sensitivity"] == pytest.approx(
        total / obr["owner_occupied_sensitivity_total"]
    )
    england = {
        measure: facts.rows_at(
            level="country", measure_id=measure, threshold_id="over_gbp_2m"
        )["E92000001"]
        for measure in ("private_sector_dwellings", "owner_occupied_homes")
    }
    derived = obr["derived_stock_england_over_2m"]
    assert derived["private_sector_over_obr_all_properties"] == pytest.approx(
        england["private_sector_dwellings"] / 165_000.0
    )
    # The stock build sets the owner-occupied share of England's 2m+ stock to
    # one less the same OBR share, so the owner-occupied ratio would repeat
    # the private-sector one: only the counts ship beside it.
    assert "owner_occupied_over_obr_owner_occupied" not in derived
    assert england["owner_occupied_homes"] / 99_000.0 == pytest.approx(
        england["private_sector_dwellings"] / 165_000.0, rel=1e-6
    )
    assert derived["owner_occupied_homes"] == england["owner_occupied_homes"]


def _pool_by_band_2026(sources_by_band: dict[str, int]) -> pd.DataFrame:
    """England owner-occupiers, one distinct source each, valued inside a band."""
    facts = high_value_homes.load_uk_high_value_home_stock_facts()
    two_m = {t.threshold_id: t for t in facts.thresholds}["over_gbp_2m"]
    deflator = two_m.basis_gbp / two_m.nominal_gbp
    midpoints = {
        "gbp_2m_to_2_5m": 2.2e6,
        "gbp_2_5m_to_3_5m": 3.0e6,
        "gbp_3_5m_to_5m": 4.0e6,
        "over_gbp_5m": 6.0e6,
    }
    values = [
        midpoints[band] * deflator
        for band, n in sources_by_band.items()
        for _ in range(n)
    ]
    n = len(values)
    return pd.DataFrame(
        {
            "region_code": ["E12000007"] * n,
            "constituency_code": ["E14000001"] * n,
            "local_authority_code": ["E09000020"] * n,
            "tenure_type": ["OWNED_OUTRIGHT"] * n,
            "main_residence_value": values,
            "source_household_id": list(range(1, n + 1)),
        }
    )


def test_obr_leg_ratios_read_the_shipped_total() -> None:
    household = _pool_by_band_2026(
        {"gbp_2m_to_2_5m": 12, "gbp_2_5m_to_3_5m": 11, "gbp_3_5m_to_5m": 10}
    )
    weights = np.full(len(household), 100.0)
    summary = uk_high_value_home_evidence(
        household,
        weights=weights,
        facts=high_value_homes.load_uk_high_value_home_stock_facts(),
        obr_rows=high_value_homes.load_vendored_obr_hvcts_facts(),
    ).summary
    bands = summary["england_owner_occupied_by_band_2026"]
    assert bands["total"] == pytest.approx(3_300.0)
    assert summary["england_bands_suppressed_small_cells"] == []
    obr = summary["obr"]
    assert obr["frame_over_owner_occupied_sensitivity"] == pytest.approx(
        3_300.0 / 99_000.0
    )
    assert obr["frame_over_all_properties_ratio"]["gbp_2m_to_2_5m"] == pytest.approx(
        1_200.0 / 71_000.0
    )
    assert obr["frame_over_all_properties_ratio"]["over_gbp_5m"] == 0.0


@pytest.mark.parametrize(
    ("period", "basis", "total"),
    [("2028", "post_behaviour", 156_000.0), ("2030", "static", 167_000.0)],
)
def test_obr_leg_selects_the_year_and_the_behaviour_basis(
    period: str, basis: str, total: float
) -> None:
    obr = _evidence_with_obr(
        high_value_homes.load_vendored_obr_hvcts_facts(),
        obr_period=period,
        obr_basis=basis,
    ).summary["obr"]
    assert obr["status"] == "available"
    assert obr["all_properties"]["total"] == pytest.approx(total)


def test_obr_leg_names_rows_that_match_no_year() -> None:
    obr = _evidence_with_obr(
        high_value_homes.load_vendored_obr_hvcts_facts(), obr_period="2031"
    ).summary["obr"]
    assert obr == {"status": "no_matching_rows", "period": "2031", "basis": "static"}
    assert _evidence_with_obr(None).summary["obr"] == {
        "status": "awaiting_vendored_facts"
    }


def test_obr_leg_names_the_year_its_share_is_from() -> None:
    # The note states the share once, for 2028-29: a later year's comparison
    # says so rather than applying it silently.
    obr = _evidence_with_obr(
        high_value_homes.load_vendored_obr_hvcts_facts(), obr_period="2030"
    ).summary["obr"]
    assert obr["status"] == "available"
    assert obr["non_owner_occupied_share"] == pytest.approx(0.4)
    assert obr["non_owner_occupied_share_fiscal_year"] == "2028"
    assert obr["owner_occupied_sensitivity_total"] == pytest.approx(167_000.0 * 0.6)


def _matching_row(rows: list[dict], band: str) -> dict:
    return next(
        row
        for row in rows
        if row["concept"] == "obr.hvcts_costing.properties_in_scope"
        and row["period"]["value"] == 2028
        and row["dimensions"].get("behaviour_basis") == "static"
        and row["dimensions"].get("property_value_band") == band
    )


@pytest.mark.parametrize("defect", ["unit", "band", "duplicate"])
def test_obr_leg_reports_malformed_rows_without_raising(defect: str) -> None:
    # The leg runs in the terminal phase after the solve, so a bad re-vendor
    # is reported as a status, never raised.
    rows = copy.deepcopy(list(high_value_homes.load_vendored_obr_hvcts_facts()))
    row = _matching_row(rows, "over_gbp_5m")
    if defect == "unit":
        row["unit"] = "thousand"
    elif defect == "band":
        row["dimensions"]["property_value_band"] = "over_gbp_10m"
    else:
        rows.append(copy.deepcopy(row))
    obr = _evidence_with_obr(rows).summary["obr"]
    assert obr["status"] == "malformed_rows"
    assert len(obr["problems"]) == 1
    assert "all_properties" not in obr


@pytest.mark.parametrize(
    ("defect", "status", "missing"),
    [
        ("concept", "incomplete_rows", ["over_gbp_5m"]),
        ("geography", "incomplete_rows", ["over_gbp_5m"]),
        ("total", "incomplete_rows", ["total"]),
        ("calendar_year", "no_matching_rows", None),
    ],
)
def test_obr_leg_selects_by_concept_geography_and_period_type(
    defect: str, status: str, missing: list[str] | None
) -> None:
    rows = copy.deepcopy(list(high_value_homes.load_vendored_obr_hvcts_facts()))
    if defect == "concept":
        _matching_row(rows, "over_gbp_5m")["concept"] = "obr.hvcts_costing.other"
    elif defect == "geography":
        _matching_row(rows, "over_gbp_5m")["geography"]["id"] = "K02000001"
    elif defect == "total":
        total = next(
            row
            for row in rows
            if row["concept"] == "obr.hvcts_costing.properties_in_scope"
            and row["period"]["value"] == 2028
            and row["dimensions"].get("behaviour_basis") == "static"
            and "property_value_band" not in row["dimensions"]
        )
        rows.remove(total)
    else:
        for row in rows:
            if row["period"]["value"] == 2028:
                row["period"]["type"] = "calendar_year"
    obr = _evidence_with_obr(rows).summary["obr"]
    assert obr["status"] == status
    assert obr.get("missing_bands") == missing
    assert "owner_occupied_sensitivity_total" not in obr


@pytest.mark.parametrize(
    ("defect", "status"),
    [
        ("duplicate", "ambiguous_rows"),
        ("unit", "malformed_row"),
        ("range", "malformed_row"),
    ],
)
def test_obr_leg_reports_a_share_it_cannot_read(defect: str, status: str) -> None:
    rows = copy.deepcopy(list(high_value_homes.load_vendored_obr_hvcts_facts()))
    share = next(
        r for r in rows if r["concept"] == "obr.hvcts_costing.non_owner_occupied_share"
    )
    if defect == "duplicate":
        rows.append(copy.deepcopy(share))
    elif defect == "unit":
        share["unit"] = "count"
    else:
        share["value"] = 140.0
    obr = _evidence_with_obr(rows).summary["obr"]
    # The band comparison stands; only the owner-occupied sensitivity goes.
    assert obr["status"] == "available"
    assert obr["non_owner_occupied_share"] is None
    assert obr["non_owner_occupied_share_status"] == status
    assert "owner_occupied_sensitivity_total" not in obr


@pytest.mark.parametrize(
    ("field", "value", "status"),
    [
        ("value", None, "malformed_rows"),
        ("value", "n/a", "malformed_rows"),
        ("value", float("nan"), "malformed_rows"),
        ("value", -5.0, "malformed_rows"),
        ("period", "2028", "incomplete_rows"),
        ("dimensions", ["static"], "incomplete_rows"),
    ],
)
def test_obr_leg_reports_unreadable_fields_without_raising(
    field: str, value, status: str
) -> None:
    # A missing, non-numeric, NaN or negative count, or a field that is not a
    # mapping, is reported, never raised and never "available"
    # (microcosm#1306 review).
    rows = copy.deepcopy(list(high_value_homes.load_vendored_obr_hvcts_facts()))
    _matching_row(rows, "over_gbp_5m")[field] = value
    obr = _evidence_with_obr(rows).summary["obr"]
    assert obr["status"] == status
    assert "all_properties" not in obr


@pytest.mark.parametrize("value", [None, "40%", float("nan"), 100.0])
def test_obr_leg_reports_an_unreadable_share_without_raising(value) -> None:
    rows = copy.deepcopy(list(high_value_homes.load_vendored_obr_hvcts_facts()))
    share = next(
        r for r in rows if r["concept"] == "obr.hvcts_costing.non_owner_occupied_share"
    )
    share["value"] = value
    obr = _evidence_with_obr(rows).summary["obr"]
    assert obr["status"] == "available"
    assert obr["non_owner_occupied_share"] is None
    assert obr["non_owner_occupied_share_status"] == "malformed_row"


def test_small_cell_suppression_withholds_a_thin_total() -> None:
    # One band on two sources and the rest empty: the total is that band,
    # so both go (microcosm#1306 review).
    household = _pool_by_band_2026({"gbp_2m_to_2_5m": 2})
    summary = uk_high_value_home_evidence(
        household,
        weights=np.full(len(household), 100.0),
        facts=high_value_homes.load_uk_high_value_home_stock_facts(),
    ).summary
    bands = summary["england_owner_occupied_by_band_2026"]
    assert bands["gbp_2m_to_2_5m"] is None and bands["total"] is None
    assert bands["gbp_3_5m_to_5m"] == 0.0
    assert summary["regional_over_2m_share"]["frame_total"] is None


def test_small_cell_suppression_protects_the_withheld_bands_together() -> None:
    # Bands on 40, 1 and 1 sources: the two thin bands rest together on 2, so
    # the total less the shipped bands would give their sum back; the next
    # smallest band goes too, and the total, on 42, ships.
    household = _pool_by_band_2026(
        {"gbp_2m_to_2_5m": 40, "gbp_2_5m_to_3_5m": 1, "gbp_3_5m_to_5m": 1}
    )
    summary = uk_high_value_home_evidence(
        household,
        weights=np.full(len(household), 100.0),
        facts=high_value_homes.load_uk_high_value_home_stock_facts(),
    ).summary
    bands = summary["england_owner_occupied_by_band_2026"]
    assert summary["england_bands_suppressed_small_cells"] == [
        "gbp_2_5m_to_3_5m",
        "gbp_2m_to_2_5m",
        "gbp_3_5m_to_5m",
    ]
    assert bands["over_gbp_5m"] == 0.0
    assert bands["total"] == pytest.approx(4_200.0)


def test_small_cell_suppression_applies_to_shares_summing_to_one() -> None:
    from microcosm.build.uk_runtime.high_value_home_gates import _suppress_small_cells

    sources = {
        "E12000007": set(range(30)),
        "E12000008": {100},
        "E12000009": {200},
    }
    shares = {"E12000007": 0.9, "E12000008": 0.06, "E12000009": 0.04}
    shipped, withheld = _suppress_small_cells(shares, sources)
    # Shares sum to one, so the two thin regions (two sources together) take
    # the third with them.
    assert withheld == ["E12000007", "E12000008", "E12000009"]
    assert set(shipped.values()) == {None}
    sources["E12000008"] = set(range(100, 112))
    shipped, withheld = _suppress_small_cells(shares, sources)
    assert withheld == ["E12000008", "E12000009"]
    assert shipped["E12000007"] == pytest.approx(0.9)


def test_benchmarks_gate_outcome_does_not_depend_on_the_obr_leg() -> None:
    facts = _facts(
        {"E14000001": 500.0, "E14000002": 200.0, "E14000003": 0.0},
        {"E12000007": 500.0, "E12000008": 200.0},
    )
    household = _household()
    rows = high_value_homes.load_vendored_obr_hvcts_facts()
    with_obr = uk_high_value_home_evidence(
        household, weights=np.ones(len(household)), facts=facts, obr_rows=rows
    ).summary
    without_obr = uk_high_value_home_evidence(
        household, weights=np.ones(len(household)), facts=facts
    ).summary
    assert with_obr["obr"]["status"] == "available"
    # A ratio to a public count would give back a suppressed band, so the
    # suppressed toy bands carry no ratio.
    ratios = with_obr["obr"]["frame_over_all_properties_ratio"]
    assert ratios["gbp_2m_to_2_5m"] is None and ratios["gbp_2_5m_to_3_5m"] is None
    for parameters in ({}, {"minimum_rank_correlation": 1.1}):
        assert (
            high_value_home_benchmarks_gate(with_obr, **parameters).passed
            == high_value_home_benchmarks_gate(without_obr, **parameters).passed
        )


def test_evidence_and_both_gates_name_the_main_residence_scope() -> None:
    # Until the coverage decision (microcosm#1240 fix 6, tracked on
    # microcosm#1278), every reading of these figures says they cover
    # owner-occupiers' main residences only.
    facts = _facts(
        {"E14000001": 500.0, "E14000002": 200.0, "E14000003": 0.0},
        {"E12000007": 500.0, "E12000008": 200.0},
    )
    summary = uk_high_value_home_evidence(
        _household(), weights=np.ones(12), facts=facts
    ).summary
    assert HIGH_VALUE_HOME_SCOPE["label"] == "owner_occupied_main_residences_only"
    assert summary["scope"] == dict(HIGH_VALUE_HOME_SCOPE)
    assert high_value_home_support_gate(summary).details["scope"] == dict(
        HIGH_VALUE_HOME_SCOPE
    )
    assert high_value_home_benchmarks_gate(summary).details["scope"] == dict(
        HIGH_VALUE_HOME_SCOPE
    )
