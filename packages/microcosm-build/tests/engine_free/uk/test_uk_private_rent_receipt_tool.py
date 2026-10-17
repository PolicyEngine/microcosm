"""The private-rent receipt's measures and disclosure control (microcosm#1262).

The receipt reads a licensed build, so these tests drive its measures with
small synthetic frames, facts and problems.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_TOOL = _TEST_PATHS.repository / "tools/uk_private_rent_receipt.py"
_SPEC = importlib.util.spec_from_file_location("uk_private_rent_receipt", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
tool = importlib.util.module_from_spec(_SPEC)
# Its dataclasses resolve their annotations through ``sys.modules``.
sys.modules[_SPEC.name] = tool
_SPEC.loader.exec_module(tool)


def _frame(rows: list[tuple]) -> pd.DataFrame:
    columns = (
        "weight",
        "region",
        "tenure_type",
        "rent",
        "num_bedrooms",
        "brma",
        "la_code",
        "source_key",
    )
    frame = pd.DataFrame(rows, columns=columns)
    frame["subrent"] = 0.0
    frame["housing_benefit_reported"] = 0.0
    frame["channel"] = "frs"
    return frame


def test_unweighted_shares_are_floored_and_small_sides_withheld() -> None:
    assert tool.floored_share(0, 100) == 0.0
    assert tool.floored_share(37, 300) == 0.12
    assert tool.floored_share(5, 100) == "<10"
    assert tool.floored_share(95, 100) == "<10"
    assert tool.floored_share(1, 0) is None


def test_weighted_shares_count_distinct_source_households_on_both_sides() -> None:
    weight = np.ones(40)
    many = np.array([f"s{i}" for i in range(40)], dtype=object)
    mask = np.arange(40) < 20
    assert tool.weighted_share(mask, weight, many) == 0.5
    # Twenty clones of three source households are still three households.
    few = np.array([f"s{i % 3}" if i < 20 else f"t{i}" for i in range(40)])
    assert tool.weighted_share(mask, weight, few) == "<10"
    assert tool.weighted_share(~mask, weight, few) == "<10"


def test_one_small_cell_takes_a_complement_with_it() -> None:
    cells = {
        "B": {"source_households": 4, "mean_monthly_rent": 900.0},
        "C": {"source_households": 40, "mean_monthly_rent": 1000.0},
        "D": {"source_households": 25, "mean_monthly_rent": 1100.0},
    }
    out = tool.suppress_cells(cells)
    assert out["B"] == {"withheld": "<10"}
    assert out["D"] == {"withheld": "complement"}
    assert out["C"] == {"source_households": 40, "mean_monthly_rent": 1000.0}
    two_small = tool.suppress_cells(
        {**cells, "C": {"source_households": 6, "mean_monthly_rent": 1000.0}}
    )
    assert two_small["B"] == two_small["C"] == {"withheld": "<10"}
    assert two_small["D"]["source_households"] == 30


def test_rent_rows_are_scored_by_family_and_nation() -> None:
    metadata = (
        {"contract_target_id": "ons.rent.private_rent", "area_code": "E06000001"},
        {"contract_target_id": "ons.rent.private_rent", "area_code": "W06000001"},
        {"contract_target_id": "ons.rent.private_rent_brma", "area_code": "S33000001"},
        {"contract_target_id": "ons.tenure.private_rent", "area_code": "E06000001"},
    )
    matrix = sparse.csr_array(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 5.0]])
    )
    weighting = tool.Weighting(
        name="design",
        matrix=matrix,
        weights=np.array([105.0, 70.0]),
        names=("e", "w", "s", "t"),
        targets=np.array([100.0, 100.0, 175.0, 1.0]),
        metadata=metadata,
    )
    summary, rows = tool.rent_row_fit(weighting)
    assert [row["target_name"] for row in rows] == ["e", "w", "s"]
    assert summary["all"]["rows"] == 3
    assert summary["all"]["rows_over_25pct"] == 1
    assert summary["ons.rent.private_rent England"]["share_within_10pct"] == 1.0
    assert summary["ons.rent.private_rent Wales"]["rows_over_25pct"] == 1
    assert summary["ons.rent.private_rent_brma Scotland"]["median_abs_error"] == 0.0


def _pipr_fact(area: str, level: str, month: str, bedrooms: str, value: float) -> str:
    return json.dumps(
        {
            "dimensions": {"bedrooms": bedrooms},
            "geography": {"id": area, "level": level},
            "layout": {"measure_id": "ons_rent_private_rent"},
            "period": {"type": "month", "value": month},
            "source": {"vintage": "pipr_test_edition"},
            "value": value,
        }
    )


def test_pipr_means_need_all_twelve_months_and_follow_the_aliases(tmp_path) -> None:
    lines = [
        _pipr_fact("E08000038", "local_authority", f"2025-{m:02d}", "two bed", 600 + m)
        for m in range(1, 13)
    ]
    lines += [
        _pipr_fact("E06000001", "local_authority", f"2025-{m:02d}", "one bed", 500)
        for m in range(1, 12)
    ]
    lines.append(
        _pipr_fact("E06000002", "local_authority", "2025-01", "two bed", 1).replace(
            "pipr_test_edition", "another_edition"
        )
    )
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text("\n".join(lines) + "\n")
    means = tool.pipr_bedroom_means(tmp_path, edition="pipr_test_edition")
    # Barnsley's 2025 code reads as the build's E08000016.
    assert means == {("la", "E08000016", "C"): pytest.approx(606.5)}


def test_renters_are_gb_private_with_positive_rent_and_carry_a_category() -> None:
    frame = _frame(
        [
            (
                1.0,
                "LONDON",
                "RENT_PRIVATELY",
                12_000.0,
                1,
                "INNER_EAST_LONDON",
                "E09000002",
                "a",
            ),
            (
                1.0,
                "LONDON",
                "RENT_PRIVATELY",
                0.0,
                2,
                "INNER_EAST_LONDON",
                "E09000002",
                "b",
            ),
            (
                1.0,
                "LONDON",
                "OWNED_OUTRIGHT",
                0.0,
                3,
                "INNER_EAST_LONDON",
                "E09000002",
                "c",
            ),
            (
                1.0,
                "NORTHERN_IRELAND",
                "RENT_PRIVATELY",
                6_000.0,
                4,
                "BELFAST",
                "N09000003",
                "d",
            ),
            (1.0, "WALES", "RENT_PRIVATELY", 9_000.0, 4, "CARDIFF", "W06000015", "e"),
        ]
    )
    renters = tool.private_renters(frame)
    assert list(renters["source_key"]) == ["a", "e"]
    assert list(renters["category"]) == ["B", "E"]
    assert list(renters["nation"]) == ["England", "Wales"]


def test_a_mix_matching_the_census_has_no_distance() -> None:
    renters = tool.with_carriers(
        _frame(
            [
                (1.0, "WALES", "RENT_PRIVATELY", 9_000.0, 2, brma, "W06000015", key)
                for key, brma in [(f"c{i}", "CARDIFF") for i in range(9)]
                + [(f"s{i}", "SWANSEA") for i in range(3)]
            ]
        )
    ).assign(category="C", nation="Wales")
    counts = {("WALES", "C"): {"CARDIFF": 300.0, "SWANSEA": 100.0}}
    summary, audit = tool.brma_mix(renters, counts)
    assert summary == {"Wales": 0.0}
    assert audit["WALES"]["census_share"] == {"CARDIFF": 0.75, "SWANSEA": 0.25}


def test_the_matched_change_reads_each_row_against_its_spine_value() -> None:
    rows = [
        (1.0, "WALES", "RENT_PRIVATELY", 11_000.0, 2, "CARDIFF", "W06000015", f"s{i}")
        for i in range(12)
    ]
    frame = _frame(rows).assign(spine_rent=10_000.0, spine_brma="CARDIFF")
    frame.loc[0, "brma"] = "SWANSEA"
    out = tool.matched_change(frame)
    assert out["gb_rows_redrawn"] == 1.0
    assert out["gb_rows_whose_brma_changed"] == "<10"
    assert out["region_level_at_design_weights"] == {"WALES": 1.1}


def test_a_pool_table_without_the_spine_values_is_refused(tmp_path) -> None:
    path = tmp_path / "pool.npz"
    np.savez(path, weight=np.ones(2), region=np.array(["WALES", "WALES"]))
    with pytest.raises(ValueError, match="lacks"):
        tool.load_pool(path)


# ----------------------------------------------------------------------------
# The quotable summary (microcosm#1305 review)


def _renter(key: str, *, region="LONDON", bedrooms=1, weight=1.0, rent=12_000.0):
    return (
        weight,
        region,
        "RENT_PRIVATELY",
        rent,
        bedrooms,
        "INNER_EAST_LONDON",
        "E09000002",
        key,
    )


def _summary(frame: pd.DataFrame, **kwargs) -> dict:
    summary, _ = tool.measure_households(frame, pipr={}, counts={}, **kwargs)
    return summary


def test_a_thin_region_category_cell_reads_under_ten_and_withholds_a_complement():
    rows = (
        [_renter(f"b{i}", bedrooms=1) for i in range(12)]
        + [_renter(f"c{i}", bedrooms=2) for i in range(15)]
        + [_renter(f"d{i}", bedrooms=3) for i in range(4)]
        + [_renter(f"e{i}", bedrooms=4) for i in range(20)]
    )
    london = _summary(_frame(rows))["rents_by_cell"]["region"]["LONDON"]
    assert london["D"] == {"withheld": "<10"}
    assert london["B"] == {"withheld": "complement"}
    assert london["C"]["source_households"] == 20
    assert london["all"]["mean_monthly_rent"] == 1000.0


def test_district_groups_need_ten_cells_of_ten_households():
    pipr = {("la", f"E0900000{i}", "B"): 1_000.0 for i in range(1, 10)}
    rows = [
        (*_renter(f"k{area}-{i}")[:6], f"E0900000{area}", f"k{area}-{i}")
        for area in range(1, 10)
        for i in range(12)
    ]
    rows += [(*_renter(f"thin{i}")[:6], "E09000001", f"thin{i}") for i in range(3)]
    summary, audit = tool.measure_households(_frame(rows), pipr=pipr, counts={})
    group = summary["pipr_by_district_and_bedrooms"]["la England"]
    # Nine districts pass the household floor: too few to summarise.
    assert group == {
        "withheld": "fewer than 10 cells",
        "cells_under_10_source_households": 0,
    }
    assert len(audit["pipr_by_district_and_bedrooms"]) == 9


def test_the_top_one_percent_share_needs_ten_rows_and_ten_households():
    # 2,000 renters: the top 20 rows are clones of two households.
    rows = [_renter(f"r{i}") for i in range(1_980)]
    rows += [_renter(f"top{i % 2}", weight=50.0) for i in range(20)]
    weights = _summary(_frame(rows))["weights"]
    assert weights["top_1pct_weight_share"] == "<10"
    rows[-20:] = [_renter(f"top{i}", weight=50.0) for i in range(20)]
    assert isinstance(_summary(_frame(rows))["weights"]["top_1pct_weight_share"], float)
    assert _summary(_frame([_renter(f"r{i}") for i in range(9)]))["weights"] == {
        "withheld": "<10"
    }


def test_coherence_counts_households_not_clone_rows():
    # 8 of 800 households, 25 clones each, report HB above their rent.
    frame = _frame([_renter(f"h{i}") for i in range(800) for _ in range(25)])
    frame.loc[
        frame["source_key"].isin([f"h{i}" for i in range(8)]),
        "housing_benefit_reported",
    ] = 20_000.0
    out = _summary(frame)["coherence"]["housing_benefit_reported_above_rent"]
    assert out == {"weighted_share": "<10", "unweighted_share": "<10"}


def test_spine_values_count_frs_channel_households_only():
    frame = _frame(
        [_renter(f"f{i}") for i in range(4)] + [_renter(f"s{i}") for i in range(6)]
    )
    frame.loc[4:, "channel"] = "spi"
    every = _summary(frame)["rents_by_cell"]["nation"]["England"]
    assert every["source_households"] == 10
    frs = _summary(frame, frs_carriers=True)["rents_by_cell"]["nation"]["England"]
    assert frs == {"withheld": "<10"}


def test_licensed_outputs_are_refused_inside_the_repository(tmp_path):
    with pytest.raises(SystemExit, match="inside the repository"):
        tool.outside_repository(tool.REPO_ROOT / "audit-1262", what="the audit")
    assert (
        tool.outside_repository(tmp_path / "audit-1262", what="the audit")
        == (tmp_path / "audit-1262").resolve()
    )
