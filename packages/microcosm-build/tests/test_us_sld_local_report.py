"""Tests for the SLD validation protocol and sidecar writer (populace#625)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.sld_local_doctrine import (
    US_SLD_LOCAL_SOLVE_DOCTRINE,
    solve_us_sld_chamber_under_doctrine,
)
from microcosm.build.us_runtime.sld_local_report import (
    achieved_vs_target_table,
    honest_boundaries_statement,
    median_income_validation,
    render_boundaries_markdown,
    statewide_coherence_report,
    weighted_median,
    write_sld_sidecar,
)
from microcosm.build.us_runtime.sld_local_solver import SldDistrictProblem
from microcosm.build.us_runtime.sld_local_targets import (
    SldTargetFacts,
    resolve_money_income_recipe,
)


def _problem(area_code: str, district_code: str) -> SldDistrictProblem:
    matrix = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0, 1.0],
        ]
    )
    base = np.full(4, 10.0)
    return SldDistrictProblem(
        area_type="sldu",
        area_code=area_code,
        state_fips="49",
        district_code=district_code,
        matrix=matrix,
        targets=matrix @ (base * 1.5),
        target_frame=pd.DataFrame(
            {
                "area_type": ["sldu"] * 2,
                "area_code": [area_code] * 2,
                "metric": ["households", "age_0_to_4"],
            }
        ),
        household_ids=np.array(
            [int(district_code) * 100 + offset for offset in range(1, 5)]
        ),
        base_weights=base,
    )


def _facts() -> SldTargetFacts:
    return SldTargetFacts(
        calibration=pd.DataFrame(
            {
                "area_type": ["sldu"],
                "area_code": ["610U900US49001"],
                "metric": ["households"],
                "value": [60.0],
            }
        ),
        validation=pd.DataFrame(
            {
                "area_type": ["sldu", "sldu"],
                "area_code": ["610U900US49001", "610U900US49009"],
                "metric": ["median_household_income"] * 2,
                "value": [50_000.0, 70_000.0],
            }
        ),
        source_path="facts.jsonl",
        source_sha256="0" * 64,
        geography_vintages=("2024_state_legislative_districts",),
    )


def _chambers():
    problems = [
        _problem("610U900US49001", "001"),
        _problem("610U900US49002", "002"),
    ]
    return [solve_us_sld_chamber_under_doctrine(problems, epochs=64)]


def test_weighted_median_crosses_at_half_mass():
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1])) == 2.0
    assert weighted_median(np.array([10.0, 20.0]), np.array([3.0, 1.0])) == 10.0
    with pytest.raises(ValueError, match="non-empty"):
        weighted_median(np.array([]), np.array([]))


def test_achieved_vs_target_covers_every_district_cell():
    table = achieved_vs_target_table(_chambers())
    assert len(table) == 4  # 2 districts x 2 targets
    assert set(table["area_code"]) == {"610U900US49001", "610U900US49002"}
    assert {"target", "final_estimate", "abs_relative_error"} <= set(table.columns)


def test_median_validation_reports_gaps_and_flags():
    chambers = _chambers()
    incomes = {
        household: 50_000.0 + 10_000.0 * index
        for index, household in enumerate(
            [*chambers[0].district_results[0].problem.household_ids]
        )
    }
    incomes.update(
        {
            household: 200_000.0
            for household in chambers[0].district_results[1].problem.household_ids
        }
    )
    table = median_income_validation(chambers, _facts(), incomes)
    # Only district 001 has a published median among solved districts.
    assert table["area_code"].tolist() == ["610U900US49001"]
    row = table.iloc[0]
    assert row["published_median"] == 50_000.0
    assert 50_000.0 <= row["achieved_median"] <= 80_000.0
    assert bool(row["review_flag"]) == (abs(row["relative_gap"]) > 0.15)


def test_statewide_coherence_sums_targets_artifact_and_solved():
    table = statewide_coherence_report(_chambers())
    households = table[table["metric"] == "households"].iloc[0]
    assert households["state_fips"] == "49"
    # Two districts x 4 households x weight 10 at the artifact anchor.
    assert households["artifact_weight_sum"] == pytest.approx(80.0)
    assert households["district_target_sum"] == pytest.approx(120.0)
    assert households["solved_vs_target_ratio"] == pytest.approx(1.0, abs=0.05)


def test_boundaries_statement_declares_the_contract():
    chambers = _chambers()
    statement = honest_boundaries_statement(
        chambers=chambers,
        facts=_facts(),
        recipe_resolution=resolve_money_income_recipe(
            [
                "employment_income_before_lsr",
                "self_employment_income_before_lsr",
                "taxable_interest_income",
                "social_security_retirement",
            ],
            [],
        ),
        membership_gate_details={
            "method_counts": {"tract_exact": 6, "puma_cd_county_draw": 2},
            "unassigned_share": 0.0,
        },
        doctrine_record=US_SLD_LOCAL_SOLVE_DOCTRINE.as_record(),
        zero_support_districts={"sldu": ("610U900US49003",)},
    )
    assert "NOT district-calibrated" in statement["calibrated_surface"]["statement"]
    assert statement["vintages"]["district_boundaries"] == [
        "2024_state_legislative_districts"
    ]
    assert statement["doctrine"]["target_loss_cap"] == 10.0
    assert statement["small_area_tails"]["zero_support_districts"]["sldu"] == [
        "610U900US49003"
    ]
    assert statement["small_area_tails"]["thin_districts"]  # 4-row districts
    markdown = render_boundaries_markdown(statement)
    assert "# SLD layer: declared boundaries" in markdown
    assert "tract_exact: 6" in markdown
    assert "supplemental_security_income" in markdown


def test_sidecar_writer_emits_hashed_bundle(tmp_path):
    chambers = _chambers()
    incomes = {
        int(household): 40_000.0
        for chamber in chambers
        for result in chamber.district_results
        for household in result.problem.household_ids
    }
    shas = write_sld_sidecar(
        tmp_path,
        chambers=chambers,
        facts=_facts(),
        recipe_resolution=resolve_money_income_recipe(
            [
                "employment_income_before_lsr",
                "self_employment_income_before_lsr",
                "taxable_interest_income",
                "social_security_retirement",
            ],
            [],
        ),
        membership_gate_details={"method_counts": {}, "unassigned_share": 0.0},
        doctrine_record=US_SLD_LOCAL_SOLVE_DOCTRINE.as_record(),
        zero_support_districts={"sldu": ()},
        money_income_by_household_id=incomes,
    )
    assert set(shas) == {
        "sld_local_weights.csv.gz",
        "sld_achieved_vs_target.csv.gz",
        "sld_local_diagnostics.json",
        "sld_local_boundaries.json",
        "sld_local_boundaries.md",
    }
    for filename, sha in shas.items():
        path = tmp_path / filename
        assert path.exists()
        assert len(sha) == 64
    weights = pd.read_csv(tmp_path / "sld_local_weights.csv.gz")
    assert list(weights.columns) == [
        "area_type",
        "area_code",
        "household_id",
        "weight",
        "weight_source",
    ]
    assert len(weights) == 8
    diagnostics = json.loads((tmp_path / "sld_local_diagnostics.json").read_text())
    assert diagnostics["doctrine"]["max_weight_ratio"] == 100.0
    assert diagnostics["target_facts"]["source_sha256"] == "0" * 64
    assert len(diagnostics["chambers"]) == 1
    # Deterministic bytes: rewriting produces identical hashes (gzip mtime=0).
    shas_again = write_sld_sidecar(
        tmp_path,
        chambers=chambers,
        facts=_facts(),
        recipe_resolution=resolve_money_income_recipe(
            [
                "employment_income_before_lsr",
                "self_employment_income_before_lsr",
                "taxable_interest_income",
                "social_security_retirement",
            ],
            [],
        ),
        membership_gate_details={"method_counts": {}, "unassigned_share": 0.0},
        doctrine_record=US_SLD_LOCAL_SOLVE_DOCTRINE.as_record(),
        zero_support_districts={"sldu": ()},
        money_income_by_household_id=incomes,
    )
    assert shas == shas_again


def test_sidecar_hashes_match_on_disk_bytes(tmp_path):
    import hashlib

    chambers = _chambers()
    incomes = {
        int(household): 40_000.0
        for chamber in chambers
        for result in chamber.district_results
        for household in result.problem.household_ids
    }
    shas = write_sld_sidecar(
        tmp_path,
        chambers=chambers,
        facts=_facts(),
        recipe_resolution=resolve_money_income_recipe(
            [
                "employment_income_before_lsr",
                "self_employment_income_before_lsr",
                "taxable_interest_income",
                "social_security_retirement",
            ],
            [],
        ),
        membership_gate_details={"method_counts": {}, "unassigned_share": 0.0},
        doctrine_record=US_SLD_LOCAL_SOLVE_DOCTRINE.as_record(),
        zero_support_districts={"sldu": ()},
        money_income_by_household_id=incomes,
    )
    for filename, declared in shas.items():
        actual = hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        assert actual == declared, filename


def test_boundaries_json_is_stable_and_serializable(tmp_path):
    statement = honest_boundaries_statement(
        chambers=_chambers(),
        facts=_facts(),
        recipe_resolution=resolve_money_income_recipe(
            [
                "employment_income_before_lsr",
                "self_employment_income_before_lsr",
                "taxable_interest_income",
                "social_security_retirement",
            ],
            [],
        ),
        membership_gate_details={"method_counts": {}, "unassigned_share": 0.0},
        doctrine_record=US_SLD_LOCAL_SOLVE_DOCTRINE.as_record(),
        zero_support_districts={},
    )
    encoded = json.dumps(statement, sort_keys=True, allow_nan=False)
    assert json.dumps(json.loads(encoded), sort_keys=True, allow_nan=False) == encoded
