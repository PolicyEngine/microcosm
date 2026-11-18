"""First-processed returns and annual tax returns have different populations."""

import json
import subprocess
import sys
from hashlib import sha256

import pytest
from hypothesis import given
from hypothesis import strategies as st

from microcosm.build.us_runtime.filing_season import filing_season_comparison
from microcosm.build.us_runtime.fiscal_targets import (
    compile_us_fiscal_target_registry,
    us_fiscal_target_exclusion_receipt,
)
from test_support.microcosm_build.us_fiscal_targets import (
    _dynamic_ledger_fact,
    _soi_eitc_filing_season_total_fact,
    packaged_reference_facts,
)
from test_support.paths import paths_for


def _fact(year, measure, value, *, week=30, band="all", unit="count"):
    record_set = f"irs_soi.ty{year}.filing_season_week{week}.all_returns"
    fact = _dynamic_ledger_fact(
        source_record_id=f"{record_set}.{band}.{measure}",
        source_name="irs_soi",
        measure_id=measure,
        value=value,
        period_value=year,
        layout_record_set_id=record_set,
        dimensions={
            "filing_season_week": week,
            "processing_cutoff": f"{year + 1}-07-25",
            "income_range": band,
        },
    )
    fact["period"]["type"] = "tax_year"
    fact["observed_measure"]["unit"] = unit
    return fact


def _compare(rows):
    return filing_season_comparison(rows, first_year=2024, last_year=2025, week=30)


def test_annual_compiler_keeps_reviewed_eitc_and_ignores_partial_observations():
    legacy = _soi_eitc_filing_season_total_fact(
        measure_id="total_earned_income_credit_returns", value=80
    )
    annual = _dynamic_ledger_fact(
        source_record_id="irs_soi.ty2025.table_2_1.itemized_all_returns.all.limited_state_local_taxes_amount",
        source_name="irs_soi",
        measure_id="limited_state_local_taxes_amount",
        value=1200,
        period_value=2025,
        dimensions={"income_range": "all"},
        layout_record_set_id="irs_soi.ty2025.table_2_1.itemized_all_returns",
    )
    partial = [
        _fact(2024, "total_earned_income_credit_returns", 73, week=21),
        _fact(2025, "limited_state_local_taxes_amount", 912, unit="usd"),
        _fact(2025, "return_count", 94),
    ]
    facts = [*packaged_reference_facts(), annual, legacy, *partial]
    result = compile_us_fiscal_target_registry(
        facts, target_period=2025, allow_unaged_dollar_targets=True
    )
    ids = {spec.name for spec in result.specs}
    assert annual["lineage"]["source_record_id"] in ids
    assert legacy["lineage"]["source_record_id"] in ids
    excluded = {f["lineage"]["source_record_id"] for f in partial}
    assert ids.isdisjoint(excluded)
    receipt = us_fiscal_target_exclusion_receipt(facts, target_period=2025)
    assert (
        set(receipt["rules"]["partial_filing_season"]["source_record_ids"]) == excluded
    )


def test_report_cli_compares_only_matched_weeks_and_distinguishes_absence(tmp_path):
    facts = [
        _fact(2024, "return_count", 100),
        _fact(2025, "return_count", 125),
        _fact(2025, "return_count", 40, week=21),
        _fact(2024, "total_itemized_deductions_returns", 20),
        _fact(2025, "total_itemized_deductions_returns", 30),
        _fact(2025, "new_deduction_amount", 0, unit="usd"),
        _fact(2024, "new_deduction_amount", 99, week=21, unit="usd"),
    ]
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text("".join(json.dumps(fact) + "\n" for fact in facts))
    out = tmp_path / "report.json"
    tool = paths_for("microcosm-build").repository / "tools/report_us_filing_season.py"
    subprocess.run(
        [
            sys.executable,
            str(tool),
            str(path),
            "--out",
            str(out),
            "--week",
            "30",
            "--first-year",
            "2024",
            "--last-year",
            "2025",
        ],
        check=True,
    )
    report = json.loads(out.read_text())
    assert report["facts_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert report["last_processing_cutoff"] == "2026-07-25"
    rows = {r["measure"]: r for r in report["rows"]}
    assert rows["return_count"]["difference"] == 25
    assert rows["total_itemized_deductions_returns"]["difference"] == 10
    assert rows["total_itemized_deductions_returns"]["change_percentage_points"] == 4
    assert rows["new_deduction_amount"]["last"] == 0
    assert rows["new_deduction_amount"]["first"] is None
    assert rows["new_deduction_amount"]["difference"] is None


def test_report_rejects_incomparable_inputs():
    first, last = (_fact(y, "return_count", 100) for y in (2024, 2025))
    with pytest.raises(ValueError, match="Both tax years"):
        _compare([first, _fact(2025, "return_count", 100, week=21)])
    with pytest.raises(ValueError, match="Duplicate"):
        _compare([first, last, first])
    with pytest.raises(ValueError, match="units"):
        _compare([first, _fact(2025, "return_count", 100, unit="usd")])
    with pytest.raises(ValueError, match="all-return count"):
        _compare([first, last, _fact(2025, "total_itemized_deductions_returns", 101)])
    outside = _fact(2025, "total_itemized_deductions_returns", 60)
    outside["dimensions"]["processing_cutoff"] = "2026-07-22"
    with pytest.raises(ValueError, match="cutoff"):
        _compare([first, last, outside])


@given(st.integers(0, 100), st.integers(0, 100))
def test_percentage_point_change_uses_each_years_own_denominator(first, last):
    rows = [
        _fact(2024, "return_count", 200),
        _fact(2025, "return_count", 100),
        _fact(2024, "total_itemized_deductions_returns", first),
        _fact(2025, "total_itemized_deductions_returns", last),
    ]
    report = _compare(rows)
    assert report == _compare(reversed(rows))
    result = report["rows"][1]
    assert result["change_percentage_points"] == pytest.approx(last - first / 2)
