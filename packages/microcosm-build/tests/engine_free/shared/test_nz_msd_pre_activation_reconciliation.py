"""Replay the pre-activation MSD evidence without a donor or country engine."""

from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction

import pytest

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from test_support.paths import paths_for
from tools import generate_nz_msd_pre_activation_reconciliation as generator

_SPEC = paths_for("microcosm-build").package / "src/microcosm/build/nz"


def _report():
    return json.loads((_SPEC / generator.OUTPUT).read_text())


def test_committed_reconciliation_replays_exactly_from_disclosed_source_rows():
    report = _report()
    rates_bytes = (_SPEC / generator.RATE_RESOURCE).read_bytes()
    replay = generator.reconcile(
        report["source_rows"],
        json.loads(rates_bytes),
        json.loads((_SPEC / "chronicle_feed.json").read_text()),
        hashlib.sha256(rates_bytes).hexdigest(),
        report["exhaustive_msd_dimension_audit"],
    )
    assert replay == report
    assert report["stage"] == "pre_activation"
    assert report["status"] == "after_abatement_reconciliation_unavailable"
    assert len(report["source_rows"]) == 19
    assert len(report["exhaustive_msd_dimension_audit"]) == 27
    source_keys = {row["aggregate_fact_key"] for row in report["source_rows"]}
    for comparison in report["comparisons"]:
        for stratum in comparison["strata"]:
            assert set(stratum["recipient_selector"]) == {"aggregate_fact_key"}
            assert stratum["recipient_selector"]["aggregate_fact_key"] in source_keys
            assert "recipient_fact_key" not in stratum


def test_all_headline_categories_and_exposure_gaps_use_exact_arithmetic():
    """Exhaust every disclosed rate option, recipient cell and exposure point."""
    for comparison in _report()["comparisons"]:
        lower = upper = Fraction(0)
        for stratum in comparison["strata"]:
            rates = [
                Fraction(row["gross_weekly_nzd"])
                for row in stratum["available_individual_rate_categories"]
            ]
            count = stratum["recipient_count"]
            assert Fraction(stratum["annual_headline_lower_nzd"]) == (
                count * 52 * min(rates)
            )
            assert Fraction(stratum["annual_headline_upper_nzd"]) == (
                count * 52 * max(rates)
            )
            lower += count * 52 * min(rates)
            upper += count * 52 * max(rates)
        published = Fraction(comparison["published_actual_expense_nzd"])
        assert comparison["category_weighted_after_abatement_total_nzd"] is None
        assert (
            comparison["headline_rate_total_nzd"] is None
            if lower != upper
            else Fraction(comparison["headline_rate_total_nzd"]) == lower
        )
        assert [
            row["full_year_exposure"] for row in comparison["exposure_sensitivity"]
        ] == list(generator.EXPOSURES)
        for row in comparison["exposure_sensitivity"]:
            exposure = Fraction(row["full_year_exposure"])
            assert Fraction(row["headline_lower_nzd"]) == lower * exposure
            assert Fraction(row["headline_upper_nzd"]) == upper * exposure
            assert Fraction(row["lower_minus_published_nzd"]) == (
                lower * exposure - published
            )
            assert Fraction(row["upper_minus_published_nzd"]) == (
                upper * exposure - published
            )


def test_six_published_expense_keys_keep_their_universe_and_period_disclosures():
    report = _report()
    assert {
        row["component"]: row["aggregate_fact_key"]
        for row in report["published_expenses"]
    } == generator.EXPENSE_KEYS
    assert [
        row["component"]
        for row in report["published_expenses"]
        if row["comparison_status"] == "unmodelled_programme_no_comparison"
    ] == ["orphans_unsupported_childs_benefit", "student_allowances"]
    for row in report["source_rows"]:
        if row["record_set_id"] == generator.EXPENSE_RECORD_SET:
            assert row["period_coverage"]["start_date"] == "2024-07-01"
            assert row["period_coverage"]["end_date"] == "2025-06-30"
    assert "no category shares" in report["method"]["headline"]
    assert "without person-level data" in report["method"]["after_abatement"]
    assert "Emergency Benefit" in report["method"]["period_and_universe"]


def test_reconciliation_refuses_missing_source_cells():
    report = _report()
    with pytest.raises(ValueError, match="Expected 12 working-age cells"):
        generator.reconcile(
            report["source_rows"][:-1],
            json.loads((_SPEC / generator.RATE_RESOURCE).read_text()),
            report["feed_identity"],
            report["rates_sha256"],
            report["exhaustive_msd_dimension_audit"],
        )


def test_pinned_feed_regenerates_reconciliation_and_check_mode(tmp_path, monkeypatch):
    artifact_path = os.environ.get("NZ_CHRONICLE_ARTIFACT")
    if not artifact_path:
        pytest.skip("NZ_CHRONICLE_ARTIFACT supplies the staged pinned feed audit")
    artifact = load_ledger_consumer_artifact(artifact_path)
    assert generator.generate(artifact) == _report()
    monkeypatch.setattr(generator, "load_ledger_consumer_artifact", lambda _: artifact)
    output = tmp_path / generator.OUTPUT
    generator.main(["--artifact", artifact_path, "--out", str(output)])
    assert output.read_bytes() == (_SPEC / generator.OUTPUT).read_bytes()
    generator.main(["--artifact", artifact_path, "--out", str(output), "--check"])
    output.write_text("{}\n")
    with pytest.raises(SystemExit, match="Stale MSD pre-activation reconciliation"):
        generator.main(["--artifact", artifact_path, "--out", str(output), "--check"])
