"""Tests for the declared UK capital gains target facts.

The published populace-uk surface carries no capital gains targets, which
leaves the gains distribution unanchored (1.47m CGT taxpayers against HMRC's
378k). These tests pin the declared facts, their provenance, and the coverage
requirement that makes their absence a build failure rather than a silence.
"""

import pytest

from populace.build.uk_runtime.fiscal_targets import (
    UK_CGT_REQUIRED_COLUMNS,
    UK_CGT_TARGET_COVERAGE_REQUIREMENTS,
    UK_CGT_TARGET_SPECS,
    UK_FISCAL_TARGET_REGISTRY,
)


def test_declares_gains_total_and_taxpayer_count():
    """Both facts are needed: a total alone does not pin the distribution."""
    assert {spec.name for spec in UK_CGT_TARGET_SPECS} == {
        "hmrc/capital_gains_total",
        "hmrc/cgt_taxpayers",
    }


def test_every_fact_carries_provenance():
    """A fact without a citation is not a fact."""
    for spec in UK_CGT_TARGET_SPECS:
        assert "gov.uk" in spec.source
        assert spec.family == "hmrc"


def test_facts_match_hmrc_2023_24_outturn():
    """Values are HMRC's published outturn, not uprated approximations."""
    by_name = {spec.name: spec for spec in UK_CGT_TARGET_SPECS}
    assert by_name["hmrc/capital_gains_total"].value == pytest.approx(65.9e9)
    assert by_name["hmrc/cgt_taxpayers"].value == pytest.approx(378_000)
    assert all(spec.period == 2023 for spec in UK_CGT_TARGET_SPECS)


def test_measures_are_declared_columns():
    """The registry refuses callables, so measures must be prepared columns."""
    measures = {spec.measure for spec in UK_CGT_TARGET_SPECS}
    assert measures == set(UK_CGT_REQUIRED_COLUMNS)
    assert all(isinstance(spec.measure, str) for spec in UK_CGT_TARGET_SPECS)


def test_facts_are_person_grain():
    """UK measures are person-level, matching the hmrc_calibration convention.

    The weights stay household-level; the frame carries them as household
    ``Weights`` while the constraint rows live on the person table.
    """
    assert all(spec.entity == "person" for spec in UK_CGT_TARGET_SPECS)


def test_measure_columns_are_prepared_names():
    """Measures name prepared columns, not raw model inputs."""
    by_name = {spec.name: spec for spec in UK_CGT_TARGET_SPECS}
    assert by_name["hmrc/capital_gains_total"].measure == "uk_cgt_measure_gains_amount"
    assert by_name["hmrc/cgt_taxpayers"].measure == "uk_cgt_measure_taxpayer_count"


def test_registry_is_uk_and_content_addressed():
    assert UK_FISCAL_TARGET_REGISTRY.country == "uk"
    assert UK_FISCAL_TARGET_REGISTRY.version


def test_coverage_requires_both_facts():
    """A build that drops either fact must fail the gate, not pass quietly."""
    (requirement,) = UK_CGT_TARGET_COVERAGE_REQUIREMENTS
    assert requirement.min_matches == 2
    assert set(requirement.accepted_names) == {
        "hmrc/capital_gains_total",
        "hmrc/cgt_taxpayers",
    }
