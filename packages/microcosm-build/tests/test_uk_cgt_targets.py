"""Tests for the UK capital gains Ledger target references.

The published populace-uk surface carries no capital gains targets, which
leaves the gains distribution unanchored (1.47m CGT taxpayers against HMRC's
378k). These tests pin the declared facts, their provenance, and the coverage
requirement that makes their absence a build failure rather than a silence.
"""

import json
from pathlib import Path

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.uk_runtime.fiscal_targets import (
    UK_CGT_REQUIRED_COLUMNS,
    UK_CGT_TARGET_COVERAGE_REQUIREMENTS,
    UK_CGT_TARGET_SPECS,
    UK_FISCAL_TARGET_REGISTRY,
)

FIXTURE_FEED_ROWS = (
    Path(__file__).parent / "fixtures" / "uk_target_reference_feed_rows.jsonl"
)


def _compiled_cgt_registry():
    spec = load_country_spec("uk")
    references = [
        reference
        for reference in spec.target_references
        if reference.name in {"hmrc.cgt.gains_total", "hmrc.cgt.taxpayers_total"}
    ]
    facts = [
        json.loads(line)
        for line in FIXTURE_FEED_ROWS.read_text().splitlines()
        if line.strip()
    ]
    return compile_ledger_target_references(facts, references, country="uk")


def test_inline_cgt_target_specs_are_retired():
    assert UK_CGT_TARGET_SPECS == ()
    assert len(UK_FISCAL_TARGET_REGISTRY) == 0


def test_compiled_references_declare_gains_total_and_taxpayer_count():
    registry = _compiled_cgt_registry()

    assert {spec.name for spec in registry.specs} == {
        "hmrc.cgt.gains_total",
        "hmrc.cgt.taxpayers_total",
    }


def test_every_compiled_fact_carries_provenance():
    """A fact without a citation is not a fact."""
    for spec in _compiled_cgt_registry().specs:
        assert "gov.uk" in spec.source
        assert spec.family == "hmrc_cgt"


def test_compiled_facts_match_hmrc_2023_24_outturn():
    by_name = {spec.name: spec for spec in _compiled_cgt_registry().specs}
    assert by_name["hmrc.cgt.gains_total"].value == 65_937_000_000
    assert by_name["hmrc.cgt.taxpayers_total"].value == 378_000
    assert all(spec.period == 2025 for spec in by_name.values())
    assert all(
        spec.metadata["ledger_fact_period"] == "2023" for spec in by_name.values()
    )


def test_measures_are_declared_columns():
    """The registry refuses callables, so measures must be prepared columns."""
    measures = {spec.measure for spec in _compiled_cgt_registry().specs}
    assert measures == {"hmrc/capital_gains_total", "hmrc/cgt_taxpayers"}
    assert set(UK_CGT_REQUIRED_COLUMNS) == {
        "uk_cgt_measure_gains_amount",
        "uk_cgt_measure_taxpayer_count",
    }


def test_facts_are_person_grain():
    """UK measures are person-level, matching the hmrc_calibration convention.

    The weights stay household-level; the frame carries them as household
    ``Weights`` while the constraint rows live on the person table.
    """
    assert all(spec.entity == "person" for spec in _compiled_cgt_registry().specs)


def test_registry_is_uk_and_content_addressed():
    assert UK_FISCAL_TARGET_REGISTRY.country == "uk"
    assert UK_FISCAL_TARGET_REGISTRY.version


def test_coverage_requires_both_facts():
    """A build that drops either fact must fail the gate, not pass quietly."""
    (requirement,) = UK_CGT_TARGET_COVERAGE_REQUIREMENTS
    assert requirement.min_matches == 2
    assert set(requirement.accepted_names) == {
        "hmrc.cgt.gains_total",
        "hmrc.cgt.taxpayers_total",
    }
