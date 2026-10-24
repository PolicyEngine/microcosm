"""NZ feed operands, reviewed factors, and the observed-quarter re-declaration."""

from __future__ import annotations

import copy
import json
import os
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from microcosm.build.ledger_artifact import (
    LexicalFloat,
    load_ledger_consumer_artifact,
    validate_chronicle_feed_pin,
)
from microcosm.build.ledger_targets import fact_matches_selector
from microcosm.build.reference_factors import (
    replay_reference_factor,
    resolve_reference_factor,
    validate_factor_context,
)
from test_support.paths import paths_for
from tools.generate_nz_reference_aging import (
    _OPERANDS,
    build_reference_aging,
)

_PATHS = paths_for("microcosm-build")
_SPEC = _PATHS.package / "src/microcosm/build/nz"
_EXPECTED = {
    "ird_ty2024_to_ty2027_low": 1115056,
    "ird_ty2024_to_ty2027_central": 1117109,
    "ird_ty2024_to_ty2027_high": 1119158,
    "p1_ty2026_to_ty2027_low": 1026690,
    "p1_ty2026_to_ty2027_central": 1028580,
    "p1_ty2026_to_ty2027_high": 1030467,
    "observed_ty2026_over_ty2024": 1086069,
}


def _document():
    return json.loads((_SPEC / "reference_aging.json").read_text())


def _entry(factor_id="ird_ty2024_to_ty2027_central"):
    return next(
        entry
        for entry in _document()["factors"]
        if entry["declaration"]["id"] == factor_id
    )


def _fraction(rational):
    return Fraction(rational["numerator"], rational["denominator"])


@pytest.fixture(scope="module")
def artifact():
    root = os.environ.get("NZ_CHRONICLE_ARTIFACT")
    if not root:
        pytest.skip("NZ_CHRONICLE_ARTIFACT is not set")
    loaded = load_ledger_consumer_artifact(Path(root), lexical_floats=True)
    validate_chronicle_feed_pin(loaded, _document()["feed_identity"])
    return loaded


@pytest.fixture(scope="module")
def operand_facts(artifact):
    keys = {f"ledger.aggregate_fact.v2:{suffix}" for _, _, suffix, _ in _OPERANDS}
    keys.add("ledger.aggregate_fact.v2:aba7d3d54489b068116e6a25")
    return tuple(fact for fact in artifact.facts if fact["aggregate_fact_key"] in keys)


def test_generated_aging_resource_reproduces_pinned_feed_and_all_receipts(artifact):
    assert build_reference_aging(artifact) == _document()
    assert set(validate_factor_context(_document())) == set(_EXPECTED)
    identity = _document()["feed_identity"]
    assert set(identity) == {
        "source_commit",
        "facts_sha256",
        "manifest_sha256",
        "artifact_schema_version",
        "consumer_fact_schema_versions",
        "consumer_fact_schema_sha256",
        "fact_row_count",
    }
    for entry in _document()["factors"]:
        assert entry["declaration"]["feed_identity"] == identity
        assert entry["receipt"]["feed_identity"] == identity


@pytest.mark.parametrize("factor_id", list(_EXPECTED))
def test_all_six_factors_and_observed_diagnostic_are_exact(factor_id, operand_facts):
    entry = _entry(factor_id)
    declaration = entry["declaration"]
    expected = {"numerator": _EXPECTED[factor_id], "denominator": 1000000}
    factor, receipt = resolve_reference_factor(
        declaration, operand_facts, _document()["feed_identity"]
    )
    assert factor == expected == declaration["expected_operational_factor"]
    assert receipt == entry["receipt"]
    assert receipt["operation"] == "projected_period_average_ratio"
    assert receipt["operation_version"] == 1
    assert receipt["subperiods_per_year"] == 4
    assert _fraction(receipt["denominator_mean"]) == (
        Fraction("1670.0725") if factor_id.startswith("p1_") else Fraction("1537.7225")
    )
    assert declaration["purpose"] == (
        "diagnostic" if factor_id.startswith("observed_") else "profile_input"
    )
    replay_reference_factor(
        declaration, operand_facts, _document()["feed_identity"], factor, receipt
    )


def test_each_reviewed_operand_resolves_once_with_exact_lexical_evidence(artifact):
    for line, period, suffix, expected in _OPERANDS:
        fact = artifact.facts[line - 1]
        assert fact["aggregate_fact_key"] == f"ledger.aggregate_fact.v2:{suffix}"
        assert fact["period"]["value"] == period
        assert fact["value"].lexical_token == expected
    checked = set()
    for entry in _document()["factors"]:
        declaration = entry["declaration"]
        slots = [*declaration["source_schedule"], *declaration["target_schedule"]]
        if declaration["growth"]:
            slots.append(declaration["growth"])
        for slot in slots:
            identity = json.dumps(
                {"selector": slot["selector"], "validation": slot["validation"]},
                sort_keys=True,
            )
            if identity in checked:
                continue
            checked.add(identity)
            matches = [
                fact
                for fact in artifact.facts
                if fact_matches_selector(fact, slot["selector"])
            ]
            if not matches:
                assert slot in declaration["target_schedule"]
                assert slot["projection_offset"] > 0
                continue
            (fact,) = matches
            validation = slot["validation"]
            assert fact["observed_measure"]["unit"] == validation["unit"]
            assert fact["source"]["source_sha256"] == validation["source_sha256"]
            assert fact["source"]["vintage"] == validation["source_vintage"]
            assert fact["entity"]["role"] == validation["entity_role"]
            assert fact["assertion"] == validation["assertion"]
            assert fact["provenance_class"] == validation["provenance_class"]
            for key in ("start_date", "end_date"):
                assert (
                    fact["period_coverage"][key] == validation["period_coverage"][key]
                )


def test_treasury_fy2027_source_column_uses_chronicle_2026_and_2027_refuses(
    operand_facts,
):
    declaration = _entry()["declaration"]
    assert declaration["growth"]["selector"]["period_value"] == 2026
    assert declaration["growth"]["validation"]["source_column_id"] == (
        "year_ending_30_june_2027"
    )
    declaration["growth"]["selector"]["period_value"] = 2027
    with pytest.raises(ValueError, match="resolve exactly once"):
        resolve_reference_factor(
            declaration, operand_facts, _document()["feed_identity"]
        )


def test_p1_high_rounds_once_to_1030467_and_old_truncation_refuses(operand_facts):
    declaration = _entry("p1_ty2026_to_ty2027_high")["declaration"]
    assert declaration["source_period"] == 2026
    assert [
        row["selector"]["period_value"] for row in declaration["source_schedule"]
    ] == ["2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1"]
    declaration["expected_operational_factor"]["numerator"] = 1030466
    with pytest.raises(ValueError, match="expected factor mismatch"):
        resolve_reference_factor(
            declaration, operand_facts, _document()["feed_identity"]
        )


def test_root_enclosure_certifies_no_rounded_quarters_and_irrational_root():
    for factor_id in list(_EXPECTED)[:6]:
        entry = _entry(factor_id)
        receipt = entry["receipt"]
        delta = Fraction(entry["declaration"]["growth"]["scenario_delta"])
        radicand = 1 + (Fraction("2.7") + delta) / 100
        lower = _fraction(receipt["root_bounds"]["lower"])
        upper = _fraction(receipt["root_bounds"]["upper"])
        assert lower**4 < radicand < upper**4
        assert receipt["root_bounds"]["exact"] is False
        denominator = _fraction(receipt["denominator_mean"])
        assert _fraction(receipt["numerator_mean_bounds"]["lower"]) == (
            Fraction("1700.68") * sum(lower**i for i in range(4)) / 4
        )
        for bound in ("lower", "upper"):
            value = _fraction(receipt["unrounded_factor_bounds"][bound])
            assert (
                value
                == _fraction(receipt["numerator_mean_bounds"][bound]) / denominator
            )
            assert round(value * 1000000) == _EXPECTED[factor_id]


def test_september_observation_reanchors_and_requires_reviewed_redeclaration(
    operand_facts,
):
    declaration = _entry()["declaration"]
    anchor = next(
        fact for fact in operand_facts if fact["period"]["value"] == "2026-Q2"
    )
    september = copy.deepcopy(anchor)
    september["aggregate_fact_key"] = "ledger.aggregate_fact.v2:synthetic_september"
    september["period"]["value"] = "2026-Q3"
    september["layout"]["record_set_id"] = "stats_nz.qes.awe.2026q3"
    september["period_coverage"].update(
        {"start_date": "2026-07-01", "end_date": "2026-09-30"}
    )
    september["value"] = LexicalFloat("1750.00")
    declaration["feed_identity"]["facts_sha256"] = "9" * 64
    facts = (*operand_facts, september)
    with pytest.raises(ValueError, match="expected factor mismatch.*re-declaration"):
        resolve_reference_factor(declaration, facts, declaration["feed_identity"])
    with localcontext() as context:
        context.prec = 80
        q = (Decimal("1.027").sqrt()).sqrt()
        numerator = (Decimal("1700.68") + Decimal(1750) * (1 + q + q * q)) / 4
        expected = round(numerator / Decimal("1537.7225") * 1000000)
    declaration["expected_operational_factor"]["numerator"] = expected
    factor, receipt = resolve_reference_factor(
        declaration, facts, declaration["feed_identity"]
    )
    assert factor["numerator"] == expected
    assert [slot["status"] for slot in receipt["target_schedule"]] == [
        "observed",
        "observed",
        "projected",
        "projected",
    ]
    lower = _fraction(receipt["root_bounds"]["lower"])
    assert (
        _fraction(receipt["numerator_mean_bounds"]["lower"])
        == (Fraction("1700.68") + 1750 * (1 + lower + lower**2)) / 4
    )
    for slot in receipt["target_schedule"][2:]:
        assert slot["projection_anchor"] == {
            "aggregate_fact_key": september["aggregate_fact_key"]
        }
