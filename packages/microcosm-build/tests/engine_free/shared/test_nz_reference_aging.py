"""NZ feed operands, reviewed factors, and the observed-quarter re-declaration."""

from __future__ import annotations

import copy
import json
import os
from datetime import date
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from microcosm.build import reference_factors
from microcosm.build.ledger_artifact import (
    LexicalFloat,
    load_ledger_consumer_artifact,
    validate_chronicle_feed_pin,
)
from microcosm.build.ledger_targets import fact_matches_selector
from microcosm.build.reference_factors import (
    replay_reference_factor,
    require_current_factors,
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
    "ird_ty2024_to_ty2027_low": 1117460,
    "ird_ty2024_to_ty2027_central": 1118491,
    "ird_ty2024_to_ty2027_high": 1119519,
    "p1_ty2026_to_ty2027_low": 1028904,
    "p1_ty2026_to_ty2027_central": 1029853,
    "p1_ty2026_to_ty2027_high": 1030799,
    "observed_ty2026_over_ty2024": 1086069,
    "donor_cy2024_to_ty2027_low": 1079413,
    "donor_cy2024_to_ty2027_central": 1080408,
    "donor_cy2024_to_ty2027_high": 1081401,
}
_DENOMINATORS = {
    "p1_": Fraction("1670.0725"),
    "donor_": Fraction("1591.925"),
}
_PURPOSES = {"observed_": "diagnostic", "donor_": "money_normalization_input"}


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


def test_packaged_factor_context_validation_ignores_the_wall_clock(monkeypatch):
    class FutureDate(date):
        @classmethod
        def today(cls):
            return cls(2027, 2, 3)

    monkeypatch.setattr(reference_factors, "date", FutureDate)

    assert set(validate_factor_context(_document())) == set(_EXPECTED)


@pytest.mark.parametrize(
    "today,current",
    [
        (date(2026, 11, 3), False),
        (date(2026, 11, 4), True),
        (date(2027, 2, 2), True),
        (date(2027, 2, 3), False),
    ],
)
def test_factor_currency_uses_inclusive_declared_boundaries(today, current):
    selected = validate_factor_context(_document())
    if current:
        require_current_factors(selected, today=today)
    else:
        with pytest.raises(ValueError, match="reviewed re-declaration") as error:
            require_current_factors(selected, today=today)
        for factor_id in selected:
            assert factor_id in str(error.value)


@pytest.mark.parametrize("factor_id", list(_EXPECTED))
def test_all_nine_factors_and_observed_diagnostic_are_exact(factor_id, operand_facts):
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
    assert _fraction(receipt["denominator_mean"]) == next(
        (
            value
            for prefix, value in _DENOMINATORS.items()
            if factor_id.startswith(prefix)
        ),
        Fraction("1537.7225"),
    )
    assert declaration["purpose"] == next(
        (value for prefix, value in _PURPOSES.items() if factor_id.startswith(prefix)),
        "profile_input",
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


def _p1_numerators(declaration):
    """A P1 factor's numerator under one half-even rounding, and what truncation
    or an intermediate rounding would declare instead, all in millionths."""
    value = {period: Decimal(text) for _, period, _, text in _OPERANDS}
    sources = [
        row["selector"]["period_value"] for row in declaration["source_schedule"]
    ]
    cent = Decimal("0.01")
    with localcontext() as context:
        context.prec = 60
        growth = Decimal("2.7") + Decimal(declaration["growth"]["scenario_delta"])
        q = (1 + growth / 100).sqrt().sqrt()
        q6 = q.quantize(Decimal("0.000001"))
        june, september = value["2026-Q2"], value["2026-Q3"]
        denominator = sum(value[period] for period in sources) / 4

        def mean(projected):
            return (june + september + sum(projected)) / 4

        def millionths(numerator, rounding=ROUND_HALF_EVEN):
            ratio = numerator / denominator * 1000000
            return int(ratio.to_integral_value(rounding=rounding))

        exact = mean([september * q, september * q * q])
        return millionths(exact), {
            "truncated": millionths(exact, ROUND_FLOOR),
            "numerator rounded to cents": millionths(exact.quantize(cent)),
            "projected quarters rounded to cents": millionths(
                mean([(september * q**power).quantize(cent) for power in (1, 2)])
            ),
            "q rounded to six digits": millionths(
                mean([september * q6, september * q6 * q6])
            ),
        }


@pytest.mark.parametrize("profile", ["low", "central", "high"])
def test_p1_rounds_once_and_refuses_truncated_or_intermediate_numerators(
    profile, operand_facts
):
    factor_id = f"p1_ty2026_to_ty2027_{profile}"
    declaration = _entry(factor_id)["declaration"]
    assert declaration["source_period"] == 2026
    assert [
        row["selector"]["period_value"] for row in declaration["source_schedule"]
    ] == ["2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1"]
    once, alternatives = _p1_numerators(declaration)
    assert once == _EXPECTED[factor_id]
    for numerator in sorted(set(alternatives.values()) - {once}):
        declaration["expected_operational_factor"]["numerator"] = numerator
        with pytest.raises(ValueError, match="expected factor mismatch"):
            resolve_reference_factor(
                declaration, operand_facts, _document()["feed_identity"]
            )


def test_p1_profiles_separate_one_rounding_from_each_alternative():
    """Each alternative misses some profile's declared numerator, so the
    refusal test above exercises every alternative."""
    missed = set()
    for profile in ("low", "central", "high"):
        declaration = _entry(f"p1_ty2026_to_ty2027_{profile}")["declaration"]
        once, alternatives = _p1_numerators(declaration)
        missed |= {
            name for name, numerator in alternatives.items() if numerator != once
        }
    assert missed == {
        "truncated",
        "numerator rounded to cents",
        "projected quarters rounded to cents",
        "q rounded to six digits",
    }


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
        assert (
            _fraction(receipt["numerator_mean_bounds"]["lower"])
            == (
                Fraction("1700.68")
                + Fraction("1714.86") * sum(lower**i for i in range(3))
            )
            / 4
        )
        for bound in ("lower", "upper"):
            value = _fraction(receipt["unrounded_factor_bounds"][bound])
            assert (
                value
                == _fraction(receipt["numerator_mean_bounds"][bound]) / denominator
            )
            assert round(value * 1000000) == _EXPECTED[factor_id]


def test_december_observation_reanchors_and_requires_reviewed_redeclaration(
    operand_facts,
):
    declaration = _entry()["declaration"]
    anchor = next(
        fact for fact in operand_facts if fact["period"]["value"] == "2026-Q3"
    )
    december = copy.deepcopy(anchor)
    december["aggregate_fact_key"] = "ledger.aggregate_fact.v2:synthetic_december"
    december["period"]["value"] = "2026-Q4"
    december["layout"]["record_set_id"] = "stats_nz.qes.awe.2026q4"
    december["period_coverage"].update(
        {"start_date": "2026-10-01", "end_date": "2026-12-31"}
    )
    december["value"] = LexicalFloat("1750.00")
    declaration["feed_identity"]["facts_sha256"] = "9" * 64
    facts = (*operand_facts, december)
    with pytest.raises(ValueError, match="expected factor mismatch.*re-declaration"):
        resolve_reference_factor(declaration, facts, declaration["feed_identity"])
    with localcontext() as context:
        context.prec = 80
        q = (Decimal("1.027").sqrt()).sqrt()
        observed = Decimal("1700.68") + Decimal("1714.86")
        numerator = (observed + Decimal(1750) * (1 + q)) / 4
        expected = round(numerator / Decimal("1537.7225") * 1000000)
    declaration["expected_operational_factor"]["numerator"] = expected
    factor, receipt = resolve_reference_factor(
        declaration, facts, declaration["feed_identity"]
    )
    assert factor["numerator"] == expected
    assert [slot["status"] for slot in receipt["target_schedule"]] == [
        "observed",
        "observed",
        "observed",
        "projected",
    ]
    lower = _fraction(receipt["root_bounds"]["lower"])
    observed_sum = Fraction("1700.68") + Fraction("1714.86")
    assert (
        _fraction(receipt["numerator_mean_bounds"]["lower"])
        == (observed_sum + 1750 * (1 + lower)) / 4
    )
    for slot in receipt["target_schedule"][3:]:
        assert slot["projection_anchor"] == {
            "aggregate_fact_key": december["aggregate_fact_key"]
        }


def test_donor_factors_average_calendar_2024_and_share_the_tax_year_numerator(
    operand_facts,
):
    """Hub-recomputed values (G8-w rulings 4, H1); the feed calculation governs."""
    for profile, numerator in (
        ("low", 1079413),
        ("central", 1080408),
        ("high", 1081401),
    ):
        entry = _entry(f"donor_cy2024_to_ty2027_{profile}")
        ird = _entry(f"ird_ty2024_to_ty2027_{profile}")
        declaration = entry["declaration"]
        assert [
            slot["validation"]["period_coverage"]
            for slot in declaration["source_schedule"]
        ][0]["start_date"] == "2024-01-01"
        assert (
            declaration["source_schedule"][-1]["validation"]["period_coverage"][
                "end_date"
            ]
            == "2024-12-31"
        )
        assert declaration["target_schedule"] == ird["declaration"]["target_schedule"]
        assert declaration["growth"] == ird["declaration"]["growth"]
        assert declaration["validity"] == ird["declaration"]["validity"]
        factor, receipt = resolve_reference_factor(
            declaration, operand_facts, _document()["feed_identity"]
        )
        assert factor == {"numerator": numerator, "denominator": 1000000}
        mean = (
            Fraction("1563.54")
            + Fraction("1586.32")
            + Fraction("1595.11")
            + Fraction("1622.73")
        ) / 4
        assert mean == Fraction("1591.925") == _fraction(receipt["denominator_mean"])
        for bound in ("lower", "upper"):
            assert (
                round(
                    _fraction(ird["receipt"]["numerator_mean_bounds"][bound])
                    / mean
                    * 1000000
                )
                == numerator
            )
