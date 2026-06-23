"""Reform-validation payload assembly, isolated from policyengine-us.

The simulation is injected, so these tests exercise the budget-effect math and
the in-sample/out-of-sample split without running a Microsimulation.
"""

from __future__ import annotations

import json

import pytest

import populace.build.us_runtime.reform_validation as reform_validation_module
from populace.build.us_runtime.reform_validation import (
    REFORM_VALIDATION_SCHEMA_VERSION,
    ReformValidationSpec,
    in_sample_reform_specs,
    out_of_sample_reform_specs,
    reform_validation_payload,
    tax_expenditure_reform_specs,
    write_reform_validation,
)


class _FakeSeries:
    def __init__(self, total: float) -> None:
        self._total = total

    def sum(self) -> float:
        return self._total


class _FakeSim:
    """A sim whose weighted total for a measure shifts by a per-reform delta."""

    def __init__(self, totals: dict[str, float]) -> None:
        self._totals = totals

    def calculate(self, measure: str, period):  # noqa: ARG002
        return _FakeSeries(self._totals[measure])


def _oos_spec(score: float, *, category: str = "Other") -> ReformValidationSpec:
    return ReformValidationSpec(
        id="obbba_salt",
        name="OBBBA — SALT cap to $40k",
        category=category,
        in_sample=False,
        period=2024,
        jct_score=score,
        jct_window="FY2025-2034",
        jct_source="JCX-00-25",
        jct_source_url="https://www.jct.gov/",
        parameter_changes={"gov.example.cap": {"2025-01-01.2034-12-31": 40000}},
    )


def test_spec_requires_exactly_one_reform_definition():
    with pytest.raises(ValueError):
        ReformValidationSpec(
            id="x",
            name="x",
            category="c",
            in_sample=False,
            period=2024,
            jct_score=1.0,
            jct_window="",
            jct_source="",
            jct_source_url="",
        )
    with pytest.raises(ValueError):
        ReformValidationSpec(
            id="x",
            name="x",
            category="c",
            in_sample=False,
            period=2024,
            jct_score=1.0,
            jct_window="",
            jct_source="",
            jct_source_url="",
            neutralized_variable="v",
            parameter_changes={"a": 1},
        )


def test_in_sample_uses_calibration_estimate_no_simulation():
    specs = (
        ReformValidationSpec(
            id="nation/jct/mortgage",
            name="Mortgage interest deduction",
            category="JCT tax expenditure",
            in_sample=True,
            period=2024,
            jct_score=30e9,
            jct_window="annual",
            jct_source="JCT",
            jct_source_url="",
            neutralized_variable="mortgage_interest_deduction",
        ),
    )

    def simulate(_reform):  # pragma: no cover - must not be called
        raise AssertionError("in-sample rows must not simulate")

    payload = reform_validation_payload(
        specs,
        period=2024,
        simulate=simulate,
        in_sample_estimates={"nation/jct/mortgage": 28e9},
    )
    row = payload["reforms"][0]
    assert row["in_sample"] is True
    assert row["populace"]["budget_effect"] == pytest.approx(28e9)
    assert row["jct"]["score"] == pytest.approx(30e9)


def test_out_of_sample_budget_effect_is_reform_minus_baseline(monkeypatch):
    # baseline income_tax total 2.0e12; under the reform it rises by 50e9.
    spec = _oos_spec(score=-60e9)
    monkeypatch.setattr(spec.__class__, "build_reform", lambda self: "REFORM")

    def simulate(reform):
        total = 2.0e12 + 50e9 if reform == "REFORM" else 2.0e12
        return _FakeSim({"income_tax": total})

    payload = reform_validation_payload([spec], period=2024, simulate=simulate)
    row = payload["reforms"][0]
    assert row["in_sample"] is False
    assert row["populace"]["budget_effect"] == pytest.approx(50e9)
    assert row["populace"]["baseline_total"] == pytest.approx(2.0e12)
    assert row["populace"]["reform_total"] == pytest.approx(2.05e12)
    assert row["jct"]["score"] == pytest.approx(-60e9)


def test_counterfactual_revert_flips_sign(monkeypatch):
    # With a single OBBBA row, the pre-OBBBA scoring baseline is the row's
    # revert patch and the component-on reform is the no-reform baseline.
    spec = _oos_spec(score=-33e9, category="OBBBA")
    object.__setattr__(spec, "effect_direction", "baseline_minus_reform")
    monkeypatch.setattr(
        reform_validation_module,
        "_build_parameter_reform",
        lambda changes: frozenset(changes),
    )

    def simulate(reform):
        total = 2.033e12 if reform else 2.0e12
        return _FakeSim({"income_tax": total})

    payload = reform_validation_payload([spec], period=2024, simulate=simulate)
    assert payload["reforms"][0]["populace"]["budget_effect"] == pytest.approx(-33e9)


def test_obbba_components_score_against_pre_obbba_baseline(monkeypatch):
    specs = (
        ReformValidationSpec(
            id="obbba_a",
            name="OBBBA A",
            category="OBBBA",
            in_sample=False,
            period=2026,
            jct_score=-100.0,
            jct_window="FY2026",
            jct_source="JCX",
            jct_source_url="",
            parameter_changes={
                "gov.example.a": {"2026-01-01.2026-12-31": 0},
            },
            effect_direction="baseline_minus_reform",
        ),
        ReformValidationSpec(
            id="obbba_b",
            name="OBBBA B",
            category="OBBBA",
            in_sample=False,
            period=2026,
            jct_score=60.0,
            jct_window="FY2026",
            jct_source="JCX",
            jct_source_url="",
            parameter_changes={
                "gov.example.b": {"2026-01-01.2026-12-31": 0},
            },
            effect_direction="baseline_minus_reform",
        ),
    )
    monkeypatch.setattr(
        reform_validation_module,
        "_build_parameter_reform",
        lambda changes: frozenset(changes),
    )

    def simulate(reform):
        # Reform keys are the provisions still turned off. The full pre-OBBBA
        # baseline has both patches applied. Component A is scored with only B
        # still off, and component B with only A still off.
        totals = {
            frozenset({"gov.example.a", "gov.example.b"}): 1_000.0,
            frozenset({"gov.example.b"}): 900.0,
            frozenset({"gov.example.a"}): 1_060.0,
            None: 950.0,
        }
        return _FakeSim({"income_tax": totals[reform]})

    payload = reform_validation_payload(specs, period=2026, simulate=simulate)
    rows = {row["id"]: row for row in payload["reforms"]}
    assert rows["obbba_a"]["populace"]["baseline_total"] == pytest.approx(1_000.0)
    assert rows["obbba_a"]["populace"]["reform_total"] == pytest.approx(900.0)
    assert rows["obbba_a"]["populace"]["budget_effect"] == pytest.approx(-100.0)
    assert rows["obbba_b"]["populace"]["baseline_total"] == pytest.approx(1_000.0)
    assert rows["obbba_b"]["populace"]["reform_total"] == pytest.approx(1_060.0)
    assert rows["obbba_b"]["populace"]["budget_effect"] == pytest.approx(60.0)


def test_shipped_obbba_config_is_out_of_sample_counterfactual():
    specs = out_of_sample_reform_specs(period=2026)
    assert {s.id for s in specs} >= {"obbba_no_tax_on_tips", "obbba_no_tax_on_overtime"}
    assert any(s.jct_score and s.jct_score < 0 for s in specs)
    assert any(s.jct_score and s.jct_score > 0 for s in specs)
    assert any(s.jct_score is None for s in specs)
    for spec in specs:
        assert spec.effect_direction == "baseline_minus_reform"
        assert spec.period == 2026
        if spec.jct_score is None:
            assert "No standalone" in spec.jct_source
        else:
            assert spec.jct_source.startswith("JCX-35-25")


def test_shipped_tax_expenditure_specs_neutralize_big_provisions():
    specs = tax_expenditure_reform_specs(period=2024)
    by_id = {s.id for s in specs}
    assert {
        "te_ctc",
        "te_eitc",
        "te_cdcc",
        "te_standard_deduction",
        "te_itemized_total",
    } <= by_id
    for spec in specs:
        assert spec.neutralized_variable  # all are repeals
        assert spec.effect_direction == "reform_minus_baseline"  # neutralize raises tax
    eitc = next(s for s in specs if s.id == "te_eitc")
    assert eitc.in_sample is True  # calibrated to SOI EITC targets
    std = next(s for s in specs if s.id == "te_standard_deduction")
    assert std.jct_score is None  # baseline in both JCT and Treasury — no benchmark


def test_null_benchmark_row_publishes_magnitude_only(monkeypatch):
    spec = ReformValidationSpec(
        id="te_std",
        name="Standard deduction",
        category="Tax expenditure",
        in_sample=False,
        period=2024,
        jct_score=None,
        jct_window="FY2024",
        jct_source="not scored",
        jct_source_url="",
        neutralized_variable="standard_deduction",
    )
    monkeypatch.setattr(spec.__class__, "build_reform", lambda self: "REFORM")

    def simulate(reform):
        return _FakeSim({"income_tax": 2.28e12 if reform is not None else 2.0e12})

    payload = reform_validation_payload([spec], period=2024, simulate=simulate)
    row = payload["reforms"][0]
    assert row["jct"]["score"] is None
    assert row["populace"]["budget_effect"] == pytest.approx(280e9)  # repeal magnitude


def test_out_of_sample_null_when_no_simulate():
    payload = reform_validation_payload([_oos_spec(-1.0)], period=2024, simulate=None)
    assert payload["reforms"][0]["populace"]["budget_effect"] is None
    assert payload["schema_version"] == REFORM_VALIDATION_SCHEMA_VERSION
    # A release built with out-of-sample reforms but no simulation must mark
    # itself, so a null budget effect is never mistaken for a genuine result.
    assert payload["out_of_sample_simulated"] is False


def test_out_of_sample_simulated_flag_true_when_simulated(monkeypatch):
    spec = _oos_spec(-1.0)
    monkeypatch.setattr(spec.__class__, "build_reform", lambda self: "REFORM")

    def simulate(reform):
        return _FakeSim({"income_tax": 2.0e12 if reform is None else 1.99e12})

    payload = reform_validation_payload([spec], period=2024, simulate=simulate)
    assert payload["out_of_sample_simulated"] is True


def test_out_of_sample_simulated_flag_true_when_only_in_sample():
    # No out-of-sample specs => the fidelity test is vacuously complete.
    spec = in_sample_reform_specs(period=2024)[0]
    payload = reform_validation_payload(
        [spec], period=2024, in_sample_estimates={spec.id: 1.0}
    )
    assert payload["out_of_sample_simulated"] is True


def test_in_sample_specs_built_from_jct_reforms():
    specs = in_sample_reform_specs(period=2024)
    assert specs, "expected at least one JCT tax-expenditure reform"
    assert all(s.in_sample for s in specs)
    assert all(s.neutralized_variable for s in specs)


def test_out_of_sample_specs_load_from_default_config():
    # The shipped OBBBA config (if present) must parse into valid specs.
    specs = out_of_sample_reform_specs(period=2024)
    for spec in specs:
        assert spec.in_sample is False
        assert spec.parameter_changes
        assert spec.jct_source


def test_write_round_trips(tmp_path):
    payload = reform_validation_payload([_oos_spec(-1.0)], period=2024, simulate=None)
    path = write_reform_validation(payload, tmp_path / "reform_validation.json")
    assert json.loads(path.read_text())["reforms"][0]["id"] == "obbba_salt"
