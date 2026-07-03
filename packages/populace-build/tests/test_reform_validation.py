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


def test_obbba_components_score_stacked_in_jcx_order(monkeypatch):
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
        # Reform keys are the provisions still repealed. Pre-OBBBA repeals both;
        # the provisions are then enacted one at a time in order: A (only B still
        # repealed), then B (nothing repealed → reform is None).
        totals = {
            frozenset({"gov.example.a", "gov.example.b"}): 1_000.0,  # pre-OBBBA
            frozenset({"gov.example.b"}): 900.0,  # A enacted
            None: 960.0,  # A and B enacted
        }
        return _FakeSim({"income_tax": totals[reform]})

    payload = reform_validation_payload(specs, period=2026, simulate=simulate)
    rows = {row["id"]: row for row in payload["reforms"]}
    # A is scored against pre-OBBBA; B is scored against the post-A state, not
    # against pre-OBBBA — that stacking is the whole point.
    assert rows["obbba_a"]["populace"]["baseline_total"] == pytest.approx(1_000.0)
    assert rows["obbba_a"]["populace"]["reform_total"] == pytest.approx(900.0)
    assert rows["obbba_a"]["populace"]["budget_effect"] == pytest.approx(-100.0)
    assert rows["obbba_b"]["populace"]["baseline_total"] == pytest.approx(900.0)
    assert rows["obbba_b"]["populace"]["reform_total"] == pytest.approx(960.0)
    assert rows["obbba_b"]["populace"]["budget_effect"] == pytest.approx(60.0)
    # Stacked line effects telescope to the true total OBBBA effect.
    total = sum(rows[i]["populace"]["budget_effect"] for i in ("obbba_a", "obbba_b"))
    assert total == pytest.approx(960.0 - 1_000.0)


def test_obbba_mixed_measures_stack_per_group(monkeypatch):
    # The estate exemption is scored on estate_tax (not a component of
    # income_tax). A mixed-measure set must NOT degrade the income-tax
    # provisions to isolated scoring: they keep stacking as one group while the
    # estate provision scores within its own single-member group, with the
    # other group's reverts still applied.
    def spec(id_, path, measure):
        return ReformValidationSpec(
            id=id_,
            name=id_,
            category="OBBBA",
            in_sample=False,
            period=2026,
            jct_score=-1.0,
            jct_window="FY2026",
            jct_source="JCX",
            jct_source_url="",
            parameter_changes={path: {"2026-01-01.2026-12-31": 0}},
            effect_direction="baseline_minus_reform",
            budget_measure=measure,
        )

    specs = (
        spec("obbba_a", "gov.example.a", "income_tax"),
        spec("obbba_b", "gov.example.b", "income_tax"),
        spec("obbba_estate", "gov.example.estate", "estate_tax"),
    )
    monkeypatch.setattr(
        reform_validation_module,
        "_build_parameter_reform",
        lambda changes: frozenset(changes),
    )

    def simulate(reform):
        # Keys are the provisions still repealed. Income group: A enacted
        # ({b, estate} repealed) then B enacted ({estate} repealed). Estate
        # group: estate enacted ({a, b} repealed). An isolated-fallback state
        # like {a, estate} is not in the table and would KeyError.
        totals = {
            frozenset({"gov.example.a", "gov.example.b", "gov.example.estate"}): {
                "income_tax": 1_000.0,
                "estate_tax": 50.0,
            },
            frozenset({"gov.example.b", "gov.example.estate"}): {
                "income_tax": 900.0,
                "estate_tax": 50.0,
            },
            frozenset({"gov.example.estate"}): {
                "income_tax": 960.0,
                "estate_tax": 50.0,
            },
            frozenset({"gov.example.a", "gov.example.b"}): {
                "income_tax": 1_000.0,
                "estate_tax": 30.0,
            },
        }
        return _FakeSim(totals[reform])

    payload = reform_validation_payload(specs, period=2026, simulate=simulate)
    rows = {row["id"]: row for row in payload["reforms"]}
    # Income-tax provisions still stack: B scores against the post-A state.
    assert rows["obbba_a"]["populace"]["budget_effect"] == pytest.approx(-100.0)
    assert rows["obbba_b"]["populace"]["baseline_total"] == pytest.approx(900.0)
    assert rows["obbba_b"]["populace"]["budget_effect"] == pytest.approx(60.0)
    # The estate provision scores on its own measure.
    assert rows["obbba_estate"]["populace"]["baseline_total"] == pytest.approx(50.0)
    assert rows["obbba_estate"]["populace"]["reform_total"] == pytest.approx(30.0)
    assert rows["obbba_estate"]["populace"]["budget_effect"] == pytest.approx(-20.0)


def test_obbba_stacked_scoring_releases_intermediate_simulations(monkeypatch):
    specs = tuple(
        ReformValidationSpec(
            id=f"obbba_{name}",
            name=f"OBBBA {name}",
            category="OBBBA",
            in_sample=False,
            period=2026,
            jct_score=None,
            jct_window="FY2026",
            jct_source="JCX",
            jct_source_url="",
            parameter_changes={
                f"gov.example.{name}": {"2026-01-01.2026-12-31": 0},
            },
            effect_direction="baseline_minus_reform",
        )
        for name in ("a", "b", "c")
    )
    monkeypatch.setattr(
        reform_validation_module,
        "_build_parameter_reform",
        lambda changes: frozenset(changes),
    )

    live_simulations = 0

    class _TrackedSim(_FakeSim):
        def __init__(self, total: float) -> None:
            nonlocal live_simulations
            assert live_simulations == 0
            live_simulations += 1
            super().__init__({"income_tax": total})

        def __del__(self) -> None:
            nonlocal live_simulations
            live_simulations -= 1

    def simulate(reform):
        totals = {
            frozenset({"gov.example.a", "gov.example.b", "gov.example.c"}): 1000.0,
            frozenset({"gov.example.b", "gov.example.c"}): 900.0,
            frozenset({"gov.example.c"}): 960.0,
            None: 970.0,
        }
        return _TrackedSim(totals[reform])

    payload = reform_validation_payload(specs, period=2026, simulate=simulate)
    assert [row["id"] for row in payload["reforms"]] == [
        "obbba_a",
        "obbba_b",
        "obbba_c",
    ]
    assert live_simulations == 0


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


def test_itemized_benefit_limit_counterfactual_keeps_pease():
    # OBBBA line 11 = OBBBA itemized-deduction cap vs present-law Pease. The
    # counterfactual must revert ONLY the OBBBA cap (obbb.applies); disabling
    # the whole limitation (limitation.applies=False) drops present-law Pease,
    # which flips the sign (measures cap-vs-nothing, +, instead of cap-vs-Pease,
    # the JCT-scored cost).
    spec = next(
        s
        for s in out_of_sample_reform_specs(period=2026)
        if s.id == "obbba_itemized_tax_benefit_limit"
    )
    paths = set(spec.parameter_changes or {})
    assert paths == {"gov.irs.deductions.itemized.limitation.obbb.applies"}, paths
    assert "gov.irs.deductions.itemized.limitation.applies" not in paths, (
        "must not disable the whole limitation (drops present-law Pease)"
    )


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


def test_out_of_sample_specs_carry_fy2027_and_emit_it():
    specs = out_of_sample_reform_specs(period=2026)
    rates = next(s for s in specs if s.id == "obbba_reduced_rates")
    # FY2027 (first full fiscal year) is larger than the FY2026 ramp figure.
    assert rates.jct_score_fy2027 == -222154000000
    assert abs(rates.jct_score_fy2027) > abs(rates.jct_score)
    payload = reform_validation_payload([rates], period=2026, simulate=None)
    assert payload["reforms"][0]["jct"]["score_fy2027"] == -222154000000


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


def test_soi_baseline_levels_load_from_default_config():
    from populace.build.us_runtime.reform_validation import soi_baseline_level_specs

    levels = soi_baseline_level_specs()
    assert len(levels) >= 8
    ids = [lv.id for lv in levels]
    assert len(ids) == len(set(ids))
    for lv in levels:
        assert lv.benchmark_value > 0
        assert lv.benchmark_year.startswith("TY")
        assert "SOI" in lv.source
        assert lv.source_url
        # Levels must be OUT of the calibration target set; the calibrated
        # national SOI concepts are income lines, EITC, and itemized
        # components — none of these variables.
        assert lv.variable not in {
            "adjusted_gross_income",
            "eitc",
            "taxable_income",
            "itemized_taxable_income_deductions",
        }
    income_tax = next(lv for lv in levels if lv.id == "soi_income_tax_net")
    assert income_tax.benchmark_value == pytest.approx(2_042_047_899_000)


def test_baseline_levels_share_one_simulation_and_emit_rows():
    from populace.build.us_runtime.reform_validation import BaselineLevelSpec

    calls = []

    def simulate(reform):
        calls.append(reform)
        return _FakeSim({"cdcc": 3.6e9, "savers_credit": 2.2e9})

    levels = (
        BaselineLevelSpec(
            id="soi_cdcc",
            name="CDCC",
            variable="cdcc",
            period=2024,
            benchmark_value=3.47e9,
            benchmark_year="TY2023",
            source="IRS SOI Pub 1304 TY2023, Table 3.3",
            source_url="https://www.irs.gov/pub/irs-soi/23in33ar.xls",
        ),
        BaselineLevelSpec(
            id="soi_savers",
            name="Saver's credit",
            variable="savers_credit",
            period=2024,
            benchmark_value=2.04e9,
            benchmark_year="TY2023",
            source="IRS SOI Pub 1304 TY2023, Table 3.3",
            source_url="https://www.irs.gov/pub/irs-soi/23in33ar.xls",
        ),
    )
    payload = reform_validation_payload(
        (), period=2024, simulate=simulate, baseline_levels=levels
    )
    # Both levels read the one shared baseline simulation.
    assert calls == [None]
    rows = {row["id"]: row for row in payload["reforms"]}
    cdcc = rows["soi_cdcc"]
    assert cdcc["category"] == "IRS SOI actual"
    assert cdcc["in_sample"] is False
    assert cdcc["jct"]["score"] == pytest.approx(3.47e9)
    assert cdcc["jct"]["score_type"] == "actual"
    assert cdcc["jct"]["window"] == "TY2023"
    assert cdcc["populace"]["budget_effect"] == pytest.approx(3.6e9)
    assert rows["soi_savers"]["populace"]["budget_effect"] == pytest.approx(2.2e9)
    assert payload["out_of_sample_simulated"] is True


def test_baseline_levels_null_without_simulate():
    from populace.build.us_runtime.reform_validation import BaselineLevelSpec

    level = BaselineLevelSpec(
        id="soi_cdcc",
        name="CDCC",
        variable="cdcc",
        period=2024,
        benchmark_value=3.47e9,
        benchmark_year="TY2023",
        source="IRS SOI",
        source_url="https://example.test",
    )
    payload = reform_validation_payload(
        (), period=2024, simulate=None, baseline_levels=(level,)
    )
    assert payload["reforms"][0]["populace"]["budget_effect"] is None
    # An unsimulated backtest must mark itself, same as skipped OBBBA rows.
    assert payload["out_of_sample_simulated"] is False


def test_capped_baseline_level_limits_credit_to_liability():
    import numpy as np

    from populace.build.us_runtime.reform_validation import BaselineLevelSpec

    class _WeightedSeries:
        def __init__(self, values, weights):
            self._values = np.asarray(values, dtype=float)
            self.weights = np.asarray(weights, dtype=float)

        def __array__(self, dtype=None):
            return self._values if dtype is None else self._values.astype(dtype)

        def sum(self):
            return float((self._values * self.weights).sum())

    class _ArraySim:
        # Two tax units (weights 10 and 1): the first has a $2,000 credit but
        # only $500 of pre-credit liability, so only $500 is usable.
        data = {
            "cdcc": ([2_000.0, 1_000.0], [10.0, 1.0]),
            "income_tax_before_credits": ([500.0, 5_000.0], [10.0, 1.0]),
        }

        def calculate(self, measure, period):
            values, weights = self.data[measure]
            return _WeightedSeries(values, weights)

    def simulate(reform):
        assert reform is None
        return _ArraySim()

    level = BaselineLevelSpec(
        id="soi_cdcc",
        name="CDCC",
        variable="cdcc",
        period=2024,
        benchmark_value=3.47e9,
        benchmark_year="TY2023",
        source="IRS SOI",
        source_url="https://example.test",
        cap_variable="income_tax_before_credits",
    )
    payload = reform_validation_payload(
        (), period=2024, simulate=simulate, baseline_levels=(level,)
    )
    row = payload["reforms"][0]
    # min(2000, 500)*10 + min(1000, 5000)*1 — NOT the uncapped 2000*10 + 1000.
    assert row["populace"]["budget_effect"] == pytest.approx(500 * 10 + 1_000)


def test_shipped_soi_levels_cap_nonrefundable_credits():
    from populace.build.us_runtime.reform_validation import soi_baseline_level_specs

    levels = {lv.id: lv for lv in soi_baseline_level_specs()}
    for capped in (
        "soi_cdcc",
        "soi_education_credits",
        "soi_savers_credit",
        "soi_ctc_nonrefundable",
    ):
        assert levels[capped].cap_variable == "income_tax_before_credits"
    # Taxes and refundable credits are clean concepts — no cap.
    for uncapped in (
        "soi_income_tax_net",
        "soi_amt",
        "soi_niit",
        "soi_se_tax",
        "soi_ctc_refundable",
    ):
        assert levels[uncapped].cap_variable is None


def _write_json(path, payload):
    path.write_text(json.dumps(payload))
    return path


def test_state_program_level_specs_carry_category_and_score_type(tmp_path):
    from populace.build.us_runtime.reform_validation import (
        state_program_level_specs,
    )

    config = _write_json(
        tmp_path / "state_program_levels.json",
        {
            "levels": [
                {
                    "id": "mn_ctc_wfc_total",
                    "name": "Minnesota CTC + Working Family Credit",
                    "variable": "mn_child_and_working_families_credits",
                    "period": 2024,
                    "benchmark": {
                        "value": 5.64e8,
                        "year": "TY2024",
                        "source": "MN DOR",
                        "source_url": "https://www.revenue.state.mn.us/",
                    },
                },
                {
                    "id": "il_eitc_total",
                    "name": "Illinois EITC (approximation)",
                    "variable": "il_eitc",
                    "period": 2024,
                    "benchmark": {
                        "value": 4.2e8,
                        "year": "TY2022",
                        "source": "20% x IRS SOI federal EITC in IL",
                        "source_url": "https://www.irs.gov/",
                        "score_type": "approximation",
                    },
                },
            ]
        },
    )
    specs = state_program_level_specs(config)
    assert [s.id for s in specs] == ["mn_ctc_wfc_total", "il_eitc_total"]
    # Default category + score_type for official rows; approximation carries
    # through from the benchmark block.
    assert specs[0].category == "State program actual"
    assert specs[0].benchmark_score_type == "actual"
    assert specs[1].benchmark_score_type == "approximation"


def test_level_rows_emit_per_spec_category_and_score_type():
    from populace.build.us_runtime.reform_validation import BaselineLevelSpec

    level = BaselineLevelSpec(
        id="mn_ctc_wfc_total",
        name="Minnesota CTC + WFC",
        variable="mn_child_and_working_families_credits",
        period=2024,
        benchmark_value=5.64e8,
        benchmark_year="TY2024",
        source="MN DOR",
        source_url="",
        category="State program actual",
        benchmark_score_type="approximation",
    )
    sim = _FakeSim({"mn_child_and_working_families_credits": 5.5e8})
    payload = reform_validation_payload(
        (),
        period=2024,
        simulate=lambda reform: sim,
        baseline_levels=(level,),
    )
    row = payload["reforms"][0]
    assert row["category"] == "State program actual"
    assert row["jct"]["score_type"] == "approximation"
    assert row["populace"]["budget_effect"] == pytest.approx(5.5e8)


def test_soi_level_rows_keep_historical_category():
    from populace.build.us_runtime.reform_validation import BaselineLevelSpec

    level = BaselineLevelSpec(
        id="soi_amt",
        name="AMT",
        variable="alternative_minimum_tax",
        period=2024,
        benchmark_value=2.7e9,
        benchmark_year="TY2023",
        source="IRS SOI",
        source_url="",
    )
    sim = _FakeSim({"alternative_minimum_tax": 2.8e9})
    payload = reform_validation_payload(
        (), period=2024, simulate=lambda reform: sim, baseline_levels=(level,)
    )
    row = payload["reforms"][0]
    assert row["category"] == "IRS SOI actual"
    assert row["jct"]["score_type"] == "actual"


def test_state_program_reform_specs_load(tmp_path):
    from populace.build.us_runtime.reform_validation import (
        state_program_reform_specs,
    )

    config = _write_json(
        tmp_path / "state_program_reforms.json",
        {
            "reforms": [
                {
                    "id": "state_mn_cwfc_repeal",
                    "name": "Repeal Minnesota CTC + Working Family Credit",
                    "neutralized_variable": "mn_child_and_working_families_credits",
                    "period": 2024,
                    "benchmark": {
                        "score": 8.3e8,
                        "window": "TY2024",
                        "source": "MN DOR (CTC $564M + WFC)",
                        "source_url": "https://www.revenue.state.mn.us/",
                    },
                }
            ]
        },
    )
    specs = state_program_reform_specs(config, period=2024)
    (spec,) = specs
    assert spec.category == "State program"
    assert spec.in_sample is False
    assert spec.budget_measure == "state_income_tax"
    assert spec.jct_score_type == "actual"
    assert spec.neutralized_variable == "mn_child_and_working_families_credits"
    assert spec.effect_direction == "reform_minus_baseline"


def test_default_baseline_level_specs_concatenates(monkeypatch, tmp_path):
    from populace.build.us_runtime import reform_validation as rv

    soi = _write_json(
        tmp_path / "soi.json",
        {
            "levels": [
                {
                    "id": "soi_x",
                    "name": "x",
                    "variable": "income_tax",
                    "period": 2024,
                    "benchmark": {"value": 1.0},
                }
            ]
        },
    )
    state = _write_json(
        tmp_path / "state.json",
        {
            "levels": [
                {
                    "id": "state_y",
                    "name": "y",
                    "variable": "mn_wfc",
                    "period": 2024,
                    "benchmark": {"value": 2.0},
                }
            ]
        },
    )
    monkeypatch.setattr(rv, "_soi_baseline_levels_config_path", lambda: soi)
    monkeypatch.setattr(rv, "_state_program_levels_config_path", lambda: state)
    specs = rv.default_baseline_level_specs()
    assert [s.id for s in specs] == ["soi_x", "state_y"]
    assert specs[0].category == "IRS SOI actual"
    assert specs[1].category == "State program actual"


def test_shipped_state_program_configs_are_well_formed():
    from populace.build.us_runtime.reform_validation import (
        state_program_level_specs,
        state_program_reform_specs,
    )

    levels = state_program_level_specs()
    reforms = state_program_reform_specs(period=2024)
    assert len(levels) >= 30
    assert len(reforms) >= 10
    ids = [s.id for s in levels] + [s.id for s in reforms]
    assert len(ids) == len(set(ids))
    for spec in levels:
        assert spec.benchmark_value > 0
        assert spec.variable
        assert spec.source and spec.source_url
        assert spec.category == "State program actual"
        assert spec.benchmark_score_type in {"actual", "approximation"}
        assert spec.period == 2024
    for spec in reforms:
        assert spec.jct_score and spec.jct_score > 0
        assert spec.neutralized_variable
        assert spec.budget_measure == "state_income_tax"
        assert spec.in_sample is False
