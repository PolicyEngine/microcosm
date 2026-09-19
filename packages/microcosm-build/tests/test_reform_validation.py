"""Reform-validation payload assembly, isolated from policyengine-us.

The simulation is injected, so these tests exercise the budget-effect math and
the in-sample/out-of-sample split without running a Microsimulation.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

import microcosm.build.us_runtime.reform_validation as reform_validation_module
from microcosm.build.us_runtime.fiscal_targets import SimpleTaxExpenditureReform
from microcosm.build.us_runtime.reform_validation import (
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


def test_obbba_claim_comparisons_match_executed_worlds(monkeypatch):
    # Invented patches and callback totals: two measures and two periods must
    # retain separate stacks, even when their rows are interleaved.
    specs = tuple(
        replace(
            _oos_spec(-1, category="OBBBA"),
            id=name,
            parameter_changes={f"gov.example.{name}": {"2027-01-01": 0}},
            budget_measure=measure,
            period=period,
            effect_direction="baseline_minus_reform",
        )
        for name, measure, period in (
            ("a", "example_tax", 2027),
            ("c", "other_tax", 2027),
            ("b", "example_tax", 2027),
            ("d", "example_tax", 2028),
        )
    )
    original = deepcopy(specs)
    patches = {
        key: value for spec in specs for key, value in spec.parameter_changes.items()
    }

    def world(names):
        return {f"gov.example.{name}": patches[f"gov.example.{name}"] for name in names}

    expected_calls = [
        (world("abcd"), "example_tax", 2027, 1000),
        (world("bcd"), "example_tax", 2027, 900),
        (world("cd"), "example_tax", 2027, 920),
        (world("abcd"), "other_tax", 2027, 50),
        (world("abd"), "other_tax", 2027, 70),
        (world("abcd"), "example_tax", 2028, 1100),
        (world("abc"), "example_tax", 2028, 1080),
    ]
    calls = []
    monkeypatch.setattr(reform_validation_module, "_build_parameter_reform", deepcopy)

    def simulate(reform):
        changes, measure, at_period, total = expected_calls[len(calls)]
        assert reform == changes
        calls.append(deepcopy(reform))

        class FakeSimulation:
            def calculate(self, actual_measure, actual_period):
                assert (actual_measure, actual_period) == (measure, at_period)
                return _FakeSeries(total)

        return FakeSimulation()

    payload = reform_validation_payload(specs, period=2026, simulate=simulate)
    rows = {row["id"]: row for row in payload["reforms"]}
    assert len(calls) == len(expected_calls)
    assert payload["obbba_pre_baseline_parameter_changes"] == patches
    assert payload["baseline_period"] == 2026
    for name, before, after, base_total, reform_total in (
        ("a", "abcd", "bcd", 1000, 900),
        ("c", "abcd", "abd", 50, 70),
        ("b", "bcd", "cd", 900, 920),
        ("d", "abcd", "abc", 1100, 1080),
    ):
        comparison = rows[name]["scoring_comparison"]
        assert comparison == {
            "baseline": {
                "framework": "policyengine_us",
                "parameter_changes": world(before),
                "neutralized_variables": None,
            },
            "reform": {
                "framework": "policyengine_us",
                "parameter_changes": world(after),
                "neutralized_variables": None,
            },
            "effect_direction": "reform_minus_baseline",
        }
        assert rows[name]["microcosm"]["baseline_total"] == base_total
        assert rows[name]["microcosm"]["reform_total"] == reform_total
        assert rows[name]["microcosm"]["budget_effect"] == reform_total - base_total
        # The configured repeal definition keeps its original convention;
        # the executed comparison explicitly records the enactment convention.
        assert rows[name]["reform"]["parameter_changes"] == world(name)
        assert rows[name]["reform"]["effect_direction"] == "baseline_minus_reform"

    rows["a"]["scoring_comparison"]["baseline"]["parameter_changes"]["gov.example.b"][
        "2027-01-01"
    ] = 99
    rows["a"]["scoring_comparison"]["reform"]["parameter_changes"]["gov.example.b"][
        "2027-01-01"
    ] = 98
    payload["obbba_pre_baseline_parameter_changes"]["gov.example.a"]["2027-01-01"] = 97
    assert rows["b"]["scoring_comparison"]["baseline"]["parameter_changes"] == world(
        "bcd"
    )
    assert rows["a"]["reform"]["parameter_changes"] == world("a")
    assert specs == original
    assert calls == [call[0] for call in expected_calls]


def test_obbba_claim_current_law_endpoint_and_identical_shared_patch(monkeypatch):
    # Identical overlapping reverts are accepted by the existing scorer. Once
    # the first row enacts the shared path, both later worlds are current law.
    first = _oos_spec(-1, category="OBBBA")
    second = replace(first, id="second")
    monkeypatch.setattr(reform_validation_module, "_build_parameter_reform", deepcopy)
    calls = []

    def simulate(reform):
        calls.append(deepcopy(reform))
        return _FakeSim({"income_tax": 30 if reform else 20})

    payload = reform_validation_payload([first, second], period=2024, simulate=simulate)
    first_row, second_row = payload["reforms"]
    current_law = {
        "framework": "policyengine_us",
        "parameter_changes": None,
        "neutralized_variables": None,
    }
    assert calls == [first.parameter_changes, None, None]
    assert first_row["scoring_comparison"]["reform"] == current_law
    assert second_row["scoring_comparison"]["baseline"] == current_law
    assert second_row["scoring_comparison"]["reform"] == current_law
    assert first_row["microcosm"]["budget_effect"] == -10
    assert second_row["microcosm"]["budget_effect"] == 0


@pytest.mark.parametrize("category", ["OBBBA", "Other"])
def test_obbba_claim_unscored_rows_do_not_claim_executed_baseline(category):
    spec = _oos_spec(-1, category=category)
    payload = reform_validation_payload([spec], period=2024)
    assert payload["out_of_sample_simulated"] is False
    assert payload["reforms"][0]["microcosm"]["budget_effect"] is None
    assert (
        payload["reforms"][0]["reform"]["parameter_changes"] == spec.parameter_changes
    )
    assert "scoring_comparison" not in payload["reforms"][0]
    assert "obbba_pre_baseline_parameter_changes" not in payload


def test_obbba_claim_duplicate_ids_follow_existing_score_lookup(monkeypatch):
    # Duplicate OBBBA IDs already use the last stacked score. Metadata must
    # describe that same comparison, even with an ordinary row sharing the ID.
    first = replace(_oos_spec(-1, category="OBBBA"), id="shared")
    second = replace(
        first,
        parameter_changes={"gov.example.other": {"2027": 0}},
        period=2027,
    )
    ordinary = replace(first, category="Other")
    monkeypatch.setattr(reform_validation_module, "_build_parameter_reform", deepcopy)
    calls = []

    def simulate(reform):
        calls.append(deepcopy(reform))
        return _FakeSim({"income_tax": [100, 90, 100, 80, 110, 105][len(calls) - 1]})

    payload = reform_validation_payload(
        [first, ordinary, second], period=2024, simulate=simulate
    )
    first_row, ordinary_row, second_row = payload["reforms"]
    assert len(calls) == 6
    assert first_row["microcosm"]["budget_effect"] == -20
    assert second_row["microcosm"]["budget_effect"] == -20
    assert ordinary_row["microcosm"]["budget_effect"] == -5
    assert first_row["scoring_comparison"] == second_row["scoring_comparison"]
    assert second_row["scoring_comparison"]["reform"]["parameter_changes"] == calls[3]
    assert ordinary_row["scoring_comparison"]["reform"]["parameter_changes"] == calls[5]
    assert ordinary_row["scoring_comparison"]["baseline"]["parameter_changes"] is None


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
    assert row["microcosm"]["budget_effect"] == pytest.approx(28e9)
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
    assert row["microcosm"]["budget_effect"] == pytest.approx(50e9)
    assert row["microcosm"]["baseline_total"] == pytest.approx(2.0e12)
    assert row["microcosm"]["reform_total"] == pytest.approx(2.05e12)
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
    assert payload["reforms"][0]["microcosm"]["budget_effect"] == pytest.approx(-33e9)


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
    assert rows["obbba_a"]["microcosm"]["baseline_total"] == pytest.approx(1_000.0)
    assert rows["obbba_a"]["microcosm"]["reform_total"] == pytest.approx(900.0)
    assert rows["obbba_a"]["microcosm"]["budget_effect"] == pytest.approx(-100.0)
    assert rows["obbba_b"]["microcosm"]["baseline_total"] == pytest.approx(900.0)
    assert rows["obbba_b"]["microcosm"]["reform_total"] == pytest.approx(960.0)
    assert rows["obbba_b"]["microcosm"]["budget_effect"] == pytest.approx(60.0)
    # Stacked line effects telescope to the true total OBBBA effect.
    total = sum(rows[i]["microcosm"]["budget_effect"] for i in ("obbba_a", "obbba_b"))
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
    assert rows["obbba_a"]["microcosm"]["budget_effect"] == pytest.approx(-100.0)
    assert rows["obbba_b"]["microcosm"]["baseline_total"] == pytest.approx(900.0)
    assert rows["obbba_b"]["microcosm"]["budget_effect"] == pytest.approx(60.0)
    # The estate provision scores on its own measure.
    assert rows["obbba_estate"]["microcosm"]["baseline_total"] == pytest.approx(50.0)
    assert rows["obbba_estate"]["microcosm"]["reform_total"] == pytest.approx(30.0)
    assert rows["obbba_estate"]["microcosm"]["budget_effect"] == pytest.approx(-20.0)


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
    # Every shipped OBBBA row carries a JCX-35-25 benchmark: the senior
    # deduction (formerly the one unbenchmarked row) is folded into the
    # personal-exemption row, matching the combined scope of Ch.1 line 3.
    assert all(s.jct_score is not None for s in specs)
    for spec in specs:
        assert spec.effect_direction == "baseline_minus_reform"
        assert spec.period == 2026
        assert spec.jct_source.startswith("JCX-35-25")


def test_shipped_obbba_line3_scores_combined_sec_70103_scope():
    # JCX-35-25 Ch.1 line 3 nets the temporary senior deduction against the
    # exemption termination (OBBBA Sec. 70103). The counterfactual patch must
    # revert BOTH, or the modeled scope is narrower than the benchmark scope.
    spec = next(
        s
        for s in out_of_sample_reform_specs(period=2026)
        if s.id == "obbba_personal_exemption_termination"
    )
    paths = set(spec.parameter_changes or {})
    assert paths == {
        "gov.irs.income.exemption.suspended",
        "gov.irs.deductions.senior_deduction.amount",
    }, paths


def test_shipped_obbba_parameter_patches_are_pairwise_disjoint():
    # stacked_obbba_effects tracks enactment by parameter path: a path shared
    # by two provisions would silently corrupt both the merged pre-OBBBA
    # baseline and the incremental scoring, so disjointness is load-bearing.
    specs = out_of_sample_reform_specs(period=2026)
    seen: dict[str, str] = {}
    for spec in specs:
        for path in spec.parameter_changes or {}:
            assert path not in seen, (
                f"{path} appears in both {seen[path]} and {spec.id}"
            )
            seen[path] = spec.id


def test_shipped_obbba_specs_are_in_jcx_line_order():
    # The stack enacts provisions in file order; JCT scores each line given
    # the lines above it, so file order must follow the JCX-35-25 document
    # order for the increments to be comparable.
    import re

    specs = out_of_sample_reform_specs(period=2026)
    keys = []
    for spec in specs:
        if spec.budget_measure != "income_tax":
            continue  # other measures stack in their own group
        m = re.search(r"Ch\.(\d+)(?:\.([A-C]))? line (\d+)", spec.jct_source or "")
        assert m, f"{spec.id}: cannot parse JCX position from {spec.jct_source!r}"
        keys.append((int(m.group(1)), m.group(2) or "", int(m.group(3))))
    assert keys == sorted(keys), keys


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
    assert row["microcosm"]["budget_effect"] == pytest.approx(280e9)  # repeal magnitude


def test_out_of_sample_null_when_no_simulate():
    payload = reform_validation_payload([_oos_spec(-1.0)], period=2024, simulate=None)
    assert payload["reforms"][0]["microcosm"]["budget_effect"] is None
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
    from microcosm.build.us_runtime.reform_validation import soi_baseline_level_specs

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
    from microcosm.build.us_runtime.reform_validation import BaselineLevelSpec

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
    assert cdcc["microcosm"]["budget_effect"] == pytest.approx(3.6e9)
    assert rows["soi_savers"]["microcosm"]["budget_effect"] == pytest.approx(2.2e9)
    assert payload["out_of_sample_simulated"] is True


def test_baseline_levels_null_without_simulate():
    from microcosm.build.us_runtime.reform_validation import BaselineLevelSpec

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
    assert payload["reforms"][0]["microcosm"]["budget_effect"] is None
    # An unsimulated backtest must mark itself, same as skipped OBBBA rows.
    assert payload["out_of_sample_simulated"] is False


def test_capped_baseline_level_limits_credit_to_liability():
    import numpy as np

    from microcosm.build.us_runtime.reform_validation import BaselineLevelSpec

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
    assert row["microcosm"]["budget_effect"] == pytest.approx(500 * 10 + 1_000)


def test_shipped_soi_levels_cap_nonrefundable_credits():
    from microcosm.build.us_runtime.reform_validation import soi_baseline_level_specs

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
    from microcosm.build.us_runtime.reform_validation import (
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
    from microcosm.build.us_runtime.reform_validation import BaselineLevelSpec

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
    assert row["microcosm"]["budget_effect"] == pytest.approx(5.5e8)


def test_soi_level_rows_keep_historical_category():
    from microcosm.build.us_runtime.reform_validation import BaselineLevelSpec

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
    from microcosm.build.us_runtime.reform_validation import (
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


def test_state_reform_specs_load(tmp_path):
    from microcosm.build.us_runtime.reform_validation import state_reform_specs

    config = _write_json(
        tmp_path / "state_reforms.json",
        {
            "reforms": [
                {
                    "id": "state.mi.hb4170",
                    "name": "MI HB4170 flat rate cut",
                    "state": "MI",
                    "period": 2025,
                    "budget_measure": "mi_income_tax",
                    "parameter_changes": {
                        "gov.states.mi.tax.income.rate": {
                            "2025-01-01.2100-12-31": 0.0405
                        }
                    },
                    "benchmark": {
                        "score": -7.0e8,
                        "window": "annual",
                        "source": "Michigan HFA fiscal note",
                        "source_url": "https://www.legislature.mi.gov/",
                    },
                }
            ]
        },
    )
    specs = state_reform_specs(config, period=2026)
    (spec,) = specs
    assert spec.category == "State reform"
    assert spec.in_sample is False
    assert spec.period == 2025
    assert spec.budget_measure == "mi_income_tax"
    assert spec.jct_score == -7.0e8
    assert spec.jct_score_type == "fiscal_note"
    assert spec.parameter_changes
    assert spec.effect_direction == "reform_minus_baseline"


def test_state_reform_specs_shipped_config_loads():
    from microcosm.build.us_runtime.reform_validation import state_reform_specs

    specs = state_reform_specs(period=2026)
    # 8 original state bills (MA H5007 dropped: its DOR estimate has no
    # primary source) plus the tracker-informed expansion: state
    # credit/deduction bills and federal benchmark rows (ARPA provisions,
    # CBO rate option, UBI mechanical check).
    assert len(specs) >= 17
    assert all(not spec.in_sample for spec in specs)
    assert all(spec.parameter_changes for spec in specs)
    assert all(spec.jct_score is not None for spec in specs)
    by_category = {}
    for spec in specs:
        by_category.setdefault(spec.category, []).append(spec)
    assert set(by_category) == {"State reform", "Federal reform", "Mechanical check"}
    # State rows each score their own state's income tax.
    assert len(by_category["State reform"]) >= 13
    assert all(
        spec.budget_measure.endswith("_income_tax")
        for spec in by_category["State reform"]
    )
    # Federal rows score federal income tax against JCT/CBO published figures.
    assert all(
        spec.budget_measure == "income_tax" for spec in by_category["Federal reform"]
    )
    # Mechanical rows measure the reform's own spending variable, so the
    # benchmark is an exact external anchor (population x amount).
    assert all(
        spec.jct_score_type == "mechanical" for spec in by_category["Mechanical check"]
    )


def test_default_baseline_level_specs_concatenates(monkeypatch, tmp_path):
    from microcosm.build.us_runtime import reform_validation as rv

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
    fed = _write_json(
        tmp_path / "fed.json",
        {
            "levels": [
                {
                    "id": "fed_eitc_z",
                    "name": "z",
                    "variable": "eitc",
                    "state": "VA",
                    "period": 2024,
                    "benchmark": {"value": 3.0},
                }
            ]
        },
    )
    spm = _write_json(
        tmp_path / "spm.json",
        {
            "levels": [
                {
                    "id": "spm_w",
                    "name": "w",
                    "variable": "in_poverty",
                    "statistic": "rate",
                    "mask_variable": "is_child",
                    "state": "AL",
                    "period": 2024,
                    "benchmark": {"value": 0.146},
                }
            ]
        },
    )
    monkeypatch.setattr(rv, "_soi_baseline_levels_config_path", lambda: soi)
    monkeypatch.setattr(rv, "_state_program_levels_config_path", lambda: state)
    monkeypatch.setattr(rv, "_federal_eitc_by_state_config_path", lambda: fed)
    monkeypatch.setattr(rv, "_state_spm_poverty_levels_config_path", lambda: spm)
    specs = rv.default_baseline_level_specs()
    assert [s.id for s in specs] == ["soi_x", "state_y", "fed_eitc_z", "spm_w"]
    assert specs[0].category == "IRS SOI actual"
    assert specs[1].category == "State program actual"
    assert specs[2].category == "Federal EITC by state"
    assert specs[2].state == "VA"
    assert specs[3].category == "Census state SPM"
    assert specs[3].statistic == "rate"
    assert specs[3].mask_variable == "is_child"


def test_shipped_state_program_configs_are_well_formed():
    from microcosm.build.us_runtime.reform_validation import (
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


def test_shipped_federal_eitc_by_state_levels_are_well_formed():
    from microcosm.build.us_runtime.reform_validation import (
        federal_eitc_state_level_specs,
    )

    specs = federal_eitc_state_level_specs()
    assert len(specs) == 18
    for spec in specs:
        assert spec.variable == "eitc"
        assert spec.state and len(spec.state) == 2
        assert spec.benchmark_value > 0
        assert spec.category == "Federal EITC by state"
        assert spec.benchmark_score_type == "actual"
        assert spec.period == 2024


def test_state_sliced_level_total_filters_households(monkeypatch):
    # A level with `state` must total the variable over that state's
    # households only, weighting by household weight.
    import numpy as np

    from microcosm.build.us_runtime.reform_validation import (
        BaselineLevelSpec,
        reform_validation_payload,
    )

    class _Series(np.ndarray):
        weights: np.ndarray

    def series(values, weights):
        arr = np.asarray(values, dtype=float).view(_Series)
        arr.weights = np.asarray(weights, dtype=float)
        return arr

    class _Sim:
        def calculate(self, variable, period, map_to=None):
            if variable == "eitc":
                assert map_to == "household"
                return series([100.0, 200.0, 300.0], [1.0, 2.0, 3.0])
            if variable == "state_code_str":
                return np.asarray(["VA", "OH", "VA"])
            raise AssertionError(variable)

    level = BaselineLevelSpec(
        id="fed_eitc_va",
        name="VA",
        variable="eitc",
        period=2024,
        benchmark_value=1_000.0,
        benchmark_year="TY2024",
        source="s",
        source_url="u",
        state="VA",
    )
    payload = reform_validation_payload(
        (), period=2024, baseline_levels=(level,), simulate=lambda reform: _Sim()
    )
    row = next(r for r in payload["reforms"] if r["id"] == "fed_eitc_va")
    # VA households: 100*1 + 300*3 = 1000
    assert row["microcosm"]["budget_effect"] == 1_000.0


def test_variable_list_level_sums_components(monkeypatch, tmp_path):
    # A list-valued `variable` totals each component and sums them (used for
    # unsplit official figures covering mutually exclusive elections).
    from microcosm.build.us_runtime import reform_validation as rv

    cfg = _write_json(
        tmp_path / "levels.json",
        {
            "levels": [
                {
                    "id": "combo",
                    "name": "combo",
                    "variable": ["a_var", "b_var"],
                    "period": 2024,
                    "benchmark": {"value": 5.0},
                }
            ]
        },
    )
    specs = rv.state_program_level_specs(cfg)
    assert specs[0].variable == ["a_var", "b_var"]

    class _Series(float):
        pass

    class _Sim:
        def calculate(self, variable, period, map_to=None):
            import numpy as np

            class _MS(np.ndarray):
                pass

            vals = {"a_var": 2.0, "b_var": 3.0}[variable]
            arr = np.asarray([vals]).view(_MS)
            return arr

    payload = rv.reform_validation_payload(
        (), period=2024, baseline_levels=specs, simulate=lambda reform: _Sim()
    )
    row = payload["reforms"][0]
    assert row["microcosm"]["budget_effect"] == 5.0


def test_md_eitc_repeal_neutralizes_tax_path_components():
    # md_eitc is a derived reporting total (md_refundable_eitc +
    # md_non_refundable_eitc) that Maryland's tax formula never reads;
    # neutralizing the umbrella is a proven no-op on md_income_tax. The
    # repeal must target both components on the tax path.
    from microcosm.build.us_runtime.reform_validation import (
        state_program_reform_specs,
    )

    spec = next(
        s
        for s in state_program_reform_specs(period=2024)
        if s.id == "state_repeal_md_eitc"
    )
    assert sorted(spec.neutralized_variable) == [
        "md_non_refundable_eitc",
        "md_refundable_eitc",
    ]


def test_neutralize_list_reform_targets_each_variable():
    # The list form of neutralized_variable must neutralize every listed
    # variable in one Reform (used where an umbrella total is off the tax
    # path and repeal needs its components). build_reform() lazily imports
    # policyengine_core, which the CI unit-test env intentionally lacks.
    pytest.importorskip("policyengine_core")
    from microcosm.build.us_runtime.reform_validation import (
        ReformValidationSpec,
    )

    spec = ReformValidationSpec(
        id="x",
        name="x",
        category="State program",
        in_sample=False,
        period=2024,
        jct_score=1.0,
        jct_window="",
        jct_source="",
        jct_source_url="",
        neutralized_variable=["a_var", "b_var"],
    )

    class _Recorder:
        def __init__(self):
            self.neutralized = []

        def neutralize_variable(self, name):
            self.neutralized.append(name)

    reform_cls = spec.build_reform()
    recorder = _Recorder()
    reform_cls.apply(recorder)
    assert recorder.neutralized == ["a_var", "b_var"]


class _FakeArraySeries:
    """Array-valued series with weights, mimicking a MicroSeries for np.asarray."""

    def __init__(self, values, weights) -> None:
        import numpy as np

        self._values = np.asarray(values)
        self.weights = np.asarray(weights)

    def __array__(self, dtype=None):
        return self._values if dtype is None else self._values.astype(dtype)

    def __len__(self) -> int:
        return len(self._values)


class _FakePersonSim:
    """Person-level fake for rate levels: named arrays plus person weights."""

    def __init__(self, arrays: dict, weights) -> None:
        self._arrays = arrays
        self._weights = weights

    def calculate(self, measure: str, period, map_to=None):  # noqa: ARG002
        return _FakeArraySeries(self._arrays[measure], self._weights)


def test_rate_level_computes_state_sliced_person_share():
    from microcosm.build.us_runtime.reform_validation import BaselineLevelSpec

    # Four persons: two in AL (fips 1), two in CA (fips 6).
    sim = _FakePersonSim(
        {
            "in_poverty": [1, 0, 1, 1],
            "is_child": [1, 1, 0, 1],
            "state_fips": [1, 1, 6, 6],
        },
        weights=[10.0, 30.0, 20.0, 20.0],
    )

    def spec(id_, *, state=None, mask=None, value=0.5):
        return BaselineLevelSpec(
            id=id_,
            name=id_,
            variable="in_poverty",
            period=2024,
            benchmark_value=value,
            benchmark_year="2022-2024",
            source="Census",
            source_url="https://census.gov/",
            statistic="rate",
            state=state,
            mask_variable=mask,
        )

    payload = reform_validation_payload(
        [],
        period=2024,
        simulate=lambda reform: sim,
        baseline_levels=[
            spec("al_overall", state="AL"),
            spec("ca_overall", state="CA"),
            spec("us_child", mask="is_child"),
            spec("al_child", state="AL", mask="is_child"),
        ],
    )
    rows = {r["id"]: r for r in payload["reforms"]}
    # AL overall: poor weight 10 of 40 residents.
    assert rows["al_overall"]["microcosm"]["budget_effect"] == pytest.approx(0.25)
    # CA overall: both poor.
    assert rows["ca_overall"]["microcosm"]["budget_effect"] == pytest.approx(1.0)
    # National child: poor children 10 + 20 of child weight 10 + 30 + 20.
    assert rows["us_child"]["microcosm"]["budget_effect"] == pytest.approx(0.5)
    # AL child: poor child 10 of child weight 40.
    assert rows["al_child"]["microcosm"]["budget_effect"] == pytest.approx(0.25)
    for row in rows.values():
        assert row["unit"] == "percent"
        assert row["in_sample"] is False


def test_rate_spec_validation_errors():
    from microcosm.build.us_runtime.reform_validation import BaselineLevelSpec

    common = dict(
        name="x",
        period=2024,
        benchmark_value=0.1,
        benchmark_year="2024",
        source="s",
        source_url="u",
    )
    with pytest.raises(ValueError):
        BaselineLevelSpec(id="bad", variable="in_poverty", statistic="share", **common)
    with pytest.raises(ValueError):
        BaselineLevelSpec(id="bad", variable=["a", "b"], statistic="rate", **common)
    with pytest.raises(ValueError):
        BaselineLevelSpec(
            id="bad",
            variable="in_poverty",
            statistic="rate",
            cap_variable="income_tax",
            **common,
        )


def test_shipped_spm_poverty_config_well_formed():
    from microcosm.build.us_runtime.reform_validation import (
        state_spm_poverty_level_specs,
    )

    specs = state_spm_poverty_level_specs()
    assert len(specs) == 104  # 51 jurisdictions x (overall, child) + 2 national
    ids = [s.id for s in specs]
    assert len(set(ids)) == len(ids)
    states = {s.state for s in specs if s.state}
    assert len(states) == 51
    for s in specs:
        assert s.statistic == "rate"
        assert s.variable == "in_poverty"
        assert 0.03 < s.benchmark_value < 0.25
        assert "census.gov" in s.source_url
        assert s.category == "Census state SPM"
        assert s.benchmark_score_type == "actual"
        if "child" in s.id:
            assert s.mask_variable == "is_child"
        else:
            assert s.mask_variable is None
    # Total-statistic rows keep the currency unit.
    payload = reform_validation_payload([], period=2024, simulate=None)
    assert payload["reforms"] == []


def test_default_simulate_factory_declares_county_spm_selection(monkeypatch, tmp_path):
    """Both release simulations name county SPM measurement explicitly.

    PolicyEngine-US 2.0.0 stopped inferring SPM geography from an absent
    county; the release H5 carries observed county FIPS, so the factory must
    declare county measurement on the baseline and on every reform rather
    than inherit whatever the engine default becomes.
    """

    import sys
    import types

    calls: list[dict[str, object]] = []

    class _RecordingMicrosimulation:
        def __init__(self, **kwargs: object) -> None:
            calls.append(dict(kwargs))

    class _RecordingDataset:
        def __init__(self, *, file_path: str) -> None:
            self.file_path = file_path

    fake_engine = types.ModuleType("policyengine_us")
    fake_engine.Microsimulation = _RecordingMicrosimulation
    fake_data = types.ModuleType("policyengine_us.data")
    fake_data.USSingleYearDataset = _RecordingDataset
    fake_engine.data = fake_data
    monkeypatch.setitem(sys.modules, "policyengine_us", fake_engine)
    monkeypatch.setitem(sys.modules, "policyengine_us.data", fake_data)

    dataset_path = tmp_path / "release.h5"
    simulate = reform_validation_module.default_simulate_factory(dataset_path)

    simulate(None)
    simulate("a reform")

    baseline, reformed = calls
    expected = {"geography_kind": "county"}
    assert reform_validation_module.US_RELEASE_SPM_SELECTION == expected
    assert baseline["spm"] == expected
    assert reformed["spm"] == expected
    assert "reform" not in baseline
    assert reformed["reform"] == "a reform"
    assert baseline["dataset"].file_path == str(dataset_path)
    assert reformed["dataset"].file_path == str(dataset_path)


def _spec(**overrides):
    fields = {
        "id": "invented_reform",
        "name": "Invented reform",
        "category": "Invented category",
        "in_sample": False,
        "period": 2031,
        "jct_score": 17.0,
        "jct_window": "Invented window",
        "jct_source": "Invented source mentioning JCT",
        "jct_source_url": "https://example.test/invented",
        "neutralized_variable": "invented_credit",
    }
    fields.update(overrides)
    return reform_validation_module.ReformValidationSpec(**fields)


def _level(**overrides):
    fields = {
        "id": "invented_level",
        "name": "Invented level",
        "variable": "invented_total",
        "period": 2032,
        "benchmark_value": 43.0,
        "benchmark_year": "Invented year",
        "source": "Invented source mentioning IRS",
        "source_url": "https://example.test/level",
    }
    fields.update(overrides)
    return reform_validation_module.BaselineLevelSpec(**fields)


class _FakeTotal:
    def __init__(self, total):
        self.total = total

    def sum(self):
        return self.total


class _FakeSimulation:
    def __init__(self, total, calls):
        self.total = total
        self.calls = calls

    def calculate(self, measure, period):
        self.calls.append((measure, period))
        return _FakeTotal(self.total)


@pytest.mark.parametrize("factory", [_spec, _level])
def test_claim_benchmark_publisher_is_optional_and_explicit(factory):
    assert factory().benchmark_publisher is None
    assert factory(benchmark_publisher="invented_agency").benchmark_publisher == (
        "invented_agency"
    )


@pytest.mark.parametrize(
    "loader,benchmark_key,definition",
    [
        (
            reform_validation_module.out_of_sample_reform_specs,
            "jct",
            {"parameter_changes": {"gov.invented.amount": {"2031": 7}}},
        ),
        (
            reform_validation_module.tax_expenditure_reform_specs,
            "benchmark",
            {"neutralized_variable": "invented_credit"},
        ),
        (
            reform_validation_module.state_program_reform_specs,
            "benchmark",
            {"neutralized_variable": ["invented_a", "invented_b"]},
        ),
        (
            reform_validation_module.state_reform_specs,
            "benchmark",
            {"parameter_changes": {"gov.invented.amount": {"2031": 7}}},
        ),
    ],
)
@pytest.mark.parametrize(
    "publisher_fields,expected",
    [
        ({}, None),
        ({"publisher": None}, None),
        ({"publisher": ""}, None),
        ({"publisher": "invented_agency"}, "invented_agency"),
    ],
)
def test_claim_reform_loaders_preserve_publisher_without_source_inference(
    tmp_path, loader, benchmark_key, definition, publisher_fields, expected
):
    raw = {
        "id": "invented_reform",
        "name": "Invented reform",
        "period": 2031,
        **definition,
        benchmark_key: {
            "score": -17.0,
            "window": "Invented window",
            "source": "Invented source mentioning JCT and IRS",
            "source_url": "https://example.test/source",
            **publisher_fields,
        },
    }
    path = tmp_path / "invented.json"
    path.write_text(json.dumps({"reforms": [raw]}))
    (spec,) = loader(path, period=2099)
    assert spec.benchmark_publisher == expected
    assert spec.period == 2031
    assert spec.jct_score == -17.0
    assert spec.jct_window == "Invented window"
    assert spec.jct_source == raw[benchmark_key]["source"]
    assert spec.jct_source_url == raw[benchmark_key]["source_url"]


@pytest.mark.parametrize(
    "loader",
    [
        reform_validation_module.soi_baseline_level_specs,
        reform_validation_module.state_program_level_specs,
        reform_validation_module.federal_eitc_state_level_specs,
        reform_validation_module.state_spm_poverty_level_specs,
    ],
)
@pytest.mark.parametrize(
    "publisher_fields,expected",
    [
        ({}, None),
        ({"publisher": None}, None),
        ({"publisher": ""}, None),
        ({"publisher": "invented_agency"}, "invented_agency"),
    ],
)
def test_claim_baseline_loader_wrappers_preserve_explicit_publisher(
    tmp_path, loader, publisher_fields, expected
):
    raw = {
        "id": "invented_level",
        "name": "Invented level",
        "variable": "invented_total",
        "period": 2032,
        "benchmark": {
            "value": 43.0,
            "year": "Invented year",
            "source": "Invented source mentioning Census and IRS",
            "source_url": "https://example.test/level",
            **publisher_fields,
        },
    }
    path = tmp_path / "invented.json"
    path.write_text(json.dumps({"levels": [raw]}))
    (spec,) = loader(path)
    assert spec.benchmark_publisher == expected
    assert spec.period == 2032
    assert spec.benchmark_value == 43.0
    assert spec.benchmark_year == "Invented year"
    assert spec.source == raw["benchmark"]["source"]
    assert spec.source_url == raw["benchmark"]["source_url"]


def test_claim_in_sample_factory_identifies_jct_without_simulation():
    reform = SimpleTaxExpenditureReform(
        target_name="invented_target",
        neutralized_variable="invented_credit",
        measure="invented_target",
        period=2001,
        source="Invented target source",
        output_variable="income_tax",
    )
    (spec,) = reform_validation_module.in_sample_reform_specs([reform], period=2031)
    assert spec.benchmark_publisher == "jct"
    assert spec.period == 2031
    assert spec.jct_source == "Invented target source"

    def simulate(_):
        pytest.fail("A supplied calibration estimate must not run a simulation")

    payload = reform_validation_module.reform_validation_payload(
        [spec],
        period=2030,
        simulate=simulate,
        in_sample_estimates={"invented_target": 11.0},
        in_sample_targets={"invented_target": 13.0},
    )
    row = payload["reforms"][0]
    assert row["jct"]["publisher"] == "jct"
    assert row["jct"]["score"] == 13.0
    assert row["microcosm"]["budget_effect"] == 11.0
    assert row["microcosm"]["measure"] == "income_tax"
    assert row["microcosm"]["baseline_total"] is None
    assert row["microcosm"]["reform_total"] is None
    assert "scoring_comparison" not in row


@pytest.mark.parametrize(
    "direction,expected_effect",
    [("reform_minus_baseline", 20.0), ("baseline_minus_reform", -20.0)],
)
def test_claim_parameter_definition_is_additive_and_preserves_actual_sign_and_year(
    monkeypatch, tmp_path, direction, expected_effect
):
    changes = {"gov.invented.amount": {"2031-01-01.2031-12-31": 7}}
    spec = _spec(
        neutralized_variable=None,
        parameter_changes=changes,
        effect_direction=direction,
        benchmark_publisher="invented_agency",
    )
    token = object()
    builds = []

    def build_reform(value):
        builds.append(value)
        return token

    monkeypatch.setattr(
        reform_validation_module.ReformValidationSpec, "build_reform", build_reform
    )
    simulations = []
    calculations = []

    def simulate(reform):
        simulations.append(reform)
        assert reform is None or reform is token
        return _FakeSimulation(100.0 if reform is None else 120.0, calculations)

    payload = reform_validation_module.reform_validation_payload(
        [spec], period=2030, simulate=simulate, release_id="invented_release"
    )
    assert simulations == [None, token]
    assert builds == [spec]
    assert calculations == [("income_tax", 2031), ("income_tax", 2031)]
    row = payload["reforms"][0]
    assert row["reform"] == {
        "framework": "policyengine_us",
        "parameter_changes": changes,
        "neutralized_variables": None,
        "effect_direction": direction,
    }
    assert row["jct"]["publisher"] == "invented_agency"
    assert row["scoring_comparison"] == {
        "baseline": {
            "framework": "policyengine_us",
            "parameter_changes": None,
            "neutralized_variables": None,
        },
        "reform": {
            "framework": "policyengine_us",
            "parameter_changes": changes,
            "neutralized_variables": None,
        },
        "effect_direction": direction,
    }
    legacy = deepcopy(payload)
    legacy["reforms"][0].pop("reform")
    legacy["reforms"][0].pop("scoring_comparison")
    legacy["reforms"][0]["jct"].pop("publisher")
    assert legacy == {
        "schema_version": 1,
        "baseline_period": 2030,
        "scoring_window": "see per-reform jct.window",
        "out_of_sample_simulated": True,
        "release_id": "invented_release",
        "reforms": [
            {
                "id": "invented_reform",
                "name": "Invented reform",
                "category": "Invented category",
                "in_sample": False,
                "period": 2031,
                "description": None,
                "jct": {
                    "score": 17.0,
                    "score_fy2027": None,
                    "score_type": "conventional",
                    "window": "Invented window",
                    "source": "Invented source mentioning JCT",
                    "source_url": "https://example.test/invented",
                },
                "microcosm": {
                    "budget_effect": expected_effect,
                    "period": 2031,
                    "window": "Invented window",
                    "measure": "income_tax",
                    "baseline_total": 100.0,
                    "reform_total": 120.0,
                },
            }
        ],
    }
    path = reform_validation_module.write_reform_validation(
        payload, tmp_path / "payload.json"
    )
    assert json.loads(path.read_text()) == payload
    row["scoring_comparison"]["reform"]["parameter_changes"]["gov.invented.amount"][
        "2031-01-01.2031-12-31"
    ] = 999
    assert changes["gov.invented.amount"]["2031-01-01.2031-12-31"] == 7
    assert row["reform"]["parameter_changes"] == changes


@pytest.mark.parametrize(
    "neutralized,expected",
    [
        ("invented_credit", ["invented_credit"]),
        (["invented_a", "invented_b"], ["invented_a", "invented_b"]),
        (
            ["invented_b", "invented_a", "invented_b"],
            ["invented_b", "invented_a", "invented_b"],
        ),
    ],
)
def test_claim_neutralizations_are_normalized_and_detached_from_inputs(
    neutralized, expected
):
    spec = _spec(neutralized_variable=neutralized)
    first = reform_validation_module.reform_validation_payload([spec], period=2031)
    second = reform_validation_module.reform_validation_payload([spec], period=2031)
    definition = first["reforms"][0]["reform"]
    assert definition == {
        "framework": "policyengine_us",
        "parameter_changes": None,
        "neutralized_variables": expected,
        "effect_direction": "reform_minus_baseline",
    }
    definition["neutralized_variables"].append("mutated_output")
    assert second["reforms"][0]["reform"]["neutralized_variables"] == expected
    assert (
        reform_validation_module.reform_validation_payload([spec], period=2031)[
            "reforms"
        ][0]["reform"]["neutralized_variables"]
        == expected
    )
    assert spec.neutralized_variable == neutralized
    if isinstance(neutralized, list):
        assert neutralized == expected


def test_claim_parameter_definitions_deepcopy_nested_inputs_across_payloads():
    changes = {"gov.invented.amount": {"2031": [7, {"value": 9}]}}
    original = deepcopy(changes)
    spec = _spec(neutralized_variable=None, parameter_changes=changes)
    first = reform_validation_module.reform_validation_payload([spec], period=2031)
    second = reform_validation_module.reform_validation_payload([spec], period=2031)
    output_changes = first["reforms"][0]["reform"]["parameter_changes"]
    output_changes["gov.invented.amount"]["2031"][1]["value"] = 999
    output_changes["gov.invented.added"] = {"2031": 2}
    assert changes == original
    assert second["reforms"][0]["reform"]["parameter_changes"] == original
    third = reform_validation_module.reform_validation_payload([spec], period=2031)
    assert third["reforms"][0]["reform"]["parameter_changes"] == original
    changes["gov.invented.amount"]["2031"][1]["value"] = 123
    assert second["reforms"][0]["reform"]["parameter_changes"] == original
    assert third["reforms"][0]["reform"]["parameter_changes"] == original


@pytest.mark.parametrize("publisher", [None, "invented_agency"])
def test_claim_baseline_rows_publish_attribution_without_reform_definition(publisher):
    level = _level(benchmark_publisher=publisher)
    simulations = []
    calculations = []

    def simulate(reform):
        simulations.append(reform)
        return _FakeSimulation(43.0, calculations)

    payload = reform_validation_module.reform_validation_payload(
        [], period=2030, simulate=simulate, baseline_levels=[level]
    )
    (row,) = payload["reforms"]
    assert "reform" not in row
    assert "scoring_comparison" not in row
    assert row["jct"]["publisher"] == publisher
    assert row["jct"]["source"] == "Invented source mentioning IRS"
    assert row["jct"]["score"] == 43.0
    assert row["microcosm"]["budget_effect"] == 43.0
    assert row["microcosm"]["reform_total"] is None
    assert simulations == [None]
    assert calculations == [("invented_total", 2032)]


def test_claim_unsimulated_rows_publish_definition_and_null_publisher():
    payload = reform_validation_module.reform_validation_payload(
        [_spec()], period=2030, baseline_levels=[_level()]
    )
    assert payload["schema_version"] == 1
    assert payload["out_of_sample_simulated"] is False
    reform, level = payload["reforms"]
    assert reform["reform"]["neutralized_variables"] == ["invented_credit"]
    assert "reform" not in level
    for row in (reform, level):
        assert "scoring_comparison" not in row
        assert row["jct"]["publisher"] is None
        assert row["microcosm"]["budget_effect"] is None
        assert row["microcosm"]["baseline_total"] is None
        assert row["microcosm"]["reform_total"] is None
        assert "populace" not in row


@pytest.mark.parametrize(
    "neutralized,changes",
    [(None, None), ([], {}), ("invented_credit", {"gov.invented": {"2031": 7}})],
)
def test_claim_ambiguous_or_absent_definitions_are_rejected(neutralized, changes):
    with pytest.raises(ValueError, match="provide exactly one"):
        _spec(neutralized_variable=neutralized, parameter_changes=changes)


def test_claim_invalid_direction_is_rejected_before_reporting():
    with pytest.raises(ValueError, match="effect_direction"):
        _spec(effect_direction="invented_unsupported_direction")


def test_claim_conflicting_obbba_patches_fail_before_callback():
    specs = [
        _spec(
            id=f"invented_{value}",
            category="OBBBA",
            neutralized_variable=None,
            parameter_changes={"gov.invented.amount": {"2031": value}},
        )
        for value in (7, 9)
    ]

    def simulate(_):
        pytest.fail("Conflicting patches must fail before any callback")

    with pytest.raises(ValueError, match="conflicts with"):
        reform_validation_module.reform_validation_payload(
            specs, period=2031, simulate=simulate
        )
