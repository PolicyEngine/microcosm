"""US release input-column coverage + reform-coverage smoke, isolated from PE-US.

populace #368. The five acceptance cases the brief pins:

1. a full required set present with signal passes;
2. a missing required column fails, named;
3. a required column present but degenerate (every value the engine default)
   fails without a reviewed exclusion;
4. a reviewed-exclusion column that has caught up (present with signal) is stale
   and fails (#286 cannot-rot);
5. a bound reform scoring ~$0 fails the reform-coverage smoke.

The frame is a real :class:`~populace.frame.Frame`; the engine is a stub exposing
only ``default_values`` (the surface the gate uses) and the simulation is injected
(and ``_build_reform`` monkeypatched), so nothing here imports policyengine-us —
the same isolation ``test_reform_validation`` relies on. Separate tests assert the
shipped manifest keeps the #368 red-by-design guarantee: the SSI countable-resource
assets stay hard requirements with no exclusion, and demoting one is rejected.
"""

from __future__ import annotations

import importlib.util
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.reform_coverage_smoke as smoke_module
from populace.build.us_runtime import (
    SSI_COUNTABLE_RESOURCE_ASSETS,
    US_QBI_OUTPUT_COLUMNS,
    US_RELEASE_INPUT_COVERAGE_RESOURCE,
    ReformCoverageProbe,
    ReleaseInputColumn,
    ReleaseInputCoverageManifest,
    assert_release_input_coverage_manifest_current,
    load_release_input_coverage_manifest,
    us_reform_coverage_smoke_gate,
    us_release_input_coverage_gate,
    us_release_reform_coverage_probes,
)
from populace.frame import EntitySchema, Frame, WeightKind, Weights

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_GENERATOR = (
    _REPO_ROOT / "tools" / "build_us_release_input_coverage_manifest.py"
)


def _person_frame(columns: dict[str, np.ndarray]) -> Frame:
    """A real single-household Frame carrying ``columns`` on the person table."""
    n = len(next(iter(columns.values())))
    person = pd.DataFrame(
        {
            "person_id": np.arange(n, dtype="int64"),
            "person_household_id": np.ones(n, dtype="int64"),
            **{name: np.asarray(values) for name, values in columns.items()},
        }
    )
    household = pd.DataFrame({"household_id": np.asarray([1], dtype="int64")})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.asarray([1000.0]), kind=WeightKind.DESIGN)},
    )


class _StubEngine:
    """Only ``default_values(names)`` — the single engine surface the gate uses."""

    def __init__(self, defaults: dict[str, object]) -> None:
        self._defaults = dict(defaults)

    def default_values(self, names) -> dict[str, object]:
        return {name: self._defaults[name] for name in names if name in self._defaults}


def _manifest(
    columns: tuple[ReleaseInputColumn, ...],
    probes: tuple[ReformCoverageProbe, ...] = (),
) -> ReleaseInputCoverageManifest:
    return ReleaseInputCoverageManifest(
        reference={"source": "test"}, columns=columns, probes=probes
    )


# A two-required-plus-one-excluded contract, reused across the gate cases.
_CONTRACT = _manifest(
    (
        ReleaseInputColumn("employment_income", "required"),
        ReleaseInputColumn("stock_assets", "required"),
        ReleaseInputColumn(
            "alimony_income",
            "reviewed_exclusion",
            reason="Residual income-source layer not yet sourced; tracked.",
            issue="PolicyEngine/populace#38",
        ),
    )
)

# Every declared column defaults to 0.0 in the stub engine, so an all-zero
# required column reads as degenerate (present but indistinguishable from absent).
_DEFAULTS = {"employment_income": 0.0, "stock_assets": 0.0, "alimony_income": 0.0}


class TestReleaseInputCoverageGate:
    def test_full_required_set_with_signal_passes(self) -> None:
        # Case 1: both required columns present and carrying signal; the excluded
        # column is absent (dormant), which is reported, not failed.
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0, 12_000.0]),
                "stock_assets": np.asarray([0.0, 1_500.0, 0.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert result.passed
        assert result.failures == ()
        assert result.details["dormant_exclusions"] == ["alimony_income"]

    def test_missing_required_column_fails(self) -> None:
        # Case 2: stock_assets is absent from the export entirely — the silent
        # zero the #368 launch failure rode in on.
        frame = _person_frame({"employment_income": np.asarray([0.0, 52_000.0])})
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert "stock_assets" in result.details["missing"]
        assert any(
            "stock_assets" in failure and "absent" in failure
            for failure in result.failures
        )

    def test_degenerate_required_column_without_exclusion_fails(self) -> None:
        # Case 3: stock_assets is present but every value is the engine default,
        # so the export writer's default-broadcast makes it indistinguishable
        # from absence — and there is no reviewed exclusion to accept it.
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "stock_assets": np.asarray([0.0, 0.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert "stock_assets" in result.details["degenerate_required"]
        assert any(
            "stock_assets" in failure and "default" in failure
            for failure in result.failures
        )

    def test_stale_reviewed_exclusion_fails(self) -> None:
        # Case 4: alimony_income is a reviewed exclusion, but the data caught up
        # — it is now present with signal, so the exclusion is stale and must be
        # promoted to a hard requirement (#286 cannot-rot).
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "stock_assets": np.asarray([0.0, 1_500.0]),
                "alimony_income": np.asarray([0.0, 800.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert result.details["stale_exclusions"] == ["alimony_income"]
        assert any(
            "Stale reviewed exclusions" in failure for failure in result.failures
        )

    def test_absent_and_degenerate_excluded_column_passes(self) -> None:
        # The exclusion accepts both an absent column and a degenerate one: with
        # alimony_income present-but-all-default, the reviewed exclusion holds.
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "stock_assets": np.asarray([0.0, 1_500.0]),
                "alimony_income": np.asarray([0.0, 0.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert result.passed
        assert result.details["reviewed_exclusions"] == {
            "alimony_income": "Residual income-source layer not yet sourced; tracked."
        }


class _Series:
    def __init__(self, total: float) -> None:
        self._total = total

    def sum(self) -> float:
        return self._total


class _Sim:
    """A simulation whose weighted total for the measure is a fixed number."""

    def __init__(self, total: float) -> None:
        self._total = total

    def calculate(self, measure: str, period):  # noqa: ARG002 - stub
        return _Series(self._total)


def _probe(min_abs_effect: float = 1_000_000_000.0) -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="ssi_probe",
        name="SSI asset limits raised to $10k / $20k",
        parameter_changes={
            "gov.ssa.ssi.eligibility.resources.limit.individual": {
                "2024-01-01.2100-12-31": 10000
            }
        },
        budget_measure="ssi",
        binding_inputs=("bank_account_assets", "stock_assets", "bond_assets"),
        min_abs_effect=min_abs_effect,
        reason="Assets absent → countable resources 0 → the relaxation scores $0.",
        issue="PolicyEngine/populace#356",
    )


def _tips_probe() -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="tips_probe",
        name="OBBBA no-tax-on-tips deduction",
        parameter_changes={
            "gov.irs.deductions.tip_income.cap": {"2026-01-01.2026-12-31": 0}
        },
        budget_measure="income_tax",
        binding_inputs=("tip_income", "treasury_tipped_occupation_code"),
        min_abs_effect=100_000_000.0,
        reason="The cap repeal must bind through qualified tip income.",
        issue="PolicyEngine/populace#38",
        effect_direction="baseline_minus_reform",
        period=2026,
        expected_sign="negative",
    )


def _overtime_probe() -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="obbba_no_tax_on_overtime",
        name="OBBBA no-tax-on-overtime deduction",
        parameter_changes={
            "gov.irs.deductions.overtime_income.cap.SINGLE": {
                "2026-01-01.2026-12-31": 0
            }
        },
        budget_measure="income_tax",
        binding_inputs=("fsla_overtime_premium",),
        min_abs_effect=100_000_000.0,
        reason="The cap repeal must bind through the FLSA overtime premium.",
        issue="PolicyEngine/populace#242",
        effect_direction="baseline_minus_reform",
        period=2026,
        expected_sign="negative",
    )


def _auto_loan_probe() -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="obbba_auto_loan_interest",
        name="OBBBA no-tax-on-auto-loan-interest deduction",
        parameter_changes={
            "gov.irs.deductions.auto_loan_interest.cap": {"2026-01-01.2026-12-31": 0}
        },
        budget_measure="income_tax",
        binding_inputs=("qualified_passenger_vehicle_loan_interest",),
        min_abs_effect=100_000_000.0,
        reason="The repeal must bind through qualifying vehicle-loan interest.",
        issue="PolicyEngine/populace#252",
        effect_direction="baseline_minus_reform",
        period=2026,
        expected_sign="negative",
    )


class TestReformCoverageSmokeGate:
    def test_zero_bound_reform_fails(self, monkeypatch) -> None:
        # Case 5: with the asset inputs absent, everyone already passes the SSI
        # resource test, so raising the limit moves nothing — baseline and reform
        # score the same total and the bound reform reads as a coverage hole.
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")

        def simulate(reform):  # noqa: ARG001 - baseline == reform on purpose
            return _Sim(4.0e10)

        result = us_reform_coverage_smoke_gate(
            simulate=simulate, probes=[_probe()], period=2024
        )
        assert not result.passed
        assert result.details["results"]["ssi_probe"]["effect"] == 0.0
        assert "did not bind" in result.failures[0]
        assert "bank_account_assets" in result.failures[0]

    def test_bound_reform_with_effect_passes(self, monkeypatch) -> None:
        # The green counterpart: when the assets are carried, the same reform
        # moves SSI by ~$1.6B (the dense-native reference), clearing the floor.
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")

        def simulate(reform):
            return _Sim(4.16e10 if reform == "REFORM" else 4.0e10)

        result = us_reform_coverage_smoke_gate(
            simulate=simulate, probes=[_probe()], period=2024
        )
        assert result.passed
        assert result.details["results"]["ssi_probe"]["effect"] == pytest.approx(1.6e9)

    def test_wrong_signed_effect_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")

        def simulate(reform):
            return _Sim(3.0e10 if reform == "REFORM" else 4.0e10)

        result = us_reform_coverage_smoke_gate(
            simulate=simulate, probes=[_probe()], period=2024
        )
        assert not result.passed
        assert result.details["results"]["ssi_probe"]["effect"] == -1.0e10
        assert "expected a positive effect" in result.failures[0]

    def test_negative_tip_effect_uses_probe_period_and_passes(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")
        periods: list[int] = []

        class RecordingSim(_Sim):
            def calculate(self, measure: str, period):
                periods.append(period)
                return super().calculate(measure, period)

        def simulate(reform):
            return RecordingSim(10.5e9 if reform == "REFORM" else 10.0e9)

        result = us_reform_coverage_smoke_gate(
            simulate=simulate,
            probes=[_tips_probe()],
            period=2024,
        )

        assert result.passed
        assert periods == [2026, 2026]
        assert result.details["default_period"] == 2024
        tip_result = result.details["results"]["tips_probe"]
        assert tip_result["period"] == 2026
        assert tip_result["effect"] == pytest.approx(-0.5e9)
        assert tip_result["expected_sign"] == "negative"

    def test_negative_overtime_effect_passes_and_wrong_sign_fails(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")

        passing = us_reform_coverage_smoke_gate(
            simulate=lambda reform: _Sim(10.5e9 if reform else 10.0e9),
            probes=[_overtime_probe()],
        )
        assert passing.passed
        assert passing.details["results"]["obbba_no_tax_on_overtime"][
            "effect"
        ] == pytest.approx(-0.5e9)

        wrong_sign = us_reform_coverage_smoke_gate(
            simulate=lambda reform: _Sim(9.5e9 if reform else 10.0e9),
            probes=[_overtime_probe()],
        )
        assert not wrong_sign.passed
        assert "expected a negative effect" in wrong_sign.failures[0]

    def test_negative_auto_loan_effect_passes_and_wrong_sign_fails(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")

        passing = us_reform_coverage_smoke_gate(
            simulate=lambda reform: _Sim(10.5e9 if reform else 10.0e9),
            probes=[_auto_loan_probe()],
        )
        assert passing.passed
        result = passing.details["results"]["obbba_auto_loan_interest"]
        assert result["effect"] == pytest.approx(-0.5e9)
        assert result["period"] == 2026

        wrong_sign = us_reform_coverage_smoke_gate(
            simulate=lambda reform: _Sim(9.5e9 if reform else 10.0e9),
            probes=[_auto_loan_probe()],
        )
        assert not wrong_sign.passed
        assert "expected a negative effect" in wrong_sign.failures[0]

    def test_probeless_gate_is_refused(self) -> None:
        # A probe-less smoke gate would pass vacuously — refuse it.
        with pytest.raises(ValueError, match="at least one probe"):
            us_reform_coverage_smoke_gate(simulate=lambda reform: _Sim(0.0), probes=[])


class TestShippedManifest:
    def test_manifest_is_current(self) -> None:
        # The checked-in-facts half runs everywhere (no engine in this env): the
        # declared surface must equal the reference eCPS populated columns and the
        # SSI assets must stay hard requirements.
        assert_release_input_coverage_manifest_current()

    def test_ssi_assets_are_required_without_exclusion(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for asset in SSI_COUNTABLE_RESOURCE_ASSETS:
            assert asset in manifest.required_columns
            assert asset not in manifest.reviewed_exclusions

    def test_post_reference_obbba_inputs_are_hard_requirements(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in (
            "fsla_overtime_premium",
            "qualified_passenger_vehicle_loan_interest",
        ):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_legacy_auto_loan_columns_are_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in ("auto_loan_balance", "auto_loan_interest"):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_education_input_family_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in (
            "qualified_tuition_expenses",
            "educational_assistance",
            "is_pursuing_credential_for_american_opportunity_credit",
            "attends_eligible_educational_institution_for_american_opportunity_credit",
            "is_enrolled_at_least_half_time_for_american_opportunity_credit",
            "has_american_opportunity_credit_1098_t_or_exception",
            "has_american_opportunity_credit_institution_ein",
        ):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_retirement_contribution_family_is_a_hard_requirement(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in (
            "traditional_401k_contributions_desired",
            "roth_401k_contributions_desired",
            "traditional_ira_contributions_desired",
            "roth_ira_contributions_desired",
            "self_employed_pension_contributions_desired",
        ):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_casualty_loss_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "casualty_loss" in manifest.required_columns
        assert "casualty_loss" not in manifest.reviewed_exclusions

    def test_alimony_family_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in ("alimony_income", "alimony_expense"):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_misc_itemized_input_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        column = "unreimbursed_business_employee_expenses"
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions

    def test_childcare_input_is_a_hard_requirement(self) -> None:
        manifest = load_release_input_coverage_manifest()
        column = "spm_unit_pre_subsidy_childcare_expenses"
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions

    def test_child_support_family_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in ("child_support_received", "child_support_expense"):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_disability_benefits_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "disability_benefits" in manifest.required_columns
        assert "disability_benefits" not in manifest.reviewed_exclusions

    def test_educator_expense_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "educator_expense" in manifest.required_columns
        assert "educator_expense" not in manifest.reviewed_exclusions

    def test_qbi_input_family_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in US_QBI_OUTPUT_COLUMNS:
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_domestic_production_ald_is_promoted_separately_from_qbi(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "domestic_production_ald" in manifest.required_columns
        assert "domestic_production_ald" not in manifest.reviewed_exclusions

    def test_shipped_ssi_probe_binds_through_the_assets(self) -> None:
        probes = us_release_reform_coverage_probes()
        assert probes, "the shipped manifest must pin at least one reform probe"
        ssi = next(probe for probe in probes if probe.id == "ssi_asset_limit_10k_20k")
        assert set(SSI_COUNTABLE_RESOURCE_ASSETS) <= set(ssi.binding_inputs)
        assert ssi.budget_measure == "ssi"
        assert ssi.min_abs_effect > 0
        assert ssi.expected_sign == "positive"

    def test_shipped_tip_probe_has_2026_period_sign_and_inputs(self) -> None:
        tip = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_no_tax_on_tips"
        )
        assert tip.period == 2026
        assert tip.expected_sign == "negative"
        assert tip.effect_direction == "baseline_minus_reform"
        assert tip.budget_measure == "income_tax"
        assert set(tip.binding_inputs) == {
            "tip_income",
            "treasury_tipped_occupation_code",
        }

    def test_shipped_aotc_probe_binds_through_education_inputs(self) -> None:
        aotc = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "aotc_abolition"
        )
        assert aotc.period == 2024
        assert aotc.expected_sign == "positive"
        assert aotc.effect_direction == "baseline_minus_reform"
        assert aotc.budget_measure == "american_opportunity_credit"
        assert set(aotc.binding_inputs) == {
            "qualified_tuition_expenses",
            "is_pursuing_credential_for_american_opportunity_credit",
            "attends_eligible_educational_institution_for_american_opportunity_credit",
            "is_enrolled_at_least_half_time_for_american_opportunity_credit",
            "has_american_opportunity_credit_1098_t_or_exception",
            "has_american_opportunity_credit_institution_ein",
        }
        assert aotc.min_abs_effect > 0
        assert set(aotc.parameter_changes) == {
            "gov.irs.credits.education.american_opportunity_credit.abolition"
        }

    def test_shipped_savers_credit_probe_binds_through_contributions(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "savers_credit_abolition"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "savers_credit"
        assert set(probe.binding_inputs) == {
            "traditional_401k_contributions_desired",
            "roth_401k_contributions_desired",
            "traditional_ira_contributions_desired",
            "roth_ira_contributions_desired",
            "self_employed_pension_contributions_desired",
        }
        assert probe.min_abs_effect == 100_000_000.0
        assert set(probe.parameter_changes) == {
            "gov.irs.credits.retirement_saving.contributions_cap"
        }

    def test_shipped_casualty_probe_has_2026_period_sign_and_input(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_casualty_loss_limit"
        )
        assert probe.period == 2026
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("casualty_loss",)
        assert probe.min_abs_effect == 1_000_000.0
        assert set(probe.parameter_changes) == {
            "gov.irs.deductions.itemized.casualty.active"
        }

    def test_shipped_alimony_probe_has_sign_period_and_expense_input(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "alimony_expense_ald_abolition"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "negative"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("alimony_expense",)
        assert probe.min_abs_effect == 1_000_000.0
        assert set(probe.parameter_changes) == {
            "gov.irs.ald.alimony_expense.divorce_year_threshold[0].amount"
        }

    def test_shipped_misc_itemized_probe_has_2026_period_sign_and_input(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_misc_itemized_deductions"
        )
        assert probe.period == 2026
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("unreimbursed_business_employee_expenses",)
        assert probe.min_abs_effect == 100_000_000.0
        assert set(probe.parameter_changes) == {
            "gov.irs.deductions.itemized.misc.applies"
        }

    def test_shipped_cdcc_probe_has_2026_period_sign_and_input(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_cdcc"
        )
        assert probe.period == 2026
        assert probe.expected_sign == "negative"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("spm_unit_pre_subsidy_childcare_expenses",)
        assert probe.min_abs_effect == 1_000_000.0
        assert set(probe.parameter_changes) == {
            "gov.irs.credits.cdcc.phase_out.max",
            "gov.irs.credits.cdcc.phase_out.min",
            "gov.irs.credits.cdcc.phase_out.amended_structure.applies",
        }

    def test_shipped_child_support_received_probe_removes_only_snap_source(
        self,
    ) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "child_support_received_snap_exclusion"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "reform_minus_baseline"
        assert probe.budget_measure == "snap"
        assert probe.binding_inputs == ("child_support_received",)
        assert probe.min_abs_effect == 1_000_000.0
        assert probe.parameter_changes == {
            "gov.usda.snap.income.sources.unearned": {
                "2024-01-01.2024-12-31": [
                    "ssi",
                    "tanf",
                    "general_assistance",
                    "pension_income",
                    "veterans_benefits",
                    "unemployment_compensation",
                    "disability_benefits",
                    "workers_compensation",
                    "social_security",
                    "retirement_distributions",
                    "rental_income",
                    "alimony_income",
                    "financial_assistance",
                    "survivor_benefits",
                    "dividend_income",
                    "interest_income",
                    "miscellaneous_income",
                ]
            }
        }

    def test_shipped_child_support_expense_probe_removes_only_snap_deduction(
        self,
    ) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "child_support_expense_snap_deduction_abolition"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "snap"
        assert probe.binding_inputs == ("child_support_expense",)
        assert probe.min_abs_effect == 1_000_000.0
        assert probe.parameter_changes == {
            "gov.usda.snap.income.deductions.allowed": {
                "2024-01-01.2024-12-31": [
                    "snap_standard_deduction",
                    "snap_earned_income_deduction",
                    "snap_dependent_care_deduction",
                    "snap_excess_medical_expense_deduction",
                    "snap_excess_shelter_expense_deduction",
                ]
            }
        }

    def test_shipped_disability_probe_removes_only_snap_unearned_source(
        self,
    ) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "disability_benefits_snap_exclusion"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "reform_minus_baseline"
        assert probe.budget_measure == "snap"
        assert probe.binding_inputs == ("disability_benefits",)
        assert probe.min_abs_effect == 1_000_000.0
        assert probe.parameter_changes == {
            "gov.usda.snap.income.sources.unearned": {
                "2024-01-01.2024-12-31": [
                    "ssi",
                    "tanf",
                    "general_assistance",
                    "pension_income",
                    "veterans_benefits",
                    "unemployment_compensation",
                    "workers_compensation",
                    "social_security",
                    "retirement_distributions",
                    "rental_income",
                    "child_support_received",
                    "alimony_income",
                    "financial_assistance",
                    "survivor_benefits",
                    "dividend_income",
                    "interest_income",
                    "miscellaneous_income",
                ]
            }
        }

    def test_shipped_educator_expense_probe_removes_only_its_ald(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "educator_expense_ald_abolition"
        )

        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "reform_minus_baseline"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("educator_expense",)
        assert probe.min_abs_effect == 1_000_000.0
        assert probe.parameter_changes == {
            "gov.irs.ald.deductions": {
                "2024-01-01.2024-12-31": [
                    "loss_ald",
                    "self_employment_tax_ald",
                    "student_loan_interest_ald",
                    "early_withdrawal_penalty",
                    "alimony_expense_ald",
                    "health_savings_account_ald",
                    "self_employed_health_insurance_ald",
                    "self_employed_pension_contribution_ald",
                    "traditional_ira_contributions",
                    "qualified_adoption_assistance_expense",
                    "us_bonds_for_higher_ed",
                    "specified_possession_income",
                    "puerto_rico_income",
                ]
            }
        }

    def test_shipped_qbi_probes_cover_reit_and_wage_property_inputs(self) -> None:
        probes = {probe.id: probe for probe in us_release_reform_coverage_probes()}
        reit = probes["qbi_reit_ptp_rate_abolition"]
        assert reit.period == 2024
        assert reit.expected_sign == "positive"
        assert reit.effect_direction == "baseline_minus_reform"
        assert reit.budget_measure == "qualified_business_income_deduction"
        assert reit.binding_inputs == ("qualified_reit_and_ptp_income",)
        assert set(reit.parameter_changes) == {
            "gov.irs.deductions.qbi.max.reit_ptp_rate"
        }

        guardrails = probes["qbi_wage_property_guardrails_zeroed"]
        assert guardrails.period == 2024
        assert guardrails.expected_sign == "positive"
        assert guardrails.effect_direction == "baseline_minus_reform"
        assert guardrails.budget_measure == "qualified_business_income_deduction"
        assert set(guardrails.binding_inputs) == {
            "w2_wages_from_qualified_business",
            "unadjusted_basis_qualified_property",
        }
        assert set(guardrails.parameter_changes) == {
            "gov.irs.deductions.qbi.max.w2_wages.rate",
            "gov.irs.deductions.qbi.max.w2_wages.alt_rate",
            "gov.irs.deductions.qbi.max.business_property.rate",
        }

    def test_shipped_domestic_production_probe_reactivates_only_its_ald(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "domestic_production_ald_reactivation"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("domestic_production_ald",)
        assert probe.min_abs_effect == 1_000_000.0
        assert set(probe.parameter_changes) == {"gov.irs.ald.deductions"}
        deductions = probe.parameter_changes["gov.irs.ald.deductions"]
        assert set(deductions) == {"2024-01-01.2024-12-31"}
        assert deductions["2024-01-01.2024-12-31"].count("domestic_production_ald") == 1

    def test_shipped_overtime_probe_has_2026_period_sign_and_input(self) -> None:
        overtime = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_no_tax_on_overtime"
        )
        assert overtime.period == 2026
        assert overtime.expected_sign == "negative"
        assert overtime.effect_direction == "baseline_minus_reform"
        assert overtime.budget_measure == "income_tax"
        assert overtime.binding_inputs == ("fsla_overtime_premium",)
        assert overtime.min_abs_effect > 0
        assert set(overtime.parameter_changes) == {
            "gov.irs.deductions.overtime_income.cap.JOINT",
            "gov.irs.deductions.overtime_income.cap.SINGLE",
            "gov.irs.deductions.overtime_income.cap.HEAD_OF_HOUSEHOLD",
            "gov.irs.deductions.overtime_income.cap.SURVIVING_SPOUSE",
            "gov.irs.deductions.overtime_income.cap.SEPARATE",
        }

    def test_shipped_auto_loan_probe_has_2026_period_sign_and_input(self) -> None:
        auto = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_auto_loan_interest"
        )
        assert auto.period == 2026
        assert auto.expected_sign == "negative"
        assert auto.effect_direction == "baseline_minus_reform"
        assert auto.budget_measure == "income_tax"
        assert auto.binding_inputs == ("qualified_passenger_vehicle_loan_interest",)
        assert set(auto.parameter_changes) == {
            "gov.irs.deductions.auto_loan_interest.cap"
        }

    def test_demoting_an_ssi_asset_to_exclusion_is_rejected(self) -> None:
        # The #368 red-gate guarantee cannot be quietly undone: turning an SSI
        # asset into a reviewed exclusion must fail the anti-rot assertion.
        manifest = load_release_input_coverage_manifest()
        tampered_columns = tuple(
            ReleaseInputColumn(
                column.name,
                "reviewed_exclusion",
                reason="pretend this gap is tracked",
                issue="PolicyEngine/populace#000",
            )
            if column.name == "stock_assets"
            else column
            for column in manifest.columns
        )
        tampered = ReleaseInputCoverageManifest(
            reference=manifest.reference,
            columns=tampered_columns,
            probes=manifest.probes,
        )
        with pytest.raises(ValueError, match="stock_assets"):
            assert_release_input_coverage_manifest_current(
                manifest=tampered, engine=None
            )

    def test_duplicate_probe_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate reform coverage probe id"):
            _manifest(_CONTRACT.columns, probes=(_probe(), _probe()))


def _load_manifest_generator():
    spec = importlib.util.spec_from_file_location(
        "build_us_release_input_coverage_manifest", _MANIFEST_GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestManifestGeneratorSync:
    def test_committed_manifest_matches_regeneration(self) -> None:
        # The committed manifest is derivable purely from the checked-in eCPS
        # parity reference + known-gaps register, so it cannot drift from them:
        # regenerating must reproduce the committed file byte-for-value.
        generator = _load_manifest_generator()
        committed = json.loads(
            files("populace.build.us")
            .joinpath(US_RELEASE_INPUT_COVERAGE_RESOURCE)
            .read_text(encoding="utf-8")
        )
        assert generator.build_manifest() == committed

    def test_generated_manifest_names_no_retired_data_package(self) -> None:
        # The manifest is not on the incumbent-reference allow-list, so its
        # provenance block must not name the retired data package (the guard
        # test_us_plan.test_no_incumbent_data_package_references_in_live_tree
        # enforces on the committed file; this pins the generator too).
        generator = _load_manifest_generator()
        rendered = json.dumps(generator.build_manifest())
        # Build the needles by concatenation so this test file does not itself
        # trip the live-tree guard it mirrors (test_us_plan does the same).
        assert ("policyengine-" + "us-data") not in rendered
        assert ("policyengine_" + "us_data") not in rendered
