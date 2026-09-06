"""Actual policyengine-us behaviour behind the housing receipt/charge consumer.

Small invented households only. The probes run the installed engine through
Microcosm's real export gate and simulation seam, so each assertion below is a
measured engine fact rather than a restatement of a formula.
"""

import numpy as np
import pytest

pytest.importorskip("policyengine_us")

from microcosm.build.us_runtime import (  # noqa: E402
    asec_housing_receipt_consumer as consumer,
)
from microcosm.build.us_runtime import housing_charge_probes as probes  # noqa: E402

# SPM-unit row order in the invented frame.
SOLE_ASSISTED = 0
SHARED_FIRST = 1  # two people, household 2
SHARED_SECOND = 2  # one person, household 2
UTILITY = 3
OWNER_FLAG_FALSE = 4
OWNER_FLAG_TRUE = 5

# Person row order.
PERSON_SOLE = 0
PERSON_SHARED_A = 1
PERSON_SHARED_B = 2
PERSON_SHARED_C = 3
PERSON_UTILITY = 4


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> dict:
    return probes.run_housing_charge_probes(
        tmp_path_factory.mktemp("us-housing-charge-probe")
    )


class TestFixedIndependentFullCharge:
    def test_the_charge_input_does_not_move_with_assistance_policy(self, probe):
        assert (
            probe["baseline"]["pre_subsidy_rent"]
            == probe["hud_abolition"]["pre_subsidy_rent"]
        )
        assert probe["baseline"]["pre_subsidy_rent"] == [
            12_000.0,
            7_000.0,
            5_000.0,
            9_600.0,
            3_600.0,
            12_000.0,
            12_000.0,
        ]

    def test_the_charge_is_an_engine_input_without_a_formula(self):
        from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

        assert "pre_subsidy_rent" in PolicyEngineUSEngine().variables()


class TestAssistancePolicyMovesHapAndTenantRent:
    def test_abolition_zeroes_assistance_and_lifts_tenant_rent_to_the_charge(
        self, probe
    ):
        baseline, reform = probe["baseline"], probe["hud_abolition"]
        assert baseline["housing_assistance"][SOLE_ASSISTED] == 8_400.0
        assert baseline["rent"][PERSON_SOLE] == 3_600.0
        assert reform["housing_assistance"] == [0.0] * 6
        assert reform["rent"] == reform["pre_subsidy_rent"]

    def test_hap_is_still_computed_under_abolition_so_hap_is_not_assistance(
        self, probe
    ):
        # gov.hud.abolition zeroes housing_assistance only; hud_hap keeps its
        # value, so an observed HAP is not the amount the engine pays.
        assert probe["hud_abolition"]["hud_hap"] == probe["baseline"]["hud_hap"]
        assert (
            probe["hud_abolition"]["housing_assistance"]
            != probe["hud_abolition"]["hud_hap"]
        )

    def test_the_capped_valuation_follows_assistance_and_is_not_it(self, probe):
        baseline = probe["baseline"]
        assert probe["hud_abolition"]["spm_unit_capped_housing_subsidy"] == [0.0] * 6
        capped = baseline["spm_unit_capped_housing_subsidy"]
        assistance = baseline["housing_assistance"]
        assert capped[SOLE_ASSISTED] < assistance[SOLE_ASSISTED]
        assert capped[UTILITY] == assistance[UTILITY]
        assert capped != assistance


class TestUtilityReimbursementCannotCreateNegativeTenantRent:
    def test_hap_exceeds_the_charge_and_tenant_rent_floors_at_zero(self, probe):
        baseline = probe["baseline"]
        charge = baseline["pre_subsidy_rent"][PERSON_UTILITY]
        assert baseline["hud_gross_rent"][UTILITY] == charge + 3_816.0
        assert baseline["hud_hap"][UTILITY] > charge
        assert baseline["rent"][PERSON_UTILITY] == 0.0

    def test_no_probe_household_reports_negative_rent_or_housing_cost(self, probe):
        for block in (probe["baseline"], probe["hud_abolition"]):
            assert min(block["rent"]) >= 0.0
            assert min(block["housing_cost"]) >= 0.0


class TestMultiSpmUnitConservationAndTheReplicationHazard:
    def test_tenant_rent_is_allocated_in_proportion_to_the_person_charge(self, probe):
        baseline = probe["baseline"]
        charges = baseline["pre_subsidy_rent"]
        rents = baseline["rent"]
        unit_charge = charges[PERSON_SHARED_A] + charges[PERSON_SHARED_B]
        unit_rent = rents[PERSON_SHARED_A] + rents[PERSON_SHARED_B]
        assert rents[PERSON_SHARED_A] == pytest.approx(
            charges[PERSON_SHARED_A] / unit_charge * unit_rent
        )
        assert rents[PERSON_SHARED_B] == pytest.approx(
            charges[PERSON_SHARED_B] / unit_charge * unit_rent
        )

    def test_the_household_bill_conserves_across_its_two_spm_units(self, probe):
        baseline = probe["baseline"]
        people = (PERSON_SHARED_A, PERSON_SHARED_B, PERSON_SHARED_C)
        units = (SHARED_FIRST, SHARED_SECOND)
        charge = sum(baseline["pre_subsidy_rent"][index] for index in people)
        rent = sum(baseline["rent"][index] for index in people)
        assistance = sum(baseline["housing_assistance"][index] for index in units)
        assert charge == 21_600.0
        assert rent == pytest.approx(charge - assistance)

    def test_each_spm_unit_independently_draws_the_whole_payment_standard(self, probe):
        baseline = probe["baseline"]
        standard = baseline["pha_payment_standard"][1]
        for unit in (SHARED_FIRST, SHARED_SECOND):
            assert baseline["hud_max_subsidy"][unit] == pytest.approx(
                standard - baseline["hud_ttp"][unit]
            )
        combined = sum(baseline["housing_assistance"][unit] for unit in (1, 2))
        assert combined == 15_300.0
        # Replicating one household answer to both units would fund two whole
        # voucher budgets inside a single housing unit, so the consumer withholds.
        assert combined > baseline["housing_assistance"][SOLE_ASSISTED]

    def test_the_consumer_withholds_exactly_this_shape(self):
        import pandas as pd

        from microcosm.build.us_runtime import asec_housing_status as housing

        observations = pd.DataFrame(
            {
                "household_id": [1],
                "income_year": [2023],
                "survey_year": [2024],
                "H_SEQ": [1],
                "H_TENURE": [2],
                "HRHTYPE": [1],
                "H_HHTYPE": [1],
                "HPUBLIC": [1],
                "HLORENT": [0],
                "I_HPUBLI": [0],
                "I_HLOREN": [0],
            },
            dtype=np.int64,
        )
        status = housing.classify_synthetic_housing_status(observations)
        interpretation = consumer.interpret_household_receipt(
            consumer.receipt_columns_from_status(status)
        )
        scope = consumer.resolve_spm_receipt_scope(
            interpretation,
            household_id=np.array([1], dtype="int64"),
            spm_unit_id=np.array([1, 2], dtype="int64"),
            person_household_id=np.array([1, 1, 1], dtype="int64"),
            person_spm_unit_id=np.array([1, 1, 2], dtype="int64"),
        )
        assert scope.assignment_counts()["withheld_scope_ambiguous_receipt"] == 2


class TestTheReceiptFlagHasRealDollarConsequences:
    def test_only_the_flag_differs_and_it_moves_assistance_and_rent(self, probe):
        baseline = probe["baseline"]
        assert baseline["is_eligible_for_housing_assistance"][OWNER_FLAG_FALSE] is False
        assert baseline["is_eligible_for_housing_assistance"][OWNER_FLAG_TRUE] is True
        assert baseline["housing_assistance"][OWNER_FLAG_FALSE] == 0.0
        assert baseline["housing_assistance"][OWNER_FLAG_TRUE] == 8_400.0
        assert baseline["rent"][5] == 12_000.0
        assert baseline["rent"][6] == 3_600.0

    def test_the_engine_default_makes_a_wrong_false_silently_cheap_to_assert(self):
        from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

        adapter = PolicyEngineUSEngine()
        assert "receives_housing_assistance" in adapter.variables()
        assert "takes_up_housing_assistance_if_eligible" in adapter.variables()
        # $8,400 a year for one invented unit is what an unjustified False costs,
        # which is why the consumer refuses a blanket engine Boolean.
        with pytest.raises(consumer.HousingReceiptRefusalError):
            consumer.compose_full_contract_charge(
                reported_contract_rent=np.array([1.0]),
                capped_spm_valuation=np.array([1.0]),
            )


class TestFormulaOwnedOutputsStayRefused:
    def test_every_formula_owned_housing_output_is_refused_as_an_input(self, probe):
        refusals = probe["formula_owned_export_refusals"]
        assert set(refusals) == set(probes.FORMULA_OWNED_HOUSING_OUTPUTS)
        for name, message in refusals.items():
            assert "formula-owned column(s) present" in message
            assert name in message
            assert "nothing was written" in message
            assert "FILE WRITTEN" not in message

    def test_the_consumer_seam_names_the_same_refused_exports(self, probe):
        seam = consumer.describe_integration_seam()
        assert set(seam["refuses_to_export"]) == set(
            probe["formula_owned_export_refusals"]
        )


class TestKnownConsumerDefects:
    def test_the_bundled_fmr_table_has_no_2024_and_falls_forward_to_fy2025(self, probe):
        resolution = probe["fmr_year_resolution"]
        assert 2024 not in resolution["available_fmr_years"]
        assert resolution["resolved_year"] == 2025
        assert resolution["falls_forward"] is True

    def test_the_consumer_reports_the_charge_as_unavailable_for_that_reason(self):
        availability = consumer.counterfactual_charge_availability()
        assert availability["available"] is False
        assert any("FY2025" in defect for defect in availability["blocking_defects"])

    def test_the_utility_allowance_is_zero_outside_the_encoded_counties(self, probe):
        # Every probe household sits in one encoded county; only the household
        # that pays its own utilities receives an allowance.
        allowances = probe["baseline"]["hud_utility_allowance"]
        assert allowances == [0.0, 0.0, 3_816.0, 0.0, 0.0]


class TestProbeProvenance:
    def test_the_receipt_marks_itself_invented_and_not_release_eligible(self, probe):
        assert probe["inputs_are_invented"] is True
        assert probe["release_eligible"] is False
        assert probe["engine"]["policyengine-us"] == "1.819.0"
        assert probe["period"] == 2024
