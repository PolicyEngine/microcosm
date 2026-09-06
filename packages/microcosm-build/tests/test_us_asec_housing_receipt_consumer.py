"""Household-to-SPM receipt interpretation over invented ASEC housing statuses.

Every fixture is explicitly invented and stays ``synthetic_unverified``; the real
production classifier derives the carried status columns the consumer reads, so
these tests exercise the shipped kernel rather than a restatement of it.
"""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import asec_housing_receipt_consumer as consumer
from microcosm.build.us_runtime import asec_housing_status as housing
from microcosm.build.us_runtime.asec_housing_status_source import ATTACHED_COLUMNS

RECEIPT_PUBLIC = {"HPUBLIC": 1, "HLORENT": 0}
RECEIPT_LOWER = {"HPUBLIC": 2, "HLORENT": 1}
NONRECEIPT = {"HPUBLIC": 2, "HLORENT": 2}
UNKNOWN_PARENT = {"HPUBLIC": 0, "HLORENT": 0}
UNKNOWN_LOWER = {"HPUBLIC": 2, "HLORENT": 0}
CONFLICT = {"HPUBLIC": 1, "HLORENT": 2}
OWNER = {"H_TENURE": 1, "HPUBLIC": 0, "HLORENT": 0}


def observations(rows):
    """Invented households; one dict per row overrides the renter-nonreceipt base."""
    base = {
        "income_year": 2023,
        "survey_year": 2024,
        "H_TENURE": 2,
        "HRHTYPE": 1,
        "H_HHTYPE": 1,
        "HPUBLIC": 2,
        "HLORENT": 2,
        "I_HPUBLI": 0,
        "I_HLOREN": 0,
    }
    records = []
    for index, row in enumerate(rows, start=1):
        record = {"household_id": index, "H_SEQ": index, **base, **row}
        records.append(record)
    return pd.DataFrame(records, dtype=np.int64)


def status_for(rows):
    return housing.classify_synthetic_housing_status(observations(rows))


def interpretation_for(rows):
    return consumer.interpret_household_receipt(
        consumer.receipt_columns_from_status(status_for(rows))
    )


def scope_for(rows, spm_units_per_household):
    """One person per SPM unit, SPM units distributed over the invented households."""
    interpretation = interpretation_for(rows)
    household_id, spm_unit_id = [], []
    person_household_id, person_spm_unit_id = [], []
    next_spm = 1
    for index, count in enumerate(spm_units_per_household, start=1):
        household_id.append(index)
        for _ in range(count):
            spm_unit_id.append(next_spm)
            person_household_id.append(index)
            person_spm_unit_id.append(next_spm)
            next_spm += 1
    return consumer.resolve_spm_receipt_scope(
        interpretation,
        household_id=np.array(household_id, dtype="int64"),
        spm_unit_id=np.array(spm_unit_id, dtype="int64"),
        person_household_id=np.array(person_household_id, dtype="int64"),
        person_spm_unit_id=np.array(person_spm_unit_id, dtype="int64"),
    )


class TestHouseholdInterpretation:
    @pytest.mark.parametrize(
        "row,view",
        [
            (RECEIPT_PUBLIC, "observed_receipt"),
            (RECEIPT_LOWER, "observed_receipt"),
            (NONRECEIPT, "observed_nonreceipt"),
            (UNKNOWN_PARENT, "unknown"),
            (UNKNOWN_LOWER, "unknown"),
            (CONFLICT, "conflict"),
            (OWNER, "not_in_universe"),
        ],
    )
    def test_view_renames_the_carried_status_without_rederiving_it(self, row, view):
        interpretation = interpretation_for([row])
        assert consumer.RECEIPT_VIEWS[int(interpretation.receipt_view[0])] == view
        assert interpretation.receipt_view.tolist() == status_for(
            [row]
        ).array("status").tolist()

    @pytest.mark.parametrize("row", [UNKNOWN_PARENT, UNKNOWN_LOWER, CONFLICT, OWNER])
    def test_no_invalid_household_becomes_observed_nonreceipt(self, row):
        interpretation = interpretation_for([row])
        nonreceipt = consumer.RECEIPT_VIEWS.index("observed_nonreceipt")
        assert int(interpretation.receipt_view[0]) != nonreceipt
        assert int(interpretation.receipt_valid[0]) == 0

    def test_frozen_zero_origin_is_preserved_not_collapsed(self):
        interpretation = interpretation_for([RECEIPT_PUBLIC])
        # HLORENT is routed out by a public-housing yes; its zero keeps the
        # unresolved frozen-fill origin rather than becoming an observed no.
        assert int(interpretation.zero_origin["HLORENT_zero_origin"][0]) == 1
        assert int(interpretation.zero_origin["HPUBLIC_zero_origin"][0]) == 0

    def test_allocation_quality_stays_a_separate_axis(self):
        allocated = interpretation_for([{**RECEIPT_PUBLIC, "I_HPUBLI": 1}])
        unallocated = interpretation_for([RECEIPT_PUBLIC])
        assert int(allocated.public_quality[0]) == 2
        assert int(unallocated.public_quality[0]) == 1
        assert allocated.receipt_view.tolist() == unallocated.receipt_view.tolist()

    def test_interview_period_kind_is_required(self):
        projection = consumer.receipt_columns_from_status(status_for([NONRECEIPT]))
        projection["source_period_kind"] = "annual_income_year"
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="UNSUPPORTED_PERIOD_KIND"
        ):
            consumer.interpret_household_receipt(projection)

    def test_a_tampered_validity_mask_is_refused(self):
        projection = consumer.receipt_columns_from_status(status_for([UNKNOWN_PARENT]))
        projection["columns"]["receipt_valid"] = np.array([1], dtype="uint8")
        with pytest.raises(
            consumer.HousingReceiptRefusalError,
            match="VALIDITY_DISAGREES_WITH_STATUS",
        ):
            consumer.interpret_household_receipt(projection)

    def test_synthetic_authentication_is_carried_through(self):
        interpretation = interpretation_for([NONRECEIPT])
        assert interpretation.source_authentication == "synthetic_unverified"


class TestAttachedColumnReader:
    def test_reads_the_attached_household_columns(self):
        status = status_for([RECEIPT_PUBLIC, NONRECEIPT])
        household = pd.DataFrame(
            {
                attached: status.array(name)
                for name, attached in zip(
                    housing.DERIVED_COLUMNS, ATTACHED_COLUMNS, strict=True
                )
            }
        )
        interpretation = consumer.interpret_household_receipt(
            consumer.receipt_columns_from_attached_household(
                household,
                source_period_kind="interview_status",
                source_authentication="checkpoint_bytes_verified",
            )
        )
        assert interpretation.view_counts()["observed_receipt"] == 1
        assert interpretation.view_counts()["observed_nonreceipt"] == 1

    def test_missing_attached_columns_refuse(self):
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="ATTACHED_COLUMNS"
        ):
            consumer.receipt_columns_from_attached_household(
                pd.DataFrame({"household_id": [1]}),
                source_period_kind="interview_status",
                source_authentication="checkpoint_bytes_verified",
            )


class TestSpmScope:
    def test_sole_spm_unit_household_assigns_both_answers(self):
        scope = scope_for([RECEIPT_PUBLIC, NONRECEIPT], [1, 1])
        assert [consumer.ASSIGNMENTS[code] for code in scope.assignment] == [
            "assigned_receipt_sole_spm_unit",
            "assigned_nonreceipt_sole_spm_unit",
        ]

    def test_multi_spm_unit_receipt_stays_scope_ambiguous(self):
        scope = scope_for([RECEIPT_LOWER], [2])
        assert set(consumer.ASSIGNMENTS[c] for c in scope.assignment) == {
            "withheld_scope_ambiguous_receipt"
        }
        assert scope.spm_units_in_household.tolist() == [2, 2]

    def test_multi_spm_unit_negative_is_separated_from_receipt_ambiguity(self):
        scope = scope_for([NONRECEIPT], [2])
        assert set(consumer.ASSIGNMENTS[c] for c in scope.assignment) == {
            "withheld_scope_ambiguous_nonreceipt"
        }

    @pytest.mark.parametrize(
        "row,assignment",
        [
            (UNKNOWN_PARENT, "withheld_unknown"),
            (UNKNOWN_LOWER, "withheld_unknown"),
            (CONFLICT, "withheld_conflict"),
            (OWNER, "withheld_not_in_universe"),
        ],
    )
    def test_invalid_households_are_withheld_at_every_spm_count(self, row, assignment):
        for count in (1, 3):
            scope = scope_for([row], [count])
            assert set(consumer.ASSIGNMENTS[c] for c in scope.assignment) == {
                assignment
            }
            assert scope.rows == count

    def test_no_household_is_dropped(self):
        rows = [RECEIPT_PUBLIC, NONRECEIPT, UNKNOWN_PARENT, CONFLICT, OWNER]
        scope = scope_for(rows, [2, 1, 1, 1, 1])
        assert scope.rows == 6
        assert sum(scope.assignment_counts().values()) == 6
        assert len(np.unique(scope.household_row)) == len(rows)

    def test_an_spm_unit_spanning_two_households_is_refused(self):
        interpretation = interpretation_for([NONRECEIPT, NONRECEIPT])
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="SPM_UNIT_SPANS_HOUSEHOLDS"
        ):
            consumer.resolve_spm_receipt_scope(
                interpretation,
                household_id=np.array([1, 2], dtype="int64"),
                spm_unit_id=np.array([1], dtype="int64"),
                person_household_id=np.array([1, 2], dtype="int64"),
                person_spm_unit_id=np.array([1, 1], dtype="int64"),
            )

    def test_unaligned_household_rows_are_refused(self):
        interpretation = interpretation_for([NONRECEIPT])
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="HOUSEHOLD_ALIGNMENT"
        ):
            consumer.resolve_spm_receipt_scope(
                interpretation,
                household_id=np.array([1, 2], dtype="int64"),
                spm_unit_id=np.array([1, 2], dtype="int64"),
                person_household_id=np.array([1, 2], dtype="int64"),
                person_spm_unit_id=np.array([1, 2], dtype="int64"),
            )

    def test_unordered_household_ids_still_map_each_spm_unit(self):
        interpretation = interpretation_for([RECEIPT_PUBLIC, NONRECEIPT])
        scope = consumer.resolve_spm_receipt_scope(
            interpretation,
            household_id=np.array([9, 4], dtype="int64"),
            spm_unit_id=np.array([70, 30], dtype="int64"),
            person_household_id=np.array([4, 9], dtype="int64"),
            person_spm_unit_id=np.array([30, 70], dtype="int64"),
        )
        assert [consumer.ASSIGNMENTS[c] for c in scope.assignment] == [
            "assigned_receipt_sole_spm_unit",
            "assigned_nonreceipt_sole_spm_unit",
        ]


class TestEngineProjection:
    def test_multi_unit_negative_policy_must_be_declared(self):
        scope = scope_for([NONRECEIPT], [2])
        with pytest.raises(TypeError):
            consumer.project_engine_receipt_inputs(scope)
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="MULTI_UNIT_NEGATIVE_POLICY"
        ):
            consumer.project_engine_receipt_inputs(scope, multi_unit_negative="maybe")

    def test_refuse_policy_withholds_the_multi_unit_negative(self):
        scope = scope_for([NONRECEIPT], [2])
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        assert projection.code_counts() == {"false": 0, "true": 0, "withheld": 2}

    def test_distribute_policy_assigns_only_the_negative(self):
        scope = scope_for([NONRECEIPT, RECEIPT_LOWER], [2, 2])
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="distribute"
        )
        assert projection.code_counts() == {"false": 2, "true": 0, "withheld": 2}
        assert projection.receipt["multi_unit_negative"] == "distribute"

    def test_a_withheld_unit_never_becomes_a_blanket_boolean(self):
        scope = scope_for([RECEIPT_PUBLIC, UNKNOWN_PARENT], [1, 1])
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match=r"WITHHELD_SPM_UNITS:1"
        ):
            consumer.engine_receives_housing_assistance(projection)
        assert projection.withheld.tolist() == [False, True]

    def test_a_complete_scope_yields_the_engine_boolean(self):
        scope = scope_for([RECEIPT_PUBLIC, NONRECEIPT], [1, 1])
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        values = consumer.engine_receives_housing_assistance(projection)
        assert values.dtype == np.dtype("bool")
        assert values.tolist() == [True, False]

    def test_take_up_and_annual_participation_are_refused(self):
        scope = scope_for([RECEIPT_PUBLIC], [1])
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        with pytest.raises(
            consumer.HousingReceiptRefusalError,
            match="TAKE_UP_NOT_DERIVABLE_FROM_INTERVIEW_STATUS",
        ):
            consumer.engine_takes_up_housing_assistance_if_eligible(projection)
        with pytest.raises(
            consumer.HousingReceiptRefusalError,
            match="ANNUAL_PARTICIPATION_NOT_OBSERVED_AT_INTERVIEW",
        ):
            consumer.annual_income_year_participation(projection)
        assert projection.receipt["annual_income_year_participation"] == (
            "not_established"
        )

    def test_receipt_records_the_synthetic_provenance_and_counts(self):
        scope = scope_for([RECEIPT_PUBLIC, OWNER, CONFLICT], [1, 1, 1])
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        receipt = projection.receipt
        assert receipt["source_authentication"] == "synthetic_unverified"
        assert receipt["release_eligible"] is False
        assert receipt["source_period_kind"] == "interview_status"
        assert receipt["withheld_rows"] == 2
        assert receipt["assignment_counts"]["withheld_not_in_universe"] == 1
        assert receipt["assignment_counts"]["withheld_conflict"] == 1


class TestCappedValuationCorroboration:
    def test_zero_valuation_does_not_revoke_observed_receipt(self):
        scope = scope_for([RECEIPT_PUBLIC], [1])
        report = consumer.corroborate_with_capped_valuation(
            scope, np.array([0.0], dtype="float64")
        )
        assert report["receipt_with_zero_valuation"] == 1
        assert report["defines_receipt"] is False
        assert scope.receipt_view.tolist() == [
            consumer.RECEIPT_VIEWS.index("observed_receipt")
        ]

    def test_positive_valuation_does_not_create_receipt(self):
        scope = scope_for([NONRECEIPT, UNKNOWN_PARENT], [1, 1])
        report = consumer.corroborate_with_capped_valuation(
            scope, np.array([4_000.0, 9_000.0], dtype="float64")
        )
        assert report["nonreceipt_with_positive_valuation"] == 1
        assert report["table"]["unknown"]["positive_valuation"] == 1
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        assert projection.code_counts()["true"] == 0

    def test_a_negative_valuation_is_refused(self):
        scope = scope_for([NONRECEIPT], [1])
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="VALUATION_NEGATIVE"
        ):
            consumer.corroborate_with_capped_valuation(
                scope, np.array([-1.0], dtype="float64")
            )


class TestQuantitySeparation:
    def test_the_five_quantities_stay_declared_and_distinct(self):
        assert set(consumer.HOUSING_QUANTITIES) == {
            "observed_receipt_status",
            "counterfactual_full_contract_charge",
            "reported_contract_rent",
            "modeled_hap",
            "capped_spm_valuation",
        }
        assert all(
            quantity["additive_to_tenant_rent"] is False
            for quantity in consumer.HOUSING_QUANTITIES.values()
        )
        defining = [
            name
            for name, quantity in consumer.HOUSING_QUANTITIES.items()
            if quantity["defines_receipt"]
        ]
        assert defining == ["observed_receipt_status"]

    @pytest.mark.parametrize(
        "quantities,reason",
        [
            (
                ("reported_contract_rent", "capped_spm_valuation"),
                "CAPPED_VALUATION_NOT_ADDITIVE_TO_RENT",
            ),
            (
                ("reported_contract_rent", "modeled_hap"),
                "MODEL_OUTPUT_CANNOT_DEFINE_ITS_OWN_INPUT",
            ),
            (
                ("observed_receipt_status", "capped_spm_valuation"),
                "CAPPED_VALUATION_CANNOT_DEFINE_RECEIPT",
            ),
            (
                ("reported_contract_rent",),
                "FULL_CONTRACT_CHARGE_SOURCE_UNAVAILABLE",
            ),
        ],
    )
    def test_every_rejected_fusion_refuses_by_name(self, quantities, reason):
        payload = {name: np.array([1.0]) for name in quantities}
        with pytest.raises(consumer.HousingReceiptRefusalError, match=reason):
            consumer.compose_full_contract_charge(**payload)

    def test_the_counterfactual_charge_is_reported_unavailable(self):
        availability = consumer.counterfactual_charge_availability()
        assert availability["available"] is False
        assert availability["refused_shortcut"].startswith("FMR_WITHOUT_YEAR_CORRECT")
        assert len(availability["blocking_defects"]) == 4


class TestDiagnostics:
    def test_counts_every_raw_status_and_unknown_reason(self):
        rows = [
            RECEIPT_PUBLIC,
            RECEIPT_LOWER,
            NONRECEIPT,
            UNKNOWN_PARENT,
            UNKNOWN_LOWER,
            CONFLICT,
            OWNER,
        ]
        scope = scope_for(rows, [1, 2, 1, 1, 1, 1, 1])
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        report = consumer.housing_receipt_diagnostics(
            scope.interpretation, scope, projection
        )
        assert report["household_view_counts"] == {
            "unknown": 2,
            "observed_receipt": 2,
            "observed_nonreceipt": 1,
            "not_in_universe": 1,
            "conflict": 1,
        }
        assert report["unknown_reason_counts"]["parent_answer"] == 1
        assert report["unknown_reason_counts"]["lower_answer"] == 1
        assert report["conflict_counts"]["parent_route"] == 1
        assert report["spm_unit_rows"] == 8
        assert report["multi_spm_unit_households"] == 1
        assert report["engine_boolean_complete"] is False
        assert report["engine_input_code_counts"]["withheld"] == 6
        assert report["zero_origin_counts"]["HLORENT_zero_origin"] == 4

    def test_the_diagnostics_digest_moves_with_the_counts(self):
        first = consumer.housing_receipt_diagnostics(interpretation_for([NONRECEIPT]))
        second = consumer.housing_receipt_diagnostics(
            interpretation_for([RECEIPT_PUBLIC])
        )
        assert first["diagnostics_sha256"] != second["diagnostics_sha256"]


class TestIntegrationSeam:
    def test_the_seam_names_the_real_modules_and_refused_exports(self):
        seam = consumer.describe_integration_seam()
        assert seam["reads"]["attached_columns"] == list(ATTACHED_COLUMNS)
        assert set(seam["refuses_to_export"]) == {
            "rent",
            "housing_assistance",
            "hud_hap",
            "spm_unit_capped_housing_subsidy",
        }
        assert seam["produces"]["takes_up_housing_assistance_if_eligible"] == (
            "not derived"
        )
        assert seam["unresolved_before_launch"]["available"] is False
