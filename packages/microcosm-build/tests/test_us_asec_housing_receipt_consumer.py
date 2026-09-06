"""Household-to-SPM receipt interpretation over invented ASEC housing evidence.

Every fixture is explicitly invented. The synthetic cases stay
``synthetic_unverified``; the one authenticated case reuses the existing
invented-byte source fixture, which substitutes pins inside invented files and
authenticates no Census artifact. The production classifiers derive the carried
status and occupied-HU columns the consumer reads, so these tests exercise the
shipped kernels rather than a restatement of them.

The consumer never reads a raw source column: it reads a validated status
artifact or a validated status attachment, and it reads occupied-HU evidence
only through the ``asec_housing_universe`` artifact or attachment.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import asec_housing_receipt_consumer as consumer
from microcosm.build.us_runtime import asec_housing_status as housing
from microcosm.build.us_runtime import asec_housing_universe as universe_module
from microcosm.build.us_runtime.asec_housing_status_source import ATTACHED_COLUMNS

REPOSITORY_ROOT = Path(__file__).parents[3]

RECEIPT_PUBLIC = {"HPUBLIC": 1, "HLORENT": 0}
RECEIPT_LOWER = {"HPUBLIC": 2, "HLORENT": 1}
NONRECEIPT = {"HPUBLIC": 2, "HLORENT": 2}
UNKNOWN_PARENT = {"HPUBLIC": 0, "HLORENT": 0}
UNKNOWN_LOWER = {"HPUBLIC": 2, "HLORENT": 0}
CONFLICT = {"HPUBLIC": 1, "HLORENT": 2}
OWNER = {"H_TENURE": 1, "HPUBLIC": 0, "HLORENT": 0}

#: Group-quarters receipt households. The frozen source classifies both HRHTYPE
#: 9 and 10 as receipt and declares its non-housing-unit scope "not_derived".
GQ_RECEIPT_9 = {**RECEIPT_PUBLIC, "HRHTYPE": 9}
GQ_RECEIPT_10 = {**RECEIPT_PUBLIC, "HRHTYPE": 10}

#: Occupied-HU overrides, applied to the universe row only.
IN_HOUSING_UNIT = {"H_LIVQRT": 1}
OTHER_QUARTERS = {"H_LIVQRT": 8}
QUARTERS_UNKNOWN = {"H_LIVQRT": 0}
#: A universe row whose household kind disagrees with the status's own
#: group-quarters column: the same household cannot be both.
KIND_DISAGREES = {"H_LIVQRT": 8, "HRHTYPE": 9}

#: Universe fields the two artifacts share and must carry identically.
SHARED_CONTEXT = ("H_TENURE", "HRHTYPE", "H_HHTYPE")


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


def universe_observations(rows, housing_unit_rows=None):
    """The same invented households seen through the occupied-HU classifier.

    Context the two artifacts share is copied from the status row, so a
    disagreement only appears where a case deliberately overrides it.
    """
    base = {"income_year": 2023, "H_HHTYPE": 1, "H_LIVQRT": 1, "HRHTYPE": 1}
    records = []
    for index, row in enumerate(rows, start=1):
        shared = {name: row[name] for name in SHARED_CONTEXT if name in row}
        record = {
            "household_id": index,
            "H_SEQ": index,
            "H_TENURE": 2,
            **base,
            **shared,
        }
        if housing_unit_rows is not None:
            record = {**record, **housing_unit_rows[index - 1]}
        records.append(record)
    return pd.DataFrame(records, dtype=np.int64)


def status_for(rows):
    return housing.classify_synthetic_housing_status(observations(rows))


def housing_unit_for(rows, housing_unit_rows=None):
    return consumer.housing_unit_evidence_from_universe(
        universe_module.classify_synthetic_housing_universe(
            universe_observations(rows, housing_unit_rows)
        )
    )


def interpretation_for(rows):
    return consumer.interpret_household_receipt(
        consumer.receipt_columns_from_status(status_for(rows))
    )


def memberships(spm_units_per_household):
    """One person per SPM unit, SPM units distributed over the invented households."""
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
    return {
        "household_id": np.array(household_id, dtype="int64"),
        "spm_unit_id": np.array(spm_unit_id, dtype="int64"),
        "person_household_id": np.array(person_household_id, dtype="int64"),
        "person_spm_unit_id": np.array(person_spm_unit_id, dtype="int64"),
    }


def scope_for(rows, spm_units_per_household, housing_unit_rows=None):
    return consumer.resolve_spm_receipt_scope(
        interpretation_for(rows),
        housing_unit=housing_unit_for(rows, housing_unit_rows),
        **memberships(spm_units_per_household),
    )


def assignments_of(scope):
    return [consumer.ASSIGNMENTS[code] for code in scope.assignment]


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
        assert (
            interpretation.receipt_view.tolist()
            == status_for([row]).array("status").tolist()
        )

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

    def test_every_carried_derived_column_is_kept(self):
        interpretation = interpretation_for([GQ_RECEIPT_9])
        status = status_for([GQ_RECEIPT_9])
        for name in ("route", "group_quarters", "HPUBLIC_valid", "HLORENT_valid"):
            assert interpretation.column(name).tolist() == status.array(name).tolist()
        assert set(interpretation.columns) == set(housing.DERIVED_COLUMNS)
        assert int(interpretation.group_quarters[0]) == 1
        assert interpretation.non_housing_unit_scope == "not_derived"

    def test_the_household_axis_and_cohort_are_carried_from_the_artifact(self):
        interpretation = interpretation_for([RECEIPT_PUBLIC, NONRECEIPT])
        status = status_for([RECEIPT_PUBLIC, NONRECEIPT])
        assert (
            interpretation.household_id.tolist()
            == status.array("household_id").tolist()
        )
        assert (
            interpretation.income_year.tolist() == status.array("income_year").tolist()
        )

    def test_every_stored_array_is_read_only(self):
        interpretation = interpretation_for([RECEIPT_PUBLIC, NONRECEIPT])
        arrays = [
            interpretation.household_id,
            interpretation.income_year,
            interpretation.receipt_view,
            interpretation.receipt_valid,
            interpretation.route,
            interpretation.group_quarters,
            *interpretation.zero_origin.values(),
        ]
        assert all(not array.flags.writeable for array in arrays)
        with pytest.raises(ValueError):
            interpretation.receipt_view[0] = 0

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

    def test_a_bit_the_carried_codebook_never_names_is_refused(self):
        projection = consumer.receipt_columns_from_status(status_for([CONFLICT]))
        conflicts = projection["columns"]["conflicts"].copy()
        conflicts[0] = 32
        projection["columns"]["conflicts"] = conflicts
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="CONFLICT_BIT_DOMAIN"
        ):
            consumer.interpret_household_receipt(projection)

    def test_a_renumbered_status_codebook_is_refused(self):
        projection = consumer.receipt_columns_from_status(status_for([NONRECEIPT]))
        codebook = dict(projection["codebook"])
        codebook["status"] = {"receipt": 0, "nonreceipt": 1}
        projection["codebook"] = codebook
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="CODEBOOK_STATUS"
        ):
            consumer.interpret_household_receipt(projection)

    def test_a_permuted_household_axis_inside_the_projection_is_refused(self):
        projection = consumer.receipt_columns_from_status(
            status_for([RECEIPT_PUBLIC, NONRECEIPT])
        )
        projection["household_id"] = np.array([1, 1], dtype="int64")
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="DUPLICATE_HOUSEHOLD_ID"
        ):
            consumer.interpret_household_receipt(projection)

    def test_synthetic_authentication_is_carried_through(self):
        interpretation = interpretation_for([NONRECEIPT])
        assert interpretation.source_authentication == "synthetic_unverified"
        assert (
            interpretation.status_content_sha256
            == status_for([NONRECEIPT]).content_sha256
        )


class TestAttachedStatusReader:
    def test_a_plain_dataframe_is_refused(self):
        status = status_for([RECEIPT_PUBLIC, NONRECEIPT])
        household = pd.DataFrame(
            {
                attached: status.array(name)
                for name, attached in zip(
                    housing.DERIVED_COLUMNS, ATTACHED_COLUMNS, strict=True
                )
            }
        )
        with pytest.raises(consumer.HousingReceiptRefusalError, match="ATTACHED_ASEC"):
            consumer.receipt_columns_from_attached_household(household)

    def test_no_caller_supplied_authentication_label_is_accepted(self):
        # The forged-label parameters are gone; the reader derives the label
        # from the attachment it validates.
        with pytest.raises(TypeError):
            consumer.receipt_columns_from_attached_household(
                pd.DataFrame({"household_id": [1]}),
                source_period_kind="interview_status",
                source_authentication="checkpoint_bytes_verified",
            )

    def test_a_synthetic_status_never_reports_a_verified_label(self):
        interpretation = interpretation_for([RECEIPT_PUBLIC])
        assert interpretation.source_authentication == "synthetic_unverified"
        scope = scope_for([RECEIPT_PUBLIC], [1])
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        assert projection.receipt["source_authentication"] == "synthetic_unverified"
        assert projection.receipt["housing_unit_authentication"] == (
            "synthetic_unverified"
        )


class TestAuthenticatedAttachment:
    """One invented-byte authenticated pass through both real attachments."""

    def test_the_reader_derives_its_provenance_from_the_validated_attachment(
        self, tmp_path, monkeypatch
    ):
        specification = importlib.util.spec_from_file_location(
            "housing_receipt_status_source_fixture",
            Path(__file__).with_name("test_us_asec_housing_status_source.py"),
        )
        fixture = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(fixture)
        from microcosm.build.us_runtime import asec_current_money_units as units
        from microcosm.build.us_runtime import (
            asec_housing_status_source,
            asec_housing_universe_source,
        )

        loaded, cohort_paths, *_ = fixture.invented_source(tmp_path, monkeypatch)
        status = asec_housing_status_source.load_authenticated_housing_status(
            loaded, cohort_paths=cohort_paths
        )
        tax_result = units.reconstruct_current_money_tax_units(loaded, loaded.ready())
        attached = asec_housing_status_source.attach_housing_status(tax_result, status)
        universe = asec_housing_universe_source.load_authenticated_housing_universe(
            loaded
        )
        attached_universe = asec_housing_universe_source.attach_housing_universe(
            attached, universe
        )

        interpretation = consumer.interpret_household_receipt(
            consumer.receipt_columns_from_attached_household(attached)
        )
        assert interpretation.source_authentication == "checkpoint_bytes_verified"
        assert interpretation.status_content_sha256 == status.content_sha256
        assert (
            interpretation.household_id.tolist()
            == status.array("household_id").tolist()
        )
        assert interpretation.income_year.tolist() == [2022, 2023, 2024]
        # The attached bytes are the status bytes; nothing is coerced.
        for name in housing.DERIVED_COLUMNS:
            assert interpretation.column(name).tobytes() == status.array(name).tobytes()

        evidence = consumer.housing_unit_evidence_from_attached_universe(
            attached_universe
        )
        assert evidence.source_authentication == "checkpoint_bytes_verified"
        assert evidence.universe_content_sha256 == universe.content_sha256
        assert evidence.occupied_hu.tolist() == [1, 1, 1]

        person = attached_universe.frame.table("person")
        scope = consumer.resolve_spm_receipt_scope(
            interpretation,
            housing_unit=evidence,
            household_id=attached_universe.frame.table("household")[
                "household_id"
            ].to_numpy(),
            spm_unit_id=attached_universe.frame.table("spm_unit")[
                "spm_unit_id"
            ].to_numpy(),
            person_household_id=person["person_household_id"].to_numpy(),
            person_spm_unit_id=person["person_spm_unit_id"].to_numpy(),
        )
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        receipt = projection.receipt
        assert receipt["source_authentication"] == "checkpoint_bytes_verified"
        assert receipt["housing_unit_authentication"] == "checkpoint_bytes_verified"
        assert receipt["status_content_sha256"] == status.content_sha256
        assert receipt["universe_content_sha256"] == universe.content_sha256
        assert receipt["release_eligible"] is False

    def test_a_synthetic_universe_cannot_certify_an_authenticated_status(
        self, tmp_path, monkeypatch
    ):
        specification = importlib.util.spec_from_file_location(
            "housing_receipt_status_source_mismatch_fixture",
            Path(__file__).with_name("test_us_asec_housing_status_source.py"),
        )
        fixture = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(fixture)
        from microcosm.build.us_runtime import asec_housing_status_source

        loaded, cohort_paths, *_ = fixture.invented_source(tmp_path, monkeypatch)
        status = asec_housing_status_source.load_authenticated_housing_status(
            loaded, cohort_paths=cohort_paths
        )
        interpretation = consumer.interpret_household_receipt(
            consumer.receipt_columns_from_status(status)
        )
        synthetic = consumer.housing_unit_evidence_from_universe(
            universe_module.classify_synthetic_housing_universe(
                pd.DataFrame(
                    {
                        "household_id": [1, 2, 3],
                        "income_year": [2022, 2023, 2024],
                        "H_SEQ": [7, 7, 7],
                        "H_HHTYPE": [1, 1, 1],
                        "H_LIVQRT": [1, 1, 1],
                        "HRHTYPE": [1, 1, 1],
                        "H_TENURE": [2, 2, 2],
                    },
                    dtype=np.int64,
                )
            )
        )
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="HOUSING_UNIT_AUTHENTICATION"
        ):
            consumer.resolve_spm_receipt_scope(
                interpretation,
                housing_unit=synthetic,
                **memberships([1, 1, 1]),
            )


class TestSpmScope:
    def test_sole_spm_unit_household_assigns_both_answers(self):
        scope = scope_for([RECEIPT_PUBLIC, NONRECEIPT], [1, 1])
        assert assignments_of(scope) == [
            "assigned_receipt_sole_spm_unit",
            "assigned_nonreceipt_sole_spm_unit",
        ]

    def test_multi_spm_unit_receipt_stays_scope_ambiguous(self):
        scope = scope_for([RECEIPT_LOWER], [2])
        assert set(assignments_of(scope)) == {"withheld_scope_ambiguous_receipt"}
        assert scope.spm_units_in_household.tolist() == [2, 2]

    def test_multi_spm_unit_negative_is_separated_from_receipt_ambiguity(self):
        scope = scope_for([NONRECEIPT], [2])
        assert set(assignments_of(scope)) == {"withheld_scope_ambiguous_nonreceipt"}

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
            assert set(assignments_of(scope)) == {assignment}
            assert scope.rows == count

    def test_no_household_is_dropped(self):
        rows = [RECEIPT_PUBLIC, NONRECEIPT, UNKNOWN_PARENT, CONFLICT, OWNER]
        scope = scope_for(rows, [2, 1, 1, 1, 1])
        assert scope.rows == 6
        assert sum(scope.assignment_counts().values()) == 6
        assert len(np.unique(scope.household_row)) == len(rows)

    def test_an_spm_unit_spanning_two_households_is_refused(self):
        rows = [NONRECEIPT, NONRECEIPT]
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="SPM_UNIT_SPANS_HOUSEHOLDS"
        ):
            consumer.resolve_spm_receipt_scope(
                interpretation_for(rows),
                housing_unit=housing_unit_for(rows),
                household_id=np.array([1, 2], dtype="int64"),
                spm_unit_id=np.array([1], dtype="int64"),
                person_household_id=np.array([1, 2], dtype="int64"),
                person_spm_unit_id=np.array([1, 1], dtype="int64"),
            )

    def test_unaligned_household_rows_are_refused(self):
        rows = [NONRECEIPT]
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="HOUSEHOLD_ALIGNMENT"
        ):
            consumer.resolve_spm_receipt_scope(
                interpretation_for(rows),
                housing_unit=housing_unit_for(rows),
                household_id=np.array([1, 2], dtype="int64"),
                spm_unit_id=np.array([1, 2], dtype="int64"),
                person_household_id=np.array([1, 2], dtype="int64"),
                person_spm_unit_id=np.array([1, 2], dtype="int64"),
            )

    def test_a_permuted_caller_household_axis_is_refused(self):
        # Positional alignment silently swapped the two answers before the
        # interpretation carried its own household identity.
        rows = [RECEIPT_PUBLIC, NONRECEIPT]
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="CALLER_HOUSEHOLD_IDENTITY"
        ):
            consumer.resolve_spm_receipt_scope(
                interpretation_for(rows),
                housing_unit=housing_unit_for(rows),
                household_id=np.array([2, 1], dtype="int64"),
                spm_unit_id=np.array([70, 30], dtype="int64"),
                person_household_id=np.array([1, 2], dtype="int64"),
                person_spm_unit_id=np.array([30, 70], dtype="int64"),
            )

    def test_household_ids_the_artifact_never_carried_are_refused(self):
        rows = [RECEIPT_PUBLIC, NONRECEIPT]
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="CALLER_HOUSEHOLD_IDENTITY"
        ):
            consumer.resolve_spm_receipt_scope(
                interpretation_for(rows),
                housing_unit=housing_unit_for(rows),
                household_id=np.array([9, 4], dtype="int64"),
                spm_unit_id=np.array([70, 30], dtype="int64"),
                person_household_id=np.array([4, 9], dtype="int64"),
                person_spm_unit_id=np.array([30, 70], dtype="int64"),
            )

    def test_permuting_only_persons_and_spm_units_keeps_every_assignment(self):
        rows = [RECEIPT_PUBLIC, NONRECEIPT]
        ordered = scope_for(rows, [1, 1])
        permuted = consumer.resolve_spm_receipt_scope(
            interpretation_for(rows),
            housing_unit=housing_unit_for(rows),
            household_id=np.array([1, 2], dtype="int64"),
            spm_unit_id=np.array([30, 7], dtype="int64"),
            person_household_id=np.array([2, 1], dtype="int64"),
            person_spm_unit_id=np.array([7, 30], dtype="int64"),
        )
        assert assignments_of(ordered) == [
            "assigned_receipt_sole_spm_unit",
            "assigned_nonreceipt_sole_spm_unit",
        ]
        by_spm_unit = dict(
            zip(permuted.spm_unit_id.tolist(), assignments_of(permuted), strict=True)
        )
        assert by_spm_unit == {
            30: "assigned_receipt_sole_spm_unit",
            7: "assigned_nonreceipt_sole_spm_unit",
        }


class TestHousingUnitEvidence:
    def test_occupied_hu_evidence_has_no_default(self):
        rows = [RECEIPT_PUBLIC]
        with pytest.raises(TypeError):
            consumer.resolve_spm_receipt_scope(
                interpretation_for(rows), **memberships([1])
            )

    @pytest.mark.parametrize("row", [GQ_RECEIPT_9, GQ_RECEIPT_10])
    def test_a_group_quarters_receipt_is_observed_but_never_assigned(self, row):
        scope = scope_for([row], [1], [OTHER_QUARTERS])
        assert assignments_of(scope) == ["withheld_outside_housing_unit"]
        # The raw observation is untouched: only the assignment is withheld.
        assert scope.receipt_view.tolist() == [
            consumer.RECEIPT_VIEWS.index("observed_receipt")
        ]
        assert scope.interpretation.view_counts()["observed_receipt"] == 1
        assert scope.occupied_hu.tolist() == [
            consumer.OCCUPIED_HU_STATES.index("outside_housing_unit")
        ]
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        assert projection.code_counts() == {"false": 0, "true": 0, "withheld": 1}

    def test_an_unestablished_housing_unit_is_withheld_not_defaulted(self):
        scope = scope_for(
            [RECEIPT_PUBLIC, NONRECEIPT], [1, 1], [QUARTERS_UNKNOWN, QUARTERS_UNKNOWN]
        )
        assert set(assignments_of(scope)) == {"withheld_housing_unit_not_established"}
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="distribute"
        )
        assert projection.code_counts() == {"false": 0, "true": 0, "withheld": 2}

    def test_a_household_kind_disagreement_is_withheld_as_a_conflict(self):
        # The status says primary household; the universe says group quarters.
        scope = scope_for([RECEIPT_PUBLIC], [1], [KIND_DISAGREES])
        assert assignments_of(scope) == ["withheld_housing_unit_conflict"]

    def test_a_bound_occupied_housing_unit_assigns_normally(self):
        scope = scope_for(
            [RECEIPT_PUBLIC, NONRECEIPT], [1, 1], [IN_HOUSING_UNIT, IN_HOUSING_UNIT]
        )
        assert assignments_of(scope) == [
            "assigned_receipt_sole_spm_unit",
            "assigned_nonreceipt_sole_spm_unit",
        ]
        assert scope.housing_unit_valid.tolist() == [True, True]

    def test_a_universe_on_other_households_is_refused(self):
        rows = [RECEIPT_PUBLIC, NONRECEIPT]
        other = universe_observations(rows)
        other["household_id"] = np.array([11, 12], dtype="int64")
        evidence = consumer.housing_unit_evidence_from_universe(
            universe_module.classify_synthetic_housing_universe(other)
        )
        with pytest.raises(
            consumer.HousingReceiptRefusalError,
            match="HOUSING_UNIT_HOUSEHOLD_IDENTITY",
        ):
            consumer.resolve_spm_receipt_scope(
                interpretation_for(rows),
                housing_unit=evidence,
                **memberships([1, 1]),
            )

    def test_a_universe_from_another_cohort_is_refused(self):
        rows = [RECEIPT_PUBLIC, NONRECEIPT]
        other = universe_observations(rows)
        other["income_year"] = np.array([2024, 2024], dtype="int64")
        evidence = consumer.housing_unit_evidence_from_universe(
            universe_module.classify_synthetic_housing_universe(other)
        )
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="HOUSING_UNIT_COHORT"
        ):
            consumer.resolve_spm_receipt_scope(
                interpretation_for(rows),
                housing_unit=evidence,
                **memberships([1, 1]),
            )

    def test_a_universe_of_the_wrong_length_is_refused(self):
        rows = [RECEIPT_PUBLIC, NONRECEIPT]
        evidence = housing_unit_for([RECEIPT_PUBLIC])
        with pytest.raises(
            consumer.HousingReceiptRefusalError, match="HOUSING_UNIT_ALIGNMENT"
        ):
            consumer.resolve_spm_receipt_scope(
                interpretation_for(rows),
                housing_unit=evidence,
                **memberships([1, 1]),
            )

    def test_the_evidence_arrays_are_read_only(self):
        evidence = housing_unit_for([RECEIPT_PUBLIC])
        assert not evidence.occupied_hu.flags.writeable
        assert not evidence.household_kind.flags.writeable
        assert not evidence.household_id.flags.writeable


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

    def test_distribute_leaves_every_other_withheld_class_unchanged(self):
        rows = [
            NONRECEIPT,
            RECEIPT_LOWER,
            UNKNOWN_PARENT,
            CONFLICT,
            OWNER,
            NONRECEIPT,
            GQ_RECEIPT_9,
        ]
        housing_unit_rows = [
            IN_HOUSING_UNIT,
            IN_HOUSING_UNIT,
            IN_HOUSING_UNIT,
            IN_HOUSING_UNIT,
            IN_HOUSING_UNIT,
            QUARTERS_UNKNOWN,
            OTHER_QUARTERS,
        ]
        scope = scope_for(rows, [2, 2, 1, 1, 1, 2, 1], housing_unit_rows)
        refused = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        distributed = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="distribute"
        )
        expected = {
            "assigned_receipt_sole_spm_unit": 0,
            "assigned_nonreceipt_sole_spm_unit": 0,
            "assigned_nonreceipt_household_negative": 0,
            "withheld_scope_ambiguous_receipt": 2,
            "withheld_scope_ambiguous_nonreceipt": 2,
            "withheld_unknown": 1,
            "withheld_conflict": 1,
            "withheld_not_in_universe": 1,
            "withheld_outside_housing_unit": 1,
            "withheld_housing_unit_not_established": 2,
            "withheld_housing_unit_conflict": 0,
        }
        assert refused.receipt["assignment_counts"] == expected
        moved = {
            **expected,
            "withheld_scope_ambiguous_nonreceipt": 0,
            "assigned_nonreceipt_household_negative": 2,
        }
        assert distributed.receipt["assignment_counts"] == moved
        assert distributed.code_counts() == {"false": 2, "true": 0, "withheld": 8}

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
        assert receipt["housing_unit_authentication"] == "synthetic_unverified"
        assert receipt["release_eligible"] is False
        assert receipt["source_period_kind"] == "interview_status"
        assert receipt["non_housing_unit_scope"] == "not_derived"
        assert receipt["withheld_rows"] == 2
        assert receipt["assignment_counts"]["withheld_not_in_universe"] == 1
        assert receipt["assignment_counts"]["withheld_conflict"] == 1

    def test_the_projection_arrays_are_read_only(self):
        scope = scope_for([RECEIPT_PUBLIC], [1])
        projection = consumer.project_engine_receipt_inputs(
            scope, multi_unit_negative="refuse"
        )
        assert not projection.code.flags.writeable
        assert not projection.assignment.flags.writeable
        assert not scope.assignment.flags.writeable


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

    def test_the_requirements_reference_is_a_portable_in_repository_document(self):
        availability = consumer.counterfactual_charge_availability()
        reference = availability["requirements"]
        assert reference == consumer.CHARGE_REQUIREMENTS_DOCUMENT
        assert not reference.startswith("/") and ".." not in reference
        assert (REPOSITORY_ROOT / reference).is_file()
        assert availability["requirements_kind"] == "required_future_measurement"
        assert availability["measured_here"] is False


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

    def test_the_routing_and_scope_context_the_docs_promise_is_reported(self):
        rows = [RECEIPT_PUBLIC, RECEIPT_LOWER, GQ_RECEIPT_10, NONRECEIPT]
        scope = scope_for(
            rows,
            [1, 1, 1, 1],
            [IN_HOUSING_UNIT, IN_HOUSING_UNIT, OTHER_QUARTERS, IN_HOUSING_UNIT],
        )
        report = consumer.housing_receipt_diagnostics(scope.interpretation, scope)
        assert report["route_counts"]["public_housing"] == 2
        assert report["route_counts"]["lower_rent"] == 1
        assert report["route_counts"]["both_negative"] == 1
        assert report["group_quarters_counts"]["gq"] == 1
        assert report["group_quarters_counts"]["not_gq"] == 3
        assert report["observed_answer_valid_counts"]["HPUBLIC_valid"] == 4
        assert report["non_housing_unit_scope"] == "not_derived"
        assert report["occupied_hu_counts"]["outside_housing_unit"] == 1
        assert report["occupied_hu_counts"]["occupied_housing_unit"] == 3
        assert report["household_kind_counts"]["group_quarters"] == 1

    def test_the_bit_names_come_from_the_carried_codebook(self):
        projection = consumer.receipt_columns_from_status(status_for([CONFLICT]))
        codebook = {
            name: (dict(value) if type(value) is dict else value)
            for name, value in projection["codebook"].items()
        }
        codebook["conflict_bits"] = {
            "renamed_excluded_scope": 1,
            "renamed_parent_route": 2,
            "renamed_public_allocation": 4,
            "renamed_lower_allocation": 8,
            "renamed_context": 16,
        }
        projection["codebook"] = codebook
        report = consumer.housing_receipt_diagnostics(
            consumer.interpret_household_receipt(projection)
        )
        assert report["conflict_counts"]["renamed_parent_route"] == 1
        assert "parent_route" not in report["conflict_counts"]

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

    def test_the_seam_requires_the_occupied_hu_attachment_it_actually_reads(self):
        from microcosm.build.us_runtime.asec_housing_universe_source import (
            ATTACHED_COLUMNS as UNIVERSE_ATTACHED_COLUMNS,
        )

        seam = consumer.describe_integration_seam()
        assert seam["reads"]["housing_unit_attachment"] == (
            "microcosm.asec_housing_universe_attachment.v1"
        )
        assert seam["reads"]["housing_unit_attached_columns"] == list(
            UNIVERSE_ATTACHED_COLUMNS
        )
        assert seam["attaches_after"].endswith("attach_housing_universe")
