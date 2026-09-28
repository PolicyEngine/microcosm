"""Invented-roster tests for the explicit Build P SPM regrouping policy."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.acs_spm_regroup import (
    AcsSpmLegacyDefaults,
    regroup_acs_spm_units,
)


def defaults():
    return AcsSpmLegacyDefaults(
        acs_spine="acs_2024_1yr",
        parent_sha256="a" * 64,
        null_register_sha256="b" * 64,
        values={
            "receives_housing_assistance": False,
            "spm_unit_energy_subsidy": 0.0,
            "takes_up_tanf_if_eligible": True,
            "takes_up_snap_if_eligible": True,
        },
    )


def fixture(*, ages=(40, 8, 30), amount=1200.0, tenure=3):
    persons = pd.DataFrame(
        {
            "person_id": range(1, len(ages) + 1),
            "person_household_id": 10,
            "person_spm_unit_id": 100,
            "person_tax_unit_id": [11] * (len(ages) - 1) + [12],
            "AGEP": ages,
            "RELSHIPP": [20] + [25] * (len(ages) - 2) + [36],
            "employment_income_before_lsr": 100.0,
        }
    )
    units = pd.DataFrame(
        {
            "spm_unit_id": [100],
            "spm_unit_pre_subsidy_childcare_expenses": [amount],
            "receives_housing_assistance": [False],
            "takes_up_housing_assistance_if_eligible": [True],
            "spm_unit_tenure_type": [
                {1: "OWNER_WITH_MORTGAGE", 2: "OWNER_WITHOUT_MORTGAGE"}.get(
                    tenure, "RENTER"
                )
            ],
            "spm_unit_energy_subsidy": [0.0],
            "spm_unit_source_id": [70],
            "spm_unit_support_channel": ["acs_2024_1yr"],
            "spm_unit_support_clone_index": [0],
            "takes_up_tanf_if_eligible": [True],
            "takes_up_snap_if_eligible": [True],
            "spm_unit_spine": ["acs_2024_1yr"],
        }
    )
    households = pd.DataFrame(
        {"household_id": [10], "TYPEHUGQ": [1], "NP": [len(ages)], "TEN": [tenure]}
    )
    membership = pd.DataFrame(
        {
            "person_id": persons.person_id,
            "new_spm_unit_id": [100] * (len(ages) - 1) + [101],
            "new_spm_unit_source_id": [80] * (len(ages) - 1) + [81],
        }
    )
    return persons, units, households, membership


def run(inputs, **kwargs):
    return regroup_acs_spm_units(
        *inputs,
        legacy_defaults=defaults(),
        tenure_policy="acs_ten4_no_mortgage_v1",
        **kwargs,
    )


def test_conserves_positive_childcare_once_and_never_mutates_inputs():
    inputs = fixture()
    before = [table.copy(deep=True) for table in inputs]
    result = run(inputs)
    result.require_complete()
    assert result.spm_units.spm_unit_pre_subsidy_childcare_expenses.tolist() == [
        1200.0,
        0.0,
    ]
    assert result.childcare_ledger.old_amount.tolist() == [1200.0]
    assert result.childcare_ledger.allocated_amount.tolist() == [1200.0]
    assert result.childcare_ledger.unresolved_amount.tolist() == [0.0]
    assert result.childcare_ledger.status.tolist() == [
        "conserved_unique_reference_successor"
    ]
    assert result.crosswalk.old_spm_unit_source_id.tolist() == [70, 70]
    assert result.crosswalk.new_spm_unit_source_id.tolist() == [80, 81]
    assert result.spm_units.spm_unit_source_id.tolist() == [80, 81]
    assert result.spm_units.takes_up_housing_assistance_if_eligible.tolist() == [
        True,
        True,
    ]
    assert result.metadata["housing_policy"] == "acs_spm_housing_common_imputation_v1"
    assert result.metadata["legacy_defaults"]["source_observed"] is False
    for after, original in zip(inputs, before, strict=True):
        pd.testing.assert_frame_equal(after, original)


def test_imputed_zero_fans_out_only_as_labeled_zero():
    result = run(fixture(ages=(40, 30), amount=0))
    result.require_complete()
    assert result.spm_units.spm_unit_pre_subsidy_childcare_expenses.eq(0).all()
    assert result.childcare_ledger.status.tolist() == ["inherited_imputed_zero"]


@pytest.mark.parametrize("age", [14, 15])
def test_age_boundary_is_potential_care_not_tax_credit_eligibility(age):
    result = run(fixture(ages=(40, age, 30)))
    result.require_complete()
    assert result.spm_units.spm_unit_pre_subsidy_childcare_expenses.tolist() == [
        1200.0,
        0.0,
    ]


@pytest.mark.parametrize("ages", [(40, 8, 15), (40, 30)])
def test_ambiguous_or_absent_childcare_recipient_is_queued(ages):
    result = run(fixture(ages=ages))
    assert result.spm_units.spm_unit_pre_subsidy_childcare_expenses.isna().all()
    assert result.childcare_ledger.unresolved_amount.tolist() == [1200.0]
    assert len(result.exceptions) == 1
    with pytest.raises(ValueError, match="unresolved"):
        result.require_complete()


def test_single_secondary_child_unit_is_not_silently_selected():
    result = run(fixture(ages=(40, 30, 15)))
    assert result.exceptions.reason.tolist() == ["sole_recipient_is_not_reference"]


@pytest.mark.parametrize("care", [True, pd.NA])
def test_older_minor_care_or_unknown_evidence_prevents_false_allocation(care):
    inputs = fixture(ages=(40, 8, 17))
    evidence = pd.Series([False, False, care], index=[1, 2, 3], dtype="boolean")
    result = run(inputs, older_care_evidence=evidence)
    assert result.spm_units.spm_unit_pre_subsidy_childcare_expenses.isna().all()
    assert len(result.exceptions) == 1


def test_older_minor_explicit_negative_care_evidence_can_resolve():
    inputs = fixture(ages=(40, 8, 17))
    evidence = pd.Series(False, index=[1, 2, 3], dtype="boolean")
    result = run(inputs, older_care_evidence=evidence)
    result.require_complete()
    assert result.spm_units.spm_unit_pre_subsidy_childcare_expenses.tolist() == [
        1200,
        0,
    ]


def test_missing_age_keeps_unknown_amount_in_exception_ledger():
    result = run(fixture(ages=(40, 8, np.nan)))
    assert result.exceptions.reason.tolist() == ["unknown_age_or_care_evidence"]
    assert result.childcare_ledger.unresolved_amount.tolist() == [1200.0]


def test_missing_old_amount_is_not_zero():
    result = run(fixture(amount=np.nan))
    assert result.childcare_ledger.old_amount.isna().all()
    assert result.childcare_ledger.unresolved_amount.isna().all()
    assert result.exceptions.reason.tolist() == ["missing_childcare_amount"]


def test_tenure4_recode_has_separate_preserve_sensitivity():
    inputs = fixture(tenure=4)
    corrected = run(inputs)
    baseline = regroup_acs_spm_units(
        *inputs,
        legacy_defaults=defaults(),
        tenure_policy="preserve_parent_tenure_v1",
    )
    assert corrected.spm_units.spm_unit_tenure_type.eq("OWNER_WITHOUT_MORTGAGE").all()
    assert baseline.spm_units.spm_unit_tenure_type.eq("RENTER").all()
    pd.testing.assert_frame_equal(
        corrected.crosswalk.drop(columns="tenure_recode"),
        baseline.crosswalk.drop(columns="tenure_recode"),
    )
    assert corrected.crosswalk.tenure_recode.all()
    assert not baseline.crosswalk.tenure_recode.any()
    assert corrected.metadata["tenure_recode_units"] == 2
    assert inputs[2].TEN.tolist() == [4]


def test_unchanged_units_preserve_values_and_source_identity():
    inputs = list(fixture())
    inputs[3]["new_spm_unit_id"] = 100
    inputs[3]["new_spm_unit_source_id"] = 70
    result = run(inputs)
    pd.testing.assert_frame_equal(result.spm_units, inputs[1])
    assert result.childcare_ledger.empty


def test_unchanged_nonacs_positive_energy_and_receipt_are_preserved():
    inputs = list(fixture())
    inputs[1]["spm_unit_spine"] = "asec_puf"
    inputs[1]["spm_unit_support_channel"] = "asec"
    inputs[1]["spm_unit_energy_subsidy"] = 600.0
    inputs[1]["receives_housing_assistance"] = True
    inputs[1]["takes_up_tanf_if_eligible"] = False
    inputs[2]["TEN"] = np.nan
    inputs[3]["new_spm_unit_id"] = 100
    inputs[3]["new_spm_unit_source_id"] = 70
    result = run(inputs)
    pd.testing.assert_frame_equal(result.spm_units, inputs[1])


def test_nonacs_split_is_not_authorized():
    inputs = list(fixture())
    inputs[1]["spm_unit_spine"] = "asec_puf"
    with pytest.raises(ValueError, match="non-ACS"):
        run(inputs)


def test_group_quarters_membership_and_placeholder_are_preserved():
    inputs = list(fixture(ages=(20, 21), amount=0))
    inputs[0]["RELSHIPP"] = 38
    inputs[2]["TYPEHUGQ"] = 3
    inputs[2]["TEN"] = np.nan
    inputs[3]["new_spm_unit_id"] = 100
    inputs[3]["new_spm_unit_source_id"] = 70
    result = run(inputs)
    pd.testing.assert_frame_equal(result.spm_units, inputs[1])
    inputs[3].loc[1, "new_spm_unit_id"] = 101
    inputs[3].loc[0, "new_spm_unit_source_id"] = 80
    inputs[3].loc[1, "new_spm_unit_source_id"] = 81
    with pytest.raises(ValueError, match="group quarters"):
        run(inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spm_unit_energy_subsidy", 20.0),
        ("takes_up_snap_if_eligible", False),
        ("receives_housing_assistance", True),
        ("takes_up_tanf_if_eligible", pd.NA),
    ],
)
def test_authentication_rejects_unexpected_acs_default_surface(field, value):
    inputs = list(fixture())
    inputs[1][field] = value
    with pytest.raises(ValueError, match="legacy-default"):
        run(inputs)


def test_incomplete_or_duplicate_person_membership_is_rejected():
    inputs = list(fixture())
    inputs[3] = inputs[3].iloc[:-1]
    with pytest.raises(ValueError, match="person"):
        run(inputs)


def test_complete_household_roster_is_required():
    inputs = list(fixture())
    inputs[2]["NP"] = 4
    with pytest.raises(ValueError, match="NP"):
        run(inputs)


def test_unknown_unit_owned_field_is_not_silently_copied():
    inputs = list(fixture())
    inputs[1]["new_monetary_aggregate"] = 500.0
    with pytest.raises(ValueError, match="unhandled"):
        run(inputs)


@pytest.mark.parametrize("amount", [-1, np.inf])
def test_negative_or_infinite_childcare_is_rejected(amount):
    with pytest.raises(ValueError, match="childcare"):
        run(fixture(amount=amount))


def test_string_false_is_not_a_boolean():
    inputs = list(fixture())
    inputs[1]["takes_up_housing_assistance_if_eligible"] = "False"
    with pytest.raises(ValueError, match="boolean"):
        run(inputs)


def test_row_permutation_does_not_change_result():
    inputs = fixture()
    expected = run(inputs)
    reordered = tuple(table.iloc[::-1].reset_index(drop=True) for table in inputs)
    actual = run(reordered)
    pd.testing.assert_frame_equal(actual.spm_units, expected.spm_units)
    pd.testing.assert_frame_equal(actual.crosswalk, expected.crosswalk)
    pd.testing.assert_frame_equal(actual.childcare_ledger, expected.childcare_ledger)


def test_sibling_components_cannot_share_a_new_source_identity():
    inputs = list(fixture())
    inputs[3]["new_spm_unit_source_id"] = 80
    with pytest.raises(ValueError, match="source identity"):
        run(inputs)


def test_merge_of_distinct_old_units_is_rejected():
    inputs = list(fixture())
    inputs[0].loc[2, "person_spm_unit_id"] = 101
    other = inputs[1].copy()
    other["spm_unit_id"] = 101
    other["spm_unit_source_id"] = 71
    inputs[1] = pd.concat([inputs[1], other], ignore_index=True)
    inputs[3]["new_spm_unit_id"] = 100
    inputs[3]["new_spm_unit_source_id"] = 80
    with pytest.raises(ValueError, match="merge"):
        run(inputs)


def test_canonical_label_and_generated_identity_remain_distinct():
    inputs = list(fixture())
    inputs[3]["canonical_spm_unit_id"] = ["10:1", "10:1", "10:3"]
    result = run(inputs)
    assert result.crosswalk.canonical_spm_unit_id.tolist() == ["10:1", "10:3"]
    assert result.crosswalk.source_identity_status.eq(
        "generated_component_identity"
    ).all()
    assert result.crosswalk.old_spm_unit_source_id.tolist() == [70, 70]


def test_one_canonical_unit_cannot_be_split_across_numeric_identities():
    inputs = list(fixture())
    inputs[3]["canonical_spm_unit_id"] = "10:1"
    with pytest.raises(ValueError, match="canonical SPM unit"):
        run(inputs)


@pytest.mark.parametrize("field", ["new_spm_unit_id", "new_spm_unit_source_id"])
def test_unchanged_member_set_cannot_be_relabeled(field):
    inputs = list(fixture())
    inputs[3]["new_spm_unit_id"] = 100
    inputs[3]["new_spm_unit_source_id"] = 70
    inputs[3][field] += 1000
    with pytest.raises(ValueError, match="Unchanged member sets"):
        run(inputs)


def test_changed_component_cannot_reuse_old_source_identity():
    inputs = list(fixture())
    inputs[3].loc[0:1, "new_spm_unit_source_id"] = 70
    with pytest.raises(ValueError, match="collides with an old origin"):
        run(inputs)


def test_nonbinary_currency_is_conserved_exactly_once():
    result = run(fixture(amount=1234.56))
    emitted = result.spm_units.spm_unit_pre_subsidy_childcare_expenses
    assert emitted.sum() == 1234.56
    assert emitted.gt(0).sum() == 1
    assert result.childcare_ledger.allocated_amount.iloc[0] == emitted.sum()


def test_whole_household_chunks_use_the_same_supplied_global_identity_map():
    first = fixture(amount=1234.56)
    second = tuple(table.copy() for table in fixture(amount=45.67))
    second[0]["person_id"] += 10
    second[0]["person_household_id"] += 10
    second[0]["person_spm_unit_id"] += 100
    second[1]["spm_unit_id"] += 100
    second[1]["spm_unit_source_id"] += 100
    second[2]["household_id"] += 10
    second[3]["person_id"] += 10
    second[3]["new_spm_unit_id"] += 100
    second[3]["new_spm_unit_source_id"] += 100
    combined = tuple(
        pd.concat([left, right], ignore_index=True)
        for left, right in zip(first, second, strict=True)
    )
    whole = run(combined)
    chunks = [run(first), run(second)]
    pd.testing.assert_frame_equal(
        whole.spm_units,
        pd.concat([chunk.spm_units for chunk in chunks], ignore_index=True),
    )
    pd.testing.assert_frame_equal(
        whole.childcare_ledger,
        pd.concat([chunk.childcare_ledger for chunk in chunks], ignore_index=True),
    )
