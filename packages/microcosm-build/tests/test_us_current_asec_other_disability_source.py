"""Invented ASEC other-disability evidence; pure fixtures confer no authority.

Every literal here is made up. The tests check the two published source slots,
the workers' compensation exclusion, the distinction between an observed zero
and NIU/missing/outside-universe, conditional allocation provenance, the exact
clone transport, and the fences that refuse a mutated or foreign owner.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_income_routing_source as routing
from microcosm.build.us_runtime import current_asec_other_disability_source as od
from microcosm.build.us_runtime import current_asec_retirement_detail_source as detail
from microcosm.build.us_runtime import disability_benefits as legacy
from microcosm.build.us_runtime import support_provenance as provenance

DETAIL_EVIDENCE = {
    "protocol": detail.PROTOCOL,
    "source_year": routing.CURRENT_INCOME_YEAR,
    "survey_year": routing.CURRENT_INCOME_YEAR + 1,
    "source_admission_issued": False,
}
UNRELATED_CELLS = {
    "PNSN_VAL": "700",
    "ANN_VAL": "-1",
    "PEN_YN": "1",
    "PEN_SC1": "3",
    "PEN_VAL1": "700",
    "SUR_YN": "1",
    "SUR_SC1": "8",
    "SUR_VAL1": "900",
    "SRVS_VAL": "900",
    "DST_VAL1": "150",
    "DBTN_VAL": "150",
    "I_PENVAL1": "4",
    "TPEN_VAL1": "1",
    "DIS_CS": "1",
    "DIS_HP": "1",
    "I_DISCS": "1",
    "I_DISHP": "1",
    "DSAB_VAL": "999",
}


def row(**changes):
    """One invented ASEC person; a single non-workers-compensation slot."""
    return {
        **{name: "0" for name in detail.READ_COLUMNS},
        "PERIDNUM": "0000000000000000000001",
        "PH_SEQ": "1",
        "A_LINENO": "1",
        "A_AGE": "40",
        "DIS_YN": "1",
        "DIS_SC1": "6",
        "DIS_VAL1": "200",
        **changes,
    }


def basis(*rows):
    literals = pd.DataFrame(list(rows))
    frame = detail.project_retirement_detail_literals(literals)
    frame.index = pd.Index(range(10, 10 + len(frame)), name="person_id")
    frame["native_person_id"] = np.arange(101, 101 + len(frame), dtype="int64")
    return literals, frame


def project(*rows):
    return od.project_other_disability(basis(*rows)[1])


def one(**changes):
    return project(row(**changes)).iloc[0]


def values(*rows):
    literals, frame = basis(*rows)
    literals.index = frame.index.copy()
    return od.compose_other_disability(
        detail.CurrentAsecRetirementDetailValues(frame, literals, dict(DETAIL_EVIDENCE))
    )


def receiving(result, *, clones=2, acs_rows=0, unrelated=True):
    """Two clones of each original source person, plus optional ACS rows."""
    records, person_id = [], 1
    for source, native in zip(
        result.person.index, result.person.native_person_id, strict=True
    ):
        for clone in range(clones):
            records.append((person_id, int(source), clone, int(native), "asec"))
            person_id += 1
    for offset in range(acs_rows):
        for clone in range(clones):
            records.append((person_id, 9000 + offset, clone, 7000 + offset, "acs"))
            person_id += 1
    people = pd.DataFrame(
        records,
        columns=[
            "person_id",
            provenance.support_source_id_column("person"),
            provenance.support_clone_index_column("person"),
            provenance.spine_source_id_column("person"),
            provenance.support_channel_column("person"),
        ],
    )
    for column in people.columns[:-1]:
        people[column] = people[column].astype("int64")
    if unrelated:
        people["age"] = np.arange(len(people), dtype="int64") + 20
        people["unrelated_leaf"] = np.arange(len(people), dtype="float64")
    return SimpleNamespace(person=people)


# --------------------------------------------------------------------------
# The two published slots and the workers' compensation exclusion
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first_code,first_amount,second_code,second_amount,total,reason",
    [
        ("6", "200", "0", "0", 200.0, "known_positive"),
        ("0", "0", "0", "0", None, "affirmed_receipt_without_reported_source"),
        ("2", "300", "7", "400", 700.0, "known_positive"),
        ("1", "500", "0", "0", 0.0, "known_zero_workers_compensation_only"),
        ("2", "300", "1", "400", 300.0, "known_positive"),
        ("1", "500", "9", "250", 250.0, "known_positive"),
        ("1", "500", "1", "400", 0.0, "known_zero_workers_compensation_only"),
    ],
)
def test_both_slots_sum_only_non_workers_compensation_sources(
    first_code, first_amount, second_code, second_amount, total, reason
):
    value = one(
        DIS_SC1=first_code,
        DIS_VAL1=first_amount,
        DIS_SC2=second_code,
        DIS_VAL2=second_amount,
    )
    assert value[od.REASON_COLUMN] == reason
    assert bool(value[od.KNOWN_COLUMN]) is (total is not None)
    if total is None:
        assert pd.isna(value[od.AMOUNT_COLUMN])
    else:
        assert value[od.AMOUNT_COLUMN] == total


@pytest.mark.parametrize("code", sorted(detail.DISABILITY_CODES))
def test_every_published_source_code_is_admitted_or_excluded_explicitly(code):
    value = one(DIS_SC1=str(code), DIS_VAL1="400")
    if code == 0:
        # A yes receipt with a nonzero amount but no source contradicts itself.
        assert value.DIS_VAL1_slot_kind == "unresolved_slot_reporting"
        assert not value[od.KNOWN_COLUMN]
        return
    if code == od.WORKERS_COMPENSATION_CODE:
        assert value.DIS_VAL1_slot_kind == "excluded_workers_compensation"
        assert value[od.AMOUNT_COLUMN] == 0.0
        assert value[od.REASON_COLUMN] == "known_zero_workers_compensation_only"
        return
    assert value.DIS_VAL1_slot_kind == "reported_source_slot"
    assert value[od.AMOUNT_COLUMN] == 400.0
    assert value.DIS_SC1_label == detail.DISABILITY_CODES[code]


def test_workers_compensation_exclusion_is_bound_to_its_printed_meaning(monkeypatch):
    codes = dict(detail.DISABILITY_CODES)
    codes[od.WORKERS_COMPENSATION_CODE] = "renamed by an editor"
    monkeypatch.setattr(detail, "SOURCE_CODES", {**detail.SOURCE_CODES, "DIS": codes})
    monkeypatch.setattr(detail, "DISABILITY_CODES", codes)
    with pytest.raises(ValueError, match="PRINTED_SEMANTICS"):
        one()


@pytest.mark.parametrize(
    "amount,status,kind",
    [
        ("500", "known_receipt", "excluded_workers_compensation"),
        ("0", "ambiguous_recipient_zero", "excluded_workers_compensation"),
        ("", "missing_amount", "excluded_workers_compensation"),
        ("abc", "invalid_amount_literal", "unresolved_slot_reporting"),
        ("1000000", "invalid_amount_literal", "unresolved_slot_reporting"),
    ],
)
def test_workers_compensation_excludes_on_a_readable_code_not_on_its_amount(
    amount, status, kind
):
    value = one(DIS_SC1="1", DIS_VAL1=amount)
    assert value.DIS_VAL1_reporting_status == status
    assert value.DIS_VAL1_slot_kind == kind
    assert bool(value[od.KNOWN_COLUMN]) is (kind == "excluded_workers_compensation")


def test_a_second_slot_is_read_independently_of_the_first():
    value = one(DIS_SC1="0", DIS_VAL1="0", DIS_SC2="4", DIS_VAL2="750")
    assert value.DIS_VAL1_slot_kind == "unused_source_slot"
    assert value.DIS_VAL2_slot_kind == "reported_source_slot"
    assert value[od.AMOUNT_COLUMN] == 750.0
    assert value.DIS_VAL1_slot_contribution == 0.0
    assert value.DIS_VAL2_slot_contribution == 750.0


# --------------------------------------------------------------------------
# Observed zero versus NIU, missing, outside universe and contradiction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes,known,reason",
    [
        (
            {"DIS_YN": "2", "DIS_SC1": "0", "DIS_VAL1": "0"},
            True,
            "known_zero_nonreceipt",
        ),
        (
            {"DIS_YN": "0", "DIS_SC1": "0", "DIS_VAL1": "0"},
            False,
            "niu_not_observed_zero",
        ),
        ({"DIS_VAL1": ""}, False, "unresolved_slot_reporting"),
        ({"DIS_YN": ""}, False, "unresolved_slot_reporting"),
        ({"DIS_SC1": ""}, False, "unresolved_slot_reporting"),
        ({"DIS_YN": "3"}, False, "unresolved_slot_reporting"),
        ({"DIS_SC1": "11"}, False, "unresolved_slot_reporting"),
        ({"DIS_YN": "2", "DIS_SC1": "6"}, False, "unresolved_slot_reporting"),
        ({"DIS_SC1": "6", "DIS_VAL1": "0"}, False, "unresolved_slot_reporting"),
        (
            {"A_AGE": "14", "DIS_YN": "0", "DIS_SC1": "0", "DIS_VAL1": "0"},
            False,
            "outside_age_universe",
        ),
        (
            {"A_AGE": "0", "DIS_YN": "0", "DIS_SC1": "0", "DIS_VAL1": ""},
            False,
            "outside_age_universe",
        ),
    ],
)
def test_unobserved_slots_never_become_an_observed_zero(changes, known, reason):
    value = one(**changes)
    assert bool(value[od.KNOWN_COLUMN]) is known
    assert value[od.REASON_COLUMN] == reason
    if not known:
        assert pd.isna(value[od.AMOUNT_COLUMN])
    else:
        assert value[od.AMOUNT_COLUMN] == 0.0


def test_a_recipient_zero_under_a_reported_source_stays_ambiguous():
    value = one(DIS_SC1="6", DIS_VAL1="0")
    assert value.DIS_VAL1_reporting_status == "ambiguous_recipient_zero"
    assert value.DIS_VAL1_slot_kind == "unresolved_slot_reporting"
    assert pd.isna(value.DIS_VAL1_slot_contribution)


def test_an_affirmed_receipt_without_a_populated_source_is_not_zero():
    value = one(DIS_SC1="0", DIS_VAL1="0", DIS_SC2="0", DIS_VAL2="0")
    assert value.DIS_VAL1_slot_kind == value.DIS_VAL2_slot_kind == "unused_source_slot"
    assert value[od.REASON_COLUMN] == "affirmed_receipt_without_reported_source"
    assert pd.isna(value[od.AMOUNT_COLUMN])


def test_one_unresolved_slot_withholds_the_whole_person():
    value = one(DIS_SC1="6", DIS_VAL1="200", DIS_SC2="2", DIS_VAL2="")
    assert value.DIS_VAL1_slot_kind == "reported_source_slot"
    assert value.DIS_VAL2_slot_kind == "unresolved_slot_reporting"
    assert not value[od.KNOWN_COLUMN]
    assert pd.isna(value[od.AMOUNT_COLUMN])


def test_reason_and_slot_vocabularies_stay_closed():
    frame = project(
        row(),
        row(DIS_SC1="1", DIS_VAL1="500"),
        row(DIS_YN="2", DIS_SC1="0", DIS_VAL1="0"),
        row(DIS_YN="0", DIS_SC1="0", DIS_VAL1="0"),
        row(A_AGE="4", DIS_YN="0", DIS_SC1="0", DIS_VAL1="0"),
        row(DIS_VAL1=""),
        row(DIS_SC1="0", DIS_VAL1="0"),
    )
    assert set(frame[od.REASON_COLUMN]) <= set(od.REASONS)
    assert set(frame[od.REASON_COLUMN]) == set(od.REASONS) - {od.UNOBSERVED_REASON}
    for name in od.AMOUNT_FIELDS:
        assert set(frame[name + "_slot_kind"]) <= set(od.SLOT_KINDS)


# --------------------------------------------------------------------------
# The retired arithmetic is reused, and its divergences are recorded
# --------------------------------------------------------------------------


def test_the_archived_two_slot_arithmetic_is_reused_verbatim():
    frame = project(
        row(DIS_SC1="2", DIS_VAL1="300", DIS_SC2="1", DIS_VAL2="400"),
        row(DIS_SC1="1", DIS_VAL1="500"),
    )
    expected = legacy.derive_us_disability_benefits_from_asec(
        pd.DataFrame(
            {
                "DIS_VAL1": [300.0, 500.0],
                "DIS_SC1": [2.0, 1.0],
                "DIS_VAL2": [400.0, 0.0],
                "DIS_SC2": [1.0, 0.0],
            },
            index=frame.index,
        )
    )[legacy.US_DISABILITY_BENEFITS_OUTPUT_COLUMNS[0]]
    assert list(frame.other_disability_archived_arithmetic_amount) == list(expected)
    assert list(frame[od.AMOUNT_COLUMN]) == list(expected)
    assert frame.other_disability_archived_arithmetic_agrees.all()


@pytest.mark.parametrize(
    "changes",
    [
        {"DIS_YN": "0", "DIS_SC1": "0", "DIS_VAL1": "0"},
        {"A_AGE": "3", "DIS_YN": "0", "DIS_SC1": "0", "DIS_VAL1": "0"},
        {"DIS_SC1": "0", "DIS_VAL1": "0", "DIS_SC2": "0", "DIS_VAL2": "0"},
    ],
)
def test_archived_zeroes_are_recorded_but_never_adopted(changes):
    value = one(**changes)
    assert value.other_disability_archived_arithmetic_evaluable
    assert value.other_disability_archived_arithmetic_amount == 0.0
    assert not value[od.KNOWN_COLUMN]
    assert pd.isna(value[od.AMOUNT_COLUMN])
    assert pd.isna(value.other_disability_archived_arithmetic_agrees)


def test_unreadable_literals_leave_the_archived_arithmetic_unevaluable():
    value = one(DIS_VAL1="abc")
    assert not value.other_disability_archived_arithmetic_evaluable
    assert pd.isna(value.other_disability_archived_arithmetic_amount)
    assert pd.isna(value.other_disability_archived_arithmetic_agrees)


def test_archived_parameter_drift_refuses(monkeypatch):
    monkeypatch.setattr(
        legacy,
        "US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS",
        ("DIS_VAL1", "DIS_SC1"),
    )
    with pytest.raises(ValueError, match="ARCHIVED_ROSTER"):
        one()


def test_owner_roster_drift_refuses(monkeypatch):
    monkeypatch.setattr(detail, "RETAINED_MONEY_FIELDS", ("PNSN_VAL",))
    with pytest.raises(ValueError, match="OWNER_ROSTER"):
        one()


@pytest.mark.parametrize(
    "name,replacement",
    [
        ("DIS_SC1", (2, 461, 44, "6C-23", "DIS_YN > 0")),
        ("DIS_SC2", (2, 463, 44, "6C-23", "All Persons aged 15+")),
    ],
)
def test_source_code_universe_drift_refuses(monkeypatch, name, replacement):
    entries = dict(detail.SOURCE_ENTRIES)
    entries[name] = replacement
    monkeypatch.setattr(detail, "SOURCE_ENTRIES", entries)
    with pytest.raises(ValueError, match="PRINTED_SEMANTICS"):
        one()


# --------------------------------------------------------------------------
# Conditional allocation provenance and topcode censoring
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag,changes,status,origin",
    [
        ("I_DISVL1", {}, "not_allocated_in_flag_universe", "published_flags_all_zero"),
        ("I_DISVL2", {}, "outside_flag_universe", "published_flags_all_zero"),
        ("I_DISVL1", {"I_DISVL1": "5"}, "publisher_allocated", "publisher_allocated"),
        (
            "I_DISVL1",
            {"I_DISVL1": ""},
            "allocation_flag_not_populated",
            "allocation_flag_not_populated",
        ),
        (
            "I_DISSC1",
            {"I_DISSC1": "3"},
            "unresolved_allocation_literal",
            "unresolved_allocation_provenance",
        ),
        ("I_DISYN", {"I_DISYN": "7"}, "publisher_allocated", "publisher_allocated"),
    ],
)
def test_allocation_flags_keep_their_conditional_universes(
    flag, changes, status, origin
):
    value = one(**changes)
    assert value[flag + "_allocation_status"] == status
    assert value.other_disability_allocation_origin == origin
    # Provenance never becomes knownness in either direction.
    assert bool(value[od.KNOWN_COLUMN]) is True
    assert value[od.AMOUNT_COLUMN] == 200.0
    assert not value.other_disability_allocation_qualifies_receipt


def test_an_unreadable_amount_leaves_its_flag_universe_unresolved():
    value = one(DIS_VAL1="")
    assert value.I_DISVL1_flag_universe is pd.NA
    assert value.I_DISVL1_allocation_status == "unresolved_flag_universe"
    assert not value[od.KNOWN_COLUMN]


def test_work_limitation_flags_are_not_income_allocation_provenance():
    assert "I_DISCS" not in od.ALLOCATION_FIELDS
    assert "I_DISHP" not in od.ALLOCATION_FIELDS
    plain = one()
    limited = one(DIS_CS="1", DIS_HP="1", I_DISCS="4", I_DISHP="4")
    assert (
        plain.other_disability_allocation_origin
        == limited.other_disability_allocation_origin
    )
    assert limited[od.AMOUNT_COLUMN] == plain[od.AMOUNT_COLUMN]


@pytest.mark.parametrize(
    "changes,censored",
    [
        ({}, False),
        ({"TDISVAL1": "1"}, True),
        ({"TDISVAL1": ""}, None),
        ({"DIS_SC1": "1", "DIS_VAL1": "500", "TDISVAL1": "1"}, False),
        ({"DIS_YN": "2", "DIS_SC1": "0", "DIS_VAL1": "0", "TDISVAL1": "1"}, False),
    ],
)
def test_topcodes_describe_only_the_dollars_this_leaf_admits(changes, censored):
    value = one(**changes)
    if censored is None:
        assert pd.isna(value.other_disability_topcoded)
    else:
        assert bool(value.other_disability_topcoded) is censored


def test_an_unknown_person_has_no_topcode_reading():
    assert pd.isna(one(DIS_VAL1="", TDISVAL1="1").other_disability_topcoded)


# --------------------------------------------------------------------------
# Owner agreement, mutation and unrelated-cell preservation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("column", sorted(set(od.required_basis_columns())))
def test_every_declared_basis_column_is_required(column):
    _, frame = basis(row())
    with pytest.raises(ValueError, match="BASIS_COLUMNS"):
        od.project_other_disability(frame.drop(columns=[column]))


def test_unrelated_source_cells_do_not_move_this_leaf():
    plain = project(row())
    noisy = project(row(**UNRELATED_CELLS))
    pd.testing.assert_frame_equal(plain, noisy)


def test_owner_knownness_disagreement_refuses():
    _, frame = basis(row())
    frame["DIS_VAL1_amount_known"] = ~frame.DIS_VAL1_amount_known
    with pytest.raises(ValueError, match="OWNER_KNOWNNESS"):
        od.project_other_disability(frame)


def test_money_owner_axes_are_cross_checked_when_supplied():
    _, frame = basis(row())
    for name in od.AMOUNT_FIELDS:
        frame[name + "_parent_statuses"] = np.int16(2)
        frame[name + "_parent_validity"] = np.int16(1)
        frame[name + "_parent_zero_origin"] = np.int16(0)
    projected = od.project_other_disability(frame)
    assert bool(projected.other_disability_money_owner_axes_present.iloc[0])
    assert projected.DIS_VAL1_parent_validity.iloc[0] == 1
    frame["DIS_VAL1_parent_validity"] = np.int16(0)
    with pytest.raises(ValueError, match="PARENT_VALIDITY"):
        od.project_other_disability(frame)


def test_money_owner_axes_are_optional_and_declared():
    projected = project(row())
    assert not projected.other_disability_money_owner_axes_present.iloc[0]
    assert "DIS_VAL1_parent_validity" not in projected


@pytest.mark.parametrize(
    "mutation",
    [
        "own_projection",
        "owner_source_codes",
        "owner_known_statuses",
        "archived_arithmetic",
        "archived_parameters",
    ],
)
def test_composition_refuses_a_mutated_owner(monkeypatch, mutation):
    literals, frame = basis(row())
    literals.index = frame.index.copy()
    owner = detail.CurrentAsecRetirementDetailValues(
        frame, literals, dict(DETAIL_EVIDENCE)
    )
    if mutation == "own_projection":
        original = od._slot_kind
        monkeypatch.setattr(od, "_slot_kind", lambda *a, **k: original(*a, **k))
    elif mutation == "owner_source_codes":
        codes = dict(detail.DISABILITY_CODES)
        codes[10] = "edited in place"
        monkeypatch.setattr(detail, "DISABILITY_CODES", codes)
    elif mutation == "owner_known_statuses":
        monkeypatch.setattr(routing, "KNOWN_AMOUNT_STATUSES", ("known_receipt",))
    elif mutation == "archived_arithmetic":
        original = legacy.derive_us_disability_benefits_from_asec
        monkeypatch.setattr(
            legacy,
            "derive_us_disability_benefits_from_asec",
            lambda *a, **k: original(*a, **k),
        )
    else:
        monkeypatch.setattr(
            legacy,
            "_EXPECTED_DIRECT_PARAMETERS",
            {**legacy._EXPECTED_DIRECT_PARAMETERS, "workers_compensation_code": 9},
        )
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        od.compose_other_disability(owner)


def test_composition_refuses_an_owner_that_changes_under_it(monkeypatch):
    literals, frame = basis(row())
    literals.index = frame.index.copy()
    owner = detail.CurrentAsecRetirementDetailValues(
        frame, literals, dict(DETAIL_EVIDENCE)
    )
    real = detail.retirement_detail_values_seal
    seen = []

    def drifting(value):
        seen.append(value)
        return real(value) + ":" + str(len(seen))

    monkeypatch.setattr(detail, "retirement_detail_values_seal", drifting)
    monkeypatch.setattr(od, "_LIVE", od._live())
    with pytest.raises(ValueError, match="DETAIL_VALUES_CHANGED"):
        od.compose_other_disability(owner)


@pytest.mark.parametrize(
    "mutation",
    ["type", "protocol", "year", "survey_year", "admission", "native"],
)
def test_composition_refuses_a_foreign_or_unqualified_owner(mutation):
    literals, frame = basis(row())
    literals.index = frame.index.copy()
    evidence = dict(DETAIL_EVIDENCE)
    if mutation == "protocol":
        evidence["protocol"] = "microcosm.us.invented.v1"
    elif mutation == "year":
        evidence["source_year"] = routing.CURRENT_INCOME_YEAR - 1
    elif mutation == "survey_year":
        evidence["survey_year"] = routing.CURRENT_INCOME_YEAR
    elif mutation == "admission":
        evidence["source_admission_issued"] = True
    elif mutation == "native":
        frame = frame.drop(columns=["native_person_id"])
    owner = (
        SimpleNamespace(person=frame, asec_literals=literals, evidence=evidence)
        if mutation == "type"
        else detail.CurrentAsecRetirementDetailValues(frame, literals, evidence)
    )
    expected = {
        "type": "DETAIL_OWNER_TYPE",
        "native": "DETAIL_NATIVE_AXIS",
    }.get(mutation, "DETAIL_EVIDENCE")
    with pytest.raises(ValueError, match=expected):
        od.compose_other_disability(owner)


def test_values_seal_covers_masked_backing_and_evidence():
    result = values(row(), row(DIS_VAL1=""))
    seal = od.other_disability_values_seal(result)
    assert od.other_disability_values_seal(copy.deepcopy(result)) == seal
    for kind in ("masked_backing", "mask", "label", "evidence"):
        changed = copy.deepcopy(result)
        if kind == "masked_backing":
            changed.person[od.AMOUNT_COLUMN].array._data[1] = 17.0
        elif kind == "mask":
            changed.person[od.AMOUNT_COLUMN].array._mask[0] = True
        elif kind == "label":
            changed.person.loc[10, od.REASON_COLUMN] = "known_positive "
        else:
            changed.evidence["invented"] = True
        assert od.other_disability_values_seal(changed) != seal


def test_evidence_claims_nothing_beyond_the_observed_slots():
    result = values(row(), row(DIS_YN="0", DIS_SC1="0", DIS_VAL1="0"))
    evidence = result.evidence
    for claim in (
        "acs_completion_assigned",
        "under15_completed_with_zero",
        "niu_completed_with_zero",
        "unknown_completed_with_zero",
        "social_security_disability_included",
        "workers_compensation_included",
        "ssi_eligibility_assigned",
        "taxability_assigned",
        "topcode_corrected",
        "allocation_flags_qualify_receipt",
        "model_fitted",
        "clone_redraw_issued",
        "raw_member_read",
        "source_admission_issued",
        "release_eligible",
    ):
        assert evidence[claim] is False
    assert evidence["canonical_leaf"] == legacy.US_DISABILITY_BENEFITS_OUTPUT_COLUMNS[0]
    assert evidence["retirement_detail_protocol"] == detail.PROTOCOL
    assert evidence["workers_compensation_label"] == "worker's compensation"
    assert evidence["rows"] == 2 and evidence["known_rows"] == 1
    assert evidence["reason_counts"]["niu_not_observed_zero"] == 1


# --------------------------------------------------------------------------
# Clone transport and receiver identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("clones", [1, 2, 3])
def test_one_source_row_is_copied_to_every_clone_without_a_redraw(clones):
    result = values(row(), row(DIS_SC1="1", DIS_VAL1="500"))
    frame = receiving(result, clones=clones)
    attachment = od.attach_other_disability_columns(result, frame)
    leaf = attachment.columns["person", legacy.US_DISABILITY_BENEFITS_OUTPUT_COLUMNS[0]]
    assert list(leaf) == [200.0] * clones + [0.0] * clones
    assert attachment.receipt["clones_per_source"] == clones
    assert attachment.receipt["redraw_issued"] is False
    assert attachment.receipt["draws_consumed"] == 0
    assert attachment.receipt["unknown_completed_with_zero"] is False
    assert attachment.receipt["values_sha256"] == od.other_disability_values_seal(
        result
    )
    reason = attachment.columns["person", od.REPORT_PREFIX + "reason"]
    assert (
        list(reason)
        == ["known_positive"] * clones
        + ["known_zero_workers_compensation_only"] * clones
    )


def test_unobserved_arms_stay_unknown_rather_than_zero():
    result = values(row())
    frame = receiving(result, acs_rows=2)
    attachment = od.attach_other_disability_columns(result, frame)
    leaf = attachment.columns["person", legacy.US_DISABILITY_BENEFITS_OUTPUT_COLUMNS[0]]
    known = attachment.columns["person", od.REPORT_PREFIX + "known"]
    reason = attachment.columns["person", od.REPORT_PREFIX + "reason"]
    assert list(leaf)[:2] == [200.0, 200.0]
    assert all(pd.isna(value) for value in list(leaf)[2:])
    assert list(known) == [True, True] + [False] * 4
    assert list(reason)[2:] == [od.UNOBSERVED_REASON] * 4
    assert attachment.receipt["unobserved_channels"] == ["acs"]
    assert attachment.receipt["unobserved_rows"] == 4
    assert attachment.receipt["observed_rows"] == 2


def test_attachment_requires_receiver_source_identity():
    result = values(row(), row(DIS_VAL1="300"))
    frame = receiving(result)
    frame.person.loc[0, provenance.spine_source_id_column("person")] = 999
    with pytest.raises(ValueError, match="RECEIVER_SOURCE_IDENTITY"):
        od.attach_other_disability_columns(result, frame)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("missing_source", "SOURCE_ROSTER"),
        ("extra_source", "SOURCE_ROSTER"),
        ("acs_collision", "SOURCE_ROSTER"),
        ("foreign_channel", "RECEIVING_CHANNEL_ROSTER"),
        ("ragged_clones", "CLONE_COUNT"),
        ("repeated_clone_index", "CLONE_INDEX_SET"),
        ("missing_column", "RECEIVING_AXES"),
        ("float_ids", "RECEIVING_AXES"),
        ("duplicate_person", "RECEIVING_AXES"),
        ("clone_native_disagreement", "CLONE_SOURCE_AGREEMENT"),
    ],
)
def test_attachment_refuses_a_broken_receiving_roster(mutation, reason):
    result = values(row(), row(DIS_VAL1="300"))
    frame = receiving(result, acs_rows=1)
    people = frame.person
    source_column = provenance.support_source_id_column("person")
    clone_column = provenance.support_clone_index_column("person")
    channel_column = provenance.support_channel_column("person")
    if mutation == "missing_source":
        frame.person = people.iloc[2:].reset_index(drop=True)
    elif mutation == "extra_source":
        people.loc[0, source_column] = 555
    elif mutation == "acs_collision":
        people.loc[4, source_column] = int(result.person.index[0])
    elif mutation == "foreign_channel":
        people.loc[0, channel_column] = "puf_tax_detail"
    elif mutation == "ragged_clones":
        frame.person = people.iloc[1:].reset_index(drop=True)
    elif mutation == "repeated_clone_index":
        people.loc[1, clone_column] = 0
    elif mutation == "missing_column":
        frame.person = people.drop(columns=[clone_column])
    elif mutation == "float_ids":
        people[source_column] = people[source_column].astype("float64")
    elif mutation == "duplicate_person":
        people.loc[1, "person_id"] = people.loc[0, "person_id"]
    else:
        # An unobserved clone pair whose two rows claim different originals.
        people.loc[5, provenance.spine_source_id_column("person")] = 7777
    with pytest.raises(ValueError, match=reason):
        od.attach_other_disability_columns(result, frame)


def test_attachment_preserves_unrelated_receiving_cells():
    result = values(row())
    frame = receiving(result, acs_rows=1)
    before = frame.person.copy(deep=True)
    attachment = od.attach_other_disability_columns(result, frame)
    pd.testing.assert_frame_equal(frame.person, before)
    attached = {name for _, name in attachment.columns}
    assert not attached & set(before.columns)
    assert attached == set(attachment.receipt["attached_columns"])
    assert all(entity == "person" for entity, _ in attachment.columns)
    assert "native_person_id" not in attached
    assert od.REPORT_PREFIX + "native_person_id" not in attached


def test_a_copied_values_object_attaches_identically():
    result = values(row(), row(DIS_SC1="2", DIS_VAL1="450"))
    frame = receiving(result)
    original = od.attach_other_disability_columns(result, frame)
    copied = od.attach_other_disability_columns(copy.deepcopy(result), frame)
    assert original.receipt == copied.receipt
    assert set(original.columns) == set(copied.columns)
    for key, series in original.columns.items():
        pd.testing.assert_series_equal(series, copied.columns[key])


def test_a_foreign_values_object_is_refused():
    result = values(row())
    frame = receiving(result)
    foreign = SimpleNamespace(person=result.person, evidence=result.evidence)
    with pytest.raises(ValueError, match="VALUES_TYPE"):
        od.attach_other_disability_columns(foreign, frame)


def test_attachment_refuses_a_mutated_implementation(monkeypatch):
    result = values(row())
    frame = receiving(result)
    monkeypatch.setattr(od, "WORKERS_COMPENSATION_CODE", 2)
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        od.attach_other_disability_columns(result, frame)


def test_attached_names_keep_the_canonical_leaf_and_one_report_prefix():
    result = values(row())
    attachment = od.attach_other_disability_columns(result, receiving(result))
    names = sorted(name for _, name in attachment.columns)
    leaf = legacy.US_DISABILITY_BENEFITS_OUTPUT_COLUMNS[0]
    assert leaf in names
    assert all(name == leaf or name.startswith(od.REPORT_PREFIX) for name in names)
    assert not any(
        name.startswith(od.REPORT_PREFIX + "other_disability") for name in names
    )
