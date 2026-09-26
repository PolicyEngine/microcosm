"""Invented literal and real-source-owner controls for the interest extension."""

import copy
import hashlib
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_interest_source as owner


def row(**changes):
    return {
        **{c: "0" for c in owner.READ_COLUMNS},
        "PERIDNUM": "0000000000000000000001",
        "PH_SEQ": "1",
        "A_LINENO": "1",
        "A_AGE": "40",
        "INT_VAL": "140",
        "INT_YN": "1",
        "TRDINT_VAL": "100",
        "RINT_YN": "1",
        "RINT_SC1": "4",
        "RINT_VAL1": "40",
        **changes,
    }


def project(**changes):
    return owner.project_interest_literals(pd.DataFrame([row(**changes)])).iloc[0]


def test_component_totals_remain_observed_and_discrepancy_is_not_balanced():
    value = project(
        INT_VAL="500", TRDINT_VAL="0", RINT_VAL1="400", RINT_SC2="7", RINT_VAL2="200"
    )
    assert value.INT_VAL_amount == 500
    assert value.TRDINT_VAL_amount == 0 and value.TRDINT_VAL_amount_known
    assert value.TRDINT_VAL_reporting_status == "observed_zero_component"
    assert value.RINT_VAL1_amount == 400 and value.RINT_VAL2_amount == 200
    assert value.published_components_sum == 600
    assert value.combined_minus_published_components == -100
    assert value.component_discrepancy_known and value.all_component_amounts_observed


def test_an_unreported_account_slot_is_not_an_observed_zero():
    value = project()
    assert value.RINT_VAL2_published_amount == 0
    assert value.RINT_VAL2_reporting_status == "unreported_account_slot"
    assert not value.RINT_VAL2_amount_known and pd.isna(value.RINT_VAL2_amount)
    assert not value.RINT_SC2_account_known
    assert (
        value.component_discrepancy_known
        and value.combined_minus_published_components == 0
    )
    assert not value.all_component_amounts_observed


@pytest.mark.parametrize(
    "changes,field,status,known",
    [
        ({"INT_VAL": "0"}, "INT_VAL", "ambiguous_recipient_zero", False),
        ({"RINT_VAL1": "0"}, "RINT_VAL1", "ambiguous_recipient_zero", False),
        ({"TRDINT_VAL": ""}, "TRDINT_VAL", "missing_amount", False),
        ({"TRDINT_VAL": "-1"}, "TRDINT_VAL", "invalid_amount_literal", False),
        ({"RINT_VAL1": "-1"}, "RINT_VAL1", "invalid_amount_literal", False),
        ({"RINT_VAL1": "4.0"}, "RINT_VAL1", "invalid_amount_literal", False),
        ({"RINT_VAL1": "NA"}, "RINT_VAL1", "invalid_amount_literal", False),
        ({"RINT_SC1": ""}, "RINT_VAL1", "missing_account_literal", False),
        ({"RINT_SC1": "9"}, "RINT_VAL1", "invalid_account_literal", False),
        ({"RINT_SC1": "0"}, "RINT_VAL1", "amount_without_account", False),
        ({"RINT_YN": ""}, "RINT_VAL1", "missing_receipt_literal", False),
        ({"RINT_YN": "2"}, "RINT_VAL1", "account_without_receipt", False),
        ({"INT_YN": "2"}, "TRDINT_VAL", "contradictory_no_nonzero", False),
        ({"INT_YN": "0", "TRDINT_VAL": "0"}, "TRDINT_VAL", "niu", False),
        ({"INT_YN": "2", "TRDINT_VAL": "0"}, "TRDINT_VAL", "known_nonreceipt", True),
    ],
)
def test_receipt_account_and_amount_conflicts_remain_explicit(
    changes, field, status, known
):
    value = project(**changes)
    assert value[field + "_reporting_status"] == status
    assert bool(value[field + "_amount_known"]) is known
    if not known:
        assert pd.isna(value[field + "_amount"])


def test_under15_universe_is_checked_without_analytical_zeros():
    value = project(
        A_AGE="14",
        INT_VAL="0",
        INT_YN="0",
        TRDINT_VAL="0",
        RINT_YN="0",
        RINT_SC1="0",
        RINT_VAL1="0",
    )
    for name in owner.AMOUNT_FIELDS:
        assert value[name + "_reporting_status"] == "outside_reporting_universe"
        assert not value[name + "_amount_known"]
    assert value.published_components_sum == 0  # Published-cell diagnostic only.


@pytest.mark.parametrize(
    "token,status",
    [
        ("", "missing"),
        ("1", "outside_printed_range"),
        ("11", "in_printed_range"),
        ("15", "in_printed_range"),
        ("x", "malformed"),
    ],
)
def test_composite_allocation_uses_printed_codes_not_header_interval(token, status):
    value = project(I_INTVAL=token)
    assert value.I_INTVAL_literal_status == status
    assert (
        value.RINT_VAL1_amount_known
    )  # Allocation never changes observation validity.
    if token in ("11", "15"):
        assert value.allocation_origin == "publisher_allocated"


def test_missing_component_or_flag_never_becomes_false_or_zero():
    value = project(RINT_VAL2="", TRINT_VAL2="", I_RINTVAL2="")
    assert pd.isna(value.published_components_sum)
    assert not value.component_discrepancy_known
    assert value.TRINT_VAL2_literal_status == "missing"
    assert value.allocation_origin == "allocation_flag_not_populated"


def test_shared_account_codes_and_pinned_total_domain_are_reused():
    assert owner.amount_entries()["INT_VAL"][4] == 999999
    assert owner.amount_entries()["TRDINT_VAL"][4] == 99999
    value = project(RINT_SC1="3")
    assert value.RINT_SC1_label == owner.routing.ACCOUNT_CODES[3] == "Roth IRA"
    assert value.RINT_SC1_account_known
    value = project(INT_YN="0", RINT_YN="0")
    assert value.INT_YN_label == value.RINT_YN_label == "niu"


def test_nullable_float_seal_preserves_masks_hidden_values_and_exact_bits():
    raw = pd.DataFrame([row()])
    values = owner.CurrentAsecInterestValues(
        owner.project_interest_literals(raw), raw, {"descriptive": True}
    )
    original = owner.interest_values_seal(values)
    assert owner.interest_values_seal(copy.deepcopy(values)) == original
    for change in ("bit", "hidden", "mask"):
        altered = copy.deepcopy(values)
        if change == "bit":
            array = altered.person.INT_VAL_amount.array
            array._data[0] = np.nextafter(array._data[0], np.inf)
        else:
            array = altered.person.RINT_VAL2_amount.array
            assert array._mask[0]
            if change == "hidden":
                array._data[0] = 17.0
            else:
                array._mask[0] = False
        assert owner.interest_values_seal(altered) != original


@pytest.mark.parametrize("change", ["bit", "validity", "literal"])
def test_retained_combined_total_requires_exact_amount_and_validity(change):
    field = SimpleNamespace(amounts=np.array([140.0]), validity=np.array([1]))
    ready = SimpleNamespace(field=lambda name: field)
    positions, literals = np.array([0]), ["140"]
    assert owner._compare_total(ready, positions, literals) is field
    if change == "bit":
        field.amounts[0] = np.nextafter(140.0, np.inf)
    elif change == "validity":
        field.validity[0] = 0
    else:
        literals[0] = "-1"
    with pytest.raises(ValueError, match="TOTAL_"):
        owner._compare_total(ready, positions, literals)


def interest_arguments(tmp_path, monkeypatch):
    """Amend original invented bytes before any preparation is issued."""
    from test_us_asec_coverage_authentication import _changed_parent
    from test_us_survey_population_preparation import fixture

    from microcosm.build.frame_checkpoint import load_frame_checkpoint
    from microcosm.build.us_runtime import asec_person_income_source as restoration

    arguments = fixture(tmp_path, monkeypatch)
    folder = arguments["source_dir"] / "asec"
    parent_path, attachment = folder / "parent.h5", folder / "household-attachment.h5"
    people = load_frame_checkpoint(parent_path).frame.person
    ids = people.person_id.to_numpy()
    amounts, ages = people.INT_VAL.to_numpy(copy=True), people.A_AGE.to_numpy(copy=True)
    literals = {
        105: row(),
        106: row(
            A_AGE="14",
            INT_VAL="0",
            INT_YN="0",
            TRDINT_VAL="0",
            RINT_YN="0",
            RINT_SC1="0",
            RINT_VAL1="0",
        ),
        107: row(
            A_AGE="70",
            INT_VAL="500",
            TRDINT_VAL="0",
            RINT_VAL1="400",
            RINT_SC2="7",
            RINT_VAL2="200",
            I_INTVAL="11",
            TRINT_VAL1="1",
        ),
        108: row(
            INT_VAL="0",
            INT_YN="2",
            TRDINT_VAL="0",
            RINT_YN="2",
            RINT_SC1="0",
            RINT_VAL1="0",
        ),
    }
    for person_id, values in literals.items():
        amounts[ids == person_id], ages[ids == person_id] = (
            int(values["INT_VAL"]),
            int(values["A_AGE"]),
        )
    _changed_parent(
        parent_path, attachment, monkeypatch, {"INT_VAL": amounts, "A_AGE": ages}
    )
    updated = load_frame_checkpoint(parent_path).frame.person.set_index("PERIDNUM")
    paths, pins = {}, []
    for year, member, archive, *_ in owner.routing.coverage._MEMBER_PINS:
        path = folder / f"pppub{year - 1999}.csv"
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        raw["INT_VAL"] = [str(int(updated.loc[key, "INT_VAL"])) for key in raw.PERIDNUM]
        raw["A_AGE"] = [str(int(updated.loc[key, "A_AGE"])) for key in raw.PERIDNUM]
        for name in (
            set(owner.READ_COLUMNS)
            - set(owner.routing.COORDINATE_COLUMNS)
            - {"INT_VAL"}
        ):
            raw[name] = [
                literals.get(int(updated.loc[key, "person_id"]), row())[name]
                for key in raw.PERIDNUM
            ]
        raw.iloc[::-1].to_csv(path, index=False)
        payload = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(payload).hexdigest(),
                len(raw),
                len(payload),
            )
        )
        paths[year] = path
    for module in (owner.routing.coverage, restoration):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = tmp_path / "interest-restored-money"
    restoration.restore_asec_person_income_source(
        parent_path, attachment, member_paths=paths, output_dir=output
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, folder / "person-income-attachment.h5"
    )
    return arguments


def prepared(tmp_path, monkeypatch):
    return owner.routing.source.prepare_authenticated_survey_population(
        **interest_arguments(tmp_path, monkeypatch)
    )


def test_actual_owner_join_keeps_source_total_and_requalifies(tmp_path, monkeypatch):
    parent = prepared(tmp_path, monkeypatch)
    result = owner.qualify_current_asec_interest(parent)
    values = result.person.set_index("native_person_id")
    assert values.loc[107, "INT_VAL_amount"] == 500
    assert values.loc[107, "combined_minus_published_components"] == -100
    assert values.loc[107, "TRINT_VAL1_code"] == 1
    assert values.loc[105, "RINT_SC1_label"] == "Regular IRA"
    assert values.loc[106, "INT_VAL_reporting_status"] == "outside_reporting_universe"
    assert values.loc[108, "RINT_VAL2_reporting_status"] == "known_nonreceipt"
    assert owner.interest_values_seal(
        owner.qualify_current_asec_interest(parent)
    ) == owner.interest_values_seal(result)
    assert (
        not result.evidence["tax_treatment_assigned"]
        and not result.evidence["source_admission_issued"]
    )
    assert result.evidence == json.loads(json.dumps(result.evidence))
    with pytest.raises(ValueError):
        owner.qualify_current_asec_interest(copy.copy(parent))

    # The actual qualifier checks the retained MoneyDomain before source capture.
    check, checked = owner._domain_agreement, []

    def rejected_domain(ready):
        check(ready)
        checked.append(True)
        raise ValueError("INVENTED_DOMAIN_REFUSAL")

    def forbidden_capture(*args):
        pytest.fail("source capture preceded live domain agreement")

    with monkeypatch.context() as patch:
        patch.setattr(owner, "_domain_agreement", rejected_domain)
        patch.setattr(owner, "_capture_member", forbidden_capture)
        with pytest.raises(ValueError, match="INVENTED_DOMAIN_REFUSAL"):
            owner.qualify_current_asec_interest(parent)
    assert checked == [True]


@pytest.mark.parametrize("coordinate", ["PH_SEQ", "A_LINENO", "A_AGE", "PERIDNUM"])
def test_actual_owner_refuses_changed_source_coordinates(
    tmp_path, monkeypatch, coordinate
):
    parent = prepared(tmp_path, monkeypatch)
    capture = owner._capture_member

    def changed(*args):
        raw = capture(*args)
        if coordinate == "PERIDNUM":
            raw.index = list(raw.index[:-1]) + ["9999999999999999999999"]
        else:
            raw.iloc[0, raw.columns.get_loc(coordinate)] = "99"
        return raw

    monkeypatch.setattr(owner, "_capture_member", changed)
    with pytest.raises(ValueError):
        owner.qualify_current_asec_interest(parent)


def test_actual_owner_refuses_mutation_after_capture_io(tmp_path, monkeypatch):
    parent = prepared(tmp_path, monkeypatch)
    capture = owner._capture_member
    called = []

    def changed(*args):
        raw = capture(*args)
        parent._checked()[2].frame.person.iloc[
            0, parent._checked()[2].frame.person.columns.get_loc("age")
        ] += 1
        called.append(True)
        return raw

    monkeypatch.setattr(owner, "_capture_member", changed)
    with pytest.raises(ValueError):
        owner.qualify_current_asec_interest(parent)
    assert called


@pytest.mark.parametrize(
    "mutation", ["amount_bit", "masked_backing", "literal", "evidence"]
)
def test_final_physical_seal_covers_complete_returned_values(
    tmp_path, monkeypatch, mutation
):
    parent = prepared(tmp_path, monkeypatch)
    seal, calls = owner.interest_values_seal, []

    def changed(value):
        before = seal(value)
        if not calls:
            calls.append(True)
            if mutation == "amount_bit":
                column = value.person["INT_VAL_amount"].array
                i = np.flatnonzero(~column._mask)[0]
                column._data[i] = np.nextafter(column._data[i], np.inf)
            elif mutation == "masked_backing":
                column = value.person["RINT_VAL2_amount"].array
                i = np.flatnonzero(column._mask)[0]
                column._data[i] = 17.0
            elif mutation == "literal":
                value.asec_literals.iloc[
                    0, value.asec_literals.columns.get_loc("RINT_SC1")
                ] = "7"
            else:
                value.evidence["tax_treatment_assigned"] = True
        return before

    monkeypatch.setattr(owner, "interest_values_seal", changed)
    with pytest.raises(ValueError, match="FINAL_VALUES_CHANGED"):
        owner.qualify_current_asec_interest(parent)


def test_default_reader_and_explicit_original_roster_agree(tmp_path):
    from test_us_current_asec_income_routing import _member_row, _write_member

    path = tmp_path / "routing.csv"
    _write_member(path, [_member_row()])
    default = owner.routing._read_capture(path, rows=1)
    explicit = owner.routing._read_capture(
        path,
        rows=1,
        columns=owner.routing.READ_COLUMNS,
        patterns=owner.routing.amount_patterns(),
    )
    pd.testing.assert_frame_equal(default, explicit)
