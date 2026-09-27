"""Child-support source semantics and invented actual-owner boundaries."""

import copy
import hashlib
import shutil

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_child_support_source as owner


def row(**changes):
    return {
        **{name: "0" for name in owner.READ_COLUMNS},
        "PERIDNUM": "0000000000000000000001",
        "PH_SEQ": "1",
        "A_LINENO": "1",
        "A_AGE": "40",
        "CSP_VAL": "1200",
        "CSP_YN": "1",
        "CHSP_VAL": "2000",
        "CHSP_YN": "1",
        "CHELSEW_YN": "1",
        **changes,
    }


def project(**changes):
    return owner.project_child_support_literals(pd.DataFrame([row(**changes)])).iloc[0]


@pytest.mark.parametrize(
    "receipt,amount,status,known",
    [
        ("1", "1200", "known_receipt", True),
        ("1", "0", "ambiguous_recipient_zero", False),
        ("2", "0", "known_nonreceipt", True),
        ("2", "1200", "contradictory_no_nonzero", False),
        ("0", "0", "niu", False),
        ("", "1200", "missing_receipt_literal", False),
    ],
)
def test_received_support_uses_receipt_but_does_not_complete_recipient_zeros(
    receipt, amount, status, known
):
    value = project(CSP_YN=receipt, CSP_VAL=amount)
    assert value.CSP_VAL_reporting_status == status
    assert bool(value.CSP_VAL_amount_known) is known


@pytest.mark.parametrize("obligation", ["0", "1", "2", ""])
def test_paid_zero_is_niu_under_every_obligation_answer(obligation):
    value = project(CHSP_VAL="0", CHSP_YN=obligation)
    assert value.CHSP_VAL_published_amount == 0
    assert value.CHSP_VAL_reporting_status == "declared_niu_amount"
    assert not value.CHSP_VAL_amount_known and pd.isna(value.CHSP_VAL_amount)
    assert not value.voluntary_payment_absence_known
    assert not value.obligation_implies_payment


@pytest.mark.parametrize(
    "elsewhere,obligation,status",
    [
        ("1", "1", "observed_positive_payment"),
        ("1", "2", "payment_outside_published_obligation_universe"),
        ("1", "0", "payment_outside_published_obligation_universe"),
        ("2", "1", "unresolved_child_elsewhere_route"),
        ("0", "1", "unresolved_child_elsewhere_route"),
        ("", "1", "unresolved_child_elsewhere_literal"),
        ("1", "", "missing_obligation_literal"),
        ("1", "x", "invalid_obligation_literal"),
    ],
)
def test_positive_paid_cells_remain_visible_when_routing_is_unresolved(
    elsewhere, obligation, status
):
    value = project(CHELSEW_YN=elsewhere, CHSP_YN=obligation)
    assert value.CHSP_VAL_reporting_status == status
    assert value.CHSP_VAL_published_amount == 2000
    assert bool(value.CHSP_VAL_amount_known) is (status == "observed_positive_payment")


@pytest.mark.parametrize("field", ["CSP_VAL", "CHSP_VAL"])
@pytest.mark.parametrize(
    "token,status",
    [
        ("", "missing"),
        ("-1", "malformed"),
        ("1.5", "malformed"),
        ("NA", "malformed"),
    ],
)
def test_amount_missing_malformed_and_negative_literals_remain_distinct(
    field, token, status
):
    value = project(**{field: token})
    assert value[field + "_literal"] == token
    assert value[field + "_literal_status"] == status
    assert not value[field + "_amount_known"]


def test_under15_observations_do_not_become_analytical_zeros():
    value = project(
        A_AGE="14", CSP_VAL="0", CSP_YN="0", CHSP_VAL="0", CHSP_YN="0", CHELSEW_YN="0"
    )
    for name in owner.AMOUNT_FIELDS:
        assert value[name + "_reporting_status"] == "outside_reporting_universe"
        assert not value[name + "_amount_known"]


def test_allocated_values_remain_known_and_missing_flags_remain_unknown():
    value = project(I_CHSPVAL="4", I_CSPVAL="", TCHSP_VAL="1")
    assert value.CHSP_VAL_amount == 2000 and value.CHSP_VAL_amount_known
    assert value.paid_allocation_origin == "publisher_allocated"
    assert value.received_allocation_origin == "allocation_flag_not_populated"
    assert value.I_CSPVAL_literal_status == "missing"
    assert value.TCHSP_VAL_code == 1


def test_pinned_domains_and_universe_discrepancy_are_preserved():
    entries = owner.amount_entries()
    assert entries["CHSP_VAL"]["domain"]["zero_semantics"] == "niu"
    assert entries["CHSP_VAL"]["domain"]["valid_dollar_range_inclusive"]["minimum"] == 1
    assert owner.RESPONSE_ENTRIES["CHSP_YN"][4] == "CHELSEW_YN"
    assert owner.ALLOCATION_ENTRIES["I_CHSPYN"][4] == "CHELSEW_YN = 1"
    assert "bare" in owner.OBLIGATION_UNIVERSE_NOTE


def test_nullable_seal_covers_values_masks_and_hidden_backing():
    raw = pd.DataFrame([row(CHSP_VAL="0")])
    values = owner.CurrentAsecChildSupportValues(
        owner.project_child_support_literals(raw), raw, {}
    )
    original = owner.child_support_values_seal(values)
    assert owner.child_support_values_seal(copy.deepcopy(values)) == original
    for change in ("bit", "mask", "hidden"):
        other = copy.deepcopy(values)
        array = (
            other.person.CSP_VAL_amount.array
            if change == "bit"
            else other.person.CHSP_VAL_amount.array
        )
        if change == "bit":
            array._data[0] = np.nextafter(array._data[0], np.inf)
        elif change == "mask":
            array._mask[0] = False
        else:
            array._data[0] = 17.0
        assert owner.child_support_values_seal(other) != original


def child_support_arguments(tmp_path, monkeypatch):
    from test_us_asec_coverage_authentication import _changed_parent
    from test_us_survey_population_preparation import fixture

    from microcosm.build.frame_checkpoint import load_frame_checkpoint
    from microcosm.build.us_runtime import asec_person_income_source as restoration

    arguments = fixture(tmp_path, monkeypatch)
    folder = arguments["source_dir"] / "asec"
    parent_path, attachment = folder / "parent.h5", folder / "household-attachment.h5"
    people = load_frame_checkpoint(parent_path).frame.person
    ids = people.person_id.to_numpy()
    literals = {
        105: row(I_CHSPVAL="4"),
        106: row(
            A_AGE="14",
            CSP_VAL="0",
            CSP_YN="0",
            CHSP_VAL="0",
            CHSP_YN="0",
            CHELSEW_YN="0",
        ),
        107: row(A_AGE="70", CSP_VAL="0", CSP_YN="2", CHSP_VAL="0", CHSP_YN="2"),
        108: row(CSP_VAL="0", CHSP_VAL="0"),
    }
    changes = {}
    for name in (*owner.AMOUNT_FIELDS, "A_AGE"):
        values = people[name].to_numpy(copy=True)
        for person_id, literal in literals.items():
            values[ids == person_id] = int(literal[name])
        changes[name] = values
    _changed_parent(parent_path, attachment, monkeypatch, changes)
    updated = load_frame_checkpoint(parent_path).frame.person.set_index("PERIDNUM")
    paths, pins = {}, []
    for year, member, archive, *_ in owner.routing.coverage._MEMBER_PINS:
        path = folder / f"pppub{year - 1999}.csv"
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        for name in (*owner.AMOUNT_FIELDS, "A_AGE"):
            raw[name] = [str(int(updated.loc[key, name])) for key in raw.PERIDNUM]
        for name in (
            *owner.RESPONSE_ENTRIES,
            *owner.ALLOCATION_ENTRIES,
            *owner.TOPCODE_ENTRIES,
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
    output = tmp_path / "child-support-restored-money"
    restoration.restore_asec_person_income_source(
        parent_path, attachment, member_paths=paths, output_dir=output
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, folder / "person-income-attachment.h5"
    )
    return arguments


def prepared(tmp_path, monkeypatch):
    return owner.routing.source.prepare_authenticated_survey_population(
        **child_support_arguments(tmp_path, monkeypatch)
    )


def test_actual_owner_keeps_obligation_distinct_and_refuses_copied_owner(
    tmp_path, monkeypatch
):
    parent = prepared(tmp_path, monkeypatch)
    result = owner.qualify_current_asec_child_support(parent)
    values = result.person.set_index("native_person_id")
    assert values.loc[105, "CHSP_VAL_amount"] == 2000
    assert values.loc[105, "CSP_VAL_amount"] == 1200
    assert values.loc[107, "CSP_VAL_amount"] == 0
    assert not values.loc[107, "CHSP_VAL_amount_known"]
    assert not values.loc[108, "CHSP_VAL_amount_known"]
    assert not values.loc[108, "CSP_VAL_amount_known"]
    assert not result.evidence["paid_niu_completed_with_zero"]
    assert owner.child_support_values_seal(
        owner.qualify_current_asec_child_support(parent)
    ) == owner.child_support_values_seal(result)
    with pytest.raises(ValueError):
        owner.qualify_current_asec_child_support(copy.copy(parent))

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
            owner.qualify_current_asec_child_support(parent)
    assert checked == [True]


@pytest.mark.parametrize("field", ["PERIDNUM", "A_AGE", "CSP_VAL", "CHSP_VAL"])
def test_actual_owner_requires_original_source_identity_and_amounts(
    tmp_path, monkeypatch, field
):
    parent = prepared(tmp_path, monkeypatch)
    capture = owner._capture_member

    def changed(*args):
        raw = capture(*args)
        if field == "PERIDNUM":
            raw.index = list(raw.index[:-1]) + ["9999999999999999999999"]
        else:
            raw.iloc[0, raw.columns.get_loc(field)] = "99"
        return raw

    monkeypatch.setattr(owner, "_capture_member", changed)
    with pytest.raises(ValueError):
        owner.qualify_current_asec_child_support(parent)


def test_actual_owner_requalifies_after_capture_io(tmp_path, monkeypatch):
    parent = prepared(tmp_path, monkeypatch)
    capture = owner._capture_member
    calls = []

    def changed(*args):
        raw = capture(*args)
        table = parent._checked()[2].frame.person
        table.iloc[0, table.columns.get_loc("age")] += 1
        calls.append(True)
        return raw

    monkeypatch.setattr(owner, "_capture_member", changed)
    with pytest.raises(ValueError):
        owner.qualify_current_asec_child_support(parent)
    assert calls


@pytest.mark.parametrize(
    "mutation", ["amount_bit", "masked_backing", "literal", "evidence"]
)
def test_actual_owner_final_seal_covers_complete_returned_values(
    tmp_path, monkeypatch, mutation
):
    parent = prepared(tmp_path, monkeypatch)
    seal, calls = owner.child_support_values_seal, []

    def changed(value):
        before = seal(value)
        if not calls:
            calls.append(True)
            if mutation == "amount_bit":
                array = value.person.CSP_VAL_amount.array
                i = np.flatnonzero(~array._mask)[0]
                array._data[i] = np.nextafter(array._data[i], np.inf)
            elif mutation == "masked_backing":
                array = value.person.CHSP_VAL_amount.array
                array._data[np.flatnonzero(array._mask)[0]] = 17.0
            elif mutation == "literal":
                value.asec_literals.iloc[
                    0, value.asec_literals.columns.get_loc("CHSP_YN")
                ] = "2"
            else:
                value.evidence["paid_niu_completed_with_zero"] = True
        return before

    monkeypatch.setattr(owner, "child_support_values_seal", changed)
    with pytest.raises(ValueError, match="FINAL_VALUES_CHANGED"):
        owner.qualify_current_asec_child_support(parent)
