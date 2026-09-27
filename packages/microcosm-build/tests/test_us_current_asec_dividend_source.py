"""Invented dividend literals, survivor routes and actual retained source owners."""

import copy
import hashlib
import json
import shutil
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_dividend_source as owner


def row(**changes):
    return {
        **{name: "0" for name in owner.READ_COLUMNS},
        "PERIDNUM": "0000000000000000000001",
        "PH_SEQ": "1",
        "A_LINENO": "1",
        "A_AGE": "40",
        "DIV_YN": "1",
        "DIV_VAL": "140",
        "SUR_YN": "2",
        **changes,
    }


def project(**changes):
    return owner.project_dividend_literals(pd.DataFrame([row(**changes)])).iloc[0]


@pytest.mark.parametrize(
    "changes,status,amount",
    [
        ({}, "known_receipt", 140),
        ({"DIV_VAL": "0", "DIV_YN": "2"}, "known_nonreceipt", 0),
        ({"DIV_VAL": "0"}, "ambiguous_recipient_zero", None),
        ({"DIV_VAL": "0", "DIV_YN": "0"}, "niu", None),
        ({"DIV_YN": "0"}, "contradictory_niu_nonzero", None),
        ({"DIV_YN": "2"}, "contradictory_no_nonzero", None),
        ({"DIV_YN": ""}, "missing_receipt_literal", None),
        ({"DIV_YN": "x"}, "unrecognized_receipt_literal", None),
        ({"DIV_VAL": ""}, "missing_amount", None),
        ({"DIV_VAL": "-1"}, "invalid_amount_literal", None),
        ({"DIV_VAL": "1.0"}, "invalid_amount_literal", None),
        (
            {"A_AGE": "14", "DIV_VAL": "0", "DIV_YN": "0"},
            "outside_reporting_universe",
            None,
        ),
        ({"A_AGE": "14"}, "contradictory_outside_reporting_universe", None),
    ],
)
def test_dividend_receipt_unknownness_never_infers_age_only_or_recipient_zeros(
    changes, status, amount
):
    value = project(**changes)
    assert value.DIV_VAL_reporting_status == status
    assert bool(value.DIV_VAL_amount_known) is (amount is not None)
    assert (
        pd.isna(value.DIV_VAL_amount)
        if amount is None
        else value.DIV_VAL_amount == amount
    )
    assert value.DIV_VAL_literal == changes.get("DIV_VAL", "140")


@pytest.mark.parametrize(
    "token,header,reference,agreement",
    [
        ("0", "in_printed_range", "in_printed_range", "header_reference_agree"),
        ("1", "in_printed_range", "in_printed_range", "header_reference_agree"),
        ("5", "outside_printed_range", "in_printed_range", "header_reference_conflict"),
        ("9", "outside_printed_range", "in_printed_range", "header_reference_conflict"),
        ("", "missing", "missing", "literal_unresolved"),
        ("x", "malformed", "malformed", "literal_unresolved"),
        ("01", "malformed", "malformed", "literal_unresolved"),
    ],
)
def test_dividend_receipt_allocation_preserves_both_conflicting_codebook_axes(
    token, header, reference, agreement
):
    value = project(I_DIVYN=token)
    assert value.I_DIVYN_literal == token
    assert value.I_DIVYN_header_range_status == header
    assert value.I_DIVYN_referenced_values_status == reference
    assert value.I_DIVYN_codebook_status == agreement
    if reference == "in_printed_range":
        assert value.I_DIVYN_code == int(token)
        assert value.I_DIVYN_literal_status == "well_formed"
    assert value.DIV_VAL_amount == 140 and value.DIV_VAL_amount_known


@pytest.mark.parametrize(
    "changes,status,clear",
    [
        ({}, "known_nonreceipt", True),
        ({"SUR_YN": "1", "SUR_SC1": "1"}, "known_other_survivor_sources", True),
        ({"SUR_YN": "1", "SUR_SC2": "9"}, "known_other_survivor_sources", True),
        ({"SUR_YN": "1", "SUR_SC2": "8"}, "possible_estate_or_trust_route", False),
        (
            {"SUR_YN": "1", "SUR_SC1": "8", "SUR_SC2": ""},
            "possible_estate_or_trust_route",
            False,
        ),
        ({"SUR_YN": "1", "SUR_SC1": "10"}, "source_type_unspecified", None),
        ({"SUR_YN": "1"}, "unreported_source_slots", None),
        (
            {"SUR_YN": "1", "SUR_SC1": "1", "SUR_SC2": ""},
            "unresolved_source_literal",
            None,
        ),
        ({"SUR_YN": "1", "SUR_SC1": "11"}, "unresolved_source_literal", None),
        ({"SUR_YN": "1", "SUR_SC1": "x"}, "unresolved_source_literal", None),
        ({"SUR_YN": "", "SUR_SC1": "8"}, "missing_receipt_literal", None),
        ({"SUR_YN": "x"}, "unrecognized_receipt_literal", None),
        ({"SUR_SC1": "8"}, "unresolved_nonreceipt_source_route", None),
        ({"SUR_SC1": ""}, "unresolved_nonreceipt_source_route", None),
        ({"SUR_YN": "0"}, "niu", None),
        ({"SUR_YN": "0", "SUR_SC1": "1"}, "unresolved_niu_route", None),
        ({"A_AGE": "14", "SUR_YN": "0"}, "outside_reporting_universe", None),
        ({"A_AGE": "14", "SUR_YN": ""}, "unresolved_outside_reporting_universe", None),
        (
            {"A_AGE": "14", "SUR_YN": "0", "SUR_SC1": ""},
            "unresolved_outside_reporting_universe",
            None,
        ),
    ],
)
def test_survivor_clearance_requires_resolved_receipt_and_sources(
    changes, status, clear
):
    value = project(**changes)
    assert value.survivor_property_route_status == status
    assert (
        pd.isna(value.survivor_property_route_clear)
        if clear is None
        else bool(value.survivor_property_route_clear) is clear
    )
    assert value.DIV_VAL_published_amount == 140  # Routes never change donor values.
    if "8" in (changes.get("SUR_SC1"), changes.get("SUR_SC2")):
        assert value.survivor_estate_or_trust_code_present
    if changes.get("SUR_SC1") == "10":
        assert value.SUR_SC1_literal_status == "in_printed_range"
        assert value.survivor_unspecified_source_code_present


@pytest.mark.parametrize(
    "field,token",
    [
        ("I_DIVVAL", "9"),
        ("I_DIVVAL", ""),
        ("I_DIVVAL", "x"),
        ("I_DIVYN", "5"),
        ("TDIV_VAL", "1"),
        ("TDIV_VAL", ""),
        ("TRNT_VAL", "1"),
        ("TRNT_VAL", ""),
    ],
)
def test_source_allocation_topcode_flags_are_never_donor_filters(field, token):
    value = project(**{field: token})
    assert value.DIV_VAL_amount == 140 and value.DIV_VAL_amount_known
    assert bool(value.survivor_property_route_clear)
    assert value[field + "_literal"] == token


def test_source_roster_and_primary_pin_have_no_survivor_amount_or_tax_outputs():
    assert owner.amount_entries()["DIV_VAL"][:4] == (6, 478, 45, "6C-24")
    assert owner.ALLOCATION_ENTRIES["I_DIVYN"][5:] == ("0:1", "See I_ANNVAL")
    assert owner.SURVIVOR_CODES[8] == "regular payments from estates or trusts"
    assert owner.SURVIVOR_CODES[10] == "other or don't know"
    assert not {
        "SUR_VAL1",
        "SUR_VAL2",
        "SRVS_VAL",
        "OI_VAL",
        "RNT_VAL",
        "CAP_VAL",
    } & set(owner.READ_COLUMNS)


def test_cached_domain_entry_does_not_expose_a_shared_mutable_mapping():
    original = owner.amount_entries()
    changed = owner.amount_entries()
    changed["DIV_VAL"] = (99, *changed["DIV_VAL"][1:])
    changed["invented"] = ()
    assert owner.amount_entries() == original
    assert type(owner._amount_entry()) is tuple
    assert project().DIV_VAL_published_amount == 140


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_field",
        "absent_vintage",
        "duplicate_vintage",
        "negative_code",
        "negative_dollars",
        "zero_semantics",
        "missing_code",
        "minimum",
        "dictionary",
    ],
)
def test_supported_domain_and_unique_current_vintage_are_required(change):
    data = json.loads(
        owner.resources.files(owner.__package__)
        .joinpath(owner.routing.DOMAINS_RESOURCE)
        .read_bytes()
    )
    field = next(f for f in data["fields"] if f["name"] == "DIV_VAL")
    current = next(v for v in field["vintages"] if v["income_year"] == 2024)
    assert owner._decode_amount_entry(data) == owner.amount_entries()["DIV_VAL"]
    if change == "duplicate_field":
        data["fields"].append(copy.deepcopy(field))
    elif change == "absent_vintage":
        field["vintages"].remove(current)
    elif change == "duplicate_vintage":
        field["vintages"].append(copy.deepcopy(current))
    elif change == "negative_code":
        field["domain"]["declared_negative_nonmoney_codes"] = [-1]
    elif change == "negative_dollars":
        field["domain"]["negative_dollars_permitted"] = True
    elif change == "zero_semantics":
        field["domain"]["zero_semantics"] = "valid_zero_dollars"
    elif change == "missing_code":
        field["domain"]["declared_other_missing_codes"] = [999999]
    elif change == "minimum":
        field["domain"]["encoded_range_inclusive"]["minimum"] = -1
    else:
        current["dictionary_spelling"] = "WRONG"
    with pytest.raises(ValueError, match="ASEC_DIVIDEND_SOURCE_(DOMAIN_|DICTIONARY_)"):
        owner._decode_amount_entry(data)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "entity", "minimum", "maximum", "zero_semantics"]
)
def test_live_ready_money_domain_must_match_the_pinned_source_contract(change):
    domain = SimpleNamespace(
        name="DIV_VAL",
        entity="person",
        minimum=0,
        maximum=999999,
        zero_semantics="none_or_niu_not_distinguishable_from_amount_alone",
    )
    fields = [domain]
    ready = SimpleNamespace(
        bindings=SimpleNamespace(spec=SimpleNamespace(fields=fields))
    )
    owner._domain_agreement(ready)
    if change == "missing":
        fields.clear()
    elif change == "duplicate":
        fields.append(copy.copy(domain))
    else:
        setattr(
            domain, change, "wrong" if change in ("entity", "zero_semantics") else 1
        )
    with pytest.raises(ValueError, match="MONEY_DOMAIN_"):
        owner._domain_agreement(ready)


@pytest.mark.parametrize("change", ["amount_bit", "validity", "literal", "missing"])
def test_current_money_total_requires_exact_bits_and_literal_validity(change):
    field = SimpleNamespace(amounts=np.array([140.0, 0.0]), validity=np.array([1, 0]))
    ready = SimpleNamespace(field=lambda name: field)
    positions, literals = np.array([0, 1]), ["140", ""]
    assert owner._compare_total(ready, positions, literals) is field
    if change == "amount_bit":
        field.amounts[0] = np.nextafter(140.0, np.inf)
    elif change == "validity":
        field.validity[1] = 1
    elif change == "literal":
        literals[0] = "-1"
    else:
        literals[1] = "0"
    with pytest.raises(ValueError, match="TOTAL_"):
        owner._compare_total(ready, positions, literals)


def test_physical_seal_covers_nullable_boolean_and_money_hidden_backing():
    raw = pd.DataFrame([row(DIV_VAL="0", SUR_YN="1")])
    values = owner.CurrentAsecDividendValues(
        owner.project_dividend_literals(raw), raw, {"descriptive": True}
    )
    original = owner.dividend_values_seal(values)
    assert owner.dividend_values_seal(copy.deepcopy(values)) == original
    for name in ("DIV_VAL_amount", "survivor_property_route_clear"):
        changed = copy.deepcopy(values)
        array = changed.person[name].array
        assert array._mask[0]
        array._data[0] = not array._data[0] if name.endswith("clear") else 99.0
        assert owner.dividend_values_seal(changed) != original


def dividend_arguments(tmp_path, monkeypatch, *, fraction=Fraction(1)):
    """Amend complete invented literals and retained money before any issuance."""
    from test_us_asec_coverage_authentication import _changed_parent
    from test_us_survey_population_preparation import fixture

    from microcosm.build.frame_checkpoint import load_frame_checkpoint
    from microcosm.build.us_runtime import asec_person_income_source as restoration

    arguments = fixture(tmp_path, monkeypatch, fraction=fraction, zero=False)
    folder = arguments["source_dir"] / "asec"
    parent_path, attachment = folder / "parent.h5", folder / "household-attachment.h5"
    people = load_frame_checkpoint(parent_path).frame.person
    ids = people.person_id.to_numpy()
    literals = {
        105: row(I_DIVYN="5"),
        106: row(A_AGE="14", DIV_VAL="0", DIV_YN="0", SUR_YN="0"),
        107: row(A_AGE="70", DIV_VAL="500", SUR_YN="1", SUR_SC2="8", TDIV_VAL="1"),
        108: row(DIV_VAL="0", SUR_YN="1", SUR_SC1="10"),
    }
    changes = {}
    for name in ("DIV_VAL", "A_AGE"):
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
        for name in ("DIV_VAL", "A_AGE"):
            raw[name] = [str(int(updated.loc[key, name])) for key in raw.PERIDNUM]
        for name in (
            *owner.RECEIPT_ENTRIES,
            *owner.SURVIVOR_ENTRIES,
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
    output = tmp_path / "dividend-restored-money"
    restoration.restore_asec_person_income_source(
        parent_path, attachment, member_paths=paths, output_dir=output
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, folder / "person-income-attachment.h5"
    )
    return arguments


def prepared(tmp_path, monkeypatch, *, fraction=Fraction(1)):
    return owner.routing.source.prepare_authenticated_survey_population(
        **dividend_arguments(tmp_path, monkeypatch, fraction=fraction)
    )


def test_actual_owner_qualified_source_values_and_requalification(
    tmp_path, monkeypatch
):
    parent = prepared(tmp_path, monkeypatch)
    result = owner.qualify_current_asec_dividend(parent)
    values = result.person.set_index("native_person_id")
    assert values.loc[105, "DIV_VAL_amount"] == 140
    assert values.loc[105, "I_DIVYN_codebook_status"] == "header_reference_conflict"
    assert values.loc[106, "DIV_VAL_reporting_status"] == "outside_reporting_universe"
    assert values.loc[107, "DIV_VAL_amount"] == 500
    assert not bool(values.loc[107, "survivor_property_route_clear"])
    assert pd.isna(values.loc[108, "survivor_property_route_clear"])
    assert not values.loc[108, "DIV_VAL_amount_known"]
    assert {
        "DIV_VAL_parent_statuses",
        "DIV_VAL_parent_validity",
        "DIV_VAL_parent_zero_origin",
    } <= set(values)
    assert owner.dividend_values_seal(
        owner.qualify_current_asec_dividend(parent)
    ) == owner.dividend_values_seal(result)
    assert (
        not result.evidence["source_admission_issued"]
        and not result.evidence["survivor_amounts_read"]
    )
    assert result.evidence == json.loads(json.dumps(result.evidence))
    with pytest.raises(ValueError):
        owner.qualify_current_asec_dividend(copy.copy(parent))


@pytest.mark.parametrize(
    "coordinate", ["PERIDNUM", "PH_SEQ", "A_LINENO", "A_AGE", "DIV_VAL"]
)
def test_actual_owner_refuses_original_coordinate_or_money_change(
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
        owner.qualify_current_asec_dividend(parent)


def test_actual_owner_compares_unselected_current_year_amounts(tmp_path, monkeypatch):
    parent = prepared(tmp_path, monkeypatch, fraction=Fraction(1, 2))
    values = owner.qualify_current_asec_dividend(parent)
    omitted = set(values.asec_literals.index) - set(values.person.native_person_id)
    assert omitted
    key = values.asec_literals.loc[next(iter(omitted)), "PERIDNUM"]
    capture = owner._capture_member

    def changed(*args):
        raw = capture(*args)
        raw.loc[key, "DIV_VAL"] = "997"
        return raw

    monkeypatch.setattr(owner, "_capture_member", changed)
    with pytest.raises(ValueError, match="TOTAL_BITS"):
        owner.qualify_current_asec_dividend(parent)


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
        owner.qualify_current_asec_dividend(parent)
    assert calls


@pytest.mark.parametrize(
    "mutation", ["amount_bit", "masked_backing", "route_mask", "literal", "evidence"]
)
def test_final_physical_seal_covers_complete_returned_values(
    tmp_path, monkeypatch, mutation
):
    parent = prepared(tmp_path, monkeypatch)
    seal, calls = owner.dividend_values_seal, []

    def changed(value):
        before = seal(value)
        if not calls:
            calls.append(True)
            if mutation == "amount_bit":
                array = value.person.DIV_VAL_amount.array
                i = np.flatnonzero(~array._mask)[0]
                array._data[i] = np.nextafter(array._data[i], np.inf)
            elif mutation == "masked_backing":
                array = value.person.DIV_VAL_amount.array
                array._data[np.flatnonzero(array._mask)[0]] = 17.0
            elif mutation == "route_mask":
                array = value.person.survivor_property_route_clear.array
                array._mask[np.flatnonzero(array._mask)[0]] = False
            elif mutation == "literal":
                value.asec_literals.iloc[
                    0, value.asec_literals.columns.get_loc("SUR_SC1")
                ] = "8"
            else:
                value.evidence["survivor_amounts_read"] = True
        return before

    monkeypatch.setattr(owner, "dividend_values_seal", changed)
    with pytest.raises(ValueError, match="FINAL_VALUES_CHANGED"):
        owner.qualify_current_asec_dividend(parent)
