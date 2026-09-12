"""Retirement detail source observations; no tax or regularity model."""

import copy
import hashlib
import importlib
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def owner():
    return importlib.import_module(
        "microcosm.build.us_runtime.current_asec_retirement_detail_source"
    )


def row(owner, **changes):
    return {
        **{name: "0" for name in owner.READ_COLUMNS},
        "PERIDNUM": "0000000000000000000001",
        "PH_SEQ": "1",
        "A_LINENO": "1",
        "A_AGE": "40",
        "PEN_YN": "1",
        "PEN_SC1": "1",
        "PEN_VAL1": "100",
        "PNSN_VAL": "100",
        "DIS_YN": "1",
        "DIS_SC1": "6",
        "DIS_VAL1": "200",
        "DSAB_VAL": "200",
        "SUR_YN": "1",
        "SUR_SC1": "8",
        "SUR_VAL1": "300",
        "SRVS_VAL": "450",
        **changes,
    }


def project(owner, **changes):
    return owner.project_retirement_detail_literals(
        pd.DataFrame([row(owner, **changes)])
    ).iloc[0]


def test_source_components_and_extra_survivor_total_remain_distinct(owner):
    value = project(owner)
    assert value.PEN_VAL1_amount == 100
    assert value.DIS_VAL1_amount == 200
    assert value.SUR_VAL1_amount == 300
    assert value.SRVS_VAL_published_amount == 450
    assert value.survivor_total_minus_visible_slots == 150
    assert not value.survivor_visible_slots_exhaustive
    assert not value.taxability_assigned
    assert not value.regularity_assigned
    assert not value.acs_retirement_component_assigned


@pytest.mark.parametrize(
    "receipt,code,amount,status,known",
    [
        ("1", "1", "100", "known_receipt", True),
        ("1", "1", "0", "ambiguous_recipient_zero", False),
        ("1", "0", "0", "unreported_source_slot", False),
        ("1", "0", "100", "contradictory_source_slot", False),
        ("2", "0", "0", "known_nonreceipt", True),
        ("2", "1", "0", "contradictory_source_slot", False),
        ("2", "0", "100", "contradictory_no_nonzero", False),
        ("0", "0", "0", "niu", False),
        ("", "1", "100", "missing_receipt_literal", False),
        ("1", "", "100", "unresolved_source_slot", False),
    ],
)
@pytest.mark.parametrize("family", ["PEN", "DIS", "SUR"])
def test_slot_knownness_requires_receipt_and_source(
    owner, family, receipt, code, amount, status, known
):
    value = project(
        owner,
        **{family + "_YN": receipt, family + "_SC1": code, family + "_VAL1": amount},
    )
    assert value[family + "_VAL1_reporting_status"] == status
    assert bool(value[family + "_VAL1_amount_known"]) is known


@pytest.mark.parametrize("age", ["0", "14"])
def test_under15_has_no_analytical_zero(owner, age):
    value = project(owner, A_AGE=age, PEN_YN="0", PEN_SC1="0", PEN_VAL1="0")
    assert value.PEN_VAL1_reporting_status == "outside_reporting_universe"
    assert not value.PEN_VAL1_amount_known


@pytest.mark.parametrize(
    "token,status",
    [
        ("", "missing"),
        ("-1", "malformed"),
        ("1.2", "malformed"),
        ("1000000", "malformed"),
        ("NaN", "malformed"),
    ],
)
def test_new_amount_literal_domains_are_strict(owner, token, status):
    value = project(owner, SUR_VAL1=token)
    assert value.SUR_VAL1_literal == token
    assert value.SUR_VAL1_literal_status == status
    assert not value.SUR_VAL1_amount_known


def test_public_amount_mapping_is_detached_and_values_immutable(owner):
    entries = owner.amount_entries()
    expected = entries["SUR_VAL1"]
    entries["SUR_VAL1"] = entries["PEN_VAL1"]
    assert owner.amount_entries()["SUR_VAL1"] == expected


def test_reference_ira_annuity_and_aggregate_do_not_acquire_new_meanings(owner):
    value = project(owner, ANN_VAL="-1", DST_VAL1="100", DST_VAL2="50", DBTN_VAL="120")
    assert value.ANN_VAL_published_amount == -1
    assert value.distribution_total_minus_main_slots == -30
    assert not value.regularity_assigned and not value.taxability_assigned


def test_full_physical_seal_covers_masked_backing_and_metadata(owner):
    raw = pd.DataFrame([row(owner, SUR_VAL1="0")])
    result = owner.CurrentAsecRetirementDetailValues(
        owner.project_retirement_detail_literals(raw), raw, {}
    )
    seal = owner.retirement_detail_values_seal(result)
    assert owner.retirement_detail_values_seal(copy.deepcopy(result)) == seal
    for kind in ("masked_backing", "mask", "literal", "evidence"):
        changed = copy.deepcopy(result)
        if kind == "masked_backing":
            changed.person.SUR_VAL1_amount.array._data[0] = 17.0
        elif kind == "mask":
            changed.person.SUR_VAL1_amount.array._mask[0] = False
        elif kind == "literal":
            changed.asec_literals.loc[0, "SUR_VAL1"] = "1"
        else:
            changed.evidence["invented"] = True
        assert owner.retirement_detail_values_seal(changed) != seal


@pytest.mark.parametrize("field", ["PEN_YN", "PEN_SC1", "PEN_VAL1"])
def test_unreadable_under15_values_are_unresolved_not_contradictions(owner, field):
    value = project(owner, **{"A_AGE": "14", field: ""})
    assert value.PEN_VAL1_reporting_status == "unresolved_outside_reporting_universe"
    assert not value.PEN_VAL1_amount_known


def test_source_allocation_domains_do_not_inherit_full_header_range(owner):
    value = project(owner, I_PENSC1="4", I_PENVAL1="4", TPEN_VAL1="1", I_SURVL2="")
    assert value.I_PENSC1_literal_status == "outside_printed_range"
    assert value.I_PENVAL1_literal_status == "in_printed_range"
    assert value.I_SURVL2_literal_status == "missing"
    assert value.PEN_VAL1_amount == 100
    assert value.TPEN_VAL1_code == 1
    assert (
        next(e for e in owner.ALLOCATION_ENTRIES if e[0] == "I_SURVL2")[4]
        == "SURV_VAL2 > 0"
    )


@pytest.mark.parametrize("change", ["bound", "duplicate", "missing", "new_retained"])
def test_live_money_roster_is_exact_for_existing_fields_only(owner, change):
    entries = owner.amount_entries()
    fields = [
        SimpleNamespace(
            name=name,
            entity="person",
            grain="person",
            column=name,
            minimum=entries[name].encoded_minimum,
            maximum=entries[name].encoded_maximum,
            zero_semantics=entries[name].zero_semantics,
        )
        for name in owner.RETAINED_MONEY_FIELDS
    ]
    owner._domain_agreement(
        SimpleNamespace(bindings=SimpleNamespace(spec=SimpleNamespace(fields=fields)))
    )
    if change == "bound":
        fields[0].maximum += 1
    elif change == "duplicate":
        fields.append(fields[0])
    elif change == "missing":
        fields.pop()
    else:
        fields.append(SimpleNamespace(name="SRVS_VAL"))
    with pytest.raises(ValueError):
        owner._domain_agreement(
            SimpleNamespace(
                bindings=SimpleNamespace(spec=SimpleNamespace(fields=fields))
            )
        )


def retirement_arguments(owner, tmp_path, monkeypatch):
    from test_us_asec_coverage_authentication import _changed_parent
    from test_us_survey_population_preparation import fixture

    from microcosm.build.frame_checkpoint import load_frame_checkpoint
    from microcosm.build.us_runtime import asec_person_income_source as restoration

    arguments = fixture(tmp_path, monkeypatch)
    folder = arguments["source_dir"] / "asec"
    parent_path, attachment = folder / "parent.h5", folder / "household-attachment.h5"
    people = load_frame_checkpoint(parent_path).frame.person
    ids = people.person_id.to_numpy()
    zeros = {
        name: "0"
        for name in (
            *owner.AMOUNT_FIELDS,
            *owner.RECEIPT_ENTRIES,
            *owner.SOURCE_ENTRIES,
        )
    }
    literals = {
        105: row(owner, I_PENVAL1="4"),
        106: row(owner, **zeros, A_AGE="14"),
        107: row(
            owner,
            **{**zeros, "A_AGE": "70", "PEN_YN": "2", "DIS_YN": "2", "SUR_YN": "2"},
        ),
        108: row(owner, PEN_VAL1="0", SUR_VAL1="0"),
    }
    changes = {}
    for name in (*owner.RETAINED_MONEY_FIELDS, "A_AGE"):
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
        for name in (*owner.RETAINED_MONEY_FIELDS, "A_AGE"):
            raw[name] = [str(int(updated.loc[key, name])) for key in raw.PERIDNUM]
        for name in (
            set(owner.READ_COLUMNS)
            - set(owner.routing.COORDINATE_COLUMNS)
            - set(owner.RETAINED_MONEY_FIELDS)
        ):
            raw[name] = [
                literals.get(int(updated.loc[key, "person_id"]), row(owner))[name]
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
    output = tmp_path / "retirement-restored-money"
    restoration.restore_asec_person_income_source(
        parent_path, attachment, member_paths=paths, output_dir=output
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, folder / "person-income-attachment.h5"
    )
    return arguments


def prepared(owner, tmp_path, monkeypatch):
    return owner.routing.source.prepare_authenticated_survey_population(
        **retirement_arguments(owner, tmp_path, monkeypatch)
    )


def test_actual_owner_qualifies_new_literals_without_fabricating_money_fields(
    owner, tmp_path, monkeypatch
):
    parent = prepared(owner, tmp_path, monkeypatch)
    compare, compared = owner._compare_amount, []

    def seen(ready, positions, name, literals):
        compared.append(name)
        assert name not in owner.INDEPENDENT_LITERAL_FIELDS
        return compare(ready, positions, name, literals)

    monkeypatch.setattr(owner, "_compare_amount", seen)
    result = owner.qualify_current_asec_retirement_detail(parent)
    assert tuple(compared) == owner.RETAINED_MONEY_FIELDS
    values = result.person.set_index("native_person_id")
    assert values.loc[105, "PEN_VAL1_amount"] == 100
    assert values.loc[105, "survivor_total_minus_visible_slots"] == 150
    assert values.loc[107, "SUR_VAL1_amount"] == 0
    assert not values.loc[106, "SUR_VAL1_amount_known"]
    assert not values.loc[108, "SUR_VAL1_amount_known"]
    assert not result.evidence["source_admission_issued"]
    assert not result.evidence["taxability_assigned"]
    assert owner.retirement_detail_values_seal(
        owner.qualify_current_asec_retirement_detail(parent)
    ) == owner.retirement_detail_values_seal(result)
    with pytest.raises(ValueError):
        owner.qualify_current_asec_retirement_detail(copy.copy(parent))


@pytest.mark.parametrize("field", ["PERIDNUM", "A_AGE", "DIS_VAL1"])
def test_actual_owner_requires_retained_identity_and_money(
    owner, tmp_path, monkeypatch, field
):
    parent = prepared(owner, tmp_path, monkeypatch)
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
        owner.qualify_current_asec_retirement_detail(parent)


def test_actual_owner_pins_new_survivor_dollars_to_original_member(
    owner, tmp_path, monkeypatch
):
    parent = prepared(owner, tmp_path, monkeypatch)
    capture = owner._capture_member

    def changed(root, pin):
        # Only the invented fixture is touched. Even an unretained money field
        # must match the exact already admitted source-member bytes.
        path = root / "asec" / pin[1]
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        raw["SRVS_VAL"] = "999"
        raw.to_csv(path, index=False)
        return capture(root, pin)

    monkeypatch.setattr(owner, "_capture_member", changed)
    with pytest.raises(ValueError):
        owner.qualify_current_asec_retirement_detail(parent)


@pytest.mark.parametrize(
    "mutation", ["amount_bit", "masked_backing", "literal", "evidence"]
)
def test_actual_owner_final_seal_covers_returned_values(
    owner, tmp_path, monkeypatch, mutation
):
    parent = prepared(owner, tmp_path, monkeypatch)
    seal, calls = owner.retirement_detail_values_seal, []

    def changed(value):
        before = seal(value)
        if not calls:
            calls.append(True)
            array = value.person.SUR_VAL1_amount.array
            if mutation == "amount_bit":
                i = np.flatnonzero(~array._mask)[0]
                array._data[i] = np.nextafter(array._data[i], np.inf)
            elif mutation == "masked_backing":
                array._data[np.flatnonzero(array._mask)[0]] = 17.0
            elif mutation == "literal":
                value.asec_literals.iloc[
                    0, value.asec_literals.columns.get_loc("SUR_YN")
                ] = "2"
            else:
                value.evidence["taxability_assigned"] = True
        return before

    monkeypatch.setattr(owner, "retirement_detail_values_seal", changed)
    with pytest.raises(ValueError, match="FINAL_VALUES_CHANGED"):
        owner.qualify_current_asec_retirement_detail(parent)


def test_actual_owner_requalifies_parent_after_last_capture(
    owner, tmp_path, monkeypatch
):
    parent = prepared(owner, tmp_path, monkeypatch)
    capture, calls = owner._capture_member, []

    def changed(*args):
        raw = capture(*args)
        table = parent._checked()[2].frame.person
        table.iloc[0, table.columns.get_loc("age")] += 1
        calls.append(True)
        return raw

    monkeypatch.setattr(owner, "_capture_member", changed)
    with pytest.raises(ValueError):
        owner.qualify_current_asec_retirement_detail(parent)
    assert calls
