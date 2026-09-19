"""Full-source donor controls on tiny genuine issuers over invented bytes."""

import hashlib
import json
import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_immigration_donor as owner
from microcosm.frame import WeightKind


def source_arguments(tmp_path, monkeypatch, *, defect=None):
    import test_us_current_survey_immigration_source as fixture

    from microcosm.build.us_runtime import asec_person_income_source as restoration
    from microcosm.build.us_runtime import native_household_origin as origin

    full, partial, donor = fixture.source_arguments(tmp_path, monkeypatch)
    root = full["source_dir"] / "asec"
    pins, paths = [], {}
    for year, member, archive, *_ in owner.literals.original.asec._MEMBER_PINS:
        path = root / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        raw["PENATVTY"] = "057"
        raw["A_MARITL"] = "7"
        raw["PEAFEVER"] = raw.A_AGE.map(lambda age: "2" if int(age) >= 17 else "-1")
        for name in ("MCARE", "CAID", "IHSFLG", "CHAMPVA", "MIL"):
            raw[name] = "2"
        raw["A_SEX"] = raw.A_LINENO.map({"1": "1", "2": "2"})
        raw["AXSEX"] = raw.A_LINENO.map({"1": "0", "2": "4"})
        raw["A_EXPRRP"] = raw.A_LINENO.map({"1": "1", "2": "5"})
        raw["P_SEQ"] = raw.A_LINENO
        if year == 2024:
            selected = raw.PERIDNUM.eq(str(donor - 100).zfill(22))
            assert selected.sum() == 1
            if defect == "blank-owning-token":
                raw.loc[selected, "MCARE"] = ""
            elif defect == "unresolved-owning-code":
                raw.loc[selected, "PENATVTY"] = "-4"
            elif defect == "missing-person":
                raw = raw.loc[~selected]
            elif defect == "duplicate-person":
                raw = pd.concat([raw, raw.loc[selected]], ignore_index=True)
        raw.iloc[::-1].to_csv(path, index=False)
        data = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(data).hexdigest(),
                len(raw),
                len(data),
            )
        )
        paths[year] = path
        shutil.copyfile(path, partial["source_dir"] / "asec" / member)
    for module in (
        owner.literals.original.asec,
        restoration,
        owner.demographics.demographic,
    ):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = tmp_path / "full-donor-restoration"
    restoration.restore_asec_person_income_source(
        root / "parent.h5",
        root / "household-attachment.h5",
        member_paths=paths,
        output_dir=output,
    )
    for args in (full, partial):
        shutil.copyfile(
            output / restoration.CHECKPOINT_FILENAME,
            args["source_dir"] / "asec" / "person-income-attachment.h5",
        )
    path = root / "hhpub25.csv"
    households = pd.read_csv(path, dtype=str, keep_default_na=False)
    households["GESTFIPS"] = households.H_SEQ.map({"00007": "06", "00008": "36"})
    if defect == "unresolved-household-weight":
        source_household = "00007" if donor == 105 else "00008"
        households.loc[households.H_SEQ.eq(source_household), "HSUP_WGT"] = ""
    households.iloc[::-1].to_csv(path, index=False)
    data = path.read_bytes()
    monkeypatch.setattr(
        origin,
        "_ASEC_MEMBER_PINS",
        tuple(
            replace(
                pin,
                member_sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            )
            for pin in origin._ASEC_MEMBER_PINS
        ),
    )
    shutil.copyfile(path, partial["source_dir"] / "asec" / "hhpub25.csv")
    return full, partial, donor


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        full_args, partial_args, donor = source_arguments(
            tmp_path_factory.mktemp("full-asec-donor"), patch
        )
        full = owner.source.prepare_authenticated_survey_population(**full_args)
        partial = owner.source.prepare_authenticated_survey_population(**partial_args)
        originals = [
            (prepared, owner.source._frame_identity(prepared._checked()[2].frame))
            for prepared in (full, partial)
        ]
        left = owner.borrow_full_asec_immigration_donor(full)
        right = owner.borrow_full_asec_immigration_donor(partial)
        yield SimpleNamespace(
            full=full, partial=partial, left=left, right=right, unselected=donor
        )
        for prepared, seal in originals:
            assert owner.source._frame_identity(prepared._checked()[2].frame) == seal


def test_full_donor_is_selection_invariant_and_exactly_design_weighted(actual):
    left, right = actual.left, actual.right
    left.validate()
    right.validate()
    assert left.frame.n("person") == 4 and left.frame.n("household") == 2
    for entity in left.frame.entities:
        pd.testing.assert_frame_equal(
            left.frame.table(entity), right.frame.table(entity)
        )
    pd.testing.assert_frame_equal(left.raw, right.raw)
    pd.testing.assert_frame_equal(left.households, right.households)
    assert actual.unselected in right.frame.person.person_id.tolist()
    selected_asec = actual.partial._checked()[2].native[1].frame.person.person_id
    assert actual.unselected not in selected_asec.tolist()
    assert right.frame.weights_for("household").kind is WeightKind.DESIGN
    assert right.frame.weights_for("household").values.tolist() == [2552.12, 100.0]
    assert [
        Fraction(r.fraction_numerator, r.fraction_denominator)
        for r in right.households.itertuples()
    ] == [Fraction(63803, 25), Fraction(100)]
    assert right.households.HSUP_WGT.tolist() == ["000255212", "000010000"]
    assert right.frame.table("household").state_fips.tolist() == [6, 36]
    assert right.frame.person.is_female.tolist() == [False, True, False, True]
    assert np.array_equal(right.frame.person.age, right.frame.person.A_AGE)
    assert right.raw.A_LFSR.tolist().count(" 7 ") == 1
    assert right.frame.person.A_LFSR.tolist().count(7) == 1
    assert right.frame.person.PEAFEVER.tolist().count(-1) == 3
    for frame in (left.frame, right.frame):
        assert not {
            "WSAL_VAL",
            "SEMP_VAL",
            "employment_income",
            "person_weight",
            "ssn_card_type",
            "immigration_status_str",
        } & set(frame.person)
    receipt = json.loads(right.receipt)
    assert receipt["observation_year"] == 2025 and receipt["income_year"] == 2024
    assert receipt["weight_source"] == "original_HSUP_WGT/100"
    assert not receipt["national_stock_alignment_qualified"]
    assert not receipt["status_assignment_performed"]


def raw_row():
    values = {name: "0" for name in owner.literals.ASEC_COLUMNS}
    values.update(
        PH_SEQ="00007",
        PERIDNUM="0000000000000000000001",
        A_LINENO="1",
        PRCITSHP="1",
        PENATVTY="057",
        A_MARITL="7",
        PEAFEVER="-1",
        SPM_CAPHOUSESUB="0.00",
    )
    return pd.DataFrame([values], columns=owner.literals.ASEC_COLUMNS)


@pytest.mark.parametrize("field", owner.literals.ASEC_VALUE_COLUMNS)
@pytest.mark.parametrize("token", ["", " ", "not-a-number"])
def test_every_owning_field_refuses_blank_or_malformed_before_coercion(field, token):
    raw = raw_row()
    raw.loc[0, field] = token
    original = raw.copy(deep=True)
    with pytest.raises(ValueError, match="MISSING_TOKEN|MALFORMED_TOKEN"):
        owner._numeric_literals(raw)
    pd.testing.assert_frame_equal(raw, original)


@pytest.mark.parametrize(
    "field,token",
    [
        ("PRCITSHP", "-4"),
        ("PRCITSHP", "0"),
        ("PENATVTY", "999"),
        ("PENATVTY", "-1"),
        ("PEINUSYR", "29"),
        ("A_AGE", "81"),
        ("A_MARITL", "0"),
        ("A_SPOUSE", "17"),
        ("A_HSCOL", "3"),
        ("A_LFSR", "5"),
        ("MCARE", "3"),
        ("PEN_SC1", "9"),
        ("RESNSS2", "9"),
        ("PEIO1COW", "9"),
        ("PEIO1COW", "-4"),
        ("A_MJOCC", "-1"),
        ("PEAFEVER", "0"),
        ("SPM_CAPHOUSESUB", "100000"),
        ("SPM_CAPHOUSESUB", "0.01"),
    ],
)
def test_unexplained_or_out_of_domain_source_codes_refuse(field, token):
    raw = raw_row()
    raw.loc[0, field] = token
    with pytest.raises(ValueError, match="UNRESOLVED_DOMAIN"):
        owner._numeric_literals(raw)


@pytest.mark.parametrize(
    "field,token,expected",
    [
        ("PEINUSYR", "27", 27),
        ("PEINUSYR", "28", 28),
        ("PEAFEVER", "-1", -1),
        ("A_LFSR", " 7 ", 7),
        ("A_AGE", "80", 80),
        ("A_AGE", "85", 85),
        ("SPM_CAPHOUSESUB", "99999.00", 99999),
    ],
)
def test_legitimate_niu_recent_arrival_and_topcoded_tokens_survive(
    field, token, expected
):
    raw = raw_row()
    raw.loc[0, field] = token
    assert owner._numeric_literals(raw).loc[0, field] == expected
    assert raw.loc[0, field] == token


@pytest.mark.parametrize("field", ["raw", "households", "frame", "receipt"])
def test_copies_or_changed_borrowed_output_cannot_gain_authority(actual, field):
    value = actual.right
    previous = getattr(value, field)
    replacement = (
        previous.copy(deep=True)
        if isinstance(previous, pd.DataFrame)
        else (b"changed" if field == "receipt" else actual.left.frame)
    )
    try:
        object.__setattr__(value, field, replacement)
        with pytest.raises(ValueError, match="DONOR_PROJECTION_CHANGED"):
            value.validate()
    finally:
        object.__setattr__(value, field, previous)
    value.validate()
    with pytest.raises(ValueError, match="ISSUED_OWNER_REQUIRED"):
        replace(value).validate()


@pytest.mark.parametrize("surface", ["raw", "households", "frame"])
def test_in_place_output_mutation_refuses(actual, surface):
    value = actual.right
    table, column = (
        (value.raw, "MCARE")
        if surface == "raw"
        else (value.households, "numerator")
        if surface == "households"
        else (value.frame.person, "A_LFSR")
    )
    index = table.index[0]
    before = table.at[index, column]
    try:
        table.at[index, column] = "changed" if surface == "raw" else 999
        with pytest.raises(ValueError, match="DONOR_PROJECTION_CHANGED"):
            value.validate()
    finally:
        table.at[index, column] = before
    value.validate()


@pytest.mark.parametrize("surface", ["raw", "frame", "owner", "implementation"])
def test_final_foreign_io_mutation_is_refused(actual, monkeypatch, surface):
    value = actual.right
    native = owner._ISSUED[id(value)][2].native
    raw_index = value.raw.index[0]
    raw_before = value.raw.at[raw_index, "MCARE"]
    age_before = value.frame.person.A_AGE.copy(deep=True)
    payload_before = native.payload
    reader = Path.read_bytes
    changed = []

    def late(path):
        result = reader(path)
        if not changed and str(path) == owner.__file__:
            changed.append(True)
            if surface == "raw":
                value.raw.at[raw_index, "MCARE"] = "changed"
            elif surface == "frame":
                value.frame.person.loc[:, "A_AGE"] += 1
            elif surface == "owner":
                object.__setattr__(native, "payload", b"changed")
            else:
                monkeypatch.setattr(owner, "_numeric_literals", lambda raw: raw)
        return result

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", late)
            with pytest.raises(ValueError):
                value.validate()
        assert changed
    finally:
        value.raw.at[raw_index, "MCARE"] = raw_before
        value.frame.person.loc[:, "A_AGE"] = age_before
        object.__setattr__(native, "payload", payload_before)
    # Outer monkeypatch restores implementation at teardown.
    if surface != "implementation":
        value.validate()


@pytest.mark.parametrize(
    "defect",
    [
        "blank-owning-token",
        "unresolved-owning-code",
        "missing-person",
        "duplicate-person",
        "unresolved-household-weight",
    ],
)
def test_genuine_issuer_refuses_incomplete_or_unresolved_original_source(
    tmp_path, monkeypatch, defect
):
    with pytest.raises(ValueError):
        _, partial_args, _ = source_arguments(tmp_path, monkeypatch, defect=defect)
        prepared = owner.source.prepare_authenticated_survey_population(**partial_args)
        if defect in {"blank-owning-token", "unresolved-owning-code"}:
            literal_owner = owner.literals.qualify_current_survey_immigration(prepared)
            literal_owner.validate()
            column, token = (
                ("MCARE", "") if defect == "blank-owning-token" else ("PENATVTY", "-4")
            )
            assert token in literal_owner.asec_full_raw[column].tolist()
        owner.borrow_full_asec_immigration_donor(prepared)


def test_changed_original_anchor_is_not_replaced_by_current_projection(actual):
    value = actual.right
    anchors = owner._ISSUED[id(value)][2].native_entry[2].anchors
    previous = anchors.payload
    try:
        object.__setattr__(anchors, "payload", previous + b"changed")
        with pytest.raises(ValueError):
            value.validate()
    finally:
        object.__setattr__(anchors, "payload", previous)
    value.validate()


def test_independent_seal_refuses_mutated_qualifier_closure_and_literal(actual):
    value = actual.right
    qualified = owner._ISSUED[id(value)][2].qualified
    index = qualified.asec_full_raw.index[0]
    before = qualified.asec_full_raw.at[index, "MCARE"]
    # The retained donor seal is outside the older literal owner's callback.
    cells = dict(
        zip(
            qualified._revalidate.__code__.co_freevars,
            qualified._revalidate.__closure__,
            strict=True,
        )
    )
    previous = cells["seals"].cell_contents
    try:
        qualified.asec_full_raw.at[index, "MCARE"] = "1"
        tables = cells["tables"].cell_contents
        cells["seals"].cell_contents = tuple(
            owner.literals._table_seal(table) for table in tables
        )
        with pytest.raises(ValueError, match="SOURCE_PROJECTION_CHANGED"):
            value.validate()
    finally:
        qualified.asec_full_raw.at[index, "MCARE"] = before
        cells["seals"].cell_contents = previous
    value.validate()
