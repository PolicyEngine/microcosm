"""Invented SPM-unit amount evidence; pure fixtures confer no source authority."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_acs_spm_native_construction import requires_assembler
from test_us_current_survey_spm_source import project, pure_fixture

from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import current_survey_spm_amount_source as amounts
from microcosm.build.us_runtime import current_survey_spm_source as spm

FIELDS = ("SPM_ENGVAL", "SPM_CHILDCAREXPNS")


def _field(name, values, *, statuses=None, validity=None, zero_origin=None):
    values = np.asarray(values, dtype="<f8")
    return money.MoneyField(
        name,
        values.tobytes(),
        np.asarray(
            [int(money.CodebookStatus.AMOUNT_NONZERO)] * len(values)
            if statuses is None
            else statuses,
            dtype="u1",
        ).tobytes(),
        np.asarray(
            [1] * len(values) if validity is None else validity, dtype="u1"
        ).tobytes(),
        np.asarray(
            [0] * len(values) if zero_origin is None else zero_origin, dtype="u1"
        ).tobytes(),
    )


def invented_projection_arguments(*, unresolved=False, acs=True):
    """Use existing literal SPM fixture; money positions deliberately differ."""
    frame, origins, keys, selected, rosters = pure_fixture()
    raw, _, evidence = project(
        (frame, origins, keys, selected, rosters),
        None if unresolved else spm.ASEC_2025_INCOME_2024_SPM_POLICY,
    )
    origins["native_person_id"] = np.array([101, 102], dtype="int64")
    origins["source_year"] = 2024
    origins["survey_year"] = 2025
    frame.spm_unit = pd.DataFrame({"spm_unit_id": [100]}, dtype="int64")
    if acs:
        frame.person = pd.concat(
            [
                frame.person,
                pd.DataFrame(
                    {
                        "person_id": [12],
                        "person_spm_unit_id": [101],
                        "person_household_id": [201],
                        "A_AGE": [33],
                    }
                ),
            ],
            ignore_index=True,
        )
        frame.spm_unit = pd.DataFrame({"spm_unit_id": [100, 101]}, dtype="int64")
        origins.loc[12] = ["acs", 1103, 2024, 2024]
        origins["native_person_id"] = origins.native_person_id.astype("int64")
        evidence.append(
            {
                "spm_unit_id": 101,
                "source": "acs",
                "status": "INCLUDED",
                "reason": "invented_acs_scope",
                "source_spm_unit_id": "acs-unit",
            }
        )
    units = pd.DataFrame(evidence).set_index("spm_unit_id")
    native = raw.PERIDNUM.tolist()
    # Year2023 repeats a literal native key; it must never supply the2024 amount.
    scope = money.AsecMoneyScope(
        person_ids=(901, 102, 101),
        household_ids=(80, 7),
        person_household_ids=(80, 7, 7),
        person_spm_ids=(900, 700, 700),
        person_years=(2023, 2024, 2024),
        household_years=(2023, 2024),
        person_native_keys=(native[0], native[1], native[0]),
        household_native_keys=("00080", "00007"),
    )
    fields = {
        "SPM_ENGVAL": _field("SPM_ENGVAL", [9999, 180, 180]),
        "SPM_CHILDCAREXPNS": _field(
            "SPM_CHILDCAREXPNS", [9999, 0, 0], statuses=[1, 3, 3], zero_origin=[0, 1, 1]
        ),
    }
    return [frame, origins, raw, units, scope, fields]


def _run(args):
    return amounts.project_spm_amounts(*args)


@pytest.mark.parametrize("axis", ["amount", "status", "validity", "zero_origin"])
def test_replicated_unit_amount_evidence_disagreement_refuses(axis):
    args = invented_projection_arguments()
    field = args[-1]["SPM_ENGVAL"]
    attribute, dtype = {
        "amount": ("amount_bytes", "<f8"),
        "status": ("status_bytes", "u1"),
        "validity": ("validity_bytes", "u1"),
        "zero_origin": ("zero_origin_bytes", "u1"),
    }[axis]
    changed = np.frombuffer(getattr(field, attribute), dtype=dtype).copy()
    changed[2] = {"amount": 181, "status": 3, "validity": 0, "zero_origin": 1}[axis]
    args[-1][field.name] = replace(field, **{attribute: changed.tobytes()})
    with pytest.raises(ValueError):
        _run(args)


@pytest.mark.parametrize(
    "mutation",
    [
        "native_id",
        "raw_key",
        "split_unit",
        "unit_literal",
        "member_count",
        "duplicate_origin",
        "source_year",
    ],
)
def test_current_native_keys_and_complete_original_units_are_required(mutation):
    args = invented_projection_arguments()
    frame, origins, raw, units, _, _ = args
    if mutation == "native_id":
        origins.loc[10, "native_person_id"] = 999
    elif mutation == "raw_key":
        raw.loc[10, "PERIDNUM"] = "0" * 21 + "9"
    elif mutation == "split_unit":
        frame.person.loc[frame.person.person_id.eq(11), "person_spm_unit_id"] = 101
    elif mutation == "unit_literal":
        raw.loc[11, "SPM_ID"] = "7"  # Literal identity must retain leading zeroes.
    elif mutation == "member_count":
        raw.loc[11, "SPM_NUMPER"] = "3"
    elif mutation == "duplicate_origin":
        origins.loc[11, "native_person_id"] = origins.loc[10, "native_person_id"]
    else:
        origins.loc[10, "source_year"] = 2023
    with pytest.raises(ValueError):
        _run(args)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "copied_name", "wrong_length"]
)
def test_exact_requested_money_field_contract_refuses(mutation):
    args = invented_projection_arguments()
    fields = args[-1]
    if mutation == "missing":
        fields.pop("SPM_ENGVAL")
    elif mutation == "extra":
        fields["WSAL_VAL"] = _field("WSAL_VAL", [1, 1, 1])
    elif mutation == "copied_name":
        fields["SPM_ENGVAL"] = replace(fields["SPM_ENGVAL"], name="WSAL_VAL")
    else:
        fields["SPM_ENGVAL"] = replace(
            fields["SPM_ENGVAL"], amount_bytes=np.array([1, 1], dtype="<f8").tobytes()
        )
    with pytest.raises(ValueError):
        _run(args)


def test_detached_inputs_cannot_enter_genuine_borrower():
    with pytest.raises(ValueError):
        amounts.qualify_current_spm_amounts(object(), SimpleNamespace())


def test_pure_projection_never_mutates_supplied_axes_or_evidence():
    args = invented_projection_arguments()
    before = [
        args[0].person.copy(deep=True),
        args[0].spm_unit.copy(deep=True),
        args[1].copy(deep=True),
        args[2].copy(deep=True),
        args[3].copy(deep=True),
    ]
    scope, fields = args[4], dict(args[5])
    result = _run(args)
    for expected, actual in zip(
        before,
        [args[0].person, args[0].spm_unit, args[1], args[2], args[3]],
        strict=True,
    ):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert args[4] is scope and args[5] == fields
    assert result.person is not args[0].person and result.units is not args[3]


def _genuine_spm_pair(tmp_path, monkeypatch):
    from test_us_acs_spm_native_construction import (
        OPTIONS,
        require_acs_spm_source_capability,
    )
    from test_us_current_survey_spm_source import _real_source_arguments

    require_acs_spm_source_capability(OPTIONS)
    arguments = _real_source_arguments(tmp_path, monkeypatch)
    preparation = spm.source.prepare_authenticated_survey_population(**arguments)
    spm_inputs = spm.qualify_current_survey_spm(
        preparation,
        acs_profile=spm.acs.ACSAnalysisProfile(2024, "acs_2024_1yr", True, True),
        asec_scope_policy=spm.ASEC_2025_INCOME_2024_SPM_POLICY,
    )
    return arguments, preparation, spm_inputs


@requires_assembler
def test_genuine_borrow_rejects_copies_cross_owner_and_source_mutation(
    tmp_path, monkeypatch
):
    """One genuine setup; destructive member mutation is deliberately last."""
    arguments, preparation, spm_inputs = _genuine_spm_pair(tmp_path, monkeypatch)
    result = amounts.qualify_current_spm_amounts(preparation, spm_inputs)
    assert len(result.person) == len(spm_inputs.source_frame.person)
    for a, b in [
        (copy.copy(preparation), spm_inputs),
        (preparation, copy.copy(spm_inputs)),
    ]:
        with pytest.raises(ValueError):
            amounts.qualify_current_spm_amounts(a, b)
    # A fresh genuine preparation over identical files is still a distinct owner.
    other = spm.source.prepare_authenticated_survey_population(**arguments)
    with pytest.raises(ValueError):
        amounts.qualify_current_spm_amounts(other, spm_inputs)
    native = spm_inputs.asec_raw
    old = native.iloc[0]["SPM_ID"]
    native.iloc[0, native.columns.get_loc("SPM_ID")] = "mutated"
    with pytest.raises(ValueError):
        amounts.qualify_current_spm_amounts(preparation, spm_inputs)
    native.iloc[0, native.columns.get_loc("SPM_ID")] = old
    restored = amounts.qualify_current_spm_amounts(preparation, spm_inputs)
    pd.testing.assert_frame_equal(result.person, restored.person, check_exact=True)
    pd.testing.assert_frame_equal(result.units, restored.units, check_exact=True)
    member = arguments["source_dir"] / "asec" / "pppub25.csv"
    member.write_bytes(member.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        amounts.qualify_current_spm_amounts(preparation, spm_inputs)


def test_exact_current_key_join_preserves_unit_grain_and_frozen_zero_origin():
    result = _run(invented_projection_arguments())
    assert result.person.index.tolist() == [10, 11, 12]
    assert result.person.index.name == "person_id"
    assert result.person.money_position.tolist() == [2, 1, -1]
    assert result.person.SPM_ENGVAL_amount.iloc[:2].tolist() == [180.0, 180.0]
    assert result.person.SPM_ENGVAL_status.iloc[:2].tolist() == [1, 1]
    assert result.person.SPM_ENGVAL_validity.iloc[:2].tolist() == [1, 1]
    assert result.person.SPM_CHILDCAREXPNS_amount.iloc[:2].tolist() == [0.0, 0.0]
    assert result.person.SPM_CHILDCAREXPNS_status.iloc[:2].tolist() == [3, 3]
    assert result.person.SPM_CHILDCAREXPNS_zero_origin.iloc[:2].tolist() == [1, 1]
    assert result.units.index.tolist() == [100, 101]
    assert result.units.loc[100, "source_spm_unit_id"] == "000007"
    assert result.units.loc[100, "member_count"] == 2
    assert (
        result.units.loc[100, "SPM_ENGVAL_amount"] == 180
    )  # Never sum repeated unit copies.
    assert result.units.loc[100, "SPM_ENGVAL_known"]
    assert result.units.loc[100, "SPM_ENGVAL_reason"] == "known_nonzero"
    assert not result.units.loc[100, "SPM_CHILDCAREXPNS_known"]
    assert result.units.loc[100, "SPM_CHILDCAREXPNS_reason"] == "zero_origin_unresolved"
    assert result.units.loc[100, "SPM_CHILDCAREXPNS_amount"] == 0
    for field in FIELDS:
        assert np.isnan(result.units.loc[101, field + "_amount"])
        assert not result.units.loc[101, field + "_known"]
        assert result.units.loc[101, field + "_reason"] == "survey_amount_unobserved"
        assert result.person.loc[12, field + "_validity"] == -1
    assert result.evidence["source_admission_issued"] is False
    assert result.evidence["release_eligible"] is False
    assert result.evidence["zero_origin_upgraded"] is False
    assert result.evidence["childcare_measure"] == "reported_uncapped_expense"
    assert result.evidence["pre_subsidy_amount_observed"] is False


def test_money_position_permutation_and_raw_row_order_do_not_change_values():
    args = invented_projection_arguments()
    expected = _run(args)
    scope = args[4]
    order = np.array([2, 0, 1])
    args[4] = replace(
        scope,
        **{
            field: tuple(getattr(scope, field)[int(i)] for i in order)
            for field in (
                "person_ids",
                "person_household_ids",
                "person_spm_ids",
                "person_years",
                "person_native_keys",
            )
        },
    )
    args[5] = {
        name: replace(
            field,
            **{
                attr: np.frombuffer(getattr(field, attr), dtype=dtype)[order].tobytes()
                for attr, dtype in (
                    ("amount_bytes", "<f8"),
                    ("status_bytes", "u1"),
                    ("validity_bytes", "u1"),
                    ("zero_origin_bytes", "u1"),
                )
            },
        )
        for name, field in args[5].items()
    }
    args[2] = args[2].iloc[::-1].copy()
    actual = _run(args)
    pd.testing.assert_frame_equal(actual.units, expected.units, check_exact=True)
    pd.testing.assert_frame_equal(
        actual.person.drop(columns="money_position"),
        expected.person.drop(columns="money_position"),
        check_exact=True,
    )
    assert actual.person.money_position.tolist() == [0, 2, -1]


def test_missing_one_money_member_keeps_complete_unit_unknown():
    args = invented_projection_arguments()
    scope = args[4]
    keep = [0, 2]  # Remove current native102; old-year duplicate cannot fill the gap.
    args[4] = replace(
        scope,
        **{
            field: tuple(getattr(scope, field)[i] for i in keep)
            for field in (
                "person_ids",
                "person_household_ids",
                "person_spm_ids",
                "person_years",
                "person_native_keys",
            )
        },
    )
    args[5] = {
        name: replace(
            field,
            **{
                attr: np.frombuffer(getattr(field, attr), dtype=dtype)[keep].tobytes()
                for attr, dtype in (
                    ("amount_bytes", "<f8"),
                    ("status_bytes", "u1"),
                    ("validity_bytes", "u1"),
                    ("zero_origin_bytes", "u1"),
                )
            },
        )
        for name, field in args[5].items()
    }
    result = _run(args)
    assert result.person.loc[10, "SPM_ENGVAL_amount"] == 180
    assert result.person.loc[11, "money_position"] == -1
    assert result.units.loc[100, "missing_money_members"] == 1
    for field in FIELDS:
        assert np.isnan(result.units.loc[100, field + "_amount"])
        assert result.units.loc[100, field + "_validity"] == -1
        assert not result.units.loc[100, field + "_known"]
        assert result.units.loc[100, field + "_reason"] == "missing_money_member"


def test_valid_values_do_not_qualify_unresolved_annual_scope():
    result = _run(invented_projection_arguments(unresolved=True))
    assert result.units.loc[100, "scope_status"] == "UNRESOLVED"
    assert result.units.loc[100, "SPM_ENGVAL_amount"] == 180
    assert result.units.loc[100, "SPM_ENGVAL_validity"] == 1
    assert not result.units.loc[100, "SPM_ENGVAL_known"]
    assert result.units.loc[100, "SPM_ENGVAL_reason"] == "annual_scope_unresolved"


@pytest.mark.parametrize("case", ["known_zero", "missing_amount"])
def test_known_zero_and_missing_amount_evidence_remain_distinct(case):
    args = invented_projection_arguments()
    if case == "known_zero":
        args[-1]["SPM_ENGVAL"] = _field(
            "SPM_ENGVAL", [9999, 0, 0], statuses=[1, 3, 3], zero_origin=[0, 2, 2]
        )
    else:
        args[-1]["SPM_ENGVAL"] = _field(
            "SPM_ENGVAL", [9999, 0, 0], statuses=[1, 0, 0], validity=[1, 0, 0]
        )
    result = _run(args)
    assert result.units.loc[100, "SPM_ENGVAL_reason"] == case
    assert bool(result.units.loc[100, "SPM_ENGVAL_known"]) is (case == "known_zero")
    assert result.units.loc[100, "SPM_ENGVAL_validity"] == (
        1 if case == "known_zero" else 0
    )
    assert result.units.loc[100, "SPM_ENGVAL_amount"] == 0
    assert not np.signbit(result.units.loc[100, "SPM_ENGVAL_amount"])


@pytest.mark.parametrize("value", [np.nan, -0.0])
@pytest.mark.parametrize("valid", [0, 1])
def test_noncanonical_missing_and_zero_amount_bytes_refuse(value, valid):
    args = invented_projection_arguments()
    args[-1]["SPM_ENGVAL"] = _field(
        "SPM_ENGVAL",
        [9999, value, value],
        statuses=[1, 3 if valid else 0, 3 if valid else 0],
        validity=[1, valid, valid],
        zero_origin=[0, 1 if valid else 0, 1 if valid else 0],
    )
    with pytest.raises(ValueError, match="FIELD_EVIDENCE"):
        _run(args)


@pytest.mark.parametrize(
    "change",
    [
        "money_alias",
        "routing_alias",
        "spm_alias",
        "zero_origin",
        "domain_pin",
        "year",
        "unit_status",
    ],
)
def test_borrower_refuses_changed_imported_semantics_before_projection(
    monkeypatch, change
):
    if change.endswith("_alias"):
        name = change.removesuffix("_alias")
        replacement = SimpleNamespace(**vars(getattr(amounts, name)))
        if name == "money":
            # These swapped labels previously upgraded genuine frozen-origin1
            # to known_zero while leaving the original owners unchanged.
            replacement.ZeroOrigin = SimpleNamespace(
                NOT_ZERO=0, FROZEN_FILLNA_UNRESOLVED=2, AUTHENTICATED_CENSUS_ENCODED=1
            )
        monkeypatch.setattr(amounts, name, replacement)
    elif change == "zero_origin":
        monkeypatch.setattr(
            money,
            "ZeroOrigin",
            SimpleNamespace(
                NOT_ZERO=0, FROZEN_FILLNA_UNRESOLVED=2, AUTHENTICATED_CENSUS_ENCODED=1
            ),
        )
    elif change == "domain_pin":
        monkeypatch.setattr(
            money, "RESOURCE_PINS", ("changed", *money.RESOURCE_PINS[1:])
        )
    elif change == "year":
        monkeypatch.setattr(amounts.routing, "CURRENT_INCOME_YEAR", 2023)
    else:
        monkeypatch.setattr(spm, "INCLUDED", "UNRESOLVED")
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        amounts.qualify_current_spm_amounts(object(), object())


@pytest.mark.parametrize("historical_amount", [12.34567, -123456789.125, 1e15 + 0.5])
def test_historical_restated_amounts_do_not_enter_current_year_domain(
    historical_amount,
):
    args = invented_projection_arguments()
    expected = _run(args)
    for name, field in args[-1].items():
        values = field.amounts.copy()
        values[0] = historical_amount  # 2023; price restatement may be fractional.
        args[-1][name] = replace(field, amount_bytes=values.astype("<f8").tobytes())
    actual = _run(args)
    pd.testing.assert_frame_equal(actual.person, expected.person, check_exact=True)
    pd.testing.assert_frame_equal(actual.units, expected.units, check_exact=True)
