"""Invented numerical and literal-reader controls; no source issuer substitute."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from test_us_asec_coverage_authentication import _changed_parent
from test_us_survey_population_preparation import fixture

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime import asec_current_money_source as money
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import current_social_security_source as source
from microcosm.build.us_runtime import survey_social_security as ss


@pytest.mark.parametrize(
    ("reason", "component"), [(1, 0), (2, 1), (3, 3), (4, 2), (5, 3), (6, 2)]
)
def test_unique_reported_component_receives_source_total(reason, component):
    total = np.asarray([1200.0])
    first, second = np.asarray([float(reason)]), np.asarray([0.0])
    values, allowed, origin = ss.asec_reason_basis(total, first, second)
    expected = np.zeros((1, 4))
    expected[0, component] = 1200
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(allowed, expected > 0)
    assert origin == ("source_reason_total_allocation",)
    np.testing.assert_array_equal(total, [1200])
    np.testing.assert_array_equal(first, [reason])


def test_two_reasons_in_same_component_do_not_duplicate_total():
    values, allowed, _ = ss.asec_reason_basis(
        np.array([300.0]), np.array([3.0]), np.array([5.0])
    )
    np.testing.assert_array_equal(values, [[0, 0, 0, 300]])
    np.testing.assert_array_equal(allowed, [[False, False, False, True]])


@pytest.mark.parametrize(
    ("first", "second", "allowed"),
    [
        (1.0, 2.0, [True, True, False, False]),
        (7.0, 0.0, [False, True, True, True]),
        (8.0, 0.0, [True, True, True, True]),
        (np.nan, 1.0, [True, True, True, True]),
        (0.0, 0.0, [True, True, True, True]),
    ],
)
def test_ambiguous_positive_total_has_no_fabricated_component(first, second, allowed):
    values, support, origin = ss.asec_reason_basis(
        np.array([100.0]), np.array([first]), np.array([second])
    )
    assert np.isnan(values).all()
    np.testing.assert_array_equal(support, [allowed])
    assert origin[0].startswith("unresolved_")


def test_known_zero_does_not_need_a_component_reason():
    values, support, origin = ss.asec_reason_basis(
        np.array([0.0]), np.array([np.nan]), np.array([np.nan])
    )
    np.testing.assert_array_equal(values, np.zeros((1, 4)))
    assert not support.any()
    assert origin == ("known_total_zero",)


def test_asec_niu_zero_remains_unknown_but_reporting_zero_is_known():
    total = np.array([0.0, 0.0, 1200.0])
    age = np.array([14.0, 15.0, 70.0])
    receipt = np.array([0.0, 2.0, 1.0])
    first, second = np.array([0.0, 0.0, 1.0]), np.zeros(3)
    inputs = [a.copy() for a in (total, age, receipt, first, second)]
    amount, basis, allowed, labels = ss.asec_reporting_basis(
        total, age, receipt, first, second
    )
    assert np.isnan(amount[0]) and np.isnan(basis[0]).all() and allowed[0].all()
    assert labels[0] == "asec_below15_outside_reporting_universe"
    np.testing.assert_array_equal(amount[1:], [0.0, 1200.0])
    np.testing.assert_array_equal(basis[1:], [[0, 0, 0, 0], [1200, 0, 0, 0]])
    for original, current in zip(
        inputs, (total, age, receipt, first, second), strict=True
    ):
        np.testing.assert_array_equal(original, current)


@pytest.mark.parametrize(
    ("amount", "age", "receipt", "error"),
    [
        (100.0, 14.0, 1.0, "OUTSIDE_REPORTING_UNIVERSE_OBSERVATION"),
        (0.0, 14.0, 2.0, "OUTSIDE_REPORTING_UNIVERSE_OBSERVATION"),
        (0.0, 15.0, 0.0, "IN_UNIVERSE_RECIPIENCY_UNKNOWN"),
        (0.0, 15.0, np.nan, "IN_UNIVERSE_RECIPIENCY_UNKNOWN"),
        (100.0, 30.0, 2.0, "RECIPIENCY_CONTRADICTION"),
        (0.0, 15.5, 2.0, "AGE_DOMAIN"),
        (0.0, 100.0, 2.0, "AGE_DOMAIN"),
        (0.0, 30.0, 3.0, "RECIPIENCY_DOMAIN"),
    ],
)
def test_reporting_universe_refuses_contradictions(amount, age, receipt, error):
    with pytest.raises(ValueError, match=error):
        ss.asec_reporting_basis(
            np.array([amount]),
            np.array([age]),
            np.array([receipt]),
            np.array([0.0]),
            np.array([0.0]),
        )


@pytest.mark.parametrize("value", [-1.0, 9.0, 1.5, np.inf])
def test_invalid_reason_is_not_age_fallback(value):
    with pytest.raises(
        ValueError, match="SURVEY_SOCIAL_SECURITY_(REASON_DOMAIN|NONFINITE|NEGATIVE)"
    ):
        ss.asec_reason_basis(np.array([100.0]), np.array([value]), np.array([0.0]))


def test_completion_preserves_known_basis_and_source_totals():
    total = np.array([120.0, 300.0, 0.0])
    source_basis = np.array([[120.0, 0, 0, 0], [np.nan] * 4, [0.0] * 4])
    allowed = np.array(
        [[True, False, False, False], [False, True, True, False], [False] * 4]
    )
    scores = np.array([[np.nan] * 4, [1000.0, 1, 2, 500], [np.nan] * 4])
    originals = [x.copy() for x in (total, source_basis, allowed, scores)]
    got = ss.complete_positive_basis(total, source_basis, allowed, scores)
    np.testing.assert_array_equal(got, [[120, 0, 0, 0], [0, 100, 200, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(got.sum(axis=1), total)
    for original, current in zip(
        originals, (total, source_basis, allowed, scores), strict=True
    ):
        np.testing.assert_array_equal(original, current)


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ([0.0, 0, 0, 0], "COMPLETION_NO_ALLOWED_MASS"),
        ([5.0, 0, 0, 0], "COMPLETION_NO_ALLOWED_MASS"),
        ([np.nan, 1, 1, 0], "COMPLETION_UNKNOWN"),
        ([0.0, -1, 1, 0], "NEGATIVE"),
    ],
)
def test_completion_refuses_invalid_modeled_basis(scores, message):
    with pytest.raises(ValueError, match=message):
        ss.complete_positive_basis(
            np.array([100.0]),
            np.full((1, 4), np.nan),
            np.array([[False, True, True, False]]),
            np.asarray([scores]),
        )


def test_completion_cannot_overwrite_a_reported_basis():
    with pytest.raises(ValueError, match="MODELED_SOURCE_OVERWRITE"):
        ss.complete_positive_basis(
            np.array([100.0]),
            np.array([[100.0, 0, 0, 0]]),
            np.array([[True, False, False, False]]),
            np.ones((1, 4)),
        )


def _literal_row(**changes):
    return {
        "PERIDNUM": "0000000000000000000001",
        "PH_SEQ": "1",
        "A_LINENO": "1",
        "A_AGE": "70",
        "SS_VAL": "12000",
        "SS_YN": "1",
        "RESNSS1": "1",
        "RESNSS2": "0",
        "RESNSSA": "0",
        "I_SSVAL": "0",
        "I_SSYN": "0",
        **changes,
    }


def _write_literals(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source.READ_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_literal_reader_preserves_missing_reason_and_allocation_provenance(tmp_path):
    path = tmp_path / "invented.csv"
    _write_literals(path, [_literal_row(RESNSS1="", I_SSVAL="15", RESNSSA="9")])
    result = source._read_capture(path, rows=1)
    assert result.iloc[0].RESNSS1 == ""
    assert result.iloc[0].PERIDNUM == "0000000000000000000001"
    assert (
        source._allocation_labels(result).iloc[0].allocation_origin
        == "publisher_allocated"
    )


@pytest.mark.parametrize("token", ["9", "-1", "1.0", "1e0", " 1", "١", 1])
def test_invalid_nonempty_code_is_never_silently_missing(token):
    with pytest.raises(ValueError, match="CURRENT_SOCIAL_SECURITY_CODE_"):
        source._codes([token], set(range(9)))


def test_blank_code_stays_unknown_separately_from_valid_niu():
    values = source._codes(["", "0", "8"], set(range(9)))
    assert np.isnan(values[0])
    np.testing.assert_array_equal(values[1:], [0, 8])


@pytest.mark.parametrize("field,token", [("I_SSVAL", "10"), ("I_SSYN", "9")])
def test_out_of_domain_allocation_code_refuses(field, token):
    with pytest.raises(ValueError, match="CODE_DOMAIN"):
        source._allocation_labels(pd.DataFrame([_literal_row(**{field: token})]))


@pytest.mark.parametrize(
    "changes",
    [
        {"RESNSS1": "12"},
        {"SS_VAL": "1e3"},
        {"PH_SEQ": ""},
        {"PERIDNUM": "1"},
        {"I_SSVAL": "-1"},
    ],
)
def test_literal_reader_refuses_noncanonical_source_tokens(tmp_path, changes):
    path = tmp_path / "invented.csv"
    _write_literals(path, [_literal_row(**changes)])
    with pytest.raises(ValueError, match="CURRENT_SOCIAL_SECURITY_"):
        source._read_capture(path, rows=1)


def test_literal_reader_refuses_duplicate_person_keys(tmp_path):
    path = tmp_path / "invented.csv"
    _write_literals(path, [_literal_row(), _literal_row(A_LINENO="2")])
    with pytest.raises(ValueError, match="COORDINATE_DUPLICATE"):
        source._read_capture(path, rows=2)


def test_a_projection_receipt_cannot_replace_live_source_authority():
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        source.qualify_current_social_security({"source_authenticated": True})


def _social_security_arguments(tmp_path, monkeypatch, *, ambiguous):
    """Construct original SS observations before fresh maintained issuance.

    The raw CSV and original checkpoint are fixture inputs, not mutations of a
    received Population. The maintained helper rebuilds identities and source
    pins, then the real person-income attachment and preparation are rebuilt.
    No source issuer, validation method, ready artifact, or model is replaced.
    """
    arguments = fixture(
        tmp_path,
        monkeypatch,
        zero=False,
        acs_ssp_values={("2024HU0000001", 1): 1200},
    )
    root = arguments["source_dir"] / "asec"
    parent_path, household_path = root / "parent.h5", root / "household-attachment.h5"
    parent = load_frame_checkpoint(parent_path).frame.person
    values = parent.SS_VAL.to_numpy(dtype=np.float64, copy=True)
    for pid, value in ((105, 12000.0), (106, 0.0), (107, 3000.0), (108, 0.0)):
        positions = np.flatnonzero(parent.person_id.to_numpy() == pid)
        assert len(positions) == 1
        values[positions[0]] = value
    _changed_parent(parent_path, household_path, monkeypatch, {"SS_VAL": values})
    updated = load_frame_checkpoint(parent_path).frame.person.set_index("PERIDNUM")
    members, pins = {}, []
    for year, member, archive, *_ in source.coverage._MEMBER_PINS:
        path = root / f"pppub{year - 1999}.csv"
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        amounts = np.asarray(
            [updated.loc[k, "SS_VAL"] for k in raw.PERIDNUM], dtype=np.float64
        )
        assert np.isfinite(amounts).all() and (amounts == np.floor(amounts)).all()
        raw["SS_VAL"] = [str(int(v)) for v in amounts]
        raw["SS_YN"] = [
            "1" if value > 0 else "0" if int(age) < 15 else "2"
            for value, age in zip(amounts, raw.A_AGE, strict=True)
        ]
        raw["RESNSS1"] = ["1" if v > 0 else "0" for v in amounts]
        raw["RESNSS2"] = "0"
        raw["RESNSSA"] = "0"
        raw["I_SSVAL"] = "0"
        raw["I_SSYN"] = "0"
        if year == 2024:
            raw.loc[raw.PERIDNUM.eq(str(5).zfill(22)), "RESNSS1"] = "2"
            k = raw.PERIDNUM.eq(str(7).zfill(22))
            raw.loc[k, "RESNSS1"] = "7" if ambiguous else "3"
            raw.loc[k, "RESNSSA"] = "9"
            raw.loc[k, "I_SSVAL"] = "15"
        # Prove the join by original identity rather than source row order.
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
        members[year] = path
    for owner in (source.coverage, restoration):
        monkeypatch.setattr(owner, "_MEMBER_PINS", tuple(pins))
    output = tmp_path / "ss-restored-money"
    restoration.restore_asec_person_income_source(
        parent_path,
        household_path,
        member_paths=members,
        output_dir=output,
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, root / "person-income-attachment.h5"
    )
    # Ensure the real source pin update is retained by the original owner.
    assert money._SOURCE_PINS[0] == hashlib.sha256(parent_path.read_bytes()).hexdigest()
    return arguments


@pytest.mark.parametrize("ambiguous", [False, True])
def test_actual_current_source_qualifier_preserves_totals_reasons_and_unknownness(
    tmp_path, monkeypatch, ambiguous
):
    arguments = _social_security_arguments(tmp_path, monkeypatch, ambiguous=ambiguous)
    prepared = source.source.prepare_authenticated_survey_population(**arguments)
    before = prepared.checked_view().frame
    originals = {e: before.table(e).copy(deep=True) for e in before.entities}
    result = source.qualify_current_social_security(prepared)
    asec = result.person.loc[result.person.source.eq("asec")].set_index(
        "native_person_id"
    )
    assert asec.loc[105, "social_security_source_total"] == 12000.0
    np.testing.assert_array_equal(
        asec.loc[105, list(ss.COMPONENTS)].to_numpy(dtype=np.float64), [0, 12000, 0, 0]
    )
    assert asec.loc[107, "social_security_source_total"] == 3000.0
    assert asec.loc[107, "allocation_origin"] == "publisher_allocated"
    if ambiguous:
        assert asec.loc[107, list(ss.COMPONENTS)].isna().all()
        assert asec.loc[107, "basis_origin"] == "unresolved_component_split"
        assert not asec.loc[107, "allowed_social_security_retirement"]
    else:
        assert asec.loc[107, "social_security_survivors"] == 3000.0
    acs = result.person.loc[result.person.source.eq("acs")]
    positive = acs.social_security_source_total.gt(0)
    assert positive.any()
    assert acs.loc[positive, list(ss.COMPONENTS)].isna().all().all()
    assert (
        acs.loc[positive, "basis_origin"]
        .eq("acs_combined_positive_requires_model")
        .all()
    )
    assert (
        not result.evidence["source_admission_issued"]
        and not result.evidence["release_eligible"]
    )
    assert (
        result.evidence["asec_income_year"]
        == result.evidence["price_basis_year"]
        == 2024
    )
    assert result.evidence["asec_interview_year"] == 2025
    assert result.evidence["acs_survey_year"] == 2024
    assert len(result.evidence["acs_native_sha256"]) == 64
    assert not result.evidence["individual_beneficiary_assignment_claim"]
    assert asec.source_reporting_unit.eq(
        "person_report_may_combine_family_payments"
    ).all()
    fresh = source.qualify_current_social_security(prepared)
    pd.testing.assert_frame_equal(fresh.person, result.person, check_exact=True)
    assert fresh.evidence == result.evidence
    result.person.loc[:, "social_security_source_total"] = -1
    result.evidence["dictionary"]["sha256"] = "0" * 64
    assert (
        source.qualify_current_social_security(prepared).evidence["dictionary"][
            "sha256"
        ]
        == source.DICTIONARY["sha256"]
        != "0" * 64
    )
    assert (
        source.qualify_current_social_security(
            prepared
        ).person.social_security_source_total.dropna()
        >= 0
    ).all()
    for entity, original in originals.items():
        pd.testing.assert_frame_equal(before.table(entity), original, check_exact=True)


@pytest.fixture(scope="module")
def full_ss_preparation(tmp_path_factory):
    root = tmp_path_factory.mktemp("full-ss-source")
    with pytest.MonkeyPatch.context() as patch:
        arguments = _social_security_arguments(root, patch, ambiguous=True)
        request_path = arguments["source_dir"] / "selection-request.json"
        request = json.loads(request_path.read_bytes())
        request["fraction"] = [2, 3]
        request_path.write_text(
            json.dumps(request, sort_keys=True, separators=(",", ":"))
        )
        arguments["fraction"] = Fraction(2, 3)
        preparation = source.source.prepare_authenticated_survey_population(**arguments)
        yield preparation, arguments


def test_full_ss_retains_original_design_donors_omitted_by_selection(
    full_ss_preparation,
):
    preparation, _ = full_ss_preparation
    selected = source.qualify_current_social_security(preparation)
    result = source.qualify_full_current_social_security(preparation)
    pd.testing.assert_frame_equal(
        result.selected.person, selected.person, check_exact=True
    )
    pd.testing.assert_frame_equal(result.selected.asec_literals, selected.asec_literals)
    assert result.selected.evidence == selected.evidence
    assert tuple(result.asec_basis.index) == (105, 106, 107, 108)
    assert result.asec_basis.index.equals(
        pd.Index(result.asec_frame.person.person_id, name="person_id")
    )
    receiving = set(
        selected.person.loc[selected.person.source.eq("asec"), "native_person_id"]
    )
    assert len(receiving) == 2
    assert set(result.asec_basis.index) - receiving
    # Seed41's selected household omits105, a positive resolved report eligible
    # for the proposed category donor basis, not merely an ambiguous report.
    assert 105 not in receiving
    assert result.asec_frame.weights_for("household").kind.value == "design"
    np.testing.assert_array_equal(
        result.asec_frame.weights_for("household").values, [2552.12, 100.0]
    )
    basis = result.asec_basis
    assert basis.loc[105, "social_security_disability"] == 12000
    assert basis.loc[105, "basis_origin"] == "source_reason_total_allocation"
    assert basis.loc[105, "source_reporting_universe"]
    assert basis.loc[105, "allocation_origin"] == "publisher_no_allocation"
    assert basis.loc[107, list(ss.COMPONENTS)].isna().all()
    assert not basis.loc[107, "allowed_social_security_retirement"]
    assert basis.loc[[106, 108], "social_security_source_total"].isna().all()
    assert basis.loc[[106, 108], list(ss.COMPONENTS)].isna().all().all()
    assert basis.loc[[106, 108], "SS_VAL"].eq("0").all()
    assert basis.loc[107, "I_SSVAL"] == "15"
    assert basis.loc[107, "reason_allocation_code"] == 9
    assert basis.loc[107, "allocation_origin"] == "publisher_allocated"
    assert not result.evidence["source_admission_issued"]
    assert not result.evidence["release_eligible"]
    assert result.evidence["donor_eligibility_selected"] is False
    before = source.full_social_security_seal(result)
    result.asec_basis.loc[105, "social_security_disability"] = 0
    assert source.full_social_security_seal(result) != before
    fresh = source.qualify_full_current_social_security(preparation)
    assert fresh.asec_basis.loc[105, "social_security_disability"] == 12000


@pytest.mark.parametrize("change", ["mapping", "mapping_type", "dictionary", "columns"])
def test_full_ss_rejects_changed_mapping_configuration(
    full_ss_preparation, monkeypatch, change
):
    preparation, _ = full_ss_preparation
    if change == "mapping":
        monkeypatch.setitem(ss.REASON_COMPONENTS, 1, (1,))
    elif change == "mapping_type":

        class ChangedMapping(dict):
            def __getitem__(self, key):
                return (0,)

        monkeypatch.setattr(
            ss, "REASON_COMPONENTS", ChangedMapping(ss.REASON_COMPONENTS)
        )
    elif change == "dictionary":
        monkeypatch.setitem(source.DICTIONARY, "sha256", "0" * 64)
    else:
        monkeypatch.setattr(source, "READ_COLUMNS", source.READ_COLUMNS[::-1])
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        source.qualify_full_current_social_security(preparation)


def test_full_ss_rechecks_detached_result_after_final_owner_io(full_ss_preparation):
    preparation, _ = full_ss_preparation
    # Use this before destructive source tests when selected directly. A new
    # genuine preparation is supplied by the module fixture for this test run.
    changed = []
    previous = sys.getprofile()

    def late_change(frame, event, value):
        if previous is not None:
            previous(frame, event, value)
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code
            is source.source.AuthenticatedSurveyPopulationPreparation._checked.__code__
            and caller is not None
            and caller.f_code is source._qualify_current_social_security.__code__
            and "result" in caller.f_locals
        ):
            caller.f_locals["result"].asec_basis.loc[
                105, "social_security_disability"
            ] = 99.0
            changed.append(True)

    try:
        sys.setprofile(late_change)
        with pytest.raises(ValueError, match="FINAL_OWNER_OR_VALUES"):
            source.qualify_full_current_social_security(preparation)
    finally:
        sys.setprofile(previous)
    assert changed == [True]


@pytest.mark.parametrize(
    "column,old,new", [("RESNSS1", "7", "8"), ("I_SSVAL", "15", "14")]
)
def test_full_ss_refuses_changed_literal_source(
    full_ss_preparation, tmp_path, column, old, new
):
    _, arguments = full_ss_preparation
    # Each destructive literal check gets fresh genuine custody. Restoring file
    # bytes does not restore ctime and must not revive the preceding owner.
    preparation = source.source.prepare_authenticated_survey_population(
        **{**arguments, "snapshot_root": tmp_path}
    )
    path = arguments["source_dir"] / "asec/pppub25.csv"
    original = path.read_bytes()
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert raw[column].eq(old).any()
    raw.loc[raw[column].eq(old), column] = new
    try:
        raw.to_csv(path, index=False)
        with pytest.raises(ValueError):
            source.qualify_full_current_social_security(preparation)
    finally:
        path.write_bytes(original)


def test_full_ss_does_not_accept_detached_projection_or_receipt():
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        source.qualify_full_current_social_security({"source_authenticated": True})
