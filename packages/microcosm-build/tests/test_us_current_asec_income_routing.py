"""Actual source readers over constructed, privately pinned invented bytes.

No native microdata, no country engine and no full PUF donor fixture: the
current ASEC income routing qualifier only needs the bounded survey preparation
fixture plus invented member literals.
"""

import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_us_asec_coverage_authentication import _changed_parent
from test_us_survey_population_preparation import fixture

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime import asec_coverage_authentication as coverage
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import current_asec_income_routing_source as owner

# Invented current-year people. Ages 55/14 come from the shared fixture; the
# older adult exercises the printed age-58 distribution route.
CURRENT_PEOPLE = (105, 106, 107, 108)
AGES = {105: 55, 106: 14, 107: 70, 108: 14}
AMOUNTS = {
    #            PNSN   ANN   DSTV1 DSTV1Y DSTV2 DSTV2Y   RNT    FRSE    OI
    105: (12000.0, -1.0, 0.0, 7000.0, 0.0, 0.0, -400.0, 0.0, 3000.0),
    106: (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    107: (9000.0, 0.0, 5000.0, 0.0, 0.0, 0.0, 500.0, -9000.0, 250.0),
    108: (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 700.0),
}
AMOUNT_ORDER = (
    "PNSN_VAL",
    "ANN_VAL",
    "DST_VAL1",
    "DST_VAL1_YNG",
    "DST_VAL2",
    "DST_VAL2_YNG",
    "RNT_VAL",
    "FRSE_VAL",
    "OI_VAL",
)
LITERALS = {
    105: {
        "PEN_YN": "1",
        "ANN_YN": "0",
        "DST_YN": "0",
        "DST_YN_YNG": "1",
        "DST_SC1_YNG": "4",
        "RNT_YN": "1",
        "FRSE_YN": "1",
        "ERN_YN": "1",
        "OI_YN": "1",
        "OI_OFF": "20",
    },
    106: {},
    107: {
        "PEN_YN": "2",
        "ANN_YN": "1",
        "DST_YN": "1",
        "DST_SC1": "9",
        "RNT_YN": "0",
        "FRSE_YN": "1",
        "ERN_YN": "1",
        "OI_YN": "1",
        "OI_OFF": "0",
    },
    108: {"OI_YN": "0", "ERN_YN": "", "FRMOTR": ""},
}
DEFAULT_LITERAL = {
    **{name: "0" for name in owner.RECEIPT_ENTRIES},
    **{name: "0" for name in owner.ACCOUNT_ENTRIES},
    "OI_OFF": "0",
    **{name: "0" for name in owner.ALLOCATION_ENTRIES},
}


def _member_literals(person_id):
    return {**DEFAULT_LITERAL, **LITERALS.get(person_id, {})}


def routing_arguments(tmp_path, monkeypatch, *, reverse=True, ages=None):
    """Extend the invented member before any owner issues a preparation.

    Only fixture registry pins change. No source issuer, ready() or native
    loader is replaced, and the person-income attachment is rebuilt from the
    changed member bytes exactly as production would.
    """
    arguments = fixture(tmp_path, monkeypatch)
    asec = arguments["source_dir"] / "asec"
    parent_path, attachment = asec / "parent.h5", asec / "household-attachment.h5"
    person = load_frame_checkpoint(parent_path).frame.person
    ids = person.person_id.to_numpy()
    ages = AGES if ages is None else ages
    changes = {}
    for offset, field in enumerate(AMOUNT_ORDER):
        data = person[field].to_numpy(dtype="float64", copy=True)
        for person_id in CURRENT_PEOPLE:
            positions = np.flatnonzero(ids == person_id)
            assert len(positions) == 1
            data[positions[0]] = AMOUNTS[person_id][offset]
        changes[field] = data
    age_column = person.A_AGE.to_numpy(copy=True)
    for person_id in CURRENT_PEOPLE:
        age_column[np.flatnonzero(ids == person_id)[0]] = ages[person_id]
    changes["A_AGE"] = age_column
    _changed_parent(parent_path, attachment, monkeypatch, changes)
    updated = load_frame_checkpoint(parent_path).frame.person.set_index("PERIDNUM")
    paths, pins = {}, []
    for year, member, archive, *_ in coverage._MEMBER_PINS:
        path = asec / f"pppub{year - 1999}.csv"
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        for field in (*AMOUNT_ORDER, "A_AGE"):
            raw[field] = [str(int(updated.loc[key, field])) for key in raw.PERIDNUM]
        keys = [int(key) for key in raw.PERIDNUM]
        current = dict(zip(keys, updated.loc[raw.PERIDNUM].person_id, strict=True))
        for name in DEFAULT_LITERAL:
            raw[name] = [_member_literals(int(current[key]))[name] for key in keys]
        assert raw.notna().all().all()
        ordered = raw.iloc[::-1] if reverse else raw
        ordered.to_csv(path, index=False)
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
    for module in (coverage, restoration):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = tmp_path / "routing-restored-money"
    restoration.restore_asec_person_income_source(
        parent_path, attachment, member_paths=paths, output_dir=output
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, asec / "person-income-attachment.h5"
    )
    return arguments


def _qualified(tmp_path, monkeypatch, **kwargs):
    arguments = routing_arguments(tmp_path, monkeypatch, **kwargs)
    prepared = owner.source.prepare_authenticated_survey_population(**arguments)
    return prepared, owner.qualify_current_asec_income_routing(prepared)


def test_actual_invented_member_joins_the_real_preparation_by_native_keys(
    tmp_path, monkeypatch
):
    prepared, qualified = _qualified(tmp_path, monkeypatch)
    person = qualified.person.set_index("native_person_id")
    assert set(person.index) == set(CURRENT_PEOPLE)
    # Borrowed money, native identity and member literal agree row by row.
    assert person.loc[105, "pension_annuity_pension_source_total"] == 12000.0
    assert person.loc[105, "pension_annuity_pension_reporting_status"] == (
        "known_receipt"
    )
    assert person.loc[105, "pension_annuity_pension_known_amount"] == 12000.0
    assert person.source_age.to_dict() == {k: float(v) for k, v in AGES.items()}
    literals = qualified.asec_literals.set_index("PERIDNUM")
    for person_id in CURRENT_PEOPLE:
        key = str(person_id - 100).zfill(22)
        assert int(literals.loc[key, "PNSN_VAL"]) == AMOUNTS[person_id][0]
        assert int(literals.loc[key, "A_AGE"]) == AGES[person_id]
    assert person.amount_validity_PNSN_VAL.eq(1).all()
    assert (
        hashlib.sha256(qualified.person.to_json(orient="table").encode()).hexdigest()
        == qualified.evidence["projection_sha256"]
    )
    assert not qualified.evidence["source_admission_issued"]
    assert not qualified.evidence["release_eligible"]
    # A host records this receipt, so it must round-trip as JSON unchanged.
    receipt = json.dumps(qualified.evidence, sort_keys=True)
    assert json.loads(receipt) == qualified.evidence
    assert qualified.evidence["joined_person_years"] == [2024]
    assert qualified.evidence["acs_channel_rows"] > 0
    assert not qualified.evidence["acs_components_modeled"]
    assert set(qualified.evidence["families"]) == set(owner.FAMILIES)
    assert (
        qualified.evidence["dictionary"]["amount_entries_source"]["sha256"]
        == owner.money.RESOURCE_PINS[0]
    )
    assert qualified.evidence["declared_niu_normalized_to_zero"] == ["ANN_VAL"]
    printed = qualified.evidence["dictionary"]
    assert set(printed["printed_universe_questions"]) == {
        "DST_VAL1",
        "DST_SC2_YNG",
        "I_DSTVAL1COMP",
        "DST_YN",
    }
    assert set(printed["printed_scope_and_code_questions"]) == {
        "RNT_YN",
        "RNT_VAL",
        "FRSE_VAL",
        "PNSN_VAL",
        "OI_YN",
        "DST_SC1",
        "FRSE_YN",
    }
    assert set(printed["ambiguous_allocation_flag_coverage"]) == set(
        owner.AMBIGUOUS_FLAG_COVERAGE
    )
    prepared.checked_view()


def test_pension_and_annuity_totals_stay_separate_and_unsplit(tmp_path, monkeypatch):
    _, qualified = _qualified(tmp_path, monkeypatch)
    person = qualified.person.set_index("native_person_id")
    # PNSN_VAL is the printed combined total over all pension sources. Neither
    # a private share nor a taxable share is derived from it.
    assert person.loc[105, "pension_annuity_combined_total_scope"] == (
        owner.PENSION_TOTAL_SCOPE
    )
    assert not person.pension_annuity_private_share_applied.any()
    assert not person.pension_annuity_taxable_amount_known.any()
    assert not person.pension_annuity_pension_total_has_published_flag.any()
    # ANN_VAL's printed -1 is a NIU code, never a one dollar annuity loss.
    assert np.isnan(person.loc[105, "pension_annuity_annuity_source_total"])
    assert person.loc[105, "pension_annuity_annuity_amount_kind"] == "declared_niu"
    assert person.loc[105, "pension_annuity_annuity_reporting_status"] == "niu"
    # ANN_VAL is the one entry here whose printed zero is valid dollars (its NIU
    # is the separate -1 code), so a yes answer with a zero amount resolves.
    assert person.loc[107, "pension_annuity_annuity_reporting_status"] == (
        "known_recipient_zero"
    )
    assert person.loc[107, "pension_annuity_annuity_known_amount"] == 0.0
    # PNSN_VAL prints "0 = none or niu", so the same pattern would not resolve
    # there; the module reads that distinction from the pinned domains artifact.
    assert owner._zero_is_dollars("ANN_VAL")
    assert not owner._zero_is_dollars("PNSN_VAL")
    # A no answer against a positive total is a retained contradiction.
    assert person.loc[107, "pension_annuity_pension_reporting_status"] == (
        "contradictory_no_nonzero"
    )
    assert person.loc[107, "pension_annuity_pension_source_total"] == 9000.0
    assert np.isnan(person.loc[107, "pension_annuity_pension_known_amount"])


def test_distribution_routing_preserves_slots_ages_and_unresolved_accounts(
    tmp_path, monkeypatch
):
    _, qualified = _qualified(tmp_path, monkeypatch)
    person = qualified.person.set_index("native_person_id")
    assert person.loc[105, "retirement_distribution_route"] == "under_age58"
    assert person.loc[107, "retirement_distribution_route"] == "age58_and_over"
    # Regular IRA is account code 4; the slot amount is not a taxable amount.
    assert person.loc[105, "retirement_distribution_slot1_young_account_code"] == 4
    assert person.loc[105, "retirement_distribution_slot1_young_account_label"] == (
        "Regular IRA"
    )
    assert person.loc[105, "retirement_distribution_regular_ira_amount"] == 7000.0
    assert person.loc[105, "retirement_distribution_regular_ira_slots"] == 1
    assert person.loc[105, "retirement_distribution_source_total"] == 7000.0
    assert not person.retirement_distribution_taxable_amount_known.any()
    assert bool(person.loc[107, "retirement_distribution_route_has_published_flag"])
    assert not bool(person.loc[105, "retirement_distribution_route_has_published_flag"])
    # An account literal outside the printed range leaves the composition and
    # the regular IRA share unresolved rather than defaulting either way.
    assert person.loc[107, "retirement_distribution_slot1_account_literal"] == "9"
    assert person.loc[107, "retirement_distribution_slot1_account_literal_status"] == (
        "outside_printed_range"
    )
    assert person.loc[107, "retirement_distribution_slot1_slot_status"] == (
        "unresolved_slot_account"
    )
    assert pd.isna(person.loc[107, "retirement_distribution_slot1_account_code"])
    assert np.isnan(person.loc[107, "retirement_distribution_regular_ira_amount"])
    assert pd.isna(person.loc[107, "retirement_distribution_regular_ira_slots"])
    # The slot total is still the retained literal sum for the applicable route.
    assert person.loc[107, "retirement_distribution_source_total"] == 5000.0
    # The printed distribution universes name only the age 58 split, so
    # coverage below age 15 is a source question, not a resolved answer.
    assert pd.isna(person.loc[106, "retirement_distribution_source_reporting_universe"])
    assert person.loc[106, "retirement_distribution_reporting_status"] == (
        "unresolved_reporting_universe"
    )


def test_signed_property_and_farm_totals_keep_losses_and_net_zero_receipt(
    tmp_path, monkeypatch
):
    _, qualified = _qualified(tmp_path, monkeypatch)
    person = qualified.person.set_index("native_person_id")
    assert person.loc[105, "net_property_source_total"] == -400.0
    assert person.loc[105, "net_property_known_amount"] == -400.0
    assert person.loc[105, "net_property_reporting_status"] == "known_receipt"
    assert bool(person.loc[105, "net_property_is_net_loss"])
    # The receipt question is wider than the amount question, so the total is
    # not independently labelled rental and no component split is claimed.
    assert person.loc[105, "net_property_receipt_scope"] == (
        owner.NET_PROPERTY_RECEIPT_SCOPE
    )
    assert person.loc[105, "net_property_amount_scope"] == (
        owner.NET_PROPERTY_AMOUNT_SCOPE
    )
    assert not person.net_property_component_split_known.any()
    # A NIU receipt literal against a positive amount stays a contradiction.
    assert person.loc[107, "net_property_reporting_status"] == (
        "contradictory_niu_nonzero"
    )
    assert person.loc[107, "net_property_source_total"] == 500.0
    assert np.isnan(person.loc[107, "net_property_known_amount"])
    # Farm receipt with a net zero stays distinct from absence, NIU and missing,
    # but FRSE_VAL prints "0 = none or niu" just as the gross entries do, so the
    # amount is not completed to a known zero.
    assert person.loc[105, "farm_reporting_status"] == "receipt_with_net_zero"
    assert np.isnan(person.loc[105, "farm_known_amount"])
    assert person.loc[105, "farm_source_total"] == 0.0
    assert "receipt_with_net_zero" not in owner.KNOWN_AMOUNT_STATUSES
    assert person.loc[107, "farm_source_total"] == -9000.0
    assert person.loc[107, "farm_reporting_status"] == "known_receipt"
    assert bool(person.loc[107, "farm_is_net_loss"])
    assert not person.farm_is_nonfarm_self_employment.any()
    # The farm universe comes from ERN_YN/FRMOTR evidence, never from age.
    assert bool(person.loc[105, "farm_source_reporting_universe"])
    assert pd.isna(person.loc[108, "farm_source_reporting_universe"])
    assert person.loc[108, "farm_reporting_status"] == "unresolved_reporting_universe"


def test_other_income_keeps_reported_alimony_apart_from_residual_rules(
    tmp_path, monkeypatch
):
    _, qualified = _qualified(tmp_path, monkeypatch)
    person = qualified.person.set_index("native_person_id")
    assert person.loc[105, "other_income_category_code"] == 20
    assert person.loc[105, "other_income_category_label"] == "alimony"
    assert person.loc[105, "other_income_routing_status"] == "reported_category"
    assert bool(person.loc[105, "other_income_is_reported_alimony"])
    assert person.loc[105, "other_income_source_total"] == 3000.0
    # A receipt without a category is retained as an unresolved pattern; no
    # residual rule assigns it to alimony or to miscellaneous income.
    assert person.loc[107, "other_income_routing_status"] == "receipt_without_category"
    assert not bool(person.loc[107, "other_income_is_reported_alimony"])
    assert not person.other_income_residual_rule_applied.any()
    # A NIU receipt with a positive amount is a contradiction, not a nonfiler.
    assert person.loc[108, "other_income_reporting_status"] == (
        "contradictory_outside_reporting_universe"
    )
    assert person.loc[108, "other_income_source_total"] == 700.0
    assert np.isnan(person.loc[108, "other_income_known_amount"])


def test_row_order_of_the_source_member_cannot_change_the_projection(
    tmp_path, monkeypatch
):
    roots = {}
    for name in ("forward", "reverse"):
        roots[name] = tmp_path / name
        roots[name].mkdir()
    with pytest.MonkeyPatch.context() as forward:
        _, ordered = _qualified(roots["forward"], forward, reverse=False)
    with pytest.MonkeyPatch.context() as backward:
        _, reversed_ = _qualified(roots["reverse"], backward, reverse=True)
    pd.testing.assert_frame_equal(ordered.person, reversed_.person, check_exact=True)
    assert (
        ordered.evidence["projection_sha256"] == reversed_.evidence["projection_sha256"]
    )
    # The member digest legitimately differs; the projection must not.
    assert (
        ordered.evidence["source_member_sha256"]
        != reversed_.evidence["source_member_sha256"]
    )


def test_repeated_qualification_is_stable_and_transport_is_not_authority(
    tmp_path, monkeypatch
):
    prepared, qualified = _qualified(tmp_path, monkeypatch)
    again = owner.qualify_current_asec_income_routing(prepared)
    assert (
        again.evidence["projection_sha256"] == qualified.evidence["projection_sha256"]
    )
    # Mutating the returned transport cannot reach a fresh live projection.
    qualified.person.loc[qualified.person.index[0], "farm_source_total"] = 99.0
    qualified.asec_literals.loc[qualified.asec_literals.index[0], "OI_OFF"] = "99"
    fresh = owner.qualify_current_asec_income_routing(prepared)
    assert not fresh.person.farm_source_total.eq(99.0).any()
    assert not fresh.asec_literals.OI_OFF.eq("99").any()
    pd.testing.assert_frame_equal(fresh.person, again.person, check_exact=True)


@pytest.mark.parametrize("defect", ("digest", "member", "rows", "bytes"))
def test_changed_member_bytes_or_pin_refuse_before_any_projection(
    tmp_path, monkeypatch, defect
):
    arguments = routing_arguments(tmp_path, monkeypatch)
    prepared = owner.source.prepare_authenticated_survey_population(**arguments)
    pins = list(coverage._MEMBER_PINS)
    if defect == "bytes":
        member = next(p[1] for p in pins if p[0] == 2024)
        path = arguments["source_dir"] / "asec" / member
        path.write_bytes(path.read_bytes().replace(b"12000", b"12001", 1))
    else:
        index = next(i for i, p in enumerate(pins) if p[0] == 2024)
        year, name, archive, digest, rows, size = pins[index]
        pins[index] = {
            "digest": (year, name, archive, "f" * 64, rows, size),
            "member": (year, "pppub99.csv", archive, digest, rows, size),
            "rows": (year, name, archive, digest, rows + 1, size),
        }[defect]
        monkeypatch.setattr(coverage, "_MEMBER_PINS", tuple(pins))
    with pytest.raises(ValueError):
        owner.qualify_current_asec_income_routing(prepared)


@pytest.mark.parametrize("swap", ("coordinate", "amount"))
def test_borrowed_member_must_match_the_parent_row_for_row(tmp_path, monkeypatch, swap):
    arguments = routing_arguments(tmp_path, monkeypatch)
    prepared = owner.source.prepare_authenticated_survey_population(**arguments)
    real = owner._read_capture

    def tampered(path, *, rows):
        frame = real(path, rows=rows)
        column = "A_LINENO" if swap == "coordinate" else "PNSN_VAL"
        values = frame[column].to_numpy(copy=True)
        frame[column] = np.concatenate([values[1:], values[:1]])
        return frame

    monkeypatch.setattr(owner, "_read_capture", tampered)
    expected = (
        "PARENT_COORDINATE_IDENTITY"
        if swap == "coordinate"
        else "CURRENT_AMOUNT_SOURCE_IDENTITY"
    )
    with pytest.raises(ValueError, match=expected):
        owner.qualify_current_asec_income_routing(prepared)


def test_a_tampered_preparation_refuses_before_any_projection(tmp_path, monkeypatch):
    """A replaced checked method is refused by the owner, not worked around."""
    arguments = routing_arguments(tmp_path, monkeypatch)
    prepared = owner.source.prepare_authenticated_survey_population(**arguments)
    real = type(prepared)._checked
    calls = []

    def late(self):
        entry = real(self)
        calls.append(entry)
        return entry if len(calls) == 1 else (entry[0], b"{}", entry[2])

    monkeypatch.setattr(type(prepared), "_checked", late)
    with pytest.raises(ValueError, match="FINAL_AUTHORITY_CHANGED|SOURCE_CHANGED"):
        owner.qualify_current_asec_income_routing(prepared)


def test_the_final_requalification_cannot_be_replaced_or_skipped(tmp_path, monkeypatch):
    """The owner refuses a substituted final requalification, so this qualifier
    cannot be made to return a projection that skipped it."""
    prepared, qualified = _qualified(tmp_path, monkeypatch)
    real = owner.source._pure_final
    seen = []

    def spy(state):
        seen.append(state)
        return real(state)

    # Scoped so undoing the substitution leaves the fixture pins in place.
    with pytest.MonkeyPatch.context() as replaced:
        replaced.setattr(owner.source, "_pure_final", spy)
        with pytest.raises(ValueError, match="FINAL_AUTHORITY_CHANGED"):
            owner.qualify_current_asec_income_routing(prepared)
    # Unreplaced, the same preparation still yields the identical projection.
    again = owner.qualify_current_asec_income_routing(prepared)
    assert (
        again.evidence["projection_sha256"] == qualified.evidence["projection_sha256"]
    )
    assert (
        hashlib.sha256(again.person.to_json(orient="table").encode()).hexdigest()
        == again.evidence["projection_sha256"]
    )


def test_the_qualifier_imports_only_the_accepted_source_neighbourhood():
    """Source-only portability: no graph, fit, engine or modelled stage import."""
    path = Path(owner.__file__)
    tree = ast.parse(path.read_text())
    absolute, relative = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            absolute.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative.update(
                    alias.name for alias in node.names
                ) if node.module is None else relative.add(node.module)
            else:
                absolute.add(node.module.split(".")[0])
    assert absolute <= {
        "__future__",
        "csv",
        "hashlib",
        "json",
        "re",
        "tempfile",
        "dataclasses",
        "functools",
        "importlib",
        "pathlib",
        "typing",
        "numpy",
        "pandas",
    }
    assert relative == {
        "asec_coverage_authentication",
        "asec_current_money",
        "source_csv_builtin",
        "survey_population_preparation",
        "support_provenance",
    }
    # The legacy modelled assumptions are deliberately not reachable from here.
    text = path.read_text()
    for forbidden in ("microcosm.graph", "policyengine", "0.590", "0.59"):
        assert forbidden not in text


def _member_row(**overrides):
    values = {
        "PERIDNUM": "0000000000000000000001",
        "PH_SEQ": "1",
        "A_LINENO": "1",
        "A_AGE": "40",
        **{name: "0" for name in owner.AMOUNT_FIELDS},
        **DEFAULT_LITERAL,
        **overrides,
    }
    return [values[name] for name in owner.READ_COLUMNS]


def _write_member(path, rows):
    lines = [",".join(owner.READ_COLUMNS)]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize(
    "universe,token,kind,net,dollars,label",
    [
        (True, "1", "nonzero", False, False, "known_receipt"),
        (True, "1", "zero", False, False, "ambiguous_recipient_zero"),
        # A signed net measure's recipient zero stays distinct, and stays
        # unknown: RNT_VAL and FRSE_VAL print the same "0 = none or niu" label
        # the gross entries print.
        (True, "1", "zero", True, False, "receipt_with_net_zero"),
        # Only an entry whose printed zero is valid dollars (ANN_VAL) resolves.
        (True, "1", "zero", False, True, "known_recipient_zero"),
        (True, "1", "nonzero", True, False, "known_receipt"),
        (True, "2", "zero", False, False, "known_nonreceipt"),
        (True, "2", "nonzero", False, False, "contradictory_no_nonzero"),
        (True, "0", "zero", False, False, "niu"),
        (True, "0", "nonzero", False, False, "contradictory_niu_nonzero"),
        (True, "0", "declared_niu", False, True, "niu"),
        (True, "1", "declared_niu", False, True, "contradictory_declared_niu_amount"),
        (True, "1", "missing", False, False, "missing_amount"),
        (True, "", "zero", False, False, "missing_receipt_literal"),
        (True, "9", "zero", False, False, "unrecognized_receipt_literal"),
        (True, "01", "zero", False, False, "unrecognized_receipt_literal"),
        (False, "0", "zero", False, False, "outside_reporting_universe"),
        (False, "0", "missing", False, False, "outside_reporting_universe"),
        (
            False,
            "1",
            "nonzero",
            False,
            False,
            "contradictory_outside_reporting_universe",
        ),
        (
            False,
            "2",
            "zero",
            False,
            False,
            "contradictory_outside_reporting_universe",
        ),
        (None, "1", "nonzero", False, False, "unresolved_reporting_universe"),
        (None, "0", "zero", True, False, "unresolved_reporting_universe"),
    ],
)
def test_receipt_and_amount_knownness_stay_separate(
    universe, token, kind, net, dollars, label
):
    receipt = owner.literal_code(token, owner.receipt_codes("PEN_YN"), width=1)
    assert (
        owner.receipt_status(
            universe, receipt, kind, net_measure=net, zero_is_dollars=dollars
        )
        == label
    )
    # Only three statuses establish a dollar reading; nothing else completes.
    assert (label in owner.KNOWN_AMOUNT_STATUSES) == (
        label in ("known_receipt", "known_recipient_zero", "known_nonreceipt")
    )


def test_only_ann_val_prints_a_zero_that_reads_as_valid_dollars():
    entries = owner.printed_amount_entries()
    dollars = {
        name
        for name, entry in entries.items()
        if entry.zero_semantics == owner.DOLLAR_ZERO_SEMANTICS
    }
    assert dollars == {"ANN_VAL"}
    for name in set(owner.AMOUNT_FIELDS) - dollars:
        assert (
            entries[name].zero_semantics
            == "none_or_niu_not_distinguishable_from_amount_alone"
        )
    assert owner._zero_is_dollars("ANN_VAL")
    assert not owner._zero_is_dollars("RNT_VAL")
    assert not owner._zero_is_dollars("FRSE_VAL")


@pytest.mark.parametrize(
    "status,value,kind,dollars",
    [
        (owner.money.CodebookStatus.AMOUNT_NONZERO, 120.0, "nonzero", 120.0),
        (owner.money.CodebookStatus.AMOUNT_NONZERO, -900.0, "nonzero", -900.0),
        (owner.money.CodebookStatus.ZERO_NONE_OR_NIU, 0.0, "zero", 0.0),
        (owner.money.CodebookStatus.ZERO_DOLLARS_AS_CODED, 0.0, "zero", 0.0),
        (owner.money.CodebookStatus.DECLARED_NIU, 0.0, "declared_niu", None),
        (owner.money.CodebookStatus.MISSING_NULL, np.nan, "missing", None),
    ],
)
def test_amount_meaning_comes_from_the_parent_status_axis(status, value, kind, dollars):
    read_kind, read_dollars = owner.amount_state(value, status)
    assert read_kind == kind
    if dollars is None:
        assert np.isnan(read_dollars)
    else:
        assert read_dollars == dollars
    # A normalized NIU zero must never be read back as a zero dollar amount.
    if status == owner.money.CodebookStatus.DECLARED_NIU:
        assert read_kind != "zero"


@pytest.mark.parametrize(
    "token,width,code,status",
    [
        ("4", 1, 4, "in_printed_range"),
        ("20", 2, 20, "in_printed_range"),
        ("", 1, None, "missing"),
        ("01", 1, None, "malformed"),
        (" 4", 1, None, "malformed"),
        ("4.0", 1, None, "malformed"),
        ("NA", 1, None, "malformed"),
        ("9", 1, None, "outside_printed_range"),
        ("21", 2, None, "outside_printed_range"),
    ],
)
def test_routing_literals_stay_unknown_rather_than_recoded(token, width, code, status):
    allowed = owner.ACCOUNT_CODES if width == 1 else owner.OTHER_INCOME_CATEGORIES
    assert owner.literal_code(token, allowed, width=width) == (code, status)


@pytest.mark.parametrize(
    "defect,match",
    [
        ("duplicate_key", "DUPLICATE_OR_INVALID_COORDINATE"),
        ("duplicate_coordinate", "DUPLICATE_OR_INVALID_COORDINATE"),
        ("short_key", "PERSON_KEY"),
        ("bad_age", "COORDINATE:A_AGE"),
        ("unsigned_negative", "AMOUNT_TOKEN:PNSN_VAL"),
        ("wide_amount", "AMOUNT_TOKEN:OI_VAL"),
        ("oversized_token", "TOKEN_BOUND:OI_OFF"),
        ("missing_column", "HEADER"),
        ("row_count", "ROW_COUNT"),
    ],
)
def test_bounded_member_reader_refuses_structural_defects(tmp_path, defect, match):
    path = tmp_path / "invented.csv"
    rows = [_member_row()]
    if defect == "duplicate_key":
        rows.append(_member_row(PH_SEQ="2"))
    elif defect == "duplicate_coordinate":
        rows.append(_member_row(PERIDNUM="0000000000000000000002"))
    elif defect == "short_key":
        rows = [_member_row(PERIDNUM="1")]
    elif defect == "bad_age":
        rows = [_member_row(A_AGE="123")]
    elif defect == "unsigned_negative":
        rows = [_member_row(PNSN_VAL="-5")]
    elif defect == "wide_amount":
        rows = [_member_row(OI_VAL="1234567")]
    elif defect == "oversized_token":
        rows = [_member_row(OI_OFF="9" * 65)]
    _write_member(path, rows)
    if defect == "missing_column":
        text = path.read_text().split("\n")
        keep = [i for i, c in enumerate(owner.READ_COLUMNS) if c != "OI_OFF"]
        path.write_text(
            "\n".join(",".join(line.split(",")[i] for i in keep) for line in text[:-1])
            + "\n"
        )
    expected = 2 if defect.startswith("duplicate") else 1
    if defect == "row_count":
        expected = 2
    with pytest.raises(ValueError, match=match):
        owner._read_capture(path, rows=expected)


def test_bounded_member_reader_accepts_signed_and_missing_amount_literals(tmp_path):
    path = _write_member(
        tmp_path / "invented.csv",
        [_member_row(RNT_VAL="-9999", FRSE_VAL="-9999999", ANN_VAL="-1", OI_VAL="")],
    )
    frame = owner._read_capture(path, rows=1)
    assert frame.iloc[0].RNT_VAL == "-9999"
    assert frame.iloc[0].FRSE_VAL == "-9999999"
    assert frame.iloc[0].ANN_VAL == "-1"
    assert frame.iloc[0].OI_VAL == ""


def _field(name, amounts, statuses, validity):
    return owner.money.MoneyField(
        name,
        np.array(amounts, dtype="<f8").tobytes(),
        bytes(int(s) for s in statuses),
        bytes(validity),
        bytes(len(amounts)),
    )


def test_money_literal_join_keeps_missing_storage_and_niu_normalization():
    positions = np.array([0, 1, 2], dtype="int64")
    field = _field(
        "PNSN_VAL",
        [0.0, 0.0, 4000.0],
        [
            owner.money.CodebookStatus.MISSING_NULL,
            owner.money.CodebookStatus.ZERO_NONE_OR_NIU,
            owner.money.CodebookStatus.AMOUNT_NONZERO,
        ],
        [0, 1, 1],
    )
    amounts, statuses = owner.amount_observations(field, positions, ["", "0", "4000"])
    assert np.isnan(amounts[0]) and amounts[1] == 0.0 and amounts[2] == 4000.0
    assert statuses[0] == owner.money.CodebookStatus.MISSING_NULL
    # Missing backing storage never becomes a survey zero.
    with pytest.raises(ValueError, match="SOURCE_VALIDITY"):
        owner.amount_observations(field, positions, ["0", "0", "4000"])
    with pytest.raises(ValueError, match="SOURCE_IDENTITY"):
        owner.amount_observations(field, positions, ["", "0", "4001"])
    # ANN_VAL's printed -1 is stored as a normalized zero under DECLARED_NIU.
    annuity = _field(
        "ANN_VAL",
        [0.0, 500.0],
        [
            owner.money.CodebookStatus.DECLARED_NIU,
            owner.money.CodebookStatus.AMOUNT_NONZERO,
        ],
        [1, 1],
    )
    amounts, statuses = owner.amount_observations(
        annuity, np.array([0, 1], dtype="int64"), ["-1", "500"]
    )
    assert amounts.tolist() == [0.0, 500.0]
    assert statuses[0] == owner.money.CodebookStatus.DECLARED_NIU
    with pytest.raises(ValueError, match="NIU_IDENTITY"):
        owner.amount_observations(
            annuity, np.array([0, 1], dtype="int64"), ["0", "500"]
        )


def test_printed_amount_entries_come_from_the_pinned_domains_artifact():
    entries = owner.printed_amount_entries()
    assert set(entries) == set(owner.AMOUNT_FIELDS)
    pension = entries["PNSN_VAL"]
    assert (pension.printed_length, pension.printed_position) == (7, 571)
    assert pension.universe_as_printed == "PEN_YN = 1"
    assert pension.printed_page == "6C-26"
    assert pension.nonmoney_codes == ()
    annuity = entries["ANN_VAL"]
    assert annuity.encoded_minimum == -1 and annuity.nonmoney_codes == (-1,)
    assert annuity.negative_dollars_permitted is False
    for name, minimum in (("RNT_VAL", -9999), ("FRSE_VAL", -9999999)):
        assert entries[name].encoded_minimum == minimum
        assert entries[name].negative_dollars_permitted is True
    assert entries["FRSE_VAL"].universe_as_printed == "ERN_YN=1 or FRMOTR=1"
    # The artifact is read under its pinned digest, never re-typed here.
    assert owner.DOMAINS_SHA256 == owner.money.RESOURCE_PINS[0]
    with pytest.raises(ValueError, match="DOMAINS_RESOURCE_SHA256"):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(owner, "DOMAINS_SHA256", "f" * 64)
            owner.printed_amount_entries.cache_clear()
            owner.printed_amount_entries()
    owner.printed_amount_entries.cache_clear()


def test_routing_code_systems_agree_with_the_existing_domain_constants():
    """Derived views stay consistent with the canonical definitions in tree."""
    from microcosm.build.us_runtime import alimony
    from microcosm.build.us_runtime import retirement_distributions as distributions

    slots = set(distributions.US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS)
    projected = {
        name
        for name in owner.READ_COLUMNS
        if name.startswith("DST_SC") or name.startswith("DST_VAL")
    }
    assert projected == slots
    assert owner.ALIMONY_CATEGORY_CODE == alimony._ASEC_ALIMONY_OTHER_INCOME_CODE
    assert owner.OTHER_INCOME_CATEGORIES[owner.ALIMONY_CATEGORY_CODE] == "alimony"
    # Compare against the canonical code domain, not a re-typed literal.
    assert set(owner.ACCOUNT_CODES) == set(distributions._VALID_ACCOUNT_CODES)
    assert set(distributions._EXPECTED_OUTPUT_BY_ACCOUNT_CODE) <= set(
        owner.ACCOUNT_CODES
    )
    assert owner.ACCOUNT_CODES[owner.REGULAR_IRA_CODE] == "Regular IRA"
    assert (
        owner.OTHER_INCOME_CATEGORIES[alimony._ASEC_STRIKE_BENEFITS_OTHER_INCOME_CODE]
        == "strike benefits"
    )
    # Published flags and unflagged fields are disjoint and jointly complete.
    named = set(owner.PUBLISHED_ALLOCATION_FLAG_BY_FIELD)
    assert named.isdisjoint(owner.UNFLAGGED_FIELDS)
    projected_fields = set(owner.READ_COLUMNS) - set(owner.COORDINATE_COLUMNS)
    projected_fields -= set(owner.ALLOCATION_ENTRIES)
    assert named | set(owner.UNFLAGGED_FIELDS) == projected_fields
    assert set(owner.PUBLISHED_ALLOCATION_FLAG_BY_FIELD.values()) <= set(
        owner.ALLOCATION_ENTRIES
    )


def test_both_distribution_recipiency_literals_are_retained_with_the_route():
    n = 2
    raw = {
        "amounts": {name: np.zeros(n) for name in owner.AMOUNT_FIELDS},
        "statuses": {
            name: np.full(n, int(owner.money.CodebookStatus.ZERO_NONE_OR_NIU), "u1")
            for name in owner.AMOUNT_FIELDS
        },
        "allocations": {
            name: ([0] * n, ["in_printed_range"] * n)
            for name in owner.ALLOCATION_ENTRIES
        },
    }
    for name in (*owner.RECEIPT_ENTRIES, *owner.ACCOUNT_ENTRIES, "OI_OFF"):
        raw[name] = ["0"] * n
    # A 70 year old whose under-58 recipiency answers yes, and a 40 year old
    # whose 58-and-over recipiency answers yes: both are off-route answers.
    raw["DST_YN_YNG"] = ["1", "0"]
    raw["DST_YN"] = ["0", "1"]
    out = owner.project_income_routing(raw, np.array([70.0, 40.0]))
    assert out.retirement_distribution_route.tolist() == [
        "age58_and_over",
        "under_age58",
    ]
    assert out.retirement_distribution_receipt_58_literal.tolist() == ["0", "1"]
    assert out.retirement_distribution_receipt_young_literal.tolist() == ["1", "0"]
    assert out.retirement_distribution_offroute_receipt.tolist() == [True, True]
    # An answered off-route recipiency contradicts the route, so the applicable
    # literal cannot resolve the row however it reads.
    assert out.retirement_distribution_reporting_status.tolist() == [
        "contradictory_offroute_evidence",
        "contradictory_offroute_evidence",
    ]
    assert out.retirement_distribution_known_amount.isna().all()
    # A clean pair reports no off-route answer. Here each person answers yes on
    # their own route with every applicable slot at the printed "none or niu"
    # zero, which stays ambiguous rather than becoming a known zero.
    raw["DST_YN_YNG"], raw["DST_YN"] = ["0", "1"], ["1", "0"]
    clean = owner.project_income_routing(raw, np.array([70.0, 40.0]))
    assert clean.retirement_distribution_offroute_receipt.tolist() == [False, False]
    assert clean.retirement_distribution_reporting_status.tolist() == [
        "ambiguous_recipient_zero",
        "ambiguous_recipient_zero",
    ]
    assert clean.retirement_distribution_known_amount.isna().all()
    # Answering no on the applicable route does give a known zero.
    raw["DST_YN_YNG"], raw["DST_YN"] = ["0", "2"], ["2", "0"]
    declined = owner.project_income_routing(raw, np.array([70.0, 40.0]))
    assert declined.retirement_distribution_reporting_status.tolist() == [
        "known_nonreceipt",
        "known_nonreceipt",
    ]
    assert declined.retirement_distribution_known_amount.tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    "receipt,category,routing,alimony",
    [
        ("1", "20", "reported_category", True),
        ("1", "19", "reported_category", False),
        ("1", "0", "receipt_without_category", False),
        ("2", "20", "category_without_receipt", False),
        ("0", "20", "category_without_receipt", False),
        ("2", "0", "niu_category", False),
        ("", "20", "unresolved_receipt_routing", False),
        ("9", "20", "unresolved_receipt_routing", False),
        ("1", "", "missing_category_literal", False),
        ("1", "21", "unrecognized_category_literal", False),
        ("1", "NA", "unrecognized_category_literal", False),
    ],
)
def test_other_income_routing_never_invents_a_reported_category(
    receipt, category, routing, alimony
):
    n = 1
    raw = {
        "amounts": {name: np.zeros(n) for name in owner.AMOUNT_FIELDS},
        "statuses": {
            name: np.full(n, int(owner.money.CodebookStatus.ZERO_NONE_OR_NIU), "u1")
            for name in owner.AMOUNT_FIELDS
        },
        "allocations": {
            name: ([0] * n, ["in_printed_range"] * n)
            for name in owner.ALLOCATION_ENTRIES
        },
    }
    for name in (*owner.RECEIPT_ENTRIES, *owner.ACCOUNT_ENTRIES, "OI_OFF"):
        raw[name] = ["0"] * n
    raw["amounts"]["OI_VAL"] = np.array([1500.0])
    raw["statuses"]["OI_VAL"] = np.full(
        n, int(owner.money.CodebookStatus.AMOUNT_NONZERO), "u1"
    )
    raw["OI_YN"], raw["OI_OFF"] = [receipt], [category]
    out = owner.project_income_routing(raw, np.array([45.0]))
    assert out.other_income_routing_status.iloc[0] == routing
    assert bool(out.other_income_is_reported_alimony.iloc[0]) is alimony
    assert out.other_income_source_total.iloc[0] == 1500.0
    # The reported category never becomes a residual assignment.
    assert not bool(out.other_income_residual_rule_applied.iloc[0])


def test_printed_flag_and_scope_claims_match_the_dictionary_as_printed():
    # I_FRMYN prints an empty Values block, so only its (0:9) range header is
    # published; no code meaning is claimed for it.
    assert owner.ALLOCATION_ENTRIES["I_FRMYN"].codes == tuple(range(10))
    assert (
        owner.ALLOCATION_ENTRIES["I_FRMYN"].codes is not owner.ALLOCATION_ANNVAL_CODES
    )
    assert owner.ALLOCATION_ENTRIES["I_DSTSC"].codes == (0, 1, 9)
    for name in ("I_DSTVAL1COMP", "I_DSTVAL2COMP", "I_DSTYNCOMP"):
        assert owner.ALLOCATION_ENTRIES[name].codes == (0, 10, 11)
    # I_DSTVAL1COMP's printed universe line is empty in the dictionary.
    assert owner.ALLOCATION_ENTRIES["I_DSTVAL1COMP"].universe_as_printed == ""
    # The DST_SC(2) notation and the empty Values block stay recorded as
    # printed ambiguities rather than resolved silently.
    assert set(owner.AMBIGUOUS_FLAG_COVERAGE) == {
        "I_DSTSC",
        "I_DSTSCCOMP",
        "I_FRMYN",
    }
    assert len(owner.UNFLAGGED_FIELDS) == 10
    # Printed labels are complete, including the farm composite clause.
    assert owner.FARM_AMOUNT_SCOPE.endswith(
        "(combined amounts in ERN_VAL, if ERN_SRCE=3, and FRM_VAL)"
    )
    assert owner.PENSION_TOTAL_SCOPE.endswith("from all pension sources")
    assert "estates or trusts" in owner.NET_PROPERTY_RECEIPT_SCOPE
    assert "estates" not in owner.NET_PROPERTY_AMOUNT_SCOPE
    # The two printed-question dictionaries stay separate: one is about printed
    # universes, the other about printed scopes and code labels.
    assert set(owner.ACCOUNT_ENTRIES["DST_SC1"][4:]) == {"DST_VAL1 > 0 and a_age ≥ 58"}
    assert owner.ACCOUNT_ENTRIES["DST_SC1_YNG"][4] == ("DST_YN_YNG = 1 and a_age < 58")
    assert owner.ACCOUNT_ENTRIES["DST_SC2_YNG"][4] == ("DST_VAL_YNG > 0 and a_age < 58")
    assert owner.RECEIPT_ENTRIES["OI_YN"][5] == "none or niu"
    assert owner.receipt_codes("OI_YN")[0] == "none or niu"
    for name in ("PEN_YN", "ANN_YN", "DST_YN", "DST_YN_YNG", "RNT_YN", "ERN_YN"):
        assert owner.receipt_codes(name)[0] == "niu"
    assert owner.receipt_codes("FRSE_YN")[0] == "Niu"
    assert owner.RECEIPT_ENTRIES["FRSE_YN"].universe_as_printed == (
        "ERN_YN=1 or FRMOTR=1"
    )
    for name in ("PEN_YN", "ANN_YN", "RNT_YN", "OI_YN"):
        assert owner.RECEIPT_ENTRIES[name].universe_as_printed == (
            "All Persons aged 15+"
        )


def _pure_rows(n, ages, **literals):
    codebook = owner.money.CodebookStatus
    amounts = {name: np.zeros(n) for name in owner.AMOUNT_FIELDS}
    statuses = {
        name: np.full(n, int(codebook.ZERO_NONE_OR_NIU), "u1")
        for name in owner.AMOUNT_FIELDS
    }
    statuses["ANN_VAL"] = np.full(n, int(codebook.ZERO_DOLLARS_AS_CODED), "u1")
    raw = {
        "amounts": amounts,
        "statuses": statuses,
        "allocations": {
            name: ([0] * n, ["in_printed_range"] * n)
            for name in owner.ALLOCATION_ENTRIES
        },
    }
    for name in (*owner.RECEIPT_ENTRIES, *owner.ACCOUNT_ENTRIES, "OI_OFF"):
        raw[name] = ["0"] * n
    raw.update(literals)
    return raw, np.asarray(ages, dtype="float64")


def _set_amount(raw, field, values):
    codebook = owner.money.CodebookStatus
    raw["amounts"][field] = np.asarray(values, dtype="float64")
    raw["statuses"][field] = np.array(
        [
            int(codebook.AMOUNT_NONZERO)
            if v
            else int(
                codebook.ZERO_DOLLARS_AS_CODED
                if field == "ANN_VAL"
                else codebook.ZERO_NONE_OR_NIU
            )
            for v in values
        ],
        dtype="u1",
    )


def test_offroute_distribution_dollars_block_a_known_absence():
    raw, ages = _pure_rows(1, [70.0], DST_YN=["2"], DST_SC1_YNG=["4"])
    _set_amount(raw, "DST_VAL1_YNG", [8000.0])
    out = owner.project_income_routing(raw, ages)
    assert out.retirement_distribution_route.iloc[0] == "age58_and_over"
    assert bool(out.retirement_distribution_offroute_nonzero.iloc[0])
    # The applicable literals alone would read as a known zero; the retained
    # off-route dollars contradict that, so nothing is resolved.
    assert out.retirement_distribution_reporting_status.iloc[0] == (
        "contradictory_offroute_evidence"
    )
    assert np.isnan(out.retirement_distribution_known_amount.iloc[0])
    assert out.retirement_distribution_slot1_young_amount.iloc[0] == 8000.0


def test_an_ambiguous_slot_zero_leaves_the_total_and_the_ira_share_unknown():
    raw, ages = _pure_rows(1, [70.0], DST_YN=["1"], DST_SC1=["4"], DST_SC2=["1"])
    _set_amount(raw, "DST_VAL1", [5000.0])
    out = owner.project_income_routing(raw, ages)
    # Slot 2 declares a 401k account whose amount is a "none or niu" zero.
    assert out.retirement_distribution_slot2_slot_status.iloc[0] == (
        "ambiguous_slot_zero"
    )
    assert bool(out.retirement_distribution_slot_zero_ambiguity.iloc[0])
    assert out.retirement_distribution_source_total.iloc[0] == 5000.0
    assert out.retirement_distribution_reporting_status.iloc[0] == (
        "unresolved_slot_composition"
    )
    assert np.isnan(out.retirement_distribution_known_amount.iloc[0])
    assert np.isnan(out.retirement_distribution_regular_ira_amount.iloc[0])
    assert pd.isna(out.retirement_distribution_regular_ira_slots.iloc[0])
    # A regular IRA slot whose own amount is that ambiguous zero is not a known
    # zero distribution from that account either.
    raw, ages = _pure_rows(1, [70.0], DST_YN=["1"], DST_SC1=["4"], DST_SC2=["1"])
    _set_amount(raw, "DST_VAL2", [5000.0])
    ambiguous = owner.project_income_routing(raw, ages)
    assert ambiguous.retirement_distribution_slot1_slot_status.iloc[0] == (
        "ambiguous_slot_zero"
    )
    assert np.isnan(ambiguous.retirement_distribution_regular_ira_amount.iloc[0])
    # A contradictory NIU slot carrying dollars is likewise unresolved.
    raw, ages = _pure_rows(1, [70.0], DST_YN=["1"], DST_SC1=["0"])
    _set_amount(raw, "DST_VAL1", [5000.0])
    contradictory = owner.project_income_routing(raw, ages)
    assert contradictory.retirement_distribution_slot1_slot_status.iloc[0] == (
        "contradictory_niu_slot_amount"
    )
    assert np.isnan(contradictory.retirement_distribution_regular_ira_amount.iloc[0])
    assert pd.isna(contradictory.retirement_distribution_regular_ira_slots.iloc[0])
    # Two fully resolved slots do give a known composition.
    raw, ages = _pure_rows(1, [70.0], DST_YN=["1"], DST_SC1=["4"], DST_SC2=["0"])
    _set_amount(raw, "DST_VAL1", [5000.0])
    resolved = owner.project_income_routing(raw, ages)
    assert resolved.retirement_distribution_reporting_status.iloc[0] == "known_receipt"
    assert resolved.retirement_distribution_known_amount.iloc[0] == 5000.0
    assert resolved.retirement_distribution_regular_ira_amount.iloc[0] == 5000.0
    assert resolved.retirement_distribution_regular_ira_slots.iloc[0] == 1


@pytest.mark.parametrize(
    "age,ern,frmotr,routing",
    [
        (14, "1", "0", "outside_reporting_universe_routing"),
        (40, "", "", "reported_category"),
    ],
)
def test_other_income_routing_follows_the_printed_receipt_universe(
    age, ern, frmotr, routing
):
    raw, ages = _pure_rows(
        1,
        [float(age)],
        OI_YN=["1"],
        OI_OFF=["20"],
        ERN_YN=[ern],
        FRMOTR=[frmotr],
    )
    _set_amount(raw, "OI_VAL", [3000.0])
    out = owner.project_income_routing(raw, ages)
    assert out.other_income_routing_status.iloc[0] == routing
    assert bool(out.other_income_is_reported_alimony.iloc[0]) == (
        routing == "reported_category"
    )
    # The farm universe on the same rows comes from ERN_YN/FRMOTR, not from age.
    assert pd.isna(out.farm_source_reporting_universe.iloc[0]) == (ern == "")


def test_allocation_origins_do_not_assert_publisher_confirmed_non_allocation():
    raw, ages = _pure_rows(1, [40.0])
    zeroed = owner.project_income_routing(raw, ages)
    # Every flag here prints a conditional universe that is not evaluated, so an
    # all-zero reading is reported as exactly that, never as "no allocation".
    assert zeroed.net_property_allocation_origin.iloc[0] == "published_flags_all_zero"
    assert zeroed.pension_annuity_allocation_origin.iloc[0] == (
        "published_flags_all_zero_with_unflagged_fields"
    )
    for column in [c for c in zeroed.columns if c.endswith("_allocation_origin")]:
        assert "no_allocation" not in zeroed[column].iloc[0]
    raw, ages = _pure_rows(1, [40.0])
    raw["allocations"]["I_RNTVAL"] = ([4], ["in_printed_range"])
    assert (
        owner.project_income_routing(raw, ages).net_property_allocation_origin.iloc[0]
        == "publisher_allocated"
    )
    raw, ages = _pure_rows(1, [40.0])
    raw["allocations"]["I_RNTYN"] = ([None], ["missing"])
    assert (
        owner.project_income_routing(raw, ages).net_property_allocation_origin.iloc[0]
        == "allocation_flag_not_populated"
    )
    raw, ages = _pure_rows(1, [40.0])
    raw["allocations"]["I_RNTYN"] = ([None], ["outside_printed_range"])
    assert (
        owner.project_income_routing(raw, ages).net_property_allocation_origin.iloc[0]
        == "unresolved_allocation_provenance"
    )


@pytest.mark.parametrize("field", ("OI_YN", "DST_SC1", "OI_OFF"))
def test_the_public_projection_validates_its_routing_token_arrays(field):
    raw, ages = _pure_rows(2, [40.0, 40.0])
    raw[field] = ["0"]
    with pytest.raises(ValueError, match="ROUTING_TOKEN_CONTRACT:" + field):
        owner.project_income_routing(raw, ages)
    raw[field] = ["0", 0]
    with pytest.raises(ValueError, match="ROUTING_TOKEN_CONTRACT:" + field):
        owner.project_income_routing(raw, ages)


def test_the_literal_transport_carries_its_own_digest(tmp_path, monkeypatch):
    _, qualified = _qualified(tmp_path, monkeypatch)
    assert (
        hashlib.sha256(
            qualified.asec_literals.to_json(orient="table").encode()
        ).hexdigest()
        == qualified.evidence["literals_sha256"]
    )
    assert (
        qualified.evidence["literals_sha256"] != qualified.evidence["projection_sha256"]
    )
    # A host comparing the documented digests detects a corrupted literal frame.
    index = qualified.asec_literals.index[0]
    qualified.asec_literals.loc[index, "PNSN_VAL"] = "999999"
    assert (
        hashlib.sha256(
            qualified.asec_literals.to_json(orient="table").encode()
        ).hexdigest()
        != qualified.evidence["literals_sha256"]
    )
