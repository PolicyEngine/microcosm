"""Actual source readers over constructed, privately pinned invented source bytes."""

import csv
import hashlib
import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from test_us_survey_population_preparation import fixture

from microcosm.build.us_runtime import asec_coverage_authentication as coverage
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import current_asec_demographics as owner
from microcosm.build.us_runtime import native_household_origin as origin


def _csv(path, headers, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    data = path.read_bytes()
    return SimpleNamespace(
        size_bytes=len(data),
        rows=len(rows),
        member_sha256=hashlib.sha256(data).hexdigest(),
    )


@pytest.mark.parametrize(
    "literal,code,status",
    (
        ("06", 6, "in_printed_range"),
        ("36", 36, "in_printed_range"),
        ("", None, "missing"),
        ("NA", None, "malformed"),
        (" 6", None, "malformed"),
        ("6.0", None, "malformed"),
        ("57", None, "outside_printed_range"),
        ("00", None, "outside_printed_range"),
    ),
)
def test_household_state_reader_preserves_literal_and_knownness(
    tmp_path, literal, code, status
):
    path = tmp_path / "invented.csv"
    pin = _csv(path, ("H_SEQ", "GESTFIPS", "unread"), [("00007", literal, "anything")])
    value = owner._read_state_capture(path, pin)[7]
    assert value == {
        "H_SEQ": "00007",
        "GESTFIPS": literal,
        "state_code": code,
        "status": status,
        "member_row_1based": 1,
    }


@pytest.mark.parametrize("defect", ("duplicate", "digest", "size", "rows", "header"))
def test_state_capture_structure_and_exact_bytes_refuse(tmp_path, defect):
    path = tmp_path / "invented.csv"
    headers, rows = ("H_SEQ", "GESTFIPS"), [("00007", "06")]
    if defect == "duplicate":
        rows.append(("7", "36"))
    if defect == "header":
        headers = ("H_SEQ", "other")
    pin = _csv(path, headers, rows)
    if defect == "digest":
        pin.member_sha256 = "f" * 64
    elif defect == "size":
        pin.size_bytes += 1
    elif defect == "rows":
        pin.rows += 1
    with pytest.raises(ValueError, match="CURRENT_ASEC_DEMOGRAPHICS_"):
        owner._read_state_capture(path, pin)


def _demographic_arguments(tmp_path, monkeypatch, *, unknown=False, zero=True):
    """Extend invented source members before actual preparation/issuance.

    Rebuild the real person-income attachment after member bytes change. Only
    fixture registry pins are changed; no source issuer or ready() is replaced.
    """
    arguments = fixture(tmp_path, monkeypatch, zero=zero)
    source = arguments["source_dir"] / "asec"
    members, pins = {}, []
    for year, member, archive, *_ in coverage._MEMBER_PINS:
        path = source / f"pppub{year - 1999}.csv"
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        lines = table.A_LINENO.map(int)
        table["A_SEX"] = lines.map({1: "1", 2: "2"})
        table["AXSEX"] = lines.map({1: "0", 2: "4"})
        table["A_EXPRRP"] = lines.map({1: "1", 2: "5"})
        table["P_SEQ"] = table.A_LINENO
        if unknown and year == 2024:
            table.loc[table.index[0], "AXSEX"] = "1"
        assert table.notna().all().all()
        # Reverse the actual source order, so receiving-row alignment cannot pass.
        table.iloc[::-1].to_csv(path, index=False)
        data = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(data).hexdigest(),
                len(table),
                len(data),
            )
        )
        members[year] = path
    for module in (coverage, restoration, owner.demographic):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = tmp_path / "demographic-restored-money"
    restoration.restore_asec_person_income_source(
        source / "parent.h5",
        source / "household-attachment.h5",
        member_paths=members,
        output_dir=output,
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, source / "person-income-attachment.h5"
    )
    path = source / "hhpub25.csv"
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table["GESTFIPS"] = table.H_SEQ.map({"00007": "06", "00008": "36"})
    if unknown:
        table.loc[table.H_SEQ.eq("00008"), "GESTFIPS"] = ""
    assert table.notna().all().all()
    table.iloc[::-1].to_csv(path, index=False)
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
    return arguments


@pytest.mark.parametrize("unknown", (False, True))
def test_actual_current_native_parent_binds_demographics_by_origin(
    tmp_path, monkeypatch, unknown
):
    arguments = _demographic_arguments(tmp_path, monkeypatch, unknown=unknown)
    prepared = owner.preparation_owner.prepare_authenticated_survey_population(
        **arguments
    )
    original = prepared.checked_view().frame
    retained = {e: original.table(e).copy(deep=True) for e in original.entities}
    result = owner.qualify_current_asec_demographics(prepared)
    receipt = json.loads(result.receipt)
    assert receipt["sex_observation_year"] == receipt["state_observation_year"] == 2025
    assert receipt["income_year"] == 2024
    assert not receipt["state_is_income_year_residence_claim"]
    assert not receipt["source_admission_issued"] and not receipt["release_eligible"]
    h = result.household.set_index("H_SEQ_integer")
    assert h.loc[7, "GESTFIPS"] == "06" and h.loc[7, "state_fips"] == 6
    if unknown:
        assert h.loc[8, "GESTFIPS"] == "" and pd.isna(h.loc[8, "state_fips"])
        assert receipt["state_unknown_households"] == 1
        assert receipt["sex_unknown_persons"] == 1
        unbound = result.person.asec_AXSEX.eq(1)
        assert result.person.loc[unbound, "is_female"].isna().all()
        assert not result.person.loc[unbound, "sex_known"].any()
    else:
        assert h.loc[8, "state_fips"] == 36
        assert (
            receipt["state_unknown_households"] == receipt["sex_unknown_persons"] == 0
        )
    allocated = result.person.asec_AXSEX.eq(4)
    assert allocated.any() and result.person.loc[allocated, "is_female"].all()
    assert result.person.loc[allocated, "sex_origin"].eq("census_allocated").all()
    for entity, expected in retained.items():
        pd.testing.assert_frame_equal(
            expected, original.table(entity), check_exact=True
        )
    assert owner.qualify_current_asec_demographics(prepared).receipt == result.receipt
    # Value transport is deliberately not an issued authority. Mutating it must
    # not affect a fresh live projection or its source-bound receipt.
    result.household.loc[result.household.index[0], "state_fips"] = 99
    fresh = owner.qualify_current_asec_demographics(prepared)
    assert not fresh.household.state_fips.eq(99).any()
