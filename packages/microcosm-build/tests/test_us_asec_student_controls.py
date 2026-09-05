"""Invented source bytes and actual tiny microunit; no genuine population runs."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build import frame_checkpoint as checkpoint
from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_source as legacy
from microcosm.build.us_runtime import asec_current_money_units as units
from microcosm.build.us_runtime import asec_housing_status_source as housing
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import asec_student_controls as student


def _fixture(tmp_path, monkeypatch, *, control_missing=True, incumbent_missing=False):
    # A dependency absence is an error in this focused contract, never a green skip.
    import microunit  # noqa: F401

    path = Path(__file__).with_name("test_us_asec_person_income_source.py")
    spec = importlib.util.spec_from_file_location("student_parent_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent, household, members = module.invented_sources(
        tmp_path, monkeypatch, housing=True
    )
    enroll = [0, 1, 0, 1, 0, 2]
    fulltime = [0, 1, 0, 2, 0, 0]
    for path in (parent, household):
        loaded = checkpoint.load_frame_checkpoint(path)
        loaded.frame.person["A_AGE"] = [55, 20] * 3
        for name, values in zip(student.CONTROLS, (enroll, fulltime), strict=True):
            loaded.frame.person[name] = (
                ([np.nan] * 4 + values[4:]) if control_missing else values
            )
        if incumbent_missing:
            loaded.frame.person.iloc[
                4, loaded.frame.person.columns.get_loc("A_ENRLW")
            ] = np.nan
        if path == household:
            pin = money._sha(parent.read_bytes())
            loaded.metadata["parent_checkpoint_sha256"] = pin
            loaded.metadata["household_observations"]["input_checkpoint_sha256"] = pin
        checkpoint.write_frame_checkpoint(path, loaded.frame, metadata=loaded.metadata)
    monkeypatch.setattr(
        legacy,
        "_SOURCE_PINS",
        (
            money._sha(parent.read_bytes()),
            money._sha(household.read_bytes()),
            legacy._SOURCE_PINS[2],
        ),
    )
    pins = []
    for year, name, archive, _, rows, _ in restoration._MEMBER_PINS:
        path = members[year]
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        # Existing fixture CSV is reverse person order. Assign observations by
        # native key, not by the output position under test.
        for index, row in table.iterrows():
            i = int(row.PERIDNUM) - 1
            table.loc[index, "A_AGE"] = str([55, 20][i % 2])
            table.loc[index, "A_ENRLW"] = str(enroll[i])
            table.loc[index, "A_FTPT"] = str(fulltime[i])
        table.to_csv(path, index=False)
        pins.append(
            (
                year,
                name,
                archive,
                money._sha(path.read_bytes()),
                rows,
                path.stat().st_size,
            )
        )
    monkeypatch.setattr(restoration, "_MEMBER_PINS", tuple(pins))
    monkeypatch.setattr(student, "_MEMBER_PINS", tuple(pins))
    output = tmp_path / "T1"
    restoration.restore_asec_person_income_source(
        parent, household, member_paths=members, output_dir=output
    )
    t1 = output / restoration.CHECKPOINT_FILENAME
    source = restoration.load_authenticated_restored_current_money_source(
        parent, household, t1, member_paths=members
    )
    return source, source.ready(), members, (parent, household, t1)


def _load(source, ready, members):
    return student.load_authenticated_student_controls(
        source, ready, member_paths=members
    )


def test_missing_cohorts_refuse_then_exact_source_attachment_reaches_real_microunit_and_housing(
    tmp_path, monkeypatch
):
    source, ready, members, parents = _fixture(tmp_path, monkeypatch)
    original = legacy._frame_signature(source.frame)
    before_bytes = [p.read_bytes() for p in parents]
    money_before = (ready.header, tuple(ready.fields))
    with pytest.raises(money.MoneyRefusalError) as error:
        units.reconstruct_current_money_tax_units(source, ready)
    assert error.value.reason == "MICROUNIT_CONTROL_MISSING"
    assert error.value.field == "A_ENRLW"
    observations = _load(source, ready, members)
    combined = student.attach_student_controls(source, ready, observations)
    assert combined.ready() is ready
    assert combined.source is source.source and combined.spec is source.spec
    assert observations.array("A_ENRLW").tolist() == [0, 1, 0, 1, 0, 2]
    assert observations.array("A_FTPT").tolist() == [0, 1, 0, 2, 0, 0]
    assert observations.receipt["annual_five_month_student_status_validated"] is False
    assert [j["incumbent_compared_rows"] for j in observations.receipt["sources"]] == [
        {"A_ENRLW": 0, "A_FTPT": 0}
    ] * 2 + [{"A_ENRLW": 2, "A_FTPT": 2}]
    view = units.current_money_microunit_view(combined, ready)
    assert view.A_ENRLW.tolist() == [0, 1, 0, 1, 0, 2]
    assert view.A_FTPT.tolist() == [0, 1, 0, 2, 0, 0]
    # Actual pinned microunit, rather than a replacement classifier.
    tax = units.reconstruct_current_money_tax_units(combined, ready)
    tax.validate()
    assert tax.money is ready
    assert tax.receipt["student_controls"] == combined.receipt
    for column in source.frame.person:
        if column not in units.TAX_PERSON_OUTPUTS:
            pd.testing.assert_series_equal(
                tax.frame.person[column], source.frame.person[column], check_exact=True
            )
    status = housing.load_authenticated_housing_status(
        source,
        cohort_paths={
            y: tmp_path / f"invented-cohort-{y}.h5" for y in (2022, 2023, 2024)
        },
    )
    final = housing.attach_housing_status(tax, status)
    final.validate()
    assert final.money is ready
    assert final.frame.person.A_ENRLW.isna().tolist() == [True] * 4 + [False] * 2
    assert final.frame.person.A_FTPT.isna().tolist() == [True] * 4 + [False] * 2
    assert legacy._frame_signature(source.frame) == original
    assert (ready.header, ready.fields) == money_before
    assert [p.read_bytes() for p in parents] == before_bytes


def test_complete_source_retains_original_microunit_path(tmp_path, monkeypatch):
    source, ready, members, _ = _fixture(tmp_path, monkeypatch, control_missing=False)
    original_tax = units.reconstruct_current_money_tax_units(source, ready)
    combined = student.attach_student_controls(
        source, ready, _load(source, ready, members)
    )
    attached_tax = units.reconstruct_current_money_tax_units(combined, ready)
    original_tax.validate()
    attached_tax.validate()
    pd.testing.assert_frame_equal(
        original_tax.frame.table("tax_unit"),
        attached_tax.frame.table("tax_unit"),
        check_exact=True,
    )
    assert original_tax.receipt["student_controls"] is None


@pytest.mark.parametrize("mutation", ["missing", "unknown_year", "swapped", "bytes"])
def test_unclosed_or_wrong_member_paths_refuse(tmp_path, monkeypatch, mutation):
    source, ready, members, _ = _fixture(tmp_path, monkeypatch)
    if mutation == "missing":
        members.pop(2022)
    elif mutation == "unknown_year":
        members[2025] = members.pop(2022)
    elif mutation == "swapped":
        members[2022], members[2023] = members[2023], members[2022]
    else:
        p = members[2022]
        p.write_bytes(p.read_bytes().replace(b"PERIDNUM", b"PERIDNUL"))
    with pytest.raises(money.MoneyRefusalError, match="MEMBER_PATHS|STUDENT_SOURCE"):
        _load(source, ready, members)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_control",
        "duplicate_header",
        "empty",
        "fraction",
        "domain",
        "duplicate_key",
        "duplicate_line",
        "missing_row",
        "unknown_key",
        "age",
        "household",
        "line",
        "age_universe",
        "enrolled_zero",
        "unenrolled_nonzero",
        "incumbent",
    ],
)
def test_authenticated_but_contradictory_source_cannot_issue(
    tmp_path, monkeypatch, mutation
):
    source, ready, members, _ = _fixture(tmp_path, monkeypatch)
    year = 2024 if mutation == "incumbent" else 2022
    path = members[year]
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    child = table.index[table.A_LINENO == "2"][0]
    adult = table.index[table.A_LINENO == "1"][0]
    if mutation == "missing_control":
        table = table.drop(columns="A_FTPT")
    elif mutation == "duplicate_header":
        table = table.rename(columns={"A_FTPT": "A_ENRLW"})
    elif mutation == "duplicate_key":
        table.loc[adult, "PERIDNUM"] = table.loc[child, "PERIDNUM"]
    elif mutation == "duplicate_line":
        table.loc[adult, "A_LINENO"] = "2"
    elif mutation == "missing_row":
        table = table.drop(index=child)
    else:
        name, value, row = {
            "empty": ("A_ENRLW", "", child),
            "fraction": ("A_FTPT", "1.5", child),
            "domain": ("A_FTPT", "3", child),
            "unknown_key": ("PERIDNUM", "9" * 22, child),
            "age": ("A_AGE", "21", child),
            "household": ("PH_SEQ", "9", child),
            "line": ("A_LINENO", "3", child),
            "age_universe": ("A_AGE", "15", child),
            "enrolled_zero": ("A_FTPT", "0", child),
            "unenrolled_nonzero": ("A_FTPT", "1", adult),
            "incumbent": ("A_ENRLW", "0", child),
        }[mutation]
        table.loc[row, name] = value
    table.to_csv(path, index=False)
    # Only S's private invented registry changes. T1 remains authenticated under
    # its original member pins, so this also checks the separate parent binding.
    monkeypatch.setattr(
        student,
        "_MEMBER_PINS",
        tuple(
            (
                y,
                n,
                a,
                money._sha(path.read_bytes()) if y == year else p,
                r,
                path.stat().st_size if y == year else s,
            )
            for y, n, a, p, r, s in student._MEMBER_PINS
        ),
    )
    with pytest.raises(money.MoneyRefusalError) as error:
        _load(source, ready, members)
    assert error.value.reason.startswith("STUDENT_")
    assert "999999" not in str(error.value)


def test_missing_current_incumbent_refuses(tmp_path, monkeypatch):
    source, ready, members, _ = _fixture(tmp_path, monkeypatch, incumbent_missing=True)
    with pytest.raises(money.MoneyRefusalError, match="STUDENT_INCUMBENT_INCOMPLETE"):
        _load(source, ready, members)


@pytest.mark.parametrize(
    "mutation",
    [
        "same",
        "truncate",
        "grow",
        "payload_valid_checksum",
        "header_valid_checksum",
        "alias_valid_checksum",
    ],
)
def test_candidate_admission_reconstructs_exact_canonical_bytes(
    tmp_path, monkeypatch, mutation
):
    import hashlib
    import struct

    source, ready, members, parents = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "S"
    receipt = student.write_student_controls(
        source, ready, member_paths=members, output_dir=output
    )
    path = output / student.FILENAME
    payload = path.read_bytes()
    if mutation == "truncate":
        payload = payload[:-1]
    elif mutation == "grow":
        payload += b"x"
    elif mutation != "same":
        n = struct.unpack("<I", payload[len(student.MAGIC) : len(student.MAGIC) + 4])[0]
        start = len(student.MAGIC) + 4
        header = json.loads(payload[start : start + n])
        body = bytearray(payload[start + n : -32])
        if mutation == "payload_valid_checksum":
            body[-8] ^= 1
            header["body_sha256"] = money._sha(body)
        elif mutation == "alias_valid_checksum":
            header["aliases"]["A_FTPT"] = "asec_A_FAKE"
        else:
            header["annual_five_month_student_status_validated"] = True
        encoded = money._json(header)
        payload = student.MAGIC + struct.pack("<I", len(encoded)) + encoded + body
        payload += hashlib.sha256(payload).digest()
    path.write_bytes(payload)

    # S replay accepts no HDF path at all; an accidental generic decoder is a bug.
    def forbid(*args, **kwargs):
        raise AssertionError("S replay must not decode any HDF")

    monkeypatch.setattr(checkpoint, "load_frame_checkpoint", forbid)
    if mutation == "same":
        replay = student.load_authenticated_student_controls(
            source, ready, member_paths=members, candidate_path=path
        )
        assert replay.content_sha256 == receipt["content_sha256"]
        assert money._sha(payload) == receipt["output_sha256"]
        assert replay.array("A_FTPT").tolist() == [0, 1, 0, 2, 0, 0]
        with pytest.raises(FileExistsError):
            student.write_student_controls(
                source, ready, member_paths=members, output_dir=output
            )
    else:
        with pytest.raises(
            money.MoneyRefusalError, match="STUDENT_SOURCE_SIZE|STUDENT_CANONICAL_BYTES"
        ):
            student.load_authenticated_student_controls(
                source, ready, member_paths=members, candidate_path=path
            )
    assert all(p.exists() for p in parents)


@pytest.mark.parametrize(
    "mutation",
    [
        "alias_value",
        "old_missing",
        "person_axis",
        "columns",
        "weight",
        "metadata",
        "history",
        "extra_column",
    ],
)
def test_owned_composition_and_tax_result_detect_mutation(
    tmp_path, monkeypatch, mutation
):
    source, ready, members, _ = _fixture(tmp_path, monkeypatch)
    original = legacy._frame_signature(source.frame)
    controls = _load(source, ready, members)
    combined = student.attach_student_controls(source, ready, controls)
    tax = units.reconstruct_current_money_tax_units(combined, ready)
    if mutation == "alias_value":
        combined.frame.person.iloc[
            0, combined.frame.person.columns.get_loc("asec_A_ENRLW")
        ] = 1
    elif mutation == "old_missing":
        combined.frame.person.iloc[
            0, combined.frame.person.columns.get_loc("A_ENRLW")
        ] = 0
    elif mutation == "person_axis":
        combined.frame.person.index.array[0] = 99
    elif mutation == "columns":
        combined.frame.person.columns.array[-1] = "forged_alias"
    elif mutation == "weight":
        values = combined.frame.weights_for("household").values
        values.setflags(write=True)
        values[0] = 7
    elif mutation == "metadata":
        object.__setattr__(combined.frame, "_metadata", {"extra": True})
    elif mutation == "history":
        object.__setattr__(combined.frame, "_mass_log", ("changed",))
    else:
        combined.frame.person["extra"] = 0
    assert legacy._frame_signature(source.frame) == original
    source.validate()
    with pytest.raises(money.MoneyRefusalError):
        combined.validate()
    with pytest.raises(money.MoneyRefusalError):
        tax.validate()


def test_immutable_buffers_constructor_and_cross_money_parent_refuse(
    tmp_path, monkeypatch
):
    source, ready, members, parents = _fixture(tmp_path, monkeypatch)
    controls = _load(source, ready, members)
    with pytest.raises(ValueError):
        controls.array("A_ENRLW").setflags(write=True)
    with pytest.raises(money.MoneyRefusalError, match="STUDENT_CONSTRUCTOR"):
        student.AuthenticatedStudentControls(controls._header, controls._body)
    with pytest.raises(money.MoneyRefusalError):
        replace(controls, _body=controls._body)
    old = legacy.load_authenticated_current_money_source(*parents[:2])
    with pytest.raises(money.MoneyRefusalError):
        student.attach_student_controls(old, ready, controls)
    fields = list(ready.fields)
    changed = fields[0].amounts.copy()
    changed[0] += 1
    fields[0] = replace(fields[0], amount_bytes=changed.tobytes())
    forged = money.ReadyCurrentMoney(
        ready.bindings, tuple(fields), _token=money._STATE_TOKEN
    )
    with pytest.raises(money.MoneyRefusalError):
        student.attach_student_controls(source, forged, controls)


def test_private_csv_and_candidate_snapshots_survive_path_replacement(
    tmp_path, monkeypatch
):
    source, ready, members, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "S"
    student.write_student_controls(
        source, ready, member_paths=members, output_dir=output
    )
    candidate = output / student.FILENAME
    read = student._read_member
    seen = []

    def private(path, **kwargs):
        assert path not in members.values()
        seen.append(path)
        return read(path, **kwargs)

    monkeypatch.setattr(student, "_read_member", private)
    snapshot = student._snapshot

    def capture(path, destination, **kwargs):
        digest = snapshot(path, destination, **kwargs)
        if Path(path) == candidate:
            candidate.write_bytes(b"replaced after private capture")
        return digest

    monkeypatch.setattr(student, "_snapshot", capture)
    value = student.load_authenticated_student_controls(
        source, ready, member_paths=members, candidate_path=candidate
    )
    value.validate()
    assert len(seen) == 3 and all(not p.exists() for p in seen)
    with pytest.raises(money.MoneyRefusalError):
        student.load_authenticated_student_controls(
            source, ready, member_paths=members, candidate_path=candidate
        )
