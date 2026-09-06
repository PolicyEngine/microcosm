"""Actual T1 plus invented PERSON members; no Census data or model execution."""

import csv
import importlib.util
import json
import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_source as legacy
from microcosm.build.us_runtime import asec_income_observations as income
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime._asec_current_money_codec import encode_current_money


def invented_income_sources(tmp_path, monkeypatch, *, incumbent="known"):
    spec = importlib.util.spec_from_file_location(
        "income_t1_fixture",
        Path(__file__).with_name("test_us_asec_person_income_source.py"),
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    parent, household, members = fixture.invented_sources(tmp_path, monkeypatch)
    ages = [55, 14, 55, 20, 55, 20]
    paw = [60000, 0, 0, 0, 0, 0]
    yn = [1, 0, 1, 2, 0, 1]
    for path in (parent, household):
        loaded = load_frame_checkpoint(path)
        loaded.frame.person["A_AGE"] = np.asarray(ages, dtype="int64")
        loaded.frame.person["WSAL_VAL"] = np.asarray(paw, dtype="float64")
        if incumbent == "bool":
            loaded.frame.person["PAW_VAL"] = np.asarray(
                [bool(v) for v in paw], dtype="bool"
            )
        elif incumbent == "nullable_integer":
            loaded.frame.person["PAW_VAL"] = pd.array(paw, dtype="Int64")
        elif incumbent != "missing_column":
            loaded.frame.person["PAW_VAL"] = np.asarray(
                [np.nan] * 6 if incumbent == "unknown" else paw, dtype="float64"
            )
        if path == household:
            digest = money._sha(parent.read_bytes())
            loaded.metadata["parent_checkpoint_sha256"] = digest
            loaded.metadata["household_observations"]["input_checkpoint_sha256"] = (
                digest
            )
        write_frame_checkpoint(path, loaded.frame, metadata=loaded.metadata)
    monkeypatch.setattr(
        legacy,
        "_SOURCE_PINS",
        (
            money._sha(parent.read_bytes()),
            money._sha(household.read_bytes()),
            legacy._SOURCE_PINS[2],
        ),
    )
    for _year, path in members.items():
        table = pd.read_csv(path, dtype=str)
        for row in range(len(table)):
            position = int(table.loc[row, "PERIDNUM"]) - 1
            table.loc[row, "A_AGE"] = str(ages[position])
            table.loc[row, "PAW_VAL"] = str(paw[position])
            table.loc[row, "PAW_YN"] = str(yn[position])
        table.to_csv(path, index=False)
    pins = tuple(
        (y, n, a, money._sha(members[y].read_bytes()), r, members[y].stat().st_size)
        for y, n, a, p, r, s in restoration._MEMBER_PINS
    )
    monkeypatch.setattr(restoration, "_MEMBER_PINS", pins)
    monkeypatch.setattr(income, "_MEMBER_PINS", pins)
    output = tmp_path / "t1"
    restoration.restore_asec_person_income_source(
        parent, household, member_paths=members, output_dir=output
    )
    source = restoration.load_authenticated_restored_current_money_source(
        parent,
        household,
        output / restoration.CHECKPOINT_FILENAME,
        member_paths=members,
    )
    return source, source.ready(), members


def _repin(monkeypatch, members):
    monkeypatch.setattr(
        income,
        "_MEMBER_PINS",
        tuple(
            (y, n, a, money._sha(members[y].read_bytes()), r, members[y].stat().st_size)
            for y, n, a, p, r, s in income._MEMBER_PINS
        ),
    )


def test_full_join_price_bits_zero_classes_and_sealed_parent(tmp_path, monkeypatch):
    source, ready, members = invented_income_sources(tmp_path, monkeypatch)
    frame_before = legacy._frame_signature(source.frame)
    money_before = encode_current_money(ready)
    value = income.load_authenticated_income_observations(
        source, ready, member_paths=members
    )
    assert value.array("PAW_VAL").tolist() == [60000, 0, 0, 0, 0, 0]
    assert value.array("PAW_YN").tolist() == [1, 0, 1, 2, 0, 1]
    assert (
        value.array("PAW_VAL_2024_price").tobytes()
        == ready.field("WSAL_VAL").amount_bytes
    )
    assert income.zero_origin_classes(
        value.array("A_AGE"), value.array("PAW_YN"), value.array("PAW_VAL")
    ).tolist() == [0, 1, 2, 3, 4, 2]
    assert value.receipt["sources"][2]["paw_yn_zero_at_age_15_plus"] == 1
    assert [
        s["incumbent_compared_rows"]["PAW_VAL"] for s in value.receipt["sources"]
    ] == [2, 2, 2]
    assert value.receipt["monetary_basis"]["2022"]["period"] == "2022"
    assert value.receipt["release_eligible"] is False
    assert frame_before == legacy._frame_signature(source.frame)
    assert money_before == encode_current_money(ready)
    assert "PAW_YN" not in source.frame.person


@pytest.mark.parametrize("incumbent", ["unknown", "missing_column"])
def test_unknown_incumbent_is_counted_but_missing_column_refuses(
    tmp_path, monkeypatch, incumbent
):
    source, ready, members = invented_income_sources(
        tmp_path, monkeypatch, incumbent=incumbent
    )
    if incumbent == "missing_column":
        with pytest.raises(money.MoneyRefusalError, match="INCOME_INCUMBENT_DTYPE"):
            income.load_authenticated_income_observations(
                source, ready, member_paths=members
            )
    else:
        value = income.load_authenticated_income_observations(
            source, ready, member_paths=members
        )
        assert [
            s["incumbent_compared_rows"]["PAW_VAL"] for s in value.receipt["sources"]
        ] == [0, 0, 0]
        assert value.array("PAW_VAL")[0] == 60000


@pytest.mark.parametrize(
    "column,token",
    [
        ("PAW_VAL", ""),
        ("PAW_VAL", "1.5"),
        ("PAW_VAL", "-1"),
        ("PAW_VAL", "100000"),
        ("PAW_VAL", "9" * 40),
        ("PAW_YN", "3"),
        ("PAW_YN", "2.0"),
        ("PERIDNUM", "9" * 22),
        ("A_AGE", "19"),
        ("PH_SEQ", "8"),
        ("A_LINENO", "7"),
    ],
)
def test_changed_authenticated_source_tokens_refuse(
    tmp_path, monkeypatch, column, token
):
    source, ready, members = invented_income_sources(tmp_path, monkeypatch)
    path = members[2024]
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table.loc[0, column] = token
    table.to_csv(path, index=False)
    _repin(monkeypatch, members)
    with pytest.raises(money.MoneyRefusalError) as error:
        income.load_authenticated_income_observations(
            source, ready, member_paths=members
        )
    assert str(path) not in str(error.value)
    assert "999999" not in str(error.value)


@pytest.mark.parametrize(
    "year,person,amount,code",
    [
        (2022, 1, 1, 0),
        (2022, 1, 0, 1),
        (2023, 0, 5, 2),
        (2023, 0, 5, 0),
    ],
)
def test_receipt_age_universe_contradictions_refuse(
    tmp_path, monkeypatch, year, person, amount, code
):
    source, ready, members = invented_income_sources(
        tmp_path, monkeypatch, incumbent="unknown"
    )
    table = pd.read_csv(members[year], dtype=str)
    # Source fixture is reversed: select by native key rather than its row index.
    key = str((year - 2022) * 2 + person + 1).zfill(22)
    table.loc[table.PERIDNUM == key, ["PAW_VAL", "PAW_YN"]] = [str(amount), str(code)]
    table.to_csv(members[year], index=False)
    _repin(monkeypatch, members)
    with pytest.raises(money.MoneyRefusalError, match="INCOME_SOURCE_UNIVERSE"):
        income.load_authenticated_income_observations(
            source, ready, member_paths=members
        )


def test_all_cohort_incumbents_are_checked(tmp_path, monkeypatch):
    source, ready, members = invented_income_sources(tmp_path, monkeypatch)
    table = pd.read_csv(members[2022], dtype=str)
    table.loc[table.PAW_VAL == "60000", "PAW_VAL"] = "60001"
    table.to_csv(members[2022], index=False)
    _repin(monkeypatch, members)
    with pytest.raises(money.MoneyRefusalError, match="INCOME_INCUMBENT_CONFLICT"):
        income.load_authenticated_income_observations(
            source, ready, member_paths=members
        )


def test_owned_binary_layout_replay_and_fully_rechecksummed_forgery(
    tmp_path, monkeypatch
):
    source, ready, members = invented_income_sources(tmp_path, monkeypatch)
    output = tmp_path / "income"
    receipt = income.write_income_observations(
        source, ready, member_paths=members, output_dir=output
    )
    candidate = output / income.FILENAME
    value = income.load_authenticated_income_observations(
        source, ready, member_paths=members, candidate_path=candidate
    )
    payload = candidate.read_bytes()
    assert len(payload) <= income.INCOME_PAYLOAD_MAX_BYTES
    assert payload.startswith(b"MCAINCOB\x01")
    assert money._sha(payload) == receipt["output_sha256"]
    for name in income.COLUMNS:
        with pytest.raises(ValueError):
            value.array(name).setflags(write=True)
    changed_receipt = value.receipt
    changed_receipt["rows"] = 999
    assert value.receipt["rows"] == 6
    with pytest.raises(money.MoneyRefusalError, match="CONSTRUCTOR_UNAVAILABLE"):
        income.AuthenticatedIncomeObservations(value._header, value._body)
    with pytest.raises(FileExistsError):
        income.write_income_observations(
            source, ready, member_paths=members, output_dir=output
        )
    header_length = struct.unpack_from("<I", payload, len(income.MAGIC))[0]
    offset = len(income.MAGIC) + 4
    header = json.loads(payload[offset : offset + header_length])
    body = bytearray(payload[offset + header_length : -32])
    # Forge a semantically valid different age, closing every inner and outer digest.
    struct.pack_into("<q", body, 4 * 6 * 8, 56)
    header["body_sha256"] = money._sha(body)
    header["column_sha256"]["A_AGE"] = money._sha(body[4 * 6 * 8 : 5 * 6 * 8])
    encoded_header = money._json(header)
    forged = (
        income.MAGIC
        + struct.pack("<I", len(encoded_header))
        + encoded_header
        + bytes(body)
    )
    candidate.write_bytes(forged + bytes.fromhex(money._sha(forged)))
    with pytest.raises(money.MoneyRefusalError, match="INCOME_CANONICAL_BYTES"):
        income.load_authenticated_income_observations(
            source, ready, member_paths=members, candidate_path=candidate
        )


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "size"])
def test_candidate_paths_are_bounded_before_reading(tmp_path, monkeypatch, kind):
    source, ready, members = invented_income_sources(tmp_path, monkeypatch)
    candidate = tmp_path / "candidate"
    if kind == "symlink":
        candidate.symlink_to(members[2022])
    elif kind == "fifo":
        os.mkfifo(candidate)
    elif kind == "directory":
        candidate.mkdir()
    else:
        candidate.write_bytes(b"short")
    with pytest.raises(money.MoneyRefusalError, match="SOURCE_FILE_KIND|SOURCE_SIZE"):
        income.load_authenticated_income_observations(
            source, ready, member_paths=members, candidate_path=candidate
        )


def test_member_change_during_capture_refuses(tmp_path, monkeypatch):
    source, ready, members = invented_income_sources(tmp_path, monkeypatch)
    capture = income._snapshot

    def changed(path, destination, *, size):
        result = capture(path, destination, size=size)
        os.utime(path, ns=(1, 1))
        return result

    monkeypatch.setattr(income, "_snapshot", changed)
    with pytest.raises(money.MoneyRefusalError, match="INCOME_SOURCE_CHANGED"):
        income.load_authenticated_income_observations(
            source, ready, member_paths=members
        )


def test_live_csv_parser_profile_is_bound_into_the_source_identity(
    tmp_path, monkeypatch
):
    """Raw PAW API: a changed csv limit is a different identity, not a silent reparse."""
    source, ready, members = invented_income_sources(tmp_path, monkeypatch)
    limit = csv.field_size_limit()
    baseline = income.load_authenticated_income_observations(
        source, ready, member_paths=members
    )
    assert baseline.receipt["implementation"]["parser_profile"] == {
        "csv.field_size_limit": limit,
        "delimiter": ",",
        "dialect": "excel",
        "doublequote": True,
        "encoding": "utf-8",
        "header_engine": "stdlib_csv_reader",
        "newline": "",
        "quotechar": '"',
        "quoting": 0,
        "skipinitialspace": False,
        "strict": False,
        "table_engine": "pandas_read_csv",
    }
    try:
        csv.field_size_limit(limit * 2)
        changed = income.load_authenticated_income_observations(
            source, ready, member_paths=members
        )
        assert (
            changed.receipt["implementation"]["parser_profile"]["csv.field_size_limit"]
            == limit * 2
        )
        assert changed.content_sha256 != baseline.content_sha256
        assert changed._body == baseline._body
        # The frozen artifact refuses to validate under the changed profile.
        with pytest.raises(money.MoneyRefusalError, match="INCOME_CONTENT_CHANGED"):
            baseline.validate()
    finally:
        csv.field_size_limit(limit)
    restored = income.load_authenticated_income_observations(
        source, ready, member_paths=members
    )
    assert restored.content_sha256 == baseline.content_sha256
    baseline.validate()


def test_changing_the_parser_limit_mid_reconstruction_refuses(tmp_path, monkeypatch):
    source, ready, members = invented_income_sources(tmp_path, monkeypatch)
    limit = csv.field_size_limit()
    original = income._read_member

    def widened(path, *, rows):
        table = original(path, rows=rows)
        csv.field_size_limit(limit * 2)
        return table

    monkeypatch.setattr(income, "_read_member", widened)
    try:
        with pytest.raises(
            money.MoneyRefusalError, match="INCOME_IMPLEMENTATION_CHANGED"
        ):
            income.load_authenticated_income_observations(
                source, ready, member_paths=members
            )
    finally:
        csv.field_size_limit(limit)


def test_unparsable_member_under_a_narrow_limit_refuses_the_source_contract(
    tmp_path, monkeypatch
):
    source, ready, members = invented_income_sources(tmp_path, monkeypatch)
    limit = csv.field_size_limit()
    try:
        csv.field_size_limit(3)
        with pytest.raises(money.MoneyRefusalError, match="INCOME_SOURCE_CONTRACT"):
            income.load_authenticated_income_observations(
                source, ready, member_paths=members
            )
    finally:
        csv.field_size_limit(limit)


def test_create_time_price_basis_crosscheck(tmp_path, monkeypatch):
    """The resource price cells and pin are checked at CREATE, not only at bind."""
    source, ready, members = invented_income_sources(tmp_path, monkeypatch)
    contract = income._contract()
    assert contract["price"]["resource_sha256"] == money.RESOURCE_PINS[1]
    assert income.price_factors(contract) == tuple(
        tuple(f) for f in ready.bindings.spec.factors
    )
    changed = json.loads(json.dumps(contract))
    changed["price"]["cells"]["2022"] = "163.7"
    monkeypatch.setattr(income, "_contract", lambda: changed)
    with pytest.raises(money.MoneyRefusalError, match="INCOME_PRICE_BASIS"):
        income.load_authenticated_income_observations(
            source, ready, member_paths=members
        )
    unpinned = json.loads(json.dumps(contract))
    unpinned["price"]["resource_sha256"] = "f" * 64
    monkeypatch.setattr(income, "_contract", lambda: unpinned)
    with pytest.raises(money.MoneyRefusalError, match="INCOME_PRICE_BASIS"):
        income.load_authenticated_income_observations(
            source, ready, member_paths=members
        )


@pytest.mark.parametrize("incumbent", ["bool", "nullable_integer"])
def test_bool_and_nullable_integer_incumbent_dtypes(tmp_path, monkeypatch, incumbent):
    source, ready, members = invented_income_sources(
        tmp_path, monkeypatch, incumbent=incumbent
    )
    if incumbent == "bool":
        with pytest.raises(money.MoneyRefusalError, match="INCOME_INCUMBENT_DTYPE"):
            income.load_authenticated_income_observations(
                source, ready, member_paths=members
            )
        return
    value = income.load_authenticated_income_observations(
        source, ready, member_paths=members
    )
    assert value.array("PAW_VAL").tolist() == [60000, 0, 0, 0, 0, 0]
