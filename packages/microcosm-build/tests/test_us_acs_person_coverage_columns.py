"""Literal coverage projections from invented, multi-member ACS archives."""

import csv
import hashlib
import io
import json
from zipfile import ZipFile

import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_person_coverage_columns as module
from microcosm.build.us_runtime import acs_pums


def _row(serial="2024HU0000001", order="1", age="30", mil="4", esr="6"):
    return dict(zip(module.READ_COLUMNS, (serial, order, age, mil, esr), strict=True))


def _archive(tmp_path, members):
    path = tmp_path / "persons.zip"
    with ZipFile(path, "w") as archive:
        for name, rows in members.items():
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(
                stream, fieldnames=module.READ_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            archive.writestr(name, stream.getvalue())
    return acs_pums.AcsPumsSource(
        tmp_path / "unused-households.zip", path, max_households=1
    )


def _keys(rows):
    return pd.DataFrame(rows).loc[:, list(module.KEYS)]


def test_private_scanner_exhausts_members_despite_consumer_return_value(tmp_path):
    first = _row(order="02", age="16", mil="", esr="1")
    second = _row(order="01", age="17", mil="4", esr=" 6")
    third = _row("2024GQ0000001", age="", mil='x,"\t', esr="")
    source = _archive(
        tmp_path,
        {"psam_pusb.csv": [second, third], "psam_pusa.csv": [first]},
    )
    received = []

    def consume(cells):
        received.append(cells)
        return True  # No return value can turn a prefix into a successful scan.

    members, rows = module._scan_acs_person_coverage(source, consume)
    assert members == ["psam_pusa.csv", "psam_pusb.csv"]
    assert rows == 3
    assert received == [
        ("2024HU0000001", "02", "16", "", "1"),
        ("2024HU0000001", "01", "17", "4", " 6"),
        ("2024GQ0000001", "1", "", 'x,"\t', ""),
    ]


@pytest.mark.parametrize(
    "tail,match",
    [
        ("2024HU0000009,1,30,4,6,EXTRA\n", "record width"),
        ("2024HU0000009,1,30,4,6\x00\n", "forbidden control"),
        ('2024HU0000009,1,30,"unfinished,6\n', "invalid literal CSV"),
    ],
)
def test_private_scanner_refuses_invalid_tail_after_delivering_expected_rows(
    tmp_path, tail, match
):
    source = _raw_archive(
        tmp_path,
        "SERIALNO,SPORDER,AGEP,MIL,ESR\n2024HU0000001,1,30,4,6\n" + tail,
    )
    received = []
    with pytest.raises(ValueError, match=match):
        module._scan_acs_person_coverage(source, received.append)
    assert received == [("2024HU0000001", "1", "30", "4", "6")]


def test_exact_household_roster_across_members_preserves_blanks_and_request_order(
    tmp_path,
):
    rows = [
        _row(mil="1", esr="4"),
        _row(order="2", age="10", mil="", esr=""),
        _row("2024GQ0000001", age="16", mil="", esr="1"),
    ]
    unrelated = _row("2024HU0000009")
    source = _archive(
        tmp_path,
        {
            "nested/psam_pusb.csv": [rows[0], unrelated],
            "psam_pusa.csv": [rows[2], rows[1]],
        },
    )
    requested = _keys([rows[1], rows[2], rows[0]])
    requested["SPORDER"] = requested.SPORDER.astype("int64")
    result, receipt = module.read_acs_person_coverage_columns(
        source, person_keys=requested, chunksize=1
    )
    assert result.MIL.tolist() == ["", "", "1"]
    assert result.ESR.tolist() == ["", "1", "4"]
    assert result.MIL_state.tolist() == [
        "outside_age_universe",
        "outside_age_universe",
        "observed_code",
    ]
    assert result.ESR_state.tolist() == [
        "outside_age_universe",
        "observed_code",
        "observed_code",
    ]
    assert result.SERIALNO.tolist() == requested.SERIALNO.tolist()
    assert result.SPORDER.tolist() == requested.SPORDER.tolist()
    assert not result.isna().any().any()
    assert receipt["source_rows_streamed"] == 4
    assert receipt["selected_person_rows"] == 3
    assert receipt["source_authenticated"] is False
    assert receipt["release_eligible"] is False
    assert receipt["legacy_max_households_applied"] is False
    assert receipt["coverage_status"] == "literal_source_fields_only"
    second, other = module.read_acs_person_coverage_columns(
        source, person_keys=requested, chunksize=3
    )
    pd.testing.assert_frame_equal(result, second)
    assert other == receipt


@pytest.mark.parametrize(
    "age,mil,esr,mil_state,esr_state",
    [
        ("17", "2", "2", "observed_code", "observed_code"),
        ("45", "3", "5", "observed_code", "observed_code"),
        ("30", "", "", "missing_in_universe", "missing_in_universe"),
        ("30", "0", "7", "unlabelled_code", "unlabelled_code"),
        ("30", "NA", "1.0", "unlabelled_code", "unlabelled_code"),
        ("30", " 1", "4 ", "unlabelled_code", "unlabelled_code"),
        ("15", "1", "4", "value_below_age_universe", "value_below_age_universe"),
        ("", "1", "4", "age_unresolved", "age_unresolved"),
        ("100", "4", "6", "age_unresolved", "age_unresolved"),
    ],
)
def test_unresolved_states_preserve_literal_source_tokens(
    tmp_path, age, mil, esr, mil_state, esr_state
):
    row = _row(age=age, mil=mil, esr=esr)
    source = _archive(tmp_path, {"psam_pusa.csv": [row]})
    table, _ = module.read_acs_person_coverage_columns(source, person_keys=_keys([row]))
    assert table.loc[0, ["AGEP", "MIL", "ESR"]].tolist() == [age, mil, esr]
    assert table.MIL_state.tolist() == [mil_state]
    assert table.ESR_state.tolist() == [esr_state]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("extra", "extra selected-household"),
        ("missing", "native roster differ"),
        ("duplicate", "keys repeat"),
        ("wrong_order", "native roster differ"),
    ],
)
def test_partial_household_or_conflicting_native_roster_refuses(
    tmp_path, mutation, match
):
    rows = [_row(), _row(order="2")]
    observed = rows.copy()
    if mutation == "extra":
        observed += [_row(order="3")]
    elif mutation == "missing":
        observed.pop()
    elif mutation == "duplicate":
        observed[1] = rows[0]
    else:
        observed[1] = _row(order="3")
    source = _archive(tmp_path, {"psam_pusa.csv": observed})
    with pytest.raises(ValueError, match=match):
        module.read_acs_person_coverage_columns(
            source, person_keys=_keys(rows), chunksize=1
        )


@pytest.mark.parametrize(
    "serial,order",
    [
        ("2023HU0000001", "1"),
        ("2024HU0000001", "0"),
        ("2024HU0000001", "21"),
        ("2024HU0000001", "1.0"),
        ("2024HU0000001\x00other", "1"),
        ("2024HU0000001", "1\x00other"),
        (None, "1"),
        ("2024HU0000001", True),
    ],
)
def test_invalid_native_keys_refuse_before_file_read(tmp_path, serial, order):
    source = acs_pums.AcsPumsSource(
        tmp_path / "absent-h.zip", tmp_path / "absent-p.zip"
    )
    with pytest.raises(ValueError, match="native"):
        module.read_acs_person_coverage_columns(
            source, person_keys=pd.DataFrame({"SERIALNO": [serial], "SPORDER": [order]})
        )


@pytest.mark.parametrize(
    "mutation", ["missing_column", "duplicate_column", "duplicate_member"]
)
def test_ambiguous_source_schema_refuses(tmp_path, mutation):
    source = _archive(tmp_path, {"psam_pusa.csv": [_row()]})
    with ZipFile(source.person_zip, "r") as archive:
        raw = archive.read("psam_pusa.csv")
    if mutation == "missing_column":
        raw = raw.replace(b"MIL", b"OTHER")
    elif mutation == "duplicate_column":
        raw = raw.replace(b"ESR", b"MIL")
    with ZipFile(source.person_zip, "w") as archive:
        archive.writestr("psam_pusa.csv", raw)
        if mutation == "duplicate_member":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("psam_pusa.csv", raw)
    with pytest.raises(ValueError, match="missing or duplicated"):
        module.read_acs_person_coverage_columns(source, person_keys=_keys([_row()]))


def test_explicit_row_bound_refuses_without_truncation(tmp_path, monkeypatch):
    row = _row()
    source = _archive(tmp_path, {"psam_pusa.csv": [row, _row("2024HU0000002")]})
    monkeypatch.setattr(module, "MAX_ROWS", 1)
    with pytest.raises(ValueError, match="row bound"):
        module.read_acs_person_coverage_columns(
            source, person_keys=_keys([row]), chunksize=1
        )


def test_selected_row_ceiling_refuses_before_opening_the_source(tmp_path, monkeypatch):
    source = acs_pums.AcsPumsSource(
        tmp_path / "absent-h.zip", tmp_path / "absent-p.zip"
    )
    monkeypatch.setattr(module, "MAX_SELECTED_ROWS", 1)
    with pytest.raises(ValueError, match="selected person count is outside the bound"):
        module.read_acs_person_coverage_columns(
            source, person_keys=_keys([_row(), _row(order="2")])
        )


def test_contract_is_owned_and_separate_from_legacy_native_roster():
    contract = module.coverage_field_contract()
    assert contract["fields"]["MIL"]["minimum_age"] == 17
    assert contract["fields"]["ESR"]["minimum_age"] == 16
    contract["fields"]["MIL"]["codes"].clear()
    assert len(module.coverage_field_contract()["fields"]["MIL"]["codes"]) == 4
    assert not {"MIL", "ESR"}.intersection(acs_pums._PERSON_REQUIRED)


def _raw_archive(tmp_path, raw):
    source = acs_pums.AcsPumsSource(
        tmp_path / "unused-households.zip", tmp_path / "literal-persons.zip"
    )
    with ZipFile(source.person_zip, "w") as archive:
        archive.writestr("psam_pusa.csv", raw)
    return source


@pytest.mark.parametrize("field", module.READ_COLUMNS)
def test_nul_suffix_refuses_before_truncating_value_or_identity(tmp_path, field):
    row = _row(mil="1")
    row[field] += "\x00garbage"
    source = _archive(tmp_path, {"psam_pusa.csv": [row]})
    with pytest.raises(ValueError, match="control"):
        module.read_acs_person_coverage_columns(source, person_keys=_keys([_row()]))


@pytest.mark.parametrize(
    "control",
    [chr(code) for code in range(32) if code not in (9, 10, 13)]
    + [chr(code) for code in range(127, 160)],
)
def test_non_csv_control_characters_refuse(tmp_path, control):
    source = _archive(tmp_path, {"psam_pusa.csv": [_row(mil="1" + control)]})
    with pytest.raises(ValueError, match="control"):
        module.read_acs_person_coverage_columns(source, person_keys=_keys([_row()]))


@pytest.mark.parametrize("location", ["header", "ignored_column", "unselected_row"])
def test_control_validation_precedes_column_and_household_selection(tmp_path, location):
    header = "SERIALNO,SPORDER,AGEP,MIL,ESR,OTHER\n"
    body = "2024HU0000001,1,30,1,6,ok\n"
    if location == "header":
        header = header.replace("OTHER", "OTHER\x00suffix")
    elif location == "ignored_column":
        body = body.replace("ok", "ok\x00suffix")
    else:
        body += "2024HU0000009,1,30,1\x00suffix,6,ok\n"
    source = _raw_archive(tmp_path, header + body)
    with pytest.raises(ValueError, match="control"):
        module.read_acs_person_coverage_columns(source, person_keys=_keys([_row()]))


@pytest.mark.parametrize("serial", ["2024HU0000001", "2024HU0000009"])
@pytest.mark.parametrize(
    "tail",
    ["1,30,4", "1,30,4,6", "1,30,4,6,ok,extra", "1,30,4,6,ok,"],
    ids=["missing_required", "missing_unused", "too_many", "trailing_delimiter"],
)
def test_full_record_width_refuses_even_outside_projection(tmp_path, serial, tail):
    source = _raw_archive(
        tmp_path,
        "SERIALNO,SPORDER,AGEP,MIL,ESR,OTHER\n" + serial + "," + tail + "\n",
    )
    with pytest.raises(ValueError, match="record width"):
        module.read_acs_person_coverage_columns(source, person_keys=_keys([_row()]))


@pytest.mark.parametrize(
    "body",
    ["\n", '2024HU0000001,1,30,4,"unterminated\n', '2024HU0000001,1,30,"1"garbage,6\n'],
)
def test_blank_records_and_malformed_quoting_refuse(tmp_path, body):
    source = _raw_archive(tmp_path, "SERIALNO,SPORDER,AGEP,MIL,ESR\n" + body)
    with pytest.raises(ValueError, match="CSV|record width"):
        module.read_acs_person_coverage_columns(source, person_keys=_keys([_row()]))


@pytest.mark.parametrize("line_end", ["\n", "\r", "\r\n"])
def test_quoted_delimiters_newlines_tabs_and_digest_preserve_tokens(tmp_path, line_end):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator=line_end)
    writer.writerow(["OTHER", "ESR", "MIL", "AGEP", "SPORDER", "SERIALNO"])
    writer.writerow(
        [
            'ignored, "quoted"\r\nfield',
            "4\r\n\t",
            '1,"quoted"\n\r',
            "30",
            "01",
            "2024HU0000001",
        ]
    )
    source = _raw_archive(tmp_path, "\ufeff" + stream.getvalue())
    table, receipt = module.read_acs_person_coverage_columns(
        source, person_keys=_keys([_row()]), chunksize=1
    )
    expected = [
        {
            "SERIALNO": "2024HU0000001",
            "SPORDER": 1,
            "AGEP": "30",
            "MIL": '1,"quoted"\n\r',
            "ESR": "4\r\n\t",
            "MIL_state": "unlabelled_code",
            "ESR_state": "unlabelled_code",
        }
    ]
    assert table.to_dict("records") == expected
    assert receipt["source_rows_streamed"] == 1

    def digest(records):
        return hashlib.sha256(
            json.dumps(
                records, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    assert receipt["projection_sha256"] == digest(expected)
    for token in ("4\n\t", "4\r\t", r"4\r\n\t", "4"):
        assert receipt["projection_sha256"] != digest([{**expected[0], "ESR": token}])


@pytest.mark.parametrize("field", ["AGEP", "MIL", "ESR"])
@pytest.mark.parametrize("control", ["\t", "\r", "\n", "\r\n"])
def test_accepted_csv_whitespace_is_literal_and_never_an_observed_code(
    tmp_path, field, control
):
    row = _row(mil="1", esr="4")
    row[field] += control
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=module.READ_COLUMNS, quoting=csv.QUOTE_ALL
    )
    writer.writeheader()
    writer.writerow(row)
    source = _raw_archive(tmp_path, stream.getvalue())
    table, _ = module.read_acs_person_coverage_columns(
        source, person_keys=_keys([_row()])
    )
    assert table.loc[0, field] == row[field]
    if field == "AGEP":
        assert (
            table.MIL_state.tolist() == table.ESR_state.tolist() == ["age_unresolved"]
        )
    else:
        assert table[field + "_state"].tolist() == ["unlabelled_code"]


@pytest.mark.parametrize("field", module.KEYS)
@pytest.mark.parametrize("suffix", ["\t", "\n", "\r\n", " other", ",other"])
def test_literal_identity_suffix_cannot_join_canonical_requested_key(
    tmp_path, field, suffix
):
    row = _row()
    row[field] += suffix
    source = _archive(tmp_path, {"psam_pusa.csv": [row]})
    with pytest.raises(ValueError, match="missing|native"):
        module.read_acs_person_coverage_columns(source, person_keys=_keys([_row()]))


def test_duplicate_canonical_person_order_in_later_member_refuses(tmp_path):
    source = _archive(
        tmp_path,
        {"psam_pusa.csv": [_row(order="01")], "psam_pusb.csv": [_row(order="1")]},
    )
    with pytest.raises(ValueError, match="keys repeat"):
        module.read_acs_person_coverage_columns(
            source, person_keys=_keys([_row(), _row(order="2")]), chunksize=1
        )


def test_invalid_utf8_refuses(tmp_path):
    source = _raw_archive(
        tmp_path, b"SERIALNO,SPORDER,AGEP,MIL,ESR\n2024HU0000001,1,30,1\xff,6\n"
    )
    with pytest.raises(ValueError):
        module.read_acs_person_coverage_columns(source, person_keys=_keys([_row()]))


@pytest.mark.parametrize("multiline", [False, True])
def test_record_character_bound_includes_all_quoted_physical_lines(
    tmp_path, monkeypatch, multiline
):
    monkeypatch.setattr(module, "MAX_CSV_RECORD_CHARS", 100, raising=False)
    suffix = ("x\n" if multiline else "xx") * 60
    source = _archive(tmp_path, {"psam_pusa.csv": [_row(mil=suffix)]})
    with pytest.raises(ValueError, match="record character bound"):
        module.read_acs_person_coverage_columns(source, person_keys=_keys([_row()]))


@pytest.mark.parametrize("line_end", ["", "\n", "\r\n"])
def test_record_character_bound_accepts_exact_limit_and_resets_per_record(
    tmp_path, monkeypatch, line_end
):
    monkeypatch.setattr(module, "MAX_CSV_RECORD_CHARS", 100)
    prefix, suffix = '2024HU0000001,1,30,"', '",6' + line_end
    token = "x\n" + "x" * (100 - len(prefix) - len(suffix) - 2)
    body = prefix + token + suffix
    assert len(body) == 100
    source = _raw_archive(tmp_path, "SERIALNO,SPORDER,AGEP,MIL,ESR\n" + body)
    table, receipt = module.read_acs_person_coverage_columns(
        source, person_keys=_keys([_row()])
    )
    assert table.MIL.tolist() == [token]
    assert receipt["source_rows_streamed"] == 1


def test_stream_opens_member_once_and_stops_at_refusal_without_full_validation(
    tmp_path, monkeypatch
):
    source = _raw_archive(
        tmp_path,
        "SERIALNO,SPORDER,AGEP,MIL,ESR\n"
        + "2024HU0000001,1,30,4,6\n" * 2
        + "2024HU0000009,1,30,4,6\n" * 20_000,
    )
    opened, consumed = [], []

    class GuardedZipFile(ZipFile):
        def open(self, name, *args, **kwargs):
            opened.append(name)
            member = super().open(name, *args, **kwargs)
            original_read, original_read1 = member.read, member.read1

            def bounded(method, size=-1):
                assert size >= 0, "unbounded archive read"
                result = method(size)
                consumed.append(len(result))
                return result

            member.read = lambda size=-1: bounded(original_read, size)
            member.read1 = lambda size=-1: bounded(original_read1, size)
            return member

        def read(self, *args, **kwargs):
            pytest.fail("whole-member validation read")

    monkeypatch.setattr(module, "ZipFile", GuardedZipFile)
    with pytest.raises(ValueError, match="extra selected-household"):
        module.read_acs_person_coverage_columns(
            source, person_keys=_keys([_row()]), chunksize=1
        )
    assert opened == ["psam_pusa.csv"]
    assert sum(consumed) < 32_768
