"""Invented ACS archives exercise exact frame reuse at both reader boundaries."""

from dataclasses import replace

import pandas as pd
import pytest
from test_us_acs_pums import _household, _person, _write_csv_zip
from test_us_source_memo import Calls, _counts, _key, _root

from microcosm.build.us_runtime import acs_person_coverage_authentication as coverage
from microcosm.build.us_runtime import acs_person_coverage_columns as literal
from microcosm.build.us_runtime import acs_pums as pums
from microcosm.build.us_runtime import source_memo as memo

TABLES = "acs_pums.load_acs_pums_tables"
COLUMNS = "acs_person_coverage_authentication._coverage_columns"
FIRST = "2024HU0000001"
SECOND = "2024HU0000002"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.delenv(memo.ROOT_ENV, raising=False)
    monkeypatch.delenv(memo.KEY_ENV, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    memo.reset_statistics()
    # Object labels and native key columns are normal with string inference
    # disabled. In particular, coverage keys must not silently bypass the memo.
    with pd.option_context("future.infer_string", False):
        yield
    memo.reset_statistics()


@pytest.fixture
def archives(tmp_path):
    source = pums.AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip")
    households = {
        "psam_husa.csv": [_household(SECOND, RNTP=500)],
        "psam_husb.csv": [_household(FIRST, NP=2)],
    }
    persons = {
        "psam_pusa.csv": [
            _person(SECOND, 1, 20, AGEP=80, MIL="4", ESR="6"),
            _person(FIRST, 1, 20, AGEP=30, MIL="1", ESR="4"),
        ],
        "psam_pusb.csv": [_person(FIRST, 2, 25, AGEP=15, MIL="", ESR="", WAGP=None)],
    }
    _write_csv_zip(source.household_zip, households)
    _write_csv_zip(source.person_zip, persons)
    return source, persons


def _paths(source):
    return {"household": source.household_zip, "person": source.person_zip}


def _keys():
    return pd.DataFrame(
        {
            "SERIALNO": pd.Series([FIRST, FIRST, SECOND], dtype=object),
            "SPORDER": pd.Series([1, 2, 1], dtype="int64"),
        }
    )


def _assert_tables(actual, expected):
    tables, metadata = actual
    expected_tables, expected_metadata = expected
    assert tuple(tables) == tuple(expected_tables) == ("household", "person")
    for name in tables:
        pd.testing.assert_frame_equal(
            tables[name], expected_tables[name], check_exact=True
        )
        assert memo.frames_identical(tables[name], expected_tables[name])
    assert memo.strict_equal(metadata, expected_metadata)


def _assert_columns(actual, expected):
    pd.testing.assert_frame_equal(actual[0], expected[0], check_exact=True)
    assert memo.frames_identical(actual[0], expected[0])
    assert memo.strict_equal(actual[1], expected[1])


def test_pums_tables_off_cold_warm_are_exact_and_warm_skips_parser(tmp_path, archives):
    source, _persons = archives
    traces, values = [], []
    for phase in ("off", "cold", "warm"):
        memo.reset_statistics()
        with (
            memo.source_memo(
                None if phase == "off" else _root(tmp_path), key_path=_key(tmp_path)
            ),
            Calls(pums._load_acs_pums_tables, pums._read_archive, pd.read_csv) as seen,
        ):
            values.append(pums.load_acs_pums_tables(source, chunksize=1))
        traces.append((dict(seen), _counts(TABLES)))
    for actual in values[1:]:
        _assert_tables(actual, values[0])
    assert pd.isna(values[0][0]["person"].loc[1, "WAGP"])
    assert traces[0] == (
        {"_load_acs_pums_tables": 1, "_read_archive": 2, "read_csv": 8},
        {},
    )
    assert traces[1] == (traces[0][0], {"miss": 1, "stored": 1})
    assert traces[2] == (
        {"_load_acs_pums_tables": 0, "_read_archive": 0, "read_csv": 0},
        {"hit": 1},
    )


def test_coverage_columns_off_cold_warm_are_exact_and_warm_skips_scan(
    tmp_path, archives
):
    source, _persons = archives
    keys = _keys()
    assert keys.SERIALNO.dtype == object
    traces, values = [], []
    for phase in ("off", "cold", "warm"):
        memo.reset_statistics()
        with (
            memo.source_memo(
                None if phase == "off" else _root(tmp_path), key_path=_key(tmp_path)
            ),
            Calls(
                literal.read_acs_person_coverage_columns,
                literal._scan_acs_person_coverage,
            ) as seen,
        ):
            values.append(coverage._coverage_columns(_paths(source), keys, 1))
        traces.append((dict(seen), _counts(COLUMNS)))
    for actual in values[1:]:
        _assert_columns(actual, values[0])
    assert values[0][0].MIL.tolist() == ["1", "", "4"]
    assert values[0][0].MIL_state.tolist() == [
        "observed_code",
        "outside_age_universe",
        "observed_code",
    ]
    assert values[0][1]["source_authenticated"] is False
    assert values[0][1]["release_eligible"] is False
    assert traces[0] == (
        {"read_acs_person_coverage_columns": 1, "_scan_acs_person_coverage": 1},
        {},
    )
    assert traces[1] == (traces[0][0], {"miss": 1, "stored": 1})
    assert traces[2] == (
        {"read_acs_person_coverage_columns": 0, "_scan_acs_person_coverage": 0},
        {"hit": 1},
    )


@pytest.mark.parametrize(
    "change", ["selection", "selection_order", "chunksize", "limit"]
)
def test_pums_parameter_changes_recompute_exact_tables(tmp_path, archives, change):
    source, _persons = archives
    initial = {"chunksize": 1, "serialnos": (FIRST, SECOND)}
    changed = dict(initial)
    changed_source = source
    if change == "selection":
        changed["serialnos"] = (FIRST,)
    elif change == "selection_order":
        changed["serialnos"] = (SECOND, FIRST)
    elif change == "chunksize":
        changed["chunksize"] = 2
    else:
        changed["serialnos"] = None
        changed_source = replace(source, max_households=1)
    plain = pums.load_acs_pums_tables(changed_source, **changed)
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        pums.load_acs_pums_tables(source, **initial)
        memo.reset_statistics()
        with Calls(pums._load_acs_pums_tables) as seen:
            actual = pums.load_acs_pums_tables(changed_source, **changed)
            warm = pums.load_acs_pums_tables(changed_source, **changed)
    _assert_tables(actual, plain)
    _assert_tables(warm, plain)
    assert seen == {"_load_acs_pums_tables": 1}
    assert _counts(TABLES) == {"miss": 1, "stored": 1, "hit": 1}


@pytest.mark.parametrize(
    "change", ["selection", "selection_order", "chunksize", "dtype"]
)
def test_coverage_parameter_changes_recompute_exact_columns(tmp_path, archives, change):
    source, _persons = archives
    keys, chunksize = _keys(), 1
    if change == "selection":
        keys = keys.iloc[:2].reset_index(drop=True)
    elif change == "selection_order":
        keys = keys.iloc[::-1].reset_index(drop=True)
    elif change == "chunksize":
        chunksize = 2
    else:
        keys["SERIALNO"] = keys.SERIALNO.astype("string")
    plain = coverage._coverage_columns(_paths(source), keys, chunksize)
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        coverage._coverage_columns(_paths(source), _keys(), 1)
        memo.reset_statistics()
        with Calls(literal._scan_acs_person_coverage) as seen:
            actual = coverage._coverage_columns(_paths(source), keys, chunksize)
            warm = coverage._coverage_columns(_paths(source), keys, chunksize)
    _assert_columns(actual, plain)
    _assert_columns(warm, plain)
    assert seen == {"_scan_acs_person_coverage": 1}
    assert _counts(COLUMNS) == {"miss": 1, "stored": 1, "hit": 1}


@pytest.mark.parametrize("reader", ["tables", "coverage"])
def test_archive_byte_changes_invalidate_frame_results(tmp_path, archives, reader):
    source, persons = archives

    def run():
        if reader == "tables":
            return pums.load_acs_pums_tables(source, chunksize=1)
        return coverage._coverage_columns(_paths(source), _keys(), 1)

    if reader == "tables":
        compare, namespace = _assert_tables, TABLES
    else:
        compare, namespace = _assert_columns, COLUMNS
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        before = run()
        persons["psam_pusa.csv"][1].update(WAGP=12_345, ESR="5")
        _write_csv_zip(source.person_zip, persons)
        memo.reset_statistics()
        actual, warm = run(), run()
        with memo.source_memo(None):
            plain = run()
    compare(actual, plain)
    compare(warm, plain)
    if reader == "tables":
        assert actual[0]["person"].loc[0, "WAGP"] == 12_345
        assert before[0]["person"].loc[0, "WAGP"] != 12_345
    else:
        assert actual[0].loc[0, "ESR"] == "5"
        assert before[0].loc[0, "ESR"] == "4"
    assert _counts(namespace) == {"miss": 1, "stored": 1, "hit": 1}


def test_unsupported_key_dtype_bypasses_memo_without_changing_read(tmp_path, archives):
    source, _persons = archives
    keys = _keys().astype({"SPORDER": "Int64"})
    plain = coverage._coverage_columns(_paths(source), keys, 1)
    with (
        memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)),
        Calls(literal._scan_acs_person_coverage) as seen,
    ):
        actual = coverage._coverage_columns(_paths(source), keys, 1)
        repeated = coverage._coverage_columns(_paths(source), keys, 1)
    _assert_columns(actual, plain)
    _assert_columns(repeated, plain)
    assert seen == {"_scan_acs_person_coverage": 2}
    assert _counts(COLUMNS) == {"bypassed:SourceMemoError": 2}


def test_invalid_exact_selection_still_refuses_after_a_warm_read(tmp_path, archives):
    source, _persons = archives
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        pums.load_acs_pums_tables(source, serialnos=(FIRST,))
        pums.load_acs_pums_tables(source, serialnos=(FIRST,))
        with pytest.raises(ValueError, match="bounded unique raw native keys"):
            pums.load_acs_pums_tables(source, serialnos=(FIRST, FIRST))
        with pytest.raises(ValueError, match="absent household keys"):
            pums.load_acs_pums_tables(source, serialnos=("2024HU9999999",))
