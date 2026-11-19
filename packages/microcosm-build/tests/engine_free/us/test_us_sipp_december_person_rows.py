"""One read of the SIPP person file must give each donor what its own read gave.

``_separate_december_read`` keeps a copy of the per-donor reads the shared read
replaced: each donor read its own columns, and they differed in how ``SSUID``
was parsed, in ``low_memory`` and in whether chunks without December rows were
kept. The comparison needs that copy; nothing else should use it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from microcosm.build.us_runtime import sipp_december_person_rows as person_rows_module
from microcosm.build.us_runtime.sipp_december_person_rows import (
    SippDecemberPersonRows,
    read_sipp_december_person_rows,
)
from microcosm.build.us_runtime.sipp_financial_assets import (
    SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
    SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES,
    SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS,
    build_sipp_2023_financial_asset_donor,
)
from microcosm.build.us_runtime.sipp_head_start import (
    SIPP_2023_HEAD_START_DONOR_SHA256,
    SIPP_2023_HEAD_START_DONOR_SIZE_BYTES,
    SIPP_HEAD_START_SOURCE_COLUMNS,
    build_sipp_2023_head_start_donor,
)
from microcosm.build.us_runtime.sipp_vehicles import (
    SIPP_2023_VEHICLE_DONOR_SHA256,
    SIPP_2023_VEHICLE_DONOR_SIZE_BYTES,
    SIPP_VEHICLE_SOURCE_COLUMNS,
    build_sipp_2023_vehicle_donor,
)
from microcosm.build.us_runtime.ssi_disability_criteria import (
    SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
    SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
    SIPP_SSI_DISABILITY_SOURCE_COLUMNS,
    build_sipp_2023_ssi_disability_donor,
    load_sipp_2023_ssi_disability_donor,
)
from microcosm.build.us_runtime.voluntary_filing import (
    SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS,
    build_sipp_2023_voluntary_filing_donor,
)

_CHUNK_ROWS = 7
_HOUSEHOLD_COUNT = 24
_UNREAD_COLUMN = "UNREAD_NOTE"
_ALL_DONOR_COLUMNS = tuple(
    sorted(
        {
            *SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS,
            *SIPP_SSI_DISABILITY_SOURCE_COLUMNS,
            *SIPP_HEAD_START_SOURCE_COLUMNS,
            *SIPP_VEHICLE_SOURCE_COLUMNS,
            *SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS,
        }
    )
)


@dataclass(frozen=True)
class _DonorRead:
    """How one donor read the file before the reads were merged."""

    columns: tuple[str, ...]
    build_donor: Callable[[SippDecemberPersonRows], pd.DataFrame]
    ssuid_as_text: bool = False
    low_memory: bool = False
    keeps_chunks_without_december_rows: bool = False


_DONOR_READS = {
    "financial_assets": _DonorRead(
        SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS,
        build_sipp_2023_financial_asset_donor,
        ssuid_as_text=True,
    ),
    "ssi_disability": _DonorRead(
        SIPP_SSI_DISABILITY_SOURCE_COLUMNS,
        build_sipp_2023_ssi_disability_donor,
    ),
    "head_start": _DonorRead(
        SIPP_HEAD_START_SOURCE_COLUMNS,
        build_sipp_2023_head_start_donor,
        keeps_chunks_without_december_rows=True,
    ),
    "vehicles": _DonorRead(
        SIPP_VEHICLE_SOURCE_COLUMNS,
        build_sipp_2023_vehicle_donor,
        low_memory=True,
    ),
    "voluntary_filing": _DonorRead(
        SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS,
        build_sipp_2023_voluntary_filing_donor,
        low_memory=True,
    ),
}
_SSUID_VARIANTS = (
    "digits_with_leading_zeros",
    "blank_in_chunk_without_december_rows",
    "blank_beside_december_rows",
    "every_identifier_blank_in_one_chunk",
    "text_identifiers",
    "text_identifiers_with_blank",
)
# A donor that parsed in low_memory blocks refuses text identifiers, so the
# comparison with its own read covers numeric identifiers only.
_COMPARABLE_DONOR_CASES = tuple(
    (donor_name, ssuid_variant)
    for donor_name, donor_read in _DONOR_READS.items()
    for ssuid_variant in _SSUID_VARIANTS
    if not (donor_read.low_memory and ssuid_variant.startswith("text_identifiers"))
)


def _person_cells(
    household_number: int,
    person_number: int,
    ssuid: str,
) -> dict[str, str]:
    """Return one person's cells, valid for every donor's transform."""

    has_spouse = household_number % 3 == 0
    is_first_person = person_number % 2 == 1
    if is_first_person or has_spouse:
        age = 30 + household_number % 30 + (0 if is_first_person else 2)
    elif household_number % 3 == 1:
        age = 3 + (household_number // 3) % 3
    else:
        age = 15
    is_adult = age >= 18
    monthly_earnings = 1_000 + 3 * household_number if is_adult else 0
    cells = dict.fromkeys(_ALL_DONOR_COLUMNS, "0")
    cells.update(
        {
            "SSUID": ssuid,
            "PNUM": str(person_number),
            "WPFINWGT": f"{1_000 + household_number * 17.125 + person_number / 7:.6f}",
            "TAGE": str(age),
            "ESEX": "1" if is_first_person else "2",
            "EMS": "1" if has_spouse and is_adult else "6",
            "TPTOTINC": str(monthly_earnings),
            "TJB1_MSUM": str(monthly_earnings),
            "TJB2_MSUM": "",
            "TVAL_BANK": ""
            if household_number % 5 == 0
            else str(household_number * 10),
            "TVAL_STMF": str(household_number),
            "TVAL_BOND": str(household_number + 1),
            "AJSSAVVAL": "2" if household_number % 4 == 0 else "0",
            "RSSI_YRYN": "1" if household_number % 6 == 1 and is_adult else "2",
            "ESSI_BRSN": "1",
            "AEDHEADST": "1" if 3 <= age <= 5 else "0",
            "EEDHEADST": "1" if household_number % 2 else "2",
            "EPNSPOUSE": str(person_number + (1 if is_first_person else -1))
            if has_spouse
            else "",
            "AFILING": "1" if is_adult else "0",
            "EFILING": "1" if household_number % 2 else "2",
            "AWILLFILE": "1",
            "EWILLFILE": "2",
            "EDEPCLM": "2",
            "TVEH_NUM": str(1 + household_number % 3),
            "THVAL_VEH": str(5_000 * household_number),
            "THVAL_HOME": str(100_000 * (household_number % 2)),
        }
    )
    return cells


def _household_identifiers(ssuid_variant: str) -> list[str]:
    if ssuid_variant.startswith("text_identifiers"):
        return [f"H{number:03d}" for number in range(_HOUSEHOLD_COUNT)]
    identifiers = [f"{number + 7:012d}" for number in range(_HOUSEHOLD_COUNT)]
    # Equal as numbers, different as text: only the financial-asset donor
    # keeps these two households apart.
    identifiers[-2:] = ["000000000123", "123"]
    return identifiers


def _write_person_file(tmp_path: Path, ssuid_variant: str) -> Path:
    identifiers = _household_identifiers(ssuid_variant)
    rows: list[dict[str, str]] = []
    # Three chunks with no December row, one with a blank age.
    for filler_position in range(3 * _CHUNK_ROWS):
        cells = _person_cells(1, 101, identifiers[1])
        cells["MONTHCODE"] = "5"
        cells["PNUM"] = str(500 + filler_position)
        if filler_position == 3:
            cells["TAGE"] = ""
            if ssuid_variant == "blank_in_chunk_without_december_rows":
                cells["SSUID"] = ""
        if (
            ssuid_variant == "every_identifier_blank_in_one_chunk"
            and filler_position < _CHUNK_ROWS
        ):
            cells["SSUID"] = ""
        rows.append(cells)
    for household_number, ssuid in enumerate(identifiers):
        first_person = 101 + 2 * (ssuid == "123")
        for person_number in (first_person, first_person + 1):
            for month in ("11", "12"):
                cells = _person_cells(household_number, person_number, ssuid)
                cells["MONTHCODE"] = month
                rows.append(cells)
    if ssuid_variant in ("blank_beside_december_rows", "text_identifiers_with_blank"):
        # This November row starts a chunk that also holds December rows.
        rows[3 * _CHUNK_ROWS]["SSUID"] = ""

    # A file order unlike any donor's column list, with a column no donor reads.
    header = [*sorted(_ALL_DONOR_COLUMNS, reverse=True), _UNREAD_COLUMN]
    lines = ["|".join(header)]
    lines.extend(
        "|".join([*(row[column] for column in header[:-1]), "free text"])
        for row in rows
    )
    path = tmp_path / "pu2023.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _separate_december_read(path: Path, donor_read: _DonorRead) -> pd.DataFrame:
    """Read December rows the way this donor's own read did."""

    options: dict[str, object] = {}
    if donor_read.ssuid_as_text:
        options["dtype"] = {"SSUID": "string"}
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        delimiter="|",
        usecols=list(donor_read.columns),
        chunksize=_CHUNK_ROWS,
        low_memory=donor_read.low_memory,
        **options,
    ):
        month = pd.to_numeric(chunk["MONTHCODE"], errors="coerce")
        december = chunk.loc[month.eq(12)].copy()
        if donor_read.keeps_chunks_without_december_rows or not december.empty:
            parts.append(december)
    return pd.concat(parts, ignore_index=True)


def _rows_from_separate_read(path: Path, donor_read: _DonorRead):
    table = _separate_december_read(path, donor_read)
    return SippDecemberPersonRows(
        rows=table,
        ssuid_text=table["SSUID"],
        december_parts_by_chunk=(),
        columns_sensitive_to_parser_blocks=frozenset(),
        file_row_count=sum(1 for _ in path.open(encoding="utf-8")) - 1,
        file_size_bytes=path.stat().st_size,
        verified_sha256=None,
        verified_size_bytes=None,
    )


def _assert_identical_tables(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Compare exactly, separating 0.0 from -0.0 and object 1 from 1.0."""

    assert_frame_equal(actual, expected, check_exact=True)
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column].dtype):
            assert (
                actual[column].to_numpy().tobytes()
                == expected[column].to_numpy().tobytes()
            ), column
        elif pd.api.types.is_object_dtype(expected[column].dtype):
            assert list(map(repr, actual[column])) == list(
                map(repr, expected[column])
            ), column


def _shared_read(path: Path) -> SippDecemberPersonRows:
    return read_sipp_december_person_rows(
        path,
        columns=_ALL_DONOR_COLUMNS,
        chunksize=_CHUNK_ROWS,
    )


@pytest.mark.parametrize(("donor_name", "ssuid_variant"), _COMPARABLE_DONOR_CASES)
def test_shared_read_gives_each_donor_the_table_of_its_own_read(
    tmp_path: Path,
    donor_name: str,
    ssuid_variant: str,
) -> None:
    donor_read = _DONOR_READS[donor_name]
    path = _write_person_file(tmp_path, ssuid_variant)
    table_without_empty_chunks = _separate_december_read(
        path,
        _DonorRead(
            donor_read.columns,
            donor_read.build_donor,
            ssuid_as_text=donor_read.ssuid_as_text,
            low_memory=donor_read.low_memory,
        ),
    )

    shared_table = _shared_read(path).source_table(
        donor_read.columns,
        ssuid_as_text=donor_read.ssuid_as_text,
        read_in_parser_blocks=donor_read.low_memory,
    )

    _assert_identical_tables(shared_table, table_without_empty_chunks)


@pytest.mark.parametrize("ssuid_variant", _SSUID_VARIANTS)
def test_head_start_table_counts_chunks_without_december_rows(
    tmp_path: Path,
    ssuid_variant: str,
) -> None:
    donor_read = _DONOR_READS["head_start"]
    path = _write_person_file(tmp_path, ssuid_variant)

    shared_table = _shared_read(path).source_table(
        donor_read.columns,
        include_chunks_without_december_rows=True,
    )

    _assert_identical_tables(shared_table, _separate_december_read(path, donor_read))


@pytest.mark.parametrize("donor_name", ("vehicles", "voluntary_filing"))
def test_low_memory_donors_refuse_text_identifiers(
    tmp_path: Path,
    donor_name: str,
) -> None:
    shared_rows = _shared_read(_write_person_file(tmp_path, "text_identifiers"))

    with pytest.raises(ValueError, match=r"SSUID.*parser blocks"):
        _DONOR_READS[donor_name].build_donor(shared_rows)


@pytest.mark.parametrize(("donor_name", "ssuid_variant"), _COMPARABLE_DONOR_CASES)
def test_donor_built_from_shared_read_equals_donor_from_its_own_read(
    tmp_path: Path,
    donor_name: str,
    ssuid_variant: str,
) -> None:
    donor_read = _DONOR_READS[donor_name]
    path = _write_person_file(tmp_path, ssuid_variant)

    shared_donor = donor_read.build_donor(_shared_read(path))
    separate_donor = donor_read.build_donor(_rows_from_separate_read(path, donor_read))

    assert not separate_donor.empty
    _assert_identical_tables(shared_donor, separate_donor)
    assert shared_donor.attrs == separate_donor.attrs


def test_building_every_donor_leaves_the_shared_rows_unchanged(tmp_path: Path) -> None:
    shared_rows = _shared_read(
        _write_person_file(tmp_path, "digits_with_leading_zeros")
    )
    rows_before = shared_rows.rows.copy(deep=True)
    ssuid_text_before = shared_rows.ssuid_text.copy(deep=True)

    for donor_read in _DONOR_READS.values():
        donor_read.build_donor(shared_rows)

    assert_frame_equal(shared_rows.rows, rows_before, check_exact=True)
    pd.testing.assert_series_equal(shared_rows.ssuid_text, ssuid_text_before)


def test_identifier_forms_keep_leading_zeros_only_as_text(tmp_path: Path) -> None:
    shared_rows = _shared_read(
        _write_person_file(tmp_path, "digits_with_leading_zeros")
    )

    as_text = shared_rows.source_table(("SSUID", "PNUM"), ssuid_as_text=True)["SSUID"]
    as_number = shared_rows.source_table(("SSUID", "PNUM"))["SSUID"]

    assert as_text.iloc[0] == "000000000007"
    assert as_number.iloc[0] == 7
    assert as_number.dtype == "int64"
    assert as_text.nunique() == as_number.nunique() + 1


def test_shared_read_counts_every_file_row_and_records_the_file(tmp_path: Path) -> None:
    path = _write_person_file(tmp_path, "digits_with_leading_zeros")
    digest = sha256(path.read_bytes()).hexdigest()

    shared_rows = read_sipp_december_person_rows(
        path,
        columns=_ALL_DONOR_COLUMNS,
        expected_sha256=digest,
        expected_size_bytes=path.stat().st_size,
        chunksize=_CHUNK_ROWS,
    )

    assert shared_rows.file_row_count == 3 * _CHUNK_ROWS + 4 * _HOUSEHOLD_COUNT
    assert len(shared_rows.rows) == 2 * _HOUSEHOLD_COUNT
    assert shared_rows.file_size_bytes == path.stat().st_size
    assert shared_rows.verified_sha256 == digest
    assert shared_rows.verified_size_bytes == path.stat().st_size


def test_all_five_donors_pin_the_one_file_the_shared_read_verifies() -> None:
    assert {
        SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
        SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
        SIPP_2023_HEAD_START_DONOR_SHA256,
        SIPP_2023_VEHICLE_DONOR_SHA256,
        SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
    } == {SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256}
    assert {
        SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES,
        SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
        SIPP_2023_HEAD_START_DONOR_SIZE_BYTES,
        SIPP_2023_VEHICLE_DONOR_SIZE_BYTES,
        SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    } == {SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES}


def test_ssi_path_loader_equals_the_donor_built_from_its_own_read(
    tmp_path: Path,
) -> None:
    path = _write_person_file(tmp_path, "digits_with_leading_zeros")
    donor_read = _DONOR_READS["ssi_disability"]

    donor = load_sipp_2023_ssi_disability_donor(
        path,
        expected_size_bytes=None,
        chunksize=_CHUNK_ROWS,
    )

    expected = donor_read.build_donor(_rows_from_separate_read(path, donor_read))
    _assert_identical_tables(donor, expected)
    assert donor.attrs == expected.attrs


def test_ssi_path_loader_passes_its_pins_and_period_through(tmp_path: Path) -> None:
    path = _write_person_file(tmp_path, "digits_with_leading_zeros")

    with pytest.raises(ValueError, match="sha-256 verification"):
        load_sipp_2023_ssi_disability_donor(
            path,
            expected_sha256="0" * 64,
            expected_size_bytes=None,
        )
    with pytest.raises(ValueError, match="byte-length verification"):
        load_sipp_2023_ssi_disability_donor(path, expected_size_bytes=1)
    donor = load_sipp_2023_ssi_disability_donor(
        path,
        expected_size_bytes=None,
        chunksize=_CHUNK_ROWS,
        time_period=2023,
    )
    assert donor.attrs["source_audit"]["time_period"] == 2023


def test_head_start_audits_only_rows_verified_against_both_pins(
    tmp_path: Path,
) -> None:
    path = _write_person_file(tmp_path, "digits_with_leading_zeros")
    person_rows = _shared_read(path)

    with pytest.raises(ValueError, match="Pinned SIPP Head Start audit drifted"):
        build_sipp_2023_head_start_donor(
            replace(
                person_rows,
                verified_sha256=SIPP_2023_HEAD_START_DONOR_SHA256,
                verified_size_bytes=SIPP_2023_HEAD_START_DONOR_SIZE_BYTES,
            )
        )
    donor = build_sipp_2023_head_start_donor(
        replace(person_rows, verified_sha256=SIPP_2023_HEAD_START_DONOR_SHA256)
    )
    assert donor.attrs["source_audit"]["pinned_transform"] is False


def test_ssi_donor_audits_a_file_of_the_pinned_size_for_2024(tmp_path: Path) -> None:
    path = _write_person_file(tmp_path, "digits_with_leading_zeros")
    pinned_size_rows = replace(
        _shared_read(path),
        file_size_bytes=SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
    )

    with pytest.raises(ValueError, match="Pinned SIPP SSI disability source audit"):
        build_sipp_2023_ssi_disability_donor(pinned_size_rows)
    donor = build_sipp_2023_ssi_disability_donor(pinned_size_rows, time_period=2023)
    assert donor.attrs["source_audit"]["pinned_transform"] is False


def _write_table(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    lines = ["|".join(header), *("|".join(row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_wide_december_table(
    path: Path,
    weight_cells: list[str],
    *,
    month_cells: list[str] | None = None,
) -> Path:
    """Write a table wide enough for ``low_memory`` to use 1,024-row blocks."""

    unread_columns = [f"UNREAD_{position}" for position in range(1_000)]
    months = month_cells or ["12"] * len(weight_cells)
    return _write_table(
        path,
        ["SSUID", "MONTHCODE", "WPFINWGT", *unread_columns],
        [
            [str(position + 1), month, weight, *[""] * len(unread_columns)]
            for position, (month, weight) in enumerate(
                zip(months, weight_cells, strict=True)
            )
        ],
    )


_WIDE_TABLE_COLUMNS = ("SSUID", "MONTHCODE", "WPFINWGT")
_FIRST_BLOCK_ROWS = 1_100
_CELLS_THAT_PARSER_BLOCKS_CHANGE = {
    "seventeen_digit_integer": ["99302078781721519"] * _FIRST_BLOCK_ROWS + ["1.5"],
    "negative_zero": ["-0"] * _FIRST_BLOCK_ROWS + ["0.5"],
    "above_int64": ["1"] * _FIRST_BLOCK_ROWS + ["18446744073709551615"],
    "text_after_numbers": ["3"] * _FIRST_BLOCK_ROWS + ["x"],
    "integer_with_many_leading_zeros": ["0" * 20 + "12"] * _FIRST_BLOCK_ROWS + ["0.5"],
}


@pytest.mark.parametrize("case_name", _CELLS_THAT_PARSER_BLOCKS_CHANGE)
def test_whole_chunk_donors_get_a_direct_parse_of_awkward_cells(
    tmp_path: Path,
    case_name: str,
) -> None:
    path = _write_wide_december_table(
        tmp_path / "wide.csv",
        _CELLS_THAT_PARSER_BLOCKS_CHANGE[case_name],
    )

    shared_table = read_sipp_december_person_rows(
        path,
        columns=_WIDE_TABLE_COLUMNS,
    ).source_table(_WIDE_TABLE_COLUMNS)

    _assert_identical_tables(
        shared_table,
        pd.read_csv(
            path,
            delimiter="|",
            usecols=list(_WIDE_TABLE_COLUMNS),
            low_memory=False,
        ),
    )


@pytest.mark.parametrize("case_name", _CELLS_THAT_PARSER_BLOCKS_CHANGE)
def test_low_memory_donors_refuse_cells_parser_blocks_change(
    tmp_path: Path,
    case_name: str,
) -> None:
    path = _write_wide_december_table(
        tmp_path / "wide.csv",
        _CELLS_THAT_PARSER_BLOCKS_CHANGE[case_name],
    )
    shared_rows = read_sipp_december_person_rows(path, columns=_WIDE_TABLE_COLUMNS)

    with pytest.raises(ValueError, match=r"WPFINWGT.*parser blocks"):
        shared_rows.source_table(_WIDE_TABLE_COLUMNS, read_in_parser_blocks=True)


def test_low_memory_donors_refuse_an_out_of_range_integer_outside_december(
    tmp_path: Path,
) -> None:
    weights = ["1.5"] * _FIRST_BLOCK_ROWS + ["99999999999999999999999"]
    path = _write_wide_december_table(
        tmp_path / "wide.csv",
        weights,
        month_cells=["12"] * _FIRST_BLOCK_ROWS + ["11"],
    )
    shared_rows = read_sipp_december_person_rows(path, columns=_WIDE_TABLE_COLUMNS)

    with pytest.raises(ValueError, match=r"WPFINWGT.*parser blocks"):
        shared_rows.source_table(_WIDE_TABLE_COLUMNS, read_in_parser_blocks=True)


def test_low_memory_donors_refuse_a_month_only_parser_blocks_read_as_december(
    tmp_path: Path,
) -> None:
    columns = ("SSUID", "MONTHCODE", "WPFINWGT")
    months = ["0" * 20 + "12"] * 3 + ["11.5"]
    other_chunk = [["9", "12", "1"]]
    path = _write_table(
        tmp_path / "months.csv",
        list(columns),
        [
            *[[str(position + 1), month, "1"] for position, month in enumerate(months)],
            *other_chunk,
        ],
    )
    shared_rows = read_sipp_december_person_rows(path, columns=columns, chunksize=4)

    with pytest.raises(ValueError, match=r"MONTHCODE.*parser blocks"):
        shared_rows.source_table(columns, read_in_parser_blocks=True)


def test_low_memory_donors_get_their_own_parse_of_ordinary_numbers(
    tmp_path: Path,
) -> None:
    path = _write_wide_december_table(
        tmp_path / "wide.csv",
        ["123456789012345"] * _FIRST_BLOCK_ROWS + ["", "1.25"],
    )

    shared_table = read_sipp_december_person_rows(
        path,
        columns=_WIDE_TABLE_COLUMNS,
    ).source_table(_WIDE_TABLE_COLUMNS, read_in_parser_blocks=True)

    _assert_identical_tables(
        shared_table,
        pd.read_csv(
            path,
            delimiter="|",
            usecols=list(_WIDE_TABLE_COLUMNS),
            low_memory=True,
        ),
    )
    assert shared_table["WPFINWGT"].dtype == np.float64


@pytest.mark.parametrize(
    "identifiers",
    (
        ["True", "False"],
        ["True", ""],
        ["0012", "A13"],
        ["12.5", "13"],
        ["+7", "-7"],
        ["12345678901234567890", "1"],
    ),
)
def test_identifiers_are_parsed_as_a_direct_parse_would(
    tmp_path: Path,
    identifiers: list[str],
) -> None:
    columns = ("SSUID", "MONTHCODE")
    path = _write_table(
        tmp_path / "identifiers.csv",
        list(columns),
        [[identifier, "12"] for identifier in identifiers],
    )

    shared_rows = read_sipp_december_person_rows(path, columns=columns)

    _assert_identical_tables(
        shared_rows.source_table(columns),
        pd.read_csv(path, delimiter="|", low_memory=False),
    )
    _assert_identical_tables(
        shared_rows.source_table(columns, ssuid_as_text=True),
        pd.read_csv(
            path,
            delimiter="|",
            dtype={"SSUID": "string"},
            low_memory=False,
        ),
    )


def test_head_start_combines_every_chunk_in_one_concatenation(tmp_path: Path) -> None:
    columns = ("SSUID", "MONTHCODE", "PNUM")
    path = _write_table(
        tmp_path / "large_identifiers.csv",
        list(columns),
        [
            ["9007199254740992", "12", "1"],
            ["9007199254740993", "12", "1"],
            ["1", "12", "1"],
            ["", "11", "1"],
            ["abc", "11", "1"],
            ["abd", "11", "1"],
        ],
    )

    shared_table = read_sipp_december_person_rows(
        path,
        columns=columns,
        chunksize=2,
    ).source_table(columns, include_chunks_without_december_rows=True)

    parts = [
        chunk.loc[chunk["MONTHCODE"].eq(12)]
        for chunk in pd.read_csv(
            path,
            delimiter="|",
            chunksize=2,
            low_memory=False,
        )
    ]
    _assert_identical_tables(shared_table, pd.concat(parts, ignore_index=True))
    assert shared_table["SSUID"].iloc[:2].tolist() == [
        9007199254740992,
        9007199254740993,
    ]


@pytest.mark.parametrize(
    "cell",
    (
        '"a|b"',
        '"line one\nline two"',
        '"doubled "" quote"',
        '"12"',
        '"NA"',
        " 12",
        "12 ",
    ),
    ids=(
        "quoted_delimiter",
        "quoted_line_feed",
        "doubled_quote",
        "quoted_number",
        "quoted_na",
        "leading_space",
        "trailing_space",
    ),
)
def test_cell_text_survives_the_round_trip_through_the_narrow_table(
    tmp_path: Path,
    cell: str,
) -> None:
    columns = ("SSUID", "MONTHCODE", "TAGE")
    path = tmp_path / "quoted.csv"
    path.write_text(
        f"SSUID|MONTHCODE|UNREAD|TAGE\n1|12|x|{cell}\n2|12|y|40\n",
        encoding="utf-8",
        newline="",
    )

    shared_table = read_sipp_december_person_rows(
        path,
        columns=columns,
    ).source_table(columns)

    _assert_identical_tables(
        shared_table,
        pd.read_csv(path, delimiter="|", usecols=list(columns), low_memory=False),
    )


@pytest.mark.parametrize(
    "file_text",
    (
        "SSUID|MONTHCODE\r1|12\r2|12\r",
        "SSUID|MONTHCODE\r\n1|12\r2|12\r\n",
        "SSUID|MONTHCODE\n1|12\n2|12\r",
        'SSUID|MONTHCODE\n"1\r2"|12\n',
    ),
    ids=("every_row", "one_row", "last_byte", "inside_quotes"),
)
@pytest.mark.parametrize("scan_block_bytes", (3, 1024))
def test_read_refuses_a_bare_carriage_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_text: str,
    scan_block_bytes: int,
) -> None:
    monkeypatch.setattr(person_rows_module, "_SCAN_BLOCK_BYTES", scan_block_bytes)
    path = tmp_path / "carriage_returns.csv"
    path.write_bytes(file_text.encode("utf-8"))

    with pytest.raises(ValueError, match="carriage return"):
        read_sipp_december_person_rows(path, columns=("SSUID", "MONTHCODE"))


@pytest.mark.parametrize("scan_block_bytes", (3, 16, 1024))
def test_read_accepts_crlf_rows_across_scan_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scan_block_bytes: int,
) -> None:
    monkeypatch.setattr(person_rows_module, "_SCAN_BLOCK_BYTES", scan_block_bytes)
    columns = ("SSUID", "MONTHCODE")
    path = tmp_path / "crlf.csv"
    path.write_bytes(b"SSUID|MONTHCODE\r\n0012|12\r\n13|12\r\n")

    shared_table = read_sipp_december_person_rows(
        path,
        columns=columns,
    ).source_table(columns)

    _assert_identical_tables(
        shared_table,
        pd.read_csv(path, delimiter="|", low_memory=False),
    )


@pytest.mark.parametrize("header_has_unread_column", (False, True))
def test_read_refuses_a_first_row_longer_than_the_header(
    tmp_path: Path,
    header_has_unread_column: bool,
) -> None:
    columns = ["SSUID", "MONTHCODE"]
    header = [*columns, "UNREAD"] if header_has_unread_column else columns
    cells = ["1", "12", "x"] if header_has_unread_column else ["1", "12"]
    path = _write_table(tmp_path / "long.csv", header, [[*cells, ""]])

    with pytest.raises(ValueError, match="longer than its header"):
        read_sipp_december_person_rows(path, columns=columns)


def test_leading_space_in_the_first_column_survives_the_parser_buffer_end(
    tmp_path: Path,
) -> None:
    # In a table of only these two columns, the cell with the leading space
    # starts on the last character of pandas' 262,144-character read buffer.
    last_buffer_offset = 262_143
    header = "SSUID|MONTHCODE\n"
    row_characters = len("000000000001|12\n")
    characters_before_cell = last_buffer_offset - len(header)
    first_row_characters = characters_before_cell % row_characters or row_characters
    ordinary_row_count = (
        characters_before_cell - first_row_characters
    ) // row_characters
    rows = [
        f"{'1' * (first_row_characters - len('|12' + chr(10)))}|12\n",
        *(f"{number:012d}|12\n" for number in range(2, ordinary_row_count + 2)),
    ]
    assert len(header) + sum(map(len, rows)) == last_buffer_offset
    narrow_lines = [*rows, " 000000000099|12\n", "000000000100|12\n"]
    path = tmp_path / "boundary.csv"
    path.write_text(
        "SSUID|MONTHCODE|UNREAD\n"
        + "".join(f"{line[:-1]}|x\n" for line in narrow_lines),
        encoding="utf-8",
    )
    columns = ("SSUID", "MONTHCODE")

    shared_rows = read_sipp_december_person_rows(path, columns=columns)

    direct_text = pd.read_csv(
        path,
        delimiter="|",
        usecols=list(columns),
        dtype={"SSUID": "string"},
        low_memory=False,
    )
    assert " 000000000099" in set(direct_text["SSUID"])
    _assert_identical_tables(
        shared_rows.source_table(columns, ssuid_as_text=True),
        direct_text,
    )


def test_second_parse_infers_types_over_the_whole_chunk(tmp_path: Path) -> None:
    # One hundred columns give 8,192-row blocks if the narrow table were parsed
    # with low_memory=True, so the text cell would leave earlier blocks numeric.
    columns = ["SSUID", "MONTHCODE", *(f"COLUMN_{n}" for n in range(98))]
    rows = [["1", "12", *["3"] * 98] for _ in range(9_000)]
    rows.append(["1", "12", "x", *["3"] * 97])
    path = _write_table(tmp_path / "text_late.csv", columns, rows)

    shared_table = read_sipp_december_person_rows(
        path,
        columns=columns,
    ).source_table(columns)

    _assert_identical_tables(
        shared_table,
        pd.read_csv(path, delimiter="|", low_memory=False),
    )


def test_scan_finds_a_carriage_return_at_the_end_of_a_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_bytes = b"SSUID|MONTHCODE\n1\r2|12\n"
    monkeypatch.setattr(
        person_rows_module,
        "_SCAN_BLOCK_BYTES",
        file_bytes.index(b"\r") + 1,
    )
    path = tmp_path / "block_end.csv"
    path.write_bytes(file_bytes)

    with pytest.raises(ValueError, match="carriage return"):
        read_sipp_december_person_rows(path, columns=("SSUID", "MONTHCODE"))


def test_low_memory_donors_accept_a_whole_chunk_boolean_column(tmp_path: Path) -> None:
    columns = ("SSUID", "MONTHCODE", "FLAG")
    path = _write_table(
        tmp_path / "flags.csv",
        list(columns),
        [["1", "12", "True"], ["2", "12", "False"]],
    )

    table = read_sipp_december_person_rows(path, columns=columns).source_table(
        columns,
        read_in_parser_blocks=True,
    )

    assert table["FLAG"].tolist() == [True, False]


def test_text_identifiers_replace_numbers_when_empty_chunks_are_included(
    tmp_path: Path,
) -> None:
    columns = ("SSUID", "MONTHCODE")
    path = _write_table(
        tmp_path / "empty_chunk.csv",
        list(columns),
        [["1", "11"], ["2", "11"], ["0012", "12"], ["0013", "12"]],
    )

    table = read_sipp_december_person_rows(
        path,
        columns=columns,
        chunksize=2,
    ).source_table(
        columns,
        ssuid_as_text=True,
        include_chunks_without_december_rows=True,
    )

    assert table["SSUID"].tolist() == ["0012", "0013"]


def test_head_start_builder_counts_chunks_without_december_rows(
    tmp_path: Path,
) -> None:
    # A blank identifier in a chunk with no December row makes SSUID float64,
    # so these two identifiers become equal, as they did in the separate read.
    identifiers = ["", "1", "9007199254740992", "9007199254740993"]
    months = ["11", "11", "12", "12"]
    columns = list(SIPP_HEAD_START_SOURCE_COLUMNS)
    rows = []
    for identifier, month in zip(identifiers, months, strict=True):
        cells = _person_cells(1, 101, identifier)
        cells["MONTHCODE"] = month
        rows.append([cells[column] for column in columns])
    path = _write_table(tmp_path / "head_start.csv", columns, rows)

    person_rows = read_sipp_december_person_rows(
        path,
        columns=columns,
        chunksize=2,
    )

    with pytest.raises(ValueError, match="not person-unique"):
        build_sipp_2023_head_start_donor(person_rows)


def test_read_keeps_non_ascii_digit_identifiers_as_text(tmp_path: Path) -> None:
    path = _write_table(
        tmp_path / "arabic_digits.csv",
        ["SSUID", "MONTHCODE"],
        [["\u0661\u0662", "12"], ["\u0663", "12"]],
    )

    shared_table = read_sipp_december_person_rows(
        path,
        columns=("SSUID", "MONTHCODE"),
    ).source_table(("SSUID", "MONTHCODE"))

    assert_frame_equal(
        shared_table,
        pd.read_csv(path, delimiter="|", low_memory=False),
        check_exact=True,
    )


def test_read_does_not_expand_a_home_directory_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_table(tmp_path / "home.csv", ["SSUID", "MONTHCODE"], [["12", "12"]])

    with pytest.raises(FileNotFoundError):
        read_sipp_december_person_rows(
            "~/home.csv",
            columns=("SSUID", "MONTHCODE"),
        )


def test_read_reports_a_file_without_data_rows_as_having_no_december_rows(
    tmp_path: Path,
) -> None:
    path = _write_table(tmp_path / "header.csv", ["SSUID", "MONTHCODE"], [])

    with pytest.raises(ValueError, match="no December rows"):
        read_sipp_december_person_rows(path, columns=("SSUID", "MONTHCODE"))


def test_read_refuses_a_file_that_fails_its_pins(tmp_path: Path) -> None:
    path = _write_table(
        tmp_path / "pinned.csv",
        ["SSUID", "MONTHCODE"],
        [["12", "12"]],
    )

    with pytest.raises(ValueError, match="byte-length verification"):
        read_sipp_december_person_rows(
            path,
            columns=("SSUID", "MONTHCODE"),
            expected_size_bytes=1,
        )
    with pytest.raises(ValueError, match="sha-256 verification"):
        read_sipp_december_person_rows(
            path,
            columns=("SSUID", "MONTHCODE"),
            expected_sha256="0" * 64,
        )


def test_read_refuses_missing_columns_and_files_without_december_rows(
    tmp_path: Path,
) -> None:
    path = _write_table(
        tmp_path / "november.csv",
        ["SSUID", "MONTHCODE"],
        [["12", "11"]],
    )

    with pytest.raises(ValueError, match=r"missing column\(s\): \['TAGE'\]"):
        read_sipp_december_person_rows(path, columns=("SSUID", "MONTHCODE", "TAGE"))
    with pytest.raises(ValueError, match="no December rows"):
        read_sipp_december_person_rows(path, columns=("SSUID", "MONTHCODE"))
    with pytest.raises(ValueError, match="chunksize"):
        read_sipp_december_person_rows(
            path,
            columns=("SSUID", "MONTHCODE"),
            chunksize=0,
        )


def test_source_table_refuses_a_column_the_read_did_not_keep(tmp_path: Path) -> None:
    path = _write_table(
        tmp_path / "december.csv",
        ["SSUID", "MONTHCODE", "TAGE"],
        [["12", "12", "40"]],
    )
    shared_rows = read_sipp_december_person_rows(path, columns=("SSUID", "MONTHCODE"))

    with pytest.raises(ValueError, match="TAGE"):
        shared_rows.source_table(("SSUID", "TAGE"))
