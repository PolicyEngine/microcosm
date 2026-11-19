"""Read the December rows of the full SIPP 2023 person-month file once.

The financial-asset, SSI-disability, Head Start, vehicle and voluntary-filing
donor tables all start from the same 3.73 GB file of about 5,000 columns. A
``low_memory=False`` parse of that file holds the position of every field of a
100,000-row chunk, including the columns no donor reads. This read avoids that
cost and still gives each donor the table its own parse gave. For each chunk:

1. The file is tokenized with ``low_memory=True``, keeping only the requested
   columns, as unconverted text.
2. That text, now only the requested columns wide, is parsed with
   ``low_memory=False``, so pandas infers each column's type from the whole chunk
   with the same parser and the same tokens as a direct parse of the file.

A file with a carriage return outside a ``\\r\\n`` pair is refused: pandas splits
such a file into different rows depending on how it is read.

The donors' own parses differed in three ways, which the result keeps apart:

* The financial-asset donor read ``SSUID`` as text, so ``"0123"`` and ``"123"``
  stay separate households. ``ssuid_text`` holds that form.
* The Head Start donor kept chunks without December rows when concatenating, so
  their column types count towards the result's types.
* The vehicle and voluntary-filing donors parsed with ``low_memory=True``, which
  converts a few hundred rows at a time and combines the results. For a column
  inferred as ``int64`` or ``bool`` over the whole chunk, the blocks agree with
  it. For a ``float64`` column they agree when every integer cell has at most
  fifteen digits and none is a negative zero. Any other column is recorded, and
  :meth:`SippDecemberPersonRows.source_table` refuses it to these two donors,
  including cases where the blocks would in fact have agreed.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "SippDecemberPersonRows",
    "read_sipp_december_person_rows",
]

_DELIMITER = "|"
_IDENTIFIER_COLUMN = "SSUID"
_MONTH_COLUMN = "MONTHCODE"
_DECEMBER = 12
_SCAN_BLOCK_BYTES = 8 * 1024 * 1024
_INTEGER_TOKEN = re.compile(r"\s*([+-]?)([0-9]+)\s*")
# Integers written with up to fifteen digits, leading zeros included, are exact
# in float64 by either conversion route.
_MAX_EXACT_INTEGER_DIGITS = 15


@dataclass(frozen=True, eq=False)
class SippDecemberPersonRows:
    """December person rows of one SIPP file, ready for the donor builders.

    ``rows`` holds the requested columns in file order, with types inferred per
    chunk. ``ssuid_text`` holds the same rows' identifiers as written in the
    file. ``december_parts_by_chunk`` is empty when every chunk had a December
    row; otherwise it holds each chunk's December rows in file order, zero-row
    tables included. Builders take their columns from :meth:`source_table`;
    pandas' copy-on-write keeps their changes out of this object.
    """

    rows: pd.DataFrame
    ssuid_text: pd.Series
    december_parts_by_chunk: tuple[pd.DataFrame, ...]
    columns_sensitive_to_parser_blocks: frozenset[str]
    file_row_count: int
    file_size_bytes: int
    verified_sha256: str | None
    verified_size_bytes: int | None

    def source_table(
        self,
        columns: Iterable[str],
        *,
        ssuid_as_text: bool = False,
        read_in_parser_blocks: bool = False,
        include_chunks_without_december_rows: bool = False,
    ) -> pd.DataFrame:
        """Return ``columns`` in file order, as one donor's own parse gave them.

        ``read_in_parser_blocks`` is for a donor that parsed with
        ``low_memory=True``; it refuses a column whose values that parse could
        have changed. ``include_chunks_without_december_rows`` lets those chunks'
        column types count, as a donor that concatenated them did.
        """

        requested = set(columns)
        missing = sorted(requested - set(self.rows.columns))
        if missing:
            raise ValueError(f"SIPP December person rows lack column(s): {missing}.")
        if read_in_parser_blocks:
            sensitive = sorted(requested & self.columns_sensitive_to_parser_blocks)
            if sensitive:
                raise ValueError(
                    f"SIPP person file column(s) {sensitive} hold values that "
                    "low_memory parser blocks could convert differently, so this "
                    "read cannot reproduce them for this donor."
                )
        table = self.rows.loc[
            :, [column for column in self.rows.columns if column in requested]
        ]
        if include_chunks_without_december_rows and self.december_parts_by_chunk:
            table = pd.concat(
                [part.loc[:, table.columns] for part in self.december_parts_by_chunk],
                ignore_index=True,
            )
        if ssuid_as_text:
            table = table.assign(**{_IDENTIFIER_COLUMN: self.ssuid_text})
        return table


def _scan_file(path: Path) -> tuple[str, bool]:
    """Return the file's SHA-256 and whether it has a bare carriage return."""

    digest = hashlib.sha256()
    has_bare_carriage_return = False
    previous_block_ended_in_carriage_return = False
    with path.open("rb") as stream:
        while block := stream.read(_SCAN_BLOCK_BYTES):
            digest.update(block)
            ends_in_carriage_return = block.endswith(b"\r")
            paired_carriage_returns = block.count(b"\r\n")
            if (
                previous_block_ended_in_carriage_return and not block.startswith(b"\n")
            ) or block.count(b"\r") != (
                paired_carriage_returns + ends_in_carriage_return
            ):
                has_bare_carriage_return = True
            previous_block_ended_in_carriage_return = ends_in_carriage_return
    return digest.hexdigest(), (
        has_bare_carriage_return or previous_block_ended_in_carriage_return
    )


def _converts_differently_in_parser_blocks(token: str) -> bool:
    """Return whether an integer cell of a float column depends on its block.

    A block holding only integers is parsed as ``int64`` and then cast to
    ``float64``, and an integer outside ``int64`` leaves its block unconverted;
    the whole chunk goes straight through the float parser.
    """

    integer = _INTEGER_TOKEN.fullmatch(token)
    if integer is None:
        return False
    sign, digits = integer.groups()
    is_negative_zero = sign == "-" and not digits.strip("0")
    return is_negative_zero or len(digits) > _MAX_EXACT_INTEGER_DIGITS


def _columns_sensitive_to_parser_blocks(
    cell_text: pd.DataFrame,
    parsed: pd.DataFrame,
) -> set[str]:
    """Return columns a ``low_memory=True`` parse of this chunk could change."""

    sensitive: set[str] = set()
    for column in parsed.columns:
        dtype = parsed[column].dtype
        if dtype == "int64" or pd.api.types.is_bool_dtype(dtype):
            continue
        if dtype != "float64" or any(
            _converts_differently_in_parser_blocks(token)
            for token in cell_text[column].dropna().unique()
        ):
            sensitive.add(column)
    return sensitive


def read_sipp_december_person_rows(
    path: str | Path,
    *,
    columns: Iterable[str],
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    chunksize: int = 100_000,
) -> SippDecemberPersonRows:
    """Verify the file, then read it once and keep its December rows."""

    source_path = Path(path)
    file_size_bytes = source_path.stat().st_size
    if expected_size_bytes is not None and file_size_bytes != expected_size_bytes:
        raise ValueError(
            "SIPP person file failed byte-length verification: "
            f"expected {expected_size_bytes}, got {file_size_bytes}."
        )
    actual_sha256, has_bare_carriage_return = _scan_file(source_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            "SIPP person file failed sha-256 verification: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    if has_bare_carriage_return:
        raise ValueError(
            "SIPP person file has a carriage return outside a CRLF pair, so its "
            "rows depend on how it is parsed."
        )
    if chunksize < 1:
        raise ValueError("chunksize must be a positive integer")

    requested_columns = tuple(columns)
    first_row = pd.read_csv(source_path, delimiter=_DELIMITER, nrows=1)
    missing = sorted(set(requested_columns) - set(first_row.columns))
    if missing:
        raise ValueError(f"SIPP person file missing column(s): {missing}.")
    # pandas turns a first field beyond the header into an index and shifts the
    # columns, but only when some header column is left unread.
    if not isinstance(first_row.index, pd.RangeIndex):
        raise ValueError("SIPP person file's first row is longer than its header.")

    december_parts: list[pd.DataFrame] = []
    ssuid_text_parts: list[pd.Series] = []
    columns_sensitive_to_parser_blocks: set[str] = set()
    file_row_count = 0
    with pd.read_csv(
        source_path,
        delimiter=_DELIMITER,
        usecols=list(requested_columns),
        chunksize=int(chunksize),
        dtype=object,
        na_filter=False,
        low_memory=True,
    ) as chunks:
        for cell_text in chunks:
            if cell_text.empty:
                continue
            file_row_count += len(cell_text)
            # The index column keeps data cells off the start of each row, where
            # the parser can drop whitespace that ends its read buffer.
            requested_csv = cell_text.to_csv(sep=_DELIMITER).encode()
            parsed = pd.read_csv(
                io.BytesIO(requested_csv),
                delimiter=_DELIMITER,
                index_col=0,
                low_memory=False,
            ).reset_index(drop=True)
            columns_sensitive_to_parser_blocks |= _columns_sensitive_to_parser_blocks(
                cell_text,
                parsed,
            )
            month = pd.to_numeric(parsed[_MONTH_COLUMN], errors="coerce")
            is_december = month.eq(_DECEMBER)
            december_parts.append(parsed.loc[is_december])
            if not is_december.any():
                continue
            ssuid_text = pd.read_csv(
                io.BytesIO(requested_csv),
                delimiter=_DELIMITER,
                usecols=[_IDENTIFIER_COLUMN],
                dtype={_IDENTIFIER_COLUMN: "string"},
                low_memory=False,
            )[_IDENTIFIER_COLUMN]
            ssuid_text_parts.append(ssuid_text[is_december])
    if not ssuid_text_parts:
        raise ValueError("SIPP person file has no December rows.")
    parts_with_rows = [part for part in december_parts if not part.empty]
    return SippDecemberPersonRows(
        rows=pd.concat(parts_with_rows, ignore_index=True),
        ssuid_text=pd.concat(ssuid_text_parts, ignore_index=True),
        december_parts_by_chunk=(
            tuple(december_parts) if len(parts_with_rows) < len(december_parts) else ()
        ),
        columns_sensitive_to_parser_blocks=frozenset(
            columns_sensitive_to_parser_blocks
        ),
        file_row_count=file_row_count,
        file_size_bytes=file_size_bytes,
        verified_sha256=expected_sha256,
        verified_size_bytes=expected_size_bytes,
    )
