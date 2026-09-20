"""Differential raw-byte framing tests; no data/source authority is minted."""

import itertools
import random

import pytest

from microcosm.build.us_runtime import asec_coverage_authentication as coverage


class _ReferenceBounds(coverage._CsvBounds):
    # Original bcd6f0d97 byte loop, with module globals explicitly qualified.
    def feed(self, chunk):
        for byte in chunk:
            if self.after_cr and byte == 10:
                self.after_cr = False
                continue
            self.after_cr = False
            self.row += 1
            self.field += 1
            coverage._require(
                self.row
                <= (coverage._CSV_HEADER_MAX if self.header else coverage._ROW_MAX),
                "COVERAGE_CSV_RECORD_BYTES",
            )
            coverage._require(
                self.field <= coverage._TOKEN_MAX, "COVERAGE_CSV_TOKEN_BYTES"
            )
            if self.header:
                self.header_bytes.append(byte)
            else:
                if self.row == 1:
                    # Row length + six integers + three lengths + fixed native
                    # key + a conservative 32 bytes for any closed status.
                    self.budget[0] -= 4 + coverage._NUMBERS.size + 12 + 22 + 32
                if self.column == self.token_column:
                    self.budget[0] -= 1
                coverage._require(self.budget[0] >= 0, "COVERAGE_BODY_BYTES")
            if self.quoted:
                if byte == 34:
                    self.quoted, self.after_quote = False, True
            elif self.after_quote and byte == 34:
                self.quoted, self.after_quote = True, False
            elif byte in (10, 13):
                if self.header:
                    self._end_header()
                self.row = self.field = 0
                self.column = 0
                self.header = self.after_quote = False
                self.start = True
                self.after_cr = byte == 13
            elif byte == 44:
                self.field = 0
                self.column += 1
                self.start, self.after_quote = True, False
            else:
                self.quoted = self.start and byte == 34
                self.start = self.after_quote = False


def _snapshot(guard):
    return tuple(
        getattr(guard, name)
        for name in (
            "row",
            "field",
            "header",
            "start",
            "quoted",
            "after_quote",
            "after_cr",
            "column",
            "token_column",
        )
    ) + (bytes(guard.header_bytes), guard.budget[0])


def _feed(guard, chunk):
    try:
        guard.feed(chunk)
    except Exception as error:
        return type(error), error.args
    return None


def _compare(chunks, *, token_column=None, budget=1000000):
    reference, actual = _ReferenceBounds([budget]), coverage._CsvBounds([budget])
    if token_column is not None:
        # A fresh body record state reachable after a valid header.
        for guard in (reference, actual):
            guard.header = False
            guard.token_column = token_column
    for index, chunk in enumerate(chunks):
        expected = _feed(reference, chunk)
        observed = _feed(actual, chunk)
        assert (observed, _snapshot(actual)) == (expected, _snapshot(reference)), (
            index,
            chunk,
            observed,
            expected,
            _snapshot(actual),
            _snapshot(reference),
        )
        if expected is not None:
            break
    return actual


@pytest.mark.parametrize("token_column", [0, 1, 3])
def test_exhaustive_short_body_streams(monkeypatch, token_column):
    monkeypatch.setattr(coverage, "_ROW_MAX", 4)
    monkeypatch.setattr(coverage, "_TOKEN_MAX", 3)
    alphabet = (b"a", b'"', b",", b"\r", b"\n", b"\xc3", b"\0")
    for length in range(7):
        for parts in itertools.product(alphabet, repeat=length):
            raw = b"".join(parts)
            for chunks in ((raw,), parts, (raw[:2], raw[2:])):
                _compare(chunks, token_column=token_column, budget=130)


@pytest.mark.parametrize("row_limit", [0, 1, 4, 20])
@pytest.mark.parametrize("field_limit", [0, 1, 3, 20])
def test_exact_failure_state_and_budget_order(monkeypatch, row_limit, field_limit):
    monkeypatch.setattr(coverage, "_ROW_MAX", row_limit)
    monkeypatch.setattr(coverage, "_TOKEN_MAX", field_limit)
    for budget in (-1, 0, 117, 118, 119, 250):
        for raw in (b"", b"\n", b"\r\n\r\n", b"a,bc\n", b'"\na",b\n'):
            for token_column in (0, 1, 3):
                _compare((raw,), token_column=token_column, budget=budget)
                _compare(
                    tuple(bytes([v]) for v in raw),
                    token_column=token_column,
                    budget=budget,
                )


@pytest.mark.parametrize(
    "header",
    [
        b"PRPERTYP,x",
        b"x,PRPERTYP",
        b"x,PRPERTYP,y",
        b'"PRPERTYP",x',
        b'"x,\ny",PRPERTYP',
        b"PRPERTYP,PRPERTYP",
        b"x,y",
        b"\xff,PRPERTYP",
    ],
)
def test_real_header_and_all_byte_values(header):
    raw = header + b"\r\n" + bytes(range(256)) + b"\r\n"
    for width in (1, 2, 17, 65536):
        _compare(tuple(raw[i : i + width] for i in range(0, len(raw), width)))


def test_seeded_long_records_and_chunk_boundaries():
    rng = random.Random(881726)
    records = []
    for _ in range(50):
        fields = []
        for _ in range(80):
            token = bytes(rng.choice(b"0123456789xyz") for _ in range(rng.randrange(8)))
            if rng.randrange(12) == 0:
                token = b'"' + token + b',\r\n""more"'
            fields.append(token)
        records.append(b",".join(fields) + rng.choice((b"\n", b"\r", b"\r\n")))
    raw = b"x,PRPERTYP,y\r\n" + b"".join(records)
    for width in (1, 7, 63, 1024, 65536):
        _compare(tuple(raw[i : i + width] for i in range(0, len(raw), width)))
    _compare((raw,), budget=500)


def test_complete_unquoted_records_use_fast_path(monkeypatch):
    guard = coverage._CsvBounds([10000])
    guard.feed(b"x,PRPERTYP,y\n")
    monkeypatch.setattr(guard, "_slow_record", lambda *args: pytest.fail("slow path"))
    guard.feed(b"a,123,b\n\r\n0,456,2\r")
    assert guard.budget[0] == 10000 - 3 * 118 - 2 * 4
    assert guard.after_cr is True


@pytest.mark.parametrize("terminator", [b"\n", b"\r", b"\r\n"])
def test_every_nonstructural_byte_uses_fast_admission(monkeypatch, terminator):
    reference = _ReferenceBounds([1000000])
    actual = coverage._CsvBounds([1000000])
    for guard in (reference, actual):
        guard.feed(b"x,PRPERTYP,y\n")
    monkeypatch.setattr(actual, "_slow_record", lambda *args: pytest.fail("slow path"))
    for byte in range(256):
        if byte in (10, 13, 34, 44):
            continue
        token = bytes([byte])
        record = token + b"," + token + b"," + token + terminator
        reference.feed(record)
        actual.feed(record)
        assert _snapshot(actual) == _snapshot(reference), byte


def test_empty_crlf_records_are_each_charged():
    guard = _compare((b"\r\n\r\n",), token_column=0, budget=1000)
    assert guard.budget[0] == 1000 - 2 * (118 + 1)


def test_record_crossing_chunk_uses_unchanged_fallback():
    chunks = (b"PRPERTYP,x\r", b"\na,", b'"b\r', b'\n""c"\nnext,x', b"\n")
    _compare(chunks)


def test_long_unquoted_records_and_absent_newline_kind():
    fields = [str(i % 17).encode() for i in range(800)]
    for token_column in (0, 400, 799, 800):
        for terminator in (b"\n", b"\r", b"\r\n"):
            raw = (b",".join(fields) + terminator) * 25
            for width in (63, 65536, len(raw)):
                _compare(
                    tuple(raw[i : i + width] for i in range(0, len(raw), width)),
                    token_column=token_column,
                )
