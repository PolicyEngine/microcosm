"""Invented byte streams: preserve the original scanner's outputs and refusals."""

import io
import itertools

import pytest

from microcosm.build.us_runtime import acs_native_coverage_binding as native_owner
from microcosm.build.us_runtime import acs_person_coverage_authentication as owner


# Bytewise reference from a96fc850; only its name and owner bindings differ.
def _reference_records(stream):
    """Fence raw logical records/tokens before UTF-8 decoding or CSV allocation.

    Quote parity only locates boundaries; the unchanged strict literal parser
    subsequently owns CSV validity. CR, LF and CRLF are preserved, including
    inside quotes. The conservative token ceiling includes raw CSV quoting.
    """
    MAX_CSV_HEADER_BYTES = owner.MAX_CSV_HEADER_BYTES  # noqa: N806
    MAX_RECORD_BYTES = owner.MAX_RECORD_BYTES  # noqa: N806
    MAX_TOKEN_BYTES = owner.MAX_TOKEN_BYTES  # noqa: N806
    _require = owner._require
    record = bytearray()
    quoted, pending_cr, first, token_bytes = False, False, True, 0
    cap = MAX_CSV_HEADER_BYTES
    while block := stream.read(4096):
        for byte in block:
            if pending_cr:
                if byte == 10:
                    _require(len(record) < cap, "CSV_RECORD_BYTES")
                    record.append(byte)
                yield bytes(record)
                record.clear()
                first, pending_cr, token_bytes = False, False, 0
                if byte == 10:
                    continue
            cap = MAX_CSV_HEADER_BYTES if first else MAX_RECORD_BYTES
            _require(len(record) < cap, "CSV_RECORD_BYTES")
            if byte == 44 and not quoted:
                token_bytes = 0
            else:
                token_bytes += 1
                _require(token_bytes <= MAX_TOKEN_BYTES, "CSV_TOKEN_BYTES")
            record.append(byte)
            if byte == 34:
                quoted = not quoted
            if not quoted and byte in (10, 13):
                if byte == 13:
                    pending_cr = True
                else:
                    yield bytes(record)
                    record.clear()
                    first, token_bytes = False, 0
    if record:
        yield bytes(record)


class ShortReads(io.BytesIO):
    def __init__(self, data, chunk):
        super().__init__(data)
        self.chunk = chunk
        self.requests = []

    def read(self, size=-1):
        self.requests.append(size)
        assert size == 4096
        return super().read(min(size, self.chunk))


def _outcome(reader, data, chunk):
    stream = ShortReads(data, chunk)
    records = []
    try:
        records.extend(reader(stream))
    except owner.ACSCoverageAuthenticationError as exc:
        refusal = str(exc)
    else:
        refusal = None
    return records, refusal, stream.tell(), stream.requests


@pytest.mark.parametrize("chunk", [1, 2, 3, 4095, 4096])
@pytest.mark.parametrize(
    "data,expected",
    [
        (b"", []),
        (b"a,b", [b"a,b"]),
        (b"a,b\n", [b"a,b\n"]),
        (b"a,b\r", [b"a,b\r"]),
        (b"a,b\r\nx,y\r\n", [b"a,b\r\n", b"x,y\r\n"]),
        (b"a\rb\nc\r\n", [b"a\r", b"b\n", b"c\r\n"]),
        (b'"a\r\nb",c\n', [b'"a\r\nb",c\n']),
        (b'"a""b",c\r\n', [b'"a""b",c\r\n']),
        (b'"unclosed\nnext', [b'"unclosed\nnext']),
        (
            b"\xef\xbb\xbfA,B\n\xc3\xa9,\x00\t\n",
            [b"\xef\xbb\xbfA,B\n", b"\xc3\xa9,\x00\t\n"],
        ),
    ],
)
def test_literal_bytes_endings_and_quote_parity_are_unchanged(data, expected, chunk):
    actual = _outcome(owner._records, data, chunk)
    assert actual[:2] == (expected, None)
    assert actual == _outcome(_reference_records, data, chunk)


def test_exhaustive_small_streams_preserve_refusal_order_and_consumption(monkeypatch):
    alphabet = (b"a", b",", b'"', b"\r", b"\n")
    for header, body, token in ((9, 9, 9), (4, 3, 2), (1, 2, 1)):
        monkeypatch.setattr(owner, "MAX_CSV_HEADER_BYTES", header)
        monkeypatch.setattr(owner, "MAX_RECORD_BYTES", body)
        monkeypatch.setattr(owner, "MAX_TOKEN_BYTES", token)
        for size in range(5):
            for letters in itertools.product(alphabet, repeat=size):
                data = b"".join(letters)
                for chunk in (1, 3, 4096):
                    assert _outcome(owner._records, data, chunk) == _outcome(
                        _reference_records, data, chunk
                    )


@pytest.mark.parametrize("padding", [4093, 4094, 4095, 4096, 4097])
@pytest.mark.parametrize("ending", [b"\n", b"\r", b"\r\n", b""])
def test_read_boundary_and_long_quoted_tokens(padding, ending, monkeypatch):
    monkeypatch.setattr(owner, "MAX_TOKEN_BYTES", 20000)
    monkeypatch.setattr(owner, "MAX_CSV_HEADER_BYTES", 30000)
    monkeypatch.setattr(owner, "MAX_RECORD_BYTES", 30000)
    header = b"A,B\r\n"
    body = b'"' + b"x" * padding + b'\r\n""y",z' + ending
    for chunk in (1, 4095, 4096):
        actual = _outcome(owner._records, header + body, chunk)
        assert actual[:2] == ([header, body], None)
        assert actual == _outcome(_reference_records, header + body, chunk)


@pytest.mark.parametrize(
    "header,body,token,data,code",
    [
        (3, 8, 3, b"abcd", "CSV_RECORD_BYTES"),
        (4, 8, 2, b"abcd", "CSV_TOKEN_BYTES"),
        (2, 8, 4, b"abcd", "CSV_RECORD_BYTES"),
        (4, 8, 1, b"a,b\n", "CSV_TOKEN_BYTES"),
        (4, 3, 8, b"h\r\nabc\r\n", "CSV_RECORD_BYTES"),
        (4, 8, 2, b"h\nabc", "CSV_TOKEN_BYTES"),
        (4, 4, 4, b"h\nabc\r\n", "CSV_RECORD_BYTES"),
    ],
)
def test_exact_resource_refusal_precedence(
    header, body, token, data, code, monkeypatch
):
    monkeypatch.setattr(owner, "MAX_CSV_HEADER_BYTES", header)
    monkeypatch.setattr(owner, "MAX_RECORD_BYTES", body)
    monkeypatch.setattr(owner, "MAX_TOKEN_BYTES", token)
    for chunk in (1, 2, 4096):
        actual = _outcome(owner._records, data, chunk)
        assert actual[1] == code
        assert actual == _outcome(_reference_records, data, chunk)


@pytest.mark.parametrize("byte", [b",", b'"', b"\n", b"x"])
def test_delimiter_dense_blocks_do_not_repeat_suffix_searches(byte, monkeypatch):
    monkeypatch.setattr(owner, "MAX_TOKEN_BYTES", 8192)
    monkeypatch.setattr(owner, "MAX_CSV_HEADER_BYTES", 8192)
    monkeypatch.setattr(owner, "MAX_RECORD_BYTES", 8192)
    searched = []

    class TracedBlock(bytes):
        def find(self, delimiter, start=0, end=None):
            limit = len(self) if end is None else end
            found = super().find(delimiter, start, limit)
            searched.append((found + 1 if found >= 0 else limit) - start)
            return found

    class TracedStream(io.BytesIO):
        def read(self, size=-1):
            assert size == 4096
            return TracedBlock(super().read(size))

    data = byte * 4096
    assert list(owner._records(TracedStream(data))) == list(
        _reference_records(io.BytesIO(data))
    )
    # Each of four independent delimiter searches covers a byte at most once.
    assert sum(searched) <= 4 * len(data)


def test_reviewed_scanner_is_admitted_by_native_owner_and_old_pin_refuses(monkeypatch):
    producer = native_owner._producer()
    assert producer["accepted_age_transform"] == (
        "literal_numeric_identity_AGEP_to_AGEP_A_AGE_age"
    )
    monkeypatch.setitem(
        native_owner._ACCEPTED,
        "acs_person_coverage_authentication.py",
        "475aa795c8a5b49a0dd3405a0877866dddd03012a1f2b9743447fab1fe85bcff",
    )
    with pytest.raises(
        native_owner.ACSNativeCoverageBindingError, match="^UNREVIEWED_PREPARATION$"
    ):
        native_owner._producer()
