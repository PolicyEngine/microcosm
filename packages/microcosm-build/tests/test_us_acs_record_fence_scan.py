"""Byte-exact parity for the ACS record fence's chunked scan.

Invented byte strings only: no archive, no source tree, no engine.  ``_records``
now locates record boundaries with ``bytes.find``/``bytes.count`` over 1 MiB
blocks instead of a Python loop over individual bytes.  The sequence of yielded
records and the refusal it raises must be identical to the byte loop's for every
input, so every test here runs both implementations over the same bytes and
compares the whole outcome -- records and refusal together.

``_reference_records`` below is a verbatim copy of the pre-change body.
"""

from __future__ import annotations

import io
import random

import pytest

from microcosm.build.us_runtime import acs_person_coverage_authentication as owner

MAX_RECORD_BYTES = owner.MAX_RECORD_BYTES
MAX_CSV_HEADER_BYTES = owner.MAX_CSV_HEADER_BYTES
MAX_TOKEN_BYTES = owner.MAX_TOKEN_BYTES


def _reference_records(stream):
    """The exact pre-change body, kept verbatim as the parity reference."""

    record = bytearray()
    quoted, pending_cr, first, token_bytes = False, False, True, 0
    cap = MAX_CSV_HEADER_BYTES
    while block := stream.read(4096):
        for byte in block:
            if pending_cr:
                if byte == 10:
                    owner._require(len(record) < cap, "CSV_RECORD_BYTES")
                    record.append(byte)
                yield bytes(record)
                record.clear()
                first, pending_cr, token_bytes = False, False, 0
                if byte == 10:
                    continue
            cap = MAX_CSV_HEADER_BYTES if first else MAX_RECORD_BYTES
            owner._require(len(record) < cap, "CSV_RECORD_BYTES")
            if byte == 44 and not quoted:
                token_bytes = 0
            else:
                token_bytes += 1
                owner._require(token_bytes <= MAX_TOKEN_BYTES, "CSV_TOKEN_BYTES")
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


def _outcome(fence, data):
    """Every record the fence yielded, and the refusal code that stopped it."""

    records, refusal = [], None
    try:
        for record in fence(io.BytesIO(data)):
            records.append(record)
    except owner.ACSCoverageAuthenticationError as error:
        refusal = str(error)
    return tuple(records), refusal


def _assert_identical(data):
    reference = _outcome(_reference_records, data)
    assert _outcome(owner._records, data) == reference
    # The records must also reassemble the input exactly up to a refusal.
    if reference[1] is None:
        assert b"".join(reference[0]) == data
    return reference


BOUNDARIES = {
    "empty": b"",
    "no-terminator": b"a,b,c",
    "line-feed": b"a,b\nc,d\n",
    "carriage-line-feed": b"a,b\r\nc,d\r\n",
    "carriage-only": b"a,b\rc,d\r",
    "carriage-then-byte": b"a\rb\n",
    "mixed-endings": b"a\r\nb\rc\nd",
    "quoted-line-feed": b'"a\nb",c\nd,e\n',
    "quoted-carriage": b'"a\rb",c\nd,e\n',
    "quoted-carriage-line-feed": b'"a\r\nb",c\nd,e\n',
    "odd-quote-runs-on": b'a,"b\nc,d\n',
    "parity-carried-across-records": b'a"b\nc"d\ne\n',
    "byte-order-mark": b"\xef\xbb\xbfa,b\nc,d\n",
    "trailing-carriage-at-eof": b"a,b\r",
    "empty-lines": b"\n\n\n",
    "carriage-carriage": b"a\r\rb\n",
    "quote-inside-unquoted-field": b'a"b",c\n',
    "commas-only": b",,,,\n",
    "tab-and-space": b"a\t ,b\n",
    "nul-byte": b"a\x00b,c\n",
}


@pytest.mark.parametrize("name", sorted(BOUNDARIES))
def test_boundary_cases_fence_identically(name):
    _assert_identical(BOUNDARIES[name])


BLOCK = owner._SCAN_BLOCK
SPANNING = {
    "record-spans-two-blocks": b"x" * (BLOCK + 500) + b"\n",
    "carriage-line-feed-split-across-blocks": b"x" * (BLOCK - 1) + b"\r\ny\n",
    "carriage-at-eof-on-a-block-edge": b"x" * (BLOCK - 1) + b"\r",
    "carriage-then-byte-across-blocks": b"x" * (BLOCK - 1) + b"\rz\n",
    "quoted-region-spans-two-blocks": b'"' + b"x" * BLOCK + b'"\ny\n',
    "terminator-exactly-at-block-edge": b"x" * (BLOCK - 1) + b"\ny\n",
}


@pytest.mark.parametrize("name", sorted(SPANNING))
def test_block_boundary_cases_fence_identically(name):
    _assert_identical(SPANNING[name])


CEILINGS = {
    "first-record-one-over-its-cap": (
        b"x" * MAX_CSV_HEADER_BYTES + b"\n",
        "CSV_RECORD_BYTES",
    ),
    "first-record-exactly-at-its-cap": (
        b"x" * (MAX_CSV_HEADER_BYTES - 1) + b"\n",
        None,
    ),
    "later-record-token-one-over": (
        b"h\n" + b"x" * MAX_TOKEN_BYTES + b"\n",
        "CSV_TOKEN_BYTES",
    ),
    "later-record-token-exactly-at-cap": (
        b"h\n" + b"x" * (MAX_TOKEN_BYTES - 1) + b"\n",
        None,
    ),
    "later-record-many-short-tokens-over-cap": (
        b"h\n" + b",".join([b"y" * 1000] * 500) + b"\n",
        "CSV_RECORD_BYTES",
    ),
    "token-ceiling-fires-before-record-ceiling": (
        b"h\n" + b"a," * 100 + b"w" * (MAX_TOKEN_BYTES + 1) + b"\n",
        "CSV_TOKEN_BYTES",
    ),
    "unterminated-record-past-its-cap": (
        b"h\n" + b"q" * (MAX_RECORD_BYTES + 100),
        "CSV_TOKEN_BYTES",
    ),
    "quoted-terminators-do-not-reset-the-ceiling": (
        b"h\n" + b'"' + b"z" * MAX_TOKEN_BYTES + b"\n",
        "CSV_TOKEN_BYTES",
    ),
}


@pytest.mark.parametrize("name", sorted(CEILINGS))
def test_ceiling_cases_refuse_identically(name):
    data, expected = CEILINGS[name]
    assert _assert_identical(data)[1] == expected


def test_records_yielded_before_a_refusal_are_the_same_records():
    """A refusal stops the fence; everything it already yielded still matches."""

    data = b"h\n" + b"good,row\n" + b"x" * (MAX_TOKEN_BYTES + 1) + b"\n"
    records, refusal = _assert_identical(data)
    assert refusal == "CSV_TOKEN_BYTES"
    assert records == (b"h\n", b"good,row\n")


def test_randomised_byte_strings_fence_identically():
    """4,200 random strings over the alphabet the fence actually branches on."""

    alphabet = b'ab,"\r\n\t 0'
    generator = random.Random(20260915)
    for _ in range(4200):
        _assert_identical(
            bytes(generator.choice(alphabet) for _ in range(generator.randrange(0, 60)))
        )


def test_randomised_record_shaped_inputs_fence_identically():
    """Row-shaped inputs, each ending with one of the three line endings."""

    generator = random.Random(20260916)
    for _ in range(300):
        rows = []
        for _ in range(generator.randrange(0, 40)):
            fields = [
                bytes(
                    generator.choice(b"abc0123")
                    for _ in range(generator.randrange(0, 6))
                )
                for _ in range(generator.randrange(1, 6))
            ]
            if generator.random() < 0.2:
                fields[0] = b'"' + fields[0] + b'\n"'
            rows.append(b",".join(fields))
        ending = generator.choice([b"\n", b"\r\n", b"\r"])
        data = ending.join(rows)
        if generator.random() < 0.7:
            data += ending
        _assert_identical(data)


def test_quote_parity_still_persists_across_records():
    """The fence never resets parity at a boundary; both implementations agree."""

    records, refusal = _assert_identical(b'a"b\nc,d\ne"f\ng,h\n')
    assert refusal is None
    # The unbalanced quote swallows the next terminator, exactly as before.
    assert records == (b'a"b\nc,d\ne"f\n', b"g,h\n")


# The suite's other fence tests drive the ceilings by monkeypatching them to
# small values, which is the only practical way to exercise the bounded paths.
# The scan must agree with the byte loop under those ceilings too -- the
# short-circuit that skips the replay is bounded by the smaller of the two live
# ceilings, not by a compile-time relationship between them.
CEILING_SETTINGS = (
    {"MAX_RECORD_BYTES": 8},
    {"MAX_RECORD_BYTES": 12},
    {"MAX_CSV_HEADER_BYTES": 4},
    {"MAX_TOKEN_BYTES": 3},
    {"MAX_TOKEN_BYTES": 2, "MAX_RECORD_BYTES": 9},
    {"MAX_RECORD_BYTES": 5, "MAX_CSV_HEADER_BYTES": 9},
    {"MAX_TOKEN_BYTES": 100_000, "MAX_RECORD_BYTES": 6},
    {"MAX_CSV_HEADER_BYTES": 100_000, "MAX_TOKEN_BYTES": 4},
)


@pytest.mark.parametrize("settings", CEILING_SETTINGS, ids=repr)
def test_small_ceilings_refuse_identically(monkeypatch, settings):
    for name, value in settings.items():
        monkeypatch.setattr(owner, name, value)
    global MAX_RECORD_BYTES, MAX_CSV_HEADER_BYTES, MAX_TOKEN_BYTES
    previous = (MAX_RECORD_BYTES, MAX_CSV_HEADER_BYTES, MAX_TOKEN_BYTES)
    MAX_RECORD_BYTES = owner.MAX_RECORD_BYTES
    MAX_CSV_HEADER_BYTES = owner.MAX_CSV_HEADER_BYTES
    MAX_TOKEN_BYTES = owner.MAX_TOKEN_BYTES
    try:
        for data in BOUNDARIES.values():
            _assert_identical(data)
        generator = random.Random(20260917)
        alphabet = b'ab,"\r\n\t 0'
        for _ in range(600):
            _assert_identical(
                bytes(
                    generator.choice(alphabet)
                    for _ in range(generator.randrange(0, 40))
                )
            )
    finally:
        MAX_RECORD_BYTES, MAX_CSV_HEADER_BYTES, MAX_TOKEN_BYTES = previous


def test_the_small_ceiling_case_the_first_scan_got_wrong(monkeypatch):
    """A record cap below the token cap must still refuse at the record cap.

    The first version of the scan short-circuited on MAX_TOKEN_BYTES alone,
    which silently skipped the record ceiling whenever the record ceiling was
    the smaller of the two. This is that case, pinned.
    """

    body = b'"x\r\n\ty",z\n'
    monkeypatch.setattr(owner, "MAX_RECORD_BYTES", len(body) - 1)
    assert owner.MAX_RECORD_BYTES < owner.MAX_TOKEN_BYTES
    with pytest.raises(owner.ACSCoverageAuthenticationError, match="CSV_RECORD_BYTES"):
        list(owner._records(io.BytesIO(b"A,B\n" + body)))


def _exhaustive(alphabet, length):
    yield b""
    current = [b""]
    for _ in range(length):
        current = [
            prefix + bytes((symbol,)) for prefix in current for symbol in alphabet
        ]
        yield from current


EXHAUSTIVE_SETTINGS = (
    {},
    {"MAX_TOKEN_BYTES": 2},
    {"MAX_RECORD_BYTES": 3, "MAX_CSV_HEADER_BYTES": 3},
    {"MAX_RECORD_BYTES": 4, "MAX_TOKEN_BYTES": 2},
    {"MAX_CSV_HEADER_BYTES": 2, "MAX_RECORD_BYTES": 5, "MAX_TOKEN_BYTES": 4},
)


@pytest.mark.parametrize("settings", EXHAUSTIVE_SETTINGS, ids=repr)
def test_every_short_string_over_the_branching_alphabet(monkeypatch, settings):
    """Exhaustive: all 9,331 strings up to five bytes over the fence's alphabet.

    Every byte the fence branches on is in the alphabet -- comma, quote, CR, LF
    -- plus one ordinary byte and one whitespace byte. Under each ceiling
    setting the scan and the byte loop must agree on every one of them.
    """

    for name, value in settings.items():
        monkeypatch.setattr(owner, name, value)
    global MAX_RECORD_BYTES, MAX_CSV_HEADER_BYTES, MAX_TOKEN_BYTES
    previous = (MAX_RECORD_BYTES, MAX_CSV_HEADER_BYTES, MAX_TOKEN_BYTES)
    MAX_RECORD_BYTES = owner.MAX_RECORD_BYTES
    MAX_CSV_HEADER_BYTES = owner.MAX_CSV_HEADER_BYTES
    MAX_TOKEN_BYTES = owner.MAX_TOKEN_BYTES
    checked = 0
    try:
        for data in _exhaustive(b',"\r\n a', 5):
            assert _outcome(owner._records, data) == _outcome(
                _reference_records, data
            ), data
            checked += 1
    finally:
        MAX_RECORD_BYTES, MAX_CSV_HEADER_BYTES, MAX_TOKEN_BYTES = previous
    assert checked == 9331
