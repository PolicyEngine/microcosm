"""Bounded, checksummed codec preserving target-only money source authority."""

import struct
from hashlib import sha256

from .asec_current_money import (
    _STATE_TOKEN,
    HEADER_MAX_BYTES,
    MAX_HOUSEHOLDS,
    MAX_PERSONS,
    AuthenticatedAsecSource,
    MoneyBindings,
    MoneyField,
    ReadyCurrentMoney,
    RestatedAsecMoney,
    SyntheticReadyCurrentMoney,
    _parse,
    _require,
    _validate_money,
    require_complete_current_money,
)

MAGIC = b"MCAM2024\x01"
_CONTENT_DOMAIN = b"microcosm/asec-current-money-content/1\0"
# 32 person fields, one household observation, 8+1+1+1 bytes per field/row.
PAYLOAD_MAX_BYTES = (
    len(MAGIC) + 4 + HEADER_MAX_BYTES + 11 * (32 * MAX_PERSONS + MAX_HOUSEHOLDS) + 32
)


def _ready_parts(
    ready: SyntheticReadyCurrentMoney | ReadyCurrentMoney,
) -> tuple[bytes, ...]:
    """One canonical serialization shared by transport and content identity."""
    _require(
        type(ready) in (SyntheticReadyCurrentMoney, ReadyCurrentMoney),
        "SYNTHETIC_READY_REQUIRED",
    )
    _validate_money(ready, ready.bindings.spec, nominal=False)
    # Recheck completeness even for a directly constructed or replaced value.
    require_complete_current_money(
        RestatedAsecMoney(ready.bindings, ready.fields, _token=_STATE_TOKEN),
        ready.bindings.spec,
        production=type(ready) is ReadyCurrentMoney,
    )
    parts = [MAGIC, struct.pack("<I", len(ready.header)), ready.header]
    for field in ready.fields:
        parts.extend(
            (
                field.amount_bytes,
                field.status_bytes,
                field.validity_bytes,
                field.zero_origin_bytes,
            )
        )
    _require(sum(map(len, parts)) + 32 <= PAYLOAD_MAX_BYTES, "PAYLOAD_SIZE")
    return tuple(parts)


def current_money_content_sha256(
    ready: SyntheticReadyCurrentMoney | ReadyCurrentMoney,
) -> str:
    """Bind the full canonical header and field body, apart from transport checksum."""
    digest = sha256(_CONTENT_DOMAIN)
    for part in _ready_parts(ready):
        digest.update(part)
    return digest.hexdigest()


def encode_current_money(
    ready: SyntheticReadyCurrentMoney | ReadyCurrentMoney,
) -> bytes:
    """Encode immutable target amounts and evidence, including field-scoped origins.

    Schema 1 retains legacy zero provenance. Schema 2 binds the restored-source
    PTOTVAL origin policy through the exact header/spec; the byte widths and
    checksum framing are unchanged. Per-field validation rejects swapped codes.
    """
    parts = _ready_parts(ready)
    checksum = sha256()
    for part in parts:
        checksum.update(part)
    return b"".join((*parts, checksum.digest()))


def decode_current_money(
    payload: bytes,
    expected_bindings: MoneyBindings,
    *,
    expected: ReadyCurrentMoney | None = None,
) -> SyntheticReadyCurrentMoney | ReadyCurrentMoney:
    """Authenticate replay content against issued readiness before state creation.

    Verified source bindings alone cannot authorize a restated amount body.
    Synthetic replay retains its binding-only API and refuses an expected object.
    """
    _require(
        type(payload) is bytes
        and len(MAGIC) + 4 + 32 < len(payload) <= PAYLOAD_MAX_BYTES,
        "PAYLOAD_SIZE",
    )
    _require(type(expected_bindings) is MoneyBindings, "EXPECTED_BINDINGS_REQUIRED")
    expected_bindings.__post_init__()
    authenticated = type(expected_bindings.spec.source) is AuthenticatedAsecSource
    if authenticated:
        _require(type(expected) is ReadyCurrentMoney, "EXPECTED_CONTENT_REQUIRED")
        _require(expected.bindings == expected_bindings, "EXPECTED_CONTENT_BINDING")
        _validate_money(expected, expected_bindings.spec, nominal=False)
    else:
        _require(expected is None, "EXPECTED_CONTENT_SYNTHETIC")
    view = memoryview(payload)
    _require(payload.startswith(MAGIC), "PAYLOAD_VERSION")
    size = struct.unpack_from("<I", payload, len(MAGIC))[0]
    start = len(MAGIC) + 4
    _require(
        0 < size <= HEADER_MAX_BYTES and start + size <= len(payload) - 32,
        "HEADER_SIZE",
    )
    header = payload[start : start + size]
    data = _parse(header, HEADER_MAX_BYTES)
    _require(header == expected_bindings.header, "REPLAY_BINDING")
    expected_size = (
        start + size + 11 * (32 * data["person_rows"] + data["household_rows"]) + 32
    )
    _require(len(payload) == expected_size, "PAYLOAD_LENGTH")
    _require(sha256(view[:-32]).digest() == payload[-32:], "PAYLOAD_CHECKSUM")
    cursor = start + size
    fields = []
    for domain in expected_bindings.spec.fields:
        count = data[domain.entity + "_rows"]
        buffers = []
        for width in (8, 1, 1, 1):
            buffers.append(payload[cursor : cursor + width * count])
            cursor += width * count
        fields.append(MoneyField(domain.name, *buffers))
    if authenticated:
        _require(tuple(fields) == expected.fields, "REPLAY_CONTENT")
    value = RestatedAsecMoney(expected_bindings, tuple(fields), _token=_STATE_TOKEN)
    return require_complete_current_money(
        value,
        expected_bindings.spec,
        production=data["source_authentication"] == "checkpoint_bytes_verified",
    )
