"""Bounded, checksummed binary codec for target-only synthetic money artifacts."""

import struct
from hashlib import sha256

from .asec_current_money import (
    _STATE_TOKEN,
    HEADER_MAX_BYTES,
    MAX_HOUSEHOLDS,
    MAX_PERSONS,
    MoneyBindings,
    MoneyField,
    RestatedAsecMoney,
    SyntheticReadyCurrentMoney,
    _parse,
    _require,
    _validate_money,
    require_complete_current_money,
)

MAGIC = b"MCAM2024\x01"
# 32 person fields, one household observation, 8+1+1+1 bytes per field/row.
PAYLOAD_MAX_BYTES = (
    len(MAGIC) + 4 + HEADER_MAX_BYTES + 11 * (32 * MAX_PERSONS + MAX_HOUSEHOLDS) + 32
)


def encode_current_money(ready: SyntheticReadyCurrentMoney) -> bytes:
    """Encode immutable target amounts/statuses; never serialize nominal arrays."""
    _require(type(ready) is SyntheticReadyCurrentMoney, "SYNTHETIC_READY_REQUIRED")
    _validate_money(ready, ready.bindings.spec, nominal=False)
    # Recheck completeness even for a directly constructed or replaced value.
    require_complete_current_money(
        RestatedAsecMoney(ready.bindings, ready.fields, _token=_STATE_TOKEN),
        ready.bindings.spec,
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
    checksum = sha256()
    for part in parts:
        checksum.update(part)
    parts.append(checksum.digest())
    _require(sum(map(len, parts)) <= PAYLOAD_MAX_BYTES, "PAYLOAD_SIZE")
    return b"".join(parts)


def decode_current_money(
    payload: bytes, expected_bindings: MoneyBindings
) -> SyntheticReadyCurrentMoney:
    """Validate complete framing, expected scope/input/spec, and field invariants."""
    _require(
        type(payload) is bytes
        and len(MAGIC) + 4 + 32 < len(payload) <= PAYLOAD_MAX_BYTES,
        "PAYLOAD_SIZE",
    )
    _require(type(expected_bindings) is MoneyBindings, "EXPECTED_BINDINGS_REQUIRED")
    expected_bindings.__post_init__()
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
    value = RestatedAsecMoney(expected_bindings, tuple(fields), _token=_STATE_TOKEN)
    return require_complete_current_money(value, expected_bindings.spec)
