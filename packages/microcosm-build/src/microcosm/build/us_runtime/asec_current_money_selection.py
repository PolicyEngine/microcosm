"""Typed subset of an authenticated ASEC current-money body.

The authenticated ``ReadyCurrentMoney`` object never leaves the preparation
process. Its canonical encoded bytes travel as a typed graph artifact, and this
module parses those bytes into a *body* that carries amounts and evidence
without carrying source readiness. Selecting whole households from a body
produces a second typed artifact whose header names its parent and the exact
selected coordinates.

Neither type subclasses ``RestatedAsecMoney``, so every existing
``type(x) is ReadyCurrentMoney`` check refuses them: a subset can never be
minted as full-source readiness, and full readiness can only originate in the
reviewed source loaders.
"""

from __future__ import annotations

import struct
from dataclasses import InitVar, dataclass
from hashlib import sha256

import numpy as np

from microcosm.graph import ArtifactType

from ._asec_current_money_codec import _CONTENT_DOMAIN
from ._asec_current_money_codec import MAGIC as BODY_MAGIC
from ._asec_current_money_codec import PAYLOAD_MAX_BYTES as BODY_PAYLOAD_MAX_BYTES
from .asec_current_money import (
    FIELDS,
    HEADER_MAX_BYTES,
    MAX_HOUSEHOLDS,
    MAX_PERSONS,
    RECIPE,
    MoneyField,
    MoneyRefusalError,
    _digest,
    _json,
    _parse,
    _require,
    _sha,
)

US_ASEC_CURRENT_MONEY_BODY_TYPE = ArtifactType(
    "microcosm.us.asec_current_money_body", 1
)
US_ASEC_PREPARED_RECEIPT_TYPE = ArtifactType("microcosm.us.asec_prepared_receipt", 3)
US_ASEC_SELECTION_TYPE = ArtifactType("microcosm.us.asec_household_selection", 2)
US_ASEC_SELECTED_MONEY_TYPE = ArtifactType(
    "microcosm.us.asec_selected_current_money", 1
)

SELECTED_MAGIC = b"MCASELM1\x01"
SELECTED_ARTIFACT_KIND = "microcosm.us.asec_selected_current_money.v1"
_ENTITIES = ("person", "household")
_BODY_TOKEN = object()
_SELECTED_TOKEN = object()
_SELECTED_HEADER_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "release_eligible",
        "recipe",
        "target_year",
        "state",
        "semantic",
        "source_authentication",
        "parent_header_sha256",
        "parent_content_sha256",
        "prepared_receipt_sha256",
        "selection_sha256",
        "person_rows",
        "household_rows",
        "person_identity_sha256",
        "household_identity_sha256",
        "fields",
        "field_entities",
        "dtype",
        "evidence_dtype",
    }
)
# 32 person fields plus one household observation, four evidence axes each.
SELECTED_PAYLOAD_MAX_BYTES = (
    len(SELECTED_MAGIC)
    + 4
    + HEADER_MAX_BYTES
    + 11 * (32 * MAX_PERSONS + MAX_HOUSEHOLDS)
    + 32
)


def _entities(field_entities) -> tuple[str, ...]:
    """Validate a producer-declared field/entity roster against the closed fields."""
    _require(
        type(field_entities) is tuple and len(field_entities) == len(FIELDS),
        "FIELD_ENTITY_ROSTER",
    )
    names = []
    for item in field_entities:
        _require(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and item[1] in _ENTITIES,
            "FIELD_ENTITY_ROSTER",
        )
        names.append(item[0])
    _require(tuple(names) == FIELDS, "FIELD_ENTITY_ROSTER")
    return tuple(item[1] for item in field_entities)


def _decoded_entities(data: dict) -> tuple[str, ...]:
    roster = data.get("field_entities")
    _require(
        type(roster) is list
        and all(type(item) is list and len(item) == 2 for item in roster),
        "FIELD_ENTITY_ROSTER",
    )
    return _entities(tuple(tuple(item) for item in roster))


def _positions(values: np.ndarray, *, rows: int, reason: str) -> np.ndarray:
    _require(
        type(values) is np.ndarray
        and values.dtype == np.dtype("int64")
        and values.ndim == 1,
        reason,
    )
    _require(0 < len(values) <= rows, reason)
    _require((values[:-1] < values[1:]).all() if len(values) > 1 else True, reason)
    _require(int(values[0]) >= 0 and int(values[-1]) < rows, reason)
    return values


def _split(
    payload: bytes, cursor: int, entities, counts: dict[str, int]
) -> tuple[MoneyField, ...]:
    fields = []
    for name, entity in zip(FIELDS, entities, strict=True):
        count = counts[entity]
        buffers = []
        for width in (8, 1, 1, 1):
            buffers.append(payload[cursor : cursor + width * count])
            cursor += width * count
        fields.append(MoneyField(name, *buffers))
    return tuple(fields)


def _field_shapes(fields, entities, counts: dict[str, int]) -> None:
    for field, entity in zip(fields, entities, strict=True):
        count = counts[entity]
        widths = (
            len(field.amount_bytes),
            len(field.status_bytes),
            len(field.validity_bytes),
            len(field.zero_origin_bytes),
        )
        _require(widths == (8 * count, count, count, count), "FIELD_BUFFER_SHAPE")
        amounts = field.amounts
        _require(np.isfinite(amounts).all(), "FIELD_ENCODING", field.name)
        _require(
            np.isin(field.statuses, [0, 1, 2, 3, 4]).all()
            and np.isin(field.validity, [0, 1]).all()
            and np.isin(field.zero_origin, [0, 1, 2]).all(),
            "FIELD_ENCODING",
            field.name,
        )
        _require(field.validity.all(), "MISSING_REQUIRED_AMOUNT", field.name)


@dataclass(frozen=True)
class BoundCurrentMoneyBody:
    """Amount/evidence buffers bound to a producer-declared body identity.

    This is emphatically not readiness: it carries no spec, no source evidence
    and no authenticated authority, only bytes that a named producer already
    authenticated inside its own process.
    """

    header: bytes
    content_sha256: str
    field_entities: tuple[tuple[str, str], ...]
    fields: tuple[MoneyField, ...]
    person_rows: int
    household_rows: int
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _BODY_TOKEN, "BODY_CONSTRUCTOR_UNAVAILABLE")

    @property
    def header_data(self) -> dict:
        return _parse(self.header, HEADER_MAX_BYTES)

    @property
    def header_sha256(self) -> str:
        return _sha(self.header)

    def field(self, name: str) -> MoneyField:
        _require(name in FIELDS, "UNKNOWN_FIELD")
        return self.fields[FIELDS.index(name)]

    def entity_of(self, name: str) -> str:
        _require(name in FIELDS, "UNKNOWN_FIELD")
        return self.field_entities[FIELDS.index(name)][1]


@dataclass(frozen=True)
class SelectedCurrentMoney:
    """Whole-household subset of a body, bound to its exact selected coordinates."""

    header: bytes
    fields: tuple[MoneyField, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _SELECTED_TOKEN, "SELECTED_CONSTRUCTOR_UNAVAILABLE")

    @property
    def header_data(self) -> dict:
        return _parse(self.header, HEADER_MAX_BYTES)

    @property
    def header_sha256(self) -> str:
        return _sha(self.header)

    @property
    def person_rows(self) -> int:
        return self.header_data["person_rows"]

    @property
    def household_rows(self) -> int:
        return self.header_data["household_rows"]

    def field(self, name: str) -> MoneyField:
        _require(name in FIELDS, "UNKNOWN_FIELD")
        return self.fields[FIELDS.index(name)]

    def entity_of(self, name: str) -> str:
        _require(name in FIELDS, "UNKNOWN_FIELD")
        return self.header_data["field_entities"][FIELDS.index(name)][1]

    def amounts(self, name: str) -> np.ndarray:
        """Return a writable copy of one field's selected target-year amounts."""
        return self.field(name).amounts.copy()


def parse_current_money_body(
    payload: bytes,
    *,
    expected_header_sha256: str,
    expected_content_sha256: str,
    field_entities: tuple[tuple[str, str], ...],
) -> BoundCurrentMoneyBody:
    """Parse canonical money bytes without reconstructing replayable readiness.

    The caller supplies the producer's header and content digests; both must
    match exactly. Digests, not header bytes, are what a producer receipt
    carries across a graph edge, so binding on them keeps the consumer honest
    without re-transporting the header. This deliberately cannot mint a
    ``ReadyCurrentMoney``: the authenticated replay contract stays with
    ``decode_current_money``.
    """
    _require(
        type(payload) is bytes
        and len(BODY_MAGIC) + 4 + 32 < len(payload) <= BODY_PAYLOAD_MAX_BYTES,
        "PAYLOAD_SIZE",
    )
    _require(_digest(expected_header_sha256), "EXPECTED_HEADER_DIGEST")
    _require(_digest(expected_content_sha256), "EXPECTED_CONTENT_DIGEST")
    entities = _entities(field_entities)
    _require(payload.startswith(BODY_MAGIC), "PAYLOAD_VERSION")
    size = struct.unpack_from("<I", payload, len(BODY_MAGIC))[0]
    start = len(BODY_MAGIC) + 4
    _require(
        0 < size <= HEADER_MAX_BYTES and start + size <= len(payload) - 32,
        "HEADER_SIZE",
    )
    header = payload[start : start + size]
    _require(_sha(header) == expected_header_sha256, "BODY_HEADER_BINDING")
    data = _parse(header, HEADER_MAX_BYTES)
    _require(
        data.get("recipe") == RECIPE
        and data.get("target_year") == 2024
        and data.get("state") == "target_current"
        and data.get("semantic") == "annual_current_money"
        and data.get("dtype") == "<f8"
        and data.get("evidence_dtype") == "|u1"
        and data.get("fields") == list(FIELDS),
        "BODY_HEADER_SCHEMA",
    )
    counts = {}
    for entity in _ENTITIES:
        rows = data.get(entity + "_rows")
        limit = MAX_PERSONS if entity == "person" else MAX_HOUSEHOLDS
        _require(type(rows) is int and 0 < rows <= limit, "BODY_ROW_BOUND")
        counts[entity] = rows
    expected_size = (
        start + size + 11 * (sum(counts[entity] for entity in entities)) + 32
    )
    _require(len(payload) == expected_size, "PAYLOAD_LENGTH")
    _require(
        sha256(memoryview(payload)[:-32]).digest() == payload[-32:], "PAYLOAD_CHECKSUM"
    )
    content = sha256(_CONTENT_DOMAIN)
    content.update(memoryview(payload)[:-32])
    _require(content.hexdigest() == expected_content_sha256, "BODY_CONTENT_BINDING")
    fields = _split(payload, start + size, entities, counts)
    _field_shapes(fields, entities, counts)
    return BoundCurrentMoneyBody(
        header,
        expected_content_sha256,
        tuple(field_entities),
        fields,
        counts["person"],
        counts["household"],
        _token=_BODY_TOKEN,
    )


def select_current_money(
    body: BoundCurrentMoneyBody,
    *,
    person_positions: np.ndarray,
    household_positions: np.ndarray,
    prepared_receipt_sha256: str,
    selection_sha256: str,
    person_identity_sha256: str,
    household_identity_sha256: str,
) -> SelectedCurrentMoney:
    """Slice a body to selected positions and bind the selected coordinates."""
    _require(type(body) is BoundCurrentMoneyBody, "TYPED_BODY_REQUIRED")
    _require(
        all(
            _digest(value)
            for value in (
                prepared_receipt_sha256,
                selection_sha256,
                person_identity_sha256,
                household_identity_sha256,
            )
        ),
        "SELECTION_BINDING_DIGEST",
    )
    entities = _entities(body.field_entities)
    positions = {
        "person": _positions(
            person_positions, rows=body.person_rows, reason="PERSON_POSITIONS"
        ),
        "household": _positions(
            household_positions,
            rows=body.household_rows,
            reason="HOUSEHOLD_POSITIONS",
        ),
    }
    selected = []
    for field, entity in zip(body.fields, entities, strict=True):
        index = positions[entity]
        selected.append(
            MoneyField(
                field.name,
                field.amounts[index].tobytes(),
                field.statuses[index].tobytes(),
                field.validity[index].tobytes(),
                field.zero_origin[index].tobytes(),
            )
        )
    parent = body.header_data
    header = _json(
        {
            "schema_version": 1,
            "artifact_kind": SELECTED_ARTIFACT_KIND,
            "release_eligible": False,
            "recipe": RECIPE,
            "target_year": 2024,
            "state": "target_current",
            "semantic": "annual_current_money",
            "source_authentication": parent["source_authentication"],
            "parent_header_sha256": body.header_sha256,
            "parent_content_sha256": body.content_sha256,
            "prepared_receipt_sha256": prepared_receipt_sha256,
            "selection_sha256": selection_sha256,
            "person_rows": len(positions["person"]),
            "household_rows": len(positions["household"]),
            "person_identity_sha256": person_identity_sha256,
            "household_identity_sha256": household_identity_sha256,
            "fields": list(FIELDS),
            "field_entities": [list(item) for item in body.field_entities],
            "dtype": "<f8",
            "evidence_dtype": "|u1",
        }
    )
    value = SelectedCurrentMoney(header, tuple(selected), _token=_SELECTED_TOKEN)
    _validate_selected(value)
    return value


def _validate_selected(selected: SelectedCurrentMoney) -> None:
    data = _parse(selected.header, HEADER_MAX_BYTES)
    _require(set(data) == _SELECTED_HEADER_KEYS, "SELECTED_HEADER_SCHEMA")
    _require(selected.header == _json(data), "SELECTED_HEADER_SCHEMA")
    _require(
        data["schema_version"] == 1
        and data["artifact_kind"] == SELECTED_ARTIFACT_KIND
        and data["release_eligible"] is False
        and data["recipe"] == RECIPE
        and data["target_year"] == 2024
        and data["state"] == "target_current"
        and data["semantic"] == "annual_current_money"
        and data["dtype"] == "<f8"
        and data["evidence_dtype"] == "|u1"
        and data["fields"] == list(FIELDS),
        "SELECTED_HEADER_BINDING",
    )
    _require(
        all(
            _digest(data[name])
            for name in (
                "parent_header_sha256",
                "parent_content_sha256",
                "prepared_receipt_sha256",
                "selection_sha256",
                "person_identity_sha256",
                "household_identity_sha256",
            )
        ),
        "SELECTED_HEADER_DIGEST",
    )
    entities = _decoded_entities(data)
    counts = {}
    for entity in _ENTITIES:
        rows = data[entity + "_rows"]
        limit = MAX_PERSONS if entity == "person" else MAX_HOUSEHOLDS
        _require(type(rows) is int and 0 < rows <= limit, "SELECTED_ROW_BOUND")
        counts[entity] = rows
    _require(
        type(selected.fields) is tuple and len(selected.fields) == len(FIELDS),
        "FIELD_ROSTER",
    )
    _require(tuple(field.name for field in selected.fields) == FIELDS, "FIELD_ROSTER")
    _field_shapes(selected.fields, entities, counts)


def _selected_parts(selected: SelectedCurrentMoney) -> tuple[bytes, ...]:
    _require(type(selected) is SelectedCurrentMoney, "TYPED_SELECTED_REQUIRED")
    _validate_selected(selected)
    parts = [
        SELECTED_MAGIC,
        struct.pack("<I", len(selected.header)),
        selected.header,
    ]
    for field in selected.fields:
        parts.extend(
            (
                field.amount_bytes,
                field.status_bytes,
                field.validity_bytes,
                field.zero_origin_bytes,
            )
        )
    _require(sum(map(len, parts)) + 32 <= SELECTED_PAYLOAD_MAX_BYTES, "PAYLOAD_SIZE")
    return tuple(parts)


def encode_selected_current_money(selected: SelectedCurrentMoney) -> bytes:
    """Encode the selected subset with a transport checksum over its bytes."""
    parts = _selected_parts(selected)
    checksum = sha256()
    for part in parts:
        checksum.update(part)
    return b"".join((*parts, checksum.digest()))


def decode_selected_current_money(
    payload: bytes,
    *,
    expected_parent_header_sha256: str,
    expected_parent_content_sha256: str,
    expected_prepared_receipt_sha256: str,
    expected_selection_sha256: str,
) -> SelectedCurrentMoney:
    """Decode a selected subset only against its declared parent and selection."""
    _require(
        type(payload) is bytes
        and len(SELECTED_MAGIC) + 4 + 32 < len(payload) <= SELECTED_PAYLOAD_MAX_BYTES,
        "PAYLOAD_SIZE",
    )
    _require(
        all(
            _digest(value)
            for value in (
                expected_parent_header_sha256,
                expected_parent_content_sha256,
                expected_prepared_receipt_sha256,
                expected_selection_sha256,
            )
        ),
        "SELECTION_BINDING_DIGEST",
    )
    _require(payload.startswith(SELECTED_MAGIC), "PAYLOAD_VERSION")
    size = struct.unpack_from("<I", payload, len(SELECTED_MAGIC))[0]
    start = len(SELECTED_MAGIC) + 4
    _require(
        0 < size <= HEADER_MAX_BYTES and start + size <= len(payload) - 32,
        "HEADER_SIZE",
    )
    header = payload[start : start + size]
    data = _parse(header, HEADER_MAX_BYTES)
    _require(set(data) == _SELECTED_HEADER_KEYS, "SELECTED_HEADER_SCHEMA")
    _require(
        data.get("parent_header_sha256") == expected_parent_header_sha256
        and data.get("parent_content_sha256") == expected_parent_content_sha256
        and data.get("prepared_receipt_sha256") == expected_prepared_receipt_sha256
        and data.get("selection_sha256") == expected_selection_sha256,
        "SELECTED_REPLAY_BINDING",
    )
    entities = _decoded_entities(data)
    counts = {entity: data[entity + "_rows"] for entity in _ENTITIES}
    _require(
        all(type(value) is int and value > 0 for value in counts.values()),
        "SELECTED_ROW_BOUND",
    )
    expected_size = start + size + 11 * sum(counts[entity] for entity in entities) + 32
    _require(len(payload) == expected_size, "PAYLOAD_LENGTH")
    _require(
        sha256(memoryview(payload)[:-32]).digest() == payload[-32:], "PAYLOAD_CHECKSUM"
    )
    fields = _split(payload, start + size, entities, counts)
    value = SelectedCurrentMoney(header, fields, _token=_SELECTED_TOKEN)
    _validate_selected(value)
    return value


__all__ = [
    "BoundCurrentMoneyBody",
    "MoneyRefusalError",
    "SELECTED_ARTIFACT_KIND",
    "SELECTED_MAGIC",
    "SELECTED_PAYLOAD_MAX_BYTES",
    "SelectedCurrentMoney",
    "US_ASEC_CURRENT_MONEY_BODY_TYPE",
    "US_ASEC_PREPARED_RECEIPT_TYPE",
    "US_ASEC_SELECTED_MONEY_TYPE",
    "US_ASEC_SELECTION_TYPE",
    "decode_selected_current_money",
    "encode_selected_current_money",
    "parse_current_money_body",
    "select_current_money",
]
