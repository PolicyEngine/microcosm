"""Frame-less ACS HU evidence transport; source and graph authority stay distinct.

Only a real all-source reader object can be exported. Replaying its typed
transport never constructs an AuthenticatedACSHousingSource or a Frame.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import InitVar, dataclass

from microcosm.graph import ArtifactType
from microcosm.graph.canonical import canonical_json
from microcosm.graph.keys import opaque_artifact_key

from . import acs_housing_universe_source as source

TYPE = ArtifactType("microcosm.us.acs_hu_source_projection", 1)
KERNEL_REF = "us.cd.acs_hu_source_evidence@1"
OUTPUT = "projection"
MAGIC = b"MCCDACS\x01"
MAX_BYTES = 64 * 1024**2
HEADER_MAX_BYTES = 65536
SCHEMA = "microcosm.us.acs-hu-source-transport.v1"
_TOKEN = object()


class SourceTransportError(ValueError):
    """Closed refusal codes contain no original row identities."""


def require(condition, code):
    if not condition:
        raise SourceTransportError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _tables(raw):
    return source._tables(source._parse_json(raw, MAX_BYTES))


def _receipt(raw):
    receipt = source._parse_json(raw, source.ACS_HU_RECEIPT_MAX_BYTES)
    require(
        receipt["format"] == "microcosm.acs_housing_universe_source.v1"
        and receipt["release_eligible"] is False
        and type(receipt["vintage"]) is int
        and receipt["vintage"] == 2024
        and receipt["selection"] == "all"
        and receipt["source_weight_kind"] == "design"
        and receipt["calibrated_descendants_approved"] is False
        and receipt["definition_sha256"] == source._definition()[1]
        and receipt["implementation_sha256"] == source._implementation(),
        "ACS_TRANSPORT_SOURCE_SCOPE",
    )
    require(
        canonical_json(receipt["archives"])
        == canonical_json(
            [
                {"role": role, "filename": name, "sha256": digest, "bytes": size}
                for role, name, digest, size in source._pins()
            ]
        ),
        "ACS_TRANSPORT_ARCHIVE_PINS",
    )
    return receipt


def _lineage(projection, receipt):
    household, person, *_ = source._tables(projection)
    for role, table in (("household", household), ("person", person)):
        members = {item["name"]: item for item in receipt["members"][role]}
        require(len(members) == len(receipt["members"][role]), "ACS_MEMBER_DUPLICATE")
        for member, ordinal in table[
            ["source_member", "source_row_ordinal"]
        ].itertuples(index=False, name=None):
            require(
                member in members and 0 < ordinal <= members[member]["rows"],
                "ACS_MEMBER_ROW_LINEAGE",
            )


def encode_source_projection(observation, *, selected_serialnos):
    """Preserve the authentic full receipt and derive a bounded selected view."""
    require(
        type(observation) is source.AuthenticatedACSHousingSource,
        "ACS_AUTHENTICATED_SOURCE_REQUIRED",
    )
    receipt = _receipt(observation.receipt_json)
    require(
        sha(observation.projection_json) == receipt["projection_sha256"]
        and len(observation.projection_json) == receipt["projection_bytes"],
        "ACS_SOURCE_PROJECTION_PIN",
    )
    projection = source._parse_json(
        observation.projection_json, source.ACS_HU_SOURCE_MAX_BYTES
    )
    selected, keys, counts = source._select(projection, selected_serialnos)
    require(
        canonical_json(counts) == canonical_json(receipt["full_counts"])
        and canonical_json(receipt["selected_counts"])
        == canonical_json(
            {"households": counts["households"], "persons": counts["persons"]}
        ),
        "ACS_FULL_CUSTODY",
    )
    view = source._json(selected)
    _lineage(selected, receipt)
    header = {
        "schema": SCHEMA,
        "release_eligible": False,
        "frame_authority": False,
        "source_receipt_bytes": len(observation.receipt_json),
        "source_receipt_sha256": sha(observation.receipt_json),
        "original_full_projection_sha256": receipt["full_projection_sha256"],
        "original_sorted_projection_sha256": receipt["projection_sha256"],
        "selected_projection_bytes": len(view),
        "selected_projection_sha256": sha(view),
        "selected_serialnos_sha256": sha(source._json(keys)),
        "selected_counts": {
            "households": len(selected["households"]),
            "persons": len(selected["persons"]),
        },
    }
    raw_header = canonical_json(header)
    require(len(raw_header) <= HEADER_MAX_BYTES, "ACS_HEADER_CAP")
    raw = MAGIC + struct.pack("<I", len(raw_header)) + raw_header
    raw += observation.receipt_json + view
    raw += bytes.fromhex(sha(raw))
    require(len(raw) <= MAX_BYTES, "ACS_TRANSPORT_CAP")
    return raw


@dataclass(frozen=True)
class BoundACSSourceProjection:
    """Verified immutable graph transport, explicitly not source authentication."""

    payload: bytes
    receipt_json: bytes
    projection_json: bytes
    header_json: bytes
    edge_json: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        require(_token is _TOKEN, "ACS_BOUND_CONSTRUCTOR")

    @property
    def receipt(self):
        return json.loads(self.receipt_json)

    @property
    def projection(self):
        return json.loads(self.projection_json)

    @property
    def header(self):
        return json.loads(self.header_json)

    @property
    def edge(self):
        return json.loads(self.edge_json)


def bind_source_projection(value, *, expected_edge, selected_serialnos):
    """Compare actual typed edge with an independent integrating graph pin."""
    try:
        edge = {
            "producer_key": value.producer_key,
            "artifact_key": value.key,
            "payload_sha256": sha(value.payload),
            "type": {
                "name": value.type.name,
                "schema_version": value.type.schema_version,
            },
        }
        require(
            value.type == TYPE
            and value.key == opaque_artifact_key(value.producer_key, OUTPUT)
            and canonical_json(edge) == canonical_json(expected_edge),
            "ACS_TRANSPORT_ACTUAL_EDGE",
        )
        raw = value.payload
        require(
            type(raw) is bytes
            and len(MAGIC) + 4 + 32 < len(raw) <= MAX_BYTES
            and raw.startswith(MAGIC)
            and sha(raw[:-32]) == raw[-32:].hex(),
            "ACS_TRANSPORT_PAYLOAD",
        )
        n = struct.unpack_from("<I", raw, len(MAGIC))[0]
        start = len(MAGIC) + 4
        require(
            0 < n <= HEADER_MAX_BYTES and start + n < len(raw) - 32, "ACS_HEADER_CAP"
        )
        header_bytes = raw[start : start + n]
        header = source._parse_json(header_bytes, HEADER_MAX_BYTES)
        expected_fields = {
            "schema",
            "release_eligible",
            "frame_authority",
            "source_receipt_bytes",
            "source_receipt_sha256",
            "original_full_projection_sha256",
            "original_sorted_projection_sha256",
            "selected_projection_bytes",
            "selected_projection_sha256",
            "selected_serialnos_sha256",
            "selected_counts",
        }
        require(
            set(header) == expected_fields
            and canonical_json(header) == header_bytes
            and header["schema"] == SCHEMA
            and header["release_eligible"] is False
            and header["frame_authority"] is False,
            "ACS_HEADER_SCHEMA",
        )
        nr, np = header["source_receipt_bytes"], header["selected_projection_bytes"]
        require(
            type(nr) is int
            and 0 < nr <= source.ACS_HU_RECEIPT_MAX_BYTES
            and type(np) is int
            and 0 < np <= MAX_BYTES
            and start + n + nr + np + 32 == len(raw),
            "ACS_SECTION_LENGTH",
        )
        receipt_bytes = raw[start + n : start + n + nr]
        view = raw[start + n + nr : -32]
        require(
            sha(receipt_bytes) == header["source_receipt_sha256"]
            and sha(view) == header["selected_projection_sha256"],
            "ACS_SECTION_PIN",
        )
        receipt = _receipt(receipt_bytes)
        require(
            header["original_full_projection_sha256"]
            == receipt["full_projection_sha256"]
            and header["original_sorted_projection_sha256"]
            == receipt["projection_sha256"],
            "ACS_ORIGINAL_CUSTODY",
        )
        projection = source._parse_json(view, MAX_BYTES)
        selected, keys, counts = source._select(projection, selected_serialnos)
        require(
            source._json(selected) == view
            and tuple(sorted(selected_serialnos)) == keys
            and sha(source._json(keys)) == header["selected_serialnos_sha256"]
            and canonical_json(header["selected_counts"])
            == canonical_json(
                {"households": counts["households"], "persons": counts["persons"]}
            )
            and all(counts[k] <= receipt["full_counts"][k] for k in counts),
            "ACS_SELECTED_CUSTODY",
        )
        _lineage(projection, receipt)
        return BoundACSSourceProjection(
            raw, receipt_bytes, view, header_bytes, canonical_json(edge), _token=_TOKEN
        )
    except SourceTransportError:
        raise
    except (AttributeError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        raise SourceTransportError("ACS_TRANSPORT_CONTRACT") from None
