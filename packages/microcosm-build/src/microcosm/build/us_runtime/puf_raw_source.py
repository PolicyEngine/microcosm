"""The first typed artifact read straight from the 2015 PUF delivery bytes.

One ordinary graph node reads the two accepted parent captures through a
dedicated pinned raw-byte codec and produces one opaque, typed artifact:
``microcosm.us.puf_2015_raw_return_status`` at source-return grain, 207,696
rows in delivered main-file order.

What it types is deliberately small. Six of the 222 main columns (``RECID``,
``FLPDYR``, ``FLPDMO``, ``MARS``, ``DSI``, ``S006``) and the six demographic
columns of the companion file. The other 216 main columns are checked for
header name, order and record width and then dropped: no amount is parsed,
no growth factor is admitted, no household or person is invented, and the
node owns no population columns at all. Its host population is whatever the
caller names.

Three separations are load-bearing.

*Structure is not provenance.* :func:`decode_puf_raw_source` validates CSV
structure against a definition it is handed. It authorizes nothing. The
production wrapper — :func:`puf_raw_source_codecs` and
:class:`USPufRawSourceKernel` — is what binds the closed packaged pins, and
it refuses any definition whose canonical bytes are not the packaged file's.

*Fixtures take an explicit non-production route.* :func:`fixture_definition`
builds a definition marked ``route="test_fixture"``, which must carry the
invented-fixture authority and must not restate a packaged source pin. There
is no "expected hash omitted, so admit anything" path anywhere.

*The private cap is private.* :data:`PUF_RAW_SOURCE_MAX_BYTES` is 128 MiB
because the pinned main delivery is 126,034,649 bytes.
:data:`microcosm.graph.codecs.RAW_BYTES_MAX_BYTES` stays at 64 MiB.

Period semantics stay unresolved on purpose. The header audit observed
``FLPDYR`` in {2012, 2013, 2014, 2015} while the booklet prints 2011–2014;
the artifact records the raw code and the conflict, and resolves neither.
``S006`` is kept as exact integer hundredths and never divided.

The four publisher aggregate records stay in the artifact, tagged
``disclosure_aggregate``, never disaggregated. The 88,021 returns with no
demographic record carry ``demographic_status = 0``, the ``-1`` sentinel and
empty lexical columns; absence is never spelled as a zero code.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from importlib import resources
from pathlib import Path
from types import MappingProxyType

import numpy as np

from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    SourceRef,
    StructuralDelta,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SourceCodecRegistry, load_source_bytes

__all__ = [
    "BODY_MAX_BYTES",
    "DEMOGRAPHIC_CODEC",
    "DEMOGRAPHIC_SOURCE_NAME",
    "HEADER_MAX_BYTES",
    "MAIN_CODEC",
    "MAIN_SOURCE_NAME",
    "PUF_AGGREGATE_RECIDS",
    "PUF_RAW_SOURCE_MAX_BYTES",
    "PUF_RAW_SOURCE_NODE",
    "PUF_RAW_SOURCE_STAGE",
    "RETURN_STATUS_SUMMARY_NODE",
    "RETURN_STATUS_TYPE",
    "PufRawReturnStatus",
    "PufRawSourceDefinition",
    "PufRawSourceRefusalError",
    "SourcePin",
    "USPufRawReturnStatusSummaryKernel",
    "USPufRawSourceKernel",
    "csv_acceptance_profile",
    "decode_puf_raw_source",
    "definition_document_json",
    "decode_return_status",
    "encode_return_status",
    "fixture_definition",
    "load_pinned_puf_bytes",
    "packaged_definition",
    "puf_raw_source_codecs",
    "register_us_puf_raw_source_codecs",
    "register_us_puf_raw_source_kernels",
    "us_puf_raw_return_status_summary_node",
    "us_puf_raw_source_node",
    "us_puf_raw_source_refs",
]

#: The most this module reads from one PUF delivery file (128 MiB). Private:
#: the shared ``raw-bytes-v1`` cap is untouched at 64 MiB.
PUF_RAW_SOURCE_MAX_BYTES = 128 * 1024 * 1024

#: Bound on the encoded artifact's column bodies, below the 64 MiB a shared
#: opaque payload is expected to stay under. The full slice is ~35 MB.
BODY_MAX_BYTES = 64 * 1024 * 1024

#: Bound on the canonical JSON header, as in the ACS rent placement envelope.
HEADER_MAX_BYTES = 64 * 1024

RETURN_STATUS_TYPE = ArtifactType("microcosm.us.puf_2015_raw_return_status", 1)
RETURN_STATUS_MAGIC = b"microcosm.us.puf_2015_raw_return_status/1\n"

MAIN_SOURCE_NAME = "puf_2015_main"
DEMOGRAPHIC_SOURCE_NAME = "puf_2015_demographic"
MAIN_CODEC = "us-puf-2015-main-raw-v1"
DEMOGRAPHIC_CODEC = "us-puf-2015-demographic-raw-v1"

PUF_RAW_SOURCE_STAGE = "us_puf_raw_source"
PUF_RAW_SOURCE_NODE = f"{PUF_RAW_SOURCE_STAGE}.return_status"
RETURN_STATUS_SUMMARY_NODE = f"{PUF_RAW_SOURCE_STAGE}.return_status_summary"

#: The publisher's four aggregate record identifiers. The literal values, not
#: an import of ``puf_aggregate_records``: that module pulls nine derivation
#: modules and the calibrate package behind it.
PUF_AGGREGATE_RECIDS = (999996, 999997, 999998, 999999)

_DEFINITION_RESOURCE = "puf_2015_raw_source_definition.json"
_DEFINITION_PACKAGE = "microcosm.build.us_runtime"
_FIXTURE_AUTHORITY = "invented_fixture_nonauthority"

_MAIN_FIELDS = ("RECID", "FLPDYR", "FLPDMO", "MARS", "DSI", "S006")
_DEMOGRAPHIC_FIELDS = (
    "AGEDP1",
    "AGEDP2",
    "AGEDP3",
    "AGERANGE",
    "EARNSPLIT",
    "GENDER",
)
#: Typed column name to its explicit little-endian numpy dtype. Bodies are
#: written with these exact dtypes, so the payload is platform-independent.
_TYPED_DTYPES = {
    "RECID": "<i8",
    "FLPDYR": "<i2",
    "FLPDMO": "<i1",
    "MARS": "<i1",
    "DSI": "<i1",
    "S006": "<i8",
    "AGEDP1": "<i1",
    "AGEDP2": "<i1",
    "AGEDP3": "<i1",
    "AGERANGE": "<i1",
    "EARNSPLIT": "<i1",
    "GENDER": "<i1",
    "disclosure_aggregate": "<u1",
    "demographic_status": "<i1",
}
#: The definition names dtypes logically; the envelope writes numpy typestrings.
#: This is the only bridge between the two, and a test walks every declared
#: field through it, so the definition can never drift from what is stored.
_DECLARED_DTYPE = MappingProxyType(
    {"int64": "<i8", "int16": "<i2", "int8": "<i1", "uint8": "<u1"}
)
_TYPED_COLUMNS = tuple(_TYPED_DTYPES)
_LEXICAL_COLUMNS = _MAIN_FIELDS + _DEMOGRAPHIC_FIELDS
_ABSENT_SENTINEL = -1
#: The widest lexical allocation any field may declare, matching the closed
#: definition's own ``0 < lexical_width <= 64`` bound.
_LEXICAL_WIDTH_MAX = 64

_TYPED_KIND = "typed"
_LEXICAL_KIND = "lexical_fixed_ascii_nul_padded"

#: The envelope header is a closed schema. Every key set below is exact: an
#: unknown key, a missing key or a renamed key is a refusal, not a default.
_ENVELOPE_HEADER_KEYS = frozenset(
    {
        "schema_version",
        "type",
        "rows",
        "definition_sha256",
        "definition_route",
        "sources",
        "header_length_endianness",
        "body_endianness",
        "columns",
        "facts",
    }
)
_ENVELOPE_TYPED_COLUMN_KEYS = frozenset({"name", "kind", "dtype", "bytes", "sha256"})
_ENVELOPE_LEXICAL_COLUMN_KEYS = frozenset({"name", "kind", "width", "bytes", "sha256"})
_ENVELOPE_SOURCE_KEYS = frozenset(
    {"sha256", "git_blob_sha1", "bytes", "header_record_canonical_sha256"}
)
_ENVELOPE_FACT_KEYS = frozenset(
    {
        "rows",
        "main_records",
        "demographic_records",
        "matched_keys",
        "main_unmatched_records",
        "demographic_orphan_records",
        "aggregate_records",
        "aggregate_demographic_records",
        "join_status",
        "join_method",
        "code_frequencies",
        "period_semantics",
        "weight_units",
        "amounts_decoded",
        "untyped_main_columns",
    }
)
_INT64_MAX = 2**63 - 1
_DIGITS = frozenset(b"0123456789")


class PufRawSourceRefusalError(ValueError):
    """A refusal. The reason is a code and aggregates, never a source token."""

    def __init__(self, reason: str, *detail: object) -> None:
        super().__init__(" ".join((reason, *(str(part) for part in detail))).strip())
        self.reason = reason


def _refuse(reason: str, *detail: object) -> PufRawSourceRefusalError:
    return PufRawSourceRefusalError(reason, *detail)


# --------------------------------------------------------------------------
# The closed definition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourcePin:
    """The closed content identity of one delivered PUF file."""

    name: str
    codec: str
    source_name: str
    bytes: int
    sha256: str
    git_blob_sha1: str
    header_record_canonical_sha256: str
    delivered_header: tuple[str, ...]
    data_records: int


@dataclass(frozen=True)
class PufRawSourceDefinition:
    """A closed source declaration plus the route that produced it."""

    route: str
    document: Mapping[str, object]
    canonical: bytes
    sha256: str
    main: SourcePin
    demographic: SourcePin

    @property
    def params_text(self) -> str:
        """The canonical JSON the node carries verbatim as a parameter."""

        return self.canonical.decode("ascii")


def _hex64(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _refuse("DEFINITION_DIGEST", label)
    return value


def _hex40(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _refuse("DEFINITION_DIGEST", label)
    return value


def _pin(document: Mapping[str, object], name: str) -> SourcePin:
    entry = document["sources"][name]
    header = entry["delivered_header"]
    if not isinstance(header, list) or not header:
        raise _refuse("DEFINITION_HEADER", name)
    if len(header) != entry["delivered_header_width"]:
        raise _refuse("DEFINITION_HEADER_WIDTH", name)
    if len(set(header)) != len(header):
        raise _refuse("DEFINITION_HEADER_DUPLICATE", name)
    if _canonical_header_digest(header) != entry["header_record_canonical_sha256"]:
        raise _refuse("DEFINITION_HEADER_DIGEST", name)
    size = entry["bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise _refuse("DEFINITION_SOURCE_BYTES", name)
    if size > PUF_RAW_SOURCE_MAX_BYTES:
        raise _refuse("DEFINITION_SOURCE_OVER_CAP", name)
    return SourcePin(
        name=name,
        codec=entry["codec"],
        source_name=entry["source_name"],
        bytes=size,
        sha256=_hex64(entry["sha256"], f"{name}.sha256"),
        git_blob_sha1=_hex40(entry["git_blob_sha1"], f"{name}.git_blob_sha1"),
        header_record_canonical_sha256=_hex64(
            entry["header_record_canonical_sha256"], f"{name}.header"
        ),
        delivered_header=tuple(header),
        data_records=entry["data_records"],
    )


def _frozen(value: object) -> object:
    """A deeply immutable view: mappings become proxies, sequences tuples.

    ``MappingProxyType`` only freezes the top level. ``definition.document``
    is reached through ordinary read-only attribute access, so handing out a
    mutable nested container let a caller change a declared lexical width or
    a CSV limit — changing what parses — while ``canonical``, ``sha256`` and
    therefore ``params_text`` and the node key all stayed identical. Freezing
    the whole tree closes that without any caller sandboxing: the document is
    simply a value, as its canonical bytes always were.
    """

    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def _canonical_header_digest(header: Sequence[str]) -> str:
    """The audit's header identity: sha256 of the canonical JSON header list."""

    payload = json.dumps(
        list(header),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _definition_from_document(
    document: Mapping[str, object], *, route: str
) -> PufRawSourceDefinition:
    if document.get("schema") != "microcosm.us.puf_2015_raw_source_definition":
        raise _refuse("DEFINITION_SCHEMA")
    if document.get("schema_version") != 1:
        raise _refuse("DEFINITION_SCHEMA_VERSION")
    if document.get("route") != route:
        raise _refuse("DEFINITION_ROUTE")
    fields = document["fields"]
    declared = tuple(field["name"] for field in fields)
    if declared != _MAIN_FIELDS + _DEMOGRAPHIC_FIELDS:
        raise _refuse("DEFINITION_FIELDS")
    for field in fields:
        width = field["lexical_width"]
        if not isinstance(width, int) or isinstance(width, bool) or not 0 < width <= 64:
            raise _refuse("DEFINITION_LEXICAL_WIDTH", field["name"])
        if field["grammar"] != "ascii_decimal_unsigned":
            raise _refuse("DEFINITION_GRAMMAR", field["name"])
    if tuple(document["aggregates"]["recids"]) != PUF_AGGREGATE_RECIDS:
        raise _refuse("DEFINITION_AGGREGATE_RECIDS")
    canonical = canonical_json(dict(document))
    # The document is re-derived from the canonical bytes and then deeply
    # frozen, so the two can never disagree and neither can be edited.
    return PufRawSourceDefinition(
        route=route,
        document=_frozen(json.loads(canonical.decode("ascii"))),
        canonical=canonical,
        sha256=hashlib.sha256(canonical).hexdigest(),
        main=_pin(document, "main"),
        demographic=_pin(document, "demographic"),
    )


def _thawed(value: object) -> object:
    """The plain-container form of a frozen document, for canonical rendering."""

    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thawed(item) for item in value]
    return value


def _packaged_bytes() -> bytes:
    return (
        resources.files(_DEFINITION_PACKAGE).joinpath(_DEFINITION_RESOURCE).read_bytes()
    )


_PACKAGED: PufRawSourceDefinition | None = None


def packaged_definition() -> PufRawSourceDefinition:
    """The one closed definition a production run may use."""

    global _PACKAGED
    if _PACKAGED is None:
        document = json.loads(_packaged_bytes().decode("ascii"))
        _PACKAGED = _definition_from_document(document, route="packaged")
    return _PACKAGED


def definition_document_json(
    definition: PufRawSourceDefinition,
) -> dict[str, object]:
    """A fresh, plain, mutable JSON copy of a definition's document.

    The definition's own document is deeply frozen. A caller that wants to
    build a *different* document from this one — a fixture, say — starts here
    and gets a copy it owns, rather than a handle on the original.
    """

    return json.loads(definition.canonical.decode("ascii"))


def fixture_definition(document: Mapping[str, object]) -> PufRawSourceDefinition:
    """Build a definition on the explicit non-production test route.

    A fixture definition must declare ``route="test_fixture"`` and the
    invented-fixture authority, and it may not restate either packaged source
    pin. There is no route where an omitted or optional expected hash admits
    a file: a fixture carries its own real pins over its own invented bytes.
    """

    definition = _definition_from_document(document, route="test_fixture")
    if document.get("authority") != _FIXTURE_AUTHORITY:
        raise _refuse("FIXTURE_AUTHORITY")
    packaged = packaged_definition()
    genuine = {packaged.main.sha256, packaged.demographic.sha256}
    if {definition.main.sha256, definition.demographic.sha256} & genuine:
        raise _refuse("FIXTURE_RESTATES_PACKAGED_PIN")
    return definition


# --------------------------------------------------------------------------
# The pinned raw-byte codec
# --------------------------------------------------------------------------


def _identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _git_blob_sha1(payload: bytes) -> str:
    header = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(header + payload).hexdigest()


def load_pinned_puf_bytes(
    path: Path, *, store: object | None = None, pin: SourcePin
) -> bytes:
    """Read one pinned PUF delivery file, or refuse.

    The codec authenticates the bytes itself: exact length, SHA-256 and git
    blob SHA-1, all three from the descriptor it opened. It never consults a
    capture receipt or run summary, so a forged receipt cannot admit changed
    bytes.

    ``O_NOFOLLOW`` protects the path *this function receives*. The graph
    executor resolves source paths strictly before codec lookup, so under the
    executor the codec sees an already-resolved path and a user's symlink is
    followed upstream; on a direct call a symlink is refused here.
    ``O_NONBLOCK`` means a FIFO cannot make the codec wait.
    """

    del store  # identity is content; no content store is consulted
    if not isinstance(pin, SourcePin):
        raise _refuse("CODEC_PIN_TYPE")
    if pin.bytes > PUF_RAW_SOURCE_MAX_BYTES:
        raise _refuse("CODEC_PIN_OVER_CAP", pin.name)
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise _refuse("CODEC_OPEN", pin.name) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _refuse("CODEC_NOT_REGULAR_FILE", pin.name)
        if before.st_size != pin.bytes:
            raise _refuse("CODEC_SIZE", pin.name)
        if _identity(source.lstat()) != _identity(before):
            raise _refuse("CODEC_PATH_IDENTITY", pin.name)
        digest = hashlib.sha256()
        blob = hashlib.sha1(b"blob " + str(pin.bytes).encode("ascii") + b"\0")
        chunks: list[bytes] = []
        total = 0
        limit = pin.bytes + 1
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise _refuse("CODEC_GREW", pin.name)
            digest.update(chunk)
            blob.update(chunk)
            chunks.append(chunk)
        if total != pin.bytes:
            raise _refuse("CODEC_LENGTH", pin.name)
        if _identity(os.fstat(descriptor)) != _identity(before):
            raise _refuse("CODEC_CHANGED_DURING_READ", pin.name)
        if digest.hexdigest() != pin.sha256:
            raise _refuse("CODEC_SHA256", pin.name)
        if blob.hexdigest() != pin.git_blob_sha1:
            raise _refuse("CODEC_GIT_BLOB_SHA1", pin.name)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def register_us_puf_raw_source_codecs(
    registry: SourceCodecRegistry, *, definition: PufRawSourceDefinition | None = None
) -> SourceCodecRegistry:
    """Bind the two pinned byte codecs into ``registry`` through ``register_bytes``."""

    resolved = packaged_definition() if definition is None else definition
    registry.register_bytes(
        resolved.main.codec, partial(load_pinned_puf_bytes, pin=resolved.main)
    )
    registry.register_bytes(
        resolved.demographic.codec,
        partial(load_pinned_puf_bytes, pin=resolved.demographic),
    )
    return registry


def puf_raw_source_codecs(
    definition: PufRawSourceDefinition | None = None,
) -> SourceCodecRegistry:
    """A private registry holding only this slice's two pinned byte codecs."""

    return register_us_puf_raw_source_codecs(
        SourceCodecRegistry(), definition=definition
    )


# --------------------------------------------------------------------------
# The structural decoder
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PufRawReturnStatus:
    """The decoded slice: typed arrays, lexical tokens and aggregate facts."""

    typed: Mapping[str, np.ndarray]
    lexical: Mapping[str, tuple[str, ...]]
    facts: Mapping[str, object]

    @property
    def rows(self) -> int:
        return int(self.facts["rows"])


def csv_acceptance_profile() -> dict[str, object]:
    """The effective CSV acceptance state this decoder actually parses under.

    ``csv.field_size_limit`` is process-global state that silently changes
    which deliveries parse: at a lowered limit a file that was admitted is
    refused, and at a raised limit a file that was refused is admitted. It is
    therefore part of the producer's identity, exactly as
    :mod:`asec_income_observations` and :mod:`acs_housing_universe_source`
    already bind it. This function only *reads* it — nothing in this module
    ever sets it, so no other stage's acceptance is disturbed.

    The remaining entries are the reader semantics :func:`_reader` fixes
    itself; the delimiter, quote character and doublequote flag come from the
    closed definition and are checked against it at decode time.
    """

    return {
        "engine": "stdlib_csv_reader",
        "encoding": "utf-8-strict",
        "newline": "",
        "quoting": int(csv.QUOTE_MINIMAL),
        "strict": True,
        "csv.field_size_limit": csv.field_size_limit(),
    }


def _check_csv_profile(profile: Mapping[str, object]) -> None:
    """Refuse when the effective acceptance state is not the pinned one.

    The packaged definition states ``field_size_limit``. Before this check it
    was a decorative fact: the decoder relied on the process default happening
    to match. Now the pinned value is the value the parse actually ran under,
    or the decode refuses — a lowered limit is a refusal, never a silent
    reparse, and a raised limit is a refusal, never a silent widening.
    """

    declared = profile["field_size_limit"]
    if not isinstance(declared, int) or isinstance(declared, bool) or declared <= 0:
        raise _refuse("DEFINITION_CSV_FIELD_SIZE_LIMIT")
    if csv.field_size_limit() != declared:
        raise _refuse("CSV_FIELD_SIZE_LIMIT")
    if profile["quoting"] != "QUOTE_MINIMAL" or profile["strict"] is not True:
        raise _refuse("DEFINITION_CSV_PROFILE")
    if profile["encoding"] != "utf-8-strict" or profile["newline"] != "":
        raise _refuse("DEFINITION_CSV_PROFILE")


def _reader(payload: bytes, profile: Mapping[str, object]):
    if not isinstance(payload, bytes):
        raise _refuse("DECODE_PAYLOAD_TYPE")
    stream = io.TextIOWrapper(
        io.BytesIO(payload), encoding="utf-8", errors="strict", newline=""
    )
    return csv.reader(
        stream,
        delimiter=profile["delimiter"],
        quotechar=profile["quotechar"],
        doublequote=bool(profile["doublequote"]),
        quoting=csv.QUOTE_MINIMAL,
        strict=True,
    )


def _token(value: str, *, width: int, label: str) -> str:
    """An unsigned ASCII decimal token within its declared allocation."""

    if not isinstance(value, str):
        raise _refuse("TOKEN_TYPE", label)
    raw = value.encode("ascii", errors="strict") if value.isascii() else None
    if raw is None:
        raise _refuse("TOKEN_NON_ASCII", label)
    if b"\x00" in raw:
        raise _refuse("TOKEN_EMBEDDED_NUL", label)
    if not raw:
        raise _refuse("TOKEN_EMPTY", label)
    if len(raw) > width:
        raise _refuse("TOKEN_OVER_WIDTH", label)
    if any(byte not in _DIGITS for byte in raw):
        raise _refuse("TOKEN_GRAMMAR", label)
    return value


def _int64(token: str, label: str) -> int:
    value = int(token)
    if value > _INT64_MAX:
        raise _refuse("TOKEN_INT64_OVERFLOW", label)
    return value


def _widths(definition: PufRawSourceDefinition) -> dict[str, int]:
    return {
        field["name"]: field["lexical_width"] for field in definition.document["fields"]
    }


def _domains(definition: PufRawSourceDefinition) -> dict[str, frozenset[int]]:
    domains: dict[str, frozenset[int]] = {}
    for field in definition.document["fields"]:
        values = field.get("domain")
        if values is not None:
            domains[field["name"]] = frozenset(int(value) for value in values)
    return domains


def _agerange_domains(
    definition: PufRawSourceDefinition,
) -> dict[int, frozenset[int]]:
    for field in definition.document["fields"]:
        if field["name"] == "AGERANGE":
            cases = field["conditional_domain"]["cases"]
            return {
                int(key): frozenset(int(value) for value in values)
                for key, values in cases.items()
            }
    raise _refuse("DEFINITION_AGERANGE")


def _check_header(
    reader, pin: SourcePin, *, profile: Mapping[str, object]
) -> list[str]:
    try:
        header = next(reader)
    except StopIteration as error:
        raise _refuse("HEADER_MISSING", pin.name) from error
    except (csv.Error, UnicodeError) as error:
        raise _refuse("HEADER_UNREADABLE", pin.name) from error
    if len(header) > profile["header_field_cap"]:
        raise _refuse("HEADER_WIDTH_LIMIT", pin.name)
    if tuple(header) != pin.delivered_header:
        raise _refuse("HEADER_MISMATCH", pin.name)
    if _canonical_header_digest(header) != pin.header_record_canonical_sha256:
        raise _refuse("HEADER_DIGEST", pin.name)
    return header


def _records(reader, pin: SourcePin, *, profile: Mapping[str, object]):
    width = len(pin.delivered_header)
    cap = profile["logical_record_character_cap"]
    limit = profile["record_cap_per_file"]
    count = 0
    while True:
        try:
            record = next(reader)
        except StopIteration:
            return
        except (csv.Error, UnicodeError) as error:
            raise _refuse("RECORD_UNREADABLE", pin.name) from error
        count += 1
        if count > limit:
            raise _refuse("RECORD_CAP", pin.name)
        if len(record) != width:
            raise _refuse("RECORD_WIDTH", pin.name)
        if sum(len(cell) for cell in record) > cap:
            raise _refuse("RECORD_CHARACTER_CAP", pin.name)
        yield record


def decode_puf_raw_source(
    main_bytes: bytes,
    demographic_bytes: bytes,
    definition: PufRawSourceDefinition,
) -> PufRawReturnStatus:
    """Decode the two delivery payloads into the typed status slice.

    This is structure only. It checks headers, widths, token grammar, closed
    domains and the exact lexical join against the definition it is handed,
    and it authorizes nothing about where the bytes came from.
    """

    if not isinstance(definition, PufRawSourceDefinition):
        raise _refuse("DECODE_DEFINITION_TYPE")
    profile = definition.document["csv_profile"]
    _check_csv_profile(profile)
    widths = _widths(definition)
    domains = _domains(definition)
    agerange = _agerange_domains(definition)
    aggregates = frozenset(PUF_AGGREGATE_RECIDS)

    demographic = _decode_demographic(
        demographic_bytes, definition, profile=profile, widths=widths
    )
    main = _decode_main(
        main_bytes,
        definition,
        profile=profile,
        widths=widths,
        domains=domains,
        aggregates=aggregates,
    )

    rows = len(main["RECID_lexical"])
    orphans = set(demographic["by_key"]) - set(main["RECID_lexical"])
    if orphans:
        raise _refuse("DEMOGRAPHIC_ORPHAN_KEYS", len(orphans))

    typed: dict[str, np.ndarray] = {
        name: np.empty(rows, dtype=_TYPED_DTYPES[name]) for name in _TYPED_COLUMNS
    }
    for name in _MAIN_FIELDS:
        typed[name][:] = np.asarray(main[name], dtype=_TYPED_DTYPES[name])
    typed["disclosure_aggregate"][:] = np.asarray(
        main["disclosure_aggregate"], dtype=_TYPED_DTYPES["disclosure_aggregate"]
    )

    lexical: dict[str, list[str]] = {
        name: list(
            main["RECID_lexical"] if name == "RECID" else main[f"{name}_lexical"]
        )
        for name in _MAIN_FIELDS
    }
    for name in _DEMOGRAPHIC_FIELDS:
        lexical[name] = [""] * rows

    status = np.zeros(rows, dtype=_TYPED_DTYPES["demographic_status"])
    for name in _DEMOGRAPHIC_FIELDS:
        typed[name][:] = _ABSENT_SENTINEL

    aggregate_demographic_rows = 0
    for index, key in enumerate(main["RECID_lexical"]):
        record = demographic["by_key"].get(key)
        if record is None:
            continue
        status[index] = 1
        dsi = int(typed["DSI"][index])
        allowed = agerange.get(dsi)
        if allowed is None:
            raise _refuse("AGERANGE_CONDITION_UNKNOWN")
        for name in _DEMOGRAPHIC_FIELDS:
            token = record[name]
            value = _int64(token, name)
            permitted = allowed if name == "AGERANGE" else domains[name]
            if value not in permitted:
                raise _refuse("DEMOGRAPHIC_DOMAIN", name)
            typed[name][index] = value
            lexical[name][index] = token
        if bool(typed["disclosure_aggregate"][index]):
            aggregate_demographic_rows += 1
    typed["demographic_status"][:] = status

    del aggregate_demographic_rows  # re-derived below from the arrays themselves
    if demographic["records"] != int(status.sum()):
        raise _refuse("DEMOGRAPHIC_RECORD_COUNT")
    facts = _facts_from_arrays(
        typed,
        join_method=_join_method(definition),
        untyped_main_columns=_untyped_main_columns(definition),
    )
    _check_declared_expectations(facts, definition)
    return PufRawReturnStatus(
        typed=MappingProxyType({name: array for name, array in typed.items()}),
        lexical=MappingProxyType(
            {name: tuple(values) for name, values in lexical.items()}
        ),
        facts=MappingProxyType(facts),
    )


def _check_declared_expectations(
    facts: Mapping[str, object], definition: PufRawSourceDefinition
) -> None:
    """Cross-check the derived counts against the definition's declared ones.

    The packaged definition states how many records each delivery holds, how
    many keys should match, how many main records should stay unmatched and
    how many aggregate records to expect. Nothing compared them, so they were
    documentation. The byte pins make them implied, but implied is not
    checked: a delivery that hashed correctly and parsed to a different shape
    would have gone unnoticed.

    This runs on the packaged route only. A fixture definition carries its own
    invented counts over its own invented bytes, and inventing a fixture that
    restates the genuine expectations would be the wrong kind of fidelity.
    """

    if definition.route != "packaged":
        return
    join = definition.document["join"]
    for label, derived, declared in (
        ("rows", facts["rows"], definition.main.data_records),
        (
            "demographic_records",
            facts["demographic_records"],
            definition.demographic.data_records,
        ),
        ("matched_keys", facts["matched_keys"], join["expected_matched_keys"]),
        (
            "main_unmatched_records",
            facts["main_unmatched_records"],
            join["expected_main_unmatched_records"],
        ),
        (
            "aggregate_records",
            facts["aggregate_records"],
            definition.document["aggregates"]["expected_records"],
        ),
    ):
        if derived != declared:
            raise _refuse("DECLARED_EXPECTATION", label)


def _join_method(definition: PufRawSourceDefinition) -> str:
    method = definition.document["join"]["method"]
    if not isinstance(method, str) or not method:
        raise _refuse("DEFINITION_JOIN_METHOD")
    return method


def _untyped_main_columns(definition: PufRawSourceDefinition) -> int:
    return len(definition.main.delivered_header) - len(_MAIN_FIELDS)


def _facts_from_arrays(
    typed: Mapping[str, np.ndarray],
    *,
    join_method: str,
    untyped_main_columns: int,
) -> dict[str, object]:
    """Derive every reported fact from the typed arrays themselves.

    The producer builds its facts here and the envelope readback rebuilds them
    here, so a header can never report a count the arrays do not carry. Only
    ``join_method`` and ``untyped_main_columns`` come from outside: both are
    properties of the definition, not of the decoded rows.
    """

    rows = int(typed["RECID"].shape[0])
    status = typed["demographic_status"]
    aggregate = typed["disclosure_aggregate"]
    matched = int(np.count_nonzero(status == 1))
    return {
        "rows": rows,
        "main_records": rows,
        "demographic_records": matched,
        "matched_keys": matched,
        "main_unmatched_records": rows - matched,
        "demographic_orphan_records": 0,
        "aggregate_records": int(np.count_nonzero(aggregate == 1)),
        "aggregate_demographic_records": int(
            np.count_nonzero((aggregate == 1) & (status == 1))
        ),
        "join_status": "one_to_one_subset_compatible",
        "join_method": join_method,
        "code_frequencies": _code_frequencies(typed),
        "period_semantics": "unresolved_raw_FLPDYR",
        "weight_units": "S006_hundredths",
        "amounts_decoded": False,
        "untyped_main_columns": untyped_main_columns,
    }


def _code_frequencies(typed: Mapping[str, np.ndarray]) -> dict[str, dict[str, int]]:
    counted = ("FLPDYR", "FLPDMO", "MARS", "DSI", "demographic_status")
    frequencies: dict[str, dict[str, int]] = {}
    for name in (*counted, *_DEMOGRAPHIC_FIELDS):
        values, counts = np.unique(typed[name], return_counts=True)
        frequencies[name] = {
            str(int(value)): int(count)
            for value, count in zip(values.tolist(), counts.tolist(), strict=True)
        }
    return frequencies


def _decode_main(
    payload: bytes,
    definition: PufRawSourceDefinition,
    *,
    profile,
    widths,
    domains,
    aggregates,
) -> dict[str, object]:
    pin = definition.main
    reader = _reader(payload, profile)
    header = _check_header(reader, pin, profile=profile)
    index = {name: header.index(name) for name in _MAIN_FIELDS}
    columns: dict[str, list] = {name: [] for name in _MAIN_FIELDS}
    lexical: dict[str, list[str]] = {f"{name}_lexical": [] for name in _MAIN_FIELDS}
    keys: list[str] = []
    numeric_keys: set[int] = set()
    seen: set[str] = set()
    aggregate_flags: list[int] = []
    for record in _records(reader, pin, profile=profile):
        values: dict[str, int] = {}
        for name in _MAIN_FIELDS:
            token = _token(record[index[name]], width=widths[name], label=name)
            lexical[f"{name}_lexical"].append(token)
            values[name] = _int64(token, name)
        key = lexical["RECID_lexical"][-1]
        if key in seen:
            raise _refuse("MAIN_DUPLICATE_LEXICAL_KEY")
        seen.add(key)
        recid = values["RECID"]
        if recid in numeric_keys:
            raise _refuse("MAIN_NUMERIC_KEY_COLLISION")
        numeric_keys.add(recid)
        keys.append(key)
        for name in ("FLPDYR", "FLPDMO", "MARS", "DSI"):
            if values[name] not in domains[name]:
                raise _refuse("MAIN_DOMAIN", name)
        aggregate = recid in aggregates
        if aggregate != (values["MARS"] == 0):
            raise _refuse("AGGREGATE_MARS_DISAGREEMENT")
        aggregate_flags.append(1 if aggregate else 0)
        for name in _MAIN_FIELDS:
            columns[name].append(values[name])
    if not keys:
        raise _refuse("MAIN_NO_RECORDS")
    result: dict[str, object] = dict(columns)
    result.update(lexical)
    result["disclosure_aggregate"] = aggregate_flags
    return result


def _decode_demographic(
    payload: bytes, definition: PufRawSourceDefinition, *, profile, widths
) -> dict[str, object]:
    pin = definition.demographic
    reader = _reader(payload, profile)
    header = _check_header(reader, pin, profile=profile)
    index = {name: header.index(name) for name in ("RECID", *_DEMOGRAPHIC_FIELDS)}
    by_key: dict[str, dict[str, str]] = {}
    numeric_keys: set[int] = set()
    records = 0
    for record in _records(reader, pin, profile=profile):
        key = _token(record[index["RECID"]], width=widths["RECID"], label="RECID")
        if key in by_key:
            raise _refuse("DEMOGRAPHIC_DUPLICATE_LEXICAL_KEY")
        recid = _int64(key, "RECID")
        if recid in numeric_keys:
            raise _refuse("DEMOGRAPHIC_NUMERIC_KEY_COLLISION")
        numeric_keys.add(recid)
        by_key[key] = {
            name: _token(record[index[name]], width=widths[name], label=name)
            for name in _DEMOGRAPHIC_FIELDS
        }
        records += 1
    return {"by_key": by_key, "records": records}


# --------------------------------------------------------------------------
# The typed envelope
# --------------------------------------------------------------------------


def _fixed_ascii(values: Sequence[str], width: int, label: str) -> bytes:
    body = bytearray(len(values) * width)
    for position, value in enumerate(values):
        raw = value.encode("ascii")
        if len(raw) > width:
            raise _refuse("LEXICAL_OVER_WIDTH", label)
        if b"\x00" in raw:
            raise _refuse("LEXICAL_EMBEDDED_NUL", label)
        start = position * width
        body[start : start + len(raw)] = raw
    return bytes(body)


def _read_fixed_ascii(
    body: bytes, rows: int, width: int, label: str
) -> tuple[str, ...]:
    values = []
    for position in range(rows):
        cell = body[position * width : (position + 1) * width]
        stripped = cell.rstrip(b"\x00")
        if b"\x00" in stripped:
            raise _refuse("LEXICAL_EMBEDDED_NUL", label)
        try:
            values.append(stripped.decode("ascii"))
        except UnicodeDecodeError as error:
            raise _refuse("LEXICAL_NON_ASCII", label) from error
    return tuple(values)


def _payload_bound(
    rows: int, definition: PufRawSourceDefinition
) -> tuple[int, dict[str, int]]:
    """The exact body size this artifact will allocate, and its per-column parts."""

    widths = _widths(definition)
    sizes = {
        name: rows * np.dtype(_TYPED_DTYPES[name]).itemsize for name in _TYPED_COLUMNS
    }
    sizes.update({f"{name}_lexical": rows * widths[name] for name in _LEXICAL_COLUMNS})
    return sum(sizes.values()), sizes


def encode_return_status(
    status: PufRawReturnStatus, definition: PufRawSourceDefinition
) -> bytes:
    """Encode the slice into the bounded, self-describing artifact envelope.

    Magic line, a 4-byte big-endian header length matching the ACS rent
    placement precedent, a canonical JSON header, then explicit little-endian
    integer bodies and fixed-width NUL-padded ASCII lexical bodies, in the
    declared order. Every body carries its own SHA-256 in the header.
    """

    rows = status.rows
    total, sizes = _payload_bound(rows, definition)
    if total > BODY_MAX_BYTES:
        raise _refuse("BODY_OVER_BOUND", total)
    widths = _widths(definition)
    bodies: list[bytes] = []
    columns: list[dict[str, object]] = []
    for name in _TYPED_COLUMNS:
        array = np.ascontiguousarray(status.typed[name], dtype=_TYPED_DTYPES[name])
        if array.shape != (rows,):
            raise _refuse("COLUMN_SHAPE", name)
        body = array.tobytes()
        if len(body) != sizes[name]:
            raise _refuse("COLUMN_SIZE", name)
        bodies.append(body)
        columns.append(
            {
                "name": name,
                "kind": "typed",
                "dtype": _TYPED_DTYPES[name],
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    for name in _LEXICAL_COLUMNS:
        body = _fixed_ascii(status.lexical[name], widths[name], name)
        if len(body) != sizes[f"{name}_lexical"]:
            raise _refuse("COLUMN_SIZE", name)
        bodies.append(body)
        columns.append(
            {
                "name": f"{name}_lexical",
                "kind": "lexical_fixed_ascii_nul_padded",
                "width": widths[name],
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    header = {
        "schema_version": 1,
        "type": [RETURN_STATUS_TYPE.name, RETURN_STATUS_TYPE.schema_version],
        "rows": rows,
        "definition_sha256": definition.sha256,
        "definition_route": definition.route,
        "sources": {
            "main": {
                "sha256": definition.main.sha256,
                "git_blob_sha1": definition.main.git_blob_sha1,
                "bytes": definition.main.bytes,
                "header_record_canonical_sha256": (
                    definition.main.header_record_canonical_sha256
                ),
            },
            "demographic": {
                "sha256": definition.demographic.sha256,
                "git_blob_sha1": definition.demographic.git_blob_sha1,
                "bytes": definition.demographic.bytes,
                "header_record_canonical_sha256": (
                    definition.demographic.header_record_canonical_sha256
                ),
            },
        },
        "header_length_endianness": "big",
        "body_endianness": "little",
        "columns": columns,
        "facts": dict(status.facts),
    }
    encoded = canonical_json(header)
    if len(encoded) > HEADER_MAX_BYTES:
        raise _refuse("HEADER_OVER_BOUND", len(encoded))
    return b"".join(
        (RETURN_STATUS_MAGIC, len(encoded).to_bytes(4, "big"), encoded, *bodies)
    )


def _exact_keys(
    value: object, keys: frozenset[str], label: str
) -> Mapping[str, object]:
    """A mapping whose key set is exactly ``keys``. Nothing extra, nothing missing."""

    if not isinstance(value, dict) or set(value) != set(keys):
        raise _refuse("ENVELOPE_SCHEMA", label)
    return value


def _envelope_int(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise _refuse("ENVELOPE_INTEGER", label)
    return value


def _resolve_envelope_definition(
    header: Mapping[str, object], definition: PufRawSourceDefinition | None
) -> PufRawSourceDefinition | None:
    """The definition this payload may be checked against, or ``None``.

    A caller may hand one in. Otherwise a payload that declares the *packaged*
    route is checked against this module's own packaged definition, because
    that one is closed and importable. A fixture-route payload has no
    reconstructible definition, so it is checked structurally only: the
    production and fixture boundaries stay exactly where they are.
    """

    route = header["definition_route"]
    if route not in ("packaged", "test_fixture"):
        raise _refuse("ENVELOPE_DEFINITION_ROUTE")
    digest = _hex64(header["definition_sha256"], "envelope.definition_sha256")
    if definition is not None:
        if not isinstance(definition, PufRawSourceDefinition):
            raise _refuse("DECODE_DEFINITION_TYPE")
        if definition.route != route or definition.sha256 != digest:
            raise _refuse("ENVELOPE_DEFINITION_MISMATCH")
        return definition
    if route == "packaged":
        packaged = packaged_definition()
        if packaged.sha256 != digest:
            raise _refuse("ENVELOPE_DEFINITION_NOT_PACKAGED")
        return packaged
    return None


def _check_envelope_sources(
    header: Mapping[str, object], definition: PufRawSourceDefinition | None
) -> None:
    sources = _exact_keys(
        header["sources"], frozenset({"main", "demographic"}), "sources"
    )
    for name in ("main", "demographic"):
        entry = _exact_keys(sources[name], _ENVELOPE_SOURCE_KEYS, f"sources.{name}")
        _hex64(entry["sha256"], f"sources.{name}.sha256")
        _hex40(entry["git_blob_sha1"], f"sources.{name}.git_blob_sha1")
        _hex64(entry["header_record_canonical_sha256"], f"sources.{name}.header")
        size = _envelope_int(entry["bytes"], f"sources.{name}.bytes", minimum=1)
        if size > PUF_RAW_SOURCE_MAX_BYTES:
            raise _refuse("ENVELOPE_SOURCE_OVER_CAP", name)
        if definition is None:
            continue
        pin = definition.main if name == "main" else definition.demographic
        if (
            entry["sha256"] != pin.sha256
            or entry["git_blob_sha1"] != pin.git_blob_sha1
            or size != pin.bytes
            or entry["header_record_canonical_sha256"]
            != pin.header_record_canonical_sha256
        ):
            raise _refuse("ENVELOPE_SOURCE_PIN", name)


def _envelope_columns(
    header: Mapping[str, object],
    rows: int,
    definition: PufRawSourceDefinition | None,
) -> tuple[list[Mapping[str, object]], dict[str, int]]:
    """Close the column schema: exact order, kind, dtype, width and size."""

    columns = header["columns"]
    if not isinstance(columns, list):
        raise _refuse("ENVELOPE_COLUMNS")
    declared = [
        *_TYPED_COLUMNS,
        *(f"{name}_lexical" for name in _LEXICAL_COLUMNS),
    ]
    if len(columns) != len(declared):
        raise _refuse("ENVELOPE_COLUMNS")
    widths = _widths(definition) if definition is not None else None
    resolved: list[Mapping[str, object]] = []
    sizes: dict[str, int] = {}
    total = 0
    for column, name in zip(columns, declared, strict=True):
        if not isinstance(column, dict) or column.get("name") != name:
            raise _refuse("ENVELOPE_COLUMNS")
        if name in _TYPED_DTYPES:
            entry = _exact_keys(column, _ENVELOPE_TYPED_COLUMN_KEYS, name)
            if entry["kind"] != _TYPED_KIND:
                raise _refuse("ENVELOPE_COLUMN_KIND", name)
            # Exact dtype string: this closes width, signedness and endianness
            # at once, so ``<f8`` can never stand in for ``<i8``.
            if entry["dtype"] != _TYPED_DTYPES[name]:
                raise _refuse("ENVELOPE_COLUMN_DTYPE", name)
            expected = rows * np.dtype(_TYPED_DTYPES[name]).itemsize
        else:
            entry = _exact_keys(column, _ENVELOPE_LEXICAL_COLUMN_KEYS, name)
            if entry["kind"] != _LEXICAL_KIND:
                raise _refuse("ENVELOPE_COLUMN_KIND", name)
            field = name.removesuffix("_lexical")
            width = _envelope_int(entry["width"], f"{name}.width", minimum=1)
            if width > _LEXICAL_WIDTH_MAX:
                raise _refuse("ENVELOPE_COLUMN_WIDTH", name)
            if widths is not None and width != widths[field]:
                raise _refuse("ENVELOPE_COLUMN_WIDTH", name)
            expected = rows * width
            sizes[field] = width
        _hex64(entry["sha256"], f"{name}.sha256")
        size = _envelope_int(entry["bytes"], f"{name}.bytes")
        if size != expected:
            raise _refuse("ENVELOPE_COLUMN_SIZE", name)
        total += size
        if total > BODY_MAX_BYTES:
            raise _refuse("ENVELOPE_BODY_OVER_BOUND", total)
        resolved.append(entry)
    return resolved, sizes


def _verify_readback(
    typed: Mapping[str, np.ndarray],
    lexical: Mapping[str, tuple[str, ...]],
    widths: Mapping[str, int],
    definition: PufRawSourceDefinition | None,
) -> None:
    """Re-prove the slice's own invariants from the arrays that were read back.

    Lexical and numeric spellings must agree, absence must stay absence rather
    than a zero code, the aggregate flag must follow from ``RECID`` and agree
    with ``MARS``, and the join keys must still be unique both ways.
    """

    rows = int(typed["RECID"].shape[0])
    status = typed["demographic_status"]
    aggregate = typed["disclosure_aggregate"]
    if not np.isin(status, (0, 1)).all():
        raise _refuse("READBACK_DEMOGRAPHIC_STATUS")
    if not np.isin(aggregate, (0, 1)).all():
        raise _refuse("READBACK_AGGREGATE_FLAG")
    if int(typed["S006"].min(initial=0)) < 0:
        raise _refuse("READBACK_S006_NEGATIVE")

    present = status == 1
    for name in _MAIN_FIELDS:
        tokens = lexical[name]
        if len(tokens) != rows:
            raise _refuse("READBACK_LEXICAL_ROWS", name)
        width = widths[name]
        values = typed[name]
        for position, token in enumerate(tokens):
            _token(token, width=width, label=name)
            if int(token) != int(values[position]):
                raise _refuse("READBACK_LEXICAL_DISAGREEMENT", name)
    for name in _DEMOGRAPHIC_FIELDS:
        tokens = lexical[name]
        if len(tokens) != rows:
            raise _refuse("READBACK_LEXICAL_ROWS", name)
        width = widths[name]
        values = typed[name]
        for position, token in enumerate(tokens):
            if not present[position]:
                # Absence is the sentinel and an empty spelling, never a zero.
                if token or int(values[position]) != _ABSENT_SENTINEL:
                    raise _refuse("READBACK_ABSENCE_NOT_EMPTY", name)
                continue
            _token(token, width=width, label=name)
            if int(token) != int(values[position]):
                raise _refuse("READBACK_LEXICAL_DISAGREEMENT", name)

    keys = lexical["RECID"]
    if len(set(keys)) != rows:
        raise _refuse("READBACK_DUPLICATE_LEXICAL_KEY")
    recids = typed["RECID"]
    if len(np.unique(recids)) != rows:
        raise _refuse("READBACK_NUMERIC_KEY_COLLISION")
    expected_aggregate = np.isin(recids, np.asarray(PUF_AGGREGATE_RECIDS, dtype="<i8"))
    if not np.array_equal(aggregate == 1, expected_aggregate):
        raise _refuse("READBACK_AGGREGATE_RECID")
    if not np.array_equal(expected_aggregate, typed["MARS"] == 0):
        raise _refuse("READBACK_AGGREGATE_MARS_DISAGREEMENT")

    if definition is None:
        return
    domains = _domains(definition)
    agerange = _agerange_domains(definition)
    for name in ("FLPDYR", "FLPDMO", "MARS", "DSI"):
        if not np.isin(typed[name], sorted(domains[name])).all():
            raise _refuse("READBACK_MAIN_DOMAIN", name)
    for name in _DEMOGRAPHIC_FIELDS:
        values = typed[name][present]
        if name == "AGERANGE":
            for condition, allowed in agerange.items():
                selected = values[typed["DSI"][present] == condition]
                if selected.size and not np.isin(selected, sorted(allowed)).all():
                    raise _refuse("READBACK_DEMOGRAPHIC_DOMAIN", name)
            if not np.isin(typed["DSI"][present], sorted(agerange)).all():
                raise _refuse("READBACK_AGERANGE_CONDITION_UNKNOWN")
            continue
        if values.size and not np.isin(values, sorted(domains[name])).all():
            raise _refuse("READBACK_DEMOGRAPHIC_DOMAIN", name)


def decode_return_status(
    payload: bytes, definition: PufRawSourceDefinition | None = None
) -> PufRawReturnStatus:
    """Decode an artifact payload and re-prove it, never trusting its header.

    The header is a closed schema: exact key sets, an exact dtype string for
    every typed column, an exact width and body size for every lexical column,
    and a total body bound. Every reported fact is then **recomputed** from the
    arrays actually read back and must agree exactly, so a payload cannot
    report a row count, a match count or a code frequency it does not carry.

    This stays structural. It is not source authority: it proves that a payload
    is internally consistent and, when a definition is available, that it
    matches that definition's widths, domains and source pins. It authorizes
    nothing about where the delivery bytes came from — the codec does that.
    """

    if not isinstance(payload, bytes) or not payload.startswith(RETURN_STATUS_MAGIC):
        raise _refuse("ENVELOPE_MAGIC")
    start = len(RETURN_STATUS_MAGIC)
    if len(payload) < start + 4:
        raise _refuse("ENVELOPE_TRUNCATED")
    length = int.from_bytes(payload[start : start + 4], "big")
    if not 0 < length <= HEADER_MAX_BYTES or len(payload) < start + 4 + length:
        raise _refuse("ENVELOPE_HEADER_LENGTH")
    encoded = payload[start + 4 : start + 4 + length]
    try:
        header = json.loads(encoded.decode("ascii"))
    except (UnicodeError, ValueError, RecursionError) as error:
        # A deeply nested or malformed header is a refusal, not a stray
        # ``RecursionError`` or ``UnicodeDecodeError`` out of the JSON reader.
        raise _refuse("ENVELOPE_HEADER_JSON") from error
    try:
        recanonical = canonical_json(header)
    except (TypeError, ValueError, RecursionError) as error:
        raise _refuse("ENVELOPE_HEADER_JSON") from error
    if recanonical != encoded:
        raise _refuse("ENVELOPE_HEADER_NOT_CANONICAL")
    _exact_keys(header, _ENVELOPE_HEADER_KEYS, "header")
    if header["schema_version"] != 1 or header["type"] != [
        RETURN_STATUS_TYPE.name,
        RETURN_STATUS_TYPE.schema_version,
    ]:
        raise _refuse("ENVELOPE_TYPE")
    if (
        header["header_length_endianness"] != "big"
        or header["body_endianness"] != "little"
    ):
        raise _refuse("ENVELOPE_ENDIANNESS")
    rows = _envelope_int(header["rows"], "rows", minimum=1)
    resolved = _resolve_envelope_definition(header, definition)
    _check_envelope_sources(header, resolved)
    columns, widths = _envelope_columns(header, rows, resolved)

    cursor = start + 4 + length
    typed: dict[str, np.ndarray] = {}
    lexical: dict[str, tuple[str, ...]] = {}
    for column in columns:
        name = column["name"]
        size = column["bytes"]
        body = payload[cursor : cursor + size]
        if len(body) != size:
            raise _refuse("ENVELOPE_BODY_TRUNCATED", name)
        if hashlib.sha256(body).hexdigest() != column["sha256"]:
            raise _refuse("ENVELOPE_BODY_DIGEST", name)
        cursor += size
        if column["kind"] == _TYPED_KIND:
            typed[name] = np.frombuffer(body, dtype=np.dtype(column["dtype"]))
        else:
            field = name.removesuffix("_lexical")
            lexical[field] = _read_fixed_ascii(body, rows, column["width"], field)
    if cursor != len(payload):
        raise _refuse("ENVELOPE_TRAILING_BYTES")

    _verify_readback(typed, lexical, widths, resolved)
    facts = header["facts"]
    _exact_keys(facts, _ENVELOPE_FACT_KEYS, "facts")
    if resolved is not None:
        join_method = _join_method(resolved)
        untyped_main_columns = _untyped_main_columns(resolved)
    else:
        join_method = facts["join_method"]
        if not isinstance(join_method, str) or not join_method:
            raise _refuse("ENVELOPE_FACTS", "join_method")
        untyped_main_columns = _envelope_int(
            facts["untyped_main_columns"], "facts.untyped_main_columns"
        )
    recomputed = _facts_from_arrays(
        typed,
        join_method=join_method,
        untyped_main_columns=untyped_main_columns,
    )
    if canonical_json(recomputed) != canonical_json(dict(facts)):
        raise _refuse("ENVELOPE_FACTS_INCONSISTENT")
    if recomputed["rows"] != rows:
        raise _refuse("ENVELOPE_ROWS")
    if resolved is not None:
        _check_declared_expectations(recomputed, resolved)
    return PufRawReturnStatus(
        typed=MappingProxyType(typed),
        lexical=MappingProxyType(lexical),
        facts=MappingProxyType(recomputed),
    )


# --------------------------------------------------------------------------
# The graph producer, consumer and registry
# --------------------------------------------------------------------------

_IMPLEMENTATION_MODULES = ("microcosm.build.us_runtime.puf_raw_source",)
_IMPLEMENTATION_IDENTITY_MAGIC = b"us.puf.raw_source/implementation/2\n"


class USPufRawSourceKernel(KernelBase):
    """Read the two pinned deliveries and produce one typed status artifact."""

    ref = "us.puf.raw_source.return_status@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=("numpy",),
    )

    def __init__(
        self,
        *,
        definition: PufRawSourceDefinition | None = None,
        source_codecs: SourceCodecRegistry | None = None,
    ) -> None:
        self.definition = packaged_definition() if definition is None else definition
        if self.definition.route not in ("packaged", "test_fixture"):
            raise _refuse("KERNEL_DEFINITION_ROUTE")
        if self.definition.route == "packaged":
            # The production wrapper, not the bare decoder, is the authority:
            # the definition must be the packaged file's own canonical bytes.
            # Compare the bytes, not only their digest, and re-prove that the
            # document this definition carries is still those exact bytes.
            packaged = packaged_definition()
            if (
                self.definition.canonical != packaged.canonical
                or self.definition.sha256 != packaged.sha256
                or canonical_json(_thawed(self.definition.document))
                != self.definition.canonical
            ):
                raise _refuse("KERNEL_DEFINITION_NOT_PACKAGED")
        self.source_codecs = (
            puf_raw_source_codecs(self.definition)
            if source_codecs is None
            else source_codecs
        )

    def implementation_hash(self) -> str:
        """Module source, dependency versions **and** the effective CSV profile.

        The acceptance profile is process state the parse runs under, so it
        belongs in the producer's identity: change it and this node's key
        moves, which is a recompute or a ``require`` miss. It can no longer
        reuse an admission made under a different acceptance.
        """

        from importlib import import_module

        base = source_hash(
            *(import_module(name) for name in _IMPLEMENTATION_MODULES),
            dependencies=self.capabilities.dependencies,
        )
        digest = hashlib.sha256(_IMPLEMENTATION_IDENTITY_MAGIC)
        digest.update(base.encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(csv_acceptance_profile()))
        return digest.hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        definition = self.definition
        if (
            context.node.kernel != self.ref
            or context.node.structural is not StructuralDelta.NONE
            or context.node.sources
            != (definition.main.source_name, definition.demographic.source_name)
            or context.node.inputs
            or context.node.outputs
            or context.node.artifact_inputs
            or context.node.artifact_outputs
            != (ArtifactOutput("return_status", RETURN_STATUS_TYPE),)
            or set(context.params) != {"definition"}
        ):
            raise _refuse("NODE_DECLARATION")
        if context.params["definition"] != definition.params_text:
            raise _refuse("NODE_DEFINITION_PARAM")
        main_bytes = load_source_bytes(
            definition.main.codec,
            context.sources[definition.main.source_name],
            registry=self.source_codecs,
        )
        demographic_bytes = load_source_bytes(
            definition.demographic.codec,
            context.sources[definition.demographic.source_name],
            registry=self.source_codecs,
        )
        status = decode_puf_raw_source(main_bytes, demographic_bytes, definition)
        payload = encode_return_status(status, definition)
        return KernelResult(
            artifacts={"return_status": payload},
            receipt={
                "puf_raw_return_status": {
                    "definition_sha256": definition.sha256,
                    "definition_route": definition.route,
                    "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                    "artifact_bytes": len(payload),
                    "main_sha256": definition.main.sha256,
                    "demographic_sha256": definition.demographic.sha256,
                    "csv_acceptance_profile": csv_acceptance_profile(),
                    **{
                        key: value
                        for key, value in status.facts.items()
                        if key != "code_frequencies"
                    },
                }
            },
        )


class USPufRawReturnStatusSummaryKernel(KernelBase):
    """A downstream consumer: decode the typed artifact and report counts."""

    ref = "us.puf.raw_source.return_status_summary@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=("numpy",),
    )

    def implementation_hash(self) -> str:
        from importlib import import_module

        return source_hash(
            *(import_module(name) for name in _IMPLEMENTATION_MODULES),
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        if (
            context.node.kernel != self.ref
            or context.node.structural is not StructuralDelta.NONE
            or context.node.sources
            or context.node.inputs
            or context.node.outputs
            or context.node.artifact_outputs
            or len(context.node.artifact_inputs) != 1
            or context.node.artifact_inputs[0].type != RETURN_STATUS_TYPE
            or context.params
        ):
            raise _refuse("SUMMARY_NODE_DECLARATION")
        alias = context.node.artifact_inputs[0].name
        value = context.artifacts[alias]
        if value.type != RETURN_STATUS_TYPE:
            raise _refuse("SUMMARY_ARTIFACT_TYPE")
        status = decode_return_status(value.payload)
        return KernelResult(
            receipt={
                "puf_raw_return_status_summary": {
                    "producer_key": value.producer_key,
                    "rows": status.rows,
                    "matched_keys": status.facts["matched_keys"],
                    "main_unmatched_records": status.facts["main_unmatched_records"],
                    "aggregate_records": status.facts["aggregate_records"],
                    "period_semantics": status.facts["period_semantics"],
                    "weight_units": status.facts["weight_units"],
                    "amounts_decoded": status.facts["amounts_decoded"],
                }
            }
        )


def us_puf_raw_source_refs(
    definition: PufRawSourceDefinition | None = None,
) -> tuple[SourceRef, ...]:
    """The two ``SourceRef`` declarations this slice's node reads."""

    resolved = packaged_definition() if definition is None else definition
    return (
        SourceRef(
            resolved.main.source_name,
            resolved.main.codec,
            description="2015 PUF main delivery, pinned by content.",
        ),
        SourceRef(
            resolved.demographic.source_name,
            resolved.demographic.codec,
            description="2015 PUF demographic delivery, pinned by content.",
        ),
    )


def us_puf_raw_source_node(
    *,
    population: str,
    definition: PufRawSourceDefinition | None = None,
    stage: str = PUF_RAW_SOURCE_STAGE,
) -> Node:
    """Declare the producer node inside an explicitly named host population."""

    if not isinstance(population, str) or not population:
        raise _refuse("NODE_POPULATION_REQUIRED")
    resolved = packaged_definition() if definition is None else definition
    return Node(
        f"{stage}.return_status",
        USPufRawSourceKernel.ref,
        population=population,
        sources=(resolved.main.source_name, resolved.demographic.source_name),
        params={"definition": resolved.params_text},
        artifact_outputs=(ArtifactOutput("return_status", RETURN_STATUS_TYPE),),
        description=(
            "Type the 2015 PUF return status slice from the pinned delivery "
            "bytes. Owns no population columns."
        ),
    )


def us_puf_raw_return_status_summary_node(
    *,
    population: str,
    producer: str = PUF_RAW_SOURCE_NODE,
    stage: str = PUF_RAW_SOURCE_STAGE,
) -> Node:
    """Declare an ordinary consumer of the typed artifact."""

    if not isinstance(population, str) or not population:
        raise _refuse("NODE_POPULATION_REQUIRED")
    return Node(
        f"{stage}.return_status_summary",
        USPufRawReturnStatusSummaryKernel.ref,
        population=population,
        artifact_inputs=(
            ArtifactInput(
                "return_status", producer, "return_status", RETURN_STATUS_TYPE
            ),
        ),
        description="Read the typed PUF status artifact and report its counts.",
    )


def register_us_puf_raw_source_kernels(
    registry: KernelRegistry,
    *,
    definition: PufRawSourceDefinition | None = None,
    source_codecs: SourceCodecRegistry | None = None,
) -> KernelRegistry:
    """Register the producer and consumer kernels into ``registry``."""

    registry.register(
        USPufRawSourceKernel(definition=definition, source_codecs=source_codecs)
    )
    registry.register(USPufRawReturnStatusSummaryKernel())
    return registry
