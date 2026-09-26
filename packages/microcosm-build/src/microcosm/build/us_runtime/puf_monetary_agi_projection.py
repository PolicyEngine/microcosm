"""The thirteen-field 2015 PUF monetary source projection: the twelve accepted
columns plus the publisher's reported ``E00100``.

The reviewed twelve-field increment
(:mod:`microcosm.build.us_runtime.puf_monetary_source`) decodes wages, the two
interest columns, the two dividend columns, the Schedule C and Schedule F net
results, the two pension columns and the three Schedule D quantities. The PUF
growth contract additionally requires ``E00100`` as a declared money field
(:class:`~microcosm.build.us_runtime.puf_growth.GrowthRecipe`), and the
provenance AGI column that contract writes is a *copy of its own input*, not an
independent source. So a growth run over this lineage needs a projection that
carries reported AGI, and this module is that projection.

Why a separate module, and what it preserves
--------------------------------------------
The twelve-field kernel hashes the bytes of ``puf_monetary_source.py`` and
``puf_raw_source.py`` into its implementation hash, and the executor folds that
hash into the artifact store key. Editing either module would move the accepted
twelve-field artifact's key. A repository test also pins the twelve-field
generator's own sha256 to the packaged document. **Nothing here edits any of
them.** This module imports the reviewed decoder, the reviewed envelope
helpers and the reviewed fact derivation, so the thirteenth column is decoded
by exactly the code that decodes the other twelve rather than by a second
implementation of the same contract. The private names it imports are imported
deliberately, and each one is a piece of the reviewed contract this edition
must not restate.

What is the same
----------------
Every property of the twelve-field contract holds here unchanged: whole-dollar
signed integer tokens on individual returns, the separately bounded
disclosure-aggregate lexical grammar, the minimum-int64 out-of-universe
sentinel, the ``amount_known == (disclosure_aggregate == 0)`` rule derived from
the authenticated flag on every access, all 207,696 delivered rows retained,
the three disjoint row classes, the binding to the reviewed status artifact,
and ``target_year_use: refused_pending_independent_growth_decision`` on every
column including ``E00100``.

What is different
-----------------
One more column, and a different artifact identity. The type is
``microcosm.us.puf_2015_monetary_agi_source_projection`` version 1, with its
own magic line. A twelve-field payload fed to this reader is refused **by
name** rather than reinterpreted, because the two artifacts carry different
column sets and silently reading one as the other is exactly the failure the
twelve-field module's superseded-magic list exists to prevent. Nothing here
derives, reconstructs or reconciles AGI: ``E00100`` is the delivered reported
field, and the document records that the booklets state no component identity
between it and the twelve.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib import resources
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
    StructuralDelta,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SourceCodecRegistry, load_source_bytes

# The reviewed twelve-field contract, imported rather than restated. The
# private names are deliberate: the thirteenth column is authenticated,
# encoded, read back and described by exactly the reviewed code, so a change
# there cannot leave a stale second copy of the same contract here. What this
# module owns is its own artifact identity, not its own parser.
from .puf_monetary_source import (
    _AGGREGATE_COLUMN_FACT_KEYS,
    _ENVELOPE_FACT_KEYS,
    _ENVELOPE_HEADER_KEYS,
    _LEXICAL_KIND,
    _RETURN_COLUMN_FACT_KEYS,
    _STRUCTURAL_DTYPES,
    _TYPED_KIND,
    AGGREGATE_ROW_CLASS,
    BODY_MAX_BYTES,
    HEADER_MAX_BYTES,
    MONETARY_PROJECTION_MAGIC,
    ROW_CLASSES,
    SUPERSEDED_MAGIC,
    PufMonetaryProjection,
    PufMonetarySourceProjection,
    _check_envelope_sources,
    _envelope_column_order,
    _envelope_columns,
    _envelope_int,
    _exact_keys,
    _facts_from_arrays,
    _fixed_ascii,
    _hex64,
    _projection_from_document,
    _read_fixed_ascii,
    _refuse,
    _thawed,
    _verify_readback,
    decode_puf_monetary_source,
    packaged_projection,
    payload_bound,
)
from .puf_raw_source import (
    PUF_RAW_SOURCE_NODE,
    RETURN_STATUS_TYPE,
    PufRawSourceDefinition,
    csv_acceptance_profile,
    decode_return_status,
    puf_raw_source_codecs,
)
from .puf_raw_source import packaged_definition as packaged_raw_definition

__all__ = [
    "AGI_FIELD",
    "AGI_PROJECTION_EDITION",
    "AGI_PROJECTION_MAGIC",
    "AGI_PROJECTION_NODE",
    "AGI_PROJECTION_STAGE",
    "AGI_PROJECTION_TYPE",
    "TWELVE_FIELD_MAGIC",
    "USPufMonetaryAgiSourceKernel",
    "decode_agi_projection",
    "encode_agi_projection",
    "fixture_agi_projection_document",
    "packaged_agi_projection",
    "register_us_puf_monetary_agi_source_kernels",
    "us_puf_monetary_agi_source_node",
]

#: The one column this edition adds to the accepted twelve.
AGI_FIELD = "E00100"

#: The edition marker the packaged document must declare about itself.
AGI_PROJECTION_EDITION = "thirteen_field_reported_agi"

AGI_PROJECTION_TYPE = ArtifactType(
    "microcosm.us.puf_2015_monetary_agi_source_projection", 1
)
AGI_PROJECTION_MAGIC = b"microcosm.us.puf_2015_monetary_agi_source_projection/1\n"

#: Magic lines this reader refuses by name rather than reinterpreting. These
#: are the twelve-field artifact's own magics: those payloads carry a
#: different column set, and reading one here would report a twelve-column
#: artifact as a thirteen-column one.
TWELVE_FIELD_MAGIC = (MONETARY_PROJECTION_MAGIC, *SUPERSEDED_MAGIC)

AGI_PROJECTION_STAGE = "us_puf_monetary_agi_source"
AGI_PROJECTION_NODE = f"{AGI_PROJECTION_STAGE}.projection"

_AGI_RESOURCE = "puf_2015_monetary_agi_source_projection.json"
_AGI_PACKAGE = "microcosm.build.us_runtime"
_FIXTURE_AUTHORITY = "invented_fixture_nonauthority"

_STATUS_ALIAS = "return_status"
_PROJECTION_ALIAS = "monetary_agi_projection"


# --------------------------------------------------------------------------
# The packaged thirteen-field document
# --------------------------------------------------------------------------


def _check_agi_edition(
    document: Mapping[str, object], projection: PufMonetaryProjection
) -> None:
    """Bind this edition to the accepted twelve-field document, at load time.

    The edition block is not decoration: it names the parent document by
    digest and lists the parent's projected fields, so a thirteen-field
    document that quietly dropped or renamed one of the accepted twelve — or
    that was generated against a different parent — is refused here rather
    than discovered downstream.
    """

    edition = document.get("projection_edition")
    if not isinstance(edition, Mapping):
        raise _refuse("AGI_PROJECTION_EDITION_MISSING")
    if edition.get("edition") != AGI_PROJECTION_EDITION:
        raise _refuse("AGI_PROJECTION_EDITION")
    if list(edition.get("adds", ())) != [AGI_FIELD]:
        raise _refuse("AGI_PROJECTION_EDITION_ADDS")
    parent = packaged_projection()
    if edition.get("parent_document_canonical_sha256") != parent.sha256:
        raise _refuse("AGI_PROJECTION_PARENT_DOCUMENT")
    if tuple(edition.get("parent_projected_fields", ())) != parent.fields:
        raise _refuse("AGI_PROJECTION_PARENT_FIELDS")
    fields = projection.fields
    if AGI_FIELD not in fields:
        raise _refuse("AGI_PROJECTION_FIELD_MISSING", AGI_FIELD)
    if set(fields) != set(parent.fields) | {AGI_FIELD}:
        raise _refuse("AGI_PROJECTION_FIELD_SET")
    if len(fields) != len(parent.fields) + 1:
        raise _refuse("AGI_PROJECTION_FIELD_COUNT")
    # Every accepted column must still declare exactly what it declared
    # upstream. This edition adds a column; it does not restate the twelve.
    upstream = {column.field: column for column in parent.columns}
    for column in projection.columns:
        if column.field == AGI_FIELD:
            continue
        if column != upstream[column.field]:
            raise _refuse("AGI_PROJECTION_PARENT_COLUMN", column.field)


def _packaged_agi_bytes() -> bytes:
    return resources.files(_AGI_PACKAGE).joinpath(_AGI_RESOURCE).read_bytes()


_PACKAGED_AGI: PufMonetaryProjection | None = None


def packaged_agi_projection() -> PufMonetaryProjection:
    """The one closed thirteen-field document a production run may use."""

    global _PACKAGED_AGI
    if _PACKAGED_AGI is None:
        document = json.loads(_packaged_agi_bytes().decode("utf-8"))
        projection = _projection_from_document(
            document, route="packaged", definition=packaged_raw_definition()
        )
        _check_agi_edition(document, projection)
        _PACKAGED_AGI = projection
    return _PACKAGED_AGI


def fixture_agi_projection_document(
    document: Mapping[str, object], definition: PufRawSourceDefinition
) -> PufMonetaryProjection:
    """Build a thirteen-field projection on the explicit non-production route.

    Same rules as the twelve-field fixture route: ``route="test_fixture"``,
    the invented-fixture authority, and no restatement of a packaged source
    pin. It must still name ``E00100``, because a fixture without it would be
    testing the twelve-field contract under this module's name.
    """

    projection = _projection_from_document(
        document, route="test_fixture", definition=definition
    )
    if document.get("authority") != _FIXTURE_AUTHORITY:
        raise _refuse("FIXTURE_AUTHORITY")
    packaged_raw = packaged_raw_definition()
    genuine = {packaged_raw.main.sha256, packaged_raw.demographic.sha256}
    stated = {
        document["sources"]["main"]["sha256"],
        document["sources"]["demographic"]["sha256"],
    }
    if stated & genuine:
        raise _refuse("FIXTURE_RESTATES_PACKAGED_PIN")
    if AGI_FIELD not in projection.fields:
        raise _refuse("AGI_PROJECTION_FIELD_MISSING", AGI_FIELD)
    return projection


# --------------------------------------------------------------------------
# The typed envelope
# --------------------------------------------------------------------------


def encode_agi_projection(
    decoded: PufMonetarySourceProjection,
    projection: PufMonetaryProjection,
    definition: PufRawSourceDefinition,
    *,
    status_artifact_sha256: str,
) -> bytes:
    """Encode a thirteen-field projection into its own bounded envelope.

    Byte-for-byte the twelve-field layout — magic line, big-endian header
    length, canonical JSON header, little-endian typed bodies then fixed-width
    lexical bodies — under a different magic and type, with the column bodies
    the shared order helper produces for this projection.
    """

    if AGI_FIELD not in projection.fields:
        raise _refuse("AGI_PROJECTION_FIELD_MISSING", AGI_FIELD)
    rows = decoded.rows
    total, sizes = payload_bound(rows, projection)
    if total > BODY_MAX_BYTES:
        raise _refuse("BODY_OVER_BOUND", total)
    dtypes = {
        **_STRUCTURAL_DTYPES,
        **{column.field: column.numpy_dtype for column in projection.columns},
    }
    widths = {column.field: column.lexical_width for column in projection.columns}
    bodies: list[bytes] = []
    columns: list[dict[str, object]] = []
    for name in _envelope_column_order(projection):
        if name.endswith("_lexical"):
            field = name.removesuffix("_lexical")
            body = _fixed_ascii(decoded.lexical[field], widths[field], field)
            entry: dict[str, object] = {
                "name": name,
                "kind": _LEXICAL_KIND,
                "width": widths[field],
            }
        else:
            array = np.ascontiguousarray(decoded.typed[name], dtype=dtypes[name])
            if array.shape != (rows,):
                raise _refuse("COLUMN_SHAPE", name)
            body = array.tobytes()
            entry = {"name": name, "kind": _TYPED_KIND, "dtype": dtypes[name]}
        if len(body) != sizes[name]:
            raise _refuse("COLUMN_SIZE", name)
        bodies.append(body)
        entry["bytes"] = len(body)
        entry["sha256"] = hashlib.sha256(body).hexdigest()
        columns.append(entry)
    header = {
        "schema_version": AGI_PROJECTION_TYPE.schema_version,
        "type": [AGI_PROJECTION_TYPE.name, AGI_PROJECTION_TYPE.schema_version],
        "rows": rows,
        "projection_sha256": projection.sha256,
        "projection_route": projection.route,
        "raw_source_definition_sha256": definition.sha256,
        "status_artifact_sha256": _hex64(
            status_artifact_sha256, "status_artifact_sha256"
        ),
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
        "facts": dict(decoded.facts),
    }
    encoded = canonical_json(header)
    if len(encoded) > HEADER_MAX_BYTES:
        raise _refuse("HEADER_OVER_BOUND", len(encoded))
    return b"".join(
        (AGI_PROJECTION_MAGIC, len(encoded).to_bytes(4, "big"), encoded, *bodies)
    )


def _resolve_agi_projection(
    header: Mapping[str, object], projection: PufMonetaryProjection | None
) -> PufMonetaryProjection:
    """The thirteen-field projection this payload may be checked against."""

    route = header["projection_route"]
    if route not in ("packaged", "test_fixture"):
        raise _refuse("ENVELOPE_PROJECTION_ROUTE")
    digest = _hex64(header["projection_sha256"], "envelope.projection_sha256")
    if projection is not None:
        if not isinstance(projection, PufMonetaryProjection):
            raise _refuse("DECODE_PROJECTION_TYPE")
        if projection.route != route or projection.sha256 != digest:
            raise _refuse("ENVELOPE_PROJECTION_MISMATCH")
        if AGI_FIELD not in projection.fields:
            raise _refuse("AGI_PROJECTION_FIELD_MISSING", AGI_FIELD)
        return projection
    if route == "packaged":
        packaged = packaged_agi_projection()
        if packaged.sha256 != digest:
            raise _refuse("ENVELOPE_PROJECTION_NOT_PACKAGED")
        return packaged
    # A fixture-route payload has no reconstructible projection document, so
    # there is nothing to close its column set against.
    raise _refuse("ENVELOPE_PROJECTION_REQUIRED")


def decode_agi_projection(
    payload: bytes,
    projection: PufMonetaryProjection | None = None,
    *,
    expected_status_artifact_sha256: str | None = None,
) -> PufMonetarySourceProjection:
    """Decode a thirteen-field payload and re-prove it, never trusting it.

    Every reported fact is recomputed from the arrays actually read back by
    the reviewed derivation and must agree exactly, and every row re-parses
    under the universe its authenticated flag selects.
    """

    if not isinstance(payload, bytes):
        raise _refuse("ENVELOPE_MAGIC")
    if not payload.startswith(AGI_PROJECTION_MAGIC):
        if any(payload.startswith(magic) for magic in TWELVE_FIELD_MAGIC):
            raise _refuse("ENVELOPE_TWELVE_FIELD_MAGIC")
        raise _refuse("ENVELOPE_MAGIC")
    start = len(AGI_PROJECTION_MAGIC)
    if len(payload) < start + 4:
        raise _refuse("ENVELOPE_TRUNCATED")
    length = int.from_bytes(payload[start : start + 4], "big")
    if not 0 < length <= HEADER_MAX_BYTES or len(payload) < start + 4 + length:
        raise _refuse("ENVELOPE_HEADER_LENGTH")
    encoded = payload[start + 4 : start + 4 + length]
    try:
        header = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as error:
        raise _refuse("ENVELOPE_HEADER_JSON") from error
    try:
        recanonical = canonical_json(header)
    except (TypeError, ValueError, RecursionError) as error:
        raise _refuse("ENVELOPE_HEADER_JSON") from error
    if recanonical != encoded:
        raise _refuse("ENVELOPE_HEADER_NOT_CANONICAL")
    _exact_keys(header, _ENVELOPE_HEADER_KEYS, "header")
    if header["schema_version"] != AGI_PROJECTION_TYPE.schema_version or header[
        "type"
    ] != [AGI_PROJECTION_TYPE.name, AGI_PROJECTION_TYPE.schema_version]:
        raise _refuse("ENVELOPE_TYPE")
    if (
        header["header_length_endianness"] != "big"
        or header["body_endianness"] != "little"
    ):
        raise _refuse("ENVELOPE_ENDIANNESS")
    _hex64(header["raw_source_definition_sha256"], "raw_source_definition_sha256")
    _hex64(header["status_artifact_sha256"], "status_artifact_sha256")
    if expected_status_artifact_sha256 is not None:
        _hex64(expected_status_artifact_sha256, "expected_status_artifact_sha256")
        if header["status_artifact_sha256"] != expected_status_artifact_sha256:
            raise _refuse("ENVELOPE_STATUS_ARTIFACT_MISMATCH")
    rows = _envelope_int(header["rows"], "rows", minimum=1)
    resolved = _resolve_agi_projection(header, projection)
    if resolved.raw_definition_sha256 != header["raw_source_definition_sha256"]:
        raise _refuse("ENVELOPE_RAW_DEFINITION_MISMATCH")
    _check_envelope_sources(header, resolved)
    columns = _envelope_columns(header, rows, resolved)

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

    _verify_readback(typed, lexical, resolved)
    facts = header["facts"]
    _exact_keys(facts, _ENVELOPE_FACT_KEYS, "facts")
    for field, classes in facts["column_facts"].items():
        if field not in resolved.fields:
            raise _refuse("ENVELOPE_FACTS", "column_facts")
        _exact_keys(classes, frozenset(ROW_CLASSES), f"column_facts.{field}")
        for name, entry in classes.items():
            keys = (
                _AGGREGATE_COLUMN_FACT_KEYS
                if name == AGGREGATE_ROW_CLASS
                else _RETURN_COLUMN_FACT_KEYS
            )
            _exact_keys(entry, keys, f"column_facts.{field}.{name}")
    recomputed = _facts_from_arrays(typed, lexical, resolved)
    if canonical_json(recomputed) != canonical_json(dict(facts)):
        raise _refuse("ENVELOPE_FACTS_INCONSISTENT")
    if recomputed["rows"] != rows:
        raise _refuse("ENVELOPE_ROWS")
    return PufMonetarySourceProjection(
        typed=MappingProxyType(typed),
        lexical=MappingProxyType(lexical),
        facts=MappingProxyType(recomputed),
        column_metadata=MappingProxyType(dict(recomputed["column_metadata"])),
    )


# --------------------------------------------------------------------------
# The graph producer
# --------------------------------------------------------------------------

_IMPLEMENTATION_MODULES = (
    "microcosm.build.us_runtime.puf_raw_source",
    "microcosm.build.us_runtime.puf_monetary_source",
    "microcosm.build.us_runtime.puf_monetary_agi_projection",
)
_IMPLEMENTATION_IDENTITY_MAGIC = b"us.puf.monetary_agi_source/implementation/1\n"


class USPufMonetaryAgiSourceKernel(KernelBase):
    """Read the pinned main delivery and produce the thirteen-field projection."""

    ref = "us.puf.monetary_agi_source.projection@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=("numpy",),
    )

    def __init__(
        self,
        *,
        projection: PufMonetaryProjection | None = None,
        definition: PufRawSourceDefinition | None = None,
        source_codecs: SourceCodecRegistry | None = None,
    ) -> None:
        self.definition = (
            packaged_raw_definition() if definition is None else definition
        )
        self.projection = (
            packaged_agi_projection() if projection is None else projection
        )
        if self.projection.route not in ("packaged", "test_fixture"):
            raise _refuse("KERNEL_PROJECTION_ROUTE")
        if self.projection.route != self.definition.route:
            raise _refuse("KERNEL_ROUTE_DISAGREEMENT")
        if self.projection.raw_definition_sha256 != self.definition.sha256:
            raise _refuse("KERNEL_DEFINITION_MISMATCH")
        if AGI_FIELD not in self.projection.fields:
            raise _refuse("AGI_PROJECTION_FIELD_MISSING", AGI_FIELD)
        if self.projection.route == "packaged":
            packaged = packaged_agi_projection()
            if (
                self.projection.canonical != packaged.canonical
                or self.projection.sha256 != packaged.sha256
                or canonical_json(_thawed(self.projection.document))
                != self.projection.canonical
            ):
                raise _refuse("KERNEL_PROJECTION_NOT_PACKAGED")
        self.source_codecs = (
            puf_raw_source_codecs(self.definition)
            if source_codecs is None
            else source_codecs
        )

    def implementation_hash(self) -> str:
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
        projection = self.projection
        if (
            context.node.kernel != self.ref
            or context.node.structural is not StructuralDelta.NONE
            or context.node.sources != (definition.main.source_name,)
            or context.node.inputs
            or context.node.outputs
            or len(context.node.artifact_inputs) != 1
            or context.node.artifact_inputs[0].type != RETURN_STATUS_TYPE
            or context.node.artifact_outputs
            != (ArtifactOutput(_PROJECTION_ALIAS, AGI_PROJECTION_TYPE),)
            or set(context.params) != {"definition", "projection"}
        ):
            raise _refuse("NODE_DECLARATION")
        if context.params["definition"] != definition.params_text:
            raise _refuse("NODE_DEFINITION_PARAM")
        if context.params["projection"] != projection.params_text:
            raise _refuse("NODE_PROJECTION_PARAM")
        alias = context.node.artifact_inputs[0].name
        value = context.artifacts[alias]
        if value.type != RETURN_STATUS_TYPE:
            raise _refuse("STATUS_ARTIFACT_TYPE")
        status = decode_return_status(value.payload, definition)
        main_bytes = load_source_bytes(
            definition.main.codec,
            context.sources[definition.main.source_name],
            registry=self.source_codecs,
        )
        decoded = decode_puf_monetary_source(main_bytes, status, projection, definition)
        payload = encode_agi_projection(
            decoded,
            projection,
            definition,
            status_artifact_sha256=hashlib.sha256(value.payload).hexdigest(),
        )
        return KernelResult(
            artifacts={_PROJECTION_ALIAS: payload},
            receipt={
                "puf_monetary_agi_source_projection": {
                    "projection_sha256": projection.sha256,
                    "projection_route": projection.route,
                    "projection_edition": AGI_PROJECTION_EDITION,
                    "definition_sha256": definition.sha256,
                    "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                    "artifact_bytes": len(payload),
                    "status_artifact_sha256": hashlib.sha256(value.payload).hexdigest(),
                    "main_sha256": definition.main.sha256,
                    "csv_acceptance_profile": csv_acceptance_profile(),
                    "projected_fields": list(projection.fields),
                    "agi_field": AGI_FIELD,
                    "agi_is_derived": False,
                    **{
                        key: value
                        for key, value in decoded.facts.items()
                        if key != "column_facts"
                    },
                }
            },
        )


def us_puf_monetary_agi_source_node(
    *,
    population: str,
    producer: str = PUF_RAW_SOURCE_NODE,
    producer_output: str = "return_status",
    projection: PufMonetaryProjection | None = None,
    definition: PufRawSourceDefinition | None = None,
    stage: str = AGI_PROJECTION_STAGE,
) -> Node:
    """Declare the thirteen-field producer inside an explicitly named host."""

    if not isinstance(population, str) or not population:
        raise _refuse("NODE_POPULATION_REQUIRED")
    resolved_definition = (
        packaged_raw_definition() if definition is None else definition
    )
    resolved = packaged_agi_projection() if projection is None else projection
    if resolved.raw_definition_sha256 != resolved_definition.sha256:
        raise _refuse("NODE_DEFINITION_MISMATCH")
    if AGI_FIELD not in resolved.fields:
        raise _refuse("AGI_PROJECTION_FIELD_MISSING", AGI_FIELD)
    return Node(
        f"{stage}.projection",
        USPufMonetaryAgiSourceKernel.ref,
        population=population,
        sources=(resolved_definition.main.source_name,),
        params={
            "definition": resolved_definition.params_text,
            "projection": resolved.params_text,
        },
        artifact_inputs=(
            ArtifactInput(_STATUS_ALIAS, producer, producer_output, RETURN_STATUS_TYPE),
        ),
        artifact_outputs=(ArtifactOutput(_PROJECTION_ALIAS, AGI_PROJECTION_TYPE),),
        description=(
            "Decode the twelve accepted 2015 PUF monetary source columns plus "
            "reported E00100 at return grain, bound to the reviewed status "
            "artifact. Owns no population columns and admits no target year."
        ),
    )


def register_us_puf_monetary_agi_source_kernels(
    registry: KernelRegistry,
    *,
    projection: PufMonetaryProjection | None = None,
    definition: PufRawSourceDefinition | None = None,
    source_codecs: SourceCodecRegistry | None = None,
) -> KernelRegistry:
    """Register the thirteen-field producer kernel into ``registry``."""

    registry.register(
        USPufMonetaryAgiSourceKernel(
            projection=projection,
            definition=definition,
            source_codecs=source_codecs,
        )
    )
    return registry
