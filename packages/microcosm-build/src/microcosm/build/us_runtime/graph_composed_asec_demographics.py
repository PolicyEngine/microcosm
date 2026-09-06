"""Bind the accepted ASEC demographic projection to the composed ASEC arm.

:mod:`.graph_composed_asec_binding` records ``is_female`` and
``is_household_head`` as **unbound** on the composed prepared-ASEC/native-ACS
population, and says exactly why: binding either one needs an explicit source
contract plus a reviewed cross-arm decision that the arm-binding stage may not
make. :mod:`.asec_demographic_source` is that source contract — the printed
``A_SEX``/``AXSEX``/``A_EXPRRP`` entries, their per-person binding states, and
the household-membership fence the headship verdict rests on. This module is
the decision that joins the two, and nothing more.

The stage is ``composed_asec_demographic_binding_v1``. It is an engineering
intermediate: it is not a certified population, a calibration, a transfer, a
score, a national sex calibration or a release, and every receipt it writes
says ``release_eligible: false``.

Two nodes
---------

``composed_population.asec_demographic_source``
    Reads one pinned projection artifact through the generic ``raw-bytes-v1``
    codec and re-publishes it as a typed artifact together with an identity
    document. It touches no population table.

``composed_population.asec_demographic_bind``
    Resolves the composed ASEC arm's rows against that projection by declared
    source, cohort and native identities, and owns five cells per bound person
    under the arm-binding stage's boolean row mask.

What the projection is, and what admitting it does and does not claim
---------------------------------------------------------------------
The artifact is the byte-for-byte bundle
:func:`~.asec_demographic_source.write_asec_demographic_source` emits. Its
producer admits candidate bytes only after independently reconstructing them
from the pinned Census members; **this module cannot and does not repeat that
reconstruction**, because a graph binding may not re-read a restricted source.
So :func:`admit_asec_demographic_projection` issues a deliberately weaker type
than :class:`~.asec_demographic_source.AuthenticatedAsecDemographicSource`, and
the difference is the whole point of naming it differently:

* it verifies the transport frame, the canonical header, the declared column
  roster, the body length, the body digest and every per-column digest;
* it requires the header's ``contract`` to equal this checkout's
  :func:`~.asec_demographic_source.contract_document` and its ``implementation``
  to equal this checkout's demographic implementation identity, so a projection
  produced under a different mapping or a different reader refuses here rather
  than being bound under today's code meanings;
* it **recomputes** every derived state, the household-membership verification
  and the full diagnostics block from the projection's own carried
  observations, with the accepted
  :func:`~.asec_demographic_source.classify_asec_demographic_observations` and
  :func:`~.asec_demographic_source.verify_household_membership`, and refuses a
  header that disagrees with its own body; and
* it does **not** attest that those observations came from the Census members.
  That attestation is the producer's, is recorded in the accepted extraction
  receipt, and reaches the graph as the declared source pin the executor keys
  on — not as anything this module could check.

The published-household grouping the classifier needs is not carried by the
artifact, and is not invented here: the header records a proven bidirectional
bijection between the parent roster's ``person_household_id`` partition and the
``(income_year, source_household_id)`` partition, so the recomputation groups by
the latter, which the bijection makes the identical partition. That recorded
bijection is itself re-derived from the carried coordinates before it is used.

How a composed row is resolved to a projection row
--------------------------------------------------
By the composed row's declared cohort and the ASEC arm's own pre-remap person
identifier — never by row position, and never by a provenance column. This
module reads no support channel and no spine-source-ID column: the arm's rows
arrive as the declared boolean mask ``person_composed_asec_row`` and the arm's
own identifiers arrive inside the typed ``arm_rows`` artifact, both already
resolved, digested and admitted by the arm-binding node.

1. The join key is the pair ``(source_year, arm person identifier)``. The
   native identifier alone is not the key: the three cohorts are distinct
   populations and an identifier that resolves in another cohort is a cohort
   mismatch, reported as one.
2. A repeated key on either side refuses (ambiguous), a key the projection
   never declares refuses (unresolved), and neither ever drops a row.
3. ``source_household_id``, ``A_LINENO`` and ``A_AGE`` — the crosschecks the
   projection itself declares — must agree cell by cell at the resolved rows.
4. The household-membership fence is restated on the composed arm with the
   accepted verifier: the composed ``person_household_id`` partition must be
   bijective with ``(source_year, source_household_id)`` there too. A native
   household id reused by a different cohort is admitted and counted; a merged
   or split household refuses.
5. Household reference status is a whole-household verdict, so the arm is held
   to it: a bound household has exactly one reference person and no unknown
   member, an unbound household is unknown for every member, and a household
   that is partly bound refuses.

The mapping, stated once
------------------------
Only the printed, self-labelled codes bind, through the states the accepted
classifier already derived:

===========================  ==============================  ===============
Canonical column             Source state                    Known codes
===========================  ==============================  ===============
``is_female``                ``asec_sex_binding_state``      1 -> ``False``,
                                                             2 -> ``True``
``is_household_head``        ``asec_household_reference_``   1 -> ``True``,
                             ``state``                       2 -> ``False``
===========================  ==============================  ===============

State ``0`` is *unbound* and stays unbound: it is written as a declared null in
a nullable ``boolean`` cell, never as ``False``. Nothing here maps "not female"
onto male or "no reference code" onto not-head, and no row is defaulted to
zero. The source's own allocation state and both unbound-reason codes are
carried alongside as ``Int64`` cells, so an unknown is not merely null but
says which printed-code condition failed.

``is_household_head`` is the **household reference person** the Census prints,
and nothing else. It is not a tax-unit head, not a family reference person and
not a relationship inference: ``A_FAMREL`` is not read, ``A_LINENO`` is a join
crosscheck, and ``P_SEQ`` is carried by the source as a diagnostic that never
supplies a verdict. Microunit's relationship construction is untouched.

What is not harmonized, and what is not claimed
-----------------------------------------------
The ASEC observation is the interview household one year after the income year,
so the three cohorts are observed in survey years 2023/2024/2025 while the
native ACS arm is observed in its own vintage. No person is aged from any date
and the source's own observed age stays unharmonized. The native ACS arm is
**not** bound: both canonical cells stay declared-null on every ACS row, which
is a recorded gap and not a measurement. A half-populated canonical column is
not a national sex distribution, is not calibrated to any control total, and
does not make this population launch-eligible.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import load_source_bytes

from . import asec_demographic_source as demographic
from .graph_asec_prepared import PreparedGraphError, _artifact, _same_producer
from .graph_composed_asec_binding import (
    BIND_NODE,
    COHORT_COLUMN,
    US_COMPOSED_ASEC_ARM_ROWS_TYPE,
    US_COMPOSED_ASEC_BINDING_TYPE,
    arm_row_column,
    bind_composed_asec_arm_rows,
    bind_composed_asec_document,
)
from .graph_composed_asec_measures import (
    REPORTED_INCOME_NODE,
    composed_asec_measure_nodes,
    composed_asec_population_registry,
)
from .graph_composed_population import (
    COMPOSED_SOURCES,
    GEOGRAPHY_PHASE,
    HARMONIZE_NODE,
    composed_population_nodes,
)
from .graph_context import US_FRAME_CONTEXT_TYPE, _decode, _mass_records, _row_identity
from .graph_geography import LOOKUP_SOURCES, us_geography_nodes
from .graph_implementation import (
    STAGE_DEPENDENCIES,
    implementation_hash,
    implementation_manifest,
)

DEMOGRAPHIC_STAGE = "composed_asec_demographic_binding_v1"
DEMOGRAPHIC_PHASE = "bind_composed_asec_demographics"

SOURCE_NODE = "composed_population.asec_demographic_source"
DEMOGRAPHIC_BIND_NODE = "composed_population.asec_demographic_bind"

#: The one external input this stage reads. It is the artifact the accepted
#: producer writes, admitted through the generic byte codec and pinned by the
#: executor's own content key; this module interprets the payload.
PROJECTION_SOURCE_NAME = "us_asec_demographic_projection"
DEMOGRAPHIC_SOURCES = (
    SourceRef(
        PROJECTION_SOURCE_NAME,
        "raw-bytes-v1",
        "Accepted ASEC demographic source projection bundle.",
    ),
)

US_ASEC_DEMOGRAPHIC_PROJECTION_TYPE = ArtifactType(
    "microcosm.us.asec_demographic_projection", 1
)
US_ASEC_DEMOGRAPHIC_SOURCE_DOCUMENT_TYPE = ArtifactType(
    "microcosm.us.asec_demographic_source_document", 1
)
US_COMPOSED_ASEC_DEMOGRAPHIC_BINDING_TYPE = ArtifactType(
    "microcosm.us.composed_asec_demographic_binding", 1
)

PROJECTION_SCHEMA = "microcosm.us.asec-demographic-projection.v1"
DEMOGRAPHIC_BINDING_SCHEMA = "microcosm.us.composed-asec-demographic-binding.v1"

#: Canonical population cells this stage owns, all masked to the ASEC arm.
FEMALE_COLUMN = "is_female"
HOUSEHOLD_HEAD_COLUMN = "is_household_head"
#: Retained source provenance: the printed allocation flag's state and the two
#: unbound-reason codes. An unknown canonical cell is a declared null *and*
#: names the printed-code condition that failed.
SEX_ALLOCATION_COLUMN = "asec_sex_allocation_state"
SEX_UNBOUND_REASON_COLUMN = "asec_sex_unbound_reason"
HOUSEHOLD_REFERENCE_UNBOUND_REASON_COLUMN = "asec_household_reference_unbound_reason"

BOOLEAN_DTYPE = "boolean"
STATE_DTYPE = "Int64"

#: ``asec_sex_binding_state`` -> ``is_female``. State 0 is unbound and has no
#: entry: it becomes a declared null, never ``False``.
SEX_KNOWN_CODES: Mapping[int, bool] = {1: False, 2: True}
#: ``asec_household_reference_state`` -> ``is_household_head``. Same rule.
HOUSEHOLD_REFERENCE_KNOWN_CODES: Mapping[int, bool] = {1: True, 2: False}

#: (population cell, source column, dtype) for the retained integer provenance.
_RETAINED_STATES = (
    (SEX_ALLOCATION_COLUMN, "asec_sex_allocation_state"),
    (SEX_UNBOUND_REASON_COLUMN, "asec_sex_unbound_reason"),
    (
        HOUSEHOLD_REFERENCE_UNBOUND_REASON_COLUMN,
        "asec_household_reference_unbound_reason",
    ),
)

#: Every population cell this stage owns, in one canonical order. All five are
#: new columns, so the native ACS arm keeps a declared null on each of them.
OWNED_COLUMNS: tuple[str, ...] = (
    FEMALE_COLUMN,
    HOUSEHOLD_HEAD_COLUMN,
    SEX_ALLOCATION_COLUMN,
    SEX_UNBOUND_REASON_COLUMN,
    HOUSEHOLD_REFERENCE_UNBOUND_REASON_COLUMN,
)
#: The two canonical booleans, which are the only cells a downstream model
#: reads under an existing name.
CANONICAL_COLUMNS: tuple[str, ...] = (FEMALE_COLUMN, HOUSEHOLD_HEAD_COLUMN)

#: The composed person columns this node reads, beyond the arm mask and the
#: implicit identity/membership columns. None of them is a provenance column.
_CROSSCHECK_COLUMNS = ("source_household_id", "A_LINENO", "A_AGE")

_HOUSEHOLD_MEMBERSHIP_COLUMN = US_SCHEMA.membership_column("household")
_PERSON_ID_COLUMN = US_SCHEMA.entity_id_column("person")

_PROJECTION_HEADER_KEYS = frozenset(
    {
        "aliases",
        "artifact_kind",
        "body_sha256",
        "column_sha256",
        "columns",
        "contract",
        "crosschecks",
        "diagnostics",
        "dtype",
        "encoding_contract",
        "household_grouping_column",
        "household_membership",
        "implementation",
        "join_keys",
        "native_person_keys_sha256",
        "release_eligible",
        "rows",
        "schema_version",
        "source_frame_sha256",
        "source_identity",
        "sources",
    }
)
_PROJECTION_FRAME_BYTES = len(demographic.MAGIC) + 4
_PROJECTION_MAX_BYTES = (
    _PROJECTION_FRAME_BYTES
    + demographic._HEADER_MAX
    + demographic._MAX_PERSONS * len(demographic.COLUMNS) * 8
    + 32
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PreparedGraphError(reason)


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not set(value) - set("0123456789abcdef")
    )


def _int64_view(series: pd.Series, *, reason: str) -> np.ndarray:
    """Narrow the stack's nullable storage on the arm's own rows, never filling.

    Stacking widens an arm-specific integer column to ``Int64`` so the opposite
    arm's absence stays a declared null. On this arm's own rows nothing may be
    missing and every verifier compares native ``int64`` buffers, so the
    narrowing is explicit and refuses rather than filling.
    """
    _require(pd.api.types.is_integer_dtype(series.dtype), f"{reason}_DTYPE")
    _require(not bool(series.isna().any()), f"{reason}_MISSING")
    return series.to_numpy(dtype="int64")


def _grouping_from_published(
    income_year: np.ndarray, source_household_id: np.ndarray
) -> np.ndarray:
    """Number the published households ``1..n`` without combining coordinates.

    The projection does not carry the parent roster's ``person_household_id``,
    and this module will not invent one. It uses the published key instead,
    which the header's own recorded bijection makes the identical partition —
    and that record is re-derived from these same coordinates before use.
    """
    codes, _count = demographic._dense_codes(income_year, source_household_id)
    return codes + 1


def _recomputed_membership(
    income_year: np.ndarray, source_household_id: np.ndarray
) -> dict:
    return demographic.verify_household_membership(
        income_year=income_year,
        source_household_id=source_household_id,
        household_id=_grouping_from_published(income_year, source_household_id),
    )


class AdmittedAsecDemographicProjection:
    """Framed projection bytes whose header is proven against its own body.

    Weaker than :class:`~.asec_demographic_source.AuthenticatedAsecDemographicSource`
    by exactly one claim, and the name says which: nothing here reconstructs the
    observations from a Census member. What is proven is transport integrity,
    contract and implementation identity, and full internal consistency between
    the declared header and the carried columns.
    """

    __slots__ = ("_body", "_header", "_header_bytes", "_payload_sha256")

    def __init__(self, header_bytes: bytes, body: bytes, payload_sha256: str, _token):
        _require(_token is _PROJECTION_TOKEN, "DEMOGRAPHIC_PROJECTION_CONSTRUCTOR")
        self._header_bytes = header_bytes
        self._header = json.loads(header_bytes.decode("ascii"))
        self._body = body
        self._payload_sha256 = payload_sha256

    @property
    def header(self) -> dict:
        return json.loads(self._header_bytes.decode("ascii"))

    @property
    def rows(self) -> int:
        return int(self._header["rows"])

    @property
    def content_sha256(self) -> str:
        """The producer's own ``header + body`` identity."""
        return demographic._sha(self._header_bytes + self._body)

    @property
    def payload_sha256(self) -> str:
        """The identity of the framed bytes the executor keyed the source on."""
        return self._payload_sha256

    def array(self, name: str) -> np.ndarray:
        _require(name in demographic.COLUMNS, f"DEMOGRAPHIC_PROJECTION_COLUMN:{name}")
        rows = self.rows
        offset = demographic.COLUMNS.index(name) * rows * 8
        return np.frombuffer(self._body, dtype="<i8", count=rows, offset=offset)


_PROJECTION_TOKEN = object()


def admit_asec_demographic_projection(
    payload: bytes,
) -> AdmittedAsecDemographicProjection:
    """Admit pinned projection bytes; refuse anything that is not self-proving."""
    _require(
        type(payload) is bytes
        and _PROJECTION_FRAME_BYTES + 32 < len(payload) <= _PROJECTION_MAX_BYTES,
        "DEMOGRAPHIC_PROJECTION_SIZE",
    )
    _require(payload.startswith(demographic.MAGIC), "DEMOGRAPHIC_PROJECTION_MAGIC")
    _require(
        demographic._sha(payload[:-32]) == payload[-32:].hex(),
        "DEMOGRAPHIC_PROJECTION_CHECKSUM",
    )
    size = struct.unpack_from("<I", payload, len(demographic.MAGIC))[0]
    start = _PROJECTION_FRAME_BYTES
    _require(
        0 < size <= demographic._HEADER_MAX and start + size < len(payload) - 32,
        "DEMOGRAPHIC_PROJECTION_HEADER_SIZE",
    )
    header_bytes = payload[start : start + size]
    try:
        header = json.loads(header_bytes.decode("ascii"))
    except (UnicodeError, ValueError):
        raise PreparedGraphError("DEMOGRAPHIC_PROJECTION_HEADER_JSON") from None
    _require(isinstance(header, dict), "DEMOGRAPHIC_PROJECTION_HEADER_JSON")
    _require(
        set(header) == _PROJECTION_HEADER_KEYS, "DEMOGRAPHIC_PROJECTION_HEADER_SCHEMA"
    )
    _require(
        demographic._json(header) == header_bytes,
        "DEMOGRAPHIC_PROJECTION_HEADER_CANONICAL",
    )
    _require(
        header["schema_version"] == 1
        and header["artifact_kind"] == demographic.ARTIFACT_KIND
        and header["dtype"] == "little_endian_int64"
        and header["release_eligible"] is False
        and header["encoding_contract"]
        == "independently_reconstructed_canonical_numeric_bytes_v1",
        "DEMOGRAPHIC_PROJECTION_HEADER_BINDING",
    )
    _require(
        header["columns"] == list(demographic.COLUMNS)
        and header["aliases"]
        == dict(zip(demographic.OBSERVATIONS, demographic.ALIASES, strict=True))
        and header["join_keys"] == ["source_year", "PERIDNUM"]
        and header["crosschecks"] == list(_CROSSCHECK_COLUMNS)
        and header["household_grouping_column"]
        == demographic.HOUSEHOLD_GROUPING_COLUMN,
        "DEMOGRAPHIC_PROJECTION_ROSTER",
    )
    # The mapping and the reader that produced these states must be this
    # checkout's, or the code meanings bound below are not the ones the
    # projection was derived under.
    _require(
        demographic._json(header["contract"])
        == demographic._json(demographic.contract_document()),
        "DEMOGRAPHIC_PROJECTION_CONTRACT",
    )
    _require(
        demographic._json(header["implementation"])
        == demographic._json(demographic._implementation()),
        "DEMOGRAPHIC_PROJECTION_IMPLEMENTATION",
    )
    _require(
        _is_digest(header["body_sha256"])
        and _is_digest(header["source_frame_sha256"])
        and _is_digest(header["native_person_keys_sha256"])
        and isinstance(header["source_identity"], str)
        and bool(header["source_identity"]),
        "DEMOGRAPHIC_PROJECTION_IDENTITIES",
    )
    rows = header["rows"]
    _require(
        type(rows) is int and 0 < rows <= demographic._MAX_PERSONS,
        "DEMOGRAPHIC_PROJECTION_ROWS",
    )
    body = payload[start + size : -32]
    _require(
        len(body) == rows * len(demographic.COLUMNS) * 8,
        "DEMOGRAPHIC_PROJECTION_BODY_LENGTH",
    )
    _require(
        demographic._sha(body) == header["body_sha256"],
        "DEMOGRAPHIC_PROJECTION_BODY",
    )
    _require(
        header["column_sha256"]
        == {
            name: demographic._sha(body[index * rows * 8 : (index + 1) * rows * 8])
            for index, name in enumerate(demographic.COLUMNS)
        },
        "DEMOGRAPHIC_PROJECTION_COLUMN_DIGESTS",
    )
    value = AdmittedAsecDemographicProjection(
        header_bytes, body, demographic._sha(payload), _PROJECTION_TOKEN
    )
    _verify_projection_body(value, header)
    return value


def _verify_projection_body(
    value: AdmittedAsecDemographicProjection, header: Mapping[str, object]
) -> None:
    """Re-derive the header's verdicts from the body with the accepted helpers."""
    income_year = value.array("income_year")
    source_household_id = value.array("source_household_id")
    _require(
        bool(np.isin(income_year, list(demographic.COHORTS)).all()),
        "DEMOGRAPHIC_PROJECTION_COHORT",
    )
    _require(
        bool((source_household_id > 0).all()),
        "DEMOGRAPHIC_PROJECTION_COORDINATE",
    )
    person_id = value.array("person_id")
    _require(
        len(np.unique(person_id)) == len(person_id),
        "DEMOGRAPHIC_PROJECTION_PERSON_ID",
    )
    try:
        membership = _recomputed_membership(income_year, source_household_id)
        classification = demographic.classify_asec_demographic_observations(
            income_year=income_year,
            household_id=_grouping_from_published(income_year, source_household_id),
            a_sex=value.array("asec_A_SEX"),
            axsex=value.array("asec_AXSEX"),
            a_exprrp=value.array("asec_A_EXPRRP"),
            p_seq=value.array("asec_P_SEQ"),
        )
    except demographic.DemographicSourceRefusalError as error:
        raise PreparedGraphError(
            f"DEMOGRAPHIC_PROJECTION_RECOMPUTATION:{error.field}:{error.reason}"
        ) from None
    _require(
        demographic._json(header["household_membership"])
        == demographic._json(membership),
        "DEMOGRAPHIC_PROJECTION_MEMBERSHIP",
    )
    _require(
        demographic._json(header["diagnostics"])
        == demographic._json(classification.diagnostics),
        "DEMOGRAPHIC_PROJECTION_DIAGNOSTICS",
    )
    for name in demographic.DERIVED:
        _require(
            np.array_equal(value.array(name), classification.states[name]),
            f"DEMOGRAPHIC_PROJECTION_STATE:{name}",
        )
    declared_sources = header["sources"]
    _require(
        isinstance(declared_sources, list)
        and len(declared_sources) == len(demographic.COHORTS)
        and all(isinstance(item, dict) for item in declared_sources)
        and sorted(int(item["income_year"]) for item in declared_sources)
        == sorted(demographic.COHORTS),
        "DEMOGRAPHIC_PROJECTION_SOURCES",
    )
    _require(
        all(
            int(item["incumbent_conflicts"]) == 0
            and int(item["native_key_or_age_conflicts"]) == 0
            and int(item["unreferenced_source_rows"]) == 0
            and all(
                int(count) == 0
                for count in item["accepted_source_missing_tokens"].values()
            )
            for item in declared_sources
        ),
        "DEMOGRAPHIC_PROJECTION_SOURCE_COUNTERS",
    )


def projection_document(
    value: AdmittedAsecDemographicProjection, *, declared: Mapping[str, object]
) -> dict:
    """The identity the bind node is held to, and the claim it does not make."""
    header = value.header
    return {
        "schema": PROJECTION_SCHEMA,
        "phase": DEMOGRAPHIC_PHASE,
        "release_eligible": False,
        "certified": False,
        "artifact_kind": demographic.ARTIFACT_KIND,
        "source_name": PROJECTION_SOURCE_NAME,
        "codec": DEMOGRAPHIC_SOURCES[0].codec,
        "declared": {
            "projection_sha256": declared["projection_sha256"],
            "rows": declared["rows"],
            "cohort_rows": declared["cohort_rows"],
        },
        "payload_sha256": value.payload_sha256,
        "content_sha256": value.content_sha256,
        "header_sha256": demographic._sha(value._header_bytes),
        "rows": value.rows,
        "columns": list(demographic.COLUMNS),
        "cohort_rows": _cohort_rows(value),
        "source_identity": header["source_identity"],
        "source_frame_sha256": header["source_frame_sha256"],
        "native_person_keys_sha256": header["native_person_keys_sha256"],
        "contract_sha256": demographic._sha(demographic._json(header["contract"])),
        "implementation_sha256": demographic._sha(
            demographic._json(header["implementation"])
        ),
        "household_membership": header["household_membership"],
        "admission": {
            "transport_frame_verified": True,
            "header_canonical_bytes_verified": True,
            "column_digests_verified": True,
            "contract_identical_to_this_checkout": True,
            "implementation_identical_to_this_checkout": True,
            "derived_states_recomputed_from_carried_observations": True,
            "household_membership_recomputed_from_carried_coordinates": True,
            "diagnostics_recomputed_from_carried_observations": True,
            "reconstructed_from_census_members_here": False,
            "reconstruction_attested_by": (
                "the accepted extraction receipt, carried into this graph as the "
                "pinned source content key; not re-derived by this stage"
            ),
        },
    }


def _cohort_rows(value: AdmittedAsecDemographicProjection) -> dict:
    years = value.array("income_year")
    return {
        str(int(year)): int((years == year).sum())
        for year in sorted({int(item) for item in np.unique(years)})
    }


def declared_projection_identity(
    *, projection_sha256: str, rows: int, cohort_rows: tuple
) -> dict:
    """Normalize the node's declared identity params into document shape."""
    _require(_is_digest(projection_sha256), "DEMOGRAPHIC_DECLARED_DIGEST")
    _require(type(rows) is int and rows > 0, "DEMOGRAPHIC_DECLARED_ROWS")
    _require(
        isinstance(cohort_rows, tuple)
        and bool(cohort_rows)
        and all(
            isinstance(item, tuple)
            and len(item) == 2
            and type(item[0]) is int
            and type(item[1]) is int
            and item[1] > 0
            for item in cohort_rows
        ),
        "DEMOGRAPHIC_DECLARED_COHORTS",
    )
    declared = {str(year): count for year, count in cohort_rows}
    _require(
        len(declared) == len(cohort_rows)
        and sum(declared.values()) == rows
        and set(declared) <= {str(year) for year in demographic.COHORTS},
        "DEMOGRAPHIC_DECLARED_COHORTS",
    )
    return {
        "projection_sha256": projection_sha256,
        "rows": rows,
        "cohort_rows": declared,
    }


def bind_projection_document(payload: bytes) -> dict:
    """Decode the source node's identity document as a typed shape."""
    document = json.loads(payload)
    _require(canonical_json(document) == payload, "DEMOGRAPHIC_SOURCE_CANONICAL")
    _require(isinstance(document, dict), "DEMOGRAPHIC_SOURCE_DOCUMENT")
    _require(document["schema"] == PROJECTION_SCHEMA, "DEMOGRAPHIC_SOURCE_SCHEMA")
    _require(document["phase"] == DEMOGRAPHIC_PHASE, "DEMOGRAPHIC_SOURCE_PHASE")
    _require(
        document["release_eligible"] is False and document["certified"] is False,
        "DEMOGRAPHIC_SOURCE_RELEASE_CLAIM",
    )
    _require(
        document["artifact_kind"] == demographic.ARTIFACT_KIND
        and document["columns"] == list(demographic.COLUMNS),
        "DEMOGRAPHIC_SOURCE_ROSTER",
    )
    _require(
        document["declared"]["projection_sha256"] == document["payload_sha256"]
        and int(document["declared"]["rows"]) == int(document["rows"])
        and document["declared"]["cohort_rows"] == document["cohort_rows"],
        "DEMOGRAPHIC_SOURCE_DECLARED_IDENTITY",
    )
    _require(
        document["admission"]["reconstructed_from_census_members_here"] is False
        and document["admission"]["derived_states_recomputed_from_carried_observations"]
        is True
        and document["admission"][
            "household_membership_recomputed_from_carried_coordinates"
        ]
        is True,
        "DEMOGRAPHIC_SOURCE_ADMISSION_CLAIM",
    )
    _require(
        all(
            _is_digest(value)
            for key, value in document.items()
            if key.endswith("_sha256")
        ),
        "DEMOGRAPHIC_SOURCE_DIGESTS",
    )
    return document


@dataclass(frozen=True)
class ComposedAsecDemographicBinding:
    """One resolved demographic binding: the arm's cells and its document."""

    document: Mapping[str, object]
    context_document: Mapping[str, object]
    person_ids: np.ndarray
    booleans: Mapping[str, np.ndarray]
    unknown: Mapping[str, np.ndarray]
    states: Mapping[str, np.ndarray]

    @property
    def payload(self) -> bytes:
        return canonical_json(self.document)


def _counts(values: np.ndarray, meanings: Mapping[int, str]) -> dict:
    return {
        name: int((values == code).sum()) for code, name in sorted(meanings.items())
    }


def _resolved_projection_rows(
    projection: AdmittedAsecDemographicProjection,
    *,
    income_years: np.ndarray,
    person_ids: np.ndarray,
) -> np.ndarray:
    """Locate ``(cohort, native person id)`` in the projection's own identity.

    Every failure mode refuses rather than dropping a row, and a cohort
    mismatch is named as one instead of hiding inside "not found": the three
    income cohorts are distinct populations, so an identifier that resolves
    only under a different year is a real join failure, not an absence.
    """
    _require(len(person_ids) > 0, "DEMOGRAPHIC_BINDING_EMPTY")
    source_years = projection.array("income_year")
    source_ids = projection.array("person_id")
    source_index = pd.MultiIndex.from_arrays([source_years, source_ids])
    _require(not source_index.has_duplicates, "DEMOGRAPHIC_BINDING_AMBIGUOUS_SOURCE")
    requested = pd.MultiIndex.from_arrays([income_years, person_ids])
    _require(not requested.has_duplicates, "DEMOGRAPHIC_BINDING_AMBIGUOUS_ROWS")
    positions = source_index.get_indexer(requested).astype("int64")
    if not bool((positions >= 0).all()):
        missing = person_ids[positions < 0]
        _require(
            not bool(np.isin(missing, source_ids).any()),
            "DEMOGRAPHIC_BINDING_COHORT_MISMATCH",
        )
        raise PreparedGraphError("DEMOGRAPHIC_BINDING_UNRESOLVED")
    return positions


def _household_reference_constraint(
    households: np.ndarray, states: np.ndarray
) -> tuple[int, int]:
    """Hold the composed arm to the source's whole-household headship verdict.

    A bound household has exactly one reference person and no unknown member;
    an unbound household is unknown for every member. A household that is
    partly bound cannot be either, so it refuses instead of being reduced to a
    majority. Returns the bound and unbound household counts.
    """
    _, inverse = np.unique(households, return_inverse=True)
    total = int(inverse.max()) + 1 if len(inverse) else 0
    unknown = np.bincount(
        inverse, weights=(states == 0).astype(np.int64), minlength=total
    )
    references = np.bincount(
        inverse, weights=(states == 1).astype(np.int64), minlength=total
    )
    members = np.bincount(inverse, minlength=total)
    fully_unknown = unknown == members
    bound = (unknown == 0) & (references == 1)
    _require(
        bool((fully_unknown | bound).all()),
        "DEMOGRAPHIC_HOUSEHOLD_REFERENCE_CARDINALITY",
    )
    return int(bound.sum()), int(fully_unknown.sum())


def resolve_composed_asec_demographics(
    person: pd.DataFrame,
    *,
    projection: AdmittedAsecDemographicProjection,
    binding_document: Mapping[str, object],
    arm_rows,
    source_document: Mapping[str, object],
    context_document: Mapping[str, object],
    producers: Mapping[str, str],
) -> ComposedAsecDemographicBinding:
    """Bind the projection to the composed ASEC arm, or refuse without writing."""
    mask_column = arm_row_column("person")
    _require(mask_column in person, "DEMOGRAPHIC_ARM_MASK_COLUMN")
    _require(bool(person[mask_column].to_numpy().all()), "DEMOGRAPHIC_ARM_MASK")
    recorded = binding_document["arm"]["person"]
    person_ids = person[_PERSON_ID_COLUMN].to_numpy(dtype="int64")
    _require(len(person_ids) == int(recorded["rows"]), "DEMOGRAPHIC_ARM_ROWS")
    _require(
        _row_identity(pd.DataFrame({_PERSON_ID_COLUMN: person_ids}), "person")[
            "ordered_ids_sha256"
        ]
        == recorded["composed_ordered_ids_sha256"],
        "DEMOGRAPHIC_ARM_IDENTITY",
    )
    native_ids = arm_rows.array("person", "original_ids")
    _require(len(native_ids) == len(person_ids), "DEMOGRAPHIC_ARM_ROWS")

    income_years = _int64_view(person[COHORT_COLUMN], reason="DEMOGRAPHIC_ARM_COHORT")
    _require(
        {str(int(year)) for year in np.unique(income_years)}
        <= set(binding_document["cohorts"]),
        "DEMOGRAPHIC_ARM_COHORT_DECLARED",
    )
    positions = _resolved_projection_rows(
        projection, income_years=income_years, person_ids=native_ids
    )
    for column in _CROSSCHECK_COLUMNS:
        _require(column in person, f"DEMOGRAPHIC_CROSSCHECK_COLUMN:{column}")
        _require(
            np.array_equal(
                _int64_view(person[column], reason=f"DEMOGRAPHIC_CROSSCHECK:{column}"),
                projection.array(column)[positions],
            ),
            f"DEMOGRAPHIC_CROSSCHECK:{column}",
        )

    households = _int64_view(
        person[_HOUSEHOLD_MEMBERSHIP_COLUMN], reason="DEMOGRAPHIC_ARM_HOUSEHOLD"
    )
    source_households = _int64_view(
        person["source_household_id"], reason="DEMOGRAPHIC_ARM_NATIVE_HOUSEHOLD"
    )
    try:
        membership = demographic.verify_household_membership(
            income_year=income_years,
            source_household_id=source_households,
            household_id=households,
        )
    except demographic.DemographicSourceRefusalError as error:
        raise PreparedGraphError(
            f"DEMOGRAPHIC_COMPOSED_MEMBERSHIP:{error.field}:{error.reason}"
        ) from None

    sex_state = projection.array("asec_sex_binding_state")[positions]
    head_state = projection.array("asec_household_reference_state")[positions]
    _require(
        bool(np.isin(sex_state, [0, *SEX_KNOWN_CODES]).all()),
        "DEMOGRAPHIC_SEX_STATE",
    )
    _require(
        bool(np.isin(head_state, [0, *HOUSEHOLD_REFERENCE_KNOWN_CODES]).all()),
        "DEMOGRAPHIC_HOUSEHOLD_REFERENCE_STATE",
    )
    states = {
        cell: projection.array(name)[positions] for cell, name in _RETAINED_STATES
    }
    # An unbound state must carry a reason and a bound state must not: the
    # unknown is explicit on both axes or it is not explicit at all.
    _require(
        bool(((states[SEX_UNBOUND_REASON_COLUMN] == 0) == (sex_state != 0)).all()),
        "DEMOGRAPHIC_SEX_STATE_REASON",
    )
    _require(
        bool(
            (
                (states[HOUSEHOLD_REFERENCE_UNBOUND_REASON_COLUMN] == 0)
                == (head_state != 0)
            ).all()
        ),
        "DEMOGRAPHIC_HOUSEHOLD_REFERENCE_STATE_REASON",
    )
    bound_households, unbound_households = _household_reference_constraint(
        households, head_state
    )

    booleans = {
        FEMALE_COLUMN: np.array(
            [SEX_KNOWN_CODES.get(int(code), False) for code in sex_state], dtype=bool
        ),
        HOUSEHOLD_HEAD_COLUMN: np.array(
            [
                HOUSEHOLD_REFERENCE_KNOWN_CODES.get(int(code), False)
                for code in head_state
            ],
            dtype=bool,
        ),
    }
    unknown = {
        FEMALE_COLUMN: sex_state == 0,
        HOUSEHOLD_HEAD_COLUMN: head_state == 0,
    }

    document = _demographic_document(
        binding_document=binding_document,
        source_document=source_document,
        context_document=context_document,
        producers=producers,
        person_ids=person_ids,
        native_ids=native_ids,
        positions=positions,
        income_years=income_years,
        households=households,
        membership=membership,
        sex_state=sex_state,
        head_state=head_state,
        states=states,
        bound_households=bound_households,
        unbound_households=unbound_households,
    )
    updated = json.loads(canonical_json(context_document))
    for cell in OWNED_COLUMNS:
        columns = updated["entities"]["person"]["columns"]
        _require(cell not in columns, f"DEMOGRAPHIC_INCUMBENT_COLUMN:{cell}")
        columns.append(cell)
    return ComposedAsecDemographicBinding(
        document=document,
        context_document=updated,
        person_ids=person_ids,
        booleans=booleans,
        unknown=unknown,
        states=states,
    )


def _binding_block(
    *,
    sex_state: np.ndarray,
    head_state: np.ndarray,
    states: Mapping[str, np.ndarray],
    households: np.ndarray,
) -> dict:
    """Count what was actually bound on these rows; never a source measurement."""
    bound_households, unbound_households = _household_reference_constraint(
        households, head_state
    )
    return {
        "rows": int(len(sex_state)),
        "households": int(len(np.unique(households))),
        FEMALE_COLUMN: {
            "true": int((sex_state == 2).sum()),
            "false": int((sex_state == 1).sum()),
            "unknown": int((sex_state == 0).sum()),
            "unknown_by_reason": {
                name: int(
                    (
                        (sex_state == 0) & (states[SEX_UNBOUND_REASON_COLUMN] == code)
                    ).sum()
                )
                for code, name in sorted(demographic.SEX_UNBOUND_REASONS.items())
                if code != 0
            },
            "unknown_defaulted_to_male": False,
        },
        HOUSEHOLD_HEAD_COLUMN: {
            "true": int((head_state == 1).sum()),
            "false": int((head_state == 2).sum()),
            "unknown": int((head_state == 0).sum()),
            "unknown_by_reason": {
                name: int(
                    (
                        (head_state == 0)
                        & (states[HOUSEHOLD_REFERENCE_UNBOUND_REASON_COLUMN] == code)
                    ).sum()
                )
                for code, name in sorted(
                    demographic.HOUSEHOLD_REFERENCE_UNBOUND_REASONS.items()
                )
                if code != 0
            },
            "unknown_defaulted_to_not_head": False,
            "households_with_one_reference_person": bound_households,
            "households_unknown": unbound_households,
        },
        SEX_ALLOCATION_COLUMN: _counts(
            states[SEX_ALLOCATION_COLUMN], demographic.SEX_ALLOCATION_STATES
        ),
    }


def mapping_document() -> dict:
    """The explicit known-code semantics, and every mapping refused beside them."""
    return {
        "bound": [
            {
                "column": FEMALE_COLUMN,
                "dtype": BOOLEAN_DTYPE,
                "source_state_column": "asec_sex_binding_state",
                "asec_source_columns": ["A_SEX", "AXSEX"],
                "known_codes": {
                    str(code): bool(value)
                    for code, value in sorted(SEX_KNOWN_CODES.items())
                },
                "state_meanings": {
                    str(code): name
                    for code, name in sorted(demographic.SEX_BINDING_STATES.items())
                },
                "unknown_state": 0,
                "unknown_representation": "declared_null",
                "unknown_defaulted_to_male": False,
                "allocation_flag_retained_as": SEX_ALLOCATION_COLUMN,
                "unbound_reason_retained_as": SEX_UNBOUND_REASON_COLUMN,
                "acs_arm_bound": False,
            },
            {
                "column": HOUSEHOLD_HEAD_COLUMN,
                "dtype": BOOLEAN_DTYPE,
                "source_state_column": "asec_household_reference_state",
                "asec_source_columns": ["A_EXPRRP"],
                "known_codes": {
                    str(code): bool(value)
                    for code, value in sorted(HOUSEHOLD_REFERENCE_KNOWN_CODES.items())
                },
                "state_meanings": {
                    str(code): name
                    for code, name in sorted(
                        demographic.HOUSEHOLD_REFERENCE_STATES.items()
                    )
                },
                "unknown_state": 0,
                "unknown_representation": "declared_null",
                "unknown_defaulted_to_not_head": False,
                "unbound_reason_retained_as": (
                    HOUSEHOLD_REFERENCE_UNBOUND_REASON_COLUMN
                ),
                "household_reference_person_codes": list(
                    demographic.HOUSEHOLD_REFERENCE_CODES
                ),
                "is_tax_unit_head": False,
                "is_family_reference_person": False,
                "infers_relationship": False,
                "acs_arm_bound": False,
            },
        ],
        "not_adopted": [dict(item) for item in demographic.NON_ADOPTED_HEAD_SIGNALS],
        "family_relationship_field_read": False,
        "line_number_used_for_headship": False,
        "household_and_family_roles_separate": True,
        "microunit_relationship_construction_changed": False,
    }


def claims_document() -> dict:
    """Everything this stage explicitly does not establish."""
    return {
        "release_eligible": False,
        "certified": False,
        "launch_eligible": False,
        "national_sex_calibration": False,
        "calibrated_to_any_control_total": False,
        "representative_of_any_population": False,
        "congressional_district_fitness": False,
        "cross_arm_period_harmonized": False,
        "aged_from_a_date": False,
        "source_year_observed_age_harmonized": False,
        "acs_arm_demographics_bound": False,
        "acs_arm_cells_are_declared_null_not_measured": True,
        "claim": "engineering_binding_only_no_calibration_or_launch_claim",
    }


def _demographic_document(
    *,
    binding_document: Mapping[str, object],
    source_document: Mapping[str, object],
    context_document: Mapping[str, object],
    producers: Mapping[str, str],
    person_ids: np.ndarray,
    native_ids: np.ndarray,
    positions: np.ndarray,
    income_years: np.ndarray,
    households: np.ndarray,
    membership: Mapping[str, object],
    sex_state: np.ndarray,
    head_state: np.ndarray,
    states: Mapping[str, np.ndarray],
    bound_households: int,
    unbound_households: int,
) -> dict:
    cohorts = sorted({int(year) for year in np.unique(income_years)})
    return {
        "schema": DEMOGRAPHIC_BINDING_SCHEMA,
        "phase": DEMOGRAPHIC_PHASE,
        "stage": DEMOGRAPHIC_STAGE,
        "claims": claims_document(),
        "mapping": mapping_document(),
        "owned_columns": {
            "person": [
                {
                    "column": FEMALE_COLUMN,
                    "dtype": BOOLEAN_DTYPE,
                    "rows": arm_row_column("person"),
                },
                {
                    "column": HOUSEHOLD_HEAD_COLUMN,
                    "dtype": BOOLEAN_DTYPE,
                    "rows": arm_row_column("person"),
                },
                *(
                    {
                        "column": cell,
                        "dtype": STATE_DTYPE,
                        "rows": arm_row_column("person"),
                    }
                    for cell, _name in _RETAINED_STATES
                ),
            ]
        },
        "resolution": {
            "keys": [COHORT_COLUMN, "arm_person_identifier"],
            "source_identity": "asec_demographic_source.person_id",
            "crosschecks": list(_CROSSCHECK_COLUMNS),
            "uses_composed_row_position": False,
            "uses_provenance_column": False,
            "uses_source_row_order": False,
            "requires_monotone_source_positions": False,
            "arm_rows_artifact": "composed_asec_arm_rows",
        },
        "arm": {
            "rows": int(len(person_ids)),
            "composed_ordered_ids_sha256": _row_identity(
                pd.DataFrame({_PERSON_ID_COLUMN: person_ids}), "person"
            )["ordered_ids_sha256"],
            "original_ordered_ids_sha256": _row_identity(
                pd.DataFrame({_PERSON_ID_COLUMN: native_ids}), "person"
            )["ordered_ids_sha256"],
            "projection_positions_sha256": demographic._sha(
                np.ascontiguousarray(positions, dtype="<i8").tobytes()
            ),
            "projection_is_whole_source": bool(
                len(positions) == int(source_document["rows"])
            ),
        },
        "composed": {
            entity: {
                "rows": int(context_document["entities"][entity]["rows"]),
                "ordered_ids_sha256": context_document["entities"][entity][
                    "ordered_ids_sha256"
                ],
            }
            for entity in US_SCHEMA.entities
        },
        "household_membership": membership,
        "observed_period": {
            "kind": demographic.OBSERVED_PERIOD_KIND,
            "by_cohort": {
                str(year): {
                    "income_year": int(year),
                    "asec_survey_year": int(demographic.COHORTS[int(year)]),
                    "aged_from_a_date": False,
                    "cross_arm_period_harmonized": False,
                }
                for year in cohorts
            },
        },
        "bound": {
            "totals": _binding_block(
                sex_state=sex_state,
                head_state=head_state,
                states=states,
                households=households,
            ),
            "by_cohort": {
                str(year): _binding_block(
                    sex_state=sex_state[income_years == year],
                    head_state=head_state[income_years == year],
                    states={
                        name: values[income_years == year]
                        for name, values in states.items()
                    },
                    households=households[income_years == year],
                )
                for year in cohorts
            },
        },
        "households_with_one_reference_person": bound_households,
        "households_unknown": unbound_households,
        "inputs": {
            "asec_binding_sha256": demographic._sha(
                canonical_json(dict(binding_document))
            ),
            "demographic_source_sha256": demographic._sha(
                canonical_json(dict(source_document))
            ),
            "frame_context_sha256": demographic._sha(
                canonical_json(dict(context_document))
            ),
            "projection_payload_sha256": source_document["payload_sha256"],
            "projection_content_sha256": source_document["content_sha256"],
            "projection_contract_sha256": source_document["contract_sha256"],
            "projection_implementation_sha256": source_document[
                "implementation_sha256"
            ],
            "producers": {name: producers[name] for name in sorted(producers)},
        },
    }


_DOCUMENT_KEYS = frozenset(
    {
        "arm",
        "bound",
        "claims",
        "composed",
        "household_membership",
        "households_unknown",
        "households_with_one_reference_person",
        "inputs",
        "mapping",
        "observed_period",
        "owned_columns",
        "phase",
        "resolution",
        "schema",
        "stage",
    }
)


def bind_composed_asec_demographic_document(payload: bytes) -> dict:
    """Decode the demographic binding artifact as a typed shape.

    Like the arm binding's own decoder this checks canonical bytes and declared
    shape; it verifies no digest against any population. What it will not admit
    is a document that has quietly changed what the mapping means, what the
    unknowns became, or what the stage claims.
    """
    document = json.loads(payload)
    _require(canonical_json(document) == payload, "DEMOGRAPHIC_BINDING_CANONICAL")
    _require(isinstance(document, dict), "DEMOGRAPHIC_BINDING_DOCUMENT")
    _require(set(document) == _DOCUMENT_KEYS, "DEMOGRAPHIC_BINDING_KEYS")
    _require(
        document["schema"] == DEMOGRAPHIC_BINDING_SCHEMA
        and document["phase"] == DEMOGRAPHIC_PHASE
        and document["stage"] == DEMOGRAPHIC_STAGE,
        "DEMOGRAPHIC_BINDING_SCHEMA",
    )
    _require(document["claims"] == claims_document(), "DEMOGRAPHIC_BINDING_CLAIMS")
    _require(document["mapping"] == mapping_document(), "DEMOGRAPHIC_BINDING_MAPPING")
    _require(
        [item["column"] for item in document["owned_columns"]["person"]]
        == list(OWNED_COLUMNS)
        and all(
            item["rows"] == arm_row_column("person")
            for item in document["owned_columns"]["person"]
        )
        and [item["dtype"] for item in document["owned_columns"]["person"]]
        == [BOOLEAN_DTYPE, BOOLEAN_DTYPE, *([STATE_DTYPE] * len(_RETAINED_STATES))],
        "DEMOGRAPHIC_BINDING_OWNED_COLUMNS",
    )
    resolution = document["resolution"]
    _require(
        resolution["keys"] == [COHORT_COLUMN, "arm_person_identifier"]
        and resolution["crosschecks"] == list(_CROSSCHECK_COLUMNS)
        and resolution["uses_composed_row_position"] is False
        and resolution["uses_provenance_column"] is False
        and resolution["uses_source_row_order"] is False,
        "DEMOGRAPHIC_BINDING_RESOLUTION",
    )
    membership = document["household_membership"]
    _require(
        membership["equivalence"] == "bidirectional_partition_bijection"
        and membership["bijective"] is True
        and all(
            int(membership[counter]) == 0
            for counter in demographic.MEMBERSHIP_REFUSAL_COUNTERS
        ),
        "DEMOGRAPHIC_BINDING_MEMBERSHIP",
    )
    period = document["observed_period"]
    _require(
        period["kind"] == demographic.OBSERVED_PERIOD_KIND
        and bool(period["by_cohort"])
        and all(
            entry["aged_from_a_date"] is False
            and entry["cross_arm_period_harmonized"] is False
            and int(entry["asec_survey_year"]) == int(entry["income_year"]) + 1
            for entry in period["by_cohort"].values()
        ),
        "DEMOGRAPHIC_BINDING_PERIOD",
    )
    totals = document["bound"]["totals"]
    _require(
        totals[FEMALE_COLUMN]["unknown_defaulted_to_male"] is False
        and totals[HOUSEHOLD_HEAD_COLUMN]["unknown_defaulted_to_not_head"] is False
        and totals["rows"]
        == sum(int(block["rows"]) for block in document["bound"]["by_cohort"].values()),
        "DEMOGRAPHIC_BINDING_COUNTS",
    )
    _require(
        all(
            _is_digest(value)
            for key, value in document["inputs"].items()
            if key.endswith("_sha256")
        )
        and all(_is_digest(value) for value in document["inputs"]["producers"].values())
        and all(
            _is_digest(value)
            for key, value in document["arm"].items()
            if key.endswith("_sha256")
        ),
        "DEMOGRAPHIC_BINDING_DIGESTS",
    )
    return document


def demographic_owned_declarations() -> tuple[Owned, ...]:
    """The five cells, every one masked to the arm the binding node declared.

    ``Slice.rows`` masks the table a kernel receives and
    ``population._patch_columns`` refuses any change to storage outside an
    ``Owned`` mask, missingness included, so the native ACS arm's cells are held
    at the declared null the executor created them with. The two canonical cells
    are nullable ``boolean`` for the same reason: a masked non-nullable ``bool``
    column cannot be null outside its owned rows, so it could not represent the
    ACS arm's absence — or an ASEC arm unknown — without inventing ``False``.
    """
    mask = arm_row_column("person")
    return (
        Owned("person", FEMALE_COLUMN, BOOLEAN_DTYPE, rows=mask),
        Owned("person", HOUSEHOLD_HEAD_COLUMN, BOOLEAN_DTYPE, rows=mask),
        *(
            Owned("person", cell, STATE_DTYPE, rows=mask)
            for cell, _name in _RETAINED_STATES
        ),
    )


def demographic_bind_inputs() -> tuple[Slice, ...]:
    """The declared person view this node reads, and nothing else."""
    return (
        Slice(
            "person",
            (COHORT_COLUMN, *_CROSSCHECK_COLUMNS, arm_row_column("person")),
            rows=arm_row_column("person"),
        ),
    )


_SOURCE_ARTIFACTS = (
    ArtifactOutput("demographic_projection", US_ASEC_DEMOGRAPHIC_PROJECTION_TYPE),
    ArtifactOutput("demographic_source", US_ASEC_DEMOGRAPHIC_SOURCE_DOCUMENT_TYPE),
)
_BIND_ARTIFACTS = (
    ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
    ArtifactOutput(
        "asec_demographic_binding", US_COMPOSED_ASEC_DEMOGRAPHIC_BINDING_TYPE
    ),
)
_BIND_ARTIFACT_TYPES = (
    ("frame_context", US_FRAME_CONTEXT_TYPE),
    ("asec_binding", US_COMPOSED_ASEC_BINDING_TYPE),
    ("arm_rows", US_COMPOSED_ASEC_ARM_ROWS_TYPE),
    ("demographic_projection", US_ASEC_DEMOGRAPHIC_PROJECTION_TYPE),
    ("demographic_source", US_ASEC_DEMOGRAPHIC_SOURCE_DOCUMENT_TYPE),
)
#: The projection and its identity document are one source read, so they must
#: come from one producing node execution.
_SOURCE_EDGES = ("demographic_projection", "demographic_source")
#: The binding document and the resolved arm rows are positional over one arm
#: resolution, so they must come from one producing node execution too.
_ARM_EDGES = ("asec_binding", "arm_rows")


def _source_params(*, projection_sha256: str, rows: int, cohort_rows: tuple) -> dict:
    return {
        "phase": DEMOGRAPHIC_PHASE,
        "projection_sha256": projection_sha256,
        "rows": rows,
        "cohort_rows": cohort_rows,
    }


def demographic_source_node(
    *, population: str, projection_sha256: str, rows: int, cohort_rows: tuple
) -> Node:
    """Declare the pinned projection import; the identity is normative."""
    declared_projection_identity(
        projection_sha256=projection_sha256, rows=rows, cohort_rows=cohort_rows
    )
    return Node(
        id=SOURCE_NODE,
        kernel=USAsecDemographicSourceKernel.ref,
        population=population,
        sources=(PROJECTION_SOURCE_NAME,),
        params=_source_params(
            projection_sha256=projection_sha256, rows=rows, cohort_rows=cohort_rows
        ),
        artifact_outputs=_SOURCE_ARTIFACTS,
    )


def demographic_bind_node(*, population: str, context_producer: str) -> Node:
    """Declare the one node that writes the canonical demographic cells."""
    return Node(
        id=DEMOGRAPHIC_BIND_NODE,
        kernel=USComposedAsecDemographicBindKernel.ref,
        population=population,
        inputs=demographic_bind_inputs(),
        outputs=demographic_owned_declarations(),
        params={"phase": DEMOGRAPHIC_PHASE},
        artifact_inputs=(
            ArtifactInput(
                "frame_context",
                context_producer,
                "frame_context",
                US_FRAME_CONTEXT_TYPE,
            ),
            ArtifactInput(
                "asec_binding", BIND_NODE, "asec_binding", US_COMPOSED_ASEC_BINDING_TYPE
            ),
            ArtifactInput(
                "arm_rows", BIND_NODE, "arm_rows", US_COMPOSED_ASEC_ARM_ROWS_TYPE
            ),
            ArtifactInput(
                "demographic_projection",
                SOURCE_NODE,
                "demographic_projection",
                US_ASEC_DEMOGRAPHIC_PROJECTION_TYPE,
            ),
            ArtifactInput(
                "demographic_source",
                SOURCE_NODE,
                "demographic_source",
                US_ASEC_DEMOGRAPHIC_SOURCE_DOCUMENT_TYPE,
            ),
        ),
        artifact_outputs=_BIND_ARTIFACTS,
    )


class USAsecDemographicSourceKernel(KernelBase):
    """Admit the pinned demographic projection; write no population cell."""

    ref = "us.composed_population.asec_demographic_source@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=STAGE_DEPENDENCIES[DEMOGRAPHIC_STAGE],
    )

    def implementation_hash(self) -> str:
        return implementation_hash(DEMOGRAPHIC_STAGE)

    def run(self, context: KernelContext) -> KernelResult:
        node = context.node
        _require(node.kernel == self.ref, "NODE_KERNEL")
        _require(
            set(context.params)
            == {"phase", "projection_sha256", "rows", "cohort_rows"},
            "NODE_PARAMS",
        )
        _require(context.params["phase"] == DEMOGRAPHIC_PHASE, "NODE_PHASE")
        _require(node.structural is StructuralDelta.NONE, "NODE_STRUCTURAL")
        _require(tuple(node.sources) == (PROJECTION_SOURCE_NAME,), "NODE_SOURCES")
        _require(not node.inputs and not node.outputs, "NODE_SLICES")
        _require(not node.artifact_inputs, "NODE_ARTIFACT_INPUTS")
        _require(node.artifact_outputs == _SOURCE_ARTIFACTS, "NODE_ARTIFACT_OUTPUTS")
        declared = declared_projection_identity(
            projection_sha256=context.params["projection_sha256"],
            rows=context.params["rows"],
            cohort_rows=context.params["cohort_rows"],
        )
        payload = load_source_bytes(
            DEMOGRAPHIC_SOURCES[0].codec, context.sources[PROJECTION_SOURCE_NAME]
        )
        projection = admit_asec_demographic_projection(payload)
        # The declaration is normative: bytes that are internally perfect but
        # are not the artifact this graph declared are refused here, before any
        # population sees them.
        _require(
            projection.payload_sha256 == declared["projection_sha256"],
            "DEMOGRAPHIC_DECLARED_PROJECTION_DIGEST",
        )
        _require(projection.rows == declared["rows"], "DEMOGRAPHIC_DECLARED_ROWS")
        _require(
            _cohort_rows(projection) == declared["cohort_rows"],
            "DEMOGRAPHIC_DECLARED_COHORT_ROWS",
        )
        document = projection_document(projection, declared=declared)
        return KernelResult(
            artifacts={
                "demographic_projection": payload,
                "demographic_source": canonical_json(document),
            },
            receipt={
                "phase": DEMOGRAPHIC_PHASE,
                "implementation": implementation_manifest(DEMOGRAPHIC_STAGE),
                "projection": document,
                "release_eligible": False,
                "certified": False,
            },
        )


def _context(value) -> dict:
    """Decode the producing node's context, keeping every graph-context check."""
    document = _decode(value.payload)
    _mass_records(document["mass_log"])
    _require(canonical_json(document) == value.payload, "DEMOGRAPHIC_CONTEXT_CANONICAL")
    return document


def _bound_population(document: dict, binding: Mapping[str, object]) -> None:
    """The context and the arm binding must describe one composed population."""
    for entity in US_SCHEMA.entities:
        declared = document["entities"][entity]
        recorded = binding["composed"][entity]
        _require(
            declared["rows"] == recorded["rows"]
            and declared["ordered_ids_sha256"] == recorded["ordered_ids_sha256"],
            f"DEMOGRAPHIC_POPULATION_IDENTITY:{entity}",
        )
    _require(
        arm_row_column("person") in document["entities"]["person"]["columns"],
        "DEMOGRAPHIC_ARM_COLUMN",
    )


class USComposedAsecDemographicBindKernel(KernelBase):
    """Write the canonical sex and household-reference cells on the ASEC arm."""

    ref = "us.composed_population.asec_demographic_bind@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=STAGE_DEPENDENCIES[DEMOGRAPHIC_STAGE],
    )

    def implementation_hash(self) -> str:
        return implementation_hash(DEMOGRAPHIC_STAGE)

    def run(self, context: KernelContext) -> KernelResult:
        node = context.node
        _require(node.kernel == self.ref, "NODE_KERNEL")
        _require(set(context.params) == {"phase"}, "NODE_PARAMS")
        _require(context.params["phase"] == DEMOGRAPHIC_PHASE, "NODE_PHASE")
        _require(node.structural is StructuralDelta.NONE, "NODE_STRUCTURAL")
        _require(not node.sources, "NODE_SOURCES")
        _require(node.inputs == demographic_bind_inputs(), "NODE_SLICES")
        _require(node.outputs == demographic_owned_declarations(), "NODE_OUTPUTS")
        _require(
            set(context.artifacts) == {name for name, _kind in _BIND_ARTIFACT_TYPES},
            "NODE_ARTIFACT_INPUTS",
        )
        _require(node.artifact_outputs == _BIND_ARTIFACTS, "NODE_ARTIFACT_OUTPUTS")
        _require(
            {edge.producer for edge in node.artifact_inputs if edge.name in _ARM_EDGES}
            == {BIND_NODE},
            "NODE_ARM_PRODUCER",
        )
        _require(
            {
                edge.producer
                for edge in node.artifact_inputs
                if edge.name in _SOURCE_EDGES
            }
            == {SOURCE_NODE},
            "NODE_SOURCE_PRODUCER",
        )
        values = {
            name: _artifact(context, name, kind) for name, kind in _BIND_ARTIFACT_TYPES
        }
        arm_producer = _same_producer(context, _ARM_EDGES)
        source_producer = _same_producer(context, _SOURCE_EDGES)
        binding = bind_composed_asec_document(values["asec_binding"].payload)
        source_document = bind_projection_document(values["demographic_source"].payload)
        projection = admit_asec_demographic_projection(
            values["demographic_projection"].payload
        )
        _require(
            projection.payload_sha256 == source_document["payload_sha256"]
            and projection.content_sha256 == source_document["content_sha256"],
            "DEMOGRAPHIC_SOURCE_DOCUMENT_PAYLOAD",
        )
        document = _context(values["frame_context"])
        _bound_population(document, binding)
        arm_rows = bind_composed_asec_arm_rows(
            values["arm_rows"].payload, binding_document=binding
        )
        resolved = resolve_composed_asec_demographics(
            context.tables["person"],
            projection=projection,
            binding_document=binding,
            arm_rows=arm_rows,
            source_document=source_document,
            context_document=document,
            producers={
                "population": values["frame_context"].producer_key,
                "asec_arm_binding": arm_producer,
                "demographic_source": source_producer,
            },
        )
        index = pd.Index(resolved.person_ids, name=_PERSON_ID_COLUMN)
        columns = {}
        for cell in CANONICAL_COLUMNS:
            array = pd.array(resolved.booleans[cell], dtype=BOOLEAN_DTYPE)
            array[resolved.unknown[cell]] = pd.NA
            columns[("person", cell)] = pd.Series(
                array, index=index, dtype=BOOLEAN_DTYPE
            )
        for cell, _name in _RETAINED_STATES:
            columns[("person", cell)] = pd.Series(
                pd.array(resolved.states[cell], dtype=STATE_DTYPE),
                index=index,
                dtype=STATE_DTYPE,
            )
        payload = resolved.payload
        return KernelResult(
            columns=columns,
            artifacts={
                "frame_context": canonical_json(resolved.context_document),
                "asec_demographic_binding": payload,
            },
            receipt={
                "phase": DEMOGRAPHIC_PHASE,
                "implementation": implementation_manifest(DEMOGRAPHIC_STAGE),
                "demographic_binding_sha256": demographic._sha(payload),
                "projection_payload_sha256": projection.payload_sha256,
                "arm_person_rows": int(len(resolved.person_ids)),
                "bound": resolved.document["bound"],
                "mapping": resolved.document["mapping"],
                "claims": resolved.document["claims"],
                "observed_period": resolved.document["observed_period"],
                "release_eligible": False,
                "certified": False,
            },
        )


def composed_asec_demographic_nodes(
    *,
    population: str,
    population_context: str,
    projection_sha256: str,
    rows: int,
    cohort_rows: tuple,
):
    """The arm binding and its measurements, then the demographic binding."""
    measures = composed_asec_measure_nodes(
        population=population, population_context=population_context
    )
    return (
        *measures,
        demographic_source_node(
            population=population,
            projection_sha256=projection_sha256,
            rows=rows,
            cohort_rows=cohort_rows,
        ),
        demographic_bind_node(
            population=population, context_producer=REPORTED_INCOME_NODE
        ),
    )


def composed_asec_demographic_population_graph(
    columns,
    *,
    sample_fraction: float,
    sample_seed: int,
    projection_sha256: str,
    rows: int,
    cohort_rows: tuple,
    geography: bool = True,
) -> Graph:
    """The bound composed population plus this arm's demographic binding."""
    composed = composed_population_nodes(
        columns, sample_fraction=sample_fraction, sample_seed=sample_seed
    )
    identity = {
        "projection_sha256": projection_sha256,
        "rows": rows,
        "cohort_rows": cohort_rows,
    }
    if not geography:
        return Graph(
            country="us",
            sources=(*COMPOSED_SOURCES, *DEMOGRAPHIC_SOURCES),
            nodes=(
                *composed,
                *composed_asec_demographic_nodes(
                    population=HARMONIZE_NODE,
                    population_context=HARMONIZE_NODE,
                    **identity,
                ),
            ),
        )
    return Graph(
        country="us",
        sources=(*COMPOSED_SOURCES, *LOOKUP_SOURCES, *DEMOGRAPHIC_SOURCES),
        nodes=(
            *composed,
            *us_geography_nodes(
                columns, base=HARMONIZE_NODE, context_producer=HARMONIZE_NODE
            ),
            *composed_asec_demographic_nodes(
                population=f"{GEOGRAPHY_PHASE}.boundary",
                population_context=GEOGRAPHY_PHASE,
                **identity,
            ),
        ),
    )


def composed_asec_demographic_registry(*, geography: bool = True) -> KernelRegistry:
    """The bound composed registry plus exactly this stage's two kernels."""
    registry = composed_asec_population_registry(geography=geography)
    registry.register(USAsecDemographicSourceKernel())
    registry.register(USComposedAsecDemographicBindKernel())
    return registry


__all__ = [
    "BOOLEAN_DTYPE",
    "CANONICAL_COLUMNS",
    "DEMOGRAPHIC_BINDING_SCHEMA",
    "DEMOGRAPHIC_BIND_NODE",
    "DEMOGRAPHIC_PHASE",
    "DEMOGRAPHIC_SOURCES",
    "DEMOGRAPHIC_STAGE",
    "FEMALE_COLUMN",
    "HOUSEHOLD_HEAD_COLUMN",
    "HOUSEHOLD_REFERENCE_KNOWN_CODES",
    "HOUSEHOLD_REFERENCE_UNBOUND_REASON_COLUMN",
    "OWNED_COLUMNS",
    "PROJECTION_SCHEMA",
    "PROJECTION_SOURCE_NAME",
    "SEX_ALLOCATION_COLUMN",
    "SEX_KNOWN_CODES",
    "SEX_UNBOUND_REASON_COLUMN",
    "SOURCE_NODE",
    "STATE_DTYPE",
    "US_ASEC_DEMOGRAPHIC_PROJECTION_TYPE",
    "US_ASEC_DEMOGRAPHIC_SOURCE_DOCUMENT_TYPE",
    "US_COMPOSED_ASEC_DEMOGRAPHIC_BINDING_TYPE",
    "AdmittedAsecDemographicProjection",
    "ComposedAsecDemographicBinding",
    "USAsecDemographicSourceKernel",
    "USComposedAsecDemographicBindKernel",
    "admit_asec_demographic_projection",
    "bind_composed_asec_demographic_document",
    "bind_projection_document",
    "claims_document",
    "composed_asec_demographic_nodes",
    "composed_asec_demographic_population_graph",
    "composed_asec_demographic_registry",
    "declared_projection_identity",
    "demographic_bind_inputs",
    "demographic_bind_node",
    "demographic_owned_declarations",
    "demographic_source_node",
    "mapping_document",
    "projection_document",
    "resolve_composed_asec_demographics",
]
