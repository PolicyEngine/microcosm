"""Typed ACS source evidence and exact whole-household graph selection.

Only CREATE authenticates source bytes. Selection keeps the original evidence as
an ancestor and emits selected identities/positions; it never mints source
authority or asserts calibrated representation.
"""

from __future__ import annotations

import json
import struct
from dataclasses import InitVar, dataclass, replace
from pathlib import Path

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
    Slice,
    SourceRef,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.keys import opaque_artifact_key

from . import acs_housing_universe_source as source
from .graph_context import (
    US_FRAME_CONTEXT_TYPE,
    encode_us_frame_context,
    us_frame_from_context,
)
from .graph_implementation import (
    STAGE_DEPENDENCIES,
    implementation_hash,
    implementation_manifest,
)
from .graph_sources import frame_column_declarations

ACS_HU_PAYLOAD_MAX_BYTES = 64 * 1024**2
ACS_HU_HEADER_MAX_BYTES = 65536
MAGIC = bytes.fromhex("4d43414353485502")
US_ACS_HOUSING_UNIVERSE_TYPE = ArtifactType("microcosm.us.acs_housing_universe", 2)
US_ACS_HOUSING_PREPARATION_TYPE = ArtifactType(
    "microcosm.us.acs_housing_preparation", 2
)
US_ACS_HOUSING_SELECTION_TYPE = ArtifactType("microcosm.us.acs_housing_selection", 2)
CREATE_NODE = "acs_housing_create"
SELECT_NODE = "acs_housing_select"
SOURCE_NAME = "acs_housing_source"
PHASE = "acs_housing_universe_graph_v2"
_TOKEN = object()
_CODES = (
    "interview_scope",
    "physical_unit",
    "household_kind",
    "tenure_subtype",
    "occupied_hu",
    "hu_tenure_class",
    "unresolved_reasons",
    "TEN_valid",
)
_HEADER_FIELDS = frozenset(
    {
        "format",
        "release_eligible",
        "source_receipt_bytes",
        "projection_bytes",
        "household_rows",
        "person_rows",
        "code_columns",
        "body_bytes",
        "source_receipt_sha256",
        "projection_sha256",
        "prepared_receipt_sha256",
        "frame_context_sha256",
        "frame_sha256",
        "definition_sha256",
        "implementation_sha256",
    }
)
_ARTIFACT_OUTPUTS = (
    ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
    ArtifactOutput("prepared_receipt", US_ACS_HOUSING_PREPARATION_TYPE),
    ArtifactOutput("housing_universe", US_ACS_HOUSING_UNIVERSE_TYPE),
)
_PREPARED_FIELDS = frozenset(
    {
        "format",
        "release_eligible",
        "source_receipt_sha256",
        "projection_sha256",
        "implementation_sha256",
        "same_snapshot_frame_and_observations",
        "full_source_frame_before_selection",
        "full_source_lexical_projection",
        "native_selection_before_person_accumulation",
        "selection_kind",
        "requested_serialnos",
        "full_source_inclusion_probability",
        "pre_promotion_frame_sha256",
        "frame_sha256",
        "dtype_transitions",
        "entity_rows",
        "weight_kind",
        "HU_columns",
    }
)


def _require(value, code):
    source._require(value, "GRAPH_ACS_" + code)


@dataclass(frozen=True)
class BoundACSHousingEvidence:
    """Immutable graph transport, explicitly not AuthenticatedACSHousingSource."""

    header_json: bytes
    source_receipt_json: bytes
    projection_json: bytes
    codes: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "BOUND_CONSTRUCTOR")


def _verify_preparation(prepared, receipt, household, frame_context):
    """Check the v2 construction claim without minting source authority."""
    _require(
        set(prepared) == _PREPARED_FIELDS
        and prepared["format"] == "microcosm.acs_housing_preparation.v2"
        and prepared["release_eligible"] is False
        and prepared["HU_columns"] == "artifact_only"
        and prepared["weight_kind"] == "design"
        and prepared["same_snapshot_frame_and_observations"] is True
        and prepared["full_source_lexical_projection"] is True
        and prepared["full_source_inclusion_probability"] is None,
        "PREPARED_SCHEMA",
    )
    requested = prepared["requested_serialnos"]
    subset = requested is not None
    _require(
        not subset
        or (
            type(requested) is list
            and bool(requested)
            and all(type(key) is str for key in requested)
            and len(set(requested)) == len(requested)
        ),
        "PREPARED_SELECTION",
    )
    chosen = tuple(sorted(household.SERIALNO))
    _require(
        prepared["full_source_frame_before_selection"] is (not subset)
        and prepared["native_selection_before_person_accumulation"] is subset
        and prepared["selection_kind"]
        == ("engineering_exact_keys" if subset else "all")
        and receipt["selection"] == ("exact_serialnos" if subset else "all")
        and (not subset or tuple(sorted(requested)) == chosen)
        and receipt["selected_serialnos_sha256"] == source._sha(canonical_json(chosen)),
        "PREPARED_SELECTION",
    )
    context = source._parse_json(frame_context, ACS_HU_PAYLOAD_MAX_BYTES)
    rows = prepared["entity_rows"]
    entities = context.get("entities")
    _require(
        type(rows) is dict
        and type(entities) is dict
        and set(rows) == set(entities) == set(US_SCHEMA.entities)
        and all(
            type(rows[entity]) is int
            and rows[entity] >= 0
            and type(entities[entity]) is dict
            and type(entities[entity].get("rows")) is int
            and rows[entity] == entities[entity]["rows"]
            for entity in US_SCHEMA.entities
        ),
        "PREPARED_CONTEXT",
    )


def encode_acs_housing_evidence(prepared):
    _require(
        type(prepared) is source.PreparedACSHousingPopulation
        and type(prepared.source) is source.AuthenticatedACSHousingSource,
        "PREPARED_TYPE",
    )
    receipt = source._parse_json(prepared.receipt_json, source.ACS_HU_RECEIPT_MAX_BYTES)
    observation = prepared.source
    projection = source._parse_json(
        observation.projection_json, ACS_HU_PAYLOAD_MAX_BYTES
    )
    _h, _p, _t, _lines, _pw, codes = source._tables(projection)
    source.verify_acs_frame_projection(prepared.frame, projection)
    _require(
        receipt["source_receipt_sha256"] == source._sha(observation.receipt_json)
        and receipt["projection_sha256"] == source._sha(observation.projection_json)
        and receipt["frame_sha256"] == source.frame_content_sha256(prepared.frame),
        "PREPARED_BINDING",
    )
    context = encode_us_frame_context(prepared.frame)
    _verify_preparation(
        receipt,
        source._parse_json(observation.receipt_json, source.ACS_HU_RECEIPT_MAX_BYTES),
        _h,
        context,
    )
    native = b"".join(codes[n].to_numpy(dtype="uint8").tobytes() for n in _CODES)
    body = observation.receipt_json + observation.projection_json + native
    header = {
        "format": "microcosm.acs_housing_graph_evidence.v2",
        "release_eligible": False,
        "source_receipt_bytes": len(observation.receipt_json),
        "projection_bytes": len(observation.projection_json),
        "household_rows": len(_h),
        "person_rows": len(_p),
        "code_columns": list(_CODES),
        "body_bytes": len(body),
        "source_receipt_sha256": source._sha(observation.receipt_json),
        "projection_sha256": source._sha(observation.projection_json),
        "prepared_receipt_sha256": source._sha(prepared.receipt_json),
        "frame_context_sha256": source._sha(context),
        "frame_sha256": receipt["frame_sha256"],
        "definition_sha256": source._definition()[1],
        "implementation_sha256": source._implementation(),
    }
    encoded = canonical_json(header)
    _require(len(encoded) <= ACS_HU_HEADER_MAX_BYTES, "HEADER_SIZE")
    payload = MAGIC + struct.pack("<I", len(encoded)) + encoded + body
    payload += bytes.fromhex(source._sha(payload))
    _require(len(payload) <= ACS_HU_PAYLOAD_MAX_BYTES, "PAYLOAD_SIZE")
    return payload


def bind_acs_housing_evidence(payload: bytes, *, prepared_receipt, frame_context):
    """Parse only bounded canonical transport under an explicit graph receipt."""
    try:
        _require(
            type(payload) is bytes
            and len(MAGIC) + 4 + 32 < len(payload) <= ACS_HU_PAYLOAD_MAX_BYTES,
            "PAYLOAD_SIZE",
        )
        _require(
            payload.startswith(MAGIC)
            and source._sha(payload[:-32]) == payload[-32:].hex(),
            "CHECKSUM",
        )
        size = struct.unpack_from("<I", payload, len(MAGIC))[0]
        start = len(MAGIC) + 4
        _require(
            0 < size <= ACS_HU_HEADER_MAX_BYTES and start + size < len(payload) - 32,
            "HEADER_SIZE",
        )
        raw_header = payload[start : start + size]
        header = source._parse_json(raw_header, ACS_HU_HEADER_MAX_BYTES)
        _require(
            set(header) == _HEADER_FIELDS
            and header["format"] == "microcosm.acs_housing_graph_evidence.v2"
            and header["release_eligible"] is False
            and header["code_columns"] == list(_CODES),
            "HEADER_SCHEMA",
        )
        for name in (
            "source_receipt_bytes",
            "projection_bytes",
            "household_rows",
            "person_rows",
            "body_bytes",
        ):
            _require(
                type(header[name]) is int
                and 0 <= header[name] <= ACS_HU_PAYLOAD_MAX_BYTES,
                "SECTION_SIZE",
            )
        nreceipt, nprojection, nhouse = (
            header[n]
            for n in ("source_receipt_bytes", "projection_bytes", "household_rows")
        )
        _require(
            0 < nreceipt <= source.ACS_HU_RECEIPT_MAX_BYTES
            and nprojection > 0
            and nhouse > 0,
            "SECTION_SIZE",
        )
        _require(
            header["body_bytes"] == nreceipt + nprojection + nhouse * len(_CODES)
            and len(payload) == start + size + header["body_bytes"] + 32,
            "BODY_SIZE",
        )
        cursor = start + size
        receipt_bytes = payload[cursor : cursor + nreceipt]
        projection_bytes = payload[cursor + nreceipt : cursor + nreceipt + nprojection]
        codes = payload[cursor + nreceipt + nprojection : -32]
        receipt = source._parse_json(receipt_bytes, source.ACS_HU_RECEIPT_MAX_BYTES)
        projection = source._parse_json(projection_bytes, ACS_HU_PAYLOAD_MAX_BYTES)
        prepared = source._parse_json(prepared_receipt, source.ACS_HU_RECEIPT_MAX_BYTES)
        household, person, _typed, _lines, _pw, derived = source._tables(projection)
        _require(
            source._sha(prepared_receipt) == header["prepared_receipt_sha256"]
            and source._sha(frame_context) == header["frame_context_sha256"],
            "CONTEXT_BINDING",
        )
        _verify_preparation(prepared, receipt, household, frame_context)
        _require(
            source._sha(receipt_bytes)
            == header["source_receipt_sha256"]
            == prepared["source_receipt_sha256"]
            and source._sha(projection_bytes)
            == header["projection_sha256"]
            == prepared["projection_sha256"]
            == receipt["projection_sha256"],
            "SOURCE_BINDING",
        )
        _require(
            header["definition_sha256"]
            == receipt["definition_sha256"]
            == source._definition()[1]
            and header["implementation_sha256"]
            == prepared["implementation_sha256"]
            == receipt["implementation_sha256"]
            == source._implementation(),
            "IMPLEMENTATION_BINDING",
        )
        _require(header["frame_sha256"] == prepared["frame_sha256"], "FRAME_BINDING")
        # Coverage of the pins is otherwise only transitive, through the
        # implementation hash over the archive manifest. Comparing them here
        # makes the evidence say what it rests on.
        _require(
            receipt["format"] == "microcosm.acs_housing_universe_source.v1"
            and receipt["release_eligible"] is False
            and receipt["vintage"] == 2024
            and receipt["archives"]
            == [
                {"role": r, "filename": n, "sha256": d, "bytes": s}
                for r, n, d, s in source._pins()
            ],
            "SOURCE_PINS",
        )
        _require(
            len(household) == nhouse == receipt["selected_counts"]["households"]
            and len(person)
            == header["person_rows"]
            == receipt["selected_counts"]["persons"],
            "ROW_COUNTS",
        )
        _require(
            codes
            == b"".join(
                derived[name].to_numpy(dtype="uint8").tobytes() for name in _CODES
            ),
            "NATIVE_CODES",
        )
        return BoundACSHousingEvidence(
            raw_header, receipt_bytes, projection_bytes, codes, _token=_TOKEN
        )
    except source.ACSHousingSourceError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        struct.error,
        AttributeError,
    ):
        raise source.ACSHousingSourceError("GRAPH_ACS_TRANSPORT_CONTRACT") from None


def _artifact(context, name, expected_type):
    value = context.artifacts.get(name)
    _require(value is not None and value.type == expected_type, "ARTIFACT_TYPE")
    declarations = [item for item in context.node.artifact_inputs if item.name == name]
    _require(
        len(declarations) == 1
        and value.key
        == opaque_artifact_key(value.producer_key, declarations[0].artifact),
        "ARTIFACT_KEY",
    )
    return value


def _keys(value):
    if value is None:
        return None
    _require(
        isinstance(value, (tuple, list))
        and bool(value)
        and all(type(v) is str for v in value)
        and len(set(value)) == len(value),
        "SELECTION_KEYS",
    )
    return tuple(value)


def _slices(columns):
    # The compiler does not treat implicit entity IDs as owned data columns.
    # Structural-only groups are reconstructed from declared person membership
    # and checked against the parent's typed ID digest, never read as fake cells.
    return tuple(
        Slice(entity, tuple(c.column for c in columns if c.entity == entity))
        for entity in US_SCHEMA.entities
        if any(c.entity == entity for c in columns)
    )


def _complete_structural_tables(context, frame_context):
    document = json.loads(frame_context)
    tables = dict(context.tables)
    _require("person" in tables, "PERSON_VIEW")
    for group in US_SCHEMA.group_entities:
        if group in tables:
            continue
        id_column = US_SCHEMA.entity_id_column(group)
        _require(
            document["entities"][group]["columns"] == [id_column],
            "UNDECLARED_GROUP_CELLS",
        )
        membership = tables["person"][US_SCHEMA.membership_column(group)]
        _require(membership.dtype == np.dtype("int64"), "MEMBERSHIP_DTYPE")
        tables[group] = pd.DataFrame({id_column: np.unique(membership.to_numpy())})
    # us_frame_from_context checks every reconstructed group's actual IDs
    # against the original producer's ordered-ID digest before Frame creation.
    return replace(context, tables=tables)


class _Kernel(KernelBase):
    def implementation_hash(self):
        return implementation_hash(source.ACS_HU_STAGE)


class ACSHousingCreateKernel(_Kernel):
    ref = "us.acs_housing.create@2"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        structural=StructuralDelta.CREATE,
        dependencies=STAGE_DEPENDENCIES[source.ACS_HU_STAGE],
    )

    def __init__(self, snapshot_root):
        self.snapshot_root = Path(snapshot_root)

    def run(self, context: KernelContext):
        node = context.node
        _require(
            node.kernel == self.ref
            and node.structural is StructuralDelta.CREATE
            and tuple(node.sources) == (SOURCE_NAME,)
            and not node.inputs
            and not node.artifact_inputs
            and node.artifact_outputs == _ARTIFACT_OUTPUTS,
            "CREATE_DECLARATION",
        )
        _require(
            set(context.params) == {"phase", "serialnos"}
            and context.params["phase"] == PHASE,
            "PARAMS",
        )
        prepared = source.prepare_acs_housing_population(
            context.sources[SOURCE_NAME],
            snapshot_root=self.snapshot_root,
            serialnos=_keys(context.params["serialnos"]),
        )
        _require(
            frame_column_declarations(prepared.frame) == node.outputs,
            "COLUMN_INVENTORY",
        )
        payload = encode_acs_housing_evidence(prepared)
        return KernelResult(
            frame=prepared.frame,
            artifacts={
                "frame_context": encode_us_frame_context(prepared.frame),
                "prepared_receipt": prepared.receipt_json,
                "housing_universe": payload,
            },
            receipt={
                "phase": PHASE,
                "prepared": prepared.receipt,
                "housing_universe_sha256": source._sha(payload),
                "implementation": implementation_manifest(source.ACS_HU_STAGE),
                "release_eligible": False,
            },
        )


class ACSHousingSelectKernel(_Kernel):
    ref = "us.acs_housing.select@2"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        structural=StructuralDelta.FILTER,
        dependencies=STAGE_DEPENDENCIES[source.ACS_HU_STAGE],
    )

    def run(self, context: KernelContext):
        node = context.node
        _require(
            node.kernel == self.ref
            and node.structural is StructuralDelta.FILTER
            and node.mass == "declared"
            and not node.outputs
            and not node.sources,
            "SELECT_DECLARATION",
        )
        _require(
            set(context.params) == {"phase", "serialnos"}
            and context.params["phase"] == PHASE,
            "PARAMS",
        )
        _require(
            set(context.artifacts)
            == {"frame_context", "prepared_receipt", "housing_universe"}
            and node.artifact_outputs
            == (
                ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
                ArtifactOutput("selection", US_ACS_HOUSING_SELECTION_TYPE),
            ),
            "SELECT_ARTIFACTS",
        )
        values = [
            _artifact(context, n, t)
            for n, t in (
                ("frame_context", US_FRAME_CONTEXT_TYPE),
                ("prepared_receipt", US_ACS_HOUSING_PREPARATION_TYPE),
                ("housing_universe", US_ACS_HOUSING_UNIVERSE_TYPE),
            )
        ]
        _require(
            len({v.producer_key for v in values}) == 1
            and len({a.producer for a in node.artifact_inputs}) == 1
            and next(iter(node.artifact_inputs)).producer == node.base,
            "SAME_PRODUCER",
        )
        bound = bind_acs_housing_evidence(
            values[2].payload,
            prepared_receipt=values[1].payload,
            frame_context=values[0].payload,
        )
        frame = us_frame_from_context(
            _complete_structural_tables(context, values[0].payload)
        )
        _require(
            tuple(node.inputs) == _slices(frame_column_declarations(frame)),
            "COMPLETE_FRAME_DECLARATION",
        )
        header = json.loads(bound.header_json)
        _require(
            source.frame_content_sha256(frame) == header["frame_sha256"], "FRAME_CELLS"
        )
        projection = json.loads(bound.projection_json)
        source.verify_acs_frame_projection(frame, projection)
        selected_projection, chosen, _counts = source._select(
            projection, _keys(context.params["serialnos"])
        )
        household = frame.table("household")
        household_keep = household.SERIALNO.isin(chosen).to_numpy()
        ids = household.household_id.to_numpy()[household_keep]
        keep = frame.person.person_household_id.isin(ids)
        _require(bool(keep.any()), "EMPTY_SELECTION")
        selected = frame.select(keep)
        source.verify_acs_frame_projection(selected, selected_projection)
        positions = {
            "person": np.flatnonzero(keep).astype("int64"),
            "household": np.flatnonzero(household_keep).astype("int64"),
        }
        _require(
            selected.weights_for("household").values.tobytes()
            == frame.weights_for("household").values[positions["household"]].tobytes(),
            "SELECTED_WEIGHT_BYTES",
        )
        before_mass, after_mass = frame.stratum_mass(), selected.stratum_mass()
        native_households = dict(
            zip(
                selected.table("household").household_id,
                selected.table("household").SERIALNO,
                strict=True,
            )
        )
        selection = {
            "format": "microcosm.acs_housing_selection.v2",
            "release_eligible": False,
            "representative": False,
            "source_authority": "original_CREATE_only",
            "producer_key": values[0].producer_key,
            "source_payload_sha256": source._sha(values[2].payload),
            "source_prepared_receipt_sha256": source._sha(values[1].payload),
            "source_frame_context_sha256": source._sha(values[0].payload),
            "selected_frame_sha256": source.frame_content_sha256(selected),
            "selected_frame_context_sha256": source._sha(
                encode_us_frame_context(selected)
            ),
            "selected_projection_sha256": source._sha(
                canonical_json(selected_projection)
            ),
            "selected_serialnos_sha256": source._sha(canonical_json(chosen)),
            "positions": {e: a.tolist() for e, a in positions.items()},
            "frame_identity_scope": "actual_CREATE_frame",
            "native_keys": {
                "household": selected.table("household").SERIALNO.tolist(),
                "person": [
                    [native_households[household_id], int(line)]
                    for household_id, line in selected.person[
                        ["person_household_id", "SPORDER"]
                    ].itertuples(index=False, name=None)
                ],
            },
            "ordered_ids": {
                e: selected.table(e)[US_SCHEMA.entity_id_column(e)].tolist()
                for e in US_SCHEMA.entities
            },
            "weight_kind": "design",
            "mass_policy": "declared_no_normalization",
        }
        return KernelResult(
            keep=pd.Series(
                keep.to_numpy(dtype=bool),
                index=pd.Index(frame.person.person_id.to_numpy(), name="person_id"),
                dtype=bool,
            ),
            artifacts={
                "frame_context": encode_us_frame_context(selected),
                "selection": canonical_json(selection),
            },
            receipt={
                "phase": PHASE,
                "implementation": implementation_manifest(source.ACS_HU_STAGE),
                "selection_sha256": source._sha(canonical_json(selection)),
                "release_eligible": False,
                "mass": {
                    "policy": "declared",
                    "before": float(before_mass.sum()),
                    "after": float(after_mass.sum()),
                    "stratum_before": {
                        str(k): float(v) for k, v in before_mass.items()
                    },
                    "stratum_after": {str(k): float(v) for k, v in after_mass.items()},
                },
            },
        )


def acs_housing_graph(columns, *, serialnos=None, selected_serialnos=None):
    """Explicit seed source selection followed by exact further household pruning."""
    columns = tuple(columns)
    _keys(serialnos)
    _keys(selected_serialnos)
    create = Node(
        id=CREATE_NODE,
        kernel=ACSHousingCreateKernel.ref,
        structural=StructuralDelta.CREATE,
        sources=(SOURCE_NAME,),
        outputs=columns,
        params={"phase": PHASE, "serialnos": serialnos},
        artifact_outputs=_ARTIFACT_OUTPUTS,
    )
    select = Node(
        id=SELECT_NODE,
        kernel=ACSHousingSelectKernel.ref,
        structural=StructuralDelta.FILTER,
        mass="declared",
        base=CREATE_NODE,
        inputs=_slices(columns),
        params={"phase": PHASE, "serialnos": selected_serialnos},
        artifact_inputs=tuple(
            ArtifactInput(a.name, CREATE_NODE, a.name, a.type)
            for a in _ARTIFACT_OUTPUTS
        ),
        artifact_outputs=(
            ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
            ArtifactOutput("selection", US_ACS_HOUSING_SELECTION_TYPE),
        ),
    )
    return Graph(
        country="us",
        sources=(SourceRef(SOURCE_NAME, codec=source.ACS_HU_CODEC),),
        nodes=(create, select),
    )


def acs_housing_registry(*, snapshot_root):
    registry = KernelRegistry()
    registry.register(ACSHousingCreateKernel(snapshot_root))
    registry.register(ACSHousingSelectKernel())
    return registry
