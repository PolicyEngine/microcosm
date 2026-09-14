"""Compose the prepared ASEC population with the native ACS population.

This is a separately versioned seam. The reviewed ``asec_raw_stage`` assembly
kernel in :mod:`.graph_sources` keeps its kernel ref, contract, params, sources
and node ids unchanged; nothing here rewrites them. What changes is only which
authenticated ASEC source the ASEC arm reads: the reviewed prepared
current-money directory (``us-asec-prepared-current-money-v3``) instead of the
raw-stage checkpoint.

Node *keys* are a different matter and do move. Every stage manifest binds the
inventory and the bytes of ``graph_implementation.py``, which is in every
stage's module list, and declaring a new stage necessarily edits both. So every
existing US node key changes and every existing US cache is invalidated —
exactly as when any other attested module changes.

Why this is a two-source ``CREATE`` and not two parent populations
------------------------------------------------------------------
``Node.base`` is a single population version. A ``CREATE`` node may declare no
base at all, a structural node may not declare a second ``population``, and a
tuple ``base`` is refused as "not a structural node". So the executor cannot
structurally combine two parent populations; an ``EXPAND`` over one arm can
only invent entrant rows, never bind the other arm's rows or lineage. The
composed population is therefore one ``CREATE`` over two declared sources —
exactly the shape :class:`~.graph_sources.USAssemblyCreateKernel` already uses.
The refusals are recorded in the review root's two-parent probe.

The stack itself is not reimplemented. Sampling, stacking, provenance and
allocation stay in :mod:`.stacked_spine` / :mod:`.spine_assembly`, and the
harmonization node reuses :class:`~.graph_sources.USSpineHarmonizeKernel`
unchanged by emitting the same two typed producer artifacts it declares.

Nothing here is a release, a calibration, a certified population, a transfer or
a score. It is an engineering intermediate whose receipts say so.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.build.serialization_dtypes import (
    CANONICAL_STRING_DTYPE,
    canonicalize_frame_string_dtypes,
)
from microcosm.frame import US_SCHEMA, Frame
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
    SeedSource,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
)
from microcosm.graph.canonical import canonical_json

from .asec_current_money import _sha
from .asec_current_money_selection import (
    US_ASEC_CURRENT_MONEY_BODY_TYPE,
    US_ASEC_PREPARED_RECEIPT_TYPE,
)
from .asec_prepared_source import (
    PREPARED_SOURCE_FILES,
    PREPARED_SOURCE_KIND,
    prepare_asec_current_money_population,
)
from .graph_asec_income import US_ASEC_INCOME_OBSERVATIONS_TYPE
from .graph_asec_prepared import (
    ASEC_PREPARED_CODEC,
    ASEC_PREPARED_SOURCE_NAME,
    PreparedGraphError,
)
from .graph_composed_contracts import CREATE_NODE as CREATE_NODE
from .graph_context import US_FRAME_CONTEXT_TYPE, _json_data, encode_us_frame_context
from .graph_geography import (
    GEOGRAPHY_PHASE,
    LOOKUP_SOURCES,
    register_us_geography_kernels,
    us_geography_nodes,
)
from .graph_housing_universe import US_ASEC_HOUSING_UNIVERSE_TYPE
from .graph_implementation import (
    STAGE_DEPENDENCIES,
    implementation_hash,
    implementation_manifest,
)
from .graph_sources import (
    ACS_CODEC,
    ASSEMBLY_PHASE,
    US_STACK_PREPARATION_TYPE,
    USSpineHarmonizeKernel,
    _canonical_assembly,
    frame_column_declarations,
    load_graph_acs,
    us_source_codecs,
)
from .stacked_spine import ACS_STACKED_SUPPORT_CHANNEL, prepare_stacked_spine
from .support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    spine_source_id_column,
    support_channel_column,
    support_source_id_column,
)

COMPOSED_STAGE = "composed_population_v1"
COMPOSED_PHASE = "compose_prepared_asec_native_acs"
ACS_NATIVE_SOURCE_NAME = "acs_native"
COMPOSED_DEPENDENCIES = STAGE_DEPENDENCIES[COMPOSED_STAGE]

PREFIX = "composed_population"
HARMONIZE_NODE = PREFIX

US_COMPOSED_SOURCE_ORIGIN_TYPE = ArtifactType(
    "microcosm.us.composed_population_source_origin", 1
)

COMPOSED_SOURCES = (
    SourceRef(
        ASEC_PREPARED_SOURCE_NAME,
        ASEC_PREPARED_CODEC,
        "Reviewed ASEC restoration inputs: P, H, T with its receipt, three "
        "housing cohort HDFs and three official Census PERSON CSV members.",
    ),
    SourceRef(
        ACS_NATIVE_SOURCE_NAME,
        ACS_CODEC,
        "2024 ACS one-year national PUMS archives.",
    ),
)

#: Native source-record identity carried by each arm, on both entities. These
#: are the columns a later node must be able to bind back to an actual archive
#: member, so the origin artifact digests them in addition to — never instead of
#: — the arm's own pre-remap IDs. Every declared column must be present: a
#: silently skipped one would leave an arm unbindable while still receipting.
NATIVE_IDENTITY_COLUMNS = {
    BASE_ASEC_SUPPORT_CHANNEL: (
        ("household", "asec_H_SEQ"),
        ("person", "PERIDNUM"),
        ("person", "PH_SEQ"),
    ),
    ACS_STACKED_SUPPORT_CHANNEL: (
        ("household", "SERIALNO"),
        ("person", "source_household_id"),
        ("person", "source_person_id"),
    ),
}
#: The per-arm cohort/vintage axis. ASEC restores three income years into one
#: prepared population, so a single scalar vintage would misdescribe it.
COHORT_COLUMN = "source_year"

_SOURCE_ORIGIN_SCHEMA = "microcosm.us.composed-source-origin.v1"


@dataclass(frozen=True)
class ComposedPopulationResult:
    """A direct-call composition and everything its CREATE kernel emits."""

    frame: Frame
    preparation: object
    dtype_transitions: tuple[dict[str, str], ...]
    storage_bridge: tuple[dict[str, str], ...]
    source_origin: dict[str, object]
    asec_frame_context: bytes
    money_payload: bytes
    receipt_payload: bytes
    housing_universe_payload: bytes
    income_observations_payload: bytes
    prepared_receipt: Mapping[str, object]


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PreparedGraphError(reason)


def bridge_prepared_string_storage(frame: Frame) -> tuple[dict[str, str], ...]:
    """Put the prepared arm on the pre-assembly string storage, in place.

    The one explicit schema difference between the two arms. The prepared
    source hands back graph column storage (``string[python, pd.NA]``) because
    its own CREATE writes graph cells directly; every pre-assembly build
    boundary, and therefore the native ACS arm, uses ``CANONICAL_STRING_DTYPE``
    (``string[python, nan]``). ``_stack_spine_tables`` requires shared columns
    to have identical dtypes, so one arm has to move. Moving the prepared arm
    down leaves the ACS arm byte-identical to today's production assembly, and
    :func:`~.graph_sources._canonical_assembly` promotes the assembled frame
    back to graph storage afterwards under its own null/value equality checks.

    This is a physical storage bridge only: no value, no missingness and no
    column is added, dropped or filled.
    """
    transitions = []
    for entity in US_SCHEMA.entities:
        table = frame.table(entity)
        for column in table.columns:
            dtype = table[column].dtype
            if isinstance(dtype, pd.StringDtype) and dtype != CANONICAL_STRING_DTYPE:
                transitions.append(
                    {
                        "entity": entity,
                        "column": column,
                        "from": str(dtype),
                        "to": str(CANONICAL_STRING_DTYPE),
                    }
                )
    before = {
        (item["entity"], item["column"]): frame.table(item["entity"])[item["column"]]
        .isna()
        .to_numpy(copy=True)
        for item in transitions
    }
    canonicalize_frame_string_dtypes(
        frame, boundary="US composed population source bridge", in_place=True
    )
    for (entity, column), missing in before.items():
        actual = frame.table(entity)[column]
        _require(
            actual.dtype == CANONICAL_STRING_DTYPE
            and np.array_equal(actual.isna().to_numpy(), missing),
            f"STRING_BRIDGE_MISSINGNESS:{entity}.{column}",
        )
    return tuple(transitions)


def _arm_origin(source: Frame, channel: str) -> dict[str, object]:
    """Digest one arm's native identity and cohort axis before assembly."""
    identity = {}
    for entity, column in NATIVE_IDENTITY_COLUMNS[channel]:
        table = source.table(entity)
        _require(column in table, f"ARM_NATIVE_IDENTITY:{channel}.{entity}.{column}")
        series = table[column]
        identity[f"{entity}.{column}"] = {
            "dtype": str(series.dtype),
            "rows": int(len(series)),
            "missing": int(series.isna().sum()),
            "ordered_sha256": _sha(
                canonical_json(
                    [
                        None if value is None or value is pd.NA else str(value)
                        for value in series.tolist()
                    ]
                )
            ),
        }
    person = source.table("person")
    _require(COHORT_COLUMN in person, f"ARM_COHORT_COLUMN:{channel}")
    years = person[COHORT_COLUMN]
    counts = years.value_counts(dropna=False).to_dict()
    cohorts = {
        ("missing" if pd.isna(year) else str(int(year))): int(count)
        for year, count in counts.items()
    }
    cohorts = {key: cohorts[key] for key in sorted(cohorts)}
    return {
        "channel": channel,
        "rows": {entity: int(source.n(entity)) for entity in source.entities},
        "household_mass": float(source.weights_for("household").total),
        "weight_kind": source.weights_for("household").kind.value,
        "native_identity": identity,
        "person_cohorts": cohorts,
        "columns": {
            entity: sorted(source.table(entity).columns) for entity in source.entities
        },
    }


def _source_origin(
    frame: Frame,
    sources: Mapping[str, Frame],
    *,
    sampling: Mapping[str, object],
    preparation_sha256: str,
    prepared_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Bind each output row to the arm and the arm's own pre-remap source ID.

    The mapping is read back off the assembled frame's own receipt-validated
    support-provenance columns, so it records what assembly actually did rather
    than restating what the caller intended. Both provenance IDs are digested:
    ``*_spine_source_id`` is the arm's raw pre-remap identifier and is what
    binds an output row back to its source row, while ``*_source_id`` is the
    assembly-unique pre-clone identifier the clone stage will later offset. At
    CREATE the latter equals the entity ID, so digesting it alone would bind
    nothing.
    """
    mapping: dict[str, object] = {}
    for entity in US_SCHEMA.entities:
        table = frame.table(entity)
        channel_column = support_channel_column(entity)
        spine_column = spine_source_id_column(entity)
        source_column = support_source_id_column(entity)
        _require(channel_column in table, f"ORIGIN_CHANNEL_MISSING:{entity}")
        _require(spine_column in table, f"ORIGIN_SPINE_SOURCE_ID_MISSING:{entity}")
        _require(source_column in table, f"ORIGIN_SOURCE_ID_MISSING:{entity}")
        channels = table[channel_column].astype(str).tolist()
        spine_ids = table[spine_column].to_numpy(dtype="int64").tolist()
        ids = table[source_column].to_numpy(dtype="int64").tolist()
        outputs = (
            table[US_SCHEMA.entity_id_column(entity)].to_numpy(dtype="int64").tolist()
        )
        mapping[entity] = {
            "rows_by_channel": {
                channel: int(channels.count(channel))
                for channel in sorted(set(channels))
            },
            "ordered_channels_sha256": _sha(canonical_json(channels)),
            "ordered_spine_source_ids_sha256": _sha(canonical_json(spine_ids)),
            "ordered_source_to_output_sha256": _sha(
                canonical_json(
                    [
                        [channel, spine_id, source_id, output_id]
                        for channel, spine_id, source_id, output_id in zip(
                            channels, spine_ids, ids, outputs, strict=True
                        )
                    ]
                )
            ),
        }
    asec = sources[BASE_ASEC_SUPPORT_CHANNEL]
    composed_asec = {
        entity: int(
            (
                frame.table(entity)[support_channel_column(entity)]
                == BASE_ASEC_SUPPORT_CHANNEL
            ).sum()
        )
        for entity in US_SCHEMA.entities
    }
    return {
        "schema": _SOURCE_ORIGIN_SCHEMA,
        "phase": COMPOSED_PHASE,
        "preparation_sha256": preparation_sha256,
        "sampling": {
            "sample_fraction": float(sampling["sample_fraction"]),
            "sample_seed": int(sampling["sample_seed"]),
            "survey_samples": {
                channel: {
                    key: sample[key]
                    for key in (
                        "realized_household_count",
                        "selected_household_ids_sha256",
                        "incoming_household_mass",
                        "sampled_household_mass",
                    )
                }
                for channel, sample in sampling["survey_samples"].items()
            },
        },
        # The four carried prepared artifacts describe the whole prepared ASEC
        # population, not the composed rows. They coincide only when the draw
        # keeps the arm entire. Recording both counts makes the difference
        # readable instead of leaving a later consumer to assume alignment.
        "asec_evidence_alignment": {
            "basis": "full_prepared_population",
            "evidence_rows": {
                entity: int(count)
                for entity, count in sorted(prepared_receipt["entity_rows"].items())
            },
            "arm_rows": {entity: int(asec.n(entity)) for entity in asec.entities},
            "composed_rows": composed_asec,
            "composed_is_whole_arm": all(
                composed_asec[entity] == int(asec.n(entity))
                for entity in US_SCHEMA.entities
            ),
        },
        "arms": {
            channel: _arm_origin(source, channel)
            for channel, source in sorted(sources.items())
        },
        "assembled": mapping,
        "release_eligible": False,
    }


def compose_from_sources(
    asec_prepared_path,
    acs_path,
    *,
    sample_fraction: float,
    sample_seed: int,
) -> ComposedPopulationResult:
    """The direct-call parity oracle the CREATE kernel shares.

    Source preparation is deliberately separate from sampling: the prepared
    ASEC directory and the ACS archives are read whole, and only then does
    :func:`prepare_stacked_spine` draw its declared seeded sample. Reading a
    fraction of a source is not what this does and would not cost a fraction
    of a whole-source read.
    """
    prepared = prepare_asec_current_money_population(asec_prepared_path)
    asec = prepared.frame
    # The carried evidence is positional over the whole prepared population.
    # If the receipt and the arm ever disagreed, everything downstream that
    # reads the evidence by position would be silently misaligned.
    _require(
        {
            entity: int(count)
            for entity, count in prepared.receipt["entity_rows"].items()
        }
        == {entity: int(asec.n(entity)) for entity in prepared.receipt["entity_rows"]},
        "PREPARED_EVIDENCE_ROWS",
    )
    acs = load_graph_acs(acs_path)
    storage_bridge = bridge_prepared_string_storage(asec)
    # Encoded before stacking, so the identity it declares is the prepared
    # arm's own ordered identity, which is what the evidence is positional in.
    asec_context = encode_us_frame_context(asec)
    arms = {
        BASE_ASEC_SUPPORT_CHANNEL: asec,
        ACS_STACKED_SUPPORT_CHANNEL: acs,
    }
    stacked = prepare_stacked_spine(
        asec, acs, sample_fraction=sample_fraction, sample_seed=sample_seed
    )
    transitions = _canonical_assembly(stacked.frame, (asec, acs))
    preparation_payload = canonical_json(_json_data(stacked.receipt))
    return ComposedPopulationResult(
        frame=stacked.frame,
        preparation=stacked.receipt,
        dtype_transitions=transitions,
        storage_bridge=storage_bridge,
        source_origin=_source_origin(
            stacked.frame,
            arms,
            sampling=stacked.receipt["sampling"],
            preparation_sha256=_sha(preparation_payload),
            prepared_receipt=prepared.receipt,
        ),
        asec_frame_context=asec_context,
        money_payload=prepared.money_payload,
        receipt_payload=prepared.receipt_payload,
        housing_universe_payload=prepared.housing_universe_payload,
        income_observations_payload=prepared.income_observations_payload,
        prepared_receipt=prepared.receipt,
    )


class USComposedPopulationCreateKernel(KernelBase):
    """Read both authenticated sources whole and stack them into one spine."""

    ref = "us.composed_population.prepare@1"
    capabilities = Capabilities(
        determinism=Determinism.SEEDED,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.PARAM,
        structural=StructuralDelta.CREATE,
        dependencies=COMPOSED_DEPENDENCIES,
    )

    def implementation_hash(self) -> str:
        # Runs before cache lookup, including resume=require, exactly as the
        # raw-stage assembly kernel does: a renamed or substituted public
        # loader cannot claim the declared codec implementation.
        us_source_codecs()
        return implementation_hash(COMPOSED_STAGE)

    def run(self, context: KernelContext) -> KernelResult:
        node = context.node
        _require(node.kernel == self.ref, "NODE_KERNEL")
        _require(
            set(context.params) == {"phase", "sample_fraction", "sample_seed"},
            "NODE_PARAMS",
        )
        _require(context.params["phase"] == COMPOSED_PHASE, "NODE_PHASE")
        _require(
            tuple(node.sources) == (ASEC_PREPARED_SOURCE_NAME, ACS_NATIVE_SOURCE_NAME),
            "NODE_SOURCES",
        )
        _require(not node.inputs and not node.artifact_inputs, "NODE_INPUTS")
        _require(node.artifact_outputs == _CREATE_ARTIFACTS, "NODE_ARTIFACT_OUTPUTS")
        result = compose_from_sources(
            context.sources[ASEC_PREPARED_SOURCE_NAME],
            context.sources[ACS_NATIVE_SOURCE_NAME],
            sample_fraction=context.params["sample_fraction"],
            sample_seed=context.params["sample_seed"],
        )
        receipt = result.prepared_receipt
        _require(receipt["source_kind"] == PREPARED_SOURCE_KIND, "PREPARED_SOURCE_KIND")
        _require(
            receipt["file_roster"] == list(PREPARED_SOURCE_FILES),
            "PREPARED_FILE_ROSTER",
        )
        _require(
            frame_column_declarations(result.frame) == node.outputs,
            "COMPOSED_COLUMN_INVENTORY",
        )
        return KernelResult(
            frame=result.frame,
            artifacts={
                "frame_context": encode_us_frame_context(result.frame),
                "preparation": canonical_json(_json_data(result.preparation)),
                "source_origin": canonical_json(result.source_origin),
                "asec_frame_context": result.asec_frame_context,
                "current_money": result.money_payload,
                "prepared_receipt": result.receipt_payload,
                "housing_universe": result.housing_universe_payload,
                "income_observations": result.income_observations_payload,
            },
            receipt={
                "phase": COMPOSED_PHASE,
                "implementation": implementation_manifest(COMPOSED_STAGE),
                "preparation": result.preparation,
                "prepared": receipt,
                "dtype_transitions": result.dtype_transitions,
                "storage_bridge": result.storage_bridge,
                "release_eligible": False,
                "certified": False,
            },
        )


_CREATE_ARTIFACTS = (
    ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
    ArtifactOutput("preparation", US_STACK_PREPARATION_TYPE),
    ArtifactOutput("source_origin", US_COMPOSED_SOURCE_ORIGIN_TYPE),
    # The prepared arm's own typed context, encoded before stacking. The four
    # evidence artifacts below are positional over that population, so a later
    # binding node needs its ordered identity to align them to composed rows.
    ArtifactOutput("asec_frame_context", US_FRAME_CONTEXT_TYPE),
    ArtifactOutput("current_money", US_ASEC_CURRENT_MONEY_BODY_TYPE),
    ArtifactOutput("prepared_receipt", US_ASEC_PREPARED_RECEIPT_TYPE),
    ArtifactOutput("housing_universe", US_ASEC_HOUSING_UNIVERSE_TYPE),
    ArtifactOutput("income_observations", US_ASEC_INCOME_OBSERVATIONS_TYPE),
)


#: Top-level keys the origin document must carry, exactly.
_ORIGIN_KEYS = frozenset(
    {
        "schema",
        "phase",
        "preparation_sha256",
        "sampling",
        "asec_evidence_alignment",
        "arms",
        "assembled",
        "release_eligible",
    }
)
_ORIGIN_ENTITY_KEYS = frozenset(
    {
        "rows_by_channel",
        "ordered_channels_sha256",
        "ordered_spine_source_ids_sha256",
        "ordered_source_to_output_sha256",
    }
)


def bind_composed_source_origin(payload: bytes) -> dict[str, object]:
    """Decode the origin artifact as a typed shape, refusing a malformed one.

    This checks canonical bytes and the document's declared shape: its exact
    key set, its schema/phase, that both arms and every US entity are present
    with well-formed digests and counts, and that it makes no release claim. It
    verifies **no** digest against any population — it cannot, since it receives
    only bytes. A consumer that needs the mapping to be true of a frame in hand
    must recompute the digests against that frame.
    """
    document = json.loads(payload)
    _require(canonical_json(document) == payload, "ORIGIN_CANONICAL")
    _require(isinstance(document, dict), "ORIGIN_DOCUMENT")
    _require(set(document) == _ORIGIN_KEYS, "ORIGIN_KEYS")
    _require(document["schema"] == _SOURCE_ORIGIN_SCHEMA, "ORIGIN_SCHEMA")
    _require(document["phase"] == COMPOSED_PHASE, "ORIGIN_PHASE")
    _require(_is_digest(document["preparation_sha256"]), "ORIGIN_PREPARATION_DIGEST")
    _require(document["release_eligible"] is False, "ORIGIN_RELEASE_CLAIM")
    channels = {BASE_ASEC_SUPPORT_CHANNEL, ACS_STACKED_SUPPORT_CHANNEL}
    _require(set(document["arms"]) == channels, "ORIGIN_ARMS")
    sampling = document["sampling"]
    _require(
        set(sampling) == {"sample_fraction", "sample_seed", "survey_samples"}
        and set(sampling["survey_samples"]) == channels,
        "ORIGIN_SAMPLING",
    )
    alignment = document["asec_evidence_alignment"]
    _require(
        set(alignment)
        == {
            "basis",
            "evidence_rows",
            "arm_rows",
            "composed_rows",
            "composed_is_whole_arm",
        }
        and alignment["basis"] == "full_prepared_population"
        and isinstance(alignment["composed_is_whole_arm"], bool),
        "ORIGIN_ALIGNMENT",
    )
    _require(set(document["assembled"]) == set(US_SCHEMA.entities), "ORIGIN_ENTITIES")
    for entity, record in document["assembled"].items():
        _require(set(record) == _ORIGIN_ENTITY_KEYS, f"ORIGIN_ENTITY_KEYS:{entity}")
        _require(
            all(
                _is_digest(record[key])
                for key in _ORIGIN_ENTITY_KEYS
                if key.endswith("_sha256")
            ),
            f"ORIGIN_ENTITY_DIGEST:{entity}",
        )
        _require(
            set(record["rows_by_channel"]) <= channels
            and all(
                isinstance(count, int) and count >= 0
                for count in record["rows_by_channel"].values()
            ),
            f"ORIGIN_ENTITY_ROWS:{entity}",
        )
    return document


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not set(value) - set("0123456789abcdef")
    )


def composed_population_nodes(
    columns: Sequence[Owned], *, sample_fraction: float, sample_seed: int
) -> tuple[Node, ...]:
    """The two composition nodes: whole-source CREATE, then reused harmonize."""
    inventory = {(owned.entity, owned.column) for owned in columns}
    _require(len(inventory) == len(tuple(columns)), "COLUMN_INVENTORY_REPEATS")
    for entity in US_SCHEMA.entities:
        _require(
            (entity, support_channel_column(entity)) in inventory,
            f"MISSING_SUPPORT_CHANNEL:{entity}",
        )
        _require(
            (entity, support_source_id_column(entity)) in inventory,
            f"MISSING_SUPPORT_SOURCE_ID:{entity}",
        )
    return (
        Node(
            id=CREATE_NODE,
            kernel=USComposedPopulationCreateKernel.ref,
            structural=StructuralDelta.CREATE,
            sources=(ASEC_PREPARED_SOURCE_NAME, ACS_NATIVE_SOURCE_NAME),
            outputs=tuple(columns),
            params={
                "phase": COMPOSED_PHASE,
                "sample_fraction": sample_fraction,
                "sample_seed": sample_seed,
            },
            artifact_outputs=_CREATE_ARTIFACTS,
        ),
        Node(
            id=HARMONIZE_NODE,
            kernel=USSpineHarmonizeKernel.ref,
            base=CREATE_NODE,
            structural=StructuralDelta.REWEIGHT,
            inputs=(
                Slice("household", (support_channel_column("household"),)),
                Slice("person", (support_channel_column("person"),)),
            ),
            weights=WeightTransition("household", "importance", mass="declared"),
            mass="declared",
            # The reused production harmonization kernel owns this phase name;
            # composing a different ASEC source does not make its operation a
            # different one, and its contract is deliberately not re-versioned.
            params={"phase": ASSEMBLY_PHASE},
            artifact_inputs=(
                ArtifactInput(
                    "frame_context", CREATE_NODE, "frame_context", US_FRAME_CONTEXT_TYPE
                ),
                ArtifactInput(
                    "preparation",
                    CREATE_NODE,
                    "preparation",
                    US_STACK_PREPARATION_TYPE,
                ),
            ),
            artifact_outputs=(ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),),
        ),
    )


def composed_population_graph(
    columns: Sequence[Owned],
    *,
    sample_fraction: float,
    sample_seed: int,
    geography: bool = True,
) -> Graph:
    """Declare the composed development population, never a release pool."""
    nodes = composed_population_nodes(
        columns, sample_fraction=sample_fraction, sample_seed=sample_seed
    )
    if not geography:
        return Graph(country="us", sources=COMPOSED_SOURCES, nodes=nodes)
    return Graph(
        country="us",
        sources=(*COMPOSED_SOURCES, *LOOKUP_SOURCES),
        nodes=(
            *nodes,
            *us_geography_nodes(
                columns, base=HARMONIZE_NODE, context_producer=HARMONIZE_NODE
            ),
        ),
    )


def composed_population_registry(*, geography: bool = True) -> KernelRegistry:
    """Register exactly the kernels this graph runs; two of three are reused."""
    registry = KernelRegistry()
    registry.register(USComposedPopulationCreateKernel())
    registry.register(USSpineHarmonizeKernel())
    if geography:
        register_us_geography_kernels(registry)
    return registry


__all__ = [
    "ACS_NATIVE_SOURCE_NAME",
    "COHORT_COLUMN",
    "COMPOSED_DEPENDENCIES",
    "COMPOSED_PHASE",
    "COMPOSED_SOURCES",
    "COMPOSED_STAGE",
    "CREATE_NODE",
    "GEOGRAPHY_PHASE",
    "HARMONIZE_NODE",
    "NATIVE_IDENTITY_COLUMNS",
    "US_COMPOSED_SOURCE_ORIGIN_TYPE",
    "ComposedPopulationResult",
    "USComposedPopulationCreateKernel",
    "bind_composed_source_origin",
    "bridge_prepared_string_storage",
    "compose_from_sources",
    "composed_population_graph",
    "composed_population_nodes",
    "composed_population_registry",
]
