"""Graph source producers and population-grain native household lineage.

Only source kernels open original members. Consumers preserve the executor's
typed producer edges when decoding transport; a nominal artifact type by itself
is not source authentication. These nodes issue no benchmark approval or score.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from microcosm.frame import US_SCHEMA, Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    SourceRef,
)
from microcosm.graph.artifact_edges import value_from_descriptor
from microcosm.graph.canonical import canonical_json
from microcosm.graph.keys import opaque_artifact_key

from . import native_household_origin as native
from .graph_native_origin_implementation import (
    DEPENDENCIES,
    implementation_hash,
    implementation_manifest,
)

SOURCE_TYPE = ArtifactType("microcosm.us.native_household_origins", 1)
BINDING_TYPE = ArtifactType("microcosm.us.population_household_origins", 1)
SOURCE_OUTPUT = (ArtifactOutput("origins", SOURCE_TYPE),)
BINDING_OUTPUT = (ArtifactOutput("binding", BINDING_TYPE),)
PREFIX = "native_household_origins"
ACS_NODE = PREFIX + ".acs"
ASEC_NODE = PREFIX + ".asec"
ACS_SOURCE = "native_origin_acs"
ASEC_SOURCES = tuple(f"native_origin_asec_{year}" for year in (2022, 2023, 2024))
SOURCES = (
    SourceRef(ACS_SOURCE, native.acs.ACS_HU_CODEC),
    *(SourceRef(name, "raw-bytes-v1") for name in ASEC_SOURCES),
)


class _Kernel(KernelBase):
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=DEPENDENCIES,
    )

    def implementation_hash(self):
        return implementation_hash()


class NativeACSOriginKernel(_Kernel):
    """Authenticate original ACS archive members through the accepted codec."""

    ref = "us.native_household_origin.acs@1"

    def run(self, context):
        native._require(
            tuple(context.node.sources) == (ACS_SOURCE,)
            and not context.node.inputs
            and not context.node.artifact_inputs
            and context.node.artifact_outputs == SOURCE_OUTPUT
            and not context.params,
            "ACS_ORIGIN_DECLARATION",
        )
        with tempfile.TemporaryDirectory(prefix="graph-native-acs-") as tmp:
            root = Path(tmp).resolve()
            snapshots = root / "snapshots"
            snapshots.mkdir()
            source = native.produce_acs_native_origins(
                context.sources[ACS_SOURCE],
                snapshot_root=snapshots,
                output_dir=root / "projection",
            )
        return _source_result(source)


class NativeASECOriginKernel(_Kernel):
    """Authenticate the registered original HOUSEHOLD CSV projections."""

    ref = "us.native_household_origin.asec@1"

    def run(self, context):
        native._require(
            tuple(context.node.sources) == ASEC_SOURCES
            and not context.node.inputs
            and not context.node.artifact_inputs
            and context.node.artifact_outputs == SOURCE_OUTPUT
            and not context.params,
            "ASEC_ORIGIN_DECLARATION",
        )
        source = native.produce_asec_native_origins(
            {
                year: context.sources[name]
                for year, name in zip((2022, 2023, 2024), ASEC_SOURCES, strict=True)
            }
        )
        return _source_result(source)


def _source_result(source):
    document = source.document
    return KernelResult(
        artifacts={"origins": source.payload},
        receipt={
            "phase": "native_household_origin_source",
            "arm": document["arm"],
            "households": len(document["records"]),
            "source_projection_sha256": native._sha(source.payload),
            "source_receipt_sha256": document["source_receipt_sha256"],
            "implementation": implementation_manifest(),
            "release_eligible": False,
        },
    )


def _artifact(context, name, expected_type, output):
    value = context.artifacts[name]
    native._require(
        value.type == expected_type
        and value.key == opaque_artifact_key(value.producer_key, output),
        "ORIGIN_ARTIFACT_EDGE",
    )
    return value


class PopulationOriginBindingKernel(_Kernel):
    """Bind carried source evidence to every current row and clone membership."""

    ref = "us.native_household_origin.bind@1"

    def run(self, context):
        node = context.node
        native._require(
            not node.sources
            and not node.outputs
            and node.artifact_outputs == BINDING_OUTPUT
            and set(context.params) == {"columns", "parent_binding"},
            "ORIGIN_BIND_DECLARATION",
        )
        expected = {
            entity: [US_SCHEMA.entity_id_column(entity)]
            for entity in US_SCHEMA.entities
        }
        expected[US_SCHEMA.person_entity].extend(
            US_SCHEMA.membership_column(group) for group in US_SCHEMA.group_entities
        )
        for entity, column, _dtype in context.params["columns"]:
            expected[entity].append(column)
        native._require(
            set(context.tables) == set(expected)
            and all(
                set(context.tables[e]) == set(columns)
                for e, columns in expected.items()
            ),
            "ORIGIN_BIND_COLUMN_INVENTORY",
        )
        # Graph context supplies IDs and memberships implicitly. All remaining
        # source cells must be declared by the integrating graph. The returned
        # content digest must also be verified against the complete materialized
        # population: this projected context cannot discover omitted columns.
        tables = {
            e: context.tables[e].loc[:, columns] for e, columns in expected.items()
        }
        frame = Frame(tables, US_SCHEMA, dict(context.weights), context.strata)
        aliases = {"acs", "asec"}
        if context.params["parent_binding"]:
            aliases.add("parent_binding")
        native._require(
            set(context.artifacts) == aliases, "ORIGIN_BIND_ARTIFACT_ROSTER"
        )
        values = [
            _artifact(context, arm, SOURCE_TYPE, "origins") for arm in ("acs", "asec")
        ]
        # Typed immutable transport was content-checked by the executor. This
        # does not reopen sources or mint an independent source-authentication
        # claim; the concrete producer keys are carried into the new artifact.
        sources = []
        for arm, value in zip(("acs", "asec"), values, strict=True):
            native._require(
                native._source_document(value.payload)["arm"] == arm,
                "ORIGIN_SOURCE_ARM",
            )
            sources.append(
                native.AuthenticatedNativeOriginSource(
                    value.payload, _token=native._TOKEN
                )
            )
        parent = None
        if "parent_binding" in aliases:
            value = _artifact(context, "parent_binding", BINDING_TYPE, "binding")
            doc = json.loads(value.payload)
            native._require(
                canonical_json(doc) == value.payload
                and doc["schema"] == native.BINDING_SCHEMA,
                "ORIGIN_PARENT_PAYLOAD",
            )
            parent = native.PopulationOriginBinding(
                value.payload, _token=native._BOUND_TOKEN
            )
        result = native.bind_population_origins(frame, sources=sources, parent=parent)
        document = result.document
        document["graph_parent_edges"] = {
            alias: {
                "producer_key": value.producer_key,
                "artifact_key": value.key,
                "payload_sha256": native._sha(value.payload),
            }
            for alias, value in context.artifacts.items()
        }
        document["population_node"] = node.population
        payload = canonical_json(document)
        return KernelResult(
            artifacts={"binding": payload},
            receipt={
                "phase": "native_household_origin_binding",
                "summary": result.summary,
                "binding_sha256": native._sha(payload),
                "parent_artifact_keys": {
                    alias: value.key for alias, value in context.artifacts.items()
                },
                "implementation": implementation_manifest(),
                "release_eligible": False,
            },
        )


def native_origin_source_nodes(*, population: str) -> tuple[Node, ...]:
    """Declare source-only producers at an explicit graph population version.

    The graph requires the version to disambiguate structural branches. These
    kernels declare no population columns and change no rows.
    """
    return (
        Node(
            ACS_NODE,
            NativeACSOriginKernel.ref,
            population=population,
            sources=(ACS_SOURCE,),
            artifact_outputs=SOURCE_OUTPUT,
        ),
        Node(
            ASEC_NODE,
            NativeASECOriginKernel.ref,
            population=population,
            sources=ASEC_SOURCES,
            artifact_outputs=SOURCE_OUTPUT,
        ),
    )


def native_origin_binding_node(
    columns: Sequence[Owned],
    *,
    population: str,
    node_id: str,
    parent_binding_node: str | None = None,
) -> Node:
    """Bind the exact declared full population; optionally extend native ancestry."""
    grouped = {entity: [] for entity in US_SCHEMA.entities}
    for cell in columns:
        grouped[cell.entity].append(cell.column)
    native._require(
        all(grouped.values()) and all(len(v) == len(set(v)) for v in grouped.values()),
        "ORIGIN_COLUMN_DECLARATION",
    )
    artifacts = [
        ArtifactInput("acs", ACS_NODE, "origins", SOURCE_TYPE),
        ArtifactInput("asec", ASEC_NODE, "origins", SOURCE_TYPE),
    ]
    if parent_binding_node is not None:
        artifacts.append(
            ArtifactInput(
                "parent_binding", parent_binding_node, "binding", BINDING_TYPE
            )
        )
    return Node(
        node_id,
        PopulationOriginBindingKernel.ref,
        population=population,
        inputs=tuple(
            Slice(entity, tuple(values)) for entity, values in grouped.items()
        ),
        artifact_inputs=tuple(artifacts),
        artifact_outputs=BINDING_OUTPUT,
        params={
            "columns": tuple((c.entity, c.column, c.dtype) for c in columns),
            "parent_binding": parent_binding_node is not None,
        },
    )


def register_native_origin_kernels(registry: KernelRegistry) -> None:
    """Register only the three new kernels, without editing upstream identities."""
    for kernel in (
        NativeACSOriginKernel(),
        NativeASECOriginKernel(),
        PopulationOriginBindingKernel(),
    ):
        registry.register(kernel)


def verify_materialized_population_origins(
    manifest,
    store,
    *,
    population_node: str,
    binding_node: str,
    parent_binding_node: str | None = None,
) -> native.PopulationOriginBinding:
    """Verify graph evidence against its complete materialized population.

    This is a mandatory integration boundary: a kernel cannot know whether its
    declared slices omitted another population column. The manifest must come
    from the reviewed graph and store; matching local receipts do not constitute
    an external producer authorization or approve any benchmark evaluation.
    No original source is reopened, and no kernel is executed here.
    """
    expected_implementation = implementation_hash()

    def read(node_id, output, expected_type, expected_kernel):
        receipt = manifest.nodes[node_id]
        descriptor = receipt.typed_artifacts["outputs"][output]
        payload = store.load_bytes(receipt.opaque_artifacts[output])
        value = value_from_descriptor(payload, descriptor)
        native._require(
            receipt.kernel_ref == expected_kernel
            and receipt.kernel_impl_hash == expected_implementation
            and descriptor["producer"] == node_id
            and descriptor["artifact"] == output
            and value.producer_key == receipt.key
            and value.key == receipt.opaque_artifacts[output]
            and value.type == expected_type,
            "ORIGIN_MATERIALIZED_PRODUCER",
        )
        return value

    try:
        values = {
            arm: read(node_id, "origins", SOURCE_TYPE, kernel.ref)
            for arm, node_id, kernel in (
                ("acs", ACS_NODE, NativeACSOriginKernel),
                ("asec", ASEC_NODE, NativeASECOriginKernel),
            )
        }
        sources = [
            native.AuthenticatedNativeOriginSource(value.payload, _token=native._TOKEN)
            for value in values.values()
        ]
        parent = None
        if parent_binding_node is not None:
            values["parent_binding"] = read(
                parent_binding_node,
                "binding",
                BINDING_TYPE,
                PopulationOriginBindingKernel.ref,
            )
            parent = native.PopulationOriginBinding(
                values["parent_binding"].payload, _token=native._BOUND_TOKEN
            )
            # The native parent's own projection must also describe its actual
            # complete population. This boundary supports the first full clone
            # (or one native household selection), not an implicit ancestry DAG.
            verify_materialized_population_origins(
                manifest,
                store,
                population_node=parent.document["population_node"],
                binding_node=parent_binding_node,
            )
        artifact = read(
            binding_node, "binding", BINDING_TYPE, PopulationOriginBindingKernel.ref
        )
        actual = native.PopulationOriginBinding(
            artifact.payload, _token=native._BOUND_TOKEN
        )
        document = actual.document
        frame = manifest.population(population_node)
        native.verify_population_origin_binding(frame, actual)
        expected = native.bind_population_origins(
            frame, sources=sources, parent=parent
        ).document
        expected["population_node"] = population_node
        expected["graph_parent_edges"] = {
            alias: {
                "producer_key": value.producer_key,
                "artifact_key": value.key,
                "payload_sha256": native._sha(value.payload),
            }
            for alias, value in values.items()
        }
        native._require(
            document == expected and canonical_json(document) == artifact.payload,
            "ORIGIN_MATERIALIZED_LINEAGE",
        )
        return actual
    except native.NativeOriginError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        raise native.NativeOriginError("ORIGIN_MATERIALIZED_CONTRACT") from None
