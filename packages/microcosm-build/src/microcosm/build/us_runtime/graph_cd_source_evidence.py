"""Three terminal artifact-only nodes; no existing source or population rewrite."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from microcosm.frame import US_SCHEMA, Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    Slice,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.keys import opaque_artifact_key

from . import acs_hu_source_transport as transport
from . import cd_current_source_evidence as current
from . import graph_composed_population as composed
from . import graph_geography as geography
from . import graph_native_household_origin as native
from .graph_housing_universe import US_ASEC_HOUSING_UNIVERSE_TYPE, bind_housing_universe
from .graph_native_origin_implementation import DEPENDENCIES
from .support_provenance import support_channel_column

PREFIX = "cd.source_evidence"
ACS_NODE = PREFIX + ".acs"
UNIVERSE_NODE = PREFIX + ".universe"
GEOGRAPHY_NODE = PREFIX + ".geography"
PRIVATE_TYPE = ArtifactType("microcosm.us.cd_current_source_evidence", 1)
PRIVATE_MAX_BYTES = 64 * 1024**2


def implementation_hash():
    from .graph_cd_source_implementation import implementation_hash as identity

    return identity()


def edge_pin(value):
    return {
        "producer_key": value.producer_key,
        "artifact_key": value.key,
        "payload_sha256": current.sha(value.payload),
        "type": {"name": value.type.name, "schema_version": value.type.schema_version},
    }


def artifact(context, alias):
    declaration = next(
        item for item in context.node.artifact_inputs if item.name == alias
    )
    value = context.artifacts[alias]
    current.require(
        value.type == declaration.type
        and value.key == opaque_artifact_key(value.producer_key, declaration.artifact),
        "CD_TYPED_ARTIFACT_EDGE",
    )
    return value


def document(value):
    current.require(
        type(value.payload) is bytes and len(value.payload) <= PRIVATE_MAX_BYTES,
        "CD_PRIVATE_PAYLOAD_CAP",
    )
    doc = json.loads(value.payload)
    current.require(
        type(doc) is dict and canonical_json(doc) == value.payload,
        "CD_PRIVATE_CANONICAL",
    )
    return doc


def context_frame(context):
    """Declared complete cells; independent full Population verification required."""
    tables = {e: context.tables[e].copy() for e in US_SCHEMA.entities}
    return Frame(tables, US_SCHEMA, dict(context.weights), context.strata)


def _result(kind, body, context):
    payload = canonical_json(
        {
            **body,
            "producer_edges": {
                name: edge_pin(value) for name, value in context.artifacts.items()
            },
            "population_node": context.node.population,
        }
    )
    current.require(len(payload) <= PRIVATE_MAX_BYTES, "CD_PRIVATE_PAYLOAD_CAP")
    return KernelResult(
        artifacts={"evidence": payload},
        receipt={
            "phase": kind,
            "payload_sha256": current.sha(payload),
            "row_count": len(body["records"]),
            "release_eligible": False,
            "definition_admission": "unresolved",
        },
    )


class _Kernel(KernelBase):
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=DEPENDENCIES,
    )

    def implementation_hash(self):
        return implementation_hash()


class ACSSourceEvidenceKernel(_Kernel):
    ref = transport.KERNEL_REF

    def run(self, context):
        current.require(
            context.node.artifact_outputs
            == (ArtifactOutput("projection", transport.TYPE),)
            and tuple(context.node.sources) == (native.ACS_SOURCE,)
            and not context.node.outputs
            and not context.params,
            "CD_ACS_SOURCE_DECLARATION",
        )
        # Read after the existing provenance-ownership claim, as do the two
        # consumers. This adds no ancestor to any existing structural node.
        document(artifact(context, "clone_binding"))
        table = context.tables["household"]
        chosen = tuple(
            sorted(
                set(
                    table.loc[
                        table[support_channel_column("household")] == "acs", "SERIALNO"
                    ]
                )
            )
        )
        with tempfile.TemporaryDirectory(prefix="cd-acs-source-") as name:
            root = Path(name).resolve()
            snapshots = root / "snapshots"
            snapshots.mkdir()
            observation = transport.source.produce_acs_housing_source(
                context.sources[native.ACS_SOURCE],
                snapshot_root=snapshots,
                output_dir=root / "projection",
            )
            payload = transport.encode_source_projection(
                observation, selected_serialnos=chosen
            )
        return KernelResult(
            artifacts={"projection": payload},
            receipt={
                "phase": "cd_acs_source_projection",
                "source_receipt_sha256": current.sha(observation.receipt_json),
                "projection_sha256": current.sha(payload),
                "selected_households": len(chosen),
                "release_eligible": False,
            },
        )


def native_geography_body(frame, binding, *, assignment, ladder, crosswalk):
    """Attest observed output cells and real lookup roster without rerunning RNG."""
    native.native._require(
        binding["population_content_sha256"]
        == native.native._population_content(frame),
        "CD_NATIVE_GEOGRAPHY_FULL_CONTENT",
    )
    _ladder, expected = geography._lookups(ladder.payload, crosswalk.payload)
    current.require(len(expected) == 436, "CD_GEOGRAPHY_LOOKUP_ROSTER")
    household = frame.table("household").set_index("household_id")
    records = []
    for row in binding["households"]:
        current.require(row["clone_index"] == 0, "CD_NATIVE_GEOGRAPHY_ROLE")
        cell = household.loc[row["household_id"]]
        values = {}
        for column in current.GEOGRAPHY_COLUMNS:
            value = cell[column]
            values[column] = value.item() if hasattr(value, "item") else value
        records.append(
            {
                "household_id": row["household_id"],
                "origin_key": row["origin_key"],
                **values,
            }
        )
    return {
        "schema": current.SCHEMA,
        "kind": "native_geography",
        "records": records,
        "assignment_evidence": {
            "seed": 0,
            "assign_tract": False,
            "district_vintage": geography.CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
            "target_district_count": len(expected),
            "assignment_edge": edge_pin(assignment),
            "ladder_edge": edge_pin(ladder),
            "crosswalk_edge": edge_pin(crosswalk),
        },
        "release_eligible": False,
    }


def universe_body(frame, binding, *, projection, acs_origins, housing, prepared):
    table = frame.table("household")
    keys = tuple(
        sorted(
            set(
                table.loc[
                    table[support_channel_column("household")] == "acs", "SERIALNO"
                ]
            )
        )
    )
    bound = transport.bind_source_projection(
        projection, expected_edge=edge_pin(projection), selected_serialnos=keys
    )
    receipt = json.loads(prepared.payload)
    body = bind_housing_universe(housing.payload, prepared_receipt=receipt)
    return current.classify_current(
        frame,
        binding,
        acs_projection=bound,
        acs_origins=document(acs_origins),
        asec_body=body,
        source_household_ids=body.array("household_id"),
    )


class CurrentUniverseEvidenceKernel(_Kernel):
    ref = "us.cd.current_universe_evidence@1"

    def run(self, context):
        body = universe_body(
            context_frame(context),
            document(artifact(context, "clone_binding")),
            **{
                name: artifact(context, name)
                for name in ("projection", "acs_origins", "housing", "prepared")
            },
        )
        return _result("cd_current_universe", body, context)


class CurrentGeographyEvidenceKernel(_Kernel):
    ref = "us.cd.current_geography_evidence@1"

    def run(self, context):
        frame = context_frame(context)
        binding = document(artifact(context, "clone_binding"))
        prior = artifact(context, "native_binding")
        current.require(
            binding["parent_binding_sha256"] == current.sha(prior.payload),
            "CD_NATIVE_BINDING_EDGE",
        )
        rows, _entities = current._rows(frame, binding)
        household = frame.table("household").set_index("household_id")
        # This graph-context result establishes pair agreement only. Mandatory
        # full materialized verification below independently compares these
        # values with the actual preclone population before control issuance.
        native_rows = []
        for row in rows:
            if row["clone_index"] == 0:
                cell = household.loc[row["household_id"]]
                native_rows.append(
                    {
                        "origin_key": row["origin_key"],
                        **{
                            c: cell[c].item() if hasattr(cell[c], "item") else cell[c]
                            for c in current.GEOGRAPHY_COLUMNS
                        },
                    }
                )
        _ladder, expected = geography._lookups(
            artifact(context, "ladder").payload, artifact(context, "crosswalk").payload
        )
        body = current.attest_current_geography(
            frame,
            binding,
            native_rows=native_rows,
            assignment_evidence={
                "seed": 0,
                "assign_tract": False,
                "district_vintage": geography.CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
                "target_district_count": len(expected),
                "assignment_edge": edge_pin(artifact(context, "assignment")),
                "ladder_edge": edge_pin(artifact(context, "ladder")),
                "crosswalk_edge": edge_pin(artifact(context, "crosswalk")),
            },
        )
        return _result("cd_current_geography", body, context)


def evidence_nodes(
    columns,
    *,
    native_population,
    current_population,
    native_binding,
    clone_binding,
    prepared_node,
    native_columns=None,
):
    """Three terminal artifact-only nodes; all predecessor declarations stay exact."""
    grouped = {entity: [] for entity in US_SCHEMA.entities}
    for cell in columns:
        grouped[cell.entity].append(cell.column)
    slices = tuple(Slice(entity, tuple(values)) for entity, values in grouped.items())
    lookup = geography.GEOGRAPHY_PHASE + ".lookup"
    assign = geography.GEOGRAPHY_PHASE
    binding = ArtifactInput(
        "clone_binding", clone_binding, "binding", native.BINDING_TYPE
    )
    return (
        Node(
            ACS_NODE,
            ACSSourceEvidenceKernel.ref,
            population=current_population,
            inputs=(
                Slice("household", (support_channel_column("household"), "SERIALNO")),
            ),
            artifact_inputs=(binding,),
            sources=(native.ACS_SOURCE,),
            artifact_outputs=(ArtifactOutput("projection", transport.TYPE),),
        ),
        Node(
            UNIVERSE_NODE,
            CurrentUniverseEvidenceKernel.ref,
            population=current_population,
            inputs=slices,
            artifact_inputs=(
                binding,
                ArtifactInput("projection", ACS_NODE, "projection", transport.TYPE),
                ArtifactInput(
                    "acs_origins", native.ACS_NODE, "origins", native.SOURCE_TYPE
                ),
                ArtifactInput(
                    "housing",
                    prepared_node,
                    "housing_universe",
                    US_ASEC_HOUSING_UNIVERSE_TYPE,
                ),
                ArtifactInput(
                    "prepared",
                    prepared_node,
                    "prepared_receipt",
                    composed.US_ASEC_PREPARED_RECEIPT_TYPE,
                ),
            ),
            artifact_outputs=(ArtifactOutput("evidence", PRIVATE_TYPE),),
        ),
        Node(
            GEOGRAPHY_NODE,
            CurrentGeographyEvidenceKernel.ref,
            population=current_population,
            inputs=slices,
            artifact_inputs=(
                binding,
                ArtifactInput(
                    "native_binding", native_binding, "binding", native.BINDING_TYPE
                ),
                ArtifactInput(
                    "assignment",
                    assign,
                    "frame_context",
                    geography.US_FRAME_CONTEXT_TYPE,
                ),
                ArtifactInput(
                    "ladder", lookup, "ladder", geography.US_PUMA_LOOKUP_TYPE
                ),
                ArtifactInput(
                    "crosswalk", lookup, "crosswalk", geography.US_CD_CROSSWALK_TYPE
                ),
            ),
            artifact_outputs=(ArtifactOutput("evidence", PRIVATE_TYPE),),
        ),
    )


def register(registry):
    for kernel in (
        ACSSourceEvidenceKernel(),
        CurrentUniverseEvidenceKernel(),
        CurrentGeographyEvidenceKernel(),
    ):
        registry.register(kernel)


def verify_materialized(
    manifest,
    store,
    graph,
    observed,
    *,
    expected_node_keys,
    native_population,
    clone_population,
    current_population,
    native_binding_node,
    clone_binding_node,
    source_git_head,
    parent=None,
    successor=None,
):
    """Actual full-boundary proof, mandatory on cold AND cached execution.

    ``observed`` contains the executor's real Population objects, including
    anchors/owners/ledgers. Keys are independently computed/pinned by the caller;
    the candidate bundle is not an authority for its own producer identity.
    """
    from ..cd_benchmark_io import source_adapters as io
    from . import enriched_origin_calibration as full

    current.require(
        type(source_git_head) is str
        and len(source_git_head) == 40
        and all(c in "0123456789abcdef" for c in source_git_head),
        "CD_SOURCE_GIT_HEAD",
    )
    current.require(
        set(expected_node_keys) == set(manifest.nodes), "CD_MANIFEST_NODE_ROSTER"
    )
    for node_id, receipt in manifest.nodes.items():
        current.require(
            receipt.key == expected_node_keys[node_id]
            and receipt.kernel_ref == graph.node(node_id).kernel,
            "CD_EXPECTED_PRODUCER_KEY",
        )
    for node_id in (ACS_NODE, UNIVERSE_NODE, GEOGRAPHY_NODE):
        current.require(
            manifest.node(node_id).kernel_impl_hash == implementation_hash(),
            "CD_NEW_IMPLEMENTATION",
        )
    assignment_node = graph.node(geography.GEOGRAPHY_PHASE)
    current.require(
        assignment_node.kernel == geography.USGeographyAssignKernel.ref
        and canonical_json(assignment_node.params)
        == canonical_json(
            {"phase": geography.GEOGRAPHY_PHASE, "seed": 0, "assign_tract": False}
        )
        and manifest.node(assignment_node.id).kernel_impl_hash
        == geography.USGeographyAssignKernel().implementation_hash(),
        "CD_ACTUAL_GEOGRAPHY_PRODUCER",
    )
    native_bound = native.verify_materialized_population_origins(
        manifest,
        store,
        population_node=native_population,
        binding_node=native_binding_node,
    )
    clone_bound = native.verify_materialized_population_origins(
        manifest,
        store,
        population_node=clone_population,
        binding_node=clone_binding_node,
        parent_binding_node=native_binding_node,
    )
    before = observed[current_population]
    ancestry = current.authenticate_current_lineage(
        observed[clone_population],
        before,
        clone_bound,
        parent=parent,
        successor=successor,
    )
    for node_id in (ACS_NODE, UNIVERSE_NODE, GEOGRAPHY_NODE):
        current.verify_population_unchanged(before, observed[node_id])

    def values(node_id):
        result = {}
        for declaration in graph.node(node_id).artifact_inputs:
            value = full.edge_value(
                manifest,
                store,
                declaration.producer,
                declaration.artifact,
                declaration.type,
            )
            descriptor = manifest.node(declaration.producer).typed_artifacts["outputs"][
                declaration.artifact
            ]
            current.require(
                descriptor["producer"] == declaration.producer
                and descriptor["artifact"] == declaration.artifact,
                "CD_ACTUAL_DESCRIPTOR_PRODUCER",
            )
            result[declaration.name] = value
        return result

    geo_values = values(GEOGRAPHY_NODE)
    native_geo = native_geography_body(
        observed[native_population].frame,
        native_bound.document,
        assignment=geo_values["assignment"],
        ladder=geo_values["ladder"],
        crosswalk=geo_values["crosswalk"],
    )
    universe_values = values(UNIVERSE_NODE)
    universe = universe_body(
        before.frame,
        clone_bound.document,
        **{
            name: universe_values[name]
            for name in ("projection", "acs_origins", "housing", "prepared")
        },
    )
    geo = current.attest_current_geography(
        before.frame,
        clone_bound.document,
        native_rows=native_geo["records"],
        assignment_evidence=native_geo["assignment_evidence"],
    )
    expected_bodies = {UNIVERSE_NODE: universe, GEOGRAPHY_NODE: geo}
    actual_values = {}
    for node_id, body in expected_bodies.items():
        value = full.edge_value(manifest, store, node_id, "evidence", PRIVATE_TYPE)
        expected = {
            **body,
            "producer_edges": {
                name: edge_pin(v) for name, v in values(node_id).items()
            },
            "population_node": graph.node(node_id).population,
        }
        current.require(
            value.payload == canonical_json(expected),
            "CD_MATERIALIZED_EVIDENCE_MISMATCH",
        )
        actual_values[node_id] = value
    population_identity = current.current_population_identity(before)
    semantic = current.digest(population_identity)
    identity = {
        "candidate_id": "invented-cd-source-evidence",
        "population_artifact_id": current_population,
        "population_content_sha256": population_identity["content_sha256"],
        "population_semantic_snapshot_sha256": semantic,
        "producer_run_id": current.digest(expected_node_keys),
        "producer_git_head": source_git_head,
        "producer_implementation_sha256": implementation_hash(),
        "model_identity_or_not_invoked": "not_invoked",
    }
    slots = {}
    for slot, node_id, body in (
        (io.UNIVERSE, UNIVERSE_NODE, universe),
        (io.GEOGRAPHY, GEOGRAPHY_NODE, geo),
    ):
        value = actual_values[node_id]
        slots[slot] = {
            "dependencies": {},
            "evidence": {
                "graph_artifact": edge_pin(value),
                "ordered_rows_sha256": current.digest(body["records"]),
                "entity_membership_sha256": current.digest(body["entities"]),
                "entity_rows": {e: len(rows) for e, rows in body["entities"].items()},
                "household_rows": len(body["records"]),
                "zero_weight_household_rows": sum(
                    float.fromhex(r["weight_hex"]) == 0
                    for r in body["entities"]["household"]
                ),
                "materialized_population": population_identity,
                "source_lineage": ancestry,
                "source_edges_sha256": current.digest(
                    {alias: edge_pin(v) for alias, v in values(node_id).items()}
                ),
            },
        }
    return {"identity": identity, "slots": slots}
