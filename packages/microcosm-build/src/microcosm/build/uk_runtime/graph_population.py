"""Population operations in the single UK full-build graph.

Source interpretation stays in UK adapters. Selection, row ancestry, typed
weights, content storage and replay are enforced by the shared graph runtime.
The legacy numerical functions remain the only implementations of the draws.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from microcosm.frame import Frame, Weights
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
    Owned,
    SeedSource,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightUpdate,
    compile_graph,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import dtype_for_token
from microcosm.graph.weight_update import weight_update_receipt

from . import geography_ladder, national_sampling, rowwise_dataset
from .geography_ladder import (
    UK_GEOGRAPHY_LADDER_COLUMNS,
    derive_uk_ladder_locations,
    draw_uk_ladder_locations,
    load_uk_oa_ladder,
)
from .national_frame import UK_NATIONAL_SCHEMA
from .rowwise_dataset import expand_uk_geographic_pool, ladder_clone_index_column

POPULATION_RECEIPT_TYPE = ArtifactType("microcosm.uk.population-receipt", 1)
LOCATION_DRAW_TYPE = ArtifactType("microcosm.uk.ladder-location-draw", 1)
GEOGRAPHY_GATE_TYPE = ArtifactType("microcosm.uk.geography-gate", 1)


def context_frame(context: KernelContext) -> Frame:
    """Rebuild only declared slices, with the shared immutable weight context."""

    return Frame(
        {
            entity: context.tables[entity]
            .loc[
                :,
                list(
                    context.frame_column_order.get(
                        entity, tuple(context.tables[entity].columns)
                    )
                ),
            ]
            .copy(deep=True)
            for entity in UK_NATIONAL_SCHEMA.entities
        },
        UK_NATIONAL_SCHEMA,
        {"household": context.weights["household"]},
        context.strata.copy(deep=True),
        mass_log=getattr(context, "frame_mass_log", ()),
        metadata=getattr(context, "frame_metadata", {})
        or {"time_period": str(context.params["time_period"])},
    )


def population_columns(graph: Graph, population: str) -> dict[tuple[str, str], str]:
    """Resolve the carried and newly owned columns in a compiled version."""

    compiled = compile_graph(graph)
    holder = graph.node(population)
    cells = {} if holder.base is None else population_columns(graph, holder.base)
    for node in graph.nodes:
        if compiled.versions[node.id] == population:
            cells.update({(o.entity, o.column): o.dtype for o in node.outputs})
    return cells


def population_slices(cells: Mapping[tuple[str, str], str]) -> tuple[Slice, ...]:
    return tuple(
        Slice(entity, tuple(sorted(c for e, c in cells if e == entity)))
        for entity in sorted({e for e, _ in cells})
    )


def _series(frame: Frame, entity: str, column: str, dtype: str) -> pd.Series:
    table = frame.table(entity)
    ids = pd.Index(
        table[frame.schema.entity_id_column(entity)],
        name=frame.schema.entity_id_column(entity),
    )
    return pd.Series(table[column].array, index=ids, name=column).astype(
        dtype_for_token(dtype)
    )


def _mass_records(before: Frame, after: Frame) -> list[dict]:
    if after.mass_log[: len(before.mass_log)] != before.mass_log:
        raise ValueError(
            "A graph population operation replaced its incoming mass ledger."
        )
    return [asdict(record) for record in after.mass_log[len(before.mass_log) :]]


def _declared_mass(before: Frame, after: Frame) -> dict:
    old, new = before.stratum_mass(), after.stratum_mass()
    return {
        "policy": "declared",
        "before": float(old.sum()),
        "after": float(new.sum()),
        "stratum_before": {k: float(v) for k, v in old.items()},
        "stratum_after": {k: float(v) for k, v in new.items()},
    }


class _PopulationKernel(KernelBase):
    def implementation_hash(self) -> str:
        from . import graph_evidence

        return source_hash(
            type(self),
            geography_ladder,
            national_sampling,
            rowwise_dataset,
            graph_evidence,
        )


class UKFamilySampleKernel(_PopulationKernel):
    ref = "uk.full.sample@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        seed_source=SeedSource.PARAM,
        structural=StructuralDelta.FILTER,
    )

    def run(self, context: KernelContext) -> KernelResult:
        from .graph_evidence import require_uk_spine_gate_admission

        require_uk_spine_gate_admission(context)
        before = context_frame(context)
        fraction = float(context.params["fraction"])
        seed = int(context.params["seed"])
        if fraction == 1.0:
            after = before
            receipt = {
                "fraction": fraction,
                "seed": seed,
                "sampled": False,
                "pre_household_count": before.n("household"),
                "post_household_count": before.n("household"),
                "rung_token": "f100",
            }
        else:
            after, receipt = national_sampling.sample_uk_spine_frame(
                before, fraction=fraction, seed=seed
            )
            receipt = {"sampled": True, **receipt}
        ids = before.table("person")["person_id"]
        keep = pd.Series(
            ids.isin(after.table("person")["person_id"]).to_numpy(),
            index=pd.Index(ids, name="person_id"),
            dtype=bool,
        )
        # The sampling function computes the historical normalization once.
        # Installing its weights is a separate declared same-kind operation.
        payload = {
            "receipt": receipt,
            "household_ids": after.table("household")["household_id"].tolist(),
            "weights": after.weights_for("household").values.tolist(),
            "mass_log_append": _mass_records(before, after),
        }
        return KernelResult(keep=keep, artifacts={"sampling": canonical_json(payload)})


class UKSampleNormalizationKernel(_PopulationKernel):
    ref = "uk.full.normalize@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
    )

    def run(self, context: KernelContext) -> KernelResult:
        frame = context_frame(context)
        payload = json.loads(context.artifacts["sampling"].payload)
        ids = frame.table("household")["household_id"].tolist()
        if ids != payload["household_ids"]:
            raise ValueError("Sample normalization household axis changed.")
        weights = Weights(
            np.asarray(payload["weights"], dtype=np.float64),
            kind=context.weights["household"].kind,
        )
        after = Frame(
            {e: frame.table(e) for e in frame.entities},
            frame.schema,
            {"household": weights},
            frame.strata,
            metadata=frame.metadata,
        )
        return KernelResult(
            weights=weights,
            receipt={
                "weight_update": weight_update_receipt(ids),
                "mass": _declared_mass(frame, after),
                "frame_mass_log_append": payload["mass_log_append"],
            },
        )


class UKGeographicExpansionKernel(_PopulationKernel):
    ref = "uk.full.expand@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.EXPAND
    )

    def run(self, context: KernelContext) -> KernelResult:
        before = context_frame(context)
        household = before.table("household").copy()
        household["household_weight"] = before.weights_for("household").values
        pool = expand_uk_geographic_pool(
            person=before.table("person"),
            benunit=before.table("benunit"),
            household=household,
            n_clones=int(context.params["n_clones"]),
            source_year=int(context.params["source_year"]),
            time_period=str(context.params["time_period"]),
            household_weight_kind=before.weights_for("household").kind,
            mass_log=before.mass_log,
            source_lineage_modulus=context.params.get("source_lineage_modulus"),
        )
        after = pool.frame
        ancestry = {}
        for entity in before.entities:
            table = after.table(entity)
            id_column = before.schema.entity_id_column(entity)
            added = table.iloc[before.n(entity) :]
            source_ids = (
                added[id_column].to_numpy()
                - added[ladder_clone_index_column(entity)].to_numpy()
                * pool.id_multiplier
            )
            ancestry[entity] = pd.Series(
                source_ids, index=pd.Index(added[id_column], name=id_column)
            )
        columns = {
            (e, c): _series(after, e, c, dtype)
            for e, c, dtype in context.params["expand_cells"]
        }
        return KernelResult(
            expand=ancestry,
            columns=columns,
            weights=after.weights_for("household"),
            artifacts={
                "expansion": canonical_json(
                    {"n_clones": pool.n_clones, "id_multiplier": pool.id_multiplier}
                )
            },
            receipt={"frame_mass_log_append": _mass_records(before, after)},
        )


class UKLocationDrawKernel(_PopulationKernel):
    ref = "uk.full.locations@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC, seed_source=SeedSource.PARAM)

    def run(self, context: KernelContext) -> KernelResult:
        ladder = load_uk_oa_ladder(context.sources["uk_ladder"])
        household = context.tables["household"]
        indices = draw_uk_ladder_locations(
            household,
            ladder,
            seed=int(context.params["seed"]),
            expected_constituency_vintage=str(context.params["constituency_vintage"]),
        )
        return KernelResult(
            artifacts={
                "locations": canonical_json(
                    {
                        "household_ids": household["household_id"].tolist(),
                        "indices": indices.tolist(),
                        "seed": int(context.params["seed"]),
                        "layer_vintages": ladder.layer_vintages,
                        "constituency_sampling_basis": ladder.metadata[
                            "constituency_sampling_basis"
                        ],
                        "oa_sampling_basis": ladder.metadata["oa_sampling_basis"],
                    }
                )
            }
        )


class UKGeographyMappingKernel(_PopulationKernel):
    ref = "uk.full.geography_mapping@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context: KernelContext) -> KernelResult:
        payload = json.loads(context.artifacts["locations"].payload)
        household = context.tables["household"]
        if household["household_id"].tolist() != payload["household_ids"]:
            raise ValueError("Location draw is bound to a different household axis.")
        assigned = derive_uk_ladder_locations(
            household,
            load_uk_oa_ladder(context.sources["uk_ladder"]),
            np.asarray(payload["indices"], dtype=np.int64),
        )
        index = pd.Index(household["household_id"], name="household_id")
        return KernelResult(
            columns={
                (owned.entity, owned.column): pd.Series(
                    assigned[owned.column].array, index=index, name=owned.column
                ).astype(dtype_for_token(owned.dtype))
                for owned in context.node.outputs
            }
        )


class UKGeographyGateKernel(_PopulationKernel):
    ref = "uk.full.geography_gate@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context: KernelContext) -> KernelResult:
        frame = context_frame(context)
        gate = geography_ladder.uk_geography_ladder_gate(
            frame.table("household"), frame.weights_for("household").values
        )
        # Persist the failure. Downstream target materialization explicitly
        # refuses this outcome; cached failure must never turn into a pass.
        payload = asdict(gate)
        return KernelResult(
            artifacts={"gate": canonical_json(payload)},
            receipt={"outcome": "pass" if gate.passed else "fail", "evidence": payload},
        )


def append_uk_population_nodes(
    graph: Graph,
    *,
    population: str,
    time_period: str,
    weight_kind: str,
    sample_fraction: float = 1.0,
    sample_seed: int = national_sampling.UK_SAMPLE_SEED_DEFAULT,
    n_clones: int = 1,
    seed: int = 42,
    source_year: int = 2024,
    constituency_vintage: str = "2024_pcon",
    source_lineage_modulus: int | None = None,
) -> Graph:
    """Compose sampling → replication → location draw → derivation → gate."""

    from .graph_kernels import UKClaimKernel

    cells = population_columns(graph, population)
    if type(n_clones) is not int or n_clones < 1:
        raise ValueError("Geographic replicate count K must be a positive integer.")
    national_sampling.validate_sample_fraction(sample_fraction, label="UK full build")
    national_sampling.validate_sample_seed(sample_seed, label="UK full build")
    common = {"time_period": str(time_period)}
    spine_gate_inputs = ()
    spine_gate_params = {}
    if any(node.id == "spine.gates.transferred" for node in graph.nodes):
        from .graph_evidence import SPINE_GATE_REPORT_TYPE

        gate = graph.node("spine.gates.transferred")
        spine_gate_inputs = (
            ArtifactInput("spine_gate", gate.id, "gate_report", SPINE_GATE_REPORT_TYPE),
        )
        spine_gate_params = {
            "spine_gate_phase": "transferred",
            "spine_gate_release_candidate": bool(gate.params["release_candidate"]),
        }
    nodes = list(graph.nodes)
    nodes.append(
        Node(
            id="uk.full.sample",
            kernel=UKFamilySampleKernel.ref,
            inputs=population_slices(cells),
            base=population,
            structural=StructuralDelta.FILTER,
            mass="free",
            params={
                **common,
                "fraction": float(sample_fraction),
                "seed": sample_seed,
                **spine_gate_params,
            },
            artifact_inputs=spine_gate_inputs,
            artifact_outputs=(ArtifactOutput("sampling", POPULATION_RECEIPT_TYPE),),
            description="Select intact source families; calculate their historical mass normalization.",
        )
    )
    nodes.append(
        Node(
            id="uk.full.normalize",
            kernel=UKSampleNormalizationKernel.ref,
            inputs=population_slices(cells),
            base="uk.full.sample",
            structural=StructuralDelta.REWEIGHT,
            mass="declared",
            weights=WeightUpdate(
                "household", weight_kind, "Normalize sampled source-family mass."
            ),
            artifact_inputs=(
                ArtifactInput(
                    "sampling", "uk.full.sample", "sampling", POPULATION_RECEIPT_TYPE
                ),
            ),
            params=common,
            description="Install normalized weights without changing their kind.",
        )
    )
    expansion_cells = {
        ("household", "source_household_id"): "int64",
        ("household", "source_year"): "int64",
        ("household", "source_household_key"): "string",
        **{
            (entity, ladder_clone_index_column(entity)): "int64"
            for entity in UK_NATIONAL_SCHEMA.entities
        },
    }
    # Existing source lineage is carried unchanged, except the explicitly
    # requested historical modulus conversion, which owns its declarations.
    if source_lineage_modulus is not None:
        expansion_cells.update(
            {("household", rowwise_dataset.POOL_SOURCE_LINEAGE_COLUMN): "int64"}
        )
    expansion_cells = {
        key: cells.get(key, dtype) for key, dtype in expansion_cells.items()
    }
    nodes.append(
        Node(
            id="uk.full.expand",
            kernel=UKGeographicExpansionKernel.ref,
            inputs=population_slices(cells),
            structural=StructuralDelta.EXPAND,
            base="uk.full.normalize",
            mass="conserve",
            params={
                **common,
                "source_year": source_year,
                "n_clones": n_clones,
                "source_lineage_modulus": source_lineage_modulus,
                "expand_weight_entity": "household",
                "expand_weight_kind": weight_kind,
                "expand_cells": tuple(
                    (e, c, dtype) for (e, c), dtype in sorted(expansion_cells.items())
                ),
            },
            artifact_outputs=(ArtifactOutput("expansion", POPULATION_RECEIPT_TYPE),),
            description="Prepare source lineage and expand linked entities into K geographic copies.",
        )
    )
    nodes.append(
        Node(
            id="uk.full.expand.owned",
            kernel=UKClaimKernel.ref,
            population="uk.full.expand",
            outputs=tuple(
                Owned(e, c, dtype, rewrite=(e, c) in cells)
                for (e, c), dtype in sorted(expansion_cells.items())
            ),
            params={
                "materialized_expand_outputs": tuple(
                    f"{e}.{c}" for e, c in expansion_cells if (e, c) not in cells
                )
            },
            description="Declare the lineage and replicate columns materialized by expansion.",
        )
    )
    cells.update(expansion_cells)
    nodes.append(
        Node(
            id="uk.full.locations",
            kernel=UKLocationDrawKernel.ref,
            population="uk.full.expand",
            inputs=(Slice("household", ("region",)),),
            sources=("uk_ladder",),
            params={"seed": seed, "constituency_vintage": constituency_vintage},
            artifact_outputs=(ArtifactOutput("locations", LOCATION_DRAW_TYPE),),
            description="Draw constituency then atomic area with the current sequential RNG.",
        )
    )
    nodes.append(
        Node(
            id="uk.full.geography_mapping",
            kernel=UKGeographyMappingKernel.ref,
            population="uk.full.expand",
            inputs=(Slice("household", ("region",)),),
            sources=("uk_ladder",),
            artifact_inputs=(
                ArtifactInput(
                    "locations", "uk.full.locations", "locations", LOCATION_DRAW_TYPE
                ),
            ),
            outputs=tuple(
                Owned("household", col, "string", rewrite=("household", col) in cells)
                for col in UK_GEOGRAPHY_LADDER_COLUMNS
            ),
            description="Derive every geography from the drawn atomic-area index.",
        )
    )
    cells.update({("household", col): "string" for col in UK_GEOGRAPHY_LADDER_COLUMNS})
    nodes.append(
        Node(
            id="uk.full.geography_gate",
            kernel=UKGeographyGateKernel.ref,
            population="uk.full.expand",
            inputs=population_slices(cells),
            params=common,
            artifact_outputs=(ArtifactOutput("gate", GEOGRAPHY_GATE_TYPE),),
            description="Validate geography integrity before selected-target contributions.",
        )
    )
    sources = (
        graph.sources
        if any(s.name == "uk_ladder" for s in graph.sources)
        else (
            *graph.sources,
            SourceRef(
                "uk_ladder",
                "raw-bytes-v1",
                "Full UK atomic-area ladder with pinned vintages.",
            ),
        )
    )
    return replace(graph, nodes=tuple(nodes), sources=tuple(sources))


def register_uk_population_kernels(registry: KernelRegistry) -> None:
    for kernel in (
        UKFamilySampleKernel(),
        UKSampleNormalizationKernel(),
        UKGeographicExpansionKernel(),
        UKLocationDrawKernel(),
        UKGeographyMappingKernel(),
        UKGeographyGateKernel(),
    ):
        registry.register(kernel)
