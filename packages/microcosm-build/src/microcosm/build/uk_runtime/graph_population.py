"""Population operations in the single UK full-build graph.

Source interpretation stays in UK adapters. Selection, row ancestry, typed
weights, content storage and replay are enforced by the shared graph runtime.
The legacy numerical functions remain the only implementations of the draws.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

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
    Numeric,
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

from . import (
    atomic_area_support,
    atomic_household_identity,
    geography_ladder,
    local_authority_input,
    national_sampling,
    property_localisation,
    rowwise_dataset,
)
from .geography_ladder import (
    UK_GEOGRAPHY_LADDER_COLUMNS,
    derive_uk_ladder_locations,
    draw_uk_ladder_locations,
    load_uk_oa_ladder,
)
from .household_lineage import (
    HIGH_VALUE_SUPPORT_COPY_COLUMN,
    UK_HOUSEHOLD_LINEAGE_COLUMNS,
    UK_OPTIONAL_LINEAGE_COLUMNS,
    uk_household_clone_paths,
)
from .local_authority_input import (
    UK_LOCAL_AUTHORITY_CODE_COLUMN,
    UK_LOCAL_AUTHORITY_INPUT_COLUMN,
    UK_LOCAL_AUTHORITY_VINTAGE,
    resolve_local_authority_engine_keys,
    verify_local_authority_engine_domain,
)
from .national_frame import UK_NATIONAL_SCHEMA
from .rowwise_dataset import expand_uk_geographic_pool, ladder_clone_index_column

POPULATION_RECEIPT_TYPE = ArtifactType("microcosm.uk.population-receipt", 1)
LOCATION_DRAW_TYPE = ArtifactType("microcosm.uk.ladder-location-draw", 1)
GEOGRAPHY_GATE_TYPE = ArtifactType("microcosm.uk.geography-gate", 1)
GEOGRAPHY_ASSIGNMENTS = ("atomic", "legacy")
#: Lineage columns the post-clone household identity is keyed on. The support
#: channel, the CGT support split's copy flag and index (microcosm#1045) and
#: the clone flag come from the spine stages; the geographic clone index from
#: ``uk.full.expand``. Calibration K, weights and remapped ids are not inputs.
UK_GEOGRAPHY_IDENTITY_INPUTS = UK_HOUSEHOLD_LINEAGE_COLUMNS
PROPERTY_LOCALISATION_MODES = property_localisation.PROPERTY_LOCALISATION_MODES
PROPERTY_QUANTILE_TYPE = ArtifactType("microcosm.uk.property-quantile-draw", 1)
#: Household columns the post-geography property nodes read and rewrite.
_PROPERTY_VALUE_COLUMNS = (
    "region",
    "tenure_type",
    "main_residence_value",
    "property_wealth",
)
_PROPERTY_BAND_COLUMNS = ("council_tax_band",)


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


class UKGeographyIdentityKernel(_PopulationKernel):
    """Key every post-clone household by its source id and ordered clone path.

    The path is read from the spine's explicit lineage columns, never inferred
    from ids, weights or financial values: ``spi_support_channel`` with the
    support clone index when the household is an SPI support copy,
    ``cgt_support_split`` with the copy index when it is a support copy
    (microcosm#1045), ``cgt_incidence_clone`` (ordinal 1) from its flag,
    ``geographic_support`` with the pool clone index when it is not the
    original copy. Growing K never changes an existing household's key.
    """

    ref = "uk.full.identity@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def implementation_hash(self) -> str:
        return source_hash(
            type(self), atomic_household_identity, atomic_area_support, rowwise_dataset
        )

    def run(self, context: KernelContext) -> KernelResult:
        household = context.tables["household"]
        source = str(context.params["source"])
        vintage = str(context.params["source_vintage"])
        source_ids, paths = uk_household_clone_paths(household)
        keys = [
            atomic_household_identity.household_draw_key(
                source=source,
                source_vintage=vintage,
                source_household_id=source_id,
                clone_path=path,
            )
            for source_id, path in zip(source_ids, paths, strict=True)
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("Geography identity keys are not unique.")
        index = pd.Index(household["household_id"], name="household_id")
        column = atomic_area_support.IDENTITY_COLUMN
        return KernelResult(
            columns={
                ("household", column): pd.Series(
                    keys, index=index, name=column, dtype=dtype_for_token("string")
                )
            },
            receipt={
                "households": len(keys),
                "source": source,
                "source_vintage": vintage,
                "inputs": list(UK_GEOGRAPHY_IDENTITY_INPUTS),
            },
        )


class UKPropertyValueKernel(_PopulationKernel):
    """Read each owner clone's home value off its constituency's distribution.

    microcosm#1240 fixes 3 and 5. Every pool clone of one spine household
    keeps one uniform (its weighted rank among the owner households of its
    FRS region, ties broken by a keyed draw; high-value support copies take a
    keyed draw inside the top stratum), and reads its value at that uniform
    from the assigned constituency's uprated latest-sale knots. Weights are
    untouched; property wealth moves by the value's change.
    """

    ref = "uk.full.property.value@1"
    capabilities = Capabilities(
        determinism=Determinism.SEEDED,
        seed_source=SeedSource.KEYED,
        numeric=Numeric.PLATFORM_BITWISE,
    )

    def implementation_hash(self) -> str:
        from . import household_lineage

        return source_hash(
            type(self),
            property_localisation,
            household_lineage,
            atomic_household_identity,
        )

    def run(self, context: KernelContext) -> KernelResult:
        household = context.tables["household"]
        weights = np.asarray(context.weights["household"].values, dtype=float)
        knots = property_localisation.load_value_knots(
            Path(context.sources[property_localisation.QUANTILES_SOURCE]).read_bytes()
        )
        stream = tuple(context.params["stream"])
        keys = property_localisation.spine_household_keys(
            household,
            source=str(context.params["source"]),
            source_vintage=str(context.params["source_vintage"]),
        )
        support = (
            household[HIGH_VALUE_SUPPORT_COPY_COLUMN].to_numpy().astype(bool)
            if HIGH_VALUE_SUPPORT_COPY_COLUMN in household
            else None
        )
        u, uniform_receipt = property_localisation.owner_uniforms(
            spine_key=keys,
            region=household["region"].to_numpy(),
            tenure_type=household["tenure_type"].to_numpy(),
            # The spine's regional values (microcosm#1112) keep the value they
            # replaced; ranking on it keeps this node's draws unchanged.
            value=property_localisation.main_residence_rank_basis(
                household,
                declared=bool(context.params.get("rank_basis_declared", False)),
            ),
            weight=weights,
            stream=stream,
            support_copy=support,
            support_stratum_lower=float(context.params["support_stratum_lower"]),
        )
        new_value, new_wealth, receipt = property_localisation.localise_values(
            u=u,
            constituency_code=household["constituency_code"].to_numpy(),
            tenure_type=household["tenure_type"].to_numpy(),
            value=household["main_residence_value"].to_numpy(),
            property_wealth=household["property_wealth"].to_numpy(),
            knots=knots,
        )
        index = pd.Index(household["household_id"], name="household_id")
        household_ids = [int(value) for value in household["household_id"].to_numpy()]
        payload = {
            "schema": "microcosm.uk.property-quantile-draw.v1",
            "household_ids": household_ids,
            "u": [float(value) for value in u],
            "cap": knots.cap,
            "stream": list(stream),
            "method": property_localisation.METHOD_VERSION,
            "knots_inputs_sha256": knots.inputs_sha256,
            "basis_year": knots.basis_year,
            "rows_value_rewritten": receipt["rows_rewritten"],
        }
        return KernelResult(
            columns={
                ("household", "main_residence_value"): pd.Series(
                    new_value, index=index, name="main_residence_value", dtype="float64"
                ),
                ("household", "property_wealth"): pd.Series(
                    new_wealth, index=index, name="property_wealth", dtype="float64"
                ),
            },
            artifacts={"quantiles": canonical_json(payload)},
            receipt={
                **receipt,
                "uniforms": uniform_receipt,
                "weights": "unchanged",
                "household_mass": float(weights.sum()),
                "knots_inputs_sha256": knots.inputs_sha256,
                "basis_year": knots.basis_year,
                "stream": list(stream),
            },
        )


class UKPropertyBandKernel(_PopulationKernel):
    """Draw each England-and-Wales clone's council-tax band from its LSOA's VOA mix.

    microcosm#1240 fix 4: the band is read at the same uniform the value node
    placed the home value with, so a clone's value and band agree by
    construction. Rows outside England and Wales keep the spine's band.
    """

    ref = "uk.full.property.band@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC, numeric=Numeric.PLATFORM_BITWISE
    )

    def implementation_hash(self) -> str:
        return source_hash(type(self), property_localisation)

    def run(self, context: KernelContext) -> KernelResult:
        household = context.tables["household"]
        payload = json.loads(context.artifacts["quantiles"].payload)
        if payload.get("schema") != "microcosm.uk.property-quantile-draw.v1":
            raise ValueError(
                "Property band node received an unknown quantile artifact."
            )
        ids = [int(value) for value in household["household_id"].to_numpy()]
        if payload["household_ids"] != ids:
            raise ValueError(
                "Property quantile draws do not align with the pool households."
            )
        u = np.asarray(payload["u"], dtype=float)
        mix = property_localisation.load_band_mix(
            Path(context.sources[property_localisation.BAND_MIX_SOURCE]).read_bytes()
        )
        band_dtype = household["council_tax_band"].dtype
        tenure = household["tenure_type"].to_numpy()
        new_band, receipt = property_localisation.localise_bands(
            u=u,
            lsoa_code=household["lsoa_code"].to_numpy(),
            band=household["council_tax_band"].to_numpy(),
            tenure_type=tenure,
            mix=mix,
        )
        lsoa = household["lsoa_code"].to_numpy().astype(str)
        rewritten = (
            np.char.startswith(lsoa, "E") | np.char.startswith(lsoa, "W")
        ) & property_localisation.owner_mask(tenure)
        violations = property_localisation.monotone_violations(
            u[rewritten], new_band[rewritten], lsoa[rewritten]
        )
        if violations:
            raise ValueError("Property band draws are not monotone in the uniform.")
        index = pd.Index(household["household_id"], name="household_id")
        return KernelResult(
            columns={
                ("household", "council_tax_band"): pd.Series(
                    new_band, index=index, name="council_tax_band"
                ).astype(band_dtype),
            },
            receipt={
                **receipt,
                "monotone_violations": violations,
                "band_mix_sha256": mix.source_sha256,
            },
        )


class UKLocalAuthorityResolveKernel(_PopulationKernel):
    """Resolve the engine's ``local_authority`` input from the derived code.

    The shared derive step writes ``local_authority_code``; the engine input
    is the member name resolved fail-closed on the April 2023 roster
    (microcosm#953), exactly as the legacy ladder derive writes it. The
    engine itself is never imported here; the UK distribution gate checks
    the assigned names against the installed enum.
    """

    ref = "uk.full.geography.local_authority@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def implementation_hash(self) -> str:
        return source_hash(type(self), local_authority_input)

    def run(self, context: KernelContext) -> KernelResult:
        household = context.tables["household"]
        keys = resolve_local_authority_engine_keys(
            household[UK_LOCAL_AUTHORITY_CODE_COLUMN]
        )
        index = pd.Index(household["household_id"], name="household_id")
        column = UK_LOCAL_AUTHORITY_INPUT_COLUMN
        return KernelResult(
            columns={
                ("household", column): pd.Series(
                    keys, index=index, name=column, dtype=dtype_for_token("string")
                )
            },
            receipt={
                "households": int(len(keys)),
                "local_authority_vintage": UK_LOCAL_AUTHORITY_VINTAGE,
            },
        )


class UKPoolCheckpointKernel(_PopulationKernel):
    """Checkpoint the complete pool; refuse unless the shared atomic gate passed.

    The shared ``geography.gate@1`` result is a platform-scoped typed artifact
    (amendment 17), and a bitwise UK kernel consuming it as a typed input
    would be refused. This FILTER node consumes it and emits no typed
    artifact of its own, so that refusal is avoided here. It is only the
    typed-artifact edge that stops: the shared assignment kernels are
    ``platform_bitwise`` too, and their columns reach the UK target and
    calibration chain through population slices, which the executor already
    reads at platform scope. Whether the shared kernels can honestly declare
    ``bitwise`` (exact keyed draws, integer inverse CDF) is the open question
    recorded on microcosm#932.
    """

    ref = "uk.full.pool@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        structural=StructuralDelta.FILTER,
    )

    def run(self, context: KernelContext) -> KernelResult:
        receipt = {"scope": "complete_pool_checkpoint"}
        if "atomic_validation" in context.artifacts:
            validation = json.loads(context.artifacts["atomic_validation"].payload)
            if validation.get("outcome") != "pass":
                raise ValueError("Shared atomic geography validation did not pass.")
            receipt["atomic_validation"] = validation
        if "property_quantiles" in context.artifacts:
            quantiles = json.loads(context.artifacts["property_quantiles"].payload)
            household_ids = [
                int(value)
                for value in context.tables["household"]["household_id"].to_numpy()
            ]
            if quantiles.get("household_ids") != household_ids:
                raise ValueError(
                    "Property quantile draws do not align with the checkpointed pool."
                )
            receipt["property_localisation"] = {
                key: quantiles[key]
                for key in (
                    "cap",
                    "stream",
                    "method",
                    "knots_inputs_sha256",
                    "basis_year",
                    "rows_value_rewritten",
                )
            }
        person = context.tables[UK_NATIONAL_SCHEMA.person_entity]
        id_column = UK_NATIONAL_SCHEMA.person_id_column
        ids = pd.Index(person[id_column].to_numpy(copy=True), name=id_column)
        return KernelResult(
            keep=pd.Series(True, index=ids, dtype="bool"), receipt=receipt
        )


def uk_pool_validation_inputs(
    geography_assignment: str, property_localisation_mode: str = "none"
) -> tuple[ArtifactInput, ...]:
    """The typed edges that order the shared atomic gate (and the property draw) before the checkpoint."""
    if geography_assignment != "atomic":
        return ()
    from microcosm.build.graph_atomic_geography import (
        ATOMIC_GEOGRAPHY_VALIDATION_TYPE,
    )

    inputs = (
        ArtifactInput(
            "atomic_validation",
            "uk.full.geography.gate",
            "validation",
            ATOMIC_GEOGRAPHY_VALIDATION_TYPE,
        ),
    )
    if property_localisation_mode == "price_paid":
        inputs = (
            *inputs,
            ArtifactInput(
                "property_quantiles",
                "uk.full.property.value",
                "quantiles",
                PROPERTY_QUANTILE_TYPE,
            ),
        )
    return inputs


class UKGeographyGateKernel(_PopulationKernel):
    ref = "uk.full.geography_gate@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context: KernelContext) -> KernelResult:
        frame = context_frame(context)
        household = frame.table("household")
        gate = geography_ladder.uk_geography_ladder_gate(
            household, frame.weights_for("household").values
        )
        # Fail closed against the installed engine (microcosm#953): every
        # assigned local_authority member name must exist in this pin's
        # LocalAuthority enum. Only assigned keys are checked so a toy ladder
        # stays evaluable and an engine-free lane never imports the engine.
        if gate.passed and importlib.util.find_spec("policyengine_uk") is not None:
            verify_local_authority_engine_domain(household["local_authority"].unique())
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
    geography_assignment: str = "atomic",
    atomic_geography_definition: Mapping | None = None,
    source_vintage: str | None = None,
    identity_source: str = "frs",
    property_localisation_mode: str = "none",
) -> Graph:
    """Compose sampling → replication → identity → assignment → derivation → gate.

    ``property_localisation_mode="price_paid"`` (atomic assignment only) adds
    the microcosm#1240 value and band nodes between the local-authority
    resolve and the gates; ``"none"`` leaves the spine's values and bands on
    every clone.

    ``geography_assignment="atomic"`` (the default) keys every post-clone
    household by its lineage and runs the shared atomic-geography operators
    on the three UK support artifacts declared by ``atomic_geography_definition``
    (``uk_atomic_assignment_definition``). ``"legacy"`` keeps the sequential
    ladder draw for measurement builds only.
    """

    from .graph_kernels import UKClaimKernel

    cells = population_columns(graph, population)
    if type(n_clones) is not int or n_clones < 1:
        raise ValueError("Geographic replicate count K must be a positive integer.")
    if geography_assignment not in GEOGRAPHY_ASSIGNMENTS:
        raise ValueError(
            f"Geography assignment must be one of {GEOGRAPHY_ASSIGNMENTS}."
        )
    if geography_assignment == "atomic":
        if atomic_geography_definition is None:
            raise ValueError(
                "Atomic geography assignment requires the UK assignment definition."
            )
        if not isinstance(source_vintage, str) or not source_vintage:
            raise ValueError("Atomic geography assignment requires the FRS vintage.")
    elif atomic_geography_definition is not None:
        raise ValueError("Legacy geography assignment takes no atomic definition.")
    if property_localisation_mode not in PROPERTY_LOCALISATION_MODES:
        raise ValueError(
            f"Property localisation must be one of {PROPERTY_LOCALISATION_MODES}."
        )
    if property_localisation_mode != "none" and geography_assignment != "atomic":
        raise ValueError("Property localisation needs the atomic geography assignment.")
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
            "spine_gate_synthetic_smoke": bool(gate.params["synthetic_smoke"]),
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
    if geography_assignment == "legacy":
        geography_nodes, geography_sources = _legacy_geography_nodes(
            cells, common=common, seed=seed, constituency_vintage=constituency_vintage
        )
    else:
        geography_nodes, geography_sources = _atomic_geography_nodes(
            cells,
            common=common,
            seed=seed,
            definition=atomic_geography_definition,
            identity_source=identity_source,
            source_vintage=source_vintage,
            property_localisation_mode=property_localisation_mode,
        )
    nodes.extend(geography_nodes)
    declared = {s.name for s in graph.sources}
    sources = (
        *graph.sources,
        *(ref for ref in geography_sources if ref.name not in declared),
    )
    return replace(graph, nodes=tuple(nodes), sources=tuple(sources))


def _legacy_geography_nodes(
    cells: dict, *, common: Mapping, seed: int, constituency_vintage: str
) -> tuple[tuple[Node, ...], tuple[SourceRef, ...]]:
    """The sequential ladder draw, kept verbatim for measurement builds."""

    nodes = (
        Node(
            id="uk.full.locations",
            kernel=UKLocationDrawKernel.ref,
            population="uk.full.expand",
            inputs=(Slice("household", ("region",)),),
            sources=("uk_ladder",),
            params={"seed": seed, "constituency_vintage": constituency_vintage},
            artifact_outputs=(ArtifactOutput("locations", LOCATION_DRAW_TYPE),),
            description="Draw constituency then atomic area with the current sequential RNG.",
        ),
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
        ),
    )
    cells.update({("household", col): "string" for col in UK_GEOGRAPHY_LADDER_COLUMNS})
    gate = Node(
        id="uk.full.geography_gate",
        kernel=UKGeographyGateKernel.ref,
        population="uk.full.expand",
        inputs=population_slices(cells),
        params=dict(common),
        artifact_outputs=(ArtifactOutput("gate", GEOGRAPHY_GATE_TYPE),),
        description="Validate geography integrity before selected-target contributions.",
    )
    ladder = SourceRef(
        "uk_ladder", "raw-bytes-v1", "Full UK atomic-area ladder with pinned vintages."
    )
    return (*nodes, gate), (ladder,)


def _atomic_geography_nodes(
    cells: dict,
    *,
    common: Mapping,
    seed: int,
    definition: Mapping,
    identity_source: str,
    source_vintage: str,
    property_localisation_mode: str = "none",
) -> tuple[tuple[Node, ...], tuple[SourceRef, ...]]:
    """Identity-keyed assignment on the shared operators (release path order)."""

    from microcosm.build.atomic_geography import validate_assignment_spec
    from microcosm.build.graph_atomic_geography import atomic_geography_nodes

    spec = validate_assignment_spec(definition)
    if spec["identity"] != [atomic_area_support.IDENTITY_COLUMN]:
        raise ValueError("UK atomic assignment must key on the geography identity.")
    if spec["stream"][3] != seed:
        raise ValueError("UK atomic assignment seed differs from the build seed.")
    if {system["source"] for system in spec["systems"]} != set(
        atomic_area_support.SOURCES.values()
    ):
        raise ValueError("UK atomic assignment must declare the three UK supports.")
    lineage_inputs = tuple(
        sorted(
            column
            for column in (*UK_HOUSEHOLD_LINEAGE_COLUMNS, *UK_OPTIONAL_LINEAGE_COLUMNS)
            if ("household", column) in cells
        )
    )
    identity = Node(
        id="uk.full.identity",
        kernel=UKGeographyIdentityKernel.ref,
        population="uk.full.expand",
        inputs=(Slice("household", lineage_inputs),),
        outputs=(Owned("household", atomic_area_support.IDENTITY_COLUMN, "string"),),
        params={
            **common,
            "source": identity_source,
            "source_vintage": source_vintage,
        },
        description="Key every post-clone household by its source id and ordered clone path.",
    )
    cells[("household", atomic_area_support.IDENTITY_COLUMN)] = "string"
    shared = atomic_geography_nodes(
        spec,
        tuple(Owned(e, c, dtype) for (e, c), dtype in sorted(cells.items())),
        base="uk.full.expand",
        prefix="uk.full.geography",
        emit_validation_artifact=True,
    )
    for node in shared:
        cells.update({(o.entity, o.column): o.dtype for o in node.outputs})
    # The engine's local_authority enum input (microcosm#953) is UK-side: the
    # shared derive writes the GSS code, this node resolves the member name
    # the export gate and the engine require, between derive and the gates.
    local_authority = Node(
        id="uk.full.geography.local_authority",
        kernel=UKLocalAuthorityResolveKernel.ref,
        population="uk.full.expand",
        inputs=(Slice("household", (UK_LOCAL_AUTHORITY_CODE_COLUMN,)),),
        outputs=(Owned("household", UK_LOCAL_AUTHORITY_INPUT_COLUMN, "string"),),
        params=dict(common),
        description="Resolve the engine local_authority input from the derived GSS code.",
    )
    cells[("household", UK_LOCAL_AUTHORITY_INPUT_COLUMN)] = "string"
    property_nodes: tuple[Node, ...] = ()
    property_sources: tuple[SourceRef, ...] = ()
    if property_localisation_mode == "price_paid":
        property_nodes, property_sources = _property_localisation_nodes(
            cells,
            common=common,
            seed=seed,
            identity_source=identity_source,
            source_vintage=source_vintage,
            lineage_inputs=lineage_inputs,
        )
    # The shared gate's typed validation artifact is platform-scoped, so it is
    # consumed by the pool checkpoint (uk.full.pool, no typed outputs), not by
    # this distribution gate, whose own artifact feeds the bitwise UK chain.
    gate = Node(
        id="uk.full.geography_gate",
        kernel=UKGeographyGateKernel.ref,
        population="uk.full.expand",
        inputs=population_slices(cells),
        params=dict(common),
        artifact_outputs=(ArtifactOutput("gate", GEOGRAPHY_GATE_TYPE),),
        description="Validate the UK geography distribution on the identity-keyed assignment.",
    )
    sources = (
        SourceRef(
            "uk_ladder",
            "raw-bytes-v1",
            "Full UK atomic-area ladder with pinned vintages.",
        ),
        *(
            SourceRef(
                atomic_area_support.SOURCES[system],
                "raw-bytes-v1",
                f"UK atomic-area support artifact for {system}.",
            )
            for system in atomic_area_support.SYSTEMS
        ),
        *property_sources,
    )
    return (
        identity,
        *shared[:-1],
        local_authority,
        *property_nodes,
        shared[-1],
        gate,
    ), sources


def _property_localisation_nodes(
    cells: dict,
    *,
    common: Mapping,
    seed: int,
    identity_source: str,
    source_vintage: str,
    lineage_inputs: tuple[str, ...],
) -> tuple[tuple[Node, ...], tuple[SourceRef, ...]]:
    """The microcosm#1240 value and band nodes, after derivation and before the gates."""
    for column in (*_PROPERTY_VALUE_COLUMNS, *_PROPERTY_BAND_COLUMNS):
        if ("household", column) not in cells:
            raise ValueError(
                f"Property localisation needs the household column {column!r} on the pool."
            )
    for column in ("constituency_code", "lsoa_code"):
        if ("household", column) not in cells:
            raise ValueError(
                f"Property localisation runs after geography derives {column!r}."
            )
    stream = (
        "sha256-u53-v1",
        property_localisation.PROPERTY_STREAM_EXPERIMENT,
        0,
        int(seed),
    )
    # The regional home values stage (microcosm#1112) keeps the value it
    # replaced; the node must receive it to rank owners as it did before.
    rank_basis_declared = (
        "household",
        property_localisation.RANK_BASIS_COLUMN,
    ) in cells
    rank_basis_inputs = (
        (property_localisation.RANK_BASIS_COLUMN,) if rank_basis_declared else ()
    )
    value = Node(
        id="uk.full.property.value",
        kernel=UKPropertyValueKernel.ref,
        population="uk.full.expand",
        inputs=(
            Slice(
                "household",
                tuple(
                    sorted(
                        {
                            *lineage_inputs,
                            *_PROPERTY_VALUE_COLUMNS,
                            *rank_basis_inputs,
                            "constituency_code",
                        }
                    )
                ),
            ),
        ),
        outputs=(
            Owned(
                "household",
                "main_residence_value",
                cells[("household", "main_residence_value")],
                rewrite=True,
            ),
            Owned(
                "household",
                "property_wealth",
                cells[("household", "property_wealth")],
                rewrite=True,
            ),
        ),
        sources=(property_localisation.QUANTILES_SOURCE,),
        params={
            **common,
            "source": identity_source,
            "source_vintage": source_vintage,
            "stream": stream,
            "support_stratum_lower": property_localisation.SUPPORT_STRATUM_LOWER,
            "method": property_localisation.METHOD_VERSION,
            "rank_basis_declared": rank_basis_declared,
        },
        artifact_outputs=(ArtifactOutput("quantiles", PROPERTY_QUANTILE_TYPE),),
        description="Read each owner clone's home value off its constituency's uprated sale distribution at its region rank.",
    )
    band = Node(
        id="uk.full.property.band",
        kernel=UKPropertyBandKernel.ref,
        population="uk.full.expand",
        inputs=(
            Slice(
                "household",
                tuple(sorted({*_PROPERTY_BAND_COLUMNS, "lsoa_code", "tenure_type"})),
            ),
        ),
        outputs=(
            Owned(
                "household",
                "council_tax_band",
                cells[("household", "council_tax_band")],
                rewrite=True,
            ),
        ),
        sources=(property_localisation.BAND_MIX_SOURCE,),
        params={**common, "method": property_localisation.METHOD_VERSION},
        artifact_inputs=(
            ArtifactInput("quantiles", value.id, "quantiles", PROPERTY_QUANTILE_TYPE),
        ),
        description="Draw each England-and-Wales clone's council-tax band from its LSOA's VOA mix at the same uniform.",
    )
    sources = (
        SourceRef(
            property_localisation.QUANTILES_SOURCE,
            "raw-bytes-v1",
            "Per-constituency quantile knots of the uprated Land Registry sale values.",
        ),
        SourceRef(
            property_localisation.BAND_MIX_SOURCE,
            "raw-bytes-v1",
            "Per-LSOA cumulative VOA council-tax band shares.",
        ),
    )
    return (value, band), sources


def register_uk_population_kernels(registry: KernelRegistry) -> None:
    from microcosm.build.graph_atomic_geography import (
        register_atomic_geography_kernels,
    )

    for kernel in (
        UKFamilySampleKernel(),
        UKSampleNormalizationKernel(),
        UKGeographicExpansionKernel(),
        UKGeographyIdentityKernel(),
        UKPropertyValueKernel(),
        UKPropertyBandKernel(),
        UKLocalAuthorityResolveKernel(),
        UKLocationDrawKernel(),
        UKGeographyMappingKernel(),
        UKGeographyGateKernel(),
        UKPoolCheckpointKernel(),
    ):
        registry.register(kernel)
    register_atomic_geography_kernels(registry)
