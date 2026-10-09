"""Authenticated donor CREATE and declared transport rewrites.

Country data enters only through node parameters and declared facts. The
CREATE installs destination-scale design weights before the graph captures
its design anchor; the records retain their donor support provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import fsum

import numpy as np
import pandas as pd

import microcosm.frame.adapters.axiom as axiom_module
import microcosm.frame.adapters.policyengine_us_concepts as donor_mapping_module
import microcosm.frame.bundle as bundle_module
import microcosm.frame.concept_mapping as mapping_module
import microcosm.frame.concepts as concepts_module
import microcosm.frame.materialize as materialize_module
import microcosm.frame.scaling as scaling_module
import microcosm.frame.schema as schema_module
import microcosm.frame.transport as transport_module
import microcosm.frame.unit_construction as units_module
import microcosm.frame.weights as weights_module
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.adapters.axiom import NZ_SCHEMA
from microcosm.frame.concepts import (
    concept_for_column,
    concept_schema_sha256,
    split_for_transport,
)
from microcosm.frame.transport import (
    currency_bridge,
    derive_transport_seed,
    quantile_map,
    read_populace_us_donor,
)
from microcosm.frame.unit_construction import BenefitUnitRule, build_benefit_units
from microcosm.graph import (
    ROWS_ALL,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
    StructuralDelta,
    source_hash,
)

from . import graph_inputs, target_kernels
from .artifact_types import TARGET_SURFACE_TYPE
from .graph_inputs import (
    canonical_document_param,
    require_params,
    sha256_param,
    string_param,
)
from .target_kernels import compile_target_surface, decode_target_surface

__all__ = [
    "TRANSPORT_BOUNDARY",
    "TRANSPORT_CREATE",
    "TRANSPORT_CURRENCY",
    "TRANSPORT_QUANTILE_MAP",
    "TransportBoundaryKernel",
    "TransportCreateKernel",
    "TransportCurrencyKernel",
    "TransportQuantileMapKernel",
    "register_population_kernels",
]


def _whole_rows(context: KernelContext, ref: str) -> None:
    if any(item.rows != ROWS_ALL for item in context.node.inputs):
        raise ValueError(f"{ref} requires whole-row slices.")


def _column_result(
    context: KernelContext, ref: str, values: Mapping[tuple[str, str], pd.Series]
) -> KernelResult:
    declared = {(item.entity, item.column) for item in context.node.outputs}
    if declared != set(values):
        raise ValueError(f"{ref} outputs must equal its declared rewritten columns.")
    columns = {}
    for (entity, name), series in values.items():
        table = context.tables[entity]
        ids = table[f"{entity}_id"]
        columns[entity, name] = pd.Series(
            series.to_numpy(copy=True),
            index=pd.Index(ids.to_numpy(copy=True), name=f"{entity}_id"),
            name=name,
            dtype=series.dtype,
        )
    return KernelResult(columns=columns)


class TransportCreateKernel(KernelBase):
    """``transport.create@1``: verified support records at destination mass."""

    ref = "transport.create@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.KEYED,
        structural=StructuralDelta.CREATE,
        dependencies=("numpy", "pandas", "tables"),
    )
    _required = frozenset(
        {
            "donor_source",
            "facts_source",
            "donor_sha256",
            "donor_size",
            "donor_pin_sha256",
            "unit_rule",
            "unit_rule_sha256",
            "mass_reference",
            "mass_reference_sha256",
            "country",
            "concept_schema_sha256",
            "seed_stream",
        }
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            target_kernels,
            transport_module,
            units_module,
            concepts_module,
            mapping_module,
            donor_mapping_module,
            materialize_module,
            scaling_module,
            axiom_module,
            bundle_module,
            schema_module,
            weights_module,
            *target_kernels._COMPILE_MODULES,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context,
            self.ref,
            required=self._required,
            optional=frozenset({"facts_sha256"}),
        )
        donor_source = string_param(context, self.ref, "donor_source")
        facts_source = string_param(context, self.ref, "facts_source")
        if (
            donor_source == facts_source
            or set(context.node.sources) != {donor_source, facts_source}
            or any(name not in context.sources for name in context.node.sources)
        ):
            raise ValueError(f"{self.ref} reads exactly its donor and facts sources.")
        for name in ("donor_pin_sha256", "unit_rule_sha256", "concept_schema_sha256"):
            sha256_param(context, self.ref, name)
        if context.params["concept_schema_sha256"] != concept_schema_sha256():
            raise ValueError("CREATE concept schema SHA-256 differs from this reader.")
        country = string_param(context, self.ref, "country")
        surface_bytes, registry = compile_target_surface(
            context.sources[facts_source],
            canonical_document_param(context, self.ref, "mass_reference"),
            country=country,
            references_sha256=sha256_param(context, self.ref, "mass_reference_sha256"),
            expected_facts_sha256=(
                sha256_param(context, self.ref, "facts_sha256")
                if "facts_sha256" in context.params
                else None
            ),
        )
        if len(registry.specs) != 1:
            raise ValueError(
                "CREATE requires exactly one destination population reference."
            )
        mass_spec = registry.specs[0]
        if (
            mass_spec.entity != "person"
            or mass_spec.filter
            or mass_spec.metadata.get("ledger_measure_unit") != "count"
        ):
            raise ValueError(
                "CREATE destination population reference must be an unfiltered person count fact."
            )
        mass = float(mass_spec.value)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("Destination population mass must be finite and positive.")
        donor = read_populace_us_donor(
            context.sources[donor_source],
            sha256=sha256_param(context, self.ref, "donor_sha256"),
            size=context.params["donor_size"],
        )
        tables, dropped = split_for_transport(donor.tables)
        # pandas 3's inferred text dtype spells itself "str"; graph columns
        # have an explicit "string" token. Keep the concept values intact
        # while fixing the portable dtype at this boundary.
        for entity, table in tables.items():
            for column in table:
                item = concept_for_column(entity, column)
                if item is not None and item.dtype == "str":
                    table[column] = table[column].astype("string")
        rule = BenefitUnitRule.from_dict(
            canonical_document_param(context, self.ref, "unit_rule")
        )
        if rule.entity != "family":
            raise ValueError("CREATE's declared schema requires family benefit units.")
        family, membership = build_benefit_units(
            tables["person"], tables["household"], rule
        )
        tables["family"] = family
        person, household = tables["person"], tables["household"]
        person[rule.membership_column] = membership
        person["take_up_seed"] = derive_transport_seed(
            donor.source_person_ids, string_param(context, self.ref, "seed_stream")
        )
        household["donor_support_stratum"] = pd.array(
            donor.support_strata, dtype="string"
        )
        household_ids = household["household_id"]
        donor_weights = pd.Series(donor.weights, index=household_ids)
        person_weights = person["person_household_id"].map(donor_weights)
        donor_mass = fsum(person_weights.to_numpy(dtype=np.float64))
        if not np.isfinite(donor_mass) or donor_mass <= 0:
            raise ValueError("Donor person mass must be finite and positive.")
        scale = mass / donor_mass
        weights = Weights(donor.weights * scale, WeightKind.DESIGN)
        strata = pd.Series(
            person["person_household_id"]
            .map(pd.Series(donor.support_strata, index=household_ids))
            .to_numpy(copy=True),
            index=person.index,
            name="stratum",
        )
        surface = decode_target_surface(surface_bytes)
        frame = Frame(
            tables,
            NZ_SCHEMA,
            {"household": weights},
            strata,
            metadata={
                "content_basis": donor.content_basis.value,
                "donor_country": donor.donor_country,
                "country": country,
                "donor_currency": donor.currency,
            },
        )
        return KernelResult(
            frame=frame,
            receipt={
                "content_basis": donor.content_basis.value,
                "donor_country": donor.donor_country,
                "country": country,
                "dropped_columns": {
                    entity: list(names) for entity, names in dropped.items()
                },
                "dropped_parent_cycle_edges": donor.dropped_parent_cycle_edges,
                "donor_mass": donor_mass,
                "destination_mass": mass,
                "design_scale": scale,
                "n_persons": len(person),
                "n_households": len(household),
                "n_units": len(family),
                "mass_reference": registry.specs[0].name,
                "facts": dict(surface.facts),
            },
        )


class TransportBoundaryKernel(KernelBase):
    """``transport.boundary@1``: keep every person and conserve mass."""

    ref = "transport.boundary@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            _whole_rows,
            graph_inputs,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(context, self.ref, required=frozenset())
        _whole_rows(context, self.ref)
        if not any(item.entity == "person" for item in context.node.inputs):
            raise ValueError(
                "Boundary FILTER requires a declared person data-column slice."
            )
        if context.node.mass != "conserve":
            raise ValueError("Boundary FILTER requires mass='conserve'.")
        ids = context.tables["person"]["person_id"]
        return KernelResult(
            keep=pd.Series(True, index=pd.Index(ids.to_numpy(), name="person_id")),
            receipt={"kept_persons": len(ids)},
        )


class TransportCurrencyKernel(KernelBase):
    """``transport.currency@1``: one declared currency multiplication."""

    ref = "transport.currency@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            _whole_rows,
            _column_result,
            graph_inputs,
            transport_module,
            concepts_module,
            scaling_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context,
            self.ref,
            required=frozenset({"columns", "rate", "currency_sha256"}),
        )
        _whole_rows(context, self.ref)
        sha256_param(context, self.ref, "currency_sha256")
        selected = canonical_document_param(context, self.ref, "columns")
        if not selected or any(
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            for names in selected.values()
        ):
            raise ValueError(
                "Currency columns must name nonempty lists of money concepts."
            )
        tables = currency_bridge(context.tables, selected, context.params["rate"])
        return _column_result(
            context,
            self.ref,
            {
                (entity, name): tables[entity][name]
                for entity, names in selected.items()
                for name in names
            },
        )


def _surface_bands(document: Mapping[str, object], surface) -> object:
    """Resolve band shares from the exact declared population facts."""
    values = {spec.name: float(spec.value) for spec in surface.registry.specs}

    def rows(raw):
        if not isinstance(raw, list) or not raw:
            raise ValueError("Quantile bands require a nonempty list.")
        if any(
            not isinstance(row, dict) or set(row) != {"lower", "upper", "reference"}
            for row in raw
        ):
            raise ValueError("Each quantile band declares lower, upper and reference.")
        names = [row["reference"] for row in raw]
        if any(not isinstance(name, str) or name not in values for name in names):
            raise ValueError("A quantile band's reference is absent from its surface.")
        if len(set(names)) != len(names):
            raise ValueError("Quantile band references must be distinct.")
        counts = [values[name] for name in names]
        if any(not np.isfinite(value) or value < 0 for value in counts):
            raise ValueError(
                "Quantile band population counts must be finite and nonnegative."
            )
        total = fsum(counts)
        if total <= 0 or not np.isfinite(total):
            raise ValueError(
                "Quantile band population total must be positive and finite."
            )
        return [
            {"lower": row["lower"], "upper": row["upper"], "share": count / total}
            for row, count in zip(raw, counts, strict=True)
        ]

    if set(document) == {"bands"}:
        return rows(document["bands"])
    if set(document) == {"positive", "negative"}:
        return {sign: rows(raw) for sign, raw in document.items()}
    if set(document) == {"components"}:
        components = document["components"]
        if not isinstance(components, list) or not components:
            raise ValueError("Component bands require a nonempty list.")
        result = {}
        for component in components:
            if not isinstance(component, dict) or set(component) != {"value", "bands"}:
                raise ValueError("Each component declares value and bands.")
            value = component["value"]
            if (
                not isinstance(value, (str, int))
                or isinstance(value, bool)
                or value in result
            ):
                raise ValueError(
                    "Band component values must be distinct string or integer codes."
                )
            result[value] = rows(component["bands"])
        return {"components": result}
    raise ValueError("Quantile bands declare bands or positive and negative rows.")


class TransportQuantileMapKernel(KernelBase):
    """``transport.quantile_map@1``: ranks mapped to compiled fact bands.

    ``aggregate_entity`` requests household totals and pro rata member
    allocation. ``component_column`` requests independent distributions
    within a sliced column (for example a household region).
    """

    ref = "transport.quantile_map@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            _surface_bands,
            _whole_rows,
            _column_result,
            graph_inputs,
            target_kernels,
            transport_module,
            *target_kernels._COMPILE_MODULES,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context,
            self.ref,
            required=frozenset(
                {"entity", "column", "bands", "interpolation", "bands_sha256"}
            ),
            optional=frozenset({"component_column", "aggregate_entity"}),
        )
        _whole_rows(context, self.ref)
        sha256_param(context, self.ref, "bands_sha256")
        artifact = context.artifacts.get("surface")
        if artifact is None or artifact.type != TARGET_SURFACE_TYPE:
            raise ValueError(
                "Quantile map requires a target surface under alias 'surface'."
            )
        bands = _surface_bands(
            canonical_document_param(context, self.ref, "bands"),
            decode_target_surface(artifact.payload),
        )
        interpolation = canonical_document_param(context, self.ref, "interpolation")
        if interpolation == {"method": "uniform"}:
            interpolation = "uniform"
        entity = string_param(context, self.ref, "entity")
        column = string_param(context, self.ref, "column")
        table = context.tables[entity]
        component = (
            table[string_param(context, self.ref, "component_column")].to_numpy()
            if "component_column" in context.params
            else None
        )
        group = component
        if "aggregate_entity" in context.params:
            aggregate = string_param(context, self.ref, "aggregate_entity")
            if entity != "person":
                raise ValueError(
                    "Aggregate quantile maps require person membership rows."
                )
            group = {"entity_ids": table[f"person_{aggregate}_id"].to_numpy()}
            if component is not None:
                group["components"] = component
                membership_components = (
                    pd.DataFrame(
                        {"entity": group["entity_ids"], "component": component}
                    )
                    .groupby("entity")["component"]
                    .nunique(dropna=False)
                )
                if (membership_components > 1).any():
                    raise ValueError(
                        "Aggregate members must repeat one component code."
                    )
        values = table[column].to_numpy()
        weights = context.weights[entity].values
        if isinstance(bands, dict) and set(bands) == {"components"}:
            if component is None or pd.isna(component).any():
                raise ValueError(
                    "Component-specific bands require a nonmissing component_column."
                )
            if set(component) != set(bands["components"]):
                raise ValueError(
                    "Component bands must exactly cover the sliced component codes."
                )
            mapped = np.empty(len(values), dtype=np.float64)
            for value, component_bands in bands["components"].items():
                selected = component == value
                component_group = (
                    {"entity_ids": group["entity_ids"][selected]}
                    if isinstance(group, dict)
                    else None
                )
                mapped[selected] = quantile_map(
                    values[selected],
                    weights[selected],
                    component_bands,
                    interpolation=interpolation,
                    group=component_group,
                )
        else:
            mapped = quantile_map(
                values,
                weights,
                bands,
                interpolation=interpolation,
                group=group,
            )
        return _column_result(
            context,
            self.ref,
            {(entity, column): pd.Series(mapped, index=table.index, name=column)},
        )


TRANSPORT_CREATE = TransportCreateKernel()
TRANSPORT_BOUNDARY = TransportBoundaryKernel()
TRANSPORT_CURRENCY = TransportCurrencyKernel()
TRANSPORT_QUANTILE_MAP = TransportQuantileMapKernel()


def register_population_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Register the population kernels explicitly and idempotently."""
    for kernel in (
        TRANSPORT_CREATE,
        TRANSPORT_BOUNDARY,
        TRANSPORT_CURRENCY,
        TRANSPORT_QUANTILE_MAP,
    ):
        registry.register(kernel)
    return registry
