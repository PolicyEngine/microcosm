"""Cross-reference closure for normalized country bundles."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .canonical import canonical_json_bytes
from .identity_contracts import IdentityContractError, resolve_seed_site_bindings
from .model import (
    ArtifactSpec,
    ColumnSpec,
    EntitySpec,
    FrozenMap,
    ScopeSpec,
    SeedSiteBinding,
    SymbolRef,
    freeze_json,
)
from .seeds import LEGACY_V1_PROTOCOL, SeedProtocol
from .take_up_semantics import validate_take_up_semantics
from .typed_closure import TypedClosureError, resolve_typed_closure
from .vintage_authorities import (
    VintageAuthorityError,
    resolve_vintage_authorities,
)


class SpecResolutionError(ValueError):
    """A typed reference is duplicate, dangling, or structurally invalid."""


@dataclass(frozen=True, slots=True)
class KernelRegistry:
    """Closed kernel-reference namespace with explicit implementation status.

    ``ids`` contains every contract id accepted during cross-reference
    resolution.  ``implemented_ids`` is the strict subset allowed to back an
    executable producer node.  Contract-only ids let a walking-skeleton bundle
    prove schema expressibility without pretending that F0 has an executor
    binding for the named kernel.
    """

    ids: frozenset[str]
    implemented_ids: frozenset[str]
    implementation_sha256: str

    @classmethod
    def from_ids(
        cls,
        ids: Iterable[str],
        *,
        contract_only_ids: Iterable[str] = (),
    ) -> KernelRegistry:
        """Build a registry, treating ``ids`` as implemented by default."""

        implemented = frozenset(_strip_prefix(item, "kernel") for item in ids)
        contract_only = frozenset(
            _strip_prefix(item, "kernel") for item in contract_only_ids
        )
        overlap = sorted(implemented & contract_only)
        if overlap:
            raise ValueError(
                "kernel ids cannot be both implemented and contract-only: "
                f"{overlap!r}"
            )
        digest = hashlib.sha256(canonical_json_bytes(sorted(implemented))).hexdigest()
        return cls(implemented | contract_only, implemented, digest)

    def contains(self, value: str) -> bool:
        return _strip_prefix(value, "kernel") in self.ids

    def has_implementation(self, value: str) -> bool:
        """Return whether ``value`` has an executable F0 implementation pin."""

        return _strip_prefix(value, "kernel") in self.implemented_ids

    @property
    def contract_only_ids(self) -> frozenset[str]:
        """Accepted references that cannot back executable producer nodes."""

        return self.ids - self.implemented_ids


# The implementation namespace is compiler-owned code inventory, not country
# configuration.  Keeping it explicit and independent of bundle contents is
# what makes an unknown ``kernel:`` reference fail closed: a bundle cannot
# make a misspelling valid merely by declaring it.  These are the implemented
# generation-0 kernels referenced by the packaged US F0 bundle.
F0_IMPLEMENTED_KERNEL_IDS = frozenset(
    {
        "acs_pums_earnings_universe",
        "acs_transfer",
        "assert_exact_k_support",
        "assign_us_puma_ladder",
        "binary_assignment",
        "build_acs_pums_unit_frame",
        "by_origin_battery",
        "calibrate",
        "calibrate_exact_k_ladder",
        "calibrate_l0_refit",
        "count_calibration",
        "engine_default",
        "exact_k_ladder_manifest_payload",
        "exact_k_pcg64_rng",
        "housing_imputed_transfer",
        "housing_measured_map",
        "impute_us_housing_assistance_to_puf_support",
        "joint_count_calibration",
        "load_acs_2022_rent_donor",
        "load_asec_raw_stage_checkpoint",
        "load_congressional_district_vintage_crosswalk",
        "load_puf_tax_unit_donor",
        "load_us_puma_ladder",
        "marketplace_assignment",
        "marketplace_measured_map",
        "medicare_measured_map",
        "primary_puf_qrf",
        "puf_capital_gains_tail_post_selection_gate",
        "refit_l0_selection",
        "regime_gated_qrf",
        "select_exact_k",
        "seeded_rate",
        "snap_probability_seed",
        "source_finalizer",
        "stacked_completeness_gate",
        "ssi_delivery_gate",
        "ssi_probability_seed",
        "tail_concentration_gate",
        "torch_adam",
        "torch_manual_seed",
        "us_puma_ladder_gate",
        "with_us_adult_care_inputs",
        "with_us_child_support_inputs",
        "with_us_childcare_inputs",
        "with_us_disability_benefits",
        "with_us_education_inputs",
        "with_us_energy_subsidy_input",
        "with_us_immigration_inputs",
        "with_us_medicare_take_up_input",
        "with_us_pregnancy_inputs",
        "with_us_prior_year_income_inputs",
        "with_us_retirement_contribution_inputs",
        "with_us_retirement_distribution_inputs",
        "with_us_weeks_unemployed",
        "with_us_wic_claim_input",
        "with_us_workers_compensation",
        "weighted_qrf_imputed_transfer",
    }
)

# UK and BE use these reviewed ids to prove that their shared-core bundle
# shapes resolve through the same compiler.  F0 does not yet bind them into the
# generic executor: the Belgian implementations are absent, and the existing
# UK functions have no compiler-owned producer binding or per-kernel
# implementation attestation.  They therefore close references but must never
# be presented as executable producer kernels.
F0_CONTRACT_ONLY_KERNEL_IDS = frozenset(
    {
        "assign_uk_geography_ladder",
        "be_commune_geography_gate",
        "clone_assign_communes",
        "load_uk_national_frame",
        "silc_load",
        "uk_geography_ladder_gate",
    }
)

# Compatibility name: every kernel contract accepted by the F0 resolver.
F0_KERNEL_IDS = F0_IMPLEMENTED_KERNEL_IDS | F0_CONTRACT_ONLY_KERNEL_IDS


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    references: tuple[SymbolRef, ...]
    entities: tuple[EntitySpec, ...]
    artifacts: tuple[ArtifactSpec, ...]
    scopes: tuple[ScopeSpec, ...]
    columns: tuple[ColumnSpec, ...]
    vintage_authorities: FrozenMap
    generated_authorities: FrozenMap
    seed_protocol: SeedProtocol
    seed_site_bindings: tuple[SeedSiteBinding, ...]


def _strip_prefix(value: str, namespace: str) -> str:
    prefix = f"{namespace}:"
    return value[len(prefix) :] if value.startswith(prefix) else value


F0_KERNEL_REGISTRY = KernelRegistry.from_ids(
    F0_IMPLEMENTED_KERNEL_IDS,
    contract_only_ids=F0_CONTRACT_ONLY_KERNEL_IDS,
)


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SpecResolutionError(f"{location}: expected mapping")
    return value


def _array(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise SpecResolutionError(f"{location}: expected array")
    return value


def _declare_array_ids(
    value: object,
    *,
    location: str,
    field: str = "id",
) -> frozenset[str]:
    declared: set[str] = set()
    for index, row in enumerate(_array(value, location)):
        item = _mapping(row, f"{location}/{index}")
        candidate = item.get(field)
        if not isinstance(candidate, str):
            raise SpecResolutionError(
                f"{location}/{index}/{field}: expected identifier"
            )
        if candidate in declared:
            raise SpecResolutionError(
                f"{location}/{index}/{field}: duplicate id {candidate!r}"
            )
        declared.add(candidate)
    return frozenset(declared)


def _declare_mapping_ids(value: object, *, location: str) -> frozenset[str]:
    mapping = _mapping(value, location)
    return frozenset(mapping)


def _unique_strings(value: object, *, location: str) -> tuple[str, ...]:
    """Return an identifier-like array while rejecting ambiguity."""

    result: list[str] = []
    for index, item in enumerate(_array(value, location)):
        if not isinstance(item, str):
            raise SpecResolutionError(f"{location}/{index}: expected string")
        if item in result:
            raise SpecResolutionError(f"{location}/{index}: duplicate value {item!r}")
        result.append(item)
    return tuple(result)


def _validate_imputation_concept_coverage(
    imputation: Mapping[str, object],
    *,
    families: object,
) -> None:
    """Require each target's family predictors to cover its named concepts.

    A concept is covered only when *all* columns in its registry entry occur
    in the union of the family's referenced predictor blocks.  F0's one
    approved exception is an explicit F-P record: the target must point to
    that record, the record must list the target, and its per-target missing
    concept set must equal the resolver's result exactly.  This makes the
    migration waiver auditable and self-expiring when predictor sets change.
    """

    concept_rows = _mapping(imputation.get("concepts", {}), "imputation/concepts")
    concept_columns: dict[str, frozenset[str]] = {}
    for concept_id, value in concept_rows.items():
        columns = _unique_strings(
            value,
            location=f"imputation/concepts/{concept_id}",
        )
        if not columns:
            raise SpecResolutionError(
                f"imputation/concepts/{concept_id}: at least one predictor "
                "column is required"
            )
        concept_columns[concept_id] = frozenset(columns)

    block_rows = _mapping(
        imputation.get("predictor_blocks", {}), "imputation/predictor_blocks"
    )
    predictor_blocks: dict[str, frozenset[str]] = {}
    for block_id, value in block_rows.items():
        block = _mapping(value, f"imputation/predictor_blocks/{block_id}")
        columns = _unique_strings(
            block.get("columns", []),
            location=f"imputation/predictor_blocks/{block_id}/columns",
        )
        if not columns:
            raise SpecResolutionError(
                f"imputation/predictor_blocks/{block_id}/columns: at least "
                "one predictor column is required"
            )
        predictor_blocks[block_id] = frozenset(columns)

    waiver_targets: dict[str, frozenset[str]] = {}
    waiver_missing: dict[str, dict[str, frozenset[str]]] = {}
    for waiver_index, value in enumerate(
        _array(imputation.get("waiver_records", []), "imputation/waiver_records")
    ):
        location = f"imputation/waiver_records/{waiver_index}"
        waiver = _mapping(value, location)
        waiver_id = waiver.get("id")
        if not isinstance(waiver_id, str):
            raise SpecResolutionError(f"{location}/id: expected identifier")
        if waiver_id in waiver_targets:
            raise SpecResolutionError(
                f"{location}/id: duplicate waiver id {waiver_id!r}"
            )
        if (
            waiver.get("code") != "F-P"
            or waiver.get("marker") != "F-P: eligibility concepts absent"
            or waiver.get("reason") != "eligibility_concepts_absent"
            or waiver.get("coverage_status")
            != "required_concepts_not_covered_by_current_predictors"
        ):
            raise SpecResolutionError(
                f"{location}: unsupported eligibility-concept waiver contract"
            )
        targets = frozenset(
            _unique_strings(waiver.get("targets", []), location=f"{location}/targets")
        )
        if not targets:
            raise SpecResolutionError(f"{location}/targets: must not be empty")
        missing_rows = _mapping(
            waiver.get("missing_concepts_by_target", {}),
            f"{location}/missing_concepts_by_target",
        )
        missing_by_target: dict[str, frozenset[str]] = {}
        for target_name, missing_value in missing_rows.items():
            missing = frozenset(
                _unique_strings(
                    missing_value,
                    location=(f"{location}/missing_concepts_by_target/{target_name}"),
                )
            )
            if not missing:
                raise SpecResolutionError(
                    f"{location}/missing_concepts_by_target/{target_name}: "
                    "must not be empty"
                )
            unknown = sorted(missing - concept_columns.keys())
            if unknown:
                raise SpecResolutionError(
                    f"{location}/missing_concepts_by_target/{target_name}: "
                    f"dangling concept references {unknown!r}"
                )
            missing_by_target[target_name] = missing
        if set(missing_by_target) != targets:
            raise SpecResolutionError(
                f"{location}: targets must exactly equal "
                "missing_concepts_by_target keys"
            )
        declared_union = frozenset(
            _unique_strings(
                waiver.get("requires_concepts", []),
                location=f"{location}/requires_concepts",
            )
        )
        actual_union = frozenset().union(*missing_by_target.values())
        if not declared_union or declared_union != actual_union:
            raise SpecResolutionError(
                f"{location}/requires_concepts: must exactly equal the union "
                "of missing_concepts_by_target"
            )
        waiver_targets[waiver_id] = targets
        waiver_missing[waiver_id] = missing_by_target

    seen_target_names: set[str] = set()
    validated_waiver_targets: set[tuple[str, str]] = set()
    for family_index, value in enumerate(_array(families, "imputation/families")):
        family_location = f"imputation/families/{family_index}"
        family = _mapping(value, family_location)
        predictor_columns: set[str] = set()
        for block_index, block_id in enumerate(
            _unique_strings(
                family.get("predictors", []),
                location=f"{family_location}/predictors",
            )
        ):
            if block_id not in predictor_blocks:
                raise SpecResolutionError(
                    f"{family_location}/predictors/{block_index}: dangling "
                    f"predictor-block reference {block_id!r}"
                )
            predictor_columns.update(predictor_blocks[block_id])

        for target_index, target_value in enumerate(
            _array(family.get("targets", []), f"{family_location}/targets")
        ):
            target_location = f"{family_location}/targets/{target_index}"
            target = _mapping(target_value, target_location)
            target_name = target.get("name")
            if not isinstance(target_name, str):
                raise SpecResolutionError(
                    f"{target_location}/name: expected identifier"
                )
            seen_target_names.add(target_name)
            required = _unique_strings(
                target.get("requires_concepts", []),
                location=f"{target_location}/requires_concepts",
            )
            missing_columns: dict[str, list[str]] = {}
            for concept_index, concept_id in enumerate(required):
                if concept_id not in concept_columns:
                    raise SpecResolutionError(
                        f"{target_location}/requires_concepts/{concept_index}: "
                        f"dangling concept reference {concept_id!r}"
                    )
                missing = sorted(concept_columns[concept_id] - predictor_columns)
                if missing:
                    missing_columns[concept_id] = missing

            waiver_id = target.get("waiver")
            if not missing_columns:
                if waiver_id is not None:
                    raise SpecResolutionError(
                        f"{target_location}/waiver: stale waiver {waiver_id!r}; "
                        "all required concepts are covered"
                    )
                continue
            if not isinstance(waiver_id, str) or waiver_id not in waiver_targets:
                raise SpecResolutionError(
                    f"{target_location}: required concepts "
                    f"{sorted(missing_columns)!r} have missing predictor columns "
                    f"{missing_columns!r}; a valid target-listed F-P waiver is required"
                )
            if target_name not in waiver_targets[waiver_id]:
                raise SpecResolutionError(
                    f"{target_location}/waiver: F-P waiver {waiver_id!r} does "
                    f"not list target {target_name!r}"
                )
            recorded = waiver_missing[waiver_id].get(target_name, frozenset())
            expected = frozenset(missing_columns)
            if recorded != expected:
                raise SpecResolutionError(
                    f"{target_location}/waiver: F-P waiver {waiver_id!r} does "
                    "not exactly record missing concepts; "
                    f"expected={sorted(expected)!r}, recorded={sorted(recorded)!r}"
                )
            validated_waiver_targets.add((waiver_id, target_name))

    for waiver_id, targets in waiver_targets.items():
        unknown_targets = sorted(targets - seen_target_names)
        if unknown_targets:
            raise SpecResolutionError(
                f"imputation waiver {waiver_id!r} lists unknown targets "
                f"{unknown_targets!r}"
            )
        unvalidated = sorted(
            target
            for target in targets
            if (waiver_id, target) not in validated_waiver_targets
        )
        if unvalidated:
            raise SpecResolutionError(
                f"imputation waiver {waiver_id!r} is stale or is not attached "
                f"to targets {unvalidated!r}"
            )


def _identifier(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecResolutionError(f"{location}: expected identifier")
    return value


def _validate_source_stage_references(
    graph: Mapping[str, object],
    *,
    source_stage_ids: frozenset[str],
) -> None:
    """Resolve compact producer-resource references into ``sources.stages``.

    Post-clone producer resources deliberately carry only a stage id; the
    source-domain row is inflated by the legacy adapter.  Treating the id as
    an opaque string here would let a bundle validate while naming a stage
    that the adapter cannot reconstruct.
    """

    for node_index, node_value in enumerate(
        _array(graph.get("nodes", []), "imputation/producer_graph/nodes")
    ):
        node_location = f"imputation/producer_graph/nodes/{node_index}"
        node = _mapping(node_value, node_location)
        for resource_index, resource_value in enumerate(
            _array(
                node.get("virtual_resources", []),
                f"{node_location}/virtual_resources",
            )
        ):
            resource_location = f"{node_location}/virtual_resources/{resource_index}"
            resource = _mapping(resource_value, resource_location)
            binding = _mapping(
                resource.get("binding", {}), f"{resource_location}/binding"
            )
            if "source_stage_ref" not in binding:
                continue
            reference = binding["source_stage_ref"]
            if reference is None:
                continue
            reference_location = f"{resource_location}/binding/source_stage_ref"
            reference_row = _mapping(reference, reference_location)
            stage_id = _identifier(
                reference_row.get("stage_id"),
                location=f"{reference_location}/stage_id",
            )
            _require(
                stage_id,
                source_stage_ids,
                namespace="source stage",
                location=f"{reference_location}/stage_id",
            )


def _validate_imputation_structure(
    imputation: Mapping[str, object],
    *,
    families: object,
    family_ids: frozenset[str],
    graph: Mapping[str, object],
    node_ids: frozenset[str],
) -> None:
    """Close the authored family, producer-graph, and chaining semantics.

    JSON Schema establishes the shape of these records.  This pass establishes
    the relationships which the compiler relies on: a single canonical DAG,
    stage-appropriate execution contracts, lossless late-family bindings, and
    derived primary predictor chains.  Minimal country bundles may omit the
    imputation domain entirely, in which case every defaulted collection below
    is empty and resolution remains a no-op.
    """

    graph_location = "imputation/producer_graph"
    scope_coverage = _mapping(
        graph.get("scope_coverage", {}), f"{graph_location}/scope_coverage"
    )
    declared_output_scopes = frozenset(
        _mapping(
            scope_coverage.get("declared", {}),
            f"{graph_location}/scope_coverage/declared",
        )
    )
    stale_graph_fields = sorted(
        field
        for field in (
            "edges",
            "input_inventories",
            "incomparable_node_policy",
            "order",
            "ordering",
            "transfer_groups",
            "waves",
        )
        if field in graph
    )
    if stale_graph_fields:
        raise SpecResolutionError(
            f"{graph_location}: stale compiler-derived fields {stale_graph_fields!r}"
        )
    node_rows = _array(graph.get("nodes", []), f"{graph_location}/nodes")
    nodes_by_id: dict[str, Mapping[str, object]] = {}
    for node_index, value in enumerate(node_rows):
        location = f"{graph_location}/nodes/{node_index}"
        node = _mapping(value, location)
        node_id = _identifier(node.get("id"), location=f"{location}/id")
        name = _identifier(node.get("name"), location=f"{location}/name")
        if node_id != name:
            raise SpecResolutionError(
                f"{location}: node id must equal name; id={node_id!r}, name={name!r}"
            )
        if "write_scopes" in node:
            raise SpecResolutionError(
                f"{location}/write_scopes: stale compiler-derived assertion"
            )
        nodes_by_id[node_id] = node

    external_stage_ids = frozenset(
        _unique_strings(
            graph.get("external_stages", []),
            location=f"{graph_location}/external_stages",
        )
    )
    overlapping_stages = sorted(node_ids & external_stage_ids)
    if overlapping_stages:
        raise SpecResolutionError(
            f"{graph_location}/external_stages: stages also declared as producer "
            f"nodes {overlapping_stages!r}"
        )

    valid_input_stages = node_ids | external_stage_ids
    input_predecessors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for node_index, node in enumerate(node_rows):
        node_map = _mapping(node, f"{graph_location}/nodes/{node_index}")
        node_id = _identifier(
            node_map.get("id"), location=f"{graph_location}/nodes/{node_index}/id"
        )
        for input_index, value in enumerate(
            _array(
                node_map.get("inputs", []),
                f"{graph_location}/nodes/{node_index}/inputs",
            )
        ):
            location = f"{graph_location}/nodes/{node_index}/inputs/{input_index}"
            input_row = _mapping(value, location)
            producing_stage = _identifier(
                input_row.get("producing_stage"),
                location=f"{location}/producing_stage",
            )
            if producing_stage not in valid_input_stages:
                raise SpecResolutionError(
                    f"{location}/producing_stage: dangling stage reference "
                    f"{producing_stage!r}"
                )
            if producing_stage in node_ids:
                input_predecessors[node_id].add(producing_stage)

    predecessors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    successors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    seen_edges: set[tuple[str, str]] = set()
    for edge_index, value in enumerate(
        _array(graph.get("edges", []), f"{graph_location}/edges")
    ):
        location = f"{graph_location}/edges/{edge_index}"
        edge = _array(value, location)
        if len(edge) != 2:
            raise SpecResolutionError(
                f"{location}: expected exactly [producer, consumer]"
            )
        producer = _identifier(edge[0], location=f"{location}/0")
        consumer = _identifier(edge[1], location=f"{location}/1")
        for endpoint_index, endpoint in enumerate((producer, consumer)):
            if endpoint not in node_ids:
                raise SpecResolutionError(
                    f"{location}/{endpoint_index}: dangling producer-node "
                    f"reference {endpoint!r}"
                )
        pair = (producer, consumer)
        if pair in seen_edges:
            raise SpecResolutionError(f"{location}: duplicate edge {pair!r}")
        seen_edges.add(pair)
        predecessors[consumer].add(producer)
        successors[producer].add(consumer)

    # RAW dependencies have one authority: each input's producing_stage.
    # Global edges and equality-only depends_on arrays are compiler outputs.
    predecessors = {
        node_id: set(input_predecessors[node_id]) for node_id in node_ids
    }
    successors = {node_id: set() for node_id in node_ids}
    for consumer, producers in predecessors.items():
        for producer in producers:
            successors[producer].add(consumer)

    for node_index, node in enumerate(node_rows):
        node_map = _mapping(node, f"{graph_location}/nodes/{node_index}")
        node_id = _identifier(
            node_map.get("id"), location=f"{graph_location}/nodes/{node_index}/id"
        )
        dependency_location = f"{graph_location}/nodes/{node_index}/depends_on"
        if "depends_on" in node_map:
            raise SpecResolutionError(
                f"{dependency_location}: stale equality-only dependency assertion; "
                "inputs[].producing_stage is the sole RAW authority"
            )

    indegree = {node_id: len(values) for node_id, values in predecessors.items()}
    remaining = set(node_ids)
    node_rank = {
        _identifier(
            _mapping(node, f"{graph_location}/nodes/{index}").get("id"),
            location=f"{graph_location}/nodes/{index}/id",
        ): index
        for index, node in enumerate(node_rows)
    }
    derived_waves: list[tuple[str, ...]] = []
    while remaining:
        wave = tuple(
            sorted(
                (node_id for node_id in remaining if indegree[node_id] == 0),
                key=node_rank.__getitem__,
            )
        )
        if not wave:
            raise SpecResolutionError(
                f"{graph_location}/nodes: producer graph contains a cycle among "
                f"{sorted(remaining)!r}"
            )
        derived_waves.append(wave)
        remaining.difference_update(wave)
        for producer in wave:
            for consumer in sorted(successors[producer]):
                indegree[consumer] -= 1

    order_location = f"{graph_location}/order"
    derived_order = tuple(node_id for wave in derived_waves for node_id in wave)
    order = _unique_strings(
        graph.get("order", list(derived_order)), location=order_location
    )
    missing_order_nodes = sorted(node_ids - set(order))
    extra_order_nodes = sorted(set(order) - node_ids)
    if missing_order_nodes or extra_order_nodes:
        raise SpecResolutionError(
            f"{order_location}: must be an exact producer-node permutation; "
            f"missing={missing_order_nodes!r}, extra={extra_order_nodes!r}"
        )
    if order != derived_order:
        raise SpecResolutionError(
            f"{order_location}: must equal the deterministic topological order; "
            f"expected={list(derived_order)!r}, actual={list(order)!r}"
        )

    waves_location = f"{graph_location}/waves"
    declared_waves: list[tuple[str, ...]] = []
    seen_wave_nodes: set[str] = set()
    for wave_index, value in enumerate(
        _array(
            graph.get("waves", [list(wave) for wave in derived_waves]),
            waves_location,
        )
    ):
        location = f"{waves_location}/{wave_index}"
        wave = _unique_strings(value, location=location)
        if not wave:
            raise SpecResolutionError(f"{location}: producer wave must not be empty")
        for node_index, node_id in enumerate(wave):
            if node_id not in node_ids:
                raise SpecResolutionError(
                    f"{location}/{node_index}: dangling producer-node reference "
                    f"{node_id!r}"
                )
            if node_id in seen_wave_nodes:
                raise SpecResolutionError(
                    f"{location}/{node_index}: producer node {node_id!r} occurs "
                    "in more than one wave"
                )
            seen_wave_nodes.add(node_id)
        declared_waves.append(wave)
    missing_wave_nodes = sorted(node_ids - seen_wave_nodes)
    if missing_wave_nodes:
        raise SpecResolutionError(
            f"{waves_location}: waves do not cover producer nodes "
            f"{missing_wave_nodes!r}"
        )
    flattened_waves = tuple(node_id for wave in declared_waves for node_id in wave)
    if flattened_waves != order:
        raise SpecResolutionError(
            f"{waves_location}: flattened waves must exactly equal graph order; "
            f"expected={list(order)!r}, actual={list(flattened_waves)!r}"
        )
    if tuple(declared_waves) != tuple(derived_waves):
        raise SpecResolutionError(
            f"{waves_location}: must equal deterministic topological waves; "
            f"expected={[list(wave) for wave in derived_waves]!r}, "
            f"actual={[list(wave) for wave in declared_waves]!r}"
        )

    resource_semantics = _mapping(
        graph.get("resource_semantics", {}),
        f"{graph_location}/resource_semantics",
    )
    if "producer_order" in resource_semantics:
        raise SpecResolutionError(
            f"{graph_location}/resource_semantics/producer_order: stale derived "
            "assertion; graph order is the sole producer order"
        )

    transfer_execution = _mapping(
        imputation.get("transfer_execution", {}), "imputation/transfer_execution"
    )
    profiles = _mapping(
        transfer_execution.get("profiles", {}),
        "imputation/transfer_execution/profiles",
    )
    profile_ids = frozenset(
        _identifier(profile_id, location="imputation/transfer_execution/profiles")
        for profile_id in profiles
    )
    schedule = _mapping(
        imputation.get("gap_fill_schedule", {}), "imputation/gap_fill_schedule"
    )
    direction_ids = _declare_array_ids(
        schedule.get("directions", []),
        location="imputation/gap_fill_schedule/directions",
        field="name",
    )

    family_rows = _array(families, "imputation/families")
    families_by_id: dict[str, Mapping[str, object]] = {}
    family_targets: dict[str, tuple[tuple[str, str], ...]] = {}
    late_families: list[tuple[int, Mapping[str, object]]] = []
    primary_families: list[tuple[int, Mapping[str, object]]] = []
    used_direction_ids: set[str] = set()
    gap_target_keys: set[tuple[str, str, str, str]] = set()
    for family_index, value in enumerate(family_rows):
        location = f"imputation/families/{family_index}"
        family = _mapping(value, location)
        family_id = _identifier(family.get("id"), location=f"{location}/id")
        families_by_id[family_id] = family
        entities = _unique_strings(
            family.get("entities", []), location=f"{location}/entities"
        )
        targets: list[tuple[str, str]] = []
        seen_target_names: set[str] = set()
        for target_index, target_value in enumerate(
            _array(family.get("targets", []), f"{location}/targets")
        ):
            target_location = f"{location}/targets/{target_index}"
            target = _mapping(target_value, target_location)
            target_name = _identifier(
                target.get("name"), location=f"{target_location}/name"
            )
            if target_name in seen_target_names:
                raise SpecResolutionError(
                    f"{target_location}/name: duplicate family target {target_name!r}"
                )
            seen_target_names.add(target_name)
            target_entity = _identifier(
                target.get("entity"), location=f"{target_location}/entity"
            )
            targets.append((target_name, target_entity))
        target_entities = frozenset(entity for _, entity in targets)
        if frozenset(entities) != target_entities:
            raise SpecResolutionError(
                f"{location}/entities: must exactly cover target entities; "
                f"expected={sorted(target_entities)!r}, "
                f"actual={sorted(entities)!r}"
            )
        family_targets[family_id] = tuple(targets)

        stage = _identifier(family.get("stage"), location=f"{location}/stage")
        contract = _identifier(
            family.get("execution_contract"),
            location=f"{location}/execution_contract",
        )
        if stage == "primary_puf_qrf":
            primary_families.append((family_index, family))
            producer = nodes_by_id.get(contract)
            if producer is None:
                raise SpecResolutionError(
                    f"{location}/execution_contract: dangling primary producer-node "
                    f"reference {contract!r}"
                )
            if producer.get("kind") != "primary_puf":
                raise SpecResolutionError(
                    f"{location}/execution_contract: primary contract {contract!r} "
                    "must resolve to a primary_puf producer node"
                )
        elif stage in {"gap_fill_stacked_spine", "late_producer_dag"}:
            if contract not in profile_ids:
                raise SpecResolutionError(
                    f"{location}/execution_contract: dangling transfer-execution "
                    f"profile reference {contract!r}"
                )
        elif contract not in profile_ids | node_ids:
            raise SpecResolutionError(
                f"{location}/execution_contract: dangling execution-contract "
                f"reference {contract!r}"
            )

        direction = family.get("direction")
        if stage == "gap_fill_stacked_spine" and direction is None:
            raise SpecResolutionError(
                f"{location}/direction: gap-fill family requires a direction"
            )
        if direction is not None:
            direction_id = _identifier(direction, location=f"{location}/direction")
            if direction_id not in direction_ids:
                raise SpecResolutionError(
                    f"{location}/direction: dangling gap-fill direction reference "
                    f"{direction_id!r}"
                )
            used_direction_ids.add(direction_id)
        for target_index, target_value in enumerate(
            _array(family.get("targets", []), f"{location}/targets")
        ):
            target_location = f"{location}/targets/{target_index}"
            target = _mapping(target_value, target_location)
            output_scope = target.get("output_coverage_scope")
            if stage in {"primary_puf_qrf", "late_producer_dag"}:
                output_scope_id = _identifier(
                    output_scope,
                    location=f"{target_location}/output_coverage_scope",
                )
                if output_scope_id not in declared_output_scopes:
                    raise SpecResolutionError(
                        f"{target_location}/output_coverage_scope: dangling graph "
                        f"scope reference {output_scope_id!r}"
                    )
            elif "output_coverage_scope" in target:
                raise SpecResolutionError(
                    f"{target_location}/output_coverage_scope: only primary and "
                    "late family targets own producer outputs"
                )
            producer_binding = target.get("producer_binding")
            if stage != "gap_fill_stacked_spine":
                if producer_binding is not None:
                    raise SpecResolutionError(
                        f"{target_location}/producer_binding: only gap-fill "
                        "targets may own producer schedule bindings"
                    )
                continue
            producer = _mapping(
                producer_binding if producer_binding is not None else {},
                f"{target_location}/producer_binding",
            )
            _identifier(
                producer.get("operator"),
                location=f"{target_location}/producer_binding/operator",
            )
            _identifier(
                producer.get("execution_scope"),
                location=f"{target_location}/producer_binding/execution_scope",
            )
            _identifier(
                producer.get("stage"),
                location=f"{target_location}/producer_binding/stage",
            )
            order_index = producer.get("order_index")
            if (
                isinstance(order_index, bool)
                or not isinstance(order_index, int)
                or order_index < 0
            ):
                raise SpecResolutionError(
                    f"{target_location}/producer_binding/order_index: expected "
                    "a nonnegative integer"
                )
            target_key = (
                str(direction),
                str(target.get("entity")),
                str(family.get("name")),
                str(target.get("name")),
            )
            if target_key in gap_target_keys:
                raise SpecResolutionError(
                    f"{target_location}: duplicate gap-fill scheduled target "
                    f"{target_key!r}"
                )
            gap_target_keys.add(target_key)
        if stage == "late_producer_dag":
            late_families.append((family_index, family))

    unused_direction_ids = sorted(direction_ids - used_direction_ids)
    if unused_direction_ids:
        raise SpecResolutionError(
            "imputation/gap_fill_schedule/directions: directions without "
            f"gap-fill families {unused_direction_ids!r}"
        )

    group_by_name: dict[str, tuple[int, Mapping[str, object]]] = {}
    for group_index, value in enumerate(
        _array(graph.get("transfer_groups", []), f"{graph_location}/transfer_groups")
    ):
        location = f"{graph_location}/transfer_groups/{group_index}"
        group = _mapping(value, location)
        name = _identifier(group.get("name"), location=f"{location}/name")
        if name in group_by_name:
            raise SpecResolutionError(
                f"{location}/name: duplicate transfer-group name {name!r}"
            )
        group_by_name[name] = (group_index, group)

    late_runtime_names: set[str] = set()
    for family_index, family in late_families:
        location = f"imputation/families/{family_index}"
        family_id = _identifier(family.get("id"), location=f"{location}/id")
        runtime_name = _identifier(
            family.get("runtime_name"), location=f"{location}/runtime_name"
        )
        if runtime_name in late_runtime_names:
            raise SpecResolutionError(
                f"{location}/runtime_name: duplicate late-family runtime name "
                f"{runtime_name!r}"
            )
        late_runtime_names.add(runtime_name)
        node = nodes_by_id.get(runtime_name)
        if node is None:
            raise SpecResolutionError(
                f"{location}/runtime_name: dangling late producer-node reference "
                f"{runtime_name!r}"
            )
        if node.get("kind") != "late_transfer":
            raise SpecResolutionError(
                f"{location}/runtime_name: node {runtime_name!r} is not a "
                "late_transfer producer"
            )
        group_match = group_by_name.get(runtime_name)
        if group_match is None:
            entities = _unique_strings(
                family.get("entities", []), location=f"{location}/entities"
            )
            if len(entities) != 1:
                raise SpecResolutionError(
                    f"{location}/entities: late family must have one entity"
                )
            runtime_prefix = f"transfer:{entities[0]}/"
            if not runtime_name.startswith(runtime_prefix):
                raise SpecResolutionError(
                    f"{location}/runtime_name: must begin {runtime_prefix!r}"
                )
            expected_family_id = (
                f"late/{entities[0]}/{runtime_name.removeprefix(runtime_prefix)}"
            )
            if family_id != expected_family_id:
                raise SpecResolutionError(
                    f"{location}/id: expected {expected_family_id!r}"
                )
            continue
        group_index, group = group_match
        group_location = f"{graph_location}/transfer_groups/{group_index}"
        group_entity = _identifier(
            group.get("entity"), location=f"{group_location}/entity"
        )
        group_family = _identifier(
            group.get("family"), location=f"{group_location}/family"
        )
        expected_family_id = f"late/{group_entity}/{group_family}"
        if family_id != expected_family_id:
            raise SpecResolutionError(
                f"{group_location}: transfer group resolves family "
                f"{expected_family_id!r}, not {family_id!r}"
            )
        entities = _unique_strings(
            family.get("entities", []), location=f"{location}/entities"
        )
        if entities != (group_entity,):
            raise SpecResolutionError(
                f"{group_location}/entity: must equal the late family's sole "
                f"entity; expected={list(entities)!r}, actual={group_entity!r}"
            )
        group_targets = _unique_strings(
            group.get("targets", []), location=f"{group_location}/targets"
        )
        expected_targets = tuple(name for name, _ in family_targets[family_id])
        if group_targets != expected_targets:
            raise SpecResolutionError(
                f"{group_location}/targets: must exactly equal late family target "
                f"order; expected={list(expected_targets)!r}, "
                f"actual={list(group_targets)!r}"
            )
    extra_groups = sorted(set(group_by_name) - late_runtime_names)
    if extra_groups:
        raise SpecResolutionError(
            f"{graph_location}/transfer_groups: groups without late families "
            f"{extra_groups!r}"
        )
    orphan_late_nodes = sorted(
        node_id
        for node_id, node in nodes_by_id.items()
        if node.get("kind") == "late_transfer" and node_id not in late_runtime_names
    )
    if orphan_late_nodes:
        raise SpecResolutionError(
            f"{graph_location}/nodes: late-transfer nodes without families "
            f"{orphan_late_nodes!r}"
        )

    producer_family_links: dict[str, tuple[int, Mapping[str, object]]] = {}
    for family_index, family in [*primary_families, *late_families]:
        family_location = f"imputation/families/{family_index}"
        stage = _identifier(
            family.get("stage"), location=f"{family_location}/stage"
        )
        producer_field = (
            "execution_contract" if stage == "primary_puf_qrf" else "runtime_name"
        )
        producer_id = _identifier(
            family.get(producer_field),
            location=f"{family_location}/{producer_field}",
        )
        previous = producer_family_links.get(producer_id)
        if previous is not None:
            previous_index, _ = previous
            raise SpecResolutionError(
                f"{family_location}/{producer_field}: producer node {producer_id!r} "
                "is linked from more than one family; "
                f"first=imputation/families/{previous_index}"
            )
        producer_family_links[producer_id] = (family_index, family)

    orphan_primary_nodes = sorted(
        node_id
        for node_id, node in nodes_by_id.items()
        if node.get("kind") == "primary_puf"
        and node_id not in producer_family_links
    )
    if orphan_primary_nodes:
        raise SpecResolutionError(
            f"{graph_location}/nodes: primary-PUF nodes without families "
            f"{orphan_primary_nodes!r}"
        )

    # Modeled outputs are authored once on family targets.  Graph nodes retain
    # structural/virtual outputs only; compilation joins the two sets and sorts
    # them by (entity, column) to recreate the constants-era node contracts.
    for node_index, node_value in enumerate(node_rows):
        node_location = f"{graph_location}/nodes/{node_index}"
        node = _mapping(node_value, node_location)
        node_id = _identifier(node.get("id"), location=f"{node_location}/id")
        authored_outputs: dict[tuple[str, str], str] = {}
        for output_index, output_value in enumerate(
            _array(node.get("outputs", []), f"{node_location}/outputs")
        ):
            output_location = f"{node_location}/outputs/{output_index}"
            output = _mapping(output_value, output_location)
            entity = _identifier(
                output.get("entity"), location=f"{output_location}/entity"
            )
            column = _identifier(
                output.get("column"), location=f"{output_location}/column"
            )
            coverage_scope = _identifier(
                output.get("coverage_scope"),
                location=f"{output_location}/coverage_scope",
            )
            if coverage_scope not in declared_output_scopes:
                raise SpecResolutionError(
                    f"{output_location}/coverage_scope: dangling graph scope "
                    f"reference {coverage_scope!r}"
                )
            key = (entity, column)
            if key in authored_outputs:
                relation = (
                    "duplicate"
                    if authored_outputs[key] == coverage_scope
                    else "conflicting"
                )
                raise SpecResolutionError(
                    f"{output_location}: {relation} authored producer output "
                    f"{key!r}"
                )
            authored_outputs[key] = coverage_scope

        link = producer_family_links.get(node_id)
        if link is None:
            continue
        family_index, family = link
        family_location = f"imputation/families/{family_index}"
        expanded_outputs: dict[tuple[str, str], str] = {}
        for target_index, target_value in enumerate(
            _array(family.get("targets", []), f"{family_location}/targets")
        ):
            target_location = f"{family_location}/targets/{target_index}"
            target = _mapping(target_value, target_location)
            key = (
                _identifier(
                    target.get("entity"), location=f"{target_location}/entity"
                ),
                _identifier(
                    target.get("name"), location=f"{target_location}/name"
                ),
            )
            coverage_scope = _identifier(
                target.get("output_coverage_scope"),
                location=f"{target_location}/output_coverage_scope",
            )
            if key in expanded_outputs:
                relation = (
                    "duplicate"
                    if expanded_outputs[key] == coverage_scope
                    else "conflicting"
                )
                raise SpecResolutionError(
                    f"{target_location}: {relation} expanded producer output "
                    f"{key!r}"
                )
            expanded_outputs[key] = coverage_scope
            if key in authored_outputs:
                relation = (
                    "duplicates"
                    if authored_outputs[key] == coverage_scope
                    else "conflicts with"
                )
                raise SpecResolutionError(
                    f"{node_location}/outputs: authored output {key!r} {relation} "
                    f"family-owned output at {target_location}"
                )
        # Materialize the deterministic compile order here so resolution fails
        # before an adapter can observe an ambiguous union.
        compiled_output_keys = tuple(
            sorted({**authored_outputs, **expanded_outputs})
        )
        if len(compiled_output_keys) != len(authored_outputs) + len(expanded_outputs):
            raise SpecResolutionError(
                f"{node_location}/outputs: duplicate expanded producer output"
            )

    chaining = _mapping(imputation.get("chaining", {}), "imputation/chaining")
    if "primary_effective_predictor_tuples" in chaining:
        raise SpecResolutionError(
            "imputation/chaining/primary_effective_predictor_tuples: stale "
            "derived assertion; primary family order and predictor block are "
            "the sole authority"
        )
    seen_splits: set[tuple[str, str]] = set()
    for split_index, value in enumerate(
        _array(chaining.get("split_after", []), "imputation/chaining/split_after")
    ):
        location = f"imputation/chaining/split_after/{split_index}"
        split = _mapping(value, location)
        family_id = _identifier(split.get("family"), location=f"{location}/family")
        if family_id not in family_ids:
            raise SpecResolutionError(
                f"{location}/family: dangling family reference {family_id!r}"
            )
        after_target = _identifier(
            split.get("after_target"), location=f"{location}/after_target"
        )
        pair = (family_id, after_target)
        if pair in seen_splits:
            raise SpecResolutionError(f"{location}: duplicate split boundary {pair!r}")
        seen_splits.add(pair)
        targets = tuple(name for name, _ in family_targets[family_id])
        if after_target not in targets:
            raise SpecResolutionError(
                f"{location}/after_target: target {after_target!r} is not in "
                f"family {family_id!r}"
            )
        target_index = targets.index(after_target)
        family = families_by_id[family_id]
        max_targets = family.get("max_targets_per_fit")
        at_family_end = target_index == len(targets) - 1
        at_fit_boundary = (
            isinstance(max_targets, int)
            and not isinstance(max_targets, bool)
            and max_targets > 0
            and (target_index + 1) % max_targets == 0
        )
        if not (at_family_end or at_fit_boundary):
            raise SpecResolutionError(
                f"{location}/after_target: {after_target!r} is not a declared "
                "family-end or max_targets_per_fit boundary"
            )

    if len(primary_families) > 1:
        raise SpecResolutionError(
            "imputation/families: primary_puf_qrf stage must have exactly one family"
        )
    if not primary_families:
        return

    primary_index, primary = primary_families[0]
    primary_location = f"imputation/families/{primary_index}"
    if primary.get("chaining") != "base_plus_preceding_declared_targets":
        raise SpecResolutionError(
            f"{primary_location}/chaining: primary family must declare "
            "base_plus_preceding_declared_targets"
        )
    predictor_refs = _unique_strings(
        primary.get("predictors", []),
        location=f"{primary_location}/predictors",
    )
    if len(predictor_refs) != 1:
        raise SpecResolutionError(
            f"{primary_location}/predictors: primary family must reference "
            "exactly one base predictor block"
        )
    predictor_blocks = _mapping(
        imputation.get("predictor_blocks", {}), "imputation/predictor_blocks"
    )
    block_id = predictor_refs[0]
    if block_id not in predictor_blocks:
        raise SpecResolutionError(
            f"{primary_location}/predictors/0: dangling predictor-block "
            f"reference {block_id!r}"
        )
    block = _mapping(
        predictor_blocks[block_id], f"imputation/predictor_blocks/{block_id}"
    )
    base_predictors = _unique_strings(
        block.get("columns", []),
        location=f"imputation/predictor_blocks/{block_id}/columns",
    )
    if not base_predictors:
        raise SpecResolutionError(
            f"imputation/predictor_blocks/{block_id}/columns: primary base "
            "predictor block must not be empty"
        )


def _walk_strings(value: object, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_strings(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{path}/{index}")
    elif isinstance(value, str):
        yield path or "/", value


def _require(
    target: str,
    declared: frozenset[str],
    *,
    namespace: str,
    location: str,
) -> None:
    if target not in declared:
        raise SpecResolutionError(
            f"{location}: dangling {namespace} reference {target!r}"
        )


_SOURCE_REF_FIELDS = frozenset(
    {
        "fact_ref",
        "identified_county_source",
        "ladder_source",
        "source_ref",
        "targets",
        "training",
    }
)


def _is_declared_source_ref_location(kind: str, pointer: str) -> bool:
    """Distinguish typed source refs from producer ids such as ``source:x``.

    The producer graph intentionally uses the ``source:`` spelling for stage
    names inherited from the constants-era DAG.  Prefix scanning alone would
    misclassify those ids as external inputs.  Source references are therefore
    resolved only at schema-declared reference fields (plus a vintage record's
    optional ``source`` field).
    """

    segments = tuple(segment for segment in pointer.split("/") if segment)
    named_segments = tuple(segment for segment in segments if not segment.isdigit())
    if not named_segments:
        return False
    field = named_segments[-1]
    return field in _SOURCE_REF_FIELDS or (kind == "vintages" and field == "source")


_ALLOWED_VINTAGE_COMPATIBILITY_KIND_PAIRS = frozenset(
    {
        frozenset({"survey_period_ref"}),
        frozenset({"geography_vintage_ref"}),
        frozenset({"survey_period_ref", "target_period_ref"}),
        frozenset({"tax_period_ref", "target_period_ref"}),
        frozenset({"policy_engine_surface_ref", "target_period_ref"}),
        frozenset({"release_series_ref", "target_period_ref"}),
    }
)
_SOURCE_VINTAGE_KINDS = frozenset(
    {
        "tax_period_ref",
        "survey_period_ref",
        "target_period_ref",
        "geography_vintage_ref",
    }
)


def _validate_vintage_compatibility(
    resources: Mapping[str, object],
    *,
    vintages: Mapping[str, object],
) -> None:
    """Validate reviewed, reciprocal vintage relations and their uses.

    Equal literal values are deliberately irrelevant: compatibility is an
    authored review result.  Multi-vintage source pins additionally require a
    complete pairwise relation, so adding an individually valid reference
    cannot silently create an unreviewed composite input.
    """

    record_by_id: dict[str, Mapping[str, object]] = {}
    compatible_by_id: dict[str, frozenset[str]] = {}
    for index, value in enumerate(
        _array(vintages.get("records", []), "vintages/records")
    ):
        location = f"vintages/records/{index}"
        record = _mapping(value, location)
        record_id = _identifier(record.get("id"), location=f"{location}/id")
        if record_id in record_by_id:
            raise SpecResolutionError(f"{location}/id: duplicate id {record_id!r}")
        record_by_id[record_id] = record
        references = _unique_strings(
            record.get("compatible_with", []),
            location=f"{location}/compatible_with",
        )
        if not references:
            raise SpecResolutionError(
                f"{location}/compatible_with: at least one reviewed relation required"
            )
        compatible_by_id[record_id] = frozenset(
            _strip_prefix(reference, "vintage") for reference in references
        )

    for record_id, targets in compatible_by_id.items():
        source = record_by_id[record_id]
        source_kind = _identifier(
            source.get("kind"), location=f"vintages/{record_id}/kind"
        )
        for target_id in sorted(targets):
            if target_id == record_id:
                raise SpecResolutionError(
                    f"vintages/{record_id}/compatible_with: self relation is forbidden"
                )
            target = record_by_id.get(target_id)
            if target is None:
                raise SpecResolutionError(
                    f"vintages/{record_id}/compatible_with: dangling vintage "
                    f"reference {target_id!r}"
                )
            target_kind = _identifier(
                target.get("kind"), location=f"vintages/{target_id}/kind"
            )
            kind_pair = frozenset({source_kind, target_kind})
            if kind_pair not in _ALLOWED_VINTAGE_COMPATIBILITY_KIND_PAIRS:
                raise SpecResolutionError(
                    f"vintages/{record_id}/compatible_with: incompatible kind "
                    f"pair {source_kind!r} -> {target_kind!r}"
                )
            if record_id not in compatible_by_id.get(target_id, frozenset()):
                raise SpecResolutionError(
                    f"vintages/{record_id}/compatible_with: relation to "
                    f"{target_id!r} is not reciprocal"
                )

    sources = _mapping(resources.get("sources", {}), "sources")
    for source_index, value in enumerate(
        _array(sources.get("sources", []), "sources/sources")
    ):
        location = f"sources/sources/{source_index}/vintages"
        source = _mapping(value, f"sources/sources/{source_index}")
        source_vintages = tuple(
            _strip_prefix(reference, "vintage")
            for reference in _unique_strings(
                source.get("vintages", []), location=location
            )
        )
        for vintage_id in source_vintages:
            record = record_by_id.get(vintage_id)
            if record is None:
                # The generic reference scan reports the same dangling id with
                # its exact JSON pointer after this semantic pass.
                continue
            kind = str(record.get("kind"))
            if kind not in _SOURCE_VINTAGE_KINDS:
                raise SpecResolutionError(
                    f"{location}: {vintage_id!r} has non-source vintage kind {kind!r}"
                )
        for left_index, left in enumerate(source_vintages):
            for right in source_vintages[left_index + 1 :]:
                if right not in compatible_by_id.get(left, frozenset()):
                    raise SpecResolutionError(
                        f"{location}: composite source vintages {left!r} and "
                        f"{right!r} lack a reviewed compatibility relation"
                    )


def _require_exact_mapping(
    value: object,
    expected: Mapping[str, object],
    *,
    location: str,
) -> Mapping[str, object]:
    actual = _mapping(value, location)
    if dict(actual) != dict(expected):
        raise SpecResolutionError(
            f"{location}: expected reviewed typed reference {dict(expected)!r}, "
            f"got {dict(actual)!r}"
        )
    return actual


def _validate_single_authority_references(
    resources: Mapping[str, object],
    *,
    bundle: Mapping[str, object],
    spine: Mapping[str, object],
    imputation: Mapping[str, object],
) -> None:
    """Close cross-domain refs whose targets replace legacy duplicate payloads."""

    publication_value = resources.get("publication")
    if publication_value is not None:
        publication = _mapping(publication_value, "publication")
        release = _mapping(publication.get("release", {}), "publication/release")
        rows = _array(
            release.get("rung_fractions", []),
            "publication/release/rung_fractions",
        )
        expected_tokens = ("f001", "f004", "f010", "f025", "f100")
        actual_tokens: list[str] = []
        for index, value in enumerate(rows):
            location = f"publication/release/rung_fractions/{index}"
            row = _mapping(value, location)
            token = _identifier(row.get("token"), location=f"{location}/token")
            actual_tokens.append(token)
            try:
                percent = int(token[1:])
            except (ValueError, IndexError) as error:
                raise SpecResolutionError(
                    f"{location}/token: invalid rung token {token!r}"
                ) from error
            if row.get("fraction") != percent / 100:
                raise SpecResolutionError(
                    f"{location}/fraction: inconsistent with token {token!r}"
                )
            if row.get("percent_basis_points") != percent * 100:
                raise SpecResolutionError(
                    f"{location}/percent_basis_points: inconsistent with "
                    f"token {token!r}"
                )
        if tuple(actual_tokens) != expected_tokens:
            raise SpecResolutionError(
                "publication/release/rung_fractions: expected ordered tokens "
                f"{expected_tokens!r}, got {tuple(actual_tokens)!r}"
            )

        # Publication is part of the shared country-spec core and can stand
        # alone in a minimal bundle.  Close this cross-domain reference only
        # when a country actually declares spine sampling.
        if "sampling" in spine:
            sampling = _mapping(spine["sampling"], "spine/sampling")
            fraction = _mapping(
                sampling.get("fraction", {}),
                "spine/sampling/fraction",
            )
            _require_exact_mapping(
                fraction.get("rungs_ref"),
                {"domain": "publication", "pointer": "/release/rung_fractions"},
                location="spine/sampling/fraction/rungs_ref",
            )

    graph = _mapping(imputation.get("producer_graph", {}), "imputation/producer_graph")
    nodes = [
        _mapping(value, f"imputation/producer_graph/nodes/{index}")
        for index, value in enumerate(
            _array(graph.get("nodes", []), "imputation/producer_graph/nodes")
        )
    ]
    if not nodes:
        return

    metadata = _mapping(
        graph.get("resource_semantics", {}),
        "imputation/producer_graph/resource_semantics",
    )
    if not ({"resolution", "source_execution_defaults"} & metadata.keys()):
        return

    support_roles = [
        _mapping(value, f"spine/support_roles/{index}")
        for index, value in enumerate(
            _array(spine.get("support_roles", []), "spine/support_roles")
        )
    ]
    puf_roles = [row for row in support_roles if row.get("id") == "puf_tax_detail"]
    if len(puf_roles) != 1:
        raise SpecResolutionError(
            "spine/support_roles: puf_tax_detail reference must resolve once"
        )
    puf_role = puf_roles[0]
    _mapping(
        puf_role.get("attachment"), "spine/support_roles/puf_tax_detail/attachment"
    )
    tail_support = _mapping(
        puf_role.get("tail_support"),
        "spine/support_roles/puf_tax_detail/tail_support",
    )
    _mapping(
        tail_support.get("legacy_contract"),
        "spine/support_roles/puf_tax_detail/tail_support/legacy_contract",
    )

    pipeline = _mapping(spine.get("pipeline_contract"), "spine/pipeline_contract")
    _unique_strings(
        pipeline.get("post_clone_source_operator_order"),
        location="spine/pipeline_contract/post_clone_source_operator_order",
    )
    resolution = _mapping(
        metadata.get("resolution"),
        "imputation/producer_graph/resource_semantics/resolution",
    )
    _require_exact_mapping(
        resolution.get("clone_attachment_ref"),
        {
            "domain": "spine",
            "support_role": "puf_tax_detail",
            "pointer": "/attachment",
        },
        location=(
            "imputation/producer_graph/resource_semantics/resolution/"
            "clone_attachment_ref"
        ),
    )
    _require_exact_mapping(
        resolution.get("build_model_seed_ref"),
        {
            "domain": "seed_protocol",
            "value_source": "run_request.build_model_seed",
        },
        location=(
            "imputation/producer_graph/resource_semantics/resolution/"
            "build_model_seed_ref"
        ),
    )
    source_defaults = _mapping(
        metadata.get("source_execution_defaults"),
        "imputation/producer_graph/resource_semantics/source_execution_defaults",
    )
    _require_exact_mapping(
        source_defaults.get("operator_registry_ref"),
        {
            "domain": "spine",
            "pointer": "/pipeline_contract/post_clone_source_operator_order",
        },
        location=(
            "imputation/producer_graph/resource_semantics/"
            "source_execution_defaults/operator_registry_ref"
        ),
    )
    _require_exact_mapping(
        source_defaults.get("time_period_ref"),
        {"domain": "bundle", "pointer": "/dataset_run/target_period"},
        location=(
            "imputation/producer_graph/resource_semantics/"
            "source_execution_defaults/time_period_ref"
        ),
    )
    dataset_run = _mapping(bundle.get("dataset_run"), "bundle/dataset_run")
    if "target_period" not in dataset_run:
        raise SpecResolutionError(
            "bundle/dataset_run/target_period: referenced value is absent"
        )

    primary_nodes = [row for row in nodes if row.get("name") == "primary_puf_qrf"]
    if len(primary_nodes) != 1:
        raise SpecResolutionError(
            "imputation/producer_graph/nodes: primary_puf_qrf must resolve once"
        )
    primary_resources = [
        _mapping(value, "primary_puf_qrf virtual resource")
        for value in _array(
            primary_nodes[0].get("virtual_resources", []),
            "primary_puf_qrf/virtual_resources",
        )
        if _mapping(value, "primary_puf_qrf virtual resource").get("id")
        == "tax_unit.@primary_puf_execution_config"
    ]
    if len(primary_resources) != 1:
        raise SpecResolutionError(
            "primary_puf_qrf/virtual_resources: execution config must resolve once"
        )
    primary_binding = _mapping(
        primary_resources[0].get("binding"), "primary_puf_qrf execution binding"
    )
    execution_tail = _mapping(
        primary_binding.get("capital_gains_tail"),
        "primary_puf_qrf execution binding/capital_gains_tail",
    )
    _require_exact_mapping(
        execution_tail.get("support_contract_ref"),
        {
            "domain": "spine",
            "support_role": "puf_tax_detail",
            "pointer": "/tail_support/legacy_contract",
        },
        location="primary_puf_qrf capital_gains_tail/support_contract_ref",
    )

    calibration_value = resources.get("calibration")
    if calibration_value is None:
        return
    calibration = _mapping(calibration_value, "calibration")
    calibration_tails = _mapping(
        calibration.get("tail_contracts"), "calibration/tail_contracts"
    )
    calibration_puf = _mapping(
        calibration_tails.get("puf_capital_gains_tail"),
        "calibration/tail_contracts/puf_capital_gains_tail",
    )
    _require_exact_mapping(
        calibration_puf.get("support_contract_ref"),
        {
            "domain": "spine",
            "support_role": "puf_tax_detail",
            "pointer": "/tail_support/legacy_contract",
        },
        location="calibration/tail_contracts/puf_capital_gains_tail/support_contract_ref",
    )
    _require_exact_mapping(
        calibration_puf.get("execution_binding_ref"),
        {
            "domain": "imputation",
            "producer": "primary_puf_qrf",
            "resource": "tax_unit.@primary_puf_execution_config",
            "pointer": "/binding/capital_gains_tail",
        },
        location=(
            "calibration/tail_contracts/puf_capital_gains_tail/execution_binding_ref"
        ),
    )


def resolve_cross_references(
    resources: Mapping[str, object],
    *,
    kernel_registry: KernelRegistry,
    generated_authorities: Mapping[str, object] | None = None,
) -> ResolutionResult:
    """Resolve every typed namespace used by the selected country bundle."""

    sources = resources.get("sources", {})
    vintages = resources.get("vintages", {})
    spine = resources.get("spine", {})
    imputation = resources.get("imputation", {})
    bundle = _mapping(resources.get("bundle", {}), "bundle")

    source_ids = _declare_array_ids(
        _mapping(sources, "sources").get("sources", []),
        location="sources/sources",
    )
    source_stage_ids = _declare_array_ids(
        _mapping(sources, "sources").get("stages", []),
        location="sources/stages",
        field="stage",
    )
    take_up = _mapping(resources.get("take_up", {}), "take_up")
    if "take_up" in resources:
        validate_take_up_semantics(
            take_up,
            sources_document=_mapping(sources, "sources"),
        )
    vintages_map = _mapping(vintages, "vintages")
    vintage_ids = _declare_array_ids(
        vintages_map.get("records", []),
        location="vintages/records",
    )
    _validate_vintage_compatibility(resources, vintages=vintages_map)
    try:
        vintage_authorities = resolve_vintage_authorities(
            resources,
            generated_authorities=generated_authorities,
        )
    except VintageAuthorityError as error:
        raise SpecResolutionError(str(error)) from error
    spine_map = _mapping(spine, "spine")
    channel_ids = _declare_array_ids(
        spine_map.get("channels", []), location="spine/channels"
    )
    support_ids = _declare_array_ids(
        spine_map.get("support_roles", []), location="spine/support_roles"
    )
    imputation_map = _mapping(imputation, "imputation")
    model_ids = _declare_mapping_ids(
        imputation_map.get("models", {}), location="imputation/models"
    )
    family_rows = imputation_map.get("families", [])
    family_ids = _declare_array_ids(family_rows, location="imputation/families")
    _validate_imputation_concept_coverage(
        imputation_map,
        families=family_rows,
    )
    graph = _mapping(imputation_map.get("producer_graph", {}), "producer_graph")
    node_ids = _declare_array_ids(
        graph.get("nodes", []), location="imputation/producer_graph/nodes"
    )
    _validate_source_stage_references(
        graph,
        source_stage_ids=source_stage_ids,
    )
    _validate_imputation_structure(
        imputation_map,
        families=family_rows,
        family_ids=family_ids,
        graph=graph,
        node_ids=node_ids,
    )
    _validate_single_authority_references(
        resources,
        bundle=bundle,
        spine=spine_map,
        imputation=imputation_map,
    )

    protocol_id = bundle.get("seed_protocol")
    if protocol_id != LEGACY_V1_PROTOCOL.id:
        # derived-v2 is valid grammar but intentionally has no F0 execution
        # protocol.  It can compile only after its immutable registry lands.
        raise SpecResolutionError(
            f"bundle/seed_protocol: unsupported F0 protocol {protocol_id!r}"
        )

    try:
        seed_site_bindings = resolve_seed_site_bindings(
            spine_map,
            protocol=LEGACY_V1_PROTOCOL,
            source_stage_ids=source_stage_ids,
            producer_node_ids=node_ids,
        )
    except IdentityContractError as error:
        raise SpecResolutionError(str(error)) from error

    references: list[SymbolRef] = []
    for kind, resource in sorted(resources.items()):
        for pointer, string in _walk_strings(resource):
            location = f"{kind}{pointer}"
            if string.startswith("kernel:"):
                target = _strip_prefix(string, "kernel")
                if not kernel_registry.contains(string):
                    raise SpecResolutionError(
                        f"{location}: dangling kernel reference {target!r}"
                    )
                references.append(SymbolRef("kernel", target, location))
            elif string.startswith("source:"):
                if not _is_declared_source_ref_location(kind, pointer):
                    continue
                target = _strip_prefix(string, "source")
                _require(target, source_ids, namespace="source", location=location)
                references.append(SymbolRef("source", target, location))
            elif string.startswith("vintage:"):
                target = _strip_prefix(string, "vintage")
                _require(target, vintage_ids, namespace="vintage", location=location)
                references.append(SymbolRef("vintage", target, location))
            elif string.startswith("stream:"):
                target = _strip_prefix(string, "stream")
                _require(
                    target,
                    LEGACY_V1_PROTOCOL.streams,
                    namespace="stream",
                    location=location,
                )
                references.append(SymbolRef("stream", target, location))

    if channel_ids:
        for channel_index, channel_row in enumerate(
            _array(spine_map.get("channels", []), "spine/channels")
        ):
            channel = _mapping(channel_row, f"spine/channels/{channel_index}")
            source_value = channel.get("source")
            channel_sources = (
                [source_value] if isinstance(source_value, str) else source_value
            )
            for source_index, source_id in enumerate(
                _array(channel_sources, f"spine/channels/{channel_index}/source")
            ):
                if isinstance(source_id, str):
                    _require(
                        source_id,
                        source_ids,
                        namespace="source",
                        location=(
                            f"spine/channels/{channel_index}/source/{source_index}"
                        ),
                    )
        assembly = _mapping(spine_map.get("assembly", {}), "spine/assembly")
        anchor = assembly.get("mass_anchor_channel")
        if isinstance(anchor, str):
            _require(
                anchor,
                channel_ids,
                namespace="channel",
                location="spine/assembly/mass_anchor_channel",
            )
        for index, row in enumerate(_array(family_rows, "imputation/families")):
            family = _mapping(row, f"imputation/families/{index}")
            donor = _mapping(family.get("donor", {}), f"families/{index}/donor")
            if isinstance(donor.get("channel"), str):
                _require(
                    str(donor["channel"]),
                    channel_ids,
                    namespace="channel",
                    location=f"imputation/families/{index}/donor/channel",
                )
            if isinstance(donor.get("support_role"), str):
                _require(
                    str(donor["support_role"]),
                    support_ids,
                    namespace="support_role",
                    location=f"imputation/families/{index}/donor/support_role",
                )
            model = family.get("model")
            if isinstance(model, str):
                _require(
                    model,
                    model_ids,
                    namespace="model",
                    location=f"imputation/families/{index}/model",
                )

    # One compiler-owned closure pass resolves columns, virtual artifacts, and
    # the finite row-scope algebra.  Minimal country bundles may keep these
    # inventories empty; any declared inventory is closed completely.
    entity_by_id = {
        name: EntitySpec(name)
        for name in (
            "person",
            "tax_unit",
            "spm_unit",
            "household",
            "family",
            "benunit",
            "marital_unit",
            "frame",
        )
    }
    entities = tuple(entity_by_id.values())
    try:
        typed = resolve_typed_closure(resources, entities=entities)
    except TypedClosureError as error:
        raise SpecResolutionError(str(error)) from error
    frozen_generated_authorities = freeze_json(
        {} if generated_authorities is None else generated_authorities
    )
    assert isinstance(frozen_generated_authorities, FrozenMap)
    return ResolutionResult(
        references=tuple(sorted(references, key=lambda row: (row.source_path, row.id))),
        entities=entities,
        artifacts=typed.artifacts,
        scopes=typed.scopes,
        columns=typed.columns,
        vintage_authorities=vintage_authorities,
        generated_authorities=frozen_generated_authorities,
        seed_protocol=LEGACY_V1_PROTOCOL,
        seed_site_bindings=seed_site_bindings,
    )
