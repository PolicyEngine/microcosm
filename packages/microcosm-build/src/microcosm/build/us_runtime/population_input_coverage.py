"""Describe required inputs on one actual graph Population and attached manifest.

This is a diagnostic, never an issuer, source qualifier, applicability model,
release gate or claim of non-default statistical signal. Call inside an actual
host's checked lifetime after its source/producer checks. Population and manifest
constructors are public: even coherent supplied objects cannot prove original
source ancestry. There is deliberately no supplied expected hash/list shortcut.

The roster is a versioned coverage contract, not a claim that every name is
scientifically independent or consumed by every engine route. The current
source manifest does not declare grain or applicability. Present
columns report their actual grains; missing names retain unresolved grain.
Every row's applicability remains unresolved, including declared ABSENT cells.
Column ownership identifies the last writer, not the measurement source of all
rows: writer-mask counts and carried counts stay separate. Source names and
typed artifact ancestry below describe declarations/receipts, not fresh reads.

Only the current group-linked survey schema is supported. Experimental link
tables refuse, rather than being dropped by the maintained replay comparator.
The single possible borrowed I/O is the attached manifest population lookup.
All detached output is built afterward, with final pure input seals.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from microcosm.graph import CompiledGraph, Population, RunManifest, compile_graph
from microcosm.graph.availability import execution_state
from microcosm.graph.canonical import canonical_json
from microcosm.graph.decl import Ownership, StructuralDelta
from microcosm.graph.population import _storage_parts, owned_ids
from microcosm.graph.serialize import graph_to_json
from microcosm.graph.store import (
    _axis_name_payload,
    _encode_frame_metadata,
    _encode_object_scalar,
)

from .input_coverage_profile import (
    ASSIGNED_BLOCK_COLUMN,
    MANIFEST_SHA256,
    USInputProfile,
    required_us_inputs,
)
from .support_provenance import (
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
)
from .survey_population_replay import same_replayed_frame


@dataclass(frozen=True)
class CoverageGroup:
    entity: str
    origin: str
    clone: int
    id_dtype: str
    ordered_ids: tuple[int | str, ...]
    source_id_dtype: str
    ordered_source_ids: tuple[int | str, ...]


@dataclass(frozen=True)
class CoverageCounts:
    group: int
    known: int
    unknown: int
    invalid: int
    writer_rows: int
    carried_rows: int
    applicability_unresolved: int
    # Neither nulls nor Ownership.ABSENT are evidence of non-applicability.
    not_applicable: int = 0


@dataclass(frozen=True)
class InputCoverage:
    name: str
    entity: str | None
    dtype: str | None
    owner: str | None
    declaration: str
    writer_mask: str | None
    counts: tuple[CoverageCounts, ...]


@dataclass(frozen=True)
class CoverageProducer:
    node: str
    key: str
    kernel_ref: str
    implementation: str
    declared_sources: tuple[tuple[str, str], ...]
    predecessors: tuple[tuple[str, str], ...]
    typed_artifacts_json: str


@dataclass(frozen=True)
class PopulationInputCoverage:
    profile: USInputProfile
    version: str
    graph_sha256: str
    manifest_key: str
    population_storage_sha256: str
    groups: tuple[CoverageGroup, ...]
    inputs: tuple[InputCoverage, ...]
    producers: tuple[CoverageProducer, ...]
    missing_inputs: tuple[str, ...]
    ambiguous_grains: tuple[str, ...]
    assigned_block: InputCoverage
    block_storage_issues: tuple[str, ...]
    protocol: str = "microcosm.us.input_coverage_diagnostic.v1"
    profile_reference_sha256: str = MANIFEST_SHA256
    profile_authority: str = "tracked_input_coverage_contract_v1"
    source_ancestry_verified: bool = False
    applicability_complete: bool = False
    statistical_signal_verified: bool = False
    release_eligible: bool = False

    def to_bytes(self) -> bytes:
        return canonical_json(asdict(self))


def _require(condition, reason):
    if not condition:
        raise ValueError("US_INPUT_COVERAGE_" + reason)


def _series_stamp(series):
    digest = hashlib.sha256()
    digest.update(canonical_json((str(series.dtype), _axis_name_payload(series.name))))
    if pd.api.types.is_object_dtype(series.dtype):
        # Typed scalar encoding, never process-specific PyObject addresses.
        for value in series:
            part = _encode_object_scalar(value)
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    else:
        for part in _storage_parts(series, np.ones(len(series), dtype=np.bool_)):
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    return digest.hexdigest()


def _axis_stamp(axis):
    _require(not isinstance(axis, pd.MultiIndex), "MULTIINDEX")
    return (type(axis).__name__, _series_stamp(pd.Series(axis.array, name=axis.name)))


def _frame_stamp(frame):
    _require(frame.links == (), "LINK_TABLES_UNSUPPORTED")
    tables = []
    for entity in frame.entities:
        table = frame.table(entity)
        tables.append(
            (
                entity,
                _axis_stamp(table.index),
                _axis_stamp(table.columns),
                table.flags.allows_duplicate_labels,
                tuple((column, _series_stamp(table[column])) for column in table),
            )
        )
    return hashlib.sha256(
        canonical_json(
            {
                "schema": asdict(frame.schema),
                "tables": tables,
                "strata": (
                    _axis_stamp(frame.strata.index),
                    _series_stamp(frame.strata),
                ),
                "weights": tuple(
                    (
                        entity,
                        frame.weights_for(entity).kind.value,
                        frame.weights_for(entity).values.dtype.str,
                        frame.weights_for(entity).values.tobytes().hex(),
                    )
                    for entity in frame.weighted_entities
                ),
                "metadata": _encode_frame_metadata(frame.metadata),
                "mass_log": tuple(asdict(record) for record in frame.mass_log),
            }
        )
    ).hexdigest()


def _population_stamp(population):
    return hashlib.sha256(
        canonical_json(
            {
                "frame": _frame_stamp(population.frame),
                "version": population.version,
                "owners": sorted(population.owners.items()),
                "kinds": sorted(
                    (k, v.value) for k, v in population.weight_kind.items()
                ),
                "ledger": tuple(asdict(record) for record in population.mass_ledger),
                "design": tuple(
                    (k, v.dtype.str, v.shape, v.tobytes().hex())
                    for k, v in sorted(population.design_weights.items())
                ),
            }
        )
    ).hexdigest()


def _ids(series):
    _require(not series.isna().any(), "UNKNOWN_ID")
    if pd.api.types.is_integer_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
        series.dtype
    ):
        return tuple(int(value) for value in series)
    _require(isinstance(series.dtype, pd.StringDtype), "ID_DTYPE")
    return tuple(str(value) for value in series)


def _groups(frame):
    groups, selections, roles = [], [], {}
    for entity in frame.entities:
        table = frame.table(entity)
        names = (
            support_channel_column(entity),
            support_clone_index_column(entity),
            spine_source_id_column(entity),
        )
        _require(all(name in table for name in names), "ORIGIN_COLUMNS:" + entity)
        channel, clone, native = (table[name] for name in names)
        _require(
            not channel.isna().any() and set(channel) <= {"acs", "asec"},
            "ORIGIN:" + entity,
        )
        _require(
            pd.api.types.is_integer_dtype(clone.dtype)
            and not pd.api.types.is_bool_dtype(clone.dtype)
            and not clone.isna().any()
            and set(clone) <= {0, 1},
            "CLONE:" + entity,
        )
        ids = _ids(table[frame.schema.entity_id_column(entity)])
        native_ids = _ids(native)
        _require(len(set(ids)) == len(ids), "DUPLICATE_ID:" + entity)
        row_roles = tuple((str(a), int(b)) for a, b in zip(channel, clone, strict=True))
        _require(
            len(set(zip(row_roles, native_ids, strict=True))) == len(ids),
            "DUPLICATE_SOURCE_ROLE:" + entity,
        )
        roles[entity] = dict(zip(ids, row_roles, strict=True))
        for origin, role in sorted(set(row_roles)):
            selected = np.array(
                [v == (origin, role) for v in row_roles], dtype=np.bool_
            )
            groups.append(
                CoverageGroup(
                    entity,
                    origin,
                    role,
                    str(table[frame.schema.entity_id_column(entity)].dtype),
                    tuple(v for v, use in zip(ids, selected, strict=True) if use),
                    str(native.dtype),
                    tuple(
                        v for v, use in zip(native_ids, selected, strict=True) if use
                    ),
                )
            )
            selections.append(selected)
    person = frame.table(frame.schema.person_entity)
    person_ids = _ids(person[frame.schema.person_id_column])
    for entity in frame.schema.group_entities:
        members = _ids(person[frame.schema.membership_column(entity)])
        _require(
            all(
                member in roles[entity]
                and roles[entity][member] == roles[frame.schema.person_entity][pid]
                for pid, member in zip(person_ids, members, strict=True)
            ),
            "MEMBERSHIP_ORIGIN_CLONE:" + entity,
        )
    return tuple(groups), tuple(selections)


def _writer(population, compiled, entity, name):
    owner = population.owners[entity, name]
    node = compiled.graph.node(owner)
    outputs = [o for o in node.outputs if (o.entity, o.column) == (entity, name)]
    if not outputs:
        _require(node.structural is not StructuralDelta.NONE, "OWNER_DECLARATION")
        return (
            owner,
            "structural_carrier",
            None,
            np.zeros(population.frame.n(entity), dtype=np.bool_),
        )
    _require(len(outputs) == 1, "OWNER_DECLARATION")
    output = outputs[0]
    table = population.frame.table(entity)
    selected = (
        table[population.frame.schema.entity_id_column(entity)]
        .isin(owned_ids(population, output))
        .to_numpy(dtype=np.bool_)
    )
    if output.ownership is Ownership.ABSENT:
        _require(table[name].loc[selected].isna().all(), "ABSENT_VALUE")
    return owner, output.ownership.value, output.rows, selected


def _input(population, compiled, groups, selections, name, entity):
    if entity is None:
        return InputCoverage(
            name, None, None, None, "missing_unresolved_grain", None, ()
        )
    frame, table = population.frame, population.frame.table(entity)
    if name == "household_weight":
        series = pd.Series(frame.weights_for(entity).values)
        owner, declaration, mask = population.version, "typed_weight_carrier", None
        writer = np.zeros(len(series), dtype=np.bool_)
    else:
        series = table[name]
        owner, declaration, mask, writer = _writer(population, compiled, entity, name)
    unknown = series.isna().to_numpy(dtype=np.bool_)
    invalid = np.zeros(len(series), dtype=np.bool_)
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = series.to_numpy(dtype=np.float64, na_value=np.nan)
        invalid = ~unknown & ~np.isfinite(numeric)
    known = ~unknown & ~invalid
    counts = tuple(
        CoverageCounts(
            i,
            int(known[selected].sum()),
            int(unknown[selected].sum()),
            int(invalid[selected].sum()),
            int(writer[selected].sum()),
            int((~writer[selected]).sum()),
            int(selected.sum()),
        )
        for i, (group, selected) in enumerate(zip(groups, selections, strict=True))
        if group.entity == entity
    )
    return InputCoverage(
        name, entity, str(series.dtype), owner, declaration, mask, counts
    )


def diagnose_us_input_coverage(
    population: Population,
    *,
    compiled: CompiledGraph,
    manifest: RunManifest,
    profile: USInputProfile = USInputProfile.NATIONAL_CD,
) -> PopulationInputCoverage:
    """Bind descriptive coverage to actual tables and current graph ownership.

    This verifies supplied graph consistency, not actual-run/source issuance.
    The final host must retain/requalify the real upstream owners separately.
    No applicability exemption is accepted from caller metadata or a hash.
    """
    names = required_us_inputs(profile)
    _require(
        type(population) is Population
        and type(compiled) is CompiledGraph
        and type(manifest) is RunManifest,
        "TYPES",
    )
    graph_json = graph_to_json(compiled.graph)
    fresh = compile_graph(compiled.graph)
    _require(compiled == fresh, "COMPILED_GRAPH")
    _require(
        manifest.country == compiled.graph.country == "us"
        and tuple(manifest.nodes) == compiled.order,
        "MANIFEST_ROSTER",
    )
    _require(not manifest.known_failures, "MANIFEST_FAILURE")
    for node_id in compiled.order:
        record = manifest.node(node_id)
        _require(
            record.kernel_ref == compiled.graph.node(node_id).kernel
            and execution_state(record.receipt) is None,
            "PRODUCER_RECEIPT",
        )
    frame = population.frame
    frame.revalidate()
    _require(
        set(population.owners)
        == {
            (entity, str(column))
            for entity in frame.entities
            for column in frame.table(entity)
        },
        "PHYSICAL_OWNERS",
    )
    _require(
        compiled.versions.get(population.version) == population.version,
        "POPULATION_VERSION",
    )
    # CompiledGraph.owners contains explicit claims, not carried columns.
    # Every structural step reowns its complete carrier; only writers in the
    # current version replace those owners. Derive the roster from declarations
    # along the base chain, never by accepting arbitrary physical columns.
    expected_owners = {}
    version = population.version
    while True:
        for (declared_version, entity, column), owner in compiled.owners.items():
            if declared_version == version:
                expected_owners.setdefault(
                    (entity, column),
                    owner if version == population.version else population.version,
                )
        holder = compiled.graph.node(version)
        if holder.structural is StructuralDelta.CREATE:
            break
        version = holder.base
    structural = {
        (entity, frame.schema.entity_id_column(entity)) for entity in frame.entities
    }
    structural.update(
        (frame.schema.person_entity, frame.schema.membership_column(entity))
        for entity in frame.schema.group_entities
    )
    expected_owners.update(dict.fromkeys(structural, population.version))
    _require(
        expected_owners and dict(population.owners) == expected_owners, "CURRENT_OWNERS"
    )
    _require(
        population.weight_kind
        == {
            entity: frame.weights_for(entity).kind for entity in frame.weighted_entities
        },
        "WEIGHT_KIND",
    )
    before, manifest_json = _population_stamp(population), manifest.to_json()
    # A lazy manifest can perform store I/O here. Never construct detached output
    # before this final borrow. Replay normalization is restricted to this edge.
    materialized = manifest.population(population.version)
    ledger = manifest.mass_ledger(population.version)
    same_replayed_frame(frame, materialized)
    _require(
        canonical_json(tuple(asdict(r) for r in ledger))
        == canonical_json(tuple(asdict(r) for r in population.mass_ledger)),
        "MASS_LEDGER",
    )
    _require(
        _population_stamp(population) == before and manifest.to_json() == manifest_json,
        "BORROW_MUTATION",
    )
    materialized_stamp = _frame_stamp(materialized)
    groups, selections = _groups(frame)
    inputs, missing, ambiguous = [], [], []
    for name in names:
        entities = (
            ["household"]
            if name == "household_weight" and "household" in frame.weighted_entities
            else [entity for entity in frame.entities if name in frame.table(entity)]
        )
        if name == "household_weight" and entities != ["household"]:
            # Never accept a redundant ordinary column instead of typed weights.
            entities = []
        if not entities:
            missing.append(name)
            inputs.append(_input(population, compiled, groups, selections, name, None))
        else:
            if len(entities) > 1:
                ambiguous.append(name)
            inputs.extend(
                _input(population, compiled, groups, selections, name, entity)
                for entity in entities
            )
    household = frame.table("household")
    block = _input(
        population,
        compiled,
        groups,
        selections,
        ASSIGNED_BLOCK_COLUMN,
        "household" if ASSIGNED_BLOCK_COLUMN in household else None,
    )
    block_issues = []
    if block.entity is None:
        block_issues.append("missing_assigned_block")
    else:
        values = household[ASSIGNED_BLOCK_COLUMN]
        if values.isna().any():
            block_issues.append("unknown_assigned_block")
        if any(
            not isinstance(v, str) or len(v) != 15 or not v.isascii() or not v.isdigit()
            for v in values[values.notna()]
        ):
            block_issues.append("malformed_assigned_block")
    sources = {source.name: source.codec for source in compiled.graph.sources}
    producers = tuple(
        CoverageProducer(
            node_id,
            manifest.node(node_id).key,
            manifest.node(node_id).kernel_ref,
            manifest.node(node_id).kernel_impl_hash,
            tuple(
                (name, sources[name]) for name in compiled.graph.node(node_id).sources
            ),
            tuple(
                (name, manifest.node(name).key)
                for name in compiled.predecessors[node_id]
            ),
            canonical_json(manifest.node(node_id).typed_artifacts).decode(),
        )
        for node_id in compiled.order
    )
    # Pure checks after decoding and grouping. No source/store/owner borrow follows.
    _require(
        _population_stamp(population) == before
        and _frame_stamp(materialized) == materialized_stamp
        and manifest.to_json() == manifest_json
        and graph_to_json(compiled.graph) == graph_json
        and compiled == fresh,
        "FINAL_MUTATION",
    )
    return PopulationInputCoverage(
        profile,
        population.version,
        hashlib.sha256(graph_json.encode()).hexdigest(),
        manifest.key,
        before,
        groups,
        tuple(inputs),
        producers,
        tuple(missing),
        tuple(ambiguous),
        block,
        tuple(block_issues),
    )
