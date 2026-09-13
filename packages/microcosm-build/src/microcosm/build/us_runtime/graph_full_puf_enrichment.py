"""Bounded PUF profile attachment to an independently retained Population.

This is a placement owner, not source admission or a source-successor issuer.
The caller must already have qualified the complete survey predictor stages and
canonical donor. The current diagnostic survey host has only five diagnostic
features; neither its matrix nor a standalone Frame qualifies for this boundary.
No missing predictor is supplied here, and prior wages are outside this chain.

The upstream Population must already be an inherited version in which the output
incumbents can be rewritten. These two ordinary nodes add no structural version,
weight transition, or mass record. Retain independent expected state before
execution and call the materialized verifier after cold AND required warm runs.
KernelContext alone cannot expose metadata, owners or design anchors.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import chain
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import (
    graph_legacy_apply_matrix,
    graph_legacy_qrf,
    model_input,
    qrf,
    qrf_target,
)
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
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
    Owned,
    Slice,
    StructuralDelta,
    artifact_edges,
    source_hash,
)
from microcosm.graph import population as population_ops
from microcosm.graph import store as store_ops
from microcosm.graph.canonical import canonical_json
from microcosm.graph.kernel import ArtifactValue
from microcosm.graph.keys import node_key, opaque_artifact_key, platform_fingerprint
from microcosm.graph.population import Population

from . import (
    acs_income_universe,
    operator_column_contracts,
    qbi_inputs,
    support_provenance,
)
from . import full_puf_enrichment as full
from . import survey_population_replay as replay

PERSON_MASK = "full_puf_person_mask"
TAX_UNIT_MASK = "full_puf_tax_unit_mask"
MASKS = MappingProxyType({"person": PERSON_MASK, "tax_unit": TAX_UNIT_MASK})
PLACEMENT_TYPE = ArtifactType("microcosm.us.full_puf_placement", 1)
SCOPE = "full65_attachment_complete_upstream_required"
require = full._require


def _rosters(profile):
    profile = full.require_puf_output_profile(profile)
    return (("person", profile.person_outputs), ("tax_unit", profile.tax_unit_outputs))


def _scope(profile):
    profile = full.require_puf_output_profile(profile)
    return profile.value + "_attachment_complete_upstream_required"


def _profile_chain(profile, fit_nodes, apply_nodes):
    profile = full.require_puf_output_profile(profile)
    require(
        not {"prior_year_wages", "employment_income_last_year"} & set(profile.targets)
        and type(fit_nodes) is type(apply_nodes) is tuple
        and len(fit_nodes) == len(apply_nodes) == len(profile.targets),
        "FULL_PUF_CHAIN_ROSTER",
    )
    require(
        all(
            type(fit) is type(apply) is Node
            and fit.params.get("predictors") == profile.predictors
            and fit.params.get("targets") == profile.targets
            and fit.params.get("target") == apply.params.get("target") == target
            and fit.params.get("phase") == apply.params.get("phase") == profile.phase
            for fit, apply, target in zip(
                fit_nodes, apply_nodes, profile.targets, strict=True
            )
        ),
        "FULL_PUF_CHAIN_PROFILE",
    )


def _digest(parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _series_parts(series):
    yield repr(series.dtype).encode()
    if pd.api.types.is_object_dtype(series.dtype):
        # Portable typed scalar bytes, never Python object pointers.
        yield from (store_ops._encode_object_scalar(v) for v in series)
    else:
        yield from population_ops._storage_parts(
            series, np.ones(len(series), dtype=np.bool_)
        )


def _axis_parts(axis):
    require(not isinstance(axis, pd.MultiIndex), "FULL_PUF_AXIS")
    yield type(axis).__name__.encode()
    yield canonical_json(store_ops._axis_name_payload(axis.name))
    yield from _series_parts(pd.Series(axis.array, copy=False))


def _table_stamp(table):
    return _digest(
        chain(
            _axis_parts(table.index),
            _axis_parts(table.columns),
            (str(table.flags.allows_duplicate_labels).encode(),),
            (part for column in table for part in _series_parts(table[column])),
        )
    )


def _population_stamp(population):
    """In-process mutation seal only; never source or successor qualification."""
    require(type(population) is Population, "FULL_PUF_POPULATION_REQUIRED")
    frame = population.frame
    return _digest(
        (
            canonical_json(
                {
                    "schema": asdict(frame.schema),
                    "entities": frame.entities,
                    "links": frame.links,
                    "metadata": store_ops._encode_frame_metadata(frame.metadata),
                    "mass_log": [asdict(r) for r in frame.mass_log],
                    "version": population.version,
                    "owners": sorted(population.owners.items()),
                    "weight_kind": tuple(population.weight_kind.items()),
                    "mass_ledger": [asdict(r) for r in population.mass_ledger],
                }
            ),
            *(_table_stamp(frame.table(e)).encode() for e in frame.entities),
            *_axis_parts(frame.strata.index),
            canonical_json(store_ops._axis_name_payload(frame.strata.name)),
            *_series_parts(frame.strata),
            *(
                canonical_json(
                    (
                        e,
                        frame.weights_for(e).kind,
                        str(frame.weights_for(e).values.dtype),
                        frame.weights_for(e).values.shape,
                    )
                )
                + frame.weights_for(e).values.tobytes()
                for e in frame.weighted_entities
            ),
            *(
                canonical_json((e, str(a.dtype), a.shape)) + a.tobytes()
                for e, a in population.design_weights.items()
            ),
        )
    )


def _structural(frame, entity):
    schema = frame.schema
    return (
        schema.entity_id_column(entity),
        *(
            tuple(schema.membership_column(e) for e in schema.group_entities)
            if entity == schema.person_entity
            else ()
        ),
    )


def _inputs(frame, *, masks=False, profile=full.FULL65):
    coordinates = frozenset((e, c) for e, cols in _rosters(profile) for c in cols)
    result = []
    for entity in frame.entities:
        columns = tuple(
            c
            for c in frame.table(entity)
            if c not in _structural(frame, entity)
            and (not masks or (entity, c) not in coordinates)
        )
        # Full attachment must project every table, including its structural IDs.
        # The graph cannot express a Slice with zero nonstructural columns.
        require(columns or masks, "FULL_PUF_EMPTY_ENTITY_PROJECTION:" + entity)
        if columns:
            result.append(Slice(entity, columns))
    return tuple(result)


def _outputs(frame, *, profile=full.FULL65):
    result = []
    for entity, columns in _rosters(profile):
        table = frame.table(entity)
        for column in columns:
            boolean = column in full.support._PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS
            dtype = "boolean" if boolean else "float64"
            if column in table:
                token = population_ops.token_for_dtype(table[column].dtype)
                require(
                    token in ({"bool", "boolean"} if boolean else {"float64"}),
                    "FULL_PUF_INCUMBENT_DTYPE:" + column,
                )
                dtype = token
            result.append(
                Owned(
                    entity,
                    column,
                    dtype,
                    rows=MASKS[entity],
                    rewrite=column in table,
                )
            )
    return tuple(result)


def _masks(frame):
    masks = {
        entity: full.support.puf_tax_detail_clone_mask(
            frame.table(entity), entity=entity
        )
        for entity in MASKS
    }
    person, tax_unit = frame.table("person"), frame.table("tax_unit")
    ids = tax_unit.loc[masks["tax_unit"], "tax_unit_id"]
    require(
        bool(masks["person"].any())
        and bool(masks["tax_unit"].any())
        and np.array_equal(masks["person"], person.person_tax_unit_id.isin(ids))
        and set(person.loc[masks["person"], "person_tax_unit_id"]) == set(ids),
        "FULL_PUF_MASK_MEMBERSHIP",
    )
    return masks


@dataclass(frozen=True)
class FullPufAttachmentBinding:
    """Retained profile and comparison inputs; no source authority is conferred."""

    population: Population
    expected_population: Population
    population_node: Node
    input_owners: Mapping
    donor: pd.DataFrame
    predictor_known: pd.DataFrame
    matrix: ArtifactValue
    matrix_edge: ArtifactInput
    fit_nodes: tuple[Node, ...]
    apply_nodes: tuple[Node, ...]
    prefix: str
    population_stamp: str
    expected_stamp: str
    donor_stamp: str
    known_stamp: str
    profile: full.PufOutputProfile = full.FULL65

    def check(self):
        _profile_chain(self.profile, self.fit_nodes, self.apply_nodes)
        require(
            _population_stamp(self.population) == self.population_stamp
            and _population_stamp(self.expected_population) == self.expected_stamp,
            "FULL_PUF_RETAINED_POPULATION_CHANGED",
        )
        require(_table_stamp(self.donor) == self.donor_stamp, "FULL_PUF_DONOR_CHANGED")
        require(
            _table_stamp(self.predictor_known) == self.known_stamp,
            "FULL_PUF_KNOWNNESS_CHANGED",
        )
        require(
            dict(self.input_owners) == dict(self.expected_population.owners),
            "FULL_PUF_INPUT_OWNERS",
        )


def _value(edge, value, *, producer_key=None):
    require(
        type(value) is ArtifactValue
        and value.type == edge.type
        and value.key == opaque_artifact_key(value.producer_key, edge.artifact)
        and (producer_key is None or value.producer_key == producer_key)
        and value.numerics.numeric in (Numeric.BITWISE, Numeric.PLATFORM_BITWISE)
        and (
            value.numerics.numeric is not Numeric.PLATFORM_BITWISE
            or value.numerics.platform == platform_fingerprint()
        ),
        "FULL_PUF_ARTIFACT_IDENTITY:" + edge.name,
    )
    return value


def retain_full_puf_attachment(
    *,
    population,
    expected_population,
    population_node,
    input_owners,
    donor,
    predictor_known,
    matrix,
    fit_nodes,
    apply_nodes,
    prefix="puf_full_attachment",
    profile=full.FULL65,
):
    """Pin prepared upstream values, selected outputs and a real matrix edge.

    ``profile`` defaults to the complete legacy FULL65 roster. PUF59 owns only
    its selected columns; any excluded incumbent remains upstream-owned and
    participates only in the complete Population preservation checks.

    ``population`` is executor-observed; ``expected_population`` is a separately
    retained upstream comparison value, not derived from the attachment result.
    ``input_owners`` and ``population_node`` come from that upstream graph. The
    integrating source owner remains responsible for qualification and for opening
    a version with inherited incumbents. This function admits neither a donor nor
    a survey source and does not construct an upstream Population from a Frame.
    """
    profile = full.require_puf_output_profile(profile)
    require(
        type(population) is type(expected_population) is Population
        and population is not expected_population
        and population.frame is not expected_population.frame,
        "FULL_PUF_INDEPENDENT_POPULATION_REQUIRED",
    )
    replay.same_replayed_population(expected_population, population)
    require(
        population.frame.schema == full.support.US_SCHEMA
        and not population.frame.links,
        "FULL_PUF_US_POPULATION_REQUIRED",
    )
    require(
        type(population_node) is Node
        and population_node.id == population.version
        and population_node.base is not None
        and population_node.structural
        not in (StructuralDelta.CREATE, StructuralDelta.NONE)
        and dict(input_owners) == dict(population.owners),
        "FULL_PUF_UPSTREAM_GRAPH_BINDING",
    )
    _profile_chain(profile, fit_nodes, apply_nodes)
    first = fit_nodes[0]
    matrix_edges = tuple(
        e for e in apply_nodes[0].artifact_inputs if e.name == "matrix"
    )
    require(len(matrix_edges) == 1, "FULL_PUF_MATRIX_EDGE")
    edge = matrix_edges[0]
    require(
        edge.artifact == "matrix" and edge.type == model_input.RECIPIENT_MATRIX_TYPE,
        "FULL_PUF_MATRIX_EDGE",
    )
    expected_fits, expected_applies = full.full_puf_train_apply_nodes(
        donor_population=first.population,
        recipient_population=population.version,
        matrix_producer=edge.producer,
        seed=first.params["seed"],
        n_estimators=first.params["n_estimators"],
        zero_atol=first.params["zero_atol"],
        prefix=first.id.removesuffix(".fit.000"),
        profile=profile,
    )
    require(
        fit_nodes == expected_fits and apply_nodes == expected_applies,
        "FULL_PUF_CHAIN_DECLARATION",
    )
    # Structural nodes have no ordinary output declarations. The compiler is
    # the enforcer of same-version ownership conflicts and inherited rewrites;
    # the integrating host must compile the entire graph before execution.
    for entity, name in MASKS.items():
        require(
            name not in population.frame.table(entity), "FULL_PUF_MASK_ALREADY_PRESENT"
        )
    _outputs(population.frame, profile=profile)
    _inputs(population.frame, profile=profile)
    _masks(population.frame)
    prepared = full.prepare_full_puf_inputs(
        population.frame, donor, predictor_known=predictor_known, profile=profile
    )
    _value(edge, matrix)
    require(matrix.payload == prepared.matrix, "FULL_PUF_PREPARED_MATRIX")
    binding = FullPufAttachmentBinding(
        population,
        expected_population,
        population_node,
        MappingProxyType(dict(input_owners)),
        donor,
        predictor_known,
        matrix,
        edge,
        fit_nodes,
        apply_nodes,
        prefix,
        _population_stamp(population),
        _population_stamp(expected_population),
        _table_stamp(donor),
        _table_stamp(predictor_known),
        profile,
    )
    binding.check()
    return binding


def _params(binding):
    # The physical Population seals are in-process mutation checks only.
    # Nullable backing storage beneath nulls is canonicalized by Frame-store
    # v2. Graph column/frame keys bind the upstream values across sessions.
    return {
        "scope": _scope(binding.profile),
        "output_profile": binding.profile.value,
        "donor_values": binding.donor_stamp,
        "predictor_knownness": binding.known_stamp,
        "matrix_producer_key": binding.matrix.producer_key,
        "matrix_sha256": codec.sha(binding.matrix.payload),
        "source_admission_issued": False,
    }


def _attach_edges(binding):
    return (
        binding.matrix_edge,
        ArtifactInput(
            "placement", binding.prefix + ".mask", "placement", PLACEMENT_TYPE
        ),
        *(
            ArtifactInput(f"raw_{i:03d}", node.id, "raw_draw", codec.RAW_TARGET_TYPE)
            for i, node in enumerate(binding.apply_nodes)
        ),
        ArtifactInput(
            "apply_state",
            binding.apply_nodes[-1].id,
            "apply_state",
            graph_legacy_apply_matrix.MATRIX_APPLY_STATE_TYPE,
        ),
        ArtifactInput(
            "training_state",
            binding.fit_nodes[-1].id,
            "training_state",
            codec.TRAINING_STATE_TYPE,
        ),
        ArtifactInput(
            "last_model",
            binding.fit_nodes[-1].id,
            "model",
            qrf_target.LEGACY_QRF_TARGET_TYPE,
        ),
    )


def full_puf_attachment_nodes(binding):
    binding.check()
    frame = binding.expected_population.frame
    masks = Node(
        binding.prefix + ".mask",
        FullPufMaskKernel.ref,
        population=binding.population.version,
        # Reading fields rewritten by attach would create a graph dependency cycle.
        inputs=_inputs(frame, masks=True, profile=binding.profile),
        outputs=tuple(Owned(e, c, "bool") for e, c in MASKS.items()),
        params=_params(binding),
        artifact_inputs=(binding.matrix_edge,),
        artifact_outputs=(ArtifactOutput("placement", PLACEMENT_TYPE),),
    )
    inputs = list(_inputs(frame, profile=binding.profile))
    for entity, mask in MASKS.items():
        for i, item in enumerate(inputs):
            if item.entity == entity:
                inputs[i] = Slice(entity, (*item.columns, mask))
                break
        else:
            inputs.append(Slice(entity, (mask,)))
    attach = Node(
        binding.prefix + ".attach",
        FullPufAttachKernel.ref,
        population=binding.population.version,
        inputs=tuple(inputs),
        outputs=_outputs(frame, profile=binding.profile),
        params=_params(binding),
        artifact_inputs=_attach_edges(binding),
    )
    return masks, attach


def _context_projection(context, node, frame):
    """Check the expected declaration's complete projection, never actual-chosen columns."""
    require(
        context.node == node and dict(context.params) == dict(node.params),
        "FULL_PUF_DECLARATION",
    )
    require(not context.sources, "FULL_PUF_UNDECLARED_SOURCE")
    entities = {s.entity for s in node.inputs} | {o.entity for o in node.outputs}
    require(set(context.tables) == entities, "FULL_PUF_CONTEXT_TABLES")
    weights = {}
    for entity in sorted(entities):
        columns = list(_structural(frame, entity))
        columns.extend(c for s in node.inputs if s.entity == entity for c in s.columns)
        columns.extend(
            o.column for o in node.outputs if o.entity == entity and o.rewrite
        )
        expected = frame.table(entity).loc[
            np.ones(frame.n(entity), dtype=np.bool_), list(dict.fromkeys(columns))
        ]
        actual = context.tables[entity]
        replay._axis(expected.index, actual.index)
        replay._axis(expected.columns, actual.columns)
        require(expected.flags == actual.flags, "FULL_PUF_CONTEXT_FLAGS")
        for column in expected:
            replay._series(expected[column], actual[column])
        try:
            weights[entity] = frame.resolve_weights(entity)
        except ValueError:
            if entity in frame.weighted_entities:
                raise
    require(set(context.weights) == set(weights), "FULL_PUF_CONTEXT_WEIGHTS")
    for entity, expected in weights.items():
        actual = context.weights[entity]
        require(
            actual.kind is expected.kind
            and replay._array_bytes_equal(expected.values, actual.values),
            "FULL_PUF_CONTEXT_WEIGHT_VALUES",
        )
    strata = frame.strata.loc[np.ones(len(frame.strata), dtype=np.bool_)]
    replay._axis(strata.index, context.strata.index)
    require(
        replay._name_bytes(strata.name) == replay._name_bytes(context.strata.name),
        "FULL_PUF_CONTEXT_STRATA_NAME",
    )
    replay._series(strata, context.strata)


def _mask_result(binding):
    frame = binding.expected_population.frame
    return KernelResult(
        columns={
            (entity, MASKS[entity]): pd.Series(
                mask,
                index=frame.table(entity)[frame.schema.entity_id_column(entity)],
                dtype=bool,
            )
            for entity, mask in _masks(frame).items()
        }
    )


def _placement(binding):
    frame = binding.expected_population.frame
    return codec.encode_json(
        {
            **_params(binding),
            "target_order": list(binding.profile.targets),
            "selected_ids": {
                entity: frame.table(entity)
                .loc[mask, frame.schema.entity_id_column(entity)]
                .tolist()
                for entity, mask in _masks(frame).items()
            },
            "release_eligible": False,
        }
    )


def _checked_artifacts(binding, edges, artifacts, producer_keys=None):
    require(set(artifacts) == {e.name for e in edges}, "FULL_PUF_ARTIFACT_ROSTER")
    values = {}
    for edge in edges:
        if producer_keys is not None:
            require(edge.producer in producer_keys, "FULL_PUF_PRODUCER_ROSTER")
        values[edge.name] = _value(
            edge,
            artifacts[edge.name],
            producer_key=None
            if producer_keys is None
            else producer_keys[edge.producer],
        )
    require(values["matrix"] == binding.matrix, "FULL_PUF_MATRIX_PIN")
    if "placement" in values:
        require(
            values["placement"].payload == _placement(binding),
            "FULL_PUF_PLACEMENT_BINDING",
        )
        last_raw = f"raw_{len(binding.profile.targets) - 1:03d}"
        for names in ((last_raw, "apply_state"), ("last_model", "training_state")):
            require(
                len({values[name].producer_key for name in names}) == 1,
                "FULL_PUF_SIBLING_PRODUCER",
            )
    return values


def _finalized_columns(binding, values):
    frame = binding.expected_population.frame
    finalized, evidence = full.finalize_full_puf(
        frame,
        binding.donor,
        predictor_known=binding.predictor_known,
        matrix=values["matrix"].payload,
        matrix_producer_key=values["matrix"].producer_key,
        raw_draws={
            target: values[f"raw_{i:03d}"].payload
            for i, target in enumerate(binding.profile.targets)
        },
        apply_state=values["apply_state"].payload,
        training_state=values["training_state"].payload,
        last_model=values["last_model"].payload,
        seed=binding.fit_nodes[0].params["seed"],
        profile=binding.profile,
    )
    masks, columns = _masks(frame), {}
    for owned in _outputs(frame, profile=binding.profile):
        table, mask = finalized.table(owned.entity), masks[owned.entity]
        selected = table.loc[mask, owned.column]
        require(selected.notna().all(), "FULL_PUF_FINALIZED_UNKNOWN:" + owned.column)
        require(
            np.isfinite(selected.to_numpy(dtype=np.float64)).all(),
            "FULL_PUF_FINALIZED_NONFINITE:" + owned.column,
        )
        # Cast only owned values: an incumbent physical bool must remain bool.
        columns[(owned.entity, owned.column)] = pd.Series(
            selected.astype(owned.dtype).array,
            index=table.loc[mask, frame.schema.entity_id_column(owned.entity)],
            dtype=owned.dtype,
        )
    binding.check()
    return KernelResult(columns=columns, receipt=evidence)


class _FullPufKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, binding):
        self.binding = binding

    def implementation_hash(self):
        # Independent placement closure; never refresh a source owner's inventory.
        return source_hash(
            sys.modules[__name__],
            full,
            full.support,
            support_provenance,
            operator_column_contracts,
            qbi_inputs,
            acs_income_universe,
            replay,
            population_ops,
            store_ops,
            artifact_edges,
            codec,
            model_input,
            qrf,
            qrf_target,
            graph_legacy_qrf,
            graph_legacy_apply_matrix,
            dependencies=self.capabilities.dependencies,
        )


class FullPufMaskKernel(_FullPufKernel):
    ref = "us.full_puf.mask@1"

    def run(self, context):
        binding = self.binding
        node, _ = full_puf_attachment_nodes(binding)
        _context_projection(context, node, binding.expected_population.frame)
        _checked_artifacts(binding, node.artifact_inputs, context.artifacts)
        return KernelResult(
            columns=_mask_result(binding).columns,
            artifacts={"placement": _placement(binding)},
            receipt={
                "scope": _scope(binding.profile),
                "output_profile": binding.profile.value,
                "source_admission_issued": False,
            },
        )


class FullPufAttachKernel(_FullPufKernel):
    ref = "us.full_puf.attach@1"

    def run(self, context):
        binding = self.binding
        mask, attach = full_puf_attachment_nodes(binding)
        expected_masked = population_ops.patch(
            binding.expected_population, mask, _mask_result(binding)
        )
        _context_projection(context, attach, expected_masked.frame)
        values = _checked_artifacts(binding, attach.artifact_inputs, context.artifacts)
        return _finalized_columns(binding, values)


def verify_materialized_full_puf_attachment(
    binding, *, upstream_population, population, artifacts, producer_keys
):
    """Mandatory independent complete Population comparison, cold and replay.

    Pass actual executor-observed Populations and actual loaded typed artifacts;
    ``producer_keys`` must come from the same run manifest, not payload labels.
    ``load_full_puf_attachment_artifacts`` performs the manifest/store descriptor
    checks. These comparisons issue no source, successor, or release evidence.
    """
    binding.check()
    replay.same_replayed_population(binding.expected_population, upstream_population)
    mask, attach = full_puf_attachment_nodes(binding)
    values = _checked_artifacts(
        binding, attach.artifact_inputs, artifacts, producer_keys
    )
    result = _finalized_columns(binding, values)
    expected = population_ops.patch(
        binding.expected_population, mask, _mask_result(binding)
    )
    expected = population_ops.patch(expected, attach, result)
    replay.same_replayed_population(expected, population)
    binding.check()
    return {
        **result.receipt,
        "scope": _scope(binding.profile),
        "output_profile": binding.profile.value,
        "complete_population_compared": True,
    }


def _manifest_contract(binding, compiled, manifest):
    """Bind declarations and real kernel identities for the selected complete chain."""
    mask, attach = full_puf_attachment_nodes(binding)
    require(
        compiled.graph.node(binding.population_node.id) == binding.population_node,
        "FULL_PUF_MANIFEST_UPSTREAM_DECLARATION",
    )
    kernels = (
        LegacyQRFTrainKernel(),
        graph_legacy_apply_matrix.LegacyQRFApplyMatrixKernel(),
        FullPufMaskKernel(binding),
        FullPufAttachKernel(binding),
    )
    contracts = {k.ref: (k.capabilities, k.implementation_hash()) for k in kernels}
    keys = {name: record.key for name, record in manifest.nodes.items()}
    for node in (*binding.fit_nodes, *binding.apply_nodes, mask, attach):
        require(compiled.graph.node(node.id) == node, "FULL_PUF_MANIFEST_DECLARATION")
        record = manifest.node(node.id)
        capabilities, implementation = contracts[node.kernel]
        require(
            record.kernel_ref == node.kernel
            and record.capabilities == capabilities
            and record.kernel_impl_hash == implementation
            and record.key
            == node_key(
                compiled,
                node.id,
                keys,
                implementation,
                {},
                kernel_capabilities=capabilities,
            ),
            "FULL_PUF_MANIFEST_KERNEL:" + node.id,
        )
        expected_outputs = {
            output.name: artifact_edges.descriptor(
                producer=node.id,
                artifact=output.name,
                type_=output.type,
                producer_key=record.key,
                capabilities=capabilities,
            )
            for output in node.artifact_outputs
        }
        require(
            record.typed_artifacts["outputs"] == expected_outputs
            and dict(record.opaque_artifacts)
            == {name: d["key"] for name, d in expected_outputs.items()},
            "FULL_PUF_MANIFEST_OUTPUTS:" + node.id,
        )
        expected_inputs = {}
        for edge in node.artifact_inputs:
            producer = manifest.node(edge.producer)
            descriptor = artifact_edges.descriptor(
                producer=edge.producer,
                artifact=edge.artifact,
                type_=edge.type,
                producer_key=producer.key,
                capabilities=producer.capabilities,
            )
            require(
                producer.typed_artifacts["outputs"][edge.artifact] == descriptor,
                "FULL_PUF_MANIFEST_ANCESTRY:" + edge.name,
            )
            expected_inputs[edge.name] = descriptor
        require(
            record.typed_artifacts["inputs"] == expected_inputs,
            "FULL_PUF_MANIFEST_INPUTS:" + node.id,
        )


def load_full_puf_attachment_artifacts(binding, *, compiled, manifest, store):
    """Load real producer bytes after verifying the graph and both ends of edges.

    ``compiled`` is the actual graph used in this run. Recompute selected node keys
    from its exact declarations and maintained kernel identities before loading
    trusted model bytes. Upstream source qualification remains the caller's duty.
    """
    _manifest_contract(binding, compiled, manifest)
    _, attach = full_puf_attachment_nodes(binding)
    consumer = manifest.node(attach.id)
    require(consumer.kernel_ref == attach.kernel, "FULL_PUF_MANIFEST_CONSUMER")
    descriptors = consumer.typed_artifacts["inputs"]
    require(
        set(descriptors) == {e.name for e in attach.artifact_inputs},
        "FULL_PUF_MANIFEST_ROSTER",
    )
    artifacts, keys = {}, {}
    for edge in attach.artifact_inputs:
        record = manifest.node(edge.producer)
        descriptor = record.typed_artifacts["outputs"][edge.artifact]
        require(
            descriptor == descriptors[edge.name]
            and descriptor["producer"] == edge.producer
            and descriptor["artifact"] == edge.artifact
            and descriptor["producer_key"] == record.key
            and descriptor["key"]
            == record.opaque_artifacts[edge.artifact]
            == opaque_artifact_key(record.key, edge.artifact),
            "FULL_PUF_MANIFEST_EDGE:" + edge.name,
        )
        value = artifact_edges.value_from_descriptor(
            store.load_bytes(descriptor["key"]), descriptor
        )
        artifacts[edge.name] = _value(edge, value, producer_key=record.key)
        keys[edge.producer] = record.key
    _checked_artifacts(binding, attach.artifact_inputs, artifacts, keys)
    return artifacts, keys
