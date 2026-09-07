"""Bounded PUF-before-calibration integration with original-origin cap lineage.

This runner authenticates materialized boundaries on cold and REQUIRED replay.
It does not grant source/fit/scoring approval. The caller supplies the exact
reviewed source paths and pins; the real source codecs remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import InitVar, asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.fit import model_input
from microcosm.frame import WeightKind
from microcosm.graph import (
    ArtifactType,
    KernelResult,
    Numeric,
    Slice,
    SourceRef,
    compile_graph,
    run_graph,
    source_hash,
)
from microcosm.graph.artifact_edges import value_from_descriptor
from microcosm.graph.canonical import canonical_json, sha256_domain
from microcosm.graph.keys import opaque_artifact_key
from microcosm.graph.population import _mass_policy, _mass_record, storage_equal
from microcosm.graph.store import _encode_frame_metadata, _encode_object_scalar

from . import graph_national_age_counts as ages
from . import graph_puf_detail_transfer as transfer
from . import graph_puf_diagnostic_consumer as puf
from . import origin_group_budget as budgets
from . import origin_group_calibration_graph as caps
from . import puf_diagnostic_consumer as consumer

ENRICHED_SCHEMA = "microcosm.us.puf-age-enriched-origin-parent.v1"
SUCCESSOR_SCHEMA = "microcosm.us.puf-age-weight-only-origin-successor.v1"
ENRICHED_TYPE = ArtifactType("microcosm.us.puf_age_enriched_origin_parent", 1)
SUCCESSOR_TYPE = ArtifactType("microcosm.us.puf_age_weight_only_origin_successor", 1)
RECIPE = {
    "epochs": 500,
    "learning_rate": 0.05,
    "seed": 0,
    "incoming_multiplier": 4.0,
    "design_multiplier": 8.0,
    "mass": "free",
}
BOUNDARY = puf.PREFIX + ".population"
MATRIX = puf.PREFIX + ".matrix"
ATTACH = puf.PREFIX + ".attach"
COUNT = "national.age_counts"
_TOKEN = object()
detail, codec = transfer.detail, transfer.codec


def require(condition: object, reason: str) -> None:
    if not condition:
        raise budgets.OriginBudgetError("ENRICHED_" + reason)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hash() -> str:
    """Bind this admission code, all reused cap code and PUF/source identities."""
    return sha(
        canonical_json(
            {
                "runner": source_hash(
                    sys.modules[__name__],
                    budgets,
                    caps,
                    ages,
                    dependencies=("numpy", "pandas"),
                ),
                "puf": puf.DiagnosticAttachKernel().implementation_hash(),
                "solver": caps.GroupedDemographicCalibrationKernel().implementation_hash(),
            }
        )
    )


def enriched_age_count_node():
    """The exact accepted count schema, downstream of the real PUF output."""
    original = ages.national_age_count_node(population=BOUNDARY)
    return replace(
        original,
        kernel=EnrichedAgeCountKernel.ref,
        inputs=(*original.inputs, Slice("tax_unit", (detail.OUTPUT,))),
    )


class EnrichedAgeCountKernel(ages.NationalAgeCountKernel):
    """Validate one dependency projection, then execute the unchanged count code."""

    ref = "us.puf_age_enriched_counts@1"
    capabilities = replace(
        ages.NationalAgeCountKernel.capabilities, numeric=Numeric.PLATFORM_BITWISE
    )

    def implementation_hash(self):
        return implementation_hash()

    def run(self, context):
        require(
            context.node.normative() == enriched_age_count_node().normative(),
            "COUNT_DEPENDENCY_DECLARATION",
        )
        require(
            not context.artifacts and not context.sources, "COUNT_DEPENDENCY_CONTEXT"
        )
        require(
            context.tables["tax_unit"][detail.OUTPUT].dtype == np.dtype("float64"),
            "COUNT_DEPENDENCY_DTYPE",
        )
        original = ages.national_age_count_node(population=BOUNDARY)
        projected = replace(
            context,
            node=original,
            tables={e: context.tables[e] for e in ("person", "household")},
        )
        return super().run(projected)


def _physical_metadata(frame) -> bytes:
    return canonical_json(_encode_frame_metadata(frame.metadata))


def _log(population) -> bytes:
    return canonical_json([asdict(record) for record in population.mass_ledger])


def _frame_log(frame) -> bytes:
    return canonical_json([asdict(record) for record in frame.mass_log])


def cache_canonical_storage_equal(left, right) -> bool:
    """Only a replay boundary may canonicalize masked integer/boolean slots.

    The accepted nullable-integer-v1 / nullable-boolean-v1 store encodings
    require zero backing in missing slots. Dtype, mask and observed bits are
    still exact. Other dtypes, including nullable Float64, remain physical.
    This function is never used for an in-phase kernel transition.
    """
    if left.dtype != right.dtype or len(left) != len(right):
        return False
    if isinstance(left.array, (pd.arrays.IntegerArray, pd.arrays.BooleanArray)):
        mask = left.isna().to_numpy(dtype=bool)
        return np.array_equal(
            mask, right.isna().to_numpy(dtype=bool)
        ) and storage_equal(left, right, ~mask)
    return storage_equal(left, right)


def _same_axis(left, right):
    return _physical_axis_name(left) == _physical_axis_name(right) and storage_equal(
        pd.Series(left.array), pd.Series(right.array)
    )


def cache_strata_equal(left, right):
    """Compare replayed object strata by the accepted store's scalar tags.

    Object ndarray bytes are memory addresses. This rule is limited to strata;
    it never replaces the physical comparator for Frame data columns.
    """
    if left.dtype != right.dtype or not _same_axis(left.index, right.index):
        return False
    if left.dtype != np.dtype("object"):
        return cache_canonical_storage_equal(left, right)
    return np.array_equal(left.isna().to_numpy(), right.isna().to_numpy()) and tuple(
        _encode_object_scalar(v) for v in left.to_numpy(object)
    ) == tuple(_encode_object_scalar(v) for v in right.to_numpy(object))


def _same_frame(a, b, *, additions=None, reweight=False, replay=False) -> None:
    additions = additions or {}
    a.revalidate()
    b.revalidate()
    require(a.schema == b.schema and a.entities == b.entities, "SCHEMA")
    require(_physical_metadata(a) == _physical_metadata(b), "FRAME_METADATA")
    require(_frame_log(a) == _frame_log(b), "FRAME_MASS_LOG")
    require(a.weighted_entities == b.weighted_entities, "WEIGHT_ROSTER")
    equal = cache_canonical_storage_equal if replay else storage_equal
    for entity in a.entities:
        left, right = a.table(entity), b.table(entity)
        extra = tuple(additions.get(entity, ()))
        require(not set(extra) & set(left.columns), "ADDITION_COLLISION")
        require(
            list(right.columns) == [*left.columns, *extra]
            and _physical_axis_name(left.columns) == _physical_axis_name(right.columns)
            and _same_axis(left.index, right.index),
            "ORDERED_COLUMNS_OR_AXIS",
        )
        require(all(equal(left[c], right[c]) for c in left), "CARRIED_CELL")
        if not reweight:
            # Donor/source Frames need not define effective weights on all tables.
            if entity not in a.weighted_entities:
                continue
            old, new = a.resolve_weights(entity), b.resolve_weights(entity)
            require(
                old.kind is new.kind and old.values.tobytes() == new.values.tobytes(),
                "CARRIED_WEIGHT",
            )
    require(
        (a.strata is None and b.strata is None)
        or (
            a.strata is not None
            and b.strata is not None
            and _same_axis(a.strata.index, b.strata.index)
            and (
                cache_strata_equal(a.strata, b.strata)
                if replay
                else storage_equal(a.strata, b.strata)
            )
        ),
        "STRATA",
    )


def _physical_axis_name(axis):
    return canonical_json(_encode_frame_metadata({"name": axis.name}))


def same_carried(before, after, *, additions=None, reweight=False) -> None:
    """Strict physical comparison within one actual runtime phase."""
    require(before.frame.weighted_entities == ("household",), "WEIGHT_ROSTER")
    _same_frame(before.frame, after.frame, additions=additions, reweight=reweight)
    if not reweight:
        for entity in before.frame.entities:
            old, new = (p.frame.resolve_weights(entity) for p in (before, after))
            require(
                old.kind is new.kind and old.values.tobytes() == new.values.tobytes(),
                "CARRIED_WEIGHT",
            )
    require(budgets._anchors(before) == budgets._anchors(after), "ORIGINAL_ANCHORS")


def verify_cache_replay(before, after):
    """Check a prior phase against the current replay, never an ordinary patch."""
    _same_frame(before.frame, after.frame, replay=True)
    require(
        before.version == after.version and before.owners == after.owners,
        "REPLAY_OWNERSHIP",
    )
    require(_log(before) == _log(after), "REPLAY_LEDGER")
    require(
        tuple(before.design_weights) == tuple(after.design_weights)
        and all(
            old.dtype == after.design_weights[e].dtype
            and old.tobytes() == after.design_weights[e].tobytes()
            for e, old in before.design_weights.items()
        ),
        "REPLAY_ORIGINAL_ANCHORS",
    )


def verify_donor(manifest, store, materialized, expected_frame, expected_payload):
    """Reconcile the actual complete20 donor before any fit can execute."""
    _same_frame(expected_frame, materialized.frame, replay=True)
    require(materialized.frame.weighted_entities == ("tax_unit",), "DONOR_WEIGHTS")
    weights = materialized.frame.weights_for("tax_unit")
    require(weights.kind is WeightKind.DESIGN, "DONOR_DESIGN")
    require(
        tuple(materialized.design_weights) == ("tax_unit",)
        and materialized.design_weights["tax_unit"].tobytes()
        == weights.values.tobytes()
        and materialized.mass_ledger == (),
        "DONOR_ORIGINAL_ANCHOR",
    )
    value = edge_value(
        manifest, store, puf.PREFIX + ".donor", "transport", puf.TRANSPORT_TYPE
    )
    require(value.payload == expected_payload, "DONOR_TRANSPORT")
    require(
        manifest.node(puf.PREFIX + ".donor").receipt["transport_sha256"]
        == sha(expected_payload),
        "DONOR_TRANSPORT_RECEIPT",
    )


def check_ledger_append(before, after, node, *, partition) -> None:
    """Recompute the one actual structural record and preserve its whole prefix."""
    expected = _mass_record(
        before.frame,
        after.frame,
        node,
        KernelResult(),
        _mass_policy(node),
        mass_partition=partition,
    )
    require(
        _log(after)
        == canonical_json([asdict(r) for r in (*before.mass_ledger, expected)]),
        "GRAPH_LEDGER_APPEND",
    )


def edge_value(manifest, store, node_id, name, expected_type):
    """Resolve the actual typed output; a caller envelope is not authority."""
    receipt = manifest.node(node_id)
    payload = store.load_bytes(receipt.opaque_artifacts[name])
    value = value_from_descriptor(payload, receipt.typed_artifacts["outputs"][name])
    require(
        value.type == expected_type
        and value.producer_key == receipt.key
        and value.key
        == receipt.opaque_artifacts[name]
        == opaque_artifact_key(receipt.key, name),
        "TYPED_PRODUCER",
    )
    return value


def edge_pin(value) -> dict:
    return {
        "producer_key": value.producer_key,
        "artifact_key": value.key,
        "payload_sha256": sha(value.payload),
    }


def host_values(manifest, store, *, native_binding, clone_binding, expected):
    edges = puf.host_edges(native_binding=native_binding, clone_binding=clone_binding)
    require(set(expected) == {e.name for e in edges}, "HOST_PIN_ROSTER")
    result = {}
    for e in edges:
        result[e.name] = edge_value(manifest, store, e.producer, e.artifact, e.type)
        require(edge_pin(result[e.name]) == expected[e.name], "HOST_PIN")
    return result


def verify_matrix(manifest, store, boundary, materialized, *, values, host_pins, route):
    """Rebuild exact mask, matrix and placement without calling a graph kernel."""
    same_carried(boundary, materialized, additions={"tax_unit": (detail.MASK,)})
    require(_log(boundary) == _log(materialized), "MATRIX_LEDGER")
    scope = consumer.qualify_host_population(boundary.frame, values)
    scope.update(route=route, actual_producer_edges=host_pins)
    expected, mask = consumer.recipient_matrix(boundary.frame)
    matrix = edge_value(
        manifest, store, MATRIX, "matrix", model_input.RECIPIENT_MATRIX_TYPE
    )
    placement = edge_value(manifest, store, MATRIX, "placement", puf.PLACEMENT_TYPE)
    require(matrix.producer_key == placement.producer_key, "MATRIX_SIBLINGS")
    require(matrix.payload == expected, "MATRIX_BYTES")
    actual = materialized.frame.table("tax_unit")[detail.MASK]
    require(
        actual.dtype == np.dtype("bool") and np.array_equal(actual.to_numpy(), mask),
        "MASK",
    )
    binding = {
        "scope": consumer.SCOPE,
        "population": BOUNDARY,
        "host_content_sha256": detail.population_content(boundary.frame),
        "source_qualified_host": scope,
        "matrix_sha256": sha(expected),
        "features": list(detail.FEATURES),
        "targets": [detail.TARGET],
        "output": detail.OUTPUT,
        "selected_ids": boundary.frame.table("tax_unit")
        .tax_unit_id.to_numpy()[mask]
        .tolist(),
    }
    require(placement.payload == codec.encode_json(binding), "PLACEMENT_BYTES")
    return matrix, placement, mask


def verify_draw(manifest, store, matrix, placement, mask, attached):
    """Authenticate materialized OUTPUT against actual raw/state artifacts on hits."""
    inputs = manifest.node(ATTACH).typed_artifacts["inputs"]
    raw_spec, state_spec = inputs["raw_draw"], inputs["apply_state"]
    raw = value_from_descriptor(store.load_bytes(raw_spec["key"]), raw_spec)
    state = value_from_descriptor(store.load_bytes(state_spec["key"]), state_spec)
    require(
        raw.type == codec.RAW_TARGET_TYPE
        and state.type == transfer.MATRIX_APPLY_STATE_TYPE
        and raw.producer_key == state.producer_key,
        "RAW_STATE_TYPES",
    )
    # Authenticate both payloads against their actual producer receipt too.
    producers = {record.key: name for name, record in manifest.nodes.items()}
    require(raw.producer_key in producers, "RAW_STATE_PRODUCER")
    producer = producers[raw.producer_key]
    for name, value in (("raw_draw", raw), ("apply_state", state)):
        actual = edge_value(manifest, store, producer, name, value.type)
        require(actual == value, "RAW_STATE_EDGE")
    require(
        inputs["matrix"]["key"] == matrix.key
        and inputs["placement"]["key"] == placement.key,
        "ATTACH_MATRIX_EDGE",
    )
    packet = transfer.decode_matrix_apply_state(state.payload)
    require(
        packet["matrix_sha256"] == sha(matrix.payload)
        and packet["matrix_producer_key"] == matrix.producer_key,
        "STATE_MATRIX",
    )
    application, chain = codec.read_application(
        codec.encode_json(packet["application"])
    )
    prepared = model_input.decode_recipient_matrix(matrix.payload)
    require(
        tuple(chain.completed_targets) == tuple(chain.targets) == (detail.TARGET,)
        and tuple(chain.predictors) == detail.FEATURES
        and chain.entity == prepared.entity == "tax_unit"
        and chain.recipient_index
        == transfer.qrf._index_identity(prepared.features.index)
        and application["seed"] == 578
        and application["raw_targets"]
        == [{"target": detail.TARGET, "sha256": sha(raw.payload)}],
        "RAW_STATE_BINDING",
    )
    draws = codec.read_raw_target(
        raw.payload, target=detail.TARGET, index=prepared.features.index
    )
    table = attached.frame.table("tax_unit")
    output = table[detail.OUTPUT]
    require(output.dtype == np.dtype("float64"), "OUTPUT_DTYPE")
    require(
        np.array_equal(table.tax_unit_id.to_numpy()[mask], prepared.entity_ids)
        and np.array_equal(output.isna().to_numpy(), ~mask)
        and output.to_numpy()[mask].tobytes()
        == np.asarray(draws, dtype="float64").tobytes(),
        "OUTPUT_RAW_BYTES",
    )
    require(np.array_equal(table[detail.MASK].to_numpy(), mask), "ATTACH_MASK")
    return {
        "matrix": edge_pin(matrix),
        "placement": edge_pin(placement),
        "raw_draw": edge_pin(raw),
        "apply_state": edge_pin(state),
    }


def verify_counts(before, after) -> None:
    """Recount each observed person by membership independently of cached output."""
    bands = ages.NATIONAL_AGE_ACTIVATION.bands
    same_carried(before, after, additions={"household": tuple(b.column for b in bands)})
    require(_log(before) == _log(after), "COUNT_LEDGER")
    p, h = before.frame.table("person"), after.frame.table("household")
    age = ages._ages(p)
    memberships = p.person_household_id.to_numpy()
    ids = h.household_id.to_numpy()
    for band in bands:
        inside = (age >= band.low) & (True if band.high is None else age <= band.high)
        expected = np.asarray(
            [np.count_nonzero(inside & (memberships == i)) for i in ids], dtype="int64"
        )
        column = h[band.column]
        require(
            column.dtype == np.dtype("int64")
            and column.to_numpy().tobytes() == expected.tobytes(),
            "COUNT_VALUES",
        )


@dataclass(frozen=True)
class EnrichedBinding:
    """Issued only after actual full-boundary comparison; transport is not admission."""

    payload: bytes
    _token: InitVar[object] = None
    _digest: str = field(init=False, repr=False)

    def __post_init__(self, _token):
        require(_token is _TOKEN and type(self.payload) is bytes, "BINDING_CONSTRUCTOR")
        object.__setattr__(self, "_digest", sha(self.payload))

    @property
    def digest(self):
        require(sha(self.payload) == self._digest, "BINDING_MUTATION")
        return self._digest

    @property
    def document(self):
        _ = self.digest
        return json.loads(self.payload)


def enriched_binding(
    budget,
    clone,
    boundary,
    matrix_pop,
    attached,
    counted,
    *,
    manifest,
    store,
    graph,
    host,
    host_pins,
    route,
):
    """Full original-clone → PUF → age-count admission, independently of run/hit."""
    require(
        budgets._frame_identity(clone.frame)
        == budget.document["expanded_population_content_sha256"],
        "ORIGINAL_CLONE",
    )
    same_carried(clone, boundary)
    check_ledger_append(
        clone, boundary, graph.node(BOUNDARY), partition=graph.mass_partition
    )
    matrix, placement, mask = verify_matrix(
        manifest,
        store,
        boundary,
        matrix_pop,
        values=host,
        host_pins=host_pins,
        route=route,
    )
    same_carried(matrix_pop, attached, additions={"tax_unit": (detail.OUTPUT,)})
    require(_log(matrix_pop) == _log(attached), "ATTACH_LEDGER")
    transfer.verify_materialized_transfer(
        boundary.frame,
        attached.frame,
        placement.payload,
        before_design=boundary.design_weights,
        after_design=attached.design_weights,
        before_ledger=boundary.mass_ledger,
        after_ledger=attached.mass_ledger,
        boundary_node=BOUNDARY,
    )
    edges = verify_draw(manifest, store, matrix, placement, mask, attached)
    verify_counts(attached, counted)
    require(
        budgets._anchors(counted)
        == budget.document["expanded_original_design_anchors"],
        "BUDGET_ANCHORS",
    )
    budget.grouped_bounds.check(
        counted.frame.weights_for("household").values, positive=True
    )
    require(
        graph.node(COUNT).normative() == enriched_age_count_node().normative(),
        "COUNT_DECLARATION",
    )
    return EnrichedBinding(
        canonical_json(
            {
                "schema": ENRICHED_SCHEMA,
                "budget_sha256": budget.digest,
                "original_clone_content_sha256": budget.document[
                    "expanded_population_content_sha256"
                ],
                "enriched_content_sha256": budgets._frame_identity(counted.frame),
                "enriched_metadata_sha256": sha(_physical_metadata(counted.frame)),
                "enriched_graph_ledger_sha256": sha(_log(counted)),
                "enriched_frame_mass_log_sha256": sha(_frame_log(counted.frame)),
                "enriched_population_version": counted.version,
                "puf_host_cells_sha256": detail.population_content(boundary.frame),
                "puf_edges": edges,
                "host_edges": host_pins,
                "count_node_key": manifest.node(COUNT).key,
                "count_artifacts": {
                    f"{e}.{c}": key
                    for (e, c), key in manifest.node(COUNT).artifacts.items()
                },
                "runner_implementation_sha256": implementation_hash(),
                "scope": consumer.SCOPE,
                "recipe": RECIPE,
                "release_eligible": False,
            }
        ),
        _token=_TOKEN,
    )


def weight_successor(parent, budget, before, after, *, node, partition):
    require(
        type(parent) is EnrichedBinding
        and parent.document["schema"] == ENRICHED_SCHEMA,
        "PARENT_TYPE",
    )
    require(
        parent.document["budget_sha256"] == budget.digest
        and parent.document["enriched_content_sha256"]
        == budgets._frame_identity(before.frame),
        "PARENT_CONTENT",
    )
    require(
        parent.document["enriched_graph_ledger_sha256"] == sha(_log(before))
        and parent.document["enriched_frame_mass_log_sha256"]
        == sha(_frame_log(before.frame)),
        "PARENT_LEDGER",
    )
    same_carried(before, after, reweight=True)
    require(
        before.frame.weights_for("household").kind is WeightKind.IMPORTANCE
        and after.frame.weights_for("household").kind is WeightKind.CALIBRATED,
        "WEIGHT_TRANSITION",
    )
    budgets._derived_weights(after.frame)
    require(
        budgets._anchors(after) == budget.document["expanded_original_design_anchors"],
        "SUCCESSOR_ANCHORS",
    )
    check_ledger_append(before, after, node, partition=partition)
    require(
        all(owner == after.version for owner in after.owners.values()),
        "SUCCESSOR_OWNERSHIP",
    )
    budget.grouped_bounds.check(
        before.frame.weights_for("household").values, positive=True
    )
    values = after.frame.weights_for("household").values
    budget.grouped_bounds.check(values, positive=True)
    return EnrichedBinding(
        canonical_json(
            {
                "schema": SUCCESSOR_SCHEMA,
                "budget_sha256": budget.digest,
                "enriched_parent_sha256": parent.digest,
                "old_content_sha256": budgets._frame_identity(before.frame),
                "new_content_sha256": budgets._frame_identity(after.frame),
                "graph_ledger_sha256": sha(_log(after)),
                "original_anchors": budgets._anchors(after),
                "accepted_weight_sha256": sha(values.tobytes()),
                "diagnostics": budget.diagnostics(values),
                "release_eligible": False,
            }
        ),
        _token=_TOKEN,
    )


def verify_final_diagnostics(final, store, budget, current):
    """Reconcile budget, receipts, accepted float64 weights and diagnostics."""
    receipt = final.node(caps.CALIBRATION_NODE)
    values = current.frame.weights_for("household").values
    corrected = receipt.receipt["last_corrected_group_count"]
    diagnostic = edge_value(
        final,
        store,
        caps.CALIBRATION_NODE,
        "diagnostics",
        caps.demographic.DIAGNOSTICS_TYPE,
    )
    document = json.loads(diagnostic.payload)
    materialized_budget = edge_value(
        final, store, caps.BUDGET_NODE, "budget", budgets.BUDGET_TYPE
    )
    require(materialized_budget.payload == budget.payload, "MATERIALIZED_BUDGET")
    expected = budget.diagnostics(values, corrected)
    require(
        canonical_json(document) == diagnostic.payload
        and sha(values.tobytes()) == receipt.receipt["accepted_weight_sha256"]
        and sha(diagnostic.payload) == receipt.receipt["diagnostics_sha256"]
        and document["build"].get("budget_sha256") == budget.digest
        and document["build"].get("budget_artifact_key") == materialized_budget.key
        and document["build"].get("accepted_weight_sha256") == sha(values.tobytes()),
        "FINAL_DIAGNOSTICS",
    )
    require(
        canonical_json({n: receipt.receipt.get(n) for n in expected})
        == canonical_json(expected)
        and canonical_json({n: document["build"].get(n) for n in expected})
        == canonical_json(expected)
        and canonical_json(document["options"]["grouped_upper_bounds"])
        == canonical_json(budget.grouped_bounds.diagnostics(values, corrected)),
        "DIAGNOSTIC_RECONSTRUCTION",
    )


def _persist(path, payload, *, warm):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == payload, "FROZEN_BYTES")
    else:
        require(not warm, "WARM_FROZEN_ABSENT")
        with path.open("xb") as output:
            output.write(payload)


def _subset(graph, ids):
    nodes = tuple(n for n in graph.nodes if n.id in ids)
    used = {s for n in nodes for s in n.sources}
    return replace(
        graph, nodes=nodes, sources=tuple(s for s in graph.sources if s.name in used)
    )


def run_enriched_calibration(
    survey_graph,
    *,
    sources,
    store,
    kernels,
    native_binding,
    clone_binding,
    clone_population,
    columns,
    price_pins,
    host_pins,
    route,
    target_registry,
    evidence_directory,
    resume="auto",
):
    """Run the 17→20→23→24→26 admission sequence, never skipping warm checks."""
    require(resume in ("auto", "require"), "RESUME")
    warm = resume == "require"
    require(len(survey_graph.nodes) == 17, "SURVEY_NODE_COUNT")
    require(
        tuple(s.name for s in target_registry)
        == tuple(b.variable for b in ages.NATIONAL_AGE_ACTIVATION.bands)
        and tuple(s.measure for s in target_registry)
        == tuple(b.column for b in ages.NATIONAL_AGE_ACTIVATION.bands),
        "TARGET_RECIPE",
    )
    evidence_directory = Path(evidence_directory)
    evidence_directory.mkdir(exist_ok=True)
    live, phases = {}, []
    replay_boundaries = {
        native_binding,
        clone_binding,
        BOUNDARY,
        MATRIX,
        ATTACH,
        COUNT,
        caps.CALIBRATION_NODE,
    }

    def run(graph, additional_sources=None):
        used = {s.name for s in graph.sources}
        actual = {**sources, **(additional_sources or {})}
        previous = {
            name: pop for name, pop in live.items() if name in replay_boundaries
        }
        compiled = compile_graph(graph)
        manifest = run_graph(
            compiled,
            sources={n: actual[n] for n in used},
            store=store,
            kernels=kernels,
            resume=resume,
            _population_observer=lambda name, pop: live.__setitem__(name, pop),
        )
        for name in previous.keys() & live.keys():
            verify_cache_replay(previous[name], live[name])
        phases.append(
            {
                "nodes": len(graph.nodes),
                "order": list(compiled.order),
                "hits": sum(r.hit for r in manifest.nodes.values()),
                "keys": {n: r.key for n, r in manifest.nodes.items()},
            }
        )
        return manifest

    prefix = run(survey_graph)
    native, clone = live[native_binding], live[clone_binding]
    arguments = dict(
        native_population=native,
        expanded_population=clone,
        native_binding_node=native_binding,
        expanded_binding_node=clone_binding,
        design_multiplier=RECIPE["design_multiplier"],
        incoming_multiplier=RECIPE["incoming_multiplier"],
    )
    budget = budgets.freeze_origin_group_budget(prefix, store, **arguments)
    budget.grouped_bounds.check(
        clone.frame.weights_for("household").values, positive=True
    )
    host = host_values(
        prefix,
        store,
        native_binding=native_binding,
        clone_binding=clone_binding,
        expected=host_pins,
    )
    consumer.qualify_host_population(clone.frame, host)
    payloads = {
        role: Path(sources[name]).read_bytes()
        for role, name in puf.SOURCE_NAMES.items()
    }
    expected_donor, expected_transport = consumer.qualify_price_transport(
        payloads, price_pins, route=route, recipe=consumer.recipe_document()
    )
    extra = puf.diagnostic_nodes(
        columns,
        base=clone_population,
        native_binding=native_binding,
        clone_binding=clone_binding,
        price_pins=price_pins,
        host_pins=host_pins,
        route=route,
    )
    full = replace(
        survey_graph,
        sources=(*survey_graph.sources, *puf.SOURCES),
        nodes=(*survey_graph.nodes, *extra),
    )
    prefit_ids = {n.id for n in survey_graph.nodes} | {
        puf.PREFIX + ".donor",
        BOUNDARY,
        MATRIX,
    }
    prefit = run(_subset(full, prefit_ids))
    verify_donor(
        prefit, store, live[puf.PREFIX + ".donor"], expected_donor, expected_transport
    )
    # Prefix replay uses the store's specified null canonicalization. Strict
    # transition checks below use this phase's actual clone, not cold buffers.
    clone = live[clone_binding]
    boundary, matrix_pop = live[BOUNDARY], live[MATRIX]
    same_carried(clone, boundary)
    check_ledger_append(
        clone, boundary, full.node(BOUNDARY), partition=full.mass_partition
    )
    verify_matrix(
        prefit,
        store,
        boundary,
        matrix_pop,
        values=host,
        host_pins=host_pins,
        route=route,
    )
    # No fit/apply/attach node exists in the preceding compiled graph.
    run(full)
    attached = live[ATTACH]
    counted_graph = replace(full, nodes=(*full.nodes, enriched_age_count_node()))
    counted_manifest = run(counted_graph)
    # Capture these immutable historical boundaries before later version updates.
    boundary, matrix_pop, attached, counted = (
        live[n] for n in (BOUNDARY, MATRIX, ATTACH, COUNT)
    )
    parent = enriched_binding(
        budget,
        live[clone_binding],
        boundary,
        matrix_pop,
        attached,
        counted,
        manifest=counted_manifest,
        store=store,
        graph=counted_graph,
        host=host,
        host_pins=host_pins,
        route=route,
    )
    budget_path = evidence_directory / "original-budget.json"
    _persist(budget_path, budget.payload, warm=warm)
    _persist(evidence_directory / "enriched-parent.json", parent.payload, warm=warm)
    edge_args = dict(
        base=BOUNDARY,
        native_binding_node=native_binding,
        expanded_binding_node=clone_binding,
    )
    final_graph = replace(
        counted_graph,
        sources=(*counted_graph.sources, SourceRef(caps.BUDGET_SOURCE, "raw-bytes-v1")),
        nodes=(
            *counted_graph.nodes,
            caps.origin_budget_node(**edge_args),
            caps.grouped_demographic_calibration_node(
                target_registry,
                **edge_args,
                epochs=RECIPE["epochs"],
                learning_rate=RECIPE["learning_rate"],
                design_multiplier=RECIPE["design_multiplier"],
                incoming_multiplier=RECIPE["incoming_multiplier"],
            ),
        ),
    )
    require(len(final_graph.nodes) == 26, "FINAL_NODE_COUNT")
    final = run(final_graph, {caps.BUDGET_SOURCE: budget_path})
    verify_donor(
        final, store, live[puf.PREFIX + ".donor"], expected_donor, expected_transport
    )
    budgets.verify_origin_group_budget(
        budget,
        final,
        store,
        **dict(
            arguments,
            native_population=live[native_binding],
            expanded_population=live[clone_binding],
        ),
    )
    restored = enriched_binding(
        budget,
        live[clone_binding],
        live[BOUNDARY],
        live[MATRIX],
        live[ATTACH],
        live[COUNT],
        manifest=final,
        store=store,
        graph=final_graph,
        host=host,
        host_pins=host_pins,
        route=route,
    )
    require(restored.payload == parent.payload, "FINAL_PARENT_REPLAY")
    current = live[caps.CALIBRATION_NODE]
    successor = weight_successor(
        parent,
        budget,
        live[COUNT],
        current,
        node=final_graph.node(caps.CALIBRATION_NODE),
        partition=final_graph.mass_partition,
    )
    receipt = final.node(caps.CALIBRATION_NODE)
    verify_final_diagnostics(final, store, budget, current)
    _persist(evidence_directory / "weight-successor.json", successor.payload, warm=warm)
    key = sha256_domain(
        "puf-age-enriched-weight-successor",
        canonical_json(
            {
                "calibration_node": receipt.key,
                "parent": parent.digest,
                "payload": successor.digest,
                "type": SUCCESSOR_SCHEMA,
            }
        ),
    )
    store.put_bytes(key, successor.payload)
    require(store.load_bytes(key) == successor.payload, "STORED_SUCCESSOR")
    return {
        "manifest": final,
        "budget": budget,
        "parent": parent,
        "successor": successor,
        "successor_key": key,
        "phases": phases,
        "populations": live,
        "graph": final_graph,
    }
