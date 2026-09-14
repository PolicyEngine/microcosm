"""Country admission for survey age profiles with an explicit source prefix.

Every call reconstructs the accepted source preparation. The numerical kernels
are values-only; source, full-population, group/row-cap and cache checks live
here. The original entry point remains invented-only; the named development
entry point derives its targets from an explicitly pinned S0101-only capture.
Both permit only the first IMPORTANCE-to-CALIBRATED transition and no release.
An optional atomic-geography recipe enriches the accepted allocation before
cloning; the sampling budget continues to retain the original raw allocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from microcosm.frame import Frame
from microcosm.graph import (
    CompiledGraph,
    KernelRegistry,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.artifact_edges import typed_contracts
from microcosm.graph.canonical import canonical_json
from microcosm.graph.executor import (
    _all_node_keys,
    _input_writers,
    _project_context,
    _source_paths_and_keys,
    _tolerance_writer_payload,
)
from microcosm.graph.keys import (
    _capabilities_projection,
    artifact_key,
    frame_key,
    opaque_artifact_key,
    seed,
    weights_key,
)
from microcosm.graph.manifest import _freeze_json
from microcosm.graph.population import (
    Population,
    mass_record_receipt,
    weight_cap_receipt,
)
from microcosm.graph.store import _verified_meta

from . import graph_combined_clone as clone
from . import graph_survey_age_artifact as ages
from . import graph_survey_budget as transport
from . import graph_survey_calibration as numerical
from . import graph_survey_population as source_graph
from . import survey_age_activation as age_activation
from . import survey_calibration_diagnostics as diagnostic_check
from . import survey_origin_budget as budgets
from . import survey_population_replay as replay

COUNT_NODE = "survey.age_count_matrix"


def _require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_AGE_RUN_" + reason)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _node_identity(node):
    """Immutable snapshot excluding execution time and cache-hit bookkeeping."""
    return canonical_json(
        {
            "key": node.key,
            "ref": node.kernel_ref,
            "implementation": node.kernel_impl_hash,
            "capabilities": _capabilities_projection(node.capabilities),
            "typed_artifacts": node.typed_artifacts,
            "seed": node.seed,
            "frame_key": node.frame_key,
            "weight_key": node.weight_key,
            "artifacts": [
                [entity, column, key]
                for (entity, column), key in sorted(node.artifacts.items())
            ],
            "opaque_artifacts": node.opaque_artifacts,
            "receipt": node.receipt,
            "legacy_capabilities": node.legacy_capabilities,
        }
    )


def _measurement(population):
    node = ages.survey_age_count_artifact_node(
        population=population.version, node_id=COUNT_NODE
    )
    context = _project_context(
        node,
        population,
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
    )
    measured = ages.SurveyAgeCountArtifactKernel().run(context)
    payload = measured.artifacts["counts"]
    values = ages.decode_survey_age_counts(payload)
    _require(
        tuple(values.household_ids)
        == tuple(population.frame.table("household").household_id),
        "MEASUREMENT_HOUSEHOLD_ORDER",
    )
    return payload, json.loads(canonical_json(measured.receipt))


def _measure(population):
    return _measurement(population)[0]


def _load_diagnostics(store, key):
    """Bound the diagnostic payload buffer before reading it.

    The existing store verifier still authenticates metadata and streams its
    payload hashes. This does not claim a bound on that verifier's metadata
    decoding or total I/O. Only the selected diagnostic payload is buffered here.
    """
    path = store.object_path(key)
    payload_path = path / "payload.bin"
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(payload_path, flags)
    try:
        before = os.fstat(fd)
        limit = diagnostic_check.MAX_DIAGNOSTIC_BYTES
        _require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit,
            "DIAGNOSTIC_STORE_LIMIT",
        )
        metadata = _verified_meta(path, expected_kind="bytes")
        _require(
            set(metadata["payloads"]) == {"payload.bin"}, "DIAGNOSTIC_STORE_ROSTER"
        )
        record = metadata["payloads"]["payload.bin"]
        _require(record["size"] == before.st_size, "DIAGNOSTIC_STORE_SIZE")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(limit + 1)
        after = os.fstat(fd)
        current = os.stat(payload_path, follow_symlinks=False)

        def seal(s):
            return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

        _require(
            seal(before) == seal(after) == seal(current)
            and stat.S_ISREG(current.st_mode)
            and len(payload) == record["size"] <= limit
            and _sha(payload) == record["sha256"],
            "DIAGNOSTIC_STORE_CHANGED",
        )
        return payload
    finally:
        os.close(fd)


def _expected_nodes(
    compiled: CompiledGraph,
    kernels: KernelRegistry,
    keys: dict[str, str],
    implementations: dict[str, str],
    prefix_identities: dict[str, bytes],
    frame: Frame,
):
    """Detach declaration expectations before any graph kernel/store execution."""
    states = {}
    cells = tuple((e, str(c)) for e in frame.entities for c in frame.table(e))
    for node in compiled.graph.nodes:
        if node.id in prefix_identities:
            state = json.loads(prefix_identities[node.id])
            state["artifacts"] = {(e, c): k for e, c, k in state["artifacts"]}
            # The accepted prefix runner already independently sealed this
            # receipt. No receipt from the receiving run is trusted here.
            states[node.id] = state
            continue
        key = keys[node.id]
        structural = node.structural is not StructuralDelta.NONE
        outputs = (
            cells if structural else tuple((o.entity, o.column) for o in node.outputs)
        )
        states[node.id] = {
            "key": key,
            "ref": node.kernel,
            "implementation": implementations[node.id],
            "capabilities": json.loads(
                canonical_json(
                    _capabilities_projection(kernels.get(node.kernel).capabilities)
                )
            ),
            "typed_artifacts": json.loads(
                canonical_json(typed_contracts(compiled, node, keys, kernels))
            ),
            "seed": seed(key),
            "frame_key": frame_key(key) if structural else None,
            "weight_key": weights_key(key, node.weights.entity)
            if node.weights
            else None,
            "artifacts": {(e, c): artifact_key(key, e, c) for e, c in outputs},
            "opaque_artifacts": {
                o.name: opaque_artifact_key(key, o.name) for o in node.artifact_outputs
            },
        }
    return states


def _complete_receipts(
    compiled: CompiledGraph,
    states: dict[str, dict[str, object]],
    authored: dict[str, dict[str, object]],
    observed: dict[str, Population],
):
    """Normalize independently authored receipts through the executor's rules."""
    previous = {}
    for node_id in compiled.order:
        state = states[node_id]
        if node_id in authored:
            node = compiled.graph.node(node_id)
            receipt = json.loads(canonical_json(authored[node_id]))
            receipt["capabilities"] = dict(state["capabilities"])
            writers = _tolerance_writer_payload(
                _input_writers(compiled, node_id, receipts=previous)
            )
            if writers:
                receipt["capabilities"]["tolerance_writers"] = writers
            if node.structural not in {StructuralDelta.NONE, StructuralDelta.CREATE}:
                receipt["mass"] = mass_record_receipt(observed[node_id].mass_ledger[-1])
            receipt.update(weight_cap_receipt(observed[node_id], node))
            state["receipt"] = json.loads(canonical_json(receipt))
        previous[node_id] = SimpleNamespace(receipt=state["receipt"])


def _check_manifest(manifest, compiled, states):
    _require(
        manifest.country == compiled.graph.country and manifest.decisions == (),
        "MANIFEST_COUNTRY_DECISIONS",
    )
    _require(
        set(manifest.populations) == set(compiled.versions.values())
        and set(manifest.mass_ledgers) == set(compiled.versions.values()),
        "MANIFEST_POPULATION_VERSIONS",
    )
    # Detached expectations use JSON arrays. NodeReceipt freezes those arrays
    # as tuples, including nested capabilities and typed artifact descriptors.
    # Match its representation without weakening the canonical value check.
    frozen_states = {
        node_id: {
            **state,
            "receipt": _freeze_json(state["receipt"]),
            "typed_artifacts": _freeze_json(state["typed_artifacts"]),
        }
        for node_id, state in states.items()
    }
    source_graph._check_node_states(manifest, frozen_states)
    for node_id, state in states.items():
        expected = {**state, "legacy_capabilities": False}
        expected["artifacts"] = [
            [e, c, key] for (e, c), key in sorted(state["artifacts"].items())
        ]
        _require(
            _node_identity(manifest.node(node_id)) == canonical_json(expected),
            "MANIFEST_CANONICAL_VALUES",
        )


@dataclass(frozen=True)
class SurveyAgeCalibrationRun:
    """Values validated at return; retained source handles still require rechecks."""

    manifest: object
    compiled: object
    budget: object
    successor: object
    diagnostics: dict
    counts_sha256: str
    numeric_bounds_sha256: str
    release_eligible: bool = False


def run_survey_age_calibration(
    source_dir,
    *,
    snapshot_root,
    store_root,
    fraction,
    seed_value,
    target_registry,
    epochs,
    learning_rate,
    resume="auto",
    geography_config=None,
):
    """Reconstruct sources and admit a complete invented calibration graph.

    The source and store paths are the caller's explicitly approved local
    inputs. Cache reuse never bypasses fresh source preparation or the complete
    receiving-population and sampling-reference checks. This v1 does not admit
    genuine targets, donor-detail descendants, pruning or repeated calibration.
    """
    # Refuse unsupported targets/options before any source I/O.
    calibration = numerical.survey_age_calibration_node(
        target_registry,
        base=clone.COMBINED_CLONE_NODE,
        budget_node=transport.BUDGET_NODE,
        count_node=COUNT_NODE,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    return _run_survey_age_calibration(
        source_dir,
        snapshot_root=snapshot_root,
        store_root=store_root,
        fraction=fraction,
        seed_value=seed_value,
        epochs=epochs,
        learning_rate=learning_rate,
        resume=resume,
        calibration=calibration,
        geography_config=geography_config,
    )


def run_survey_age_development(
    source_dir,
    *,
    age_source_dir,
    activation,
    snapshot_root,
    store_root,
    fraction,
    seed_value,
    epochs,
    learning_rate,
    resume="auto",
    geography_config=None,
):
    """Run development calibration after deriving targets from exact source pins.

    The caller separately reviews the declaration and explicitly admits these
    local source paths. There is no default capture, caller-supplied registry,
    mixed-inventory fallback, production attestation or release permission.
    """
    _require(type(resume) is str and resume in {"auto", "require"}, "RESUME")
    _require(type(epochs) is int and 1 <= epochs <= 1000, "EPOCHS")
    _require(
        type(learning_rate) in (int, float)
        and np.isfinite(learning_rate)
        and 0 < learning_rate <= 1,
        "LEARNING_RATE",
    )
    binding = age_activation.activation_binding(activation)
    registry = age_activation.activate_survey_age_targets(
        age_source_dir, declaration=activation
    )
    calibration = numerical.survey_age_development_node(
        registry,
        activation_binding=binding,
        base=clone.COMBINED_CLONE_NODE,
        budget_node=transport.BUDGET_NODE,
        count_node=COUNT_NODE,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    return _run_survey_age_calibration(
        source_dir,
        snapshot_root=snapshot_root,
        store_root=store_root,
        fraction=fraction,
        seed_value=seed_value,
        epochs=epochs,
        learning_rate=learning_rate,
        resume=resume,
        calibration=calibration,
        age_source_dir=age_source_dir,
        activation=activation,
        geography_config=geography_config,
    )


def _run_survey_age_calibration(
    source_dir,
    *,
    snapshot_root,
    store_root,
    fraction,
    seed_value,
    epochs,
    learning_rate,
    resume,
    calibration,
    age_source_dir=None,
    activation=None,
    geography_config=None,
):
    """Shared population, budget, replay, diagnostic and final-return checks."""
    _require(type(resume) is str and resume in {"auto", "require"}, "RESUME")
    activation_identity = None
    calibration_binding = numerical.calibration_activation_binding(calibration.params)
    _require(
        (calibration_binding is None) == (activation is None), "ACTIVATION_REQUIRED"
    )
    if activation is not None:
        activation_identity = canonical_json(
            age_activation.activation_binding(activation)
        )
        _require(
            canonical_json(calibration_binding) == activation_identity,
            "ACTIVATION_BINDING",
        )
    geography = None
    geography_identity = None
    geography_config_payload = None
    prefix_populations = {}
    if geography_config is None:
        prefix = source_graph.run_authenticated_survey_population(
            source_dir,
            snapshot_root=snapshot_root,
            store_root=store_root,
            fraction=fraction,
            seed=seed_value,
            resume=resume,
            clones=True,
            return_values=True,
        )
    else:
        from . import graph_atomic_survey_population as atomic_source

        geography_config_payload = (
            atomic_source.reconstruction.AtomicSurveyReconstruction.to_bytes(
                geography_config
            )
        )
        prefix = atomic_source.run_atomic_survey_population(
            source_dir,
            snapshot_root=snapshot_root,
            store_root=store_root,
            fraction=fraction,
            seed=seed_value,
            resume=resume,
            geography_config=geography_config,
            return_values=True,
        )
        geography = atomic_source.reconstruction.reconstruct_atomic_survey_geography(
            prefix.preparation, prefix.allocated_population, geography_config
        )
        replay.same_replayed_population(
            geography.population, prefix.geography_population
        )
        geography_identity = budgets._population_identity(prefix.geography_population)
        prefix_populations = {
            stage.node.id: stage.population for stage in geography.stages
        }
    source_owner, source_view = source_graph._checked_preparation(prefix.preparation)
    preparation_entry = source_owner._ISSUED.get(id(prefix.preparation))
    instructions = source_graph.allocation_instructions(
        source_view.selection_plan, source_view.receipt["origins"]["households"]
    )
    _, allocated_context, allocation_payload, _, _ = source_graph._allocation_output(
        source_view.frame, source_view.context, instructions, _sha(source_view.payload)
    )
    prefix_artifacts = (
        (
            source_graph.CREATE_NODE,
            "preparation",
            source_graph.PREPARATION_TYPE,
            source_view.payload,
            source_graph.SurveyPopulationCreateKernel.capabilities,
        ),
        (
            source_graph.CREATE_NODE,
            "frame_context",
            source_graph.US_FRAME_CONTEXT_TYPE,
            source_view.context,
            source_graph.SurveyPopulationCreateKernel.capabilities,
        ),
        (
            source_graph.ALLOCATION_NODE,
            "allocation",
            source_graph.ALLOCATION_TYPE,
            allocation_payload,
            source_graph.SurveyPopulationAllocationKernel.capabilities,
        ),
        (
            source_graph.ALLOCATION_NODE,
            "frame_context",
            source_graph.US_FRAME_CONTEXT_TYPE,
            allocated_context,
            source_graph.SurveyPopulationAllocationKernel.capabilities,
        ),
    )
    if geography is not None:
        prefix_artifacts += tuple(
            (
                node.id,
                "support",
                atomic_source.ATOMIC_SUPPORT_TYPE,
                geography.support_payload,
                atomic_source.AtomicSupportImportKernel.capabilities,
            )
            for node in geography.nodes
            if node.kernel == atomic_source.AtomicSupportImportKernel.ref
        )
        gate = geography.stages[-1]
        prefix_artifacts += (
            (
                gate.node.id,
                "validation",
                atomic_source.reconstruction.atomic_graph.ATOMIC_GEOGRAPHY_VALIDATION_TYPE,
                gate.receipt,
                atomic_source.reconstruction.atomic_graph.AtomicGeographyGateKernel.capabilities,
            ),
        )
    initial = prefix.clone_population
    _require(initial is not None, "COMPLETE_CLONE_REQUIRED")
    budget = budgets.freeze_survey_origin_budget(
        prefix.preparation,
        allocated_population=prefix.allocated_population,
        clone_population=initial,
        geography_config=geography_config,
    )
    budget_view = budget.checked_view()
    budget_payload = budget_view.payload
    numeric_payload = transport.numeric_survey_budget_payload(budget_payload)
    numeric_bounds = numerical.decode_numeric_survey_bounds(numeric_payload)
    count_payload, count_receipt = _measurement(initial)
    _require(
        numeric_bounds.grouped.household_ids
        == tuple(initial.frame.table("household").household_id),
        "BUDGET_HOUSEHOLD_ORDER",
    )
    prefix_identities = {
        name: _node_identity(row) for name, row in prefix.manifest.nodes.items()
    }
    initial_identity = budgets._population_identity(initial)
    allocation_identity = budgets._population_identity(prefix.allocated_population)
    prefix_population_identities = {
        name: budgets._population_identity(population)
        for name, population in prefix_populations.items()
    }
    prefix_budget = budget
    prefix_budget_entry = budgets._entry(prefix_budget, budgets.SamplingOriginBudget)
    kernels = prefix.kernels
    kernels.register(transport.SurveySamplingBudgetKernel(budget_payload))
    kernels.register(ages.SurveyAgeCountArtifactKernel())
    kernels.register(numerical.SurveyAgeCalibrationKernel())
    age_nodes = (
        transport.survey_sampling_budget_node(budget_sha256=_sha(budget_payload)),
        ages.survey_age_count_artifact_node(
            population=initial.version, node_id=COUNT_NODE
        ),
        calibration,
    )
    graph = replace(
        prefix.compiled.graph,
        nodes=(*prefix.compiled.graph.nodes, *age_nodes),
    )
    compiled = compile_graph(graph)
    prefix_roster = set(prefix.compiled.order)
    age_roster = {transport.BUDGET_NODE, COUNT_NODE, numerical.CALIBRATION_NODE}
    _require(
        set(prefix_identities) == prefix_roster
        and not prefix_roster & age_roster
        and tuple(node.id for node in age_nodes)
        == (transport.BUDGET_NODE, COUNT_NODE, numerical.CALIBRATION_NODE)
        and set(compiled.order) == prefix_roster | age_roster
        and len(compiled.order) == len(prefix.compiled.order) + 3
        and tuple(name for name in compiled.order if name in prefix_roster)
        == prefix.compiled.order,
        "EXACT_GRAPH",
    )
    _require(
        clone.COMBINED_CLONE_CLAIM_NODE in compiled.predecessors[transport.BUDGET_NODE],
        "OWNERSHIP_CLAIM_EDGE",
    )
    _paths, source_keys = _source_paths_and_keys(compiled, prefix.sources, prefix.store)
    keys, implementations = _all_node_keys(compiled, kernels, source_keys)
    expected_nodes = _expected_nodes(
        compiled, kernels, keys, implementations, prefix_identities, initial.frame
    )
    observed, snapshots = {}, {}
    admitted = None
    successor_entry = None
    successor_payload = None
    receiving_initial = None
    receiving_budget_entry = None
    receiving_node = (
        clone.COMBINED_CLONE_CLAIM_NODE
        if geography is None
        else geography.stages[-1].node.id
    )

    def observe(node_id, population):
        nonlocal admitted, budget, receiving_initial, receiving_budget_entry
        nonlocal successor_entry, successor_payload
        _require(node_id not in observed, "DUPLICATE_OBSERVATION")
        if node_id == numerical.CALIBRATION_NODE:
            _require(receiving_initial is not None, "RECEIVING_BUDGET_REQUIRED")
            # Independent of the numerical kernel, on both execution and load.
            numerical.check_numeric_survey_weights(
                numeric_bounds, population.frame.weights_for("household").values
            )
            admitted = budgets.admit_survey_weight_only_population(
                budget,
                previous=receiving_initial,
                current=population,
            )
            successor_entry = budgets._entry(admitted, budgets.SamplingOriginSuccessor)
            successor_payload = admitted.payload
        elif node_id in {transport.BUDGET_NODE, COUNT_NODE}:
            # Read-only count/transport may precede the geography gate. Compare
            # their complete current version, including whichever prefix writes
            # have actually occurred. Calibration waits for every base member.
            if geography is None:
                replay.same_replayed_frame(initial.frame, population.frame)
            else:
                preceding = next(
                    name
                    for name in reversed(observed)
                    if name in prefix_populations
                    and compiled.versions[name] == initial.version
                )
                replay.same_replayed_population(
                    prefix_populations[preceding], population
                )
            _require(
                population.frame.weights_for("household").values.tobytes()
                == numeric_bounds.incoming.tobytes(),
                "RECEIVING_INCOMING_WEIGHTS",
            )
            if node_id == transport.BUDGET_NODE and geography is None:
                _require(receiving_initial is not None, "RECEIVING_BUDGET_REQUIRED")
                _require(
                    budgets._population_identity(population)
                    == snapshots[clone.COMBINED_CLONE_CLAIM_NODE],
                    "COMPLETE_CLAIMED_CLONE",
                )
            elif node_id == COUNT_NODE:
                _require(_measure(population) == count_payload, "REMEASURED_COUNTS")
        elif node_id == source_graph.ALLOCATION_NODE:
            replay.same_replayed_population(prefix.allocated_population, population)
        elif node_id == receiving_node:
            replay.same_replayed_population(initial, population)
            _require(source_graph.ALLOCATION_NODE in observed, "ALLOCATION_REQUIRED")
            # Actual source reconstruction rebinds an issued budget to this run's
            # receiving objects. Candidate bytes grant no authority by themselves.
            budget = budgets.freeze_survey_origin_budget(
                prefix.preparation,
                allocated_population=observed[source_graph.ALLOCATION_NODE],
                clone_population=population,
                candidate=budget_payload,
                geography_config=geography_config,
            )
            receiving_budget_entry = budgets._entry(
                budget, budgets.SamplingOriginBudget
            )
            receiving_initial = population
        elif node_id == source_graph.CREATE_NODE:
            replay.same_replayed_frame(
                prefix.manifest.population(node_id), population.frame
            )
        elif node_id == clone.COMBINED_CLONE_NODE:
            actual_allocation = observed[
                source_graph.ALLOCATION_NODE
                if geography is None
                else atomic_source.projection.NODE
            ]
            source_graph._verify_cloned_frame(
                actual_allocation.frame,
                population.frame,
                actual_allocation.design_weights["household"],
            )
            if geography is None:
                replay.same_replayed_frame(initial.frame, population.frame)
            else:
                replay.same_replayed_population(prefix_populations[node_id], population)
        elif node_id in prefix_populations:
            replay.same_replayed_population(prefix_populations[node_id], population)
        else:
            _require(False, "UNEXPECTED_NODE")
        observed[node_id] = population
        snapshots[node_id] = budgets._population_identity(population)

    manifest = run_graph(
        compiled,
        sources=prefix.sources,
        store=prefix.store,
        kernels=kernels,
        resume=resume,
        _population_observer=observe,
    )
    _require(
        tuple(observed) == compiled.order and admitted is not None,
        "OBSERVATION_COVERAGE",
    )
    _require(set(manifest.nodes) == set(compiled.order), "MANIFEST_NODES")
    for node in graph.nodes:
        actual = manifest.node(node.id)
        _require(
            actual.key == keys[node.id]
            and actual.kernel_ref == node.kernel
            and actual.kernel_impl_hash == implementations[node.id]
            and actual.typed_artifacts == expected_nodes[node.id]["typed_artifacts"]
            and actual.seed == seed(keys[node.id])
            and actual.legacy_capabilities is False,
            "MANIFEST_NODE_IDENTITY",
        )
        if node.id in prefix_identities:
            _require(
                _node_identity(actual) == prefix_identities[node.id],
                "PREFIX_RECEIPT_REPLAY",
            )
    for node_id, name, type_, payload, capabilities in (
        *prefix_artifacts,
        (
            transport.BUDGET_NODE,
            "budget",
            budgets.BUDGET_TYPE,
            budget_payload,
            transport.SurveySamplingBudgetKernel.capabilities,
        ),
        (
            transport.BUDGET_NODE,
            "numeric_bounds",
            numerical.BOUNDS_TYPE,
            numeric_payload,
            transport.SurveySamplingBudgetKernel.capabilities,
        ),
        (
            COUNT_NODE,
            "counts",
            ages.COUNTS_TYPE,
            count_payload,
            ages.SurveyAgeCountArtifactKernel.capabilities,
        ),
    ):
        source_graph._final_artifact(
            manifest,
            prefix.store,
            node_id=node_id,
            name=name,
            type_=type_,
            payload=payload,
            capabilities=capabilities,
        )
    receipt = manifest.node(numerical.CALIBRATION_NODE)
    diagnostic_payload = _load_diagnostics(
        prefix.store,
        expected_nodes[numerical.CALIBRATION_NODE]["opaque_artifacts"]["diagnostics"],
    )
    current = observed[numerical.CALIBRATION_NODE]
    anchors = {
        "budget_sha256": _sha(budget_payload),
        "numeric_bounds_sha256": _sha(numeric_payload),
        "counts_sha256": _sha(count_payload),
        "accepted_weight_sha256": _sha(
            current.frame.weights_for("household").values.tobytes()
        ),
        "constraint_digest": numeric_bounds.grouped.digest,
        "weight_anchor": calibration.params["weight_anchor"],
        "cap_enforcement": calibration.params["cap_enforcement"],
        "fixed_zero_rows": int(np.count_nonzero(numeric_bounds.incoming == 0)),
    }
    _require(
        _sha(diagnostic_payload) == receipt.receipt.get("diagnostics_sha256"),
        "DIAGNOSTICS_DIGEST",
    )
    _require(
        all(receipt.receipt.get(k) == v for k, v in anchors.items()), "RECEIPT_ANCHORS"
    )
    registry = numerical.demographic._registry_from_json(calibration.params["registry"])
    diagnostics = diagnostic_check.validate_survey_calibration_diagnostics(
        diagnostic_payload,
        counts_payload=count_payload,
        bounds_payload=numeric_payload,
        weights=current.frame.weights_for("household").values,
        registry=registry,
        epochs=epochs,
        learning_rate=learning_rate,
        anchors=anchors,
        activation_binding=calibration_binding,
    )
    diagnostic_identity = canonical_json(diagnostics)
    _complete_receipts(
        compiled,
        expected_nodes,
        {
            transport.BUDGET_NODE: {
                "budget_sha256": _sha(budget_payload),
                "numeric_bounds_sha256": _sha(numeric_payload),
                "constraint_digest": numeric_bounds.grouped.digest,
                "group_count": numeric_bounds.grouped.group_count,
                "release_eligible": False,
                "source_admission": "required_from_country_runner",
            },
            COUNT_NODE: count_receipt,
            numerical.CALIBRATION_NODE: {
                **anchors,
                "diagnostics_sha256": _sha(diagnostic_payload),
                "scope": numerical.calibration_receipt_scope(calibration.params),
                "release_eligible": False,
                "source_admission": "required_from_country_runner",
            },
        },
        observed,
    )
    _check_manifest(manifest, compiled, expected_nodes)
    result = SurveyAgeCalibrationRun(
        manifest,
        compiled,
        budget,
        admitted,
        diagnostics,
        _sha(count_payload),
        _sha(numeric_payload),
    )
    final_numbers = numerical.decode_numeric_survey_bounds(numeric_payload)
    # A population version can be shared by several nonstructural nodes. Check
    # the actual last receiving state for every version, including the prefix.
    terminal_nodes = tuple({compiled.versions[n]: n for n in compiled.order}.values())
    terminal_frames = {
        node_id: manifest.population(compiled.versions[node_id])
        for node_id in terminal_nodes
    }
    terminal_ledgers = {
        node_id: manifest.mass_ledger(compiled.versions[node_id])
        for node_id in terminal_nodes
    }
    # Finish storage and key/implementation I/O before final source/target
    # revalidation and complete-population checks. No store read follows.
    final_keys, final_implementations = _all_node_keys(compiled, kernels, source_keys)
    _require(
        final_keys == keys and final_implementations == implementations,
        "FINAL_IMPLEMENTATIONS",
    )
    _require(_measure(initial) == count_payload, "FINAL_REMEASUREMENT")
    _require(
        _measure(receiving_initial) == count_payload, "FINAL_RECEIVING_MEASUREMENT"
    )
    # Retain both authorities; the original source/planned budget is not waived
    # when a candidate-identical handle is issued for the actual receiving run.
    budgets.verify_survey_origin_budget(prefix_budget)
    budgets.verify_survey_weight_only_successor(admitted)
    if activation is not None:
        # Recheck target evidence before the optional final support read;
        # complete population, configuration and owner seals follow both.
        age_activation.verify_survey_age_targets(
            age_source_dir, declaration=activation, registry=registry
        )
    if geography is not None:
        _require(
            atomic_source.reconstruction.AtomicSurveyReconstruction.to_bytes(
                geography_config
            )
            == geography_config_payload,
            "FINAL_GEOGRAPHY_CONFIG",
        )
        final_support, _support_identity = atomic_source.reconstruction._read_support(
            geography_config
        )
        _require(final_support == geography.support_payload, "FINAL_GEOGRAPHY_SUPPORT")
    _require(
        source_owner._ISSUED.get(id(prefix.preparation)) is preparation_entry
        and preparation_entry is not None
        and prefix.preparation.payload == preparation_entry[1],
        "FINAL_SOURCE_ISSUANCE",
    )
    source_owner._pure_final(preparation_entry[2])
    if geography is not None:
        _require(
            atomic_source.reconstruction.AtomicSurveyReconstruction.to_bytes(
                geography_config
            )
            == geography_config_payload
            and budgets._population_identity(prefix.geography_population)
            == geography_identity
            and all(
                budgets._population_identity(population)
                == prefix_population_identities[name]
                for name, population in prefix_populations.items()
            ),
            "FINAL_GEOGRAPHY_PREFIX",
        )
    _require(budget.payload == budget_payload, "FINAL_BUDGET_BYTES")
    _require(
        budgets._population_identity(initial) == initial_identity
        and budgets._population_identity(prefix.allocated_population)
        == allocation_identity,
        "FINAL_PREFIX_POPULATIONS",
    )
    for node_id, population in observed.items():
        _require(
            budgets._population_identity(population) == snapshots[node_id],
            "FINAL_POPULATION_STATE",
        )
        # A nonstructural node can precede the claim within the same version;
        # its own snapshot is checked above. Terminal version state is checked
        # on the declared weight transition and complete ownership claim.
        if node_id in terminal_nodes:
            replay.same_replayed_frame(population.frame, terminal_frames[node_id])
            _require(
                terminal_ledgers[node_id] == population.mass_ledger,
                "FINAL_MASS_LEDGER",
            )
    numerical.check_numeric_survey_weights(
        final_numbers,
        current.frame.weights_for("household").values,
    )
    if activation is not None:
        _require(
            canonical_json(age_activation.activation_binding(activation))
            == activation_identity
            and calibration.params.get("activation") == activation_identity.decode(),
            "FINAL_ACTIVATION_BINDING",
        )
    # No decoding, store access, source borrow or implementation helper follows
    # this seal. It covers the complete selected prefix and age trio, including
    # scope and release flags, using expectations detached before execution.
    _check_manifest(manifest, compiled, expected_nodes)
    _require(
        canonical_json(result.diagnostics) == diagnostic_identity,
        "FINAL_DIAGNOSTICS_VALUES",
    )
    # The verifications above perform I/O. Finish with the retained issuance
    # entries, so the second verification cannot invalidate the first handle.
    _require(
        budgets._entry(prefix_budget, budgets.SamplingOriginBudget)
        is prefix_budget_entry
        and budgets._entry(budget, budgets.SamplingOriginBudget)
        is receiving_budget_entry
        and prefix_budget.payload == budget.payload == budget_payload,
        "FINAL_BUDGET_ISSUANCE",
    )
    _require(
        budgets._entry(admitted, budgets.SamplingOriginSuccessor) is successor_entry
        and admitted.payload == successor_payload,
        "FINAL_SUCCESSOR_ISSUANCE",
    )
    return result
