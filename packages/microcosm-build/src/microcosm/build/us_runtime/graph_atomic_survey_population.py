"""Complete initial support clones before assigning qualified survey geography.

The existing runner independently admits the raw allocation. This extension
reconstructs every added column from the live preparation and pinned normalized
support, then checks the complete graph populations on cold and warm execution.
Normalized support integrity does not establish its publisher provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from microcosm.build.graph_atomic_geography import (
    ATOMIC_SUPPORT_TYPE as ATOMIC_SUPPORT_TYPE,
)
from microcosm.build.graph_atomic_geography import (
    AtomicSupportImportKernel as AtomicSupportImportKernel,
)
from microcosm.build.graph_atomic_geography import (
    register_atomic_geography_kernels,
)
from microcosm.graph import (
    Graph,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.artifact_edges import typed_contracts
from microcosm.graph.codecs import load_raw_bytes
from microcosm.graph.executor import (
    _all_node_keys,
    _input_writers,
    _source_paths_and_keys,
    _tolerance_writer_payload,
)
from microcosm.graph.keys import (
    _capabilities_projection,
    artifact_key,
    frame_key,
    opaque_artifact_key,
    weights_key,
)
from microcosm.graph.keys import seed as node_seed
from microcosm.graph.manifest import _freeze_json
from microcosm.graph.population import (
    Population,
    mass_record_receipt,
    weight_cap_receipt,
)

from . import graph_combined_clone as clone
from . import graph_current_survey_geography as projection
from . import graph_survey_population as survey
from . import survey_atomic_geography as reconstruction
from .survey_population_replay import same_replayed_population


@dataclass(frozen=True)
class AtomicSurveyPopulationRunValues:
    """Actual run values, without a new source or release certificate."""

    manifest: object
    preparation: object
    allocated_population: Population
    observed_population: Population
    expanded_population: Population
    geography_population: Population
    clone_population: Population
    geography_config: reconstruction.AtomicSurveyReconstruction
    compiled: object
    store: object
    kernels: object
    sources: dict


def _states(compiled, kernels, source_keys, expected_populations, raw_receipts):
    """Bind independent domain receipts to current implementation and graph keys."""
    keys, implementations = _all_node_keys(compiled, kernels, source_keys)
    states, writer_receipts = {}, {}
    for node_id in compiled.order:
        node = compiled.graph.node(node_id)
        population = expected_populations[node_id]
        key = keys[node_id]
        structural = node.structural is not StructuralDelta.NONE
        cells = (
            tuple(
                (e, str(c))
                for e in population.frame.entities
                for c in population.frame.table(e)
            )
            if structural
            else tuple((o.entity, o.column) for o in node.outputs)
        )
        weight_entity = (
            node.weights.entity
            if node.weights is not None
            else node.params.get("expand_weight_entity")
            if node.structural is StructuralDelta.EXPAND
            else None
        )
        capabilities = _capabilities_projection(kernels.get(node.kernel).capabilities)
        receipt = dict(raw_receipts[node_id])
        receipt["capabilities"] = dict(capabilities)
        writers = _tolerance_writer_payload(
            _input_writers(compiled, node_id, receipts=writer_receipts)
        )
        if writers:
            receipt["capabilities"]["tolerance_writers"] = writers
        if structural and node.structural is not StructuralDelta.CREATE:
            receipt["mass"] = {
                **receipt.get("mass", {}),
                **mass_record_receipt(population.mass_ledger[-1]),
            }
        receipt.update(weight_cap_receipt(population, node))
        states[node_id] = {
            "key": key,
            "ref": node.kernel,
            "capabilities": capabilities,
            "implementation": implementations[node_id],
            "typed_artifacts": typed_contracts(compiled, node, keys, kernels),
            "seed": node_seed(key),
            "frame_key": frame_key(key) if structural else None,
            "weight_key": weights_key(key, weight_entity) if weight_entity else None,
            "artifacts": {(e, c): artifact_key(key, e, c) for e, c in cells},
            "opaque_artifacts": {
                o.name: opaque_artifact_key(key, o.name) for o in node.artifact_outputs
            },
            "receipt": _freeze_json(receipt),
        }
        writer_receipts[node_id] = SimpleNamespace(receipt=states[node_id]["receipt"])
    return states


def run_atomic_survey_population(
    source_dir,
    *,
    snapshot_root,
    store_root,
    fraction,
    seed,
    geography_config,
    resume="auto",
    return_values=False,
):
    """Assign each completed source/clone household and independently verify replay."""
    survey._require(type(return_values) is bool, "RETURN_VALUES_FLAG")
    prefix = survey.run_authenticated_survey_population(
        source_dir,
        snapshot_root=snapshot_root,
        store_root=store_root,
        fraction=fraction,
        seed=seed,
        resume=resume,
        clones=False,
        return_values=True,
    )
    source_owner = survey._source_owner()
    preparation_entry = prefix.preparation._checked()
    geography = reconstruction.reconstruct_atomic_survey_geography(
        prefix.preparation, prefix.allocated_population, geography_config
    )
    _, view = survey._checked_preparation(prefix.preparation)
    instructions = survey.allocation_instructions(
        view.selection_plan, view.receipt["origins"]["households"]
    )
    _, allocated_context, allocation_payload, _, _ = survey._allocation_output(
        view.frame, view.context, instructions, survey._sha(view.payload)
    )
    prefix_artifacts = {
        (survey.CREATE_NODE, "preparation"): view.payload,
        (survey.CREATE_NODE, "frame_context"): view.context,
        (survey.ALLOCATION_NODE, "allocation"): allocation_payload,
        (survey.ALLOCATION_NODE, "frame_context"): allocated_context,
    }
    additions = geography.nodes
    compiled = compile_graph(
        Graph(
            "us",
            (
                *prefix.compiled.graph.sources,
                *(SourceRef(name, "raw-bytes-v1") for name, _ in geography.sources),
            ),
            (*prefix.compiled.graph.nodes, *additions),
        )
    )
    survey._require(len(compiled.order) == 9, "ATOMIC_COMPILER_ROSTER")
    sources = {**prefix.sources, **dict(geography.sources)}
    kernels, store = prefix.kernels, prefix.store
    store.codecs.register_bytes("raw-bytes-v1", load_raw_bytes)
    kernels.register(projection.CurrentSurveyGeographyKernel(prefix.preparation))
    register_atomic_geography_kernels(kernels)
    clone.register_us_combined_survey_clone_kernels(kernels)
    expected = {
        survey.CREATE_NODE: Population.from_frame(
            prefix.manifest.population(survey.CREATE_NODE), survey.CREATE_NODE
        ),
        survey.ALLOCATION_NODE: prefix.allocated_population,
        **{stage.node.id: stage.population for stage in geography.stages},
    }
    receipts = {
        **{name: record.receipt for name, record in prefix.manifest.nodes.items()},
        **{stage.node.id: json.loads(stage.receipt) for stage in geography.stages},
    }
    expected_stamps = {
        name: reconstruction._population_stamp(value)
        for name, value in expected.items()
    }
    raw_stamp = reconstruction._population_stamp(prefix.allocated_population)
    source_items = tuple(sorted(sources.items()))
    config_bytes = geography_config.to_bytes()
    _, source_keys = _source_paths_and_keys(compiled, sources, store)
    states = _states(compiled, kernels, source_keys, expected, receipts)
    observed, observed_stamps = {}, {}

    def observe(node_id, population):
        survey._require(node_id not in observed, "DUPLICATE_POPULATION_OBSERVATION")
        same_replayed_population(expected[node_id], population)
        observed[node_id] = population
        observed_stamps[node_id] = reconstruction._population_stamp(population)

    manifest = run_graph(
        compiled,
        sources=sources,
        store=store,
        kernels=kernels,
        resume=resume,
        _population_observer=observe,
    )
    survey._require(tuple(observed) == compiled.order, "POPULATION_OBSERVER_COVERAGE")
    survey._check_node_states(manifest, states)
    for stage in geography.stages:
        for name, payload in stage.artifacts:
            output = next(o for o in stage.node.artifact_outputs if o.name == name)
            survey._final_artifact(
                manifest,
                store,
                node_id=stage.node.id,
                name=name,
                type_=output.type,
                payload=payload,
                capabilities=kernels.get(stage.node.kernel).capabilities,
            )
    # Validate prefix artifacts separately from column materialization.
    for (node_id, name), payload in prefix_artifacts.items():
        key = prefix.manifest.node(node_id).opaque_artifacts[name]
        survey._require(
            manifest.node(node_id).opaque_artifacts.get(name) == key,
            "ATOMIC_PREFIX_ARTIFACT_KEY",
        )
        survey._require(
            store.load_bytes(key) == payload, "ATOMIC_PREFIX_ARTIFACT_BYTES"
        )
    geography_terminal = geography.nodes[-1].id
    result = AtomicSurveyPopulationRunValues(
        manifest=manifest,
        preparation=prefix.preparation,
        allocated_population=prefix.allocated_population,
        observed_population=observed[projection.NODE],
        expanded_population=observed[clone.COMBINED_CLONE_CLAIM_NODE],
        geography_population=observed[geography_terminal],
        # This historical field is the terminal receiving population. The
        # explicit expanded_population is the preassignment clone snapshot.
        clone_population=observed[geography_terminal],
        geography_config=geography_config,
        compiled=compiled,
        store=store,
        kernels=kernels,
        sources=sources,
    )
    current_keys, current_implementations = _all_node_keys(
        compiled, kernels, source_keys
    )
    survey._require(
        current_keys == {name: value["key"] for name, value in states.items()}
        and current_implementations
        == {name: value["implementation"] for name, value in states.items()},
        "ATOMIC_FINAL_IMPLEMENTATIONS",
    )
    # Finish source/support I/O before final comparisons of returned objects.
    fresh = reconstruction.reconstruct_atomic_survey_geography(
        prefix.preparation, prefix.allocated_population, geography_config
    )
    survey._require(
        source_owner._ISSUED.get(id(prefix.preparation)) is preparation_entry
        and prefix.preparation.payload == preparation_entry[1],
        "ATOMIC_FINAL_PREPARATION",
    )
    source_owner._pure_final(preparation_entry[2])
    survey._require(
        fresh.config_sha256 == geography.config_sha256
        and fresh.projection_receipt == geography.projection_receipt
        and fresh.support_payload == geography.support_payload
        and fresh.definition == geography.definition
        and geography_config.to_bytes() == config_bytes
        and result.geography_config is geography_config
        and result.preparation is prefix.preparation
        and result.allocated_population is prefix.allocated_population
        and result.observed_population is observed[projection.NODE]
        and result.expanded_population is observed[clone.COMBINED_CLONE_CLAIM_NODE]
        and result.geography_population is observed[geography_terminal]
        and result.clone_population is result.geography_population
        and result.manifest is manifest
        and result.compiled is compiled
        and result.store is store
        and result.kernels is kernels
        and tuple(sorted(result.sources.items())) == source_items
        and reconstruction._population_stamp(prefix.allocated_population) == raw_stamp,
        "ATOMIC_FINAL_RECONSTRUCTION",
    )
    same_replayed_population(fresh.observed_population, result.observed_population)
    same_replayed_population(fresh.expanded_population, result.expanded_population)
    same_replayed_population(fresh.population, result.geography_population)
    for node_id, population in expected.items():
        survey._require(
            reconstruction._population_stamp(population) == expected_stamps[node_id]
            and reconstruction._population_stamp(observed[node_id])
            == observed_stamps[node_id],
            "ATOMIC_FINAL_POPULATION_MUTATION",
        )
        same_replayed_population(population, observed[node_id])
    latest = {}
    for node_id in compiled.order:
        latest[compiled.versions[node_id]] = node_id
    for version, node_id in latest.items():
        survey._same_frame(expected[node_id].frame, manifest.population(version))
        survey._require(
            manifest.mass_ledger(version) == expected[node_id].mass_ledger,
            "ATOMIC_FINAL_MASS_LEDGER",
        )
    survey._check_node_states(manifest, states)
    return result if return_values else manifest
