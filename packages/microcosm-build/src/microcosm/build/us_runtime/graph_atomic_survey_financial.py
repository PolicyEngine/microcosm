"""Checked atomic survey geography plus current financial development output.

The base nineteen nodes retain the raw allocation and the pre-financial clone
separately. An explicit property-income option adds sixteen nodes and retains
the complete legacy financial population alongside its extended output. Support
bytes establish integrity, not publisher provenance or release eligibility.
"""

from __future__ import annotations

import json
import sys
import weakref
from dataclasses import dataclass, replace
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.fit import qrf_target
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.graph import KernelResult, StructuralDelta, compile_graph, run_graph
from microcosm.graph import population as population_ops
from microcosm.graph.artifact_edges import typed_contracts
from microcosm.graph.executor import _all_node_keys, _source_paths_and_keys
from microcosm.graph.keys import _capabilities_projection
from microcosm.graph.population import Population
from microcosm.graph.serialize import graph_to_json

from . import graph_atomic_survey_population as atomic
from . import graph_current_survey_predictors as financial

values = financial.values
codec = financial.codec
survey = atomic.survey
reconstruction = atomic.reconstruction
require = values.require
RUN_PROTOCOL = "microcosm.us.atomic-survey-financial-run.v1"
_ISSUED_RUNS = {}


@dataclass(frozen=True)
class AtomicSurveyFinancialRunValues:
    """Development output, retaining the independently admitted prefix values."""

    prefix: atomic.AtomicSurveyPopulationRunValues
    financial_population: Population
    manifest: object
    compiled: object
    store: object
    kernels: object
    sources: dict
    projection: bytes
    matrix: bytes

    def checked_view(self):
        """Recheck a completed actual run; public dataclass copies are unissued."""
        return check_atomic_survey_financial_run(self)


@dataclass(frozen=True)
class CheckedAtomicSurveyFinancialRun:
    """Descriptive values; authority remains in the actual issued run handle."""

    payload: bytes
    digest: str
    population: Population


@dataclass(frozen=True)
class _FinancialRunState:
    prefix: object
    prefix_objects: tuple
    preparation_entry: tuple
    financial_population: Population
    populations: tuple
    manifest: object
    manifest_bytes: bytes
    prefix_manifest_bytes: bytes
    compiled: object
    declaration: str
    prefix_declaration: str
    store: object
    kernels: object
    source_items: tuple
    config_bytes: bytes
    projection: bytes
    matrix: bytes
    pins: bytes
    n_estimators: int
    demographic_conditioning: bool
    source_keys: tuple
    keys: tuple
    implementations: tuple
    artifact_hashes: tuple
    manifest_populations: tuple
    prefix_manifest_populations: tuple
    live: dict
    property_income: object = None
    property_income_bytes: bytes | None = None
    legacy_financial_population: Population | None = None
    legacy_financial_stamp: object = None


def _property_module():
    """Load the explicitly selected extension without changing the default path."""
    from . import graph_current_survey_property

    return graph_current_survey_property


def _manifest_population_seals(manifest, compiled):
    """Include transient attached Frames, which portable JSON deliberately omits."""
    return tuple(
        (
            version,
            reconstruction._population_stamp(
                Population.from_frame(
                    manifest.population(version),
                    version,
                    mass_ledger=manifest.mass_ledger(version),
                )
            ),
        )
        for version in sorted(set(compiled.versions.values()))
    )


def _run_entry(run):
    entry = _ISSUED_RUNS.get(id(run))
    require(
        type(run) is AtomicSurveyFinancialRunValues
        and entry is not None
        and entry[0]() is run,
        "UNISSUED_FINANCIAL_RUN",
    )
    return entry


def financial_output_node(run):
    """Name the final writer of an already issued financial run."""
    state = _run_entry(run)[2]
    return (
        financial.ATTACH_NODE
        if state.property_income is None
        else _property_module().ATTACH_NODE
    )


def _run_document(run, state):
    """Portable ancestry omits timing/cache hits and private physical seals."""
    return codec.encode_json(
        {
            "protocol": RUN_PROTOCOL,
            "graph_sha256": codec.sha(state.declaration.encode()),
            "manifest_key": state.manifest.key,
            "preparation_sha256": codec.sha(state.preparation_entry[1]),
            "geography_config_sha256": codec.sha(state.config_bytes),
            "projection_sha256": codec.sha(state.projection),
            "matrix_sha256": codec.sha(state.matrix),
            "host_edges": codec.decode_json(state.pins),
            "node_keys": dict(state.keys),
            "artifact_payload_sha256": [list(row) for row in state.artifact_hashes],
            "financial_frame_sha256": values.source._frame_identity(
                run.financial_population.frame
            ),
            "financial_version": run.financial_population.version,
            "financial_owners": sorted(
                (e, c, writer)
                for (e, c), writer in run.financial_population.owners.items()
            ),
            "owned_columns": [
                *values.OUTPUTS,
                *(
                    ()
                    if state.property_income is None
                    else (owned.column for owned in _property_module().owned_columns())
                ),
            ],
            "demographic_conditioning": state.demographic_conditioning,
            "n_estimators": state.n_estimators,
            "release_eligible": False,
            **(
                {}
                if state.property_income is None
                else {
                    "property_income": codec.decode_json(state.property_income_bytes),
                    "property_node_count": 16,
                    "tax_split_rebased": False,
                    "capital_gains_conditioning": _property_module().CAP_LIMITATION,
                    "legacy_financial_frame_sha256": values.source._frame_identity(
                        state.legacy_financial_population.frame
                    ),
                }
            ),
        }
    )


def _pure_run(run, entry):
    """Check retained graph/source/Population state after all external I/O."""
    require(_run_entry(run) is entry, "FINAL_FINANCIAL_RUN_ISSUANCE")
    state = entry[2]
    prefix = state.prefix
    require(
        run.prefix is prefix
        and run.financial_population is state.financial_population
        and run.manifest is state.manifest
        and run.compiled is state.compiled
        and run.store is state.store
        and run.kernels is state.kernels
        and run.projection == state.projection
        and run.matrix == state.matrix
        and tuple(sorted(run.sources.items())) == state.source_items
        and tuple(sorted(prefix.sources.items())) == state.source_items
        and all(
            a is b
            for a, b in zip(
                (
                    prefix.preparation,
                    prefix.manifest,
                    prefix.compiled,
                    prefix.store,
                    prefix.kernels,
                    prefix.sources,
                    prefix.geography_config,
                ),
                state.prefix_objects,
                strict=True,
            )
        )
        and values.host.survey_budget._config_payload(prefix.geography_config)
        == state.config_bytes
        and run.manifest.to_json_bytes() == state.manifest_bytes
        and prefix.manifest.to_json_bytes() == state.prefix_manifest_bytes
        and graph_to_json(run.compiled.graph) == state.declaration
        and graph_to_json(prefix.compiled.graph) == state.prefix_declaration
        and run.compiled == compile_graph(run.compiled.graph)
        and prefix.compiled == compile_graph(prefix.compiled.graph),
        "FINANCIAL_RUN_BINDINGS_CHANGED",
    )
    require(
        _manifest_population_seals(run.manifest, run.compiled)
        == state.manifest_populations
        and _manifest_population_seals(prefix.manifest, prefix.compiled)
        == state.prefix_manifest_populations,
        "FINANCIAL_RUN_ATTACHED_POPULATION_CHANGED",
    )
    require(
        values.source._ISSUED.get(id(prefix.preparation)) is state.preparation_entry
        and prefix.preparation.payload == state.preparation_entry[1],
        "FINANCIAL_RUN_SOURCE_CHANGED",
    )
    values.source._pure_final(state.preparation_entry[2])
    if state.property_income is not None:
        require(
            type(state.property_income) is _property_module().PropertyIncomeOptions
            and state.property_income.to_bytes() == state.property_income_bytes
            and reconstruction._population_stamp(state.legacy_financial_population)
            == state.legacy_financial_stamp,
            "PROPERTY_OPTIONS_OR_LEGACY_POPULATION_CHANGED",
        )
    for name, population, stamp in state.populations:
        actual = (
            run.financial_population if name == "financial" else getattr(prefix, name)
        )
        require(
            actual is population and reconstruction._population_stamp(actual) == stamp,
            "FINANCIAL_RUN_POPULATION_CHANGED",
        )
    require(
        _run_document(run, state) == entry[1]
        and _live(state.property_income) == state.live,
        "FINAL_FINANCIAL_RUN_SEAL",
    )
    require(_run_entry(run) is entry, "FINAL_FINANCIAL_RUN_ISSUANCE")


def check_atomic_survey_financial_run(run):
    """Requalify existing artifacts and live owners without fitting or execution."""
    entry = _run_entry(run)
    _pure_run(run, entry)
    state, prefix = entry[2], entry[2].prefix
    _, source_keys = _source_paths_and_keys(run.compiled, run.sources, run.store)
    keys, implementations = _all_node_keys(run.compiled, run.kernels, source_keys)
    require(
        tuple(sorted(source_keys.items())) == state.source_keys
        and tuple(sorted(keys.items())) == state.keys
        and tuple(sorted(implementations.items())) == state.implementations,
        "FINANCIAL_RUN_IMPLEMENTATIONS_CHANGED",
    )
    loaded = _artifacts(
        run.manifest, run.compiled, run.store, run.kernels, keys, implementations
    )
    require(
        tuple(
            sorted(
                (node, name, codec.sha(payload))
                for (node, name), payload in loaded.items()
            )
        )
        == state.artifact_hashes,
        "FINANCIAL_RUN_ARTIFACT_CHANGED",
    )
    # These exact fitted artifacts were checked at actual execution/required
    # replay before issuance. Rechecking their identities needs no new pickle
    # decode or fit; the materialized verifier freshly derives current values.
    financial.verify_materialized_current_survey_predictors(
        prefix.preparation,
        prefix.allocated_population,
        prefix.clone_population,
        population=(
            run.financial_population
            if state.property_income is None
            else state.legacy_financial_population
        ),
        projection=state.projection,
        matrix=state.matrix,
        matrix_producer_key=keys[financial.PROJECTION_NODE],
        raw_draws=tuple(
            loaded[f"{financial.APPLY_PREFIX}.{i:03d}", "raw_draw"] for i in range(3)
        ),
        apply_states=tuple(
            loaded[f"{financial.APPLY_PREFIX}.{i:03d}", "apply_state"] for i in range(3)
        ),
        host_pins=codec.decode_json(state.pins),
        n_estimators=state.n_estimators,
        demographic_conditioning=state.demographic_conditioning,
        geography_config=prefix.geography_config,
    )
    if state.property_income is not None:
        _property_module().verify_materialized_property_income(
            prefix.preparation,
            prefix.allocated_population,
            prefix.clone_population,
            legacy_population=state.legacy_financial_population,
            population=run.financial_population,
            host_pins=codec.decode_json(state.pins),
            options=state.property_income,
            artifacts=loaded,
            legacy_matrix_producer_key=keys[financial.PROJECTION_NODE],
            demographic_conditioning=state.demographic_conditioning,
            geography_config=prefix.geography_config,
        )
    result = CheckedAtomicSurveyFinancialRun(
        entry[1], codec.sha(entry[1]), run.financial_population
    )
    _pure_run(run, entry)
    return result


def _issue_run(
    result,
    *,
    preparation_entry,
    pins,
    n_estimators,
    demographic_conditioning,
    source_keys,
    keys,
    implementations,
    loaded,
    live,
    property_income=None,
    legacy_financial_population=None,
):
    """Called only after this runner's complete materialization/replay checks."""
    prefix = result.prefix
    populations = tuple(
        (name, population, reconstruction._population_stamp(population))
        for name, population in (
            *(
                (name, getattr(prefix, name))
                for name in (
                    "allocated_population",
                    "observed_population",
                    "expanded_population",
                    "geography_population",
                    "clone_population",
                )
            ),
            ("financial", result.financial_population),
        )
    )
    state = _FinancialRunState(
        prefix,
        (
            prefix.preparation,
            prefix.manifest,
            prefix.compiled,
            prefix.store,
            prefix.kernels,
            prefix.sources,
            prefix.geography_config,
        ),
        preparation_entry,
        result.financial_population,
        populations,
        result.manifest,
        result.manifest.to_json_bytes(),
        prefix.manifest.to_json_bytes(),
        result.compiled,
        graph_to_json(result.compiled.graph),
        graph_to_json(prefix.compiled.graph),
        result.store,
        result.kernels,
        tuple(sorted(result.sources.items())),
        prefix.geography_config.to_bytes(),
        result.projection,
        result.matrix,
        codec.encode_json(pins),
        n_estimators,
        demographic_conditioning,
        tuple(sorted(source_keys.items())),
        tuple(sorted(keys.items())),
        tuple(sorted(implementations.items())),
        tuple(
            sorted(
                (node, name, codec.sha(payload))
                for (node, name), payload in loaded.items()
            )
        ),
        _manifest_population_seals(result.manifest, result.compiled),
        _manifest_population_seals(prefix.manifest, prefix.compiled),
        live,
        property_income,
        None if property_income is None else property_income.to_bytes(),
        legacy_financial_population,
        None
        if legacy_financial_population is None
        else reconstruction._population_stamp(legacy_financial_population),
    )
    identifier = id(result)

    def forget(reference):
        entry = _ISSUED_RUNS.get(identifier)
        if entry is not None and entry[0] is reference:
            _ISSUED_RUNS.pop(identifier, None)

    reference = weakref.ref(result, forget)
    _ISSUED_RUNS[identifier] = (reference, _run_document(result, state), state)
    _pure_run(result, _run_entry(result))


def _live(property_income=None):
    """Pure final fence over this composition and its existing owner closure."""
    result = dict(values.host.survey_budget._live())
    modules = (
        sys.modules[__name__],
        atomic,
        financial,
        values,
        codec,
        qrf_target,
        financial.qrf,
        financial.model_input,
        sys.modules[LegacyQRFTrainKernel.__module__],
        sys.modules[LegacyQRFApplyMatrixKernel.__module__],
    )
    if property_income is not None:
        from . import current_asec_property_basis, graph_property_income_receipts

        extension = _property_module()
        modules = (
            *modules,
            extension,
            extension.sources,
            current_asec_property_basis,
            extension.sources.acs,
            extension.sources.interest,
            extension.sources.routing,
            extension.sources.dividend,
            extension.model,
            extension.signed_graph,
            graph_property_income_receipts,
            sys.modules[extension.LegacyQRFApplyKernel.__module__],
        )
        result["property_contract"] = values.source._runtime_marker(
            (
                extension.PROTOCOL,
                extension.PREFIX,
                extension.CAP_LIMITATION,
                extension.LEGACY_DIFFERENCE,
                extension.BASIS_DIAGNOSTICS,
                extension.PROPERTY_COMPONENTS,
                extension.PROPERTY_DRAW_COLUMNS,
                extension.PROPERTY_REPORTED_TOTAL,
                extension.model.PROTOCOL,
                extension.sources.PROTOCOL,
                current_asec_property_basis.OTHER_PROPERTY_CATEGORIES,
                current_asec_property_basis.OTHER_UNSPECIFIED_CATEGORY,
            )
        )
    for module in modules:
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                result[module.__name__, name] = values.source._function_seal(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                result[module.__name__, name] = value
                for method, function in vars(value).items():
                    if isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        result[module.__name__, name, method] = (
                            values.source._function_seal(function)
                        )
    result["financial_contract"] = values.source._runtime_marker(
        (
            RUN_PROTOCOL,
            values.FEATURES,
            values.DEMOGRAPHIC_FEATURES,
            values.TARGETS,
            values.OUTPUTS,
            values.PROTOCOL,
            values.PHASE,
            values.SEED,
        )
    )
    return result


def _artifacts(manifest, compiled, store, kernels, keys, implementations):
    """Read only the declared outputs after binding their actual producer keys."""
    require(set(manifest.nodes) == set(compiled.order), "ATOMIC_NODE_ROSTER")
    loaded = {}
    for node_id in compiled.order:
        node, record = compiled.graph.node(node_id), manifest.node(node_id)
        kernel = kernels.get(node.kernel)
        require(
            record.key == keys[node_id]
            and record.kernel_ref == node.kernel
            and record.kernel_impl_hash == implementations[node_id]
            and _capabilities_projection(record.capabilities)
            == _capabilities_projection(kernel.capabilities)
            and record.typed_artifacts
            == typed_contracts(compiled, node, keys, kernels),
            "ATOMIC_ARTIFACT_PRODUCER",
        )
        require(
            set(record.opaque_artifacts) == {o.name for o in node.artifact_outputs},
            "ATOMIC_ARTIFACT_ROSTER",
        )
        for output in node.artifact_outputs:
            payload = store.load_bytes(record.opaque_artifacts[output.name])
            survey._final_artifact(
                manifest,
                store,
                node_id=node_id,
                name=output.name,
                type_=output.type,
                payload=payload,
                capabilities=kernel.capabilities,
            )
            loaded[node_id, output.name] = payload
    return loaded


def _model_receipts(nodes, donor, qualified, loaded, matrix_key):
    """Bind fitted checkpoints to the exact source-derived design-weight donor.

    The start-chain operation only resolves inputs and initializes RNG state;
    this verifier does not fit a second model. Pickles are decoded only after
    the caller has checked the actual store and typed graph producer closure.
    """
    fits = tuple(n for n in nodes if n.kernel == LegacyQRFTrainKernel.ref)
    first = fits[0]
    model_frame = codec.model_frame(
        SimpleNamespace(weights={"person": donor.frame.resolve_weights("person")}),
        first.inputs[0],
        donor.frame.person,
    )
    predictors = values.feature_columns(qualified.demographic_conditioning)
    model = financial.qrf.RegimeGatedQRF(
        seed=values.SEED,
        n_estimators=first.params["n_estimators"],
        zero_atol=0,
        max_samples_leaf=None,
    )
    before = qrf_target.LegacyQRFTrainingState.from_chain(
        model.start_chain(
            model_frame, list(predictors), list(values.TARGETS), weights="design"
        )
    )
    receipts, history = {}, []
    for i, fit in enumerate(fits):
        target = values.TARGETS[i]
        payload = loaded[fit.id, "model"]
        packet, after = codec.read_training(loaded[fit.id, "training_state"])
        require(
            len(packet["models"]) == i + 1
            and packet["models"][:i] == history
            and packet["models"][-1]["sha256"] == codec.sha(payload),
            "ATOMIC_TRAINING_HISTORY",
        )
        fitted = qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
            payload, expected_sha256=packet["models"][-1]["sha256"]
        )
        require(
            fitted.training_state == before
            and fitted.next_training_state == after
            and fitted.donor_sha256
            == qrf_target._consumed_values_sha256(
                model_frame.person, (*predictors, *values.TARGETS[:i], target)
            ),
            "ATOMIC_TRAINING_DONOR",
        )
        history = [
            *history,
            {
                "target": target,
                "sha256": codec.sha(payload),
                "training_id": fitted.training_id,
            },
        ]
        require(packet["models"] == history, "ATOMIC_TRAINING_HISTORY")
        before = after
        receipts[fit.id] = {
            "phase": values.PHASE,
            "target": target,
            "training_id": fitted.training_id,
            "model_sha256": codec.sha(payload),
            "donor_rows": len(model_frame.person),
            "entity": "person",
            "weight_kind": "design",
            "regime": fitted.regime,
        }
        apply_id = f"{financial.APPLY_PREFIX}.{i:03d}"
        application = financial.decode_matrix_apply_state(
            loaded[apply_id, "apply_state"]
        )
        require(
            application["application"]["models"] == history,
            "ATOMIC_APPLY_MODEL_HISTORY",
        )
        receipts[apply_id] = {
            "phase": values.PHASE,
            "target": target,
            "recipient_rows": len(
                financial.model_input.decode_recipient_matrix(qualified.matrix).features
            ),
            "entity": "person",
            "model_sha256": codec.sha(payload),
            "raw_sha256": codec.sha(loaded[apply_id, "raw_draw"]),
            "regime": fitted.regime,
            "matrix_sha256": codec.sha(qualified.matrix),
            "matrix_producer_key": matrix_key,
        }
    return receipts


def run_atomic_survey_financial(
    source_dir,
    *,
    snapshot_root,
    store_root,
    fraction,
    seed,
    geography_config,
    demographic_conditioning=False,
    n_estimators=100,
    property_income=None,
    resume="auto",
    return_values=False,
):
    """Verify the base financial graph and its explicitly selected extension."""
    require(type(return_values) is bool, "RETURN_VALUES_FLAG")
    values.feature_columns(demographic_conditioning)
    property_graph = None if property_income is None else _property_module()
    if property_graph is not None:
        require(
            type(property_income) is property_graph.PropertyIncomeOptions,
            "PROPERTY_OPTIONS_TYPE",
        )
        property_income_bytes = property_income.to_bytes()
    else:
        property_income_bytes = None
    live = _live(property_income)
    config_bytes = values.host.survey_budget._config_payload(geography_config)
    require(config_bytes is not None, "ATOMIC_GEOGRAPHY_REQUIRED")
    prefix = atomic.run_atomic_survey_population(
        source_dir,
        snapshot_root=snapshot_root,
        store_root=store_root,
        fraction=fraction,
        seed=seed,
        geography_config=geography_config,
        resume=resume,
        return_values=True,
    )
    entry = prefix.preparation._checked()
    prefix_objects = (
        prefix.preparation,
        prefix.manifest,
        prefix.compiled,
        prefix.store,
        prefix.kernels,
        prefix.sources,
    )
    prefix_manifest_bytes = prefix.manifest.to_json_bytes()
    prefix_declaration = graph_to_json(prefix.compiled.graph)
    retained = {
        name: (
            getattr(prefix, name),
            reconstruction._population_stamp(getattr(prefix, name)),
        )
        for name in (
            "allocated_population",
            "observed_population",
            "expanded_population",
            "geography_population",
            "clone_population",
        )
    }
    qualified = values.qualify_current_survey_predictors(
        prefix.preparation,
        prefix.allocated_population,
        prefix.clone_population,
        demographic_conditioning=demographic_conditioning,
        geography_config=geography_config,
    )
    projection_bytes, matrix_bytes = qualified.projection, qualified.matrix
    property_qualified = (
        None
        if property_graph is None
        else property_graph.sources.qualify_current_property_income_sources(
            prefix.preparation,
            prefix.allocated_population,
            prefix.clone_population,
            demographic_conditioning=demographic_conditioning,
            geography_config=geography_config,
        )
    )
    geography = reconstruction.reconstruct_atomic_survey_geography(
        prefix.preparation, prefix.allocated_population, geography_config
    )
    pins, prefix_artifacts = {}, {}
    for node_id, record in prefix.manifest.nodes.items():
        for name, key in record.opaque_artifacts.items():
            prefix_artifacts[node_id, name] = prefix.store.load_bytes(key)
    for edge in (
        *financial.host.current_survey_host_edges(),
        financial._geography_edge(),
    ):
        record = prefix.manifest.node(edge.producer)
        pins[edge.name] = {
            "producer_key": record.key,
            "artifact_key": record.opaque_artifacts[edge.artifact],
            "payload_sha256": codec.sha(prefix_artifacts[edge.producer, edge.artifact]),
        }
    nodes = financial.current_survey_predictor_nodes(
        qualified,
        prefix.clone_population.frame,
        host_pins=pins,
        n_estimators=n_estimators,
    )
    property_nodes = (
        ()
        if property_graph is None
        else property_graph.current_survey_property_nodes(
            property_qualified,
            prefix.clone_population.frame,
            host_pins=pins,
            options=property_income,
        )
    )
    compiled = compile_graph(
        replace(
            prefix.compiled.graph,
            nodes=(*prefix.compiled.graph.nodes, *nodes, *property_nodes),
        )
    )
    require(
        len(prefix.compiled.order) == 9
        and len(property_nodes) == (0 if property_graph is None else 16)
        and len(compiled.order) == 19 + len(property_nodes),
        "ATOMIC_COMPILER_ROSTER",
    )
    gate_edge = financial._geography_edge()
    require(
        gate_edge.producer in compiled.predecessors[financial.DONOR_NODE]
        and prefix_artifacts[gate_edge.producer, gate_edge.artifact]
        == qualified.geography_validation,
        "ATOMIC_GEOGRAPHY_GATE_EDGE",
    )
    declaration = graph_to_json(compiled.graph)
    kernels, store, sources = prefix.kernels, prefix.store, dict(prefix.sources)
    source_items = tuple(sorted(sources.items()))
    for cls in (
        financial.CurrentSurveyPredictorProjectionKernel,
        financial.CurrentSurveyPredictorDonorFilterKernel,
        financial.CurrentSurveyPredictorDonorColumnsKernel,
        financial.CurrentSurveyPredictorAttachKernel,
    ):
        kernels.register(
            cls(
                prefix.preparation,
                prefix.allocated_population,
                prefix.clone_population,
                host_pins=pins,
                n_estimators=n_estimators,
                demographic_conditioning=demographic_conditioning,
                geography_config=geography_config,
            )
        )
    kernels.register(LegacyQRFTrainKernel())
    kernels.register(LegacyQRFApplyMatrixKernel())
    if property_graph is not None:
        property_graph.register_property_kernels(
            kernels,
            prefix.preparation,
            prefix.allocated_population,
            prefix.clone_population,
            host_pins=pins,
            options=property_income,
            demographic_conditioning=demographic_conditioning,
            geography_config=geography_config,
        )
    _, source_keys = _source_paths_and_keys(compiled, sources, store)
    keys, implementations = _all_node_keys(compiled, kernels, source_keys)
    base_expected = {
        survey.CREATE_NODE: Population.from_frame(entry[2].frame, survey.CREATE_NODE),
        survey.ALLOCATION_NODE: prefix.allocated_population,
        **{s.node.id: s.population for s in geography.stages},
    }
    receipts = {
        **{n: r.receipt for n, r in prefix.manifest.nodes.items()},
        **{s.node.id: json.loads(s.receipt) for s in geography.stages},
    }
    donor_node = compiled.graph.node(financial.DONOR_NODE)
    donor = population_ops.patch(
        base_expected[survey.CREATE_NODE],
        donor_node,
        KernelResult(frame=qualified.donor_frame),
    )
    receipts[donor_node.id] = {
        "selection": "ASEC_native_whole_households",
        "fit_weight_kind": "design",
        "release_eligible": False,
    }
    columns_node = compiled.graph.node(financial.DONOR_COLUMNS_NODE)
    donor_columns = population_ops.patch(
        donor,
        columns_node,
        KernelResult(
            columns={
                ("person", c): qualified.donor_columns[c]
                for c in (
                    *values.feature_columns(demographic_conditioning),
                    *values.TARGETS,
                )
            }
        ),
    )
    receipts[financial.PROJECTION_NODE] = qualified.evidence
    receipts[columns_node.id] = qualified.evidence
    observed, observed_stamps = {}, {}

    def observe(node_id, population):
        require(node_id not in observed, "ATOMIC_OBSERVER_DUPLICATE")
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
    require(tuple(observed) == compiled.order, "ATOMIC_OBSERVER_ROSTER")
    loaded = _artifacts(manifest, compiled, store, kernels, keys, implementations)
    require(
        all(loaded[k] == v for k, v in prefix_artifacts.items()),
        "ATOMIC_PREFIX_ARTIFACTS",
    )
    require(
        loaded[financial.PROJECTION_NODE, "projection"] == projection_bytes
        and loaded[financial.PROJECTION_NODE, "matrix"] == matrix_bytes,
        "ATOMIC_SOURCE_ARTIFACTS",
    )
    raw = tuple(
        loaded[f"{financial.APPLY_PREFIX}.{i:03d}", "raw_draw"] for i in range(3)
    )
    applications = tuple(
        loaded[f"{financial.APPLY_PREFIX}.{i:03d}", "apply_state"] for i in range(3)
    )
    matrix_key = keys[financial.PROJECTION_NODE]
    drawn = financial.read_current_survey_draws(
        matrix_bytes,
        matrix_key,
        raw,
        applications,
        demographic_conditioning=demographic_conditioning,
    )
    receipts.update(
        _model_receipts(nodes, donor_columns, qualified, loaded, matrix_key)
    )
    receipts[financial.ATTACH_NODE] = {
        **qualified.evidence,
        "raw_sha256": [codec.sha(r) for r in raw],
        "all_output_cells_available": True,
        "ACS_financial_origin": "modeled",
        "paired_draws": "source_origin_join",
        "host_weights_changed": False,
    }
    property_results = (
        {}
        if property_graph is None
        else property_graph.reconstruct_property_results(
            property_qualified,
            prefix.clone_population.frame,
            host_pins=pins,
            options=property_income,
            artifacts=loaded,
            legacy_matrix_producer_key=matrix_key,
        )
    )
    receipts.update({name: result.receipt for name, result in property_results.items()})
    expected, current = {}, {}
    for node_id in compiled.order:
        node, version = compiled.graph.node(node_id), compiled.versions[node_id]
        if node_id in property_results:
            incumbent = (
                version if node.structural is StructuralDelta.NONE else node.base
            )
            property_result = property_results[node_id]
            if node.structural is StructuralDelta.FILTER:
                frame = current[incumbent].frame
                entity = frame.schema.person_entity
                id_column = frame.schema.entity_id_column(entity)
                ids = pd.Index(frame.table(entity)[id_column], name=id_column)
                mask = property_result.keep.reindex(ids).to_numpy(
                    dtype=np.bool_, copy=True
                )
                property_result = replace(
                    property_result, frame=frame.select(mask), keep=None
                )
            population = population_ops.patch(current[incumbent], node, property_result)
        elif node_id == donor_node.id:
            population = donor
        elif node_id == columns_node.id:
            population = donor_columns
        elif node_id == financial.ATTACH_NODE:
            population = population_ops.patch(
                current[version],
                node,
                KernelResult(
                    columns=values.complete_predictor_columns(
                        qualified, prefix.clone_population.frame, drawn
                    )
                ),
            )
        elif node_id in base_expected:
            population = base_expected[node_id]
        else:
            population = current[version]
        expected[node_id] = current[version] = population
        atomic.same_replayed_population(population, observed[node_id])
    if property_graph is not None:
        from .graph_property_income_receipts import verify_property_model_receipts

        receipts.update(
            verify_property_model_receipts(
                property_nodes,
                expected[property_graph.DONOR_COLUMNS_NODE],
                expected[property_graph.RECIPIENT_COLUMNS_NODE],
                loaded,
            )
        )
    expected_stamps = {
        n: reconstruction._population_stamp(p) for n, p in expected.items()
    }
    states = atomic._states(compiled, kernels, source_keys, expected, receipts)
    survey._check_node_states(manifest, states)
    final_node = (
        financial.ATTACH_NODE if property_graph is None else property_graph.ATTACH_NODE
    )
    legacy_population = observed[financial.ATTACH_NODE]
    result = AtomicSurveyFinancialRunValues(
        prefix,
        observed[final_node],
        manifest,
        compiled,
        store,
        kernels,
        sources,
        projection_bytes,
        matrix_bytes,
    )
    # Implementation/source/store I/O precedes the last owner/support borrow.
    current_keys, current_implementations = _all_node_keys(
        compiled, kernels, source_keys
    )
    require(
        current_keys == keys and current_implementations == implementations,
        "ATOMIC_FINAL_IMPLEMENTATIONS",
    )
    financial.verify_materialized_current_survey_predictors(
        prefix.preparation,
        prefix.allocated_population,
        prefix.clone_population,
        population=legacy_population,
        projection=projection_bytes,
        matrix=matrix_bytes,
        matrix_producer_key=matrix_key,
        raw_draws=raw,
        apply_states=applications,
        host_pins=pins,
        n_estimators=n_estimators,
        demographic_conditioning=demographic_conditioning,
        geography_config=geography_config,
    )
    if property_graph is not None:
        property_graph.verify_materialized_property_income(
            prefix.preparation,
            prefix.allocated_population,
            prefix.clone_population,
            legacy_population=legacy_population,
            population=result.financial_population,
            host_pins=pins,
            options=property_income,
            artifacts=loaded,
            legacy_matrix_producer_key=matrix_key,
            demographic_conditioning=demographic_conditioning,
            geography_config=geography_config,
        )
        require(
            property_income.to_bytes() == property_income_bytes,
            "PROPERTY_OPTIONS_CHANGED",
        )
    values.source._pure_final(entry[2])
    require(
        values.source._ISSUED.get(id(prefix.preparation)) is entry
        and prefix.preparation.payload == entry[1]
        and prefix.geography_config is geography_config
        and values.host.survey_budget._config_payload(geography_config) == config_bytes
        and result.prefix is prefix
        and result.manifest is manifest
        and result.compiled is compiled
        and result.store is store
        and result.kernels is kernels
        and result.financial_population is observed[final_node]
        and result.projection == projection_bytes
        and result.matrix == matrix_bytes
        and all(
            a is b
            for a, b in zip(
                (
                    prefix.preparation,
                    prefix.manifest,
                    prefix.compiled,
                    prefix.store,
                    prefix.kernels,
                    prefix.sources,
                ),
                prefix_objects,
                strict=True,
            )
        )
        and prefix.manifest.to_json_bytes() == prefix_manifest_bytes
        and graph_to_json(prefix.compiled.graph) == prefix_declaration
        and prefix.compiled == compile_graph(prefix.compiled.graph)
        and tuple(sorted(result.sources.items())) == source_items
        and tuple(sorted(prefix.sources.items())) == source_items
        and graph_to_json(compiled.graph) == declaration
        and compiled == compile_graph(compiled.graph)
        and _live(property_income) == live,
        "ATOMIC_FINAL_BINDINGS",
    )
    for name, (population, stamp) in retained.items():
        require(
            getattr(prefix, name) is population
            and reconstruction._population_stamp(population) == stamp,
            "ATOMIC_FINAL_PREFIX_MUTATION",
        )
    for version, population in (
        (survey.CREATE_NODE, base_expected[survey.CREATE_NODE]),
        (survey.ALLOCATION_NODE, prefix.observed_population),
        (atomic.clone.COMBINED_CLONE_NODE, prefix.clone_population),
    ):
        survey._same_frame(population.frame, prefix.manifest.population(version))
        require(
            population.mass_ledger == prefix.manifest.mass_ledger(version),
            "ATOMIC_FINAL_PREFIX_LEDGER",
        )
    for node_id, population in expected.items():
        require(
            reconstruction._population_stamp(population) == expected_stamps[node_id]
            and reconstruction._population_stamp(observed[node_id])
            == observed_stamps[node_id],
            "ATOMIC_FINAL_POPULATION_MUTATION",
        )
        atomic.same_replayed_population(population, observed[node_id])
    for version, population in current.items():
        survey._same_frame(population.frame, manifest.population(version))
        require(
            population.mass_ledger == manifest.mass_ledger(version),
            "ATOMIC_FINAL_LEDGER",
        )
    survey._check_node_states(manifest, states)
    _issue_run(
        result,
        preparation_entry=entry,
        pins=pins,
        n_estimators=n_estimators,
        demographic_conditioning=demographic_conditioning,
        source_keys=source_keys,
        keys=keys,
        implementations=implementations,
        loaded=loaded,
        live=live,
        property_income=property_income,
        legacy_financial_population=None
        if property_graph is None
        else legacy_population,
    )
    return result if return_values else manifest
