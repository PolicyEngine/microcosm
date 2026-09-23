"""Opt-in continuation of a genuine completed survey enrichment owner.

The first stage supplies real terminal bytes and a complete receiving Population.
This boundary keeps that owner live; it cannot create authority from a Frame.
"""

from __future__ import annotations

import json
import sys
import weakref
from dataclasses import replace
from types import FunctionType, SimpleNamespace

import pandas as pd

from microcosm.graph import (
    ArtifactInput,
    ContentStore,
    KernelBase,
    KernelRegistry,
    KernelResult,
    SourceRef,
    StructuralDelta,
    codecs,
    compile_graph,
    executor,
    run_graph,
    source_hash,
)
from microcosm.graph import (
    population as population_ops,
)
from microcosm.graph.serialize import graph_to_json

from . import graph_current_survey_other_disability_completion as fragment
from . import graph_us_survey_enrichment as host

codec, values = fragment.codec, fragment.values
parent, physical, original_host = host.parent, host.physical, host.original_host
require = values.require
PROTOCOL = "microcosm.us.other-disability-host.v1"
DISPATCH_POLICY = "inherit_exact_kernel_objects_and_cache_keys_refuse_prefix_execution"


def validate_options(enabled, seed):
    require(type(enabled) is bool, "HOST_OPTION")
    require(
        (enabled and type(seed) is int and seed >= 0) or (not enabled and seed is None),
        "HOST_SEED",
    )


def _continuation_store(predecessor_store):
    """Install the new source decoder without changing the retained registry."""
    added = codecs.SourceCodecRegistry()
    added.register("frame-store", codecs.load_frame_store)
    registry = parent._registry(predecessor_store.codecs, added)
    return ContentStore(predecessor_store.root, codecs=registry)


def _require_reusable_kernel(incumbent, kernel):
    """Only the two stateless QRF implementations may serve both stages."""
    require(
        type(incumbent) is type(kernel)
        and type(kernel)
        in (parent.LegacyQRFTrainKernel, parent.LegacyQRFApplyMatrixKernel)
        and incumbent.implementation_hash() == kernel.implementation_hash(),
        "HOST_KERNEL_COLLISION",
    )


def _live():
    seal = values.source._function_seal
    members = []
    for name, item in vars(sys.modules[__name__]).items():
        if type(item) is FunctionType:
            members.append((name, seal(item)))
        elif isinstance(item, type) and item.__module__ == __name__:
            members.append((name, item))
            for member, function in vars(item).items():
                if isinstance(function, staticmethod):
                    function = function.__func__
                if type(function) is FunctionType:
                    members.append((name, member, seal(function)))
    return tuple(members), PROTOCOL, DISPATCH_POLICY, fragment._live(), host._live()


class _GuardedKernel(KernelBase):
    """Keep inherited cache identities while forbidding inherited cold dispatch."""

    def __init__(self, kernel, boundary, inherited):
        self.kernel, self.boundary, self.inherited = kernel, boundary, inherited
        self.ref, self.capabilities = kernel.ref, kernel.capabilities
        self.identity = (kernel, boundary, inherited, self.ref, self.capabilities)

    def implementation_hash(self):
        require(
            self.identity
            == (
                self.kernel,
                self.boundary,
                self.inherited,
                self.ref,
                self.capabilities,
            ),
            "HOST_KERNEL_CHANGED",
        )
        # The wrapper adds only a refusal. Existing implementation identities
        # remain exact; its own bytes/live state are bound by the host owner.
        return self.kernel.implementation_hash()

    def state(self):
        return (
            self.kernel,
            self.boundary,
            self.inherited,
            self.ref,
            self.capabilities,
            self.identity,
        )

    def run(self, context):
        require(context.node.id not in self.inherited, "INHERITED_COLD_DISPATCH")
        self.boundary.context(context)
        result = self.kernel.run(context)
        stamp = fragment._result_stamp(result)
        self.boundary.pure()
        require(fragment._result_stamp(result) == stamp, "HOST_KERNEL_RESULT_CHANGED")
        return result


class ContinuationBoundary:
    """Concrete retained host state, not a source issuer or forwarding proxy."""

    _original_after = staticmethod(host._original_after)

    def __init__(self, predecessor, *, seed, n_estimators, original_application_seed):
        self.predecessor = predecessor
        self.predecessor_view = host.check_survey_enrichment_run(predecessor)
        self.predecessor_entry = host._ISSUED.get(id(predecessor))
        require(self.predecessor_entry is not None, "HOST_PREDECESSOR_UNISSUED")
        previous = self.predecessor_entry[1]
        require(type(previous) is host.Boundary, "HOST_PREDECESSOR_TYPE")
        require(previous.original is None, "HOST_ORIGINAL_ALREADY_PLACED")
        self.run = predecessor.parent_run
        self.parent_entry = parent._run_entry(self.run)
        self.preparation = self.run.financial_run.prefix.preparation
        self.parent_view = parent.check_survey_puf55_run(self.run)
        terminal = previous.receiving_terminal
        require(
            terminal.population == predecessor.population.version
            and len(terminal.artifact_outputs) == 1,
            "HOST_TERMINAL",
        )
        output = terminal.artifact_outputs[0]
        record = predecessor.manifest.node(terminal.id)
        payload = predecessor.store.load_bytes(record.opaque_artifacts[output.name])
        self.terminal_binding = (terminal, record.key, output, payload)
        self.fragment = fragment._CompletionBoundary(
            self.preparation,
            predecessor.population.frame,
            receiving_version=predecessor.population.version,
            after=ArtifactInput("after", terminal.id, output.name, output.type),
            after_payload=payload,
            seed=seed,
            n_estimators=n_estimators,
        )
        self.seed, self.n_estimators = seed, n_estimators
        self.configuration = seed, n_estimators, original_application_seed
        self.receiving_terminal = self.fragment.nodes[-1]
        require(
            self.receiving_terminal.id == fragment.ATTACH_NODE
            and self.receiving_terminal.population is not None,
            "HOST_RECEIVING_TERMINAL",
        )
        self.original_application_seed = original_application_seed
        self.original = (
            None
            if original_application_seed is None
            else original_host.Binding(
                self,
                terminal=self.receiving_terminal,
                application_seed=original_application_seed,
            )
        )
        self.nodes = (
            *self.fragment.nodes,
            *(self.original.nodes if self.original is not None else ()),
        )
        self.declaration = tuple(self.nodes)
        self.predecessor_objects = (
            predecessor.parent_run,
            predecessor.population,
            predecessor.manifest,
            predecessor.compiled,
            predecessor.store,
            predecessor.kernels,
            predecessor.sources,
        )
        self.implementation = source_hash(sys.modules[__name__])
        self.live = _live()
        self.compiled = self.store = self.kernels = None
        self.paths = self.source_keys = self.keys = self.implementations = None
        self.pure()

    def pure(self):
        require(_live() == self.live, "HOST_IMPLEMENTATION_CHANGED")
        predecessor = self.predecessor
        require(
            host._ISSUED.get(id(predecessor)) is self.predecessor_entry
            and self.predecessor_entry[0]() is predecessor,
            "HOST_PREDECESSOR_IDENTITY",
        )
        self.predecessor_entry[1].pure()
        require(
            host._run_seal(predecessor) == self.predecessor_entry[2]
            and all(
                a is b
                for a, b in zip(
                    self.predecessor_objects,
                    (
                        predecessor.parent_run,
                        predecessor.population,
                        predecessor.manifest,
                        predecessor.compiled,
                        predecessor.store,
                        predecessor.kernels,
                        predecessor.sources,
                    ),
                    strict=True,
                )
            )
            and self.run is predecessor.parent_run
            and parent._run_entry(self.run) is self.parent_entry
            and self.preparation is self.run.financial_run.prefix.preparation
            and self.fragment.preparation is self.preparation
            and self.fragment.receiving is predecessor.population.frame
            and self.configuration
            == (self.seed, self.n_estimators, self.original_application_seed)
            and self.fragment.configuration["seed"] == self.seed
            and self.fragment.configuration["n_estimators"] == self.n_estimators,
            "HOST_PREDECESSOR_CHANGED",
        )
        terminal, key, output, payload = self.terminal_binding
        require(
            terminal == self.predecessor_entry[1].receiving_terminal
            and predecessor.manifest.node(terminal.id).key == key
            and output == terminal.artifact_outputs[0]
            and self.fragment.configuration["after_payload"] == payload
            and self.fragment.configuration["receiving_version"]
            == predecessor.population.version
            and self.fragment.configuration["after"]
            == ArtifactInput("after", terminal.id, output.name, output.type),
            "HOST_TERMINAL_CHANGED",
        )
        self.fragment.pure()
        require(
            self.receiving_terminal == self.fragment.nodes[-1]
            and self.receiving_terminal.id == fragment.ATTACH_NODE
            and self.receiving_terminal.population is not None
            and self.declaration
            == self.nodes
            == (
                *self.fragment.nodes,
                *(self.original.nodes if self.original is not None else ()),
            ),
            "HOST_DECLARATIONS_CHANGED",
        )
        if self.original_application_seed is None:
            require(self.original is None, "HOST_ORIGINAL_DISABLED")
        else:
            require(
                type(self.original) is original_host.Binding
                and self.original.host is self
                and self.original.terminal == self.receiving_terminal
                and self.original.seeds["original_application_seed"]
                == self.original_application_seed,
                "HOST_ORIGINAL_BINDING",
            )
            self.original.pure()
        if self.compiled is not None:
            require(
                all(
                    a is b
                    for a, b in zip(
                        self.bound_objects,
                        (self.compiled, self.store, self.kernels, self.store.codecs),
                        strict=True,
                    )
                )
                and tuple(self.kernels.as_mapping().items()) == self.kernel_items
                and tuple(
                    (ref, kernel.state())
                    for ref, kernel in self.kernels.as_mapping().items()
                )
                == self.dispatch_states
                and all(
                    self.kernels.as_mapping()[ref].kernel is kernel
                    for ref, kernel in predecessor.kernels.as_mapping().items()
                )
                and tuple(self.store.codecs.as_mapping().items()) == self.codec_items
                and tuple(self.store.codecs.as_bytes_mapping().items())
                == self.bytes_codec_items
                and self.compiled == compile_graph(self.compiled.graph)
                and graph_to_json(self.compiled.graph) == self.graph_json,
                "HOST_BOUND_REGISTRY_CHANGED",
            )

    def borrow(self):
        require(
            host.check_survey_enrichment_run(self.predecessor).payload
            == self.predecessor_view.payload,
            "HOST_PREDECESSOR_REVALIDATION",
        )
        self.fragment.validate()
        require(
            source_hash(sys.modules[__name__]) == self.implementation,
            "HOST_SOURCE_CHANGED",
        )
        if self.compiled is not None:
            paths, source_keys = executor._source_paths_and_keys(
                self.compiled, dict(self.paths), self.store
            )
            keys, implementations = executor._all_node_keys(
                self.compiled, self.kernels, source_keys
            )
            require(
                tuple(sorted(paths.items())) == self.paths
                and tuple(sorted(source_keys.items())) == self.source_keys
                and tuple(sorted(keys.items())) == self.keys
                and tuple(sorted(implementations.items())) == self.implementations,
                "HOST_SOURCE_OR_IMPLEMENTATION_CHANGED",
            )
        self.pure()

    def requalify(self):
        self.predecessor_entry[1].requalify()
        self.fragment.validate()
        if self.original is not None:
            self.original.requalify()
        self.pure()

    def context(self, context):
        self.pure()
        require(
            context.node in self.nodes
            and context.params == context.node.params
            and set(context.artifacts) == {e.name for e in context.node.artifact_inputs}
            and set(context.sources) == set(context.node.sources)
            and all(context.sources[n] == dict(self.paths)[n] for n in context.sources),
            "HOST_CONTEXT_DECLARATION",
        )
        for edge in context.node.artifact_inputs:
            value = host.values.predictors.host.shared.artifact(
                context, edge.name, edge.type
            )
            require(
                value.producer_key == dict(self.keys)[edge.producer],
                "HOST_ARTIFACT_PRODUCER",
            )
        self.pure()


def _construct(predecessor, *, seed, n_estimators, original_application_seed):
    boundary = ContinuationBoundary(
        predecessor,
        seed=seed,
        n_estimators=n_estimators,
        original_application_seed=original_application_seed,
    )
    graph = predecessor.compiled.graph
    require(
        fragment.SOURCE_NAME not in {s.name for s in graph.sources},
        "HOST_SOURCE_COLLISION",
    )
    compiled = compile_graph(
        replace(
            graph,
            sources=(*graph.sources, SourceRef(fragment.SOURCE_NAME, "frame-store")),
            nodes=(*graph.nodes, *boundary.nodes),
        )
    )
    inherited = frozenset(predecessor.compiled.order)
    registry = KernelRegistry()
    for kernel in predecessor.kernels.as_mapping().values():
        registry.register(_GuardedKernel(kernel, boundary, inherited))
    added = list(
        fragment.other_disability_kernel_registry(boundary.fragment)
        .as_mapping()
        .values()
    )
    if boundary.original is not None:
        added.extend(original_host.kernels(boundary.original))
        for cls in (
            original_host.recipient_graph.Puf55SurveyRecipientProjectionKernel,
            original_host.recipient_graph.Puf55SurveyRecipientMatrixKernel,
        ):
            incumbent = registry.as_mapping()[cls.ref].kernel
            require(
                type(incumbent) is cls
                and incumbent._financial_run is boundary.run.financial_run,
                "ORIGINAL_RECIPIENT_KERNEL",
            )
    for kernel in added:
        if kernel.ref in registry.refs():
            incumbent = registry.as_mapping()[kernel.ref].kernel
            _require_reusable_kernel(incumbent, kernel)
        else:
            registry.register(_GuardedKernel(kernel, boundary, inherited))
    store = _continuation_store(predecessor.store)
    q = boundary.fragment.qualified
    paths = dict(predecessor.sources)
    paths[fragment.SOURCE_NAME] = store.put_frame(
        codec.sha(q.donor_projection), q.source_frame
    )
    paths, source_keys = executor._source_paths_and_keys(compiled, paths, store)
    keys, implementations = executor._all_node_keys(compiled, registry, source_keys)
    require(
        all(
            compiled.graph.node(n) == predecessor.compiled.graph.node(n)
            and keys[n] == predecessor.manifest.node(n).key
            and implementations[n] == predecessor.manifest.node(n).kernel_impl_hash
            for n in predecessor.compiled.order
        ),
        "HOST_PREFIX_KEY_CHANGED",
    )
    boundary.compiled, boundary.store, boundary.kernels = compiled, store, registry
    boundary.paths = tuple(sorted(paths.items()))
    boundary.source_keys = tuple(sorted(source_keys.items()))
    boundary.keys = tuple(sorted(keys.items()))
    boundary.implementations = tuple(sorted(implementations.items()))
    boundary.bound_objects = compiled, store, registry, store.codecs
    boundary.kernel_items = tuple(registry.as_mapping().items())
    boundary.dispatch_states = tuple(
        (ref, kernel.state()) for ref, kernel in registry.as_mapping().items()
    )
    boundary.codec_items = tuple(store.codecs.as_mapping().items())
    boundary.bytes_codec_items = tuple(store.codecs.as_bytes_mapping().items())
    boundary.graph_json = graph_to_json(compiled.graph)
    boundary.borrow()
    return boundary


def _result(boundary, node, artifacts):
    """Independent reconstruction from qualified values and authenticated artifacts."""
    q = boundary.fragment.qualified
    if node.id == fragment.SOURCE_NODE:
        return KernelResult(
            frame=q.source_frame, artifacts={"donor_projection": q.donor_projection}
        )
    if node.id == fragment.COLUMNS_NODE:
        return KernelResult(
            columns={
                ("person", c): q.donor_columns[c].copy(deep=True)
                for c in q.donor_columns
            }
        )
    if node.id == fragment.DONOR_NODE:
        return KernelResult(keep=q.donor_columns[values.ELIGIBLE].copy(deep=True))
    if node.id == fragment.MATRIX_NODE:
        return KernelResult(
            artifacts={
                "recipient_projection": q.recipient_projection,
                **({} if q.matrix is None else {"matrix": q.matrix}),
            }
        )
    context = SimpleNamespace(node=node, artifacts=artifacts)
    draws, model = fragment.read_draws(q, context, seed=boundary.seed)
    columns, payload = fragment.attachment_payload(
        q, boundary.fragment.receiving, draws, model
    )
    version = codec.encode_json(
        {
            "protocol": values.PROTOCOL,
            "selection": "keep_all",
            "receiving_sha256": boundary.fragment.receiving_seal,
            "attachment_sha256": codec.sha(payload),
        }
    )
    if node.id == fragment.VERSION_NODE:
        return KernelResult(
            keep=pd.Series(
                True,
                index=pd.Index(
                    boundary.fragment.receiving.person.person_id, name="person_id"
                ),
                dtype="bool",
            ),
            artifacts={"version": version},
        )
    require(
        node.id == fragment.ATTACH_NODE and artifacts["version"].payload == version,
        "HOST_VERSION_BINDING",
    )
    return KernelResult(columns=columns, artifacts={"attachment": payload})


def _materialize_filter_result(incoming, node, result):
    """Independently select the full Frame before structural population patching."""
    require(
        node.id in (fragment.DONOR_NODE, fragment.VERSION_NODE)
        and node.structural is StructuralDelta.FILTER
        and node.base == incoming.version
        and result.frame is None,
        "HOST_FILTER_DECLARATION",
    )
    # Match the executor's typed person-ID mask contract, including nullable
    # boolean masks without nulls. Selection remains independent of its output.
    executor._validate_filter_mask(node, result.keep, incoming)
    frame = incoming.frame
    person = frame.schema.person_entity
    identifier = frame.schema.entity_id_column(person)
    ids = pd.Index(frame.table(person)[identifier].to_numpy(copy=True), name=identifier)
    selected = frame.select(result.keep.reindex(ids).to_numpy(dtype=bool, copy=True))
    return replace(result, frame=selected, keep=None)


def _verify_model(boundary, loaded, donor):
    q = boundary.fragment.qualified
    if q.matrix is None:
        require(donor is None, "HOST_UNEXPECTED_DONOR")
        return
    require(donor is not None, "HOST_MISSING_DONOR")
    first = boundary.compiled.graph.node(fragment.FIT_PREFIX + ".000")
    frame = codec.model_frame(
        SimpleNamespace(weights={"person": donor.frame.resolve_weights("person")}),
        first.inputs[0],
        donor.frame.person,
    )
    model = fragment.qrf.RegimeGatedQRF(
        seed=boundary.seed,
        n_estimators=boundary.n_estimators,
        zero_atol=0,
        max_samples_leaf=None,
    )
    before = fragment.qrf_target.LegacyQRFTrainingState.from_chain(
        model.start_chain(
            frame, list(values.FEATURES), [values.TARGET], weights="design"
        )
    )
    payload = loaded[first.id, "model"]
    packet, after = codec.read_training(loaded[first.id, "training_state"])
    fitted = fragment.qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
        payload, expected_sha256=codec.sha(payload)
    )
    history = [
        {
            "target": values.TARGET,
            "sha256": codec.sha(payload),
            "training_id": fitted.training_id,
        }
    ]
    require(
        fitted.training_state == before
        and fitted.next_training_state == after
        and fitted.donor_sha256
        == fragment.qrf_target._consumed_values_sha256(
            frame.person, (*values.FEATURES, values.TARGET)
        )
        and packet["models"] == history
        and fragment.decode_matrix_apply_state(
            loaded[fragment.APPLY_PREFIX + ".000", "apply_state"]
        )["application"]["models"]
        == history,
        "HOST_TRAINING_DONOR_OR_HISTORY",
    )


def _compare_manifest_population(population, manifest, version):
    """Compare the Frame/ledger surface without inventing execution owners."""
    physical.replay.same_replayed_population(
        population_ops.Population.from_frame(
            population.frame, version, mass_ledger=population.mass_ledger
        ),
        population_ops.Population.from_frame(
            manifest.population(version),
            version,
            mass_ledger=manifest.mass_ledger(version),
        ),
    )


def _compare_predecessor_populations(predecessor, observed, receiving_node):
    # The retained receiving owner has full execution context. Check that
    # context before comparing the manifest's intentionally narrower surface.
    physical.replay.same_replayed_population(
        predecessor.population, observed[receiving_node]
    )
    for version in predecessor.manifest.populations:
        inherited_terminal = next(
            observed[n]
            for n in reversed(predecessor.compiled.order)
            if predecessor.compiled.versions[n] == version
        )
        _compare_manifest_population(inherited_terminal, predecessor.manifest, version)


def run_continuation(
    predecessor, *, seed, n_estimators, resume, original_application_seed
):
    require(resume in ("auto", "require"), "HOST_RESUME")
    boundary = _construct(
        predecessor,
        seed=seed,
        n_estimators=n_estimators,
        original_application_seed=original_application_seed,
    )
    observed, stamps = {}, {}

    def observe(node_id, population):
        require(node_id not in observed, "HOST_OBSERVER_DUPLICATE")
        observed[node_id] = population
        stamps[node_id] = physical._population_stamp(population)
        if boundary.original is not None:
            boundary.original.observe_terminal(node_id, population)

    manifest = run_graph(
        boundary.compiled,
        sources=dict(boundary.paths),
        store=boundary.store,
        kernels=boundary.kernels,
        resume=resume,
        _population_observer=observe,
    )
    require(tuple(observed) == boundary.compiled.order, "HOST_OBSERVER_ROSTER")
    require(
        all(manifest.node(n).hit for n in predecessor.compiled.order),
        "HOST_PREFIX_MISS",
    )
    if resume == "require":
        require(all(n.hit for n in manifest.nodes.values()), "HOST_REQUIRED_MISS")
    boundary.borrow()
    loaded = parent.financial._artifacts(
        manifest,
        boundary.compiled,
        boundary.store,
        boundary.kernels,
        dict(boundary.keys),
        dict(boundary.implementations),
    )
    current, donor = {}, None
    fragment_ids = {n.id for n in boundary.fragment.nodes}
    original_ids = (
        set()
        if boundary.original is None
        else {
            boundary.original.source_version.id,
            *(n.id for n in boundary.original.placement_nodes),
        }
    )
    if boundary.original is not None:
        boundary.original.verify_sources(manifest, loaded)
    for node_id in boundary.compiled.order:
        node = boundary.compiled.graph.node(node_id)
        version = boundary.compiled.versions[node_id]
        if node_id in predecessor.compiled.order:
            old = predecessor.manifest.node(node_id)
            for name, key in old.opaque_artifacts.items():
                require(
                    loaded[node_id, name] == predecessor.store.load_bytes(key),
                    "HOST_PREFIX_ARTIFACT_CHANGED",
                )
            current[version] = observed[node_id]
            continue
        artifacts = parent._loaded_values(boundary, manifest, loaded, node)
        persisted = {a.name: loaded[node_id, a.name] for a in node.artifact_outputs}
        if node_id in original_ids:
            incoming = current[
                node.base if node.structural is StructuralDelta.FILTER else version
            ]
            expected = boundary.original.reconstruct(
                node, incoming, artifacts, persisted
            )
        elif node_id in fragment_ids and not node_id.startswith(
            (fragment.FIT_PREFIX + ".", fragment.APPLY_PREFIX + ".")
        ):
            result = _result(boundary, node, artifacts)
            require(persisted == result.artifacts, "HOST_RESULT_ARTIFACT")
            if node.structural is StructuralDelta.FILTER:
                result = _materialize_filter_result(current[node.base], node, result)
            expected = (
                population_ops.Population.from_frame(result.frame, node.id)
                if node.structural is StructuralDelta.CREATE
                else population_ops.patch(
                    current[
                        node.base
                        if node.structural is StructuralDelta.FILTER
                        else version
                    ],
                    node,
                    result,
                )
            )
            if node_id == fragment.DONOR_NODE:
                donor = expected
        else:
            expected = current[version]
        physical.replay.same_replayed_population(expected, observed[node_id])
        current[version] = expected
    _compare_predecessor_populations(
        predecessor, observed, boundary.terminal_binding[0].id
    )
    _verify_model(boundary, loaded, donor)
    for version, population in current.items():
        _compare_manifest_population(population, manifest, version)
    attachment = codec.decode_json(loaded[fragment.ATTACH_NODE, "attachment"])
    receipt = json.loads(predecessor.receipt)
    receipt.update(
        manifest_key=manifest.key,
        node_count=len(boundary.compiled.order),
        complete_population_compared=True,
        inherited_enrichment={
            "receipt_sha256": boundary.predecessor_view.digest,
            "terminal": boundary.terminal_binding[0].id,
            "scope": "Source and fragment artifacts from the checked predecessor; no inherited final-candidate quantity claim.",
            "spm_and_amount_evidence_scope": "Unchanged source/projection/attachment artifacts; their source-defined coordinates are not disability outputs.",
        },
        other_disability_completion={
            "enabled": True,
            "protocol": PROTOCOL,
            "seed": seed,
            "n_estimators": n_estimators,
            "full_original_donors": True,
            "projection_sha256": codec.sha(
                boundary.fragment.qualified.donor_projection
            ),
            "recipient_sha256": codec.sha(
                boundary.fragment.qualified.recipient_projection
            ),
            "attachment_sha256": codec.sha(loaded[fragment.ATTACH_NODE, "attachment"]),
            "source_knownness_preserved": True,
            "reporting_age_minimum": values.observed.REPORTING_AGE,
            "under15_completed_with_zero": False,
            "scientific_qualification": "pending",
            "source_evidence": boundary.fragment.qualified.evidence,
            "attachment_receipt": attachment["receipt"],
            "model_binding": attachment["model_binding"],
            "execution_policy": {
                "policy": DISPATCH_POLICY,
                "host_implementation_sha256": boundary.implementation,
                "inherited_kernel_objects_retained": True,
                "inherited_dispatch_forbidden": True,
                "node_id_scope": list(predecessor.compiled.order),
                "computation_identity": "Delegated original kernel implementation; the separate host policy only refuses inherited execution.",
            },
        },
        release_eligible=False,
    )
    if boundary.original is not None:
        receipt["original_puf_development"] = {
            "fit_seed": boundary.original.seeds["clone_one_seed"],
            "application_seed": original_application_seed,
            "receiving_terminal": boundary.receiving_terminal.id,
            "receiving_version": boundary.receiving_terminal.population,
            "placement_sha256": codec.sha(
                loaded[original_host.fragment.ATTACH_NODE, "placement"]
            ),
            "scientific_qualification": "pending",
        }
    terminal = (
        original_host.fragment.ATTACH_NODE
        if boundary.original is not None
        else fragment.ATTACH_NODE
    )
    output = host.SurveyEnrichmentRun(
        boundary.run,
        observed[terminal],
        manifest,
        boundary.compiled,
        boundary.store,
        boundary.kernels,
        boundary.paths,
        codec.encode_json(receipt),
    )
    objects = (
        output.parent_run,
        output.population,
        output.manifest,
        output.compiled,
        output.store,
        output.kernels,
        output.sources,
    )
    stamp = host._run_seal(output)
    hashes = tuple(sorted((n, a, codec.sha(p)) for (n, a), p in loaded.items()))
    boundary.requalify()
    boundary.borrow()
    fresh = parent.financial._artifacts(
        manifest,
        boundary.compiled,
        boundary.store,
        boundary.kernels,
        dict(boundary.keys),
        dict(boundary.implementations),
    )
    boundary.borrow()
    boundary.pure()
    require(
        tuple(sorted((n, a, codec.sha(p)) for (n, a), p in fresh.items())) == hashes
        and host._run_seal(output) == stamp
        and all(physical._population_stamp(observed[n]) == stamps[n] for n in observed),
        "HOST_FINAL_OUTPUT_CHANGED",
    )
    ident = id(output)
    require(ident not in host._ISSUED, "HOST_RUN_ALREADY_ISSUED")

    def forget(ref):
        old = host._ISSUED.get(ident)
        if old is not None and old[0] is ref:
            del host._ISSUED[ident]

    host._ISSUED[ident] = (
        weakref.ref(output, forget),
        boundary,
        stamp,
        objects,
        hashes,
    )
    return output
