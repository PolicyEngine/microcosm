"""Extend an issued postclone financial run with two PUF55 routes.

The existing financial run is an explicit prerequisite. This host compiles and
executes the extension once; it does not claim that the upstream run's entire
lifetime used one execution. No donor or FILTER issuer is run in preparation.
Required replay is a separate explicit call over the same source paths/store.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, replace
from pathlib import Path

from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.frame import US_SCHEMA
from microcosm.graph import (
    ContentStore,
    KernelRegistry,
    artifact_edges,
    codecs,
    compile_graph,
    run_graph,
)
from microcosm.graph.executor import _all_node_keys, _source_paths_and_keys
from microcosm.graph.serialize import graph_to_json

from . import graph_puf55_route_attachment as attach

require = attach.require
financial, recipient, canonical = attach.financial, attach.recipient, attach.canonical
physical, codec, population_ops = attach.physical, attach.codec, attach.population_ops
RUN_PROTOCOL = "microcosm.us.survey-puf55-checked-run.v1"
_ISSUED_RUNS = {}


@dataclass(frozen=True)
class SurveyPuf55Run:
    """Actual result handle; constructor/copies do not issue run authority."""

    financial_run: financial.AtomicSurveyFinancialRunValues
    population: population_ops.Population
    manifest: object
    compiled: object
    store: ContentStore
    kernels: KernelRegistry
    sources: tuple[tuple[str, Path], ...]
    receipt: bytes

    def checked_view(self):
        """Recheck this retained handle without fitting or graph execution."""
        return check_survey_puf55_run(self)


@dataclass(frozen=True)
class CheckedSurveyPuf55Run:
    """Descriptive values; only the original run retains in-process authority.

    A downstream owner must check that run immediately before consumption and
    after its last relevant I/O, before returning or exporting a successor.
    Reconstructing this view from JSON or copying it cannot authorize a run.
    Neither this view nor the run grants calibration or release admission.
    """

    payload: bytes
    digest: str
    population: population_ops.Population


@dataclass(frozen=True)
class _RunState:
    boundary: attach.Boundary
    population: population_ops.Population
    population_stamp: str
    expected: population_ops.Population
    expected_stamp: str
    manifest: object
    manifest_bytes: str
    manifest_populations: tuple
    receipt: bytes
    artifact_hashes: tuple


def _run_entry(run):
    entry = _ISSUED_RUNS.get(id(run))
    require(
        type(run) is SurveyPuf55Run and entry is not None and entry[0]() is run,
        "UNISSUED_PUF55_RUN",
    )
    return entry


def _forget_run(run, entry):
    if _ISSUED_RUNS.get(id(run)) is entry:
        _ISSUED_RUNS.pop(id(run))


def _run_document(state):
    """Describe checked ancestry without granting authority to portable bytes."""
    boundary = state.boundary
    return codec.encode_json(
        {
            "protocol": RUN_PROTOCOL,
            "graph_sha256": codec.sha(boundary.declaration.encode()),
            "manifest_key": state.manifest.key,
            "financial_run_sha256": codec.sha(boundary.entry[1]),
            "receipt_sha256": codec.sha(state.receipt),
            "node_count": len(boundary.compiled.order),
            "node_keys": dict(boundary.keys),
            "source_keys": dict(boundary.source_keys),
            "artifact_payload_sha256": [list(row) for row in state.artifact_hashes],
            "population_version": state.population.version,
            "population_physical_sha256": state.population_stamp,
            "source_admission_issued": False,
            "population_admission_issued": False,
            "release_eligible": False,
        }
    )


def _pure_run(run, entry):
    """Check the retained output after all external reads have completed."""
    require(_run_entry(run) is entry, "FINAL_PUF55_RUN_ISSUANCE")
    state, boundary = entry[2], entry[2].boundary
    require(
        run.financial_run is boundary.run
        and run.population is state.population
        and run.manifest is state.manifest
        and run.compiled is boundary.compiled
        and run.store is boundary.store
        and run.kernels is boundary.kernels
        and run.sources is boundary.paths
        and run.receipt == state.receipt,
        "PUF55_RUN_BINDINGS_CHANGED",
    )
    require(
        physical._population_stamp(run.population) == state.population_stamp
        and physical._population_stamp(state.expected) == state.expected_stamp,
        "PUF55_RUN_POPULATION_CHANGED",
    )
    require(
        set(run.manifest.populations)
        == set(run.manifest.mass_ledgers)
        == set(run.compiled.versions.values()),
        "PUF55_RUN_MANIFEST_ROSTER_CHANGED",
    )
    require(
        run.manifest.to_json() == state.manifest_bytes
        and _heterogeneous_manifest_seals(run.manifest, run.compiled)
        == state.manifest_populations,
        "PUF55_RUN_MANIFEST_CHANGED",
    )
    boundary.pure()
    require(
        graph_to_json(run.compiled.graph) == boundary.declaration
        and run.compiled == compile_graph(run.compiled.graph),
        "PUF55_RUN_DECLARATION_CHANGED",
    )
    physical.replay.same_replayed_population(state.expected, run.population)
    require(_run_document(state) == entry[1], "PUF55_RUN_DOCUMENT_CHANGED")
    require(_run_entry(run) is entry, "FINAL_PUF55_RUN_ISSUANCE")


def check_survey_puf55_run(run):
    """Requalify actual source/store ancestry, then seal the complete output.

    Does not fit, execute a graph, decode model pickles, or reconstruct the PUF
    donor. The actual execution already verified those exact artifact bytes.
    A failed check revokes this handle; restoring its fields cannot reissue it.
    Downstream owners must recheck before consumption and after their last
    relevant I/O before returning/exporting a successor. A successful check is
    a point-in-time claim, not a lease permitting future unchecked mutation.
    """
    entry = _run_entry(run)
    try:
        _pure_run(run, entry)
        state, boundary = entry[2], entry[2].boundary
        boundary.borrow()
        loaded = financial._artifacts(
            run.manifest,
            boundary.compiled,
            boundary.store,
            boundary.kernels,
            dict(boundary.keys),
            dict(boundary.implementations),
        )
        boundary.borrow()
        require(
            tuple(sorted((n, a, codec.sha(p)) for (n, a), p in loaded.items()))
            == state.artifact_hashes,
            "PUF55_RUN_ARTIFACT_CHANGED",
        )
        result = CheckedSurveyPuf55Run(entry[1], codec.sha(entry[1]), run.population)
        _pure_run(run, entry)
        return result
    except BaseException:
        _forget_run(run, entry)
        raise


def _issue_run(output, state):
    """Retain only a fully verified execution, with no new baseline reads."""
    identifier = id(output)
    require(identifier not in _ISSUED_RUNS, "PUF55_RUN_REISSUANCE")

    def forget(reference):
        entry = _ISSUED_RUNS.get(identifier)
        if entry is not None and entry[0] is reference:
            _ISSUED_RUNS.pop(identifier)

    entry = (weakref.ref(output, forget), _run_document(state), state)
    _ISSUED_RUNS[identifier] = entry
    try:
        _pure_run(output, entry)
    except BaseException:
        _forget_run(output, entry)
        raise
    return entry


def _registry(existing, donor):
    require(type(existing) is codecs.SourceCodecRegistry, "UPSTREAM_CODEC_REGISTRY")
    registry = codecs.SourceCodecRegistry()
    for old in (existing, donor):
        for name, loader in old.as_mapping().items():
            require(
                name not in (*registry.names(), *registry.bytes_names()),
                "CODEC_COLLISION",
            )
            registry.register(name, loader)
        for name, loader in old.as_bytes_mapping().items():
            require(
                name not in (*registry.names(), *registry.bytes_names()),
                "CODEC_COLLISION",
            )
            registry.register_bytes(name, loader)
    return registry


def _construct(
    financial_run, donor_sources, *, fixture_definition, seed, n_estimators, zero_atol
):
    """Construct actual declarations and bind actual source keys without running."""
    financial.check_atomic_survey_financial_run(financial_run)
    upstream_count = len(financial_run.compiled.order)
    require(type(seed) is int and 0 <= seed < 2**64, "SEED")
    require(type(n_estimators) is int and n_estimators > 0, "TREE_COUNT")
    qualified = recipient.values.qualify_puf55_survey_recipients(financial_run)
    donor = canonical.CanonicalPuf55DonorKernel(
        seed=seed, fixture_definition=fixture_definition
    )
    require(
        type(donor_sources) is dict
        and set(donor_sources) == {s.name for s in donor.source_refs},
        "DONOR_SOURCE_ROSTER",
    )
    donor_paths = {
        name: Path(path).resolve(strict=True) for name, path in donor_sources.items()
    }
    require(not set(donor_paths) & set(financial_run.sources), "SOURCE_NAME_COLLISION")
    boundary = attach.Boundary(
        financial_run,
        qualified,
        donor,
        donor_paths,
        seed=seed,
        n_estimators=n_estimators,
        zero_atol=zero_atol,
    )
    recipient_nodes = recipient.puf55_survey_recipient_nodes(qualified)
    nodes = (
        *financial_run.compiled.graph.nodes,
        *recipient_nodes,
        boundary.donor_node,
        boundary.nodes[0],
        *(node for route in boundary.routes for node in (*route.fits, *route.applies)),
        *boundary.nodes[1:],
    )
    graph = replace(
        financial_run.compiled.graph,
        sources=(*financial_run.compiled.graph.sources, *donor.source_refs),
        nodes=nodes,
    )
    compiled = compile_graph(graph)
    require(
        len(compiled.order) == upstream_count + 6 + 110 * len(boundary.routes),
        "NODE_COUNT",
    )
    order = compiled.order
    require(
        order.index(financial.financial_output_node(financial_run))
        < order.index(recipient.PROJECTION_NODE)
        < order.index(recipient.MATRIX_NODE)
        < order.index(attach.FILTER_NODE)
        < order.index(attach.MASK_NODE)
        < order.index(attach.ATTACH_NODE),
        "EXTENSION_ORDER",
    )
    # Private registries preserve the original issued run's retained containers.
    kernels = KernelRegistry()
    for kernel in financial_run.kernels.as_mapping().values():
        kernels.register(kernel)
    for kernel in (
        recipient.Puf55SurveyRecipientProjectionKernel(financial_run),
        recipient.Puf55SurveyRecipientMatrixKernel(financial_run),
        donor,
        attach.SurveyPuf55KeepAllKernel(boundary),
        attach.SurveyPuf55MaskKernel(boundary),
        attach.SurveyPuf55AttachKernel(boundary),
    ):
        require(kernel.ref not in kernels.refs(), "KERNEL_COLLISION")
        kernels.register(kernel)
    for kernel in (LegacyQRFTrainKernel(), LegacyQRFApplyMatrixKernel()):
        if kernel.ref not in kernels.refs():
            kernels.register(kernel)
        else:
            require(type(kernels.get(kernel.ref)) is type(kernel), "LEGACY_KERNEL_TYPE")
    store = ContentStore(
        financial_run.store.root,
        codecs=_registry(financial_run.store.codecs, donor.source_codecs),
    )
    paths, sources = _source_paths_and_keys(
        compiled, {**financial_run.sources, **donor_paths}, store
    )
    keys, implementations = _all_node_keys(compiled, kernels, sources)
    # The preserved prefix must have the actual issued keys, not merely the
    # same user-facing IDs. New donor sources are unused by the original nodes.
    state = financial._run_entry(financial_run)[2]
    require(
        all(keys[n] == k for n, k in state.keys)
        and all(implementations[n] == k for n, k in state.implementations),
        "UPSTREAM_PRODUCER_KEYS",
    )
    boundary.bind(compiled, store, kernels, paths, sources, keys, implementations)
    boundary.borrow()
    return boundary


def _heterogeneous_manifest_seals(manifest, compiled):
    """Seal every attached version of this mixed-schema run, donor included.

    The survey helper stamps through ``source._frame_identity``, which requires
    the six US_SCHEMA entity groups. This host attaches a tax_unit-only PUF
    donor CREATE, so that helper cannot seal this manifest at all. The physical
    seal is schema-aware and already seals exactly this donor frame elsewhere
    (``canonical._frame_seal``). It records link *names* but never link bodies,
    so a frame carrying link tables is refused here rather than sealed blind.
    Reconstructed versions carry default owners and anchors derived from their
    stored weights. This manifest-content seal does not recover independently
    retained original owner/anchor/history evidence.
    """
    seals = []
    for version in sorted(set(compiled.versions.values())):
        frame = manifest.population(version)
        require(not frame.links, "MANIFEST_SEAL_LINK_TABLES")
        seals.append(
            (
                version,
                physical._population_stamp(
                    population_ops.Population.from_frame(
                        frame, version, mass_ledger=manifest.mass_ledger(version)
                    )
                ),
            )
        )
    return tuple(seals)


def _check_replayed_survey_manifest(expected_manifest, actual_manifest, compiled):
    """Compare an exact upstream survey roster across store representations.

    The caller keeps the original issued manifest and its physical lifetime
    seals. These Frame/ledger views reconstruct the same default owners and
    design anchors as the former manifest stamp; they do not recover true
    execution ownership. Actual observed Populations are checked separately.
    Manifest access may materialize lazy attachments, so the host brackets this
    comparison with its retained-owner checks and keeps its final physical seal.
    """
    versions = tuple(sorted(set(compiled.versions.values())))
    wanted = set(versions)
    require(
        expected_manifest.country
        == actual_manifest.country
        == compiled.graph.country
        == "us",
        "UPSTREAM_SURVEY_COUNTRY",
    )
    require(
        bool(versions)
        and set(expected_manifest.nodes) == set(compiled.order)
        and set(compiled.order) <= set(actual_manifest.nodes)
        and set(expected_manifest.populations) == wanted
        and set(expected_manifest.mass_ledgers) == wanted
        and wanted <= set(actual_manifest.populations)
        and wanted <= set(actual_manifest.mass_ledgers),
        "UPSTREAM_SURVEY_ROSTER",
    )
    for version in versions:
        expected = expected_manifest.population(version)
        actual = actual_manifest.population(version)
        require(expected.schema == actual.schema == US_SCHEMA, "UPSTREAM_SURVEY_SCHEMA")
        physical.replay.same_replayed_population(
            population_ops.Population.from_frame(
                expected, version, mass_ledger=expected_manifest.mass_ledger(version)
            ),
            population_ops.Population.from_frame(
                actual, version, mass_ledger=actual_manifest.mass_ledger(version)
            ),
        )


def _loaded_values(boundary, manifest, loaded, node):
    """Only construct typed inputs from the already-checked real manifest."""
    record = manifest.node(node.id)
    return {
        edge.name: artifact_edges.value_from_descriptor(
            loaded[edge.producer, edge.artifact],
            record.typed_artifacts["inputs"][edge.name],
        )
        for edge in node.artifact_inputs
    }


def _reconstruct(boundary, observed, values, result):
    """Use original retained upstream and independently source-derived donor."""
    run = boundary.run
    # The original issuer retains final views per structural version. Its
    # existing nineteen-node keys/types are checked above; the complete retained
    # final views below bind the unchanged prefix, without inventing a second
    # upstream observer history or re-executing a preparation graph.
    physical.replay.same_replayed_population(
        run.financial_population, observed[recipient.PROJECTION_NODE]
    )
    physical.replay.same_replayed_population(
        run.financial_population, observed[recipient.MATRIX_NODE]
    )
    physical.replay.same_replayed_population(
        boundary.expected, observed[attach.FILTER_NODE]
    )
    donor_frame, donors, donor_seal = boundary.canonical_donor(values)
    require(
        attach._donor_seal(donor_frame, donors) == donor_seal, "DONOR_RESULT_CHANGED"
    )
    donor_node = boundary.donor_node
    # CREATE assigns every loaded column, including linkage, to this version.
    donor_population = population_ops.Population.from_frame(donor_frame, donor_node.id)
    physical.replay.same_replayed_population(donor_population, observed[donor_node.id])
    for route in boundary.routes:
        for node in route.fits:
            physical.replay.same_replayed_population(
                donor_population, observed[node.id]
            )
        for node in route.applies:
            # Ordinary fit/apply nodes own no Population cells. Compiler order
            # can put the independent mask before any apply, so compare with
            # the preceding state of this version, not an assumed fixed order.
            require(not node.outputs and node.weights is None, "CHAIN_POPULATION_WRITE")
    current = {}
    for node_id in boundary.compiled.order:
        node = boundary.compiled.graph.node(node_id)
        version = boundary.compiled.versions[node_id]
        if node_id in run.compiled.order:
            current[version] = observed[node_id]
            continue
        if node_id == donor_node.id:
            expected = donor_population
        elif node_id == attach.FILTER_NODE:
            expected = boundary.expected
        elif node_id == attach.MASK_NODE:
            expected = population_ops.patch(
                current[version], node, attach._mask_result(boundary.expected.frame)
            )
        elif node_id == attach.ATTACH_NODE:
            expected = population_ops.patch(current[version], node, result)
        else:
            expected = current[version]
        physical.replay.same_replayed_population(expected, observed[node_id])
        current[version] = expected
    return current[attach.FILTER_NODE]


def run_survey_puf55(
    financial_run,
    *,
    donor_sources,
    seed=578,
    n_estimators=100,
    zero_atol=1e-8,
    fixture_definition=None,
    resume="auto",
):
    """Run one extension over an actual issued run; no pre-run or donor injection.

    ``fixture_definition`` is the canonical owner's explicit invented-source
    route. It cannot claim packaged-source admission. A separate call with
    ``resume='require'`` must reproduce every source/producer and all values.
    """
    require(resume in ("auto", "require"), "RESUME")
    boundary = _construct(
        financial_run,
        donor_sources,
        fixture_definition=fixture_definition,
        seed=seed,
        n_estimators=n_estimators,
        zero_atol=zero_atol,
    )
    observed, stamps = {}, {}

    def observe(node_id, population):
        require(node_id not in observed, "OBSERVER_DUPLICATE")
        observed[node_id] = population
        stamps[node_id] = physical._population_stamp(population)

    manifest = run_graph(
        boundary.compiled,
        sources=dict(boundary.paths),
        store=boundary.store,
        kernels=boundary.kernels,
        resume=resume,
        _population_observer=observe,
    )
    require(tuple(observed) == boundary.compiled.order, "OBSERVER_ROSTER")
    if resume == "require":
        require(
            all(manifest.node(n).hit for n in boundary.compiled.order), "REQUIRED_HITS"
        )
    # Complete source-key/implementation/type ancestry before loading any
    # trusted model or independently invoking the replay numerical verifier.
    boundary.borrow()
    loaded = financial._artifacts(
        manifest,
        boundary.compiled,
        boundary.store,
        boundary.kernels,
        dict(boundary.keys),
        dict(boundary.implementations),
    )
    # The mixed-manifest physical baseline includes donor and survey versions
    # and is captured once. Cross-store comparison uses the still-issued
    # upstream manifest as expected, retaining its original lifetime seals.
    full_seals = _heterogeneous_manifest_seals(manifest, boundary.compiled)
    boundary.pure()
    _check_replayed_survey_manifest(
        financial_run.manifest, manifest, financial_run.compiled
    )
    boundary.pure()
    values = _loaded_values(boundary, manifest, loaded, boundary.nodes[2])
    boundary.projections(values)
    if boundary.computed is None:
        result = boundary.finalize(values)
    else:
        boundary.pure()
        result = boundary.computed[0]
    require(
        loaded[attach.ATTACH_NODE, "finalization"] == result.artifacts["finalization"],
        "FINALIZATION_ARTIFACT",
    )
    expected = _reconstruct(boundary, observed, values, result)
    population = observed[attach.ATTACH_NODE]
    physical.replay.same_replayed_population(expected, population)
    receipt = codec.encode_json(
        {
            "protocol": attach.PROTOCOL,
            "node_count": len(boundary.compiled.order),
            "profiles": [r.profile.value for r in boundary.routes],
            "financial_run_sha256": codec.sha(boundary.entry[1]),
            "manifest_key": manifest.key,
            "one_extension_execution": True,
            "upstream_already_issued": True,
            "complete_population_compared": True,
            "source_admission_issued": False,
            "population_admission_issued": False,
            "release_eligible": False,
        }
    )
    output = SurveyPuf55Run(
        financial_run,
        population,
        manifest,
        boundary.compiled,
        boundary.store,
        boundary.kernels,
        boundary.paths,
        receipt,
    )
    manifest_bytes = manifest.to_json()
    expected_stamp = physical._population_stamp(expected)
    artifact_hashes = tuple(
        sorted((n, a, codec.sha(p)) for (n, a), p in loaded.items())
    )
    # Last reads include actual source owners/code, manifest store identities,
    # and every returned Population. Nothing below the fence performs I/O.
    boundary.borrow()
    fresh = financial._artifacts(
        manifest,
        boundary.compiled,
        boundary.store,
        boundary.kernels,
        dict(boundary.keys),
        dict(boundary.implementations),
    )
    boundary.borrow()
    require(
        tuple(sorted((n, a, codec.sha(p)) for (n, a), p in fresh.items()))
        == artifact_hashes,
        "LATE_ARTIFACT_CHANGED",
    )
    boundary.pure()
    require(
        output.financial_run is financial_run
        and output.population is population
        and output.manifest is manifest
        and output.compiled is boundary.compiled
        and output.store is boundary.store
        and output.kernels is boundary.kernels
        and output.sources == boundary.paths
        and output.receipt == receipt
        and manifest.to_json() == manifest_bytes
        and _heterogeneous_manifest_seals(manifest, boundary.compiled) == full_seals
        and physical._population_stamp(expected) == expected_stamp
        and all(physical._population_stamp(observed[n]) == stamps[n] for n in observed),
        "FINAL_OUTPUT_CHANGED",
    )
    physical.replay.same_replayed_population(expected, population)
    state = _RunState(
        boundary,
        population,
        stamps[attach.ATTACH_NODE],
        expected,
        expected_stamp,
        manifest,
        manifest_bytes,
        full_seals,
        receipt,
        artifact_hashes,
    )
    # Baselines were captured before the preceding last-I/O fence. The entry
    # retains the final output, not all 245 executor-observed snapshots.
    try:
        entry = _issue_run(output, state)
        _pure_run(output, entry)
    except BaseException:
        candidate = _ISSUED_RUNS.get(id(output))
        if candidate is not None and candidate[0]() is output:
            _forget_run(output, candidate)
        raise
    return output
