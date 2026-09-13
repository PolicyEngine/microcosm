"""Fixed US post-PUF host for source-qualified survey enrichment fragments.

The owning boundary retains a real checked PUF run and authenticated source
projection. Models use separate original-design donor branches; attachment
adds the declared amount and coverage families to the existing receiving frame.
"""

from __future__ import annotations

import sys
import weakref
from dataclasses import dataclass, replace
from types import FunctionType, SimpleNamespace

import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input, qrf, qrf_target
from microcosm.fit.graph_legacy_apply_matrix import (
    MATRIX_APPLY_STATE_TYPE,
    decode_matrix_apply_state,
)
from microcosm.fit.graph_legacy_qrf import (
    legacy_qrf_apply_matrix_nodes,
    legacy_qrf_train_nodes,
)
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    ContentStore,
    Determinism,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    StructuralDelta,
    codecs,
    compile_graph,
    run_graph,
    source_hash,
)
from microcosm.graph import population as population_ops
from microcosm.graph.executor import _all_node_keys, _source_paths_and_keys
from microcosm.graph.serialize import graph_to_json

from . import current_survey_amounts as values
from . import graph_current_survey_health as health_graph
from . import graph_current_survey_predictors as predictor_graph

parent = values.parent_host
physical = values.physical
require = values.require
PROJECTION_NODE = "survey_amounts.source_projection"
ATTACH_NODE = "survey_amounts.attach"
PROJECTION_TYPE = ArtifactType("microcosm.us.current_survey_amount_projection", 1)
ATTACHMENT_TYPE = ArtifactType("microcosm.us.current_survey_amount_attachment", 1)
_ISSUED = {}


def _live():
    result = []
    for module in (
        sys.modules[__name__],
        values,
        values.unemployment,
        health_graph,
        health_graph.health,
        health_graph.source,
    ):
        for name, item in vars(module).items():
            if type(item) is FunctionType:
                result.append(
                    (
                        module.__name__,
                        name,
                        values.predictors.source._function_seal(item),
                    )
                )
            elif isinstance(item, type) and item.__module__ == module.__name__:
                result.append((module.__name__, name, item))
                for member, function in vars(item).items():
                    if isinstance(function, (classmethod, staticmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if type(function) is FunctionType:
                        result.append(
                            (
                                module.__name__,
                                name,
                                member,
                                values.predictors.source._function_seal(function),
                            )
                        )
    result.append(
        (
            "amount_configuration",
            values.SEED,
            tuple((g.key, g.fields, g.targets) for g in values.GROUPS),
            values.UC_REPORT_COLUMNS,
            values.unemployment.PROTOCOL,
            values.unemployment.READ_COLUMNS,
            codec.encode_json(values.unemployment.DICTIONARY),
        )
    )
    result.append(
        (
            "health_configuration",
            health_graph.health.PROTOCOL,
            tuple(
                (f.output, f.asec, f.acs, f.acs_gap) for f in health_graph.health.FIELDS
            ),
            health_graph.health.RAW_COLUMNS,
            health_graph.health.SOURCE_PREFIX,
            health_graph.source.ASEC_COLUMNS,
            health_graph.source.ACS_COLUMNS,
        )
    )
    return tuple(result)


def _amount_edge():
    return ArtifactInput(
        "amount_attachment", ATTACH_NODE, "attachment", ATTACHMENT_TYPE
    )


def _upstream_edge():
    return ArtifactInput(
        "puf_finalization",
        parent.attach.ATTACH_NODE,
        "finalization",
        parent.attach.FINALIZATION_TYPE,
    )


def _projection_edge():
    return ArtifactInput("projection", PROJECTION_NODE, "projection", PROJECTION_TYPE)


def _ids(group):
    base = "survey_amounts." + group.spec.key
    return base + ".donor", base + ".columns", base + ".fit", base + ".apply"


def amount_nodes(qualified, receiving, *, parent_digest, n_estimators):
    require(
        type(n_estimators) is int and n_estimators > 0 and codec._hash(parent_digest),
        "NODE_PARAMETERS",
    )
    params = {
        "projection_sha256": codec.sha(qualified.projection),
        "parent_sha256": parent_digest,
        "n_estimators": n_estimators,
        "groups": tuple(g.spec.key for g in qualified.groups),
        "protocol": values.PROTOCOL,
    }
    nodes = [
        Node(
            PROJECTION_NODE,
            CurrentSurveyAmountProjectionKernel.ref,
            population=parent.attach.FILTER_NODE,
            inputs=predictor_graph._inputs(receiving),
            params=params,
            artifact_inputs=(_upstream_edge(),),
            artifact_outputs=(
                ArtifactOutput("projection", PROJECTION_TYPE),
                *(
                    ArtifactOutput(
                        g.spec.key + "_matrix", model_input.RECIPIENT_MATRIX_TYPE
                    )
                    for g in qualified.groups
                ),
            ),
            description="Qualify original current survey amount and reporting-universe evidence; preserve source unknowns.",
        )
    ]
    attach_edges = [_projection_edge(), _upstream_edge()]
    for group in qualified.groups:
        donor, columns, fit_prefix, apply_prefix = _ids(group)
        group_params = {**params, "group": group.spec.key}
        matrix_edge = ArtifactInput(
            group.spec.key + "_matrix",
            PROJECTION_NODE,
            group.spec.key + "_matrix",
            model_input.RECIPIENT_MATRIX_TYPE,
        )
        nodes.extend(
            (
                Node(
                    donor,
                    CurrentSurveyAmountDonorKernel.ref,
                    base=values.predictors.host.survey_graph.CREATE_NODE,
                    structural=StructuralDelta.FILTER,
                    mass="free",
                    inputs=predictor_graph._inputs(qualified.source_frame),
                    params=group_params,
                    artifact_inputs=(_projection_edge(),),
                    description="Select source ASEC persons with jointly known targets; retain original design weights before allocation and clones.",
                ),
                Node(
                    columns,
                    CurrentSurveyAmountColumnsKernel.ref,
                    population=donor,
                    inputs=predictor_graph._inputs(group.donor_frame),
                    params=group_params,
                    outputs=tuple(
                        Owned("person", c, "float64")
                        for c in (*qualified.features, *group.spec.targets)
                    ),
                    artifact_inputs=(_projection_edge(),),
                    description="Map qualified current amounts and source predictors without numeric or reporting-unknown fills.",
                ),
            )
        )
        fits = legacy_qrf_train_nodes(
            fit_prefix,
            population=donor,
            entity="person",
            predictors=qualified.features,
            targets=group.spec.targets,
            seed=values.SEED,
            n_estimators=n_estimators,
            zero_atol=0,
            phase="current_survey_amounts." + group.spec.key,
        )
        applies = legacy_qrf_apply_matrix_nodes(
            apply_prefix,
            population=parent.attach.FILTER_NODE,
            fit_nodes=fits,
            matrix_producer=PROJECTION_NODE,
            matrix_artifact=group.spec.key + "_matrix",
            seed=values.SEED,
            phase="current_survey_amounts." + group.spec.key,
        )
        nodes.extend((*fits, *applies))
        attach_edges.append(matrix_edge)
        for i, node in enumerate(applies):
            attach_edges.extend(
                (
                    ArtifactInput(
                        f"{group.spec.key}_raw_{i}",
                        node.id,
                        "raw_draw",
                        codec.RAW_TARGET_TYPE,
                    ),
                    ArtifactInput(
                        f"{group.spec.key}_state_{i}",
                        node.id,
                        "apply_state",
                        MATRIX_APPLY_STATE_TYPE,
                    ),
                )
            )
    columns = pd.concat([qualified.native, qualified.reports], axis=1)
    nodes.append(
        Node(
            ATTACH_NODE,
            CurrentSurveyAmountAttachKernel.ref,
            population=parent.attach.FILTER_NODE,
            inputs=predictor_graph._inputs(receiving),
            params=params,
            outputs=tuple(Owned("person", c, str(columns[c].dtype)) for c in columns),
            artifact_inputs=tuple(attach_edges),
            artifact_outputs=(ArtifactOutput("attachment", ATTACHMENT_TYPE),),
            description="Join source observations and ACS conditional draws to both support clones; retain all PUF and source columns, geography, weights and ledger.",
        )
    )
    return tuple(nodes)


class Boundary:
    """Internal retained-value seam; a detached projection cannot construct it."""

    def __init__(self, run, *, groups, n_estimators):
        self.run = run
        self.parent_view = parent.check_survey_puf55_run(run)
        self.parent_entry = parent._run_entry(run)
        self.qualified = values.qualify_current_survey_amounts(run, groups=groups)
        self.qualified_stamp = values.seal(self.qualified)
        self.parent_stamp = physical._population_stamp(run.population)
        self.preparation = run.financial_run.prefix.preparation
        self.health = health_graph.qualify_health_coverage(self.preparation)
        self.health_stamp = health_graph.health_coverage_seal(self.health)
        self.n_estimators = n_estimators
        self.amount_nodes = amount_nodes(
            self.qualified,
            run.population.frame,
            parent_digest=self.parent_view.digest,
            n_estimators=n_estimators,
        )
        self.health_nodes = health_graph.health_coverage_nodes(
            self.health,
            receiving_version=parent.attach.FILTER_NODE,
            after=_amount_edge(),
        )
        self.nodes = (*self.amount_nodes, *self.health_nodes)
        self.declaration = tuple(self.nodes)
        self.live = _live()
        self.compiled = self.kernels = self.store = None
        self.paths = self.source_keys = self.keys = self.implementations = None
        self.parent_objects = (
            run.population,
            run.compiled,
            run.manifest,
            run.store,
            run.kernels,
            run.sources,
        )

    def pure(self):
        parent._pure_run(self.run, self.parent_entry)
        require(
            parent._run_entry(self.run) is self.parent_entry
            and all(
                a is b
                for a, b in zip(
                    self.parent_objects,
                    (
                        self.run.population,
                        self.run.compiled,
                        self.run.manifest,
                        self.run.store,
                        self.run.kernels,
                        self.run.sources,
                    ),
                    strict=True,
                )
            )
            and physical._population_stamp(self.run.population) == self.parent_stamp
            and values.seal(self.qualified) == self.qualified_stamp
            and self.run.financial_run.prefix.preparation is self.preparation
            and health_graph.health_coverage_seal(self.health) == self.health_stamp
            and self.nodes == self.declaration
            and _live() == self.live,
            "BOUNDARY_CHANGED",
        )
        require(
            self.amount_nodes
            == amount_nodes(
                self.qualified,
                self.run.population.frame,
                parent_digest=self.parent_view.digest,
                n_estimators=self.n_estimators,
            )
            and self.health_nodes
            == health_graph.health_coverage_nodes(
                self.health,
                receiving_version=parent.attach.FILTER_NODE,
                after=_amount_edge(),
            )
            and self.nodes == (*self.amount_nodes, *self.health_nodes),
            "BOUNDARY_DECLARATIONS",
        )
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
                and tuple(self.store.codecs.as_mapping().items()) == self.codec_items
                and tuple(self.store.codecs.as_bytes_mapping().items())
                == self.bytes_codec_items,
                "BOUND_REGISTRY_CHANGED",
            )
            require(
                self.compiled == compile_graph(self.compiled.graph)
                and graph_to_json(self.compiled.graph) == self.graph_json,
                "COMPILED_CHANGED",
            )

    def borrow(self):
        require(
            parent.check_survey_puf55_run(self.run).payload == self.parent_view.payload,
            "PARENT_IDENTITY",
        )
        if self.compiled is not None:
            paths, source_keys = _source_paths_and_keys(
                self.compiled, dict(self.paths), self.store
            )
            keys, implementations = _all_node_keys(
                self.compiled, self.kernels, source_keys
            )
            require(
                tuple(sorted(paths.items())) == self.paths
                and tuple(sorted(source_keys.items())) == self.source_keys
                and tuple(sorted(keys.items())) == self.keys
                and tuple(sorted(implementations.items())) == self.implementations,
                "SOURCE_OR_IMPLEMENTATION_CHANGED",
            )
        self.pure()

    def context(self, context):
        # Entry/final host fences check the complete PUF owner. These ordinary
        # fragment calls consume retained, source-qualified values and check
        # their pure seals; they do not repeatedly reread the full PUF pipeline.
        self.pure()
        require(
            not context.sources
            and context.node in self.nodes
            and set(context.artifacts)
            == {e.name for e in context.node.artifact_inputs},
            "CONTEXT_DECLARATION",
        )
        for edge in context.node.artifact_inputs:
            value = values.predictors.host.shared.artifact(
                context, edge.name, edge.type
            )
            require(
                value.producer_key == dict(self.keys)[edge.producer],
                "ARTIFACT_PRODUCER_KEY",
            )
            if edge.name == "projection":
                require(value.payload == self.qualified.projection, "PROJECTION_BYTES")
            if edge == _upstream_edge():
                record = self.run.manifest.node(edge.producer)
                payload = self.run.store.load_bytes(
                    record.opaque_artifacts[edge.artifact]
                )
                require(
                    value.producer_key == record.key
                    and value.key == record.opaque_artifacts[edge.artifact]
                    and value.payload == payload,
                    "PARENT_ARTIFACT",
                )
        self.pure()
        return self.qualified

    def requalify(self):
        """Reconstruct owned source transformations at the host's final I/O fence."""
        fresh = values.qualify_current_survey_amounts(
            self.run, groups=tuple(g.spec.key for g in self.qualified.groups)
        )
        require(
            values.seal(fresh) == self.qualified_stamp, "SOURCE_REQUALIFICATION_CHANGED"
        )
        fresh_health = health_graph.qualify_health_coverage(self.preparation)
        require(
            health_graph.health_coverage_seal(fresh_health) == self.health_stamp,
            "HEALTH_SOURCE_REQUALIFICATION_CHANGED",
        )
        self.pure()


class _Kernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=predictor_graph._Kernel.capabilities.dependencies,
    )

    def __init__(self, boundary):
        self.boundary = boundary

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            values,
            values.unemployment,
            health_graph,
            parent,
            physical,
            predictor_graph,
            values.predictors,
            values.unemployment.coverage,
            values.unemployment.source_csv_builtin,
            population_ops,
            model_input,
            qrf,
            qrf_target,
            dependencies=self.capabilities.dependencies,
        )


class CurrentSurveyAmountProjectionKernel(_Kernel):
    ref = "us.survey_amounts.source_projection@1"

    def run(self, context):
        qualified = self.boundary.context(context)
        values.predictors.host._current_context_frame(
            context, self.boundary.run.population.frame
        )
        return KernelResult(
            artifacts={
                "projection": qualified.projection,
                **{g.spec.key + "_matrix": g.matrix for g in qualified.groups},
            },
            receipt=qualified.evidence,
        )


class CurrentSurveyAmountDonorKernel(_Kernel):
    ref = "us.survey_amounts.source_donor@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.FILTER)

    def run(self, context):
        qualified = self.boundary.context(context)
        values.predictors.host._current_context_frame(context, qualified.source_frame)
        group = next(
            g for g in qualified.groups if g.spec.key == context.params["group"]
        )
        return KernelResult(
            keep=pd.Series(
                group.keep.copy(),
                index=pd.Index(
                    qualified.source_frame.person.person_id.to_numpy(), name="person_id"
                ),
            ),
            receipt={
                "group": group.spec.key,
                "donor_persons": int(group.keep.sum()),
                "selection": "joint_known_current_ASEC_person_targets",
                "weight_kind": "design",
            },
        )


class CurrentSurveyAmountColumnsKernel(_Kernel):
    ref = "us.survey_amounts.source_columns@1"

    def run(self, context):
        qualified = self.boundary.context(context)
        group = next(
            g for g in qualified.groups if g.spec.key == context.params["group"]
        )
        values.predictors.host._current_context_frame(context, group.donor_frame)
        return KernelResult(
            columns={
                ("person", c): group.donor_columns[c] for c in group.donor_columns
            },
            receipt={
                "group": group.spec.key,
                "projection_sha256": codec.sha(qualified.projection),
            },
        )


def read_draws(qualified, artifacts):
    """Check target history, exact matrix producer and immutable draw bytes."""
    draws = {}
    for group in qualified.groups:
        name = group.spec.key
        matrix_value = artifacts[name + "_matrix"]
        require(matrix_value.payload == group.matrix, "MATRIX_BYTES")
        matrix = model_input.decode_recipient_matrix(group.matrix)
        require(tuple(matrix.features) == qualified.features, "MATRIX_FEATURES")
        result = pd.DataFrame(index=matrix.features.index)
        models, raw_history = [], []
        for i, target in enumerate(group.spec.targets):
            raw = artifacts[f"{name}_raw_{i}"].payload
            state_value = artifacts[f"{name}_state_{i}"]
            require(
                state_value.producer_key == artifacts[f"{name}_raw_{i}"].producer_key,
                "RAW_STATE_SIBLINGS",
            )
            packet = decode_matrix_apply_state(state_value.payload)
            application, chain = codec.read_application(
                codec.encode_json(packet["application"])
            )
            raw_history.append({"target": target, "sha256": codec.sha(raw)})
            require(
                packet["matrix_sha256"] == codec.sha(group.matrix)
                and packet["matrix_producer_key"] == matrix_value.producer_key
                and chain.entity == "person"
                and tuple(chain.predictors) == qualified.features
                and tuple(chain.targets) == group.spec.targets
                and tuple(chain.completed_targets) == group.spec.targets[: i + 1]
                and chain.recipient_index == qrf._index_identity(matrix.features.index)
                and application["seed"] == values.SEED
                and application["raw_targets"] == raw_history
                and application["models"][:i] == models
                and len(application["models"]) == i + 1,
                "DRAW_CHAIN",
            )
            models = application["models"]
            result[target] = codec.read_raw_target(
                raw, target=target, index=matrix.features.index
            )
        draws[name] = result
    return draws


def _attachment_result(boundary, artifacts):
    qualified = boundary.qualified
    require(
        artifacts["projection"].payload == qualified.projection, "ATTACHMENT_PROJECTION"
    )
    draws = read_draws(qualified, artifacts)
    columns = values.attach_columns(qualified, boundary.run.population.frame, draws)
    receipt = {
        "protocol": values.PROTOCOL,
        "parent_sha256": boundary.parent_view.digest,
        "projection_sha256": codec.sha(qualified.projection),
        "draw_sha256": {
            n: codec.sha(v.payload) for n, v in artifacts.items() if "_raw_" in n
        },
        "source_unknowns_preserved": True,
        "weights_changed": False,
        "prior_wages_consumed": False,
        "release_eligible": False,
    }
    return KernelResult(
        columns=columns,
        artifacts={"attachment": codec.encode_json(receipt)},
        receipt=receipt,
    )


class CurrentSurveyAmountAttachKernel(_Kernel):
    ref = "us.survey_amounts.attach@1"

    def run(self, context):
        self.boundary.context(context)
        values.predictors.host._current_context_frame(
            context, self.boundary.run.population.frame
        )
        result = _attachment_result(self.boundary, context.artifacts)
        self.boundary.pure()
        return result


def _construct(run, *, groups, n_estimators):
    boundary = Boundary(run, groups=groups, n_estimators=n_estimators)
    compiled = compile_graph(
        replace(run.compiled.graph, nodes=(*run.compiled.graph.nodes, *boundary.nodes))
    )
    kernels = KernelRegistry()
    for kernel in run.kernels.as_mapping().values():
        kernels.register(kernel)
    for cls in (
        CurrentSurveyAmountProjectionKernel,
        CurrentSurveyAmountDonorKernel,
        CurrentSurveyAmountColumnsKernel,
        CurrentSurveyAmountAttachKernel,
    ):
        require(cls.ref not in kernels.refs(), "KERNEL_COLLISION")
        kernels.register(cls(boundary))
    for kernel in health_graph.health_coverage_kernels(
        boundary.health,
        receiving_version=parent.attach.FILTER_NODE,
        after=_amount_edge(),
        require_current=boundary.pure,
    ):
        require(kernel.ref not in kernels.refs(), "HEALTH_KERNEL_COLLISION")
        kernels.register(kernel)
    registry = parent._registry(run.store.codecs, codecs.SourceCodecRegistry())
    store = ContentStore(run.store.root, codecs=registry)
    paths, source_keys = _source_paths_and_keys(compiled, dict(run.sources), store)
    keys, implementations = _all_node_keys(compiled, kernels, source_keys)
    require(
        all(
            keys[n] == run.manifest.node(n).key
            and implementations[n] == run.manifest.node(n).kernel_impl_hash
            and compiled.graph.node(n) == run.compiled.graph.node(n)
            for n in run.compiled.order
        ),
        "PREFIX_DECLARATION_OR_KEY",
    )
    boundary.compiled, boundary.kernels, boundary.store = compiled, kernels, store
    boundary.paths = tuple(sorted(paths.items()))
    boundary.source_keys = tuple(sorted(source_keys.items()))
    boundary.keys = tuple(sorted(keys.items()))
    boundary.implementations = tuple(sorted(implementations.items()))
    boundary.graph_json = graph_to_json(compiled.graph)
    boundary.bound_objects = (compiled, store, kernels, store.codecs)
    boundary.kernel_items = tuple(kernels.as_mapping().items())
    boundary.codec_items = tuple(store.codecs.as_mapping().items())
    boundary.bytes_codec_items = tuple(store.codecs.as_bytes_mapping().items())
    boundary.borrow()
    return boundary


def _verify_models(boundary, loaded, donors):
    """Verify persisted models against actual donor values after artifact admission."""
    for group in boundary.qualified.groups:
        donor_id, _, fit_prefix, apply_prefix = _ids(group)
        donor = donors[donor_id]
        first = boundary.compiled.graph.node(fit_prefix + ".000")
        model_frame = codec.model_frame(
            SimpleNamespace(weights={"person": donor.frame.resolve_weights("person")}),
            first.inputs[0],
            donor.frame.person,
        )
        model = qrf.RegimeGatedQRF(
            seed=values.SEED,
            n_estimators=boundary.n_estimators,
            zero_atol=0,
            max_samples_leaf=None,
        )
        before = qrf_target.LegacyQRFTrainingState.from_chain(
            model.start_chain(
                model_frame,
                list(boundary.qualified.features),
                list(group.spec.targets),
                weights="design",
            )
        )
        history = []
        for i, target in enumerate(group.spec.targets):
            node = f"{fit_prefix}.{i:03d}"
            payload = loaded[node, "model"]
            packet, after = codec.read_training(loaded[node, "training_state"])
            fitted = qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
                payload, expected_sha256=codec.sha(payload)
            )
            require(
                fitted.training_state == before
                and fitted.next_training_state == after
                and fitted.donor_sha256
                == qrf_target._consumed_values_sha256(
                    model_frame.person,
                    (*boundary.qualified.features, *group.spec.targets[:i], target),
                ),
                "TRAINING_DONOR",
            )
            history.append(
                {
                    "target": target,
                    "sha256": codec.sha(payload),
                    "training_id": fitted.training_id,
                }
            )
            require(
                packet["models"] == history
                and decode_matrix_apply_state(
                    loaded[f"{apply_prefix}.{i:03d}", "apply_state"]
                )["application"]["models"]
                == history,
                "TRAINING_APPLY_HISTORY",
            )
            before = after


@dataclass(frozen=True)
class SurveyEnrichmentRun:
    parent_run: parent.SurveyPuf55Run
    population: population_ops.Population
    manifest: object
    compiled: object
    store: ContentStore
    kernels: KernelRegistry
    sources: tuple
    receipt: bytes

    def checked_view(self):
        return check_survey_enrichment_run(self)


@dataclass(frozen=True)
class CheckedSurveyEnrichmentRun:
    """Descriptive values; authority remains in the original retained host."""

    payload: bytes
    digest: str
    population: population_ops.Population


def _run_seal(run):
    return (
        run.receipt,
        run.manifest.to_json(),
        graph_to_json(run.compiled.graph),
        physical._population_stamp(run.population),
        parent._heterogeneous_manifest_seals(run.manifest, run.compiled),
        tuple(sorted(run.manifest.populations)),
        tuple(sorted(run.manifest.mass_ledgers)),
    )


def check_survey_enrichment_run(run):
    entry = _ISSUED.get(id(run))
    require(
        type(run) is SurveyEnrichmentRun and entry is not None and entry[0]() is run,
        "UNISSUED_RUN",
    )
    _, boundary, stamp, objects, artifacts = entry
    try:
        boundary.borrow()
        require(
            all(
                a is b
                for a, b in zip(
                    objects,
                    (
                        run.parent_run,
                        run.population,
                        run.manifest,
                        run.compiled,
                        run.store,
                        run.kernels,
                        run.sources,
                    ),
                    strict=True,
                )
            ),
            "RUN_OBJECTS",
        )
        loaded = parent.financial._artifacts(
            run.manifest,
            run.compiled,
            run.store,
            run.kernels,
            dict(boundary.keys),
            dict(boundary.implementations),
        )
        require(
            tuple(sorted((n, a, codec.sha(p)) for (n, a), p in loaded.items()))
            == artifacts,
            "RUN_ARTIFACTS",
        )
        boundary.borrow()
        require(_run_seal(run) == stamp, "RUN_CHANGED")
        boundary.pure()
    except BaseException:
        if _ISSUED.get(id(run)) is entry:
            del _ISSUED[id(run)]
        raise
    return CheckedSurveyEnrichmentRun(
        run.receipt, codec.sha(run.receipt), run.population
    )


def run_us_survey_enrichment(
    run, *, groups=("unemployment", "health_costs"), n_estimators=100, resume="auto"
):
    """Execute one extension and verify its complete observed parent and output."""
    require(resume in ("auto", "require"), "RESUME")
    boundary = _construct(run, groups=groups, n_estimators=n_estimators)
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
        require(all(record.hit for record in manifest.nodes.values()), "REQUIRED_HITS")
    boundary.borrow()
    loaded = parent.financial._artifacts(
        manifest,
        boundary.compiled,
        boundary.store,
        boundary.kernels,
        dict(boundary.keys),
        dict(boundary.implementations),
    )
    for node in run.compiled.graph.nodes:
        old = run.manifest.node(node.id)
        for name, key in old.opaque_artifacts.items():
            require(
                loaded[node.id, name] == run.store.load_bytes(key), "PREFIX_ARTIFACT"
            )
    physical.replay.same_replayed_population(
        run.population, observed[parent.attach.ATTACH_NODE]
    )
    require(
        loaded[PROJECTION_NODE, "projection"] == boundary.qualified.projection,
        "SOURCE_PROJECTION",
    )
    attach = boundary.compiled.graph.node(ATTACH_NODE)
    artifacts = parent._loaded_values(boundary, manifest, loaded, attach)
    result = _attachment_result(boundary, artifacts)
    require(
        loaded[ATTACH_NODE, "attachment"] == result.artifacts["attachment"],
        "ATTACHMENT_ARTIFACT",
    )
    # Compare every unchanged prefix terminal Frame and ledger against the
    # retained parent manifest. Full execution owners/design anchors on the
    # receiving population are checked separately above against the original.
    prefix_terminal = {}
    for node_id in run.compiled.order:
        prefix_terminal[run.compiled.versions[node_id]] = observed[node_id]
    for version, population in prefix_terminal.items():
        physical.replay.same_replayed_population(
            population_ops.Population.from_frame(
                population.frame, version, mass_ledger=population.mass_ledger
            ),
            population_ops.Population.from_frame(
                run.manifest.population(version),
                version,
                mass_ledger=run.manifest.mass_ledger(version),
            ),
        )
    current, donors = {}, {}
    health_ids = {n.id for n in boundary.health_nodes}
    group_nodes = {_ids(g)[0]: g for g in boundary.qualified.groups}
    column_nodes = {_ids(g)[1]: g for g in boundary.qualified.groups}
    original = population_ops.Population.from_frame(
        boundary.qualified.source_frame, values.predictors.host.survey_graph.CREATE_NODE
    )
    for node_id in boundary.compiled.order:
        node = boundary.compiled.graph.node(node_id)
        version = boundary.compiled.versions[node_id]
        if node_id in run.compiled.order:
            current[version] = observed[node_id]
            continue
        if node_id in health_ids:
            health_artifacts = parent._loaded_values(boundary, manifest, loaded, node)
            expected = health_graph.expected_health_population(
                node_id,
                current.get(version),
                qualified=boundary.health,
                node=node,
                artifacts=health_artifacts,
            )
            # Bind every persisted source/recode/attachment artifact to its
            # independent domain result, including attachment metadata.
            expected_result = health_graph._result(
                boundary.health,
                node,
                None if current.get(version) is None else current[version].frame.person,
            )
            require(
                all(
                    loaded[node_id, name] == payload
                    for name, payload in expected_result.artifacts.items()
                ),
                "HEALTH_RESULT_ARTIFACT",
            )
        elif node_id in group_nodes:
            expected = population_ops.patch(
                original, node, KernelResult(frame=group_nodes[node_id].donor_frame)
            )
        elif node_id in column_nodes:
            group = column_nodes[node_id]
            expected = population_ops.patch(
                current[version],
                node,
                KernelResult(
                    columns={
                        ("person", c): group.donor_columns[c]
                        for c in group.donor_columns
                    }
                ),
            )
            donors[_ids(group)[0]] = expected
        elif node_id == ATTACH_NODE:
            expected = population_ops.patch(current[version], node, result)
        else:
            expected = current[version]
        physical.replay.same_replayed_population(expected, observed[node_id])
        current[version] = expected
    _verify_models(boundary, loaded, donors)
    for version, population in current.items():
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
    receipt = codec.encode_json(
        {
            "protocol": values.PROTOCOL,
            "parent_sha256": boundary.parent_view.digest,
            "manifest_key": manifest.key,
            "node_count": len(boundary.compiled.order),
            "groups": list(groups),
            "complete_population_compared": True,
            "projection_sha256": codec.sha(boundary.qualified.projection),
            "attachment_sha256": codec.sha(result.artifacts["attachment"]),
            "health_projection_sha256": codec.sha(boundary.health.projection),
            "health_attachment_sha256": codec.sha(
                loaded[health_graph.ATTACH_NODE, "attachment"]
            ),
            "health_fields": [f.output for f in health_graph.health.FIELDS],
            "release_eligible": False,
        }
    )
    output = SurveyEnrichmentRun(
        run,
        observed[health_graph.ATTACH_NODE],
        manifest,
        boundary.compiled,
        boundary.store,
        boundary.kernels,
        boundary.paths,
        receipt,
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
    stamp = _run_seal(output)
    artifact_hashes = tuple(
        sorted((n, a, codec.sha(p)) for (n, a), p in loaded.items())
    )
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
    require(
        tuple(sorted((n, a, codec.sha(p)) for (n, a), p in fresh.items()))
        == artifact_hashes
        and _run_seal(output) == stamp
        and all(physical._population_stamp(observed[n]) == stamps[n] for n in observed),
        "FINAL_OUTPUT",
    )
    boundary.pure()
    require(id(output) not in _ISSUED, "RUN_ALREADY_ISSUED")
    ident = id(output)

    def forget(ref):
        old = _ISSUED.get(ident)
        if old is not None and old[0] is ref:
            del _ISSUED[ident]

    entry = (weakref.ref(output, forget), boundary, stamp, objects, artifact_hashes)
    _ISSUED[ident] = entry
    return output
