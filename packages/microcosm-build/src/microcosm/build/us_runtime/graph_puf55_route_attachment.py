"""One source-bound PUF55 attachment over the complete survey clone cohort.

This internal seam is constructed only from an actual retained financial run.
It authenticates source and typed producer inputs before trusted model decoding.
Its mutation seals are in-process checks, not a new source/admission authority.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from types import FunctionType

import numpy as np
import pandas as pd

from microcosm.fit.graph_legacy_qrf import (
    legacy_qrf_apply_matrix_nodes,
    legacy_qrf_train_nodes,
)
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
from microcosm.graph.executor import _all_node_keys, _source_paths_and_keys
from microcosm.graph.serialize import graph_to_json

from . import graph_full_puf_enrichment as physical
from . import graph_puf55_canonical_donor as canonical
from . import graph_puf55_survey_recipients as recipient
from . import puf55_route_finalization as numerical

financial = recipient.financial
full, codec = physical.full, physical.codec
PROFILES = numerical.PROFILES
FILTER_NODE = "survey_puf55.receiving"
MASK_NODE = "survey_puf55.mask"
ATTACH_NODE = "survey_puf55.attach"
PLACEMENT_TYPE = ArtifactType("microcosm.us.survey_puf55.placement", 1)
FINALIZATION_TYPE = ArtifactType("microcosm.us.survey_puf55.finalization", 2)
PROTOCOL = "microcosm.us.survey_puf55.attachment.v1"


def require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_PUF55_" + reason)


def _live():
    result = []
    for name in (__name__, __package__ + ".graph_survey_puf55"):
        module = sys.modules.get(name)
        require(module is not None, "HOST_MODULE")
        for key, value in vars(module).items():
            if type(value) is FunctionType:
                result.append(
                    (name, key, financial.values.source._function_seal(value))
                )
            elif isinstance(value, type) and value.__module__ == name:
                result.append((name, key, value))
                for member, function in vars(value).items():
                    if isinstance(function, (classmethod, staticmethod)):
                        function = function.__func__
                    if type(function) is FunctionType:
                        result.append(
                            (
                                name,
                                key,
                                member,
                                financial.values.source._function_seal(function),
                            )
                        )
    return tuple(result)


def _result_seal(result):
    require(
        type(result) is KernelResult and type(result.columns) is dict, "RESULT_TYPE"
    )
    require(
        result.frame
        is result.keep
        is result.expand
        is result.weights
        is result.strata
        is None
        and type(result.artifacts) is dict
        and all(
            type(k) is str and type(v) is bytes for k, v in result.artifacts.items()
        ),
        "RESULT_FIELDS",
    )
    return (
        tuple(
            (e, c, physical._table_stamp(s.to_frame()))
            for (e, c), s in sorted(result.columns.items())
        ),
        tuple(sorted(result.artifacts.items())),
        codec.encode_json(dict(result.receipt)),
    )


def _artifact_seal(values):
    """Detach both typed provenance and exact immutable payloads before I/O."""
    require(type(values) is dict, "ARTIFACT_VALUES_TYPE")
    result = []
    for name, value in sorted(values.items()):
        require(
            type(name) is str
            and type(value) is physical.ArtifactValue
            and type(value.payload) is bytes
            and type(value.key) is str
            and type(value.producer_key) is str,
            "ARTIFACT_VALUE_TYPE",
        )
        result.append(
            (
                name,
                value.payload,
                codec.encode_json(
                    {
                        "type": {
                            "name": value.type.name,
                            "schema_version": value.type.schema_version,
                        },
                        "key": value.key,
                        "producer_key": value.producer_key,
                        "numerics": artifact_edges.scope_payload(value.numerics),
                    }
                ),
            )
        )
    return tuple(result)


def _donor_seal(frame, donors):
    require(
        type(frame) is full.Frame
        and type(donors) is tuple
        and all(type(donor) is pd.DataFrame for donor in donors),
        "DONOR_RESULT_TYPE",
    )
    return canonical._frame_seal(frame), tuple(physical._table_stamp(d) for d in donors)


@dataclass(frozen=True)
class Route:
    profile: full.PufOutputProfile
    fits: tuple[Node, ...]
    applies: tuple[Node, ...]


def route_nodes(profile, *, seed, n_estimators, zero_atol):
    """Both training Slices read the one canonical donor's unchanged row axis."""
    require(profile in PROFILES, "PROFILE")
    prefix = "survey_puf55." + ("nine" if profile is PROFILES[0] else "eight")
    fits = legacy_qrf_train_nodes(
        prefix=prefix + ".fit",
        population=canonical.CANONICAL_DONOR_NODE,
        entity="tax_unit",
        predictors=profile.predictors,
        targets=profile.targets,
        seed=seed,
        phase=profile.phase,
        n_estimators=n_estimators,
        zero_atol=zero_atol,
    )
    applies = legacy_qrf_apply_matrix_nodes(
        prefix=prefix + ".apply",
        population=FILTER_NODE,
        fit_nodes=fits,
        matrix_producer=recipient.MATRIX_NODE,
        matrix_artifact=recipient._NAMES[profile.value],
        seed=seed,
        phase=profile.phase,
    )
    physical._profile_chain(profile, fits, applies)
    return Route(profile, fits, applies)


def _base_edges(profiles):
    return (
        ArtifactInput(
            "recipient_projection",
            recipient.PROJECTION_NODE,
            "projection",
            recipient.PROJECTION_TYPE,
        ),
        *(
            ArtifactInput(
                recipient._NAMES[p.value],
                recipient.MATRIX_NODE,
                recipient._NAMES[p.value],
                recipient.model_input.RECIPIENT_MATRIX_TYPE,
            )
            for p in profiles
        ),
    )


def _donor_edges():
    node = canonical.CANONICAL_DONOR_NODE
    return (
        ArtifactInput(
            "full_return_source",
            node,
            "full_return_source",
            canonical.source_graph.FULL_RETURN_SOURCE_TYPE,
        ),
        ArtifactInput(
            "canonical_donor", node, "canonical_donor", canonical.CANONICAL_DONOR_TYPE
        ),
        ArtifactInput(
            "donor_projection",
            node,
            "donor_projection",
            canonical.DONOR_PROJECTION_TYPE,
        ),
    )


def _chain_edges(routes):
    result = []
    for r, route in enumerate(routes):
        for i, (fit, apply) in enumerate(zip(route.fits, route.applies, strict=True)):
            # Every model/history is an explicit typed dependency. Last-model
            # equality alone cannot authenticate an earlier executable pickle.
            result.extend(
                (
                    ArtifactInput(
                        f"r{r}_model_{i:03d}",
                        fit.id,
                        "model",
                        physical.qrf_target.LEGACY_QRF_TARGET_TYPE,
                    ),
                    ArtifactInput(
                        f"r{r}_training_{i:03d}",
                        fit.id,
                        "training_state",
                        codec.TRAINING_STATE_TYPE,
                    ),
                    ArtifactInput(
                        f"r{r}_raw_{i:03d}", apply.id, "raw_draw", codec.RAW_TARGET_TYPE
                    ),
                    ArtifactInput(
                        f"r{r}_apply_{i:03d}",
                        apply.id,
                        "apply_state",
                        physical.graph_legacy_apply_matrix.MATRIX_APPLY_STATE_TYPE,
                    ),
                )
            )
    return tuple(result)


def _mask_result(frame):
    return KernelResult(
        columns={
            (e, physical.MASKS[e]): pd.Series(
                mask, index=frame.table(e)[frame.schema.entity_id_column(e)], dtype=bool
            )
            for e, mask in physical._masks(frame).items()
        }
    )


def _keep_result(frame):
    entity = frame.schema.person_entity
    return KernelResult(
        keep=pd.Series(
            True,
            index=frame.table(entity)[frame.schema.entity_id_column(entity)],
            dtype=bool,
        )
    )


def _params(qualified, donor_node, routes, seed):
    return {
        "protocol": PROTOCOL,
        "financial_run_sha256": codec.sha(
            financial._run_entry(qualified.financial_run)[1]
        ),
        "recipient_projection_sha256": codec.sha(qualified.receipt),
        "route_matrices": tuple((name, codec.sha(p)) for name, p in qualified.matrices),
        "donor_recipe": codec.encode_json(dict(donor_node.params)).decode(),
        "route_profiles": tuple(r.profile.value for r in routes),
        "seed": seed,
    }


def attachment_nodes(qualified, donor_node, routes, *, seed):
    """Declare the checked keep-all rewrite version, mask and single attachment."""
    run = qualified.financial_run
    frame = run.financial_population.frame
    profiles = tuple(r.profile for r in routes)
    params = _params(qualified, donor_node, routes, seed)
    survey_sources = tuple(
        name for name, _ in financial._run_entry(run)[2].source_items
    )
    all_sources = tuple(sorted((*survey_sources, *donor_node.sources)))
    keep = Node(
        FILTER_NODE,
        SurveyPuf55KeepAllKernel.ref,
        base=run.financial_population.version,
        structural=StructuralDelta.FILTER,
        mass="conserve",
        inputs=physical._inputs(frame, profile=PROFILES[0]),
        params=params,
        sources=survey_sources,
        artifact_inputs=_base_edges(profiles),
    )
    mask = Node(
        MASK_NODE,
        SurveyPuf55MaskKernel.ref,
        population=FILTER_NODE,
        inputs=physical._inputs(frame, masks=True, profile=PROFILES[0]),
        outputs=tuple(Owned(e, c, "bool") for e, c in physical.MASKS.items()),
        sources=all_sources,
        params=params,
        artifact_inputs=(*_base_edges(profiles), *_donor_edges()),
        artifact_outputs=(ArtifactOutput("placement", PLACEMENT_TYPE),),
    )
    attach = Node(
        ATTACH_NODE,
        SurveyPuf55AttachKernel.ref,
        population=FILTER_NODE,
        inputs=(
            *physical._inputs(frame, profile=PROFILES[0]),
            *(Slice(e, (c,)) for e, c in physical.MASKS.items()),
        ),
        outputs=physical._outputs(frame, profile=PROFILES[0]),
        sources=all_sources,
        params=params,
        artifact_inputs=(
            *_base_edges(profiles),
            *_donor_edges(),
            ArtifactInput("placement", MASK_NODE, "placement", PLACEMENT_TYPE),
            *_chain_edges(routes),
        ),
        artifact_outputs=(ArtifactOutput("finalization", FINALIZATION_TYPE),),
    )
    return keep, mask, attach


class Boundary:
    """Private retained live-owner boundary, created before extension execution."""

    def __init__(
        self,
        run,
        qualified,
        donor_kernel,
        donor_paths,
        *,
        seed,
        n_estimators,
        zero_atol,
    ):
        require(qualified.financial_run is run, "RECIPIENT_RUN")
        self.run, self.entry, self.qualified = run, financial._run_entry(run), qualified
        self.qualified_seal = recipient.values._result_stamp(qualified)
        self.donor_kernel, self.donor_state = donor_kernel, donor_kernel._state
        self.donor_paths = tuple(sorted(donor_paths.items()))
        self.seed = seed
        names = tuple(name for name, _ in qualified.matrices)
        require(
            names and names == tuple(p.value for p in PROFILES if p.value in names),
            "ROUTES",
        )
        self.routes = tuple(
            route_nodes(p, seed=seed, n_estimators=n_estimators, zero_atol=zero_atol)
            for p in PROFILES
            if p.value in names
        )
        self.donor_node = canonical._node(
            donor_kernel._definition, json.loads(donor_kernel._state[2])
        )
        self.nodes = attachment_nodes(
            qualified, self.donor_node, self.routes, seed=seed
        )
        # Pure reconstruction from the retained upstream. No issuer/filter run.
        kept = run.financial_population.frame.select(
            np.ones(run.financial_population.frame.n("person"), dtype=bool)
        )
        self.expected = population_ops.patch(
            run.financial_population, self.nodes[0], KernelResult(frame=kept)
        )
        self.masked = population_ops.patch(
            self.expected, self.nodes[1], _mask_result(self.expected.frame)
        )
        self.expected_stamp, self.masked_stamp = (
            physical._population_stamp(p) for p in (self.expected, self.masked)
        )
        self.live = _live()
        self.compiled = self.store = self.kernels = None
        self.paths = self.source_keys = self.keys = self.implementations = (
            self.declaration
        ) = None
        self.configuration = (
            self.run,
            self.entry,
            self.qualified,
            self.qualified_seal,
            self.donor_kernel,
            self.donor_state,
            self.donor_paths,
            self.seed,
            self.routes,
            self.donor_node,
            self.nodes,
            self.expected,
            self.masked,
            self.expected_stamp,
            self.masked_stamp,
            self.live,
        )
        self.computed = None

    def bind(self, compiled, store, kernels, paths, source_keys, keys, implementations):
        require(self.compiled is None, "REBIND")
        self.compiled, self.store, self.kernels = compiled, store, kernels
        self.paths = tuple(sorted(paths.items()))
        self.source_keys = tuple(sorted(source_keys.items()))
        self.keys = tuple(sorted(keys.items()))
        self.implementations = tuple(sorted(implementations.items()))
        self.declaration = graph_to_json(compiled.graph)
        self.kernel_items = tuple(sorted(kernels.as_mapping().items()))
        self.codec_items = (
            tuple(sorted(store.codecs.as_mapping().items())),
            tuple(sorted(store.codecs.as_bytes_mapping().items())),
        )
        self.bound_objects = (compiled, store, kernels, store.codecs)
        self.bound_values = (
            self.paths,
            self.source_keys,
            self.keys,
            self.implementations,
            self.declaration,
            self.kernel_items,
            self.codec_items,
            store.root,
        )
        self.pure()

    def pure(self):
        require(
            all(
                a is b
                for a, b in zip(
                    (
                        self.run,
                        self.entry,
                        self.qualified,
                        self.qualified_seal,
                        self.donor_kernel,
                        self.donor_state,
                        self.donor_paths,
                        self.seed,
                        self.routes,
                        self.donor_node,
                        self.nodes,
                        self.expected,
                        self.masked,
                        self.expected_stamp,
                        self.masked_stamp,
                        self.live,
                    ),
                    self.configuration,
                    strict=True,
                )
            ),
            "RETAINED_CONFIGURATION_CHANGED",
        )
        financial._pure_run(self.run, self.entry)
        self.donor_kernel._check_state(self.donor_state)
        require(
            recipient.values._result_stamp(self.qualified) == self.qualified_seal,
            "RECIPIENT_CHANGED",
        )
        require(
            physical._population_stamp(self.expected) == self.expected_stamp
            and physical._population_stamp(self.masked) == self.masked_stamp
            and _live() == self.live,
            "RECEIVING_OR_CODE_CHANGED",
        )
        require(
            self.nodes
            == attachment_nodes(
                self.qualified, self.donor_node, self.routes, seed=self.seed
            ),
            "NODE_CHANGED",
        )
        if self.compiled is not None:
            require(
                all(
                    a is b
                    for a, b in zip(
                        (self.compiled, self.store, self.kernels, self.store.codecs),
                        self.bound_objects,
                        strict=True,
                    )
                )
                and (
                    self.paths,
                    self.source_keys,
                    self.keys,
                    self.implementations,
                    self.declaration,
                    self.kernel_items,
                    self.codec_items,
                    self.store.root,
                )
                == self.bound_values
                and graph_to_json(self.compiled.graph) == self.declaration
                and tuple(sorted(self.kernels.as_mapping().items()))
                == self.kernel_items
                and (
                    tuple(sorted(self.store.codecs.as_mapping().items())),
                    tuple(sorted(self.store.codecs.as_bytes_mapping().items())),
                )
                == self.codec_items,
                "HOST_BINDING_CHANGED",
            )
            require(
                all(
                    self.compiled.graph.node(node.id) == node
                    for route in self.routes
                    for node in (*route.fits, *route.applies)
                ),
                "ROUTE_DECLARATION_CHANGED",
            )
        if self.computed is not None:
            require(
                _result_seal(self.computed[0]) == self.computed[1],
                "DETACHED_RESULT_CHANGED",
            )

    def borrow(self):
        """All source/store/owner reads, followed by pure retained-object seals."""
        self.pure()
        paths, sources = _source_paths_and_keys(
            self.compiled, dict(self.paths), self.store
        )
        keys, implementations = _all_node_keys(self.compiled, self.kernels, sources)
        fresh = recipient.values.qualify_puf55_survey_recipients(self.run)
        self.donor_kernel._read_resources(self.donor_state)
        require(
            tuple(sorted(paths.items())) == self.paths
            and tuple(sorted(sources.items())) == self.source_keys
            and tuple(sorted(keys.items())) == self.keys
            and tuple(sorted(implementations.items())) == self.implementations
            and recipient.values._result_stamp(fresh) == self.qualified_seal,
            "SOURCE_IMPLEMENTATION_OR_RECIPIENT_CHANGED",
        )
        self.pure()

    def context(self, context, node, population):
        self.pure()
        require(
            tuple(sorted(context.sources.items()))
            == tuple((n, p) for n, p in self.paths if n in node.sources),
            "CONTEXT_SOURCE_PATHS",
        )
        # The maintained projection helper deliberately handles source-free
        # contexts. Validate the declared source roster above, then reuse only
        # its complete table/weight/strata projection comparison.
        physical._context_projection(
            replace(context, sources={}), node, population.frame
        )
        require(
            set(context.artifacts) == {e.name for e in node.artifact_inputs},
            "CONTEXT_ARTIFACT_ROSTER",
        )
        keys = dict(self.keys)
        values = {
            e.name: physical._value(
                e, context.artifacts[e.name], producer_key=keys[e.producer]
            )
            for e in node.artifact_inputs
        }
        self.projections(values)
        return values

    def stored_artifacts(self, values):
        """Bind every declared byte input to its actual computed producer key.

        The kernel context's public ArtifactValue constructor does not issue
        producer authority. These reads use the retained actual store and graph
        keys, before any numerical helper can decode a trusted model.
        """
        seal = _artifact_seal(values)
        retained_payloads = {name: payload for name, payload, _ in seal}
        keys = dict(self.keys)
        for edge in self.nodes[2].artifact_inputs:
            value = physical._value(
                edge, values[edge.name], producer_key=keys[edge.producer]
            )
            payload = self.store.load_bytes(value.key)
            require(
                type(payload) is bytes and payload == retained_payloads[edge.name],
                "STORED_ARTIFACT_BYTES",
            )
        require(_artifact_seal(values) == seal, "ARTIFACT_CHANGED_DURING_STORE_READ")
        self.pure()

    def projections(self, values):
        require(
            values["recipient_projection"].payload == self.qualified.receipt,
            "RECIPIENT_PROJECTION",
        )
        for name, payload in self.qualified.matrices:
            require(
                values[recipient._NAMES[name]].payload == payload, "RECIPIENT_MATRIX"
            )

    def canonical_donor(self, values):
        """Reconstruct from the actual pinned originals, before any pickle read."""
        state = self.donor_state
        definition, _, _, _, seed, scheme, *_ = state
        ordered = tuple(
            (p.source_name, dict(self.donor_paths)[p.source_name])
            for p in (definition.main, definition.demographic)
        )
        buffers = self.donor_kernel._read_sources(ordered, state)
        self.donor_kernel._read_resources(state)
        decoded = canonical.source.decode_full_puf_source(*buffers, definition)
        raw_payload = canonical.source.encode_full_puf_source(decoded)
        constructed = canonical.canonical.construct_canonical_puf59(
            decoded,
            interest_bands=canonical.interest.US_PUF_E19200_AGI_BANDS,
            interest_asset_sha256=canonical.canonical.INTEREST_ASSET_SHA256,
            seed=seed,
            growth_scheme=scheme,
        )
        payload = canonical.envelope.encode_canonical_puf59(
            constructed, expected_growth_scheme=scheme
        )
        require(
            values["full_return_source"].payload == raw_payload
            and values["canonical_donor"].payload == payload,
            "CANONICAL_SOURCE_RECONSTRUCTION",
        )
        donors = tuple(
            canonical.projection.canonical_puf55_donor_from_artifact(
                payload,
                expected_artifact_sha256=codec.sha(payload),
                expected_growth_scheme=scheme,
                profile=route.profile,
            )[0]
            for route in self.routes
        )
        shared, evidence = canonical.projection.canonical_puf55_donor_from_artifact(
            payload,
            expected_artifact_sha256=codec.sha(payload),
            expected_growth_scheme=scheme,
            profile=PROFILES[0],
        )
        frame = numerical._model_donor_frame(
            full._validated_model_donor(shared, profile=PROFILES[0])
        )
        ids = decoded.status["RECID"][decoded.ordinary]
        require(
            np.array_equal(frame.table("tax_unit").index.to_numpy(), ids),
            "DONOR_SOURCE_AXIS",
        )
        require(
            frame.weights_for("tax_unit").values.tobytes()
            == (
                decoded.status["S006"][decoded.ordinary].astype(np.float64) / 100
            ).tobytes(),
            "DONOR_SOURCE_WEIGHTS",
        )
        receipt = codec.decode_json(values["donor_projection"].payload)
        require(
            receipt["protocol"] == canonical.PHASE
            and receipt["projection"] == evidence
            and receipt["model_frame_sha256"] == canonical._frame_seal(frame)
            and receipt["full_return_source_sha256"] == codec.sha(raw_payload)
            and receipt["canonical_donor_sha256"] == codec.sha(payload)
            and all(receipt[k] == v for k, v in json.loads(state[2]).items()),
            "DONOR_PROJECTION",
        )
        # Carry this immutable seal through the function-return boundary. The
        # model Frame omits incidence capacity, so seal complete donor tables too.
        seal = _donor_seal(frame, donors)
        self.pure()
        require(_donor_seal(frame, donors) == seal, "DONOR_RESULT_CHANGED")
        return frame, donors, seal

    def placement(self):
        frame = self.expected.frame
        return codec.encode_json(
            {
                **dict(self.nodes[1].params),
                "selected_ids": {
                    e: frame.table(e).loc[m, frame.schema.entity_id_column(e)].tolist()
                    for e, m in physical._masks(frame).items()
                },
                "target_order": list(PROFILES[0].targets),
                "release_eligible": False,
            }
        )

    def histories(self, values):
        """Check every typed model's byte hash against both recorded histories."""
        for r, route in enumerate(self.routes):
            models, raw = [], []
            for i, target in enumerate(route.profile.targets):
                model = values[f"r{r}_model_{i:03d}"].payload
                packet, _ = codec.read_training(
                    values[f"r{r}_training_{i:03d}"].payload
                )
                app = physical.graph_legacy_apply_matrix.decode_matrix_apply_state(
                    values[f"r{r}_apply_{i:03d}"].payload
                )
                require(
                    len(packet["models"]) == i + 1
                    and packet["models"][:i] == models
                    and packet["models"][-1]["target"] == target
                    and packet["models"][-1]["sha256"] == codec.sha(model),
                    "TRAINING_HISTORY",
                )
                models = packet["models"]
                require(app["application"]["models"] == models, "APPLY_MODEL_HISTORY")
                raw.append((target, values[f"r{r}_raw_{i:03d}"].payload))
            # The maintained raw merger checks each target's chain, matrix,
            # phase/seed and complete entity-ID coverage without deserialization.
            yield numerical.Puf55RouteDraws(
                route.profile,
                values[recipient._NAMES[route.profile.value]].payload,
                dict(self.keys)[recipient.MATRIX_NODE],
                tuple(raw),
                values[f"r{r}_apply_{len(route.applies) - 1:03d}"].payload,
                values[f"r{r}_training_{len(route.fits) - 1:03d}"].payload,
            )

    def finalize(self, values):
        require(
            set(values) == {e.name for e in self.nodes[2].artifact_inputs},
            "FINAL_ARTIFACT_ROSTER",
        )
        keys = dict(self.keys)
        values = {
            e.name: physical._value(e, values[e.name], producer_key=keys[e.producer])
            for e in self.nodes[2].artifact_inputs
        }
        artifact_seal = _artifact_seal(values)
        self.borrow()
        self.stored_artifacts(values)
        self.projections(values)
        require(values["placement"].payload == self.placement(), "PLACEMENT")
        donor_frame, donors, donor_seal = self.canonical_donor(values)
        require(_donor_seal(donor_frame, donors) == donor_seal, "DONOR_RESULT_CHANGED")
        require(
            _artifact_seal(values) == artifact_seal, "ARTIFACT_CHANGED_BEFORE_DECODE"
        )
        self.pure()
        draws = tuple(self.histories(values))
        # JSON/history decoders are also callback boundaries. No trusted pickle
        # is decoded until the retained source-derived inputs still match.
        require(
            _artifact_seal(values) == artifact_seal, "ARTIFACT_CHANGED_BEFORE_DECODE"
        )
        require(_donor_seal(donor_frame, donors) == donor_seal, "DONOR_RESULT_CHANGED")
        self.pure()
        # Full receiving Population stamps above and below close the values-only
        # numerical helper's deliberately narrower receiving-axis boundary.
        candidate, evidence = numerical._finalize_puf55_routes(
            self.expected.frame,
            recipient_matrices=self.qualified.matrices,
            routes=tuple(
                numerical.Puf55RouteFinalizationInput(
                    draw,
                    donor,
                    values[f"r{r}_model_{len(route.fits) - 1:03d}"].payload,
                    route.profile.phase,
                )
                for r, (route, draw, donor) in enumerate(
                    zip(self.routes, draws, donors, strict=True)
                )
            ),
            seed=self.seed,
        )
        require(type(evidence) is bytes, "NUMERICAL_RECEIPT_TYPE")
        numerical_receipt = codec.decode_json(evidence)
        candidate_stamp = numerical_receipt.get("candidate_frame_sha256")
        protocol = numerical_receipt.get("protocol")
        # The decoder's mutable mapping is descriptive. Capture exact scalar
        # literals and compare its canonical bytes with the retained receipt.
        require(
            type(protocol) is str
            and protocol == numerical.FINALIZATION_PROTOCOL
            and type(candidate_stamp) is str
            and codec.encode_json(numerical_receipt) == evidence,
            "NUMERICAL_RECEIPT_BINDING",
        )
        require(
            candidate_stamp == numerical._candidate_frame_sha256(candidate),
            "NUMERICAL_OUTPUT_CHANGED",
        )
        # The later-I/O baseline is the immutable digest made by the callee,
        # never a first stamp of possibly changed post-return candidate values.
        columns = {}
        for owned in self.nodes[2].outputs:
            mask = physical._masks(self.expected.frame)[owned.entity]
            table = candidate.table(owned.entity)
            selected = table.loc[mask, owned.column]
            require(
                selected.notna().all()
                and np.isfinite(selected.to_numpy(dtype=np.float64)).all(),
                "FINALIZED_VALUES",
            )
            columns[owned.entity, owned.column] = pd.Series(
                selected.astype(owned.dtype).array,
                index=table.loc[mask, candidate.schema.entity_id_column(owned.entity)],
                dtype=owned.dtype,
            )
        # KernelResult/receipt constructors are detached-output boundaries too.
        # Retain selected-column stamps before giving the mutable dict to either.
        column_seal = tuple(
            (e, c, physical._table_stamp(series.to_frame()))
            for (e, c), series in sorted(columns.items())
        )
        result = KernelResult(
            columns=columns,
            artifacts={"finalization": evidence},
            receipt=codec.decode_json(evidence),
        )
        require(
            result.artifacts == {"finalization": evidence}
            and codec.encode_json(dict(result.receipt)) == evidence,
            "NUMERICAL_RECEIPT_BINDING",
        )
        seal = _result_seal(result)
        require(seal[0] == column_seal, "NUMERICAL_COLUMN_BINDING")
        # The first complete seal must bind both mutable metadata containers to
        # the immutable callee receipt, even if the earlier encoder return was
        # a mutation boundary. Never adopt their current values as authority.
        require(
            seal[1:] == ((("finalization", evidence),), evidence),
            "NUMERICAL_RECEIPT_BINDING",
        )
        # Last I/O, then only immutable baselines and complete physical checks.
        self.borrow()
        require(
            numerical._candidate_frame_sha256(candidate) == candidate_stamp
            and _donor_seal(donor_frame, donors) == donor_seal
            and _artifact_seal(values) == artifact_seal
            and _result_seal(result) == seal,
            "FINALIZATION_OUTPUT_CHANGED",
        )
        self.pure()
        if self.computed is not None:
            require(self.computed[1] == seal, "REPEATED_FINALIZATION_CHANGED")
        self.computed = (result, seal)
        return result


class _Kernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, boundary):
        self.boundary = boundary

    def implementation_hash(self):
        # Code and package resources are freshly bound on cache hits as well.
        return codec.sha(
            codec.encode_json(
                {
                    "attachment": source_hash(
                        sys.modules[__name__],
                        sys.modules[__package__ + ".graph_survey_puf55"],
                        physical,
                        numerical,
                        recipient,
                        recipient.values,
                        *canonical._modules(),
                        dependencies=self.capabilities.dependencies,
                    ),
                    "financial": recipient._Kernel.implementation_hash(self),
                }
            )
        )


class SurveyPuf55KeepAllKernel(_Kernel):
    ref = "us.survey_puf55.keep_all@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.FILTER)

    def run(self, context):
        b = self.boundary
        b.context(context, b.nodes[0], b.run.financial_population)
        fresh = recipient.values.qualify_puf55_survey_recipients(b.run)
        require(
            recipient.values._result_stamp(fresh) == b.qualified_seal,
            "FILTER_RECIPIENT",
        )
        result = _keep_result(b.run.financial_population.frame)
        seal = physical._table_stamp(result.keep.to_frame())
        financial.check_atomic_survey_financial_run(b.run)
        b.context(context, b.nodes[0], b.run.financial_population)
        require(physical._table_stamp(result.keep.to_frame()) == seal, "KEEP_CHANGED")
        return result


class SurveyPuf55MaskKernel(_Kernel):
    ref = "us.survey_puf55.mask@1"

    def run(self, context):
        b = self.boundary
        b.borrow()
        b.context(context, b.nodes[1], b.expected)
        result = _mask_result(b.expected.frame)
        result.artifacts["placement"] = b.placement()
        seal = _result_seal(result)
        b.borrow()
        b.context(context, b.nodes[1], b.expected)
        require(_result_seal(result) == seal, "MASK_CHANGED")
        return result


class SurveyPuf55AttachKernel(_Kernel):
    ref = "us.survey_puf55.attach@1"

    def run(self, context):
        b = self.boundary
        values = b.context(context, b.nodes[2], b.masked)
        result = b.finalize(values)
        b.context(context, b.nodes[2], b.masked)
        b.pure()
        return result
