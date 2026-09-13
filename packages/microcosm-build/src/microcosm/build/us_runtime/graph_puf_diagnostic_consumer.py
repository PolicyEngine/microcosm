"""Source-qualified development diagnostic, preserving the fixture-only path.

The runner must first perform the bounded reader/feature check before allowing
fit. Graph fit keys deliberately have no host ancestry: changing recipient
features can reuse the same donor model. No source metadata is promoted.
"""

from __future__ import annotations

import json
import sys

from microcosm.fit import model_input
from microcosm.fit.graph_legacy_qrf import (
    legacy_qrf_apply_matrix_nodes,
    legacy_qrf_train_nodes,
)
from microcosm.frame import US_SCHEMA, Frame
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
    SourceRef,
    StructuralDelta,
    source_hash,
)

from . import graph_asec_income as income
from . import graph_combined_clone as survey_clone
from . import graph_composed_contracts as composed_contracts
from . import graph_context
from . import graph_native_household_origin as origins
from . import graph_puf_detail_transfer as shared
from . import graph_survey_population as survey_graph
from . import puf_diagnostic_consumer as consumer
from . import survey_origin_budget as survey_budget
from . import survey_population_preparation as survey_source
from . import survey_population_replay as survey_replay
from .asec_current_money_selection import (
    US_ASEC_PREPARED_RECEIPT_TYPE,
    US_ASEC_SELECTED_MONEY_TYPE,
)

detail, codec, require = shared.detail, shared.codec, shared.require
TRANSPORT_TYPE = ArtifactType("microcosm.us.puf_diagnostic_transport", 1)
PLACEMENT_TYPE = ArtifactType("microcosm.us.puf_diagnostic_placement", 1)
PREFIX = "puf_diagnostic"
SOURCE_NAMES = {n: PREFIX + "_" + n for n in consumer.ROLES}
SOURCES = tuple(SourceRef(n, "raw-bytes-v1") for n in SOURCE_NAMES.values())


class _Kernel(shared._Kernel):
    def implementation_hash(self):
        # Only the legacy diagnostic attests the composed measurement owner.
        from . import graph_composed_asec_measures as measures

        return codec.sha(
            codec.encode_json(
                {
                    "consumer": source_hash(
                        sys.modules[__name__],
                        consumer,
                        composed_contracts,
                        dependencies=self.capabilities.dependencies,
                    ),
                    "shared": shared._Kernel().implementation_hash(),
                    "asec": measures.USComposedAsecReportedIncomeKernel().implementation_hash(),
                    "native_origins": origins.PopulationOriginBindingKernel().implementation_hash(),
                }
            )
        )


class DiagnosticDonorKernel(_Kernel):
    ref = "us.puf_diagnostic.donor@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context):
        require(
            set(context.params) == {"expected_pins", "route", "recipe"}
            and not context.node.inputs
            and not context.artifacts
            and set(context.sources) == set(SOURCE_NAMES.values()),
            "DIAGNOSTIC_DONOR_DECLARATION",
        )
        pins = codec.decode_json(context.params["expected_pins"].encode())
        require(set(pins) == set(consumer.ROLES), "PRICE_TRANSPORT_ROSTER")
        payloads = {}
        for role, name in SOURCE_NAMES.items():
            size = pins[role]["bytes"]
            require(
                type(size) is int and 0 < size <= detail.MAX_BYTES,
                "PRICE_TRANSPORT_SIZE",
            )
            with context.sources[name].open("rb") as handle:
                payloads[role] = handle.read(size + 1)
            consumer.check_pin(
                payloads[role], pins[role], "PRICE_TRANSPORT_PIN:" + role
            )
        frame, payload = consumer.qualify_price_transport(
            payloads,
            pins,
            route=context.params["route"],
            recipe=codec.decode_json(context.params["recipe"].encode()),
        )
        return KernelResult(
            frame=frame,
            artifacts={"transport": payload},
            receipt={
                "scope": consumer.SCOPE,
                "route": context.params["route"],
                "recipe": consumer.recipe_document(),
                "transport_sha256": codec.sha(payload),
                "original_source_fit_admission": detail.SOURCE_ADMISSION,
                "release_eligible": False,
            },
        )


def host_edges(*, native_binding, clone_binding):
    return (
        ArtifactInput("acs", origins.ACS_NODE, "origins", origins.SOURCE_TYPE),
        ArtifactInput("asec", origins.ASEC_NODE, "origins", origins.SOURCE_TYPE),
        ArtifactInput(
            "parent_binding", native_binding, "binding", origins.BINDING_TYPE
        ),
        ArtifactInput("clone_binding", clone_binding, "binding", origins.BINDING_TYPE),
        ArtifactInput(
            "asec_binding",
            composed_contracts.BIND_NODE,
            "asec_binding",
            composed_contracts.US_COMPOSED_ASEC_BINDING_TYPE,
        ),
        ArtifactInput(
            "arm_rows",
            composed_contracts.BIND_NODE,
            "arm_rows",
            composed_contracts.US_COMPOSED_ASEC_ARM_ROWS_TYPE,
        ),
        ArtifactInput(
            "selected_current_money",
            composed_contracts.BIND_NODE,
            "selected_current_money",
            US_ASEC_SELECTED_MONEY_TYPE,
        ),
        ArtifactInput(
            "prepared_receipt",
            composed_contracts.CREATE_NODE,
            "prepared_receipt",
            US_ASEC_PREPARED_RECEIPT_TYPE,
        ),
        ArtifactInput(
            "income_observations",
            composed_contracts.CREATE_NODE,
            "income_observations",
            income.US_ASEC_INCOME_OBSERVATIONS_TYPE,
        ),
        ArtifactInput(
            "frame_context",
            composed_contracts.LEAVES_NODE,
            "frame_context",
            graph_context.US_FRAME_CONTEXT_TYPE,
        ),
        ArtifactInput(
            "reported_income",
            composed_contracts.REPORTED_INCOME_NODE,
            "reported_income",
            income.US_ASEC_REPORTED_INCOME_TYPE,
        ),
    )


def verify_host(context, frame):
    """Typed actual producer edges plus externally pinned accepted host evidence."""
    require(
        context.params["scope"] == consumer.SCOPE
        and context.params["recipe"].encode()
        == codec.encode_json(consumer.recipe_document())
        and context.params["route"] in ("packaged", "test_fixture"),
        "DIAGNOSTIC_HOST_RECIPE",
    )
    expected = codec.decode_json(context.params["host_edges"].encode())
    types = {
        e.name: e.type
        for e in host_edges(native_binding="native", clone_binding="clones")
    }
    edges = {e.name: e for e in context.node.artifact_inputs if e.name in expected}
    require(set(expected) == set(edges) == set(types), "DIAGNOSTIC_HOST_EDGE_ROSTER")
    values = {}
    for name in edges:
        value = shared.artifact(context, name, types[name])
        require(
            expected[name]
            == {
                "producer_key": value.producer_key,
                "artifact_key": value.key,
                "payload_sha256": codec.sha(value.payload),
            },
            "DIAGNOSTIC_HOST_EDGE_PIN:" + name,
        )
        values[name] = value
    shared.siblings(context, ("asec_binding", "arm_rows", "selected_current_money"))
    shared.siblings(context, ("prepared_receipt", "income_observations"))
    evidence = consumer.qualify_host_population(frame, values)
    evidence["route"] = context.params["route"]
    evidence["actual_producer_edges"] = expected
    return evidence


class DiagnosticMatrixKernel(shared.DetailMatrixKernel, _Kernel):
    ref = "us.puf_diagnostic.matrix@1"
    scope = consumer.SCOPE
    host_binding_name = "source_qualified_host"
    verify_host = staticmethod(verify_host)
    make_matrix = staticmethod(consumer.recipient_matrix)
    implementation_hash = _Kernel.implementation_hash


class DiagnosticAttachKernel(shared.DetailAttachKernel, _Kernel):
    ref = "us.puf_diagnostic.attach@1"
    scope = consumer.SCOPE
    host_binding_name = "source_qualified_host"
    placement_type = PLACEMENT_TYPE
    verify_host = staticmethod(verify_host)
    make_matrix = staticmethod(consumer.recipient_matrix)
    implementation_hash = _Kernel.implementation_hash


def diagnostic_nodes(
    columns, *, base, native_binding, clone_binding, price_pins, host_pins, route
):
    """Extend the complete bound clone pool with one raw diagnostic target."""
    require(route in ("packaged", "test_fixture"), "DIAGNOSTIC_SOURCE_ROUTE")
    donor = PREFIX + ".donor"
    boundary = PREFIX + ".population"
    matrix = PREFIX + ".matrix"
    recipe = consumer.recipe_document()
    fits = legacy_qrf_train_nodes(
        PREFIX + ".fit",
        population=donor,
        entity="tax_unit",
        predictors=detail.FEATURES,
        targets=(detail.TARGET,),
        seed=578,
        n_estimators=2,
        zero_atol=0.0,
        phase=consumer.SCOPE,
    )
    applies = legacy_qrf_apply_matrix_nodes(
        PREFIX + ".apply",
        population=boundary,
        fit_nodes=fits,
        matrix_producer=matrix,
        seed=578,
        phase=consumer.SCOPE,
    )
    grouped = {
        e: tuple(o.column for o in columns if o.entity == e) for e in US_SCHEMA.entities
    }
    inputs = tuple(Slice(e, names) for e, names in grouped.items())
    edges = host_edges(native_binding=native_binding, clone_binding=clone_binding)
    params = {
        "scope": consumer.SCOPE,
        "route": route,
        "recipe": codec.encode_json(recipe).decode(),
        "host_edges": codec.encode_json(host_pins).decode(),
    }
    return (
        Node(
            donor,
            DiagnosticDonorKernel.ref,
            structural=StructuralDelta.CREATE,
            sources=tuple(SOURCE_NAMES.values()),
            outputs=tuple(
                Owned("tax_unit", n, "float64")
                for n in (*detail.FEATURES, detail.TARGET)
            ),
            params={
                "route": route,
                "recipe": codec.encode_json(recipe).decode(),
                "expected_pins": codec.encode_json(price_pins).decode(),
            },
            artifact_outputs=(ArtifactOutput("transport", TRANSPORT_TYPE),),
        ),
        Node(
            boundary,
            shared.DetailBoundaryKernel.ref,
            base=base,
            structural=StructuralDelta.FILTER,
            inputs=(
                Slice(
                    "person", (shared.provenance.support_source_id_column("person"),)
                ),
            ),
        ),
        Node(
            matrix,
            DiagnosticMatrixKernel.ref,
            population=boundary,
            inputs=inputs,
            outputs=(Owned("tax_unit", detail.MASK, "bool"),),
            params=params,
            artifact_inputs=edges,
            artifact_outputs=(
                ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),
                ArtifactOutput("placement", PLACEMENT_TYPE),
            ),
        ),
        *fits,
        *applies,
        Node(
            PREFIX + ".attach",
            DiagnosticAttachKernel.ref,
            population=boundary,
            inputs=tuple(
                Slice(e, (*names, detail.MASK) if e == "tax_unit" else names)
                for e, names in grouped.items()
            ),
            outputs=(Owned("tax_unit", detail.OUTPUT, "float64", rows=detail.MASK),),
            params=params,
            artifact_inputs=(
                *edges,
                ArtifactInput(
                    "matrix", matrix, "matrix", model_input.RECIPIENT_MATRIX_TYPE
                ),
                ArtifactInput("placement", matrix, "placement", PLACEMENT_TYPE),
                ArtifactInput(
                    "raw_draw", applies[-1].id, "raw_draw", codec.RAW_TARGET_TYPE
                ),
                ArtifactInput(
                    "apply_state",
                    applies[-1].id,
                    "apply_state",
                    shared.MATRIX_APPLY_STATE_TYPE,
                ),
            ),
        ),
    )


def register_diagnostic_kernels(registry):
    for kernel in (
        DiagnosticDonorKernel(),
        DiagnosticMatrixKernel(),
        DiagnosticAttachKernel(),
    ):
        registry.register(kernel)


# This host-only route neither declares nor admits a donor, fit, draw or attach.
# It extends the maintained four-node current survey prefix, not the legacy host.
CURRENT_PROJECTION_TYPE = ArtifactType("microcosm.us.survey_puf_host_projection", 1)
CURRENT_PROJECTION_NODE = "puf_diagnostic.current_survey_host"
CURRENT_MATRIX_NODE = "puf_diagnostic.current_survey_matrix"


def qualify_current_survey_host(preparation, allocated_population, clone_population):
    """Check one live current prefix and project its wages once.

    Returning bytes is transport only. Call verify_materialized_current_survey_host
    with actual executor/store outputs on cold and replay boundaries. No budget,
    successor, source capsule or legacy artifact is issued here.
    """
    require(
        type(preparation) is survey_source.AuthenticatedSurveyPopulationPreparation,
        "SURVEY_HOST_PREPARATION_TYPE",
    )
    entry = preparation._checked()  # One full source preparation check.
    state = entry[2]
    view = survey_source.CheckedSurveyPopulationView(
        entry[1], state.context, state.frame, state.plan, json.loads(entry[1])
    )
    # The actual maintained owner checks allocation, all clone cells, membership,
    # pair weights, full receiving state, design anchors and mass ledgers.
    _, allocation = survey_budget._initial(view, allocated_population, clone_population)
    identities = tuple(
        survey_budget._population_identity(p)
        for p in (allocated_population, clone_population)
    )
    asec = survey_source._current_survey_wage_projection(preparation, entry)
    document = {
        "protocol": consumer.CURRENT_SURVEY_PROJECTION_PROTOCOL,
        "route": consumer.CURRENT_SURVEY_ROUTE,
        "preparation_sha256": codec.sha(entry[1]),
        "allocation_sha256": codec.sha(allocation),
        "clone_sha256": survey_source._frame_identity(clone_population.frame),
        "asec": json.loads(asec),
        "release_eligible": False,
    }
    projection = survey_graph._bounded_json(document, consumer.CURRENT_SURVEY_MAX_BYTES)
    matrix, mask, evidence = consumer.current_survey_recipient_matrix(
        clone_population.frame, projection
    )
    # No I/O after the projection; compare complete retained receiving state and
    # actual source issuance after readiness and feature calculation.
    require(
        identities
        == tuple(
            survey_budget._population_identity(p)
            for p in (allocated_population, clone_population)
        ),
        "SURVEY_HOST_FINAL_POPULATION",
    )
    survey_budget._preparation_entry(preparation, entry[1], entry)
    return projection, matrix, mask, evidence


def current_survey_host_edges():
    return (
        ArtifactInput(
            "preparation",
            survey_graph.CREATE_NODE,
            "preparation",
            survey_graph.PREPARATION_TYPE,
        ),
        ArtifactInput(
            "allocation",
            survey_graph.ALLOCATION_NODE,
            "allocation",
            survey_graph.ALLOCATION_TYPE,
        ),
        ArtifactInput(
            "frame_context",
            survey_graph.ALLOCATION_NODE,
            "frame_context",
            graph_context.US_FRAME_CONTEXT_TYPE,
        ),
    )


def _current_params(document):
    return {
        "host_route": consumer.CURRENT_SURVEY_ROUTE,
        **{
            key: document[key]
            for key in ("preparation_sha256", "allocation_sha256", "clone_sha256")
        },
    }


def _current_context_frame(context, expected):
    # The executor puts structural columns before the declared data columns.
    # Validate the full unmasked declaration from the retained Frame first;
    # actual context columns must never choose their own comparison projection.
    ordered = {}
    inputs = []
    for entity in US_SCHEMA.entities:
        structural = [US_SCHEMA.entity_id_column(entity)]
        if entity == US_SCHEMA.person_entity:
            structural.extend(
                US_SCHEMA.membership_column(group) for group in US_SCHEMA.group_entities
            )
        columns = tuple(
            c for c in expected.table(entity).columns if c not in structural
        )
        inputs.append(Slice(entity, columns))
        ordered[entity] = [*structural, *columns]
    require(context.node.inputs == tuple(inputs), "SURVEY_HOST_INPUTS")
    require(set(context.tables) == set(US_SCHEMA.entities), "SURVEY_HOST_TABLES")

    # Match the executor's effective-weight roster, including inherited weights.
    # Ambiguous inherited group weights are omitted by the executor; an invalid
    # explicitly stored weight is an error rather than an absent context entry.
    weights = {}
    for entity in US_SCHEMA.entities:
        try:
            weights[entity] = expected.resolve_weights(entity)
        except ValueError:
            if entity in expected.weighted_entities:
                raise
    require(set(context.weights) == set(weights), "SURVEY_HOST_WEIGHTS")
    require(
        type(context.strata) is type(expected.strata)
        and type(context.strata.name) is type(expected.strata.name)
        and context.strata.name == expected.strata.name,
        "SURVEY_HOST_STRATA_NAME",
    )
    # Use the same all-row boolean selection so index class/name/dtype survive
    # exactly as they do in a KernelContext. Frame normalizes the strata name,
    # which was checked above before either comparison Frame is constructed.
    tables = {
        entity: expected.table(entity).loc[
            [True] * len(expected.table(entity)), ordered[entity]
        ]
        for entity in US_SCHEMA.entities
    }
    strata = expected.strata.loc[[True] * len(expected.strata)]
    actual = Frame(
        dict(context.tables), US_SCHEMA, dict(context.weights), context.strata
    )
    projected = Frame(tables, US_SCHEMA, weights, strata)
    survey_replay.same_replayed_frame(projected, actual)
    # KernelContext has no Population metadata. The mandatory materialized
    # verifier separately checks metadata, ledgers, owners and design anchors.
    return actual


class _CurrentSurveyKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=(
            *survey_graph.STAGE_DEPENDENCIES,
            "scikit-learn",
            "quantile-forest",
        ),
    )

    def implementation_hash(self):
        # No PUF raw/price definitions or acceptance locators are consulted.
        # The reviewed survey inventory remains an independent fail-closed gate.
        return codec.sha(
            codec.encode_json(
                {
                    "survey": survey_graph._Kernel().implementation_hash(),
                    "host": source_hash(
                        sys.modules[__name__],
                        consumer,
                        composed_contracts,
                        shared,
                        shared.qrf,
                        detail,
                        codec,
                        model_input,
                        survey_budget,
                        survey_replay,
                        consumer.current_money,
                        dependencies=self.capabilities.dependencies,
                    ),
                }
            )
        )


class CurrentSurveyHostProjectionKernel(_CurrentSurveyKernel):
    ref = "us.puf_diagnostic.current_survey_host@1"

    def __init__(self, preparation, allocated_population, clone_population):
        self.preparation = preparation
        self.allocated_population = allocated_population
        self.clone_population = clone_population

    def run(self, context):
        require(
            context.node.id == CURRENT_PROJECTION_NODE
            and context.node.population == survey_clone.COMBINED_CLONE_NODE
            and context.node.artifact_inputs == current_survey_host_edges()
            and context.node.artifact_outputs
            == (ArtifactOutput("projection", CURRENT_PROJECTION_TYPE),)
            and not context.node.outputs
            and not context.sources
            and set(context.artifacts)
            == {"preparation", "allocation", "frame_context"},
            "SURVEY_HOST_DECLARATION",
        )
        projection, _, _, evidence = qualify_current_survey_host(
            self.preparation, self.allocated_population, self.clone_population
        )
        document = codec.decode_json(projection)
        require(
            set(context.params) == set(_current_params(document)) | {"host_edges"},
            "SURVEY_HOST_PARAMS",
        )
        require(
            {k: context.params[k] for k in _current_params(document)}
            == _current_params(document),
            "SURVEY_HOST_PARAMS",
        )
        pins_bytes = context.params["host_edges"].encode()
        require(len(pins_bytes) <= 4096, "SURVEY_HOST_EDGE_PIN_BOUND")
        pins = codec.decode_json(pins_bytes)
        require(
            set(pins) == {e.name for e in current_survey_host_edges()},
            "SURVEY_HOST_EDGE_PIN_ROSTER",
        )
        _current_context_frame(context, self.clone_population.frame)
        prepared = shared.artifact(
            context, "preparation", survey_graph.PREPARATION_TYPE
        )
        allocated = shared.artifact(context, "allocation", survey_graph.ALLOCATION_TYPE)
        frame_context = shared.artifact(
            context, "frame_context", graph_context.US_FRAME_CONTEXT_TYPE
        )
        shared.siblings(context, ("allocation", "frame_context"))
        for edge, value in zip(
            current_survey_host_edges(),
            (prepared, allocated, frame_context),
            strict=True,
        ):
            require(
                pins[edge.name]
                == {
                    "producer_key": value.producer_key,
                    "artifact_key": value.key,
                    "payload_sha256": codec.sha(value.payload),
                },
                "SURVEY_HOST_EDGE_PIN",
            )
        require(
            prepared.payload == self.preparation.payload
            and codec.sha(allocated.payload) == document["allocation_sha256"]
            and codec.sha(frame_context.payload)
            == codec.decode_json(allocated.payload)["output_context_sha256"],
            "SURVEY_HOST_EDGE_BYTES",
        )
        return KernelResult(artifacts={"projection": projection}, receipt=evidence)


class CurrentSurveyMatrixKernel(_CurrentSurveyKernel):
    ref = "us.puf_diagnostic.current_survey_matrix@1"

    def run(self, context):
        require(
            context.node.id == CURRENT_MATRIX_NODE
            and context.node.population == survey_clone.COMBINED_CLONE_NODE
            and context.node.artifact_inputs
            == (
                ArtifactInput(
                    "projection",
                    CURRENT_PROJECTION_NODE,
                    "projection",
                    CURRENT_PROJECTION_TYPE,
                ),
            )
            and context.node.artifact_outputs
            == (ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),)
            and not context.node.outputs
            and not context.sources
            and set(context.artifacts) == {"projection"},
            "SURVEY_MATRIX_DECLARATION",
        )
        projection = shared.artifact(
            context, "projection", CURRENT_PROJECTION_TYPE
        ).payload
        require(
            type(projection) is bytes
            and len(projection) <= consumer.CURRENT_SURVEY_MAX_BYTES,
            "SURVEY_HOST_PROJECTION_BOUND",
        )
        require(
            dict(context.params) == _current_params(codec.decode_json(projection)),
            "SURVEY_MATRIX_PARAMS",
        )
        # Do not strip any carried cells; attachment/mask columns are not produced.
        frame = Frame(
            dict(context.tables), US_SCHEMA, dict(context.weights), context.strata
        )
        matrix, _, evidence = consumer.current_survey_recipient_matrix(
            frame, projection
        )
        return KernelResult(artifacts={"matrix": matrix}, receipt=evidence)


def current_survey_host_nodes(
    columns, *, preparation_sha256, allocation_sha256, clone_sha256, host_pins
):
    """Two artifact-only nodes over the existing pre-calibration clone version.

    Digest arguments name values; they do not authenticate them. This deliberately
    cannot be substituted into the eleven-edge legacy donor/attachment factory.
    """
    document = dict(
        preparation_sha256=preparation_sha256,
        allocation_sha256=allocation_sha256,
        clone_sha256=clone_sha256,
    )
    require(all(codec._hash(v) for v in document.values()), "SURVEY_HOST_DIGEST")
    require(columns and all(type(o) is Owned for o in columns), "SURVEY_HOST_COLUMNS")
    require(
        len({(o.entity, o.column) for o in columns}) == len(columns),
        "SURVEY_HOST_COLUMNS",
    )
    inputs = tuple(
        Slice(e, tuple(o.column for o in columns if o.entity == e))
        for e in US_SCHEMA.entities
    )
    require(all(item.columns for item in inputs), "SURVEY_HOST_COLUMNS")
    params = _current_params(document)
    require(
        set(host_pins) == {e.name for e in current_survey_host_edges()},
        "SURVEY_HOST_EDGE_PIN_ROSTER",
    )
    for pin in host_pins.values():
        require(
            set(pin) == {"producer_key", "artifact_key", "payload_sha256"}
            and all(codec._hash(value) for value in pin.values()),
            "SURVEY_HOST_EDGE_PIN",
        )
    encoded_pins = codec.encode_json(host_pins)
    require(len(encoded_pins) <= 4096, "SURVEY_HOST_EDGE_PIN_BOUND")
    return (
        Node(
            CURRENT_PROJECTION_NODE,
            CurrentSurveyHostProjectionKernel.ref,
            population=survey_clone.COMBINED_CLONE_NODE,
            inputs=inputs,
            params={**params, "host_edges": encoded_pins.decode()},
            artifact_inputs=current_survey_host_edges(),
            artifact_outputs=(ArtifactOutput("projection", CURRENT_PROJECTION_TYPE),),
        ),
        Node(
            CURRENT_MATRIX_NODE,
            CurrentSurveyMatrixKernel.ref,
            population=survey_clone.COMBINED_CLONE_NODE,
            inputs=inputs,
            params=params,
            artifact_inputs=(
                ArtifactInput(
                    "projection",
                    CURRENT_PROJECTION_NODE,
                    "projection",
                    CURRENT_PROJECTION_TYPE,
                ),
            ),
            artifact_outputs=(
                ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),
            ),
        ),
    )


def verify_materialized_current_survey_host(
    preparation,
    allocated_population,
    clone_population,
    *,
    population,
    projection,
    matrix,
):
    """Mandatory cold/replay check of actual outputs, not manifest receipt labels.

    The integrating runner first checks typed descriptors and actual content-store
    keys, then supplies their loaded bytes and the executor-observed Population.
    This verifier reconstructs live source projection once and compares the full
    materialized Population after that last source I/O.
    """
    expected_projection, expected_matrix, _, evidence = qualify_current_survey_host(
        preparation, allocated_population, clone_population
    )
    require(
        type(projection) is bytes and projection == expected_projection,
        "SURVEY_HOST_MATERIALIZED_PROJECTION",
    )
    require(
        type(matrix) is bytes and matrix == expected_matrix,
        "SURVEY_HOST_MATERIALIZED_MATRIX",
    )
    survey_replay.same_replayed_population(clone_population, population)
    return evidence
