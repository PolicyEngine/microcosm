"""Source-qualified development diagnostic, preserving the fixture-only path.

The runner must first perform the bounded reader/feature check before allowing
fit. Graph fit keys deliberately have no host ancestry: changing recipient
features can reuse the same donor model. No source metadata is promoted.
"""

from __future__ import annotations

import sys

from microcosm.fit import model_input
from microcosm.fit.graph_legacy_qrf import (
    legacy_qrf_apply_matrix_nodes,
    legacy_qrf_train_nodes,
)
from microcosm.frame import US_SCHEMA
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
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
from . import graph_composed_asec_binding as binding
from . import graph_composed_asec_measures as measures
from . import graph_composed_population as composed
from . import graph_context
from . import graph_native_household_origin as origins
from . import graph_puf_detail_transfer as shared
from . import puf_diagnostic_consumer as consumer
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
        return codec.sha(
            codec.encode_json(
                {
                    "consumer": source_hash(
                        sys.modules[__name__],
                        consumer,
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
        pins = shared.plain(context.params["expected_pins"])
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
            recipe=shared.plain(context.params["recipe"]),
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
            binding.BIND_NODE,
            "asec_binding",
            binding.US_COMPOSED_ASEC_BINDING_TYPE,
        ),
        ArtifactInput(
            "arm_rows",
            binding.BIND_NODE,
            "arm_rows",
            binding.US_COMPOSED_ASEC_ARM_ROWS_TYPE,
        ),
        ArtifactInput(
            "selected_current_money",
            binding.BIND_NODE,
            "selected_current_money",
            US_ASEC_SELECTED_MONEY_TYPE,
        ),
        ArtifactInput(
            "prepared_receipt",
            composed.CREATE_NODE,
            "prepared_receipt",
            US_ASEC_PREPARED_RECEIPT_TYPE,
        ),
        ArtifactInput(
            "income_observations",
            composed.CREATE_NODE,
            "income_observations",
            income.US_ASEC_INCOME_OBSERVATIONS_TYPE,
        ),
        ArtifactInput(
            "frame_context",
            measures.LEAVES_NODE,
            "frame_context",
            graph_context.US_FRAME_CONTEXT_TYPE,
        ),
        ArtifactInput(
            "reported_income",
            measures.REPORTED_INCOME_NODE,
            "reported_income",
            income.US_ASEC_REPORTED_INCOME_TYPE,
        ),
    )


def verify_host(context, frame):
    """Typed actual producer edges plus externally pinned accepted host evidence."""
    require(
        context.params["scope"] == consumer.SCOPE
        and codec.encode_json(shared.plain(context.params["recipe"]))
        == codec.encode_json(consumer.recipe_document())
        and context.params["route"] in ("packaged", "test_fixture"),
        "DIAGNOSTIC_HOST_RECIPE",
    )
    expected = shared.plain(context.params["host_edges"])
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
        "recipe": recipe,
        "host_edges": host_pins,
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
            params={"route": route, "recipe": recipe, "expected_pins": price_pins},
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
