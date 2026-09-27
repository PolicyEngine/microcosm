"""A fixture-only price → donor → matrix → raw draw → masked placement graph.

These kernels do not grant genuine source or fit admission. Existing source,
growth and targetwise QRF kernels perform all parsing, growth, fit and draw math.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from importlib import import_module

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import graph_legacy_apply_matrix, model_input, qrf
from microcosm.fit.graph_legacy_apply_matrix import (
    MATRIX_APPLY_STATE_TYPE,
    decode_matrix_apply_state,
)
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights
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
    source_hash,
)
from microcosm.graph.keys import opaque_artifact_key

from . import puf_detail_transfer as detail
from . import puf_growth as growth
from . import puf_growth_graph as growth_graph
from . import puf_monetary_agi_projection as agi
from . import puf_monetary_source as monetary
from . import puf_price_baseline as price
from . import puf_raw_source as raw
from . import support_provenance as provenance

require = detail.require
PRICE_TYPE = ArtifactType("microcosm.us.fixture_puf_price_arrays", 1)
PRICE_BINDING_TYPE = ArtifactType("microcosm.us.fixture_puf_price_binding", 1)
HOST_TYPE = ArtifactType("microcosm.us.fixture_combined_host", 1)
PLACEMENT_TYPE = ArtifactType("microcosm.us.fixture_puf_detail_placement", 1)
SOURCE_EDGES = (
    ArtifactInput(
        "status", raw.PUF_RAW_SOURCE_NODE, "return_status", raw.RETURN_STATUS_TYPE
    ),
    ArtifactInput(
        "projection",
        agi.AGI_PROJECTION_NODE,
        "monetary_agi_projection",
        agi.AGI_PROJECTION_TYPE,
    ),
)


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def artifact(context, alias, expected_type):
    edges = [e for e in context.node.artifact_inputs if e.name == alias]
    require(len(edges) == 1, "DETAIL_ARTIFACT_DECLARATION")
    edge, value = edges[0], context.artifacts[alias]
    require(
        edge.type == value.type == expected_type
        and value.key == opaque_artifact_key(value.producer_key, edge.artifact),
        "DETAIL_ARTIFACT_IDENTITY",
    )
    return value


def siblings(context, names):
    edges = {e.name: e for e in context.node.artifact_inputs}
    require(
        len({edges[name].producer for name in names}) == 1
        and len({context.artifacts[name].producer_key for name in names}) == 1,
        "DETAIL_SIBLING_PRODUCER",
    )


class _Kernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self):
        # Separate extension identity. Never edit the shared upstream inventory.
        base = source_hash(
            sys.modules[__name__],
            detail,
            raw,
            agi,
            monetary,
            price,
            growth,
            growth_graph,
            codec,
            model_input,
            graph_legacy_apply_matrix,
            qrf,
            provenance,
            detail.origin,
            detail.origin.acs,
            *(import_module(name) for name in growth_graph._IMPLEMENTATION_MODULES),
            dependencies=self.capabilities.dependencies,
        )
        return codec.sha(
            codec.encode_json(
                {
                    "modules": base,
                    "csv_profile": raw.csv_acceptance_profile(),
                    "source_definition": raw.packaged_definition().sha256,
                    "monetary_definition": monetary.packaged_projection().sha256,
                    "agi_definition": agi.packaged_agi_projection().sha256,
                }
            )
        )


def fixture_inputs(context):
    """Consume authenticated fixture source artifacts and the actual CPI codec bytes."""
    params = context.params
    require(params["scope"] == detail.SCOPE, "GENUINE_TRANSFER_NOT_ADMITTED")
    definition, projection = detail.fixture_sources(
        params["definition"], params["projection"]
    )
    status_value = artifact(context, "status", raw.RETURN_STATUS_TYPE)
    projection_value = artifact(context, "projection", agi.AGI_PROJECTION_TYPE)
    status = raw.decode_return_status(status_value.payload, definition)
    decoded = agi.decode_agi_projection(
        projection_value.payload,
        projection,
        expected_status_artifact_sha256=codec.sha(status_value.payload),
    )
    cpi = context.sources["fixture_cpi"].read_bytes()
    resource = codec.decode_json(params["resource"].encode())
    compiled = price.compile_price_baseline(
        detail.MONEY_FIELDS, resource_bytes=cpi, **resource
    )
    adaptation = price.adapt_projection_to_money_view(
        decoded, status, compiled, resource_bytes=cpi
    )
    require(
        compiled.release_eligible is False
        and adaptation.excluded["rows"] == 4
        and len(set(adaptation.excluded["recid"])) == 4
        and set(adaptation.excluded["recid"]) == set(raw.PUF_AGGREGATE_RECIDS),
        "FIXTURE_PRICE_UNIVERSE",
    )
    expected = {
        "definition": definition.canonical,
        "projection_definition": projection.canonical,
        "status": status_value.payload,
        "projection": projection_value.payload,
    }
    inputs = codec.decode_json(params["inputs"].encode())
    require(set(inputs) == set(expected), "FIXTURE_INPUT_ROSTER")
    for name, body in expected.items():
        require(
            inputs[name]["sha256"] == codec.sha(body)
            and inputs[name]["bytes"] == len(body),
            "FIXTURE_SOURCE_BINDING",
        )
    return definition, projection, status, decoded, compiled, adaptation


class FixturePriceAdaptKernel(_Kernel):
    ref = "us.fixture_puf_detail.price_adapt@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context):
        _, projection, _, _, compiled, adapted = fixture_inputs(context)
        table = adapted.money_view.copy()
        for name in compiled.money_fields:
            require(
                np.all(np.abs(table[name].to_numpy(dtype=object)) <= 2**53),
                "FIXTURE_EXACT_MONEY_FLOAT",
            )
            table[name] = table[name].astype("float64")
        require(np.all(adapted.design_weight <= 2**53), "FIXTURE_EXACT_WEIGHT_FLOAT")
        table["S006"] = adapted.design_weight.astype("float64")
        table["FLPDYR"] = adapted.flpdyr.astype("int32")
        table["demographic_status"] = adapted.demographic_status.astype("int32")
        ids = np.arange(1, len(table) + 1, dtype="int64")
        table.insert(0, "tax_unit_id", ids)
        return KernelResult(
            frame=Frame(
                {
                    "tax_unit": table,
                    "person": pd.DataFrame(
                        {
                            "person_id": ids,
                            "person_tax_unit_id": ids,
                            "is_primary_filer": np.ones(len(ids), dtype=bool),
                        }
                    ),
                },
                EntitySchema(group_entities=("tax_unit",)),
                {
                    "tax_unit": Weights(
                        adapted.design_weight.astype("float64"), WeightKind.DESIGN
                    )
                },
            ),
            receipt={
                "scope": detail.SCOPE,
                "source_projection": projection.canonical.decode("utf-8"),
                "source_admission": detail.SOURCE_ADMISSION,
                "money_view": plain(adapted.receipt),
            },
        )


class FixturePriceExportKernel(_Kernel):
    ref = "us.fixture_puf_detail.price_export@1"

    def run(self, context):
        definition, projection, status, decoded, compiled, adapted = fixture_inputs(
            context
        )
        table = context.tables["tax_unit"]
        arrays = {
            name: table[name].to_numpy(copy=True)
            for name in detail.ARRAY_DTYPES
            if "_delivered_" not in name
        }
        arrays.update(
            S006_delivered_int64=adapted.design_weight.copy(),
            FLPDYR_delivered_int16=adapted.flpdyr.copy(),
            demographic_status_delivered_int8=adapted.demographic_status.copy(),
        )
        # Exact dea helper envelope; this fixture producer states its own actual
        # code identities. It never impersonates a completed genuine price run.
        header = {
            "schema": "local.puf.price_restatement.arrays.v1",
            "input_binding_sha256": codec.sha(context.params["inputs"].encode()),
            "contract_sha256": compiled.sha256,
            "source_head": context.params["code_head"],
            "helper_head": context.params["code_head"],
            "release_eligible": False,
        }
        identities = {
            n: {"dtype": v.dtype.str, "rows": len(v), "sha256": codec.sha(v.tobytes())}
            for n, v in sorted(arrays.items())
        }
        encoded = codec.encode_json({**header, "columns": identities})
        payload = (
            detail.MAGIC
            + len(encoded).to_bytes(8, "big")
            + encoded
            + b"".join(
                np.ascontiguousarray(arrays[n]).tobytes() for n in sorted(arrays)
            )
        )
        binding = {
            "scope": detail.SCOPE,
            "payload_sha256": codec.sha(payload),
            "header": header,
            "definition_sha256": definition.sha256,
            "projection_sha256": projection.sha256,
            "status_sha256": codec.sha(context.artifacts["status"].payload),
            "source_projection_document": projection.canonical.decode("utf-8"),
            "factor_document": compiled.factors.document.decode("utf-8"),
            "recipe": compiled.recipe.identity.decode("utf-8"),
            "resource": codec.decode_json(context.params["resource"].encode()),
        }
        # Exercise the import at production time too, without doing growth twice.
        checked = detail.decode_price_arrays(
            payload, expected=header, expected_sha256=codec.sha(payload)
        )
        detail.donor_frame(checked, status, decoded, projection)
        return KernelResult(
            artifacts={"price": payload, "price_binding": codec.encode_json(binding)},
            receipt={
                "scope": detail.SCOPE,
                "price_payload_sha256": codec.sha(payload),
                "source_admission": detail.SOURCE_ADMISSION,
                "release_eligible": False,
            },
        )


class FixtureDonorKernel(_Kernel):
    ref = "us.fixture_puf_detail.donor@1"
    capabilities = FixturePriceAdaptKernel.capabilities

    def run(self, context):
        definition, projection, status, decoded, compiled, _ = fixture_inputs(context)
        value = artifact(context, "price", PRICE_TYPE)
        bound = artifact(context, "price_binding", PRICE_BINDING_TYPE)
        siblings(context, ("price", "price_binding"))
        binding = codec.decode_json(bound.payload)
        require(
            binding["scope"] == detail.SCOPE
            and binding["definition_sha256"] == definition.sha256
            and binding["projection_sha256"] == projection.sha256
            and binding["status_sha256"]
            == codec.sha(context.artifacts["status"].payload)
            and binding["source_projection_document"]
            == projection.canonical.decode("utf-8")
            and binding["factor_document"] == compiled.factors.document.decode("utf-8")
            and binding["recipe"] == compiled.recipe.identity.decode("utf-8")
            and binding["resource"]
            == codec.decode_json(context.params["resource"].encode()),
            "DONOR_PRICE_BINDING",
        )
        header = binding["header"]
        require(
            header["source_head"]
            == header["helper_head"]
            == context.params["code_head"]
            and header["contract_sha256"] == compiled.sha256
            and header["input_binding_sha256"]
            == codec.sha(context.params["inputs"].encode()),
            "DONOR_PRICE_AUTHORITY",
        )
        arrays = detail.decode_price_arrays(
            value.payload, expected=header, expected_sha256=binding["payload_sha256"]
        )
        frame = detail.donor_frame(arrays, status, decoded, projection)
        return KernelResult(
            frame=frame,
            receipt={
                "scope": detail.SCOPE,
                "source_admission": detail.SOURCE_ADMISSION,
                "source_projection_document": projection.canonical.decode("utf-8"),
                "donor_design_weight_units": "delivered_S006_integer_hundredths",
                "release_eligible": False,
            },
        )


class DetailBoundaryKernel(_Kernel):
    ref = "us.fixture_puf_detail.boundary@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context):
        return KernelResult(
            keep=pd.Series(True, index=context.tables["person"].person_id, dtype=bool)
        )


def context_frame(context):
    tables = {
        e: context.tables[e].drop(columns=[detail.MASK, detail.OUTPUT], errors="ignore")
        if e == "tax_unit"
        else context.tables[e]
        for e in US_SCHEMA.entities
    }
    return Frame(tables, US_SCHEMA, dict(context.weights), context.strata)


def verify_host_scope(context, frame):
    value = artifact(context, "fixture_host", HOST_TYPE)
    scope = codec.decode_json(value.payload)
    require(
        set(scope) == {"scope", "native_content_sha256", "source_fixture_sha256"}
        and scope["scope"] == detail.SCOPE
        and scope["source_fixture_sha256"] == context.params["source_fixture_sha256"],
        "HOST_FIXTURE_AUTHORITY",
    )
    native_tables = {
        e: frame.table(e)
        .loc[frame.table(e)[provenance.support_clone_index_column(e)].eq(0)]
        .reset_index(drop=True)
        for e in frame.entities
    }
    native_person = (
        frame.person[provenance.support_clone_index_column("person")].eq(0).to_numpy()
    )
    native_hh = (
        frame.table("household")[provenance.support_clone_index_column("household")]
        .eq(0)
        .to_numpy()
    )
    require(
        frame.weights_for("household").kind is WeightKind.IMPORTANCE, "HOST_WEIGHT_KIND"
    )
    native = Frame(
        native_tables,
        US_SCHEMA,
        {
            "household": Weights(
                frame.weights_for("household").values[native_hh] * 2,
                WeightKind.IMPORTANCE,
            )
        },
        frame.strata.loc[native_person].reset_index(drop=True),
    )
    require(
        detail.population_content(native) == scope["native_content_sha256"],
        "HOST_NATIVE_SOURCE_BINDING",
    )
    return scope


class DetailMatrixKernel(_Kernel):
    ref = "us.fixture_puf_detail.matrix@1"
    scope = detail.SCOPE
    host_binding_name = "fixture_host"
    verify_host = staticmethod(verify_host_scope)
    make_matrix = staticmethod(detail.recipient_matrix)

    def run(self, context):
        require(context.params["scope"] == self.scope, "GENUINE_TRANSFER_NOT_ADMITTED")
        frame = context_frame(context)
        scope = self.verify_host(context, frame)
        payload, mask = self.make_matrix(frame)
        binding = {
            "scope": self.scope,
            "population": context.node.population,
            "host_content_sha256": detail.population_content(frame),
            self.host_binding_name: scope,
            "matrix_sha256": codec.sha(payload),
            "features": list(detail.FEATURES),
            "targets": [detail.TARGET],
            "output": detail.OUTPUT,
            "selected_ids": frame.table("tax_unit")
            .tax_unit_id.to_numpy()[mask]
            .tolist(),
        }
        return KernelResult(
            columns={
                ("tax_unit", detail.MASK): pd.Series(
                    mask, index=frame.table("tax_unit").tax_unit_id, dtype=bool
                )
            },
            artifacts={"matrix": payload, "placement": codec.encode_json(binding)},
            receipt={
                "scope": self.scope,
                "recipient_rows": int(mask.sum()),
                "zero_weight_rows_retained": True,
                "source_admission": detail.SOURCE_ADMISSION,
                "release_eligible": False,
            },
        )


class DetailAttachKernel(_Kernel):
    ref = "us.fixture_puf_detail.attach@1"
    scope = detail.SCOPE
    host_binding_name = "fixture_host"
    placement_type = PLACEMENT_TYPE
    verify_host = staticmethod(verify_host_scope)
    make_matrix = staticmethod(detail.recipient_matrix)

    def run(self, context):
        require(
            context.params["scope"] == self.scope
            and context.node.outputs
            == (Owned("tax_unit", detail.OUTPUT, "float64", rows=detail.MASK),),
            "DETAIL_OWNERSHIP",
        )
        frame = context_frame(context)
        scope = self.verify_host(context, frame)
        matrix = artifact(context, "matrix", model_input.RECIPIENT_MATRIX_TYPE)
        placement = artifact(context, "placement", self.placement_type)
        raw_draw = artifact(context, "raw_draw", codec.RAW_TARGET_TYPE)
        state = artifact(context, "apply_state", MATRIX_APPLY_STATE_TYPE)
        siblings(context, ("matrix", "placement"))
        siblings(context, ("raw_draw", "apply_state"))
        binding = codec.decode_json(placement.payload)
        expected, mask = self.make_matrix(frame)
        require(
            np.array_equal(context.tables["tax_unit"][detail.MASK], mask),
            "DETAIL_MASK_CHANGED",
        )
        require(
            matrix.payload == expected
            and binding
            == {
                "scope": self.scope,
                "population": context.node.population,
                "host_content_sha256": detail.population_content(frame),
                self.host_binding_name: scope,
                "matrix_sha256": codec.sha(expected),
                "features": list(detail.FEATURES),
                "targets": [detail.TARGET],
                "output": detail.OUTPUT,
                "selected_ids": frame.table("tax_unit")
                .tax_unit_id.to_numpy()[mask]
                .tolist(),
            },
            "DETAIL_PLACEMENT_BINDING",
        )
        packet = decode_matrix_apply_state(state.payload)
        require(
            packet["matrix_sha256"] == codec.sha(matrix.payload)
            and packet["matrix_producer_key"] == matrix.producer_key,
            "DETAIL_MATRIX_PRODUCER",
        )
        application, chain = codec.read_application(
            codec.encode_json(packet["application"])
        )
        require(
            tuple(chain.completed_targets) == (detail.TARGET,)
            and tuple(chain.predictors) == detail.FEATURES
            and application["seed"] == 578
            and application["raw_targets"]
            == [{"target": detail.TARGET, "sha256": codec.sha(raw_draw.payload)}],
            "DETAIL_RAW_STATE_BINDING",
        )
        prepared = model_input.decode_recipient_matrix(matrix.payload)
        require(
            chain.entity == prepared.entity == "tax_unit"
            and tuple(chain.targets) == (detail.TARGET,)
            and chain.recipient_index == qrf._index_identity(prepared.features.index),
            "DETAIL_STATE_ENTITY_INDEX",
        )
        values = codec.read_raw_target(
            raw_draw.payload, target=detail.TARGET, index=prepared.features.index
        )
        return KernelResult(
            columns={
                ("tax_unit", detail.OUTPUT): pd.Series(
                    values, index=prepared.entity_ids, dtype="float64"
                )
            },
            receipt={
                "scope": self.scope,
                "raw_sha256": codec.sha(raw_draw.payload),
                "matrix_producer_key": matrix.producer_key,
                "application_producer_key": state.producer_key,
                "native_output_cells_written": 0,
                "host_weights_changed": False,
                "release_eligible": False,
            },
        )


def host_nodes(
    columns,
    *,
    base,
    host_producer,
    source_fixture_sha256,
    fit_nodes,
    prefix="fixture_puf_detail",
):
    """Declare the clone-only matrix and exact raw attachment after whole-household cloning."""
    from microcosm.fit.graph_legacy_qrf import legacy_qrf_apply_matrix_nodes

    grouped = {
        e: tuple(o.column for o in columns if o.entity == e) for e in US_SCHEMA.entities
    }
    inputs = tuple(Slice(e, values) for e, values in grouped.items())
    boundary = prefix + ".population"
    matrix_id = prefix + ".matrix"
    host_edge = ArtifactInput("fixture_host", host_producer, "fixture_host", HOST_TYPE)
    params = {"scope": detail.SCOPE, "source_fixture_sha256": source_fixture_sha256}
    matrix = Node(
        matrix_id,
        DetailMatrixKernel.ref,
        population=boundary,
        inputs=inputs,
        outputs=(Owned("tax_unit", detail.MASK, "bool"),),
        params=params,
        artifact_inputs=(host_edge,),
        artifact_outputs=(
            ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),
            ArtifactOutput("placement", PLACEMENT_TYPE),
        ),
    )
    applies = legacy_qrf_apply_matrix_nodes(
        prefix + ".apply",
        population=boundary,
        fit_nodes=fit_nodes,
        matrix_producer=matrix_id,
        seed=578,
        phase=detail.SCOPE,
    )
    attach_inputs = tuple(
        Slice(e, (*values, detail.MASK) if e == "tax_unit" else values)
        for e, values in grouped.items()
    )
    attach = Node(
        prefix + ".attach",
        DetailAttachKernel.ref,
        population=boundary,
        inputs=attach_inputs,
        params=params,
        outputs=(Owned("tax_unit", detail.OUTPUT, "float64", rows=detail.MASK),),
        artifact_inputs=(
            host_edge,
            ArtifactInput(
                "matrix", matrix_id, "matrix", model_input.RECIPIENT_MATRIX_TYPE
            ),
            ArtifactInput("placement", matrix_id, "placement", PLACEMENT_TYPE),
            ArtifactInput(
                "raw_draw", applies[-1].id, "raw_draw", codec.RAW_TARGET_TYPE
            ),
            ArtifactInput(
                "apply_state", applies[-1].id, "apply_state", MATRIX_APPLY_STATE_TYPE
            ),
        ),
    )
    return (
        Node(
            boundary,
            DetailBoundaryKernel.ref,
            base=base,
            structural=StructuralDelta.FILTER,
            inputs=(Slice("person", (provenance.support_source_id_column("person"),)),),
        ),
        matrix,
        *applies,
        attach,
    )


def verify_materialized_transfer(
    before,
    after,
    binding_payload,
    *,
    before_design,
    after_design,
    before_ledger,
    after_ledger,
    boundary_node,
):
    """Detect omitted projected columns and any non-owned materialized change."""
    binding = codec.decode_json(binding_payload)
    require(
        detail.population_content(before) == binding["host_content_sha256"],
        "MATERIALIZED_HOST_BINDING",
    )
    tables = {
        e: after.table(e).drop(columns=[detail.MASK, detail.OUTPUT], errors="ignore")
        if e == "tax_unit"
        else after.table(e)
        for e in after.entities
    }
    restored = Frame(
        tables,
        after.schema,
        {e: after.weights_for(e) for e in after.weighted_entities},
        after.strata,
    )
    require(
        detail.population_content(restored) == detail.population_content(before),
        "MATERIALIZED_NONOWNED_CHANGED",
    )
    require(before.mass_log == after.mass_log, "MATERIALIZED_FRAME_LEDGER_CHANGED")
    require(before_design.keys() == after_design.keys(), "MATERIALIZED_DESIGN_ROSTER")
    for entity in before_design:
        require(
            np.array_equal(
                before_design[entity].view("uint64"),
                after_design[entity].view("uint64"),
            ),
            "MATERIALIZED_DESIGN_CHANGED",
        )
    require(
        bool(before_ledger)
        and tuple(after_ledger) == tuple(before_ledger)
        and after_ledger[-1].node_id == boundary_node,
        "MATERIALIZED_GRAPH_LEDGER_CHANGED",
    )
    native = after.table("tax_unit")[
        provenance.support_clone_index_column("tax_unit")
    ].eq(0)
    require(
        after.table("tax_unit").loc[native, detail.OUTPUT].isna().all(),
        "MATERIALIZED_NATIVE_OUTPUT",
    )
    return {
        "all_nonowned_preserved": True,
        "native_output_null": True,
        "design_anchors_preserved": True,
        "design_anchor_entities": len(before_design),
        "graph_ledger_preserved": True,
        "scope": binding["scope"],
    }


def price_nodes(params, compiled):
    """Real growth and strict donor import, independent of any survey host."""
    inputs = ("fixture_cpi",)
    adapted = "fixture_puf_price.adapted"
    export = "fixture_puf_price.export"
    grown = growth_graph.us_puf_growth_nodes(
        compiled,
        base=adapted,
        entity="tax_unit",
        person_boundary_columns=("is_primary_filer",),
    )
    money_columns = (
        "RECID",
        *detail.MONEY_FIELDS,
        "S006",
        "FLPDYR",
        "demographic_status",
    )
    adapted_outputs = (
        Owned("person", "is_primary_filer", "bool"),
        *(
            Owned(
                "tax_unit",
                name,
                "int64"
                if name == "RECID"
                else "int32"
                if name in ("FLPDYR", "demographic_status")
                else "float64",
            )
            for name in money_columns
        ),
    )
    export_columns = tuple(n for n in detail.ARRAY_DTYPES if "_delivered_" not in n)
    donor_outputs = tuple(
        Owned("tax_unit", name, "float64") for name in (*detail.FEATURES, detail.TARGET)
    )
    return (
        Node(
            adapted,
            FixturePriceAdaptKernel.ref,
            structural=StructuralDelta.CREATE,
            sources=inputs,
            params=params,
            artifact_inputs=SOURCE_EDGES,
            outputs=adapted_outputs,
        ),
        *grown,
        Node(
            export,
            FixturePriceExportKernel.ref,
            population=growth_graph.PUF_GROWTH_BOUNDARY_NODE,
            inputs=(Slice("tax_unit", export_columns),),
            sources=inputs,
            params=params,
            artifact_inputs=SOURCE_EDGES,
            artifact_outputs=(
                ArtifactOutput("price", PRICE_TYPE),
                ArtifactOutput("price_binding", PRICE_BINDING_TYPE),
            ),
        ),
        Node(
            "fixture_puf_donor",
            FixtureDonorKernel.ref,
            structural=StructuralDelta.CREATE,
            outputs=donor_outputs,
            sources=inputs,
            params=params,
            artifact_inputs=(
                *SOURCE_EDGES,
                ArtifactInput("price", export, "price", PRICE_TYPE),
                ArtifactInput(
                    "price_binding", export, "price_binding", PRICE_BINDING_TYPE
                ),
            ),
        ),
    )
