"""Typed source-to-probability-to-report fragment; no beneficiary input writes.

The private boundary borrows an existing genuine preparation. It must be
validated before graph/store work and after the final I/O, including replay.
No result, receipt or reconstructed Frame establishes a source capability.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import FunctionType

import numpy as np
import pandas as pd

from microcosm.fit import categorical, model_input
from microcosm.fit import graph_categorical as fitted
from microcosm.frame import weights as frame_weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    Determinism,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
    load_source,
    platform_fingerprint,
    source_hash,
)
from microcosm.graph import population as population_ops
from microcosm.graph.keys import _capabilities_projection, opaque_artifact_key

from . import current_survey_ss_completion as values

source = values.source
codec = values.codec
require = values.require
SOURCE_NAME = "survey_ss_full_original_asec"
SOURCE_NODE = "survey_ss_report.full_source"
COLUMNS_NODE = "survey_ss_report.model_columns"
DONOR_NODE = "survey_ss_report.resolved_donors"
MATRIX_NODE = "survey_ss_report.original_matrix"
FIT_NODE = "survey_ss_report.fit"
PROBABILITY_NODE = "survey_ss_report.probabilities"
REPORT_NODE = "survey_ss_report.complete"
PROJECTION_TYPE = ArtifactType("microcosm.us.survey_ss_report_projection", 1)
REPORT_TYPE = ArtifactType("microcosm.us.survey_ss_completed_reports", 1)
_MAGIC = b"microcosm.us.survey_ss_completed_reports/1\n"
_HEADER_MAX = 1024 * 1024


def _edge(name, artifact):
    return ArtifactInput(name, SOURCE_NODE, artifact, PROJECTION_TYPE)


def _donor_edge():
    return _edge("donor_projection", "donor_projection")


def _recipient_edge():
    return _edge("recipient_projection", "recipient_projection")


def ss_report_completion_nodes(qualified, *, seed, config):
    """Build seven real-model nodes, or a two-node source-only no-recipient path."""
    require(type(qualified) is values.QualifiedSSModelInputs, "QUALIFIED_TYPE")
    require(type(config) is categorical.CategoricalConfig, "CONFIG_TYPE")
    config.parameters(seed)
    common = dict(
        protocol=values.PROTOCOL,
        donor_sha256=codec.sha(qualified.donor_projection),
        recipient_sha256=codec.sha(qualified.recipient_projection),
    )
    result = [
        Node(
            SOURCE_NODE,
            SourceKernel.ref,
            sources=(SOURCE_NAME,),
            structural=StructuralDelta.CREATE,
            params=common,
            outputs=(
                Owned(
                    "person",
                    "age",
                    population_ops.token_for_dtype(
                        qualified.source_frame.person.age.dtype
                    ),
                ),
            ),
            artifact_outputs=(
                ArtifactOutput("donor_projection", PROJECTION_TYPE),
                ArtifactOutput("recipient_projection", PROJECTION_TYPE),
            ),
            description="Borrow every original current ASEC source person with unchanged DESIGN weights; no selected-support truncation.",
        )
    ]
    if qualified.matrix is not None:
        result.extend(
            (
                Node(
                    COLUMNS_NODE,
                    ColumnsKernel.ref,
                    population=SOURCE_NODE,
                    inputs=(Slice("person", ("age",)),),
                    params=common,
                    outputs=tuple(
                        Owned(
                            "person",
                            c,
                            population_ops.token_for_dtype(
                                qualified.donor_columns[c].dtype
                            ),
                        )
                        for c in qualified.donor_columns
                    ),
                    artifact_inputs=(_donor_edge(),),
                ),
                Node(
                    DONOR_NODE,
                    DonorKernel.ref,
                    base=SOURCE_NODE,
                    structural=StructuralDelta.FILTER,
                    mass="free",
                    inputs=(Slice("person", (values.ELIGIBLE,)),),
                    params=common,
                    artifact_inputs=(_donor_edge(),),
                ),
                Node(
                    MATRIX_NODE,
                    MatrixKernel.ref,
                    population=SOURCE_NODE,
                    params=common,
                    artifact_inputs=(_recipient_edge(),),
                    artifact_outputs=(
                        ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),
                    ),
                ),
            )
        )
        fit = fitted.categorical_fit_node(
            FIT_NODE,
            population=DONOR_NODE,
            entity="person",
            predictors=values.FEATURES,
            label=values.LABEL,
            classes=values.basis.COMPONENTS,
            seed=seed,
            config=config,
            source_projection=_edge("source_projection", "donor_projection"),
            source_sha256=common["donor_sha256"],
        )
        # The recipient-axis artifact nodes own no cells. The compiler makes a
        # FILTER depend on every member of its base version, so a member of
        # the full-source version that consumes the fit would close a cycle
        # through the donor FILTER. They therefore join the donor version.
        probabilities = fitted.categorical_probability_node(
            PROBABILITY_NODE,
            population=DONOR_NODE,
            fit_node=fit,
            matrix=ArtifactInput(
                "matrix", MATRIX_NODE, "matrix", model_input.RECIPIENT_MATRIX_TYPE
            ),
            matrix_sha256=codec.sha(qualified.matrix),
            source_projection=_edge("source_projection", "recipient_projection"),
            source_sha256=common["recipient_sha256"],
        )
        result.extend((fit, probabilities))
        inputs = (
            _donor_edge(),
            _recipient_edge(),
            ArtifactInput("model", FIT_NODE, "model", categorical.MODEL_TYPE),
            ArtifactInput(
                "model_metadata", FIT_NODE, "model_metadata", fitted.MODEL_METADATA_TYPE
            ),
            ArtifactInput(
                "matrix", MATRIX_NODE, "matrix", model_input.RECIPIENT_MATRIX_TYPE
            ),
            ArtifactInput(
                "probabilities",
                PROBABILITY_NODE,
                "probabilities",
                fitted.PROBABILITY_TYPE,
            ),
        )
    else:
        inputs = (_donor_edge(), _recipient_edge())
    result.append(
        Node(
            REPORT_NODE,
            ReportKernel.ref,
            population=SOURCE_NODE if qualified.matrix is None else DONOR_NODE,
            params=common,
            artifact_inputs=inputs,
            artifact_outputs=(ArtifactOutput("report", REPORT_TYPE),),
            description="Preserve source report basis and raw probabilities; complete only unresolved positive originals with exact total/reason support. No beneficiary columns.",
        )
    )
    return tuple(result)


def _artifact(context, name):
    edge = next((e for e in context.node.artifact_inputs if e.name == name), None)
    require(edge is not None and name in context.artifacts, "ARTIFACT_EDGE")
    value = context.artifacts[name]
    require(
        type(value) is ArtifactValue
        and value.type == edge.type
        and value.key == opaque_artifact_key(value.producer_key, edge.artifact),
        "ARTIFACT_IDENTITY",
    )
    require(
        value.numerics.numeric is Numeric.PLATFORM_BITWISE
        and value.numerics.platform == platform_fingerprint(),
        "ARTIFACT_PLATFORM",
    )
    require(type(value.payload) is bytes, "ARTIFACT_BYTES")
    return value


def _probabilities(qualified, context):
    if qualified.matrix is None:
        require(
            set(context.artifacts) == {"donor_projection", "recipient_projection"},
            "EMPTY_BRANCH_ARTIFACTS",
        )
        return pd.DataFrame(
            index=qualified.recipient_features.index,
            columns=values.basis.COMPONENTS,
            dtype="float64",
        ), {}
    model = _artifact(context, "model")
    metadata = _artifact(context, "model_metadata")
    matrix = _artifact(context, "matrix")
    probabilities = _artifact(context, "probabilities")
    require(model.producer_key == metadata.producer_key, "MODEL_PRODUCER_PAIR")
    require(matrix.payload == qualified.matrix, "RECIPIENT_MATRIX_CHANGED")
    doc = codec.decode_json(metadata.payload)
    require(doc["model_sha256"] == codec.sha(model.payload), "MODEL_BYTES_CHANGED")
    binding = dict(
        model_sha256=doc["model_sha256"],
        model_producer_key=model.producer_key,
        training_id=doc["training"]["training_id"],
        donor_source_sha256=codec.sha(qualified.donor_projection),
        matrix_sha256=codec.sha(qualified.matrix),
        matrix_producer_key=matrix.producer_key,
        source_sha256=codec.sha(qualified.recipient_projection),
        source_producer_key=_artifact(context, "recipient_projection").producer_key,
        classes=list(values.basis.COMPONENTS),
    )
    result = fitted.read_probabilities(
        probabilities.payload,
        expected_binding=binding,
        recipient_matrix=qualified.matrix,
    )
    return result, {
        **binding,
        "probability_producer_key": probabilities.producer_key,
        "probability_sha256": codec.sha(probabilities.payload),
    }


def _source_bytes(originals):
    """Fixed source arrays, including exact nullable float bits and status codes."""
    catalogs = {
        name: sorted(set(originals[name]))
        for name in ("basis_origin", "allocation_origin")
    }
    require(
        all(
            all(type(v) is str for v in catalog) and len(catalog) < 256
            for catalog in catalogs.values()
        ),
        "SOURCE_STATUSES",
    )
    parts = [
        originals.index.to_numpy(dtype="<i8", copy=True).tobytes(),
        originals.native_person_id.to_numpy(dtype="<i8", copy=True).tobytes(),
        originals.source.map({"asec": 0, "acs": 1}).to_numpy(dtype="uint8").tobytes(),
        values._bits(originals.social_security_source_total.to_numpy()),
        values._bits(originals.loc[:, values.basis.COMPONENTS].to_numpy()),
        values._allowed(originals).astype("uint8").tobytes(),
        originals.source_reporting_universe.to_numpy(dtype="uint8").tobytes(),
    ]
    for name, catalog in catalogs.items():
        parts.append(
            originals[name]
            .map({v: i for i, v in enumerate(catalog)})
            .to_numpy(dtype="uint8")
            .tobytes()
        )
    return b"".join(parts), catalogs


def encode_report(qualified, probabilities, model_binding):
    """Encode source, raw scores and completed report bits as distinct arrays."""
    completed = values.complete_reports(qualified.originals, probabilities)
    source_bytes, catalogs = _source_bytes(qualified.originals)
    raw = np.full((len(completed), 4), np.nan, dtype="float64")
    _, requested = values._report_selection(qualified.originals)
    raw[requested] = probabilities.to_numpy(copy=True)
    body = source_bytes + values._bits(raw) + values._bits(completed.to_numpy())
    header = codec.encode_json(
        dict(
            protocol=REPORT_TYPE.name + "/1",
            rows=len(completed),
            classes=list(values.basis.COMPONENTS),
            source_bytes=len(source_bytes),
            status_catalogs=catalogs,
            source_basis_sha256=values._table(qualified.originals),
            donor_projection_sha256=codec.sha(qualified.donor_projection),
            recipient_projection_sha256=codec.sha(qualified.recipient_projection),
            model_binding=model_binding,
            modeled_rows=int(requested.sum()),
            body_sha256=codec.sha(body),
            reporting_grain="source_person_report_may_combine_family_payments",
            scientific_qualification="pending",
            individual_beneficiary_assignment_claim=False,
            release_eligible=False,
        )
    )
    require(
        len(header) <= _HEADER_MAX and len(body) <= source.MAX_ROSTER_BYTES,
        "REPORT_SIZE",
    )
    return _MAGIC + len(header).to_bytes(4, "big") + header + body


def read_report(payload, *, qualified, expected_model_binding):
    """Independently reconstruct numeric completion; descriptive, not authority."""
    offset = len(_MAGIC)
    require(
        type(payload) is bytes
        and payload.startswith(_MAGIC)
        and len(payload) > offset + 4,
        "REPORT_ENVELOPE",
    )
    size = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    require(0 < size <= _HEADER_MAX and offset + size < len(payload), "REPORT_HEADER")
    header = codec.decode_json(payload[offset : offset + size])
    body = payload[offset + size :]
    originals = qualified.originals
    source_bytes, _ = _source_bytes(originals)
    n = len(originals)
    require(
        len(body) == len(source_bytes) + n * 64
        and body[: len(source_bytes)] == source_bytes,
        "REPORT_SOURCE_BYTES",
    )
    raw = np.frombuffer(
        body, dtype="<f8", count=n * 4, offset=len(source_bytes)
    ).reshape(n, 4)
    _, requested = values._report_selection(originals)
    require(np.isnan(raw[~requested]).all(), "REPORT_SOURCE_SCORES")
    probabilities = pd.DataFrame(
        raw[requested].copy(),
        index=qualified.recipient_features.index,
        columns=values.basis.COMPONENTS,
    )
    require(
        header.get("model_binding") == expected_model_binding, "REPORT_MODEL_BINDING"
    )
    require(
        encode_report(qualified, probabilities, expected_model_binding) == payload,
        "REPORT_RECONSTRUCTION",
    )
    completed = (
        np.frombuffer(body, dtype="<f8", count=n * 4, offset=len(source_bytes) + n * 32)
        .reshape(n, 4)
        .copy()
    )
    return pd.DataFrame(
        completed, index=originals.index.copy(), columns=values.basis.COMPONENTS
    ), probabilities


class _ReportBoundary:
    """Private graph custody over an existing preparation; no new issued handle.

    The owning caller validates before *any* graph/store I/O and after the final
    store read/write, including all-required replay, then calls ``pure`` last.
    """

    def __init__(self, preparation, *, seed, config):
        require(_live() == _LIVE, "GRAPH_IMPLEMENTATION_CHANGED")
        self.preparation = preparation
        self.entry = preparation._checked()
        self.qualified = values.qualify_current_survey_ss_model_inputs(preparation)
        self.seal = values.qualified_seal(self.qualified)
        self.seed, self.config = seed, config
        self.nodes = ss_report_completion_nodes(
            self.qualified, seed=seed, config=config
        )
        self.declarations = tuple(n.normative() for n in self.nodes)
        self.implementation = _implementation_hash()
        self.pure()

    def pure(self):
        require(
            _live() == _LIVE and values._live() == values._LIVE,
            "GRAPH_IMPLEMENTATION_CHANGED",
        )
        require(
            source._ISSUED.get(id(self.preparation)) is self.entry
            and self.preparation.payload == self.entry[1],
            "PREPARATION_IDENTITY",
        )
        source._pure_final(self.entry[2])
        require(
            values.qualified_seal(self.qualified) == self.seal
            and tuple(n.normative() for n in self.nodes) == self.declarations
            and tuple(
                n.normative()
                for n in ss_report_completion_nodes(
                    self.qualified, seed=self.seed, config=self.config
                )
            )
            == self.declarations,
            "QUALIFIED_OR_CONFIGURATION_CHANGED",
        )

    def validate(self):
        self.pure()
        with source.verification_epoch(join=True):
            require(self.preparation._checked() is self.entry, "PREPARATION_IDENTITY")
            fresh = values.qualify_current_survey_ss_model_inputs(self.preparation)
            require(values.qualified_seal(fresh) == self.seal, "SOURCE_REQUALIFICATION")
        require(_implementation_hash() == self.implementation, "GRAPH_SOURCE_CHANGED")
        self.pure()

    def context(self, context):
        self.pure()
        require(
            any(context.node.normative() == n for n in self.declarations)
            and dict(context.params) == dict(context.node.params),
            "NODE_DECLARATION",
        )
        require(
            set(context.artifacts) == {e.name for e in context.node.artifact_inputs},
            "CONTEXT_ARTIFACTS",
        )
        for name, expected in (
            ("donor_projection", self.qualified.donor_projection),
            ("recipient_projection", self.qualified.recipient_projection),
        ):
            if name in context.artifacts:
                require(
                    _artifact(context, name).payload == expected,
                    "SOURCE_PROJECTION_CHANGED",
                )
        if context.node.id in (COLUMNS_NODE, DONOR_NODE):
            require(set(context.tables) == {"person"}, "CONTEXT_TABLES")
            people = context.tables["person"]
            ids = self.qualified.donor_columns.index
            require(
                np.array_equal(people.person_id.to_numpy(), ids.to_numpy()),
                "CONTEXT_PERSON_AXIS",
            )
            if context.node.id == COLUMNS_NODE:
                require(
                    "age" in people
                    and population_ops.storage_equal(
                        people.age.reset_index(drop=True),
                        self.qualified.source_frame.person.age.reset_index(drop=True),
                    ),
                    "CONTEXT_SOURCE_AGE",
                )
            if context.node.id == DONOR_NODE:
                require(
                    people[values.ELIGIBLE]
                    .reset_index(drop=True)
                    .equals(
                        self.qualified.donor_columns[values.ELIGIBLE].reset_index(
                            drop=True
                        )
                    ),
                    "CONTEXT_DONOR_MASK",
                )
        else:
            require(not context.tables, "CONTEXT_TABLES")
        require(
            set(context.sources)
            == ({SOURCE_NAME} if context.node.id == SOURCE_NODE else set()),
            "CONTEXT_SOURCES",
        )


def _result_stamp(result):
    return (
        None if result.frame is None else source._frame_identity(result.frame),
        None if result.keep is None else values._table(result.keep.to_frame("keep")),
        tuple(
            (key, values._table(series.to_frame(key[1])))
            for key, series in sorted(result.columns.items())
        ),
        tuple(sorted(result.artifacts.items())),
        codec.encode_json(result.receipt),
    )


def _implementation_hash():
    return source_hash(
        sys.modules[__name__],
        values,
        values.observed,
        values.basis,
        values.predictors,
        values.predictors.universe,
        source,
        source.asec_native,
        source.acs_native,
        model_input,
        categorical,
        fitted,
        codec,
        population_ops,
        frame_weights,
        dependencies=categorical.DEPENDENCIES,
    )


class _Kernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=categorical.DEPENDENCIES,
    )

    def __init__(self, boundary):
        require(type(boundary) is _ReportBoundary, "BOUNDARY_TYPE")
        self.boundary = boundary

    def implementation_hash(self):
        value = _implementation_hash()
        require(value == self.boundary.implementation, "GRAPH_SOURCE_CHANGED")
        self.boundary.pure()
        return value

    def run(self, context):
        boundary = self.boundary
        boundary.context(context)
        q = boundary.qualified
        name = context.node.id
        if name == SOURCE_NODE:
            frame = load_source("frame-store", context.sources[SOURCE_NAME])
            require(
                source._frame_identity(frame) == source._frame_identity(q.source_frame),
                "SOURCE_FRAME_CHANGED",
            )
            result = KernelResult(
                frame=frame,
                artifacts={
                    "donor_projection": q.donor_projection,
                    "recipient_projection": q.recipient_projection,
                },
            )
        elif name == COLUMNS_NODE:
            result = KernelResult(
                columns={
                    ("person", c): q.donor_columns[c].copy(deep=True)
                    for c in q.donor_columns
                }
            )
        elif name == DONOR_NODE:
            result = KernelResult(keep=q.donor_columns[values.ELIGIBLE].copy(deep=True))
        elif name == MATRIX_NODE:
            require(type(q.matrix) is bytes, "MATRIX_ABSENT")
            result = KernelResult(artifacts={"matrix": q.matrix})
        else:
            require(name == REPORT_NODE, "NODE")
            probabilities, binding = _probabilities(q, context)
            report = encode_report(q, probabilities, binding)
            result = KernelResult(
                artifacts={"report": report},
                receipt=dict(
                    protocol=values.PROTOCOL,
                    report_sha256=codec.sha(report),
                    modeled_rows=len(probabilities),
                    individual_beneficiary_assignment_claim=False,
                    release_eligible=False,
                ),
            )
        stamp = _result_stamp(result)
        boundary.context(context)
        boundary.pure()
        require(_result_stamp(result) == stamp, "FINAL_RESULT_CHANGED")
        return result


class SourceKernel(_Kernel):
    ref = "us.ss_report.source@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.CREATE)


class ColumnsKernel(_Kernel):
    ref = "us.ss_report.columns@1"


class DonorKernel(_Kernel):
    ref = "us.ss_report.donors@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.FILTER)


class MatrixKernel(_Kernel):
    ref = "us.ss_report.matrix@1"


class ReportKernel(_Kernel):
    ref = "us.ss_report.complete@1"


def ss_report_kernel_registry(boundary):
    """Fixed implementation roster, not a caller-provided evaluator registry."""
    result = KernelRegistry()
    for cls in (SourceKernel, ColumnsKernel, DonorKernel, MatrixKernel, ReportKernel):
        result.register(cls(boundary))
    result.register(fitted.CategoricalFitKernel())
    result.register(fitted.CategoricalProbabilityKernel())
    return result


def _live():
    return (
        tuple(
            (
                m,
                tuple(
                    (n, source._function_seal(v))
                    for n, v in vars(m).items()
                    if type(v) is FunctionType
                ),
            )
            for m in (sys.modules[__name__], values, categorical, fitted, model_input)
        ),
        tuple(
            (
                cls,
                tuple(
                    (n, source._function_seal(v))
                    for n, v in vars(cls).items()
                    if type(v) is FunctionType
                ),
            )
            for cls in (
                _ReportBoundary,
                _Kernel,
                SourceKernel,
                ColumnsKernel,
                DonorKernel,
                MatrixKernel,
                ReportKernel,
                categorical.CategoricalConfig,
                categorical.CategoricalModel,
                fitted.CategoricalFitKernel,
                fitted.CategoricalProbabilityKernel,
            )
        ),
        tuple(
            (cls, cls.ref, _capabilities_projection(cls.capabilities))
            for cls in (
                SourceKernel,
                ColumnsKernel,
                DonorKernel,
                MatrixKernel,
                ReportKernel,
                fitted.CategoricalFitKernel,
                fitted.CategoricalProbabilityKernel,
            )
        ),
        source,
        codec,
        require,
        values,
        SOURCE_NAME,
        SOURCE_NODE,
        COLUMNS_NODE,
        DONOR_NODE,
        MATRIX_NODE,
        FIT_NODE,
        PROBABILITY_NODE,
        REPORT_NODE,
        PROJECTION_TYPE,
        REPORT_TYPE,
        _MAGIC,
        _HEADER_MAX,
        values.PROTOCOL,
        values.FEATURES,
        values.basis.COMPONENTS,
    )


_LIVE = _live()
