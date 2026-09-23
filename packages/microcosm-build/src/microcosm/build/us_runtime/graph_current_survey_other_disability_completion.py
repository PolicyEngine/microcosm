"""Optional other-disability QRF fragment and explicit canonical version barrier.

The private boundary borrows genuine preparation custody. The owning host must
also validate its retained receiving run before and after final graph/store I/O,
including replay; passing a Frame or artifact never issues authority.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import FunctionType

import numpy as np
import pandas as pd

from microcosm.fit import graph_legacy_qrf as fitted
from microcosm.fit import model_input, qrf, qrf_target
from microcosm.fit.graph_legacy_apply_matrix import (
    MATRIX_APPLY_STATE_TYPE,
    LegacyQRFApplyMatrixKernel,
    decode_matrix_apply_state,
)
from microcosm.fit.kernels import FIT_QRF_DEPENDENCIES
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

from . import current_survey_other_disability_completion as values

source, codec, require = values.source, values.codec, values.require
SOURCE_NAME = "survey_other_disability_full_original_asec"
SOURCE_NODE = "survey_other_disability.full_source"
COLUMNS_NODE = "survey_other_disability.model_columns"
DONOR_NODE = "survey_other_disability.donors"
MATRIX_NODE = "survey_other_disability.recipient_projection"
FIT_PREFIX = "survey_other_disability.fit"
APPLY_PREFIX = "survey_other_disability.apply"
VERSION_NODE = "survey_other_disability.version"
ATTACH_NODE = "survey_other_disability.attach"
DONOR_TYPE = ArtifactType("microcosm.us.other_disability_donor_projection", 1)
RECIPIENT_TYPE = ArtifactType("microcosm.us.other_disability_recipient_projection", 1)
VERSION_TYPE = ArtifactType("microcosm.us.other_disability_version", 1)
ATTACHMENT_TYPE = ArtifactType("microcosm.us.other_disability_attachment", 1)
PHASE = "survey_other_disability_completion_development"
FRAME_REPORTS = (
    values.observed.attached_name(values.observed.KNOWN_COLUMN),
    values.observed.attached_name(values.observed.REASON_COLUMN),
    values.MODEL_APPLICABLE,
    values.CANONICAL_KNOWN,
    values.VALUE_ORIGIN,
)


def _edge(name, producer, artifact, type_):
    return ArtifactInput(name, producer, artifact, type_)


def _donor_edge():
    return _edge("donor_projection", SOURCE_NODE, "donor_projection", DONOR_TYPE)


def _recipient_edge():
    return _edge(
        "recipient_projection", MATRIX_NODE, "recipient_projection", RECIPIENT_TYPE
    )


def other_disability_completion_nodes(
    qualified, receiving, *, receiving_version, after, after_payload, seed, n_estimators
):
    """Eight-node amount path, or four nodes when no ACS original is applicable."""
    require(
        type(qualified) is values.QualifiedOtherDisabilityCompletion, "QUALIFIED_TYPE"
    )
    require(
        type(after) is ArtifactInput
        and after.name == "after"
        and type(after_payload) is bytes,
        "TERMINAL_BINDING",
    )
    require(
        type(seed) is int
        and seed >= 0
        and type(n_estimators) is int
        and n_estimators > 0,
        "MODEL_CONFIGURATION",
    )
    dtypes = values.attachment_dtypes(qualified, receiving)
    common = {
        "protocol": values.PROTOCOL,
        "donor_sha256": codec.sha(qualified.donor_projection),
        "recipient_sha256": codec.sha(qualified.recipient_projection),
        "after_sha256": codec.sha(after_payload),
        "receiving_sha256": source._frame_identity(receiving),
    }
    nodes = [
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
            artifact_outputs=(ArtifactOutput("donor_projection", DONOR_TYPE),),
            description="Every original current ASEC person and original DESIGN weight; unknown target observations are not zero donors.",
        )
    ]
    if qualified.matrix is not None:
        nodes.extend(
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
            )
        )
    identity_columns = tuple(
        values.provenance.__dict__[name]("person")
        for name in (
            "support_source_id_column",
            "support_clone_index_column",
            "spine_source_id_column",
            "support_channel_column",
        )
    )
    nodes.append(
        Node(
            MATRIX_NODE,
            RecipientKernel.ref,
            population=receiving_version,
            params=common,
            inputs=(Slice("person", identity_columns),),
            artifact_inputs=(after,),
            artifact_outputs=(ArtifactOutput("recipient_projection", RECIPIENT_TYPE),)
            + (
                ()
                if qualified.matrix is None
                else (ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),)
            ),
            description="Selected original ACS age-15+ recipients, ordered by original person ID; clone rows are never a model matrix.",
        )
    )
    final_inputs = [_donor_edge(), _recipient_edge(), after]
    if qualified.matrix is not None:
        fits = fitted.legacy_qrf_train_nodes(
            FIT_PREFIX,
            population=DONOR_NODE,
            entity="person",
            predictors=values.FEATURES,
            targets=(values.TARGET,),
            seed=seed,
            n_estimators=n_estimators,
            zero_atol=0,
            phase=PHASE,
        )
        applies = fitted.legacy_qrf_apply_matrix_nodes(
            APPLY_PREFIX,
            population=receiving_version,
            fit_nodes=fits,
            matrix_producer=MATRIX_NODE,
            seed=seed,
            phase=PHASE,
        )
        nodes.extend((*fits, *applies))
        final_inputs.extend(
            (
                _edge(
                    "matrix", MATRIX_NODE, "matrix", model_input.RECIPIENT_MATRIX_TYPE
                ),
                _edge("model", fits[0].id, "model", qrf_target.LEGACY_QRF_TARGET_TYPE),
                _edge(
                    "training_state",
                    fits[0].id,
                    "training_state",
                    codec.TRAINING_STATE_TYPE,
                ),
                _edge("raw_draw", applies[0].id, "raw_draw", codec.RAW_TARGET_TYPE),
                _edge(
                    "apply_state", applies[0].id, "apply_state", MATRIX_APPLY_STATE_TYPE
                ),
            )
        )
    nodes.append(
        Node(
            VERSION_NODE,
            VersionKernel.ref,
            base=receiving_version,
            structural=StructuralDelta.FILTER,
            params=common,
            inputs=(
                Slice(
                    "person", (values.provenance.support_clone_index_column("person"),)
                ),
            ),
            artifact_inputs=tuple(final_inputs),
            artifact_outputs=(ArtifactOutput("version", VERSION_TYPE),),
            description="Keep all receiving rows and weights in a new version before the explicit disability-benefits replacement.",
        )
    )
    nodes.append(
        Node(
            ATTACH_NODE,
            AttachKernel.ref,
            population=VERSION_NODE,
            params=common,
            outputs=tuple(
                Owned("person", c, dtypes[c], rewrite=c in receiving.person)
                for c in (values.observed.OUTPUT, *FRAME_REPORTS)
            ),
            artifact_inputs=(
                *final_inputs,
                _edge("version", VERSION_NODE, "version", VERSION_TYPE),
            ),
            artifact_outputs=(ArtifactOutput("attachment", ATTACHMENT_TYPE),),
            description="Canonical dollars plus supported knownness/origin reports. Exact nullable Float64/Int16 source reports, backing bytes and masks remain in the typed attachment artifact, not coerced Frame leaves.",
        )
    )
    return tuple(nodes)


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
        and value.numerics.platform == platform_fingerprint()
        and type(value.payload) is bytes,
        "ARTIFACT_PLATFORM_OR_BYTES",
    )
    return value


def read_draws(qualified, context, *, seed):
    """Bind draw bytes, sibling state, fit history and the exact recipient matrix."""
    if qualified.matrix is None:
        return pd.Series(index=qualified.recipient_features.index, dtype="float64"), {}
    matrix = _artifact(context, "matrix")
    require(matrix.payload == qualified.matrix, "MATRIX_BYTES")
    decoded = model_input.decode_recipient_matrix(matrix.payload)
    require(
        decoded.entity == "person"
        and decoded.features.equals(qualified.recipient_features),
        "MATRIX_AXIS",
    )
    model, training_value = (
        _artifact(context, "model"),
        _artifact(context, "training_state"),
    )
    require(model.producer_key == training_value.producer_key, "MODEL_STATE_SIBLINGS")
    training, training_chain = codec.read_training(training_value.payload)
    require(
        len(training["models"]) == 1
        and training["models"][0]["target"] == values.TARGET
        and training["models"][0]["sha256"] == codec.sha(model.payload)
        and tuple(training_chain.to_dict()["completed_targets"]) == (values.TARGET,),
        "MODEL_HISTORY",
    )
    raw, state_value = _artifact(context, "raw_draw"), _artifact(context, "apply_state")
    require(raw.producer_key == state_value.producer_key, "RAW_STATE_SIBLINGS")
    packet = decode_matrix_apply_state(state_value.payload)
    application, chain = codec.read_application(
        codec.encode_json(packet["application"])
    )
    require(
        packet["matrix_sha256"] == codec.sha(qualified.matrix)
        and packet["matrix_producer_key"] == matrix.producer_key
        and chain.entity == "person"
        and tuple(chain.predictors) == values.FEATURES
        and tuple(chain.targets) == (values.TARGET,)
        and tuple(chain.completed_targets) == (values.TARGET,)
        and chain.recipient_index == qrf._index_identity(decoded.features.index)
        and application["seed"] == seed
        and application["models"] == training["models"]
        and application["raw_targets"]
        == [{"target": values.TARGET, "sha256": codec.sha(raw.payload)}],
        "DRAW_CHAIN",
    )
    draw = codec.read_raw_target(
        raw.payload, target=values.TARGET, index=decoded.features.index
    )
    return pd.Series(draw, index=decoded.features.index, dtype="float64"), {
        "model_sha256": codec.sha(model.payload),
        "model_producer_key": model.producer_key,
        "training_id": training["models"][0]["training_id"],
        "matrix_sha256": codec.sha(matrix.payload),
        "matrix_producer_key": matrix.producer_key,
        "raw_sha256": codec.sha(raw.payload),
        "apply_producer_key": raw.producer_key,
        "seed": seed,
    }


def _source_report_storage(table):
    """Closed lossless source-report encoding; never round nullable data or fill masks."""
    columns = []
    for name in table:
        series = table[name]
        array = series.array
        if isinstance(
            series.dtype,
            (pd.Float64Dtype, pd.Int16Dtype, pd.Int64Dtype, pd.BooleanDtype),
        ):
            item = {
                "data_dtype": array._data.dtype.str,
                "data_hex": array._data.tobytes().hex(),
                "mask_hex": array._mask.tobytes().hex(),
            }
        elif isinstance(series.dtype, pd.StringDtype):
            item = {
                "storage": series.dtype.storage,
                "values": [None if value is pd.NA else value for value in array],
            }
        else:
            require(series.dtype == np.dtype("int64"), "SOURCE_REPORT_DTYPE")
            item = {
                "data_dtype": series.dtype.str,
                "data_hex": series.to_numpy().tobytes().hex(),
            }
        columns.append({"name": name, "dtype": str(series.dtype), **item})
    return {
        "index_name": table.index.name,
        "index_dtype": table.index.dtype.str,
        "index_hex": table.index.to_numpy().tobytes().hex(),
        "columns": columns,
        "physical_sha256": values._table(table),
    }


def attachment_payload(qualified, receiving, draws, model_binding):
    attached = values.attach_other_disability_completion(qualified, receiving, draws)
    frame_names = (values.observed.OUTPUT, *FRAME_REPORTS)
    columns = {
        key: value for key, value in attached.columns.items() if key[1] in frame_names
    }
    payload = codec.encode_json(
        {
            "protocol": values.PROTOCOL,
            "model_binding": model_binding,
            "donor_projection_sha256": codec.sha(qualified.donor_projection),
            "recipient_projection_sha256": codec.sha(qualified.recipient_projection),
            "source_reports": _source_report_storage(qualified.selected_source.person),
            "source_report_scope": "selected_original_asec_before_clone_transport",
            "artifact_only_source_fields": [
                name
                for name in qualified.selected_source.person
                if values.observed.attached_name(name) not in FRAME_REPORTS
            ],
            "frame_projection": list(frame_names),
            "frame_columns_sha256": values._table(
                pd.DataFrame({name: col for (_, name), col in columns.items()})
            ),
            "receipt": attached.receipt,
        }
    )
    return columns, payload


class _CompletionBoundary:
    """Borrowed private custody. The caller validates the actual receiving owner."""

    def __init__(
        self,
        preparation,
        receiving,
        *,
        receiving_version,
        after,
        after_payload,
        seed,
        n_estimators,
    ):
        require(_live() == _LIVE, "GRAPH_IMPLEMENTATION_CHANGED")
        self.preparation, self.receiving = preparation, receiving
        self.entry = preparation._checked()
        self.qualified = values.qualify_current_survey_other_disability_completion(
            preparation
        )
        self.seal = values.qualified_seal(self.qualified)
        self.receiving_seal = source._frame_identity(receiving)
        self.configuration = dict(
            receiving_version=receiving_version,
            after=after,
            after_payload=after_payload,
            seed=seed,
            n_estimators=n_estimators,
        )
        self.nodes = other_disability_completion_nodes(
            self.qualified, receiving, **self.configuration
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
            and source._frame_identity(self.receiving) == self.receiving_seal
            and tuple(n.normative() for n in self.nodes) == self.declarations
            and tuple(
                n.normative()
                for n in other_disability_completion_nodes(
                    self.qualified, self.receiving, **self.configuration
                )
            )
            == self.declarations,
            "QUALIFIED_OR_CONFIGURATION_CHANGED",
        )

    def validate(self):
        self.pure()
        with source.verification_epoch(join=True):
            require(self.preparation._checked() is self.entry, "PREPARATION_IDENTITY")
            fresh = values.qualify_current_survey_other_disability_completion(
                self.preparation
            )
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
            "ARTIFACT_ROSTER",
        )
        for name in context.artifacts:
            value = _artifact(context, name)
            expected = {
                "donor_projection": self.qualified.donor_projection,
                "recipient_projection": self.qualified.recipient_projection,
                "after": self.configuration["after_payload"],
            }.get(name)
            if expected is not None:
                require(value.payload == expected, "PROJECTION_BYTES:" + name)
        name = context.node.id
        require(
            set(context.sources) == ({SOURCE_NAME} if name == SOURCE_NODE else set()),
            "CONTEXT_SOURCES",
        )
        if name in (COLUMNS_NODE, DONOR_NODE, MATRIX_NODE, VERSION_NODE):
            require(set(context.tables) == {"person"}, "CONTEXT_TABLES")
            people = context.tables["person"]
            expected_people = (
                self.qualified.source_frame.person
                if name in (COLUMNS_NODE, DONOR_NODE)
                else self.receiving.person
            )
            require(
                np.array_equal(
                    people.person_id.to_numpy(), expected_people.person_id.to_numpy()
                ),
                "CONTEXT_PERSON_AXIS",
            )
            for column in context.node.inputs[0].columns:
                expected = (
                    self.qualified.donor_columns[column]
                    if name == DONOR_NODE
                    else expected_people[column]
                )
                require(
                    population_ops.storage_equal(
                        people[column].reset_index(drop=True),
                        expected.reset_index(drop=True),
                    ),
                    "CONTEXT_COLUMN:" + column,
                )
        elif name == ATTACH_NODE:
            require(set(context.tables) == {"person"}, "CONTEXT_TABLES")
            people = context.tables["person"]
            structural = (
                self.receiving.schema.entity_id_column("person"),
                *(
                    self.receiving.schema.membership_column(e)
                    for e in self.receiving.schema.group_entities
                ),
            )
            incumbents = tuple(o.column for o in context.node.outputs if o.rewrite)
            require(
                tuple(people.columns) == (*structural, *incumbents),
                "ATTACH_CONTEXT_COLUMNS",
            )
            for column in people:
                require(
                    population_ops.storage_equal(
                        people[column].reset_index(drop=True),
                        self.receiving.person[column].reset_index(drop=True),
                    ),
                    "ATTACH_CONTEXT_IDENTITY:" + column,
                )
        else:
            require(not context.tables, "CONTEXT_TABLES")


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
        values.observed.detail,
        values.predictors,
        values.predictors.universe,
        source,
        source.asec_native,
        source.acs_native,
        fitted,
        model_input,
        codec,
        qrf,
        qrf_target,
        LegacyQRFApplyMatrixKernel,
        population_ops,
        dependencies=FIT_QRF_DEPENDENCIES,
    )


class _Kernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=FIT_QRF_DEPENDENCIES,
    )

    def __init__(self, boundary):
        require(type(boundary) is _CompletionBoundary, "BOUNDARY_TYPE")
        self.boundary = boundary

    def implementation_hash(self):
        result = _implementation_hash()
        require(result == self.boundary.implementation, "GRAPH_SOURCE_CHANGED")
        self.boundary.pure()
        return result

    def run(self, context):
        boundary = self.boundary
        boundary.context(context)
        q, name = boundary.qualified, context.node.id
        if name == SOURCE_NODE:
            frame = load_source("frame-store", context.sources[SOURCE_NAME])
            require(
                source._frame_identity(frame) == source._frame_identity(q.source_frame),
                "SOURCE_FRAME_CHANGED",
            )
            result = KernelResult(
                frame=frame, artifacts={"donor_projection": q.donor_projection}
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
            result = KernelResult(
                artifacts={
                    "recipient_projection": q.recipient_projection,
                    **({} if q.matrix is None else {"matrix": q.matrix}),
                }
            )
        else:
            draws, binding = read_draws(q, context, seed=boundary.configuration["seed"])
            columns, payload = attachment_payload(q, boundary.receiving, draws, binding)
            version = codec.encode_json(
                {
                    "protocol": values.PROTOCOL,
                    "selection": "keep_all",
                    "receiving_sha256": boundary.receiving_seal,
                    "attachment_sha256": codec.sha(payload),
                }
            )
            if name == VERSION_NODE:
                result = KernelResult(
                    keep=pd.Series(
                        True,
                        index=pd.Index(
                            context.tables["person"].person_id, name="person_id"
                        ),
                        dtype="bool",
                    ),
                    artifacts={"version": version},
                )
            else:
                require(
                    name == ATTACH_NODE
                    and _artifact(context, "version").payload == version,
                    "VERSION_BINDING",
                )
                result = KernelResult(
                    columns=columns,
                    artifacts={"attachment": payload},
                    receipt={
                        "protocol": values.PROTOCOL,
                        "modeled_originals": len(draws),
                        "source_reports_storage": "typed_attachment_artifact",
                        "under15_completed_with_zero": False,
                        "release_eligible": False,
                    },
                )
        stamp = _result_stamp(result)
        boundary.context(context)
        boundary.pure()
        require(_result_stamp(result) == stamp, "FINAL_RESULT_CHANGED")
        return result


class SourceKernel(_Kernel):
    ref = "us.other_disability.source@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.CREATE)


class ColumnsKernel(_Kernel):
    ref = "us.other_disability.columns@1"


class DonorKernel(_Kernel):
    ref = "us.other_disability.donors@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.FILTER)


class RecipientKernel(_Kernel):
    ref = "us.other_disability.recipient_projection@1"


class VersionKernel(_Kernel):
    ref = "us.other_disability.version@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.FILTER)


class AttachKernel(_Kernel):
    ref = "us.other_disability.attach@1"


def other_disability_kernel_registry(boundary):
    result = KernelRegistry()
    for cls in (
        SourceKernel,
        ColumnsKernel,
        DonorKernel,
        RecipientKernel,
        VersionKernel,
        AttachKernel,
    ):
        result.register(cls(boundary))
    result.register(fitted.LegacyQRFTrainKernel())
    result.register(LegacyQRFApplyMatrixKernel())
    return result


def _live():
    classes = (
        _CompletionBoundary,
        _Kernel,
        SourceKernel,
        ColumnsKernel,
        DonorKernel,
        RecipientKernel,
        VersionKernel,
        AttachKernel,
        fitted.LegacyQRFTrainKernel,
        LegacyQRFApplyMatrixKernel,
    )
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
            for m in (
                sys.modules[__name__],
                values,
                fitted,
                model_input,
                codec,
                qrf,
                qrf_target,
            )
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
            for cls in classes
        ),
        tuple(
            (cls, cls.ref, _capabilities_projection(cls.capabilities))
            for cls in classes[2:]
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
        FIT_PREFIX,
        APPLY_PREFIX,
        VERSION_NODE,
        ATTACH_NODE,
        DONOR_TYPE,
        RECIPIENT_TYPE,
        VERSION_TYPE,
        ATTACHMENT_TYPE,
        PHASE,
        FRAME_REPORTS,
        decode_matrix_apply_state,
        source._function_seal(decode_matrix_apply_state),
        FIT_QRF_DEPENDENCIES,
    )


_LIVE = _live()
