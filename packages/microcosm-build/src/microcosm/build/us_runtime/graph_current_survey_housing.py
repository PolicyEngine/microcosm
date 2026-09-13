"""Household participation fragment in the common survey enrichment graph.

The source projection retains interview observations separately from modeled
annual participation. A household-design-weighted binary QRF completes only
declared missing recipients. The country model owns awards and SPM valuation.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

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
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    Owned,
    StructuralDelta,
    source_hash,
)

from . import current_survey_amounts as amount_values
from . import current_survey_housing as housing
from . import graph_current_survey_predictors as predictor_graph
from . import housing_participation as participation

PROJECTION_NODE = "survey_housing.source_projection"
DONOR_NODE = "survey_housing.asec_known_donor"
COLUMNS_NODE = "survey_housing.donor_columns"
FIT_PREFIX = "survey_housing.fit"
APPLY_PREFIX = "survey_housing.apply"
ATTACH_NODE = "survey_housing.attach"
PROJECTION_TYPE = ArtifactType("microcosm.us.current_survey_housing_projection", 1)
ATTACHMENT_TYPE = ArtifactType("microcosm.us.current_survey_housing_attachment", 1)
SEED = housing.SEED
PHASE = "current_survey_housing.participation"
require = housing.require
physical = amount_values.physical


def projection_edge():
    return ArtifactInput(
        "housing_projection", PROJECTION_NODE, "projection", PROJECTION_TYPE
    )


def housing_nodes(qualified, receiving, *, after, n_estimators):
    """Declare a household-grain donor fit and source-keyed clone attachment."""
    require(type(n_estimators) is int and n_estimators > 0, "HOUSING_ESTIMATORS")
    params = {
        "projection_sha256": codec.sha(qualified.projection),
        "n_estimators": n_estimators,
        "protocol": housing.PROTOCOL,
    }
    version = amount_values.parent_host.attach.FILTER_NODE
    source_outputs = [ArtifactOutput("projection", PROJECTION_TYPE)]
    if qualified.matrix is not None:
        source_outputs.append(
            ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE)
        )
    nodes = [
        Node(
            PROJECTION_NODE,
            HousingProjectionKernel.ref,
            population=version,
            inputs=predictor_graph._inputs(receiving),
            params=params,
            artifact_inputs=(after,),
            artifact_outputs=tuple(source_outputs),
            description="Bind independent household housing observations, source knownness and explicit annual participation assumptions.",
        )
    ]
    attach_edges = [projection_edge()]
    if qualified.matrix is not None:
        require(qualified.donor_frame is not None, "HOUSING_DONOR_REQUIRED")
        nodes.extend(
            (
                Node(
                    DONOR_NODE,
                    HousingDonorKernel.ref,
                    base=amount_values.predictors.host.survey_graph.CREATE_NODE,
                    structural=StructuralDelta.FILTER,
                    mass="free",
                    inputs=predictor_graph._inputs(qualified.source_frame),
                    params=params,
                    artifact_inputs=(projection_edge(),),
                    description="Select complete known original ASEC donor households and retain their original household design weights.",
                ),
                Node(
                    COLUMNS_NODE,
                    HousingColumnsKernel.ref,
                    population=DONOR_NODE,
                    inputs=predictor_graph._inputs(qualified.donor_frame),
                    outputs=tuple(
                        Owned("household", name, "float64")
                        for name in (*qualified.features, housing.TARGET)
                    ),
                    params=params,
                    artifact_inputs=(projection_edge(),),
                    description="One binary source participation label and one predictor row per donor household.",
                ),
            )
        )
        fits = legacy_qrf_train_nodes(
            FIT_PREFIX,
            population=DONOR_NODE,
            entity="household",
            predictors=qualified.features,
            targets=(housing.TARGET,),
            seed=SEED,
            phase=PHASE,
            n_estimators=n_estimators,
            zero_atol=0,
        )
        applies = legacy_qrf_apply_matrix_nodes(
            APPLY_PREFIX,
            population=version,
            fit_nodes=fits,
            matrix_producer=PROJECTION_NODE,
            seed=SEED,
            phase=PHASE,
        )
        nodes.extend((*fits, *applies))
        attach_edges.extend(
            (
                ArtifactInput(
                    "housing_matrix",
                    PROJECTION_NODE,
                    "matrix",
                    model_input.RECIPIENT_MATRIX_TYPE,
                ),
                ArtifactInput(
                    "housing_raw", applies[0].id, "raw_draw", codec.RAW_TARGET_TYPE
                ),
                ArtifactInput(
                    "housing_state",
                    applies[0].id,
                    "apply_state",
                    MATRIX_APPLY_STATE_TYPE,
                ),
            )
        )
    nodes.append(
        Node(
            ATTACH_NODE,
            HousingAttachKernel.ref,
            population=version,
            inputs=predictor_graph._inputs(receiving),
            outputs=housing.output_columns(qualified, receiving),
            params=params,
            artifact_inputs=tuple(attach_edges),
            artifact_outputs=(ArtifactOutput("attachment", ATTACHMENT_TYPE),),
            description="Preserve observed knownness, join one modeled draw to each household's clones, and route assistance only to the householder's SPM unit.",
        )
    )
    return tuple(nodes)


def read_draw(qualified, artifacts):
    """Bind the binary draw to its actual household matrix and model history."""
    require(
        artifacts["housing_projection"].payload == qualified.projection,
        "HOUSING_PROJECTION_BYTES",
    )
    if qualified.matrix is None:
        require(set(artifacts) == {"housing_projection"}, "HOUSING_NO_FIT_ROSTER")
        return None
    matrix_value = artifacts["housing_matrix"]
    require(matrix_value.payload == qualified.matrix, "HOUSING_MATRIX_BYTES")
    matrix = model_input.decode_recipient_matrix(qualified.matrix)
    raw_value, state_value = artifacts["housing_raw"], artifacts["housing_state"]
    packet = decode_matrix_apply_state(state_value.payload)
    application, chain = codec.read_application(
        codec.encode_json(packet["application"])
    )
    raw = raw_value.payload
    require(
        raw_value.producer_key == state_value.producer_key
        and packet["matrix_sha256"] == codec.sha(qualified.matrix)
        and packet["matrix_producer_key"] == matrix_value.producer_key
        and chain.entity == "household"
        and tuple(chain.predictors) == qualified.features
        and tuple(chain.targets) == (housing.TARGET,)
        and tuple(chain.completed_targets) == (housing.TARGET,)
        and chain.recipient_index == qrf._index_identity(matrix.features.index)
        and application["seed"] == SEED
        and application["raw_targets"]
        == [{"target": housing.TARGET, "sha256": codec.sha(raw)}]
        and len(application["models"]) == 1,
        "HOUSING_DRAW_CHAIN",
    )
    draw = pd.DataFrame(index=matrix.features.index)
    draw[housing.TARGET] = codec.read_raw_target(
        raw, target=housing.TARGET, index=matrix.features.index
    )
    require(draw[housing.TARGET].isin((0.0, 1.0)).all(), "HOUSING_BINARY_DRAW")
    return draw


def attachment_result(boundary, artifacts):
    qualified = boundary.housing
    draw = read_draw(qualified, artifacts)
    columns = housing.attach_columns(qualified, boundary.run.population.frame, draw)
    receipt = {
        "protocol": housing.PROTOCOL,
        "parent_sha256": boundary.parent_view.digest,
        "projection_sha256": codec.sha(qualified.projection),
        "draw_sha256": None
        if draw is None
        else codec.sha(artifacts["housing_raw"].payload),
        "assumptions": qualified.evidence["assumptions"],
        "donor_entity": "household",
        "donor_weight_kind": "design",
        "source_knownness_preserved": True,
        "program_dollars_produced": False,
        "spm_caps_consumed": False,
        "weights_changed": False,
        "release_eligible": False,
    }
    return KernelResult(
        columns=columns,
        artifacts={"attachment": codec.encode_json(receipt)},
        receipt=receipt,
    )


class _HousingKernel(KernelBase):
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
            housing,
            housing.source_reader,
            housing.status,
            housing.household_source,
            participation,
            amount_values,
            amount_values.predictors,
            predictor_graph,
            physical,
            model_input,
            qrf,
            qrf_target,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        self.boundary.context(context)
        qualified = self.boundary.housing
        require(context.node in self.boundary.housing_nodes, "HOUSING_NODE")
        if context.node.id == DONOR_NODE:
            incoming = qualified.source_frame
            result = KernelResult(
                keep=pd.Series(
                    qualified.keep.copy(),
                    index=pd.Index(
                        incoming.person.person_id.to_numpy(), name="person_id"
                    ),
                ),
                receipt={
                    "donor_households": qualified.donor_frame.n("household"),
                    "donor_persons": qualified.donor_frame.n("person"),
                    "weight_kind": "design",
                },
            )
        elif context.node.id == COLUMNS_NODE:
            incoming = qualified.donor_frame
            result = KernelResult(
                columns={
                    ("household", name): qualified.donor_columns[name]
                    for name in qualified.donor_columns
                }
            )
        else:
            incoming = self.boundary.run.population.frame
            if context.node.id == PROJECTION_NODE:
                artifacts = {"projection": qualified.projection}
                if qualified.matrix is not None:
                    artifacts["matrix"] = qualified.matrix
                result = KernelResult(artifacts=artifacts, receipt=qualified.evidence)
            else:
                require(context.node.id == ATTACH_NODE, "HOUSING_ATTACH_NODE")
                result = attachment_result(self.boundary, context.artifacts)
        amount_values.predictors.host._current_context_frame(context, incoming)
        # Capture actual values before callbacks; a callback may not silently
        # replace a mapper or mutate an already-constructed output.
        stamp = result_stamp(result)
        self.boundary.pure()
        require(result_stamp(result) == stamp, "HOUSING_RESULT_CHANGED")
        return result


def result_stamp(result):
    return (
        tuple(
            (entity, name, physical._table_stamp(series.to_frame(name)))
            for (entity, name), series in sorted(result.columns.items())
        ),
        None
        if result.keep is None
        else physical._table_stamp(result.keep.to_frame("keep")),
        tuple(sorted(result.artifacts.items())),
        codec.encode_json(result.receipt),
    )


class HousingProjectionKernel(_HousingKernel):
    ref = "us.survey_housing.source_projection@1"


class HousingDonorKernel(_HousingKernel):
    ref = "us.survey_housing.known_household_donor@1"
    capabilities = replace(
        _HousingKernel.capabilities, structural=StructuralDelta.FILTER
    )


class HousingColumnsKernel(_HousingKernel):
    ref = "us.survey_housing.donor_columns@1"


class HousingAttachKernel(_HousingKernel):
    ref = "us.survey_housing.attach@1"


def kernels(boundary):
    refs = {node.kernel for node in boundary.housing_nodes}
    return tuple(
        cls(boundary)
        for cls in (
            HousingProjectionKernel,
            HousingDonorKernel,
            HousingColumnsKernel,
            HousingAttachKernel,
        )
        if cls.ref in refs
    )


def verify_model(boundary, loaded, donor):
    """Reconstruct the training state against actual household donor weights."""
    qualified = boundary.housing
    if qualified.matrix is None:
        require(donor is None, "HOUSING_UNEXPECTED_DONOR")
        return
    fit_id, apply_id = FIT_PREFIX + ".000", APPLY_PREFIX + ".000"
    first = boundary.compiled.graph.node(fit_id)
    model_frame = codec.model_frame(
        SimpleNamespace(
            weights={"household": donor.frame.resolve_weights("household")}
        ),
        first.inputs[0],
        donor.frame.table("household"),
    )
    model = qrf.RegimeGatedQRF(
        seed=SEED,
        n_estimators=boundary.n_estimators,
        zero_atol=0,
        max_samples_leaf=None,
    )
    before = qrf_target.LegacyQRFTrainingState.from_chain(
        model.start_chain(
            model_frame, list(qualified.features), [housing.TARGET], weights="design"
        )
    )
    payload = loaded[fit_id, "model"]
    packet, after = codec.read_training(loaded[fit_id, "training_state"])
    fitted = qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
        payload, expected_sha256=codec.sha(payload)
    )
    history = [
        {
            "target": housing.TARGET,
            "sha256": codec.sha(payload),
            "training_id": fitted.training_id,
        }
    ]
    require(
        fitted.training_state == before
        and fitted.next_training_state == after
        and fitted.donor_sha256
        == qrf_target._consumed_values_sha256(
            model_frame.table("household"), (*qualified.features, housing.TARGET)
        )
        and packet["models"] == history
        and decode_matrix_apply_state(loaded[apply_id, "apply_state"])["application"][
            "models"
        ]
        == history,
        "HOUSING_TRAINING_BINDING",
    )
