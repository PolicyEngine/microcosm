"""Opt-in donor fit/apply fragment within the retained survey enrichment host."""

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
    Slice,
    StructuralDelta,
    source_hash,
)
from microcosm.graph import population as population_ops

from . import current_survey_health_completion as values
from . import graph_current_survey_health as observed
from . import graph_current_survey_predictors as predictors

SOURCE_NODE = "survey_health_completion.full_asec_source"
COLUMNS_NODE = "survey_health_completion.model_columns"
DONOR_NODE = "survey_health_completion.known_asec_donor"
MATRIX_NODE = "survey_health_completion.original_acs_matrix"
FIT_PREFIX = "survey_health_completion.fit"
APPLY_PREFIX = "survey_health_completion.apply"
PROJECTION_TYPE = ArtifactType("microcosm.us.current_survey_health_completion", 1)
PHASE = "current_survey_health_completion.asec2025_to_acs2024"
require = values.require
physical = observed.physical


def projection_edge():
    return ArtifactInput(
        "health_completion_projection", SOURCE_NODE, "projection", PROJECTION_TYPE
    )


def nodes(qualified, observations, *, receiving_version, after, n_estimators):
    require(type(n_estimators) is int and n_estimators > 0, "ESTIMATORS")
    params = {
        "protocol": values.PROTOCOL,
        "projection_sha256": codec.sha(qualified.projection),
        "n_estimators": n_estimators,
    }
    schema = qualified.source_frame.schema
    structural = {
        schema.person_id_column,
        *(schema.membership_column(e) for e in schema.group_entities),
    }
    source_columns = tuple(
        c for c in qualified.source_frame.person if c not in structural
    )
    result = [
        Node(
            SOURCE_NODE,
            SourceKernel.ref,
            structural=StructuralDelta.CREATE,
            sources=(observed.SOURCE_NAME,),
            params=params,
            outputs=tuple(
                Owned(
                    "person",
                    c,
                    population_ops.token_for_dtype(
                        qualified.source_frame.person[c].dtype
                    ),
                )
                for c in source_columns
            ),
            artifact_inputs=(observed._source_edge(),),
            artifact_outputs=(ArtifactOutput("projection", PROJECTION_TYPE),),
            description="Borrow all original ASEC donor persons with original DESIGN weights and exact health source literals.",
        ),
        Node(
            COLUMNS_NODE,
            ColumnsKernel.ref,
            population=SOURCE_NODE,
            inputs=(Slice("person", source_columns),),
            params=params,
            outputs=observed._outputs(qualified.columns),
            artifact_inputs=(projection_edge(),),
            description="Encode seven known source yes/no targets and explicit complete-case donor eligibility; unknown is never false.",
        ),
        Node(
            DONOR_NODE,
            DonorKernel.ref,
            base=SOURCE_NODE,
            structural=StructuralDelta.FILTER,
            mass="free",
            inputs=(Slice("person", (values.ELIGIBLE,)),),
            params=params,
            artifact_inputs=(projection_edge(),),
            description="Select jointly known original ASEC targets and source age/sex/state; retain original design weights.",
        ),
        Node(
            MATRIX_NODE,
            MatrixKernel.ref,
            population=receiving_version,
            params=params,
            artifact_inputs=(projection_edge(), observed._source_edge()),
            artifact_outputs=(
                ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),
            ),
            description="Bind one predictor row per original ACS person, before transport to existing clones.",
        ),
    ]
    fits = legacy_qrf_train_nodes(
        FIT_PREFIX,
        population=DONOR_NODE,
        entity="person",
        predictors=values.FEATURES,
        targets=values.TARGETS,
        seed=values.SEED,
        phase=PHASE,
        n_estimators=n_estimators,
        zero_atol=0,
    )
    applies = legacy_qrf_apply_matrix_nodes(
        APPLY_PREFIX,
        population=receiving_version,
        fit_nodes=fits,
        matrix_producer=MATRIX_NODE,
        seed=values.SEED,
        phase=PHASE,
    )
    result.extend((*fits, *applies))
    attachment = observed.health_coverage_nodes(
        observations, receiving_version=receiving_version, after=after
    )[-1]
    result.append(
        replace(
            attachment,
            kernel=AttachKernel.ref,
            params=params,
            outputs=(
                *attachment.outputs,
                *(
                    Owned("person", field.output + "__imputed", "bool")
                    for field in values.FIELDS
                ),
            ),
            artifact_inputs=(
                *attachment.artifact_inputs,
                projection_edge(),
                ArtifactInput(
                    "health_completion_matrix",
                    MATRIX_NODE,
                    "matrix",
                    model_input.RECIPIENT_MATRIX_TYPE,
                ),
                *(
                    edge
                    for i, node in enumerate(applies)
                    for edge in (
                        ArtifactInput(
                            f"health_completion_raw_{i}",
                            node.id,
                            "raw_draw",
                            codec.RAW_TARGET_TYPE,
                        ),
                        ArtifactInput(
                            f"health_completion_state_{i}",
                            node.id,
                            "apply_state",
                            MATRIX_APPLY_STATE_TYPE,
                        ),
                    )
                ),
            ),
            description="Complete only ACS semantic gaps from one original draw; preserve all source observations/knownness and fan to existing clones.",
        )
    )
    return tuple(result)


def read_draws(qualified, artifacts):
    require(
        artifacts["health_completion_projection"].payload == qualified.projection,
        "PROJECTION_BYTES",
    )
    matrix_value = artifacts["health_completion_matrix"]
    require(matrix_value.payload == qualified.matrix, "MATRIX_BYTES")
    matrix = model_input.decode_recipient_matrix(qualified.matrix)
    require(tuple(matrix.features) == values.FEATURES, "MATRIX_FEATURES")
    result = pd.DataFrame(index=matrix.features.index)
    history = []
    for i, target in enumerate(values.TARGETS):
        raw_value = artifacts[f"health_completion_raw_{i}"]
        state_value = artifacts[f"health_completion_state_{i}"]
        packet = decode_matrix_apply_state(state_value.payload)
        application, chain = codec.read_application(
            codec.encode_json(packet["application"])
        )
        history.append({"target": target, "sha256": codec.sha(raw_value.payload)})
        require(
            raw_value.producer_key == state_value.producer_key
            and packet["matrix_sha256"] == codec.sha(qualified.matrix)
            and packet["matrix_producer_key"] == matrix_value.producer_key
            and chain.entity == "person"
            and tuple(chain.predictors) == values.FEATURES
            and tuple(chain.targets) == values.TARGETS
            and tuple(chain.completed_targets) == values.TARGETS[: i + 1]
            and chain.recipient_index == qrf._index_identity(matrix.features.index)
            and application["seed"] == values.SEED
            and application["raw_targets"] == history
            and len(application["models"]) == i + 1,
            "DRAW_CHAIN",
        )
        result[target] = codec.read_raw_target(
            raw_value.payload, target=target, index=matrix.features.index
        )
    return result


def result(qualified, observations, node, artifacts, people=None):
    if node.id == SOURCE_NODE:
        require(
            artifacts["health_projection"].payload == observations.projection,
            "OBSERVATION_PROJECTION",
        )
        return KernelResult(
            frame=values.source._copy_source(qualified.source_frame),
            artifacts={"projection": qualified.projection},
        )
    require(
        artifacts["health_completion_projection"].payload == qualified.projection,
        "PROJECTION_BYTES",
    )
    if node.id == COLUMNS_NODE:
        columns = values.model_columns(qualified.source_frame.person)
        require(columns.equals(qualified.columns), "MODEL_COLUMNS")
        return KernelResult(
            columns={("person", c): columns[c].copy(deep=True) for c in columns}
        )
    if node.id == DONOR_NODE:
        return KernelResult(keep=qualified.columns[values.ELIGIBLE].copy(deep=True))
    if node.id == MATRIX_NODE:
        require(
            artifacts["health_projection"].payload == observations.projection,
            "OBSERVATION_PROJECTION",
        )
        return KernelResult(artifacts={"matrix": qualified.matrix})
    require(node.id == observed.ATTACH_NODE, "NODE")
    # Existing literal and recode artifacts remain bound to the exact source.
    observed._check_artifacts(node, artifacts, observations)
    draws = read_draws(qualified, artifacts)
    columns = values.completed_columns(observations.raw, draws)
    attached = values.health.attach_columns(
        observations.origins, SimpleNamespace(person=people), columns
    )
    receipt = {
        "protocol": values.PROTOCOL,
        "projection_sha256": codec.sha(qualified.projection),
        "health_projection_sha256": codec.sha(observations.projection),
        "receiving_version": node.population,
        "rows": len(people),
        "source_knownness_preserved": True,
        "one_draw_per_original": True,
        "asec_observation_year": 2025,
        "acs_observation_year": 2024,
        "temporal_equivalence_claim": False,
        "scientific_qualification": "pending",
        "source_admission_issued": False,
        "release_eligible": False,
    }
    return KernelResult(
        columns=attached,
        artifacts={"attachment": codec.encode_json(receipt)},
        receipt=receipt,
    )


def expected_population(incoming, node, value):
    """Reconstruct the host's independently checked population transition."""
    if node.structural is StructuralDelta.FILTER:
        require(incoming is not None, "RECONSTRUCTION_BASE")
        frame = incoming.frame
        id_column = frame.schema.person_id_column
        ids = pd.Index(frame.person[id_column].to_numpy(), name=id_column)
        keep = value.keep
        require(
            isinstance(keep, pd.Series)
            and keep.index.is_unique
            and len(keep) == len(ids)
            and set(keep.index) == set(ids)
            and pd.api.types.is_bool_dtype(keep.dtype)
            and not keep.isna().any(),
            "RECONSTRUCTION_FILTER_MASK",
        )
        value = replace(
            value,
            frame=frame.select(keep.reindex(ids).to_numpy(dtype=bool, copy=True)),
            keep=None,
        )
    return (
        population_ops.Population.from_frame(value.frame, node.id)
        if node.structural is StructuralDelta.CREATE
        else population_ops.patch(incoming, node, value)
    )


def result_stamp(value):
    return (
        None if value.frame is None else values.source._frame_identity(value.frame),
        None
        if value.keep is None
        else physical._table_stamp(value.keep.to_frame("keep")),
        tuple(
            (key, physical._table_stamp(column.to_frame(key[1])))
            for key, column in sorted(value.columns.items())
        ),
        tuple(sorted(value.artifacts.items())),
        codec.encode_json(value.receipt),
    )


class _Kernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=predictors._Kernel.capabilities.dependencies,
    )

    def __init__(self, boundary):
        self.boundary = boundary

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            values,
            values.literals,
            values.health,
            values.demographics,
            values.demographics.demographic,
            values.demographics.household,
            values.source,
            values.source.asec_native,
            values.source.acs_native,
            values.literals.asec,
            values.literals.records,
            values.literals.housing,
            values.literals.source_csv_builtin,
            observed,
            physical,
            predictors,
            model_input,
            qrf,
            qrf_target,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        boundary = self.boundary
        boundary.context(context)
        require(context.node in boundary.health_completion_nodes, "NODE_DECLARATION")
        qualified = boundary.health_completion
        if context.node.id == COLUMNS_NODE:
            expected = qualified.source_frame.person.set_index("person_id")
            actual = context.tables["person"].set_index("person_id")
            for selection in context.node.inputs:
                for column in selection.columns:
                    require(
                        population_ops.storage_equal(actual[column], expected[column]),
                        "SOURCE_SLICE",
                    )
        if context.node.id == DONOR_NODE:
            actual = context.tables["person"].set_index("person_id")[values.ELIGIBLE]
            require(
                population_ops.storage_equal(
                    actual, qualified.columns[values.ELIGIBLE]
                ),
                "ELIGIBLE_SLICE",
            )
        output = result(
            qualified,
            boundary.health,
            context.node,
            context.artifacts,
            context.tables.get("person"),
        )
        stamp = result_stamp(output)
        boundary.pure()
        require(result_stamp(output) == stamp, "RESULT_CHANGED")
        return output


class SourceKernel(_Kernel):
    ref = "us.survey_health_completion.full_asec_source@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.CREATE)


class ColumnsKernel(_Kernel):
    ref = "us.survey_health_completion.model_columns@1"


class DonorKernel(_Kernel):
    ref = "us.survey_health_completion.known_donor@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.FILTER)


class MatrixKernel(_Kernel):
    ref = "us.survey_health_completion.original_acs_matrix@1"


class AttachKernel(_Kernel):
    ref = "us.survey_health_completion.attach@1"


def kernels(boundary):
    return (
        tuple(
            cls(boundary)
            for cls in (
                SourceKernel,
                ColumnsKernel,
                DonorKernel,
                MatrixKernel,
                AttachKernel,
            )
        )
        if boundary.health_completion is not None
        else ()
    )


def verify_models(boundary, loaded, donor):
    """Bind fitted artifacts to exact full-source donor values and typed weights."""
    if boundary.health_completion is None:
        require(donor is None, "UNEXPECTED_DONOR")
        return
    require(donor is not None, "MISSING_DONOR")
    first = boundary.compiled.graph.node(FIT_PREFIX + ".000")
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
            model_frame, list(values.FEATURES), list(values.TARGETS), weights="design"
        )
    )
    history = []
    for i, target in enumerate(values.TARGETS):
        node = f"{FIT_PREFIX}.{i:03d}"
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
                model_frame.person, (*values.FEATURES, *values.TARGETS[:i], target)
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
                loaded[f"{APPLY_PREFIX}.{i:03d}", "apply_state"]
            )["application"]["models"]
            == history,
            "TRAINING_APPLY_HISTORY",
        )
        before = after
