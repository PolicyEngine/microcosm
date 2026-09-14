"""Current survey-first E00900 diagnostic placement, not donor admission.

The source-qualified host is the accepted ACS+ASEC allocation and whole-household
clone. Donor qualification belongs to a separately reviewed upstream producer.
This module writes a development tax-unit diagnostic, never an engine input.
The mandatory materialized check also runs after replay, using the retained live
preparation and the real executor-observed Population.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from microcosm.fit import model_input
from microcosm.fit.graph_legacy_apply_matrix import (
    MATRIX_APPLY_STATE_TYPE,
    decode_matrix_apply_state,
)
from microcosm.fit.graph_legacy_qrf import (
    legacy_qrf_apply_matrix_nodes,
    legacy_qrf_train_nodes,
)
from microcosm.frame import US_SCHEMA
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    KernelResult,
    Node,
    Owned,
    Slice,
    source_hash,
)
from microcosm.graph import population as population_ops

from . import graph_puf_diagnostic_consumer as host
from . import survey_population_replay as replay

shared, detail, codec, require = host.shared, host.detail, host.codec, host.require
SCOPE = "current_survey_e00900_development_not_tax_inputs"
PLACEMENT_TYPE = ArtifactType("microcosm.us.current_survey_puf_placement", 1)
PLACEMENT_NODE = "puf_diagnostic.current_survey_placement"
APPLY_PREFIX = "puf_diagnostic.current_survey_apply"
ATTACH_NODE = "puf_diagnostic.current_survey_attach"
SEED = 578


def _inputs(frame):
    values = []
    for entity in US_SCHEMA.entities:
        structural = {US_SCHEMA.entity_id_column(entity)}
        if entity == US_SCHEMA.person_entity:
            structural.update(
                US_SCHEMA.membership_column(e) for e in US_SCHEMA.group_entities
            )
        values.append(
            Slice(entity, tuple(c for c in frame.table(entity) if c not in structural))
        )
    return tuple(values)


def _edges():
    return (
        ArtifactInput(
            "projection",
            host.CURRENT_PROJECTION_NODE,
            "projection",
            host.CURRENT_PROJECTION_TYPE,
        ),
        ArtifactInput(
            "matrix",
            host.CURRENT_MATRIX_NODE,
            "matrix",
            model_input.RECIPIENT_MATRIX_TYPE,
        ),
    )


def _attach_edges():
    return (
        *_edges(),
        ArtifactInput("placement", PLACEMENT_NODE, "placement", PLACEMENT_TYPE),
        ArtifactInput(
            "raw_draw", APPLY_PREFIX + ".000", "raw_draw", codec.RAW_TARGET_TYPE
        ),
        ArtifactInput(
            "apply_state", APPLY_PREFIX + ".000", "apply_state", MATRIX_APPLY_STATE_TYPE
        ),
    )


def _placement_node(frame):
    require(
        detail.MASK not in frame.table("tax_unit")
        and detail.OUTPUT not in frame.table("tax_unit"),
        "SURVEY_TRANSFER_ALREADY_PRESENT",
    )
    return Node(
        PLACEMENT_NODE,
        CurrentSurveyPlacementKernel.ref,
        population=host.survey_clone.COMBINED_CLONE_NODE,
        inputs=_inputs(frame),
        outputs=(Owned("tax_unit", detail.MASK, "bool"),),
        params={"scope": SCOPE},
        artifact_inputs=_edges(),
        artifact_outputs=(ArtifactOutput("placement", PLACEMENT_TYPE),),
    )


def _attach_node(frame):
    inputs = tuple(
        Slice(
            s.entity, (*s.columns, detail.MASK) if s.entity == "tax_unit" else s.columns
        )
        for s in _inputs(frame)
    )
    return Node(
        ATTACH_NODE,
        CurrentSurveyAttachKernel.ref,
        population=host.survey_clone.COMBINED_CLONE_NODE,
        inputs=inputs,
        outputs=(Owned("tax_unit", detail.OUTPUT, "float64", rows=detail.MASK),),
        params={"scope": SCOPE},
        artifact_inputs=_attach_edges(),
    )


def _binding(projection, matrix, matrix_producer_key, frame, mask):
    return codec.encode_json(
        {
            "scope": SCOPE,
            "projection_sha256": codec.sha(projection),
            "matrix_sha256": codec.sha(matrix),
            "matrix_producer_key": matrix_producer_key,
            "features": list(detail.FEATURES),
            "targets": [detail.TARGET],
            "output": detail.OUTPUT,
            "selected_ids": frame.table("tax_unit")
            .tax_unit_id.to_numpy()[mask]
            .tolist(),
            "source_admission": detail.SOURCE_ADMISSION,
            "release_eligible": False,
        }
    )


def _mask_result(frame, mask):
    return KernelResult(
        columns={
            ("tax_unit", detail.MASK): pd.Series(
                mask, index=frame.table("tax_unit").tax_unit_id, dtype=bool
            ),
        }
    )


def _raw_values(matrix, matrix_producer_key, raw_draw, apply_state):
    packet = decode_matrix_apply_state(apply_state)
    require(
        packet["matrix_sha256"] == codec.sha(matrix)
        and packet["matrix_producer_key"] == matrix_producer_key,
        "SURVEY_TRANSFER_MATRIX_PRODUCER",
    )
    application, chain = codec.read_application(
        codec.encode_json(packet["application"])
    )
    require(
        tuple(chain.completed_targets) == (detail.TARGET,)
        and tuple(chain.predictors) == detail.FEATURES
        and application["seed"] == SEED
        and application["raw_targets"]
        == [
            {
                "target": detail.TARGET,
                "sha256": codec.sha(raw_draw),
            }
        ],
        "SURVEY_TRANSFER_RAW_STATE",
    )
    prepared = model_input.decode_recipient_matrix(matrix)
    require(
        chain.entity == prepared.entity == "tax_unit"
        and tuple(chain.targets) == (detail.TARGET,)
        and chain.recipient_index
        == shared.qrf._index_identity(prepared.features.index),
        "SURVEY_TRANSFER_RAW_INDEX",
    )
    values = codec.read_raw_target(
        raw_draw, target=detail.TARGET, index=prepared.features.index
    )
    require(np.isfinite(values).all(), "SURVEY_TRANSFER_NONFINITE_DRAW")
    return pd.Series(values, index=prepared.entity_ids, dtype="float64")


class _CurrentTransferKernel(host._CurrentSurveyKernel):
    def __init__(self, preparation, allocated_population, clone_population):
        self.preparation = preparation
        self.allocated_population = allocated_population
        self.clone_population = clone_population

    def implementation_hash(self):
        # Independent extension closure; do not relabel or refresh survey inventory.
        return codec.sha(
            codec.encode_json(
                {
                    "host": super().implementation_hash(),
                    "transfer": source_hash(
                        sys.modules[__name__],
                        population_ops,
                        replay,
                        sys.modules[decode_matrix_apply_state.__module__],
                        sys.modules[legacy_qrf_apply_matrix_nodes.__module__],
                        dependencies=self.capabilities.dependencies,
                    ),
                }
            )
        )

    def _qualified(self, context):
        expected_projection, expected_matrix, mask, evidence = (
            host.qualify_current_survey_host(
                self.preparation, self.allocated_population, self.clone_population
            )
        )
        projection = shared.artifact(
            context, "projection", host.CURRENT_PROJECTION_TYPE
        )
        matrix = shared.artifact(context, "matrix", model_input.RECIPIENT_MATRIX_TYPE)
        require(
            projection.payload == expected_projection
            and matrix.payload == expected_matrix,
            "SURVEY_TRANSFER_HOST_ARTIFACTS",
        )
        return projection, matrix, mask, evidence


class CurrentSurveyPlacementKernel(_CurrentTransferKernel):
    ref = "us.puf_diagnostic.current_survey_placement@1"

    def run(self, context):
        require(
            context.node == _placement_node(self.clone_population.frame)
            and not context.sources
            and set(context.artifacts) == {"projection", "matrix"},
            "SURVEY_TRANSFER_PLACEMENT_DECLARATION",
        )
        projection, matrix, mask, evidence = self._qualified(context)
        host._current_context_frame(context, self.clone_population.frame)
        result = _mask_result(self.clone_population.frame, mask)
        return KernelResult(
            columns=result.columns,
            artifacts={
                "placement": _binding(
                    projection.payload,
                    matrix.payload,
                    matrix.producer_key,
                    self.clone_population.frame,
                    mask,
                )
            },
            receipt={
                **evidence,
                "scope": SCOPE,
                "source_admission": detail.SOURCE_ADMISSION,
            },
        )


class CurrentSurveyAttachKernel(_CurrentTransferKernel):
    ref = "us.puf_diagnostic.current_survey_attach@1"

    def run(self, context):
        require(
            context.node == _attach_node(self.clone_population.frame)
            and not context.sources
            and set(context.artifacts)
            == {"projection", "matrix", "placement", "raw_draw", "apply_state"},
            "SURVEY_TRANSFER_ATTACH_DECLARATION",
        )
        projection, matrix, mask, evidence = self._qualified(context)
        placement = shared.artifact(context, "placement", PLACEMENT_TYPE)
        raw = shared.artifact(context, "raw_draw", codec.RAW_TARGET_TYPE)
        state = shared.artifact(context, "apply_state", MATRIX_APPLY_STATE_TYPE)
        shared.siblings(context, ("raw_draw", "apply_state"))
        require(
            placement.payload
            == _binding(
                projection.payload,
                matrix.payload,
                matrix.producer_key,
                self.clone_population.frame,
                mask,
            ),
            "SURVEY_TRANSFER_PLACEMENT_BINDING",
        )
        # Comparison-only normal patch; no new source/successor owner is issued.
        masked = population_ops.patch(
            self.clone_population,
            _placement_node(self.clone_population.frame),
            _mask_result(self.clone_population.frame, mask),
        )
        host._current_context_frame(context, masked.frame)
        values = _raw_values(
            matrix.payload, matrix.producer_key, raw.payload, state.payload
        )
        return KernelResult(
            columns={("tax_unit", detail.OUTPUT): values},
            receipt={
                **evidence,
                "scope": SCOPE,
                "source_admission": detail.SOURCE_ADMISSION,
                "raw_sha256": codec.sha(raw.payload),
                "native_output_cells_written": 0,
                "host_weights_changed": False,
            },
        )


def current_survey_transfer_nodes(frame, *, fit_nodes):
    """Connect the independently qualified donor fits to the accepted host matrix.

    This E00900 mechanism does not admit a donor source. A future complete PUF
    chain uses the same maintained train/apply family with its full ordered
    targets and separately reviewed person/tax-unit finalization contract.
    """
    require(
        type(fit_nodes) is tuple and len(fit_nodes) == 1, "SURVEY_TRANSFER_FIT_CHAIN"
    )
    fit = fit_nodes[0]
    expected = legacy_qrf_train_nodes(
        "__comparison__",
        population=fit.population,
        entity="tax_unit",
        predictors=detail.FEATURES,
        targets=(detail.TARGET,),
        seed=SEED,
        phase=SCOPE,
        n_estimators=2,
        zero_atol=0,
    )[0]
    from dataclasses import replace

    require(fit == replace(expected, id=fit.id), "SURVEY_TRANSFER_FIT_DECLARATION")
    applies = legacy_qrf_apply_matrix_nodes(
        APPLY_PREFIX,
        population=host.survey_clone.COMBINED_CLONE_NODE,
        fit_nodes=fit_nodes,
        matrix_producer=host.CURRENT_MATRIX_NODE,
        seed=SEED,
        phase=SCOPE,
    )
    return (_placement_node(frame), *applies, _attach_node(frame))


def verify_materialized_current_survey_transfer(
    preparation,
    allocated_population,
    clone_population,
    *,
    population,
    projection,
    matrix,
    matrix_producer_key,
    placement,
    raw_draw,
    apply_state,
):
    """Mandatory after actual typed-artifact/key/store checks, cold and replay.

    Inputs are the actual loaded payloads and producer key, not receipt labels.
    Last source I/O requalifies the retained host. Pure comparison then verifies
    every native/clone cell, axis, metadata value, weight, owner and mass ledger.
    Reconstructed expected populations are comparison values, never source owners.
    """
    require(codec._hash(matrix_producer_key), "SURVEY_TRANSFER_MATRIX_KEY")
    expected_projection, expected_matrix, mask, evidence = (
        host.qualify_current_survey_host(
            preparation, allocated_population, clone_population
        )
    )
    require(
        projection == expected_projection and matrix == expected_matrix,
        "SURVEY_TRANSFER_MATERIALIZED_HOST",
    )
    require(
        placement
        == _binding(
            projection, matrix, matrix_producer_key, clone_population.frame, mask
        ),
        "SURVEY_TRANSFER_PLACEMENT_BINDING",
    )
    values = _raw_values(matrix, matrix_producer_key, raw_draw, apply_state)
    expected = population_ops.patch(
        clone_population,
        _placement_node(clone_population.frame),
        _mask_result(clone_population.frame, mask),
    )
    expected = population_ops.patch(
        expected,
        _attach_node(clone_population.frame),
        KernelResult(columns={("tax_unit", detail.OUTPUT): values}),
    )
    replay.same_replayed_population(expected, population)
    return {**evidence, "scope": SCOPE, "source_admission": detail.SOURCE_ADMISSION}
