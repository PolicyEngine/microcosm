"""Declarations for explicit legacy donor-fit and recipient-apply target chains.

Callers own scientifically reviewed preparation, donor support, encoding, and
finalization. Donor populations must be independent of recipient source ancestry
to reuse fits when recipient rungs change. Table indices remain the existing
pandas indices; graph entity IDs are never substituted as QRF checkpoint indices.
"""

from __future__ import annotations

from dataclasses import replace

from microcosm.fit._graph_legacy_qrf import (
    APPLY_STATE_TYPE,
    RAW_TARGET_TYPE,
    TRAINING_STATE_TYPE,
    names,
    read_raw_target,
)
from microcosm.fit.graph_legacy_apply import LegacyQRFApplyKernel
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.fit.qrf import DEFAULT_N_ESTIMATORS, DEFAULT_ZERO_ATOL
from microcosm.fit.qrf_target import LEGACY_QRF_TARGET_TYPE
from microcosm.graph import ArtifactInput, ArtifactOutput, Node, Slice

__all__ = [
    "LegacyQRFTrainKernel",
    "LegacyQRFApplyKernel",
    "TRAINING_STATE_TYPE",
    "APPLY_STATE_TYPE",
    "RAW_TARGET_TYPE",
    "legacy_qrf_train_nodes",
    "legacy_qrf_apply_nodes",
    "legacy_qrf_apply_matrix_nodes",
    "read_raw_target",
]


def legacy_qrf_train_nodes(
    prefix,
    *,
    population,
    entity,
    predictors,
    targets,
    seed,
    phase,
    n_estimators=DEFAULT_N_ESTIMATORS,
    zero_atol=DEFAULT_ZERO_ATOL,
    max_samples_leaf=None,
):
    """Declare independent per-target fits; each successor reads small state only."""
    predictors = names(predictors, "predictors")
    targets = names(targets, "targets")
    nodes = []
    for i, target in enumerate(targets):
        nodes.append(
            Node(
                f"{prefix}.{i:03d}",
                LegacyQRFTrainKernel.ref,
                population=population,
                inputs=(Slice(entity, (*predictors, *targets)),),
                params={
                    "predictors": predictors,
                    "targets": targets,
                    "target": target,
                    "seed": seed,
                    "n_estimators": n_estimators,
                    "zero_atol": zero_atol,
                    "max_samples_leaf": max_samples_leaf,
                    "phase": phase,
                },
                artifact_inputs=()
                if i == 0
                else (
                    ArtifactInput(
                        "training_state",
                        nodes[-1].id,
                        "training_state",
                        TRAINING_STATE_TYPE,
                    ),
                ),
                artifact_outputs=(
                    ArtifactOutput("model", LEGACY_QRF_TARGET_TYPE),
                    ArtifactOutput("training_state", TRAINING_STATE_TYPE),
                ),
            )
        )
    return tuple(nodes)


def legacy_qrf_apply_nodes(prefix, *, population, fit_nodes, seed, phase):
    """Declare recipient-only draws with immutable ordered raw target dependencies."""
    if not isinstance(fit_nodes, tuple) or not fit_nodes:
        raise ValueError("fit_nodes must be a nonempty ordered tuple.")
    first = fit_nodes[0]
    expected_targets = first.params["targets"]
    if len(fit_nodes) != len(expected_targets):
        raise ValueError("fit_nodes must contain the complete target chain.")
    nodes = []
    for i, fit in enumerate(fit_nodes):
        if (
            fit.kernel != LegacyQRFTrainKernel.ref
            or fit.params["target"] != expected_targets[i]
            or fit.params["targets"] != expected_targets
            or fit.inputs != first.inputs
            or fit.population != first.population
        ):
            raise ValueError("fit_nodes must describe one ordered donor chain.")
        edges = [
            ArtifactInput("model", fit.id, "model", LEGACY_QRF_TARGET_TYPE),
            ArtifactInput(
                "training_state", fit.id, "training_state", TRAINING_STATE_TYPE
            ),
        ]
        if nodes:
            edges.append(
                ArtifactInput(
                    "apply_state", nodes[-1].id, "apply_state", APPLY_STATE_TYPE
                )
            )
            edges.extend(
                ArtifactInput(f"prior_{j:03d}", node.id, "raw_draw", RAW_TARGET_TYPE)
                for j, node in enumerate(nodes)
            )
        nodes.append(
            Node(
                f"{prefix}.{i:03d}",
                LegacyQRFApplyKernel.ref,
                population=population,
                inputs=(Slice(fit.inputs[0].entity, fit.params["predictors"]),),
                params={"target": fit.params["target"], "seed": seed, "phase": phase},
                artifact_inputs=tuple(edges),
                artifact_outputs=(
                    ArtifactOutput("raw_draw", RAW_TARGET_TYPE),
                    ArtifactOutput("apply_state", APPLY_STATE_TYPE),
                ),
            )
        )
    return tuple(nodes)


def legacy_qrf_apply_matrix_nodes(
    prefix,
    *,
    population,
    fit_nodes,
    matrix_producer,
    matrix_artifact="matrix",
    seed,
    phase,
):
    """Declare one fixed recipient-matrix producer for the complete target chain."""
    from microcosm.fit.graph_legacy_apply_matrix import (
        MATRIX_APPLY_STATE_TYPE,
        LegacyQRFApplyMatrixKernel,
    )
    from microcosm.fit.model_input import RECIPIENT_MATRIX_TYPE

    nodes = legacy_qrf_apply_nodes(
        prefix, population=population, fit_nodes=fit_nodes, seed=seed, phase=phase
    )
    matrix = ArtifactInput(
        "matrix", matrix_producer, matrix_artifact, RECIPIENT_MATRIX_TYPE
    )
    return tuple(
        replace(
            node,
            kernel=LegacyQRFApplyMatrixKernel.ref,
            inputs=(),
            artifact_inputs=(
                *(
                    replace(edge, type=MATRIX_APPLY_STATE_TYPE)
                    if edge.name == "apply_state"
                    else edge
                    for edge in node.artifact_inputs
                ),
                matrix,
            ),
            artifact_outputs=(
                ArtifactOutput("raw_draw", RAW_TARGET_TYPE),
                ArtifactOutput("apply_state", MATRIX_APPLY_STATE_TYPE),
            ),
        )
        for node in nodes
    )
