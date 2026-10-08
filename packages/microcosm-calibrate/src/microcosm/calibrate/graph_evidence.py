"""Aggregate graph evidence supplied through shared calibration codecs."""

from __future__ import annotations

import numpy as np

from microcosm.frame import Frame
from microcosm.graph.evidence import ArtifactSummaryContext

from .artifacts import (
    PROBLEM_TYPE,
    RESULT_TYPE,
    SOLUTION_TYPE,
    decode_calibration_result,
    decode_problem,
    decode_solution,
)


def weight_summary(weights) -> dict:
    values = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("Weight summary requires finite nonnegative weights.")
    total = float(values.sum())
    squares = float(np.square(values).sum())
    return {
        "records": len(values),
        "nonzero": int(np.count_nonzero(values)),
        "total": total,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "quantiles": dict(
            zip(
                ("p0", "p25", "p50", "p75", "p90", "p99", "p100"),
                np.quantile(values, [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).tolist(),
                strict=True,
            )
        ),
        "effective_sample_size": total * total / squares if squares else 0.0,
    }


def problem_summary(context: ArtifactSummaryContext) -> dict:
    ordered = decode_problem(context.payload)
    return {
        "overview": {
            "problem_sha256": ordered.sha256,
            "weight_entity": ordered.problem.weight_entity,
            "initial_weight_kind": ordered.problem.initial_weights.kind.value,
            "targets": len(ordered.problem.targets),
            "matrix_shape": list(ordered.problem.matrix.shape),
            "matrix_nonzero": int(ordered.problem.matrix.nnz),
            "initial_weights": weight_summary(ordered.problem.initial_weights.values),
            "skipped_targets": len(ordered.problem.skipped),
        }
    }


def _problem_payload(context: ArtifactSummaryContext) -> bytes:
    receipt = context.run.manifest.nodes[context.node_id]
    for direction in ("outputs", "inputs"):
        for entry in receipt.typed_artifacts.get(direction, {}).values():
            if entry["type"] == {
                "name": PROBLEM_TYPE.name,
                "schema_version": PROBLEM_TYPE.schema_version,
            }:
                return context.store.load_bytes(entry["key"])
    raise ValueError("Calibration summary requires a declared ordered problem.")


def solution_summary(context: ArtifactSummaryContext) -> dict:
    solution = decode_solution(context.payload)
    # Installation solutions in an exact-count refit bind the original problem;
    # the separately declared refit solution binds the compact problem.
    receipt = context.run.manifest.nodes[context.node_id]
    problems = [
        decode_problem(context.store.load_bytes(entry["key"]))
        for direction in ("outputs", "inputs")
        for entry in receipt.typed_artifacts.get(direction, {}).values()
        if entry["type"]
        == {"name": PROBLEM_TYPE.name, "schema_version": PROBLEM_TYPE.schema_version}
    ]
    problem = next(
        (item for item in problems if item.sha256 == solution.problem_sha256), None
    )
    selected = set(solution.entity_ids)
    if (
        problem is None
        or tuple(identity for identity in problem.entity_ids if identity in selected)
        != solution.entity_ids
    ):
        raise ValueError("Solution summary does not match a declared problem.")
    return {
        "overview": {
            "problem_sha256": solution.problem_sha256,
            "weights": weight_summary(solution.weights),
        }
    }


def result_summary_data(result, ordered) -> dict:
    """Project recorded diagnostics, never record axes or individual weights."""
    rows = []
    for index, diagnostic in enumerate(result.diagnostics):
        target = ordered.problem.targets[index]
        metadata = ordered.target_metadata[index]
        se = metadata.get("se")
        rows.append(
            {
                "name": diagnostic.name,
                "entity": target.entity,
                "period": target.period,
                "target": diagnostic.target,
                "initial": diagnostic.initial_estimate,
                "achieved": diagnostic.final_estimate,
                "residual": diagnostic.final_estimate - diagnostic.target,
                "relative_error": diagnostic.relative_error,
                "tolerance": target.tolerance,
                "within_tolerance": diagnostic.within_tolerance,
                "uncertainty": {"status": "recorded", "se": se}
                if isinstance(se, (int, float))
                and not isinstance(se, bool)
                and np.isfinite(se)
                and se > 0
                else {"status": "not_recorded"},
                "source": target.source,
            }
        )
    return {
        "overview": {
            "problem_sha256": ordered.sha256,
            "weight_entity": result.weight_entity,
            "initial_weight_kind": ordered.problem.initial_weights.kind.value,
            "final_weight_kind": result.frame.weights_for(
                result.weight_entity
            ).kind.value,
            "initial_weights": weight_summary(result.initial_weights),
            "final_weights": weight_summary(result.weights),
            "target_count": len(rows),
            "final_loss": result.final_loss,
            "l0_lambda": result.l0_lambda,
            "record_scope": "Target aggregates and weight distribution; no entity IDs or record values.",
        },
        "tables": {"targets": rows},
    }


def result_summary(context: ArtifactSummaryContext) -> dict:
    ordered = decode_problem(_problem_payload(context))
    version = context.run.compiled.versions[context.node_id]
    receipt = context.run.manifest.nodes[version]
    if receipt.frame_key is None:
        raise ValueError("Calibration summary requires a saved population boundary.")
    frame = context.store.load_frame(receipt.frame_key)
    entity = ordered.problem.weight_entity
    id_column = frame.schema.entity_id_column(entity)
    if tuple(frame.table(entity)[id_column]) != ordered.entity_ids:
        person = frame.schema.person_entity
        membership = (
            id_column if entity == person else frame.schema.membership_column(entity)
        )
        frame = frame.select(
            frame.table(person)[membership].isin(ordered.entity_ids).to_numpy()
        )
    if tuple(frame.table(entity)[id_column]) != ordered.entity_ids:
        raise ValueError("Calibration summary differs from its saved population axis.")
    initial_frame = Frame(
        {
            **{e: frame.table(e) for e in frame.entities},
            **{link.name: frame.link(link.name) for link in frame.schema.links},
        },
        frame.schema,
        {entity: ordered.problem.initial_weights},
        frame.strata,
    )
    result = decode_calibration_result(
        context.payload, frame=initial_frame, problem=ordered
    )
    outputs = context.run.manifest.nodes[context.node_id].typed_artifacts.get(
        "outputs", {}
    )
    matching_solutions = [
        decode_solution(context.store.load_bytes(entry["key"]))
        for entry in outputs.values()
        if entry["type"]
        == {"name": SOLUTION_TYPE.name, "schema_version": SOLUTION_TYPE.schema_version}
    ]
    for solution in matching_solutions:
        if solution.problem_sha256 == ordered.sha256:
            if solution.entity_ids != ordered.entity_ids or not np.array_equal(
                solution.weights, result.weights
            ):
                raise ValueError(
                    "Calibration result disagrees with its recorded solution."
                )
    return result_summary_data(result, ordered)


CALIBRATION_SUMMARY_PROVIDERS = {
    (PROBLEM_TYPE.name, PROBLEM_TYPE.schema_version): problem_summary,
    (SOLUTION_TYPE.name, SOLUTION_TYPE.schema_version): solution_summary,
    (RESULT_TYPE.name, RESULT_TYPE.schema_version): result_summary,
}
