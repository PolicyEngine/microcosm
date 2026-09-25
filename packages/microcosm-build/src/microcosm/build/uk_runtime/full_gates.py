"""Complete UK gate ownership for one full build and explicit target filters.

Country-only target selection removes local fit claims, while geographic
integrity, source coverage, register completeness and population checks remain.
Gate evaluation and persistence use the shared battery and graph artifacts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse

from microcosm.build.country_spec import GatesManifest, load_country_spec
from microcosm.build.gate_battery import EvidenceContext
from microcosm.build.uk_runtime.calibration_run import uk_aggregate_admin_totals
from microcosm.build.uk_runtime.graph_evidence import uk_spine_gate_artifacts
from microcosm.build.uk_runtime.parity_reference import load_efrs_parity_reference
from microcosm.calibrate import TargetRegistry
from microcosm.calibrate.artifacts import OrderedProblem, OrderedSolution
from microcosm.calibrate.solve import CalibrationResult, _build_diagnostics
from microcosm.frame import Frame
from microcosm.graph.canonical import canonical_json

_LOCAL_FIT_GATES = frozenset(
    {"uk_local_target_fit", "uk_local_per_family_fit", "uk_local_area_support"}
)
_LOCAL_LEVELS = frozenset({"constituency", "local_authority", "la"})


def _scope(selection_receipt: Mapping[str, Any] | None) -> tuple[bool, dict[str, str]]:
    if selection_receipt is None:
        return True, {}
    if selection_receipt.get("schema") != "microcosm.calibrate.target-selection.v1":
        raise ValueError("UK gate scope requires a versioned target-selection receipt.")
    included = selection_receipt.get("included")
    if not isinstance(included, list) or not included:
        raise ValueError("UK gate scope requires nonempty selected targets.")
    levels = {str(row["geography_level"]) for row in included}
    unknown = levels - (_LOCAL_LEVELS | {"country", "region"})
    if unknown:
        raise ValueError(
            f"UK gate scope has unknown geography levels {sorted(unknown)}."
        )
    has_local = bool(levels & _LOCAL_LEVELS)
    selector = selection_receipt.get("selector", {})
    if selector.get("geography_levels") is None:
        # An unfiltered source that omits all local targets is a completeness
        # failure, not permission to declare national-only scope.
        if not has_local:
            raise ValueError("An unfiltered UK full build must include local targets.")
        return True, {}
    excluded = (
        {}
        if has_local
        else {
            gate_id: "No local targets were selected; local fit and per-area fit-support claims are inapplicable."
            for gate_id in sorted(_LOCAL_FIT_GATES)
        }
    )
    return has_local, excluded


def uk_full_gate_manifest(
    selection_receipt: Mapping[str, Any] | None = None,
    *,
    source: GatesManifest | None = None,
) -> GatesManifest:
    """Return every declared UK gate, except explicitly inapplicable fit checks."""
    source = load_country_spec("uk").gates if source is None else source
    _, exclusions = _scope(selection_receipt)
    return replace(
        source,
        policy=f"{source.policy}; full_build_scope",
        gates=tuple(entry for entry in source.gates if entry.id not in exclusions),
    )


def uk_full_gate_scope_receipt(
    selection_receipt: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Describe selected-fit scope without changing invariant gate ownership."""
    has_local, exclusions = _scope(selection_receipt)
    return {
        "schema": "microcosm.build.uk.full-gate-scope.v1",
        "posture": "full_build" if has_local else "full_build_filtered_targets",
        "local_fit_claim": has_local,
        "scope_exclusions": exclusions,
        "target_selection": None
        if selection_receipt is None
        else dict(selection_receipt),
    }


def build_full_gate_context(
    frame: Frame,
    *,
    ordered_problem: OrderedProblem,
    solution: OrderedSolution,
    selection_receipt: Mapping[str, Any],
    stage_evidence: Mapping[str, Mapping[str, Any]],
    supporting_evidence: Mapping[str, Any],
) -> EvidenceContext:
    """Reconstruct gate evidence from identified rows and the installed solution.

    Supporting source/validation artifacts are forwarded under their existing
    gate-binding names. Missing artifacts remain missing so shared evaluation
    records an evidence failure; measured quantities are never filled with zero.
    """
    _scope(selection_receipt)
    bindings = ordered_problem.bindings
    embedded = bindings.get("target_selection")
    digest = bindings.get("target_selection_sha256")
    if embedded is None and digest is None:
        raise ValueError("Gate problem has no target-selection binding.")
    if embedded is not None and canonical_json(embedded) != canonical_json(
        selection_receipt
    ):
        raise ValueError("Gate target selection differs from the problem binding.")
    if (
        digest is not None
        and digest != hashlib.sha256(canonical_json(selection_receipt)).hexdigest()
    ):
        raise ValueError(
            "Gate target-selection digest differs from the problem binding."
        )
    if solution.problem_sha256 != ordered_problem.sha256:
        raise ValueError("Gate solution belongs to a different ordered problem.")
    if solution.entity_ids != ordered_problem.entity_ids:
        raise ValueError("Gate solution and problem have different household axes.")
    entity = ordered_problem.problem.weight_entity
    ids = tuple(frame.table(entity)[frame.schema.entity_id_column(entity)])
    if ids != ordered_problem.entity_ids:
        raise ValueError(
            "Gate population differs from the ordered problem household axis."
        )
    if not np.array_equal(frame.weights_for(entity).values, solution.weights):
        raise ValueError("Gate population does not carry the bound solution weights.")
    if ordered_problem.problem.skipped:
        raise ValueError(
            "Full-build gate context cannot omit skipped selected targets."
        )
    problem = ordered_problem.problem
    if len(ordered_problem.target_metadata) != len(problem.targets):
        raise ValueError("Gate target metadata does not align with the ordered matrix.")
    calibration_result = supporting_evidence.get("calibration_result")
    if calibration_result is None:
        solved_rows = _build_diagnostics(
            problem, frame, problem.initial_weights.values, solution.weights
        )
    else:
        if not isinstance(calibration_result, CalibrationResult):
            raise TypeError(
                "Gate calibration_result must be a decoded CalibrationResult."
            )
        result_problem = calibration_result.problem
        result_ids = tuple(
            calibration_result.frame.table(entity)[
                calibration_result.frame.schema.entity_id_column(entity)
            ]
        )
        if (
            calibration_result.weight_entity != entity
            or result_ids != ordered_problem.entity_ids
            or not np.array_equal(calibration_result.weights, solution.weights)
            or not np.array_equal(
                calibration_result.initial_weights, problem.initial_weights.values
            )
            or result_problem.names != problem.names
            or result_problem.matrix.shape != problem.matrix.shape
            or (
                sparse.csr_array(result_problem.matrix)
                != sparse.csr_array(problem.matrix)
            ).nnz
            or not np.array_equal(result_problem.target_vector, problem.target_vector)
            or calibration_result.skipped
        ):
            raise ValueError(
                "Gate calibration_result differs from the bound numerical problem/solution."
            )
        solved_rows = calibration_result.diagnostics
        if tuple(
            row.name for row in solved_rows
        ) != problem.names or not np.array_equal(
            [row.target for row in solved_rows], problem.target_vector
        ):
            raise ValueError(
                "Gate calibration_result diagnostics differ from the target axis."
            )
    national_errors: dict[str, float] = {}
    local_rows = []
    diagnostics = []
    for index, (target, metadata) in enumerate(
        zip(problem.targets, ordered_problem.target_metadata, strict=True)
    ):
        level = str(metadata.get("geography_level", ""))
        materialization = metadata.get("materialization")
        if level not in _LOCAL_LEVELS | {"country", "region"}:
            raise ValueError(f"Target {target.row_name!r} lacks classified geography.")
        if materialization not in {"uk_local_surface", "uk_national_measure"}:
            raise ValueError(
                f"Target {target.row_name!r} lacks a materialization owner."
            )
        row = {
            "name": target.row_name,
            "target_name": target.row_name,
            "target": float(target.value),
            "estimate": float(solved_rows[index].final_estimate),
            "relative_error": float(solved_rows[index].relative_error),
            "abs_relative_error": float(abs(solved_rows[index].relative_error)),
            "family": str(metadata.get("family", "")),
            "geography_level": level,
            "area_type": "local_authority" if level == "la" else level,
            "area_code": str(metadata.get("geography_id", "")),
            "metric": str(
                metadata.get("metric", metadata.get("contract_target_id", target.name))
            ),
        }
        if not row["family"]:
            raise ValueError(f"Target {target.row_name!r} lacks a diagnostic family.")
        diagnostics.append(row)
        if materialization == "uk_local_surface":
            local_rows.append(row)
        else:
            national_errors[target.row_name] = float(solved_rows[index].relative_error)
    reference_registry = supporting_evidence.get("reference_registry")
    if not isinstance(reference_registry, TargetRegistry):
        raise ValueError(
            "Full-build gates require the complete approved national reference registry."
        )
    selected = {
        (str(row["name"]), row["period"]) for row in selection_receipt["included"]
    }
    national_reference = {
        spec.to_target().row_name
        for spec in reference_registry.specs
        if (spec.name, spec.period) in selected
    }
    reference = load_efrs_parity_reference()
    manifest = uk_full_gate_manifest(selection_receipt)
    admin_totals, admin_receipt = uk_aggregate_admin_totals(frame, manifest)
    artifacts = dict(supporting_evidence)
    artifacts.update(
        {
            "stage_evidence": dict(stage_evidence),
            "build_stage_names": tuple(stage_evidence),
            "national_calibration": {
                "activated_reference_count": len(selection_receipt["included"]),
                "resolved_reference_count": len(problem.targets),
                "matrix_target_count": len(problem.names),
            },
            "parity_evidence": SimpleNamespace(
                candidate_columns={
                    f"{entity}.{column}"
                    for entity in frame.entities
                    for column in frame.table(entity).columns
                },
                reference_columns={
                    f"{entity}.{name}"
                    for name, entity in reference.input_entities.items()
                },
                candidate_targets=set(national_errors),
                reference_targets=national_reference,
                target_relative_errors=national_errors,
            ),
            "local_target_diagnostics": local_rows,
            "target_diagnostics": diagnostics,
            "aggregate_admin": admin_totals,
            "aggregate_admin_measurement": admin_receipt,
            "full_gate_scope": uk_full_gate_scope_receipt(selection_receipt),
        }
    )
    if "rules_engine" not in artifacts and "coverage_engine" in artifacts:
        artifacts["rules_engine"] = artifacts["coverage_engine"]
    if "rules_engine" in artifacts:
        for name, value in uk_spine_gate_artifacts(artifacts["rules_engine"]).items():
            artifacts.setdefault(name, value)
    return EvidenceContext(frame=frame, artifacts=artifacts)


def classify_full_gate_outcomes(
    report,
    *,
    sample_fraction: float,
    release_candidate: bool,
) -> dict[str, object]:
    """Preserve declaration enforcement and the existing local export exception.

    The combined driver exports a diagnostic candidate after failures of its
    five local fit/support/weight checks, but stops on a non-passing geography
    ladder. Below f100 those five checks are recorded without enforcement.
    All newly included national/source checks keep the shared battery's own
    BLOCKS_ARTIFACT policy, including its declared missing-evidence rules.
    """
    from microcosm.build.gate_battery import GatePhaseReport, GateStatus
    from microcosm.build.uk_runtime.calibration_run import UK_LOCAL_GATE_SCOPE

    if not isinstance(report, GatePhaseReport):
        raise TypeError("Classify a validated GatePhaseReport, not unbound JSON.")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1].")
    geography = "uk_local_geography_ladder_post_calibration"
    local_export_exception = set(UK_LOCAL_GATE_SCOPE) - {geography}
    blocking = {
        outcome.entry.id
        for outcome in report.blocking_outcomes(release_candidate=release_candidate)
    }
    stop, exported_failures, unenforced, diagnostic = [], [], [], []
    for outcome in report.outcomes:
        gate_id = outcome.entry.id
        if gate_id == geography and outcome.status is not GateStatus.PASSED:
            stop.append(gate_id)
            continue
        if outcome.status not in {GateStatus.FAILED, GateStatus.EVIDENCE_ABSENT}:
            continue
        if outcome.entry.criticality == "diagnostic":
            diagnostic.append(gate_id)
        elif gate_id in local_export_exception:
            if gate_id in blocking and sample_fraction == 1.0:
                exported_failures.append(gate_id)
            else:
                unenforced.append(gate_id)
        elif gate_id in blocking:
            stop.append(gate_id)
        else:
            unenforced.append(gate_id)
    return {
        "schema": "microcosm.build.uk.full-gate-enforcement.v1",
        "sample_fraction": sample_fraction,
        "release_candidate": release_candidate,
        "structural_failures": stop,
        "enforced_blocking": stop + exported_failures,
        "exportable_blocking": exported_failures,
        "unenforced_release_failures": unenforced,
        "diagnostic_failures": diagnostic,
        "artifact_permitted": not stop,
        "release_blocking_gates_passed": not (stop or exported_failures or unenforced),
    }
