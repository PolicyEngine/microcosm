"""Scope and population identity checks for the complete UK gate context."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import full_gates as runtime
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.artifacts import OrderedProblem, OrderedSolution
from microcosm.calibrate.matrix import CalibrationProblem
from microcosm.calibrate.solve import CalibrationResult
from microcosm.frame import WeightKind, Weights


def _selection(levels=None):
    rows = [{"name": "national", "period": 2024, "geography_level": "country"}]
    if levels is None:
        rows.append(
            {"name": "local", "period": 2024, "geography_level": "constituency"}
        )
    return {
        "schema": "microcosm.calibrate.target-selection.v1",
        "selector": {"geography_levels": levels, "explicit": levels is not None},
        "included": rows,
        "excluded": [],
    }


def test_default_full_gate_scope_owns_every_country_gate():
    assert {gate.id for gate in runtime.uk_full_gate_manifest().gates} == {
        gate.id for gate in load_country_spec("uk").gates.gates
    }
    assert runtime.uk_full_gate_scope_receipt()["scope_exclusions"] == {}


def test_country_filter_retains_integrity_and_registry_checks_without_local_fit_claims():
    selected = _selection(["country"])
    receipt = runtime.uk_full_gate_scope_receipt(selected)
    gates = {gate.id for gate in runtime.uk_full_gate_manifest(selected).gates}
    assert receipt["local_fit_claim"] is False
    assert set(receipt["scope_exclusions"]) == {
        "uk_local_target_fit",
        "uk_local_per_family_fit",
        "uk_local_area_support",
    }
    assert {
        "uk_local_geography_ladder_post_calibration",
        "uk_weight_ess",
        "uk_release_family_build_stages",
        "uk_ledger_compile_parity_local_incumbent_2025",
        "uk_target_surface_local_default_2025",
    } <= gates


def test_unfiltered_build_cannot_silently_become_national_only():
    selection = _selection(["country"])
    selection["selector"]["geography_levels"] = None
    with pytest.raises(ValueError, match="unfiltered UK full build"):
        runtime.uk_full_gate_manifest(selection)


@pytest.fixture
def evidence(monkeypatch):
    specs = [
        TargetSpec(
            name="national",
            entity="household",
            value=0.5,
            measure="income",
            period=2024,
            source="test",
            family="income",
        ),
        TargetSpec(
            name="local",
            entity="household",
            value=2.0,
            measure="income",
            period=2024,
            source="test",
            family="population",
        ),
    ]
    targets = tuple(spec.to_target() for spec in specs)
    weights = Weights(np.ones(2), WeightKind.CALIBRATED)
    problem = CalibrationProblem(
        sparse.csr_array([[0.1, 0.2], [1.0, 1.0]]),
        np.array([0.5, 2.0]),
        tuple(target.row_name for target in targets),
        Weights(np.ones(2), WeightKind.IMPORTANCE),
        "household",
        targets,
    )
    metadata = (
        {
            "geography_level": "country",
            "geography_id": "UK",
            "family": "income",
            "materialization": "uk_national_measure",
        },
        {
            "geography_level": "constituency",
            "geography_id": "E14000001",
            "family": "population",
            "materialization": "uk_local_surface",
        },
    )
    ordered = OrderedProblem(
        problem, (1, 2), metadata, {"target_selection": _selection()}, "a" * 64
    )
    solution = OrderedSolution(np.ones(2), (1, 2), "a" * 64, {}, "b" * 64)
    table = pd.DataFrame({"household_id": [1, 2], "income": [1.0, 1.0]})
    frame = SimpleNamespace(
        entities=("household",),
        table=lambda entity: table,
        schema=SimpleNamespace(entity_id_column=lambda entity: "household_id"),
        weights_for=lambda entity: weights,
    )
    monkeypatch.setattr(
        runtime,
        "load_efrs_parity_reference",
        lambda: SimpleNamespace(input_entities={"income": "household"}),
    )
    monkeypatch.setattr(
        runtime,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: ({"admin": 2.0}, [{"measured": 2.0}]),
    )
    return (
        frame,
        ordered,
        solution,
        {
            "reference_registry": TargetRegistry([specs[0]], country="uk"),
            "coverage_engine": object(),
        },
    )


def _context(evidence, **overrides):
    frame, ordered, solution, supporting = evidence
    return runtime.build_full_gate_context(
        frame,
        ordered_problem=ordered,
        solution=overrides.get("solution", solution),
        selection_receipt=_selection(),
        stage_evidence={"frs_spine": {}},
        supporting_evidence=supporting,
    )


def test_gate_diagnostics_use_bound_matrix_and_row_identity(evidence):
    context = _context(evidence)
    assert context.artifacts["parity_evidence"].target_relative_errors == pytest.approx(
        {"national@2024": -0.4}
    )

    assert context.artifacts["parity_evidence"].reference_targets == {"national@2024"}
    assert context.artifacts["local_target_diagnostics"][0]["area_code"] == "E14000001"
    assert context.artifacts["local_target_diagnostics"][0]["relative_error"] == 0.0
    assert context.artifacts["national_calibration"]["matrix_target_count"] == 2
    assert context.artifacts["rules_engine"] is context.artifacts["coverage_engine"]


def _completed_result(evidence):
    frame, ordered, solution, _ = evidence
    problem = ordered.problem
    return CalibrationResult(
        frame=frame,
        weight_entity="household",
        weights=solution.weights,
        initial_weights=problem.initial_weights.values,
        diagnostics=runtime._build_diagnostics(
            problem, frame, problem.initial_weights.values, solution.weights
        ),
        loss_trajectory=np.array([0.2]),
        skipped=(),
        problem=problem,
        l0_lambda=0.0,
        n_nonzero=2,
        closing_loss=0.2,
        target_loss_weights=np.ones(2),
        target_loss_scales=np.ones(2),
        target_loss_cap=10.0,
    )


def test_gate_context_reuses_authenticated_final_diagnostics(evidence, monkeypatch):
    evidence[3]["calibration_result"] = _completed_result(evidence)
    monkeypatch.setattr(
        runtime,
        "_build_diagnostics",
        lambda *args: pytest.fail("diagnostics computed twice"),
    )
    assert _context(evidence).artifacts["parity_evidence"].target_relative_errors[
        "national@2024"
    ] == pytest.approx(-0.4)


@pytest.mark.parametrize("mutation", ["weights", "problem", "diagnostic_axis"])
def test_gate_context_rejects_foreign_completed_result(evidence, mutation):
    result = _completed_result(evidence)
    if mutation == "weights":
        result = replace(result, weights=np.array([1.5, 0.5]))
    elif mutation == "problem":
        result = replace(
            result, problem=replace(result.problem, matrix=2 * result.problem.matrix)
        )
    else:
        result = replace(result, diagnostics=result.diagnostics[::-1])
    evidence[3]["calibration_result"] = result
    with pytest.raises(ValueError, match="calibration_result"):
        _context(evidence)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"problem_sha256": "c" * 64}, "different ordered problem"),
        ({"entity_ids": (2, 1)}, "different household axes"),
        ({"weights": np.array([2.0, 1.0])}, "bound solution weights"),
    ],
)
def test_gate_context_rejects_wrong_solution_identity(evidence, change, match):
    with pytest.raises(ValueError, match=match):
        _context(evidence, solution=replace(evidence[2], **change))


def test_gate_context_rejects_selection_binding_drift(evidence):
    frame, ordered, solution, supporting = evidence
    ordered = replace(ordered, bindings={"target_selection_sha256": "c" * 64})
    with pytest.raises(ValueError, match="target-selection digest"):
        _context((frame, ordered, solution, supporting))


def _phase_failure(gate_id, *, absent=False):
    from microcosm.build.gate_battery import GateOutcome, GatePhaseReport, GateStatus
    from microcosm.build.gates import GateResult

    entry = next(g for g in load_country_spec("uk").gates.gates if g.id == gate_id)
    return GatePhaseReport(
        entry.phase,
        (
            GateOutcome(
                entry=entry,
                status=GateStatus.EVIDENCE_ABSENT if absent else GateStatus.FAILED,
                result=None
                if absent
                else GateResult(
                    name=entry.gate, passed=False, failures=("test failure",)
                ),
                reason="missing evidence" if absent else None,
            ),
        ),
    )


def test_existing_local_statistical_failure_can_export_but_blocks_full_release():
    report = _phase_failure("uk_local_target_fit")
    full = runtime.classify_full_gate_outcomes(
        report, sample_fraction=1.0, release_candidate=True
    )
    assert full["artifact_permitted"] is True
    assert full["exportable_blocking"] == ["uk_local_target_fit"]
    assert full["release_blocking_gates_passed"] is False
    rung = runtime.classify_full_gate_outcomes(
        report, sample_fraction=0.1, release_candidate=False
    )
    assert rung["enforced_blocking"] == []
    assert rung["unenforced_release_failures"] == ["uk_local_target_fit"]


@pytest.mark.parametrize(
    "gate_id",
    [
        "uk_local_geography_ladder_post_calibration",
        "uk_nonnegative_columns",
        "uk_input_mass_parity",
        "uk_target_fit",
        "uk_calibration_reference_coverage",
    ],
)
def test_geography_and_migrated_national_gates_keep_artifact_enforcement(gate_id):
    for fraction in (0.1, 1.0):
        result = runtime.classify_full_gate_outcomes(
            _phase_failure(gate_id), sample_fraction=fraction, release_candidate=False
        )
        assert result["artifact_permitted"] is False
        assert result["structural_failures"] == [gate_id]


def test_missing_evidence_uses_declared_development_policy():
    ordinary = _phase_failure("uk_input_mass_parity", absent=True)
    assert (
        runtime.classify_full_gate_outcomes(
            ordinary, sample_fraction=1.0, release_candidate=False
        )["artifact_permitted"]
        is True
    )
    assert (
        runtime.classify_full_gate_outcomes(
            ordinary, sample_fraction=1.0, release_candidate=True
        )["artifact_permitted"]
        is False
    )
    strict = _phase_failure("uk_weights_audit", absent=True)
    assert (
        runtime.classify_full_gate_outcomes(
            strict, sample_fraction=1.0, release_candidate=False
        )["artifact_permitted"]
        is False
    )
