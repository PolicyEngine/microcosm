"""Adaptive bands accumulate pressure only during observed tolerance violations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.calibrate import (
    Target,
    TargetSet,
    calibrate,
    calibrate_l0_refit,
    refit_l0_selection,
)
from microcosm.calibrate import solve as solve_module
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _problem(n_minor: int = 2, n: int = 24) -> tuple[Frame, TargetSet, np.ndarray]:
    """One critical total competes with individually small, higher totals."""
    x = np.linspace(1.0, 3.0, n)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame({"household_id": range(n), "x": x}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.full(n, 100.0), kind=WeightKind.DESIGN)},
    )
    total = float(100.0 * x.sum())
    targets = TargetSet(
        (
            Target(name="critical", entity="household", value=total, measure="x"),
            *(
                Target(
                    name=f"minor_{i}",
                    entity="household",
                    value=1.2 * total,
                    measure="x",
                )
                for i in range(n_minor)
            ),
        )
    )
    tolerances = np.full(n_minor + 1, np.nan)
    tolerances[0] = 0.05
    return frame, targets, tolerances


def _legacy_term(self, estimate, targets):
    """The pre-adaptive band's original arithmetic, including operation order."""
    errors = torch.abs(estimate[self.index] - targets[self.index]) / self.scales
    hinge = torch.clamp(errors - self.tolerances, min=0.0)
    return self.penalty * hinge.sum()


_FIXED_PATHS = [
    pytest.param({}, id="adam-free"),
    pytest.param({"mass": "conserve"}, id="adam-conserved"),
    pytest.param({"max_weight_ratio": 1.1}, id="adam-ratio-cap"),
    pytest.param(
        {"mass": "conserve", "mass_parametrization": "softmax"},
        id="adam-softmax",
    ),
    pytest.param(
        {
            "mass": "conserve",
            "mass_parametrization": "softmax",
            "max_weight_ratio": 1.1,
        },
        id="adam-softmax-ratio-cap",
    ),
    pytest.param({"l2_lambda": 0.01}, id="adam-l2-closing"),
    pytest.param(
        {"l2_lambda": 0.01, "retain_best_objective": True},
        id="adam-l2-best-objective",
    ),
    pytest.param(
        {"l2_lambda": 0.01, "l2_basis": "chi_square"}, id="adam-l2-chi-square"
    ),
    pytest.param({"l0_lambda": 1e-4}, id="adam-l0"),
    pytest.param({"target_records": 12, "budget_iters": 2}, id="adam-budget-search"),
    pytest.param({"method": "prox", "l1_lambda": 0.05}, id="prox-free"),
    pytest.param(
        {"method": "prox", "l1_lambda": 0.05, "mass": "conserve"},
        id="prox-conserved",
    ),
]


def _assert_same_result(actual, expected) -> None:
    # Comparing bytes, rather than a tolerance, also catches signed-zero changes.
    assert actual.weights.tobytes() == expected.weights.tobytes()
    assert actual.loss_trajectory.tobytes() == expected.loss_trajectory.tobytes()
    assert actual.final_loss == expected.final_loss
    assert actual.options == expected.options


@pytest.mark.parametrize("path", _FIXED_PATHS)
def test_step_zero_preserves_the_legacy_band_on_every_solve_path(
    monkeypatch, path
) -> None:
    frame, targets, tolerances = _problem()
    kwargs = dict(
        epochs=30,
        learning_rate=0.02,
        seed=7,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        **path,
    )
    with monkeypatch.context() as legacy:
        legacy.setattr(solve_module._TargetBand, "term", _legacy_term)
        legacy.setattr(solve_module._TargetBand, "update", lambda *args: None)
        baseline = calibrate(frame, targets, **kwargs)
    default = calibrate(frame, targets, **kwargs)
    explicit_zero = calibrate(frame, targets, target_band_adaptive_step=0.0, **kwargs)
    _assert_same_result(default, baseline)
    _assert_same_result(explicit_zero, baseline)


def test_step_zero_preserves_both_l0_refit_phases(monkeypatch) -> None:
    frame, targets, tolerances = _problem()
    kwargs = dict(
        epochs=30,
        l0_lambda=1e-4,
        seed=7,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
    )
    with monkeypatch.context() as legacy:
        legacy.setattr(solve_module._TargetBand, "term", _legacy_term)
        legacy.setattr(solve_module._TargetBand, "update", lambda *args: None)
        baseline = calibrate_l0_refit(frame, targets, **kwargs)
    actual = calibrate_l0_refit(frame, targets, target_band_adaptive_step=0.0, **kwargs)
    _assert_same_result(actual.selection, baseline.selection)
    _assert_same_result(actual.refit, baseline.refit)
    np.testing.assert_array_equal(actual.selected_mask, baseline.selected_mask)


@settings(max_examples=60, deadline=None)
@given(
    cap=st.floats(min_value=0.01, max_value=10.0),
    step=st.floats(min_value=1e-4, max_value=2.0),
    estimates=st.lists(
        st.tuples(
            st.floats(min_value=-2.0, max_value=2.0),
            st.floats(min_value=-2.0, max_value=2.0),
        ),
        min_size=1,
        max_size=30,
    ),
)
def test_property_multipliers_rise_only_for_violations_and_respect_the_cap(
    cap, step, estimates
) -> None:
    # Include a zero target (absolute-error scale) and a negative target.
    targets = torch.tensor([0.0, -1.0])
    band = solve_module._TargetBand.build(
        np.array([0.2, 0.1]), targets.numpy(), cap, adaptive_step=step
    )
    assert torch.equal(band.multipliers, torch.zeros_like(band.multipliers))
    first = [None, None]
    last = [None, None]
    for epoch, values in enumerate(estimates):
        estimate = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        hinge = band.hinge(estimate, targets).detach().to(torch.float64)
        previous = band.multipliers.clone()
        band.update(estimate, targets, epoch)
        current = band.multipliers
        expected = torch.clamp(previous + step * hinge, max=cap)
        torch.testing.assert_close(current, expected, rtol=1e-14, atol=1e-15)
        assert bool((current >= previous).all())
        assert bool((current <= cap).all())
        assert torch.equal(current[hinge == 0.0], previous[hinge == 0.0])
        unsaturated_violation = (hinge > 0.0) & (previous < cap)
        assert bool(
            (current[unsaturated_violation] > previous[unsaturated_violation]).all()
        )
        assert current.requires_grad is False
        assert current.grad_fn is None
        for row in range(2):
            if hinge[row] > 0.0:
                first[row] = epoch if first[row] is None else first[row]
                last[row] = epoch
    assert band.first_violation_epoch.tolist() == [
        -1 if epoch is None else epoch for epoch in first
    ]
    assert band.last_violation_epoch.tolist() == [
        -1 if epoch is None else epoch for epoch in last
    ]


@settings(max_examples=50, deadline=None)
@given(
    tolerance=st.floats(min_value=0.1, max_value=1.0),
    relative_error=st.floats(min_value=-0.5, max_value=0.5),
    step=st.floats(min_value=1e-4, max_value=10.0),
)
def test_property_an_always_satisfied_row_has_zero_multiplier_and_gradient(
    tolerance, relative_error, step
) -> None:
    targets = torch.tensor([10.0, 100.0])
    band = solve_module._TargetBand.build(
        np.array([tolerance, np.nan]), targets.numpy(), 1.0, adaptive_step=step
    )
    # Strictly inside the tolerance, including after float32 rounding. The
    # unbanded row may be arbitrarily far away without entering the gradient.
    estimate = torch.tensor(
        [10.0 * (1.0 + tolerance * relative_error), -1e6], requires_grad=True
    )
    for epoch in range(4):
        band.update(estimate, targets, epoch)
    term = band.term(estimate, targets)
    term.backward()
    assert term.item() == 0.0
    assert band.multipliers.tolist() == [0.0]
    assert band.first_violation_epoch.tolist() == [-1]
    assert band.last_violation_epoch.tolist() == [-1]
    assert estimate.grad.tolist() == [0.0, 0.0]


def test_a_cleared_row_keeps_its_multiplier_and_stops_rising() -> None:
    targets = torch.tensor([10.0])
    band = solve_module._TargetBand.build(
        np.array([0.05]), targets.numpy(), 1.0, adaptive_step=0.1
    )
    band.update(torch.tensor([12.0]), targets, 0)
    previous = band.multipliers.clone()
    assert previous.item() > 0.0
    for epoch in range(1, 5):
        band.update(torch.tensor([10.0]), targets, epoch)
    assert torch.equal(band.multipliers, previous)
    assert band.first_violation_epoch.tolist() == [0]
    assert band.last_violation_epoch.tolist() == [0]


@pytest.mark.parametrize("method", ["adam", "prox"])
def test_diagnostics_record_the_observed_violation_epochs(monkeypatch, method) -> None:
    frame, targets, tolerances = _problem()
    tolerances[2] = 10.0  # the second banded row never violates
    observations = []
    original_update = solve_module._TargetBand.update

    def observe_update(self, estimate, target_vector, epoch):
        hinge = self.hinge(estimate, target_vector).detach().clone()
        original_update(self, estimate, target_vector, epoch)
        observations.append((epoch, hinge, self.multipliers.clone()))

    monkeypatch.setattr(solve_module._TargetBand, "update", observe_update)
    result = calibrate(
        frame,
        targets,
        method=method,
        epochs=80,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
    )
    assert [epoch for epoch, _, _ in observations] == list(range(80))
    for band_index, row in enumerate(result.options["target_band"]["targets"]):
        violating_epochs = [
            epoch for epoch, hinge, _ in observations if hinge[band_index] > 0.0
        ]
        assert row["first_violation_epoch"] == (
            min(violating_epochs) if violating_epochs else None
        )
        assert row["last_violation_epoch"] == (
            max(violating_epochs) if violating_epochs else None
        )
        assert row["final_multiplier"] == observations[-1][2][band_index].item()


@pytest.mark.parametrize("method", ["adam", "prox"])
def test_the_first_update_uses_the_estimate_its_step_came_from(
    monkeypatch, method
) -> None:
    frame, targets, tolerances = _problem()
    # Start with the critical row outside the band. Its first gradient uses
    # multiplier zero; the multiplier increases only after that optimizer step.
    warm_start = np.full(24, 120.0)
    kwargs = dict(
        epochs=1,
        method=method,
        seed=7,
        warm_start_weights=warm_start,
    )
    if method == "adam":
        kwargs["l2_lambda"] = 1e-6  # select the closing state on the plain solve
    plain = calibrate(frame, targets, **kwargs)
    hinges = []
    original_update = solve_module._TargetBand.update

    def observe_update(self, estimate, target_vector, epoch):
        hinges.append(self.hinge(estimate, target_vector).item())
        original_update(self, estimate, target_vector, epoch)

    monkeypatch.setattr(solve_module._TargetBand, "update", observe_update)
    adaptive = calibrate(
        frame,
        targets,
        **kwargs,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
    )
    assert adaptive.weights.tobytes() == plain.weights.tobytes()
    band = adaptive.options["target_band"]
    [row] = band["targets"]
    assert band["adaptive_step"] == 0.1
    assert len(hinges) == 1
    assert hinges[0] > 0.0
    assert row["final_multiplier"] == pytest.approx(0.1 * hinges[0], abs=1e-7)
    assert row["first_violation_epoch"] == 0
    assert row["last_violation_epoch"] == 0


@pytest.mark.parametrize("method", ["adam", "prox"])
def test_a_never_binding_adaptive_band_leaves_the_closing_solve_unchanged(
    method,
) -> None:
    frame, targets, tolerances = _problem()
    tolerances[0] = 10.0
    # L2 makes ordinary Adam retain its closing state too.
    kwargs = dict(epochs=60, method=method, seed=7)
    if method == "adam":
        kwargs["l2_lambda"] = 0.001
    plain = calibrate(frame, targets, **kwargs)
    wide = calibrate(
        frame,
        targets,
        **kwargs,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
    )
    assert plain.weights.tobytes() == wide.weights.tobytes()
    assert plain.loss_trajectory.tobytes() == wide.loss_trajectory.tobytes()
    [row] = wide.options["target_band"]["targets"]
    assert row["final_multiplier"] == 0.0
    assert row["first_violation_epoch"] is None
    assert row["last_violation_epoch"] is None


@pytest.mark.parametrize("method", ["adam", "prox"])
def test_adaptive_solve_is_identical_for_a_fixed_seed(method) -> None:
    frame, targets, tolerances = _problem()
    kwargs = dict(
        method=method,
        epochs=100,
        seed=13,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
    )
    _assert_same_result(
        calibrate(frame, targets, **kwargs), calibrate(frame, targets, **kwargs)
    )


@pytest.mark.parametrize("method", ["adam", "prox"])
def test_adaptive_bands_work_with_a_sparse_constraint_matrix(
    monkeypatch, method
) -> None:
    frame, targets, tolerances = _problem()
    monkeypatch.setattr(solve_module, "_SPARSE_MIN_CELLS", 0)
    monkeypatch.setattr(solve_module, "_SPARSE_DENSITY_CUTOFF", 1.0)
    layouts = []
    original_apply = solve_module._apply_constraint

    def observe_apply(matrix, weights):
        layouts.append(matrix.layout)
        return original_apply(matrix, weights)

    monkeypatch.setattr(solve_module, "_apply_constraint", observe_apply)
    result = calibrate(
        frame,
        targets,
        method=method,
        epochs=80,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
    )
    assert layouts
    assert all(layout != torch.strided for layout in layouts)
    assert np.isfinite(result.weights).all()
    [row] = result.options["target_band"]["targets"]
    assert 0.0 < row["final_multiplier"] <= 1.0


@pytest.mark.parametrize("mass_parametrization", ["projection", "softmax"])
def test_adaptive_bands_preserve_conserved_mass_and_the_ratio_cap(
    mass_parametrization,
) -> None:
    frame, targets, tolerances = _problem()
    result = calibrate(
        frame,
        targets,
        epochs=80,
        mass="conserve",
        mass_parametrization=mass_parametrization,
        max_weight_ratio=1.1,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
    )
    np.testing.assert_allclose(result.weights.sum(), 24 * 100.0, rtol=1e-12)
    assert bool((result.weights <= 1.1 * result.initial_weights).all())
    assert result.options["iterate_selection"] == "closing_state"


@pytest.mark.parametrize("l2_lambda", [0.0, 0.01])
def test_adaptive_objectives_always_select_the_closing_state(l2_lambda) -> None:
    frame, targets, tolerances = _problem()
    kwargs = dict(
        epochs=80,
        seed=7,
        l2_lambda=l2_lambda,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
    )
    closing = calibrate(frame, targets, **kwargs)
    requested_best = calibrate(frame, targets, retain_best_objective=True, **kwargs)
    assert requested_best.weights.tobytes() == closing.weights.tobytes()
    assert requested_best.loss_trajectory.tobytes() == closing.loss_trajectory.tobytes()
    assert requested_best.options["iterate_selection"] == "closing_state"


def test_adaptive_solve_clears_a_critical_row_without_worsening_fixed_band_fit() -> (
    None
):
    # A single record makes the tradeoff transparent: 100 minor targets prefer
    # weight 120, while the critical target requires 100 +/- 5. Prox returns
    # the closing state in both modes, so the same iteration budget compares
    # the same selection rule. The normalized step is 0.02 in ratio space.
    frame, targets, tolerances = _problem(n_minor=100, n=1)
    kwargs = dict(method="prox", epochs=600, learning_rate=0.02, seed=7)
    plain = calibrate(frame, targets, **kwargs)
    fixed = calibrate(
        frame,
        targets,
        **kwargs,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
    )
    adaptive = calibrate(
        frame,
        targets,
        **kwargs,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
    )
    plain_error = abs(plain.problem.estimates(plain.weights)[0] / 100.0 - 1.0)
    assert plain_error > 0.15
    [row] = adaptive.options["target_band"]["targets"]
    assert abs(row["final_relative_error"]) <= row["tolerance"]
    assert row["final_violation"] == 0.0
    assert row["final_multiplier"] > 0.0
    assert adaptive.final_loss <= fixed.final_loss


@pytest.mark.parametrize("path", [{"l0_lambda": 1e-4}, {"target_records": 12}])
def test_adaptive_bands_refuse_stochastic_l0_paths(path) -> None:
    frame, targets, tolerances = _problem()
    with pytest.raises(ValueError, match="adaptive.*L0|L0.*adaptive"):
        calibrate(
            frame,
            targets,
            epochs=2,
            target_band_tolerances=tolerances,
            target_band_penalty=1.0,
            target_band_adaptive_step=0.1,
            **path,
        )


def test_adaptive_bands_refuse_the_combined_l0_selection_wrapper() -> None:
    frame, targets, tolerances = _problem()
    with pytest.raises(ValueError, match="adaptive.*L0|L0.*adaptive"):
        calibrate_l0_refit(
            frame,
            targets,
            epochs=2,
            l0_lambda=1e-4,
            target_band_tolerances=tolerances,
            target_band_penalty=1.0,
            target_band_adaptive_step=0.1,
        )


def test_frozen_l0_support_can_be_refit_with_an_adaptive_band() -> None:
    frame, targets, tolerances = _problem()
    selection = calibrate(frame, targets, l0_lambda=1e-4, epochs=10, seed=7)
    result = refit_l0_selection(
        frame,
        targets,
        selection,
        epochs=100,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
    )
    assert result.refit.options["target_band"]["adaptive_step"] == 0.1
    assert result.refit.options["iterate_selection"] == "closing_state"


@pytest.mark.parametrize("step", [-1.0, float("nan"), float("inf")])
def test_malformed_adaptive_steps_are_refused(step) -> None:
    frame, targets, _ = _problem()
    with pytest.raises(
        ValueError, match="target_band_adaptive_step.*finite.*non-negative"
    ):
        calibrate(frame, targets, epochs=2, target_band_adaptive_step=step)


def test_adaptive_mode_requires_a_positive_cap() -> None:
    frame, targets, tolerances = _problem()
    with pytest.raises(ValueError, match="target_band_adaptive_step > 0 requires"):
        calibrate(
            frame,
            targets,
            epochs=2,
            target_band_tolerances=tolerances,
            target_band_penalty=0.0,
            target_band_adaptive_step=0.1,
        )


@pytest.mark.parametrize("method", ["adam", "prox"])
def test_adaptive_mode_adds_no_forward_pass(monkeypatch, method) -> None:
    frame, targets, tolerances = _problem()
    original = solve_module._apply_constraint
    calls = [0]

    def counted(matrix, weights):
        calls[0] += 1
        return original(matrix, weights)

    monkeypatch.setattr(solve_module, "_apply_constraint", counted)
    kwargs = dict(
        epochs=30,
        method=method,
        seed=7,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
    )
    calibrate(frame, targets, **kwargs)
    fixed_calls = calls[0]
    calls[0] = 0
    calibrate(frame, targets, target_band_adaptive_step=0.1, **kwargs)
    # Fixed Adam may add a closing comparison that adaptive solves skip.
    assert calls[0] <= fixed_calls


@settings(max_examples=50, deadline=None)
@given(
    warmup=st.integers(min_value=0, max_value=8),
    step=st.floats(min_value=1e-4, max_value=2.0),
    epochs=st.integers(min_value=1, max_value=12),
)
def test_property_warmup_holds_multipliers_and_keeps_recording_violations(
    warmup, step, epochs
) -> None:
    targets = torch.tensor([10.0])
    band = solve_module._TargetBand.build(
        np.array([0.05]),
        targets.numpy(),
        1000.0,
        adaptive_step=step,
        adaptive_warmup_epochs=warmup,
    )
    estimate = torch.tensor([12.0])
    hinge = band.hinge(estimate, targets).to(torch.float64)
    for epoch in range(epochs):
        band.update(estimate, targets, epoch)
        expected = step * hinge * max(epoch + 1 - warmup, 0)
        torch.testing.assert_close(band.multipliers, expected, rtol=1e-12, atol=0.0)
    assert band.first_violation_epoch.tolist() == [0]
    assert band.last_violation_epoch.tolist() == [epochs - 1]


@pytest.mark.parametrize("method", ["adam", "prox"])
def test_a_warmup_covering_every_gradient_matches_the_plain_solve(method) -> None:
    frame, targets, tolerances = _problem()
    epochs = 40
    # L2 makes ordinary Adam retain its closing state too.
    kwargs = dict(epochs=epochs, method=method, seed=7)
    if method == "adam":
        kwargs["l2_lambda"] = 0.001
    plain = calibrate(frame, targets, **kwargs)
    held = calibrate(
        frame,
        targets,
        **kwargs,
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
        target_band_adaptive_warmup_epochs=epochs - 1,
    )
    # The update after the last step comes too late to touch any gradient.
    assert held.weights.tobytes() == plain.weights.tobytes()
    assert held.options["target_band"]["adaptive_warmup_epochs"] == epochs - 1


@pytest.mark.parametrize(
    ("step", "warmup", "message"),
    [
        (0.1, -1, "non-negative integer"),
        (0.1, 1.5, "non-negative integer"),
        (0.1, True, "non-negative integer"),
        (0.0, 5, "requires target_band_adaptive_step > 0"),
        (0.1, 10, "must be below epochs"),
        (0.1, 11, "must be below epochs"),
    ],
)
def test_a_malformed_warmup_is_refused(step, warmup, message) -> None:
    frame, targets, tolerances = _problem()
    with pytest.raises(ValueError, match=message):
        calibrate(
            frame,
            targets,
            epochs=10,
            target_band_tolerances=tolerances,
            target_band_penalty=1.0,
            target_band_adaptive_step=step,
            target_band_adaptive_warmup_epochs=warmup,
        )


def test_warmup_lets_a_row_the_fit_clears_alone_collect_no_pressure() -> None:
    frame, targets, tolerances = _problem(n_minor=0)
    # The row starts outside its band with nothing pulling against the fit.
    kwargs = dict(
        epochs=400,
        seed=7,
        warm_start_weights=np.full(24, 120.0),
        target_band_tolerances=tolerances,
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
    )
    eager = calibrate(frame, targets, **kwargs)
    patient = calibrate(
        frame, targets, target_band_adaptive_warmup_epochs=200, **kwargs
    )
    [eager_row] = eager.options["target_band"]["targets"]
    [patient_row] = patient.options["target_band"]["targets"]
    assert eager_row["first_violation_epoch"] == 0
    assert patient_row["first_violation_epoch"] == 0
    assert eager_row["final_multiplier"] > 0.0
    assert patient_row["final_multiplier"] == 0.0
    assert patient_row["final_violation"] == 0.0
