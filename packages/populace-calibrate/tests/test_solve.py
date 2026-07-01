"""Behavioral contracts for the calibration solver.

Each test pins a property the charter promises: calibration reduces target loss
and hits feasible targets, produces CALIBRATED weights, conserves declared mass
only when asked, respects the hard weight-ratio bound (and so cannot detonate a
landmine), stacks multi-period targets over one weight vector, and prunes toward
a record budget with L0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.calibrate import (
    L0RefitResult,
    Target,
    TargetSet,
    calibrate,
    calibrate_l0_refit,
    default_target_loss_scales,
    refit_l0_selection,
    relative_error_loss,
)
from populace.frame import EntitySchema, Frame, WeightKind, Weights


def _income_target(truth: float, factor: float) -> Target:
    return Target(
        name="income",
        entity="household",
        value=truth * factor,
        measure="income",
    )


def _population_target(truth: float, factor: float) -> Target:
    return Target(
        name="population",
        entity="household",
        value=truth * factor,
        measure="household_count",
    )


def _effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def _l2_concentration_fixture() -> tuple[Frame, TargetSet, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 160
    income = rng.lognormal(10.5, 1.0, n)
    is_renter = (income < np.quantile(income, 0.35)).astype(float)
    initial_weights = np.full(n, 1000.0)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {
                    "household_id": range(n),
                    "income": income,
                    "is_renter": is_renter,
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=initial_weights, kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                value=float((income * initial_weights).sum()) * 1.3,
                measure="income",
            ),
            Target(
                name="renters",
                entity="household",
                value=float((is_renter * initial_weights).sum()) * 0.7,
                measure="is_renter",
            ),
        )
    )
    return frame, targets, initial_weights


def test_method_prox_l1_selects_sparse_subset() -> None:
    """method='prox' with an L1 penalty drives a strict subset to exact zero."""
    frame, targets, w0 = _l2_concentration_fixture()
    n = w0.size
    dense = calibrate(frame, targets, method="prox", l1_lambda=0.0, epochs=300, seed=0)
    assert dense.n_nonzero == n  # no penalty -> nothing pruned
    assert dense.final_loss < dense.initial_loss  # the prox path actually fits

    sparse = calibrate(frame, targets, method="prox", l1_lambda=0.5, epochs=300, seed=0)
    weights = np.asarray(sparse.weights)
    assert 0 < sparse.n_nonzero < n  # a budget-controllable sparse subset
    assert np.all(weights >= 0.0)  # weights stay non-negative
    assert (weights == 0.0).any()  # and exactly zero, not merely small


def test_l1_lambda_is_budget_monotone() -> None:
    """A larger L1 penalty retains no more records -- the budget knob is monotone."""
    frame, targets, _ = _l2_concentration_fixture()
    counts = [
        calibrate(
            frame, targets, method="prox", l1_lambda=lam, epochs=300, seed=0
        ).n_nonzero
        for lam in (0.3, 0.5, 1.0)
    ]
    assert counts[0] >= counts[1] >= counts[2]


def test_apg_method_alias_normalizes_to_adam() -> None:
    """Existing configs can pass 'apg', but manifests record the real Adam path."""
    frame, targets, _ = _l2_concentration_fixture()
    with pytest.deprecated_call(match="method='apg' is deprecated"):
        result = calibrate(frame, targets, method="apg", epochs=10, seed=0)

    assert result.options["method"] == "adam"


def test_l1_lambda_requires_prox_method() -> None:
    """l1_lambda needs the proximal solver; Adam cannot soft-threshold to zero."""
    frame, targets, _ = _l2_concentration_fixture()
    with pytest.raises(ValueError, match="l1_lambda requires method='prox'"):
        calibrate(frame, targets, method="adam", l1_lambda=0.5, epochs=10, seed=0)


def test_method_prox_rejects_l2_lambda() -> None:
    """The prox path must not record an L2 objective it does not optimize."""
    frame, targets, _ = _l2_concentration_fixture()
    with pytest.raises(ValueError, match="method='prox' does not implement l2_lambda"):
        calibrate(frame, targets, method="prox", l2_lambda=0.001, epochs=10, seed=0)


def test_method_prox_conserve_cap_infeasible_names_l1_remedy() -> None:
    """The prox projection error must not tell users to tune L0 knobs."""
    frame, targets, _ = _l2_concentration_fixture()
    with pytest.raises(
        ValueError,
        match=(
            "L1 proximal pruning.*mass='conserve'.*max_weight_ratio=.*lower l1_lambda"
        ),
    ):
        calibrate(
            frame,
            targets,
            method="prox",
            l1_lambda=0.5,
            epochs=300,
            seed=0,
            mass="conserve",
            max_weight_ratio=1.05,
        )


def test_method_prox_zero_target_all_zero_raises_named_error() -> None:
    """The prox path names all-zero optima before the frame kernel rejects them."""
    initial_weights = np.full(8, 100.0)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(8), "person_household_id": range(8)}
            ),
            "household": pd.DataFrame(
                {"household_id": range(8), "household_count": np.ones(8)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=initial_weights, kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (
            Target(
                name="zero_population",
                entity="household",
                value=0.0,
                measure="household_count",
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="method='prox' returned every calibrated weight as zero",
    ):
        calibrate(
            frame,
            targets,
            method="prox",
            l1_lambda=0.0,
            learning_rate=1.0,
            epochs=1,
            seed=0,
            target_loss_cap=1_000_000.0,
        )


def test_method_prox_l1_uses_mean_ratio_penalty_scale() -> None:
    """The prox shrink matches l1_lambda * mean(weight / initial_weight)."""
    initial_weights = np.array([100.0, 200.0, 300.0, 400.0])
    n = initial_weights.size
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {"household_id": range(n), "household_count": np.ones(n)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=initial_weights, kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                value=float(initial_weights.sum()),
                measure="household_count",
            ),
        )
    )
    learning_rate = 0.2
    l1_lambda = 0.8

    result = calibrate(
        frame,
        targets,
        method="prox",
        l1_lambda=l1_lambda,
        learning_rate=learning_rate,
        epochs=1,
        seed=0,
    )

    expected_ratio = 1.0 - (learning_rate * l1_lambda / n)
    np.testing.assert_allclose(result.weights / initial_weights, expected_ratio)
    assert result.options["l1_penalty"] == "mean_initial_weight_ratio_abs"


def test_method_prox_l1_uses_effective_step_with_nonzero_gradient() -> None:
    """The L1 threshold uses the RMS-normalized smooth step, not raw lr."""
    initial_weights = np.array([100.0, 200.0, 300.0, 400.0])
    n = initial_weights.size
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {"household_id": range(n), "household_count": np.ones(n)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=initial_weights, kind=WeightKind.DESIGN)},
    )
    target = float(initial_weights.sum() * 2.0)
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                value=target,
                measure="household_count",
            ),
        )
    )
    learning_rate = 0.2
    l1_lambda = 0.8

    result = calibrate(
        frame,
        targets,
        method="prox",
        l1_lambda=l1_lambda,
        learning_rate=learning_rate,
        epochs=1,
        seed=0,
    )

    grad = -initial_weights / target
    step_size = learning_rate / np.sqrt(np.mean(grad**2))
    expected_ratio = 1.0 - (step_size * grad) - (step_size * l1_lambda / n)
    np.testing.assert_allclose(result.weights / initial_weights, expected_ratio)


def test_calibration_reduces_loss_and_hits_feasible_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    # Shift both targets by the same factor so a uniform rescale hits both.
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.4),
            _income_target(truths["income"], 1.4),
        )
    )
    result = calibrate(frame, targets, epochs=400, seed=0)
    assert result.final_loss < result.initial_loss * 0.01
    for diag in result.diagnostics:
        assert abs(diag.relative_error) < 0.01  # within 1%


def test_calibrated_weights_are_calibrated_kind(feasible_frame) -> None:
    frame, truths = feasible_frame()
    targets = TargetSet((_population_target(truths["population"], 1.2),))
    result = calibrate(frame, targets, epochs=200, seed=0)
    assert result.frame.resolve_weights("household").kind is WeightKind.CALIBRATED


def test_conserve_mass_holds_total_free_mass_moves(feasible_frame) -> None:
    frame, truths = feasible_frame()
    initial_total = frame.resolve_weights("household").values.sum()
    # A population target above the current total: free mass should grow to it.
    targets = TargetSet((_population_target(truths["population"], 1.5),))

    free = calibrate(frame, targets, epochs=300, seed=0, mass="free")
    free_total = free.frame.resolve_weights("household").values.sum()
    assert free_total > initial_total * 1.3  # moved toward the larger target

    conserved = calibrate(frame, targets, epochs=300, seed=0, mass="conserve")
    conserved_total = conserved.frame.resolve_weights("household").values.sum()
    assert abs(conserved_total - initial_total) / initial_total < 1e-6


def test_target_loss_weights_prioritize_conflicting_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    low = truths["population"] * 0.5
    high = truths["population"] * 1.5
    targets = TargetSet(
        (
            Target(
                name="population_low",
                entity="household",
                value=low,
                measure="household_count",
            ),
            Target(
                name="population_high",
                entity="household",
                value=high,
                measure="household_count",
            ),
        )
    )

    uniform = calibrate(frame, targets, epochs=300, seed=0)
    prioritized = calibrate(
        frame,
        targets,
        epochs=300,
        seed=0,
        target_loss_weights=np.asarray([1.0, 1_000.0]),
    )

    uniform_total = uniform.weights.sum()
    prioritized_total = prioritized.weights.sum()
    assert abs(prioritized_total - high) < abs(uniform_total - high)
    assert prioritized.options["target_loss_weights"]["kind"] == "provided"


def test_target_loss_weights_follow_skipped_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    targets = TargetSet(
        (
            Target(
                name="missing_measure",
                entity="household",
                value=1.0,
                measure="missing_measure",
            ),
            _population_target(truths["population"], 1.0),
        )
    )

    result = calibrate(
        frame,
        targets,
        epochs=50,
        seed=0,
        target_loss_weights=np.asarray([1_000.0, 1.0]),
    )

    assert [skip.target.name for skip in result.skipped] == ["missing_measure"]
    assert result.options["target_loss_weights"]["n"] == 1


def test_target_loss_weights_must_survive_skipped_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    targets = TargetSet(
        (
            Target(
                name="missing_measure",
                entity="household",
                value=1.0,
                measure="missing_measure",
            ),
            _population_target(truths["population"], 1.0),
        )
    )

    with pytest.raises(ValueError, match="positive total weight"):
        calibrate(
            frame,
            targets,
            epochs=50,
            seed=0,
            target_loss_weights=np.asarray([1.0, 0.0]),
        )


def test_target_loss_scales_follow_skipped_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    targets = TargetSet(
        (
            Target(
                name="missing_measure",
                entity="household",
                value=1.0,
                measure="missing_measure",
            ),
            _population_target(truths["population"], 1.0),
        )
    )

    result = calibrate(
        frame,
        targets,
        epochs=50,
        seed=0,
        target_loss_scales=np.asarray([10.0, 20.0]),
    )

    assert [skip.target.name for skip in result.skipped] == ["missing_measure"]
    assert result.options["target_loss_scales"]["kind"] == "provided"
    assert result.options["target_loss_scales"]["n"] == 1
    assert result.options["target_loss_scales"]["min"] == pytest.approx(20.0)
    assert result.options["target_loss_scales"]["max"] == pytest.approx(20.0)


def test_target_loss_scales_must_survive_skipped_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    targets = TargetSet(
        (
            Target(
                name="missing_measure",
                entity="household",
                value=1.0,
                measure="missing_measure",
            ),
            _population_target(truths["population"], 1.0),
        )
    )

    with pytest.raises(ValueError, match="target_loss_scales"):
        calibrate(
            frame,
            targets,
            epochs=50,
            seed=0,
            target_loss_scales=np.asarray([10.0, 0.0]),
        )


def test_max_weight_ratio_is_respected(feasible_frame) -> None:
    frame, truths = feasible_frame()
    w0 = frame.resolve_weights("household").values
    targets = TargetSet((_income_target(truths["income"], 3.0),))
    result = calibrate(frame, targets, epochs=300, seed=0, max_weight_ratio=2.0)
    w = result.frame.resolve_weights("household").values
    assert (w <= 2.0 * w0 + 1e-9).all()


def test_weight_ratio_bound_prevents_a_landmine(landmine_frame) -> None:
    """A capital-gains target reachable only by inflating one rare donor.

    Unbounded calibration detonates the donor's weight to hit the target;
    the ratio bound caps it, so the donor's weighted capital-gains stays sane.
    """
    frame, donor_index, donor_value = landmine_frame()
    w0 = frame.resolve_weights("household").values
    # Target far above the current weighted capital gains (donor weight is ~1).
    far_target = TargetSet(
        (
            Target(
                name="capital_gains",
                entity="household",
                value=donor_value * 5000.0,
                measure="capital_gains",
            ),
        )
    )

    unbounded = calibrate(frame, far_target, epochs=300, seed=0)
    unbounded_w = unbounded.frame.resolve_weights("household").values
    bounded = calibrate(frame, far_target, epochs=300, seed=0, max_weight_ratio=10.0)
    bounded_w = bounded.frame.resolve_weights("household").values

    # Unbounded blows the donor weight far past the bound; bounded cannot.
    assert unbounded_w[donor_index] > 100.0 * w0[donor_index]
    assert bounded_w[donor_index] <= 10.0 * w0[donor_index] + 1e-9


def test_multi_period_targets_share_one_weight_vector(multiperiod_frame) -> None:
    frame, truth_2026, truth_2030 = multiperiod_frame()
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                value=truth_2026 * 1.3,
                measure="income_2026",
                period=2026,
            ),
            Target(
                name="income",
                entity="household",
                value=truth_2030 * 1.3,
                measure="income_2030",
                period=2030,
            ),
        )
    )
    result = calibrate(frame, targets, epochs=400, seed=0)
    # One weight vector, both periods' targets improved.
    assert result.problem.matrix.shape[0] == 2  # two (target, period) rows
    for diag in result.diagnostics:
        assert abs(diag.relative_error) < 0.05


def test_l0_prunes_toward_a_record_budget(feasible_frame) -> None:
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    budget = 120
    result = calibrate(
        frame,
        targets,
        epochs=500,
        seed=0,
        target_records=budget,
        l0_lambda=1e-3,
    )
    w = result.frame.resolve_weights("household").values
    nonzero = int((w > 1e-6).sum())
    # Pruned well below the 400 records, in the neighborhood of the budget.
    assert nonzero < 250
    assert result.l0_lambda > 0.0


def test_target_records_controls_the_budget_not_just_lambda(
    feasible_frame,
) -> None:
    """``target_records`` actually drives the achieved non-zero count (Finding 3).

    Previously ``target_records`` was dead: the *number* never entered the
    optimization (pruning was controlled entirely by ``l0_lambda``), so budget
    10 and budget 350 at the same lambda produced bitwise-identical weights. With
    budget control, two different budgets must produce materially different
    non-zero counts, each tracking its budget.
    """
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    low = calibrate(frame, targets, epochs=250, seed=0, target_records=100)
    high = calibrate(frame, targets, epochs=250, seed=0, target_records=250)

    # The budgets produce materially different non-zero counts...
    assert high.n_nonzero - low.n_nonzero > 50, (low.n_nonzero, high.n_nonzero)
    # ...and the lower budget really is the smaller pool, ordered by budget.
    assert low.n_nonzero < high.n_nonzero
    # Each tracks its budget within a loose tolerance (the search is on a noisy,
    # discrete count, so the band is generous but far tighter than "anything").
    assert abs(low.n_nonzero - 100) <= 40, low.n_nonzero
    assert abs(high.n_nonzero - 250) <= 60, high.n_nonzero
    # A budget-controlled run reports the penalty it settled on.
    assert low.l0_lambda > 0.0 and high.l0_lambda > 0.0
    # The smaller pool took the stronger penalty (more pruning pressure).
    assert low.l0_lambda > high.l0_lambda


def test_l0_lambda_alone_is_a_fixed_penalty_pruning_control(feasible_frame) -> None:
    """Without ``target_records``, ``l0_lambda`` alone prunes monotonically.

    The fixed-penalty path (no budget search) is still a real control: a stronger
    ``l0_lambda`` keeps strictly fewer records, and the reported ``l0_lambda`` is
    exactly the value passed in (no search overrides it).
    """
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    weak = calibrate(frame, targets, epochs=300, seed=0, l0_lambda=3e-4)
    strong = calibrate(frame, targets, epochs=300, seed=0, l0_lambda=3e-3)
    assert strong.n_nonzero < weak.n_nonzero
    # Reported penalty is the value supplied, unchanged.
    assert weak.l0_lambda == 3e-4
    assert strong.l0_lambda == 3e-3


def test_l0_refit_returns_refit_on_selected_support(feasible_frame) -> None:
    frame, truths = feasible_frame(n=220)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )

    result = calibrate_l0_refit(
        frame,
        targets,
        epochs=180,
        refit_epochs=180,
        seed=0,
        target_records=80,
        mass="conserve",
    )

    assert isinstance(result, L0RefitResult)
    assert result.selection.l0_lambda > 0.0
    assert result.refit.l0_lambda == 0.0
    assert result.l0_lambda == result.selection.l0_lambda
    assert result.final_loss == result.refit.final_loss
    assert result.frame is result.refit.frame
    assert result.frame.n("household") == result.selection.n_nonzero
    assert len(result.selected_entity_ids) == result.selection.n_nonzero
    assert result.n_nonzero == result.selection.n_nonzero
    # The production weights are the ordinary no-L0 refit on the L0-selected
    # support, not the raw gated selection weights.
    assert result.final_loss <= result.selection.final_loss + 0.02


def test_refit_l0_selection_reuses_existing_selection(feasible_frame) -> None:
    frame, truths = feasible_frame(n=180)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    selection = calibrate(
        frame,
        targets,
        epochs=160,
        seed=0,
        target_records=70,
        mass="conserve",
    )

    result = refit_l0_selection(
        frame,
        targets,
        selection,
        epochs=160,
        seed=0,
        mass="conserve",
    )

    assert result.selection is selection
    assert result.refit.l0_lambda == 0.0
    assert result.frame.n("household") == selection.n_nonzero
    assert len(result.selected_entity_ids) == selection.n_nonzero


def test_l0_refit_requires_l0_pruning_control(feasible_frame) -> None:
    frame, truths = feasible_frame(n=40)
    targets = TargetSet((_population_target(truths["population"], 1.0),))

    with pytest.raises(ValueError, match="target_records|l0_lambda"):
        calibrate_l0_refit(frame, targets, epochs=10, seed=0)


def test_l2_lambda_zero_matches_default(feasible_frame) -> None:
    frame, truths = feasible_frame(n=80)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.2),
            _income_target(truths["income"], 1.2),
        )
    )

    default = calibrate(frame, targets, epochs=120, seed=0)
    explicit_zero = calibrate(frame, targets, epochs=120, seed=0, l2_lambda=0.0)

    np.testing.assert_array_equal(explicit_zero.weights, default.weights)


@pytest.mark.parametrize("l2_lambda", [-1.0, float("inf"), float("nan")])
def test_l2_lambda_must_be_finite_and_non_negative(
    feasible_frame,
    l2_lambda: float,
) -> None:
    frame, truths = feasible_frame(n=40)
    targets = TargetSet((_population_target(truths["population"], 1.0),))

    with pytest.raises(ValueError, match="l2_lambda"):
        calibrate(frame, targets, epochs=50, seed=0, l2_lambda=l2_lambda)


def test_l2_lambda_records_provenance(feasible_frame) -> None:
    frame, truths = feasible_frame(n=40)
    targets = TargetSet((_population_target(truths["population"], 1.1),))

    result = calibrate(frame, targets, epochs=50, seed=0, l2_lambda=0.001)

    assert result.options["l2_lambda"] == 0.001
    assert result.options["l2_penalty"] == "mean_initial_pre_gate_weight_ratio_squared"


def test_l2_lambda_reduces_weight_concentration() -> None:
    """A positive L2 penalty is a smooth concentration/ESS control.

    The shifted targets below are easiest to fit by moving mass toward the
    highest-income non-renter records. A small L2 penalty keeps the fit close
    while making that concentration materially less extreme.
    """

    frame, targets, initial_weights = _l2_concentration_fixture()

    baseline = calibrate(
        frame,
        targets,
        epochs=500,
        seed=0,
        mass="conserve",
        learning_rate=0.05,
    )
    penalized = calibrate(
        frame,
        targets,
        epochs=500,
        seed=0,
        mass="conserve",
        learning_rate=0.05,
        l2_lambda=0.001,
    )

    baseline_ratio = baseline.weights / initial_weights
    penalized_ratio = penalized.weights / initial_weights
    assert penalized_ratio.max() < baseline_ratio.max() * 0.5
    assert _effective_sample_size(penalized.weights) > (
        _effective_sample_size(baseline.weights) * 2.0
    )
    assert penalized.final_loss <= baseline.final_loss + 0.02


def test_l2_lambda_reduces_concentration_with_l0_gates_active() -> None:
    frame, targets, initial_weights = _l2_concentration_fixture()

    baseline = calibrate(
        frame,
        targets,
        epochs=400,
        seed=0,
        mass="conserve",
        learning_rate=0.05,
        l0_lambda=0.003,
    )
    penalized = calibrate(
        frame,
        targets,
        epochs=400,
        seed=0,
        mass="conserve",
        learning_rate=0.05,
        l0_lambda=0.003,
        l2_lambda=0.001,
    )

    assert baseline.n_nonzero < len(initial_weights)
    assert penalized.n_nonzero < len(initial_weights)
    baseline_ratio = baseline.weights / initial_weights
    penalized_ratio = penalized.weights / initial_weights
    assert penalized_ratio.max() < baseline_ratio.max() * 0.5
    assert _effective_sample_size(penalized.weights) > (
        _effective_sample_size(baseline.weights) * 1.5
    )
    assert penalized.final_loss <= baseline.final_loss + 0.1


def test_l2_lambda_is_fixed_during_target_record_budget_search(feasible_frame) -> None:
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )

    result = calibrate(
        frame,
        targets,
        epochs=250,
        seed=0,
        target_records=120,
        l2_lambda=0.001,
    )

    assert abs(result.n_nonzero - 120) <= 60, result.n_nonzero
    assert result.l0_lambda > 0.0
    assert result.options["l2_lambda"] == 0.001
    assert result.options["l2_penalty"] == "mean_initial_pre_gate_weight_ratio_squared"


def test_budget_iters_must_be_positive(feasible_frame) -> None:
    """A non-positive ``budget_iters`` is rejected with a named error."""
    frame, truths = feasible_frame(n=50)
    targets = TargetSet((_population_target(truths["population"], 1.0),))
    with pytest.raises(ValueError, match="budget_iters"):
        calibrate(frame, targets, epochs=50, seed=0, target_records=10, budget_iters=0)


def test_prune_with_conserve_and_cap_keeps_pruned_records_pruned(
    feasible_frame,
) -> None:
    """L0 pruning survives mass-conservation + a weight cap (Finding 4).

    With ``mass="conserve"`` and a ``max_weight_ratio`` cap, the post-pruning
    mass deficit must be filled only over the surviving (gate-open) records.
    Filling it over *all* records with headroom — including the gate-closed,
    ~0-weight pruned ones — resurrects every record (the bug: free-mass pruned
    hundreds, conserve+cap returned zero pruned).

    The conserve+cap run uses ``target_records`` budget control rather than a
    fixed penalty, because a fixed ``l0_lambda`` prunes platform-dependently:
    3e-3 left tens of survivors on arm64 but 3 of 400 on x86 CI, where the
    freed mass had nowhere to go under the 100x cap (feasibility needs
    ``survivors >= n / ratio`` = 4) — while a cap loose enough to make any
    count feasible lets the conserve rescale feed an all-gates-closed
    collapse. The budget search adapts the penalty until the survivor count
    tracks the budget on *any* platform, keeping the scenario away from both
    cliffs. The resurrection bug is still caught: under it every probe
    returns ~400 nonzero records, nowhere near the budget or the assertion
    bound.
    """
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    free = calibrate(frame, targets, epochs=400, seed=0, l0_lambda=3e-2)
    free_nonzero = int((free.frame.resolve_weights("household").values > 1e-6).sum())
    assert free_nonzero < 100  # free mass prunes hard

    capped = calibrate(
        frame,
        targets,
        epochs=400,
        seed=0,
        target_records=80,
        mass="conserve",
        max_weight_ratio=100.0,
    )
    capped_w = capped.frame.resolve_weights("household").values
    capped_nonzero = int((capped_w > 1e-6).sum())
    # The pruned records stay pruned: the survivor count tracks the budget,
    # nowhere near the full 400 the resurrection bug returned.
    assert capped_nonzero < 150, capped_nonzero
    # Mass is still conserved on the survivors.
    initial_total = frame.resolve_weights("household").values.sum()
    assert abs(capped_w.sum() - initial_total) / initial_total < 1e-6


def test_all_gates_closed_raises_a_named_error(feasible_frame) -> None:
    """An L0 penalty that closes every gate raises a named error.

    A penalty far above the fit loss drives every gate shut (under Adam the
    per-parameter step is sign-dominated, so the penalty's gradient marches
    every gate logit down regardless of scale) and the calibrated vector is
    all zeros. The kernel would reject that with an opaque "Weights cannot be
    all zero"; calibrate must instead name the cause — the penalty — and the
    remedies, before the kernel ever sees the weights.
    """
    frame, truths = feasible_frame(n=50)
    targets = TargetSet((_income_target(truths["income"], 1.0),))
    with pytest.raises(ValueError, match="l0_lambda"):
        calibrate(frame, targets, epochs=300, seed=0, l0_lambda=100.0)


def test_cap_below_one_with_conserve_is_rejected_a_priori(feasible_frame) -> None:
    """``max_weight_ratio < 1`` with ``mass="conserve"`` is infeasible (Finding 7).

    Every capped weight is below its initial, so ``sum(cap) < total`` and the
    input mass can never be restored. This is infeasible before any optimization
    and must be rejected in argument validation, naming the three causes — not
    surfaced later as an opaque kernel mass-conservation failure.
    """
    frame, truths = feasible_frame(n=100)
    targets = TargetSet((_income_target(truths["income"], 1.0),))
    with pytest.raises(ValueError, match="max_weight_ratio"):
        calibrate(
            frame,
            targets,
            epochs=50,
            seed=0,
            mass="conserve",
            max_weight_ratio=0.5,
        )


def test_prune_conserve_cap_infeasible_when_survivors_lack_headroom(
    feasible_frame,
) -> None:
    """If the surviving records cannot absorb the deficit under the cap, raise.

    A tight cap (just above 1) leaves the few survivors almost no headroom, so
    they cannot soak up the mass freed by pruning. That is genuinely infeasible
    and must raise a clear error naming pruning + conserve + cap, rather than
    silently resurrecting the pruned records to balance the books.
    """
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    with pytest.raises(ValueError, match="prun.*conserve.*cap|conserve.*cap.*prun"):
        calibrate(
            frame,
            targets,
            epochs=400,
            seed=0,
            l0_lambda=5e-3,
            mass="conserve",
            max_weight_ratio=1.05,
        )


def test_final_loss_describes_the_returned_weights(feasible_frame) -> None:
    """``final_loss`` is the loss of the weights actually returned (Finding 8).

    The loss trajectory records each epoch's value *before* that epoch's step and
    before the closing mass/cap projections, so its tail does not describe the
    returned vector. With ``mass="conserve"`` and a cap, the closing projection
    moves the weights, so the trajectory tail and the true loss on the returned
    weights diverge. ``final_loss`` must equal the loss recomputed on the
    returned weights.
    """
    frame, truths = feasible_frame(n=200)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.5),
            _income_target(truths["income"], 1.5),
        )
    )
    result = calibrate(
        frame,
        targets,
        epochs=300,
        seed=0,
        mass="conserve",
        max_weight_ratio=2.0,
        learning_rate=0.1,
    )
    # Recompute the capped weighted-MAPE loss on the returned weights directly.
    # final_loss is a float64 closing eval, so it matches to machine epsilon.
    b = result.problem.target_vector
    scales = default_target_loss_scales(b)
    est = result.problem.estimates(result.weights)
    true_loss = relative_error_loss(est, b, target_loss_scales=scales)
    assert abs(result.final_loss - true_loss) < 1e-9
    # final_loss is NOT merely the trajectory tail (which is pre-projection): on
    # this conserve+cap run they differ materially.
    assert abs(result.final_loss - float(result.loss_trajectory[-1])) > 1e-4
    # initial_loss still describes the input weights (the trajectory head),
    # to float32 precision since the trajectory is computed in float32 torch.
    est0 = result.problem.estimates(result.initial_weights)
    true_initial = relative_error_loss(est0, b, target_loss_scales=scales)
    assert abs(result.initial_loss - true_initial) < 1e-5


def test__given_warm_start_weights__then_optimizer_starts_from_warm_loss(
    feasible_frame,
) -> None:
    frame, truths = feasible_frame(n=80)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.2),
            _income_target(truths["income"], 0.9),
        )
    )

    cold = calibrate(frame, targets, epochs=1, seed=0, mass="conserve")
    prior = calibrate(frame, targets, epochs=80, seed=0, mass="conserve")
    warm = calibrate(
        frame,
        targets,
        epochs=1,
        seed=0,
        mass="conserve",
        warm_start_weights=prior.weights,
    )

    assert warm.initial_loss < cold.initial_loss
    assert warm.options["warm_start_weights"] == {
        "enabled": True,
        "kind": "explicit",
    }
    np.testing.assert_allclose(warm.initial_weights, prior.initial_weights)
    assert warm.diagnostics[0].initial_estimate == cold.diagnostics[0].initial_estimate


def test__given_bad_warm_start_shape__then_solver_rejects_it(feasible_frame) -> None:
    frame, truths = feasible_frame(n=20)
    targets = TargetSet((_population_target(truths["population"], 1.0),))
    bad = np.ones(19)

    with pytest.raises(ValueError, match="warm_start_weights shape"):
        calibrate(frame, targets, epochs=1, seed=0, warm_start_weights=bad)


def test__given_l0_budget_and_warm_start__then_solver_rejects_it(
    feasible_frame,
) -> None:
    frame, truths = feasible_frame(n=40)
    targets = TargetSet((_population_target(truths["population"], 1.0),))
    weights = frame.resolve_weights("household").values.copy()

    with pytest.raises(ValueError, match="warm_start_weights.*L0"):
        calibrate(
            frame,
            targets,
            epochs=1,
            seed=0,
            target_records=20,
            warm_start_weights=weights,
        )


def test_default_target_loss_scales_ignore_initial_estimates() -> None:
    targets = np.asarray([0.0, 10.0, 1_000.0])
    initial_estimates = np.asarray([1_000_000.0, 20_000.0, 1.0])

    scales = default_target_loss_scales(targets, initial_estimates)

    np.testing.assert_allclose(scales, np.asarray([1.0, 10.0, 1_000.0]))


def test_default_target_loss_scales_ignore_nonfinite_initial_estimates() -> None:
    targets = np.asarray([0.0, 10.0, 1_000.0])
    initial_estimates = np.asarray([np.inf, np.nan, -np.inf])

    scales = default_target_loss_scales(targets, initial_estimates)

    np.testing.assert_allclose(scales, np.asarray([1.0, 10.0, 1_000.0]))


def test_small_valued_targets_converge_to_the_value_not_value_minus_one() -> None:
    """The loss is minimized at est == target, not est == target - 1.

    A +1 in the loss numerator biases the optimum to target - 1 — negligible at
    $2T magnitudes, fatal for small targets: a count of 5 converges to 4. The
    numerator must be the raw residual.
    """
    import numpy as np
    import pandas as pd

    from populace.frame import EntitySchema, Frame, WeightKind, Weights

    n = 50
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {"household_id": range(n), "household_count": np.ones(n)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.full(n, 0.1), kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (
            Target(
                name="count",
                entity="household",
                measure="household_count",
                value=5.0,
            ),
        )
    )
    result = calibrate(frame, targets, epochs=400, seed=0)
    estimate = result.frame.resolve_weights("household").values.sum()
    assert abs(estimate - 5.0) < 0.05  # not 4.0


def test_budget_search_survives_a_mid_search_infeasible_probe(feasible_frame) -> None:
    """A feasible budget under conserve + cap must not crash on a search probe.

    The L0 budget search bisects ``l0_lambda``; an intermediate (too-large)
    penalty can over-prune past what the surviving records can carry under the
    cap, tripping the conserve+cap infeasibility raise *inside* the search. That
    must be caught and steered (toward a smaller penalty), not surfaced as a
    failure that wrongly blames the user's budget — which is itself feasible.
    """
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    # 395 survivors at cap 1.05x can carry the input total (395*1.05 > 400), so
    # the budget is feasible — the search must reach it, not raise.
    result = calibrate(
        frame,
        targets,
        epochs=200,
        seed=0,
        target_records=395,
        mass="conserve",
        max_weight_ratio=1.05,
    )
    assert result.n_nonzero > 0
    # Mass is still conserved on the survivors.
    initial_total = frame.resolve_weights("household").values.sum()
    final_total = result.frame.resolve_weights("household").values.sum()
    assert abs(final_total - initial_total) / initial_total < 1e-6


def test_calibrate_reports_epoch_progress(feasible_frame) -> None:
    frame, truths = feasible_frame(n=30)
    targets = TargetSet((_population_target(truths["population"], 1.0),))
    events: list[dict[str, object]] = []

    calibrate(frame, targets, epochs=5, seed=0, progress_callback=events.append)

    assert [event["kind"] for event in events] == ["calibration_epoch"] * 5
    assert [event["epoch"] for event in events] == [1, 2, 3, 4, 5]
    assert all(event["epochs"] == 5 for event in events)
    assert all(isinstance(event["loss"], float) for event in events)
