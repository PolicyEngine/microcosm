"""Behavioral contracts for the calibration solver.

Each test pins a property the charter promises: calibration reduces target loss
and hits feasible targets, produces CALIBRATED weights, conserves declared mass
only when asked, respects the hard weight-ratio bound (and so cannot detonate a
landmine), stacks multi-period targets over one weight vector, and prunes toward
a record budget with L0.
"""

from __future__ import annotations

import pytest

from populace.calibrate import Target, TargetSet, calibrate
from populace.frame import WeightKind


def _income_target(truth: float, factor: float) -> Target:
    return Target(
        name="income",
        entity="household",
        aggregation="sum",
        value=truth * factor,
        measure="income",
    )


def _population_target(truth: float, factor: float) -> Target:
    return Target(
        name="population",
        entity="household",
        aggregation="count",
        value=truth * factor,
    )


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
    assert result.final_loss < result.initial_loss * 1e-3
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


def test_max_weight_ratio_is_respected(feasible_frame) -> None:
    frame, truths = feasible_frame()
    w0 = frame.resolve_weights("household").values
    targets = TargetSet((_income_target(truths["income"], 3.0),))
    result = calibrate(
        frame, targets, epochs=300, seed=0, max_weight_ratio=2.0
    )
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
                aggregation="sum",
                value=donor_value * 5000.0,
                measure="capital_gains",
            ),
        )
    )

    unbounded = calibrate(frame, far_target, epochs=300, seed=0)
    unbounded_w = unbounded.frame.resolve_weights("household").values
    bounded = calibrate(
        frame, far_target, epochs=300, seed=0, max_weight_ratio=10.0
    )
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
                aggregation="sum",
                value=truth_2026 * 1.3,
                measure="income_2026",
                period=2026,
            ),
            Target(
                name="income",
                entity="household",
                aggregation="sum",
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


def test_prune_with_conserve_and_cap_keeps_pruned_records_pruned(
    feasible_frame,
) -> None:
    """L0 pruning survives mass-conservation + a weight cap (Finding 4).

    With ``mass="conserve"`` and a ``max_weight_ratio`` cap, the post-pruning
    mass deficit must be filled only over the surviving (gate-open) records.
    Filling it over *all* records with headroom — including the gate-closed,
    ~0-weight pruned ones — resurrects every record (the bug: free-mass pruned
    hundreds, conserve+cap returned zero pruned).

    The cap here is loose enough that the survivors *can* absorb the freed mass
    (a count target at factor 1.0 means the survivors must carry the full input
    total, so a tight cap is genuinely infeasible — tested separately).
    """
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    free = calibrate(frame, targets, epochs=400, seed=0, l0_lambda=3e-3)
    free_nonzero = int((free.frame.resolve_weights("household").values > 1e-6).sum())
    assert free_nonzero < 100  # free mass prunes hard

    capped = calibrate(
        frame, targets, epochs=400, seed=0, l0_lambda=3e-3,
        mass="conserve", max_weight_ratio=100.0,
    )
    capped_w = capped.frame.resolve_weights("household").values
    capped_nonzero = int((capped_w > 1e-6).sum())
    # The pruned records stay pruned: the survivor count is still small, nowhere
    # near the full 400 the resurrection bug returned.
    assert capped_nonzero < 150, capped_nonzero
    # Mass is still conserved on the survivors.
    initial_total = frame.resolve_weights("household").values.sum()
    assert abs(capped_w.sum() - initial_total) / initial_total < 1e-6


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
            frame, targets, epochs=50, seed=0,
            mass="conserve", max_weight_ratio=0.5,
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
            frame, targets, epochs=400, seed=0, l0_lambda=5e-3,
            mass="conserve", max_weight_ratio=1.05,
        )


def test_small_valued_targets_converge_to_the_value_not_value_minus_one() -> None:
    """The loss is minimized at est == target, not est == target - 1.

    A +1 in the loss numerator (the eCPS reference carries one) biases the
    optimum to target - 1 — negligible at $2T magnitudes, fatal for small
    targets: a count of 5 converges to 4. The numerator must be the raw
    residual.
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
            "household": pd.DataFrame({"household_id": range(n)}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.full(n, 0.1), kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (Target(name="count", entity="household", aggregation="count", value=5.0),)
    )
    result = calibrate(frame, targets, epochs=400, seed=0)
    estimate = result.frame.resolve_weights("household").values.sum()
    assert abs(estimate - 5.0) < 0.05  # not 4.0
