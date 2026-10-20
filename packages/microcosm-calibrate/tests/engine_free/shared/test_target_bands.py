"""Target bands: tolerances the solve holds instead of trading them for fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.calibrate import Target, TargetSet, calibrate, calibrate_l0_refit
from microcosm.calibrate import solve as solve_module
from microcosm.calibrate.solve import target_band_violations
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _frame(n: int = 120, seed: int = 0) -> tuple[Frame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.lognormal(8.0, 0.6, n)
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
    return frame, x


def _conflict(frame: Frame, x: np.ndarray) -> TargetSet:
    """Two targets on one feature, 20% apart: the critical one and a heavier one."""

    total = float((x * 100.0).sum())
    return TargetSet(
        (
            Target(name="critical", entity="household", value=total, measure="x"),
            Target(name="heavier", entity="household", value=total * 1.2, measure="x"),
        )
    )


_SOLVE = dict(epochs=600, learning_rate=0.02, mass="free", seed=0)
_LOSS_WEIGHTS = np.array([1.0, 9.0])  # the heavier target carries 90% of the loss


def _critical_error(result) -> float:
    estimate = float(result.problem.estimates(result.weights)[0])
    target = float(result.problem.target_vector[0])
    return (estimate - target) / target


def test_without_a_band_the_heavier_target_wins() -> None:
    frame, x = _frame()
    result = calibrate(
        frame, _conflict(frame, x), target_loss_weights=_LOSS_WEIGHTS, **_SOLVE
    )
    assert _critical_error(result) > 0.15
    assert result.options["target_band"] == {"enabled": False, "penalty": 0.0}


def test_a_band_holds_the_critical_target_inside_its_tolerance() -> None:
    frame, x = _frame()
    result = calibrate(
        frame,
        _conflict(frame, x),
        target_loss_weights=_LOSS_WEIGHTS,
        target_band_tolerances=np.array([0.05, np.nan]),
        target_band_penalty=1.0,
        **_SOLVE,
    )
    assert abs(_critical_error(result)) <= 0.05 + 2e-3
    band = result.options["target_band"]
    assert band["enabled"] is True
    assert band["banded_targets"] == 1
    [row] = band["targets"]
    assert row["target"] == "critical@0"
    assert row["tolerance"] == 0.05
    assert row["final_violation"] == pytest.approx(
        max(abs(row["final_relative_error"]) - 0.05, 0.0)
    )
    # The fit loss itself is unchanged: final_loss is still relative_error_loss.
    assert result.final_loss == pytest.approx(
        solve_module.relative_error_loss(
            result.problem.estimates(result.weights),
            result.problem.target_vector,
            target_loss_weights=_LOSS_WEIGHTS,
            target_loss_scales=result.target_loss_scales,
            target_loss_cap=result.target_loss_cap,
        )
    )


def test_a_band_that_never_binds_changes_nothing() -> None:
    frame, x = _frame()
    targets = _conflict(frame, x)
    plain = calibrate(frame, targets, target_loss_weights=_LOSS_WEIGHTS, **_SOLVE)
    wide = calibrate(
        frame,
        targets,
        target_loss_weights=_LOSS_WEIGHTS,
        target_band_tolerances=np.array([10.0, np.nan]),
        target_band_penalty=1.0,
        **_SOLVE,
    )
    np.testing.assert_array_equal(plain.weights, wide.weights)


def test_tolerances_without_a_penalty_change_nothing() -> None:
    frame, x = _frame()
    targets = _conflict(frame, x)
    plain = calibrate(frame, targets, target_loss_weights=_LOSS_WEIGHTS, **_SOLVE)
    unpenalized = calibrate(
        frame,
        targets,
        target_loss_weights=_LOSS_WEIGHTS,
        target_band_tolerances=np.array([0.05, np.nan]),
        target_band_penalty=0.0,
        **_SOLVE,
    )
    np.testing.assert_array_equal(plain.weights, unpenalized.weights)
    assert unpenalized.options["target_band"]["enabled"] is False


def test_the_band_reaches_both_phases_of_an_l0_refit() -> None:
    frame, x = _frame()
    result = calibrate_l0_refit(
        frame,
        _conflict(frame, x),
        epochs=300,
        mass="conserve",
        l0_lambda=1e-4,
        target_loss_weights=_LOSS_WEIGHTS,
        target_band_tolerances=np.array([0.05, np.nan]),
        target_band_penalty=1.0,
    )
    assert result.selection.options["target_band"]["enabled"] is True
    assert result.refit.options["target_band"]["enabled"] is True


@pytest.mark.parametrize(
    ("tolerances", "penalty", "message"),
    [
        (None, 1.0, "needs target_band_tolerances"),
        (np.array([np.nan, np.nan]), 1.0, "bands no row"),
        (np.array([-0.1, np.nan]), 1.0, "finite and non-negative"),
        (np.array([np.inf, np.nan]), 1.0, "finite and non-negative"),
        (np.array([0.05]), 1.0, "must match the target shape"),
        (np.array([0.05, np.nan]), -1.0, "target_band_penalty must be"),
        (np.array([0.05, np.nan]), float("nan"), "target_band_penalty must be"),
    ],
)
def test_malformed_bands_are_refused(tolerances, penalty, message) -> None:
    frame, x = _frame()
    with pytest.raises(ValueError, match=message):
        calibrate(
            frame,
            _conflict(frame, x),
            epochs=2,
            target_band_tolerances=tolerances,
            target_band_penalty=penalty,
        )


# --------------------------------------------------------------------------- #
# The violation measure                                                       #
# --------------------------------------------------------------------------- #

_VALUES = st.floats(
    min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False
)
# Targets are counts or dollars: zero, or at least 1e-3 in magnitude, which
# float32 (the solve's dtype) represents without underflow.
_TARGETS = st.one_of(
    st.just(0.0),
    st.floats(min_value=1e-3, max_value=1e9),
    st.floats(min_value=-1e9, max_value=-1e-3),
)


@st.composite
def _rows(draw):
    n = draw(st.integers(min_value=1, max_value=12))
    estimates = np.array(draw(st.lists(_VALUES, min_size=n, max_size=n)))
    targets = np.array(draw(st.lists(_TARGETS, min_size=n, max_size=n)))
    tolerances = np.array(
        draw(
            st.lists(
                st.one_of(
                    st.just(float("nan")), st.floats(min_value=0.0, max_value=2.0)
                ),
                min_size=n,
                max_size=n,
            )
        )
    )
    return estimates, targets, tolerances


def _relative_errors(estimates, targets):
    return np.where(
        targets == 0.0,
        estimates - targets,
        (estimates - targets) / np.where(targets == 0.0, 1.0, targets),
    )


@settings(max_examples=200, deadline=None)
@given(rows=_rows())
def test_property_violation_is_the_distance_outside_the_band(rows) -> None:
    estimates, targets, tolerances = rows
    violations = target_band_violations(estimates, targets, tolerances)
    errors = np.abs(_relative_errors(estimates, targets))
    banded = ~np.isnan(tolerances)
    assert (violations >= 0.0).all()
    assert (violations[~banded] == 0.0).all()
    inside = banded & (errors <= tolerances)
    assert (violations[inside] == 0.0).all()
    outside = banded & (errors > tolerances)
    np.testing.assert_allclose(
        violations[outside], errors[outside] - tolerances[outside], rtol=1e-12, atol=0.0
    )


@settings(max_examples=100, deadline=None)
@given(rows=_rows(), scale=st.floats(min_value=1e-3, max_value=1e3))
def test_property_violation_does_not_depend_on_units(rows, scale) -> None:
    estimates, targets, tolerances = rows
    targets = np.where(targets == 0.0, 1.0, targets)  # zero targets use absolute error
    base = target_band_violations(estimates, targets, tolerances)
    scaled = target_band_violations(estimates * scale, targets * scale, tolerances)
    np.testing.assert_allclose(scaled, base, rtol=1e-9, atol=1e-12)


@settings(max_examples=100, deadline=None)
@given(rows=_rows(), penalty=st.floats(min_value=0.01, max_value=100.0))
def test_property_torch_term_matches_the_numpy_reference(rows, penalty) -> None:
    estimates, targets, tolerances = rows
    if np.isnan(tolerances).all():
        tolerances = tolerances.copy()
        tolerances[0] = 0.1
    band = solve_module._TargetBand.build(tolerances, targets, penalty)
    term = float(
        band.term(
            torch.tensor(estimates, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )
    )
    reference = penalty * float(
        target_band_violations(estimates, targets, tolerances).sum()
    )
    # float32 against float64: relative tolerance on the total, plus an absolute floor.
    assert term == pytest.approx(reference, rel=1e-4, abs=1e-3 * penalty)


def test_violations_refuse_misaligned_or_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="must align"):
        target_band_violations(np.zeros(2), np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        target_band_violations(np.array([np.nan]), np.array([1.0]), np.array([0.1]))
