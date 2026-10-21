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
    y = rng.lognormal(7.0, 0.8, n)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame({"household_id": range(n), "x": x, "y": y}),
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
    kwargs = dict(
        epochs=600,
        learning_rate=0.02,
        mass="free",
        l0_lambda=1e-4,
        target_loss_weights=_LOSS_WEIGHTS,
    )
    plain = calibrate_l0_refit(frame, _conflict(frame, x), **kwargs)
    result = calibrate_l0_refit(
        frame,
        _conflict(frame, x),
        **kwargs,
        target_band_tolerances=np.array([0.05, np.nan]),
        target_band_penalty=1.0,
    )
    assert result.selection.options["target_band"]["enabled"] is True
    assert result.refit.options["target_band"]["enabled"] is True
    # The selection phase solves through stochastic gates with an L0 penalty,
    # so the band pulls it in without holding it; the refit, whose weights
    # ship, holds it.
    assert _critical_error(plain.selection) > 0.15
    assert _critical_error(result.selection) < _critical_error(plain.selection) - 0.1
    assert _critical_error(plain.refit) > 0.15
    assert abs(_critical_error(result.refit)) <= 0.05 + 2e-3
    assert result.refit.options["target_band"]["max_final_violation"] <= 2e-3


@pytest.mark.parametrize(
    ("path", "solve"),
    [
        ("prox", dict(method="prox", epochs=600, learning_rate=0.02, mass="free")),
        (
            "budget search",
            dict(epochs=300, learning_rate=0.02, mass="free", target_records=60),
        ),
        (
            "conserve mass with a ratio cap",
            dict(epochs=600, learning_rate=0.1, mass="conserve", max_weight_ratio=5.0),
        ),
    ],
)
def test_the_band_holds_on_every_solve_path(path, solve) -> None:
    frame, x = _frame()
    targets = _conflict(frame, x)
    plain = calibrate(
        frame, targets, target_loss_weights=_LOSS_WEIGHTS, seed=0, **solve
    )
    banded = calibrate(
        frame,
        targets,
        target_loss_weights=_LOSS_WEIGHTS,
        seed=0,
        target_band_tolerances=np.array([0.05, np.nan]),
        target_band_penalty=1.0,
        **solve,
    )
    assert _critical_error(plain) > 0.10, path
    assert abs(_critical_error(banded)) <= 0.05 + 2e-3, path
    assert banded.options["target_band"]["max_final_violation"] <= 2e-3, path
    if solve["mass"] == "conserve":
        np.testing.assert_allclose(banded.weights.sum(), 100.0 * len(x), rtol=1e-9)
        assert (banded.weights <= 5.0 * 100.0 * (1 + 1e-12)).all()


def test_a_band_on_only_skipped_targets_is_refused() -> None:
    # The input check refuses a band on no row; a band whose every row the
    # constraint matrix skipped (here, a missing column) must not solve
    # silently without a band.
    frame, x = _frame()
    targets = TargetSet(
        (
            Target(name="missing", entity="household", value=1.0, measure="absent"),
            Target(name="heavier", entity="household", value=1.0e6, measure="x"),
        )
    )
    with pytest.raises(ValueError, match="skipped: missing@0"):
        calibrate(
            frame,
            targets,
            epochs=2,
            target_band_tolerances=np.array([0.05, np.nan]),
            target_band_penalty=1.0,
        )


def test_a_skipped_banded_target_is_listed_and_the_rest_stay_aligned() -> None:
    frame, x = _frame()
    conflict = _conflict(frame, x).targets
    targets = TargetSet(
        (
            Target(name="missing", entity="household", value=1.0, measure="absent"),
            conflict[1],
            conflict[0],
        )
    )
    result = calibrate(
        frame,
        targets,
        target_loss_weights=np.array([1.0, 9.0, 1.0]),
        target_band_tolerances=np.array([0.05, np.nan, 0.05]),
        target_band_penalty=1.0,
        **_SOLVE,
    )
    band = result.options["target_band"]
    assert band["skipped_banded_targets"] == ["missing@0"]
    assert [row["target"] for row in band["targets"]] == ["critical@0"]
    critical = [t.name for t in result.problem.targets].index("critical")
    estimate = float(result.problem.estimates(result.weights)[critical])
    target = float(result.problem.target_vector[critical])
    assert abs(estimate / target - 1.0) <= 0.05 + 2e-3


def _tiny_frame(n: int = 50) -> tuple[Frame, np.ndarray]:
    x = np.random.default_rng(0).lognormal(8.0, 0.6, n)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {"household_id": range(n), "x": x, "tiny": np.full(n, 1e-52)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.full(n, 100.0), kind=WeightKind.DESIGN)},
    )
    return frame, x


def test_an_unbanded_target_below_float32_range_cannot_poison_the_band() -> None:
    # A nonzero target too small for float32 has a zero float32 scale. Only
    # banded rows enter the band term, so this unbanded one cannot divide 0 by
    # 0 in the gradient.
    frame, x = _tiny_frame()
    targets = TargetSet(
        (
            Target(
                name="x", entity="household", value=float((x * 100).sum()), measure="x"
            ),
            Target(name="tiny", entity="household", value=2e-50, measure="tiny"),
        )
    )
    result = calibrate(
        frame,
        targets,
        epochs=50,
        mass="free",
        seed=0,
        target_band_tolerances=np.array([0.05, np.nan]),
        target_band_penalty=1.0,
    )
    assert np.isfinite(result.weights).all()


def test_a_banded_target_below_float32_range_is_refused() -> None:
    frame, x = _tiny_frame()
    targets = TargetSet(
        (Target(name="tiny", entity="household", value=2e-50, measure="tiny"),)
    )
    with pytest.raises(ValueError, match="rounds to zero in float32"):
        calibrate(
            frame,
            targets,
            epochs=2,
            target_band_tolerances=np.array([0.05]),
            target_band_penalty=1.0,
        )


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
# Targets are counts or dollars: zero, or at least 1e-3 in magnitude. The
# floor keeps the float32 term (the solve's dtype) away from subnormal
# magnitudes, where it cannot match the float64 reference to 1e-4; the
# below-float32-range cases have their own tests above.
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
