"""Contracts for the L2 basis and the softmax mass parametrization.

Invariants pinned here (each holds for every input the strategies draw):

1. The chi-square penalty is zero exactly at ``w = anchor`` and nonnegative
   everywhere; the solver's float32 torch form agrees with the float64
   reference :func:`chi_square_distance`.
2. When ``sum(w) == sum(d)`` the distance equals ``sum(w**2 / d) / sum(d) - 1``
   (the weighting effect relative to ``d`` minus one), and with uniform ``d``
   it equals ``n / ESS(w) - 1``.
3. Under a mass constraint the chi-square penalty is stationary at
   ``w = d`` and the record penalty at ``w ∝ d**2`` (the reduced gradient
   vanishes there).
4. With no targets and ``mass="conserve"``, the chi-square solve returns the
   design weights from any warm start, under both mass parametrizations.
5. The softmax parametrization conserves the input total, respects the ratio
   cap, and keeps every weight positive.
6. Each solve under softmax-conserve and free mass lands within a bound of
   the exact optimum of the same convex program (differential against
   CLARABEL on these tests' own problems, fixture
   ``tests/fixtures/l2_basis_path_reference.json``), and the realized
   chi-square distance is nonincreasing in ``l2_lambda`` to within the slack
   that bound implies (the regularization-path identity), so with uniform
   design weights Kish ESS is nondecreasing too.
7. The defaults (``l2_basis="record"``, ``mass_parametrization="projection"``)
   reproduce the pre-change optimizer's bytes.

Intended violation, pinned as such: under the default projection
parametrization invariant 6 does not hold in general. When every record's
gradient shares a sign the per-step mass shift cancels Adam's normalized step,
so the solve stalls (``test_projection_parametrization_stalls_under_uniform
_mass_pressure``); the softmax option exists for that reason.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import json
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.calibrate import (
    L2_BASIS_CHI_SQUARE,
    L2_BASIS_RECORD,
    MASS_PARAMETRIZATION_PROJECTION,
    MASS_PARAMETRIZATION_SOFTMAX,
    Target,
    TargetSet,
    calibrate,
    calibrate_l0_refit,
    chi_square_distance,
    effective_sample_size,
    refit_l0_selection,
)
from microcosm.calibrate import solve as solve_module
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-calibrate")

# Solver-backed properties run a torch optimization per example, so they use
# few, derandomized examples; the pure penalty algebra runs many.
_SOLVER_SETTINGS = settings(max_examples=8, deadline=None, derandomize=True)
_ALGEBRA_SETTINGS = settings(max_examples=200, deadline=None)


def _frame(columns: dict[str, np.ndarray], design: np.ndarray) -> Frame:
    n = len(design)
    return Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame({"household_id": range(n), **columns}),
        },
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                values=np.asarray(design, float), kind=WeightKind.DESIGN
            )
        },
    )


def _problem(
    seed: int,
    *,
    n: int = 80,
    k: int = 5,
    factors: tuple[float, float] = (0.7, 1.4),
    uniform: bool = False,
) -> tuple[Frame, TargetSet, np.ndarray]:
    """A sparse synthetic surface whose targets pull away from the design."""
    rng = np.random.default_rng(seed)
    values = rng.lognormal(0.0, 1.0, (n, k)) * (rng.random((n, k)) < 0.6)
    design = np.full(n, 10.0) if uniform else rng.lognormal(2.0, 0.8, n)
    scale = rng.uniform(*factors, k)
    columns = {f"x{j}": values[:, j] for j in range(k)}
    targets = TargetSet(
        tuple(
            Target(
                name=f"x{j}",
                entity="household",
                value=float((values[:, j] * design).sum() * scale[j]),
                measure=f"x{j}",
            )
            for j in range(k)
        )
    )
    return _frame(columns, design), targets, design


def _null_problem(design: np.ndarray) -> tuple[Frame, TargetSet]:
    """A target the weights cannot affect: zero measure, zero value.

    Its scaled miss is exactly zero for every weight vector, and so is its
    gradient, which makes the solve's objective the L2 penalty alone.
    """
    n = len(design)
    frame = _frame({"null": np.zeros(n)}, design)
    targets = TargetSet(
        (Target(name="null", entity="household", value=0.0, measure="null"),)
    )
    return frame, targets


def _positive_vectors(min_size: int = 1, max_size: int = 40):
    return st.integers(min_size, max_size).flatmap(
        lambda n: st.tuples(
            st.lists(
                st.floats(1e-3, 1e4, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            ),
            st.lists(
                st.floats(0.0, 1e4, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            ),
        )
    )


def _torch_penalty(weights: np.ndarray, anchor: np.ndarray, basis: str) -> float:
    anchor_t = torch.tensor(anchor, dtype=torch.float32)
    share_t = torch.tensor(anchor / anchor.sum(), dtype=torch.float32)
    return float(
        solve_module._l2_penalty(
            torch.tensor(weights, dtype=torch.float32), anchor_t, share_t, basis
        )
    )


# --- 1-3: the penalty's algebra ---------------------------------------------


@_ALGEBRA_SETTINGS
@given(_positive_vectors())
def test_chi_square_penalty_is_zero_at_the_anchor_and_nonnegative(vectors) -> None:
    anchor, weights = (np.asarray(v, dtype=np.float64) for v in vectors)
    assert _torch_penalty(anchor, anchor, L2_BASIS_CHI_SQUARE) == 0.0
    assert chi_square_distance(anchor, anchor) == 0.0
    assert _torch_penalty(weights, anchor, L2_BASIS_CHI_SQUARE) >= 0.0
    distance = chi_square_distance(weights, anchor)
    assert distance >= 0.0
    if not np.array_equal(weights, anchor):
        assert distance > 0.0


@_ALGEBRA_SETTINGS
@given(_positive_vectors())
def test_torch_chi_square_penalty_matches_the_float64_reference(vectors) -> None:
    """Differential: the solver's float32 penalty vs the float64 definition."""
    anchor, weights = (np.asarray(v, dtype=np.float64) for v in vectors)
    reference = chi_square_distance(weights, anchor)
    assert _torch_penalty(weights, anchor, L2_BASIS_CHI_SQUARE) == pytest.approx(
        reference, rel=1e-4, abs=1e-6
    )
    # The record basis is the historical mean((w / d) ** 2), 1 at w = d.
    assert _torch_penalty(weights, anchor, L2_BASIS_RECORD) == pytest.approx(
        float(np.mean((weights / anchor) ** 2)), rel=1e-4, abs=1e-6
    )
    assert _torch_penalty(anchor, anchor, L2_BASIS_RECORD) == pytest.approx(1.0)


@_ALGEBRA_SETTINGS
@given(_positive_vectors(min_size=2))
def test_chi_square_distance_is_the_weighting_effect_at_equal_totals(vectors) -> None:
    anchor, raw = (np.asarray(v, dtype=np.float64) for v in vectors)
    if raw.sum() == 0.0:
        raw = raw + 1.0
    weights = raw * (anchor.sum() / raw.sum())
    expected = float(np.sum(weights**2 / anchor) / anchor.sum() - 1.0)
    assert chi_square_distance(weights, anchor) == pytest.approx(
        expected, rel=1e-9, abs=1e-9
    )
    uniform = np.full_like(anchor, anchor.mean())
    rescaled = raw * (uniform.sum() / raw.sum())
    assert chi_square_distance(rescaled, uniform) == pytest.approx(
        len(uniform) / effective_sample_size(rescaled) - 1.0, rel=1e-9, abs=1e-9
    )


def test_chi_square_distance_edges_and_validation() -> None:
    assert chi_square_distance(np.array([0.0, 2.0]), np.array([0.0, 2.0])) == 0.0
    assert chi_square_distance(np.array([1.0, 2.0]), np.array([0.0, 2.0])) == (
        float("inf")
    )
    with pytest.raises(ValueError, match="shape"):
        chi_square_distance(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="non-negative anchor"):
        chi_square_distance(np.ones(2), np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="non-negative weights"):
        chi_square_distance(np.array([1.0, np.nan]), np.ones(2))
    with pytest.raises(ValueError, match="positive anchor total"):
        chi_square_distance(np.zeros(2), np.zeros(2))


@_ALGEBRA_SETTINGS
@given(
    st.lists(st.floats(1e-2, 1e3, allow_nan=False), min_size=2, max_size=30),
)
def test_mass_constrained_stationary_points_of_each_basis(anchor_values) -> None:
    """Chi-square is stationary at ``w = d``; the record basis at ``w ∝ d**2``.

    Stationary under ``sum(w) = D`` means the log-weight gradient is
    proportional to ``w`` (the constraint's normal in log space), i.e. the
    reduced gradient ``grad_i / w_i - mean_w(grad / w)`` is zero.
    """
    anchor = np.asarray(anchor_values, dtype=np.float64)
    total = anchor.sum()
    for basis, optimum in (
        (L2_BASIS_CHI_SQUARE, anchor.copy()),
        (L2_BASIS_RECORD, anchor**2 * total / np.sum(anchor**2)),
    ):
        log_w = torch.tensor(np.log(optimum), dtype=torch.float64, requires_grad=True)
        penalty = solve_module._l2_penalty(
            torch.exp(log_w),
            torch.tensor(anchor, dtype=torch.float64),
            torch.tensor(anchor / total, dtype=torch.float64),
            basis,
        )
        penalty.backward()
        per_unit = log_w.grad.numpy() / optimum
        reduced = per_unit - np.average(per_unit, weights=optimum)
        # 2 / total is the per-unit gradient of a unit ratio deviation; the
        # chi-square gradient at its optimum is roundoff of exp(log d) - d.
        scale = max(np.abs(per_unit).max(), 2.0 / total)
        assert np.abs(reduced).max() <= 1e-9 * scale, basis


# --- 4-5: target-free and softmax solves -------------------------------------


@_SOLVER_SETTINGS
@given(st.integers(0, 2**32 - 1))
def test_target_free_chi_square_solve_returns_the_design_weights(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 40))
    design = rng.lognormal(2.0, 1.2, n)
    frame, targets = _null_problem(design)
    warm = design * rng.uniform(0.25, 4.0, n)
    for parametrization in (
        MASS_PARAMETRIZATION_PROJECTION,
        MASS_PARAMETRIZATION_SOFTMAX,
    ):
        result = calibrate(
            frame,
            targets,
            epochs=1500,
            learning_rate=0.05,
            mass="conserve",
            l2_lambda=1.0,
            l2_basis=L2_BASIS_CHI_SQUARE,
            mass_parametrization=parametrization,
            warm_start_weights=warm,
            seed=0,
        )
        np.testing.assert_allclose(result.weights, design, rtol=1e-2)
        assert result.chi_square_distance < 1e-4
        assert result.weights.sum() == pytest.approx(design.sum(), rel=1e-12)


@_SOLVER_SETTINGS
@given(
    st.integers(0, 2**32 - 1),
    st.sampled_from([0.0, 1e-3, 1e-2, 1e-1, 1.0]),
    st.sampled_from([None, 1.5, 5.0]),
    st.sampled_from([L2_BASIS_RECORD, L2_BASIS_CHI_SQUARE]),
)
def test_softmax_parametrization_conserves_mass_and_respects_the_cap(
    seed: int, l2_lambda: float, cap: float | None, basis: str
) -> None:
    frame, targets, design = _problem(seed, n=40, k=3, factors=(0.5, 2.0))
    result = calibrate(
        frame,
        targets,
        epochs=300,
        learning_rate=0.05,
        mass="conserve",
        max_weight_ratio=cap,
        l2_lambda=l2_lambda,
        l2_basis=basis,
        mass_parametrization=MASS_PARAMETRIZATION_SOFTMAX,
        seed=0,
    )
    assert result.weights.sum() == pytest.approx(design.sum(), rel=1e-12)
    assert (result.weights > 0.0).all()
    if cap is not None:
        assert (result.weights <= cap * design * (1.0 + 1e-12)).all()
    assert result.options["mass_parametrization"] == MASS_PARAMETRIZATION_SOFTMAX
    assert result.options["iterate_selection"] == "closing_state"
    exhausted = result.options["iterate_selection_receipt"][
        "softmax_cap_rounds_exhausted_epochs"
    ]
    assert isinstance(exhausted, int) and 0 <= exhausted <= 300


# --- 6: the regularization path ----------------------------------------------

#: Dense at large lambda, where the derived slack 2 * eps / (lam_b - lam_a) is
#: small enough for the path identity to say something.
_LAMBDAS = (0.0, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0)
#: The path tests' problems: three seeds, heterogeneous and uniform design
#: weights, under the two settings whose Adam solves reach the optimum.
PATH_CASES = tuple(
    {"seed": seed, "uniform": uniform, "mass": mass, "parametrization": param}
    for seed in range(3)
    for uniform in (False, True)
    for mass, param in (
        ("conserve", MASS_PARAMETRIZATION_SOFTMAX),
        ("free", MASS_PARAMETRIZATION_PROJECTION),
    )
)


def path_case_id(case: dict) -> str:
    design = "uniform" if case["uniform"] else "lognormal"
    return f"s{case['seed']}_{design}_{case['mass']}_{case['parametrization']}"


def _path_reference() -> dict:
    path = _TEST_PATHS.tests / "fixtures" / "l2_basis_path_reference.json"
    return json.loads(path.read_text())


def _problem_digest(frame: Frame, targets: TargetSet) -> str:
    household = frame.table("household")
    digest = hashlib.sha256()
    for target in targets:
        digest.update(household[target.measure].to_numpy(np.float64).tobytes())
        digest.update(np.float64(target.value).tobytes())
    digest.update(
        frame.resolve_weights("household").values.astype(np.float64).tobytes()
    )
    return digest.hexdigest()


def _objective_parts(frame, targets, weights: np.ndarray) -> tuple[float, float]:
    household = frame.table("household")
    design = frame.resolve_weights("household").values
    misses = [
        abs(float(household[t.measure].to_numpy() @ weights) - t.value)
        / max(abs(t.value), 1.0)
        for t in targets
    ]
    return float(np.mean(misses)), chi_square_distance(weights, design)


@functools.cache
def _solved_path(case_id: str) -> tuple:
    case = next(c for c in PATH_CASES if path_case_id(c) == case_id)
    frame, targets, _ = _problem(case["seed"], uniform=case["uniform"])
    results = [
        calibrate(
            frame,
            targets,
            epochs=1500,
            learning_rate=0.02,
            mass=case["mass"],
            max_weight_ratio=5.0,
            l2_lambda=lam,
            l2_basis=L2_BASIS_CHI_SQUARE,
            mass_parametrization=case["parametrization"],
            seed=0,
        )
        for lam in _LAMBDAS
    ]
    return frame, targets, results


@pytest.mark.parametrize("case", PATH_CASES, ids=path_case_id)
def test_path_solves_reach_the_exact_optimum(case: dict) -> None:
    """Differential: each solve's objective against CLARABEL's exact optimum.

    The fixture holds the exact optimum of the same convex program for these
    very problems (``experiments/us-acs-local-l2-basis-20260928/
    test_path_reference.py``); its problem digests guard against generator
    drift. The loss cap never binds on them, so the solver's capped loss and
    the program's loss coincide.
    """
    reference = _path_reference()
    entry = reference["cases"][path_case_id(case)]
    frame, targets, results = _solved_path(path_case_id(case))
    assert _problem_digest(frame, targets) == entry["problem_sha256"]
    bound = reference["excess_objective_bound"]
    for lam, result in zip(_LAMBDAS, results, strict=True):
        loss, distance = _objective_parts(frame, targets, result.weights)
        assert loss == pytest.approx(result.final_loss, rel=1e-4, abs=1e-6)
        excess = loss + lam * distance - entry["optimum"][str(lam)]["objective"]
        assert excess <= bound, (lam, excess)


@pytest.mark.parametrize("case", PATH_CASES, ids=path_case_id)
def test_chi_square_distance_is_nonincreasing_in_lambda(case: dict) -> None:
    """The regularization-path identity, with its slack derived, not fitted.

    For exact minimizers at ``lam_a < lam_b``, adding the two optimality
    inequalities gives ``(lam_b - lam_a) * (P_b - P_a) <= 0``. Solves within
    ``eps`` of optimal (the differential test above) weaken it to
    ``P_b <= P_a + 2 * eps / (lam_b - lam_a)``. Only pairs where that slack is
    below the exact optimum's own ``P_a`` say anything; the others are skipped
    by name, and every case keeps at least two informative pairs. With uniform
    design weights under mass conservation, ``ESS = n / (1 + P)`` exactly, so
    Kish ESS is nondecreasing on the same pairs.
    """
    reference = _path_reference()
    optimum = reference["cases"][path_case_id(case)]["optimum"]
    eps = reference["excess_objective_bound"]
    frame, targets, results = _solved_path(path_case_id(case))
    exact = [optimum[str(lam)]["chi_square_distance"] for lam in _LAMBDAS]
    assert all(b <= a + 1e-9 for a, b in pairwise(exact)), exact
    informative = 0
    for (lam_a, a, exact_a), (lam_b, b, _) in pairwise(
        zip(_LAMBDAS, results, exact, strict=True)
    ):
        slack = 2.0 * eps / (lam_b - lam_a)
        if slack >= exact_a:
            continue
        informative += 1
        assert b.chi_square_distance <= a.chi_square_distance + slack, (lam_a, lam_b)
        if case["uniform"] and case["mass"] == "conserve":
            n = len(b.weights)
            for result in (a, b):
                assert result.effective_sample_size == pytest.approx(
                    n / (1.0 + result.chi_square_distance), rel=1e-9
                )
    assert informative >= 2


def test_projection_parametrization_stalls_under_uniform_mass_pressure() -> None:
    """Pins the mechanism the softmax option exists for (an intended violation).

    Every target sits above its design total, so every record's gradient
    points up. Adam normalizes each coordinate to roughly the same step and
    the per-step mass shift then cancels it: the projection solve leaves the
    design weights untouched for hundreds of epochs, while the softmax solve,
    whose gradient has the mass component removed, moves at once.
    """
    frame, targets, design = _problem(4, factors=(1.02, 1.4))
    common = dict(
        epochs=200,
        learning_rate=0.02,
        mass="conserve",
        max_weight_ratio=5.0,
        l2_lambda=0.01,
        l2_basis=L2_BASIS_CHI_SQUARE,
        seed=0,
    )
    stalled = calibrate(
        frame, targets, mass_parametrization=MASS_PARAMETRIZATION_PROJECTION, **common
    )
    moved = calibrate(
        frame, targets, mass_parametrization=MASS_PARAMETRIZATION_SOFTMAX, **common
    )
    assert stalled.chi_square_distance < 1e-6
    assert stalled.final_loss == pytest.approx(stalled.initial_loss, rel=1e-4)
    assert moved.final_loss < 0.5 * moved.initial_loss
    # And the record basis alone, target-free: its log-space gradient is
    # positive for every record, so the projection solve cannot reach w ∝ d².
    free_frame, free_targets = _null_problem(design)
    record = calibrate(
        free_frame,
        free_targets,
        epochs=200,
        learning_rate=0.05,
        mass="conserve",
        l2_lambda=1.0,
        l2_basis=L2_BASIS_RECORD,
        seed=0,
    )
    np.testing.assert_allclose(record.weights, design, rtol=1e-4)
    record_optimum = design**2 * design.sum() / np.sum(design**2)
    assert np.abs(record.weights / record_optimum - 1.0).max() > 0.5


# --- options, validation and threading ---------------------------------------


@pytest.mark.parametrize(
    ("basis", "anchor", "explicit", "label"),
    [
        (
            L2_BASIS_RECORD,
            "initial",
            False,
            "mean_initial_pre_gate_weight_ratio_squared",
        ),
        (
            L2_BASIS_RECORD,
            "uniform",
            False,
            "mean_uniform_pre_gate_weight_ratio_squared",
        ),
        (
            L2_BASIS_RECORD,
            "design",
            True,
            "mean_explicit_anchor_pre_gate_weight_ratio_squared",
        ),
        (
            L2_BASIS_CHI_SQUARE,
            "initial",
            False,
            "chi_square_initial_pre_gate_weight_distance",
        ),
        (
            L2_BASIS_CHI_SQUARE,
            "uniform",
            False,
            "chi_square_uniform_pre_gate_weight_distance",
        ),
        (
            L2_BASIS_CHI_SQUARE,
            "design",
            True,
            "chi_square_explicit_anchor_pre_gate_weight_distance",
        ),
    ],
)
def test_options_record_the_basis_and_penalty_label(
    basis: str, anchor: str, explicit: bool, label: str
) -> None:
    frame, targets, design = _problem(0, n=30, k=2)
    result = calibrate(
        frame,
        targets,
        epochs=5,
        mass="conserve",
        l2_lambda=0.01,
        l2_basis=basis,
        l2_anchor=anchor,
        l2_anchor_weights=design.copy() if explicit else None,
    )
    assert result.options["l2_basis"] == basis
    assert result.options["l2_penalty"] == label
    assert result.options["mass_parametrization"] == MASS_PARAMETRIZATION_PROJECTION


def test_explicit_design_anchor_is_the_initial_anchor_under_chi_square() -> None:
    frame, targets, design = _problem(1, n=30, k=2)
    common = dict(
        epochs=60, mass="conserve", l2_lambda=0.05, l2_basis=L2_BASIS_CHI_SQUARE
    )
    preset = calibrate(frame, targets, **common)
    explicit = calibrate(
        frame, targets, l2_anchor="design", l2_anchor_weights=design.copy(), **common
    )
    np.testing.assert_array_equal(explicit.weights, preset.weights)


def test_invalid_basis_and_parametrization_are_refused() -> None:
    frame, targets, _ = _problem(0, n=20, k=2)
    with pytest.raises(ValueError, match="l2_basis"):
        calibrate(frame, targets, epochs=2, l2_basis="bogus")
    with pytest.raises(ValueError, match="mass_parametrization must be one of"):
        calibrate(frame, targets, epochs=2, mass_parametrization="bogus")
    for bad in (
        dict(mass="free"),
        dict(mass="conserve", l0_lambda=0.01),
        dict(mass="conserve", target_records=5),
        dict(mass="conserve", method="prox"),
    ):
        with pytest.raises(ValueError, match="softmax"):
            calibrate(
                frame,
                targets,
                epochs=2,
                mass_parametrization=MASS_PARAMETRIZATION_SOFTMAX,
                **bad,
            )
    with pytest.raises(ValueError, match="refit_l2_basis"):
        calibrate_l0_refit(
            frame, targets, epochs=2, l0_lambda=0.003, refit_l2_basis="bogus"
        )
    with pytest.raises(ValueError, match="refit_mass_parametrization"):
        calibrate_l0_refit(
            frame,
            targets,
            epochs=2,
            l0_lambda=0.003,
            refit_mass_parametrization=MASS_PARAMETRIZATION_SOFTMAX,
        )


def test_l0_refit_threads_the_basis_and_the_refit_parametrization() -> None:
    frame, targets, _ = _problem(2, n=60, k=3)
    common = dict(epochs=40, seed=0, mass="conserve", l0_lambda=0.003)
    inherited = calibrate_l0_refit(
        frame, targets, l2_lambda=0.01, l2_basis=L2_BASIS_CHI_SQUARE, **common
    )
    assert inherited.options["selection_options"]["l2_basis"] == L2_BASIS_CHI_SQUARE
    assert inherited.options["l2_basis"] == L2_BASIS_CHI_SQUARE
    overridden = calibrate_l0_refit(
        frame,
        targets,
        l2_lambda=0.01,
        l2_basis=L2_BASIS_CHI_SQUARE,
        refit_l2_basis=L2_BASIS_RECORD,
        refit_mass_parametrization=MASS_PARAMETRIZATION_SOFTMAX,
        **common,
    )
    assert overridden.options["selection_options"]["l2_basis"] == L2_BASIS_CHI_SQUARE
    assert (
        overridden.options["selection_options"]["mass_parametrization"]
        == MASS_PARAMETRIZATION_PROJECTION
    )
    assert overridden.options["l2_basis"] == L2_BASIS_RECORD
    assert overridden.options["mass_parametrization"] == MASS_PARAMETRIZATION_SOFTMAX
    direct = refit_l0_selection(
        frame,
        targets,
        overridden.selection,
        epochs=10,
        mass="conserve",
        l2_lambda=0.01,
        l2_anchor="design",
        l2_basis=L2_BASIS_CHI_SQUARE,
        mass_parametrization=MASS_PARAMETRIZATION_SOFTMAX,
    )
    assert direct.refit.options["l2_penalty"] == (
        "chi_square_explicit_anchor_pre_gate_weight_distance"
    )


def test_result_reports_the_realized_chi_square_distance() -> None:
    frame, targets, design = _problem(3, n=40, k=3)
    result = calibrate(frame, targets, epochs=50, mass="conserve", seed=0)
    assert result.chi_square_distance == pytest.approx(
        chi_square_distance(result.weights, design), rel=1e-12
    )
    assert result.chi_square_distance > 0.0


# --- 7: the defaults are the pre-change optimizer, byte for byte --------------

_ORACLE_MODULE = "optimizer_bda72cb02.py"
_ORACLE_MODULE_SHA256 = (
    "88ce8f77dcbfb48096dd16f4dc4de9de4c39367dc479a36db79b35f6c05ec55b"
)
_ORACLE_FUNCTION_SHA256 = (
    "c5c8923788d33ef50861599e5599028611b62fb4f5a7238f7c09773daaee5155"
)
#: The live helpers the oracle imports. These digests equal the helpers'
#: sources at bda72cb02 (checked with ``git show bda72cb02:.../solve.py`` when
#: the pin was written), so the oracle runs the pre-change closure exactly.
_HELPER_SOURCE_SHA256 = {
    "_apply_constraint": (
        "86281bdf18bdc17acafbbe79e15fb6b4c4102840a1315c3b6bbf52b170d8749c"
    ),
    "_prepare_warm_start_weights": (
        "2e50a07a704ad2ab9d6529a965b7554f7aaba3972ec046a14e5a519e4f0904ed"
    ),
    "_project_to_total": (
        "f220ae141fc9144a4a1eec2c894f49065e88bbc1fd18be837ddec2d96cdd52f0"
    ),
    "_relative_error_loss": (
        "e7bba6d694211827780e2d5a0c60436decbaa1dc9310a7caa62945ac4d50843d"
    ),
}


def _pre_l2_basis_oracle():
    path = _TEST_PATHS.tests / "fixtures/pre_l2_basis" / _ORACLE_MODULE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _ORACLE_MODULE_SHA256
    for name, expected in _HELPER_SOURCE_SHA256.items():
        source = inspect.getsource(getattr(solve_module, name)).rstrip("\n")
        assert hashlib.sha256(source.encode()).hexdigest() == expected, name
    spec = importlib.util.spec_from_file_location("pre_l2_basis_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    source = inspect.getsource(oracle._optimize).rstrip("\n")
    assert hashlib.sha256(source.encode()).hexdigest() == _ORACLE_FUNCTION_SHA256
    return oracle


_DEFAULT_CASES = {
    "free_best_iterate": dict(mass="free"),
    "conserve_capped": dict(mass="conserve", max_weight_ratio=5.0),
    "conserve_l2_initial": dict(mass="conserve", max_weight_ratio=5.0, l2_lambda=0.05),
    "free_l2_uniform": dict(mass="free", l2_lambda=0.05, l2_anchor="uniform"),
    "conserve_l2_explicit": dict(mass="conserve", l2_lambda=0.05, l2_anchor="design"),
    "conserve_l0_gated": dict(mass="conserve", l0_lambda=0.002, l2_lambda=0.01),
    "conserve_warm_start": dict(
        mass="conserve", max_weight_ratio=5.0, l2_lambda=0.02, warm=True
    ),
}


@pytest.mark.parametrize("case_id", sorted(_DEFAULT_CASES))
def test_defaults_reproduce_the_pre_change_optimizer_bytes(case_id: str) -> None:
    """Regression pin: the default path is the bda72cb02 optimizer exactly.

    The oracle is the verbatim pre-change ``_optimize``; both sides get the
    same compiled inputs and seed, and every returned array must match byte
    for byte, with the defaults implicit and with them spelled out.
    """
    oracle = _pre_l2_basis_oracle()
    options = dict(_DEFAULT_CASES[case_id])
    warm = options.pop("warm", False)
    frame, targets, design = _problem(7, n=90, k=4)
    rng = np.random.default_rng(11)
    warm_start = design * rng.uniform(0.5, 2.0, len(design)) if warm else None
    anchor_weights = design * 1.5 if options.get("l2_anchor") == "design" else None
    epochs, learning_rate, seed = 120, 0.03, 3

    problem = build_constraint_matrix(frame, targets, "household")
    torch.manual_seed(seed)
    old = oracle._optimize(
        solve_module._torch_constraint_matrix(problem.matrix),
        torch.tensor(problem.target_vector, dtype=torch.float32),
        None,
        torch.tensor(
            solve_module.default_target_loss_scales(problem.target_vector),
            dtype=torch.float32,
        ),
        solve_module._DEFAULT_TARGET_LOSS_CAP,
        problem.initial_weights.values,
        warm_start_weights=warm_start,
        epochs=epochs,
        learning_rate=learning_rate,
        conserve_mass=options["mass"] == "conserve",
        max_weight_ratio=options.get("max_weight_ratio"),
        l0_lambda=options.get("l0_lambda", 0.0),
        l2_lambda=options.get("l2_lambda", 0.0),
        l2_anchor=options.get("l2_anchor", "initial"),
        l2_anchor_weights=anchor_weights,
        target_records=None,
        init_mean=0.999,
        temperature=0.25,
        return_gate_open_probabilities=True,
    )
    old_weights, old_trajectory, old_gates = old
    common = dict(
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        warm_start_weights=warm_start,
        l2_anchor_weights=anchor_weights,
        **options,
    )
    for spelled in (
        {},
        {
            "l2_basis": L2_BASIS_RECORD,
            "mass_parametrization": MASS_PARAMETRIZATION_PROJECTION,
        },
    ):
        result = calibrate(frame, targets, **common, **spelled)
        assert result.weights.tobytes() == old_weights.tobytes()
        assert result.loss_trajectory.tobytes() == old_trajectory.tobytes()
        if old_gates is None:
            assert result.gate_open_probabilities is None
        else:
            assert result.gate_open_probabilities.tobytes() == old_gates.tobytes()
