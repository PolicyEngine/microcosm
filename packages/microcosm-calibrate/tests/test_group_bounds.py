"""Invented strict grouped-cap numerical and prepatch byte contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from microcosm.calibrate import (
    GroupedUpperBounds,
    Target,
    TargetSet,
    TargetSnapshotCadence,
    TargetSnapshotObserver,
    calibrate,
    solve,
)
from microcosm.calibrate.initialization import GateInitialization
from microcosm.calibrate.target_snapshots import (
    EVERY_EPOCH,
    ITERATE_CURRENT,
    ITERATE_SELECTED,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _frame(weights=(40.0, 40.0, 40.0, 40.0), ids=(0, 1, 2, 3)):
    n = len(weights)
    return Frame(
        {
            "person": pd.DataFrame({"person_id": range(n), "person_household_id": ids}),
            "household": pd.DataFrame(
                {
                    "household_id": ids,
                    "native": np.array([1.0, 0.0] * (n // 2)),
                    "detail": np.array([0.0, 1.0] * (n // 2)),
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.asarray(weights), kind=WeightKind.IMPORTANCE)},
    )


def _targets():
    return TargetSet(
        (
            Target(name="native", entity="household", value=380.0, measure="native"),
            Target(name="detail", entity="household", value=20.0, measure="detail"),
        )
    )


def _groups(bounds=(200.0, 200.0), ids=(0, 1, 2, 3)):
    return GroupedUpperBounds(ids, np.array([0, 0, 1, 1]), np.array(bounds))


def _internal(initial, groups, **overrides):
    options = dict(
        epochs=2,
        learning_rate=0.02,
        conserve_mass=False,
        max_weight_ratio=None,
        l0_lambda=0.0,
        l2_lambda=0.0,
        target_records=None,
        init_mean=0.999,
        temperature=0.25,
        grouped_upper_bounds=groups,
    )
    options.update(overrides)
    return solve._optimize(
        torch.ones((1, len(initial)), dtype=torch.float32),
        torch.tensor([200.0], dtype=torch.float32),
        None,
        torch.tensor([200.0], dtype=torch.float32),
        10.0,
        np.asarray(initial, dtype=np.float64),
        **options,
    )


def test_two_reviewed_examples_and_strict_binding_counts():
    design = GroupedUpperBounds(
        (0, 1), np.array([0, 0]), np.array([min(2 * 100.0, 3 * 80.0)])
    )
    design.check(np.array([190.0, 10.0]))
    with pytest.raises(ValueError, match="exceed"):
        design.check(np.array([120.0, 120.0]))
    incoming = GroupedUpperBounds(
        (0, 1), np.array([0, 0]), np.array([min(4 * 100.0, 2 * 80.0)])
    )
    incoming.check(np.array([150.0, 10.0]))
    with pytest.raises(ValueError, match="exceed"):
        incoming.check(np.array([170.0, 0.0]))
    assert incoming.diagnostics(np.array([150.0, 10.0]))["binding_group_count"] == 1
    assert (
        incoming.diagnostics(np.array([np.nextafter(150.0, 0.0), 10.0]))[
            "binding_group_count"
        ]
        == 0
    )


def test_groups_are_immutable_and_complete_even_with_zero_rows():
    ids = [2, 1]
    indices = np.array([0, 0])
    bounds = np.array([0.0])
    groups = GroupedUpperBounds(ids, indices, bounds)
    indices[0] = 9
    bounds[0] = 9
    ids[0] = 9
    assert groups.household_ids == (2, 1)
    groups.check(np.zeros(2))
    for array in (groups.group_indices, groups.absolute_bounds):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(ValueError, match="exceed"):
        groups.check(np.array([0.0, np.nextafter(0.0, 1.0)]))
    with pytest.raises(ValueError, match="align"):
        groups.check(np.zeros(1))


@pytest.mark.parametrize(
    "count", [-1, 3, True, False, np.bool_(True), 1.0, 0.5, np.nan, np.inf, "1", None]
)
def test_diagnostic_corrected_count_refuses_invalid_domain(count):
    with pytest.raises(ValueError, match="last corrected group count"):
        _groups().diagnostics(np.full(4, 40.0), count)


@pytest.mark.parametrize("count", [0, 1, 2, np.int64(2)])
def test_diagnostic_corrected_count_retains_integer_domain(count):
    recorded = _groups().diagnostics(np.full(4, 40.0), count)
    assert recorded["last_corrected_group_count"] == int(count)
    assert type(recorded["last_corrected_group_count"]) is int


@pytest.mark.parametrize(
    "ids,indices,bounds",
    [
        ((1, 1), [0, 0], [1.0]),
        ((1, 2), [0], [1.0]),
        ((1, 2), [0.0, 0.0], [1.0]),
        ((1, 2), [0, 2], [1.0, 1.0]),
        ((1, 2), [0, 0], [1.0, 1.0]),
        ((1, 2), [0, 0], [np.nan]),
        ((1, 2), [0, 0], [np.inf]),
        ((1, 2), [0, 0], [-1.0]),
    ],
)
def test_invalid_maps_refuse(ids, indices, bounds):
    with pytest.raises(ValueError):
        GroupedUpperBounds(ids, np.array(indices), np.array(bounds))


@pytest.mark.parametrize(
    "values", [[np.inf, 1.0], [np.nan, 1.0], [-1.0, 1.0], [1e308, 1e308]]
)
def test_nonfinite_negative_or_overflowed_sums_refuse(values):
    groups = GroupedUpperBounds((0, 1), np.array([0, 0]), np.array([1e308]))
    with pytest.raises(ValueError):
        groups.check(np.array(values))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_weight_ratio": 2.0},
        {"method": "prox"},
        {"mass": "conserve"},
        {"l0_lambda": 0.001},
        {"target_records": 2},
        {"l1_lambda": 0.1},
        {"target_records": 2, "budget_basis": "open_probability_mass"},
        {
            "target_records": 2,
            "budget_basis": "open_probability_mass",
            "feasible_draw_pi_hi": 0.9,
        },
        {
            "l0_lambda": 0.001,
            "gate_initialization": GateInitialization(
                np.full(4, 0.5), np.array([True, False, False, False])
            ),
        },
    ],
)
def test_public_unsupported_modes_refuse_before_optimizer(monkeypatch, kwargs):
    calls = []

    def forbidden(*args, **kw):
        calls.append(1)
        raise AssertionError("optimizer constructed")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    with pytest.raises(ValueError, match="grouped upper bounds"):
        calibrate(_frame(), _targets(), grouped_upper_bounds=_groups(), **kwargs)
    assert not calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_weight_ratio": 2.0},
        {"conserve_mass": True},
        {"l0_lambda": 0.001},
        {"target_records": 2},
        {
            "gate_initialization": GateInitialization(
                np.full(4, 0.5), np.array([True, False, False, False])
            ),
        },
    ],
)
def test_internal_unsupported_modes_refuse_before_optimizer(monkeypatch, kwargs):
    calls = []
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="grouped upper bounds"):
        _internal([40.0, 40.0, 40.0, 40.0], _groups(), **kwargs)
    assert not calls


@pytest.mark.parametrize(
    "initial,bounds,warm",
    [
        ([120.0, 120.0, 40.0, 40.0], (200.0, 200.0), None),
        ([40.0, 40.0, 40.0, 40.0], (0.0, 200.0), None),
        ([0.0, 40.0, 40.0, 40.0], (200.0, 200.0), None),
        ([40.0, 40.0, 40.0, 40.0], (200.0, 200.0), [120.0, 120.0, 40.0, 40.0]),
        ([40.0, 40.0, 40.0, 40.0], (200.0, 200.0), [0.0, 40.0, 40.0, 40.0]),
        ([40.0, 40.0, 40.0, 40.0], (200.0, 200.0), [np.nan, 40.0, 40.0, 40.0]),
    ],
)
def test_initial_and_effective_warm_refuse_before_optimizer(
    monkeypatch, initial, bounds, warm
):
    calls = []
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError):
        _internal(
            initial,
            _groups(bounds),
            warm_start_weights=None if warm is None else np.array(warm),
        )
    assert calls == []


def test_public_overbudget_warm_checks_effective_prepared_vector(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="exceed"):
        calibrate(
            _frame(),
            _targets(),
            epochs=2,
            grouped_upper_bounds=_groups(),
            warm_start_weights=np.array([150.0, 100.0, 40.0, 40.0]),
        )
    assert calls == []
    # Test the actual preparation boundary too, rather than only raw warm input.
    monkeypatch.setattr(
        solve,
        "_prepare_warm_start_weights",
        lambda *a, **k: np.array([150.0, 100.0, 40.0, 40.0]),
    )
    with pytest.raises(ValueError, match="exceed"):
        calibrate(_frame(), _targets(), epochs=2, grouped_upper_bounds=_groups())
    assert calls == []


@pytest.mark.parametrize("sparse", [False, True])
def test_real_adam_redistributes_and_every_accepted_update_is_observed(
    monkeypatch, sparse
):
    groups = _groups()
    observed = []
    layouts = []
    apply = solve._apply_constraint

    def checked_apply(matrix, weights):
        assert matrix.dtype == weights.dtype == torch.float32
        assert weights.grad_fn is not None
        layouts.append(matrix.layout)
        return apply(matrix, weights)

    monkeypatch.setattr(solve, "_apply_constraint", checked_apply)
    if sparse:
        monkeypatch.setattr(solve, "_SPARSE_MIN_CELLS", 1)
        monkeypatch.setattr(solve, "_SPARSE_DENSITY_CUTOFF", 1.0)

    def observer(payload):
        weights = payload["weights"]
        assert weights.dtype == np.float64
        assert np.array_equal(
            groups.check(weights, positive=True), payload["group_totals"]
        )
        assert payload["household_ids"] == groups.household_ids
        observed.append(payload)

    result = calibrate(
        _frame(),
        _targets(),
        epochs=60,
        learning_rate=0.08,
        grouped_upper_bounds=groups,
        _post_projection_observer=observer,
    )
    assert len(observed) == 61
    assert [x["epoch"] for x in observed] == list(range(61))
    assert result.weights[0] > 120.0  # exceeds removed role-wise rI * 40 cap
    assert result.weights[1] < 40.0
    assert result.weights.tobytes() == observed[-1]["weights"].tobytes()
    assert (
        result.frame.resolve_weights("household").values.tobytes()
        == result.weights.tobytes()
    )
    assert all(x == (torch.sparse_csr if sparse else torch.strided) for x in layouts)
    assert result.options["grouped_upper_bounds"] == groups.diagnostics(
        result.weights, observed[-1]["corrected_group_count"]
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_float32_matmul_remains_differentiable_to_float64_log_weights(sparse):
    matrix = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    if sparse:
        matrix = matrix.to_sparse_csr()
    log_w = torch.tensor(np.log([40.0, 40.0]), dtype=torch.float64, requires_grad=True)
    estimate = solve._apply_constraint(matrix, torch.exp(log_w).to(torch.float32))
    assert estimate.dtype == torch.float32
    estimate.sum().backward()
    assert log_w.grad.dtype == torch.float64
    assert torch.isfinite(log_w.grad).all() and torch.all(log_w.grad != 0)
    assert matrix.dtype == torch.float32


def test_no_update_final_is_effective_initial_bytes_without_reexp():
    initial = np.array([100.0, 120.0, 80.0, 40.0])
    assert np.exp(np.log(initial)).tobytes() != initial.tobytes()
    observed = []
    final, trajectory = _internal(
        initial,
        _groups((220.0, 120.0)),
        epochs=0,
        _post_projection_observer=observed.append,
    )
    assert not len(trajectory)
    assert final.tobytes() == initial.tobytes() == observed[-1]["weights"].tobytes()


def test_projection_is_common_factor_from_original_candidate_and_strict():
    rng = np.random.default_rng(11)
    found_roundoff = False
    for _ in range(1000):
        candidate = rng.uniform(1.0, 200.0, 3)
        bound = float(rng.uniform(1.0, 100.0))
        groups = GroupedUpperBounds(
            (30, 10, 20), np.zeros(3, dtype=int), np.array([bound])
        )
        total = math.fsum(float(candidate[i]) for i in (1, 2, 0))
        if total <= bound:
            continue
        factor = np.float64(bound / total)
        naive = candidate * factor
        naive_total = math.fsum(float(naive[i]) for i in (1, 2, 0))
        if naive_total <= bound:
            continue
        found_roundoff = True
        for _ in range(64):
            factor = np.nextafter(factor, 0.0)
            expected = candidate * factor
            if math.fsum(float(expected[i]) for i in (1, 2, 0)) <= bound:
                break
        accepted, count = groups.project(candidate)
        assert count == 1
        assert accepted.tobytes() == expected.tobytes()
        assert groups.project(accepted)[0].tobytes() == accepted.tobytes()
        assert groups.check(accepted)[0] <= bound
        break
    assert found_roundoff, "invented roundoff boundary must be exercised"


def test_reorder_requires_aligned_ids_and_preserves_by_key_solution():
    groups = _groups()
    changed = GroupedUpperBounds(
        (2, 3, 0, 1), np.array([1, 1, 0, 0]), np.array([200.0, 200.0])
    )
    with pytest.raises(ValueError, match="ordered IDs"):
        calibrate(_frame(), _targets(), epochs=2, grouped_upper_bounds=changed)
    # Frame requires sorted IDs. Its lower numerical seam accepts an explicit
    # order, where simultaneous vector/map alignment must preserve keyed values.
    baseline, _ = _internal([10.0, 20.0, 30.0, 40.0], groups, epochs=3)
    reordered, _ = _internal([30.0, 40.0, 10.0, 20.0], changed, epochs=3)
    assert dict(zip(groups.household_ids, baseline, strict=True)) == dict(
        zip(changed.household_ids, reordered, strict=True)
    )
    assert changed.constraint_digest != groups.constraint_digest


def test_singleton_admission_is_exactly_direct_inequalities():
    design = np.array([100.0, 20.0])
    incoming = np.array([80.0, 10.0])
    bound = np.minimum(2 * design, 3 * incoming)
    groups = GroupedUpperBounds((0, 1), np.array([0, 1]), bound)
    for candidate in (
        np.array([190.0, 10.0]),
        np.array([210.0, 10.0]),
        np.array([100.0, 31.0]),
        np.array([0.0, 0.0]),
    ):
        direct = bool(
            np.all(candidate <= 2 * design) and np.all(candidate <= 3 * incoming)
        )
        if direct:
            groups.check(candidate)
        else:
            with pytest.raises(ValueError):
                groups.check(candidate)


def test_saved_prepatch_legacy_exact_bytes():
    from microcosm.graph import platform_fingerprint

    path = Path(__file__).parent / "fixtures/group_bounds/legacy_prepatch.json"
    frozen = json.loads(path.read_text())
    running = {
        "platform_fingerprint": platform_fingerprint(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "backend": "Adam",
        "parameter_dtype": "float32",
        "matrix_dtype": "float32",
        "output_dtype": "float64",
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "python_binary_sha256": hashlib.sha256(
            Path(sys.executable).read_bytes()
        ).hexdigest(),
    }
    if running != frozen["settings"]:
        pytest.skip(
            "Matching configuration required; recorded="
            + json.dumps(frozen["settings"], sort_keys=True)
            + " running="
            + json.dumps(running, sort_keys=True)
        )
    inputs = frozen["inputs"]
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(4), "person_household_id": range(4)}
            ),
            "household": pd.DataFrame(
                {"household_id": range(4), "adjudicated_measure": inputs["measure"]}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                values=np.array(inputs["weights"]), kind=WeightKind.DESIGN
            )
        },
    )
    targets = TargetSet(
        (
            Target(
                name="adjudicated",
                entity="household",
                value=inputs["target"],
                measure="adjudicated_measure",
            ),
        )
    )
    result = calibrate(frame, targets, **frozen["options"])
    for label, values in (
        ("weights", result.weights),
        ("trajectory", result.loss_trajectory),
    ):
        data = np.asarray(values, dtype="<f8").tobytes()
        assert data.hex() == frozen[label]["bytes_hex"]
        assert hashlib.sha256(data).hexdigest() == frozen[label]["sha256"]


def test_final_noop_projection_is_mandatory_and_byte_preserving(monkeypatch):
    groups = _groups((220.0, 120.0))
    project = GroupedUpperBounds.project
    calls = []

    def corrupt_final(self, weights, **kwargs):
        calls.append(1)
        accepted, count = project(self, weights, **kwargs)
        accepted[0] = np.nextafter(accepted[0], 0.0)
        return accepted, count

    monkeypatch.setattr(GroupedUpperBounds, "project", corrupt_final)
    with pytest.raises(ValueError, match="preserve accepted weight bytes"):
        _internal([100.0, 120.0, 80.0, 40.0], groups, epochs=0)
    assert len(calls) == 1


def test_observer_payload_cannot_mutate_the_accepted_state():
    expected, _ = _internal(
        [100.0, 120.0, 80.0, 40.0], _groups((220.0, 120.0)), epochs=2
    )

    def alter(payload):
        payload["weights"][:] = 0.0
        payload["group_indices"][:] = 0
        payload["absolute_bounds"][:] = 0.0
        payload["group_totals"][:] = 0.0

    actual, _ = _internal(
        [100.0, 120.0, 80.0, 40.0],
        _groups((220.0, 120.0)),
        epochs=2,
        _post_projection_observer=alter,
    )
    assert actual.tobytes() == expected.tobytes()


def _grouped_run(**overrides):
    options = dict(epochs=6, learning_rate=0.05, grouped_upper_bounds=_groups())
    options.update(overrides)
    return calibrate(_frame(), _targets(), **options)


def test_public_grouped_apg_reaches_the_adam_path_and_warns(monkeypatch):
    """The deprecated alias is normalized before the grouped mode check."""
    entered = []
    grouped = solve._optimize_grouped

    def recorded(*args, **kwargs):
        entered.append(True)
        return grouped(*args, **kwargs)

    monkeypatch.setattr(solve, "_optimize_grouped", recorded)
    with pytest.warns(DeprecationWarning, match="apg"):
        aliased = _grouped_run(method="apg")
    plain = _grouped_run(method="adam")
    assert entered == [True, True]
    assert aliased.options["method"] == "adam" == plain.options["method"]
    assert aliased.weights.tobytes() == plain.weights.tobytes()
    assert (
        aliased.options["grouped_upper_bounds"] == plain.options["grouped_upper_bounds"]
    )


def test_ungrouped_apg_alias_is_unchanged_by_the_reordering():
    """Control: the ungrouped alias behaviour the move must not disturb."""
    with pytest.warns(DeprecationWarning, match="apg"):
        aliased = calibrate(_frame(), _targets(), epochs=6, method="apg")
    plain = calibrate(_frame(), _targets(), epochs=6, method="adam")
    assert aliased.options["method"] == "adam"
    assert aliased.weights.tobytes() == plain.weights.tobytes()


@pytest.mark.parametrize("mass", ["conserved", "Free", "", "none", None, 0, True])
def test_public_grouped_invalid_mass_is_reported_as_mass(monkeypatch, mass):
    """An unknown mass must not be silently read as a conserve-mass request."""
    calls = []
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="mass must be") as caught:
        _grouped_run(epochs=2, mass=mass)
    assert "grouped upper bounds" not in str(caught.value)
    assert not calls


@pytest.mark.parametrize("method", ["sgd", "APG", "", "adam "])
def test_public_grouped_unknown_method_is_reported_as_method(monkeypatch, method):
    """An unknown method is a method error, not a grouped-mode refusal."""
    calls = []
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="Unknown method") as caught:
        _grouped_run(epochs=2, method=method)
    assert "grouped upper bounds" not in str(caught.value)
    assert not calls


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"method": "prox"}, "grouped upper bounds"),
        ({"mass": "conserve"}, "grouped upper bounds"),
        ({"method": "apg", "max_weight_ratio": 2.0}, "grouped upper bounds"),
        ({"method": "apg", "target_records": 2}, "grouped upper bounds"),
        ({"method": "apg", "mass": "conserve"}, "grouped upper bounds"),
    ],
)
def test_grouped_mode_constraints_survive_the_reordering(monkeypatch, kwargs, expected):
    """Valid-valued but unsupported grouped modes still refuse as before."""
    calls = []
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match=expected):
        _grouped_run(epochs=2, **kwargs)
    assert not calls


def _snapshot_observer(seen, **kwargs):
    return TargetSnapshotObserver(
        sink=seen.append,
        run_id="grouped-bounds",
        cadence=TargetSnapshotCadence(every=EVERY_EPOCH),
        **kwargs,
    )


def test_zero_epochs_through_the_internal_grouped_path_adds_no_evaluation(monkeypatch):
    """A bound observer on a no-update run neither emits nor evaluates.

    ``calibrate`` refuses zero epochs publicly, so the internal seam is the
    only way to reach this state; it must stay a pure admission path.
    """
    applications = []
    apply_constraint = solve._apply_constraint

    def counted(matrix, weights):
        applications.append(matrix.layout)
        return apply_constraint(matrix, weights)

    monkeypatch.setattr(solve, "_apply_constraint", counted)
    initial = np.array([100.0, 120.0, 80.0, 40.0])
    seen = []
    bound = _snapshot_observer(seen).bind(
        names=("population@2024",), targets=np.array([200.0])
    )
    final, trajectory = _internal(
        initial, _groups((220.0, 120.0)), epochs=0, snapshots=bound
    )
    assert not len(trajectory)
    assert final.tobytes() == initial.tobytes()
    assert applications == []
    assert seen == []


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "csr"])
def test_a_binding_bound_that_actually_corrects_is_unchanged_by_snapshots(
    monkeypatch, sparse
):
    """Projection still corrects, identically, with the observer attached.

    Caps of 81 sit just above the initial per-group total of 80, so the very
    first Adam step overshoots them and the projection has to pull each group
    back on every epoch — the case where an observer could plausibly disturb
    the accepted state.
    """
    if sparse:
        monkeypatch.setattr(solve, "_SPARSE_MIN_CELLS", 1)
        monkeypatch.setattr(solve, "_SPARSE_DENSITY_CUTOFF", 1.0)

    def run(**extra):
        observations = []
        result = _grouped_run(
            grouped_upper_bounds=_groups((81.0, 81.0)),
            _post_projection_observer=observations.append,
            **extra,
        )
        return result, observations

    off, observed_off = run()
    seen = []
    on, observed_on = run(target_snapshots=_snapshot_observer(seen))

    # The bound really binds on this fixture, or the test proves nothing.
    assert sum(item["corrected_group_count"] for item in observed_off) > 0

    assert on.weights.tobytes() == off.weights.tobytes()
    assert on.loss_trajectory.tobytes() == off.loss_trajectory.tobytes()
    assert on.options == off.options
    assert on.options["grouped_upper_bounds"] == off.options["grouped_upper_bounds"]
    for payload_on, payload_off in zip(observed_on, observed_off, strict=True):
        assert payload_on["weights"].tobytes() == payload_off["weights"].tobytes()
        assert (
            payload_on["group_totals"].tobytes()
            == payload_off["group_totals"].tobytes()
        )
        assert (
            payload_on["corrected_group_count"] == payload_off["corrected_group_count"]
        )

    in_loop = [item for item in seen if item["iterate"] != ITERATE_SELECTED]
    assert [item["epoch"] for item in in_loop] == list(range(6))
    assert all(item["iterate"] == ITERATE_CURRENT for item in in_loop)
    assert all(item["precision"] == "float32" for item in in_loop)
    assert all(
        item["selection"]
        == {
            "rule": "closing_state",
            "constraint_mode": "grouped_upper_bounds",
            "grouped_preserve_zeros": False,
        }
        for item in seen
    )
    selected = [item for item in seen if item["iterate"] == ITERATE_SELECTED]
    assert len(selected) == 1
    assert selected[0]["epoch"] == 6 and selected[0]["precision"] == "float64"
    assert [row["estimate"] for row in selected[0]["targets"]] == [
        diagnostic.final_estimate for diagnostic in on.diagnostics
    ]
