"""Invented full-population contracts for opt-in fixed-zero grouped fitting."""

from __future__ import annotations

import hashlib

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
from microcosm.calibrate.matrix import CalibrationProblem
from microcosm.calibrate.target_snapshots import (
    EVERY_EPOCH,
    ITERATE_CURRENT,
    ITERATE_SELECTED,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _frame(weights=(0.0, 0.0, 40.0, 40.0, 0.0, 20.0)):
    return Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(6), "person_household_id": range(6)}
            ),
            "household": pd.DataFrame(
                {
                    "household_id": range(6),
                    "native": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    "detail": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.asarray(weights), kind=WeightKind.IMPORTANCE)},
    )


def _groups():
    # A zero-only origin, a positive clone pair and a mixed-support group.
    return GroupedUpperBounds(
        tuple(range(6)), np.array([0, 0, 1, 1, 2, 2]), np.array([0.0, 100.0, 30.0])
    )


def _targets():
    return TargetSet(
        (
            Target(name="native", entity="household", value=150.0, measure="native"),
            Target(name="detail", entity="household", value=10.0, measure="detail"),
        )
    )


def _fit(**kwargs):
    options = dict(
        epochs=20,
        learning_rate=0.08,
        grouped_upper_bounds=_groups(),
        grouped_preserve_zeros=True,
    )
    options.update(kwargs)
    return calibrate(_frame(), _targets(), **options)


def _observer(seen, **kwargs):
    """The recording sink every grouped snapshot case below shares."""
    return TargetSnapshotObserver(
        sink=seen.append,
        run_id="grouped-run",
        cadence=TargetSnapshotCadence(every=EVERY_EPOCH),
        **kwargs,
    )


def _count_evaluations(monkeypatch):
    """Count the real forward and closing evaluations without changing them.

    Returns the counter, the list of float32 estimate vectors the loss was
    actually computed from (in epoch order, so a test can prove a snapshot
    carries that exact tensor rather than a recomputation), and the matrix
    layouts those evaluations actually ran on, so a case labelled ``csr``
    cannot silently take the dense path.
    """
    counts = {"apply": 0, "estimates": 0}
    observed = []
    layouts = []
    apply_constraint = solve._apply_constraint
    problem_estimates = CalibrationProblem.estimates

    def apply(matrix, weights):
        counts["apply"] += 1
        layouts.append(matrix.layout)
        value = apply_constraint(matrix, weights)
        observed.append(value.detach().cpu().numpy().copy())
        return value

    def estimates(self, weights):
        counts["estimates"] += 1
        return problem_estimates(self, weights)

    monkeypatch.setattr(solve, "_apply_constraint", apply)
    monkeypatch.setattr(CalibrationProblem, "estimates", estimates)
    return counts, observed, layouts


def _rng_state():
    """The whole legacy numpy state, not just its key.

    Dropping ``pos``/``has_gauss``/``cached_gaussian`` would let up to 623
    draws hide inside one refill block.
    """
    key, values, pos, has_gauss, cached = np.random.get_state()
    return (
        torch.random.get_rng_state().clone(),
        (key, values.copy(), pos, has_gauss, cached),
    )


def _seeded():
    torch.manual_seed(0)
    np.random.seed(0)


def _positive_groups():
    """The same ordered map, with a bound group zero can meet when positive.

    ``_groups`` caps the origin group at zero, which is exactly what makes it
    a fixed-zero fixture; the default grouped contract needs every row strictly
    positive, so the positive arm of the cross product raises that one bound and
    leaves the map, the ordering and the other two bounds alone.
    """
    return GroupedUpperBounds(
        tuple(range(6)), np.array([0, 0, 1, 1, 2, 2]), np.array([40.0, 100.0, 30.0])
    )


def _binding_groups():
    """``_groups``' map with caps the very first Adam step overshoots.

    ``_groups`` and ``_positive_groups`` leave every non-degenerate group
    strictly interior for the whole run, so on their own they prove observer
    neutrality only where ``project`` never corrects. These caps sit just above
    the initial group totals of 0, 80 and 20, so the projection pulls a group
    back on most epochs -- including with the shared warm start and with
    nonzero L2.
    """
    return GroupedUpperBounds(
        tuple(range(6)), np.array([0, 0, 1, 1, 2, 2]), np.array([0.0, 85.0, 25.0])
    )


def _positive_binding_groups():
    """The binding caps for the strictly positive arm's initial totals."""
    return GroupedUpperBounds(
        tuple(range(6)), np.array([0, 0, 1, 1, 2, 2]), np.array([21.0, 85.0, 26.0])
    )


#: Feasible under _groups(): group 0 stays at its zero bound, group 1 totals 80
#: against 100 and group 2 totals 25 against 30, with the fixed zeros intact.
_WARM = np.array([0.0, 0.0, 30.0, 50.0, 0.0, 25.0])

_VARIANTS = [
    pytest.param({}, id="plain"),
    pytest.param({"warm_start_weights": _WARM}, id="warm-start"),
    pytest.param({"l2_lambda": 0.01}, id="l2"),
]


@pytest.mark.parametrize("sparse", [False, True])
def test_full_population_survives_positive_only_log_parameters(monkeypatch, sparse):
    if sparse:
        monkeypatch.setattr(solve, "_SPARSE_MIN_CELLS", 1)
        monkeypatch.setattr(solve, "_SPARSE_DENSITY_CUTOFF", 1.0)
    original_adam = torch.optim.Adam
    parameter_sizes = []

    def adam(parameters, **kwargs):
        parameters = list(parameters)
        parameter_sizes.extend(x.numel() for x in parameters)
        return original_adam(parameters, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", adam)
    original_apply = solve._apply_constraint

    def apply(matrix, weights):
        assert weights.shape == (6,)
        assert weights.dtype == torch.float32 and weights.grad_fn is not None
        assert torch.equal(weights[[0, 1, 4]], torch.zeros(3))
        assert matrix.layout == (torch.sparse_csr if sparse else torch.strided)
        return original_apply(matrix, weights)

    monkeypatch.setattr(solve, "_apply_constraint", apply)
    observations = []
    result = _fit(_post_projection_observer=observations.append)
    assert parameter_sizes == [3]
    assert len(observations) == 21
    original = _frame()
    for entity in ("household", "person"):
        pd.testing.assert_frame_equal(
            result.frame.table(entity), original.table(entity)
        )
    for observation in observations:
        assert observation["weights"].shape == (6,)
        assert observation["household_ids"] == tuple(range(6))
        assert observation["weights"][[0, 1, 4]].tobytes() == np.zeros(3).tobytes()
        np.testing.assert_array_equal(
            _groups().check(observation["weights"]), observation["group_totals"]
        )
    assert result.weights[2] > 40.0 and result.weights[3] < 40.0
    assert result.weights.tobytes() == observations[-1]["weights"].tobytes()
    assert (
        result.frame.resolve_weights("household").values.tobytes()
        == result.weights.tobytes()
    )
    assert (
        result.initial_weights.tobytes()
        == original.resolve_weights("household").values.tobytes()
    )
    assert result.options["grouped_preserve_zeros"] == {
        "enabled": True,
        "fixed_zero_count": 3,
        "ordered_zero_mask_sha256": hashlib.sha256(
            bytes([1, 1, 0, 0, 1, 0])
        ).hexdigest(),
    }


@pytest.mark.parametrize(
    "warm",
    [
        [1.0, 0.0, 40.0, 40.0, 0.0, 20.0],
        [0.0, 0.0, 0.0, 40.0, 0.0, 20.0],
        [0.0, 0.0, 60.0, 60.0, 0.0, 20.0],
        [0.0, 0.0, 40.0, 40.0, -1.0, 20.0],
        [0.0, 0.0, float("nan"), 40.0, 0.0, 20.0],
        [0.0, 0.0, 40.0, 40.0, 0.0],
    ],
)
def test_invalid_warm_support_refuses_before_optimizer(monkeypatch, warm):
    calls = []
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError):
        _fit(warm_start_weights=np.array(warm))
    assert calls == []


def test_valid_warm_starts_keep_original_support_and_initial_diagnostics():
    first = _fit()
    seen = []
    second = _fit(
        epochs=2,
        warm_start_weights=first.weights,
        _post_projection_observer=seen.append,
    )
    assert seen[0]["weights"].tobytes() == first.weights.tobytes()
    assert (
        second.initial_weights.tobytes()
        == _frame().resolve_weights("household").values.tobytes()
    )
    assert second.weights[[0, 1, 4]].tobytes() == np.zeros(3).tobytes()


@pytest.mark.parametrize("value", [1, "yes", None, np.bool_(True)])
def test_nonboolean_zero_option_refuses(value):
    with pytest.raises(ValueError, match="boolean"):
        _fit(grouped_preserve_zeros=value)


def test_zero_option_requires_grouping_and_default_keeps_positive_contract():
    with pytest.raises(ValueError, match="requires grouped"):
        _fit(grouped_upper_bounds=None)
    with pytest.raises(ValueError, match="strictly positive"):
        _fit(grouped_preserve_zeros=False)


@pytest.mark.parametrize("anchor", ["initial", "uniform", "explicit"])
def test_l2_penalty_is_finite_with_fixed_zero_support(anchor):
    options = {"l2_lambda": 0.01, "l2_anchor": anchor}
    if anchor == "explicit":
        options["l2_anchor_weights"] = np.array([1.0, 1.0, 40.0, 40.0, 1.0, 20.0])
    result = _fit(**options)
    assert (
        np.isfinite(result.loss_trajectory).all() and np.isfinite(result.weights).all()
    )
    assert result.weights[[0, 1, 4]].tobytes() == np.zeros(3).tobytes()


def test_all_zero_numeric_support_refuses_before_optimizer(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="positive initial"):
        solve._optimize(
            torch.ones((1, 2), dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
            None,
            torch.ones(1, dtype=torch.float32),
            10.0,
            np.zeros(2),
            epochs=2,
            learning_rate=0.02,
            conserve_mass=False,
            max_weight_ratio=None,
            l0_lambda=0.0,
            l2_lambda=0.0,
            target_records=None,
            init_mean=0.999,
            temperature=0.25,
            grouped_upper_bounds=GroupedUpperBounds(
                (0, 1), np.array([0, 0]), np.array([0.0])
            ),
            grouped_preserve_zeros=True,
        )
    assert calls == []


def test_observer_cannot_change_fixed_mask_or_accepted_weights():
    baseline = _fit()

    def corrupt(payload):
        payload["weights"][:] = 1.0
        payload["group_indices"][:] = 0
        payload["absolute_bounds"][:] = 99.0
        payload["group_totals"][:] = 99.0

    changed = _fit(_post_projection_observer=corrupt)
    assert changed.weights.tobytes() == baseline.weights.tobytes()


def test_materialized_zero_activation_refuses(monkeypatch):
    apply = solve._apply_weights

    def corrupt(*args, **kwargs):
        result = apply(*args, **kwargs)
        current = result.resolve_weights("household")
        values = current.values.copy()
        # Within a positive group bound, so group totals alone cannot catch this.
        values[4] = np.nextafter(0.0, 1.0)
        return result.with_weights(
            "household", current.with_values(values, kind=current.kind), mass="conserve"
        )

    monkeypatch.setattr(solve, "_apply_weights", corrupt)
    with pytest.raises(ValueError, match="zero support"):
        _fit()


def test_aligned_reordering_required_with_zero_rows():
    reordered = GroupedUpperBounds(
        (1, 0, 2, 3, 4, 5), np.array([0, 0, 1, 1, 2, 2]), np.array([0.0, 100.0, 30.0])
    )
    with pytest.raises(ValueError, match="ordered IDs"):
        _fit(grouped_upper_bounds=reordered)


def test_retained_zero_ratio_is_finite_and_activation_is_infinite():
    result = _fit()
    active = result.initial_weights > 0
    expected = float((result.weights[active] / result.initial_weights[active]).max())
    assert result.realized_max_weight_ratio == expected
    # A diagnostic must not hide invalid positive mass on an initial zero.
    from dataclasses import replace

    changed = result.weights.copy()
    changed[0] = 1.0
    assert replace(result, weights=changed).realized_max_weight_ratio == float("inf")


def test_returned_id_swap_refuses_even_when_weight_bytes_match(monkeypatch):
    apply = solve._apply_weights

    def corrupt(*args, **kwargs):
        result = apply(*args, **kwargs)
        result.table("household").loc[[0, 2], "household_id"] = [2, 0]
        return result

    monkeypatch.setattr(solve, "_apply_weights", corrupt)
    with pytest.raises(ValueError, match="ordered IDs"):
        _fit()


def test_late_options_identity_mutation_refuses(monkeypatch):
    apply = solve._apply_weights
    diagnostics = GroupedUpperBounds.diagnostics
    frames = []

    def capture(*args, **kwargs):
        result = apply(*args, **kwargs)
        frames.append(result)
        return result

    def corrupt(self, weights, last_corrected_count=0):
        result = diagnostics(self, weights, last_corrected_count)
        frames[-1].table("household").loc[[0, 2], "household_id"] = [2, 0]
        return result

    monkeypatch.setattr(solve, "_apply_weights", capture)
    monkeypatch.setattr(GroupedUpperBounds, "diagnostics", corrupt)
    with pytest.raises(ValueError, match="ordered IDs"):
        _fit()


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "csr"])
@pytest.mark.parametrize("preserve_zeros", [False, True], ids=["positive", "fixed"])
@pytest.mark.parametrize("bound", [False, True], ids=["drifts-into-cap", "at-cap"])
@pytest.mark.parametrize("extra", _VARIANTS)
def test_snapshots_leave_the_grouped_run_bit_identical(
    monkeypatch, sparse, preserve_zeros, bound, extra
):
    """Enabling the observer changes nothing a grouped run produces or draws.

    The cross product is dense/CSR times positive/fixed-zero support times two
    cap tightnesses times a plain, a warm-started and an L2 run, over the shared
    mixed and zero-only groups. Both arms start strictly interior and reach a cap
    during the run, which the test asserts rather than assuming, so neutrality is
    never proved only on a run whose projection is a no-op; the ``at-cap`` arm
    puts the caps close enough to the initial totals to change how early and how
    often the projection has to correct.
    """
    if sparse:
        monkeypatch.setattr(solve, "_SPARSE_MIN_CELLS", 1)
        monkeypatch.setattr(solve, "_SPARSE_DENSITY_CUTOFF", 1.0)
    options = dict(extra)
    if preserve_zeros:
        weights = (0.0, 0.0, 40.0, 40.0, 0.0, 20.0)
        groups = _binding_groups() if bound else _groups()
    else:
        # The default grouped contract is strictly positive, so lift the zeros.
        weights = (10.0, 10.0, 40.0, 40.0, 5.0, 20.0)
        groups = _positive_binding_groups() if bound else _positive_groups()
        if "warm_start_weights" in options:
            options["warm_start_weights"] = np.array([5.0, 5.0, 30.0, 50.0, 5.0, 20.0])

    def run(observer):
        _seeded()
        with monkeypatch.context() as scoped:
            counts, observed, layouts = _count_evaluations(scoped)
            proof = []
            result = calibrate(
                _frame(weights),
                _targets(),
                epochs=20,
                learning_rate=0.08,
                grouped_upper_bounds=groups,
                grouped_preserve_zeros=preserve_zeros,
                _post_projection_observer=proof.append,
                **({} if observer is None else {"target_snapshots": observer}),
                **options,
            )
        return result, counts, observed, layouts, proof, _rng_state()

    off, counts_off, observed_off, layouts_off, proof_off, rng_off = run(None)
    seen = []
    on, counts_on, observed_on, layouts_on, proof_on, rng_on = run(_observer(seen))

    # The case really took the layout its id claims.
    expected_layout = torch.sparse_csr if sparse else torch.strided
    assert layouts_off and layouts_on
    assert set(layouts_off) == set(layouts_on) == {expected_layout}

    # A bound really binds here, so observer neutrality is not being proved
    # only on a run whose projection is a no-op.
    assert sum(item["corrected_group_count"] for item in proof_off) > 0

    assert on.weights.tobytes() == off.weights.tobytes()
    assert on.loss_trajectory.tobytes() == off.loss_trajectory.tobytes()
    assert on.options == off.options
    assert float(on.closing_loss) == float(off.closing_loss)
    assert [
        (d.name, d.target, d.initial_estimate, d.final_estimate) for d in on.diagnostics
    ] == [
        (d.name, d.target, d.initial_estimate, d.final_estimate)
        for d in off.diagnostics
    ]
    stored_on = on.frame.resolve_weights("household").values
    assert (
        stored_on.tobytes() == off.frame.resolve_weights("household").values.tobytes()
    )
    assert on.initial_weights.tobytes() == off.initial_weights.tobytes()
    assert (
        tuple(on.frame.table("household")["household_id"])
        == tuple(off.frame.table("household")["household_id"])
        == tuple(range(6))
    )
    if preserve_zeros:
        zeros = np.asarray(weights) == 0
        assert stored_on[zeros].tobytes() == np.zeros(int(zeros.sum())).tobytes()

    # The private proof seam is byte-for-byte the same sequence of payloads.
    assert len(proof_on) == len(proof_off) == 21
    for payload_on, payload_off in zip(proof_on, proof_off, strict=True):
        assert payload_on.keys() == payload_off.keys()
        assert payload_on["epoch"] == payload_off["epoch"]
        assert payload_on["weights"].tobytes() == payload_off["weights"].tobytes()
        assert (
            payload_on["group_totals"].tobytes()
            == payload_off["group_totals"].tobytes()
        )
        assert (
            payload_on["corrected_group_count"] == payload_off["corrected_group_count"]
        )
        assert payload_on["household_ids"] == payload_off["household_ids"]

    # Cadence buys observation, not evaluation, and advances no RNG stream.
    # The RNG halves are weak evidence by construction: grouped Adam draws
    # nothing after calibrate's seeding and the grouped validators refuse every
    # gated mode, so these can only catch a draw introduced by the emission
    # itself, not a reordering of a stream that does not exist.
    assert counts_on == counts_off == {"apply": 20, "estimates": 3}
    assert len(observed_on) == len(observed_off) == 20
    assert torch.equal(rng_on[0], rng_off[0])
    assert rng_on[1][0] == rng_off[1][0]
    assert (rng_on[1][1] == rng_off[1][1]).all()
    assert rng_on[1][2:] == rng_off[1][2:]
    assert len(seen) == 21


@pytest.mark.parametrize("preserve_zeros", [False, True], ids=["positive", "fixed"])
def test_grouped_snapshots_report_a_closing_state_run_with_no_retained_best(
    preserve_zeros,
):
    """Grouped labels say "this solver retains nothing", not "best is unset"."""
    if preserve_zeros:
        weights = (0.0, 0.0, 40.0, 40.0, 0.0, 20.0)
        groups = _groups()
    else:
        weights = (10.0, 10.0, 40.0, 40.0, 5.0, 20.0)
        groups = _positive_groups()
    seen = []
    result = calibrate(
        _frame(weights),
        _targets(),
        epochs=20,
        learning_rate=0.08,
        grouped_upper_bounds=groups,
        grouped_preserve_zeros=preserve_zeros,
        target_snapshots=_observer(seen),
    )
    in_loop = [item for item in seen if item["iterate"] != ITERATE_SELECTED]
    selected = [item for item in seen if item["iterate"] == ITERATE_SELECTED]
    assert len(selected) == 1
    assert [item["epoch"] for item in in_loop] == list(range(1, 21))
    for item in seen:
        assert item["best_retained"] == {
            "available": False,
            "epoch": None,
            "loss": None,
        }
        assert item["epochs"] == 20
        assert item["selection"]["rule"] == "closing_state"
        assert item["selection"]["constraint_mode"] == "grouped_upper_bounds"
        assert item["selection"]["grouped_preserve_zeros"] is preserve_zeros
    for item in in_loop:
        assert item["iterate"] == ITERATE_CURRENT
        assert item["precision"] == "float32"
        assert item["selection"]["epoch_convention"] == (
            solve.GROUPED_IN_LOOP_EPOCH_CONVENTION
        )
    assert selected[0]["precision"] == "float64"
    assert selected[0]["epoch"] == 20
    assert selected[0]["selection"]["epoch_convention"] == (
        solve.GROUPED_CLOSING_EPOCH_CONVENTION
    )
    # Two rows are stamped epoch == epochs, one update apart. iterate and the
    # convention label are the only things that separate them, so both must be
    # present and different.
    assert [item["epoch"] for item in seen].count(20) == 2
    assert (
        solve.GROUPED_IN_LOOP_EPOCH_CONVENTION != solve.GROUPED_CLOSING_EPOCH_CONVENTION
    )
    # The rule label is the option the result itself records, not a second
    # opinion, and the receipt the emitter read really was empty.
    assert result.options["iterate_selection"] == "closing_state"
    assert result.options["iterate_selection_receipt"] == {}


def test_a_bounded_grouped_cadence_emits_first_multiples_and_last():
    """The cadence gate is evaluated in a mode where it can return False.

    Every other grouped case runs at ``EVERY_EPOCH``, where ``emits`` is always
    true and a gate written against the wrong epoch base would still pass.
    """
    seen = []
    calibrate(
        _frame(),
        _targets(),
        epochs=20,
        learning_rate=0.08,
        grouped_upper_bounds=_groups(),
        grouped_preserve_zeros=True,
        target_snapshots=TargetSnapshotObserver(
            sink=seen.append,
            run_id="grouped-run",
            cadence=TargetSnapshotCadence(every=5),
        ),
    )
    in_loop = [item for item in seen if item["iterate"] != ITERATE_SELECTED]
    assert [item["epoch"] for item in in_loop] == [1, 5, 10, 15, 20]
    assert [item["iterate"] for item in seen[-1:]] == [ITERATE_SELECTED]
    assert len(seen) == 6


def test_a_populated_grouped_receipt_cannot_claim_a_retained_best(monkeypatch):
    """Retain-best is read off the solver, not inferred from an empty receipt.

    Grouped ``_optimize`` returns before ``selection_receipt`` is ever touched,
    so every real grouped run leaves it empty and the guard is invisible. This
    forces the state a future grouped retain-best rule would create.
    """
    optimize = solve._optimize

    def populating(*args, **kwargs):
        receipt = kwargs.get("selection_receipt")
        result = optimize(*args, **kwargs)
        assert receipt == {}, "grouped _optimize is expected to leave this empty"
        receipt.update(
            {
                "rule": "best_feasible_loss",
                "selected_epoch": 3,
                "epochs_executed": kwargs["epochs"],
                "selected_loss_float32": 0.5,
            }
        )
        return result

    monkeypatch.setattr(solve, "_optimize", populating)
    seen = []
    result = _fit(target_snapshots=_observer(seen))
    assert result.options["iterate_selection_receipt"]["selected_epoch"] == 3
    selected = [item for item in seen if item["iterate"] == ITERATE_SELECTED]
    assert len(selected) == 1
    assert selected[0]["best_retained"] == {
        "available": False,
        "epoch": None,
        "loss": None,
    }
    assert selected[0]["epoch"] == 20
    assert selected[0]["selection"]["rule"] == "closing_state"


def test_grouped_in_loop_estimates_are_the_observed_loss_tensor(monkeypatch):
    """Each row is the epoch's own float32 forward pass, not a recomputation.

    The row is stamped ``epoch = e + 1`` but its parameters are the accepted
    state the private observer reported at ``epoch = e``: a systematic one-epoch
    lag. Asserting both halves of that pins the pairing a consumer must use, and
    the difference from ``epoch = e + 1``'s accepted state shows why producing
    totals for the accepted vector would need another evaluation.
    """
    counts, observed, _ = _count_evaluations(monkeypatch)
    seen = []
    proof = []
    result = calibrate(
        _frame(),
        _targets(),
        epochs=20,
        learning_rate=0.08,
        grouped_upper_bounds=_groups(),
        grouped_preserve_zeros=True,
        _post_projection_observer=proof.append,
        target_snapshots=_observer(seen),
    )
    in_loop = [item for item in seen if item["iterate"] != ITERATE_SELECTED]
    assert len(in_loop) == len(observed) == 20
    for epoch, item in enumerate(in_loop):
        emitted = np.asarray([row["estimate"] for row in item["targets"]])
        # The exact tensor the loss was computed from.
        assert list(emitted) == list(observed[epoch].astype(np.float64))
        assert item["loss"] == float(result.loss_trajectory[epoch])
        # Its parameters are the accepted state of the PREVIOUS projection,
        # recoverable to float32 round-off through exp(log(x)).
        lagged = result.problem.estimates(proof[epoch]["weights"])
        assert np.allclose(emitted, lagged, rtol=1e-6, atol=0.0)
        # And it is not the accepted state of the projection that follows.
        following = result.problem.estimates(proof[epoch + 1]["weights"])
        assert not np.array_equal(emitted, following)

    selected = [item for item in seen if item["iterate"] == ITERATE_SELECTED][0]
    finals = [diagnostic.final_estimate for diagnostic in result.diagnostics]
    assert [row["estimate"] for row in selected["targets"]] == finals
    assert selected["loss"] == float(result.closing_loss)
    # Three closing evaluations belong to the solver (two diagnostics, one
    # reused final estimate); the 40 extra are this test's own recomputations.
    assert counts["estimates"] == 3 + 40
    assert counts["apply"] == 20


def test_a_sink_mutating_delivered_payloads_cannot_reach_the_solver():
    """Delivered metadata and target rows are the sink's to ruin, alone.

    The one mutable object the grouped integration introduces is the selection
    label mapping, built once per run and handed to every in-loop emission, so
    a leak there would show up as the previous payload's damage arriving in the
    next one.
    """
    baseline = _fit()
    seen = []
    delivered = []

    def hostile(payload):
        # Read the payload as delivered, then ruin every mutable part of it.
        delivered.append(
            (
                payload["epoch"],
                payload["iterate"],
                dict(payload["selection"]),
                dict(payload["best_retained"]),
                dict(payload["context"]),
                [row["name"] for row in payload["targets"]],
            )
        )
        seen.append(payload)
        payload["selection"]["rule"] = "tampered"
        payload["best_retained"]["available"] = True
        payload["context"]["tampered"] = True
        payload["epoch"] = -1
        for row in payload["targets"]:
            row["estimate"] = 0.0
            row["name"] = "tampered"

    result = _fit(
        target_snapshots=TargetSnapshotObserver(
            sink=hostile,
            run_id="grouped-run",
            cadence=TargetSnapshotCadence(every=EVERY_EPOCH),
            context={"lane": "control"},
        )
    )
    assert result.weights.tobytes() == baseline.weights.tobytes()
    assert result.loss_trajectory.tobytes() == baseline.loss_trajectory.tobytes()
    assert result.options == baseline.options
    assert float(result.closing_loss) == float(baseline.closing_loss)
    assert (
        result.frame.resolve_weights("household").values.tobytes()
        == baseline.frame.resolve_weights("household").values.tobytes()
    )
    assert [d.final_estimate for d in result.diagnostics] == [
        d.final_estimate for d in baseline.diagnostics
    ]
    # Every later payload arrived freshly built rather than carrying the damage.
    assert len(delivered) == 21
    assert [item[0] for item in delivered] == list(range(1, 21)) + [20]
    assert [item[1] for item in delivered[:-1]] == [ITERATE_CURRENT] * 20
    assert delivered[-1][1] == ITERATE_SELECTED
    for _, _, selection, best_retained, context, names in delivered:
        assert selection["rule"] == "closing_state"
        assert best_retained == {"available": False, "epoch": None, "loss": None}
        assert context == {"lane": "control"}
        assert names == ["native@0", "detail@0"]
    # And the sink's damage really did land on the objects it was handed.
    assert seen[-1]["targets"][0]["name"] == "tampered"


def test_a_sink_mutating_live_frame_ids_still_trips_the_final_admission():
    """The last thing to run is still the identity guard, not the sink.

    The delivered count pins which guard raised: all 21 payloads arrive first,
    so the loop and the closing emission both completed and the in-loop
    ``grouped_observer`` copies of the same check cannot be the one raising.
    """
    frame = _frame()
    delivered = []

    def saboteur(payload):
        delivered.append(payload["iterate"])
        if payload["iterate"] == ITERATE_SELECTED:
            frame.table("household").loc[[0, 2], "household_id"] = [2, 0]

    with pytest.raises(ValueError, match="ordered IDs"):
        calibrate(
            frame,
            _targets(),
            epochs=20,
            learning_rate=0.08,
            grouped_upper_bounds=_groups(),
            grouped_preserve_zeros=True,
            target_snapshots=TargetSnapshotObserver(
                sink=saboteur,
                run_id="grouped-run",
                cadence=TargetSnapshotCadence(every=EVERY_EPOCH),
            ),
        )
    assert delivered == [ITERATE_CURRENT] * 20 + [ITERATE_SELECTED]


def test_a_sink_cannot_write_through_the_live_weight_arrays_at_all():
    """The weight arm of the admission guard is closed by construction.

    A sink can reach the frame's weight values, but they are read-only, so it
    cannot stage the mutation the accepted-byte and fixed-support checks exist
    to catch. Only the household-ID column is writable, which is why the
    ordered-ID arm is the one a sink can actually reach.
    """
    frame = _frame()
    values = frame.resolve_weights("household").values

    def saboteur(payload):
        if payload["iterate"] == ITERATE_SELECTED:
            values[3] = 7.0

    with pytest.raises(ValueError, match="read-only"):
        calibrate(
            frame,
            _targets(),
            epochs=5,
            learning_rate=0.08,
            grouped_upper_bounds=_groups(),
            grouped_preserve_zeros=True,
            target_snapshots=TargetSnapshotObserver(
                sink=saboteur,
                run_id="grouped-run",
                cadence=TargetSnapshotCadence(every=EVERY_EPOCH),
            ),
        )
    assert not values.flags.writeable


def test_grouped_snapshots_carry_no_record_level_content():
    """Aggregate target rows only: no weights, IDs, group map or bounds.

    Structural rather than length-keyed: the compiled target rows are the only
    sequence a payload may contain at any depth, so a record-level vector of
    any length fails regardless of how many households the fixture has.
    """
    seen = []
    _fit(target_snapshots=_observer(seen))
    forbidden = {"household_id", "household_ids", "weights", "group_indices"}

    def scan(value, *, inside_targets):
        if isinstance(value, dict):
            assert not (set(map(str, value)) & forbidden), sorted(value)
            for item in value.values():
                scan(item, inside_targets=inside_targets)
        elif isinstance(value, (list, tuple)):
            # The only sequence anywhere in a payload is one target row list.
            assert inside_targets, value
            for item in value:
                scan(item, inside_targets=False)

    for payload in seen:
        for key, value in payload.items():
            scan(value, inside_targets=(key == "targets"))
        assert payload["n_targets"] == 2
        assert len(payload["targets"]) == 2


def test_grouped_rows_follow_the_compiled_order_through_duplicate_names():
    """The bind takes its identity from the compiled problem, not the caller.

    Two targets share a name here, so a snapshot that keyed rows by name rather
    than by compiled position would collapse them; and the ordering asserted is
    ``problem.names``, not the order the TargetSet was written in.
    """
    # int 2024 and str "2024" are distinct TargetSet keys with the same lossy
    # f"{name}@{period}" row label, which is how a collision is reachable.
    targets = TargetSet(
        (
            Target(
                name="detail",
                period=2024,
                entity="household",
                value=10.0,
                measure="detail",
            ),
            Target(
                name="native",
                period=2024,
                entity="household",
                value=150.0,
                measure="native",
            ),
            Target(
                name="native",
                period="2024",
                entity="household",
                value=140.0,
                measure="native",
            ),
        )
    )
    seen = []
    result = calibrate(
        _frame(),
        targets,
        epochs=6,
        learning_rate=0.08,
        grouped_upper_bounds=_groups(),
        grouped_preserve_zeros=True,
        target_snapshots=_observer(seen),
    )
    assert len(result.problem.names) == 3
    assert [row["name"] for row in seen[0]["targets"]] == list(result.problem.names)
    duplicated = [
        index
        for index, name in enumerate(result.problem.names)
        if list(result.problem.names).count(name) > 1
    ]
    assert len(duplicated) == 2
    for payload in seen:
        assert [row["index"] for row in payload["targets"]] == [0, 1, 2]
        assert [row["target"] for row in payload["targets"]] == list(
            result.problem.target_vector
        )
        assert payload["n_targets"] == 3
    # The two same-named rows stay separable by position and by target value.
    first, second = (seen[0]["targets"][index] for index in duplicated)
    assert first["name"] == second["name"]
    assert first["index"] != second["index"]
    assert first["target"] != second["target"]
