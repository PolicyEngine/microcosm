"""Per-target calibration estimate snapshots (microcosm#908, slice 1).

Covers the shared codec (schema identity, ordering, finiteness, signed
relative error with the zero-target convention, aggregate-only refusal),
the cadence, the atomic/immutable local store, and the solver seam:
emission from real Adam iterations, honest current/best_retained/selected
labelling, multi-phase and budget-search identity, and byte-identical
optimizer results with the observer off and on.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from microcosm.calibrate.solve import CONSERVE_MASS, calibrate, calibrate_l0_refit
from microcosm.calibrate.target import Target, TargetSet
from microcosm.calibrate.target_snapshots import (
    EVERY_EPOCH,
    ITERATE_BEST_RETAINED,
    ITERATE_CURRENT,
    ITERATE_SELECTED,
    LATEST_SNAPSHOT_FILENAME,
    TARGET_SNAPSHOT_SCHEMA,
    TARGET_SNAPSHOT_SCHEMA_VERSION,
    TargetSnapshotCadence,
    TargetSnapshotError,
    TargetSnapshotObserver,
    TargetSnapshotWriter,
    iter_history,
    signed_relative_error,
    target_identity_digest,
    validate_target_snapshot,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

#: One person per household, so household weights are the calibrated vector.
_SCHEMA = EntitySchema(group_entities=("household",))

#: A tiny invented frame: four households with an income and an adult count.
_INCOME = np.array([100.0, 200.0, 300.0, 400.0])
_ADULTS = np.array([1.0, 2.0, 1.0, 2.0])
_WEIGHTS = np.array([10.0, 10.0, 10.0, 10.0])


def _frame() -> Frame:
    import pandas as pd

    household_ids = np.arange(len(_WEIGHTS), dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": household_ids,
            "person_household_id": household_ids,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "income": _INCOME,
            "adults": _ADULTS,
        }
    )
    return Frame(
        {"person": person, "household": household},
        _SCHEMA,
        {"household": Weights(values=_WEIGHTS.copy(), kind=WeightKind.DESIGN)},
    )


def _targets() -> TargetSet:
    """Targets deliberately off the frame's aggregates, so the solver has to move."""
    return TargetSet(
        [
            Target(
                name="income",
                period=2024,
                entity="household",
                measure="income",
                value=12000.0,
            ),
            Target(
                name="adults",
                period=2024,
                entity="household",
                measure="adults",
                value=50.0,
            ),
        ]
    )


def _collect(**observer_kwargs) -> tuple[list[dict], TargetSnapshotObserver]:
    seen: list[dict] = []
    observer = TargetSnapshotObserver(
        sink=seen.append, run_id="run-1", **observer_kwargs
    )
    return seen, observer


# --------------------------------------------------------------------------
# Codec: identity, ordering, finiteness, zero-target semantics
# --------------------------------------------------------------------------


def test_target_identity_digest_is_order_sensitive_and_value_sensitive():
    names = ("a@2024", "b@2024")
    base = target_identity_digest(names, (1.0, 2.0))
    assert base == target_identity_digest(names, (1.0, 2.0))
    assert base != target_identity_digest(("b@2024", "a@2024"), (2.0, 1.0))
    assert base != target_identity_digest(names, (1.0, 2.5))
    assert len(base) == 64


def test_target_identity_digest_separates_duplicate_names_by_row():
    """Duplicate row labels must not abort a run the observer is only watching.

    A compiled problem can produce two rows labelled the same (row_name is the
    lossy f"{name}@{period}"), so the digest carries the row index and stays
    unambiguous instead of refusing.
    """
    digest = target_identity_digest(("a", "a"), (1.0, 2.0))
    assert digest != target_identity_digest(("a", "a"), (2.0, 1.0))
    assert len(digest) == 64


def test_target_identity_digest_digests_a_non_finite_target_as_null():
    nan = target_identity_digest(("a", "b"), (1.0, float("nan")))
    inf = target_identity_digest(("a", "b"), (1.0, float("inf")))
    assert nan == inf  # both are "no finite target value"
    assert nan != target_identity_digest(("a", "b"), (1.0, 2.0))


def test_target_identity_digest_refuses_misaligned_or_empty_input():
    with pytest.raises(TargetSnapshotError):
        target_identity_digest(("a", "b"), (1.0,))
    with pytest.raises(TargetSnapshotError):
        target_identity_digest((), ())
    with pytest.raises(TargetSnapshotError):
        target_identity_digest(("",), (1.0,))


def test_signed_relative_error_matches_the_solver_zero_target_convention():
    assert signed_relative_error(90.0, 100.0) == pytest.approx(-0.1)
    assert signed_relative_error(110.0, 100.0) == pytest.approx(0.1)
    # Zero target: the relative form is undefined, so the solver's diagnostics
    # report the signed absolute miss. The snapshot must agree exactly.
    assert signed_relative_error(3.0, 0.0) == pytest.approx(3.0)
    assert signed_relative_error(-3.0, 0.0) == pytest.approx(-3.0)


def test_validate_rejects_reordered_rows_and_wrong_digest():
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    payload = observer.bind(
        names=("a@2024", "b@2024"), targets=np.array([10.0, 20.0])
    ).snapshot(np.array([9.0, 21.0]), epoch=1, epochs=1, iterate=ITERATE_CURRENT)
    validate_target_snapshot(payload)

    reordered = json.loads(json.dumps(payload))
    reordered["targets"] = list(reversed(reordered["targets"]))
    with pytest.raises(TargetSnapshotError, match="index"):
        validate_target_snapshot(reordered)

    wrong_digest = json.loads(json.dumps(payload))
    wrong_digest["targets_sha256"] = "0" * 64
    with pytest.raises(TargetSnapshotError, match="targets_sha256"):
        validate_target_snapshot(wrong_digest)


def test_validate_rejects_non_finite_and_inconsistent_relative_error():
    _, observer = _collect()
    bound = observer.bind(names=("a@2024",), targets=np.array([10.0]))
    payload = bound.snapshot(
        np.array([9.0]), epoch=1, epochs=1, iterate=ITERATE_CURRENT
    )
    broken = json.loads(json.dumps(payload))
    broken["targets"][0]["relative_error"] = 0.5
    with pytest.raises(TargetSnapshotError, match="relative_error"):
        validate_target_snapshot(broken)

    # A non-finite estimate is recorded as null, not raised: an unobserved run
    # would have completed, so the observer must not be what kills it.
    diverged = bound.snapshot(
        np.array([float("inf")]), epoch=1, epochs=1, iterate=ITERATE_CURRENT
    )
    assert diverged["targets"][0]["estimate"] is None
    assert diverged["targets"][0]["relative_error"] is None
    assert diverged["non_finite_rows"] == 1
    validate_target_snapshot(diverged)


def test_validate_refuses_record_level_identifiers_in_context():
    """A record-level identifier is refused even as a bare scalar.

    The closed metadata contract now catches this where the caller set it, at
    observer construction, rather than at the first mid-run emission.
    """
    with pytest.raises(TargetSnapshotError, match="aggregate-only"):
        _collect(context={"household_id": 4})
    _, observer = _collect()
    payload = observer.bind(names=("a@2024",), targets=np.array([10.0])).snapshot(
        np.array([9.0]), epoch=1, epochs=1, iterate=ITERATE_CURRENT
    )
    payload["context"] = {"household_id": 4}
    with pytest.raises(TargetSnapshotError, match="aggregate-only"):
        validate_target_snapshot(payload)


def test_validate_refuses_unknown_top_level_keys():
    _, observer = _collect()
    payload = observer.bind(names=("a@2024",), targets=np.array([10.0])).snapshot(
        np.array([9.0]), epoch=1, epochs=1, iterate=ITERATE_CURRENT
    )
    smuggled = json.loads(json.dumps(payload))
    smuggled["household_weights"] = [1.0, 2.0]
    with pytest.raises(TargetSnapshotError, match="unknown"):
        validate_target_snapshot(smuggled)


def test_schema_identity_is_pinned():
    _, observer = _collect()
    payload = observer.bind(names=("a@2024",), targets=np.array([10.0])).snapshot(
        np.array([9.0]), epoch=1, epochs=1, iterate=ITERATE_CURRENT
    )
    assert payload["schema"] == TARGET_SNAPSHOT_SCHEMA
    assert payload["schema_version"] == TARGET_SNAPSHOT_SCHEMA_VERSION
    assert payload["iterate"] in {
        ITERATE_CURRENT,
        ITERATE_BEST_RETAINED,
        ITERATE_SELECTED,
    }


# --------------------------------------------------------------------------
# Cadence
# --------------------------------------------------------------------------


def test_cadence_every_epoch_emits_every_epoch():
    cadence = TargetSnapshotCadence(every=EVERY_EPOCH)
    assert [cadence.emits(e, 5) for e in range(1, 6)] == [True] * 5


def test_bounded_cadence_emits_first_multiples_and_last():
    cadence = TargetSnapshotCadence(every=4)
    assert [cadence.emits(e, 10) for e in range(1, 11)] == [
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        True,
    ]


def test_cadence_rejects_non_positive_every():
    with pytest.raises(ValueError):
        TargetSnapshotCadence(every=0)


# --------------------------------------------------------------------------
# Local store: atomic latest, immutable bounded history
# --------------------------------------------------------------------------


def _payloads(n: int) -> list[dict]:
    _, observer = _collect()
    bound = observer.bind(names=("a@2024",), targets=np.array([10.0]))
    return [
        bound.snapshot(
            np.array([9.0 + i]), epoch=i + 1, epochs=n, iterate=ITERATE_CURRENT
        )
        for i in range(n)
    ]


def test_writer_replaces_latest_atomically_and_never_leaves_partial_json(tmp_path):
    writer = TargetSnapshotWriter(tmp_path)
    for payload in _payloads(3):
        writer(payload)
        text = (tmp_path / LATEST_SNAPSHOT_FILENAME).read_text()
        validate_target_snapshot(json.loads(text))
    assert json.loads((tmp_path / LATEST_SNAPSHOT_FILENAME).read_text())["epoch"] == 3
    # No temporary files survive a completed write.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [LATEST_SNAPSHOT_FILENAME, "history", "history_index.json"]
    )


def test_writer_history_chunks_are_write_once(tmp_path):
    writer = TargetSnapshotWriter(tmp_path)
    payloads = _payloads(2)
    writer(payloads[0])
    chunk = next((tmp_path / "history").iterdir())
    with pytest.raises(TargetSnapshotError, match="immutable"):
        writer._write_chunk(chunk, payloads[0])


def test_writer_bounds_retained_history_and_records_what_it_dropped(tmp_path):
    writer = TargetSnapshotWriter(tmp_path, history_limit=2)
    for payload in _payloads(5):
        writer(payload)
    chunks = sorted(p.name for p in (tmp_path / "history").iterdir())
    assert len(chunks) == 2
    index = json.loads((tmp_path / "history_index.json").read_text())
    assert index["retained"] == chunks
    assert index["dropped"] == 3
    assert index["last_sequence"] == 5
    # The retained chunks are the most recent ones and still validate.
    for name in chunks:
        validate_target_snapshot(json.loads((tmp_path / "history" / name).read_text()))


# --------------------------------------------------------------------------
# Solver seam
# --------------------------------------------------------------------------


def test_observer_off_is_the_default_and_changes_nothing():
    baseline = calibrate(_frame(), _targets(), epochs=20, seed=0)
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    observed = calibrate(
        _frame(), _targets(), epochs=20, seed=0, target_snapshots=observer
    )
    np.testing.assert_array_equal(
        baseline.weights,
        observed.weights,
    )
    np.testing.assert_array_equal(baseline.loss_trajectory, observed.loss_trajectory)
    assert seen, "an enabled observer must actually receive snapshots"


def test_every_epoch_emits_one_snapshot_per_epoch_plus_the_selected_one():
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    calibrate(_frame(), _targets(), epochs=6, seed=0, target_snapshots=observer)
    epochs = [s["epoch"] for s in seen if s["iterate"] != ITERATE_SELECTED]
    assert epochs == [0, 1, 2, 3, 4, 5]
    selected = [s for s in seen if s["iterate"] == ITERATE_SELECTED]
    assert len(selected) == 1
    for snapshot in seen:
        validate_target_snapshot(snapshot)


def test_bounded_cadence_emits_fewer_snapshots_than_every_epoch():
    dense, dense_observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    calibrate(_frame(), _targets(), epochs=40, seed=0, target_snapshots=dense_observer)
    sparse, sparse_observer = _collect(cadence=TargetSnapshotCadence(every=10))
    calibrate(_frame(), _targets(), epochs=40, seed=0, target_snapshots=sparse_observer)
    assert len(sparse) < len(dense)
    assert [s["epoch"] for s in sparse if s["iterate"] != ITERATE_SELECTED] == [
        0,
        9,
        19,
        29,
        39,
    ]


def test_snapshot_values_are_the_real_iterate_estimates():
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    result = calibrate(
        _frame(), _targets(), epochs=30, seed=0, target_snapshots=observer
    )
    selected = [s for s in seen if s["iterate"] == ITERATE_SELECTED][-1]
    by_name = {d.name: d for d in result.diagnostics}
    for row in selected["targets"]:
        diagnostic = by_name[row["name"]]
        assert row["estimate"] == pytest.approx(diagnostic.final_estimate, rel=1e-9)
        assert row["target"] == pytest.approx(diagnostic.target, rel=1e-9)
        assert row["relative_error"] == pytest.approx(
            diagnostic.relative_error, rel=1e-9
        )


def test_current_iterates_are_never_labelled_best_when_they_are_not():
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    calibrate(_frame(), _targets(), epochs=30, seed=0, target_snapshots=observer)
    epoch_snapshots = [s for s in seen if s["iterate"] != ITERATE_SELECTED]
    best_so_far = math.inf
    for snapshot in epoch_snapshots:
        loss = snapshot["loss"]
        is_incumbent = loss < best_so_far
        best_so_far = min(best_so_far, loss)
        expected = ITERATE_BEST_RETAINED if is_incumbent else ITERATE_CURRENT
        assert snapshot["iterate"] == expected, (
            f"epoch {snapshot['epoch']} labelled {snapshot['iterate']}"
        )
        assert snapshot["best_retained"]["available"] is True
    assert any(s["iterate"] == ITERATE_BEST_RETAINED for s in epoch_snapshots)


def test_a_run_that_retains_no_best_says_so_instead_of_claiming_one():
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    calibrate(
        _frame(),
        _targets(),
        epochs=10,
        seed=0,
        mass=CONSERVE_MASS,
        target_snapshots=observer,
    )
    epoch_snapshots = [s for s in seen if s["iterate"] != ITERATE_SELECTED]
    assert epoch_snapshots
    assert all(s["iterate"] == ITERATE_CURRENT for s in epoch_snapshots)
    assert all(s["best_retained"]["available"] is False for s in epoch_snapshots)


def test_multi_phase_and_search_identity_is_unambiguous():
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=5))
    calibrate_l0_refit(
        _frame(),
        _targets(),
        epochs=10,
        refit_epochs=10,
        target_records=2,
        budget_iters=2,
        seed=0,
        target_snapshots=observer,
    )
    phases = {s["phase"] for s in seen}
    assert {"l0_selection", "post_l0_refit"} <= phases
    searched = [s for s in seen if s["search"] is not None]
    assert searched, "budget-search probes must carry their search identity"
    assert {s["search"]["budget_iteration"] for s in searched} >= {1, 2}
    # (phase, search iteration, epoch, iterate) must identify a snapshot.
    keys = [
        (
            s["phase"],
            None if s["search"] is None else s["search"]["budget_iteration"],
            s["epoch"],
            s["iterate"],
            s["sequence"],
        )
        for s in seen
    ]
    assert len({k[:4] for k in keys}) == len(keys)
    assert [k[4] for k in keys] == sorted(k[4] for k in keys)
    for snapshot in seen:
        validate_target_snapshot(snapshot)


def test_sink_exceptions_follow_the_progress_callback_contract():
    def boom(_payload):
        raise RuntimeError("sink down")

    observer = TargetSnapshotObserver(
        sink=boom, run_id="run-1", cadence=TargetSnapshotCadence(every=EVERY_EPOCH)
    )
    with pytest.raises(RuntimeError, match="sink down"):
        calibrate(_frame(), _targets(), epochs=4, seed=0, target_snapshots=observer)


def test_sink_mutation_cannot_reach_the_next_snapshot():
    seen: list[dict] = []

    def mutating(payload):
        payload["targets"].clear()
        payload["run_id"] = "mutated"
        seen.append(json.loads(json.dumps(payload)))

    observer = TargetSnapshotObserver(
        sink=mutating,
        run_id="run-1",
        cadence=TargetSnapshotCadence(every=EVERY_EPOCH),
    )
    kept: list[dict] = []
    observer_two = TargetSnapshotObserver(
        sink=kept.append,
        run_id="run-1",
        cadence=TargetSnapshotCadence(every=EVERY_EPOCH),
    )
    mutated_result = calibrate(
        _frame(), _targets(), epochs=5, seed=0, target_snapshots=observer
    )
    clean_result = calibrate(
        _frame(), _targets(), epochs=5, seed=0, target_snapshots=observer_two
    )
    np.testing.assert_array_equal(
        mutated_result.weights,
        clean_result.weights,
    )
    assert all(len(s["targets"]) == 0 for s in seen)
    assert all(len(s["targets"]) == 2 for s in kept)


def test_snapshots_carry_no_record_level_vectors(tmp_path: Path):
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    result = calibrate(
        _frame(), _targets(), epochs=5, seed=0, target_snapshots=observer
    )
    n_records = int(result.weights.shape[0])
    for snapshot in seen:
        text = json.dumps(snapshot)
        assert "household_id" not in text
        assert "household_weight" not in text
        for value in snapshot.values():
            assert not (
                isinstance(value, list)
                and len(value) == n_records
                and value
                and isinstance(value[0], float)
            )


# --------------------------------------------------------------------------
# Regressions found by adversarial review of the first draft of this slice
# --------------------------------------------------------------------------


def test_a_diverging_run_still_returns_weights_with_the_observer_on():
    """The observer must never be the thing that aborts an otherwise-good run.

    At this learning rate the estimate blows up to inf mid-run; the capped loss
    absorbs it and calibrate returns weights. The first draft raised on the
    non-finite estimate and destroyed a run that would have succeeded.
    """
    baseline = calibrate(_frame(), _targets(), epochs=60, seed=0, learning_rate=14.0)
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    observed = calibrate(
        _frame(),
        _targets(),
        epochs=60,
        seed=0,
        learning_rate=14.0,
        target_snapshots=observer,
    )
    np.testing.assert_array_equal(baseline.weights, observed.weights)
    np.testing.assert_array_equal(baseline.loss_trajectory, observed.loss_trajectory)
    assert any(s["non_finite_rows"] > 0 for s in seen), (
        "this fixture is meant to drive a non-finite estimate"
    )
    for snapshot in seen:
        validate_target_snapshot(snapshot)


def test_duplicate_compiled_row_labels_do_not_abort_the_run():
    """`row_name` is the lossy f"{name}@{period}", so labels can collide."""
    targets = TargetSet(
        [
            Target(
                name="income",
                period=2024,
                entity="household",
                measure="income",
                value=12000.0,
            ),
            Target(
                name="income",
                period="2024",
                entity="household",
                measure="income",
                value=13000.0,
            ),
        ]
    )
    baseline = calibrate(_frame(), targets, epochs=10, seed=0)
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    observed = calibrate(
        _frame(), targets, epochs=10, seed=0, target_snapshots=observer
    )
    np.testing.assert_array_equal(baseline.weights, observed.weights)
    assert seen
    names = [row["name"] for row in seen[0]["targets"]]
    assert names == ["income@2024", "income@2024"]
    for snapshot in seen:
        validate_target_snapshot(snapshot)


def test_the_selected_snapshot_names_the_epoch_it_actually_selected():
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    result = calibrate(
        _frame(),
        _targets(),
        epochs=30,
        seed=0,
        learning_rate=0.1,
        target_snapshots=observer,
    )
    receipt = result.options["iterate_selection_receipt"]
    selected = [s for s in seen if s["iterate"] == ITERATE_SELECTED][-1]
    assert selected["epoch"] == receipt["selected_epoch"]


def test_retained_and_selected_snapshots_use_completed_update_identity():
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    result = calibrate(
        _frame(),
        _targets(),
        epochs=30,
        seed=0,
        learning_rate=0.1,
        target_snapshots=observer,
    )
    receipt = result.options["iterate_selection_receipt"]
    assert receipt["selected_epoch"] < 30  # Exercise an earlier retained state.
    retained = [s for s in seen if s["iterate"] == ITERATE_BEST_RETAINED][-1]
    selected = seen[-1]
    assert retained["epoch"] == selected["epoch"]
    assert retained["best_retained"]["epoch"] == selected["best_retained"]["epoch"]
    assert retained["loss"] == receipt["selected_loss_float32"]


def test_selected_budget_snapshot_identifies_the_winning_probe():
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    result = calibrate(
        _frame(),
        _targets(),
        epochs=8,
        seed=0,
        target_records=2,
        budget_iters=2,
        target_snapshots=observer,
    )
    selected = seen[-1]
    assert selected["iterate"] == ITERATE_SELECTED
    assert selected["search"] is not None
    receipt = result.options["budget_search"]
    assert selected["search"] == {
        "budget_iteration": receipt["selected_budget_iteration"],
        "budget_iters": receipt["budget_iters"],
        "l0_lambda": receipt["selected_l0_lambda"],
    }
    matching = [s for s in seen[:-1] if s["search"] == selected["search"]]
    assert matching


@pytest.mark.parametrize("estimate,target", [(1.0, 5e-324), (1e308, -1e308)])
def test_overflowing_relative_error_remains_a_counted_null_diagnostic(estimate, target):
    _, observer = _collect()
    snapshot = observer.bind(names=("tiny",), targets=np.array([target])).snapshot(
        np.array([estimate]),
        epoch=0,
        epochs=1,
        iterate=ITERATE_CURRENT,
    )
    assert snapshot["targets"][0] == {
        "index": 0,
        "name": "tiny",
        "target": target,
        "estimate": estimate,
        "relative_error": None,
    }
    assert snapshot["non_finite_rows"] == 1
    validate_target_snapshot(snapshot)
    broken = copy.deepcopy(snapshot)
    broken["targets"][0]["relative_error"] = 0.0
    with pytest.raises(TargetSnapshotError, match="relative_error"):
        validate_target_snapshot(broken)


def test_a_store_refuses_to_share_a_directory_with_another_run():
    """Sequences restart at 1 per observer, so a shared directory collides."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        first = TargetSnapshotWriter(Path(directory))
        for payload in _payloads(2):
            first(payload)
        with pytest.raises(TargetSnapshotError, match="already holds"):
            TargetSnapshotWriter(Path(directory))
        # Even the explicit adoption escape hatch keeps chunks write-once: a
        # colliding sequence is refused, never silently overwritten.
        adopted = TargetSnapshotWriter(Path(directory), allow_existing_history=True)
        with pytest.raises(TargetSnapshotError, match="immutable"):
            adopted(_payloads(1)[0])


def test_pruning_never_deletes_another_runs_retained_chunks():
    import tempfile

    _, observer = _collect()
    bound = observer.bind(names=("a@2024",), targets=np.array([10.0]))
    with tempfile.TemporaryDirectory() as directory:
        first = TargetSnapshotWriter(Path(directory))
        for index in range(2):
            first(
                bound.snapshot(
                    np.array([9.0 + index]),
                    epoch=index + 1,
                    epochs=2,
                    iterate=ITERATE_CURRENT,
                )
            )
        # A second writer adopting the directory continues the same observer's
        # sequence, so nothing collides; its own bounded history must prune only
        # its own chunks.
        second = TargetSnapshotWriter(
            Path(directory), history_limit=1, allow_existing_history=True
        )
        for index in range(3):
            second(
                bound.snapshot(
                    np.array([20.0 + index]),
                    epoch=index + 1,
                    epochs=3,
                    iterate=ITERATE_CURRENT,
                )
            )
        retained = sorted(p.name for p in (Path(directory) / "history").iterdir())
        assert "00000001.json" in retained
        assert "00000002.json" in retained
        assert retained[-1] == "00000005.json"


def test_a_store_refuses_a_snapshot_from_a_different_run():
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        writer = TargetSnapshotWriter(Path(directory))
        first = TargetSnapshotObserver(sink=lambda _p: None, run_id="run-a").bind(
            names=("a@2024",), targets=np.array([10.0])
        )
        second = TargetSnapshotObserver(sink=lambda _p: None, run_id="run-b").bind(
            names=("a@2024",), targets=np.array([10.0])
        )
        writer(
            first.snapshot(np.array([9.0]), epoch=1, epochs=1, iterate=ITERATE_CURRENT)
        )
        with pytest.raises(TargetSnapshotError, match="run 'run-a'"):
            writer(
                second.snapshot(
                    np.array([9.0]), epoch=1, epochs=1, iterate=ITERATE_CURRENT
                )
            )


def test_validation_refuses_string_typed_numbers():
    _, observer = _collect()
    payload = observer.bind(names=("a@2024",), targets=np.array([10.0])).snapshot(
        np.array([9.0]), epoch=1, epochs=1, iterate=ITERATE_CURRENT
    )
    for field_name in ("target", "estimate", "relative_error"):
        broken = json.loads(json.dumps(payload))
        broken["targets"][0][field_name] = str(broken["targets"][0][field_name])
        with pytest.raises(TargetSnapshotError, match="JSON number"):
            validate_target_snapshot(broken)
    broken = json.loads(json.dumps(payload))
    broken["loss"] = "0.5"
    with pytest.raises(TargetSnapshotError, match="JSON number"):
        validate_target_snapshot(broken)


def test_the_aggregate_only_scan_covers_the_whole_payload():
    _, observer = _collect()
    payload = observer.bind(names=("a@2024",), targets=np.array([10.0])).snapshot(
        np.array([9.0]), epoch=1, epochs=1, iterate=ITERATE_CURRENT
    )
    smuggled = json.loads(json.dumps(payload))
    smuggled["selection"] = {"rule": "x", "row_id": 7}
    with pytest.raises(TargetSnapshotError, match="aggregate-only"):
        validate_target_snapshot(smuggled)

    # best_retained is a closed triple, so a record-level key has no slot at
    # all rather than needing the denylist to catch it.
    smuggled = json.loads(json.dumps(payload))
    smuggled["best_retained"]["source_values"] = [1.0, 2.0]
    with pytest.raises(TargetSnapshotError, match="unknown keys"):
        validate_target_snapshot(smuggled)

    smuggled = json.loads(json.dumps(payload))
    smuggled["targets"][0]["name"] = "a@2024"
    smuggled["search"] = {"budget_iteration": 1, "benunit_id": 3}
    with pytest.raises(TargetSnapshotError, match="aggregate-only"):
        validate_target_snapshot(smuggled)


def test_the_selected_snapshot_admits_the_run_retained_a_best():
    """A retained-best run must not report best_retained.available=false."""
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    result = calibrate(
        _frame(),
        _targets(),
        epochs=30,
        seed=0,
        learning_rate=0.1,
        target_snapshots=observer,
    )
    receipt = result.options["iterate_selection_receipt"]
    assert receipt, "this fixture is meant to exercise the retain-best rule"
    selected = [s for s in seen if s["iterate"] == ITERATE_SELECTED][-1]
    assert selected["best_retained"]["available"] is True
    if receipt["selected_epoch"] < 30:
        assert selected["best_retained"]["epoch"] == receipt["selected_epoch"]
    else:
        assert selected["best_retained"]["epoch"] is None


def test_a_closing_state_run_still_reports_no_retained_best_on_selected():
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    calibrate(
        _frame(),
        _targets(),
        epochs=10,
        seed=0,
        mass=CONSERVE_MASS,
        target_snapshots=observer,
    )
    selected = [s for s in seen if s["iterate"] == ITERATE_SELECTED][-1]
    assert selected["best_retained"]["available"] is False


# --------------------------------------------------------------------------
# Closing the four reviewed findings (#908 independent review, head bae1887ff)
# --------------------------------------------------------------------------


def _payload(**observer_kwargs) -> dict:
    _, observer = _collect(**observer_kwargs)
    return observer.bind(names=("a@2024",), targets=np.array([10.0])).snapshot(
        np.array([9.0]), epoch=1, epochs=1, iterate=ITERATE_CURRENT
    )


# Finding 1 — a closed, typed, bounded aggregate metadata contract.


def test_context_refuses_the_record_vectors_the_review_smuggled_through():
    """The exact counterexample from the review probe, at every location."""
    vectors = {
        "household_weights": [10.0, 20.0],
        "tax_unit_id": [101, 102],
        "spm_unit_id": [201, 202],
    }
    with pytest.raises(TargetSnapshotError):
        TargetSnapshotObserver(sink=lambda _p: None, run_id="r", context=vectors)
    _, observer = _collect()
    bound = observer.bind(names=("a@2024",), targets=np.array([10.0]))
    with pytest.raises(TargetSnapshotError):
        bound.with_search(**vectors)
    payload = _payload()
    for location in ("context", "search", "selection"):
        smuggled = json.loads(json.dumps(payload))
        smuggled[location] = dict(vectors)
        with pytest.raises(TargetSnapshotError):
            validate_target_snapshot(smuggled)


def test_metadata_refuses_arbitrary_nested_payloads():
    """A nested mapping is not aggregate metadata, whatever its key names are."""
    with pytest.raises(TargetSnapshotError):
        TargetSnapshotObserver(
            sink=lambda _p: None, run_id="r", context={"display": {"unit": "USD"}}
        )
    payload = _payload()
    for location in ("context", "search", "selection"):
        nested = json.loads(json.dumps(payload))
        nested[location] = {"display": {"unit": "USD"}}
        with pytest.raises(TargetSnapshotError):
            validate_target_snapshot(nested)


def test_identifier_fields_must_be_strings():
    with pytest.raises(TargetSnapshotError):
        TargetSnapshotObserver(
            sink=lambda _p: None, run_id="r", candidate_id={"tax_unit_id": [101, 102]}
        )
    with pytest.raises(TargetSnapshotError):
        TargetSnapshotObserver(sink=lambda _p: None, run_id=7)
    for broken in ({"unexpected": [1, 2]}, 7, ["a"]):
        payload = _payload()
        payload["candidate_id"] = broken
        with pytest.raises(TargetSnapshotError):
            validate_target_snapshot(payload)


def test_supported_scalar_metadata_survives_and_is_bounded():
    supported = {
        "dataset": "invented_2024",
        "replicate": 3,
        "tolerance": 0.5,
        "dense": True,
        "note": None,
    }
    payload = _payload(context=supported)
    assert payload["context"] == supported
    with pytest.raises(TargetSnapshotError):
        TargetSnapshotObserver(
            sink=lambda _p: None,
            run_id="r",
            context={f"k{i}": i for i in range(1000)},
        )
    with pytest.raises(TargetSnapshotError):
        TargetSnapshotObserver(
            sink=lambda _p: None, run_id="r", context={"note": "x" * 100_000}
        )


def test_the_structured_metadata_solve_emits_round_trips():
    """The shapes solve.py actually emits stay supported, unchanged."""
    _, observer = _collect()
    bound = observer.bind(names=("a@2024",), targets=np.array([10.0])).with_search(
        budget_iteration=2, budget_iters=8, l0_lambda=0.25
    )
    payload = bound.snapshot(
        np.array([9.0]),
        epoch=1,
        epochs=1,
        iterate=ITERATE_SELECTED,
        precision="float64",
        loss=0.5,
        best_retained={"available": True, "epoch": 1, "loss": 0.4},
        selection={
            "rule": "best_feasible_loss",
            "selected_epoch": 1,
            "epochs_executed": 1,
            "epoch_convention": "completed_optimizer_updates; zero is start",
            "selected_loss_float32": 0.4,
            "closing_iterate_loss_float32": 0.6,
        },
    )
    assert payload["search"] == {
        "budget_iteration": 2,
        "budget_iters": 8,
        "l0_lambda": 0.25,
    }
    assert payload["selection"]["rule"] == "best_feasible_loss"
    assert payload["best_retained"] == {"available": True, "epoch": 1, "loss": 0.4}
    validate_target_snapshot(json.loads(json.dumps(payload)))


# Finding 2 — a sink must not be able to reach caller metadata or later snapshots.


def test_sink_mutation_cannot_reach_caller_metadata_or_the_next_snapshot():
    context = {"dataset": "invented_2024", "replicate": 1}
    seen: list[dict] = []

    def mutating(payload):
        payload["context"]["dataset"] = "changed-by-sink"
        payload["search"]["budget_iteration"] = -1
        payload["selection"]["rule"] = "changed-by-sink"
        payload["best_retained"]["available"] = False
        payload["targets"][0]["name"] = "changed-by-sink"
        seen.append(json.loads(json.dumps(payload)))

    observer = TargetSnapshotObserver(sink=mutating, run_id="r", context=context)
    bound = observer.bind(names=("a@2024",), targets=np.array([10.0])).with_search(
        budget_iteration=2, budget_iters=8, l0_lambda=0.25
    )
    kwargs = dict(
        epochs=2,
        iterate=ITERATE_CURRENT,
        best_retained={"available": True, "epoch": 1, "loss": 0.4},
        selection={"rule": "best_feasible_loss", "selected_epoch": 1},
    )
    bound.emit(np.array([9.0]), epoch=1, **kwargs)
    second = bound.snapshot(np.array([9.0]), epoch=2, **kwargs)

    assert context == {"dataset": "invented_2024", "replicate": 1}
    assert second["context"] == {"dataset": "invented_2024", "replicate": 1}
    assert second["search"]["budget_iteration"] == 2
    assert second["selection"]["rule"] == "best_feasible_loss"
    assert second["best_retained"]["available"] is True
    assert second["targets"][0]["name"] == "a@2024"
    assert seen[0]["context"]["dataset"] == "changed-by-sink"


def test_a_delivered_snapshot_shares_no_mutable_object_with_its_caller():
    """The detachment guarantee, pinned structurally rather than by example.

    Every supported metadata value is an immutable scalar and every container
    is freshly built, so a delivered payload has no mutable object in common
    with the observer, the bound view, the caller's arguments, or the previous
    snapshot. A future seam that reintroduced an alias would fail here.
    """

    def mutable_ids(value, seen=None):
        seen = set() if seen is None else seen
        if isinstance(value, (dict, list)):
            seen.add(id(value))
            items = value.values() if isinstance(value, dict) else value
            for item in items:
                mutable_ids(item, seen)
        return seen

    context = {"dataset": "invented_2024"}
    best = {"available": True, "epoch": 1, "loss": 0.4}
    selection = {"rule": "best_feasible_loss"}
    observer = TargetSnapshotObserver(sink=lambda _p: None, run_id="r", context=context)
    bound = observer.bind(names=("a@2024",), targets=np.array([10.0])).with_search(
        budget_iteration=1
    )
    kwargs = dict(
        epochs=2, iterate=ITERATE_CURRENT, best_retained=best, selection=selection
    )
    first = bound.snapshot(np.array([9.0]), epoch=1, **kwargs)
    second = bound.snapshot(np.array([9.0]), epoch=2, **kwargs)
    caller_side = (
        mutable_ids(context)
        | mutable_ids(dict(observer.context))
        | mutable_ids(best)
        | mutable_ids(selection)
        | mutable_ids(dict(bound.search))
        | mutable_ids(second)
    )
    assert not (mutable_ids(first) & caller_side)


# Finding 3 — a history chunk becomes visible only once its bytes are complete.


def test_a_history_chunk_is_published_only_after_its_bytes_are_complete(
    tmp_path: Path, monkeypatch
):
    from microcosm.calibrate import target_snapshots as module

    writer = TargetSnapshotWriter(tmp_path)
    real_link = module.os.link
    published: list[dict] = []

    def observing_link(source, target, *args, **kwargs):
        # At publication time the bytes must already be complete and the
        # immutable name must not exist yet.
        published.append(
            {
                "retained_before": writer.retained(),
                "complete": json.loads(Path(source).read_text(encoding="utf-8")),
                "target_exists": Path(target).exists(),
            }
        )
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(module.os, "link", observing_link)
    writer(_payload())
    assert len(published) == 1, "the chunk was not published through a temporary file"
    assert published[0]["retained_before"] == ()
    assert published[0]["target_exists"] is False
    assert published[0]["complete"]["schema"] == TARGET_SNAPSHOT_SCHEMA
    assert len(list(iter_history(tmp_path))) == 1


def test_a_failed_chunk_write_leaves_no_partial_or_leftover_file(
    tmp_path: Path, monkeypatch
):
    from microcosm.calibrate import target_snapshots as module

    writer = TargetSnapshotWriter(tmp_path)
    real_fsync = module.os.fsync
    calls: list[int] = []

    def failing_fsync(fd):
        calls.append(fd)
        if len(calls) == 1:
            raise OSError("invented disk failure mid-chunk")
        return real_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        writer(_payload())
    monkeypatch.undo()

    assert writer.retained() == ()
    assert list(iter_history(tmp_path)) == []
    leftovers = sorted(p.name for p in (tmp_path / "history").iterdir())
    assert leftovers == [], f"failed write left {leftovers} behind"


def test_history_publication_still_refuses_to_overwrite_an_existing_chunk(
    tmp_path: Path,
):
    writer = TargetSnapshotWriter(tmp_path)
    payload = _payload()
    writer(payload)
    with pytest.raises(TargetSnapshotError, match="immutable"):
        writer(payload)
    assert len(writer.retained()) == 1
    assert sorted(p.name for p in (tmp_path / "history").iterdir()) == ["00000001.json"]


# Finding 4 — the public codec refuses impossible identity and best metadata.


def test_codec_refuses_the_impossible_payloads_the_review_reproduced():
    # The emitter's omitted best iterate is a complete triple; the public
    # codec must not silently turn a serialized null into that default.
    assert _payload()["best_retained"] == {
        "available": False,
        "epoch": None,
        "loss": None,
    }
    cases = {
        "null_best_metadata": {"best_retained": None},
        "epoch_exceeds_epochs": {"epoch": 99, "epochs": 1},
        "invalid_timestamp": {"created_at": "not-a-time"},
        "naive_timestamp": {"created_at": "2026-09-12T00:00:00"},
        "invalid_best_metadata": {
            "iterate": ITERATE_BEST_RETAINED,
            "best_retained": {"available": True, "epoch": -7, "loss": "not-a-number"},
        },
        "candidate_id_not_string": {"candidate_id": {"unexpected": [1, 2]}},
        "best_epoch_exceeds_epochs": {
            "best_retained": {"available": True, "epoch": 99, "loss": 0.1}
        },
        "unavailable_best_carries_an_epoch": {
            "best_retained": {"available": False, "epoch": 1, "loss": None}
        },
        "unavailable_best_carries_a_loss": {
            "best_retained": {"available": False, "epoch": None, "loss": 0.1}
        },
        "best_retained_unknown_key": {
            "best_retained": {
                "available": True,
                "epoch": 1,
                "loss": 0.1,
                "extra": "x",
            }
        },
        "non_finite_rows_exceeds_n_targets": {"non_finite_rows": 99},
    }
    for name, changes in cases.items():
        payload = _payload()
        payload.update(changes)
        try:
            validate_target_snapshot(payload)
        except TargetSnapshotError:
            continue
        raise AssertionError(f"the codec accepted an impossible payload: {name}")


def test_codec_accepts_the_epochs_the_solver_actually_selects():
    """A retained-best run returns an earlier iterate; epoch 0 is its floor."""
    _, observer = _collect()
    bound = observer.bind(names=("a@2024",), targets=np.array([10.0]))
    for epoch in (0, 3, 7):
        payload = bound.snapshot(
            np.array([9.0]),
            epoch=epoch,
            epochs=7,
            iterate=ITERATE_SELECTED,
            precision="float64",
            best_retained={"available": True, "epoch": epoch, "loss": 0.4},
        )
        assert payload["epoch"] == epoch
        validate_target_snapshot(json.loads(json.dumps(payload)))


def test_non_finite_diagnostics_stay_null_statuses_rather_than_aborting():
    """The observer must never be the thing that ends a run that would finish."""
    _, observer = _collect()
    bound = observer.bind(names=("a@2024", "b@2024"), targets=np.array([10.0, 0.0]))
    payload = bound.emit(
        np.array([float("nan"), 1.0]),
        epoch=1,
        epochs=1,
        iterate=ITERATE_CURRENT,
        loss=float("inf"),
        best_retained={"available": True, "epoch": 1, "loss": float("nan")},
        selection={"rule": "best_feasible_loss", "selected_loss_float32": float("inf")},
    )
    assert payload["loss"] is None
    assert payload["non_finite_rows"] == 1
    assert payload["best_retained"] == {"available": True, "epoch": 1, "loss": None}
    assert payload["selection"]["selected_loss_float32"] is None
    validate_target_snapshot(json.loads(json.dumps(payload)))


def test_the_observer_stays_passive_on_the_l0_budget_refit_and_prox_paths():
    """Strict validation must not change — or end — the runs it only watches.

    The Adam path already had this parity test; the sparse-selection paths are
    where the tightened codec has the most metadata to reject (search
    identity, phase labels, selection receipts), so they get one too.
    """
    l0_kwargs = dict(epochs=8, refit_epochs=8, target_records=2, budget_iters=2, seed=0)
    baseline = calibrate_l0_refit(_frame(), _targets(), **l0_kwargs)
    seen, observer = _collect(cadence=TargetSnapshotCadence(every=EVERY_EPOCH))
    observed = calibrate_l0_refit(
        _frame(), _targets(), target_snapshots=observer, **l0_kwargs
    )
    np.testing.assert_array_equal(baseline.weights, observed.weights)
    assert {s["phase"] for s in seen} >= {"l0_selection", "post_l0_refit"}
    assert any(s["search"] is not None for s in seen)

    prox_kwargs = dict(epochs=8, seed=0, method="prox", l1_lambda=0.01)
    prox_baseline = calibrate(_frame(), _targets(), **prox_kwargs)
    prox_seen, prox_observer = _collect(
        cadence=TargetSnapshotCadence(every=EVERY_EPOCH)
    )
    prox_observed = calibrate(
        _frame(), _targets(), target_snapshots=prox_observer, **prox_kwargs
    )
    np.testing.assert_array_equal(prox_baseline.weights, prox_observed.weights)
    np.testing.assert_array_equal(
        prox_baseline.loss_trajectory, prox_observed.loss_trajectory
    )
    assert prox_seen, "an enabled observer must actually receive snapshots"
    for snapshot in (*seen, *prox_seen):
        validate_target_snapshot(snapshot)


def test_the_codec_is_stricter_than_the_emitting_edge_and_says_so():
    """A validated payload is a payload that serializes.

    The emitter coerces what a running solver honestly produces (numpy
    scalars, a non-finite loss) so the observer can never end a run; the codec
    refuses those same values, so `json.dumps` can never be the thing that
    fails on a payload the codec has already blessed.
    """
    emitted = _payload(context={"tolerance": np.float64(0.5)})
    assert emitted["context"] == {"tolerance": 0.5}
    assert isinstance(emitted["context"]["tolerance"], float)
    json.dumps(emitted)

    for location in ("context", "search", "selection"):
        for bad in (float("nan"), float("inf"), np.float64(0.5)):
            payload = _payload()
            payload[location] = {"tolerance": bad}
            with pytest.raises(TargetSnapshotError):
                validate_target_snapshot(payload)

    for bad_best in (
        {"available": True},
        {"available": True, "epoch": 1},
        {"available": True, "epoch": 1, "loss": float("nan")},
        {"available": True, "epoch": np.int64(1), "loss": 0.1},
    ):
        payload = _payload()
        payload["best_retained"] = bad_best
        with pytest.raises(TargetSnapshotError):
            validate_target_snapshot(payload)


def test_created_at_is_a_timezone_aware_timestamp():
    payload = _payload()
    parsed = datetime.fromisoformat(payload["created_at"])
    assert parsed.tzinfo is not None
    naive = _payload(clock=lambda: datetime(2026, 9, 12))  # noqa: DTZ001
    assert datetime.fromisoformat(naive["created_at"]).tzinfo is not None
