"""Per-stage L2 penalties and target weightings in the UK size chain (microcosm#1124)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from microcosm.build.uk_runtime import dataset_size
from microcosm.build.uk_runtime.dataset_size import (
    UK_SIZE_L2_PARAM_KEYS,
    UKSizeDraw,
    UKSizeL2,
    draw_uk_dataset_size,
    refit_uk_dataset_size,
    select_uk_dataset_size,
    uk_size_l2,
    uk_size_l2_from_options,
)
from microcosm.build.uk_runtime.target_weights import (
    UKStageTargetWeighting,
    uk_target_rows,
)
from microcosm.calibrate import Target, TargetSet, calibrate
from test_support.microcosm_build.uk_local_rowwise import _clone_frame

_COMMON = dict(households=2, epochs=2, learning_rate=0.02, seed=7, pi_hi=0.5)
_NEW_RECEIPT_KEYS = {
    "selection_l2",
    "refit_l2",
    "refit_iterate_selection",
    "selection_target_weighting",
    "refit_target_weighting",
    "compact_loss_under_dense_weights",
}


def _dense(frame):
    return calibrate(
        frame,
        TargetSet(
            [
                Target("count", "household", lambda f: np.ones(f.n("household")), 3),
                Target(
                    "first",
                    "household",
                    lambda f: (
                        f.table("household").household_id.to_numpy() == 101
                    ).astype(float),
                    1,
                ),
            ]
        ),
        epochs=2,
    )


def _weighting(dense, rule="grain_family_equal", held_out=()):
    names = dense.problem.names
    rows = uk_target_rows(
        names,
        [{"family": "count_family"}, {"family": "first_family"}],
        np.asarray(dense.problem.target_vector, dtype=np.float64),
        local=[False, False],
    )
    return UKStageTargetWeighting(rule, rows, held_out=held_out)


def _capturing(monkeypatch):
    calls = {"calibrate": [], "refit_l0_selection": []}
    real_calibrate = dataset_size.calibrate
    real_refit = dataset_size.refit_l0_selection

    def calibrate_spy(*args, **kwargs):
        calls["calibrate"].append(kwargs)
        return real_calibrate(*args, **kwargs)

    def refit_spy(*args, **kwargs):
        calls["refit_l0_selection"].append(kwargs)
        return real_refit(*args, **kwargs)

    monkeypatch.setattr(dataset_size, "calibrate", calibrate_spy)
    monkeypatch.setattr(dataset_size, "refit_l0_selection", refit_spy)
    return calls


def test_uk_size_l2_validates_and_is_off_by_default() -> None:
    assert uk_size_l2("selection") is None
    assert uk_size_l2("refit", l2_lambda=0) is None
    on = uk_size_l2("refit", l2_lambda=0.03, anchor="uniform")
    assert on == UKSizeL2("refit", 0.03, "uniform", "chi_square")
    assert on.solver_kwargs() == {
        "l2_lambda": 0.03,
        "l2_anchor": "uniform",
        "l2_basis": "chi_square",
    }
    assert on.params() == {
        "refit_l2_lambda": 0.03,
        "refit_l2_anchor": "uniform",
        "refit_l2_basis": "chi_square",
    }
    assert set(on.params()) <= set(UK_SIZE_L2_PARAM_KEYS)
    with pytest.raises(ValueError, match="without a positive"):
        uk_size_l2("refit", anchor="uniform")
    with pytest.raises(ValueError, match="'record' is refused"):
        uk_size_l2("refit", l2_lambda=0.1, basis="record")
    with pytest.raises(ValueError, match="'design' is not a size-stage anchor"):
        uk_size_l2("refit", l2_lambda=0.1, anchor="design")
    for bad in (-0.1, float("nan"), float("inf"), True, "0.1"):
        with pytest.raises(ValueError, match="finite number"):
            uk_size_l2("selection", l2_lambda=bad)
    with pytest.raises(ValueError, match="stage"):
        uk_size_l2("dense", l2_lambda=0.1)


def test_default_calls_pass_no_l2_keyword_and_keep_the_receipt(monkeypatch) -> None:
    frame = _clone_frame()
    dense = _dense(frame)
    calls = _capturing(monkeypatch)
    sized = refit_uk_dataset_size(frame, dense, **_COMMON)
    for stage in ("calibrate", "refit_l0_selection"):
        (kwargs,) = calls[stage]
        assert not {"l2_lambda", "l2_anchor", "l2_basis"} & set(kwargs)
        assert kwargs["target_loss_weights"] is dense.target_loss_weights
    assert not _NEW_RECEIPT_KEYS & set(sized.receipt)
    assert sized.result.options["l2_lambda"] == 0.0


def test_refit_l2_reaches_the_refit_only_and_is_receipted(monkeypatch) -> None:
    frame = _clone_frame()
    dense = _dense(frame)
    calls = _capturing(monkeypatch)
    sized = refit_uk_dataset_size(
        frame, dense, refit_l2_lambda=0.05, refit_l2_anchor="uniform", **_COMMON
    )
    (search_kwargs,) = calls["calibrate"]
    assert "l2_lambda" not in search_kwargs
    (refit_kwargs,) = calls["refit_l0_selection"]
    assert refit_kwargs["l2_lambda"] == 0.05
    assert refit_kwargs["l2_anchor"] == "uniform"
    assert refit_kwargs["l2_basis"] == "chi_square"
    options = sized.result.options
    assert options["l2_lambda"] == 0.05 and options["l2_basis"] == "chi_square"
    assert sized.receipt["refit_iterate_selection"] == options["iterate_selection"]
    block = sized.receipt["refit_l2"]
    assert block["lambda"] == 0.05 and block["anchor"] == "uniform"
    assert block["anchor_reference"] == "normalized_horvitz_thompson_w_over_q_mean"
    assert block["chi_square_distance_to_anchor"] >= 0.0
    assert "selection_l2" not in sized.receipt
    floored = refit_uk_dataset_size(
        frame, dense, refit_l2_lambda=0.05, baseline_pi_floor=0.5, **_COMMON
    )
    assert floored.receipt["refit_l2"]["anchor_reference"] == (
        "normalized_horvitz_thompson_w_over_q_floored"
    )


def test_selection_l2_reaches_the_search_and_binds_reuse() -> None:
    frame = _clone_frame()
    dense = _dense(frame)
    selection = select_uk_dataset_size(
        frame, dense, selection_l2_lambda=0.01, **_COMMON
    )
    assert uk_size_l2_from_options(selection.selection.options) == {
        "lambda": 0.01,
        "anchor": "initial",
        "basis": "chi_square",
    }
    sized = refit_uk_dataset_size(
        frame, dense, selection=selection, selection_l2_lambda=0.01, **_COMMON
    )
    assert sized.receipt["selection_l2"]["anchor_reference"] == "pool_design"
    with pytest.raises(ValueError, match="selection L2: searched under"):
        refit_uk_dataset_size(frame, dense, selection=selection, **_COMMON)
    plain = select_uk_dataset_size(frame, dense, **_COMMON)
    with pytest.raises(ValueError, match="selection L2: searched under None"):
        refit_uk_dataset_size(
            frame, dense, selection=plain, selection_l2_lambda=0.01, **_COMMON
        )


def test_stage_weightings_are_named_rules_on_the_problem_axis() -> None:
    frame = _clone_frame()
    dense = _dense(frame)
    weighting = _weighting(dense)
    sized = refit_uk_dataset_size(
        frame, dense, refit_target_weighting=weighting, **_COMMON
    )
    expected = weighting.loss_weights(dense.problem.names)
    np.testing.assert_array_equal(sized.result.target_loss_weights, expected)
    assert sized.receipt["refit_target_weighting"]["rule"] == "grain_family_equal"
    assert sized.receipt["compact_loss_under_dense_weights"] >= 0.0
    assert "selection_target_weighting" not in sized.receipt
    with pytest.raises(TypeError, match="never a raw vector"):
        refit_uk_dataset_size(
            frame, dense, refit_target_weighting=np.array([0.5, 0.5]), **_COMMON
        )
    reversed_rows = uk_target_rows(
        dense.problem.names[::-1],
        [{"family": "a"}, {"family": "b"}],
        [1.0, 3.0],
        local=[False, False],
    )
    with pytest.raises(ValueError, match="another row axis"):
        refit_uk_dataset_size(
            frame,
            dense,
            refit_target_weighting=UKStageTargetWeighting("uniform", reversed_rows),
            **_COMMON,
        )
    # A search under a weighting must be reused with that same weighting.
    searched = select_uk_dataset_size(
        frame, dense, target_weighting=weighting, **_COMMON
    )
    with pytest.raises(ValueError, match="target loss weights"):
        refit_uk_dataset_size(frame, dense, selection=searched, **_COMMON)
    reused = refit_uk_dataset_size(
        frame,
        dense,
        selection=searched,
        selection_target_weighting=weighting,
        **_COMMON,
    )
    assert reused.receipt["selection_target_weighting"]["rule"] == "grain_family_equal"


def test_held_out_rows_train_at_zero_weight() -> None:
    frame = _clone_frame()
    dense = _dense(frame)
    weighting = _weighting(dense, held_out=(1,))
    sized = refit_uk_dataset_size(
        frame, dense, refit_target_weighting=weighting, **_COMMON
    )
    assert sized.result.target_loss_weights[1] == 0.0
    assert sized.result.target_loss_weights[0] > 0.0
    assert sized.receipt["refit_target_weighting"]["n_held_out"] == 1


def test_a_full_pool_size_refuses_size_stage_settings() -> None:
    frame = _clone_frame()
    dense = _dense(frame)
    full = dict(households=3, epochs=1, learning_rate=0.02, seed=7)
    assert refit_uk_dataset_size(frame, dense, **full).result is dense
    for extra in (
        {"refit_l2_lambda": 0.1},
        {"selection_l2_lambda": 0.1},
        {"refit_target_weighting": _weighting(dense)},
    ):
        with pytest.raises(ValueError, match="full-pool size"):
            refit_uk_dataset_size(frame, dense, **full, **extra)


def test_stored_draw_payload_round_trips() -> None:
    frame = _clone_frame()
    dense = _dense(frame)
    selection = select_uk_dataset_size(frame, dense, **_COMMON)
    draw = draw_uk_dataset_size(
        frame, dense, selection=selection, households=2, seed=7, pi_hi=0.5
    )
    payload = json.dumps(
        {
            "problem_sha256": "abc",
            "method": "exact_count",
            "support": draw.support.tolist(),
            "sampling": draw.sampling,
            "inclusion_probabilities": draw.inclusion_probabilities.tolist(),
            "feasibility": draw.feasibility,
            "seed": draw.seed,
            "pi_hi": draw.pi_hi,
            "probabilities_sha256": draw.probabilities_sha256,
        }
    )
    restored = UKSizeDraw.from_payload(payload, problem_sha256="abc")
    np.testing.assert_array_equal(restored.support, draw.support)
    np.testing.assert_array_equal(
        restored.inclusion_probabilities, draw.inclusion_probabilities
    )
    assert restored.probabilities_sha256 == draw.probabilities_sha256
    with pytest.raises(ValueError, match="another ordered problem"):
        UKSizeDraw.from_payload(payload, problem_sha256="other")
    full = json.dumps({"problem_sha256": "abc", "method": "full_pool", "support": []})
    assert UKSizeDraw.from_payload(full, problem_sha256="abc") is None
    with pytest.raises(ValueError, match="completed exact-count draw"):
        UKSizeDraw.from_payload(
            json.dumps({"problem_sha256": "abc", "method": "other"}),
            problem_sha256="abc",
        )


def test_checkpoint_binds_the_selection_l2_and_refuses_weighted_searches(
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.size_checkpoint import (
        load_uk_size_checkpoint,
        uk_size_checkpoint_identity,
        write_uk_size_checkpoint,
    )

    frame = _clone_frame()
    dense = _dense(frame)
    plain = select_uk_dataset_size(frame, dense, **_COMMON)
    penalized = select_uk_dataset_size(
        frame, dense, selection_l2_lambda=0.01, **_COMMON
    )
    l2_identity = {
        "run": "x",
        "selection_l2": uk_size_l2("selection", l2_lambda=0.01).as_dict(),
    }
    write_uk_size_checkpoint(
        tmp_path / "plain",
        frame=frame,
        dense=dense,
        selection=plain,
        identity={"run": "x"},
    )
    write_uk_size_checkpoint(
        tmp_path / "l2",
        frame=frame,
        dense=dense,
        selection=penalized,
        identity=l2_identity,
    )
    targets = TargetSet(list(dense.problem.targets))
    restored = load_uk_size_checkpoint(
        tmp_path / "l2", frame=frame, target_set=targets, identity=l2_identity
    )
    assert (
        uk_size_l2_from_options(restored.selection.selection.options)["lambda"] == 0.01
    )
    # An L2 run cannot import an unpenalized search, nor the reverse.
    with pytest.raises(ValueError, match="selection_l2: absent in checkpoint"):
        load_uk_size_checkpoint(
            tmp_path / "plain", frame=frame, target_set=targets, identity=l2_identity
        )
    with pytest.raises(ValueError, match="searched under L2"):
        load_uk_size_checkpoint(
            tmp_path / "l2", frame=frame, target_set=targets, identity={"run": "x"}
        )
    # The writer refuses an identity that contradicts the search, and a search
    # run under other loss weights than the dense solve's.
    with pytest.raises(ValueError, match="disagrees with the L2 penalty"):
        write_uk_size_checkpoint(
            tmp_path / "bad",
            frame=frame,
            dense=dense,
            selection=penalized,
            identity={"run": "x"},
        )
    weighted = select_uk_dataset_size(
        frame, dense, target_weighting=_weighting(dense), **_COMMON
    )
    with pytest.raises(ValueError, match="one loss-weight vector"):
        write_uk_size_checkpoint(
            tmp_path / "weighted",
            frame=frame,
            dense=dense,
            selection=weighted,
            identity={"run": "x"},
        )
    # The identity gains the key only when the selection penalty is on.
    assert "selection_l2" not in _identity(
        uk_size_checkpoint_identity, SimpleNamespace()
    )
    assert _identity(
        uk_size_checkpoint_identity, SimpleNamespace(selection_l2_lambda=0.02)
    )["selection_l2"] == {"lambda": 0.02, "anchor": "initial", "basis": "chi_square"}


def _identity(builder, extra):
    from argparse import Namespace

    from microcosm.build.uk_runtime.rowwise_posture import UK_ROWWISE_DENSE_POSTURE

    args = Namespace(
        _posture=UK_ROWWISE_DENSE_POSTURE,
        ledger_facts_sha256="f",
        ledger_manifest_sha256="m",
        seed=7,
        selection_seed=None,
        n_clones=2,
        dataset_households=2,
        epochs=2,
        learning_rate=0.02,
        sample_fraction=1.0,
        sample_seed=1,
        source_lineage_modulus=None,
        target_weight_rule="grain_equal",
        engine_blocks=1,
        measure_exclusions=None,
        **vars(extra),
    )
    return builder(
        args,
        pins={"dataset": {"sha256": "d"}, "ladder": {"sha256": "l"}},
        source_year=2024,
    )
