"""UK local solve doctrine: one uniform operator, declared bounds (#495 inc 4).

Doctrine (#492/#493): no per-target calibration knobs. The doctrine solve
exposes no per-target weight or scale parameters at all; its loss cap and
weight-ratio stretch are declared constants that a change must touch (and
therefore review); and the past-cap census makes the rows a solve wrote off
first-class diagnostics instead of silent triage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import (
    UK_LOCAL_MAX_WEIGHT_RATIO,
    UK_LOCAL_SOLVE_DOCTRINE,
    UK_LOCAL_TARGET_LOSS_CAP,
    UKLocalSolveDoctrine,
    past_cap_census,
    uk_local_doctrine_with_overrides,
    uk_local_target_loss_weights,
)


def test_past_cap_census_classifies_every_transition() -> None:
    targets = np.array([100.0, 100.0, 100.0, 100.0])
    initial = np.array([100.0, 350.0, 300.0, 90.0])
    final = np.array([102.0, 120.0, 320.0, 350.0])
    frame = pd.DataFrame(
        {
            "target_index": [0, 1, 2, 3],
            "area_code": ["A", "B", "C", "D"],
            "metric": ["m", "m", "m", "m"],
        }
    )
    census = past_cap_census(
        initial,
        final,
        targets,
        target_loss_cap=1.0,
        target_frame=frame,
    )
    assert census["target_loss_cap"] == 1.0
    assert census["n_targets"] == 4
    assert census["past_at_init"] == 2
    assert census["past_at_final"] == 2
    assert census["escaped"] == 1
    assert census["frozen"] == 1
    assert census["pushed_out"] == 1
    (pushed,) = census["pushed_out_rows"]
    assert pushed["area_code"] == "D"
    assert pushed["initial_abs_relative_error"] == pytest.approx(0.1)
    assert pushed["final_abs_relative_error"] == pytest.approx(2.5)


def test_past_cap_census_defaults_scales_and_validates_shapes() -> None:
    targets = np.array([10.0, 0.0])
    # Default scale is max(|target|, 1): row 0 scale 10, row 1 scale 1.
    census = past_cap_census(
        np.array([10.0, 5.0]),
        np.array([25.0, 0.0]),
        targets,
        target_loss_cap=1.0,
    )
    assert census["past_at_init"] == 1  # |5-0|/1 = 5 > 1
    assert census["past_at_final"] == 1  # |25-10|/10 = 1.5 > 1
    assert census["pushed_out"] == 1
    assert census["escaped"] == 1
    assert census["pushed_out_rows"][0]["target_index"] == 0

    with pytest.raises(ValueError, match="align"):
        past_cap_census(
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            target_loss_cap=1.0,
        )


def test_doctrine_constants_are_the_declared_contract() -> None:
    assert UK_LOCAL_TARGET_LOSS_CAP == 10.0
    assert UK_LOCAL_MAX_WEIGHT_RATIO == 100.0
    assert UK_LOCAL_SOLVE_DOCTRINE.target_loss_cap == UK_LOCAL_TARGET_LOSS_CAP
    assert UK_LOCAL_SOLVE_DOCTRINE.max_weight_ratio == UK_LOCAL_MAX_WEIGHT_RATIO
    assert UK_LOCAL_SOLVE_DOCTRINE.scale_rule == "default_target_loss_scales"
    assert UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule == "uniform"
    assert UKLocalSolveDoctrine(
        target_weight_rule="grain_equal"
    ).target_weight_rule == ("grain_equal")


def test_grain_equal_weights_give_each_grain_one_equal_share() -> None:
    labels = ["national", "national", "constituency", "la", "la", "la"]
    weights = uk_local_target_loss_weights(labels, rule="grain_equal")
    assert weights is not None
    assert weights[:2].sum() == pytest.approx(1 / 3)
    assert weights[2:3].sum() == pytest.approx(1 / 3)
    assert weights[3:].sum() == pytest.approx(1 / 3)
    assert uk_local_target_loss_weights(labels, rule="uniform") is None


def test_local_doctrine_override_is_receipted_and_bounds_are_frozen() -> None:
    doctrine, receipt = uk_local_doctrine_with_overrides(
        UK_LOCAL_SOLVE_DOCTRINE,
        {"target_weight_rule": "grain_equal"},
    )
    assert doctrine.target_weight_rule == "grain_equal"
    assert receipt == {
        "target_weight_rule": {"default": "uniform", "effective": "grain_equal"}
    }
    with pytest.raises(ValueError, match="reviewed constants"):
        uk_local_doctrine_with_overrides(
            UK_LOCAL_SOLVE_DOCTRINE,
            {"target_loss_cap": 2.0},
        )


def test_doctrine_rejects_tampered_bounds() -> None:
    with pytest.raises(ValueError, match="target_loss_cap"):
        UKLocalSolveDoctrine(target_loss_cap=0.0)
    with pytest.raises(ValueError, match="target_loss_cap"):
        UKLocalSolveDoctrine(target_loss_cap=float("nan"))
    with pytest.raises(ValueError, match="max_weight_ratio"):
        UKLocalSolveDoctrine(max_weight_ratio=0.5)
    with pytest.raises(ValueError, match="scale_rule"):
        UKLocalSolveDoctrine(scale_rule="bespoke")
    with pytest.raises(ValueError, match="target_weight_rule"):
        UKLocalSolveDoctrine(target_weight_rule="per_target")


def test_census_boundary_and_scale_refusals() -> None:
    targets = np.array([100.0, 100.0])
    # A row exactly AT the cap is not past it: strictly-greater semantics
    # match the torch loss, where a tie still carries gradient.
    census = past_cap_census(
        np.array([200.0, 100.0]),
        np.array([200.0, 100.0]),
        targets,
        target_loss_cap=1.0,
    )
    assert census["past_at_init"] == 0
    assert census["past_at_final"] == 0

    with pytest.raises(ValueError, match="finite and positive"):
        past_cap_census(
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            targets,
            target_loss_cap=1.0,
            target_loss_scales=np.array([1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="estimates must be finite"):
        past_cap_census(
            np.array([np.nan, 1.0]),
            np.array([1.0, 1.0]),
            targets,
            target_loss_cap=1.0,
        )


def test_census_lists_every_pushed_out_row_unless_bounded() -> None:
    n = 130
    targets = np.full(n, 100.0)
    initial = np.full(n, 100.0)
    final = np.full(n, 100.0 + 100.0 * 5.0)
    unbounded = past_cap_census(initial, final, targets, target_loss_cap=1.0)
    assert unbounded["pushed_out"] == n
    assert len(unbounded["pushed_out_rows"]) == n
    assert unbounded["pushed_out_rows_truncated"] is False

    bounded = past_cap_census(
        initial, final, targets, target_loss_cap=1.0, max_listed_rows=100
    )
    assert len(bounded["pushed_out_rows"]) == 100
    assert bounded["pushed_out_rows_truncated"] is True
