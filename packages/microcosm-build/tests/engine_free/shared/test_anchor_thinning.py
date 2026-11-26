"""Keyed thinning of reported take-up anchors to a residual count (#1720)."""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.anchor_thinning import (
    anchor_keep_uniforms,
    thin_anchors_to_residuals,
)
from microcosm.build.stochastic_assignment import stable_identity_uniforms

STREAM = "test_anchor_keep"


def _rows(n: int, *, groups=("06", "36"), seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "groups": np.asarray(groups)[np.arange(n) % len(groups)],
        "weights": rng.uniform(0.5, 2.0, n),
        "counted": rng.random(n) < 0.8,
        "fixed": np.zeros(n, dtype=bool),
        "anchors": rng.random(n) < 0.5,
        "keys": np.asarray([f"2024HU{i:07d}:1" for i in range(n)], dtype=object),
    }


def _thin(rows, residuals, *, seed: int = 7, stream: str = STREAM):
    return thin_anchors_to_residuals(
        **rows, residuals=residuals, seed=seed, stream=stream
    )


def _mass(rows, selected, group) -> float:
    chosen = selected & rows["counted"] & (rows["groups"] == group)
    return float(rows["weights"][chosen].sum())


def test_anchors_that_fit_the_residual_are_all_kept() -> None:
    rows = _rows(400)
    anchored = {group: _mass(rows, rows["anchors"], group) for group in ("06", "36")}
    result = _thin(rows, {group: mass + 1.0 for group, mass in anchored.items()})
    assert result.keep_probability == {"06": 1.0, "36": 1.0}
    assert result.thinned_groups == []
    assert np.array_equal(result.kept, rows["anchors"])
    assert result.anchor_mass == pytest.approx(anchored)


def test_the_keep_probability_meets_the_residual_after_the_fixed_mass() -> None:
    rows = _rows(400)
    rows["fixed"] = ~rows["anchors"] & (np.arange(400) % 7 == 0)
    fixed = {group: _mass(rows, rows["fixed"], group) for group in ("06", "36")}
    anchored = {group: _mass(rows, rows["anchors"], group) for group in ("06", "36")}
    residuals = {"06": fixed["06"] + 0.25 * anchored["06"], "36": 0.5 * fixed["36"]}
    result = _thin(rows, residuals)
    assert result.keep_probability["06"] == pytest.approx(0.25)
    # Fixed rows alone exceed the residual: no anchor is kept there.
    assert result.keep_probability["36"] == 0.0
    assert result.thinned_groups == ["06", "36"]
    assert not result.kept[rows["groups"] == "36"].any()
    assert result.fixed_mass == pytest.approx(fixed)
    # Only anchors are kept; a fixed row is not an anchor.
    assert not (result.kept & ~rows["anchors"]).any()


def test_uncounted_anchors_are_thinned_with_their_group() -> None:
    rows = _rows(4_000)
    residual = 0.5 * _mass(rows, rows["anchors"], "06")
    result = _thin(rows, {"06": residual})
    uncounted = rows["anchors"] & ~rows["counted"] & (rows["groups"] == "06")
    assert 0.4 < result.kept[uncounted].mean() < 0.6
    # A group without a residual keeps every anchor.
    other = rows["anchors"] & (rows["groups"] == "36")
    assert result.kept[other].all()
    assert "36" not in result.keep_probability


def test_the_kept_mass_meets_the_residual_in_expectation() -> None:
    rows = _rows(200_000, groups=("06",))
    rows["weights"] = np.ones(200_000)
    anchored = _mass(rows, rows["anchors"], "06")
    result = _thin(rows, {"06": 0.5 * anchored})
    kept = _mass(rows, result.kept, "06")
    # Independent keep draws: the relative noise is about 1 / sqrt(n p / (1 - p)),
    # here near 0.3%, so the kept mass sits well within 2% of the residual.
    assert kept == pytest.approx(0.5 * anchored, rel=0.02)


def test_a_record_keeps_its_result_in_any_row_order() -> None:
    rows = _rows(2_000)
    residuals = {
        group: 0.4 * _mass(rows, rows["anchors"], group) for group in ("06", "36")
    }
    result = _thin(rows, residuals)
    order = np.random.default_rng(1).permutation(2_000)
    shuffled = _thin({name: values[order] for name, values in rows.items()}, residuals)
    assert np.array_equal(shuffled.kept, result.kept[order])
    assert shuffled.keep_probability == pytest.approx(result.keep_probability)
    # Another seed or another stream keeps other records.
    assert not np.array_equal(_thin(rows, residuals, seed=8).kept, result.kept)
    assert not np.array_equal(_thin(rows, residuals, stream="other").kept, result.kept)


def test_the_keep_draw_is_the_shared_keyed_uniform() -> None:
    keys = ["2024HU0000001:1", "2024HU0000002:3"]
    assert np.array_equal(
        anchor_keep_uniforms(keys, seed=5, stream=STREAM),
        stable_identity_uniforms(keys, seed=5, salt=STREAM),
    )


def test_misaligned_rows_are_refused() -> None:
    rows = _rows(10)
    rows["weights"] = rows["weights"][:9]
    with pytest.raises(ValueError, match="aligned"):
        _thin(rows, {"06": 1.0})
