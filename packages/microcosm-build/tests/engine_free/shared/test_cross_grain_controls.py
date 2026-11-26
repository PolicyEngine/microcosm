"""The controls, allocators and checks beside the cross-grain operator.

``microcosm.build.cross_grain_controls`` holds what a country's sources can
need around the operator (microcosm#1760). The allocators must give, bit for
bit, the arithmetic they replace; the ``_old_*`` functions here state that
arithmetic as it was written before the operator ran it.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
import struct

import numpy as np
import pandas as pd
import pytest

from microcosm.build import cross_grain_controls as controls
from microcosm.build.cross_grain import (
    CrossGrainRule,
    apply_cross_grain_reconciliation,
)

SEEDS = range(200)


def _bits(values) -> bytes:
    return b"".join(struct.pack("<d", float(value)) for value in values)


def _block(seed: int, *, signed: bool = False) -> tuple[np.ndarray, float]:
    """One leg's rows and its control, sometimes with equal or zero rows."""

    rng = np.random.default_rng(seed)
    size = int(rng.integers(1, 60))
    values = rng.lognormal(mean=rng.uniform(0.0, 14.0), sigma=2.0, size=size)
    if rng.random() < 0.3:
        values = np.round(values)
    if size > 2 and rng.random() < 0.3:
        values[int(rng.integers(size))] = values.max()  # equal largest rows
    if size > 1 and rng.random() < 0.3:
        values[int(rng.integers(size))] = 0.0
    if signed and rng.random() < 0.5:
        values[int(rng.integers(size))] *= -0.25
    if not values.any():
        values[0] = 1.0
    parent = float(values.sum() * rng.uniform(0.5, 2.0))
    if rng.random() < 0.2:
        parent = float(round(parent))
    return values, parent


def _old_proportional(values: list[float], parent: float):
    child_sum = math.fsum(float(value) for value in values)
    factor = 1.0 if child_sum == 0.0 else parent / child_sum
    return [float(value) * factor for value in values], factor, child_sum


def _old_share_rows(values: list[float], parent: float):
    total = math.fsum(values)
    shares = [value / total for value in values]
    largest = max(range(len(values)), key=lambda index: shares[index])
    shares[largest] += 1.0 - math.fsum(shares)
    targets = [parent * share for share in shares]
    targets[largest] += parent - math.fsum(targets)
    return shares, targets, total


def _old_scaled_rows(values: list[float], parent: float):
    total = math.fsum(values)
    scaled = [float(parent) * value / total for value in values]
    largest = max(range(len(values)), key=lambda index: scaled[index])
    scaled[largest] += float(parent) - math.fsum(scaled)
    return scaled, total


def test_proportional_fsum_is_the_old_rebase_bit_for_bit() -> None:
    for seed in SEEDS:
        values, parent = _block(seed, signed=True)
        new_values, factor, raw_sum = controls.proportional_fsum(values, parent)
        old_values, old_factor, old_sum = _old_proportional(values.tolist(), parent)
        assert _bits(new_values) == _bits(old_values), seed
        assert _bits([factor, raw_sum]) == _bits([old_factor, old_sum]), seed
        # The factor does not depend on the order of the rows.
        shuffled = np.random.default_rng(seed).permutation(values)
        assert controls.proportional_fsum(shuffled, parent)[1:] == (factor, raw_sum)


def test_share_residual_to_largest_is_the_old_share_rows_bit_for_bit() -> None:
    for seed in SEEDS:
        values, parent = _block(seed)
        new_values, factor, raw_sum = controls.share_residual_to_largest(values, parent)
        shares, targets, total = _old_share_rows(values.tolist(), parent)
        assert _bits(new_values) == _bits(targets), seed
        assert raw_sum == total and factor == parent / total, seed
        new_shares, new_total, _largest = controls.residual_shares(values.tolist())
        assert _bits(new_shares) == _bits(shares) and new_total == total, seed
        assert math.isclose(math.fsum(new_shares), 1.0, rel_tol=0.0, abs_tol=1e-15)
        assert math.isclose(math.fsum(new_values), parent, rel_tol=1e-15), seed


def test_scale_residual_to_largest_is_the_old_scaled_rows_bit_for_bit() -> None:
    for seed in SEEDS:
        values, parent = _block(seed)
        new_values, factor, raw_sum = controls.scale_residual_to_largest(values, parent)
        scaled, total = _old_scaled_rows(values.tolist(), parent)
        assert _bits(new_values) == _bits(scaled), seed
        assert raw_sum == total and factor == parent / total, seed
        assert math.isclose(math.fsum(new_values), parent, rel_tol=1e-15), seed


def test_the_residual_goes_to_the_first_of_equal_largest_rows() -> None:
    values = np.array([1.0, 3.0, 3.0, 0.1])
    shares, total, largest = controls.residual_shares(values.tolist())
    assert (total, largest) == (7.1, 1)
    untouched = [value / total for value in values.tolist()]
    assert [shares[0], shares[2], shares[3]] == [
        untouched[0],
        untouched[2],
        untouched[3],
    ]
    assert math.fsum(shares) == 1.0


@pytest.mark.parametrize("name", sorted(controls.ALLOCATORS))
def test_an_allocator_keeps_rows_that_sum_to_zero(name: str) -> None:
    values = np.array([0.0, 0.0, 0.0])
    new_values, factor, raw_sum = controls.ALLOCATORS[name](values, 5.0)
    assert new_values.tolist() == [0.0, 0.0, 0.0]
    assert (factor, raw_sum) == (1.0, 0.0)
    # The operator then refuses the vanishing leg under a nonzero control.
    surface = pd.DataFrame(
        [("higher", "P", "control", 5.0)]
        + [("lower", f"P{index}", "rows", 0.0) for index in range(3)],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="vanishing lower-leg total"):
        apply_cross_grain_reconciliation(
            surface, ("control",), _SIGNATURES, _rule(controls.ALLOCATORS[name])
        )


def test_a_declaration_names_an_allocator() -> None:
    assert sorted(controls.ALLOCATORS) == [
        "proportional_fsum",
        "scale_residual_to_largest",
        "share_residual_to_largest",
    ]
    assert controls.allocator_named("proportional_fsum") is controls.proportional_fsum
    with pytest.raises(ValueError, match="Unknown cross-grain allocator 'numpy_sum'"):
        controls.allocator_named("numpy_sum")


_SIGNATURES = {
    "control": {"measurement": {"concept": "quantity"}},
    "rows": {"measurement": {"concept": "quantity"}},
}


def _rule(allocator) -> CrossGrainRule:
    return CrossGrainRule(
        grain_precedence=("higher", "lower"),
        signature_fields=("concept",),
        bridges=(),
        leg_of_area=lambda area: area[0],
        parent_geography_legs={"P": ("P",)},
        allocator=allocator,
    )


@pytest.mark.parametrize("name", sorted(controls.ALLOCATORS))
def test_the_operator_runs_each_allocator(name: str) -> None:
    allocator = controls.ALLOCATORS[name]
    for seed in range(40):
        values, parent = _block(seed)
        surface = pd.DataFrame(
            [("higher", "P", "control", parent)]
            + [
                ("lower", f"P{index}", "rows", value)
                for index, value in enumerate(values.tolist())
            ],
            columns=["grain", "geography_id", "target_id", "value"],
        )
        reconciled, receipt = apply_cross_grain_reconciliation(
            surface, ("control",), _SIGNATURES, _rule(allocator)
        )
        direct, factor, raw_sum = allocator(values, parent)
        assert _bits(reconciled["value"].to_numpy()[1:]) == _bits(direct), seed
        (leg,) = receipt["groups"][0]["legs"]
        assert (leg["allocator"], leg["declared_factor"], leg["old_total"]) == (
            name,
            factor,
            raw_sum,
        )


def test_derived_controls() -> None:
    # Exactly rounded: a running float total would lose the 1.0.
    assert controls.sum_of_lower_rows([1e16, 1.0, -1e16]) == 1.0
    assert controls.sum_of_lower_rows([0.1] * 10) == 1.0
    assert controls.sibling_ratio(110.0, 100.0) == 110.0 / 100.0
    scaled, factor = controls.scaled_by_sibling_ratio(40.0, 110.0, 100.0)
    assert (scaled, factor) == (40.0 * (110.0 / 100.0), 110.0 / 100.0)


def test_level_ratio_is_undefined_at_zero_or_across_signs() -> None:
    assert controls.level_ratio(6.0, 4.0) == 1.5
    assert controls.level_ratio(-6.0, -4.0) == 1.5
    assert controls.level_ratio(0.0, 0.0) != controls.level_ratio(0.0, 0.0)  # NaN
    for higher, lower in ((6.0, 0.0), (6.0, -4.0), (-6.0, 4.0), (0.0, 4.0)):
        assert math.isnan(controls.level_ratio(higher, lower)), (higher, lower)
    # Zero over a negative level: neither is positive, so a ratio is defined.
    assert controls.level_ratio(0.0, -4.0) == 0.0


def test_control_ratio_is_the_proportional_factor() -> None:
    for seed in range(40):
        values, parent = _block(seed, signed=True)
        ratio, lower_sum = controls.control_ratio(parent, values.tolist())
        _values, factor, raw_sum = controls.proportional_fsum(values, parent)
        if raw_sum == 0.0:
            assert ratio is None
        else:
            assert (ratio, lower_sum) == (factor, raw_sum)
    assert controls.control_ratio(5.0, [1.0, -1.0]) == (None, 0.0)


def test_median_by_group_reads_finite_ratios_only() -> None:
    ratios = {
        ("a", "x"): 1.0,
        ("b", "x"): 3.0,
        ("c", "x"): math.nan,
        ("d", "x"): math.inf,
        ("a", "y"): -2.0,
        ("b", "y"): 0.0,
        ("c", "y"): 4.0,
        ("a", "z"): math.nan,
    }
    assert controls.median_by_group(ratios, lambda key: key[1]) == {
        "x": 2.0,
        "y": 0.0,
    }
    assert controls.median_by_group(ratios, lambda key: key[1], positive_only=True) == {
        "x": 2.0,
        "y": 4.0,
    }


def test_band_verdict() -> None:
    inside = controls.band_verdict(1.2, 1.0, 1.25)
    assert inside == controls.BandVerdict(True, 1.2, 1.0, 1.2)
    assert controls.band_verdict(1.25, 1.0, 1.25).ok
    assert controls.band_verdict(0.8, 1.0, 1.25).ok
    assert not controls.band_verdict(1.26, 1.0, 1.25).ok
    assert not controls.band_verdict(0.79, 1.0, 1.25).ok
    # No median, a zero median or a ratio that is not finite: no verdict.
    for ratio, median in ((1.0, math.nan), (1.0, 0.0), (math.nan, 1.0), (math.inf, 1)):
        verdict = controls.band_verdict(ratio, median, 1.25)
        assert not verdict.ok and math.isnan(verdict.relative), (ratio, median)
    # An unbounded band still refuses a ratio on the other side of zero.
    assert controls.band_verdict(1e9, 1.0, math.inf).ok
    assert not controls.band_verdict(-1.0, 1.0, math.inf).ok


def test_leg_completeness() -> None:
    complete = controls.leg_completeness(["a", "b"], {"a", "b"})
    assert complete.complete and complete.merged == {}
    report = controls.leg_completeness(["a", "a", "x"], ["a", "b", "c"])
    assert not report.complete
    assert (report.missing, report.unexpected, report.duplicated) == (
        ("b", "c"),
        ("x",),
        ("a",),
    )
    # A declared combined cell covers an absent member through its host.
    merged = controls.leg_completeness(
        ["a", "b"], ["a", "b", "c"], combined_cells={"c": "b", "z": "a"}
    )
    assert merged.complete and merged.merged == {"c": "b"}
    # The host must be present and expected.
    for present, expected in ((["a"], ["a", "b", "c"]), (["a", "b"], ["a", "c"])):
        report = controls.leg_completeness(present, expected, combined_cells={"c": "b"})
        assert "c" in report.missing and report.merged == {}


def test_sum_gap() -> None:
    gap = controls.sum_gap(100.0, 99.0)
    assert (gap.gap, gap.scale, gap.relative) == (1.0, 100.0, 0.01)
    assert gap.within(0.01) and not gap.exceeds(0.01)
    assert gap.exceeds(0.001) and not gap.within(0.001)
    # A total near zero is judged on the absolute gap.
    small = controls.sum_gap(0.0, 0.5)
    assert (small.gap, small.scale) == (0.5, 1.0)
    assert controls.sum_gap(0.0, 0.5, floor=0.25).scale == 0.25
    # A gap that is not a number is neither within nor beyond a tolerance.
    undefined = controls.sum_gap(math.nan, 1.0)
    assert not undefined.within(1.0) and not undefined.exceeds(1.0)


def test_the_module_holds_no_country_and_the_operator_does_not_import_it() -> None:
    from microcosm.build import cross_grain

    source = inspect.getsource(controls)
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert imported <= {"__future__", "collections", "dataclasses", "math", "numpy"} | {
        "typing"
    }
    named = re.findall(
        r"\b(us|uk|us_runtime|uk_runtime|irs|soi|census|hmrc)\b", source.lower()
    )
    assert named == []
    assert "cross_grain_controls" not in inspect.getsource(cross_grain)
