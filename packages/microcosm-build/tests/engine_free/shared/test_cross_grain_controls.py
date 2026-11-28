"""The row-level pass and the allocators beside the cross-grain operator.

``microcosm.build.cross_grain_controls`` runs the operator on target rows and
says what it did with each row (microcosm#1760). Its allocators must give, bit
for bit, the arithmetic they replace; the ``_old_*`` functions here state that
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
    EXACT_SIGNATURE_CONTROL,
    CrossGrainBridge,
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


def test_sum_of_lower_rows_is_exactly_rounded() -> None:
    # A running float total would lose the 1.0.
    assert controls.sum_of_lower_rows([1e16, 1.0, -1e16]) == 1.0
    assert controls.sum_of_lower_rows([0.1] * 10) == 1.0


def _row_rule(*bridges: CrossGrainBridge) -> CrossGrainRule:
    return CrossGrainRule(
        grain_precedence=("nation", "state", "district"),
        signature_fields=("concept",),
        bridges=bridges,
        leg_of_area=lambda area: area[0],
        parent_geography_legs={state: (state,) for state in "ABCDE"},
        control_grains=("nation", "state"),
    )


def _rows(target: str, factors: dict[str, float]) -> list[controls.SurfaceRow]:
    """A state total and two districts in each state; the factor is their gap."""

    rows = []
    for state, factor in factors.items():
        rows.append(controls.SurfaceRow("state", state, "total", 100.0 * factor))
        rows.append(controls.SurfaceRow("district", f"{state}1", target, 30.0))
        rows.append(controls.SurfaceRow("district", f"{state}2", target, 70.0))
    return rows


def test_reconcile_rows_names_the_control_of_each_row() -> None:
    """Rows of the control's own measurement: no bridge, one parent row."""

    signatures = {
        "total": {"measurement": {"concept": "quantity"}},
        "parts": {"measurement": {"concept": "quantity"}},
    }
    rows = _rows("parts", {"A": 1.1, "B": 0.5})
    outcomes, receipt = controls.reconcile_rows(
        rows, ("total",), signatures, _row_rule()
    )
    assert len(outcomes) == len(rows)
    assert [outcome.status for outcome in outcomes] == [
        controls.UNTOUCHED,
        controls.RECONCILED,
        controls.RECONCILED,
    ] * 2
    assert [outcome.value for outcome in outcomes] == pytest.approx(
        [110.0, 33.0, 77.0, 50.0, 15.0, 35.0]
    )
    # A control row keeps its value and names nothing.
    assert outcomes[0] == controls.RowOutcome(value=rows[0].value)
    for position, parent, factor in ((1, 0, 1.1), (2, 0, 1.1), (4, 3, 0.5)):
        outcome = outcomes[position]
        assert outcome.control == EXACT_SIGNATURE_CONTROL
        assert outcome.parent == parent
        assert outcome.factor == pytest.approx(factor)
        assert outcome.lower_total == 100.0
    assert len(receipt["groups"]) == 1


def test_reconcile_rows_reads_a_bridge_and_its_unbound_legs() -> None:
    """Two sources: the bridge is the control, and a leg out of bound unbinds."""

    signatures = {
        "total": {"measurement": {"concept": "source one"}},
        "parts": {"measurement": {"concept": "source two"}},
    }
    bridge = CrossGrainBridge(
        "total_vs_parts",
        concept="one quantity",
        higher_target_ids=("total",),
        lower_side="parts",
        max_factor_shift=0.25,
        min_factor_shift=-0.2,
        shift_around="median",
        out_of_bound="unbind",
    )
    factors = {"A": 2.0, "B": 2.0, "C": 2.0, "D": 2.0, "E": 4.0}
    rows = _rows("parts", factors)
    outcomes, receipt = controls.reconcile_rows(
        rows, ("total",), signatures, _row_rule(bridge)
    )
    for index, state in enumerate(factors):
        total, first, second = outcomes[3 * index : 3 * index + 3]
        assert total.status == controls.UNTOUCHED
        assert first.control == second.control == "total_vs_parts"
        assert first.parent == second.parent == 3 * index
        if state == "E":
            # The leg keeps its values. The caller takes it off its surface.
            assert first.status == second.status == controls.UNBOUND
            assert (first.value, second.value) == (30.0, 70.0)
            assert first.factor == pytest.approx(4.0)
        else:
            assert first.status == second.status == controls.RECONCILED
            assert (first.value, second.value) == pytest.approx((60.0, 140.0))
            assert first.factor == pytest.approx(2.0)
            assert first.lower_total == 100.0
    assert [leg["leg"] for leg in receipt["unbound_legs"]] == ["E"]


def test_reconcile_rows_leaves_a_row_that_no_group_covers() -> None:
    signatures = {
        "total": {"measurement": {"concept": "quantity"}},
        "parts": {"measurement": {"concept": "quantity"}},
        "other": {"measurement": {"concept": "another quantity"}},
    }
    rows = [
        *_rows("parts", {"A": 1.0}),
        controls.SurfaceRow("district", "A1", "other", 7.0),
    ]
    outcomes, _receipt = controls.reconcile_rows(
        rows, ("total",), signatures, _row_rule()
    )
    assert outcomes[-1] == controls.RowOutcome(value=7.0)


def test_reconcile_rows_refuses_a_row_under_two_controls() -> None:
    """A row names one control and one factor."""

    signatures = {
        "bridged": {"measurement": {"concept": "source one"}},
        "total": {"measurement": {"concept": "quantity"}},
        "parts": {"measurement": {"concept": "quantity"}},
    }
    bridge = CrossGrainBridge(
        "bridged_vs_parts",
        concept="one quantity",
        higher_target_ids=("bridged",),
        lower_side="parts",
    )
    rows = [
        controls.SurfaceRow("state", "A", "bridged", 120.0),
        controls.SurfaceRow("state", "A", "total", 120.0),
        controls.SurfaceRow("district", "A1", "parts", 20.0),
        controls.SurfaceRow("district", "A2", "parts", 40.0),
    ]
    with pytest.raises(ValueError, match="a row names one control"):
        controls.reconcile_rows(
            rows, ("bridged", "total"), signatures, _row_rule(bridge)
        )


def test_reconcile_rows_refuses_a_leg_id_that_holds_a_plus() -> None:
    """The operator's receipt joins the legs of one control with ``+``."""

    signatures = {
        "total": {"measurement": {"concept": "quantity"}},
        "parts": {"measurement": {"concept": "quantity"}},
    }
    rule = CrossGrainRule(
        grain_precedence=("nation", "state", "district"),
        signature_fields=("concept",),
        bridges=(),
        leg_of_area=lambda area: "A+B",
        parent_geography_legs={"A+B": ("A+B",)},
        control_grains=("nation", "state"),
    )
    rows = [
        controls.SurfaceRow("state", "A+B", "total", 120.0),
        controls.SurfaceRow("district", "A+B-1", "parts", 20.0),
        controls.SurfaceRow("district", "A+B-2", "parts", 40.0),
    ]
    with pytest.raises(ValueError, match=r"a leg id that holds '\+'"):
        controls.reconcile_rows(rows, ("total",), signatures, rule)


def test_the_module_holds_no_country_and_the_operator_does_not_import_it() -> None:
    from microcosm.build import cross_grain

    source = inspect.getsource(controls)
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert imported <= {
        "__future__",
        "collections.abc",
        "dataclasses",
        "math",
        "typing",
        "numpy",
        "pandas",
        "microcosm.build.cross_grain",
    }
    named = re.findall(
        r"\b(us|uk|us_runtime|uk_runtime|irs|soi|census|hmrc)\b", source.lower()
    )
    assert named == []
    assert "cross_grain_controls" not in inspect.getsource(cross_grain)
