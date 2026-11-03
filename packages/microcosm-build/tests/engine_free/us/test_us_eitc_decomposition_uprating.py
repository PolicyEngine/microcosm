"""Cross-period uprating of the SOI Table 2.5 EITC child decomposition.

Each child split's record set carries its AGI bins and an all-AGI total row
that sums them. The uprating first scales a stale row by (active EITC in its
AGI group) / (source-period EITC in that group, summed over child splits).
That source sum must count each split once: a split's total row and its bins
describe the same returns. With AGI groups that grow at different rates, the
total rows (all-AGI child split) and the bins (each group's child split) then
disagree, so the bins are raked (iterative proportional fitting) to both
margins: the splits' total rows and the active AGI groups.

Invariants exercised by the property tests (decompositions with bins that tile
AGI and total rows equal to their bins, active filing-season AGI bins that
coarsen the Table 2.5 bins):

* conservation: the uprated total rows sum to the active EITC, and so do the
  uprated bins;
* the total rows keep the source's all-AGI child split: every total row takes
  active EITC / source total rows;
* each split's bins sum to its total row;
* each active AGI group's bins, over all splits, sum to the group's EITC;
* every bin of one split in one AGI group takes one factor, so the source's
  shape inside the cell is kept;
* the rake keeps the source's cross-product ratios between splits and groups
  (what makes it iterative proportional fitting and not some other
  reconciliation);
* when every group grows by the same factor the rake changes nothing.

``_rake_two_margins`` is also checked against the closed-form 2 x 2 solution.
"""

from __future__ import annotations

import itertools
import math
import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.optimize import linprog

from microcosm.build.us_runtime.fiscal_targets import _rake_two_margins
from test_support.microcosm_build.us_fiscal_targets import (
    _soi_eitc_child_fact,
    _soi_eitc_child_total_facts,
    _soi_eitc_filing_season_agi_fact,
    _soi_eitc_filing_season_total_fact,
    compile_us_fiscal_target_registry,
    packaged_reference_facts,
)

_SPLITS = (
    "no_qualifying_children",
    "one_qualifying_child",
    "two_qualifying_children",
    "three_or_more_qualifying_children",
)
# (Table 2.5 measure, active filing-season measure, uprating index)
_KINDS = {
    "amount": ("eitc_total", "total_earned_income_credit_amount", "agi_eitc_amount"),
    "returns": (
        "eitc_returns",
        "total_earned_income_credit_returns",
        "agi_eitc_returns",
    ),
}
_CUTS = (1, 1_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 50_000)
_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _bins(cuts: tuple[float, ...]) -> list[tuple[float | None, float | None]]:
    edges: list[float | None] = [None, *cuts, None]
    return list(zip(edges[:-1], edges[1:], strict=True))


def _bin_id(lower: float | None, upper: float | None) -> str:
    return f"{'under' if lower is None else int(lower)}_to_{'up' if upper is None else int(upper)}"


def _decomposition_facts(
    *,
    kind: str,
    source: dict[str, dict[tuple[float | None, float | None], float]],
    active: dict[tuple[float | None, float | None], float],
    total_rows: bool = True,
) -> list[dict[str, object]]:
    """TY2023 Table 2.5 rows (bins plus total rows) and TY2024 active bins."""

    measure, active_measure, _ = _KINDS[kind]
    facts: list[dict[str, object]] = [
        *packaged_reference_facts(),
        _soi_eitc_filing_season_total_fact(
            measure_id=active_measure, value=sum(active.values())
        ),
    ]
    if total_rows:
        facts.extend(
            _soi_eitc_child_total_facts(
                2023,
                measure_id=measure,
                values={split: sum(rows.values()) for split, rows in source.items()},
            )
        )
    for (lower, upper), value in active.items():
        facts.append(
            _soi_eitc_filing_season_agi_fact(
                measure_id=active_measure,
                value=value,
                income_range=_bin_id(lower, upper),
                lower=lower,
                upper=upper,
            )
        )
    for split, rows in source.items():
        for (lower, upper), value in rows.items():
            facts.append(_child_fact(measure, split, (lower, upper), value))
    return facts


def _child_fact(measure: str, split: str, bounds, value: float) -> dict[str, object]:
    income_range = _bin_id(*bounds)
    return _soi_eitc_child_fact(
        2023,
        source_record_id=(
            "irs_soi.ty2023.table_2_5.eitc_by_agi_children."
            f"{split}.{income_range}.{measure}"
        ),
        measure_id=measure,
        value=value,
        income_range=income_range,
        child_group=split,
        lower=bounds[0],
        upper=bounds[1],
    )


def _compile(facts: list[dict[str, object]]):
    registry = compile_us_fiscal_target_registry(
        facts, target_period=2024, allow_unaged_dollar_targets=True
    )
    return {spec.name: spec for spec in registry.specs}


def _total_name(split: str, measure: str) -> str:
    return f"irs_soi.ty2023.table_2_5.eitc_by_agi_children.{split}.total.{measure}"


def _bin_name(split: str, measure: str, bounds) -> str:
    return (
        "irs_soi.ty2023.table_2_5.eitc_by_agi_children."
        f"{split}.{_bin_id(*bounds)}.{measure}"
    )


_LOW, _HIGH = (None, 25_000.0), (25_000.0, None)
# Two splits' bins (100 + 300, 100 + 200) under active groups of 500 and 900.
_TWO_BY_TWO = {
    "one_qualifying_child": {_LOW: 100, _HIGH: 300},
    "two_qualifying_children": {_LOW: 100, _HIGH: 200},
}
_TWO_BY_TWO_ACTIVE = {_LOW: 500, _HIGH: 900}


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_all_agi_total_row_counts_each_child_split_once(kind: str) -> None:
    """Both splits' bins and total rows (400, 300) sit inside the all-AGI
    group. The source EITC there is 700, not 1,400, so the 1,400 active EITC
    doubles each total row instead of leaving it at its TY2023 value."""

    measure, _, index = _KINDS[kind]
    by_name = _compile(
        _decomposition_facts(kind=kind, source=_TWO_BY_TWO, active=_TWO_BY_TWO_ACTIVE)
    )

    one = by_name[_total_name("one_qualifying_child", measure)]
    two = by_name[_total_name("two_qualifying_children", measure)]
    assert one.value == 800
    assert two.value == 600
    for total in (one, two):
        assert total.metadata["uprating_index"] == index
        assert total.metadata["uprating_factor"] == "2"
        assert total.metadata["uprating_agi_lower_bound"] == "-inf"
        assert total.metadata["uprating_agi_upper_bound"] == "inf"
        assert total.metadata["uprating_reconciliation"] == "two_margin_rake"
    assert one.value + two.value == 1_400


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_bins_rake_to_their_total_rows_and_the_active_groups(kind: str) -> None:
    """Uprated per group alone, the bins would be 250 + 540 and 250 + 360:
    the one-child bins would sum to 790 against a total row of 800. Raking
    keeps the total rows and the active groups and the source's
    cross-product ratio (100 x 200) / (300 x 100) = 2/3. With ``a`` the
    one-child bin under $25,000, the other bins are 800 - a, 500 - a and
    100 + a, so a (100 + a) = 2/3 (800 - a)(500 - a), that is
    a^2 + 2,900 a - 800,000 = 0."""

    measure, _, index = _KINDS[kind]
    by_name = _compile(
        _decomposition_facts(kind=kind, source=_TWO_BY_TWO, active=_TWO_BY_TWO_ACTIVE)
    )
    a = (-2_900 + math.sqrt(2_900**2 + 4 * 800_000)) / 2  # 253.67...
    expected = {
        ("one_qualifying_child", _LOW): a,
        ("one_qualifying_child", _HIGH): 800 - a,
        ("two_qualifying_children", _LOW): 500 - a,
        ("two_qualifying_children", _HIGH): 100 + a,
    }
    for (split, bounds), value in expected.items():
        spec = by_name[_bin_name(split, measure, bounds)]
        assert spec.value == pytest.approx(value, rel=1e-12)
        assert spec.metadata["uprating_index"] == index
        assert spec.metadata["uprating_reconciliation"] == "two_margin_rake"
        source = _TWO_BY_TWO[split][bounds]
        group_factor = _TWO_BY_TWO_ACTIVE[bounds] / sum(
            rows[bounds] for rows in _TWO_BY_TWO.values()
        )
        assert float(spec.metadata["uprating_factor"]) == pytest.approx(
            value / source, rel=1e-12
        )
        assert float(spec.metadata["uprating_agi_group_factor"]) == pytest.approx(
            group_factor, rel=1e-12
        )
        assert float(spec.metadata["uprating_child_split_adjustment"]) == pytest.approx(
            value / source / group_factor, rel=1e-12
        )


def test_a_subtotal_row_takes_the_sum_of_its_raked_bins() -> None:
    """A row between the bins and the total row (here $25,000 and over, over
    bins $25,000-$50,000 and $50,000 and over) is the sum of its raked bins."""

    measure, _, _ = _KINDS["amount"]
    middle, top = (25_000.0, 50_000.0), (50_000.0, None)
    source = {
        "one_qualifying_child": {_LOW: 100, middle: 200, top: 100},
        "two_qualifying_children": {_LOW: 100, middle: 150, top: 50},
    }
    facts = _decomposition_facts(
        kind="amount", source=source, active={_LOW: 500, _HIGH: 900}
    )
    for split, rows in source.items():
        facts.append(_child_fact(measure, split, _HIGH, rows[middle] + rows[top]))
    by_name = _compile(facts)

    for split in source:
        subtotal = by_name[_bin_name(split, measure, _HIGH)]
        bins = [by_name[_bin_name(split, measure, b)] for b in (middle, top)]
        assert subtotal.value == pytest.approx(sum(b.value for b in bins), rel=1e-12)
        assert subtotal.metadata["uprating_reconciliation"] == "two_margin_rake"
        total = by_name[_total_name(split, measure)].value
        low = by_name[_bin_name(split, measure, _LOW)].value
        assert total == pytest.approx(low + subtotal.value, rel=1e-12)
    # The subtotals are the high group's whole: the bins under it are not
    # counted again.
    assert sum(
        by_name[_bin_name(split, measure, _HIGH)].value for split in source
    ) == pytest.approx(900, rel=1e-12)


_MIDDLE, _TOP = (25_000.0, 50_000.0), (50_000.0, None)


def test_a_group_with_no_eitc_in_either_year_takes_a_zero_factor() -> None:
    """microcosm#1235 review: a $50,000-and-over group with no EITC in either
    year made the per-row uprating fall back to the all-returns total for its
    bins, and that used to skip the rake for the whole decomposition."""

    measure, _, index = _KINDS["amount"]
    source = {
        "one_qualifying_child": {_LOW: 100, _MIDDLE: 300, _TOP: 0},
        "two_qualifying_children": {_LOW: 100, _MIDDLE: 200, _TOP: 0},
    }
    by_name = _compile(
        _decomposition_facts(
            kind="amount", source=source, active={_LOW: 500, _MIDDLE: 900, _TOP: 0}
        )
    )

    for split, rows in source.items():
        total = by_name[_total_name(split, measure)].value
        bins = [by_name[_bin_name(split, measure, bounds)] for bounds in rows]
        assert sum(spec.value for spec in bins) == pytest.approx(total, rel=1e-12)
        top = by_name[_bin_name(split, measure, _TOP)]
        assert top.value == 0
        assert top.metadata["uprating_index"] == index
        assert top.metadata["uprating_agi_lower_bound"] == "50000"
        assert top.metadata["uprating_factor"] == "0"
        assert top.metadata["uprating_agi_group_factor"] == "0"
        assert top.metadata["uprating_child_split_adjustment"] == "1"
        assert {spec.metadata["uprating_reconciliation"] for spec in bins} == {
            "two_margin_rake"
        }


def test_a_group_with_active_but_no_source_eitc_is_refused() -> None:
    """No source bin can carry the $10 of active EITC at $50,000 and over, so
    neither level can hold; the compile refuses rather than drop it."""

    with pytest.raises(ValueError, match="has no source EITC to scale"):
        _compile(
            _decomposition_facts(
                kind="amount",
                source={
                    "one_qualifying_child": {_LOW: 100, _MIDDLE: 300, _TOP: 0},
                    "two_qualifying_children": {_LOW: 100, _MIDDLE: 200, _TOP: 0},
                },
                active={_LOW: 500, _MIDDLE: 900, _TOP: 10},
            )
        )


def test_an_ill_conditioned_decomposition_is_raked() -> None:
    """microcosm#1235 review: the splits sit almost wholly in opposite
    groups, and the groups' margins sit near the edge of what the source
    allows; 10,000 IPF sweeps did not converge on this."""

    measure, _, _ = _KINDS["amount"]
    source = {
        "one_qualifying_child": {_LOW: 10_000, _HIGH: 1},
        "two_qualifying_children": {_LOW: 1, _HIGH: 10_000},
    }
    active = {_LOW: 10_011.001, _HIGH: 9_990.999}
    by_name = _compile(
        _decomposition_facts(kind="amount", source=source, active=active)
    )

    for split, rows in source.items():
        total = by_name[_total_name(split, measure)].value
        assert sum(
            by_name[_bin_name(split, measure, bounds)].value for bounds in rows
        ) == pytest.approx(total, rel=1e-12)
    for bounds, value in active.items():
        assert sum(
            by_name[_bin_name(split, measure, bounds)].value for split in source
        ) == pytest.approx(value, rel=1e-11)


def test_strict_slack_below_rounding_is_raked() -> None:
    """microcosm#1235 review round 2: the one-child split's total falls short
    of the low group by less than 1e-12 of the total. That is strict slack,
    which the two-child split's $1,000 bin there fills; it used to be taken
    for an exact (infeasible) fill and refused."""

    measure, _, _ = _KINDS["amount"]
    source = {
        "one_qualifying_child": {_LOW: 1_048_575_000, _HIGH: 0},
        "two_qualifying_children": {_LOW: 1_000, _HIGH: 1_048_576_000},
    }
    active = {_LOW: 1_048_576_000, _HIGH: 1_048_578_000}
    by_name = _compile(
        _decomposition_facts(kind="amount", source=source, active=active)
    )

    for split, rows in source.items():
        total = by_name[_total_name(split, measure)].value
        assert sum(
            by_name[_bin_name(split, measure, bounds)].value for bounds in rows
        ) == pytest.approx(total, rel=1e-12)
    for bounds, value in active.items():
        assert sum(
            by_name[_bin_name(split, measure, bounds)].value for split in source
        ) == pytest.approx(value, rel=1e-11)


def test_a_zero_subtotal_over_positive_bins_is_refused() -> None:
    """microcosm#1235 review round 2: a published $25,000-and-over subtotal of
    zero over bins of $1 and $1 contradicts itself; the per-row uprating
    found no source EITC in that group, so no factor it records is true."""

    measure, _, _ = _KINDS["amount"]
    source = {
        "one_qualifying_child": {_LOW: 100, _MIDDLE: 1, _TOP: 1},
        "two_qualifying_children": {_LOW: 100, _MIDDLE: 1, _TOP: 1},
    }
    facts = _decomposition_facts(
        kind="amount", source=source, active={_LOW: 500, _HIGH: 10}
    )
    for split in source:
        facts.append(_child_fact(measure, split, _HIGH, 0))

    with pytest.raises(ValueError, match="sum to zero"):
        _compile(facts)


def test_rounded_equality_with_strict_slack_is_raked() -> None:
    """microcosm#1235 review round 3: the one- and two-child splits hold
    2**-24 less than the low group, but a float sum of their totals rounds
    to the group's total exactly; summed exactly, it is slack, which the
    three-or-more split's $1 bin there fills."""

    measure, _, _ = _KINDS["amount"]
    source = {
        "one_qualifying_child": {_LOW: 1_056_964_608, _HIGH: 0},
        "two_qualifying_children": {_LOW: 16_777_215, _HIGH: 0},
        "three_or_more_qualifying_children": {_LOW: 1, _HIGH: 16_777_216},
    }
    active = {_LOW: 1_073_741_887, _HIGH: 16_777_218}
    by_name = _compile(
        _decomposition_facts(kind="amount", source=source, active=active)
    )

    for split, rows in source.items():
        total = by_name[_total_name(split, measure)].value
        assert sum(
            by_name[_bin_name(split, measure, bounds)].value for bounds in rows
        ) == pytest.approx(total, rel=1e-12)
    for bounds, value in active.items():
        assert sum(
            by_name[_bin_name(split, measure, bounds)].value for split in source
        ) == pytest.approx(value, rel=1e-11)


def test_a_zero_subtotal_inside_a_group_is_refused() -> None:
    """microcosm#1235 review round 3: a published zero over the
    $25,000-$40,000 bins ($1 and $1) sits inside the $25,000-and-over
    group, whose own source is not zero. The subtotal would become $3.33 on a
    zero source, so no factor for it would be true."""

    measure, _, _ = _KINDS["amount"]
    first, second, top = (25_000.0, 30_000.0), (30_000.0, 40_000.0), (40_000.0, None)
    source = {
        split: {_LOW: 100, first: 1, second: 1, top: 1}
        for split in ("one_qualifying_child", "two_qualifying_children")
    }
    facts = _decomposition_facts(
        kind="amount", source=source, active={_LOW: 500, _HIGH: 10}
    )
    for split in source:
        facts.append(_child_fact(measure, split, (25_000.0, 40_000.0), 0))

    with pytest.raises(ValueError, match="is zero, but bins inside it are not"):
        _compile(facts)


def test_the_adjustment_is_relative_to_the_recorded_group_factor() -> None:
    """microcosm#1235 review: a published subtotal ($301) can differ from its
    bins ($200 + $100) by rounding. The per-row factor of the $25,000-and-over
    group divides by the subtotal (900 / 501); the adjustment is relative to
    that recorded factor, so the two multiply to the bin's factor."""

    measure, _, _ = _KINDS["amount"]
    source = {
        "one_qualifying_child": {_LOW: 100, _MIDDLE: 200, _TOP: 100},
        "two_qualifying_children": {_LOW: 100, _MIDDLE: 150, _TOP: 50},
    }
    facts = _decomposition_facts(
        kind="amount", source=source, active={_LOW: 500, _HIGH: 900}
    )
    facts.append(_child_fact(measure, "one_qualifying_child", _HIGH, 301))
    facts.append(_child_fact(measure, "two_qualifying_children", _HIGH, 200))
    by_name = _compile(facts)

    for split, rows in source.items():
        for bounds in rows:
            metadata = by_name[_bin_name(split, measure, bounds)].metadata
            assert float(metadata["uprating_agi_group_factor"]) * float(
                metadata["uprating_child_split_adjustment"]
            ) == pytest.approx(float(metadata["uprating_factor"]), rel=1e-12)
        middle = by_name[_bin_name(split, measure, _MIDDLE)].metadata
        assert float(middle["uprating_agi_group_factor"]) == pytest.approx(900 / 501)


def test_a_decomposition_without_total_rows_keeps_its_group_factors() -> None:
    """With no total row there is nothing to rake to: each bin keeps its AGI
    group's factor, 500 / 200 and 900 / 500."""

    measure, _, _ = _KINDS["amount"]
    by_name = _compile(
        _decomposition_facts(
            kind="amount",
            source=_TWO_BY_TWO,
            active=_TWO_BY_TWO_ACTIVE,
            total_rows=False,
        )
    )

    expected = {
        ("one_qualifying_child", _LOW): 250,
        ("one_qualifying_child", _HIGH): 540,
        ("two_qualifying_children", _LOW): 250,
        ("two_qualifying_children", _HIGH): 360,
    }
    for (split, bounds), value in expected.items():
        spec = by_name[_bin_name(split, measure, bounds)]
        assert spec.value == value
        assert "uprating_reconciliation" not in spec.metadata


def test_a_decomposition_on_the_all_returns_total_is_not_raked() -> None:
    """Without active AGI bins every row takes the all-returns factor,
    1,400 / 700, so the levels already agree and the rows keep it."""

    measure, active_measure, _ = _KINDS["amount"]
    facts = [
        *packaged_reference_facts(),
        _soi_eitc_filing_season_total_fact(measure_id=active_measure, value=1_400),
        *_soi_eitc_child_total_facts(
            2023,
            measure_id=measure,
            values={split: sum(rows.values()) for split, rows in _TWO_BY_TWO.items()},
        ),
        *(
            _child_fact(measure, split, bounds, value)
            for split, rows in _TWO_BY_TWO.items()
            for bounds, value in rows.items()
        ),
    ]
    by_name = _compile(facts)

    for split, rows in _TWO_BY_TWO.items():
        total = by_name[_total_name(split, measure)]
        assert total.value == 2 * sum(rows.values())
        assert "uprating_reconciliation" not in total.metadata
        for bounds, value in rows.items():
            spec = by_name[_bin_name(split, measure, bounds)]
            assert spec.value == 2 * value
            assert spec.metadata["uprating_factor"] == "2"
            assert "uprating_reconciliation" not in spec.metadata


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(((None, 30_000.0), (20_000.0, None)), id="crossing"),
        pytest.param(((None, 25_000.0), (None, 25_000.0)), id="identical"),
    ],
)
def test_overlapping_decomposition_rows_are_refused(rows) -> None:
    """A split whose AGI rows cross, or repeat one interval under two ids,
    can't be summed without counting some returns twice, so the compile
    refuses it rather than guess."""

    measure, _, _ = _KINDS["amount"]
    facts = _decomposition_facts(
        kind="amount",
        source={"one_qualifying_child": {_LOW: 100, _HIGH: 300}},
        active=_TWO_BY_TWO_ACTIVE,
    )
    for (lower, upper), income_range in zip(
        rows, ("overlap_a", "overlap_b"), strict=True
    ):
        facts.append(
            _soi_eitc_child_fact(
                2023,
                source_record_id=(
                    "irs_soi.ty2023.table_2_5.eitc_by_agi_children."
                    f"two_qualifying_children.{income_range}.{measure}"
                ),
                measure_id=measure,
                value=10,
                income_range=income_range,
                child_group="two_qualifying_children",
                lower=lower,
                upper=upper,
            )
        )

    with pytest.raises(ValueError, match="has overlapping AGI rows"):
        _compile(facts)


@pytest.mark.parametrize(
    ("seed", "rows", "columns", "message"),
    [
        ([[1.0, -1.0]], [0.0], [1.0, -1.0], "cannot rake negative EITC"),
        ([[1.0, 1.0]], [3.0], [1.0, 1.0], "the rake needs one total"),
        ([[1.0, 0.0], [1.0, 0.0]], [1.0, 1.0], [1.0, 1.0], "has no source EITC"),
        ([[0.0, 0.0], [1.0, 1.0]], [1.0, 1.0], [1.0, 1.0], "has no EITC in any AGI"),
        # Block-diagonal: split one only in group one, split two only in group
        # two, so both margins hold only if they match, and here they don't.
        ([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0], [2.0, 1.0], "meets both margins"),
        # Split two reaches only group two and fills it exactly, so split
        # one's positive bin in group two would have to be zero.
        ([[1.0, 1.0], [0.0, 1.0]], [1.0, 1.0], [1.0, 1.0], "meets both margins"),
        # Seventeen splits: checked over the two groups instead. Group two
        # holds only split one's bin and fills split one exactly, so split
        # one's bin in group one would have to be zero.
        (
            [[1.0, 1.0], *([[1.0, 0.0]] * 16)],
            [1.0] * 17,
            [16.0, 1.0],
            "AGI groups \\[1\\] total 1.0",
        ),
        # Seventeen splits and seventeen groups: too many to check exactly.
        (
            [
                [1.0 if i == j or (i, j) == (1, 0) else 0.0 for j in range(17)]
                for i in range(17)
            ],
            [1.0] * 17,
            [1.0] * 17,
            "at most 16 child splits or AGI groups",
        ),
    ],
)
def test_rake_refuses_margins_it_cannot_meet(seed, rows, columns, message) -> None:
    with pytest.raises(ValueError, match=message):
        _rake_two_margins(seed, rows, columns, label="test")


@pytest.mark.parametrize(
    ("seed", "rows", "columns"),
    [
        # microcosm#1235 review: 10,000 IPF sweeps did not converge on these.
        ([[1e4, 1.0], [1.0, 1e4]], [10_001.0, 10_001.0], [10_011.0, 9_991.0]),
        ([[1e4, 1.0], [1.0, 1e4]], [10_001.0, 10_001.0], [10_011.001, 9_990.999]),
        # Factors spanning about 1e15.
        ([[1e7, 1.0], [1.0, 1e7]], [1e7 + 1, 1e7 + 1], [2e7 + 1.9, 0.1]),
        # Two separate blocks whose margins match.
        ([[1.0, 0.0], [0.0, 1.0]], [2.0, 1.0], [2.0, 1.0]),
        # A group with no EITC in either year.
        ([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], [3.0, 4.0], [5.0, 2.0, 0.0]),
        # Review round 2: split one's 1,048,575,999.999 sits below group one's
        # 1,048,576,000 by less than 1e-12 of the total; it is strict slack,
        # filled by split two's bin there, not an exact (infeasible) fill.
        (
            [[1_048_575_000.0, 0.0], [1_000.0, 1_048_576_000.0]],
            [1_048_575_000.0 * (1 + 2**-20), 1_048_577_000.0 * (1 + 2**-20)],
            [1_048_576_000.0, 1_048_578_000.0],
        ),
        # Review round 2: rounding makes the Newton system singular.
        ([[1e16, 1.0], [1.0, 1e16]], [1.0, 1.0], [1.01, 0.99]),
        # Review round 2: the seed's sum overflows.
        ([[1e308, 1e308], [1e308, 1e308]], [1.0, 1.0], [1.0, 1.0]),
        # Seventeen splits, checked over the two groups.
        ([[1.0, 1.0]] * 17, [1.0] * 17, [10.0, 7.0]),
        # Review round 3: splits one and two hold 2**-24 less than group one,
        # but their float sum rounds to the group's total exactly.
        (
            [[1_056_964_608.0, 0.0], [16_777_215.0, 0.0], [1.0, 16_777_216.0]],
            [
                1_056_964_608.0 * (1 + 2**-24),
                16_777_215.0 * (1 + 2**-24),
                16_777_217.0 * (1 + 2**-24),
            ],
            [1_073_741_887.0, 16_777_218.0],
        ),
        # Review round 3: dividing 1e-308 by the largest cell underflowed.
        ([[1e-308, 0.0], [0.0, 1e308]], [1.0, 1.0], [1.0, 1.0]),
    ],
)
def test_rake_meets_feasible_ill_conditioned_margins(seed, rows, columns) -> None:
    r, s = _rake_two_margins(seed, rows, columns, label="test")
    _assert_meets_margins(seed, rows, columns, r, s)


def _assert_meets_margins(seed, rows, columns, r, s) -> None:
    for i, total in enumerate(rows):
        fitted = sum(r[i] * s[j] * seed[i][j] for j in range(len(columns)))
        assert fitted == pytest.approx(total, rel=1e-12, abs=1e-12)
    for j, total in enumerate(columns):
        fitted = sum(r[i] * s[j] * seed[i][j] for i in range(len(rows)))
        assert fitted == pytest.approx(total, rel=1e-11, abs=1e-12)


def _positively_scalable(seed, rows, columns) -> bool:
    """An independent oracle: a linear program for the largest t such that
    some matrix with the seed's positive cells, each at least t, meets both
    margins. A positive scaling exists exactly when that t is positive
    (zero-margin rows and columns drop out, as the rake gives them zero)."""

    cells = [
        (i, j)
        for i, total in enumerate(rows)
        for j, column_total in enumerate(columns)
        if total > 0 and column_total > 0 and seed[i][j] > 0
    ]
    live_rows = [i for i, total in enumerate(rows) if total > 0]
    live_columns = [j for j, total in enumerate(columns) if total > 0]
    if not live_rows:
        return True
    if any(not any(c[0] == i for c in cells) for i in live_rows) or any(
        not any(c[1] == j for c in cells) for j in live_columns
    ):
        return False
    n = len(cells)
    equalities = []
    targets = []
    for i in live_rows:
        equalities.append([1.0 if c[0] == i else 0.0 for c in cells] + [0.0])
        targets.append(rows[i])
    for j in live_columns:
        equalities.append([1.0 if c[1] == j else 0.0 for c in cells] + [0.0])
        targets.append(columns[j])
    # t - y_c <= 0 for every cell.
    upper = [[-1.0 if k == c else 0.0 for k in range(n)] + [1.0] for c in range(n)]
    result = linprog(
        c=[0.0] * n + [-1.0],
        A_ub=upper,
        b_ub=[0.0] * n,
        A_eq=equalities,
        b_eq=targets,
        bounds=[(0, None)] * n + [(0, 1)],
        method="highs",
    )
    return bool(result.status == 0 and -result.fun > 1e-7)


@st.composite
def _sparse_margins(draw):
    n_rows = draw(st.integers(1, 4))
    n_columns = draw(st.integers(1, 5))
    seed = [
        [float(draw(st.sampled_from([0, 0, 1, 2, 5, 100]))) for _ in range(n_columns)]
        for _ in range(n_rows)
    ]
    rows = [float(draw(st.integers(0, 12))) for _ in range(n_rows)]
    total = sum(rows)
    columns = [float(draw(st.integers(0, 12))) for _ in range(n_columns - 1)]
    last = total - sum(columns)
    if last < 0:
        columns = [value * total / sum(columns) for value in columns]
        last = 0.0
    return seed, rows, [*columns, last]


@given(_sparse_margins())
@_SETTINGS
def test_rake_succeeds_exactly_when_a_positive_scaling_exists(case) -> None:
    """Against the linear-program oracle: the rake returns a scaling meeting
    both margins whenever one exists, keeping the seed's cross-product
    ratios, and refuses otherwise."""

    seed, rows, columns = case
    feasible = _positively_scalable(seed, rows, columns)
    try:
        r, s = _rake_two_margins(seed, rows, columns, label="test")
    except ValueError as error:
        assert not feasible, error
        assert re.search(
            "meets both margins|has no source EITC|has no EITC in any AGI", str(error)
        )
        return
    assert feasible
    _assert_meets_margins(seed, rows, columns, r, s)
    for (i, k), (j, m) in itertools.product(
        itertools.combinations(range(len(rows)), 2),
        itertools.combinations(range(len(columns)), 2),
    ):
        corners = (seed[i][j], seed[k][m], seed[i][m], seed[k][j])
        if min(corners) > 0 and min(r[i], r[k], s[j], s[m]) > 0:
            fitted = (r[i] * s[j] * seed[i][j] * r[k] * s[m] * seed[k][m]) / (
                r[i] * s[m] * seed[i][m] * r[k] * s[j] * seed[k][j]
            )
            assert fitted == pytest.approx(
                (seed[i][j] * seed[k][m]) / (seed[i][m] * seed[k][j]), rel=1e-9
            )


@given(
    exponents=st.lists(st.floats(-3, 7), min_size=12, max_size=12),
    rows=st.lists(st.integers(1, 10**8), min_size=3, max_size=3),
    shares=st.lists(st.floats(0.001, 1.0), min_size=4, max_size=4),
)
@_SETTINGS
def test_rake_meets_margins_on_positive_seeds_spanning_ten_orders(
    exponents, rows, shares
) -> None:
    seed = [[10.0**e for e in exponents[k * 4 : (k + 1) * 4]] for k in range(3)]
    row_totals = [float(value) for value in rows]
    columns = [sum(row_totals) * share / sum(shares) for share in shares]
    r, s = _rake_two_margins(seed, row_totals, columns, label="test")
    _assert_meets_margins(seed, row_totals, columns, r, s)


def _two_by_two_reference(seed, rows, columns) -> list[list[float]]:
    """The 2 x 2 rake in closed form, by bisection on the cross-product ratio.

    With ``a`` the top-left cell the others are r1 - a, c1 - a and
    c2 - r1 + a. IPF keeps theta = (p v) / (q u), so ``a`` solves
    a (c2 - r1 + a) = theta (r1 - a)(c1 - a), whose left side minus right
    side is negative at the smallest feasible ``a`` and positive at the
    largest.
    """

    (p, q), (u, v) = seed
    r1, _ = rows
    c1, c2 = columns
    theta = (p * v) / (q * u)
    lo, hi = max(0.0, r1 - c2), min(r1, c1)
    for _ in range(400):
        mid = (lo + hi) / 2
        if mid * (c2 - r1 + mid) - theta * (r1 - mid) * (c1 - mid) < 0:
            lo = mid
        else:
            hi = mid
    a = (lo + hi) / 2
    return [[a, r1 - a], [c1 - a, c2 - r1 + a]]


@given(
    seed=st.lists(st.integers(1, 10**6), min_size=4, max_size=4),
    rows=st.tuples(st.integers(1, 10**7), st.integers(1, 10**7)),
    share=st.floats(0.01, 0.99),
)
@_SETTINGS
def test_rake_matches_the_closed_form_two_by_two(seed, rows, share) -> None:
    matrix = [[float(seed[0]), float(seed[1])], [float(seed[2]), float(seed[3])]]
    row_totals = [float(rows[0]), float(rows[1])]
    grand = sum(row_totals)
    column_totals = [grand * share, grand - grand * share]
    r, s = _rake_two_margins(matrix, row_totals, column_totals, label="test")
    expected = _two_by_two_reference(matrix, row_totals, column_totals)
    for i, j in itertools.product(range(2), range(2)):
        assert r[i] * s[j] * matrix[i][j] == pytest.approx(
            expected[i][j], rel=1e-7, abs=1e-6 * grand
        )


@st.composite
def _decompositions(draw, *, uniform_growth: bool = False):
    kind = draw(st.sampled_from(sorted(_KINDS)))
    cuts = tuple(sorted(draw(st.sets(st.sampled_from(_CUTS), min_size=1, max_size=5))))
    # Active filing-season bins coarsen the Table 2.5 bins, and never into a
    # single all-AGI bin (that is the all-returns total, not an AGI bucket).
    active_cuts = tuple(
        sorted(draw(st.sets(st.sampled_from(cuts), min_size=1, max_size=len(cuts))))
    )
    splits = draw(
        st.lists(st.sampled_from(_SPLITS), min_size=1, max_size=4, unique=True)
    )
    source = {
        split: {
            bounds: draw(st.integers(min_value=1, max_value=10**7))
            for bounds in _bins(cuts)
        }
        for split in splits
    }
    growth = draw(st.integers(min_value=50, max_value=200)) / 100
    active = {}
    for lower, upper in _bins(active_cuts):
        group_source = _group_sum(source, lower, upper)
        active[(lower, upper)] = (
            group_source * growth
            if uniform_growth
            else draw(st.integers(min_value=1, max_value=10**8))
        )
    return kind, source, active


def _inside(lower, upper, outer_lower, outer_upper) -> bool:
    lo = -float("inf") if lower is None else lower
    hi = float("inf") if upper is None else upper
    outer_lo = -float("inf") if outer_lower is None else outer_lower
    outer_hi = float("inf") if outer_upper is None else outer_upper
    return outer_lo <= lo and hi <= outer_hi


def _group_sum(cells, lower, upper) -> float:
    return sum(
        value
        for rows in cells.values()
        for bounds, value in rows.items()
        if _inside(*bounds, lower, upper)
    )


@given(_decompositions())
@_SETTINGS
def test_uprated_decomposition_reconciles_its_child_splits_and_agi_groups(
    case,
) -> None:
    kind, source, active = case
    measure, _, _ = _KINDS[kind]
    by_name = _compile(_decomposition_facts(kind=kind, source=source, active=active))
    active_total = sum(active.values())
    source_total = sum(sum(rows.values()) for rows in source.values())
    totals = {split: by_name[_total_name(split, measure)].value for split in source}
    uprated = {
        split: {bounds: by_name[_bin_name(split, measure, bounds)] for bounds in rows}
        for split, rows in source.items()
    }
    values = {
        split: {bounds: spec.value for bounds, spec in rows.items()}
        for split, rows in uprated.items()
    }

    # Conservation, and the total rows keep the all-AGI child split.
    assert sum(totals.values()) == pytest.approx(active_total, rel=1e-9)
    for split, total in totals.items():
        assert total == pytest.approx(
            sum(source[split].values()) * active_total / source_total, rel=1e-9
        )
    # Each split's bins sum to its total row.
    for split, rows in values.items():
        assert sum(rows.values()) == pytest.approx(totals[split], rel=1e-9)
    # Each active group's bins sum to its active EITC.
    for (lower, upper), group_active in active.items():
        assert _group_sum(values, lower, upper) == pytest.approx(group_active, rel=1e-9)
    # One factor per (split, group) cell, which the metadata records.
    for split, rows in uprated.items():
        for lower, upper in active:
            factors = {
                spec.value / source[split][bounds]
                for bounds, spec in rows.items()
                if _inside(*bounds, lower, upper)
            }
            assert max(factors) == pytest.approx(min(factors), rel=1e-12)
        for bounds, spec in rows.items():
            assert float(spec.metadata["uprating_factor"]) == pytest.approx(
                spec.value / source[split][bounds], rel=1e-12
            )
    # The source's cross-product ratios between splits and groups survive.
    groups = list(active)
    for one, two in itertools.combinations(source, 2):
        for g, h in itertools.combinations(groups, 2):

            def ratio(cells, one=one, two=two, g=g, h=h):
                def cell(split, group):
                    return _group_sum({split: cells[split]}, *group)

                return (cell(one, g) * cell(two, h)) / (cell(one, h) * cell(two, g))

            assert ratio(values) == pytest.approx(ratio(source), rel=1e-6)


@given(_decompositions(uniform_growth=True))
@_SETTINGS
def test_uniform_growth_leaves_the_rake_nothing_to_do(case) -> None:
    kind, source, active = case
    measure, _, _ = _KINDS[kind]
    by_name = _compile(_decomposition_facts(kind=kind, source=source, active=active))
    growth = sum(active.values()) / sum(sum(rows.values()) for rows in source.values())

    for split, rows in source.items():
        total = by_name[_total_name(split, measure)].value
        assert total == pytest.approx(sum(rows.values()) * growth, rel=1e-9)
        for bounds, value in rows.items():
            spec = by_name[_bin_name(split, measure, bounds)]
            assert spec.value == pytest.approx(value * growth, rel=1e-9)
            assert float(
                spec.metadata["uprating_child_split_adjustment"]
            ) == pytest.approx(1, rel=1e-9)
