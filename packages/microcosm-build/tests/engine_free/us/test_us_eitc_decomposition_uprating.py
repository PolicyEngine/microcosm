"""Cross-period uprating of the SOI Table 2.5 EITC child decomposition.

Each child split's record set carries its AGI bins and an all-AGI total row
that sums them. The uprating scales a stale row by (active EITC in its AGI
group) / (source-period EITC in that group, summed over child splits). That
source sum must count each split once: a split's total row and its bins
describe the same returns.

Invariants exercised by the property test (decompositions with bins that tile
AGI and total rows equal to their bins, active filing-season AGI bins that
coarsen the Table 2.5 bins):

* conservation: the uprated total rows sum to the active EITC, and so do the
  uprated bins;
* the all-AGI group keeps the source child split: every total row takes the
  same factor, active EITC / source total rows;
* each AGI group keeps its source child split: every bin takes active / source
  EITC of its group;
* when every group grows by the same factor, each total row equals the sum of
  its uprated bins. Otherwise the two can differ, by design: a total row keeps
  the all-AGI child split while its bins keep each group's.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
) -> list[dict[str, object]]:
    """TY2023 Table 2.5 rows (bins plus total rows) and TY2024 active bins."""

    measure, active_measure, _ = _KINDS[kind]
    facts: list[dict[str, object]] = [
        *packaged_reference_facts(),
        _soi_eitc_filing_season_total_fact(
            measure_id=active_measure, value=sum(active.values())
        ),
        *_soi_eitc_child_total_facts(
            2023,
            measure_id=measure,
            values={split: sum(rows.values()) for split, rows in source.items()},
        ),
    ]
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
            income_range = _bin_id(lower, upper)
            facts.append(
                _soi_eitc_child_fact(
                    2023,
                    source_record_id=(
                        "irs_soi.ty2023.table_2_5.eitc_by_agi_children."
                        f"{split}.{income_range}.{measure}"
                    ),
                    measure_id=measure,
                    value=value,
                    income_range=income_range,
                    child_group=split,
                    lower=lower,
                    upper=upper,
                )
            )
    return facts


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


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_all_agi_total_row_counts_each_child_split_once(kind: str) -> None:
    """Both splits' bins (100 + 300, 100 + 200) and total rows (400, 300) sit
    inside the all-AGI group. The source EITC there is 700, not 1,400, so the
    1,400 active EITC doubles each total row instead of leaving it at its
    TY2023 value."""

    measure, _, index = _KINDS[kind]
    low, high = (None, 25_000.0), (25_000.0, None)
    by_name = _compile(
        _decomposition_facts(
            kind=kind,
            source={
                "one_qualifying_child": {low: 100, high: 300},
                "two_qualifying_children": {low: 100, high: 200},
            },
            active={low: 500, high: 900},
        )
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
    # The bins keep their own groups' factors, 500 / 200 and 900 / 500.
    assert by_name[_bin_name("one_qualifying_child", measure, low)].value == 250
    assert by_name[_bin_name("one_qualifying_child", measure, high)].value == 540
    assert by_name[_bin_name("two_qualifying_children", measure, low)].value == 250
    assert by_name[_bin_name("two_qualifying_children", measure, high)].value == 360
    # Both levels conserve the active EITC; per split they differ by design.
    assert one.value + two.value == 1_400


def test_overlapping_decomposition_rows_are_refused() -> None:
    """A split whose AGI rows cross can't be summed without counting some
    returns twice, so the compile refuses it rather than guess."""

    measure, _, _ = _KINDS["amount"]
    low, high = (None, 25_000.0), (25_000.0, None)
    facts = _decomposition_facts(
        kind="amount",
        source={"one_qualifying_child": {low: 100, high: 300}},
        active={low: 500, high: 900},
    )
    for (lower, upper), income_range in (
        ((None, 30_000.0), "crossing_low"),
        ((20_000.0, None), "crossing_high"),
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
        group_source = sum(
            value
            for rows in source.values()
            for (row_lower, row_upper), value in rows.items()
            if _inside(row_lower, row_upper, lower, upper)
        )
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


@given(_decompositions())
@_SETTINGS
def test_uprated_decomposition_conserves_active_eitc_and_child_splits(
    case,
) -> None:
    kind, source, active = case
    measure, _, _ = _KINDS[kind]
    by_name = _compile(_decomposition_facts(kind=kind, source=source, active=active))
    active_total = sum(active.values())
    source_total = sum(sum(rows.values()) for rows in source.values())

    totals = {split: by_name[_total_name(split, measure)] for split in source}
    assert sum(spec.value for spec in totals.values()) == pytest.approx(
        active_total, rel=1e-9
    )
    for split, spec in totals.items():
        assert spec.value == pytest.approx(
            sum(source[split].values()) * active_total / source_total, rel=1e-9
        )

    bins_sum = 0.0
    for (lower, upper), group_active in active.items():
        group_source = sum(
            value
            for rows in source.values()
            for (row_lower, row_upper), value in rows.items()
            if _inside(row_lower, row_upper, lower, upper)
        )
        for split, rows in source.items():
            for bounds, value in rows.items():
                if not _inside(*bounds, lower, upper):
                    continue
                spec = by_name[_bin_name(split, measure, bounds)]
                assert spec.value == pytest.approx(
                    value * group_active / group_source, rel=1e-9
                )
                bins_sum += spec.value
    assert bins_sum == pytest.approx(active_total, rel=1e-9)


@given(_decompositions(uniform_growth=True))
@_SETTINGS
def test_uniform_growth_keeps_total_rows_equal_to_their_bins(case) -> None:
    kind, source, active = case
    measure, _, _ = _KINDS[kind]
    by_name = _compile(_decomposition_facts(kind=kind, source=source, active=active))

    for split, rows in source.items():
        total = by_name[_total_name(split, measure)].value
        bins = sum(by_name[_bin_name(split, measure, bounds)].value for bounds in rows)
        assert total == pytest.approx(bins, rel=1e-9)
