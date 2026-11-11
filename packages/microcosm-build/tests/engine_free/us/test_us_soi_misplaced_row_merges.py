"""SOI row merges and the EITC AGI uprating-group coarsening, on synthetic rows.

``test_us_eitc_table_2_5_irs_reconciliation.py`` checks both on the real
Table 2.5 and filing-season rows. This module checks their guards and states
their invariants as properties:

* a merge replaces its members with one row at their sum, in the first
  member's place, and leaves every other row as it was; a registry it does
  not apply to comes back unchanged;
* coarsening returns groups that tile what the input groups tile, end on
  input edges, conserve each control's value, leave no source row
  straddling a group and no leaf outside a single group, and come back
  unchanged when no row needs a merge; it is idempotent and agrees with an
  independent edge-removal reading of the same rule;
* through the real uprating, for any filing-season AGI groups and any child
  splits whose bins tile AGI, each final group's uprated bins add to its
  controls and the uprated split totals add to all of them.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.fiscal_targets import (
    US_SOI_MISPLACED_ROW_MERGES,
    SoiRowMerge,
    _coarsen_eitc_agi_groups,
    _EitcAgiTotal,
    _merge_misplaced_soi_rows,
    _uprate_cross_period_eitc_decompositions,
)
from microcosm.calibrate import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyDimension,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
)

_RECORD_SET = "irs_soi.ty2023.table_9_9.test_split"
_MEASURE = "eitc_returns"
_AGI = "us:statutes/26/62#adjusted_gross_income"
_INF = float("inf")

_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _label(lower: float, upper: float) -> str:
    return f"${lower:,.0f} under ${upper:,.0f}"


def _row(
    row_id: str,
    lower: float,
    upper: float,
    value: float,
    *,
    record_set: str = _RECORD_SET,
    measure: str = _MEASURE,
    **metadata: str,
) -> TargetSpec:
    name = f"{record_set}.{row_id}.{measure}"
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        period=2024,
        source="test source",
        family="irs_soi",
        metadata={
            "ledger_layout_record_set_id": record_set,
            "source_measure_id": measure,
            "source_period": "2023",
            "agi_lower_bound": f"{lower:.1f}",
            "agi_upper_bound": f"{upper:.1f}",
            f"ledger_filter_{_AGI}_lower_bound": f"{lower:.0f}",
            f"ledger_filter_{_AGI}_upper_bound": f"{upper:.0f}",
            "ledger_filter_income_range": row_id,
            "ledger_layout_groupby_value_id": row_id,
            "ledger_layout_groupby_value_label": _label(lower, upper),
            "ledger_source_record_id": name,
            "ledger_fact_key": f"fact:{row_id}",
            "ledger_aggregate_fact_key": f"fact:{row_id}",
            "ledger_semantic_fact_key": f"semantic:{row_id}",
            "ledger_legacy_fact_key": f"legacy:{row_id}",
            "ledger_dimension_set_key": f"dimensions:{row_id}",
            "ledger_universe_constraint_set_key": f"universe:{row_id}",
            "ledger_fact_label": f"returns ({row_id})",
            "requires_total_eitc_uprating": "true",
            **metadata,
        },
        hierarchy=CalibrationHierarchy(
            provider=HierarchyNode("irs_soi", "IRS Statistics of Income"),
            category=HierarchyCategory("irs_soi.eitc", "EITC", "irs_soi"),
            geography=HierarchyGeography("0100000US", "United States", "country"),
            dimensions=(
                HierarchyDimension(
                    _AGI, "Adjusted gross income", row_id, _label(lower, upper)
                ),
                HierarchyDimension(
                    "income_range", "Income range", row_id, _label(lower, upper)
                ),
            ),
            target=HierarchyNode(name, "EITC"),
        ),
    )


def _merge(
    row_ids: tuple[str, ...] = ("a", "b"),
    published: tuple[float, ...] = (10.0, 0.0),
) -> SoiRowMerge:
    return SoiRowMerge(
        record_set_id=_RECORD_SET,
        measure_id=_MEASURE,
        source_row_ids=row_ids,
        published_values=published,
        merged_row_id="".join(row_ids),
        merged_row_label="merged label",
        reason="test",
    )


def _other() -> TargetSpec:
    return _row("other", 0, 100, 7.0, record_set="irs_soi.ty2023.other")


def _registry(*specs: TargetSpec) -> TargetRegistry:
    return TargetRegistry(specs, country="us")


def test_merge_replaces_members_with_one_summed_row_in_the_first_member_s_place():
    registry = _registry(
        _other(), _row("a", 100, 200, 10.0), _other_row("x"), _row("b", 200, 300, 0.0)
    )
    merged = _merge_misplaced_soi_rows(registry, (_merge(),))
    names = [spec.name for spec in merged.specs]
    assert names == [
        "irs_soi.ty2023.other.other.eitc_returns",
        f"{_RECORD_SET}.ab.{_MEASURE}",
        "irs_soi.ty2023.other.x.eitc_returns",
    ]
    assert merged.country == "us"
    spec = merged.specs[1]
    assert spec.value == 10.0
    assert spec.measure == spec.name
    metadata = spec.metadata
    assert (metadata["agi_lower_bound"], metadata["agi_upper_bound"]) == (
        "100.0",
        "300.0",
    )
    assert (
        metadata[f"ledger_filter_{_AGI}_lower_bound"],
        metadata[f"ledger_filter_{_AGI}_upper_bound"],
    ) == ("100", "300")
    assert metadata["ledger_filter_income_range"] == "ab"
    assert metadata["ledger_layout_groupby_value_id"] == "ab"
    assert metadata["ledger_layout_groupby_value_label"] == "merged label"
    assert metadata["derived_operation"] == "sum_of_misplaced_soi_rows"
    assert metadata["derived_source_record_ids"] == (
        f"{_RECORD_SET}.a.{_MEASURE},{_RECORD_SET}.b.{_MEASURE}"
    )
    assert not [key for key in metadata if key.endswith("_key")]
    assert "ledger_source_record_id" not in metadata
    assert "ledger_fact_label" not in metadata
    assert metadata["requires_total_eitc_uprating"] == "true"
    assert [
        (dimension.id, dimension.value_id, dimension.value_label)
        for dimension in spec.hierarchy.dimensions
    ] == [(_AGI, "ab", "merged label"), ("income_range", "ab", "merged label")]
    assert spec.hierarchy.target == HierarchyNode(spec.name, "EITC")


def _other_row(row_id: str) -> TargetSpec:
    return _row(row_id, 0, 100, 3.0, record_set="irs_soi.ty2023.other")


def test_merge_leaves_a_registry_without_its_members_unchanged():
    registry = _registry(_other(), _other_row("x"))
    assert _merge_misplaced_soi_rows(registry, (_merge(),)) is registry


def test_merge_leaves_a_registry_whose_members_hold_other_values_unchanged():
    """One member, not at its published value: a different feed (here, a
    synthetic test registry), not the reviewed table."""

    registry = _registry(_row("a", 100, 200, 5.0))
    assert _merge_misplaced_soi_rows(registry, (_merge(),)) is registry


def test_merge_refuses_a_feed_with_one_member_at_its_published_value():
    registry = _registry(_row("a", 100, 200, 10.0))
    with pytest.raises(ValueError, match="carries 1 of its 2 rows"):
        _merge_misplaced_soi_rows(registry, (_merge(),))


@pytest.mark.parametrize("values", [(11.0, 0.0), (10.0, 1.0), (0.0, 10.0)])
def test_merge_refuses_members_whose_values_moved(values):
    registry = _registry(_row("a", 100, 200, values[0]), _row("b", 200, 300, values[1]))
    with pytest.raises(ValueError, match="differ from the reviewed"):
        _merge_misplaced_soi_rows(registry, (_merge(),))


def test_merge_refuses_a_member_named_twice():
    """The registry refuses a repeated (name, period); a member at two periods
    is still two targets, and the merge cannot tell which one to join."""

    registry = _registry(
        _row("a", 100, 200, 10.0),
        replace(_row("a", 100, 200, 10.0), period=2025),
        _row("b", 200, 300, 0.0),
    )
    with pytest.raises(ValueError, match="a member must be one target"):
        _merge_misplaced_soi_rows(registry, (_merge(),))


def test_merge_refuses_members_that_do_not_tile():
    registry = _registry(_row("a", 100, 200, 10.0), _row("b", 250, 300, 0.0))
    with pytest.raises(ValueError, match="do not share an AGI bound"):
        _merge_misplaced_soi_rows(registry, (_merge(),))


def test_merge_refuses_members_that_disagree_on_other_metadata():
    registry = _registry(
        _row("a", 100, 200, 10.0, filing_status="All"),
        _row("b", 200, 300, 0.0, filing_status="Single"),
    )
    with pytest.raises(ValueError, match="'filing_status'"):
        _merge_misplaced_soi_rows(registry, (_merge(),))


def test_merge_refuses_members_with_a_standard_error():
    registry = _registry(
        _row("a", 100, 200, 10.0), replace(_row("b", 200, 300, 0.0), se=1.0)
    )
    with pytest.raises(ValueError, match="standard error"):
        _merge_misplaced_soi_rows(registry, (_merge(),))


def test_merge_refuses_a_member_from_another_measure_s_metadata():
    registry = _registry(
        _row("a", 100, 200, 10.0),
        _row("b", 200, 300, 0.0, source_measure_id="eitc_total"),
    )
    with pytest.raises(ValueError, match="does not measure"):
        _merge_misplaced_soi_rows(registry, (_merge(),))


def test_merge_refuses_a_registry_that_already_holds_the_merged_name():
    registry = _registry(
        _row("a", 100, 200, 10.0), _row("b", 200, 300, 0.0), _row("ab", 100, 300, 1.0)
    )
    with pytest.raises(ValueError, match="already holds"):
        _merge_misplaced_soi_rows(registry, (_merge(),))


def test_declared_merges_are_well_formed():
    """Each reviewed entry joins two or more Table 2.5 rows under a new row
    id, one published value per row, and no two entries share a row."""

    seen: set[str] = set()
    for merge in US_SOI_MISPLACED_ROW_MERGES:
        assert ".table_2_5." in merge.record_set_id
        assert len(merge.source_row_ids) >= 2
        assert len(merge.published_values) == len(merge.source_row_ids)
        assert len(set(merge.source_row_ids)) == len(merge.source_row_ids)
        assert merge.merged_row_id not in merge.source_row_ids
        assert merge.reason
        assert not seen & set(merge.member_names)
        seen.update(merge.member_names)


@st.composite
def _merge_cases(draw):
    """A merge over 2 to 4 tiling rows at their published values, among
    unrelated rows."""

    size = draw(st.integers(min_value=2, max_value=4))
    edges = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=10**6),
                min_size=size + 1,
                max_size=size + 1,
                unique=True,
            )
        )
    )
    published = tuple(
        float(draw(st.integers(min_value=0, max_value=10**9))) for _ in range(size)
    )
    row_ids = tuple(f"r{index}" for index in range(size))
    members = [
        _row(row_id, edges[index], edges[index + 1], published[index])
        for index, row_id in enumerate(row_ids)
    ]
    others = [
        _other_row(f"o{index}")
        for index in range(draw(st.integers(min_value=0, max_value=3)))
    ]
    specs = [*others, *members]
    order = draw(st.permutations(range(len(specs))))
    return _merge(row_ids, published), [specs[index] for index in order]


@given(_merge_cases())
@_SETTINGS
def test_merge_conserves_value_and_touches_only_its_members(case):
    merge, specs = case
    registry = _registry(*specs)
    merged = _merge_misplaced_soi_rows(registry, (merge,))

    member_names = set(merge.member_names)
    assert len(merged.specs) == len(specs) - len(member_names) + 1
    assert sum(spec.value for spec in merged.specs) == sum(spec.value for spec in specs)
    (row,) = [spec for spec in merged.specs if spec.name == merge.merged_name]
    assert row.value == sum(merge.published_values)
    assert [spec for spec in merged.specs if spec.name != merge.merged_name] == [
        spec for spec in specs if spec.name not in member_names
    ]
    first_member = min(
        index for index, spec in enumerate(specs) if spec.name in member_names
    )
    assert merged.specs[first_member] is row
    # Idempotent: the members are gone, so a second pass changes nothing.
    assert _merge_misplaced_soi_rows(merged, (merge,)) is merged


# --- Coarsening -------------------------------------------------------------


def _group(lower: float, upper: float, value: float, index: int = 0) -> _EitcAgiTotal:
    return _EitcAgiTotal(
        value=value,
        source_periods=("2024",),
        source_record_ids=(f"control.{index}",),
        period_key=(1, 2024, "2024"),
        lower=lower,
        upper=upper,
    )


def _bounds(groups) -> list[tuple[float, float]]:
    return [(group.lower, group.upper) for group in groups]


def test_coarsening_joins_the_groups_a_merged_row_straddles():
    """The Table 2.5 case: a $25,000 to $35,000 row across the filing
    season's $25,000 to $30,000 and $30,000 to $40,000 bins."""

    groups = [
        _group(-_INF, 25_000, 1.0, 0),
        _group(25_000, 30_000, 2.0, 1),
        _group(30_000, 40_000, 3.0, 2),
        _group(40_000, _INF, 4.0, 3),
    ]
    rows = [(20_000, 25_000), (25_000, 35_000), (35_000, 40_000), (-_INF, _INF)]
    coarsened = _coarsen_eitc_agi_groups(groups, rows)
    assert _bounds(coarsened) == [
        (-_INF, 25_000),
        (25_000, 40_000),
        (40_000, _INF),
    ]
    assert coarsened[1].value == 5.0
    assert coarsened[1].source_record_ids == ("control.1", "control.2")
    assert coarsened[0] is groups[0]
    assert coarsened[2] is groups[3]


def test_coarsening_leaves_a_row_past_the_groups_range_alone():
    """No merge of groups covering $0 to $20 holds a row that runs to $30, so
    such a row keeps the all-returns fallback, straddling or leaf alike
    (test_cross_period_soi_eitc_uprating_requires_complete_agi_interval)."""

    groups = [_group(0, 10, 1.0), _group(10, 20, 1.0, 1)]
    assert _coarsen_eitc_agi_groups(groups, [(5, 30)]) == tuple(groups)
    assert _coarsen_eitc_agi_groups(groups, [], leaves=[(5, 30)]) == tuple(groups)
    assert _coarsen_eitc_agi_groups(groups, [], leaves=[(0, 30)]) == tuple(groups)


def test_coarsening_refuses_groups_with_a_gap():
    groups = [_group(0, 10, 1.0), _group(12, 20, 1.0, 1)]
    with pytest.raises(ValueError, match="do not tile one interval holding it"):
        _coarsen_eitc_agi_groups(groups, [(5, 15)])
    with pytest.raises(ValueError, match="do not tile one interval holding it"):
        _coarsen_eitc_agi_groups(groups, [], leaves=[(5, 11)])


def test_coarsening_refuses_to_duplicate_a_nested_total():
    groups = [_group(0, 10, 1.0), _group(10, 20, 1.0, 1), _group(0, 20, 2.5, 2)]
    with pytest.raises(ValueError, match="nested AGI totals"):
        _coarsen_eitc_agi_groups(groups, [(5, 15)])


def test_coarsening_gives_a_leaf_that_holds_groups_one_group():
    """A leaf (a row with no row of its own split inside it) that holds two
    groups exactly sits in neither, so each group's source total would miss
    it: the merged $25,000 to $35,000 row against a filing season with a
    $35,000 edge, and a coarse $25,000 to $40,000 bin against the $30,000
    edge. A row with its split's rows inside it may hold several groups."""

    groups = [
        _group(-_INF, 25_000, 1.0, 0),
        _group(25_000, 30_000, 2.0, 1),
        _group(30_000, 35_000, 3.0, 2),
        _group(35_000, 40_000, 4.0, 3),
        _group(40_000, _INF, 5.0, 4),
    ]
    assert _coarsen_eitc_agi_groups(groups, [(25_000, 35_000)]) == tuple(groups)
    coarsened = _coarsen_eitc_agi_groups(
        groups, [(-_INF, _INF)], leaves=[(20_000, 25_000), (25_000, 35_000)]
    )
    assert _bounds(coarsened) == [
        (-_INF, 25_000),
        (25_000, 35_000),
        (35_000, 40_000),
        (40_000, _INF),
    ]
    assert coarsened[1].value == 5.0

    season = [
        _group(-_INF, 25_000, 1.0, 0),
        _group(25_000, 30_000, 2.0, 1),
        _group(30_000, 40_000, 3.0, 2),
        _group(40_000, _INF, 4.0, 3),
    ]
    coarse = _coarsen_eitc_agi_groups(season, [], leaves=[(25_000, 40_000)])
    assert _bounds(coarse) == [(-_INF, 25_000), (25_000, 40_000), (40_000, _INF)]
    assert coarse[1].value == 5.0


_POINTS = st.integers(min_value=0, max_value=40)


@st.composite
def _coarsening_cases(draw):
    """Groups tiling the whole AGI line, with a value each, and source rows
    whose bounds may fall anywhere, the infinite ends included; some of the
    rows are leaves."""

    inner = sorted(draw(st.sets(_POINTS, min_size=0, max_size=8)))
    edges = [-_INF, *map(float, inner), _INF]
    groups = [
        _group(lower, upper, float(draw(st.integers(0, 10**6))), index)
        for index, (lower, upper) in enumerate(zip(edges, edges[1:], strict=False))
    ]
    ends = st.one_of(st.just(-_INF), st.just(_INF), _POINTS.map(float))
    rows = []
    for _ in range(draw(st.integers(min_value=0, max_value=8))):
        lower, upper = sorted(draw(st.lists(ends, min_size=2, max_size=2, unique=True)))
        rows.append((lower, upper))
    leaves = [row for row in rows if draw(st.booleans())]
    return groups, rows, leaves


def _straddles(group: _EitcAgiTotal, row: tuple[float, float]) -> bool:
    lower, upper = row
    overlap = group.lower < upper and lower < group.upper
    group_holds_row = group.lower <= lower and upper <= group.upper
    row_holds_group = lower <= group.lower and group.upper <= upper
    return overlap and not group_holds_row and not row_holds_group


def _held(groups, row: tuple[float, float]) -> bool:
    return any(group.lower <= row[0] and row[1] <= group.upper for group in groups)


def _edge_removal(groups, rows, leaves=()) -> list[tuple[float, float]]:
    """The same rule read as edges, over groups that tile the line: a leaf
    removes every edge inside it, and any other row that crosses an edge
    without starting and ending on edges does too, until no row does."""

    leaves = set(leaves)
    edges = sorted({bound for group in groups for bound in (group.lower, group.upper)})
    changed = True
    while changed:
        changed = False
        for lower, upper in [*rows, *leaves]:
            inside = [edge for edge in edges if lower < edge < upper]
            if inside and (
                (lower, upper) in leaves or not (lower in edges and upper in edges)
            ):
                edges = [edge for edge in edges if edge not in inside]
                changed = True
    return list(zip(edges, edges[1:], strict=False))


@given(_coarsening_cases())
@_SETTINGS
def test_coarsening_invariants(case):
    groups, rows, leaves = case
    coarsened = _coarsen_eitc_agi_groups(groups, rows, leaves=leaves)

    # Tiles the line, on input edges only.
    bounds = _bounds(coarsened)
    assert bounds[0][0] == -_INF and bounds[-1][1] == _INF
    assert all(
        left[1] == right[0] for left, right in zip(bounds, bounds[1:], strict=False)
    )
    input_edges = {bound for group in groups for bound in (group.lower, group.upper)}
    assert {bound for pair in bounds for bound in pair} <= input_edges
    # Each coarsened control is the sum of the input controls inside it.
    for total in coarsened:
        inside = [
            group
            for group in groups
            if total.lower <= group.lower and group.upper <= total.upper
        ]
        assert total.value == sum(group.value for group in inside)
    assert sum(total.value for total in coarsened) == sum(
        group.value for group in groups
    )
    # No row straddles a group, and one group holds each leaf.
    assert not [
        (row, total.lower, total.upper)
        for row in [*rows, *leaves]
        for total in coarsened
        if _straddles(total, row)
    ]
    assert all(_held(coarsened, leaf) for leaf in leaves)
    # Agrees with the edge reading and is idempotent.
    assert bounds == _edge_removal(groups, rows, leaves)
    assert _coarsen_eitc_agi_groups(coarsened, rows, leaves=leaves) == coarsened
    # Unchanged when no row needs a merge.
    if not any(
        _straddles(group, row) for row in [*rows, *leaves] for group in groups
    ) and all(_held(groups, leaf) for leaf in leaves):
        assert list(coarsened) == groups


_SEASON = "irs_soi.ty2024.filing_season_week47.eitc_by_agi"
_SPLIT_SET = "irs_soi.ty2023.table_2_5.eitc_by_agi_children.split_{index}"


def _bound_text(bound: float) -> str:
    return "-inf" if bound == -_INF else "inf" if bound == _INF else f"{bound:.0f}"


def _season_total(index: int, lower: float, upper: float, value: float) -> TargetSpec:
    name = f"{_SEASON}.group_{index}.total_earned_income_credit_returns"
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        period=2024,
        source="test",
        family="irs_soi",
        metadata={
            "source_measure_id": "total_earned_income_credit_returns",
            "filing_status": "All",
            "source_period": "2024",
            "agi_lower_bound": _bound_text(lower),
            "agi_upper_bound": _bound_text(upper),
            "ledger_source_record_id": name,
        },
    )


def _split_row(
    split: int, row_id: str, lower: float, upper: float, value: float
) -> TargetSpec:
    record_set = _SPLIT_SET.format(index=split)
    name = f"{record_set}.{row_id}.eitc_returns"
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        period=2024,
        source="test",
        family="irs_soi",
        metadata={
            "source_measure_id": "eitc_returns",
            "ledger_layout_record_set_id": record_set,
            "filing_status": "All",
            "source_period": "2023",
            "agi_lower_bound": _bound_text(lower),
            "agi_upper_bound": _bound_text(upper),
            "requires_total_eitc_uprating": "true",
        },
    )


@st.composite
def _uprating_cases(draw):
    """Filing-season groups tiling AGI, each with a control, and one to four
    child splits whose bins tile AGI at their own edges, each with an
    all-AGI total row and, sometimes, a subtotal over a run of its bins."""

    cuts = st.sets(st.integers(min_value=1, max_value=39), min_size=1, max_size=8)
    season_edges = [-_INF, *(1_000.0 * edge for edge in sorted(draw(cuts))), _INF]
    season = [
        (lower, upper, float(draw(st.integers(1, 10**6))))
        for lower, upper in zip(season_edges, season_edges[1:], strict=False)
    ]
    splits = []
    for _ in range(draw(st.integers(min_value=1, max_value=4))):
        edges = [-_INF, *(1_000.0 * edge for edge in sorted(draw(cuts))), _INF]
        bins = [
            (lower, upper, float(draw(st.integers(1, 10**6))))
            for lower, upper in zip(edges, edges[1:], strict=False)
        ]
        subtotal = None
        if len(bins) >= 3 and draw(st.booleans()):
            start = draw(st.integers(0, len(bins) - 2))
            stop = draw(st.integers(start + 2, len(bins)))
            if (start, stop) != (0, len(bins)):
                subtotal = (start, stop)
        splits.append((bins, subtotal))
    return season, splits


@given(_uprating_cases())
@_SETTINGS
def test_uprating_conserves_each_group_s_controls_for_any_split_tiling(case):
    """Invariant: after coarsening, each final uprating group's uprated bins,
    over all splits, add to the controls inside it, and the uprated split
    totals add to every control. Each bin takes one group's factor, and that
    group holds it. Reviewers found that a bin holding two filing-season
    groups exactly broke the first property before the leaf rule."""

    season, splits = case
    specs = [
        _season_total(index, lower, upper, value)
        for index, (lower, upper, value) in enumerate(season)
    ]
    bin_names = set()
    total_names = set()
    for split, (bins, subtotal) in enumerate(splits):
        for index, (lower, upper, value) in enumerate(bins):
            spec = _split_row(split, f"bin_{index}", lower, upper, value)
            specs.append(spec)
            bin_names.add(spec.name)
        if subtotal is not None:
            start, stop = subtotal
            specs.append(
                _split_row(
                    split,
                    "subtotal",
                    bins[start][0],
                    bins[stop - 1][1],
                    sum(value for _, _, value in bins[start:stop]),
                )
            )
        total = _split_row(
            split, "total", -_INF, _INF, sum(value for _, _, value in bins)
        )
        specs.append(total)
        total_names.add(total.name)
    uprated = _uprate_cross_period_eitc_decompositions(
        TargetRegistry(specs, country="us")
    )
    rows = [spec for spec in uprated.specs if ".table_2_5." in spec.name]
    assert len(rows) == len(specs) - len(season)
    by_group: dict[tuple[float, float], float] = {}
    for spec in rows:
        group = (
            float(spec.metadata["uprating_agi_lower_bound"]),
            float(spec.metadata["uprating_agi_upper_bound"]),
        )
        lower = float(spec.metadata["agi_lower_bound"])
        upper = float(spec.metadata["agi_upper_bound"])
        assert group[0] <= lower and upper <= group[1], spec.name
        if spec.name in bin_names:
            by_group[group] = by_group.get(group, 0.0) + spec.value
    for (lower, upper), value in by_group.items():
        controls = sum(
            control
            for control_lower, control_upper, control in season
            if lower <= control_lower and control_upper <= upper
        )
        assert math.isclose(value, controls, rel_tol=1e-9), (lower, upper)
    assert math.isclose(
        sum(by_group.values()), sum(value for _, _, value in season), rel_tol=1e-9
    )
    assert math.isclose(
        sum(spec.value for spec in rows if spec.name in total_names),
        sum(value for _, _, value in season),
        rel_tol=1e-9,
    )
