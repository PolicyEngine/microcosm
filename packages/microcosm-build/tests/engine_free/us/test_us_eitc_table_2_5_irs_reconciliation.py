"""SOI Table 2.5 EITC child splits on the real TY2023 rows, in CI.

``tests/fixtures/us_soi_eitc_table_2_5_feed_rows.jsonl`` holds, verbatim and in
source-record-id order, the 274 rows of the pinned Chronicle feed
(``consumer_facts_us_41b2fb2.jsonl``, sha256 1d45a89c...) that the EITC
decomposition uprating reads:

* IRS SOI Publication 1304 Table 2.5, tax year 2023 (``23in25ic.xls``, sha256
  c6a58f63...): the all-returns total row, and for each of the four
  qualifying-child splits its 28 AGI bins and its total row, as returns and
  amount;
* IRS SOI Filing Season Statistics Table 1, 2025 week 47, tax year 2024
  (``25inweek47.xlsx``, sha256 5b8f0573...): EITC returns and amount in 16 AGI
  bins and in total.

TY2023's three-or-more-children column prints its $30,000 to $35,000 cell
(0 returns, $0) in the $25,000 to $30,000 row, so the compile binds the two
rows as one $25,000 to $35,000 row (``US_SOI_MISPLACED_ROW_MERGES``), and the
filing season's $25,000 to $30,000 and $30,000 to $40,000 bins as one
uprating group. The tests below read the source rows as the compile binds
them.

The pinned-feed tests in ``test_us_fiscal_targets.py`` skip in CI, which has no
feed. These run everywhere, so the halving that microcosm#1227 fixed cannot
come back unnoticed: the compile summed each split's AGI bins and its total row
into the all-AGI source total, uprated the four total rows by 0.4877 (returns)
and 0.5209 (amount), and shipped them at 11.92 million returns and $34.52
billion against the TY2024 filing season's 23.84 million and $69.04 billion.

IRS-published EITC (EITC statistics pages, read 2026-10-14):

* TY2023, returns IRS processed by December 2024: "more than 23 million"
  workers and families received "about $64 billion"; 27% had no qualifying
  child, 36% one, 24% two and 13% three or more.
* TY2024, returns IRS processed by December 2025: about 24 million and $69.6
  billion.

IRS labels those "returns processed in" the filing year, "as of December".
Publication 1304 Table 2.5 gives SOI's sample estimate for "Tax Year 2023
(Filing Year 2024)"; its footnote 1 defines EIC earned income "for prior-year
returns", which implies that the table includes them. Its TY2023 EITC
(24,439,936 returns, $66.27 billion) exceeds the TY2023 December figures by
6.3% and 3.5%; the two products' bases differ, and this module does not
attribute the gap. The reconciliation allows 10%: enough for that observed gap,
and far too little for a halving. The tests compare child shares within 1.5
percentage points because IRS rounds them to whole percents.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import replace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from test_support.microcosm_build.us_fiscal_targets import (
    _load_repo_tool,
    compile_us_fiscal_target_registry,
    packaged_reference_facts,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_FIXTURE = _TEST_PATHS.tests / "fixtures" / "us_soi_eitc_table_2_5_feed_rows.jsonl"

_SPLITS = (
    "no_qualifying_children",
    "one_qualifying_child",
    "two_qualifying_children",
    "three_or_more_qualifying_children",
)
_TABLE_2_5 = "irs_soi.ty2023.table_2_5."
_CHILD_RECORD_SET = f"{_TABLE_2_5}eitc_by_agi_children."
_ALL_RETURNS_RECORD_SET = f"{_TABLE_2_5}eitc_all_returns"
_FILING_SEASON = "irs_soi.ty2024.filing_season_week47."
_FILING_SEASON_BINS = f"{_FILING_SEASON}eitc_by_agi."
_FILING_SEASON_TOTAL = f"{_FILING_SEASON}eitc_all_returns."
# kind -> (Table 2.5 child-split measure, all-returns and filing-season measure)
_KINDS = {
    "returns": ("eitc_returns", "total_earned_income_credit_returns"),
    "amount": ("eitc_total", "total_earned_income_credit_amount"),
}

# IRS-published EITC; see the module docstring.
_IRS_PUBLISHED = {
    2023: {"returns": 23_000_000, "amount": 64_000_000_000},
    2024: {"returns": 24_000_000, "amount": 69_600_000_000},
}
_IRS_TY2023_CHILD_SHARES = {
    "no_qualifying_children": 0.27,
    "one_qualifying_child": 0.36,
    "two_qualifying_children": 0.24,
    "three_or_more_qualifying_children": 0.13,
}
# split -> (member AGI row ids, merged row id) for the rows the compile binds
# as one; see the module docstring.
_MERGED_ROWS = {
    "three_or_more_qualifying_children": (("25k_to_30k", "30k_to_35k"), "25k_to_35k"),
}
_PUBLISHED_TOLERANCE = 0.10
_SHARE_TOLERANCE = 0.015
# SOI rounds each cell ("Detail may not add to totals because of rounding"):
# counts to the return, amounts to $1,000. In the TY2023 workbook a split's
# bins and its total row differ by at most 3 returns and $1,000.
_ROUNDING = {"returns": 5, "amount": 5_000}

_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@functools.cache
def _feed_rows() -> tuple[dict[str, object], ...]:
    return tuple(load_ledger_consumer_artifact(_FIXTURE).facts)


def _record_id(fact: dict[str, object]) -> str:
    return fact["lineage"]["source_record_id"]


def _agi_bounds(fact: dict[str, object]) -> tuple[float, float]:
    lower, upper = -float("inf"), float("inf")
    for constraint in fact.get("universe_constraints", {}).get("constraints", ()):
        if not constraint["variable"].endswith("adjusted_gross_income"):
            continue
        if constraint["operator"] == ">=":
            lower = float(constraint["value"])
        elif constraint["operator"] == "<":
            upper = float(constraint["value"])
        else:  # pragma: no cover - the fixture uses only >= and <
            raise AssertionError(f"unexpected AGI operator in {_record_id(fact)}")
    return lower, upper


def _inside(inner: tuple[float, float], outer: tuple[float, float]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _split_rows(kind: str) -> dict[str, dict[str, dict[str, object]]]:
    """split -> AGI row id (28 bins and ``total``) -> fact."""

    measure = _KINDS[kind][0]
    rows: dict[str, dict[str, dict[str, object]]] = {}
    for fact in _feed_rows():
        layout = fact["layout"]
        record_set = layout["record_set_id"]
        if record_set.startswith(_CHILD_RECORD_SET) and layout["measure_id"] == measure:
            split = record_set.removeprefix(_CHILD_RECORD_SET)
            rows.setdefault(split, {})[layout["groupby_value_id"]] = fact
    return rows


def _filing_season_bins(kind: str) -> dict[str, dict[str, object]]:
    measure = _KINDS[kind][1]
    return {
        fact["layout"]["groupby_value_id"]: fact
        for fact in _feed_rows()
        if _record_id(fact).startswith(_FILING_SEASON_BINS)
        and fact["layout"]["measure_id"] == measure
    }


def _single(prefix: str, measure: str) -> dict[str, object]:
    (fact,) = (
        fact
        for fact in _feed_rows()
        if _record_id(fact).startswith(prefix)
        and fact["layout"]["measure_id"] == measure
    )
    return fact


def _compile(facts) -> dict[str, object]:
    registry = compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *facts],
        target_period=2024,
        allow_unaged_dollar_targets=True,
    )
    return {spec.name: spec for spec in registry.specs}


@functools.cache
def _compiled_real() -> dict[str, object]:
    return _compile(_feed_rows())


def _bound_rows(kind: str) -> dict[str, dict[str, tuple[tuple[float, float], float]]]:
    """split -> AGI row id -> (AGI bounds, source value), as the compile binds
    the rows: each split's 28 bins and total row, with a merged row in place of
    its members."""

    bound: dict[str, dict[str, tuple[tuple[float, float], float]]] = {}
    for split, rows in _split_rows(kind).items():
        split_rows = {
            row_id: (_agi_bounds(fact), fact["value"]) for row_id, fact in rows.items()
        }
        if split in _MERGED_ROWS:
            members, merged = _MERGED_ROWS[split]
            parts = [split_rows.pop(row_id) for row_id in members]
            split_rows[merged] = (
                (parts[0][0][0], parts[-1][0][1]),
                sum(value for _, value in parts),
            )
        bound[split] = split_rows
    return bound


def _compiled_split_rows(by_name, kind: str) -> dict[str, dict[str, object]]:
    """split -> AGI row id -> compiled spec, for every bound Table 2.5 row."""

    measure = _KINDS[kind][0]
    return {
        split: {
            row_id: by_name[f"{_CHILD_RECORD_SET}{split}.{row_id}.{measure}"]
            for row_id in rows
        }
        for split, rows in _bound_rows(kind).items()
    }


def _uprating_groups(
    kind: str,
) -> dict[tuple[float, float], list[tuple[str, str]]]:
    """Map each AGI uprating group to the (split, row id) pairs the compile
    uprates in it.

    The groups start as the filing-season bins. A bound row that crosses a
    group edge without starting and ending on edges (the merged $25,000 to
    $35,000 row crosses the $30,000 edge) removes the edges inside it, until
    no row does. A row then sits in the narrowest group that contains it or,
    when the groups run finer (Table 2.5's $50,000-and-over bin), in its own
    interval, which the groups inside it cover. The compile merges groups
    rather than removing edges; the two give the same groups when the
    filing-season bins tile AGI, which
    ``test_filing_season_agi_bins_sum_to_the_filing_season_total`` checks.
    """

    season = [_agi_bounds(fact) for fact in _filing_season_bins(kind).values()]
    edges = sorted({bound for bounds in season for bound in bounds})
    rows = [
        (split, row_id, bounds)
        for split, split_rows in _bound_rows(kind).items()
        for row_id, (bounds, _) in split_rows.items()
        if row_id != "total"
    ]
    changed = True
    while changed:
        changed = False
        for _, _, (lower, upper) in rows:
            inside = [edge for edge in edges if lower < edge < upper]
            if inside and not (lower in edges and upper in edges):
                edges = [edge for edge in edges if edge not in inside]
                changed = True
    coarsened = list(zip(edges, edges[1:], strict=False))
    groups: dict[tuple[float, float], list[tuple[str, str]]] = {}
    for split, row_id, bounds in rows:
        containing = [group for group in coarsened if _inside(bounds, group)]
        group = (
            min(containing, key=lambda item: item[1] - item[0])
            if containing
            else bounds
        )
        groups.setdefault(group, []).append((split, row_id))
    return groups


def test_fixture_holds_every_table_2_5_and_filing_season_row() -> None:
    """Four child splits x (28 AGI bins + total) x 2 measures, the 8-measure
    all-returns total row, and 16 filing-season AGI bins plus a total for each
    measure: 274 rows."""

    assert len(_feed_rows()) == 274
    for kind in _KINDS:
        rows = _split_rows(kind)
        assert set(rows) == set(_SPLITS)
        for split_rows in rows.values():
            assert len(split_rows) == 29
            assert "total" in split_rows
        assert len(_filing_season_bins(kind)) == 16
    assert (
        sum(
            1
            for fact in _feed_rows()
            if fact["layout"]["record_set_id"] == _ALL_RETURNS_RECORD_SET
        )
        == 8
    )


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_each_child_split_s_agi_bins_tile_agi_and_sum_to_its_total_row(
    kind: str,
) -> None:
    """In each split, the 28 Table 2.5 AGI bins tile AGI and add to the split's
    total row within SOI rounding, so the extraction neither drops a bin, reads
    one twice nor takes one from the total row. A sum can't see two bins'
    values swapped; the verbatim-feed check and the extraction's row guards
    catch that."""

    for split, rows in _split_rows(kind).items():
        bins = {row_id: fact for row_id, fact in rows.items() if row_id != "total"}
        edges = sorted(_agi_bounds(fact) for fact in bins.values())
        assert edges[0][0] == -float("inf")
        assert edges[-1][1] == float("inf")
        assert all(
            left[1] == right[0] for left, right in zip(edges, edges[1:], strict=False)
        ), split
        assert _agi_bounds(rows["total"]) == (-float("inf"), float("inf"))
        gap = sum(fact["value"] for fact in bins.values()) - rows["total"]["value"]
        assert abs(gap) <= _ROUNDING[kind], (split, gap)


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_filing_season_agi_bins_sum_to_the_filing_season_total(kind: str) -> None:
    """Source check for the uprating index: the 16 TY2024 filing-season AGI
    bins tile AGI and add to its all-returns total within SOI rounding."""

    bins = _filing_season_bins(kind)
    edges = sorted(_agi_bounds(fact) for fact in bins.values())
    assert edges[0][0] == -float("inf")
    assert edges[-1][1] == float("inf")
    assert all(
        left[1] == right[0] for left, right in zip(edges, edges[1:], strict=False)
    )
    total = _single(_FILING_SEASON_TOTAL, _KINDS[kind][1])["value"]
    assert abs(sum(fact["value"] for fact in bins.values()) - total) <= _ROUNDING[kind]


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_child_split_totals_add_to_the_all_returns_total(kind: str) -> None:
    """The four splits' total rows add to Table 2.5's all-returns total
    (24,439,936 returns; $66,270,000,000), so no split holds only a subset such
    as one filing status or one component of the credit."""

    split_total = sum(rows["total"]["value"] for rows in _split_rows(kind).values())
    all_returns = _single(_ALL_RETURNS_RECORD_SET, _KINDS[kind][1])["value"]
    assert abs(split_total - all_returns) <= _ROUNDING[kind]


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_ty2023_table_2_5_totals_reconcile_with_irs_published(kind: str) -> None:
    """The TY2023 child-split total rows add to within 10% of the IRS-published
    TY2023 EITC (more than 23 million returns, about $64 billion): 24.44
    million is 6.3% above it and $66.27 billion 3.5% above."""

    total = sum(rows["total"]["value"] for rows in _split_rows(kind).values())
    assert total == pytest.approx(_IRS_PUBLISHED[2023][kind], rel=_PUBLISHED_TOLERANCE)


def test_ty2023_child_split_shares_match_irs_published() -> None:
    """Childless 27.7%, one child 35.5%, two 23.7%, three or more 13.1%,
    against IRS's published 27/36/24/13."""

    rows = _split_rows("returns")
    total = sum(split_rows["total"]["value"] for split_rows in rows.values())
    for split, share in _IRS_TY2023_CHILD_SHARES.items():
        assert rows[split]["total"]["value"] / total == pytest.approx(
            share, abs=_SHARE_TOLERANCE
        ), split


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_compiled_total_rows_sum_to_the_filing_season_and_irs_published(
    kind: str,
) -> None:
    """The regression for microcosm#1227 on real data. Each total row takes
    one factor, the filing-season EITC over the source total rows (23,837,149
    / 24,439,936 and $69,041,650,000 / $66,269,999,000), so the four total rows
    add to the filing season's AGI bins, as their 111 compiled bins do, and to
    within 10% of IRS's published TY2024 EITC (about 24 million returns and
    $69.6 billion; the compiled values are 0.7% and 0.8% below). The bins are
    the 112 Table 2.5 child rows less one: the three-or-more split's two
    misplaced rows compile as one."""

    by_name = _compiled_real()
    compiled = _compiled_split_rows(by_name, kind)
    source = _split_rows(kind)
    active = sum(fact["value"] for fact in _filing_season_bins(kind).values())
    source_total = sum(rows["total"]["value"] for rows in source.values())

    totals = {split: rows["total"] for split, rows in compiled.items()}
    bins = [
        spec
        for rows in compiled.values()
        for row_id, spec in rows.items()
        if row_id != "total"
    ]
    assert len(bins) == 4 * 28 - 1
    assert sum(spec.value for spec in totals.values()) == pytest.approx(
        active, rel=1e-12
    )
    assert sum(spec.value for spec in bins) == pytest.approx(active, rel=1e-12)
    for split, spec in totals.items():
        assert float(spec.metadata["uprating_factor"]) == pytest.approx(
            active / source_total, rel=1e-12
        )
        assert spec.metadata["uprating_agi_lower_bound"] == "-inf"
        assert spec.metadata["uprating_agi_upper_bound"] == "inf"
        assert spec.value == pytest.approx(
            source[split]["total"]["value"] * active / source_total, rel=1e-12
        )
    compiled_total = sum(spec.value for spec in totals.values())
    assert compiled_total == pytest.approx(
        _IRS_PUBLISHED[2024][kind], rel=_PUBLISHED_TOLERANCE
    )


def test_merged_rows_match_the_reviewed_declaration_and_the_fixture() -> None:
    """``_MERGED_ROWS`` names the rows ``US_SOI_MISPLACED_ROW_MERGES``
    declares, and the fixture carries each member at the value the declaration
    reviewed. The $30,000 to $35,000 three-or-more cell prints 0 returns and
    $0 between 407,149 returns ($2,919,017,000) at $20,000 to $25,000 and
    312,506 ($1,408,203,000) at $35,000 to $40,000, while the $25,000 to
    $30,000 cell holds 635,160 returns ($3,812,252,000)."""

    from microcosm.build.us_runtime.fiscal_targets import US_SOI_MISPLACED_ROW_MERGES

    declared = {
        (merge.record_set_id, merge.measure_id): merge
        for merge in US_SOI_MISPLACED_ROW_MERGES
    }
    assert set(declared) == {
        (f"{_CHILD_RECORD_SET}{split}", measure)
        for split in _MERGED_ROWS
        for measure, _ in _KINDS.values()
    }
    neighbours = {
        "returns": (407_149, 312_506),
        "amount": (2_919_017_000, 1_408_203_000),
    }
    for kind, (measure, _) in _KINDS.items():
        for split, (members, merged_id) in _MERGED_ROWS.items():
            merge = declared[(f"{_CHILD_RECORD_SET}{split}", measure)]
            assert merge.source_row_ids == members
            assert merge.merged_row_id == merged_id
            rows = _split_rows(kind)[split]
            assert (
                tuple(rows[row_id]["value"] for row_id in members)
                == merge.published_values
            )
            assert rows["30k_to_35k"]["value"] == 0
            assert (
                rows["20k_to_25k"]["value"],
                rows["35k_to_40k"]["value"],
            ) == neighbours[kind]


_IDENTITY_KEYS = frozenset(
    {
        "ledger_source_record_id",
        "ledger_fact_key",
        "ledger_aggregate_fact_key",
        "ledger_semantic_fact_key",
        "ledger_legacy_fact_key",
        "ledger_dimension_set_key",
        "ledger_universe_constraint_set_key",
        "ledger_fact_label",
    }
)


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_compiled_merged_row_takes_the_joined_group_s_factor(kind: str) -> None:
    """The compile drops the two misplaced rows and binds one $25,000 to
    $35,000 row at their sum, uprated by the filing season's $25,000 to
    $40,000 EITC over every split's Table 2.5 EITC in that range. The row
    names its members and carries none of their per-fact identities."""

    by_name = _compiled_real()
    measure = _KINDS[kind][0]
    split = "three_or_more_qualifying_children"
    members, merged_id = _MERGED_ROWS[split]
    member_ids = [
        f"{_CHILD_RECORD_SET}{split}.{row_id}.{measure}" for row_id in members
    ]
    assert not [name for name in member_ids if name in by_name]
    spec = by_name[f"{_CHILD_RECORD_SET}{split}.{merged_id}.{measure}"]

    season = _filing_season_bins(kind)
    group_active = season["25k_to_30k"]["value"] + season["30k_to_40k"]["value"]
    group_source = sum(
        value
        for rows in _bound_rows(kind).values()
        for bounds, value in rows.values()
        if _inside(bounds, (25_000.0, 40_000.0))
    )
    factor = group_active / group_source
    source_value = sum(_split_rows(kind)[split][row_id]["value"] for row_id in members)
    assert spec.value == pytest.approx(source_value * factor, rel=1e-12)
    metadata = spec.metadata
    assert float(metadata["uprating_factor"]) == pytest.approx(factor, rel=1e-12)
    assert (
        metadata["uprating_agi_lower_bound"],
        metadata["uprating_agi_upper_bound"],
    ) == ("25000", "40000")
    assert metadata["uprating_index_source_record_ids"] == ",".join(
        _record_id(season[row_id]) for row_id in ("25k_to_30k", "30k_to_40k")
    )
    assert (metadata["agi_lower_bound"], metadata["agi_upper_bound"]) == (
        "25000.0",
        "35000.0",
    )
    agi_filter = "ledger_filter_us:statutes/26/62#adjusted_gross_income"
    assert (
        metadata[f"{agi_filter}_lower_bound"],
        metadata[f"{agi_filter}_upper_bound"],
    ) == ("25000", "35000")
    assert metadata["ledger_filter_income_range"] == merged_id
    assert metadata["ledger_layout_groupby_value_id"] == merged_id
    assert metadata["ledger_layout_groupby_value_label"] == "$25,000 under $35,000"
    assert metadata["derived_operation"] == "sum_of_misplaced_soi_rows"
    assert metadata["derived_source_record_ids"] == ",".join(member_ids)
    assert not _IDENTITY_KEYS & set(metadata)
    assert {
        (dimension.id, dimension.value_id, dimension.value_label)
        for dimension in spec.hierarchy.dimensions
    } == {
        ("us:statutes/26/62#adjusted_gross_income", merged_id, "$25,000 under $35,000"),
        ("income_range", merged_id, "$25,000 under $35,000"),
    }
    assert spec.hierarchy.target.id == spec.name == spec.measure


def test_compiled_table_2_5_rows_pass_the_materializer_filter_guards() -> None:
    """The release materializer accepts every compiled Table 2.5 child row's
    Ledger filters, the merged rows included, and its restated-AGI guard reads
    the merged rows: an upper bound that disagrees with the compiled one
    stops it."""

    builder = _load_repo_tool("build_us_fiscal_refresh_release")
    rows = [
        spec
        for name, spec in _compiled_real().items()
        if name.startswith(_CHILD_RECORD_SET)
    ]
    builder._assert_supported_ledger_filter_metadata(rows)
    merged = [spec for spec in rows if ".25k_to_35k." in spec.name]
    assert len(merged) == 2
    for spec in merged:
        assert builder._unsupported_soi_ledger_filters(spec.metadata) == ()
        moved = replace(
            spec,
            metadata={
                **spec.metadata,
                "ledger_filter_us:statutes/26/62#adjusted_gross_income_upper_bound": (
                    "30000"
                ),
            },
        )
        with pytest.raises(RuntimeError, match=spec.name):
            builder._assert_supported_ledger_filter_metadata([moved])


def _with_filing_season(
    values: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    """Return the fixture rows with ``values`` (a map from kind and AGI row id)
    in place of the filing-season AGI bins, and each filing-season total equal
    to their sum."""

    measure_kind = {measures[1]: kind for kind, measures in _KINDS.items()}
    totals = {
        kind: sum(
            value for (value_kind, _), value in values.items() if value_kind == kind
        )
        for kind in _KINDS
    }
    rows = []
    for fact in _feed_rows():
        record_id = _record_id(fact)
        kind = measure_kind.get(fact["layout"]["measure_id"])
        if record_id.startswith(_FILING_SEASON_BINS):
            fact = {**fact, "value": values[(kind, fact["layout"]["groupby_value_id"])]}
        elif record_id.startswith(_FILING_SEASON_TOTAL):
            fact = {**fact, "value": totals[kind]}
        rows.append(fact)
    return rows


@st.composite
def _filing_seasons(draw):
    """Any nonnegative TY2024 filing-season EITC by AGI, returns and amount,
    with at least one return and one dollar in total."""

    values = {}
    for kind, ceiling in (("returns", 10**7), ("amount", 5 * 10**10)):
        for row_id in _filing_season_bins(kind):
            values[(kind, row_id)] = draw(st.integers(min_value=0, max_value=ceiling))
        if not any(
            value for (value_kind, _), value in values.items() if value_kind == kind
        ):
            values[(kind, "10k_to_15k")] = 1
    return values


@given(_filing_seasons())
@_SETTINGS
def test_compiled_agi_bins_and_total_rows_both_sum_to_any_filing_season(values) -> None:
    """For any filing-season EITC by AGI, on the real Table 2.5 layout:

    * the four total rows add to the filing season's total, and keep the
      TY2023 child split of the total rows;
    * the 111 bound AGI rows add to the same total, and within each uprating
      AGI group to the filing season's EITC in that group, keeping the group's
      TY2023 child split.

    The real layout covers every grouping path: Table 2.5 bins inside one
    filing-season bin, the $50,000-and-over bin, which spans seven, and the
    merged $25,000 to $35,000 row, which joins two filing-season bins into one
    group."""

    by_name = _compile(_with_filing_season(values))
    for kind in _KINDS:
        compiled = _compiled_split_rows(by_name, kind)
        source = _split_rows(kind)
        bound = _bound_rows(kind)
        season = {
            row_id: (_agi_bounds(fact), values[(kind, row_id)])
            for row_id, fact in _filing_season_bins(kind).items()
        }
        active = sum(value for _, value in season.values())
        source_total = sum(rows["total"]["value"] for rows in source.values())

        totals = sum(rows["total"].value for rows in compiled.values())
        assert totals == pytest.approx(active, rel=1e-9, abs=1e-6)
        for split, rows in compiled.items():
            assert rows["total"].value == pytest.approx(
                source[split]["total"]["value"] * active / source_total,
                rel=1e-9,
                abs=1e-6,
            )

        bins_total = 0.0
        for group, members in _uprating_groups(kind).items():
            group_active = sum(
                value for bounds, value in season.values() if _inside(bounds, group)
            )
            group_source = sum(bound[split][row_id][1] for split, row_id in members)
            group_compiled = 0.0
            for split, row_id in members:
                spec = compiled[split][row_id]
                assert spec.value == pytest.approx(
                    bound[split][row_id][1] * group_active / group_source,
                    rel=1e-9,
                    abs=1e-6,
                )
                group_compiled += spec.value
            assert group_compiled == pytest.approx(group_active, rel=1e-9, abs=1e-6)
            bins_total += group_compiled
        assert bins_total == pytest.approx(active, rel=1e-9, abs=1e-6)


@given(growth=st.integers(min_value=25, max_value=400))
@_SETTINGS
def test_uniform_growth_keeps_each_split_s_bins_summing_to_its_total_row(
    growth: int,
) -> None:
    """When every AGI group's EITC grows by one factor, each split's compiled
    AGI bins (28, or 27 for three or more children) add to its compiled total
    row within twice the source's
    rounding allowance (10 returns or $10,000) times the factor.

    The compile scales each total row by the filing-season bins' sum over the
    four total rows' sum, here the factor times all Table 2.5 bins over all
    total rows. A split's compiled gap (bins less total row) then equals
    ``factor * (g - t / T * G)``, where ``g`` is its source gap, ``t`` its
    source total row, ``T`` the four source total rows' sum and ``G`` the four
    source gaps' sum. On this fixture ``g - t / T * G`` stays within one
    allowance, so each compiled gap stays within one allowance times the
    factor.
    """

    factor = growth / 100
    values: dict[tuple[str, str], float] = {}
    for kind in _KINDS:
        bound = _bound_rows(kind)
        season = {
            row_id: _agi_bounds(fact)
            for row_id, fact in _filing_season_bins(kind).items()
        }
        for row_id in season:
            values[(kind, row_id)] = 0.0
        for group, members in _uprating_groups(kind).items():
            group_source = sum(bound[split][row_id][1] for split, row_id in members)
            # Put the whole group on one filing-season bin inside it; the
            # $50,000-and-over group sums its seven filing-season bins, the
            # $25,000 to $40,000 group its two.
            first = min(
                (row_id for row_id, bounds in season.items() if _inside(bounds, group)),
                key=lambda row_id: season[row_id],
            )
            values[(kind, first)] = group_source * factor

    by_name = _compile(_with_filing_season(values))
    for kind in _KINDS:
        for split, rows in _compiled_split_rows(by_name, kind).items():
            bins = sum(spec.value for row_id, spec in rows.items() if row_id != "total")
            assert bins == pytest.approx(
                rows["total"].value, abs=2 * _ROUNDING[kind] * factor
            ), (kind, split)


def test_fixture_rows_are_verbatim_pinned_feed_rows() -> None:
    """On a machine that has the pinned feed (CI has none), the fixture matches
    the feed's rows for these record sets byte for byte.

    It first checks the feed's sha256 against the pin in ``chronicle_feed.json``.
    It fails after a re-pin that changes any of those rows. To refresh the
    fixture, take the new feed's lines for the same record sets, sort them by
    ``lineage.source_record_id`` and write one per line. Compare the old and
    new rows field by field first, and check any changed value against the
    workbooks this module's docstring names.
    """

    feed_path = _load_repo_tool("build_us_target_parity_manifest").DEFAULT_FEED_PATH
    if not feed_path.exists():
        pytest.skip(f"pinned feed not present at {feed_path}")
    from microcosm.build.us_runtime.chronicle_feed import load_us_chronicle_feed

    pinned = load_us_chronicle_feed().facts_sha256
    with feed_path.open("rb") as feed:
        digest = hashlib.file_digest(feed, "sha256").hexdigest()
    assert digest == pinned, (
        f"{feed_path.name} has sha256 {digest}; chronicle_feed.json pins {pinned}"
    )
    record_sets = {fact["layout"]["record_set_id"] for fact in _feed_rows()}
    with feed_path.open(encoding="utf-8") as feed:
        feed_rows = sorted(
            line.rstrip("\n")
            for line in feed
            if json.loads(line)["layout"]["record_set_id"] in record_sets
        )
    fixture_rows = sorted(
        line.rstrip("\n")
        for line in _FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    assert fixture_rows == feed_rows, (
        f"{_FIXTURE.name} differs from {feed_path.name}'s rows for its record "
        "sets; refresh it as this test's docstring says"
    )
