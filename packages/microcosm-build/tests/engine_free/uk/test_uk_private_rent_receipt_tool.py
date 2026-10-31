"""The private-rent receipt's measures and disclosure control (microcosm#1262).

The receipt reads a licensed build, so these tests drive its measures with
small synthetic frames, facts and problems.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_TOOL = _TEST_PATHS.repository / "tools/uk_private_rent_receipt.py"
_SPEC = importlib.util.spec_from_file_location("uk_private_rent_receipt", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
tool = importlib.util.module_from_spec(_SPEC)
# Its dataclasses resolve their annotations through ``sys.modules``.
sys.modules[_SPEC.name] = tool
_SPEC.loader.exec_module(tool)


def _frame(rows: list[tuple]) -> pd.DataFrame:
    columns = (
        "weight",
        "region",
        "tenure_type",
        "rent",
        "num_bedrooms",
        "brma",
        "la_code",
        "source_key",
    )
    frame = pd.DataFrame(rows, columns=columns)
    frame["subrent"] = 0.0
    frame["housing_benefit_reported"] = 0.0
    frame["channel"] = "frs"
    return frame


def test_unweighted_shares_are_floored_and_small_sides_withheld() -> None:
    assert tool.floored_share(0, 100) == 0.0
    assert tool.floored_share(37, 300) == 0.12
    # The floor of three: one or two records, or a share leaving out one or
    # two, read "<3"; three read exactly.
    assert tool.floored_share(2, 100) == "<3"
    assert tool.floored_share(98, 100) == "<3"
    assert tool.floored_share(3, 100) == 0.03
    assert tool.floored_share(1, 0) is None


def test_weighted_shares_count_distinct_source_households_on_both_sides() -> None:
    weight = np.ones(40)
    many = np.array([f"s{i}" for i in range(40)], dtype=object)
    mask = np.arange(40) < 20
    assert tool.weighted_share(mask, weight, many) == 0.5
    # Twenty clones of two source households are still two households.
    few = np.array([f"s{i % 2}" if i < 20 else f"t{i}" for i in range(40)])
    assert tool.weighted_share(mask, weight, few) == "<3"
    assert tool.weighted_share(~mask, weight, few) == "<3"


def test_one_small_cell_takes_a_complement_with_it() -> None:
    cells = {
        "B": {"source_households": 2, "mean_monthly_rent": 900.0},
        "C": {"source_households": 40, "mean_monthly_rent": 1000.0},
        "D": {"source_households": 25, "mean_monthly_rent": 1100.0},
    }
    out = tool.suppress_cells(cells)
    assert out["B"] == {"withheld": "<3"}
    assert out["D"] == {"withheld": "complement"}
    assert out["C"] == {"source_households": 40, "mean_monthly_rent": 1000.0}
    two_small = tool.suppress_cells(
        {**cells, "C": {"source_households": 1, "mean_monthly_rent": 1000.0}}
    )
    # Two and one households are three together: no complement is needed.
    assert two_small["B"] == two_small["C"] == {"withheld": "<3"}
    assert two_small["D"]["source_households"] == 30


def test_thin_cells_resting_on_under_three_together_take_a_complement():
    """Two withheld cells of one household each are two together."""

    out = tool.suppress_cells(
        {
            "B": {"source_households": 1, "mean_monthly_rent": 900.0},
            "C": {"source_households": 1, "mean_monthly_rent": 950.0},
            "D": {"source_households": 25, "mean_monthly_rent": 1100.0},
            "E": {"source_households": 40, "mean_monthly_rent": 1300.0},
        }
    )
    assert out["B"] == out["C"] == {"withheld": "<3"}
    assert out["D"] == {"withheld": "complement"}
    assert out["E"]["source_households"] == 40


def test_a_cell_with_rows_but_no_counted_carrier_reads_under_three():
    """microcosm#1305 re-review: zero counted carriers is fewer than three."""

    assert tool.suppress_cells(
        {"all": {"source_households": 0, "mean_monthly_rent": 900.0}}
    ) == {"all": {"withheld": "<3"}}
    weight = np.ones(20)
    mask = np.arange(20) < 10
    blank = np.array([""] * 10 + [f"s{i}" for i in range(10)], dtype=object)
    assert tool.weighted_share(mask, weight, blank) == "<3"
    assert tool.unweighted_share(mask, blank) == "<3"
    # An empty side has nothing to describe.
    assert tool.weighted_share(np.ones(20, dtype=bool), weight, blank[::-1]) == 1.0


def _rent_weighting(carriers=True, *, thin_brma=False, per_group=12):
    """``per_group`` English, Welsh and Scottish BRMA rent rows, and a tenure
    row, over 40 households.

    The English rows cover households 0-19, the Welsh ones 20-39 and the
    BRMA rows all forty; with ``thin_brma`` the first BRMA row covers 0-1
    only, under the floor of three. Each household is its own source household.
    """

    metadata = [
        *(
            {"contract_target_id": "ons.rent.private_rent", "area_code": f"E06{i:06d}"}
            for i in range(per_group)
        ),
        *(
            {"contract_target_id": "ons.rent.private_rent", "area_code": f"W06{i:06d}"}
            for i in range(per_group)
        ),
        *(
            {
                "contract_target_id": "ons.rent.private_rent_brma",
                "area_code": f"S33{i:06d}",
            }
            for i in range(per_group)
        ),
        {"contract_target_id": "ons.tenure.private_rent", "area_code": "E06000001"},
    ]
    rows = np.zeros((len(metadata), 40))
    rows[:per_group, :20] = 1.0
    rows[per_group : 2 * per_group, 20:] = 1.0
    rows[2 * per_group : 3 * per_group] = 1.0
    if thin_brma:
        rows[2 * per_group, 2:] = 0.0
    rows[-1] = 5.0
    weights = np.where(np.arange(40) < 20, 105.0, 70.0) / 20
    targets = np.array(
        [100.0] * (2 * per_group)
        + [10.5 if thin_brma and i == 0 else 175.0 for i in range(per_group)]
        + [1.0]
    )
    return tool.Weighting(
        name="design",
        matrix=sparse.csr_array(rows),
        weights=weights,
        names=tuple(f"row{i}" for i in range(len(metadata))),
        targets=targets,
        metadata=tuple(metadata),
        carriers=(
            np.array([f"s{i}" for i in range(40)], dtype=object) if carriers else None
        ),
    )


def test_rent_rows_are_scored_by_family_and_nation() -> None:
    summary, rows = tool.rent_row_fit(_rent_weighting())
    assert len(rows) == 36
    assert [row["source_households"] for row in rows[::12]] == [20, 20, 40]
    assert summary["all"]["rows"] == 36
    assert summary["all"]["rows_over_25pct"] == 12
    assert summary["ons.rent.private_rent England"]["share_within_10pct"] == 1.0
    assert summary["ons.rent.private_rent Wales"]["rows_over_25pct"] == 12
    assert summary["ons.rent.private_rent_brma Scotland"]["median_abs_error"] == 0.0
    assert "rows_withheld" not in summary["all"]


def test_a_rent_row_on_fewer_than_three_households_is_counted_never_summarised():
    """microcosm#1305 re-review: the rent rows follow the build's BRMA rule.

    With the thin row in its groups, the median and the within-10% count
    would give its error back against the published rows.
    """

    summary, rows = tool.rent_row_fit(_rent_weighting(thin_brma=True))
    assert rows[24]["source_households"] == 2
    assert summary["all"]["rows"] == 35 and summary["all"]["rows_withheld"] == 1
    brma = summary["ons.rent.private_rent_brma Scotland"]
    assert brma["rows"] == 11 and brma["rows_withheld"] == 1
    assert brma["median_abs_error"] == 0.0
    # Without the households' keys no row can be floored, so none is summarised.
    summary, rows = tool.rent_row_fit(_rent_weighting(carriers=False))
    assert summary == {"withheld": tool.NO_SOURCE_KEYS} and len(rows) == 36


def test_a_group_of_fewer_than_ten_published_rows_reads_withheld():
    """The third review of #1305: a median over one row is that row's error."""

    summary, _ = tool.rent_row_fit(_rent_weighting(thin_brma=True, per_group=3))
    assert summary["ons.rent.private_rent_brma Scotland"] == {
        "withheld": "fewer than 10 published rows",
        "rows": 2,
        "rows_withheld": 1,
    }
    assert summary["ons.rent.private_rent England"] == {
        "withheld": "fewer than 10 published rows",
        "rows": 3,
    }


def test_all_reads_withheld_while_a_group_it_covers_does():
    """The fourth review of #1305: "all" less England and Wales gave a floored
    Scottish group's counts back."""

    summary, _ = tool.rent_row_fit(_rent_weighting(thin_brma=True, per_group=10))
    scotland = summary["ons.rent.private_rent_brma Scotland"]
    assert scotland["withheld"] == "fewer than 10 published rows"
    assert scotland["rows"] == 9
    assert summary["ons.rent.private_rent England"]["share_within_10pct"] == 1.0
    assert summary["all"] == {
        "withheld": tool.ALL_WITH_A_GROUP_WITHHELD,
        "rows": 29,
        "rows_withheld": 1,
    }


def test_carriers_follow_the_problem_axis_and_its_positive_weights():
    keys = pd.Series(["k1", "k2", "k3"], index=pd.Index([10, 11, 12]))
    out = tool.carriers_on([12, 10, 13], np.array([1.0, 0.0, 0.0]), keys)
    assert out.tolist() == ["k3", "", ""]
    # A positive-weight household without a key leaves the axis unkeyed.
    assert tool.carriers_on([12, 13], np.array([1.0, 1.0]), keys) is None
    assert tool.carriers_on([12], np.ones(1), None) is None


def _pipr_fact(area: str, level: str, month: str, bedrooms: str, value: float) -> str:
    return json.dumps(
        {
            "dimensions": {"bedrooms": bedrooms},
            "geography": {"id": area, "level": level},
            "layout": {"measure_id": "ons_rent_private_rent"},
            "period": {"type": "month", "value": month},
            "source": {"vintage": "pipr_test_edition"},
            "value": value,
        }
    )


def test_pipr_means_need_all_twelve_months_and_follow_the_aliases(tmp_path) -> None:
    lines = [
        _pipr_fact("E08000038", "local_authority", f"2025-{m:02d}", "two bed", 600 + m)
        for m in range(1, 13)
    ]
    lines += [
        _pipr_fact("E06000001", "local_authority", f"2025-{m:02d}", "one bed", 500)
        for m in range(1, 12)
    ]
    lines.append(
        _pipr_fact("E06000002", "local_authority", "2025-01", "two bed", 1).replace(
            "pipr_test_edition", "another_edition"
        )
    )
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text("\n".join(lines) + "\n")
    means = tool.pipr_bedroom_means(tmp_path, edition="pipr_test_edition")
    # Barnsley's 2025 code reads as the build's E08000016.
    assert means == {("la", "E08000016", "C"): pytest.approx(606.5)}


def test_renters_are_gb_private_with_positive_rent_and_carry_a_category() -> None:
    frame = _frame(
        [
            (
                1.0,
                "LONDON",
                "RENT_PRIVATELY",
                12_000.0,
                1,
                "INNER_EAST_LONDON",
                "E09000002",
                "a",
            ),
            (
                1.0,
                "LONDON",
                "RENT_PRIVATELY",
                0.0,
                2,
                "INNER_EAST_LONDON",
                "E09000002",
                "b",
            ),
            (
                1.0,
                "LONDON",
                "OWNED_OUTRIGHT",
                0.0,
                3,
                "INNER_EAST_LONDON",
                "E09000002",
                "c",
            ),
            (
                1.0,
                "NORTHERN_IRELAND",
                "RENT_PRIVATELY",
                6_000.0,
                4,
                "BELFAST",
                "N09000003",
                "d",
            ),
            (1.0, "WALES", "RENT_PRIVATELY", 9_000.0, 4, "CARDIFF", "W06000015", "e"),
        ]
    )
    renters = tool.private_renters(frame)
    assert list(renters["source_key"]) == ["a", "e"]
    assert list(renters["category"]) == ["B", "E"]
    assert list(renters["nation"]) == ["England", "Wales"]


def test_a_mix_matching_the_census_has_no_distance() -> None:
    renters = tool.with_carriers(
        _frame(
            [
                (1.0, "WALES", "RENT_PRIVATELY", 9_000.0, 2, brma, "W06000015", key)
                for key, brma in [(f"c{i}", "CARDIFF") for i in range(9)]
                + [(f"s{i}", "SWANSEA") for i in range(3)]
            ]
        )
    ).assign(category="C", nation="Wales")
    counts = {("WALES", "C"): {"CARDIFF": 300.0, "SWANSEA": 100.0}}
    summary, audit = tool.brma_mix(renters, counts)
    assert summary == {"Wales": 0.0}
    assert audit["WALES"]["census_share"] == {"CARDIFF": 0.75, "SWANSEA": 0.25}


def test_the_matched_change_reads_each_row_against_its_spine_value() -> None:
    rows = [
        (1.0, "WALES", "RENT_PRIVATELY", 11_000.0, 2, "CARDIFF", "W06000015", f"s{i}")
        for i in range(12)
    ]
    frame = _frame(rows).assign(spine_rent=10_000.0, spine_brma="CARDIFF")
    frame.loc[0, "brma"] = "SWANSEA"
    out = tool.matched_change(frame)
    assert out["gb_rows_redrawn"] == 1.0
    assert out["gb_rows_whose_brma_changed"] == "<3"
    assert out["region_level_at_design_weights"] == {"WALES": 1.1}


def test_a_pool_table_without_the_spine_values_is_refused(tmp_path) -> None:
    path = tmp_path / "pool.npz"
    np.savez(path, weight=np.ones(2), region=np.array(["WALES", "WALES"]))
    with pytest.raises(ValueError, match="lacks"):
        tool.load_pool(path)


# ----------------------------------------------------------------------------
# The quotable summary (microcosm#1305 review)


def _renter(key: str, *, region="LONDON", bedrooms=1, weight=1.0, rent=12_000.0):
    return (
        weight,
        region,
        "RENT_PRIVATELY",
        rent,
        bedrooms,
        "INNER_EAST_LONDON",
        "E09000002",
        key,
    )


def _summary(frame: pd.DataFrame, **kwargs) -> dict:
    summary, _ = tool.measure_households(frame, pipr={}, counts={}, **kwargs)
    return summary


def test_a_thin_region_category_cell_reads_under_three_and_withholds_a_complement():
    rows = (
        [_renter(f"b{i}", bedrooms=1) for i in range(12)]
        + [_renter(f"c{i}", bedrooms=2) for i in range(15)]
        + [_renter(f"d{i}", bedrooms=3) for i in range(2)]
        + [_renter(f"e{i}", bedrooms=4) for i in range(20)]
    )
    london = _summary(_frame(rows))["rents_by_cell"]["region"]["LONDON"]
    assert london["D"] == {"withheld": "<3"}
    assert london["B"] == {"withheld": "complement"}
    assert london["C"]["source_households"] == 20
    assert london["all"]["mean_monthly_rent"] == 1000.0


def test_district_groups_need_ten_cells_of_three_households():
    pipr = {("la", f"E0900000{i}", "B"): 1_000.0 for i in range(1, 10)}
    rows = [
        (*_renter(f"k{area}-{i}")[:6], f"E0900000{area}", f"k{area}-{i}")
        for area in range(1, 10)
        for i in range(12)
    ]
    rows += [(*_renter(f"thin{i}")[:6], "E09000001", f"thin{i}") for i in range(3)]
    summary, audit = tool.measure_households(_frame(rows), pipr=pipr, counts={})
    group = summary["pipr_by_district_and_bedrooms"]["la England"]
    # Nine districts pass the household floor: too few to summarise.
    assert group == {
        "withheld": "fewer than 10 cells",
        "cells_under_floor_source_households": 0,
    }
    assert len(audit["pipr_by_district_and_bedrooms"]) == 9


def test_the_top_one_percent_share_needs_three_rows_and_three_households():
    # 2,000 renters: the top 20 rows are clones of two households.
    rows = [_renter(f"r{i}") for i in range(1_980)]
    rows += [_renter(f"top{i % 2}", weight=50.0) for i in range(20)]
    weights = _summary(_frame(rows))["weights"]
    assert weights["top_1pct_weight_share"] == "<3"
    rows[-20:] = [_renter(f"top{i}", weight=50.0) for i in range(20)]
    assert isinstance(_summary(_frame(rows))["weights"]["top_1pct_weight_share"], float)
    assert _summary(_frame([_renter(f"r{i}") for i in range(2)]))["weights"] == {
        "withheld": "<3"
    }


def test_coherence_counts_households_not_clone_rows():
    # 2 of 800 households, 25 clones each, report HB above their rent: 50
    # rows, but two households, under the floor of three.
    frame = _frame([_renter(f"h{i}") for i in range(800) for _ in range(25)])
    frame.loc[
        frame["source_key"].isin([f"h{i}" for i in range(2)]),
        "housing_benefit_reported",
    ] = 20_000.0
    out = _summary(frame)["coherence"]["housing_benefit_reported_above_rent"]
    assert out == {"weighted_share": "<3", "unweighted_share": "<3"}


def test_spine_values_count_frs_channel_households_only():
    frame = _frame(
        [_renter(f"f{i}") for i in range(2)] + [_renter(f"s{i}") for i in range(6)]
    )
    frame.loc[2:, "channel"] = "spi"
    every = _summary(frame)["rents_by_cell"]["nation"]["England"]
    # All eight households carry the cell, so it reads "3-9", not "<3".
    assert every["source_households"] == "3-9"
    frs = _summary(frame, frs_carriers=True)["rents_by_cell"]["nation"]["England"]
    assert frs == {"withheld": "<3"}
    # A cell of SPI rows only has no FRS-channel carrier: it reads "<3".
    spi = frame.assign(channel="spi")
    assert _summary(spi, frs_carriers=True)["rents_by_cell"]["nation"] == {
        "England": {"withheld": "<3"}
    }


def test_zero_weight_rows_carry_nothing():
    # Twelve rows, ten of them of zero weight: two carriers, under the floor.
    frame = _frame([_renter(f"r{i}") for i in range(12)])
    frame.loc[:9, "weight"] = 0.0
    cell = _summary(frame)["rents_by_cell"]["nation"]["England"]
    assert cell == {"withheld": "<3"}


def _work_tree(path):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    return path


def test_licensed_outputs_are_refused_inside_any_git_work_tree(tmp_path):
    from microcosm.build.uk_runtime.receipt_counts import outside_work_trees

    with pytest.raises(SystemExit, match="inside the git work tree"):
        outside_work_trees(_TEST_PATHS.repository / "audit-1262", what="the audit")
    other = _work_tree(tmp_path / "other-checkout") / "not" / "made"
    with pytest.raises(SystemExit, match="inside the git work tree"):
        outside_work_trees(other, what="the audit")
    assert (
        outside_work_trees(tmp_path / "audit-1262", what="the audit")
        == (tmp_path / "audit-1262").resolve()
    )


def _load_script(name: str, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_writer_refuses_a_work_tree_before_any_work(tmp_path):
    """The guard runs first: no run, store, H5 or pool is read before it."""

    missing = tmp_path / "missing"
    inside = _work_tree(tmp_path / "checkout") / "audit"
    with pytest.raises(SystemExit, match="inside the git work tree"):
        tool.main(
            ["--run", str(missing), "--ledger-facts", str(missing)]
            + ["--audit-dir", str(inside)]
        )
    pool = _load_script(
        "uk_private_rent_pool", _TEST_PATHS.repository / "tools/uk_private_rent_pool.py"
    )
    with pytest.raises(SystemExit, match="inside the git work tree"):
        pool.main(
            ["--run", str(missing), "--graph-store", str(missing)]
            + ["--out", str(inside / "pool.npz")]
        )
    engine = _load_script(
        "engine_rent_diagnostics",
        _TEST_PATHS.repository
        / "docs/evidence/uk-private-rent-1262/scripts/engine_rent_diagnostics.py",
    )
    with pytest.raises(SystemExit, match="inside the git work tree"):
        engine.main(["--h5", str(missing), "--out", str(inside / "engine.json")])


def test_the_engine_report_refuses_an_unpinned_engine_before_reading(
    tmp_path, monkeypatch
):
    """#1308's freeze: the report measures the pinned policyengine-uk only."""

    from microcosm.build.uk_runtime import hmrc_uprating

    def refuse() -> str:
        raise RuntimeError("policyengine-uk 0.0 is installed, but the build is pinned")

    monkeypatch.setattr(hmrc_uprating, "assert_installed_engine_is_pinned", refuse)
    engine = _load_script(
        "engine_rent_diagnostics",
        _TEST_PATHS.repository
        / "docs/evidence/uk-private-rent-1262/scripts/engine_rent_diagnostics.py",
    )
    missing = tmp_path / "missing.h5"
    with pytest.raises(RuntimeError, match="pinned"):
        engine.main(["--h5", str(missing), "--out", str(tmp_path / "engine.json")])


def _synthetic_run(tmp_path):
    """A run directory, its pool table and an empty feed for ``main``.

    Thirty London private renters in category B. Two are FRS-channel
    households and 28 SPI-channel rows, so the cell's spine values rest on
    two FRS carriers, under the floor of three, and its built values on thirty
    source households. The local-authority rent row covers all thirty; the
    BRMA row two.
    """
    pytest.importorskip("tables")
    from microcosm.calibrate import Target, TargetSet, build_constraint_matrix
    from microcosm.calibrate.artifacts import encode_problem
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    n = 30
    ids = np.arange(100, 100 + n, dtype=np.int64)
    keys = np.array([f"2024:{i}" for i in range(n)], dtype=object)
    channel = np.array(["frs"] * 2 + ["spi"] * 28)
    rent = np.full(n, 12_000.0)
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_weight": np.ones(n),
            "region": "LONDON",
            "tenure_type": "RENT_PRIVATELY",
            "rent": rent,
            "subrent": 0.0,
            "num_bedrooms": 1.0,
            "brma": "INNER_EAST_LONDON",
            "la_code_oa": "E09000002",
            "source_household_key": keys.astype(str),
            "household_support_channel": channel,
        }
    )
    run = tmp_path / "run"
    run.mkdir()
    h5 = run / "microcosm_uk_2024_25_local.h5"
    household.to_hdf(h5, key="household", format="table")
    pd.DataFrame(
        {"person_household_id": ids, "housing_benefit_reported": np.zeros(n)}
    ).to_hdf(h5, key="person", format="table")
    frame = Frame(
        {
            "person": pd.DataFrame({"person_id": ids, "person_household_id": ids}),
            "household": household[["household_id", "rent"]],
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )
    thin = np.arange(n) < 2
    targets = TargetSet(
        [
            Target("rent_la", "household", lambda f: rent, 360_000.0),
            Target("rent_brma", "household", lambda f: rent * thin, 24_000.0),
        ]
    )
    (run / "uk.full.problem.problem.artifact").write_bytes(
        encode_problem(
            build_constraint_matrix(frame, targets, weight_entity="household"),
            entity_ids=ids.tolist(),
            target_metadata=[
                {
                    "contract_target_id": "ons.rent.private_rent",
                    "area_code": "E09000002",
                },
                {
                    "contract_target_id": "ons.rent.private_rent_brma",
                    "area_code": "S33000001",
                },
            ],
            bindings={},
        )
    )
    pool = tmp_path / "pool.npz"
    np.savez(
        pool,
        household_id=ids,
        weight=np.ones(n),
        region=np.full(n, "LONDON"),
        tenure_type=np.full(n, "RENT_PRIVATELY"),
        rent=rent,
        subrent=np.zeros(n),
        num_bedrooms=np.ones(n),
        brma=np.full(n, "INNER_EAST_LONDON"),
        la_code=np.full(n, "E09000002"),
        source_key=keys.astype(str),
        channel=channel,
        housing_benefit_reported=np.zeros(n),
        spine_rent=rent,
        spine_brma=np.full(n, "INNER_EAST_LONDON"),
    )
    facts = tmp_path / "facts.jsonl"
    facts.write_text("")
    return run, pool, facts


def test_main_floors_rent_rows_and_spine_values_on_a_synthetic_run(tmp_path):
    """microcosm#1305 re-review: the summary writer's floors, end to end."""

    run, pool, facts = _synthetic_run(tmp_path)
    audit = tmp_path / "audit"
    assert (
        tool.main(
            ["--run", str(run), "--ledger-facts", str(facts), "--pool", str(pool)]
            + ["--pipr-edition", "synthetic", "--audit-dir", str(audit)]
        )
        == 0
    )
    head = json.loads((audit / "private_rent_summary.json").read_text())["runs"]["head"]
    assert head["rent_rows"]["design"]["all"]["rows"] == 1
    assert head["rent_rows"]["design"]["all"]["rows_withheld"] == 1
    built = head["design_weights"]["rents_by_cell"]["region"]["LONDON"]
    assert built["B"]["source_households"] == 30
    spine = head["design_weights_spine_values"]["rents_by_cell"]["region"]["LONDON"]
    assert spine["B"] == {"withheld": "<3"}
    private = json.loads((audit / "private_rent_audit.json").read_text())
    rows = private["runs"]["head"]["rent_rows"]["design"]
    assert [row["source_households"] for row in rows] == [30, 2]
