"""The PIP points stage (microcosm#1095, P1 of the November event-data plan)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import pip_points
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.pip_points import (
    PIP_AGE_GROUPS,
    PIP_DAILY_LIVING_BANDS,
    PIP_MOBILITY_TOTALS,
    PIP_NOT_ASSESSED,
    PIP_POINTS_RESOURCE,
    PIP_POINTS_STAGE_NAME,
    UKPIPPointsStageTransform,
    draw_pip_points,
    load_pip_points_distributions,
    pip_points_distributions_from_rows,
    published_pip_points_shares,
    store_pip_points,
)
from microcosm.frame import WeightKind

_GROUPS = [name for name, _, _ in PIP_AGE_GROUPS]


def _rows(dl=lambda group, band: 100.0 / 12, mobility=lambda group, total: 100.0 / 11):
    rows = []
    for group in [*_GROUPS, "all"]:
        for band, _, _ in PIP_DAILY_LIVING_BANDS:
            rows.append(
                {
                    "concept": "dwp.pip_daily_living_points_share",
                    "measure_id": f"points_{band}",
                    "dimensions": {
                        "age_band": group,
                        "pip_daily_living_points_band": band,
                    },
                    "value": dl(group, band),
                }
            )
        for total in PIP_MOBILITY_TOTALS:
            rows.append(
                {
                    "concept": "dwp.pip_mobility_points_share",
                    "measure_id": f"points_{total}",
                    "dimensions": {"age_band": group, "pip_mobility_points": total},
                    "value": mobility(group, total),
                }
            )
        rows.append(
            {
                "concept": "dwp.pip_daily_living_points_claimants",
                "measure_id": "claimants",
                "dimensions": {"age_band": group},
                "value": 1000.0,
            }
        )
    return rows


def test_the_committed_tables_give_every_age_groups_distribution() -> None:
    distributions = load_pip_points_distributions()
    for group in [*_GROUPS, "all"]:
        values, weights = distributions["daily_living"][group]
        # The twelve bands spread over 0 to 40, and the open band at 41.
        assert values.tolist() == list(range(0, 42))
        assert (weights >= 0.0).all()
        values, weights = distributions["mobility"][group]
        assert values.tolist() == list(PIP_MOBILITY_TOTALS)
    # Table 2.9 prints whole percentages; its 16 to 19 row adds to 92 as
    # ported, inside the tolerance, and every row's total is kept for the
    # receipt.
    totals = {
        group: sum(cells.values())
        for group, cells in published_pip_points_shares(
            vendored_rows(PIP_POINTS_RESOURCE)
        )["daily_living"].items()
    }
    assert totals["16_to_19"] == pytest.approx(92.0)
    assert all(abs(total - 100.0) <= 10.0 for total in totals.values())


def test_the_reader_refuses_broken_tables() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="repeat"):
        pip_points_distributions_from_rows([*rows, rows[0]])
    with pytest.raises(ValueError, match="miss"):
        pip_points_distributions_from_rows(rows[1:])
    stray = dict(
        rows[0],
        dimensions={"age_band": "70_to_79", "pip_daily_living_points_band": "0_to_3"},
    )
    with pytest.raises(ValueError, match="unexpected age group"):
        pip_points_distributions_from_rows([*rows, stray])
    with pytest.raises(ValueError, match="not about 100"):
        pip_points_distributions_from_rows(_rows(dl=lambda group, band: 5.0))
    with pytest.raises(ValueError, match="is -1.0"):
        pip_points_distributions_from_rows(
            _rows(dl=lambda group, band: -1.0 if band == "0_to_3" else 100.0 / 11)
        )


def test_draws_keep_every_award_and_leave_non_recipients_unassessed() -> None:
    distributions = pip_points_distributions_from_rows(_rows())
    count = 600
    rng = np.random.default_rng(0)
    category = np.asarray(["NONE", "STANDARD", "ENHANCED"] * (count // 3), dtype=object)
    recipient = np.ones(count, dtype=bool)
    recipient[::7] = False
    category[~recipient] = "NONE"
    group = np.asarray(rng.choice(_GROUPS, count), dtype=object)
    for component in ("daily_living", "mobility"):
        points, fallbacks = draw_pip_points(
            category=category,
            recipient=recipient,
            age_group=group,
            uniforms=rng.uniform(size=count),
            distributions=distributions[component],
        )
        assert fallbacks == 0
        assert (points[~recipient] == PIP_NOT_ASSESSED).all()
        none = recipient & (category == "NONE")
        standard = recipient & (category == "STANDARD")
        enhanced = recipient & (category == "ENHANCED")
        assert ((points[none] >= 0) & (points[none] < 8)).all()
        assert ((points[standard] >= 8) & (points[standard] < 12)).all()
        assert (points[enhanced] >= 12).all()
    # An unknown category is refused.
    with pytest.raises(ValueError, match="unknown"):
        draw_pip_points(
            category=np.asarray(["LOWER"], dtype=object),
            recipient=np.asarray([True]),
            age_group=np.asarray(["30_to_39"], dtype=object),
            uniforms=np.asarray([0.5]),
            distributions=distributions["daily_living"],
        )


def test_an_age_group_with_no_share_in_range_takes_the_all_ages_row() -> None:
    # The 30 to 39 group prints no enhanced daily living share at all.
    distributions = pip_points_distributions_from_rows(
        _rows(
            dl=lambda group, band: (
                (
                    100.0 / 4
                    if band in ("0_to_3", "4_to_7", "8_to_9", "10_to_11")
                    else 0.0
                )
                if group == "30_to_39"
                else 100.0 / 12
            )
        )
    )
    points, fallbacks = draw_pip_points(
        category=np.asarray(["ENHANCED", "STANDARD"], dtype=object),
        recipient=np.asarray([True, True]),
        age_group=np.asarray(["30_to_39", "30_to_39"], dtype=object),
        uniforms=np.asarray([0.5, 0.5]),
        distributions=distributions["daily_living"],
    )
    assert points[0] >= 12 and 8 <= points[1] < 12
    assert fallbacks == 1


def _frame(categories, ages, regions, sources):
    count = len(ages)
    person = pd.DataFrame(
        {
            "person_id": [100 + i for i in range(count)],
            "person_benunit_id": list(range(1, count + 1)),
            "person_household_id": list(range(1, count + 1)),
            "age": ages,
            "pip_dl_category": [dl for dl, _ in categories],
            "pip_m_category": [m for _, m in categories],
        }
    )
    benunit = pd.DataFrame({"benunit_id": list(range(1, count + 1))})
    household = pd.DataFrame(
        {
            "household_id": list(range(1, count + 1)),
            "region": regions,
            "household_source_id": sources,
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.full(count, 10.0),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def _implied(points):
    return np.where(points >= 12, "ENHANCED", np.where(points >= 8, "STANDARD", "NONE"))


def test_store_pip_points_writes_both_points_and_keeps_every_award() -> None:
    awards = [
        ("ENHANCED", "STANDARD"),
        ("NONE", "ENHANCED"),
        ("STANDARD", "NONE"),
        ("NONE", "NONE"),
    ] * 5
    frame = _frame(
        awards,
        ages=[18, 25, 45, 70] * 5,
        regions=["WALES", "LONDON", "SCOTLAND", "NORTH_EAST"] * 5,
        sources=list(range(1000, 1020)),
    )
    rows = _rows()
    result = store_pip_points(
        frame,
        distributions=pip_points_distributions_from_rows(rows),
        published_shares=published_pip_points_shares(rows),
    )
    person = result.frame.table("person")
    dl = person["pip_dl_points"].to_numpy()
    m = person["pip_m_points"].to_numpy()
    assert (_implied(dl) == person["pip_dl_category"].to_numpy()).all()
    assert (_implied(m) == person["pip_m_category"].to_numpy()).all()
    # Neither component: not assessed on both.
    neither = (person["pip_dl_category"] == "NONE") & (
        person["pip_m_category"] == "NONE"
    )
    assert (dl[neither.to_numpy()] == PIP_NOT_ASSESSED).all()
    assert (m[neither.to_numpy()] == PIP_NOT_ASSESSED).all()
    # The weights and the categories pass through.
    assert (
        result.frame.weights_for("household").total
        == frame.weights_for("household").total
    )
    assert result.frame.mass_log[-1].reason == pip_points.PIP_POINTS_MASS_CHANGE_REASON
    pd.testing.assert_series_equal(
        person["pip_dl_category"], frame.table("person")["pip_dl_category"]
    )
    receipt = result.evidence()
    assert receipt["stage"] == PIP_POINTS_STAGE_NAME
    for component in ("daily_living", "mobility"):
        entry = receipt["components"][component]
        assert entry["category_mismatches"] == 0
        # England and Wales only, by age group, with support bands; a cell on
        # fewer than three source households is withheld.
        for group in entry["england_and_wales"]:
            for cell in group["cells"]:
                if cell["support"] in ("<3",):
                    assert cell["share"] in ("withheld", "<3")
    # Every country's recipients, so a threshold reform can be netted out by
    # country: Wales, London and Scotland hold five recipients each, weight 10.
    assert receipt["recipients_by_country"] == {
        "england": {"records": "3-9", "support": "3-9", "weighted": 50},
        "wales": {"records": "3-9", "support": "3-9", "weighted": 50},
        "scotland": {"records": "3-9", "support": "3-9", "weighted": 50},
        "northern_ireland": {"records": 0, "support": 0, "weighted": 0},
    }


def test_a_country_on_fewer_than_three_households_shows_no_weighted_count() -> None:
    awards = [("STANDARD", "NONE")] * 5 + [("NONE", "NONE")]
    frame = _frame(
        awards,
        ages=[40] * 6,
        regions=["SCOTLAND", "SCOTLAND", "LONDON", "LONDON", "LONDON", "WALES"],
        sources=[1, 2, 3, 4, 5, 6],
    )
    rows = _rows()
    receipt = store_pip_points(
        frame, distributions=pip_points_distributions_from_rows(rows)
    ).evidence()
    by_country = receipt["recipients_by_country"]
    assert by_country["scotland"] == {
        "records": "<3",
        "support": "<3",
        "weighted": "<3",
    }
    assert by_country["england"] == {"records": "3-9", "support": "3-9", "weighted": 30}
    # Wales's one household holds no recipient.
    assert by_country["wales"] == {"records": 0, "support": 0, "weighted": 0}


def test_store_pip_points_is_the_points_only_writer_and_needs_its_inputs() -> None:
    frame = _frame(
        [("STANDARD", "NONE")] * 3,
        ages=[30] * 3,
        regions=["WALES"] * 3,
        sources=[1, 2, 3],
    )
    distributions = pip_points_distributions_from_rows(_rows())
    stored = store_pip_points(frame, distributions=distributions).frame
    with pytest.raises(ValueError, match="only writer"):
        store_pip_points(stored, distributions=distributions)
    person = frame.table("person").drop(columns=["pip_m_category"])
    bare = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        household_weights=frame.weights_for("household").values,
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )
    with pytest.raises(KeyError, match="pip_m_category"):
        store_pip_points(bare, distributions=distributions)


def test_the_draw_is_keyed_on_person_id() -> None:
    awards = [("ENHANCED", "STANDARD")] * 4
    distributions = pip_points_distributions_from_rows(_rows())
    first = store_pip_points(
        _frame(awards, ages=[30] * 4, regions=["WALES"] * 4, sources=[1, 2, 3, 4]),
        distributions=distributions,
    ).frame.table("person")
    again = store_pip_points(
        _frame(awards, ages=[30] * 4, regions=["WALES"] * 4, sources=[1, 2, 3, 4]),
        distributions=distributions,
    ).frame.table("person")
    pd.testing.assert_series_equal(first["pip_dl_points"], again["pip_dl_points"])
    pd.testing.assert_series_equal(first["pip_m_points"], again["pip_m_points"])


def test_the_committed_stage_binds_its_operations_closed_world() -> None:
    stage = load_country_spec("uk").sources.stage_map()[PIP_POINTS_STAGE_NAME]
    transform = UKPIPPointsStageTransform(stage=stage)
    assert transform.output_columns() == ("pip_dl_points", "pip_m_points")
    operations = list(stage.operations)
    drifted = replace(
        operations[1],
        parameters={**dict(operations[1].parameters), "open_band_points": 45},
    )
    with pytest.raises(ValueError, match="parameters drifted"):
        UKPIPPointsStageTransform(
            stage=replace(stage, operations=(operations[0], drifted, operations[2]))
        )
