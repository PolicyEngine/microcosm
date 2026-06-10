"""Target / TargetSet construction and validation."""

from __future__ import annotations

import numpy as np
import pytest

from populace.calibrate import Target, TargetSet


def test_sum_count_mean_aggregations_construct() -> None:
    for agg in ("sum", "count", "mean"):
        t = Target(
            name=f"t_{agg}",
            entity="household",
            aggregation=agg,
            value=1.0,
            measure="income" if agg != "count" else None,
        )
        assert t.aggregation == agg
        assert t.entity == "household"


def test_unknown_aggregation_is_refused() -> None:
    with pytest.raises(ValueError, match="aggregation"):
        Target(
            name="bad",
            entity="household",
            aggregation="median",
            value=1.0,
            measure="income",
        )


def test_measure_may_be_a_column_or_a_callable() -> None:
    column = Target(
        name="by_column",
        entity="household",
        aggregation="sum",
        value=1.0,
        measure="income",
    )
    assert column.measure == "income"

    def half_income(frame):
        return frame.table("household")["income"].to_numpy() * 0.5

    callable_target = Target(
        name="by_callable",
        entity="household",
        aggregation="sum",
        value=1.0,
        measure=half_income,
    )
    assert callable(callable_target.measure)


def test_targetset_is_an_ordered_container() -> None:
    a = Target(name="a", entity="household", aggregation="count", value=1.0)
    b = Target(name="b", entity="household", aggregation="count", value=2.0)
    ts = TargetSet((a, b))
    assert len(ts) == 2
    assert list(ts)[0].name == "a"


def test_period_defaults_and_is_carried(landmine_frame) -> None:
    t = Target(
        name="cg",
        entity="household",
        aggregation="sum",
        value=1.0,
        measure="capital_gains",
        period=2030,
    )
    assert t.period == 2030
    _ = np  # keep import used if fixtures change
