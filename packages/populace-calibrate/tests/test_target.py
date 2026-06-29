"""Target / TargetSet construction and validation."""

from __future__ import annotations

import numpy as np
import pytest

from populace.calibrate import Target, TargetSet


def test_target_constructs_sum_constraint() -> None:
    target = Target(
        name="income",
        entity="household",
        value=1.0,
        measure="income",
    )
    assert target.entity == "household"
    assert target.measure == "income"


def test_aggregation_argument_is_not_supported() -> None:
    with pytest.raises(TypeError, match="aggregation"):
        Target(
            name="bad",
            entity="household",
            aggregation="median",
            value=1.0,
            measure="income",
        )


@pytest.mark.parametrize("measure", [None, ""])
def test_measure_is_required(measure) -> None:
    with pytest.raises(ValueError, match="measure is required"):
        Target(
            name="population",
            entity="household",
            measure=measure,
            value=1.0,
        )


def test_measure_may_be_a_column_or_a_callable() -> None:
    column = Target(
        name="by_column",
        entity="household",
        value=1.0,
        measure="income",
    )
    assert column.measure == "income"

    def half_income(frame):
        return frame.table("household")["income"].to_numpy() * 0.5

    callable_target = Target(
        name="by_callable",
        entity="household",
        value=1.0,
        measure=half_income,
    )
    assert callable(callable_target.measure)


def test_metadata_is_carried_on_target() -> None:
    target = Target(
        name="jct/salt",
        entity="household",
        value=1.0,
        measure="income_tax_delta",
        metadata={"kind": "neutralize_variable"},
    )
    assert target.metadata == {"kind": "neutralize_variable"}

    with pytest.raises(ValueError, match="metadata"):
        Target(
            name="bad",
            entity="household",
            measure="household_count",
            value=1.0,
            metadata={"kind": ""},
        )


def test_targetset_is_an_ordered_container() -> None:
    a = Target(name="a", entity="household", measure="household_count", value=1.0)
    b = Target(name="b", entity="household", measure="household_count", value=2.0)
    ts = TargetSet((a, b))
    assert len(ts) == 2
    assert list(ts)[0].name == "a"


def test_period_defaults_and_is_carried(landmine_frame) -> None:
    t = Target(
        name="cg",
        entity="household",
        value=1.0,
        measure="capital_gains",
        period=2030,
    )
    assert t.period == 2030
    _ = np  # keep import used if fixtures change
