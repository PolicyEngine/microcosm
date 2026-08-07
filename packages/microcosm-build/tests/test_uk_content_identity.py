"""Content identity for UK national frames (#612 increment 3).

The identity must move with every content dimension a substitution fence
cares about — payload values, column names, typed weights and their kind,
the mass log, and frame metadata — while two independent reconstructions of
the same content agree.
"""

from __future__ import annotations

import pandas as pd
import pytest

from microcosm.build.uk_runtime import uk_frame_content_identity
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import MassChangeRecord, WeightKind


def _frame(
    *,
    pay: float = 20_000.0,
    weight: float = 10.0,
    time_period: str = "2023",
    weight_kind: WeightKind = WeightKind.DESIGN,
    mass_log: tuple[MassChangeRecord, ...] = (),
    pay_column: str = "pay",
):
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_household_id": [1, 1],
                "person_benunit_id": [1, 1],
                pay_column: [pay, 0.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame(
            {
                "household_id": [1],
                "household_weight": [weight],
            }
        ),
        time_period=time_period,
        weight_kind=weight_kind,
        mass_log=mass_log,
    )


def test_independent_reconstructions_of_the_same_content_agree() -> None:
    first = _frame()
    second = _frame()
    assert first is not second
    assert uk_frame_content_identity(first) == uk_frame_content_identity(second)


@pytest.mark.parametrize(
    "variant",
    [
        dict(pay=20_001.0),
        dict(weight=11.0),
        dict(time_period="2024"),
        dict(weight_kind=WeightKind.IMPORTANCE),
        dict(pay_column="pay_renamed"),
        dict(
            mass_log=(
                MassChangeRecord(
                    entity="household",
                    old_total=10.0,
                    new_total=10.0,
                    declared_factor=None,
                    reason="test",
                ),
            )
        ),
    ],
    ids=["value", "weight", "metadata", "weight-kind", "column-name", "mass-log"],
)
def test_identity_moves_with_every_content_dimension(variant: dict) -> None:
    assert uk_frame_content_identity(_frame()) != uk_frame_content_identity(
        _frame(**variant)
    )


def test_identity_requires_a_frame() -> None:
    with pytest.raises(TypeError, match="microcosm Frame"):
        uk_frame_content_identity(object())  # type: ignore[arg-type]


def test_identity_moves_with_strata() -> None:
    """Strata are part of the content (the v2 digest closes the v1 gap)."""

    import numpy as np

    from microcosm.frame import EntitySchema, Frame, Weights

    def _stratified(labels):
        return Frame(
            {
                "person": pd.DataFrame(
                    {"person_id": [1, 2], "person_household_id": [1, 1]}
                ),
                "household": pd.DataFrame({"household_id": [1]}),
            },
            EntitySchema(group_entities=("household",)),
            {
                "household": Weights(
                    np.array([1.0], dtype=np.float64), WeightKind.DESIGN
                )
            },
            pd.Series(labels, dtype=object),
        )

    assert uk_frame_content_identity(
        _stratified(["s1", "s1"])
    ) != uk_frame_content_identity(_stratified(["s1", "s2"]))
