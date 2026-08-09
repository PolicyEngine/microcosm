"""The shared engine-tables materializer: typed weights are authoritative.

Both engine adapters delegate here, and country build runtimes without an
adapter class (the UK writer) call it directly, so this is the one place the
weight-column contract is enforced: any existing ``{entity}_weight`` column
is overwritten in place (preserving its position), an absent one is appended,
and a stale column can never override calibrated weights on export.
"""

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights, engine_tables


def _uk_frame(*, stale_weight_column: bool) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([10, 11, 12], dtype="int64"),
            "person_benunit_id": np.asarray([1, 1, 2], dtype="int64"),
            "person_household_id": np.asarray([5, 5, 6], dtype="int64"),
            "age": [34.0, 3.0, 61.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.asarray([1, 2], dtype="int64")})
    household_columns: dict[str, object] = {
        "household_id": np.asarray([5, 6], dtype="int64"),
        "region": ["North East", "Wales"],
    }
    if stale_weight_column:
        # Deliberately wrong values, placed *between* other columns so the
        # in-place overwrite (position preserved) is observable.
        household_columns = {
            "household_id": np.asarray([5, 6], dtype="int64"),
            "household_weight": [1.0, 1.0],
            "region": ["North East", "Wales"],
        }
    household = pd.DataFrame(household_columns)
    return Frame(
        tables={"person": person, "benunit": benunit, "household": household},
        schema=EntitySchema(group_entities=("benunit", "household")),
        weights={
            "household": Weights(
                values=np.array([120.0, 250.0]), kind=WeightKind.DESIGN
            )
        },
    )


def test_stale_weight_column_is_overwritten_in_place() -> None:
    frame = _uk_frame(stale_weight_column=True)

    tables = engine_tables(frame)

    household = tables["household"]
    np.testing.assert_array_equal(
        household["household_weight"].to_numpy(), np.array([120.0, 250.0])
    )
    # Overwriting assigns in place: the column keeps its original position.
    assert list(household.columns) == ["household_id", "household_weight", "region"]
    # The frame's own table is untouched — the materializer copies.
    np.testing.assert_array_equal(
        frame.table("household")["household_weight"].to_numpy(), np.array([1.0, 1.0])
    )


def test_absent_weight_column_is_appended_and_order_follows_entities() -> None:
    frame = _uk_frame(stale_weight_column=False)

    tables = engine_tables(frame)

    assert list(tables) == ["person", "benunit", "household"]
    assert list(tables["household"].columns) == [
        "household_id",
        "region",
        "household_weight",
    ]
    # Only entities carrying explicit typed weights are materialized.
    assert "person_weight" not in tables["person"].columns
    assert "benunit_weight" not in tables["benunit"].columns


def test_weighted_entities_pin_rejects_unweighted_entity() -> None:
    frame = _uk_frame(stale_weight_column=False)

    pinned = engine_tables(frame, weighted_entities=("household",))
    np.testing.assert_array_equal(
        pinned["household"]["household_weight"].to_numpy(), np.array([120.0, 250.0])
    )

    # Inherited weights are never materialized implicitly: pinning an entity
    # without its own explicit vector raises, exactly as weights_for does.
    with pytest.raises(ValueError):
        engine_tables(frame, weighted_entities=("person",))


def test_simple_schema_bundle_matches_typed_weights(make_bundle) -> None:
    bundle = make_bundle(kind=WeightKind.CALIBRATED)

    tables = engine_tables(bundle)

    np.testing.assert_array_equal(
        tables["household"]["household_weight"].to_numpy(),
        bundle.weights_for("household").values,
    )
