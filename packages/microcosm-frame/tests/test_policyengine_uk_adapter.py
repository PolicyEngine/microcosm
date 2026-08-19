"""PolicyEngine-UK adapter import/protocol behavior."""

from __future__ import annotations

import pytest

from microcosm.frame import RulesEngine
from microcosm.frame.adapters.policyengine_uk import (
    UK_SCHEMA,
    PolicyEngineUKEngine,
)


def test_policyengine_uk_adapter_satisfies_rules_protocol_without_importing_engine() -> (
    None
):
    adapter = PolicyEngineUKEngine()

    assert isinstance(adapter, RulesEngine)
    assert adapter.country == "uk"
    assert adapter.entity_schema() == UK_SCHEMA


def test_policyengine_uk_adapter_export_side_is_not_implemented() -> None:
    adapter = PolicyEngineUKEngine()

    with pytest.raises(NotImplementedError, match="write_uk_national_frame"):
        adapter.write_dataset(object(), "unused.h5", period=2023)  # type: ignore[arg-type]


def test_policyengine_uk_adapter_builds_a_real_engine_dataset() -> None:
    # Regression: the real UKSingleYearDataset constructor takes fiscal_year,
    # not time_period — a kwarg mismatch the protocol tests above cannot see
    # because they never import the engine. Dataset construction is cheap
    # (no simulation), so this runs wherever the uk extra is installed.
    pytest.importorskip("policyengine_uk")
    import numpy as np
    import pandas as pd

    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    person = pd.DataFrame(
        {
            "person_id": np.asarray([10, 11, 12], dtype="int64"),
            "person_benunit_id": np.asarray([1, 1, 2], dtype="int64"),
            "person_household_id": np.asarray([5, 5, 6], dtype="int64"),
            "age": [34.0, 3.0, 61.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.asarray([1, 2], dtype="int64")})
    household = pd.DataFrame({"household_id": np.asarray([5, 6], dtype="int64")})
    frame = Frame(
        tables={"person": person, "benunit": benunit, "household": household},
        schema=EntitySchema(group_entities=("benunit", "household")),
        weights={
            "household": Weights(
                values=np.array([120.0, 250.0]), kind=WeightKind.DESIGN
            )
        },
    )

    dataset = PolicyEngineUKEngine()._build_dataset(frame, 2023)

    stamp = str(getattr(dataset, "time_period", getattr(dataset, "fiscal_year", "")))
    assert stamp.startswith("2023")
