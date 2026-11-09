"""The scoring route through ``prepare_uk_target_frame`` on a synthetic surface.

The route is the calibration stage's resolve-inject-materialize path without
the solve. Since the region fan-out, cross-grain measures inject one input
name at two entities, which the prepared frame's global-column-uniqueness
rule refuses unless the injected inputs are dropped after materialization;
every score of a candidate export was refused that way and nothing in the
suite caught it (review finding on PR #967). This fixture is that case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import national_calibration
from microcosm.build.uk_runtime.national_calibration import (
    CalibrationFrameAdapter,
    inject_measure_inputs,
    prepare_uk_target_frame,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind

PEOPLE = "test/people_in_scope"
HOUSEHOLDS = "test/households_in_scope"


class _CrossGrainResolver:
    """One input name, ``in_scope``, resolved at the person and the household
    entity: the shape the region fan-out produces."""

    contract_targets = {
        "test.people_in_scope": {
            "target_id": "test.people_in_scope",
            "family": "test",
            "bindings": {
                "policyengine": {
                    "metric_name": PEOPLE,
                    "from_entity": "person",
                    "value_variable": "in_scope",
                }
            },
        },
        "test.households_in_scope": {
            "target_id": "test.households_in_scope",
            "family": "test",
            "bindings": {
                "policyengine": {
                    "metric_name": HOUSEHOLDS,
                    "from_entity": "household",
                    "value_variable": "in_scope",
                }
            },
        },
    }

    def __init__(self) -> None:
        self.computed: list[tuple[str, str]] = []

    def knows(self, entity: str, variable: str) -> bool:
        return variable == "in_scope" and entity in {"person", "household"}

    def compute(self, entity: str, variable: str):
        self.computed.append((entity, variable))
        if entity == "person":
            return np.array([1.0, 0.0, 1.0, 1.0]), "stub_person"
        return np.array([1.0, 0.0]), "stub_household"

    def receipt(self):
        return {"provider": "stub"}


def _frame():
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [11, 12, 21, 22],
                "person_benunit_id": [1, 1, 2, 2],
                "person_household_id": [1, 1, 2, 2],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame({"household_id": [1, 2]}),
        time_period="2025",
        weight_kind=WeightKind.CALIBRATED,
        household_weights=np.array([10.0, 20.0]),
    )


@pytest.fixture(autouse=True)
def _synthetic_contract(monkeypatch):
    """The materializer reads the packaged contract; this surface is synthetic."""
    from microcosm.build.uk_runtime import ledger_targets

    monkeypatch.setattr(
        ledger_targets,
        "_uk_contract_targets",
        lambda: dict(_CrossGrainResolver.contract_targets),
    )


def _registry() -> TargetRegistry:
    return TargetRegistry(
        [
            TargetSpec(
                name="test.people_in_scope",
                entity="person",
                measure=PEOPLE,
                value=50.0,
                period=2025,
                source="synthetic",
                family="test",
                metadata={"contract_target_id": "test.people_in_scope"},
            ),
            TargetSpec(
                name="test.households_in_scope",
                entity="household",
                measure=HOUSEHOLDS,
                value=10.0,
                period=2025,
                source="synthetic",
                family="test",
                metadata={"contract_target_id": "test.households_in_scope"},
            ),
        ],
        country="uk",
    )


def test_cross_grain_measures_materialize_and_the_injected_inputs_are_dropped():
    resolver = _CrossGrainResolver()

    prepared, receipt = prepare_uk_target_frame(
        _frame(), _registry(), period=2025, measure_resolver=resolver
    )

    assert sorted(resolver.computed) == [
        ("household", "in_scope"),
        ("person", "in_scope"),
    ]
    person = prepared.table("person")
    household = prepared.table("household")
    assert person[PEOPLE].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert household[HOUSEHOLDS].tolist() == [1.0, 0.0]
    # The injected inputs are gone from both tables, as the stage drops them.
    assert "in_scope" not in person.columns
    assert "in_scope" not in household.columns
    assert receipt is not None
    assert set(receipt["attached"]) == {"person.in_scope", "household.in_scope"}


def test_the_prepared_frame_refuses_the_injected_inputs_if_they_stayed():
    """The defect the route fixes: one name on two tables is refused."""
    adapter = CalibrationFrameAdapter(_frame())
    inject_measure_inputs(
        adapter,
        {
            ("person", "in_scope"): np.array([1.0, 0.0, 1.0, 1.0]),
            ("household", "in_scope"): np.array([1.0, 0.0]),
        },
    )
    with pytest.raises(ValueError, match="in_scope"):
        adapter.prepared_frame()


def test_the_band_edge_register_reaches_materialization(monkeypatch):
    """A pruned scoring surface never redraws its own band edges (#803)."""
    seen: list[object] = []
    original = national_calibration.materialize_uk_ledger_targets

    def spy(adapter, registry, *, period, band_edge_registry=None):
        seen.append(band_edge_registry)
        return original(
            adapter, registry, period=period, band_edge_registry=band_edge_registry
        )

    monkeypatch.setattr(national_calibration, "materialize_uk_ledger_targets", spy)
    full = _registry()
    surface = TargetRegistry([full.specs[0]], country="uk")

    prepare_uk_target_frame(
        _frame(),
        surface,
        period=2025,
        measure_resolver=_CrossGrainResolver(),
        band_edge_registry=full,
    )

    assert seen and all(edges is full for edges in seen)
