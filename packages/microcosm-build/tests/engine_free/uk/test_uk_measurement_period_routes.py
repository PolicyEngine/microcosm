"""Both calibration routes measure a stored value at 2025 (microcosm#1324).

The national graph route and the dense line's full-measure route resolve the
packaged contract's OBR statutory maternity pay row, whose value variable the
frame stores at 2024, through the engine at the calibration period. The
stored column itself is never overwritten.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.target_materialization import ENGINE_PERIOD_COLUMN_PREFIX
from microcosm.build.uk_runtime import measure_simulation
from microcosm.build.uk_runtime.graph_national import materialize_uk_national_rows
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind

STORED = np.array([100.0, 0.0, 200.0])
GROWTH = 1.034
MEASURE = "obr/statutory_maternity_pay"


def _frame():
    ids = np.arange(3, dtype=np.int64)
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "age": [30, 35, 40],
                "statutory_maternity_pay": STORED.copy(),
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": [2.0, 3.0, 5.0],
                "region": "LONDON",
                "council_tax": 0.0,
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )


def _registry():
    return TargetRegistry(
        [
            TargetSpec(
                name="obr.statutory_maternity_pay",
                entity="person",
                measure=MEASURE,
                value=1000.0,
                period=2025,
                source="synthetic measurement-period fixture",
                family="obr",
                metadata={
                    "contract_target_id": "obr.statutory_maternity_pay",
                    "ledger_fact_period": "2025",
                },
            )
        ],
        country="uk",
    )


class _Simulation:
    def __init__(self):
        self.calls = []
        self.tax_benefit_system = SimpleNamespace(
            variables={
                "statutory_maternity_pay": SimpleNamespace(
                    entity=SimpleNamespace(key="person"), value_type=float
                )
            }
        )

    def calculate(self, variable, year, map_to=None):
        self.calls.append((variable, year))
        return STORED * (GROWTH if year == 2025 else 1.0)


@pytest.mark.parametrize("route", ["national", "dense"])
def test_both_routes_measure_the_stored_value_at_the_calibration_period(
    monkeypatch, tmp_path, route
):
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    original = _frame()
    simulation = _Simulation()
    monkeypatch.setattr(
        measure_simulation,
        "_policyengine_uk_module",
        lambda: SimpleNamespace(__version__="test"),
    )

    def resolver_factory(**kwargs):
        kwargs.setdefault("simulation_source", None)
        return UKMeasureResolver(
            **kwargs, microsimulation_factory=lambda **_: simulation
        )

    registry = _registry()
    if route == "national":
        prepared, restore, _rows, _evidence = materialize_uk_national_rows(
            original,
            registry,
            period=2025,
            band_edge_registry=registry,
            resolver_factory=resolver_factory,
            scratch_dir=tmp_path / "engine",
        )
    else:
        from microcosm.build.uk_runtime import full_measure

        monkeypatch.setattr(
            full_measure,
            "compute_household_metrics",
            lambda _sim, _area, *, period, input_period, household_ids: pd.DataFrame(
                {"households": np.ones(len(household_ids))}, index=household_ids
            ),
        )
        prepared, restore, _national, _metrics, _receipt = (
            full_measure.resolve_uk_full_measures(
                original,
                registry,
                period=2025,
                scratch_dir=tmp_path / "engine",
                resolver_factory=resolver_factory,
            )
        )

    person = prepared.table("person")
    np.testing.assert_allclose(person[MEASURE], STORED * GROWTH)
    assert ("statutory_maternity_pay", 2025) in simulation.calls
    # The stored column is the frame's own, and no engine column is left.
    np.testing.assert_allclose(person["statutory_maternity_pay"], STORED)
    assert not any(
        str(column).startswith(ENGINE_PERIOD_COLUMN_PREFIX) for column in person
    )
    restored = restore(prepared)
    pd.testing.assert_frame_equal(restored.table("person"), original.table("person"))
