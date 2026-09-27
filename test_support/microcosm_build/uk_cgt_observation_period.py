"""Dated CGT measurements retain base-year rows through both calibration paths."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import measure_simulation
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_calibration import UKNationalCalibrationStage
from microcosm.build.uk_runtime.national_doctrine import UKNationalSolveDoctrine
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind


def _frame():
    ids = np.arange(3, dtype=np.int64)
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "age": [40, 50, 60],
                "capital_gains": [3000.0, 5000.0, 20000.0],
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
                name=name,
                entity="person",
                measure=measure,
                value=value,
                period=2025,
                source="synthetic observation-year integration fixture",
                family="hmrc_cgt",
                metadata={"contract_target_id": name, "ledger_fact_period": "2024"},
            )
            for name, measure, value in [
                ("hmrc.cgt.taxpayers_total", "hmrc/cgt_taxpayers", 16.0),
                ("hmrc.cgt.gains_total", "hmrc/capital_gains_total", 230000.0),
                ("hmrc.cgt.liability_total", "hmrc/cgt_liability", 32760.0),
            ]
        ],
        country="uk",
    )


class _Simulation:
    def __init__(self):
        self.calls = []
        self.tax_benefit_system = SimpleNamespace(
            variables={
                name: SimpleNamespace(
                    entity=SimpleNamespace(key="person"), value_type=float
                )
                for name in ("capital_gains", "capital_gains_tax")
            }
        )

    def calculate(self, variable, year):
        self.calls.append((variable, year))
        gains = np.array([3000.0, 5000.0, 20000.0]) * (1 if year == 2024 else 1.1)
        return (
            gains if variable == "capital_gains" else np.maximum(gains - 3000, 0) * 0.18
        )


def _assert_export(original, fitted, path):
    # Both source bytes and output period survive; fitted weights do not revert.
    assert fitted.metadata["time_period"] == "2024"
    for entity in original.entities:
        columns = [c for c in original.table(entity) if c != "household_weight"]
        pd.testing.assert_frame_equal(
            original.table(entity)[columns], fitted.table(entity)[columns]
        )
    assert not np.array_equal(
        original.weights_for("household").values, fitted.weights_for("household").values
    )
    write_uk_national_frame(fitted, path)
    reread, _ = load_uk_national_frame(path)
    assert reread.metadata["time_period"] == "2024"
    np.testing.assert_array_equal(
        reread.table("person").capital_gains, original.table("person").capital_gains
    )
    np.testing.assert_array_equal(
        reread.weights_for("household").values, fitted.weights_for("household").values
    )
    assert not any(
        str(c).startswith("cgt_2024_") or "/" in str(c) for c in reread.table("person")
    )


__all__ = [name for name in globals() if not name.startswith("__")]
