"""Contract tests for the UK bus-spending imputation plan and calibration targets.

Mirrors ``test_us_plan.py`` (plan assembly + donor citations) and the calibrate
registry tests (target values + provenance), for the two DfT-anchored bus
variables ``bus_fare_spending`` and ``bus_subsidy_spending``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.uk import (
    UK_BUS_DONORS,
    UK_BUS_NONNEGATIVE_SOURCE_OUTPUTS,
    UK_BUS_SOURCE_MANIFEST,
    UK_BUS_STAGE_NAMES,
    UK_BUS_TARGET_REGISTRY,
    calibrate_bus_spending_levels,
    uk_bus_plan,
    uk_bus_targets,
)

EXPECTED_STAGES = {"bus_fare_spending", "bus_subsidy_spending"}


def _noop_implementations() -> dict:
    return {name: (lambda frame: frame) for name in UK_BUS_STAGE_NAMES}


class TestUkBusManifest:
    def test_manifest_is_uk_with_both_bus_stages(self) -> None:
        assert UK_BUS_SOURCE_MANIFEST.country == "uk"
        assert UK_BUS_SOURCE_MANIFEST.version >= 1
        assert set(UK_BUS_STAGE_NAMES) == EXPECTED_STAGES

    def test_every_stage_outputs_its_named_variable_nonnegative(self) -> None:
        for stage in UK_BUS_SOURCE_MANIFEST.stages:
            assert stage.outputs == (stage.stage,)
            assert stage.stage in UK_BUS_NONNEGATIVE_SOURCE_OUTPUTS

    def test_every_stage_imputes_then_clips(self) -> None:
        # The realism clip is what keeps the imputation from concentrating
        # spending in too few households at implausibly high amounts.
        for stage in UK_BUS_SOURCE_MANIFEST.stages:
            kinds = [op.kind for op in stage.operations]
            assert "fit_weighted_qrf" in kinds
            assert "support_clip" in kinds


class TestUkBusPlan:
    def test_plan_assembles_with_donor_citations(self) -> None:
        plan = uk_bus_plan(_noop_implementations())
        assert tuple(stage.name for stage in plan.stages) == UK_BUS_STAGE_NAMES
        donor_stages = dict(plan.donors())
        assert set(donor_stages) == set(UK_BUS_DONORS)
        for spec in donor_stages.values():
            assert spec.source.startswith("https://")

    def test_missing_stage_refuses_to_assemble(self) -> None:
        implementations = _noop_implementations()
        del implementations["bus_fare_spending"]
        with pytest.raises(ValueError, match="missing"):
            uk_bus_plan(implementations)

    def test_unknown_stage_refuses_to_assemble(self) -> None:
        implementations = _noop_implementations()
        implementations["not_a_stage"] = lambda frame: frame
        with pytest.raises(ValueError, match="Unknown stage"):
            uk_bus_plan(implementations)


class TestUkBusTargets:
    def test_targets_cover_both_bus_variables(self) -> None:
        measures = {spec.measure for spec in UK_BUS_TARGET_REGISTRY.specs}
        assert measures == EXPECTED_STAGES

    def test_fare_and_subsidy_anchored_near_dft_uk_totals(self) -> None:
        by_measure = {spec.measure: spec for spec in UK_BUS_TARGET_REGISTRY.specs}
        # DfT England totals uplifted to UK (~1.18x): fare ~GBP 4.0bn,
        # subsidy ~GBP 3.5bn.
        assert 3.9e9 < by_measure["bus_fare_spending"].value < 4.1e9
        assert 3.4e9 < by_measure["bus_subsidy_spending"].value < 3.7e9

    def test_every_target_is_a_sourced_household_sum(self) -> None:
        for spec in UK_BUS_TARGET_REGISTRY.specs:
            assert spec.entity == "household"
            assert spec.aggregation == "sum"
            assert spec.family == "dft"
            assert spec.source  # provenance is required


class TestUkBusCalibration:
    @staticmethod
    def _household(fare: list[float], subsidy: list[float], weight: list[float]):
        return pd.DataFrame(
            {
                "household_weight": weight,
                "bus_fare_spending": fare,
                "bus_subsidy_spending": subsidy,
            }
        )

    def test_scales_each_variable_to_its_target_total(self) -> None:
        # Deliberately wrong levels (fare too high, subsidy too low) — the same
        # failure direction as the published Populace UK population.
        household = self._household(
            fare=[0.0, 5_000.0, 5_000.0, 0.0],
            subsidy=[100.0, 0.0, 100.0, 0.0],
            weight=[1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
        )
        targets = {"bus_fare_spending": 4.0e9, "bus_subsidy_spending": 3.5e9}
        calibrated, scales = calibrate_bus_spending_levels(household, targets=targets)
        w = calibrated["household_weight"].to_numpy(float)
        for column, target in targets.items():
            total = float(np.sum(calibrated[column].to_numpy(float) * w))
            assert total == pytest.approx(target, rel=1e-9)
        assert scales["bus_fare_spending"] < 1  # fare scaled down
        assert scales["bus_subsidy_spending"] > 1  # subsidy scaled up

    def test_value_scaling_preserves_the_spender_set(self) -> None:
        household = self._household(
            fare=[0.0, 5_000.0, 5_000.0, 0.0],
            subsidy=[0.0, 0.0, 100.0, 0.0],
            weight=[1e6, 1e6, 1e6, 1e6],
        )
        before = household["bus_fare_spending"].to_numpy(float) > 0
        calibrated, _ = calibrate_bus_spending_levels(
            household, targets={"bus_fare_spending": 4.0e9}
        )
        after = calibrated["bus_fare_spending"].to_numpy(float) > 0
        # Scaling changes the level, never who spends.
        assert np.array_equal(before, after)

    def test_default_targets_come_from_the_dft_registry(self) -> None:
        assert uk_bus_targets() == {
            spec.measure: float(spec.value) for spec in UK_BUS_TARGET_REGISTRY.specs
        }

    def test_missing_column_is_a_clear_error(self) -> None:
        household = pd.DataFrame({"household_weight": [1.0, 2.0]})
        with pytest.raises(KeyError, match="bus_fare_spending"):
            calibrate_bus_spending_levels(
                household, targets={"bus_fare_spending": 4.0e9}
            )

    def test_zero_aggregate_refuses_to_scale(self) -> None:
        household = self._household(
            fare=[0.0, 0.0], subsidy=[0.0, 0.0], weight=[1e6, 1e6]
        )
        with pytest.raises(ValueError, match="weighted aggregate"):
            calibrate_bus_spending_levels(
                household, targets={"bus_fare_spending": 4.0e9}
            )
