"""US hours-worked stage tests (populace #242/#248)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.source_manifest import SourceStageSpec
from populace.build.source_runtime import SourceRuntimeError
from populace.build.us_runtime import (
    US_HOURS_WORKED_OUTPUT_COLUMNS,
    US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS,
    US_HOURS_WORKED_STAGE_NAME,
    US_SOURCE_MANIFEST,
    derive_us_hours_worked_from_manifest,
    us_hours_worked_signal_gate,
    us_hours_worked_stage_spec,
    us_hours_worked_summary,
    with_us_hours_worked_inputs,
)
from populace.build.us_runtime.source_runtime import us_source_operation_handlers
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024


def _person_table(rows: list[dict]) -> pd.DataFrame:
    """A raw-ASEC person table: baseline is a non-working adult."""

    baseline = {"HRSWK": 0, "A_HRS1": 0, "WKSWORK": 0}
    records = []
    for index, row in enumerate(rows):
        record = dict(baseline)
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("person_household_id", index + 1)
        records.append(record)
    return pd.DataFrame(records)


def _us_frame(
    person_rows: list[dict],
    *,
    household_weights: list[float] | None = None,
) -> Frame:
    person = _person_table(person_rows)
    n = len(person)
    household_ids = person["person_household_id"].to_numpy()
    unique_households = np.unique(household_ids)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_spm_unit_id"] = person["person_household_id"] + 2_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(n, dtype="int64") + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": unique_households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": unique_households + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": unique_households + 2_000}),
        "family": pd.DataFrame({"family_id": unique_households + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(n, dtype="int64") + 4_000}
        ),
    }
    weights = household_weights or [1.0] * len(unique_households)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.asarray(weights, dtype=np.float64),
                kind=WeightKind.DESIGN,
            )
        },
    )


def _worker(hrswk: float, a_hrs1: float, wkswork: float, **extra) -> dict:
    return {"HRSWK": hrswk, "A_HRS1": a_hrs1, "WKSWORK": wkswork, **extra}


#: Half workers at typical full-time hours, half non-workers: lands inside
#: both plausibility bands.
_PLAUSIBLE_ROWS = [
    _worker(40, 40, 52),
    _worker(38, 35, 50),
    _worker(20, 22, 26),
    _worker(45, 40, 52),
    _worker(0, 0, 0),
    _worker(0, 0, 0),
    _worker(0, 0, 0),
    _worker(0, 5, 0),
]


class TestManifestDeclaration:
    def test_stage_is_declared_with_the_two_leaf_output_columns(self) -> None:
        spec = us_hours_worked_stage_spec()
        assert spec.stage == US_HOURS_WORKED_STAGE_NAME
        assert tuple(spec.outputs) == US_HOURS_WORKED_OUTPUT_COLUMNS
        assert set(spec.nonnegative_outputs) == set(US_HOURS_WORKED_OUTPUT_COLUMNS)

    def test_org_wages_no_longer_claims_weekly_hours(self) -> None:
        org_wages = US_SOURCE_MANIFEST.stage_map()["org_wages"]
        assert "weekly_hours_worked_before_lsr" not in org_wages.outputs

    def test_handler_is_registered(self) -> None:
        handlers = us_source_operation_handlers()
        assert handlers["derive_hours_worked"] is derive_us_hours_worked_from_manifest


class TestDerivation:
    def _derive(self, table: pd.DataFrame) -> pd.DataFrame:
        spec = us_hours_worked_stage_spec()
        operation = next(
            op for op in spec.operations if op.kind == "derive_hours_worked"
        )
        return derive_us_hours_worked_from_manifest(table, operation, None)

    def test_maps_the_asec_columns_directly(self) -> None:
        table = _person_table([_worker(38, 40, 50), _worker(0, 0, 0)])
        result = self._derive(table)
        assert result["weekly_hours_worked_before_lsr"].tolist() == [38.0, 0.0]
        assert result["hours_worked_last_week"].tolist() == [40.0, 0.0]
        assert "weeks_worked" not in result
        assert result["WKSWORK"].tolist() == [50, 0]

    def test_negative_hour_sentinels_floor_at_zero(self) -> None:
        table = _person_table([_worker(-1, -4, 99)])
        result = self._derive(table)
        assert result["weekly_hours_worked_before_lsr"].tolist() == [0.0]
        assert result["hours_worked_last_week"].tolist() == [0.0]
        assert "weeks_worked" not in result
        assert result["WKSWORK"].tolist() == [99]

    def test_nan_raw_values_become_zero(self) -> None:
        table = _person_table([_worker(np.nan, np.nan, np.nan)])
        result = self._derive(table)
        for column in US_HOURS_WORKED_OUTPUT_COLUMNS:
            assert result[column].tolist() == [0.0]

    def test_missing_raw_column_is_named(self) -> None:
        table = _person_table([_worker(40, 40, 52)]).drop(columns=["HRSWK"])
        with pytest.raises(SourceRuntimeError, match="HRSWK"):
            self._derive(table)

    def test_unexpected_parameters_are_refused(self) -> None:
        operation_spec = SourceStageSpec.from_mapping(
            {
                "stage": US_HOURS_WORKED_STAGE_NAME,
                "survey": "test ASEC",
                "source": "https://example.com",
                "grain": "person",
                "operations": [
                    {"kind": "read_table", "table": "person"},
                    {"kind": "derive_hours_worked", "surprise": True},
                ],
                "outputs": list(US_HOURS_WORKED_OUTPUT_COLUMNS),
            }
        )
        operation = operation_spec.operations[1]
        with pytest.raises(SourceRuntimeError, match="surprise"):
            derive_us_hours_worked_from_manifest(
                _person_table([_worker(40, 40, 52)]), operation, None
            )


class TestFrameIntegration:
    def test_with_us_hours_worked_inputs_writes_only_the_two_leaves(self) -> None:
        frame = with_us_hours_worked_inputs(
            _us_frame(_PLAUSIBLE_ROWS), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person")
        assert person["weekly_hours_worked_before_lsr"].tolist()[:4] == [
            40.0,
            38.0,
            20.0,
            45.0,
        ]
        assert person["hours_worked_last_week"].iloc[7] == 5.0
        assert "weeks_worked" not in person
        assert person["WKSWORK"].max() == 52

    def test_frame_with_signal_passes_through_untouched(self) -> None:
        derived = with_us_hours_worked_inputs(
            _us_frame(_PLAUSIBLE_ROWS), seed=0, time_period=TIME_PERIOD
        )
        again = with_us_hours_worked_inputs(derived, seed=1, time_period=TIME_PERIOD)
        assert again is derived

    def test_constant_forty_landmine_is_recomputed_from_raw(self) -> None:
        # The published failure mode: both leaves present, but weekly
        # hours broadcast at the engine's 40-hour default.
        rows = [
            _worker(
                38,
                40,
                50,
                weekly_hours_worked_before_lsr=40.0,
                hours_worked_last_week=40.0,
                weeks_worked=52.0,
            ),
            _worker(
                0,
                0,
                0,
                weekly_hours_worked_before_lsr=40.0,
                hours_worked_last_week=40.0,
                weeks_worked=52.0,
            ),
        ]
        frame = with_us_hours_worked_inputs(
            _us_frame(rows), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person")
        assert person["weekly_hours_worked_before_lsr"].tolist() == [38.0, 0.0]
        assert person["hours_worked_last_week"].tolist() == [40.0, 0.0]
        assert "weeks_worked" not in person

    def test_partial_surface_is_recomputed(self) -> None:
        rows = [
            _worker(38, 40, 50, hours_worked_last_week=40.0, weeks_worked=52.0),
            _worker(0, 0, 0, hours_worked_last_week=0.0, weeks_worked=0.0),
        ]
        frame = with_us_hours_worked_inputs(
            _us_frame(rows), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person")
        assert person["weekly_hours_worked_before_lsr"].tolist() == [38.0, 0.0]
        assert "weeks_worked" not in person

    def test_missing_raw_columns_without_signal_raise(self) -> None:
        frame = _us_frame([{"person_id": 1}])
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["person"] = tables["person"].drop(
            columns=list(US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS)
        )
        stripped = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        )
        with pytest.raises(SourceRuntimeError, match="HRSWK"):
            with_us_hours_worked_inputs(stripped, seed=0, time_period=TIME_PERIOD)


class TestGate:
    def test_plausible_distribution_passes(self) -> None:
        frame = with_us_hours_worked_inputs(
            _us_frame(_PLAUSIBLE_ROWS), seed=0, time_period=TIME_PERIOD
        )
        gate = us_hours_worked_signal_gate(frame)
        assert gate.passed
        summary = us_hours_worked_summary(frame)
        assert 0.35 <= summary["worked_share"] <= 0.62
        assert 30.0 <= summary["mean_weekly_hours_workers"] <= 45.0

    def test_missing_columns_fail(self) -> None:
        gate = us_hours_worked_signal_gate(_us_frame([_worker(40, 40, 52)]))
        assert not gate.passed
        assert "missing" in gate.failures[0]

    def test_constant_forty_fails(self) -> None:
        rows = [
            _worker(
                0,
                0,
                0,
                weekly_hours_worked_before_lsr=40.0,
                hours_worked_last_week=40.0,
                weeks_worked=52.0,
            )
            for _ in range(4)
        ]
        frame = _us_frame(rows)
        gate = us_hours_worked_signal_gate(frame)
        assert not gate.passed
        assert any("constant" in failure for failure in gate.failures)

    def test_everyone_working_fails_the_share_band(self) -> None:
        frame = with_us_hours_worked_inputs(
            _us_frame([_worker(40, 40, 52), _worker(38, 38, 52)]),
            seed=0,
            time_period=TIME_PERIOD,
        )
        gate = us_hours_worked_signal_gate(frame)
        assert not gate.passed
        assert any("worked share" in failure for failure in gate.failures)

    def test_implausible_mean_hours_fail(self) -> None:
        rows = [
            _worker(3, 3, 52),
            _worker(2, 2, 52),
            _worker(0, 0, 0),
            _worker(0, 0, 0),
        ]
        frame = with_us_hours_worked_inputs(
            _us_frame(rows), seed=0, time_period=TIME_PERIOD
        )
        gate = us_hours_worked_signal_gate(frame)
        assert not gate.passed
        assert any("mean weekly hours" in failure for failure in gate.failures)
