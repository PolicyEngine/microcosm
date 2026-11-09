"""US SNAP state take-up stage tests (microcosm #372)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import (
    US_SNAP_HOUSEHOLDS_TARGET_TABLE,
    US_SNAP_STATE_TAKE_UP_ANCHOR,
    US_SNAP_STATE_TAKE_UP_STAGE,
    US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    US_SOURCE_MANIFEST,
    us_snap_state_take_up_gate,
    with_us_snap_state_take_up,
)
from microcosm.build.us_runtime.snap_state_take_up import (
    US_SNAP_ELIGIBILITY_COLUMN,
    us_snap_source_spm_unit_table,
    with_us_snap_state_take_up_rate,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024


def _person_table(rows: list[dict]) -> pd.DataFrame:
    baseline = {"SPM_SNAPSUB": 0.0}
    records = []
    for index, row in enumerate(rows):
        record = dict(baseline)
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("person_spm_unit_id", index + 1)
        records.append(record)
    return pd.DataFrame(records)


def _us_frame(
    person_rows: list[dict],
    *,
    household_weights: list[float] | None = None,
) -> Frame:
    person = _person_table(person_rows)
    n = len(person)
    person["person_household_id"] = person["person_spm_unit_id"]
    household_ids = person["person_household_id"].to_numpy()
    unique_households = np.unique(household_ids)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(n, dtype="int64") + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": unique_households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": unique_households + 1_000}),
        "spm_unit": pd.DataFrame(
            {"spm_unit_id": np.unique(person["person_spm_unit_id"])}
        ),
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


def _state_targets(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": state,
                "target": target,
                "source_record_id": f"usda_snap.fy2024.test.{state}",
            }
            for state, target in rows
        ]
    )


class TestManifestDeclaration:
    def test_stage_is_declared_with_calibrated_ops(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()[US_SNAP_STATE_TAKE_UP_STAGE]
        assert tuple(stage.outputs) == (US_SNAP_TAKE_UP_OUTPUT_COLUMN,)
        kinds = [operation.kind for operation in stage.operations]
        assert kinds == [
            "read_table",
            "assign_binary_from_rate",
            "calibrate_binary_assignment",
        ]
        calibrate = stage.operations[-1]
        assert calibrate.parameters["targets"] == [US_SNAP_HOUSEHOLDS_TARGET_TABLE]
        assert calibrate.parameters["preserve_true_anchor"] == (
            US_SNAP_STATE_TAKE_UP_ANCHOR
        )
        assert calibrate.parameters["domain"] == US_SNAP_ELIGIBILITY_COLUMN


class TestSourceTable:
    def test_anchor_derives_from_reported_subsidy(self) -> None:
        frame = _us_frame(
            [{"SPM_SNAPSUB": 2_400.0}, {"SPM_SNAPSUB": 0.0}, {"SPM_SNAPSUB": 0.0}]
        )
        table = us_snap_source_spm_unit_table(
            frame,
            is_snap_eligible=np.asarray([True, True, False]),
            state_fips=np.asarray(["6", "6", "36"]),
            seed=0,
        )
        assert table[US_SNAP_STATE_TAKE_UP_ANCHOR].tolist() == [True, False, False]
        assert table["state_fips"].tolist() == ["06", "06", "36"]
        assert table["spm_unit_weight"].tolist() == [1.0, 1.0, 1.0]

    def test_misaligned_inputs_are_refused(self) -> None:
        frame = _us_frame([{}, {}])
        with pytest.raises(ValueError, match="must align with the spm_unit table"):
            us_snap_source_spm_unit_table(
                frame,
                is_snap_eligible=np.asarray([True]),
                state_fips=np.asarray(["06", "06"]),
                seed=0,
            )

    def test_missing_raw_column_is_refused(self) -> None:
        frame = _us_frame([{}, {}])
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["person"] = tables["person"].drop(columns=["SPM_SNAPSUB"])
        stripped = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        )
        with pytest.raises(ValueError, match="SPM_SNAPSUB"):
            us_snap_source_spm_unit_table(
                stripped,
                is_snap_eligible=np.asarray([True, True]),
                state_fips=np.asarray(["06", "06"]),
                seed=0,
            )


class TestRatePrior:
    def test_rate_is_target_over_weighted_eligibles(self) -> None:
        table = pd.DataFrame(
            {
                "state_fips": ["06", "06", "06", "36"],
                US_SNAP_ELIGIBILITY_COLUMN: [True, True, False, True],
                "spm_unit_weight": [2.0, 2.0, 2.0, 1.0],
            }
        )
        rated = with_us_snap_state_take_up_rate(
            table, _state_targets([("06", 2.0), ("36", 5.0)])
        )
        rates = rated.set_index(rated.index)["snap_take_up_rate"]
        assert rates.iloc[0] == pytest.approx(0.5)  # 2 / (2 + 2)
        assert rates.iloc[3] == 1.0  # 5 / 1 clipped


class TestAssignment:
    def test_unsaturated_state_hits_the_fns_count(self) -> None:
        # State 06: one eligible reporter + five eligible non-reporters,
        # unit weight 1, FNS count 3 -> reporter plus exactly two fills.
        rows = [{"SPM_SNAPSUB": 1_200.0}] + [{}] * 5
        frame = _us_frame(rows)
        result, diagnostics = with_us_snap_state_take_up(
            frame,
            is_snap_eligible=np.ones(6, dtype=bool),
            state_fips=np.asarray(["06"] * 6),
            state_targets=_state_targets([("06", 3.0)]),
            seed=0,
        )
        spm_unit = result.table("spm_unit")
        takes_up = spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert bool(takes_up.iloc[0]) is True  # the reporter
        assert int(takes_up.sum()) == 3
        state_row = diagnostics["states"][0]
        assert state_row["caseload_weight"] == pytest.approx(3.0)
        assert state_row["saturated"] is False
        assert us_snap_state_take_up_gate(diagnostics).passed

    def test_saturated_state_enrolls_every_eligible_unit(self) -> None:
        rows = [{}] * 3
        frame = _us_frame(rows)
        result, diagnostics = with_us_snap_state_take_up(
            frame,
            is_snap_eligible=np.asarray([True, True, False]),
            state_fips=np.asarray(["36"] * 3),
            state_targets=_state_targets([("36", 10.0)]),
            seed=0,
        )
        spm_unit = result.table("spm_unit")
        takes_up = spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert bool(takes_up.iloc[0]) and bool(takes_up.iloc[1])
        state_row = diagnostics["states"][0]
        assert state_row["saturated"] is True
        assert diagnostics["saturated_states"] == ["36"]
        gate = us_snap_state_take_up_gate(diagnostics)
        assert gate.passed

    def test_reported_recipients_survive_downward_calibration(self) -> None:
        # Two eligible reporters against an FNS count of one: calibration
        # cannot remove anchored units, so both keep the flag and the gate
        # treats the anchor mass as the reachable floor.
        rows = [{"SPM_SNAPSUB": 900.0}, {"SPM_SNAPSUB": 600.0}, {}, {}]
        frame = _us_frame(rows)
        result, diagnostics = with_us_snap_state_take_up(
            frame,
            is_snap_eligible=np.ones(4, dtype=bool),
            state_fips=np.asarray(["48"] * 4),
            state_targets=_state_targets([("48", 1.0)]),
            seed=0,
        )
        takes_up = result.table("spm_unit")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert bool(takes_up.iloc[0]) and bool(takes_up.iloc[1])
        assert diagnostics["national"]["anchored_not_taking_up_count"] == 0
        assert us_snap_state_take_up_gate(diagnostics).passed

    def test_empty_targets_are_refused(self) -> None:
        frame = _us_frame([{}])
        with pytest.raises(ValueError, match="non-empty FNS state household"):
            with_us_snap_state_take_up(
                frame,
                is_snap_eligible=np.asarray([True]),
                state_fips=np.asarray(["06"]),
                state_targets=pd.DataFrame(),
                seed=0,
            )


class TestGate:
    def test_state_without_target_fails(self) -> None:
        frame = _us_frame([{}, {}])
        _, diagnostics = with_us_snap_state_take_up(
            frame,
            is_snap_eligible=np.asarray([True, True]),
            state_fips=np.asarray(["06", "36"]),
            state_targets=_state_targets([("06", 1.0)]),
            seed=0,
        )
        gate = us_snap_state_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("36" in failure for failure in gate.failures)

    def test_positive_target_with_zero_eligibility_fails(self) -> None:
        frame = _us_frame([{}, {}])
        _, diagnostics = with_us_snap_state_take_up(
            frame,
            is_snap_eligible=np.asarray([False, False]),
            state_fips=np.asarray(["06", "06"]),
            state_targets=_state_targets([("06", 2.0)]),
            seed=0,
        )
        gate = us_snap_state_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("eligibility feed collapsed" in f for f in gate.failures)

    def test_dropped_anchor_fails_even_when_counts_agree(self) -> None:
        diagnostics = {
            "states": [
                {
                    "state_fips": "06",
                    "target": 2.0,
                    "eligible_weight": 5.0,
                    "anchored_eligible_weight": 1.0,
                    "caseload_weight": 2.0,
                    "saturated": False,
                    "anchored_not_taking_up_count": 1,
                    "max_unit_weight": 1.0,
                }
            ],
            "states_without_targets": [],
        }
        gate = us_snap_state_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("anchor was not preserved" in f for f in gate.failures)

    def test_unsaturated_count_miss_fails(self) -> None:
        diagnostics = {
            "states": [
                {
                    "state_fips": "17",
                    "target": 100.0,
                    "eligible_weight": 500.0,
                    "anchored_eligible_weight": 10.0,
                    "caseload_weight": 60.0,
                    "saturated": False,
                    "anchored_not_taking_up_count": 0,
                    "max_unit_weight": 1.0,
                }
            ],
            "states_without_targets": [],
        }
        gate = us_snap_state_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("misses the FNS count" in f for f in gate.failures)

    def test_universal_take_up_landmine_fails(self) -> None:
        diagnostics = {
            "states": [
                {
                    "state_fips": "17",
                    "target": 100.0,
                    "eligible_weight": 500.0,
                    "anchored_eligible_weight": 10.0,
                    "caseload_weight": 500.0,
                    "saturated": False,
                    "anchored_not_taking_up_count": 0,
                    "max_unit_weight": 1.0,
                }
            ],
            "states_without_targets": [],
        }
        gate = us_snap_state_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("universal-take-up landmine" in f for f in gate.failures)
