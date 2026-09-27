"""US SNAP take-up stage tests (microcosm #243)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeError,
    run_source_stage,
)
from microcosm.build.us_runtime import (
    US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    US_SNAP_TAKE_UP_STAGE_NAME,
    derive_us_snap_take_up_from_manifest,
    us_snap_take_up_signal_gate,
    us_snap_take_up_stage_spec,
    us_snap_take_up_summary,
    with_us_snap_take_up_inputs,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024
_HANDLERS = {"derive_snap_take_up": derive_us_snap_take_up_from_manifest}


def _stage_spec(*, rate: float, source: str = "https://example.com") -> SourceStageSpec:
    return SourceStageSpec.from_mapping(
        {
            "stage": US_SNAP_TAKE_UP_STAGE_NAME,
            "survey": "test ASEC",
            "source": "https://example.com",
            "grain": "spm_unit",
            "operations": [
                {"kind": "read_table", "table": "person"},
                {
                    "kind": "derive_snap_take_up",
                    "seed_from_build_config": True,
                    "take_up_rate": {"value": rate, "source": source},
                },
            ],
            "outputs": [US_SNAP_TAKE_UP_OUTPUT_COLUMN],
        }
    )


def _person_table(rows: list[dict]) -> pd.DataFrame:
    baseline = {"SPM_SNAPSUB": 0.0, "person_weight": 1.0}
    records = []
    for index, row in enumerate(rows):
        record = dict(baseline)
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("person_spm_unit_id", index + 1)
        records.append(record)
    return pd.DataFrame(records)


def _run(rows: list[dict], *, rate: float, seed: int = 0) -> pd.DataFrame:
    return run_source_stage(
        _stage_spec(rate=rate),
        tables={"person": _person_table(rows)},
        operation_handlers=_HANDLERS,
        config=SourceRuntimeConfig(seed=seed, target_year=TIME_PERIOD),
    )


def _us_frame(
    person_rows: list[dict],
    *,
    household_weights: list[float] | None = None,
    spm_extra: dict | None = None,
) -> Frame:
    person = _person_table(person_rows).drop(columns=["person_weight"])
    n = len(person)
    person["person_household_id"] = person["person_spm_unit_id"]
    household_ids = person["person_household_id"].to_numpy()
    unique_households = np.unique(household_ids)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(n, dtype="int64") + 4_000
    spm_unit = pd.DataFrame({"spm_unit_id": np.unique(person["person_spm_unit_id"])})
    for column, values in (spm_extra or {}).items():
        spm_unit[column] = values
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": unique_households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": unique_households + 1_000}),
        "spm_unit": spm_unit,
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


class TestManifestDeclaration:
    def test_stage_is_declared_with_cited_rate(self) -> None:
        spec = us_snap_take_up_stage_spec()
        assert spec.stage == US_SNAP_TAKE_UP_STAGE_NAME
        assert tuple(spec.outputs) == (US_SNAP_TAKE_UP_OUTPUT_COLUMN,)
        derive = next(op for op in spec.operations if op.kind == "derive_snap_take_up")
        rate = derive.parameters["take_up_rate"]
        assert 0.5 < float(rate["value"]) <= 1.0
        assert str(rate["source"]).startswith("https://")

    def test_handler_is_registered(self) -> None:
        handlers = us_source_operation_handlers()
        assert handlers["derive_snap_take_up"] is derive_us_snap_take_up_from_manifest


class TestDerivation:
    def test_reported_recipients_always_take_up(self) -> None:
        rows = [{"SPM_SNAPSUB": 3_000.0}] + [{"SPM_SNAPSUB": 0.0}] * 9
        output = _run(rows, rate=0.1)
        by_unit = output.set_index("spm_unit_id")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert bool(by_unit.loc[1])

    def test_non_reporters_fill_to_the_weighted_rate(self) -> None:
        # 2 reporting units + 98 non-reporting, equal weights, rate 0.5:
        # non-reporter fill rate is (50 - 2) / 98, so roughly half take up.
        rows = [{"SPM_SNAPSUB": 1_200.0}] * 2 + [{"SPM_SNAPSUB": 0.0}] * 98
        output = _run(rows, rate=0.5)
        share = output[US_SNAP_TAKE_UP_OUTPUT_COLUMN].mean()
        assert 0.35 <= share <= 0.65

    def test_reporters_beyond_the_rate_grant_no_non_reporters(self) -> None:
        rows = [{"SPM_SNAPSUB": 1_200.0}] * 8 + [{"SPM_SNAPSUB": 0.0}] * 2
        output = _run(rows, rate=0.5)
        by_unit = output.set_index("spm_unit_id")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert by_unit.iloc[:8].all()
        assert not by_unit.iloc[8:].any()

    def test_draws_are_seed_stable(self) -> None:
        rows = [{"SPM_SNAPSUB": 0.0}] * 40
        first = _run(rows, rate=0.5, seed=7)
        second = _run(rows, rate=0.5, seed=7)
        assert first[US_SNAP_TAKE_UP_OUTPUT_COLUMN].tolist() == (
            second[US_SNAP_TAKE_UP_OUTPUT_COLUMN].tolist()
        )

    def test_source_identity_keys_clone_consistent_draws(self) -> None:
        # Two "clones" of the same source unit must always agree.
        clone = {
            "SPM_SNAPSUB": 0.0,
            "source_year": 2023,
            "source_household_id": 555,
            "source_person_id": 7,
        }
        rows = [
            {**clone, "person_spm_unit_id": 1},
            {**clone, "person_spm_unit_id": 2},
        ] + [{"SPM_SNAPSUB": 0.0, "person_spm_unit_id": 3 + i} for i in range(20)]
        output = _run(rows, rate=0.5)
        by_unit = output.set_index("spm_unit_id")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert bool(by_unit.loc[1]) == bool(by_unit.loc[2])

    def test_multi_person_units_report_once(self) -> None:
        rows = [
            {"person_spm_unit_id": 1, "SPM_SNAPSUB": 2_400.0},
            {"person_spm_unit_id": 1, "SPM_SNAPSUB": 2_400.0},
            {"person_spm_unit_id": 2, "SPM_SNAPSUB": 0.0},
        ]
        output = _run(rows, rate=0.0001)
        assert len(output) == 2
        by_unit = output.set_index("spm_unit_id")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert bool(by_unit.loc[1])

    def test_rate_without_citation_is_refused(self) -> None:
        spec = _stage_spec(rate=0.82, source="")
        with pytest.raises(SourceRuntimeError, match="citation"):
            run_source_stage(
                spec,
                tables={"person": _person_table([{"SPM_SNAPSUB": 0.0}])},
                operation_handlers=_HANDLERS,
                config=SourceRuntimeConfig(seed=0, target_year=TIME_PERIOD),
            )

    def test_missing_raw_column_is_named(self) -> None:
        table = _person_table([{"SPM_SNAPSUB": 0.0}]).drop(columns=["SPM_SNAPSUB"])
        with pytest.raises(SourceRuntimeError, match="SPM_SNAPSUB"):
            run_source_stage(
                _stage_spec(rate=0.82),
                tables={"person": table},
                operation_handlers=_HANDLERS,
                config=SourceRuntimeConfig(seed=0, target_year=TIME_PERIOD),
            )


class TestFrameIntegration:
    def _mixed_rows(self, n_reporters: int = 20, n_total: int = 100) -> list[dict]:
        return [{"SPM_SNAPSUB": 2_000.0}] * n_reporters + [{"SPM_SNAPSUB": 0.0}] * (
            n_total - n_reporters
        )

    def test_with_inputs_writes_the_spm_unit_column(self) -> None:
        frame = with_us_snap_take_up_inputs(
            _us_frame(self._mixed_rows()), seed=0, time_period=TIME_PERIOD
        )
        spm_unit = frame.table("spm_unit")
        assert US_SNAP_TAKE_UP_OUTPUT_COLUMN in spm_unit.columns
        assert spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN].iloc[:20].all()

    def test_frame_with_signal_passes_through_untouched(self) -> None:
        derived = with_us_snap_take_up_inputs(
            _us_frame(self._mixed_rows()), seed=0, time_period=TIME_PERIOD
        )
        again = with_us_snap_take_up_inputs(derived, seed=9, time_period=TIME_PERIOD)
        assert again is derived

    def test_constant_true_landmine_is_recomputed(self) -> None:
        # The published failure mode: takes_up constant True for every unit.
        frame = _us_frame(
            self._mixed_rows(n_reporters=2, n_total=10),
            spm_extra={US_SNAP_TAKE_UP_OUTPUT_COLUMN: [True] * 10},
        )
        healed = with_us_snap_take_up_inputs(frame, seed=0, time_period=TIME_PERIOD)
        values = healed.table("spm_unit")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert values.iloc[:2].all()
        assert values.nunique() == 2


class TestGate:
    def test_plausible_assignment_passes(self) -> None:
        frame = with_us_snap_take_up_inputs(
            _us_frame([{"SPM_SNAPSUB": 2_000.0}] * 20 + [{"SPM_SNAPSUB": 0.0}] * 80),
            seed=0,
            time_period=TIME_PERIOD,
        )
        gate = us_snap_take_up_signal_gate(frame)
        assert gate.passed
        summary = us_snap_take_up_summary(frame)
        assert 0.70 <= summary["take_up_share"] <= 0.95
        assert summary["reporters_not_taking_up"] == 0

    def test_missing_column_fails(self) -> None:
        gate = us_snap_take_up_signal_gate(_us_frame([{"SPM_SNAPSUB": 0.0}] * 4))
        assert not gate.passed
        assert "missing" in gate.failures[0]

    def test_constant_true_fails(self) -> None:
        frame = _us_frame(
            [{"SPM_SNAPSUB": 0.0}] * 4,
            spm_extra={US_SNAP_TAKE_UP_OUTPUT_COLUMN: [True] * 4},
        )
        gate = us_snap_take_up_signal_gate(frame)
        assert not gate.passed
        assert any("constant" in failure for failure in gate.failures)

    def test_reporter_without_take_up_fails(self) -> None:
        frame = _us_frame(
            [{"SPM_SNAPSUB": 3_000.0}] + [{"SPM_SNAPSUB": 0.0}] * 3,
            spm_extra={US_SNAP_TAKE_UP_OUTPUT_COLUMN: [False, True, True, False]},
        )
        gate = us_snap_take_up_signal_gate(frame)
        assert not gate.passed
        assert any("reported SNAP receipt but" in failure for failure in gate.failures)

    def test_share_outside_band_fails(self) -> None:
        frame = _us_frame(
            [{"SPM_SNAPSUB": 0.0}] * 10,
            spm_extra={US_SNAP_TAKE_UP_OUTPUT_COLUMN: [True] + [False] * 9},
        )
        gate = us_snap_take_up_signal_gate(frame)
        assert not gate.passed
        assert any("take-up share" in failure for failure in gate.failures)
