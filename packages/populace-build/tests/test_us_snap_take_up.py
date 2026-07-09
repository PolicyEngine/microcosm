"""US state-conditional SNAP take-up tests (populace #372)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.source_manifest import SourceStageSpec
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeError,
    run_source_stage,
)
from populace.build.us_runtime import (
    US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    US_SNAP_TAKE_UP_STAGE_NAME,
    derive_us_snap_take_up_from_manifest,
    us_snap_take_up_diagnostics,
    us_snap_take_up_signal_gate,
    us_snap_take_up_stage_spec,
    with_us_snap_take_up_inputs,
)
from populace.build.us_runtime.source_runtime import us_source_operation_handlers
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024
_HANDLERS = {"derive_snap_take_up": derive_us_snap_take_up_from_manifest}


def _stage_spec(
    *,
    rates: dict[str, float],
    source: str = "https://example.com/fns.pdf",
) -> SourceStageSpec:
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
                    "state_take_up_rates": {
                        "fiscal_year": 2022,
                        "measure": "eligible_person_share",
                        "source": source,
                        "values": rates,
                    },
                },
            ],
            "outputs": [US_SNAP_TAKE_UP_OUTPUT_COLUMN],
        }
    )


def _person_table(rows: list[dict]) -> pd.DataFrame:
    baseline = {
        "SPM_SNAPSUB": 0.0,
        "person_weight": 1.0,
        "state_fips": "05",
        "is_snap_eligible": True,
        "snap_unit_size": 1,
    }
    records = []
    for index, row in enumerate(rows):
        record = dict(baseline)
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("person_spm_unit_id", index + 1)
        records.append(record)
    return pd.DataFrame(records)


def _run(
    rows: list[dict],
    *,
    rates: dict[str, float],
    seed: int = 0,
) -> pd.DataFrame:
    return run_source_stage(
        _stage_spec(rates=rates),
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
    person = _person_table(person_rows).drop(
        columns=["person_weight", "state_fips", "is_snap_eligible", "snap_unit_size"]
    )
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


def _with_inputs(
    frame: Frame,
    *,
    state: str = "05",
    seed: int = 0,
) -> tuple[Frame, dict[str, object]]:
    count = frame.n("spm_unit")
    return with_us_snap_take_up_inputs(
        frame,
        is_snap_eligible=np.ones(count, dtype=bool),
        snap_unit_size=np.ones(count, dtype=np.int64),
        state_fips=np.full(count, state),
        seed=seed,
        time_period=TIME_PERIOD,
    )


def _assigned_for_gate(
    *,
    takes_up: list[bool],
    reported: list[bool] | None = None,
    eligible_weights: list[float] | None = None,
    target_rate: float | None = 0.59,
) -> pd.DataFrame:
    count = len(takes_up)
    return pd.DataFrame(
        {
            "state_fips": ["05"] * count,
            "eligible_person_weight": eligible_weights or [1.0] * count,
            "reported_snap_receipt": reported or [False] * count,
            "stable_spm_unit_draw": np.linspace(0.01, 0.99, count),
            "snap_take_up_target_rate": [target_rate] * count,
            US_SNAP_TAKE_UP_OUTPUT_COLUMN: takes_up,
        }
    )


class TestManifestDeclaration:
    def test_stage_declares_all_state_rates_with_official_source(self) -> None:
        spec = us_snap_take_up_stage_spec()
        assert spec.stage == US_SNAP_TAKE_UP_STAGE_NAME
        assert tuple(spec.outputs) == (US_SNAP_TAKE_UP_OUTPUT_COLUMN,)
        derive = next(op for op in spec.operations if op.kind == "derive_snap_take_up")
        declaration = derive.parameters["state_take_up_rates"]
        assert declaration["fiscal_year"] == 2022
        assert len(declaration["values"]) == 51
        assert declaration["values"]["05"] == 0.59
        assert declaration["values"]["17"] == 1.0
        assert str(declaration["source"]).startswith("https://www.fns.usda.gov/")

    def test_handler_is_registered(self) -> None:
        handlers = us_source_operation_handlers()
        assert handlers["derive_snap_take_up"] is derive_us_snap_take_up_from_manifest


class TestDerivation:
    def test_reported_recipients_always_take_up(self) -> None:
        rows = [{"SPM_SNAPSUB": 3_000.0}] + [{}] * 9
        output = _run(rows, rates={"05": 0.1})
        by_unit = output.set_index("spm_unit_id")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert bool(by_unit.loc[1])

    def test_state_rates_change_the_amount_of_recipient_support(self) -> None:
        rows = ([{"state_fips": "05"}] * 100) + ([{"state_fips": "17"}] * 100)
        output = _run(rows, rates={"05": 0.59, "17": 1.0})
        shares = output.groupby("state_fips")[US_SNAP_TAKE_UP_OUTPUT_COLUMN].mean()
        assert shares["05"] == 0.59
        assert shares["17"] == 1.0

    def test_reporters_beyond_target_are_the_floor(self) -> None:
        rows = [{"SPM_SNAPSUB": 1_200.0}] * 8 + [{}] * 2
        output = _run(rows, rates={"05": 0.5})
        by_unit = output.set_index("spm_unit_id")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert by_unit.iloc[:8].all()
        assert not by_unit.iloc[8:].any()

    def test_calibration_uses_eligible_person_weight(self) -> None:
        sizes = [1, 2, 3] * 20
        rows = [{"snap_unit_size": size} for size in sizes]
        output = _run(rows, rates={"05": 0.59})
        modeled = output.loc[
            output[US_SNAP_TAKE_UP_OUTPUT_COLUMN], "eligible_person_weight"
        ].sum()
        target = 0.59 * output["eligible_person_weight"].sum()
        assert abs(modeled - target) <= max(sizes)

    def test_draws_are_seed_stable(self) -> None:
        rows = [{}] * 40
        first = _run(rows, rates={"05": 0.5}, seed=7)
        second = _run(rows, rates={"05": 0.5}, seed=7)
        assert (
            first[US_SNAP_TAKE_UP_OUTPUT_COLUMN].tolist()
            == second[US_SNAP_TAKE_UP_OUTPUT_COLUMN].tolist()
        )

    def test_source_identity_clones_are_selected_together(self) -> None:
        clone = {
            "source_year": 2023,
            "source_household_id": 555,
            "source_person_id": 7,
        }
        rows = [
            {**clone, "person_spm_unit_id": 1},
            {**clone, "person_spm_unit_id": 2},
        ] + [{"person_spm_unit_id": 3 + index} for index in range(20)]
        output = _run(rows, rates={"05": 0.5})
        by_unit = output.set_index("spm_unit_id")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert bool(by_unit.loc[1]) == bool(by_unit.loc[2])

    def test_multi_person_units_are_emitted_once(self) -> None:
        rows = [
            {
                "person_spm_unit_id": 1,
                "SPM_SNAPSUB": 2_400.0,
                "snap_unit_size": 2,
            },
            {
                "person_spm_unit_id": 1,
                "SPM_SNAPSUB": 2_400.0,
                "snap_unit_size": 2,
            },
            {"person_spm_unit_id": 2},
        ]
        output = _run(rows, rates={"05": 0.1})
        assert len(output) == 2
        assert bool(
            output.set_index("spm_unit_id").loc[1, US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        )

    def test_rate_without_citation_is_refused(self) -> None:
        spec = _stage_spec(rates={"05": 0.59}, source="")
        with pytest.raises(SourceRuntimeError, match="citation"):
            run_source_stage(
                spec,
                tables={"person": _person_table([{}])},
                operation_handlers=_HANDLERS,
                config=SourceRuntimeConfig(seed=0, target_year=TIME_PERIOD),
            )

    def test_represented_state_without_rate_is_refused(self) -> None:
        with pytest.raises(SourceRuntimeError, match="no cited FNS"):
            _run([{"state_fips": "06"}], rates={"05": 0.59})

    def test_missing_policy_input_is_named(self) -> None:
        table = _person_table([{}]).drop(columns=["is_snap_eligible"])
        with pytest.raises(SourceRuntimeError, match="is_snap_eligible"):
            run_source_stage(
                _stage_spec(rates={"05": 0.59}),
                tables={"person": table},
                operation_handlers=_HANDLERS,
                config=SourceRuntimeConfig(seed=0, target_year=TIME_PERIOD),
            )


class TestFrameIntegration:
    def _mixed_rows(self, n_reporters: int = 20, n_total: int = 100) -> list[dict]:
        return [{"SPM_SNAPSUB": 2_000.0}] * n_reporters + [{}] * (n_total - n_reporters)

    def test_with_inputs_writes_the_spm_unit_column_and_diagnostics(self) -> None:
        frame, diagnostics = _with_inputs(_us_frame(self._mixed_rows()))
        spm_unit = frame.table("spm_unit")
        assert US_SNAP_TAKE_UP_OUTPUT_COLUMN in spm_unit.columns
        assert spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN].iloc[:20].all()
        assert diagnostics["states"][0]["target_rate"] == 0.59

    def test_owned_column_is_always_recomputed(self) -> None:
        derived, _ = _with_inputs(_us_frame(self._mixed_rows()), seed=0)
        original = derived.table("spm_unit")[US_SNAP_TAKE_UP_OUTPUT_COLUMN].copy()
        recomputed, _ = _with_inputs(derived, seed=9)
        updated = recomputed.table("spm_unit")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert not original.equals(updated)

    def test_constant_true_landmine_is_recomputed(self) -> None:
        frame = _us_frame(
            self._mixed_rows(n_reporters=2, n_total=10),
            spm_extra={US_SNAP_TAKE_UP_OUTPUT_COLUMN: [True] * 10},
        )
        healed, _ = _with_inputs(frame)
        values = healed.table("spm_unit")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert values.iloc[:2].all()
        assert values.nunique() == 2

    def test_input_arrays_must_align_with_spm_units(self) -> None:
        frame = _us_frame(self._mixed_rows(n_reporters=2, n_total=10))
        with pytest.raises(ValueError, match="10 units, 9 values"):
            with_us_snap_take_up_inputs(
                frame,
                is_snap_eligible=np.ones(9, dtype=bool),
                snap_unit_size=np.ones(10),
                state_fips=np.full(10, "05"),
                seed=0,
                time_period=TIME_PERIOD,
            )


class TestGate:
    def test_state_faithful_assignment_passes(self) -> None:
        _, diagnostics = _with_inputs(
            _us_frame([{"SPM_SNAPSUB": 2_000.0}] * 20 + [{}] * 80)
        )
        gate = us_snap_take_up_signal_gate(diagnostics)
        assert gate.passed
        state = diagnostics["states"][0]
        assert abs(state["modeled_participation_rate"] - 0.59) <= 0.01
        assert state["reported_not_taking_up_count"] == 0

    def test_missing_rate_fails(self) -> None:
        diagnostics = us_snap_take_up_diagnostics(
            _assigned_for_gate(takes_up=[True, False], target_rate=None)
        )
        gate = us_snap_take_up_signal_gate(diagnostics)
        assert not gate.passed
        assert "without cited FNS" in gate.failures[0]

    def test_constant_true_below_full_target_fails(self) -> None:
        diagnostics = us_snap_take_up_diagnostics(
            _assigned_for_gate(takes_up=[True] * 10)
        )
        gate = us_snap_take_up_signal_gate(diagnostics)
        assert not gate.passed
        assert any("constant column" in failure for failure in gate.failures)

    def test_reporter_without_take_up_fails(self) -> None:
        diagnostics = us_snap_take_up_diagnostics(
            _assigned_for_gate(
                takes_up=[False, True, False, True],
                reported=[True, False, False, False],
            )
        )
        gate = us_snap_take_up_signal_gate(diagnostics)
        assert not gate.passed
        assert any("reported SNAP recipient" in failure for failure in gate.failures)

    def test_collapsed_eligibility_fails(self) -> None:
        diagnostics = us_snap_take_up_diagnostics(
            _assigned_for_gate(
                takes_up=[False, True],
                eligible_weights=[0.0, 0.0],
            )
        )
        gate = us_snap_take_up_signal_gate(diagnostics)
        assert not gate.passed
        assert any(
            "eligibility surface collapsed" in failure for failure in gate.failures
        )

    def test_rate_miss_outside_granularity_fails(self) -> None:
        diagnostics = us_snap_take_up_diagnostics(
            _assigned_for_gate(takes_up=[True] + [False] * 99)
        )
        gate = us_snap_take_up_signal_gate(diagnostics)
        assert not gate.passed
        assert any(
            "misses the FNS/anchor floor" in failure for failure in gate.failures
        )
