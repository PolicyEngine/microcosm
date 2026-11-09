"""ASEC retirement-contribution restoration and PUF-half QRF treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.retirement_contributions as module
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.retirement_contributions import (
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    derive_us_retirement_contributions_from_manifest,
    us_retirement_contributions_signal_gate,
    us_retirement_contributions_stage_spec,
    us_retirement_contributions_summary,
    with_us_retirement_contribution_inputs,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _person_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
            "person_household_id": np.asarray([10, 20, 30, 40], dtype="int64"),
            "person_tax_unit_id": np.asarray([100, 200, 300, 400], dtype="int64"),
            "person_spm_unit_id": np.asarray([1_000, 2_000, 3_000, 4_000]),
            "person_family_id": np.asarray([10_000, 20_000, 30_000, 40_000]),
            "person_marital_unit_id": np.asarray([100_000, 200_000, 300_000, 400_000]),
            # Wage-only, self-employment-only, both, and neither.
            "RETCB_VAL": [1.0, 1.0, 1.0, 1.0],
            "WSAL_VAL": [50_000.0, 0.0, 50_000.0, 0.0],
            "SEMP_VAL": [0.0, 30_000.0, 30_000.0, 0.0],
            "employment_income_before_lsr": [50_000.0, 0.0, 50_000.0, 0.0],
            "self_employment_income_before_lsr": [0.0, 30_000.0, 30_000.0, 0.0],
            "age": [35, 45, 55, 25],
            "is_female": [False, True, False, True],
            "has_esi": [True, False, True, False],
            "tax_unit_role_input": ["HEAD", "HEAD", "HEAD", "HEAD"],
            "social_security_retirement": [0.0, 0.0, 2_000.0, 0.0],
            "social_security_disability": [0.0, 0.0, 0.0, 0.0],
            "social_security_dependents": [0.0, 0.0, 0.0, 0.0],
            "social_security_survivors": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _frame() -> Frame:
    person = _person_source()
    ids = {
        "household": [10, 20, 30, 40],
        "tax_unit": [100, 200, 300, 400],
        "spm_unit": [1_000, 2_000, 3_000, 4_000],
        "family": [10_000, 20_000, 30_000, 40_000],
        "marital_unit": [100_000, 200_000, 300_000, 400_000],
    }
    tables = {
        entity: pd.DataFrame({f"{entity}_id": np.asarray(values, dtype="int64")})
        for entity, values in ids.items()
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = [
        "SINGLE",
        "SINGLE",
        "SINGLE",
        "SINGLE",
    ]
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([1.0, 1.0, 1.0, 1.0]),
                WeightKind.DESIGN,
            )
        },
    )


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    operation = next(
        operation
        for operation in us_retirement_contributions_stage_spec().operations
        if operation.kind == "derive_retirement_contributions"
    )
    return derive_us_retirement_contributions_from_manifest(frame, operation, None)


def _stacked_frame() -> Frame:
    direct = with_us_retirement_contribution_inputs(
        _frame(),
        seed=0,
        time_period=2024,
    )
    stacked = clone_us_frame_for_puf_support(direct)
    person = stacked.table("person")
    source_record = np.tile(np.arange(4, dtype=np.int64), 2)
    person["person_spine_source_id"] = source_record
    person["person_support_channel"] = np.where(
        source_record < 2,
        "asec",
        "acs",
    )
    acs = person["person_support_channel"].eq("acs")
    person.loc[acs, ["RETCB_VAL", "WSAL_VAL", "SEMP_VAL"]] = np.nan
    return stacked


def test_stage_manifest_pins_sources_operations_and_five_desired_leaves() -> None:
    spec = us_retirement_contributions_stage_spec()

    assert spec.stage == "retirement_contributions"
    assert spec.grain == "person"
    assert tuple(spec.outputs) == US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_retirement_contributions",
        "impute_retirement_contributions_to_puf_support",
    ]
    derive = spec.operations[1]
    assert derive.parameters == {
        "se_pension_share": 0.046,
        "dc_share_of_remainder": 0.908,
        "roth_dc_share": 0.15,
        "traditional_ira_share": 0.392,
    }


def test_direct_split_matches_archived_wage_and_self_employment_cases() -> None:
    result = _derive(_person_source())

    np.testing.assert_allclose(
        result[list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)].to_numpy(),
        np.asarray(
            [
                [0.7718, 0.1362, 0.036064, 0.055936, 0.0],
                [0.0, 0.0, 0.373968, 0.580032, 0.046],
                [0.7362972, 0.1299348, 0.034405056, 0.053362944, 0.046],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        rtol=0,
        atol=1e-12,
    )


def test_direct_split_fails_closed_without_measured_asec_source() -> None:
    with pytest.raises(SourceRuntimeError, match="RETCB_VAL"):
        _derive(_person_source().drop(columns=["RETCB_VAL"]))


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [(np.nan, "nonfinite"), (-1.0, "negative")],
)
def test_direct_split_rejects_invalid_measured_totals(
    bad_value: float,
    message: str,
) -> None:
    person = _person_source()
    person.loc[0, "RETCB_VAL"] = bad_value

    with pytest.raises(SourceRuntimeError, match=message):
        _derive(person)


def test_with_inputs_materializes_signal_and_preserves_reported_total() -> None:
    result = with_us_retirement_contribution_inputs(
        _frame(),
        seed=0,
        time_period=2024,
    )

    gate = us_retirement_contributions_signal_gate(result)
    assert gate.passed, gate.failures
    person = result.table("person")
    allocated = person[list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)].sum(axis=1)
    np.testing.assert_allclose(allocated.iloc[:3], person["RETCB_VAL"].iloc[:3])
    assert allocated.iloc[3] == 0.0


def test_stacked_gate_validates_physical_source_and_reconciles_direct_role() -> None:
    stacked = _stacked_frame()

    gate = us_retirement_contributions_signal_gate(stacked)

    assert gate.passed, gate.failures
    assert gate.details["source_rows"] == 4
    assert gate.details["source_reconciliation_rows"] == 2
    assert gate.details["allocation_mismatch_rows"] == 0

    person = stacked.table("person")
    asec_puf_role = person["person_support_channel"].eq("asec") & person[
        "person_support_clone_index"
    ].eq(1)
    person.loc[person.index[asec_puf_role][0], "RETCB_VAL"] = np.nan
    with pytest.raises(SourceRuntimeError, match="RETCB_VAL"):
        us_retirement_contributions_summary(stacked)


def test_puf_half_uses_qrf_predictions_and_applies_income_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = with_us_retirement_contribution_inputs(
        _frame(),
        seed=0,
        time_period=2024,
    )
    expanded = clone_us_frame_for_puf_support(direct)
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            calls["test"] = test.copy()
            return pd.DataFrame(
                {
                    column: np.full(len(test), index + 1.0)
                    for index, column in enumerate(
                        US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
                    )
                },
                index=test.index,
            )

    class FakeQRF:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def fit(
            self,
            training: pd.DataFrame,
            predictors: list[str],
            targets: list[str],
            *,
            weights: np.ndarray,
        ) -> FakeFitted:
            calls["training"] = training.copy()
            calls["predictors"] = predictors
            calls["targets"] = targets
            calls["weights"] = weights.copy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)

    result = with_us_retirement_contribution_inputs(
        expanded,
        seed=7,
        time_period=2024,
    )

    assert calls["init"] == {"n_estimators": 100, "seed": 7}
    assert len(calls["training"]) == 4
    assert len(calls["test"]) == 4
    person = result.table("person")
    puf = person[person["person_support_channel"] == "puf_tax_detail"]
    # Fake predictions are 1,2,3,4,5. The no-wage rows lose both 401(k)
    # values, and the no-self-employment rows lose the SE-pension value.
    np.testing.assert_allclose(
        puf[list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)].to_numpy(),
        np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 0.0],
                [0.0, 0.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 0.0, 3.0, 4.0, 0.0],
            ]
        ),
    )


def test_signal_gate_rejects_a_default_leaf() -> None:
    result = with_us_retirement_contribution_inputs(
        _frame(),
        seed=0,
        time_period=2024,
    )
    result.table("person")["roth_ira_contributions_desired"] = 0.0

    gate = us_retirement_contributions_signal_gate(result)

    assert not gate.passed
    assert any("roth_ira_contributions_desired" in failure for failure in gate.failures)
