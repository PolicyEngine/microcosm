"""ASEC childcare restoration and PUF-half QRF treatment."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.childcare as module
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.us_runtime.childcare import (
    US_CHILDCARE_OUTPUT_COLUMNS,
    derive_us_childcare_from_manifest,
    impute_us_childcare_to_puf_support_from_manifest,
    us_childcare_signal_gate,
    us_childcare_stage_spec,
    with_us_childcare_inputs,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)

_OUTPUT = US_CHILDCARE_OUTPUT_COLUMNS[0]


def _person_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype="int64"),
            "person_household_id": [10, 10, 20, 30, 40, 50],
            "person_tax_unit_id": [100, 100, 200, 300, 400, 500],
            "person_spm_unit_id": [1_000, 1_000, 2_000, 3_000, 4_000, 5_000],
            "person_family_id": [10_000, 10_000, 20_000, 30_000, 40_000, 50_000],
            "person_marital_unit_id": [
                100_000,
                100_000,
                200_000,
                300_000,
                400_000,
                500_000,
            ],
            "SPM_CHILDCAREXPNS": [800.0, 800.0, 0.0, 1_200.0, 0.0, 0.0],
            "WSAL_VAL": [50_000.0, 20_000.0, 0.0, 35_000.0, 10_000.0, 0.0],
            "SEMP_VAL": [0.0, 0.0, 20_000.0, 0.0, 5_000.0, 0.0],
            "employment_income_before_lsr": [
                50_000.0,
                20_000.0,
                0.0,
                35_000.0,
                10_000.0,
                0.0,
            ],
            "self_employment_income_before_lsr": [
                0.0,
                0.0,
                20_000.0,
                0.0,
                5_000.0,
                0.0,
            ],
            "age": [35, 33, 45, 29, 55, 70],
            "is_female": [False, True, True, False, True, False],
            "has_esi": [True, True, False, True, False, False],
            "tax_unit_role_input": [
                "HEAD",
                "SPOUSE",
                "HEAD",
                "HEAD",
                "HEAD",
                "HEAD",
            ],
            "social_security_retirement": [0.0, 0.0, 0.0, 0.0, 0.0, 15_000.0],
            "social_security_disability": [0.0] * 6,
            "social_security_dependents": [0.0] * 6,
            "social_security_survivors": [0.0] * 6,
        }
    )


def _frame() -> Frame:
    person = _person_source()
    ids = {
        "household": [10, 20, 30, 40, 50],
        "tax_unit": [100, 200, 300, 400, 500],
        "spm_unit": [1_000, 2_000, 3_000, 4_000, 5_000],
        "family": [10_000, 20_000, 30_000, 40_000, 50_000],
        "marital_unit": [100_000, 200_000, 300_000, 400_000, 500_000],
    }
    tables = {
        entity: pd.DataFrame({f"{entity}_id": np.asarray(values, dtype="int64")})
        for entity, values in ids.items()
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = [
        "JOINT",
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
                np.ones(5, dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    operation = next(
        operation
        for operation in us_childcare_stage_spec().operations
        if operation.kind == "derive_childcare_inputs"
    )
    return derive_us_childcare_from_manifest(frame, operation, None)


def test_stage_manifest_pins_source_qrf_and_first_person_reduction() -> None:
    spec = us_childcare_stage_spec()

    assert spec.stage == "childcare_inputs"
    assert spec.grain == "person"
    assert tuple(spec.outputs) == US_CHILDCARE_OUTPUT_COLUMNS
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_childcare_inputs",
        "impute_childcare_to_puf_support",
    ]
    impute = spec.operations[2]
    assert impute.parameters["max_train_samples"] == 5_000
    assert impute.parameters["n_estimators"] == 100
    assert impute.parameters["weight"] == "person_weight"
    assert impute.parameters["reduction"] == "value_from_first_person"


def test_direct_source_is_exact_and_replicated() -> None:
    result = _derive(_person_source())

    assert result[_OUTPUT].tolist() == [800.0, 800.0, 0.0, 1_200.0, 0.0, 0.0]


def test_direct_source_fails_closed_when_missing() -> None:
    with pytest.raises(SourceRuntimeError, match="SPM_CHILDCAREXPNS"):
        _derive(_person_source().drop(columns=["SPM_CHILDCAREXPNS"]))


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [(np.nan, "nonfinite"), (-1.0, "negative")],
)
def test_direct_source_rejects_invalid_values(
    bad_value: float,
    message: str,
) -> None:
    person = _person_source()
    person.loc[[0, 1], "SPM_CHILDCAREXPNS"] = bad_value

    with pytest.raises(SourceRuntimeError, match=message):
        _derive(person)


def test_direct_source_rejects_inconsistent_unit_replicas() -> None:
    person = _person_source()
    person.loc[1, "SPM_CHILDCAREXPNS"] = 700.0

    with pytest.raises(SourceRuntimeError, match="disagrees within replicated"):
        _derive(person)


def test_with_inputs_materializes_spm_unit_values() -> None:
    result = with_us_childcare_inputs(_frame(), seed=0, time_period=2024)

    assert result.table("spm_unit")[_OUTPUT].tolist() == [
        800.0,
        0.0,
        1_200.0,
        0.0,
        0.0,
    ]
    gate = us_childcare_signal_gate(result)
    assert gate.passed, gate.failures


def test_existing_output_is_recomputed_from_measured_source() -> None:
    frame = _frame()
    frame.table("spm_unit")[_OUTPUT] = [111.0, 222.0, 333.0, 444.0, 555.0]

    result = with_us_childcare_inputs(frame, seed=0, time_period=2024)

    assert result.table("spm_unit")[_OUTPUT].tolist() == [
        800.0,
        0.0,
        1_200.0,
        0.0,
        0.0,
    ]


def test_release_frame_without_raw_source_preserves_valid_output() -> None:
    materialized = with_us_childcare_inputs(_frame(), seed=0, time_period=2024)
    tables = {
        entity: materialized.table(entity).copy() for entity in materialized.entities
    }
    tables["person"] = tables["person"].drop(columns=["SPM_CHILDCAREXPNS"])
    release = Frame(
        tables,
        materialized.schema,
        {
            entity: materialized.weights_for(entity)
            for entity in materialized.weighted_entities
        },
        materialized.strata,
    )

    with pytest.raises(ValueError, match="cannot heal.*without measured"):
        with_us_childcare_inputs(release, seed=0, time_period=2024)

    result = with_us_childcare_inputs(
        release,
        seed=0,
        time_period=2024,
        allow_existing_without_source=True,
    )

    assert result is release


def test_puf_half_uses_qrf_and_first_person_spm_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = with_us_childcare_inputs(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            calls["test"] = test.copy()
            return pd.DataFrame(
                {_OUTPUT: np.arange(100.0, 100.0 + len(test))},
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

    result = with_us_childcare_inputs(expanded, seed=7, time_period=2024)

    assert calls["init"] == {"n_estimators": 100, "seed": 7}
    assert len(calls["training"]) == 6
    assert len(calls["test"]) == 6
    spm = result.table("spm_unit")
    asec = spm[spm["spm_unit_support_channel"] == "asec"]
    puf = spm[spm["spm_unit_support_channel"] == "puf_tax_detail"]
    assert asec[_OUTPUT].tolist() == [800.0, 0.0, 1_200.0, 0.0, 0.0]
    # The first two PUF people share an SPM unit, so prediction 101 is ignored.
    assert puf[_OUTPUT].tolist() == [100.0, 102.0, 103.0, 104.0, 105.0]


def test_puf_qrf_caps_training_at_5000_and_keeps_aligned_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = next(
        operation
        for operation in us_childcare_stage_spec().operations
        if operation.kind == "impute_childcare_to_puf_support"
    )
    predictors = tuple(operation.parameters["predictors"])
    asec_rows = 5_010
    puf_rows = 2
    frame = pd.DataFrame(
        {
            "person_support_channel": ["asec"] * asec_rows
            + ["puf_tax_detail"] * puf_rows,
            "person_weight": np.arange(1.0, asec_rows + puf_rows + 1.0),
            _OUTPUT: np.tile([0.0, 500.0], (asec_rows + puf_rows + 1) // 2)[
                : asec_rows + puf_rows
            ],
            **{
                f"childcare_predictor_{predictor}": np.arange(
                    asec_rows + puf_rows, dtype=np.float64
                )
                for predictor in predictors
            },
        }
    )
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            calls["test_rows"] = len(test)
            return pd.DataFrame({_OUTPUT: np.zeros(len(test))}, index=test.index)

    class FakeQRF:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def fit(
            self,
            training: pd.DataFrame,
            predictor_names: list[str],
            targets: list[str],
            *,
            weights: np.ndarray,
        ) -> FakeFitted:
            calls["training_rows"] = len(training)
            calls["weights"] = weights.copy()
            calls["training_index"] = training.index.to_numpy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)
    context = SourceRuntimeContext(
        config=SourceRuntimeConfig(seed=19, target_year=2024),
        tables={},
    )

    impute_us_childcare_to_puf_support_from_manifest(frame, operation, context)

    assert calls["init"] == {"n_estimators": 100, "seed": 19}
    assert calls["training_rows"] == 5_000
    assert calls["test_rows"] == 2
    np.testing.assert_allclose(
        calls["weights"],
        frame.loc[calls["training_index"], "person_weight"].to_numpy(),
    )


def test_signal_gate_rejects_missing_default_and_invalid_surfaces() -> None:
    frame = with_us_childcare_inputs(_frame(), seed=0, time_period=2024)
    for values in (
        None,
        [0.0] * 5,
        [800.0, 0.0, -1.0, 0.0, 0.0],
        [800.0, 0.0, np.nan, 0.0, 0.0],
    ):
        candidate = _frame()
        if values is not None:
            candidate.table("spm_unit")[_OUTPUT] = values
        gate = us_childcare_signal_gate(candidate)
        assert not gate.passed

    assert us_childcare_signal_gate(frame).passed


@requires_us
def test_policyengine_us_contract_is_spm_unit_year_input_leaf() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    variable = variables[_OUTPUT]

    assert variable.is_input_variable()
    assert variable.entity.key == "spm_unit"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0
    assert not variables["cdcc_relevant_expenses"].is_input_variable()
