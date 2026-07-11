"""ASEC child-support restoration and retired PUF-half joint QRF treatment."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.child_support as module
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from populace.build.us_runtime.child_support import (
    CHILD_SUPPORT_ARCHIVED_PUF_IMPUTATION_URL,
    CHILD_SUPPORT_ARCHIVED_PUF_OUTPUTS_URL,
    CHILD_SUPPORT_EXPENSE_ARCHIVED_DERIVATION_URL,
    CHILD_SUPPORT_RECEIVED_ARCHIVED_DERIVATION_URL,
    US_CHILD_SUPPORT_OUTPUT_COLUMNS,
    derive_us_child_support_from_manifest,
    impute_us_child_support_to_puf_support_from_manifest,
    us_child_support_signal_gate,
    us_child_support_stage_spec,
    with_us_child_support_inputs,
)
from populace.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)

_RECEIVED, _EXPENSE = US_CHILD_SUPPORT_OUTPUT_COLUMNS


def _person_source() -> pd.DataFrame:
    count = 10
    return pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1, dtype="int64"),
            "person_household_id": np.arange(1, count + 1, dtype="int64") * 10,
            "person_tax_unit_id": np.arange(1, count + 1, dtype="int64") * 100,
            "person_spm_unit_id": np.arange(1, count + 1, dtype="int64") * 1_000,
            "person_family_id": np.arange(1, count + 1, dtype="int64") * 10_000,
            "person_marital_unit_id": (
                np.arange(1, count + 1, dtype="int64") * 100_000
            ),
            "CSP_VAL": [3_600.0, *([0.0] * 9)],
            "CHSP_VAL": [2_400.0, *([0.0] * 9)],
            "WSAL_VAL": np.linspace(10_000.0, 100_000.0, count),
            "SEMP_VAL": np.zeros(count),
            "employment_income_before_lsr": np.linspace(10_000.0, 100_000.0, count),
            "self_employment_income_before_lsr": np.zeros(count),
            "age": np.arange(25, 25 + count),
            "is_female": np.asarray([False, True] * 5),
            "has_esi": np.asarray([True, False] * 5),
            "tax_unit_role_input": ["HEAD"] * count,
            "social_security_retirement": np.zeros(count),
            "social_security_disability": np.zeros(count),
            "social_security_dependents": np.zeros(count),
            "social_security_survivors": np.zeros(count),
        }
    )


def _frame() -> Frame:
    person = _person_source()
    count = len(person)
    ids = {
        "household": person["person_household_id"].to_numpy(),
        "tax_unit": person["person_tax_unit_id"].to_numpy(),
        "spm_unit": person["person_spm_unit_id"].to_numpy(),
        "family": person["person_family_id"].to_numpy(),
        "marital_unit": person["person_marital_unit_id"].to_numpy(),
    }
    tables = {
        entity: pd.DataFrame({f"{entity}_id": values}) for entity, values in ids.items()
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = ["SINGLE"] * count
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.arange(1.0, count + 1.0),
                WeightKind.DESIGN,
            )
        },
    )


def _derive(person: pd.DataFrame) -> pd.DataFrame:
    operation = next(
        operation
        for operation in us_child_support_stage_spec().operations
        if operation.kind == "derive_child_support_inputs"
    )
    return derive_us_child_support_from_manifest(person, operation, None)


def test_archived_sources_are_sha_and_line_pinned() -> None:
    commit = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
    assert commit in CHILD_SUPPORT_RECEIVED_ARCHIVED_DERIVATION_URL
    assert CHILD_SUPPORT_RECEIVED_ARCHIVED_DERIVATION_URL.endswith(
        "datasets/cps/cps.py#L1493-L1496"
    )
    assert commit in CHILD_SUPPORT_EXPENSE_ARCHIVED_DERIVATION_URL
    assert CHILD_SUPPORT_EXPENSE_ARCHIVED_DERIVATION_URL.endswith(
        "datasets/cps/cps.py#L1572-L1574"
    )
    assert commit in CHILD_SUPPORT_ARCHIVED_PUF_OUTPUTS_URL
    assert CHILD_SUPPORT_ARCHIVED_PUF_OUTPUTS_URL.endswith(
        "datasets/cps/extended_cps.py#L135-L194"
    )
    assert commit in CHILD_SUPPORT_ARCHIVED_PUF_IMPUTATION_URL
    assert CHILD_SUPPORT_ARCHIVED_PUF_IMPUTATION_URL.endswith(
        "datasets/cps/extended_cps.py#L639-L745"
    )


def test_stage_manifest_pins_direct_sources_and_one_joint_qrf() -> None:
    spec = us_child_support_stage_spec()

    assert spec.stage == "child_support_inputs"
    assert spec.grain == "person"
    assert tuple(spec.outputs) == US_CHILD_SUPPORT_OUTPUT_COLUMNS
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_child_support_inputs",
        "impute_child_support_to_puf_support",
    ]
    assert spec.operations[1].parameters == {
        "received_source": "CSP_VAL",
        "received_output": _RECEIVED,
        "expense_source": "CHSP_VAL",
        "expense_output": _EXPENSE,
    }
    assert spec.operations[2].parameters == {
        "predictors": [
            "age",
            "is_male",
            "has_esi",
            "tax_unit_is_joint",
            "tax_unit_count_dependents",
            "employment_income",
            "self_employment_income",
            "social_security",
        ],
        "max_train_samples": 5_000,
        "n_estimators": 100,
        "seed_from_build_config": True,
        "weight": "person_weight",
    }


def test_direct_carry_preserves_positive_annual_values_and_topcodes() -> None:
    source = pd.DataFrame(
        {
            "CSP_VAL": [0.0, 99_999.0, 1_500.0, 800.0],
            "CHSP_VAL": [90_000.0, 0.0, 700.0, 800.0],
        }
    )
    original = source.copy(deep=True)

    result = _derive(source)

    assert result[_RECEIVED].tolist() == source["CSP_VAL"].tolist()
    assert result[_EXPENSE].tolist() == source["CHSP_VAL"].tolist()
    assert result.loc[3, [_RECEIVED, _EXPENSE]].tolist() == [800.0, 800.0]
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize("missing", ["CSP_VAL", "CHSP_VAL"])
def test_direct_carry_fails_closed_when_source_is_missing(missing: str) -> None:
    with pytest.raises(SourceRuntimeError, match=missing):
        _derive(_person_source().drop(columns=[missing]))


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("CSP_VAL", np.nan, "nonfinite"),
        ("CHSP_VAL", np.inf, "nonfinite"),
        ("CSP_VAL", -1.0, "negative"),
        ("CHSP_VAL", -1.0, "negative"),
    ],
)
def test_direct_carry_rejects_invalid_source_values(
    column: str,
    bad_value: float,
    message: str,
) -> None:
    person = _person_source()
    person.loc[0, column] = bad_value

    with pytest.raises(SourceRuntimeError, match=message):
        _derive(person)


def test_with_inputs_materializes_person_leaves_without_mutating_source() -> None:
    frame = _frame()
    original = frame.table("person").copy(deep=True)

    result = with_us_child_support_inputs(frame, seed=0, time_period=2024)

    pd.testing.assert_frame_equal(frame.table("person"), original)
    assert result.table("person")[_RECEIVED].tolist() == original["CSP_VAL"].tolist()
    assert result.table("person")[_EXPENSE].tolist() == original["CHSP_VAL"].tolist()
    gate = us_child_support_signal_gate(result)
    assert gate.passed, gate.failures


def test_release_frame_requires_opt_in_to_preserve_valid_existing_surface() -> None:
    materialized = with_us_child_support_inputs(_frame(), seed=0, time_period=2024)
    tables = {
        entity: materialized.table(entity).copy() for entity in materialized.entities
    }
    tables["person"] = tables["person"].drop(columns=["CSP_VAL", "CHSP_VAL"])
    release = Frame(
        tables,
        materialized.schema,
        {
            entity: materialized.weights_for(entity)
            for entity in materialized.weighted_entities
        },
        materialized.strata,
        mass_log=materialized.mass_log,
    )

    with pytest.raises(ValueError, match="cannot heal.*without measured"):
        with_us_child_support_inputs(release, seed=0, time_period=2024)

    assert (
        with_us_child_support_inputs(
            release,
            seed=0,
            time_period=2024,
            allow_existing_without_source=True,
        )
        is release
    )


def test_puf_half_uses_one_joint_qrf_in_archived_target_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = with_us_child_support_inputs(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame) -> pd.DataFrame:
            calls["test"] = test.copy()
            received = np.zeros(len(test), dtype=np.float64)
            expense = np.zeros(len(test), dtype=np.float64)
            received[0] = 7_200.0
            expense[0] = 4_800.0
            return pd.DataFrame(
                {_RECEIVED: received, _EXPENSE: expense},
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
            calls["fit_count"] = int(calls.get("fit_count", 0)) + 1
            calls["training"] = training.copy()
            calls["predictors"] = predictors
            calls["targets"] = targets
            calls["weights"] = weights.copy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)

    result = with_us_child_support_inputs(expanded, seed=7, time_period=2024)

    assert calls["fit_count"] == 1
    assert calls["init"] == {"n_estimators": 100, "seed": 7}
    assert calls["targets"] == [_RECEIVED, _EXPENSE]
    assert calls["predictors"] == list(
        us_child_support_stage_spec().operations[2].parameters["predictors"]
    )
    training = calls["training"]
    assert isinstance(training, pd.DataFrame)
    assert len(training) == 10
    weights = calls["weights"]
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (10,)

    person = result.table("person")
    asec = person[person["person_support_channel"] == "asec"]
    puf = person[person["person_support_channel"] == "puf_tax_detail"]
    assert asec[_RECEIVED].tolist() == _person_source()["CSP_VAL"].tolist()
    assert asec[_EXPENSE].tolist() == _person_source()["CHSP_VAL"].tolist()
    assert puf[_RECEIVED].tolist() == [7_200.0, *([0.0] * 9)]
    assert puf[_EXPENSE].tolist() == [4_800.0, *([0.0] * 9)]
    gate = us_child_support_signal_gate(result)
    assert gate.passed, gate.failures


def test_puf_qrf_caps_training_at_5000_and_keeps_weights_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = us_child_support_stage_spec().operations[2]
    predictors = tuple(operation.parameters["predictors"])
    asec_rows = 5_010
    puf_rows = 2
    rows = asec_rows + puf_rows
    frame = pd.DataFrame(
        {
            "person_support_channel": ["asec"] * asec_rows
            + ["puf_tax_detail"] * puf_rows,
            "person_weight": np.arange(1.0, rows + 1.0),
            _RECEIVED: np.tile([0.0, 500.0], (rows + 1) // 2)[:rows],
            _EXPENSE: np.tile([300.0, 0.0], (rows + 1) // 2)[:rows],
            **{
                f"child_support_predictor_{predictor}": np.arange(
                    rows, dtype=np.float64
                )
                for predictor in predictors
            },
        }
    )
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame) -> pd.DataFrame:
            calls["test_rows"] = len(test)
            return pd.DataFrame(
                {
                    _RECEIVED: np.zeros(len(test)),
                    _EXPENSE: np.zeros(len(test)),
                },
                index=test.index,
            )

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
            calls["training_index"] = training.index.to_numpy()
            calls["weights"] = weights.copy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)
    context = SourceRuntimeContext(
        config=SourceRuntimeConfig(seed=19, target_year=2024),
        tables={},
    )

    impute_us_child_support_to_puf_support_from_manifest(frame, operation, context)

    assert calls["init"] == {"n_estimators": 100, "seed": 19}
    assert calls["training_rows"] == 5_000
    assert calls["test_rows"] == 2
    np.testing.assert_allclose(
        calls["weights"],
        frame.loc[calls["training_index"], "person_weight"].to_numpy(),
    )


def test_signal_gate_rejects_missing_default_invalid_and_dead_puf_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = with_us_child_support_inputs(_frame(), seed=0, time_period=2024)
    assert us_child_support_signal_gate(valid).passed

    candidates: list[Frame] = []
    for column, replacement in (
        (_RECEIVED, None),
        (_RECEIVED, np.zeros(10)),
        (_EXPENSE, np.zeros(10)),
        (_RECEIVED, np.asarray([-1.0, *([0.0] * 9)])),
        (_EXPENSE, np.asarray([np.nan, *([0.0] * 9)])),
    ):
        candidate = with_us_child_support_inputs(_frame(), seed=0, time_period=2024)
        if replacement is None:
            candidate.table("person").drop(columns=[column], inplace=True)
        else:
            candidate.table("person")[column] = replacement
        candidates.append(candidate)
    assert all(
        not us_child_support_signal_gate(candidate).passed for candidate in candidates
    )

    class ZeroFitted:
        def predict(self, test: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    _RECEIVED: np.zeros(len(test)),
                    _EXPENSE: np.zeros(len(test)),
                },
                index=test.index,
            )

    class ZeroQRF:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(self, *args: object, **kwargs: object) -> ZeroFitted:
            return ZeroFitted()

    monkeypatch.setattr(module, "QRF", ZeroQRF)
    expanded = clone_us_frame_for_puf_support(valid)
    dead_puf = with_us_child_support_inputs(expanded, seed=0, time_period=2024)
    gate = us_child_support_signal_gate(dead_puf)
    assert not gate.passed
    assert any("puf_tax_detail nonzero share" in failure for failure in gate.failures)


@requires_us
def test_policyengine_us_contract_is_two_person_year_input_leaves() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    for name in US_CHILD_SUPPORT_OUTPUT_COLUMNS:
        variable = variables[name]
        assert variable.is_input_variable()
        assert variable.entity.key == "person"
        assert str(variable.definition_period).lower() == "year"
        assert variable.default_value == 0


@requires_us
def test_policyengine_us_graph_uses_positive_annual_received_and_expense() -> None:
    from policyengine_us import Simulation

    def situation(received: float, expense: float) -> dict[str, object]:
        return {
            "people": {
                "adult": {
                    "age": {"2024": 35},
                    _RECEIVED: {"2024": received},
                    _EXPENSE: {"2024": expense},
                }
            },
            "spm_units": {"spm_unit": {"members": ["adult"]}},
            "households": {
                "household": {
                    "members": ["adult"],
                    "state_code": {"2024": "CA"},
                }
            },
        }

    baseline = Simulation(situation=situation(0.0, 0.0))
    active = Simulation(situation=situation(3_600.0, 2_400.0))

    assert (
        active.calculate("spm_unit_benefits", 2024)[0]
        - baseline.calculate("spm_unit_benefits", 2024)[0]
    ) == pytest.approx(3_600.0)
    assert (
        active.calculate("spm_unit_spm_expenses", 2024)[0]
        - baseline.calculate("spm_unit_spm_expenses", 2024)[0]
    ) == pytest.approx(2_400.0)
    assert active.calculate("snap_child_support_deduction", "2024-01")[
        0
    ] == pytest.approx(200.0)
