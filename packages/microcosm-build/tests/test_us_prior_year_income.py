"""Archived adjacent-ASEC prior-year-income restoration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.prior_year_income as module
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.prior_year_income import (
    PRIOR_YEAR_INCOME_ARCHIVED_DERIVATION_URL,
    PRIOR_YEAR_INCOME_ARCHIVED_FINALIZER_URL,
    PRIOR_YEAR_INCOME_ARCHIVED_FORMULA_OUTPUT_URL,
    PRIOR_YEAR_INCOME_ARCHIVED_PUF_IMPUTATION_URL,
    PRIOR_YEAR_INCOME_ARCHIVED_PUF_OUTPUTS_URL,
    PRIOR_YEAR_INCOME_ARCHIVED_PUF_SPLICE_URL,
    US_PRIOR_YEAR_INCOME_NONCONSTANT_PERSON_COLUMNS,
    US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS,
    US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS,
    derive_us_prior_year_income_from_manifest,
    impute_us_prior_year_income_to_puf_support_from_manifest,
    us_prior_year_income_signal_gate,
    us_prior_year_income_source_reconciliation_gate,
    us_prior_year_income_stage_spec,
    with_us_prior_year_income_inputs,
)
from microcosm.build.us_runtime.puf_support import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
)
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
    load_release_input_coverage_manifest,
    us_release_reform_coverage_probes,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)


def _frame(person: pd.DataFrame) -> Frame:
    person = person.reset_index(drop=True).copy()
    n = len(person)
    ids = np.arange(1, n + 1, dtype=np.int64)
    defaults: dict[str, object] = {
        "person_id": ids,
        "person_household_id": ids,
        "person_tax_unit_id": ids + 100,
        "person_spm_unit_id": ids + 200,
        "person_family_id": ids + 300,
        "person_marital_unit_id": ids + 400,
        "tax_unit_role_input": np.full(n, "HEAD", dtype=object),
        "age": np.linspace(25, 65, n),
        "is_female": ids % 2 == 0,
        "has_esi": ids % 3 == 0,
        "employment_income_before_lsr": np.zeros(n),
        "self_employment_income_before_lsr": np.zeros(n),
        "social_security_retirement": np.zeros(n),
        "social_security_disability": np.zeros(n),
        "social_security_dependents": np.zeros(n),
        "social_security_survivors": np.zeros(n),
    }
    for column, values in defaults.items():
        if column not in person:
            person[column] = values
    person["employment_income_before_lsr"] = pd.to_numeric(
        person.get("WSAL_VAL", person["employment_income_before_lsr"]),
        errors="coerce",
    ).replace({-1: 0, -9999: 0})
    person["self_employment_income_before_lsr"] = pd.to_numeric(
        person.get("SEMP_VAL", person["self_employment_income_before_lsr"]),
        errors="coerce",
    ).replace({-1: 0, -9999: 0})
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": ids, "state_fips": np.resize([6, 36], n)}
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": ids + 100,
                "filing_status_input": np.resize(["SINGLE", "JOINT"], n),
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids + 200}),
        "family": pd.DataFrame({"family_id": ids + 300}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids + 400}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.linspace(1.0, 2.0, n), WeightKind.DESIGN)},
    )


def _source_frame() -> Frame:
    return _frame(
        pd.DataFrame(
            {
                "source_year": [2022, 2023, 2024, 2023, 2024, 2023, 2024, 2024],
                "PERIDNUM": ["A", "A", "A", "B", "B", "C", "C", "D"],
                "WSAL_VAL": [100, 200, 300, 10, 400, 300, 500, 600],
                "SEMP_VAL": [-20, 30, 40, 20, -5, -9999, 50, 60],
                "I_ERNVAL": [0, 0, 0, 1, 0, 0, 0, 0],
                "I_SEVAL": [0, 0, 0, 0, 0, 0, 0, 0],
            }
        )
    )


def test_stage_manifest_pins_archived_join_signedness_and_puf_qrf() -> None:
    spec = us_prior_year_income_stage_spec()

    assert spec.stage == "prior_year_income"
    assert spec.outputs == US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS
    assert spec.nonnegative_outputs == ("employment_income_last_year",)
    operations = {operation.kind: operation for operation in spec.operations}
    assert tuple(operations) == (
        "read_table",
        "derive_prior_year_income",
        "impute_prior_year_income_to_puf_support",
    )
    derive = operations["derive_prior_year_income"].parameters
    assert derive["person_id"] == "PERIDNUM"
    assert derive["prior_year_offset"] == -1
    assert derive["employment_allocation_flag"] == "I_ERNVAL"
    assert derive["self_employment_allocation_flag"] == "I_SEVAL"
    assert derive["unallocated_flag"] == 0
    assert derive["sentinels"] == [-1, -9999]
    assert derive["fallback_to_current"] is True
    assert derive["no_prior_artifact"] == "leave_defaults"
    qrf = operations["impute_prior_year_income_to_puf_support"].parameters
    assert qrf["predictors"] == [
        "age",
        "is_male",
        "has_esi",
        "tax_unit_is_joint",
        "tax_unit_count_dependents",
        "employment_income",
        "self_employment_income",
        "social_security",
    ]
    assert qrf["outputs"] == [
        "employment_income_last_year",
        "self_employment_income_last_year",
    ]
    assert qrf["max_train_samples"] == 5_000
    assert qrf["n_estimators"] == 100
    assert qrf["weight"] == "person_weight"
    assert all(
        url.startswith(
            "https://github.com/PolicyEngine/policyengine-" + "us-data/blob/"
            "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
        )
        for url in (
            PRIOR_YEAR_INCOME_ARCHIVED_DERIVATION_URL,
            PRIOR_YEAR_INCOME_ARCHIVED_PUF_OUTPUTS_URL,
            PRIOR_YEAR_INCOME_ARCHIVED_PUF_IMPUTATION_URL,
            PRIOR_YEAR_INCOME_ARCHIVED_PUF_SPLICE_URL,
            PRIOR_YEAR_INCOME_ARCHIVED_FORMULA_OUTPUT_URL,
            PRIOR_YEAR_INCOME_ARCHIVED_FINALIZER_URL,
        )
    )


def test_adjacent_join_matches_flags_sentinels_fallback_and_signed_losses() -> None:
    result = with_us_prior_year_income_inputs(
        _source_frame(), seed=0, time_period=2024
    ).table("person")

    assert result["previous_year_income_available"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert result["employment_income_last_year"].tolist() == [
        0,
        100,
        200,
        10,
        400,
        300,
        500,
        600,
    ]
    assert result["self_employment_income_last_year"].tolist() == [
        0,
        -20,
        30,
        20,
        -5,
        0,
        50,
        60,
    ]


def test_valid_zero_prior_values_are_available_not_fallbacks() -> None:
    frame = _frame(
        pd.DataFrame(
            {
                "source_year": [2023, 2024],
                "PERIDNUM": ["0000000000000000000001"] * 2,
                "WSAL_VAL": [0, 50_000],
                "SEMP_VAL": [0, 10_000],
                "I_ERNVAL": [0, 0],
                "I_SEVAL": [0, 0],
            }
        )
    )

    person = with_us_prior_year_income_inputs(frame, seed=0, time_period=2024).table(
        "person"
    )

    assert person["previous_year_income_available"].tolist() == [False, True]
    assert person["employment_income_last_year"].tolist() == [0, 0]
    assert person["self_employment_income_last_year"].tolist() == [0, 0]


def test_existing_default_outputs_rederive_when_raw_sources_remain() -> None:
    frame = _source_frame()
    person = frame.table("person").copy()
    person["employment_income_last_year"] = 0.0
    person["self_employment_income_last_year"] = 0.0
    person["previous_year_income_available"] = False
    stale = module._replace_person_table(frame, person)

    result = with_us_prior_year_income_inputs(stale, seed=0, time_period=2024).table(
        "person"
    )

    assert result["previous_year_income_available"].any()
    assert result["self_employment_income_last_year"].tolist() != [0.0] * len(result)


def test_join_rejects_duplicate_source_year_person_key() -> None:
    frame = _source_frame()
    person = frame.table("person").copy()
    duplicate = person.iloc[[0]].copy()
    duplicate["person_id"] = 999
    duplicate["person_household_id"] = 999
    duplicate["person_tax_unit_id"] = 999
    duplicate["person_spm_unit_id"] = 999
    duplicate["person_family_id"] = 999
    duplicate["person_marital_unit_id"] = 999
    operation = us_prior_year_income_stage_spec().operations[1]

    with pytest.raises(SourceRuntimeError, match="duplicate key"):
        derive_us_prior_year_income_from_manifest(
            pd.concat([person, duplicate], ignore_index=True), operation, None
        )


def test_join_refuses_missing_allocation_source() -> None:
    person = _source_frame().table("person").drop(columns=["I_SEVAL"])
    operation = us_prior_year_income_stage_spec().operations[1]

    with pytest.raises(SourceRuntimeError, match="I_SEVAL"):
        derive_us_prior_year_income_from_manifest(person, operation, None)


class _Fitted:
    def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
        n = len(test)
        self_employment = np.zeros(n, dtype=np.float64)
        self_employment[:2] = [125.0, -25.0]
        return pd.DataFrame(
            {
                "employment_income_last_year": np.arange(n) + 1_000.0,
                "self_employment_income_last_year": self_employment,
            },
            index=test.index,
        )


class _QRF:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def fit(
        self,
        training: pd.DataFrame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: np.ndarray,
    ) -> _Fitted:
        self.calls.append(
            {
                "kwargs": self.kwargs,
                "training": training.copy(),
                "predictors": predictors,
                "targets": targets,
                "weights": weights.copy(),
            }
        )
        return _Fitted()


def test_puf_support_joint_qrf_is_weighted_signed_and_drops_formula_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _QRF)
    _QRF.calls.clear()
    direct = with_us_prior_year_income_inputs(_source_frame(), seed=7, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)

    result = with_us_prior_year_income_inputs(expanded, seed=7, time_period=2024)
    person = result.table("person")
    assert "employment_income_last_year" not in person
    channel = person["person_support_channel"].astype(str)
    puf_values = person.loc[
        channel == PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        "self_employment_income_last_year",
    ].to_numpy()
    assert puf_values[:2].tolist() == [125.0, -25.0]
    asec_values = person.loc[
        channel == BASE_ASEC_SUPPORT_CHANNEL,
        "self_employment_income_last_year",
    ].to_numpy()
    assert asec_values.tolist() == [0, -20, 30, 20, -5, 0, 50, 60]

    assert len(_QRF.calls) == 1
    call = _QRF.calls[0]
    assert call["targets"] == [
        "employment_income_last_year",
        "self_employment_income_last_year",
    ]
    assert call["predictors"] == list(module._PUF_PREDICTORS)
    assert np.all(np.asarray(call["weights"]) > 0)
    assert call["kwargs"] == {"n_estimators": 100, "seed": 7}


def test_puf_qrf_rejects_zero_weight_capped_training_sample() -> None:
    n_asec = 5_001
    sampled = (
        pd.DataFrame(index=np.arange(n_asec)).sample(n=5_000, random_state=0).index
    )
    omitted = int(next(iter(set(range(n_asec)) - set(sampled))))
    weights = np.zeros(n_asec + 1, dtype=np.float64)
    weights[omitted] = 1.0
    person = pd.DataFrame(
        {
            "person_support_channel": [BASE_ASEC_SUPPORT_CHANNEL] * n_asec
            + [PUF_TAX_DETAIL_SUPPORT_CHANNEL],
            "person_weight": weights,
            "employment_income_last_year": np.ones(n_asec + 1),
            "self_employment_income_last_year": np.ones(n_asec + 1),
        }
    )
    for predictor in module._PUF_PREDICTORS:
        person[module._PUF_PREDICTOR_PREFIX + predictor] = 1.0
    operation = us_prior_year_income_stage_spec().operations[2]

    with pytest.raises(SourceRuntimeError, match="sampled training weights"):
        impute_us_prior_year_income_to_puf_support_from_manifest(
            person, operation, None
        )


def _signal_frame() -> Frame:
    n = 100
    return _frame(
        pd.DataFrame(
            {
                "self_employment_income_last_year": np.where(
                    np.arange(n) < 4,
                    np.resize([10_000.0, -2_000.0], n),
                    0.0,
                ),
                "previous_year_income_available": np.arange(n) < 20,
            }
        )
    )


def test_signal_gate_accepts_signed_source_signal_and_rejects_defaults() -> None:
    passing = us_prior_year_income_signal_gate(_signal_frame())
    assert passing.passed, passing.failures
    assert passing.details["self_employment_income_last_year_negative_rows"] > 0

    frame = _signal_frame()
    person = frame.table("person").copy()
    person["previous_year_income_available"] = False
    failing = us_prior_year_income_signal_gate(
        module._replace_person_table(frame, person)
    )
    assert not failing.passed
    assert "availability" in " ".join(failing.failures)


def test_source_reconciliation_detects_plausible_but_wrong_asec_carry() -> None:
    derived = with_us_prior_year_income_inputs(
        _source_frame(), seed=0, time_period=2024
    )
    passing = us_prior_year_income_source_reconciliation_gate(derived)
    assert passing.passed, passing.failures

    person = derived.table("person").copy()
    person.loc[1, "self_employment_income_last_year"] = 30.0
    corrupted = module._replace_person_table(derived, person)
    failing = us_prior_year_income_source_reconciliation_gate(corrupted)
    assert not failing.passed
    assert failing.details["mismatch_counts"] == {
        "employment_income_last_year": 0,
        "self_employment_income_last_year": 1,
        "previous_year_income_available": 0,
    }


def test_release_contract_promotes_both_persisted_inputs_without_wage_formula() -> None:
    manifest = load_release_input_coverage_manifest()
    for column in US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS:
        assert column in RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
        assert column in US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions
    assert US_PRIOR_YEAR_INCOME_NONCONSTANT_PERSON_COLUMNS == (
        "self_employment_income_last_year",
        "previous_year_income_available",
    )
    assert "employment_income_last_year" not in manifest.required_columns

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "prior_year_self_employment_neutralization"
    )
    assert probe.neutralized_variable == "self_employment_income_last_year"
    assert probe.parameter_changes == {}
    assert probe.binding_inputs == ("self_employment_income_last_year",)
    assert probe.budget_measure == "tax_unit_earned_income_last_year"
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "positive"


@requires_us
def test_policyengine_17646_input_and_dependency_contract() -> None:
    engine = PolicyEngineUSEngine()
    assert set(US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS) <= set(engine.variables())
    assert engine.formula_owned_outputs(US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS) == {
        "employment_income_last_year"
    }
    system = engine._tax_benefit_system()
    assert system.variables["earned_income_last_year"].adds == [
        "employment_income_last_year",
        "self_employment_income_last_year",
    ]
    assert system.variables["tax_unit_earned_income_last_year"].entity.key == "tax_unit"


@requires_us
def test_export_ready_support_persists_inputs_and_excludes_formula_owned_wage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(module, "QRF", _QRF)
    direct = with_us_prior_year_income_inputs(_source_frame(), seed=0, time_period=2024)
    result = with_us_prior_year_income_inputs(
        clone_us_frame_for_puf_support(direct), seed=0, time_period=2024
    )
    output = tmp_path / "prior_year_income.h5"

    PolicyEngineUSEngine().write_dataset(result, output, period=2024)

    with pd.HDFStore(output, mode="r") as store:
        person = store["person"]
    assert "self_employment_income_last_year" in person
    assert "previous_year_income_available" in person
    assert "employment_income_last_year" not in person
