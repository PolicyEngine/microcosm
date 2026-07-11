"""ASEC disability-benefit restoration and PUF-half QRF treatment."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.disability_benefits as module
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from populace.build.us_runtime.disability_benefits import (
    DISABILITY_BENEFITS_ARCHIVED_DERIVATION_URL,
    DISABILITY_BENEFITS_ARCHIVED_PUF_IMPUTATION_URL,
    DISABILITY_BENEFITS_ARCHIVED_PUF_OUTPUTS_URL,
    DISABILITY_BENEFITS_ARCHIVED_SOURCE_COLUMNS_URL,
    US_DISABILITY_BENEFITS_OUTPUT_COLUMNS,
    US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS,
    derive_us_disability_benefits_from_manifest,
    impute_us_disability_benefits_to_puf_support_from_manifest,
    us_disability_benefits_signal_gate,
    us_disability_benefits_stage_spec,
    with_us_disability_benefits,
)
from populace.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from populace.build.us_runtime.release_input_coverage import (
    us_release_reform_coverage_probes,
)
from populace.build.us_runtime.source_runtime import us_source_operation_handlers
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)

_OUTPUT = US_DISABILITY_BENEFITS_OUTPUT_COLUMNS[0]
_PREDICTORS = (
    "age",
    "is_male",
    "has_esi",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "employment_income",
    "self_employment_income",
    "social_security",
)


def _person_source() -> pd.DataFrame:
    count = 100
    first_amount = np.zeros(count)
    first_code = np.zeros(count)
    second_amount = np.zeros(count)
    second_code = np.zeros(count)
    first_amount[0] = 3_600.0
    first_code[0] = 2.0
    # A workers'-compensation amount in the other slot must remain excluded.
    second_amount[0] = 1_200.0
    second_code[0] = 1.0
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
            "DIS_VAL1": first_amount,
            "DIS_SC1": first_code,
            "DIS_VAL2": second_amount,
            "DIS_SC2": second_code,
            "WSAL_VAL": np.linspace(0.0, 99_000.0, count),
            "SEMP_VAL": np.zeros(count),
            "employment_income_before_lsr": np.linspace(0.0, 99_000.0, count),
            "self_employment_income_before_lsr": np.zeros(count),
            "age": np.arange(20, 20 + count),
            "is_female": np.tile([False, True], count // 2),
            "has_esi": np.tile([True, False], count // 2),
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
                np.ones(count, dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _derive(person: pd.DataFrame) -> pd.DataFrame:
    operation = next(
        operation
        for operation in us_disability_benefits_stage_spec().operations
        if operation.kind == "derive_disability_benefits"
    )
    return derive_us_disability_benefits_from_manifest(person, operation, None)


def test_archived_sources_are_sha_and_line_pinned_and_sources_available() -> None:
    commit = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
    urls = (
        DISABILITY_BENEFITS_ARCHIVED_DERIVATION_URL,
        DISABILITY_BENEFITS_ARCHIVED_SOURCE_COLUMNS_URL,
        DISABILITY_BENEFITS_ARCHIVED_PUF_OUTPUTS_URL,
        DISABILITY_BENEFITS_ARCHIVED_PUF_IMPUTATION_URL,
    )
    assert all(commit in url for url in urls)
    assert DISABILITY_BENEFITS_ARCHIVED_DERIVATION_URL.endswith(
        "datasets/cps/cps.py#L1561-L1571"
    )
    assert DISABILITY_BENEFITS_ARCHIVED_SOURCE_COLUMNS_URL.endswith(
        "datasets/cps/census_cps.py#L306-L381"
    )
    assert DISABILITY_BENEFITS_ARCHIVED_PUF_OUTPUTS_URL.endswith(
        "datasets/cps/extended_cps.py#L135-L194"
    )
    assert DISABILITY_BENEFITS_ARCHIVED_PUF_IMPUTATION_URL.endswith(
        "datasets/cps/extended_cps.py#L639-L745"
    )
    assert US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS == (
        "DIS_VAL1",
        "DIS_SC1",
        "DIS_VAL2",
        "DIS_SC2",
    )


def test_stage_manifest_pins_two_slot_formula_and_one_output_qrf() -> None:
    spec = us_disability_benefits_stage_spec()

    assert spec.stage == "disability_benefits_input"
    assert spec.survey == "Census CPS ASEC"
    assert spec.grain == "person"
    assert tuple(spec.outputs) == (_OUTPUT,)
    assert tuple(spec.nonnegative_outputs) == (_OUTPUT,)
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_disability_benefits",
        "impute_disability_benefits_to_puf_support",
    ]
    assert spec.operations[0].parameters == {
        "table": "person",
        "weight": "person_weight",
    }
    assert spec.operations[1].parameters == {
        "first_amount_source": "DIS_VAL1",
        "first_code_source": "DIS_SC1",
        "second_amount_source": "DIS_VAL2",
        "second_code_source": "DIS_SC2",
        "workers_compensation_code": 1,
        "output": _OUTPUT,
    }
    assert spec.operations[2].parameters == {
        "predictors": list(_PREDICTORS),
        "max_train_samples": 5_000,
        "n_estimators": 100,
        "seed_from_build_config": True,
        "weight": "person_weight",
    }

    handlers = us_source_operation_handlers()
    assert (
        handlers["derive_disability_benefits"]
        is derive_us_disability_benefits_from_manifest
    )
    assert (
        handlers["impute_disability_benefits_to_puf_support"]
        is impute_us_disability_benefits_to_puf_support_from_manifest
    )


def test_direct_formula_excludes_only_code_one_and_preserves_topcodes() -> None:
    source = pd.DataFrame(
        {
            "DIS_VAL1": [100.0, 99_999.0, 500.0, 800.0, 300.0],
            "DIS_SC1": [1, 2, 1, 2, 0],
            "DIS_VAL2": [200.0, 90_000.0, 700.0, 900.0, 400.0],
            "DIS_SC2": [1, 0, 2, 1, 3],
        }
    )
    original = source.copy(deep=True)

    result = _derive(source)

    assert result[_OUTPUT].tolist() == [0.0, 189_999.0, 700.0, 800.0, 700.0]
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize("missing", US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS)
def test_direct_formula_fails_closed_when_source_is_missing(missing: str) -> None:
    with pytest.raises(SourceRuntimeError, match=missing):
        _derive(_person_source().drop(columns=[missing]))


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("DIS_VAL1", np.nan, "nonfinite"),
        ("DIS_SC1", np.inf, "nonfinite"),
        ("DIS_VAL2", np.inf, "nonfinite"),
        ("DIS_SC2", np.nan, "nonfinite"),
        ("DIS_VAL1", -1.0, "negative"),
        ("DIS_VAL2", -1.0, "negative"),
    ],
)
def test_direct_formula_rejects_invalid_sources(
    column: str,
    bad_value: float,
    message: str,
) -> None:
    person = _person_source()
    person.loc[0, column] = bad_value

    with pytest.raises(SourceRuntimeError, match=message):
        _derive(person)


def test_with_inputs_materializes_exact_asec_values_without_mutation() -> None:
    frame = _frame()
    original = frame.table("person").copy(deep=True)

    result = with_us_disability_benefits(frame, seed=0, time_period=2024)

    pd.testing.assert_frame_equal(frame.table("person"), original)
    expected = np.where(original["DIS_SC1"] != 1, original["DIS_VAL1"], 0) + np.where(
        original["DIS_SC2"] != 1,
        original["DIS_VAL2"],
        0,
    )
    np.testing.assert_allclose(result.table("person")[_OUTPUT], expected)
    gate = us_disability_benefits_signal_gate(result)
    assert gate.passed, gate.failures


def test_release_requires_opt_in_to_preserve_valid_existing_surface() -> None:
    materialized = with_us_disability_benefits(_frame(), seed=0, time_period=2024)
    tables = {
        entity: materialized.table(entity).copy() for entity in materialized.entities
    }
    tables["person"] = tables["person"].drop(
        columns=list(US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS)
    )
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
        with_us_disability_benefits(release, seed=0, time_period=2024)

    assert (
        with_us_disability_benefits(
            release,
            seed=0,
            time_period=2024,
            allow_existing_without_source=True,
        )
        is release
    )


def test_puf_half_uses_one_output_qrf_and_preserves_asec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = with_us_disability_benefits(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)
    original = expanded.table("person").copy(deep=True)
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame) -> pd.DataFrame:
            calls["test"] = test.copy()
            predicted = np.zeros(len(test), dtype=np.float64)
            predicted[0] = 7_200.0
            return pd.DataFrame({_OUTPUT: predicted}, index=test.index)

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

    result = with_us_disability_benefits(expanded, seed=7, time_period=2024)

    assert calls["fit_count"] == 1
    assert calls["init"] == {"n_estimators": 100, "seed": 7}
    assert calls["predictors"] == list(_PREDICTORS)
    assert calls["targets"] == [_OUTPUT]
    training = calls["training"]
    assert isinstance(training, pd.DataFrame)
    assert list(training.columns) == [*_PREDICTORS, _OUTPUT]
    asec_mask = original["person_support_channel"] == "asec"
    expected_weights = expanded.resolve_weights("person").values[asec_mask]
    np.testing.assert_allclose(calls["weights"], expected_weights)

    person = result.table("person")
    asec = person[person["person_support_channel"] == "asec"]
    puf = person[person["person_support_channel"] == "puf_tax_detail"]
    assert asec[_OUTPUT].tolist() == [3_600.0, *([0.0] * 99)]
    assert puf[_OUTPUT].tolist() == [7_200.0, *([0.0] * 99)]
    pd.testing.assert_frame_equal(expanded.table("person"), original)
    gate = us_disability_benefits_signal_gate(result)
    assert gate.passed, gate.failures


def test_puf_qrf_caps_training_at_5000_and_keeps_weights_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = us_disability_benefits_stage_spec().operations[2]
    asec_rows = 5_010
    puf_rows = 2
    rows = asec_rows + puf_rows
    frame = pd.DataFrame(
        {
            "person_support_channel": ["asec"] * asec_rows
            + ["puf_tax_detail"] * puf_rows,
            "person_weight": np.arange(1.0, rows + 1.0),
            _OUTPUT: np.tile([0.0, 500.0], (rows + 1) // 2)[:rows],
            **{
                f"disability_benefits_predictor_{predictor}": np.arange(
                    rows, dtype=np.float64
                )
                for predictor in _PREDICTORS
            },
        }
    )
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame) -> pd.DataFrame:
            calls["test_rows"] = len(test)
            return pd.DataFrame({_OUTPUT: np.zeros(len(test))}, index=test.index)

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
            calls["training_rows"] = len(training)
            calls["training_index"] = training.index.to_numpy()
            calls["weights"] = weights.copy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)
    context = SourceRuntimeContext(
        config=SourceRuntimeConfig(seed=19, target_year=2024),
        tables={},
    )

    impute_us_disability_benefits_to_puf_support_from_manifest(
        frame,
        operation,
        context,
    )

    assert calls["init"] == {"n_estimators": 100, "seed": 19}
    assert calls["training_rows"] == 5_000
    assert calls["test_rows"] == 2
    np.testing.assert_allclose(
        calls["weights"],
        frame.loc[calls["training_index"], "person_weight"].to_numpy(),
    )


def test_signal_gate_rejects_missing_default_and_invalid_surfaces() -> None:
    valid = with_us_disability_benefits(_frame(), seed=0, time_period=2024)
    assert us_disability_benefits_signal_gate(valid).passed

    candidates: list[Frame] = []
    for replacement in (
        None,
        np.zeros(100),
        np.asarray([-1.0, *([0.0] * 99)]),
        np.asarray([np.nan, *([0.0] * 99)]),
    ):
        candidate = with_us_disability_benefits(_frame(), seed=0, time_period=2024)
        if replacement is None:
            candidate.table("person").drop(columns=[_OUTPUT], inplace=True)
        else:
            candidate.table("person")[_OUTPUT] = replacement
        candidates.append(candidate)

    assert all(
        not us_disability_benefits_signal_gate(candidate).passed
        for candidate in candidates
    )


@pytest.mark.parametrize("dead_channel", ["asec", "puf_tax_detail"])
def test_signal_gate_rejects_either_dead_support_channel(dead_channel: str) -> None:
    direct = with_us_disability_benefits(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)
    assert us_disability_benefits_signal_gate(expanded).passed

    channel = expanded.table("person")["person_support_channel"]
    expanded.table("person").loc[channel == dead_channel, _OUTPUT] = 0.0
    gate = us_disability_benefits_signal_gate(expanded)

    assert not gate.passed
    assert any(dead_channel in failure for failure in gate.failures)


@requires_us
def test_policyengine_us_1_764_6_contract_and_positive_annual_behavior() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "1.764.6"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0

    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                _OUTPUT: {"2024": 6_000.0},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
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
    simulation = Simulation(situation=situation)

    assert simulation.calculate(_OUTPUT, 2024)[0] == pytest.approx(6_000.0)
    assert simulation.calculate(_OUTPUT, "2024-01")[0] == pytest.approx(500.0)
    assert simulation.calculate("snap_unearned_income", "2024-01")[0] == pytest.approx(
        500.0
    )


@requires_us
def test_shipped_snap_exclusion_probe_binds_with_positive_sign() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "disability_benefits_snap_exclusion"
    )
    reform = Reform.from_dict(dict(probe.parameter_changes), country_id="us")
    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": 12_000.0},
                _OUTPUT: {"2024": 6_000.0},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
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
    baseline = Simulation(situation=situation)
    reformed = Simulation(
        tax_benefit_system=CountryTaxBenefitSystem(reform=(reform,)),
        situation=situation,
    )

    effect = reformed.calculate("snap", 2024)[0] - baseline.calculate("snap", 2024)[0]
    assert effect > 1_000.0
