"""ASEC retirement-distribution restoration and PUF-half QRF treatment."""

from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.retirement_distributions as module
from populace.build.source_manifest import SourceStageSpec
from populace.build.source_runtime import SourceRuntimeError
from populace.build.us_runtime.asec_pool import load_asec_h5_tables
from populace.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from populace.build.us_runtime.release_input_coverage import (
    us_release_reform_coverage_probes,
)
from populace.build.us_runtime.retirement_distributions import (
    RETIREMENT_DISTRIBUTIONS_ARCHIVED_DERIVATION_URL,
    RETIREMENT_DISTRIBUTIONS_ARCHIVED_PARAMETERS_URL,
    US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS,
    US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS,
    derive_us_retirement_distributions_from_manifest,
    us_retirement_distributions_signal_gate,
    us_retirement_distributions_stage_spec,
    with_us_retirement_distribution_inputs,
)
from populace.build.us_runtime.source_runtime import us_source_operation_handlers
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

ROOT = Path(__file__).resolve().parents[3]
policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)

_OUTPUTS = US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
_PUF_QRF_OUTPUTS = (
    "taxable_401k_distributions",
    "taxable_403b_distributions",
    "keogh_distributions",
    "taxable_sep_distributions",
)
_SLOT_SUFFIXES = ("1", "2", "1_YNG", "2_YNG")


def _person_source() -> pd.DataFrame:
    count = 8
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1, dtype="int64"),
            "person_household_id": np.arange(10, 10 + count, dtype="int64"),
            "person_tax_unit_id": np.arange(100, 100 + count, dtype="int64"),
            "person_spm_unit_id": np.arange(1_000, 1_000 + count, dtype="int64"),
            "person_family_id": np.arange(10_000, 10_000 + count, dtype="int64"),
            "person_marital_unit_id": np.arange(
                100_000, 100_000 + count, dtype="int64"
            ),
            "age": [65, 66, 67, 68, 69, 70, 71, 40],
            "is_female": [False, True, False, True, False, True, False, True],
            "has_esi": [False] * count,
            "tax_unit_role_input": ["HEAD"] * count,
            "employment_income_before_lsr": [50_000.0] * count,
            "self_employment_income_before_lsr": [0.0] * count,
            "social_security_retirement": [10_000.0] * count,
            "social_security_disability": [0.0] * count,
            "social_security_dependents": [0.0] * count,
            "social_security_survivors": [0.0] * count,
        }
    )
    for suffix in _SLOT_SUFFIXES:
        person[f"DST_SC{suffix}"] = 0
        person[f"DST_VAL{suffix}"] = 0.0
    person["DST_SC1"] = np.arange(1, count + 1) % 8
    person["DST_VAL1"] = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 0.0]
    # A second 401(k) slot proves that repeated account codes sum by person.
    person.loc[0, ["DST_SC2", "DST_VAL2"]] = [1, 50.0]
    return person


def _frame() -> Frame:
    person = _person_source()
    identifiers = {
        "household": person["person_household_id"].to_numpy(),
        "tax_unit": person["person_tax_unit_id"].to_numpy(),
        "spm_unit": person["person_spm_unit_id"].to_numpy(),
        "family": person["person_family_id"].to_numpy(),
        "marital_unit": person["person_marital_unit_id"].to_numpy(),
    }
    tables = {
        entity: pd.DataFrame({f"{entity}_id": values})
        for entity, values in identifiers.items()
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = ["SINGLE"] * len(person)
    # Total = 10,000. Rare outputs receive smaller weights so the fixture
    # reproduces plausible population shares while retaining every account.
    weights = np.asarray([1_000, 100, 100, 1_000, 1, 100, 100, 7_599], dtype=float)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.DESIGN)},
    )


def _operation(kind: str = "derive_retirement_distributions"):
    return next(
        operation
        for operation in us_retirement_distributions_stage_spec().operations
        if operation.kind == kind
    )


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    return derive_us_retirement_distributions_from_manifest(
        frame,
        _operation(),
        None,
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_stage_manifest_pins_direct_mapping_and_retired_qrf() -> None:
    spec = us_retirement_distributions_stage_spec()

    assert spec.stage == "retirement_distributions"
    assert spec.grain == "person"
    assert tuple(spec.outputs) == _OUTPUTS
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_retirement_distributions",
        "impute_retirement_distributions_to_puf_support",
    ]
    assert spec.operations[1].parameters["output_by_account_code"] == {
        "1": "taxable_401k_distributions",
        "2": "taxable_403b_distributions",
        "3": "tax_exempt_ira_distributions",
        "4": "taxable_ira_distributions",
        "5": "keogh_distributions",
        "6": "taxable_sep_distributions",
    }
    impute = spec.operations[2].parameters
    assert impute["predictors"] == [
        "age",
        "is_male",
        "has_esi",
        "tax_unit_is_joint",
        "tax_unit_count_dependents",
        "employment_income",
        "self_employment_income",
        "social_security",
    ]
    assert impute["max_train_samples"] == 5_000
    assert impute["n_estimators"] == 100
    assert RETIREMENT_DISTRIBUTIONS_ARCHIVED_DERIVATION_URL.endswith(
        "cps.py#L1448-L1481"
    )
    assert RETIREMENT_DISTRIBUTIONS_ARCHIVED_PARAMETERS_URL.endswith(
        "imputation_parameters.yaml#L10-L15"
    )


def test_handler_is_registered() -> None:
    handlers = us_source_operation_handlers()
    assert (
        handlers["derive_retirement_distributions"]
        is derive_us_retirement_distributions_from_manifest
    )
    assert (
        handlers["impute_retirement_distributions_to_puf_support"]
        is module.impute_us_retirement_distributions_to_puf_support_from_manifest
    )


def test_direct_mapping_sums_all_four_measured_slots() -> None:
    person = _person_source()
    person.loc[0, ["DST_SC1_YNG", "DST_VAL1_YNG"]] = [1, 25.0]
    person.loc[0, ["DST_SC2_YNG", "DST_VAL2_YNG"]] = [1, 75.0]

    result = _derive(person)

    np.testing.assert_array_equal(
        result[list(_OUTPUTS)].to_numpy(),
        np.asarray(
            [
                [250.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 200.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 300.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 400.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 500.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 600.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


@pytest.mark.parametrize("missing", US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS)
def test_direct_mapping_fails_closed_without_each_source_column(missing: str) -> None:
    with pytest.raises(SourceRuntimeError, match=missing):
        _derive(_person_source().drop(columns=[missing]))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("DST_SC1", np.nan, "account codes"),
        ("DST_SC1", 8, "account codes"),
        ("DST_SC1", 1.5, "account codes"),
        ("DST_VAL1", np.nan, "nonnegative amounts"),
        ("DST_VAL1", -1, "nonnegative amounts"),
    ],
)
def test_direct_mapping_rejects_invalid_source_values(
    column: str,
    value: float,
    message: str,
) -> None:
    person = _person_source()
    person[column] = person[column].astype(float)
    person.loc[0, column] = value

    with pytest.raises(SourceRuntimeError, match=message):
        _derive(person)


def test_direct_mapping_rejects_amount_on_not_in_universe_slot() -> None:
    person = _person_source()
    person.loc[7, "DST_VAL1"] = 1.0
    with pytest.raises(SourceRuntimeError, match="NIU code 0"):
        _derive(person)


def test_wrong_operation_and_missing_table_fail_closed() -> None:
    wrong = SourceStageSpec.from_mapping(
        {
            "stage": "test",
            "survey": "test",
            "source": "https://example.com",
            "grain": "person",
            "operations": [{"kind": "derive"}],
            "outputs": list(_OUTPUTS),
        }
    ).operations[0]
    with pytest.raises(SourceRuntimeError, match="unexpected operation"):
        derive_us_retirement_distributions_from_manifest(_person_source(), wrong, None)
    with pytest.raises(SourceRuntimeError, match="person table"):
        derive_us_retirement_distributions_from_manifest(None, _operation(), None)


def test_frame_integration_gate_and_idempotence() -> None:
    result = with_us_retirement_distribution_inputs(_frame(), seed=0, time_period=2024)
    gate = us_retirement_distributions_signal_gate(result)

    assert gate.passed, gate.failures
    assert all(value == 0 for value in gate.details["source_mismatches"].values())
    assert (
        with_us_retirement_distribution_inputs(result, seed=0, time_period=2024)
        is result
    )


def test_puf_half_uses_qrf_and_asec_half_remains_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = with_us_retirement_distribution_inputs(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)
    expanded_person = expanded.table("person")
    puf_mask = expanded_person["person_support_channel"] == "puf_tax_detail"
    # The PUF tax-detail stage owns this leaf; the retirement stage must not
    # replace it with either the copied ASEC value or a QRF prediction.
    puf_indices = expanded_person.index[puf_mask]
    expanded_person.loc[puf_mask, "taxable_ira_distributions"] = 0.0
    expanded_person.loc[puf_indices[3], "taxable_ira_distributions"] = 999.0
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            calls["test"] = test.copy()
            return pd.DataFrame(
                0.0,
                index=test.index,
                columns=list(_PUF_QRF_OUTPUTS),
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
    result = with_us_retirement_distribution_inputs(
        expanded,
        seed=7,
        time_period=2024,
        force_puf_imputation=True,
    )

    assert calls["init"] == {"n_estimators": 100, "seed": 7}
    assert len(calls["training"]) == len(direct.table("person"))
    assert len(calls["test"]) == len(direct.table("person"))
    assert calls["targets"] == list(_PUF_QRF_OUTPUTS)
    person = result.table("person")
    asec = person[person["person_support_channel"] == "asec"]
    puf = person[person["person_support_channel"] == "puf_tax_detail"]
    np.testing.assert_array_equal(
        asec[list(_OUTPUTS)].to_numpy(),
        direct.table("person")[list(_OUTPUTS)].to_numpy(),
    )
    assert not puf[list(_PUF_QRF_OUTPUTS)].to_numpy().any()
    np.testing.assert_array_equal(
        puf["tax_exempt_ira_distributions"].to_numpy(),
        direct.table("person")["tax_exempt_ira_distributions"].to_numpy(),
    )
    np.testing.assert_array_equal(
        puf["taxable_ira_distributions"].to_numpy(),
        [0.0, 0.0, 0.0, 999.0, 0.0, 0.0, 0.0, 0.0],
    )
    gate = us_retirement_distributions_signal_gate(result)
    assert gate.passed, gate.failures


def test_completed_puf_surface_survives_narrowed_support_without_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = with_us_retirement_distribution_inputs(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)

    class ZeroFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            return pd.DataFrame(
                0.0,
                index=test.index,
                columns=list(_PUF_QRF_OUTPUTS),
            )

    class ZeroQRF:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(self, *args: object, **kwargs: object) -> ZeroFitted:
            return ZeroFitted()

    monkeypatch.setattr(module, "QRF", ZeroQRF)
    completed = with_us_retirement_distribution_inputs(
        expanded,
        seed=7,
        time_period=2024,
        force_puf_imputation=True,
    )

    # Model frozen-support recovery by removing one all-zero PUF household.
    # Frame.select deliberately preserves the surviving person index, so the
    # selected frame also covers the non-RangeIndex path that changed the
    # historical 5,000-row donor sample.
    person = completed.table("person")
    puf_zero = person["person_support_channel"].eq("puf_tax_detail") & ~person[
        list(_PUF_QRF_OUTPUTS)
    ].any(axis=1)
    drop_index = person.index[puf_zero][0]
    selected = completed.select(person.index != drop_index)
    before = us_retirement_distributions_signal_gate(selected)
    assert before.passed, before.failures
    before_share = before.details["nonzero_shares"]["keogh_distributions"]
    assert 0.0000001 <= before_share <= 0.005

    class UnexpectedQRF:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("a completed retirement surface must not be refit")

    monkeypatch.setattr(module, "QRF", UnexpectedQRF)
    result = with_us_retirement_distribution_inputs(
        selected,
        seed=7,
        time_period=2024,
    )

    assert result is selected
    after = us_retirement_distributions_signal_gate(result)
    assert after.passed, after.failures
    assert after.details["nonzero_shares"]["keogh_distributions"] == before_share


def test_gate_rejects_a_default_or_source_divergent_leaf() -> None:
    result = with_us_retirement_distribution_inputs(_frame(), seed=0, time_period=2024)
    result.table("person")["keogh_distributions"] = 0.0
    gate = us_retirement_distributions_signal_gate(result)
    assert not gate.passed
    assert any("keogh_distributions" in failure for failure in gate.failures)


def test_all_sha_locked_asec_artifacts_carry_exact_source_signal() -> None:
    expected = {
        "census_cps_2022.h5": (
            "7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e",
            4,
            44_000.0,
        ),
        "census_cps_2023.h5": (
            "cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88",
            5,
            130_040.0,
        ),
        "census_cps_2024.h5": (
            "ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d",
            4,
            166_600.0,
        ),
    }
    summary = json.loads(
        (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )
    paths = {
        Path(item["path"]).name: Path(item["path"])
        for item in summary["base_source"]["sources"]
    }
    if not all(paths[name].is_file() for name in expected):
        pytest.skip("SHA-locked ASEC artifacts are not mounted")

    for name, (digest, positives, total) in expected.items():
        path = paths[name]
        assert _sha256(path) == digest
        person = load_asec_h5_tables(path)["person"]
        assert set(US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS) <= set(
            person.columns
        )
        derived = _derive(person)["keogh_distributions"]
        assert int((derived > 0).sum()) == positives
        assert float(derived.sum()) == total


@requires_us
def test_policyengine_1_764_6_contract_and_live_binding() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "1.764.6"
    variables = CountryTaxBenefitSystem().variables
    for name in _OUTPUTS:
        variable = variables[name]
        assert variable.is_input_variable()
        assert variable.entity.key == "person"
        assert variable.value_type is float
        assert variable.default_value == 0
        assert str(variable.definition_period).lower() == "year"

    situation = {
        "people": {
            "adult": {
                "age": {"2024": 65},
                "employment_income": {"2024": 60_000},
                "keogh_distributions": {"2024": 10_000},
            }
        },
        "tax_units": {"tax_unit": {"members": ["adult"]}},
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
    }

    class NeutralizeKeogh(Reform):
        def apply(self) -> None:
            self.neutralize_variable("keogh_distributions")

    baseline = Simulation(situation=situation)
    neutralized = Simulation(situation=situation, reform=NeutralizeKeogh)
    assert baseline.calculate("taxable_retirement_distributions", 2024)[0] == 10_000
    assert neutralized.calculate("taxable_retirement_distributions", 2024)[0] == 0
    for measure in ("irs_gross_income", "adjusted_gross_income"):
        assert baseline.calculate(measure, 2024)[0] == pytest.approx(
            neutralized.calculate(measure, 2024)[0] + 10_000
        )
    assert (
        baseline.calculate("income_tax", 2024)[0]
        > neutralized.calculate("income_tax", 2024)[0]
    )


def test_shipped_keogh_neutralization_probe() -> None:
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "keogh_distribution_neutralization"
    )
    assert probe.neutralized_variable == "keogh_distributions"
    assert probe.binding_inputs == ("keogh_distributions",)
    assert probe.budget_measure == "income_tax"
    assert probe.period == 2024
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "positive"
    assert probe.min_abs_effect >= 1_000_000.0
