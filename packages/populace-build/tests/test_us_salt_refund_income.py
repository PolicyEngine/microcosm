"""Contracts for the IRS PUF state-and-local-tax-refund income input."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    puf_tax_unit_donor_from_arrays,
    support_channel_column,
)
from populace.build.us_runtime import puf_support as puf_support_module
from populace.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from populace.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_salt_refund_income_from_source,
    derive_puf_policyengine_variables,
)
from populace.build.us_runtime.reform_coverage_smoke import _build_reform
from populace.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
    load_release_input_coverage_manifest,
    us_release_reform_coverage_probes,
)
from populace.build.us_runtime.salt_refund_income import (
    SALT_REFUND_ARCHIVED_DERIVATION_URL,
    SALT_REFUND_ARCHIVED_EXPORT_URL,
    SALT_REFUND_ARCHIVED_IMPUTATION_URL,
    SALT_REFUND_ARCHIVED_PERSON_ALLOCATION_URL,
    SALT_REFUND_ARCHIVED_PUF_ARTIFACT_URL,
    derive_us_salt_refund_income_from_puf,
    us_salt_refund_income_signal_gate,
    us_salt_refund_income_stage_spec,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)


class _ResolvedWeights:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _PersonFrame:
    def __init__(
        self,
        person: pd.DataFrame,
        weights: np.ndarray | None = None,
    ) -> None:
        self._person = person
        self._weights = np.ones(len(person)) if weights is None else weights

    def table(self, entity: str) -> pd.DataFrame:
        assert entity == "person"
        return self._person

    def resolve_weights(self, entity: str) -> _ResolvedWeights:
        assert entity == "person"
        return _ResolvedWeights(np.asarray(self._weights, dtype=np.float64))


def _joint_support_frame() -> Frame:
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.asarray([1, 2], dtype="int64"),
                "person_household_id": np.asarray([10, 10], dtype="int64"),
                "person_tax_unit_id": np.asarray([100, 100], dtype="int64"),
                "person_spm_unit_id": np.asarray([1_000, 1_000], dtype="int64"),
                "person_family_id": np.asarray([10_000, 10_000], dtype="int64"),
                "person_marital_unit_id": np.asarray([100_000, 100_000], dtype="int64"),
                "employment_income_before_lsr": [50_000.0, 20_000.0],
                "self_employment_income_before_lsr": [0.0, 40_000.0],
            }
        ),
        "household": pd.DataFrame({"household_id": np.asarray([10], dtype="int64")}),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([100], dtype="int64"),
                "filing_status_input": ["JOINT"],
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([1_000], dtype="int64")}),
        "family": pd.DataFrame({"family_id": np.asarray([10_000], dtype="int64")}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.asarray([100_000], dtype="int64")}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray([100.0]), WeightKind.DESIGN)},
    )


def test_archived_coordinates_pin_derivation_export_and_imputation() -> None:
    commit = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"

    assert commit in SALT_REFUND_ARCHIVED_DERIVATION_URL
    assert SALT_REFUND_ARCHIVED_DERIVATION_URL.endswith("puf.py#L707")
    assert commit in SALT_REFUND_ARCHIVED_EXPORT_URL
    assert SALT_REFUND_ARCHIVED_EXPORT_URL.endswith("puf.py#L804-L850")
    assert commit in SALT_REFUND_ARCHIVED_IMPUTATION_URL
    assert SALT_REFUND_ARCHIVED_IMPUTATION_URL.endswith("puf_impute.py#L90-L198")
    assert SALT_REFUND_ARCHIVED_PERSON_ALLOCATION_URL.endswith("puf.py#L1513-L1601")
    assert SALT_REFUND_ARCHIVED_PUF_ARTIFACT_URL.endswith("puf.py#L1655-L1660")


def test_archived_e00700_mapping_is_an_exact_carry() -> None:
    source = pd.DataFrame({"E00700": [0.0, 125.5, 9_000.0]})

    result = derive_us_salt_refund_income_from_puf(source)

    assert "salt_refund_income" not in source.columns
    assert result["salt_refund_income"].tolist() == [0.0, 125.5, 9_000.0]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame({"other": [1.0]}), "requires source column"),
        (pd.DataFrame({"E00700": ["not numeric"]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E00700": [np.inf]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E00700": [-1.0]}), "negative value"),
    ],
)
def test_source_derivation_fails_closed(
    source: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_us_salt_refund_income_from_puf(source)


def test_shared_puf_derivation_and_post_disaggregation_reconciliation() -> None:
    source = pd.DataFrame(
        {
            "E00600": [0.0, 0.0, 0.0],
            "E00650": [0.0, 0.0, 0.0],
            "E00700": [0.0, 3_000.0, 7_500.0],
        }
    )

    derived = derive_puf_policyengine_variables(
        source,
        salt_refund_income_source="E00700",
    )
    derived["salt_refund_income"] = 999.0
    result = _reconcile_puf_salt_refund_income_from_source(derived)

    assert result["salt_refund_income"].tolist() == [0.0, 3_000.0, 7_500.0]


def test_shared_puf_stage_declares_exact_source_output_and_artifact() -> None:
    stage = us_salt_refund_income_stage_spec()
    operation = next(
        operation
        for operation in stage.operations
        if operation.kind == "derive_puf_policyengine_variables"
    )

    assert operation.parameters["salt_refund_income_source"] == "E00700"
    assert operation.parameters["salt_refund_income_output"] == "salt_refund_income"
    assert "salt_refund_income" in stage.outputs
    assert "salt_refund_income" in stage.nonnegative_outputs
    assert any(
        "E00700" in str(artifact.get("columns", "")) + str(artifact.get("locator", ""))
        for artifact in stage.artifacts
    )
    assert any(
        "irs-soi-puf/1.8.0/puf_2024.h5" in str(artifact.get("locator"))
        for artifact in stage.artifacts
    )


def test_puf_support_keeps_salt_refunds_nonnegative_sparse_and_unsplit() -> None:
    output = "salt_refund_income"

    assert output in puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert output in puf_support_module._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
    assert output in puf_support_module._PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS
    assert output not in puf_support_module._PERSON_OUTPUT_DISTRIBUTION_BASIS


def test_processed_puf_people_aggregate_to_one_tax_unit_refund() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 20, 20],
            "salt_refund_income": [0.0, 100.0, 200.0],
        },
        person_outputs=("salt_refund_income",),
        tax_unit_outputs=(),
    )

    assert donor["salt_refund_income"].tolist() == [0.0, 300.0]


def test_puf_imputation_preserves_tax_unit_total_without_inventing_split() -> None:
    expanded = clone_us_frame_for_puf_support(_joint_support_frame())
    donor = pd.DataFrame(
        {
            "filing_status_code": [2.0, 2.0],
            "tax_unit_person_count": [2.0, 2.0],
            "salt_refund_income": [900.0, 900.0],
            "weight": [1.0, 1.0],
        }
    )

    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=(
            "puf_predictor_filing_status_code",
            "puf_predictor_tax_unit_person_count",
        ),
        person_outputs=("salt_refund_income",),
        tax_unit_outputs=(),
        n_estimators=4,
        seed=0,
    )
    person = imputed.table("person")
    channel = support_channel_column("person")
    asec = person[person[channel] == BASE_ASEC_SUPPORT_CHANNEL]
    puf = person[person[channel] == PUF_TAX_DETAIL_SUPPORT_CHANNEL].sort_values(
        "person_id"
    )

    assert asec["salt_refund_income"].tolist() == [0.0, 0.0]
    np.testing.assert_allclose(puf["salt_refund_income"].to_numpy(), [900.0, 0.0])
    assert puf["salt_refund_income"].sum() == pytest.approx(900.0)


def test_signal_gate_accepts_source_aligned_nondefault_values() -> None:
    values = np.zeros(2_000)
    values[np.arange(1_000, 1_100)] = np.linspace(100.0, 10_000.0, 100)
    channels = np.asarray(["asec"] * 1_000 + ["puf_tax_detail"] * 1_000)
    frame = _PersonFrame(
        pd.DataFrame(
            {
                "salt_refund_income": values,
                "person_support_channel": channels,
            }
        )
    )

    result = us_salt_refund_income_signal_gate(frame)  # type: ignore[arg-type]

    assert result.passed, result.failures
    assert result.details["positive_share"] == pytest.approx(0.05)
    assert result.details["channels"]["asec"]["weighted_total"] == 0.0
    assert result.details["channels"]["puf_tax_detail"]["positive_share"] == (
        pytest.approx(0.10)
    )


@pytest.mark.parametrize(
    "person",
    [
        pd.DataFrame({"other": [0.0, 1.0]}),
        pd.DataFrame({"salt_refund_income": [0.0, 0.0]}),
        pd.DataFrame({"salt_refund_income": [0.0, -1.0]}),
        pd.DataFrame({"salt_refund_income": [0.0, np.nan]}),
        pd.DataFrame(
            {
                "salt_refund_income": [10.0, 0.0],
                "person_support_channel": ["asec", "puf_tax_detail"],
            }
        ),
    ],
)
def test_signal_gate_rejects_missing_default_invalid_or_asec_signal(
    person: pd.DataFrame,
) -> None:
    result = us_salt_refund_income_signal_gate(  # type: ignore[arg-type]
        _PersonFrame(person)
    )

    assert not result.passed


def test_release_wiring_keeps_restored_input_hard_required() -> None:
    output = "salt_refund_income"

    assert output in US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS
    assert output in RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
    assert output in load_release_input_coverage_manifest().required_columns


def test_shipped_neutralization_probe_binds_only_through_salt_refunds() -> None:
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "salt_refund_income_neutralization"
    )

    assert probe.parameter_changes == {}
    assert probe.neutralized_variable == "salt_refund_income"
    assert probe.binding_inputs == ("salt_refund_income",)
    assert probe.period == 2024
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "negative"
    assert probe.budget_measure == "state_income_tax"
    assert probe.min_abs_effect == 1_000_000.0


@requires_us
def test_policyengine_us_contract_and_state_tax_binding() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    variable = CountryTaxBenefitSystem().variables["salt_refund_income"]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0

    year = 2024
    adult = {
        "age": {year: 40},
        "employment_income": {year: 100_000},
        "salt_refund_income": {year: 10_000},
    }
    entities = {
        "tax_units": {"unit": {"members": ["adult"]}},
        "families": {"family": {"members": ["adult"]}},
        "spm_units": {"spm": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {year: "SC"},
            }
        },
        "marital_units": {"marital": {"members": ["adult"]}},
    }
    baseline = Simulation(situation={"people": {"adult": adult}, **entities})
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "salt_refund_income_neutralization"
    )
    neutralized = Simulation(
        situation={"people": {"adult": adult}, **entities},
        reform=_build_reform(probe),
    )

    assert baseline.calculate("sc_subtractions", year)[0] == pytest.approx(10_000)
    assert neutralized.calculate("sc_subtractions", year)[0] == 0
    assert (
        baseline.calculate("state_income_tax", year)[0]
        < neutralized.calculate("state_income_tax", year)[0]
    )
