"""Contracts for the IRS PUF Form 4952 elected-investment-income input."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    puf_tax_unit_donor_from_arrays,
    support_channel_column,
)
from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime.form_4952 import (
    FORM_4952_ARCHIVED_DERIVATION_URL,
    FORM_4952_ARCHIVED_EXPORT_URL,
    FORM_4952_ARCHIVED_IMPUTATION_URL,
    FORM_4952_ARCHIVED_PERSON_ALLOCATION_URL,
    FORM_4952_ARCHIVED_PUF_ARTIFACT_URL,
    derive_us_form_4952_election_from_puf,
    us_form_4952_election_signal_gate,
    us_form_4952_election_stage_spec,
)
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_form_4952_election_from_source,
    derive_puf_policyengine_variables,
)
from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
    load_release_input_coverage_manifest,
    us_release_reform_coverage_probes,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

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

    assert commit in FORM_4952_ARCHIVED_DERIVATION_URL
    assert FORM_4952_ARCHIVED_DERIVATION_URL.endswith("puf.py#L708")
    assert commit in FORM_4952_ARCHIVED_EXPORT_URL
    assert FORM_4952_ARCHIVED_EXPORT_URL.endswith("puf.py#L804-L850")
    assert commit in FORM_4952_ARCHIVED_IMPUTATION_URL
    assert FORM_4952_ARCHIVED_IMPUTATION_URL.endswith("puf_impute.py#L90-L198")
    assert FORM_4952_ARCHIVED_PERSON_ALLOCATION_URL.endswith("puf.py#L477-L546")
    assert FORM_4952_ARCHIVED_PUF_ARTIFACT_URL.endswith("puf.py#L1655-L1660")


def test_archived_e58990_mapping_is_an_exact_carry() -> None:
    source = pd.DataFrame({"E58990": [0.0, 125.5, 9_000.0]})

    result = derive_us_form_4952_election_from_puf(source)

    assert "investment_income_elected_form_4952" not in source.columns
    assert result["investment_income_elected_form_4952"].tolist() == [
        0.0,
        125.5,
        9_000.0,
    ]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame({"other": [1.0]}), "requires source column"),
        (pd.DataFrame({"E58990": ["not numeric"]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E58990": [np.inf]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E58990": [-1.0]}), "negative value"),
    ],
)
def test_source_derivation_fails_closed(
    source: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_us_form_4952_election_from_puf(source)


def test_shared_puf_derivation_and_post_disaggregation_reconciliation() -> None:
    source = pd.DataFrame(
        {
            "E00600": [0.0, 0.0, 0.0],
            "E00650": [0.0, 0.0, 0.0],
            "E58990": [0.0, 3_000.0, 7_500.0],
        }
    )

    derived = derive_puf_policyengine_variables(
        source,
        investment_income_elected_form_4952_source="E58990",
    )
    derived["investment_income_elected_form_4952"] = 999.0
    result = _reconcile_puf_form_4952_election_from_source(derived)

    assert result["investment_income_elected_form_4952"].tolist() == [
        0.0,
        3_000.0,
        7_500.0,
    ]


def test_shared_puf_stage_declares_exact_source_output_and_artifact() -> None:
    stage = us_form_4952_election_stage_spec()
    operation = next(
        operation
        for operation in stage.operations
        if operation.kind == "derive_puf_policyengine_variables"
    )

    assert (
        operation.parameters["investment_income_elected_form_4952_source"] == "E58990"
    )
    assert (
        operation.parameters["investment_income_elected_form_4952_output"]
        == "investment_income_elected_form_4952"
    )
    assert "investment_income_elected_form_4952" in stage.outputs
    assert "investment_income_elected_form_4952" in stage.nonnegative_outputs
    assert any("E58990" in str(artifact.get("locator")) for artifact in stage.artifacts)
    assert any(
        "irs-soi-puf/1.8.0/puf_2024.h5" in str(artifact.get("locator"))
        for artifact in stage.artifacts
    )


def test_puf_support_keeps_form_4952_nonnegative_and_sparse() -> None:
    output = "investment_income_elected_form_4952"

    assert output in puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert output in puf_support_module._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
    assert output in puf_support_module._PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS
    assert output not in puf_support_module._PERSON_OUTPUT_DISTRIBUTION_BASIS


def test_processed_puf_people_aggregate_to_one_tax_unit_election() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 20, 20],
            "investment_income_elected_form_4952": [0.0, 100.0, 200.0],
        },
        person_outputs=("investment_income_elected_form_4952",),
        tax_unit_outputs=(),
    )

    assert donor["investment_income_elected_form_4952"].tolist() == [0.0, 300.0]


def test_puf_imputation_preserves_tax_unit_total_without_inventing_split() -> None:
    expanded = clone_us_frame_for_puf_support(_joint_support_frame())
    donor = pd.DataFrame(
        {
            "filing_status_code": [2.0, 2.0],
            "tax_unit_person_count": [2.0, 2.0],
            "investment_income_elected_form_4952": [900.0, 900.0],
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
        person_outputs=("investment_income_elected_form_4952",),
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

    assert asec["investment_income_elected_form_4952"].tolist() == [0.0, 0.0]
    np.testing.assert_allclose(
        puf["investment_income_elected_form_4952"].to_numpy(),
        [900.0, 0.0],
    )
    assert puf["investment_income_elected_form_4952"].sum() == pytest.approx(900.0)


def test_signal_gate_accepts_source_aligned_sparse_nondefault_values() -> None:
    values = np.zeros(2_000)
    values[[1_010, 1_020, 1_030]] = [1_000.0, 2_000.0, 3_000.0]
    channels = np.asarray(["asec"] * 1_000 + ["puf_tax_detail"] * 1_000)
    frame = _PersonFrame(
        pd.DataFrame(
            {
                "investment_income_elected_form_4952": values,
                "person_support_channel": channels,
            }
        )
    )

    result = us_form_4952_election_signal_gate(frame)  # type: ignore[arg-type]

    assert result.passed, result.failures
    assert result.details["positive_share"] == pytest.approx(0.0015)
    assert result.details["channels"]["asec"]["weighted_total"] == 0.0
    assert result.details["channels"]["puf_tax_detail"]["positive_share"] == (
        pytest.approx(0.003)
    )


@pytest.mark.parametrize(
    "person",
    [
        pd.DataFrame({"other": [0.0, 1.0]}),
        pd.DataFrame({"investment_income_elected_form_4952": [0.0, 0.0]}),
        pd.DataFrame({"investment_income_elected_form_4952": [0.0, -1.0]}),
        pd.DataFrame({"investment_income_elected_form_4952": [0.0, np.nan]}),
        pd.DataFrame(
            {
                "investment_income_elected_form_4952": [10.0, 0.0],
                "person_support_channel": ["asec", "puf_tax_detail"],
            }
        ),
    ],
)
def test_signal_gate_rejects_missing_default_invalid_or_asec_signal(
    person: pd.DataFrame,
) -> None:
    result = us_form_4952_election_signal_gate(  # type: ignore[arg-type]
        _PersonFrame(person)
    )

    assert not result.passed


def test_release_wiring_keeps_restored_input_hard_required() -> None:
    output = "investment_income_elected_form_4952"

    assert output in US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS
    assert output in RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
    assert output in load_release_input_coverage_manifest().required_columns


def test_shipped_neutralization_probe_binds_only_through_form_4952() -> None:
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "form_4952_election_neutralization"
    )

    assert probe.parameter_changes == {}
    assert probe.neutralized_variable == "investment_income_elected_form_4952"
    assert probe.binding_inputs == ("investment_income_elected_form_4952",)
    assert probe.period == 2024
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "positive"
    assert probe.budget_measure == "income_tax"
    assert probe.min_abs_effect == 1_000_000.0


@requires_us
def test_policyengine_us_contract_and_net_capital_gain_binding() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    variable = CountryTaxBenefitSystem().variables[
        "investment_income_elected_form_4952"
    ]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0

    adult = {
        "age": {2024: 45},
        "long_term_capital_gains_before_response": {2024: 100_000},
        "qualified_dividend_income": {2024: 0},
        "short_term_capital_gains": {2024: 0},
    }
    entities = {
        "tax_units": {
            "unit": {"members": ["adult"], "filing_status": {2024: "SINGLE"}}
        },
        "families": {"family": {"members": ["adult"]}},
        "spm_units": {"spm": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code_str": {2024: "CA"},
            }
        },
        "marital_units": {"marital": {"members": ["adult"]}},
    }
    baseline = Simulation(situation={"people": {"adult": adult}, **entities})
    election = Simulation(
        situation={
            "people": {
                "adult": {
                    **adult,
                    "investment_income_elected_form_4952": {2024: 25_000},
                }
            },
            **entities,
        }
    )

    assert baseline.calculate("net_capital_gain", 2024)[0] == pytest.approx(100_000)
    assert election.calculate("net_capital_gain", 2024)[0] == pytest.approx(75_000)
    assert (
        election.calculate("income_tax", 2024)[0]
        > baseline.calculate("income_tax", 2024)[0]
    )

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "form_4952_election_neutralization"
    )
    neutralized = Simulation(
        situation={
            "people": {
                "adult": {
                    **adult,
                    "investment_income_elected_form_4952": {2024: 25_000},
                }
            },
            **entities,
        },
        reform=_build_reform(probe),
    )
    assert (
        election.calculate("income_tax", 2024)[0]
        > neutralized.calculate("income_tax", 2024)[0]
    )
