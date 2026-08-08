"""Contracts for source-backed PUF capital-gain detail inputs."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime.capital_gain_details import (
    CAPITAL_GAIN_DETAILS_ARCHIVED_DERIVATION_URL,
    CAPITAL_GAIN_DETAILS_ARCHIVED_EXPORT_URL,
    CAPITAL_GAIN_DETAILS_ARCHIVED_IMPUTATION_URL,
    CAPITAL_GAIN_DETAILS_ARCHIVED_PERSON_ALLOCATION_URL,
    CAPITAL_GAIN_DETAILS_ARCHIVED_PUF_ARTIFACT_URL,
    US_CAPITAL_GAIN_DETAILS_NONCONSTANT_PERSON_COLUMNS,
    US_CAPITAL_GAIN_DETAILS_NONCONSTANT_TAX_UNIT_COLUMNS,
    US_CAPITAL_GAIN_DETAILS_OUTPUT_COLUMNS,
    derive_us_capital_gain_details_from_puf,
    us_capital_gain_details_signal_gate,
    us_capital_gain_details_stage_spec,
)
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_TAX_UNIT_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_capital_gain_details_from_sources,
    derive_puf_policyengine_variables,
)
from microcosm.build.us_runtime.puf_support import puf_tax_unit_donor_from_arrays
from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
    load_release_input_coverage_manifest,
    us_release_reform_coverage_probes,
)

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)


class _ResolvedWeights:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _CapitalGainFrame:
    def __init__(
        self,
        *,
        person: pd.DataFrame,
        tax_unit: pd.DataFrame,
        person_weights: np.ndarray | None = None,
        tax_unit_weights: np.ndarray | None = None,
    ) -> None:
        self._tables = {"person": person, "tax_unit": tax_unit}
        self._weights = {
            "person": (
                np.ones(len(person)) if person_weights is None else person_weights
            ),
            "tax_unit": (
                np.ones(len(tax_unit)) if tax_unit_weights is None else tax_unit_weights
            ),
        }

    def table(self, entity: str) -> pd.DataFrame:
        return self._tables[entity]

    def resolve_weights(self, entity: str) -> _ResolvedWeights:
        return _ResolvedWeights(np.asarray(self._weights[entity], dtype=np.float64))


def test_archived_coordinates_pin_derivation_export_and_imputation() -> None:
    commit = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"

    assert commit in CAPITAL_GAIN_DETAILS_ARCHIVED_DERIVATION_URL
    assert CAPITAL_GAIN_DETAILS_ARCHIVED_DERIVATION_URL.endswith("puf.py#L636-L702")
    assert CAPITAL_GAIN_DETAILS_ARCHIVED_EXPORT_URL.endswith("puf.py#L804-L850")
    assert CAPITAL_GAIN_DETAILS_ARCHIVED_IMPUTATION_URL.endswith(
        "puf_impute.py#L90-L198"
    )
    assert CAPITAL_GAIN_DETAILS_ARCHIVED_PERSON_ALLOCATION_URL.endswith(
        "puf.py#L1513-L1601"
    )
    assert CAPITAL_GAIN_DETAILS_ARCHIVED_PUF_ARTIFACT_URL.endswith("puf.py#L1655-L1660")


def test_archived_puf_mappings_are_exact_carries() -> None:
    source = pd.DataFrame(
        {
            "E24518": [0.0, 125.5, 9_000.0],
            "E24515": [10.0, 0.0, 7_500.0],
        }
    )

    result = derive_us_capital_gain_details_from_puf(source)

    assert set(US_CAPITAL_GAIN_DETAILS_OUTPUT_COLUMNS).isdisjoint(source.columns)
    assert result["long_term_capital_gains_on_collectibles"].tolist() == [
        0.0,
        125.5,
        9_000.0,
    ]
    assert result["unrecaptured_section_1250_gain"].tolist() == [
        10.0,
        0.0,
        7_500.0,
    ]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame({"E24518": [1.0]}), "requires source columns"),
        (
            pd.DataFrame({"E24518": ["bad"], "E24515": [0.0]}),
            "nonnumeric or nonfinite",
        ),
        (
            pd.DataFrame({"E24518": [0.0], "E24515": [np.inf]}),
            "nonnumeric or nonfinite",
        ),
        (
            pd.DataFrame({"E24518": [-1.0], "E24515": [0.0]}),
            "negative value",
        ),
    ],
)
def test_source_derivation_fails_closed(
    source: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_us_capital_gain_details_from_puf(source)


def test_shared_puf_derivation_and_post_disaggregation_reconciliation() -> None:
    source = pd.DataFrame(
        {
            "E00600": [0.0, 0.0, 0.0],
            "E00650": [0.0, 0.0, 0.0],
            "E24518": [0.0, 3_000.0, 7_500.0],
            "E24515": [100.0, 0.0, 8_000.0],
        }
    )

    derived = derive_puf_policyengine_variables(
        source,
        collectibles_capital_gain_source="E24518",
        unrecaptured_section_1250_gain_source="E24515",
    )
    for output in US_CAPITAL_GAIN_DETAILS_OUTPUT_COLUMNS:
        derived[output] = 999.0
    result = _reconcile_puf_capital_gain_details_from_sources(derived)

    assert result["long_term_capital_gains_on_collectibles"].tolist() == [
        0.0,
        3_000.0,
        7_500.0,
    ]
    assert result["unrecaptured_section_1250_gain"].tolist() == [
        100.0,
        0.0,
        8_000.0,
    ]


def test_shared_puf_stage_declares_sources_outputs_and_artifact() -> None:
    stage = us_capital_gain_details_stage_spec()
    operation = next(
        operation
        for operation in stage.operations
        if operation.kind == "derive_puf_policyengine_variables"
    )

    assert operation.parameters["collectibles_capital_gain_source"] == "E24518"
    assert operation.parameters["unrecaptured_section_1250_gain_source"] == "E24515"
    assert set(US_CAPITAL_GAIN_DETAILS_OUTPUT_COLUMNS) <= set(stage.outputs)
    assert set(US_CAPITAL_GAIN_DETAILS_OUTPUT_COLUMNS) <= set(stage.nonnegative_outputs)
    assert any(
        all(field in str(artifact.get("locator")) for field in ("E24515", "E24518"))
        for artifact in stage.artifacts
    )
    assert any(
        "irs-soi-puf/1.8.0/puf_2024.h5" in str(artifact.get("locator"))
        for artifact in stage.artifacts
    )


def test_puf_support_keeps_both_details_nonnegative_and_sparse() -> None:
    person_output = US_CAPITAL_GAIN_DETAILS_NONCONSTANT_PERSON_COLUMNS[0]
    tax_unit_output = US_CAPITAL_GAIN_DETAILS_NONCONSTANT_TAX_UNIT_COLUMNS[0]

    assert person_output in puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert tax_unit_output in puf_support_module.PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    assert person_output in puf_support_module._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
    assert tax_unit_output in puf_support_module._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
    assert person_output in puf_support_module._PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS
    assert tax_unit_output in puf_support_module._PUF_TAX_DETAIL_SPARSE_TAX_UNIT_OUTPUTS
    assert person_output not in puf_support_module._PERSON_OUTPUT_DISTRIBUTION_BASIS


def test_processed_puf_arrays_preserve_identifiable_tax_unit_totals() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 20, 20],
            "long_term_capital_gains_on_collectibles": [0.0, 100.0, 200.0],
            "unrecaptured_section_1250_gain": [50.0, 400.0],
        },
        person_outputs=("long_term_capital_gains_on_collectibles",),
        tax_unit_outputs=("unrecaptured_section_1250_gain",),
    )

    assert donor["long_term_capital_gains_on_collectibles"].tolist() == [0.0, 300.0]
    assert donor["unrecaptured_section_1250_gain"].tolist() == [50.0, 400.0]


def test_signal_gate_accepts_sparse_source_aligned_values() -> None:
    person_values = np.zeros(2_000)
    person_values[[1_010, 1_020]] = [1_000.0, 2_000.0]
    tax_unit_values = np.zeros(2_000)
    tax_unit_values[[1_010, 1_020, 1_030, 1_040]] = [100.0, 200.0, 300.0, 400.0]
    channels = np.asarray(["asec"] * 1_000 + ["puf_tax_detail"] * 1_000)
    frame = _CapitalGainFrame(
        person=pd.DataFrame(
            {
                "long_term_capital_gains_on_collectibles": person_values,
                "person_support_channel": channels,
            }
        ),
        tax_unit=pd.DataFrame(
            {
                "unrecaptured_section_1250_gain": tax_unit_values,
                "tax_unit_support_channel": channels,
            }
        ),
    )

    result = us_capital_gain_details_signal_gate(frame)  # type: ignore[arg-type]

    assert result.passed, result.failures
    assert result.details["long_term_capital_gains_on_collectibles"][
        "positive_share"
    ] == pytest.approx(0.001)
    assert result.details["unrecaptured_section_1250_gain"][
        "positive_share"
    ] == pytest.approx(0.002)


def test_release_wiring_promotes_only_source_backed_detail_leaves() -> None:
    manifest = load_release_input_coverage_manifest()
    person_output = US_CAPITAL_GAIN_DETAILS_NONCONSTANT_PERSON_COLUMNS[0]
    tax_unit_output = US_CAPITAL_GAIN_DETAILS_NONCONSTANT_TAX_UNIT_COLUMNS[0]

    assert person_output in US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS
    assert tax_unit_output in US_RELEASE_REQUIRED_TAX_UNIT_SOURCE_COLUMNS
    assert set(US_CAPITAL_GAIN_DETAILS_OUTPUT_COLUMNS) <= (
        RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
    )
    assert set(US_CAPITAL_GAIN_DETAILS_OUTPUT_COLUMNS) <= manifest.required_columns
    assert "investment_interest_expense" in RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
    assert "investment_interest_expense" in manifest.required_columns
    assert "investment_interest_expense" not in manifest.reviewed_exclusions


def test_shipped_neutralization_probes_bind_each_restored_leaf() -> None:
    probes = {probe.id: probe for probe in us_release_reform_coverage_probes()}
    collectibles = probes["collectibles_gain_neutralization"]
    unrecaptured = probes["unrecaptured_section_1250_gain_neutralization"]

    assert collectibles.neutralized_variable == (
        "long_term_capital_gains_on_collectibles"
    )
    assert collectibles.binding_inputs == ("long_term_capital_gains_on_collectibles",)
    assert collectibles.budget_measure == "income_tax"
    assert collectibles.expected_sign == "positive"
    assert unrecaptured.neutralized_variable == "unrecaptured_section_1250_gain"
    assert unrecaptured.binding_inputs == ("unrecaptured_section_1250_gain",)
    assert unrecaptured.budget_measure == "income_tax"
    assert unrecaptured.expected_sign == "positive"


@requires_us
def test_policyengine_us_contracts_and_household_bindings() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    system = CountryTaxBenefitSystem()
    collectibles = system.variables["long_term_capital_gains_on_collectibles"]
    unrecaptured = system.variables["unrecaptured_section_1250_gain"]
    assert collectibles.is_input_variable()
    assert collectibles.entity.key == "person"
    assert unrecaptured.is_input_variable()
    assert unrecaptured.entity.key == "tax_unit"

    adult = {
        "age": {2024: 45},
        "employment_income": {2024: 120_000},
        "long_term_capital_gains_before_response": {2024: 100_000},
        "long_term_capital_gains_on_collectibles": {2024: 25_000},
    }
    entities = {
        "tax_units": {
            "unit": {
                "members": ["adult"],
                "filing_status": {2024: "SINGLE"},
                "unrecaptured_section_1250_gain": {2024: 20_000},
            }
        },
        "families": {"family": {"members": ["adult"]}},
        "spm_units": {"spm": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code_str": {2024: "MA"},
            }
        },
        "marital_units": {"marital": {"members": ["adult"]}},
    }
    baseline = Simulation(situation={"people": {"adult": adult}, **entities})

    probes = {probe.id: probe for probe in us_release_reform_coverage_probes()}
    no_collectibles = Simulation(
        situation={"people": {"adult": adult}, **entities},
        reform=_build_reform(probes["collectibles_gain_neutralization"]),
    )
    no_unrecaptured = Simulation(
        situation={"people": {"adult": adult}, **entities},
        reform=_build_reform(probes["unrecaptured_section_1250_gain_neutralization"]),
    )

    assert (
        baseline.calculate("income_tax", 2024)[0]
        > no_collectibles.calculate("income_tax", 2024)[0]
    )
    assert (
        baseline.calculate("income_tax", 2024)[0]
        > no_unrecaptured.calculate("income_tax", 2024)[0]
    )
