"""Contracts for the IRS PUF educator-expense input restoration."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version

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
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime.educator_expenses import (
    EDUCATOR_EXPENSE_ARCHIVED_ALLOCATION_URL,
    EDUCATOR_EXPENSE_ARCHIVED_DERIVATION_URL,
    EDUCATOR_EXPENSE_ARCHIVED_EXPORT_URL,
    EDUCATOR_EXPENSE_ARCHIVED_PUF_IMPUTATION_URL,
    derive_us_educator_expense_from_puf,
    us_educator_expense_signal_gate,
    us_educator_expense_stage_spec,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_educator_expense_from_source,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)


class _ResolvedWeights:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _PersonFrame:
    def __init__(self, person: pd.DataFrame, weights: np.ndarray | None = None) -> None:
        self._person = person
        self._weights = np.ones(len(person)) if weights is None else weights

    def table(self, entity: str) -> pd.DataFrame:
        assert entity == "person"
        return self._person

    def resolve_weights(self, entity: str) -> _ResolvedWeights:
        assert entity == "person"
        return _ResolvedWeights(np.asarray(self._weights, dtype=np.float64))


def _minimal_us_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
            "person_family_id": np.asarray([1_000, 1_000, 2_000], dtype="int64"),
            "person_marital_unit_id": np.asarray(
                [10_000, 10_000, 20_000], dtype="int64"
            ),
            "employment_income_before_lsr": np.asarray([75_000.0, 25_000.0, 50_000.0]),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "state_fips": np.asarray([6, 36], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10, 20], dtype="int64"),
                "filing_status_input": ["JOINT", "SINGLE"],
            }
        ),
        "spm_unit": pd.DataFrame(
            {"spm_unit_id": np.asarray([100, 200], dtype="int64")}
        ),
        "family": pd.DataFrame(
            {"family_id": np.asarray([1_000, 2_000], dtype="int64")}
        ),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.asarray([10_000, 20_000], dtype="int64")}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.asarray([100.0, 100.0]),
                kind=WeightKind.DESIGN,
            )
        },
        pd.Series(["asec_2024", "asec_2024", "asec_2024"], name="stratum"),
    )


def test_immutable_archive_urls_pin_derivation_allocation_export_and_qrf() -> None:
    assert EDUCATOR_EXPENSE_ARCHIVED_DERIVATION_URL == (
        _ARCHIVED_ROOT + "datasets/puf/puf.py#L636-L649"
    )
    assert EDUCATOR_EXPENSE_ARCHIVED_ALLOCATION_URL == (
        _ARCHIVED_ROOT + "datasets/puf/puf.py#L617-L620"
    )
    assert EDUCATOR_EXPENSE_ARCHIVED_EXPORT_URL == (
        _ARCHIVED_ROOT + "datasets/puf/puf.py#L804-L815"
    )
    assert EDUCATOR_EXPENSE_ARCHIVED_PUF_IMPUTATION_URL == (
        _ARCHIVED_ROOT + "calibration/puf_impute.py#L940-L1075"
    )


def test_archived_e03220_mapping_is_exact_and_preserves_observed_topcode() -> None:
    source = pd.DataFrame(
        {
            "E03220": [0.0, 250.0, 500.0, 99_999.0],
            "other": [1, 2, 3, 4],
        }
    )
    before = source.copy(deep=True)

    result = derive_us_educator_expense_from_puf(source)

    pd.testing.assert_frame_equal(source, before)
    assert "educator_expense" not in source
    assert result["educator_expense"].tolist() == [0.0, 250.0, 500.0, 99_999.0]
    assert result["other"].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame({"other": [1.0]}), "requires source column"),
        (pd.DataFrame({"E03220": ["not numeric"]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E03220": [np.nan]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E03220": [np.inf]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E03220": [-1.0]}), "negative value"),
    ],
)
def test_source_derivation_fails_closed(
    source: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_us_educator_expense_from_puf(source)


def test_shared_puf_stage_pins_source_artifact_mapping_and_qrf_contract() -> None:
    stage = us_educator_expense_stage_spec()
    operations = {operation.kind: operation for operation in stage.operations}
    derive = operations["derive_puf_policyengine_variables"]
    qrf = operations["fit_weighted_qrf"]

    assert stage.stage == "puf_tax_detail"
    assert stage.survey == "IRS PUF 2015 (uprated)"
    assert stage.source == (
        "https://www.irs.gov/statistics/"
        "soi-tax-stats-individual-public-use-microdata-files"
    )
    assert stage.grain == "tax_unit"
    assert derive.parameters["educator_expense_source"] == "E03220"
    assert derive.parameters["educator_expense_output"] == "educator_expense"
    assert qrf.parameters["predictors"] == [
        "employment_income",
        "self_employment_income",
        "taxable_interest_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "capital_gains",
        "filing_status",
    ]
    assert "educator_expense" in stage.outputs
    assert "educator_expense" in stage.nonnegative_outputs
    locators = [str(artifact.get("locator")) for artifact in stage.artifacts]
    assert any("E03220" in locator for locator in locators)
    assert any(
        "release://policyengine/irs-soi-puf/1.8.0/puf_2024.h5" in locator
        for locator in locators
    )
    assert any(
        "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe" in locator for locator in locators
    )


def test_post_disaggregation_reconciliation_uses_final_e03220() -> None:
    source = pd.DataFrame(
        {
            "E03220": [0.0, 300.0, 600.0],
            "educator_expense": [999.0, 999.0, 999.0],
        }
    )
    before = source.copy(deep=True)

    result = _reconcile_puf_educator_expense_from_source(source)

    pd.testing.assert_frame_equal(source, before)
    assert result["educator_expense"].tolist() == [0.0, 300.0, 600.0]


@requires_us
def test_processed_puf_person_array_is_available_to_tax_unit_donor() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 20, 20],
            "educator_expense": [300.0, 150.0, 150.0],
        },
        person_outputs=("educator_expense",),
        tax_unit_outputs=(),
    )

    assert donor["educator_expense"].tolist() == [300.0, 300.0]
    assert donor["weight"].tolist() == [100.0, 200.0]
    assert donor["tax_unit_person_count"].tolist() == [1, 2]


def test_puf_support_declares_sparse_nonnegative_employment_allocation() -> None:
    assert (
        "educator_expense" in puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    )
    assert (
        "educator_expense" in puf_support_module._PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS
    )
    assert "educator_expense" in puf_support_module._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
    assert puf_support_module._PERSON_OUTPUT_DISTRIBUTION_BASIS["educator_expense"] == (
        "employment_income_before_lsr",
    )


@requires_us
def test_weighted_qrf_writes_only_puf_channel_and_allocates_by_employment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQRF:
        def __init__(self, *, n_estimators: int, seed: int) -> None:
            assert n_estimators == 4
            assert seed == 9

        def fit(
            self,
            frame,
            predictors,
            outputs,
            *,
            weights,
        ) -> FakeQRF:
            assert predictors == [
                "puf_predictor_filing_status_code",
                "puf_predictor_tax_unit_person_count",
            ]
            assert outputs == ["educator_expense"]
            assert weights == "design"
            return self

        def predict(
            self, features: pd.DataFrame, *, release_models: bool = False
        ) -> pd.DataFrame:
            assert len(features) == 2
            assert release_models is True
            # Sparse snapping maps these to observed donor values 600 and 0.
            return pd.DataFrame(
                {"educator_expense": [500.0, 100.0]},
                index=features.index,
            )

    monkeypatch.setattr(puf_support_module, "QRF", FakeQRF)
    expanded = clone_us_frame_for_puf_support(_minimal_us_frame())
    donor = pd.DataFrame(
        {
            "puf_predictor_filing_status_code": [1.0, 2.0],
            "puf_predictor_tax_unit_person_count": [1.0, 2.0],
            "educator_expense": [0.0, 600.0],
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
        person_outputs=("educator_expense",),
        tax_unit_outputs=(),
        n_estimators=4,
        seed=9,
    )

    person = imputed.table("person")
    channel = support_channel_column("person")
    asec = person[person[channel] == BASE_ASEC_SUPPORT_CHANNEL]
    puf = person[person[channel] == PUF_TAX_DETAIL_SUPPORT_CHANNEL]
    assert asec["educator_expense"].tolist() == [0.0, 0.0, 0.0]
    assert puf["educator_expense"].tolist() == [450.0, 150.0, 0.0]
    assert puf.groupby("person_tax_unit_id")["educator_expense"].sum().tolist() == [
        600.0,
        0.0,
    ]
    assert "educator_expense" not in expanded.table("person")


def test_signal_gate_requires_exact_zero_asec_and_sparse_nonzero_puf() -> None:
    n_per_channel = 1_000
    values = np.zeros(2 * n_per_channel)
    values[n_per_channel : n_per_channel + 20] = 300.0
    channels = np.asarray(
        [BASE_ASEC_SUPPORT_CHANNEL] * n_per_channel
        + [PUF_TAX_DETAIL_SUPPORT_CHANNEL] * n_per_channel
    )
    frame = _PersonFrame(
        pd.DataFrame(
            {
                "educator_expense": values,
                support_channel_column("person"): channels,
            }
        )
    )

    result = us_educator_expense_signal_gate(frame)  # type: ignore[arg-type]

    assert result.passed, result.failures
    assert result.details["positive_share"] == pytest.approx(0.01)
    assert result.details["channels"][BASE_ASEC_SUPPORT_CHANNEL][
        "positive_share"
    ] == pytest.approx(0.0)
    assert result.details["channels"][PUF_TAX_DETAIL_SUPPORT_CHANNEL][
        "positive_share"
    ] == pytest.approx(0.02)

    values[0] = 300.0
    contaminated = _PersonFrame(
        pd.DataFrame(
            {
                "educator_expense": values,
                support_channel_column("person"): channels,
            }
        )
    )
    failed = us_educator_expense_signal_gate(contaminated)  # type: ignore[arg-type]
    assert not failed.passed
    assert any("ASEC support channel" in failure for failure in failed.failures)


@requires_us
def test_policyengine_us_17646_contract_is_person_year_input_in_ald_graph() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    assert version("policyengine-us") == "1.764.6"
    system = CountryTaxBenefitSystem()
    variable = system.variables["educator_expense"]

    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0
    assert (
        variable.uprating
        == "calibration.gov.cbo.income_by_source.adjusted_gross_income"
    )
    assert system.variables["above_the_line_deductions"].adds == (
        "gov.irs.ald.deductions"
    )
    assert "educator_expense" in system.parameters.gov.irs.ald.deductions("2024-01-01")


@requires_us
def test_shipped_abolition_probe_has_positive_sign_and_binds_live() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "educator_expense_ald_abolition"
    )
    assert probe.period == 2024
    assert probe.expected_sign == "positive"
    assert probe.effect_direction == "reform_minus_baseline"
    assert probe.budget_measure == "income_tax"
    assert probe.binding_inputs == ("educator_expense",)
    assert probe.min_abs_effect == 1_000_000.0
    deductions = probe.parameter_changes["gov.irs.ald.deductions"]
    assert set(deductions) == {"2024-01-01.2024-12-31"}
    assert "educator_expense" not in deductions["2024-01-01.2024-12-31"]

    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": 100_000},
                "educator_expense": {"2024": 300},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
            }
        },
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    reform = Reform.from_dict(dict(probe.parameter_changes), country_id="us")
    baseline = Simulation(situation=situation)
    reformed = Simulation(
        tax_benefit_system=CountryTaxBenefitSystem(reform=(reform,)),
        situation=situation,
    )

    assert baseline.calculate("above_the_line_deductions", 2024)[0] == 300.0
    assert reformed.calculate("above_the_line_deductions", 2024)[0] == 0.0
    assert (
        reformed.calculate("income_tax", 2024)[0]
        - baseline.calculate("income_tax", 2024)[0]
        > 0.0
    )
