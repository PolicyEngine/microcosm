"""Signed ASEC/PUF farm-business input restoration contracts."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.puf_support as puf_support_module
from populace.build.us_runtime.cps_carried import derive_us_cps_carried_inputs
from populace.build.us_runtime.farm_business_income import (
    FARM_BUSINESS_INCOME_ARCHIVED_CPS_FARM_INCOME_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_DERIVATION_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_EXPORT_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_IMPUTATION_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_OVERRIDE_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_PUF_ARTIFACT_URL,
    US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS,
    derive_us_farm_business_income_from_puf,
    us_farm_business_income_signal_gate,
    us_farm_business_income_stage_spec,
)
from populace.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_farm_business_income_from_sources,
)
from populace.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    puf_tax_unit_donor_from_arrays,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)

_OPERATIONS, _RENT = US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS


def _asec_frame() -> Frame:
    count = 20
    ids = np.arange(1, count + 1, dtype=np.int64)
    farm_operations = np.zeros(count, dtype=np.float64)
    farm_operations[:4] = [500.0, -200.0, 800.0, -300.0]
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids + 100,
            "person_tax_unit_id": ids + 200,
            "person_spm_unit_id": ids + 300,
            "person_family_id": ids + 400,
            "person_marital_unit_id": ids + 500,
            "A_AGE": np.arange(30, 30 + count),
            "A_SEX": np.tile([1, 2], count // 2),
            "WSAL_VAL": np.arange(count) * 1_000.0,
            "SEMP_VAL": np.arange(count) * 100.0,
            "FRSE_VAL": farm_operations,
            "OI_VAL": np.zeros(count),
            "OI_OFF": np.zeros(count),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": ids + 100, "state_fips": np.full(count, 6)}
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": ids + 200,
                "filing_status_input": ["SINGLE"] * count,
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids + 300}),
        "family": pd.DataFrame({"family_id": ids + 400}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids + 500}),
    }
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


def _imputed_frame(monkeypatch: pytest.MonkeyPatch) -> Frame:
    asec = derive_us_cps_carried_inputs(_asec_frame())
    expanded = clone_us_frame_for_puf_support(asec)
    predictions = {
        _OPERATIONS: np.asarray([700.0, -400.0, 900.0, -500.0, *([0.0] * 16)]),
        _RENT: np.asarray([300.0, -100.0, 600.0, -200.0, *([0.0] * 16)]),
    }

    class FakeFitted:
        def predict(self, test: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(predictions, index=test.index)

    class FakeQRF:
        def __init__(self, *, n_estimators: int, seed: int) -> None:
            assert n_estimators == 100
            assert seed == 11

        def fit(
            self,
            frame: Frame,
            predictors: list[str],
            outcomes: list[str],
            *,
            weights: str,
        ) -> FakeFitted:
            assert predictors == []
            assert outcomes == list(US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS)
            assert weights == "design"
            return FakeFitted()

    monkeypatch.setattr(puf_support_module, "QRF", FakeQRF)
    donor = pd.DataFrame(
        {
            _OPERATIONS: [10.0, -20.0, 0.0, 30.0],
            _RENT: [-5.0, 15.0, 0.0, 25.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    return impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=(),
        person_outputs=US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS,
        tax_unit_outputs=(),
        seed=11,
    )


def test_archive_urls_pin_asec_puf_export_qrf_override_and_artifact() -> None:
    commit = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
    urls = (
        FARM_BUSINESS_INCOME_ARCHIVED_CPS_FARM_INCOME_URL,
        FARM_BUSINESS_INCOME_ARCHIVED_DERIVATION_URL,
        FARM_BUSINESS_INCOME_ARCHIVED_EXPORT_URL,
        FARM_BUSINESS_INCOME_ARCHIVED_IMPUTATION_URL,
        FARM_BUSINESS_INCOME_ARCHIVED_OVERRIDE_URL,
        FARM_BUSINESS_INCOME_ARCHIVED_PUF_ARTIFACT_URL,
    )
    assert all(commit in url for url in urls)
    assert FARM_BUSINESS_INCOME_ARCHIVED_CPS_FARM_INCOME_URL.endswith(
        "datasets/cps/cps.py#L1363-L1382"
    )
    assert FARM_BUSINESS_INCOME_ARCHIVED_DERIVATION_URL.endswith(
        "datasets/puf/puf.py#L636-L704"
    )
    assert FARM_BUSINESS_INCOME_ARCHIVED_EXPORT_URL.endswith(
        "datasets/puf/puf.py#L804-L875"
    )
    assert FARM_BUSINESS_INCOME_ARCHIVED_IMPUTATION_URL.endswith(
        "calibration/puf_impute.py#L80-L198"
    )
    assert FARM_BUSINESS_INCOME_ARCHIVED_OVERRIDE_URL.endswith(
        "calibration/puf_impute.py#L513-L672"
    )
    assert FARM_BUSINESS_INCOME_ARCHIVED_PUF_ARTIFACT_URL.endswith(
        "datasets/puf/puf.py#L1655-L1660"
    )


def test_shared_puf_stage_pins_exact_signed_sources_and_outputs() -> None:
    spec = us_farm_business_income_stage_spec()
    operation = next(
        operation
        for operation in spec.operations
        if operation.kind == "derive_puf_policyengine_variables"
    )

    assert operation.parameters["farm_operations_income_source"] == "E02100"
    assert operation.parameters["farm_operations_income_output"] == _OPERATIONS
    assert operation.parameters["farm_rent_income_source"] == "E27200"
    assert operation.parameters["farm_rent_income_output"] == _RENT
    assert set(US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS) <= set(spec.outputs)
    assert not set(US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS).intersection(
        spec.nonnegative_outputs
    )
    assert "preserves measured ASEC operations income" in spec.notes


def test_direct_puf_mapping_preserves_both_signs_and_does_not_mutate() -> None:
    source = pd.DataFrame(
        {
            "E02100": [10_000.0, -8_000.0, 0.0, 99_999.0],
            "E27200": [-3_000.0, 7_000.0, 0.0, 88_888.0],
            "T27800": [1.0, 2.0, 3.0, 4.0],
        }
    )
    original = source.copy(deep=True)

    result = derive_us_farm_business_income_from_puf(source)

    assert result[_OPERATIONS].tolist() == [10_000.0, -8_000.0, 0.0, 99_999.0]
    assert result[_RENT].tolist() == [-3_000.0, 7_000.0, 0.0, 88_888.0]
    assert result["T27800"].tolist() == [1.0, 2.0, 3.0, 4.0]
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("E02100", np.nan, "nonnumeric or nonfinite"),
        ("E02100", np.inf, "nonnumeric or nonfinite"),
        ("E27200", -np.inf, "nonnumeric or nonfinite"),
    ],
)
def test_direct_puf_mapping_fails_closed_on_invalid_not_negative_values(
    column: str,
    bad_value: float,
    message: str,
) -> None:
    source = pd.DataFrame({"E02100": [-1.0], "E27200": [-2.0]})
    source.loc[0, column] = bad_value

    with pytest.raises(ValueError, match=message):
        derive_us_farm_business_income_from_puf(source)


def test_cps_frse_maps_to_operations_not_schedule_j_farm_income() -> None:
    source = _asec_frame()
    original = source.table("person").copy(deep=True)

    result = derive_us_cps_carried_inputs(source)

    pd.testing.assert_frame_equal(source.table("person"), original)
    np.testing.assert_allclose(
        result.table("person")[_OPERATIONS],
        original["FRSE_VAL"],
    )
    assert "farm_income" not in result.table("person")
    assert _RENT not in result.table("person")


def test_processed_puf_donor_aggregates_signed_person_values() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 10, 20],
            _OPERATIONS: [2_000.0, -3_000.0, -4_000.0],
            _RENT: [-500.0, 1_500.0, -2_000.0],
        },
        person_outputs=US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS,
        tax_unit_outputs=(),
    )

    assert donor[_OPERATIONS].tolist() == [-1_000.0, -4_000.0]
    assert donor[_RENT].tolist() == [1_000.0, -2_000.0]


def test_post_disaggregation_reconciliation_uses_final_signed_sources() -> None:
    source = pd.DataFrame(
        {
            "E02100": [-7_000.0, 9_000.0],
            "E27200": [3_000.0, -5_000.0],
            _OPERATIONS: [999.0, 999.0],
            _RENT: [999.0, 999.0],
        }
    )

    result = _reconcile_puf_farm_business_income_from_sources(source)

    assert result[_OPERATIONS].tolist() == [-7_000.0, 9_000.0]
    assert result[_RENT].tolist() == [3_000.0, -5_000.0]


def test_puf_runtime_keeps_signed_outputs_out_of_clipping_and_sparse_sets() -> None:
    for output in US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS:
        assert output in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
        assert output not in puf_support_module._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
        assert output not in puf_support_module._PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS


def test_weighted_qrf_preserves_asec_and_writes_signed_puf_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _imputed_frame(monkeypatch)
    person = result.table("person")
    asec = person[person["person_support_channel"] == "asec"]
    puf = person[person["person_support_channel"] == "puf_tax_detail"]

    assert asec[_OPERATIONS].tolist()[:4] == [500.0, -200.0, 800.0, -300.0]
    assert not asec[_RENT].any()
    assert puf[_OPERATIONS].tolist()[:4] == [700.0, -400.0, 900.0, -500.0]
    assert puf[_RENT].tolist()[:4] == [300.0, -100.0, 600.0, -200.0]
    gate = us_farm_business_income_signal_gate(result)
    assert gate.passed, gate.failures


def test_signal_gate_rejects_clipped_losses_and_asec_rent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clipped = _imputed_frame(monkeypatch)
    person = clipped.table("person")
    person[_OPERATIONS] = person[_OPERATIONS].clip(lower=0.0)
    person[_RENT] = person[_RENT].clip(lower=0.0)
    gate = us_farm_business_income_signal_gate(clipped)
    assert not gate.passed
    assert any("farm losses" in failure for failure in gate.failures)

    misplaced = _imputed_frame(monkeypatch)
    asec_mask = misplaced.table("person")["person_support_channel"] == "asec"
    misplaced.table("person").loc[asec_mask, _RENT] = 1.0
    gate = us_farm_business_income_signal_gate(misplaced)
    assert not gate.passed
    assert any("ASEC support carries nonzero" in failure for failure in gate.failures)


@requires_us
def test_policyengine_us_1_764_6_qbi_graph_reads_each_farm_leaf() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "1.764.6"
    system = CountryTaxBenefitSystem()
    for output in US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS:
        variable = system.variables[output]
        assert variable.is_input_variable()
        assert variable.entity.key == "person"
        assert str(variable.definition_period).lower() == "year"
        assert variable.default_value == 0

    reform = Reform.from_dict(
        {
            "gov.irs.deductions.qbi.income_definition": {
                "2026-01-01.2026-12-31": [
                    "self_employment_income",
                    "partnership_s_corp_income",
                    "rental_income",
                    "estate_income",
                ]
            }
        },
        country_id="us",
    )

    def situation(output: str) -> dict[str, object]:
        return {
            "people": {
                "adult": {
                    "age": {"2026": 40},
                    output: {"2026": 10_000.0},
                }
            },
            "tax_units": {
                "tax_unit": {
                    "members": ["adult"],
                    "filing_status": {"2026": "SINGLE"},
                }
            },
            "spm_units": {"spm_unit": {"members": ["adult"]}},
            "households": {
                "household": {
                    "members": ["adult"],
                    "state_code": {"2026": "CA"},
                }
            },
        }

    reformed_system = CountryTaxBenefitSystem(reform=(reform,))
    for output in US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS:
        baseline = Simulation(tax_benefit_system=system, situation=situation(output))
        reformed = Simulation(
            tax_benefit_system=reformed_system,
            situation=situation(output),
        )
        assert baseline.calculate("qualified_business_income", 2026)[0] > 9_000.0
        assert baseline.calculate("qualified_business_income_deduction", 2026)[
            0
        ] == pytest.approx(400.0)
        assert reformed.calculate("qualified_business_income", 2026)[0] == 0.0
        assert reformed.calculate("qualified_business_income_deduction", 2026)[0] == 0.0
