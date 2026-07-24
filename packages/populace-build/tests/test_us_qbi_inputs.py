"""Contracts for the retired eCPS Section 199A input family."""

from __future__ import annotations

import importlib.util
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime import puf_support as puf_support_module
from populace.build.us_runtime.qbi_inputs import (
    QBI_ARCHIVED_ASSUMPTIONS_URL,
    QBI_ARCHIVED_CLONE_URL,
    QBI_ARCHIVED_DERIVATION_URL,
    QBI_ARCHIVED_EXPORT_URL,
    QBI_ARCHIVED_IMPUTATION_URL,
    QBI_ARCHIVED_PUF_ARTIFACT_URL,
    QBI_ARCHIVED_SIMULATION_URL,
    US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    US_QBI_NONNEGATIVE_OUTPUT_COLUMNS,
    US_QBI_OUTPUT_COLUMNS,
    us_qbi_inputs_signal_gate,
    us_qbi_inputs_stage_spec,
    us_qbi_inputs_summary,
    with_host_sstb_classification,
    with_us_qbi_input_reconciliation,
)
from populace.build.us_runtime.qbi_simulation import (
    QBI_SIMULATION_V2,
    load_qbi_simulation_assumptions,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights
from populace.frame.schema import EntitySchema

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)


def _frame(person: pd.DataFrame) -> Frame:
    person = person.copy(deep=True).reset_index(drop=True)
    n = len(person)
    ids = np.arange(1, n + 1, dtype=np.int64)
    person.insert(0, "person_id", ids)
    for entity in ("household", "tax_unit", "spm_unit", "family", "marital_unit"):
        person[f"person_{entity}_id"] = ids
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )


def _qbi_person(n: int = 200) -> pd.DataFrame:
    person = pd.DataFrame(
        {
            column: np.zeros(n, dtype=bool)
            if column in US_QBI_BOOLEAN_OUTPUT_COLUMNS
            else np.zeros(n, dtype=np.float64)
            for column in US_QBI_OUTPUT_COLUMNS
        }
    )
    person["person_support_channel"] = np.where(
        np.arange(n) < n // 2, "asec", "puf_tax_detail"
    )
    person["self_employment_income_before_lsr"] = 0.0
    person["partnership_income"] = 0.0
    person["s_corp_income"] = 0.0
    person["estate_income"] = 0.0
    person["rental_income"] = 0.0
    person["non_qualified_dividend_income"] = 0.0

    puf = np.arange(n // 2, n)
    for index, column in enumerate(US_QBI_BOOLEAN_OUTPUT_COLUMNS):
        if column == "business_is_sstb":
            continue
        person.loc[puf[index % 5 :: 5], column] = False
        person.loc[puf, column] |= np.arange(len(puf)) % 5 != index % 5

    sstb = puf[:10]
    person.loc[sstb, "business_is_sstb"] = True
    person.loc[sstb, "self_employment_income_would_be_qualified"] = True
    person.loc[sstb, "self_employment_income_before_lsr"] = 10_000.0
    person.loc[sstb, "w2_wages_from_qualified_business"] = 2_000.0
    person.loc[sstb, "unadjusted_basis_qualified_property"] = 5_000.0

    person.loc[puf[:2], "non_qualified_dividend_income"] = 1_000.0
    person.loc[puf[:2], "qualified_bdc_income"] = 20.0
    person.loc[puf[:20], "non_qualified_dividend_income"] = 1_000.0
    person.loc[puf[:20], "qualified_reit_and_ptp_income"] = 40.0
    person.loc[puf[:25], "w2_wages_from_qualified_business"] = 2_000.0
    person.loc[puf[:30], "unadjusted_basis_qualified_property"] = 5_000.0
    return person


@pytest.fixture
def ready_sstb_crosswalk() -> dict[str, object]:
    return {
        "schema_version": 1,
        "crosswalk_version": "synthetic-host-test-v1",
        "status": "ready",
        "occupation_code_system": "synthetic Census occupation",
        "industry_code_system": None,
        "mapping": {
            "occupation": {
                "1010": "clear_sstb",
                "2020": "non_sstb",
                "3030": "ambiguous",
            },
            "industry": {},
        },
    }


def _v2_host_person() -> pd.DataFrame:
    person = _qbi_person()
    person["PEIOOCC"] = 2020
    person["AGI"] = 150_000.0
    person["farm_operations_income"] = 0.0
    person.loc[100:109, "farm_operations_income"] = 1_000.0
    person.loc[[103, 105, 110], "partnership_income"] = 2_000.0

    person.loc[100, "PEIOOCC"] = 1010
    person.loc[101, "PEIOOCC"] = 2020
    person.loc[102, "PEIOOCC"] = 3030

    # Passive-only, low-AGI partnership income.
    person.loc[103, "self_employment_income_before_lsr"] = 0.0
    person.loc[103, "sstb_self_employment_income_before_lsr"] = 0.0
    person.loc[103, "AGI"] = 50_000.0

    # Passive-only estate income in a zero-probability AGI band.
    person.loc[104, "self_employment_income_before_lsr"] = 0.0
    person.loc[104, "sstb_self_employment_income_before_lsr"] = 0.0
    person.loc[104, "estate_income"] = 2_000.0
    person.loc[104, "estate_income_would_be_qualified"] = True

    # A non-SSTB Schedule C signal takes precedence over the passive prior.
    person.loc[105, "PEIOOCC"] = 2020
    person.loc[105, "AGI"] = 50_000.0

    # No positive qualified mapped source, despite a one-valued AGI prior.
    person.loc[106, "self_employment_income_before_lsr"] = 0.0
    person.loc[106, "sstb_self_employment_income_before_lsr"] = 0.0
    person.loc[106, "AGI"] = 50_000.0
    return person


def _host_test_assumptions():
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_V2)
    classification = assumptions.sstb_classification
    bands = tuple(
        replace(
            band,
            probability=1.0 if band.label == "-inf:100000" else 0.0,
        )
        for band in classification.passive_passthrough_sstb_prior_by_agi
    )
    return replace(
        assumptions,
        sstb_classification=replace(
            classification,
            ambiguous_prior=1.0,
            passive_passthrough_sstb_prior_by_agi=bands,
        ),
    )


def test_archived_coordinates_pin_algorithms_export_clone_and_artifact() -> None:
    urls = (
        QBI_ARCHIVED_ASSUMPTIONS_URL,
        QBI_ARCHIVED_CLONE_URL,
        QBI_ARCHIVED_DERIVATION_URL,
        QBI_ARCHIVED_EXPORT_URL,
        QBI_ARCHIVED_IMPUTATION_URL,
        QBI_ARCHIVED_PUF_ARTIFACT_URL,
        QBI_ARCHIVED_SIMULATION_URL,
    )
    for url in urls:
        assert "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe" in url
        assert "#L" in url


def test_output_contract_is_exact_and_source_declared() -> None:
    expected = {
        "business_is_sstb",
        "estate_income_would_be_qualified",
        "farm_operations_income_would_be_qualified",
        "farm_rent_income_would_be_qualified",
        "partnership_s_corp_income_would_be_qualified",
        "rental_income_would_be_qualified",
        "self_employment_income_would_be_qualified",
        "sstb_self_employment_income_would_be_qualified",
        "qualified_bdc_income",
        "qualified_reit_and_ptp_income",
        "sstb_self_employment_income_before_lsr",
        "sstb_unadjusted_basis_qualified_property",
        "sstb_w2_wages_from_qualified_business",
        "unadjusted_basis_qualified_property",
        "w2_wages_from_qualified_business",
    }
    assert set(US_QBI_OUTPUT_COLUMNS) == expected
    assert set(US_QBI_BOOLEAN_OUTPUT_COLUMNS) == {
        name for name in expected if name.endswith("_would_be_qualified")
    } | {"business_is_sstb"}
    assert set(US_QBI_NONNEGATIVE_OUTPUT_COLUMNS) == {
        "qualified_bdc_income",
        "qualified_reit_and_ptp_income",
        "sstb_unadjusted_basis_qualified_property",
        "sstb_w2_wages_from_qualified_business",
        "unadjusted_basis_qualified_property",
        "w2_wages_from_qualified_business",
    }

    stage = us_qbi_inputs_stage_spec()
    assert expected <= set(stage.outputs)
    assert set(US_QBI_NONNEGATIVE_OUTPUT_COLUMNS) <= set(stage.nonnegative_outputs)
    assert "release://policyengine/irs-soi-puf/1.8.0/puf_2024.h5" in str(
        stage.artifacts
    )


def test_legacy_processed_puf_sstb_alias_is_resolved_for_donor() -> None:
    source = {"sstb_self_employment_income": np.array([-20.0, 0.0, 300.0])}

    values = puf_support_module._person_source_values(
        source, "sstb_self_employment_income_before_lsr"
    )

    assert values.tolist() == [-20.0, 0.0, 300.0]


def test_boolean_tax_unit_counts_are_preserved_and_source_aligned() -> None:
    person = pd.DataFrame(
        {
            "person_tax_unit_id": [1, 1, 1, 2, 2],
            "self_employment_income_before_lsr": [0.0, 50.0, 100.0, 0.0, 25.0],
            "business_is_sstb": 0.0,
        }
    )

    puf_support_module._write_person_tax_unit_boolean_counts(
        person,
        mask=pd.Series(True, index=person.index),
        column="business_is_sstb",
        totals=pd.Series({1: 2.0, 2: 1.0}),
        fallback_basis_columns=("self_employment_income_before_lsr",),
    )

    assert person.groupby("person_tax_unit_id")["business_is_sstb"].sum().to_dict() == {
        1: 2.0,
        2: 1.0,
    }
    assert person["business_is_sstb"].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]


def test_reconciliation_restores_sstb_routes_total_pools_and_exposure_caps() -> None:
    person = _qbi_person(200)
    original_total = (
        person["self_employment_income_before_lsr"]
        + person["sstb_self_employment_income_before_lsr"]
    ).to_numpy(copy=True)
    person.loc[100, "qualified_bdc_income"] = 2_000.0
    person.loc[100, "qualified_reit_and_ptp_income"] = 4_000.0

    result = with_us_qbi_input_reconciliation(_frame(person))
    reconciled = result.table("person")

    np.testing.assert_allclose(
        reconciled["self_employment_income_before_lsr"]
        + reconciled["sstb_self_employment_income_before_lsr"],
        original_total,
    )
    sstb = reconciled["business_is_sstb"].to_numpy()
    assert np.all(reconciled.loc[sstb, "self_employment_income_before_lsr"] == 0)
    assert np.all(reconciled.loc[~sstb, "sstb_self_employment_income_before_lsr"] == 0)
    np.testing.assert_allclose(
        reconciled["sstb_w2_wages_from_qualified_business"],
        np.where(sstb, reconciled["w2_wages_from_qualified_business"], 0.0),
    )
    np.testing.assert_allclose(
        reconciled["sstb_unadjusted_basis_qualified_property"],
        np.where(sstb, reconciled["unadjusted_basis_qualified_property"], 0.0),
    )
    # The base W-2 and UBIA leaves remain total pools; they are not zeroed on
    # all-SSTB rows.
    assert reconciled.loc[sstb, "w2_wages_from_qualified_business"].sum() > 0
    assert reconciled.loc[sstb, "unadjusted_basis_qualified_property"].sum() > 0
    assert reconciled.loc[100, "qualified_bdc_income"] == 1_000.0
    assert reconciled.loc[100, "qualified_reit_and_ptp_income"] == 1_000.0


def test_host_sstb_classification_fails_closed_on_packaged_placeholder() -> None:
    person = _v2_host_person()

    with pytest.raises(ValueError, match="status is 'placeholder'"):
        with_host_sstb_classification(
            _frame(person),
            qbi_simulation_version=QBI_SIMULATION_V2,
        )


def test_host_sstb_classification_routes_host_and_passive_records(
    ready_sstb_crosswalk: dict[str, object],
) -> None:
    person = _v2_host_person()
    source = _frame(person)
    source_before = source.table("person").copy(deep=True)
    original_total = (
        person["self_employment_income_before_lsr"]
        + person["sstb_self_employment_income_before_lsr"]
    ).to_numpy(copy=True)
    original_w2 = person["w2_wages_from_qualified_business"].to_numpy(copy=True)
    original_ubia = person["unadjusted_basis_qualified_property"].to_numpy(copy=True)
    assumptions = _host_test_assumptions()

    first = with_host_sstb_classification(
        source,
        qbi_simulation_version=QBI_SIMULATION_V2,
        assumptions=assumptions,
        sstb_crosswalk=ready_sstb_crosswalk,
    )
    second = with_host_sstb_classification(
        source,
        qbi_simulation_version=QBI_SIMULATION_V2,
        assumptions=assumptions,
        sstb_crosswalk=ready_sstb_crosswalk,
    )
    result = first.table("person")

    pd.testing.assert_frame_equal(source_before, source.table("person"))
    pd.testing.assert_frame_equal(result, second.table("person"))
    assert result.loc[100:106, "business_is_sstb"].tolist() == [
        True,
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    np.testing.assert_allclose(
        result["self_employment_income_before_lsr"]
        + result["sstb_self_employment_income_before_lsr"],
        original_total,
    )
    np.testing.assert_array_equal(
        result["w2_wages_from_qualified_business"],
        original_w2,
    )
    np.testing.assert_array_equal(
        result["unadjusted_basis_qualified_property"],
        original_ubia,
    )
    business = result["business_is_sstb"].to_numpy()
    np.testing.assert_array_equal(
        result["sstb_w2_wages_from_qualified_business"],
        np.where(business, original_w2, 0.0),
    )
    np.testing.assert_array_equal(
        result["sstb_unadjusted_basis_qualified_property"],
        np.where(business, original_ubia, 0.0),
    )
    ordinary_route = result["self_employment_income_would_be_qualified"].to_numpy()
    sstb_route = result["sstb_self_employment_income_would_be_qualified"].to_numpy()
    assert not np.any(ordinary_route & sstb_route)

    summary = us_qbi_inputs_summary(first)
    assert all(count == 0 for count in summary["invariants"].values())
    gate = us_qbi_inputs_signal_gate(first)
    assert gate.passed, gate.failures


def test_host_sstb_stream_is_independent_of_qualification_mode(
    ready_sstb_crosswalk: dict[str, object],
) -> None:
    person = _v2_host_person()
    person["partnership_s_corp_income_would_be_qualified"] = True
    assumptions = _host_test_assumptions()
    prior_partnership = replace(
        assumptions,
        qualification_derivations=tuple(
            replace(
                derivation,
                mode="prior",
                prior_probability=1.0,
            )
            if derivation.source == "partnership_s_corp_income"
            else derivation
            for derivation in assumptions.qualification_derivations
        ),
    )

    baseline = with_host_sstb_classification(
        _frame(person),
        qbi_simulation_version=QBI_SIMULATION_V2,
        assumptions=assumptions,
        sstb_crosswalk=ready_sstb_crosswalk,
    ).table("person")
    changed = with_host_sstb_classification(
        _frame(person),
        qbi_simulation_version=QBI_SIMULATION_V2,
        assumptions=prior_partnership,
        sstb_crosswalk=ready_sstb_crosswalk,
    ).table("person")

    assert not np.array_equal(
        baseline["partnership_s_corp_income_would_be_qualified"],
        changed["partnership_s_corp_income_would_be_qualified"],
    )
    for column in (
        "business_is_sstb",
        "sstb_self_employment_income_before_lsr",
        "sstb_w2_wages_from_qualified_business",
        "sstb_unadjusted_basis_qualified_property",
    ):
        assert baseline[column].to_numpy().tobytes() == (
            changed[column].to_numpy().tobytes()
        )


def test_host_sstb_uses_industry_first_then_occupation_fallback(
    ready_sstb_crosswalk: dict[str, object],
) -> None:
    person = _v2_host_person()
    person["PEIOIND"] = 0
    person.loc[100, "PEIOIND"] = 4040
    person.loc[101, "PEIOIND"] = 5050
    assumptions = _host_test_assumptions()
    assumptions = replace(
        assumptions,
        sstb_classification=replace(
            assumptions.sstb_classification,
            industry_column="PEIOIND",
        ),
    )
    crosswalk = {
        **ready_sstb_crosswalk,
        "industry_code_system": "synthetic Census industry",
        "mapping": {
            **ready_sstb_crosswalk["mapping"],
            "industry": {
                "4040": "non_sstb",
                "5050": "clear_sstb",
            },
        },
    }

    result = with_host_sstb_classification(
        _frame(person),
        qbi_simulation_version=QBI_SIMULATION_V2,
        assumptions=assumptions,
        sstb_crosswalk=crosswalk,
    ).table("person")

    # Industry overrides occupation on the first two rows; the unmapped
    # industry on row 102 falls back to its ambiguous occupation and prior.
    assert result.loc[100:102, "business_is_sstb"].tolist() == [
        False,
        True,
        True,
    ]


def test_sstb_requires_a_positive_qualified_mapped_source() -> None:
    person = _qbi_person(200)
    row = 100
    person.loc[row, "business_is_sstb"] = True
    person.loc[row, "self_employment_income_before_lsr"] = 500.0
    person.loc[row, "self_employment_income_would_be_qualified"] = False
    person.loc[row, "sstb_self_employment_income_would_be_qualified"] = False
    person.loc[row, "partnership_income"] = 0.0
    person.loc[row, "s_corp_income"] = 0.0
    person.loc[row, "estate_income"] = 0.0

    result = with_us_qbi_input_reconciliation(_frame(person)).table("person")

    assert not result.loc[row, "business_is_sstb"]
    assert result.loc[row, "self_employment_income_before_lsr"] == 500.0
    assert result.loc[row, "sstb_self_employment_income_before_lsr"] == 0.0


def test_asec_channel_policy_is_explicit_and_reconciliation_is_idempotent() -> None:
    first = with_us_qbi_input_reconciliation(_frame(_qbi_person()))
    second = with_us_qbi_input_reconciliation(first)
    asec = second.table("person").query("person_support_channel == 'asec'")

    assert not asec["business_is_sstb"].any()
    assert not asec["sstb_self_employment_income_would_be_qualified"].any()
    for column in US_QBI_BOOLEAN_OUTPUT_COLUMNS:
        if column not in {
            "business_is_sstb",
            "sstb_self_employment_income_would_be_qualified",
        }:
            assert asec[column].all()
    pd.testing.assert_frame_equal(first.table("person"), second.table("person"))


def test_signal_gate_passes_reconciled_nondefault_family() -> None:
    reconciled = with_us_qbi_input_reconciliation(_frame(_qbi_person()))

    result = us_qbi_inputs_signal_gate(reconciled)

    assert result.passed, result.failures
    assert all(value == 0 for value in result.details["invariants"].values())


def test_signal_gate_reports_missing_negative_and_identity_failures() -> None:
    person = _qbi_person()
    person.loc[100, "qualified_bdc_income"] = -1.0
    person.loc[100, "business_is_sstb"] = True
    person.loc[100, "self_employment_income_before_lsr"] = 100.0
    person.loc[100, "sstb_self_employment_income_before_lsr"] = 0.0
    frame = _frame(person)

    result = us_qbi_inputs_signal_gate(frame)

    assert not result.passed
    assert any("qualified_bdc_income" in failure for failure in result.failures)
    assert any(
        "sstb_rows_with_non_sstb_income" in failure for failure in result.failures
    )

    missing = person.drop(columns=["qualified_reit_and_ptp_income"])
    missing_result = us_qbi_inputs_signal_gate(_frame(missing))
    assert not missing_result.passed
    assert "qualified_reit_and_ptp_income" in missing_result.failures[0]


def test_reconciliation_rejects_wrong_schema_and_missing_inputs() -> None:
    wrong = Frame(
        {
            "person": pd.DataFrame({"person_id": [1], "person_household_id": [1]}),
            "household": pd.DataFrame({"household_id": [1]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(1), WeightKind.DESIGN)},
    )
    with pytest.raises(ValueError, match="US schema"):
        with_us_qbi_input_reconciliation(wrong)

    with pytest.raises(ValueError, match="requires PUF-imputed"):
        with_us_qbi_input_reconciliation(
            _frame(_qbi_person().drop(columns=["qualified_bdc_income"]))
        )


def test_summary_does_not_mutate_source_frame() -> None:
    reconciled = with_us_qbi_input_reconciliation(_frame(_qbi_person()))
    before = reconciled.table("person").copy(deep=True)

    summary = us_qbi_inputs_summary(reconciled)

    assert set(summary) == {"columns", "invariants"}
    pd.testing.assert_frame_equal(before, reconciled.table("person"))


@requires_us
def test_all_qbi_outputs_are_live_policyengine_us_input_leaves() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    for column in US_QBI_OUTPUT_COLUMNS:
        variable = variables[column]
        assert variable.entity.key == "person"
        assert not variable.formulas
        assert getattr(variable, "adds", None) is None
        assert getattr(variable, "subtracts", None) is None
