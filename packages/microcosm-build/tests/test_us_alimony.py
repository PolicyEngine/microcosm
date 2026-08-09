"""Contracts for the retired eCPS alimony input family."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime.alimony import (
    ALIMONY_ASEC_ARCHIVED_DERIVATION_URL,
    ALIMONY_PUF_ARCHIVED_DERIVATION_URL,
    STRIKE_BENEFITS_ASEC_ARCHIVED_DERIVATION_URL,
    derive_us_alimony_from_asec,
    derive_us_alimony_from_puf,
    us_alimony_signal_gate,
    us_alimony_stage_spec,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_alimony_from_sources,
)

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)


class _ResolvedWeights:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _PersonFrame:
    def __init__(self, person: pd.DataFrame) -> None:
        self._person = person

    def table(self, entity: str) -> pd.DataFrame:
        assert entity == "person"
        return self._person

    def resolve_weights(self, entity: str) -> _ResolvedWeights:
        assert entity == "person"
        return _ResolvedWeights(np.ones(len(self._person)))


def test_archived_coordinates_are_commit_and_line_pinned() -> None:
    for url in (
        ALIMONY_ASEC_ARCHIVED_DERIVATION_URL,
        ALIMONY_PUF_ARCHIVED_DERIVATION_URL,
        STRIKE_BENEFITS_ASEC_ARCHIVED_DERIVATION_URL,
    ):
        assert "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe" in url
        assert "#L" in url


def test_asec_mapping_splits_alimony_and_miscellaneous_income_exactly() -> None:
    source = pd.DataFrame(
        {
            "OI_OFF": [20, 12, 19, 0],
            "OI_VAL": [5_000.0, 700.0, 900.0, 0.0],
        }
    )

    result = derive_us_alimony_from_asec(source)

    assert result["alimony_income"].tolist() == [5_000.0, 0.0, 0.0, 0.0]
    assert result["strike_benefits"].tolist() == [0.0, 700.0, 0.0, 0.0]
    assert result["miscellaneous_income"].tolist() == [0.0, 0.0, 900.0, 0.0]
    assert all(
        result[column].dtype == np.dtype("float64")
        for column in (
            "alimony_income",
            "strike_benefits",
            "miscellaneous_income",
        )
    )
    assert "alimony_income" not in source
    assert "strike_benefits" not in source
    assert "miscellaneous_income" not in source


def test_strike_benefits_matches_archived_ecps_mapping_semantics() -> None:
    source = pd.DataFrame(
        {
            "OI_OFF": [12, 20, 19, 12, 0],
            "OI_VAL": [700.0, 5_000.0, 900.0, 0.0, 0.0],
        }
    )

    result = derive_us_alimony_from_asec(source)
    archived_ecps = (source["OI_OFF"] == 12) * source["OI_VAL"]

    np.testing.assert_array_equal(result["strike_benefits"], archived_ecps)


def test_asec_other_income_split_conserves_oi_val_per_person() -> None:
    source = pd.DataFrame(
        {
            "OI_OFF": [12, 20, 19, 0, 7],
            "OI_VAL": [700.0, 5_000.0, 900.0, 0.0, 125.5],
        }
    )

    result = derive_us_alimony_from_asec(source)
    split_total = result.loc[
        :, ["alimony_income", "strike_benefits", "miscellaneous_income"]
    ].sum(axis="columns")

    np.testing.assert_array_equal(split_total, source["OI_VAL"])


def test_asec_mapping_preserves_an_already_materialized_split() -> None:
    source = pd.DataFrame(
        {
            "alimony_income": [10.0],
            "strike_benefits": [30.0],
            "miscellaneous_income": [20.0],
        }
    )

    result = derive_us_alimony_from_asec(source)

    assert result.equals(source)
    assert result is not source


def test_asec_mapping_replaces_partial_stale_miscellaneous_carry() -> None:
    source = pd.DataFrame(
        {
            "OI_OFF": [20, 19],
            "OI_VAL": [5_000.0, 700.0],
            "miscellaneous_income": [5_000.0, 999.0],
        }
    )

    result = derive_us_alimony_from_asec(source)

    assert result["alimony_income"].tolist() == [5_000.0, 0.0]
    assert result["miscellaneous_income"].tolist() == [0.0, 700.0]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame({"OI_VAL": [1.0]}), "requires both raw source columns"),
        (
            pd.DataFrame({"OI_VAL": [1.0], "OI_OFF": [np.inf]}),
            "nonnumeric or nonfinite",
        ),
        (
            pd.DataFrame({"OI_VAL": [-1.0], "OI_OFF": [20]}),
            "negative value",
        ),
        (
            pd.DataFrame({"OI_VAL": [1.0], "OI_OFF": [20.5]}),
            "noninteger code",
        ),
    ],
)
def test_asec_mapping_fails_closed(source: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        derive_us_alimony_from_asec(source)


def test_puf_mapping_is_an_exact_two_field_carry() -> None:
    source = pd.DataFrame(
        {
            "E00800": [0.0, 125.5, 9_000.0],
            "E03500": [40.0, 0.0, 700.0],
        }
    )

    result = derive_us_alimony_from_puf(source)

    assert result["alimony_income"].tolist() == [0.0, 125.5, 9_000.0]
    assert result["alimony_expense"].tolist() == [40.0, 0.0, 700.0]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame({"E00800": [1.0]}), "requires source column 'E03500'"),
        (
            pd.DataFrame({"E00800": [np.nan], "E03500": [0.0]}),
            "nonnumeric or nonfinite",
        ),
        (
            pd.DataFrame({"E00800": [0.0], "E03500": [-1.0]}),
            "negative value",
        ),
    ],
)
def test_puf_mapping_fails_closed(source: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        derive_us_alimony_from_puf(source)


def test_post_disaggregation_reconciliation_uses_final_raw_fields() -> None:
    source = pd.DataFrame(
        {
            "E00800": [0.0, 3_000.0],
            "E03500": [2_000.0, 0.0],
            "alimony_income": [999.0, 999.0],
            "alimony_expense": [999.0, 999.0],
        }
    )

    result = _reconcile_puf_alimony_from_sources(source)

    assert result["alimony_income"].tolist() == [0.0, 3_000.0]
    assert result["alimony_expense"].tolist() == [2_000.0, 0.0]


def test_shared_puf_stage_declares_exact_sources_and_outputs() -> None:
    stage = us_alimony_stage_spec()
    operation = next(
        operation
        for operation in stage.operations
        if operation.kind == "derive_puf_policyengine_variables"
    )

    assert operation.parameters["alimony_income_source"] == "E00800"
    assert operation.parameters["alimony_income_output"] == "alimony_income"
    assert operation.parameters["alimony_expense_source"] == "E03500"
    assert operation.parameters["alimony_expense_output"] == "alimony_expense"
    assert set(("alimony_income", "alimony_expense")) <= set(stage.outputs)
    assert set(("alimony_income", "alimony_expense")) <= set(stage.nonnegative_outputs)


def test_puf_support_marks_both_leaves_sparse_nonnegative_and_preserves_income() -> (
    None
):
    outputs = {"alimony_income", "alimony_expense"}
    assert outputs <= set(puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS)
    assert outputs <= puf_support_module._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
    assert outputs <= puf_support_module._PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS
    assert "alimony_income" in (
        puf_support_module._PUF_TAX_DETAIL_PRESERVE_BASE_ASEC_OUTPUTS
    )
    assert "alimony_expense" not in (
        puf_support_module._PUF_TAX_DETAIL_PRESERVE_BASE_ASEC_OUTPUTS
    )


def test_sparse_puf_splice_does_not_prune_reported_asec_alimony_income() -> None:
    person = pd.DataFrame(
        {
            "person_tax_unit_id": [1, 2, 3, 4],
            "person_household_id": [1, 2, 3, 4],
            "person_support_channel": [
                "asec",
                "asec",
                "puf_tax_detail",
                "puf_tax_detail",
            ],
            "alimony_income": [1_000.0, 0.0, 500.0, 250.0],
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": [1, 2, 3, 4]}),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": [1, 2, 3, 4],
                "tax_unit_support_channel": [
                    "asec",
                    "asec",
                    "puf_tax_detail",
                    "puf_tax_detail",
                ],
            }
        ),
    }

    puf_support_module._sparsify_tax_unit_person_output_to_donor_positive_rate(
        tables,
        column="alimony_income",
        donor_positive_rate=0.0,
        household_weights=np.ones(4),
        person_channel="person_support_channel",
        tax_unit_channel="tax_unit_support_channel",
    )

    assert person.loc[:1, "alimony_income"].tolist() == [1_000.0, 0.0]
    assert person.loc[2:, "alimony_income"].tolist() == [750.0, 0.0]


def test_signal_gate_accepts_reference_like_sparse_values() -> None:
    income = np.zeros(1_000)
    expense = np.zeros(1_000)
    income[10] = 2_000.0
    expense[20] = 3_000.0
    frame = _PersonFrame(
        pd.DataFrame({"alimony_income": income, "alimony_expense": expense})
    )

    result = us_alimony_signal_gate(frame)  # type: ignore[arg-type]

    assert result.passed, result.failures
    assert result.details["columns"]["alimony_income"][
        "positive_share"
    ] == pytest.approx(0.001)


def test_signal_gate_rejects_asec_alimony_left_in_miscellaneous_income() -> None:
    n = 1_000
    codes = np.zeros(n)
    amounts = np.zeros(n)
    income = np.zeros(n)
    expense = np.zeros(n)
    miscellaneous = np.zeros(n)
    codes[10] = 20
    amounts[10] = 2_000.0
    income[10] = 2_000.0
    miscellaneous[10] = 2_000.0
    expense[20] = 3_000.0
    frame = _PersonFrame(
        pd.DataFrame(
            {
                "OI_OFF": codes,
                "OI_VAL": amounts,
                "alimony_income": income,
                "alimony_expense": expense,
                "strike_benefits": np.zeros(n),
                "miscellaneous_income": miscellaneous,
            }
        )
    )

    result = us_alimony_signal_gate(frame)  # type: ignore[arg-type]

    assert not result.passed
    assert any(
        "miscellaneous_income does not exclude alimony/strike" in failure
        for failure in result.failures
    )


def test_signal_gate_rejects_discarded_asec_strike_benefits() -> None:
    n = 1_000
    codes = np.zeros(n)
    amounts = np.zeros(n)
    income = np.zeros(n)
    expense = np.zeros(n)
    strike_benefits = np.zeros(n)
    miscellaneous = np.zeros(n)
    codes[10] = 12
    amounts[10] = 2_000.0
    expense[20] = 3_000.0
    frame = _PersonFrame(
        pd.DataFrame(
            {
                "OI_OFF": codes,
                "OI_VAL": amounts,
                "alimony_income": income,
                "alimony_expense": expense,
                "strike_benefits": strike_benefits,
                "miscellaneous_income": miscellaneous,
            }
        )
    )

    result = us_alimony_signal_gate(frame)  # type: ignore[arg-type]

    assert not result.passed
    assert any(
        "strike_benefits disagrees with OI_OFF == 12" in failure
        for failure in result.failures
    )
    assert any("does not conserve OI_VAL" in failure for failure in result.failures)


@pytest.mark.parametrize(
    "person",
    [
        pd.DataFrame({"alimony_income": [1.0]}),
        pd.DataFrame({"alimony_income": [0.0], "alimony_expense": [0.0]}),
        pd.DataFrame({"alimony_income": [-1.0], "alimony_expense": [1.0]}),
        pd.DataFrame({"alimony_income": [1.0], "alimony_expense": [np.nan]}),
    ],
)
def test_signal_gate_rejects_missing_default_or_invalid_surface(
    person: pd.DataFrame,
) -> None:
    result = us_alimony_signal_gate(_PersonFrame(person))  # type: ignore[arg-type]

    assert not result.passed


@requires_us
def test_policyengine_us_contract_includes_strike_person_year_input_leaf() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    for name in ("alimony_income", "alimony_expense", "strike_benefits"):
        variable = variables[name]
        assert variable.is_input_variable()
        assert variable.entity.key == "person"
        assert str(variable.definition_period).lower() == "year"
        assert variable.default_value == 0

    assert "alimony_expense" in (
        variables["alimony_expense_ald"].formula.__code__.co_consts
    )
