"""Tests split from packages/microcosm-build/tests/test_us_educator_expenses.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_educator_expenses import *


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
