"""Tests split from packages/microcosm-build/tests/test_us_misc_itemized.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_misc_itemized import *


def test_archived_e20400_proxy_is_an_exact_carry() -> None:
    source = pd.DataFrame({"E20400": [0.0, 125.5, 9_000.0]})

    result = derive_us_misc_itemized_from_puf(source)

    assert "unreimbursed_business_employee_expenses" not in source.columns
    assert result["unreimbursed_business_employee_expenses"].tolist() == [
        0.0,
        125.5,
        9_000.0,
    ]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame({"other": [1.0]}), "requires source column"),
        (pd.DataFrame({"E20400": ["not numeric"]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E20400": [np.inf]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E20400": [-1.0]}), "negative value"),
    ],
)
def test_source_derivation_fails_closed(
    source: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_us_misc_itemized_from_puf(source)


def test_post_disaggregation_reconciliation_uses_final_e20400() -> None:
    output = "unreimbursed_business_employee_expenses"
    source = pd.DataFrame(
        {
            "E20400": [0.0, 3_000.0, 7_500.0],
            output: [999.0, 999.0, 999.0],
        }
    )

    result = _reconcile_puf_misc_itemized_from_source(source)

    assert result[output].tolist() == [0.0, 3_000.0, 7_500.0]


def test_shared_puf_stage_declares_exact_source_and_output() -> None:
    stage = us_misc_itemized_stage_spec()
    operation = next(
        operation
        for operation in stage.operations
        if operation.kind == "derive_puf_policyengine_variables"
    )

    assert (
        operation.parameters["unreimbursed_business_employee_expenses_source"]
        == "E20400"
    )
    assert (
        operation.parameters["unreimbursed_business_employee_expenses_output"]
        == "unreimbursed_business_employee_expenses"
    )
    assert "unreimbursed_business_employee_expenses" in stage.outputs
    assert "unreimbursed_business_employee_expenses" in stage.nonnegative_outputs
    assert any("E20400" in str(artifact.get("locator")) for artifact in stage.artifacts)


def test_puf_support_keeps_misc_itemized_dense_and_nonnegative() -> None:
    output = "unreimbursed_business_employee_expenses"
    assert output in puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert output in puf_support_module._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
    assert output not in puf_support_module._PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS


def test_signal_gate_accepts_reference_like_nondefault_values() -> None:
    values = np.zeros(100)
    values[:30] = np.arange(1, 31, dtype=np.float64) * 100.0
    frame = _PersonFrame(
        pd.DataFrame({"unreimbursed_business_employee_expenses": values})
    )

    result = us_misc_itemized_signal_gate(frame)  # type: ignore[arg-type]

    assert result.passed, result.failures
    assert result.details["positive_share"] == pytest.approx(0.30)


@pytest.mark.parametrize(
    "person",
    [
        pd.DataFrame({"other": [0.0, 1.0]}),
        pd.DataFrame({"unreimbursed_business_employee_expenses": [0.0, 0.0]}),
        pd.DataFrame({"unreimbursed_business_employee_expenses": [0.0, -1.0]}),
        pd.DataFrame({"unreimbursed_business_employee_expenses": [0.0, np.nan]}),
    ],
)
def test_signal_gate_rejects_missing_default_or_invalid_surface(
    person: pd.DataFrame,
) -> None:
    result = us_misc_itemized_signal_gate(  # type: ignore[arg-type]
        _PersonFrame(person)
    )

    assert not result.passed
