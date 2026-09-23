"""Tests split from packages/microcosm-build/tests/test_us_casualty_losses.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_casualty_losses import *


def test_archived_puf_mapping_is_an_exact_carry() -> None:
    source = pd.DataFrame({"E20500": [0.0, 125.5, 9_000.0]})

    result = derive_us_casualty_loss_from_puf(source)

    assert "casualty_loss" not in source.columns
    assert result["casualty_loss"].tolist() == [0.0, 125.5, 9_000.0]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame({"other": [1.0]}), "requires source column"),
        (pd.DataFrame({"E20500": ["not numeric"]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E20500": [np.inf]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E20500": [-1.0]}), "negative value"),
    ],
)
def test_source_derivation_fails_closed(
    source: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_us_casualty_loss_from_puf(source)


def test_post_disaggregation_reconciliation_uses_final_e20500() -> None:
    source = pd.DataFrame(
        {
            "E20500": [0.0, 3_000.0, 7_500.0],
            "casualty_loss": [999.0, 999.0, 999.0],
        }
    )

    result = _reconcile_puf_casualty_loss_from_source(source)

    assert result["casualty_loss"].tolist() == [0.0, 3_000.0, 7_500.0]


def test_shared_puf_stage_declares_exact_source_and_output() -> None:
    stage = us_casualty_loss_stage_spec()
    operation = next(
        operation
        for operation in stage.operations
        if operation.kind == "derive_puf_policyengine_variables"
    )

    assert operation.parameters["casualty_loss_source"] == "E20500"
    assert operation.parameters["casualty_loss_output"] == "casualty_loss"
    assert "casualty_loss" in stage.outputs
    assert "casualty_loss" in stage.nonnegative_outputs
    assert any("E20500" in str(artifact.get("locator")) for artifact in stage.artifacts)


def test_puf_support_keeps_casualty_loss_sparse_and_earnings_distributed() -> None:
    assert "casualty_loss" in puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "casualty_loss" in puf_support_module._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
    assert "casualty_loss" in puf_support_module._PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS
    assert puf_support_module._PERSON_OUTPUT_DISTRIBUTION_BASIS["casualty_loss"] == (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
    )


def test_signal_gate_accepts_plausibly_sparse_nondefault_values() -> None:
    values = np.zeros(1_000)
    values[[10, 20, 30]] = [1_000.0, 2_000.0, 3_000.0]
    frame = _PersonFrame(pd.DataFrame({"casualty_loss": values}))

    result = us_casualty_loss_signal_gate(frame)  # type: ignore[arg-type]

    assert result.passed, result.failures
    assert result.details["positive_share"] == pytest.approx(0.003)


@pytest.mark.parametrize(
    "person",
    [
        pd.DataFrame({"other": [0.0, 1.0]}),
        pd.DataFrame({"casualty_loss": [0.0, 0.0]}),
        pd.DataFrame({"casualty_loss": [0.0, -1.0]}),
        pd.DataFrame({"casualty_loss": [0.0, np.nan]}),
    ],
)
def test_signal_gate_rejects_missing_default_or_invalid_surface(
    person: pd.DataFrame,
) -> None:
    result = us_casualty_loss_signal_gate(  # type: ignore[arg-type]
        _PersonFrame(person)
    )

    assert not result.passed
