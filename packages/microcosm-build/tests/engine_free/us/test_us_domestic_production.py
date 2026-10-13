"""Tests split from packages/microcosm-build/tests/test_us_domestic_production.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_domestic_production import *


def test_archived_coordinates_are_immutable_and_cover_the_full_source_path() -> None:
    commit = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
    for url in (
        DOMESTIC_PRODUCTION_ALD_ARCHIVED_DERIVATION_URL,
        DOMESTIC_PRODUCTION_ALD_ARCHIVED_EXPORT_URL,
        DOMESTIC_PRODUCTION_ALD_ARCHIVED_IMPUTATION_URL,
        DOMESTIC_PRODUCTION_ALD_ARCHIVED_PUF_ARTIFACT_URL,
    ):
        assert commit in url
    assert DOMESTIC_PRODUCTION_ALD_ARCHIVED_DERIVATION_URL.endswith(
        "datasets/puf/puf.py#L646"
    )
    assert DOMESTIC_PRODUCTION_ALD_ARCHIVED_EXPORT_URL.endswith(
        "datasets/puf/puf.py#L808-L815"
    )
    assert DOMESTIC_PRODUCTION_ALD_ARCHIVED_IMPUTATION_URL.endswith(
        "calibration/puf_impute.py#L90-L198"
    )
    assert DOMESTIC_PRODUCTION_ALD_ARCHIVED_PUF_ARTIFACT_URL.endswith(
        "datasets/puf/puf.py#L1655-L1660"
    )


def test_archived_e03240_mapping_is_an_exact_carry() -> None:
    source = pd.DataFrame({"E03240": [0.0, 125.5, 9_000.0]})

    result = derive_us_domestic_production_ald_from_puf(source)

    assert "domestic_production_ald" not in source.columns
    assert result["domestic_production_ald"].tolist() == [0.0, 125.5, 9_000.0]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame({"other": [1.0]}), "requires source column"),
        (pd.DataFrame({"E03240": ["not numeric"]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E03240": [np.inf]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E03240": [-1.0]}), "negative value"),
    ],
)
def test_source_derivation_fails_closed(
    source: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_us_domestic_production_ald_from_puf(source)


def test_post_disaggregation_reconciliation_uses_final_e03240() -> None:
    source = pd.DataFrame(
        {
            "E03240": [0.0, 3_000.0, 7_500.0],
            "domestic_production_ald": [999.0, 999.0, 999.0],
        }
    )

    result = _reconcile_puf_domestic_production_ald_from_source(source)

    assert result["domestic_production_ald"].tolist() == [0.0, 3_000.0, 7_500.0]


def test_shared_puf_stage_declares_exact_source_and_output() -> None:
    stage = us_domestic_production_ald_stage_spec()
    operation = next(
        operation
        for operation in stage.operations
        if operation.kind == "derive_puf_policyengine_variables"
    )

    assert operation.parameters["domestic_production_ald_source"] == "E03240"
    assert (
        operation.parameters["domestic_production_ald_output"]
        == "domestic_production_ald"
    )
    assert "domestic_production_ald" in stage.outputs
    assert "domestic_production_ald" in stage.nonnegative_outputs
    assert any("E03240" in str(artifact.get("locator")) for artifact in stage.artifacts)


def test_puf_support_keeps_input_at_tax_unit_grain_and_sparse() -> None:
    output = "domestic_production_ald"
    assert output in puf_support_module.PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    assert output not in puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert output in puf_support_module._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
    assert output in puf_support_module._PUF_TAX_DETAIL_SPARSE_TAX_UNIT_OUTPUTS
    assert output not in puf_support_module._PUF_TAX_DETAIL_DISCRETE_TAX_UNIT_OUTPUTS


def test_tax_unit_sparsifier_preserves_weighted_total_and_source_channel() -> None:
    tables = {
        "household": pd.DataFrame({"household_id": [1, 2, 3, 4, 5]}),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": [10, 20, 30, 40, 50],
                "tax_unit_support_channel": [
                    "puf_tax_detail",
                    "puf_tax_detail",
                    "puf_tax_detail",
                    "puf_tax_detail",
                    "asec",
                ],
                "domestic_production_ald": [100.0, 200.0, 300.0, 400.0, 0.0],
            }
        ),
        "person": pd.DataFrame(
            {
                "person_tax_unit_id": [10, 20, 30, 40, 50],
                "person_household_id": [1, 2, 3, 4, 5],
            }
        ),
    }

    puf_support_module._sparsify_tax_unit_output_to_donor_positive_rate(
        tables,
        column="domestic_production_ald",
        donor_positive_rate=0.25,
        household_weights=np.ones(5),
        tax_unit_channel="tax_unit_support_channel",
    )

    values = tables["tax_unit"]["domestic_production_ald"]
    assert int((values.iloc[:4] > 0.0).sum()) == 1
    assert values.iloc[:4].sum() == pytest.approx(1_000.0)
    assert values.iloc[4] == 0.0


def test_signal_gate_accepts_plausibly_sparse_nondefault_values() -> None:
    values = np.zeros(1_000)
    values[[10, 20, 30]] = [1_000.0, 2_000.0, 3_000.0]
    frame = _TaxUnitFrame(pd.DataFrame({"domestic_production_ald": values}))

    result = us_domestic_production_ald_signal_gate(frame)  # type: ignore[arg-type]

    assert result.passed, result.failures
    assert result.details["positive_share"] == pytest.approx(0.003)


def test_signal_gate_rejects_positive_values_on_the_asec_support_channel() -> None:
    values = np.zeros(1_000)
    values[[10, 20, 30]] = [1_000.0, 2_000.0, 3_000.0]
    channels = np.full(1_000, "puf_tax_detail", dtype=object)
    channels[10] = "asec"
    frame = _TaxUnitFrame(
        pd.DataFrame(
            {
                "domestic_production_ald": values,
                "tax_unit_support_channel": channels,
            }
        )
    )

    result = us_domestic_production_ald_signal_gate(frame)  # type: ignore[arg-type]

    assert not result.passed
    assert any("ASEC support channel" in failure for failure in result.failures)


@pytest.mark.parametrize(
    "tax_unit",
    [
        pd.DataFrame({"other": [0.0, 1.0]}),
        pd.DataFrame({"domestic_production_ald": [0.0, 0.0]}),
        pd.DataFrame({"domestic_production_ald": [0.0, -1.0]}),
        pd.DataFrame({"domestic_production_ald": [0.0, np.nan]}),
    ],
)
def test_signal_gate_rejects_missing_default_or_invalid_surface(
    tax_unit: pd.DataFrame,
) -> None:
    result = us_domestic_production_ald_signal_gate(  # type: ignore[arg-type]
        _TaxUnitFrame(tax_unit)
    )

    assert not result.passed
