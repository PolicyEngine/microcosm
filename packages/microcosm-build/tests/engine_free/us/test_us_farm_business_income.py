"""Tests split from packages/microcosm-build/tests/test_us_farm_business_income.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_farm_business_income import *


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
    # farm_operations_income now carries the signed-mass calibration: each
    # PUF-channel leg is rescaled so its per-unit-weight mass equals the donor's
    # (positive 40/4 = 10.0, negative -20/4 = -5.0), pinning the imputed net to
    # the source. The raw draw [700, -400, 900, -500] at PUF weight 0.5 has
    # per-weight legs 80.0 / -45.0, so the positive leg scales by 10/80 = 0.125
    # and the negative by 5/45 = 1/9; the signs and the measured ASEC leg are
    # untouched. farm_rent_income is not signed-mass calibrated.
    assert puf[_OPERATIONS].tolist()[:4] == pytest.approx(
        [87.5, -400.0 / 9.0, 112.5, -500.0 / 9.0]
    )
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
