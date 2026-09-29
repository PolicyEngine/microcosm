"""Tests split from packages/microcosm-build/tests/test_uk_spi_income.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_spi_income import *


def test_spi_preserves_observed_children_outside_the_donor_age_domain(
    monkeypatch, tmp_path
) -> None:
    """Preserve FRS inputs below the constructed donor-age support guard."""
    support = _dead_support()
    person = support.person.copy()
    channel = support_channel_column("person")
    child = person.index[person[channel].eq("spi")][0]
    person.loc[child, "age"] = 15
    preserved = [
        "employment_income",
        "self_employment_income",
        "private_pension_income",
        "state_pension_reported",
        "universal_credit_reported",
        "savings_interest_income",
        "tax_free_savings_income",
        "dla_sc_reported",
    ]
    person.loc[child, preserved] = [0, 0, 0, 0, 0, 12, 3, 42]
    support = replace(support, person=person)
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    result = impute_uk_spi_income_support(
        support, donor_path, seed=9, n_estimators=3, donor_sample_size=None
    )

    pd.testing.assert_series_equal(
        result.person.loc[child, preserved], person.loc[child, preserved]
    )
    assert result.spi_prediction_rows == int(person[channel].eq("spi").sum()) - 1
    adults = person[channel].eq("spi") & person.age.ge(16)
    assert result.person.loc[adults, "gift_aid"].eq(10).all()


def test_spi_rebases_income_before_frs_fill_and_preserves_child_inputs(
    monkeypatch, tmp_path
) -> None:
    support = _dead_support()
    person = support.person.copy()
    channel = support_channel_column("person")
    base_child = person.index[person[channel].eq("frs")][0]
    person.loc[base_child, "age"] = 15
    person.loc[base_child, "dividend_income"] = 19.0
    support = replace(support, person=person)
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)
    factors = dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 2.0)
    monkeypatch.setattr(
        spi_income,
        "_spi_income_uprating_factors",
        lambda year: (factors, {"from_period": 2022, "to_period": year}),
        raising=False,
    )
    options = dict(
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
        stage1_base_redraw_columns=("dividend_income",),
    )
    reference = impute_uk_spi_income_support(support, donor_path, **options)
    result = impute_uk_spi_income_support(
        support,
        donor_path,
        rebase_income_to_build_period=True,
        build_period=2024,
        **options,
    )
    spi = person[channel].eq("spi")
    np.testing.assert_array_equal(
        result.person.loc[spi, "employment_income"],
        2 * reference.person.loc[spi, "employment_income"],
    )
    # Only taxable SPI interest is rebased; the FRS fill's £5 tax-free draw
    # is already in the build-year basis and must not be doubled.
    assert result.person.loc[spi, "savings_interest_income"].eq(205).all()
    assert (
        result.person.loc[base_child, "dividend_income"]
        == person.loc[base_child, "dividend_income"]
    )
    base_adult = person[channel].eq("frs") & person.age.ge(16)
    np.testing.assert_array_equal(
        result.person.loc[base_adult, "dividend_income"],
        2 * reference.person.loc[base_adult, "dividend_income"],
    )
    assert result.income_uprating == {"from_period": 2022, "to_period": 2024}


def test_child_exclusion_keeps_the_base_dividend_random_stream(
    monkeypatch, tmp_path
) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    original_predict = _FakeFittedQRF.predict

    def predict(self, predictors):
        result = original_predict(self, predictors)
        if "dividend_income" in result:
            consumed = getattr(self, "consumed", 0)
            result["dividend_income"] = consumed + np.arange(len(result), dtype=float)
            self.consumed = consumed + len(result)
        return result

    monkeypatch.setattr(_FakeFittedQRF, "predict", predict)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)
    options = dict(
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
        stage1_base_redraw_columns=("dividend_income",),
    )
    reference = impute_uk_spi_income_support(support, donor_path, **options)
    person = support.person.copy()
    channel = support_channel_column("person")
    child = person.index[person[channel].eq("spi")][0]
    person.loc[child, "age"] = 15
    candidate = impute_uk_spi_income_support(
        replace(support, person=person), donor_path, **options
    )
    base = person[channel].eq("frs")
    np.testing.assert_array_equal(
        reference.person.loc[base, "dividend_income"],
        candidate.person.loc[base, "dividend_income"],
    )


def test_stage2_pension_bridge_uses_observed_and_drawn_receipt(
    monkeypatch, tmp_path
) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    captured = {}
    original_predict = _FakeFittedQRF.predict

    def predict(self, predictors):
        result = original_predict(self, predictors)
        if SPI_HMRC_STATE_PENSION_INCOME_COLUMN in result:
            result[SPI_HMRC_STATE_PENSION_INCOME_COLUMN] = np.arange(len(result)) % 2
        if "state_pension_reported" in result:
            captured["recipient"] = predictors["state_pension_receipt"].to_numpy()
        return result

    class CapturingQRF(_FakeQRF):
        def fit(self, frame, predictors, targets, *, weights):
            if "state_pension_reported" in targets:
                captured["training"] = frame.table("person")[
                    "state_pension_receipt"
                ].to_numpy()
            return super().fit(frame, predictors, targets, weights=weights)

    monkeypatch.setattr(_FakeFittedQRF, "predict", predict)
    monkeypatch.setattr(spi_income, "QRF", CapturingQRF)
    _bypass_reviewed_donor_identity(monkeypatch)
    result = impute_uk_spi_income_support(
        support,
        donor_path,
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
        condition_on_state_pension_receipt=True,
    )
    np.testing.assert_array_equal(
        captured["recipient"], np.arange(result.spi_prediction_rows) % 2
    )
    assert captured["training"].all()
    assert (
        result.pension_receipt_bridge["recipient_source"]
        == "hmrc_spi_state_pension_income > 0"
    )


def test_spi_qrf_stages_use_typed_weights_and_restore_gross_savings(
    monkeypatch,
    tmp_path,
) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    _FakeQRF.fit_weight_kinds = []
    _FakeQRF.fit_weight_values = []
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    before = support.person.copy()
    result = impute_uk_spi_income_support(
        support,
        donor_path,
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
    )

    assert _FakeQRF.fit_weight_kinds == ["design", "importance"]
    assert [record.weight_kind for record in result.fit_weight_records] == [
        "design",
        "importance",
    ]
    assert result.reviewed_absent_stage2_outputs == (SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS)
    assert result.donor_rows == 4
    assert len(result.donor_sha256) == 64
    assert HMRC_SPI_ASSESSABLE_INCOME_COLUMN not in SPI_INCOME_QRF_OUTPUT_COLUMNS

    channel = support_channel_column("person")
    spi_people = result.person[channel] == "spi"
    base_people = ~spi_people
    assert result.person.loc[spi_people, "savings_interest_income"].eq(105.0).all()
    assert result.person.loc[spi_people, "other_investment_income"].eq(25.0).all()
    assert result.person.loc[spi_people, "gift_aid"].eq(10.0).all()
    assert result.person.loc[spi_people, "charitable_investment_gifts"].eq(2.0).all()
    assert (
        result.person.loc[
            spi_people,
            SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
        ]
        .eq(-3.0)
        .all()
    )
    assert result.person.loc[spi_people, "is_disabled_for_benefits"].all()
    assert {
        "aa_category",
        "dla_sc_category",
        "dla_m_category",
        "pip_m_category",
        "pip_dl_category",
    }.issubset(result.person.columns)
    pd.testing.assert_frame_equal(
        result.person.loc[base_people, before.columns],
        before.loc[base_people],
    )
    for subset in (
        FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
        FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
    ):
        pd.testing.assert_series_equal(result.person[subset], before[subset])

    unavailable_on_frs = (
        *spi_income.FRS_HMRC_UNAVAILABLE_FULL_CONCEPT_COLUMNS,
        SPI_HMRC_EMPLOYED_INCOME_COLUMN,
        SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
        SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
        HMRC_SPI_ASSESSABLE_INCOME_COLUMN,
    )
    # These full concepts are unmeasured on the FRS instrument, so the FRS
    # channel carries the adjudicated stage-time zero rather than NaN — the
    # artifact must load through an engine that refuses NaN inputs, and the
    # calibration seam's finiteness fence stays fail-loud because of it.
    assert result.person.loc[base_people, list(unavailable_on_frs)].eq(0.0).all().all()

    expected_employed = (
        np.maximum(
            result.person.loc[spi_people, SPI_HMRC_PAY_COLUMN]
            + result.person.loc[spi_people, SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN]
            - result.person.loc[spi_people, SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN],
            0.0,
        )
        + result.person.loc[spi_people, SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN]
        + result.person.loc[
            spi_people,
            SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
        ]
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, SPI_HMRC_EMPLOYED_INCOME_COLUMN],
        expected_employed,
    )
    expected_pe_employment = (
        result.person.loc[spi_people, SPI_HMRC_PAY_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN]
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, "employment_income"],
        expected_pe_employment,
    )
    expected_total_earned = (
        expected_employed
        + result.person.loc[spi_people, SPI_HMRC_OTHER_INCOME_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_STATE_PENSION_INCOME_COLUMN]
        + result.person.loc[spi_people, "self_employment_income"]
        + result.person.loc[spi_people, "private_pension_income"]
    )
    expected_total_investment = (
        result.person.loc[spi_people, "savings_interest_income"]
        - result.person.loc[spi_people, "tax_free_savings_income"]
        + result.person.loc[spi_people, "dividend_income"]
        + result.person.loc[spi_people, "property_income"]
        + result.person.loc[spi_people, "other_investment_income"]
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN],
        expected_total_earned,
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN],
        expected_total_investment,
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, HMRC_SPI_ASSESSABLE_INCOME_COLUMN],
        expected_total_earned + expected_total_investment,
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, HMRC_SPI_ASSESSABLE_INCOME_COLUMN],
        result.person.loc[spi_people, SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN],
    )
    assert (
        result.person.loc[
            spi_people,
            SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
        ]
        .lt(0.0)
        .all()
    )

    spi_households = support.household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN]
    assert support.household.loc[spi_households, "household_weight"].gt(0).all()
    assert support.household_weight_kind is WeightKind.IMPORTANCE


def test_spi_stage2_does_not_require_frs_other_investment_income(
    monkeypatch,
    tmp_path,
) -> None:
    support = _dead_support(drop_income_component="other_investment_income")
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    result = impute_uk_spi_income_support(
        support,
        donor_path,
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
    )

    channel = support_channel_column("person")
    spi_people = result.person[channel] == "spi"
    # The FRS channel carries the stage-time zero, not NaN: the column is a
    # full concept the FRS instrument does not measure, and the artifact has
    # to load through an engine that refuses NaN inputs.
    assert result.person.loc[~spi_people, "other_investment_income"].eq(0.0).all()
    assert result.person.loc[spi_people, "other_investment_income"].eq(25.0).all()


def test_spi_weighted_bootstrap_does_not_apply_fact_twice(
    monkeypatch,
    tmp_path,
) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    _FakeQRF.fit_weight_kinds = []
    _FakeQRF.fit_weight_values = []
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    result = impute_uk_spi_income_support(
        support,
        donor_path,
        donor_sample_size=8,
    )

    assert result.donor_rows == 8
    np.testing.assert_array_equal(_FakeQRF.fit_weight_values[0], np.ones(8))
    assert _FakeQRF.fit_weight_kinds[0] == "design"


def test_the_spi_channel_ships_no_structural_nan_on_the_frs_channel(
    monkeypatch,
    tmp_path,
) -> None:
    """The FRS channel carries stage-time zero, never NaN (#747).

    The SPI stage populates full-concept income columns the FRS instrument
    does not measure. Shipping NaN on the FRS rows was assessment-era
    honesty that made the artifact unloadable: the engine's ``validate()``
    refuses NaN inputs, and the calibration seam's finiteness fence refuses
    the frame — so the first armed campaign had to zero-fill twelve person
    columns outside the build before it could calibrate at all. Zero is the
    adjudicated stage-time semantics, and the auxiliary-crosswalk guard
    already stops the QRF mistaking the fill for measured data.
    """

    __import__("policyengine_uk")
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    result = impute_uk_spi_income_support(
        support,
        donor_path,
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
    )

    person = result.person
    nan_columns = sorted(
        column
        for column in person.columns
        if person[column].dtype.kind == "f" and bool(person[column].isna().any())
    )
    assert nan_columns == [], (
        f"the SPI stage left NaN in {nan_columns}; the artifact must ship "
        "stage-time zeros so the engine can load it and the calibration "
        "seam's finiteness fence can stay fail-loud"
    )


@pytest.mark.parametrize("age", [15, 16])
@pytest.mark.parametrize("channel", ["frs", "spi"])
def test_spi_donor_age_boundary_applies_to_both_recipient_channels(
    monkeypatch, tmp_path, age, channel
) -> None:
    support = _dead_support()
    person = support.person.copy()
    recipient = person.index[person[support_channel_column("person")].eq(channel)][0]
    person.loc[recipient, "age"] = age
    person.loc[recipient, "dividend_income"] = 987.0
    person.loc[recipient, "universal_credit_reported"] = 1234.0
    support = replace(support, person=person)
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    result = impute_uk_spi_income_support(
        support,
        donor_path,
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
        stage1_base_redraw_columns=("dividend_income",),
    )

    if age == 15:
        assert result.person.loc[recipient, "dividend_income"] == 987.0
        assert result.person.loc[recipient, "universal_credit_reported"] == 1234.0
    else:
        assert result.person.loc[recipient, "dividend_income"] == 3.0
        if channel == "spi":
            assert result.person.loc[recipient, "universal_credit_reported"] != 1234.0
        else:
            assert result.person.loc[recipient, "universal_credit_reported"] == 1234.0


def test_spi_uprating_uses_actual_engine_indices_and_explicit_nominal_holdouts() -> (
    None
):
    from policyengine_uk import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    source = system.parameters("2022-01-01").gov.economic_assumptions.indices
    target = system.parameters("2024-01-01").gov.economic_assumptions.indices
    expected = {
        "self_employment_income": (
            "obr.per_capita.mixed_income",
            target.obr.per_capita.mixed_income / source.obr.per_capita.mixed_income,
        ),
        "savings_interest_income": (
            "ons.household_interest_income",
            target.ons.household_interest_income / source.ons.household_interest_income,
        ),
        "dividend_income": (
            "obr.per_capita.gdp",
            target.obr.per_capita.gdp / source.obr.per_capita.gdp,
        ),
        "private_pension_income": (
            "obr.private_pension_index",
            target.obr.private_pension_index / source.obr.private_pension_index,
        ),
        "employment_income_before_lsr": (
            "obr.average_earnings",
            target.obr.average_earnings / source.obr.average_earnings,
        ),
    }
    expected["property_income"] = expected["dividend_income"]
    expected["miscellaneous_income"] = expected["dividend_income"]
    expected["other_investment_income"] = expected["dividend_income"]
    nominal = {
        "gift_aid",
        "charitable_investment_gifts",
        "hmrc_spi_incapacity_benefit_income",
        "hmrc_spi_other_social_security_income",
        "hmrc_spi_unemployment_benefit_income",
        "hmrc_spi_state_pension_income",
    }

    factors, receipt = spi_income._spi_income_uprating_factors.__wrapped__(2024)

    assert receipt["from_period"] == 2022
    assert receipt["to_period"] == 2024
    assert set(factors) == set(SPI_INCOME_QRF_OUTPUT_COLUMNS)
    assert {
        column
        for column, row in receipt["columns"].items()
        if row["basis"] == "held_nominal"
    } == nominal
    for column, row in receipt["columns"].items():
        if column in nominal:
            assert row == {
                "variable": None,
                "index": None,
                "factor": 1.0,
                "basis": "held_nominal",
            }
        else:
            path, ratio = expected[row["variable"]]
            assert row["index"] == "gov.economic_assumptions.indices." + path
            assert row["basis"] == "model_index"
            assert factors[column] == pytest.approx(float(ratio))
            assert np.isfinite(factors[column]) and factors[column] > 0

    unchanged, source_receipt = spi_income._spi_income_uprating_factors.__wrapped__(
        2022
    )
    assert set(unchanged.values()) == {1.0}
    assert source_receipt["from_period"] == source_receipt["to_period"] == 2022
    with pytest.raises(ValueError, match="before its source year"):
        spi_income._spi_income_uprating_factors.__wrapped__(2021)
