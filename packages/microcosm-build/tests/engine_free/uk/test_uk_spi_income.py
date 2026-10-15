"""Tests split from packages/microcosm-build/tests/test_uk_spi_income.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_spi_income import *


def test_spi_disability_refresh_reuses_frs_derivation_for_spi_people() -> None:
    weeks = 365.25 / 7
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [0.0, 19 * weeks, 0.0],
            "afcs_reported": [0.0, 0.0, 1.0],
        },
        index=[10, 20, 30],
    )
    for column in frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS:
        if column.endswith("_category"):
            person[column] = ["BASE", "STALE", "STALE"]
        else:
            person[column] = [True, False, False]
    spi_people = pd.Series([False, True, True], index=person.index)
    category_rates = frs_disability.UKDWPDisabilityCategoryRates(
        aa_lower=10,
        aa_higher=20,
        dla_sc_lower=10,
        dla_sc_middle=20,
        dla_sc_higher=30,
        dla_m_lower=10,
        dla_m_higher=20,
        pip_m_standard=10,
        pip_m_enhanced=20,
        pip_dl_standard=10,
        pip_dl_enhanced=20,
        instant="2024-01-01",
        source="fixture",
    )
    flag_rates = frs_disability.UKDWPDisabilityFlagRates(
        aa_higher=20,
        dla_sc_higher=30,
        pip_dl_enhanced=20,
        instant="2024-01-01",
        source="fixture",
    )
    output_columns = list(frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS)
    non_spi_before = person.loc[~spi_people, output_columns].copy()
    expected = frs_disability.derive_frs_disability(
        person.loc[spi_people],
        category_rates=category_rates,
        flag_rates=flag_rates,
    )

    result = spi_income._refresh_disability_derived_inputs(
        person,
        spi_people=spi_people,
        category_rates=category_rates,
        flag_rates=flag_rates,
    )

    pd.testing.assert_frame_equal(result.loc[spi_people, output_columns], expected)
    pd.testing.assert_frame_equal(
        result.loc[~spi_people, output_columns], non_spi_before
    )


def test_spi_carer_take_up_refresh_follows_the_refilled_receipt() -> None:
    """The carer flag is re-derived from the refilled receipt on SPI rows only."""

    person = pd.DataFrame(
        {
            "carers_allowance_reported": [0.0, 0.0, 12.0, 5.0],
            # Root-stage flags: row 20 kept its donor's True although the fill
            # zeroed the receipt; row 30 kept a False although the fill gave
            # it one; row 40 is a base row whose stale True must survive.
            "would_claim_carers_allowance": [False, True, False, True],
        },
        index=[10, 20, 30, 40],
    )
    person["would_claim_carers_allowance"] = person[
        "would_claim_carers_allowance"
    ].astype(bool)
    spi_people = pd.Series([False, True, True, False], index=person.index)

    result = spi_income._refresh_carer_take_up_input(person, spi_people=spi_people)

    assert result["would_claim_carers_allowance"].tolist() == [False, False, True, True]
    assert result["would_claim_carers_allowance"].dtype == bool
    # A frame without the flag (an older synthetic manifest) passes through.
    bare = pd.DataFrame({"carers_allowance_reported": [1.0]}, index=[1])
    assert (
        "would_claim_carers_allowance"
        not in spi_income._refresh_carer_take_up_input(
            bare, spi_people=pd.Series([True], index=[1])
        )
    )


def test_spi_uc_gainful_self_employment_follows_the_rows_drawn_incomes() -> None:
    """A drawn row's gainful self-employment takes the root's whole rule."""

    person = pd.DataFrame(
        {
            "employment_status": [
                "FT_EMPLOYED",
                "FT_SELF_EMPLOYED",
                "PT_EMPLOYED",
                "FT_SELF_EMPLOYED",
                "FT_SELF_EMPLOYED",
                "FT_EMPLOYED",
            ],
            # Row 10's drawn profit no longer out-earns its pay. Rows 20 and 40
            # keep a self-employed main job the draw conditioned on, so the
            # main-job route flags them whatever their profit (microcosm#840).
            # Rows 30 and 50 draw a profit above their pay, whatever the
            # status. Row 60 is a base row whose flag stays.
            "self_employment_income": [500.0, 0.0, 9_000.0, 3_000.0, 8_000.0, 0.0],
            "employment_income": [20_000.0, 0.0, 4_000.0, 25_000.0, 1_000.0, 30_000.0],
            "uc_is_in_gainful_self_employment": [True, True, False, True, False, True],
        },
        index=[10, 20, 30, 40, 50, 60],
    )
    spi_people = pd.Series([True, True, True, True, True, False], index=person.index)

    result, receipt = spi_income._refresh_uc_gainful_self_employment(
        person, spi_people=spi_people
    )

    assert result["uc_is_in_gainful_self_employment"].tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    assert result["uc_is_in_gainful_self_employment"].dtype == bool
    assert receipt == {
        "rule": receipt["rule"],
        "rows": 5,
        "flagged_rows": 4,
        "main_job_route_rows": 3,
        "earnings_route_rows": 2,
        "main_job_route_only_rows": 2,
    }
    assert "main job" in receipt["rule"] and "twin's flag" in receipt["rule"]
    bare = pd.DataFrame({"employment_status": ["FT_EMPLOYED"]}, index=[1])
    bare_result, bare_receipt = spi_income._refresh_uc_gainful_self_employment(
        bare, spi_people=pd.Series([True], index=[1])
    )
    assert "uc_is_in_gainful_self_employment" not in bare_result
    assert bare_receipt is None


def test_a_redrawn_row_drops_its_frs_twins_landlord_carrier() -> None:
    """The carrier is the FRS respondent's answer; a redrawn row's letting is its donor's."""

    person = pd.DataFrame(
        {
            "property_income": [0.0, 6_000.0, 0.0],
            "reports_rent_from_other_property": [True, True, True],
        },
        index=[10, 20, 30],
    )
    spi_people = pd.Series([True, True, False], index=person.index)

    result = spi_income._refresh_landlord_carrier(person, spi_people=spi_people)

    # Row 30 is a base row and keeps its answer.
    assert result["reports_rent_from_other_property"].tolist() == [False, False, True]
    assert result["reports_rent_from_other_property"].dtype == bool
    assert result["property_income"].tolist() == [0.0, 6_000.0, 0.0]
    bare = pd.DataFrame({"property_income": [0.0]}, index=[1])
    assert "reports_rent_from_other_property" not in (
        spi_income._refresh_landlord_carrier(
            bare, spi_people=pd.Series([True], index=[1])
        )
    )


def test_spi_uc_start_up_period_holds_only_while_the_row_stays_self_employed() -> None:
    """A redrawn row keeps its donor's start-up period only while self-employed."""

    person = pd.DataFrame(
        {
            "employment_status": [
                "FT_EMPLOYED",
                "FT_SELF_EMPLOYED",
                "FT_EMPLOYED",
                "FT_EMPLOYED",
            ],
            # Row 20 keeps a self-employed status, row 30 keeps imputed
            # self-employment income, row 40 is a base row whose flag stays.
            "self_employment_income": [0.0, 0.0, 800.0, 0.0],
            "uc_is_in_startup_period": [True, True, True, True],
        },
        index=[10, 20, 30, 40],
    )
    spi_people = pd.Series([True, True, True, False], index=person.index)

    result = spi_income._refresh_uc_start_up_period(person, spi_people=spi_people)

    assert result["uc_is_in_startup_period"].tolist() == [False, True, True, True]
    assert result["uc_is_in_startup_period"].dtype == bool
    bare = pd.DataFrame({"employment_status": ["FT_EMPLOYED"]}, index=[1])
    assert "uc_is_in_startup_period" not in spi_income._refresh_uc_start_up_period(
        bare, spi_people=pd.Series([True], index=[1])
    )


def test_finite_numeric_diagnostic_names_columns_and_counts() -> None:
    frame = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0],
            "bad": [np.nan, np.inf, 3.0],
            "also_bad": ["not-numeric", 1.0, 2.0],
        }
    )

    with pytest.raises(ValueError) as error:
        spi_income._require_finite_numeric(frame, label="diagnostic fixture")

    message = str(error.value)
    assert "'bad': 2" in message
    assert "'also_bad': 1" in message
    assert "good" not in message


def test_spi_qrf_fails_closed_on_missing_donor_component(monkeypatch, tmp_path) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path, drop="OTHERINV")
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    with pytest.raises(ValueError, match="OTHERINV"):
        impute_uk_spi_income_support(
            support,
            donor_path,
            donor_sample_size=None,
            state_pension_age=66,
            allow_in_sample_reference=True,
        )


def test_spi_donor_preserves_documented_unattributed_sex_code(tmp_path) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")
    raw.loc[0, "SEX"] = 0

    donor = spi_income._prepare_spi_donor(raw, seed=7)

    assert donor.loc[0, "gender"] == "UNKNOWN"
    assert set(donor["gender"]) == {"UNKNOWN", "MALE", "FEMALE"}


def test_spi_donor_keeps_narrow_pe_employment_and_broad_hmrc_measure(
    tmp_path,
) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")

    donor = spi_income._prepare_spi_donor(raw, seed=7)
    expected_pe_employment = raw["PAY"] + raw["EPB"] + raw["TAXTERM"]
    np.testing.assert_array_equal(
        donor["employment_income"],
        expected_pe_employment,
    )
    np.testing.assert_array_equal(
        donor[SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN],
        raw["MOTHINC"],
    )
    assert donor[SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN].lt(0.0).any()
    assert donor[SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN].gt(0.0).any()

    derived = spi_income.derive_hmrc_income_auxiliaries(
        donor.assign(tax_free_savings_income=0.0)
    )
    expected_hmrc_employed = (
        (raw["PAY"] + raw["EPB"] - raw["EXPS"]).clip(lower=0.0)
        + raw["INCPBEN"]
        + raw["OSSBEN"]
        + raw["TAXTERM"]
        + raw["UBISJA"]
        + raw["MOTHINC"]
    )
    np.testing.assert_array_equal(
        derived[SPI_HMRC_EMPLOYED_INCOME_COLUMN],
        expected_hmrc_employed,
    )
    np.testing.assert_array_equal(
        derived[HMRC_SPI_ASSESSABLE_INCOME_COLUMN],
        derived[SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN]
        + derived[SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN],
    )


def test_spi_donor_rejects_leaf_reconciliation_drift(tmp_path) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")
    # Preserve the published TI = TEI + TII identity while breaking Annex A's
    # source-leaf formula, proving the two source diagnostics are independent.
    raw["TEI"] += 1_000.0
    raw["TI"] += 1_000.0

    with pytest.raises(ValueError, match="source-leaf reconciliation"):
        spi_income._prepare_spi_donor(raw, seed=7)


def test_spi_donor_accepts_reviewed_composite_reconciliation_envelope(
    tmp_path,
) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")
    raw.loc[0, "AGERANGE"] = -1
    raw.loc[0, ["TEI", "TI"]] += 180.0

    spi_income._prepare_spi_donor(raw, seed=7)

    raw.loc[0, ["TEI", "TI"]] += 5.0
    with pytest.raises(ValueError, match="composite TEI"):
        spi_income._prepare_spi_donor(raw, seed=7)


@pytest.mark.skipif(
    not _PINNED_SPI_DONOR_PATH.is_file(),
    reason="licensed pinned SPI donor is not staged locally",
)
def test_real_pinned_spi_donor_reconciles_documented_source_leaves() -> None:
    spi_income._verify_spi_donor_identity(_PINNED_SPI_DONOR_PATH)
    raw = pd.read_csv(_PINNED_SPI_DONOR_PATH, delimiter="\t")

    donor = spi_income._prepare_spi_donor(raw, seed=42)

    assert len(donor) == 836_850


def test_spi_donor_rejects_undocumented_sex_code(tmp_path) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")
    raw.loc[0, "SEX"] = 3

    with pytest.raises(ValueError, match="documented codes 0/1/2"):
        spi_income._prepare_spi_donor(raw, seed=7)


def test_spi_donor_carries_restricted_finance_costs_with_their_relief(
    tmp_path,
) -> None:
    """microcosm#1106: LLIR_RESTRICT_AMT_TOT becomes the property_finance_costs
    leaf, and the donor is refused unless TAX_CRED carries 20% relief on it."""

    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")

    donor = spi_income._prepare_spi_donor(raw, seed=7)

    assert donor["property_finance_costs"].tolist() == [0.0, 0.0, 200.0, 300.0]
    check = donor.attrs["property_finance_costs"]
    assert check["rows_with_restricted_costs"] == 2
    assert check["weighted_restricted_costs_gbp"] == pytest.approx(
        3.0 * 200.0 + 4.0 * 300.0
    )
    assert check["median_relief_ratio"] == pytest.approx(0.2)
    # The leaf is drawn last, so it conditions on every income leaf and moves
    # no earlier draw.
    assert spi_income.SPI_INCOME_QRF_OUTPUT_COLUMNS[-1] == "property_finance_costs"

    wrong = raw.copy()
    wrong["TAX_CRED"] = wrong["LLIR_RESTRICT_AMT_TOT"] * 0.4
    with pytest.raises(ValueError, match="20% finance-cost relief"):
        spi_income._prepare_spi_donor(wrong, seed=7)
    negative = raw.copy()
    negative.loc[0, "LLIR_RESTRICT_AMT_TOT"] = -1.0
    with pytest.raises(ValueError, match="negative costs"):
        spi_income._prepare_spi_donor(negative, seed=7)
    none = raw.copy()
    none[["LLIR_RESTRICT_AMT_TOT", "TAX_CRED"]] = 0.0
    assert (
        spi_income._prepare_spi_donor(none, seed=7).attrs["property_finance_costs"][
            "median_relief_ratio"
        ]
        is None
    )


def test_spi_qrf_fails_closed_on_unreviewed_stage2_gap(monkeypatch, tmp_path) -> None:
    support = _dead_support(drop_stage2="universal_credit_reported")
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    with pytest.raises(ValueError, match="universal_credit_reported"):
        impute_uk_spi_income_support(
            support,
            donor_path,
            donor_sample_size=None,
            state_pension_age=66,
            allow_in_sample_reference=True,
        )


def test_spi_qrf_fails_closed_on_missing_retained_frs_hmrc_leaf(
    monkeypatch,
    tmp_path,
) -> None:
    missing = FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN
    support = _dead_support(drop_hmrc_leaf=missing)
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    with pytest.raises(ValueError, match=missing):
        impute_uk_spi_income_support(
            support,
            donor_path,
            donor_sample_size=None,
            state_pension_age=66,
            allow_in_sample_reference=True,
        )


@pytest.mark.parametrize(
    "unavailable_full_concept",
    spi_income.FRS_HMRC_UNAVAILABLE_FULL_CONCEPT_COLUMNS,
)
def test_spi_qrf_forbids_source_absent_full_concepts_on_frs(
    monkeypatch,
    tmp_path,
    unavailable_full_concept,
) -> None:
    support = _dead_support()
    person = support.person.copy()
    person[unavailable_full_concept] = 0.0
    support = replace(support, person=person)
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    with pytest.raises(ValueError, match=unavailable_full_concept):
        impute_uk_spi_income_support(
            support,
            donor_path,
            donor_sample_size=None,
            state_pension_age=66,
            allow_in_sample_reference=True,
        )


def test_spi_qrf_requires_current_donor_filename(tmp_path) -> None:
    support = _dead_support()
    donor_path = tmp_path / "put2021uk.tab"
    _write_donor(donor_path)

    with pytest.raises(ValueError, match=SPI_DONOR_FILENAME):
        impute_uk_spi_income_support(
            support, donor_path, allow_in_sample_reference=True
        )


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing_variable", "Missing SPI uprating variable"),
        ("missing_index", "Unsupported SPI uprating index"),
        ("blank_index", "Unsupported SPI uprating index"),
        ("callable_index", "Unsupported SPI uprating index"),
        ("missing_parameter", "Missing SPI uprating parameter"),
        ("nonpositive_index", "SPI uprating index must be positive"),
        ("nonfinite_index", "SPI uprating index must be positive"),
    ],
)
def test_declared_spi_uprating_mapping_cannot_silently_become_nominal(
    monkeypatch, defect, message
) -> None:
    variables = {
        name: SimpleNamespace(uprating="index")
        for name in spi_income.SPI_INCOME_UPRATING_VARIABLES.values()
        if name is not None
    }
    variable = "self_employment_income"
    if defect == "missing_variable":
        del variables[variable]
    elif defect == "missing_index":
        variables[variable] = SimpleNamespace()
    elif defect == "blank_index":
        variables[variable].uprating = ""
    elif defect == "callable_index":
        variables[variable].uprating = lambda _: 1.0

    def parameters(period):
        if defect == "missing_parameter":
            return SimpleNamespace()
        value = 100.0 if period.startswith("2022") else 120.0
        if defect == "nonpositive_index":
            value = 0.0
        elif defect == "nonfinite_index":
            value = np.nan
        return SimpleNamespace(index=value)

    system = SimpleNamespace(variables=variables, parameters=parameters)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(CountryTaxBenefitSystem=lambda: system),
    )

    # Bypass memoization so one runtime's cached factors cannot conceal a
    # changed or incomplete installed-engine contract.
    with pytest.raises(ValueError, match=message):
        spi_income._spi_income_uprating_factors.__wrapped__(2024)


def test_person_bridge_matches_the_tape_formulas() -> None:
    person = pd.DataFrame(
        {
            "age": [30.0, 70.0, 45.0, 50.0],
            "employment_income": [20_000.0, 0.0, 0.0, 3_000.0],
            "self_employment_income": [0.0, 0.0, 0.0, 9_000.0],
            "private_pension_income": [0.0, 4_000.0, 0.0, 0.0],
            "state_pension_reported": [0.0, 9_000.0, 0.0, 0.0],
            "employment_status": [
                "FT_EMPLOYED",
                "RETIRED",
                "PT_SELF_EMPLOYED",
                "FT_SELF_EMPLOYED",
            ],
        }
    )
    bridge = spi_income._person_bridge(person, state_pension_age=66)
    assert bridge["main_source"].tolist() == ["pay", "pension", "other", "trade"]
    assert bridge["has_pay"].tolist() == [1.0, 0.0, 0.0, 1.0]
    # A self-employed main job at a loss still files trade pages.
    assert bridge["has_trade"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert bridge["at_or_over_spa"].tolist() == [0.0, 1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="missing column"):
        spi_income._person_bridge(
            person.drop(columns=["employment_status"]), state_pension_age=66
        )


def test_taxable_income_mask_applies_the_no_record_rule() -> None:
    columns = {
        "age": 40.0,
        "employment_income": 0.0,
        "self_employment_income": 0.0,
        "private_pension_income": 0.0,
        "state_pension_reported": 0.0,
        "employment_status": "UNEMPLOYED",
        "savings_interest_income": 0.0,
        "tax_free_savings_income": 0.0,
        "dividend_income": 0.0,
        "property_income": 0.0,
        "miscellaneous_income": 0.0,
    }
    rows = [
        dict(columns),  # nothing at all
        {
            **columns,
            "savings_interest_income": 1_500.0,
            "tax_free_savings_income": 600.0,
        },
        {**columns, "dividend_income": 1_200.0},
        {**columns, "employment_status": "FT_SELF_EMPLOYED"},
        {**columns, "state_pension_reported": 10.0},
        {**columns, "property_income": 600.0, "miscellaneous_income": 500.0},
    ]
    mask = spi_income._taxable_income_mask(pd.DataFrame(rows))
    assert mask.tolist() == [False, False, True, True, True, True]


def test_stage_refuses_in_sample_reference_unless_allowed(
    monkeypatch, tmp_path
) -> None:
    """A full build never holds too few predictor cells to cross-fit the
    reference; when the fold draw yields one fold, the stage refuses unless a
    sampled rung or a synthetic fixture says it may locate in-sample."""

    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)
    monkeypatch.setattr(
        spi_income,
        "cell_fold_ids",
        lambda predictors, *, folds, seed, used=None: (
            np.zeros(len(predictors), dtype=np.int64),
            1,
        ),
    )
    with pytest.raises(ValueError, match="allowed only on a sampled rung"):
        impute_uk_spi_income_support(
            support,
            donor_path,
            seed=9,
            n_estimators=3,
            donor_sample_size=None,
            state_pension_age=66,
        )


def test_stage_requires_the_identity_key(monkeypatch, tmp_path) -> None:
    support = _dead_support()
    person = support.person.drop(columns=["person_source_id"])
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)
    with pytest.raises(ValueError, match="person_source_id"):
        impute_uk_spi_income_support(
            replace(support, person=person),
            donor_path,
            seed=9,
            n_estimators=3,
            donor_sample_size=None,
            state_pension_age=66,
            allow_in_sample_reference=True,
        )


def test_spi_donor_refuses_undocumented_bridge_codes(tmp_path) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, sep="\t")
    for column, bad in (("MAINSRCE", 7), ("SEINC_NUM", 2), ("SPA", 3)):
        drifted = raw.copy()
        drifted.loc[0, column] = bad
        with pytest.raises(ValueError, match=f"{column} has undocumented code"):
            spi_income.prepare_spi_donor_table(drifted, seed=1)
    fractional = raw.copy()
    fractional["SPA"] = fractional["SPA"].astype(float)
    fractional.loc[0, "SPA"] = 0.5
    with pytest.raises(ValueError, match="integer codes"):
        spi_income.prepare_spi_donor_table(fractional, seed=1)


def test_spi_donor_bridge_is_validated_against_the_published_codes(tmp_path) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, sep="\t")
    donor = spi_income.prepare_spi_donor_table(raw, seed=1)
    assert donor["main_source"].tolist() == ["pay"] * 4
    assert donor["has_pay"].tolist() == [1.0] * 4
    assert donor["has_trade"].tolist() == [1.0] * 4
    assert donor["at_or_over_spa"].tolist() == [0.0, 0.0, 1.0, 1.0]
    receipt = donor.attrs["bridge"]
    assert receipt["main_source"]["weighted_agreement"] == 1.0
    assert (
        receipt["main_source"]["by_published_code"]["main source pay"]["agreement"]
        == 1.0
    )
    assert receipt["has_pay"]["pay_share_among_main_source_pay"] == 1.0
    assert receipt["has_trade"]["seinc_num_share_among_positive_profits"] == 1.0
    assert receipt["has_trade"]["positive_profits_share_among_seinc_num"] == 1.0
    assert receipt["at_or_over_spa"]["srp_share_among_spa"] == 1.0
    assert receipt["at_or_over_spa"]["spa_share_among_srp"] == 1.0
    assert receipt["composite_weighted_share"] == 0.0
    # The indicator codes, not the leaves, set has_trade and at_or_over_spa on
    # a coded record: a record that files no trade pages with a profit, and a
    # State Pension recipient under State Pension age, follow their codes,
    # and the receipt reports both directions of each code against its leaf.
    coded = raw.copy()
    coded.loc[0, "SEINC_NUM"] = 0
    # Record 1's trade (PROFITS 2,000 less CAPALL 100) leaves its totals too.
    coded.loc[1, ["PROFITS", "CAPALL", "LOSSBF"]] = [0.0, 0.0, 0.0]
    coded.loc[1, ["TEI", "TI"]] = coded.loc[1, ["TEI", "TI"]] - 1_900.0
    coded.loc[2, "SPA"] = 0
    prepared = spi_income.prepare_spi_donor_table(coded, seed=1)
    # Record 0 has a profit but no pages: not a trade. Record 1 files the
    # pages with no profit: a trade, and the reverse share drops below one.
    assert prepared.loc[0, "has_trade"] == 0.0 and prepared.loc[1, "has_trade"] == 1.0
    assert prepared.loc[2, "at_or_over_spa"] == 0.0
    assert prepared.loc[3, "at_or_over_spa"] == 1.0
    bridge = prepared.attrs["bridge"]
    assert bridge["has_trade"]["seinc_num_share_among_positive_profits"] < 1.0
    assert 0.0 < bridge["has_trade"]["positive_profits_share_among_seinc_num"] < 1.0
    assert bridge["at_or_over_spa"]["spa_share_among_srp"] < 1.0
    assert bridge["at_or_over_spa"]["srp_share_among_spa"] == 1.0
    # Composites carry no code and follow their leaves.
    composite = raw.copy()
    composite.loc[3, ["AGERANGE", "MAINSRCE", "SEINC_NUM", "SPA"]] = [-1, -1, -1, -1]
    prepared = spi_income.prepare_spi_donor_table(composite, seed=1)
    assert prepared.loc[3, ["has_trade", "at_or_over_spa"]].tolist() == [1.0, 1.0]
    assert prepared.attrs["bridge"]["composite_weighted_share"] == 0.4
    # A tape whose published code the leaves no longer translate is refused.
    disagreeing = raw.copy()
    disagreeing["MAINSRCE"] = 2
    with pytest.raises(ValueError, match="no longer translates"):
        spi_income.prepare_spi_donor_table(disagreeing, seed=1)


class _Tape:
    def __init__(self, classes: dict[str, tuple[int, ...]]) -> None:
        self._classes = classes

    def sign_classes(self, target: str) -> tuple[int, ...]:
        return self._classes.get(target, (0, 1))


def test_sign_class_mismatches_name_the_located_targets_only() -> None:
    """A reference whose gate orders other classes than the tape's is a
    mismatch on a located target; an identity or degenerate target (no
    reference classes) is not."""
    receipt = {
        "targets": {
            target: {"reference_classes": [0, 1]}
            for target in spi_income.SPI_INCOME_QRF_OUTPUT_COLUMNS
        }
    }
    receipt["targets"]["private_pension_income"]["reference_classes"] = [1]
    receipt["targets"]["hmrc_spi_incapacity_benefit_income"]["reference_classes"] = None
    tape = _Tape({"hmrc_spi_incapacity_benefit_income": (0,)})
    assert spi_income._sign_class_mismatches(receipt, tape) == {
        "private_pension_income": {"reference": [1], "tape": [0, 1]}
    }
    receipt["targets"]["private_pension_income"]["reference_classes"] = [0, 1]
    assert spi_income._sign_class_mismatches(receipt, tape) == {}


def test_stage_refuses_a_sign_class_mismatch_unless_sampled(
    monkeypatch, tmp_path
) -> None:
    """A full build refuses a located target whose reference sign classes
    differ from the tape forest's; a sampled rung or a synthetic fixture
    records the mismatch in the transport receipt instead."""

    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)
    mismatch = {"private_pension_income": {"reference": [1], "tape": [0, 1]}}
    monkeypatch.setattr(
        spi_income, "_sign_class_mismatches", lambda receipt, tape: mismatch
    )
    # The synthetic support cannot cross-fit (too few cells); let the transport
    # locate in-sample either way so the sign-class check is what decides.
    real_cross_fit = spi_income.cross_fit_transport
    monkeypatch.setattr(
        spi_income,
        "cross_fit_transport",
        lambda *args, **kwargs: real_cross_fit(
            *args, **{**kwargs, "allow_in_sample": True}
        ),
    )
    options = dict(seed=9, n_estimators=3, donor_sample_size=None, state_pension_age=66)
    with pytest.raises(ValueError, match="sign classes differ from the tape forest's"):
        impute_uk_spi_income_support(
            support, donor_path, allow_in_sample_reference=False, **options
        )
    # The sampled path reaches the stage-1 draw (the disability refresh then
    # needs the engine on this synthetic frame), with the mismatch receipted.
    seen: dict[str, object] = {}
    original = _FakeFittedQRF.predict_from_uniforms

    def predict_from_uniforms(self, predictors, *, quantiles, sign_uniforms):
        seen["reached"] = True
        return original(
            self, predictors, quantiles=quantiles, sign_uniforms=sign_uniforms
        )

    monkeypatch.setattr(_FakeFittedQRF, "predict_from_uniforms", predict_from_uniforms)
    try:
        result = impute_uk_spi_income_support(
            support, donor_path, allow_in_sample_reference=True, **options
        )
    except ImportError:
        result = None
    assert seen.get("reached") is True
    if result is not None:
        assert result.stage1_transport["sign_class_mismatches"] == mismatch
