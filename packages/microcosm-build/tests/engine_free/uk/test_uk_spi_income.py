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
        )


def test_spi_qrf_requires_current_donor_filename(tmp_path) -> None:
    support = _dead_support()
    donor_path = tmp_path / "put2021uk.tab"
    _write_donor(donor_path)

    with pytest.raises(ValueError, match=SPI_DONOR_FILENAME):
        impute_uk_spi_income_support(support, donor_path)


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
