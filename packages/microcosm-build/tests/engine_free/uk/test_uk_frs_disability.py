"""Tests split from packages/microcosm-build/tests/test_uk_frs_disability.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_frs_disability import *


def test_disability_category_threshold_and_overwrite() -> None:
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [
                (20 - 1) * WEEKS_IN_YEAR,
                (20 - 1.01) * WEEKS_IN_YEAR,
            ],
            "dla_sc_reported": [0, 0],
            "dla_m_reported": [0, 0],
            "pip_m_reported": [0, 0],
            "pip_dl_reported": [0, 0],
        }
    )

    result = derive_frs_disability(
        person, category_rates=_category_rates(), flag_rates=_flags()
    )

    assert result["aa_category"].tolist() == ["HIGHER", "LOWER"]


def test_disability_flag_operator_asymmetry_and_afcs() -> None:
    boundary = (30 - 1) * WEEKS_IN_YEAR
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [0, 0],
            "dla_sc_reported": [boundary, 0],
            "dla_m_reported": [0, 0],
            "pip_m_reported": [0, 0],
            "pip_dl_reported": [0, 0],
            "sda_reported": [0, 0],
            "incapacity_benefit_reported": [0, 0],
            "iidb_reported": [0, 0],
            "afcs_reported": [0, 1],
            "esa_contrib_reported": [0, 0],
            "esa_income_reported": [0, 0],
        }
    )

    result = derive_frs_disability(
        person, category_rates=_category_rates(), flag_rates=_flags()
    )

    assert result["is_enhanced_disabled_for_benefits"].tolist() == [False, False]
    # DLA care at the highest rate counts; an Armed Forces Compensation Scheme
    # payment alone does not, since FRS code 8 cannot be told apart from Armed
    # Forces Independence Payment (pe-uk#1946, uk-data#494).
    assert result["is_severely_disabled_for_benefits"].tolist() == [True, False]


@pytest.mark.parametrize(
    ("attendance_allowance", "severe"),
    [(0.0, False), (10.0, False), (20.0, True)],
)
def test_severe_flag_counts_only_higher_rate_attendance_allowance(
    attendance_allowance: float, severe: bool
) -> None:
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [attendance_allowance * WEEKS_IN_YEAR],
            "dla_sc_reported": [0.0],
            "dla_m_reported": [0.0],
            "pip_m_reported": [0.0],
            "pip_dl_reported": [0.0],
        }
    )

    result = derive_frs_disability(
        person, category_rates=_category_rates(), flag_rates=_flags()
    )

    assert result["is_severely_disabled_for_benefits"].tolist() == [severe]


def test_severe_flag_is_the_tax_credit_condition_on_the_categories() -> None:
    """Every row's flag equals the category rule, across the rate boundaries."""

    rates = _category_rates()
    offsets = (0.0, 0.5, -0.5, -0.99, -1.0, -1.01, -1.5)
    rows = []
    for column, levels in (
        ("attendance_allowance_reported", (rates.aa_lower, rates.aa_higher)),
        (
            "dla_sc_reported",
            (rates.dla_sc_lower, rates.dla_sc_middle, rates.dla_sc_higher),
        ),
        ("pip_dl_reported", (rates.pip_dl_standard, rates.pip_dl_enhanced)),
    ):
        for level in levels:
            for offset in offsets:
                row = dict.fromkeys(
                    (
                        "attendance_allowance_reported",
                        "dla_sc_reported",
                        "dla_m_reported",
                        "pip_m_reported",
                        "pip_dl_reported",
                    ),
                    0.0,
                )
                row[column] = (level + offset) * WEEKS_IN_YEAR
                rows.append(row)
    person = pd.DataFrame(rows)

    result = derive_frs_disability(person, category_rates=rates, flag_rates=_flags())

    expected = (
        (result["aa_category"] == "HIGHER")
        | (result["dla_sc_category"] == "HIGHER")
        | (result["pip_dl_category"] == "ENHANCED")
    )
    assert result["is_severely_disabled_for_benefits"].tolist() == expected.tolist()
    # Not vacuous: middle-rate DLA care and standard PIP daily living rows
    # are present, and the flag takes both values.
    assert (result["dla_sc_category"] == "MIDDLE").any()
    assert (result["pip_dl_category"] == "STANDARD").any()
    assert expected.any() and not expected.all()


@pytest.mark.parametrize("operation_index", [0, 1])
def test_disability_stage_rejects_year_rule_drift(operation_index: int) -> None:
    stage = load_country_spec("uk").sources.stage_map()["frs_disability"]
    operations = list(stage.operations)
    operation = operations[operation_index]
    operations[operation_index] = SourceOperationSpec(
        operation.kind,
        {**operation.parameters, "year_rule": "calibration_year"},
    )

    with pytest.raises(ValueError, match="declaration drifted"):
        _assert_frs_disability_stage_parameters(
            replace(stage, operations=tuple(operations))
        )


def test_disability_stage_resolves_the_survey_year_not_the_frame_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Survey year and frame period are both 2024 on the current release, so a
    # frame stamped 2023 is the only way to see which one picks the rates.
    from microcosm.build.uk_runtime import frs_disability
    from microcosm.build.uk_runtime.national_frame import (
        uk_national_frame,
        uk_time_period,
    )

    stage = load_country_spec("uk").sources.stage_map()["frs_disability"]
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1],
                "person_benunit_id": [1],
                "person_household_id": [1],
                "attendance_allowance_reported": [20 * WEEKS_IN_YEAR],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame({"household_id": [1]}),
        time_period="2023",
        household_weights=[1.0],
    )
    years: list[int] = []

    def _category_reader(year: int) -> UKDWPDisabilityCategoryRates:
        years.append(year)
        return _category_rates()

    def _flag_reader(year: int) -> UKDWPDisabilityFlagRates:
        years.append(year)
        return _flags()

    monkeypatch.setattr(
        frs_disability, "uk_dwp_disability_category_rates", _category_reader
    )
    monkeypatch.setattr(frs_disability, "uk_dwp_disability_flag_rates", _flag_reader)

    result = frs_disability.UKFRSDisabilityStageTransform(stage=stage)(frame)

    assert years == [2024, 2024]
    assert uk_time_period(result) == "2023"
    assert result.table("person")["aa_category"].tolist() == ["HIGHER"]


def _carer_link_people() -> pd.DataFrame:
    """Eleven one- or two-person households, each carer at 35 hours a week.

    Households 1 and 3: the carer's main recipient (member 2) qualifies
    (middle-rate DLA care, lower-rate Attendance Allowance). Households 2 and
    4: member 2 gets none of the benefits. Households 5 to 8: the main
    recipient lives outside. Household 9: care hours and no link in the care
    tab. Household 10: no care at all. Household 11: a qualifying member and
    reported Carer's Allowance, so it stays out of the share's base.
    """

    from microcosm.build.uk_runtime.frs_spine import FRS_CARE_OUTSIDE_BIT

    rows = []
    member = 1 << 2
    outside = 1 << FRS_CARE_OUTSIDE_BIT
    carers = {
        1: (member, "MIDDLE", "NONE", 0.0),
        2: (member, "NONE", "NONE", 0.0),
        3: (member, "NONE", "LOWER", 0.0),
        4: (member, "NONE", "NONE", 0.0),
        5: (outside, None, None, 0.0),
        6: (outside, None, None, 0.0),
        7: (outside, None, None, 0.0),
        8: (outside, None, None, 0.0),
        9: (0, None, None, 0.0),
        10: (0, None, None, 0.0),
        11: (member, "HIGHER", "NONE", 5000.0),
    }
    for household, (mask, dla, aa, reported) in carers.items():
        rows.append(
            {
                "person_id": household * 1000 + 1,
                "person_household_id": household,
                "frs_care_recipients": mask,
                "care_hours": 0.0 if household == 10 else 35.0,
                "carers_allowance_reported": reported,
                "aa_category": "NONE",
                "dla_sc_category": "NONE",
                "pip_dl_category": "NONE",
            }
        )
        if dla is not None:
            rows.append(
                {
                    "person_id": household * 1000 + 2,
                    "person_household_id": household,
                    "frs_care_recipients": 0,
                    "care_hours": 0.0,
                    "carers_allowance_reported": 0.0,
                    "aa_category": aa,
                    "dla_sc_category": dla,
                    "pip_dl_category": "NONE",
                }
            )
    return pd.DataFrame(rows)


def test_the_carer_link_follows_the_main_recipient_and_draws_outside_carers() -> None:
    """A3's choices of 10 Nov 2026: a qualifying household recipient makes it
    true, a non-qualifying one false, and outside carers are drawn at the share
    of non-reporting household carers at the line whose recipient qualifies."""

    from microcosm.build.uk_runtime import frs_disability

    people = _carer_link_people()
    link, evidence = frs_disability.derive_carer_link(
        people, weights=np.ones(len(people)), min_hours=35.0
    )
    by_id = dict(zip(people["person_id"], link, strict=True))

    assert by_id[1001] and by_id[3001] and by_id[11001]
    assert not by_id[2001] and not by_id[4001]
    # Two of the four household carers in the base qualify: a share of a half,
    # which the weight-balanced draw gives exactly two equal-weight outside
    # carers.
    assert evidence["share"] == 0.5
    assert sum(by_id[household * 1000 + 1] for household in (5, 6, 7, 8)) == 2
    # No link in the care tab keeps the engine's default; no care is false.
    assert by_id[9001]
    assert not by_id[10001]
    # Recipients themselves aren't carers.
    assert not by_id[1002] and not by_id[2002]
    routes = evidence["routes_at_the_line"]
    assert routes["outside_drawn"] == {"records": "3-9", "support": "3-9", "true": "<3"}
    assert routes["no_link"]["true"] == "<3"
    assert evidence["rule"] == frs_disability.CARER_LINK_PARAMETERS


def test_the_carer_link_draw_is_reproducible() -> None:
    from microcosm.build.uk_runtime import frs_disability

    people = _carer_link_people()
    first, _ = frs_disability.derive_carer_link(
        people, weights=np.ones(len(people)), min_hours=35.0
    )
    second, _ = frs_disability.derive_carer_link(
        people.iloc[::-1].reset_index(drop=True),
        weights=np.ones(len(people)),
        min_hours=35.0,
    )
    first_by_id = dict(zip(people["person_id"], first, strict=True))
    second_by_id = dict(zip(people["person_id"].iloc[::-1], second, strict=True))
    assert first_by_id == second_by_id


def test_the_carer_link_reads_a_copy_households_members_by_person_number() -> None:
    """SPI and CGT copies shift person and household ids by a power of ten, so
    the link keys a member on household and person_id modulo 1,000. The outside
    draw is keyed on identity, so only the other routes are compared."""

    from microcosm.build.uk_runtime import frs_disability

    people = _carer_link_people()
    shift = 10**8
    copies = people.assign(
        person_id=people["person_id"] + shift,
        person_household_id=people["person_household_id"] + shift,
    )
    link, _ = frs_disability.derive_carer_link(
        copies, weights=np.ones(len(copies)), min_hours=35.0
    )
    original, _ = frs_disability.derive_carer_link(
        people, weights=np.ones(len(people)), min_hours=35.0
    )
    not_drawn = ~people["person_household_id"].isin([5, 6, 7, 8]).to_numpy()
    assert link[not_drawn].tolist() == original[not_drawn].tolist()
    assert link[people["person_id"].eq(1001).to_numpy()].all()
