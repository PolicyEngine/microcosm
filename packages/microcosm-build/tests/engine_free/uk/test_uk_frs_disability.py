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
    # The tax credit severe disability condition is policyengine-uk's formula
    # on the categories (microcosm#1095, A4), so the stage stores no flag: DLA
    # care at the highest rate is a HIGHER category, and an Armed Forces
    # Compensation Scheme payment alone gives none, since FRS code 8 cannot be
    # told apart from Armed Forces Independence Payment (pe-uk#1946).
    assert result["dla_sc_category"].tolist() == ["HIGHER", "NONE"]
    assert "is_severely_disabled_for_benefits" not in result


@pytest.mark.parametrize(
    ("attendance_allowance", "higher"),
    [(0.0, False), (10.0, False), (20.0, True)],
)
def test_only_higher_rate_attendance_allowance_is_a_higher_category(
    attendance_allowance: float, higher: bool
) -> None:
    """The engine's severe disability formula reads this category (A4)."""

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

    assert (result["aa_category"] == "HIGHER").tolist() == [higher]
    assert "is_severely_disabled_for_benefits" not in result


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


def test_the_carer_link_takes_a_share_from_other_households() -> None:
    """The SPI channel draws its outside carers at the survey's own share
    (spi_income), so the share can be passed in; a share resting on one or two
    households, and a route carried by one or two, is withheld."""

    from microcosm.build.uk_runtime import frs_disability

    people = _carer_link_people()
    weights = np.ones(len(people))
    # Households 1 to 4 make the base: 11 reports the allowance.
    base = frs_disability.carer_link_share(people, weights=weights, min_hours=35.0)
    assert base == (0.5, 4)
    outside = people["person_household_id"].isin([5, 6, 7, 8]).to_numpy()
    for share, drawn in ((1.0, 4), (0.0, 0)):
        link, evidence = frs_disability.derive_carer_link(
            people, weights=weights, min_hours=35.0, share=(share, 30)
        )
        assert int(link[outside].sum()) == drawn
        assert evidence["share"] == share
        assert evidence["share_support"] == "30+"
    _, thin = frs_disability.derive_carer_link(
        people, weights=weights, min_hours=35.0, share=(1.0, 2)
    )
    assert thin["share"] == "<3"
    _, collapsed = frs_disability.derive_carer_link(
        people,
        weights=weights,
        min_hours=35.0,
        support_keys=np.zeros(len(people), dtype=np.int64),
    )
    assert collapsed["routes_at_the_line"]["outside_drawn"] == {
        "records": "<3",
        "support": "<3",
        "true": "<3",
    }


def test_the_carer_link_is_checked_against_dwps_uc_carer_element() -> None:
    """A3 (f): DWP's record of the UC carer element checks the link, and
    changes no value. A claim counts once, by its UC row's carrier, against
    whether any claimant or partner is a carer for Universal Credit."""

    from microcosm.build.uk_runtime.frs_disability import carer_link_admin_check
    from microcosm.build.uk_runtime.frs_spine import FRS_UC_CARER_ELEMENT_COLUMN
    from microcosm.build.uk_runtime.receipt_counts import (
        UK_WITHHELD,
        rounded_count,
        support_band,
    )

    classes = {
        # DWP's carer element, care hours, the link, Carer's Allowance reported.
        "both": ("CARER", 35.0, True, 0.0),
        "dwp_only": ("CARER", 35.0, False, 0.0),
        "stage_only": ("NONE", 0.0, False, 81.9 * WEEKS_IN_YEAR),
        "neither": ("NONE", 20.0, True, 0.0),
    }
    rows = []
    household = 0
    for status, hours, linked, reported in classes.values():
        for _ in range(3):
            household += 1
            rows.append(
                {
                    "person_id": household * 1000 + 1,
                    "person_household_id": household,
                    "person_benunit_id": household * 10 + 1,
                    "is_claimant_or_partner": True,
                    "care_hours": hours,
                    "carers_allowance_reported": reported,
                    FRS_UC_CARER_ELEMENT_COLUMN: status,
                    "link": linked,
                }
            )
            # A partner off the UC row carries NOT_LINKED and is no carer.
            rows.append(
                {
                    "person_id": household * 1000 + 2,
                    "person_household_id": household,
                    "person_benunit_id": household * 10 + 1,
                    "is_claimant_or_partner": True,
                    "care_hours": 0.0,
                    "carers_allowance_reported": 0.0,
                    FRS_UC_CARER_ELEMENT_COLUMN: "NOT_LINKED",
                    "link": False,
                }
            )
    # A claim with no linked record is left out, whatever its carer.
    rows.append(
        {
            "person_id": 99001,
            "person_household_id": 99,
            "person_benunit_id": 991,
            "is_claimant_or_partner": True,
            "care_hours": 35.0,
            "carers_allowance_reported": 0.0,
            FRS_UC_CARER_ELEMENT_COLUMN: "NOT_LINKED",
            "link": True,
        }
    )
    frame = pd.DataFrame(rows)
    person = frame.drop(columns="link")
    link = frame["link"].to_numpy(dtype=bool)

    check = carer_link_admin_check(person, link=link, min_hours=35.0)

    assert check["linked_claims"] == {
        "claims": rounded_count(12),
        "support": support_band(12),
    }
    for name in classes:
        assert check["classes"][name] == {
            "claims": rounded_count(3),
            "support": support_band(3),
        }
    assert check["dwp_carer_claims_the_stage_holds"] == 0.5
    assert check["stage_carer_claims_dwp_holds"] == 0.5
    # The check reads the link and changes nothing in the frame.
    pd.testing.assert_frame_equal(person, frame.drop(columns="link"))

    # A class on one or two households is withheld, with the shares it sits in.
    thin = person.loc[
        ~person["person_household_id"].isin([4, 5])
    ]  # dwp_only keeps one household
    thin_link = link[thin.index.to_numpy()]
    thin_check = carer_link_admin_check(thin, link=thin_link, min_hours=35.0)
    assert thin_check["classes"]["dwp_only"]["claims"] == UK_WITHHELD
    assert thin_check["dwp_carer_claims_the_stage_holds"] == UK_WITHHELD

    # A frame without the carrier, such as the SPI channel's copies, has none.
    assert (
        carer_link_admin_check(
            person.drop(columns=FRS_UC_CARER_ELEMENT_COLUMN), link=link, min_hours=35.0
        )
        is None
    )
