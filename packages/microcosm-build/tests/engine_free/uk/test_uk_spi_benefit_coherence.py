"""The SPI-channel benefit coherence pass (microcosm#1095, uk-data#514)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime import frs_disability
from microcosm.build.uk_runtime.frs_take_up import UKTakeUpPopulationPolicy
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_benefit_coherence import (
    SPI_BENEFIT_COHERENCE_STAGE_NAME,
    SPI_RESTORED_REPORT_COLUMNS,
    SPI_WITHIN_LIMIT_REPORT_COLUMNS,
    SPI_ZEROED_REPORT_COLUMNS,
    UKSPIPermittedWorkLimit,
    _within_limit_receipt,
    apply_spi_benefit_coherence,
    assert_spi_benefit_coherence_stage_parameters,
)
from microcosm.build.uk_runtime.spi_support import (
    support_channel_column,
    support_source_id_column,
)
from microcosm.frame import WeightKind

CATEGORY_RATES = frs_disability.UKDWPDisabilityCategoryRates(
    aa_lower=72.65,
    aa_higher=108.55,
    dla_sc_lower=28.70,
    dla_sc_middle=72.65,
    dla_sc_higher=108.55,
    dla_m_lower=28.70,
    dla_m_higher=75.75,
    pip_m_standard=28.70,
    pip_m_enhanced=75.75,
    pip_dl_standard=72.65,
    pip_dl_enhanced=108.55,
    instant="2024-01-01",
    source="test",
)
FLAG_RATES = frs_disability.UKDWPDisabilityFlagRates(
    aa_higher=108.55,
    dla_sc_higher=108.55,
    pip_dl_enhanced=108.55,
    instant="2024-01-01",
    source="test",
)
POLICY = UKTakeUpPopulationPolicy(
    adult_age=18, state_pension_age=66, instant="2024-01-01", source="test"
)
#: The 2024-25 limits: ESA's permitted work (GBP 183.50 a week, GBP 9,542 a
#: year, under 16 hours) and JSA's age-related amounts, GBP 5 disregard,
#: 16-hour remunerative work and GBP 50 pension disregard.
LIMIT = UKSPIPermittedWorkLimit(
    esa_weekly_limit=183.5,
    esa_hours_limit=16.0,
    jsa_amount_under_25=71.7,
    jsa_amount_from_25=90.5,
    jsa_earnings_disregard=5.0,
    jsa_remunerative_hours=16.0,
    jsa_pension_disregard=50.0,
    year=2024,
    source="test",
)


class _Contract:
    def rate(self, key: str) -> float:
        assert key == "universal_credit"
        return 0.55


REPORT_COLUMNS = tuple(
    dict.fromkeys(
        (
            *SPI_ZEROED_REPORT_COLUMNS,
            *SPI_RESTORED_REPORT_COLUMNS,
            *SPI_WITHIN_LIMIT_REPORT_COLUMNS,
            *frs_disability.UK_DISABILITY_FLAG_REPORTED_COLUMNS,
            "universal_credit_reported",
        )
    )
)


def _person(
    person_id, benunit_id, household_id, channel, source_id, age, **reports
) -> dict:
    row = {
        "person_id": person_id,
        "person_benunit_id": benunit_id,
        "person_household_id": household_id,
        "age": age,
        # The fixtures' adults are their units' UC claimants or partners.
        "is_uc_claimant": age >= 18,
        support_channel_column("person"): channel,
        support_source_id_column("person"): source_id,
    }
    row.update(dict.fromkeys(REPORT_COLUMNS, 0.0))
    row.update(
        employment_income=0.0,
        self_employment_income=0.0,
        hours_worked=0.0,
        private_pension_income=0.0,
    )
    row.update(reports)
    return row


def _frame(*, base_own_right_override: bool | None = None):
    people = [
        # FRS twins. Person 11 reports UC and IIDB; person 21 reports JSA and
        # AFCS; person 31 is a pensioner reporting bereavement support.
        _person(
            11,
            1,
            1,
            "frs",
            11,
            40,
            universal_credit_reported=900.0,
            iidb_reported=500.0,
        ),
        _person(
            21, 2, 2, "frs", 21, 35, jsa_contrib_reported=300.0, afcs_reported=800.0
        ),
        _person(31, 3, 3, "frs", 31, 70, bsp_reported=1200.0),
        # SPI copies with drawn reports; their source ids name the twins.
        _person(
            105,
            50,
            5,
            "spi",
            11,
            40,
            jsa_contrib_reported=2_000.0,
            iidb_reported=9_999.0,
            income_support_reported=400.0,
            sda_reported=100.0,
        ),
        _person(
            106,
            60,
            6,
            "spi",
            21,
            35,
            universal_credit_reported=1_200.0,
            working_tax_credit_reported=300.0,
            afcs_reported=0.0,
            bsp_reported=650.0,
        ),
        _person(
            107,
            70,
            7,
            "spi",
            31,
            70,
            ssmg_reported=500.0,
            child_tax_credit_reported=50.0,
        ),
    ]
    person = pd.DataFrame(people)
    rule = (
        person[
            [
                "universal_credit_reported",
                "jsa_contrib_reported",
                "jsa_income_reported",
                "esa_contrib_reported",
                "esa_income_reported",
            ]
        ].sum(axis=1)
        > 0
    )
    # The twins' stored flag follows the rule; the SPI copies keep the twin's.
    twin_flag = dict(zip(person["person_id"], rule, strict=True))
    person["receives_benefits_in_own_right"] = [
        twin_flag[source] for source in person[support_source_id_column("person")]
    ]
    if base_own_right_override is not None:
        person.loc[0, "receives_benefits_in_own_right"] = base_own_right_override
    derived = frs_disability.derive_frs_disability(
        person, category_rates=CATEGORY_RATES, flag_rates=FLAG_RATES
    )
    for column in frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS:
        person[column] = derived[column].to_numpy()
    benunit = pd.DataFrame(
        {
            "benunit_id": [1, 2, 3, 50, 60, 70],
            support_channel_column("benunit"): [
                "frs",
                "frs",
                "frs",
                "spi",
                "spi",
                "spi",
            ],
            # Every SPI unit inherited a claim from its twin.
            "would_claim_uc": [True, True, False, True, True, True],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 5, 6, 7],
            # Each SPI copy repeats its twin's household.
            "household_source_id": [1, 2, 3, 1, 2, 3],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.asarray([10.0, 20.0, 30.0, 1.0, 2.0, 3.0]),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def _apply(frame):
    return apply_spi_benefit_coherence(
        frame,
        contract=_Contract(),
        population_policy=POLICY,
        category_rates=CATEGORY_RATES,
        flag_rates=FLAG_RATES,
        permitted_work_limit=LIMIT,
    )


def _with(frame, **columns_by_person):
    """The frame with person columns set, ``{person_id: {column: value}}``."""

    person = frame.table("person").copy()
    for person_id, values in columns_by_person.items():
        for column, value in values.items():
            person.loc[person["person_id"] == int(person_id), column] = value
    return uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        household_weights=frame.weights_for("household").values,
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def test_spi_rows_lose_out_of_work_and_legacy_reports() -> None:
    result = _apply(_frame())
    person = result.frame.table("person").set_index("person_id")

    for column in SPI_ZEROED_REPORT_COLUMNS:
        assert (person.loc[[105, 106, 107], column] == 0.0).all(), column
    assert result.zeroed["income_support_reported"]["rows_reporting_before"] == 1
    assert "jsa_contrib_reported" not in result.zeroed
    assert all(
        receipt["rows_reporting_after"] == 0 for receipt in result.zeroed.values()
    )


def test_spi_rows_lose_income_related_esa_and_their_drawn_contributory_esa() -> None:
    # policyengine-uk pays a drawn income-related ESA report as the award,
    # behind a capital test alone, whatever the copy's new incomes. A drawn
    # contributory ESA report gives way to the twin's, who reports none.
    result = _apply(
        _with(
            _frame(),
            **{"106": {"esa_income_reported": 600.0, "esa_contrib_reported": 700.0}},
        )
    )
    after = result.frame.table("person").set_index("person_id")

    assert after.loc[106, "esa_income_reported"] == 0.0
    assert after.loc[106, "esa_contrib_reported"] == 0.0
    assert result.zeroed["esa_income_reported"]["rows_reporting_before"] == 1
    assert "esa_contrib_reported" not in result.zeroed


def test_spi_rows_take_the_twins_contributory_awards_within_their_limits() -> None:
    # Twin 21 reports contributory JSA and ESA; its copy 106 earns nothing and
    # works no hours, so it takes both. Copy 105 drew JSA, but its twin 11
    # reports none.
    frame = _with(_frame(), **{"21": {"esa_contrib_reported": 700.0}})
    result = _apply(frame)
    person = result.frame.table("person").set_index("person_id")

    assert person.loc[106, "jsa_contrib_reported"] == 300.0
    assert person.loc[106, "esa_contrib_reported"] == 700.0
    assert person.loc[105, "jsa_contrib_reported"] == 0.0
    assert person.loc[[21, 106], "receives_benefits_in_own_right"].all()
    receipt = result.restored_within_earnings_limit
    assert all(item["rows_breaking_rule_after"] == 0 for item in receipt.values())
    assert result.permitted_work_limit["esa_weekly_limit"] == 183.5
    assert result.permitted_work_limit["jsa_amount_from_25"] == 90.5


#: Contributory JSA at the 2024-25 rate from 25, GBP 90.50 a week.
JSA_FULL = 90.5 * 52


@pytest.mark.parametrize(
    "copy,expected",
    [
        # No work: the twin's whole award.
        ({}, JSA_FULL),
        # The disregard itself costs nothing.
        ({"employment_income": 5.0 * 52}, JSA_FULL),
        # GBP 50 a week cuts the award by the GBP 45 above the disregard
        # (Jobseekers Act 1995 s. 4(1)(b)), to GBP 45.50 a week.
        ({"employment_income": 50.0 * 52}, 45.5 * 52),
        # At the prescribed amount, GBP 95.49 a week, the award is 1p a week.
        ({"employment_income": 95.49 * 52}, 0.01 * 52),
        # Above it, nothing (JSA Regs 1996 reg 56).
        ({"employment_income": 95.5 * 52}, 0.0),
        # GBP 150 a week is within ESA's limit but not JSA's (Vahid's case).
        ({"employment_income": 150.0 * 52}, 0.0),
        ({"self_employment_income": 50.0 * 52}, 45.5 * 52),
        # Remunerative work from 16 hours a week (reg 51(1)(a)), paid or not.
        ({"hours_worked": 16.0 * 52}, 0.0),
        ({"hours_worked": 15.5 * 52}, JSA_FULL),
        # Under 25 the lower amount, GBP 71.70, sets the limit and the award.
        ({"age": 22, "employment_income": 50.0 * 52}, (71.7 - 45.0) * 52),
        ({"age": 22, "employment_income": 76.71 * 52}, 0.0),
    ],
)
def test_contributory_jsa_follows_its_own_earnings_and_hours_limits(
    copy, expected
) -> None:
    frame = _with(
        _frame(), **{"21": {"jsa_contrib_reported": JSA_FULL}, "106": dict(copy)}
    )
    result = _apply(frame)
    person = result.frame.table("person").set_index("person_id")

    assert person.loc[106, "jsa_contrib_reported"] == pytest.approx(expected)
    # The twin keeps its own report whatever the copy earns.
    assert person.loc[21, "jsa_contrib_reported"] == JSA_FULL
    receipt = result.restored_within_earnings_limit["jsa_contrib_reported"]
    assert receipt["rows_breaking_rule_after"] == 0


def test_contributory_jsa_never_exceeds_the_twins_award() -> None:
    # Twin 21's GBP 300 a year is below the copy's GBP 45.50 a week.
    frame = _with(_frame(), **{"106": {"employment_income": 50.0 * 52}})
    person = _apply(frame).frame.table("person").set_index("person_id")

    assert person.loc[106, "jsa_contrib_reported"] == 300.0


@pytest.mark.parametrize(
    "copy,takes_award",
    [
        # Exactly the limit is within it (ESA Regs 2008 reg 45(4)).
        ({"employment_income": 183.5 * 52}, True),
        ({"employment_income": 183.5 * 52 + 1.0}, False),
        ({"self_employment_income": 20_000.0}, False),
        # A self-employment loss is not set against pay.
        ({"employment_income": 9_600.0, "self_employment_income": -1_000.0}, False),
        ({"employment_income": 6_000.0, "self_employment_income": 4_000.0}, False),
        # Permitted work is under 16 hours a week.
        ({"hours_worked": 15.5 * 52}, True),
        ({"hours_worked": 16.0 * 52}, False),
    ],
)
def test_contributory_esa_stops_outside_permitted_work(copy, takes_award) -> None:
    frame = _with(
        _frame(), **{"21": {"esa_contrib_reported": 700.0}, "106": dict(copy)}
    )
    result = _apply(frame)
    person = result.frame.table("person").set_index("person_id")

    assert person.loc[106, "esa_contrib_reported"] == (700.0 if takes_award else 0.0)
    assert person.loc[21, "esa_contrib_reported"] == 700.0
    receipt = result.restored_within_earnings_limit["esa_contrib_reported"]
    assert receipt["twin_reporters_outside_limits"] == (0 if takes_award else "<3")
    assert receipt["rows_breaking_rule_after"] == 0


def test_the_within_limit_receipt_withholds_figures_under_three_households() -> None:
    # Copy 105 loses its drawn JSA and copy 106 takes its twin's: one source
    # household carries each figure, and two the change between them.
    result = _apply(_frame())
    receipt = result.restored_within_earnings_limit["jsa_contrib_reported"]

    assert receipt["twin_reporters"] == "<3"
    assert receipt["changed"] == "<3"
    assert receipt["twin_reporters_outside_limits"] == 0
    for key in (
        "weighted_amount_before",
        "weighted_amount_after",
        "weighted_reporters_before",
        "weighted_reporters_after",
    ):
        assert receipt[key] == "<3", key
    assert result.permitted_work_limit["spi_source_households_within_limits"] == {
        "jsa_contrib_reported": "3-9",
        "esa_contrib_reported": "3-9",
    }


def test_the_within_limit_receipt_gives_figures_on_three_households() -> None:
    frame = _with(
        _frame(),
        **{
            "11": {"esa_contrib_reported": 100.0},
            "21": {"esa_contrib_reported": 200.0},
            "31": {
                "esa_contrib_reported": 300.0,
                "receives_benefits_in_own_right": True,
            },
        },
    )
    receipt = _apply(frame).restored_within_earnings_limit["esa_contrib_reported"]

    assert receipt["twin_reporters"] == "3-9"
    assert receipt["changed"] == "3-9"
    # Each copy's weight is its household's: 1, 2 and 3.
    assert receipt["weighted_amount_after"] == pytest.approx(1_400.0)
    assert receipt["weighted_reporters_after"] == pytest.approx(6.0)
    assert receipt["weighted_amount_before"] == 0.0


def test_the_receipt_gives_the_awards_resting_on_pension_payments() -> None:
    # Each copy carries GBP 300 a week of pension, above both thresholds: ESA
    # Regs 2008 reg 74 would take half the GBP 215 above GBP 85 off each
    # copy's ESA, but the awards stand, and the receipt gives how much rests
    # on them (Vahid's review of microcosm#1426).
    pensions = {
        copy: {"private_pension_income": 300.0 * 52} for copy in ("105", "106", "107")
    }
    frame = _with(
        _frame(),
        **{
            "11": {"esa_contrib_reported": 100.0},
            "21": {"esa_contrib_reported": 200.0},
            "31": {
                "esa_contrib_reported": 300.0,
                "receives_benefits_in_own_right": True,
            },
        },
        **pensions,
    )
    result = _apply(frame)
    person = result.frame.table("person").set_index("person_id")

    assert person.loc[[105, 106, 107], "esa_contrib_reported"].tolist() == [
        100.0,
        200.0,
        300.0,
    ]
    receipt = result.restored_within_earnings_limit["esa_contrib_reported"]
    # Each copy's weight is its household's: 1, 2 and 3.
    assert receipt["pension_payments_above_threshold"] == {
        "source_households": "3-9",
        "weighted_amount_after": pytest.approx(1_400.0),
        "weighted_reporters_after": pytest.approx(6.0),
    }
    assert result.permitted_work_limit["jsa_pension_disregard"] == 50.0
    assert result.permitted_work_limit["esa_pension_payments_threshold"] == 85.0


@pytest.mark.parametrize(
    "column,weekly_pension,above",
    [
        # JSA Regs 1996 reg 81 takes off the pension payments above GBP 50 a
        # week and ESA Regs 2008 reg 74 half of those above GBP 85: at the
        # threshold itself nothing would come off.
        ("jsa_contrib_reported", 50.0, False),
        ("jsa_contrib_reported", 50.01, True),
        ("esa_contrib_reported", 85.0, False),
        ("esa_contrib_reported", 85.01, True),
    ],
)
def test_the_pension_thresholds_are_reg_81s_and_reg_74s(
    column, weekly_pension, above
) -> None:
    frame = _with(
        _frame(),
        **{
            "21": {column: 700.0},
            "106": {"private_pension_income": weekly_pension * 52},
        },
    )
    result = _apply(frame)
    person = result.frame.table("person").set_index("person_id")

    # Neither award is reduced for pension payments.
    assert person.loc[106, column] == 700.0
    block = result.restored_within_earnings_limit[column][
        "pension_payments_above_threshold"
    ]
    # The one copy above the threshold rests on one source household.
    assert block["source_households"] == ("<3" if above else 0)


def _pension_block(above, sources) -> dict[str, object]:
    """The pension block of a receipt on six copies each taking GBP 100."""

    taking = np.ones(6, dtype=bool)
    return _within_limit_receipt(
        old=np.zeros(6),
        new=np.full(6, 100.0),
        spi=taking,
        twin_reports=taking,
        outside=~taking,
        reduced=~taking,
        pension_above=np.asarray(above, dtype=bool),
        weights=np.arange(1.0, 7.0),
        sources=np.asarray(sources),
    )["pension_payments_above_threshold"]


@pytest.mark.parametrize(
    "above,sources,expected",
    [
        # All six copies above the threshold, on six source households of
        # weights 1 to 6.
        ([1] * 6, range(6), ("3-9", 2_100.0, 21.0)),
        # Three above beside three below.
        ([1, 1, 1, 0, 0, 0], range(6), ("3-9", 600.0, 6.0)),
        # Four above beside two: the four's figures beside the six's would
        # give back the two's.
        ([1, 1, 1, 1, 0, 0], range(6), ("3-9", "<3", "<3")),
        ([1, 1, 0, 0, 0, 0], range(6), ("<3", "<3", "<3")),
        ([0] * 6, range(6), (0, 0.0, 0.0)),
        # Six copies of two source households count as two.
        ([1] * 6, [0, 0, 0, 1, 1, 1], ("<3", "<3", "<3")),
    ],
)
def test_the_pension_figures_withhold_what_rests_on_one_or_two_households(
    above, sources, expected
) -> None:
    support, amount, reporters = expected
    assert _pension_block(above, list(sources)) == {
        "source_households": support,
        "weighted_amount_after": amount,
        "weighted_reporters_after": reporters,
    }


def test_spi_rows_take_injury_service_and_bereavement_reports_from_their_twin() -> None:
    result = _apply(_frame())
    person = result.frame.table("person").set_index("person_id")

    assert person.loc[105, "iidb_reported"] == 500.0
    assert person.loc[106, "afcs_reported"] == 800.0
    assert person.loc[106, "bsp_reported"] == 0.0
    assert person.loc[107, "bsp_reported"] == 1200.0
    assert all(
        receipt["rows_differing_from_twin_after"] == 0
        for receipt in result.restored.values()
    )


def test_twin_restore_follows_source_ids_not_row_order() -> None:
    frame = _frame()
    shuffled = frame.table("person").sample(frac=1.0, random_state=7)
    reordered = uk_national_frame(
        person=shuffled.reset_index(drop=True),
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        household_weights=frame.weights_for("household").values,
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )
    person = _apply(reordered).frame.table("person").set_index("person_id")

    assert person.loc[105, "iidb_reported"] == 500.0
    assert person.loc[107, "bsp_reported"] == 1200.0


def test_benefits_in_own_right_follow_the_rows_own_reports() -> None:
    result = _apply(_frame())
    person = result.frame.table("person").set_index("person_id")

    # 105 inherited True (twin reports UC) but its only own report was a drawn
    # JSA its twin does not report; 106 reports UC on its own; 107 reports
    # none.
    assert not person.loc[105, "receives_benefits_in_own_right"]
    assert person.loc[106, "receives_benefits_in_own_right"]
    assert not person.loc[107, "receives_benefits_in_own_right"]
    assert result.benefits_in_own_right["mismatches_after"] == 0


def test_a_base_row_that_disagrees_with_its_own_reports_refuses() -> None:
    with pytest.raises(ValueError, match="FRS-channel person"):
        _apply(_frame(base_own_right_override=False))


def test_disability_flags_follow_the_final_reports() -> None:
    before = _frame().table("person").set_index("person_id")
    person = _apply(_frame()).frame.table("person").set_index("person_id")

    # SDA (zeroed) no longer flags 105; its restored IIDB flags it instead.
    assert before.loc[105, "is_disabled_for_benefits"]
    assert person.loc[105, "is_disabled_for_benefits"]
    # 106 drew no AFCS but its twin reports it.
    assert not before.loc[106, "is_disabled_for_benefits"]
    assert person.loc[106, "is_disabled_for_benefits"]


def test_spi_universal_credit_take_up_follows_the_units_own_reports() -> None:
    result = _apply(_frame())
    benunit = result.frame.table("benunit").set_index("benunit_id")

    # 60 reports UC; 70 is a pensioner unit outside the population and reports
    # nothing; base units keep their flags.
    assert benunit.loc[60, "would_claim_uc"]
    assert not benunit.loc[70, "would_claim_uc"]
    assert benunit.loc[[1, 2, 3], "would_claim_uc"].tolist() == [True, True, False]
    assert result.universal_credit_take_up["reporters_not_claiming"] == 0
    assert (
        result.universal_credit_take_up["outside_population_non_reporters_claiming"]
        == 0
    )


def test_base_rows_are_never_modified() -> None:
    frame = _frame()
    result = _apply(frame)
    for entity in ("person", "benunit"):
        before = frame.table(entity)
        after = result.frame.table(entity)
        channel = support_channel_column(entity)
        base = before[channel] == "frs"
        pd.testing.assert_frame_equal(
            before.loc[base].reset_index(drop=True),
            after.loc[after[channel] == "frs"].reset_index(drop=True),
        )


def test_the_pass_is_idempotent() -> None:
    once = _apply(_frame()).frame
    twice = _apply(once).frame
    for entity in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(once.table(entity), twice.table(entity))


def _committed_stage() -> SourceStageSpec:
    spec = load_country_spec("uk")
    return next(
        stage
        for stage in spec.sources.stages
        if stage.stage == SPI_BENEFIT_COHERENCE_STAGE_NAME
    )


def test_committed_manifest_declares_the_stage_the_code_runs() -> None:
    stage = _committed_stage()
    assert_spi_benefit_coherence_stage_parameters(stage)
    names = [entry.stage for entry in load_country_spec("uk").sources.stages]
    # After the last UC report writer, before every UC, PC and CB consumer.
    position = names.index(SPI_BENEFIT_COHERENCE_STAGE_NAME)
    assert names[position - 1] == "uc_reporter_redraw"
    assert names[position + 1] == "uc_capital_coherence"


def test_manifest_drift_refuses() -> None:
    stage = _committed_stage()
    operations = [
        {"kind": operation.kind, **dict(operation.parameters)}
        for operation in stage.operations
    ]
    operations[0]["columns"] = operations[0]["columns"][:-1]
    drifted = SourceStageSpec.from_mapping({**stage.__dict__, "operations": operations})
    with pytest.raises(ValueError, match="operations drifted"):
        assert_spi_benefit_coherence_stage_parameters(drifted)
