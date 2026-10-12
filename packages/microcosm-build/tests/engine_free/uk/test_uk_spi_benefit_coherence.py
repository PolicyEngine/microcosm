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
    SPI_ZEROED_REPORT_COLUMNS,
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


class _Contract:
    def rate(self, key: str) -> float:
        assert key == "universal_credit"
        return 0.55


REPORT_COLUMNS = tuple(
    dict.fromkeys(
        (
            *SPI_ZEROED_REPORT_COLUMNS,
            *SPI_RESTORED_REPORT_COLUMNS,
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
        support_channel_column("person"): channel,
        support_source_id_column("person"): source_id,
    }
    row.update(dict.fromkeys(REPORT_COLUMNS, 0.0))
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
    household = pd.DataFrame({"household_id": [1, 2, 3, 5, 6, 7]})
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
    )


def test_spi_rows_lose_out_of_work_and_legacy_reports() -> None:
    result = _apply(_frame())
    person = result.frame.table("person").set_index("person_id")

    for column in SPI_ZEROED_REPORT_COLUMNS:
        assert (person.loc[[105, 106, 107], column] == 0.0).all(), column
    assert person.loc[21, "jsa_contrib_reported"] == 300.0
    assert result.zeroed["jsa_contrib_reported"]["rows_reporting_before"] == 1
    assert all(
        receipt["rows_reporting_after"] == 0 for receipt in result.zeroed.values()
    )


def test_spi_rows_lose_income_related_esa_and_keep_contributory_esa() -> None:
    # policyengine-uk pays a drawn income-related ESA report as the award,
    # behind a capital test alone, whatever the copy's new incomes; contributory
    # ESA is open to new claims and not income-tested, so the copy keeps it.
    frame = _frame()
    person = frame.table("person").copy()
    copy = person["person_id"] == 106
    person.loc[copy, "esa_income_reported"] = 600.0
    person.loc[copy, "esa_contrib_reported"] = 700.0
    drawn = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        household_weights=frame.weights_for("household").values,
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )
    result = _apply(drawn)
    after = result.frame.table("person").set_index("person_id")

    assert after.loc[106, "esa_income_reported"] == 0.0
    assert after.loc[106, "esa_contrib_reported"] == 700.0
    assert result.zeroed["esa_income_reported"]["rows_reporting_before"] == 1
    assert "esa_contrib_reported" not in result.zeroed


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

    # 105 inherited True (twin reports UC) but its only own report was JSA,
    # now zeroed; 106 reports UC on its own; 107 reports none.
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
