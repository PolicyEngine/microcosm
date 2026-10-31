"""The FRS annual income descriptors (frs_annual_income_spells).

Synthetic adult, job and household tables in the shape the stage reads them
after ``normalize_ids``: one spell rule per person, beside forty donors who
are current employees in their first two years and report usual pay.
"""

from __future__ import annotations

import json
import math
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.uk_runtime import frs_annual_income as fai
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

INTERVIEW = pd.Timestamp("2024-10-15")
DONORS = 40
DONOR_WEEKLY = 500.0
DONOR_USUAL = 490.0
RATIO = DONOR_WEEKLY / DONOR_USUAL
EMPLOYEE, UNEMPLOYED, SELF_EMPLOYED = 1, 5, 3


def _calendar(now: int, months: dict[int, int] | None = None) -> dict[str, float]:
    """SELFDEMP and SDEMP01-12: ``months`` maps months ago to a code."""

    row = {"selfdemp": float(now)}
    for month in range(1, 13):
        row[f"sdemp{month:02d}"] = float((months or {}).get(month, now))
    return row


def _population(people: dict[str, dict]) -> tuple:
    """Donors in households 1..40 and one household per named person."""

    adult_rows, job_rows, person_rows = [], [], []
    households: list[int] = []

    def add(household: int, adult: dict | None, job: dict | None, person: dict) -> None:
        person_id = household * 1000 + 1
        households.append(household)
        person_rows.append(
            {
                "person_id": person_id,
                "person_household_id": household,
                "age": 40,
                "gender": "MALE",
                "employment_income": 0.0,
                "self_employment_income": 0.0,
                **person,
            }
        )
        if adult is not None:
            adult_rows.append(
                {
                    "household_id": household,
                    "benunit_id": household * 100 + 1,
                    "person_id": person_id,
                    "samesit": 2.0,
                    "working": 1.0,
                    "jobaway": np.nan,
                    "numjob2": np.nan,
                    "currjobm": np.nan,
                    "prevjobm": np.nan,
                    "lstwrk1": np.nan,
                    "lstwrk2": np.nan,
                    "lstyr": np.nan,
                    "retire": np.nan,
                    "retire1": np.nan,
                    "empstati": 1.0,
                    **_calendar(EMPLOYEE),
                    **adult,
                }
            )
        if job is not None:
            job_rows.append(
                {
                    "household_id": household,
                    "benunit_id": household * 100 + 1,
                    "person_id": person_id,
                    "jobtype": 1.0,
                    "etype": 1.0,
                    "ftpt": 1.0,
                    "wrkprev": np.nan,
                    "ugross": np.nan,
                    "ugrspay": np.nan,
                    "sejblong": np.nan,
                    "sejbmths": np.nan,
                    "selwks": np.nan,
                    "seincamt": np.nan,
                    "profit1": np.nan,
                    "profit2": np.nan,
                    **job,
                }
            )

    for donor in range(1, DONORS + 1):
        add(
            donor,
            {"currjobm": 20.0},
            {"ugross": DONOR_USUAL, "ugrspay": DONOR_WEEKLY},
            {"employment_income": DONOR_WEEKLY * WEEKS_IN_YEAR},
        )
    index: dict[str, int] = {}
    for offset, (name, spec) in enumerate(people.items()):
        household = 101 + offset
        index[name] = household * 1000 + 1
        add(household, spec.get("adult"), spec.get("job"), spec.get("person", {}))
    person = pd.DataFrame(person_rows)
    adult = pd.DataFrame(adult_rows)
    job = pd.DataFrame(job_rows) if job_rows else pd.DataFrame(columns=adult.columns)
    interview = pd.Series(INTERVIEW, index=pd.Index(households, name="household_id"))
    return person, adult, job, interview, index


def _derive(people: dict[str, dict], *, growth: float = 0.0):
    person, adult, job, interview, index = _population(people)
    descriptors, receipt = fai.derive_frs_annual_income_spells(
        person,
        person_weights=np.full(len(person), 1000.0),
        adult=adult,
        job=job,
        interview=interview,
        earnings_growth=growth,
    )
    descriptors.index = person["person_id"].to_numpy()
    return descriptors, receipt, index


LEAVER = {"working": 2.0, "jobaway": 2.0, "numjob2": 1.0, "empstati": 5.0}


def test_full_year_employee_keeps_the_whole_year() -> None:
    descriptors, _, index = _derive(
        {
            "full": {
                "adult": {"currjobm": 100.0},
                "job": {},
                "person": {"employment_income": 20000.0},
            }
        }
    )
    row = descriptors.loc[index["full"]]
    assert row[fai.EMPLOYMENT_INCOME_ANNUAL_SHARE] == 1.0
    assert row[fai.EMPLOYMENT_INCOME_OTHER_SPELLS] == 0.0


def test_entrant_is_paid_for_the_months_in_the_job() -> None:
    descriptors, _, index = _derive(
        {
            "entrant": {
                # CURRJOBM 5.4 is the sixth month in the job; nothing before it.
                "adult": {"currjobm": 5.4},
                "job": {"wrkprev": 3.0},
                "person": {"employment_income": 30000.0},
            }
        }
    )
    assert descriptors.loc[index["entrant"], fai.EMPLOYMENT_INCOME_ANNUAL_SHARE] == 0.5


def test_job_changer_keeps_the_year_unless_the_calendar_records_a_gap() -> None:
    gap = _calendar(EMPLOYEE, {4: UNEMPLOYED, 5: UNEMPLOYED, 6: UNEMPLOYED})
    descriptors, _, index = _derive(
        {
            "continuous": {
                "adult": {"currjobm": 3.0},
                "job": {"wrkprev": 1.0},
                "person": {"employment_income": 30000.0},
            },
            # Four months in the job; the calendar shows three months out of
            # work before it and employee months 7 to 11 months ago.
            "gap": {
                "adult": {"currjobm": 3.0, "samesit": 1.0, **gap},
                "job": {"wrkprev": 1.0},
                "person": {"employment_income": 30000.0},
            },
        }
    )
    assert (
        descriptors.loc[index["continuous"], fai.EMPLOYMENT_INCOME_ANNUAL_SHARE] == 1.0
    )
    assert descriptors.loc[index["gap"], fai.EMPLOYMENT_INCOME_ANNUAL_SHARE] == 0.75


def test_leaver_is_paid_the_last_jobs_usual_pay_for_their_weeks() -> None:
    descriptors, receipt, index = _derive(
        {
            "leaver": {
                "adult": {**LEAVER, "lstyr": 26.0, "lstwrk2": 2024.0, "lstwrk1": 7.0},
                "job": {"ugross": 300.0},
            }
        }
    )
    expected = 300.0 * RATIO * WEEKS_IN_YEAR * 6.0 / 12.0
    row = descriptors.loc[index["leaver"]]
    assert row[fai.EMPLOYMENT_INCOME_OTHER_SPELLS] == pytest.approx(expected)
    # Contributory benefits are in payment for the six months not worked.
    assert row[fai.CONTRIBUTORY_BENEFITS_ANNUAL_SHARE] == pytest.approx(0.5)
    assert receipt["pay_concept_ratio"] == pytest.approx(RATIO)


def test_leaver_pay_is_uprated_for_the_months_since_leaving() -> None:
    plain, _, index = _derive(
        {
            "leaver": {
                "adult": {**LEAVER, "lstyr": 26.0, "lstwrk2": 2024.0, "lstwrk1": 4.0},
                "job": {"ugross": 300.0},
            }
        }
    )
    uprated, _, _ = _derive(
        {
            "leaver": {
                "adult": {**LEAVER, "lstyr": 26.0, "lstwrk2": 2024.0, "lstwrk1": 4.0},
                "job": {"ugross": 300.0},
            }
        },
        growth=0.05,
    )
    column = fai.EMPLOYMENT_INCOME_OTHER_SPELLS
    assert uprated.loc[index["leaver"], column] == pytest.approx(
        plain.loc[index["leaver"], column] * 1.05 ** (6 / 12)
    )


def test_zero_weeks_of_regular_work_fall_through_to_the_leaving_date() -> None:
    descriptors, _, index = _derive(
        {
            "irregular": {
                "adult": {**LEAVER, "lstyr": 0.0, "lstwrk2": 2024.0, "lstwrk1": 8.0},
                "job": {"ugross": 300.0},
            }
        }
    )
    months = fai.LEAVING_DATE_SHARE * (12 - 2)
    assert descriptors.loc[
        index["irregular"], fai.EMPLOYMENT_INCOME_OTHER_SPELLS
    ] == pytest.approx(300.0 * RATIO * WEEKS_IN_YEAR * months / 12.0)


def test_a_leaving_date_before_the_window_gives_no_months() -> None:
    descriptors, _, index = _derive(
        {
            "long_ago": {
                "adult": {**LEAVER, "lstwrk2": 2023.0, "lstwrk1": 3.0},
                "job": {"ugross": 300.0},
            }
        }
    )
    assert descriptors.loc[index["long_ago"], fai.EMPLOYMENT_INCOME_OTHER_SPELLS] == 0.0


def test_recent_retiree_without_a_leaving_date_uses_the_month_retired() -> None:
    descriptors, _, index = _derive(
        {
            "retired": {
                # Retired in April, interviewed in October: six months ago.
                "adult": {**LEAVER, "retire": 1.0, "retire1": 4.0, "empstati": 6.0},
                "job": {"ugross": 200.0},
            }
        }
    )
    row = descriptors.loc[index["retired"]]
    months = fai.LEAVING_DATE_SHARE * (12 - 6)
    assert row[fai.EMPLOYMENT_INCOME_OTHER_SPELLS] == pytest.approx(
        200.0 * RATIO * WEEKS_IN_YEAR * months / 12.0
    )
    assert row[fai.PRIVATE_PENSION_INCOME_ANNUAL_SHARE] == pytest.approx(6.5 / 12)


def test_people_in_work_are_never_leavers() -> None:
    switch = _calendar(SELF_EMPLOYED, {m: EMPLOYEE for m in range(6, 12)})
    descriptors, receipt, index = _derive(
        {
            # Employee six to eleven months ago, self-employed since.
            "switcher": {
                "adult": {"samesit": 1.0, "empstati": 3.0, **switch},
                "job": {"etype": 3.0, "sejblong": 1.0},
                "person": {"self_employment_income": 15000.0},
            }
        }
    )
    row = descriptors.loc[index["switcher"]]
    # A donor's weekly pay for the six employee months.
    assert row[fai.EMPLOYMENT_INCOME_OTHER_SPELLS] == pytest.approx(
        DONOR_WEEKLY * WEEKS_IN_YEAR * 6 / 12
    )
    assert row[fai.SELF_EMPLOYMENT_INCOME_ANNUAL_SHARE] == pytest.approx(6 / 12)
    # Nothing on a contributory benefit stops: they never left work.
    assert row[fai.CONTRIBUTORY_BENEFITS_ANNUAL_SHARE] == 1.0


def test_self_employment_new_and_ceased() -> None:
    descriptors, _, index = _derive(
        {
            "new_business": {
                "adult": {"empstati": 3.0, **_calendar(SELF_EMPLOYED)},
                "job": {"etype": 4.0, "sejblong": 0.0, "sejbmths": 5.0},
                "person": {"self_employment_income": 10000.0},
            },
            "ceased": {
                "adult": {**LEAVER, **_calendar(UNEMPLOYED)},
                "job": {"etype": 4.0, "seincamt": 250.0, "selwks": 26.0},
            },
        }
    )
    assert descriptors.loc[
        index["new_business"], fai.SELF_EMPLOYMENT_INCOME_ANNUAL_SHARE
    ] == pytest.approx(6 / 12)
    assert descriptors.loc[
        index["ceased"], fai.SELF_EMPLOYMENT_INCOME_OTHER_SPELLS
    ] == pytest.approx(250.0 * WEEKS_IN_YEAR * 6 / 12)
    # A ceased self-employed person is not an employee leaver.
    assert descriptors.loc[index["ceased"], fai.EMPLOYMENT_INCOME_OTHER_SPELLS] == 0.0


def test_the_calendar_bounds_a_business_only_when_it_is_the_main_activity() -> None:
    descriptors, _, index = _derive(
        {
            # Self-employed for the last four months, an employee before.
            "moved_into_business": {
                "adult": {
                    "samesit": 1.0,
                    "empstati": 3.0,
                    **_calendar(
                        SELF_EMPLOYED, {month: EMPLOYEE for month in range(4, 13)}
                    ),
                },
                "job": {"etype": 4.0, "sejblong": 1.0},
                "person": {"self_employment_income": 8000.0},
            },
            # An employee all year beside a business the calendar cannot see.
            "side_business": {
                "adult": {"samesit": 1.0, "currjobm": 30.0},
                "job": {"sejblong": 3.0},
                "person": {
                    "employment_income": 26000.0,
                    "self_employment_income": 3000.0,
                },
            },
        }
    )
    share = fai.SELF_EMPLOYMENT_INCOME_ANNUAL_SHARE
    assert descriptors.loc[index["moved_into_business"], share] == pytest.approx(4 / 12)
    assert descriptors.loc[index["side_business"], share] == 1.0


def test_people_outside_adult_tab_keep_their_current_columns() -> None:
    descriptors, _, index = _derive({"child": {"person": {"age": 8}}})
    row = descriptors.loc[index["child"]]
    for column in fai.ANNUAL_SHARE_COLUMNS:
        assert row[column] == 1.0
    for column in fai.OTHER_SPELLS_COLUMNS:
        assert row[column] == 0.0


def test_draws_are_keyed_on_identity() -> None:
    people = {
        "calendar_leaver": {
            "adult": {
                "working": 2.0,
                "samesit": 1.0,
                "empstati": 5.0,
                **_calendar(UNEMPLOYED, {m: EMPLOYEE for m in range(3, 9)}),
            }
        }
    }
    first, _, _ = _derive(people)
    second, _, _ = _derive(people)
    pd.testing.assert_frame_equal(first, second)


def test_receipt_gives_bands_and_structural_zeros() -> None:
    _, receipt, _ = _derive(
        {
            "entrant": {
                "adult": {"currjobm": 5.4},
                "job": {"wrkprev": 3.0},
                "person": {"employment_income": 30000.0},
            }
        }
    )
    assert set(receipt["spell_sources"]) == set(fai.SPELL_SOURCES)
    # A source carried by one household gives a band, never a count.
    assert receipt["spell_sources"][fai.SOURCE_ENTRANT] == "<3"
    for key in (
        "full_year_rows_not_whole",
        "rows_outside_share_bounds",
        "rows_negative_other_spells",
    ):
        assert receipt[key] == 0


@pytest.mark.parametrize(
    ("broken", "count"),
    [(0, 0), (1, "<3"), (3, "3-9"), (12, "10-49"), (150, 200)],
)
def test_receipt_bands_its_structural_counts(broken: int, count: object) -> None:
    # A count of rows that break an invariant says something about the tail,
    # so the receipt bands it as the other stages' receipts do, zero exact.
    rows = 200
    descriptors = pd.DataFrame(
        {column: np.ones(rows) for column in fai.ANNUAL_SHARE_COLUMNS}
        | {column: np.zeros(rows) for column in fai.OTHER_SPELLS_COLUMNS}
    )
    employment_share = np.ones(rows)
    employment_share[:broken] = 0.5
    receipt = fai._receipt(
        pd.DataFrame({"person_household_id": np.arange(rows)}),
        weights=np.ones(rows),
        descriptors=descriptors,
        current=np.zeros(rows),
        se_current=np.zeros(rows),
        pay=np.zeros(rows, dtype=bool),
        source=np.full(rows, fai.SOURCE_FULL_YEAR),
        observed=np.zeros(rows, dtype=bool),
        needs_donor=np.zeros(rows, dtype=bool),
        donor_cells={},
        ratio=1.0,
        ratio_support=0,
        earnings_growth=0.0,
        employment_share=employment_share,
    )
    assert receipt["full_year_rows_not_whole"] == count
    assert receipt["rows_outside_share_bounds"] == 0
    assert receipt["rows_negative_other_spells"] == 0


def _gate(evidence: dict):
    """The declared stage-health gate, called as the gate battery calls it."""

    gates = json.loads(files("microcosm.build.uk").joinpath("gates.json").read_text())
    entry = next(
        gate
        for gate in gates["gates"]
        if gate["id"] == "uk_stage_frs_annual_income_spells"
    )
    assert entry["gate"] == "stage_health"
    assert entry["evidence_absent_blocks"] is True
    parameters = entry["parameters"]
    return uk_stage_health_gate(
        evidence=evidence,
        stage=parameters["stage"],
        check=parameters["check"],
        parameters=parameters,
    )


def _leaver_evidence() -> dict:
    _, receipt, _ = _derive(
        {
            "leaver": {
                "adult": {**LEAVER, "lstyr": 26.0, "lstwrk2": 2024.0, "lstwrk1": 7.0},
                "job": {"ugross": 300.0},
            }
        }
    )
    return {"stage": fai.FRS_ANNUAL_INCOME_STAGE_NAME, **receipt}


def test_gate_passes_the_stages_own_receipt() -> None:
    gate = _gate(_leaver_evidence())
    assert gate.passed, gate.failures
    # One household carries the pay outside the current job: a band, unmeasured.
    assert gate.details["other_spells_pay_observed_share"] == "<3"
    assert gate.details["observed_pay_share_measured"] is False


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        # The receipt bands a count of broken rows (extreme_count).
        ("full_year_rows_not_whole", "<3", "full_year_rows_not_whole is '<3'"),
        ("rows_outside_share_bounds", 100, "rows_outside_share_bounds is 100"),
        ("rows_negative_other_spells", None, "rows_negative_other_spells is None"),
        # Weekly pay read as a monthly amount gives a ratio far above one.
        ("pay_concept_ratio", 3.58, "pay_concept_ratio 3.5800 lies outside"),
        # Spells add part-year earners; they cannot remove a current one.
        ("annual_over_current_earners", 0.98, "annual_over_current_earners 0.9800"),
        ("other_spells_pay_observed_share", 0.5, "only 0.5000 of the pay"),
        ("other_spells_pay_observed_share", None, "observed_share is None"),
    ],
)
def test_gate_fails_closed(key: str, value: object, message: str) -> None:
    # A measured ratio rests on ten or more households.
    gate = _gate(
        {**_leaver_evidence(), "pay_concept_ratio_support": "10-29", key: value}
    )
    assert not gate.passed
    assert any(message in failure for failure in gate.failures), gate.failures


def test_gate_refuses_another_stages_receipt() -> None:
    gate = _gate({**_leaver_evidence(), "stage": "frs_hmrc_spine_leaves"})
    assert not gate.passed
    assert "does not match" in gate.failures[0]


def test_pay_concept_ratio_measures_few_reporters_and_refuses_none() -> None:
    """Fewer than ten reporting households only flags (María's ruling of
    2026-10-26): the ratio is measured, and the receipt prints it from three
    households up ("<3" below, the disclosure floor of 2026-10-29)."""
    person, adult, job, interview, _ = _population({})
    ratio, support = fai.pay_concept_ratio(
        person["person_id"].to_numpy(),
        np.ones(len(person)),
        job.iloc[:5],
        np.ones(len(person), dtype=bool),
    )
    assert 0 < support < 10 and math.isfinite(ratio)
    with pytest.raises(ValueError, match="no household reports both"):
        fai.pay_concept_ratio(
            person["person_id"].to_numpy(),
            np.ones(len(person)),
            job.iloc[:0],
            np.ones(len(person), dtype=bool),
        )


def _reporting_jobs(households: int, ratio: float) -> tuple:
    """One current employee main job a household, at one DV-to-usual ratio."""
    person_ids = np.arange(1, households + 1)
    job = pd.DataFrame(
        {
            "person_id": person_ids,
            "household_id": person_ids,
            "jobtype": 1,
            "etype": fai.ETYPE_EMPLOYEE_CODE,
            "ugross": 500.0,
            "ugrspay": 500.0 * ratio,
        }
    )
    return person_ids, np.ones(households), job, np.ones(households, dtype=bool)


@pytest.mark.parametrize("measured, used", [(1.5, 1.10), (0.5, 0.95), (1.02, 1.02)])
def test_a_thin_pay_concept_ratio_is_held_to_its_bounds(measured, used) -> None:
    """On fewer than ten households a ratio past the plausibility bounds is
    noise: it is held to them, and the build goes on (Vahid's second review
    of #1396). On ten or more it still refuses."""
    ratio, support = fai.pay_concept_ratio(*_reporting_jobs(5, measured))
    assert support == 5 and ratio == pytest.approx(used)
    with pytest.raises(ValueError, match="lies outside"):
        fai.pay_concept_ratio(*_reporting_jobs(10, 1.5))


@pytest.mark.parametrize("key", ["pay_concept_ratio", "annual_over_current_earners"])
def test_gate_flags_a_ratio_on_fewer_than_three_households(key: str) -> None:
    gate = _gate({**_leaver_evidence(), key: "<3"})
    assert gate.passed, gate.failures
    assert gate.details[f"{key}_measured"] is False


@pytest.mark.parametrize(("support", "measured"), [("3-9", False), ("10-29", True)])
def test_gate_reads_a_ratio_held_to_its_bounds_unmeasured(support, measured) -> None:
    """The stage holds a ratio on fewer than ten households to its bounds, so
    the gate cannot judge it, though the receipt prints it from three up."""
    gate = _gate(
        {
            **_leaver_evidence(),
            "pay_concept_ratio": 1.10,
            "pay_concept_ratio_support": support,
        }
    )
    assert gate.passed, gate.failures
    assert gate.details.get("pay_concept_ratio_measured", True) is measured
    assert gate.details.get("pay_concept_ratio_held_to_bounds", False) is not measured


def _stage(**overrides) -> SourceStageSpec:
    operations = tuple(
        SourceOperationSpec(kind=kind, parameters=parameters)
        for kind, parameters in fai.frs_annual_income_operation_parameters()
    )
    fields = {
        "stage": fai.FRS_ANNUAL_INCOME_STAGE_NAME,
        "survey": "Family Resources Survey 2024-25",
        "source": "test",
        "grain": "person",
        "artifacts": (
            {"role": "frs_table", "table": "adult", "locator": "adult.tab"},
            {"role": "frs_table", "table": "job", "locator": "job.tab"},
            {"role": "frs_table", "table": "househol", "locator": "househol.tab"},
            {"role": "earnings_index", "resource": fai.EARNINGS_INDEX_RESOURCE},
        ),
        "operations": operations,
        "outputs": fai.FRS_ANNUAL_INCOME_OUTPUT_COLUMNS,
    }
    fields.update(overrides)
    return SourceStageSpec(**fields)


def test_stage_parameters_are_closed_world() -> None:
    fai._assert_stage_parameters(_stage())
    drifted = tuple(
        SourceOperationSpec(
            kind=operation.kind,
            parameters={**operation.parameters, "leaving_date_share": 0.5}
            if operation.kind == fai.FRS_ANNUAL_INCOME_OPERATION_KIND
            else operation.parameters,
        )
        for operation in _stage().operations
    )
    with pytest.raises(ValueError, match="drifted"):
        fai._assert_stage_parameters(_stage(operations=drifted))
    with pytest.raises(ValueError, match="no rewrites"):
        fai._assert_stage_parameters(_stage(rewrites=("employment_income",)))


def test_earnings_growth_comes_from_the_pinned_index() -> None:
    growth = fai.load_earnings_growth()
    assert 0.0 < growth < 0.1
