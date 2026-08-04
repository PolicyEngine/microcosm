"""FRS occupation merge: synthetic adult.tab join, missingness, stage contract.

The real FRS adult.tab is UKDS EUL microdata and never committed; every test
builds a tiny fake adult table plus a matching person table using the
PolicyEngine-UK single-year composite-ID convention
(``person_id = SERNUM * 1000 + PERSON``; ``BENUNIT`` does not enter). The
expected IDs are written as literals taken from the upstream PolicyEngine-UK
FRS data build's contract (cited precisely in the join-keys section of
:mod:`populace.build.uk_runtime.frs_occupation`), not recomputed with the
code under test, so a formula regression cannot hide behind its own fixtures —
the pre-fix ``SERNUM*100 + BENUNIT*10 + PERSON`` composite passed exactly
such self-consistent tests while matching zero real adults. The final,
skipped-by-default test joins the real adult.tab against the same-vintage
PolicyEngine-UK H5 when both are available locally.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from populace.build.plan import Stage, StagePlan
from populace.build.uk_runtime.frs_occupation import (
    FRS_OCCUPATION_DONOR,
    SOC_MAJOR_GROUP_COLUMN,
    UK_FRS_OCCUPATION_STAGE_NAME,
    attach_soc_major_group,
    frs_major_group_to_digit,
    frs_occupation_stage,
    frs_person_ids,
    load_frs_adult_table,
    soc_major_group_for_persons,
)
from populace.frame import EntitySchema, Frame, WeightKind, Weights

SCHEMA = EntitySchema(group_entities=("household",))


def _fake_adult() -> pd.DataFrame:
    """Three adults in two households; one adult with an invalid SOC.

    ``BENUNIT`` deliberately varies (household 10 spans two benefit units)
    to pin that it does not enter the person ID.
    """

    return pd.DataFrame(
        {
            "SERNUM": [10, 10, 20],
            "BENUNIT": [1, 2, 1],
            "PERSON": [1, 2, 1],
            "SOC2020": [2000, 9000, -1],
        }
    )


def _fake_person() -> pd.DataFrame:
    """The two households' persons: three adults plus one child (no adult row)."""

    return pd.DataFrame(
        {
            "person_id": [10001, 10002, 20001, 20002],
            "person_household_id": [10, 10, 20, 20],
            "age": [40, 38, 30, 5],
        }
    )


def _frame(person: pd.DataFrame) -> Frame:
    household = pd.DataFrame(
        {"household_id": sorted(person["person_household_id"].unique())}
    )
    return Frame(
        {"person": person, "household": household},
        SCHEMA,
        {"household": Weights(values=np.ones(len(household)), kind=WeightKind.DESIGN)},
    )


def test_frs_person_ids_composite_convention():
    """``SERNUM * 1000 + PERSON``, expected values as upstream literals."""

    ids = frs_person_ids(_fake_adult())
    assert ids.tolist() == [10001, 10002, 20001]


def test_frs_person_ids_ignore_benunit():
    """Same SERNUM/PERSON under different BENUNITs → identical IDs."""

    adult = _fake_adult()
    rebenunited = adult.assign(BENUNIT=[9, 8, 7])
    assert frs_person_ids(adult).tolist() == frs_person_ids(rebenunited).tolist()


def test_frs_person_ids_does_not_require_benunit():
    ids = frs_person_ids(pd.DataFrame({"SERNUM": [5], "PERSON": [3]}))
    assert ids.tolist() == [5003]


def test_frs_person_ids_requires_key_columns():
    with pytest.raises(ValueError, match="key column"):
        frs_person_ids(pd.DataFrame({"SERNUM": [1], "BENUNIT": [1]}))


def test_join_correctness_and_missingness_report():
    values, report = soc_major_group_for_persons(_fake_person(), _fake_adult())
    assert values.name == SOC_MAJOR_GROUP_COLUMN
    # Adults with valid SOC keep the raw 1000-9000 coding.
    assert values.iloc[0] == 2000.0
    assert values.iloc[1] == 9000.0
    # Adult with invalid SOC (-1) and the child both stay NaN.
    assert np.isnan(values.iloc[2])
    assert np.isnan(values.iloc[3])
    assert report == {
        "n_persons": 4,
        "adults_matched": 3,
        "adults_with_valid_soc": 2,
        "adults_missing_soc": 1,
        "persons_without_adult_row": 1,
    }


def test_duplicate_adult_keys_rejected():
    adult = pd.concat([_fake_adult(), _fake_adult().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated SERNUM"):
        soc_major_group_for_persons(_fake_person(), adult)


def test_duplicate_person_across_benunits_rejected():
    """Two adults sharing SERNUM/PERSON collide even in different benunits.

    The composite assumes PERSON is unique within a household; if a vintage
    ever numbers PERSON within the benefit unit instead, the join must fail
    loudly rather than silently mis-assigning occupations.
    """

    adult = pd.DataFrame(
        {
            "SERNUM": [10, 10],
            "BENUNIT": [1, 2],
            "PERSON": [1, 1],
            "SOC2020": [2000, 3000],
        }
    )
    with pytest.raises(ValueError, match="duplicated SERNUM"):
        soc_major_group_for_persons(_fake_person(), adult)


def test_zero_match_raises_instead_of_all_nan():
    """Person IDs on a different convention → loud failure, not silent NaN.

    This is the regression shape of the original bug: an ID-convention
    mismatch joined 0 of 34,966 real persons and would have produced an
    all-NaN column. The fixture's IDs are exactly the pre-fix composite
    (``SERNUM*100 + BENUNIT*10 + PERSON``) for the same fake households.
    """

    person = _fake_person().assign(person_id=[1011, 1022, 2011, 2012])
    with pytest.raises(ValueError, match="convention"):
        soc_major_group_for_persons(person, _fake_adult())


def test_load_frs_adult_table_reads_tab_delimited(tmp_path):
    path = tmp_path / "adult.tab"
    _fake_adult().assign(EXTRA=1).to_csv(path, sep="\t", index=False)
    table = load_frs_adult_table(path)
    assert sorted(table.columns) == ["BENUNIT", "PERSON", "SERNUM", "SOC2020"]
    assert len(table) == 3


def test_load_frs_adult_table_missing_column(tmp_path):
    path = tmp_path / "adult.tab"
    _fake_adult().drop(columns=["SOC2020"]).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="SOC2020"):
        load_frs_adult_table(path)


def test_frs_major_group_to_digit():
    digits = frs_major_group_to_digit([1000, 9000, -1, np.nan, 4000])
    assert digits.tolist() == ["1", "9", None, None, "4"]


def test_attach_soc_major_group_on_frame():
    frame = attach_soc_major_group(_frame(_fake_person()), _fake_adult())
    column = frame.person[SOC_MAJOR_GROUP_COLUMN]
    assert column.tolist()[:2] == [2000.0, 9000.0]
    assert column.isna().sum() == 2


def test_attach_refuses_double_run():
    frame = attach_soc_major_group(_frame(_fake_person()), _fake_adult())
    with pytest.raises(ValueError, match="already exists"):
        attach_soc_major_group(frame, _fake_adult())


def test_stage_contract():
    stage = frs_occupation_stage(_fake_adult())
    assert isinstance(stage, Stage)
    assert stage.name == UK_FRS_OCCUPATION_STAGE_NAME
    assert stage.produces == (SOC_MAJOR_GROUP_COLUMN,)
    assert stage.consumes == ("person_id",)
    assert stage.donor == FRS_OCCUPATION_DONOR
    out = stage.transform(_frame(_fake_person()))
    assert SOC_MAJOR_GROUP_COLUMN in out.person.columns


def test_stage_reads_adult_tab_path_lazily(tmp_path):
    path = tmp_path / "adult.tab"
    stage = frs_occupation_stage(path)  # not written yet: read at run time
    _fake_adult().to_csv(path, sep="\t", index=False)
    out = stage.transform(_frame(_fake_person()))
    assert out.person[SOC_MAJOR_GROUP_COLUMN].iloc[0] == 2000.0


def test_stage_slots_before_exposure_imputation_in_a_plan():
    """StagePlan accepts frs_occupation producing what a consumer needs."""

    consumer = Stage(
        name="ai_exposure_imputation",
        transform=lambda frame: frame,
        consumes=(SOC_MAJOR_GROUP_COLUMN,),
    )
    plan = StagePlan([frs_occupation_stage(_fake_adult()), consumer])
    assert [stage.name for stage in plan.stages] == [
        UK_FRS_OCCUPATION_STAGE_NAME,
        "ai_exposure_imputation",
    ]


_REAL_ADULT_TAB = os.environ.get("POPULACE_UK_FRS_ADULT_TAB", "")
_REAL_FRS_H5 = os.environ.get("POPULACE_UK_FRS_H5", "")


@pytest.mark.skipif(
    not (_REAL_ADULT_TAB and _REAL_FRS_H5),
    reason=(
        "requires locally held UKDS EUL FRS microdata: set "
        "POPULACE_UK_FRS_ADULT_TAB (adult.tab path) and POPULACE_UK_FRS_H5 "
        "(the same vintage's PolicyEngine-UK single-year H5)"
    ),
)
def test_real_adult_tab_ids_match_policyengine_uk_h5():
    """Every real adult.tab row joins the same-vintage PE-UK person table.

    The synthetic tests pin the formula but construct both sides of the
    join, so they cannot catch upstream drift in the PolicyEngine-UK data
    build's ID assignment — the failure mode that shipped the original bug. This one
    can: it rebuilds IDs from the raw adult.tab and requires every one to
    appear in the real H5's person table (verified 1:1 on FRS 2023-24 and
    2024-25).
    """

    pytest.importorskip("tables")  # pandas HDF reader for the PE-UK H5
    adult = load_frs_adult_table(_REAL_ADULT_TAB)
    ids = frs_person_ids(adult)
    person_ids = pd.read_hdf(_REAL_FRS_H5, "person")["person_id"].astype("int64")
    matched = np.isin(ids, person_ids.to_numpy())
    assert matched.all(), (
        f"{int((~matched).sum())} of {len(ids)} adult.tab rows failed to "
        "match the H5 person table; the upstream PolicyEngine-UK data "
        "build's ID convention has drifted from SERNUM * 1000 + PERSON."
    )
