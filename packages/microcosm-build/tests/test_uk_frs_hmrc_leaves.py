from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.frs_hmrc_leaves import (
    FRS_HMRC_INCPBEN_COLUMN,
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    FRS_HMRC_PAY_COLUMN,
    FRS_HMRC_RETAINED_LEAF_COLUMNS,
    FRS_HMRC_RETAINED_LEAF_SOURCE_EVIDENCE,
    FRS_HMRC_RETAINED_LEAVES_STAGE_NAME,
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
    FRS_HMRC_UBISJA_COLUMN,
    FRS_WEEKS_IN_YEAR,
    UKFRSHMRCRetainedLeavesStageTransform,
    retain_uk_frs_hmrc_leaves,
)
from microcosm.build.uk_runtime.national_build import (
    UKNationalStage,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_support import (
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN,
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
)
from microcosm.frame import Frame


def _candidate() -> tuple[Frame, np.ndarray]:
    raw_households = {
        1: (1001, 1002),
        2: (2001,),
    }
    raw_person_ids = tuple(
        person_id for people in raw_households.values() for person_id in people
    )
    spi_person_offset = max(raw_person_ids) + 1
    spi_household_offset = max(raw_households) + 1

    pre_capital_stacks = [
        (household_id, False, people) for household_id, people in raw_households.items()
    ]
    pre_capital_stacks.append(
        (
            1 + spi_household_offset,
            True,
            tuple(person_id + spi_person_offset for person_id in raw_households[1]),
        )
    )
    capital_person_offset = (
        max(
            person_id
            for _household_id, _spi, people in pre_capital_stacks
            for person_id in people
        )
        + 1
    )
    capital_household_offset = (
        max(household_id for household_id, _spi, _people in pre_capital_stacks) + 1
    )
    clone_zero_stacks = [
        (household_id, spi, False, people)
        for household_id, spi, people in pre_capital_stacks
    ] + [
        (
            household_id + capital_household_offset,
            spi,
            True,
            tuple(person_id + capital_person_offset for person_id in people),
        )
        for household_id, spi, people in pre_capital_stacks
    ]
    clone_multiplier = 10 ** len(
        str(
            max(
                person_id
                for _household_id, _spi, _capital, people in clone_zero_stacks
                for person_id in people
            )
        )
    )

    household_rows: list[dict[str, object]] = []
    person_rows: list[dict[str, object]] = []
    for clone_index in range(2):
        clone_offset = clone_index * clone_multiplier
        for household_id, spi, capital_gains, people in clone_zero_stacks:
            descendant_household_id = household_id + clone_offset
            household_rows.append(
                {
                    "household_id": descendant_household_id,
                    "household_weight": 0.0 if spi else 1.0,
                    "clone_index": clone_index,
                    "household_is_spi_synthetic": spi,
                    "household_is_capital_gains_clone": capital_gains,
                }
            )
            for person_id in people:
                descendant_person_id = person_id + clone_offset
                source_person_id = (
                    person_id
                    - int(spi) * spi_person_offset
                    - int(capital_gains) * capital_person_offset
                )
                person_rows.append(
                    {
                        "person_id": descendant_person_id,
                        "person_household_id": descendant_household_id,
                        "person_benunit_id": descendant_person_id,
                        "expected_source_person_id": source_person_id,
                    }
                )
    # Frame requires group ids sorted ascending (it raises, never reorders),
    # so the group tables sort; the person table stays SHUFFLED, which keeps
    # this fixture's teeth: person row i never corresponds positionally to
    # household row i, so lineage resolution must stay id-keyed — the
    # 2024-25 FRS bug class this candidate exists to catch.
    household = pd.DataFrame(household_rows).sort_values(
        "household_id", ignore_index=True
    )
    person = pd.DataFrame(person_rows).sample(
        frac=1.0, random_state=7, ignore_index=True
    )
    source_person_ids = person.pop("expected_source_person_id").to_numpy(dtype=int)
    benunit = pd.DataFrame(
        {"benunit_id": person["person_benunit_id"].copy()}
    ).sort_values("benunit_id", ignore_index=True)
    return (
        uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        ),
        source_person_ids,
    )


def _write_raw_tables(
    directory: Path,
    *,
    include_incapacity: bool = True,
) -> tuple[Path, Path]:
    adult_path = directory / "adult.tab"
    benefits_path = directory / "benefits.tab"
    pd.DataFrame(
        {
            "SERNUM": [1, 1, 2],
            "PERSON": [1, 2, 1],
            "INEARNS": [100.0, -1.0, 50.0],
            "UNUSED": ["not", "extracted", "ever"],
        }
    ).to_csv(adult_path, sep="\t", index=False)
    benefit_rows = [
        {"SERNUM": 1, "PERSON": 1, "BENEFIT": 14, "BENAMT": 10.0, "VAR2": 0},
        {"SERNUM": 1, "PERSON": 1, "BENEFIT": 19, "BENAMT": 2.0, "VAR2": 0},
        {"SERNUM": 1, "PERSON": 2, "BENEFIT": 13, "BENAMT": 4.0, "VAR2": 0},
        {"SERNUM": 1, "PERSON": 2, "BENEFIT": 16, "BENAMT": 5.0, "VAR2": 1},
        {"SERNUM": 1, "PERSON": 2, "BENEFIT": 16, "BENAMT": 99.0, "VAR2": 2},
        {"SERNUM": 1, "PERSON": 2, "BENEFIT": 16, "BENAMT": 6.0, "VAR2": 3},
        {"SERNUM": 2, "PERSON": 1, "BENEFIT": 5, "BENAMT": 6.0, "VAR2": 0},
        {"SERNUM": 2, "PERSON": 1, "BENEFIT": 999, "BENAMT": -1.0, "VAR2": 0},
    ]
    if include_incapacity:
        benefit_rows.append(
            {
                "SERNUM": 2,
                "PERSON": 1,
                "BENEFIT": 17,
                "BENAMT": 3.0,
                "VAR2": 0,
            }
        )
    pd.DataFrame(benefit_rows).to_csv(benefits_path, sep="\t", index=False)
    return adult_path, benefits_path


def _expected_leaves(source_person_ids: np.ndarray) -> pd.DataFrame:
    weekly = {
        1001: {
            FRS_HMRC_PAY_COLUMN: 100.0,
            FRS_HMRC_UBISJA_COLUMN: 12.0,
        },
        1002: {
            FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN: 15.0,
        },
        2001: {
            FRS_HMRC_PAY_COLUMN: 50.0,
            FRS_HMRC_INCPBEN_COLUMN: 3.0,
            FRS_HMRC_SRP_REGULAR_CODE5_COLUMN: 6.0,
        },
    }
    expected = pd.DataFrame(
        0.0,
        index=np.arange(len(source_person_ids)),
        columns=FRS_HMRC_RETAINED_LEAF_COLUMNS,
    )
    for index, source_person_id in enumerate(source_person_ids):
        for column, value in weekly[int(source_person_id)].items():
            expected.loc[index, column] = value * FRS_WEEKS_IN_YEAR
    return expected


def test_retains_source_faithful_leaves_across_all_candidate_descendants(
    tmp_path: Path,
) -> None:
    dataset, source_person_ids = _candidate()
    adult_path, benefits_path = _write_raw_tables(tmp_path)

    result = retain_uk_frs_hmrc_leaves(
        dataset,
        adult_tab_path=adult_path,
        benefits_tab_path=benefits_path,
    )

    actual = result.frame.person.loc[:, list(FRS_HMRC_RETAINED_LEAF_COLUMNS)]
    assert np.array_equal(
        actual.to_numpy(), _expected_leaves(source_person_ids).to_numpy()
    )
    assert result.clone_id_multiplier == 10_000
    assert result.spi_person_id_offset == 2_002
    assert result.capital_gains_person_id_offset == 3_005
    assert result.raw_source_people == 3
    assert result.candidate_people == len(dataset.person)
    assert result.structural_zero_columns == ()
    assert result.source_signal_rows == {
        FRS_HMRC_PAY_COLUMN: 2,
        FRS_HMRC_UBISJA_COLUMN: 1,
        FRS_HMRC_INCPBEN_COLUMN: 1,
        FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN: 1,
        FRS_HMRC_SRP_REGULAR_CODE5_COLUMN: 1,
    }
    assert (
        result.adult_source.sha256
        == hashlib.sha256(adult_path.read_bytes()).hexdigest()
    )
    assert (
        result.benefits_source.sha256
        == hashlib.sha256(benefits_path.read_bytes()).hexdigest()
    )
    assert result.adult_source.extracted_columns == ("sernum", "person", "inearns")
    evidence = result.evidence()
    assert evidence["stage"] == FRS_HMRC_RETAINED_LEAVES_STAGE_NAME
    assert evidence["retained_leaves"][FRS_HMRC_PAY_COLUMN]["spi_concept"] == "PAY"
    assert (
        evidence["retained_leaves"][FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN]["scope"]
        == "identifiable_subset"
    )


def test_partial_leaves_never_populate_full_ossben_or_srp_columns(
    tmp_path: Path,
) -> None:
    dataset, _source_person_ids = _candidate()
    adult_path, benefits_path = _write_raw_tables(tmp_path)

    result = retain_uk_frs_hmrc_leaves(
        dataset,
        adult_tab_path=adult_path,
        benefits_tab_path=benefits_path,
    )

    forbidden = {
        SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
        SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN,
        SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN,
        SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
        SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
        SPI_HMRC_OTHER_INCOME_COLUMN,
        SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    }
    assert forbidden.isdisjoint(result.frame.person.columns)
    assert (
        FRS_HMRC_RETAINED_LEAF_SOURCE_EVIDENCE[
            FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN
        ]["scope"]
        == "identifiable_subset"
    )


def test_incpben_is_an_honest_structural_zero_when_code17_is_unobserved(
    tmp_path: Path,
) -> None:
    dataset, _source_person_ids = _candidate()
    adult_path, benefits_path = _write_raw_tables(tmp_path, include_incapacity=False)

    result = retain_uk_frs_hmrc_leaves(
        dataset,
        adult_tab_path=adult_path,
        benefits_tab_path=benefits_path,
    )

    assert (result.frame.person[FRS_HMRC_INCPBEN_COLUMN] == 0.0).all()
    assert FRS_HMRC_INCPBEN_COLUMN in result.structural_zero_columns
    assert result.evidence()["retained_leaves"][FRS_HMRC_INCPBEN_COLUMN][
        "structural_zero"
    ]


def test_future_code17_observation_flows_without_a_schema_change(
    tmp_path: Path,
) -> None:
    dataset, source_person_ids = _candidate()
    adult_path, benefits_path = _write_raw_tables(tmp_path, include_incapacity=True)

    result = retain_uk_frs_hmrc_leaves(
        dataset,
        adult_tab_path=adult_path,
        benefits_tab_path=benefits_path,
    )

    expected = np.where(source_person_ids == 2001, 3.0 * FRS_WEEKS_IN_YEAR, 0.0)
    assert np.array_equal(
        result.frame.person[FRS_HMRC_INCPBEN_COLUMN].to_numpy(), expected
    )
    assert FRS_HMRC_INCPBEN_COLUMN not in result.structural_zero_columns


def test_transform_is_national_stage_compatible_and_retains_evidence(
    tmp_path: Path,
) -> None:
    dataset, _source_person_ids = _candidate()
    _write_raw_tables(tmp_path)
    transform = UKFRSHMRCRetainedLeavesStageTransform.from_raw_frs_directory(tmp_path)
    stage = UKNationalStage(
        name=FRS_HMRC_RETAINED_LEAVES_STAGE_NAME,
        transform=transform,
    )

    staged = stage.run(dataset)

    assert set(FRS_HMRC_RETAINED_LEAF_COLUMNS).issubset(staged.person.columns)
    assert transform.last_result is not None
    assert transform.last_result.frame is staged


def test_raw_source_identity_must_exist_on_candidate_base(tmp_path: Path) -> None:
    dataset, _source_person_ids = _candidate()
    adult_path, benefits_path = _write_raw_tables(tmp_path)
    adult = pd.read_csv(adult_path, sep="\t")
    adult.loc[len(adult)] = {"SERNUM": 9, "PERSON": 1, "INEARNS": 1, "UNUSED": "x"}
    adult.to_csv(adult_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="absent from the certified candidate base"):
        retain_uk_frs_hmrc_leaves(
            dataset,
            adult_tab_path=adult_path,
            benefits_tab_path=benefits_path,
        )


def test_sampled_rung_receipts_the_dropped_raw_surface(tmp_path: Path) -> None:
    """A #627 rung build restricts the raw surface and receipts the drop.

    The completeness fence (every raw-survey person present in the base)
    cannot hold when the base deliberately carries a sampled subset of
    source families; declaring ``sampled_rung`` converts the raise into a
    receipted count while the surviving surface stays source-faithful.
    """

    dataset, _source_person_ids = _candidate()
    (tmp_path / "clean").mkdir()
    (tmp_path / "extra").mkdir()
    clean_adult_path, clean_benefits_path = _write_raw_tables(tmp_path / "clean")
    strict = retain_uk_frs_hmrc_leaves(
        dataset,
        adult_tab_path=clean_adult_path,
        benefits_tab_path=clean_benefits_path,
    )
    adult_path, benefits_path = _write_raw_tables(tmp_path / "extra")
    adult = pd.read_csv(adult_path, sep="\t")
    adult.loc[len(adult)] = {"SERNUM": 9, "PERSON": 1, "INEARNS": 1, "UNUSED": "x"}
    adult.to_csv(adult_path, sep="\t", index=False)

    result = retain_uk_frs_hmrc_leaves(
        dataset,
        adult_tab_path=adult_path,
        benefits_tab_path=benefits_path,
        sampled_rung=True,
    )

    assert result.source_people_outside_candidate == 1
    assert result.evidence()["lineage"]["source_people_outside_candidate"] == 1
    assert strict.source_people_outside_candidate == 0
    # The surviving surface attaches exactly what the strict run attaches.
    pd.testing.assert_frame_equal(
        result.frame.table("person"), strict.frame.table("person")
    )


def test_candidate_clone_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    dataset, _source_person_ids = _candidate()
    adult_path, benefits_path = _write_raw_tables(tmp_path)
    person = dataset.person.copy()
    household = dataset.table("household")
    clone_households = set(household.loc[household["clone_index"] == 1, "household_id"])
    tampered_row = person["person_household_id"].isin(clone_households).idxmax()
    person.loc[tampered_row, "person_id"] += 500
    tampered = uk_national_frame(
        person=person,
        benunit=dataset.table("benunit"),
        household=household,
        time_period="2023",
    )

    with pytest.raises(ValueError, match="person IDs do not reverse"):
        retain_uk_frs_hmrc_leaves(
            tampered,
            adult_tab_path=adult_path,
            benefits_tab_path=benefits_path,
        )


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("INEARNS", "missing required column"),
        ("PERSON", "missing required column"),
    ],
)
def test_missing_required_raw_column_fails_closed(
    tmp_path: Path,
    column: str,
    message: str,
) -> None:
    dataset, _source_person_ids = _candidate()
    adult_path, benefits_path = _write_raw_tables(tmp_path)
    adult = pd.read_csv(adult_path, sep="\t").drop(columns=[column])
    adult.to_csv(adult_path, sep="\t", index=False)

    with pytest.raises(ValueError, match=message):
        retain_uk_frs_hmrc_leaves(
            dataset,
            adult_tab_path=adult_path,
            benefits_tab_path=benefits_path,
        )


def test_negative_relevant_benefit_amount_fails_closed(tmp_path: Path) -> None:
    dataset, _source_person_ids = _candidate()
    adult_path, benefits_path = _write_raw_tables(tmp_path)
    benefits = pd.read_csv(benefits_path, sep="\t")
    benefits.loc[benefits["BENEFIT"] == 14, "BENAMT"] = -1.0
    benefits.to_csv(benefits_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="must be non-negative"):
        retain_uk_frs_hmrc_leaves(
            dataset,
            adult_tab_path=adult_path,
            benefits_tab_path=benefits_path,
        )


@pytest.mark.parametrize(
    ("table", "column", "message"),
    [
        ("adult", "INEARNS", "ADULT.INEARNS"),
        ("benefits", "BENAMT", "relevant BENEFITS.BENAMT"),
    ],
)
def test_nonfinite_source_amount_fails_closed(
    tmp_path: Path,
    table: str,
    column: str,
    message: str,
) -> None:
    dataset, _source_person_ids = _candidate()
    adult_path, benefits_path = _write_raw_tables(tmp_path)
    path = adult_path if table == "adult" else benefits_path
    frame = pd.read_csv(path, sep="\t")
    frame.loc[0, column] = np.inf
    frame.to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match=message):
        retain_uk_frs_hmrc_leaves(
            dataset,
            adult_tab_path=adult_path,
            benefits_tab_path=benefits_path,
        )


def test_missing_code16_var2_fails_closed(tmp_path: Path) -> None:
    dataset, _source_person_ids = _candidate()
    adult_path, benefits_path = _write_raw_tables(tmp_path)
    benefits = pd.read_csv(benefits_path, sep="\t")
    benefits.loc[benefits["BENEFIT"] == 16, "VAR2"] = np.nan
    benefits.to_csv(benefits_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="VAR2 for BENEFIT=16"):
        retain_uk_frs_hmrc_leaves(
            dataset,
            adult_tab_path=adult_path,
            benefits_tab_path=benefits_path,
        )


def test_checkpoint_metadata_round_trips_the_descent_evidence(tmp_path) -> None:
    """A fresh process resumes the retained stage from its record alone.

    The rehydrated result exposes exactly the surface the SPI stage's
    descent fence reads — evidence and both content identities — and a
    checkpoint whose content no longer matches its recorded output identity
    is refused as drifted.
    """

    from microcosm.build.uk_runtime.content_identity import (
        uk_frame_content_identity,
    )

    dataset, _source_person_ids = _candidate()
    _write_raw_tables(tmp_path)
    transform = UKFRSHMRCRetainedLeavesStageTransform.from_raw_frs_directory(tmp_path)
    staged = transform(dataset)
    metadata = transform.checkpoint_metadata()

    resumed = UKFRSHMRCRetainedLeavesStageTransform.from_raw_frs_directory(tmp_path)
    resumed.resume_from_checkpoint(metadata, staged)
    assert resumed.last_result is not None
    assert resumed.last_result.frame is staged
    assert resumed.last_result.evidence() == transform.last_result.evidence()
    assert resumed.last_result.input_content_identity == uk_frame_content_identity(
        dataset
    )
    assert resumed.last_result.output_content_identity == uk_frame_content_identity(
        staged
    )

    drifted = UKFRSHMRCRetainedLeavesStageTransform.from_raw_frs_directory(tmp_path)
    with pytest.raises(RuntimeError, match="drifted record"):
        drifted.resume_from_checkpoint(metadata, dataset)

    empty = UKFRSHMRCRetainedLeavesStageTransform.from_raw_frs_directory(tmp_path)
    with pytest.raises(RuntimeError, match="cannot prove descent"):
        empty.resume_from_checkpoint({}, staged)

    unrun = UKFRSHMRCRetainedLeavesStageTransform.from_raw_frs_directory(tmp_path)
    with pytest.raises(RuntimeError, match="completed retained-leaves run"):
        unrun.checkpoint_metadata()
