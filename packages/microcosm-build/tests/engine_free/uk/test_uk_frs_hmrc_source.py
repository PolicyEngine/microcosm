"""Raw FRS extraction shared by the canonical spine income stages."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.frs_hmrc_source import (
    FRS_HMRC_INCPBEN_COLUMN,
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    FRS_HMRC_PAY_COLUMN,
    FRS_HMRC_RETAINED_LEAF_COLUMNS,
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
    FRS_HMRC_UBISJA_COLUMN,
    FRS_WEEKS_IN_YEAR,
    _materialize_source_leaves,
    _read_raw_frs_table,
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


def _read_sources(directory):
    adult, identity = _read_raw_frs_table(
        directory / "adult.tab",
        expected_filename="adult.tab",
        source_vintage="2024-25",
        required_columns=("sernum", "person", "inearns"),
    )
    benefits, _ = _read_raw_frs_table(
        directory / "benefits.tab",
        expected_filename="benefits.tab",
        required_columns=("sernum", "person", "benefit", "benamt", "var2"),
    )
    return adult, benefits, identity


def test_raw_source_extraction_preserves_income_concepts(tmp_path):
    adult_path, _ = _write_raw_tables(tmp_path)
    adult, benefits, identity = _read_sources(tmp_path)
    actual = _materialize_source_leaves(adult, benefits)
    expected = _expected_leaves(actual.index.to_numpy())
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)
    assert list(adult) == ["sernum", "person", "inearns"]
    assert identity.sha256 == hashlib.sha256(adult_path.read_bytes()).hexdigest()
    assert identity.rows == 3


def test_absent_incapacity_stays_zero(tmp_path):
    _write_raw_tables(tmp_path, include_incapacity=False)
    adult, benefits, _ = _read_sources(tmp_path)
    assert (
        _materialize_source_leaves(adult, benefits)[FRS_HMRC_INCPBEN_COLUMN] == 0
    ).all()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "unique"),
        ("negative_benefit", "non-negative"),
        ("nan_earnings", "finite"),
    ],
)
def test_invalid_raw_source_values_are_refused(tmp_path, mutation, match):
    _write_raw_tables(tmp_path)
    adult, benefits, _ = _read_sources(tmp_path)
    if mutation == "duplicate":
        adult = pd.concat([adult, adult.iloc[:1]], ignore_index=True)
    elif mutation == "negative_benefit":
        benefits.loc[0, "benamt"] = -1
    else:
        adult.loc[0, "inearns"] = np.nan
    with pytest.raises(ValueError, match=match):
        _materialize_source_leaves(adult, benefits)


def test_candidate_restoration_module_is_removed():
    assert (
        importlib.util.find_spec("microcosm.build.uk_runtime.frs_hmrc_leaves") is None
    )
