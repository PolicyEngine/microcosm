from __future__ import annotations

import hashlib
import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import microcosm.build.us_runtime.acs_release_predictors as module
from microcosm.build.us_runtime.acs_release_predictors import (
    ACS_OCCP_TO_POCCU2,
    ACS_RELEASE_PREDICTOR_CROSSWALK_SHA256,
    acs_release_predictor_crosswalk_payload,
    join_acs_release_predictors,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_CPS_PREDICTORS = (
    "PEDISDRS",
    "PEDISEAR",
    "PEDISEYE",
    "PEDISOUT",
    "PEDISPHY",
    "PEDISREM",
    "SSI_VAL",
    "PRDTRACE",
    "PRDTHSP",
    "PEIOOCC",
    "POCCU2",
    "SPM_TENMORTSTATUS",
)


def _source_frame(*, acs: bool) -> Frame:
    if acs:
        household_ids = np.asarray([1, 2], dtype=np.int64)
        person_ids = np.asarray([0, 1, 2], dtype=np.int64)
        memberships = np.asarray([1, 1, 2], dtype=np.int64)
        ages = np.asarray([4.0, 40.0, 30.0])
    else:
        household_ids = np.asarray([1], dtype=np.int64)
        person_ids = np.asarray([0], dtype=np.int64)
        memberships = np.asarray([1], dtype=np.int64)
        ages = np.asarray([35.0])

    offsets = {
        "tax_unit": 100,
        "spm_unit": 200,
        "family": 300,
        "marital_unit": 400,
    }
    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "person_household_id": memberships,
            "age": ages,
        }
    )
    for entity, offset in offsets.items():
        person[f"person_{entity}_id"] = memberships + offset

    household = pd.DataFrame({"household_id": household_ids})
    tables: dict[str, pd.DataFrame] = {"person": person, "household": household}
    for entity, offset in offsets.items():
        tables[entity] = pd.DataFrame({f"{entity}_id": household_ids + offset})

    if acs:
        person["source_row_id"] = person_ids
        person["source_year"] = 2024
        person["source_household_id"] = memberships
        person["source_person_id"] = ["1", "2", "1"]
        person["SPORDER"] = [1, 2, 1]
        person["ssi_reported"] = [np.nan, 900.0, 0.0]
        household["SERIALNO"] = ["2024HU0000001", "2024GQ0000002"]
        household["TEN"] = [1.0, np.nan]
    else:
        for column in _CPS_PREDICTORS:
            person[column] = 2.0
        person["SSI_VAL"] = 0.0
        person["PRDTRACE"] = 1.0
        person["PRDTHSP"] = 0.0
        person["PEIOOCC"] = 1005.0
        person["POCCU2"] = 8.0
        person["SPM_TENMORTSTATUS"] = 1.0

    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(household_ids), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["acs_2024_1yr" if acs else "asec_2024"] * len(person)),
    )


def _stacked_frame() -> Frame:
    assembled = assemble_spines(
        {"asec": _source_frame(acs=False), "acs": _source_frame(acs=True)},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    return clone_us_frame_for_puf_support(assembled)


def _raw_person() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SERIALNO": ["2024HU0000001", "2024HU0000001", "2024GQ0000002"],
            "SPORDER": [1, 2, 1],
            "AGEP": [4, 40, 30],
            "DEAR": [2, 1, 2],
            "DEYE": [2, 2, 2],
            "DREM": [np.nan, 2, 2],
            "DPHY": [np.nan, 2, 2],
            "DDRS": [np.nan, 2, 2],
            "DOUT": [np.nan, 2, 2],
            "RAC1P": [1, 6, 2],
            "HISP": [1, 1, 2],
            "OCCP": [np.nan, 1005, 9800],
            "ESR": [np.nan, 1, 4],
        }
    )


def _raw_household() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SERIALNO": ["2024HU0000001", "2024GQ0000002"],
            "NP": [2, 1],
            "TYPEHUGQ": [1, 2],
            "TEN": [1.0, np.nan],
        }
    )


def _write_zip(path: Path, members: dict[str, pd.DataFrame]) -> str:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for member, table in members.items():
            buffer = io.StringIO()
            table.to_csv(buffer, index=False)
            archive.writestr(member, buffer.getvalue())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archives(
    tmp_path: Path,
    *,
    person: pd.DataFrame | None = None,
    household: pd.DataFrame | None = None,
) -> tuple[Path, str, Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw_person = _raw_person() if person is None else person
    raw_household = _raw_household() if household is None else household
    person_path = tmp_path / "csv_pus.zip"
    household_path = tmp_path / "csv_hus.zip"
    person_sha = _write_zip(
        person_path,
        {
            "psam_pusa.csv": raw_person.iloc[:2],
            "psam_pusb.csv": raw_person.iloc[2:],
        },
    )
    household_sha = _write_zip(
        household_path,
        {
            "psam_husa.csv": raw_household.iloc[:1],
            "psam_husb.csv": raw_household.iloc[1:],
        },
    )
    return person_path, person_sha, household_path, household_sha


def _join(
    frame: Frame,
    archives: tuple[Path, str, Path, str],
    monkeypatch: pytest.MonkeyPatch,
):
    person_path, person_sha, household_path, household_sha = archives
    monkeypatch.setattr(module, "ACS_2024_PERSON_ZIP_SHA256", person_sha)
    monkeypatch.setattr(module, "ACS_2024_HOUSEHOLD_ZIP_SHA256", household_sha)
    return join_acs_release_predictors(
        frame,
        person_zip=person_path,
        person_sha256=person_sha,
        household_zip=household_path,
        household_sha256=household_sha,
        chunksize=1,
    )


def test_crosswalk_digest_and_every_consumed_occupation_bin_are_pinned() -> None:
    assert module._computed_crosswalk_sha256() == (
        ACS_RELEASE_PREDICTOR_CROSSWALK_SHA256
    )
    assert len(ACS_OCCP_TO_POCCU2) == 530
    assert set(ACS_OCCP_TO_POCCU2.values()) == set(range(1, 54))
    assert ACS_OCCP_TO_POCCU2[3250] == 26
    assert ACS_OCCP_TO_POCCU2[3255] == 25
    assert ACS_OCCP_TO_POCCU2[1005] == 8
    assert ACS_OCCP_TO_POCCU2[6005] == 41
    assert ACS_OCCP_TO_POCCU2[9800] == 52
    assert ACS_OCCP_TO_POCCU2[9920] == 53

    payload = acs_release_predictor_crosswalk_payload()
    assert payload["disability"]["DREM"]["minimum_question_age"] == 5
    assert payload["disability"]["DOUT"]["minimum_question_age"] == 15
    assert payload["disability"]["DOUT"]["codes"]["below_universe_blank"] == -1
    assert payload["race"]["RAC1P_to_consumed_PRDTRACE"] == {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 3,
        "5": 3,
        "6": 4,
        "7": 3,
        "8": 3,
        "9": 3,
    }


def test_release_join_is_exact_total_clone_stable_and_receipted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _stacked_frame()
    before = frame.table("person").copy(deep=True)

    result = _join(frame, _archives(tmp_path), monkeypatch)
    person = result.frame.table("person")
    asec = person["person_support_channel"].eq("asec")
    acs = person["person_support_channel"].eq("acs")

    assert_frame_equal(
        person.loc[asec, list(_CPS_PREDICTORS)],
        before.loc[asec, list(_CPS_PREDICTORS)],
    )
    assert person.loc[acs, list(module._OUTPUT_COLUMNS)].notna().all().all()
    assert person.loc[acs, "SSI_VAL"].isna().all()
    assert person.loc[acs, "ssi_reported"].isna().sum() == 2

    by_source = person.loc[
        acs,
        [
            "person_source_id",
            "PEDISDRS",
            "PRDTRACE",
            "PRDTHSP",
            "PEIOOCC",
            "POCCU2",
            "SPM_TENMORTSTATUS",
        ],
    ].drop_duplicates("person_source_id")
    assert sorted(by_source["PEDISDRS"].tolist()) == [-1.0, 2.0, 2.0]
    assert sorted(by_source["PRDTRACE"].tolist()) == [1.0, 2.0, 4.0]
    assert sorted(by_source["PRDTHSP"].tolist()) == [0.0, 0.0, 1.0]
    assert sorted(by_source["PEIOOCC"].tolist()) == [0.0, 1005.0, 9800.0]
    assert sorted(by_source["POCCU2"].tolist()) == [0.0, 8.0, 52.0]
    assert sorted(by_source["SPM_TENMORTSTATUS"].tolist()) == [1.0, 1.0, 3.0]

    for _, clones in person.loc[acs].groupby("person_source_id"):
        for column in module._OUTPUT_COLUMNS:
            assert clones[column].nunique(dropna=False) == 1

    receipt = result.receipt
    assert receipt["crosswalk"]["sha256"] == ACS_RELEASE_PREDICTOR_CROSSWALK_SHA256
    assert receipt["join"] == {
        "semantic_key": ["household.SERIALNO", "person.SPORDER"],
        "clone_fanout_key": "person_source_id",
        "acs_source_people": 3,
        "acs_support_rows": 6,
        "acs_support_rows_by_clone_index": {"0": 3, "1": 3},
        "selected_raw_person_rows": 3,
        "selected_raw_household_rows": 2,
        "unmatched_pool_source_people": 0,
        "source_identity_collisions": 0,
        "semantic_key_sha256": receipt["join"]["semantic_key_sha256"],
    }
    race_counts = receipt["models"]["scf_wealth"]["predictors"]["PRDTRACE"]
    assert race_counts == {"asec_native": 2, "acs_joined": 6, "still_null": 0}
    reporter = receipt["models"]["ssi_disability_criteria"]["predictors"][
        "reported_ssi_anchor"
    ]
    assert reporter["asec_native"] == 2
    assert reporter["acs_joined"] == 4
    assert reporter["still_null"] == 2

    repeated = _join(result.frame, _archives(tmp_path / "again"), monkeypatch)
    assert_frame_equal(repeated.frame.table("person"), person)


def test_release_join_refuses_a_missing_raw_person(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_person().iloc[:-1].copy()
    with pytest.raises(ValueError, match="not total over pool source people"):
        _join(
            _stacked_frame(),
            _archives(tmp_path, person=raw),
            monkeypatch,
        )


def test_release_join_refuses_raw_person_key_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = pd.concat([_raw_person(), _raw_person().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="raw person key collision"):
        _join(
            _stacked_frame(),
            _archives(tmp_path, person=raw),
            monkeypatch,
        )


def test_release_join_refuses_source_identity_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _stacked_frame()
    person = frame.table("person")
    acs = person["person_support_channel"].eq("acs")
    source_id = person.loc[acs, "person_source_id"].iloc[0]
    clone = (
        acs
        & person["person_source_id"].eq(source_id)
        & person["person_support_clone_index"].eq(1)
    )
    person.loc[clone, "SPORDER"] = 9

    with pytest.raises(ValueError, match="source_person_id does not equal"):
        _join(frame, _archives(tmp_path), monkeypatch)


def test_release_join_verifies_pin_before_opening_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    person_path, person_sha, household_path, household_sha = _archives(tmp_path)
    monkeypatch.setattr(module, "ACS_2024_PERSON_ZIP_SHA256", "0" * 64)
    monkeypatch.setattr(module, "ACS_2024_HOUSEHOLD_ZIP_SHA256", household_sha)
    with pytest.raises(ValueError, match="pin must be the reviewed"):
        join_acs_release_predictors(
            _stacked_frame(),
            person_zip=person_path,
            person_sha256=person_sha,
            household_zip=household_path,
            household_sha256=household_sha,
        )


def test_no_acs_frame_is_an_identity_without_archive_options() -> None:
    frame = _source_frame(acs=False)
    result = join_acs_release_predictors(
        frame,
        person_zip=None,
        person_sha256=None,
        household_zip=None,
        household_sha256=None,
    )
    assert result.frame is frame
    assert result.receipt == {
        "enabled": False,
        "reason": "no physical ACS source rows",
    }
