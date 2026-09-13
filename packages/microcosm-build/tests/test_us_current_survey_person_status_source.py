"""Actual source qualification over small invented and privately pinned members."""

import copy
import csv
import hashlib
import io
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_current_survey_person_status import acs, asec

from microcosm.build.us_runtime import current_survey_person_status as status
from microcosm.build.us_runtime import current_survey_person_status_source as owner


def _source_arguments(tmp_path, monkeypatch):
    import test_us_survey_population_preparation as preparation_fixture
    from test_us_current_asec_demographics import _demographic_arguments

    from microcosm.build.us_runtime import asec_person_income_source as restoration
    from microcosm.build.us_runtime import current_asec_demographics as demographic

    original_person = preparation_fixture._person
    original_asec = preparation_fixture.asec_fixture

    def person(*args, **kwargs):
        row = original_person(*args, **kwargs)
        observations = acs(int(row["AGEP"]))
        row.update(
            {
                c: v
                for c, v in observations.items()
                if c not in ("SERIALNO", "SPORDER", "AGEP")
            }
        )
        return row

    def asec_fixture(*args, **kwargs):
        # Correct the invented parent before its real checkpoint/owner issuance.
        # The shared fixture has ages 55/14, so source enrollment is NIU, not 2.
        kwargs["person_changes"] = {
            "A_ENRLW": np.zeros(6, dtype=np.int64),
            "A_FTPT": np.zeros(6, dtype=np.int64),
        }
        return original_asec(*args, **kwargs)

    monkeypatch.setattr(preparation_fixture, "_person", person)
    monkeypatch.setattr(preparation_fixture, "asec_fixture", asec_fixture)
    arguments = _demographic_arguments(tmp_path, monkeypatch, zero=False)
    source_dir = arguments["source_dir"] / "asec"
    members, pins = {}, []
    for year, member, archive, *_ in owner.original.asec._MEMBER_PINS:
        path = source_dir / member
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        # ASEC fixture already has the precise PRPERTYP roster and original age.
        for position, row in table.iterrows():
            observations = asec(int(row.A_AGE), row.PRPERTYP)
            for c, value in observations.items():
                if c not in ("PERIDNUM", "PH_SEQ", "A_LINENO", "A_AGE", "PRPERTYP"):
                    table.loc[position, c] = value
        table.iloc[::-1].to_csv(path, index=False)
        data = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(data).hexdigest(),
                len(table),
                len(data),
            )
        )
        members[year] = path
    for module in (
        owner.original.asec,
        restoration,
        demographic.demographic,
        owner.student,
    ):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = tmp_path / "status-restored-money"
    restoration.restore_asec_person_income_source(
        source_dir / "parent.h5",
        source_dir / "household-attachment.h5",
        member_paths=members,
        output_dir=output,
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME,
        source_dir / "person-income-attachment.h5",
    )
    return arguments


def test_actual_qualifier_retains_complete_sources_and_closes_mutation_seals(
    tmp_path, monkeypatch
):
    arguments = _source_arguments(tmp_path, monkeypatch)
    prepared = owner.source.prepare_authenticated_survey_population(**arguments)
    original = prepared.checked_view().frame
    before = owner.source._frame_identity(original)
    result = owner.qualify_current_survey_person_status(prepared)
    result.validate()
    receipt = json.loads(result.receipt)
    assert result.source_frame is original
    assert result.origins.index.tolist() == original.person.person_id.tolist()
    assert result.raw.index.equals(result.observations.index)
    assert receipt["student_controls_income_cohorts"] == [2022, 2023, 2024]
    assert receipt["student_controls_selected_income_cohort"] == 2024
    assert (
        receipt["asec_observation_year"] == 2025
        and receipt["acs_observation_year"] == 2024
    )
    assert not receipt["canonical_eligibility_assigned"]
    assert not receipt["annual_five_month_student_status_validated"]
    assert not receipt["source_admission_issued"] and not receipt["release_eligible"]
    assert not {"is_blind", "is_disabled", "is_full_time_college_student"} & set(
        result.observations
    )
    assert owner.source._frame_identity(original) == before
    asec_mask = result.origins.source.eq("asec")
    assert result.raw.loc[asec_mask, "PERIDNUM"].str.len().eq(22).all()
    assert result.raw.loc[~asec_mask, "PERIDNUM"].isna().all()
    assert (
        result.observations.loc[
            ~asec_mask, "survey_full_time_college_student_last_week"
        ]
        .isna()
        .all()
    )
    assert (
        result.observations.loc[asec_mask, "survey_full_time_college_student_last_week"]
        .isna()
        .all()
    )
    with pytest.raises(ValueError, match="PROJECTION_OBJECT_CHANGED"):
        replace(result).validate()
    original_callback = result._revalidate
    object.__setattr__(result, "_revalidate", lambda _: None)
    with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
        result.validate()
    object.__setattr__(result, "_revalidate", original_callback)
    result.validate()
    with pytest.raises(ValueError):
        owner.qualify_current_survey_person_status(copy.copy(prepared))
    # No source reconstruction or normalization can hide a mutated projection.
    pid = result.raw.index[0]
    previous = result.raw.at[pid, "DEYE"]
    result.raw.at[pid, "DEYE"] = "1"
    with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
        result.validate()
    result.raw.at[pid, "DEYE"] = previous
    result.validate()
    old_receipt = result.receipt
    object.__setattr__(result, "receipt", b"{}")
    with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
        result.validate()
    object.__setattr__(result, "receipt", old_receipt)
    # Mutating the retained issuer object must fail, even when detached rows agree.
    closure = dict(
        zip(
            result._revalidate.__code__.co_freevars,
            (c.cell_contents for c in result._revalidate.__closure__),
            strict=True,
        )
    )
    controls = closure["controls"]
    body = controls._body
    object.__setattr__(controls, "_body", body + b"x")
    with pytest.raises(ValueError):
        result.validate()
    object.__setattr__(controls, "_body", body)
    result.validate()
    assert owner.source._frame_identity(original) == before
    original_read = Path.read_bytes
    triggered = []

    def mutate_after_source_io(path):
        data = original_read(path)
        if not triggered:
            triggered.append(str(path))
            result.raw.at[pid, "DEYE"] = "1"
        return data

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", mutate_after_source_io)
        with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
            result.validate()
    assert len(triggered) == 1
    result.raw.at[pid, "DEYE"] = previous
    result.validate()


def _stream(rows, columns):
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(columns)
    writer.writerows([[r[c] for c in columns] for r in rows])
    return io.BytesIO(text.getvalue().encode())


def test_scanner_joins_exact_native_keys_without_float_coercion():
    first = asec(PERIDNUM="9999999999999999999999")
    second = asec(PERIDNUM="9999999999999999999998", A_LINENO="2")
    keys = {owner.original._key(r, "asec") for r in (first, second)}
    selected = {}
    assert (
        owner._scan(
            _stream([second, first], status.ASEC_COLUMNS),
            survey="asec",
            wanted=keys,
            selected=selected,
            maximum=2,
        )
        == 2
    )
    assert selected[owner.original._key(first, "asec")]["PERIDNUM"] == first["PERIDNUM"]
    with pytest.raises(ValueError, match="DUPLICATE_SELECTED_SOURCE_KEY"):
        owner._scan(
            _stream([first, first], status.ASEC_COLUMNS),
            survey="asec",
            wanted=keys,
            selected={},
            maximum=2,
        )


@pytest.mark.parametrize("defect", ["missing", "duplicate", "key", "age"])
def test_source_join_refuses_missing_duplicate_mismatched_keys_and_original_age(defect):
    row = acs(age=85)
    key = owner.original._key(row, "acs")
    index = pd.Index([2**53 + 1], name="person_id", dtype="int64")
    origins = pd.DataFrame({"source": ["acs"], "native_person_id": [3]}, index=index)
    frame = SimpleNamespace(
        person=pd.DataFrame({"person_id": index.to_numpy(), "A_AGE": [85], "age": [90]})
    )
    keys, selected = {("acs", key): int(index[0])}, {"acs": {key: row}, "asec": {}}
    if defect == "missing":
        selected["acs"] = {}
    elif defect == "duplicate":
        selected["acs"][("2024HU0000001", "2", 2)] = row.copy()
    elif defect == "key":
        row["SERIALNO"] = "2024HU0000002"
    else:
        row["AGEP"] = "90"  # Mapped age never replaces the original-source age.
    with pytest.raises(ValueError, match="SURVEY_PERSON_STATUS_"):
        owner._combine(frame, origins, keys, selected)


def test_source_join_preserves_original_age_and_large_stacked_id():
    row = acs(age=85)
    key = owner.original._key(row, "acs")
    index = pd.Index([2**53 + 1], name="person_id", dtype="int64")
    origins = pd.DataFrame({"source": ["acs"], "native_person_id": [3]}, index=index)
    frame = SimpleNamespace(
        person=pd.DataFrame({"person_id": index.to_numpy(), "A_AGE": [85], "age": [90]})
    )
    raw, values = owner._combine(
        frame, origins, {("acs", key): int(index[0])}, {"acs": {key: row}, "asec": {}}
    )
    assert values.index[0] == 2**53 + 1
    assert raw.AGEP.iloc[0] == "85"
    assert values.survey_status_original_age.iloc[0] == 85
