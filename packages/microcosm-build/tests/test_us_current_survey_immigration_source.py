"""Genuine source issuers on invented bytes, never a rules engine or real data."""

import hashlib
import io
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from microcosm.build.us_runtime import current_survey_immigration_source as owner


def source_arguments(tmp_path, monkeypatch):
    import test_us_child_property_income_source_owner as fixture
    import test_us_survey_population_preparation as preparation_fixture

    from microcosm.build.us_runtime import asec_person_income_source as restoration

    original_person = preparation_fixture._person

    def person(*args, **kwargs):
        row = original_person(*args, **kwargs)
        row.update(CIT="1", POBP="001", YOEP="")
        return row

    monkeypatch.setattr(preparation_fixture, "_person", person)
    full, partial, donor = fixture.source_arguments(tmp_path, monkeypatch)
    folder = full["source_dir"] / "asec"
    pins, paths = [], {}
    for year, member, archive, *_ in owner.original.asec._MEMBER_PINS:
        path = folder / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        for name in owner.ASEC_VALUE_COLUMNS:
            if name not in raw:
                raw[name] = "0"
        raw["PRCITSHP"] = "1"
        raw["A_LFSR"] = " 7 "
        raw.loc[raw.A_AGE.map(int).lt(16), "A_LFSR"] = "0"
        raw["SPM_CAPHOUSESUB"] = "0.00"
        raw.to_csv(path, index=False)
        payload = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(payload).hexdigest(),
                len(raw),
                len(payload),
            )
        )
        paths[year] = path
        shutil.copyfile(path, partial["source_dir"] / "asec" / member)
    for module in (owner.original.asec, restoration):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    restored = tmp_path / "immigration-restored-money"
    restoration.restore_asec_person_income_source(
        folder / "parent.h5",
        folder / "household-attachment.h5",
        member_paths=paths,
        output_dir=restored,
    )
    for arguments in (full, partial):
        shutil.copyfile(
            restored / restoration.CHECKPOINT_FILENAME,
            arguments["source_dir"] / "asec" / "person-income-attachment.h5",
        )
    return full, partial, donor


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        full_args, partial_args, donor = source_arguments(
            tmp_path_factory.mktemp("immigration-source"), patch
        )
        full = owner.source.prepare_authenticated_survey_population(**full_args)
        partial = owner.source.prepare_authenticated_survey_population(**partial_args)
        full_seal = owner.source._frame_identity(full._checked()[2].frame)
        partial_seal = owner.source._frame_identity(partial._checked()[2].frame)
        yield SimpleNamespace(
            full=full,
            partial=partial,
            donor=donor,
            values=owner.qualify_current_survey_immigration(full),
            smaller=owner.qualify_current_survey_immigration(partial),
        )
        assert owner.source._frame_identity(full._checked()[2].frame) == full_seal
        assert owner.source._frame_identity(partial._checked()[2].frame) == partial_seal


def test_complete_current_asec_evidence_survives_support_selection(actual):
    left, right = actual.values, actual.smaller
    left.validate()
    right.validate()
    pd.testing.assert_frame_equal(left.asec_full_raw, right.asec_full_raw)
    assert len(right.asec_selected_raw) < len(right.asec_full_raw)
    assert str(actual.donor - 100).zfill(22) in right.asec_full_raw.PERIDNUM.tolist()
    # The deliberately unselected donor is age 15, leaving one adult.
    assert right.asec_full_raw.A_LFSR.tolist().count(" 7 ") == 1
    assert not {"WSAL_VAL", "SEMP_VAL", "person_weight"} & set(left.asec_full_raw)
    evidence = json.loads(left.receipt)
    assert evidence["asec_income_year"] == 2024
    assert evidence["asec_observation_year"] == 2025
    assert evidence["acs_observation_year"] == 2024
    assert evidence["status_assignment_performed"] is False
    assert evidence["national_stock_alignment_qualified"] is False
    assert evidence["rules_head"] == "7ee36ac1bea125218912996b1030b77a208db963"


def test_owning_literal_blank_and_nonowning_missing_remain_distinct(actual):
    value = actual.values
    assert value.person_evidence.index.equals(value.origins.index)
    acs = value.origins.source.eq("acs")
    asec = ~acs
    assert value.person_evidence.loc[acs, owner.PREFIX + "YOEP"].eq("").all()
    assert value.person_evidence.loc[asec, owner.PREFIX + "YOEP"].isna().all()
    assert value.person_evidence.loc[acs, owner.PREFIX + "A_LFSR"].isna().all()
    assert "immigration_status_str" not in value.person_evidence
    pd.testing.assert_frame_equal(
        actual.smaller.person_evidence,
        value.person_evidence.loc[actual.smaller.origins.index],
    )


def receiving_people(value):
    import test_us_current_survey_hours_source as hours_fixture

    return hours_fixture.receiving_people(value)


def test_clone_evidence_is_exact_and_does_not_assign_status(actual):
    people = receiving_people(actual.values)
    before = people.copy(deep=True)
    columns = owner.borrow_cloned_immigration_evidence(actual.values, people)
    for (_, name), values in columns.items():
        source_ids = people[
            owner.attachment.provenance.support_source_id_column("person")
        ]
        expected = (
            actual.values.person_evidence[name]
            .reindex(source_ids)
            .reset_index(drop=True)
        )
        pd.testing.assert_series_equal(
            values.reset_index(drop=True), expected, check_names=False
        )
    pd.testing.assert_frame_equal(people, before)
    assert all(name.startswith(owner.PREFIX) for _, name in columns)


@pytest.mark.parametrize(
    "defect", ["source", "native", "clone", "channel", "collision"]
)
def test_clone_evidence_refuses_bad_ancestry_and_overwrite(actual, defect):
    people = receiving_people(actual.values)
    p = owner.attachment.provenance
    if defect == "collision":
        people[owner.PREFIX + "A_LFSR"] = "7"
    elif defect == "channel":
        people.loc[0, p.support_channel_column("person")] = "foreign"
    else:
        column = {
            "source": p.support_source_id_column("person"),
            "native": p.spine_source_id_column("person"),
            "clone": p.support_clone_index_column("person"),
        }[defect]
        people.loc[0, column] = 99999
    with pytest.raises(ValueError):
        owner.borrow_cloned_immigration_evidence(actual.values, people)


@pytest.mark.parametrize(
    "field,column",
    [
        ("origins", "source"),
        ("asec_full_raw", "A_LFSR"),
        ("asec_selected_raw", "SPM_CAPHOUSESUB"),
        ("acs_raw", "YOEP"),
        ("person_evidence", "immigration_source_A_LFSR"),
    ],
)
def test_changed_projection_refuses(actual, field, column):
    table = getattr(actual.values, field)
    pid, previous = table.index[0], table.iloc[0][column]
    try:
        table.at[pid, column] = "changed"
        with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
            actual.values.validate()
    finally:
        table.at[pid, column] = previous
    actual.values.validate()


@pytest.mark.parametrize(
    "field",
    ["origins", "asec_full_raw", "asec_selected_raw", "acs_raw", "person_evidence"],
)
def test_equal_detached_table_is_not_retained_identity(actual, field):
    previous = getattr(actual.values, field)
    try:
        object.__setattr__(actual.values, field, previous.copy(deep=True))
        with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
            actual.values.validate()
    finally:
        object.__setattr__(actual.values, field, previous)


def test_copy_or_forged_callback_cannot_issue_source_owner(actual):
    with pytest.raises(ValueError, match="PROJECTION_OBJECT_CHANGED"):
        replace(actual.values).validate()
    with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
        replace(actual.values, _revalidate=lambda _: None).validate()


def test_unissued_preparation_refuses(actual):
    fake = object.__new__(type(actual.full))
    object.__setattr__(fake, "payload", actual.full.payload)
    with pytest.raises(ValueError):
        owner.qualify_current_survey_immigration(fake)


def test_readset_matches_the_reviewed_successor_interface_without_income_proxies():
    from microcosm.build.us_runtime import immigration

    assert set(owner.ASEC_VALUE_COLUMNS) == (
        set(immigration.US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS)
        - {"WSAL_VAL", "SEMP_VAL"}
    ) | {"A_LFSR"}


@pytest.mark.parametrize("which", ["receiving", "borrowed"])
def test_final_owner_io_cannot_change_transport_output(actual, monkeypatch, which):
    people = receiving_people(actual.values)
    validation = owner.QualifiedSurveyImmigrationSources.validate
    attach = owner.attachment.attach_columns
    observed = SimpleNamespace(calls=0, columns={})

    def capture(*args):
        columns = attach(*args)
        observed.columns.update(columns)
        return columns

    def validate(value):
        validation(value)
        observed.calls += 1
        if observed.calls == 2:
            if which == "receiving":
                people.loc[0, "person_id"] += 1
            else:
                observed.columns["person", owner.PREFIX + "A_LFSR"].iloc[0] = "changed"

    with monkeypatch.context() as patch:
        patch.setattr(owner.attachment, "attach_columns", capture)
        patch.setattr(owner.QualifiedSurveyImmigrationSources, "validate", validate)
        patch.setattr(owner, "_LIVE", owner._live())
        with pytest.raises(
            ValueError, match="RECEIVING_CHANGED|BORROWED_COLUMNS_CHANGED"
        ):
            owner.borrow_cloned_immigration_evidence(actual.values, people)
    actual.values.validate()


def test_final_io_cannot_mutate_a_projection(actual, monkeypatch):
    value = actual.values
    table, pid = value.asec_full_raw, value.asec_full_raw.index[0]
    previous, read_bytes, triggered = table.at[pid, "A_LFSR"], Path.read_bytes, []

    def mutate(path):
        result = read_bytes(path)
        if not triggered:
            triggered.append(True)
            table.at[pid, "A_LFSR"] = "4"
        return result

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", mutate)
            with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
                value.validate()
        assert triggered
    finally:
        table.at[pid, "A_LFSR"] = previous
    value.validate()


@pytest.mark.parametrize("bad", ["missing_header", "duplicate_key", "short_row"])
def test_literal_scanner_refuses_incomplete_or_duplicate_source(bad):
    row = dict.fromkeys(owner.ASEC_COLUMNS, "0")
    row.update(PERIDNUM="0000000000000000000001", PH_SEQ="7", A_LINENO="1")
    columns = owner.ASEC_COLUMNS
    if bad == "missing_header":
        columns = tuple(c for c in columns if c != "A_LFSR")
    values = [row[c] for c in columns]
    if bad == "short_row":
        values.pop()
    text = ",".join(columns) + "\n" + ",".join(values) + "\n"
    if bad == "duplicate_key":
        text += ",".join(values) + "\n"
    with pytest.raises(ValueError):
        owner._scan(
            io.BytesIO(text.encode()),
            survey="asec",
            wanted=set(),
            selected={},
            complete={},
            maximum=2,
        )
