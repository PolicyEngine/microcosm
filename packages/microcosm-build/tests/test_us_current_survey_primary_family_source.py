"""Genuine source issuers over small, wholly invented original survey archives."""

import hashlib
import io
import json
import shutil
import sys
from dataclasses import replace

import pandas as pd
import pytest

from microcosm.build.us_runtime import current_survey_primary_family_source as owner


def primary_family_source_arguments(tmp_path, patch):
    import test_us_survey_population_preparation as fixture
    from test_us_acs_person_coverage_authentication import build_fixture
    from test_us_current_asec_demographics import _demographic_arguments

    from microcosm.build.us_runtime import asec_demographic_source as demographics
    from microcosm.build.us_runtime import asec_person_income_source as restoration
    from microcosm.build.us_runtime import native_household_origin as origin

    build = build_fixture

    def archives(*args, **kwargs):
        # Add an actual resident spouse before archive pins or owners exist.
        h = pd.read_csv(
            io.BytesIO(kwargs["household_members"][0].data),
            dtype=str,
            keep_default_na=False,
        )
        p = pd.read_csv(
            io.BytesIO(kwargs["person_members"][0].data),
            dtype=str,
            keep_default_na=False,
        )
        p["MAR"] = "5"
        hh = "2024HU0000001"
        head = p.SERIALNO.eq(hh) & p.SPORDER.eq("1")
        p.loc[head, "MAR"] = "1"
        child = p.SERIALNO.eq(hh) & p.SPORDER.eq("2")
        p.loc[child, "AGEP"] = "5"
        spouse = p.loc[head].copy(deep=True)
        spouse["SPORDER"], spouse["RELSHIPP"], spouse["AGEP"] = "3", "23", "32"
        p = pd.concat([p, spouse], ignore_index=True)
        h.loc[h.SERIALNO.eq(hh), "NP"] = "3"
        kwargs["household_members"] = (
            fixture.Member("psam_husa.csv", fixture._csv(h.to_dict("records"))),
        )
        kwargs["person_members"] = (
            fixture.Member(
                "psam_pusa.csv", fixture._csv(p.iloc[::-1].to_dict("records"))
            ),
        )
        return build(*args, **kwargs)

    patch.setattr(fixture, "build_fixture", archives)
    arguments = _demographic_arguments(tmp_path, patch)
    folder = arguments["source_dir"] / "asec"
    pins, paths = [], {}
    for year, member, archive, *_ in owner.reader.source_reader.asec._MEMBER_PINS:
        path = folder / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        raw["A_MARITL"], raw["A_SPOUSE"] = "7", "0"
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
    for module in (owner.reader.source_reader.asec, restoration, demographics):
        patch.setattr(module, "_MEMBER_PINS", tuple(pins))
    restored = tmp_path / "primary-restored-income"
    restoration.restore_asec_person_income_source(
        folder / "parent.h5",
        folder / "household-attachment.h5",
        member_paths=paths,
        output_dir=restored,
    )
    shutil.copyfile(
        restored / restoration.CHECKPOINT_FILENAME,
        folder / "person-income-attachment.h5",
    )
    path = folder / "hhpub25.csv"
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw["HRHTYPE"] = "3"
    raw.to_csv(path, index=False)
    payload = path.read_bytes()
    patch.setattr(
        origin,
        "_ASEC_MEMBER_PINS",
        tuple(
            replace(
                pin,
                member_sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
            for pin in origin._ASEC_MEMBER_PINS
        ),
    )
    return arguments


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        arguments = primary_family_source_arguments(
            tmp_path_factory.mktemp("primary-family"), patch
        )
        preparation = owner.source.prepare_authenticated_survey_population(**arguments)
        original = owner.source._frame_identity(preparation._checked()[2].frame)
        yield preparation, arguments
        assert owner.source._frame_identity(preparation._checked()[2].frame) == original


def test_original_archives_qualify_primary_children_and_exclude_gq(actual):
    preparation, _ = actual
    result = owner.qualify_current_survey_primary_family(preparation)
    family = owner.classifier
    table = result.household
    married = table.loc[table[family.PRIMARY].eq(1)]
    assert len(married) == 1
    assert married[family.CHILD_COUNT].tolist() == [1]
    assert married[family.CHILD_U6].tolist() == [1]
    assert table[family.HOUSEHOLD_COUNT].sum() == 4
    assert table[family.HOUSEHOLD_POPULATION].sum() == 8
    assert table["household_status"].eq("excluded_group_quarters").sum() == 2
    assert table[family.HOUSEHOLD_COUNT].equals(
        table[list(family.SIZE_BINS)].sum(axis=1).astype("Int64")
    )
    asec = result.origins.source.eq("asec")
    assert table.loc[asec, family.SPOUSE_PAIRS].tolist() == [0, 0]
    assert table.loc[~asec, family.SPOUSE_PAIRS].isna().all()
    receipt = json.loads(result.receipt)
    assert receipt["readsets"] == [list(columns) for columns in family.READSETS]
    assert receipt["projection_sha256"] == owner.roles._sha(result.projection)
    assert not any(
        receipt[name]
        for name in (
            "source_admission_issued",
            "population_admission_issued",
            "graph_attachment_qualified",
            "release_eligible",
        )
    )
    assert (
        owner.qualify_current_survey_primary_family(preparation).receipt
        == result.receipt
    )
    result.household.iloc[0, result.household.columns.get_loc(family.PRIMARY)] = 99
    fresh = owner.qualify_current_survey_primary_family(preparation)
    assert not fresh.household[family.PRIMARY].eq(99).any()


def test_copied_preparation_cannot_supply_source_authority(actual):
    import copy

    preparation, _ = actual
    with pytest.raises(ValueError):
        owner.qualify_current_survey_primary_family(copy.copy(preparation))


@pytest.mark.parametrize("mutation", ["metric", "status", "raw", "function"])
def test_final_owner_callback_cannot_change_projection_or_literal_evidence(
    actual, monkeypatch, mutation
):
    preparation, _ = actual
    qualify_code = owner.qualify_current_survey_primary_family.__code__
    checked_code = type(preparation)._checked.__code__
    fired = []

    def profile(frame, event, arg):
        if (
            event != "return"
            or frame.f_code is not checked_code
            or frame.f_back is None
            or frame.f_back.f_code is not qualify_code
        ):
            return
        local = frame.f_back.f_locals
        if "result" not in local:
            return
        fired.append(True)
        if mutation in ("metric", "status"):
            column = (
                owner.classifier.PRIMARY if mutation == "metric" else "household_status"
            )
            local["result"].household.iloc[
                0, local["result"].household.columns.get_loc(column)
            ] = 99 if mutation == "metric" else "fabricated"
        elif mutation == "raw":
            next(iter(local["selected"]["asec_person"].values()))["A_SPOUSE"] = "16"
        else:
            monkeypatch.setattr(
                owner.classifier, "METRICS", owner.classifier.METRICS[:-1]
            )

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        if mutation == "raw":
            # Detached literal mutation after its freeze cannot change output.
            result = owner.qualify_current_survey_primary_family(preparation)
            assert result.household[owner.classifier.SPOUSE_PAIRS].dropna().eq(0).all()
        else:
            with pytest.raises(
                ValueError, match="IMPLEMENTATION_CHANGED|FINAL_PROJECTION"
            ):
                owner.qualify_current_survey_primary_family(preparation)
    finally:
        sys.setprofile(previous)
    assert fired == [True]


def test_original_member_change_refuses_before_classification(tmp_path, monkeypatch):
    arguments = primary_family_source_arguments(tmp_path, monkeypatch)
    preparation = owner.source.prepare_authenticated_survey_population(**arguments)
    path = arguments["source_dir"] / "asec" / "pppub25.csv"
    payload = path.read_bytes()
    path.write_bytes(payload.replace(b"A_SPOUSE", b"A_SPOUSX", 1))
    with pytest.raises(ValueError):
        owner.qualify_current_survey_primary_family(preparation)


def test_readset_extension_requires_original_join_fields(actual):
    preparation, _ = actual
    readsets = list(owner.classifier.READSETS)
    readsets[1] = tuple(name for name in readsets[1] if name != "PH_SEQ")
    with pytest.raises(ValueError, match="ORIGINAL_READSETS"):
        owner.reader._original_columns(preparation, readsets=tuple(readsets))
