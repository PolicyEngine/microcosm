from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from populace.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    AcsPumsSource,
    build_acs_pums_unit_frame,
    load_acs_pums_tables,
)


def _write_csv_zip(
    path: Path,
    members: dict[str, list[dict[str, object]]],
) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, rows in members.items():
            archive.writestr(name, pd.DataFrame(rows).to_csv(index=False))


def _household(serialno: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "SERIALNO": serialno,
        "ST": "06",
        "PUMA": "12345",
        "WGTP": 10,
        "ADJHSG": 1_000_000,
        "TEN": 1,
        "RNTP": None,
        "GRNTP": None,
        "TAXAMT": 2_400,
    }
    row.update(overrides)
    return row


def _person(
    serialno: str,
    sporder: int,
    relationship: int,
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "SERIALNO": serialno,
        "SPORDER": sporder,
        "RELSHIPP": relationship,
        "AGEP": 40,
        "SEX": 1,
        "MAR": 1,
        "ADJINC": 1_000_000,
        "WAGP": 50_000,
        "SEMP": 0,
        "SSP": 0,
        "SSIP": 0,
        "RETP": 0,
        "INTP": 0,
    }
    row.update(overrides)
    return row


def _source(tmp_path: Path) -> AcsPumsSource:
    household_zip = tmp_path / "csv_hus.zip"
    person_zip = tmp_path / "csv_pus.zip"
    _write_csv_zip(
        household_zip,
        {
            "psam_husa.csv": [
                _household("2024HU0000002", ST="36", PUMA="00100", WGTP=20),
                _household("2024HU0000003", ST="12", PUMA="00500", WGTP=30),
            ],
            "psam_husb.csv": [_household("2024HU0000001")],
        },
    )
    _write_csv_zip(
        person_zip,
        {
            "psam_pusa.csv": [
                _person("2024HU0000002", 1, 20, MAR=6, WAGP=30_000),
                _person(
                    "2024HU0000001",
                    1,
                    20,
                    AGEP=42,
                    WAGP=60_000,
                ),
            ],
            "psam_pusb.csv": [
                _person(
                    "2024HU0000001",
                    2,
                    21,
                    AGEP=40,
                    SEX=2,
                    WAGP=45_000,
                ),
                _person(
                    "2024HU0000001",
                    3,
                    25,
                    AGEP=10,
                    MAR=6,
                    WAGP=None,
                ),
            ],
        },
    )
    return AcsPumsSource(household_zip=household_zip, person_zip=person_zip)


def test_load_acs_pums_tables_streams_all_csv_members_and_keeps_native_blanks(
    tmp_path: Path,
) -> None:
    tables, metadata = load_acs_pums_tables(_source(tmp_path), chunksize=1)

    assert tables["household"]["SERIALNO"].tolist() == [
        "2024HU0000001",
        "2024HU0000002",
    ]
    assert tables["person"]["SERIALNO"].tolist() == [
        "2024HU0000001",
        "2024HU0000001",
        "2024HU0000001",
        "2024HU0000002",
    ]
    assert pd.isna(tables["person"].loc[2, "WAGP"])
    assert metadata["vacant_household_rows_dropped"] == 1
    assert metadata["household_csv_members"] == ["psam_husa.csv", "psam_husb.csv"]
    assert metadata["person_csv_members"] == ["psam_pusa.csv", "psam_pusb.csv"]


def test_build_acs_pums_unit_frame_preserves_lineage_geography_and_weights(
    tmp_path: Path,
) -> None:
    frame, metadata = build_acs_pums_unit_frame(_source(tmp_path), chunksize=1)

    assert frame.n("household") == 2
    assert frame.n("person") == 4
    assert set(frame.strata) == {ACS_2024_1YR_SPINE}
    household = frame.table("household")
    assert household["SERIALNO"].tolist() == [
        "2024HU0000001",
        "2024HU0000002",
    ]
    assert household["ST"].tolist() == ["06", "36"]
    assert household["PUMA"].tolist() == ["12345", "00100"]
    assert household["state_fips"].tolist() == [6, 36]
    assert household["puma_geoid"].tolist() == ["0612345", "3600100"]
    assert frame.weights_for("household").values.tolist() == [10.0, 20.0]
    assert "SERIALNO" not in frame.table("person")
    assert frame.table("person")["source_row_id"].tolist() == [
        "acs_2024_1yr:2024HU0000001:1",
        "acs_2024_1yr:2024HU0000001:2",
        "acs_2024_1yr:2024HU0000001:3",
        "acs_2024_1yr:2024HU0000002:1",
    ]
    assert metadata["weighted_household_population"] == pytest.approx(30.0)


def test_build_acs_pums_unit_frame_derives_only_structural_relationship_fields(
    tmp_path: Path,
) -> None:
    frame, _metadata = build_acs_pums_unit_frame(_source(tmp_path), chunksize=2)
    people = frame.table("person").set_index("source_row_id")
    head = people.loc["acs_2024_1yr:2024HU0000001:1"]
    spouse = people.loc["acs_2024_1yr:2024HU0000001:2"]
    child = people.loc["acs_2024_1yr:2024HU0000001:3"]
    lone_head = people.loc["acs_2024_1yr:2024HU0000002:1"]

    assert (head["A_SPOUSE"], spouse["A_SPOUSE"], child["A_SPOUSE"]) == (2, 1, 0)
    assert (child["PEPAR1"], child["PEPAR2"]) == (1, 2)
    assert (head["A_EXPRRP"], spouse["A_EXPRRP"], child["A_EXPRRP"]) == (1, 4, 5)
    assert lone_head["A_EXPRRP"] == 2
    assert head["person_marital_unit_id"] == spouse["person_marital_unit_id"]
    assert child["person_marital_unit_id"] != head["person_marital_unit_id"]


def test_load_acs_pums_tables_rejects_duplicate_household_serialno(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    _write_csv_zip(
        source.household_zip,
        {
            "psam_husa.csv": [_household("2024HU0000001")],
            "psam_husb.csv": [_household("2024HU0000001")],
        },
    )

    with pytest.raises(ValueError, match="duplicate household SERIALNO"):
        load_acs_pums_tables(source, chunksize=1)


def test_load_acs_pums_tables_rejects_orphan_people(tmp_path: Path) -> None:
    source = _source(tmp_path)
    _write_csv_zip(
        source.person_zip,
        {"psam_pusa.csv": [_person("2024HU9999999", 1, 20)]},
    )

    with pytest.raises(ValueError, match="missing from the household archive"):
        load_acs_pums_tables(source, chunksize=1)


def test_load_acs_pums_tables_requires_native_structure_columns(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    _write_csv_zip(
        source.person_zip,
        {
            "psam_pusa.csv": [
                {
                    "SERIALNO": "2024HU0000001",
                    "SPORDER": 1,
                    "AGEP": 40,
                    "SEX": 1,
                    "MAR": 1,
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="RELSHIPP"):
        load_acs_pums_tables(source)


def test_build_acs_pums_unit_frame_uses_person_weight_for_gq_placeholder(
    tmp_path: Path,
) -> None:
    household_zip = tmp_path / "csv_hus.zip"
    person_zip = tmp_path / "csv_pus.zip"
    _write_csv_zip(
        household_zip,
        {
            "psam_husa.csv": [
                _household(
                    "2024GQ0000001",
                    WGTP=0,
                    TYPEHUGQ=2,
                    TEN=None,
                    TAXAMT=None,
                )
            ]
        },
    )
    _write_csv_zip(
        person_zip,
        {
            "psam_pusa.csv": [
                _person(
                    "2024GQ0000001",
                    1,
                    37,
                    MAR=6,
                    PWGTP=99,
                )
            ]
        },
    )

    frame, _metadata = build_acs_pums_unit_frame(
        AcsPumsSource(household_zip=household_zip, person_zip=person_zip)
    )

    assert frame.weights_for("household").values.tolist() == [99.0]
    assert frame.table("person")["A_EXPRRP"].tolist() == [14]
