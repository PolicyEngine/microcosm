from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE
from microcosm.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    AcsPumsSource,
    build_acs_pums_unit_frame,
    load_acs_pums_tables,
)
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


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
        "NP": 1,
        "ADJHSG": 1_000_000,
        "TEN": 1,
        "RNTP": None,
        "GRNTP": None,
        "TAXAMT": 2_400,
        "TYPEHUGQ": 1,
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
        "PWGTP": 10,
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
                _household(
                    "2024HU0000003",
                    ST="12",
                    PUMA="00500",
                    WGTP=30,
                    NP=0,
                ),
            ],
            "psam_husb.csv": [_household("2024HU0000001", NP=3)],
        },
    )
    _write_csv_zip(
        person_zip,
        {
            "psam_pusa.csv": [
                _person("2024HU0000002", 1, 20, MAR=4, WAGP=30_000),
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
                    MAR=5,
                    WAGP=None,
                ),
            ],
        },
    )
    return AcsPumsSource(household_zip=household_zip, person_zip=person_zip)


def test_acs_loader_preserves_hours_and_allocation_without_filling_blanks(tmp_path):
    household_zip = tmp_path / "hours-hh.zip"
    person_zip = tmp_path / "hours-person.zip"
    _write_csv_zip(household_zip, {"psam_husa.csv": [_household("hours", NP=2)]})
    _write_csv_zip(
        person_zip,
        {
            "psam_pusa.csv": [
                _person("hours", 1, 20, WKHP=40, WKL=1, FWKHP=1),
                _person("hours", 2, 25, AGEP=12, WKHP=None, WKL=None, FWKHP=0),
            ]
        },
    )
    tables, _ = load_acs_pums_tables(AcsPumsSource(household_zip, person_zip))
    assert tables["person"]["WKHP"].iloc[0] == 40
    assert pd.isna(tables["person"]["WKHP"].iloc[1])
    assert tables["person"]["FWKHP"].tolist() == [1, 0]
    assert "weekly_hours_worked_before_lsr" not in tables["person"]


_IMMIGRATION_SOURCES = {
    "CIT": 5,
    "YOEP": 2015,
    "POBP": 303,
    "HINS3": 2,
    "HINS4": 1,
    "HINS5": 2,
    "HINS6": 2,
    "HINS7": 2,
    "COW": 5,
    "SCHG": 15,
    "ESR": 4,
    "MIL": 2,
}


def _immigration_source(tmp_path: Path) -> AcsPumsSource:
    household_zip = tmp_path / "immigration-hh.zip"
    person_zip = tmp_path / "immigration-person.zip"
    _write_csv_zip(household_zip, {"psam_husa.csv": [_household("imm", NP=2)]})
    _write_csv_zip(
        person_zip,
        {
            "psam_pusa.csv": [
                _person("imm", 1, 20, **_IMMIGRATION_SOURCES),
                # A US-born child: Census blanks for entry year, class of
                # worker, school, employment and military service.
                _person(
                    "imm",
                    2,
                    25,
                    AGEP=10,
                    MAR=5,
                    WAGP=None,
                    **{
                        **_IMMIGRATION_SOURCES,
                        "CIT": 1,
                        "YOEP": None,
                        "POBP": 6,
                        "HINS4": 2,
                        "COW": None,
                        "SCHG": None,
                        "ESR": None,
                        "MIL": None,
                    },
                ),
            ]
        },
    )
    return AcsPumsSource(household_zip, person_zip)


def test_acs_loader_keeps_immigration_sources_and_renames_military_service(
    tmp_path: Path,
) -> None:
    """microcosm#1020: ACS MIL (military service) must not land in the CPS
    ``MIL`` (military health coverage) column the ASEC stage reads."""

    tables, _ = load_acs_pums_tables(_immigration_source(tmp_path))
    person = tables["person"]
    assert "MIL" not in person
    assert person["ACS_MIL"].iloc[0] == 2
    for column, value in _IMMIGRATION_SOURCES.items():
        assert person["ACS_MIL" if column == "MIL" else column].iloc[0] == value
    # Census blanks stay missing; they are never read as a code or a zero.
    for column in ("YOEP", "COW", "SCHG", "ESR", "ACS_MIL"):
        assert pd.isna(person[column].iloc[1])
    assert person["CIT"].tolist() == [5, 1]


def test_acs_loader_leaves_absent_immigration_sources_absent(tmp_path: Path) -> None:
    tables, _ = load_acs_pums_tables(_source(tmp_path))
    for column in (*_IMMIGRATION_SOURCES, "ACS_MIL"):
        assert column not in tables["person"]


_WORK_DISABILITY_SOURCES = {
    "WKWN": 26,
    "DDRS": 2,
    "DEAR": 1,
    "DEYE": 2,
    "DOUT": 2,
    "DPHY": 2,
    "DREM": 2,
}


def _work_disability_source(tmp_path: Path) -> AcsPumsSource:
    household_zip = tmp_path / "work-disability-hh.zip"
    person_zip = tmp_path / "work-disability-person.zip"
    _write_csv_zip(household_zip, {"psam_husa.csv": [_household("wd", NP=2)]})
    _write_csv_zip(
        person_zip,
        {
            "psam_pusa.csv": [
                _person(
                    "wd", 1, 20, WKHP=20, WKL=1, FWKHP=0, **_WORK_DISABILITY_SOURCES
                ),
                # A 3-year-old: Census blanks for weeks worked and for the
                # items asked from age 5 (DREM/DPHY/DDRS) and 15 (DOUT).
                _person(
                    "wd",
                    2,
                    25,
                    AGEP=3,
                    MAR=5,
                    WAGP=None,
                    **{
                        **_WORK_DISABILITY_SOURCES,
                        "WKWN": None,
                        "DEAR": 2,
                        "DREM": None,
                        "DPHY": None,
                        "DDRS": None,
                        "DOUT": None,
                    },
                ),
            ]
        },
    )
    return AcsPumsSource(household_zip, person_zip)


def test_acs_loader_keeps_weeks_worked_and_disability_items(tmp_path: Path) -> None:
    """microcosm#1021: the local lane maps these natively; blanks stay blank."""

    tables, _ = load_acs_pums_tables(_work_disability_source(tmp_path))
    person = tables["person"]
    for column, value in _WORK_DISABILITY_SOURCES.items():
        assert person[column].iloc[0] == value
    for column in ("WKWN", "DREM", "DPHY", "DDRS", "DOUT"):
        assert pd.isna(person[column].iloc[1])
    assert person["DEAR"].tolist() == [1, 2]
    for column in ("weeks_worked", "is_disabled", "is_blind"):
        assert column not in person


def test_acs_loader_leaves_absent_work_disability_sources_absent(
    tmp_path: Path,
) -> None:
    tables, _ = load_acs_pums_tables(_source(tmp_path))
    for column in _WORK_DISABILITY_SOURCES:
        assert column not in tables["person"]


def _receipt_source(tmp_path: Path) -> AcsPumsSource:
    household_zip = tmp_path / "receipt-hh.zip"
    person_zip = tmp_path / "receipt-person.zip"
    _write_csv_zip(
        household_zip,
        {
            "psam_husa.csv": [
                _household("rcpt1", NP=2, FS=1),
                # Group quarters: FS is a housing-unit item, blank here.
                _household("rcpt2", NP=1, WGTP=0, TYPEHUGQ=3, TEN=None, FS=None),
            ]
        },
    )
    _write_csv_zip(
        person_zip,
        {
            "psam_pusa.csv": [
                _person("rcpt1", 1, 20, MAR=5, PAP=2_400),
                # Under 15: PAP is a Census blank.
                _person("rcpt1", 2, 25, AGEP=9, MAR=5, WAGP=None, PAP=None),
                _person("rcpt2", 1, 38, MAR=5, PAP=0),
            ]
        },
    )
    return AcsPumsSource(household_zip, person_zip)


def test_acs_loader_keeps_snap_recipiency_and_public_assistance(
    tmp_path: Path,
) -> None:
    """microcosm#1022: the local lane anchors receives_snap on household FS and
    records person PAP; blanks stay blank."""

    tables, _ = load_acs_pums_tables(_receipt_source(tmp_path))
    household = tables["household"]
    assert household["FS"].iloc[0] == 1
    assert pd.isna(household["FS"].iloc[1])
    person = tables["person"]
    assert person["PAP"].iloc[0] == 2_400
    assert pd.isna(person["PAP"].iloc[1])
    assert person["PAP"].iloc[2] == 0
    for column in ("receives_snap", "receives_tanf"):
        assert column not in household
        assert column not in person


def test_built_acs_frame_carries_fs_on_the_household_table(tmp_path: Path) -> None:
    pytest.importorskip("microunit")  # sanctioned tax-unit constructor (us extra)
    frame, _metadata = build_acs_pums_unit_frame(_receipt_source(tmp_path))
    household = frame.table("household")
    assert household["FS"].iloc[0] == 1
    assert pd.isna(household["FS"].iloc[1])
    assert "FS" not in frame.table("person")
    assert "receives_snap" not in frame.table("spm_unit")


def test_acs_loader_leaves_absent_receipt_sources_absent(tmp_path: Path) -> None:
    tables, _ = load_acs_pums_tables(_source(tmp_path))
    assert "FS" not in tables["household"]
    assert "PAP" not in tables["person"]


def _vehicle_source(tmp_path: Path) -> AcsPumsSource:
    household_zip = tmp_path / "vehicle-hh.zip"
    person_zip = tmp_path / "vehicle-person.zip"
    _write_csv_zip(
        household_zip,
        {
            "psam_husa.csv": [
                _household("veh1", NP=1, VEH=6),
                _household("veh2", NP=1, VEH=0),
                # Group quarters: VEH is a housing-unit item, blank here.
                _household("veh3", NP=1, WGTP=0, TYPEHUGQ=3, TEN=None, VEH=None),
            ]
        },
    )
    _write_csv_zip(
        person_zip,
        {
            "psam_pusa.csv": [
                _person("veh1", 1, 20, MAR=5),
                _person("veh2", 1, 20, MAR=5),
                _person("veh3", 1, 38, MAR=5),
            ]
        },
    )
    return AcsPumsSource(household_zip, person_zip)


def test_acs_loader_keeps_vehicles_available(tmp_path: Path) -> None:
    """microcosm#1022: the local lane writes VEH as the household vehicle
    count; the top code and the group-quarters blank are kept as read."""

    tables, _ = load_acs_pums_tables(_vehicle_source(tmp_path))
    household = tables["household"].set_index("SERIALNO")
    assert household.loc["veh1", "VEH"] == 6
    assert household.loc["veh2", "VEH"] == 0
    assert pd.isna(household.loc["veh3", "VEH"])
    assert "household_vehicles_owned" not in household
    assert "VEH" not in load_acs_pums_tables(_source(tmp_path))[0]["household"]


def test_built_acs_frame_carries_veh_on_the_household_table(tmp_path: Path) -> None:
    pytest.importorskip("microunit")  # sanctioned tax-unit constructor (us extra)
    frame, _metadata = build_acs_pums_unit_frame(_vehicle_source(tmp_path))
    household = frame.table("household").set_index("SERIALNO")
    assert household.loc["veh1", "VEH"] == 6
    assert pd.isna(household.loc["veh3", "VEH"])
    assert "VEH" not in frame.table("person")
    assert "household_vehicles_owned" not in household


def test_built_acs_frame_carries_the_immigration_sources(tmp_path: Path) -> None:
    pytest.importorskip("microunit")  # sanctioned tax-unit constructor (us extra)
    frame, _metadata = build_acs_pums_unit_frame(_immigration_source(tmp_path))
    person = frame.table("person").sort_values("source_row_id")
    assert "MIL" not in person
    assert person["ACS_MIL"].iloc[0] == 2
    assert person["CIT"].tolist() == [5, 1]
    assert person["SPORDER"].tolist() == [1, 2]
    assert frame.table("household")["SERIALNO"].tolist() == ["imm"]


def _asec_shaped_frame() -> Frame:
    """Return one ASEC-like row sharing only structural and lineage fields."""

    person = pd.DataFrame(
        {
            "person_id": np.asarray([101], dtype=np.int64),
            "person_household_id": np.asarray([201], dtype=np.int64),
            "person_tax_unit_id": np.asarray([301], dtype=np.int64),
            "person_spm_unit_id": np.asarray([401], dtype=np.int64),
            "person_family_id": np.asarray([501], dtype=np.int64),
            "person_marital_unit_id": np.asarray([601], dtype=np.int64),
            "source_year": np.asarray([2024], dtype=np.int64),
            "source_household_id": np.asarray([7], dtype=np.int64),
            # Pinned to the canonical storage, exactly as checkpoint restore
            # delivers the real ASEC channel. A bare astype(str) would follow
            # the environment's storage default and mask the cross-channel
            # divergence this fixture exists to model (pyarrow environments
            # resolve fresh 'str' casts to pyarrow storage).
            "source_person_id": pd.Series(["7-1"]).astype(CANONICAL_STRING_DTYPE),
            "source_row_id": np.asarray([0], dtype=np.int64),
            "A_AGE": np.asarray([55], dtype=np.int64),
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {"household_id": np.asarray([201], dtype=np.int64)}
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([301], dtype=np.int64)}
            ),
            "spm_unit": pd.DataFrame(
                {"spm_unit_id": np.asarray([401], dtype=np.int64)}
            ),
            "family": pd.DataFrame({"family_id": np.asarray([501], dtype=np.int64)}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([601], dtype=np.int64)}
            ),
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([10.0], dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["asec_2024"], dtype=object),
    )


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
    pytest.importorskip("microunit")  # sanctioned tax-unit constructor (us extra)
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
    assert household["puma"].tolist() == ["0612345", "3600100"]
    assert household["puma_geoid"].tolist() == ["0612345", "3600100"]
    assert frame.weights_for("household").values.tolist() == [10.0, 20.0]
    assert "SERIALNO" not in frame.table("person")
    person = frame.table("person")
    assert person["source_household_id"].tolist() == [1, 1, 1, 2]
    assert person["source_person_id"].tolist() == ["1", "2", "3", "1"]
    assert person["source_row_id"].tolist() == [0, 1, 2, 3]
    assert person["source_household_id"].dtype == np.dtype(np.int64)
    assert pd.api.types.is_string_dtype(person["source_person_id"].dtype)
    assert person["source_row_id"].dtype == np.dtype(np.int64)
    assert metadata["weighted_household_population"] == pytest.approx(30.0)


def test_build_acs_pums_unit_frame_derives_only_structural_relationship_fields(
    tmp_path: Path,
) -> None:
    pytest.importorskip("microunit")  # sanctioned tax-unit constructor (us extra)
    frame, _metadata = build_acs_pums_unit_frame(_source(tmp_path), chunksize=2)
    people = frame.table("person").set_index(
        ["source_household_id", "source_person_id"]
    )
    head = people.loc[(1, "1")]
    spouse = people.loc[(1, "2")]
    child = people.loc[(1, "3")]
    lone_head = people.loc[(2, "1")]

    assert (head["A_SPOUSE"], spouse["A_SPOUSE"], child["A_SPOUSE"]) == (2, 1, 0)
    assert (child["PEPAR1"], child["PEPAR2"]) == (1, 2)
    assert (head["A_EXPRRP"], spouse["A_EXPRRP"], child["A_EXPRRP"]) == (1, 4, 5)
    assert lone_head["A_EXPRRP"] == 2
    assert (head["A_MARITL"], spouse["A_MARITL"], child["A_MARITL"]) == (
        1,
        1,
        7,
    )
    assert lone_head["A_MARITL"] == 6
    assert head["person_marital_unit_id"] == spouse["person_marital_unit_id"]
    assert child["person_marital_unit_id"] != head["person_marital_unit_id"]


def test_built_acs_lineage_assembles_with_asec_without_measured_coercion(
    tmp_path: Path,
) -> None:
    pytest.importorskip("microunit")  # sanctioned tax-unit constructor (us extra)
    acs, _metadata = build_acs_pums_unit_frame(_source(tmp_path), chunksize=1)
    measured_wages = acs.table("person")["WAGP"].copy()
    raw_serials = acs.table("household")["SERIALNO"].copy()

    assembled = assemble_spines(
        {"asec": _asec_shaped_frame(), "acs": acs},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )

    person = assembled.table("person")
    acs_person = person.loc[person["person_support_channel"].eq("acs")].sort_values(
        "source_row_id"
    )
    pd.testing.assert_series_equal(
        acs_person["WAGP"].reset_index(drop=True),
        measured_wages.reset_index(drop=True),
    )
    assert person["source_household_id"].dtype == np.dtype(np.int64)
    assert (
        person["source_person_id"].dtype
        == _asec_shaped_frame().table("person")["source_person_id"].dtype
    )
    assert person["source_row_id"].dtype == np.dtype(np.int64)

    household = assembled.table("household")
    acs_household = household.loc[household["household_support_channel"].eq("acs")]
    pd.testing.assert_series_equal(
        acs_household["SERIALNO"].reset_index(drop=True),
        raw_serials.reset_index(drop=True),
        check_dtype=False,
    )


def test_build_acs_pums_unit_frame_canonicalizes_string_storage(
    tmp_path: Path,
) -> None:
    """The parse boundary must not leak environment-resolved string storage.

    With pyarrow installed, pandas resolves fresh ``astype(str)`` casts to
    pyarrow-backed storage while checkpoint-restored channels carry the
    canonical python-backed dtype — the exact split that failed the first
    full-scale spine assembly (both sides printed as 'str').
    """

    pytest.importorskip("microunit")
    frame, _metadata = build_acs_pums_unit_frame(_source(tmp_path), chunksize=1)

    for entity in frame.entities:
        table = frame.table(entity)
        offending = {
            column: dtype
            for column, dtype in table.dtypes.items()
            if isinstance(dtype, pd.StringDtype) and dtype != CANONICAL_STRING_DTYPE
        }
        assert not offending, f"{entity} carries non-canonical string storage"
    assert frame.table("person")["source_person_id"].dtype == CANONICAL_STRING_DTYPE


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
    pytest.importorskip("microunit")  # sanctioned tax-unit constructor (us extra)
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
                    MAR=5,
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


def test_build_acs_pums_unit_frame_handles_gq_alongside_multi_person_housing(
    tmp_path: Path,
) -> None:
    pytest.importorskip("microunit")  # sanctioned tax-unit constructor (us extra)
    household_zip = tmp_path / "csv_hus.zip"
    person_zip = tmp_path / "csv_pus.zip"
    _write_csv_zip(
        household_zip,
        {
            "psam_husa.csv": [
                _household("2024HU0000001", WGTP=10, NP=2),
                _household(
                    "2024GQ0000001",
                    WGTP=0,
                    TYPEHUGQ=2,
                    TEN=None,
                    TAXAMT=None,
                ),
            ]
        },
    )
    _write_csv_zip(
        person_zip,
        {
            "psam_pusa.csv": [
                _person("2024HU0000001", 1, 20, PWGTP=11),
                _person("2024HU0000001", 2, 25, MAR=5, PWGTP=12),
                _person("2024GQ0000001", 1, 37, MAR=5, PWGTP=99),
            ]
        },
    )

    frame, _metadata = build_acs_pums_unit_frame(
        AcsPumsSource(household_zip=household_zip, person_zip=person_zip)
    )

    assert frame.weights_for("household").values.tolist() == [99.0, 10.0]


def test_build_acs_pums_unit_frame_uses_adjusted_income_for_dependency_test(
    tmp_path: Path,
) -> None:
    pytest.importorskip("microunit")  # sanctioned tax-unit constructor (us extra)
    household_zip = tmp_path / "csv_hus.zip"
    person_zip = tmp_path / "csv_pus.zip"
    _write_csv_zip(
        household_zip,
        {"psam_husa.csv": [_household("2024HU0000001", NP=2)]},
    )
    _write_csv_zip(
        person_zip,
        {
            "psam_pusa.csv": [
                _person("2024HU0000001", 1, 20, MAR=5, WAGP=50_000),
                _person(
                    "2024HU0000001",
                    2,
                    33,
                    AGEP=30,
                    MAR=5,
                    ADJINC=1_100_000,
                    WAGP=100_000,
                ),
            ]
        },
    )

    frame, _metadata = build_acs_pums_unit_frame(
        AcsPumsSource(household_zip=household_zip, person_zip=person_zip)
    )
    people = frame.table("person")

    assert people["person_tax_unit_id"].nunique() == 2
    assert "WSAL_VAL" not in people
    assert "SEMP_VAL" not in people


def test_max_households_prioritizes_housing_units_and_filters_person_chunks(
    tmp_path: Path,
) -> None:
    household_zip = tmp_path / "csv_hus.zip"
    person_zip = tmp_path / "csv_pus.zip"
    _write_csv_zip(
        household_zip,
        {
            "psam_husa.csv": [
                _household("2024HU0000001", NP=2),
                _household("2024HU0000002", NP=1),
                _household("2024HU0000003", NP=0),
                _household(
                    "2024GQ0000001",
                    NP=1,
                    WGTP=0,
                    TYPEHUGQ=2,
                    TEN=None,
                    TAXAMT=None,
                ),
            ]
        },
    )
    _write_csv_zip(
        person_zip,
        {
            "psam_pusa.csv": [
                _person("2024HU0000001", 1, 20),
                _person("2024HU0000001", 2, 25, MAR=5),
                _person("2024HU0000002", 1, 20, MAR=5),
                _person("2024GQ0000001", 1, 37, MAR=5, PWGTP=99),
            ]
        },
    )

    tables, metadata = load_acs_pums_tables(
        AcsPumsSource(
            household_zip=household_zip,
            person_zip=person_zip,
            max_households=2,
        ),
        chunksize=1,
    )

    assert set(tables["household"]["TYPEHUGQ"]) == {1}
    assert len(tables["household"]) == 2
    assert len(tables["person"]) == 3
    assert metadata["vacant_household_rows_dropped"] == 1


def test_load_acs_pums_tables_rejects_np_person_count_mismatch(
    tmp_path: Path,
) -> None:
    household_zip = tmp_path / "csv_hus.zip"
    person_zip = tmp_path / "csv_pus.zip"
    _write_csv_zip(
        household_zip,
        {"psam_husa.csv": [_household("2024HU0000001", NP=2)]},
    )
    _write_csv_zip(
        person_zip,
        {"psam_pusa.csv": [_person("2024HU0000001", 1, 20)]},
    )

    with pytest.raises(ValueError, match="NP/person row-count mismatch"):
        load_acs_pums_tables(
            AcsPumsSource(household_zip=household_zip, person_zip=person_zip)
        )


def test_load_acs_pums_tables_accepts_official_state_header_alias(
    tmp_path: Path,
) -> None:
    household_zip = tmp_path / "csv_hus.zip"
    person_zip = tmp_path / "csv_pus.zip"
    housing = _household("2024HU0000001")
    housing["STATE"] = housing.pop("ST")
    _write_csv_zip(household_zip, {"psam_husa.csv": [housing]})
    _write_csv_zip(
        person_zip,
        {"psam_pusa.csv": [_person("2024HU0000001", 1, 20)]},
    )

    tables, _metadata = load_acs_pums_tables(
        AcsPumsSource(household_zip=household_zip, person_zip=person_zip)
    )

    assert tables["household"]["ST"].tolist() == ["06"]


def test_load_acs_pums_tables_requires_native_mapping_columns(
    tmp_path: Path,
) -> None:
    household_zip = tmp_path / "csv_hus.zip"
    person_zip = tmp_path / "csv_pus.zip"
    housing = _household("2024HU0000001")
    del housing["TAXAMT"]
    _write_csv_zip(household_zip, {"psam_husa.csv": [housing]})
    _write_csv_zip(
        person_zip,
        {"psam_pusa.csv": [_person("2024HU0000001", 1, 20)]},
    )

    with pytest.raises(ValueError, match="TAXAMT"):
        load_acs_pums_tables(
            AcsPumsSource(household_zip=household_zip, person_zip=person_zip)
        )
