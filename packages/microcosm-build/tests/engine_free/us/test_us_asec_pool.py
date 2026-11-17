from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.asec_pool import (
    ASEC_PERSON_WEIGHT_COLUMN,
    AsecSource,
    build_pooled_asec_unit_frame,
    load_asec_h5_tables,
    pool_asec_sources,
)
from microcosm.build.us_runtime.base_pool import (
    ACS_2024_1YR_SPINE,
    spine_column,
    with_optional_acs_spine,
)
from microcosm.build.us_runtime.h5_io import write_nullable_us_h5
from microcosm.build.us_runtime.puf_support import (
    clone_us_frame_for_puf_support,
    support_source_id_column,
)
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.build.us_runtime.support_provenance import (
    spine_source_id_column,
    support_channel_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

pytest.importorskip("microunit")
pytest.importorskip("tables")


def _asec_person(
    year_household: int,
    line: int,
    *,
    spouse: int = 0,
    parent1: int = 0,
) -> dict:
    sex = 1 if line == 1 else 2
    relationship_recode = 2
    if line == 1:
        relationship_recode = 1 if spouse else 2
    elif spouse:
        relationship_recode = 3 if sex == 1 else 4
    elif parent1:
        relationship_recode = 5
    return {
        "PERIDNUM": f"{year_household}-{line}",
        "PH_SEQ": year_household,
        "P_SEQ": line,
        "A_LINENO": line,
        "PF_SEQ": 1,
        "A_AGE": 40 if line == 1 else 38 if spouse else 10,
        "A_SEX": sex,
        "A_MARITL": 1 if spouse else 7,
        "A_SPOUSE": spouse,
        "PEPAR1": parent1,
        "PEPAR2": 0,
        "SPM_ID": year_household,
        "A_EXPRRP": relationship_recode,
        "A_FNLWGT": 10_000,
    }


def _write_asec(
    path: Path,
    *,
    households: list[
        tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]
    ],
    include_relationship_recode: bool = True,
    landlord_fields: bool = True,
    hpublic: int | None = None,
) -> None:
    person_rows = []
    household_rows = []
    for household in households:
        if len(household) == 2:
            household_id, n_people = household
            household_weight = 10_000
            state_fips = 6
        elif len(household) == 3:
            household_id, n_people, household_weight = household
            state_fips = 6
        else:
            household_id, n_people, household_weight, state_fips = household
        tenure = 1 + (household_id % 2)
        row = {
            "H_SEQ": household_id,
            "HSUP_WGT": household_weight,
            "GESTFIPS": state_fips,
            "H_TENURE": tenure,
        }
        if landlord_fields:
            # As the ASEC asks them: HPUBLIC of every non-owner household,
            # HLORENT only where HPUBLIC is 2 (no).
            public = 0 if tenure == 1 else 1 if household_id % 4 == 1 else 2
            if hpublic is not None:
                public = hpublic
            row["HPUBLIC"] = public
            row["HLORENT"] = 0 if public != 2 else 1 if household_id % 3 == 0 else 2
        household_rows.append(row)
        if n_people >= 2:
            person_rows.append(_asec_person(household_id, 1, spouse=2))
            person_rows.append(_asec_person(household_id, 2, spouse=1))
            for line in range(3, n_people + 1):
                person_rows.append(_asec_person(household_id, line, parent1=1))
        else:
            person_rows.append(_asec_person(household_id, 1))
    person = pd.DataFrame(person_rows)
    if not include_relationship_recode:
        person = person.drop(columns=["A_EXPRRP"])
    person.to_hdf(path, key="person", mode="w")
    pd.DataFrame(household_rows).to_hdf(path, key="household", mode="a")


def test_pool_asec_sources_makes_households_unique_and_scales_person_population(
    tmp_path: Path,
) -> None:
    current = tmp_path / "asec_2024.h5"
    prior = tmp_path / "asec_2023.h5"
    _write_asec(current, households=[(1, 2), (2, 1)])
    _write_asec(
        prior,
        households=[(1, 2), (3, 2)],
        include_relationship_recode=False,
    )

    pooled = pool_asec_sources(
        [
            AsecSource(year=2024, path=current),
            AsecSource(year=2023, path=prior),
        ]
    )
    person = pooled.person

    assert person["source_household_id"].tolist() == [1, 1, 2, 1, 1, 3, 3]
    assert person.groupby("source_year")["PH_SEQ"].nunique().to_dict() == {
        2023: 2,
        2024: 2,
    }
    assert (
        person.groupby("source_year")["PH_SEQ"]
        .apply(set)
        .iloc[0]
        .isdisjoint(person.groupby("source_year")["PH_SEQ"].apply(set).iloc[1])
    )

    weighted_population = person.groupby("source_year")[ASEC_PERSON_WEIGHT_COLUMN].sum()
    assert weighted_population.loc[2024] == pytest.approx(150.0)
    assert weighted_population.loc[2023] == pytest.approx(150.0)
    assert pooled.metadata["weighted_person_population"] == pytest.approx(300.0)
    assert {
        item["relationship_recode_source"] for item in pooled.metadata["sources"]
    } == {"source:A_EXPRRP", "derived:line_spouse_parent"}
    assert {item["source_file"] for item in pooled.metadata["sources"]} == {
        "asec_2023.h5",
        "asec_2024.h5",
    }
    assert person.loc[person["source_year"] == 2024, "A_EXPRRP"].tolist() == [
        1,
        4,
        2,
    ]
    assert person.loc[person["source_year"] == 2023, "A_EXPRRP"].tolist() == [
        1,
        4,
        1,
        4,
    ]
    assert set(person.loc[person["source_year"] == 2024, "SPM_ID"]).isdisjoint(
        set(person.loc[person["source_year"] == 2023, "SPM_ID"])
    )

    # Spouse pointers are line numbers within household; they must not be
    # offset with globally unique household ids.
    assert person.loc[0, "A_SPOUSE"] == 2
    assert person.loc[1, "A_SPOUSE"] == 1


def test_pool_asec_sources_rejects_duplicate_source_years(tmp_path: Path) -> None:
    first = tmp_path / "asec_2023_a.h5"
    second = tmp_path / "asec_2023_b.h5"
    _write_asec(first, households=[(1, 1)])
    _write_asec(second, households=[(1, 1)])

    with pytest.raises(ValueError, match="duplicate year"):
        pool_asec_sources(
            [
                AsecSource(year=2023, path=first),
                AsecSource(year=2023, path=second),
            ]
        )


def test_build_pooled_asec_unit_frame_runs_unit_assignment_on_pooled_source(
    tmp_path: Path,
) -> None:
    current = tmp_path / "asec_2024.h5"
    prior = tmp_path / "asec_2023.h5"
    _write_asec(current, households=[(1, 2, 10_000, 6)])
    _write_asec(prior, households=[(1, 2, 10_000, 36)])

    frame, metadata = build_pooled_asec_unit_frame(
        [
            AsecSource(year=2024, path=current),
            AsecSource(year=2023, path=prior),
        ],
        target_year=2024,
    )

    assert frame.n("person") == 4
    assert frame.n("household") == 2
    assert frame.n("spm_unit") == 2
    assert frame.n("tax_unit") == 2
    assert set(frame.table("person")["source_year"]) == {2023, 2024}
    assert set(frame.table("tax_unit")["filing_status_input"]) == {"JOINT"}
    assert frame.weights_for("household").total == pytest.approx(100.0)
    assert metadata["anchor_year"] == 2024
    assert metadata["target_person_population"] == pytest.approx(200.0)
    assert metadata["weighted_person_population"] == pytest.approx(200.0)
    assert np.array_equal(frame.table("household")["household_id"], np.array([1, 2]))
    assert frame.table("household")["state_fips"].tolist() == [6, 36]
    assert frame.table("household")["H_TENURE"].tolist() == [2, 2]
    assert "state_fips" not in frame.table("person")
    assert "H_TENURE" not in frame.table("person")


def test_the_landlord_fields_ride_on_the_household_table_unchanged(
    tmp_path: Path,
) -> None:
    # Households 1 (public housing), 3 (subsidised private renter), 4 (owner)
    # and 7 (unsubsidised private renter), by the fixture's codes.
    current = tmp_path / "asec_2024.h5"
    _write_asec(current, households=[(1, 1), (3, 1), (4, 1), (7, 1)])

    frame, _ = build_pooled_asec_unit_frame(
        [AsecSource(year=2024, path=current)],
        target_year=2024,
    )

    household = frame.table("household").sort_values("household_id")
    assert household["H_TENURE"].tolist() == [2, 2, 1, 2]
    assert household["HPUBLIC"].tolist() == [1, 2, 0, 2]
    assert household["HLORENT"].tolist() == [0, 1, 0, 2]
    for column in ("HPUBLIC", "HLORENT"):
        assert column not in frame.table("person")


def _one_acs_household() -> Frame:
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": [1],
                "person_household_id": [1],
                "person_tax_unit_id": [10],
                "person_spm_unit_id": [100],
                "person_family_id": [1_000],
                "person_marital_unit_id": [10_000],
            }
        ),
        "household": pd.DataFrame(
            {"household_id": [1], "state_fips": [6], "puma": ["0600101"]}
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": [10]}),
        "spm_unit": pd.DataFrame({"spm_unit_id": [100]}),
        "family": pd.DataFrame({"family_id": [1_000]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [10_000]}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray([50.0]), WeightKind.DESIGN)},
    )


def test_the_landlord_fields_survive_the_puf_clone_and_the_acs_spine(
    tmp_path: Path,
) -> None:
    # Both PUF clone halves keep their CPS household's codes; an ACS
    # household, which has no such field, is left missing.
    current = tmp_path / "asec_2024.h5"
    _write_asec(current, households=[(1, 1), (3, 1), (4, 1), (7, 1)])
    frame, _ = build_pooled_asec_unit_frame(
        [AsecSource(year=2024, path=current)],
        target_year=2024,
    )

    cloned = clone_us_frame_for_puf_support(frame)
    household = cloned.table("household")
    source = frame.table("household").set_index("household_id")
    origin = household[support_source_id_column("household")].to_numpy()
    assert len(household) == 2 * len(source)
    for column in ("H_TENURE", "HPUBLIC", "HLORENT"):
        assert household[column].tolist() == source.loc[origin, column].tolist()

    pooled = with_optional_acs_spine(cloned, _one_acs_household())
    table = pooled.table("household")
    acs = (table[spine_column("household")] == ACS_2024_1YR_SPINE).to_numpy()
    assert acs.sum() == 1
    for column in ("H_TENURE", "HPUBLIC", "HLORENT"):
        assert table.loc[acs, column].isna().all()
        assert table.loc[~acs, column].tolist() == household[column].tolist()


def test_the_landlord_fields_survive_assembly_the_clone_and_the_h5_export(
    tmp_path: Path,
) -> None:
    # Review r1: production assembles the CPS and ACS spines first, clones the
    # assembled frame (the preassembled branch the stacked PUF pass takes),
    # then writes the nullable H5. The codes survive each step, an ACS
    # household is left missing, and the export reads back as written.
    current = tmp_path / "asec_2024.h5"
    _write_asec(current, households=[(1, 1), (3, 1), (4, 1), (7, 1)])
    frame, _ = build_pooled_asec_unit_frame(
        [AsecSource(year=2024, path=current)],
        target_year=2024,
    )
    codes = frame.table("household").set_index("household_id")[
        ["H_TENURE", "HPUBLIC", "HLORENT"]
    ]

    assembled = assemble_spines(
        {"asec": frame, "acs": _one_acs_household()},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    cloned = clone_us_frame_for_puf_support(assembled)
    household = cloned.table("household")
    channel = household[support_channel_column("household")]
    assert (channel == "acs").sum() == 2
    assert (channel == "asec").sum() == 2 * len(codes)

    path = tmp_path / "export.h5"
    write_nullable_us_h5(cloned, path, period=2024, artifact_kind="test")
    written = pd.read_hdf(path, key="household")
    assert len(written) == len(household)
    spine_id = written[spine_source_id_column("household")]
    for column in ("H_TENURE", "HPUBLIC", "HLORENT"):
        on_acs = written.loc[written[support_channel_column("household")] == "acs"]
        assert on_acs[column].isna().all()
        on_asec = written[support_channel_column("household")] == "asec"
        expected = codes.loc[spine_id[on_asec], column].to_numpy()
        assert written.loc[on_asec, column].astype("int64").tolist() == (
            expected.tolist()
        )


def test_a_landlord_field_missing_from_one_pooled_year_is_refused(
    tmp_path: Path,
) -> None:
    current = tmp_path / "asec_2024.h5"
    prior = tmp_path / "asec_2023.h5"
    _write_asec(current, households=[(1, 2)])
    _write_asec(prior, households=[(3, 2)], landlord_fields=False)

    with pytest.raises(
        ValueError, match=r"HPUBLIC is missing from pooled year\(s\) \[2023\]"
    ):
        pool_asec_sources(
            [
                AsecSource(year=2024, path=current),
                AsecSource(year=2023, path=prior),
            ]
        )


def test_a_landlord_code_outside_the_census_values_is_refused(
    tmp_path: Path,
) -> None:
    current = tmp_path / "asec_2024.h5"
    _write_asec(current, households=[(1, 1)], hpublic=9)

    with pytest.raises(ValueError, match=r"HPUBLIC must be 0, 1 or 2.*\[1\]"):
        pool_asec_sources([AsecSource(year=2024, path=current)])


def test_duplicate_household_rows_that_disagree_are_refused(tmp_path: Path) -> None:
    # Review r1: duplicate H_SEQ rows kept the first row's codes and skipped
    # the rest's validation. Rows that agree still pool.
    current = tmp_path / "asec_2024.h5"
    _write_asec(current, households=[(1, 1), (3, 1)])
    household = pd.read_hdf(current, key="household")
    agreeing = pd.concat([household, household.iloc[[0]]], ignore_index=True)
    agreeing.to_hdf(current, key="household", mode="a")
    pooled = pool_asec_sources([AsecSource(year=2024, path=current)])
    assert pooled.person["HPUBLIC"].notna().all()

    for column, value in (("HPUBLIC", 9), ("HLORENT", 1), ("H_TENURE", 3)):
        conflicting = household.iloc[[0]].copy()
        conflicting[column] = value
        pd.concat([household, conflicting], ignore_index=True).to_hdf(
            current, key="household", mode="a"
        )
        with pytest.raises(
            ValueError,
            match=rf"{column} differs between duplicate household rows.*\[1\]",
        ):
            pool_asec_sources([AsecSource(year=2024, path=current)])


def test_sources_without_the_landlord_fields_still_pool(tmp_path: Path) -> None:
    current = tmp_path / "asec_2024.h5"
    _write_asec(current, households=[(1, 1)], landlord_fields=False)

    frame, _ = build_pooled_asec_unit_frame(
        [AsecSource(year=2024, path=current)],
        target_year=2024,
    )

    assert "HPUBLIC" not in frame.table("household")
    assert "HLORENT" not in frame.table("household")


def test_build_pooled_asec_unit_frame_aligns_household_weights_by_remapped_id(
    tmp_path: Path,
) -> None:
    current = tmp_path / "asec_2024.h5"
    _write_asec(current, households=[(2, 1, 10_000), (1, 1, 40_000)])

    frame, metadata = build_pooled_asec_unit_frame(
        [AsecSource(year=2024, path=current)],
        target_year=2024,
    )

    assert metadata["target_person_population"] == pytest.approx(500.0)
    assert frame.table("household")["household_id"].tolist() == [1, 2]
    assert frame.weights_for("household").values.tolist() == [400.0, 100.0]


def test_build_pooled_asec_unit_frame_anchors_population_to_target_year(
    tmp_path: Path,
) -> None:
    current = tmp_path / "asec_2024.h5"
    prior = tmp_path / "asec_2023.h5"
    _write_asec(current, households=[(1, 2), (2, 1)])
    _write_asec(prior, households=[(1, 2), (3, 2)])

    _frame, metadata = build_pooled_asec_unit_frame(
        [
            AsecSource(year=2023, path=prior),
            AsecSource(year=2024, path=current),
        ],
        target_year=2024,
    )

    assert metadata["anchor_year"] == 2024
    assert metadata["target_person_population"] == pytest.approx(300.0)
    by_year = {
        item["year"]: item["weighted_person_population"] for item in metadata["sources"]
    }
    assert by_year == {2023: pytest.approx(150.0), 2024: pytest.approx(150.0)}


def test_load_asec_h5_tables_opens_source_read_only(tmp_path: Path) -> None:
    source = tmp_path / "asec_2024.h5"
    _write_asec(source, households=[(1, 2)])
    source.chmod(0o444)
    try:
        tables = load_asec_h5_tables(source)
    finally:
        source.chmod(0o644)

    assert set(tables) == {"person", "household"}
    assert len(tables["person"]) == 2
