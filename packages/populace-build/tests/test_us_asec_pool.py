from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.asec_pool import (
    ASEC_PERSON_WEIGHT_COLUMN,
    AsecSource,
    build_pooled_asec_unit_frame,
    load_asec_h5_tables,
    pool_asec_sources,
)

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
        household_rows.append(
            {
                "H_SEQ": household_id,
                "HSUP_WGT": household_weight,
                "GESTFIPS": state_fips,
            }
        )
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
    assert "state_fips" not in frame.table("person")


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
