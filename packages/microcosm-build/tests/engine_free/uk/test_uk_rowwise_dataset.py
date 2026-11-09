from __future__ import annotations

import pandas as pd
import pytest

from microcosm.build.uk_runtime import (
    ROWWISE_GEOGRAPHY_COLUMNS,
    clone_uk_dataset_tables_with_rowwise_geography,
    clone_uk_dataset_with_rowwise_geography,
    validate_uk_rowwise_dataset_tables,
    write_uk_rowwise_dataset,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import MassChangeRecord, WeightKind


class FakeUKDataset:
    time_period = "2023"
    household_weight_kind = WeightKind.DESIGN

    def __init__(
        self,
        *,
        person: pd.DataFrame,
        benunit: pd.DataFrame,
        household: pd.DataFrame,
    ):
        self.person = person
        self.benunit = benunit
        self.household = household


def household_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "WALES"],
        }
    )


def person_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1001, 2001, 2002],
            "person_household_id": [1, 2, 2],
            "person_benunit_id": [101, 201, 201],
            "dividend_income": [5.0, 9.0, 11.0],
        }
    )


def benunit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "benunit_id": [101, 201],
            "would_claim_uc": [True, False],
        }
    )


def crosswalk_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "oa_code": "E0001",
                "lsoa_code": "E0101",
                "msoa_code": "E0201",
                "la_code": "E06000063",
                "constituency_code": "E14000001",
                "region_code": "E12000007",
                "country": "England",
                "population": 100,
            },
            {
                "oa_code": "W0001",
                "lsoa_code": "W0101",
                "msoa_code": "W0201",
                "la_code": "W06000001",
                "constituency_code": "W07000041",
                "region_code": "W99999999",
                "country": "Wales",
                "population": 80,
            },
        ]
    )


def test_clone_uk_dataset_tables_assigns_geography_and_remaps_links() -> None:
    result = clone_uk_dataset_tables_with_rowwise_geography(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        crosswalk=crosswalk_frame(),
        n_clones=2,
        seed=1,
        source_year=2023,
    )

    assert result.time_period == "2023"
    assert result.id_multiplier == 10_000
    assert result.household["household_id"].tolist() == [1, 2, 10001, 10002]
    assert result.household["household_weight"].sum() == pytest.approx(30.0)
    assert set(ROWWISE_GEOGRAPHY_COLUMNS).issubset(result.household.columns)
    assert result.household["clone_index"].tolist() == [0, 0, 1, 1]
    assert result.household["source_household_key"].tolist() == [
        "2023:1",
        "2023:2",
        "2023:1",
        "2023:2",
    ]
    assert result.household["region_code_oa"].tolist() == [
        "E12000007",
        "W99999999",
        "E12000007",
        "W99999999",
    ]

    assert result.person["person_id"].tolist() == [
        1001,
        2001,
        2002,
        11001,
        12001,
        12002,
    ]
    assert result.person["person_household_id"].tolist() == [
        1,
        2,
        2,
        10001,
        10002,
        10002,
    ]
    assert result.person["person_benunit_id"].tolist() == [
        101,
        201,
        201,
        10101,
        10201,
        10201,
    ]
    assert result.person["dividend_income"].tolist() == [5.0, 9.0, 11.0] * 2
    assert result.person["clone_index"].tolist() == [0, 0, 0, 1, 1, 1]

    assert result.benunit["benunit_id"].tolist() == [101, 201, 10101, 10201]
    assert result.benunit["clone_index"].tolist() == [0, 0, 1, 1]
    validate_uk_rowwise_dataset_tables(
        result.person,
        result.benunit,
        result.household,
    )


def test_clone_uk_dataset_rejects_duck_typed_object() -> None:
    dataset = FakeUKDataset(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
    )

    with pytest.raises(TypeError, match="duck-typed in-memory carrier retired"):
        clone_uk_dataset_with_rowwise_geography(
            dataset,
            crosswalk_frame(),
            n_clones=1,
            seed=1,
        )


def test_clone_uk_dataset_accepts_a_frame_without_downgrading_kind() -> None:
    frame = uk_national_frame(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=30.0,
                new_total=30.0,
                declared_factor=1.0,
                reason="Toy reviewed record.",
            ),
        ),
    )

    result = clone_uk_dataset_with_rowwise_geography(
        frame,
        crosswalk_frame(),
        n_clones=1,
        seed=1,
    )

    assert result.time_period == "2023"
    # The typed weights carry the kind; a frame input can never silently
    # downgrade to design the way a bare duck-carrier once could.
    assert result.household_weight_kind is WeightKind.IMPORTANCE
    assert len(result.mass_log) == 2
    assert result.household["household_id"].tolist() == [1, 2]
    assert result.person["person_household_id"].tolist() == [1, 2, 2]


def test_validate_uk_rowwise_dataset_tables_rejects_broken_household_link() -> None:
    result = clone_uk_dataset_tables_with_rowwise_geography(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        crosswalk=crosswalk_frame(),
        source_year=2023,
    )
    person = result.person.copy()
    person.loc[0, "person_household_id"] = 999

    with pytest.raises(ValueError, match="person_household_id"):
        validate_uk_rowwise_dataset_tables(person, result.benunit, result.household)


def test_clone_uk_dataset_h5_roundtrip(tmp_path) -> None:
    pytest.importorskip("tables")
    result = clone_uk_dataset_tables_with_rowwise_geography(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        crosswalk=crosswalk_frame(),
        source_year=2023,
    )
    source = write_uk_rowwise_dataset(result, tmp_path / "source.h5")

    output = tmp_path / "cloned.h5"
    cloned = clone_uk_dataset_with_rowwise_geography(
        source,
        crosswalk_frame(),
        output_path=output,
        n_clones=2,
        seed=2,
    )

    assert cloned.output_path == output
    assert output.exists()
    with pd.HDFStore(output, mode="r") as store:
        assert set(store.keys()) == {
            "/benunit",
            "/household",
            "/person",
            "/time_period",
        }
        assert store["time_period"].iloc[0] == "2023"
        assert len(store["household"]) == 4
        assert len(store["person"]) == 6
        assert len(store["benunit"]) == 4
