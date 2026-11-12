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

    with pytest.raises(ValueError, match="person_household_id") as caught:
        validate_uk_rowwise_dataset_tables(person, result.benunit, result.household)
    # The refusals count rows and name no id (microcosm#931).
    assert str(caught.value) == (
        "person.person_household_id names a household absent from household on "
        "1 person row(s)."
    )
    person = result.person.copy()
    person.loc[0, "person_benunit_id"] = 987_654
    with pytest.raises(ValueError, match="person_benunit_id") as caught:
        validate_uk_rowwise_dataset_tables(person, result.benunit, result.household)
    assert "987654" not in str(caught.value)
    household = result.household.copy()
    household.loc[1, "household_id"] = int(household["household_id"].iloc[0])
    with pytest.raises(ValueError, match="must be unique") as caught:
        validate_uk_rowwise_dataset_tables(result.person, result.benunit, household)
    assert str(caught.value) == (
        "household.household_id must be unique; 1 row(s) repeat an earlier row's value."
    )


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


# --- the row-wise lane on a clone-first spine (microcosm#931) ----------------------


def _clone_first_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The toy tables as a clone-first spine at one clone: the clone indices
    and household source columns its geographic_expand stage writes."""

    from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column

    tables = {
        "person": person_frame(),
        "benunit": benunit_frame(),
        "household": household_frame(),
    }
    for entity, table in tables.items():
        table[ladder_clone_index_column(entity)] = 0
    household = tables["household"]
    household["source_household_id"] = household["household_id"]
    household["source_year"] = 2023
    household["source_household_key"] = [
        f"2023:{source}" for source in household["source_household_id"]
    ]
    return tables["person"], tables["benunit"], household


@pytest.mark.parametrize("n_clones", [1, 2])
def test_the_rowwise_pool_refuses_a_clone_first_spine_at_any_clone_count(
    n_clones: int,
) -> None:
    """The lane's expansion clones a spine itself, so it refuses a spine that
    carries geographic_expand's clone index, at one clone as at more, pointing
    at the dense graph, which builds on such a spine as it stands; never the
    overflow refusal, and no later group is promised (microcosm#931)."""

    from microcosm.build.uk_runtime.rowwise_dataset import expand_uk_geographic_pool

    person, benunit, household = _clone_first_tables()
    with pytest.raises(ValueError) as caught:
        expand_uk_geographic_pool(
            person=person,
            benunit=benunit,
            household=household,
            n_clones=n_clones,
            source_year=2023,
        )
    message = str(caught.value)
    assert message == (
        "input person table already carries the reserved in-memory clone column "
        "'person_clone_index'; the ladder clone owns that name. A clone-first "
        "spine, whose geographic_expand stage writes it, carries its geographic "
        "clones already: the dense graph builds on a geography-first spine as it "
        "stands (microcosm#931), and this clone refuses one at any n_clones."
    )
    assert "group (c)" not in message
    assert "overflow" not in message


def test_the_rowwise_lineage_defaults_only_without_the_clone_index() -> None:
    """The lane defaults a household's source columns from its own ids only on
    a spine without the geographic clone index; on a clone-first spine its
    geographic_expand stage writes them, and a table carrying the index
    without them is refused by column name (microcosm#931)."""

    from microcosm.build.uk_runtime.rowwise_dataset import _attach_source_lineage

    household = pd.DataFrame({"household_id": [1, 2]})
    defaulted = _attach_source_lineage(household.copy(), source_year=2023)
    assert defaulted["source_household_id"].tolist() == [1, 2]
    assert defaulted["source_year"].tolist() == [2023, 2023]
    assert defaulted["source_household_key"].tolist() == ["2023:1", "2023:2"]

    cloned = household.assign(household_clone_index=[0, 1], source_household_id=[1, 1])
    with pytest.raises(ValueError) as caught:
        _attach_source_lineage(cloned.copy(), source_year=2023)
    assert str(caught.value) == (
        "The household table carries 'household_clone_index' but not "
        "['source_year', 'source_household_key']: on the clone-first spine "
        "geographic_expand writes every household's source columns, and a "
        "clone's own ids are never taken for its source (microcosm#931)."
    )
    complete = cloned.assign(
        source_year=[2023, 2023], source_household_key=["2023:1", "2023:1"]
    )
    pd.testing.assert_frame_equal(
        _attach_source_lineage(complete.copy(), source_year=2023), complete
    )
