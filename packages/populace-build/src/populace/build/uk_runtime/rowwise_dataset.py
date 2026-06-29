"""Dataset-level UK row-wise geography assignment.

This module ties the row-wise household geography primitives to the
PolicyEngine-UK single-year table contract. It deliberately works with explicit
frames or H5 files and does not import an incumbent data package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from populace.build.uk_runtime.rowwise_geography import (
    ROWWISE_GEOGRAPHY_COLUMNS,
    RowwiseGeographyAssignment,
    assign_household_geography,
    clone_entity_frame,
    id_multiplier_for_values,
)

PERSON_ID_COLUMNS = (
    "person_id",
    "person_household_id",
    "person_benunit_id",
)
BENUNIT_ID_COLUMNS = ("benunit_id",)
HOUSEHOLD_ID_COLUMNS = ("household_id",)
UK_SINGLE_YEAR_TABLES = ("person", "benunit", "household", "time_period")


@dataclass(frozen=True)
class UKRowwiseDatasetResult:
    """Cloned UK single-year tables and row-wise geography metadata."""

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    assignment: RowwiseGeographyAssignment
    time_period: str
    output_path: Path | None = None

    @property
    def id_multiplier(self) -> int:
        return self.assignment.id_multiplier

    @property
    def n_clones(self) -> int:
        return self.assignment.n_clones


def clone_uk_dataset_tables_with_rowwise_geography(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    crosswalk: pd.DataFrame,
    n_clones: int = 1,
    seed: int = 42,
    time_period: int | str | None = None,
    source_year: int | None = None,
    id_multiplier: int | None = None,
    household_id_column: str = "household_id",
    household_weight_column: str = "household_weight",
    country_column: str | None = None,
    region_column: str = "region",
    clone_index_column: str | None = "clone_index",
    require_all_countries: bool = True,
    require_constituency: bool = True,
    constrain_to_region: bool = True,
    allow_zero_population_distribution: bool = False,
    avoid_constituency_collisions: bool = True,
) -> UKRowwiseDatasetResult:
    """Clone UK person/benunit/household tables and assign row-wise geography."""

    person_frame = person.copy()
    benunit_frame = benunit.copy()
    household_frame = household.copy()
    _validate_input_tables(
        person_frame,
        benunit_frame,
        household_frame,
        household_id_column=household_id_column,
        household_weight_column=household_weight_column,
    )

    if id_multiplier is None:
        id_multiplier = id_multiplier_for_values(
            household_frame[household_id_column],
            person_frame["person_id"],
            person_frame["person_household_id"],
            person_frame["person_benunit_id"],
            benunit_frame["benunit_id"],
        )
    assignment = assign_household_geography(
        household_frame,
        crosswalk,
        n_clones=n_clones,
        seed=seed,
        id_multiplier=id_multiplier,
        household_id_column=household_id_column,
        weight_column=household_weight_column,
        country_column=country_column,
        region_column=region_column,
        source_year=source_year,
        require_all_countries=require_all_countries,
        require_constituency=require_constituency,
        constrain_to_region=constrain_to_region,
        allow_zero_population_distribution=allow_zero_population_distribution,
        avoid_constituency_collisions=avoid_constituency_collisions,
    )

    cloned_person = clone_entity_frame(
        person_frame,
        id_columns=PERSON_ID_COLUMNS,
        n_clones=n_clones,
        id_multiplier=assignment.id_multiplier,
        clone_index_column=clone_index_column,
    )
    cloned_benunit = clone_entity_frame(
        benunit_frame,
        id_columns=BENUNIT_ID_COLUMNS,
        n_clones=n_clones,
        id_multiplier=assignment.id_multiplier,
        clone_index_column=clone_index_column,
    )
    result = UKRowwiseDatasetResult(
        person=cloned_person.reset_index(drop=True),
        benunit=cloned_benunit.reset_index(drop=True),
        household=assignment.household.reset_index(drop=True),
        assignment=assignment,
        time_period=_normalise_time_period(time_period, source_year=source_year),
    )
    validate_uk_rowwise_dataset_tables(result.person, result.benunit, result.household)
    return result


def clone_uk_dataset_with_rowwise_geography(
    dataset: Any | str | Path,
    crosswalk: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    n_clones: int = 1,
    seed: int = 42,
    source_year: int | None = None,
    id_multiplier: int | None = None,
    country_column: str | None = None,
    region_column: str = "region",
    clone_index_column: str | None = "clone_index",
    require_all_countries: bool = True,
    require_constituency: bool = True,
    constrain_to_region: bool = True,
    allow_zero_population_distribution: bool = False,
    avoid_constituency_collisions: bool = True,
) -> UKRowwiseDatasetResult:
    """Clone a UK single-year dataset object or H5 path with row-wise geography."""

    tables = _dataset_tables(dataset, source_year=source_year)
    result = clone_uk_dataset_tables_with_rowwise_geography(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        crosswalk=crosswalk,
        n_clones=n_clones,
        seed=seed,
        time_period=tables["time_period"],
        source_year=source_year,
        id_multiplier=id_multiplier,
        country_column=country_column,
        region_column=region_column,
        clone_index_column=clone_index_column,
        require_all_countries=require_all_countries,
        require_constituency=require_constituency,
        constrain_to_region=constrain_to_region,
        allow_zero_population_distribution=allow_zero_population_distribution,
        avoid_constituency_collisions=avoid_constituency_collisions,
    )
    if output_path is None:
        return result
    path = write_uk_rowwise_dataset(result, output_path)
    return UKRowwiseDatasetResult(
        person=result.person,
        benunit=result.benunit,
        household=result.household,
        assignment=result.assignment,
        time_period=result.time_period,
        output_path=path,
    )


def write_uk_rowwise_dataset(
    result: UKRowwiseDatasetResult,
    output_path: str | Path,
) -> Path:
    """Write cloned UK row-wise tables as a valid single-year H5 dataset."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with pd.HDFStore(path) as store:
        store.put("person", result.person, format="table", data_columns=True)
        store.put("benunit", result.benunit, format="table", data_columns=True)
        store.put("household", result.household, format="table", data_columns=True)
        store.put(
            "time_period",
            pd.Series([result.time_period]),
            format="table",
            data_columns=True,
        )
    return path


def validate_uk_rowwise_dataset_tables(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
) -> None:
    """Validate entity IDs and row-wise household geography columns."""

    _require_columns(person, PERSON_ID_COLUMNS, label="person")
    _require_columns(benunit, BENUNIT_ID_COLUMNS, label="benunit")
    _require_columns(household, HOUSEHOLD_ID_COLUMNS, label="household")
    _require_columns(household, ROWWISE_GEOGRAPHY_COLUMNS, label="household")

    _require_unique(person, "person_id", label="person")
    _require_unique(benunit, "benunit_id", label="benunit")
    _require_unique(household, "household_id", label="household")

    household_ids = set(household["household_id"])
    missing_households = sorted(set(person["person_household_id"]) - household_ids)
    if missing_households:
        raise ValueError(
            "person.person_household_id contains value(s) absent from household: "
            f"{missing_households[:5]}."
        )

    benunit_ids = set(benunit["benunit_id"])
    missing_benunits = sorted(set(person["person_benunit_id"]) - benunit_ids)
    if missing_benunits:
        raise ValueError(
            "person.person_benunit_id contains value(s) absent from benunit: "
            f"{missing_benunits[:5]}."
        )


def _dataset_tables(
    dataset: Any | str | Path,
    *,
    source_year: int | None,
) -> dict[str, pd.DataFrame | str]:
    if isinstance(dataset, str | Path):
        return _read_uk_single_year_h5(dataset)
    missing = [
        name
        for name in ("person", "benunit", "household")
        if not hasattr(dataset, name)
    ]
    if missing:
        raise ValueError(f"dataset is missing table attribute(s): {missing}.")
    time_period = getattr(dataset, "time_period", None)
    return {
        "person": dataset.person.copy(),
        "benunit": dataset.benunit.copy(),
        "household": dataset.household.copy(),
        "time_period": _normalise_time_period(time_period, source_year=source_year),
    }


def _read_uk_single_year_h5(path: str | Path) -> dict[str, pd.DataFrame | str]:
    dataset_path = Path(path)
    if dataset_path.suffix != ".h5":
        raise ValueError("UK single-year dataset path must end with '.h5'.")
    if not dataset_path.exists():
        raise FileNotFoundError(f"UK single-year dataset not found: {dataset_path}.")
    with pd.HDFStore(dataset_path, mode="r") as store:
        keys = {key.lstrip("/") for key in store.keys()}
        missing = sorted(set(UK_SINGLE_YEAR_TABLES) - keys)
        if missing:
            raise ValueError(f"UK single-year dataset is missing table(s): {missing}.")
        time_period = str(store["time_period"].iloc[0])
        return {
            "person": store["person"],
            "benunit": store["benunit"],
            "household": store["household"],
            "time_period": time_period,
        }


def _validate_input_tables(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    household_id_column: str,
    household_weight_column: str,
) -> None:
    _require_columns(person, PERSON_ID_COLUMNS, label="person")
    _require_columns(benunit, BENUNIT_ID_COLUMNS, label="benunit")
    _require_columns(household, [household_id_column], label="household")
    _require_columns(household, [household_weight_column], label="household")
    _require_unique(person, "person_id", label="person")
    _require_unique(benunit, "benunit_id", label="benunit")
    _require_unique(household, household_id_column, label="household")


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} table is missing column(s): {missing}.")


def _require_unique(frame: pd.DataFrame, column: str, *, label: str) -> None:
    if frame[column].isna().any():
        raise ValueError(f"{label}.{column} contains missing values.")
    if frame[column].duplicated().any():
        duplicates = frame.loc[frame[column].duplicated(), column].unique()
        raise ValueError(
            f"{label}.{column} must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )


def _normalise_time_period(
    value: int | str | None,
    *,
    source_year: int | None,
) -> str:
    if value is None:
        if source_year is None:
            raise ValueError("time_period or source_year is required.")
        value = source_year
    return str(value)
