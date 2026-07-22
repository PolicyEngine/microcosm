"""Row-wise UK geography assignment primitives.

These helpers own the Populace-side equivalent of the US "assign the finest
available geography to each cloned record" workflow. They deliberately accept
explicit crosswalk and household frames rather than importing a data package or
downloading source files internally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ROWWISE_GEOGRAPHY_COLUMNS = (
    "oa_code",
    "lsoa_code",
    "msoa_code",
    "la_code_oa",
    "constituency_code_oa",
    "region_code_oa",
)

CROSSWALK_COLUMNS = (
    "oa_code",
    "lsoa_code",
    "msoa_code",
    "la_code",
    "constituency_code",
    "region_code",
    "country",
    "population",
)

AREA_TYPE_TO_CROSSWALK_COLUMN = {
    "oa": "oa_code",
    "lsoa": "lsoa_code",
    "msoa": "msoa_code",
    "la": "la_code",
    "local_authority": "la_code",
    "constituency": "constituency_code",
    "region": "region_code",
}

FRS_REGION_TO_COUNTRY = {
    "NORTH_EAST": "England",
    "NORTH_WEST": "England",
    "YORKSHIRE": "England",
    "EAST_MIDLANDS": "England",
    "WEST_MIDLANDS": "England",
    "EAST_OF_ENGLAND": "England",
    "LONDON": "England",
    "SOUTH_EAST": "England",
    "SOUTH_WEST": "England",
    "WALES": "Wales",
    "SCOTLAND": "Scotland",
    "NORTHERN_IRELAND": "Northern Ireland",
}

FRS_REGION_TO_REGION_CODE = {
    "NORTH_EAST": "E12000001",
    "NORTH_WEST": "E12000002",
    "YORKSHIRE": "E12000003",
    "EAST_MIDLANDS": "E12000004",
    "WEST_MIDLANDS": "E12000005",
    "EAST_OF_ENGLAND": "E12000006",
    "LONDON": "E12000007",
    "SOUTH_EAST": "E12000008",
    "SOUTH_WEST": "E12000009",
    "WALES": "W99999999",
    "SCOTLAND": "S99999999",
    "NORTHERN_IRELAND": "N99999999",
}

COUNTRY_LABELS = {
    "ENGLAND": "England",
    "WALES": "Wales",
    "SCOTLAND": "Scotland",
    "NORTHERN_IRELAND": "Northern Ireland",
    "NORTHERN IRELAND": "Northern Ireland",
}

FRS_COUNTRY_CODE_TO_COUNTRY = {
    1: "England",
    2: "Wales",
    3: "Scotland",
    4: "Northern Ireland",
}


@dataclass(frozen=True)
class RowwiseGeographyAssignment:
    """Household geography assignment output and ID remapping metadata."""

    household: pd.DataFrame
    id_multiplier: int
    original_household_ids: tuple[Any, ...]
    n_clones: int


def prepare_geography_crosswalk(
    crosswalk: pd.DataFrame,
    *,
    require_constituency: bool = True,
) -> pd.DataFrame:
    """Validate and canonicalize a UK geography crosswalk.

    The input is expected to be one row per finest geography record. For
    Northern Ireland this can be a Data Zone row treated as the OA-equivalent.
    """

    missing = sorted(set(CROSSWALK_COLUMNS) - set(crosswalk.columns))
    if missing:
        raise ValueError(f"crosswalk is missing column(s): {missing}.")

    frame = crosswalk.loc[:, CROSSWALK_COLUMNS].copy()
    for column in CROSSWALK_COLUMNS:
        if column == "population":
            continue
        frame[column] = _normalise_string_column(
            frame[column],
            column=column,
            allow_blank=column == "constituency_code" and not require_constituency,
        )

    frame["country"] = _normalise_country_series(frame["country"], column="country")
    duplicated = frame["oa_code"][frame["oa_code"].duplicated()].unique()
    if len(duplicated):
        raise ValueError(
            "crosswalk oa_code values must be unique; duplicate value(s): "
            f"{list(map(str, duplicated[:5]))}."
        )
    if require_constituency and (frame["constituency_code"] == "").any():
        countries = sorted(
            frame.loc[frame["constituency_code"] == "", "country"].unique()
        )
        raise ValueError(
            "crosswalk has blank constituency_code values for: " + ", ".join(countries)
        )

    population = pd.to_numeric(frame["population"], errors="coerce")
    if population.isna().any():
        raise ValueError("crosswalk population contains non-numeric values.")
    values = population.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("crosswalk population must be finite and non-negative.")
    frame["population"] = values
    return frame


def geography_coverage_summary(
    crosswalk: pd.DataFrame,
    area_codes_by_type: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Summarize whether the crosswalk covers requested target areas."""

    frame = prepare_geography_crosswalk(crosswalk, require_constituency=False)
    rows: list[dict[str, Any]] = []
    sampleable_by_column = _sampleable_codes_by_column(frame)
    for area_type, area_codes in area_codes_by_type.items():
        column = _area_type_column(area_type)
        codes = _unique_string_tuple(area_codes, label=f"{area_type} area codes")
        present = set(frame[column][frame[column] != ""])
        sampleable = sampleable_by_column[column]
        missing = tuple(code for code in codes if code not in present)
        unsampleable = tuple(
            code for code in codes if code in present and code not in sampleable
        )
        sampleable_count = len(codes) - len(missing) - len(unsampleable)
        rows.append(
            {
                "area_type": area_type,
                "crosswalk_column": column,
                "target_areas": len(codes),
                "covered_areas": len(codes) - len(missing),
                "missing_areas": len(missing),
                "coverage_share": (
                    1.0 if len(codes) == 0 else (len(codes) - len(missing)) / len(codes)
                ),
                "missing_area_codes": missing,
                "sampleable_areas": sampleable_count,
                "unsampleable_areas": len(unsampleable),
                "sampleable_share": (
                    1.0 if len(codes) == 0 else sampleable_count / len(codes)
                ),
                "unsampleable_area_codes": unsampleable,
            }
        )
    return pd.DataFrame(rows)


def validate_geography_coverage(
    crosswalk: pd.DataFrame,
    *,
    required_countries: Sequence[str] | None = None,
    area_codes_by_type: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Validate crosswalk country and target-area coverage.

    Returns a coverage summary when ``area_codes_by_type`` is supplied.
    """

    frame = prepare_geography_crosswalk(crosswalk, require_constituency=False)
    failures: list[str] = []
    if required_countries is not None:
        countries = _unique_string_tuple(
            [_normalise_country(country) for country in required_countries],
            label="required_countries",
        )
        present = set(frame["country"])
        missing = [country for country in countries if country not in present]
        if missing:
            failures.append("missing countries: " + ", ".join(missing))

    summary = pd.DataFrame()
    if area_codes_by_type is not None:
        summary = geography_coverage_summary(frame, area_codes_by_type)
        missing_rows = summary[summary["missing_areas"] > 0]
        for row in missing_rows.itertuples(index=False):
            codes = ", ".join(map(str, row.missing_area_codes[:5]))
            failures.append(
                f"{row.area_type} missing {row.missing_areas} area(s): {codes}"
            )
        unsampleable_rows = summary[summary["unsampleable_areas"] > 0]
        for row in unsampleable_rows.itertuples(index=False):
            codes = ", ".join(map(str, row.unsampleable_area_codes[:5]))
            failures.append(
                f"{row.area_type} has {row.unsampleable_areas} unsampleable "
                f"area(s): {codes}"
            )

    if failures:
        raise ValueError("geography coverage is incomplete; " + "; ".join(failures))
    return summary


def assign_household_geography(
    household: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    n_clones: int = 1,
    seed: int = 42,
    id_multiplier: int | None = None,
    household_id_column: str = "household_id",
    weight_column: str = "household_weight",
    country_column: str | None = None,
    region_column: str = "region",
    source_year: int | None = None,
    require_all_countries: bool = True,
    require_constituency: bool = True,
    constrain_to_region: bool = True,
    allow_zero_population_distribution: bool = False,
    avoid_constituency_collisions: bool = True,
    max_collision_retries: int = 50,
) -> RowwiseGeographyAssignment:
    """Clone households and assign a population-weighted finest geography.

    Each clone keeps source-household lineage and receives one row-wise
    geography assignment constrained to the household country. Household
    weights are divided by ``n_clones`` so aggregate mass is preserved.

    Pass ``id_multiplier`` from :func:`id_multiplier_for_values` when cloning
    linked person or benefit-unit frames so household, person, and foreign-key
    IDs all share a collision-free offset.
    """

    if not isinstance(n_clones, int) or n_clones <= 0:
        raise ValueError("n_clones must be a positive integer.")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if not isinstance(max_collision_retries, int) or max_collision_retries < 0:
        raise ValueError("max_collision_retries must be a non-negative integer.")

    households = _prepare_households(
        household,
        household_id_column=household_id_column,
        weight_column=weight_column,
        country_column=country_column,
        region_column=region_column,
        source_year=source_year,
        constrain_to_region=constrain_to_region,
    )
    geography = prepare_geography_crosswalk(
        crosswalk,
        require_constituency=require_constituency,
    )
    countries = households["_assignment_country"].to_numpy(dtype=object)
    region_codes = households["_assignment_region_code"].to_numpy(dtype=object)
    use_region_constraint = constrain_to_region and bool(
        (households["_assignment_region_code"] != "").any()
    )
    distributions = _assignment_distributions(
        geography,
        use_region=use_region_constraint,
        allow_zero_population=allow_zero_population_distribution,
    )
    household_keys = _assignment_keys(
        countries,
        region_codes,
        use_region=use_region_constraint,
    )
    missing_keys = sorted(set(household_keys) - set(distributions))
    missing_countries = sorted({country for country, _region_code in missing_keys})
    if missing_countries and require_all_countries:
        raise ValueError(
            "No geography distribution available for: " + ", ".join(missing_countries)
        )

    rng = np.random.default_rng(seed)
    n_households = len(households)
    if id_multiplier is None:
        id_multiplier = id_multiplier_for_values(households["household_id"])
    elif id_multiplier <= 0:
        raise ValueError("id_multiplier must be positive.")
    original_ids = tuple(households["household_id"].tolist())
    assigned_constituencies = np.full((n_clones, n_households), "", dtype=object)
    clone_frames: list[pd.DataFrame] = []

    for clone_index in range(n_clones):
        clone = households.drop(
            columns=["_assignment_country", "_assignment_region_code"]
        ).copy()
        clone["household_id"] = _remap_ids(
            households["household_id"].to_numpy(),
            clone_index=clone_index,
            id_multiplier=id_multiplier,
        )
        clone["household_weight"] = (
            households["household_weight"].to_numpy(dtype=np.float64) / n_clones
        )
        clone["clone_index"] = clone_index
        for column in ROWWISE_GEOGRAPHY_COLUMNS:
            clone[column] = ""

        for key in sorted(set(household_keys)):
            positions = _positions_for_key(household_keys, key)
            if key not in distributions:
                continue
            sampled = _sample_country_rows(
                distributions[key],
                size=len(positions),
                rng=rng,
            )
            if avoid_constituency_collisions and clone_index > 0:
                sampled = _resample_constituency_collisions(
                    sampled,
                    distributions[key],
                    positions=positions,
                    assigned_constituencies=assigned_constituencies,
                    clone_index=clone_index,
                    rng=rng,
                    max_retries=max_collision_retries,
                )
            _write_assignment_columns(clone, positions, sampled)
            assigned_constituencies[clone_index, positions] = sampled[
                "constituency_code"
            ].to_numpy(dtype=object)

        clone_frames.append(clone)

    result = pd.concat(clone_frames, ignore_index=True)
    if result["household_id"].duplicated().any():
        duplicates = result.loc[result["household_id"].duplicated(), "household_id"]
        raise ValueError(
            "remapped 'household_id' values are not unique; id_multiplier is "
            "too small for household cloning. Duplicate value(s): "
            f"{list(map(str, duplicates.unique()[:5]))}."
        )
    return RowwiseGeographyAssignment(
        household=result,
        id_multiplier=id_multiplier,
        original_household_ids=original_ids,
        n_clones=n_clones,
    )


def expected_uk_rowwise_area_support(
    household: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    n_clones: int = 1,
    household_id_column: str = "household_id",
    weight_column: str = "household_weight",
    country_column: str | None = None,
    region_column: str = "region",
    source_year: int | None = None,
    require_all_countries: bool = True,
    require_constituency: bool = True,
    constrain_to_region: bool = True,
    allow_zero_population_distribution: bool = False,
) -> pd.DataFrame:
    """Exact expected per-area row support of :func:`assign_household_geography`.

    Reuses the sampler's own household preparation and per-(country, region)
    sampling distributions, so this expectation cannot drift from the
    distributions it describes. It is the **collision-free** expectation:
    with ``avoid_constituency_collisions`` (the sampler's default) realized
    support can differ substantially whenever ``n_clones`` is comparable to
    the number of sampleable constituencies in a group, because resampling
    pushes same-source clones into distinct constituencies. For exact
    planning use the dry-run's realized assignment, which runs the real
    sampler at the build seed.

    Returns a long frame with ``area_type`` (``constituency``/``la``),
    ``area_code``, and ``expected_rows``, covering every sampleable area of
    the crosswalk for the represented groups (including zero-support areas).
    """

    if not isinstance(n_clones, int) or n_clones <= 0:
        raise ValueError("n_clones must be a positive integer.")
    households = _prepare_households(
        household,
        household_id_column=household_id_column,
        weight_column=weight_column,
        country_column=country_column,
        region_column=region_column,
        source_year=source_year,
        constrain_to_region=constrain_to_region,
    )
    geography = prepare_geography_crosswalk(
        crosswalk,
        require_constituency=require_constituency,
    )
    countries = households["_assignment_country"].to_numpy(dtype=object)
    region_codes = households["_assignment_region_code"].to_numpy(dtype=object)
    use_region_constraint = constrain_to_region and bool(
        (households["_assignment_region_code"] != "").any()
    )
    distributions = _assignment_distributions(
        geography,
        use_region=use_region_constraint,
        allow_zero_population=allow_zero_population_distribution,
    )
    household_keys = _assignment_keys(
        countries,
        region_codes,
        use_region=use_region_constraint,
    )
    missing_keys = sorted(set(household_keys) - set(distributions))
    missing_countries = sorted({country for country, _region_code in missing_keys})
    if missing_countries and require_all_countries:
        raise ValueError(
            "No geography distribution available for: " + ", ".join(missing_countries)
        )

    expected: dict[tuple[str, str], float] = {}
    key_counts: dict[tuple[str, str], int] = {}
    for key in household_keys:
        key_counts[key] = key_counts.get(key, 0) + 1
    for key, group_size in sorted(key_counts.items()):
        if key not in distributions:
            continue
        distribution = distributions[key]
        rows = (
            distribution["_sample_probability"].to_numpy(dtype=np.float64)
            * group_size
            * n_clones
        )
        for area_column, area_type in (
            ("constituency_code", "constituency"),
            ("la_code", "la"),
        ):
            codes = distribution[area_column].to_numpy(dtype=object)
            for code, value in zip(codes, rows, strict=True):
                code = str(code)
                if not code:
                    continue
                expected[(area_type, code)] = expected.get(
                    (area_type, code), 0.0
                ) + float(value)

    result = pd.DataFrame(
        [
            {
                "area_type": area_type,
                "area_code": area_code,
                "expected_rows": value,
            }
            for (area_type, area_code), value in expected.items()
        ]
    )
    if result.empty:
        return pd.DataFrame(columns=["area_type", "area_code", "expected_rows"])
    return result.sort_values(["area_type", "area_code"], kind="mergesort").reset_index(
        drop=True
    )


def clone_entity_frame(
    frame: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    n_clones: int,
    id_multiplier: int,
    clone_index_column: str | None = None,
) -> pd.DataFrame:
    """Clone an entity frame and remap each ID/foreign-key column."""

    if not isinstance(n_clones, int) or n_clones <= 0:
        raise ValueError("n_clones must be a positive integer.")
    if id_multiplier <= 0:
        raise ValueError("id_multiplier must be positive.")
    columns = _unique_string_tuple(id_columns, label="id_columns")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"entity frame is missing ID column(s): {missing}.")
    unique_input_columns = [column for column in columns if frame[column].is_unique]

    clones: list[pd.DataFrame] = []
    for clone_index in range(n_clones):
        clone = frame.copy()
        for column in columns:
            clone[column] = _remap_ids(
                clone[column].to_numpy(),
                clone_index=clone_index,
                id_multiplier=id_multiplier,
            )
        if clone_index_column is not None:
            clone[clone_index_column] = clone_index
        clones.append(clone)
    result = pd.concat(clones, ignore_index=True)
    for column in unique_input_columns:
        if result[column].duplicated().any():
            duplicates = result.loc[result[column].duplicated(), column].unique()
            raise ValueError(
                f"remapped {column!r} values are not unique; id_multiplier "
                "is too small for linked entity cloning. Duplicate value(s): "
                f"{list(map(str, duplicates[:5]))}."
            )
    return result


def id_multiplier_for_values(*values: Sequence[Any]) -> int:
    """Return a clone ID offset larger than all supplied ID values."""

    if not values:
        raise ValueError("at least one ID value sequence is required.")
    max_id = 0
    for sequence in values:
        numeric = pd.to_numeric(pd.Series(sequence), errors="raise").astype("int64")
        if len(numeric):
            max_id = max(max_id, int(numeric.abs().max()))
    return 10 ** max(1, len(str(max_id)))


def _prepare_households(
    household: pd.DataFrame,
    *,
    household_id_column: str,
    weight_column: str,
    country_column: str | None,
    region_column: str,
    source_year: int | None,
    constrain_to_region: bool,
) -> pd.DataFrame:
    missing = [
        column
        for column in (household_id_column, weight_column)
        if column not in household.columns
    ]
    if missing:
        raise ValueError(f"household frame is missing column(s): {missing}.")
    if country_column is None and region_column not in household.columns:
        raise ValueError(
            f"household frame must include {region_column!r} when country_column "
            "is not supplied."
        )
    if country_column is not None and country_column not in household.columns:
        raise ValueError(f"household frame is missing {country_column!r}.")

    frame = household.copy()
    if household_id_column != "household_id":
        frame = frame.rename(columns={household_id_column: "household_id"})
    if weight_column != "household_weight":
        frame = frame.rename(columns={weight_column: "household_weight"})
    if frame["household_id"].isna().any():
        raise ValueError("household_id contains missing values.")
    if frame["household_id"].duplicated().any():
        duplicates = frame.loc[frame["household_id"].duplicated(), "household_id"]
        raise ValueError(
            "household_id must be unique before cloning; duplicate value(s): "
            f"{list(map(str, duplicates.unique()[:5]))}."
        )
    weights = pd.to_numeric(frame["household_weight"], errors="coerce")
    if weights.isna().any():
        raise ValueError("household_weight contains non-numeric values.")
    weight_values = weights.to_numpy(dtype=np.float64)
    if not np.isfinite(weight_values).all() or (weight_values < 0).any():
        raise ValueError("household_weight must be finite and non-negative.")
    frame["household_weight"] = weight_values

    if country_column is not None:
        frame["_assignment_country"] = _normalise_country_series(
            frame[country_column],
            column=country_column,
        )
    else:
        frame["_assignment_country"] = _countries_from_region(
            frame[region_column],
            column=region_column,
        )
    if constrain_to_region and region_column in frame.columns:
        frame["_assignment_region_code"] = _region_codes_from_region(
            frame[region_column],
            column=region_column,
        )
    else:
        frame["_assignment_region_code"] = ""

    if "source_household_id" not in frame.columns:
        frame["source_household_id"] = frame["household_id"]
    if "source_year" not in frame.columns and source_year is not None:
        frame["source_year"] = source_year
    if "source_household_key" not in frame.columns:
        frame["source_household_key"] = _source_household_keys(
            frame["source_year"] if "source_year" in frame.columns else None,
            frame["source_household_id"],
            source_year=source_year,
        )
    return frame.reset_index(drop=True)


def _assignment_distributions(
    crosswalk: pd.DataFrame,
    *,
    use_region: bool,
    allow_zero_population: bool,
) -> dict[tuple[str, str], pd.DataFrame]:
    distributions: dict[tuple[str, str], pd.DataFrame] = {}
    group_columns = ["country", "region_code"] if use_region else "country"
    for group_key, subset in crosswalk.groupby(group_columns, sort=True):
        weights = subset["population"].to_numpy(dtype=np.float64)
        if weights.sum() == 0:
            if not allow_zero_population:
                raise ValueError(
                    "geography distribution has zero total population for "
                    f"{_format_distribution_key(group_key)}."
                )
            probabilities = np.ones(len(subset), dtype=np.float64) / len(subset)
        else:
            probabilities = weights / weights.sum()
        distribution = subset.copy().reset_index(drop=True)
        distribution["_sample_probability"] = probabilities
        country, region_code = _distribution_key_tuple(group_key, use_region=use_region)
        distributions[(country, region_code)] = distribution
    return distributions


def _sampleable_codes_by_column(crosswalk: pd.DataFrame) -> dict[str, set[str]]:
    result = {column: set() for column in set(AREA_TYPE_TO_CROSSWALK_COLUMN.values())}
    sampleable = crosswalk[crosswalk["population"] > 0]
    for column in result:
        result[column].update(sampleable[column][sampleable[column] != ""])
    return result


def _assignment_keys(
    countries: Sequence[Any],
    region_codes: Sequence[Any],
    *,
    use_region: bool,
) -> tuple[tuple[Any, Any], ...]:
    if use_region:
        return tuple(zip(countries, region_codes, strict=True))
    return tuple((country, "") for country in countries)


def _positions_for_key(
    keys: Sequence[tuple[Any, Any]],
    key: tuple[Any, Any],
) -> np.ndarray:
    return np.asarray(
        [index for index, value in enumerate(keys) if value == key],
        dtype=np.int64,
    )


def _distribution_key_tuple(
    group_key: Any,
    *,
    use_region: bool,
) -> tuple[str, str]:
    if use_region:
        country, region_code = group_key
        return str(country), str(region_code)
    return str(group_key), ""


def _format_distribution_key(group_key: Any) -> str:
    if isinstance(group_key, tuple):
        return " / ".join(map(str, group_key))
    return str(group_key)


def _sample_country_rows(
    distribution: pd.DataFrame,
    *,
    size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    indices = rng.choice(
        len(distribution),
        size=size,
        p=distribution["_sample_probability"].to_numpy(dtype=np.float64),
    )
    return distribution.iloc[indices].reset_index(drop=True)


def _resample_constituency_collisions(
    sampled: pd.DataFrame,
    distribution: pd.DataFrame,
    *,
    positions: np.ndarray,
    assigned_constituencies: np.ndarray,
    clone_index: int,
    rng: np.random.Generator,
    max_retries: int,
) -> pd.DataFrame:
    result = sampled.copy()
    probabilities = distribution["_sample_probability"].to_numpy(dtype=np.float64)
    for _ in range(max_retries):
        collisions = np.zeros(len(result), dtype=bool)
        current = result["constituency_code"].to_numpy(dtype=object)
        for previous_clone in range(clone_index):
            collisions |= current == assigned_constituencies[previous_clone, positions]
        if not collisions.any():
            break
        new_indices = rng.choice(
            len(distribution),
            size=int(collisions.sum()),
            p=probabilities,
        )
        result.loc[collisions, :] = distribution.iloc[new_indices].to_numpy()
    return result.reset_index(drop=True)


def _write_assignment_columns(
    household: pd.DataFrame,
    positions: np.ndarray,
    sampled: pd.DataFrame,
) -> None:
    mapping = {
        "oa_code": "oa_code",
        "lsoa_code": "lsoa_code",
        "msoa_code": "msoa_code",
        "la_code": "la_code_oa",
        "constituency_code": "constituency_code_oa",
        "region_code": "region_code_oa",
    }
    for source, target in mapping.items():
        household.iloc[positions, household.columns.get_loc(target)] = sampled[
            source
        ].to_numpy(dtype=object)


def _normalise_string_column(
    values: pd.Series,
    *,
    column: str,
    allow_blank: bool = False,
) -> pd.Series:
    if values.isna().any() and not allow_blank:
        raise ValueError(f"crosswalk {column!r} contains missing values.")
    strings = values.fillna("").astype(str).str.strip()
    if not allow_blank and (strings == "").any():
        raise ValueError(f"crosswalk {column!r} contains blank values.")
    return strings


def _normalise_country_series(values: pd.Series, *, column: str) -> pd.Series:
    countries = values.map(_normalise_country)
    if countries.isna().any():
        bad = values[countries.isna()].astype(str).unique()
        raise ValueError(
            f"{column!r} contains unrecognised country value(s): "
            f"{list(map(str, bad[:5]))}."
        )
    return countries


def _normalise_country(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return FRS_COUNTRY_CODE_TO_COUNTRY.get(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return FRS_COUNTRY_CODE_TO_COUNTRY.get(int(value))
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return FRS_COUNTRY_CODE_TO_COUNTRY.get(int(text))
    key = text.upper().replace("-", "_")
    return COUNTRY_LABELS.get(key, text if text in COUNTRY_LABELS.values() else None)


def _countries_from_region(values: pd.Series, *, column: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{column!r} contains missing values.")
    keys = values.astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)
    countries = keys.map(FRS_REGION_TO_COUNTRY)
    if countries.isna().any():
        bad = values[countries.isna()].astype(str).unique()
        raise ValueError(
            f"{column!r} contains unrecognised UK region value(s): "
            f"{list(map(str, bad[:5]))}."
        )
    return countries


def _region_codes_from_region(values: pd.Series, *, column: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{column!r} contains missing values.")
    keys = values.astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)
    region_codes = keys.map(FRS_REGION_TO_REGION_CODE)
    if region_codes.isna().any():
        bad = values[region_codes.isna()].astype(str).unique()
        raise ValueError(
            f"{column!r} contains unrecognised UK region value(s): "
            f"{list(map(str, bad[:5]))}."
        )
    return region_codes


def _area_type_column(area_type: str) -> str:
    key = str(area_type).strip().lower()
    if key not in AREA_TYPE_TO_CROSSWALK_COLUMN:
        expected = ", ".join(sorted(AREA_TYPE_TO_CROSSWALK_COLUMN))
        raise ValueError(
            f"Unknown area_type {area_type!r}; expected one of {expected}."
        )
    return AREA_TYPE_TO_CROSSWALK_COLUMN[key]


def _unique_string_tuple(values: Sequence[Any], *, label: str) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if any(value == "" for value in result):
        raise ValueError(f"{label} must not contain blank values.")
    duplicates = sorted({value for value in result if result.count(value) > 1})
    if duplicates:
        raise ValueError(f"{label} must be unique; duplicate value(s): {duplicates}.")
    return result


def _remap_ids(
    ids: Sequence[Any],
    *,
    clone_index: int,
    id_multiplier: int,
) -> np.ndarray:
    values = pd.to_numeric(pd.Series(ids), errors="raise").astype("int64").to_numpy()
    if clone_index == 0:
        return values.copy()
    offset = clone_index * id_multiplier
    max_value = int(values.max()) if len(values) else 0
    if max_value > 0 and offset > np.iinfo(np.int64).max - max_value:
        raise ValueError(
            "clone ID remapping would overflow int64: clone_index "
            f"{clone_index} x id_multiplier {id_multiplier} + max id "
            f"{max_value} exceeds the representable range."
        )
    return values + offset


def _source_household_keys(
    years: pd.Series | None,
    source_household_ids: pd.Series,
    *,
    source_year: int | None,
) -> list[str]:
    if years is None:
        year_values = [source_year] * len(source_household_ids)
    else:
        year_values = years.tolist()
    keys = []
    for year, household_id in zip(year_values, source_household_ids, strict=True):
        if year is None or pd.isna(year):
            keys.append(str(household_id))
        else:
            keys.append(f"{year}:{household_id}")
    return keys
