"""UK SPI support rows for enhanced-FRS style imputations.

The enhanced FRS pipeline creates a zero-weight FRS copy, fills that copy with
SPI-trained income imputations, and lets calibration upweight those synthetic
high-income rows where they help fit SPI targets. These helpers keep that
structural step in Populace while preserving source-household lineage for
row-wise local geography.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace.build.uk_runtime.rowwise_geography import id_multiplier_for_values

BASE_FRS_SUPPORT_CHANNEL = "frs"
SPI_SYNTHETIC_SUPPORT_CHANNEL = "spi"
UK_SPI_SUPPORT_STAGE_NAME = "spi_support_channel"
DEFAULT_SPI_SUPPORT_HOUSEHOLDS = 10_000
HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN = "household_is_spi_synthetic"

SPI_INCOME_COMPONENT_COLUMNS = (
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
)

# Mirrors the eFRS SPI-trained first-stage QRF output surface. Gift Aid and
# qualifying investment gifts are relief variables, not income components, but
# they need to be drawn jointly with high-income SPI rows.
SPI_INCOME_IMPUTATION_COLUMNS = SPI_INCOME_COMPONENT_COLUMNS + (
    "gift_aid",
    "charitable_investment_gifts",
)

FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS = (
    "age",
    "gender",
    "region",
    *SPI_INCOME_COMPONENT_COLUMNS,
)

# Mirrors the eFRS second-stage FRS-only QRF output surface. These fields are
# replaced on SPI support rows so high-income synthetic rows do not retain a
# random middle-income FRS donor's benefit receipt or pension behavior.
FRS_ONLY_SPI_FILL_PERSON_COLUMNS = (
    "employee_pension_contributions",
    "employer_pension_contributions",
    "personal_pension_contributions",
    "pension_contributions_via_salary_sacrifice",
    "tax_free_savings_income",
    "universal_credit_reported",
    "pension_credit_reported",
    "child_benefit_reported",
    "housing_benefit_reported",
    "income_support_reported",
    "working_tax_credit_reported",
    "child_tax_credit_reported",
    "attendance_allowance_reported",
    "state_pension_reported",
    "dla_sc_reported",
    "dla_m_reported",
    "pip_m_reported",
    "pip_dl_reported",
    "sda_reported",
    "carers_allowance_reported",
    "iidb_reported",
    "afcs_reported",
    "bsp_reported",
    "incapacity_benefit_reported",
    "maternity_allowance_reported",
    "winter_fuel_allowance_reported",
    "council_tax_benefit_reported",
    "jsa_contrib_reported",
    "jsa_income_reported",
    "esa_contrib_reported",
    "esa_income_reported",
)

_PERSON_ID_COLUMNS = (
    "person_id",
    "person_household_id",
    "person_benunit_id",
)
_BENUNIT_ID_COLUMNS = ("benunit_id",)
_HOUSEHOLD_ID_COLUMNS = ("household_id",)


@dataclass(frozen=True)
class UKSPISupportResult:
    """UK entity tables with an enhanced-FRS SPI support channel."""

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    id_multiplier: int
    spi_household_ids: tuple[Any, ...]

    @property
    def n_spi_households(self) -> int:
        return len(self.spi_household_ids)


def create_uk_spi_support_tables(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    spi_household_count: int | None = DEFAULT_SPI_SUPPORT_HOUSEHOLDS,
    seed: int = 42,
    source_year: int | None = None,
    id_multiplier: int | None = None,
) -> UKSPISupportResult:
    """Create a zero-weight SPI support copy from UK single-year tables.

    The base FRS channel keeps all original rows and weights. The SPI support
    channel contains a sampled FRS copy with remapped IDs, zero household
    weights, ``household_is_spi_synthetic=True``, and source-household lineage
    columns that deliberately point back to the original FRS household. Keeping
    that lineage stable prevents long local-geography support summaries from
    counting the SPI copy as additional independent FRS support.
    """

    person_frame = person.copy()
    benunit_frame = benunit.copy()
    household_frame = _prepare_household_lineage(
        household.copy(),
        source_year=source_year,
    )
    _validate_uk_support_inputs(person_frame, benunit_frame, household_frame)
    _reject_metadata_collisions(person_frame, benunit_frame, household_frame)

    if id_multiplier is None:
        id_multiplier = id_multiplier_for_values(
            household_frame["household_id"],
            person_frame["person_id"],
            person_frame["person_household_id"],
            person_frame["person_benunit_id"],
            benunit_frame["benunit_id"],
        )
    elif id_multiplier <= 0:
        raise ValueError("id_multiplier must be positive.")

    selected_household_ids = _sample_spi_household_ids(
        household_frame,
        spi_household_count=spi_household_count,
        seed=seed,
    )
    selected_household_set = set(selected_household_ids)
    selected_person = person_frame[
        person_frame["person_household_id"].isin(selected_household_set)
    ]
    selected_benunit_ids = set(selected_person["person_benunit_id"])
    selected_benunit = benunit_frame[
        benunit_frame["benunit_id"].isin(selected_benunit_ids)
    ]

    base_household = _clone_support_frame(
        household_frame,
        entity="household",
        id_columns=_HOUSEHOLD_ID_COLUMNS,
        channel=BASE_FRS_SUPPORT_CHANNEL,
        clone_index=0,
        id_multiplier=id_multiplier,
    )
    base_household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN] = False

    spi_household = _clone_support_frame(
        household_frame[household_frame["household_id"].isin(selected_household_set)],
        entity="household",
        id_columns=_HOUSEHOLD_ID_COLUMNS,
        channel=SPI_SYNTHETIC_SUPPORT_CHANNEL,
        clone_index=1,
        id_multiplier=id_multiplier,
    )
    spi_household["household_weight"] = 0.0
    spi_household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN] = True

    base_person = _clone_support_frame(
        person_frame,
        entity="person",
        id_columns=_PERSON_ID_COLUMNS,
        channel=BASE_FRS_SUPPORT_CHANNEL,
        clone_index=0,
        id_multiplier=id_multiplier,
    )
    spi_person = _clone_support_frame(
        selected_person,
        entity="person",
        id_columns=_PERSON_ID_COLUMNS,
        channel=SPI_SYNTHETIC_SUPPORT_CHANNEL,
        clone_index=1,
        id_multiplier=id_multiplier,
    )

    base_benunit = _clone_support_frame(
        benunit_frame,
        entity="benunit",
        id_columns=_BENUNIT_ID_COLUMNS,
        channel=BASE_FRS_SUPPORT_CHANNEL,
        clone_index=0,
        id_multiplier=id_multiplier,
    )
    spi_benunit = _clone_support_frame(
        selected_benunit,
        entity="benunit",
        id_columns=_BENUNIT_ID_COLUMNS,
        channel=SPI_SYNTHETIC_SUPPORT_CHANNEL,
        clone_index=1,
        id_multiplier=id_multiplier,
    )

    result = UKSPISupportResult(
        person=pd.concat([base_person, spi_person], ignore_index=True),
        benunit=pd.concat([base_benunit, spi_benunit], ignore_index=True),
        household=pd.concat([base_household, spi_household], ignore_index=True),
        id_multiplier=id_multiplier,
        spi_household_ids=selected_household_ids,
    )
    _validate_uk_support_outputs(result.person, result.benunit, result.household)
    return result


def fill_support_channel_from_source(
    frame: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    entity: str,
    columns: Sequence[str],
    channel: str = SPI_SYNTHETIC_SUPPORT_CHANNEL,
    donor_id_column: str | None = None,
    fill_missing_columns_with: Any = 0.0,
) -> pd.DataFrame:
    """Fill selected columns on one support channel from source-keyed values.

    ``donor`` should contain one row per original source entity ID, such as a
    QRF prediction frame keyed by original ``person_id``. Rows outside
    ``channel`` are left unchanged. Missing target columns are initialized to
    ``fill_missing_columns_with`` before the channel-specific update, matching
    the eFRS treatment of SPI-only variables such as charitable-giving fields.
    """

    entity = _require_entity(entity)
    values = tuple(columns)
    if not values:
        raise ValueError("columns must include at least one column name.")
    invalid_columns = [
        column for column in values if not isinstance(column, str) or not column
    ]
    if invalid_columns:
        raise ValueError("columns must be non-empty strings.")

    donor_id_column = donor_id_column or _entity_id_column(entity)
    missing_donor = sorted({donor_id_column, *values} - set(donor.columns))
    if missing_donor:
        raise ValueError(f"donor is missing column(s): {missing_donor}.")
    if donor[donor_id_column].isna().any():
        raise ValueError(f"donor.{donor_id_column} contains missing values.")
    if donor[donor_id_column].duplicated().any():
        duplicates = donor.loc[donor[donor_id_column].duplicated(), donor_id_column]
        raise ValueError(
            f"donor.{donor_id_column} must be unique; duplicate value(s): "
            f"{list(map(str, duplicates.unique()[:5]))}."
        )

    channel_column = support_channel_column(entity)
    source_id_column = support_source_id_column(entity)
    missing_frame = sorted({channel_column, source_id_column} - set(frame.columns))
    if missing_frame:
        raise ValueError(
            f"frame is missing support metadata column(s): {missing_frame}."
        )

    out = frame.copy()
    for column in values:
        if column not in out.columns:
            out[column] = fill_missing_columns_with

    mask = out[channel_column] == channel
    if not mask.any():
        raise ValueError(f"frame has no rows in support channel {channel!r}.")

    donor_indexed = donor.set_index(donor_id_column, drop=False)
    source_ids = out.loc[mask, source_id_column]
    missing_ids = source_ids[~source_ids.isin(donor_indexed.index)].unique()
    if len(missing_ids):
        raise ValueError(
            "donor is missing source ID value(s) required by the support "
            f"channel: {list(map(str, missing_ids[:5]))}."
        )

    aligned = donor_indexed.loc[source_ids.to_numpy()]
    for column in values:
        out.loc[mask, column] = aligned[column].to_numpy()
    return out


def support_channel_column(entity: str) -> str:
    """Return the entity-prefixed support-channel column name."""

    return f"{_require_entity(entity)}_support_channel"


def support_clone_index_column(entity: str) -> str:
    """Return the entity-prefixed support-clone-index column name."""

    return f"{_require_entity(entity)}_support_clone_index"


def support_source_id_column(entity: str) -> str:
    """Return the entity-prefixed original-ID provenance column name."""

    return f"{_require_entity(entity)}_source_id"


def _clone_support_frame(
    frame: pd.DataFrame,
    *,
    entity: str,
    id_columns: tuple[str, ...],
    channel: str,
    clone_index: int,
    id_multiplier: int,
) -> pd.DataFrame:
    clone = frame.copy()
    source_id = support_source_id_column(entity)
    clone[source_id] = clone[_entity_id_column(entity)].to_numpy()
    clone[support_channel_column(entity)] = channel
    clone[support_clone_index_column(entity)] = clone_index
    for column in id_columns:
        clone[column] = _remap_ids(
            clone[column].to_numpy(),
            clone_index=clone_index,
            id_multiplier=id_multiplier,
        )
    return clone


def _prepare_household_lineage(
    household: pd.DataFrame,
    *,
    source_year: int | None,
) -> pd.DataFrame:
    frame = household.copy()
    if "source_household_id" not in frame.columns:
        frame["source_household_id"] = frame["household_id"]
    if "source_year" not in frame.columns and source_year is not None:
        frame["source_year"] = source_year
    if "source_household_key" not in frame.columns:
        years = frame["source_year"] if "source_year" in frame.columns else None
        frame["source_household_key"] = _source_household_keys(
            years,
            frame["source_household_id"],
            source_year=source_year,
        )
    return frame


def _sample_spi_household_ids(
    household: pd.DataFrame,
    *,
    spi_household_count: int | None,
    seed: int,
) -> tuple[Any, ...]:
    if spi_household_count is None:
        return tuple(household["household_id"].tolist())
    if not isinstance(spi_household_count, int) or spi_household_count <= 0:
        raise ValueError("spi_household_count must be a positive integer or None.")
    if spi_household_count > len(household):
        raise ValueError(
            "spi_household_count cannot exceed the number of household rows."
        )
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    rng = np.random.default_rng(seed)
    selected_positions = set(
        rng.choice(len(household), size=spi_household_count, replace=False).tolist()
    )
    selected = household.iloc[
        [index for index in range(len(household)) if index in selected_positions]
    ]
    return tuple(selected["household_id"].tolist())


def _validate_uk_support_inputs(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
) -> None:
    _require_columns(person, _PERSON_ID_COLUMNS, label="person")
    _require_columns(benunit, _BENUNIT_ID_COLUMNS, label="benunit")
    _require_columns(
        household,
        (*_HOUSEHOLD_ID_COLUMNS, "household_weight"),
        label="household",
    )
    _require_unique(person, "person_id", label="person")
    _require_unique(benunit, "benunit_id", label="benunit")
    _require_unique(household, "household_id", label="household")

    weights = household["household_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("household.household_weight must be finite and non-negative.")

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


def _validate_uk_support_outputs(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
) -> None:
    _validate_uk_support_inputs(person, benunit, household)
    if not household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN].isin([True, False]).all():
        raise ValueError(
            f"{HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN} must contain boolean values."
        )


def _reject_metadata_collisions(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
) -> None:
    tables = {
        "person": person,
        "benunit": benunit,
        "household": household,
    }
    expected = {
        entity: {
            support_channel_column(entity),
            support_clone_index_column(entity),
            support_source_id_column(entity),
        }
        for entity in tables
    }
    expected["household"].add(HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN)
    collisions = {
        entity: sorted(columns & set(tables[entity].columns))
        for entity, columns in expected.items()
        if columns & set(tables[entity].columns)
    }
    if collisions:
        raise ValueError(
            "UK SPI support metadata column(s) already exist: "
            f"{collisions}. The stage should run exactly once."
        )


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
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


def _source_household_keys(
    years: pd.Series | None,
    source_household_ids: pd.Series,
    *,
    source_year: int | None,
) -> list[str]:
    year_values = [source_year] * len(source_household_ids) if years is None else years
    keys = []
    for year, household_id in zip(year_values, source_household_ids, strict=True):
        if year is None or pd.isna(year):
            keys.append(str(household_id))
        else:
            keys.append(f"{year}:{household_id}")
    return keys


def _remap_ids(
    ids: Sequence[Any],
    *,
    clone_index: int,
    id_multiplier: int,
) -> np.ndarray:
    values = pd.to_numeric(pd.Series(ids), errors="raise").astype("int64").to_numpy()
    if clone_index == 0:
        return values.copy()
    return values + clone_index * id_multiplier


def _entity_id_column(entity: str) -> str:
    entity = _require_entity(entity)
    if entity == "household":
        return "household_id"
    return f"{entity}_id"


def _require_entity(entity: str) -> str:
    if entity not in {"person", "benunit", "household"}:
        raise ValueError("entity must be one of: 'person', 'benunit', 'household'.")
    return entity


__all__ = [
    "BASE_FRS_SUPPORT_CHANNEL",
    "DEFAULT_SPI_SUPPORT_HOUSEHOLDS",
    "FRS_ONLY_SPI_FILL_PERSON_COLUMNS",
    "FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS",
    "HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN",
    "SPI_SYNTHETIC_SUPPORT_CHANNEL",
    "SPI_INCOME_COMPONENT_COLUMNS",
    "SPI_INCOME_IMPUTATION_COLUMNS",
    "UKSPISupportResult",
    "UK_SPI_SUPPORT_STAGE_NAME",
    "create_uk_spi_support_tables",
    "fill_support_channel_from_source",
    "support_channel_column",
    "support_clone_index_column",
    "support_source_id_column",
]
