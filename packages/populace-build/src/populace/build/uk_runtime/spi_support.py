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

from populace.build.uk_runtime.hmrc_income import (
    HMRC_SPI_ASSESSABLE_INCOME_COLUMN,
)
from populace.build.uk_runtime.rowwise_geography import id_multiplier_for_values
from populace.frame import (
    EntitySchema,
    Frame,
    MassChange,
    MassChangeRecord,
    WeightKind,
    Weights,
)

BASE_FRS_SUPPORT_CHANNEL = "frs"
SPI_SYNTHETIC_SUPPORT_CHANNEL = "spi"
UK_SPI_SUPPORT_STAGE_NAME = "spi_support_channel"
DEFAULT_SPI_SUPPORT_HOUSEHOLDS = 10_000
DEFAULT_SPI_PRIOR_MASS_SHARE = 0.5
HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN = "household_is_spi_synthetic"
SPI_REPLACEMENT_STRATA_COLUMNS = (
    "clone_index",
    "household_is_capital_gains_clone",
    "region",
)
SPI_PRIOR_MASS_CHANGE_REASON = (
    "Allocate 50% of certified UK national household prior mass to the "
    "rebuilt 2022-23 SPI support channel; total national mass is conserved."
)

SPI_INCOME_COMPONENT_COLUMNS = (
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
    "other_investment_income",
)

# Mirrors the eFRS SPI-trained first-stage QRF output surface. Gift Aid and
# qualifying investment gifts are relief variables, not income components, but
# they need to be drawn jointly with high-income SPI rows.
SPI_INCOME_IMPUTATION_COLUMNS = SPI_INCOME_COMPONENT_COLUMNS + (
    "gift_aid",
    "charitable_investment_gifts",
)
SPI_HMRC_AUXILIARY_COLUMNS = (HMRC_SPI_ASSESSABLE_INCOME_COLUMN,)
SPI_INCOME_QRF_OUTPUT_COLUMNS = (
    *SPI_INCOME_IMPUTATION_COLUMNS,
    *SPI_HMRC_AUXILIARY_COLUMNS,
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
    household_weight_kind: WeightKind | None = None
    mass_log: tuple[MassChangeRecord, ...] = ()
    replaced_spi_households: int = 0
    spi_prior_mass_share: float = 0.0

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
    selected_household_ids: Sequence[Any] | None = None,
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

    if selected_household_ids is None:
        selected_household_ids = _sample_spi_household_ids(
            household_frame,
            spi_household_count=spi_household_count,
            seed=seed,
        )
    else:
        selected_household_ids = _validate_selected_household_ids(
            household_frame,
            selected_household_ids,
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


def replace_uk_spi_support_tables(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    seed: int = 42,
    source_year: int | None = None,
    spi_prior_mass_share: float = DEFAULT_SPI_PRIOR_MASS_SHARE,
    input_weight_kind: WeightKind = WeightKind.DESIGN,
    mass_log: tuple[MassChangeRecord, ...] = (),
    strata_columns: Sequence[str] = SPI_REPLACEMENT_STRATA_COLUMNS,
) -> UKSPISupportResult:
    """Drop one dead SPI channel and rebuild it with conserved positive mass.

    The certified Populace UK candidate contains a complete enhanced-FRS base
    plus a zero-weight SPI-synthetic channel. This transform removes that dead
    channel, samples replacement source households within the exact reviewed
    clone/capital-gains/region quotas, rebuilds linked rows once, and allocates
    a fixed share of national household mass to the new channel. The allocation
    advances weights to ``IMPORTANCE`` and records a deliberate, factor-one
    :class:`MassChangeRecord`; national mass never doubles or disappears.
    """

    if not isinstance(input_weight_kind, WeightKind):
        raise TypeError("input_weight_kind must be a WeightKind.")
    if not isinstance(mass_log, tuple) or any(
        not isinstance(record, MassChangeRecord) for record in mass_log
    ):
        raise TypeError("mass_log must be a tuple of MassChangeRecord.")
    share = float(spi_prior_mass_share)
    if not np.isfinite(share) or not 0.0 < share < 1.0:
        raise ValueError("spi_prior_mass_share must be finite and in (0, 1).")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    person_frame = person.copy()
    benunit_frame = benunit.copy()
    household_frame = household.copy()
    _validate_uk_support_inputs(person_frame, benunit_frame, household_frame)
    if HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN not in household_frame:
        raise ValueError(
            "Certified UK base is missing household_is_spi_synthetic; the "
            "replacement stage requires one explicit existing SPI channel."
        )
    synthetic = _strict_bool_mask(
        household_frame[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN],
        label=HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    )
    if not synthetic.any():
        raise ValueError(
            "Certified UK base has no SPI-synthetic households to replace."
        )
    incoming_weights = pd.to_numeric(
        household_frame["household_weight"], errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)
    if not np.allclose(incoming_weights[synthetic], 0.0, rtol=0.0, atol=0.0):
        raise ValueError(
            "Existing SPI-synthetic households must all have zero incoming "
            "weight before replacement; refusing to discard live population mass."
        )

    stratum_columns = tuple(str(column) for column in strata_columns)
    if not stratum_columns or any(not column for column in stratum_columns):
        raise ValueError("strata_columns must contain non-empty column names.")
    _require_columns(household_frame, stratum_columns, label="household")
    replacement_source_ids = _sample_replacement_household_ids(
        household_frame,
        synthetic=synthetic,
        strata_columns=stratum_columns,
        seed=seed,
    )

    dead_household_ids = set(household_frame.loc[synthetic, "household_id"])
    dead_people = person_frame["person_household_id"].isin(dead_household_ids)
    dead_benunit_ids = set(person_frame.loc[dead_people, "person_benunit_id"])
    surviving_people = person_frame.loc[~dead_people].copy()
    surviving_benunits = benunit_frame.loc[
        ~benunit_frame["benunit_id"].isin(dead_benunit_ids)
    ].copy()
    surviving_households = household_frame.loc[~synthetic].copy()
    surviving_people = _strip_support_metadata(surviving_people, entity="person")
    surviving_benunits = _strip_support_metadata(surviving_benunits, entity="benunit")
    surviving_households = _strip_support_metadata(
        surviving_households,
        entity="household",
    ).drop(columns=[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN])

    rebuilt = create_uk_spi_support_tables(
        person=surviving_people,
        benunit=surviving_benunits,
        household=surviving_households,
        spi_household_count=None,
        seed=seed,
        source_year=source_year,
        selected_household_ids=replacement_source_ids,
    )
    allocated = _allocate_spi_prior_mass(
        rebuilt,
        spi_prior_mass_share=share,
        input_weight_kind=input_weight_kind,
        mass_log=mass_log,
    )
    return UKSPISupportResult(
        person=allocated.person,
        benunit=allocated.benunit,
        household=allocated.household,
        id_multiplier=allocated.id_multiplier,
        spi_household_ids=allocated.spi_household_ids,
        household_weight_kind=allocated.household_weight_kind,
        mass_log=allocated.mass_log,
        replaced_spi_households=int(synthetic.sum()),
        spi_prior_mass_share=share,
    )


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


def _validate_selected_household_ids(
    household: pd.DataFrame,
    selected_household_ids: Sequence[Any],
) -> tuple[Any, ...]:
    if isinstance(selected_household_ids, str | bytes):
        raise TypeError("selected_household_ids must be a sequence of IDs.")
    selected = tuple(selected_household_ids)
    if not selected:
        raise ValueError("selected_household_ids must not be empty.")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_household_ids must be unique.")
    available = set(household["household_id"])
    missing = [value for value in selected if value not in available]
    if missing:
        raise ValueError(
            "selected_household_ids contains value(s) absent from household: "
            f"{list(map(str, missing[:5]))}."
        )
    selected_set = set(selected)
    return tuple(
        value for value in household["household_id"].tolist() if value in selected_set
    )


def _sample_replacement_household_ids(
    household: pd.DataFrame,
    *,
    synthetic: np.ndarray,
    strata_columns: tuple[str, ...],
    seed: int,
) -> tuple[Any, ...]:
    base = household.loc[~synthetic]
    dead = household.loc[synthetic]
    base_groups = base.groupby(
        list(strata_columns),
        sort=True,
        dropna=False,
    ).indices
    dead_groups = dead.groupby(
        list(strata_columns),
        sort=True,
        dropna=False,
    ).indices
    rng = np.random.default_rng(seed)
    selected_ids: list[Any] = []
    for key in sorted(dead_groups, key=_stratum_sort_key):
        quota = len(dead_groups[key])
        candidates = base_groups.get(key)
        if candidates is None or len(candidates) < quota:
            capacity = 0 if candidates is None else len(candidates)
            raise ValueError(
                "Cannot rebuild the certified SPI quota for stratum "
                f"{key!r}: need {quota} base household(s), found {capacity}."
            )
        chosen_positions = rng.choice(candidates, size=quota, replace=False)
        selected_ids.extend(base.iloc[chosen_positions]["household_id"].tolist())
    return _validate_selected_household_ids(base, selected_ids)


def _stratum_sort_key(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, tuple) else (value,)
    return tuple("<NA>" if pd.isna(item) else str(item) for item in values)


def _strict_bool_mask(values: pd.Series, *, label: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"{label} contains missing values.")
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if not numeric.isin([0, 1]).all():
            raise ValueError(f"{label} must contain only boolean/0/1 values.")
        return numeric.to_numpy(dtype=float) != 0.0
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"{label} must contain only boolean/0/1 values.")
    return normalized.isin(["true", "1"]).to_numpy(dtype=bool)


def _strip_support_metadata(frame: pd.DataFrame, *, entity: str) -> pd.DataFrame:
    metadata = {
        support_channel_column(entity),
        support_clone_index_column(entity),
        support_source_id_column(entity),
    }
    return frame.drop(columns=sorted(metadata & set(frame.columns)))


def _allocate_spi_prior_mass(
    result: UKSPISupportResult,
    *,
    spi_prior_mass_share: float,
    input_weight_kind: WeightKind,
    mass_log: tuple[MassChangeRecord, ...],
) -> UKSPISupportResult:
    household = result.household.copy()
    channels = household[support_channel_column("household")]
    base_mask = channels.eq(BASE_FRS_SUPPORT_CHANNEL).to_numpy()
    spi_mask = channels.eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy()
    if not base_mask.any() or not spi_mask.any() or np.any(~(base_mask | spi_mask)):
        raise ValueError(
            "Rebuilt UK support must contain exactly FRS and SPI channels."
        )

    pre_weights = pd.to_numeric(
        household["household_weight"], errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)
    old_total = float(pre_weights.sum())
    source_weights = pd.Series(
        pre_weights[base_mask],
        index=household.loc[base_mask, "household_id"].to_numpy(),
    )
    spi_source_ids = household.loc[
        spi_mask,
        support_source_id_column("household"),
    ]
    spi_raw = spi_source_ids.map(source_weights).to_numpy(dtype=np.float64)
    if not np.isfinite(spi_raw).all() or not (spi_raw > 0.0).all():
        raise ValueError(
            "Every rebuilt SPI household must inherit a strictly positive "
            "source-household prior."
        )
    final_weights = np.zeros_like(pre_weights)
    final_weights[base_mask] = pre_weights[base_mask] * (1.0 - spi_prior_mass_share)
    final_weights[spi_mask] = spi_raw * (
        old_total * spi_prior_mass_share / float(spi_raw.sum())
    )

    schema = EntitySchema(group_entities=("benunit", "household"))
    audit_frame = Frame(
        {
            "person": result.person[
                ["person_id", "person_benunit_id", "person_household_id"]
            ].copy(),
            "benunit": result.benunit[["benunit_id"]].copy(),
            "household": result.household[["household_id"]].copy(),
        },
        schema,
        {"household": Weights(pre_weights, input_weight_kind)},
        mass_log=mass_log,
    )
    allocated_frame = audit_frame.with_weights(
        "household",
        Weights(final_weights, WeightKind.IMPORTANCE),
        mass=MassChange(
            factor=1.0,
            reason=_spi_prior_mass_change_reason(spi_prior_mass_share),
        ),
    )
    household["household_weight"] = allocated_frame.weights_for("household").values
    if not np.isclose(
        float(household["household_weight"].sum()),
        old_total,
        rtol=1e-9,
        atol=0.0,
    ):
        raise ValueError("UK SPI prior allocation failed to conserve national mass.")
    if not (household.loc[spi_mask, "household_weight"] > 0.0).all():
        raise ValueError("Rebuilt UK SPI channel contains dead zero-weight rows.")
    return UKSPISupportResult(
        person=result.person,
        benunit=result.benunit,
        household=household,
        id_multiplier=result.id_multiplier,
        spi_household_ids=result.spi_household_ids,
        household_weight_kind=WeightKind.IMPORTANCE,
        mass_log=allocated_frame.mass_log,
        spi_prior_mass_share=spi_prior_mass_share,
    )


def _spi_prior_mass_change_reason(share: float) -> str:
    if np.isclose(share, DEFAULT_SPI_PRIOR_MASS_SHARE, rtol=0.0, atol=0.0):
        return SPI_PRIOR_MASS_CHANGE_REASON
    return (
        f"Allocate {share:.12g} of certified UK national household prior mass "
        "to the rebuilt 2022-23 SPI support channel; total national mass is "
        "conserved."
    )


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
    "DEFAULT_SPI_PRIOR_MASS_SHARE",
    "DEFAULT_SPI_SUPPORT_HOUSEHOLDS",
    "FRS_ONLY_SPI_FILL_PERSON_COLUMNS",
    "FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS",
    "HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN",
    "SPI_SYNTHETIC_SUPPORT_CHANNEL",
    "SPI_INCOME_COMPONENT_COLUMNS",
    "SPI_HMRC_AUXILIARY_COLUMNS",
    "SPI_INCOME_IMPUTATION_COLUMNS",
    "SPI_INCOME_QRF_OUTPUT_COLUMNS",
    "SPI_PRIOR_MASS_CHANGE_REASON",
    "SPI_REPLACEMENT_STRATA_COLUMNS",
    "UKSPISupportResult",
    "UK_SPI_SUPPORT_STAGE_NAME",
    "create_uk_spi_support_tables",
    "fill_support_channel_from_source",
    "replace_uk_spi_support_tables",
    "support_channel_column",
    "support_clone_index_column",
    "support_source_id_column",
]
