"""Scale-ladder sampling for the UK spine (#627).

The UK full build samples the Microcosm spine at its ``uk.full.sample`` node,
before geographic expansion. Every derivative row on the spine names its raw
FRS household in ``source_household_id``, so the UK policy over the shared
:mod:`microcosm.build.frame_sampling` core is:

- **The sampling unit is the source family**: the raw FRS household and
  every row whose ``source_household_id`` names it. A drawn family is kept
  whole, so a derived row never survives without its raw row; the families
  are derived again on the sample, and a family left incomplete is refused.
- **The draw is stratified by the raw rows' region.** Derivative flags and
  derivative rows' own regions never split a family across strata, and a
  family whose raw rows span more than one region is refused.
- **Sampled mass is renormalized to the full-source household total** (the
  US stacked semantics #627 names).

A spine without lineage columns is sampled household by household,
stratified by region. The module also holds the rung tokens and the default
sampling seed that the UK builds share.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from microcosm.build.frame_sampling import (
    normalize_sampled_household_mass,
    sample_frame_households,
    validate_sample_fraction,
    validate_sample_seed,
)
from microcosm.build.uk_runtime.national_frame import validate_uk_national_frame
from microcosm.build.uk_runtime.spi_support import (
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_REPLACEMENT_STRATA_COLUMNS,
)
from microcosm.frame import Frame

__all__ = [
    "UK_SAMPLE_RUNG_TOKENS",
    "UK_SAMPLE_SEED_DEFAULT",
    "sample_uk_spine_frame",
    "uk_spine_source_family_units",
]

#: The #624 scale-ladder rungs and their identity tokens, shared with the US
#: stacked pipeline (``LOGBOOK_RUNGS``): ~1% smoke, 10% dev, full scale.
UK_SAMPLE_RUNG_TOKENS: Mapping[float, str] = {
    0.01: "f001",
    0.10: "f010",
    1.00: "f100",
}

#: Default survey-sampling seed, matching the US stacked pipeline's. Distinct
#: from the build ``--seed`` (SPI replacement draw, donor bootstrap, QRF) so
#: dev-scale seed sweeps vary one draw at a time.
UK_SAMPLE_SEED_DEFAULT = 578

(
    _,
    _CAPITAL_GAINS_CLONE_COLUMN,
    _REGION_COLUMN,
) = SPI_REPLACEMENT_STRATA_COLUMNS


def _strict_bool(values: pd.Series, *, label: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"UK sample: {label} contains missing values.")
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if not numeric.isin([0, 1]).all():
            raise ValueError(
                f"UK sample: {label} must contain only boolean/0/1 values."
            )
        return numeric.to_numpy(dtype=float) != 0.0
    raise ValueError(f"UK sample: {label} must be a boolean column.")


def _int_column(values: pd.Series, *, label: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"UK sample: {label} contains missing values.")
    if not pd.api.types.is_integer_dtype(values.dtype):
        raise ValueError(
            f"UK sample: {label} must be integer-typed; got dtype {values.dtype}."
        )
    return values.to_numpy(dtype=np.int64)


def _str_column(values: pd.Series, *, label: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"UK sample: {label} contains missing values.")
    array = values.to_numpy(dtype=object)
    if not all(isinstance(value, str) and value for value in array):
        raise ValueError(
            f"UK sample: {label} must contain non-empty strings; numeric "
            "codes are refused because equal codes of different dtypes "
            "(1 vs 1.0) would silently split one stratum into two."
        )
    return array


_SPINE_SOURCE_HOUSEHOLD_ID_COLUMN = "source_household_id"
_SPINE_SUPPORT_CLONE_INDEX_COLUMN = "household_support_clone_index"
_SPINE_CGT_SUPPORT_COPY_COLUMN = "household_is_cgt_support_copy"
_SPINE_CGT_RESIDENTIAL_ARM_COLUMN = "household_is_cgt_residential_clone"


def uk_spine_source_family_units(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """Return the UK spine's explicit source-family keys and raw-row strata.

    Every derivative row travels with the raw FRS household named by its
    ``source_household_id``. The raw household owns the family's region, so
    derivative flags and incidental derivative regions never split one
    source family across sampling strata.
    """

    household = frame.table("household")
    household_id_column = "household_id"
    lineage_columns = {
        _SPINE_SOURCE_HOUSEHOLD_ID_COLUMN,
        _SPINE_SUPPORT_CLONE_INDEX_COLUMN,
        HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
        _CAPITAL_GAINS_CLONE_COLUMN,
        _SPINE_CGT_SUPPORT_COPY_COLUMN,
        _SPINE_CGT_RESIDENTIAL_ARM_COLUMN,
    }
    required = {household_id_column, _REGION_COLUMN, *lineage_columns}
    missing = sorted(required - set(household.columns))
    present_lineage = lineage_columns & set(household.columns)
    if not present_lineage:
        raw_required = {household_id_column, _REGION_COLUMN}
        raw_missing = sorted(raw_required - set(household.columns))
        if raw_missing:
            raise ValueError(
                "UK spine sample requires household identity and region columns; "
                f"household is missing {raw_missing}."
            )
        household_ids = _int_column(
            household[household_id_column], label=household_id_column
        )
        regions = _str_column(household[_REGION_COLUMN], label=_REGION_COLUMN)
        return household_ids, np.asarray(
            [f"region={region}" for region in regions], dtype=object
        )
    if missing:
        raise ValueError(
            "UK spine sample has a partial lineage surface; "
            f"household is missing {missing}."
        )

    household_ids = _int_column(household["household_id"], label="household_id")
    units = _int_column(
        household[_SPINE_SOURCE_HOUSEHOLD_ID_COLUMN],
        label=_SPINE_SOURCE_HOUSEHOLD_ID_COLUMN,
    )
    support_clone = _int_column(
        household[_SPINE_SUPPORT_CLONE_INDEX_COLUMN],
        label=_SPINE_SUPPORT_CLONE_INDEX_COLUMN,
    )
    if (support_clone < 0).any():
        raise ValueError(
            "UK sample: household_support_clone_index must be non-negative."
        )
    spi = _strict_bool(
        household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN],
        label=HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    )
    capital_gains = _strict_bool(
        household[_CAPITAL_GAINS_CLONE_COLUMN],
        label=_CAPITAL_GAINS_CLONE_COLUMN,
    )
    # A support split's root keeps its raw-row role; its created copies join
    # the family through source_household_id (microcosm#1045).
    support_copy = _strict_bool(
        household[_SPINE_CGT_SUPPORT_COPY_COLUMN],
        label=_SPINE_CGT_SUPPORT_COPY_COLUMN,
    )
    # A residential split's incumbent arm keeps its raw-row role and ids;
    # the residential arms join the family the same way (microcosm#1063).
    residential_arm = _strict_bool(
        household[_SPINE_CGT_RESIDENTIAL_ARM_COLUMN],
        label=_SPINE_CGT_RESIDENTIAL_ARM_COLUMN,
    )
    regions = _str_column(household[_REGION_COLUMN], label=_REGION_COLUMN)

    raw = (
        (support_clone == 0) & ~spi & ~capital_gains & ~support_copy & ~residential_arm
    )
    source_rows = raw & (household_ids == units)
    source_keys = set(int(value) for value in units[source_rows])
    missing_sources = len(set(int(value) for value in units) - source_keys)
    if missing_sources:
        raise ValueError(
            f"UK spine sample: {missing_sources} source_household_id family "
            "key(s) lack a household_id-bearing raw row."
        )

    raw_region_by_unit: dict[int, str] = {}
    multi_region: list[int] = []
    for unit in sorted(set(int(value) for value in units)):
        family_regions = set(str(value) for value in regions[raw & (units == unit)])
        if len(family_regions) > 1:
            multi_region.append(unit)
        else:
            raw_region_by_unit[unit] = next(iter(family_regions))
    if multi_region:
        raise ValueError(
            f"UK spine sample: {len(multi_region)} source family(ies) span more "
            "than one raw-row region."
        )

    strata = np.asarray(
        [f"region={raw_region_by_unit[int(unit)]}" for unit in units],
        dtype=object,
    )
    return units, strata


def sample_uk_spine_frame(
    frame: Frame,
    *,
    fraction: float,
    seed: int,
) -> tuple[Frame, dict[str, object]]:
    """Sample complete explicit source families from a UK Microcosm spine."""

    validate_sample_fraction(fraction, label="UK spine sample")
    validate_sample_seed(seed, label="UK spine sample")
    validate_uk_national_frame(frame)

    household = frame.table("household")
    units, strata = uk_spine_source_family_units(frame)
    pre_counts = pd.Series(units).value_counts(sort=False)
    full_mass = float(frame.weights_for("household").total)
    sampled, core_receipt = sample_frame_households(
        frame,
        fraction=fraction,
        seed=seed,
        source_name="UK spine",
        unit_ids=units,
        unit_strata=strata,
        unit_noun="source family",
        floor_context="the UK rowwise candidate",
    )
    normalized, factor = normalize_sampled_household_mass(
        sampled,
        target_mass=full_mass,
        source_name="UK spine",
    )

    post_units, _ = uk_spine_source_family_units(normalized)
    post_counts = pd.Series(post_units).value_counts(sort=False)
    incomplete = sorted(
        int(unit)
        for unit, count in post_counts.items()
        if int(count) != int(pre_counts.loc[unit])
    )
    if incomplete:
        raise ValueError(
            f"UK spine sample retained {len(incomplete)} incomplete source family(ies)."
        )
    validate_uk_national_frame(normalized)
    return normalized, {
        "fraction": float(fraction),
        "seed": int(seed),
        "rung_token": UK_SAMPLE_RUNG_TOKENS.get(float(fraction)),
        "pre_household_count": int(len(household)),
        "post_household_count": int(len(normalized.table("household"))),
        "pre_family_count": int(len(pre_counts)),
        "post_family_count": int(len(post_counts)),
        "normalization_factor": float(factor),
        "strata_count": int(len(np.unique(strata))),
        "receipt": dict(core_receipt),
    }
