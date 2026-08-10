"""Scale-ladder sampling policy for the UK national build (#627).

The UK samples a mid-pipeline artifact: the certified compact already carries
geography/CG clone families (``clone_index`` reverses to the canonical
``clone_index == 0`` surface by ID arithmetic) and the zero-weight SPI dead
channel whose rebuild enforces a per-stratum ``#base >= #dead`` quota.  A
uniform household draw would break both, so the UK policy over the shared
:mod:`microcosm.build.frame_sampling` core is:

- **The sampling unit is the canonical clone family.**  A drawn family brings
  its canonical household and every clone, so clone reversal always lands on
  a surviving ``clone_index == 0`` row.
- **The clone multiplier is pinned by forced retention.**  The stage fence
  re-derives ``10 ** len(str(max canonical id))`` from surviving data; the
  families carrying the argmax canonical household and person ids are always
  retained, so the digit count — and therefore every reversal — is stable.
- **The draw is stratified by the family-level channel flags and the
  canonical row's region** (SPI-synthetic x capital-gains-clone x the
  ``clone_index == 0`` region).  Clones agree with their canonical on both
  flags (the lineage fence asserts it) and the canonical region is
  family-constant by construction, so the labels are family-constant, and
  the per-group floor is monotone — the SPI quota inequality survives
  sampling cell by cell, including the clone-0 cells whose region varies by
  family.  A post-sample check re-asserts the full per-cell quota and fails
  closed, converting the uniform-clone-structure assumption into a runtime
  receipt.
- **Sampled mass is renormalized to the full-source household total** (the
  US stacked semantics #627 names), so the anchor population and the SPI
  50/50 mass-share allocation behave identically at every rung.
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
    "sample_uk_national_frame",
    "uk_canonical_family_units",
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
    _CLONE_INDEX_COLUMN,
    _CAPITAL_GAINS_CLONE_COLUMN,
    _REGION_COLUMN,
) = SPI_REPLACEMENT_STRATA_COLUMNS


def _required_household_columns() -> tuple[str, ...]:
    return (
        "household_id",
        "household_weight",
        HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
        *SPI_REPLACEMENT_STRATA_COLUMNS,
    )


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
    return values.to_numpy(dtype=np.int64)


def uk_canonical_family_units(
    frame: Frame,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Per-household-row canonical clone-family key, mirroring the stage fence.

    Reproduces the ``_resolve_candidate_lineage`` arithmetic: the multiplier
    is ``10 ** max(1, len(str(canonical_max)))`` over the ``clone_index == 0``
    household **and** person ids, and every row's family key is
    ``household_id - clone_index * multiplier``.  Fails closed when any clone
    does not reverse onto an existing canonical household — the same defect
    the stage fence would refuse later, surfaced before any draw.

    Returns:
        The per-household-row family keys, the forced family keys (the
        families carrying the argmax canonical household and person ids —
        retaining them pins the multiplier's digit count), and the multiplier.
    """

    household = frame.table("household")
    missing = sorted(set(_required_household_columns()) - set(household.columns))
    if missing:
        raise ValueError(
            "UK sample requires the certified compact's lineage and channel "
            f"columns; household is missing {missing}."
        )
    person = frame.table("person")
    household_ids = _int_column(household["household_id"], label="household_id")
    clone_index = _int_column(household[_CLONE_INDEX_COLUMN], label=_CLONE_INDEX_COLUMN)
    if (clone_index < 0).any():
        raise ValueError("UK sample: clone_index must be non-negative.")
    canonical_households = clone_index == 0
    if not canonical_households.any():
        raise ValueError("UK sample requires clone_index=0 canonical rows.")

    person_ids = _int_column(person["person_id"], label="person_id")
    person_household = _int_column(
        person["person_household_id"], label="person_household_id"
    )
    clone_by_household = pd.Series(clone_index, index=household_ids)
    person_clone = clone_by_household.reindex(person_household)
    if person_clone.isna().any():
        raise ValueError(
            "UK sample: person_household_id cannot map every person to a "
            "household clone level."
        )
    person_clone_index = person_clone.to_numpy(dtype=np.int64)
    canonical_people = person_clone_index == 0
    if not canonical_people.any():
        raise ValueError("UK sample requires clone_index=0 person rows.")

    canonical_max = max(
        int(household_ids[canonical_households].max()),
        int(person_ids[canonical_people].max()),
    )
    multiplier = 10 ** max(1, len(str(canonical_max)))
    units = household_ids - clone_index * multiplier
    if (units <= 0).any():
        raise ValueError(
            "UK sample: clone reversal produced non-positive canonical ids; "
            "the input's clone lineage does not match the stage fence's "
            "arithmetic."
        )
    canonical_set = household_ids[canonical_households]
    unknown = np.setdiff1d(np.unique(units), canonical_set)
    if len(unknown):
        raise ValueError(
            f"UK sample: {len(unknown)} clone household(s) do not reverse to "
            "a clone_index=0 canonical row; refusing to sample an input the "
            "stage fence would reject."
        )

    forced_household_family = int(household_ids[canonical_households].max())
    canonical_person_ids = np.where(canonical_people, person_ids, -1)
    argmax_person_position = int(np.argmax(canonical_person_ids))
    forced_person_family = int(person_household[argmax_person_position])
    forced = tuple(sorted({forced_household_family, forced_person_family}))
    return units, forced, multiplier


def _assert_spi_replacement_quota(household: pd.DataFrame) -> None:
    """Re-assert the SPI rebuild's per-cell ``#base >= #dead`` on the sample.

    The stratified family draw preserves the inequality when every family has
    a uniform clone structure; this check converts that assumption into a
    fail-closed runtime receipt, using the same stratum cells the rebuild's
    ``_sample_replacement_household_ids`` will group by.
    """

    synthetic = _strict_bool(
        household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN],
        label=HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    )
    dead = household.loc[synthetic]
    if dead.empty:
        return
    strata = list(SPI_REPLACEMENT_STRATA_COLUMNS)
    base = household.loc[~synthetic]
    dead_counts = dead.groupby(strata, sort=True, dropna=False).size()
    base_counts = base.groupby(strata, sort=True, dropna=False).size()
    for key, quota in dead_counts.items():
        capacity = int(base_counts.get(key, 0))
        if capacity < int(quota):
            raise ValueError(
                "UK sample violates the SPI replacement quota for stratum "
                f"{key!r}: need {int(quota)} base household(s), found "
                f"{capacity}. The stratified family draw preserves this "
                "inequality under a uniform clone structure; an irregular "
                "structure in the input is the likely cause."
            )


def sample_uk_national_frame(
    frame: Frame,
    *,
    fraction: float,
    seed: int,
) -> tuple[Frame, dict[str, object]]:
    """Draw one seeded UK rung sample and renormalize it to full mass.

    Runs on the loaded certified compact **before** provenance binding, so
    the certified-candidate fence attests the frame the stages actually
    consume.  The result satisfies :func:`validate_uk_national_frame`: the
    kernel mints the renormalization :class:`MassChangeRecord`, and the
    exported ``household_weight`` column is refreshed from the typed vector.

    Returns:
        The sampled frame and the shared sampling receipt extended with the
        normalization fields and the UK policy block.
    """

    validate_sample_fraction(fraction, label="UK sample")
    validate_sample_seed(seed, label="UK sample")
    validate_uk_national_frame(frame)

    household = frame.table("household")
    units, forced, multiplier = uk_canonical_family_units(frame)
    spi_flag = _strict_bool(
        household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN],
        label=HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    )
    cg_flag = _strict_bool(
        household[_CAPITAL_GAINS_CLONE_COLUMN],
        label=_CAPITAL_GAINS_CLONE_COLUMN,
    )
    household_ids = _int_column(household["household_id"], label="household_id")
    clone_zero = (
        _int_column(household[_CLONE_INDEX_COLUMN], label=_CLONE_INDEX_COLUMN) == 0
    )
    region_values = household[_REGION_COLUMN]
    if region_values.isna().any():
        raise ValueError(f"UK sample: {_REGION_COLUMN} contains missing values.")
    canonical_region = dict(
        zip(
            household_ids[clone_zero].tolist(),
            region_values.to_numpy()[clone_zero].tolist(),
            strict=True,
        )
    )
    # The stratum region is the family's canonical-row region: clone-0 cells
    # vary by family, so proportionality must hold within each canonical
    # region, not just per channel.
    family_region = np.asarray(
        [str(canonical_region[int(unit)]) for unit in units], dtype=object
    )
    strata = np.asarray(
        [
            f"spi={bool(spi)}|cg={bool(cg)}|region={region}"
            for spi, cg, region in zip(spi_flag, cg_flag, family_region, strict=True)
        ],
        dtype=object,
    )
    sampled, receipt = sample_frame_households(
        frame,
        fraction=fraction,
        seed=seed,
        source_name="UK national",
        unit_ids=units,
        unit_strata=strata,
        forced_unit_ids=forced,
        unit_noun="clone family",
        floor_context="the UK national build",
    )
    if sampled is not frame:
        full_mass = float(receipt["incoming_household_mass"])
        sampled, factor = normalize_sampled_household_mass(
            sampled, target_mass=full_mass, source_name="UK national"
        )
        # The typed weights are authoritative; refresh the exported column in
        # place so its position — and therefore the staging payload's column
        # order — is preserved.
        sampled.table("household")["household_weight"] = sampled.weights_for(
            "household"
        ).values
        receipt["normalization_factor"] = factor
        receipt["normalized_household_mass"] = float(
            sampled.weights_for("household").total
        )
    _assert_spi_replacement_quota(sampled.table("household"))
    validate_uk_national_frame(sampled)
    receipt["uk_policy"] = {
        "sampling_unit": "canonical_clone_family",
        "strata_columns": [
            HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
            _CAPITAL_GAINS_CLONE_COLUMN,
            f"{_REGION_COLUMN} (of the clone_index=0 canonical row)",
        ],
        "clone_multiplier": multiplier,
        "forced_retention": "argmax canonical household and person id families",
        "spi_replacement_quota_checked": True,
    }
    return sampled, receipt
