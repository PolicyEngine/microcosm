"""Scale-ladder sampling policy for the UK national build (#627).

The UK samples a mid-pipeline artifact: the certified compact already carries
geography/CG clone families (``clone_index`` reverses to the canonical
``clone_index == 0`` surface by ID arithmetic), SPI-synthetic and
capital-gains derivative rows that reverse **to the raw FRS surface** by
max-derived offsets (``max(canonical raw ids) + 1``), and the zero-weight SPI
dead channel whose rebuild enforces a per-stratum ``#base >= #dead`` quota.
A uniform household draw would break all three, so the UK policy over the
shared :mod:`microcosm.build.frame_sampling` core is:

- **The sampling unit is the source FRS family**: the raw canonical
  household plus every SPI/CG derivative and every geography clone that
  reverses onto it through both ID arithmetics.  A drawn family therefore
  keeps clone reversal *and* the SPI/CG source reversal intact — a derived
  row never survives without the raw row it reverses to.
- **The reversal constants are pinned by forced retention.**  The stage
  fence re-derives the clone multiplier (``10 ** len(str(max canonical
  id))``) and the SPI/CG offsets (``max(canonical raw id) + 1``,
  ``max(canonical pre-CG id) + 1``) from surviving data, so the families
  carrying each argmax id — canonical, raw, and pre-CG, household and
  person — are always retained and every constant is stable.
- **The draw is stratified by the raw canonical household's region.**
  Channel flags vary *within* a source family (raw + SPI + CG rows travel
  together), so they are not strata; the per-cell SPI quota is preserved
  structurally instead — a dead row's in-cell base source is in its own
  family — and a post-sample check re-asserts the full per-cell quota,
  failing closed on any irregular input.
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
    "uk_source_family_units",
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


def uk_source_family_units(
    frame: Frame,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Per-household-row source-FRS-family key, mirroring the stage fence.

    Reproduces ``_resolve_candidate_lineage``'s two-layer arithmetic: the
    clone multiplier is ``10 ** max(1, len(str(canonical_max)))`` over the
    ``clone_index == 0`` household **and** person ids; the SPI and CG
    household offsets are ``max(canonical raw household id) + 1`` and
    ``max(canonical pre-CG household id) + 1``; and every row's family key is

    ``household_id - clone_index*multiplier - spi*spi_offset - cg*cg_offset``

    which must land on an existing canonical raw FRS household.  Fails closed
    on any row the stage fence would refuse later, surfaced before any draw.

    Returns:
        The per-household-row family keys, the forced family keys (the
        families carrying every argmax id the fence's constants are derived
        from — retaining them pins the multiplier's digit count and both
        offsets exactly), and the clone multiplier.
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
    spi_flag = _strict_bool(
        household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN],
        label=HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    )
    cg_flag = _strict_bool(
        household[_CAPITAL_GAINS_CLONE_COLUMN],
        label=_CAPITAL_GAINS_CLONE_COLUMN,
    )
    canonical_households = clone_index == 0
    if not canonical_households.any():
        raise ValueError("UK sample requires clone_index=0 canonical rows.")

    person_ids = _int_column(person["person_id"], label="person_id")
    person_household = _int_column(
        person["person_household_id"], label="person_household_id"
    )
    household_position = pd.Series(np.arange(len(household_ids)), index=household_ids)
    person_household_position = household_position.reindex(person_household)
    if person_household_position.isna().any():
        raise ValueError(
            "UK sample: person_household_id cannot map every person to a household row."
        )
    person_positions = person_household_position.to_numpy(dtype=np.int64)
    person_clone_index = clone_index[person_positions]
    person_spi = spi_flag[person_positions]
    person_cg = cg_flag[person_positions]
    canonical_people = person_clone_index == 0
    if not canonical_people.any():
        raise ValueError("UK sample requires clone_index=0 person rows.")

    canonical_max = max(
        int(household_ids[canonical_households].max()),
        int(person_ids[canonical_people].max()),
    )
    multiplier = 10 ** max(1, len(str(canonical_max)))
    clone_reversed = household_ids - clone_index * multiplier
    if (clone_reversed <= 0).any():
        raise ValueError(
            "UK sample: clone reversal produced non-positive canonical ids; "
            "the input's clone lineage does not match the stage fence's "
            "arithmetic."
        )
    canonical_set = household_ids[canonical_households]
    unknown_clones = np.setdiff1d(np.unique(clone_reversed), canonical_set)
    if len(unknown_clones):
        raise ValueError(
            f"UK sample: {len(unknown_clones)} clone household(s) do not "
            "reverse to a clone_index=0 canonical row; refusing to sample an "
            "input the stage fence would reject."
        )

    raw_households = canonical_households & ~spi_flag & ~cg_flag
    pre_cg_households = canonical_households & ~cg_flag
    if not raw_households.any():
        raise ValueError("UK sample requires canonical raw FRS households.")
    spi_offset = int(household_ids[raw_households].max()) + 1
    cg_offset = int(household_ids[pre_cg_households].max()) + 1
    units = (
        clone_reversed
        - spi_flag.astype(np.int64) * spi_offset
        - cg_flag.astype(np.int64) * cg_offset
    )
    raw_set = frozenset(int(value) for value in household_ids[raw_households])
    if (units <= 0).any() or not set(int(v) for v in np.unique(units)) <= raw_set:
        bad = len(set(int(v) for v in np.unique(units)) - raw_set)
        raise ValueError(
            f"UK sample: {bad} household(s) do not reverse to the raw FRS "
            "surface; refusing to sample an input the stage fence would "
            "reject."
        )

    def _unit_of_max_household(mask: np.ndarray) -> int:
        position = int(np.argmax(np.where(mask, household_ids, -1)))
        return int(units[position])

    def _unit_of_max_person(mask: np.ndarray) -> int:
        position = int(np.argmax(np.where(mask, person_ids, -1)))
        return int(units[person_positions[position]])

    raw_people = canonical_people & ~person_spi & ~person_cg
    pre_cg_people = canonical_people & ~person_cg
    if not raw_people.any():
        raise ValueError("UK sample requires canonical raw FRS person rows.")
    forced = tuple(
        sorted(
            {
                # Clone-multiplier pins (digit count of the canonical max).
                _unit_of_max_household(canonical_households),
                _unit_of_max_person(canonical_people),
                # SPI-offset pins (max canonical raw id, exactly).
                _unit_of_max_household(raw_households),
                _unit_of_max_person(raw_people),
                # CG-offset pins (max canonical pre-CG id, exactly).
                _unit_of_max_household(pre_cg_households),
                _unit_of_max_person(pre_cg_people),
            }
        )
    )
    return units, forced, multiplier


def _assert_spi_replacement_quota(household: pd.DataFrame) -> None:
    """Re-assert the SPI rebuild's per-cell ``#base >= #dead`` on the sample.

    Source-family sampling preserves the inequality structurally when every
    dead row's source lives in its own cell; this check converts that
    assumption into a fail-closed runtime receipt, using the same stratum
    cells the rebuild's ``_sample_replacement_household_ids`` groups by.
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
                f"{capacity}. Source-family sampling preserves this "
                "inequality when dead rows share their source's cell; an "
                "irregular structure in the input is the likely cause."
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
    units, forced, multiplier = uk_source_family_units(frame)
    household_ids = _int_column(household["household_id"], label="household_id")
    region_values = household[_REGION_COLUMN]
    if region_values.isna().any():
        raise ValueError(f"UK sample: {_REGION_COLUMN} contains missing values.")
    region_by_household = dict(
        zip(
            household_ids.tolist(),
            region_values.to_numpy().tolist(),
            strict=True,
        )
    )
    # The stratum is the raw canonical household's region: clone-0 quota
    # cells vary by family region, so proportionality must hold within each
    # canonical region. Channel flags vary within a source family and are
    # deliberately not strata — derived rows travel with their source.
    strata = np.asarray(
        [f"region={region_by_household[int(unit)]}" for unit in units],
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
        unit_noun="source family",
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
        "sampling_unit": "source_frs_family",
        "strata_columns": [
            f"{_REGION_COLUMN} (of the raw clone_index=0 canonical row)"
        ],
        "clone_multiplier": multiplier,
        "forced_retention": (
            "families carrying the argmax canonical, raw, and pre-CG "
            "household and person ids (clone-multiplier and SPI/CG offset "
            "pins)"
        ),
        "spi_replacement_quota_checked": True,
    }
    return sampled, receipt
