"""UK SPI income band donors: reserved support rows for the top of the distribution.

The synthetic SPI channel (``spi_support_channel``) copies FRS households at
zero weight and lets the income stage draw their leaves from a quantile forest
conditioned on age, gender and region. A draw lands above GBP 2m with the
population probability, so ten thousand synthetic households carry an
expected two such people and the built spine carries none; HMRC's Table 2.5
puts ten thousand taxpayers and over twenty billion pounds of liabilities
there. This stage reserves rows for every Table 2.5 band from GBP 200,000 the
way the retired ``cgt_band_donors`` stage reserved rows for the HMRC gain
bands (``cgt_support_split`` replaced it in microcosm#1045): whole FRS
households copied into the SPI channel, one carrier adult each, drawn by the
tape's own propensity for the band given region, sex and age. The income stage
then gives each carrier a band-conditional draw from the tape
(PolicyEngine/chronicle#280 lane).

The donors are a support channel, not created mass (microcosm#1063). Each band
seats enough donors that none weighs more than a declared maximum, at equal
weights of the band's published taxpayers over its count, and the mass they
carry is taken from the households of the funding stratum each donor was drawn
from, in proportion, exactly as the SPI support channel takes its copies' mass
from the households of their region. Total household mass and every funding
stratum's mass are conserved; the seats are drawn from identity-keyed uniforms,
so the draw can be reconstructed from the ids.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.rowwise_geography import id_multiplier_for_values
from microcosm.build.uk_runtime.spi_income import (
    SPI_MINIMUM_RECIPIENT_AGE,
    SPIDonorAgeModel,
    VerifiedSPIDonorIdentity,
    load_spi_donor_age_model,
    prepare_spi_donor_table,
    verify_spi_donor_identity,
)
from microcosm.build.uk_runtime.spi_support import (
    _BENUNIT_ID_COLUMNS,
    _HOUSEHOLD_ID_COLUMNS,
    _PERSON_ID_COLUMNS,
    BASE_FRS_SUPPORT_CHANNEL,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    _clone_support_frame,
    _exact_total_correction,
    support_channel_column,
)
from microcosm.frame import Frame, MassChangeRecord, WeightKind, engine_tables

UK_SPI_INCOME_BAND_DONORS_STAGE_NAME = "spi_income_band_donors"
#: HMRC Income Tax liabilities statistics Table 2.5 total-income bands from
#: GBP 200,000: 200k-500k, 500k-1m, 1m-2m and 2m and over.
SPI_INCOME_BAND_DONOR_LOWER_BOUNDS: tuple[int, ...] = (
    200_000,
    500_000,
    1_000_000,
    2_000_000,
)
#: Every band seats at least this many donor households, and enough of them
#: that no donor starts above the maximum weight: the count is
#: ``max(minimum, ceil(published taxpayers / maximum weight))`` and the weight
#: the published taxpayers over the count, equal within the band.
SPI_INCOME_BAND_MINIMUM_DONORS = 120
SPI_INCOME_BAND_MAXIMUM_DONOR_WEIGHT = 250.0
#: Small frames: donors may take at most this share of the eligible base
#: households (the seating scale), and of any funding stratum's mass (the mass
#: scale). Both scales are one on the full frame, where the published band
#: masses are met exactly.
SPI_INCOME_BAND_MAXIMUM_DONOR_HOUSEHOLD_SHARE = 0.15
SPI_INCOME_BAND_MAXIMUM_REALLOCATION_SHARE = 0.5
#: The stratum a donor's mass is taken from: the households of its source
#: household's region, base and synthetic channel alike.
SPI_INCOME_BAND_FUNDING_STRATA: tuple[str, ...] = ("region",)
SPI_INCOME_BAND_DONOR_SEED = 3
#: The tape is prepared exactly as ``hmrc_spi_income_spine`` prepares it (its
#: stage-1 seed and its ONS age draw), so the propensity table and the income
#: stage's band pools read one set of donor ages. It matters for the records
#: the tape publishes no age band for (the composites) and for State Pension
#: recipients, whom the age draw keeps at or above State Pension age.
SPI_INCOME_BAND_DONOR_TAPE_SEED = 42
SPI_INCOME_BAND_DONOR_TAPE_PREPARATION = (
    "as hmrc_spi_income_spine prepares the tape: its stage-1 seed and "
    "draw_spi_donor_ages_by_population"
)
SPI_INCOME_BAND_DONOR_DRAW_SALT = "spi_income_band_donor_draw"
#: The support channel copies at clone index 1; the reserved copies sit at 2.
SPI_INCOME_BAND_DONOR_CLONE_INDEX = 2
HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR = "household_is_spi_income_band_donor"
SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN = "spi_income_band_donor_lower_bound"
PERSON_IS_SPI_INCOME_BAND_CARRIER = "person_is_spi_income_band_carrier"
SPI_INCOME_BAND_TAXPAYER_RESOURCE = "hmrc_itl_taxpayer_counts.json"
SPI_INCOME_BAND_TAXPAYER_MEASURE = "total_taxpayer_count"
SPI_INCOME_BAND_TAXPAYER_PERIOD = "build_period_tax_year"
SPI_INCOME_BAND_DONOR_CANDIDATES = "raw FRS base channel adults aged 16 and over"
SPI_INCOME_BAND_DONOR_PROPENSITY = (
    "SPI 2022-23 FACT-weighted band share by region, sex and age band"
)
SPI_INCOME_BAND_DONOR_COUNT_RULE = (
    "max(minimum_donors_per_band, ceil(published band taxpayers / "
    "maximum_donor_weight))"
)
SPI_INCOME_BAND_DONOR_WEIGHT_RULE = (
    "published band taxpayers / band donor count, equal within a band"
)
SPI_INCOME_BAND_DONOR_DRAW = (
    "weighted_without_replacement_identity_keyed: Efraimidis-Spirakis keys "
    "log(u) / (household weight x propensity) on stable_identity_uniforms("
    "person_id, seed, salt:band); the largest keys are seated, one carrier per "
    "household, the highest band first"
)
SPI_INCOME_BAND_DONOR_FUNDING = (
    "stratum reallocation: every incumbent household of the stratum a donor's "
    "source household belongs to is scaled by 1 - donor mass / stratum mass"
)
SPI_INCOME_BAND_DONOR_SEATING_SCALE = (
    "min(1, maximum_donor_household_share x eligible base households / expected "
    "donor count); band counts floored, at least one per band"
)
SPI_INCOME_BAND_DONOR_MASS_SCALE = (
    "min(1, min over funding strata of maximum_reallocation_share x stratum "
    "mass / seated donor mass); every band's weight scaled alike"
)
SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON = (
    "Reallocate the published HMRC Table 2.5 taxpayer mass of each total-income "
    "band from GBP 200,000 from the households of each donor's region to its "
    "income band donor support rows; every region's household mass and the "
    "total national mass are conserved."
)
#: Age bands for the propensity table: the tape's own AGERANGE cut points,
#: with the open 75-and-over band absorbing the oldest respondents.
SPI_INCOME_BAND_PROPENSITY_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (16, 25),
    (25, 35),
    (35, 45),
    (45, 55),
    (55, 65),
    (65, 75),
    (75, 1_000),
)
UK_SPI_INCOME_BAND_DONOR_OUTPUT_COLUMNS = (
    HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
    SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
    PERSON_IS_SPI_INCOME_BAND_CARRIER,
)


@dataclass(frozen=True)
class UKSPIIncomeBandDonorResult:
    """Frame with the reserved band rows and the receipt the gate reads."""

    frame: Frame
    band_rows: tuple[Mapping[str, object], ...]
    funding_rows: tuple[Mapping[str, object], ...]
    minimum_donors_per_band: int
    maximum_donor_weight: float
    maximum_donor_household_share: float
    maximum_reallocation_share: float
    seating_scale: float
    mass_scale: float
    eligible_households: int
    expected_donor_count: int
    donor_count: int
    reallocated_mass: float
    old_total: float
    new_total: float
    seed: int
    sample_fraction: float
    taxpayer_period: int
    carrier_region_counts: Mapping[str, int]
    propensity_cells: int

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_SPI_INCOME_BAND_DONORS_STAGE_NAME,
            "minimum_donors_per_band": self.minimum_donors_per_band,
            "maximum_donor_weight": self.maximum_donor_weight,
            "maximum_donor_household_share": self.maximum_donor_household_share,
            "maximum_reallocation_share": self.maximum_reallocation_share,
            "funding_strata": list(SPI_INCOME_BAND_FUNDING_STRATA),
            "seating_scale": self.seating_scale,
            "mass_scale": self.mass_scale,
            "sample_fraction": self.sample_fraction,
            "eligible_households": self.eligible_households,
            "expected_donor_count": self.expected_donor_count,
            "donor_count": self.donor_count,
            "reallocated_mass": self.reallocated_mass,
            "mass": {"old_total": self.old_total, "new_total": self.new_total},
            "draw": {"seed": self.seed, "salt": SPI_INCOME_BAND_DONOR_DRAW_SALT},
            "taxpayer_count_resource": SPI_INCOME_BAND_TAXPAYER_RESOURCE,
            "taxpayer_count_period": self.taxpayer_period,
            "propensity": SPI_INCOME_BAND_DONOR_PROPENSITY,
            "propensity_cells": self.propensity_cells,
            "bands": [dict(row) for row in self.band_rows],
            "funding": [dict(row) for row in self.funding_rows],
            "carrier_region_counts": dict(self.carrier_region_counts),
        }


def spi_income_band_donor_plan(
    band_taxpayers: Mapping[int, float],
    *,
    minimum_donors: int = SPI_INCOME_BAND_MINIMUM_DONORS,
    maximum_donor_weight: float = SPI_INCOME_BAND_MAXIMUM_DONOR_WEIGHT,
) -> dict[int, tuple[int, float]]:
    """Each band's donor count and equal donor weight on the full frame.

    The count is the smallest that keeps every donor at or below the maximum
    weight, and never below the minimum; the weight is the band's published
    taxpayers over that count, so the band's donors carry its published mass.
    """

    if not isinstance(minimum_donors, int) or minimum_donors <= 0:
        raise ValueError("minimum_donors must be a positive integer.")
    if not np.isfinite(maximum_donor_weight) or maximum_donor_weight <= 0.0:
        raise ValueError("maximum_donor_weight must be positive.")
    plan: dict[int, tuple[int, float]] = {}
    for band in sorted(int(value) for value in band_taxpayers):
        taxpayers = float(band_taxpayers[band])
        if not np.isfinite(taxpayers) or taxpayers <= 0.0:
            raise ValueError(
                f"SPI income band donors need positive published taxpayers; the "
                f"band from {band} carries {taxpayers!r}."
            )
        count = max(minimum_donors, math.ceil(taxpayers / maximum_donor_weight))
        weight = taxpayers / count
        if not 0.0 < weight <= maximum_donor_weight * (1.0 + 1e-12):
            raise ValueError(
                f"SPI income band donor weight {weight!r} for the band from {band} "
                f"is outside (0, {maximum_donor_weight}]."
            )
        plan[band] = (count, weight)
    return plan


def spi_income_band_donor_seats(
    plan: Mapping[int, tuple[int, float]],
    *,
    eligible_households: int,
    maximum_donor_household_share: float = (
        SPI_INCOME_BAND_MAXIMUM_DONOR_HOUSEHOLD_SHARE
    ),
) -> tuple[float, dict[int, int]]:
    """The seating scale and each band's seats on a frame of this size.

    The scale is one while the planned donors fit inside the declared share of
    the eligible base households; below that every band's count is scaled
    alike and floored, never below one, so a small frame keeps every band
    without copying most of its households.
    """

    expected = sum(count for count, _ in plan.values())
    if eligible_households <= 0:
        raise ValueError("SPI income band donors found no eligible base household.")
    scale = min(1.0, maximum_donor_household_share * eligible_households / expected)
    if scale >= 1.0:
        return 1.0, {band: count for band, (count, _) in plan.items()}
    return scale, {
        band: max(1, math.floor(count * scale)) for band, (count, _) in plan.items()
    }


def age_band_lower_bound(ages: np.ndarray) -> np.ndarray:
    """Map ages to the propensity table's age-band lower bounds."""

    lowers = np.asarray([low for low, _ in SPI_INCOME_BAND_PROPENSITY_AGE_BANDS])
    index = np.clip(np.searchsorted(lowers, ages, side="right") - 1, 0, len(lowers) - 1)
    return lowers[index]


def income_band_lower_bound(
    total_income: np.ndarray,
    *,
    lower_bounds: Sequence[int] = SPI_INCOME_BAND_DONOR_LOWER_BOUNDS,
) -> np.ndarray:
    """Band lower bound for each total income, or 0 below the first band."""

    lowers = np.asarray(sorted(int(value) for value in lower_bounds), dtype=float)
    index = np.searchsorted(lowers, np.asarray(total_income, dtype=float), side="right")
    result = np.zeros(len(total_income), dtype=float)
    positive = index > 0
    result[positive] = lowers[index[positive] - 1]
    return result


def load_hmrc_itl_band_taxpayers(
    build_period: int,
    *,
    lower_bounds: Sequence[int] = SPI_INCOME_BAND_DONOR_LOWER_BOUNDS,
) -> dict[int, float]:
    """Published Table 2.5 taxpayers per reserved band for the build tax year.

    Read through the vendored resource so a copy that lags the feed pin refuses
    before a weight is set; the build year's row is HMRC's own projection for
    that year (the 2023-24 outturn for a 2023 build).
    """

    taxpayers: dict[int, float] = {}
    for lower in lower_bounds:
        rows = vendored_rows(
            SPI_INCOME_BAND_TAXPAYER_RESOURCE,
            measure_id=SPI_INCOME_BAND_TAXPAYER_MEASURE,
            period_type="tax_year",
            period_value=int(build_period),
            dimensions={"total_income_lower_bound": int(lower)},
        )
        if len(rows) != 1:
            raise ValueError(
                f"{SPI_INCOME_BAND_TAXPAYER_RESOURCE} must carry exactly one "
                f"{SPI_INCOME_BAND_TAXPAYER_MEASURE} row for the band from "
                f"{lower} in tax year {build_period}; found {len(rows)}."
            )
        value = float(rows[0]["value"])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Table 2.5 taxpayers for the band from {lower} in tax year "
                f"{build_period} must be positive, got {value!r}."
            )
        taxpayers[int(lower)] = value
    return taxpayers


def spi_income_band_propensity(
    donor: pd.DataFrame,
    *,
    lower_bounds: Sequence[int] = SPI_INCOME_BAND_DONOR_LOWER_BOUNDS,
) -> pd.DataFrame:
    """FACT-weighted share of each reserved band by region, sex and age band.

    ``donor`` is the prepared tape (:func:`prepare_spi_donor_table`): ``age``
    drawn within the published range, ``gender``, ``region``, ``FACT`` and
    ``total_income`` (TEI + TII as published). One row per occupied cell and
    band; a cell with no tape mass for a band has propensity zero.
    """

    required = ("age", "gender", "region", "FACT", "total_income")
    missing = sorted(set(required) - set(donor.columns))
    if missing:
        raise ValueError(f"SPI band propensity needs donor columns {missing}.")
    frame = pd.DataFrame(
        {
            "region": donor["region"].astype(str).to_numpy(),
            "gender": donor["gender"].astype(str).to_numpy(),
            "age_band": age_band_lower_bound(donor["age"].to_numpy(dtype=float)),
            "band": income_band_lower_bound(
                donor["total_income"].to_numpy(dtype=float),
                lower_bounds=lower_bounds,
            ),
            "weight": donor["FACT"].to_numpy(dtype=float),
        }
    )
    cell_weight = frame.groupby(["region", "gender", "age_band"], sort=True)[
        "weight"
    ].sum()
    band_weight = (
        frame.loc[frame["band"] > 0.0]
        .groupby(["region", "gender", "age_band", "band"], sort=True)["weight"]
        .sum()
    )
    rows: list[dict[str, object]] = []
    for (region, gender, age_band, band), weight in band_weight.items():
        cell = float(cell_weight.loc[(region, gender, age_band)])
        rows.append(
            {
                "region": region,
                "gender": gender,
                "age_band": int(age_band),
                "band": int(band),
                "band_weight": float(weight),
                "cell_weight": cell,
                "propensity": float(weight) / cell,
            }
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "region",
            "gender",
            "age_band",
            "band",
            "band_weight",
            "cell_weight",
            "propensity",
        ],
    )
    for lower in lower_bounds:
        if not (result["band"] == int(lower)).any():
            raise ValueError(
                f"The SPI donor tape carries no record with total income from "
                f"{lower}; the band cannot be reserved."
            )
    return result


def spi_income_band_donor_propensity(
    raw: pd.DataFrame,
    *,
    age_model: SPIDonorAgeModel,
    tape_seed: int = SPI_INCOME_BAND_DONOR_TAPE_SEED,
) -> pd.DataFrame:
    """The stage's propensity table from the raw tape.

    One preparation for the stage and for the identity receipt that reruns
    it: the income stage's seed and age draw, then the band shares.
    """

    donor = prepare_spi_donor_table(raw, seed=tape_seed, age_model=age_model)
    return spi_income_band_propensity(
        donor, lower_bounds=SPI_INCOME_BAND_DONOR_LOWER_BOUNDS
    )


def _candidate_propensity(
    candidates: pd.DataFrame,
    propensity: pd.DataFrame,
    *,
    band: int,
) -> np.ndarray:
    table = propensity.loc[propensity["band"] == int(band)]
    lookup = table.set_index(["region", "gender", "age_band"])["propensity"]
    keys = pd.MultiIndex.from_arrays(
        [
            candidates["region"].astype(str),
            candidates["gender"].astype(str),
            candidates["age_band"].astype(int),
        ]
    )
    return lookup.reindex(keys).fillna(0.0).to_numpy(dtype=float)


def stack_spi_income_band_donors(
    frame: Frame,
    *,
    propensity: pd.DataFrame,
    band_taxpayers: Mapping[int, float],
    minimum_donors: int = SPI_INCOME_BAND_MINIMUM_DONORS,
    maximum_donor_weight: float = SPI_INCOME_BAND_MAXIMUM_DONOR_WEIGHT,
    maximum_donor_household_share: float = (
        SPI_INCOME_BAND_MAXIMUM_DONOR_HOUSEHOLD_SHARE
    ),
    maximum_reallocation_share: float = SPI_INCOME_BAND_MAXIMUM_REALLOCATION_SHARE,
    seed: int = SPI_INCOME_BAND_DONOR_SEED,
    taxpayer_period: int,
    sample_fraction: float = 1.0,
) -> UKSPIIncomeBandDonorResult:
    """Seat each band's donors and fund them from their source strata.

    Carriers are adults of the raw FRS base channel, seated without
    replacement with probability proportional to their household's weight
    times the tape's propensity for the band given their region, sex and age
    band, so the donors' composition follows the population's rather than the
    sample's; a household is seated at most once across bands, the highest
    band first. Every copy carries the support channel's lineage columns at
    clone index 2, the synthetic flag (so the income stage draws for it), the
    donor flag, its band and one carrier flag, and starts at its band's equal
    weight. The mass the donors carry leaves the incumbent households of each
    donor's funding stratum in proportion, so the total and every stratum's
    mass are unchanged.
    """

    validate_uk_national_frame(frame)
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    for label, share in (
        ("maximum_donor_household_share", maximum_donor_household_share),
        ("maximum_reallocation_share", maximum_reallocation_share),
    ):
        if not np.isfinite(share) or not 0.0 < share < 1.0:
            raise ValueError(f"{label} must lie strictly between zero and one.")
    tables = engine_tables(frame)
    person = tables["person"].copy()
    benunit = tables["benunit"].copy()
    household = tables["household"].copy()
    household_channel = support_channel_column("household")
    person_channel = support_channel_column("person")
    for column, table, label in (
        (household_channel, household, "household"),
        (person_channel, person, "person"),
        ("region", household, "household"),
        ("age", person, "person"),
        ("gender", person, "person"),
    ):
        if column not in table.columns:
            raise ValueError(
                f"SPI income band donors require {label} column {column!r}; "
                "run the stage after spi_support_channel."
            )
    if HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR in household.columns:
        raise ValueError("SPI income band donors were already stacked.")
    plan = spi_income_band_donor_plan(
        band_taxpayers,
        minimum_donors=minimum_donors,
        maximum_donor_weight=maximum_donor_weight,
    )
    bands = sorted(plan)
    incoming = frame.weights_for("household")
    incoming_weights = np.asarray(incoming.values, dtype=float)
    weight_by_household = pd.Series(
        incoming_weights, index=household["household_id"].to_numpy()
    )

    base_households = household.loc[
        household[household_channel].eq(BASE_FRS_SUPPORT_CHANNEL)
    ]
    region = base_households.set_index("household_id")["region"]
    candidates = person.loc[
        person[person_channel].eq(BASE_FRS_SUPPORT_CHANNEL)
        & person["person_household_id"].isin(set(base_households["household_id"]))
        & pd.to_numeric(person["age"], errors="coerce").ge(SPI_MINIMUM_RECIPIENT_AGE)
    ].copy()
    if candidates.empty:
        raise ValueError("SPI income band donors found no adult FRS candidates.")
    candidates["region"] = candidates["person_household_id"].map(region).astype(str)
    candidates["age_band"] = age_band_lower_bound(
        pd.to_numeric(candidates["age"], errors="coerce").to_numpy(dtype=float)
    )
    candidates = candidates.sort_values("person_id", kind="stable").reset_index(
        drop=True
    )
    candidate_ids = candidates["person_id"].to_numpy()
    candidate_households = candidates["person_household_id"].to_numpy()
    candidate_weights = (
        candidates["person_household_id"].map(weight_by_household).to_numpy(float)
    )
    if not np.isfinite(candidate_weights).all() or (candidate_weights < 0.0).any():
        raise ValueError("SPI income band donor candidates need finite weights.")

    # Rate of seating by weight x propensity for each band; a candidate with a
    # zero rate for every band can never be seated and is not eligible.
    rates: dict[int, np.ndarray] = {}
    for band in bands:
        probabilities = _candidate_propensity(candidates, propensity, band=band)
        if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
            raise ValueError("SPI band propensities must be finite and non-negative.")
        rates[band] = candidate_weights * probabilities
    any_rate = np.zeros(len(candidates), dtype=bool)
    for band in bands:
        any_rate |= rates[band] > 0.0
    eligible_households = int(pd.unique(candidate_households[any_rate]).size)
    seating_scale, seats = spi_income_band_donor_seats(
        plan,
        eligible_households=eligible_households,
        maximum_donor_household_share=maximum_donor_household_share,
    )

    selected_household: dict[int, int] = {}
    carrier_person: dict[int, int] = {}
    band_eligible: dict[int, int] = {}
    for band in sorted(bands, reverse=True):
        rate = rates[band]
        taken_before = np.fromiter(
            (int(value) in selected_household for value in candidate_households),
            dtype=bool,
            count=len(candidates),
        )
        eligible = (rate > 0.0) & ~taken_before
        band_eligible[band] = int(pd.unique(candidate_households[eligible]).size)
        if band_eligible[band] < seats[band]:
            raise ValueError(
                f"SPI income band donors need {seats[band]} candidate households "
                f"with positive weight and propensity for the band from {band}; "
                f"found {band_eligible[band]}."
            )
        uniforms = stable_identity_uniforms(
            candidate_ids,
            seed=seed,
            salt=f"{SPI_INCOME_BAND_DONOR_DRAW_SALT}:{band}",
        )
        # Efraimidis-Spirakis: the largest u ** (1 / rate) are a weighted draw
        # without replacement; log(u) / rate orders them identically.
        keys = np.full(len(candidates), -np.inf)
        keys[eligible] = (
            np.log(np.maximum(uniforms[eligible], np.finfo(float).tiny))
            / rate[eligible]
        )
        order = np.lexsort((candidate_ids, -keys))
        taken: set[int] = set()
        for position in order:
            if not eligible[position]:
                break
            household_id = int(candidate_households[position])
            if household_id in taken:
                continue
            taken.add(household_id)
            selected_household[household_id] = band
            carrier_person[int(candidate_ids[position])] = band
            if len(taken) == seats[band]:
                break
        if len(taken) != seats[band]:
            raise ValueError(
                f"SPI income band donors could not seat {seats[band]} households "
                f"for the band from {band}."
            )

    selected_ids = set(selected_household)
    multiplier = id_multiplier_for_values(
        person["person_id"],
        person["person_household_id"],
        person["person_benunit_id"],
        benunit["benunit_id"],
        household["household_id"],
    )
    donor_household = _clone_support_frame(
        household.loc[household["household_id"].isin(selected_ids)],
        entity="household",
        id_columns=_HOUSEHOLD_ID_COLUMNS,
        channel=SPI_SYNTHETIC_SUPPORT_CHANNEL,
        clone_index=SPI_INCOME_BAND_DONOR_CLONE_INDEX,
        id_multiplier=multiplier,
    )
    donor_person = _clone_support_frame(
        person.loc[person["person_household_id"].isin(selected_ids)],
        entity="person",
        id_columns=_PERSON_ID_COLUMNS,
        channel=SPI_SYNTHETIC_SUPPORT_CHANNEL,
        clone_index=SPI_INCOME_BAND_DONOR_CLONE_INDEX,
        id_multiplier=multiplier,
    )
    selected_benunits = set(
        person.loc[
            person["person_household_id"].isin(selected_ids), "person_benunit_id"
        ]
    )
    donor_benunit = _clone_support_frame(
        benunit.loc[benunit["benunit_id"].isin(selected_benunits)],
        entity="benunit",
        id_columns=_BENUNIT_ID_COLUMNS,
        channel=SPI_SYNTHETIC_SUPPORT_CHANNEL,
        clone_index=SPI_INCOME_BAND_DONOR_CLONE_INDEX,
        id_multiplier=multiplier,
    )
    source_household = donor_household["household_id"].to_numpy() - (
        SPI_INCOME_BAND_DONOR_CLONE_INDEX * multiplier
    )
    donor_band = np.asarray(
        [selected_household[int(value)] for value in source_household], dtype=float
    )
    household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR] = False
    household[SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN] = 0.0
    donor_household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN] = True
    donor_household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR] = True
    donor_household[SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN] = donor_band
    person[PERSON_IS_SPI_INCOME_BAND_CARRIER] = False
    source_person = donor_person["person_id"].to_numpy() - (
        SPI_INCOME_BAND_DONOR_CLONE_INDEX * multiplier
    )
    donor_person[PERSON_IS_SPI_INCOME_BAND_CARRIER] = np.isin(
        source_person, np.asarray(list(carrier_person), dtype="int64")
    )
    if int(donor_person[PERSON_IS_SPI_INCOME_BAND_CARRIER].sum()) != len(
        selected_household
    ):
        raise ValueError("SPI income band donors lost a carrier in the copy.")

    # Funding: the donors' mass leaves the incumbent households of each donor's
    # stratum in proportion (the SPI support channel's own rule), so no mass is
    # created and every stratum keeps its mass.
    planned_weights = np.asarray(
        [plan[int(value)][1] for value in donor_band], dtype=float
    )
    incumbent_strata = _funding_strata(household)
    donor_strata = _funding_strata(donor_household)
    incumbent_mass = (
        pd.Series(incoming_weights).groupby(incumbent_strata.to_numpy()).sum()
    )
    donor_mass = pd.Series(planned_weights).groupby(donor_strata.to_numpy()).sum()
    unfunded = sorted(set(donor_mass.index) - set(incumbent_mass.index))
    if unfunded or not (incumbent_mass.reindex(donor_mass.index) > 0.0).all():
        raise ValueError(
            "SPI income band donors were drawn from a stratum without incumbent "
            f"mass: {unfunded}."
        )
    capacity = (
        maximum_reallocation_share
        * incumbent_mass.reindex(donor_mass.index)
        / donor_mass
    )
    mass_scale = float(min(1.0, capacity.min()))
    donor_weights = planned_weights * mass_scale
    factor_by_stratum = (
        1.0 - (donor_mass * mass_scale / incumbent_mass.reindex(donor_mass.index))
    ).reindex(incumbent_mass.index, fill_value=1.0)
    incumbent_factor = incumbent_strata.map(factor_by_stratum).to_numpy(dtype=float)
    final_incumbent = incoming_weights * incumbent_factor
    if (final_incumbent[incoming_weights > 0.0] <= 0.0).any():
        raise ValueError("SPI income band donor funding emptied an incumbent row.")

    final_person = pd.concat([person, donor_person], ignore_index=True)
    final_benunit = pd.concat([benunit, donor_benunit], ignore_index=True)
    final_household = pd.concat([household, donor_household], ignore_index=True)
    old_total = float(incoming.total)
    final_weights = _exact_incumbent_total(
        np.r_[final_incumbent, donor_weights],
        target=old_total,
        incumbent_rows=len(final_incumbent),
    )
    new_total = float(final_weights.sum())
    receipt = MassChangeRecord(
        entity="household",
        old_total=old_total,
        new_total=new_total,
        declared_factor=1.0,
        reason=SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=final_person,
        benunit=final_benunit,
        household=final_household,
        time_period=uk_time_period(frame),
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=final_weights,
        mass_log=(*frame.mass_log, receipt),
    )
    validate_uk_national_frame(result)

    carrier_rows = donor_person.loc[donor_person[PERSON_IS_SPI_INCOME_BAND_CARRIER]]
    carrier_region = (
        carrier_rows["person_household_id"]
        .map(donor_household.set_index("household_id")["region"])
        .astype(str)
    )
    carrier_ages = pd.to_numeric(carrier_rows["age"], errors="coerce")
    band_of_carrier = carrier_rows["person_household_id"].map(
        donor_household.set_index("household_id")[
            SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN
        ]
    )
    band_rows: list[dict[str, object]] = []
    for band in bands:
        mask = donor_band == band
        carriers = band_of_carrier == band
        band_rows.append(
            {
                "lower_bound": band,
                "expected_donor_households": int(plan[band][0]),
                "donor_households": int(mask.sum()),
                "carriers": int(carriers.sum()),
                "eligible_households": band_eligible[band],
                "planned_donor_weight": float(plan[band][1]),
                "donor_weight": float(plan[band][1] * mass_scale),
                "weighted_taxpayers": float(donor_weights[mask].sum()),
                "published_taxpayers": float(band_taxpayers[band]),
                "carrier_mean_age": float(carrier_ages[carriers].mean()),
                "carrier_regions": int(carrier_region[carriers].nunique()),
            }
        )
    donor_rows_by_stratum = (
        pd.Series(1, index=donor_strata.to_numpy()).groupby(level=0).sum()
    )
    funding_rows = [
        {
            "stratum": str(stratum),
            "donor_households": int(donor_rows_by_stratum.get(stratum, 0)),
            "incumbent_mass": float(incumbent_mass[stratum]),
            "donor_mass": float(donor_mass.get(stratum, 0.0) * mass_scale),
            "factor": float(factor_by_stratum[stratum]),
        }
        for stratum in sorted(incumbent_mass.index, key=str)
    ]
    return UKSPIIncomeBandDonorResult(
        frame=result,
        band_rows=tuple(band_rows),
        funding_rows=tuple(funding_rows),
        minimum_donors_per_band=int(minimum_donors),
        maximum_donor_weight=float(maximum_donor_weight),
        maximum_donor_household_share=float(maximum_donor_household_share),
        maximum_reallocation_share=float(maximum_reallocation_share),
        seating_scale=float(seating_scale),
        mass_scale=mass_scale,
        eligible_households=eligible_households,
        expected_donor_count=int(sum(count for count, _ in plan.values())),
        donor_count=len(selected_household),
        reallocated_mass=float(donor_weights.sum()),
        old_total=old_total,
        new_total=new_total,
        seed=seed,
        sample_fraction=float(sample_fraction),
        taxpayer_period=int(taxpayer_period),
        carrier_region_counts={
            str(key): int(value)
            for key, value in carrier_region.value_counts().sort_index().items()
        },
        propensity_cells=int(len(propensity)),
    )


def _funding_strata(household: pd.DataFrame) -> pd.Series:
    """Each household's funding stratum label."""

    missing = sorted(set(SPI_INCOME_BAND_FUNDING_STRATA) - set(household.columns))
    if missing:
        raise ValueError(f"SPI income band donor funding needs columns {missing}.")
    columns = [
        household[column].astype(str) for column in SPI_INCOME_BAND_FUNDING_STRATA
    ]
    label = columns[0]
    for column in columns[1:]:
        label = label + "|" + column
    return label.reset_index(drop=True)


#: Rows tried for the exact-total correction before the stage refuses.
_EXACT_TOTAL_CORRECTION_CANDIDATES = 256


def _exact_incumbent_total(
    values: np.ndarray, *, target: float, incumbent_rows: int
) -> np.ndarray:
    """Weights whose sum is exactly ``target``, corrected on an incumbent row.

    The reallocation conserves mass arithmetically; the float sum can still
    miss the incoming total in its last bits. The correction goes onto one
    incumbent row, never a donor, so the donors keep their equal band weights
    bit for bit. A heavy row moves the sum in coarse steps that can jump over
    the target, so the heaviest rows are tried first and the lightest
    positive rows, whose steps are finer, straight after; the order depends
    only on the weights, so the corrected row is reproducible.
    """

    if float(values.sum()) == target:
        return values
    incumbents = values[:incumbent_rows]
    positive = np.flatnonzero(incumbents > 0.0)
    heaviest_first = positive[np.argsort(-incumbents[positive], kind="stable")]
    half = _EXACT_TOTAL_CORRECTION_CANDIDATES // 2
    candidates = dict.fromkeys(
        int(index) for index in (*heaviest_first[:half], *heaviest_first[::-1][:half])
    )
    for index in candidates:
        corrected = _exact_total_correction(
            values, target=target, correction_index=index
        )
        if corrected is not None:
            return corrected
    raise ValueError(
        "SPI income band donor funding has no representable exact-total correction."
    )


@dataclass(frozen=True)
class UKSPIIncomeBandDonorStageTransform:
    """Whole-stage transform for the reserved SPI income band rows."""

    spi_tab_path: Path
    stage: SourceStageSpec
    seed: int = SPI_INCOME_BAND_DONOR_SEED
    # The survey-side sample fraction of a #627 scale-ladder build. The stage
    # no longer scales its counts by it: the seating and mass scales follow
    # the frame itself. It is carried into the receipt so the gate can require
    # both scales to be one on a full build.
    sample_fraction: float = 1.0
    donor_table: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    band_taxpayers: Mapping[int, float] | None = None
    # The income stage's donor age model; read from the vendored ONS resource
    # and the engine's State Pension age at the build period when not given.
    age_model: SPIDonorAgeModel | None = field(default=None, repr=False, compare=False)
    last_result: UKSPIIncomeBandDonorResult | None = field(default=None, init=False)

    def __init__(
        self,
        spi_tab_path: str | Path,
        *,
        stage: SourceStageSpec,
        seed: int = SPI_INCOME_BAND_DONOR_SEED,
        sample_fraction: float = 1.0,
        donor_table: pd.DataFrame | None = None,
        band_taxpayers: Mapping[int, float] | None = None,
        age_model: SPIDonorAgeModel | None = None,
    ) -> None:
        if donor_table is not None and not isinstance(donor_table, pd.DataFrame):
            raise TypeError("donor_table must be a pandas DataFrame.")
        if age_model is not None and not isinstance(age_model, SPIDonorAgeModel):
            raise TypeError("age_model must be an SPIDonorAgeModel.")
        object.__setattr__(self, "age_model", age_model)
        object.__setattr__(self, "spi_tab_path", Path(spi_tab_path))
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "sample_fraction", float(sample_fraction))
        object.__setattr__(
            self,
            "donor_table",
            None if donor_table is None else donor_table.copy(deep=True),
        )
        object.__setattr__(
            self,
            "band_taxpayers",
            None if band_taxpayers is None else dict(band_taxpayers),
        )
        object.__setattr__(self, "last_result", None)

    def __call__(self, frame: Frame) -> Frame:
        validate_uk_national_frame(frame)
        _assert_band_donor_stage_parameters(self.stage, seed=self.seed)
        if self.donor_table is None:
            identity = verify_spi_donor_identity(self.spi_tab_path)
            raw = pd.read_csv(identity.path, delimiter="\t")
            if not isinstance(identity, VerifiedSPIDonorIdentity):
                raise TypeError("SPI donor identity verification returned no proof.")
        else:
            raw = self.donor_table.copy(deep=True)
        period = int(uk_time_period(frame))
        propensity = spi_income_band_donor_propensity(
            raw,
            age_model=(
                load_spi_donor_age_model(period)
                if self.age_model is None
                else self.age_model
            ),
        )
        band_taxpayers = (
            load_hmrc_itl_band_taxpayers(period)
            if self.band_taxpayers is None
            else dict(self.band_taxpayers)
        )
        result = stack_spi_income_band_donors(
            frame,
            propensity=propensity,
            band_taxpayers=band_taxpayers,
            seed=self.seed,
            taxpayer_period=period,
            sample_fraction=self.sample_fraction,
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_SPI_INCOME_BAND_DONOR_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def _operation(stage: SourceStageSpec, kind: str) -> SourceOperationSpec:
    matches = [operation for operation in stage.operations if operation.kind == kind]
    if len(matches) != 1:
        raise ValueError(
            f"Stage {stage.stage!r} must declare exactly one {kind!r} operation."
        )
    return matches[0]


def spi_income_band_donor_operation_parameters(
    *, seed: int = SPI_INCOME_BAND_DONOR_SEED
) -> dict[str, dict[str, object]]:
    """The reviewed operation payload the manifests must carry verbatim."""

    return {
        "stack_income_band_donor_households": {
            "taxpayer_count_resource": SPI_INCOME_BAND_TAXPAYER_RESOURCE,
            "taxpayer_count_measure": SPI_INCOME_BAND_TAXPAYER_MEASURE,
            "taxpayer_count_period": SPI_INCOME_BAND_TAXPAYER_PERIOD,
            "band_lower_bounds": list(SPI_INCOME_BAND_DONOR_LOWER_BOUNDS),
            "minimum_donors_per_band": SPI_INCOME_BAND_MINIMUM_DONORS,
            "maximum_donor_weight": SPI_INCOME_BAND_MAXIMUM_DONOR_WEIGHT,
            "count_rule": SPI_INCOME_BAND_DONOR_COUNT_RULE,
            "weight_rule": SPI_INCOME_BAND_DONOR_WEIGHT_RULE,
            "expected_band_count": len(SPI_INCOME_BAND_DONOR_LOWER_BOUNDS),
            "candidate_population": SPI_INCOME_BAND_DONOR_CANDIDATES,
            "candidate_order": "person_id ascending breaks equal draw keys",
            "draw": SPI_INCOME_BAND_DONOR_DRAW,
            "propensity": SPI_INCOME_BAND_DONOR_PROPENSITY,
            "tape_preparation": SPI_INCOME_BAND_DONOR_TAPE_PREPARATION,
            "tape_preparation_seed": SPI_INCOME_BAND_DONOR_TAPE_SEED,
            "seed": seed,
            "salt": SPI_INCOME_BAND_DONOR_DRAW_SALT,
            "clone_index": SPI_INCOME_BAND_DONOR_CLONE_INDEX,
            "flag_column": HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
            "band_column": SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
            "carrier_column": PERSON_IS_SPI_INCOME_BAND_CARRIER,
            "funding": SPI_INCOME_BAND_DONOR_FUNDING,
            "funding_strata": list(SPI_INCOME_BAND_FUNDING_STRATA),
            "maximum_reallocation_share": SPI_INCOME_BAND_MAXIMUM_REALLOCATION_SHARE,
            "maximum_donor_household_share": (
                SPI_INCOME_BAND_MAXIMUM_DONOR_HOUSEHOLD_SHARE
            ),
            "seating_scale": SPI_INCOME_BAND_DONOR_SEATING_SCALE,
            "mass_scale": SPI_INCOME_BAND_DONOR_MASS_SCALE,
            "never_zero_weight": True,
            "weight_kind_out": WeightKind.IMPORTANCE.value,
            "conservation": "exact_total",
            "declared_factor": 1.0,
            "reason": SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON,
        }
    }


def _assert_band_donor_stage_parameters(stage: SourceStageSpec, *, seed: int) -> None:
    """Bind the stage manifest to the reviewed constants."""

    if stage.stage != UK_SPI_INCOME_BAND_DONORS_STAGE_NAME:
        raise ValueError(
            f"SPI income band donor transform received stage {stage.stage!r}."
        )
    kinds = tuple(operation.kind for operation in stage.operations)
    if kinds != ("stack_income_band_donor_households",):
        raise ValueError(
            "spi_income_band_donors must declare exactly one "
            f"stack_income_band_donor_households operation, got {kinds}."
        )
    operation = _operation(stage, "stack_income_band_donor_households")
    expected = spi_income_band_donor_operation_parameters(seed=seed)[
        "stack_income_band_donor_households"
    ]
    actual = dict(operation.parameters)
    if actual != expected:
        drifted = sorted(
            key for key in {*actual, *expected} if actual.get(key) != expected.get(key)
        )
        raise ValueError(
            "spi_income_band_donors stack_income_band_donor_households declaration "
            f"drifted from the reviewed mapping on parameter(s) {drifted}."
        )


__all__ = [
    "HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR",
    "PERSON_IS_SPI_INCOME_BAND_CARRIER",
    "SPI_INCOME_BAND_DONOR_CLONE_INDEX",
    "SPI_INCOME_BAND_DONOR_DRAW_SALT",
    "SPI_INCOME_BAND_DONOR_LOWER_BOUNDS",
    "SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN",
    "SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON",
    "SPI_INCOME_BAND_DONOR_SEED",
    "SPI_INCOME_BAND_DONOR_TAPE_SEED",
    "SPI_INCOME_BAND_FUNDING_STRATA",
    "SPI_INCOME_BAND_MAXIMUM_DONOR_HOUSEHOLD_SHARE",
    "SPI_INCOME_BAND_MAXIMUM_DONOR_WEIGHT",
    "SPI_INCOME_BAND_MAXIMUM_REALLOCATION_SHARE",
    "SPI_INCOME_BAND_MINIMUM_DONORS",
    "UKSPIIncomeBandDonorResult",
    "UKSPIIncomeBandDonorStageTransform",
    "UK_SPI_INCOME_BAND_DONORS_STAGE_NAME",
    "UK_SPI_INCOME_BAND_DONOR_OUTPUT_COLUMNS",
    "income_band_lower_bound",
    "load_hmrc_itl_band_taxpayers",
    "spi_income_band_donor_operation_parameters",
    "spi_income_band_donor_plan",
    "spi_income_band_donor_propensity",
    "spi_income_band_donor_seats",
    "spi_income_band_propensity",
    "stack_spi_income_band_donors",
]
