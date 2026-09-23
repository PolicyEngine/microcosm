"""UK SPI income band donors: reserved support rows for the top of the distribution.

The synthetic SPI channel (``spi_support_channel``) copies FRS households at
zero weight and lets the income stage draw their leaves from a quantile forest
conditioned on age, gender and region. A draw lands above GBP 2m with the
population probability, so ten thousand synthetic households carry an
expected two such people and the built spine carries none; HMRC's Table 2.5
puts ten thousand taxpayers and over twenty billion pounds of liabilities
there. This stage reserves rows for every Table 2.5 band from GBP 200,000 the
way ``cgt_band_donors`` reserves rows for the HMRC gain bands: whole FRS
households copied into the SPI channel, one carrier adult each, drawn by the
tape's own propensity for the band given region, sex and age, at the band's
published taxpayer count over the donors per band. The income stage then
gives each carrier a band-conditional draw from the tape (microcosm#280 lane).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.rowwise_geography import id_multiplier_for_values
from microcosm.build.uk_runtime.spi_income import (
    SPI_MINIMUM_RECIPIENT_AGE,
    VerifiedSPIDonorIdentity,
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
SPI_INCOME_BAND_DONORS_PER_BAND = 120
SPI_INCOME_BAND_DONOR_COUNT = SPI_INCOME_BAND_DONORS_PER_BAND * len(
    SPI_INCOME_BAND_DONOR_LOWER_BOUNDS
)
SPI_INCOME_BAND_DONOR_SEED = 3
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
SPI_INCOME_BAND_DONOR_INITIAL_WEIGHT = "published band taxpayers / donors_per_band"
SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON = (
    "SPI income band donors stack positive-weight support households per HMRC "
    "Table 2.5 total-income band from GBP 200,000, one carrier adult each; the "
    "published band taxpayer mass is added explicitly and the base and "
    "synthetic channels keep their weights."
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
    donors_per_band: int
    taxpayer_period: int
    carrier_region_counts: Mapping[str, int]
    propensity_cells: int

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_SPI_INCOME_BAND_DONORS_STAGE_NAME,
            "donors_per_band": self.donors_per_band,
            "taxpayer_count_resource": SPI_INCOME_BAND_TAXPAYER_RESOURCE,
            "taxpayer_count_period": self.taxpayer_period,
            "propensity": SPI_INCOME_BAND_DONOR_PROPENSITY,
            "propensity_cells": self.propensity_cells,
            "bands": [dict(row) for row in self.band_rows],
            "carrier_region_counts": dict(self.carrier_region_counts),
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
    donors_per_band: int = SPI_INCOME_BAND_DONORS_PER_BAND,
    seed: int = SPI_INCOME_BAND_DONOR_SEED,
    taxpayer_period: int,
) -> UKSPIIncomeBandDonorResult:
    """Copy ``donors_per_band`` FRS households per band into the SPI channel.

    Carriers are adults of the raw FRS base channel drawn without replacement
    with probability proportional to the tape's propensity for the band given
    their region, sex and age band; a household is drawn at most once across
    bands, the highest band first. Every copy carries the support channel's
    lineage columns at clone index 2, the synthetic flag (so the income stage
    draws for it), the donor flag, its band and one carrier flag, and starts
    at the band's published taxpayers over ``donors_per_band``.
    """

    validate_uk_national_frame(frame)
    if not isinstance(donors_per_band, int) or donors_per_band <= 0:
        raise ValueError("donors_per_band must be a positive integer.")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
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
    bands = sorted(int(value) for value in band_taxpayers)
    weights_by_band = {}
    for band in bands:
        taxpayers = float(band_taxpayers[band])
        weight = taxpayers / donors_per_band
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"SPI income band donors must carry positive initial weight; "
                f"band from {band} implies {weight!r}."
            )
        weights_by_band[band] = weight

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

    rng = np.random.default_rng(seed)
    selected_household: dict[int, int] = {}
    carrier_person: dict[int, int] = {}
    for band in sorted(bands, reverse=True):
        available = ~candidates["person_household_id"].isin(selected_household)
        pool = candidates.loc[available]
        probabilities = _candidate_propensity(pool, propensity, band=band)
        if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
            raise ValueError("SPI band propensities must be finite and non-negative.")
        positive = probabilities > 0.0
        if positive.sum() < donors_per_band:
            raise ValueError(
                f"SPI income band donors need {donors_per_band} candidate "
                f"households with positive propensity for the band from {band}; "
                f"found {int(positive.sum())}."
            )
        # Draw persons, one per household: a person draw whose household was
        # already taken in this band is skipped and the draw continues, so
        # the selection stays a weighted draw without replacement over
        # households at the person-level propensity.
        eligible = np.flatnonzero(positive)
        order = rng.choice(
            eligible,
            size=len(eligible),
            replace=False,
            p=probabilities[eligible] / probabilities[eligible].sum(),
        )
        taken: set[int] = set()
        for position in order:
            row = pool.iloc[int(position)]
            household_id = int(row["person_household_id"])
            if household_id in taken:
                continue
            taken.add(household_id)
            selected_household[household_id] = band
            carrier_person[int(row["person_id"])] = band
            if len(taken) == donors_per_band:
                break
        if len(taken) != donors_per_band:
            raise ValueError(
                f"SPI income band donors could not seat {donors_per_band} "
                f"households for the band from {band}."
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

    donor_weights = np.asarray(
        [weights_by_band[int(value)] for value in donor_band], dtype=float
    )
    final_person = pd.concat([person, donor_person], ignore_index=True)
    final_benunit = pd.concat([benunit, donor_benunit], ignore_index=True)
    final_household = pd.concat([household, donor_household], ignore_index=True)
    incoming = frame.weights_for("household")
    final_weights = np.r_[incoming.values, donor_weights]
    old_total = float(incoming.total)
    new_total = float(final_weights.sum())
    receipt = MassChangeRecord(
        entity="household",
        old_total=old_total,
        new_total=new_total,
        declared_factor=None,
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
                "donor_households": int(mask.sum()),
                "carriers": int(carriers.sum()),
                "donor_weight": float(weights_by_band[band]),
                "weighted_taxpayers": float(donor_weights[mask].sum()),
                "published_taxpayers": float(band_taxpayers[band]),
                "carrier_mean_age": float(carrier_ages[carriers].mean()),
                "carrier_regions": int(carrier_region[carriers].nunique()),
            }
        )
    return UKSPIIncomeBandDonorResult(
        frame=result,
        band_rows=tuple(band_rows),
        donors_per_band=donors_per_band,
        taxpayer_period=int(taxpayer_period),
        carrier_region_counts={
            str(key): int(value)
            for key, value in carrier_region.value_counts().sort_index().items()
        },
        propensity_cells=int(len(propensity)),
    )


@dataclass(frozen=True)
class UKSPIIncomeBandDonorStageTransform:
    """Whole-stage transform for the reserved SPI income band rows."""

    spi_tab_path: Path
    stage: SourceStageSpec
    seed: int = SPI_INCOME_BAND_DONOR_SEED
    # A #627 scale-ladder receipt build scales the reserved count with the
    # survey-side sample fraction, as the support stack does.
    sample_fraction: float = 1.0
    donor_table: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    band_taxpayers: Mapping[int, float] | None = None
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
    ) -> None:
        if donor_table is not None and not isinstance(donor_table, pd.DataFrame):
            raise TypeError("donor_table must be a pandas DataFrame.")
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
        donors_per_band = _assert_band_donor_stage_parameters(
            self.stage, seed=self.seed
        )
        if self.sample_fraction != 1.0:
            donors_per_band = max(1, int(round(donors_per_band * self.sample_fraction)))
        if self.donor_table is None:
            identity = verify_spi_donor_identity(self.spi_tab_path)
            raw = pd.read_csv(identity.path, delimiter="\t")
            if not isinstance(identity, VerifiedSPIDonorIdentity):
                raise TypeError("SPI donor identity verification returned no proof.")
        else:
            raw = self.donor_table.copy(deep=True)
        donor = prepare_spi_donor_table(raw, seed=self.seed)
        propensity = spi_income_band_propensity(
            donor, lower_bounds=SPI_INCOME_BAND_DONOR_LOWER_BOUNDS
        )
        period = int(uk_time_period(frame))
        band_taxpayers = (
            load_hmrc_itl_band_taxpayers(period)
            if self.band_taxpayers is None
            else dict(self.band_taxpayers)
        )
        result = stack_spi_income_band_donors(
            frame,
            propensity=propensity,
            band_taxpayers=band_taxpayers,
            donors_per_band=donors_per_band,
            seed=self.seed,
            taxpayer_period=period,
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


def _assert_band_donor_stage_parameters(stage: SourceStageSpec, *, seed: int) -> int:
    """Bind the stage manifest to the reviewed constants; return donors per band."""

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
    expected = {
        "taxpayer_count_resource": SPI_INCOME_BAND_TAXPAYER_RESOURCE,
        "taxpayer_count_measure": SPI_INCOME_BAND_TAXPAYER_MEASURE,
        "taxpayer_count_period": SPI_INCOME_BAND_TAXPAYER_PERIOD,
        "band_lower_bounds": list(SPI_INCOME_BAND_DONOR_LOWER_BOUNDS),
        "donors_per_band": SPI_INCOME_BAND_DONORS_PER_BAND,
        "expected_band_count": len(SPI_INCOME_BAND_DONOR_LOWER_BOUNDS),
        "expected_donor_count": SPI_INCOME_BAND_DONOR_COUNT,
        "candidate_population": SPI_INCOME_BAND_DONOR_CANDIDATES,
        "candidate_order": "person_id ascending",
        "draw": "weighted_without_replacement",
        "propensity": SPI_INCOME_BAND_DONOR_PROPENSITY,
        "seed": seed,
        "clone_index": SPI_INCOME_BAND_DONOR_CLONE_INDEX,
        "flag_column": HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
        "band_column": SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
        "carrier_column": PERSON_IS_SPI_INCOME_BAND_CARRIER,
        "initial_weight": SPI_INCOME_BAND_DONOR_INITIAL_WEIGHT,
        "never_zero_weight": True,
        "weight_kind_out": WeightKind.IMPORTANCE.value,
        "reason": SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON,
    }
    actual = dict(operation.parameters)
    if actual != expected:
        drifted = sorted(
            key for key in {*actual, *expected} if actual.get(key) != expected.get(key)
        )
        raise ValueError(
            "spi_income_band_donors stack_income_band_donor_households declaration "
            f"drifted from the reviewed mapping on parameter(s) {drifted}."
        )
    return int(expected["donors_per_band"])


__all__ = [
    "HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR",
    "PERSON_IS_SPI_INCOME_BAND_CARRIER",
    "SPI_INCOME_BAND_DONOR_CLONE_INDEX",
    "SPI_INCOME_BAND_DONOR_COUNT",
    "SPI_INCOME_BAND_DONOR_LOWER_BOUNDS",
    "SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN",
    "SPI_INCOME_BAND_DONOR_SEED",
    "SPI_INCOME_BAND_DONORS_PER_BAND",
    "UK_SPI_INCOME_BAND_DONOR_OUTPUT_COLUMNS",
    "UK_SPI_INCOME_BAND_DONORS_STAGE_NAME",
    "UKSPIIncomeBandDonorResult",
    "UKSPIIncomeBandDonorStageTransform",
    "income_band_lower_bound",
    "load_hmrc_itl_band_taxpayers",
    "spi_income_band_propensity",
    "stack_spi_income_band_donors",
]
