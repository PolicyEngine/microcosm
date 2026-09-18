"""Local-bus travel imputed onto the UK spine from the National Travel Survey.

The ``nts_bus_travel`` stage (microcosm#930) reads the NTS microdata (UKDS
SN 5340, End User Licence: Household, Individual and Trip tables linked by
``HouseholdID`` and ``IndividualID``; England residents only since 2013) and
imputes, per person of the FRS spine, the interview frequency-of-use band for
local buses, the annual local-bus trips that band carries in the diary, split
into the two published series (bus in London, other local bus), and the
statutory concessionary-travel eligibility of the person's area. It replaces
the take-up style incidence draw of microcosm#890 (published shares only) with
a conditional model (region, age band, sex, cars, household size, income band)
fitted on the survey, so that the lcfs stage can price journeys at a published
yield instead of raking fares to the receipts it is calibrated to.

Why the band, not the diary count: a seven-day diary annualised per person is
zero for about three quarters of people and a multiple of 52 for the rest,
right in aggregate and wrong in distribution. The interview band (the variable
behind DfT table NTS0313) carries incidence and level; the diary supplies the
mean annual trips per person in each band and residence group (London / rest
of England), weighted by W5 x JJXSC (short walks grossed) over W2 persons and
annualised by 52.14. If the deposited extract carries no band variable the
stage falls back to the published NTS0313 / NTS0621 shares (the #890 draw) and
scales declared band midpoints to the published trip rate; the receipt says
which path ran.

Every publisher value the stage checks itself against comes from vendored
Chronicle rows (``nts_trip_rates.json``, ``nts_bus_use_frequency.json``,
``nts_car_availability.json``); nothing derived from the licensed rows is
committed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.bus_use_incidence import (
    assign_bus_use_incidence,
    nts_band_shares,
)
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.support_clip import (
    UKSupportClipReceipt,
    support_clip_to_donor_with_receipt,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

UK_NTS_BUS_TRAVEL_STAGE_NAME = "nts_bus_travel"
UK_NTS_HOUSEHOLD_TAB_ROLE = "nts_household_tab"
UK_NTS_INDIVIDUAL_TAB_ROLE = "nts_individual_tab"
UK_NTS_TRIP_TAB_ROLE = "nts_trip_tab"

CLEAN_NTS_TRAVEL_TABLES_KIND = "clean_nts_travel_tables"
IMPUTE_BUS_USE_BAND_KIND = "impute_bus_use_band"
ASSIGN_TRIPS_FROM_BAND_MEANS_KIND = "assign_trips_from_band_means"
ASSIGN_BUS_PASS_ELIGIBILITY_KIND = "assign_bus_pass_eligibility"

BUS_IN_LONDON = "bus_in_london"
OTHER_LOCAL_BUS = "other_local_bus"
SERIES: tuple[str, str] = (BUS_IN_LONDON, OTHER_LOCAL_BUS)
SERIES_TRIP_COLUMNS: Mapping[str, str] = {
    BUS_IN_LONDON: "bus_in_london_trips",
    OTHER_LOCAL_BUS: "other_local_bus_trips",
}
SERIES_TRIP_RATE_CONCEPTS: Mapping[str, str] = {
    BUS_IN_LONDON: "dft.bus_in_london_trips_per_person",
    OTHER_LOCAL_BUS: "dft.other_local_bus_trips_per_person",
}

BAND_COLUMN = "local_bus_use_band"
TRIPS_COLUMN = "local_bus_trips"
ELIGIBLE_COLUMN = "bus_pass_eligible"
HOUSEHOLD_TRIPS_COLUMN = "household_local_bus_trips"
UK_NTS_PERSON_OUTPUT_COLUMNS: tuple[str, ...] = (
    BAND_COLUMN,
    SERIES_TRIP_COLUMNS[BUS_IN_LONDON],
    SERIES_TRIP_COLUMNS[OTHER_LOCAL_BUS],
    TRIPS_COLUMN,
    ELIGIBLE_COLUMN,
)
UK_NTS_HOUSEHOLD_OUTPUT_COLUMNS: tuple[str, ...] = (HOUSEHOLD_TRIPS_COLUMN,)
UK_NTS_BUS_TRAVEL_OUTPUT_COLUMNS: tuple[str, ...] = (
    *UK_NTS_PERSON_OUTPUT_COLUMNS,
    *UK_NTS_HOUSEHOLD_OUTPUT_COLUMNS,
)
#: The columns the person-level support clip covers (band and trips).
UK_NTS_SUPPORT_CLIP_COLUMNS: tuple[str, ...] = (
    BAND_COLUMN,
    SERIES_TRIP_COLUMNS[BUS_IN_LONDON],
    SERIES_TRIP_COLUMNS[OTHER_LOCAL_BUS],
    TRIPS_COLUMN,
)
#: Vendored resources the stage reads (the register's consumer truthfulness).
UK_NTS_TRIP_RATES_RESOURCE = "nts_trip_rates.json"
UK_NTS_CAR_AVAILABILITY_RESOURCE = "nts_car_availability.json"
UK_NTS_BUS_USE_FREQUENCY_RESOURCE = "nts_bus_use_frequency.json"
UK_NTS_BUS_TRAVEL_VENDORED_RESOURCES: frozenset[str] = frozenset(
    {
        UK_NTS_TRIP_RATES_RESOURCE,
        UK_NTS_CAR_AVAILABILITY_RESOURCE,
        UK_NTS_BUS_USE_FREQUENCY_RESOURCE,
    }
)

#: Ordinal bands, 0 = the non-user band. Higher is more frequent. Ids match the
#: NTS0313 band ids declared on the incidence resource so the fact checks join
#: on the same vocabulary.
BAND_IDS: tuple[str, ...] = (
    "less_than_once_a_year_or_never",
    "once_or_twice_a_year",
    "less_than_monthly_more_than_twice_a_year",
    "once_or_twice_a_month",
    "less_than_weekly_more_than_twice_a_month",
    "once_or_twice_a_week",
    "three_or_more_times_a_week",
)
NON_USER_BAND = 0
MAX_BAND = len(BAND_IDS) - 1
BAND_SALT_QUANTILE = "nts_bus_use_band_quantile"
BAND_SALT_SIGN = "nts_bus_use_band_sign"
WEEKS_IN_YEAR = 52.14
LONDON_GROUP = "london"
REST_OF_ENGLAND_GROUP = "rest_of_england"
RESIDENCE_GROUPS: tuple[str, str] = (LONDON_GROUP, REST_OF_ENGLAND_GROUP)
FREQUENCY_SOURCE_INTERVIEW = "interview_band"
FREQUENCY_SOURCE_PUBLISHED_SHARES = "published_shares"


class NTSBusTravelError(ValueError):
    """Raised when the NTS tabs or the declaration do not carry what the stage needs."""


# --------------------------------------------------------------------------
# Declaration readers
# --------------------------------------------------------------------------


def _artifact(stage: SourceStageSpec, role: str) -> Mapping[str, Any]:
    for artifact in stage.artifacts:
        if artifact.get("role") == role:
            return artifact
    raise NTSBusTravelError(f"{stage.stage} declares no {role!r} artifact.")


def _operation(stage: SourceStageSpec, kind: str) -> Mapping[str, Any]:
    for operation in stage.operations:
        if operation.kind == kind:
            return dict(operation.parameters)
    raise NTSBusTravelError(f"{stage.stage} declares no {kind!r} operation.")


def _optional_operation(stage: SourceStageSpec, kind: str) -> Mapping[str, Any] | None:
    for operation in stage.operations:
        if operation.kind == kind:
            return dict(operation.parameters)
    return None


@dataclass(frozen=True)
class NTSColumns:
    """The declared NTS column names and codes (one place to correct the codebook)."""

    household_id: str
    individual_id: str
    trip_id: str
    survey_year: str
    region: str
    income_band: str
    cars: str
    household_size: str
    household_weight: str
    age_band: str
    sex: str
    frequency: str | None
    main_mode: str
    trip_weight: str
    short_walk_multiplier: str
    region_codes: Mapping[int, str]
    london_region_code: int
    income_band_codes: Mapping[int, int]
    age_band_edges: tuple[int, ...]
    female_code: int
    mode_codes: Mapping[str, int]
    frequency_codes: Mapping[int, int]
    survey_years: tuple[int, ...]

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> NTSColumns:
        columns = _mapping(parameters.get("columns"), "derive.columns")
        codes = _mapping(parameters.get("codes"), "derive.codes")
        required = (
            "household_id",
            "individual_id",
            "trip_id",
            "survey_year",
            "region",
            "income_band",
            "cars",
            "household_size",
            "household_weight",
            "age_band",
            "sex",
            "main_mode",
            "trip_weight",
            "short_walk_multiplier",
        )
        missing = [name for name in required if not isinstance(columns.get(name), str)]
        if missing:
            raise NTSBusTravelError(f"derive.columns lacks {missing}.")
        frequency = columns.get("frequency")
        if frequency is not None and not isinstance(frequency, str):
            raise NTSBusTravelError(
                "derive.columns.frequency must be a string or absent."
            )
        region_codes = {
            int(code): str(name)
            for code, name in _mapping(codes.get("region"), "codes.region").items()
        }
        london = [code for code, name in region_codes.items() if name == "LONDON"]
        if len(london) != 1:
            raise NTSBusTravelError("codes.region must map exactly one code to LONDON.")
        income_codes = {
            int(code): int(band)
            for code, band in _mapping(
                codes.get("income_band"), "codes.income_band"
            ).items()
        }
        if sorted(set(income_codes.values())) != [1, 2, 3]:
            raise NTSBusTravelError(
                "codes.income_band must map onto the bands 1, 2, 3."
            )
        edges = tuple(int(e) for e in parameters.get("age_band_edges", ()))
        if len(edges) < 2 or list(edges) != sorted(edges) or edges[0] != 0:
            raise NTSBusTravelError(
                "age_band_edges must be an ascending list of lower bounds starting at 0."
            )
        mode_codes = {
            str(series): int(code)
            for series, code in _mapping(
                codes.get("main_mode"), "codes.main_mode"
            ).items()
        }
        if set(mode_codes) != set(SERIES):
            raise NTSBusTravelError(
                f"codes.main_mode must name exactly the series {list(SERIES)}."
            )
        frequency_codes = {
            int(code): int(band)
            for code, band in _mapping(
                codes.get("frequency", {}), "codes.frequency"
            ).items()
        }
        if frequency is not None:
            if sorted(set(frequency_codes.values())) != list(range(len(BAND_IDS))):
                raise NTSBusTravelError(
                    "codes.frequency must map the interview codes onto every band "
                    f"ordinal 0..{MAX_BAND}."
                )
        years = tuple(int(y) for y in parameters.get("survey_years", ()))
        if not years:
            raise NTSBusTravelError("derive.survey_years must name at least one year.")
        return cls(
            household_id=columns["household_id"].lower(),
            individual_id=columns["individual_id"].lower(),
            trip_id=columns["trip_id"].lower(),
            survey_year=columns["survey_year"].lower(),
            region=columns["region"].lower(),
            income_band=columns["income_band"].lower(),
            cars=columns["cars"].lower(),
            household_size=columns["household_size"].lower(),
            household_weight=columns["household_weight"].lower(),
            age_band=columns["age_band"].lower(),
            sex=columns["sex"].lower(),
            frequency=None if frequency is None else frequency.lower(),
            main_mode=columns["main_mode"].lower(),
            trip_weight=columns["trip_weight"].lower(),
            short_walk_multiplier=columns["short_walk_multiplier"].lower(),
            region_codes=region_codes,
            london_region_code=int(london[0]),
            income_band_codes=income_codes,
            age_band_edges=edges,
            female_code=int(codes.get("female", 2)),
            mode_codes=mode_codes,
            frequency_codes=frequency_codes,
            survey_years=years,
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NTSBusTravelError(f"{label} must be a mapping.")
    return value


# --------------------------------------------------------------------------
# Donor cleaning
# --------------------------------------------------------------------------


def _lower(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def _require(table: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise NTSBusTravelError(
            f"NTS {label} table lacks the declared columns {missing}; correct the "
            "stage's derive.columns against the deposited lookup tables."
        )


def age_band_ordinal(
    ages: Sequence[float] | np.ndarray, edges: Sequence[int]
) -> np.ndarray:
    """The declared age band (0-based ordinal) of each age."""

    values = np.asarray(ages, dtype=float)
    bounds = np.asarray(edges[1:], dtype=float)
    return np.searchsorted(bounds, values, side="right").astype(np.int64)


@dataclass(frozen=True)
class NTSDonor:
    """The cleaned NTS person donor and its diary trips."""

    person: pd.DataFrame
    frequency_source: str
    receipt: dict[str, Any]


def clean_nts_travel_tables(
    household: pd.DataFrame,
    individual: pd.DataFrame,
    trip: pd.DataFrame,
    *,
    columns: NTSColumns,
    cars_clip: tuple[int, int] = (0, 5),
) -> NTSDonor:
    """One row per NTS individual of the declared survey years, with predictors.

    Columns: ``region`` (FRS region name via the declared GOR codes),
    ``residence_group`` (``london`` / ``rest_of_england``), ``age_band``
    (declared ordinal), ``is_female``, ``num_vehicles`` (clipped), ``household_size``,
    ``income_band`` (1..3), ``weight`` (W2, the diary household weight), the
    interview ``local_bus_use_band`` when the extract carries it, and the
    annual diary trips per series (52.14 x sum of trip weight x short-walk
    multiplier over the person's trips whose main mode is that series).
    """

    hh = _lower(household)
    ind = _lower(individual)
    tr = _lower(trip)
    _require(
        hh,
        (
            columns.household_id,
            columns.survey_year,
            columns.region,
            columns.income_band,
            columns.cars,
            columns.household_size,
            columns.household_weight,
        ),
        "household",
    )
    _require(
        ind,
        (columns.individual_id, columns.household_id, columns.age_band, columns.sex),
        "individual",
    )
    _require(
        tr,
        (
            columns.trip_id,
            columns.individual_id,
            columns.main_mode,
            columns.trip_weight,
            columns.short_walk_multiplier,
        ),
        "trip",
    )
    years = pd.to_numeric(hh[columns.survey_year], errors="coerce")
    hh = hh.loc[years.isin(columns.survey_years)].copy()
    if hh.empty:
        raise NTSBusTravelError(
            f"NTS household table has no rows for survey years {columns.survey_years}."
        )
    weight = pd.to_numeric(hh[columns.household_weight], errors="coerce")
    hh = hh.loc[weight.notna() & (weight > 0)].copy()
    region_code = pd.to_numeric(hh[columns.region], errors="coerce")
    region = region_code.map(columns.region_codes)
    unmapped = int(region.isna().sum())
    hh = hh.loc[region.notna()].copy()
    hh["region"] = region.loc[hh.index].astype(str)
    hh["residence_group"] = np.where(
        pd.to_numeric(hh[columns.region], errors="coerce")
        == columns.london_region_code,
        LONDON_GROUP,
        REST_OF_ENGLAND_GROUP,
    )
    income_code = pd.to_numeric(hh[columns.income_band], errors="coerce")
    income = income_code.map(columns.income_band_codes)
    income_missing = int(income.isna().sum())
    hh = hh.loc[income.notna()].copy()
    hh["income_band"] = income.loc[hh.index].astype(np.int64)
    cars = pd.to_numeric(hh[columns.cars], errors="coerce").fillna(0.0)
    hh["num_vehicles"] = cars.clip(lower=cars_clip[0], upper=cars_clip[1]).astype(
        np.int64
    )
    hh["household_size"] = (
        pd.to_numeric(hh[columns.household_size], errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0)
    ).astype(np.int64)
    hh["weight"] = pd.to_numeric(hh[columns.household_weight], errors="coerce").astype(
        float
    )
    hh_keep = hh[
        [
            columns.household_id,
            columns.survey_year,
            "region",
            "residence_group",
            "income_band",
            "num_vehicles",
            "household_size",
            "weight",
        ]
    ].rename(columns={columns.survey_year: "survey_year"})
    person = ind[
        [columns.individual_id, columns.household_id, columns.age_band, columns.sex]
    ]
    person = person.merge(hh_keep, on=columns.household_id, how="inner")
    age_code = pd.to_numeric(person[columns.age_band], errors="coerce")
    # The NTS band code is 1-based and ordered like the declared edges.
    person["age_band"] = (age_code - 1).clip(
        lower=0, upper=len(columns.age_band_edges) - 1
    )
    person = person.loc[age_code.notna()].copy()
    person["age_band"] = person["age_band"].astype(np.int64)
    sex = pd.to_numeric(person[columns.sex], errors="coerce")
    person["is_female"] = (sex == columns.female_code).astype(np.int64)
    frequency_source = FREQUENCY_SOURCE_PUBLISHED_SHARES
    if columns.frequency is not None and columns.frequency in ind.columns:
        freq = pd.to_numeric(
            ind.set_index(columns.individual_id)[columns.frequency], errors="coerce"
        )
        band = person[columns.individual_id].map(freq).map(columns.frequency_codes)
        known = band.notna()
        person = person.loc[known].copy()
        person[BAND_COLUMN] = band.loc[person.index].astype(np.int64)
        frequency_source = FREQUENCY_SOURCE_INTERVIEW
    # Diary trips per person per series, annualised.
    tr = tr.loc[tr[columns.individual_id].isin(person[columns.individual_id])].copy()
    mode = pd.to_numeric(tr[columns.main_mode], errors="coerce")
    trip_weight = pd.to_numeric(tr[columns.trip_weight], errors="coerce").fillna(0.0)
    multiplier = pd.to_numeric(
        tr[columns.short_walk_multiplier], errors="coerce"
    ).fillna(0.0)
    counted = trip_weight * multiplier
    for series in SERIES:
        code = columns.mode_codes[series]
        per_person = (
            counted.where(mode == code, 0.0).groupby(tr[columns.individual_id]).sum()
        )
        person[SERIES_TRIP_COLUMNS[series]] = (
            person[columns.individual_id]
            .map(per_person)
            .fillna(0.0)
            .to_numpy(dtype=float)
            * WEEKS_IN_YEAR
        )
    person[TRIPS_COLUMN] = (
        person[SERIES_TRIP_COLUMNS[BUS_IN_LONDON]]
        + person[SERIES_TRIP_COLUMNS[OTHER_LOCAL_BUS]]
    )
    person = person.rename(
        columns={
            columns.individual_id: "individual_id",
            columns.household_id: "household_id",
        }
    )
    keep = [
        "individual_id",
        "household_id",
        "survey_year",
        "region",
        "residence_group",
        "age_band",
        "is_female",
        "num_vehicles",
        "household_size",
        "income_band",
        "weight",
        SERIES_TRIP_COLUMNS[BUS_IN_LONDON],
        SERIES_TRIP_COLUMNS[OTHER_LOCAL_BUS],
        TRIPS_COLUMN,
    ]
    if frequency_source == FREQUENCY_SOURCE_INTERVIEW:
        keep.append(BAND_COLUMN)
    person = person[keep].reset_index(drop=True)
    if person.empty:
        raise NTSBusTravelError("NTS cleaning left no donor persons.")
    receipt = {
        "survey_years": list(columns.survey_years),
        "households": int(len(hh_keep)),
        "persons": int(len(person)),
        "weighted_persons": float(person["weight"].sum()),
        "households_unmapped_region": unmapped,
        "households_missing_income_band": income_missing,
        "frequency_source": frequency_source,
        "frequency_column": columns.frequency,
        "annualisation_weeks": WEEKS_IN_YEAR,
        "trips_per_person": {
            series: float(np.average(person[column], weights=person["weight"]))
            for series, column in SERIES_TRIP_COLUMNS.items()
        },
    }
    return NTSDonor(person=person, frequency_source=frequency_source, receipt=receipt)


# --------------------------------------------------------------------------
# Recipient predictors
# --------------------------------------------------------------------------

UK_NTS_ENGINE_PREDICTORS: tuple[str, ...] = ("household_gross_income",)


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def recipient_predictors(
    frame: Frame,
    engine: object,
    *,
    columns: NTSColumns,
    income_band_edges: Sequence[float],
    region_remap: Mapping[str, str],
    cars_clip: tuple[int, int] = (0, 5),
) -> pd.DataFrame:
    """Person-grain predictors of the FRS spine in the donor's vocabulary."""

    person = frame.table("person")
    household = frame.table("household")
    for column in ("person_id", "person_household_id", "age", "gender"):
        if column not in person:
            raise NTSBusTravelError(f"recipient person table lacks {column!r}.")
    for column in ("household_id", "region", "num_vehicles"):
        if column not in household:
            raise NTSBusTravelError(
                f"recipient household table lacks {column!r}"
                + (" (the was_wealth draw)." if column == "num_vehicles" else ".")
            )
    materialized = engine.materialize(
        frame, UK_NTS_ENGINE_PREDICTORS, uk_time_period(frame)
    )
    by_household = household.set_index("household_id")
    hh_ids = person["person_household_id"].to_numpy()
    region = by_household["region"].map(_enum_name).reindex(hh_ids).to_numpy()
    if pd.isna(region).any():
        raise NTSBusTravelError(
            "every recipient person must map to a household region."
        )
    remapped = np.asarray(
        [region_remap.get(str(r), str(r)) for r in region], dtype=object
    )
    unknown = sorted(set(remapped) - set(columns.region_codes.values()))
    if unknown:
        raise NTSBusTravelError(
            f"recipient regions {unknown} are outside the donor's regions; declare a "
            "region_remap for them."
        )
    gross = np.asarray(materialized["household_gross_income"], dtype=float)
    if len(gross) != len(household):
        raise NTSBusTravelError(
            "household_gross_income did not materialize at household grain."
        )
    gross_by_household = (
        pd.Series(gross, index=by_household.index).reindex(hh_ids).to_numpy()
    )
    edges = np.asarray(list(income_band_edges), dtype=float)
    income_band = np.searchsorted(edges, gross_by_household, side="right") + 1
    sizes = person.groupby("person_household_id").size()
    cars = pd.to_numeric(by_household["num_vehicles"], errors="coerce").fillna(0.0)
    result = pd.DataFrame(
        {
            "person_id": person["person_id"].to_numpy(),
            "person_household_id": hh_ids,
            "region": remapped,
            "residence_region": region,
            "residence_group": np.where(
                region == "LONDON", LONDON_GROUP, REST_OF_ENGLAND_GROUP
            ),
            "age": pd.to_numeric(person["age"], errors="coerce").fillna(0.0).to_numpy(),
            "age_band": age_band_ordinal(
                pd.to_numeric(person["age"], errors="coerce").fillna(0.0).to_numpy(),
                columns.age_band_edges,
            ),
            "is_female": (
                person["gender"].map(_enum_name).str.upper().isin(("FEMALE", "F", "2"))
            )
            .astype(np.int64)
            .to_numpy(),
            "num_vehicles": cars.clip(lower=cars_clip[0], upper=cars_clip[1])
            .reindex(hh_ids)
            .astype(np.int64)
            .to_numpy(),
            "household_size": sizes.reindex(hh_ids).astype(np.int64).to_numpy(),
            "income_band": income_band.astype(np.int64),
        }
    )
    return result


# --------------------------------------------------------------------------
# Band imputation
# --------------------------------------------------------------------------

BAND_PREDICTORS: tuple[str, ...] = (
    "age_band",
    "is_female",
    "num_vehicles",
    "household_size",
    "income_band",
)


def _encode_pair(
    donor: pd.DataFrame, recipient: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base = list(BAND_PREDICTORS) + ["region"]
    combined = pd.concat(
        [donor[base].reset_index(drop=True), recipient[base].reset_index(drop=True)],
        ignore_index=True,
    )
    encoded = pd.get_dummies(
        combined, columns=["region"], drop_first=False, dtype=float
    )
    encoded = encoded.reindex(sorted(encoded.columns), axis=1).astype(float)
    predictors = list(encoded.columns)
    donor_encoded = encoded.iloc[: len(donor)].copy().reset_index(drop=True)
    recipient_encoded = encoded.iloc[len(donor) :].copy().reset_index(drop=True)
    return donor_encoded, recipient_encoded, predictors


@dataclass(frozen=True)
class BandImputation:
    band: np.ndarray
    receipt: dict[str, Any]
    fit_weight_records: tuple[FitWeightRecord, ...]


def impute_bus_use_band(
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int,
) -> BandImputation:
    """Draw each recipient person's interview band from a QRF fitted on the donor.

    The band is an ordinal with a zero-inflated positive regime (0 = non-user):
    the gate draws user / non-user, the positive forest the band among users,
    both from identity-keyed uniforms (``stable_identity_uniforms`` on the
    person id), so the draw is invariant to ordering and batching. Draws are
    rounded to the nearest band.
    """

    from microcosm.fit.qrf import RegimeGatedQRF

    if BAND_COLUMN not in donor:
        raise NTSBusTravelError("the donor carries no interview band to fit.")
    donor_encoded, recipient_encoded, predictors = _encode_pair(donor, recipient)
    train = donor_encoded.copy()
    train[BAND_COLUMN] = donor[BAND_COLUMN].to_numpy(dtype=float)
    train["weight"] = donor["weight"].to_numpy(dtype=float)
    model = RegimeGatedQRF(n_estimators=n_estimators, seed=seed)
    fitted = model.fit(train, predictors, [BAND_COLUMN], weights="weight")
    ids = recipient["person_id"].to_numpy()
    quantiles = stable_identity_uniforms(ids, seed=seed, salt=BAND_SALT_QUANTILE)
    signs = stable_identity_uniforms(ids, seed=seed, salt=BAND_SALT_SIGN)
    drawn = fitted.predict_from_uniforms(
        recipient_encoded,
        quantiles={BAND_COLUMN: quantiles},
        sign_uniforms={BAND_COLUMN: signs},
    )[BAND_COLUMN].to_numpy(dtype=float)
    band = np.clip(np.rint(drawn), NON_USER_BAND, MAX_BAND).astype(np.int64)
    weights = donor["weight"].to_numpy(dtype=float)
    donor_shares = _band_shares(donor[BAND_COLUMN].to_numpy(), weights)
    receipt = {
        "method": "regime_gated_qrf",
        "regime": str(fitted.regimes().get(BAND_COLUMN, "")),
        "n_estimators": int(n_estimators),
        "seed": int(seed),
        "predictors": predictors,
        "quantile_salt": BAND_SALT_QUANTILE,
        "sign_salt": BAND_SALT_SIGN,
        "rounding": f"nearest band, clipped to 0..{MAX_BAND}",
        "donor_band_shares": donor_shares,
        "donor_persons": int(len(donor)),
        "recipient_persons": int(len(recipient)),
    }
    records = (
        FitWeightRecord(
            f"{UK_NTS_BUS_TRAVEL_STAGE_NAME}:{BAND_COLUMN}", fitted.weight_kind
        ),
    )
    return BandImputation(band=band, receipt=receipt, fit_weight_records=records)


def _band_shares(bands: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    total = float(weights.sum())
    out: dict[str, float] = {}
    for ordinal, band_id in enumerate(BAND_IDS):
        mask = bands == ordinal
        out[band_id] = float(weights[mask].sum() / total) if total > 0 else 0.0
    return out


def draw_band_from_published_shares(
    recipient: pd.DataFrame,
    *,
    stage: SourceStageSpec,
    household_weights: Mapping[object, float],
    seed: int,
) -> BandImputation:
    """The fallback: the #890 draw from the vendored NTS0313 / NTS0621 shares."""

    parameters = _operation(stage, IMPUTE_BUS_USE_BAND_KIND)
    fallback = _mapping(parameters.get("fallback"), "impute_bus_use_band.fallback")
    shares, shares_receipt = nts_band_shares(fallback)
    person = pd.DataFrame(
        {
            "person_id": recipient["person_id"].to_numpy(),
            "person_household_id": recipient["person_household_id"].to_numpy(),
            "age": recipient["age"].to_numpy(),
        }
    )
    household = pd.DataFrame(
        {"household_id": sorted(set(person["person_household_id"]))}
    )
    weights = np.asarray(
        [float(household_weights[h]) for h in household["household_id"]], dtype=float
    )
    result = assign_bus_use_incidence(
        person,
        household,
        household_weights=weights,
        shares=shares,
        seed=seed,
    )
    ordinal_by_id = {band_id: ordinal for ordinal, band_id in enumerate(BAND_IDS)}
    band = np.asarray(
        [ordinal_by_id[str(b)] for b in result.person_band], dtype=np.int64
    )
    receipt = {
        "method": "published_shares",
        **{k: v for k, v in result.receipt.items() if k != "person_band_shares"},
        "shares_receipt": shares_receipt,
        "person_band_shares": result.receipt.get("person_band_shares"),
    }
    return BandImputation(band=band, receipt=receipt, fit_weight_records=())


# --------------------------------------------------------------------------
# Trips from band means
# --------------------------------------------------------------------------


def band_means_from_donor(
    donor: pd.DataFrame,
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any]]:
    """Weighted mean annual trips per person by series, residence group and band.

    A (group, band) cell with no donor persons falls back to the band's
    all-England mean, receipted; a band with no persons at all is refused.
    """

    if BAND_COLUMN not in donor:
        raise NTSBusTravelError("band means need the donor's interview band.")
    weights = donor["weight"].to_numpy(dtype=float)
    bands = donor[BAND_COLUMN].to_numpy()
    groups = donor["residence_group"].to_numpy().astype(str)
    means: dict[str, dict[str, dict[int, float]]] = {}
    fallbacks: list[dict[str, object]] = []
    for series, column in SERIES_TRIP_COLUMNS.items():
        values = donor[column].to_numpy(dtype=float)
        means[series] = {}
        for group in RESIDENCE_GROUPS:
            means[series][group] = {}
            for band in range(len(BAND_IDS)):
                in_band = bands == band
                if not in_band.any():
                    raise NTSBusTravelError(f"no donor person is in band {band}.")
                cell = in_band & (groups == group)
                if weights[cell].sum() > 0:
                    means[series][group][band] = float(
                        np.average(values[cell], weights=weights[cell])
                    )
                else:
                    means[series][group][band] = float(
                        np.average(values[in_band], weights=weights[in_band])
                    )
                    fallbacks.append({"series": series, "group": group, "band": band})
    receipt = {
        "source": "donor_diary_band_means",
        "annualisation_weeks": WEEKS_IN_YEAR,
        "means": {
            series: {
                group: {BAND_IDS[b]: v for b, v in by_band.items()}
                for group, by_band in by_group.items()
            }
            for series, by_group in means.items()
        },
        "cells_filled_from_all_england": fallbacks,
    }
    return means, receipt


def band_means_from_published_rate(
    stage: SourceStageSpec,
    *,
    band_shares: Mapping[str, float],
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any]]:
    """The fallback: declared band midpoints scaled to the vendored NTS0303 rate.

    ``sum(share_b x midpoint_b)`` is scaled to the published trips per person
    of each series; London residents are given the bus-in-London series and
    the rest the other-local-bus series (the fallback has no diary to say
    otherwise), which the receipt states.
    """

    parameters = _operation(stage, ASSIGN_TRIPS_FROM_BAND_MEANS_KIND)
    midpoints = _mapping(parameters.get("band_midpoints"), "band_midpoints")
    mid = {
        ordinal: float(midpoints[band_id])
        for ordinal, band_id in enumerate(BAND_IDS)
        if band_id in midpoints
    }
    if len(mid) != len(BAND_IDS):
        raise NTSBusTravelError("band_midpoints must name every band.")
    published = published_trip_rates(parameters)
    implied = sum(float(band_shares.get(BAND_IDS[b], 0.0)) * mid[b] for b in mid)
    if implied <= 0:
        raise NTSBusTravelError(
            "the band shares and midpoints imply no local-bus trips."
        )
    total_published = sum(published.values())
    scale = total_published / implied
    means: dict[str, dict[str, dict[int, float]]] = {series: {} for series in SERIES}
    for group in RESIDENCE_GROUPS:
        series_here = BUS_IN_LONDON if group == LONDON_GROUP else OTHER_LOCAL_BUS
        for series in SERIES:
            means[series][group] = {
                b: (mid[b] * scale if series == series_here else 0.0) for b in mid
            }
    receipt = {
        "source": "published_rate_scaled_midpoints",
        "published_trips_per_person": published,
        "implied_trips_per_person_at_midpoints": implied,
        "scale": scale,
        "series_by_residence_group": {
            LONDON_GROUP: BUS_IN_LONDON,
            REST_OF_ENGLAND_GROUP: OTHER_LOCAL_BUS,
        },
    }
    return means, receipt


def published_trip_rates(parameters: Mapping[str, Any]) -> dict[str, float]:
    """The vendored NTS0303 all-ages trips per person per year, per series."""

    resource = str(parameters.get("trip_rates_resource") or "")
    if resource != UK_NTS_TRIP_RATES_RESOURCE:
        raise NTSBusTravelError(
            f"trip_rates_resource must be {UK_NTS_TRIP_RATES_RESOURCE!r}, not {resource!r}."
        )
    period_value = int(parameters["trip_rates_period_value"])
    out: dict[str, float] = {}
    for series, concept in SERIES_TRIP_RATE_CONCEPTS.items():
        # NTS0303 rows carry the survey year as their groupby value; the
        # NTS0705a all-households and NTS0601 rows share the concept with
        # other groupby values and dimensions.
        rows = vendored_rows(
            resource,
            concept=concept,
            period_type="calendar_year",
            period_value=period_value,
            groupby_value_id=f"cy{period_value}",
            dimensions={},
        )
        if len(rows) != 1:
            raise NTSBusTravelError(
                f"{resource}: expected one NTS0303 row for {concept} {period_value}, "
                f"found {len(rows)}."
            )
        out[series] = float(rows[0]["value"])
    return out


def assign_trips_from_band_means(
    band: np.ndarray,
    residence_group: np.ndarray,
    means: Mapping[str, Mapping[str, Mapping[int, float]]],
) -> dict[str, np.ndarray]:
    """Each person's annual trips per series from their band and residence group."""

    out: dict[str, np.ndarray] = {}
    groups = np.asarray(residence_group).astype(str)
    for series in SERIES:
        values = np.zeros(len(band), dtype=float)
        for group in RESIDENCE_GROUPS:
            by_band = means[series][group]
            lookup = np.asarray(
                [by_band[int(b)] for b in range(len(BAND_IDS))], dtype=float
            )
            mask = groups == group
            values[mask] = lookup[np.asarray(band)[mask]]
        out[SERIES_TRIP_COLUMNS[series]] = values
    out[TRIPS_COLUMN] = (
        out[SERIES_TRIP_COLUMNS[BUS_IN_LONDON]]
        + out[SERIES_TRIP_COLUMNS[OTHER_LOCAL_BUS]]
    )
    return out


# --------------------------------------------------------------------------
# Concessionary eligibility
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EligibilityRule:
    regions: tuple[str, ...]
    min_age: int | None
    max_age_exclusive: int | None

    def applies(self, ages: np.ndarray) -> np.ndarray:
        eligible = np.zeros(len(ages), dtype=bool)
        if self.min_age is not None:
            eligible |= ages >= self.min_age
        if self.max_age_exclusive is not None:
            eligible |= ages < self.max_age_exclusive
        return eligible


def eligibility_rules(parameters: Mapping[str, Any]) -> dict[str, EligibilityRule]:
    """The declared statutory rules, keyed by area label."""

    rules = _mapping(parameters.get("rules"), "assign_bus_pass_eligibility.rules")
    out: dict[str, EligibilityRule] = {}
    seen: set[str] = set()
    for label, rule in rules.items():
        rule = _mapping(rule, f"rules.{label}")
        regions = tuple(str(r) for r in rule.get("regions", ()))
        if not regions:
            raise NTSBusTravelError(f"rules.{label} names no regions.")
        dup = seen & set(regions)
        if dup:
            raise NTSBusTravelError(f"rules.{label} repeats regions {sorted(dup)}.")
        seen |= set(regions)
        min_age = rule.get("min_age")
        max_age = rule.get("max_age_exclusive")
        if min_age is None and max_age is None:
            raise NTSBusTravelError(f"rules.{label} declares no age bound.")
        out[str(label)] = EligibilityRule(
            regions=regions,
            min_age=None if min_age is None else int(min_age),
            max_age_exclusive=None if max_age is None else int(max_age),
        )
    return out


def assign_bus_pass_eligibility(
    ages: np.ndarray, regions: np.ndarray, rules: Mapping[str, EligibilityRule]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Eligibility per person from the rule of the person's residence region."""

    ages = np.asarray(ages, dtype=float)
    regions = np.asarray(regions).astype(str)
    eligible = np.zeros(len(ages), dtype=bool)
    covered = np.zeros(len(ages), dtype=bool)
    by_area: dict[str, dict[str, object]] = {}
    for label, rule in rules.items():
        mask = np.isin(regions, rule.regions)
        eligible[mask] = rule.applies(ages[mask])
        covered |= mask
        by_area[label] = {
            "regions": list(rule.regions),
            "min_age": rule.min_age,
            "max_age_exclusive": rule.max_age_exclusive,
            "persons": int(mask.sum()),
            "eligible": int(eligible[mask].sum()),
        }
    uncovered = sorted(set(regions[~covered]))
    if uncovered:
        raise NTSBusTravelError(f"no eligibility rule covers regions {uncovered}.")
    return eligible, {"by_area": by_area, "eligible_persons": int(eligible.sum())}


# --------------------------------------------------------------------------
# Fact checks (published values from vendored rows)
# --------------------------------------------------------------------------


def published_band_shares(
    stage: SourceStageSpec,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """The vendored NTS0313 (all ages) and NTS0621 (60+) band shares, as fractions.

    Read through the stage's declared fallback block (the same declaration the
    published-shares draw uses), so the fact check and the fallback agree on
    the rows.
    """

    parameters = _operation(stage, IMPUTE_BUS_USE_BAND_KIND)
    fallback = _mapping(parameters.get("fallback"), "impute_bus_use_band.fallback")
    shares, receipt = nts_band_shares(fallback)
    return dict(shares.all_ages), dict(shares.older), receipt


def published_car_availability(parameters: Mapping[str, Any]) -> dict[str, float]:
    """The vendored NTS0205 England shares (fractions) for the declared year."""

    resource = str(parameters.get("car_availability_resource") or "")
    if resource != UK_NTS_CAR_AVAILABILITY_RESOURCE:
        raise NTSBusTravelError(
            f"car_availability_resource must be {UK_NTS_CAR_AVAILABILITY_RESOURCE!r}."
        )
    period_value = int(parameters["car_availability_period_value"])
    out: dict[str, float] = {}
    for key, concept in (
        ("no_car", "dft.households_no_car_share"),
        ("one_car", "dft.households_one_car_share"),
        ("two_plus_cars", "dft.households_two_plus_cars_share"),
    ):
        rows = vendored_rows(
            resource,
            concept=concept,
            geography_id="E92000001",
            period_type="calendar_year",
            period_value=period_value,
        )
        if len(rows) != 1:
            raise NTSBusTravelError(
                f"{resource}: expected one NTS0205 row for {concept} {period_value}."
            )
        out[key] = float(rows[0]["value"]) / 100.0
    return out


def frame_car_availability(
    household: pd.DataFrame, weights: np.ndarray, *, england_regions: Sequence[str]
) -> dict[str, float]:
    """The frame's design-weighted no / one / two-plus car shares over England households."""

    region = household["region"].map(_enum_name).to_numpy().astype(str)
    cars = (
        pd.to_numeric(household["num_vehicles"], errors="coerce").fillna(0.0).to_numpy()
    )
    mask = np.isin(region, list(england_regions))
    w = np.asarray(weights, dtype=float)[mask]
    c = cars[mask]
    total = float(w.sum())
    if total <= 0:
        raise NTSBusTravelError("no England households to compare car availability on.")
    return {
        "no_car": float(w[c <= 0].sum() / total),
        "one_car": float(w[c == 1].sum() / total),
        "two_plus_cars": float(w[c >= 2].sum() / total),
    }


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


@dataclass
class UKNTSBusTravelResult:
    frame: Frame
    support_clip: UKSupportClipReceipt
    donor: Mapping[str, Any]
    band: Mapping[str, Any]
    incidence: Mapping[str, Any]
    band_means: Mapping[str, Any]
    trip_rates: Mapping[str, Any]
    eligibility: Mapping[str, Any]
    vehicle_shares: Mapping[str, Any]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_NTS_BUS_TRAVEL_STAGE_NAME,
            "support_clip": self.support_clip.evidence(),
            "donor": dict(self.donor),
            "band": dict(self.band),
            "incidence": dict(self.incidence),
            "band_means": dict(self.band_means),
            "trip_rates": dict(self.trip_rates),
            "eligibility": dict(self.eligibility),
            "vehicle_shares": dict(self.vehicle_shares),
        }


@dataclass
class UKNTSBusTravelStageTransform:
    """Whole-stage callable for the NTS local-bus travel imputation."""

    stage: SourceStageSpec
    engine: object
    nts_household_tab_path: str | Path | None = None
    nts_individual_tab_path: str | Path | None = None
    nts_trip_tab_path: str | Path | None = None
    nts_household: pd.DataFrame | None = None
    nts_individual: pd.DataFrame | None = None
    nts_trip: pd.DataFrame | None = None
    last_result: UKNTSBusTravelResult | None = field(default=None, init=False)
    last_fit_weight_records: tuple[FitWeightRecord, ...] | None = field(
        default=None, init=False
    )

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        if self.last_fit_weight_records is None:
            return ()
        return tuple(self.last_fit_weight_records)

    def _table(self, supplied: pd.DataFrame | None, path: str | Path | None, role: str):
        if supplied is not None:
            return supplied
        if path is None:
            raise NTSBusTravelError(
                f"{UK_NTS_BUS_TRAVEL_STAGE_NAME} requires the caller-supplied {role} tab."
            )
        return read_pinned_tab(
            Path(path).expanduser().resolve(), _artifact(self.stage, role)
        )

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        derive = _operation(self.stage, CLEAN_NTS_TRAVEL_TABLES_KIND)
        columns = NTSColumns.from_parameters(derive)
        cars_clip = tuple(int(v) for v in derive.get("cars_clip", (0, 5)))
        donor = clean_nts_travel_tables(
            self._table(
                self.nts_household,
                self.nts_household_tab_path,
                UK_NTS_HOUSEHOLD_TAB_ROLE,
            ),
            self._table(
                self.nts_individual,
                self.nts_individual_tab_path,
                UK_NTS_INDIVIDUAL_TAB_ROLE,
            ),
            self._table(self.nts_trip, self.nts_trip_tab_path, UK_NTS_TRIP_TAB_ROLE),
            columns=columns,
            cars_clip=(cars_clip[0], cars_clip[1]),
        )
        band_parameters = _operation(self.stage, IMPUTE_BUS_USE_BAND_KIND)
        income_edges = [float(v) for v in band_parameters["income_band_edges"]]
        region_remap = {
            str(k): str(v)
            for k, v in _mapping(
                band_parameters.get("region_remap", {}), "region_remap"
            ).items()
        }
        recipient = recipient_predictors(
            frame,
            self.engine,
            columns=columns,
            income_band_edges=income_edges,
            region_remap=region_remap,
            cars_clip=(cars_clip[0], cars_clip[1]),
        )
        seed = int(band_parameters.get("seed", 0))
        household_weights = frame.weights_for("household").values
        household_table = frame.table("household")
        weight_by_household = dict(
            zip(household_table["household_id"], household_weights, strict=True)
        )
        person_weights = np.asarray(
            [weight_by_household[h] for h in recipient["person_household_id"]],
            dtype=float,
        )
        if donor.frequency_source == FREQUENCY_SOURCE_INTERVIEW:
            imputation = impute_bus_use_band(
                donor.person,
                recipient,
                seed=seed,
                n_estimators=int(band_parameters.get("n_estimators", 100)),
            )
            means, means_receipt = band_means_from_donor(donor.person)
        else:
            imputation = draw_band_from_published_shares(
                recipient,
                stage=self.stage,
                household_weights=weight_by_household,
                seed=seed,
            )
            shares = _band_shares(imputation.band, person_weights)
            means, means_receipt = band_means_from_published_rate(
                self.stage, band_shares=shares
            )
        band = imputation.band
        trips = assign_trips_from_band_means(
            band, recipient["residence_group"].to_numpy(), means
        )
        rules = eligibility_rules(
            _operation(self.stage, ASSIGN_BUS_PASS_ELIGIBILITY_KIND)
        )
        eligible, eligibility_receipt = assign_bus_pass_eligibility(
            recipient["age"].to_numpy(), recipient["residence_region"].to_numpy(), rules
        )
        # Donor-side eligible share of trips per series (the fare-paying share
        # the pricing step reads beside the publisher's concessionary share).
        donor_regions = donor.person["region"].to_numpy().astype(str)
        donor_ages = _band_lower_ages(
            donor.person["age_band"].to_numpy(), columns.age_band_edges
        )
        donor_eligible, _ = assign_bus_pass_eligibility(
            donor_ages, donor_regions, rules
        )
        donor_w = donor.person["weight"].to_numpy(dtype=float)
        eligible_trip_share = {}
        for series, column in SERIES_TRIP_COLUMNS.items():
            v = donor.person[column].to_numpy(dtype=float) * donor_w
            eligible_trip_share[series] = (
                float(v[donor_eligible].sum() / v.sum()) if v.sum() > 0 else None
            )
        eligibility_receipt["donor_eligible_trip_share"] = eligible_trip_share
        eligibility_receipt["donor_age_basis"] = (
            "band lower bound (the extract carries banded ages)"
        )
        # Person-level support clip on the band and the trips.
        draws = pd.DataFrame({BAND_COLUMN: band.astype(float), **trips})
        donor_support = pd.DataFrame(
            {
                BAND_COLUMN: donor.person[BAND_COLUMN].to_numpy(dtype=float)
                if BAND_COLUMN in donor.person
                else np.arange(len(BAND_IDS), dtype=float),
                **{
                    column: np.asarray(
                        [
                            v
                            for group in RESIDENCE_GROUPS
                            for v in means[series][group].values()
                        ],
                        dtype=float,
                    )
                    for series, column in SERIES_TRIP_COLUMNS.items()
                },
            }
            if BAND_COLUMN not in donor.person
            else donor.person[
                [BAND_COLUMN, *SERIES_TRIP_COLUMNS.values(), TRIPS_COLUMN]
            ]
        )
        if TRIPS_COLUMN not in donor_support:
            donor_support[TRIPS_COLUMN] = (
                donor_support[SERIES_TRIP_COLUMNS[BUS_IN_LONDON]]
                + donor_support[SERIES_TRIP_COLUMNS[OTHER_LOCAL_BUS]]
            )
        clip = support_clip_to_donor_with_receipt(
            draws,
            donor_support,
            columns=UK_NTS_SUPPORT_CLIP_COLUMNS,
            stage=UK_NTS_BUS_TRAVEL_STAGE_NAME,
        )
        clipped = clip.clipped
        # Outputs.
        person = frame.table("person").copy()
        person[BAND_COLUMN] = clipped[BAND_COLUMN].to_numpy().astype(np.int64)
        for column in (
            SERIES_TRIP_COLUMNS[BUS_IN_LONDON],
            SERIES_TRIP_COLUMNS[OTHER_LOCAL_BUS],
            TRIPS_COLUMN,
        ):
            person[column] = clipped[column].to_numpy(dtype=float)
        person[ELIGIBLE_COLUMN] = eligible
        household = household_table.copy()
        by_household = (
            pd.Series(
                person[TRIPS_COLUMN].to_numpy(),
                index=person["person_household_id"].to_numpy(),
            )
            .groupby(level=0)
            .sum()
        )
        household[HOUSEHOLD_TRIPS_COLUMN] = (
            by_household.reindex(household["household_id"])
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        # Evidence.
        recipient_shares = _band_shares(person[BAND_COLUMN].to_numpy(), person_weights)
        user_share = 1.0 - recipient_shares[BAND_IDS[NON_USER_BAND]]
        older = recipient["age"].to_numpy() >= 60
        older_shares = _band_shares(
            person[BAND_COLUMN].to_numpy()[older], person_weights[older]
        )
        incidence = {
            "frequency_source": donor.frequency_source,
            "person_band_shares": recipient_shares,
            "person_user_share": user_share,
            "aged_60_and_over_band_shares": older_shares,
            "aged_60_and_over_user_share": 1.0 - older_shares[BAND_IDS[NON_USER_BAND]],
            "household_user_share": float(
                np.asarray(household_weights)[
                    household["household_id"]
                    .isin(
                        person.loc[
                            person[BAND_COLUMN] > NON_USER_BAND, "person_household_id"
                        ]
                    )
                    .to_numpy()
                ].sum()
                / float(np.asarray(household_weights).sum())
            ),
        }
        trip_rates = _trip_rate_receipt(
            person, recipient, person_weights, band_parameters
        )
        vehicle_shares = {
            "frame": frame_car_availability(
                household_table,
                household_weights,
                england_regions=[r for r in columns.region_codes.values()],
            ),
            "published": published_car_availability(band_parameters),
            "basis": "design-weighted England households; num_vehicles is the was_wealth draw",
        }
        result = uk_national_frame(
            person=person,
            benunit=frame.table("benunit").copy(),
            household=household,
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=household_weights,
            mass_log=frame.mass_log,
        )
        validate_uk_national_frame(result)
        self.last_fit_weight_records = imputation.fit_weight_records
        self.last_result = UKNTSBusTravelResult(
            frame=result,
            support_clip=clip.receipt,
            donor=donor.receipt,
            band=imputation.receipt,
            incidence=incidence,
            band_means=means_receipt,
            trip_rates=trip_rates,
            eligibility=eligibility_receipt,
            vehicle_shares=vehicle_shares,
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_NTS_BUS_TRAVEL_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            return {}
        return {"evidence": self.last_result.evidence()}


def _band_lower_ages(age_band: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    lower = np.asarray(edges, dtype=float)
    return lower[np.clip(np.asarray(age_band, dtype=int), 0, len(lower) - 1)]


def _trip_rate_receipt(
    person: pd.DataFrame,
    recipient: pd.DataFrame,
    weights: np.ndarray,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    published = published_trip_rates(parameters)
    frame_rates = {
        series: float(np.average(person[column].to_numpy(dtype=float), weights=weights))
        for series, column in SERIES_TRIP_COLUMNS.items()
    }
    by_age: dict[str, dict[str, float]] = {}
    ages = recipient["age"].to_numpy(dtype=float)
    for label, lower, upper in (
        ("0 to 16", 0, 17),
        ("17 to 20", 17, 21),
        ("21 to 29", 21, 30),
        ("30 to 39", 30, 40),
        ("40 to 49", 40, 50),
        ("50 to 59", 50, 60),
        ("60 to 69", 60, 70),
        ("70 and over", 70, 10_000),
    ):
        mask = (ages >= lower) & (ages < upper)
        if weights[mask].sum() > 0:
            by_age[label] = {
                series: float(
                    np.average(
                        person[column].to_numpy(dtype=float)[mask],
                        weights=weights[mask],
                    )
                )
                for series, column in SERIES_TRIP_COLUMNS.items()
            }
    return {
        "basis": "design-weighted persons of the frame",
        "frame_trips_per_person": frame_rates,
        "published_trips_per_person": published,
        "published_period_value": int(parameters["trip_rates_period_value"]),
        "relative_deviation": {
            series: (frame_rates[series] / published[series] - 1.0)
            if published[series]
            else None
            for series in SERIES
        },
        "frame_trips_per_person_by_age_band": by_age,
    }
