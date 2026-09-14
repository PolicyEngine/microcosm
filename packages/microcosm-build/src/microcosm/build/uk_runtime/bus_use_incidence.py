"""Local-bus use incidence calibrated to NTS frequency of use (microcosm#890 I).

The LCFS diary under-records who buys bus fares (a two-week window against an
annual concept) and the consumption QRF reproduces that. This module decides
*who* uses local buses take-up style, from the vendored DfT National Travel
Survey frequency-of-use shares (NTS0313 for all ages, NTS0621 for people aged
60 and over), before the amount side is drawn and raked:

* each person is assigned a frequency band by an identity-keyed uniform draw
  from the age-specific band shares (under-60 shares are derived from the two
  published series and the frame's design-weighted 60-and-over population
  share, since DfT publishes no other breakdown);
* a household uses local buses when any member is in a user band (the declared
  ``user_definition``, at least once a year);
* each band carries declared trips-per-year weights (band midpoints, a ruling
  default) so the fare rake can spread an area's receipts over user households
  in proportion to their members' trips.

Everything read is returned in a receipt the stage records as evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

ASSIGN_BUS_USE_INCIDENCE_KIND = "assign_bus_use_incidence"
UK_NTS_BUS_USE_FREQUENCY_RESOURCE = "nts_bus_use_frequency.json"
NTS_MODE_USE_FREQUENCY_SHARE_CONCEPT = "dft.nts.mode_use_frequency_share"
USER_DEFINITION_AT_LEAST_ONCE_A_YEAR = "at_least_once_a_year"
PERSON_DRAW_SALT = "lcfs_uses_local_bus"
_SHARE_SUM_TOLERANCE = 0.2  # percentage points; the published rows are rounded


@dataclass(frozen=True)
class BusUseBandShares:
    """Band shares (fractions) per age group, in declared band order."""

    band_ids: tuple[str, ...]
    trips_per_year: Mapping[str, float]
    all_ages: Mapping[str, float]
    older: Mapping[str, float]
    non_user_band: str
    older_age_band: str
    age_threshold: int


@dataclass(frozen=True)
class BusUseIncidenceResult:
    household_user: np.ndarray
    household_trips_per_year: np.ndarray
    person_band: np.ndarray
    receipt: dict[str, Any]


def incidence_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    """The stage's declared ``assign_bus_use_incidence`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == ASSIGN_BUS_USE_INCIDENCE_KIND:
            return dict(operation.parameters)
    return None


def nts_band_shares(parameters: Mapping[str, Any]) -> tuple[BusUseBandShares, dict]:
    """Read both NTS series from the vendored resource under the declaration.

    Bands are joined by the declared ``labels`` per band id (the two tables
    word two of the bands differently), never by position or by string
    equality between the tables. Every published row must map to a declared
    band and every declared band must appear in both series; each series must
    sum to 100 within rounding.
    """

    resource = str(parameters.get("resource") or "")
    if resource != UK_NTS_BUS_USE_FREQUENCY_RESOURCE:
        raise ValueError(
            "assign_bus_use_incidence must read "
            f"{UK_NTS_BUS_USE_FREQUENCY_RESOURCE!r}, not {resource!r}."
        )
    if parameters.get("user_definition", USER_DEFINITION_AT_LEAST_ONCE_A_YEAR) != (
        USER_DEFINITION_AT_LEAST_ONCE_A_YEAR
    ):
        raise ValueError("unsupported bus user_definition.")
    period_value = int(parameters["period_value"])
    mode_groupby = str(parameters.get("transport_mode_groupby_value_id") or "local_bus")
    older_age_band = str(parameters.get("older_age_band") or "60 and over")
    age_threshold = int(parameters["age_threshold"])
    non_user_band = str(parameters["non_user_band"])
    bands = list(parameters["bands"])
    band_ids = tuple(str(band["id"]) for band in bands)
    if len(set(band_ids)) != len(band_ids):
        raise ValueError("assign_bus_use_incidence declares duplicate band ids.")
    if non_user_band not in band_ids:
        raise ValueError(f"non_user_band {non_user_band!r} is not a declared band.")
    label_to_band: dict[str, str] = {}
    trips: dict[str, float] = {}
    for band in bands:
        trips[str(band["id"])] = float(band["trips_per_year"])
        for label in band["labels"]:
            if str(label) in label_to_band:
                raise ValueError(f"band label {label!r} declared twice.")
            label_to_band[str(label)] = str(band["id"])
    if trips[non_user_band] != 0.0:
        raise ValueError("the non-user band must carry zero trips per year.")
    if any(value < 0 for value in trips.values()):
        raise ValueError("trips_per_year must be nonnegative.")

    all_rows = [
        row
        for row in vendored_rows(
            resource,
            concept=NTS_MODE_USE_FREQUENCY_SHARE_CONCEPT,
            period_type="calendar_year",
            period_value=period_value,
            groupby_value_id=mode_groupby,
        )
        if "age_band" not in (row.get("dimensions") or {})
    ]
    older_rows = vendored_rows(
        resource,
        concept=NTS_MODE_USE_FREQUENCY_SHARE_CONCEPT,
        period_type="calendar_year",
        period_value=period_value,
        dimensions={"age_band": older_age_band},
    )
    if not all_rows or not older_rows:
        raise ValueError(f"{resource}: missing an NTS series for {period_value}.")
    all_ages, all_ids = _series_from_rows(all_rows, label_to_band, band_ids, resource)
    older, older_ids = _series_from_rows(older_rows, label_to_band, band_ids, resource)
    shares = BusUseBandShares(
        band_ids=band_ids,
        trips_per_year=trips,
        all_ages=all_ages,
        older=older,
        non_user_band=non_user_band,
        older_age_band=older_age_band,
        age_threshold=age_threshold,
    )
    receipt = {
        "resource": resource,
        "period_value": period_value,
        "transport_mode_groupby_value_id": mode_groupby,
        "older_age_band": older_age_band,
        "age_threshold": age_threshold,
        "user_definition": USER_DEFINITION_AT_LEAST_ONCE_A_YEAR,
        "non_user_band": non_user_band,
        "bands": [{"id": band, "trips_per_year": trips[band]} for band in band_ids],
        "all_ages_shares": dict(all_ages),
        "older_shares": dict(older),
        "all_ages_user_share": 1.0 - all_ages[non_user_band],
        "older_user_share": 1.0 - older[non_user_band],
        "source_record_ids": [*all_ids, *older_ids],
    }
    return shares, receipt


def _series_from_rows(
    rows: Sequence[Mapping[str, Any]],
    label_to_band: Mapping[str, str],
    band_ids: Sequence[str],
    resource: str,
) -> tuple[dict[str, float], list[str]]:
    shares: dict[str, float] = {}
    record_ids: list[str] = []
    for row in rows:
        dimensions = row.get("dimensions") or {}
        if "use_frequency_band" not in dimensions:
            continue
        label = str(dimensions["use_frequency_band"])
        if label not in label_to_band:
            raise ValueError(f"{resource}: published band {label!r} is not declared.")
        band_id = label_to_band[label]
        if band_id in shares:
            raise ValueError(f"{resource}: band {band_id!r} appears twice.")
        if str(row.get("unit")) != "percent":
            raise ValueError(f"{resource}: expected percent rows.")
        shares[band_id] = float(row["value"]) / 100.0
        record_ids.append(str(row.get("source_record_id", "")))
    missing = [band for band in band_ids if band not in shares]
    if missing:
        raise ValueError(f"{resource}: series lacks declared bands {missing}.")
    total = sum(shares.values()) * 100.0
    if abs(total - 100.0) > _SHARE_SUM_TOLERANCE:
        raise ValueError(f"{resource}: band shares sum to {total:.3f}, not 100.")
    return shares, record_ids


def under_threshold_shares(
    shares: BusUseBandShares, *, older_population_share: float
) -> dict[str, float]:
    """Derive the under-threshold band shares from the two published series.

    p_all = s * p_older + (1 - s) * p_under, with s the design-weighted share
    of people at or above the age threshold on the frame being drawn; DfT
    publishes no under-60 table. Refuses a derived share outside [0, 1].
    """

    s = float(older_population_share)
    if not 0.0 <= s < 1.0:
        raise ValueError("older population share must lie in [0, 1).")
    under: dict[str, float] = {}
    for band in shares.band_ids:
        value = (shares.all_ages[band] - s * shares.older[band]) / (1.0 - s)
        if value < -1e-9 or value > 1.0 + 1e-9:
            raise ValueError(
                f"derived under-{shares.age_threshold} share for {band!r} is "
                f"{value:.4f}; the published series are inconsistent with an "
                f"older population share of {s:.4f}."
            )
        under[band] = float(min(max(value, 0.0), 1.0))
    total = sum(under.values())
    if abs(total - 1.0) > _SHARE_SUM_TOLERANCE / 100.0:
        raise ValueError(f"derived under-threshold shares sum to {total:.4f}.")
    return under


def assign_bus_use_incidence(
    person: pd.DataFrame,
    household: pd.DataFrame,
    *,
    household_weights: Sequence[float],
    shares: BusUseBandShares,
    seed: int,
    salt: str = PERSON_DRAW_SALT,
) -> BusUseIncidenceResult:
    """Draw a frequency band per person and roll up to household incidence.

    ``person`` needs ``person_id``, ``person_household_id`` and ``age``;
    ``household`` needs ``household_id`` aligned with ``household_weights``.
    """

    for column in ("person_id", "person_household_id", "age"):
        if column not in person:
            raise KeyError(f"person table is missing {column!r}.")
    if "household_id" not in household:
        raise KeyError("household table is missing 'household_id'.")
    weights = np.asarray(household_weights, dtype=float)
    if len(weights) != len(household):
        raise ValueError("household_weights must align with the household table.")
    household_ids = household["household_id"].to_numpy()
    weight_by_household = pd.Series(weights, index=household_ids)
    person_weights = (
        person["person_household_id"].map(weight_by_household).fillna(0.0).to_numpy()
    )
    ages = pd.to_numeric(person["age"], errors="coerce").fillna(0.0).to_numpy()
    older = ages >= shares.age_threshold
    total_weight = float(person_weights.sum())
    older_share = (
        float(person_weights[older].sum() / total_weight) if total_weight > 0 else 0.0
    )
    under = under_threshold_shares(shares, older_population_share=older_share)
    uniforms = stable_identity_uniforms(
        person["person_id"].to_numpy(), seed=seed, salt=salt
    )
    band_index = np.empty(len(person), dtype=np.int64)
    for mask, probabilities in (
        (older, [shares.older[band] for band in shares.band_ids]),
        (~older, [under[band] for band in shares.band_ids]),
    ):
        if not mask.any():
            continue
        edges = np.cumsum(np.asarray(probabilities, dtype=float))
        edges[-1] = max(edges[-1], 1.0)
        band_index[mask] = np.searchsorted(edges, uniforms[mask], side="right")
    band_index = np.clip(band_index, 0, len(shares.band_ids) - 1)
    band_ids = np.asarray(shares.band_ids, dtype=object)
    person_band = band_ids[band_index]
    trips = np.asarray(
        [shares.trips_per_year[band] for band in person_band], dtype=float
    )
    person_user = person_band != shares.non_user_band
    by_household = pd.DataFrame(
        {
            "household": person["person_household_id"].to_numpy(),
            "user": person_user,
            "trips": trips,
        }
    ).groupby("household")
    user_any = by_household["user"].any()
    trips_sum = by_household["trips"].sum()
    household_user = user_any.reindex(household_ids).fillna(False).to_numpy(dtype=bool)
    household_trips = trips_sum.reindex(household_ids).fillna(0.0).to_numpy(dtype=float)
    receipt = {
        "seed": int(seed),
        "salt": salt,
        "persons": int(len(person)),
        "older_population_share": older_share,
        "under_threshold_shares": under,
        "person_user_share": _weighted_share(person_user, person_weights),
        "person_user_share_older": _weighted_share(
            person_user[older], person_weights[older]
        ),
        "person_user_share_under": _weighted_share(
            person_user[~older], person_weights[~older]
        ),
        "household_user_share": _weighted_share(household_user, weights),
        "person_band_shares": {
            band: _weighted_share(person_band == band, person_weights)
            for band in shares.band_ids
        },
        "mean_trips_per_person": (
            float(np.dot(trips, person_weights) / total_weight)
            if total_weight > 0
            else 0.0
        ),
    }
    return BusUseIncidenceResult(
        household_user=household_user,
        household_trips_per_year=household_trips,
        person_band=person_band,
        receipt=receipt,
    )


def _weighted_share(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    return float(np.sum(weights[np.asarray(mask, dtype=bool)]) / total)
