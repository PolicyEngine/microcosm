"""A built UK file's single years of age against ONS's mid-2025 estimates.

uk-data#600 checks each build's population by single year of age against
ONS, which nothing in Microcosm measured (microcosm#1095). The report reads a
built file's person table and final household weights (no engine) and gives,
for each sex and each single year of age (0 to 89, and 90 and over), the
file's weighted persons beside ONS's mid-2025 UK estimate (MYE2, vendored in
``ons_mid2025_single_year_age_by_sex.json``) and their ratio, and the same for
uk-data#600's tail bands (80 to 84, 85 to 89, 90 and over) and for 80 and
over. The estimates are report-only: no stage reads them and no target binds
them.

ONS counts usual residents, communal establishments included; the frame holds
private households, so the file runs below ONS where many people live in care
homes or halls of residence. The FRS's own ages stop at 80 (AGE80): the
age_tail stage draws each such person's band from ONS's 80 to 84, 85 to 89
and 90 and over, with a uniform age within the band, so the single years above
80 show that draw. Every cell names its carriers as
distinct source households and reads "<10" in place of its figures when fewer
than 10 carry it, and a band that holds a withheld age reads "<10" too, so no
difference gives a withheld age back (UKDS disclosure control).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.diagnostics import UK_DIAGNOSTIC_MIN_SOURCE_HOUSEHOLDS
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

UK_SINGLE_YEAR_AGE_RESOURCE = "ons_mid2025_single_year_age_by_sex.json"
UK_SINGLE_YEAR_AGE_PERIOD = 2025
#: ONS's open age: the last single-year cell is 90 and over.
UK_OPEN_AGE = 90
#: uk-data#600's tail bands, the age_tail stage's draw bands, then 80 and over,
#: the FRS's top code (AGE80).
UK_TAIL_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (80, 84),
    (85, 89),
    (90, 90),
    (80, 90),
)
UK_SINGLE_YEAR_AGE_SEXES = ("FEMALE", "MALE")
#: The columns that identify the spine household a row was copied from, in
#: the full build's order of preference (uk_runtime.full_measure).
_SOURCE_IDENTITY_COLUMNS = (
    "household_source_id",
    "source_household_id",
    "source_household_key",
)


def _age_key(age: int) -> str:
    return "90_plus" if age == UK_OPEN_AGE else str(age)


def _band_key(lower: int, upper: int) -> str:
    return f"{lower}_plus" if upper == UK_OPEN_AGE else f"{lower}_{upper}"


def load_ons_single_year_ages() -> dict[tuple[str, int], float]:
    """ONS's mid-2025 UK persons by sex and single year (90 is 90 and over)."""

    populations: dict[tuple[str, int], float] = {}
    for row in vendored_rows(
        UK_SINGLE_YEAR_AGE_RESOURCE, period_value=str(UK_SINGLE_YEAR_AGE_PERIOD)
    ):
        dimensions = row.get("dimensions") or {}
        raw_age = str(dimensions.get("age", ""))
        key = (
            str(dimensions.get("sex", "")).upper(),
            UK_OPEN_AGE if raw_age == "90_plus" else int(raw_age),
        )
        if key in populations:
            raise ValueError(f"{UK_SINGLE_YEAR_AGE_RESOURCE} repeats {key!r}.")
        populations[key] = float(row["value"])
    expected = {
        (sex, age) for sex in UK_SINGLE_YEAR_AGE_SEXES for age in range(UK_OPEN_AGE + 1)
    }
    if set(populations) != expected:
        missing = sorted(expected - set(populations))[:5]
        raise ValueError(
            f"{UK_SINGLE_YEAR_AGE_RESOURCE} must carry ages 0 to 90 for both sexes "
            f"at mid-{UK_SINGLE_YEAR_AGE_PERIOD}; missing {missing}."
        )
    return populations


def household_source_ids(household: pd.DataFrame) -> np.ndarray:
    """Each household row's source household, which every copy of it shares."""

    for column in _SOURCE_IDENTITY_COLUMNS:
        if column in household:
            return household[column].to_numpy()
    raise ValueError(
        "The single-year age report counts each cell's carriers as source "
        f"households and needs one of {list(_SOURCE_IDENTITY_COLUMNS)}."
    )


def single_year_age_report(
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
    *,
    ons: Mapping[tuple[str, int], float] | None = None,
    minimum_carriers: int = UK_DIAGNOSTIC_MIN_SOURCE_HOUSEHOLDS,
) -> dict[str, Any]:
    """The file's persons by sex and single year of age beside ONS's."""

    ons = load_ons_single_year_ages() if ons is None else ons
    weights = np.asarray(household_weights, dtype=float)
    if weights.shape != (len(household),):
        raise ValueError("The report needs one final weight per household row.")
    position = pd.Index(household["household_id"]).get_indexer(
        person["person_household_id"]
    )
    if (position < 0).any():
        raise ValueError("Every person must belong to a household row.")
    person_weight = weights[position]
    person_source = household_source_ids(household)[position]
    raw_age = person["age"].to_numpy(dtype=float)
    if not np.isfinite(raw_age).all() or (raw_age < 0).any():
        raise ValueError("Every person needs a finite, non-negative age.")
    age = np.minimum(np.floor(raw_age), UK_OPEN_AGE).astype(int)
    sex = person["gender"].astype(str).str.upper().to_numpy()
    withheld = f"<{minimum_carriers}"

    def cell(mask: np.ndarray, ons_value: float) -> dict[str, Any]:
        carriers = int(np.unique(person_source[mask]).size)
        if carriers < minimum_carriers:
            return {"source_households": withheld}
        persons = float(person_weight[mask].sum())
        return {
            "source_households": carriers,
            "weighted_persons": persons,
            "ons_persons": ons_value,
            "ratio_to_ons": persons / ons_value,
        }

    by_sex: dict[str, Any] = {}
    for label in UK_SINGLE_YEAR_AGE_SEXES:
        of_sex = sex == label
        single = {
            _age_key(year): cell(of_sex & (age == year), ons[(label, year)])
            for year in range(UK_OPEN_AGE + 1)
        }
        bands: dict[str, Any] = {}
        for lower, upper in UK_TAIL_AGE_BANDS:
            years = range(lower, upper + 1)
            if any(
                single[_age_key(year)]["source_households"] == withheld
                for year in years
            ):
                # A band beside its shown ages would give a withheld one back.
                bands[_band_key(lower, upper)] = {"source_households": withheld}
                continue
            bands[_band_key(lower, upper)] = cell(
                of_sex & (age >= lower) & (age <= upper),
                sum(ons[(label, year)] for year in years),
            )
        by_sex[label.lower()] = {"single_years": single, "tail_bands": bands}
    return {
        "schema_version": 1,
        "ons_resource": UK_SINGLE_YEAR_AGE_RESOURCE,
        "ons_period": f"mid-{UK_SINGLE_YEAR_AGE_PERIOD}",
        "basis": (
            "ONS counts usual residents including communal establishments; the "
            "file holds private households. The FRS's ages stop at 80 (AGE80); "
            "the age_tail stage draws ages above it by ONS band, uniform within "
            "the band."
        ),
        "minimum_source_households": minimum_carriers,
        "sexes": by_sex,
    }


__all__ = [
    "UK_OPEN_AGE",
    "UK_SINGLE_YEAR_AGE_PERIOD",
    "UK_SINGLE_YEAR_AGE_RESOURCE",
    "UK_SINGLE_YEAR_AGE_SEXES",
    "UK_TAIL_AGE_BANDS",
    "household_source_ids",
    "load_ons_single_year_ages",
    "single_year_age_report",
]
