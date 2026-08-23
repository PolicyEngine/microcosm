"""Disaggregate the FRS age top-code so 85+ population targets can bind.

The licensed FRS delivery records no age above 80: the raw ``AGE`` column is
blanked and ``age80`` caps at 80, so every person aged 80 or older arrives as
exactly 80. That single pile produces two distortions at once — the 85-89 and
90+ population targets are structurally unbindable (estimate 0), and the
80-84 band starts roughly double its target because it carries the entire
80+ population.

This stage reassigns each piled person an age drawn from the ONS mid-year
band populations (the same chronicle facts the calibration targets bind, so
the imputation source and the target denominators cannot drift apart). The
draw is:

- keyed on ``person_source_id``, so a household and its capital-gains clone
  twin receive the same age and the payload-identity discipline holds;
- deterministic under a declared seed (sha256 counter stream, no global RNG);
- sex-specific, using the MALE/FEMALE 80-84 / 85-89 / 90+ populations as an
  inverse CDF, with a uniform integer age within the drawn band.

Ages are assigned, not weighted, toward the ONS distribution: the achieved
weighted shares land near the ONS shares by construction and calibration
still owns the totals, so the age-band targets remain honest constraints
rather than tautologies.

Written for the #623 assessment runner and shaped to cherry-pick into the
WS-E spine as a declarative source stage later.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "UK_AGE_TOP_CODE",
    "UK_AGE_TAIL_BANDS",
    "disaggregate_uk_age_top_code",
]

UK_AGE_TOP_CODE = 80

# Band name -> (lowest assigned age, number of integer ages drawn).
# 90+ is drawn over 90-97: wide enough to be demographically honest, narrow
# enough that no simulated rule changes past 90 are being invented.
UK_AGE_TAIL_BANDS: tuple[tuple[str, int, int], ...] = (
    ("80_84", 80, 5),
    ("85_89", 85, 5),
    ("90_plus", 90, 8),
)


def _unit_draw(source_id: object, seed: int, stream: str) -> float:
    """Deterministic uniform in [0, 1) keyed on a stable identity."""

    digest = hashlib.sha256(
        f"uk_age_tail:{stream}:{seed}:{source_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def disaggregate_uk_age_top_code(
    frame: Any,
    *,
    band_populations: Mapping[tuple[str, str], float],
    seed: int = 0,
    top_code: int = UK_AGE_TOP_CODE,
) -> dict[str, Any]:
    """Reassign top-coded ages in place and return the receipt.

    ``band_populations`` maps ``(gender, band)`` — gender in MALE/FEMALE,
    band in 80_84/85_89/90_plus — to the ONS mid-year population, taken from
    the compiled calibration register so there is exactly one source of
    truth. All six cells are required; a missing cell aborts.
    """

    person = frame.table("person")
    ages = pd.to_numeric(person["age"], errors="raise").to_numpy(dtype=float)
    if (ages > top_code).any():
        raise ValueError(
            f"input already has ages above {top_code}; refusing to "
            "disaggregate a surface that is not top-coded."
        )
    piled = ages == float(top_code)
    if not piled.any():
        raise ValueError(f"no persons at the top-code age {top_code}.")

    genders = person["gender"].astype(str).to_numpy()
    observed = set(np.unique(genders[piled]))
    if not observed <= {"MALE", "FEMALE"}:
        raise ValueError(f"unexpected gender labels in the pile: {observed}")

    band_names = [name for name, _, _ in UK_AGE_TAIL_BANDS]
    cdf: dict[str, np.ndarray] = {}
    for gender in ("MALE", "FEMALE"):
        populations = []
        for band in band_names:
            value = band_populations.get((gender, band))
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"band population ({gender}, {band}) is missing or "
                    f"non-positive: {value!r}"
                )
            populations.append(float(value))
        shares = np.asarray(populations) / sum(populations)
        cdf[gender] = np.cumsum(shares)

    source_ids = person["person_source_id"].to_numpy()
    new_ages = ages.copy()
    assigned_counts: dict[tuple[str, str], int] = {}
    for index in np.flatnonzero(piled):
        gender = genders[index]
        band_draw = _unit_draw(source_ids[index], seed, "band")
        band_index = int(np.searchsorted(cdf[gender], band_draw, side="right"))
        band_index = min(band_index, len(band_names) - 1)
        name, low, width = UK_AGE_TAIL_BANDS[band_index]
        within_draw = _unit_draw(source_ids[index], seed, "within")
        new_ages[index] = low + int(within_draw * width)
        key = (gender, name)
        assigned_counts[key] = assigned_counts.get(key, 0) + 1

    person["age"] = new_ages
    check = pd.to_numeric(frame.table("person")["age"], errors="raise")
    if int((check == float(top_code)).sum()) >= int(piled.sum()):
        raise RuntimeError("age disaggregation did not persist on the frame.")

    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    household = frame.table("household")
    weight_by_household = pd.Series(
        weights, index=household["household_id"].to_numpy()
    )
    person_weights = weight_by_household.loc[
        person["person_household_id"].to_numpy()
    ].to_numpy()

    achieved: dict[str, dict[str, float]] = {}
    for gender in ("MALE", "FEMALE"):
        gender_rows: dict[str, float] = {}
        for band, low, width in UK_AGE_TAIL_BANDS:
            mask = (
                piled
                & (genders == gender)
                & (new_ages >= low)
                & (new_ages < low + width)
            )
            gender_rows[band] = float(person_weights[mask].sum())
        achieved[gender] = gender_rows

    return {
        "stage": "uk_age_tail_disaggregation",
        "seed": seed,
        "top_code": top_code,
        "piled_persons": int(piled.sum()),
        "assigned_unweighted": {
            f"{gender}:{band}": count
            for (gender, band), count in sorted(assigned_counts.items())
        },
        "achieved_weighted": achieved,
        "band_populations": {
            f"{gender}:{band}": float(value)
            for (gender, band), value in sorted(band_populations.items())
        },
        "draw_key": "person_source_id (clone-twin consistent)",
    }
