"""Synthetic gambling participation facts for the top-up step's tests (microcosm#1359).

The values are made up, not publisher rows: they exercise the step's
arithmetic and its checks, so no test copies a published figure. The real
rows will come from the vendored Chronicle resource of the GSGB tables; the
layout follows that package's, down to the online sub-row an activity
carries beside its all-channels row.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

PARTICIPATION_RESOURCE = "gambling_participation_facts.json"
PARTICIPATION = "synthetic.gambling_participation"
FOUR_WEEKS = "past_4_weeks"
TWELVE_MONTHS = "past_12_months"
ANY = "any_gambling"
NON_DRAW = "excluding_lottery_draw_only"
BANDS = ("age_18_to_34", "age_35_to_54", "age_55_plus")


def _row(
    window: str, scope: str, sex: str, band: str, value: float, **more: str
) -> dict[str, Any]:
    dimensions = {
        "participation_window": window,
        "participation_scope": scope,
        "sex": sex,
        "age_band": band,
        **more,
    }
    return {
        "concept": PARTICIPATION,
        "unit": "share",
        "period": {"type": "calendar_year", "value": 2025},
        "period_coverage": {"start_date": "2025-01-13", "end_date": "2026-01-22"},
        "geography": {"id": "K03000001"},
        "dimensions": dimensions,
        "value": value,
        "source_record_id": f"{PARTICIPATION}.{'.'.join(dimensions.values())}.2025",
    }


#: Four-week rates of the two populations by band, for men, women and all
#: adults, and the twelve-month rate as a multiple of each: round numbers no
#: publisher printed.
_POPULATIONS = {
    ANY: ({"male": 0.4, "female": 0.3, "all": 0.35}, 1.3),
    NON_DRAW: ({"male": 0.45, "female": 0.35, "all": 0.4}, 1.5),
}
_BAND_SCALE = {"age_18_to_34": 1.0, "age_35_to_54": 1.2, "age_55_plus": 0.8}
#: Four-week rates of each activity among all adults, by band: above the
#: synthetic donor's, so the step adds households.
ACTIVITIES = {
    "national_lottery_draws": (0.1, 0.25, 0.2),
    "national_lottery_scratchcards": (0.05, 0.04, 0.02),
    "betting": (0.25, 0.2, 0.12),
    "bingo": (0.05, 0.05, 0.06),
    "casino_games": (0.1, 0.05, 0.02),
    "other_online_instant_win_games": (0.03, 0.02, 0.01),
}
#: An activity played only online, which the activity table prints as an
#: online row alone.
ONLINE_ONLY = ("other_online_instant_win_games",)
#: The online part of any gambling and of two activities: a row of the
#: activity table beside the all-channels one (or, for any gambling, beside
#: the population table's row).
ONLINE = {ANY: 0.4, "national_lottery_draws": 0.6, "betting": 0.5}
#: The worksheet each window's population rates and the activities come from.
SHEETS = {FOUR_WEEKS: "Table B.3", TWELVE_MONTHS: "Table B.1"}
ACTIVITY_SHEET = "Table B.9"

PARTICIPATION_FACT_ROWS: list[dict[str, Any]] = [
    *(
        _row(
            window,
            scope,
            sex,
            band,
            round(rate * _BAND_SCALE[band] * multiple, 6),
            source_sheet=SHEETS[window],
        )
        for scope, (rates, annual) in _POPULATIONS.items()
        for window, multiple in ((FOUR_WEEKS, 1.0), (TWELVE_MONTHS, annual))
        for sex, rate in rates.items()
        for band in BANDS
    ),
    *(
        _row(
            FOUR_WEEKS,
            activity,
            "all",
            band,
            value,
            source_sheet=ACTIVITY_SHEET,
            gambling_channel="online" if activity in ONLINE_ONLY else "all",
        )
        for activity, values in ACTIVITIES.items()
        for band, value in zip(BANDS, values, strict=True)
    ),
    *(
        _row(
            FOUR_WEEKS,
            scope,
            "all",
            band,
            round(value * share, 6),
            source_sheet=ACTIVITY_SHEET,
            gambling_channel="online",
        )
        for scope, share in ONLINE.items()
        for band, value in zip(
            BANDS,
            ACTIVITIES.get(scope)
            or [_POPULATIONS[ANY][0]["all"] * _BAND_SCALE[band] for band in BANDS],
            strict=True,
        )
    ),
]

PARTICIPATION_DECLARATION: dict[str, Any] = {
    "kind": "top_up_gambling_participation",
    "resource": PARTICIPATION_RESOURCE,
    "concept": PARTICIPATION,
    "period_type": "calendar_year",
    "period_value": 2025,
    "geography_id": "K03000001",
    "dimensions": {
        "window": "participation_window",
        "scope": "participation_scope",
        "sex": "sex",
        "age_band": "age_band",
    },
    "population_dimensions": {
        "regular": {"source_sheet": SHEETS[FOUR_WEEKS]},
        "occasional": {"source_sheet": SHEETS[TWELVE_MONTHS]},
    },
    "activity_dimensions": {"regular": {"source_sheet": ACTIVITY_SHEET}},
    "windows": {"regular": FOUR_WEEKS, "occasional": TWELVE_MONTHS},
    "sexes": {"MALE": "male", "FEMALE": "female", "all": "all"},
    "age_bands": [[18, "age_18_to_34"], [35, "age_35_to_54"], [55, "age_55_plus"]],
    "keys": {
        "national_lottery": {
            "population": ANY,
            "activities": {
                "national_lottery_draws": {"gambling_channel": "all"},
                "national_lottery_scratchcards": {"gambling_channel": "all"},
            },
            "theta": 0.9,
        },
        "betting_and_gaming": {
            "population": NON_DRAW,
            "activities": {
                "betting": {"gambling_channel": "all"},
                "bingo": {"gambling_channel": "all"},
                "casino_games": {"gambling_channel": "all"},
                "other_online_instant_win_games": {"gambling_channel": "online"},
            },
            "theta": 0.1,
        },
    },
    "occasional_stake_share": 0.1,
    "strata": {"adults_cap": 3, "youngest_adult_bounds": [35, 55]},
    "propensity": {
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "validation_fraction": 0.1,
    },
    "n_estimators": 4,
    "seed": 0,
    "salt": "test_gambling_participation",
}


def participation_declaration(**changes: Any) -> dict[str, Any]:
    """The synthetic declaration with top-level ``changes``."""

    declaration = copy.deepcopy(PARTICIPATION_DECLARATION)
    declaration.update(copy.deepcopy(changes))
    return declaration


def participation_inputs(
    n: int = 2400, *, seed: int = 7, declaration: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Synthetic arguments for ``top_up_gambling_participation``, rows included.

    ``n`` households of one, two or three adults (a third each) and up to two
    children, at weights between 0.5 and 2; a donor of ``n // 2`` households
    recording each key's stakes on about a quarter, more often at higher
    income; and drawn stakes on about one household in twelve.
    """

    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    adults = 1 + np.arange(n) % 3
    children = rng.integers(0, 3, n)
    members = adults + children
    household_of = np.repeat(ids, members)
    is_adult = np.concatenate(
        [
            np.r_[np.ones(a, dtype=bool), np.zeros(c, dtype=bool)]
            for a, c in zip(adults, children, strict=True)
        ]
    )
    age = np.where(
        is_adult,
        rng.uniform(18.0, 90.0, len(is_adult)),
        rng.uniform(0.0, 17.0, len(is_adult)),
    ).round()
    person = pd.DataFrame(
        {
            "person_household_id": household_of,
            "age": age,
            "gender": np.where(np.arange(len(age)) % 2 == 0, "MALE", "FEMALE"),
        }
    )

    def features(rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "income": rng.lognormal(10.0, 0.6, rows),
                "renting": (rng.random(rows) < 0.35).astype(float),
                "recreation": rng.uniform(100.0, 3000.0, rows),
            }
        )

    donors = n // 2
    donor_features = features(donors)
    recipient_features = features(n)
    lean = (np.log(donor_features["income"].to_numpy()) - 10.0) / 0.6

    def stakes(rows: int, rate: np.ndarray | float) -> np.ndarray:
        return np.where(rng.random(rows) < rate, rng.lognormal(3.0, 1.0, rows), 0.0)

    keys = ("national_lottery", "betting_and_gaming")
    return {
        "keys": {key: stakes(n, 1.0 / 12.0) for key in keys},
        "person": person,
        "household_ids": ids,
        "weights": rng.uniform(0.5, 2.0, n),
        "sources": None,
        "donor_features": donor_features,
        "donor_keys": {
            key: stakes(donors, 1.0 / (1.0 + np.exp(-(lean - 1.1)))) for key in keys
        },
        "donor_weights": rng.uniform(0.5, 2.0, donors),
        "recipient_features": recipient_features,
        "parameters": copy.deepcopy(declaration or PARTICIPATION_DECLARATION),
        "fit_name": "test",
        "rows": serve_participation_rows,
    }


def serve_participation_rows(name: str, **criteria: Any) -> list[dict[str, Any]]:
    """The synthetic rows matching ``criteria``, as the vendored reader serves them."""

    from microcosm.build.uk_runtime.ledger_fact_vendoring import rows_matching

    if name != PARTICIPATION_RESOURCE:
        raise ValueError(f"no synthetic rows for {name!r}.")
    return rows_matching({"rows": PARTICIPATION_FACT_ROWS}, **criteria)


def patch_gambling_participation_facts(
    monkeypatch, declaration: dict[str, Any] | None = None
) -> None:
    """Serve the synthetic rows for the participation resource, the real ones otherwise.

    With ``declaration`` the stage and the gate also read it as the declared
    step, in place of the committed spec's.
    """

    from microcosm.build.uk_runtime import gambling_participation, lcfs_consumption

    real = gambling_participation.vendored_rows

    def vendored_rows(name, *, fiscal_start=None, **criteria):
        if name != PARTICIPATION_RESOURCE:
            return real(name, fiscal_start=fiscal_start, **criteria)
        return serve_participation_rows(name, **criteria)

    monkeypatch.setattr(gambling_participation, "vendored_rows", vendored_rows)
    if declaration is not None:
        for module in (gambling_participation, lcfs_consumption):
            monkeypatch.setattr(
                module,
                "gambling_participation_operation",
                lambda stage: copy.deepcopy(declaration),
            )
