"""How much the gambling keys rest on the union's independence and on theta (microcosm#1359).

Usage, from the tree:
  <venv>/bin/python -I <this> <gsgb consumer facts jsonl> <spine.h5> <out.json>

A key's adult rate is the union of its activities (GSGB B.9), taken as
independent within the population that plays any of them (B.3). This script
checks that against what GSGB publishes beside it, all for all sexes by age
band:

- **B.13**, the number of activities each four-week gambler took part in. Its
  share with one activity and its standard deviation, against what independence
  among players gives on B.9's thirteen top-level activities.
- **Published unions.** Where B.9 or B.3 prints a union beside its parts (any
  lottery draw, scratchcards, online instant win games, and every activity but
  the two draws within the population that plays more than draws), the union
  that independence gives, and the one a shared-intensity model gives.
- **The shared-intensity model.** Each adult carries an intensity drawn from a
  gamma distribution with mean 1; given it, the activities are independent, each
  played with chance ``1 - exp(-intensity * h)``. The ``h`` reproduce each
  activity's published rate, and the gamma's shape the published rate of the
  key's population (any gambling for the National Lottery, more than draws for
  betting and gaming). B.13 is then the check.

It then sets the frame's adults (age, sex, household, prior weight) against the
key's rates and writes the national share of households the step targets over
four weeks and twelve months (the share that carries the duty, since the walk
fills each stratum to its share), at the declared rates and theta, at other
thetas, and with each key's union from the model and at its bounds.

GSGB figures are public. The frame's figures are national shares of the frame's
households at prior weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import optimize, stats

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.gambling_participation import (
    OCCASIONAL,
    REGULAR,
    gambling_participation_operation,
    household_participation,
    participation_rates,
)

GREAT_BRITAIN = "K03000001"
PARTICIPATION = "gambling_commission.gsgb.participation"
COUNTS = "gambling_commission.gsgb.activity_count_distribution"
MEAN_COUNT = "gambling_commission.gsgb.mean_activity_count"
AGES = (
    "all",
    "age_18_to_24",
    "age_25_to_34",
    "age_35_to_44",
    "age_45_to_54",
    "age_55_to_64",
    "age_65_to_74",
    "age_75_plus",
)
#: B.9's top-level activities, each once, with the channel row that holds it.
ACTIVITIES = {
    "national_lottery_draws": "all",
    "other_charity_lottery_draws": "all",
    "national_lottery_scratchcards": "all",
    "other_scratchcards": "in_person",
    "national_lottery_online_instant_win_games": "online",
    "other_online_instant_win_games": "online",
    "betting": "all",
    "bingo": "all",
    "casino_games": "all",
    "fruit_and_slots_games": "all",
    "another_form_of_gambling": "all",
    "football_pools": "all",
    "private_betting": "all",
}
DRAWS = ("national_lottery_draws", "other_charity_lottery_draws")
NON_DRAWS = tuple(name for name in ACTIVITIES if name not in DRAWS)
#: Each key's population (B.3) and the activities it may hold.
POPULATIONS = {
    "national_lottery": ("any_gambling", tuple(ACTIVITIES)),
    "betting_and_gaming": ("excluding_lottery_draw_only", NON_DRAWS),
}
#: Unions B.9 prints beside their parts, and the population they sit in.
PUBLISHED_UNIONS = {
    "any_lottery_draw": (("any_lottery_draw", "all"), DRAWS, "national_lottery"),
    "scratchcards": (
        ("scratchcards", "all"),
        ("national_lottery_scratchcards", "other_scratchcards"),
        "betting_and_gaming",
    ),
    "online_instant_win_games": (
        ("online_instant_win_games", "online"),
        (
            "national_lottery_online_instant_win_games",
            "other_online_instant_win_games",
        ),
        "betting_and_gaming",
    ),
}
COUNT_BANDS = {
    "one": (1, 1),
    "two_to_three": (2, 3),
    "four_to_six": (4, 6),
    "seven_or_more": (7, len(ACTIVITIES)),
}
THETAS = {
    "national_lottery": (0.0, 0.5, 1.0),
    "betting_and_gaming": (0.0, 0.25, 0.5, 1.0),
}
QUANTILES = 4000


def _rows(path: Path) -> list[dict]:
    rows = []
    for line in open(path):
        fact = json.loads(line)
        if fact["geography"]["id"] == GREAT_BRITAIN:
            rows.append(fact)
    return rows


def _value(rows, concept, sheet, age, **dimensions) -> float:
    found = [
        fact["value"]
        for fact in rows
        if fact["concept_alignment"]["canonical_concept"] == concept
        and fact["dimensions"].get("source_sheet") == sheet
        and fact["dimensions"].get("sex") == "all"
        and fact["dimensions"].get("age_band") == age
        and all(fact["dimensions"].get(k) == v for k, v in dimensions.items())
    ]
    if len(found) != 1:
        raise SystemExit(f"{sheet} {age} {dimensions}: {len(found)} rows.")
    return float(found[0])


def _rate(rows, scope, age, channel=None, sheet="Table B.9") -> float:
    selector = {"participation_scope": scope, "participation_window": "past_4_weeks"}
    if channel is not None:
        selector["gambling_channel"] = channel
    return _value(rows, PARTICIPATION, sheet, age, **selector)


def _independent(rates, names, within) -> float:
    """The union among all adults, independent within ``within``."""

    return within * (1.0 - np.prod([1.0 - min(rates[n] / within, 1.0) for n in names]))


class SharedIntensity:
    """Activities independent given a gamma intensity, fitted to a population's rate."""

    def __init__(self, rates: dict[str, float], names, population: float):
        self.names = tuple(names)
        played = np.array([rates[n] for n in self.names])

        def loads(shape):
            return shape * ((1.0 - played) ** (-1.0 / shape) - 1.0)

        def gap(log_shape):
            shape = np.exp(log_shape)
            return 1.0 - (1.0 + loads(shape).sum() / shape) ** -shape - population

        self.shape = float(
            np.exp(optimize.brentq(gap, np.log(1e-2), np.log(1e5), xtol=1e-12))
        )
        self.load = dict(zip(self.names, loads(self.shape), strict=True))

    def union(self, names) -> float:
        total = sum(self.load[n] for n in names)
        return 1.0 - (1.0 + total / self.shape) ** -self.shape

    def counts(self) -> np.ndarray:
        """The distribution of the number of activities, from 0."""

        intensity = stats.gamma.ppf(
            (np.arange(QUANTILES) + 0.5) / QUANTILES, self.shape, scale=1 / self.shape
        )
        chance = 1.0 - np.exp(-np.outer(intensity, [self.load[n] for n in self.names]))
        return _poisson_binomial(chance).mean(axis=0)


def _poisson_binomial(chance: np.ndarray) -> np.ndarray:
    chance = np.atleast_2d(chance)
    out = np.zeros((chance.shape[0], chance.shape[1] + 1))
    out[:, 0] = 1.0
    for column in range(chance.shape[1]):
        step = out * (1.0 - chance[:, [column]])
        step[:, 1:] += out[:, :-1] * chance[:, [column]]
        out = step
    return out


def _among_players(counts: np.ndarray) -> dict[str, float]:
    played = counts[1:] / counts[1:].sum()
    n = np.arange(1, len(counts))
    mean = float(played @ n)
    shares = {
        band: float(played[low - 1 : high].sum())
        for band, (low, high) in COUNT_BANDS.items()
    }
    return {
        **{band: round(share, 3) for band, share in shares.items()},
        "mean": round(mean, 2),
        "standard_deviation": round(float(np.sqrt(played @ n**2 - mean**2)), 2),
    }


def published_checks(rows) -> tuple[dict, dict]:
    """The B.13 and union checks, and each key's union share by age band."""

    report: dict = {}
    unions: dict = {key: {} for key in POPULATIONS}
    for age in AGES:
        rates = {n: _rate(rows, n, age, c) for n, c in ACTIVITIES.items()}
        populations = {
            key: _rate(rows, scope, age, sheet="Table B.3")
            for key, (scope, _) in POPULATIONS.items()
        }
        models = {
            key: SharedIntensity(rates, names, populations[key])
            for key, (_, names) in POPULATIONS.items()
        }
        within = populations["national_lottery"]
        independent = np.array([min(rates[n] / within, 1.0) for n in ACTIVITIES])
        players = _poisson_binomial(independent)[0]
        b13 = {
            band: round(
                _value(
                    rows,
                    COUNTS,
                    "Table B.13",
                    age,
                    activity_count_band=band,
                    participation_window="past_4_weeks",
                ),
                3,
            )
            for band in COUNT_BANDS
        }
        b13["mean"] = round(
            _value(
                rows, MEAN_COUNT, "Table B.13", age, participation_window="past_4_weeks"
            ),
            2,
        )
        entry = {
            "four_week_rate": {
                "any_gambling": round(within, 4),
                "excluding_lottery_draw_only": round(
                    populations["betting_and_gaming"], 4
                ),
            },
            "activities_among_four_week_players": {
                "gsgb_b13": b13,
                "independent_among_players": _among_players(players)
                | {"share_left_with_none": round(float(players[0]), 3)},
                "shared_intensity": _among_players(models["national_lottery"].counts()),
            },
            "shared_intensity_shape": {
                key: round(model.shape, 3) for key, model in models.items()
            },
            "published_unions": {},
            "keys": {},
        }
        excluding = populations["betting_and_gaming"]
        entry["published_unions"]["excluding_lottery_draw_only"] = {
            "published": round(excluding, 4),
            "independent_within_any_gambling": round(
                _independent(rates, NON_DRAWS, within), 4
            ),
            "shared_intensity_fitted_to_any_gambling": round(
                models["national_lottery"].union(NON_DRAWS), 4
            ),
        }
        for name, ((scope, channel), parts, key) in PUBLISHED_UNIONS.items():
            entry["published_unions"][name] = {
                "published": round(_rate(rows, scope, age, channel), 4),
                "population": POPULATIONS[key][0],
                "independent_within_population": round(
                    _independent(rates, parts, populations[key]), 4
                ),
                "shared_intensity": round(models[key].union(parts), 4),
            }
        for key, names in (
            ("national_lottery", KEY_ACTIVITIES["national_lottery"]),
            ("betting_and_gaming", KEY_ACTIVITIES["betting_and_gaming"]),
        ):
            population = populations[key]
            low = max(rates[n] for n in names)
            high = min(sum(rates[n] for n in names), population)
            step = _independent(rates, names, population)
            model = models[key].union(names)
            entry["keys"][key] = {
                "independent_within_population": round(step, 4),
                "shared_intensity": round(model, 4),
                "shared_intensity_over_independent": round(model / step, 3),
                "bounds": [round(low, 4), round(high, 4)],
            }
            if age != "all":
                unions[key][age] = {
                    "step": step / population,
                    "shared_intensity": model / population,
                    "lower_bound": low / population,
                    "upper_bound": high / population,
                }
        report[age] = entry
    return report, unions


#: Each key's activities, as the committed declaration names them.
KEY_ACTIVITIES: dict[str, tuple[str, ...]] = {}


def household_targets(spine: Path, unions: dict) -> dict:
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    parameters = gambling_participation_operation(stage)
    rates = participation_rates(parameters)
    bands = rates.bands
    names = {int(lower): str(value) for lower, value in parameters["age_bands"]}
    with h5py.File(spine, "r") as handle:
        person = handle["person/table"].fields(
            ["person_household_id", "age", "gender"]
        )[:]
        household = handle["household/table"].fields(
            ["household_id", "household_weight"]
        )[:]
    person = pd.DataFrame(
        {
            "person_household_id": person["person_household_id"],
            "age": person["age"].astype(float),
            "gender": [g.decode() for g in person["gender"]],
        }
    )
    ids = household["household_id"]
    weight = household["household_weight"].astype(float)
    mass = float(weight.sum())

    def share(cells, theta):
        q = household_participation(person, ids, cells, bands=bands, theta=theta)
        return round(float(weight @ q / mass), 4)

    report: dict = {"households": int(len(ids)), "keys": {}}
    for key in POPULATIONS:
        theta = float(parameters["keys"][key]["theta"])
        declared = rates.rates[key]
        entry: dict = {"declared_theta": theta, "theta": {}, "union": {}}
        for value in sorted({theta, *THETAS[key]}):
            entry["theta"][f"{value:g}"] = {
                window: share(declared[cls], value)
                for window, cls in (
                    ("four_weeks", REGULAR),
                    ("twelve_months", OCCASIONAL),
                )
            }
        for variant in ("shared_intensity", "lower_bound", "upper_bound"):
            scale = {
                lower: unions[key][names[lower]][variant]
                / unions[key][names[lower]]["step"]
                for lower in bands
            }
            entry["union"][variant] = {}
            for window, cls in (("four_weeks", REGULAR), ("twelve_months", OCCASIONAL)):
                cells = {
                    (sex, lower): min(value * scale[lower], 1.0)
                    for (sex, lower), value in declared[cls].items()
                }
                entry["union"][variant][window] = share(cells, theta)
        report["keys"][key] = entry
    return report


def main() -> None:
    feed, spine, out = (Path(arg) for arg in sys.argv[1:4])
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    declared = gambling_participation_operation(stage)["keys"]
    for key in POPULATIONS:
        KEY_ACTIVITIES[key] = tuple(declared[key]["activities"])
        if declared[key]["population"] != POPULATIONS[key][0]:
            raise SystemExit(f"{key}: the declaration's population moved.")
        if not set(KEY_ACTIVITIES[key]) <= set(POPULATIONS[key][1]):
            raise SystemExit(f"{key}: an activity sits outside its population.")
    rows = _rows(feed)
    published, unions = published_checks(rows)
    report = {
        "gsgb": published,
        "frame": household_targets(spine, unions),
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["frame"], indent=1))


if __name__ == "__main__":
    main()
