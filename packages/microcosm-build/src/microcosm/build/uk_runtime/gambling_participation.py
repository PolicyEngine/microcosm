"""Bring the LCFS stage's gambling keys to published participation (microcosm#1359).

The LCFS diary under-records gambling. On the 2024-25 diary about a fifth of
the households that GSGB's four-week participation implies record a stake in
their two weeks. Under any steady rate of play the two-week share is at least
half the four-week share, so the gap is play the diary never records (online
play, subscriptions, small stakes forgotten), not its window. The declared
``top_up_gambling_participation`` step therefore takes how many households
play from the Gambling Commission's GSGB, and which households and how much
from the diary:

- **Adult rates.** Each adult's chance of playing in four weeks and in twelve
  months comes from the published rates by sex and age band. A key's rate
  (the National Lottery's products, or the betting and gaming the other
  duties charge) is the union of its activities, taken as independent within
  the published population that plays any of them, per age band.
- **Households.** A household plays when any adult does, combined at the
  key's declared theta: ``1 - (1 - p_max) * prod(1 - p_other) ** theta``.
  Theta 0 is one player per household (adults who play together), theta 1
  adults who play independently.
- **The walk.** Within each household stratum (its adults, and its youngest
  adult's age band), households without a stake are added by the gas
  connection walk's weight-fitting rule (microcosm#1164), in an
  identity-keyed order drawn in proportion to the donor's propensity times
  the household's published chance, until the stratum's four-week share is
  met (regular players) and then its twelve-month share (occasional
  players). The chain's predictors carry no age or sex, so the published
  chance brings GSGB's age and sex mix inside a stratum, and the propensity
  the diary's gradient by income, tenure, region and spending. Households the
  diary saw keep their stakes.
- **Stakes.** An added household's stake is a draw from the donor's positive
  stakes at a quantile keyed on its id, scaled so the added regular players'
  mean is the mean of the households the diary saw.
  An occasional player's is the declared share of that.

``level_excise_quantities`` then shares each national base by the topped-up
keys, so every base still sums to its published figure.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import EXPLICIT_KIND, FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.energy_pricing import _connect_walk
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    extreme_count,
    rounded_count,
    support_band,
)

TOP_UP_GAMBLING_PARTICIPATION_KIND = "top_up_gambling_participation"
UK_GAMBLING_PARTICIPATION_RESOURCE = "gambling_participation_facts.json"
UK_GAMBLING_PARTICIPATION_RECEIPT = "gambling_participation"
#: The keys the step tops up: the stakes ``level_excise_quantities`` shares
#: Lottery Duty's base by, and those it shares every other gambling base by.
UK_GAMBLING_PARTICIPATION_KEYS = ("national_lottery", "betting_and_gaming")
UK_GAMBLING_ADULT_AGE = 18
#: The two classes, by the window whose share they fill: regular players to
#: the four-week share, occasional players on to the twelve-month share.
REGULAR = "regular"
OCCASIONAL = "occasional"
#: The frame's sexes, as its person table spells them.
UK_GAMBLING_SEXES = ("MALE", "FEMALE")
#: The stratum of households with no adult, which play nothing.
NO_ADULTS = "no_adults"
#: The propensity classifier's settings a declaration sets; early stopping on
#: the held-out fraction is always on.
UK_GAMBLING_PROPENSITY_SETTINGS = (
    "max_leaf_nodes",
    "min_samples_leaf",
    "l2_regularization",
    "validation_fraction",
)
#: The fewest distinct source households a receipt cell may rest on, and the
#: fewest donor households a key's fits may learn from (CD171 section 5.2.1).
SMALL_CELL = 10


@dataclass(frozen=True)
class ParticipationRates:
    """Each key's adult chance of playing, by class, sex and age band.

    ``bands`` are the age bands' lower bounds in years, ascending. ``rates``
    maps a key and a class to each (sex, band) cell's rate. ``receipt``
    records the rows read and the rates.
    """

    bands: tuple[int, ...]
    rates: Mapping[str, Mapping[str, Mapping[tuple[str, int], float]]]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class GamblingParticipation:
    """The topped-up keys, the receipt and the fits' weight records."""

    keys: Mapping[str, np.ndarray]
    receipt: Mapping[str, Any]
    fit_weight_records: tuple[FitWeightRecord, ...]


def gambling_participation_operation(
    stage: SourceStageSpec,
) -> Mapping[str, Any] | None:
    """The stage's declared ``top_up_gambling_participation`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == TOP_UP_GAMBLING_PARTICIPATION_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def _require(parameters: Mapping[str, Any], name: str) -> Any:
    value = parameters.get(name)
    if value is None or (isinstance(value, str | Mapping | list) and not value):
        raise ValueError(f"{TOP_UP_GAMBLING_PARTICIPATION_KIND} must declare {name}.")
    return value


def _number(value: object, *, what: str, low: float, high: float) -> float:
    """A declared number inside [low, high], or a refusal naming it."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not low <= float(value) <= high
    ):
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: {what} must be a number in "
            f"[{low}, {high}], not {value!r}."
        )
    return float(value)


def participation_settings(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """The declared step's settings, checked: what its receipt records.

    Each key's theta lies in [0, 1]; the occasional players' stake share in
    (0, 1]; the strata cap the adults at a count of one or more and band the
    youngest adult by ascending bounds above 18; the propensity sets exactly
    ``UK_GAMBLING_PROPENSITY_SETTINGS``.
    """

    keys = _require(parameters, "keys")
    if not isinstance(keys, Mapping) or set(keys) != set(
        UK_GAMBLING_PARTICIPATION_KEYS
    ):
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND} keys must be "
            f"{UK_GAMBLING_PARTICIPATION_KEYS}."
        )
    theta = {
        key: _number(
            (keys[key] or {}).get("theta"), what=f"{key} theta", low=0.0, high=1.0
        )
        for key in UK_GAMBLING_PARTICIPATION_KEYS
    }
    share = _number(
        parameters.get("occasional_stake_share"),
        what="occasional_stake_share",
        low=0.0,
        high=1.0,
    )
    if share == 0.0:
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: occasional_stake_share must be "
            "above zero, or the occasional players carry no stake."
        )
    strata = _require(parameters, "strata")
    cap = strata.get("adults_cap")
    bounds = list(strata.get("youngest_adult_bounds") or ())
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: strata.adults_cap must be a "
            f"count of one or more, not {cap!r}."
        )
    if (
        not all(isinstance(b, int) and not isinstance(b, bool) for b in bounds)
        or bounds != sorted(set(bounds))
        or (bounds and bounds[0] <= UK_GAMBLING_ADULT_AGE)
    ):
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: strata.youngest_adult_bounds "
            f"must be ascending ages above {UK_GAMBLING_ADULT_AGE}, not {bounds!r}."
        )
    propensity = _require(parameters, "propensity")
    if set(propensity) != set(UK_GAMBLING_PROPENSITY_SETTINGS):
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: propensity must set "
            f"{UK_GAMBLING_PROPENSITY_SETTINGS}, not {sorted(propensity)}."
        )
    n_estimators = int(_require(parameters, "n_estimators"))
    if n_estimators < 1:
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: n_estimators must be positive."
        )
    return {
        "theta": theta,
        "occasional_stake_share": share,
        "strata": {"adults_cap": cap, "youngest_adult_bounds": bounds},
        "propensity": {
            "max_leaf_nodes": int(propensity["max_leaf_nodes"]),
            "min_samples_leaf": int(propensity["min_samples_leaf"]),
            "l2_regularization": _number(
                propensity["l2_regularization"],
                what="propensity.l2_regularization",
                low=0.0,
                high=float("inf"),
            ),
            "validation_fraction": _number(
                propensity["validation_fraction"],
                what="propensity.validation_fraction",
                low=0.01,
                high=0.5,
            ),
        },
        "n_estimators": n_estimators,
        "seed": int(parameters.get("seed", 0)),
        "salt": str(_require(parameters, "salt")),
    }


def participation_rates(
    parameters: Mapping[str, Any],
    *,
    rows: Callable[..., list[dict[str, Any]]] | None = None,
) -> ParticipationRates:
    """Resolve each key's adult rates from the declared published rows.

    The declaration names the resource, its concept, period and geography; the
    dimension that carries each of the window, the scope (an activity or a
    population), sex and age band; the values of the two windows, of each sex
    and of all sexes; the age bands as ``[lower bound, value]`` pairs; and, per
    key, the population its activities sit within and each activity, read for
    the four weeks only, with the further dimensions of its row (an activity
    with online and in-person rows names its all-channels row, an online-only
    activity its online row). ``population_dimensions`` and
    ``activity_dimensions`` map a class to the further dimensions all its rows
    carry, such as the worksheet a window's population rates come from. Every
    cell must match exactly one row.
    """

    read = vendored_rows if rows is None else rows
    resource = str(_require(parameters, "resource"))
    if resource != UK_GAMBLING_PARTICIPATION_RESOURCE:
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND} must read "
            f"{UK_GAMBLING_PARTICIPATION_RESOURCE!r}, not {resource!r}."
        )
    concept = str(_require(parameters, "concept"))
    period_type = str(_require(parameters, "period_type"))
    period_value = int(_require(parameters, "period_value"))
    geography_id = str(_require(parameters, "geography_id"))
    dimensions = _require(parameters, "dimensions")
    windows = _require(parameters, "windows")
    sexes = _require(parameters, "sexes")
    keys = _require(parameters, "keys")
    for name in ("window", "scope", "sex", "age_band"):
        if not dimensions.get(name):
            raise ValueError(
                f"{TOP_UP_GAMBLING_PARTICIPATION_KIND} must name the {name} dimension."
            )
    if set(windows) != {REGULAR, OCCASIONAL}:
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND} windows must map {REGULAR!r} "
            f"(four weeks) and {OCCASIONAL!r} (twelve months)."
        )
    if set(sexes) != {*UK_GAMBLING_SEXES, "all"}:
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND} sexes must map "
            f"{(*UK_GAMBLING_SEXES, 'all')}."
        )
    if not isinstance(keys, Mapping) or set(keys) != set(
        UK_GAMBLING_PARTICIPATION_KEYS
    ):
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND} keys must be "
            f"{UK_GAMBLING_PARTICIPATION_KEYS}."
        )
    declared_bands = list(_require(parameters, "age_bands"))
    bands = tuple(int(lower) for lower, _ in declared_bands)
    if bands[0] != UK_GAMBLING_ADULT_AGE or list(bands) != sorted(set(bands)):
        raise ValueError(
            f"{TOP_UP_GAMBLING_PARTICIPATION_KIND} age bands must rise from "
            f"{UK_GAMBLING_ADULT_AGE} without repeats."
        )
    band_values = {int(lower): str(value) for lower, value in declared_bands}
    fixed: dict[str, dict[str, dict[str, str]]] = {}
    for kind in ("population", "activity"):
        declared = parameters.get(f"{kind}_dimensions") or {}
        if not isinstance(declared, Mapping) or not set(declared) <= {
            REGULAR,
            OCCASIONAL,
        }:
            raise ValueError(
                f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: {kind}_dimensions must map "
                f"the classes {REGULAR!r} and {OCCASIONAL!r} to dimensions."
            )
        fixed[kind] = {
            window: {str(name): str(value) for name, value in selector.items()}
            for window, selector in declared.items()
        }
        roles = {dimensions[name] for name in dimensions}
        if any(set(selector) & roles for selector in fixed[kind].values()):
            raise ValueError(
                f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: {kind}_dimensions may not "
                "fix the window, scope, sex or age band dimension."
            )
    read_rows: set[str] = set()

    def cell(
        kind: str,
        window: str,
        scope: str,
        sex: str,
        band: int,
        extra: Mapping[str, str] | None = None,
    ) -> float:
        selector = {
            **fixed[kind].get(window, {}),
            **(extra or {}),
            dimensions["window"]: windows[window],
            dimensions["scope"]: scope,
            dimensions["sex"]: sexes[sex],
            dimensions["age_band"]: band_values[band],
        }
        found = read(
            resource,
            concept=concept,
            period_type=period_type,
            period_value=period_value,
            geography_id=geography_id,
            dimensions=selector,
        )
        if len(found) != 1:
            raise ValueError(
                f"{resource}: expected one {concept} row {selector}, found {len(found)}."
            )
        value = float(found[0]["value"])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{resource}: {selector} is no share: {value}.")
        read_rows.add(str(found[0].get("source_record_id") or selector))
        return value

    rates: dict[str, dict[str, dict[tuple[str, int], float]]] = {}
    for key in UK_GAMBLING_PARTICIPATION_KEYS:
        population = str(keys[key].get("population") or "")
        declared_activities = keys[key].get("activities")
        if not population or not isinstance(declared_activities, Mapping):
            raise ValueError(
                f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: {key} must declare its "
                "population and map each activity to the further dimensions of "
                "its row."
            )
        activities = {
            str(activity): {
                str(name): str(value) for name, value in (row or {}).items()
            }
            for activity, row in declared_activities.items()
        }
        roles = {dimensions[name] for name in dimensions}
        if not activities or any(set(row) & roles for row in activities.values()):
            raise ValueError(
                f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: {key} must declare an "
                "activity, and no activity may fix the window, scope, sex or age "
                "band dimension."
            )
        rates[key] = {REGULAR: {}, OCCASIONAL: {}}
        for band in bands:
            # Activities independent within the population that plays any of
            # them, so the union is that population's share of it.
            within = cell("population", REGULAR, population, "all", band)
            product = 1.0
            for activity, row in activities.items():
                played = cell("activity", REGULAR, activity, "all", band, row)
                product *= 1.0 - (min(played / within, 1.0) if within > 0 else 0.0)
            union = 1.0 - product
            for sex in UK_GAMBLING_SEXES:
                regular = cell("population", REGULAR, population, sex, band)
                annual = cell("population", OCCASIONAL, population, sex, band)
                rates[key][REGULAR][(sex, band)] = regular * union
                # The cell's own twelve-month to four-week ratio.
                rates[key][OCCASIONAL][(sex, band)] = (
                    min(regular * union * annual / regular, 1.0) if regular > 0 else 0.0
                )
    receipt = {
        "resource": resource,
        "concept": concept,
        "period_type": period_type,
        "period_value": period_value,
        "geography_id": geography_id,
        "rows_read": len(read_rows),
        "rates": {
            key: {
                window: {f"{sex}:{band}": value for (sex, band), value in cells.items()}
                for window, cells in by_window.items()
            }
            for key, by_window in rates.items()
        },
    }
    return ParticipationRates(bands, rates, receipt)


def _sex_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).upper()


def _adults(
    person: pd.DataFrame, household_ids: Sequence[object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each person's household row, age, sex and whether an adult."""

    rows = pd.Index(household_ids).get_indexer(person["person_household_id"].to_numpy())
    if (rows < 0).any():
        raise ValueError("Some people name a household the frame does not hold.")
    age = pd.to_numeric(person["age"], errors="raise").to_numpy(dtype=float)
    sex = np.array([_sex_name(value) for value in person["gender"]], dtype=object)
    adult = age >= UK_GAMBLING_ADULT_AGE
    unknown = set(sex[adult]) - set(UK_GAMBLING_SEXES)
    if unknown:
        raise ValueError(
            f"The frame's adults carry a gender outside {UK_GAMBLING_SEXES}: "
            f"{sorted(unknown)}."
        )
    return rows, age, sex, adult


def household_participation(
    person: pd.DataFrame,
    household_ids: Sequence[object],
    cells: Mapping[tuple[str, int], float],
    *,
    bands: Sequence[int],
    theta: float,
) -> np.ndarray:
    """Each household's chance that any adult plays, combined at ``theta``.

    ``1 - (1 - p_max) * prod_{other adults} (1 - p_i) ** theta``, where each
    adult's ``p_i`` is its (sex, age band) cell. A household with no adult
    plays nothing.
    """

    if not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta {theta} is outside [0, 1].")
    rows, age, sex, adult = _adults(person, household_ids)
    band = np.maximum(np.searchsorted(np.asarray(bands), age, side="right") - 1, 0)
    table = np.array(
        [[cells[(name, int(lower))] for lower in bands] for name in UK_GAMBLING_SEXES]
    )
    p = np.where(adult, table[(sex == UK_GAMBLING_SEXES[1]).astype(int), band], 0.0)
    p = np.minimum(p, 1.0 - 1e-12)
    households = len(household_ids)
    top = np.zeros(households)
    np.maximum.at(top, rows, p)
    log_none = np.zeros(households)
    np.add.at(log_none, rows, np.log1p(-p))
    log_others = log_none - np.log1p(-top)
    return 1.0 - (1.0 - top) * np.exp(theta * log_others)


def household_strata(
    person: pd.DataFrame,
    household_ids: Sequence[object],
    *,
    adults_cap: int,
    youngest_bounds: Sequence[int],
) -> np.ndarray:
    """Each household's stratum: its adults (capped) and its youngest adult's band."""

    rows, age, _sex, adult = _adults(person, household_ids)
    households = len(household_ids)
    adults = np.zeros(households, dtype=int)
    np.add.at(adults, rows[adult], 1)
    has_adult = adults > 0
    lowest = np.full(households, np.inf)
    np.minimum.at(lowest, rows[adult], age[adult])
    youngest = np.where(has_adult, lowest, float(UK_GAMBLING_ADULT_AGE))
    edges = [UK_GAMBLING_ADULT_AGE, *youngest_bounds]
    spans = [
        f"{lower}_to_{edges[i + 1] - 1}" if i + 1 < len(edges) else f"{lower}_plus"
        for i, lower in enumerate(edges)
    ]
    place = np.searchsorted(np.asarray(edges), youngest, side="right") - 1
    count = np.minimum(adults, adults_cap)
    labels = np.array(
        [
            f"adults_{c}{'_plus' if c == adults_cap else ''}:youngest_{spans[p]}"
            for c, p in zip(count, place, strict=True)
        ],
        dtype=object,
    )
    labels[~has_adult] = NO_ADULTS
    return labels


def _propensity(
    donor_features: np.ndarray,
    donor_played: np.ndarray,
    donor_weights: np.ndarray,
    recipient_features: np.ndarray,
    *,
    seed: int,
    settings: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """The donor's chance of recording the key, for each recipient.

    A regularised gradient-boosted classifier that stops early on a held-out
    fraction, so its probabilities on households it did not see do not sink
    the way the chain's default gate's do for a rare positive (2026-10-24). It
    only orders the walk, which sets how many households are added.
    """

    from sklearn.ensemble import HistGradientBoostingClassifier

    classifier = HistGradientBoostingClassifier(
        random_state=seed, early_stopping=True, **dict(settings)
    )
    classifier.fit(
        donor_features, donor_played.astype(int), sample_weight=donor_weights
    )
    positive = list(classifier.classes_).index(1)
    probability = classifier.predict_proba(recipient_features)[:, positive]
    on_donor = classifier.predict_proba(donor_features)[:, positive]
    receipt = {
        "donor_rate": float(donor_weights[donor_played].sum() / donor_weights.sum()),
        "donor_mean_probability": float(np.average(on_donor, weights=donor_weights)),
        "iterations": int(classifier.n_iter_),
        "positive_support": support_band(int(donor_played.sum())),
    }
    return probability, receipt


def _positive_stakes(
    donor_features: pd.DataFrame,
    donor_stakes: np.ndarray,
    donor_weights: np.ndarray,
    recipient_features: pd.DataFrame,
    household_ids: np.ndarray,
    *,
    seed: int,
    salt: str,
    n_estimators: int,
) -> tuple[np.ndarray, Any]:
    """A positive stake for every recipient, from the donor's positive stakes.

    The gas connection walk's amount model (``gas_payer_draws``): a
    regime-gated QRF fitted on the donors who record the key, drawn from its
    positive forest at quantiles keyed on the household id.
    """

    from microcosm.fit.qrf import Regime, RegimeGatedQRF

    target = "stake"
    positive = donor_stakes > 0
    train = donor_features.loc[positive].reset_index(drop=True)
    train[target] = donor_stakes[positive]
    train["household_weight"] = donor_weights[positive]
    fitted = RegimeGatedQRF(n_estimators=n_estimators, seed=seed).fit(
        train, list(donor_features.columns), [target], weights="household_weight"
    )
    regime = str(fitted.regimes()[target])
    if regime != Regime.POSITIVE_ONLY:
        raise ValueError(
            f"the donor's positive stakes fitted regime {regime!r}, not "
            f"{Regime.POSITIVE_ONLY!r}."
        )
    stakes = fitted.predict_positive_from_uniforms(
        recipient_features.reset_index(drop=True),
        quantiles={
            target: stable_identity_uniforms(household_ids, seed=seed, salt=salt)
        },
    )[target].to_numpy(dtype=float)
    return stakes, fitted.weight_kind


def _ships(
    sources: np.ndarray, weight: np.ndarray, cells: Sequence[np.ndarray]
) -> bool:
    """Whether every cell rests on none or on ``SMALL_CELL`` or more source households.

    A receipt gives a group's figures only when each of its cells does: its
    figures are sums of these cells, so one small cell would come back by
    difference (CD171 section 5.2.1).
    """

    return all(
        (count := distinct_count(sources, cell, weight)) == 0 or count >= SMALL_CELL
        for cell in cells
    )


def top_up_gambling_participation(
    keys: Mapping[str, np.ndarray],
    *,
    person: pd.DataFrame,
    household_ids: Sequence[object],
    weights: Sequence[float],
    sources: Sequence[object] | None,
    donor_features: pd.DataFrame,
    donor_keys: Mapping[str, np.ndarray],
    donor_weights: Sequence[float],
    recipient_features: pd.DataFrame,
    parameters: Mapping[str, Any],
    fit_name: str,
    rows: Callable[..., list[dict[str, Any]]] | None = None,
) -> GamblingParticipation:
    """Top each gambling key up to its published household participation.

    ``keys`` are the frame's drawn stakes per key, in household order, and
    ``donor_keys`` the donor's own, on its rows; ``donor_features`` and
    ``recipient_features`` are the same predictor columns on each side.
    ``sources`` are the households' source households, for the receipt's
    floors (``None`` when every household is its own). Every household the
    diary saw keeps its stake, and the walk adds households with drawn stakes
    until each stratum's four-week and twelve-month shares are met; a stratum
    that runs out of households first is refused.

    The receipt gives, per key, the targets and the shares before and after,
    nationally and per stratum, the share of the key's stakes each class
    carries and the fits' donor support as a band: shares only, and a
    group's figures only where every cell under them rests on ten or more
    source households.
    """

    ids = np.asarray(household_ids)
    weight = np.asarray(weights, dtype=float)
    mass = float(weight.sum())
    source_ids = np.arange(len(ids)) if sources is None else np.asarray(sources)
    donor_weight = np.asarray(donor_weights, dtype=float)
    settings = participation_settings(parameters)
    rates = participation_rates(parameters, rows=rows)
    strata = household_strata(
        person,
        ids,
        adults_cap=settings["strata"]["adults_cap"],
        youngest_bounds=settings["strata"]["youngest_adult_bounds"],
    )
    labels = sorted(set(strata))
    seed, salt = settings["seed"], settings["salt"]
    share = settings["occasional_stake_share"]
    donor_x = donor_features.to_numpy(dtype=float)
    recipient_x = recipient_features.to_numpy(dtype=float)
    everyone = np.ones(len(ids), dtype=bool)
    topped_keys: dict[str, np.ndarray] = {}
    records: list[FitWeightRecord] = []
    receipt: dict[str, Any] = {
        "operation": TOP_UP_GAMBLING_PARTICIPATION_KIND,
        "facts": rates.receipt,
        "settings": settings,
        "keys": {},
    }
    for key in UK_GAMBLING_PARTICIPATION_KEYS:
        drawn = np.asarray(keys[key], dtype=float)
        donor_stakes = np.asarray(donor_keys[key], dtype=float)
        played = donor_stakes > 0
        if not SMALL_CELL <= int(played.sum()) <= len(played) - SMALL_CELL:
            raise ValueError(
                f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: the donor must record {key} "
                f"stakes on {SMALL_CELL} or more households and leave {SMALL_CELL} "
                "or more without, to fit and report its share."
            )
        chance = {
            window: household_participation(
                person,
                ids,
                rates.rates[key][window],
                bands=rates.bands,
                theta=settings["theta"][key],
            )
            for window in (REGULAR, OCCASIONAL)
        }
        propensity, propensity_receipt = _propensity(
            donor_x,
            played,
            donor_weight,
            recipient_x,
            seed=seed,
            settings=settings["propensity"],
        )
        draws, stake_kind = _positive_stakes(
            donor_features,
            donor_stakes,
            donor_weight,
            recipient_features,
            ids,
            seed=seed,
            salt=f"{salt}:{key}:stake",
            n_estimators=settings["n_estimators"],
        )
        seen = drawn > 0
        ordering = propensity * chance[OCCASIONAL]
        uniforms = stable_identity_uniforms(ids, seed=seed, salt=f"{salt}:{key}:order")
        with np.errstate(divide="ignore"):
            order = np.where(ordering > 0, -np.log(uniforms) / ordering, np.inf)
        taken = seen.copy()
        regular = np.zeros(len(ids), dtype=bool)
        targets: dict[str, dict[str, float]] = {label: {} for label in labels}
        for window in (REGULAR, OCCASIONAL):
            for label in labels:
                members = np.flatnonzero(strata == label)
                total = float(weight[members].sum())
                target = float(
                    np.average(chance[window][members], weights=weight[members])
                )
                targets[label][window] = target
                walk = _connect_walk(
                    taken,
                    rows=members,
                    weight=weight,
                    key=order,
                    total=total,
                    before=float(weight[members][taken[members]].sum()) / total,
                    target=target,
                )
                if walk["candidates_exhausted"] and walk["shortfall"] > 1e-9:
                    raise ValueError(
                        f"{TOP_UP_GAMBLING_PARTICIPATION_KIND}: {key} stratum {label} "
                        f"ran out of households before its {window} share."
                    )
            if window == REGULAR:
                regular = taken & ~seen
        occasional = taken & ~seen & ~regular
        # The added regular players' mean stake is the seen households' (the
        # donor's positive mean where the frame drew none): the walk picks
        # households whose predictors draw larger stakes, so scaling to the
        # donor's mean would put them above the players the diary saw.
        anchor = (
            np.average(drawn[seen], weights=weight[seen])
            if seen.any()
            else np.average(donor_stakes[played], weights=donor_weight[played])
        )
        scale = (
            float(anchor / np.average(draws[regular], weights=weight[regular]))
            if regular.any()
            else 1.0
        )
        topped = drawn.copy()
        topped[regular] = scale * draws[regular]
        topped[occasional] = share * scale * draws[occasional]
        topped_keys[key] = topped
        strata_receipt: dict[str, Any] = {}
        for label in labels:
            member = strata == label
            cells = (member, member & seen, member & regular, member & occasional)
            if not _ships(source_ids, weight, cells):
                strata_receipt[label] = {"withheld": True}
                continue
            total = float(weight[member].sum())
            strata_receipt[label] = {
                "regular_target": targets[label][REGULAR],
                "annual_target": targets[label][OCCASIONAL],
                "before": float(weight[cells[1]].sum()) / total,
                "after_regular": float(weight[cells[1] | cells[2]].sum()) / total,
                "after_annual": float(weight[cells[1] | cells[2] | cells[3]].sum())
                / total,
            }
        entry: dict[str, Any] = {"withheld": True}
        stakes = float(weight @ topped)
        if stakes > 0 and _ships(
            source_ids, weight, (everyone, seen, regular, occasional)
        ):
            entry = {
                "regular_target": float(weight @ chance[REGULAR]) / mass,
                "annual_target": float(weight @ chance[OCCASIONAL]) / mass,
                "before": float(weight[seen].sum()) / mass,
                "after_regular": float(weight[seen | regular].sum()) / mass,
                # As level_excise_quantities reads its keys' positive shares.
                "after_annual": float(weight[topped > 0].sum() / mass),
                "stake_shares": {
                    "seen": float(weight[seen] @ topped[seen]) / stakes,
                    "added_regular": float(weight[regular] @ topped[regular]) / stakes,
                    "added_occasional": float(weight[occasional] @ topped[occasional])
                    / stakes,
                },
                "amount_scale": scale,
            }
        receipt["keys"][key] = {
            **entry,
            "added_without_stake": extreme_count(
                int(((regular | occasional) & ~(topped > 0)).sum())
            ),
            "propensity": propensity_receipt,
            "stake_training_rows": rounded_count(int(played.sum())),
            "strata": strata_receipt,
        }
        records.extend(
            (
                FitWeightRecord(f"{fit_name}:{key}_participation", EXPLICIT_KIND),
                FitWeightRecord(f"{fit_name}:{key}_participation_stake", stake_kind),
            )
        )
    return GamblingParticipation(topped_keys, receipt, tuple(records))
