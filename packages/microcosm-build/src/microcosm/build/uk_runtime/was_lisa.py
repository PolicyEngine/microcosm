"""Lifetime ISA holdings imputed onto the UK spine from the Wealth and Assets Survey.

The ``was_lisa`` stage (microcosm#1003) reads the WAS round-8 person and
household tabs (UKDS SN 7215, End User Licence; interviews April 2020 to March
2022, Great Britain) and imputes, for every adult of the FRS spine, whether the
person holds a Lifetime ISA (LISA) and the current value across their LISAs.
The household total is the sum over the household's persons.

WAS asks each adult which ISA types they hold (``FISA``, code 3 = Lifetime ISA)
and the current value (``FLISAV``, or the band ``FLISAB`` when the exact value is
unknown). ONS imputes item non-response and the value within a reported band
and releases the completed value as ``DVFLISAvR8`` (``= FLISAVR8_i``), which sums
exactly to the household aggregate ``DVFLISAVR8_aggr``; the stage refuses a tab
where it does not. Each donor keeps its response class (observed, banded,
imputed) as evidence, so missing answers stay distinct from observed
non-ownership.

Two parts, both drawn from uniforms keyed on ``person_id`` so that adding or
reordering rows never moves a draw:

* Who holds: a weighted logistic regression with a light ridge penalty on age
  group, the log of the person's earnings and of household net income, gross
  financial wealth, cash ISA, stocks-and-shares ISA and savings, private
  renting, sex and the number of children. The intercept is unpenalised, so the
  model's weighted mean probability on the donor equals the donor's weighted
  ownership share. A person holds when their uniform falls below their
  probability. The house regime gate (a gradient-boosting classifier) is
  miscalibrated for an outcome held by under one per cent of adults with about
  120 donor holders, which is why this part is a logistic.
* How much: the regime-gated QRF fitted on the holders alone and drawn at the
  owner's keyed quantile (``predict_positive_from_uniforms``), so a balance is
  positive exactly when the person holds.

A declared credibility rule treats donor records the product rules make
impossible. LISAs have been opened only by people aged 18 to 39 since 6 April
2017, so no holder was older than 44 by March 2022: holders recorded in an age
band starting at 45 or above are read as a misreported ISA type and recoded to
non-holders (the value stays inside the household's gross financial wealth).
Holders above a declared balance ceiling stay holders but leave the balance
fit. Under-18s never hold; no upper age limit is imposed on recipients.

WAS counts LISAs inside gross financial wealth, so a household's LISA total
never exceeds its ``gross_financial_wealth``: a household over it has its
persons' balances scaled down pro rata, with a receipt. WAS records holdings,
not annual contributions; nothing here derives a contribution or a bonus from a
balance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import EXPLICIT_KIND, FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.national_frame import (
    uk_household_mass_conservation_receipt,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.support_clip import (
    UKSupportClipReceipt,
    support_clip_to_donor_with_receipt,
)
from microcosm.build.uk_runtime.was_wealth import (
    clean_was_household_table,
)
from microcosm.build.uk_runtime.was_wealth import (
    recipient_predictors as was_household_predictors,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

__all__ = [
    "UK_WAS_LISA_MASS_CONSERVATION_REASON",
    "UK_WAS_LISA_OUTPUT_COLUMNS",
    "UK_WAS_LISA_STAGE_NAME",
    "BalanceModelSpec",
    "OwnershipModelSpec",
    "UKWASLISAStageTransform",
    "WASLISAColumns",
    "WASLISAError",
    "cap_lifetime_isa_to_financial_wealth",
    "clean_was_lisa_donor",
    "fit_ownership_model",
    "impute_lifetime_isa_balance",
    "impute_lifetime_isa_ownership",
    "ownership_design",
    "recipient_predictors",
    "was_lisa_support_ranges",
]

UK_WAS_LISA_STAGE_NAME = "was_lisa"
#: The household-mass receipt this stage records (the manifest's
#: ``record_mass_conservation_receipt`` operation repeats it): the terminal
#: family gate requires exactly this reason on a valid mass-conserving record.
UK_WAS_LISA_MASS_CONSERVATION_REASON = (
    "WAS Lifetime ISA imputation on the source spine: household weights pass "
    "through unchanged and total household mass is conserved."
)
#: The household tab is the ``was_wealth`` donor (same pins); the person tab is
#: this stage's own.
UK_WAS_HOUSEHOLD_TAB_ROLE = "was_qrf_donor"
UK_WAS_PERSON_TAB_ROLE = "was_person_tab"

CLEAN_WAS_LISA_DONOR_KIND = "clean_was_lisa_donor"
IMPUTE_LIFETIME_ISA_OWNERSHIP_KIND = "impute_lifetime_isa_ownership"
IMPUTE_LIFETIME_ISA_BALANCE_KIND = "impute_lifetime_isa_balance"
CAP_LIFETIME_ISA_TO_FINANCIAL_WEALTH_KIND = "cap_lifetime_isa_to_financial_wealth"

OWNERSHIP_COLUMN = "has_lifetime_isa"
BALANCE_COLUMN = "lifetime_isa_balance"
HOUSEHOLD_BALANCE_COLUMN = "household_lifetime_isa_balance"
FINANCIAL_WEALTH_COLUMN = "gross_financial_wealth"
CAP_METHOD = "pro_rata_within_household"
UK_WAS_LISA_PERSON_OUTPUT_COLUMNS: tuple[str, ...] = (
    OWNERSHIP_COLUMN,
    BALANCE_COLUMN,
)
UK_WAS_LISA_HOUSEHOLD_OUTPUT_COLUMNS: tuple[str, ...] = (HOUSEHOLD_BALANCE_COLUMN,)
UK_WAS_LISA_OUTPUT_COLUMNS: tuple[str, ...] = (
    *UK_WAS_LISA_PERSON_OUTPUT_COLUMNS,
    *UK_WAS_LISA_HOUSEHOLD_OUTPUT_COLUMNS,
)
UK_WAS_LISA_NONNEGATIVE_OUTPUT_COLUMNS: tuple[str, ...] = (
    BALANCE_COLUMN,
    HOUSEHOLD_BALANCE_COLUMN,
)
UK_WAS_LISA_SUPPORT_CLIP_COLUMNS: tuple[str, ...] = (BALANCE_COLUMN,)
UK_WAS_LISA_FIT_NAME = "uk_was_round_8_lifetime_isa"

#: The household columns of the cleaned WAS household tab that condition the
#: person imputation (``clean_was_household_table`` names; ``is_renting`` there
#: is the private-renter flag ``DVPriRntR8 == 1``).
HOUSEHOLD_PREDICTORS: tuple[str, ...] = (
    "household_net_income",
    "num_adults",
    "num_children",
    "gross_financial_wealth",
    "savings",
    "cash_isa",
    "stocks_and_shares_isa",
)
PRIVATE_RENTER_COLUMN = "is_private_renter"
#: Engine inputs read at person grain on the recipients.
PERSON_ENGINE_PREDICTORS: tuple[str, ...] = (
    "employment_income",
    "self_employment_income",
)
PRIVATE_RENTER_TENURE = "RENT_PRIVATELY"

#: Donor response classes (evidence only; never predictors).
OBSERVED_NON_HOLDER = "observed_non_holder"
IMPUTED_NON_HOLDER = "imputed_non_holder"
OBSERVED_HOLDER_EXACT_VALUE = "observed_holder_exact_value"
HOLDER_BANDED_VALUE = "holder_banded_value"
HOLDER_IMPUTED_VALUE = "holder_imputed_value"
IMPUTED_HOLDER = "imputed_holder"
RULE_IMPOSSIBLE_HOLDER = "rule_impossible_holder"
RESPONSE_CLASSES: tuple[str, ...] = (
    OBSERVED_NON_HOLDER,
    IMPUTED_NON_HOLDER,
    OBSERVED_HOLDER_EXACT_VALUE,
    HOLDER_BANDED_VALUE,
    HOLDER_IMPUTED_VALUE,
    IMPUTED_HOLDER,
    RULE_IMPOSSIBLE_HOLDER,
)


class WASLISAError(ValueError):
    """Raised when the WAS tabs or the declaration do not carry what the stage needs."""


# --------------------------------------------------------------------------
# Declaration readers
# --------------------------------------------------------------------------


def _artifact(stage: SourceStageSpec, role: str) -> Mapping[str, Any]:
    for artifact in stage.artifacts:
        if artifact.get("role") == role:
            return artifact
    raise WASLISAError(f"{stage.stage} declares no {role!r} artifact.")


def _operation(stage: SourceStageSpec, kind: str) -> Mapping[str, Any]:
    for operation in stage.operations:
        if operation.kind == kind:
            return dict(operation.parameters)
    raise WASLISAError(f"{stage.stage} declares no {kind!r} operation.")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WASLISAError(f"{label} must be a mapping.")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise WASLISAError(f"{label} must be a list of names.")
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class WASLISAColumns:
    """The declared WAS column names and codes (one place to correct the codebook)."""

    household_key: str
    person_number: str
    is_dependent_child: str
    age_band: str
    sex: str
    employment_income: str
    self_employment_income: str
    holds_lifetime_isa: str
    holding_imputed: str
    reported_value: str
    value_imputed: str
    reported_band: str
    band_imputed: str
    lifetime_isa_balance: str
    household_aggregate: str
    aggregate_tolerance_gbp: float
    non_dependent_code: int
    female_code: int
    age_band_edges: tuple[int, ...]
    sentinel_codes: frozenset[float]
    sentinel_recode_columns: tuple[str, ...]
    household_predictors: tuple[str, ...]
    maximum_holder_age_band_lower: int
    balance_ceiling_gbp: float

    _PERSON_KEYS = (
        "person_number",
        "is_dependent_child",
        "age_band",
        "sex",
        "employment_income",
        "self_employment_income",
        "holds_lifetime_isa",
        "holding_imputed",
        "reported_value",
        "value_imputed",
        "reported_band",
        "band_imputed",
        "lifetime_isa_balance",
    )

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> WASLISAColumns:
        person = _mapping(parameters.get("person_columns"), "person_columns")
        missing = [key for key in cls._PERSON_KEYS if key not in person]
        if missing:
            raise WASLISAError(f"person_columns lacks {missing}.")
        rule = _mapping(parameters.get("credibility_rule"), "credibility_rule")
        edges = tuple(int(edge) for edge in parameters["age_band_edges"])
        if list(edges) != sorted(set(edges)) or not edges or edges[0] != 0:
            raise WASLISAError(
                "age_band_edges must be strictly increasing from 0 (the lower edge "
                "of each coded band, in code order)."
            )
        predictors = _strings(
            parameters.get("household_predictors", ()), "household_predictors"
        )
        if tuple(predictors) != HOUSEHOLD_PREDICTORS:
            raise WASLISAError(
                f"household_predictors drifted: manifest declares {predictors!r}, "
                f"runtime uses {HOUSEHOLD_PREDICTORS!r}."
            )
        recode = _strings(
            parameters.get("sentinel_recode_columns", ()), "sentinel_recode_columns"
        )
        unknown = sorted(set(recode) - set(PERSON_ENGINE_PREDICTORS))
        if unknown:
            raise WASLISAError(
                f"sentinel_recode_columns {unknown} are not person income predictors."
            )
        return cls(
            household_key=str(parameters["household_key"]),
            person_number=str(person["person_number"]),
            is_dependent_child=str(person["is_dependent_child"]),
            age_band=str(person["age_band"]),
            sex=str(person["sex"]),
            employment_income=str(person["employment_income"]),
            self_employment_income=str(person["self_employment_income"]),
            holds_lifetime_isa=str(person["holds_lifetime_isa"]),
            holding_imputed=str(person["holding_imputed"]),
            reported_value=str(person["reported_value"]),
            value_imputed=str(person["value_imputed"]),
            reported_band=str(person["reported_band"]),
            band_imputed=str(person["band_imputed"]),
            lifetime_isa_balance=str(person["lifetime_isa_balance"]),
            household_aggregate=str(parameters["household_aggregate"]),
            aggregate_tolerance_gbp=float(parameters["aggregate_tolerance_gbp"]),
            non_dependent_code=int(parameters["non_dependent_code"]),
            female_code=int(parameters["female_code"]),
            age_band_edges=edges,
            sentinel_codes=frozenset(
                float(code) for code in parameters["sentinel_codes"]
            ),
            sentinel_recode_columns=recode,
            household_predictors=predictors,
            maximum_holder_age_band_lower=int(rule["maximum_holder_age_band_lower"]),
            balance_ceiling_gbp=float(rule["balance_ceiling_gbp"]),
        )

    def person_raw_columns(self) -> tuple[str, ...]:
        """Every raw person-tab column the stage reads (the selective read)."""

        return (
            self.household_key,
            *(getattr(self, key) for key in self._PERSON_KEYS),
        )


@dataclass(frozen=True)
class OwnershipModelSpec:
    """The declared ownership model (``impute_lifetime_isa_ownership``)."""

    output: str
    seed: int
    salt: str
    penalty_c: float
    solver: str
    max_iter: int
    age_group_lower_bounds: tuple[int, ...]
    log1p_predictors: tuple[str, ...]
    indicator_predictors: tuple[str, ...]
    count_predictors: tuple[str, ...]
    minimum_age: int

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> OwnershipModelSpec:
        if parameters.get("output") != OWNERSHIP_COLUMN:
            raise WASLISAError(
                f"impute_lifetime_isa_ownership must write {OWNERSHIP_COLUMN!r}."
            )
        if parameters.get("model") != "weighted_ridge_logistic":
            raise WASLISAError("the ownership model must be weighted_ridge_logistic.")
        if parameters.get("standardise") != "donor_unweighted_mean_sd":
            raise WASLISAError("ownership predictors standardise on the donor.")
        bounds = tuple(int(b) for b in parameters["age_group_lower_bounds"])
        if list(bounds) != sorted(set(bounds)) or not bounds or bounds[0] != 0:
            raise WASLISAError(
                "age_group_lower_bounds must be strictly increasing from 0."
            )
        population = _mapping(parameters.get("population"), "population")
        penalty_c = float(parameters["penalty_c"])
        if not np.isfinite(penalty_c) or penalty_c <= 0:
            raise WASLISAError("penalty_c must be a positive finite number.")
        return cls(
            output=OWNERSHIP_COLUMN,
            seed=int(parameters["seed"]),
            salt=str(parameters["salt"]),
            penalty_c=penalty_c,
            solver=str(parameters["solver"]),
            max_iter=int(parameters["max_iter"]),
            age_group_lower_bounds=bounds,
            log1p_predictors=_strings(
                parameters.get("log1p_predictors", ()), "log1p_predictors"
            ),
            indicator_predictors=_strings(
                parameters.get("indicator_predictors", ()), "indicator_predictors"
            ),
            count_predictors=_strings(
                parameters.get("count_predictors", ()), "count_predictors"
            ),
            minimum_age=int(population["minimum_age"]),
        )

    def feature_names(self) -> tuple[str, ...]:
        return (
            *(f"age_group_{lower}" for lower in self.age_group_lower_bounds),
            *(f"log1p_{column}" for column in self.log1p_predictors),
            *self.indicator_predictors,
            *self.count_predictors,
        )

    def group_labels(self) -> tuple[str, ...]:
        """Recipient-facing labels (the first group starts at the minimum age)."""

        labels = []
        bounds = self.age_group_lower_bounds
        for index, lower in enumerate(bounds):
            start = max(lower, self.minimum_age) if index == 0 else lower
            if index + 1 < len(bounds):
                labels.append(f"{start}-{bounds[index + 1] - 1}")
            else:
                labels.append(f"{start}+")
        return tuple(labels)


@dataclass(frozen=True)
class BalanceModelSpec:
    """The declared balance model (``impute_lifetime_isa_balance``)."""

    output: str
    seed: int
    salt: str
    n_estimators: int
    predictors: tuple[str, ...]
    expected_regime: str

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> BalanceModelSpec:
        if parameters.get("output") != BALANCE_COLUMN:
            raise WASLISAError(
                f"impute_lifetime_isa_balance must write {BALANCE_COLUMN!r}."
            )
        if parameters.get("model") != "regime_gated_qrf":
            raise WASLISAError("the balance model must be regime_gated_qrf.")
        if parameters.get("training_rows") != "credible_holders":
            raise WASLISAError("the balance model trains on the credible holders.")
        n_estimators = int(parameters.get("n_estimators", 100))
        if n_estimators < 1:
            raise WASLISAError("n_estimators must be positive.")
        return cls(
            output=BALANCE_COLUMN,
            seed=int(parameters["seed"]),
            salt=str(parameters["salt"]),
            n_estimators=n_estimators,
            predictors=_strings(parameters.get("predictors", ()), "predictors"),
            expected_regime=str(parameters["expected_regime"]),
        )


# --------------------------------------------------------------------------
# Donor
# --------------------------------------------------------------------------


def _lower_lookup(table: pd.DataFrame, label: str) -> dict[str, str]:
    lowered: dict[str, str] = {}
    for column in table.columns:
        key = str(column).lower()
        if key in lowered:
            raise WASLISAError(f"the WAS {label} tab has duplicate column {key!r}.")
        lowered[key] = column
    return lowered


def _column(
    table: pd.DataFrame, lowered: Mapping[str, str], name: str, label: str
) -> pd.Series:
    actual = lowered.get(name.lower())
    if actual is None:
        raise WASLISAError(f"the WAS {label} tab lacks the declared column {name!r}.")
    return pd.to_numeric(table[actual], errors="coerce")


def _age_band_index(values: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    """The 0-based band of each age (the ``nts_bus_travel`` age-band convention)."""

    bounds = np.asarray(list(edges)[1:], dtype=float)
    return np.searchsorted(bounds, np.asarray(values, dtype=float), side="right")


def _age_group_index(values: np.ndarray, bounds: Sequence[int]) -> np.ndarray:
    return np.searchsorted(
        np.asarray(list(bounds)[1:], dtype=float),
        np.asarray(values, dtype=float),
        side="right",
    )


def _weighted_share(mask: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(weights.sum())
    if total <= 0:
        return None
    return float(weights[mask].sum() / total)


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]
) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if values.size == 0:
        return {f"p{int(round(q * 100))}": None for q in quantiles}
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return {
        f"p{int(round(q * 100))}": float(
            values[order][
                min(
                    int(np.searchsorted(cumulative, q * cumulative[-1])),
                    values.size - 1,
                )
            ]
        )
        for q in quantiles
    }


@dataclass(frozen=True)
class WASLISADonor:
    """The cleaned donor adults and the cleaning receipt."""

    person: pd.DataFrame
    receipt: Mapping[str, Any]


def clean_was_lisa_donor(
    person_raw: pd.DataFrame,
    household_raw: pd.DataFrame,
    *,
    columns: WASLISAColumns,
) -> WASLISADonor:
    """Join WAS persons to their cleaned household and apply the declared rules.

    Returns one row per non-dependent adult with the person and household
    predictors, the design weight (the household cross-sectional weight each
    person carries), ``holds`` after the credibility rule, the credible balance
    (0 for non-holders, NaN for holders above the declared ceiling, which leave
    the balance fit) and the response class. Refuses an orphan person, a
    missing or negative released value, an age band outside the declared codes,
    and a person total that does not reproduce the household aggregate.
    """

    person_lower = _lower_lookup(person_raw, "person")
    household_lower = _lower_lookup(household_raw, "household")
    household = clean_was_household_table(household_raw)
    if not household.index.equals(household_raw.index):
        raise WASLISAError("the cleaned WAS household table lost its row alignment.")
    household = household.assign(
        household_key=_column(
            household_raw, household_lower, columns.household_key, "household"
        ).to_numpy(),
        household_aggregate=_column(
            household_raw, household_lower, columns.household_aggregate, "household"
        ).to_numpy(),
    )
    if household["household_key"].isna().any():
        raise WASLISAError("the WAS household tab has a missing case number.")
    if household["household_key"].duplicated().any():
        raise WASLISAError("the WAS household tab repeats a case number.")
    raw = pd.DataFrame(
        {
            "household_key": _column(
                person_raw, person_lower, columns.household_key, "person"
            ),
            **{
                key: _column(person_raw, person_lower, getattr(columns, key), "person")
                for key in WASLISAColumns._PERSON_KEYS
            },
        }
    )
    if raw["household_key"].isna().any():
        raise WASLISAError("the WAS person tab has a missing case number.")
    if raw.duplicated(["household_key", "person_number"]).any():
        raise WASLISAError("the WAS person tab repeats a (case, person) pair.")
    orphans = ~raw["household_key"].isin(household["household_key"])
    if orphans.any():
        raise WASLISAError(
            f"{int(orphans.sum())} WAS persons have no household row; the person and "
            "household tabs are not the same deposit."
        )
    released = raw["lifetime_isa_balance"]
    # The pin check: the released person values reproduce the household
    # aggregate on every household (the ONS aggregate is the sum of the person
    # values), which also proves the column is the one the aggregate sums.
    person_totals = (
        released.clip(lower=0.0)
        .fillna(0.0)
        .groupby(raw["household_key"].to_numpy())
        .sum()
    )
    aggregate = household.set_index("household_key")["household_aggregate"].fillna(0.0)
    difference = (aggregate - person_totals.reindex(aggregate.index).fillna(0.0)).abs()
    if float(difference.max()) > columns.aggregate_tolerance_gbp:
        raise WASLISAError(
            f"person {columns.lifetime_isa_balance} does not sum to the household "
            f"{columns.household_aggregate} on {int((difference > columns.aggregate_tolerance_gbp).sum())} "
            f"households (max difference {float(difference.max()):.2f})."
        )
    joined = raw.merge(
        household[
            [
                "household_key",
                "weight",
                "is_renting",
                *HOUSEHOLD_PREDICTORS,
            ]
        ],
        on="household_key",
        how="left",
        validate="many_to_one",
    )
    dependent = joined["is_dependent_child"] != columns.non_dependent_code
    dependent_holders = int((dependent & (joined["lifetime_isa_balance"] > 0)).sum())
    adults = joined.loc[~dependent].reset_index(drop=True)
    value = adults["lifetime_isa_balance"]
    if value.isna().any() or (value < 0).any():
        raise WASLISAError(
            f"{columns.lifetime_isa_balance} is missing or negative for "
            f"{int((value.isna() | (value < 0)).sum())} donor adults; a missing value "
            "is never read as a zero balance."
        )
    band_code = adults["age_band"]
    valid_band = band_code.between(1, len(columns.age_band_edges))
    if not valid_band.all():
        raise WASLISAError(
            f"{int((~valid_band).sum())} donor adults carry an age band outside codes "
            f"1..{len(columns.age_band_edges)}."
        )
    band = band_code.astype(np.int64).to_numpy() - 1
    band_lower = np.asarray(columns.age_band_edges, dtype=float)[band]
    weight = adults["weight"].to_numpy(dtype=float)
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise WASLISAError("donor adults must carry a positive finite weight.")
    sentinel_recodes: dict[str, int] = {}
    incomes: dict[str, np.ndarray] = {}
    for column in PERSON_ENGINE_PREDICTORS:
        values = adults[column].astype(float)
        if column in columns.sentinel_recode_columns:
            is_sentinel = values.isin(sorted(columns.sentinel_codes))
            sentinel_recodes[column] = int(is_sentinel.sum())
            values = values.where(~is_sentinel, 0.0)
        incomes[column] = values.fillna(0.0).to_numpy(dtype=float)
    holds_released = value.to_numpy(dtype=float) > 0.0
    flag = adults["holds_lifetime_isa"].to_numpy(dtype=float) == 1.0
    rule_impossible = holds_released & (
        band_lower >= columns.maximum_holder_age_band_lower
    )
    holds = holds_released & ~rule_impossible
    above_ceiling = holds & (value.to_numpy(dtype=float) > columns.balance_ceiling_gbp)
    balance = np.where(holds, value.to_numpy(dtype=float), 0.0)
    balance[above_ceiling] = np.nan
    response = np.full(len(adults), OBSERVED_NON_HOLDER, dtype=object)
    holding_imputed = adults["holding_imputed"].to_numpy(dtype=float) == 1.0
    response[~holds_released & holding_imputed] = IMPUTED_NON_HOLDER
    reported = adults["reported_value"].to_numpy(dtype=float)
    value_imputed = adults["value_imputed"].to_numpy(dtype=float) == 1.0
    banded = adults["reported_band"].between(1, 8).to_numpy() | (
        adults["band_imputed"].to_numpy(dtype=float) == 1.0
    )
    holder = holds_released & ~rule_impossible
    response[holder & ~holding_imputed & (reported > 0) & ~value_imputed] = (
        OBSERVED_HOLDER_EXACT_VALUE
    )
    response[
        holder & ~holding_imputed & ~((reported > 0) & ~value_imputed) & banded
    ] = HOLDER_BANDED_VALUE
    response[
        holder & ~holding_imputed & ~((reported > 0) & ~value_imputed) & ~banded
    ] = HOLDER_IMPUTED_VALUE
    response[holder & holding_imputed] = IMPUTED_HOLDER
    response[rule_impossible] = RULE_IMPOSSIBLE_HOLDER
    person = pd.DataFrame(
        {
            "household_key": adults["household_key"].to_numpy(),
            "person_number": adults["person_number"].to_numpy(),
            "age_band": band.astype(np.int64),
            "age_floor": band_lower,
            "is_female": (
                adults["sex"].to_numpy(dtype=float) == columns.female_code
            ).astype(float),
            "employment_income": incomes["employment_income"],
            "self_employment_income": incomes["self_employment_income"],
            **{
                column: adults[column].astype(float).to_numpy()
                for column in HOUSEHOLD_PREDICTORS
            },
            PRIVATE_RENTER_COLUMN: adults["is_renting"].astype(float).to_numpy(),
            "weight": weight,
            "holds": holds.astype(np.int64),
            BALANCE_COLUMN: balance,
            "lisa_response_class": response,
        }
    )
    released_values = value.to_numpy(dtype=float)
    mass = released_values * weight
    total_mass = float(mass[holds_released].sum())
    holder_weight = float(weight[holds_released].sum())
    classes = {
        name: {
            "persons": int((response == name).sum()),
            "weight": float(weight[response == name].sum()),
        }
        for name in RESPONSE_CLASSES
    }
    receipt = {
        "persons_read": int(len(raw)),
        "households": int(len(household)),
        "donor_adults": int(len(adults)),
        "dependent_children_excluded": int(dependent.sum()),
        "dependent_children_with_a_value": dependent_holders,
        "holders_released": int(holds_released.sum()),
        "holders_after_rule": int(holds.sum()),
        "credible_holders_in_balance_fit": int((holds & ~above_ceiling).sum()),
        "released_flag_value_disagreements": int((flag != holds_released).sum()),
        "response_classes": classes,
        "sentinel_recodes": sentinel_recodes,
        "aggregate_check": {
            "households_checked": int(len(aggregate)),
            "max_abs_difference_gbp": float(difference.max()),
            "tolerance_gbp": columns.aggregate_tolerance_gbp,
        },
        "credibility_rule": {
            "maximum_holder_age_band_lower": columns.maximum_holder_age_band_lower,
            "balance_ceiling_gbp": columns.balance_ceiling_gbp,
            "holders_recoded_for_age": int(rule_impossible.sum()),
            "recoded_weighted_share_of_holders": (
                float(weight[rule_impossible].sum() / holder_weight)
                if holder_weight > 0
                else None
            ),
            "recoded_weighted_share_of_lisa_mass": (
                float(mass[rule_impossible].sum() / total_mass)
                if total_mass > 0
                else None
            ),
            "holders_above_ceiling": int(above_ceiling.sum()),
            "above_ceiling_weighted_share_of_holders": (
                float(weight[above_ceiling].sum() / holder_weight)
                if holder_weight > 0
                else None
            ),
            "above_ceiling_weighted_share_of_lisa_mass": (
                float(mass[above_ceiling].sum() / total_mass)
                if total_mass > 0
                else None
            ),
        },
        "weighted_adult_ownership_share": _weighted_share(holds, weight),
        "weighted_adult_ownership_share_released": _weighted_share(
            holds_released, weight
        ),
        "credible_holder_balance_quantiles": _weighted_quantiles(
            balance[holds & ~above_ceiling],
            weight[holds & ~above_ceiling],
            (0.1, 0.25, 0.5, 0.75, 0.9),
        ),
        "credible_holder_weighted_mean_balance": (
            float(
                np.average(
                    balance[holds & ~above_ceiling],
                    weights=weight[holds & ~above_ceiling],
                )
            )
            if (holds & ~above_ceiling).any()
            else None
        ),
    }
    return WASLISADonor(person=person, receipt=receipt)


def was_lisa_support_ranges(donor: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Exact donor ranges of the clipped columns (in-run clip and bounds tool)."""

    ranges: dict[str, tuple[float, float]] = {}
    for column in UK_WAS_LISA_SUPPORT_CLIP_COLUMNS:
        values = pd.to_numeric(donor[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            ranges[column] = (float(finite.min()), float(finite.max()))
    return ranges


# --------------------------------------------------------------------------
# Recipients
# --------------------------------------------------------------------------


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def recipient_predictors(
    frame: Frame,
    engine: object,
    *,
    age_band_edges: Sequence[int],
) -> pd.DataFrame:
    """Person-grain predictors of the FRS spine in the donor's vocabulary.

    Household predictors reuse ``was_wealth.recipient_predictors`` (the same
    definitions the WAS household donor is read with); earnings are the engine's
    person inputs; tenure is the private-renter flag; the financial columns are
    the household's ``was_wealth`` draws. Rows align with the person table.
    """

    person = frame.table("person")
    household = frame.table("household")
    for column in ("person_id", "person_household_id", "age", "gender"):
        if column not in person:
            raise WASLISAError(f"recipient person table lacks {column!r}.")
    for column in (
        "household_id",
        "tenure_type",
        "gross_financial_wealth",
        "savings",
        "cash_isa",
        "stocks_and_shares_isa",
    ):
        if column not in household:
            raise WASLISAError(
                f"recipient household table lacks {column!r} (the was_wealth draw "
                "runs first)."
            )
    household_predictors = was_household_predictors(frame, engine)
    by_household = pd.DataFrame(
        {
            "household_net_income": household_predictors[
                "household_net_income"
            ].to_numpy(dtype=float),
            "num_adults": household_predictors["num_adults"].to_numpy(dtype=float),
            "num_children": household_predictors["num_children"].to_numpy(dtype=float),
            **{
                column: pd.to_numeric(household[column], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
                for column in (
                    "gross_financial_wealth",
                    "savings",
                    "cash_isa",
                    "stocks_and_shares_isa",
                )
            },
            PRIVATE_RENTER_COLUMN: (
                household["tenure_type"].map(_enum_name) == PRIVATE_RENTER_TENURE
            )
            .astype(float)
            .to_numpy(),
        },
        index=pd.Index(household["household_id"].to_numpy()),
    )
    materialized = engine.materialize(
        frame, PERSON_ENGINE_PREDICTORS, uk_time_period(frame)
    )
    for column in PERSON_ENGINE_PREDICTORS:
        declared = str(engine.variable_metadata(column).entity)
        if declared != "person":
            raise WASLISAError(
                f"engine declares {column!r} at entity {declared!r}; the LISA stage "
                "reads it per person."
            )
    earnings = {}
    for column in PERSON_ENGINE_PREDICTORS:
        values = np.asarray(materialized[column], dtype=float)
        if values.shape != (len(person),):
            raise WASLISAError(
                f"materialized {column!r} has shape {values.shape}; expected "
                f"({len(person)},)."
            )
        earnings[column] = values
    household_ids = person["person_household_id"].to_numpy()
    mapped = by_household.reindex(household_ids)
    if mapped.isna().any().any():
        raise WASLISAError("every recipient person must map to a household row.")
    age = (
        pd.to_numeric(person["age"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    result = pd.DataFrame(
        {
            "person_id": person["person_id"].to_numpy(),
            "person_household_id": household_ids,
            "age": age,
            "age_floor": age,
            "age_band": _age_band_index(age, age_band_edges).astype(np.int64),
            "is_female": person["gender"]
            .map(_enum_name)
            .str.upper()
            .isin(("FEMALE", "F", "2"))
            .astype(float)
            .to_numpy(),
            **earnings,
        },
        index=person.index,
    )
    for column in mapped.columns:
        result[column] = mapped[column].to_numpy(dtype=float)
    return result


# --------------------------------------------------------------------------
# Ownership: weighted ridge logistic, identity-keyed draw
# --------------------------------------------------------------------------


def ownership_design(frame: pd.DataFrame, spec: OwnershipModelSpec) -> pd.DataFrame:
    """The ownership features of donor or recipient rows, in declared order."""

    groups = _age_group_index(
        frame["age_floor"].to_numpy(dtype=float), spec.age_group_lower_bounds
    )
    features: dict[str, np.ndarray] = {}
    for index, lower in enumerate(spec.age_group_lower_bounds):
        features[f"age_group_{lower}"] = (groups == index).astype(float)
    for column in spec.log1p_predictors:
        features[f"log1p_{column}"] = np.log1p(
            np.clip(frame[column].to_numpy(dtype=float), 0.0, None)
        )
    for column in (*spec.indicator_predictors, *spec.count_predictors):
        features[column] = frame[column].to_numpy(dtype=float)
    design = pd.DataFrame(features, index=frame.index)
    if not np.isfinite(design.to_numpy()).all():
        raise WASLISAError("ownership predictors must be finite.")
    return design


@dataclass(frozen=True)
class FittedOwnershipModel:
    spec: OwnershipModelSpec
    mean: np.ndarray
    scale: np.ndarray
    model: Any

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        design = ownership_design(frame, self.spec).to_numpy(dtype=float)
        standardised = (design - self.mean) / self.scale
        return np.asarray(self.model.predict_proba(standardised)[:, 1], dtype=float)


def fit_ownership_model(
    donor: pd.DataFrame, spec: OwnershipModelSpec
) -> tuple[FittedOwnershipModel, dict[str, Any]]:
    """Fit the weighted ridge logistic on the donor adults."""

    from sklearn.linear_model import LogisticRegression

    design = ownership_design(donor, spec)
    target = donor["holds"].to_numpy(dtype=np.int64)
    if target.sum() == 0 or target.sum() == len(target):
        raise WASLISAError("the ownership model needs both holders and non-holders.")
    weights = donor["weight"].to_numpy(dtype=float)
    values = design.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1) if len(values) > 1 else np.ones(values.shape[1])
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    model = LogisticRegression(
        C=spec.penalty_c, solver=spec.solver, max_iter=spec.max_iter
    )
    model.fit((values - mean) / scale, target, sample_weight=weights / weights.mean())
    iterations = int(np.max(model.n_iter_))
    if iterations >= spec.max_iter:
        raise WASLISAError(
            f"the ownership logistic did not converge in {spec.max_iter} iterations."
        )
    fitted = FittedOwnershipModel(spec=spec, mean=mean, scale=scale, model=model)
    probability = fitted.probabilities(donor)
    groups = _age_group_index(
        donor["age_floor"].to_numpy(dtype=float), spec.age_group_lower_bounds
    )
    by_group = {}
    for index, lower in enumerate(spec.age_group_lower_bounds):
        mask = groups == index
        if mask.any():
            by_group[str(lower)] = {
                "donor_share": _weighted_share(target[mask] == 1, weights[mask]),
                "model_mean_probability": float(
                    np.average(probability[mask], weights=weights[mask])
                ),
                "donor_adults": int(mask.sum()),
            }
    receipt = {
        "model": "weighted_ridge_logistic",
        "penalty_c": spec.penalty_c,
        "solver": spec.solver,
        "iterations": iterations,
        "features": list(spec.feature_names()),
        "standardisation": {
            "basis": "donor_unweighted_mean_sd",
            "mean": {
                n: float(m) for n, m in zip(spec.feature_names(), mean, strict=True)
            },
            "scale": {
                n: float(s) for n, s in zip(spec.feature_names(), scale, strict=True)
            },
        },
        "intercept": float(model.intercept_[0]),
        "coefficients": {
            n: float(c)
            for n, c in zip(spec.feature_names(), model.coef_[0], strict=True)
        },
        "donor_weighted_ownership_share": _weighted_share(target == 1, weights),
        "donor_weighted_mean_probability": float(
            np.average(probability, weights=weights)
        ),
        "by_age_group": by_group,
        "donor_adults": int(len(donor)),
        "donor_holders": int(target.sum()),
    }
    return fitted, receipt


@dataclass(frozen=True)
class OwnershipImputation:
    holds: np.ndarray
    probability: np.ndarray
    receipt: Mapping[str, Any]
    fit_weight_records: tuple[FitWeightRecord, ...]


def impute_lifetime_isa_ownership(
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    *,
    spec: OwnershipModelSpec,
) -> OwnershipImputation:
    """Draw ownership for recipient adults from the donor-fitted logistic."""

    fitted, receipt = fit_ownership_model(donor, spec)
    probability = fitted.probabilities(recipient)
    uniforms = stable_identity_uniforms(
        recipient["person_id"].to_numpy(), seed=spec.seed, salt=spec.salt
    )
    adult = recipient["age"].to_numpy(dtype=float) >= spec.minimum_age
    holds = adult & (uniforms < probability)
    probability = np.where(adult, probability, 0.0)
    receipt = {
        **receipt,
        "seed": spec.seed,
        "salt": spec.salt,
        "draw": "holds when the person-keyed uniform falls below the probability",
        "minimum_age": spec.minimum_age,
    }
    records = (
        FitWeightRecord(f"{UK_WAS_LISA_FIT_NAME}:{OWNERSHIP_COLUMN}", EXPLICIT_KIND),
    )
    return OwnershipImputation(
        holds=holds,
        probability=probability,
        receipt=receipt,
        fit_weight_records=records,
    )


# --------------------------------------------------------------------------
# Balance: holders-only QRF at the owner's keyed quantile
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BalanceImputation:
    balance: np.ndarray
    receipt: Mapping[str, Any]
    fit_weight_records: tuple[FitWeightRecord, ...]


def impute_lifetime_isa_balance(
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    *,
    holds: np.ndarray,
    spec: BalanceModelSpec,
) -> BalanceImputation:
    """Draw each owner's balance from the QRF fitted on the credible holders."""

    from microcosm.fit.qrf import RegimeGatedQRF

    predictors = list(spec.predictors)
    missing = sorted({c for c in predictors if c not in donor or c not in recipient})
    if missing:
        raise WASLISAError(f"balance predictors {missing} are not on both sides.")
    value = donor[BALANCE_COLUMN].to_numpy(dtype=float)
    credible = (donor["holds"].to_numpy() == 1) & np.isfinite(value) & (value > 0)
    if not credible.any():
        raise WASLISAError("the balance model has no credible donor holders.")
    train = donor.loc[credible, [*predictors, BALANCE_COLUMN, "weight"]].astype(float)
    train = train.reset_index(drop=True)
    fitted = RegimeGatedQRF(n_estimators=spec.n_estimators, seed=spec.seed).fit(
        train, predictors, [BALANCE_COLUMN], weights="weight"
    )
    regime = str(fitted.regimes().get(BALANCE_COLUMN, ""))
    if regime != spec.expected_regime:
        raise WASLISAError(
            f"the balance model fitted regime {regime!r}, not the declared "
            f"{spec.expected_regime!r}."
        )
    holds = np.asarray(holds, dtype=bool)
    balance = np.zeros(len(recipient), dtype=float)
    if holds.any():
        owners = recipient.loc[holds, predictors].astype(float).reset_index(drop=True)
        quantiles = stable_identity_uniforms(
            recipient.loc[holds, "person_id"].to_numpy(), seed=spec.seed, salt=spec.salt
        )
        drawn = fitted.predict_positive_from_uniforms(
            owners, quantiles={BALANCE_COLUMN: quantiles}
        )[BALANCE_COLUMN].to_numpy(dtype=float)
        balance[holds] = drawn
    receipt = {
        "model": "regime_gated_qrf",
        "regime": regime,
        "training_rows": int(credible.sum()),
        "distinct_training_values": int(np.unique(value[credible]).size),
        "n_estimators": spec.n_estimators,
        "predictors": predictors,
        "seed": spec.seed,
        "salt": spec.salt,
        "draw": "predict_positive_from_uniforms at the owner-keyed quantile",
        "owners_drawn": int(holds.sum()),
    }
    records = (
        FitWeightRecord(f"{UK_WAS_LISA_FIT_NAME}:{BALANCE_COLUMN}", fitted.weight_kind),
    )
    return BalanceImputation(
        balance=balance, receipt=receipt, fit_weight_records=records
    )


# --------------------------------------------------------------------------
# Coherence with gross financial wealth, household totals
# --------------------------------------------------------------------------


def cap_lifetime_isa_to_financial_wealth(
    *,
    person_household_ids: np.ndarray,
    balances: np.ndarray,
    household_ids: np.ndarray,
    household_financial_wealth: np.ndarray,
    household_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale a household's balances down pro rata to its gross financial wealth.

    WAS counts LISAs inside gross financial wealth, so a household total above
    the household's ``gross_financial_wealth`` draw is incoherent; its persons'
    balances are multiplied by ``wealth / total`` (zero when the wealth is not
    positive). ``was_wealth`` columns are never rewritten.
    """

    balances = np.asarray(balances, dtype=float)
    frame = pd.DataFrame(
        {"household": np.asarray(person_household_ids), "balance": balances}
    )
    totals = frame.groupby("household")["balance"].sum()
    wealth = pd.Series(
        np.asarray(household_financial_wealth, dtype=float),
        index=pd.Index(np.asarray(household_ids)),
    )
    weight = pd.Series(
        np.asarray(household_weights, dtype=float),
        index=pd.Index(np.asarray(household_ids)),
    )
    wealth_at = wealth.reindex(totals.index)
    if wealth_at.isna().any():
        raise WASLISAError("every person must map to a household's financial wealth.")
    over = totals > wealth_at.clip(lower=0.0) + 1e-9
    factor = pd.Series(1.0, index=totals.index)
    positive = over & (wealth_at > 0)
    factor[positive] = wealth_at[positive] / totals[positive]
    factor[over & ~(wealth_at > 0)] = 0.0
    person_factor = factor.reindex(frame["household"]).to_numpy(dtype=float)
    capped = balances * person_factor
    person_weight = weight.reindex(frame["household"]).to_numpy(dtype=float)
    removed = balances - capped
    total_mass = float((balances * person_weight).sum())
    receipt = {
        "households_over_financial_wealth": int(over.sum()),
        "households_over_with_zero_financial_wealth": int(
            (over & ~(wealth_at > 0)).sum()
        ),
        "persons_scaled": int(((removed > 0) & (capped > 0)).sum()),
        "owners_cleared": int(((balances > 0) & (capped <= 0)).sum()),
        "gbp_removed_unweighted": float(removed.sum()),
        "gbp_removed_weighted": float((removed * person_weight).sum()),
        "share_of_weighted_lisa_mass_removed": (
            float((removed * person_weight).sum() / total_mass)
            if total_mass > 0
            else None
        ),
        "rule": "household LISA total <= household gross_financial_wealth (pro rata)",
    }
    return capped, receipt


def _household_totals(
    person_household_ids: np.ndarray, values: np.ndarray, household_ids: np.ndarray
) -> np.ndarray:
    totals = (
        pd.Series(
            np.asarray(values, dtype=float), index=np.asarray(person_household_ids)
        )
        .groupby(level=0)
        .sum()
    )
    return totals.reindex(np.asarray(household_ids)).fillna(0.0).to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Evidence of the realised draw
# --------------------------------------------------------------------------


def _tertiles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    if cumulative.size == 0 or cumulative[-1] <= 0:
        return np.zeros(len(values), dtype=np.int64)
    cuts = np.interp([1 / 3, 2 / 3], cumulative / cumulative[-1], values[order])
    return np.digitize(values, cuts).astype(np.int64)


def _profile(
    frame: pd.DataFrame,
    holds: np.ndarray,
    weights: np.ndarray,
    spec: OwnershipModelSpec,
    *,
    channel: np.ndarray | None = None,
) -> dict[str, Any]:
    groups = _age_group_index(
        frame["age_floor"].to_numpy(dtype=float), spec.age_group_lower_bounds
    )
    labels = spec.group_labels()
    female = frame["is_female"].to_numpy(dtype=float) == 1.0
    renter = frame[PRIVATE_RENTER_COLUMN].to_numpy(dtype=float) == 1.0
    tertile = _tertiles(frame["household_net_income"].to_numpy(dtype=float), weights)

    def share(mask: np.ndarray) -> float | None:
        return _weighted_share(holds[mask], weights[mask]) if mask.any() else None

    profile: dict[str, Any] = {
        "all_adults": _weighted_share(holds, weights),
        "by_age_group": {
            labels[index]: share(groups == index) for index in range(len(labels))
        },
        "by_sex": {"female": share(female), "male": share(~female)},
        "by_private_renting": {
            "private_renter": share(renter),
            "other": share(~renter),
        },
        "by_household_net_income_tertile": {
            str(index + 1): share(tertile == index) for index in range(3)
        },
        "adults": int(len(frame)),
        "holders": int(np.asarray(holds, dtype=bool).sum()),
    }
    if channel is not None:
        profile["by_channel"] = {
            str(value): share(channel == value) for value in sorted(set(channel))
        }
    return profile


def _co_holding_share(
    household_ids: np.ndarray, holds: np.ndarray, weights: np.ndarray
) -> float | None:
    table = pd.DataFrame(
        {"household": household_ids, "holds": holds.astype(int), "weight": weights}
    )
    owners = (
        table[table["holds"] == 1]
        .groupby("household")
        .agg(owners=("holds", "sum"), weight=("weight", "first"))
    )
    if owners.empty:
        return None
    return float(
        owners.loc[owners["owners"] > 1, "weight"].sum() / owners["weight"].sum()
    )


@dataclass
class UKWASLISAResult:
    frame: Frame
    support_clip: UKSupportClipReceipt
    donor: Mapping[str, Any]
    ownership_model: Mapping[str, Any]
    balance_model: Mapping[str, Any]
    realised: Mapping[str, Any]
    financial_wealth_coherence: Mapping[str, Any]
    population: Mapping[str, Any]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_WAS_LISA_STAGE_NAME,
            "support_clip": self.support_clip.evidence(),
            "donor": dict(self.donor),
            "ownership_model": dict(self.ownership_model),
            "balance_model": dict(self.balance_model),
            "realised": dict(self.realised),
            "financial_wealth_coherence": dict(self.financial_wealth_coherence),
            "population": dict(self.population),
        }


# --------------------------------------------------------------------------
# Stage transform
# --------------------------------------------------------------------------


@dataclass
class UKWASLISAStageTransform:
    """Whole-stage callable for the WAS Lifetime ISA imputation."""

    stage: SourceStageSpec
    engine: object
    was_tab_path: str | Path | None = None
    was_person_tab_path: str | Path | None = None
    #: Raw tabs as read (tests and the synthetic fixture); column matching is
    #: case-insensitive either way.
    donor_household: pd.DataFrame | None = None
    donor_person: pd.DataFrame | None = None
    last_result: UKWASLISAResult | None = field(default=None, init=False)
    last_fit_weight_records: tuple[FitWeightRecord, ...] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        if self.last_fit_weight_records is None:
            return ()
        return tuple(self.last_fit_weight_records)

    def _table(
        self,
        supplied: pd.DataFrame | None,
        path: str | Path | None,
        role: str,
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if supplied is not None:
            return supplied
        if path is None:
            raise WASLISAError(
                f"{UK_WAS_LISA_STAGE_NAME} requires the caller-supplied {role} tab."
            )
        return read_pinned_tab(
            Path(path).expanduser().resolve(),
            _artifact(self.stage, role),
            columns=columns,
        )

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        columns = WASLISAColumns.from_parameters(
            _operation(self.stage, CLEAN_WAS_LISA_DONOR_KIND)
        )
        ownership_spec = OwnershipModelSpec.from_parameters(
            _operation(self.stage, IMPUTE_LIFETIME_ISA_OWNERSHIP_KIND)
        )
        balance_spec = BalanceModelSpec.from_parameters(
            _operation(self.stage, IMPUTE_LIFETIME_ISA_BALANCE_KIND)
        )
        cap = _operation(self.stage, CAP_LIFETIME_ISA_TO_FINANCIAL_WEALTH_KIND)
        if (
            cap.get("column") != BALANCE_COLUMN
            or cap.get("cap_column") != FINANCIAL_WEALTH_COLUMN
            or cap.get("method") != CAP_METHOD
            or cap.get("ownership_output") != OWNERSHIP_COLUMN
        ):
            raise WASLISAError(
                "cap_lifetime_isa_to_financial_wealth must cap "
                f"{BALANCE_COLUMN!r} at {FINANCIAL_WEALTH_COLUMN!r} "
                f"({CAP_METHOD}) and re-derive {OWNERSHIP_COLUMN!r}."
            )
        donor = clean_was_lisa_donor(
            self._table(
                self.donor_person,
                self.was_person_tab_path,
                UK_WAS_PERSON_TAB_ROLE,
                columns=columns.person_raw_columns(),
            ),
            self._table(
                self.donor_household, self.was_tab_path, UK_WAS_HOUSEHOLD_TAB_ROLE
            ),
            columns=columns,
        )
        recipient = recipient_predictors(
            frame, self.engine, age_band_edges=columns.age_band_edges
        )
        ownership = impute_lifetime_isa_ownership(
            donor.person, recipient, spec=ownership_spec
        )
        balance = impute_lifetime_isa_balance(
            donor.person, recipient, holds=ownership.holds, spec=balance_spec
        )
        clip = support_clip_to_donor_with_receipt(
            pd.DataFrame({BALANCE_COLUMN: balance.balance}, index=recipient.index),
            donor.person[[BALANCE_COLUMN]],
            columns=UK_WAS_LISA_SUPPORT_CLIP_COLUMNS,
            stage=UK_WAS_LISA_STAGE_NAME,
        )
        household_table = frame.table("household")
        household_weights = np.asarray(
            frame.weights_for("household").values, dtype=float
        )
        household_ids = household_table["household_id"].to_numpy()
        capped, coherence = cap_lifetime_isa_to_financial_wealth(
            person_household_ids=recipient["person_household_id"].to_numpy(),
            balances=clip.clipped[BALANCE_COLUMN].to_numpy(dtype=float),
            household_ids=household_ids,
            household_financial_wealth=pd.to_numeric(
                household_table[FINANCIAL_WEALTH_COLUMN], errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(dtype=float),
            household_weights=household_weights,
        )
        holds = capped > 0.0
        person = frame.table("person").copy()
        # Declared output order (the graph's cell order).
        person[OWNERSHIP_COLUMN] = holds
        person[BALANCE_COLUMN] = capped
        household = household_table.copy()
        household[HOUSEHOLD_BALANCE_COLUMN] = _household_totals(
            recipient["person_household_id"].to_numpy(), capped, household_ids
        )
        # Evidence.
        weight_by_household = pd.Series(household_weights, index=household_ids)
        person_weights = weight_by_household.reindex(
            recipient["person_household_id"].to_numpy()
        ).to_numpy(dtype=float)
        adult = recipient["age"].to_numpy(dtype=float) >= ownership_spec.minimum_age
        channel = None
        if "household_support_channel" in household_table:
            channel = (
                household_table.set_index("household_id")["household_support_channel"]
                .map(_enum_name)
                .reindex(recipient["person_household_id"].to_numpy())
                .astype(str)
                .to_numpy()
            )
        owner_weights = person_weights[holds]
        realised = {
            "recipient": _profile(
                recipient.loc[adult],
                holds[adult],
                person_weights[adult],
                ownership_spec,
                channel=None if channel is None else channel[adult],
            ),
            "donor": _profile(
                donor.person,
                donor.person["holds"].to_numpy() == 1,
                donor.person["weight"].to_numpy(dtype=float),
                ownership_spec,
            ),
            "owner_balance_quantiles": _weighted_quantiles(
                capped[holds], owner_weights, (0.1, 0.25, 0.5, 0.75, 0.9)
            ),
            "owner_weighted_mean_balance": (
                float(np.average(capped[holds], weights=owner_weights))
                if holds.any() and owner_weights.sum() > 0
                else None
            ),
            "weighted_total_gbp": float((capped * person_weights).sum()),
            "donor_credible_holder_balance_quantiles": donor.receipt[
                "credible_holder_balance_quantiles"
            ],
            "co_holding_share_of_owner_households": {
                "recipient": _co_holding_share(
                    recipient["person_household_id"].to_numpy(), holds, person_weights
                ),
                "donor": _co_holding_share(
                    donor.person["household_key"].to_numpy(),
                    donor.person["holds"].to_numpy() == 1,
                    donor.person["weight"].to_numpy(dtype=float),
                ),
            },
            "model_expected_adult_ownership_share": _weighted_expectation(
                ownership.probability[adult], person_weights[adult]
            ),
        }
        population = {
            "persons": int(len(recipient)),
            "adults": int(adult.sum()),
            "under_minimum_age_zeroed": int((~adult).sum()),
            "minimum_age": ownership_spec.minimum_age,
        }
        result = uk_national_frame(
            person=person,
            benunit=frame.table("benunit").copy(),
            household=household,
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=household_weights,
            mass_log=(
                *frame.mass_log,
                uk_household_mass_conservation_receipt(
                    frame, UK_WAS_LISA_MASS_CONSERVATION_REASON
                ),
            ),
        )
        validate_uk_national_frame(result)
        self.last_fit_weight_records = (
            *ownership.fit_weight_records,
            *balance.fit_weight_records,
        )
        self.last_result = UKWASLISAResult(
            frame=result,
            support_clip=clip.receipt,
            donor=donor.receipt,
            ownership_model=ownership.receipt,
            balance_model=balance.receipt,
            realised=realised,
            financial_wealth_coherence=coherence,
            population=population,
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_WAS_LISA_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            return {}
        return {"evidence": self.last_result.evidence()}


def _weighted_expectation(values: np.ndarray, weights: np.ndarray) -> float | None:
    if len(values) == 0 or float(np.sum(weights)) <= 0:
        return None
    return float(np.average(values, weights=weights))
