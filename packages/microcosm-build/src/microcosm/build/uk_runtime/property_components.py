"""Landlords' receipts and finance costs on the UK spine (microcosm#1106).

policyengine-uk 2.123.0 (PolicyEngine/policyengine-uk#2172) reads a landlord's
property business in three persisted inputs: ``property_income``, the profit
after allowable expenses and before the property allowance and before
dwelling-related finance costs (the SPI's concept); ``property_rental_income``,
the receipts before expenses, which decide whether the property allowance or
the expenses route applies (zero, or less than the profit, means unknown); and
``property_finance_costs``, the year's dwelling-related finance costs that
ITTOIA 2005 s. 272A restricts and s. 274A relieves as a tax reduction. The
``property_components`` stage fills the two missing pieces.

**Finance costs on the FRS concept.** The SPI channel draws its landlords'
restricted finance costs from the tape (``hmrc_spi_income_spine``). FRS
landlords report ROYYR1 after show card K6, which nets mortgage payments and
loan interest, so their profit sits below the SPI concept by the interest. The
stage draws their finance costs from the SPI 2022-23 tape, uprated to the
build period with the engine's indices for each variable, among the tape's
landlords whose profit after restricted finance costs is positive, the analog
of an FRS landlord reporting a profit:

* whether a landlord has finance costs: a person-keyed uniform below the
  tape's FACT-weighted share with finance costs in the landlord's cell of
  profit after finance costs (FACT-weighted deciles of the analog landlords,
  so every cell has tape support);
* how much: the regime-gated QRF fitted on the analog landlords with finance
  costs, given their profit after finance costs and their age, drawn at a
  second person-keyed quantile (``predict_positive_from_uniforms``).

The drawn costs are added back to the landlord's ``property_income`` (the
recommended ruling of decision 6, to be confirmed with Max before merge), so
the engine relieves the interest once, as a tax reduction. The card also nets
mortgage capital, which stays netted: a known shortfall in FRS profit.

**Receipts for every landlord.** No microdata source carries receipts beside
profit, so the stage constructs them from PRIS 2024-25 (the vendored
``hmrc_property_rental_income_facts.json``), the way the CGT stage allocates
its Table 3 bands: landlords are ranked by profit (both channels, after the
add-back), walked top band first into Table 13's eleven bands of rental
income at their individuals-basis landlord counts (one seeded systematic
offset, whole persons), and given receipts linear in their weight-ranked
position within the band, from its upper bound for the highest profit down
to its lower bound, never below their profit. The open top band takes
receipts proportional to profit, ``max(100,000, k x profit)``, with ``k >= 1``
solved so that receipts reach the individuals' Table 2 total. Receipts less
profit are the deductible expenses: PRIS's allowable expenses other than
residential finance costs.

Sub-letting rent (SUBRENT, on the household reference person) stays in
``property_income`` as reported until Rent a Room is released
(PolicyEngine/policyengine-uk#2002); its receipts are the same amount, with no
expenses. Landlords with no profit (FRS loss-makers, whose loss counts as
zero, and tape landlords at zero) get no receipts: a recorded residual, since
PRIS counts them among its landlords.
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
from microcosm.build.uk_runtime.hmrc_property_rental import (
    HMRC_PROPERTY_RENTAL_RECEIPTS_BAND_LOWER_BOUNDS,
    HMRC_PROPERTY_RENTAL_RECORD_SETS,
    HMRC_PROPERTY_RENTAL_RESOURCE,
    HMRCPropertyRentalFacts,
    load_hmrc_property_rental_facts,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_mass_conservation_receipt,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.spi_income import (
    SPI_DONOR_INCOME_YEAR,
    SPI_INCOME_UPRATING_VARIABLES,
    SPI_MINIMUM_RECIPIENT_AGE,
    SPI_RECIPIENT_ROLE_COLUMN,
    SPIDonorAgeModel,
    load_spi_donor_age_model,
    prepare_spi_donor_table,
    verify_spi_donor_identity,
)
from microcosm.build.uk_runtime.spi_support import (
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    support_channel_column,
)
from microcosm.frame import Frame

__all__ = [
    "ALLOCATE_PROPERTY_RENTAL_INCOME_KIND",
    "IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND",
    "UK_PROPERTY_COMPONENTS_MASS_CONSERVATION_REASON",
    "UK_PROPERTY_COMPONENTS_NONNEGATIVE_OUTPUT_COLUMNS",
    "UK_PROPERTY_COMPONENTS_OUTPUT_COLUMNS",
    "UK_PROPERTY_COMPONENTS_REWRITE_COLUMNS",
    "UK_PROPERTY_COMPONENTS_STAGE_NAME",
    "FinanceCostModelSpec",
    "PropertyComponentsError",
    "ReceiptsAllocationSpec",
    "UKPropertyComponentsStageTransform",
    "allocate_property_rental_income",
    "frs_concept_rental_profit",
    "impute_frs_property_finance_costs",
    "spi_tape_finance_cost_donor",
]

UK_PROPERTY_COMPONENTS_STAGE_NAME = "property_components"
#: The household-mass receipt this stage records (the manifest's
#: ``record_mass_conservation_receipt`` operation repeats it).
UK_PROPERTY_COMPONENTS_MASS_CONSERVATION_REASON = (
    "Landlords' receipts and finance costs on the source spine: household "
    "weights pass through unchanged and total household mass is conserved."
)

IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND = "impute_frs_property_finance_costs"
ALLOCATE_PROPERTY_RENTAL_INCOME_KIND = "allocate_property_rental_income"
VERIFY_VENDORED_FACT_RESOURCE_KIND = "verify_vendored_fact_resource"

PROFIT_COLUMN = "property_income"
FINANCE_COSTS_COLUMN = "property_finance_costs"
RECEIPTS_COLUMN = "property_rental_income"
SUBLET_COLUMN = "subrent"
HOUSEHOLD_HEAD_COLUMN = "is_household_head"
#: The tape's profit after restricted finance costs: the FRS PropRent concept.
PROFIT_AFTER_FINANCE_COSTS = "profit_after_finance_costs"

UK_PROPERTY_COMPONENTS_OUTPUT_COLUMNS: tuple[str, ...] = (RECEIPTS_COLUMN,)
UK_PROPERTY_COMPONENTS_REWRITE_COLUMNS: tuple[str, ...] = (
    PROFIT_COLUMN,
    FINANCE_COSTS_COLUMN,
)
UK_PROPERTY_COMPONENTS_NONNEGATIVE_OUTPUT_COLUMNS: tuple[str, ...] = (
    RECEIPTS_COLUMN,
    PROFIT_COLUMN,
    FINANCE_COSTS_COLUMN,
)
UK_PROPERTY_COMPONENTS_FIT_NAME = "uk_spi_2022_23_property_finance_costs"

#: The income stage's stage-1 seed: the tape is prepared exactly as
#: ``hmrc_spi_income_spine`` prepares it (ages drawn by ONS shares).
UK_PROPERTY_COMPONENTS_TAPE_SEED = 42

#: Coherence evidence: landlords whose household holds rental property.
_RENTAL_PROPERTY_COLUMNS = (
    "other_residential_property_value",
    "non_residential_property_value",
)
_BISECTION_STEPS = 200


class PropertyComponentsError(ValueError):
    """The stage's inputs or declarations cannot support the allocation."""


def _operation(stage: SourceStageSpec, kind: str) -> Mapping[str, Any]:
    matches = [op for op in stage.operations if op.kind == kind]
    if len(matches) != 1:
        raise PropertyComponentsError(
            f"{UK_PROPERTY_COMPONENTS_STAGE_NAME} must declare exactly one "
            f"{kind!r} operation; found {len(matches)}."
        )
    return matches[0].parameters


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]
) -> dict[str, float] | None:
    if len(values) == 0 or float(weights.sum()) <= 0.0:
        return None
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return {
        f"p{int(round(q * 100)):02d}": float(
            values[order][min(np.searchsorted(cumulative, q), len(values) - 1)]
        )
        for q in quantiles
    }


# --------------------------------------------------------------------------
# Declared parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FinanceCostModelSpec:
    """The ``impute_frs_property_finance_costs`` operation."""

    incidence_cells: int
    incidence_seed: int
    incidence_salt: str
    amount_seed: int
    amount_salt: str
    n_estimators: int
    predictors: tuple[str, ...]
    expected_regime: str
    tape_preparation_seed: int

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> FinanceCostModelSpec:
        cells = int(parameters["incidence_cells"])
        if cells < 1:
            raise PropertyComponentsError("incidence_cells must be at least 1.")
        predictors = tuple(str(v) for v in parameters["predictors"])
        if PROFIT_AFTER_FINANCE_COSTS not in predictors:
            raise PropertyComponentsError(
                f"the finance-cost model must condition on {PROFIT_AFTER_FINANCE_COSTS!r}."
            )
        if parameters.get("output") != FINANCE_COSTS_COLUMN:
            raise PropertyComponentsError(
                f"impute_frs_property_finance_costs must write {FINANCE_COSTS_COLUMN!r}."
            )
        if parameters.get("add_back_to") != PROFIT_COLUMN:
            raise PropertyComponentsError(
                f"the drawn finance costs must be added back to {PROFIT_COLUMN!r}."
            )
        spec = cls(
            incidence_cells=cells,
            incidence_seed=int(parameters["incidence_seed"]),
            incidence_salt=str(parameters["incidence_salt"]),
            amount_seed=int(parameters["amount_seed"]),
            amount_salt=str(parameters["amount_salt"]),
            n_estimators=int(parameters["n_estimators"]),
            predictors=predictors,
            expected_regime=str(parameters["expected_regime"]),
            tape_preparation_seed=int(parameters["tape_preparation_seed"]),
        )
        if spec.tape_preparation_seed != UK_PROPERTY_COMPONENTS_TAPE_SEED:
            raise PropertyComponentsError(
                "tape_preparation_seed must be the income stage's stage-1 seed "
                f"{UK_PROPERTY_COMPONENTS_TAPE_SEED}."
            )
        if spec.incidence_salt == spec.amount_salt:
            raise PropertyComponentsError("the two draws need distinct salts.")
        uprating = dict(parameters["income_uprating_variables"])
        expected = {
            column: SPI_INCOME_UPRATING_VARIABLES[column]
            for column in (PROFIT_COLUMN, FINANCE_COSTS_COLUMN)
        }
        if uprating != expected or int(parameters["donor_income_period"]) != (
            SPI_DONOR_INCOME_YEAR
        ):
            raise PropertyComponentsError(
                "the tape is rebased as the income stage rebases it: "
                f"income_uprating_variables {expected} from {SPI_DONOR_INCOME_YEAR}."
            )
        return spec


@dataclass(frozen=True)
class ReceiptsAllocationSpec:
    """The ``allocate_property_rental_income`` operation."""

    band_lower_bounds: tuple[int, ...]
    walk_seed: int
    maximum_top_band_multiplier: float

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> ReceiptsAllocationSpec:
        bounds = tuple(int(v) for v in parameters["band_lower_bounds"])
        if bounds != HMRC_PROPERTY_RENTAL_RECEIPTS_BAND_LOWER_BOUNDS:
            raise PropertyComponentsError(
                "band_lower_bounds must be PRIS Table 13's bands "
                f"{HMRC_PROPERTY_RENTAL_RECEIPTS_BAND_LOWER_BOUNDS}."
            )
        if parameters.get("output") != RECEIPTS_COLUMN:
            raise PropertyComponentsError(
                f"allocate_property_rental_income must write {RECEIPTS_COLUMN!r}."
            )
        if parameters.get("resource") != HMRC_PROPERTY_RENTAL_RESOURCE:
            raise PropertyComponentsError(
                f"the allocation reads {HMRC_PROPERTY_RENTAL_RESOURCE}."
            )
        multiplier = float(parameters["maximum_top_band_multiplier"])
        if not np.isfinite(multiplier) or multiplier <= 1.0:
            raise PropertyComponentsError(
                "maximum_top_band_multiplier must be finite and above 1."
            )
        return cls(
            band_lower_bounds=bounds,
            walk_seed=int(parameters["walk_seed"]),
            maximum_top_band_multiplier=multiplier,
        )


# --------------------------------------------------------------------------
# The FRS concept and the tape donor
# --------------------------------------------------------------------------


def _person_weights(frame: Frame) -> np.ndarray:
    household = frame.table("household")
    weights = pd.Series(
        np.asarray(frame.weights_for("household").values, dtype=float),
        index=household["household_id"].to_numpy(),
    )
    person = frame.table("person")
    return weights.reindex(person["person_household_id"].to_numpy()).to_numpy(
        dtype=float
    )


def _tape_concept_rows(person: pd.DataFrame) -> np.ndarray:
    """Rows whose income leaves the income stage drew from the SPI tape.

    The SPI support channel's claimants and partners aged 16 and over (the
    income stage's recipient rule); every other row keeps FRS-reported
    property income, including the channel's dependants on their twin values.
    """

    channel = person[support_channel_column("person")].astype(str).to_numpy()
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=float)
    role = person[SPI_RECIPIENT_ROLE_COLUMN].to_numpy(dtype=bool)
    return (
        (channel == SPI_SYNTHETIC_SUPPORT_CHANNEL)
        & (age >= SPI_MINIMUM_RECIPIENT_AGE)
        & role
    )


def frs_concept_rental_profit(
    person: pd.DataFrame, household: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Each person's sub-letting rent and their profit from other property.

    ``frs_property_income`` credits the household's SUBRENT (annual, floored
    at zero) to its reference person; the rest of an FRS row's
    ``property_income`` is ROYYR1. Returns ``(sublet, rental)`` for the FRS
    concept; the caller applies them to FRS-concept rows only.
    """

    profit = pd.to_numeric(person[PROFIT_COLUMN], errors="coerce").to_numpy(dtype=float)
    if SUBLET_COLUMN in household:
        household_sublet = pd.Series(
            pd.to_numeric(household[SUBLET_COLUMN], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=float),
            index=household["household_id"].to_numpy(),
        )
        sublet = household_sublet.reindex(
            person["person_household_id"].to_numpy()
        ).to_numpy(dtype=float)
        sublet = np.where(
            person[HOUSEHOLD_HEAD_COLUMN].to_numpy(dtype=bool), sublet, 0.0
        )
    else:
        sublet = np.zeros(len(person), dtype=float)
    sublet = np.minimum(sublet, np.maximum(profit, 0.0))
    rental = np.maximum(profit - sublet, 0.0)
    return sublet, rental


@dataclass(frozen=True)
class FinanceCostDonor:
    """The tape's analog landlords: profit after finance costs above zero."""

    analog: pd.DataFrame
    #: Interior edges of the incidence cells (profit after finance costs);
    #: a value at an edge belongs to the cell above it.
    edges: np.ndarray
    incidence: tuple[dict[str, float | None], ...]
    receipt: Mapping[str, Any]

    def cell(self, values: np.ndarray) -> np.ndarray:
        return np.searchsorted(
            self.edges, np.asarray(values, dtype=float), side="right"
        )


def _incidence_edges(values: np.ndarray, weights: np.ndarray, cells: int) -> np.ndarray:
    """Interior edges at the FACT-weighted ``k / cells`` quantiles, every cell
    non-empty: repeated quantiles merge, and an edge at the minimum (which
    would leave the lowest cell empty) is dropped."""

    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    edges = []
    for k in range(1, cells):
        value = float(
            values[order][min(np.searchsorted(cumulative, k / cells), len(values) - 1)]
        )
        if value > float(values.min()) and (not edges or value > edges[-1]):
            edges.append(value)
    return np.asarray(edges, dtype=float)


def spi_tape_finance_cost_donor(
    prepared: pd.DataFrame,
    *,
    uprating_factors: Mapping[str, float],
    spec: FinanceCostModelSpec,
) -> FinanceCostDonor:
    """The analog landlords of the prepared tape, at build-period prices.

    ``prepared`` is :func:`prepare_spi_donor_table`'s output (2022-23 money);
    ``uprating_factors`` are the income stage's factors for the same columns,
    so the tape's profit moves with ``obr.per_capita.gdp`` and its finance
    costs with ``obr.mortgage_interest``, as the engine uprates the inputs.
    """

    missing = sorted(
        {PROFIT_COLUMN, FINANCE_COSTS_COLUMN, "age", "FACT"} - set(prepared.columns)
    )
    if missing:
        raise PropertyComponentsError(f"the prepared tape lacks {missing}.")
    for column in (PROFIT_COLUMN, FINANCE_COSTS_COLUMN):
        factor = float(uprating_factors[column])
        if not np.isfinite(factor) or factor <= 0.0:
            raise PropertyComponentsError(f"{column} uprating factor must be positive.")
    profit = prepared[PROFIT_COLUMN].to_numpy(dtype=float) * float(
        uprating_factors[PROFIT_COLUMN]
    )
    costs = prepared[FINANCE_COSTS_COLUMN].to_numpy(dtype=float) * float(
        uprating_factors[FINANCE_COSTS_COLUMN]
    )
    weight = prepared["FACT"].to_numpy(dtype=float)
    landlord = profit > 0.0
    after = profit - costs
    analog_mask = landlord & (after > 0.0)
    if not analog_mask.any():
        raise PropertyComponentsError(
            "the tape has no landlord with profit after costs."
        )
    analog = pd.DataFrame(
        {
            PROFIT_AFTER_FINANCE_COSTS: after[analog_mask],
            "age": prepared["age"].to_numpy(dtype=float)[analog_mask],
            FINANCE_COSTS_COLUMN: costs[analog_mask],
            "weight": weight[analog_mask],
        }
    ).reset_index(drop=True)
    edges = _incidence_edges(
        analog[PROFIT_AFTER_FINANCE_COSTS].to_numpy(dtype=float),
        analog["weight"].to_numpy(dtype=float),
        spec.incidence_cells,
    )
    cell = np.searchsorted(edges, analog[PROFIT_AFTER_FINANCE_COSTS], side="right")
    has = analog[FINANCE_COSTS_COLUMN].to_numpy() > 0.0
    bounds = (0.0, *edges.tolist())
    incidence = []
    for index, lower in enumerate(bounds):
        in_cell = cell == index
        mass = float(analog["weight"].to_numpy()[in_cell].sum())
        if mass <= 0.0:
            raise PropertyComponentsError(
                "an incidence cell has no analog tape landlords."
            )
        incidence.append(
            {
                "lower_bound": lower,
                "upper_bound": bounds[index + 1] if index + 1 < len(bounds) else None,
                "tape_records": int(in_cell.sum()),
                "tape_weight": mass,
                "share_with_finance_costs": (
                    float(analog["weight"].to_numpy()[in_cell & has].sum()) / mass
                ),
            }
        )
    receipt = {
        "tape_landlords": int(landlord.sum()),
        "tape_landlord_weight": float(weight[landlord].sum()),
        "tape_landlords_with_finance_costs_weight": float(
            weight[landlord & (costs > 0.0)].sum()
        ),
        "tape_profit_gbp": float((profit * weight)[landlord].sum()),
        "tape_finance_costs_gbp": float((costs * weight)[landlord].sum()),
        "analog_landlords": int(analog_mask.sum()),
        "analog_weight": float(weight[analog_mask].sum()),
        "analog_rule": (
            "tape landlords (property_income above zero) whose profit after "
            "restricted finance costs is above zero: the analog of an FRS "
            "landlord reporting a profit after show card K6"
        ),
        "uprating_factors": {
            PROFIT_COLUMN: float(uprating_factors[PROFIT_COLUMN]),
            FINANCE_COSTS_COLUMN: float(uprating_factors[FINANCE_COSTS_COLUMN]),
        },
        "incidence_cells": spec.incidence_cells,
        "incidence_by_profit_after_finance_costs": incidence,
    }
    return FinanceCostDonor(
        analog=analog, edges=edges, incidence=tuple(incidence), receipt=receipt
    )


@dataclass(frozen=True)
class FinanceCostImputation:
    costs: np.ndarray
    receipt: Mapping[str, Any]
    fit_weight_records: tuple[FitWeightRecord, ...]


def impute_frs_property_finance_costs(
    donor: FinanceCostDonor,
    *,
    person_ids: np.ndarray,
    profit: np.ndarray,
    age: np.ndarray,
    weights: np.ndarray,
    spec: FinanceCostModelSpec,
) -> FinanceCostImputation:
    """Draw finance costs for FRS-concept landlords (``profit`` above zero).

    Incidence: a person-keyed uniform below the tape share of the landlord's
    cell of profit after finance costs. Amount: the QRF fitted on the analog
    landlords with finance costs, drawn at a second person-keyed quantile.
    Rows with no profit get no draw and no costs.
    """

    from microcosm.fit.qrf import RegimeGatedQRF

    profit = np.asarray(profit, dtype=float)
    landlord = profit > 0.0
    costs = np.zeros(len(profit), dtype=float)
    shares = np.asarray(
        [cell["share_with_finance_costs"] for cell in donor.incidence], dtype=float
    )
    band = donor.cell(profit)
    probability = np.where(landlord, shares[band], 0.0)
    uniforms = stable_identity_uniforms(
        person_ids, seed=spec.incidence_seed, salt=spec.incidence_salt
    )
    has = landlord & (uniforms < probability)

    train = donor.analog.loc[
        donor.analog[FINANCE_COSTS_COLUMN] > 0.0,
        [*spec.predictors, FINANCE_COSTS_COLUMN, "weight"],
    ].reset_index(drop=True)
    if train.empty:
        raise PropertyComponentsError("no analog tape landlord has finance costs.")
    fitted = RegimeGatedQRF(n_estimators=spec.n_estimators, seed=spec.amount_seed).fit(
        train.astype(float),
        list(spec.predictors),
        [FINANCE_COSTS_COLUMN],
        weights="weight",
    )
    regime = str(fitted.regimes().get(FINANCE_COSTS_COLUMN, ""))
    if regime != spec.expected_regime:
        raise PropertyComponentsError(
            f"the finance-cost model fitted regime {regime!r}, not the declared "
            f"{spec.expected_regime!r}."
        )
    if has.any():
        recipients = pd.DataFrame(
            {PROFIT_AFTER_FINANCE_COSTS: profit[has], "age": np.asarray(age)[has]}
        )[list(spec.predictors)].astype(float)
        quantiles = stable_identity_uniforms(
            np.asarray(person_ids)[has], seed=spec.amount_seed, salt=spec.amount_salt
        )
        costs[has] = fitted.predict_positive_from_uniforms(
            recipients, quantiles={FINANCE_COSTS_COLUMN: quantiles}
        )[FINANCE_COSTS_COLUMN].to_numpy(dtype=float)
    weights = np.asarray(weights, dtype=float)
    realised = []
    for index, cell in enumerate(donor.incidence):
        in_band = landlord & (band == index)
        mass = float(weights[in_band].sum())
        realised.append(
            {
                "lower_bound": cell["lower_bound"],
                "landlords": int(in_band.sum()),
                "landlord_weight": mass,
                "tape_share": cell["share_with_finance_costs"],
                "realised_share": (
                    float(weights[in_band & has].sum()) / mass if mass > 0.0 else None
                ),
            }
        )
    receipt = {
        "model": "regime_gated_qrf",
        "regime": regime,
        "predictors": list(spec.predictors),
        "n_estimators": spec.n_estimators,
        "training_rows": int(len(train)),
        "incidence": {"seed": spec.incidence_seed, "salt": spec.incidence_salt},
        "amount": {
            "seed": spec.amount_seed,
            "salt": spec.amount_salt,
            "draw": "predict_positive_from_uniforms at the person-keyed quantile",
        },
        "recipients": int(landlord.sum()),
        "recipient_weight": float(weights[landlord].sum()),
        "drawn": int(has.sum()),
        "drawn_weight": float(weights[has].sum()),
        "finance_costs_gbp": float((costs * weights).sum()),
        "incidence_by_profit_after_finance_costs": realised,
    }
    records = (
        FitWeightRecord(
            f"{UK_PROPERTY_COMPONENTS_FIT_NAME}:{FINANCE_COSTS_COLUMN}",
            fitted.weight_kind,
        ),
    )
    return FinanceCostImputation(
        costs=costs, receipt=receipt, fit_weight_records=records
    )


# --------------------------------------------------------------------------
# Receipts
# --------------------------------------------------------------------------


def _walk_bands(
    weights: np.ndarray, masses: Sequence[float], offset: float
) -> np.ndarray:
    """Band index of each ranked person, highest band first.

    The CGT Table 3 walk's rule (``cgt_imputation._walk_bands``): a person
    joins the band whose cumulative boundary first covers ``(1 - offset)`` of
    their weight, with one systematic offset per walk, so no person splits
    across bands and a band smaller than one person's weight still receives
    that person on the share of walks its boundary implies. ``masses`` run
    from the top band down; the last entry takes everyone left.
    """

    cumulative = np.cumsum(weights)
    position = cumulative - (1.0 - offset) * weights
    boundaries = np.cumsum(np.asarray(masses, dtype=float))
    band = np.searchsorted(boundaries, position, side="left")
    return np.minimum(band, len(masses) - 1)


@dataclass(frozen=True)
class ReceiptsAllocation:
    receipts: np.ndarray
    receipt: Mapping[str, Any]


def allocate_property_rental_income(
    *,
    person_ids: np.ndarray,
    profit: np.ndarray,
    weights: np.ndarray,
    facts: HMRCPropertyRentalFacts,
    spec: ReceiptsAllocationSpec,
) -> ReceiptsAllocation:
    """Receipts for every person with ``profit`` above zero (zero elsewhere).

    Ranked by profit (person id breaks ties), walked top band first into
    Table 13's bands at their individuals-basis counts, linear within a
    bounded band from its upper bound down, never below profit; the open top
    band takes ``max(lower bound, k x profit)`` with ``k`` solved to the
    individuals' receipts total.
    """

    profit = np.asarray(profit, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ids = np.asarray(person_ids)
    landlord = np.flatnonzero(profit > 0.0)
    receipts = np.zeros(len(profit), dtype=float)
    bands = facts.individuals_receipts_bands()
    top_down = list(reversed(bands))
    if not len(landlord):
        raise PropertyComponentsError("no landlord with profit above zero to allocate.")
    order = landlord[np.lexsort((ids[landlord], -profit[landlord]))]
    ranked_weights = weights[order]
    offset = float(np.random.default_rng(spec.walk_seed).random())
    band_index = _walk_bands(
        ranked_weights,
        [band.landlords for band in top_down[:-1]] + [np.inf],
        offset,
    )
    cells: list[dict[str, Any]] = []
    floored_weight = 0.0
    floored_rows = 0
    above_band_rows = 0
    bounded_receipts_at_published_counts = 0.0
    top_members = np.array([], dtype=int)
    for index, band in enumerate(top_down):
        members = order[band_index == index]
        cell: dict[str, Any] = {
            "lower_bound": band.lower_bound,
            "upper_bound": band.upper_bound,
            "target_landlords": band.landlords,
            "walk_landlords_weight": float(weights[members].sum()),
            "walk_records": int(len(members)),
        }
        if band.upper_bound is None:
            top_members = members
            cells.append(cell)
            continue
        if len(members):
            member_weights = weights[members]
            mass = float(member_weights.sum())
            from_top = (np.cumsum(member_weights) - 0.5 * member_weights) / mass
            linear = band.upper_bound - from_top * (band.upper_bound - band.lower_bound)
            values = np.maximum(linear, profit[members])
            floored = profit[members] > linear
            floored_rows += int(floored.sum())
            floored_weight += float(member_weights[floored].sum())
            above_band_rows += int((profit[members] > band.upper_bound).sum())
            receipts[members] = values
            mean = float(np.average(values, weights=member_weights))
        else:
            mean = 0.5 * (band.lower_bound + band.upper_bound)
        cell["mean_receipts_gbp"] = mean
        bounded_receipts_at_published_counts += mean * band.landlords
        cells.append(cell)

    top = top_down[0]
    target_total = float(facts.receipts["individual"])
    top_target_mean = (target_total - bounded_receipts_at_published_counts) / max(
        top.landlords, 1.0
    )
    multiplier = 1.0
    if len(top_members):
        top_weights = weights[top_members]
        top_profit = profit[top_members]

        def mean_at(k: float) -> float:
            return float(
                np.average(
                    np.maximum(top.lower_bound, k * top_profit), weights=top_weights
                )
            )

        low, high = 1.0, spec.maximum_top_band_multiplier
        if mean_at(low) >= top_target_mean:
            multiplier = low
        elif mean_at(high) <= top_target_mean:
            multiplier = high
        else:
            for _ in range(_BISECTION_STEPS):
                middle = 0.5 * (low + high)
                if mean_at(middle) < top_target_mean:
                    low = middle
                else:
                    high = middle
            multiplier = 0.5 * (low + high)
        receipts[top_members] = np.maximum(top.lower_bound, multiplier * top_profit)
        cells[0]["mean_receipts_gbp"] = mean_at(multiplier)
    cells[0]["target_mean_receipts_gbp"] = top_target_mean
    cells[0]["profit_multiplier"] = multiplier

    expenses = receipts - np.where(profit > 0.0, profit, 0.0)
    share = np.divide(
        expenses, receipts, out=np.zeros_like(receipts), where=receipts > 0.0
    )
    on = profit > 0.0
    receipt = {
        "population": "persons with profit from property other than sub-letting above zero",
        "order": "profit descending, person_id ascending",
        "walk": {"seed": spec.walk_seed, "offset": offset},
        "band_basis": (
            "PRIS 2024-25 Table 13 (all tax entities) x Table 1 individuals' "
            "share of landlords"
        ),
        "individuals_landlord_share": facts.individuals_share("landlords"),
        "bands_top_down": cells,
        "rows_floored_at_profit": floored_rows,
        "weight_floored_at_profit": floored_weight,
        "rows_with_profit_above_their_band": above_band_rows,
        "top_band_multiplier_bounds": [1.0, spec.maximum_top_band_multiplier],
        "landlords": int(on.sum()),
        "landlord_weight": float(weights[on].sum()),
        "largest_landlord_weight": float(weights[on].max()),
        "receipts_gbp": float((receipts * weights).sum()),
        "profit_gbp": float((np.where(on, profit, 0.0) * weights).sum()),
        "deductible_expenses_gbp": float((expenses * weights).sum()),
        "published": {
            "individuals_receipts_gbp": target_total,
            "individuals_deductible_expenses_gbp": facts.individuals_deductible_expenses(),
            "individuals_landlords": float(facts.landlords["individual"]),
        },
        "expense_share_quantiles": _weighted_quantiles(
            share[on], weights[on], (0.1, 0.25, 0.5, 0.75, 0.9)
        ),
    }
    return ReceiptsAllocation(receipts=receipts, receipt=receipt)


# --------------------------------------------------------------------------
# Stage transform
# --------------------------------------------------------------------------


@dataclass
class UKPropertyComponentsResult:
    frame: Frame
    facts: Mapping[str, Any]
    donor: Mapping[str, Any]
    finance_costs: Mapping[str, Any]
    receipts: Mapping[str, Any]
    channels: Mapping[str, Any]
    residuals: Mapping[str, Any]
    coherence: Mapping[str, Any]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_PROPERTY_COMPONENTS_STAGE_NAME,
            "facts": dict(self.facts),
            "donor": dict(self.donor),
            "finance_costs": dict(self.finance_costs),
            "receipts": dict(self.receipts),
            "channels": dict(self.channels),
            "residuals": dict(self.residuals),
            "coherence": dict(self.coherence),
        }


@dataclass
class UKPropertyComponentsStageTransform:
    """Whole-stage callable for landlords' receipts and finance costs."""

    stage: SourceStageSpec
    spi_tab_path: str | Path | None = None
    #: The raw tape as read (tests and the synthetic fixture).
    donor_table: pd.DataFrame | None = field(default=None, repr=False)
    #: The income stage's donor age model; read from the vendored ONS
    #: resource and the engine's State Pension age when not given.
    age_model: SPIDonorAgeModel | None = field(default=None, repr=False)
    #: The income stage's 2022-to-build-period factors for profit and finance
    #: costs; derived from the engine's parameters when not given.
    uprating_factors: Mapping[str, float] | None = None
    facts: HMRCPropertyRentalFacts | None = field(default=None, repr=False)
    last_result: UKPropertyComponentsResult | None = field(default=None, init=False)
    last_fit_weight_records: tuple[FitWeightRecord, ...] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        return tuple(self.last_fit_weight_records or ())

    def _raw_tape(self) -> pd.DataFrame:
        if self.donor_table is not None:
            return self.donor_table.copy(deep=True)
        if self.spi_tab_path is None:
            raise PropertyComponentsError(
                f"{UK_PROPERTY_COMPONENTS_STAGE_NAME} requires the caller-supplied "
                "SPI 2022-23 tape."
            )
        identity = verify_spi_donor_identity(self.spi_tab_path)
        return pd.read_csv(identity.path, delimiter="\t")

    def __call__(self, frame: Frame) -> Frame:
        validate_uk_national_frame(frame)
        finance_spec = FinanceCostModelSpec.from_parameters(
            _operation(self.stage, IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND)
        )
        receipts_spec = ReceiptsAllocationSpec.from_parameters(
            _operation(self.stage, ALLOCATE_PROPERTY_RENTAL_INCOME_KIND)
        )
        verify = _operation(self.stage, VERIFY_VENDORED_FACT_RESOURCE_KIND)
        if (
            verify.get("resource") != HMRC_PROPERTY_RENTAL_RESOURCE
            or tuple(verify.get("record_sets", ())) != HMRC_PROPERTY_RENTAL_RECORD_SETS
        ):
            raise PropertyComponentsError(
                f"verify_vendored_fact_resource must pin {HMRC_PROPERTY_RENTAL_RESOURCE} "
                f"and its record sets {HMRC_PROPERTY_RENTAL_RECORD_SETS}."
            )
        facts = self.facts or load_hmrc_property_rental_facts()
        period = int(uk_time_period(frame))

        person = frame.table("person").copy()
        household = frame.table("household")
        missing = sorted(
            {
                PROFIT_COLUMN,
                FINANCE_COSTS_COLUMN,
                HOUSEHOLD_HEAD_COLUMN,
                SPI_RECIPIENT_ROLE_COLUMN,
                support_channel_column("person"),
                "age",
            }
            - set(person.columns)
        )
        if missing:
            raise PropertyComponentsError(f"the spine's person table lacks {missing}.")
        weights = _person_weights(frame)
        person_ids = person["person_id"].to_numpy()
        tape_concept = _tape_concept_rows(person)
        profit_in = pd.to_numeric(person[PROFIT_COLUMN], errors="coerce").to_numpy(
            dtype=float
        )
        costs_in = pd.to_numeric(
            person[FINANCE_COSTS_COLUMN], errors="coerce"
        ).to_numpy(dtype=float)
        if not (np.isfinite(profit_in).all() and np.isfinite(costs_in).all()):
            raise PropertyComponentsError(
                "property income and finance costs must be finite."
            )
        if (profit_in < 0.0).any() or (costs_in < 0.0).any():
            raise PropertyComponentsError(
                "property income and finance costs must be non-negative."
            )
        if (costs_in[~tape_concept] != 0.0).any():
            raise PropertyComponentsError(
                "FRS-concept rows already carry finance costs; the income stage "
                "draws them on the SPI channel only."
            )
        sublet, rental = frs_concept_rental_profit(person, household)
        sublet = np.where(tape_concept, 0.0, sublet)
        rental = np.where(tape_concept, profit_in, rental)

        # Finance costs on the FRS concept, added back to profit.
        if self.uprating_factors is None:
            from microcosm.build.uk_runtime.spi_income import (
                _spi_income_uprating_factors,
            )

            factors, _ = _spi_income_uprating_factors(period)
        else:
            factors = dict(self.uprating_factors)
        prepared = prepare_spi_donor_table(
            self._raw_tape(),
            seed=finance_spec.tape_preparation_seed,
            age_model=(
                load_spi_donor_age_model(period)
                if self.age_model is None
                else self.age_model
            ),
        )
        donor = spi_tape_finance_cost_donor(
            prepared, uprating_factors=factors, spec=finance_spec
        )
        frs_rental = np.where(tape_concept, 0.0, rental)
        imputation = impute_frs_property_finance_costs(
            donor,
            person_ids=person_ids,
            profit=frs_rental,
            age=pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=float),
            weights=weights,
            spec=finance_spec,
        )
        added = np.where(tape_concept, 0.0, imputation.costs)
        costs_out = costs_in + added
        profit_out = profit_in + added
        rental_out = rental + added

        # Receipts.
        allocation = allocate_property_rental_income(
            person_ids=person_ids,
            profit=rental_out,
            weights=weights,
            facts=facts,
            spec=receipts_spec,
        )
        receipts = allocation.receipts + sublet
        if (receipts + 1e-6 < profit_out).any():
            raise PropertyComponentsError("receipts fell below profit on some row.")

        person[PROFIT_COLUMN] = profit_out
        person[FINANCE_COSTS_COLUMN] = costs_out
        # Declared output order (the graph's cell order).
        person[RECEIPTS_COLUMN] = receipts

        channel = person[support_channel_column("person")].astype(str).to_numpy()
        channels = {}
        for name in sorted(set(channel)):
            rows = channel == name
            on = rows & (profit_out > 0.0)
            channels[name] = {
                "landlords": int(on.sum()),
                "landlord_weight": float(weights[on].sum()),
                "profit_gbp": float((profit_out * weights)[rows].sum()),
                "finance_costs_gbp": float((costs_out * weights)[rows].sum()),
                "receipts_gbp": float((receipts * weights)[rows].sum()),
            }
        sublet_on = sublet > 0.0
        residuals = {
            "sublet_receipts": {
                "rows": int(sublet_on.sum()),
                "weight": float(weights[sublet_on].sum()),
                "gbp": float((sublet * weights).sum()),
                "note": (
                    "sub-letting rent stays in property_income with equal receipts "
                    "and no expenses until Rent a Room is released "
                    "(PolicyEngine/policyengine-uk#2002); the engine's receipts "
                    "concept excludes Rent a Room receipts"
                ),
            },
            "landlords_without_profit": (
                "FRS loss-makers (their loss counts as zero) and tape landlords "
                "at zero profit carry no receipts; PRIS counts them"
            ),
            "frs_mortgage_capital": (
                "show card K6 also nets mortgage capital, which stays netted from "
                "FRS profit"
            ),
            "finance_costs_brought_forward": (
                "property_finance_costs_brought_forward stays unset (single-year "
                "data): the engine's s. 274AA carry-forward starts from nothing"
            ),
        }
        coherence: dict[str, Any] = {}
        if all(column in household for column in _RENTAL_PROPERTY_COLUMNS):
            holds = (
                household[list(_RENTAL_PROPERTY_COLUMNS)]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .sum(axis=1)
                .to_numpy(dtype=float)
                > 0.0
            )
            by_household = pd.Series(holds, index=household["household_id"].to_numpy())
            person_holds = by_household.reindex(
                person["person_household_id"].to_numpy()
            ).to_numpy(dtype=bool)
            on = rental_out > 0.0
            coherence["landlord_weight_share_in_households_with_rental_property"] = (
                float(weights[on & person_holds].sum()) / float(weights[on].sum())
                if weights[on].sum() > 0.0
                else None
            )
        result = uk_national_frame(
            person=person,
            benunit=frame.table("benunit").copy(),
            household=household.copy(),
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=np.asarray(frame.weights_for("household").values),
            mass_log=(
                *frame.mass_log,
                uk_household_mass_conservation_receipt(
                    frame, UK_PROPERTY_COMPONENTS_MASS_CONSERVATION_REASON
                ),
            ),
        )
        validate_uk_national_frame(result)
        self.last_fit_weight_records = imputation.fit_weight_records
        self.last_result = UKPropertyComponentsResult(
            frame=result,
            facts=facts.evidence(),
            donor=donor.receipt,
            finance_costs={
                **imputation.receipt,
                "added_back_to": PROFIT_COLUMN,
                "rows_added_back": int((added > 0.0).sum()),
            },
            receipts={
                **allocation.receipt,
                "rows_receipts_below_profit": int((receipts + 1e-6 < profit_out).sum()),
                "sublet_receipts_gbp": float((sublet * weights).sum()),
                "total_receipts_gbp": float((receipts * weights).sum()),
            },
            channels=channels,
            residuals=residuals,
            coherence=coherence,
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_PROPERTY_COMPONENTS_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            return {}
        return {"evidence": self.last_result.evidence()}
