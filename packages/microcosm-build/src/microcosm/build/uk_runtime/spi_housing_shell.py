"""Impute the SPI support households' housing from their own incomes.

The SPI support channel and the income band donors are whole copies of FRS
households whose adults then receive SPI incomes. Without this stage every copy
keeps its FRS parent's housing: tenure, dwelling type, bedrooms, rent,
mortgage, council tax and the housing-linked benefit reports, all drawn for the
parent's income. This stage is the household counterpart of the SPI stage-2
FRS-only fill: it trains on the FRS base households and draws each SPI
household's housing from its conditional distribution given the household's
region, composition and incomes.

The categorical steps (tenure, dwelling type, bedrooms, council-tax band) use a
weighted multiclass classifier and an inverse-CDF draw; the amounts use one
chained regime-gated QRF conditioned on the drawn categories. Every draw reads
identity-keyed uniforms, so adding or reordering rows never moves a draw.
Support rules learned from the training rows then hold the bundle together: an
amount is zero wherever the FRS has no positive value for that tenure or
region, and Housing Benefit needs a rented tenure because the engine pays it to
renting reporters only. A council-tax band its nation does not use (band I
outside Wales, any band but the placeholder in Northern Ireland) is drawn again
from the classifier's probabilities over that nation's bands. FRS base rows are
never modified.

Council tax stays a target of the chained QRF on every row. After the support
rules, a Great Britain row's drawn bill is replaced by the 2024-25 statute at
its reference Band D (``spi_housing_council_tax``): the expected Band D of the
cell the SPI-copy area stage (``spi_copy_areas``), which runs after the
shell, draws the row's area from: its region by the households of its drawn
band where the region holds that band, else by households (María's ruling of
2026-11-25), for the row's nation and drawn band, less the
single-adult discount where ``council_tax_single_adult_raw`` is 1. A Northern
Ireland row keeps the chained draw (María's ruling of 2026-11-14), and every
band and every other chained amount is the draw's. The drawn council tax
benefit is then capped at the bill written, and the rebate set equal to it.
The bill reads the row's region and no area column, so the council-tax
localisation (``council_tax_localise``) localises it once. The stage refuses
unless the bill's checks hold on what it writes, and its receipt carries the
bill's method, inputs and checks under ``council_tax``, with no number.

The stage declares the Scottish Band D resource's ``verify_vendored_fact_resource``
operation, the imputation and a ``derive`` of ``council_tax``, and the two Band D
artifacts. The transform takes the conditioned supports the area stages read,
and runs the Scottish resource's runtime arm with the frame's period before it
reads a row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from microcosm.build.atomic_geography import AtomicSupport
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    uk_conditioned_assignment_definition,
    uk_conditioned_supports,
)
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.council_tax_band_d import (
    UKCouncilTaxBandD,
    check_scotland_band_d_declaration,
    load_uk_council_tax_band_d,
)
from microcosm.build.uk_runtime.council_tax_constants import (
    UK_COUNCIL_TAX_BANDS_BY_NATION,
    UK_COUNCIL_TAX_BANDS_BY_REGION,
    UK_NATION_BY_REGION,
)
from microcosm.build.uk_runtime.household_lineage import (
    GEOGRAPHIC_FOLD_EVIDENCE,
    fold_geographic_clones,
    geographic_fold_receipt,
)
from microcosm.build.uk_runtime.qrf_gates import (
    UK_DEFAULT_GATE_CALIBRATION,
    UK_GATE_OUTCOMES_EVIDENCE,
    declared_gate_calibration,
    gate_outcomes,
)
from microcosm.build.uk_runtime.spi_housing_council_tax import (
    UK_SPI_HOUSING_COUNCIL_TAX_ARTIFACTS,
    UK_SPI_HOUSING_COUNCIL_TAX_CHECKS,
    UK_SPI_HOUSING_COUNCIL_TAX_DERIVE_KIND,
    UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE,
    UK_SPI_HOUSING_COUNCIL_TAX_VERIFY_KIND,
    UKSPIHousingCouncilTaxInputs,
    uk_spi_council_tax_bills,
    uk_spi_housing_council_tax_checks,
    uk_spi_housing_council_tax_derive_parameters,
    uk_spi_housing_council_tax_evidence,
    uk_spi_housing_council_tax_inputs,
    uk_spi_housing_council_tax_verify_parameters,
)

__all__ = [
    "HOUSING_AMOUNT_COLUMNS",
    "HOUSING_CATEGORICAL_COLUMNS",
    "HOUSING_PERSON_COLUMNS",
    "UK_SPI_HOUSING_SHELL_STAGE_NAME",
    "FittedHousingModel",
    "fit_housing_model",
    "household_housing_predictors",
    "household_housing_targets",
    "class_scores",
    "impute_spi_housing_shell",
    "restrict_bands_to_nation",
]

UK_SPI_HOUSING_SHELL_STAGE_NAME = "spi_housing_shell"
FRS_CHANNEL = "frs"
SPI_CHANNEL = "spi"
RENTED_TENURES = ("RENT_PRIVATELY", "RENT_FROM_COUNCIL", "RENT_FROM_HA")

#: Drawn in this order, each conditioning on the predictors and every earlier
#: category.
HOUSING_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "tenure_type",
    "accommodation_type",
    "num_bedrooms",
    "council_tax_band",
)
#: Household amounts, one chained regime-gated QRF in this order.
HOUSING_AMOUNT_COLUMNS: tuple[str, ...] = (
    "council_tax",
    "council_tax_reported",
    "rent",
    "mortgage_interest_repayment",
    "mortgage_capital_repayment",
    "structural_insurance_payments",
    "housing_service_charges",
    "water_and_sewerage_charges",
    "domestic_rates",
    "subrent",
)
#: Housing-linked benefit reports, drawn at household level after the amounts
#: and placed on the household reference person.
HOUSING_BENEFIT_TARGETS: dict[str, str] = {
    "housing_benefit_head_unit": "housing_benefit_reported",
    "council_tax_benefit_hrp": "council_tax_benefit_reported",
}
HOUSING_PERSON_COLUMNS: tuple[str, ...] = tuple(HOUSING_BENEFIT_TARGETS.values())
#: Written from the drawn HRP council tax benefit, as the FRS does (both are
#: CTREBAMT on the household reference person).
COUNCIL_TAX_REBATE_COLUMN = "council_tax_rebate"

CATEGORICAL_PREDICTORS: tuple[str, ...] = ("region", "ons_household_type")
NUMERIC_PREDICTORS: tuple[str, ...] = (
    "num_adults",
    "num_children",
    "single_adult",
    "hrp_age",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "investment_income",
    "state_pension_reported",
)
#: Interest is ISA interest included, as gross interest held it before the
#: additive contract (microcosm#1360).
_INVESTMENT_COLUMNS = (
    "savings_interest_income",
    "individual_savings_account_interest_income",
    "dividend_income",
    "property_income",
    "other_investment_income",
)
_ADULT_AGE = 18


def household_housing_predictors(
    person: pd.DataFrame, household: pd.DataFrame
) -> pd.DataFrame:
    """One predictor row per household, indexed by ``household_id``."""

    required = {"person_household_id", "age", "is_household_head"}
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(f"Housing predictors need person columns {missing}.")
    people = person.copy()
    age = pd.to_numeric(people["age"], errors="raise")
    people["_adult"] = (age >= _ADULT_AGE).astype(float)
    people["_child"] = (age < _ADULT_AGE).astype(float)
    people["_hrp_age"] = np.where(people["is_household_head"].astype(bool), age, np.nan)
    people["investment_income"] = sum(
        pd.to_numeric(people[column], errors="raise") if column in people else 0.0
        for column in _INVESTMENT_COLUMNS
    )
    sums = {
        column: pd.to_numeric(people[column], errors="raise")
        for column in (
            "employment_income",
            "self_employment_income",
            "private_pension_income",
            "state_pension_reported",
        )
    }
    frame = pd.DataFrame(
        {
            "person_household_id": people["person_household_id"],
            "num_adults": people["_adult"],
            "num_children": people["_child"],
            "hrp_age": people["_hrp_age"],
            "investment_income": people["investment_income"],
            **sums,
        }
    )
    grouped = frame.groupby("person_household_id")
    out = grouped[
        [
            "num_adults",
            "num_children",
            "investment_income",
            *sums,
        ]
    ].sum()
    out["hrp_age"] = grouped["hrp_age"].max()
    household_indexed = household.set_index("household_id")
    out = out.reindex(household_indexed.index)
    if out.isna().any().any():
        raise ValueError(
            "Every household needs people and exactly one reference person."
        )
    out["single_adult"] = (
        pd.to_numeric(household_indexed["council_tax_single_adult_raw"]) == 1
    ).astype(float)
    for column in CATEGORICAL_PREDICTORS:
        out[column] = household_indexed[column].astype(str)
    return out[[*CATEGORICAL_PREDICTORS, *NUMERIC_PREDICTORS]]


def household_housing_targets(
    person: pd.DataFrame, household: pd.DataFrame
) -> pd.DataFrame:
    """The housing bundle per household, benefits aggregated as the FRS holds them."""

    household_indexed = household.set_index("household_id")
    out = household_indexed[
        [*HOUSING_CATEGORICAL_COLUMNS, *HOUSING_AMOUNT_COLUMNS]
    ].copy()
    heads = person.loc[person["is_household_head"].astype(bool)]
    head_benunit = heads.set_index("person_household_id")["person_benunit_id"]
    in_head_unit = (
        person["person_benunit_id"].to_numpy()
        == person["person_household_id"].map(head_benunit).to_numpy()
    )
    hb = (
        pd.to_numeric(person["housing_benefit_reported"], errors="raise")
        .where(in_head_unit, 0.0)
        .groupby(person["person_household_id"])
        .sum()
    )
    ctb = (
        pd.to_numeric(heads["council_tax_benefit_reported"], errors="raise")
        .groupby(heads["person_household_id"])
        .sum()
    )
    out["housing_benefit_head_unit"] = hb.reindex(out.index).fillna(0.0)
    out["council_tax_benefit_hrp"] = ctb.reindex(out.index).fillna(0.0)
    return out


def _one_hot(
    frame: pd.DataFrame, columns: Sequence[str], levels: Mapping
) -> pd.DataFrame:
    parts = [frame.drop(columns=list(columns))]
    for column in columns:
        values = frame[column].astype(str).to_numpy()
        for level in levels[column]:
            parts.append(
                pd.Series(
                    (values == level).astype(float),
                    index=frame.index,
                    name=f"{column}={level}",
                )
            )
    return pd.concat(parts, axis=1)


def class_scores(classifier: Any, features: np.ndarray) -> np.ndarray:
    """The classifier's raw class scores, one column per class.

    A softmax of these rows is ``predict_proba``; a binary classifier scores
    ``classes_[1]`` against ``classes_[0]``, so its column for class 0 is zero.
    """

    scores = np.asarray(classifier.decision_function(features), dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([np.zeros_like(scores), scores])
    return scores


def restrict_bands_to_nation(
    choice: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    regions: np.ndarray,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Redraw each band its household's nation does not use.

    A drawn band outside the nation's bands is replaced by an inverse-CDF draw
    from the classifier conditioned on those bands: a softmax of the raw class
    scores over the nation's bands, with its own identity-keyed uniform. On the
    scores, the conditional survives a softmax that rounded the nation's bands
    to probability zero, as it does for a few English rows on licensed data. A
    row keeps its first draw whenever that draw is valid, so the redrawn rows
    end up distributed as the classifier conditioned on the nation's bands, and
    no valid row moves.

    Returns the corrected class indices and the rows redrawn, by nation.
    """

    unknown = sorted(set(regions) - set(UK_COUNCIL_TAX_BANDS_BY_REGION))
    if unknown:
        raise ValueError(f"Council-tax bands need a nation for regions {unknown}.")
    allowed = np.zeros(scores.shape, dtype=bool)
    for region, bands in UK_COUNCIL_TAX_BANDS_BY_REGION.items():
        rows = regions == region
        if rows.any():
            allowed[np.ix_(rows, np.isin(classes, bands))] = True
    outside = ~allowed[np.arange(len(choice)), choice]
    # Exact row counts for the stage's own evidence, which no gate reads: band
    # them before any gate detail or report carries them (microcosm#1278).
    redrawn = {
        f"council_tax_band:redrawn:{nation}": int(
            (outside & (pd.Series(regions).map(UK_NATION_BY_REGION) == nation)).sum()
        )
        for nation in UK_COUNCIL_TAX_BANDS_BY_NATION
    }
    if not outside.any():
        return choice, redrawn
    logits = np.where(allowed[outside], scores[outside], -np.inf)
    top = logits.max(axis=1)
    if not np.isfinite(top).all():
        raise ValueError(
            "A household's nation has no council-tax band the classifier knows."
        )
    cumulative = np.cumsum(np.exp(logits - top[:, None]), axis=1)
    corrected = choice.copy()
    corrected[outside] = (
        cumulative > (uniforms[outside] * cumulative[:, -1])[:, None]
    ).argmax(axis=1)
    return corrected, redrawn


@dataclass
class FittedHousingModel:
    """The fitted categorical steps and the chained amount model."""

    levels: dict[str, tuple[str, ...]]
    classifiers: dict[str, HistGradientBoostingClassifier]
    amount_model: Any
    amount_targets: tuple[str, ...]
    structural_zeros: dict[str, dict[str, tuple[str, ...]]]
    seed: int
    support: dict[str, tuple[float, float]] = field(default_factory=dict)

    def _features(self, predictors: pd.DataFrame, drawn: pd.DataFrame) -> pd.DataFrame:
        frame = pd.concat([predictors, drawn], axis=1)
        categorical = [
            column
            for column in (*CATEGORICAL_PREDICTORS, *drawn.columns)
            if column in self.levels
        ]
        return _one_hot(frame, categorical, self.levels)

    def draw(
        self,
        predictors: pd.DataFrame,
        *,
        firings: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        """Draw the housing bundle for ``predictors`` (indexed by household id).

        The council-tax band is kept inside the household's nation's bands
        (:func:`restrict_bands_to_nation`); ``firings`` receives the rows
        redrawn, by nation.
        """

        ids = predictors.index.to_numpy()
        drawn = pd.DataFrame(index=predictors.index)
        for column in HOUSING_CATEGORICAL_COLUMNS:
            features = self._features(predictors, drawn)
            classifier = self.classifiers[column]
            matrix = features.to_numpy(dtype=float)
            probabilities = classifier.predict_proba(matrix)
            uniforms = stable_identity_uniforms(
                ids, seed=self.seed, salt=f"{UK_SPI_HOUSING_SHELL_STAGE_NAME}:{column}"
            )
            cumulative = np.cumsum(probabilities, axis=1)
            cumulative[:, -1] = 1.0
            choice = (cumulative > uniforms[:, None]).argmax(axis=1)
            classes = np.asarray(classifier.classes_).astype(str)
            if column == "council_tax_band":
                choice, redrawn = restrict_bands_to_nation(
                    choice,
                    class_scores(classifier, matrix),
                    classes,
                    predictors["region"].astype(str).to_numpy(),
                    stable_identity_uniforms(
                        ids,
                        seed=self.seed,
                        salt=f"{UK_SPI_HOUSING_SHELL_STAGE_NAME}:{column}:nation",
                    ),
                )
                if firings is not None:
                    firings.update(redrawn)
            drawn[column] = classes[choice]
        features = self._features(predictors, drawn)
        quantiles = {
            target: stable_identity_uniforms(
                ids,
                seed=self.seed,
                salt=f"{UK_SPI_HOUSING_SHELL_STAGE_NAME}:{target}:quantile",
            )
            for target in self.amount_targets
        }
        signs = {
            target: stable_identity_uniforms(
                ids,
                seed=self.seed,
                salt=f"{UK_SPI_HOUSING_SHELL_STAGE_NAME}:{target}:sign",
            )
            for target in self.amount_targets
        }
        amounts = self.amount_model.predict_from_uniforms(
            features, quantiles=quantiles, sign_uniforms=signs
        )
        for target in self.amount_targets:
            low, high = self.support[target]
            drawn[target] = np.clip(amounts[target].to_numpy(dtype=float), low, high)
        return drawn


def _structural_zeros(
    predictors: pd.DataFrame, targets: pd.DataFrame, amount_targets: Sequence[str]
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Per amount target, the tenures and regions where training has no positive value."""

    zeros: dict[str, dict[str, tuple[str, ...]]] = {}
    for target in amount_targets:
        positive = targets[target].to_numpy(dtype=float) > 0.0
        rules: dict[str, tuple[str, ...]] = {}
        for key, labels in (
            ("tenure_type", targets["tenure_type"].astype(str)),
            ("region", predictors["region"].astype(str)),
        ):
            empty = sorted(
                level
                for level, has_positive in pd.Series(positive, index=labels.index)
                .groupby(labels.to_numpy())
                .any()
                .items()
                if not has_positive
            )
            if empty:
                rules[key] = tuple(empty)
        zeros[target] = rules
    return zeros


def fit_housing_model(
    predictors: pd.DataFrame,
    targets: pd.DataFrame,
    weights: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    gate_calibration: str = UK_DEFAULT_GATE_CALIBRATION,
) -> FittedHousingModel:
    """Fit the categorical steps and the chained amounts on training households.

    ``gate_calibration`` is the amounts' declared
    :class:`~microcosm.fit.GateCalibration`.
    """

    from microcosm.fit.qrf import RegimeGatedQRF

    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(predictors) or not np.isfinite(weights).all():
        raise ValueError("Housing training weights must be finite and row-aligned.")
    if (weights <= 0).any():
        raise ValueError("Housing training rows must carry positive weight.")
    levels: dict[str, tuple[str, ...]] = {
        column: tuple(sorted(predictors[column].astype(str).unique()))
        for column in CATEGORICAL_PREDICTORS
    }
    for column in HOUSING_CATEGORICAL_COLUMNS:
        levels[column] = tuple(sorted(targets[column].astype(str).unique()))
    amount_targets = (*HOUSING_AMOUNT_COLUMNS, *HOUSING_BENEFIT_TARGETS)
    model = FittedHousingModel(
        levels=levels,
        classifiers={},
        amount_model=None,
        amount_targets=amount_targets,
        structural_zeros=_structural_zeros(predictors, targets, amount_targets),
        seed=seed,
    )
    observed = pd.DataFrame(index=predictors.index)
    for column in HOUSING_CATEGORICAL_COLUMNS:
        features = model._features(predictors, observed)
        classifier = HistGradientBoostingClassifier(random_state=seed)
        classifier.fit(
            features.to_numpy(dtype=float),
            targets[column].astype(str).to_numpy(),
            sample_weight=weights,
        )
        model.classifiers[column] = classifier
        observed[column] = targets[column].astype(str)
    features = model._features(predictors, observed)
    table = pd.concat(
        [features, targets[list(amount_targets)].astype(float)], axis=1
    ).reset_index(drop=True)
    model.amount_model = RegimeGatedQRF(
        n_estimators=n_estimators, seed=seed, gate_calibration=gate_calibration
    ).fit(
        table,
        list(features.columns),
        list(amount_targets),
        weights=weights,
    )
    model.support = {
        target: (
            float(targets[target].astype(float).min()),
            float(targets[target].astype(float).max()),
        )
        for target in amount_targets
    }
    return model


def apply_support_rules(
    draws: pd.DataFrame,
    region: pd.Series,
    structural_zeros: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Force the learned structural zeros and the declared tenure rules.

    An amount is zero wherever training holds no positive value for the row's
    tenure or region, mortgage payments need a mortgaged tenure and Housing
    Benefit a rented one. Returns the ruled draws and, keyed as the receipt's
    ``rule_firings``, the rows each rule changed.
    """

    out = draws.copy()
    fired: dict[str, int] = {}
    labels = {
        "tenure_type": out["tenure_type"].astype(str),
        "region": region.reindex(out.index).astype(str),
    }
    for target, rules in structural_zeros.items():
        forced = np.zeros(len(out), dtype=bool)
        for key, levels in rules.items():
            forced |= labels[key].isin(levels).to_numpy()
        hit = forced & (out[target].to_numpy(dtype=float) != 0.0)
        fired[target] = int(hit.sum())
        out.loc[forced, target] = 0.0
    mortgaged = (out["tenure_type"].astype(str) == "OWNED_WITH_MORTGAGE").to_numpy()
    for target in ("mortgage_interest_repayment", "mortgage_capital_repayment"):
        hit = ~mortgaged & (out[target].to_numpy(dtype=float) != 0.0)
        fired[f"{target}:not_mortgaged"] = int(hit.sum())
        out.loc[~mortgaged, target] = 0.0
    rented = out["tenure_type"].astype(str).isin(RENTED_TENURES).to_numpy()
    hb_hit = ~rented & (out["housing_benefit_head_unit"].to_numpy(dtype=float) > 0.0)
    fired["housing_benefit_head_unit:not_rented"] = int(hb_hit.sum())
    out.loc[~rented, "housing_benefit_head_unit"] = 0.0
    return out, fired


def cap_council_tax_benefit(
    draws: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Cap the drawn council tax benefit at the council tax the draws carry.

    Returns the draws with ``council_tax_benefit_hrp`` at most ``council_tax``
    on every row and, keyed ``council_tax_benefit_hrp:cap`` as the receipt's
    ``rule_firings``, the rows whose benefit exceeded it.
    """

    out = draws.copy()
    ctb = out["council_tax_benefit_hrp"].to_numpy(dtype=float)
    cap = out["council_tax"].to_numpy(dtype=float)
    fired = {"council_tax_benefit_hrp:cap": int((ctb > cap).sum())}
    out["council_tax_benefit_hrp"] = np.minimum(ctb, cap)
    return out, fired


def apply_structural_rules(
    draws: pd.DataFrame,
    region: pd.Series,
    structural_zeros: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Force the learned structural zeros and the declared tenure rules, then
    cap the council tax benefit at the drawn council tax.

    :func:`apply_support_rules` then :func:`cap_council_tax_benefit`, their
    firings in that order. The housing shell runs the two apart, with each
    Great Britain row's statute bill written between them (microcosm#931).
    """

    supported, fired = apply_support_rules(draws, region, structural_zeros)
    capped, cap_fired = cap_council_tax_benefit(supported)
    return capped, {**fired, **cap_fired}


#: Each council-tax check of the SPI-channel rows, and the fixed text the
#: stage refuses with where it fails (microcosm#931).
_COUNCIL_TAX_CHECK_REFUSALS: Mapping[str, str] = MappingProxyType(
    {
        "bill_at_statute_of_reference": (
            "a Great Britain row's council_tax is not the 2024-25 statute at its "
            "reference Band D"
        ),
        "northern_ireland_bill_as_chained": (
            "a Northern Ireland row's council_tax is not its chained draw"
        ),
        "reduction_within_bill": (
            "a row's council_tax or council tax reduction is not finite and "
            "nonnegative, or the reduction exceeds the bill"
        ),
        "rebate_equals_reduction": (
            "a row's council_tax_rebate is not its reference person's council tax "
            "reduction"
        ),
    }
)


def _require_council_tax_inputs(
    inputs: object, recipients: pd.DataFrame
) -> UKSPIHousingCouncilTaxInputs:
    """The council-tax inputs of exactly the SPI-channel rows, with their
    regions, in table order: the bill's nation is the inputs' region's."""

    if not isinstance(inputs, UKSPIHousingCouncilTaxInputs):
        raise TypeError(
            "Housing shell prices council tax with council_tax_inputs built by "
            "uk_spi_housing_council_tax_inputs."
        )
    if list(inputs.household_ids) != recipients["household_id"].tolist():
        raise ValueError(
            "Housing shell prices council tax on its SPI-channel rows; "
            "council_tax_inputs must hold their household_id values in table "
            "order, and these do not."
        )
    if list(inputs.regions) != recipients["region"].astype(str).tolist():
        raise ValueError(
            "Housing shell prices council tax on its SPI-channel rows; "
            "council_tax_inputs must hold their region values in table order, "
            "and these do not."
        )
    return inputs


def _written_council_tax_checks(
    chained: pd.Series,
    *,
    inputs: UKSPIHousingCouncilTaxInputs,
    draws: pd.DataFrame,
    single_adult: pd.Series,
    household_out: pd.DataFrame,
    person_out: pd.DataFrame,
    recipients: np.ndarray,
    spi_people: np.ndarray,
) -> dict[str, bool]:
    """The bill's checks on what the stage wrote, each asserted.

    The bill and the rebate are read from the household rows written, and the
    reduction from the people written, summed per household (the reference
    person's), each on the inputs' household ids in their order.
    """

    index = pd.Index(list(inputs.household_ids))
    written = household_out.loc[recipients]
    reduction = (
        person_out.loc[spi_people]
        .groupby("person_household_id")["council_tax_benefit_reported"]
        .sum()
        .reindex(written["household_id"].to_numpy())
    )
    checks = uk_spi_housing_council_tax_checks(
        chained,
        inputs=inputs,
        bills=pd.Series(written["council_tax"].to_numpy(dtype=np.float64), index=index),
        bands=draws["council_tax_band"],
        single_adult=single_adult,
        reduction=pd.Series(reduction.to_numpy(dtype=np.float64), index=index),
        rebate=pd.Series(
            written[COUNCIL_TAX_REBATE_COLUMN].to_numpy(dtype=np.float64), index=index
        ),
    )
    for check in UK_SPI_HOUSING_COUNCIL_TAX_CHECKS:
        if checks[check] is not True:
            raise ValueError(
                f"Housing shell's council tax failed its check {check!r}: "
                f"{_COUNCIL_TAX_CHECK_REFUSALS[check]}."
            )
    return checks


def impute_spi_housing_shell(
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
    *,
    council_tax_inputs: UKSPIHousingCouncilTaxInputs,
    seed: int = 0,
    n_estimators: int = 100,
    gate_calibration: str = UK_DEFAULT_GATE_CALIBRATION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Redraw every SPI-channel household's housing; FRS rows are untouched.

    ``council_tax_inputs`` are the SPI-channel rows' council-tax inputs
    (:func:`~microcosm.build.uk_runtime.spi_housing_council_tax.uk_spi_housing_council_tax_inputs`
    on ``household``), required so that no path skips the statute. After the
    support rules, each Great Britain row's ``council_tax`` becomes the
    statute at its reference
    (:func:`~microcosm.build.uk_runtime.spi_housing_council_tax.uk_spi_council_tax_bills`),
    a Northern Ireland row's stays the chained draw, and the drawn council tax
    benefit is then capped at the bill written. The household's bill, its
    ``council_tax_rebate`` and the reference person's reduction carry them;
    the bill's checks
    (:func:`~microcosm.build.uk_runtime.spi_housing_council_tax.uk_spi_housing_council_tax_checks`)
    must hold on what is written, and the receipt carries the bill's evidence
    under ``council_tax``.

    When ``gate_calibration`` is not the in-sample gate, the receipt names each
    amount target's gate outcome.

    Refused, naming no household: ``council_tax_inputs`` not built by
    ``uk_spi_housing_council_tax_inputs``, or not holding the SPI-channel
    rows' ``household_id`` and ``region`` values in table order; and a check
    that fails, with its fixed message.
    """

    channel = household["household_support_channel"].astype(str).to_numpy()
    if not set(channel) <= {FRS_CHANNEL, SPI_CHANNEL}:
        raise ValueError("Housing shell expects only frs and spi support channels.")
    weights = pd.Series(
        np.asarray(household_weights, dtype=float), index=household["household_id"]
    )
    predictors = household_housing_predictors(person, household)
    targets = household_housing_targets(person, household)
    # The model trains on each survey household once: its geographic clones
    # that agree on the predictors and targets fold into its lowest-id clone
    # with their weight, clones that differ stay rows of their own, and every
    # other copy stays its own row; a frame without geographic clones keeps
    # every row and weight (household_lineage.fold_geographic_clones,
    # microcosm#931). On more than one clone the receipt records the fold.
    fold = fold_geographic_clones(
        {"household": household},
        "household",
        (channel == FRS_CHANNEL) & (weights.to_numpy() > 0.0),
        weights.to_numpy(),
        inputs=pd.concat(
            [predictors.add_prefix("predictor:"), targets.add_prefix("target:")],
            axis=1,
        )
        .reindex(household["household_id"].to_numpy())
        .reset_index(drop=True),
    )
    geographic_fold = geographic_fold_receipt(
        {"household": household},
        "household",
        fold,
        stage=UK_SPI_HOUSING_SHELL_STAGE_NAME,
    )
    training = fold.kept
    fit_weights = pd.Series(fold.weights, index=household["household_id"])
    training_ids = household.loc[training, "household_id"]
    recipient_ids = household.loc[channel == SPI_CHANNEL, "household_id"]
    if training_ids.empty or recipient_ids.empty:
        raise ValueError("Housing shell needs FRS training rows and SPI recipients.")
    _require_council_tax_inputs(
        council_tax_inputs,
        household.loc[channel == SPI_CHANNEL, ["household_id", "region"]],
    )

    model = fit_housing_model(
        predictors.loc[training_ids],
        targets.loc[training_ids],
        fit_weights.loc[training_ids].to_numpy(),
        seed=seed,
        n_estimators=n_estimators,
        gate_calibration=gate_calibration,
    )
    band_redraws: dict[str, int] = {}
    raw = model.draw(predictors.loc[recipient_ids], firings=band_redraws)
    supported, fired = apply_support_rules(
        raw, predictors["region"], model.structural_zeros
    )
    # Each Great Britain row's bill by the statute at its reference Band D,
    # after the learned zeros and the tenure rules; a Northern Ireland row
    # keeps the chained draw (María's ruling of 2026-11-14). The cap then
    # reads the bill written (microcosm#931).
    chained = supported["council_tax"].copy()
    single_adult = predictors.loc[recipient_ids, "single_adult"]
    supported["council_tax"] = uk_spi_council_tax_bills(
        chained,
        inputs=council_tax_inputs,
        bands=supported["council_tax_band"],
        single_adult=single_adult,
    )
    draws, cap_fired = cap_council_tax_benefit(supported)
    fired.update(cap_fired)
    fired.update(band_redraws)

    household_out = household.copy()
    recipients = household_out["household_id"].isin(recipient_ids).to_numpy()
    aligned = draws.reindex(household_out.loc[recipients, "household_id"].to_numpy())
    for column in (*HOUSING_CATEGORICAL_COLUMNS, *HOUSING_AMOUNT_COLUMNS):
        values = aligned[column].to_numpy()
        if column == "num_bedrooms":
            values = values.astype(np.int64)
        household_out.loc[recipients, column] = values
    household_out.loc[recipients, COUNCIL_TAX_REBATE_COLUMN] = aligned[
        "council_tax_benefit_hrp"
    ].to_numpy(dtype=float)

    person_out = person.copy()
    spi_people = person_out["person_household_id"].isin(recipient_ids).to_numpy()
    hrp = spi_people & person_out["is_household_head"].astype(bool).to_numpy()
    for target, column in HOUSING_BENEFIT_TARGETS.items():
        person_out.loc[spi_people, column] = 0.0
        person_out.loc[hrp, column] = (
            person_out.loc[hrp, "person_household_id"]
            .map(draws[target])
            .to_numpy(dtype=float)
        )
    checks = _written_council_tax_checks(
        chained,
        inputs=council_tax_inputs,
        draws=draws,
        single_adult=single_adult,
        household_out=household_out,
        person_out=person_out,
        recipients=recipients,
        spi_people=spi_people,
    )

    receipt = {
        "training_households": int(len(training_ids)),
        "recipient_households": int(len(recipient_ids)),
        "structural_zero_rules": {
            target: {key: list(levels) for key, levels in rules.items()}
            for target, rules in model.structural_zeros.items()
        },
        "rule_firings": fired,
        "category_shares": {
            column: {
                "training": _weighted_shares(
                    targets.loc[training_ids, column], fit_weights.loc[training_ids]
                ),
                "recipients": _weighted_shares(
                    draws[column], weights.loc[recipient_ids]
                ),
            }
            for column in HOUSING_CATEGORICAL_COLUMNS
        },
        UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE: uk_spi_housing_council_tax_evidence(
            council_tax_inputs, checks=checks
        ),
    }
    if gate_calibration != UK_DEFAULT_GATE_CALIBRATION:
        receipt[UK_GATE_OUTCOMES_EVIDENCE] = gate_outcomes(
            UK_SPI_HOUSING_SHELL_STAGE_NAME, model.amount_model
        )
    if geographic_fold is not None:
        receipt[GEOGRAPHIC_FOLD_EVIDENCE] = geographic_fold
    return person_out, household_out, receipt


def _weighted_shares(values: pd.Series, weights: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "value": values.astype(str).to_numpy(),
            "weight": weights.to_numpy(dtype=float),
        }
    )
    totals = frame.groupby("value")["weight"].sum()
    return {str(key): float(value / totals.sum()) for key, value in totals.items()}


#: The manifest's operation, mirrored here and checked at stage time.
UK_SPI_HOUSING_SHELL_OPERATION: dict[str, Any] = {
    "training_rows": "base_frs_channel_positive_weight",
    "rows": "spi_channel",
    "identity": "household_id",
    "predictors": [*CATEGORICAL_PREDICTORS, *NUMERIC_PREDICTORS],
    "categorical_predictors": list(CATEGORICAL_PREDICTORS),
    "categorical_targets": list(HOUSING_CATEGORICAL_COLUMNS),
    "categorical_fit": "weighted_hist_gradient_boosting_classifier",
    "amount_targets": list(HOUSING_AMOUNT_COLUMNS),
    "benefit_targets": list(HOUSING_PERSON_COLUMNS),
    "amount_fit": "regime_gated_qrf",
    "support_clip": "donor_range",
    "structural_zero_keys": ["tenure_type", "region"],
    "rules": [
        "mortgage_requires_mortgaged_tenure",
        "housing_benefit_requires_rented_tenure",
        "council_tax_benefit_capped_at_council_tax",
        "council_tax_rebate_equals_hrp_council_tax_benefit",
        "council_tax_band_within_its_nation",
    ],
    "landing": "household_reference_person",
    "n_estimators": 100,
    "seed": 0,
    "gate_calibration": "cross_fitted",
}
UK_SPI_HOUSING_SHELL_REWRITES: tuple[str, ...] = (
    *HOUSING_CATEGORICAL_COLUMNS,
    *HOUSING_AMOUNT_COLUMNS,
    COUNCIL_TAX_REBATE_COLUMN,
    *HOUSING_PERSON_COLUMNS,
)


#: The imputation's operation kind.
_IMPUTE_KIND = "impute_spi_housing_shell"
#: The seed the transform builds the area draw's definition at: the bill's
#: references read the definition's systems, constraints and weight rules,
#: never its stream, so no seed moves one.
_DEFINITION_SEED = 0


def _stage_operations() -> tuple[tuple[str, dict[str, Any]], ...]:
    """The reviewed operations the stage must declare verbatim, in order: the
    Scottish Band D resource's feed check, as the council-tax localisation
    declares it; the imputation; and the ``derive`` of the Great Britain
    rows' ``council_tax``."""

    return (
        (
            UK_SPI_HOUSING_COUNCIL_TAX_VERIFY_KIND,
            uk_spi_housing_council_tax_verify_parameters(),
        ),
        (_IMPUTE_KIND, dict(UK_SPI_HOUSING_SHELL_OPERATION)),
        (
            UK_SPI_HOUSING_COUNCIL_TAX_DERIVE_KIND,
            uk_spi_housing_council_tax_derive_parameters(),
        ),
    )


def _assert_stage_parameters(stage) -> None:
    """The stage the transform runs as: its name, the reviewed operations
    verbatim and in order, no new outputs, the housing rewrites, and exactly
    the two Band D artifacts, each with ``runtime_sha256_required``."""

    if stage.stage != UK_SPI_HOUSING_SHELL_STAGE_NAME:
        raise ValueError(
            f"Housing shell received stage {stage.stage!r}, expected "
            f"{UK_SPI_HOUSING_SHELL_STAGE_NAME!r}."
        )
    _assert_closed_world_operations(stage, _stage_operations())
    if stage.outputs != () or tuple(stage.rewrites) != UK_SPI_HOUSING_SHELL_REWRITES:
        raise ValueError(
            "spi_housing_shell must declare no new outputs and exactly the "
            f"housing rewrites {list(UK_SPI_HOUSING_SHELL_REWRITES)}."
        )
    declared = sorted(
        (str(artifact.get("role")), str(artifact.get("resource")))
        for artifact in stage.artifacts
        if artifact.get("runtime_sha256_required") is True
    )
    if len(stage.artifacts) != len(UK_SPI_HOUSING_COUNCIL_TAX_ARTIFACTS) or (
        declared != sorted(UK_SPI_HOUSING_COUNCIL_TAX_ARTIFACTS)
    ):
        raise ValueError(
            "spi_housing_shell reads the artifacts "
            f"{list(UK_SPI_HOUSING_COUNCIL_TAX_ARTIFACTS)} (role, resource), each "
            "with runtime_sha256_required: true, and no other; the stage declares "
            "another, or lacks one of them or its flag."
        )


def _operation(stage, kind: str) -> Mapping[str, Any]:
    """The parameters of the stage's one operation of ``kind``."""

    (operation,) = (op for op in stage.operations if op.kind == kind)
    return operation.parameters


@dataclass(frozen=True)
class UKSPIHousingShellStageTransform:
    """Whole-stage callable: impute the SPI support households' housing.

    ``supports`` holds the three conditioned supports' bytes, keyed by
    system, each held to canonical bytes, as the council-tax localisation's
    transform takes them. ``band_d`` is the Band D the Great Britain bills
    are priced at: the committed resources
    (:func:`~microcosm.build.uk_runtime.council_tax_band_d.load_uk_council_tax_band_d`,
    which checks the Scottish resource's feed against the pin) unless a test
    supplies one. Constructing the transform checks the stage it runs as,
    builds the area draw's definition at seed 0 and decodes the supports;
    running it checks the stage again, runs the Scottish resource's runtime
    arm with the frame's period before it reads a row, takes the SPI-channel
    rows' council-tax inputs
    (:func:`~microcosm.build.uk_runtime.spi_housing_council_tax.uk_spi_housing_council_tax_inputs`),
    imputes (:func:`impute_spi_housing_shell`) and checks the housing
    invariants. The receipt starts with the stage.
    """

    stage: Any
    supports: Mapping[str, bytes] = field(repr=False)
    n_estimators: int | None = None
    band_d: UKCouncilTaxBandD | None = field(default=None, repr=False, compare=False)
    definition: Mapping[str, object] = field(init=False, repr=False, compare=False)
    decoded_supports: Mapping[str, AtomicSupport] = field(
        init=False, repr=False, compare=False
    )
    last_receipt: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _assert_stage_parameters(self.stage)
        if self.band_d is not None and not isinstance(self.band_d, UKCouncilTaxBandD):
            raise TypeError("Housing shell prices council tax at a UKCouncilTaxBandD.")
        # Building the definition holds every support to canonical bytes.
        definition = uk_conditioned_assignment_definition(
            self.supports, seed=_DEFINITION_SEED
        )
        decoded = uk_conditioned_supports(self.supports)
        band_d = (
            self.band_d if self.band_d is not None else load_uk_council_tax_band_d()
        )
        object.__setattr__(self, "definition", definition)
        object.__setattr__(self, "decoded_supports", MappingProxyType(decoded))
        object.__setattr__(self, "band_d", band_d)

    def __call__(self, frame):
        from microcosm.build.uk_runtime.national_frame import (
            uk_household_weight_kind,
            uk_national_frame,
            uk_time_period,
            validate_uk_national_frame,
        )

        _assert_stage_parameters(self.stage)
        check_scotland_band_d_declaration(
            _operation(self.stage, UK_SPI_HOUSING_COUNCIL_TAX_VERIFY_KIND),
            stage=UK_SPI_HOUSING_SHELL_STAGE_NAME,
            period=int(uk_time_period(frame)),
        )
        validate_uk_national_frame(frame)
        household = frame.table("household")
        weights = np.asarray(frame.weights_for("household").values, dtype=float)
        council_tax_inputs = uk_spi_housing_council_tax_inputs(
            household,
            definition=self.definition,
            supports=self.decoded_supports,
            band_d=self.band_d,
        )
        person_out, household_out, receipt = impute_spi_housing_shell(
            frame.table("person"),
            household,
            weights,
            council_tax_inputs=council_tax_inputs,
            seed=int(UK_SPI_HOUSING_SHELL_OPERATION["seed"]),
            n_estimators=int(
                self.n_estimators or UK_SPI_HOUSING_SHELL_OPERATION["n_estimators"]
            ),
            gate_calibration=declared_gate_calibration(self.stage, _IMPUTE_KIND),
        )
        _assert_invariants(frame.table("person"), household, person_out, household_out)
        result = uk_national_frame(
            person=person_out,
            benunit=frame.table("benunit").copy(),
            household=household_out,
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=frame.weights_for("household").values,
            mass_log=frame.mass_log,
        )
        validate_uk_national_frame(result)
        object.__setattr__(
            self,
            "last_receipt",
            {"stage": UK_SPI_HOUSING_SHELL_STAGE_NAME, **receipt},
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_receipt is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_receipt}


def _assert_invariants(
    person_in: pd.DataFrame,
    household_in: pd.DataFrame,
    person_out: pd.DataFrame,
    household_out: pd.DataFrame,
) -> None:
    """FRS rows unchanged, and the housing rules hold on every SPI row."""

    frs = (
        household_in["household_support_channel"].astype(str).to_numpy() == FRS_CHANNEL
    )
    if not household_in.loc[frs].equals(household_out.loc[frs]):
        raise ValueError("Housing shell modified FRS base households.")
    frs_ids = set(household_in.loc[frs, "household_id"])
    frs_people = person_in["person_household_id"].isin(frs_ids).to_numpy()
    if not person_in.loc[frs_people].equals(person_out.loc[frs_people]):
        raise ValueError("Housing shell modified FRS base persons.")
    spi = ~frs
    spi_ids = set(household_out.loc[spi, "household_id"])
    tenure = household_out.set_index("household_id")["tenure_type"].astype(str)
    spi_people = person_out.loc[person_out["person_household_id"].isin(spi_ids)]
    hb = spi_people["housing_benefit_reported"].to_numpy(dtype=float)
    hb_tenure = spi_people["person_household_id"].map(tenure).to_numpy()
    if ((hb > 0) & ~np.isin(hb_tenure, RENTED_TENURES)).any():
        raise ValueError("SPI Housing Benefit landed on a non-rented household.")
    off_hrp = ~spi_people["is_household_head"].astype(bool).to_numpy()
    for column in HOUSING_PERSON_COLUMNS:
        if (spi_people[column].to_numpy(dtype=float)[off_hrp] != 0).any():
            raise ValueError(
                f"SPI {column} must sit on the household reference person."
            )
    ctb = spi_people.groupby("person_household_id")[
        "council_tax_benefit_reported"
    ].sum()
    council_tax = household_out.set_index("household_id")["council_tax"]
    if (ctb > council_tax.reindex(ctb.index) + 1e-9).any():
        raise ValueError("SPI council tax benefit exceeds council tax.")
    spi_households = household_out.loc[spi]
    mortgage = spi_households["mortgage_interest_repayment"].to_numpy(
        dtype=float
    ) + spi_households["mortgage_capital_repayment"].to_numpy(dtype=float)
    if (
        (mortgage > 0)
        & (
            spi_households["tenure_type"].astype(str).to_numpy()
            != "OWNED_WITH_MORTGAGE"
        )
    ).any():
        raise ValueError("SPI mortgage payments without a mortgaged tenure.")
