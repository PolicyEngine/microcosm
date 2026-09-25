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
renting reporters only. FRS base rows are never modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from microcosm.build.stochastic_assignment import stable_identity_uniforms

__all__ = [
    "HOUSING_AMOUNT_COLUMNS",
    "HOUSING_CATEGORICAL_COLUMNS",
    "HOUSING_PERSON_COLUMNS",
    "UK_SPI_HOUSING_SHELL_STAGE_NAME",
    "FittedHousingModel",
    "fit_housing_model",
    "household_housing_predictors",
    "household_housing_targets",
    "impute_spi_housing_shell",
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
_INVESTMENT_COLUMNS = (
    "savings_interest_income",
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

    def draw(self, predictors: pd.DataFrame) -> pd.DataFrame:
        """Draw the housing bundle for ``predictors`` (indexed by household id)."""

        ids = predictors.index.to_numpy()
        drawn = pd.DataFrame(index=predictors.index)
        for column in HOUSING_CATEGORICAL_COLUMNS:
            features = self._features(predictors, drawn)
            classifier = self.classifiers[column]
            probabilities = classifier.predict_proba(features.to_numpy(dtype=float))
            uniforms = stable_identity_uniforms(
                ids, seed=self.seed, salt=f"{UK_SPI_HOUSING_SHELL_STAGE_NAME}:{column}"
            )
            cumulative = np.cumsum(probabilities, axis=1)
            cumulative[:, -1] = 1.0
            choice = (cumulative > uniforms[:, None]).argmax(axis=1)
            drawn[column] = np.asarray(classifier.classes_)[choice].astype(str)
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
) -> FittedHousingModel:
    """Fit the categorical steps and the chained amounts on training households."""

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
    model.amount_model = RegimeGatedQRF(n_estimators=n_estimators, seed=seed).fit(
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


def apply_structural_rules(
    draws: pd.DataFrame,
    region: pd.Series,
    structural_zeros: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Force the learned structural zeros and the declared tenure rules."""

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
    ctb = out["council_tax_benefit_hrp"].to_numpy(dtype=float)
    cap = out["council_tax"].to_numpy(dtype=float)
    fired["council_tax_benefit_hrp:cap"] = int((ctb > cap).sum())
    out["council_tax_benefit_hrp"] = np.minimum(ctb, cap)
    return out, fired


def impute_spi_housing_shell(
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
    *,
    seed: int = 0,
    n_estimators: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Redraw every SPI-channel household's housing; FRS rows are untouched."""

    channel = household["household_support_channel"].astype(str).to_numpy()
    if not set(channel) <= {FRS_CHANNEL, SPI_CHANNEL}:
        raise ValueError("Housing shell expects only frs and spi support channels.")
    weights = pd.Series(
        np.asarray(household_weights, dtype=float), index=household["household_id"]
    )
    training_ids = household.loc[
        (channel == FRS_CHANNEL) & (weights.to_numpy() > 0.0), "household_id"
    ]
    recipient_ids = household.loc[channel == SPI_CHANNEL, "household_id"]
    if training_ids.empty or recipient_ids.empty:
        raise ValueError("Housing shell needs FRS training rows and SPI recipients.")

    predictors = household_housing_predictors(person, household)
    targets = household_housing_targets(person, household)
    model = fit_housing_model(
        predictors.loc[training_ids],
        targets.loc[training_ids],
        weights.loc[training_ids].to_numpy(),
        seed=seed,
        n_estimators=n_estimators,
    )
    raw = model.draw(predictors.loc[recipient_ids])
    draws, fired = apply_structural_rules(
        raw, predictors["region"], model.structural_zeros
    )

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
                    targets.loc[training_ids, column], weights.loc[training_ids]
                ),
                "recipients": _weighted_shares(
                    draws[column], weights.loc[recipient_ids]
                ),
            }
            for column in HOUSING_CATEGORICAL_COLUMNS
        },
    }
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
    ],
    "landing": "household_reference_person",
    "n_estimators": 100,
    "seed": 0,
}
UK_SPI_HOUSING_SHELL_REWRITES: tuple[str, ...] = (
    *HOUSING_CATEGORICAL_COLUMNS,
    *HOUSING_AMOUNT_COLUMNS,
    COUNCIL_TAX_REBATE_COLUMN,
    *HOUSING_PERSON_COLUMNS,
)


def _assert_stage_parameters(stage) -> None:
    if stage.stage != UK_SPI_HOUSING_SHELL_STAGE_NAME:
        raise ValueError(
            f"Housing shell received stage {stage.stage!r}, expected "
            f"{UK_SPI_HOUSING_SHELL_STAGE_NAME!r}."
        )
    kinds = [operation.kind for operation in stage.operations]
    if kinds != ["impute_spi_housing_shell"]:
        raise ValueError(
            f"spi_housing_shell operations drifted: expected "
            f"['impute_spi_housing_shell'], got {kinds}."
        )
    actual = dict(stage.operations[0].parameters)
    if actual != UK_SPI_HOUSING_SHELL_OPERATION:
        raise ValueError(
            "spi_housing_shell parameters drifted: expected "
            f"{UK_SPI_HOUSING_SHELL_OPERATION}, got {actual}."
        )
    if stage.outputs != () or tuple(stage.rewrites) != UK_SPI_HOUSING_SHELL_REWRITES:
        raise ValueError(
            "spi_housing_shell must declare no new outputs and exactly the "
            f"housing rewrites {list(UK_SPI_HOUSING_SHELL_REWRITES)}."
        )


@dataclass
class UKSPIHousingShellStageTransform:
    """Whole-stage callable: impute the SPI support households' housing."""

    stage: Any
    n_estimators: int | None = None
    last_receipt: dict[str, Any] | None = field(default=None, init=False)

    def __call__(self, frame):
        from microcosm.build.uk_runtime.national_frame import (
            uk_household_weight_kind,
            uk_national_frame,
            uk_time_period,
            validate_uk_national_frame,
        )

        _assert_stage_parameters(self.stage)
        validate_uk_national_frame(frame)
        household = frame.table("household")
        weights = np.asarray(frame.weights_for("household").values, dtype=float)
        person_out, household_out, receipt = impute_spi_housing_shell(
            frame.table("person"),
            household,
            weights,
            seed=int(UK_SPI_HOUSING_SHELL_OPERATION["seed"]),
            n_estimators=int(
                self.n_estimators or UK_SPI_HOUSING_SHELL_OPERATION["n_estimators"]
            ),
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
        self.last_receipt = {"stage": UK_SPI_HOUSING_SHELL_STAGE_NAME, **receipt}
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
