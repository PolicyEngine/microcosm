"""Redraw the SPI-drawn rows' occupation from their own incomes, with a coherent industry.

The SPI support channel and the income band donors are whole copies of FRS
households, and ``hmrc_spi_income_spine`` then draws their claimants' and
partners' incomes from the SPI tape. A drawn row keeps its FRS twin's SOC 2020
major group, industry and sector, all chosen with the twin's pay, not the
incomes it drew; held out on the FRS, the twin's group misses the occupation
mix of people at the drawn pay by 0.15 to 0.38 (total variation) in every paid
band. This stage is the person counterpart of ``spi_housing_shell``: it trains
on the FRS rows' adults with an occupation and draws each drawn row's major
group from its age, sex, region, employment-status group and incomes, then
copies an industry division and sector pair from an FRS adult with the drawn
group, the same status group and the nearest age and incomes, so occupation and
industry stay a pair that occurs and the sector one the row's status holds. A
row draws only among the groups that FRS adults of its own status group hold,
so such an adult always exists. A drawn row's pay is the tape's PAY
(``hmrc_spi_pay``): its ``employment_income`` also holds the tape's EPB and
TAXTERM, and termination pay says nothing of an occupation.

The model is ``microcosm.fit``'s country-agnostic categorical classifier, the
equal mixture of a regularised boosted classifier and a spline logit; the held-
out evidence and the donor choice are its held-out scores and keyed nearest
donors. Every draw reads uniforms keyed on ``person_id``, so adding or
reordering rows never moves a draw. FRS rows, undrawn copies, and drawn rows
whose twin had no occupation and who drew no pay or profit are byte-unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.stochastic_assignment import (
    integer_record_keys,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.household_lineage import collapse_geographic_clones
from microcosm.build.uk_runtime.occupation_constants import (
    UK_OCCUPATION_STATUS_GROUP,
    UK_SOC2020_NO_OCCUPATION,
    uk_sic2007_sections,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    rounded_count,
    small_cell_share,
    source_household_ids,
    support_band,
)
from microcosm.fit import (
    BoostedClassifierSettings,
    CategoricalClassifier,
    SplineLogitSettings,
    can_deal_folds,
    categorical_scores,
    donor_distance_features,
    inverse_cdf_draw,
    keyed_nearest_donors,
    out_of_fold_proba,
    stratified_folds,
)

__all__ = [
    "UK_SPI_EMPLOYMENT_SHELL_OPERATION",
    "UK_SPI_EMPLOYMENT_SHELL_REWRITES",
    "UK_SPI_EMPLOYMENT_SHELL_STAGE_NAME",
    "UKSPIEmploymentShellStageTransform",
    "drawn_rows",
    "impute_spi_employment_shell",
    "occupation_model",
]

UK_SPI_EMPLOYMENT_SHELL_STAGE_NAME = "spi_employment_shell"
FRS_CHANNEL = "frs"
OCCUPATION_COLUMN = "soc2020_major_group"
INDUSTRY_COLUMNS = ("sic_industry_division", "employment_sector")
UK_SPI_EMPLOYMENT_SHELL_REWRITES: tuple[str, ...] = (
    OCCUPATION_COLUMN,
    *INDUSTRY_COLUMNS,
)
STATUS_GROUP = "status_group"
#: Investment income is savings interest plus dividends, as the SPI draws them.
INVESTMENT_COMPONENTS = ("savings_interest_income", "dividend_income")
INCOME_COLUMNS = (
    "employment_income",
    "self_employment_income",
    "investment_income",
    "private_pension_income",
)
OCCUPATION_PREDICTORS = ("age", "gender", "region", STATUS_GROUP, *INCOME_COLUMNS)
CATEGORICAL_PREDICTORS = ("gender", "region", STATUS_GROUP)
#: The SPI tape's own PAY: ``employment_income`` can be termination pay or
#: benefits alone, which says nothing of an occupation.
SPI_PAY_COLUMN = "hmrc_spi_pay"
PAY_BANDS = (
    "no_pay",
    "below_p50",
    "p50_p80",
    "p80_p90",
    "p90_p95",
    "p95_p99",
    "top_1pc",
)
_ADULT_AGE = 16
_PUBLIC_SECTOR = "PUBLIC"

#: The manifest's operation, mirrored here and checked at stage time.
UK_SPI_EMPLOYMENT_SHELL_OPERATION: dict[str, Any] = {
    "training_rows": "frs_channel_adults_with_occupation_positive_weight",
    "rows": "spi_income_recipients_with_occupation_pay_or_profit",
    "identity": "person_id",
    "pay": "employment_income_or_the_tape_pay_of_spi_income_recipients",
    "draw_support": "groups_frs_adults_of_the_rows_status_group_hold",
    "seed": 0,
    "occupation": {
        "target": OCCUPATION_COLUMN,
        "predictors": list(OCCUPATION_PREDICTORS),
        "categorical_predictors": list(CATEGORICAL_PREDICTORS),
        "arcsinh_columns": list(INCOME_COLUMNS),
        "arcsinh_scale": 1000.0,
        "boosted": {
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 40,
            "l2_regularization": 1.0,
            "early_stopping": True,
            "validation_fraction": 0.2,
            "n_iter_no_change": 20,
            "max_iter": 500,
        },
        "logit": {
            "n_knots": 6,
            "degree": 3,
            "knots": "quantile",
            "extrapolation": "constant",
            "penalty_c": 1.0,
            "solver": "lbfgs",
            "max_iter": 3000,
        },
        "boosted_weight": 0.5,
    },
    "industry": {
        "columns": list(INDUSTRY_COLUMNS),
        "cells": [OCCUPATION_COLUMN, STATUS_GROUP],
        "neighbours": 50,
        "rank_columns": ["age"],
        "presence_rank_columns": list(INCOME_COLUMNS),
        "region_scale": 0.3,
    },
    "evidence": {"folds": 5, "pay_band_quantiles": [0.5, 0.8, 0.9, 0.95, 0.99]},
}


def _plain(value: object) -> object:
    """A declaration as plain dicts and lists, for an exact comparison."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def drawn_rows(person: pd.DataFrame) -> np.ndarray:
    """The rows the stage redraws: SPI income recipients whose twin had an
    occupation, or who drew pay or a profit of their own."""
    recipient = person["person_is_spi_income_recipient"].astype(bool).to_numpy()
    coded = person[OCCUPATION_COLUMN].astype(str).to_numpy() != UK_SOC2020_NO_OCCUPATION
    works = (person[SPI_PAY_COLUMN].to_numpy(dtype=np.float64) > 0) | (
        person["self_employment_income"].to_numpy(dtype=np.float64) > 0
    )
    return recipient & (coded | works)


def occupation_model(
    operation: Mapping[str, Any], *, boosted_max_iter: int | None = None
) -> CategoricalClassifier:
    """The declared classifier.

    ``boosted_max_iter`` caps the boosted learner on a small fixture, and turns
    its early stopping off there: the validation split early stopping holds out
    is stratified, and a fixture of a few households has classes too thin to
    split.
    """
    occupation = operation["occupation"]
    boosted = dict(occupation["boosted"])
    if boosted_max_iter is not None:
        boosted["max_iter"] = int(boosted_max_iter)
        boosted["early_stopping"] = False
    return CategoricalClassifier(
        boosted_weight=float(occupation["boosted_weight"]),
        boosted=BoostedClassifierSettings(**boosted),
        logit=SplineLogitSettings(**dict(occupation["logit"])),
        categorical_predictors=tuple(occupation["categorical_predictors"]),
        arcsinh_scales={
            column: float(occupation["arcsinh_scale"])
            for column in occupation["arcsinh_columns"]
        },
        seed=int(operation["seed"]),
    )


def _pay_bands(
    pay: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    """No pay, then bands at the weighted quantiles of positive pay."""
    band = np.zeros(len(pay), dtype=np.int64)
    positive = pay > 0
    if positive.any():
        order = np.argsort(pay[positive], kind="stable")
        cumulative = np.cumsum(weights[positive][order])
        cuts = np.interp(quantiles, cumulative / cumulative[-1], pay[positive][order])
        band[positive] = np.digitize(pay[positive], cuts) + 1
    return band


def _person_table(person: pd.DataFrame, household: pd.DataFrame) -> pd.DataFrame:
    """The predictors, target, keys and channel of every person.

    An SPI income recipient's ``employment_income`` predictor is the tape's
    PAY: the column itself also holds the tape's EPB and TAXTERM (less the
    company car benefits), which say nothing of an occupation.
    """
    households = household.set_index("household_id")
    by_person = person["person_household_id"].to_numpy()
    pay = np.where(
        person["person_is_spi_income_recipient"].astype(bool).to_numpy(),
        person[SPI_PAY_COLUMN].to_numpy(dtype=np.float64),
        person["employment_income"].to_numpy(dtype=np.float64),
    )
    table = pd.DataFrame(
        {
            "person_id": person["person_id"].to_numpy(),
            "household_id": by_person,
            "age": person["age"].to_numpy(dtype=np.float64),
            "gender": person["gender"].astype(str).to_numpy(),
            "region": households["region"].astype(str).reindex(by_person).to_numpy(),
            STATUS_GROUP: person["employment_status"]
            .astype(str)
            .map(UK_OCCUPATION_STATUS_GROUP)
            .to_numpy(dtype=object),
            "employment_income": pay,
            "self_employment_income": person["self_employment_income"].to_numpy(
                dtype=np.float64
            ),
            "investment_income": sum(
                person[column].to_numpy(dtype=np.float64)
                for column in INVESTMENT_COMPONENTS
            ),
            "private_pension_income": person["private_pension_income"].to_numpy(
                dtype=np.float64
            ),
            OCCUPATION_COLUMN: person[OCCUPATION_COLUMN].astype(str).to_numpy(),
            "channel": households["household_support_channel"]
            .astype(str)
            .reindex(by_person)
            .to_numpy(),
        },
        index=person.index,
    )
    return table


def _level_gaps(
    levels: Mapping[object, Mapping[str, float]],
    names: Sequence[str] | None,
    support: Mapping[object, int],
) -> tuple[dict[str, object], float, float]:
    """Banded per-level share gaps, and their weighted mean and maximum."""
    out: dict[str, object] = {}
    total = weighted = largest = 0.0
    for level, values in levels.items():
        name = names[int(level)] if names is not None else str(level)
        households = int(support.get(level, 0))
        out[name] = {
            "support": support_band(households),
            "share_gap": small_cell_share(float(values["share_gap"]), households),
        }
        total += float(values["weight"])
        weighted += float(values["weight"]) * float(values["share_gap"])
        largest = max(largest, float(values["share_gap"]))
    return out, (weighted / total if total else 0.0), largest


def _held_out_evidence(
    model: CategoricalClassifier,
    train: pd.DataFrame,
    weights: np.ndarray,
    sources: np.ndarray,
    folds: np.ndarray,
    quantiles: Sequence[float],
) -> dict[str, object]:
    """Out-of-fold scores of the model, each learner and the constant."""
    predictors = list(OCCUPATION_PREDICTORS)
    out = out_of_fold_proba(
        model, train, predictors, OCCUPATION_COLUMN, weights=weights, folds=folds
    )
    labels = train[OCCUPATION_COLUMN].to_numpy()
    bands = _pay_bands(train["employment_income"].to_numpy(), weights, quantiles)
    status = train[STATUS_GROUP].to_numpy()
    scores = {
        name: categorical_scores(
            out[name],
            labels,
            out["classes"],
            weights,
            strata={"pay_band": bands, "status_group": status},
        )
        for name in ("model", "boosted", "spline_logit", "marginal")
        if name in out
    }
    model_scores = scores["model"]
    band_support = {
        band: distinct_count(sources, bands == band) for band in np.unique(bands)
    }
    status_support = {
        group: distinct_count(sources, status == group) for group in np.unique(status)
    }
    pay, pay_mean, _ = _level_gaps(
        model_scores["strata"]["pay_band"], PAY_BANDS, band_support
    )
    groups, _, status_max = _level_gaps(
        model_scores["strata"]["status_group"], None, status_support
    )
    return {
        "status": "measured",
        "folds": int(len(np.unique(folds))),
        "grouping": "household",
        "log_loss": {name: float(value["log_loss"]) for name, value in scores.items()},
        "calibration_slope": model_scores["calibration_slope"],
        "calibration_outcome": model_scores["calibration_outcome"],
        "share_gap": float(model_scores["share_gap"]),
        "pay_band_share_gap_mean": float(pay_mean),
        "pay_bands": pay,
        "status_group_share_gap_max": float(status_max),
        "status_groups": groups,
    }


def _copy_cells(groups: np.ndarray, status: np.ndarray) -> list[np.ndarray]:
    """The one donor cell, group and status group: a copied sector is always one
    the row's status holds, and a (group, section) pair one that occurs."""
    return [
        np.asarray(
            [f"{g}|{s}" for g, s in zip(groups, status, strict=True)], dtype=object
        )
    ]


def _copy_donors(
    recipients: pd.DataFrame,
    donors: pd.DataFrame,
    recipient_groups: np.ndarray,
    donor_weights: np.ndarray,
    uniforms: np.ndarray,
    industry: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    r, d = donor_distance_features(
        recipients,
        donors,
        rank_columns=list(industry["rank_columns"]),
        presence_rank_columns=list(industry["presence_rank_columns"]),
        scaled_categoricals={"region": float(industry["region_scale"])},
    )
    return keyed_nearest_donors(
        r,
        d,
        recipient_cells=_copy_cells(
            recipient_groups, recipients[STATUS_GROUP].to_numpy()
        ),
        donor_cells=_copy_cells(
            donors[OCCUPATION_COLUMN].to_numpy(), donors[STATUS_GROUP].to_numpy()
        ),
        donor_weights=donor_weights,
        uniforms=uniforms,
        k=int(industry["neighbours"]),
    )


def _held_out_industry(
    train: pd.DataFrame,
    industry_values: pd.DataFrame,
    weights: np.ndarray,
    folds: np.ndarray,
    industry: Mapping[str, Any],
    seed: int,
) -> dict[str, object]:
    """Copy each fold's pairs from the other folds, given the adults' own groups.

    The gaps are the weighted mean, over major groups, of the total variation
    between the copied and observed SIC section shares, and of the absolute
    public-sector share difference. An adult whose group and status group no
    adult of the other folds holds has no donor; it is left out and counted.
    """
    copied = np.full(len(train), -1, dtype=np.int64)
    keys = train["person_id"].to_numpy()
    cells = _copy_cells(
        train[OCCUPATION_COLUMN].to_numpy(), train[STATUS_GROUP].to_numpy()
    )[0]
    for fold in np.unique(folds):
        rest = folds != fold
        test = (folds == fold) & np.isin(cells, cells[rest & (weights > 0)])
        if not test.any():
            continue
        positions, _ = _copy_donors(
            train.loc[test],
            train.loc[rest],
            train.loc[test, OCCUPATION_COLUMN].to_numpy(),
            weights[rest],
            stable_identity_uniforms(
                keys[test],
                seed=seed,
                salt=f"{UK_SPI_EMPLOYMENT_SHELL_STAGE_NAME}:held_out_industry",
            ),
            industry,
        )
        copied[test] = np.flatnonzero(rest)[positions]
    section = uk_sic2007_sections(industry_values["sic_industry_division"].to_numpy())
    public = (
        industry_values["employment_sector"].astype(str).to_numpy() == _PUBLIC_SECTOR
    )
    groups = train[OCCUPATION_COLUMN].to_numpy()
    total = section_gap = public_gap = 0.0
    for group in np.unique(groups):
        rows = (groups == group) & (copied >= 0)
        if not rows.any():
            continue
        w = weights[rows]
        observed = pd.Series(w).groupby(section[rows]).sum() / w.sum()
        drawn = pd.Series(w).groupby(section[copied[rows]]).sum() / w.sum()
        levels = observed.index.union(drawn.index)
        gap = (
            0.5
            * (
                observed.reindex(levels, fill_value=0)
                - drawn.reindex(levels, fill_value=0)
            )
            .abs()
            .sum()
        )
        section_gap += w.sum() * float(gap)
        public_gap += w.sum() * abs(
            float(w @ public[rows] - w @ public[copied[rows]]) / w.sum()
        )
        total += w.sum()
    without = rounded_count(int((copied < 0).sum()))
    if not total:
        return {"status": "not_measured", "adults_without_a_donor": without}
    return {
        "status": "measured",
        "section_share_gap_within_group": section_gap / total,
        "public_sector_share_gap_within_group": public_gap / total,
        "adults_without_a_donor": without,
    }


def _draw_held_groups(
    fitted,
    recipients: pd.DataFrame,
    train: pd.DataFrame,
    train_weights: np.ndarray,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Each row's group, drawn among the groups FRS adults of its status group hold.

    The other groups' probabilities go to zero and the rest are renormalised,
    so the industry copy always has a donor of the row's own group and status
    group. Rows that lose no probability draw exactly as the classifier gives.
    Returns the drawn groups and which rows lost probability; a row whose status
    group holds none of the groups refuses the stage.
    """
    classes = fitted.classes_
    proba = fitted.predict_proba(recipients)
    holders = train.loc[train_weights > 0]
    status = recipients[STATUS_GROUP].to_numpy()
    allowed = np.zeros(proba.shape, dtype=bool)
    for status_group, groups in (
        holders.groupby(STATUS_GROUP)[OCCUPATION_COLUMN].unique().items()
    ):
        rows = np.flatnonzero(status == status_group)
        allowed[np.ix_(rows, np.flatnonzero(np.isin(classes, groups)))] = True
    restricted = (proba * ~allowed).sum(axis=1) > 0
    if restricted.any():
        kept = np.where(allowed[restricted], proba[restricted], 0.0)
        total = kept.sum(axis=1)
        if (total <= 0).any():
            raise ValueError(
                f"{UK_SPI_EMPLOYMENT_SHELL_STAGE_NAME}: a drawn row's status group "
                "holds none of the groups it could draw."
            )
        proba = proba.copy()
        proba[restricted] = kept / total[:, None]
    return classes[inverse_cdf_draw(proba, uniforms)], restricted


def _weighted_shares(
    values: np.ndarray, weights: np.ndarray, sources: np.ndarray
) -> dict[str, object]:
    out: dict[str, object] = {}
    total = weights.sum()
    for value in np.unique(values):
        rows = values == value
        out[str(value)] = small_cell_share(
            float(weights[rows].sum() / total), distinct_count(sources, rows)
        )
    return out


def impute_spi_employment_shell(
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
    *,
    operation: Mapping[str, Any] = UK_SPI_EMPLOYMENT_SHELL_OPERATION,
    boosted_max_iter: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Redraw the drawn SPI rows' occupation, industry and sector; FRS rows untouched."""

    stage = UK_SPI_EMPLOYMENT_SHELL_STAGE_NAME
    table = _person_table(person, household)
    weight_by_household = pd.Series(
        np.asarray(household_weights, dtype=np.float64),
        index=household["household_id"].to_numpy(),
    )
    weights = weight_by_household.reindex(table["household_id"]).to_numpy()
    source_by_household = pd.Series(
        source_household_ids(household, stage=stage),
        index=household["household_id"].to_numpy(),
    )
    sources = source_by_household.reindex(table["household_id"]).to_numpy()
    coded = table[OCCUPATION_COLUMN].to_numpy() != UK_SOC2020_NO_OCCUPATION
    training = (
        (table["channel"].to_numpy() == FRS_CHANNEL)
        & (weights > 0)
        & (table["age"].to_numpy() >= _ADULT_AGE)
        & coded
    )
    recipient = person["person_is_spi_income_recipient"].astype(bool).to_numpy()
    drawn = drawn_rows(person)
    if (recipient & (table["channel"].to_numpy() == FRS_CHANNEL)).any():
        raise ValueError(
            f"{stage}: an FRS-channel row is marked an SPI income recipient."
        )
    unmapped = (training | drawn) & pd.isna(table[STATUS_GROUP]).to_numpy()
    if unmapped.any():
        statuses = sorted(set(person.loc[unmapped, "employment_status"].astype(str)))
        raise ValueError(
            f"{stage}: rows to train on or draw for hold employment statuses "
            f"{statuses} that have no occupation status group."
        )
    if not training.any():
        raise ValueError(f"{stage} needs FRS adults with an occupation to train on.")
    # The fit and the donors see each survey record once: its geographic
    # clones that agree on the predictors, the occupation and the industry it
    # donates fold into its first row with their weight, clones that differ
    # stay rows of their own, and every other copy stays its own row; a frame
    # without geographic clones keeps every row and weight
    # (household_lineage.collapse_geographic_clones, microcosm#931).
    training, fit_weights = collapse_geographic_clones(
        {"person": person, "household": household},
        "person",
        training,
        weights,
        inputs=pd.concat(
            [
                table[[*OCCUPATION_PREDICTORS, OCCUPATION_COLUMN]].reset_index(
                    drop=True
                ),
                person[list(INDUSTRY_COLUMNS)].reset_index(drop=True),
            ],
            axis=1,
        ),
    )

    # The training rows in person_id order, so neither the fit (the early
    # stopping split, the folds) nor the donor ties depend on how the frame
    # orders its rows.
    train_rows = np.flatnonzero(training)
    train_rows = train_rows[
        np.argsort(table["person_id"].to_numpy()[train_rows], kind="stable")
    ]
    train = table.iloc[train_rows].reset_index(drop=True)
    train_weights = fit_weights[train_rows]
    train_sources = sources[train_rows]
    model = occupation_model(operation, boosted_max_iter=boosted_max_iter)
    seed = int(operation["seed"])
    evidence_settings = operation["evidence"]
    fold_count = int(evidence_settings["folds"])
    labels = train[OCCUPATION_COLUMN].to_numpy()
    if can_deal_folds(labels, fold_count, groups=train["household_id"].to_numpy()):
        folds = stratified_folds(
            labels,
            fold_count,
            np.random.default_rng(np.random.SeedSequence(seed)),
            groups=train["household_id"].to_numpy(),
        )
        held_out = _held_out_evidence(
            model,
            train,
            train_weights,
            train_sources,
            folds,
            list(evidence_settings["pay_band_quantiles"]),
        )
        industry_held_out: dict[str, object] = _held_out_industry(
            train,
            person.iloc[train_rows][list(INDUSTRY_COLUMNS)].reset_index(drop=True),
            train_weights,
            folds,
            operation["industry"],
            seed,
        )
    else:
        held_out = {
            "status": "not_measured",
            "reason": "a major group has fewer training households than folds",
            "log_loss": None,
            "calibration_slope": None,
        }
        industry_held_out = {"status": "not_measured"}

    fitted = model.fit(
        train, list(OCCUPATION_PREDICTORS), [OCCUPATION_COLUMN], weights=train_weights
    )
    person_out = person.copy()
    recipients = table.loc[drawn].reset_index(drop=True)
    keys = integer_record_keys(
        recipients["person_id"].to_numpy(), label=f"{stage} person_id"
    )
    new_groups = np.asarray([], dtype=object)
    restricted = np.zeros(0, dtype=bool)
    mismatched = 0
    if len(recipients):
        new_groups, restricted = _draw_held_groups(
            fitted,
            recipients,
            train,
            train_weights,
            stable_identity_uniforms(
                keys, seed=seed, salt=f"{stage}:{OCCUPATION_COLUMN}"
            ),
        )
        donor_positions, _ = _copy_donors(
            recipients,
            train,
            new_groups,
            train_weights,
            stable_identity_uniforms(keys, seed=seed, salt=f"{stage}:industry"),
            operation["industry"],
        )
        mismatched = int(
            (
                train[STATUS_GROUP].to_numpy()[donor_positions]
                != recipients[STATUS_GROUP].to_numpy()
            ).sum()
        )
        donor_rows = train_rows[donor_positions]
        person_out.loc[drawn, OCCUPATION_COLUMN] = new_groups
        for column in INDUSTRY_COLUMNS:
            person_out.loc[drawn, column] = person[column].to_numpy()[donor_rows]

    drawn_weights = weights[drawn]
    receipt: dict[str, Any] = {
        "training": {
            "people": rounded_count(int(training.sum())),
            "source_households": rounded_count(distinct_count(sources, training)),
        },
        "drawn": {
            "people": rounded_count(int(drawn.sum())),
            "twin_had_an_occupation": rounded_count(int((drawn & coded).sum())),
            "newly_coded": rounded_count(int((drawn & ~coded).sum())),
            "kept_no_occupation": rounded_count(int((recipient & ~drawn).sum())),
        },
        "fit": fitted.diagnostics,
        "held_out": held_out,
        "category_shares": {
            "training": _weighted_shares(labels, train_weights, train_sources),
            "drawn": (
                _weighted_shares(new_groups, drawn_weights, sources[drawn])
                if len(recipients)
                else {}
            ),
        },
        "industry_copy": {
            "neighbours": int(operation["industry"]["neighbours"]),
            "cell": "group_and_status_group",
            "restricted_draws": rounded_count(int(restricted.sum())),
            "donors_of_another_status_group": rounded_count(mismatched),
            "held_out": industry_held_out,
        },
    }
    return person_out, receipt


def _assert_stage_parameters(stage) -> None:
    if stage.stage != UK_SPI_EMPLOYMENT_SHELL_STAGE_NAME:
        raise ValueError(
            f"Employment shell received stage {stage.stage!r}, expected "
            f"{UK_SPI_EMPLOYMENT_SHELL_STAGE_NAME!r}."
        )
    kinds = [operation.kind for operation in stage.operations]
    if kinds != ["impute_spi_employment_shell"]:
        raise ValueError(
            "spi_employment_shell operations drifted: expected "
            f"['impute_spi_employment_shell'], got {kinds}."
        )
    actual = _plain(dict(stage.operations[0].parameters))
    if actual != _plain(UK_SPI_EMPLOYMENT_SHELL_OPERATION):
        raise ValueError(
            "spi_employment_shell parameters drifted: expected "
            f"{UK_SPI_EMPLOYMENT_SHELL_OPERATION}, got {actual}."
        )
    if stage.outputs != () or tuple(stage.rewrites) != UK_SPI_EMPLOYMENT_SHELL_REWRITES:
        raise ValueError(
            "spi_employment_shell must declare no new outputs and exactly the "
            f"rewrites {list(UK_SPI_EMPLOYMENT_SHELL_REWRITES)}."
        )


def _assert_invariants(
    person_in: pd.DataFrame, person_out: pd.DataFrame, drawn: np.ndarray
) -> None:
    """Only drawn rows change, and only in the rewritten columns."""
    untouched = [
        c for c in person_in.columns if c not in UK_SPI_EMPLOYMENT_SHELL_REWRITES
    ]
    if not person_in[untouched].equals(person_out[untouched]):
        raise ValueError("Employment shell modified a column it does not rewrite.")
    if not person_in.loc[~drawn].equals(person_out.loc[~drawn]):
        raise ValueError("Employment shell modified a row it does not draw.")
    groups = person_out.loc[drawn, OCCUPATION_COLUMN].astype(str)
    if (groups == UK_SOC2020_NO_OCCUPATION).any():
        raise ValueError("Employment shell left a drawn row with no occupation.")


@dataclass
class UKSPIEmploymentShellStageTransform:
    """Whole-stage callable: redraw the drawn SPI rows' occupation and industry."""

    stage: Any
    boosted_max_iter: int | None = None
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
        person = frame.table("person")
        person_out, receipt = impute_spi_employment_shell(
            person,
            frame.table("household"),
            np.asarray(frame.weights_for("household").values, dtype=float),
            boosted_max_iter=self.boosted_max_iter,
        )
        _assert_invariants(person, person_out, drawn_rows(person))
        result = uk_national_frame(
            person=person_out,
            benunit=frame.table("benunit").copy(),
            household=frame.table("household").copy(),
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=frame.weights_for("household").values,
            mass_log=frame.mass_log,
        )
        validate_uk_national_frame(result)
        self.last_receipt = {"stage": UK_SPI_EMPLOYMENT_SHELL_STAGE_NAME, **receipt}
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_receipt is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_receipt}
