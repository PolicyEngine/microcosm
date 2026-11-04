"""Impute each person's SOC 2020 sub-major group within their major group, from the APS.

The FRS codes an occupation to the major group only, and the SPI employment
shell redraws that group on the SPI-drawn rows. The Annual Population Survey
codes the minor group of the main job, and of the last job for people not in
work, so its adults are the donors for the sub-major group: the minor group's
first two digits. Each major group gets its own model over its sub-major
groups, ``microcosm.fit``'s categorical classifier (the equal mixture of a
regularised boosted classifier and a spline logit) on age, sex, region,
employment-status group, annual pay and SIC 2007 section, the predictors both
surveys record alike. A person with no major group keeps NONE.

The donors are the APS adults with a minor group whose pay is known: an
employee's pay is asked in the income waves only, so an employee with pay
stands for all employees through the income weight, and everyone else carries
the person weight. The held-out evidence deals five stratified folds by person,
since the End User Licence file carries no household identifier.

Draws read uniforms keyed on ``person_source_id``, so a copy whose predictors
are its FRS twin's draws its twin's group, except for the SPI income
recipients, whose incomes and occupation the SPI stages drew: they key on
their own ``person_id`` under a salt of their own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.stochastic_assignment import (
    integer_record_keys,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.occupation_constants import (
    UK_OCCUPATION_STATUS_GROUP,
    UK_SIC2007_SECTION_DIVISIONS,
    UK_SOC2020_MAJOR_GROUP_OF_SUB_MAJOR_GROUP,
    UK_SOC2020_MAJOR_GROUPS,
    UK_SOC2020_NO_OCCUPATION,
    UK_SOC2020_SUB_MAJOR_GROUPS,
    uk_sic2007_sections,
)
from microcosm.build.uk_runtime.receipt_counts import (
    UK_DISCLOSURE_FLOOR,
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
    out_of_fold_proba,
    stratified_folds,
)

__all__ = [
    "CLEAN_APS_OCCUPATION_TABLE_KIND",
    "IMPUTE_SOC2020_SUB_MAJOR_GROUP_KIND",
    "UK_APS_OCCUPATION_CLEANING",
    "UK_APS_OCCUPATION_IMPUTATION",
    "UK_APS_OCCUPATION_OUTPUT",
    "UK_APS_OCCUPATION_STAGE_NAME",
    "UK_APS_PERSON_TAB_ROLE",
    "UKAPSOccupationStageTransform",
    "clean_aps_occupation_table",
    "impute_soc2020_sub_major_group",
    "sub_major_model",
]

UK_APS_OCCUPATION_STAGE_NAME = "aps_occupation"
UK_APS_PERSON_TAB_ROLE = "aps_person_tab"
CLEAN_APS_OCCUPATION_TABLE_KIND = "clean_aps_occupation_table"
IMPUTE_SOC2020_SUB_MAJOR_GROUP_KIND = "impute_soc2020_sub_major_group"
UK_APS_OCCUPATION_OUTPUT = "soc2020_sub_major_group"
MAJOR_GROUP_COLUMN = "soc2020_major_group"
STATUS_GROUP = "status_group"
SECTION = "sic_section"
PAY = "employment_income"
PREDICTORS = ("age", "gender", "region", STATUS_GROUP, PAY, SECTION)
CATEGORICAL_PREDICTORS = ("gender", "region", STATUS_GROUP, SECTION)
FRS_CHANNEL = "frs"
#: The stratum names the evidence pools share gaps over.
STRATA = ("pay_band", "region", "age_band", "status_group")

#: The manifest's cleaning operation, mirrored here and checked at stage time.
UK_APS_OCCUPATION_CLEANING: dict[str, Any] = {
    "lowercase_columns": True,
    "adult_age": 16,
    "age_cap": 80,
    "pay_weeks": 52,
    "columns": {
        "age": "AGE",
        "sex": "SEX",
        "region": "GOVTOF",
        "economic_activity": "ILODEFR",
        "status": "INECAC05",
        "main_minor_group": "SC20MMN",
        "last_minor_group": "SC20LMN",
        "weekly_pay": "GRSSWK",
        "person_weight": "PWTA22",
        "income_weight": "PIWTA22",
        "main_section": "INDS07M",
        "last_section": "INDS07L",
    },
    "codes": {
        "in_employment": 1,
        "under_16": 26,
        "employee": 1,
        "sex": {"1": "MALE", "2": "FEMALE"},
        "region": {
            "1": "NORTH_EAST",
            "2": "NORTH_WEST",
            "3": "NORTH_WEST",
            "4": "YORKSHIRE",
            "5": "EAST_MIDLANDS",
            "6": "WEST_MIDLANDS",
            "7": "EAST_OF_ENGLAND",
            "8": "LONDON",
            "9": "SOUTH_EAST",
            "10": "SOUTH_WEST",
            "11": "WALES",
            "12": "SCOTLAND",
            "13": "NORTHERN_IRELAND",
        },
        "status_group": {
            "1": "EMPLOYEE",
            "2": "SELF_EMPLOYED",
            "16": "RETIRED",
            "24": "RETIRED",
        },
        "other_status_group": "OTHER",
        "sections": list(UK_SIC2007_SECTION_DIVISIONS),
    },
}

#: The manifest's imputation operation, mirrored here and checked at stage time.
UK_APS_OCCUPATION_IMPUTATION: dict[str, Any] = {
    "output": UK_APS_OCCUPATION_OUTPUT,
    "within": MAJOR_GROUP_COLUMN,
    "rows": "people_with_a_major_group",
    "identity": "person_source_id_or_spi_recipient_person_id",
    "seed": 0,
    "predictors": list(PREDICTORS),
    "categorical_predictors": list(CATEGORICAL_PREDICTORS),
    "unseen_categories": "ignore",
    "arcsinh_columns": [PAY],
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
    "evidence": {
        "folds": 5,
        "grouping": "person",
        "pay_band_quantiles": [0.5, 0.8, 0.9],
        "age_band_edges": [25, 35, 45, 55, 65],
    },
}


def _plain(value: object) -> object:
    """A declaration as plain dicts and lists, for an exact comparison."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _coded(values: pd.Series, codes: Mapping[str, str]) -> np.ndarray:
    """Each numeric code's label from a string-keyed codebook map; None elsewhere."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    out = np.full(len(numeric), None, dtype=object)
    for code, label in codes.items():
        out[numeric == float(code)] = label
    return out


def clean_aps_occupation_table(
    raw: pd.DataFrame, cleaning: Mapping[str, Any] = UK_APS_OCCUPATION_CLEANING
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The APS donors in the frame's codings, with a receipt of who left and why.

    A donor is an adult aged ``adult_age`` or over (``under_16`` status
    excluded) with a minor group: the main job's when in employment, the last
    job's otherwise. An employee needs a weekly pay of zero or more to stay,
    since the income weight carries the employees asked; annual pay is the
    weekly pay times ``pay_weeks`` where one is recorded and zero elsewhere.
    Region, sex and status follow the declared codes, section codes 1 to 21
    are the SIC 2007 sections in order, and anything else is "" (no section).
    Ages are capped at ``age_cap``, as the frame's are.
    """

    columns = {key: str(value) for key, value in cleaning["columns"].items()}
    table = raw.copy()
    if cleaning["lowercase_columns"]:
        table.columns = [str(column).lower() for column in table.columns]
        columns = {key: value.lower() for key, value in columns.items()}
    missing = sorted(set(columns.values()) - set(table.columns))
    if missing:
        raise ValueError(f"The APS table lacks the declared columns {missing}.")
    codes = cleaning["codes"]

    def number(key: str) -> np.ndarray:
        return pd.to_numeric(table[columns[key]], errors="coerce").to_numpy(
            dtype=np.float64
        )

    age = number("age")
    status = number("status")
    employed = number("economic_activity") == float(codes["in_employment"])
    main_minor = number("main_minor_group")
    last_minor = number("last_minor_group")
    minor = np.where(employed, main_minor, last_minor)
    section_code = np.where(employed, number("main_section"), number("last_section"))
    pay = number("weekly_pay")
    employee = status == float(codes["employee"])
    region = _coded(table[columns["region"]], codes["region"])
    sex = _coded(table[columns["sex"]], codes["sex"])
    sub_major = np.full(len(table), None, dtype=object)
    coded = np.isfinite(minor) & (minor > 0)
    sub_major[coded] = (
        pd.Series(minor[coded] // 10)
        .astype(np.int64)
        .map(UK_SOC2020_SUB_MAJOR_GROUPS)
        .to_numpy(dtype=object)
    )

    adult = (age >= float(cleaning["adult_age"])) & (status != float(codes["under_16"]))
    with_code = adult & coded
    unmapped = with_code & pd.isna(sub_major)
    if unmapped.any():
        bad = sorted({int(code) for code in minor[unmapped]})
        raise ValueError(
            f"The APS carries SOC 2020 minor group codes {bad} outside the "
            "sub-major groups; map them from the release's data dictionary."
        )
    weight = np.where(employee, number("income_weight"), number("person_weight"))
    # Each adult with a code leaves for the first reason that applies.
    no_region = with_code & pd.isna(region)
    no_sex = with_code & ~no_region & pd.isna(sex)
    pay_unknown = with_code & ~no_region & ~no_sex & employee & ~(pay >= 0)
    no_weight = with_code & ~no_region & ~no_sex & ~pay_unknown & ~(weight > 0)
    keep = with_code & ~(no_region | no_sex | pay_unknown | no_weight)

    sections = np.asarray(codes["sections"], dtype=object)
    in_range = np.isfinite(section_code) & (section_code >= 1)
    in_range &= section_code <= len(sections)
    section = np.full(len(table), "", dtype=object)
    section[in_range] = sections[section_code[in_range].astype(np.int64) - 1]
    status_group = _coded(table[columns["status"]], codes["status_group"])
    status_group[pd.isna(status_group)] = codes["other_status_group"]
    donors = pd.DataFrame(
        {
            "age": np.minimum(age, float(cleaning["age_cap"]))[keep],
            "gender": sex[keep],
            "region": region[keep],
            STATUS_GROUP: status_group[keep],
            PAY: np.where(pay >= 0, pay * float(cleaning["pay_weeks"]), 0.0)[keep],
            SECTION: section[keep],
            MAJOR_GROUP_COLUMN: np.asarray(
                [UK_SOC2020_MAJOR_GROUP_OF_SUB_MAJOR_GROUP[g] for g in sub_major[keep]],
                dtype=object,
            ),
            UK_APS_OCCUPATION_OUTPUT: sub_major[keep],
            "weight": weight[keep],
        }
    )
    receipt = {
        "people": rounded_count(len(table)),
        "adults_with_a_minor_group": rounded_count(int(with_code.sum())),
        "donors": rounded_count(int(keep.sum())),
        "dropped": {
            "no_region": rounded_count(int(no_region.sum())),
            "no_sex": rounded_count(int(no_sex.sum())),
            "employee_pay_unknown": rounded_count(int(pay_unknown.sum())),
            "no_weight": rounded_count(int(no_weight.sum())),
        },
    }
    return donors, receipt


def sub_major_model(
    imputation: Mapping[str, Any], *, boosted_max_iter: int | None = None
) -> CategoricalClassifier:
    """The declared classifier.

    ``boosted_max_iter`` caps the boosted learner on a small fixture and turns
    its early stopping off there, as the employment shell's does.
    """
    boosted = dict(imputation["boosted"])
    if boosted_max_iter is not None:
        boosted["max_iter"] = int(boosted_max_iter)
        boosted["early_stopping"] = False
    return CategoricalClassifier(
        boosted_weight=float(imputation["boosted_weight"]),
        boosted=BoostedClassifierSettings(**boosted),
        logit=SplineLogitSettings(**dict(imputation["logit"])),
        categorical_predictors=tuple(imputation["categorical_predictors"]),
        arcsinh_scales={
            column: float(imputation["arcsinh_scale"])
            for column in imputation["arcsinh_columns"]
        },
        seed=int(imputation["seed"]),
        unseen_categories=str(imputation["unseen_categories"]),
    )


def _frame_table(
    person: pd.DataFrame, household: pd.DataFrame, age_cap: float
) -> pd.DataFrame:
    """Every person's predictors, major group, identities and channel."""
    households = household.set_index("household_id")
    by_person = person["person_household_id"].to_numpy()
    return pd.DataFrame(
        {
            "person_id": person["person_id"].to_numpy(),
            "person_source_id": person["person_source_id"].to_numpy(),
            "household_id": by_person,
            "age": np.minimum(person["age"].to_numpy(dtype=np.float64), age_cap),
            "gender": person["gender"].astype(str).to_numpy(),
            "region": households["region"].astype(str).reindex(by_person).to_numpy(),
            STATUS_GROUP: person["employment_status"]
            .astype(str)
            .map(UK_OCCUPATION_STATUS_GROUP)
            .to_numpy(dtype=object),
            PAY: person[PAY].to_numpy(dtype=np.float64),
            SECTION: uk_sic2007_sections(person["sic_industry_division"].to_numpy()),
            MAJOR_GROUP_COLUMN: person[MAJOR_GROUP_COLUMN].astype(str).to_numpy(),
            "spi_income_recipient": person["person_is_spi_income_recipient"]
            .astype(bool)
            .to_numpy(),
            "channel": households["household_support_channel"]
            .astype(str)
            .reindex(by_person)
            .to_numpy(),
        },
        index=person.index,
    )


def _keyed_uniforms(table: pd.DataFrame, seed: int) -> np.ndarray:
    """One uniform per row: the source person's, or a recipient's own.

    A copy shares its FRS twin's ``person_source_id``, so the twin's uniform is
    drawn once and given to every copy; an SPI income recipient keys on its own
    ``person_id`` under the recipients' salt.
    """
    stage = UK_APS_OCCUPATION_STAGE_NAME
    salt = f"{stage}:{UK_APS_OCCUPATION_OUTPUT}"
    recipient = table["spi_income_recipient"].to_numpy()
    uniforms = np.empty(len(table), dtype=np.float64)
    sources = table["person_source_id"].to_numpy()[~recipient]
    if len(sources):
        unique, inverse = np.unique(sources, return_inverse=True)
        keys = integer_record_keys(unique, label=f"{stage} person_source_id")
        uniforms[~recipient] = stable_identity_uniforms(keys, seed=seed, salt=salt)[
            inverse
        ]
    if recipient.any():
        keys = integer_record_keys(
            table["person_id"].to_numpy()[recipient], label=f"{stage} person_id"
        )
        uniforms[recipient] = stable_identity_uniforms(
            keys, seed=seed, salt=f"{salt}:spi_income_recipient"
        )
    return uniforms


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


def _weighted_shares(
    values: np.ndarray, weights: np.ndarray, sources: np.ndarray
) -> dict[str, object]:
    """Each value's weighted share, withheld where one or two households carry it."""
    total = weights.sum()
    out: dict[str, object] = {}
    for value in np.unique(values):
        rows = values == value
        out[str(value)] = small_cell_share(
            float(weights[rows].sum() / total), distinct_count(sources, rows)
        )
    return out


def _held_out_group(
    model: CategoricalClassifier,
    donors: pd.DataFrame,
    folds: np.ndarray,
    evidence: Mapping[str, Any],
) -> tuple[dict[str, dict[str, object]], float]:
    """One major group's out-of-fold scores for the model, its learners and the constant."""
    weights = donors["weight"].to_numpy(dtype=np.float64)
    out = out_of_fold_proba(
        model,
        donors,
        list(PREDICTORS),
        UK_APS_OCCUPATION_OUTPUT,
        weights=weights,
        folds=folds,
    )
    strata = {
        "pay_band": _pay_bands(
            donors[PAY].to_numpy(), weights, list(evidence["pay_band_quantiles"])
        ),
        "region": donors["region"].to_numpy(),
        "age_band": np.digitize(
            donors["age"].to_numpy(), list(evidence["age_band_edges"])
        ),
        "status_group": donors[STATUS_GROUP].to_numpy(),
    }
    labels = donors[UK_APS_OCCUPATION_OUTPUT].to_numpy()
    scores = {
        name: categorical_scores(
            out[name], labels, out["classes"], weights, strata=strata
        )
        for name in ("model", "boosted", "spline_logit", "marginal")
        if name in out
    }
    return scores, float(weights.sum())


def _pooled_evidence(
    groups: Mapping[str, tuple[dict[str, dict[str, object]], float]],
    support: Mapping[str, int],
) -> dict[str, object]:
    """Log losses and share gaps pooled over the major groups, weight-weighted.

    A learner is pooled only where every group fitted it: a group with one
    sub-major group among its donors fits neither learner.
    """
    total = sum(weight for _, weight in groups.values())
    learners = [
        name
        for name in ("model", "boosted", "spline_logit", "marginal")
        if all(name in scores for scores, _ in groups.values())
    ]
    log_loss = {
        name: sum(
            weight * float(scores[name]["log_loss"])
            for scores, weight in groups.values()
        )
        / total
        for name in learners
    }
    gaps: dict[str, float] = {}
    for stratum in STRATA:
        weighted = weight_total = 0.0
        for scores, _ in groups.values():
            for level in scores["model"]["strata"][stratum].values():
                weighted += float(level["weight"]) * float(level["share_gap"])
                weight_total += float(level["weight"])
        gaps[f"{stratum}_share_gap"] = weighted / weight_total
    per_group: dict[str, object] = {}
    for group, (scores, _) in groups.items():
        model = scores["model"]
        per_group[group] = {
            "support": support_band(support[group]),
            "log_loss": {
                name: float(value["log_loss"]) for name, value in scores.items()
            },
            "calibration_slope": model["calibration_slope"],
            "calibration_outcome": model["calibration_outcome"],
        }
    return {
        "status": "measured",
        "log_loss": log_loss,
        **gaps,
        "major_groups": per_group,
    }


def impute_soc2020_sub_major_group(
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
    donors: pd.DataFrame,
    *,
    imputation: Mapping[str, Any] = UK_APS_OCCUPATION_IMPUTATION,
    age_cap: float = float(UK_APS_OCCUPATION_CLEANING["age_cap"]),
    boosted_max_iter: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], tuple[FitWeightRecord, ...]]:
    """Draw every person's sub-major group within their major group; NONE stays NONE."""

    stage = UK_APS_OCCUPATION_STAGE_NAME
    if UK_APS_OCCUPATION_OUTPUT in person.columns:
        raise ValueError(
            f"{stage}: the frame already carries {UK_APS_OCCUPATION_OUTPUT}."
        )
    table = _frame_table(person, household, age_cap)
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
    majors = table[MAJOR_GROUP_COLUMN].to_numpy()
    unknown = sorted(
        set(majors) - {UK_SOC2020_NO_OCCUPATION, *UK_SOC2020_MAJOR_GROUPS.values()}
    )
    if unknown:
        raise ValueError(f"{stage}: major groups {unknown} are not SOC 2020's.")
    coded = majors != UK_SOC2020_NO_OCCUPATION
    unmapped = coded & pd.isna(table[STATUS_GROUP]).to_numpy()
    if unmapped.any():
        statuses = sorted(set(person.loc[unmapped, "employment_status"].astype(str)))
        raise ValueError(
            f"{stage}: people with a major group hold employment statuses "
            f"{statuses} that have no occupation status group."
        )

    seed = int(imputation["seed"])
    evidence = imputation["evidence"]
    fold_count = int(evidence["folds"])
    model = sub_major_model(imputation, boosted_max_iter=boosted_max_iter)
    uniforms = _keyed_uniforms(table, seed)
    drawn = np.full(len(table), UK_SOC2020_NO_OCCUPATION, dtype=object)
    frs_rows = (table["channel"].to_numpy() == FRS_CHANNEL) & (weights > 0)
    records: list[FitWeightRecord] = []
    held_out: dict[str, tuple[dict[str, dict[str, object]], float]] = {}
    not_measured: list[str] = []
    fits: dict[str, object] = {}
    donor_support: dict[str, int] = {}
    frame_gaps: dict[str, object] = {}
    frame_gap_values: list[float] = []
    shares: dict[str, dict[str, object]] = {"donors": {}, "frame": {}}
    unseen = np.zeros(len(table), dtype=bool)
    for code, group in UK_SOC2020_MAJOR_GROUPS.items():
        group_donors = donors.loc[donors[MAJOR_GROUP_COLUMN] == group].reset_index(
            drop=True
        )
        if group_donors.empty:
            raise ValueError(f"{stage}: the APS has no donors in major group {group}.")
        donor_support[group] = len(group_donors)
        donor_weights = group_donors["weight"].to_numpy(dtype=np.float64)
        labels = group_donors[UK_APS_OCCUPATION_OUTPUT].to_numpy()
        if can_deal_folds(labels, fold_count):
            folds = stratified_folds(
                labels,
                fold_count,
                np.random.default_rng(np.random.SeedSequence([seed, code])),
            )
            held_out[group] = _held_out_group(model, group_donors, folds, evidence)
        else:
            not_measured.append(group)
        fitted = model.fit(
            group_donors,
            list(PREDICTORS),
            [UK_APS_OCCUPATION_OUTPUT],
            weights=donor_weights,
        )
        records.append(
            FitWeightRecord(
                f"{stage}:{UK_APS_OCCUPATION_OUTPUT}:{group}", fitted.weight_kind
            )
        )
        fits[group] = fitted.diagnostics
        donor_shares = (
            pd.Series(donor_weights).groupby(labels).sum() / donor_weights.sum()
        )
        shares["donors"][group] = {
            str(name): small_cell_share(float(value), int((labels == name).sum()))
            for name, value in donor_shares.items()
        }
        rows = majors == group
        if not rows.any():
            continue
        group_table = table.loc[rows]
        unseen[rows] = fitted.unseen_category_rows(group_table)
        drawn[rows] = fitted.predict_from_uniforms(group_table, uniforms[rows])[
            UK_APS_OCCUPATION_OUTPUT
        ].to_numpy()
        frame = rows & frs_rows
        if not frame.any():
            continue
        # The FRS rows' expected shares against the donors': how far the
        # FRS's predictor mix within the group moves the draw from the APS.
        households = distinct_count(sources, frame)
        expected = (
            weights[frame]
            @ fitted.predict_proba(table.loc[frame])
            / weights[frame].sum()
        )
        observed = donor_shares.reindex(fitted.classes_, fill_value=0.0).to_numpy()
        gap = float(0.5 * np.abs(expected - observed).sum())
        frame_gaps[group] = {
            "support": support_band(households),
            "share_gap": small_cell_share(gap, households),
        }
        if households >= UK_DISCLOSURE_FLOOR:
            frame_gap_values.append(gap)
        shares["frame"][group] = _weighted_shares(
            drawn[frame], weights[frame], sources[frame]
        )

    if not_measured:
        held_out_evidence: dict[str, object] = {
            "status": "not_measured",
            "reason": "a sub-major group has fewer donors than folds",
            "major_groups_not_measured": not_measured,
            "log_loss": None,
        }
    else:
        held_out_evidence = _pooled_evidence(held_out, donor_support)
    held_out_evidence.update({"folds": fold_count, "grouping": evidence["grouping"]})

    person_out = person.copy()
    person_out[UK_APS_OCCUPATION_OUTPUT] = drawn
    receipt: dict[str, Any] = {
        "donors": {
            group: rounded_count(count) for group, count in donor_support.items()
        },
        "drawn": {
            "people": rounded_count(int(coded.sum())),
            "source_households": rounded_count(distinct_count(sources, coded)),
            "spi_income_recipients": rounded_count(
                int((coded & table["spi_income_recipient"].to_numpy()).sum())
            ),
            "unseen_category_households": rounded_count(
                distinct_count(sources, unseen)
            ),
        },
        "fit": fits,
        "held_out": held_out_evidence,
        "frame_against_donors": {
            "share_gap_max": max(frame_gap_values) if frame_gap_values else None,
            "major_groups": frame_gaps,
        },
        "category_shares": shares,
    }
    return person_out, receipt, tuple(records)


def _assert_stage_parameters(stage) -> None:
    if stage.stage != UK_APS_OCCUPATION_STAGE_NAME:
        raise ValueError(
            f"APS occupation received stage {stage.stage!r}, expected "
            f"{UK_APS_OCCUPATION_STAGE_NAME!r}."
        )
    kinds = [operation.kind for operation in stage.operations]
    expected = [CLEAN_APS_OCCUPATION_TABLE_KIND, IMPUTE_SOC2020_SUB_MAJOR_GROUP_KIND]
    if kinds != expected:
        raise ValueError(
            f"aps_occupation operations drifted: expected {expected}, got {kinds}."
        )
    for operation, declared in zip(
        stage.operations,
        (UK_APS_OCCUPATION_CLEANING, UK_APS_OCCUPATION_IMPUTATION),
        strict=True,
    ):
        actual = _plain(dict(operation.parameters))
        if actual != _plain(declared):
            raise ValueError(
                f"aps_occupation {operation.kind} parameters drifted: expected "
                f"{declared}, got {actual}."
            )
    if tuple(stage.outputs) != (UK_APS_OCCUPATION_OUTPUT,) or tuple(stage.rewrites):
        raise ValueError(
            "aps_occupation must declare exactly the output "
            f"{UK_APS_OCCUPATION_OUTPUT!r} and no rewrites."
        )


def _assert_invariants(person_in: pd.DataFrame, person_out: pd.DataFrame) -> None:
    """Only the new column is added, and it nests in each row's major group."""
    if not person_in.equals(person_out.drop(columns=[UK_APS_OCCUPATION_OUTPUT])):
        raise ValueError("APS occupation modified a column it does not write.")
    majors = person_out[MAJOR_GROUP_COLUMN].astype(str).to_numpy()
    nested = (
        person_out[UK_APS_OCCUPATION_OUTPUT]
        .map(UK_SOC2020_MAJOR_GROUP_OF_SUB_MAJOR_GROUP)
        .to_numpy(dtype=object)
    )
    if not (nested == majors).all():
        raise ValueError(
            "APS occupation drew a sub-major group outside a row's major group."
        )


def _artifact(stage, role: str) -> Mapping[str, Any]:
    for artifact in stage.artifacts:
        if artifact.get("role") == role:
            return artifact
    raise ValueError(f"{stage.stage} declares no {role!r} artifact.")


@dataclass
class UKAPSOccupationStageTransform:
    """Whole-stage callable: draw every person's SOC 2020 sub-major group."""

    stage: Any
    aps_tab_path: str | Path | None = None
    donor: pd.DataFrame | None = None
    boosted_max_iter: int | None = None
    last_receipt: dict[str, Any] | None = field(default=None, init=False)
    last_fit_weight_records: tuple[FitWeightRecord, ...] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        return (
            () if self.last_fit_weight_records is None else self.last_fit_weight_records
        )

    def _raw_donor(self) -> pd.DataFrame:
        if self.donor is not None:
            return self.donor
        if self.aps_tab_path is None:
            raise ValueError(
                f"{UK_APS_OCCUPATION_STAGE_NAME} requires the caller-supplied APS "
                "person tab."
            )
        from microcosm.build.uk_runtime.frs_spine import read_pinned_tab

        return read_pinned_tab(
            Path(self.aps_tab_path).expanduser().resolve(),
            _artifact(self.stage, UK_APS_PERSON_TAB_ROLE),
            columns=list(UK_APS_OCCUPATION_CLEANING["columns"].values()),
        )

    def __call__(self, frame):
        from microcosm.build.uk_runtime.national_frame import (
            uk_household_weight_kind,
            uk_national_frame,
            uk_time_period,
            validate_uk_national_frame,
        )

        _assert_stage_parameters(self.stage)
        validate_uk_national_frame(frame)
        donors, cleaning_receipt = clean_aps_occupation_table(self._raw_donor())
        person = frame.table("person")
        person_out, receipt, records = impute_soc2020_sub_major_group(
            person,
            frame.table("household"),
            np.asarray(frame.weights_for("household").values, dtype=float),
            donors,
            boosted_max_iter=self.boosted_max_iter,
        )
        _assert_invariants(person, person_out)
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
        self.last_fit_weight_records = records
        self.last_receipt = {
            "stage": UK_APS_OCCUPATION_STAGE_NAME,
            "donor_table": cleaning_receipt,
            **receipt,
        }
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (UK_APS_OCCUPATION_OUTPUT,)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_receipt is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_receipt}
