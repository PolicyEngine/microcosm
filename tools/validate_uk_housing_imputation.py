"""Holdout check of the SPI housing-shell imputation against donor matching.

Development diagnostic for ``uk_runtime/spi_housing_shell.py``. On a built UK
spine, the FRS base households are split by identity-keyed uniforms into a
training share and a held-out share. The held-out households' housing is then
filled two ways from the same predictors and compared with what they report:

* ``imputation``: the stage's model (classifier steps + chained regime-gated
  QRF + learned structural zeros);
* ``matching``: whole-record nearest-k donor matching within region and
  single-adult status, composition ladder on ``ons_household_type``, k nearest
  in log income, household-weighted draw.

The JSON output is aggregate only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.spi_housing_shell import (
    HOUSING_AMOUNT_COLUMNS,
    HOUSING_CATEGORICAL_COLUMNS,
    RENTED_TENURES,
    apply_structural_rules,
    fit_housing_model,
    household_housing_predictors,
    household_housing_targets,
)

INCOME_BANDS = (-np.inf, 20e3, 50e3, 100e3, 200e3, np.inf)
INCOME_LABELS = ("<20k", "20-50k", "50-100k", "100-200k", "200k+")
INCOME_COLUMNS = (
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "investment_income",
    "state_pension_reported",
)


def _total_income(predictors: pd.DataFrame) -> pd.Series:
    return predictors[list(INCOME_COLUMNS)].sum(axis=1)


def _weighted_quantiles(values, weights, qs):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return [float(values[order][np.searchsorted(cumulative, q)]) for q in qs]


def _weighted_ks(a, wa, b, wb) -> float:
    grid = np.unique(np.concatenate([a, b]))

    def cdf(x, w):
        order = np.argsort(x, kind="stable")
        cw = np.cumsum(w[order]) / w.sum()
        return np.interp(grid, x[order], cw, left=0.0, right=1.0)

    return float(np.max(np.abs(cdf(a, wa) - cdf(b, wb))))


def _tvd_by_group(observed, filled, groups, weights) -> float:
    """Weighted mean over groups of the total variation distance of shares."""

    total, mass = 0.0, 0.0
    for group in np.unique(groups):
        m = groups == group
        w = weights[m]
        if w.sum() <= 0:
            continue
        o = pd.Series(w, index=observed[m]).groupby(level=0).sum() / w.sum()
        f = pd.Series(w, index=filled[m]).groupby(level=0).sum() / w.sum()
        tvd = 0.5 * float(o.subtract(f, fill_value=0.0).abs().sum())
        total += tvd * w.sum()
        mass += w.sum()
    return total / mass


def match_housing(train_pred, train_tgt, train_w, test_pred, *, k=10, seed=0):
    """Whole-record nearest-k matching (the design the imputation replaced)."""

    income_train = np.log1p(np.maximum(_total_income(train_pred).to_numpy(), 0))
    income_test = np.log1p(np.maximum(_total_income(test_pred).to_numpy(), 0))
    keys = ["region", "single_adult"]
    uniforms = stable_identity_uniforms(
        test_pred.index.to_numpy(), seed=seed, salt="match"
    )
    chosen = []
    for position, (_, row) in enumerate(test_pred.iterrows()):
        pool = np.ones(len(train_pred), dtype=bool)
        for key in keys:
            pool &= train_pred[key].to_numpy() == row[key]
        narrowed = pool & (
            train_pred["ons_household_type"].to_numpy() == row["ons_household_type"]
        )
        if narrowed.sum() >= 20:
            pool = narrowed
        idx = np.flatnonzero(pool)
        near = idx[
            np.argsort(
                np.abs(income_train[idx] - income_test[position]), kind="stable"
            )[:k]
        ]
        w = train_w[near]
        cdf = np.cumsum(w) / w.sum()
        chosen.append(near[np.searchsorted(cdf, uniforms[position])])
    filled = train_tgt.iloc[chosen].copy()
    filled.index = test_pred.index
    return filled, np.asarray(chosen)


def forest_joint_draw(
    train_pred,
    train_tgt,
    train_w,
    test_pred,
    *,
    seed=0,
    n_estimators=100,
    min_samples_leaf=5,
):
    """Draw one whole training record per recipient from the forest's conditional weights.

    A multi-output random forest is grown on the housing bundle (one-hot
    categories, log amounts); a recipient's weight on training household j is
    the mean over trees of 1/|leaf| for the leaf they share, times j's survey
    weight. The drawn household's full bundle is copied, so the joint is a real
    record and the draw follows the forest's conditional distribution.
    """

    from sklearn.ensemble import RandomForestRegressor

    def design(pred):
        cats = pd.get_dummies(
            pred[["region", "ons_household_type"]].astype(str), dtype=float
        )
        num = pred.drop(columns=["region", "ons_household_type"]).astype(float)
        return pd.concat([num, cats], axis=1)

    x_train = design(train_pred)
    x_test = design(test_pred).reindex(columns=x_train.columns, fill_value=0.0)
    y = pd.concat(
        [
            pd.get_dummies(
                train_tgt[list(HOUSING_CATEGORICAL_COLUMNS)].astype(str), dtype=float
            ),
            np.log1p(
                train_tgt[
                    list(HOUSING_AMOUNT_COLUMNS)
                    + ["housing_benefit_head_unit", "council_tax_benefit_hrp"]
                ].clip(lower=0)
            ),
        ],
        axis=1,
    )
    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    ).fit(x_train.to_numpy(), y.to_numpy(), sample_weight=train_w)
    leaves_train = forest.apply(x_train.to_numpy())
    leaves_test = forest.apply(x_test.to_numpy())
    uniforms = stable_identity_uniforms(
        test_pred.index.to_numpy(), seed=seed, salt="forest-joint"
    )
    n_train = len(train_pred)
    chosen = np.empty(len(test_pred), dtype=np.int64)
    for i in range(len(test_pred)):
        weight = np.zeros(n_train)
        for t in range(leaves_train.shape[1]):
            members = np.flatnonzero(leaves_train[:, t] == leaves_test[i, t])
            weight[members] += 1.0 / len(members)
        weight *= train_w
        cdf = np.cumsum(weight) / weight.sum()
        chosen[i] = np.searchsorted(cdf, uniforms[i])
    filled = train_tgt.iloc[chosen].copy()
    filled.index = test_pred.index
    return filled, chosen


def noise_floor(observed, predictors, *, seed=0):
    """The held-out records shuffled within income band x composition x region cells."""

    band = pd.cut(_total_income(predictors), INCOME_BANDS, labels=INCOME_LABELS).astype(
        str
    )
    cell = (
        band
        + "|"
        + predictors["ons_household_type"].astype(str)
        + "|"
        + predictors["region"].astype(str)
    )
    rng = np.random.default_rng(seed)
    order = np.arange(len(observed))
    for _, positions in (
        pd.Series(np.arange(len(observed))).groupby(cell.to_numpy()).indices.items()
    ):
        order[positions] = rng.permutation(positions)
    shuffled = observed.iloc[order].copy()
    shuffled.index = observed.index
    return shuffled


def evaluate(observed, filled, predictors, weights) -> dict:
    band = (
        pd.cut(_total_income(predictors), INCOME_BANDS, labels=INCOME_LABELS)
        .astype(str)
        .to_numpy()
    )
    composition = predictors["ons_household_type"].astype(str).to_numpy()
    out: dict = {"categorical_tvd": {}}
    for column in HOUSING_CATEGORICAL_COLUMNS:
        o = observed[column].astype(str).to_numpy()
        f = filled[column].astype(str).to_numpy()
        out["categorical_tvd"][column] = {
            "by_income_band": _tvd_by_group(o, f, band, weights),
            "by_composition": _tvd_by_group(o, f, composition, weights),
        }
    conditional = {
        "rent|rented": (
            "rent",
            observed["tenure_type"].isin(RENTED_TENURES),
            filled["tenure_type"].isin(RENTED_TENURES),
        ),
        "mortgage_interest|mortgaged": (
            "mortgage_interest_repayment",
            observed["tenure_type"] == "OWNED_WITH_MORTGAGE",
            filled["tenure_type"] == "OWNED_WITH_MORTGAGE",
        ),
        "council_tax|all": (
            "council_tax",
            observed["council_tax"] >= 0,
            filled["council_tax"] >= 0,
        ),
        "housing_benefit|all": (
            "housing_benefit_head_unit",
            observed["council_tax"] >= 0,
            filled["council_tax"] >= 0,
        ),
    }
    out["conditional_distributions"] = {}
    for name, (column, mo, mf) in conditional.items():
        a = observed.loc[mo, column].to_numpy(dtype=float)
        b = filled.loc[mf, column].to_numpy(dtype=float)
        wa, wb = weights[mo.to_numpy()], weights[mf.to_numpy()]
        out["conditional_distributions"][name] = {
            "observed_p25_p50_p75_p90": _weighted_quantiles(
                a, wa, [0.25, 0.5, 0.75, 0.9]
            ),
            "filled_p25_p50_p75_p90": _weighted_quantiles(
                b, wb, [0.25, 0.5, 0.75, 0.9]
            ),
            "weighted_ks": _weighted_ks(a, wa, b, wb),
        }
    income = _total_income(predictors).to_numpy()
    size = (predictors["num_adults"] + predictors["num_children"]).to_numpy()
    renters_o = observed["tenure_type"].isin(RENTED_TENURES).to_numpy()
    renters_f = filled["tenure_type"].isin(RENTED_TENURES).to_numpy()
    out["rank_correlations"] = {
        "income~rent (renters)": {
            "observed": float(
                spearmanr(
                    income[renters_o], observed["rent"].to_numpy()[renters_o]
                ).correlation
            ),
            "filled": float(
                spearmanr(
                    income[renters_f], filled["rent"].to_numpy()[renters_f]
                ).correlation
            ),
        },
        "income~council_tax": {
            "observed": float(spearmanr(income, observed["council_tax"]).correlation),
            "filled": float(spearmanr(income, filled["council_tax"]).correlation),
        },
        "bedrooms~household_size": {
            "observed": float(
                spearmanr(size, observed["num_bedrooms"].astype(float)).correlation
            ),
            "filled": float(
                spearmanr(size, filled["num_bedrooms"].astype(float)).correlation
            ),
        },
        "income~detached": {
            "observed": float(
                spearmanr(
                    income, observed["accommodation_type"] == "HOUSE_DETACHED"
                ).correlation
            ),
            "filled": float(
                spearmanr(
                    income, filled["accommodation_type"] == "HOUSE_DETACHED"
                ).correlation
            ),
        },
    }
    rented_f = filled["tenure_type"].isin(RENTED_TENURES)
    out["coherence_violations"] = {
        "mortgage>0 & not mortgaged": int(
            (
                (
                    filled["mortgage_interest_repayment"]
                    + filled["mortgage_capital_repayment"]
                    > 0
                )
                & (filled["tenure_type"] != "OWNED_WITH_MORTGAGE")
            ).sum()
        ),
        "housing_benefit>0 & not rented": int(
            ((filled["housing_benefit_head_unit"] > 0) & ~rented_f).sum()
        ),
        "rent>0 & not rented": int(((filled["rent"] > 0) & ~rented_f).sum()),
        "rent>0 & not rented (observed)": int(
            (
                (observed["rent"] > 0) & ~observed["tenure_type"].isin(RENTED_TENURES)
            ).sum()
        ),
    }

    def shares(frame):
        w = weights / weights.sum()
        return {
            "owned": float(
                w[
                    frame["tenure_type"].astype(str).str.startswith("OWNED").to_numpy()
                ].sum()
            ),
            "social_rent": float(
                w[
                    frame["tenure_type"]
                    .isin(["RENT_FROM_COUNCIL", "RENT_FROM_HA"])
                    .to_numpy()
                ].sum()
            ),
            "detached": float(
                w[(frame["accommodation_type"] == "HOUSE_DETACHED").to_numpy()].sum()
            ),
            "band_F_plus": float(
                w[frame["council_tax_band"].isin(["F", "G", "H", "I"]).to_numpy()].sum()
            ),
            "bedrooms_4_plus": float(
                w[frame["num_bedrooms"].astype(float).to_numpy() >= 4].sum()
            ),
            "council_tax_p50": _weighted_quantiles(
                frame["council_tax"].to_numpy(dtype=float), weights, [0.5]
            )[0],
        }

    out["shares"] = {"observed": shares(observed), "filled": shares(filled)}
    top = income >= np.quantile(income, 0.9)
    out["top_decile_variety"] = {
        "distinct_rent": int(filled.loc[top, "rent"].round(0).nunique()),
        "distinct_council_tax": int(filled.loc[top, "council_tax"].round(0).nunique()),
        "distinct_bundles": int(
            filled.loc[top, list(HOUSING_CATEGORICAL_COLUMNS)]
            .astype(str)
            .agg("|".join, axis=1)
            .nunique()
        ),
        "households": int(top.sum()),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spine-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-share", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--holdout",
        choices=("random", "top"),
        default="random",
        help="random: identity-keyed share; top: households above the income quantile "
        "--top-quantile, trained on the rest (the extrapolation SPI copies need).",
    )
    parser.add_argument("--top-quantile", type=float, default=0.95)
    args = parser.parse_args()

    with pd.HDFStore(args.spine_h5, mode="r") as store:
        person, household = store["person"], store["household"]
    base = household[
        (household["household_support_channel"] == "frs")
        & ~household["household_is_capital_gains_clone"].astype(bool)
        & ~household["household_is_cgt_band_donor"].astype(bool)
    ]
    person = person[person["person_household_id"].isin(base["household_id"])]
    predictors = household_housing_predictors(person, base)
    targets = household_housing_targets(person, base)
    weights = (
        base.set_index("household_id")["household_weight"]
        .reindex(predictors.index)
        .to_numpy(dtype=float)
    )
    if args.holdout == "top":
        income = _total_income(predictors).to_numpy()
        held = income >= np.quantile(income, args.top_quantile)
    else:
        held = (
            stable_identity_uniforms(
                predictors.index.to_numpy(), seed=args.seed, salt="housing-holdout"
            )
            < args.holdout_share
        )
    train, test = ~held, held

    model = fit_housing_model(
        predictors[train],
        targets[train],
        weights[train],
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    raw = model.draw(predictors[test])
    imputed, fired = apply_structural_rules(
        raw, predictors.loc[test, "region"], model.structural_zeros
    )
    matched, chosen = match_housing(
        predictors[train],
        targets[train],
        weights[train],
        predictors[test],
        seed=args.seed,
    )

    forest_filled, forest_chosen = forest_joint_draw(
        predictors[train],
        targets[train],
        weights[train],
        predictors[test],
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    observed = targets[test]
    result = {
        "spine_h5": str(args.spine_h5),
        "training_households": int(train.sum()),
        "held_out_households": int(test.sum()),
        "imputation": evaluate(observed, imputed, predictors[test], weights[test]),
        "imputation_rule_firings": fired,
        "imputation_raw_violations_before_rules": evaluate(
            observed, raw, predictors[test], weights[test]
        )["coherence_violations"],
        "matching": evaluate(observed, matched, predictors[test], weights[test]),
        "forest_joint": evaluate(
            observed, forest_filled, predictors[test], weights[test]
        ),
        "forest_joint_donor_reuse": {
            "distinct_donors": int(len(np.unique(forest_chosen))),
            "max_uses_of_one_donor": int(np.bincount(forest_chosen).max()),
        },
        "noise_floor_shuffled_within_cells": evaluate(
            observed,
            noise_floor(observed, predictors[test], seed=args.seed),
            predictors[test],
            weights[test],
        ),
        "matching_donor_reuse": {
            "distinct_donors": int(len(np.unique(chosen))),
            "max_uses_of_one_donor": int(np.bincount(chosen).max()),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
