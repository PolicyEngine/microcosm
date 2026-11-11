"""microcosm#1003 receipt: ownership-model comparison on the WAS donor, 5-fold by household.

Uses the stage's own code on the sha-pinned tabs: the committed was_lisa declaration,
clean_was_lisa_donor (credibility rule applied), ownership_design and
fit_ownership_model for the logistic, and the committed balance spec for the
holders-only QRF. The alternatives are the house regime gate (scikit-learn's
HistGradientBoostingClassifier at library defaults, as RegimeGatedQRF fits it),
the same classifier tuned for rare events, and donor weighted rates by age group.
Reports expected (probability-weighted) ownership shares, held-out weighted log
loss, and held-out holder balance quantiles. Aggregates only.

    .venv/bin/python <this>      (from the populace-1003 worktree)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from microcosm.build.country_spec import load_country_spec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.was_lisa import (
    BALANCE_COLUMN,
    CLEAN_WAS_LISA_DONOR_KIND,
    IMPUTE_LIFETIME_ISA_BALANCE_KIND,
    IMPUTE_LIFETIME_ISA_OWNERSHIP_KIND,
    BalanceModelSpec,
    OwnershipModelSpec,
    WASLISAColumns,
    clean_was_lisa_donor,
    fit_ownership_model,
)

warnings.filterwarnings("ignore")
A = Path("/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/acceptance/1003-lisa")
U = Path("/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/was_2006_22")

stage = load_country_spec("uk").sources.stage_map()["was_lisa"]
params = {op.kind: dict(op.parameters) for op in stage.operations}
artifacts = {a["role"]: a for a in stage.artifacts}
columns = WASLISAColumns.from_parameters(params[CLEAN_WAS_LISA_DONOR_KIND])
own_spec = OwnershipModelSpec.from_parameters(
    params[IMPUTE_LIFETIME_ISA_OWNERSHIP_KIND]
)
bal_spec = BalanceModelSpec.from_parameters(params[IMPUTE_LIFETIME_ISA_BALANCE_KIND])
household = read_pinned_tab(
    U / "was_round_8_hhold_eul_may_2025_230525.tab", artifacts["was_qrf_donor"]
)
person = read_pinned_tab(
    U / "was_round_8_person_eul_may_2025_230525.tab",
    artifacts["was_person_tab"],
    columns=columns.person_raw_columns(),
)
donor = clean_was_lisa_donor(person, household, columns=columns).person.reset_index(
    drop=True
)
w = donor["weight"].to_numpy(float)
y = donor["holds"].to_numpy(int)
groups = np.searchsorted(
    np.asarray(own_spec.age_group_lower_bounds[1:], float),
    donor["age_floor"].to_numpy(float),
    side="right",
)
labels = own_spec.group_labels()
RAW = [
    "age_band",
    "is_female",
    "employment_income",
    "household_net_income",
    "num_adults",
    "num_children",
    "is_private_renter",
    "gross_financial_wealth",
    "savings",
    "cash_isa",
    "stocks_and_shares_isa",
]


def tertiles(values):
    order = np.argsort(values, kind="stable")
    cw = np.cumsum(w[order]) / w.sum()
    cuts = np.interp([1 / 3, 2 / 3], cw, values[order])
    return np.digitize(values, cuts)


inc_t = tertiles(donor["household_net_income"].to_numpy(float))
gfw_t = tertiles(donor["gross_financial_wealth"].to_numpy(float))
renter = donor["is_private_renter"].to_numpy(float) == 1.0


def profile(prob):
    out = {"all_adults": float((w * prob).sum() / w.sum())}
    for i, label in enumerate(labels):
        m = groups == i
        out[f"age {label}"] = float((w[m] * prob[m]).sum() / w[m].sum())
    for name, mask in (("private renters", renter), ("other tenures", ~renter)):
        out[name] = float((w[mask] * prob[mask]).sum() / w[mask].sum())
    for i in range(3):
        m = inc_t == i
        out[f"household net income tertile {i + 1}"] = float(
            (w[m] * prob[m]).sum() / w[m].sum()
        )
        m = gfw_t == i
        out[f"gross financial wealth tertile {i + 1}"] = float(
            (w[m] * prob[m]).sum() / w[m].sum()
        )
    # The donor's bottom income tertile holds fewer than 10 holders, so the published cell merges tertiles 1-2.
    m = inc_t <= 1
    out["household net income tertiles 1-2"] = float(
        (w[m] * prob[m]).sum() / w[m].sum()
    )
    return out


def logistic(tr, te):
    fitted, _ = fit_ownership_model(donor.loc[tr].reset_index(drop=True), own_spec)
    return fitted.probabilities(donor.loc[te])


def hgb_default(tr, te):
    c = HistGradientBoostingClassifier(random_state=0).fit(
        donor.loc[tr, RAW], y[tr], sample_weight=w[tr]
    )
    return c.predict_proba(donor.loc[te, RAW])[:, 1]


def hgb_tuned(tr, te):
    c = HistGradientBoostingClassifier(
        random_state=0,
        early_stopping=False,
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=8,
        min_samples_leaf=200,
        l2_regularization=1.0,
    ).fit(donor.loc[tr, RAW], y[tr], sample_weight=w[tr])
    return c.predict_proba(donor.loc[te, RAW])[:, 1]


def age_rates(tr, te):
    rate = {
        g: (w[tr][groups[tr] == g] * y[tr][groups[tr] == g]).sum()
        / w[tr][groups[tr] == g].sum()
        for g in np.unique(groups[tr])
    }
    return np.asarray([rate.get(g, 0.0) for g in groups[te]])


households = donor["household_key"].unique()
rng = np.random.default_rng(7)
rng.shuffle(households)
folds = np.array_split(households, 5)
result = {
    "donor": profile(y.astype(float)),
    "donor_adults": int(len(donor)),
    "donor_holders": int(y.sum()),
    "folds": 5,
    "fold_seed": 7,
}
for name, fn in (
    ("stage logistic", logistic),
    ("house gate (boosting, defaults)", hgb_default),
    ("boosting tuned for rare events", hgb_tuned),
    ("age-group rates", age_rates),
):
    prob = np.zeros(len(donor))
    for fold in folds:
        te = donor["household_key"].isin(fold).to_numpy()
        prob[te] = fn(~te, te)
    result[name] = profile(prob)
    clipped = np.clip(prob, 1e-9, 1 - 1e-9)
    result[name]["held_out_weighted_log_loss_x1000"] = float(
        -np.average(y * np.log(clipped) + (1 - y) * np.log(1 - clipped), weights=w)
        * 1000
    )

# Balance: the committed holders-only QRF on held-out credible holders.
from microcosm.fit.qrf import RegimeGatedQRF  # noqa: E402

value = donor[BALANCE_COLUMN].to_numpy(float)
credible = (y == 1) & np.isfinite(value) & (value > 0)
drawn = np.full(len(donor), np.nan)
ids = (
    (donor["household_key"] * 100 + donor["person_number"]).astype(np.int64).to_numpy()
)
for fold in folds:
    te = donor["household_key"].isin(fold).to_numpy()
    tr = ~te & credible
    fitted = RegimeGatedQRF(n_estimators=bal_spec.n_estimators, seed=bal_spec.seed).fit(
        donor.loc[tr, [*bal_spec.predictors, BALANCE_COLUMN, "weight"]]
        .astype(float)
        .reset_index(drop=True),
        list(bal_spec.predictors),
        [BALANCE_COLUMN],
        weights="weight",
    )
    hold = te & credible
    if hold.any():
        q = stable_identity_uniforms(ids[hold], seed=bal_spec.seed, salt=bal_spec.salt)
        drawn[hold] = fitted.predict_positive_from_uniforms(
            donor.loc[hold, list(bal_spec.predictors)]
            .astype(float)
            .reset_index(drop=True),
            quantiles={BALANCE_COLUMN: q},
        )[BALANCE_COLUMN].to_numpy(float)


def wq(v, ww):
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(ww[o])
    return {
        f"p{int(q * 100)}": float(
            v[o][min(int(np.searchsorted(cw, q * cw[-1])), len(v) - 1)]
        )
        for q in (0.1, 0.25, 0.5, 0.75, 0.9)
    }


result["balance"] = {
    "credible_holders": int(credible.sum()),
    "donor_quantiles": wq(value[credible], w[credible]),
    "donor_weighted_mean": float(np.average(value[credible], weights=w[credible])),
    "held_out_draw_quantiles": wq(drawn[credible], w[credible]),
    "held_out_draw_weighted_mean": float(
        np.average(drawn[credible], weights=w[credible])
    ),
}
(A / "receipts").mkdir(exist_ok=True)
json.dump(result, open(A / "receipts" / "model-comparison.json", "w"), indent=1)
table = (
    pd.DataFrame(
        {k: v for k, v in result.items() if isinstance(v, dict) and k != "balance"}
    )
    .drop(index="held_out_weighted_log_loss_x1000", errors="ignore")
    .mul(100)
    .round(2)
)
print(table.to_string())
print(
    "log loss x1000:",
    {
        k: round(v["held_out_weighted_log_loss_x1000"], 2)
        for k, v in result.items()
        if isinstance(v, dict) and "held_out_weighted_log_loss_x1000" in v
    },
)
print(json.dumps(result["balance"], indent=1))
