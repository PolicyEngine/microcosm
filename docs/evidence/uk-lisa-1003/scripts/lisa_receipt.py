"""microcosm#1003 receipts: the was_lisa stage on a licensed candidate spine.

Reads the build sidecar's was_lisa stage evidence and the candidate H5, recomputes
the realised ownership and balances from the H5 as an independent check of the
evidence, and runs the release-side column checks (degenerate, support bounds,
nonnegative, export allow-list) on the three new cells. Aggregates only; run
from the populace-1003 worktree's venv:

    .venv/bin/python <this> spine-lisa
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.gates import support_gate
from microcosm.build.uk_runtime.terminal_gates import (
    UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
    _degenerate_kind,
)

A = Path("/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/acceptance/1003-lisa")
TREE = Path("/Users/mariajuaristi/Desktop/PolicyEngine/repos/populace-1003")
NEW = {
    "person": ("has_lifetime_isa", "lifetime_isa_balance"),
    "household": ("household_lifetime_isa_balance",),
}
GROUPS = (
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 200, "55+"),
)


def wq(values, weights, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values, kind="stable")
    cw = np.cumsum(weights[order])
    return {
        f"p{int(q * 100)}": float(
            values[order][min(int(np.searchsorted(cw, q * cw[-1])), len(values) - 1)]
        )
        for q in qs
    }


def share(mask, weights):
    return float(weights[mask].sum() / weights.sum()) if weights.sum() > 0 else None


def main(name: str) -> None:
    build = json.load(open(A / name / f"{name}.build.json"))
    evidence = build["stage_evidence"]["was_lisa"]
    store = pd.HDFStore(A / name / f"{name}.h5", "r")
    keys = store.keys()
    person = (
        store["/person"]
        if "/person" in keys
        else store[[k for k in keys if k.endswith("person")][0]]
    )
    household = (
        store["/household"]
        if "/household" in keys
        else store[[k for k in keys if k.endswith("household")][0]]
    )
    store.close()
    weight_column = "household_weight"
    hw = pd.Series(
        household[weight_column].to_numpy(float),
        index=household["household_id"].to_numpy(),
    )
    pw = hw.reindex(person["person_household_id"].to_numpy()).to_numpy(float)
    holds = person["has_lifetime_isa"].to_numpy(bool)
    balance = person["lifetime_isa_balance"].to_numpy(float)
    age = person["age"].to_numpy(float)
    adult = age >= 18
    female = person["gender"].astype(str).str.upper().str.contains("FEMALE").to_numpy()
    tenure = (
        household.set_index("household_id")["tenure_type"]
        .astype(str)
        .reindex(person["person_household_id"].to_numpy())
        .to_numpy()
    )
    renter = np.char.find(tenure.astype(str), "RENT_PRIVATELY") >= 0
    channel = (
        household.set_index("household_id")["household_support_channel"]
        .astype(str)
        .reindex(person["person_household_id"].to_numpy())
        .to_numpy()
        if "household_support_channel" in household
        else np.array(["frs"] * len(person))
    )
    recomputed = {
        "persons": int(len(person)),
        "adults": int(adult.sum()),
        "holders": int(holds.sum()),
        "weighted_holders": float(pw[holds].sum()),
        "weighted_adults": float(pw[adult].sum()),
        "adult_ownership_share": share(holds[adult], pw[adult]),
        "by_age_group": {
            label: share(
                holds[adult & (age >= lo) & (age <= hi)],
                pw[adult & (age >= lo) & (age <= hi)],
            )
            for lo, hi, label in GROUPS
        },
        "by_sex": {
            "female": share(holds[adult & female], pw[adult & female]),
            "male": share(holds[adult & ~female], pw[adult & ~female]),
        },
        "by_private_renting": {
            "private_renter": share(holds[adult & renter], pw[adult & renter]),
            "other": share(holds[adult & ~renter], pw[adult & ~renter]),
        },
        "by_channel": {
            str(c): share(holds[adult & (channel == c)], pw[adult & (channel == c)])
            for c in sorted(set(channel))
        },
        "holders_on_spi_rows_weighted_share": share(channel[holds] != "frs", pw[holds])
        if holds.any()
        else None,
        "under_18_holders": int((holds & ~adult).sum()),
        "owner_balance_quantiles": wq(balance[holds], pw[holds])
        if holds.any()
        else None,
        "owner_weighted_mean_balance": float(
            np.average(balance[holds], weights=pw[holds])
        )
        if holds.any()
        else None,
        "weighted_total_gbp_bn": float((balance * pw).sum() / 1e9),
        "flag_iff_positive_balance": bool(np.array_equal(holds, balance > 0)),
        "min_positive_balance": float(balance[holds].min()) if holds.any() else None,
        "max_balance": float(balance.max()),
    }
    totals = (
        pd.Series(balance, index=person["person_household_id"].to_numpy())
        .groupby(level=0)
        .sum()
        .reindex(household["household_id"].to_numpy())
        .fillna(0.0)
        .to_numpy()
    )
    recomputed["household_total_identity_max_abs_gbp"] = float(
        np.abs(
            totals - household["household_lifetime_isa_balance"].to_numpy(float)
        ).max()
    )
    recomputed["households_over_gross_financial_wealth"] = int(
        (
            household["household_lifetime_isa_balance"].to_numpy(float)
            > household["gross_financial_wealth"].to_numpy(float) + 1e-6
        ).sum()
    )
    owner_households = (
        pd.Series(holds.astype(int), index=person["person_household_id"].to_numpy())
        .groupby(level=0)
        .sum()
    )
    owner_households = owner_households[owner_households > 0]
    hwo = hw.reindex(owner_households.index).to_numpy(float)
    recomputed["co_holding_share_of_owner_households"] = (
        float(hwo[owner_households.to_numpy() > 1].sum() / hwo.sum())
        if len(hwo)
        else None
    )
    bounds = json.load(
        open(
            TREE
            / "packages/microcosm-build/src/microcosm/build/uk/was_lisa_support_bounds.json"
        )
    )["bounds"]
    release_checks = {
        "degenerate": {
            f"{entity}.{column}": (
                _degenerate_kind((person if entity == "person" else household)[column])
                or (None,)
            )[0]
            for entity, columns in NEW.items()
            for column in columns
        },
        "support_gate": support_gate(
            {"lifetime_isa_balance": balance}, {k: tuple(v) for k, v in bounds.items()}
        ).passed,
        "nonnegative": bool(
            (balance >= 0).all()
            and (household["household_lifetime_isa_balance"].to_numpy(float) >= 0).all()
        ),
        "export_allow_listed": all(
            f"{entity}.{column}" in UK_ALLOWED_EXTRA_EXPORT_COLUMNS
            for entity, columns in NEW.items()
            for column in columns
        ),
        "dtypes": {
            f"{entity}.{column}": str(
                (person if entity == "person" else household)[column].dtype
            )
            for entity, columns in NEW.items()
            for column in columns
        },
    }
    realised = evidence["realised"]
    cross_check = {
        "adult_ownership_share": (
            realised["recipient"]["all_adults"],
            recomputed["adult_ownership_share"],
        ),
        "owner_weighted_mean_balance": (
            realised["owner_weighted_mean_balance"],
            recomputed["owner_weighted_mean_balance"],
        ),
        "weighted_total_gbp": (
            realised["weighted_total_gbp"],
            recomputed["weighted_total_gbp_bn"] * 1e9,
        ),
    }
    out = {
        "name": name,
        "evidence": evidence,
        "recomputed": recomputed,
        "release_checks": release_checks,
        "cross_check_evidence_vs_h5": cross_check,
    }
    (A / "receipts").mkdir(exist_ok=True)
    json.dump(
        out, open(A / "receipts" / f"receipt-{name}.json", "w"), indent=1, default=str
    )
    print(
        json.dumps(
            {
                "recomputed": recomputed,
                "release_checks": release_checks,
                "cross_check": cross_check,
            },
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "spine-lisa")
