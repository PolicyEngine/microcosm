"""PUF-channel wage audit on a US base frame or release H5 (microcosm#1191).

Usage: python puf_wage_audit.py <frame.h5> <out.json> [--same-rows-as <other.h5>]

Extends raw_h5_checks.py. Reads stored columns only (no engine). Reports, by
support channel and clone index, weighted wages and desired traditional 401(k)
deferrals; for source persons present on both channels, how the PUF-channel
wage relates to the same person's ASEC wage (person and tax-unit grain); the
weighted wage distribution on each channel; and the pre-tax amount the engine
would subtract, rebuilt from stored columns with policyengine-us 2.2.1's 2024
elective-deferral limit. With --same-rows-as it also checks, person by person,
which wage and retirement-contribution values differ in a second file.
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

WAGE = "employment_income_before_lsr"
DEFERRAL = "traditional_401k_contributions_desired"
ROTH = "roth_401k_contributions_desired"
CHANNEL = "person_support_channel"
CLONE = "person_support_clone_index"
KEYS = ["source_year", "source_person_id"]
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
LIMIT_401K_2024 = 23_000.0
CATCH_UP_401K_2024 = 7_500.0
CATCH_UP_AGE = 50
CONTRIBUTIONS = (
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "traditional_ira_contributions_desired",
    "roth_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    with pd.HDFStore(path, mode="r") as store:
        person = store["person"]
        household = store["household"]
    weight = (
        household.set_index("household_id")["household_weight"]
        .reindex(person["person_household_id"].to_numpy())
        .to_numpy(dtype=np.float64)
    )
    return person, weight


def weighted_quantiles(values, weights, quantiles=QUANTILES):
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return {f"p{q * 100:g}": float(np.interp(q, cumulative, values)) for q in quantiles}


def ratio_summary(ratio):
    points = (5, 25, 50, 75, 95)
    out = {f"p{p}": float(np.percentile(ratio, p)) for p in points}
    out["within_1pct"] = float(np.mean(np.abs(ratio - 1.0) <= 0.01))
    out["within_10pct"] = float(np.mean(np.abs(ratio - 1.0) <= 0.10))
    return out


def spearman(a, b):
    return float(
        np.corrcoef(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())[
            0, 1
        ]
    )


def engine_pre_tax_401k(deferral, roth, age):
    """Traditional 401(k) amount policyengine-us 2.2.1 subtracts in 2024.

    traditional_401k_contributions = desired * min(limit / max(total, 1), 1),
    limit = 23,000 plus 7,500 from age 50 (elective_deferral_limit,
    k401_catch_up_limit, parameters gov.irs.gross_income.retirement_contributions).
    403(b) desired leaves, pre-tax health premiums and HSA payroll
    contributions are not stored in these files, so the engine reads them as 0.
    """

    limit = LIMIT_401K_2024 + np.where(age >= CATCH_UP_AGE, CATCH_UP_401K_2024, 0.0)
    scale = np.minimum(limit / np.maximum(deferral + roth, 1.0), 1.0)
    return deferral * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5")
    parser.add_argument("out")
    parser.add_argument("--same-rows-as")
    args = parser.parse_args()

    person, weight = load(args.h5)
    out = {
        "h5": os.path.basename(args.h5),
        "h5_sha256": sha256(args.h5),
        "units": "USD weighted sums unless a key says count or share",
    }
    wage = person[WAGE].to_numpy(dtype=np.float64)
    deferral = person[DEFERRAL].to_numpy(dtype=np.float64)
    roth = (
        person[ROTH].to_numpy(dtype=np.float64)
        if ROTH in person
        else np.zeros(len(person))
    )
    channel = person[CHANNEL].astype(str).to_numpy()
    clone = person[CLONE].to_numpy(dtype=np.int64)
    age = person["age"].to_numpy(dtype=np.float64)
    tax_unit = person["person_tax_unit_id"].to_numpy()
    pre_tax = engine_pre_tax_401k(deferral, roth, age)
    taxed = np.maximum(wage - pre_tax, 0.0)

    out["totals"] = {
        "rows_count": int(len(person)),
        "weighted_persons_count": float(weight.sum()),
        "wages": float(np.sum(weight * wage)),
        "desired_traditional_401k": float(np.sum(weight * deferral)),
        "negative_wage_rows_count": int((wage < 0).sum()),
        "negative_deferral_rows_count": int((deferral < 0).sum()),
        "engine_pre_tax_401k": float(np.sum(weight * pre_tax)),
        "engine_taxed_wage": float(np.sum(weight * taxed)),
        "gross_minus_taxed": float(np.sum(weight * (wage - taxed))),
        "lost_to_zero_floor": float(np.sum(weight * np.maximum(pre_tax - wage, 0.0))),
    }

    # 1. By support channel and clone index.
    groups = {}
    for name in np.unique(channel):
        for index in np.unique(clone[channel == name]):
            groups[f"{name}/clone{index}"] = (channel == name) & (clone == index)
        groups[f"{name}/all"] = channel == name
    by_group = {}
    for label, mask in groups.items():
        w, x, d = weight[mask], wage[mask], deferral[mask]
        earner, contributor = x > 0, d > 0
        by_group[label] = {
            "rows_count": int(mask.sum()),
            "weighted_persons_count": float(w.sum()),
            "wage_earners_count": float(w[earner].sum()),
            "wages": float(np.sum(w * x)),
            "desired_traditional_401k": float(np.sum(w * d)),
            "deferral_share_of_wages": float(np.sum(w * d) / np.sum(w * x)),
            "deferral_capped_at_wage": float(np.sum(w * np.minimum(d, x))),
            "engine_pre_tax_401k": float(np.sum(w * pre_tax[mask])),
            "engine_taxed_wage": float(np.sum(w * taxed[mask])),
            "gross_minus_taxed": float(np.sum(w * (x - taxed[mask]))),
            "gross_minus_taxed_share_of_wages": float(
                np.sum(w * (x - taxed[mask])) / np.sum(w * x)
            ),
            "desired_roth_401k": float(np.sum(w * roth[mask])),
            "contributors_count": float(w[contributor].sum()),
            "contributors_without_wages_count": float(w[contributor & ~earner].sum()),
            "contributor_share_of_earners": float(
                w[contributor & earner].sum() / w[earner].sum()
            ),
            "deferral_share_of_contributor_wages": float(
                np.sum((w * d)[contributor]) / np.sum((w * x)[contributor])
            ),
            "mean_wage_of_earners": float(np.sum((w * x)[earner]) / w[earner].sum()),
            "unweighted_mean_wage": float(x.mean()),
            "wage_earners_under_15_rows_count": int((earner & (age[mask] < 15)).sum()),
        }
    out["by_channel_and_clone"] = by_group

    # Deferral share of wages by wage band, each channel (does the PUF-half
    # deferral follow the survey relationship at the wage it was given?).
    edges = [0, 25_000, 50_000, 100_000, 200_000, 500_000, np.inf]
    bands = {}
    for name in np.unique(channel):
        mask = channel == name
        rows = []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            band = mask & (wage > low) & (wage <= high)
            total = np.sum((weight * wage)[band])
            rows.append(
                {
                    "wage_over": low,
                    "wage_up_to": None if np.isinf(high) else high,
                    "wages": float(total),
                    "desired_traditional_401k": float(
                        np.sum((weight * deferral)[band])
                    ),
                    "deferral_share_of_wages": float(
                        np.sum((weight * deferral)[band]) / total
                    )
                    if total
                    else None,
                    "contributor_share_count_basis": float(
                        weight[band & (deferral > 0)].sum() / weight[band].sum()
                    )
                    if weight[band].sum()
                    else None,
                }
            )
        bands[name] = rows
    out["deferral_by_wage_band"] = bands

    # 2. Source persons on both channels: the ASEC row and its PUF clone.
    frame = person[KEYS + ["person_id", "person_tax_unit_id"]].copy()
    frame["wage"], frame["deferral"], frame["weight"] = wage, deferral, weight
    frame["age"] = age
    asec = frame[(channel == "asec") & (clone == 0)]
    puf = frame[(channel == "puf_tax_detail") & (clone == 1)]
    tail = frame[(channel == "puf_tax_detail") & (clone == 2)]
    duplicated = int(asec.duplicated(KEYS).sum() + puf.duplicated(KEYS).sum())
    both = asec.set_index(KEYS).join(
        puf.set_index(KEYS), lsuffix="_asec", rsuffix="_puf", how="inner"
    )
    a, p = both["wage_asec"].to_numpy(), both["wage_puf"].to_numpy()
    positive = (a > 0) & (p > 0)
    matched = {
        "asec_rows_count": int(len(asec)),
        "puf_clone1_rows_count": int(len(puf)),
        "matched_count": int(len(both)),
        "duplicate_keys_count": duplicated,
        "pearson_all": float(np.corrcoef(a, p)[0, 1]),
        "spearman_all": spearman(a, p),
        "both_positive_count": int(positive.sum()),
        "pearson_both_positive": float(np.corrcoef(a[positive], p[positive])[0, 1]),
        "spearman_both_positive": spearman(a[positive], p[positive]),
        "puf_over_asec_ratio_both_positive": ratio_summary(p[positive] / a[positive]),
        "asec_positive_puf_zero_count": int(((a > 0) & (p == 0)).sum()),
        "asec_zero_puf_positive_count": int(((a == 0) & (p > 0)).sum()),
        "exactly_equal_count": int((a == p).sum()),
        "exactly_equal_and_positive_count": int(((a == p) & (a > 0)).sum()),
        "unweighted_mean_asec": float(a.mean()),
        "unweighted_mean_puf": float(p.mean()),
        "unweighted_sum_ratio_puf_over_asec": float(p.sum() / a.sum()),
        "max_asec": float(a.max()),
        "max_puf": float(p.max()),
        "deferral_pearson_all": float(
            np.corrcoef(both["deferral_asec"], both["deferral_puf"])[0, 1]
        ),
        "deferral_exactly_equal_and_positive_count": int(
            (
                (both["deferral_asec"] == both["deferral_puf"])
                & (both["deferral_asec"] > 0)
            ).sum()
        ),
    }
    out["matched_persons"] = matched

    # Tax-unit grain: the QRF predicts a tax-unit total and the build splits it
    # across people, so compare unit totals and the within-unit person shares.
    both = both.reset_index()
    unit = both.groupby("person_tax_unit_id_asec").agg(
        asec=("wage_asec", "sum"),
        puf=("wage_puf", "sum"),
        puf_units=("person_tax_unit_id_puf", "nunique"),
    )
    ua, up = unit["asec"].to_numpy(), unit["puf"].to_numpy()
    unit_positive = (ua > 0) & (up > 0)
    both["asec_unit_total"] = both.groupby("person_tax_unit_id_asec")[
        "wage_asec"
    ].transform("sum")
    both["puf_unit_total"] = both.groupby("person_tax_unit_id_asec")[
        "wage_puf"
    ].transform("sum")
    # Units where the split is not forced: every member is 15 or older (the
    # build restricts PUF earnings to that universe) and at least two members
    # have positive ASEC wages.
    eligible_all = (
        both.groupby("person_tax_unit_id_asec")["age_asec"].transform("min") >= 15
    )
    earners = both.groupby("person_tax_unit_id_asec")["wage_asec"].transform(
        lambda s: (s > 0).sum()
    )
    share_rows = (
        (both["asec_unit_total"] > 0)
        & (both["puf_unit_total"] > 0)
        & eligible_all
        & (earners >= 2)
    )
    share_gap = (
        both.loc[share_rows, "wage_puf"] / both.loc[share_rows, "puf_unit_total"]
        - both.loc[share_rows, "wage_asec"] / both.loc[share_rows, "asec_unit_total"]
    ).abs()
    out["matched_tax_units"] = {
        "tax_units_count": int(len(unit)),
        "one_to_one_count": int((unit["puf_units"] == 1).sum()),
        "pearson_all": float(np.corrcoef(ua, up)[0, 1]),
        "spearman_all": spearman(ua, up),
        "both_positive_count": int(unit_positive.sum()),
        "puf_over_asec_ratio_both_positive": ratio_summary(
            up[unit_positive] / ua[unit_positive]
        ),
        "asec_positive_puf_zero_count": int(((ua > 0) & (up == 0)).sum()),
        "asec_zero_puf_positive_count": int(((ua == 0) & (up > 0)).sum()),
        "person_share_rows_checked_count": int(share_rows.sum()),
        "person_share_max_abs_gap": float(share_gap.max()) if len(share_gap) else None,
        "person_share_rows_gap_over_1e-9_count": int((share_gap > 1e-9).sum()),
    }

    # Tail clone (clone index 2) against the clone-1 person it was copied from.
    if len(tail):
        joined = tail.set_index(KEYS).join(
            puf.set_index(KEYS), lsuffix="_tail", rsuffix="_clone1", how="left"
        )
        out["tail_clone_vs_clone1"] = {
            "tail_rows_count": int(len(tail)),
            "matched_count": int(joined["wage_clone1"].notna().sum()),
            "wage_max_abs_difference": float(
                (joined["wage_tail"] - joined["wage_clone1"]).abs().max()
            ),
            "deferral_max_abs_difference": float(
                (joined["deferral_tail"] - joined["deferral_clone1"]).abs().max()
            ),
        }

    # 3. Distributions: positive person wages and positive tax-unit totals.
    distributions = {}
    for label, mask in {
        "asec/clone0": (channel == "asec") & (clone == 0),
        "puf_tax_detail/clone1": (channel == "puf_tax_detail") & (clone == 1),
    }.items():
        earner = mask & (wage > 0)
        units = pd.DataFrame(
            {"unit": tax_unit[mask], "wage": wage[mask], "weight": weight[mask]}
        ).groupby("unit")
        unit_wage = units["wage"].sum().to_numpy()
        unit_weight = units["weight"].first().to_numpy()
        unit_earner = unit_wage > 0
        total = np.sum((weight * wage)[mask])
        distributions[label] = {
            "person_positive_wage_quantiles": weighted_quantiles(
                wage[earner], weight[earner]
            ),
            "person_max": float(wage[mask].max()),
            "person_rows_at_max_count": int((wage[mask] == wage[mask].max()).sum()),
            "tax_unit_positive_wage_quantiles": weighted_quantiles(
                unit_wage[unit_earner], unit_weight[unit_earner]
            ),
            "tax_unit_max": float(unit_wage.max()),
            "tax_units_count": int(len(unit_wage)),
            "tax_units_with_wages_count": int(unit_earner.sum()),
            "weighted_tax_units_with_wages_count": float(
                unit_weight[unit_earner].sum()
            ),
            "weighted_tax_units_count": float(unit_weight.sum()),
            "share_of_wages_over_1m_person": float(
                np.sum((weight * wage)[mask & (wage > 1e6)]) / total
            ),
            "share_of_wages_over_250k_person": float(
                np.sum((weight * wage)[mask & (wage > 250e3)]) / total
            ),
            "tax_unit_distinct_positive_totals_count": int(
                np.unique(unit_wage[unit_earner]).size
            ),
        }
    out["distributions"] = distributions

    # ASEC-half wage against its stored survey source column.
    if "WSAL_VAL" in person:
        mask = (channel == "asec") & (clone == 0)
        source = person["WSAL_VAL"].to_numpy(dtype=np.float64)
        by_year = {}
        for year in np.unique(person["source_year"].to_numpy()[mask]):
            rows = mask & (person["source_year"].to_numpy() == year) & (source > 0)
            ratio = wage[rows] / source[rows]
            by_year[str(year)] = {
                "rows_count": int(rows.sum()),
                "wage_over_WSAL_VAL_min": float(ratio.min()),
                "wage_over_WSAL_VAL_median": float(np.median(ratio)),
                "wage_over_WSAL_VAL_max": float(ratio.max()),
            }
        out["asec_wage_over_survey_source_by_year"] = by_year
        puf_rows = (channel == "puf_tax_detail") & (clone == 1)
        out["puf_clone_wage_equals_copied_WSAL_VAL_rows_count"] = int(
            ((wage == source) & (wage > 0) & puf_rows).sum()
        )

    # 4. Same stored columns in a second file (base frame against release).
    if args.same_rows_as:
        other, other_weight = load(args.same_rows_as)
        left = person.set_index("person_id")
        right = other.set_index("person_id").reindex(left.index)
        comparison = {
            "other_h5": os.path.basename(args.same_rows_as),
            "other_h5_sha256": sha256(args.same_rows_as),
            "rows_count": int(len(left)),
            "rows_missing_in_other_count": int(right[WAGE].isna().sum()),
            "household_weights_identical": bool(np.array_equal(weight, other_weight)),
        }
        label = (
            pd.Series(channel, index=left.index)
            + "/clone"
            + pd.Series(clone, index=left.index).astype(str)
        )
        for column in (WAGE, *CONTRIBUTIONS):
            if column not in left or column not in right:
                continue
            difference = (
                left[column].astype(np.float64) - right[column].astype(np.float64)
            ).abs()
            changed = difference > 1e-9
            comparison[column] = {
                "max_abs_difference": float(difference.max()),
                "rows_changed_count": int(changed.sum()),
                "rows_changed_by_channel_count": {
                    k: int(v) for k, v in changed.groupby(label).sum().items()
                },
            }
        for column in (CHANNEL, CLONE):
            comparison[f"{column}_differing_rows_count"] = int(
                (left[column].astype(str) != right[column].astype(str)).sum()
            )
        out["same_rows_comparison"] = comparison

    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
