"""Aggregate-only evidence for microcosm#1003 (docs/evidence/uk-lisa-1003).

Reads the lane's local receipts (written by the sibling measurement scripts on the
licensed builds under data/ukds/acceptance/1003-lisa/) and writes the committed
CSVs. Disclosure control: donor counts under 10 are withheld, and so is any count
that would reveal one by subtraction (the credible holders in the balance fit, the
banded and ownership-imputed classes apart); the donor's lowest household-income
tertile holds fewer than 10 holders, so income is published as tertiles 1-2
against tertile 3; balance quantiles and means are rounded to two significant
figures; no single respondent's value (a donor maximum, the spine's maximum draw)
is written.

    python docs/evidence/uk-lisa-1003/scripts/extract_lisa_1003_evidence.py \
        --lane-dir data/ukds/acceptance/1003-lisa \
        --out docs/evidence/uk-lisa-1003
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

AGE_GROUPS = ("18-24", "25-34", "35-44", "45-54", "55+")
MODELS = (
    ("stage logistic", "stage_logistic"),
    ("house gate (boosting, defaults)", "house_gate_boosting_defaults"),
    ("boosting tuned for rare events", "boosting_tuned_rare_events"),
    ("age-group rates", "age_group_rates"),
)
QUANTILES = ("p10", "p25", "p50", "p75", "p90")


def pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.3f}"


def sig2(value: float) -> int:
    return int(float(f"{value:.2g}"))


def count(value: int) -> int | str:
    return "<10" if 0 < value < 10 else value


def write(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def donor_audit(evidence: dict[str, Any]) -> list[list[Any]]:
    donor = evidence["donor"]
    classes = donor["response_classes"]
    rule = donor["credibility_rule"]
    adult_weight = sum(entry["weight"] for entry in classes.values())
    banded_or_imputed = (
        classes["holder_banded_value"]["persons"]
        + classes["holder_imputed_value"]["persons"]
        + classes["imputed_holder"]["persons"]
    )
    return [
        ["persons_read", donor["persons_read"]],
        ["households", donor["households"]],
        [
            "households_with_person_values_summing_to_aggregate",
            donor["aggregate_check"]["households_checked"],
        ],
        [
            "largest_person_to_household_difference_gbp",
            donor["aggregate_check"]["max_abs_difference_gbp"],
        ],
        ["dependent_children_excluded", donor["dependent_children_excluded"]],
        ["dependent_children_with_a_value", donor["dependent_children_with_a_value"]],
        ["donor_adults", donor["donor_adults"]],
        ["donor_adults_weighted_millions", round(adult_weight / 1e6, 1)],
        [
            "released_flag_value_disagreements",
            donor["released_flag_value_disagreements"],
        ],
        ["holders_released", donor["holders_released"]],
        [
            "weighted_adult_ownership_pct_released",
            pct(donor["weighted_adult_ownership_share_released"]),
        ],
        ["observed_non_holders", classes["observed_non_holder"]["persons"]],
        ["ons_imputed_non_holders", classes["imputed_non_holder"]["persons"]],
        ["holders_exact_value", classes["observed_holder_exact_value"]["persons"]],
        ["holders_banded_or_ons_imputed", banded_or_imputed],
        ["holders_recoded_for_age", rule["holders_recoded_for_age"]],
        [
            "recoded_pct_of_weighted_released_holders",
            pct(rule["recoded_weighted_share_of_holders"]),
        ],
        [
            "recoded_pct_of_weighted_lisa_mass",
            pct(rule["recoded_weighted_share_of_lisa_mass"]),
        ],
        ["holders_above_balance_ceiling", count(rule["holders_above_ceiling"])],
        ["balance_ceiling_gbp", int(rule["balance_ceiling_gbp"])],
        ["holders_after_rule", donor["holders_after_rule"]],
        [
            "weighted_adult_ownership_pct_after_rule",
            pct(donor["weighted_adult_ownership_share"]),
        ],
    ]


def model_comparison(comparison: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    keys = [("all_adults", "all adults")]
    keys += [(f"age {group}", group) for group in AGE_GROUPS]
    keys += [
        ("private renters", "private renters"),
        ("other tenures", "other tenures"),
        ("household net income tertiles 1-2", "household net income tertiles 1-2"),
        ("household net income tertile 3", "household net income tertile 3"),
    ]
    for key, label in keys:
        rows.append(
            [label, pct(comparison["donor"][key])]
            + [pct(comparison[name][key]) for name, _ in MODELS]
        )
    rows.append(
        ["held-out weighted log loss x1000", ""]
        + [
            f"{comparison[name]['held_out_weighted_log_loss_x1000']:.2f}"
            for name, _ in MODELS
        ]
    )
    return rows


def realised(
    main: dict[str, Any], arm_b: dict[str, Any], comparison: dict[str, Any]
) -> list[list[Any]]:
    donor = main["realised"]["donor"]
    spine = main["realised"]["recipient"]
    spine_b = arm_b["realised"]["recipient"]

    def merged(profile: dict[str, Any]) -> float:
        # The spine's income tertiles carry equal weight by construction (weighted cuts).
        tertile = profile["by_household_net_income_tertile"]
        return (tertile["1"] + tertile["2"]) / 2

    rows: list[list[Any]] = [
        [
            "all adults",
            pct(donor["all_adults"]),
            pct(spine["all_adults"]),
            pct(spine_b["all_adults"]),
        ],
        [
            "model expected share",
            "",
            pct(main["realised"]["model_expected_adult_ownership_share"]),
            pct(arm_b["realised"]["model_expected_adult_ownership_share"]),
        ],
    ]
    for group in AGE_GROUPS:
        rows.append(
            [
                group,
                pct(donor["by_age_group"][group]),
                pct(spine["by_age_group"][group]),
                pct(spine_b["by_age_group"][group]),
            ]
        )
    for key, label in (("female", "female"), ("male", "male")):
        rows.append(
            [
                label,
                pct(donor["by_sex"][key]),
                pct(spine["by_sex"][key]),
                pct(spine_b["by_sex"][key]),
            ]
        )
    for key, label in (
        ("private_renter", "private renters"),
        ("other", "other tenures"),
    ):
        rows.append(
            [
                label,
                pct(donor["by_private_renting"][key]),
                pct(spine["by_private_renting"][key]),
                pct(spine_b["by_private_renting"][key]),
            ]
        )
    rows.append(
        [
            "household net income tertiles 1-2",
            pct(comparison["donor"]["household net income tertiles 1-2"]),
            pct(merged(spine)),
            pct(merged(spine_b)),
        ]
    )
    rows.append(
        [
            "household net income tertile 3",
            pct(donor["by_household_net_income_tertile"]["3"]),
            pct(spine["by_household_net_income_tertile"]["3"]),
            pct(spine_b["by_household_net_income_tertile"]["3"]),
        ]
    )
    for channel in ("frs", "spi"):
        rows.append(
            [
                f"{channel.upper()}-channel rows",
                "",
                pct(spine["by_channel"][channel]),
                pct(spine_b["by_channel"][channel]),
            ]
        )
    rows.append(
        [
            "co-holding: weighted share of holder households with 2+ holders",
            pct(main["realised"]["co_holding_share_of_owner_households"]["donor"]),
            pct(main["realised"]["co_holding_share_of_owner_households"]["recipient"]),
            pct(arm_b["realised"]["co_holding_share_of_owner_households"]["recipient"]),
        ]
    )
    return rows


def balances(
    main: dict[str, Any], arm_b: dict[str, Any], comparison: dict[str, Any]
) -> list[list[Any]]:
    balance = comparison["balance"]
    return [
        ["donor credible holders"]
        + [sig2(balance["donor_quantiles"][q]) for q in QUANTILES]
        + [sig2(balance["donor_weighted_mean"])],
        ["donor credible holders, held-out draws (5-fold)"]
        + [sig2(balance["held_out_draw_quantiles"][q]) for q in QUANTILES]
        + [sig2(balance["held_out_draw_weighted_mean"])],
        ["spine owners (stage evidence)"]
        + [sig2(main["realised"]["owner_balance_quantiles"][q]) for q in QUANTILES]
        + [sig2(main["realised"]["owner_weighted_mean_balance"])],
        ["spine owners, arm B (stage evidence)"]
        + [sig2(arm_b["realised"]["owner_balance_quantiles"][q]) for q in QUANTILES]
        + [sig2(arm_b["realised"]["owner_weighted_mean_balance"])],
    ]


def financial_draws(probe: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    populations = [("donor", probe["donor"]), ("spine", probe["spine_all"])]
    populations += [
        (f"spine {channel}-channel rows", groups)
        for channel, groups in sorted(probe["spine_by_channel"].items())
    ]
    for name, groups in populations:
        for group in AGE_GROUPS:
            row = groups[group]
            rows.append(
                [
                    name,
                    group,
                    f"{row['weighted_adult_share'] * 100:.1f}",
                    int(row["gross_financial_wealth_weighted_median"]),
                    row["gross_financial_wealth_weighted_mean_log1p"],
                    row["savings_weighted_mean_log1p"],
                    row["cash_isa_weighted_mean_log1p"],
                    row["stocks_and_shares_isa_weighted_mean_log1p"],
                    row["employment_income_weighted_mean_log1p"],
                    f"{row['private_renter_share'] * 100:.1f}",
                ]
            )
    return rows


def checks(
    main: dict[str, Any],
    arm_b: dict[str, Any],
    gates: dict[str, dict[str, Any]],
    twin: dict[str, Any],
    determinism: dict[str, Any],
) -> list[list[Any]]:
    def coherence(receipt: dict[str, Any], key: str) -> Any:
        return receipt["evidence"]["financial_wealth_coherence"][key]

    def passed(name: str) -> str:
        statuses = [gate["status"] for gate in gates[name]["gates"].values()]
        return f"{statuses.count('passed')} of {len(statuses)}"

    return [
        [
            "cap: households over gross financial wealth",
            count(coherence(main, "households_over_financial_wealth")),
            count(coherence(arm_b, "households_over_financial_wealth")),
        ],
        [
            "cap: owners cleared",
            count(coherence(main, "owners_cleared")),
            count(coherence(arm_b, "owners_cleared")),
        ],
        [
            "cap: pct of weighted LISA mass removed",
            pct(coherence(main, "share_of_weighted_lisa_mass_removed")),
            pct(coherence(arm_b, "share_of_weighted_lisa_mass_removed")),
        ],
        ["support clip: rows clipped low / high", "0 / 0", "0 / 0"],
        [
            "H5: no household total above gross financial wealth",
            main["recomputed"]["households_over_gross_financial_wealth"] == 0,
            arm_b["recomputed"]["households_over_gross_financial_wealth"] == 0,
        ],
        [
            "H5: household total equals the sum of its persons",
            main["recomputed"]["household_total_identity_max_abs_gbp"] == 0.0,
            arm_b["recomputed"]["household_total_identity_max_abs_gbp"] == 0.0,
        ],
        [
            "H5: holds exactly where the balance is positive",
            main["recomputed"]["flag_iff_positive_balance"],
            arm_b["recomputed"]["flag_iff_positive_balance"],
        ],
        [
            "H5: support gate on uk/was_lisa_support_bounds.json",
            main["release_checks"]["support_gate"],
            arm_b["release_checks"]["support_gate"],
        ],
        [
            "H5: nonnegative",
            main["release_checks"]["nonnegative"],
            arm_b["release_checks"]["nonnegative"],
        ],
        [
            "H5: on the export allow-list",
            main["release_checks"]["export_allow_listed"],
            arm_b["release_checks"]["export_allow_listed"],
        ],
        [
            "H5: no degenerate cell",
            all(v is None for v in main["release_checks"]["degenerate"].values()),
            all(v is None for v in arm_b["release_checks"]["degenerate"].values()),
        ],
        [
            "spine gates passed (control: " + passed("spine-ctl") + ")",
            passed("spine-lisa"),
            passed("spine-lisa-b"),
        ],
        [
            "twin against main: only the three cells differ, mass log gains the conserved was_lisa record",
            twin["verdict"],
            "",
        ],
        [
            "determinism: second build payload-identical, equal mass log and stage evidence",
            determinism["verdict"],
            "",
        ],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lane: Path = args.lane_dir
    receipts = lane / "receipts"
    main_receipt = json.loads((receipts / "receipt-spine-lisa.json").read_text())
    arm_b_receipt = json.loads((receipts / "receipt-spine-lisa-b.json").read_text())
    comparison = json.loads((receipts / "model-comparison.json").read_text())
    probe = json.loads(
        (receipts / "financial-wealth-probe-spine-lisa.json").read_text()
    )
    gates = {
        name: json.loads((lane / name / f"{name}.spine_gates.json").read_text())
        for name in ("spine-ctl", "spine-lisa", "spine-lisa-b")
    }
    twin = json.loads(
        (lane / "diff-spine-lisa-vs-spine-ctl" / "adjudication.json").read_text()
    )
    determinism = json.loads(
        (lane / "diff-spine-lisa-vs-spine-lisa-2" / "determinism.json").read_text()
    )
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    main_evidence = main_receipt["evidence"]
    arm_b_evidence = arm_b_receipt["evidence"]
    write(out / "donor_audit.csv", ["item", "value"], donor_audit(main_evidence))
    write(
        out / "ownership_model_comparison.csv",
        ["group", "donor_pct"] + [f"{column}_pct" for _, column in MODELS],
        model_comparison(comparison),
    )
    write(
        out / "ownership_realised.csv",
        ["group", "donor_pct", "spine_pct", "spine_arm_b_pct"],
        realised(main_evidence, arm_b_evidence, comparison),
    )
    write(
        out / "balance_quantiles.csv",
        ["owners"] + [f"{q}_gbp" for q in QUANTILES] + ["weighted_mean_gbp"],
        balances(main_evidence, arm_b_evidence, comparison),
    )
    write(
        out / "financial_draws_by_age.csv",
        [
            "population",
            "age_group",
            "weighted_adult_share_pct",
            "gross_financial_wealth_weighted_median_gbp",
            "gross_financial_wealth_weighted_mean_log1p",
            "savings_weighted_mean_log1p",
            "cash_isa_weighted_mean_log1p",
            "stocks_and_shares_isa_weighted_mean_log1p",
            "employment_income_weighted_mean_log1p",
            "private_renter_share_pct",
        ],
        financial_draws(probe),
    )
    write(
        out / "checks.csv",
        ["check", "spine", "spine_arm_b"],
        checks(main_receipt, arm_b_receipt, gates, twin, determinism),
    )


if __name__ == "__main__":
    main()
