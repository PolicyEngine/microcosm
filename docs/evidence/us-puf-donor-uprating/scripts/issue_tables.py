"""Render the issue's tables from the two receipts, so no number is retyped.

Usage:
    python issue_tables.py <receipts/puf_donor_factor_audit.json> \
        <receipts/release_puf_channel_cost.json> > tables.md
"""

from __future__ import annotations

import json
import sys

DONOR_GROUPS = [
    (
        "SOI per-return growth of its own line",
        [
            ("employment_income_before_lsr", "E00200"),
            ("taxable_interest_income", "E00300"),
            ("tax_exempt_interest_income", "E00400"),
            ("qualified_dividend_income", "E00650"),
            (
                "non_qualified_dividend_income",
                "E00600 less E00650, each by its own line",
            ),
            ("non_sch_d_capital_gains", "E01100"),
            ("taxable_ira_distributions", "E01400"),
            ("taxable_private_pension_income", "E01700"),
            ("social_security_retirement", "E02400"),
        ],
    ),
    (
        "Flat 2% a year for itemized deductions",
        [
            ("charitable_cash_donations", "E19800"),
            ("real_estate_taxes", "E18500"),
            (
                "home_mortgage_interest+investment_interest_expense",
                "E19200, split by Microcosm",
            ),
        ],
    ),
    (
        "AGI per-return growth (the archived default)",
        [
            ("alimony_expense", "E03500"),
            ("alimony_income", "E00800"),
            ("casualty_loss", "E20500"),
            ("charitable_non_cash_donations", "E20100"),
            ("domestic_production_ald", "E03240"),
            ("educator_expense", "E03220"),
            ("estate_income", "E26390 less E26400"),
            ("farm_income", "T27800"),
            ("farm_rent_income", "E27200"),
            ("health_savings_account_ald", "E03290"),
            ("investment_income_elected_form_4952", "E58990"),
            ("long_term_capital_gains_before_response", "P23250"),
            ("long_term_capital_gains_on_collectibles", "E24518"),
            ("miscellaneous_income", "E01200"),
            ("qualified_tuition_expenses", "E03230"),
            ("rental_income", "E25850 less E25860"),
            ("salt_refund_income", "E00700"),
            ("self_employed_pension_contributions_desired", "E03300"),
            ("short_term_capital_gains", "P22250"),
            ("student_loan_interest", "E03210"),
            ("traditional_ira_contributions_desired", "E03150"),
            ("unrecaptured_section_1250_gain", "E24515"),
            ("unreimbursed_business_employee_expenses", "E20400"),
        ],
    ),
    (
        "Sign-split business growth, then AGI growth again",
        [
            (
                "self_employment_income_before_lsr+sstb_self_employment_income_before_lsr",
                "E00900",
            ),
        ],
    ),
    (
        "None: TY2015 level",
        [
            ("partnership_income", "E26190 less E26180 plus E25980 less E25960"),
            ("farm_operations_income", "E02100"),
        ],
    ),
]

ITEMIZED = "itemized deductions (1.02 per year)"
AGI = "REMAINING_VARIABLES (adjusted_gross_income)"
WAGES = "E00200 (employment_income)"
INTEREST = "E00300 (taxable_interest_income)"
EXEMPT = "E00400 (exempt_interest)"
QUALIFIED = "E00650 (qualified_dividends)"
DISTRIBUTIONS = "E01100 (capital_gains_distributions)"
PENSIONS = "E01700 (taxable_pension_income)"
IRA = "E01400 (ira_distributions)"
SOCIAL_SECURITY = "E02400 (total_social_security)"
PROFITS = "E00900 pos_only (business_net_profits)"
PARTNERSHIP = "E26270 pos_only (partnership_and_s_corp_income)"
GAINS = "E01000 pos_only (capital_gains_gross)"

# (label, frame columns, own-series key at TY2023 or None, carried keys at
# TY2021, note). Positive mass only, so sign-mixed columns do not cancel. The
# last table column is own-series growth from TY2015 to TY2023 over the factor
# the donor column carries, both from the archived SOI table.
ARM_ROWS = [
    ("Wages", ["employment_income_before_lsr"], WAGES, [WAGES], ""),
    ("Taxable interest", ["taxable_interest_income"], INTEREST, [INTEREST], ""),
    ("Tax-exempt interest", ["tax_exempt_interest_income"], EXEMPT, [EXEMPT], ""),
    ("Qualified dividends", ["qualified_dividend_income"], QUALIFIED, [QUALIFIED], ""),
    ("Non-qualified dividends", ["non_qualified_dividend_income"], None, [], "h"),
    ("Short-term gains", ["short_term_capital_gains"], GAINS, [AGI], "d"),
    ("Long-term gains", ["long_term_capital_gains_before_response"], GAINS, [AGI], "d"),
    (
        "Capital gain distributions",
        ["non_sch_d_capital_gains"],
        DISTRIBUTIONS,
        [DISTRIBUTIONS],
        "e",
    ),
    ("Taxable pensions", ["taxable_private_pension_income"], PENSIONS, [PENSIONS], ""),
    ("Taxable IRA distributions", ["taxable_ira_distributions"], IRA, [IRA], ""),
    (
        "Social Security, retirement component",
        ["social_security_retirement"],
        SOCIAL_SECURITY,
        [SOCIAL_SECURITY],
        "a",
    ),
    (
        "Schedule C profit, with its SSTB share",
        [
            "self_employment_income_before_lsr",
            "sstb_self_employment_income_before_lsr",
        ],
        PROFITS,
        [PROFITS, AGI],
        "b",
    ),
    ("Partnership income, positive", ["partnership_income"], PARTNERSHIP, [], "c"),
    (
        "Rental income, positive",
        ["rental_income"],
        "own series (rent_and_royalty_net_income)",
        [AGI],
        "",
    ),
    (
        "Estate income, positive",
        ["estate_income"],
        "own series (estate_income)",
        [AGI],
        "",
    ),
    ("Cash charitable gifts", ["charitable_cash_donations"], ITEMIZED, [ITEMIZED], "f"),
    ("Real estate taxes", ["real_estate_taxes"], ITEMIZED, [ITEMIZED], "f"),
    ("Mortgage interest", ["home_mortgage_interest"], ITEMIZED, [ITEMIZED], "f"),
    ("Farm operations income, positive", ["farm_operations_income"], None, [], "i"),
    (
        "Partnership self-employment earnings, positive",
        ["partnership_self_employment_net_earnings"],
        None,
        [],
        "i",
    ),
]

# (label, target, source table). Ordinary dividends and taxable Social
# Security each have two national rows that differ by more than 10%; both are
# listed. Other measures with two rows (wages, pensions, IRA distributions)
# show one.
RELEASE_ROWS = [
    (
        "Wages",
        "irs_soi.ty2023.table_1_4.all.wages_salaries_amount@2024",
        "TY2023 Table 1.4",
    ),
    (
        "Taxable interest",
        "irs_soi.ty2023.table_4_3.all_returns_excluding_dependents.all.taxable_interest_amount@2024",
        "TY2023 Table 4.3, excluding dependents",
    ),
    (
        "Tax-exempt interest",
        "irs_soi.ty2022.historic_table_2.us.all.tax_exempt_interest_amount@2024",
        "TY2022 Historic Table 2",
    ),
    (
        "Ordinary dividends",
        "irs_soi.ty2023.table_4_3.all_returns_excluding_dependents.all.ordinary_dividends_amount@2024",
        "TY2023 Table 4.3, excluding dependents",
    ),
    (
        "Ordinary dividends",
        "irs_soi.ty2022.historic_table_2.us.all.ordinary_dividends_amount@2024",
        "TY2022 Historic Table 2",
    ),
    (
        "Qualified dividends",
        "irs_soi.ty2022.historic_table_2.us.all.qualified_dividends_amount@2024",
        "TY2022 Historic Table 2",
    ),
    (
        "Taxable pensions",
        "irs_soi.ty2023.table_1_4.all.taxable_pension_income_amount@2024",
        "TY2023 Table 1.4",
    ),
    (
        "Taxable IRA distributions",
        "irs_soi.ty2023.table_1_4.all.taxable_ira_distributions_amount@2024",
        "TY2023 Table 1.4",
    ),
    (
        "Taxable Social Security (a)",
        "irs_soi.ty2023.table_1_4.all.taxable_social_security_amount@2024",
        "TY2023 Table 1.4",
    ),
    (
        "Taxable Social Security (a)",
        "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_amount@2024",
        "TY2022 Historic Table 2",
    ),
    (
        "Schedule C net profit",
        "irs_soi.ty2022.historic_table_2.us.all.schedule_c_income_amount@2024",
        "TY2022 Historic Table 2",
    ),
    (
        "Partnership and S-corp net income",
        "irs_soi.ty2022.historic_table_2.us.all.partnership_scorp_income_amount@2024",
        "TY2022 Historic Table 2",
    ),
    (
        "Net capital gains",
        "irs_soi.ty2023.table_1_4.all.net_capital_gains_amount@2024",
        "TY2023 Table 1.4",
    ),
    (
        "Capital gain distributions (e)",
        "irs_soi.ty2023.table_1_4.all.capital_gain_distributions_amount@2024",
        "TY2023 Table 1.4",
    ),
    (
        "Estate and trust net income",
        "irs_soi.ty2023.table_1_4.all.estate_trust_net_income_amount@2024",
        "TY2023 Table 1.4",
    ),
    (
        "Rent and royalty net income (g)",
        "irs_soi.ty2022.historic_table_2.us.all.rental_royalty_income_amount@2024",
        "TY2022 Historic Table 2",
    ),
    (
        "Real estate taxes",
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all.real_estate_taxes_amount@2024",
        "TY2023 Table 2.1, itemizers",
    ),
    (
        "Mortgage interest",
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all.home_mortgage_interest_amount@2024",
        "TY2023 Table 2.1, itemizers",
    ),
    (
        "Charitable gifts",
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all.charitable_amount@2024",
        "TY2023 Table 2.1, itemizers",
    ),
    (
        "AGI",
        "irs_soi.ty2023.table_1_1.all.adjusted_gross_income@2024",
        "TY2023 Table 1.1",
    ),
]


def fmt(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def donor_table(donor: dict) -> str:
    columns = donor["microcosm_donor_columns"]
    lines = [
        "| Factor carried, TY2015 to TY2021 | Microcosm donor column | Source fields | Records with a nonzero source value | Share matching the archived 2015-to-2021 value | Share matching 2015, 2022 or 2023 | Modal ratio to the source value |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for group, members in DONOR_GROUPS:
        for column, fields in members:
            entry = columns[column]
            shares = entry["match_share_of_raw_nonzero"]
            other = max(
                shares[k]
                for k in ("2015 (raw, no uprating)", "2015 to 2022", "2015 to 2023")
            )
            lines.append(
                f"| {group} | `{column.replace('+', '` + `')}` | {fields} | "
                f"{entry['raw_nonzero_records']:,} | "
                f"{fmt(shares['2015 to 2021'], 4)} | {fmt(other, 4)} | "
                f"{fmt(entry.get('modal_ratio_processed_to_raw'), 4)} |"
            )
    return "\n".join(lines)


def leg_table(donor: dict) -> str:
    table = donor["archived_factor_table"]
    keys = [
        "E00200 (employment_income)",
        "E00300 (taxable_interest_income)",
        "E00400 (exempt_interest)",
        "E00600 (ordinary_dividends)",
        "E00650 (qualified_dividends)",
        "E01100 (capital_gains_distributions)",
        "E01000 pos_only (capital_gains_gross)",
        "E01400 (ira_distributions)",
        "E01700 (taxable_pension_income)",
        "E02400 (total_social_security)",
        "E02300 (unemployment_compensation)",
        "E00900 pos_only (business_net_profits)",
        "E00900 neg_only (business_net_losses)",
        "E26270 pos_only (partnership_and_s_corp_income)",
        "REMAINING_VARIABLES (adjusted_gross_income)",
        "S006 weights (return count)",
    ]
    lines = [
        "| Archived series | 2015 to 2021, applied | 2021 to 2022 | 2021 to 2023, not applied |",
        "|---|---:|---:|---:|",
    ]
    for key in keys:
        lines.append(
            f"| {key} | {fmt(table['2021'][key], 4)} | "
            f"{fmt(table['2021 to 2022'][key], 4)} | {fmt(table['2021 to 2023'][key], 4)} |"
        )
    return "\n".join(lines)


def arm_table(donor: dict, release: dict) -> str:
    table = donor["archived_factor_table"]
    donor_mass = donor["donor_weighted_mass"]["columns"]
    stored = release["stored_inputs_by_arm"]
    lines = [
        "| Donor column | Donor at its own weights, $B | PUF arm ×2 at design weights, $B | PUF arm / donor | ASEC arm ×2 at design weights, $B | Own series at TY2023 / level the donor carries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, columns, own_key, carried_keys, note in ARM_ROWS:
        donor_total = sum(donor_mass[c]["positive"] for c in columns)
        puf = 2 * sum(stored[c]["puf_arm_design_positive"] for c in columns)
        asec = 2 * sum(stored[c]["asec_design_positive"] for c in columns)
        if own_key is None:
            leg = ""
        else:
            carried = 1.0
            for key in carried_keys:
                carried *= table["2021"][key]
            leg = fmt(table["2023"][own_key] / carried)
        label = f"{label} ({note})" if note else label
        lines.append(
            f"| {label} | {donor_total / 1e9:,.1f} | {puf / 1e9:,.1f} | "
            f"{fmt(puf / donor_total, 2)} | {asec / 1e9:,.1f} | {leg} |"
        )
    units = release["weighted_units_by_arm"]["tax_units"]
    donor_units = donor["donor_weighted_mass"]["tax_units_weighted"]
    puf_units = 2 * units["puf_arm_design"]
    lines.append(
        f"| Tax units, M | {donor_units / 1e6:,.1f} | {puf_units / 1e6:,.1f} | "
        f"{fmt(puf_units / donor_units, 2)} | {2 * units['asec_design'] / 1e6:,.1f} | |"
    )
    return "\n".join(lines)


def release_table(release: dict) -> str:
    lines = [
        "| National row, aged to 2024 | Source | Target, $B | PUF arm ×2 / target, design weights | ASEC arm ×2 / target, design weights | Estimate / target, design weights | Estimate / target, final weights | PUF-arm part, final / design | ASEC-arm part, final / design |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, name, source in RELEASE_ROWS:
        row = release["national_targets"][name]
        target = row["target"]
        channels = row["channels"]
        puf_initial = (
            channels["puf_clone"]["initial"] + channels["cg_tail_clone"]["initial"]
        )
        puf_final = channels["puf_clone"]["final"] + channels["cg_tail_clone"]["final"]
        asec_initial = channels["asec"]["initial"]
        asec_final = channels["asec"]["final"]
        lines.append(
            f"| {label} | {source} | {target / 1e9:,.1f} | "
            f"{fmt(2 * puf_initial / target)} | "
            f"{fmt(2 * asec_initial / target)} | {fmt(row['initial_estimate'] / target)} | "
            f"{fmt(row['final_estimate'] / target)} | "
            f"{fmt(puf_final / puf_initial) if puf_initial else ''} | "
            f"{fmt(asec_final / asec_initial) if asec_initial else 'n/a'} |"
        )
    return "\n".join(lines)


# National rows fed by one donor column on the PUF arm: (label, target,
# own-series key at TY2023, carried keys at TY2021).
REBASE_ROWS = [
    (
        "Wages",
        "irs_soi.ty2023.table_1_4.all.wages_salaries_amount@2024",
        WAGES,
        [WAGES],
    ),
    (
        "Taxable interest",
        "irs_soi.ty2023.table_4_3.all_returns_excluding_dependents.all.taxable_interest_amount@2024",
        INTEREST,
        [INTEREST],
    ),
    (
        "Tax-exempt interest",
        "irs_soi.ty2022.historic_table_2.us.all.tax_exempt_interest_amount@2024",
        EXEMPT,
        [EXEMPT],
    ),
    (
        "Qualified dividends",
        "irs_soi.ty2022.historic_table_2.us.all.qualified_dividends_amount@2024",
        QUALIFIED,
        [QUALIFIED],
    ),
    (
        "Partnership and S-corp net income",
        "irs_soi.ty2022.historic_table_2.us.all.partnership_scorp_income_amount@2024",
        PARTNERSHIP,
        [],
    ),
]


def rebase_table(donor: dict, release: dict) -> str:
    """Design-weight arithmetic: the PUF-arm part times the own-series factor."""

    table = donor["archived_factor_table"]
    lines = [
        "| National row | Estimate / target today | PUF arm ×2 / target × own-series factor | Estimate / target after |",
        "|---|---:|---:|---:|",
    ]
    for label, name, own_key, carried_keys in REBASE_ROWS:
        carried = 1.0
        for key in carried_keys:
            carried *= table["2021"][key]
        factor = table["2023"][own_key] / carried
        row = release["national_targets"][name]
        target = row["target"]
        channels = row["channels"]
        puf = channels["puf_clone"]["initial"] + channels["cg_tail_clone"]["initial"]
        asec = channels["asec"]["initial"]
        lines.append(
            f"| {label} | {fmt((puf + asec) / target)} | "
            f"{fmt(2 * puf / target)} × {fmt(factor)} | "
            f"{fmt((puf * factor + asec) / target)} |"
        )
    return "\n".join(lines)


def weight_table(release: dict) -> str:
    weights = release["household_weight_by_channel"]
    ratios = release["household_weight_ratio_by_channel"]
    labels = {
        "asec": "ASEC",
        "puf_clone": "PUF clone",
        "cg_tail_clone": "Capital-gains tail clone",
    }
    lines = [
        "| Arm | Households | Weight, design (M) | Weight, final (M) | Final / design | Household ratio: 5th, 50th, 95th percentile | Share at the 5x cap |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for key, label in labels.items():
        w, r = weights[key], ratios[key]
        p = r["percentiles"]
        lines.append(
            f"| {label} | {release['households'][key]:,} | {w['initial'] / 1e6:,.2f} | "
            f"{w['final'] / 1e6:,.2f} | {fmt(w['final_over_initial'])} | "
            f"{fmt(p['5'], 2)}, {fmt(p['50'], 2)}, {fmt(p['95'], 2)} | "
            f"{100 * r['share_at_or_above_4_99']:.1f}% |"
        )
    return "\n".join(lines)


def totals_table(donor: dict) -> str:
    block = donor["donor_weighted_totals_vs_soi"]
    lines = [
        "| SOI line | Donor weighted total, $B | SOI TY2021, $B | SOI TY2023, $B | Donor / TY2021 | Donor / TY2023 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in block["lines"].items():
        total = row["donor_weighted_total"]
        soi = row["soi_actual"]
        lines.append(
            f"| {label} | {total / 1e9:,.1f} | {soi['2021'] / 1e9:,.1f} | "
            f"{soi['2023'] / 1e9:,.1f} | {fmt(total / soi['2021'])} | "
            f"{fmt(total / soi['2023'])} |"
        )
    returns = block["soi_returns"]
    lines.append(
        f"| Returns (donor weight total), M | {block['donor_weight_total'] / 1e6:,.2f} | "
        f"{returns['2021'] / 1e6:,.2f} | {returns['2023'] / 1e6:,.2f} | "
        f"{fmt(block['donor_weight_total'] / returns['2021'])} | "
        f"{fmt(block['donor_weight_total'] / returns['2023'])} |"
    )
    return "\n".join(lines)


def main() -> None:
    donor = json.loads(open(sys.argv[1]).read())
    release = json.loads(open(sys.argv[2]).read())
    print("## Donor totals against SOI actuals\n")
    print(totals_table(donor))
    print("\n## Donor columns by factor\n")
    print(donor_table(donor))
    print(
        "\nFor the two TY2015 columns the hypotheses coincide: no archived list "
        "names their fields, so every hypothesis equals the raw value."
    )
    print("\n## The leg the 2024 branch never applied\n")
    print(leg_table(donor))
    print("\n## From donor to PUF arm\n")
    print(arm_table(donor, release))
    print("\n## Release, national rows\n")
    print(release_table(release))
    print("\n## A rebase alone, at design weights\n")
    print(rebase_table(donor, release))
    print("\n## Release, weights by arm\n")
    print(weight_table(release))


if __name__ == "__main__":
    main()
