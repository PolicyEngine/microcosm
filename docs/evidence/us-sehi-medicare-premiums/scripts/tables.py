"""Markdown tables from the combined run outputs (no computation beyond differences and ratios)."""

import json
import sys

SOI_RETURNS = 3_595_764
SOI_AMOUNT = 31_232_604_000
JCT_REVENUE_LOSS = 8_400_000_000

LABELS = {
    "shipped": "Status quo (proxy withheld)",
    "reported_where_not_enrolled": "B. Reported premium, non-enrollees only",
    "reported_only": "A. Reported premium, all",
    "reported_plus_modeled_part_b": "C. Reported + modeled Part B (frozen in data)",
    "engine_side_standard_part_b": "E. Engine-side: reported + standard Part B net of MSP",
    "reported_plus_reported_part_b": "C'. Reported + CPS Medicare premium (PEMCPREM)",
    "reported_plus_modeled_part_b_where_reported": "C, only enrollees reporting a premium",
    "reported_plus_modeled_part_b_sole_proprietors": "C, partner-only enrollees withheld",
    "reported_plus_all_modeled_medicare": "C + modeled Part A and Part D IRMAA",
    "engine_side": "Engine-side, full resolver (gross Part B incl. IRMAA)",
}


def b(x):
    return f"${x / 1e9:,.2f}B"


def m(x):
    return f"{x / 1e6:,.2f}M"


def signed_b(x):
    return f"{'+' if x >= 0 else '−'}${abs(x) / 1e9:,.2f}B"


def table(path):
    d = json.load(open(path))
    rows = []
    head = (
        "| Scenario | Returns with the deduction | Deduction | per return | of which Medicare proxy "
        "| Revenue loss (JCT row) | Federal income tax vs status quo | Medical-expense premiums vs status quo |"
    )
    rows.append(head)
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, label in LABELS.items():
        s = d["scenarios"].get(name)
        if s is None:
            continue
        if "error" in s:
            rows.append(
                f"| {label} | {s['error'].split('(')[0]}: {s['error'][:60]}… | | | | | | |"
            )
            continue
        ded = s["deduction"]
        per = ded["amount"] / ded["returns"] if ded["returns"] else 0
        rev = s.get("revenue_loss")
        itax = s.get("income_tax_vs_shipped")
        med = s.get("medical_expense_premiums_vs_shipped", {}).get("total_change")
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    m(ded["returns"]),
                    b(ded["amount"]),
                    f"${per:,.0f}",
                    b(ded["amount_in_medicare_proxy"]),
                    "" if rev is None else b(rev),
                    "" if itax is None else signed_b(itax),
                    "0"
                    if name == "shipped"
                    else ("" if med is None else signed_b(med)),
                ]
            )
            + " |"
        )
    rows.append(
        f"| *SOI Table 1.4, TY2023* | {m(SOI_RETURNS)} | {b(SOI_AMOUNT)} | ${SOI_AMOUNT / SOI_RETURNS:,.0f} | | | | |"
    )
    rows.append(
        f"| *JCT tax expenditure, CY2024 (calibration target)* | | | | | {b(JCT_REVENUE_LOSS)} | | |"
    )
    return "\n".join(rows), d


def detail(d):
    out = []
    for name, label in LABELS.items():
        s = d["scenarios"].get(name)
        if not s or "error" in s:
            continue
        line = [label]
        for key in (
            "state_income_tax_vs_shipped",
            "snap_vs_shipped",
            "medical_expense_deduction_vs_shipped",
        ):
            if key in s:
                line.append(f"{key.replace('_vs_shipped', '')} {signed_b(s[key])}")
        mp = s.get("medical_expense_premiums_vs_shipped")
        if mp:
            line.append(
                f"medical-expense premiums changed for {m(mp['people_changed'])} people ({mp['rows_changed']} rows)"
            )
        pb = s.get("medicare_part_b_premium_vs_shipped")
        if pb:
            line.append(f"Part B rows changed {pb['rows_changed']}")
        line.append(f"peak RSS {s.get('peak_rss_mb')} MB, {s.get('seconds')} s")
        out.append("- " + "; ".join(line))
    return "\n".join(out)


for path in sys.argv[1:]:
    t, d = table(path)
    print(
        f"### {d['label']} (policyengine-us {d['policyengine_us']}, widened={d['widened_plan_businesses']})\n"
    )
    print(t)
    print()
    print(detail(d))
    print()
    print(
        "engine-side vs frozen:",
        json.dumps(d.get("engine_side_vs_frozen_modeled_part_b")),
    )
    sh = d["scenarios"]["shipped"]
    print("shipped out-year:", json.dumps(sh.get("out_year", sh.get("out_year_error"))))
    print()
