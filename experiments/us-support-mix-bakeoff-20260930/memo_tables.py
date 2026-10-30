"""Build the memo's results tables from the report receipts (no hand copying).

Reads dimensions_by_arm.csv; arms with three ACS draws (the 300k one- and
three-year ACS arms) are shown as the mean of the draws.
Run: uv run python experiments/us-support-mix-bakeoff-20260930/memo_tables.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROWS = {
    "national": [
        ("CPS 2022–24 only", "natural.cps2022-2023-2024.acs100"),
        ("CPS 2022–24 + clones to 300k", "b300k.cps2022-2023-2024.acs000"),
        ("CPS 2024 + ACS to 300k (3 draws)", "b300k.cps2024.acs100"),
        ("CPS 2023–24 + ACS to 300k", "b300k.cps2023-2024.acs100"),
        ("**CPS 2022–24 + ACS to 300k (3 draws)**", "b300k.cps2022-2023-2024.acs100"),
        ("ACS only, 300k", "b300k.cps0.acs100"),
        ("CPS 2022–24 + 50/50 to 600k", "b600k.cps2022-2023-2024.acs050"),
        ("CPS 2022–24 + ACS to 600k", "b600k.cps2022-2023-2024.acs100"),
        ("CPS 2022–24 + ACS to 1.2M", "b1200k.cps2022-2023-2024.acs100"),
    ],
    "local": [
        ("CPS 2022–24 only", "natural.cps2022-2023-2024.acs100"),
        ("CPS 2022–24 + clones to 600k", "b600k.cps2022-2023-2024.acs000"),
        ("CPS 2022–24 + clones to 1.2M", "b1200k.cps2022-2023-2024.acs000"),
        ("CPS 2022–24 + 50/50 to 600k", "b600k.cps2022-2023-2024.acs050"),
        ("CPS 2022–24 + ACS to 600k", "b600k.cps2022-2023-2024.acs100"),
        ("**CPS 2022–24 + ACS to 1.2M**", "b1200k.cps2022-2023-2024.acs100"),
        ("CPS 2024 + ACS to 1.2M", "b1200k.cps2024.acs100"),
        ("ACS only, 1.2M", "b1200k.cps0.acs100"),
    ],
}
COLUMNS = [
    ("cps_native_weighted.state", "CPS-native, state"),
    ("cps_native_weighted.cd", "CPS-native, CD"),
    ("income_components.state", "Income, state"),
    ("tax_items.state", "Tax items, state"),
    ("program_receipt.state", "Program receipt, state"),
    ("demographics_pep.state", "Age (PEP), state"),
]


def main() -> None:
    table = pd.read_csv(HERE / "dimensions_by_arm.csv")
    table["stem"] = table["arm"].str.replace(r"\.s\d+$", "", regex=True)
    for product, rows in ROWS.items():
        print(f"### {product}\n")
        head = "| Arm | Rows | ESS | " + " | ".join(label for _, label in COLUMNS) + " | ACS-native state / CD / county |"
        print(head)
        print("|" + "---|" * (head.count("|") - 1))
        for label, stem in rows:
            group = table[(table["product"] == product) & (table["stem"] == stem)]
            if "draws" not in label:
                group = group[group["seed"] == 0]
            mean = group.mean(numeric_only=True)
            cells = [f"{mean[c]:.3f}" for c, _ in COLUMNS]
            acs = " / ".join(f"{mean[f'acs_native.{lv}']:.3f}" for lv in ("state", "cd", "county"))
            print(f"| {label} | {int(mean['rows']):,} | {mean['ess_distinct']:,.0f} | " + " | ".join(cells) + f" | {acs} |")
        spread = table[(table["product"] == product) & table["stem"].eq(rows[4][1] if product == "national" else "b300k.cps2022-2023-2024.acs100")]
        sd = spread.std(numeric_only=True)
        cells = [f"{sd[c]:.3f}" for c, _ in COLUMNS]
        acs = " / ".join(f"{sd[f'acs_native.{lv}']:.3f}" for lv in ("state", "cd", "county"))
        print("| SD across 3 draws (300k, 2022–24) | | | " + " | ".join(cells) + f" | {acs} |\n")


if __name__ == "__main__":
    main()
