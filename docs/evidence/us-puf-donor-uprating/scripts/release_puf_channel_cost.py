"""Decompose the release's calibration targets by support channel.

Usage:
    python -I release_puf_channel_cost.py \
        --checkpoint <release-checkpoints/target_frame_checkpoint.h5> \
        --calibration-npz <artifacts/populace_us_2024_calibration.npz> \
        --diagnostics <calibration_diagnostics.json> \
        --release-h5 <artifacts/populace_us_2024.h5> \
        --base-h5 <base_populace_us_2024_puf_support.h5> \
        --out <receipt.json>

The checkpoint stores every compiled target column per household and the
pre-solve design weights; the npz stores the initial and final household
weights. For each target this script splits the estimate into the ASEC
channel, the PUF clone (clone index 1) and the capital-gains tail clone
(clone index 2), at initial and at final weights, and checks that the parts
add up to the estimates the build recorded. It reports aggregates only: the
JSON receipt holds the national rows and the summaries, and a CSV beside it
holds one line per target, gzipped.

The npz holds object arrays (target names), so it is loaded with
allow_pickle=True; it is this build's own output, not a downloaded file.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CHANNELS = {
    ("asec", 0): "asec",
    ("puf_tax_detail", 1): "puf_clone",
    ("puf_tax_detail", 2): "cg_tail_clone",
}
# Calibration-target variables (target metadata ``variable``) whose PUF-channel
# values come from the PUF donor, wholly or in part.
PUF_SOURCED_TARGET_VARIABLES = (
    "employment_income",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "ordinary_dividend_income",
    "qualified_dividends",
    "taxable_pension_income",
    "ira_distributions",
    "taxable_social_security",
    "business_net_profits",
    "capital_gains_gross",
    "partnership_and_s_corp_income",
    "rent_and_royalty_net_income",
    "non_sch_d_capital_gains",
    "estate_income",
    "estate_losses",
    "real_estate_taxes",
    "salt_deduction",
    "interest_deduction",
    "deductible_mortgage_interest",
    "charitable_deduction",
    "medical_expense_deduction",
    "itemized_taxable_income_deductions",
    "adjusted_gross_income",
    "taxable_income",
    "income_tax_before_credits",
    "income_tax",
)


# Donor columns as the frame stores them (person or tax-unit grain). Person
# columns are summed to tax units before positive and negative legs are split,
# the grain of the donor table they are compared with.
STORED_INPUTS = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "sstb_self_employment_income_before_lsr",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
    "long_term_capital_gains_on_collectibles",
    "unrecaptured_section_1250_gain",
    "non_sch_d_capital_gains",
    "taxable_private_pension_income",
    "taxable_ira_distributions",
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
    "partnership_income",
    "s_corp_income",
    "partnership_self_employment_net_earnings",
    "rental_income",
    "estate_income",
    "farm_income",
    "farm_operations_income",
    "farm_rent_income",
    "miscellaneous_income",
    "alimony_income",
    "alimony_expense",
    "salt_refund_income",
    "charitable_cash_donations",
    "charitable_non_cash_donations",
    "real_estate_taxes",
    "home_mortgage_interest",
    "investment_interest_expense",
    "investment_income_elected_form_4952",
    "student_loan_interest",
    "educator_expense",
    "qualified_tuition_expenses",
    "casualty_loss",
    "unreimbursed_business_employee_expenses",
    "traditional_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
    "domestic_production_ald",
    "health_savings_account_ald",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "qualified_reit_and_ptp_income",
    "qualified_bdc_income",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(values: np.ndarray, qs=(1, 5, 25, 50, 75, 95, 99)) -> dict:
    if len(values) == 0:
        return {}
    return {str(q): float(np.percentile(values, q)) for q in qs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-npz", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--release-h5", type=Path, required=True)
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    receipt: dict = {
        "inputs": {
            name: {
                "file": getattr(args, name).name,
                "sha256": sha256(getattr(args, name)),
            }
            for name in (
                "checkpoint",
                "calibration_npz",
                "diagnostics",
                "release_h5",
                "base_h5",
            )
        }
    }
    diagnostics = json.loads(args.diagnostics.read_text())
    receipt["diagnostics_build_base_dataset_sha256"] = diagnostics["build"][
        "base_dataset_sha256"
    ]
    receipt["support_value_repairs"] = diagnostics["build"]["support_value_repairs"]
    targets = {t["name"]: t for t in diagnostics["targets"]}

    npz = np.load(args.calibration_npz, allow_pickle=True)
    initial = npz["initial_household_weight"]
    final = npz["household_weight"]
    names = [str(n) for n in npz["target_names"]]

    with h5py.File(args.checkpoint, "r") as handle:
        group = handle["tables/household/columns"]
        by_name = {group[key].attrs["name"]: key for key in group.keys()}
        channel_raw = group[by_name["household_support_channel"]]["values"][()]
        clone_raw = group[by_name["household_support_clone_index"]]["values"][()]
        design = handle["weights/household/values"][()]
        channel = np.array(
            [
                CHANNELS[(c.decode() if isinstance(c, bytes) else c, int(i))]
                for c, i in zip(channel_raw, clone_raw)
            ]
        )
        masks = {label: channel == label for label in CHANNELS.values()}
        receipt["checkpoint_design_weights_equal_npz_initial"] = bool(
            np.array_equal(design, initial)
        )
        receipt["households"] = {k: int(v.sum()) for k, v in masks.items()}
        receipt["household_weight_by_channel"] = {
            label: {
                "initial": float(initial[mask].sum()),
                "final": float(final[mask].sum()),
                "final_over_initial": float(final[mask].sum() / initial[mask].sum()),
            }
            for label, mask in masks.items()
        }
        ratio = np.divide(final, initial, out=np.zeros_like(final), where=initial > 0)
        receipt["household_weight_ratio_by_channel"] = {
            label: {
                "percentiles": pct(ratio[mask]),
                "share_at_or_above_4_99": float((ratio[mask] >= 4.99).mean()),
                "share_below_0_01": float((ratio[mask] < 0.01).mean()),
            }
            for label, mask in masks.items()
        }

        # The ASEC arm's stored wage against the survey's own wage field, by
        # the frame's source_year: is the survey half aged at all?
        person = handle["tables/person/columns"]
        person_by_name = {person[key].attrs["name"]: key for key in person.keys()}

        def person_column(name: str) -> np.ndarray:
            return person[person_by_name[name]]["values"][()]

        person_channel = np.array(
            [
                c.decode() if isinstance(c, bytes) else c
                for c in person_column("person_support_channel")
            ]
        )
        source_year = person_column("source_year")
        survey_wage = person_column("WSAL_VAL").astype(np.float64)
        stored_wage = person_column("employment_income_before_lsr").astype(np.float64)
        on_asec = person_channel == "asec"
        receipt["asec_stored_wage_vs_survey_wage"] = {
            str(int(year)): {
                "persons": int((on_asec & (source_year == year)).sum()),
                "share_stored_equals_WSAL_VAL": float(
                    (stored_wage == survey_wage)[on_asec & (source_year == year)].mean()
                ),
            }
            for year in sorted(set(source_year[on_asec].tolist()))
        }

        # Stored inputs by arm, at design and final weights: the level the
        # frame's PUF arm carries for each donor column before any target.
        household_ids = group[by_name["household_id"]]["values"][()]
        household_position = {int(h): i for i, h in enumerate(household_ids)}
        person_household = np.array(
            [household_position[int(h)] for h in person_column("person_household_id")]
        )
        person_arm = channel[person_household]
        tax_unit = handle["tables/tax_unit/columns"]
        tax_unit_by_name = {tax_unit[key].attrs["name"]: key for key in tax_unit.keys()}
        tax_unit_ids = tax_unit[tax_unit_by_name["tax_unit_id"]]["values"][()]
        first_person = {}
        for row, unit in enumerate(person_column("person_tax_unit_id")):
            first_person.setdefault(int(unit), row)
        tax_unit_household = np.array(
            [person_household[first_person[int(unit)]] for unit in tax_unit_ids]
        )
        tax_unit_arm = channel[tax_unit_household]

        person_tax_unit_position = pd.Index(tax_unit_ids).get_indexer(
            person_column("person_tax_unit_id")
        )
        if (person_tax_unit_position < 0).any():
            raise ValueError("person_tax_unit_id not found among tax_unit_id")

        def masses(values):
            """Positive and negative mass by arm, clipped at tax-unit grain."""

            values = values.astype(np.float64)
            out = {}
            for label in ("asec", "puf_arm"):
                mask = (
                    tax_unit_arm == "asec"
                    if label == "asec"
                    else tax_unit_arm != "asec"
                )
                for stage, weight in (("design", initial), ("final", final)):
                    w = weight[tax_unit_household[mask]]
                    out[f"{label}_{stage}_positive"] = float(
                        w @ np.clip(values[mask], 0, None)
                    )
                    out[f"{label}_{stage}_negative"] = float(
                        w @ np.clip(values[mask], None, 0)
                    )
            return out

        stored = {}
        for name in STORED_INPUTS:
            if name in person_by_name:
                unit_totals = np.bincount(
                    person_tax_unit_position,
                    weights=person_column(name).astype(np.float64),
                    minlength=len(tax_unit_ids),
                )
                stored[name] = {"entity": "person", **masses(unit_totals)}
            elif name in tax_unit_by_name:
                stored[name] = {
                    "entity": "tax_unit",
                    **masses(tax_unit[tax_unit_by_name[name]]["values"][()]),
                }
        receipt["stored_inputs_by_arm"] = stored
        receipt["weighted_units_by_arm"] = {
            "tax_units": {
                "asec_design": float(
                    initial[tax_unit_household[tax_unit_arm == "asec"]].sum()
                ),
                "puf_arm_design": float(
                    initial[tax_unit_household[tax_unit_arm != "asec"]].sum()
                ),
            },
            "persons": {
                "asec_design": float(
                    initial[person_household[person_arm == "asec"]].sum()
                ),
                "puf_arm_design": float(
                    initial[person_household[person_arm != "asec"]].sum()
                ),
            },
        }

        decomposition = {}
        reconciliation = []
        for name in names:
            target = targets[name]
            variable = target["metadata"].get("variable")
            if variable not in PUF_SOURCED_TARGET_VARIABLES:
                continue
            column = group[by_name[target["measure"]["name"]]]["values"][()].astype(
                np.float64
            )
            parts = {}
            for label, mask in masks.items():
                parts[label] = {
                    "initial": float(initial[mask] @ column[mask]),
                    "final": float(final[mask] @ column[mask]),
                    "nonzero_households": int((column[mask] != 0).sum()),
                }
            total_initial = sum(p["initial"] for p in parts.values())
            total_final = sum(p["final"] for p in parts.values())
            reconciliation.append(
                (
                    abs(total_initial - target["initial_estimate"])
                    / max(1.0, abs(target["initial_estimate"])),
                    abs(total_final - target["final_estimate"])
                    / max(1.0, abs(target["final_estimate"])),
                )
            )
            meta = target["metadata"]
            decomposition[name] = {
                "variable": variable,
                "unit": meta.get("ledger_measure_unit"),
                "geography_level": meta.get("ledger_geography_level"),
                "agi_lower_bound": meta.get("agi_lower_bound"),
                "agi_upper_bound": meta.get("agi_upper_bound"),
                "filing_status": meta.get("filing_status"),
                "source_period": meta.get("source_period"),
                "aging_factor": meta.get("aging_factor"),
                "aging_factor_source": meta.get("aging_factor_source"),
                "base_variable": meta.get("base_variable"),
                "target": target["target"],
                "initial_estimate": target["initial_estimate"],
                "final_estimate": target["final_estimate"],
                "relative_error": target["relative_error"],
                "target_loss_weight_share": target["target_loss_weight_share"],
                "channels": parts,
            }
        rec = np.array(reconciliation)
        receipt["decomposition_reconciles"] = {
            "targets": len(reconciliation),
            "max_relative_gap_initial": float(rec[:, 0].max()),
            "max_relative_gap_final": float(rec[:, 1].max()),
        }
        # National, all-AGI, all-filing-status rows stay in the JSON; every
        # row goes to the CSV beside it.
        receipt["national_targets"] = {
            name: row
            for name, row in decomposition.items()
            if row["geography_level"] == "country"
            and row["agi_lower_bound"] in ("-inf", None)
            and row["agi_upper_bound"] in ("inf", None)
            and row["filing_status"] in ("All", None)
        }
        csv_path = args.out.with_name(args.out.stem + "_targets.csv.gz")
        with gzip.open(csv_path, "wt", newline="") as handle:
            writer = csv.writer(handle)
            scalar = [k for k in next(iter(decomposition.values())) if k != "channels"]
            parts = [
                f"{label}_{stage}"
                for label in CHANNELS.values()
                for stage in ("initial", "final", "nonzero_households")
            ]
            writer.writerow(["name", *scalar, *parts])
            for name, row in decomposition.items():
                writer.writerow(
                    [
                        name,
                        *[row[k] for k in scalar],
                        *[
                            row["channels"][label][stage]
                            for label in CHANNELS.values()
                            for stage in ("initial", "final", "nonzero_households")
                        ],
                    ]
                )
        receipt["all_targets_csv"] = csv_path.name

        # Per variable: channel shares across all its rows, initial and final.
        summary = defaultdict(lambda: defaultdict(list))
        for row in decomposition.values():
            key = (row["variable"], row["unit"])
            for label in CHANNELS.values():
                for stage in ("initial", "final"):
                    total = (
                        row["initial_estimate"]
                        if stage == "initial"
                        else row["final_estimate"]
                    )
                    if total:
                        summary[key][f"{label}_{stage}_share"].append(
                            row["channels"][label][stage] / total
                        )
            summary[key]["initial_over_target"].append(
                row["initial_estimate"] / row["target"] if row["target"] else np.nan
            )
            summary[key]["final_over_target"].append(
                row["final_estimate"] / row["target"] if row["target"] else np.nan
            )
        receipt["by_variable"] = {
            f"{variable}|{unit}": {
                "rows": len(values["initial_over_target"]),
                **{
                    stat: float(np.nanmedian(series)) for stat, series in values.items()
                },
            }
            for (variable, unit), values in sorted(summary.items())
        }

    args.out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
