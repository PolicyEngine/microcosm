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
