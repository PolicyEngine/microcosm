"""Build a local US base H5 with a PUF tax-detail support channel.

This diagnostic builder starts from an existing Populace US H5, clones the
frame into ASEC and PUF-tax-detail support channels, imputes PUF-observed
inputs onto the PUF channel with Populace's weighted QRF, and writes a fresh
base H5 for the fiscal refresh calibration builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from populace.build.us_runtime import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    puf_tax_unit_donor_from_arrays,
    support_channel_column,
)
from populace.frame import Frame, WeightKind, Weights
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine
from populace.frame.units import US_SCHEMA

PERIOD = 2024
DATASET_FILENAME = "base_populace_us_2024_puf_support.h5"
SUMMARY_FILENAME = "base_populace_us_2024_puf_support.summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-h5", required=True, type=Path)
    parser.add_argument("--puf-h5", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-estimators", default=32, type=int)
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = out_dir / DATASET_FILENAME
    summary_path = out_dir / SUMMARY_FILENAME

    base = _load_frame(args.base_h5)
    expanded = clone_us_frame_for_puf_support(base)
    arrays = _read_h5_arrays(args.puf_h5)
    donor = puf_tax_unit_donor_from_arrays(arrays)
    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    PolicyEngineUSEngine().write_dataset(imputed, output_h5, period=PERIOD)

    summary = {
        "base_h5": str(args.base_h5.resolve()),
        "base_sha256": _sha256(args.base_h5),
        "puf_h5": str(args.puf_h5.resolve()),
        "puf_sha256": _sha256(args.puf_h5),
        "output_h5": str(output_h5),
        "output_sha256": _sha256(output_h5),
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "base_rows": _row_counts(base),
        "expanded_rows": _row_counts(imputed),
        "base_household_weight_total": float(base.weights_for("household").total),
        "expanded_household_weight_total": float(
            imputed.weights_for("household").total
        ),
        "channel_weight_totals": _channel_weight_totals(imputed),
        "puf_donor_rows": int(len(donor)),
        "puf_donor_columns": sorted(donor.columns.tolist()),
        "channel_output_totals": _channel_output_totals(imputed),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def _load_frame(path: Path) -> Frame:
    from policyengine_us.data import USSingleYearDataset

    dataset = USSingleYearDataset(file_path=str(path))
    tables = {
        "person": dataset.person.copy(),
        "household": dataset.household.copy(),
        "tax_unit": dataset.tax_unit.copy(),
        "spm_unit": dataset.spm_unit.copy(),
        "family": dataset.family.copy(),
        "marital_unit": dataset.marital_unit.copy(),
    }
    weights = tables["household"].pop("household_weight").to_numpy(dtype=np.float64)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
    )


def _read_h5_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return {name: np.asarray(dataset) for name, dataset in h5.items()}


def _row_counts(frame: Frame) -> dict[str, int]:
    return {entity: frame.n(entity) for entity in frame.entities}


def _channel_weight_totals(frame: Frame) -> dict[str, float]:
    household = frame.table("household")
    channel = support_channel_column("household")
    weights = pd.Series(frame.weights_for("household").values, index=household.index)
    return {
        str(name): float(weights.loc[group.index].sum())
        for name, group in household.groupby(channel, sort=True)
    }


def _channel_output_totals(frame: Frame) -> dict[str, dict[str, float]]:
    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    person_outputs = [
        "employment_income",
        "self_employment_income",
        "taxable_interest_income",
        "dividend_income",
        "qualified_dividend_income",
        "tax_exempt_interest_income",
        "short_term_capital_gains",
        "long_term_capital_gains",
        "charitable_cash_donations",
        "charitable_non_cash_donations",
        "real_estate_taxes",
        "home_mortgage_interest",
    ]
    tax_unit_outputs = [
        "interest_deduction",
        "state_withheld_income_tax",
        "tax_unit_partnership_s_corp_income",
    ]
    result: dict[str, dict[str, float]] = {
        BASE_ASEC_SUPPORT_CHANNEL: {},
        PUF_TAX_DETAIL_SUPPORT_CHANNEL: {},
    }
    for channel in result:
        person_mask = person[support_channel_column("person")] == channel
        tax_unit_mask = tax_unit[support_channel_column("tax_unit")] == channel
        for column in person_outputs:
            if column in person:
                result[channel][column] = float(
                    pd.to_numeric(person.loc[person_mask, column], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
        for column in tax_unit_outputs:
            if column in tax_unit:
                result[channel][column] = float(
                    pd.to_numeric(tax_unit.loc[tax_unit_mask, column], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
