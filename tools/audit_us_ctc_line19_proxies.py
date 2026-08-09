"""Audit competing SOI CTC line-19 proxies on a Microcosm US H5.

The IRS SOI historic table CTC amount maps to Form 1040 line 19: Schedule
8812 line 14, the smaller of the tentative CTC/ODC and Credit Limit Worksheet
A. This diagnostic compares the current PolicyEngine-US
``ctc_limiting_tax_liability`` proxy with a direct 2024 Worksheet A line-2
proxy and looser variants, by national SOI AGI bin.
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from microcosm.build.us_runtime import fiscal_targets as us_fiscal_targets
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from tools.build_us_fiscal_refresh_release import (
    _as_bound,
    _calculate_array,
    _dataset_from_frame,
    _load_frame,
)


WORKSHEET_A_LINE_2_VARIABLES: tuple[str, ...] = (
    # 2024 Schedule 8812 Credit Limit Worksheet A line 2:
    # Schedule 3 lines 1, 2, 3, 4, 5b, 6d, 6f, 6l, and 6m.
    "foreign_tax_credit",
    "cdcc",
    "non_refundable_american_opportunity_credit",
    "lifetime_learning_credit",
    "savers_credit",
    "energy_efficient_home_improvement_credit",
    "elderly_disabled_credit",
    "new_clean_vehicle_credit",
    "used_clean_vehicle_credit",
)


def _positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), 0.0)


def _tax_unit_weights(frame) -> np.ndarray:
    positions = _tax_unit_household_positions(frame)
    return frame.weights_for("household").values[positions]


def _tax_unit_household_positions(frame) -> np.ndarray:
    person = frame.table("person")
    household_ids = frame.table("household")["household_id"].to_numpy()
    tax_unit_ids = frame.table("tax_unit")["tax_unit_id"].to_numpy()
    membership = person[["person_tax_unit_id", "person_household_id"]].drop_duplicates()
    household_by_tax_unit = (
        membership.drop_duplicates("person_tax_unit_id")
        .set_index("person_tax_unit_id")["person_household_id"]
        .reindex(tax_unit_ids)
    )
    household_positions = pd.Series(
        np.arange(len(household_ids), dtype=np.int64),
        index=household_ids,
    )
    return household_positions.reindex(household_by_tax_unit.to_numpy()).to_numpy(
        dtype=np.int64
    )


def _target_rows(ledger_facts_path: Path) -> list[dict[str, object]]:
    facts = load_ledger_consumer_artifact(ledger_facts_path).facts
    rows: list[dict[str, object]] = []
    for fact in facts:
        if us_fiscal_targets._source_name(fact) != "irs_soi":
            continue
        if us_fiscal_targets._geography_level(fact) != "country":
            continue
        measure_id = us_fiscal_targets._measure_id(fact)
        if measure_id not in {"ctc_amount", "ctc_claims"}:
            continue
        source_record_id = us_fiscal_targets._source_record_id(fact)
        if not source_record_id:
            continue
        lower, upper = us_fiscal_targets._agi_bounds(fact)
        rows.append(
            {
                "name": source_record_id,
                "measure_id": measure_id,
                "group": us_fiscal_targets._str_at(
                    fact,
                    "layout",
                    "groupby_value_id",
                ),
                "target": us_fiscal_targets._numeric_value(fact),
                "agi_lower": _as_bound(lower),
                "agi_upper": _as_bound(upper),
            }
        )
    return rows


def _available_sum(
    simulation, variables: tuple[str, ...]
) -> tuple[np.ndarray, list[str]]:
    system = simulation.tax_benefit_system
    total: np.ndarray | None = None
    missing: list[str] = []
    for variable in variables:
        if variable not in system.variables:
            missing.append(variable)
            continue
        values = _calculate_array(simulation, variable).astype(np.float64)
        total = values if total is None else total + values
    if total is None:
        total = np.zeros_like(_calculate_array(simulation, "ctc"), dtype=np.float64)
    return total, missing


def audit(base_h5: Path, ledger_facts: Path, out_dir: Path) -> dict[str, object]:
    from policyengine_us import Microsimulation

    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(base_h5)
    simulation = Microsimulation(dataset=_dataset_from_frame(frame))
    weights = _tax_unit_weights(frame)

    ctc = _positive(_calculate_array(simulation, "ctc"))
    non_refundable_ctc = _positive(_calculate_array(simulation, "non_refundable_ctc"))
    pe_limit = _positive(_calculate_array(simulation, "ctc_limiting_tax_liability"))
    income_tax_before_credits = _positive(
        _calculate_array(simulation, "income_tax_before_credits")
    )
    refundable_ctc = _positive(_calculate_array(simulation, "refundable_ctc"))
    agi = _calculate_array(simulation, "adjusted_gross_income").astype(np.float64)

    worksheet_line_2, missing_worksheet_vars = _available_sum(
        simulation,
        WORKSHEET_A_LINE_2_VARIABLES,
    )
    residential_clean_energy, missing_residential = _available_sum(
        simulation,
        ("residential_clean_energy_credit",),
    )

    worksheet_a_limit = _positive(income_tax_before_credits - worksheet_line_2)
    worksheet_a_plus_residential_limit = _positive(
        income_tax_before_credits - worksheet_line_2 - residential_clean_energy
    )
    proxies = {
        "full_ctc": ctc,
        "policyengine_non_refundable_ctc": non_refundable_ctc,
        "current_pe_ctc_limit": np.minimum(ctc, pe_limit),
        "worksheet_a_line2_limit": np.minimum(ctc, worksheet_a_limit),
        "worksheet_a_line2_plus_residential_limit": np.minimum(
            ctc,
            worksheet_a_plus_residential_limit,
        ),
        "income_tax_before_credits_limit": np.minimum(ctc, income_tax_before_credits),
        "refundable_ctc": refundable_ctc,
    }

    target_rows = _target_rows(ledger_facts)
    by_bin_rows: list[dict[str, object]] = []
    for row in target_rows:
        mask = (agi >= row["agi_lower"]) & (agi < row["agi_upper"])
        for proxy_name, values in proxies.items():
            is_count = row["measure_id"] == "ctc_claims"
            simulated = (
                float(np.sum(weights[mask] * (values[mask] > 0)))
                if is_count
                else float(np.sum(weights[mask] * values[mask]))
            )
            target = float(row["target"])
            by_bin_rows.append(
                {
                    **row,
                    "proxy": proxy_name,
                    "estimate": simulated,
                    "relative_error": None
                    if target == 0
                    else (simulated - target) / target,
                }
            )

    component_rows = []
    component_variables = (
        *WORKSHEET_A_LINE_2_VARIABLES,
        "residential_clean_energy_credit",
        "ctc",
        "non_refundable_ctc",
        "refundable_ctc",
        "ctc_limiting_tax_liability",
        "income_tax_before_credits",
    )
    system = simulation.tax_benefit_system
    for variable in component_variables:
        if variable not in system.variables:
            component_rows.append({"variable": variable, "available": False})
            continue
        values = _positive(_calculate_array(simulation, variable))
        component_rows.append(
            {
                "variable": variable,
                "available": True,
                "weighted_amount": float(np.sum(weights * values)),
                "weighted_positive_count": float(np.sum(weights * (values > 0))),
            }
        )

    by_bin = pd.DataFrame(by_bin_rows)
    components = pd.DataFrame(component_rows)
    by_bin.to_csv(out_dir / "ctc_line19_proxy_by_agi.csv", index=False)
    components.to_csv(out_dir / "ctc_line19_components.csv", index=False)

    national = by_bin[by_bin["group"] == "all"].copy()
    summary = {
        "base_h5": str(base_h5),
        "ledger_facts": str(ledger_facts),
        "missing_worksheet_line_2_variables": missing_worksheet_vars,
        "missing_residential_variables": missing_residential,
        "national": national[
            [
                "measure_id",
                "proxy",
                "target",
                "estimate",
                "relative_error",
            ]
        ].to_dict(orient="records"),
        "component_totals": component_rows,
        "outputs": {
            "by_agi_csv": str(out_dir / "ctc_line19_proxy_by_agi.csv"),
            "components_csv": str(out_dir / "ctc_line19_components.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--ledger-facts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.base_h5, args.ledger_facts, args.out_dir)
    print(json.dumps(summary["national"], indent=2))
    print(f"Wrote {summary['outputs']['by_agi_csv']}")


if __name__ == "__main__":
    main()
