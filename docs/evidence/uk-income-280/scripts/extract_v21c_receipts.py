"""Aggregate-only receipts of the v21c national run for docs/evidence/uk-income-280.

Usage (from the repository root, with the licensed run tree present):

    python docs/evidence/uk-income-280/scripts/extract_v21c_receipts.py \
        --run-dir runs/uk-623-first-calibrated/spine-assessment-v21c \
        --out docs/evidence/uk-income-280/v21c-receipts.json

Kept: calibration summary, terminal gate verdicts and the deferral register as
applied, the target errors of the income-anchor rows (OBR, HMRC liabilities,
SPI bands, the region tier's top band), the pass-2 score against the
incumbent, and the band-donor stage receipts (counts, weights, pool sizes,
band means). Withheld: every per-record value (per-band realised minimum and
maximum total income, weights of individual households).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROW_PREFIXES = (
    "obr.income_tax",
    "obr.universal_credit",
    "obr.fuel_duties_cars",
    "hmrc.itl.",
    "hmrc/employment_income_income_band",
    "hmrc/state_pension_income_band",
    "hmrc/self_employment_income_income_band",
    "hmrc/savings_interest_income_income_band",
    "hmrc/dividend_income_income_band",
    "hmrc/private_pension_income_income_band",
    "hmrc.spi_region.total_income_by_region_200000_plus",
    "hmrc.spi_region.income_tax_by_region_12570_15000",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_rows(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in diagnostics.get("targets") or diagnostics.get("target_rows") or []:
        name = str(row.get("name") or row.get("target") or "")
        if not name.startswith(ROW_PREFIXES):
            continue
        rows.append(
            {
                "name": name,
                "family": row.get("family"),
                "target": row.get("target")
                if isinstance(row.get("target"), (int, float))
                else row.get("compiled_target"),
                "estimate": row.get("final_estimate") or row.get("estimate"),
                "relative_error": row.get("final_relative_error")
                if row.get("final_relative_error") is not None
                else row.get("relative_error"),
            }
        )
    return rows


def _band_donor_receipts(record: dict[str, Any]) -> dict[str, Any]:
    evidence = (record.get("spine_provenance") or {}).get("stage_evidence") or {}
    out: dict[str, Any] = {}
    donors = evidence.get("spi_income_band_donors")
    if isinstance(donors, dict):
        out["spi_income_band_donors"] = {
            key: value
            for key, value in donors.items()
            if key not in {"households", "carriers_by_household"}
        }
    income = evidence.get("hmrc_spi_income_spine") or {}
    resample = income.get("band_donor_resample") if isinstance(income, dict) else None
    if isinstance(resample, dict):
        bands = []
        for band in resample.get("bands") or []:
            bands.append(
                {
                    key: value
                    for key, value in band.items()
                    if key
                    not in {"realized_min_total_income", "realized_max_total_income"}
                }
            )
        out["band_donor_resample"] = {
            **{k: v for k, v in resample.items() if k != "bands"},
            "bands": bands,
        }
    return out


def _verdict(entry: dict[str, Any]) -> Any:
    for key in ("passed", "verdict", "status", "outcome"):
        if key in entry:
            return entry[key]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    diagnostics = _load(run / "calibration_diagnostics.json")
    gates = _load(run / "microcosm_uk_2024_25.terminal_gates.json")
    score = _load(run / "score_vs_enhanced_frs.json")["score_vs_enhanced_frs"]
    record = _load(run / "build_record.json")
    manifest = _load(run / "rowwise_candidate_manifest.json")
    target_fit = (gates.get("gates") or {}).get("uk_target_fit") or {}
    payload = {
        "run": {
            "build_id": record.get("build_id"),
            "code": ((manifest.get("identity") or {}).get("code") or {}).get(
                "git_commit"
            ),
            "release_id": (record.get("run_config") or {}).get("release_id"),
            "release_candidate": manifest.get("release_candidate"),
            "calibration_year": (record.get("run_config") or {}).get(
                "calibration_year"
            ),
        },
        "calibration": {
            key: diagnostics.get(key)
            for key in (
                "n_records",
                "n_nonzero",
                "initial_loss",
                "final_loss",
                "fraction_within_10pct",
                "fraction_within_25pct",
                "effective_sample_size",
                "epochs",
                "learning_rate",
                "loss_weight_rule",
            )
            if key in diagnostics
        },
        "terminal_gates": {
            "blocked_at_phase": gates.get("blocked_at_phase"),
            "verdicts": {
                gate_id: _verdict(entry)
                for gate_id, entry in (gates.get("gates") or {}).items()
            },
            "target_fit_deferrals": (target_fit.get("details") or {}).get(
                "reviewed_exclusions"
            ),
        },
        "income_anchor_rows": _target_rows(diagnostics),
        "pass_2_vs_incumbent": {
            key: score.get(key)
            for key in (
                "candidate_target_wins",
                "incumbent_target_wins",
                "candidate_full_loss",
                "incumbent_full_loss",
                "target_wins_by_family",
            )
        }
        | {
            "n_scored": (score.get("incumbent_unresolvable_pruned") or {}).get(
                "n_scored"
            ),
            "n_pruned": (score.get("incumbent_unresolvable_pruned") or {}).get(
                "n_pruned"
            ),
        },
        "band_donors": _band_donor_receipts(record),
    }
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}: {len(payload['income_anchor_rows'])} income-anchor rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
