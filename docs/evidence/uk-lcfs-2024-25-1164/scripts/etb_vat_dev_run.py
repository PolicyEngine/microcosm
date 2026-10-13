"""Stage-only development run of etb_vat on a finished spine (microcosm#1164).

Usage, from the tree whose stage code runs (see run_stage_dev_at.sh):
  python <this> <spine.h5> <out.json>

Re-runs the tree's committed etb_vat stage, on the ETB tab its own spec pins
($PE/data/ukds/etb_1977_24/), on a finished UK spine, and records at the
spine's prior weights:
- the etb_vat stage-health gate on the stage's receipts;
- the weighted mean full-rate VAT expenditure rate, overall and by quintile
  of the recipient's HBAI net income (one definition for both trees, so the
  profiles compare);
- the weighted shares at the bounds, zero and one.

Disclosure control (UKDS EUL CD137 cl. 8, CD171 s5.2.1): the JSON holds
weighted means and shares over whole quintiles; the donor-rate receipt's
counts under ten are written "<10"; the household dump stays beside the JSON,
outside the repository.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import etb_vat
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

TREE = Path.cwd()
PE = Path(os.environ.get("PE", Path.home() / "Desktop" / "PolicyEngine"))
ETB = PE / "data" / "ukds" / "etb_1977_24"
RATE = "full_rate_vat_expenditure_rate"


def _count(value: object) -> object:
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10:
        return "<10"
    return value


def _tool():
    spec = importlib.util.spec_from_file_location(
        "tool", TREE / "tools" / "verify_uk_identity_stability.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def main() -> None:
    spine, out = Path(sys.argv[1]), Path(sys.argv[2])
    started = time.time()
    frame, _ = load_uk_national_frame(spine)
    frame = _tool()._engine_safe_frame(frame)
    spec = load_country_spec("uk")
    stage = spec.sources.stage_map()["etb_vat"]
    artifact = next(a for a in stage.artifacts if a.get("role") == "etb_household_tab")
    engine = PolicyEngineUKEngine()
    transform = etb_vat.UKETBVATStageTransform(
        stage=stage, engine=engine, etb_tab_path=ETB / artifact["filename"]
    )
    result = transform(frame)
    evidence = transform.checkpoint_metadata()["evidence"]
    gates = {}
    for gate in spec.gates.gates:
        parameters = dict(gate.parameters)
        if gate.gate != "stage_health" or parameters.get("stage") != "etb_vat":
            continue
        checked = uk_stage_health_gate(
            evidence=evidence,
            stage="etb_vat",
            check=str(parameters["check"]),
            parameters=parameters,
        )
        gates[gate.id] = {"passed": checked.passed, "failures": list(checked.failures)}
    weights = frame.weights_for("household").values.astype(float)
    rate = result.table("household")[RATE].to_numpy(dtype=float)
    materialized = engine.materialize(
        frame, ("hbai_household_net_income",), uk_time_period(frame)
    )
    income = np.asarray(materialized["hbai_household_net_income"], dtype=float)
    order = np.argsort(income, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    quintile = np.empty(len(income), dtype=int)
    quintile[order] = np.minimum((cumulative * 5).astype(int), 4)
    mass = float(weights.sum())
    report = {
        "spine": spine.name,
        "seconds": round(time.time() - started, 1),
        "gates": gates,
        "weighted_mean_rate": float(weights @ rate) / mass,
        "weighted_mean_rate_by_hbai_net_income_quintile": {
            str(q + 1): float(weights[quintile == q] @ rate[quintile == q])
            / float(weights[quintile == q].sum())
            for q in range(5)
        },
        "share_at_zero": float(weights[rate <= 0].sum()) / mass,
        "share_at_one": float(weights[rate >= 1].sum()) / mass,
        "donor_rate": {
            k: _count(v) for k, v in (evidence.get("donor_rate") or {}).items()
        },
        "donor_uprating": evidence.get("donor_uprating"),
        "predictors": list(etb_vat.UK_ETB_VAT_PREDICTORS),
    }
    out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
    pd.DataFrame(
        {
            "household_id": frame.table("household")["household_id"].to_numpy(),
            RATE: rate,
            "weight": weights,
        }
    ).to_parquet(out.with_suffix(".households.parquet"))
    print(json.dumps({k: report[k] for k in report if k != "donor_uprating"}, indent=1))


if __name__ == "__main__":
    main()
