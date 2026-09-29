"""Build us-release-dry-run-margin-evidence.json from route A's artifacts.

The measurements behind the US release dry run's default margins and its
real-data check (experiments/us-release-dry-run-margin-evidence.md). It reads
machine-local route A artifacts (paths below), so it is a record of the method,
not a CI step. Usage::

    uv run python experiments/us-release-dry-run-margin-evidence.py \
        experiments/us-release-dry-run-margin-evidence.json

Sections of the output:

* ``release_timing``: the failed release's own ``build.timing``
  (``calibration_diagnostics.json``) and supervisor wall time, plus the CPU
  its process tree had used at the start of target materialization (from the
  supervisor's 30-second series).
* ``tail_share``: the release's own ``tail_concentration_gate`` on every column
  its tail gate checked that the raw base carries, at the base household
  weights and at the release's final calibrated household weights, beside the
  shares and carriers the release recorded.
* ``tail_share_d177_register_pairs``: the initial-weight and calibrated
  shares written in the d177 register's reasons.
* ``export_input_mass``: drift at base weights (the run's own preflight
  report) against drift at calibrated weights (the release's
  ``input_mass_parity.json``).
* ``dry_run_replay``: the dry run on that release's config
  (tools/build_us_fiscal_refresh_release.py --dry-run-gates-report), its
  report, the differential against the release's recorded tail surface, and
  the replay process's resource use.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from microcosm.build.gates import tail_concentration_gate

ROUTE_A = Path(
    "/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/overnight-20260923/"
    "route-a"
)
RUN = ROUTE_A / "run-310842b986d7"
BASE = RUN / "base-out/base_populace_us_2024_puf_support.h5"
RELEASE_ID = "populace-us-2024-0581707-310842b986d7-20260926T165326Z"
REL = RUN / "release-out" / RELEASE_ID / "releases" / RELEASE_ID
SUPERVISOR = RUN / "release-sup.failed-1790480658"
D177 = ROUTE_A / "qrf_tail_exclusions_routea_d177.json"
REPLAY = Path(
    "/Users/maxghenis/PolicyEngine/_reviews/us-release-dry-run-replay-20260927"
)
KW = {"top_k": 100, "max_top_share": 0.75, "min_nonzero_records": 500}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def release_timing() -> dict[str, object]:
    diagnostics = json.loads((REL / "calibration_diagnostics.json").read_text())
    timing = diagnostics["build"]["timing"]
    result = json.loads((SUPERVISOR / "RESULT.json").read_text())
    before = (
        timing["elapsed_through_calibration_seconds"]
        - timing["target_compilation_seconds"]
        - timing["calibration_seconds"]
    )
    rows = [
        row
        for row in csv.DictReader((RUN / "release.series.csv").open())
        if row["child_pid"] == "59213"
    ]
    # The supervisor writes ADMISSION.json when it launches the release.
    started = datetime.fromtimestamp(
        (SUPERVISOR / "ADMISSION.json").stat().st_mtime, UTC
    )
    boundary = started + timedelta(seconds=before)
    sample = min(
        rows,
        key=lambda row: abs(
            datetime.fromisoformat(row["utc"].replace("Z", "+00:00")) - boundary
        ),
    )
    return {
        "source": "calibration_diagnostics.json build.timing; supervisor RESULT.json",
        "wall_seconds_supervisor": result["wall_seconds"],
        "cpu_seconds_supervisor": result["cpu_seconds"],
        **timing,
        "before_target_materialization_seconds_at_most": before,
        "before_target_materialization_note": (
            "elapsed_through_calibration minus target_compilation and "
            "calibration: base load, input stages and pre-solve gates, plus the "
            "few statements between target compilation and the calibration timer"
        ),
        "tree_cpu_seconds_at_that_point": {
            "sample_utc": sample["utc"],
            "tree_cpu_seconds": float(sample["tree_cpu_seconds"]),
            "method": (
                "the supervisor's 30 s series sample nearest the launch "
                "(ADMISSION.json mtime) plus the seconds above"
            ),
        },
    }


def tail_shares() -> dict[str, object]:
    recorded = json.loads((REL / "qrf_tail_concentration.json").read_text())
    details = recorded["tail_concentration"]["details"]
    columns = sorted(
        set(recorded["surface"]["checked_sparse_columns"])
        | set(details["thin_columns"])
    )
    f = h5py.File(BASE, "r")
    household = f["household/table"].fields(["household_id", "household_weight"])[:]
    ids = household["household_id"]
    base_w = household["household_weight"].astype(np.float64)
    person_fields = set(f["person/table"].dtype.names)
    tax_fields = set(f["tax_unit/table"].dtype.names)
    person_cols = [c for c in columns if c in person_fields]
    tax_cols = [c for c in columns if c in tax_fields]
    not_in_base = [c for c in columns if c not in person_fields | tax_fields]
    person = f["person/table"].fields(
        ["person_household_id", "person_tax_unit_id", *person_cols]
    )[:]
    tax = f["tax_unit/table"].fields(["tax_unit_id", *tax_cols])[:]
    order = np.argsort(ids)

    def household_index(values: np.ndarray) -> np.ndarray:
        return order[np.searchsorted(ids[order], values)]

    person_hh = household_index(person["person_household_id"])
    first_member: dict[int, int] = {}
    for tax_unit, hh in zip(person["person_tax_unit_id"], person_hh, strict=True):
        first_member.setdefault(int(tax_unit), int(hh))
    tax_hh = np.array([first_member[int(t)] for t in tax["tax_unit_id"]])

    calibrated_ids = np.load(REL / "final_household_weight_ids.npy")
    calibrated_raw = np.load(REL / "final_household_weights.npy").astype(np.float64)
    calibrated_w = np.full(base_w.shape, np.nan)
    calibrated_w[household_index(calibrated_ids)] = calibrated_raw
    assert np.isfinite(calibrated_w).all()

    def grade(weights: np.ndarray):
        values = {c: person[c].astype(np.float64) for c in person_cols}
        values |= {c: tax[c].astype(np.float64) for c in tax_cols}
        per = {c: weights[person_hh] for c in person_cols}
        per |= {c: weights[tax_hh] for c in tax_cols}
        return tail_concentration_gate(values, per, **KW).details

    at_base, at_calibrated = grade(base_w), grade(calibrated_w)

    def carriers(gate: dict, column: str) -> int | None:
        return gate["carrier_counts"].get(column, gate["thin_columns"].get(column))

    rows = []
    for column in sorted(set(person_cols) | set(tax_cols)):
        base = at_base["top_share"].get(column)
        calibrated = at_calibrated["top_share"].get(column)
        rows.append(
            {
                "column": column,
                "entity": "person" if column in person_cols else "tax_unit",
                "base_share": base,
                "calibrated_share_recomputed": calibrated,
                "calibrated_share_recorded": details["top_share"].get(column),
                "shift": None
                if base is None or calibrated is None
                else calibrated - base,
                "carriers_base": carriers(at_base, column),
                "carriers_calibrated_recomputed": carriers(at_calibrated, column),
                "carriers_recorded": carriers(details, column),
            }
        )
    shifts = [row["shift"] for row in rows if row["shift"] is not None]
    differences = {
        row["column"]: abs(
            row["calibrated_share_recomputed"] - row["calibrated_share_recorded"]
        )
        for row in rows
        if row["calibrated_share_recomputed"] is not None
    }
    return {
        "method": (
            "microcosm.build.gates.tail_concentration_gate (top_k=100, "
            "max_top_share=0.75, min_nonzero_records=500) on each column the "
            "release's tail gate checked that the raw base carries, person and "
            "tax-unit weights broadcast from household weights, once at the base "
            "household weights and once at the release's final calibrated weights"
        ),
        "columns_checked_by_the_release": len(columns),
        "not_in_raw_base": not_in_base,
        "columns_with_both_shares": len(shifts),
        "calibrated_share_mismatches_over_1e-9": {
            column: difference
            for column, difference in differences.items()
            if difference > 1e-9
        },
        "carriers_equal_base_vs_calibrated_recomputed": all(
            row["carriers_base"] == row["carriers_calibrated_recomputed"]
            for row in rows
        ),
        "carriers_recomputed_vs_recorded_mismatches": {
            row["column"]: [
                row["carriers_calibrated_recomputed"],
                row["carriers_recorded"],
            ]
            for row in rows
            if row["carriers_calibrated_recomputed"] != row["carriers_recorded"]
        },
        "shift_min": min(shifts),
        "shift_max": max(shifts),
        "shift_median": sorted(shifts)[len(shifts) // 2],
        "rows": rows,
    }


def d177_pairs() -> list[dict[str, object]]:
    pattern = re.compile(
        r"Share (?P<calibrated>0\.\d+).*?initial-weight share about "
        r"(?P<initial>0\.\d+)"
    )
    pairs = []
    for column, reason in sorted(json.loads(D177.read_text()).items()):
        match = pattern.search(reason)
        if match:
            pairs.append(
                {
                    "column": column,
                    "initial": float(match["initial"]),
                    "calibrated": float(match["calibrated"]),
                    "shift": float(match["calibrated"]) - float(match["initial"]),
                }
            )
    return pairs


def export_mass() -> dict[str, object]:
    preflight = json.loads((RUN / "preflight-base.json").read_text())
    (check,) = [
        c for c in preflight["checks"] if c["name"] == "export_mass_parity_risk"
    ]
    base = {row["column"]: row["drift"] for row in check["rows"]}
    worst = json.loads((REL / "input_mass_parity.json").read_text())[
        "export_vs_base_frame"
    ]["details"]["worst_drifts"]
    rows = [
        {
            "column": column,
            "drift_at_base_weights": base[column],
            "drift_at_calibrated_weights": drift,
            "shift": drift - base[column],
        }
        for column, drift in sorted(worst.items(), key=lambda kv: -abs(kv[1]))
    ]
    return {
        "method": (
            "drift at base weights from the run's own preflight report (raw "
            "base, same --export-input-mass-reference-h5) against the release's "
            "export drift at calibrated weights (input_mass_parity.json "
            "export_vs_base_frame worst_drifts, its 20 largest)"
        ),
        "shift_min": min(row["shift"] for row in rows),
        "shift_max": max(row["shift"] for row in rows),
        "rows": rows,
    }


def replay() -> dict[str, object]:
    out = REPLAY / "out"
    validation = json.loads((out / "validation.json").read_text())
    report = json.loads((out / "dry_run_d177.json").read_text())
    time_block = (REPLAY / "replay.log").read_text()

    def measured(label: str) -> float:
        match = re.search(rf"^\s*([0-9.]+)\s+{label}\s*$", time_block, re.MULTILINE)
        return float(match.group(1)) if match else float("nan")

    wall = re.search(r"([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys", time_block)
    real, user, system = (float(value) for value in wall.groups())
    recorded = json.loads((REL / "qrf_tail_concentration.json").read_text())
    mismatch = recorded["surface"]["register_mismatch"]
    refused = sorted(
        {
            line.split(":", 1)[0]
            for line in recorded["tail_concentration"]["failures"]
            if not line.startswith("Stale reviewed exclusions")
        }
        | set(mismatch["stale"])
        | set(mismatch["unused"])
    )
    (tail,) = [c for c in report["checks"] if c["name"] == "qrf_tail_register"]
    rows = {row["column"]: row for row in tail["rows"]}
    return {
        "command": "tools/build_us_fiscal_refresh_release.py <route A release-config argv> "
        "--qrf-tail-concentration-exclusions qrf_tail_exclusions_routea_d177.json "
        "--dry-run-gates-report ...",
        "report_sha256": sha256(out / "dry_run_d177.json"),
        "build_commit": report["inputs"]["build_commit"],
        "exit_code": report["exit_code"],
        "checks": {check["name"]: check["status"] for check in report["checks"]},
        "qrf_tail_register_failures": [
            check["failures"]
            for check in report["checks"]
            if check["name"] == "qrf_tail_register"
        ][0],
        "columns_the_release_refused": {
            column: {
                "dry_run_status": rows[column]["status"],
                "dry_run_possible_verdicts": rows[column]["possible_verdicts"],
                "share_at_base_weights": rows[column]["top_share_at_base_weights"],
                "share_recorded_by_the_release": recorded["tail_concentration"][
                    "details"
                ]["top_share"].get(column),
            }
            for column in refused
        },
        "staged_frame_sha256": report["inputs"]["staged_frame_sha256"],
        "target_frame_checkpoint": report["inputs"]["target_frame_checkpoint"],
        "seconds_to_stop_point": report["inputs"]["seconds_to_stop_point"],
        # Measured by the replay script around the dry run's checks alone; the
        # report's own seconds_to_stop_point (above) was taken after them at
        # 21c1f9ba3, so the stop point came at most this much earlier.
        "replay_measured_check_seconds": validation["dry_run_checks_seconds"],
        "differential_against_release_record": validation["calibrated_differential"],
        "process": {
            "wall_seconds": real,
            "user_cpu_seconds": user,
            "system_cpu_seconds": system,
            "cpu_seconds_per_wall_second": (user + system) / real,
            "involuntary_context_switches": measured("involuntary context switches"),
            "peak_memory_footprint_bytes": measured("peak memory footprint"),
        },
    }


if __name__ == "__main__":
    receipt = {
        "schema_version": 2,
        "generated_by": "experiments/us-release-dry-run-margin-evidence.py",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": f"route A 310842b986d7, release {RELEASE_ID}",
        "inputs": {
            "base_h5_sha256": sha256(BASE),
            "qrf_tail_concentration_json_sha256": sha256(
                REL / "qrf_tail_concentration.json"
            ),
            "final_household_weights_npy_sha256": sha256(
                REL / "final_household_weights.npy"
            ),
            "final_household_weight_ids_npy_sha256": sha256(
                REL / "final_household_weight_ids.npy"
            ),
            "input_mass_parity_json_sha256": sha256(REL / "input_mass_parity.json"),
            "calibration_diagnostics_json_sha256": sha256(
                REL / "calibration_diagnostics.json"
            ),
            "preflight_base_json_sha256": sha256(RUN / "preflight-base.json"),
            "d177_register_sha256": sha256(D177),
        },
        "release_timing": release_timing(),
        "tail_share": tail_shares(),
        "tail_share_d177_register_pairs": d177_pairs(),
        "export_input_mass": export_mass(),
        "dry_run_replay": replay(),
    }
    Path(sys.argv[1]).write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: receipt[k] for k in ("release_timing",)}, indent=1))
