#!/usr/bin/env python3
"""Measure the base's child-support draw from the checkpoint before it.

The base's child-support stage (``child_support_post_clone``) draws both
child-support leaves onto the PUF tax-detail clones with one QRF, trained on
at most 5,000 ASEC people. Its default in-sample sign gate draws a rare class
below its rate on recipients, so the channel's nonzero share is a noisy draw.
This tool measures that draw on a real base, from the checkpoint the stage
starts from (the ``prior_year_income_post_clone`` record of a base checkpoint
directory), without running the base:

    uv run --no-sync python tools/diagnose_us_child_support_gate.py \
        "$RUN/base-checkpoints" --out child_support_gate_diagnosis.json

It reads the checkpoint directory and never writes to it. It reruns only the
child-support stage, in this process, several ways:

1. the base's own seed (it repeats the base's draw);
2. the next few seeds (``--seeds``), to show how far the share moves with the
   seed alone;
3. the base's seed with the cross-fitted sign gate
   (``GateCalibration.CROSS_FITTED``), unless ``--skip-cross-fitted``.

For each run it reports the weighted nonzero share of both leaves on both
support channels, by ASEC income year where the frame carries
``source_year``, whether the stage's signal gate passes, each leaf's gate
outcome, and how many payers and recipients the fit's training sample holds.
It reports aggregate shares and counts only, never a record's value; counts
below 10 read "<10". Each run takes minutes, in about the memory of the base
stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STAGE_BEFORE = "prior_year_income_post_clone"
OUTPUTS = ("child_support_expense", "child_support_received")
CHANNELS = ("asec", "puf_tax_detail")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _count(value: int) -> int | str:
    return "<10" if value < 10 else int(value)


def _load(root: Path):
    from microcosm.build.frame_checkpoint import load_frame_checkpoint

    context = json.loads((root / "stage_run_context.json").read_text())
    records = {record["stage"]: record for record in context["stage_records"]}
    if STAGE_BEFORE not in records:
        sys.exit(f"{root} has no completed {STAGE_BEFORE!r} stage.")
    record = records[STAGE_BEFORE]
    path = root / record["checkpoint_filename"]
    actual = _sha256(path)
    if actual != record["checkpoint_sha256"]:
        sys.exit(f"{path} digest {actual} differs from its run context.")
    frame = load_frame_checkpoint(path).frame
    return frame, context["run_config"], record["checkpoint_filename"]


def _shares(frame) -> dict[str, object]:
    from microcosm.build.us_runtime.child_support import us_child_support_signal_gate
    from microcosm.build.us_runtime.support_provenance import support_role_series

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    role = support_role_series(person, entity="person").to_numpy()
    year = person["source_year"].to_numpy() if "source_year" in person else None
    result: dict[str, object] = {}
    for output in OUTPUTS:
        values = pd.to_numeric(person[output], errors="coerce").to_numpy(np.float64)
        positive = np.isfinite(values) & (values > 0.0)
        by_channel: dict[str, object] = {}
        for channel in CHANNELS:
            mask = role == channel
            total = float(weights[mask].sum())
            entry: dict[str, object] = {
                "share": float(weights[mask & positive].sum()) / total
                if total
                else 0.0,
                "positive_rows": _count(int(np.count_nonzero(mask & positive))),
            }
            if year is not None:
                entry["by_income_year"] = {
                    str(value): (
                        float(weights[mask & positive & (year == value)].sum())
                        / float(weights[mask & (year == value)].sum())
                    )
                    for value in sorted(set(year[mask].tolist()))
                    if float(weights[mask & (year == value)].sum()) > 0.0
                }
            by_channel[channel] = entry
        result[output] = by_channel
    gate = us_child_support_signal_gate(frame)
    result["gate_passed"] = gate.passed
    result["gate_failures"] = list(gate.failures)
    return result


def _run(frame, *, seed: int, year: int, gate: str | None) -> dict[str, object]:
    from microcosm.build.us_runtime import child_support
    from microcosm.fit import RegimeGatedQRF

    captured: dict[str, object] = {}

    def factory(**kwargs):
        if gate is not None:
            kwargs["gate_calibration"] = gate
        model = RegimeGatedQRF(**kwargs)
        fit = model.fit

        def capturing_fit(*fit_args, **fit_kwargs):
            fitted = fit(*fit_args, **fit_kwargs)
            captured["fitted"] = fitted
            training = fit_args[0]
            captured["training"] = {
                "rows": int(len(training)),
                "payers": _count(int((training["child_support_expense"] > 0).sum())),
                "recipients": _count(
                    int((training["child_support_received"] > 0).sum())
                ),
            }
            return fitted

        model.fit = capturing_fit
        return model

    previous = child_support.QRF
    child_support.QRF = factory
    started = time.monotonic()
    try:
        out = child_support.with_us_child_support_inputs(
            frame, seed=seed, time_period=year
        )
    finally:
        child_support.QRF = previous
    fitted = captured.get("fitted")
    try:
        outcomes = fitted.gate_outcomes() if fitted is not None else None
    except Exception as error:  # report, never stop the diagnosis
        outcomes = f"unavailable: {error}"
    entry = {
        "seed": seed,
        "gate_calibration": gate or "in_sample (the build's)",
        "gate_outcomes": outcomes,
        "seconds": round(time.monotonic() - started, 1),
        "training_sample": captured.get("training"),
        **_shares(out),
    }
    del out
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--seeds", type=int, default=4, help="extra seeds to try")
    parser.add_argument("--skip-cross-fitted", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.checkpoint_dir.resolve()
    frame, run_config, checkpoint = _load(root)
    seed = int(run_config["seed"])
    year = int(run_config["target_year"])
    from microcosm.fit import GateCalibration

    runs = []
    plan = [(seed, None)]
    plan += [(seed + step, None) for step in range(1, args.seeds + 1)]
    if not args.skip_cross_fitted:
        plan.append((seed, GateCalibration.CROSS_FITTED))
    for run_seed, gate in plan:
        entry = _run(frame, seed=run_seed, year=year, gate=gate)
        runs.append(entry)
        expense = entry["child_support_expense"]["puf_tax_detail"]["share"]
        print(
            f"seed {run_seed} {entry['gate_calibration']}: PUF-channel expense "
            f"share {expense:.4f}, gate {'PASS' if entry['gate_passed'] else 'FAIL'} "
            f"({entry['seconds']} s)",
            flush=True,
        )
    report = {
        "checkpoint": checkpoint,
        "build_seed": seed,
        "target_year": year,
        "runs": runs,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        out = args.out.resolve()
        if root in out.parents:
            sys.exit("--out must be outside the checkpoint directory.")
        out.write_text(text + "\n")
        print(f"wrote {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
