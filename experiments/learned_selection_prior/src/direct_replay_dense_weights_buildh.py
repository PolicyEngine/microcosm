"""populace#299 Build H — direct dense CSR solve from the frame checkpoint.

Adapted from Build G's experiments/learned_selection_prior/src/
direct_replay_dense_weights.py (the proven 18-min direct solve). ONLY the feed
(v7 -> v8), registry version (e035007bba5b -> d71c59514e3a), and runtime root
(_buildg-runtime -> _buildh-runtime) change.

Why this exists: the monolithic release path (build_us_fiscal_refresh_release.py)
re-materialises the ACA base frame AND the five JCT reform income-tax vectors on
every run, then holds a multi-tens-of-GB footprint for 76-121 min while it solves
and exports -> macOS jetsam SIGKILLs it (rc=137, six times). The buildh-dense
frame checkpoint already stores the fully materialised post-ACA, post-target frame
for registry d71c59514e3a (identity_json verified). Loading it directly via the
tool's own `_read_target_frame_checkpoint` skips ALL materialisation, so the only
work left is the deterministic CSR solve (~18 min, bounded footprint). We then
save the recovered dense weights AND a warm-start calibration.npz that Step 2
(the release rerun) consumes.

--verify-only: compile the registry + load the frame from the checkpoint
(seconds), print the registry version + checkpoint identity + initial fit, and
stop. De-risks the identity match before committing to the ~18-min solve.

Build H registry differs from Build G (5533 vs 5521 specs: +10 SOI Table 1.4
signed income/loss legs, +2 itemizer-masked Table 2.1 home-mortgage targets), so
the final loss will NOT equal Build G's 0.0413937. It should land in the same
order of magnitude (~0.04). We report it and sanity-band it; we do NOT assert
equality.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "tools")
import build_us_fiscal_refresh_release as release  # noqa: E402

from populace.calibrate import calibrate  # noqa: E402

RT = Path("/Users/maxghenis/PolicyEngine/_buildh-runtime")
CKPT = RT / "checkpoints/buildh-dense/target_frame_checkpoint.h5"
FACTS = RT / "inputs/consumer_facts_buildh_v8.jsonl"
FACTS_SHA = "94b7155f7ca9e2de32ddb3a0add2fff2d8c66e73147fe5bd112cff3ba69b1669"
BASE_SHA = "18833fb68e60ee74461608d81a5c5ab7d52435e17026d9e3b062d9de18d6871f"
PEUS = "1.764.6"
REG_VERSION_EXPECTED = "d71c59514e3a"
OUT = RT / "out/buildh-run/densewts_direct"
OUT.mkdir(parents=True, exist_ok=True)

# Build G dense anchor (same order of magnitude expected, NOT equality).
BUILDG_DENSE_FINAL_LOSS = 0.041393692879099726
# Sanity band for the Build H dense final loss (same order as Build G).
SANE_LO, SANE_HI = 0.010, 0.100


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    log("loading v8 feed + compiling target registry (no microsim) ...")
    ledger = release.load_ledger_consumer_artifact(
        FACTS, expected_facts_sha256=FACTS_SHA
    )
    registry0 = release.compile_us_fiscal_target_registry(
        ledger.facts, target_period=release.PERIOD,
        include_congressional_district_targets=False,
        congressional_district_vintage_crosswalk=None,
        age_targets=True, allow_unaged_dollar_targets=False,
        extra_support_exclusions=None)
    target_specs = registry0.specs
    # Match the release tool: identity uses TargetRegistry(specs).version.
    active_registry = release.TargetRegistry(target_specs, country="us")
    log(f"registry version = {active_registry.version}  "
        f"(expected {REG_VERSION_EXPECTED})  n_specs={len(target_specs)}")
    if active_registry.version != REG_VERSION_EXPECTED:
        log(f"FATAL registry version mismatch — got {active_registry.version}, "
            f"expected {REG_VERSION_EXPECTED}. Feed/compile drift; STOP.")
        sys.exit(3)

    identity = release._target_frame_checkpoint_identity(
        base_dataset_sha256=BASE_SHA, policyengine_us_version=PEUS, seed=0,
        target_period=release.PERIOD, target_registry_version=active_registry.version,
        congressional_district_vintage_crosswalk_sha256=None)
    log(f"checkpoint identity: {identity}")

    loaded = release._read_target_frame_checkpoint(
        CKPT, identity=identity, target_specs=target_specs,
        gate_congressional_district_targets=False)
    if loaded is None:
        log("CHECKPOINT MISS — identity mismatch; cannot fast-load. STOP + report.")
        sys.exit(4)
    frame, registry, _comp = loaded
    log(f"checkpoint HIT: frame n(household)={frame.n('household')} "
        f"n_targets={len(registry)}")

    # Preserve the historical objective for a faithful replay. The current
    # release builder adds the populace#462 critical-row overlay.
    tlw = release._fiscal_target_loss_weights(
        registry,
        critical_target_loss_multiplier=1.0,
    )
    if args.verify_only:
        log("verify-only: frame + registry loaded from checkpoint OK. "
            "Ready for the solve; stopping.")
        return

    log("calibrating (adam, 1500 epochs, lr 0.02, mass=conserve, ratio 5.0, "
        "l2=0, seed 0) — ~18 min ...")
    result = calibrate(
        frame, registry.to_target_set(), epochs=1500, learning_rate=0.02,
        max_weight_ratio=5.0, seed=0, mass="conserve", l2_lambda=0.0,
        target_loss_weights=tlw, target_loss_cap=release.US_FISCAL_TARGET_LOSS_CAP,
        warm_start_weights=None)
    w = np.asarray(result.weights, dtype=np.float64)
    w0 = np.asarray(result.initial_weights, dtype=np.float64)
    ess = (w.sum() ** 2) / (w ** 2).sum()
    log(f"DONE. final_loss={result.final_loss:.10f}  "
        f"initial_loss={result.initial_loss:.6f}")
    log(f"  Build G dense anchor final_loss={BUILDG_DENSE_FINAL_LOSS:.6f} "
        f"(same order expected; registry differs 5533 vs 5521)")
    log(f"  ESS={ess:.2f}  n_nonzero={(w>0).sum()}  n_households={w.shape[0]}")

    # Save recovered dense weights.
    np.save(OUT / "dense_household_weight.npy", w)
    np.save(OUT / "dense_initial_weight.npy", w0)
    # Warm-start calibration NPZ for Step 2 (release rerun via
    # --warm-start-calibration-npz). Loader requires strictly-positive
    # household_weight + initial_household_weight matching the frame's base
    # weights (== result.initial_weights).
    np.savez(
        OUT / "populace_us_2024_calibration.npz",
        household_weight=w,
        initial_household_weight=w0,
    )
    log(f"saved dense weights + warm-start npz to {OUT}  ({time.time()-t0:.1f}s)")

    sane = np.isfinite(result.final_loss) and SANE_LO <= result.final_loss <= SANE_HI
    log(f"SANITY {'PASS' if sane else 'FAIL'} "
        f"(final_loss in [{SANE_LO}, {SANE_HI}] band)")
    if not sane:
        sys.exit(5)


if __name__ == "__main__":
    main()
