"""Build J DENSE arm — STEP D2: direct dense CSR solve from the frame checkpoint.

Adapted from Build H's experiments/learned_selection_prior/src/
direct_replay_dense_weights_buildh.py (the proven ~18-min direct solve). ONLY the
runtime root (_buildh-runtime -> _buildj-runtime), checkpoint root (buildh-dense ->
buildj-dense), and the base sha (read dynamically from the Build J base) change; the
feed (v8 94b7155f), registry (d71c59514e3a), pe-us (1.764.6), and every solver
hyperparameter are IDENTICAL to Build H, so the Build J dense final_loss is directly
comparable to Build H's dense parent.

Why this exists (codex review §3): the monolithic release path re-materialises the ACA
base frame + five JCT reform vectors, then holds a multi-tens-of-GB footprint while it
solves + exports -> macOS jetsam SIGKILLs it (rc=137). The buildj-dense frame checkpoint
(written atomically DURING the D1 release, WITH the scf_wealth asset stage) already stores
the fully materialised post-ACA/post-target frame; loading it via the tool's own
_read_target_frame_checkpoint skips ALL materialisation, leaving only the deterministic
CSR solve (~18 min, bounded footprint — the 88 GB full-pool-solve risk avoided). Saves the
dense weights AND a warm-start calibration.npz that Step D3 (the warm-start release) consumes.

--verify-only: compile the registry + load the frame (seconds) and stop; de-risks the
identity match before the ~18-min solve. STAGING/LOCAL ONLY.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

WT = Path("/Users/maxghenis/PolicyEngine/_worktrees/populace-build-j-recert")
sys.path.insert(0, str(WT / "tools"))
import build_us_fiscal_refresh_release as release  # noqa: E402

from populace.calibrate import calibrate  # noqa: E402

RT = Path("/Users/maxghenis/PolicyEngine/_buildj-runtime")
CKPT = RT / "checkpoints/buildj-dense/target_frame_checkpoint.h5"
BASE = RT / "out/base-j/base_populace_us_2024_puf_support.h5"
FACTS = Path("/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildh_v8.jsonl")
FACTS_SHA = "94b7155f7ca9e2de32ddb3a0add2fff2d8c66e73147fe5bd112cff3ba69b1669"
PEUS = "1.764.6"
# Build J dense registry = compile(v8) + the #387 RI substitution (5,534 specs).
# The version is computed live and checked against the checkpoint identity.
OUT = RT / "out/buildj-run/densewts_direct"
OUT.mkdir(parents=True, exist_ok=True)

# Build H dense anchor (Build J registry is IDENTICAL d71c59514e3a -> comparable).
BUILDH_DENSE_FINAL_LOSS = 0.04018
SANE_LO, SANE_HI = 0.010, 0.100


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    base_sha = _sha256(BASE)
    log(f"Build J base sha = {base_sha}")

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
    from populace.build.us_runtime import apply_us_medicaid_enrollment_substitutions
    registry1, sub_records = apply_us_medicaid_enrollment_substitutions(registry0)
    target_specs = registry1.specs
    active_registry = release.TargetRegistry(target_specs, country="us")
    log(f"registry version = {active_registry.version}  n_specs={len(target_specs)} "
        f"(expect 5534 = 5533 + RI substitution)")
    log(f"substitutions: {[(r['state_fips'], r['applied'], r['stale']) for r in sub_records]}")
    if len(target_specs) != 5534:
        log("FATAL spec count != 5534. STOP.")
        sys.exit(3)

    identity = release._target_frame_checkpoint_identity(
        base_dataset_sha256=base_sha, policyengine_us_version=PEUS, seed=0,
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
        log("verify-only: frame + registry loaded from checkpoint OK. stopping.")
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
    log(f"DONE. final_loss={result.final_loss:.10f}  initial_loss={result.initial_loss:.6f}")
    log(f"  Build H dense anchor final_loss={BUILDH_DENSE_FINAL_LOSS:.6f} (same registry -> comparable)")
    log(f"  ESS={ess:.2f}  n_nonzero={(w>0).sum()}  n_households={w.shape[0]}")

    np.save(OUT / "dense_household_weight.npy", w)
    np.save(OUT / "dense_initial_weight.npy", w0)
    np.savez(
        OUT / "populace_us_2024_calibration.npz",
        household_weight=w,
        initial_household_weight=w0,
    )
    log(f"saved dense weights + warm-start npz to {OUT}  ({time.time()-t0:.1f}s)")

    sane = np.isfinite(result.final_loss) and SANE_LO <= result.final_loss <= SANE_HI
    log(f"SANITY {'PASS' if sane else 'FAIL'} (final_loss in [{SANE_LO}, {SANE_HI}])")
    if not sane:
        sys.exit(5)


if __name__ == "__main__":
    main()
