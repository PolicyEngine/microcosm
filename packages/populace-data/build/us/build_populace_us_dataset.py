"""Build the publishable populace-US dataset: calibrate the v5 pool's household
weights to the full PE-native target surface via populace.calibrate, then write
the calibrated weights back into the USSingleYearDataset H5.

The pool (support) already beats eCPS in the symmetric-refit comparison; this
bakes in its own calibrated weights so the published dataset hits the
administrative targets out of the box.
"""

import shutil
import sys
import time

import numpy as np
import pandas as pd

from populace.calibrate import Target, TargetSet, calibrate
from populace.frame import EntitySchema, Frame, WeightKind, Weights

ART = "/Users/maxghenis/.claude-worktrees/microplex-spec-build/artifacts"
POOL = f"{ART}/spec_candidate_full_2024_v5_beats/candidate_policyengine_us.h5"
SURFACE = f"{ART}/v5_target_surface_raw.npz"
OUT = f"{ART}/populace_us_2024.h5"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    log("loading cached PE-native target surface...")
    surf = np.load(SURFACE, allow_pickle=True)
    A = surf["A"].astype(np.float64)  # (n_households, n_targets), RAW units
    b = surf["b"].astype(np.float64)  # (n_targets,), RAW units — the scaled
    # surface (targets ~0.01) breaks the eCPS +1 loss and collapses weights
    w0 = surf["w0"].astype(np.float64)
    names = [str(x) for x in surf["names"]]
    n_hh, n_targets = A.shape
    log(f"  {n_targets} targets over {n_hh} households; w0 sum {w0.sum()/1e6:.1f}M")

    # Minimal Frame: one person per household, the household entity weighted by
    # the pool's current weights. Calibration moves the HOUSEHOLD weights; the
    # target rows are the (precompiled, PE-derived) per-household contributions.
    household = pd.DataFrame({"household_id": np.arange(n_hh, dtype=np.int64)})
    person = pd.DataFrame(
        {
            "person_id": np.arange(n_hh, dtype=np.int64),
            "person_household_id": np.arange(n_hh, dtype=np.int64),
        }
    )
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=w0.copy(), kind=WeightKind.DESIGN)},
    )

    # Each PE-native target is a sum constraint whose per-household measure is the
    # cached A column (aligned to the household row order, verified identical w0).
    log("building target set from the PE-native surface...")
    targets = TargetSet(
        tuple(
            Target(
                name=names[t],
                entity="household",
                aggregation="sum",
                value=float(b[t]),
                measure=(lambda _f, col=A[:, t].copy(): col),
            )
            for t in range(n_targets)
        )
    )

    log("calibrating (torch APG over log-weights, full surface)...")
    t0 = time.time()
    result = calibrate(
        frame,
        targets,
        weight_entity="household",
        epochs=3000,
        learning_rate=0.15,
        mass="free",
        # Hard per-record bound (the landmine guard). Unbounded calibration
        # reached a 1.29M max household weight (16 records > 500k); ratio=50
        # costs ~0.1pt of within-10% and caps the max at ~318k with none
        # above 500k (see bounded_recal_results.json for the sweep).
        max_weight_ratio=50.0,
        seed=0,
    )
    log(f"  done in {time.time()-t0:.0f}s")
    log(f"  loss {result.initial_loss:.4f} -> {result.final_loss:.4f}")
    log(f"  targets within 10%: {result.fraction_within_10pct*100:.1f}%")
    cw = result.frame.resolve_weights("household").values
    log(f"  calibrated weight sum {cw.sum()/1e6:.1f}M (was {w0.sum()/1e6:.1f}M)")
    log(f"  weight range: {cw.min():.2f} .. {cw.max():.0f}")

    # Write calibrated weights back into the pool H5 (via PolicyEngine's loader so
    # the output stays a valid USSingleYearDataset).
    log("writing calibrated dataset...")
    from policyengine_us.data import USSingleYearDataset

    shutil.copy(POOL, OUT)
    ds = USSingleYearDataset(file_path=OUT)
    assert len(ds.household) == n_hh, "household count mismatch on write-back"
    ds.household["household_weight"] = cw
    # The pool carries a bookkeeping `year` column in several entity tables;
    # it is not a PolicyEngine variable, and Microsimulation's loader refuses
    # to flatten a column that appears in more than one entity.
    for entity_name in ("person", "household"):
        table = getattr(ds, entity_name)
        if "year" in table.columns:
            del table["year"]
    # interest_deduction is a tax-unit variable but the pool stores it on
    # persons (head carries the value, other members zero — at most one
    # nonzero contributor per unit, so a group sum is exact). Microsimulation
    # rejects a known variable stored at the wrong entity length.
    if "interest_deduction" in ds.person.columns:
        unit_sum = (
            pd.Series(ds.person["interest_deduction"].to_numpy(dtype=np.float64))
            .groupby(ds.person["person_tax_unit_id"].to_numpy())
            .sum()
        )
        ds.tax_unit["interest_deduction"] = (
            unit_sum.reindex(ds.tax_unit["tax_unit_id"].to_numpy())
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        del ds.person["interest_deduction"]
    ds.save(OUT)
    log(f"  saved {OUT}")

    # persist diagnostics
    np.savez_compressed(
        f"{ART}/populace_us_2024_calibration.npz",
        calibrated_weights=cw,
        initial_loss=result.initial_loss,
        final_loss=result.final_loss,
        within_10pct=result.fraction_within_10pct,
        loss_trajectory=result.loss_trajectory,
    )
    log("CALIBRATION COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
