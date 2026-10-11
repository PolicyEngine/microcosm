# Consumption-tax levels: measurement scripts and outputs

Evidence for [PolicyEngine/microcosm#1167](https://github.com/PolicyEngine/microcosm/pull/1167) (microcosm#1113), filed on the review's ask that the scripts behind `experiments/1113-consumption-tax-levels-receipts.md` live in the repository. `receipts.json` carries the digests of the files here and of the spine the stage runs measured.

## What was measured

- **M2 and M3, desk measurements** (`scripts/m2_m3_desk.py`, `outputs/m2_m3_desk.json`). ONS Consumer Trends energy and road-fuel spending against DESNZ Energy Trends volume at QEP prices, and the road-fuel level against the OBR cars receipts. It reads the pinned Chronicle feed, the vendored resources, the 2026-09-30 national build and the LCFS 2023-24 household file. It runs no solve and no engine.
- **M4, the masked re-solve** (`scripts/m4_resolve.py`, `scripts/run_m4.sh`, `outputs/m4_report.json`, `outputs/solve_*.json`). The stored 2026-09-30 national problem (1,090 rows) re-solved twice with the doctrine options: with all rows bound, which reproduces the stored weights exactly, and with `obr.vat`, `obr.fuel_duties_cars` and the two ONS energy rows masked together. No arm masks one row alone.
- **Stage-only runs** (`scripts/stage_dev_run.py`, `scripts/run_stage_dev_at.sh`, `outputs/stage_dev_run*.json`). The `lcfs_consumption` stage re-run on the finished head-r3 spine of microcosm#1121 (63,956 households), with every lcfs stage-health gate run on its receipts:
  - `stage_dev_run.json`: Group A's head;
  - `stage_dev_run.group-b.json`: Group B's head;
  - `stage_dev_run.base-7f235941c.json`: the base tree's stage on the same spine;
  - `stage_dev_run.review-r2.json`: the head after the second review round.

The base run is what isolates this PR's effect. The spine already holds the stages after `lcfs_consumption`, so a re-run redraws most households, and a comparison with the spine's stored columns mixes that drift with the change. Compare a run with the base run, not with `before_head_r3`.

## What is not here

- **Unit records.** The household dumps the stage runs write (`*.households.parquet`) and the solved weight vectors (`weights_*.npy`) stay out of the repository.
- **Licensed inputs.** The LCFS tabs, the spine and the national H5s.
- **Exact donor minima.** `scripts/sanitize_outputs.py` drops the stage's `donor_floor` receipt, which records each donor column's exact minimum, before an output is copied here.
- **M1.** The design- and final-weight energy volumes on the 2026-09-30 build (93.0 and 260.1 TWh at design weights, 82.9 and 205.2 TWh at final weights) were computed during the audit of 2026-10-08, and no script was saved. `outputs/m4_report.json` holds the same build's spend at design and final weights.

## Re-running

The stage runs need the licensed LCFS 2023-24 tabs and the head-r3 spine:

```bash
scripts/run_stage_dev_at.sh <tree> <label>
```

It runs `<tree>`'s own stage code with one worktree's environment, waits up to 15 minutes for a quieter machine, and writes `stage_dev_run.<label>.json`. A run takes about 70 seconds and 5 GB.

## Provenance

Paths inside the scripts and outputs are machine-local. The scripts are copies of the ones run, formatted by this repository's formatter where they are Python.
