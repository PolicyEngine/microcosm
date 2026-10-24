# Excise quantities: measurement scripts and outputs

Evidence for microcosm#1359, behind `experiments/1359-excise-quantities-receipts.md`. `receipts.json` holds the digests of the files here and of the spine the stage runs used.

## What was measured

- **Stage-only runs** (`scripts/stage_dev_run.py`, `scripts/run_stage_dev.sh`). The `lcfs_consumption` stage was re-run on a finished spine: microcosm#1333's `spine-b145-treat.h5`, 78,440 households. Both runs used policyengine-uk 2.151.0, and every LCFS stage-health gate ran on each run's receipts. The two runs:
  - `outputs/stage_dev_run.base-74a2a378e.json`: microcosm#1383's head `74a2a378e`, the branch base, on the `abf0921` feed;
  - `outputs/stage_dev_run.head-restack.json`: the branch head on the `200bf6d` feed, with the excise step declared.

  `outputs/stage_dev_run.comparison.json` (`scripts/compare_runs.py`) lists which of the base's 22 columns the head leaves byte-identical (13), and each other column's prior-weighted total in both runs. The other nine move through the re-pin, not the step:
  - petrol and diesel, by exactly the Consumer Trends revision (−4.227%);
  - the seven divisions #1383 holds at the shared frame-to-donor ratio, which the lower road fuel lowers (−0.149% each).

  The head adds the 24 excise inputs.
- **The pub lean of the spending rule** (`scripts/alcohol_channel_weights.py`, `outputs/alcohol_channel_weights.json`). Measured on the LCFS 2024-25 donor at survey weights: how far allocating alcohol volume by spending moves it, against channel weights for published on-trade volume shares of 20 to 35 percent, overall and by equivalised income decile.
- **The re-pin** (`scripts/repin_200bf6d.sh`). The commands that regenerated the vendored resources, the national and local references and membership, the local census, both compile-parity receipts and the coverage manifest. They ran on the national and local cuts of the pinned artifact that the runbook entry describes.

## What is not here

- **Unit records.** Household dumps, the spine and the LCFS tabs stay out of the repository.
- **Column digests.** The runs record the sha256 of every column they write. `scripts/sanitize_outputs.py` drops those digests, with any donor minimum or maximum, before an output is copied here; the comparison keeps only the verdicts.
- **Small cells.** Every count in the receipts is a band ("30+") or withheld ("<10"), so no figure here rests on fewer than ten households.

## Re-running

The stage runs need the licensed LCFS 2024-25 tabs and the spine:

```bash
BASE_LABEL=base-74a2a378e HEAD_LABEL=head-restack scripts/run_stage_dev.sh
```
