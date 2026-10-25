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

  A third run, `outputs/stage_dev_run.head-split.json`, is the head once the gambling stakes are split (Vahid's review): Lottery Duty's base follows the National Lottery stakes and every other gambling base the betting and gaming stakes. `outputs/stage_dev_run.split-comparison.json` sets it against the restacked head: 33 of the 46 columns are byte-identical, and only the 13 gambling inputs move. The run also records the share of households with each written column positive, at prior weights.

  A fourth run, `outputs/stage_dev_run.head-main-rebase.json`, is the head rebased onto main after microcosm#1383 merged. `outputs/stage_dev_run.rebase-comparison.json` sets it against the split head: 26 of the 46 columns are byte-identical. #1383's last levelling rule moves the seven levelled divisions, recreation among them, and the 13 gambling inputs follow recreation; the alcohol and tobacco inputs do not move.
- **The national reconciliation** (`scripts/national_reconciliation.py`, `outputs/national_reconciliation.json`). One household holds every national level the committed declaration resolves to, run through policyengine-uk 2.151.0. Every duty is linear in its inputs, so this is the frame's duty at prior weights. Each OBR row is read through its own binding's variables against the value the national compile-parity receipt records from the pinned feed. Public figures only.
- **Who carries the duty** (`scripts/excise_incidence.py`, `outputs/excise_incidence.json`). From the rebased run, the reconciliation and the LCFS 2024-25 donor:
  - each duty over the spending that shares it out, and the share of households that carry it;
  - the share of donors and of frame households with each parent total and each part positive, and each part's rate within its parent, which shows where the frame loses incidence.

  Every donor share rests on 30 or more donors, which the script checks.
- **The pub lean of the spending rule** (`scripts/alcohol_channel_weights.py`, `outputs/alcohol_channel_weights.json`). Measured on the LCFS 2024-25 donor at survey weights: how far allocating alcohol volume by spending moves it, against channel weights for published on-trade volume shares of 20 to 35 percent, overall and by equivalised income decile.
- **The re-pin** (`scripts/repin_200bf6d.sh`). The commands that regenerated the vendored resources, the national and local references and membership, the local census, both compile-parity receipts and the coverage manifest. They ran on the national and local cuts of the pinned artifact that the runbook entry describes.

## The gambling participation top-up

- **Theta** (`scripts/theta_calibration.py`, `outputs/theta_calibration.json`). How a household's adults combine, fitted on the LCFS 2024-25 donor. The script reads the committed declaration and the vendored GSGB rows, and sets theta so that the step's adult rates, applied to the donors' own adults, give the diary's ratio of the two-adult to the one-adult household rate. The results are 0.897 for the National Lottery and 0.081 for betting and gaming; the declaration rounds them to 0.9 and 0.08. The ratio for three or more adults is the check, and it fits less well. Each household-size class rests on 30 or more donors with a stake, which the script checks.
- **Stage-only runs with and without the top-up** (`scripts/stage_run_gsgb.py`, `outputs/stage_run_gsgb.json`). The branch head `348d1209b` ran on the same spine, on the `272c987` feed, twice in one process: as declared, and with `top_up_gambling_participation` removed. The difference is therefore the step's alone. The output records:
  - every LCFS gate on the declared run (all nine pass) and the participation receipt;
  - for the Lottery Duty base and one betting and gaming input, the share of households carrying it and each equivalised-income decile's share of households and of the base, in both runs;
  - by region, the share carrying each key, beside GSGB's regional adult rates (tables B.2 and B.4). This is the out-of-sample check, since region is not a target.

  Every stratum lands on its share. 62.7% of households carry Lottery Duty and 29.8% the other gambling duties, against 10.7% and 2.0% without the step. The added regular players' mean stake equals the seen households', and the occasional players' is about a tenth of it.

  Two results are left for review:
  - carrying rises with income, so the betting and gaming base moves towards the top decile;
  - outside London, the regional pattern follows GSGB only weakly.
- **The re-pin to `272c987`** (`scripts/repin_272c987.sh`, with `scripts/quiet_footprint.sh`). The same steps as for `200bf6d`, on cuts of the artifact built from PolicyEngine/chronicle#391's head, whose tree the merge commit keeps. Each heavy step waits until no other process holds a 3 GB memory footprint (resident plus compressed and swapped), since `ps`'s resident size misses a swapped-out process.

## What is not here

- **Unit records.** Household dumps, the spine and the LCFS tabs stay out of the repository.
- **Column digests.** The runs record the sha256 of every column they write. `scripts/sanitize_outputs.py` drops those digests, with any donor minimum or maximum, before an output is copied here; the comparison keeps only the verdicts.
- **Small cells.** Every count in the receipts is a band ("30+") or withheld ("<10"), so no figure here rests on fewer than ten households.

## Re-running

The stage runs need the licensed LCFS 2024-25 tabs and the spine:

```bash
BASE_LABEL=base-74a2a378e HEAD_LABEL=head-restack scripts/run_stage_dev.sh
```

The reconciliation needs only the tree and its locked engine. The incidence script reads the split run, the reconciliation and the LCFS household tab:

```bash
python scripts/national_reconciliation.py outputs/national_reconciliation.json
```

```bash
python scripts/excise_incidence.py outputs/stage_dev_run.head-main-rebase.json outputs/national_reconciliation.json <lcfs dvhh tab> outputs/excise_incidence.json
```

The top-up's runs need the LCFS tabs, the spine and the GSGB rows taken from the pinned artifact (`grep '"gambling_commission.gsgb' consumer_facts.jsonl`):

```bash
python scripts/theta_calibration.py <lcfs 2024-25 tab directory> outputs/theta_calibration.json
```

```bash
python scripts/stage_run_gsgb.py <tree> <spine.h5> <gsgb rows jsonl> outputs/stage_run_gsgb.json
```
