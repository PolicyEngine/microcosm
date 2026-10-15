# UK local surface, first #1278 PR: measurement scripts

Evidence for microcosm#1278's first PR. The results are in [experiments/1278-uk-local-surface-receipts.md](../../../experiments/1278-uk-local-surface-receipts.md); this folder holds the scripts that produced the measured ones.

## Scripts

- `scripts/compare_twin_bands.py` (C1). Compares the council-tax bands of two spines built from the licensed inputs with the same recipe and seed, one on main and one on the branch. It aligns the base households (those that are not CGT copies) by lineage and reports, by nation, the households whose band lies outside the nation's bands, the bands that changed, and the columns that differ, with the households each moves on at all and by more than 0.1% and 1%. It writes counts only as bands (0, <10, 10-29, 30+) and no weighted figure.
- `scripts/cgt_stage_residual.py` (C8). Reads one built spine at its design weights and compares the liable gainers in each of the CGT stage's own age and region groups with the HMRC margins the stage rakes onto, folded by the stage's own functions. A group below 30 distinct survey source households is pooled with its neighbour, so no group can be recovered by subtraction. Its output stays outside the repository for now.

`tools/uk_target_surface_receipt.py --communal-only` produces C7's census sizing; it reads only public census facts from the pinned Chronicle feed.

## What is not here

- **Licensed inputs and unit records.** The spines, the survey files and the twin comparison's raw output stay outside the repository.
- **C8's output.** The script was run on main's spine, and its output stays outside the repository for now. A later report should use that one spine: the same figures on the branch's spine would differ only through the households the fix moved, so the two could be subtracted.
- **Weighted households by band before and after the fix.** The difference between the two spines' weighted band totals would isolate the households the fix moved, so neither spine's weighted band totals are reported.
