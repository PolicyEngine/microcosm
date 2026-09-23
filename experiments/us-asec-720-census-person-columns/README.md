# #720 layer 1: real-data receipts (2026-09-23)

Evidence for `docs/us-asec-census-person-columns.md`. The scripts ran from
`_recovered/scratch-backup/893/overnight-20260923/asec-720-fix/` (they resolve
`checkpoints/` next to themselves) on commit `39b8e7b63`:

- `run_source_construction.sh`: route A's base inputs and flags, `--stage source_construction` only.
  `source_construction_stage_run_context.json` / `source_construction_stage_profile.json`
  are its stage context and profile; `checkpoint_sha256.txt` the checkpoints it wrote.
- `verify_720.py` → `verify_720.json`: the #744 gate before (route A's raw-stage
  checkpoint, commit `47976be6c`) and after; per-vintage reporter counts; column-by-column
  equality with the 2026-08-23 corrected H5s; structural deltas.
- `compare_823_raw_stage.py` → `compare_823_raw_stage.json`: this branch's raw-stage frame
  against the 2026-08-23 corrected raw-stage frame, entity by entity.

Checkpoints are local build evidence, not release artifacts.
