# build/us — provenance snapshot

The scripts that produced `populace_us_2024.h5`, kept verbatim for audit. They
are a **snapshot, not a turnkey pipeline**: they reference build-machine paths
(worktrees, cached surfaces, a scoring-harness checkout via
`SCORING_HARNESS_SRC`) and ran against `populace` installed from this
repository plus `policyengine-us==1.723.0`.

- `extract_target_surface.py` — extracts the raw (unscaled) PE-native target
  surface (3,704 targets) for the pool.
- `build_populace_us_dataset.py` — calibrates the pool's household weights to
  that surface with `populace.calibrate` (hard `max_weight_ratio=50`) and
  writes the published `USSingleYearDataset`, including the entity-table
  surgeries it documents.
- `bounded_recal_experiment.py` + `bounded_recal_results.json` — the cap sweep
  that selected the 50× bound.
- `hf_dataset_card.md` — the dataset card published to
  `policyengine/populace-us` on the Hugging Face Hub.
