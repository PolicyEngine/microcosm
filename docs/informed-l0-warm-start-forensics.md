# Informed-L0 / warm-start reconstruction — Phase 1 forensics (populace#328)

This records the Phase 1 forensic findings for populace#328 — whether the frozen
57,240-record support behind the certified live default can be recovered, and how
its record identities map onto a freshly-rebuilt base. It is the evidence base for
the design in `informed-l0-warm-start-design.md`.

All Hugging Face reads here were **read-only** against `policyengine/populace-us`.
No prod write path, tag, or `latest.json` was touched.

## Question

The certified live default
`populace-us-2024-sparse-l0-refit-57k-71a0887-national-only-20260701` is, per its
manifest, a 1,500-epoch dense polish (`method = "dense_no_l0"`, final loss 0.0440)
of a **frozen 57,240-record support** warm-started from
`out/sparse-default-release-20260701/refit-selected-current-surface-a7a77c4-20260701/artifacts/a7a77c4_reconstructed_warm_start.npz`.
That selection artifact was never committed and descends from an ad-hoc run dir.
Can the frozen support be recovered, and does it map onto Build F's rebuilt base
(`18833fb6…`, rebuilt from raw ASEC, so row identities differ)?

## Findings

### F1 — The named warm-start artifact is NOT on Hugging Face

`policyengine/populace-us` has one branch (`main`) and 15 tags. The certified tag
exists and carries `populace_us_2024.h5` (354 MB, 57,240 households),
`populace_us_2024_calibration.npz` (2.2 MB, the shipped weights), and the release
manifests. There is **no** `refit-selected-current-surface-a7a77c4` tag or file.
The manifest's `warm_start_calibration.path` is a local `out/…` run-dir path; the
`.npz` (sha `7ba46106…`) is not published. So the frozen support cannot be
retrieved as a named artifact.

The build manifest records **no** `base_dataset_sha256` and no lineage back to the
337,704 pool or to the selection run — the same lineage gap #328 describes.

### F2 — The frozen support IS recoverable from the published H5's record identities

The published certified H5 carries, on its **person** table, the stable source
identity triple already used elsewhere in the build
(`us_runtime/take_up.py::_SOURCE_IDENTITY_COLUMNS`):

- `source_year` ∈ {2024, 2023, 2022} (persons 57,185 / 54,665 / 54,452 — the
  3-year pool), `source_household_id`, `source_person_id` (stable ASEC person id
  strings), `person_support_channel` ∈ {asec, puf_tax_detail},
  `person_support_clone_index` ∈ {0, 1}.

and on its **household** table: `household_id` (assigned row id — order-dependent),
`household_source_id`, `household_support_channel`, `household_support_clone_index`.

The Build F base carries the identical identity columns. So the *set of source
records* the certified default selected is fully readable from the published H5,
independent of the assigned `household_id`.

### F3 — The identity join onto the rebuilt base is PERFECT and unambiguous

Household-level identity key
`(source_year, source_household_id, household_support_channel, household_support_clone_index)`:

- **Unique** in the Build F base (337,704 / 337,704 distinct) and in the certified
  support (57,240 / 57,240 distinct).
- Joining the certified support onto the Build F base by this key:
  **57,240 of 57,240 match (100.00%), 0 unmapped, 0 ambiguous** (no certified key
  hits more than one base row).

Two other keys were also verified fully unique in both frames and are viable
fallbacks: `(source_year, source_household_id, channel, min(source_person_id))`
and `(source_year, household_source_id, clone_index)`. The channel/clone component
is **required** — without it, a source ASEC household's two support-channel clones
collide.

### F4 — Positional warm-start is unusable across a rebuild (why identity-join is mandatory)

The selected base rows are **scattered**, not a prefix: in Build F base
`household_id` order they sit at positions 4, 7, 10, 15, 19, … up to 337,698. The
two existing warm-start seams are both strictly positional:

- `tools/build_us_fiscal_refresh_release.py::_load_warm_start_calibration_npz`
  requires `household_weight.shape == n(household)` **and** asserts the stored
  `initial_household_weight` matches the current frame's initial weights
  (`max_abs_initial_household_weight_delta` must be ~0 — it is exactly 0.0 in the
  certified manifest, i.e. that run's warm start was positionally aligned to its
  own frozen support).
- `us_runtime/l0_refit_export.py::load_l0_refit_npz` / `attach_l0_refit_weights`
  require a full-length vector aligned to the base's own row order and apply the
  selection mask positionally.

Against a rebuilt base (57,240-length vector vs 337,704 rows; different order),
both would shape-mismatch or silently misalign. Recovery therefore **must** join on
the stable identity, which is exactly what this reconstruction adds.

## Decision for Phase 2

Because the join is 100% clean and unambiguous, **frozen-support mode (select
exactly the named record set, then calibrate weights on that fixed support — the
certified pattern) is implementable now with no change to the calibrate library**,
and is the first mode to ship. Informed-init mode (initialize L0 selection
probabilities from the named artifact, then optimize) is designed as the drift-
robust successor but requires a new selection-probability-init parameter in
`populace-calibrate`; it is deferred behind mode (a) precisely because Phase 1
shows the current base needs no drift tolerance.

## Reproduction

- HF reads: `huggingface_hub` against `policyengine/populace-us`, revision =
  certified tag, `repo_type="dataset"` (read-only). Token from the shared
  `HUGGING_FACE_TOKEN` agent secret.
- Join verification: `_buildg-runtime/forensics/join_check.py` (kept out of the
  repo tree). Inputs: certified H5 (HF) + Build F base
  `out/base-f-20260705/base_populace_us_2024_puf_support.h5` (sha `18833fb6…`).
- Recovered selection mask over the Build F base:
  `_buildg-runtime/forensics/frozen_support_mask.npy` (337,704 bool, 57,240 True).
